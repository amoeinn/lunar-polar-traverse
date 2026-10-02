"""The time-dependent planner: earliest-arrival label-setting search.

State is (cell, arrival time, continuous shadow time). Edge traversal time is
the phase 2 static cost and does not depend on departure time, so the problem
is FIFO and leaving earlier always arrives earlier; all the time dependence
sits in feasibility. Waiting is allowed in lit cells only, which is what makes
the dominance rule sound -- see PLAN.md, "Planners", for the rule and why the
reachability clause is not optional.

Nothing here knows how the reference in src/reference.py works, and the
reference reads no interval table, so check 3 compares two implementations and
not two copies of one.
"""
import heapq
from dataclasses import dataclass, field

import numpy as np

from .cost import edge_time, legal


@dataclass
class Graph:
    """Directed 8-connected graph, legal edges only, in CSR form."""
    indptr: np.ndarray
    indices: np.ndarray
    travel: np.ndarray          # hours
    n_cells: int

    @classmethod
    def build(cls, edges, rover, n_cells):
        ok = legal(edges["pitch"], edges["roll"], rover)
        a, b = edges["a"][ok], edges["b"][ok]
        t = edge_time(edges["length"][ok], edges["pitch"][ok], edges["roll"][ok], rover)
        order = np.argsort(a, kind="stable")
        a, b, t = a[order], b[order], t[order]
        indptr = np.zeros(n_cells + 1, dtype=np.int64)
        np.add.at(indptr, a + 1, 1)
        return cls(np.cumsum(indptr), b.astype(np.int64), t.astype(float), n_cells)

    def neighbours(self, cell):
        lo, hi = self.indptr[cell], self.indptr[cell + 1]
        return self.indices[lo:hi], self.travel[lo:hi]


@dataclass
class Stats:
    timed_out: bool = False
    pruned_component: int = 0
    pruned_bound: int = 0
    labels_lit: int = 0
    labels_dark: int = 0
    pushed: int = 0
    popped: int = 0
    kept: int = 0
    dominated: int = 0
    infeasible: int = 0
    labels_per_cell: dict = field(default_factory=dict)


def _departure_candidates(intervals, cell, other, t, window_end, tau, tol):
    """Departure times worth trying while waiting is possible at `cell`.

    Waiting only delays arrival, so the earliest feasible departure is the one
    that matters -- but feasibility is not monotone in the departure time, so
    the earliest feasible one is not always a structural breakpoint. The
    maximum shadow over a traverse is the longest dark run inside the sliding
    window [d, d + tau]; as d grows the leading run shrinks and the trailing
    one grows, so the constraint is crossed at closed-form times. Including
    them keeps the search exact:

      T      a transition enters the window at its start
      T - tau  a transition enters at its end
      T - tol  the leading dark run has shrunk to the tolerance
      T + tol - tau  the trailing dark run has grown to the tolerance

    Leaving out the last two returns "unreachable" on instances that are
    reachable only by waiting past the point where the constraint clears; a
    test asserts exactly that.
    """
    hi = window_end + tau
    trans = np.concatenate([intervals.transitions_in(cell, t, hi),
                            intervals.transitions_in(other, t, hi)])
    cands = {t}
    for tr in trans:
        tr = float(tr)
        cands.update((tr, tr - tau, tr - tol, tr + tol - tau))
    return sorted(d for d in cands if t <= d < window_end)


def plan(graph, intervals, start, goal, t_start, shadow_tol, time_limit_s=None,
         components=None, positions=None, v_cap=None, shadow_grid_h=0.0,
         relax_dominance=False):
    """Earliest arrival at goal from start departing no earlier than t_start.

    Returns arrival time in hours, the cell path, and search statistics. An
    unreachable goal returns arrival None, which is a result and not an error:
    at no candidate limit pair is the region fully connected.

    time_limit_s aborts the search and sets stats.timed_out rather than
    truncating the label sets, which would make the search silently inexact.
    An aborted search is reported as aborted, never as unreachable.

    Two optional prunings, both exact. Neither may change an answer, and tests
    assert that on instances where each fires.

    components: a static connected-component label per cell, from the legal
    graph ignoring illumination. A goal in a different component is
    unreachable with the shadow constraint too, since adding a constraint only
    removes solutions.

    positions and v_cap: the admissible arrival bound. No continuation from a
    cell can be faster than the straight-line 3D distance to the goal over the
    top speed, so a label whose time plus that bound exceeds the horizon cannot
    produce an arrival and is dropped. Admissible because the bound is a lower
    bound on the remaining time, which is what makes dropping it exact.
    """
    import time as _time
    deadline = None if time_limit_s is None else _time.monotonic() + time_limit_s
    if components is not None and components[start] != components[goal]:
        st = Stats()
        st.pruned_component = 1
        return None, [], st

    bound = None
    if positions is not None and v_cap:
        goal_pos = np.asarray(positions[goal], dtype=float)
        bound = np.linalg.norm(np.asarray(positions, dtype=float) - goal_pos,
                               axis=-1) / float(v_cap)

    labels = {}                     # cell -> list of (t, s)
    parent = {}                     # label id -> (parent id, cell)
    heap = []
    st = Stats()

    def admit(cell, t, s, par):
        if s > shadow_tol:
            st.infeasible += 1
            return
        if t > intervals.horizon_h:
            st.infeasible += 1
            return
        if bound is not None and t + bound[cell] > intervals.horizon_h:
            st.pruned_bound += 1
            return
        for t2, s2 in labels.get(cell, ()):
            if t2 <= t and s2 <= s:
                if t2 == t:
                    st.dominated += 1
                    return
                if relax_dominance:
                    st.dominated += 1
                    return
                if intervals.is_lit(cell, t2) and intervals.next_change(cell, t2) > t:
                    st.dominated += 1
                    return
        if intervals.is_lit(cell, t):
            st.labels_lit += 1
        else:
            st.labels_dark += 1
        labels.setdefault(cell, []).append((t, s))
        lid = len(parent)
        parent[lid] = (par, cell)
        heapq.heappush(heap, (t, s, cell, lid))
        st.pushed += 1
        st.kept += 1

    admit(start, float(t_start), 0.0, None)
    checks = 0
    while heap:
        t, s, cell, lid = heapq.heappop(heap)
        st.popped += 1
        checks += 1
        if deadline is not None and checks % 512 == 0 and _time.monotonic() > deadline:
            st.timed_out = True
            st.labels_per_cell = {c: len(v) for c, v in labels.items()}
            return None, [], st
        if cell == goal:
            path, node = [], lid
            while node is not None:
                par, c = parent[node]
                path.append(c)
                node = par
            st.labels_per_cell = {c: len(v) for c, v in labels.items()}
            return float(t), path[::-1], st
        lit_here = intervals.is_lit(cell, t)
        window_end = intervals.next_change(cell, t) if lit_here else t
        nb, tau = graph.neighbours(cell)
        for n, tt in zip(nb, tau):
            n = int(n)
            if lit_here:
                cands = _departure_candidates(intervals, cell, n, t, window_end,
                                              float(tt), shadow_tol)
            else:
                cands = [t]
            for d in cands:
                arrive = d + tt
                s_dep = 0.0 if lit_here else s
                s_arr, s_max = intervals.shadow_along(cell, n, d, arrive, s_dep)
                if shadow_grid_h > 0.0:
                    s_arr = np.ceil(s_arr / shadow_grid_h) * shadow_grid_h
                    s_max = np.ceil(s_max / shadow_grid_h) * shadow_grid_h
                if s_max > shadow_tol:
                    st.infeasible += 1
                    continue
                admit(n, arrive, s_arr, lid)
    st.labels_per_cell = {c: len(v) for c, v in labels.items()}
    return None, [], st
