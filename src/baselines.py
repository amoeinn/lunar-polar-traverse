"""The two baselines: snapshot and mean-map. Both as good as their own idea allows.

Snapshot is not "A* that ignores time". It is the full time-dependent search run
on a world frozen at departure: the same engine, the same shadow constraint, the
same waiting rule, on an illumination field that never changes. Waiting is then
never useful, which is the point rather than a limitation, and it isolates the
error to the illumination model instead of to the search.

Mean-map has no clock and so cannot evaluate the shadow constraint at all;
pretending otherwise would be a strawman in the other direction. It plans on the
static cost multiplied by a weight derived from each cell's lit fraction over
the horizon, and replay decides whether its plan was feasible. The weighting
exponent is its hyperparameter and PLAN.md registers that the baseline is swept
over it and reported at its best setting, while the time-dependent planner gets
no such knob.
"""
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra

from .intervals import LitIntervals
from .planner import plan


def frozen_intervals(intervals, t_departure):
    """The same illumination, frozen at one instant and never changing again.

    A cell lit at departure is lit for ever and a cell dark at departure is dark
    for ever, so the search sees a static world while using the identical code
    path. No transitions at all, which is what makes waiting vacuous.
    """
    n = intervals.offsets.size - 1
    lit = np.array([intervals.is_lit(c, t_departure) for c in range(n)], dtype=bool)
    return LitIntervals(np.zeros(0, dtype=float), np.zeros(n + 1, dtype=np.int64),
                        lit, intervals.horizon_h)


def plan_snapshot(graph, intervals, start, goal, t_start, shadow_tol, **kw):
    """Plan on illumination frozen at departure, then live with the schedule."""
    return plan(graph, frozen_intervals(intervals, t_start), start, goal, t_start,
                shadow_tol, **kw)


def mean_map_costs(graph, lit_fraction, exponent, floor=1e-3):
    """Static edge costs weighted by the destination cell's lit fraction.

    exponent 0 is the pure fastest-path planner, which is one end of the sweep
    and a useful one: it says what ignoring illumination entirely costs. Larger
    exponents buy wider detours around cells that are dark more of the time.
    """
    w = 1.0 / np.maximum(np.asarray(lit_fraction, dtype=float), floor) ** exponent
    return graph.travel * w[graph.indices]


def plan_mean_map(graph, lit_fraction, start, goal, exponent, floor=1e-3):
    """Shortest weighted path. Returns the unweighted traverse time and the path.

    The weight decides the route; the time the rover actually spends is the
    static cost of that route, which is what a later arrival comparison needs.
    """
    costs = mean_map_costs(graph, lit_fraction, exponent, floor)
    mat = csr_matrix((costs, graph.indices, graph.indptr),
                     shape=(graph.n_cells, graph.n_cells))
    dist, pred = dijkstra(mat, directed=True, indices=start,
                          return_predecessors=True)
    if not np.isfinite(dist[goal]):
        return None, []
    path, c = [], goal
    while c != start:
        path.append(int(c))
        c = int(pred[c])
        if c < 0:
            return None, []
    path.append(int(start))
    path = path[::-1]
    # traverse time along the chosen route, from the unweighted edge times
    total = 0.0
    for u, v in zip(path[:-1], path[1:]):
        lo, hi = graph.indptr[u], graph.indptr[u + 1]
        k = np.flatnonzero(graph.indices[lo:hi] == v)
        total += float(graph.travel[lo:hi][k[0]])
    return total, path
