"""The executive and its auditor, each test naming what it protects.

The executive's job is to fire when continuing would bust the shadow budget and
to stay quiet otherwise, so every test here is about one of those two and about
the constraints PLAN.md registers around them. The fixtures are synthetic on
purpose: a line of cells with hand-built illumination is the only way to make a
world where the right answer is known before the executive is asked.
"""
import json
import subprocess
from pathlib import Path

import numpy as np
import pytest

from src import auditor as A
from src.executive import CONTRA_HORIZON_H

ROOT = Path(__file__).resolve().parents[1]
EXEC_BIN = ROOT / "cpp" / "build" / "executive"
PLANNER_BIN = ROOT / "cpp" / "build" / "planner"
pytestmark = pytest.mark.skipif(not EXEC_BIN.exists(),
                                reason="executive binary not built")

N = 5                      # cells in a line
HORIZON = 100.0
EDGE_H = 1.0               # travel time per edge, hours


def line_graph(td, n=N, edge_h=EDGE_H, drop_into_goal=False):
    """A line of n cells, each joined to its neighbours, travel edge_h per hop."""
    indptr, indices, travel = [0], [], []
    for c in range(n):
        for d in (c - 1, c + 1):
            if 0 <= d < n:
                if drop_into_goal and d == n - 1:
                    continue
                indices.append(d)
                travel.append(edge_h)
        indptr.append(len(indices))
    (td / "indptr.bin").write_bytes(np.array(indptr, np.int64).tobytes())
    (td / "indices.bin").write_bytes(np.array(indices, np.int32).tobytes())
    (td / "travel.bin").write_bytes(np.array(travel, np.float64).tobytes())
    return dict(indptr_path=str(td / "indptr.bin"),
                indices_path=str(td / "indices.bin"),
                travel_path=str(td / "travel.bin"),
                n_cells=n, n_edges=len(indices))


def intervals(td, lit0, transitions, n=N):
    """Per-cell lit/dark intervals: a starting state and the switch times."""
    times, offsets = [], [0]
    for c in range(n):
        times.extend(transitions.get(c, []))
        offsets.append(len(times))
    (td / "itimes.bin").write_bytes(np.array(times, np.float64).tobytes())
    (td / "ioffsets.bin").write_bytes(np.array(offsets, np.int64).tobytes())
    (td / "ilit0.bin").write_bytes(
        np.array([1 if lit0.get(c, True) else 0 for c in range(n)], np.uint8).tobytes())
    return dict(itimes_path=str(td / "itimes.bin"),
                ioffsets_path=str(td / "ioffsets.bin"),
                ilit0_path=str(td / "ilit0.bin"),
                n_transitions=len(times))


def world(td, ticks):
    """A hand-built trajectory: (t, shadow, index, kind, lit) per tick."""
    cols = list(zip(*ticks)) if ticks else [(), (), (), (), ()]
    for name, dt, col in (("t", np.float64, cols[0]), ("shadow", np.float64, cols[1]),
                          ("index", np.int32, cols[2]), ("kind", np.uint8, cols[3]),
                          ("lit", np.uint8, cols[4])):
        (td / f"traj_{name}.bin").write_bytes(np.array(col, dt).tobytes())
    return {f"traj_{k}_path": str(td / f"traj_{k}.bin")
            for k in ("t", "shadow", "index", "kind", "lit")}


def corridor_with_annex(td, annex=1200, edge_h=EDGE_H, annex_edge_h=0.01):
    """A five-cell corridor plus a long annex hanging off cell 1.

    The corridor is the plan and goes dark, so the monitor fires. The annex is
    lit and reachable only from cell 1, so a replan launched there explores all
    of it before it can conclude anything about the goal. That is what makes a
    search long enough to reach the committed engine's clock check, which it
    only performs every 512 pops.

    The annex edges are cheap for a measured reason: the search is bounded by
    the horizon, so with one-hour hops it could only walk 99 cells of a 1200
    cell annex before running out of horizon and it popped 104 labels, nowhere
    near the check. At 0.01 h a hop the whole annex fits inside the horizon.
    """
    n = N + annex
    adj = {c: set() for c in range(n)}
    for c in range(N - 1):
        adj[c].add(c + 1); adj[c + 1].add(c)
    adj[1].add(N); adj[N].add(1)
    for c in range(N, n - 1):
        adj[c].add(c + 1); adj[c + 1].add(c)
    indptr, indices, travel = [0], [], []
    for c in range(n):
        for d in sorted(adj[c]):
            indices.append(d)
            travel.append(edge_h if (c < N and d < N) else annex_edge_h)
        indptr.append(len(indices))
    (td / "indptr.bin").write_bytes(np.array(indptr, np.int64).tobytes())
    (td / "indices.bin").write_bytes(np.array(indices, np.int32).tobytes())
    (td / "travel.bin").write_bytes(np.array(travel, np.float64).tobytes())
    return dict(indptr_path=str(td / "indptr.bin"),
                indices_path=str(td / "indices.bin"),
                travel_path=str(td / "travel.bin"),
                n_cells=n, n_edges=len(indices)), n


def run(td, graph, iv, wrld, plan, times, holds=None, budget=1.0,
        replan_limit=2, replan_cap_s=30.0, bounded=True, horizon=CONTRA_HORIZON_H,
        correction=None, expect_rc=0):
    holds = np.zeros(len(plan)) if holds is None else holds
    (td / "plan.bin").write_bytes(np.array(plan, np.int32).tobytes())
    (td / "plan_times.bin").write_bytes(np.array(times, np.float64).tobytes())
    (td / "plan_holds.bin").write_bytes(np.array(holds, np.float64).tobytes())
    # the template a replan is built from, pointing at this fixture's own arrays
    tmpl = dict(graph); tmpl.update(iv)
    tmpl = {k: v for k, v in tmpl.items()}
    tmpl.update(shadow_tol_h=budget, t_start_h=0.0, horizon_h=HORIZON,
                time_limit_s=replan_cap_s, n_instances=1,
                pairs_path=str(td / "pairs.bin"))
    (td / "pairs.bin").write_bytes(np.array([0, N - 1], np.int32).tobytes())
    for k in ("arrival", "seconds", "flags", "pathlen", "stats",
              "path_cells", "path_times", "path_offsets"):
        tmpl[f"out_{k}"] = str(td / f"tmpl_out_{k}.bin")
    (td / "tmpl.json").write_text(json.dumps(tmpl, indent=2))

    p = dict(graph); p.update(iv); p.update(wrld)
    p.update(horizon_h=HORIZON, shadow_tol_h=budget, replan_limit=replan_limit,
             replan_cap_s=replan_cap_s, replans_so_far=0, vis_cadence_h=1.0 / 60.0,
             planner_bin=str(PLANNER_BIN), scratch_dir=str(td),
             template_planner_params=str(td / "tmpl.json"),
             plan_path=str(td / "plan.bin"),
             plan_times_path=str(td / "plan_times.bin"),
             plan_holds_path=str(td / "plan_holds.bin"),
             contradiction_bounded=1.0 if bounded else 0.0,
             contradiction_horizon_h=horizon)
    if horizon is None:
        del p["contradiction_horizon_h"]   # as a params file that omits it
    if correction is not None:
        (td / "correction.bin").write_bytes(
            np.asarray(correction, np.uint8).tobytes())
        p["correction_path"] = str(td / "correction.bin")
    for name in ("tick", "t", "s_exec", "s_world", "s_pred", "index", "kind",
                 "lit", "fired"):
        p[f"out_log_{name}"] = str(td / f"log_{name}.bin")
    p["out_new_plan"] = str(td / "new_plan.bin")
    p["out_new_times"] = str(td / "new_times.bin")
    p["out_summary"] = str(td / "summary.json")
    (td / "params.json").write_text(json.dumps(p, indent=2))
    r = subprocess.run([str(EXEC_BIN), str(td / "params.json")],
                       capture_output=True, text=True)
    if expect_rc != 0:
        assert r.returncode == expect_rc, (r.returncode, r.stderr[-300:])
        return {"stderr": r.stderr}
    assert r.returncode == 0, r.stderr[-600:]
    s = json.loads((td / "summary.json").read_text())
    dt = dict(tick=np.int64, t=np.float64, s_exec=np.float64, s_world=np.float64,
              s_pred=np.float64, index=np.int32, kind=np.uint8, lit=np.uint8,
              fired=np.uint8)
    s["log"] = {k: np.fromfile(p[f"out_log_{k}"], dtype=v) for k, v in dt.items()}
    return s


def all_lit_walk():
    """A five-cell walk, one hour per hop, every cell lit the whole time."""
    return [(float(i), 0.0, i, 0 if i == 0 else 1, 1) for i in range(N)]


def test_a_monitor_that_never_fires_fails_a_warranted_case(tmp_path):
    """Protects the whole point of the executive: it must fire when continuing busts.

    The fixture is built so that the right answer is known without asking the
    executive. Cells 2, 3 and 4 go dark at 1.5 h and stay dark, so a rover that
    keeps walking spends 2.5 h of unbroken darkness against a 1 h budget. A
    monitor that never fires cannot pass this, which is what makes it a test
    rather than a demonstration.
    """
    g = line_graph(tmp_path)
    iv = intervals(tmp_path, lit0={0: True, 1: True, 2: True, 3: True, 4: True},
                   transitions={2: [1.5], 3: [1.5], 4: [1.5]})
    w = world(tmp_path, all_lit_walk())
    s = run(tmp_path, g, iv, w, list(range(N)), [float(i) for i in range(N)],
            budget=1.0)
    # the forecast itself must see the bust, not merely the outcome
    assert s["log"]["s_pred"].max() > 1.0, s["log"]["s_pred"]
    assert s["outcome"] != "ran_to_end", s
    assert s["fired_at_tick"] >= 0


def test_a_plan_that_never_enters_darkness_is_left_alone(tmp_path):
    """Protects against a monitor that fires always, which would pass the test above.

    Every cell is lit for the whole horizon, so there is nothing to fire about
    and any firing is wrong. Without this, 'always fire' would be a passing
    executive.
    """
    g = line_graph(tmp_path)
    iv = intervals(tmp_path, lit0={c: True for c in range(N)}, transitions={})
    w = world(tmp_path, all_lit_walk())
    s = run(tmp_path, g, iv, w, list(range(N)), [float(i) for i in range(N)],
            budget=1.0)
    assert s["outcome"] == "ran_to_end", s
    assert s["fired_at_tick"] == -1
    assert s["log"]["fired"].sum() == 0
    assert s["ticks_consumed"] == s["ticks_total"]


def test_a_replan_is_never_launched_from_darkness(tmp_path):
    """Protects the registered replan-at-a-lit-cell constraint.

    Its reason is in the committed search: the initial label is admitted with
    zero accumulated continuous shadow, so a replan from a dark cell carrying
    darkness is not expressible. The executive must declare instead, and say
    that is why.
    """
    g = line_graph(tmp_path)
    iv = intervals(tmp_path, lit0={c: True for c in range(N)},
                   transitions={2: [0.5], 3: [0.5], 4: [0.5]})
    # the rover is dark from the first arrival onward
    ticks = [(0.0, 0.0, 0, 0, 1)] + [(float(i), 0.1 * i, i, 1, 0) for i in range(1, N)]
    w = world(tmp_path, ticks)
    s = run(tmp_path, g, iv, w, list(range(N)), [float(i) for i in range(N)],
            budget=1.0)
    assert s["outcome"] == "fired_declared", s
    assert s["reason"] == "dark_cannot_replan", s
    assert s["replans"] == 0


def test_an_unresolved_replan_is_not_reported_as_infeasible(tmp_path):
    """Protects the three-outcome rule where it is easiest to lose.

    A search that ran out of budget has not proved anything, and folding it
    into 'infeasible' would invent a proof.

    The fixture had to be rebuilt twice to express this, which is the point of
    checking that a test can fail before believing it. The committed search
    tests its clock every 512 pops and only when the cap is above zero, so a
    five-cell line with a zero cap never aborted: it proved the instance
    infeasible and the test passed against the wrong reason. Lengthening the
    dark corridor did not help either, because darkness prunes the search
    almost immediately. The annex is lit and reachable only from the cell the
    replan starts at, so the search must walk all of it, which is what finally
    carries it past the clock check.
    """
    g, n = corridor_with_annex(tmp_path)
    iv = intervals(tmp_path, lit0={c: True for c in range(n)},
                   transitions={2: [1.5], 3: [1.5], 4: [1.5]}, n=n)
    w = world(tmp_path, all_lit_walk())
    s = run(tmp_path, g, iv, w, list(range(N)), [float(i) for i in range(N)],
            budget=1.0, replan_cap_s=1e-9)
    assert s["outcome"] == "fired_declared", s
    assert s["reason"] == "unresolved", s


def test_a_transition_alone_is_enough_to_make_a_decision(tmp_path):
    """Protects the registered tick definition against a quiet regression to arrivals.

    Ticks fire at cell arrivals and at lit/dark transitions. A world whose only
    events after departure are transitions must still be monitored; if the
    executive only ever reacted to arrivals it would sit through this one.
    """
    g = line_graph(tmp_path)
    iv = intervals(tmp_path, lit0={c: True for c in range(N)},
                   transitions={2: [1.5], 3: [1.5], 4: [1.5]})
    ticks = [(0.0, 0.0, 0, 0, 1), (0.4, 0.0, 0, 2, 0), (0.8, 0.4, 0, 2, 1)]
    w = world(tmp_path, ticks)
    s = run(tmp_path, g, iv, w, list(range(N)), [float(i) for i in range(N)],
            budget=1.0)
    assert s["log"]["kind"].size > 0, "no decision was logged at a transition"
    assert set(s["log"]["kind"].tolist()) <= {2}, s["log"]["kind"]
    assert s["log"]["s_pred"].max() > 0.0


def test_the_auditor_catches_a_bookkeeping_drift():
    """Protects the drift audit: the executive's own accounting must be checked.

    The executive carries its own continuous-shadow accumulator so that a bug in
    it is visible rather than hidden behind the checker's. The auditor compares
    the two per tick against one visibility cadence. A drift the auditor cannot
    see is a state divergence nobody would notice.
    """
    tol = A.DRIFT_TOL_H
    log = dict(s_exec=np.array([0.0, 0.10, 0.20]),
               s_world=np.array([0.0, 0.10, 0.20]),
               fired=np.zeros(3, np.uint8), tick=np.arange(3),
               index=np.zeros(3, np.int32))

    class Leg:
        pass
    leg = Leg(); leg.log = log
    w = dict(traj=dict(shadow=np.array([0.0, 0.10, 0.20])), shadow=0.20)
    clean = A.audit_leg(leg, w, budget_h=1.0)
    assert clean["drift"].max() <= tol, clean["drift"]
    # now the same ticks with the executive's accumulator running away
    log2 = dict(log, s_exec=np.array([0.0, 0.10, 0.20 + 5 * tol]))
    leg2 = Leg(); leg2.log = log2
    dirty = A.audit_leg(leg2, w, budget_h=1.0)
    assert dirty["drift"].max() > tol, dirty["drift"]


def test_the_counterfactual_looks_forward_and_not_back():
    """Protects the auditor's counterfactual rule.

    The question at a tick is whether continuing from there would bust, so the
    judgement is a suffix maximum over what follows. A prefix maximum would
    answer the opposite question and would call an early tick unwarranted
    precisely when the bust is still ahead of it, which is the case the
    executive exists for.
    """
    s = np.array([0.0, 0.5, 2.5, 0.0])
    b = A.suffix_bust(s, budget_h=1.0)
    assert b.tolist() == [True, True, True, False], b.tolist()
    assert A.suffix_bust(np.array([2.5, 0.0, 0.0]), 1.0).tolist() == [True, False, False]
    assert A.suffix_bust(np.zeros(0), 1.0).size == 0


def test_the_forecast_follows_the_plan_including_its_planned_holds(tmp_path):
    """Protects against a forecast that walks the remainder on travel times alone.

    This is a bug the real data caught before any test did. On a core plan that
    holds for 2.79 h at three cells, a forecast built from travel times only
    departed early, crossed dark spans the plan was waiting out, and predicted
    6.34 h of continuous shadow where the planner had certified under 2 h. The
    plan is not a route, it is a schedule, and the holds are part of it.

    Here cells 2, 3 and 4 are dark from 1 h to 3 h. The plan waits at cell 1,
    which stays lit, until the darkness has passed, so nothing should fire. A
    forecast that ignores the hold sets off across the dark cells at 1 h and
    predicts about 2 h of shadow against a 1 h budget, so it fires. The fixture
    separates the two readings by construction.
    """
    g = line_graph(tmp_path)
    iv = intervals(tmp_path, lit0={c: True for c in range(N)},
                   transitions={2: [1.0, 3.0], 3: [1.0, 3.0], 4: [1.0, 3.0]})
    # arrive cell 1 at 1 h, hold there to 3 h, then walk out in the light
    times = [0.0, 1.0, 4.0, 5.0, 6.0]
    holds = [0.0, 2.0, 0.0, 0.0, 0.0]
    ticks = [(0.0, 0.0, 0, 0, 1), (1.0, 0.0, 1, 1, 1), (3.0, 0.0, 1, 3, 1),
             (4.0, 0.0, 2, 1, 1), (5.0, 0.0, 3, 1, 1), (6.0, 0.0, 4, 1, 1)]
    w = world(tmp_path, ticks)
    s = run(tmp_path, g, iv, w, list(range(N)), times, holds=holds, budget=1.0)
    assert s["outcome"] == "ran_to_end", s
    assert s["log"]["s_pred"].max() <= 1.0, s["log"]["s_pred"]


def test_disagreement_episodes_are_measured_at_their_true_length():
    """Protects the measurement the replacement rule's horizon is set from.

    A horizon justified by a distribution is only as good as the distribution,
    so the run-length measurement is asserted against a case whose answer is
    known by construction: the model says lit throughout, the world is dark for
    exactly five samples, then two, with agreement in between. The episodes must
    come back as five and two cadences, in that order, and the opposite
    direction must come back empty.
    """
    from src.disagree import episodes
    cad = 0.25
    model = np.ones(14, bool)                      # the model says lit all along
    world = np.ones(14, bool)
    world[2:7] = False                             # dark for five samples
    world[10:12] = False                           # and for two
    got = episodes(world, model, cadence_h=cad)
    # The size is asserted before the values, because np.allclose against an
    # empty array is True: a mutation that dropped every episode passed this
    # test until the count was checked. The assertion has to be able to fail.
    assert got.size == 2, got
    assert np.allclose(got, [5 * cad, 2 * cad]), got
    # the other direction, world lit where the model says dark, is empty here
    assert episodes(world, model, cadence_h=cad, world_dark_only=False).size == 0
    # and a run that reaches the end of the series is not dropped
    w2 = np.ones(5, bool); w2[3:] = False
    tail = episodes(w2, np.ones(5, bool), cadence_h=cad)
    assert tail.size == 1, tail
    assert np.allclose(tail, [2 * cad]), tail


def test_the_sampling_grid_follows_the_world_between_ticks():
    """Protects the piecewise-constant reading of the world's lit state.

    The checker re-reads visibility on its cadence and emits a tick when the
    state changes, so between two ticks the world's state is the earlier tick's
    and not the later one's. Reading it the other way round would shift every
    disagreement by one tick interval and would misdate the episodes the
    horizon is measured from.
    """
    from src.disagree import sample_states

    class AllLit:
        def is_lit(self, cell, t):
            return True

    # dark is declared at t=1 and the next tick at t=3 says lit again
    grid, world, model = sample_states([0.0, 1.0, 3.0], [1, 0, 1], [0, 1, 2],
                                       [0, 1, 2], AllLit(), cadence_h=0.5)
    assert grid.tolist() == [0.0, 0.5, 1.0, 1.5, 2.0, 2.5]
    assert world.tolist() == [True, True, False, False, False, False]
    assert model.all()


def test_the_contradiction_horizon_bounds_what_an_observation_projects(tmp_path):
    """Protects the registered replacement rule, against the rule it replaces.

    Where the world says dark and the model says lit, the observation wins. How
    far it is carried is the whole difference between the two rules, and this
    fixture separates them by construction: the model has no transitions at
    all, so its next change is the table's horizon, 100 h away. The first rule
    carries the darkness there and predicts tens of hours against a 1 h budget,
    so it fires. The replacement carries it 0.25 h and does not.

    The ticks are 0.1 h apart for a measured reason. At one hour apart the
    rover's own observed darkness reached the 1 h budget by the second tick, so
    both rules fired and the fixture could not tell them apart: it was testing
    the accumulator rather than the projection. Short steps keep the observed
    part small so the only thing that differs is the forecast.
    """
    g = line_graph(tmp_path)
    iv = intervals(tmp_path, lit0={c: True for c in range(N)}, transitions={})
    dt = 0.1
    ticks = [(0.0, 0.0, 0, 0, 1)] + [(i * dt, 0.0, i, 1, 0) for i in range(1, N)]
    w = world(tmp_path, ticks)
    plan, times = list(range(N)), [i * dt for i in range(N)]

    (tmp_path / "a").mkdir()
    unbounded = run(tmp_path / "a", g, iv, w, plan, times, budget=1.0,
                    bounded=False)
    assert unbounded["log"]["s_pred"].max() > 50.0, unbounded["log"]["s_pred"]
    assert unbounded["outcome"] != "ran_to_end", unbounded

    (tmp_path / "b").mkdir()
    bounded = run(tmp_path / "b", g, iv, w, plan, times, budget=1.0,
                  bounded=True, horizon=0.25)
    assert bounded["log"]["s_pred"].max() < 1.0, bounded["log"]["s_pred"]
    assert bounded["outcome"] == "ran_to_end", bounded
    assert bounded["log"]["fired"].sum() == 0


def test_the_binary_refuses_to_run_without_a_horizon(tmp_path):
    """Protects the single definition of H against a copy inside the binary.

    H lived in six places across two languages, two of them in this binary: a
    member initialiser and a fallback for a missing params key. A fallback is a
    second registered value that nothing exercises, so when H is re-registered
    it keeps the old one and a params file that omits the key runs under it
    silently. The binary now has no value of its own, and a params file without
    the key must be refused, not run.
    """
    g = line_graph(tmp_path)
    iv = intervals(tmp_path, lit0={c: True for c in range(N)}, transitions={})
    w = world(tmp_path, all_lit_walk())
    plan, times = list(range(N)), [float(i) for i in range(N)]
    r = run(tmp_path, g, iv, w, plan, times, horizon=None, expect_rc=2)
    assert "contradiction_horizon_h" in r["stderr"], r["stderr"]


def test_no_correction_stream_is_identical_to_an_all_zero_one(tmp_path):
    """Protects the whole P8 record against the correction stream's existence.

    The stream was added so a learned or computed illumination correction can
    reach the monitor's forecast. Everything already scored in this record was
    scored without it, so absent must mean absent: not nearly, and not only in
    the verdict but in every logged number. An all-zero stream exercises the
    new branch and must land on exactly the same log as supplying no stream at
    all, which is what separates "the parameter is ignored" from "the parameter
    is ignored when empty".

    The full guard is the 187-plan sample: with no stream the extended binary
    reproduced all 374 published P8 records with zero differing fields,
    recorded in PLAN.md on 2026-10-05. This is that guard in miniature, at a
    size the suite can run.
    """
    g = line_graph(tmp_path)
    iv = intervals(tmp_path, lit0={c: True for c in range(N)}, transitions={})
    w = world(tmp_path, all_lit_walk())
    plan, times = list(range(N)), [float(i) for i in range(N)]

    (tmp_path / "none").mkdir()
    a = run(tmp_path / "none", g, iv, w, plan, times, budget=1.0)
    (tmp_path / "zeros").mkdir()
    b = run(tmp_path / "zeros", g, iv, w, plan, times, budget=1.0,
            correction=np.zeros(len(plan), np.uint8))
    assert a["outcome"] == b["outcome"], (a["outcome"], b["outcome"])
    for k in a["log"]:
        assert np.array_equal(a["log"][k], b["log"][k]), (
            f"{k} differs between no stream and an all-zero stream: "
            f"{a['log'][k]} vs {b['log'][k]}")


def test_a_dark_mark_raises_the_forecast_where_the_model_says_lit(tmp_path):
    """The code-1 direction, and the test that lets the identity test fail.

    Without this, a binary that parsed correction_path and threw it away would
    pass the identity test above perfectly. The fixture is a walk whose cells
    the model calls lit for all time, so the forecast is zero darkness and the
    monitor stays quiet. Marking every step dark makes the forecast four hours
    against a one hour budget, so it must fire.

    On a model-LIT step codes 0 and 2 agree and only 1 differs, so this is the
    test that separates code 1 from both of the others. It is what catches a
    mutation treating 1 as 2.
    """
    g = line_graph(tmp_path)
    iv = intervals(tmp_path, lit0={c: True for c in range(N)}, transitions={})
    w = world(tmp_path, all_lit_walk())
    plan, times = list(range(N)), [float(i) for i in range(N)]

    (tmp_path / "off").mkdir()
    quiet = run(tmp_path / "off", g, iv, w, plan, times, budget=1.0)
    assert quiet["log"]["s_pred"].max() == 0.0, quiet["log"]["s_pred"]
    assert quiet["log"]["fired"].sum() == 0

    (tmp_path / "on").mkdir()
    loud = run(tmp_path / "on", g, iv, w, plan, times, budget=1.0,
               correction=np.ones(len(plan), np.uint8))
    assert loud["log"]["s_pred"].max() > 1.0, loud["log"]["s_pred"]
    assert loud["log"]["fired"].sum() > 0, loud


def test_a_correction_stream_of_the_wrong_length_is_refused(tmp_path):
    """A silently truncated stream would correct the wrong steps.

    The stream is indexed by plan step, so a length mismatch is not a smaller
    correction but a different one, applied to steps it was not computed for.
    The binary refuses rather than reading what it can.
    """
    g = line_graph(tmp_path)
    iv = intervals(tmp_path, lit0={c: True for c in range(N)}, transitions={})
    w = world(tmp_path, all_lit_walk())
    plan, times = list(range(N)), [float(i) for i in range(N)]
    out = run(tmp_path, g, iv, w, plan, times, budget=1.0,
              correction=np.ones(len(plan) - 1, np.uint8), expect_rc=2)
    assert "correction stream" in out["stderr"], out["stderr"]


def test_a_lit_mark_lowers_the_forecast_where_the_model_says_dark(tmp_path):
    """The code-2 direction, which is the only one that can remove a false alarm.

    The first version of this stream carried code 1 alone and could only raise
    a forecast, so it removed 0 of the executive's 17 false alarms by
    construction. Those false alarms are forecasts that are too pessimistic, so
    bringing one down needs a mark that says lit where the model says dark.

    The fixture is the mirror of the dark-mark test: the model calls every cell
    dark for all time while the world is observed lit, so the forecast projects
    four hours of modelled darkness against a one hour budget and the monitor
    fires. Marking every step lit must make it quiet.

    On a model-DARK step codes 0 and 1 agree and only 2 differs, so this is the
    test that separates code 2 from both of the others. It is what catches a
    mutation treating 2 as 1 and a mutation treating 2 as 0, and those two
    mutations are indistinguishable on this fixture precisely because 0 and 1
    agree here; the test above is what tells them apart from code 1 itself.
    """
    g = line_graph(tmp_path)
    iv = intervals(tmp_path, lit0={c: False for c in range(N)}, transitions={})
    # the world is lit the whole way, so nothing is observed dark and the only
    # thing driving the monitor is the forecast
    w = world(tmp_path, all_lit_walk())
    plan, times = list(range(N)), [float(i) for i in range(N)]

    (tmp_path / "believe").mkdir()
    loud = run(tmp_path / "believe", g, iv, w, plan, times, budget=1.0)
    assert loud["log"]["s_pred"].max() > 1.0, loud["log"]["s_pred"]
    assert loud["log"]["fired"].sum() > 0, loud

    (tmp_path / "lit").mkdir()
    quiet = run(tmp_path / "lit", g, iv, w, plan, times, budget=1.0,
                correction=np.full(len(plan), 2, np.uint8))
    assert quiet["log"]["s_pred"].max() < 1.0, quiet["log"]["s_pred"]
    assert quiet["log"]["fired"].sum() == 0, quiet


def test_a_correction_code_above_two_is_refused(tmp_path):
    """An unknown code is a different correction, not a bigger one.

    The stream is ternary. A 3 means whatever the branch chain happens to fall
    through to, which is "believe the model", so an out-of-range code would
    silently become no correction at all on that step.
    """
    g = line_graph(tmp_path)
    iv = intervals(tmp_path, lit0={c: True for c in range(N)}, transitions={})
    w = world(tmp_path, all_lit_walk())
    plan, times = list(range(N)), [float(i) for i in range(N)]
    codes = np.ones(len(plan), np.uint8); codes[2] = 3
    out = run(tmp_path, g, iv, w, plan, times, budget=1.0,
              correction=codes, expect_rc=2)
    assert "not 0, 1 or 2" in out["stderr"], out["stderr"]
