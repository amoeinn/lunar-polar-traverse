"""Driver for the executive: one instance, one or more legs.

The behaviour tree in cpp/src/executive.cpp monitors one leg and decides. This
drives it: it builds the world each leg from the replay checker's trajectory
dump, runs the tree, and if the tree adopted a new plan it re-simulates from
the firing instant and runs the tree again, up to the registered k.

The split is deliberate. Deciding and replanning are the executive's job and
live in the tree; re-simulating the world is world simulation and lives here,
so the tree stays thin and the checker stays the only source of execution time.
"""
import json
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
EXEC_BIN = ROOT / "cpp" / "build" / "executive"
REPLAY_BIN = ROOT / "cpp" / "build" / "replay"
PLANNER_BIN = ROOT / "cpp" / "build" / "planner"

# H, the contradiction rule's horizon: the one definition, shared with the C++
# executive through its params file, which carries no value of its own.
# Registered 2026-11-22 as the held-out p99 of the disagreements the rule acts
# on (world dark, model lit), measured by examples/disagreement.py on the
# corrected sun. It replaced 0.7250 h, measured the same way before the fix.
CONTRA_HORIZON_H = 0.7105

TRAJ_KEYS = ("t", "shadow", "index", "kind", "lit", "offsets")
LOG_KEYS = (("tick", np.int64), ("t", np.float64), ("s_exec", np.float64),
            ("s_world", np.float64), ("s_pred", np.float64),
            ("index", np.int32), ("kind", np.uint8), ("lit", np.uint8),
            ("fired", np.uint8))


@dataclass
class Leg:
    """One pass of the tree over one world, with the world it ran on."""
    outcome: str
    reason: str
    fired_at_t: float
    fired_at_tick: int
    ticks_consumed: int
    ticks_total: int
    replans: int
    s_exec_final: float
    s_world_final: float
    t_final: float
    log: dict
    plan: np.ndarray
    plan_times: np.ndarray
    world_shadow_max: float
    world_arrival: float
    world_verdict: bool
    new_plan: np.ndarray = field(default_factory=lambda: np.zeros(0, np.int32))
    new_times: np.ndarray = field(default_factory=lambda: np.zeros(0, np.float64))


def plan_waits(plan, times, indptr, indices, travel):
    """The hold at each cell implied by the planner's schedule.

    Identical in form to what the sweep drivers compute, and needed here for the
    same reason: the checker must hold where the plan holds or it is integrating
    a different plan.
    """
    w = np.zeros(plan.size)
    for i in range(plan.size - 1):
        u, v = int(plan[i]), int(plan[i + 1])
        s0, s1 = indptr[u], indptr[u + 1]
        k = np.flatnonzero(indices[s0:s1] == v)
        if k.size:
            w[i] = max(times[i + 1] - float(travel[s0:s1][k[0]]) - times[i], 0.0)
    return w


def simulate(replay_template, plan, start_h, waits, work):
    """Run the checker on one plan and return its outputs plus the trajectory.

    This is the world. It shares only the DEM, the projection constants, the
    radius and the rover parameters with the planner, which is what makes the
    executed time independent of the time model being monitored.
    """
    work = Path(work)
    work.mkdir(parents=True, exist_ok=True)
    p = dict(json.loads(Path(replay_template).read_text()))
    offs = np.array([0, plan.size], dtype=np.int64)
    (work / "plan_offsets.bin").write_bytes(offs.tobytes())
    (work / "plan_cells.bin").write_bytes(plan.astype(np.int32).tobytes())
    (work / "plan_start.bin").write_bytes(np.array([start_h], np.float64).tobytes())
    (work / "plan_wait.bin").write_bytes(np.asarray(waits, np.float64).tobytes())
    p["plan_offsets_path"] = str(work / "plan_offsets.bin")
    p["plan_cells_path"] = str(work / "plan_cells.bin")
    p["plan_start_path"] = str(work / "plan_start.bin")
    p["plan_wait_path"] = str(work / "plan_wait.bin")
    p["n_plans"] = 1
    for k in list(p):
        if k.startswith("out_"):
            p[k] = str(work / f"{k}.bin")
    for k in TRAJ_KEYS:
        p[f"out_traj_{k}"] = str(work / f"out_traj_{k}.bin")
    (work / "params.json").write_text(json.dumps(p, indent=2))
    subprocess.run([str(REPLAY_BIN), str(work / "params.json")], check=True,
                   capture_output=True)
    dt = dict(t=np.float64, shadow=np.float64, index=np.int32, kind=np.uint8,
              lit=np.uint8, offsets=np.int64)
    traj = {k: np.fromfile(p[f"out_traj_{k}"], dtype=dt[k]) for k in TRAJ_KEYS}
    return dict(
        arrival=float(np.fromfile(p["out_arrival"], np.float64)[0]),
        shadow=float(np.fromfile(p["out_shadow"], np.float64)[0]),
        verdict=bool(np.fromfile(p["out_verdict"], np.uint8)[0]),
        illegal=int(np.fromfile(p["out_illegal"], np.int32)[0]),
        traj=traj, params=p, work=work)


def run_leg(planner_template, graph_paths, plan, plan_times, holds, world, work,
            budget_h, replan_limit, replan_cap_s, replans_so_far,
            contra_bounded=True, contra_horizon_h=CONTRA_HORIZON_H, correction=None):
    """One pass of the tree over one world."""
    work = Path(work)
    work.mkdir(parents=True, exist_ok=True)
    (work / "plan.bin").write_bytes(plan.astype(np.int32).tobytes())
    (work / "plan_times.bin").write_bytes(plan_times.astype(np.float64).tobytes())
    (work / "plan_holds.bin").write_bytes(np.asarray(holds, np.float64).tobytes())
    tp = json.loads(Path(planner_template).read_text())
    p = dict(graph_paths)
    p.update(
        n_cells=tp["n_cells"], n_edges=tp["n_edges"],
        n_transitions=tp["n_transitions"], horizon_h=tp["horizon_h"],
        shadow_tol_h=budget_h, replan_limit=replan_limit,
        replan_cap_s=replan_cap_s, replans_so_far=replans_so_far,
        vis_cadence_h=1.0 / 60.0,
        contradiction_bounded=1.0 if contra_bounded else 0.0,
        contradiction_horizon_h=contra_horizon_h,
        planner_bin=str(PLANNER_BIN), scratch_dir=str(work),
        template_planner_params=str(planner_template),
        plan_path=str(work / "plan.bin"),
        plan_times_path=str(work / "plan_times.bin"),
        plan_holds_path=str(work / "plan_holds.bin"))
    if correction is not None:
        # One byte per plan step; the key is absent when there is no
        # correction, which the binary's guard test proves is bit-identical to
        # the run every scored figure in this record came from.
        (work / "correction.bin").write_bytes(
            np.asarray(correction, np.uint8).tobytes())
        p["correction_path"] = str(work / "correction.bin")
    for k in TRAJ_KEYS:
        if k == "offsets":
            continue
        p[f"traj_{k}_path"] = str(world["params"][f"out_traj_{k}"])
    for name, _ in LOG_KEYS:
        p[f"out_log_{name}"] = str(work / f"out_log_{name}.bin")
    p["out_new_plan"] = str(work / "out_new_plan.bin")
    p["out_new_times"] = str(work / "out_new_times.bin")
    p["out_summary"] = str(work / "out_summary.json")
    (work / "params.json").write_text(json.dumps(p, indent=2))
    subprocess.run([str(EXEC_BIN), str(work / "params.json")], check=True,
                   capture_output=True)
    s = json.loads((work / "out_summary.json").read_text())
    log = {name: np.fromfile(p[f"out_log_{name}"], dtype=dt)
           for name, dt in LOG_KEYS}
    return Leg(outcome=s["outcome"], reason=s["reason"],
               fired_at_t=float(s["fired_at_t"]),
               fired_at_tick=int(s["fired_at_tick"]),
               ticks_consumed=int(s["ticks_consumed"]),
               ticks_total=int(s["ticks_total"]), replans=int(s["replans"]),
               s_exec_final=float(s["s_exec_final"]),
               s_world_final=float(s["s_world_final"]),
               t_final=float(s["t_final"]), log=log, plan=plan,
               plan_times=plan_times, world_shadow_max=world["shadow"],
               world_arrival=world["arrival"], world_verdict=world["verdict"],
               new_plan=np.fromfile(p["out_new_plan"], np.int32),
               new_times=np.fromfile(p["out_new_times"], np.float64))


def run_instance(planner_template, replay_template, graph_paths, plan, plan_times,
                 work, budget_h, replan_limit=2, replan_cap_s=30.0,
                 contra_bounded=True, contra_horizon_h=CONTRA_HORIZON_H,
                 correction_fn=None):
    """Every leg of one instance, chaining after each adopted replan."""
    work = Path(work)
    indptr = np.fromfile(graph_paths["indptr_path"], np.int64)
    indices = np.fromfile(graph_paths["indices_path"], np.int32)
    travel = np.fromfile(graph_paths["travel_path"], np.float64)
    legs, replans = [], 0
    cur_plan, cur_times = plan, plan_times
    start = float(plan_times[0])
    while True:
        i = len(legs)
        waits = plan_waits(cur_plan, cur_times, indptr, indices, travel)
        world = simulate(replay_template, cur_plan, start, waits,
                         work / f"leg{i}_world")
        # Recomputed per leg, because a replan changes the plan the stream is
        # indexed by. A stream carried over from the previous plan would
        # correct steps that no longer exist.
        corr = (None if correction_fn is None
                else correction_fn(cur_plan, cur_times))
        leg = run_leg(planner_template, graph_paths, cur_plan, cur_times, waits,
                      world, work / f"leg{i}_exec", budget_h, replan_limit,
                      replan_cap_s, replans, contra_bounded, contra_horizon_h,
                      corr)
        legs.append(leg)
        if leg.outcome != "fired_replanned" or leg.new_plan.size < 2:
            break
        replans = leg.replans
        cur_plan, cur_times = leg.new_plan, leg.new_times
        start = float(cur_times[0])
        if replans >= replan_limit:
            break
    return legs
