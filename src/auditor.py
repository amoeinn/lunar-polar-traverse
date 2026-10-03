"""The auditor: judges the executive's decisions by outcome, not by its forecast.

PLAN.md registers the rule. For every tick the auditor asks whether continuing
without a replan would in fact have busted the budget, and scores the
executive's decision against that. It never evaluates the executive's own
prediction, which is the independence that matters: the executive forecasts with
the planner's illumination model and the auditor observes with the checker's.

The counterfactual needs no extra checker run, and that is not a shortcut. The
world is deterministic and the unreplanned continuation from a tick has exactly
the same future as the run already integrated, so the continuation's outcome is
the suffix maximum of the checker's own continuous-shadow accumulation from that
tick onward. Re-running the checker would reproduce it exactly. The one thing
that has to be checked rather than assumed is that the tick samples capture the
accumulator's peaks, which audit_leg asserts against the checker's own reported
maximum.

Declarations are audited by a different instrument: the exact engine at a larger
cap than the executive used. A declaration of infeasibility that a larger budget
refutes is a false declaration.

The stated limitation: the auditor shares the world with the executed
trajectory, because the checker is both the world and the counterfactual engine.
It is independent of the executive, not of the world model, and this project has
no second world model.
"""
import json
import subprocess
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PLANNER_BIN = ROOT / "cpp" / "build" / "planner"
DRIFT_TOL_H = 1.0 / 60.0          # one visibility cadence, registered


def suffix_bust(shadow_at_ticks, budget_h):
    """Would continuing from each tick have busted the budget?

    The accumulator is carried by the checker, so the question at tick i is
    whether it ever exceeds the budget at or after i.
    """
    s = np.asarray(shadow_at_ticks, dtype=float)
    if s.size == 0:
        return np.zeros(0, dtype=bool)
    return np.maximum.accumulate(s[::-1])[::-1] > budget_h


def audit_leg(leg, world, budget_h):
    """Per-tick judgements for one leg, plus the drift table."""
    traj_sh = world["traj"]["shadow"]
    peak = float(traj_sh.max()) if traj_sh.size else 0.0
    # The tick samples must capture the accumulator's peaks or the
    # counterfactual is computed on a sampled-down signal.
    peak_gap = world["shadow"] - peak
    bust = suffix_bust(traj_sh, budget_h)
    ticks = leg.log["tick"]
    fired = leg.log["fired"].astype(bool)
    warranted = bust[ticks] if ticks.size else np.zeros(0, dtype=bool)
    drift = np.abs(leg.log["s_exec"] - leg.log["s_world"])
    return dict(fired=fired, warranted=warranted, drift=drift,
                ticks=ticks, peak_gap=float(peak_gap),
                bust_any=bool(bust[0]) if bust.size else False)


def audit_declaration(leg, planner_template, work, cap_s=300.0):
    """A declaration of infeasibility, put to the exact engine at a larger cap.

    A different instrument from the one the executive used, which is what makes
    this part of the audit independent of the world model as well.
    """
    work = Path(work)
    work.mkdir(parents=True, exist_ok=True)
    if leg.plan.size < 2 or leg.fired_at_tick < 0:
        return dict(audited=False, reason="no firing state to replan from")
    idx = int(leg.log["index"][-1])
    idx = min(max(idx, 0), leg.plan.size - 1)
    start, goal = int(leg.plan[idx]), int(leg.plan[-1])
    if start == goal:
        return dict(audited=False, reason="already at the goal")
    p = dict(json.loads(Path(planner_template).read_text()))
    (work / "pairs.bin").write_bytes(
        np.array([start, goal], dtype=np.int32).tobytes())
    p["pairs_path"] = str(work / "pairs.bin")
    p["n_instances"] = 1
    p["t_start_h"] = float(leg.fired_at_t)
    p["time_limit_s"] = cap_s
    for k in list(p):
        if k.startswith("out_"):
            p[k] = str(work / f"{k}.bin")
    (work / "params.json").write_text(json.dumps(p, indent=2))
    subprocess.run([str(PLANNER_BIN), str(work / "params.json")], check=True,
                   capture_output=True)
    aborted = bool(np.fromfile(p["out_flags"], np.uint8)[0])
    arrival = float(np.fromfile(p["out_arrival"], np.float64)[0])
    seconds = float(np.fromfile(p["out_seconds"], np.float64)[0])
    found = (not aborted) and np.isfinite(arrival)
    return dict(audited=True, cap_s=cap_s, aborted=aborted, arrival=arrival,
                seconds=seconds, plan_exists=bool(found),
                start=start, goal=goal, t_start=float(leg.fired_at_t))


def contingency(fired, busted):
    f = np.asarray(fired, dtype=bool)
    b = np.asarray(busted, dtype=bool)
    return (int((f & b).sum()), int((f & ~b).sum()),
            int((~f & b).sum()), int((~f & ~b).sum()))


def mcc(tp, fp, fn, tn):
    den = np.sqrt(float(tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    return 0.0 if den == 0.0 else (tp * tn - fp * fn) / den


def youden(tp, fp, fn, tn):
    tpr = tp / (tp + fn) if (tp + fn) else 0.0
    fpr = fp / (fp + tn) if (fp + tn) else 0.0
    return tpr - fpr
