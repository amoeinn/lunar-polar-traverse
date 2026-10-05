"""Score a correction through the executive on the registered McNemar test.

The test is registered in PLAN.md on 2026-10-05, before this existed: paired
against the bounded rule on the same 45 monitored plans, passing if the
candidate removes enough of the bounded rule's 17 false alarms to reach
one-sided exact p <= 0.05, with zero busts lost and control firings still 0.

B3b as a monitor correction is "the model says the cell the rover is on is lit
but the next cell on the path is dark at that moment". That is the interval
table on both cells, not the exact-time horizon re-evaluation B3b was measured
with, and the substitution is justified by measurement rather than convenience:
B3a, the exact-time re-evaluation, differed from the table on 5 of 91690
samples, so on this terrain it recovers the table. Using the table costs no
SPICE call and no horizon array per worker.
"""
import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from math import comb
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import examples.run_executive as R                               # noqa: E402
from src import auditor as A                                     # noqa: E402
from src.disagree import FlatIntervals                           # noqa: E402
from src.executive import run_instance                           # noqa: E402

OUT = ROOT / "data" / "work" / "executive_correction"


def b3b_correction(graph, plan, times):
    """One byte per plan step: the model calls this cell lit and the next dark."""
    iv = FlatIntervals(graph["itimes_path"], graph["ioffsets_path"],
                       graph["ilit0_path"])
    plan = np.asarray(plan, np.int64)
    out = np.zeros(plan.size, np.uint8)
    for i in range(plan.size - 1):
        t = float(times[i])
        if iv.is_lit(int(plan[i]), t) and not iv.is_lit(int(plan[i + 1]), t):
            out[i] = 1
    return out


def one(arg):
    rec, kind = arg
    work = OUT / kind / rec["tag"] / f"plan{rec['plan_index']:03d}"
    plan = np.array(rec["plan"], np.int32)
    times = np.array(rec["times"], np.float64)
    fn = None if kind == "none" else (
        lambda p, t: b3b_correction(rec["graph"], p, t))
    legs = run_instance(rec["planner_params"], rec["replay_params"],
                        rec["graph"], plan, times, work, R.BUDGET_H,
                        R.REPLAN_LIMIT, R.REPLAN_CAP_S, True,
                        R.CONTRA_HORIZON_H, fn)
    fired_any, fired_warranted = False, False
    for n, leg in enumerate(legs):
        wd = work / f"leg{n}_world"
        traj_sh = np.fromfile(wd / "out_traj_shadow.bin", np.float64)
        w = dict(traj=dict(shadow=traj_sh), shadow=leg.world_shadow_max)
        a = A.audit_leg(leg, w, R.BUDGET_H)
        if bool(a["fired"].any()):
            fired_any = True
            k = int(np.flatnonzero(a["fired"])[0])
            fired_warranted |= bool(a["warranted"][k])
    return dict(tag=rec["tag"], plan_index=rec["plan_index"], kind=kind,
                monitored=rec["monitored"], busted=rec["busted"],
                fired=fired_any, fired_warranted=fired_warranted)


def mcnemar(b, c):
    n = b + c
    return sum(comb(n, k) for k in range(0, c + 1)) / 2 ** n if n else 1.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=6)
    args = ap.parse_args()
    recs = R.sample()
    OUT.mkdir(parents=True, exist_ok=True)
    jobs = [(r, k) for k in ("none", "b3b") for r in recs]
    print(f"{len(recs)} instances x 2 arms, {args.jobs} workers", flush=True)
    with ProcessPoolExecutor(max_workers=args.jobs) as ex:
        res = list(ex.map(one, jobs))
    (OUT / "results.json").write_text(json.dumps(res, indent=1))

    key = lambda r: (r["tag"], r["plan_index"])
    arms = {k: {key(r): r for r in res if r["kind"] == k}
            for k in ("none", "b3b")}
    base, cand = arms["none"], arms["b3b"]
    mon = [k for k in base if base[k]["monitored"]]
    print(f"\nmonitored {len(mon)}, controls {len(base)-len(mon)}")
    for nm, arm in (("bounded rule, no stream", base), ("B3b stream", cand)):
        TP = sum(1 for k in mon if arm[k]["busted"] and arm[k]["fired"])
        FN = sum(1 for k in mon if arm[k]["busted"] and not arm[k]["fired"])
        FP = sum(1 for k in mon if not arm[k]["busted"] and arm[k]["fired"])
        TN = sum(1 for k in mon if not arm[k]["busted"] and not arm[k]["fired"])
        cf = sum(1 for k in base if not base[k]["monitored"] and arm[k]["fired"])
        den = np.sqrt(float((TP+FP)*(TP+FN)*(TN+FP)*(TN+FN)))
        print(f"  {nm:26s} TP {TP} FN {FN} FP {FP} TN {TN}  controls {cf}  "
              f"unnecessary {FP/max(FP+TN,1):.4f}  MCC "
              f"{(TP*TN-FP*FN)/den if den>0 else float('nan'):+.4f}")
    nonbust = [k for k in mon if not base[k]["busted"]]
    b = sum(1 for k in nonbust if base[k]["fired"] and not cand[k]["fired"])
    c = sum(1 for k in nonbust if not base[k]["fired"] and cand[k]["fired"])
    lost = sum(1 for k in mon if base[k]["busted"] and not cand[k]["fired"])
    cf = sum(1 for k in base if not base[k]["monitored"] and cand[k]["fired"])
    p = mcnemar(b, c)
    print(f"\npaired against the bounded rule on the same plans:")
    print(f"  false alarms removed  b = {b}")
    print(f"  false alarms added    c = {c}")
    print(f"  busts lost              = {lost}   (must be 0)")
    print(f"  control firings         = {cf}   (must be 0)")
    print(f"  McNemar exact one-sided p = {p:.4f}   (bar 0.05)")
    ok = (lost == 0 and cf == 0 and p <= 0.05)
    print(f"\n  REGISTERED TEST -> {'PASS' if ok else 'FAIL'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
