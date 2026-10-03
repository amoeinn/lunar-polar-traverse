"""Define the executive's sample and score the two constant predictors on it.

PLAN.md registers that no bar on an agreement metric is written until the
constant predictor's score on the same data has been computed. This runs first,
before any threshold for phase 6 exists, and it also fixes the sample by rule
rather than by outcome: the monitored set is every core plan that enters shadow
at all, the control set is every core plan that does not.

Reads the cached replay outputs of the core sweep and nothing else. Scores
nothing about the executive, which does not exist yet.
"""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

BUDGET_H = 2.0              # the core's registered shadow tolerance
WORK = ROOT / "data" / "work" / "sweep"


def load():
    """Per-plan checker outputs for the core's feasible time-dependent plans."""
    rows = []
    for d in sorted(WORK.glob("*_replay_td_tol")):
        tag = d.name[: -len("_replay_td_tol")]
        sh = np.fromfile(d / "out_shadow.bin", dtype=np.float64)
        ar = np.fromfile(d / "out_arrival.bin", dtype=np.float64)
        il = np.fromfile(d / "out_illegal.bin", dtype=np.int32)
        for i in range(sh.size):
            rows.append(dict(tag=tag, plan=i, shadow=float(sh[i]),
                             arrival=float(ar[i]), illegal=int(il[i])))
    return rows


def contingency(fired, busted):
    """Rows are the decision, columns the counterfactual outcome."""
    tp = int(np.sum(fired & busted))
    fp = int(np.sum(fired & ~busted))
    fn = int(np.sum(~fired & busted))
    tn = int(np.sum(~fired & ~busted))
    return tp, fp, fn, tn


def mcc(tp, fp, fn, tn):
    den = np.sqrt(float(tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    return 0.0 if den == 0.0 else (tp * tn - fp * fn) / den


def youden(tp, fp, fn, tn):
    tpr = tp / (tp + fn) if (tp + fn) else 0.0
    fpr = fp / (fp + tn) if (fp + tn) else 0.0
    return tpr - fpr, tpr, fpr


def report(name, fired, busted):
    tp, fp, fn, tn = contingency(fired, busted)
    n = tp + fp + fn + tn
    j, tpr, fpr = youden(tp, fp, fn, tn)
    print(f"  {name}")
    print(f"    fired and busted {tp:3d}   fired, would not have busted {fp:3d}")
    print(f"    MISS, not fired  {fn:3d}   correctly quiet              {tn:3d}")
    print(f"    accuracy {(tp+tn)/n:.4f}   miss rate {fn/(tp+fn) if (tp+fn) else float('nan'):.4f}"
          f"   unnecessary rate {fp/(fp+tn) if (fp+tn) else float('nan'):.4f}")
    print(f"    Matthews {mcc(tp,fp,fn,tn):+.4f}   Youden J {j:+.4f}"
          f"   (TPR {tpr:.4f}, FPR {fpr:.4f})")


def main():
    rows = load()
    sh = np.array([r["shadow"] for r in rows])
    monitored = [r for r in rows if r["shadow"] > 0.0]
    controls = [r for r in rows if r["shadow"] == 0.0]
    print(f"core feasible time-dependent plans: {len(rows)}")
    print(f"  monitored set, plans entering shadow at all: {len(monitored)}")
    print(f"  control set, plans with exactly zero shadow: {len(controls)}")
    print(f"  illegal steps across all plans: {sum(r['illegal'] for r in rows)}")
    print(f"  shadow over the monitored set (h): median "
          f"{np.median([r['shadow'] for r in monitored]):.4f}, max "
          f"{max(r['shadow'] for r in monitored):.4f}")

    epochs = {}
    for r in monitored:
        epochs.setdefault(r["tag"], [0, 0])
        epochs[r["tag"]][0] += 1
        epochs[r["tag"]][1] += int(r["shadow"] > BUDGET_H)
    print(f"\n  epochs contributing to the monitored set, reported not chosen:")
    for tag in sorted(epochs):
        n, b = epochs[tag]
        print(f"    {tag:<16} {n:3d} monitored, {b} busting")

    busted = np.array([r["shadow"] > BUDGET_H for r in monitored])
    print(f"\n  ground truth on the monitored set, from the checker's own "
          f"integration of the executed trajectory:")
    print(f"    plans that bust the {BUDGET_H:.1f} h budget: {int(busted.sum())} "
          f"of {busted.size} = {busted.mean():.2%}")
    print(f"    so the positive class is {int(busted.sum())} plans; any rate on "
          f"it is a rate over that many cases")

    print(f"\nthe two constant predictors, scored before any bar is written:")
    report("never fire", np.zeros_like(busted), busted)
    report("fire at the first tick", np.ones_like(busted), busted)
    print(f"\n  what this says about thresholds. An accuracy bar is unusable: "
          f"'never fire'\n  already scores "
          f"{(~busted).mean():.4f} and no executive worth having needs to beat "
          f"that.\n  Both constant predictors score Matthews 0.0000 and Youden "
          f"J 0.0000, which is\n  why the registered bar goes on those and on "
          f"the miss count, not on accuracy.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
