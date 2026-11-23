"""Define the executive's sample and score the two constant predictors on it.

PLAN.md registers that no bar on an agreement metric is written until the
constant predictor's score on the same data has been computed. This runs first,
before any threshold is written for a host, and it fixes the sample by rule
rather than by outcome: the monitored set is every plan of the host that enters
shadow at all, the control set is every plan that does not, and a bust is
shadow above the host's own budget.

Reads the cached replay outputs of the host and nothing else. Scores nothing
about the executive. --host core is the sample phase 6 registered on; --host
tol05 is the primary branch's host.
"""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import argparse                                                  # noqa: E402
from math import comb                                            # noqa: E402

import examples.run_executive as R                               # noqa: E402


def load(host):
    """Per-plan checker outputs for the host's feasible time-dependent plans.

    The partition and the bust label come from run_executive.sample, the same
    function the executive and the correction arm read, so the sample these
    base rates describe is the sample those runs score and not a second copy of
    its definition.
    """
    rows = []
    recs = R.sample(host)
    illegal = {}
    for r in recs:
        if r["tag"] not in illegal:
            d = Path(r["replay_params"]).parent
            illegal[r["tag"]] = np.fromfile(d / "out_illegal.bin", dtype=np.int32)
        rows.append(dict(tag=r["tag"], plan=r["plan_index"], shadow=r["world_shadow"],
                         busted=r["busted"], monitored=r["monitored"],
                         illegal=int(illegal[r["tag"]][r["plan_index"]])))
    return rows


def mcnemar_p(b, c):
    """One-sided exact McNemar p for b removed against c added."""
    n = b + c
    return sum(comb(n, k) for k in range(c + 1)) / 2 ** n if n else 1.0


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
    den = float(tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)
    m = f"{mcc(tp,fp,fn,tn):+.4f}" if den else "undefined, zero denominator"
    print(f"    Matthews {m}   Youden J {j:+.4f}"
          f"   (TPR {tpr:.4f}, FPR {fpr:.4f})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", choices=sorted(R.HOSTS), default="core")
    host = R.HOSTS[ap.parse_args().host]
    budget = float(host["budget_h"])
    rows = load(host)
    monitored = [r for r in rows if r["monitored"]]
    controls = [r for r in rows if not r["monitored"]]
    print(f"host {host['name']}, budget {budget:g} h: feasible time-dependent plans: {len(rows)}")
    print(f"  monitored set, plans entering shadow at all: {len(monitored)}")
    print(f"  control set, plans with exactly zero shadow: {len(controls)}")
    print(f"  illegal steps across all plans: {sum(r['illegal'] for r in rows)}")
    print(f"  shadow over the monitored set (h): median "
          f"{np.median([r['shadow'] for r in monitored]):.4f}, max "
          f"{max(r['shadow'] for r in monitored):.4f}")

    tags = sorted({r["tag"] for r in rows})
    print(f"\n  per epoch, reported not chosen:")
    for tag in tags:
        g = [r for r in rows if r["tag"] == tag]
        m = [r for r in g if r["monitored"]]
        print(f"    {tag:<16} {len(g):3d} plans, {len(m):3d} monitored, "
              f"{len(g) - len(m):3d} controls, {sum(r['busted'] for r in m)} busting")

    busted = np.array([r["busted"] for r in monitored])
    print(f"\n  ground truth on the monitored set, from the checker's own "
          f"integration of the executed trajectory:")
    print(f"    plans that bust the {budget:g} h budget: {int(busted.sum())} "
          f"of {busted.size} = {busted.mean():.2%} (of all plans "
          f"{int(busted.sum()) / len(rows):.2%})")
    print(f"    so the positive class is {int(busted.sum())} plans; any rate on "
          f"it is a rate over that many cases")
    for r in monitored:
        if r["busted"]:
            print(f"      case: {r['tag']} replay plan {r['plan']}, shadow "
                  f"{r['shadow']:.4f} h, illegal {r['illegal']}")

    print(f"\nthe two constant predictors, scored before any bar is written:")
    report("never fire", np.zeros_like(busted), busted)
    report("fire at the first tick", np.ones_like(busted), busted)
    print(f"    on the {len(controls)} controls: never fire fires on 0, fire at "
          f"the first tick on {len(controls)}")

    print(f"\n  leaving one epoch out, what a monitor that fires on exactly the "
          f"busts would score:")
    for tag in tags:
        m = [r for r in monitored if r["tag"] != tag]
        b = np.array([r["busted"] for r in m], bool)
        tp, fp, fn, tn = contingency(b, b)
        den = float(tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)
        print(f"    without {tag:<16} monitored {len(m):3d}  busts {int(b.sum())}  "
              f"Matthews {f'{mcc(tp, fp, fn, tn):+.4f}' if den else 'UNDEFINED'}")

    quiet = int((~busted).sum())
    print(f"\n  the paired McNemar test is played on the {quiet} non-bust monitored "
          f"plans: b and c are\n  counted there, so b + c <= {quiet}. The smallest "
          f"b reaching one-sided p <= 0.05, per c:")
    for c in range(0, 7):
        b = next((b for b in range(0, 200) if mcnemar_p(b, c) <= 0.05), None)
        print(f"    c = {c}  b >= {b:2d}  p = {mcnemar_p(b, c):.4f}  "
              f"{'reachable' if b + c <= quiet else 'UNREACHABLE on this host'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
