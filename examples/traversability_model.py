"""The learned traversability model: the ternary correction, depth-limited.

Predicts the code the executive's stream carries, from features a monitor can
get cheaply at execution time:
    0 believe the model   1 treat the step as dark   2 treat it as lit
which is label 1 where the planner's table says lit and the checker says dark,
label 2 where the table says dark and the checker says lit, and 0 elsewhere.

Depth is limited because the effective sample size is a few hundred episodes and
not the 162290 samples: samples at a 60 s cadence along one trajectory are
strongly autocorrelated, so the independent unit is the episode. A model with
more capacity than that supports would score well on a split that leaks.

Features are deliberately the baselines' information plus more, so the
comparison is fair in the direction that matters: B3b and B3b-lit are exactly a
two-feature rule on model and nxt_lit, both of which are given here.
"""
import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from examples.traversability_baselines import (load, metrics, pooled,  # noqa: E402
                                               region_blocks)

FEATURES = ("model", "nxt_lit", "margin", "sun_el", "hor_el", "hor_dist")


def design(d):
    X = np.column_stack([d[k].astype(float) for k in FEATURES])
    w = d["world"].astype(bool); m = d["model"].astype(bool)
    y = np.zeros(w.size, np.int64)
    y[m & ~w] = 1          # table says lit, checker says dark -> treat as dark
    y[~m & w] = 2          # table says dark, checker says lit -> treat as lit
    return X, y


def score_ternary(y, p, label):
    print(f"\n  {label}")
    print(f"    {'code':>6}{'n':>8}{'predicted':>11}{'recall':>9}"
          f"{'precision':>11}{'MCC':>9}")
    out = {}
    for k, nm in ((1, "1 dark"), (2, "2 lit")):
        yy, pp = (y == k), (p == k)
        mm = metrics(yy, pp)
        out[k] = mm
        print(f"    {nm:>6}{int(yy.sum()):8d}{int(pp.sum()):11d}"
              f"{mm['recall']:9.4f}{mm['precision']:11.4f}{mm['mcc']:+9.4f}")
    acc = float((y == p).mean())
    print(f"    overall accuracy {acc:.4f}; the constant predictor 'all 0' "
          f"scores {float((y == 0).mean()):.4f}")
    return out


def baseline_codes(d):
    """B3b and B3b-lit together, as the ternary stream the executive was given."""
    m = d["model"].astype(bool); nl = d["nxt_lit"].astype(bool)
    p = np.zeros(m.size, np.int64)
    p[m & ~nl] = 1
    p[~m & nl] = 2
    return p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-depth", type=int, default=3)
    ap.add_argument("--max-iter", type=int, default=60)
    ap.add_argument("--n-blocks", type=int, default=5)
    ap.add_argument("--save", default="data/work/traversability_model.joblib")
    args = ap.parse_args()
    from sklearn.ensemble import HistGradientBoostingClassifier

    d = pooled([ROOT / "data" / "work" / "traversability_held-out.npz",
                ROOT / "data" / "work" / "traversability_sample.npz"])
    X, y = design(d)
    bl = baseline_codes(d)
    print(f"pooled {y.size} samples, {len(np.unique(d['plan_id']))} plans, "
          f"{len(d['tag_names'])} epochs")
    print(f"  label counts: 0 {int((y==0).sum())}  1 {int((y==1).sum())}  "
          f"2 {int((y==2).sum())}")
    print(f"  features: {', '.join(FEATURES)}")
    print(f"  depth limit {args.max_depth}, {args.max_iter} iterations")

    blk = region_blocks(d["cell"].astype(int), args.n_blocks)
    ub = np.unique(blk)
    splits = [("region, 12 of 24 blocks held out", np.isin(blk, ub[::2]))]
    for tag in ("2028-08-15_p2", "2028-11-15_p2"):
        if tag in list(d["tag_names"]):
            splits.append((f"epoch, hold out {tag}",
                           d["tag_id"] == list(d["tag_names"]).index(tag)))

    fitted = None
    for name, te in splits:
        tr = ~te
        clf = HistGradientBoostingClassifier(
            max_depth=args.max_depth, max_iter=args.max_iter,
            learning_rate=0.1, l2_regularization=1.0,
            random_state=0)
        clf.fit(X[tr], y[tr])
        print(f"\n===== split: {name} =====")
        print(f"  train {int(tr.sum())} samples, test {int(te.sum())}")
        score_ternary(y[te], clf.predict(X[te]), "learned, on the test fold")
        score_ternary(y[te], bl[te], "B3b and B3b-lit, on the same test fold")
        if fitted is None:
            fitted = clf
    if args.save:
        import joblib
        out = ROOT / args.save
        joblib.dump(dict(model=fitted, features=FEATURES), out)
        print(f"\nsaved the region-split model to {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
