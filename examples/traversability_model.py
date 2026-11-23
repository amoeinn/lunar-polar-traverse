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

--exclude-host tol05 trains the model the tol05 arm uses: every plan whose
start/goal pair the tol05 host also plans is dropped from both label sources
before anything else, at any axis point and any epoch, and the model is fit only
if the region split's training fold keeps MIN_EPISODES disagreement episodes in
each direction. Below that the arm is reported as not run and nothing is
trained or saved. Both rules are registered in PLAN.md, D132 step 13.
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

# Registered 2026-11-22 before any count: ten events per candidate predictor
# (Peduzzi et al. 1996), the episode as the event, the six features above. A
# floor below which the excluded model is not trained, not an adequacy claim.
MIN_EPISODES = 10 * len(FEATURES)

# Where each host's model arm reads its model from. The core's is trained on
# everything; tol05's without the host's own pairs.
MODEL_PATH = {"core": "data/work/traversability_model.joblib",
              "tol05": "data/work/traversability_model_tol05.joblib"}


def host_pairs(name):
    """The directed (start, goal) pairs of every plan the host's sample holds.

    A plan's first and last cells are its pair: on tol05 they equal the
    planner's pairs file entry for all 66 plans, measured 2026-11-22. No epoch
    in the key, because a plan on the same pair at another epoch walks the same
    cells and the features are per cell.
    """
    import examples.run_executive as R
    return {(int(r["plan"][0]), int(r["plan"][-1]))
            for r in R.sample(R.HOSTS[name])}


def on_pairs(d, pairs):
    """Per sample: does the sample's plan sit on one of these pairs."""
    if "start" not in d or "goal" not in d:
        raise KeyError("the label files carry no start/goal; regenerate them with "
                       "examples/traversability_labels.py")
    key = set(pairs)
    sg = np.column_stack([d["start"].astype(np.int64), d["goal"].astype(np.int64)])
    uniq, inv = np.unique(sg, axis=0, return_inverse=True)
    hit = np.array([(int(a), int(b)) in key for a, b in uniq], bool)
    return hit[inv.ravel()] if hit.size else np.zeros(sg.shape[0], bool)


def without(d, drop):
    """Every per-sample array with the dropped samples removed."""
    keep = ~np.asarray(drop, bool)
    return {k: (v if k == "tag_names" else v[keep]) for k, v in d.items()}


def episode_counts(d, fold):
    """Disagreement episodes in each direction, counted in the fold their first
    sample falls in, so each episode counts once.

    Samples are stored per plan in time order on the 60 s grid, so a maximal run
    of one code within one plan is exactly an episode as src/disagree.py
    episodes() measures it. Asserted, not assumed: a plan's samples must be
    contiguous and increasing in time.
    """
    pid = d["plan_id"].astype(np.int64)
    t = d["t"].astype(float)
    same = np.zeros(pid.size, bool)
    same[1:] = pid[1:] == pid[:-1]
    if np.unique(pid).size != int((~same).sum()):
        raise ValueError("a plan's samples are not contiguous")
    if np.any(t[1:][same[1:]] <= t[:-1][same[1:]]):
        raise ValueError("a plan's samples are not increasing in time")
    _, y = design(d)
    fold = np.asarray(fold, bool)
    out = {}
    for k in (1, 2):
        run = y == k
        prev = np.zeros(run.size, bool)
        prev[1:] = run[:-1] & same[1:]
        out[k] = int((run & ~prev & fold).sum())
    return out


def gate(counts, minimum=None):
    """(run, reason). Both directions must reach the minimum."""
    minimum = MIN_EPISODES if minimum is None else minimum
    ok = counts[1] >= minimum and counts[2] >= minimum
    reason = (f"training fold has {counts[1]} code-1 (table lit, checker dark) and "
              f"{counts[2]} code-2 (table dark, checker lit) episodes; the registered "
              f"minimum is {minimum} in each")
    return ok, reason


def make_classifier(args):
    from sklearn.ensemble import HistGradientBoostingClassifier
    return HistGradientBoostingClassifier(
        max_depth=args.max_depth, max_iter=args.max_iter,
        learning_rate=0.1, l2_regularization=1.0, random_state=0)


LABELS = (ROOT / "data" / "work" / "traversability_held-out.npz",
          ROOT / "data" / "work" / "traversability_sample.npz")


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
    ap.add_argument("--exclude-host", choices=("tol05",), default=None)
    ap.add_argument("--save", default=None,
                    help="default: the host's MODEL_PATH")
    args = ap.parse_args()
    host = args.exclude_host or "core"
    save = MODEL_PATH[host] if args.save is None else args.save

    d = pooled(list(LABELS))
    if args.exclude_host:
        pairs = host_pairs(args.exclude_host)
        drop = on_pairs(d, pairs)
        names = list(d["tag_names"])
        print(f"excluding every plan on one of the {len(pairs)} {args.exclude_host} "
              f"host pairs, at any point and any epoch")
        for q, pa in enumerate(LABELS):
            sel = d["source"] == q
            n_all = np.unique(d["plan_id"][sel]).size
            n_out = np.unique(d["plan_id"][sel & drop]).size
            print(f"  {pa.name}: {n_out} of {n_all} plans, "
                  f"{int((sel & drop).sum())} of {int(sel.sum())} samples")
        # The H check matched a pair within its epoch; this key carries none.
        # Report the plans only the epoch-free key excludes.
        by_epoch = {}
        import examples.run_executive as R
        for r in R.sample(R.HOSTS[args.exclude_host]):
            by_epoch.setdefault(r["tag"], set()).add((int(r["plan"][0]), int(r["plan"][-1])))
        tags = np.array(names)[d["tag_id"].astype(int)]
        same_epoch = np.array([(int(a), int(b)) in by_epoch.get(tg, set())
                               for a, b, tg in zip(d["start"], d["goal"], tags)], bool)
        print(f"  plans excluded only because the pair is a host pair at another "
              f"epoch: {np.unique(d['plan_id'][drop & ~same_epoch]).size}")
        d = without(d, drop)
        if on_pairs(d, pairs).any():
            raise AssertionError("a host pair survived the exclusion")
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

    if args.exclude_host:
        counts = episode_counts(d, ~splits[0][1])
        ok, reason = gate(counts)
        print(f"\n  episodes on the deployed fit's training fold: {reason}")
        if not ok:
            print(f"\n{args.exclude_host} model arm: NOT RUN. {reason}. Nothing is "
                  f"trained and nothing is saved.")
            if save and (ROOT / save).exists():
                # A model left from an earlier run would be read by the arm as
                # if this one had passed; it is not removed here, only refused.
                raise SystemExit(f"{ROOT / save} exists from an earlier run and the "
                                 f"arm would read it; move it aside")
            return 0
    else:
        counts = None

    fitted = None
    for name, te in splits:
        tr = ~te
        clf = make_classifier(args)
        clf.fit(X[tr], y[tr])
        print(f"\n===== split: {name} =====")
        print(f"  train {int(tr.sum())} samples, test {int(te.sum())}")
        score_ternary(y[te], clf.predict(X[te]), "learned, on the test fold")
        score_ternary(y[te], bl[te], "B3b and B3b-lit, on the same test fold")
        if fitted is None:
            fitted = clf
    if save:
        import joblib
        out = ROOT / save
        joblib.dump(dict(model=fitted, features=FEATURES,
                         exclude_host=args.exclude_host, episodes=counts), out)
        print(f"\nsaved the region-split model to {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
