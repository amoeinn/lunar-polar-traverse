"""B0 to B3 on the disagreement target, before any learned model exists.

Registered order and registered framing, from PLAN.md 2026-10-05. The target
is the monitor's own question: among samples where the planner's table says
the cell is lit, is the checker's ray march dark? That is the direction the
executive's contradiction rule acts on, so it is the direction scored here;
the symmetric target, any disagreement, is reported beside it.

Every baseline feeds the monitor and never the planner, so P2 and P3 stay
scored on the planners as they were.

No bar is registered here. The constant predictor's score is printed first
because at this base rate it is high, which is the trap this record has
already fallen into twice, in P5 and P6b.
"""
import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

GRID_N = 500


def confusion(y, p):
    y = np.asarray(y, bool); p = np.asarray(p, bool)
    return (int((y & p).sum()), int((y & ~p).sum()),
            int((~y & p).sum()), int((~y & ~p).sum()))


def metrics(y, p):
    TP, FN, FP, TN = confusion(y, p)
    den = np.sqrt(float((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN)))
    mcc = (TP * TN - FP * FN) / den if den > 0 else float("nan")
    rec = TP / (TP + FN) if TP + FN else float("nan")
    spec = TN / (TN + FP) if TN + FP else float("nan")
    prec = TP / (TP + FP) if TP + FP else float("nan")
    return dict(TP=TP, FN=FN, FP=FP, TN=TN, mcc=mcc, recall=rec,
                precision=prec, youden=rec + spec - 1.0,
                accuracy=(TP + TN) / max(TP + TN + FP + FN, 1))


def load(path):
    d = np.load(path, allow_pickle=False)
    return {k: d[k] for k in d.files}


def region_blocks(cell, n_blocks=5):
    """Spatial block id for each cell, on a coarse grid over the region.

    A region split tests generalisation to near-field terrain the model has
    not seen. It cannot isolate the far field: the horizon reaches to 304 km
    at 80 m, far outside this 10 km square, so every block shares essentially
    all of its far-field occluders. That is a property of the site and no
    split available here removes it.
    """
    i, j = cell // GRID_N, cell % GRID_N
    bi = (i * n_blocks) // GRID_N
    bj = (j * n_blocks) // GRID_N
    return bi * n_blocks + bj


def sweep_threshold(score, y, lo_is_positive=True, n=400):
    """Best MCC over a single threshold, and the threshold that gives it."""
    qs = np.quantile(score, np.linspace(0.001, 0.999, n))
    best = (-2.0, None)
    for t in np.unique(qs):
        p = (score <= t) if lo_is_positive else (score >= t)
        m = metrics(y, p)
        if np.isfinite(m["mcc"]) and m["mcc"] > best[0]:
            best = (m["mcc"], float(t))
    return best


def report(name, y, p, extra=""):
    m = metrics(y, p)
    print(f"  {name:28s} MCC {m['mcc']:+.4f}  J {m['youden']:+.4f}  "
          f"rec {m['recall']:.4f}  prec {m['precision']:.4f}  "
          f"acc {m['accuracy']:.4f}  TP {m['TP']:5d} FP {m['FP']:6d} "
          f"FN {m['FN']:5d}{extra}")
    return m


def pooled(paths):
    """Both label sets in one frame, with epochs remapped to common ids.

    tag ids are assigned per file, so tag 0 in one is not tag 0 in the other;
    the stored tag_names are what make the epoch split meaningful.
    """
    out, names, off = {}, [], 0
    for q, pa in enumerate(paths):
        d = load(pa)
        nm = list(d.pop("tag_names"))
        remap = np.array([names.index(t) if t in names else
                          (names.append(t) or len(names) - 1) for t in nm])
        d["tag_id"] = remap[d["tag_id"].astype(int)]
        d["plan_id"] = d["plan_id"].astype(int) + off
        off = int(d["plan_id"].max()) + 1
        d["source"] = np.full(d["world"].size, q)
        for k, v in d.items():
            out.setdefault(k, []).append(v)
    out = {k: np.concatenate(v) for k, v in out.items()}
    out["tag_names"] = np.array(names)
    return out


def evaluate(d, train, test, label, direction="dark"):
    """One split, one direction.

    direction "dark": among model-lit samples, is the world dark? The condition
      is B3b, the model calling this cell lit and the next dark.
    direction "lit": among model-dark samples, is the world lit? The condition
      is B3b-lit, the mirror: the model calling this cell dark and the next lit.
    """
    w = d["world"].astype(bool); mo = d["model"].astype(bool)
    nl = d["nxt_lit"].astype(bool)
    print(f"\n----- {label}  [{direction} direction] -----")
    for nm, sel0 in (("train", train), ("test", test)):
        sel = sel0 & (mo if direction == "dark" else ~mo)
        y = (~w[sel]) if direction == "dark" else w[sel]
        if y.sum() < 10 or (~y).sum() < 10:
            print(f"  {nm}: unusable, {int(y.sum())} positives of {int(sel.sum())}")
            return None
        mg = np.abs(d["margin"][sel]); hd = d["hor_dist"][sel]
        cond = (~nl[sel]) if direction == "dark" else nl[sel]
        print(f"  {nm}: {int(sel.sum())} samples, {int(y.sum())} positive, "
              f"base rate {100*y.mean():.3f}%, "
              f"{len(np.unique(d['plan_id'][sel]))} plans")
        print(f"    B0 constant                    accuracy "
              f"{100*(1-y.mean()):.3f}%  MCC 0.0000  J 0.0000  recall 0")
        report("B3b" if direction == "dark" else "B3b-lit", y, cond)
        if nm == "train":
            global TAU, TAU2, HMAX
            TAU = sweep_threshold(mg, y)[1]
            best = (-2, TAU, float(hd.max()))
            for ht in np.quantile(hd, np.linspace(0.05, 0.95, 19)):
                m2 = hd <= ht
                if m2.sum() < 200 or y[m2].sum() < 10:
                    continue
                v, t = sweep_threshold(mg[m2], y[m2])
                vv = metrics(y, (mg <= t) & m2)["mcc"]
                if np.isfinite(vv) and vv > best[0]:
                    best = (vv, t, float(ht))
            TAU2, HMAX = best[1], best[2]
            print(f"    thresholds on train: B1 tau {TAU:.4f} deg; "
                  f"B2 tau {TAU2:.4f} deg, dist <= {HMAX:.0f} m")
        report("B1 |margin| <= tau", y, mg <= TAU)
        report("B2 B1 & dist <= h", y, (mg <= TAU2) & (hd <= HMAX))
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-blocks", type=int, default=5)
    args = ap.parse_args()
    d = pooled([ROOT / "data" / "work" / "traversability_held-out.npz",
                ROOT / "data" / "work" / "traversability_sample.npz"])
    w = d["world"].astype(bool); mo = d["model"].astype(bool)
    print(f"pooled: {w.size} samples, {len(np.unique(d['plan_id']))} plans, "
          f"{len(d['tag_names'])} epochs")
    print(f"  world dark & model lit {int((~w & mo).sum())} "
          f"({100*(~w & mo).mean():.3f}%)   world lit & model dark "
          f"{int((w & ~mo).sum())} ({100*(w & ~mo).mean():.3f}%)")
    print(f"  the independent unit is the episode, not the sample: samples at "
          f"a 60 s cadence\n  along one trajectory are strongly autocorrelated.")

    blk = region_blocks(d["cell"].astype(int), args.n_blocks)
    ub = np.unique(blk)
    te_r = np.isin(blk, ub[::2])
    rlab = (f"split by region: {args.n_blocks}x{args.n_blocks} blocks, "
            f"{len(ub[::2])} of {len(ub)} held out, no buffer")
    for direction in ("dark", "lit"):
        evaluate(d, ~te_r, te_r, rlab, direction)
        for tag in ("2028-08-15_p2", "2028-11-15_p2"):
            if tag not in list(d["tag_names"]):
                continue
            te = d["tag_id"] == list(d["tag_names"]).index(tag)
            evaluate(d, ~te, te, f"split by epoch: hold out {tag}", direction)
    return 0


if __name__ == "__main__":
    sys.exit(main())
