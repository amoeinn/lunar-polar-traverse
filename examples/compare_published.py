"""Compare two published products against each other: NAC observed LE/RE vs _MODEL.

Registered in PLAN.md as H1 before it was run. Nothing from src/ enters: this
compares the LROC pipeline's simulated irradiance against its own observed
frames, to decide whether the _MODEL depicts the acquisition instant without
using the model that is itself under suspicion.

Metrics are Cohen's kappa between the Otsu dark mask of the observed frames
and the _MODEL's zero-irradiance mask (the criterion), and Spearman rank
correlation between I/F and irradiance (reported beside it). Each is checked
against a fixture with a known answer before it touches a product.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import rasterio
from rasterio.windows import Window
from scipy.ndimage import map_coordinates
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
NBINS = 256          # Otsu histogram bins, fixed here
SHIFT_PX = 10        # misalignment control, 10 px = 200 m
# The sub-pixel offsets of each frame from the DEM grid, measured in step 2 by an
# independent path (nearest-pixel residuals over the NAC bounding box) and recorded
# in PLAN.md. Asserted here so the resampling is tied to that earlier measurement.
GRID_OFFSET = {"LE": (-0.491, +0.197), "RE": (-0.217, -0.138)}


def otsu(values, nbins=NBINS):
    """Threshold maximising between-class variance. Returns the bin edge."""
    lo, hi = float(values.min()), float(values.max())
    counts, edges = np.histogram(values, bins=nbins, range=(lo, hi))
    w = counts.astype(float) / counts.sum()
    centres = 0.5 * (edges[:-1] + edges[1:])
    w0 = np.cumsum(w)[:-1]
    w1 = 1.0 - w0
    m = np.cumsum(w * centres)
    mu0 = np.divide(m[:-1], w0, out=np.zeros_like(w0), where=w0 > 0)
    mu1 = np.divide(m[-1] - m[:-1], w1, out=np.zeros_like(w1), where=w1 > 0)
    between = w0 * w1 * (mu0 - mu1) ** 2
    return float(edges[1:-1][int(np.argmax(between))])


def kappa(a, b):
    """Cohen's kappa for two boolean labellings of the same pixels."""
    a = np.asarray(a, dtype=bool)
    b = np.asarray(b, dtype=bool)
    n = a.size
    po = float((a == b).sum()) / n
    pa, pb = a.mean(), b.mean()
    pe = pa * pb + (1.0 - pa) * (1.0 - pb)
    return (po - pe) / (1.0 - pe) if pe < 1.0 else float("nan"), po


def rank_average(v):
    order = np.argsort(v, kind="stable")
    ranks = np.empty(v.size, dtype=float)
    sv = v[order]
    i = 0
    while i < v.size:
        j = i
        while j + 1 < v.size and sv[j + 1] == sv[i]:
            j += 1
        ranks[order[i:j + 1]] = 0.5 * (i + j) + 1.0
        i = j + 1
    return ranks


def spearman(x, y):
    rx, ry = rank_average(np.asarray(x, float)), rank_average(np.asarray(y, float))
    rx = rx - rx.mean()
    ry = ry - ry.mean()
    d = np.sqrt((rx * rx).sum() * (ry * ry).sum())
    return float((rx * ry).sum() / d) if d > 0 else float("nan")


def youden(values, positive):
    """Threshold t maximising J = TPR - FPR for the rule "dark where value < t".

    Positives are the pixels the reference calls dark. Ties are handled by
    testing each distinct value as a strict upper bound, so the returned
    threshold is always a value present in the data.
    """
    v = np.asarray(values, float)
    pos = np.sort(v[positive])
    neg = np.sort(v[~positive])
    cand = np.unique(v)
    tpr = np.searchsorted(pos, cand, side="left") / pos.size
    fpr = np.searchsorted(neg, cand, side="left") / neg.size
    j = tpr - fpr
    i = int(np.argmax(j))
    return float(cand[i]), float(j[i]), float(tpr[i]), float(fpr[i])


def fixtures():
    """Each metric answers a case whose answer is known before any product is read."""
    rng = np.random.default_rng(20260922)
    lo = rng.normal(0.10, 0.01, 4000)
    hi = rng.normal(0.60, 0.01, 4000)
    t = otsu(np.concatenate([lo, hi]))
    assert 0.11 < t < 0.59, f"otsu fixture: {t}"
    assert (lo < t).all() and (hi > t).all(), "otsu fixture does not separate the modes"

    # hand-computed: n=100, both mark 30 true, 20 of them the same pixels.
    a = np.zeros(100, bool); a[:30] = True
    b = np.zeros(100, bool); b[10:40] = True
    k, po = kappa(a, b)
    assert abs(po - 0.80) < 1e-12, po
    pe = 0.3 * 0.3 + 0.7 * 0.7
    assert abs(k - (0.80 - pe) / (1 - pe)) < 1e-12, k
    assert abs(k - 0.523809523809524) < 1e-12, k
    assert abs(kappa(a, a)[0] - 1.0) < 1e-12
    assert abs(kappa(a, np.zeros(100, bool))[0]) < 1e-12, "constant predictor must give kappa 0"

    x = rng.normal(size=500).round(1)          # ties on purpose
    y = (2.0 * x + rng.normal(size=500)).round(1)
    mine, theirs = spearman(x, y), float(spearmanr(x, y).statistic)
    assert abs(mine - theirs) < 1e-12, f"spearman {mine} vs scipy {theirs}"
    # Youden: separated populations must give J ~ 1 at a threshold in the gap;
    # identical populations must give J ~ 0.
    yp = rng.normal(0.001, 0.0002, 3000)
    yn = rng.normal(0.020, 0.0020, 3000)   # a clean gap: the optimum is unambiguous
    yv = np.concatenate([yp, yn])
    ymask = np.zeros(yv.size, bool); ymask[:yp.size] = True
    ty, jy, tpr, fpr = youden(yv, ymask)
    assert jy == 1.0 and yp.max() < ty <= yn.min(), f"youden fixture: {ty}, J {jy}"
    same = rng.normal(0.01, 0.003, 6000)
    smask = np.zeros(6000, bool); smask[:3000] = True
    assert youden(same, smask)[1] < 0.05, "youden must give J near 0 on identical populations"

    print(f"fixtures pass: otsu split {t:.4f}, kappa {k:.12f} (scipy-free), "
          f"spearman agrees with scipy to {abs(mine-theirs):.2e}, "
          f"youden J {jy:.4f} at {ty:.6f} on separated fixtures")


def read_model(path, x0, y1, n, px, margin):
    """The _MODEL is on the DEM grid exactly; assert that rather than assume it."""
    with rasterio.open(path) as ds:
        col = (x0 - ds.transform.c) / px
        row = (ds.transform.f - y1) / px
        assert abs(col - round(col)) < 1e-6 and abs(row - round(row)) < 1e-6, \
            f"_MODEL is not on the DEM grid: offset ({col%1:.4f}, {row%1:.4f}) px"
        a = ds.read(1, window=Window(round(col) - margin, round(row) - margin,
                                     n + 2 * margin, n + 2 * margin),
                    boundless=True, fill_value=np.nan).astype(np.float64)
    return a


def read_observed(path, half, x0, y1, n, px):
    """Bilinear resampling of an observed frame onto the DEM cell centres."""
    with rasterio.open(path) as ds:
        fx = ((x0 - ds.transform.c) / px) % 1.0
        fy = ((ds.transform.f - y1) / px) % 1.0
        sx, sy = (fx if fx <= 0.5 else fx - 1.0), (fy if fy <= 0.5 else fy - 1.0)
        ex, ey = GRID_OFFSET[half]
        assert abs(sx - ex) < 1e-3 and abs(sy - ey) < 1e-3, \
            f"{half} offset ({sx:+.4f}, {sy:+.4f}) px does not reproduce step 2's ({ex:+.3f}, {ey:+.3f})"
        c0 = int(np.floor((x0 - ds.transform.c) / px)) - 2
        r0 = int(np.floor((ds.transform.f - y1) / px)) - 2
        src = ds.read(1, window=Window(c0, r0, n + 5, n + 5),
                      boundless=True, fill_value=np.nan).astype(np.float64)
        src = np.where(src < -1e30, np.nan, src)       # declared nodata, if ever stored
        # DEM cell centres in the frame's own pixel coordinates
        cols = (x0 + (np.arange(n) + 0.5) * px - ds.transform.c) / px - 0.5 - c0
        rows = (ds.transform.f - (y1 - (np.arange(n) + 0.5) * px)) / px - 0.5 - r0
        cc, rr = np.meshgrid(cols, rows)
        out = map_coordinates(src, [rr, cc], order=1, mode="constant",
                              cval=np.nan, output=np.float64)
    return out, sx, sy


def report(name, obs, mod_dark, valid, thr):
    dark_obs = obs < thr
    k, po = kappa(dark_obs[valid], mod_dark[valid])
    base = max(dark_obs[valid].mean(), 1.0 - dark_obs[valid].mean())
    print(f"  {name:<10} n {valid.sum():>7}  obs dark {dark_obs[valid].mean():7.4%}  "
          f"model zero {mod_dark[valid].mean():7.4%}  agree {po:7.4%}  "
          f"constant {base:7.4%}  kappa {k:7.4f}")
    return k, po, base


def p6b_thresholds(le, re, model, model_pad, v_le, v_re, v_mod, n, px):
    """P6b step 1: the dark thresholds, and the gate. Our model does not enter."""
    mod_dark = (model == 0.0)
    print("\nP6b thresholds: Youden's J against the _MODEL zero mask, per frame,")
    print("fixed here before our model is compared with anything.")
    print(f"  {'frame':<9} {'threshold':>10} {'J':>8} {'sens':>8} {'spec':>8} "
          f"{'dark frac':>10} {'model zero':>11}")
    thr = {}
    for name, img, vf in (("LE", le, v_le), ("RE", re, v_re)):
        valid = vf & v_mod
        t_y, j, tpr, fpr = youden(img[valid], mod_dark[valid])
        thr[name] = t_y
        print(f"  {name:<9} {t_y:10.6f} {j:8.4f} {tpr:8.4f} {1-fpr:8.4f} "
              f"{(img[valid] < t_y).mean():9.4%} {mod_dark[valid].mean():10.4%}")

    obs_dark = np.where(v_le, le < thr["LE"], re < thr["RE"])
    comb = (v_le | v_re) & v_mod
    sets = (("LE", v_le & v_mod), ("RE", v_re & v_mod), ("COMBINED", comb))
    print("\nagreement of that dark mask with the _MODEL zero mask, beside a constant:")
    for name, valid in sets:
        k, po = kappa(obs_dark[valid], mod_dark[valid])
        base = max(obs_dark[valid].mean(), 1 - obs_dark[valid].mean())
        print(f"  {name:<9} n {valid.sum():>7}  agree {po:8.4%}  constant {base:8.4%}  "
              f"kappa {k:7.4f}")

    print(f"\nP6b gate: the {SHIFT_PX} px ({SHIFT_PX*px:.0f} m) four-direction shift "
          f"control must drop kappa by at least 0.2.")
    k0, _ = kappa(obs_dark[comb], mod_dark[comb])
    drops = []
    for dr, dc, lab in ((-SHIFT_PX, 0, "north"), (SHIFT_PX, 0, "south"),
                        (0, SHIFT_PX, "east"), (0, -SHIFT_PX, "west")):
        sub = model_pad[SHIFT_PX + dr:SHIFT_PX + dr + n, SHIFT_PX + dc:SHIFT_PX + dc + n]
        ks, _ = kappa(obs_dark[comb], (sub == 0.0)[comb])
        drops.append(k0 - ks)
        print(f"  {lab:<9} kappa {ks:+7.4f}  drop {k0-ks:+7.4f}")
    print(f"  unshifted kappa {k0:+7.4f}; smallest drop {min(drops):.4f}, "
          f"largest {max(drops):.4f}")
    ok = min(drops) >= 0.2
    print(f"  -> gate {'PASSES' if ok else 'FAILS'} (every direction must drop by 0.2; "
          f"smallest is {min(drops):.4f})")
    return 0 if ok else 2


def diagnose(obs_if, obs_dark, model, model_pad, valid, v_le, le, re, t_le, t_re, n, px):
    """Why the criterion reads as it does. Diagnostic only; it scores nothing."""
    v = obs_if[valid]
    mz = (model == 0.0)[valid]
    print("\n--- diagnostics, not the criterion ---")
    qs = [0, 1, 5, 10, 25, 50, 75, 90, 95, 99, 100]
    print("  observed I/F percentiles, combined valid set:")
    print("   " + "  ".join(f"p{q}={np.percentile(v, q):.5f}" for q in qs))
    print(f"  I/F where the _MODEL is zero   : median {np.median(v[mz]):.5f}, "
          f"mean {v[mz].mean():.5f}, p90 {np.percentile(v[mz], 90):.5f}, n {mz.sum()}")
    print(f"  I/F where the _MODEL is nonzero: median {np.median(v[~mz]):.5f}, "
          f"mean {v[~mz].mean():.5f}, p10 {np.percentile(v[~mz], 10):.5f}, n {(~mz).sum()}")

    # Threshold-free separation: P(I/F of a model-lit pixel > I/F of a model-zero pixel).
    r = rank_average(v)
    n1, n0 = int((~mz).sum()), int(mz.sum())
    auc = (r[~mz].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0)
    print(f"  AUC of I/F separating _MODEL lit from _MODEL zero: {auc:.4f} "
          f"(0.5 is no separation, 1.0 is perfect)")

    # What the criterion's threshold does, and what a threshold that is not the
    # question would do. Diagnostic: the registered criterion stands as computed.
    print(f"  the Otsu thresholds call {obs_dark[valid].mean():.4%} of pixels dark; "
          f"the _MODEL calls {mz.mean():.4%} zero")
    frac = mz.mean()
    t_q = np.quantile(v, frac)
    dq = obs_if < t_q
    kq, poq = kappa(dq[valid], mz)
    print(f"  quantile-matched threshold {t_q:.6f} (same dark fraction as the _MODEL): "
          f"agree {poq:.4%}, kappa {kq:.4f}")
    print("  misalignment control at that threshold:")
    for dr, dc, lab in ((0, 0, "unshifted"), (-SHIFT_PX, 0, "north"), (SHIFT_PX, 0, "south"),
                        (0, SHIFT_PX, "east"), (0, -SHIFT_PX, "west")):
        sub = model_pad[SHIFT_PX + dr:SHIFT_PX + dr + n, SHIFT_PX + dc:SHIFT_PX + dc + n]
        ks, _ = kappa(dq[valid], (sub == 0.0)[valid])
        print(f"    {lab:<10} kappa {ks:+7.4f}")
    print("--- end diagnostics ---")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", default="M139729686")
    ap.add_argument("--thresholds-only", action="store_true")
    ap.add_argument("--youden", action="store_true",
                    help="P6b: Youden thresholds and the shift-control gate, no model")
    ap.add_argument("--diagnose", action="store_true",
                    help="threshold-free diagnostics, run after the criterion is applied")
    args = ap.parse_args()

    fixtures()
    p = json.loads((DATA / "work" / "params.json").read_text())
    n, px = p["region_nx"], p["region_px"]
    x0, y1 = p["region_ox"], p["region_oy"]
    print(f"region {n}x{n} at {px} m, origin ({x0}, {y1}); image {args.image}")

    model_pad = read_model(DATA / "nac" / f"{args.image}_20MPP_MODEL.TIF",
                           x0, y1, n, px, SHIFT_PX)
    model = model_pad[SHIFT_PX:SHIFT_PX + n, SHIFT_PX:SHIFT_PX + n]
    le, lfx, lfy = read_observed(DATA / "nac" / f"{args.image}LE_20MPP.TIF", "LE", x0, y1, n, px)
    re, rfx, rfy = read_observed(DATA / "nac" / f"{args.image}RE_20MPP.TIF", "RE", x0, y1, n, px)
    print(f"sub-pixel offsets: LE ({lfx:+.3f}, {lfy:+.3f}) px, RE ({rfx:+.3f}, {rfy:+.3f}) px")

    v_mod = np.isfinite(model)
    v_le, v_re = np.isfinite(le), np.isfinite(re)
    print(f"valid after resampling: _MODEL {v_mod.mean():.4%}, LE {v_le.mean():.4%}, "
          f"RE {v_re.mean():.4%}, either {(v_le | v_re).mean():.4%}, "
          f"both {(v_le & v_re).mean():.4%}")

    t_le = otsu(le[v_le])
    t_re = otsu(re[v_re])
    print(f"\nOtsu thresholds, {NBINS} bins, each on its own frame's valid I/F over "
          f"the region, stated before any kappa:")
    print(f"  LE threshold {t_le:.6f}  (I/F range {le[v_le].min():.6f} to {le[v_le].max():.6f})")
    print(f"  RE threshold {t_re:.6f}  (I/F range {re[v_re].min():.6f} to {re[v_re].max():.6f})")
    print(f"  _MODEL irradiance range {model[v_mod].min():.4f} to {model[v_mod].max():.4f} "
          f"W/m^2/sr; zero on {(model[v_mod] == 0.0).mean():.4%} of valid pixels")
    if args.thresholds_only:
        return 0

    mod_dark = (model == 0.0)
    if args.youden:
        return p6b_thresholds(le, re, model, model_pad, v_le, v_re, v_mod, n, px)
    print("\nper frame and combined, on the registered scoring set:")
    sets = {"LE": v_le & v_mod, "RE": v_re & v_mod}
    for name, valid in sets.items():
        report(name, le if name == "LE" else re, mod_dark, valid,
               t_le if name == "LE" else t_re)

    # combined: LE where valid, RE elsewhere; each pixel uses its own frame's threshold
    comb_valid = (v_le | v_re) & v_mod
    obs_dark = np.where(v_le, le < t_le, re < t_re)
    k, po = kappa(obs_dark[comb_valid], mod_dark[comb_valid])
    base = max(obs_dark[comb_valid].mean(), 1.0 - obs_dark[comb_valid].mean())
    print(f"  {'COMBINED':<10} n {comb_valid.sum():>7}  obs dark "
          f"{obs_dark[comb_valid].mean():7.4%}  model zero {mod_dark[comb_valid].mean():7.4%}  "
          f"agree {po:7.4%}  constant {base:7.4%}  kappa {k:7.4f}")

    obs_if = np.where(v_le, le, re)
    print("\nSpearman between observed I/F and _MODEL irradiance "
          "(constant predictor: undefined, reported as 0):")
    for name, valid in list(sets.items()) + [("COMBINED", comb_valid)]:
        src = le if name == "LE" else (re if name == "RE" else obs_if)
        print(f"  {name:<10} n {valid.sum():>7}  rho {spearman(src[valid], model[valid]):+7.4f}")

    print(f"\nmisalignment control: kappa with the _MODEL mask shifted {SHIFT_PX} px "
          f"({SHIFT_PX*px:.0f} m), combined set, observed mask held fixed:")
    for dr, dc, lab in ((0, 0, "unshifted"), (-SHIFT_PX, 0, "north"), (SHIFT_PX, 0, "south"),
                        (0, SHIFT_PX, "east"), (0, -SHIFT_PX, "west")):
        sub = model_pad[SHIFT_PX + dr:SHIFT_PX + dr + n, SHIFT_PX + dc:SHIFT_PX + dc + n]
        ks, _ = kappa(obs_dark[comb_valid], (sub == 0.0)[comb_valid])
        print(f"  {lab:<10} kappa {ks:+7.4f}")

    if args.diagnose:
        diagnose(obs_if, obs_dark, model, model_pad, comb_valid, v_le, le, re, t_le, t_re, n, px)

    print(f"\nH1 criterion: kappa (combined) = {k:.4f}")
    print("  >= 0.80 products agree; <= 0.50 products disagree; between is inconclusive")
    verdict = "AGREE" if k >= 0.8 else ("DISAGREE" if k <= 0.5 else "INCONCLUSIVE")
    print(f"  -> {verdict}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
