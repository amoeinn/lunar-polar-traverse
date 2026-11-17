"""Score P6 and P6b: our shadow mask at the NAC label epoch against the observed frames.

The observed dark masks come from two thresholds, both fixed before this ran:
P6's Otsu thresholds and P6b's Youden thresholds, both recorded in PLAN.md.
The resampling, the scoring set and the metrics are the ones H1 used, imported
from examples/compare_published.py rather than restated, so there is one
implementation of each.

Our model enters here for the first time in this sequence, and only here.
"""
import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import rasterio
import spiceypy as sp
from rasterio.windows import Window, from_bounds
from scipy.ndimage import distance_transform_edt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.illumination import RegionCells, shadow_mask, sun_angles   # noqa: E402
from src.sun import sun_positions_m  # noqa: E402

spec = importlib.util.spec_from_file_location("cp", ROOT / "examples" / "compare_published.py")
cp = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cp)

DATA = ROOT / "data"
KERNELS = ["naif0012.tls", "pck00010.tpc", "moon_pa_de421_1900-2050.bpc",
           "moon_080317.tf", "de421.bsp"]
FRAME = "MOON_ME_DE421"
EPOCH = "2010-09-21T17:34:00.5"        # label midpoint, PLAN.md
LABEL_AZ, LABEL_EL = 8.989, 1.105      # recorded in PLAN.md step 8
SOLAR_RADIUS_DEG = 0.2666
EDGES = (("first light", +SOLAR_RADIUS_DEG), ("point source", 0.0),
         ("full sun", -SOLAR_RADIUS_DEG))
# Thresholds fixed before this script ran; PLAN.md steps 9 and 10.
OTSU = {"LE": 0.013051, "RE": 0.018386}
YOUDEN = {"LE": 0.003307, "RE": 0.004581}
SHIFT_PX = cp.SHIFT_PX


def shifted_kappa(obs_dark, ours, valid, dr, dc):
    """kappa with our mask displaced by (dr, dc), over the in-region overlap only."""
    n = ours.shape[0]
    r0, r1 = max(0, -dr), min(n, n - dr)
    c0, c1 = max(0, -dc), min(n, n - dc)
    sl = (slice(r0, r1), slice(c0, c1))
    sh = (slice(r0 + dr, r1 + dr), slice(c0 + dc, c1 + dc))
    v = valid[sl]
    return cp.kappa(obs_dark[sl][v], ours[sh][v])[0], int(v.sum())


def edge_distance(mask):
    """Distance in pixels to the nearest boundary of a boolean mask."""
    e = np.zeros_like(mask, dtype=bool)
    e[:-1, :] |= mask[:-1, :] != mask[1:, :]
    e[1:, :] |= mask[:-1, :] != mask[1:, :]
    e[:, :-1] |= mask[:, :-1] != mask[:, 1:]
    e[:, 1:] |= mask[:, :-1] != mask[:, 1:]
    return distance_transform_edt(~e), e


def score(label, obs_dark, ours, valid, name):
    k, po = cp.kappa(obs_dark[valid], ours[valid])
    base = max(obs_dark[valid].mean(), 1.0 - obs_dark[valid].mean())
    print(f"  {label:<9} {name:<9} n {valid.sum():>7}  obs dark {obs_dark[valid].mean():7.4%}  "
          f"ours {ours[valid].mean():7.4%}  agree {po:8.4%}  constant {base:8.4%}  "
          f"kappa {k:7.4f}")
    return po, k, base


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", default="M139729686")
    args = ap.parse_args()

    cp.fixtures()
    p = json.loads((DATA / "work" / "params.json").read_text())
    n, px, x0, y1 = p["region_nx"], p["region_px"], p["region_ox"], p["region_oy"]
    naz = p["n_azimuth"]
    side = n * px
    horizon = np.fromfile(p["out_horizon"], dtype=np.float32).reshape(n, n, naz)
    with rasterio.open(DATA / "lola" / "LDEM_80S_20MPP_ADJ.TIF") as ds:
        w = from_bounds(x0, y1 - side, x0 + side, y1, ds.transform)
        dem = ds.read(1, window=Window(round(w.col_off), round(w.row_off), n, n))
    print(f"region {n}x{n} at {px} m; horizon {horizon.shape}; "
          f"sidecar commit {p['source_commit']}")

    for k_ in KERNELS:
        sp.furnsh(str(DATA / "kernels" / k_))
    et = sp.str2et(EPOCH)
    sun = sun_positions_m(et, FRAME)[0]

    xs = x0 + (np.arange(n) + 0.5) * px
    ys = y1 - (np.arange(n) + 0.5) * px
    gx, gy = np.meshgrid(xs, ys)
    cells = RegionCells(gx.ravel(), gy.ravel(), dem.ravel().astype(float),
                        radius=p["radius_m"], observer_height=p["observer_height_m"])
    centre = RegionCells(np.array([x0 + side / 2]), np.array([y1 - side / 2]),
                         np.array([0.0]), radius=p["radius_m"])
    caz, cel = sun_angles(centre, sun)
    print(f"label epoch {EPOCH}: sun at region centre azimuth {float(caz[0]):.3f}, "
          f"elevation {float(cel[0]):.3f}")
    assert abs(float(caz[0]) - LABEL_AZ) < 5e-3 and abs(float(cel[0]) - LABEL_EL) < 5e-3, \
        "label-epoch geometry does not reproduce what PLAN.md step 8 recorded"

    profiles = horizon.reshape(n * n, naz).astype(np.float64)
    masks = {}
    for name, edge in EDGES:
        masks[name] = shadow_mask(cells, sun, profiles, edge_deg=edge).reshape(n, n)
        print(f"  our shadow fraction, {name:<12} {masks[name].mean():7.4%}")
    sp.kclear()

    model_pad = cp.read_model(DATA / "nac" / f"{args.image}_20MPP_MODEL.TIF",
                              x0, y1, n, px, SHIFT_PX)
    model = model_pad[SHIFT_PX:SHIFT_PX + n, SHIFT_PX:SHIFT_PX + n]
    le, _, _ = cp.read_observed(DATA / "nac" / f"{args.image}LE_20MPP.TIF", "LE", x0, y1, n, px)
    re, _, _ = cp.read_observed(DATA / "nac" / f"{args.image}RE_20MPP.TIF", "RE", x0, y1, n, px)
    v_mod, v_le, v_re = np.isfinite(model), np.isfinite(le), np.isfinite(re)
    comb = (v_le | v_re) & v_mod
    sets = (("LE", v_le & v_mod), ("RE", v_re & v_mod), ("COMBINED", comb))
    print(f"scoring set: LE {int((v_le&v_mod).sum())}, RE {int((v_re&v_mod).sum())}, "
          f"COMBINED {int(comb.sum())} pixels")

    results = {}
    for tag, thr in (("P6", OTSU), ("P6b", YOUDEN)):
        dark = {"LE": le < thr["LE"], "RE": re < thr["RE"]}
        dark["COMBINED"] = np.where(v_le, dark["LE"], dark["RE"])
        print(f"\n{tag}: observed dark by {'Otsu' if tag == 'P6' else 'Youden'} "
              f"thresholds LE {thr['LE']:.6f}, RE {thr['RE']:.6f}")
        for ename, _ in EDGES:
            for sname, valid in sets:
                r = score(ename, dark[sname], masks[ename], valid, sname)
                results[(tag, ename, sname)] = r
            print()
        results[(tag, "dark")] = dark

    print("gate on the scored comparison: our mask shifted "
          f"{SHIFT_PX} px ({SHIFT_PX*px:.0f} m), observed dark held fixed, point source,")
    print("over the in-region overlap; replacement gate requires every drop > 0.05")
    for tag in ("P6", "P6b"):
        dark = results[(tag, "dark")]["COMBINED"]
        k0 = results[(tag, "point source", "COMBINED")][1]
        drops = []
        line = []
        for dr, dc, lab in ((-SHIFT_PX, 0, "north"), (SHIFT_PX, 0, "south"),
                            (0, SHIFT_PX, "east"), (0, -SHIFT_PX, "west")):
            ks, nn = shifted_kappa(dark, masks["point source"], comb, dr, dc)
            drops.append(k0 - ks)
            line.append(f"{lab} {ks:+.4f} (drop {k0-ks:+.4f}, n {nn})")
        print(f"  {tag}: unshifted {k0:+.4f}; " + "; ".join(line))
        print(f"    smallest drop {min(drops):.4f} -> "
              f"{'PASSES' if min(drops) > 0.05 else 'FAILS'} the replacement gate "
              f"(> 0.05); registered 0.2 gate would {'pass' if min(drops) > 0.2 else 'fail'}")

    print("\n_MODEL-shifted form, reported alongside; it guards the threshold, not the score:")
    for tag, thr in (("P6", OTSU), ("P6b", YOUDEN)):
        dark = results[(tag, "dark")]["COMBINED"]
        k0, _ = cp.kappa(dark[comb], (model == 0.0)[comb])
        ds = []
        for dr, dc, lab in ((-SHIFT_PX, 0, "north"), (SHIFT_PX, 0, "south"),
                            (0, SHIFT_PX, "east"), (0, -SHIFT_PX, "west")):
            sub = model_pad[SHIFT_PX + dr:SHIFT_PX + dr + n, SHIFT_PX + dc:SHIFT_PX + dc + n]
            ks, _ = cp.kappa(dark[comb], (sub == 0.0)[comb])
            ds.append(k0 - ks)
        print(f"  {tag}: unshifted {k0:+.4f}, drops "
              f"{', '.join(f'{d:+.4f}' for d in ds)}, smallest {min(ds):.4f}")

    print("\nwhere the disagreement sits, relative to the edges of our own shadow mask")
    print("(point source; distance in pixels of 20 m, 0 means the pixel touches an edge):")
    dist, edges = edge_distance(masks["point source"])
    print(f"  our mask has {int(edges.sum())} edge pixels, {edges.mean():.4%} of the region")
    for tag in ("P6", "P6b"):
        dark = results[(tag, "dark")]["COMBINED"]
        wrong = (dark != masks["point source"]) & comb
        dd, da = dist[wrong], dist[comb]
        print(f"  {tag}: {int(wrong.sum())} disagreeing of {int(comb.sum())} "
              f"({wrong.sum()/comb.sum():.4%}), median distance to an edge "
              f"{np.median(dd):.2f} px against {np.median(da):.2f} px for all valid")
        for r in (0, 1, 2, 3, 5, 10):
            print(f"    within {r:>2} px: {(dd <= r).mean():7.4%} of disagreements, "
                  f"{(da <= r).mean():7.4%} of all valid pixels, "
                  f"ratio {((dd <= r).mean()/(da <= r).mean()) if (da<=r).mean() else float('nan'):5.2f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
