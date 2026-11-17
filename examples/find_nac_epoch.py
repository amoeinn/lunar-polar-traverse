"""Determine the sun geometry of a published NAC _MODEL image by epoch scan.

Scans every epoch in a window, comparing our shadow mask against the _MODEL's
zero-irradiance mask, and reports the best-agreeing epochs. The scan itself is
src.illumination.scan_epochs, which is round-trip tested and mutation-checked
in tests/test_epoch_scan.py.

The result determines the sun's direction at the image epoch, not the date:
the same geometry recurs, so the archive is still needed for the date. And the
direction is chosen to maximise agreement, so the peak agreement is an upper
bound and not evidence for the model. See PLAN.md, check 1 and P6.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import rasterio
import spiceypy as sp
from rasterio.windows import Window, from_bounds

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.frames import local_frame                       # noqa: E402
from src.illumination import RegionCells, scan_epochs, sun_angles  # noqa: E402
from src.sun import sun_positions_m  # noqa: E402

DATA = ROOT / "data"
KERNELS = ["naif0012.tls", "pck00010.tpc", "moon_pa_de421_1900-2050.bpc",
           "moon_080317.tf", "de421.bsp"]
FRAME = "MOON_ME_DE421"
N_CELLS = 12000
SEED = 20260922
TARGET_AZ, TARGET_AZ_TOL = 354.8, 1.0        # registered in PLAN.md, P6
TARGET_EL, TARGET_EL_TOL = 1.10, 0.1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=str(DATA / "nac" / "M139729686_20MPP_MODEL.TIF"))
    ap.add_argument("--start", default="2009-06-23T00:00:00")
    ap.add_argument("--stop", default="2024-01-01T00:00:00")
    ap.add_argument("--step", type=float, default=3600.0)
    ap.add_argument("--limit", type=int, default=0, help="only scan this many epochs")
    args = ap.parse_args()

    p = json.loads((DATA / "work" / "params.json").read_text())
    n, naz = p["region_nx"], p["n_azimuth"]
    horizon = np.fromfile(p["out_horizon"], dtype=np.float32).reshape(n, n, naz)

    x0, y1 = p["region_ox"], p["region_oy"]
    x1, y0 = x0 + n * p["region_px"], y1 - n * p["region_px"]
    with rasterio.open(args.model) as ds:
        w = from_bounds(x0, y0, x1, y1, ds.transform)
        model = ds.read(1, window=Window(round(w.col_off), round(w.row_off), n, n))
    with rasterio.open(DATA / "lola" / "LDEM_80S_20MPP_ADJ.TIF") as ds:
        w = from_bounds(x0, y0, x1, y1, ds.transform)
        dem = ds.read(1, window=Window(round(w.col_off), round(w.row_off), n, n))

    rng = np.random.default_rng(SEED)
    sel = rng.choice(n * n, size=N_CELLS, replace=False)
    rows, cols = sel // n, sel % n
    xs = x0 + (cols + 0.5) * p["region_px"]
    ys = y1 - (rows + 0.5) * p["region_px"]
    cells = RegionCells(xs, ys, dem[rows, cols].astype(float),
                        radius=p["radius_m"], observer_height=p["observer_height_m"])
    profiles = horizon[rows, cols].astype(np.float64)
    dark = (model[rows, cols] == 0.0)
    print(f"{N_CELLS} cells (seed {SEED}); the _MODEL calls {dark.mean():.2%} shadowed")

    for k in KERNELS:
        sp.furnsh(str(DATA / "kernels" / k))
    ets = np.arange(sp.str2et(args.start), sp.str2et(args.stop), args.step)
    if args.limit:
        ets = ets[:args.limit]
    print(f"scanning {ets.size} epochs, {sp.et2utc(ets[0],'C',0)} to {sp.et2utc(ets[-1],'C',0)}")
    suns = sun_positions_m(ets, FRAME)

    t0 = time.time()
    agreement, baseline = scan_epochs(cells, suns, profiles, dark)
    print(f"scan took {time.time()-t0:.1f} s")

    centre = RegionCells(np.array([x0 + n * p["region_px"] / 2]),
                         np.array([y1 - n * p["region_px"] / 2]), np.array([0.0]),
                         radius=p["radius_m"])
    print(f"\npeak agreement {agreement.max():.4%}; always-lit baseline {baseline:.4%}; "
          f"always-dark {1-baseline:.4%}")
    order = np.argsort(agreement)[::-1]
    print("\nten best epochs (separated by at least 48 h):")
    print(f"  {'epoch (UTC)':>22} {'agreement':>11} {'azimuth':>9} {'elevation':>10}")
    kept, best = [], None
    for i in order:
        if any(abs(int(i) - j) < 48 for j in kept):
            continue
        kept.append(int(i))
        az, el = sun_angles(centre, suns[i])
        if best is None:
            best = (float(az[0]), float(el[0]), float(agreement[i]), sp.et2utc(ets[i], "C", 0))
        print(f"  {sp.et2utc(ets[i],'C',0):>22} {agreement[i]:10.4%} {float(az[0]):9.3f} "
              f"{float(el[0]):10.3f}")
        if len(kept) >= 10:
            break

    az, el, agr, when = best
    d_az = abs((az - TARGET_AZ + 180) % 360 - 180)
    d_el = abs(el - TARGET_EL)
    print(f"\nregistered P6 target: azimuth {TARGET_AZ} +/- {TARGET_AZ_TOL}, "
          f"elevation {TARGET_EL} +/- {TARGET_EL_TOL}")
    print(f"  best epoch {when}: azimuth {az:.3f} (off by {d_az:.3f}), "
          f"elevation {el:.3f} (off by {d_el:.3f})")
    ok = d_az <= TARGET_AZ_TOL and d_el <= TARGET_EL_TOL
    print(f"  -> {'REPRODUCED' if ok else 'DOES NOT REPRODUCE'}")
    sp.kclear()
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
