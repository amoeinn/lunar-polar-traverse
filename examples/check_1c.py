"""Check 1c: the C++ horizon model against an independent Python ray march.

Runs the comparison at the cells selected by the rule registered in PLAN.md,
prints a per-cell table rather than an aggregate, and separately measures the
far-field azimuth undersampling named in Limitations.

Needs data/work/, which is produced by examples/extract_windows.py followed by
cpp/build/horizon.
"""
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tests.independent_horizon import Reference, select_cells  # noqa: E402

WORK = ROOT / "data" / "work"
SUB_RAYS_PER_DEGREE = 10


def main():
    ref = Reference(WORK)
    p = ref.params
    n, naz = p["region_nx"], p["n_azimuth"]
    H = np.fromfile(p["out_horizon"], dtype=np.float32).reshape(n, n, naz)
    D = np.fromfile(p["out_horizon_dist"], dtype=np.float32).reshape(n, n, naz)

    cells, why, near_frac, far_frac, peak = select_cells(H, D, p["switch_m"])
    print(f"check 1c against {p['out_horizon']}, {len(cells)} cells\n")

    print("=== per-cell agreement, C++ against the independent Python march ===")
    print(f"  {'cell':>12} {'selected as':>22} {'max|diff| deg':>14} {'ray':>4} "
          f"{'dist py':>10} {'dist c++':>10} {'dist same':>10}")
    worst = 0.0
    t0 = time.time()
    for (r, c) in cells:
        h, d = ref.march(r, c, np.arange(float(naz)))
        diff = np.abs(h - H[r, c].astype(np.float64))
        k = int(np.argmax(diff))
        worst = max(worst, float(diff.max()))
        print(f"  {f'({r},{c})':>12} {why[(r, c)]:>22} {diff.max():14.3e} {k:4d} "
              f"{d[k]:10.0f} {D[r, c, k]:10.0f} {int((d == D[r, c]).sum()):7d}/{naz}")
    print(f"\n  worst over all cells: {worst:.3e} deg")
    print(f"  criterion: 1e-3 deg -> {'PASS' if worst < 1e-3 else 'FAIL'}")
    print(f"  float32 ulp at the largest horizon here ({H.max():.1f} deg): "
          f"{np.spacing(np.float32(H.max())):.2e} deg")
    print(f"  ({time.time() - t0:.1f} s)")

    print(f"\n=== far-field azimuth undersampling, {SUB_RAYS_PER_DEGREE} sub-rays per degree ===")
    print("  how far the true between-ray maximum rises above the linear")
    print("  interpolation of the two adjacent integer rays (no pass threshold)")
    print(f"  {'cell':>12} {'max excess deg':>15} {'at azimuth':>11} {'true':>9} "
          f"{'interpolated':>13} {'occluder m':>11}")
    sub = np.arange(0.0, 360.0, 1.0 / SUB_RAYS_PER_DEGREE)
    frac = sub - np.floor(sub)
    i0 = np.floor(sub).astype(int) % naz
    i1 = (i0 + 1) % naz
    excesses = []
    t0 = time.time()
    for (r, c) in cells:
        hs, ds = ref.march(r, c, sub)
        prof = H[r, c].astype(np.float64)
        lin = (1.0 - frac) * prof[i0] + frac * prof[i1]
        exc = hs - lin
        j = int(np.argmax(exc))
        excesses.append(float(exc[j]))
        print(f"  {f'({r},{c})':>12} {exc[j]:15.4f} {sub[j]:11.1f} {hs[j]:9.4f} "
              f"{lin[j]:13.4f} {ds[j]:11.0f}")
    e = np.array(excesses)
    print(f"\n  across the {len(cells)} cells: min {e.min():.4f}, median {np.median(e):.4f}, "
          f"max {e.max():.4f} deg")
    print(f"  for scale, the solar angular radius is 0.2666 deg")
    print(f"  ({time.time() - t0:.1f} s)")


if __name__ == "__main__":
    main()
