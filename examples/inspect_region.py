"""Phase 1 step 1: measure the candidate planning region before any illumination is computed.

The candidate is the 10 km square centred on the pole, x and y in [-5000, 5000] m,
500 x 500 px on the 20 m grid. Reads are windowed; no raster is opened whole.
"""
from pathlib import Path

import numpy as np
import rasterio
from rasterio.windows import Window, from_bounds

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"

X0, X1 = -5000.0, 5000.0
Y0, Y1 = -5000.0, 5000.0
NAC_ID = "M139729686"


def valid_mask(a, nodata):
    """Valid = finite, not the declared nodata, and not a sentinel near the float32 floor.

    The NAC products declare nodata -3.4e38 but store NaN, so both are checked.
    """
    m = np.isfinite(a)
    if nodata is not None and np.isfinite(nodata):
        m &= a != np.float32(nodata)
    m &= a > -1e38
    return m


def square_window(ds, halo=0):
    """Window of the candidate square in this dataset's own grid, expanded by a halo."""
    w = from_bounds(X0, Y0, X1, Y1, ds.transform)
    return Window(
        round(w.col_off) - halo,
        round(w.row_off) - halo,
        round(w.width) + 2 * halo,
        round(w.height) + 2 * halo,
    )


def read_square(ds, halo=0):
    """Boundless windowed read: pixels outside the footprint come back NaN, not dropped."""
    win = square_window(ds, halo)
    a = ds.read(1, window=win, boundless=True, fill_value=float("nan")).astype(np.float64)
    return a, win


def header():
    print("== Candidate region ==")
    print(f"  x [{X0:.0f}, {X1:.0f}] m, y [{Y0:.0f}, {Y1:.0f}] m, chosen before any illumination")
    with rasterio.open(DATA / "lola" / "LDEM_80S_20MPP_ADJ.TIF") as ds:
        win = square_window(ds)
        print(f"  on the 20 m LDEM grid: window {win}, {win.width} x {win.height} px")
        print(f"  res {ds.res}, crs {ds.crs.to_string()}")
        b = rasterio.windows.bounds(win, ds.transform)
        print(f"  window bounds {tuple(round(v, 3) for v in b)} "
              f"(exact match to the request: {np.allclose(b, (X0, Y0, X1, Y1))})")


def check_elevation():
    print("\n== 1. Elevation, 20 m LDEM ==")
    with rasterio.open(DATA / "lola" / "LDEM_80S_20MPP_ADJ.TIF") as ds:
        a, _ = read_square(ds)
        nodata = ds.nodata
    m = valid_mask(a, nodata)
    z = a[m]
    print(f"  cells {a.size}, valid {z.size}, NaN/nodata {a.size - z.size}")
    print(f"  min {z.min():.2f} m   p5 {np.percentile(z, 5):.2f} m   "
          f"p95 {np.percentile(z, 95):.2f} m   max {z.max():.2f} m")
    print(f"  relief (max - min) {z.max() - z.min():.2f} m, p95 - p5 {np.percentile(z, 95) - np.percentile(z, 5):.2f} m")
    return a, m


def check_psr():
    print("\n== 2. PSR map ==")
    with rasterio.open(DATA / "lola" / "LPSR_80S_20MPP_ADJ.TIF") as ds:
        a, _ = read_square(ds)
        nodata, dt = ds.nodata, ds.dtypes[0]
    print(f"  dtype {dt}, declared nodata {nodata}")
    vals, counts = np.unique(a[np.isfinite(a)], return_counts=True)
    print(f"  distinct finite values present: {vals.size}")
    if vals.size <= 16:
        for v, c in zip(vals, counts):
            print(f"    value {v!r:>12}: {c:8d} cells  ({c / a.size:.4%} of the square)")
    else:
        print(f"    range [{vals.min()}, {vals.max()}], showing deciles:")
        for q in range(0, 101, 10):
            print(f"      p{q:<3d} {np.percentile(a[np.isfinite(a)], q)}")
    nonfinite = int((~np.isfinite(a)).sum())
    print(f"  non-finite cells: {nonfinite}")
    m = valid_mask(a, nodata)
    if vals.size <= 16:
        nz = int((a[m] != 0).sum())
        print(f"  fraction non-zero over valid cells: {nz}/{int(m.sum())} = {nz / max(int(m.sum()), 1):.4%}")
    return a


def check_nac():
    print(f"\n== 3. NAC {NAC_ID} LE and RE, each on its own grid, no resampling ==")
    for side in ("LE", "RE"):
        p = DATA / "nac" / f"{NAC_ID}{side}_20MPP.TIF"
        with rasterio.open(p) as ds:
            a, win = read_square(ds)
            nodata, dt = ds.nodata, ds.dtypes[0]
            res, bounds = ds.res, ds.bounds
        m = valid_mask(a, nodata)
        nan_ct = int(np.isnan(a).sum())
        print(f"  {p.name}: dtype {dt}, res {res}, declared nodata {nodata}")
        print(f"    image bounds {tuple(round(v) for v in bounds)}")
        print(f"    square window in this grid: {win} ({win.width} x {win.height} px)")
        print(f"    valid (finite, not nodata): {int(m.sum())}/{a.size} = {m.sum() / a.size:.4%}")
        print(f"    NaN cells {nan_ct} ({nan_ct / a.size:.4%}), "
              f"cells equal to declared nodata {int((a == np.float32(nodata)).sum()) if nodata is not None else 'n/a'}")
        if m.any():
            print(f"    valid values: min {a[m].min():.5f} max {a[m].max():.5f} median {np.median(a[m]):.5f}")


def slope_deg(h, px):
    """Central-difference slope in degrees. Rows are y, columns are x."""
    dzdy, dzdx = np.gradient(h, px, px)
    return np.degrees(np.arctan(np.hypot(dzdx, dzdy)))


def check_slope():
    print("\n== 4. Slope, 20 m LDEM, central differences ==")
    # Central differences need a one-pixel halo, so the slope pass reads with one.
    with rasterio.open(DATA / "lola" / "LDEM_80S_20MPP_ADJ.TIF") as ds:
        h, _ = read_square(ds, halo=1)
        nodata = ds.nodata
        px = abs(ds.transform.a)
    h = np.where(valid_mask(h, nodata), h, np.nan)
    slope = slope_deg(h, px)[1:-1, 1:-1]
    s = slope[np.isfinite(slope)]
    print(f"  pixel size {px:.1f} m, cells {slope.size}, valid {s.size} "
          f"(a cell is invalid if it or a 4-neighbour is nodata)")
    print(f"  median {np.median(s):.3f} deg   p95 {np.percentile(s, 95):.3f} deg   "
          f"max {s.max():.3f} deg")
    for lim in (15.0, 20.0, 25.0):
        n = int((s > lim).sum())
        print(f"  above {lim:.0f} deg: {n:7d}/{s.size} = {n / s.size:.4%}")


def check_slope_is_terrain_not_noise():
    """Protects against reading DEM pixel noise as terrain slope.

    A median slope above 22 deg over a 10 km square is steep enough to decide
    the cost model on its own, so it is not taken on trust. Noise at the pixel
    scale inflates 20 m differences and washes out when the same surface is
    differenced at 80 m; real relief does not. Two independent coarsenings are
    used, one a different product (the 80 m DEM) and one a reduction of this
    one, so agreement is not an artefact of either path.
    """
    print("\n== 4b. Is that slope terrain or DEM noise? ==")
    with rasterio.open(DATA / "lola" / "LDEM_80S_20MPP_ADJ.TIF") as ds:
        h20, _ = read_square(ds, halo=1)
        h20 = np.where(valid_mask(h20, ds.nodata), h20, np.nan)
        px20 = abs(ds.transform.a)
    with rasterio.open(DATA / "lola" / "LDEM_80S_80MPP_ADJ.TIF") as ds:
        h80, _ = read_square(ds, halo=1)
        h80 = np.where(valid_mask(h80, ds.nodata), h80, np.nan)
        px80 = abs(ds.transform.a)

    inner = h20[1:-1, 1:-1]
    n = inner.shape[0] // 4
    averaged = inner.reshape(n, 4, n, 4).mean(axis=(1, 3))   # 20 m DEM reduced to 80 m

    print("  slope at three scales over the same ground:")
    for label, h, px in (("20 m DEM, 20 m differences", h20, px20),
                         ("80 m DEM, 80 m differences", h80, px80),
                         ("20 m DEM averaged to 80 m", averaged, 80.0)):
        s = slope_deg(h, px)[1:-1, 1:-1]
        s = s[np.isfinite(s)]
        print(f"    {label:30s} median {np.median(s):6.2f}  p95 {np.percentile(s, 95):6.2f}  max {s.max():6.2f} deg")
    print("    noise inflates the 20 m figure alone; terrain holds across all three.")

    adjacent = np.abs(np.diff(inner, axis=1))
    lap = np.abs(inner[1:-1, 2:] + inner[1:-1, :-2]
                 + inner[2:, 1:-1] + inner[:-2, 1:-1] - 4 * inner[1:-1, 1:-1])
    print(f"  adjacent-cell |dz| along x: median {np.nanmedian(adjacent):.3f} m, "
          f"p95 {np.nanpercentile(adjacent, 95):.3f} m, max {np.nanmax(adjacent):.3f} m")
    print(f"  discrete Laplacian |.|:     median {np.nanmedian(lap):.3f} m, "
          f"p95 {np.nanpercentile(lap, 95):.3f} m")
    print("    a surface that is smooth but steep keeps the Laplacian well under")
    print("    the adjacent relief; pixel noise puts the two on the same order.")


if __name__ == "__main__":
    header()
    check_elevation()
    check_psr()
    check_nac()
    check_slope()
    check_slope_is_terrain_not_noise()
