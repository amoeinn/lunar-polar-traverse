"""Phase 1 step 2: run the region selection rule registered in PLAN.md on 2026-09-21.

The rule, verbatim from PLAN.md:

    Candidates are 10 km squares on the 20 m grid, centres on a 1 km lattice,
    fully inside the bounding box of NAC M139729686. A cell is traversable if
    its central-difference slope is at most 15 deg; this threshold is for
    selection only, and phase 2 sweeps the limits. A square is eligible if its
    largest 8-connected traversable component covers at least 50% of its cells.
    Eligible squares are ranked by the fraction of the square with a valid
    pixel in either the LE or RE image of M139729686; ties within 1 percentage
    point go to the centre nearer the pole. PSR fraction and relief are
    reported for every candidate but never used in ranking. If no square is
    eligible, report the best ten and stop.

No illumination is computed here and PSR never enters the ranking. Slope is
computed once over the bounding box and windowed per candidate. NAC coverage
uses nearest-pixel mapping onto the DEM grid, which is exact enough for a
coverage count; the bilinear resampling P6 needs is a later step.
"""
import math
from pathlib import Path

import numpy as np
import rasterio
from rasterio.windows import Window, from_bounds
from scipy import ndimage

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"

NAC_ID = "M139729686"
SIDE_FOR_BBOX = "LE"          # PLAN.md's Region paragraph fixes the bbox as the LE bounds
SQUARE_M = 10000.0            # 10 km square
LATTICE_M = 1000.0            # centres on a 1 km lattice, aligned to the pole at (0, 0)
SLOPE_LIMIT_DEG = 15.0        # selection only; phase 2 sweeps the limits
COMPONENT_MIN = 0.50          # largest 8-connected traversable component
TIE_PP = 1.0                  # ties within 1 percentage point go to the centre nearer the pole

EIGHT = np.ones((3, 3), bool)


def assert_eight_connected():
    """Protects against silently labelling with 4-connectivity.

    scipy.ndimage.label defaults to 4-connectivity. The rule says 8-connected,
    and on diagonal terrain the two differ by a lot: under 4-connectivity a
    diagonal ribbon of traversable cells breaks into isolated pixels, so the
    largest component collapses and eligible squares are rejected. A fixture
    that cannot tell the two apart would pass either way, so the fixture is a
    purely diagonal chain, which is one component under 8 and N under 4.
    """
    diag = np.eye(4, dtype=bool)
    assert ndimage.label(diag)[1] == 4, "expected the 4-connected default to split a diagonal"
    assert ndimage.label(diag, structure=EIGHT)[1] == 1, "EIGHT is not 8-connectivity"


def valid_mask(a, nodata):
    m = np.isfinite(a)
    if nodata is not None and np.isfinite(nodata):
        m &= a != np.float32(nodata)
    return m & (a > -1e38)


def covering_window(ds, bounds, halo=0):
    """Smallest grid-aligned window covering `bounds`, expanded by a halo.

    The NAC bounding box does not land on the DEM grid: its origin is 0.491 px
    off in x and 0.803 px in y. That is expected (phase 0 measured the same
    sub-pixel offset) and it is not a problem here, because the rule puts the
    candidate squares on the 20 m grid and uses the box only as a containment
    test in projected coordinates. So the window is snapped outward to cover
    the box rather than being required to match it.
    """
    w = from_bounds(*bounds, ds.transform)
    c, r = math.floor(w.col_off), math.floor(w.row_off)
    wd = math.ceil(w.col_off + w.width) - c
    ht = math.ceil(w.row_off + w.height) - r
    return Window(c - halo, r - halo, wd + 2 * halo, ht + 2 * halo)


def window_origin(transform, win):
    """Projected coordinates of the window's top-left corner."""
    return transform.c + win.col_off * transform.a, transform.f + win.row_off * transform.e


def load_bbox():
    """Elevation, slope, PSR and NAC coverage over the bounding box, on the DEM grid."""
    with rasterio.open(DATA / "nac" / f"{NAC_ID}{SIDE_FOR_BBOX}_20MPP.TIF") as ds:
        bounds = tuple(ds.bounds)
    print(f"== Bounding box: NAC {NAC_ID} {SIDE_FOR_BBOX} ==")
    print(f"  x [{bounds[0]:.0f}, {bounds[2]:.0f}] m, y [{bounds[1]:.0f}, {bounds[3]:.0f}] m")

    dem_path = DATA / "lola" / "LDEM_80S_20MPP_ADJ.TIF"
    with rasterio.open(dem_path) as ds:
        win = covering_window(ds, bounds, halo=1)
        z = ds.read(1, window=win, boundless=True, fill_value=np.nan).astype(np.float64)
        z = np.where(valid_mask(z, ds.nodata), z, np.nan)
        px = abs(ds.transform.a)
        dem_transform = ds.transform
        inner = covering_window(ds, bounds)
        origin = window_origin(dem_transform, inner)
    w_raw = from_bounds(*bounds, dem_transform)
    print(f"  on the 20 m DEM grid: {inner.width} x {inner.height} px, "
          f"{inner.width * inner.height / 1e6:.2f} Mpx")
    print(f"  the box is off the DEM grid by ({w_raw.col_off % 1:.3f}, {w_raw.row_off % 1:.3f}) px; "
          f"window snapped outward to cover it")
    print(f"  window origin (top-left) {origin[0]:.0f}, {origin[1]:.0f} m")

    # Slope once over the whole box. The halo makes every in-box cell a true
    # central difference rather than a one-sided edge difference.
    dzdy, dzdx = np.gradient(z, px, px)
    slope = np.degrees(np.arctan(np.hypot(dzdx, dzdy)))[1:-1, 1:-1]
    z = z[1:-1, 1:-1]
    trav = np.isfinite(slope) & (slope <= SLOPE_LIMIT_DEG)
    print(f"  slope over the box: median {np.nanmedian(slope):.3f} deg, "
          f"at or below {SLOPE_LIMIT_DEG:.0f} deg: {trav.mean():.4%}")

    with rasterio.open(DATA / "lola" / "LPSR_80S_20MPP_ADJ.TIF") as ds:
        assert ds.transform == dem_transform, "PSR grid differs from the DEM grid"
        psr = ds.read(1, window=inner, boundless=True, fill_value=np.nan).astype(np.float64)
    print(f"  PSR over the box: {np.nanmean(psr > 0.5):.4%}")

    # Nearest-pixel mapping. Both grids are 20 m with the same orientation, so
    # the mapping is one constant integer pixel shift, not a per-cell lookup.
    cover = np.zeros(trav.shape, bool)
    for side in ("LE", "RE"):
        with rasterio.open(DATA / "nac" / f"{NAC_ID}{side}_20MPP.TIF") as ds:
            dx = (dem_transform.c - ds.transform.c) / ds.transform.a
            dy = (dem_transform.f - ds.transform.f) / ds.transform.e
            assert abs(ds.transform.a - dem_transform.a) < 1e-9, f"{side} is not 20 m"
            shift_c, shift_r = round(dx), round(dy)
            w = Window(inner.col_off + shift_c, inner.row_off + shift_r,
                       inner.width, inner.height)
            a = ds.read(1, window=w, boundless=True, fill_value=np.nan).astype(np.float64)
            v = valid_mask(a, ds.nodata)
        print(f"  NAC {side}: integer shift ({shift_c:+d}, {shift_r:+d}) px, sub-pixel residual "
              f"({dx - shift_c:+.3f}, {dy - shift_r:+.3f}) px, valid over the box {v.mean():.4%}")
        cover |= v
    print(f"  NAC LE or RE valid over the box: {cover.mean():.4%}")
    return bounds, origin, z, trav, psr, cover


def candidates(bounds):
    """Centres on a pole-aligned 1 km lattice whose 10 km square is fully inside the box."""
    x0, y0, x1, y1 = bounds
    h = SQUARE_M / 2
    cx = np.arange(np.ceil((x0 + h) / LATTICE_M), np.floor((x1 - h) / LATTICE_M) + 1) * LATTICE_M
    cy = np.arange(np.ceil((y0 + h) / LATTICE_M), np.floor((y1 - h) / LATTICE_M) + 1) * LATTICE_M
    print(f"\n== Candidates ==")
    print(f"  centres x {cx[0]:.0f} to {cx[-1]:.0f} ({cx.size}), "
          f"y {cy[0]:.0f} to {cy[-1]:.0f} ({cy.size}) -> {cx.size * cy.size} squares")
    return cx, cy


def evaluate(origin, z, trav, psr, cover, cx, cy):
    ox, oy = origin           # projected coordinates of array cell (0, 0)'s top-left corner
    n = int(SQUARE_M / 20)
    cells = n * n
    rows = []
    for yc in cy:
        r_f = (oy - (yc + SQUARE_M / 2)) / 20
        r0 = int(round(r_f))
        assert abs(r_f - r0) < 1e-9, "square is not on the 20 m grid in y"
        for xc in cx:
            c_f = ((xc - SQUARE_M / 2) - ox) / 20
            c0 = int(round(c_f))
            assert abs(c_f - c0) < 1e-9, "square is not on the 20 m grid in x"
            assert 0 <= r0 and r0 + n <= trav.shape[0] and 0 <= c0 and c0 + n <= trav.shape[1], \
                "candidate square falls outside the covering window"
            sl = (slice(r0, r0 + n), slice(c0, c0 + n))
            t = trav[sl]
            lab, k = ndimage.label(t, structure=EIGHT)
            if k:
                largest = np.bincount(lab.ravel())[1:].max()
            else:
                largest = 0
            zs = z[sl]
            rows.append((
                xc, yc,
                largest / cells,          # largest 8-connected traversable component
                t.mean(),                 # traversable fraction
                cover[sl].mean(),         # NAC coverage, the ranking criterion
                np.nanmean(psr[sl] > 0.5),
                np.nanmax(zs) - np.nanmin(zs),
            ))
    dt = np.dtype([("x", "f8"), ("y", "f8"), ("comp", "f8"), ("trav", "f8"),
                   ("nac", "f8"), ("psr", "f8"), ("relief", "f8")])
    return np.array(rows, dtype=dt)


def rank_by_rule(rows, k=10):
    """Rank eligible squares by the registered rule's tie-break.

    "Ties within 1 percentage point" is not a transitive relation, so it
    cannot order a list on its own and has to be read. It is applied here
    relative to the leader: every square within 1 pp of the best remaining NAC
    coverage is tied with it, and the tied square whose centre is nearest the
    pole takes the slot. The alternative, whole-percentage-point buckets, was
    rejected because it introduces boundaries the rule does not state. The two
    readings select different regions; see PLAN.md, "Applied 2026-09-21".
    """
    rem = rows.copy()
    d = np.hypot(rem["x"], rem["y"])
    out = []
    for _ in range(min(k, rem.size)):
        tied = rem["nac"] * 100 >= rem["nac"].max() * 100 - TIE_PP
        i = np.where(tied)[0][np.argmin(d[tied])]
        out.append(rem[i])
        rem, d = np.delete(rem, i), np.delete(d, i)
    return np.array(out, dtype=rows.dtype)


def assert_tie_break_is_relative_to_leader():
    """Protects against silently reverting to the whole-percentage-point reading.

    The two readings of "ties within 1 percentage point" pick different
    regions on the real data, so the difference is not academic. A fixture
    where they agree could not catch a reversion, so the fixture is first
    asserted to discriminate: two squares 0.8 pp apart that straddle a whole
    percentage point. Bucketing splits them and the higher coverage wins;
    reading relative to the leader calls them tied and the nearer pole wins.
    """
    dt = np.dtype([("x", "f8"), ("y", "f8"), ("comp", "f8"), ("trav", "f8"),
                   ("nac", "f8"), ("psr", "f8"), ("relief", "f8")])
    far = (30000.0, 0.0, 1.0, 1.0, 0.666, 0.0, 0.0)    # higher coverage, far from the pole
    near = (10000.0, 0.0, 1.0, 1.0, 0.658, 0.0, 0.0)   # 0.8 pp lower, nearer the pole
    fixture = np.array([far, near], dtype=dt)

    buckets = np.round(fixture["nac"] * 100 / TIE_PP)
    assert buckets[0] != buckets[1], "fixture does not straddle a whole percentage point"
    assert abs(fixture["nac"][0] - fixture["nac"][1]) * 100 < TIE_PP, "fixture is not a tie"

    won = rank_by_rule(fixture, 1)[0]
    assert won["x"] == near[0], "tie-break is bucketing, not reading relative to the leader"


def show(rows, title, note):
    print(f"\n{title}")
    print(f"  {note}")
    print(f"  {'centre x':>9} {'centre y':>9} {'largest':>9} {'travers':>9} "
          f"{'NAC':>8} {'PSR':>8} {'relief':>9}  {'pole dist':>9}")
    print(f"  {'(m)':>9} {'(m)':>9} {'comp %':>9} {'%':>9} {'%':>8} {'%':>8} {'(m)':>9}  {'(km)':>9}")
    for r in rows:
        d = np.hypot(r["x"], r["y"]) / 1000
        print(f"  {r['x']:9.0f} {r['y']:9.0f} {r['comp'] * 100:9.2f} {r['trav'] * 100:9.2f} "
              f"{r['nac'] * 100:8.2f} {r['psr'] * 100:8.2f} {r['relief']:9.1f}  {d:9.2f}")


def main():
    assert_eight_connected()
    assert_tie_break_is_relative_to_leader()
    bounds, origin, z, trav, psr, cover = load_bbox()
    cx, cy = candidates(bounds)
    rows = evaluate(origin, z, trav, psr, cover, cx, cy)

    print(f"\n== Rule outcome ==")
    print(f"  largest component: max {rows['comp'].max():.4%}, "
          f"median {np.median(rows['comp']):.4%} over {rows.size} candidates")
    eligible = rows[rows["comp"] >= COMPONENT_MIN]
    print(f"  eligible (largest component >= {COMPONENT_MIN:.0%}): {eligible.size} of {rows.size}")

    dist = np.hypot(rows["x"], rows["y"])
    if eligible.size:
        near = (eligible["nac"] * 100 >= eligible["nac"].max() * 100 - TIE_PP).sum()
        print(f"  within {TIE_PP:.0f} pp of the best NAC coverage "
              f"({eligible['nac'].max():.4%}): {near} squares")
        show(rank_by_rule(eligible, 10), "== Top 10 eligible squares ==",
             "ranked by NAC LE-or-RE coverage, ties within 1 pp to the centre nearer the pole")
    else:
        order = np.lexsort((dist, -rows["comp"]))
        show(rows[order][:10], "== NO SQUARE IS ELIGIBLE: the best ten candidates ==",
             "ranked by largest component, the criterion that failed; no square reaches "
             f"{COMPONENT_MIN:.0%}")
        print("\n  The rule says: if no square is eligible, report the best ten and stop.")


if __name__ == "__main__":
    main()
