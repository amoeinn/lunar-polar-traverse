"""Score P5 and P5b: never-lit cells over an 18.6-year cycle against the PSR map.

Never-lit is evaluated as an envelope, not a time loop, exactly as registered:
the sun's Moon-fixed track is reduced once to the extreme (most southerly)
latitude in each Moon-fixed longitude bin, and every cell is then tested
against that envelope in its own local frame. The reduction is validated
against the full hourly track on sampled cells, because an envelope that
missed a lighting epoch would silently inflate the never-lit set.

Everything about the sun convention comes from src.illumination, which test (c)
and the epoch-scan round trip guard; nothing here restates it.
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
from scipy.ndimage import distance_transform_edt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.frames import horizon_lookup                       # noqa: E402
from src.illumination import RegionCells, sun_angles        # noqa: E402
from src.sun import sun_positions_m  # noqa: E402

DATA = ROOT / "data"
KERNELS = ["naif0012.tls", "pck00010.tpc", "moon_pa_de421_1900-2050.bpc",
           "moon_080317.tf", "de421.bsp"]
FRAME = "MOON_ME_DE421"
CYCLE = ("2027-01-01T00:00:00", "2045-08-07T00:00:00")   # 18.6 years, as registered
SOLAR_RADIUS_DEG = 0.2666        # the value step 11 used; the cycle peak is 0.2718
EDGES = (("first light", +SOLAR_RADIUS_DEG), ("point source", 0.0),
         ("full sun", -SOLAR_RADIUS_DEG))
PSR_FRACTION = 0.022004          # measured in step 2; asserted, not assumed
SEED = 20260922


def sun_track(step_s):
    for k in KERNELS:
        sp.furnsh(str(DATA / "kernels" / k))
    ets = np.arange(sp.str2et(CYCLE[0]), sp.str2et(CYCLE[1]), step_s)
    pos = sun_positions_m(ets, FRAME)
    sp.kclear()
    return ets, pos


def envelope(pos, nbins):
    """One real sun position per longitude bin: the most southerly in that bin.

    The selected positions are actual track samples, not bin centres, so every
    direction tested is a direction the sun really had.
    """
    u = pos / np.linalg.norm(pos, axis=1, keepdims=True)
    lat = np.degrees(np.arcsin(u[:, 2]))
    lon = np.degrees(np.arctan2(u[:, 1], u[:, 0])) % 360.0
    b = np.minimum((lon * (nbins / 360.0)).astype(int), nbins - 1)
    order = np.lexsort((lat, b))          # within each bin, most southerly first
    first = np.ones(order.size, bool)
    first[1:] = b[order][1:] != b[order][:-1]
    sel = order[first]
    return pos[sel], lat[sel], lon[sel], sel


def max_margin(cells_xy, heights, profiles, suns, radius, obs_h, chunk):
    """Per cell, the maximum over the envelope of (sun elevation - horizon)."""
    n = cells_xy[0].size
    out = np.empty(n, dtype=np.float64)
    for i in range(0, n, chunk):
        j = min(i + chunk, n)
        c = RegionCells(cells_xy[0][i:j], cells_xy[1][i:j], heights[i:j],
                        radius=radius, observer_height=obs_h)
        az, el = sun_angles_broadcast(c, suns)
        out[i:j] = np.max(el - horizon_lookup(profiles[i:j, None, :], az), axis=1)
    return out


def sun_angles_broadcast(cells, suns):
    """sun_angles with one cell axis and one epoch axis, via the same function."""
    from src.frames import azimuth_elevation_from_frame
    return azimuth_elevation_from_frame(suns[None, :, :], cells.origin[:, None, :],
                                        cells.up[:, None, :], cells.east[:, None, :],
                                        cells.north[:, None, :])


def fixed_declination(suns, lat, nbins):
    """The published convention: one declination, the cycle minimum, at every longitude.

    Built from our own track's extreme latitude and our own mean sun distance, so
    the only difference from the envelope is that the declination does not vary
    with longitude. This is a sensitivity measurement, not a rescoring.
    """
    d = np.linalg.norm(suns, axis=1).mean()
    lam = np.radians((np.arange(nbins) + 0.5) * (360.0 / nbins))
    phi = np.radians(lat.min())
    return d * np.stack([np.cos(phi) * np.cos(lam), np.cos(phi) * np.sin(lam),
                         np.full(nbins, np.sin(phi))], axis=1)


def boundary_distance(mask, shape):
    """Distance in pixels to the nearest boundary of a boolean mask."""
    m = mask.reshape(shape)
    e = np.zeros_like(m, dtype=bool)
    e[:-1, :] |= m[:-1, :] != m[1:, :]
    e[1:, :] |= m[:-1, :] != m[1:, :]
    e[:, :-1] |= m[:, :-1] != m[:, 1:]
    e[:, 1:] |= m[:, :-1] != m[:, 1:]
    return distance_transform_edt(~e).ravel(), e.ravel()


def mcc(pred, truth):
    tp = float(np.sum(pred & truth)); tn = float(np.sum(~pred & ~truth))
    fp = float(np.sum(pred & ~truth)); fn = float(np.sum(~pred & truth))
    d = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    return (tp * tn - fp * fn) / d if d > 0 else 0.0, (tp, tn, fp, fn)


def fixtures():
    """Each metric answers a case whose value is known before any product is read."""
    t = np.zeros(100, bool); t[:40] = True
    p_ = np.zeros(100, bool); p_[:30] = True          # 30 TP, 0 FP, 10 FN, 60 TN
    m, (tp, tn, fp, fn) = mcc(p_, t)
    assert (tp, tn, fp, fn) == (30, 60, 0, 10)
    expect = (30 * 60 - 0 * 10) / np.sqrt(30 * 40 * 60 * 70)
    assert abs(m - expect) < 1e-12 and abs(m - 0.8017837257372732) < 1e-12, m
    assert abs(mcc(t, t)[0] - 1.0) < 1e-12
    assert mcc(np.zeros(100, bool), t)[0] == 0.0, "constant predictor must score MCC 0"
    # the envelope reducer must keep the most southerly sample in each bin
    pos = np.array([[1, 0, -0.1], [1, 0.01, -0.3], [0, 1, -0.2], [0.01, 1, -0.05]], float)
    _, lat, _, sel = envelope(pos, 4)
    assert set(sel.tolist()) == {1, 2}, sel
    assert lat.max() < 0, lat
    print("fixtures pass: MCC 0.8017837257372732 on a hand-counted table, "
          "0 for a constant predictor; envelope keeps the southerly sample per bin")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bins", type=int, default=1800)
    ap.add_argument("--step", type=float, default=3600.0)
    ap.add_argument("--chunk", type=int, default=500)
    ap.add_argument("--cells", type=int, default=0, help="time a subset first")
    ap.add_argument("--validate", type=int, default=200,
                    help="cells per class for the full-track check")
    args = ap.parse_args()

    fixtures()
    p = json.loads((DATA / "work" / "params.json").read_text())
    n, px, x0, y1 = p["region_nx"], p["region_px"], p["region_ox"], p["region_oy"]
    naz, side = p["n_azimuth"], p["region_nx"] * p["region_px"]
    horizon = np.fromfile(p["out_horizon"], dtype=np.float32).reshape(n * n, naz)
    with rasterio.open(DATA / "lola" / "LDEM_80S_20MPP_ADJ.TIF") as ds:
        w = from_bounds(x0, y1 - side, x0 + side, y1, ds.transform)
        dem = ds.read(1, window=Window(round(w.col_off), round(w.row_off), n, n))
    with rasterio.open(DATA / "lola" / "LPSR_80S_20MPP_ADJ.TIF") as ds:
        w = from_bounds(x0, y1 - side, x0 + side, y1, ds.transform)
        psr_raw = ds.read(1, window=Window(round(w.col_off), round(w.row_off), n, n))
    vals = np.unique(psr_raw)
    assert set(vals.tolist()) <= {0.0, 1.0}, f"LPSR window is not binary: {vals[:6]}"
    psr = (psr_raw == 1.0).ravel()
    assert abs(psr.mean() - PSR_FRACTION) < 1e-6, \
        f"PSR fraction {psr.mean():.6%} does not reproduce step 2's {PSR_FRACTION:.6%}"
    print(f"region {n}x{n} at {px} m, commit {p['source_commit']}; "
          f"LPSR binary, PSR {psr.mean():.4%} reproduces step 2")

    t0 = time.time()
    ets, pos = sun_track(args.step)
    suns, lat, lon, sel = envelope(pos, args.bins)
    print(f"track {ets.size} samples at {args.step/3600:.2f} h over {CYCLE[0][:10]} to "
          f"{CYCLE[1][:10]} in {time.time()-t0:.1f} s; envelope {suns.shape[0]} of "
          f"{args.bins} bins filled")
    print(f"  envelope sub-solar latitude: min {lat.min():.4f}, median "
          f"{np.median(lat):.4f}, max {lat.max():.4f} deg (south is negative)")

    xs = x0 + (np.arange(n) + 0.5) * px
    ys = y1 - (np.arange(n) + 0.5) * px
    gx, gy = np.meshgrid(xs, ys)
    gx, gy, hh = gx.ravel(), gy.ravel(), dem.ravel().astype(float)
    m = args.cells or gx.size
    t0 = time.time()
    margin = max_margin((gx[:m], gy[:m]), hh[:m], horizon[:m].astype(np.float64),
                        suns, p["radius_m"], p["observer_height_m"], args.chunk)
    dt = time.time() - t0
    print(f"envelope test over {m} cells in {dt:.1f} s "
          f"({m/dt:.0f} cells/s, {m*suns.shape[0]/dt:.2e} pairs/s)")
    if args.cells:
        print(f"  full region would take {gx.size/(m/dt)/60:.1f} min")
        return 0

    # Validation: does the envelope ever miss a lighting epoch the full track has?
    rng = np.random.default_rng(SEED)
    nl = margin < 0.0
    pick = np.concatenate([rng.choice(np.flatnonzero(nl), args.validate, replace=False),
                           rng.choice(np.flatnonzero(~nl), args.validate, replace=False)])
    c = RegionCells(gx[pick], gy[pick], hh[pick],
                    radius=p["radius_m"], observer_height=p["observer_height_m"])
    full = np.full(pick.size, -np.inf)
    step = 8192
    for i in range(0, pos.shape[0], step):
        az, el = sun_angles_broadcast(c, pos[i:i + step])
        full = np.maximum(full, np.max(el - horizon_lookup(
            horizon[pick][:, None, :].astype(np.float64), az), axis=1))
    gap = full - margin[pick]
    print(f"\nenvelope against the full {pos.shape[0]}-sample track on {pick.size} cells:")
    print(f"  full-track margin exceeds the envelope's by median {np.median(gap):.6f}, "
          f"p99 {np.percentile(gap, 99):.6f}, max {gap.max():.6f} deg")
    flip = (margin[pick] < 0.0) & (full >= 0.0)
    print(f"  cells the envelope calls never-lit that the full track lights: "
          f"{int(flip.sum())} of {int((margin[pick] < 0).sum())}")

    print(f"\nP5 and P5b, {m} cells, PSR base rate {psr.mean():.4%}")
    print(f"  {'edge':<13} {'never-lit':>10} {'agree':>9} {'recall':>8} {'prec':>8} {'MCC':>8}")
    base_agree = 1.0 - psr.mean()
    for name, edge in EDGES:
        pred = margin < -edge
        agree = float((pred == psr).mean())
        rec = float(pred[psr].mean())
        prec = float(psr[pred].mean()) if pred.any() else float("nan")
        mc, (tp, tn, fp, fn) = mcc(pred, psr)
        print(f"  {name:<13} {pred.mean():9.4%} {agree:8.4%} {rec:7.4%} {prec:7.4%} "
              f"{mc:8.4f}   tp {int(tp)} fp {int(fp)} fn {int(fn)}")
    print(f"  {'constant':<13} {0.0:9.4%} {base_agree:8.4%} {0.0:7.4%} {'n/a':>7} "
          f"{0.0:8.4f}")
    print("\nwhere the disagreement sits, relative to the PSR map's own boundaries")
    print("(distance in pixels of 20 m; 0 means the pixel touches a boundary):")
    dist, edge = boundary_distance(psr, (n, n))
    print(f"  the PSR map has {int(edge.sum())} boundary pixels, {edge.mean():.4%} "
          f"of the region")
    for name, e in EDGES:
        pred = margin < -e
        wrong = pred != psr
        dw, da = dist[wrong], dist
        row = "  ".join(f"<={r}px {(dw <= r).mean():6.2%}/{(da <= r).mean():6.2%}"
                        for r in (0, 2, 5, 10))
        print(f"  {name:<13} n {int(wrong.sum()):>6}  median {np.median(dw):5.2f} px "
              f"against {np.median(da):5.2f}   {row}")
    print("  (each pair is disagreements / all region pixels within that radius)")

    print("\nsensitivity: the published map's convention, one fixed declination at the")
    print("cycle minimum, against our per-longitude envelope. Not a rescoring.")
    fsuns = fixed_declination(suns, lat, args.bins)
    t1 = time.time()
    fmargin = max_margin((gx, gy), hh, horizon.astype(np.float64), fsuns,
                         p["radius_m"], p["observer_height_m"], args.chunk)
    print(f"  computed in {time.time()-t1:.1f} s at declination {lat.min():.4f} deg")
    print(f"  {'edge':<13} {'never-lit':>10} {'agree':>9} {'recall':>8} {'prec':>8} {'MCC':>8}")
    for name, e in EDGES:
        pred = fmargin < -e
        mc, _ = mcc(pred, psr)
        prec = float(psr[pred].mean()) if pred.any() else float("nan")
        print(f"  {name:<13} {pred.mean():9.4%} {float((pred == psr).mean()):8.4%} "
              f"{float(pred[psr].mean()):7.4%} {prec:7.4%} {mc:8.4f}")
    print(f"  the envelope's per-longitude extreme latitude spans {lat.min():.4f} to "
          f"{lat.max():.4f} deg, a range of {lat.max()-lat.min():.4f}, which is "
          f"{(lat.max()-lat.min())/SOLAR_RADIUS_DEG:.0%} of the solar radius")

    print(f"\n  P5 asks agreement >= 95%; P5b asks recall >= 70% and MCC >= 0.5")
    print(f"  the solar radius used is {SOLAR_RADIUS_DEG} deg; at the cycle peak "
          f"0.2718 the full-sun never-lit fraction becomes "
          f"{(margin < 0.2718).mean():.4%} against {(margin < SOLAR_RADIUS_DEG).mean():.4%}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
