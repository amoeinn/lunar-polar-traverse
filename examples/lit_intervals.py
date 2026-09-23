"""Drive the lit-interval and shadow-edge run, and the registered g cross-check.

Builds the sun table for the year, the region height raster and the parameter
file, then either runs the cross-check of g between cpp/src/edges.cpp and
src/frames.py, or the full scan. PLAN.md registers the cross-check's tolerance
and its cell and time rules before either was run.
"""
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import rasterio
import spiceypy as sp
from rasterio.windows import Window, from_bounds

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.frames import horizon_lookup                      # noqa: E402
from src.frames import azimuth_elevation_from_frame        # noqa: E402
from src.illumination import RegionCells, sun_angles       # noqa: E402

DATA = ROOT / "data"
WORK = DATA / "work"
BIN = ROOT / "cpp" / "build" / "edges"
KERNELS = ["naif0012.tls", "pck00010.tpc", "moon_pa_de421_1900-2050.bpc",
           "moon_080317.tf", "de421.bsp"]
FRAME = "MOON_ME_DE421"
PIN_DEG = 1.0            # registered in PLAN.md, P1
TOL_DEG = 1e-6           # registered cross-check tolerance
SEED = 20260922
FRACTIONS = (0.13, 0.37, 0.61, 0.89)
N_ONGRID = 24


def build_inputs(year, cadence_h, rows):
    p = json.loads((WORK / "params.json").read_text())
    n, px, x0, y1 = p["region_nx"], p["region_px"], p["region_ox"], p["region_oy"]
    side = n * px
    with rasterio.open(DATA / "lola" / "LDEM_80S_20MPP_ADJ.TIF") as ds:
        w = from_bounds(x0, y1 - side, x0 + side, y1, ds.transform)
        dem = ds.read(1, window=Window(round(w.col_off), round(w.row_off), n, n))
    (WORK / "region_height.f32").write_bytes(dem.astype(np.float32).tobytes())

    for k in KERNELS:
        sp.furnsh(str(DATA / "kernels" / k))
    t0 = sp.str2et(f"{year}-01-01T00:00:00")
    t1 = sp.str2et(f"{year + 1}-01-01T00:00:00")
    ets = np.arange(t0, t1 + 1.0, cadence_h * 3600.0)
    suns = np.array([sp.spkpos("SUN", et, FRAME, "LT+S", "MOON")[0] for et in ets])
    sp.kclear()
    (WORK / "sun.f64").write_bytes(np.ascontiguousarray(suns, dtype=np.float64).tobytes())

    params = dict(
        radius_m=p["radius_m"], observer_height_m=p["observer_height_m"],
        region_nx=n, region_ny=rows or p["region_ny"],
        region_ox=x0, region_oy=y1, region_px=px, n_azimuth=p["n_azimuth"],
        pin_horizon_deg=PIN_DEG,
        horizon_path=p["out_horizon"], horizon_dist_path=p["out_horizon_dist"],
        height_path=str(WORK / "region_height.f32"),
        sun_path=str(WORK / "sun.f64"), n_sun=int(ets.size), sun_dt_hours=cadence_h,
        out_speed=str(WORK / "edge_speed.f32"), out_relief=str(WORK / "edge_relief.f32"),
        out_flags=str(WORK / "edge_flags.u8"), out_cell=str(WORK / "edge_cell.u32"),
        out_counts=str(WORK / "edge_counts.u32"),
        out_sidecar=str(WORK / "edges.json"), source_commit=p["source_commit"])
    return p, params, dem, suns, ets


def region_cells(p, dem, idx):
    n, px, x0, y1 = p["region_nx"], p["region_px"], p["region_ox"], p["region_oy"]
    rows, cols = idx // n, idx % n
    return RegionCells(x0 + (cols + 0.5) * px, y1 - (rows + 0.5) * px,
                       dem.ravel()[idx].astype(float),
                       radius=p["radius_m"], observer_height=p["observer_height_m"])


def sun_interp(suns, dt, t):
    """The same rule the C++ uses: linear in Cartesian Moon-fixed coordinates."""
    u = np.asarray(t, float) / dt
    k = np.clip(np.floor(u).astype(int), 0, suns.shape[0] - 2)
    f = (u - k)[:, None]
    return (1.0 - f) * suns[k] + f * suns[k + 1]


def probe_selection(horizon, n):
    """The registered rule, fixed before it ran."""
    mean_h = horizon.mean(axis=1)
    ptp = horizon.max(axis=1) - horizon.min(axis=1)
    corners = [0, n - 1, (n - 1) * n, n * n - 1]
    pick = list(corners)
    pick += list(np.argsort(mean_h)[::-1][:4])
    pick += list(np.argsort(mean_h)[:4])
    pick += list(np.argsort(ptp)[::-1][:4])
    pick += list(np.random.default_rng(SEED).choice(n * n, 4, replace=False))
    seen, out = set(), []
    for i in pick:
        if int(i) not in seen:
            seen.add(int(i))
            out.append(int(i))
    return np.array(out, dtype=np.int64)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--year", type=int, default=2028)
    ap.add_argument("--cadence", type=float, default=0.25)
    ap.add_argument("--rows", type=int, default=0, help="limit rows, for timing")
    ap.add_argument("--probe", action="store_true")
    args = ap.parse_args()

    p, params, dem, suns, ets = build_inputs(args.year, args.cadence, args.rows)
    n, naz = p["region_nx"], p["n_azimuth"]
    dt = args.cadence
    print(f"{args.year}: {ets.size} epochs at {dt*60:.0f} min, "
          f"{params['region_ny']}x{n} cells, commit {params['source_commit']}")

    if args.probe:
        horizon = np.fromfile(p["out_horizon"], dtype=np.float32).reshape(n * n, naz)
        idx = probe_selection(horizon, n)
        cells = region_cells(p, dem, idx)
        prof = horizon[idx].astype(np.float64)
        az, el = azimuth_elevation_from_frame(
            suns[None, :, :], cells.origin[:, None, :], cells.up[:, None, :],
            cells.east[:, None, :], cells.north[:, None, :])     # (cell, epoch)
        g = el - horizon_lookup(prof[:, None, :], az)
        print(f"probe cells {idx.size}; g over the year spans "
              f"{g.min():.3f} to {g.max():.3f} deg")

        rec_i, rec_t = [], []
        grid = np.linspace(0, ets.size - 1, N_ONGRID).astype(int)
        for j, m in enumerate(idx):
            for k in grid:
                rec_i.append(m); rec_t.append(k * dt)
            k_near = int(np.argmin(np.abs(g[j])))
            rec_i.append(m); rec_t.append(k_near * dt)
            trans = np.flatnonzero((g[j][:-1] < 0) & (g[j][1:] >= 0))
            k0 = int(trans[0]) if trans.size else max(k_near - 1, 0)
            for kk in (k0, k0 + 1):
                rec_i.append(m); rec_t.append(kk * dt)
            for f in FRACTIONS:
                rec_i.append(m); rec_t.append((k0 + f) * dt)
        rec_i = np.array(rec_i, dtype=np.int32)
        rec_t = np.array(rec_t, dtype=np.float64)
        (WORK / "probe_idx.i32").write_bytes(rec_i.tobytes())
        (WORK / "probe_t.f64").write_bytes(rec_t.tobytes())
        params.update(probe_path=str(WORK / "probe_idx.i32"),
                      probe_time_path=str(WORK / "probe_t.f64"),
                      probe_out_path=str(WORK / "probe_out.f64"), n_probe=int(rec_i.size))
        (WORK / "edge_params.json").write_text(json.dumps(params, indent=2))
        subprocess.run([str(BIN), str(WORK / "edge_params.json")], check=True)
        out = np.fromfile(WORK / "probe_out.f64", dtype=np.float64).reshape(-1, 3)

        cp = region_cells(p, dem, rec_i.astype(np.int64))
        sun_t = sun_interp(suns, dt, rec_t)
        paz, pel = sun_angles(cp, sun_t)
        pg = pel - horizon_lookup(horizon[rec_i.astype(np.int64)].astype(np.float64), paz)
        d_g = np.abs(out[:, 0] - pg)
        d_az = np.abs((out[:, 1] - paz + 180.0) % 360.0 - 180.0)
        d_el = np.abs(out[:, 2] - pel)
        on = np.isclose(rec_t / dt, np.round(rec_t / dt))
        print(f"\ncross-check of g, {rec_i.size} records, tolerance {TOL_DEG:g} deg")
        print(f"  {'quantity':<10} {'max diff':>12} {'median':>12}")
        for nm, d in (("g", d_g), ("azimuth", d_az), ("elevation", d_el)):
            print(f"  {nm:<10} {d.max():12.3e} {np.median(d):12.3e}")
        print(f"  on-grid records {int(on.sum())}, interpolated {int((~on).sum())}; "
              f"worst on-grid {d_g[on].max():.3e}, worst interpolated {d_g[~on].max():.3e}")
        worst = int(np.argmax(d_g))
        print(f"  worst record: cell {rec_i[worst]} at t {rec_t[worst]:.4f} h, "
              f"g C++ {out[worst,0]:.12f} vs Python {pg[worst]:.12f}")
        ok = max(d_g.max(), d_az.max(), d_el.max()) <= TOL_DEG
        print(f"  -> {'AGREE' if ok else 'DISAGREE'} at {TOL_DEG:g} deg")
        return 0 if ok else 1

    (WORK / "edge_params.json").write_text(json.dumps(params, indent=2))
    t0 = time.time()
    subprocess.run([str(BIN), str(WORK / "edge_params.json")], check=True)
    print(f"wall {time.time()-t0:.1f} s")
    print(json.dumps(json.loads((WORK / "edges.json").read_text()), indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
