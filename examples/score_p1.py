"""Score P1: the distribution of shadow-edge speeds over the planning region.

Reads the edge arrays written by cpp/src/edges.cpp and applies the registered
scoring rule: headline is the median over edges whose caster relief exceeds
20 m, with every pinned transition scored as speed zero. Before scoring, the
C++ edges are reproduced independently in Python at sampled cells, using
src.edges.level_set_speed, which tests/test_edge_speed_spike.py validates
against a caster whose sweep speed is analytic.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import rasterio
from rasterio.windows import Window, from_bounds

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.edges import (caster_relief, flag_degenerate,           # noqa: E402
                       kinematic_speed_bound, level_set_speed)
from src.frames import azimuth_elevation_from_frame, horizon_lookup   # noqa: E402
from src.illumination import RegionCells                      # noqa: E402

DATA = ROOT / "data"
WORK = DATA / "work"
RELIEF_MIN = 20.0        # registered in PLAN.md, P1
PIN_DEG = 1.0            # registered in PLAN.md, P1
P1_PASS, P1_FAIL = 10.0, 3.0
RATES = (20.0, 50.0, 100.0, 200.0)      # the rover progress rates PLAN.md sweeps
SEED = 20260922
H_DER = 1.0e-3


def g_at(cells, prof, suns, t, dt):
    u = np.asarray(t, float) / dt
    k = np.clip(np.floor(u).astype(int), 0, suns.shape[0] - 2)
    f = (u - k)[..., None]
    s = (1.0 - f) * suns[k] + f * suns[k + 1]
    az, el = azimuth_elevation_from_frame(s, cells.origin, cells.up,
                                          cells.east, cells.north)
    return el - horizon_lookup(prof, az), az


def verify(p, params, suns, dem, horizon, hdist, cell_ids, speeds, reliefs, flags,
           n_check):
    """Reproduce the C++ edges in Python at sampled cells, end to end."""
    n, px, x0, y1 = p["region_nx"], p["region_px"], p["region_ox"], p["region_oy"]
    dt, ne = params["sun_dt_hours"], params["n_sun"]
    rng = np.random.default_rng(SEED)
    interior = np.flatnonzero(np.isin(cell_ids, cell_ids))     # all recorded are interior
    pick = rng.choice(np.unique(cell_ids), n_check, replace=False)
    order = np.argsort(cell_ids, kind="stable")
    ids_sorted = cell_ids[order]
    worst_speed, worst_count, worst_relief, n_edges = 0.0, 0, 0.0, 0
    for m in pick:
        lo, hi = np.searchsorted(ids_sorted, m, "left"), np.searchsorted(ids_sorted, m, "right")
        sel = order[lo:hi]
        cpp_speed = np.sort(speeds[sel])
        r, c = int(m) // n, int(m) % n
        nb = [m - n, m + n, m - 1, m + 1]
        idx = np.array([m] + nb, dtype=np.int64)
        rows, cols = idx // n, idx % n
        cells = RegionCells(x0 + (cols + 0.5) * px, y1 - (rows + 0.5) * px,
                            dem.ravel()[idx].astype(float),
                            radius=p["radius_m"], observer_height=p["observer_height_m"])
        prof = horizon[idx].astype(np.float64)
        ts = np.arange(ne) * dt
        gall = np.empty((ne, 5))
        step = 4096
        for i in range(0, ne, step):
            gall[i:i + step], _ = g_at(cells, prof[None, :, :], suns, ts[i:i + step, None], dt)
        g0 = gall[:, 0]
        trans = np.flatnonzero((g0[:-1] < 0.0) & (g0[1:] >= 0.0))
        py_speed = []
        for k in trans:
            a, b = k * dt, (k + 1) * dt
            for _ in range(40):
                mid = 0.5 * (a + b)
                if g_at(cells, prof, suns, np.full(5, mid), dt)[0][0] < 0.0:
                    a = mid
                else:
                    b = mid
            tstar = 0.5 * (a + b)
            gs, az = g_at(cells, prof, suns, np.full(5, tstar), dt)
            gp = g_at(cells, prof, suns, np.full(5, tstar + H_DER), dt)[0][0]
            gm = g_at(cells, prof, suns, np.full(5, tstar - H_DER), dt)[0][0]
            dgdt = (gp - gm) / (2.0 * H_DER)
            sp = float(level_set_speed(dgdt, gs[4], gs[3], gs[1], gs[2], px))
            hz = horizon_lookup(prof, np.full(5, az[0]))
            pinned = bool(np.any(np.abs(hz[1:] - hz[0]) > PIN_DEG))
            bin_ = int(round(az[0])) % p["n_azimuth"]
            rel = float(caster_relief(horizon[m, bin_], hdist[m, bin_],
                                      dem.ravel()[m], p["radius_m"]))
            py_speed.append((sp, pinned, rel))
        py_sorted = np.sort([s for s, _, _ in py_speed])
        worst_count = max(worst_count, abs(len(py_speed) - sel.size))
        if len(py_speed) == sel.size and sel.size:
            worst_speed = max(worst_speed, float(np.max(
                np.abs(py_sorted - cpp_speed) / np.maximum(np.abs(cpp_speed), 1e-9))))
            rr = np.sort([r_ for _, _, r_ in py_speed])
            worst_relief = max(worst_relief, float(np.max(
                np.abs(rr - np.sort(reliefs[sel])) / np.maximum(np.abs(np.sort(reliefs[sel])), 1e-9))))
            n_edges += sel.size
    return worst_count, worst_speed, worst_relief, n_edges, len(pick)


def describe(name, v):
    q = np.percentile(v, np.arange(10, 100, 10))
    print(f"  {name:<28} n {v.size:>8}  median {np.median(v):8.2f}  "
          f"mean {v.mean():8.2f}  max {v.max():10.2f}")
    print(f"    deciles " + " ".join(f"{x:7.2f}" for x in q))
    print(f"    above " + "  ".join(f"{r:.0f} m/hr {(v > r).mean():6.2%}" for r in RATES))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checks", type=int, default=12)
    args = ap.parse_args()
    p = json.loads((WORK / "params.json").read_text())
    params = json.loads((WORK / "edge_params.json").read_text())
    side = p["region_nx"] * p["region_px"]
    n = p["region_nx"]

    speeds = np.fromfile(WORK / "edge_speed.f32", dtype=np.float32).astype(np.float64)
    reliefs = np.fromfile(WORK / "edge_relief.f32", dtype=np.float32).astype(np.float64)
    flags = np.fromfile(WORK / "edge_flags.u8", dtype=np.uint8)
    cell_ids = np.fromfile(WORK / "edge_cell.u32", dtype=np.uint32).astype(np.int64)
    counts = np.fromfile(WORK / "edge_counts.u32", dtype=np.uint32)
    side_car = json.loads((WORK / "edges.json").read_text())
    print(f"{side_car['n_cells']:.0f} cells, {side_car['n_epochs']:.0f} epochs at "
          f"{side_car['cadence_hours']*60:.0f} min, {side_car['transitions_total']:.0f} "
          f"transitions, {speeds.size} edges on interior cells, "
          f"{side_car['seconds']:.1f} s")

    with rasterio.open(DATA / "lola" / "LDEM_80S_20MPP_ADJ.TIF") as ds:
        w = from_bounds(p["region_ox"], p["region_oy"] - side,
                        p["region_ox"] + side, p["region_oy"], ds.transform)
        dem = ds.read(1, window=Window(round(w.col_off), round(w.row_off), n, n))
    horizon = np.fromfile(p["out_horizon"], dtype=np.float32).reshape(n * n, p["n_azimuth"])
    hdist = np.fromfile(p["out_horizon_dist"], dtype=np.float32).reshape(n * n, p["n_azimuth"])
    suns = np.fromfile(WORK / "sun.f64", dtype=np.float64).reshape(-1, 3)

    dc, ds_, dr, ne, nc = verify(p, params, suns, dem, horizon, hdist, cell_ids,
                                 speeds, reliefs, flags, args.checks)
    print(f"\nC++ against Python on {nc} sampled cells, {ne} edges reproduced end to end:")
    print(f"  transition count mismatch {dc}; worst relative speed difference {ds_:.3e}; "
          f"worst relative relief difference {dr:.3e}")

    zero = int((counts == 0).sum())
    print(f"\ncells with no first-light transition in the year: {zero} of {counts.size} "
          f"({zero/counts.size:.4%}); transitions per cell mean {counts.mean():.2f}, "
          f"median {np.median(counts):.0f}, max {counts.max()}")

    # The degeneracy screen, computed from this run's own inputs rather than
    # quoted: the reach the horizon was traced to, and the fastest the sun's
    # azimuth actually moved at the region centre over the year.
    centre = RegionCells(np.array([p["region_ox"] + side / 2]),
                         np.array([p["region_oy"] - side / 2]), np.array([0.0]),
                         radius=p["radius_m"])
    caz, _ = azimuth_elevation_from_frame(suns, centre.origin, centre.up,
                                          centre.east, centre.north)
    rate = np.abs((np.diff(caz) + 180.0) % 360.0 - 180.0) / params["sun_dt_hours"]
    bound = kinematic_speed_bound(p["reach_m"], rate.max())
    print(f"\ndegeneracy screen: reach {p['reach_m']/1000:.0f} km, sun azimuth rate at "
          f"the region centre over the year median {np.median(rate):.4f}, max "
          f"{rate.max():.4f} deg/hr")
    print(f"  kinematic bound {bound:.1f} m/hr (at phase 0's median rate of 0.5080 "
          f"it is {kinematic_speed_bound(p['reach_m'], 0.5080):.1f})")

    deg = flag_degenerate(speeds, bound)
    print(f"  degenerate edges: {int(deg.sum())} of {speeds.size} ({deg.mean():.4%}); "
          f"of those, {int((~np.isfinite(speeds)).sum())} have an exactly zero gradient")
    if deg.any():
        ex = speeds[deg & np.isfinite(speeds)] / bound
        print(f"  they exceed the bound by median {np.median(ex):.1f}x, "
              f"p90 {np.percentile(ex, 90):.1f}x, max {ex.max():.0f}x, so the screen's "
              f"exact value is not delicate")
    sp, rl, pin = speeds, reliefs, flags.astype(bool)

    keep = rl > RELIEF_MIN
    print(f"\ncaster relief: {keep.mean():.4%} of edges above {RELIEF_MIN:.0f} m "
          f"(median relief {np.median(rl):.1f} m); pinned {pin.mean():.4%} overall, "
          f"{pin[keep].mean():.4%} within the relief filter")

    headline = np.where(pin[keep], 0.0, sp[keep])
    clean = keep & ~deg
    print("\nP1, registered: median over edges with caster relief above 20 m, "
          "pinned scored as zero")
    describe("headline (pinned = 0)", headline)
    describe("resolved edges only", sp[keep & ~pin])
    describe("all edges, no relief filter", np.where(pin, 0.0, sp))
    describe("headline, degenerate dropped", np.where(pin[clean], 0.0, sp[clean]))
    med = float(np.median(headline))
    med_clean = float(np.median(np.where(pin[clean], 0.0, sp[clean])))
    print(f"\n  headline median {med:.2f} m/hr with the degenerate edges in, "
          f"{med_clean:.2f} with them out, a difference of "
          f"{abs(med-med_clean)/med:.2%}; the registered statistic is the median and "
          f"it is reported with them in")
    verdict = ("PASS" if med >= P1_PASS else
               ("FAIL, claim reported as failed" if med < P1_FAIL else
                "between: above the failure floor, below the pass bar"))
    print(f"\n  headline median {med:.2f} m/hr against P1's >= {P1_PASS:.0f} "
          f"and failure below {P1_FAIL:.0f}  -> {verdict}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
