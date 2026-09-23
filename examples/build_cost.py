"""Build the phase 2 cost model on the planning region and measure it.

Reruns from committed code every figure the design stage measured in scratch,
and runs the two registered checks that need the real DEM: check A, pitch
against the raster heights, and check C, the 1 m resampled path integration
that has no pass threshold and exists to quantify the grid's own error.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import rasterio
from rasterio.windows import Window, from_bounds
from scipy.ndimage import map_coordinates
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.cost import (Rover, build_edges, edge_time, joint_slope_bound,   # noqa: E402
                      legal, surface_gradient)
from src.frames import cell_position                                      # noqa: E402

DATA = ROOT / "data"
LIMIT_PAIRS = ((15.0, 10.0), (20.0, 15.0), (25.0, 20.0))    # registered sweep
RISE_TOL_M = 1e-8                                            # registered check A
SEED = 20260923
N_PATHS = 40
SUBSAMPLE_M = 1.0                                            # registered check C


def octile_path(r0, c0, r1, c1):
    """An 8-connected discrete line between two cells: the graph's best straight line."""
    cells = [(r0, c0)]
    r, c = r0, c0
    while (r, c) != (r1, c1):
        r += int(np.sign(r1 - r))
        c += int(np.sign(c1 - c))
        cells.append((r, c))
    return np.array(cells)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--paths", type=int, default=N_PATHS)
    args = ap.parse_args()
    p = json.loads((DATA / "work" / "params.json").read_text())
    n, px, x0, y1, R = (p["region_nx"], p["region_px"], p["region_ox"],
                        p["region_oy"], p["radius_m"])
    side = n * px
    with rasterio.open(DATA / "lola" / "LDEM_80S_20MPP_ADJ.TIF") as ds:
        w = from_bounds(x0, y1 - side, x0 + side, y1, ds.transform)
        win = Window(round(w.col_off), round(w.row_off), n, n)
        dem = ds.read(1, window=win).astype(np.float64)
        dem_pad = ds.read(1, window=Window(win.col_off - 2, win.row_off - 2,
                                           n + 4, n + 4)).astype(np.float64)
    print(f"region {n}x{n} at {px} m, origin ({x0}, {y1}), commit {p['source_commit']}")

    e = build_edges(dem, x0, y1, px, radius=R)
    xs = x0 + (np.arange(n) + 0.5) * px
    ys = y1 - (np.arange(n) + 0.5) * px
    gx, gy = np.meshgrid(xs, ys)
    pos = cell_position(gx, gy, dem, R).reshape(-1, 3)
    fx, fy = surface_gradient(dem, px, gx, gy, R)
    slope = np.degrees(np.arctan(np.hypot(fx, fy))).ravel()
    print(f"cell slope, 3x3 fit: median {np.median(slope):.2f}, p90 "
          f"{np.percentile(slope, 90):.2f}, p95 {np.percentile(slope, 95):.2f}, "
          f"max {slope.max():.2f} deg")
    print(f"directed edges {e['a'].size}; length min {e['length'].min():.3f} "
          f"max {e['length'].max():.3f} m (axis {px}, diagonal {px*np.sqrt(2):.3f})")
    for name, v in (("|pitch|", np.abs(e["pitch"])), ("|roll| ", np.abs(e["roll"]))):
        print(f"{name}: median {np.median(v):.2f}, p90 {np.percentile(v, 90):.2f}, "
              f"p99 {np.percentile(v, 99):.2f}, max {v.max():.2f} deg")

    dh = dem.ravel()[e["b"]] - dem.ravel()[e["a"]]
    res = np.abs(e["length"] * np.sin(np.radians(e["pitch"])) - dh)
    print(f"\ncheck A, pitch against the raster: worst {res.max():.3e} m of rise, "
          f"median {np.median(res):.3e}, tolerance {RISE_TOL_M:g} -> "
          f"{'PASS' if res.max() <= RISE_TOL_M else 'FAIL'}")

    print(f"\n{'pitch<=':>8}{'roll<=':>7}{'forbidden':>11}{'by pitch':>10}{'by roll':>9}"
          f"{'largest cc':>12}{'isolated':>10}{'bound':>8}{'slope>bound':>12}")
    for pm, rm in LIMIT_PAIRS:
        rover = Rover(pitch_max=pm, roll_max=rm)
        ok = legal(e["pitch"], e["roll"], rover)
        g = coo_matrix((np.ones(int(ok.sum())), (e["a"][ok], e["b"][ok])),
                       shape=(n * n, n * n))
        _, lab = connected_components(g, directed=False)
        sizes = np.bincount(lab)
        degree = np.bincount(e["a"][ok], minlength=n * n)
        disc, _ = joint_slope_bound(rover)
        print(f"{pm:8.0f}{rm:7.0f}{1-ok.mean():11.4%}"
              f"{(np.abs(e['pitch'])>pm).mean():10.4%}{(np.abs(e['roll'])>rm).mean():9.4%}"
              f"{sizes.max()/(n*n):12.4%}{(degree==0).sum():10d}{disc:8.2f}"
              f"{(slope>disc).mean():12.4%}")

    rover = Rover()
    t = edge_time(e["length"], e["pitch"], e["roll"], rover)
    ok = legal(e["pitch"], e["roll"], rover)
    print(f"\nedge time at the default limits and v_flat {rover.v_flat} m/hr, legal edges "
          f"only: median {np.median(t[ok])*3600:.1f} s, p99 {np.percentile(t[ok],99)*3600:.1f} s")
    up, down = e["pitch"] > 5.0, e["pitch"] < -5.0
    print(f"  uphill above 5 deg median {np.median(t[up&ok])*3600:.1f} s against downhill "
          f"{np.median(t[down&ok])*3600:.1f} s: the directed graph, in seconds")

    # Check C: the grid's own error, no pass threshold.
    # How often does a cell's own slope fail to explain its isolation? On a
    # plane, isolated implies slope above the discrete bound; on real terrain a
    # flat cell ringed by steep ones is isolated too, and this counts them.
    rover_d = Rover()
    okd = legal(e["pitch"], e["roll"], rover_d)
    degree = np.bincount(e["a"][okd], minlength=n * n)
    disc, _ = joint_slope_bound(rover_d)
    iso = degree == 0
    print(f"\nisolated cells at the default limits: {int(iso.sum())}, of which "
          f"{int((iso & (slope <= disc)).sum())} have their own slope below the "
          f"{disc:.2f} deg bound, i.e. are isolated by their neighbours rather than "
          f"by themselves")

    rng = np.random.default_rng(SEED)
    ratios_octile, ratios_octile_2d, ratios_dense, climb = [], [], [], []
    for _ in range(args.paths):
        r0, c0, r1, c1 = rng.integers(5, n - 5, 4)
        if abs(r1 - r0) + abs(c1 - c0) < 200:
            continue
        cells = octile_path(r0, c0, r1, c1)
        pa = pos[cells[:-1, 0] * n + cells[:-1, 1]]
        pb = pos[cells[1:, 0] * n + cells[1:, 1]]
        graph_len = float(np.linalg.norm(pb - pa, axis=1).sum())
        straight = float(np.linalg.norm(pos[r1 * n + c1] - pos[r0 * n + c0]))
        ratios_octile.append(graph_len / straight)
        # The 1/cos(22.5 deg) bound is a statement about the map plane; in 3D
        # relief adds length to the path and not to the straight line, so the
        # 3D ratio may exceed it and the 2D one may not.
        map_len = float(np.hypot(np.diff(cells[:, 1]) * px, np.diff(cells[:, 0]) * px).sum())
        map_straight = float(np.hypot((c1 - c0) * px, (r1 - r0) * px))
        ratios_octile_2d.append(map_len / map_straight)
        # the same polyline, resampled at 1 m from the DEM by a different
        # interpolator, with the window padded so the sampler is never clamped
        steps = max(int(round(px / SUBSAMPLE_M)), 1)
        f = np.linspace(0.0, 1.0, steps + 1)[:-1]
        rr = (cells[:-1, 0][:, None] + f * (cells[1:, 0] - cells[:-1, 0])[:, None]).ravel()
        cc = (cells[:-1, 1][:, None] + f * (cells[1:, 1] - cells[:-1, 1])[:, None]).ravel()
        rr = np.append(rr, cells[-1, 0])
        cc = np.append(cc, cells[-1, 1])
        h = map_coordinates(dem_pad, [rr + 2.0, cc + 2.0], order=1, mode="nearest",
                            output=np.float64)
        sx = x0 + (cc + 0.5) * px
        sy = y1 - (rr + 0.5) * px
        dense = cell_position(sx, sy, h, R)
        ratios_dense.append(float(np.linalg.norm(np.diff(dense, axis=0), axis=1).sum())
                            / graph_len)
        climb_graph = float(np.abs(np.diff(dem.ravel()[cells[:, 0] * n + cells[:, 1]])).sum())
        climb.append(float(np.abs(np.diff(h)).sum()) / climb_graph)
    ro, rd = np.array(ratios_octile), np.array(ratios_dense)
    r2, rc = np.array(ratios_octile_2d), np.array(climb)
    print(f"\ncheck C, {ro.size} paths, no pass threshold:")
    print(f"  octile bias, graph path over the straight line: median {np.median(ro):.4f}, "
          f"p90 {np.percentile(ro,90):.4f}, max {ro.max():.4f}, bound "
          f"{1/np.cos(np.radians(22.5)):.4f}")
    print(f"  the same in the map plane, where that bound applies: median "
          f"{np.median(r2):.4f}, max {r2.max():.4f}")
    print(f"  discretisation, 1 m resampled length over the 20 m graph length: median "
          f"{np.median(rd):.6f}, p90 {np.percentile(rd,90):.6f}, max {rd.max():.6f}")
    print(f"  total climb, 1 m profile over the 20 m edges: median {np.median(rc):.6f}, "
          f"max {rc.max():.6f}")
    print("  the 1 m profile is interpolated from the same 20 m DEM, so check C bounds "
          "the error of\n  treating each edge as a chord and says nothing about relief "
          "below 20 m, which is not in the data")
    return 0


if __name__ == "__main__":
    sys.exit(main())
