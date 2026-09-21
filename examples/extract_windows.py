"""Extract the DEM windows the horizon model needs for the selected region.

Reads the near-field 20 m window and the far-field 80 m window with windowed
reads (the 20 m DEM is 3.7 GB and is never opened whole), writes them as raw
float32 alongside a parameter JSON for cpp/build/horizon, into data/work/.

The region, reach and switch are the ones registered in PLAN.md. Run with
--subset N to extract for an N x N region centred on the same point, which is
what the timing run uses.
"""
import argparse
import json
import subprocess
from pathlib import Path

import numpy as np
import rasterio
from rasterio.windows import Window

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"

RADIUS_M = 1737400.0
REGION_CX, REGION_CY = 1000.0, 6000.0      # selected 2026-09-21, see PLAN.md
REGION_N = 500
REGION_PX = 20.0
REACH_M = 238000.0                          # arc distance, proved in PLAN.md
SWITCH_M = 20000.0
NEAR_STEP_M, FAR_STEP_M = 20.0, 80.0
N_AZIMUTH = 360
OBSERVER_HEIGHT_M = 0.0
EDGE_PIXELS = 2                             # bilinear needs a neighbour beyond the last sample


def arc_to_map(arc_m, radius=RADIUS_M):
    """Map radius corresponding to an arc distance. On this projection
    r = 2R tan(theta/2), so 238 km of arc is 238.4 km of map."""
    return 2.0 * radius * np.tan(0.5 * arc_m / radius)


def distance_to_rect(X, Y, x0, x1, y0, y1):
    dx = np.maximum(np.maximum(x0 - X, X - x1), 0.0)
    dy = np.maximum(np.maximum(y0 - Y, Y - y1), 0.0)
    return np.hypot(dx, dy)


def extract(path, x0, x1, y0, y1, label):
    """Smallest windowed read whose cells cover the requested map bounds."""
    with rasterio.open(path) as ds:
        t = ds.transform
        px = abs(t.a)
        col0 = int(np.floor((x0 - t.c) / px))
        col1 = int(np.ceil((x1 - t.c) / px))
        row0 = int(np.floor((t.f - y1) / px))
        row1 = int(np.ceil((t.f - y0) / px))
        assert col0 >= 0 and row0 >= 0 and col1 <= ds.width and row1 <= ds.height, (
            f"{label}: required window {col0}:{col1}, {row0}:{row1} falls outside "
            f"the {ds.width} x {ds.height} raster")
        win = Window(col0, row0, col1 - col0, row1 - row0)
        arr = ds.read(1, window=win).astype(np.float32)
        wt = ds.window_transform(win)
        nodata = ds.nodata
    if nodata is not None and np.isfinite(nodata):
        arr = np.where(arr == np.float32(nodata), np.nan, arr)
    return arr, float(wt.c), float(wt.f), px


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subset", type=int, default=0,
                    help="extract for an N x N region centred on the region centre")
    ap.add_argument("--out", default=str(DATA / "work"))
    args = ap.parse_args()

    n = args.subset if args.subset else REGION_N
    half = n * REGION_PX / 2.0
    rx0, rx1 = REGION_CX - half, REGION_CX + half
    ry0, ry1 = REGION_CY - half, REGION_CY + half
    region_ox, region_oy = rx0, ry1
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    near_map = arc_to_map(min(SWITCH_M, REACH_M))
    far_map = arc_to_map(REACH_M)
    print(f"region: {n} x {n} cells at {REGION_PX:.0f} m, "
          f"x [{rx0:.0f}, {rx1:.0f}], y [{ry0:.0f}, {ry1:.0f}]")
    print(f"near reach {min(SWITCH_M, REACH_M):.0f} m of arc = {near_map:.0f} m of map")
    print(f"far  reach {REACH_M:.0f} m of arc = {far_map:.0f} m of map "
          f"({far_map - REACH_M:+.0f} m, {(far_map / REACH_M - 1) * 100:.3f}%)")

    specs = [
        ("near", DATA / "lola" / "LDEM_80S_20MPP_ADJ.TIF", near_map, 20.0),
        ("far", DATA / "lola" / "LDEM_80S_80MPP_ADJ.TIF", far_map, 80.0),
    ]
    written = {}
    for label, src, reach_map, px in specs:
        pad = reach_map + EDGE_PIXELS * px
        arr, ox, oy, got_px = extract(src, rx0 - pad, rx1 + pad, ry0 - pad, ry1 + pad, label)
        assert abs(got_px - px) < 1e-9, f"{label}: expected {px} m pixels, got {got_px}"
        h, w = arr.shape
        x1e, y0e = ox + w * px, oy - h * px

        # The window must cover the region expanded by the reach, in every direction.
        for side, have, need in (("west", ox, rx0 - reach_map), ("east", x1e, rx1 + reach_map),
                                 ("south", y0e, ry0 - reach_map), ("north", oy, ry1 + reach_map)):
            ok = have <= need if side in ("west", "south") else have >= need
            assert ok, f"{label}: {side} edge at {have:.0f} does not cover {need:.0f}"

        # No NaN anywhere a ray can reach.
        xs = ox + (np.arange(w) + 0.5) * px
        ys = oy - (np.arange(h) + 0.5) * px
        X, Y = np.meshgrid(xs, ys)
        inside = distance_to_rect(X, Y, rx0, rx1, ry0, ry1) <= reach_map
        bad = int(np.sum(~np.isfinite(arr) & inside))
        assert bad == 0, f"{label}: {bad} non-finite cells inside the reach"

        path = out / f"{label}.f32"
        arr.tofile(path)
        written[label] = dict(path=str(path), ox=ox, oy=oy, width=w, height=h, px=px)
        print(f"{label}: {w} x {h} px, {arr.nbytes / 1e6:.1f} MB, "
              f"x [{ox:.0f}, {x1e:.0f}], y [{y0e:.0f}, {oy:.0f}], "
              f"cells inside the reach {int(inside.sum())}, non-finite inside 0")

    rev = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT,
                         capture_output=True, text=True).stdout.strip()
    dirty = subprocess.run(["git", "status", "--porcelain"], cwd=ROOT,
                           capture_output=True, text=True).stdout.strip()
    commit = rev + ("-dirty" if dirty else "")

    params = {
        "radius_m": RADIUS_M, "observer_height_m": OBSERVER_HEIGHT_M,
        "reach_m": REACH_M, "switch_m": SWITCH_M,
        "near_step_m": NEAR_STEP_M, "far_step_m": FAR_STEP_M, "n_azimuth": N_AZIMUTH,
        "region_ox": region_ox, "region_oy": region_oy,
        "region_nx": n, "region_ny": n, "region_px": REGION_PX,
        "near_path": written["near"]["path"], "near_ox": written["near"]["ox"],
        "near_oy": written["near"]["oy"], "near_width": written["near"]["width"],
        "near_height": written["near"]["height"], "near_px": written["near"]["px"],
        "far_path": written["far"]["path"], "far_ox": written["far"]["ox"],
        "far_oy": written["far"]["oy"], "far_width": written["far"]["width"],
        "far_height": written["far"]["height"], "far_px": written["far"]["px"],
        "out_horizon": str(out / "horizon.f32"),
        "out_horizon_dist": str(out / "horizon_dist.f32"),
        "out_sidecar": str(out / "horizon.json"),
        "source_commit": commit,
    }
    (out / "params.json").write_text(json.dumps(params, indent=2))
    steps = int(min(SWITCH_M, REACH_M) / NEAR_STEP_M) + int((REACH_M - SWITCH_M) / FAR_STEP_M)
    print(f"\nparams: {out / 'params.json'}   source_commit {commit}")
    print(f"steps per ray {steps}, samples {n * n * N_AZIMUTH * steps:.3e}")
    print(f"output arrays will be 2 x {n * n * N_AZIMUTH * 4 / 1e6:.1f} MB")
    if dirty:
        print("\nWARNING: working tree is dirty, so source_commit is marked -dirty")


if __name__ == "__main__":
    main()
