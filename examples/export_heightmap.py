"""Export the planning region as a Gazebo-loadable heightmap.

513 x 513 at 20 m, float32 GeoTIFF, no resampling and no vertical exaggeration.
513 because Dem.hh states the Ogre constraint, square with a side that is a
power of two plus one, and says Data() pads if necessary: handing Gazebo a
500 x 500 raster would make it invent 13 rows and columns somewhere this
project has not verified.

The heights come from the same raster window the planner read, through the same
call, so the heightmap is the terrain the cost model planned on rather than a
second rendering of the same source. The export asserts that against the cached
region array as well.
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
from src.gzmap import HEIGHTMAP_N, PX, REGION_N, REGION_OX, REGION_OY, pos_z  # noqa: E402

DATA = ROOT / "data"
DEM = DATA / "lola" / "LDEM_80S_20MPP_ADJ.TIF"
OUT = DATA / "work" / "gazebo"


def read_window(path, ox, oy, n, px):
    """The n x n window whose north-west pixel edge is (ox, oy)."""
    with rasterio.open(path) as ds:
        w = from_bounds(ox, oy - n * px, ox + n * px, oy, ds.transform)
        col, row = round(w.col_off), round(w.row_off)
        if abs(w.col_off - col) > 1e-6 or abs(w.row_off - row) > 1e-6:
            raise SystemExit(f"window is not pixel aligned: {w.col_off}, {w.row_off}")
        return ds.read(1, window=Window(col, row, n, n)).astype(np.float32), ds.crs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(OUT / "region_heightmap.tif"))
    args = ap.parse_args()
    if not DEM.exists():
        raise SystemExit(f"missing {DEM}; run examples/fetch_data.sh")
    OUT.mkdir(parents=True, exist_ok=True)
    out = Path(args.out)

    h, crs = read_window(DEM, REGION_OX, REGION_OY, HEIGHTMAP_N, PX)
    n_nan = int(np.isnan(h).sum())
    if n_nan:
        raise SystemExit(f"{n_nan} nodata cells in the window; refusing to fill them "
                         f"silently, since a filled height is not terrain")
    h_min, h_max = float(h.min()), float(h.max())

    transform = rasterio.Affine(PX, 0.0, REGION_OX, 0.0, -PX, REGION_OY)
    with rasterio.open(out, "w", driver="GTiff", width=HEIGHTMAP_N,
                       height=HEIGHTMAP_N, count=1, dtype="float32", crs=crs,
                       transform=transform) as ds:
        ds.write(h, 1)

    # C2 and C1 layer 1, asserted here so a bad export cannot reach Gazebo.
    with rasterio.open(out) as ds:
        back = ds.read(1)
        assert ds.dtypes[0] == "float32", ds.dtypes
        assert ds.width == ds.height == HEIGHTMAP_N, (ds.width, ds.height)
    exact = bool(np.array_equal(back, h))
    region = np.fromfile(DATA / "work" / "region_height.f32",
                         dtype=np.float32).reshape(REGION_N, REGION_N)
    overlap = back[:REGION_N, :REGION_N]
    same_as_planner = bool(np.array_equal(overlap, region))
    diff = int(np.count_nonzero(overlap != region))

    sidecar = dict(path=str(out), n=HEIGHTMAP_N, px=PX, ox=REGION_OX, oy=REGION_OY,
                   region_n=REGION_N, h_min=h_min, h_max=h_max,
                   h_range=h_max - h_min, pos_z=pos_z(h_min, h_max),
                   dtype="float32", crs=str(crs),
                   roundtrip_exact=exact, matches_planner_region=same_as_planner,
                   differing_cells_vs_planner=diff)
    (out.with_suffix(".json")).write_text(json.dumps(sidecar, indent=2))

    print(f"wrote {out} ({out.stat().st_size/1e6:.2f} MB)")
    print(f"  {HEIGHTMAP_N} x {HEIGHTMAP_N} at {PX:.0f} m, float32, no exaggeration")
    print(f"  heights {h_min:.3f} to {h_max:.3f} m, range {h_max-h_min:.3f} m")
    print(f"  pos z offset {pos_z(h_min, h_max):.4f} m so world z equals the DEM height")
    print(f"  read back byte-identical: {exact}")
    print(f"  region overlap equals the planner's cached array: {same_as_planner} "
          f"({diff} differing cells of {REGION_N*REGION_N})")
    return 0 if (exact and same_as_planner) else 1


if __name__ == "__main__":
    sys.exit(main())
