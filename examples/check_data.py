"""Phase 0 data check: raster grids, grid alignment, and the sun geometry the claim rests on."""
from pathlib import Path

import numpy as np
import rasterio
import spiceypy as sp
from rasterio.windows import Window
from src.sun import sun_positions_m

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
KERNELS = [
    "naif0012.tls",
    "pck00010.tpc",
    "moon_pa_de421_1900-2050.bpc",
    "moon_080317.tf",
    "de421.bsp",
]
FRAME = "MOON_ME_DE421"


def describe(path):
    with rasterio.open(path) as ds:
        h, w = ds.height, ds.width
        n = min(1000, h, w)
        a = ds.read(1, window=Window((w - n) // 2, (h - n) // 2, n, n), masked=True)
        print(f"{path.name}: {w} x {h} px, res {ds.res}, {ds.dtypes[0]}, nodata {ds.nodata}")
        print(f"  bounds {tuple(round(b) for b in ds.bounds)}")
        print(f"  crs {ds.crs.to_string() if ds.crs else None}")
        print(f"  central {n} x {n}: min {a.min():.2f} max {a.max():.2f} masked {np.ma.count_masked(a)}")
        return ds.transform, (h, w), ds.crs


def offset_in_pixels(t, ref):
    return (t.c - ref.c) / ref.a, (t.f - ref.f) / ref.e


def check_rasters():
    print("== LOLA ==")
    t20, shape20, crs20 = describe(DATA / "lola" / "LDEM_80S_20MPP_ADJ.TIF")
    t80, _, crs80 = describe(DATA / "lola" / "LDEM_80S_80MPP_ADJ.TIF")
    tpsr, shapepsr, crspsr = describe(DATA / "lola" / "LPSR_80S_20MPP_ADJ.TIF")
    print(f"PSR grid identical to 20 m LDEM: transform {tpsr == t20}, shape {shapepsr == shape20}, crs {crspsr == crs20}")
    print(f"80 m LDEM crs matches 20 m: {crs80 == crs20}")

    print("== NAC ==")
    for p in sorted((DATA / "nac").glob("*.TIF")):
        t, _, crs = describe(p)
        dx, dy = offset_in_pixels(t, t20)
        aligned = abs(t.a - t20.a) < 1e-9 and abs(dx - round(dx)) < 1e-6 and abs(dy - round(dy)) < 1e-6
        print(f"  on 20 m LDEM grid: {aligned} (offset {dx:.3f}, {dy:.3f} px), crs match {crs == crs20}")


def check_sun():
    print("== Sun, DE421, frame", FRAME, "==")
    for k in KERNELS:
        sp.furnsh(str(DATA / "kernels" / k))
    et0 = sp.str2et("2027-01-01T00:00:00")
    et1 = sp.str2et("2029-01-01T00:00:00")
    ets = np.arange(et0, et1, 3600.0)
    lat = np.empty(ets.size)
    lon = np.empty(ets.size)
    for i, et in enumerate(ets):
        # Metres rather than kilometres. reclat's radius is discarded here and
        # its longitude and latitude are scale invariant, so this reads the same
        # either way; it goes through the one conversion so there is no second
        # place in the project where spkpos is called raw.
        pos = sun_positions_m(et, FRAME)[0]
        _, lon[i], lat[i] = sp.reclat(pos)
    elev_pole = -np.degrees(lat)  # at the south pole, local up is -z
    rate = np.abs(np.diff(np.degrees(np.unwrap(lon))))  # deg per hour
    print(f"hours sampled: {ets.size} (2027 to 2028)")
    print(f"sun elevation at pole: min {elev_pole.min():.3f} max {elev_pole.max():.3f} deg (expected within about +/-1.6)")
    print(f"sun azimuth rate: median {np.median(rate):.4f} deg/hr, p5 {np.percentile(rate, 5):.4f}, "
          f"p95 {np.percentile(rate, 95):.4f} (expected about 0.51)")
    sp.kclear()


if __name__ == "__main__":
    check_rasters()
    check_sun()
