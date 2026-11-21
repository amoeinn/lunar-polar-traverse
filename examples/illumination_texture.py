"""Write our own illumination as an unlit texture for the Gazebo terrain.

Gazebo's shadows are never a source of truth for illumination here: the scene's
light casts none, and the brightness on screen comes entirely from this file.
The renderer's own shadowing is a rasterisation of a simplified scene and has no
horizon model behind it, so agreeing with it would say nothing about whether our
illumination is right.

The texture carries our model's irradiance: zero where our horizon model says
the cell is shadowed, and the Lambertian cos(incidence) the published _MODEL
products were matched against where it is lit.

It must be applied as an emissive, unlit map. As a diffuse map it would be
multiplied by Gazebo's own normal-dot-light shading and the rendered image
would be our model times Gazebo's, which is neither.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import spiceypy as sp

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import examples.plan_pilot as P                                    # noqa: E402
from examples.build_intervals import PHASE_OFFSET_DAYS, cache_path  # noqa: E402
from src.sun import sun_positions_m                               # noqa: E402
from src.cost import surface_gradient                              # noqa: E402
from src.frames import cell_position, local_frame                   # noqa: E402
from src.gzmap import HEIGHTMAP_N, PX, REGION_N                     # noqa: E402
from src.intervals import LitIntervals                              # noqa: E402

OUT = ROOT / "data" / "work" / "gazebo"
LIT_FLOOR = 0.25           # a lit cell never quantises to zero


def sun_unit_at(tag, t_hours):
    """The sun direction in the Moon-fixed frame at an epoch tag plus t hours."""
    date, phase = tag.rsplit("_p", 1)
    for k in P.KERNELS:
        sp.furnsh(str(ROOT / "data" / "kernels" / k))
    et = (sp.str2et(f"{date}T00:00:00")
          + PHASE_OFFSET_DAYS[int(phase)] * 86400.0 + t_hours * 3600.0)
    pos = sun_positions_m(et, P.FRAME)[0]
    sp.kclear()
    # A direction, not a position: every consumer renormalises it, in irradiance
    # below and in verify_gazebo.sun_direction_world, so the 1000 is a scale
    # nothing reads. Kept rather than dropped so the returned value is bit for
    # bit what it was before the units fix, which never touched this path.
    return pos * 1000.0 / np.linalg.norm(pos)


def irradiance(tag, t_hours, dem, gx, gy, radius):
    """Our model's surface irradiance per cell, normalised to its own maximum.

    cos(incidence) from the DEM's own gradient against the sun direction, gated
    by the horizon model's lit mask. Normalised rather than absolute because a
    texture carries 0 to 1; the absolute scale is the published 42.33
    W/m^2/sr this project already fitted and it is a constant factor here.
    """
    dzdx, dzdy = surface_gradient(dem, PX, gx, gy, radius)
    up, east, north = local_frame(gx.ravel(), gy.ravel(), radius)
    # outward normal of the surface in the local frame, from the map gradients
    nx, ny = -dzdx.ravel(), -dzdy.ravel()
    nrm = np.stack([nx, ny, np.ones_like(nx)], axis=1)
    nrm /= np.linalg.norm(nrm, axis=1)[:, None]
    s = sun_unit_at(tag, t_hours)
    # The sun in each cell's own local frame. Under this south polar
    # stereographic projection the map axes are the local north and east axes
    # rotated by the point's own longitude: with lam = atan2(x, y), the map +x
    # direction is cos(lam)*north - sin(lam)*east and +y is
    # sin(lam)*north + cos(lam)*east. Verify any such basis by finite difference
    # through map_to_unit before using it; a sign error here does not obviously
    # fail, and one survived every internal check in this project until an
    # external landmark caught it.
    lam = np.arctan2(gx.ravel(), gy.ravel())
    ax = np.cos(lam)[:, None] * north - np.sin(lam)[:, None] * east
    ay = np.sin(lam)[:, None] * north + np.cos(lam)[:, None] * east
    sl = np.stack([ax @ s, ay @ s, up @ s], axis=1)
    sl /= np.linalg.norm(sl, axis=1)[:, None]
    cosi = np.clip(np.sum(nrm * sl, axis=1), 0.0, None)
    return cosi


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="2028-02-15_p0")
    ap.add_argument("--hours", type=float, default=0.0)
    args = ap.parse_args()
    from src.cost import Rover
    p, n, px, dem, horizon, cells, gx, gy, edges, _ = P.load(Rover(v_flat=50.0))
    iv = LitIntervals.load(cache_path(args.tag))
    lit = np.asarray([iv.is_lit(c, args.hours) for c in range(n * n)], dtype=bool)
    cosi = irradiance(args.tag, args.hours, dem, gx, gy, p["radius_m"])
    field = np.where(lit, cosi, 0.0).reshape(n, n)

    # 513 x 513 to match the heightmap exactly; the 13 extra rows and columns
    # are outside the planning region and are written dark rather than
    # extrapolated, so nothing invented is ever shown as lit.
    tex = np.zeros((HEIGHTMAP_N, HEIGHTMAP_N), dtype=np.float64)
    tex[:REGION_N, :REGION_N] = field
    peak = float(tex.max())
    img = np.zeros_like(tex) if peak <= 0 else tex / peak
    # A lit cell must not quantise to zero. At this epoch the sun is about a
    # degree up, so cos(incidence) peaks at 0.214 and the median lit cell is
    # far dimmer; normalising by the peak and rounding to 8 bits floored most
    # lit cells to 0 and the texture stopped carrying the mask it exists to
    # carry. Lit cells are therefore floored at LIT_FLOOR, which keeps the
    # zero of the texture meaning exactly "our model says this cell is dark".
    litmap = np.zeros_like(tex, dtype=bool)
    litmap[:REGION_N, :REGION_N] = lit.reshape(n, n)
    img = np.where(litmap, np.maximum(img, LIT_FLOOR), 0.0)
    OUT.mkdir(parents=True, exist_ok=True)
    stem = f"illum_{args.tag}_{args.hours:+.4f}h"
    arr = (np.clip(img, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)
    import matplotlib.image as mpimg
    mpimg.imsave(OUT / f"{stem}.png", arr, cmap="gray", vmin=0, vmax=255)
    np.save(OUT / f"{stem}_mask.npy", lit.reshape(n, n))
    meta = dict(tag=args.tag, hours=args.hours, n=HEIGHTMAP_N, region_n=REGION_N,
                lit_fraction=float(lit.mean()), dark_fraction=float(1 - lit.mean()),
                cos_peak=peak, png=str(OUT / f"{stem}.png"),
                mask=str(OUT / f"{stem}_mask.npy"))
    (OUT / f"{stem}.json").write_text(json.dumps(meta, indent=2))
    print(f"wrote {OUT / f'{stem}.png'}")
    print(f"  epoch {args.tag} at {args.hours:+.4f} h")
    print(f"  our model: {lit.mean():.4%} of {n*n} region cells lit, "
          f"{1-lit.mean():.4%} dark")
    print(f"  cos(incidence) peak over lit cells {peak:.6f}")
    print(f"  texture {HEIGHTMAP_N} x {HEIGHTMAP_N}, 8-bit, the 13 extra rows "
          f"and columns written dark rather than extrapolated")
    return 0


if __name__ == "__main__":
    sys.exit(main())
