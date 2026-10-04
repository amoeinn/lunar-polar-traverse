"""The map-to-Gazebo coordinate mapping, and nothing else.

A silent offset here would misplace every shadow and every rover position while
the picture still looked plausible, so the arithmetic lives in one place with
its own tests rather than being spread through the export and the world file.

The conventions, both taken from the project's own code rather than restated:
region_ox and region_oy are pixel *edges*, and cell centres sit half a pixel
inside them, which is how examples/plan_pilot.load builds its grid
(xs = x0 + (arange(n) + 0.5) * px). The heightmap is aligned to the region
origin and extends east and south, so the region occupies the first 500 x 500
cells of the 513 x 513 raster.

One thing here is measured rather than assumed. Gazebo loads a DEM through
gz-common, whose Dem.hh exposes WorldWidth() but ships no implementation to
read, so whether N samples span N * px or (N - 1) * px is not knowable from the
installed headers. Both conventions are implemented and examples/verify_gazebo.py
determines which one the simulator actually uses from a synthetic ramp before
anything else is checked against it. The difference is 0.2% of the footprint,
about 10 m at a corner, which is small enough to hide behind real terrain and
large enough to matter.
"""
import numpy as np

HEIGHTMAP_N = 513          # 2^9 + 1, the Ogre constraint Dem.hh states
PX = 20.0                  # metres per cell, the DEM's own resolution
REGION_N = 500
REGION_OX = -4000.0        # pixel edge, not a cell centre
REGION_OY = 11000.0
SPAN_CONVENTIONS = ("vertices", "extent")
SAMPLING = 2               # samples per heightmap datum, the sdformat default


def requested_size(n=HEIGHTMAP_N, px=PX, sampling=SAMPLING, convention="vertices"):
    """The size element that yields a true px spacing, measured not assumed.

    Ogre lays nv = sampling * (n - 1) + 1 vertices across the terrain and spans
    size * (nv - 1) / nv, so asking for the span you want comes out short by
    one vertex interval. Measured in Gazebo 8.15.0: size 10240 gave a half
    extent of 5115 and size 10250 gave exactly 5120, which is the 10240 m span
    that puts 20 m between original cells. The correction depends on sampling,
    which is the trap: changing sampling silently rescales the terrain.
    """
    nv = sampling * (n - 1) + 1
    return span_m(n, px, convention) * nv / (nv - 1)


def span_m(n=HEIGHTMAP_N, px=PX, convention="vertices"):
    """World size of a heightmap of n samples under either convention."""
    if convention not in SPAN_CONVENTIONS:
        raise ValueError(f"unknown span convention: {convention}")
    return (n - 1) * px if convention == "vertices" else n * px


def centre_map(n=HEIGHTMAP_N, px=PX, ox=REGION_OX, oy=REGION_OY,
               convention="vertices"):
    """The heightmap's centre in map metres, which is where Gazebo pins it.

    Gazebo centres a heightmap on its model pose, so this is the point that
    world (0, 0) corresponds to.
    """
    half = span_m(n, px, convention) / 2.0
    return ox + half, oy - half


def cell_to_map(i, j, px=PX, ox=REGION_OX, oy=REGION_OY):
    """Cell centre in map metres. i is the row, increasing southward."""
    return ox + (np.asarray(j, float) + 0.5) * px, oy - (np.asarray(i, float) + 0.5) * px


def map_to_cell(x, y, px=PX, ox=REGION_OX, oy=REGION_OY):
    """Fractional cell indices of a map position, in the same convention."""
    return (oy - np.asarray(y, float)) / px - 0.5, (np.asarray(x, float) - ox) / px - 0.5


def map_to_world(x, y, convention="vertices", **kw):
    cx, cy = centre_map(convention=convention, **kw)
    return np.asarray(x, float) - cx, np.asarray(y, float) - cy


def world_to_map(wx, wy, convention="vertices", **kw):
    cx, cy = centre_map(convention=convention, **kw)
    return np.asarray(wx, float) + cx, np.asarray(wy, float) + cy


def bilinear(h, i, j):
    """Height at fractional cell indices, clamped at the raster edge.

    The probes settle between cell centres, so comparing them against the
    nearest cell would charge the comparison for the terrain's own slope.
    """
    h = np.asarray(h, float)
    n_i, n_j = h.shape
    i = np.clip(np.asarray(i, float), 0.0, n_i - 1.0)
    j = np.clip(np.asarray(j, float), 0.0, n_j - 1.0)
    i0 = np.clip(np.floor(i).astype(int), 0, n_i - 2) if n_i > 1 else np.zeros_like(i, int)
    j0 = np.clip(np.floor(j).astype(int), 0, n_j - 2) if n_j > 1 else np.zeros_like(j, int)
    i1 = np.minimum(i0 + 1, n_i - 1)
    j1 = np.minimum(j0 + 1, n_j - 1)
    fi, fj = i - i0, j - j0
    return ((1 - fi) * ((1 - fj) * h[i0, j0] + fj * h[i0, j1])
            + fi * ((1 - fj) * h[i1, j0] + fj * h[i1, j1]))


def pos_z(h_min, h_max):
    """The pos offset that makes world z equal the DEM height: zero.

    Designed as the height-range midpoint on the reasoning that Gazebo centres
    the range on the model pose. Measured otherwise: with size z set to the
    true range and pos z zero, Gazebo 8.15.0 reports the heightfield translated
    by +294.762, the midpoint, of its own accord, and the resulting extent runs
    from -853.5 to +1443.1, which is the DEM's own minimum and maximum. Adding
    the midpoint as well would have put the whole terrain 294.76 m too high.
    The arguments are kept so the midpoint stays computable and reportable.
    """
    _ = (float(h_min) + float(h_max)) / 2.0
    return 0.0
