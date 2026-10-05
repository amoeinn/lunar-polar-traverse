"""The Gazebo heightmap and its coordinate mapping, each test naming what it protects.

Three of these exist because the approved design was wrong and Gazebo said so
when it was run. A silent offset or scale error here would misplace every
shadow and every rover position while the rendered picture still looked
entirely plausible, which is the only way this subtask can be wrong: it scores
no prediction, so it cannot fail a bar, only misrepresent.
"""
import json
from pathlib import Path

import numpy as np
import pytest

from src.gzmap import (HEIGHTMAP_N, PX, REGION_N, REGION_OX, REGION_OY, SAMPLING,
                       bilinear, cell_to_map, centre_map, map_to_cell,
                       map_to_world, pos_z, requested_size, span_m,
                       world_to_map)

ROOT = Path(__file__).resolve().parents[1]
TIF = ROOT / "data" / "work" / "gazebo" / "region_heightmap.tif"
REGION_F32 = ROOT / "data" / "work" / "region_height.f32"


def test_the_export_is_the_terrain_the_planner_planned_on():
    """Protects provenance: the heightmap must be the planner's own terrain.

    Not merely a correct reading of the same source. If the heightmap were
    re-read or resampled differently from the array the cost model planned on,
    the demo would show a traverse over ground the planner never saw, and the
    difference would be invisible in any picture. The region overlap is
    therefore asserted equal to the cached region array cell for cell, and the
    raster shape and dtype are asserted because 513 and float32 are what make
    Gazebo neither pad nor quantise.
    """
    if not (TIF.exists() and REGION_F32.exists()):
        pytest.skip("heightmap not exported; run examples/export_heightmap.py")
    import rasterio
    with rasterio.open(TIF) as ds:
        assert ds.width == ds.height == HEIGHTMAP_N, (ds.width, ds.height)
        assert ds.dtypes[0] == "float32", ds.dtypes
        h = ds.read(1)
    region = np.fromfile(REGION_F32, dtype=np.float32).reshape(REGION_N, REGION_N)
    overlap = h[:REGION_N, :REGION_N]
    assert overlap.shape == region.shape
    assert np.array_equal(overlap, region), (
        f"{int(np.count_nonzero(overlap != region))} cells differ from the "
        f"planner's own array")


def test_the_span_correction_is_computed_from_sampling_and_not_written_down():
    """Protects the measured finding that the size asked for is not the span given.

    Ogre lays nv = sampling * (n - 1) + 1 vertices across the terrain and spans
    size * (nv - 1) / nv. Measured in Gazebo 8.15.0 at n = 513, sampling = 2:
    size 10240 gave a half extent of 5115, a 10230 m span and 19.9609 m cells,
    while 10250 gave exactly 5120 and 20.0000 m. The invariant asserted here is
    the one that matters and it must hold for every sampling, because changing
    sampling would otherwise rescale the terrain with nothing to notice it.
    """
    assert requested_size(513, 20.0, 2) == pytest.approx(10250.0, abs=1e-9)
    for sampling in (1, 2, 3, 4):
        nv = sampling * (HEIGHTMAP_N - 1) + 1
        req = requested_size(HEIGHTMAP_N, PX, sampling)
        delivered = req * (nv - 1) / nv
        assert delivered == pytest.approx(span_m(HEIGHTMAP_N, PX), abs=1e-9), sampling
        assert delivered / (HEIGHTMAP_N - 1) == pytest.approx(PX, abs=1e-12), sampling
    # and the correction is a real one: asking for the span you want is wrong
    assert requested_size(HEIGHTMAP_N, PX, SAMPLING) > span_m(HEIGHTMAP_N, PX)


def test_pos_z_is_zero_because_gazebo_already_centres_the_height_range():
    """Protects the measured finding that would have raised the terrain 294.76 m.

    The design specified the height-range midpoint, reasoning that Gazebo
    centres the range on the model pose and so needs it added back. Measured
    otherwise: with size z set to the true range and pos z zero, Gazebo reports
    the heightfield "translated by z=294.762" of its own accord and the extent
    runs from -853.5 to +1443.1 m, the DEM's own minimum and maximum. Adding
    the midpoint as well would have lifted the whole terrain while every
    internal check still passed, because the heightmap would have been
    self-consistently wrong.
    """
    h_min, h_max = -853.541, 1443.066
    midpoint = (h_min + h_max) / 2.0
    assert pos_z(h_min, h_max) == 0.0
    assert pos_z(h_min, h_max) != pytest.approx(midpoint), (
        "pos z is the midpoint again, which lifts the terrain by it")
    # true for any range, not just this one
    assert pos_z(-1.0, 1.0) == 0.0
    assert pos_z(0.0, 1000.0) == 0.0


def test_the_world_states_the_heightmap_size_explicitly():
    """Protects the measured finding that an omitted size loads a one-metre cube.

    sdformat's heightmap_shape.sdf says size "is used if present, otherwise
    defaults to true size of DEM". It does not on this path: with size omitted,
    Gazebo 8.15.0 loaded the heightfield as an AABB of plus or minus 0.4995 m,
    the SDF default of 1 1 1, and reported no warning. The generated world must
    therefore carry a size element, it must carry the corrected span rather
    than the raw one, and its z must be the true height range so the vertical
    scale is exactly one.
    """
    import examples.make_world as MW
    h_range = 2296.6073
    sdf = MW.build("hm.tif", "ours", "tex.png", "nrm.png", 0.0, (0, 0, -1),
                   [], None, "vertices", h_range)
    assert "<size>" in sdf, "no size element: Gazebo would load a one-metre cube"
    want = requested_size(HEIGHTMAP_N, PX, SAMPLING)
    assert f"{want:.4f}" in sdf, f"corrected span {want:.4f} absent from the world"
    assert f"{span_m(HEIGHTMAP_N, PX):.4f} {span_m(HEIGHTMAP_N, PX):.4f}" not in sdf, (
        "the uncorrected span is in the world, which shrinks every cell to "
        "19.9609 m")
    assert f"{h_range:.6f}" in sdf, "the size z is not the true height range"
    # Both geometries, collision and visual, or the physics and the picture
    # would disagree about where the ground is. Counted on the three-component
    # form, because the texture element carries a one-component size too and
    # counting bare tags caught that instead.
    assert sdf.count(f"<size>{want:.4f} {want:.4f} ") == 2, sdf


def test_the_cell_convention_is_a_pixel_edge_origin_with_centres_half_a_pixel_in():
    """Protects the convention taken from the project's own grid construction.

    examples/plan_pilot.load builds its grid as xs = x0 + (arange(n) + 0.5) * px,
    so region_ox and region_oy are pixel edges and cell centres sit half a pixel
    inside them. Dropping the half would shift every cell by 10 m, half a cell,
    which is exactly the size of error that moves a shadow edge by one cell and
    survives any check that uses the same convention on both sides.
    """
    x, y = cell_to_map(0, 0)
    assert x == pytest.approx(REGION_OX + PX / 2.0)
    assert y == pytest.approx(REGION_OY - PX / 2.0)
    # the origin itself is an edge, so no cell centre sits on it
    assert x != pytest.approx(REGION_OX)
    assert y != pytest.approx(REGION_OY)
    for i, j in ((0, 0), (7, 13), (499, 499), (512, 512)):
        bi, bj = map_to_cell(*cell_to_map(i, j))
        assert bi == pytest.approx(i, abs=1e-9), (i, j)
        assert bj == pytest.approx(j, abs=1e-9), (i, j)


def test_map_to_world_is_a_translation_and_never_a_reflection():
    """Protects against the one error this project has already been bitten by.

    The record holds a mirrored projection that "survives every internal
    check", caught only by an external landmark. The same hazard applies to
    placing a heightmap: a row or column flip is self-consistent and invisible
    in a picture. Orientation is asserted directly, and the region's four
    corners are asserted at distinct world positions so that no flip or
    transpose can leave all four unchanged.
    """
    cx, cy = centre_map()
    assert map_to_world(cx, cy) == (pytest.approx(0.0), pytest.approx(0.0))
    # increasing map x must increase world X, and the same for y
    x0, y0 = map_to_world(0.0, 0.0)
    x1, _ = map_to_world(100.0, 0.0)
    _, y1 = map_to_world(0.0, 100.0)
    assert x1 - x0 == pytest.approx(100.0)
    assert y1 - y0 == pytest.approx(100.0)
    corners = {}
    for name, (mx, my) in (("NW", (REGION_OX, REGION_OY)),
                           ("NE", (REGION_OX + REGION_N * PX, REGION_OY)),
                           ("SW", (REGION_OX, REGION_OY - REGION_N * PX)),
                           ("SE", (REGION_OX + REGION_N * PX,
                                   REGION_OY - REGION_N * PX))):
        corners[name] = tuple(float(v) for v in map_to_world(mx, my))
    assert corners["NW"] == (pytest.approx(-5120.0), pytest.approx(5120.0))
    assert corners["SE"] == (pytest.approx(4880.0), pytest.approx(-4880.0))
    assert len(set(corners.values())) == 4, corners
    # a round trip through the inverse, since both are used in the checks
    for mx, my in ((0.0, 0.0), (REGION_OX, REGION_OY), (1234.5, 6789.0)):
        bx, by = world_to_map(*map_to_world(mx, my))
        assert bx == pytest.approx(mx) and by == pytest.approx(my)


def test_bilinear_interpolates_rather_than_snapping_to_a_cell():
    """Protects the comparison the probe check is scored on.

    A settled probe rests between cell centres, so comparing its height against
    the nearest cell would charge the comparison for the terrain's own slope,
    which on this region is a median 12.37 deg and would swamp the 5 m
    tolerance. On a known ramp the interpolant must return the value between
    two cells, not either of them.
    """
    h = np.array([[0.0, 10.0], [0.0, 10.0]])
    assert bilinear(h, 0.0, 0.5) == pytest.approx(5.0)
    assert bilinear(h, 0.0, 0.25) == pytest.approx(2.5)
    assert bilinear(h, 0.0, 0.0) == pytest.approx(0.0)
    assert bilinear(h, 0.0, 1.0) == pytest.approx(10.0)
    g = np.array([[0.0, 0.0], [20.0, 20.0]])
    assert bilinear(g, 0.5, 0.0) == pytest.approx(10.0)
    # both axes at once, and clamped outside rather than wrapping
    d = np.array([[0.0, 1.0], [2.0, 3.0]])
    assert bilinear(d, 0.5, 0.5) == pytest.approx(1.5)
    assert bilinear(d, -1.0, -1.0) == pytest.approx(0.0)
    assert bilinear(d, 9.0, 9.0) == pytest.approx(3.0)
