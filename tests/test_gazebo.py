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
    # These moved by half a cell when centre_map was corrected from the
    # extent's centre to the middle sample's centre. They are written out
    # rather than derived from centre_map so that a change to the convention
    # has to be made deliberately here too, instead of following silently.
    assert corners["NW"] == (pytest.approx(-5130.0), pytest.approx(5130.0))
    assert corners["SE"] == (pytest.approx(4870.0), pytest.approx(-4870.0))
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


def test_the_gravity_element_sits_inside_the_link_where_sdformat_reads_it():
    """Protects the C5 free fall: <gravity> placed before <link> is dropped.

    <gravity> is a link element. Written as a child of <model> sdformat
    discards it with a warning and still reports the world Valid, so the rover
    fell with the element it needed sitting three lines above the link it
    belonged in. Asserting the text alone would not catch the position, so this
    checks that the element is inside a link block.
    """
    import examples.make_world as MW
    rover = ROOT / "data" / "work" / "gazebo" / "rover.sdf"
    if not rover.exists():
        pytest.skip("rover.sdf not converted; run the C4 conversion first")
    sdf = MW.build("hm.tif", "ours", "", "", 0.0, (0, 0, -1), [], None,
                   "vertices", 2296.6073, rover=str(rover))
    body = sdf[sdf.index("<model name='rover'"):]
    body = body[:body.index("</model>")]
    assert "<gravity>false</gravity>" in body, "the rover has no gravity override"
    link_at = body.index("<link name=")
    grav_at = body.index("<gravity>false</gravity>")
    assert grav_at > link_at, (
        "<gravity> is before the <link> tag, so it is a child of <model> and "
        "sdformat drops it")


def test_the_world_declares_the_system_that_provides_set_pose():
    """Protects C5's pose commanding: set_pose has no provider without it.

    Declaring plugins explicitly replaces Gazebo's defaults. With only Physics,
    Sensors and SceneBroadcaster declared, the running world listed no
    set_pose, no create and no remove, and every call returned an empty reply
    that was read for three attempts as an uninformative result rather than as
    the service not existing.
    """
    import examples.make_world as MW
    sdf = MW.build("hm.tif", "ours", "", "", 0.0, (0, 0, -1), [], None,
                   "vertices", 2296.6073)
    assert "gz-sim-user-commands-system" in sdf, (
        "UserCommands is not declared, so /world/<name>/set_pose has no "
        "provider at all")


def test_the_heightmap_centre_is_the_middle_sample_not_the_middle_of_the_extent():
    """Protects the half-cell offset that L2' found, and the odd-count rule.

    Gazebo pins a heightmap at the centre of the terrain it builds, and that
    terrain is built on the samples, so the pinned point is the middle
    sample's centre. The extent's centre is half a pixel away from it whenever
    the sample count is odd, and 513 is odd. Returning the extent's centre put
    every map-to-world conversion 14.14 m off diagonally, which the rolling
    spheres could not resolve and the fixed ray could: the median absolute
    height residual over 458752 pixels was 3.6869 m at zero shift and 0.2268 m
    half a cell across.

    The even case is asserted alongside it, because that is where the two
    definitions agree and a fix that simply added a constant 10 m would pass
    the odd case and break this one.
    """
    mid = (HEIGHTMAP_N - 1) // 2
    assert HEIGHTMAP_N % 2 == 1, "this heightmap is 2^n+1, so the count is odd"
    assert centre_map() == pytest.approx(
        tuple(float(v) for v in cell_to_map(mid, mid))), (
        "the centre is not the middle sample's centre, so world (0, 0) is half "
        "a cell from where Gazebo pins the terrain")

    extent = (REGION_OX + span_m(HEIGHTMAP_N, PX) / 2.0,
              REGION_OY - span_m(HEIGHTMAP_N, PX) / 2.0)
    off = np.hypot(centre_map()[0] - extent[0], centre_map()[1] - extent[1])
    assert off == pytest.approx(PX / 2.0 * np.sqrt(2.0)), (
        f"the gap to the extent's centre is {off:.3f} m, not the half-cell "
        f"diagonal {PX/2.0*np.sqrt(2.0):.3f} m")

    # Even sample count: here the two definitions do coincide, because the
    # span is (n-1)*px and (n-1)/2 is a whole number of cells from the origin.
    # Asserted so that a fix which simply added a constant 10 m would pass the
    # odd case above and fail here.
    n_even = 512
    mid_e = (n_even - 1) // 2
    c_even = centre_map(n=n_even)
    ext_even = (REGION_OX + span_m(n_even, PX) / 2.0,
                REGION_OY - span_m(n_even, PX) / 2.0)
    assert c_even == pytest.approx(
        tuple(float(v) for v in cell_to_map(mid_e, mid_e))), (
        "the even case is not the middle sample either")
    assert c_even == pytest.approx(ext_even), (
        "for an even count the middle sample and the extent's centre coincide; "
        "if they do not, the fix is a constant offset rather than the sample")


def _xacro_props():
    """The xacro's declared property values, by name."""
    import re
    txt = (ROOT / "gazebo" / "rover.urdf.xacro").read_text()
    return {m.group(1): m.group(2) for m in
            re.finditer(r'xacro:property name="([^"]+)"\s+value="([^"]+)"', txt)}


def test_the_sourced_sensor_figures_are_the_ones_the_model_declares():
    """Protects the second sourcing pass against a silent revert.

    Three assumption groups were sourced on 2026-10-04 and two were wrong as
    assumed: the Navcam stereo baseline was 0.42 m and is 0.424 m per the PDS
    context product, the LiDAR vertical field of view was 45.0 deg and is
    42.4 deg per the Ouster REV7.0 datasheet, and the LiDAR range was 120 m,
    the superseded Gen 1 figure at 80% reflectivity, against 90 m for a 10%
    target, which is the right column because this project's illumination
    model fits an albedo of 0.1. The old values are asserted absent, not just
    the new ones present, because a revert would otherwise pass.
    """
    p = _xacro_props()
    assert p["nav_baseline"] == "0.424", p["nav_baseline"]
    assert p["haz_fov_deg"] == "124.0", p["haz_fov_deg"]
    assert p["lidar_vfov"] == "42.4", p["lidar_vfov"]
    assert p["lidar_range"] == "90.0", p["lidar_range"]
    assert p["nav_baseline"] != "0.42", "reverted to the assumed baseline"
    assert p["lidar_vfov"] != "45.0", "reverted to the Gen 2 marketing figure"
    assert p["lidar_range"] != "120.0", "reverted to the Gen 1 range"


def test_the_predicted_stereo_disparity_follows_from_the_sourced_baseline():
    """Protects S2's prediction, which must come from the sourced figures.

    If the prediction were hard-coded, changing the baseline or the field of
    view would leave the check scoring against a stale number and still
    passing. fx comes from the declared field of view and the detector width,
    and the disparity from fx, the baseline and the range.
    """
    p = _xacro_props()
    px = float(p["nav_px"])
    fov = np.radians(float(p["nav_fov_deg"]))
    fx = px / (2.0 * np.tan(fov / 2.0))
    assert fx == pytest.approx(1236.077, abs=1e-3), fx
    disp = fx * float(p["nav_baseline"]) / 10.0
    assert disp == pytest.approx(52.410, abs=1e-3), disp
    # the 4 mm correction is deliberately below this check's resolution, and
    # that is stated rather than left for a reader to assume otherwise
    old = fx * 0.420 / 10.0
    assert abs(disp - old) < 0.5, (
        "the baseline correction is claimed to be unresolvable at 10 m but is "
        f"{abs(disp - old):.2f} px, which a 2 px tolerance would resolve")


def test_the_dark_world_renders_only_emissive_geometry():
    """Protects S2's target isolation.

    The first run thresholded a lit scene and found 240207 bright pixels, a
    quarter of the image, because the ambient-lit terrain is as bright as the
    target. The terrain cannot supply a matchable feature either: its texture
    is one texel per 20 m cell, so at a range short enough for measurable
    disparity the whole field of view is inside one texel. The dark world has
    ambient zero and no light at all, so only the emissive sphere renders.
    """
    import examples.make_world as MW
    sdf = MW.build("hm.tif", "ours", "", "", 0.0, (0, 0, -1), [], None,
                   "vertices", 2296.6073, dark=True,
                   targets=[("t0", 1.0, 2.0, 3.0, 0.5)])
    assert "<light" not in sdf, "the dark world still carries a light"
    assert "<ambient>0 0 0 1</ambient>" in sdf, "the scene ambient is not zero"
    assert "target_t0" in sdf and "<emissive>1 1 1 1</emissive>" in sdf


def test_the_rover_pose_reaches_the_world_rather_than_only_being_computed():
    """Protects the bug that voided the first sensor run.

    c4x computed a rover pose and never passed it to the world, so the rover
    stayed at its default (0, 0, 2000) while the target sphere sat 10 m from
    where the rover was supposed to be. The pose read back off the topic was
    3919.5 m from the commanded one, and S2 and S3 would have been scored
    against geometry that was not the geometry in the world.
    """
    import examples.make_world as MW
    rover = ROOT / "data" / "work" / "gazebo" / "rover.sdf"
    if not rover.exists():
        pytest.skip("rover.sdf not converted")
    def model_pose(pose):
        sdf = MW.build("hm.tif", "ours", "", "", 0.0, (0, 0, -1), [], None,
                       "vertices", 2296.6073, rover=str(rover), rover_pose=pose)
        body = sdf[sdf.index("<model name='rover'"):]
        body = body[:body.index("</model>")]
        first = body[body.index("<pose>") + 6:body.index("</pose>")]
        return [float(v) for v in first.split()], body

    # Parsed rather than string-matched. The quantity this protects is the pose
    # reaching the world, not how a zero is spelled, and a match on the exact
    # text broke the moment the pose gained a yaw field while the geometry it
    # guards was unchanged.
    want = (123.5, -456.25, 789.125)
    got, body = model_pose(want)
    assert got == [123.5, -456.25, 789.125, 0.0, 0.0, 0.0], body[:300]
    assert "2000.0" not in body.split("</pose>")[0]

    # The demo turns the rover along its path, so a four-element pose must carry
    # its yaw through to the world and must not disturb the position.
    got, body = model_pose((123.5, -456.25, 789.125, 1.25))
    assert got == [123.5, -456.25, 789.125, 0.0, 0.0, 1.25], body[:300]


def _generated_world(tmp_path, texture_rel):
    """A world built with a RELATIVE texture path, which is the trigger."""
    import examples.make_world as MW
    return MW.build(texture_rel, "ours", texture_rel, texture_rel, 0.0,
                    (0, 0, -1), [], None, "vertices", 2296.6073)


def test_every_file_uri_in_the_world_is_absolute():
    """Protects the world against being loadable only from the repo root.

    A relative file:// URI resolves against whatever working directory the
    simulator was started from. The heightmap and normal map were absolute by
    accident, one from the sidecar and one generated into an absolute
    directory, so only a user-supplied --texture came out relative. Every
    scripted run passed an absolute path, so the suite and all five
    verification parts passed while the world could not be opened from
    anywhere but the repo root: launching it by hand from a home directory
    failed with "uri [file://data/work/...] could not be resolved" and no
    world at all.

    Built here with a deliberately relative path, because passing an absolute
    one would make the assertion vacuous.
    """
    import re
    sdf = _generated_world(None, "data/work/gazebo/some_texture.png")
    uris = re.findall(r"<(?:uri|diffuse|normal)>file://([^<]*)</", sdf)
    assert uris, "no file:// URIs found, so this test is asserting nothing"
    bad = [u for u in uris if not u.startswith("/")]
    assert not bad, f"relative file:// URIs: {bad}"


@pytest.mark.skipif(not Path("/usr/bin/gz").exists(), reason="gz is not present")
def test_the_generated_world_loads_from_a_different_working_directory():
    """Loads the world in the simulator from outside the repo, as Ali did.

    The string assertion above can be satisfied by a path that is absolute and
    wrong. This one starts the server on the generated world from a temporary
    directory and fails on the parser's own resolve error, which is the
    failure that was actually hit.
    """
    import os
    import subprocess
    import tempfile
    import examples.make_world as MW
    tif = ROOT / "data" / "work" / "gazebo" / "region_heightmap.tif"
    if not tif.exists():
        pytest.skip("heightmap not exported")
    tex = sorted((ROOT / "data" / "work" / "gazebo").glob("illum_*h.png"))
    if not tex:
        pytest.skip("no illumination texture generated")
    # relative paths, as the failing command line supplied them
    rel_tif = os.path.relpath(tif, ROOT)
    rel_tex = os.path.relpath(tex[0], ROOT)
    rel_nrm = os.path.relpath(MW.flat_normal_png(
        ROOT / "data" / "work" / "gazebo" / "flat_normal.png"), ROOT)
    cwd = os.getcwd()
    try:
        os.chdir(ROOT)
        sdf = MW.build(rel_tif, "ours", rel_tex, rel_nrm, 0.0, (0, 0, -1),
                       [], None, "vertices", 2296.6073)
    finally:
        os.chdir(cwd)
    with tempfile.TemporaryDirectory() as td:
        world = Path(td) / "w.sdf"
        world.write_text(sdf)
        env = dict(os.environ, GZ_CONFIG_PATH="/usr/share/gz")
        r = subprocess.run(["/usr/bin/gz", "sim", "-s", "-r",
                            "--iterations", "1", str(world)],
                           cwd=td, env=env, capture_output=True, text=True,
                           timeout=180)
    out = r.stdout + r.stderr
    assert "could not be resolved" not in out, (
        "the world does not load outside the repo root:\n"
        + "\n".join(l for l in out.splitlines() if "resolve" in l.lower()))
    assert "Failed to load a world" not in out, out[-600:]
