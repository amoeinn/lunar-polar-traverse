"""The demo's two guards, both registered in PLAN.md before the demo was built.

One: the instance-selection rule is deterministic and reproduces the registered
candidate counts, so the demo's subject comes from the filters and not from a
preference formed after seeing the candidates.

Two: a plan the replay checker rejected is refused rather than rendered, which
is C5 ("the demo plays back only plans the replay checker accepted") expressed
as a test rather than as a convention someone has to remember.
"""
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import examples.demo_traverse as D                                   # noqa: E402

SWEEP = ROOT / "data" / "work" / "sweep"
needs_sweep = pytest.mark.skipif(
    not (SWEEP / "2028-08-15_p2_replay_td_tol" / "out_verdict.bin").exists(),
    reason="the core sweep outputs are not present")


@needs_sweep
def test_selection_is_deterministic_with_the_registered_counts():
    """Protects: a subject chosen by preference rather than by the filters.

    The rule is only worth registering if running it twice gives the same answer
    and if its counts are the ones the record carries. Both are asserted, and the
    counts are written out so a changed pipeline fails here rather than silently
    selecting a different traverse.
    """
    a1, c1, l1 = D.select_r1()
    b1, d1, m1 = D.select_r1()
    assert (a1, c1, len(l1)) == (b1, d1, len(m1))

    # R1 is empty, and the count that empties it is the pairing clause, not the
    # 24 h clause. If a later run makes R1 non-empty, this fails and the record
    # has to be re-scored rather than quietly gaining a different demo.
    assert c1["snap_plans"] == 322
    assert c1["snap_rejected"] == 131
    assert c1["td_accepted_same_pair"] == 0
    assert l1 == [] and a1 is None

    a2, c2, l2 = D.select_r2()
    assert (a2, c2, l2) == D.select_r2()[:1] + D.select_r2()[1:]
    assert c2 == dict(monitored=45, busts=4)
    assert len(l2) == 4
    assert a2 == dict(tag="2028-02-15_p0", plan_index=27)

    a3, c3, l3 = D.select_r3()
    assert (a3, c3, len(l3)) == (D.select_r3()[0], D.select_r3()[1],
                                 len(D.select_r3()[2]))
    assert c3 == dict(monitored_nonbust=41, false_alarms=17, removed_by_lit=11)
    assert len(l3) == 11
    assert a3 == dict(tag="2028-08-15_p2", plan_index=29)


@needs_sweep
def test_the_subject_is_a_plan_the_checker_accepted():
    """Protects: driving a plan the replay checker rejected, against C5.

    Asserted from the checker's own out_verdict.bin and not from a filename, as
    C5 requires, and asserted on the instance the demo actually renders rather
    than on one chosen here.
    """
    choice, _, _ = D.select_r3()
    rep = D.replay_out(choice["tag"], "td")
    assert bool(rep["verdict"][choice["plan_index"]]) is True
    assert float(rep["shadow"][choice["plan_index"]]) <= D.BUDGET_H


@needs_sweep
def test_a_rejected_plan_is_refused_rather_than_rendered():
    """Protects: the same, as behaviour rather than as a property of one choice.

    R2's bust is a plan the checker rejected. Handing it to the drive path must
    raise, and the discriminating half of this test is the first assertion: the
    fixture is only a fixture if the plan really is rejected, so that is checked
    before the refusal is.
    """
    bust, _, _ = D.select_r2()
    rep = D.replay_out(bust["tag"], "td")
    assert bool(rep["verdict"][bust["plan_index"]]) is False, \
        "the fixture must be a rejected plan or this test proves nothing"
    assert float(rep["shadow"][bust["plan_index"]]) > D.BUDGET_H

    with pytest.raises(SystemExit, match="checker rejected"):
        D.assert_drivable(bust["tag"], bust["plan_index"])

    ok, _, _ = D.select_r3()
    D.assert_drivable(ok["tag"], ok["plan_index"])    # must not raise


@needs_sweep
def test_frame_times_are_sun_table_samples():
    """Protects: a frame showing an illumination state the planner never saw.

    Every frame time must be an exact multiple of the registered 0.25 h cadence
    the sun table and the interval tables were built at, and the sequence must
    bracket the traverse rather than clip it.
    """
    t = D.frame_times(0.0, 27.6608)
    assert np.allclose(t / D.CADENCE_H, np.round(t / D.CADENCE_H))
    assert t[0] <= 0.0 and t[-1] >= 27.6608
    assert np.allclose(np.diff(t), D.CADENCE_H)
    # A departure off the sample grid must still land on it.
    t2 = D.frame_times(0.31, 1.0)
    assert np.allclose(t2 / D.CADENCE_H, np.round(t2 / D.CADENCE_H))
    assert t2[0] <= 0.31 and t2[-1] >= 1.0


def test_the_sequence_normalisation_is_shared_not_per_frame():
    """Protects: the committed per-frame normalisation leaking into a sequence.

    examples/illumination_texture.py divides each frame by that frame's own peak,
    which is right for one frame and wrong for a sequence: the sun elevation
    moves over a traverse, so the picture would hold its brightness while the
    irradiance fell. Two frames with different peaks must keep their ratio.
    """
    tmp = Path(__file__).resolve().parent / "_tmp_tex"
    tmp.mkdir(exist_ok=True)
    try:
        bright = np.zeros((D.REGION_N, D.REGION_N))
        bright[0, 0] = 0.40
        dim = np.zeros((D.REGION_N, D.REGION_N))
        dim[0, 0] = 0.20
        mask = np.zeros((D.REGION_N, D.REGION_N), bool)
        mask[0, 0] = True
        peak = 0.40
        import matplotlib.image as mpimg
        D.texture_png(bright, mask, peak, tmp / "b.png")
        D.texture_png(dim, mask, peak, tmp / "d.png")
        b = mpimg.imread(tmp / "b.png")[0, 0, 0]
        d = mpimg.imread(tmp / "d.png")[0, 0, 0]
        assert b > d, "a dimmer frame must render dimmer on a shared peak"
        assert abs(d / b - 0.5) < 0.02, (b, d)
    finally:
        for f in tmp.glob("*.png"):
            f.unlink()
        tmp.rmdir()


def test_the_lit_mask_is_not_recovered_from_the_field():
    """Protects: calling a lit cell dark because cos(incidence) clipped to zero.

    At the demo's own epoch the sun is about a degree up and cos(incidence) is
    exactly zero on 153188 of 168873 lit cells, so a mask taken as "field > 0"
    inverts the picture over 90% of the lit region. The texture must floor by
    the mask it is given.
    """
    import matplotlib.image as mpimg
    tmp = Path(__file__).resolve().parent / "_tmp_mask"
    tmp.mkdir(exist_ok=True)
    try:
        field = np.zeros((D.REGION_N, D.REGION_N))
        mask = np.zeros((D.REGION_N, D.REGION_N), bool)
        mask[5, 5] = True            # lit, but grazing: cos(incidence) == 0
        D.texture_png(field, mask, 0.4, tmp / "t.png")
        img = mpimg.imread(tmp / "t.png")
        # Cell indices, not texel indices: each cell is an UPSAMPLE-wide block,
        # so a raw (5, 5) would land inside cell (0, 0).
        u = D.UPSAMPLE
        lit = img[5 * u:6 * u, 5 * u:6 * u, 0]
        dark = img[6 * u:7 * u, 6 * u:7 * u, 0]
        assert lit.min() > 0.2, "a lit cell at zero cos must not render dark"
        assert dark.max() == 0.0, "an unlit cell must render at exactly zero"
    finally:
        for f in tmp.glob("*.png"):
            f.unlink()
        tmp.rmdir()


def test_look_at_points_the_camera_down_the_line_of_sight():
    """Protects: a Gazebo view aimed somewhere other than the rover.

    A sign error in the pitch or the yaw gives a camera pointing at empty sky,
    which is obvious once rendered and silent until then. Checked by rotating a
    unit +x vector by the returned angles and comparing it with the direction it
    was asked for.
    """
    for cam, tgt in (((0.0, 0.0, 100.0), (50.0, 0.0, 0.0)),
                     ((-30.0, -30.0, 14.0), (0.0, 0.0, 1.2)),
                     ((10.0, -5.0, 3.0), (-4.0, 8.0, 9.0))):
        _, pitch, yaw = D.look_at(cam, tgt)
        cp, sp_ = np.cos(pitch), np.sin(pitch)
        cy, sy = np.cos(yaw), np.sin(yaw)
        # Gazebo applies yaw about +z then pitch about the new +y, and +pitch
        # tips the camera's +x axis downward.
        fwd = np.array([cy * cp, sy * cp, -sp_])
        want = np.array(tgt) - np.array(cam)
        want = want / np.linalg.norm(want)
        assert np.allclose(fwd, want, atol=1e-9), (cam, tgt, fwd, want)


def test_the_texture_is_replicated_not_interpolated():
    """Protects: a resampled texture that invents values between cells.

    The chase view is readable only because each 20 m cell is written as a block
    of identical texels, so the renderer's filter has nothing to blend except at
    block edges. If the upsample ever became an interpolating resize, the
    picture would look smoother and would be showing values our model never
    produced, which is the one thing this texture must not do.

    The discriminating property is checked in the fixture first: the two cells
    used must actually differ, or a uniformity assertion proves nothing.
    """
    import matplotlib.image as mpimg
    tmp = Path(__file__).resolve().parent / "_tmp_up"
    tmp.mkdir(exist_ok=True)
    try:
        field = np.zeros((D.REGION_N, D.REGION_N))
        mask = np.zeros((D.REGION_N, D.REGION_N), bool)
        field[3, 3], mask[3, 3] = 0.40, True       # lit, bright
        mask[3, 4] = False                          # its neighbour: dark
        assert field[3, 3] != field[3, 4], "the fixture cells must differ"

        up = 8
        D.texture_png(field, mask, 0.40, tmp / "u.png", upsample=up)
        img = mpimg.imread(tmp / "u.png")[..., 0]
        assert img.shape == (D.HEIGHTMAP_N * up, D.HEIGHTMAP_N * up), img.shape

        # the lit cell is one flat block, not a blob with a falloff
        blk = img[3 * up:4 * up, 3 * up:4 * up]
        assert blk.min() == blk.max(), (blk.min(), blk.max())
        assert blk.min() > 0.2

        # its dark neighbour is flat too, and the step between them is abrupt:
        # exactly one texel wide, with no intermediate value anywhere
        nbr = img[3 * up:4 * up, 4 * up:5 * up]
        assert nbr.min() == nbr.max() == 0.0
        row = img[3 * up + up // 2, 3 * up:5 * up]
        between = row[(row > 0.01) & (row < blk.min() - 0.01)]
        assert between.size == 0, f"{between.size} interpolated texels"

        # and upsample=1 reproduces the original size, so the factor is the
        # only thing that changed
        D.texture_png(field, mask, 0.40, tmp / "o.png", upsample=1)
        assert mpimg.imread(tmp / "o.png").shape[0] == D.HEIGHTMAP_N
    finally:
        for f in tmp.glob("*.png"):
            f.unlink()
        tmp.rmdir()


def test_the_texture_footprint_is_set_by_span_not_by_pixel_count():
    """Protects: an upsampled texture mapped to a different patch of ground.

    The world's <texture><size> is the ground size the image is mapped across.
    If it ever came from the image's pixel count, upsampling by 8 would shrink
    the footprint to 1.28 km and put every shadow in the wrong place while still
    rendering a plausible-looking scene.
    """
    import examples.make_world as MW
    sdf = MW.build("hm.tif", "ours", "t.png", "n.png", 0.0, (0, 0, -1), [], None,
                   "vertices", 2296.6073)
    tex = sdf[sdf.index("<texture>"):sdf.index("</texture>")]
    size = float(tex[tex.index("<size>") + 6:tex.index("</size>")])
    assert size == 10240.0, size
    assert abs(size - (D.HEIGHTMAP_N - 1) * D.PX) < 1e-6
