"""The replay checker, each test naming what it protects.

The checker's job is to reject plans that do not work, so every test here is
about a rejection it must make. A checker that accepts everything passes no
test worth having.
"""
import numpy as np
import pytest

from src.cost import MOON_RADIUS_M, Rover
from src.replay import BINARY, ReplaySettings, replay

R = MOON_RADIUS_M
AU = 1.495978707e11
PX = 20.0
N = 60                 # region side, cells
NEAR = 120             # near raster side, cells
pytestmark = pytest.mark.skipif(not BINARY.exists(), reason="replay binary not built")


def scene(tmp_path, height, ox=4000.0, oy=11000.0):
    """A synthetic near raster and the geometry describing it.

    The reach is cut to 400 m and the far field is pushed past it, so only the
    near raster is needed and a test does not have to synthesise 238 km of DEM.
    """
    near_ox = ox - (NEAR - N) / 2 * PX
    near_oy = oy + (NEAR - N) / 2 * PX
    near = tmp_path / "near.f32"
    near.write_bytes(np.asarray(height, dtype=np.float32).tobytes())
    far = tmp_path / "far.f32"
    far.write_bytes(np.zeros((4, 4), dtype=np.float32).tobytes())
    return dict(radius_m=R, region_ox=ox, region_oy=oy, region_px=PX, region_nx=N,
                near_path=str(near), near_ox=near_ox, near_oy=near_oy, near_px=PX,
                near_width=NEAR, near_height=NEAR,
                far_path=str(far), far_ox=0.0, far_oy=0.0, far_px=1000.0,
                far_width=4, far_height=4,
                reach_m=400.0, switch_m=1.0e9, near_step_m=20.0, far_step_m=80.0)


def flat():
    return np.zeros((NEAR, NEAR))


def sun_table(elevation_deg, azimuth_deg=0.0, n=16, dt=1.0):
    """A sun held at a fixed elevation and azimuth in the region centre's frame."""
    from src.frames import local_frame
    cx, cy = 4000.0 + N * PX / 2, 11000.0 - N * PX / 2
    up, east, north = local_frame(np.array([cx]), np.array([cy]), R)
    e, a = np.radians(elevation_deg), np.radians(azimuth_deg)
    d = (np.cos(e) * (np.cos(a) * north[0] + np.sin(a) * east[0])
         + np.sin(e) * up[0])
    return np.repeat((AU * d)[None, :], n, axis=0), dt


def straight_plan(row, c0, c1):
    return [row * N + c for c in range(c0, c1 + 1)]


def test_a_flat_lit_traverse_is_accepted_with_the_expected_time():
    """Protects the baseline behaviour: the checker must not reject what works."""
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        from pathlib import Path
        td = Path(td)
        grid = scene(td, flat())
        suns, dt = sun_table(5.0)
        rover = Rover(v_flat=50.0)
        plan = straight_plan(30, 10, 30)
        r = replay([plan], [0.0], grid, rover, suns, dt, 2.0, td)
        assert r["verdict"][0], (r["illegal"][0], r["shadow"][0])
        assert r["illegal"][0] == 0
        assert r["shadow"][0] == pytest.approx(0.0)
        # 20 cells of 20 m at 50 m/hr on the level
        assert r["arrival"][0] == pytest.approx(20 * PX / 50.0, rel=0.02)
        assert np.max(np.abs(r["step_pitch"])) < 0.01
        assert np.max(np.abs(r["step_roll"])) < 0.01


def test_a_step_through_a_forbidden_edge_is_rejected():
    """Protects the legality check: a plan the planner called feasible but is not.

    The terrain is flat except for a ramp across the middle of the route steep
    enough to break the roll limit. The planner's graph would never offer that
    edge; this injects it, which is the only way to test that the checker
    notices.
    """
    import tempfile
    from pathlib import Path
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        h = flat()
        # a cross-route ramp: height rising with row, so travelling along a row
        # is pure roll
        h[:, :] = np.arange(NEAR)[:, None] * 12.0
        grid = scene(td, h)
        suns, dt = sun_table(45.0)
        rover = Rover(v_flat=50.0, pitch_max=20.0, roll_max=15.0)
        plan = straight_plan(30, 10, 30)
        r = replay([plan], [0.0], grid, rover, suns, dt, 2.0, td,
                   ReplaySettings(limit_tolerance_deg=0.0))
        assert np.max(np.abs(r["step_roll"])) > rover.roll_max, r["step_roll"].max()
        assert r["illegal"][0] > 0
        assert not r["verdict"][0]


def test_a_plan_whose_shadow_exceeds_tolerance_is_rejected():
    """Protects the shadow accounting: darkness must accumulate and be judged."""
    import tempfile
    from pathlib import Path
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        grid = scene(td, flat())
        suns, dt = sun_table(-1.0)          # sun below the local horizontal
        rover = Rover(v_flat=50.0)
        plan = straight_plan(30, 10, 30)
        # 20 cells of 20 m at 50 m/hr is 8 h, so the whole traverse in shadow
        # accrues 8 h and the tolerances have to straddle that.
        loose = replay([plan], [0.0], grid, rover, suns, dt, 10.0, td)
        assert loose["shadow"][0] > 7.0, loose["shadow"][0]
        assert loose["verdict"][0], "8 h of shadow is inside a 10 h tolerance"
        tight = replay([plan], [0.0], grid, rover, suns, dt, 0.1, td)
        assert not tight["verdict"][0]
        assert tight["shadow"][0] == pytest.approx(loose["shadow"][0], rel=1e-9)


def test_the_checker_sees_the_sun_rather_than_assuming_it():
    """Protects the ray march: the same geometry lit and unlit must differ."""
    import tempfile
    from pathlib import Path
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        grid = scene(td, flat())
        rover = Rover(v_flat=50.0)
        plan = straight_plan(30, 10, 30)
        up_s, dt = sun_table(5.0)
        down_s, _ = sun_table(-1.0)
        lit = replay([plan], [0.0], grid, rover, up_s, dt, 2.0, td)
        dark = replay([plan], [0.0], grid, rover, down_s, dt, 2.0, td)
        assert lit["shadow"][0] == pytest.approx(0.0)
        assert dark["shadow"][0] > 0.1


def test_reuse_reads_a_finished_run_back_and_refuses_a_different_one():
    """Protects the sweep's resume path, and the hazard it used to carry.

    A sweep piece that outlives its session must not lose the plans it already
    integrated, so replay(reuse=True) returns a finished run's outputs from its
    work directory instead of repeating the integration.

    Until 2026-11-17 that cache was keyed on the directory and on nothing else:
    it did not know which plans produced it, so a caller that reused a
    directory for different plans silently got the old answer. The run killed
    on 2026-11-16 left exactly that on disk, a directory whose params described
    a call that never wrote beside outputs from the call before it, and a
    resumed sweep would have read 120 mean-map verdicts back as a 64-plan
    time-dependent result. reuse now compares the plan inputs on disk with the
    ones the call would write, and the output sizes with the ones they must
    have, so a directory holding another call's work is integrated afresh
    rather than returned. Both halves are asserted.
    """
    import tempfile
    from pathlib import Path
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        grid = scene(td, flat())
        suns, dt = sun_table(5.0)
        rover = Rover(v_flat=50.0)
        work = td / "run"
        short = straight_plan(30, 10, 20)
        long_ = straight_plan(30, 10, 40)
        first = replay([short], [0.0], grid, rover, suns, dt, 2.0, work)
        again = replay([short], [0.0], grid, rover, suns, dt, 2.0, work, reuse=True)
        for k in ("arrival", "shadow", "verdict", "illegal", "step_offsets"):
            assert np.array_equal(first[k], again[k]), k

        # a different plan in the same directory is integrated afresh, with or
        # without reuse, and gives that plan's own arrival
        fresh = replay([long_], [0.0], grid, rover, suns, dt, 2.0, work)
        assert fresh["arrival"][0] > first["arrival"][0] * 1.5, (
            fresh["arrival"][0], first["arrival"][0])
        # the directory now holds the long plan's work. Asking for the short
        # plan with reuse must NOT return it.
        back = replay([short], [0.0], grid, rover, suns, dt, 2.0, work, reuse=True)
        assert back["arrival"][0] == pytest.approx(first["arrival"][0]), (
            "reuse returned another call's outputs")
        assert back["arrival"][0] != pytest.approx(fresh["arrival"][0])


def test_reuse_refuses_a_call_killed_while_writing():
    """A short output file is the signature of a killed call, and reading one
    back is a wrong answer rather than a crash."""
    import tempfile
    from pathlib import Path
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        grid = scene(td, flat())
        suns, dt = sun_table(5.0)
        rover = Rover(v_flat=50.0)
        work = td / "run"
        plan = straight_plan(30, 10, 20)
        first = replay([plan], [0.0], grid, rover, suns, dt, 2.0, work)
        f = work / "out_step_pitch.bin"
        f.write_bytes(f.read_bytes()[:-8])
        again = replay([plan], [0.0], grid, rover, suns, dt, 2.0, work, reuse=True)
        assert np.array_equal(first["arrival"], again["arrival"])
        assert f.stat().st_size == len(f.read_bytes()), "it was rewritten"
        assert again["step_pitch"].size == int(first["step_offsets"][-1])


def true_along_x(x, y):
    """True horizontal distance from map x = 0 to x along a line of constant y.

    The stereographic scale is k = 1 + (x^2 + y^2) / 4R^2, so the true length
    is the integral of dx / k, which is (4R^2 / c) atan(x / c) with
    c^2 = 4R^2 + y^2. Exact, and written from the projection, not from the
    checker.
    """
    c = np.sqrt(4.0 * MOON_RADIUS_M ** 2 + y ** 2)
    return 4.0 * MOON_RADIUS_M ** 2 / c * np.arctan(x / c)


def inclined(theta_deg, ox=4000.0, oy=11000.0):
    """A surface rising at theta per TRUE metre along map +x, a row plan's direction.

    Rising per map metre would not do: away from the pole a map metre is
    shorter than a true one, and the slope would be atan(k tan theta).
    """
    x = ox - (NEAR - N) / 2 * PX + (np.arange(NEAR) + 0.5) * PX
    y = oy + (NEAR - N) / 2 * PX - (np.arange(NEAR) + 0.5) * PX
    s = true_along_x(x[None, :], y[:, None])
    return np.tan(np.radians(theta_deg)) * (s - s[:, :1])


@pytest.mark.parametrize("theta_deg, ox", [(10.0, 4000.0), (-10.0, 4000.0),
                                           (15.0, 4000.0), (10.0, 300000.0)])
def test_time_on_a_slope_is_distance_along_the_slope_over_speed(tmp_path, theta_deg, ox):
    """Protects the checker's time integration against stepping the map distance.

    The speed model gives speed along the slope, and the planner's edge time is
    the 3D length along the slope over that speed. Until 2026-11-22 the checker
    divided the map-plane distance by that speed, so every step was short by
    cos(pitch) and every checker time ran about 1% early on this terrain, which
    was read for seven weeks as the planner being optimistic. On a plane the
    answer is analytic: the horizontal distance over cos(theta), over the speed
    at pitch theta and zero roll. Uphill and downhill both, because downhill
    runs at the speed cap and the error is in the length, not the speed.

    The map scale is checked too, which the region itself cannot do: there it
    is at most 1.3e-5, below this tolerance and below the disc fit's own
    scatter of about 8e-5, so a scale multiplied where it should divide passed.
    300 km from the pole the scale is 0.75%, well above both.
    """
    rover = Rover(v_flat=50.0)
    th = np.radians(theta_deg)
    rel_tol = 2e-4
    # The fixture can only catch the fault if cos(theta) is far from one
    # against the tolerance; a shallow plane would pass the broken code.
    assert 1.0 - np.cos(th) > 20 * rel_tol
    grid = scene(tmp_path, inclined(theta_deg, ox=ox), ox=ox)
    suns, dt = sun_table(30.0, n=64)
    c0, c1, row = 10, 30, 30
    plan = straight_plan(row, c0, c1)
    r = replay([plan], [0.0], grid, rover, suns, dt, 1e9, tmp_path)
    # the analytic traverse: centre to centre along the row, in true metres
    xa = grid["region_ox"] + (c0 + 0.5) * PX
    xb = grid["region_ox"] + (c1 + 0.5) * PX
    y = grid["region_oy"] - (row + 0.5) * PX
    horizontal = true_along_x(xb, y) - true_along_x(xa, y)
    # the far case must make the scale visible against the tolerance
    assert ox < 1e5 or (xb - xa) / horizontal - 1.0 > 20 * rel_tol
    drive = np.sin(th) + rover.mu * np.cos(th)
    v = min(rover.v_flat * rover.mu / drive if drive > 0 else np.inf, rover.v_cap)
    expected = horizontal / np.cos(th) / v
    assert r["illegal"][0] == 0
    assert np.median(r["step_pitch_mean"]) == pytest.approx(theta_deg, abs=0.05)
    assert r["arrival"][0] == pytest.approx(expected, rel=rel_tol), (
        r["arrival"][0], expected, r["arrival"][0] / expected, np.cos(th))
