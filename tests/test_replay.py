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


def scene(tmp_path, height):
    """A synthetic near raster and the geometry describing it.

    The reach is cut to 400 m and the far field is pushed past it, so only the
    near raster is needed and a test does not have to synthesise 238 km of DEM.
    """
    ox, oy = 4000.0, 11000.0
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
