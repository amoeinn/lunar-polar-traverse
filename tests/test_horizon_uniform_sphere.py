"""Test (a): uniform sphere, no relief.

Protects the curvature handling and the exact 3D elevation geometry against
a flat-plane substitute. On a sphere where every surface point sits at the
same radius, the elevation angle from one point to another separated by
central angle psi is exactly -psi/2, because the chord makes equal angles
with the two radii. A flat-plane model returns exactly 0 instead. The two
differ by only 3.3e-4 deg at the first 20 m step, so the tolerance has to be
tighter than that or the test passes for the broken model too; that is
asserted here rather than assumed.
"""
import numpy as np
import pytest

from tests.helpers import elevation_exact, run_horizon
from src.frames import MOON_RADIUS_M

CONST_HEIGHT = 1234.5      # non-zero, so a dropped or doubled height shows up
TOL_DEG = 1e-6


@pytest.fixture(scope="module")
def result(tmp_path_factory):
    return run_horizon(
        tmp_path_factory.mktemp("uniform"),
        lambda X, Y: np.full_like(X, CONST_HEIGHT, dtype=float),
        region_ox=-4000.0, region_oy=11000.0, region_nx=8, region_ny=8,
        reach_m=5000.0, switch_m=20000.0, margin_m=1500.0,
    )


def test_fixture_discriminates_flat_from_curved():
    """The fixture must be able to fail. A tolerance looser than the
    curvature signal would accept a flat-plane model."""
    analytic = float(elevation_exact(CONST_HEIGHT, 20.0, h_obs=CONST_HEIGHT))
    flat_plane_answer = 0.0
    assert analytic < 0.0, "a uniform sphere must put its own surface below the horizontal"
    assert abs(flat_plane_answer - analytic) > TOL_DEG, (
        "tolerance is looser than the curvature signal, so the flat-plane "
        "model would pass this test")
    # the closed form the analytic value must equal
    assert analytic == pytest.approx(-np.degrees(20.0 / MOON_RADIUS_M) / 2.0, rel=1e-12)


def test_horizon_equals_minus_half_the_first_step_angle(result):
    analytic = float(elevation_exact(CONST_HEIGHT, 20.0, h_obs=CONST_HEIGHT))
    assert np.max(np.abs(result.horizon - analytic)) < TOL_DEG, (
        f"max deviation {np.max(np.abs(result.horizon - analytic)):.3e} deg "
        f"from the analytic {analytic:.6e} deg")


def test_profile_is_isotropic(result):
    """A uniform sphere cannot prefer an azimuth. Any spread across the 360
    bins would be the azimuth code injecting asymmetry."""
    spread = np.max(result.horizon, axis=-1) - np.min(result.horizon, axis=-1)
    assert np.max(spread) < 1e-9, f"max spread across azimuth {np.max(spread):.3e} deg"


def test_nearest_sample_sets_the_horizon(result):
    """-psi/2 decreases with distance, so the first step must win. This is
    what makes horizon_dist meaningful and it is asserted, not assumed."""
    assert np.all(result.dist == np.float32(20.0))


def test_sidecar_records_the_convention(result):
    s = result.sidecar
    assert "local north" in s["azimuth_convention"]
    assert s["dem_sampling"] == "bilinear, near and far"
    assert s["ray_path"] == "great circle, advanced incrementally in 3D"
    assert s["samples_outside_grid"] == 0
    assert s["n_steps_per_ray"] == 250          # 5000 m of reach at 20 m steps
