"""Test (b): a single spike of known height, once inside and once beyond the
20 km near/far switch.

Protects the azimuth binning, the along-ray distance, the height handling and
the near/far DEM switch. The near spike is sampled from the 20 m grid and the
far spike from the 80 m grid, so a switch that fires at the wrong distance, or
a far grid that is never consulted, changes the answer.

The elevation assertion is analytic: whatever distance the binary reports as
setting the horizon, the reported angle must equal the exact spherical
elevation of a point of that height at that distance. The bin assertion is the
circular mean of the elevated bins, which moves by exactly one degree if the
bin index is off by one.
"""
import numpy as np
import pytest

from tests.helpers import (add, angular_diff_deg, circular_mean_deg, elevation_exact,
                           offset_point, plateau, run_horizon)

CELL_X, CELL_Y = -4000.0, 11000.0

NEAR_AZ, NEAR_ARC, NEAR_H, NEAR_HW = 30.0, 5000.0, 500.0, 400.0
FAR_AZ, FAR_ARC, FAR_H, FAR_HW = 200.0, 25000.0, 1500.0, 1200.0

SWITCH = 20000.0
FLAT_HORIZON = float(elevation_exact(0.0, 20.0, h_obs=0.0))


@pytest.fixture(scope="module")
def result(tmp_path_factory):
    nx, ny = offset_point(CELL_X + 10.0, CELL_Y - 10.0, NEAR_AZ, NEAR_ARC)
    fx, fy = offset_point(CELL_X + 10.0, CELL_Y - 10.0, FAR_AZ, FAR_ARC)
    scene = add(plateau(float(nx), float(ny), NEAR_HW, NEAR_H),
                plateau(float(fx), float(fy), FAR_HW, FAR_H))
    res = run_horizon(
        tmp_path_factory.mktemp("spike"), scene,
        region_ox=CELL_X, region_oy=CELL_Y, region_nx=1, region_ny=1,
        reach_m=30000.0, switch_m=SWITCH, margin_m=2000.0,
    )
    res.near_xy = (float(nx), float(ny))
    res.far_xy = (float(fx), float(fy))
    return res


def test_fixture_straddles_the_switch():
    """The point of two spikes is that one is served by each grid. If both
    fell on the same side, the test could not see the switch at all."""
    assert NEAR_ARC < SWITCH < FAR_ARC
    assert angular_diff_deg(NEAR_AZ, FAR_AZ) > 90.0, "spikes must not share bins"


def _clusters(profile, threshold=0.1):
    elevated = np.where(profile > threshold)[0]
    assert elevated.size > 0, "no elevated bins: the scene has no spike in it"
    gaps = np.where(np.diff(elevated) > 1)[0]
    return np.split(elevated, gaps + 1)


def test_two_distinct_clusters_of_elevated_bins(result):
    groups = _clusters(result.horizon[0, 0])
    assert len(groups) == 2, f"expected two spikes, found {len(groups)} clusters"


@pytest.mark.parametrize("which,az,arc,h,hw",
                         [("near", NEAR_AZ, NEAR_ARC, NEAR_H, NEAR_HW),
                          ("far", FAR_AZ, FAR_ARC, FAR_H, FAR_HW)])
def test_spike_lands_in_the_right_bins(result, which, az, arc, h, hw):
    profile = result.horizon[0, 0]
    group = min(_clusters(profile),
                key=lambda g: angular_diff_deg(circular_mean_deg(g), az))
    centre = circular_mean_deg(group)
    assert angular_diff_deg(centre, az) < 0.5, (
        f"{which} spike: elevated bins centre on {centre:.3f} deg, expected {az:.3f}; "
        f"a one-bin index error would show as exactly 1.0 deg")


@pytest.mark.parametrize("which,az,arc,h,hw",
                         [("near", NEAR_AZ, NEAR_ARC, NEAR_H, NEAR_HW),
                          ("far", FAR_AZ, FAR_ARC, FAR_H, FAR_HW)])
def test_spike_elevation_is_the_analytic_value(result, which, az, arc, h, hw):
    profile = result.horizon[0, 0]
    dist = result.dist[0, 0]
    group = min(_clusters(profile),
                key=lambda g: angular_diff_deg(circular_mean_deg(g), az))
    peak = group[np.argmax(profile[group])]
    d = float(dist[peak])
    # The plateau is a square in map coordinates, so a ray can enter it as much
    # as hw*sqrt(2) before the centre, at a corner.
    assert abs(d - arc) < hw * np.sqrt(2.0) + 100.0, (
        f"{which} spike: horizon set at {d:.1f} m, plateau centre at {arc:.0f} m, "
        f"half-width {hw:.0f} m")

    # Reconstruct the point the binary says set the horizon, independently, from
    # the azimuth and distance it reported, and check the scene is at full
    # plateau height there. If it were on the one-pixel bilinear ramp the
    # analytic elevation below would not apply.
    cx, cy = result.cell_xy(0, 0)
    px_, py_ = offset_point(cx, cy, float(peak), d)
    h_at = float(result.scene(np.asarray(px_), np.asarray(py_)))
    assert h_at == pytest.approx(h, abs=1e-9), (
        f"{which} spike: the reported horizon point is at height {h_at:.3f}, not on "
        f"the plateau top at {h:.1f}; the ray geometry disagrees between languages")

    expected = float(elevation_exact(h, d, h_obs=0.0))
    assert profile[peak] == pytest.approx(expected, abs=1e-4), (
        f"{which} spike: {profile[peak]:.6f} deg reported, exact geometry gives "
        f"{expected:.6f} deg at {d:.1f} m")
    flat_earth = np.degrees(np.arctan(h / d))
    assert abs(flat_earth - expected) > 1e-3, (
        "fixture does not discriminate curvature at this distance")


def test_far_spike_is_served_by_the_far_grid(result):
    profile = result.horizon[0, 0]
    group = min(_clusters(profile),
                key=lambda g: angular_diff_deg(circular_mean_deg(g), FAR_AZ))
    peak = group[np.argmax(profile[group])]
    d = float(result.dist[0, 0][peak])
    assert d > SWITCH
    offset = (d - SWITCH) / 80.0
    assert abs(offset - round(offset)) < 1e-6, (
        f"{d} m is not on the 80 m far-field schedule, so the far grid was not used")


def test_bins_away_from_both_spikes_are_the_flat_value(result):
    profile = result.horizon[0, 0]
    bins = np.arange(360.0)
    away = (angular_diff_deg(bins, NEAR_AZ) > 20.0) & (angular_diff_deg(bins, FAR_AZ) > 20.0)
    assert np.max(np.abs(profile[away] - FLAT_HORIZON)) < 1e-6, (
        "bins with no spike in them must equal the uniform-sphere horizon")
