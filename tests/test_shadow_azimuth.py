"""Test (c): end to end through the production path, both languages at once.

A Moon-fixed sun vector goes through the Python (azimuth, elevation) function,
that azimuth is looked up in the C++ binary's horizon output by the production
interpolation, and the resulting shadow must fall on the far side of the spike
from the sun, ending at the analytic tip distance.

This is the only test that sees both languages' azimuth conventions at once,
which is the whole point of it. Tests (a) and (b) would pass unchanged if the
C++ and the Python agreed with each other but were both wrong, or if only one
of them were wrong in a way the other never looks at.

The fixture is asserted to discriminate before anything is measured. A_s is
not a multiple of 45 deg, because a sign flip on east maps azimuth A to -A and
a north/east swap maps it to 90 - A, and at A_s = 45 both leave the shadow
where it was. The separations those two mutations would produce are checked
against the tolerance explicitly.
"""
import numpy as np
import pytest
from scipy.optimize import brentq

from tests.helpers import (angular_diff_deg, arc_and_azimuth, elevation_exact,
                           offset_point, plateau, run_horizon)
from src.frames import MOON_RADIUS_M, azimuth_elevation, cell_position, horizon_lookup, local_frame

SPIKE_X, SPIKE_Y = 1000.0, 6000.0
SPIKE_H, SPIKE_HW = 50.0, 60.0
SUN_AZ, SUN_EL = 112.7, 5.0            # not a multiple of 45; see the module docstring
SUN_RANGE_M = 1.5e11
CROSS_TRACK_TOL_M = 100.0          # plateau half-width plus two cells
NEAR_SPIKE_EXCLUSION_M = 150.0

NX = NY = 80
REGION_PX = 20.0
REGION_OX = SPIKE_X - NX / 2 * REGION_PX
REGION_OY = SPIKE_Y + NY / 2 * REGION_PX


def sun_position():
    """A Moon-fixed sun position placed to give SUN_AZ and SUN_EL at the spike."""
    up, east, north = local_frame(SPIKE_X, SPIKE_Y)
    a, e = np.radians(SUN_AZ), np.radians(SUN_EL)
    direction = np.cos(e) * (np.cos(a) * north + np.sin(a) * east) + np.sin(e) * up
    return cell_position(SPIKE_X, SPIKE_Y, 0.0) + direction * SUN_RANGE_M


def analytic_tip_arc():
    """Arc distance from the plateau centre at which the shadow ends.

    A cell at arc D behind the plateau sees the plateau's near top edge at
    arc D - half_width; the shadow ends where that drops to the sun's
    elevation. Solved on the exact geometry, not on the parabolic form.
    """
    f = lambda d: float(elevation_exact(SPIKE_H, d, h_obs=0.0)) - SUN_EL
    edge = brentq(f, 1.0, 50000.0)
    return edge + SPIKE_HW


@pytest.fixture(scope="module")
def result(tmp_path_factory):
    return run_horizon(
        tmp_path_factory.mktemp("shadow"),
        plateau(SPIKE_X, SPIKE_Y, SPIKE_HW, SPIKE_H),
        region_ox=REGION_OX, region_oy=REGION_OY, region_nx=NX, region_ny=NY,
        region_px=REGION_PX, reach_m=5000.0, switch_m=20000.0, margin_m=1500.0,
    )


@pytest.fixture(scope="module")
def shadow(result):
    """The shadow mask, built the way the model will build it."""
    rows, cols = np.mgrid[0:NY, 0:NX]
    xs = REGION_OX + (cols + 0.5) * REGION_PX
    ys = REGION_OY - (rows + 0.5) * REGION_PX
    heights = result.scene(xs, ys)
    az, el = azimuth_elevation(sun_position(), xs, ys, heights)
    horizon = horizon_lookup(result.horizon, az)
    return xs, ys, horizon > el, az, el


def test_fixture_discriminates_the_named_mutations():
    """Asserted before anything is measured: at A_s = 45 a sign flip on east
    and a north/east swap would both be invisible."""
    assert SUN_AZ % 45.0 != 0.0, "A_s must not be a multiple of 45 deg"
    expected = (SUN_AZ + 180.0) % 360.0
    east_negated = (-SUN_AZ + 180.0) % 360.0          # azimuth A -> -A
    axes_swapped = (90.0 - SUN_AZ + 180.0) % 360.0    # azimuth A -> 90 - A
    # The direction test bounds the MAXIMUM cross-track over shadowed cells, so
    # the margin is set by the far end of the shadow, at the tip distance.
    reach = min(analytic_tip_arc(), NX / 2 * REGION_PX)
    for name, mutated in (("east sign flip", east_negated), ("north/east swap", axes_swapped)):
        cross = reach * np.sin(np.radians(angular_diff_deg(expected, mutated)))
        assert cross > 3 * CROSS_TRACK_TOL_M, (
            f"a {name} would move the shadow only {cross:.0f} m off the line, "
            f"inside the {CROSS_TRACK_TOL_M:.0f} m tolerance, so the test could not fail")


def test_python_recovers_the_sun_angles_it_was_given(result):
    az, el = azimuth_elevation(sun_position(), SPIKE_X, SPIKE_Y, 0.0)
    assert float(az) == pytest.approx(SUN_AZ, abs=1e-9)
    assert float(el) == pytest.approx(SUN_EL, abs=1e-9)


def test_the_scene_casts_a_shadow_that_is_neither_empty_nor_everything(shadow):
    _, _, mask, _, _ = shadow
    assert mask.any(), "no cell is shadowed, so the test cannot fail"
    assert not mask.all(), "every cell is shadowed, so the test cannot fail"
    assert 0.01 < mask.mean() < 0.5, f"shadow covers {mask.mean():.3%} of the region"


def test_shadow_falls_on_the_far_side_from_the_sun(shadow):
    xs, ys, mask, _, _ = shadow
    arc, az = arc_and_azimuth(SPIKE_X, SPIKE_Y, xs, ys)
    far = mask & (arc > NEAR_SPIKE_EXCLUSION_M)
    assert far.sum() > 20, "too few shadowed cells outside the plateau to judge direction"
    expected = (SUN_AZ + 180.0) % 360.0
    # The caster is a square of half-width SPIKE_HW, so its shadow is a band of
    # that width, not a line: cells near the spike are legitimately off-axis in
    # angle. Cross-track distance from the anti-solar great circle is the
    # quantity that is actually bounded.
    cross = arc * np.sin(np.radians(angular_diff_deg(az, expected)))
    worst = float(np.max(cross[far]))
    assert worst < CROSS_TRACK_TOL_M, (
        f"a shadowed cell sits {worst:.1f} m off the anti-solar line through the "
        f"spike (azimuth {expected:.2f} deg); the shadow is in the wrong place")


def test_shadow_ends_at_the_analytic_tip(shadow):
    xs, ys, mask, _, _ = shadow
    arc, _ = arc_and_azimuth(SPIKE_X, SPIKE_Y, xs, ys)
    tip = analytic_tip_arc()
    measured = float(np.max(arc[mask]))
    assert measured == pytest.approx(tip, abs=2.0 * REGION_PX + SPIKE_HW), (
        f"shadow reaches {measured:.1f} m, exact geometry gives {tip:.1f} m")
    # Curvature is not what this test discriminates: at H=50 m and a 5 deg sun
    # the flat-earth tip differs by about a metre. Tests (a) and (b) carry the
    # curvature assertions; this one carries the azimuth convention.


def test_cells_toward_the_sun_are_lit(shadow):
    xs, ys, mask, _, _ = shadow
    arc, az = arc_and_azimuth(SPIKE_X, SPIKE_Y, xs, ys)
    sunward = (angular_diff_deg(az, SUN_AZ) < 30.0) & (arc > NEAR_SPIKE_EXCLUSION_M)
    assert sunward.sum() > 20
    assert not mask[sunward].any(), "cells between the sun and the spike are shadowed"
