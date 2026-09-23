"""Shadow-edge speed: the level-set normal speed of the sun-visibility function.

g(x, t) is the sun's elevation at cell x minus that cell's horizon at the sun's
azimuth, in degrees. A cell is lit where g > 0, so the shadow edge is the zero
level set of g and its normal speed is |dg/dt| / |grad_x g|, evaluated at the
transition. That is the quantity P1 is about: how fast a shadow edge sweeps
across the ground, in metres per hour.

Nothing here knows how g was produced. The C++ lit-interval scan supplies g at
the cell and its four axis neighbours at the transition time; this applies the
estimator to them, and tests/test_edge_speed_spike.py validates it against a
synthetic caster whose sweep speed is analytic.
"""
import numpy as np


def level_set_speed(dg_dt, g_east, g_west, g_north, g_south, spacing):
    """|dg/dt| / |grad_x g| from central differences over the four axis neighbours.

    dg_dt is in degrees per hour and spacing in metres, so the result is metres
    per hour. A zero gradient means the level set is not locally a moving curve
    and the speed is undefined; it is returned as inf rather than silently
    clipped, so the caller has to decide what that means.
    """
    gx = (np.asarray(g_east, float) - np.asarray(g_west, float)) / (2.0 * spacing)
    gy = (np.asarray(g_north, float) - np.asarray(g_south, float)) / (2.0 * spacing)
    grad = np.hypot(gx, gy)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.abs(np.asarray(dg_dt, float)) / grad


def kinematic_speed_bound(reach_m, azimuth_rate_deg_per_hr):
    """The fastest a shadow edge can sweep, from the horizon reach.

    A shadow is at most as long as the horizon reach, because terrain beyond it
    is not traced and cannot cast; a shadow of length L rotating about its
    caster sweeps its tip at L times the sun's azimuth rate. So the reach and
    the rate together bound the speed of anything that is a shadow edge.

    This is the tangential sweep only. The radial term from dE/dt is not in it,
    so the bound is a screen rather than a strict physical maximum, and the
    population it catches is dominated by marginal cases: on the 2028 run the
    flagged edges exceed it by a median of 1.4 times, though the worst is 297.
    Its exact value therefore decides the flagged count, and decides almost
    nothing else: the headline median moves 1.4% whether the flagged edges are
    counted or dropped.
    """
    return float(reach_m) * np.radians(float(azimuth_rate_deg_per_hr))


def flag_degenerate(speed, bound):
    """True where the estimator has left the regime it is valid in.

    |dg/dt| / |grad_x g| is the boundary's normal speed only where the boundary
    is locally a moving curve. Where |grad_x g| approaches zero, because the
    shadow edge runs nearly parallel to the local grazing geometry, the ratio
    grows without limit and stops describing anything. Exceeding the kinematic
    bound is the symptom; a near-zero gradient is the cause. Non-finite speeds,
    which is an exactly zero gradient, are degenerate by the same argument.
    """
    s = np.asarray(speed, float)
    return ~np.isfinite(s) | (s > bound)


def caster_relief(horizon_deg, arc_distance_m, cell_height_m, radius_m):
    """Height of the horizon-setting terrain above the cell, on the sphere.

    The inverse of the elevation-angle formula the ray march uses:
    tan(theta) = [(h - h0) - (R+h)(1 - cos psi)] / [(R+h) sin psi], psi = s/R.
    Solved for h exactly rather than by the flat-earth h = s tan(theta), which
    at 16 km and 1.5 degrees is 7% low and at 100 km is a factor of four.
    """
    psi = np.asarray(arc_distance_m, float) / radius_m
    t = np.tan(np.radians(np.asarray(horizon_deg, float)))
    c, s = 1.0 - np.cos(psi), np.sin(psi)
    denom = 1.0 - c - t * s
    h = (radius_m * (t * s + c) + np.asarray(cell_height_m, float)) / denom
    return h - np.asarray(cell_height_m, float)
