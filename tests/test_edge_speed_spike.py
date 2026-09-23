"""P1's estimator against a synthetic caster whose sweep speed is analytic.

The invariant: the level-set speed |dg/dt| / |grad_x g|, evaluated by central
differences at 20 m cell spacing, recovers the kinematic sweep speed of a
shadow edge rotating about its caster, and the shadow it measures ends at the
spherical root of elev(H, D) = E rather than at the flat-earth root. A model
without curvature puts the tip 2.9 km further out and its tip speed 18% high,
so the fixture discriminates against exactly the error it exists to catch.

Registered in PLAN.md: H = 500 m, sun at 1.5 deg, exact root 16.203 km, tip
speed 143.7 m/hr at 0.00887 rad/hr, tolerance 5%, flat-earth 169.4 m/hr.

The caster here is a cone of height H and base radius 3 km, not a
mathematical spike. A spike is unusable twice over: its shadow is narrower
than one cell, and its silhouette makes the horizon discontinuous in azimuth,
so a central difference straddles a jump instead of measuring a gradient. A
cone's horizon varies smoothly with azimuth and its apex still sets the
shadow's end, so the registered root and tip speed are unchanged: the tip lies
at the root of elev(H, D) = E from the apex, exactly as registered.

The tip speed is not measured at the tip. There the shadow pinches to a point
and the level set's normal is radial, so its normal speed is the radial term
and not the sweep. It is measured as the law it follows: the estimator returns
d * omega at every distance, so the tip sweeps at (tip distance) * omega,
which is how the registered 143.7 was derived in the first place.
"""
import numpy as np
import pytest

from src.edges import (caster_relief, flag_degenerate, kinematic_speed_bound,
                       level_set_speed)
from src.frames import (MOON_RADIUS_M, azimuth_elevation_from_frame, cell_position,
                        local_frame)

R = MOON_RADIUS_M
H = 500.0                 # caster height above the plain
R_BASE = 3000.0           # cone base radius; a spike cannot be differenced
E_DEG = 1.5               # sun elevation, held constant
OMEGA = 0.00887           # rad/hr, phase 0's measured azimuth rate
AU = 1.495978707e11
SPACING = 20.0
REGISTERED_TIP_SPEED = 143.7
REGISTERED_ROOT = 16203.0
FLAT_ROOT = 19094.0
TOL = 0.05


def elev_sphere(h, arc):
    """Elevation angle of a point h above the surface at arc distance arc."""
    psi = arc / R
    a = h - (R + h) * (1.0 - np.cos(psi))
    b = (R + h) * np.sin(psi)
    return np.degrees(np.arctan2(a, b))


def elev_flat(h, arc):
    return np.degrees(np.arctan2(h, arc))


def shadow_root(elev_fn, e_deg=E_DEG):
    """Arc distance where the caster top drops to elevation e_deg."""
    lo, hi = 1.0, 200000.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if elev_fn(H, mid) > e_deg:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def horizon_at(elev_fn, d, bearing_deg, az_deg):
    """Analytic horizon of the cone at one azimuth, from a cell d away.

    The ray at az_deg reaches its closest approach to the axis at offset
    d sin(delta) and arc distance d cos(delta); the cone's height there is
    H (1 - |offset| / R_BASE), falling to zero at the base, so the horizon is
    continuous in azimuth everywhere. In-plane geometry is planar, which at
    16 km on a 1737 km sphere is a 9e-5 relative error; the elevation angle
    itself is spherical, which is the part the fixture is about.
    """
    delta = np.radians((np.asarray(az_deg, float) - bearing_deg + 180.0) % 360.0 - 180.0)
    off = np.abs(d * np.sin(delta))
    h = H * np.maximum(0.0, 1.0 - off / R_BASE)
    s = np.maximum(d * np.cos(delta), 1.0)
    return elev_fn(h, s)


def scene(d_axis):
    """A cell d_axis from the caster axis, its four axis neighbours, and the sun.

    The caster sits at the pole, the cell due grid-north of it. Positions,
    frames and the (azimuth, elevation) decomposition all come from src.frames,
    so the fixture exercises the production frame code and not a copy of it.
    """
    pts = [(0.0, d_axis), (SPACING, d_axis), (-SPACING, d_axis),
           (0.0, d_axis + SPACING), (0.0, d_axis - SPACING)]
    x = np.array([p[0] for p in pts])
    y = np.array([p[1] for p in pts])
    up, east, north = local_frame(x, y, R)
    origin = cell_position(x, y, np.zeros_like(x), R)
    d = np.hypot(x, y)                      # each cell's own distance to the axis
    # The caster is at the south pole, and local north points away from it, so
    # the bearing to it is exactly 180 deg in every cell's own local frame. It
    # is NOT atan2 in map axes: the map frame turns with longitude, which over
    # 20 m here is 0.143 deg, the same size as the bearing change being measured.
    bearing = np.full(5, 180.0)
    return origin, up, east, north, d, bearing


def g_of_time(elev_fn, scn, az_ref):
    """g at the five cells for a sun whose azimuth at the centre cell is az_ref."""
    origin, up, east, north, d, bearing = scn
    sdir = (np.cos(np.radians(E_DEG)) * (np.cos(np.radians(az_ref)) * north[0]
                                         + np.sin(np.radians(az_ref)) * east[0])
            + np.sin(np.radians(E_DEG)) * up[0])
    sun = origin[0] + AU * sdir
    az, el = azimuth_elevation_from_frame(sun, origin, up, east, north)
    hor = np.array([horizon_at(elev_fn, d[i], bearing[i], az[i]) for i in range(5)])
    return el - hor, az


def measure(elev_fn, d_axis):
    """Bisect to the first-light transition and apply the estimator there."""
    scn = scene(d_axis)
    bearing0 = scn[5][0]
    span = np.degrees(np.arcsin(min(R_BASE / d_axis, 0.999))) + 1.0
    hi = bearing0 + span
    if g_of_time(elev_fn, scn, hi)[0][0] <= 0.0:
        return None
    if g_of_time(elev_fn, scn, bearing0)[0][0] > 0.0:
        return None                                # never shadowed: past the tip
    a, b = bearing0, hi                            # the trailing edge: dark -> lit
    for _ in range(80):
        m = 0.5 * (a + b)
        if g_of_time(elev_fn, scn, m)[0][0] <= 0.0:
            a = m
        else:
            b = m
    az_star = 0.5 * (a + b)
    daz = np.degrees(OMEGA) * 1e-4                 # 0.36 ms of sun rotation
    gp = g_of_time(elev_fn, scn, az_star + daz)[0][0]
    gm = g_of_time(elev_fn, scn, az_star - daz)[0][0]
    dg_dt = (gp - gm) / (2.0 * 1e-4)               # per hour
    g5, _ = g_of_time(elev_fn, scn, az_star)
    return float(level_set_speed(dg_dt, g5[1], g5[2], g5[3], g5[4], SPACING))


def find_tip(elev_fn):
    lo, hi = 5000.0, 30000.0
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if measure(elev_fn, mid) is not None:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def test_root_is_spherical_not_flat():
    """The registered roots are what this fixture is built on; check them first."""
    assert shadow_root(elev_sphere) == pytest.approx(REGISTERED_ROOT, abs=2.0)
    assert shadow_root(elev_flat) == pytest.approx(FLAT_ROOT, abs=2.0)


def analytic_normal_speed(d, delta_deg):
    """The level set's normal speed from the caster field itself.

    The shadow of an axisymmetric caster rotates rigidly about it, so a
    boundary point at distance d moves tangentially at d * omega and the normal
    speed is that times the cosine between the tangential direction and the
    boundary normal: omega |dg/dtheta| / |grad g| in polar coordinates about the
    caster. The partials are taken on the analytic horizon function in (d,
    theta), which shares no code with the estimator: no frames, no map
    coordinates, no 20 m central differences. The sun's elevation is held
    constant here, which is the one approximation and is worth about 1%.

    Only for a point caster is the boundary radial and the cosine 1, giving the
    d * omega of the registered tip speed; the cone's boundary is tilted, so at
    delta = 12 deg the normal speed is 7% below d * omega. That is the honest
    value for this caster and the test asserts it rather than the idealisation.
    """
    def h(dd, dl):
        off = abs(dd * np.sin(np.radians(dl)))
        return elev_sphere(H * max(0.0, 1.0 - off / R_BASE),
                           max(dd * np.cos(np.radians(dl)), 1.0))
    ed, ea = 1e-3, 1e-6
    g_d = -(h(d + ed, delta_deg) - h(d - ed, delta_deg)) / (2.0 * ed)
    g_t = -(h(d, delta_deg + np.degrees(ea)) - h(d, delta_deg - np.degrees(ea))) / (2.0 * ea)
    return OMEGA * abs(g_t) / np.hypot(g_d, g_t / d)


def edge_offset(elev_fn, d_axis):
    """Azimuth offset of the trailing shadow edge at this distance."""
    scn = scene(d_axis)
    span = np.degrees(np.arcsin(min(R_BASE / d_axis, 0.999))) + 1.0
    a, b = 180.0, 180.0 + span
    for _ in range(80):
        m = 0.5 * (a + b)
        if g_of_time(elev_fn, scn, m)[0][0] <= 0.0:
            a = m
        else:
            b = m
    return 0.5 * (a + b) - 180.0


def test_level_set_speed_formula():
    """Guards the arithmetic: on a plane boundary moving at a known speed."""
    # g = a x + b y - c t, so the boundary moves at c / hypot(a, b).
    a, b, c, s = 3.0, 4.0, 2.5, 20.0
    got = level_set_speed(-c, a * s, -a * s, b * s, -b * s, s)
    assert got == pytest.approx(c / np.hypot(a, b), rel=1e-12)
    assert np.isinf(level_set_speed(1.0, 0.0, 0.0, 0.0, 0.0, s))


def test_estimator_matches_the_analytic_normal_speed():
    """Guards the estimator: map-space differences must give the polar answer."""
    for d in (8000.0, 12000.0, 15000.0):
        got = measure(elev_sphere, d)
        want = analytic_normal_speed(d, edge_offset(elev_sphere, d))
        assert got == pytest.approx(want, rel=0.02), f"at {d} m: {got} vs {want}"
        assert got < d * OMEGA, "the cone's boundary is tilted, so it must be below"
        assert got > 0.9 * d * OMEGA, "but not far below"


def test_tip_speed_matches_the_registered_value():
    """Guards curvature in the shadow length: flat earth puts the tip 18% fast."""
    tip = find_tip(elev_sphere)
    assert tip == pytest.approx(REGISTERED_ROOT, abs=40.0), tip
    speed = tip * OMEGA
    assert abs(speed - REGISTERED_TIP_SPEED) / REGISTERED_TIP_SPEED < TOL, speed


def test_the_fixture_rejects_a_model_without_curvature():
    """The discriminating half: the same test on flat-earth geometry must fail."""
    tip = find_tip(elev_flat)
    assert tip == pytest.approx(FLAT_ROOT, abs=40.0), tip
    speed = tip * OMEGA
    assert abs(speed - REGISTERED_TIP_SPEED) / REGISTERED_TIP_SPEED > TOL, speed


def test_caster_relief_inverts_the_elevation_formula():
    """Guards the relief inversion P1 filters on: flat earth is 7% low at 16 km."""
    for arc in (500.0, 5000.0, 16203.0, 100000.0):
        theta = elev_sphere(H, arc)
        assert caster_relief(theta, arc, 0.0, R) == pytest.approx(H, rel=1e-9)
    theta = elev_sphere(H, 16203.0)
    flat = 16203.0 * np.tan(np.radians(theta))
    assert flat < 0.95 * H, flat


def test_degenerate_edges_are_flagged():
    """Guards the degeneracy screen: a near-parallel edge must not be scored.

    The estimator returns |dg/dt| / |grad g|, which stops meaning anything as
    the gradient goes to zero. The bound comes from the geometry: a shadow is
    at most as long as the horizon reach, so its tip sweeps at most reach times
    the azimuth rate. The fixture discriminates because it contains both cases
    and asserts each side, and because the shrinking-gradient sequence crosses
    the bound rather than sitting on one side of it.
    """
    bound = kinematic_speed_bound(238000.0, 0.5080)
    assert bound == pytest.approx(2110.6, abs=0.5), bound

    # A real edge: the cone at 15 km, which the tests above pin to the physics.
    real = measure(elev_sphere, 15000.0)
    assert not flag_degenerate(real, bound)

    # The same dg/dt with the spatial gradient shrinking toward zero. The four
    # neighbour values are what a boundary lying along the sampling direction
    # produces: the cell moves in time while its neighbours barely differ.
    dg_dt = 0.5

    def at_speed(target):
        """Neighbour values whose central difference gives exactly this speed."""
        e = abs(dg_dt) * SPACING / target
        return float(level_set_speed(dg_dt, e, -e, 0.0, 0.0, SPACING))

    # Straddle the bound closely, so the test pins where the threshold is and
    # not merely that huge speeds are caught.
    below, above = at_speed(0.98 * bound), at_speed(1.02 * bound)
    assert below == pytest.approx(0.98 * bound, rel=1e-9)
    assert not flag_degenerate(below, bound)
    assert flag_degenerate(above, bound)

    speeds = [below, above, float(level_set_speed(dg_dt, 1e-6, -1e-6, 0.0, 0.0, SPACING)),
              float(level_set_speed(dg_dt, 0.0, 0.0, 0.0, 0.0, SPACING)),
              float(level_set_speed(0.0, 0.0, 0.0, 0.0, 0.0, SPACING))]
    assert not np.isfinite(speeds[3]), "an exactly zero gradient must be inf"
    assert np.isnan(speeds[4]), "zero over zero must be nan, not a speed"
    assert list(flag_degenerate(speeds, bound)) == [False, True, True, True, True]
    assert flag_degenerate(np.inf, bound), "inf is degenerate, not merely large"
    assert flag_degenerate(np.nan, bound), "nan is degenerate, not merely unknown"
