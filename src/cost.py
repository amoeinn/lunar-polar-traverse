"""Static traversal cost: the directed 8-connected graph over the planning region.

An edge's cost is a time, because every prediction phase 2 feeds is about
arrival times. It is a function of pitch, roll, length and the rover parameters
and of nothing else: anything needing a clock belongs to phase 3, and the one
illumination-derived quantity phase 2 owes the mean-map baseline lives in
src/lit_fraction.py so that boundary is structural rather than a convention.

See PLAN.md, "Cost model", for the registered form, the swept and fixed
parameters, and the three checks this is held to.
"""
from dataclasses import dataclass

import numpy as np

from .frames import MOON_RADIUS_M, cell_position, local_frame

# Row, column offsets of the eight neighbours.
NEIGHBOURS = ((-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1))
HEADING_QUANTISATION_DEG = 22.5      # half the 45 deg spacing of the eight headings


@dataclass(frozen=True)
class Rover:
    """Rover parameters. v_flat is swept; the rest are fixed with a sensitivity pass."""
    v_flat: float = 50.0             # level-ground net progress, m/hr
    mu: float = 0.2                  # effective rolling resistance
    cap_ratio: float = 1.5           # descent speed cap, as a multiple of v_flat
    roll_ref: float = 15.0           # roll at which the derate halves the speed, deg
    roll_exponent: float = 2.0
    pitch_max: float = 20.0          # limits, on magnitudes, so the forbidden set
    roll_max: float = 15.0           # is symmetric and connectivity is undirected

    @property
    def v_cap(self):
        return self.cap_ratio * self.v_flat


def map_scale(x, y, radius=MOON_RADIUS_M):
    """Polar stereographic scale: map distance is this times true distance.

    rho = 2R tan(colat/2), so drho / (R dcolat) = sec^2(colat/2) = 1 + (rho/2R)^2
    to second order. Over this region it reaches 1.3e-5, which matters to nothing
    here except that leaving it out would be an error nobody could see.
    """
    return 1.0 + (np.hypot(x, y) / (2.0 * radius)) ** 2


def surface_gradient(height, px, x, y, radius=MOON_RADIUS_M):
    """True surface gradient in map axes, from a 3 x 3 least-squares plane fit.

    The fit reduces to central differences averaged over three rows, which is
    why it appears as a column sum over 6 px. Edges are padded by replication,
    so a border cell's gradient is one-sided in effect; border cells carry no
    edges of their own in build_edges and this only affects their neighbours.
    """
    height = np.asarray(height, dtype=float)
    h = np.pad(height, 1, mode="edge")
    dzdx = (h[:-2, 2:] + h[1:-1, 2:] + h[2:, 2:]
            - h[:-2, :-2] - h[1:-1, :-2] - h[2:, :-2]) / (6.0 * px)
    dzdy = (h[:-2, :-2] + h[:-2, 1:-1] + h[:-2, 2:]
            - h[2:, :-2] - h[2:, 1:-1] - h[2:, 2:]) / (6.0 * px)
    # Two corrections to the map spacing, both small and both real. The
    # stereographic scale factor makes map distance exceed true distance; and a
    # cell at height h sits on a sphere of radius R + h, so the ground distance
    # between adjacent cells is px (1 + h/R), not px. The second reaches 1.4e-3
    # over this DEM's +/- 2.5 km of relief, twenty times the first, and leaving
    # it out puts the surface pitch 0.02 deg away from the endpoint pitch on a
    # plane where they must agree. The endpoint-pitch test found it.
    scale = map_scale(x, y, radius) / (1.0 + height / radius)
    return dzdx * scale, dzdy * scale


def decompose(gradient_x, gradient_y, heading_x, heading_y):
    """Surface pitch and roll along a heading, both in degrees.

    The heading is a unit vector in map axes. The projection is conformal, so a
    direction in map axes is a true direction and needs no correction; only the
    gradient's magnitude carries the scale factor, applied in surface_gradient.
    Cross-heading is the heading rotated 90 degrees, which is the other grid
    axis for an axis edge and the other diagonal for a diagonal one. There is no
    special case, which is the whole reason roll is defined this way.
    """
    cross_x, cross_y = -heading_y, heading_x
    pitch = np.degrees(np.arctan(gradient_x * heading_x + gradient_y * heading_y))
    roll = np.degrees(np.arctan(gradient_x * cross_x + gradient_y * cross_y))
    return pitch, roll


def build_edges(height, ox, oy, px, radius=MOON_RADIUS_M, midpoint_up=True):
    """Every directed 8-connected edge of the region, with length, pitch and roll.

    midpoint_up=False selects the rejected definition, pitch measured against the
    up at the tail. It exists so a test can assert that the rejected one is not
    antisymmetric, which is the only way the antisymmetry test can fail.
    """
    height = np.asarray(height, dtype=float)
    ny, nx = height.shape
    xs = ox + (np.arange(nx) + 0.5) * px
    ys = oy - (np.arange(ny) + 0.5) * px
    gx, gy = np.meshgrid(xs, ys)
    pos = cell_position(gx, gy, height, radius).reshape(-1, 3)
    up = local_frame(gx, gy, radius)[0].reshape(-1, 3)
    grad_x, grad_y = surface_gradient(height, px, gx, gy, radius)
    grad_x, grad_y = grad_x.ravel(), grad_y.ravel()
    index = np.arange(ny * nx).reshape(ny, nx)

    starts, ends, length, pitch, roll = [], [], [], [], []
    for dr, dc in NEIGHBOURS:
        a = index[max(0, -dr):ny - max(0, dr), max(0, -dc):nx - max(0, dc)].ravel()
        b = index[max(0, dr):ny - max(0, -dr), max(0, dc):nx - max(0, -dc)].ravel()
        d = pos[b] - pos[a]
        ell = np.linalg.norm(d, axis=1)
        if midpoint_up:
            u = up[a] + up[b]
            u /= np.linalg.norm(u, axis=1, keepdims=True)
        else:
            u = up[a]
        starts.append(a)
        ends.append(b)
        length.append(ell)
        pitch.append(np.degrees(np.arcsin(np.sum(d / ell[:, None] * u, axis=1))))
        hx, hy = dc * px, -dr * px          # map +y is up the array
        norm = np.hypot(hx, hy)
        roll.append(decompose(0.5 * (grad_x[a] + grad_x[b]), 0.5 * (grad_y[a] + grad_y[b]),
                              hx / norm, hy / norm)[1])
    return dict(a=np.concatenate(starts), b=np.concatenate(ends),
                length=np.concatenate(length), pitch=np.concatenate(pitch),
                roll=np.concatenate(roll), shape=(ny, nx))


def speed(pitch_deg, roll_deg, rover):
    """Speed on a slope, in the units of rover.v_flat.

    Proportional to v_flat by construction, so every edge time scales as
    1/v_flat and the statically optimal path does not depend on rover speed.
    PLAN.md registers that as a requirement: P7 asks whether the planner's
    advantage falls with rover speed, and a second v_flat dependence here would
    answer that question with the cost model instead of with the illumination.
    """
    theta = np.radians(np.asarray(pitch_deg, dtype=float))
    drive = np.sin(theta) + rover.mu * np.cos(theta)
    with np.errstate(divide="ignore", invalid="ignore"):
        v = np.where(drive > 0.0, rover.v_flat * rover.mu / drive, np.inf)
    v = np.minimum(v, rover.v_cap)
    derate = 1.0 + (np.abs(np.asarray(roll_deg, dtype=float)) / rover.roll_ref) ** rover.roll_exponent
    return v / derate


def edge_time(length, pitch_deg, roll_deg, rover):
    """Traversal time for a directed edge, in the hours implied by v_flat."""
    return np.asarray(length, dtype=float) / speed(pitch_deg, roll_deg, rover)


def legal(pitch_deg, roll_deg, rover):
    """Edges within both limits. Limits are on magnitudes, so this is symmetric."""
    return ((np.abs(np.asarray(pitch_deg, dtype=float)) <= rover.pitch_max)
            & (np.abs(np.asarray(roll_deg, dtype=float)) <= rover.roll_max))


def joint_slope_bound(rover, n_alpha=901):
    """Slope below which no cell can be isolated, and the continuous-heading bound.

    On a locally planar surface tan(pitch) and tan(roll) are the components of
    the gradient along and across the heading, so pitch and roll are not
    independent knobs: with a free choice of heading a cell has a legal one iff
    tan(slope)^2 <= tan(pitch_max)^2 + tan(roll_max)^2, which is the second
    value returned.

    With only eight headings the choice is not free, and the achievable bound
    depends on where the fall line sits relative to the grid. It is computed
    here rather than approximated: the first value is the largest slope for
    which every fall-line azimuth still has at least one legal heading, found by
    bisection over a scan of azimuths. Approximating it by widening the
    continuous bound with cos(22.5 deg) was tried and is wrong in the unsafe
    direction, giving 16.66 deg where the true value at (15, 10) is 15.00.
    """
    tp, tr = np.tan(np.radians(rover.pitch_max)), np.tan(np.radians(rover.roll_max))
    offsets = np.radians(np.arange(8) * 45.0)[:, None] - np.radians(
        np.linspace(0.0, 45.0, n_alpha))[None, :]
    ca, sa = np.abs(np.cos(offsets)), np.abs(np.sin(offsets))

    def every_azimuth_has_a_heading(slope_deg):
        ts = np.tan(np.radians(slope_deg))
        return bool(((ts * ca <= tp) & (ts * sa <= tr)).any(axis=0).all())

    lo, hi = 0.0, 89.0
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if every_azimuth_has_a_heading(mid):
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi), float(np.degrees(np.arctan(np.hypot(tp, tr))))


def heuristic_time(position_a, position_b, rover):
    """Admissible A* heuristic: straight-line 3D distance over the top speed.

    No edge can be faster than v_cap, since the derate only slows, and the 3D
    straight line is no longer than any path between the two points.
    """
    d = np.linalg.norm(np.asarray(position_b, float) - np.asarray(position_a, float), axis=-1)
    return d / rover.v_cap
