"""Frames and lookups shared by the illumination model and its checks.

Azimuth is defined once, here and in cpp/src/horizon.cpp, and the two must
agree: from local north, positive toward local east, measured in each cell's
own local tangent frame, wrapped to [0, 360). tests/test_shadow_azimuth.py is
the only test that sees both definitions at once, and it exists for that.
"""
import numpy as np

MOON_RADIUS_M = 1737400.0
Z_AXIS = np.array([0.0, 0.0, 1.0])


def map_to_unit(x, y, radius=MOON_RADIUS_M):
    """Moon-fixed unit vector of a south polar stereographic map point.

    Colatitude from the south pole is 2 atan(r / 2R); this is the exact
    inverse of the projection on a sphere, not a small-angle form.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    r = np.hypot(x, y)
    t = r / (2.0 * radius)
    den = 1.0 + t * t
    sin_theta = 2.0 * t / den
    cos_theta = (1.0 - t * t) / den
    safe = np.where(r > 0.0, r, 1.0)
    sin_lam = np.where(r > 0.0, x / safe, 0.0)
    cos_lam = np.where(r > 0.0, y / safe, 1.0)
    return np.stack([sin_theta * sin_lam, sin_theta * cos_lam, -cos_theta], axis=-1)


def unit_to_map(d, radius=MOON_RADIUS_M):
    """Map coordinates of a Moon-fixed unit vector. The stereographic identity
    x = 2R dx / (1 - dz), which needs no trigonometry."""
    d = np.asarray(d, dtype=float)
    inv = 2.0 * radius / (1.0 - d[..., 2])
    return d[..., 0] * inv, d[..., 1] * inv


def local_frame(x, y, radius=MOON_RADIUS_M):
    """(up, east, north) at a map point, in Moon-fixed coordinates.

    east is normalize(spin axis x up), which is the standard local east; north
    is up x east. Undefined exactly at the pole, where up is parallel to the
    spin axis; the planning region excludes it.
    """
    up = map_to_unit(x, y, radius)
    east = np.cross(np.broadcast_to(Z_AXIS, up.shape), up)
    norm = np.linalg.norm(east, axis=-1, keepdims=True)
    if np.any(norm == 0.0):
        raise ValueError("local frame is singular at the pole")
    east = east / norm
    north = np.cross(up, east)
    return up, east, north


def cell_position(x, y, height, radius=MOON_RADIUS_M, observer_height=0.0):
    """Moon-fixed position of an observer over a map point."""
    up = map_to_unit(x, y, radius)
    scale = np.asarray(radius + np.asarray(height, dtype=float) + observer_height, dtype=float)
    return up * scale[..., None]


def azimuth_elevation(target, x, y, height, radius=MOON_RADIUS_M, observer_height=0.0):
    """Azimuth and elevation, in degrees, of a Moon-fixed point seen from a cell.

    `target` is a Moon-fixed position, not a direction; the vector from the
    observer to it is what gets decomposed, so the sun's finite distance is
    handled rather than assumed away.
    """
    up, east, north = local_frame(x, y, radius)
    origin = cell_position(x, y, height, radius, observer_height)
    d = np.asarray(target, dtype=float) - origin
    d = d / np.linalg.norm(d, axis=-1, keepdims=True)
    d_east = np.sum(d * east, axis=-1)
    d_north = np.sum(d * north, axis=-1)
    d_up = np.sum(d * up, axis=-1)
    azimuth = np.degrees(np.arctan2(d_east, d_north)) % 360.0
    elevation = np.degrees(np.arcsin(np.clip(d_up, -1.0, 1.0)))
    return azimuth, elevation


def horizon_lookup(profile, azimuth_deg):
    """Horizon elevation at an arbitrary azimuth.

    Rays are cast at integer azimuths, so a lookup between them is linearly
    interpolated. That is what makes the sun-visibility margin continuous in
    time, so a bisection on it refines a true root and not a step.
    """
    profile = np.asarray(profile, dtype=float)
    n = profile.shape[-1]
    a = np.mod(np.asarray(azimuth_deg, dtype=float), 360.0) * (n / 360.0)
    i0 = np.floor(a).astype(int)
    frac = a - i0
    i0 = i0 % n
    i1 = (i0 + 1) % n
    # Each cell is indexed into its own profile. np.take would form the outer
    # product of every cell with every other cell's azimuth instead.
    lead = np.broadcast_shapes(profile.shape[:-1], a.shape)
    prof = np.broadcast_to(profile, lead + (n,))
    v0 = np.take_along_axis(prof, np.broadcast_to(i0, lead)[..., None], axis=-1)[..., 0]
    v1 = np.take_along_axis(prof, np.broadcast_to(i1, lead)[..., None], axis=-1)[..., 0]
    return (1.0 - np.broadcast_to(frac, lead)) * v0 + np.broadcast_to(frac, lead) * v1
