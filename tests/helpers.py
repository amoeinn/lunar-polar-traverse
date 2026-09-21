"""Scaffolding for the horizon tests: synthetic scenes on the real grid.

The synthetic DEMs are built on the same south polar stereographic grid the
LOLA products use, at the same pixel sizes, so the tests exercise the real
coordinate path and not a flat toy grid.
"""
import json
import os
import subprocess
from pathlib import Path

import numpy as np

from src.frames import MOON_RADIUS_M, local_frame, map_to_unit, unit_to_map

ROOT = Path(__file__).resolve().parents[1]
BINARY = Path(os.environ.get("HORIZON_BIN", ROOT / "cpp" / "build" / "horizon"))


def offset_point(x0, y0, azimuth_deg, arc_m, radius=MOON_RADIUS_M):
    """Map coordinates of the point at a given azimuth and arc distance."""
    up, east, north = local_frame(x0, y0, radius)
    a = np.radians(azimuth_deg)
    t = np.cos(a) * north + np.sin(a) * east
    psi = arc_m / radius
    d = np.cos(psi) * up + np.sin(psi) * t
    return unit_to_map(d, radius)


def arc_and_azimuth(x0, y0, x1, y1, radius=MOON_RADIUS_M):
    """Arc distance and initial azimuth from one map point to another, in the
    first point's local frame."""
    up, east, north = local_frame(x0, y0, radius)
    d = map_to_unit(x1, y1, radius)
    cos_psi = np.clip(np.sum(d * up, axis=-1), -1.0, 1.0)
    arc = radius * np.arccos(cos_psi)
    tangent = d - cos_psi[..., None] * up
    nrm = np.linalg.norm(tangent, axis=-1, keepdims=True)
    tangent = np.divide(tangent, nrm, out=np.zeros_like(tangent), where=nrm > 0)
    az = np.degrees(np.arctan2(np.sum(tangent * east, axis=-1),
                               np.sum(tangent * north, axis=-1))) % 360.0
    return arc, az


def elevation_exact(h_target, arc_m, h_obs=0.0, radius=MOON_RADIUS_M, observer_height=0.0):
    """Exact elevation angle of a point of height h_target at arc distance arc_m.

    Written as the arcsine of a dot product between explicit 3D positions, a
    different expression from the atan2(A, B) form the C++ uses, so agreement
    between them is evidence rather than a tautology.
    """
    psi = np.asarray(arc_m, dtype=float) / radius
    r_t = radius + h_target
    # Vertical separation written as a difference of heights minus the exact
    # curvature drop, since cos(psi)*r_t - r_obs loses ten digits at 20 m.
    d_up = (h_target - h_obs - observer_height) - 2.0 * r_t * np.sin(0.5 * psi) ** 2
    d_horiz = np.sin(psi) * r_t
    return np.degrees(np.arcsin(d_up / np.hypot(d_horiz, d_up)))


def _grid_spec(x_lo, x_hi, y_lo, y_hi, px):
    """Smallest grid of pixel size px whose cells cover the requested bounds."""
    ox = np.floor(x_lo / px) * px
    oy = np.ceil(y_hi / px) * px
    width = int(np.ceil((x_hi - ox) / px))
    height = int(np.ceil((oy - y_lo) / px))
    return ox, oy, width, height


def _write_grid(path, height_fn, ox, oy, width, height, px):
    xs = ox + (np.arange(width) + 0.5) * px
    ys = oy - (np.arange(height) + 0.5) * px
    X, Y = np.meshgrid(xs, ys)
    Z = np.asarray(height_fn(X, Y), dtype=np.float32)
    assert Z.shape == (height, width)
    Z.tofile(path)
    return Z


class Result:
    def __init__(self, horizon, dist, sidecar, params, stdout, scene):
        self.horizon = horizon          # (ny, nx, n_az) degrees
        self.dist = dist                # (ny, nx, n_az) metres of arc
        self.sidecar = sidecar
        self.params = params
        self.stdout = stdout
        self.scene = scene

    def cell_xy(self, row, col):
        p = self.params
        return (p["region_ox"] + (col + 0.5) * p["region_px"],
                p["region_oy"] - (row + 0.5) * p["region_px"])


def run_horizon(tmp_path, height_fn, *, region_ox, region_oy, region_nx, region_ny,
                reach_m, region_px=20.0, switch_m=20000.0, near_step_m=20.0,
                far_step_m=80.0, near_px=20.0, far_px=80.0, observer_height_m=0.0,
                n_azimuth=360, margin_m=2000.0, radius_m=MOON_RADIUS_M):
    """Build a synthetic scene, run the real binary on it, and read the output."""
    tmp_path = Path(tmp_path)
    x_lo, x_hi = region_ox, region_ox + region_nx * region_px
    y_hi, y_lo = region_oy, region_oy - region_ny * region_px

    near_reach = min(switch_m, reach_m) + margin_m
    n_ox, n_oy, n_w, n_h = _grid_spec(x_lo - near_reach, x_hi + near_reach,
                                      y_lo - near_reach, y_hi + near_reach, near_px)
    f_ox, f_oy, f_w, f_h = _grid_spec(x_lo - reach_m - margin_m, x_hi + reach_m + margin_m,
                                      y_lo - reach_m - margin_m, y_hi + reach_m + margin_m, far_px)

    near_path = tmp_path / "near.f32"
    far_path = tmp_path / "far.f32"
    _write_grid(near_path, height_fn, n_ox, n_oy, n_w, n_h, near_px)
    _write_grid(far_path, height_fn, f_ox, f_oy, f_w, f_h, far_px)

    params = {
        "radius_m": radius_m, "observer_height_m": observer_height_m,
        "reach_m": reach_m, "switch_m": switch_m,
        "near_step_m": near_step_m, "far_step_m": far_step_m, "n_azimuth": n_azimuth,
        "region_ox": region_ox, "region_oy": region_oy,
        "region_nx": region_nx, "region_ny": region_ny, "region_px": region_px,
        "near_path": str(near_path), "near_ox": n_ox, "near_oy": n_oy,
        "near_width": n_w, "near_height": n_h, "near_px": near_px,
        "far_path": str(far_path), "far_ox": f_ox, "far_oy": f_oy,
        "far_width": f_w, "far_height": f_h, "far_px": far_px,
        "out_horizon": str(tmp_path / "horizon.f32"),
        "out_horizon_dist": str(tmp_path / "horizon_dist.f32"),
        "out_sidecar": str(tmp_path / "horizon.json"),
    }
    params_path = tmp_path / "params.json"
    params_path.write_text(json.dumps(params, indent=2))

    proc = subprocess.run([str(BINARY), str(params_path)], capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"horizon failed ({proc.returncode}):\n{proc.stdout}\n{proc.stderr}")

    shape = (region_ny, region_nx, n_azimuth)
    horizon = np.fromfile(params["out_horizon"], dtype=np.float32).reshape(shape)
    dist = np.fromfile(params["out_horizon_dist"], dtype=np.float32).reshape(shape)
    sidecar = json.loads(Path(params["out_sidecar"]).read_text())
    return Result(horizon, dist, sidecar, params, proc.stdout, height_fn)


def plateau(cx, cy, half_width, height):
    """A square plateau in map coordinates: exactly `height` inside, 0 outside.

    A square block rather than a single raised cell, so that bilinear sampling
    anywhere in its interior returns exactly `height` and the analytic
    elevation formula applies without an interpolation correction.
    """
    def fn(X, Y):
        return np.where((np.abs(X - cx) <= half_width) & (np.abs(Y - cy) <= half_width),
                        height, 0.0)
    return fn


def add(*fns):
    def fn(X, Y):
        total = np.zeros_like(X, dtype=float)
        for f in fns:
            total = total + f(X, Y)
        return total
    return fn


def circular_mean_deg(angles_deg, weights=None):
    a = np.radians(np.asarray(angles_deg, dtype=float))
    if weights is None:
        weights = np.ones_like(a)
    return np.degrees(np.arctan2(np.sum(weights * np.sin(a)),
                                 np.sum(weights * np.cos(a)))) % 360.0


def angular_diff_deg(a, b):
    return np.abs((np.asarray(a) - np.asarray(b) + 180.0) % 360.0 - 180.0)
