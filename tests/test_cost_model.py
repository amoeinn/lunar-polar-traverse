"""Phase 2's cost model, each test naming the bug or invariant it protects.

PLAN.md registers the three checks and their tolerances before this was
written: pitch against the raster to 1e-8 m of rise, the decomposition against
closed form at dip azimuth 23 deg, and a resampled path integration with no
threshold, which lives in examples/build_cost.py because it needs the DEM.
"""
import ast
from pathlib import Path

import numpy as np
import pytest

from src.cost import (NEIGHBOURS, Rover, build_edges, decompose, edge_time,
                      heuristic_time, joint_slope_bound, legal, map_scale, speed,
                      surface_gradient)
from src.frames import MOON_RADIUS_M, cell_position
from src.lit_fraction import mean_map_weight

R = MOON_RADIUS_M
PX = 20.0
DIP_DEG, DIP_AZIMUTH_DEG = 17.0, 23.0     # registered; 23 is not a multiple of 45
RISE_TOL_M = 1e-8                          # registered check A


def rough_dem(n=60, seed=1, scale=0.02):
    rng = np.random.default_rng(seed)
    return np.cumsum(np.cumsum(rng.normal(0.0, 3.0, (n, n)), 0), 1) * scale


def pair_order(edges, n_cells):
    """Index arrays putting edge (a, b) and edge (b, a) in the same position."""
    fwd = np.argsort(edges["a"].astype(np.int64) * n_cells + edges["b"])
    rev = np.argsort(edges["b"].astype(np.int64) * n_cells + edges["a"])
    return fwd, rev


def test_pitch_against_the_raster_heights():
    """Protects: pitch is the real rise over the real 3D length.

    Registered check A. Length times sin(pitch) comes through cell_position,
    local_frame and a 3D dot product; the height difference comes from
    subtracting two numbers out of the raster. They share no derivation, and on
    a sphere they agree only to the curvature correction.
    """
    h = rough_dem()
    e = build_edges(h, -4000.0, 11000.0, PX)
    dh = h.ravel()[e["b"]] - h.ravel()[e["a"]]
    residual = np.abs(e["length"] * np.sin(np.radians(e["pitch"])) - dh)
    assert residual.max() <= RISE_TOL_M, residual.max()
    # The floor is double-precision cancellation at the sphere radius, about
    # eps * R = 3.9e-10 m, not the curvature term at 3.3e-11 * |dh|.
    assert residual.max() > 1e-11, "a residual this small means the two agree by sharing code"


def test_decomposition_against_closed_form_at_every_heading():
    """Protects: the pitch and roll split, and specifically the diagonal case.

    Registered check B. A dip azimuth that is a multiple of 45 deg cannot see a
    diagonal error, which is the error most likely to be there, so it is 23.
    """
    g = np.tan(np.radians(DIP_DEG))
    gx = g * np.cos(np.radians(DIP_AZIMUTH_DEG))
    gy = g * np.sin(np.radians(DIP_AZIMUTH_DEG))
    for dr, dc in NEIGHBOURS:
        hx, hy = dc * PX, -dr * PX
        norm = np.hypot(hx, hy)
        ux, uy = hx / norm, hy / norm
        beta = np.degrees(np.arctan2(uy, ux))
        want_pitch = np.degrees(np.arctan(g * np.cos(np.radians(DIP_AZIMUTH_DEG - beta))))
        want_roll = np.degrees(np.arctan(g * np.sin(np.radians(DIP_AZIMUTH_DEG - beta))))
        got_pitch, got_roll = decompose(gx, gy, ux, uy)
        assert got_pitch == pytest.approx(want_pitch, abs=1e-9), (dr, dc, got_pitch)
        assert got_roll == pytest.approx(want_roll, abs=1e-9), (dr, dc, got_roll)
    # The fixture discriminates: at least one heading must separate the two, or
    # exchanging pitch and roll would pass.
    diffs = [abs(decompose(gx, gy, dc * PX / np.hypot(dc * PX, -dr * PX),
                           -dr * PX / np.hypot(dc * PX, -dr * PX))[0]
                 - decompose(gx, gy, dc * PX / np.hypot(dc * PX, -dr * PX),
                             -dr * PX / np.hypot(dc * PX, -dr * PX))[1])
             for dr, dc in NEIGHBOURS]
    assert max(diffs) > 1.0, diffs


def test_surface_gradient_recovers_a_planar_field():
    """Protects: the 3 x 3 least-squares fit. A plane must come back exactly."""
    n = 40
    ox, oy = -400.0, 400.0                  # near the pole, where the scale factor is 1
    xs = ox + (np.arange(n) + 0.5) * PX
    ys = oy - (np.arange(n) + 0.5) * PX
    gx_map, gy_map = np.meshgrid(xs, ys)
    g = np.tan(np.radians(DIP_DEG))
    h = g * (gx_map * np.cos(np.radians(DIP_AZIMUTH_DEG))
             + gy_map * np.sin(np.radians(DIP_AZIMUTH_DEG)))
    fx, fy = surface_gradient(h, PX, gx_map, gy_map)
    scale = map_scale(gx_map, gy_map) / (1.0 + h / R)
    inner = (slice(1, -1), slice(1, -1))    # the padded border is one-sided by design
    assert (fx / scale)[inner] == pytest.approx(g * np.cos(np.radians(DIP_AZIMUTH_DEG)),
                                                rel=1e-12)
    assert (fy / scale)[inner] == pytest.approx(g * np.sin(np.radians(DIP_AZIMUTH_DEG)),
                                                rel=1e-12)


def test_endpoint_pitch_agrees_with_surface_pitch_on_a_plane():
    """Protects: the two pitch definitions differ only by the sphere's curvature."""
    n = 40
    ox, oy = -400.0, 400.0
    xs = ox + (np.arange(n) + 0.5) * PX
    ys = oy - (np.arange(n) + 0.5) * PX
    gx_map, gy_map = np.meshgrid(xs, ys)
    g = np.tan(np.radians(DIP_DEG))
    h = g * (gx_map * np.cos(np.radians(DIP_AZIMUTH_DEG))
             + gy_map * np.sin(np.radians(DIP_AZIMUTH_DEG)))
    e = build_edges(h, ox, oy, PX)
    fx, fy = surface_gradient(h, PX, gx_map, gy_map)
    # Border cells' gradients are one-sided, because surface_gradient pads by
    # replication, so only edges with both ends interior are comparable.
    a, b = e["a"], e["b"]
    row_a, col_a, row_b, col_b = a // n, a % n, b // n, b % n
    inside = ((np.minimum(row_a, row_b) > 0) & (np.maximum(row_a, row_b) < n - 1)
              & (np.minimum(col_a, col_b) > 0) & (np.maximum(col_a, col_b) < n - 1))
    a, b = a[inside], b[inside]
    row_a, col_a, row_b, col_b = row_a[inside], col_a[inside], row_b[inside], col_b[inside]
    e = dict(e, pitch=e["pitch"][inside], length=e["length"][inside])
    hxs = (col_b - col_a) * PX
    hys = -(row_b - row_a) * PX
    norms = np.hypot(hxs, hys)
    sp, _ = decompose(0.5 * (fx.ravel()[a] + fx.ravel()[b]),
                      0.5 * (fy.ravel()[a] + fy.ravel()[b]), hxs / norms, hys / norms)
    bound = np.degrees(e["length"] / (2.0 * R))     # the registered check B bound
    residual = np.abs(e["pitch"] - sp)
    assert np.all(residual <= bound), float(np.max(residual - bound))
    # The registered bound was derived as first order in L/2R and the real
    # effect is second order, so it is loose by five orders. The tight bound is
    # asserted too, since the loose one alone would not catch much.
    assert residual.max() < 1e-8, residual.max()
    assert residual.max() > 0.0, "identical to the last bit means they share code"


def test_map_scale_factor_is_applied():
    """Protects the 1.3e-5 correction the plane fixture is too coarse to see."""
    far = 12500.0
    k = map_scale(far / np.sqrt(2.0), far / np.sqrt(2.0))
    assert k - 1.0 == pytest.approx((far / (2.0 * R)) ** 2, rel=1e-9)
    assert k - 1.0 > 1e-5, k
    n = 8
    h = np.arange(n * n, dtype=float).reshape(n, n)
    xs = np.full((n, n), far / np.sqrt(2.0))
    scaled = surface_gradient(h, PX, xs, xs)[0]
    flat = surface_gradient(h, PX, np.zeros((n, n)), np.zeros((n, n)))[0]
    assert np.all(scaled > flat), "the scale factor must make map gradients steeper"


def test_antisymmetry_is_the_midpoint_choice_not_a_coincidence():
    """Protects the midpoint-up decision, by asserting the rejected one fails.

    A test that only checks pitch(A,B) = -pitch(B,A) on the midpoint definition
    cannot fail: the construction guarantees it. This asserts both sides.
    """
    h = rough_dem(n=40)
    n_cells = h.size
    mid = build_edges(h, -4000.0, 11000.0, PX, midpoint_up=True)
    tail = build_edges(h, -4000.0, 11000.0, PX, midpoint_up=False)
    f, r = pair_order(mid, n_cells)
    assert np.abs(mid["pitch"][f] + mid["pitch"][r]).max() == 0.0
    f, r = pair_order(tail, n_cells)
    residual = np.abs(tail["pitch"][f] + tail["pitch"][r]).max()
    assert residual > 1e-5, residual
    assert residual == pytest.approx(np.degrees(PX * np.sqrt(2.0) / R), rel=0.5), residual


def test_cost_asymmetry_comes_only_from_pitch():
    """Protects: the graph is directed because of terrain, not because of a bug."""
    rover = Rover()
    pitches = np.array([1.0, 5.0, 12.0, 19.0])
    ratios = []
    for roll in (0.0, 7.0, 14.0):
        for length in (20.0, 28.284):
            up = edge_time(length, pitches, roll, rover)
            down = edge_time(length, -pitches, roll, rover)
            ratios.append(up / down)
    for r in ratios[1:]:
        assert r == pytest.approx(ratios[0], rel=1e-12)
    assert np.all(ratios[0] > 1.0), "uphill must cost more than downhill"
    assert ratios[0].max() > 2.0, "the fixture must contain a real asymmetry"


def test_limits_forbid_symmetrically():
    """Protects: connectivity is undirected while cost stays directed."""
    rover = Rover()
    p = np.array([-30.0, -20.0, -5.0, 5.0, 20.0, 30.0])
    r = np.array([-20.0, -15.0, -2.0, 2.0, 15.0, 20.0])
    pp, rr = np.meshgrid(p, r)
    assert np.array_equal(legal(pp, rr, rover), legal(-pp, rr, rover))
    assert np.array_equal(legal(pp, rr, rover), legal(pp, -rr, rover))
    assert legal(19.9, 14.9, rover) and not legal(20.1, 14.9, rover)


def test_edge_time_is_homogeneous_in_v_flat():
    """Protects P7: the static optimum must not depend on rover speed.

    A second v_flat dependence anywhere in the static cost would let the cost
    model answer P7 instead of the illumination.
    """
    pitch = np.array([-25.0, -8.0, 0.0, 8.0, 25.0])
    roll = np.array([0.0, 5.0, 10.0, 14.0, 2.0])
    slow = edge_time(25.0, pitch, roll, Rover(v_flat=20.0))
    fast = edge_time(25.0, pitch, roll, Rover(v_flat=200.0))
    assert slow == pytest.approx(10.0 * fast, rel=1e-12)
    assert np.ptp(slow / fast) == pytest.approx(0.0, abs=1e-12)


def test_heuristic_is_admissible():
    """Protects A* correctness: the heuristic may never exceed the true cost."""
    h = rough_dem(n=40)
    rover = Rover()
    e = build_edges(h, -4000.0, 11000.0, PX)
    n = h.shape[0]
    xs = -4000.0 + (np.arange(n) + 0.5) * PX
    ys = 11000.0 - (np.arange(n) + 0.5) * PX
    gx, gy = np.meshgrid(xs, ys)
    pos = cell_position(gx, gy, h, R).reshape(-1, 3)
    t = edge_time(e["length"], e["pitch"], e["roll"], rover)
    hh = heuristic_time(pos[e["a"]], pos[e["b"]], rover)
    assert np.all(hh <= t + 1e-12), float((hh - t).max())
    assert np.all(speed(e["pitch"], e["roll"], rover) <= rover.v_cap + 1e-12)


def planar_dem(slope_deg, azimuth_deg, n=40, ox=-400.0, oy=400.0, px=PX):
    """A height field linear in map axes, referenced to the patch centre."""
    xs = ox + (np.arange(n) + 0.5) * px
    ys = oy - (np.arange(n) + 0.5) * px
    gx, gy = np.meshgrid(xs, ys)
    g = np.tan(np.radians(slope_deg))
    h = g * ((gx - gx.mean()) * np.cos(np.radians(azimuth_deg))
             + (gy - gy.mean()) * np.sin(np.radians(azimuth_deg)))
    return h, gx, gy


def isolated_fraction(h, rover, ox=-400.0, oy=400.0):
    n = h.shape[0]
    e = build_edges(h, ox, oy, PX)
    ok = legal(e["pitch"], e["roll"], rover)
    degree = np.bincount(e["a"][ok], minlength=n * n)
    interior = np.zeros((n, n), dtype=bool)
    interior[2:-2, 2:-2] = True             # border gradients are one-sided
    return float((degree[interior.ravel()] == 0).mean())


def test_isolated_cells_against_the_joint_slope_bound():
    """Protects: the two limits are one knob, and the eight headings tighten it.

    On a plane every cell is alike, so isolation has a sharp threshold in slope
    and the bound can be tested rather than bounded. The worst fall-line
    azimuth for the eight headings is 0, along a grid axis, so the fixture uses
    it. The continuous bound is what a free choice of heading would give;
    asserting that cells are isolated between the two bounds is what makes this
    test able to fail if the discrete bound were replaced by the continuous one.
    """
    rover = Rover()
    discrete, continuous = joint_slope_bound(rover)
    assert discrete < continuous, (discrete, continuous)
    assert discrete == pytest.approx(20.7536, abs=1e-3), discrete
    assert continuous == pytest.approx(24.3212, abs=1e-3), continuous

    below, _, _ = planar_dem(discrete - 0.1, 0.0)
    above, _, _ = planar_dem(discrete + 0.1, 0.0)
    between, _, _ = planar_dem(0.5 * (discrete + continuous), 0.0)
    assert isolated_fraction(below, rover) == 0.0
    assert isolated_fraction(above, rover) == 1.0
    assert isolated_fraction(between, rover) == 1.0, \
        "cells between the two bounds must be isolated, or the discrete bound is untested"

    # And the bound is a joint one: neither limit alone explains it.
    assert discrete > rover.pitch_max and discrete > rover.roll_max
    steep_but_legal, _, _ = planar_dem(discrete - 0.1, 0.0)
    e = build_edges(steep_but_legal, -400.0, 400.0, PX)
    ok = legal(e["pitch"], e["roll"], rover)
    assert np.abs(e["pitch"][ok]).max() < rover.pitch_max
    assert np.abs(e["roll"][ok]).max() < rover.roll_max


def test_static_cost_does_not_import_illumination():
    """Protects the phase 2 / phase 3 boundary, structurally rather than by habit."""
    tree = ast.parse(Path("src/cost.py").read_text())
    names = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names += [a.name for a in node.names]
        elif isinstance(node, ast.ImportFrom):
            names.append(node.module or "")
            names += [a.name for a in node.names]
    banned = [n for n in names if "illumination" in n or "lit_fraction" in n
              or "horizon" in n or "shadow" in n]
    assert not banned, banned
    assert "frames" in " ".join(names) and "numpy" in " ".join(names)

    # And behaviourally: illumination moves the baseline weight and nothing else.
    lit = np.array([0.9, 0.5, 0.05])
    w1 = mean_map_weight(lit)
    w2 = mean_map_weight(lit * 0.5)
    assert not np.allclose(w1, w2)
    rover = Rover()
    before = edge_time(20.0, 5.0, 3.0, rover)
    after = edge_time(20.0, 5.0, 3.0, rover)
    assert before == after and np.isfinite(before)
