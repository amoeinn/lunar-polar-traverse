"""Landmarks: our map projection against the IAU Gazetteer of Planetary Nomenclature.

Protects the map convention itself. Every other test in this project takes its
reference from inside the project: check 1c compares two implementations of one
specification, test (c) builds its sun from our own frame, and a sub-solar
bearing check placed its cell with our own inverse. All three agreed to high
precision while the projection was mirrored, with the 3D x and y components
interchanged. These two are the only tests here whose reference is external, so
they are what guards the specification rather than the implementations.

Two craters far apart in longitude, because a single landmark near the meridian
a reflection is about would be moved very little by it. Shackleton at longitude
129.78 and Cabeus at -42.13 are 171.91 deg apart.

Skips without the LOLA DEMs, so a fresh clone still passes.
"""
from pathlib import Path

import numpy as np
import pytest

from src.frames import MOON_RADIUS_M, map_to_unit

rasterio = pytest.importorskip("rasterio")
ndimage = pytest.importorskip("scipy.ndimage")

ROOT = Path(__file__).resolve().parents[1]
DEM20 = ROOT / "data" / "lola" / "LDEM_80S_20MPP_ADJ.TIF"
DEM80 = ROOT / "data" / "lola" / "LDEM_80S_80MPP_ADJ.TIF"

# The search boxes are literal map coordinates, deliberately NOT produced by
# calling the production forward projection. If the box were placed with a
# mirrored forward projection and read back with a mirrored inverse, the two
# would cancel and this test would pass while the convention was wrong.
# Thresholds are set by what a convention fault would do, not by how precisely
# a catalogue centre can be reproduced. Each is one third of that landmark's
# SMALLEST convention-fault displacement, which for both is the 56.892 deg
# rotation rather than the mirror: Shackleton 9.53/3 and Cabeus 134.79/3 km.
# The first version of this test used crater size instead, which is the wrong
# quantity and gave 1 km and 10 km. See PLAN.md, 2026-09-22.
LANDMARKS = {
    "Shackleton": dict(lat=-89.67, lon=129.78, diameter=20.92e3, tol=3.18e3,
                       dem=DEM20, px=20.0,
                       box=(-7999.8, 23380.2, -22092.7, 9287.3)),
    "Cabeus": dict(lat=-85.33, lon=-42.13, diameter=100.58e3, tol=44.93e3,
                   dem=DEM80, px=80.0,
                   box=(-170481.8, -19611.8, 29644.7, 180514.7)),
}
CONTOUR_PERCENTILE = 5.0
FAULT_MARGIN = 3.0          # threshold = smallest fault displacement / this

pytestmark = pytest.mark.skipif(
    not (DEM20.exists() and DEM80.exists()), reason="the LOLA DEMs are not present")


def latlon_to_unit(lat_deg, lon_deg):
    la, lo = np.radians(lat_deg), np.radians(lon_deg)
    return np.array([np.cos(la) * np.cos(lo), np.cos(la) * np.sin(lo), np.sin(la)])


def separation_m(lat1, lon1, lat2, lon2):
    a, b = latlon_to_unit(lat1, lon1), latlon_to_unit(lat2, lon2)
    return MOON_RADIUS_M * np.arccos(np.clip(a @ b, -1.0, 1.0))


def depression_centroid(spec):
    """Centroid of the largest closed depression in the box, in map coordinates.

    The method is fixed in PLAN.md: contour at the box's 5th percentile, largest
    8-connected component below it, centroid of its cells. Its parameters were
    chosen against the crater's size and its radius from the pole, both of which
    are independent of the longitude convention this test scores.
    """
    from rasterio.windows import Window, from_bounds
    x0, x1, y0, y1 = spec["box"]
    with rasterio.open(spec["dem"]) as ds:
        w = from_bounds(x0, y0, x1, y1, ds.transform)
        win = Window(round(w.col_off), round(w.row_off), round(w.width), round(w.height))
        z = ds.read(1, window=win).astype(np.float64)
        t = ds.window_transform(win)
    below = z < np.nanpercentile(z, CONTOUR_PERCENTILE)
    labels, _ = ndimage.label(below, structure=np.ones((3, 3), bool))
    counts = np.bincount(labels.ravel())[1:]
    rows, cols = np.nonzero(labels == int(np.argmax(counts)) + 1)
    area = counts.max() * spec["px"] ** 2
    return ((t.c + (cols + 0.5) * spec["px"]).mean(),
            (t.f - (rows + 0.5) * spec["px"]).mean(),
            2.0 * np.sqrt(area / np.pi))


@pytest.mark.parametrize("name", list(LANDMARKS))
def test_the_landmark_tolerance_discriminates(name):
    """Asserted before anything is measured: the threshold must be at most a
    third of the smallest displacement a convention fault produces, or the
    test could not reliably fail. Checking the relationship here rather than
    trusting the constant means a threshold cannot be loosened past a third
    without this firing."""
    spec = LANDMARKS[name]
    rho = 2.0 * MOON_RADIUS_M * np.tan(np.radians(90.0 + spec["lat"]) / 2.0)
    lon = np.radians(spec["lon"])
    x, y = rho * np.sin(lon), rho * np.cos(lon)

    mirrored = map_to_unit(y, x)       # feeding the swapped pair reproduces the old bug
    lat_m = np.degrees(np.arcsin(mirrored[2]))
    lon_m = (np.degrees(np.arctan2(mirrored[1], mirrored[0])) + 180) % 360 - 180
    d_mirror = separation_m(lat_m, lon_m, spec["lat"], spec["lon"])

    th = np.radians(56.892)
    u_rot = map_to_unit(x * np.cos(th) - y * np.sin(th), x * np.sin(th) + y * np.cos(th))
    lat_r = np.degrees(np.arcsin(u_rot[2]))
    lon_r = (np.degrees(np.arctan2(u_rot[1], u_rot[0])) + 180) % 360 - 180
    d_rot = separation_m(lat_r, lon_r, spec["lat"], spec["lon"])

    smallest = min(d_mirror, d_rot)
    assert spec["tol"] <= smallest / FAULT_MARGIN * 1.001, (
        f"{name}: threshold {spec['tol']/1e3:.2f} km exceeds a third of the "
        f"smallest convention fault ({smallest/1e3:.2f} km, so "
        f"{smallest/FAULT_MARGIN/1e3:.2f} km); it would not reliably catch one")
    assert spec["tol"] >= smallest / FAULT_MARGIN * 0.9, (
        f"{name}: threshold {spec['tol']/1e3:.2f} km is far below a third of the "
        f"smallest fault ({smallest/FAULT_MARGIN/1e3:.2f} km); it is tighter than "
        f"the test's purpose requires and will be brittle")


@pytest.mark.parametrize("name", list(LANDMARKS))
def test_the_right_crater_was_found(name):
    """Distance from the pole depends on latitude alone, so it confirms the
    search found the intended crater without using the longitude the test
    scores. A method that had locked onto a different depression would show up
    here rather than as a convention failure."""
    spec = LANDMARKS[name]
    xc, yc, equiv_diameter = depression_centroid(spec)
    rho_expected = 2.0 * MOON_RADIUS_M * np.tan(np.radians(90.0 + spec["lat"]) / 2.0)
    radial = abs(np.hypot(xc, yc) - rho_expected)
    # Bounded by crater size, not by the convention threshold: this test asks
    # whether the search found the intended crater, a separate question from
    # whether the projection is right.
    assert radial < 0.15 * spec["diameter"], (
        f"{name}: centroid is {np.hypot(xc,yc)/1e3:.2f} km from the pole, "
        f"expected {rho_expected/1e3:.2f} km")
    assert 0.2 * spec["diameter"] < equiv_diameter < 1.5 * spec["diameter"], (
        f"{name}: the depression found is {equiv_diameter/1e3:.1f} km across, "
        f"against a catalogued {spec['diameter']/1e3:.1f} km")


@pytest.mark.parametrize("name", list(LANDMARKS))
def test_landmark_position_through_the_production_inverse(name):
    spec = LANDMARKS[name]
    xc, yc, _ = depression_centroid(spec)
    u = map_to_unit(xc, yc)
    lat = np.degrees(np.arcsin(u[2]))
    lon = (np.degrees(np.arctan2(u[1], u[0])) + 180) % 360 - 180
    d = separation_m(lat, lon, spec["lat"], spec["lon"])
    assert d <= spec["tol"], (
        f"{name}: our inverse puts the crater at lat {lat:.4f}, lon {lon:.4f}, "
        f"{d/1e3:.2f} km from the gazetteer's {spec['lat']}, {spec['lon']}, "
        f"against a {spec['tol']/1e3:.1f} km tolerance")
