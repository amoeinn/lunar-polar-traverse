"""The sphere radius used by the model must equal the one the DEM declares.

Protects against a radius that is right in the parameter file and wrong
against the data, or that drifts from it later. Check 1c does not close this:
it compares the model against an independent march that is handed the same
radius, and its mutation table shows it clears the 1e-3 deg criterion by only
a factor of 1.8 for a 1 km radius error, on 9 of 19 cells. A smaller error
would pass check 1c silently while moving every elevation angle. Reading the
radius back out of the raster's own CRS and demanding exact equality leaves no
room for it to drift at all.

Skips when data/work/ or the DEM is absent, so a fresh clone still passes.
"""
from pathlib import Path

import json
import pytest

rasterio = pytest.importorskip("rasterio")

ROOT = Path(__file__).resolve().parents[1]
SIDECAR = ROOT / "data" / "work" / "horizon.json"
DEMS = {
    "20 m LDEM": ROOT / "data" / "lola" / "LDEM_80S_20MPP_ADJ.TIF",
    "80 m LDEM": ROOT / "data" / "lola" / "LDEM_80S_80MPP_ADJ.TIF",
}

pytestmark = pytest.mark.skipif(
    not (SIDECAR.exists() and all(p.exists() for p in DEMS.values())),
    reason="data/work/horizon.json or the LOLA DEMs are not present")


def crs_sphere_radius(path):
    with rasterio.open(path) as ds:
        d = ds.crs.to_dict()
    assert "R" in d, f"{path.name}: CRS is not spherical, got {d}"
    return float(d["R"])


def test_the_dems_agree_with_each_other_on_the_radius():
    """If the two products disagreed, "the CRS radius" would be ambiguous and
    the test below would be comparing against an arbitrary choice."""
    radii = {name: crs_sphere_radius(p) for name, p in DEMS.items()}
    assert len(set(radii.values())) == 1, f"DEMs disagree on the sphere radius: {radii}"


def test_sidecar_radius_equals_the_dem_crs_radius_exactly():
    sidecar = json.loads(SIDECAR.read_text())
    declared = crs_sphere_radius(DEMS["20 m LDEM"])
    assert sidecar["radius_m"] == declared, (
        f"the run used radius {sidecar['radius_m']!r} m but the DEM's CRS declares "
        f"{declared!r} m; a difference of {sidecar['radius_m'] - declared:+.1f} m")
