"""Check 1c: the C++ horizon model against an independent Python ray march.

Protects against a fault shared by nothing else in the suite: tests (a) to (c)
run the binary on synthetic scenes that this project built, so they cannot see
an error that only appears on real terrain, at real distances, across the
near/far switch, under the real step schedule. This test marches the same rays
again in Python, written from the specification rather than from the C++, and
compares every ray at the cells selected by the rule registered in PLAN.md.

It skips when data/work/ is absent, so a fresh clone with no data still passes.
"""
from pathlib import Path

import numpy as np
import pytest

from tests.independent_horizon import Reference, select_cells

ROOT = Path(__file__).resolve().parents[1]
WORK = ROOT / "data" / "work"
CRITERION_DEG = 1.0e-3          # registered in PLAN.md check 1c before the run

_REQUIRED = ("params.json", "horizon.f32", "horizon_dist.f32", "near.f32", "far.f32")
pytestmark = pytest.mark.skipif(
    not all((WORK / f).exists() for f in _REQUIRED),
    reason="data/work/ not present; run examples/extract_windows.py then cpp/build/horizon")


@pytest.fixture(scope="module")
def marched():
    ref = Reference(WORK)
    p = ref.params
    n, naz = p["region_nx"], p["n_azimuth"]
    horizon = np.fromfile(p["out_horizon"], dtype=np.float32).reshape(n, n, naz)
    dist = np.fromfile(p["out_horizon_dist"], dtype=np.float32).reshape(n, n, naz)
    cells, why, *_ = select_cells(horizon, dist, p["switch_m"])
    rows = []
    for (r, c) in cells:
        h, d = ref.march(r, c, np.arange(float(naz)))
        rows.append(((r, c), why[(r, c)], h, d, horizon[r, c].astype(np.float64), dist[r, c]))
    return rows, naz


def test_the_selected_cells_span_the_regimes_the_check_is_for(marched):
    """A cell set that was all near-field, or all in one corner, could not
    express a fault in the far field or in the local frame."""
    rows, _ = marched
    labels = {why for _, why, *_ in rows}
    for expected in ("most near-set", "most far-set", "region corner",
                     "largest peak-to-peak", "largest adjacent jump"):
        assert expected in labels, f"the selected cells are missing: {expected}"
    assert len(rows) >= 15


def test_python_and_cpp_horizons_agree_within_the_registered_criterion(marched):
    rows, naz = marched
    worst, where = 0.0, None
    for cell, _, h, _, hc, _ in rows:
        d = float(np.max(np.abs(h - hc)))
        if d > worst:
            worst, where = d, cell
    assert worst < CRITERION_DEG, (
        f"worst disagreement {worst:.3e} deg at cell {where}, criterion {CRITERION_DEG}")


def test_the_residual_is_float32_output_only(marched):
    """The registered claim is not merely that the two agree, but that what
    is left is the float32 output rounding. If the residual grew past a few
    ulps of the stored value, the two implementations would have diverged for
    some other reason even while still passing the criterion."""
    rows, _ = marched
    for cell, _, h, _, hc, _ in rows:
        resid = np.abs(h - hc)
        ulp = np.spacing(np.abs(hc).astype(np.float32)).astype(np.float64)
        assert np.all(resid <= 4.0 * np.maximum(ulp, np.spacing(np.float32(1.0)))), (
            f"cell {cell}: residual {resid.max():.3e} deg exceeds four float32 ulps "
            f"of the stored value, so it is not output rounding")
