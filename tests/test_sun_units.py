"""The sun arrives in metres, and a kilometre value is refused rather than used.

What this protects. SPICE spkpos returns kilometres; every frame decomposition
here subtracts a cell origin in metres on the 1737400 m sphere. Seven producers
passed the kilometre value straight in, so the parallax term was a thousand
times too large and the sun sat 0.67218 deg too LOW at every cell and every
epoch. It survived the whole project because the azimuth stayed exact: measured
over six cells at five epochs the largest azimuth error was 5.7e-14 deg, which
is float round-off. cell_position is radial, so the origin is parallel to the
frame's up and subtracting it perturbs the up component alone.

That is the hard part of this fault and it is what these tests target: a bias
that moves one angle and leaves the other provably untouched is invisible to
every check that looks at bearings.
"""
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.frames import (MOON_RADIUS_M, azimuth_elevation_from_frame,  # noqa: E402
                        cell_position, local_frame)
from src.sun import (KM_TO_M, SUN_DISTANCE_MAX_M, SUN_DISTANCE_MIN_M,  # noqa: E402
                     check_sun_metres)

AU_M = 1.496e11


def _frame(n=5, seed=3):
    rng = np.random.default_rng(seed)
    x = rng.uniform(-4000.0, 6000.0, n)
    y = rng.uniform(1000.0, 11000.0, n)
    h = rng.uniform(-800.0, 1400.0, n)
    origin = cell_position(x, y, h, MOON_RADIUS_M)
    up, east, north = local_frame(x, y, MOON_RADIUS_M)
    return origin, up, east, north


def test_a_kilometre_sun_is_refused_by_the_guard():
    """Protects: the exact defect, a km array reaching a frame decomposition.

    The discriminating half is asserted first: the metre value must pass, or a
    guard that refused everything would also produce this refusal.
    """
    metres = np.array([[AU_M, 0.0, 0.0]])
    check_sun_metres(metres)                      # must not raise

    with pytest.raises(ValueError, match="kilometres"):
        check_sun_metres(metres / KM_TO_M)        # the defect

    with pytest.raises(ValueError, match="kilometres"):
        check_sun_metres(metres * KM_TO_M)        # converted twice

    # and the band's own edges, so a later widening is a deliberate act
    check_sun_metres(np.array([[SUN_DISTANCE_MIN_M, 0.0, 0.0]]))
    check_sun_metres(np.array([[SUN_DISTANCE_MAX_M, 0.0, 0.0]]))
    with pytest.raises(ValueError):
        check_sun_metres(np.array([[SUN_DISTANCE_MIN_M * 0.99, 0.0, 0.0]]))


def test_the_unit_error_moves_elevation_and_leaves_azimuth_exact():
    """Protects: the reason this survived every bearing check in the project.

    If a future change made the error show in azimuth too, this test would fail
    and the claim in the record about why it hid would no longer be true.
    """
    origin, up, east, north = _frame()
    sun_m = np.array([0.3, -0.9, 0.31])
    sun_m = sun_m / np.linalg.norm(sun_m) * AU_M

    az_m, el_m = azimuth_elevation_from_frame(sun_m, origin, up, east, north)
    az_k, el_k = azimuth_elevation_from_frame(sun_m / KM_TO_M, origin, up,
                                              east, north)

    daz = np.abs(((az_k - az_m + 180.0) % 360.0) - 180.0)
    assert daz.max() < 1e-9, f"azimuth must be invariant, saw {daz.max()}"

    del_ = el_k - el_m
    assert (del_ < 0).all(), "the kilometre sun must sit LOWER, not higher"
    assert 0.5 < -del_.mean() < 0.9, f"expected about 0.67 deg, saw {-del_.mean()}"


def test_cell_origin_is_radial_which_is_why_azimuth_is_invariant():
    """Protects: the mechanism the test above relies on.

    Azimuth is invariant only because the origin is parallel to up. If
    cell_position ever stopped being radial the invariance would quietly go,
    and the test above would start passing for the wrong reason.
    """
    origin, up, _, _ = _frame()
    cross = np.linalg.norm(np.cross(origin, up), axis=-1)
    assert (cross / np.linalg.norm(origin, axis=-1)).max() < 1e-12


def test_the_suntable_refuses_a_kilometre_table():
    """Protects: the interval tables, which every plan and sweep descends from."""
    from src.intervals import SunTable
    good = np.tile(np.array([[AU_M, 0.0, 0.0]]), (4, 1))
    SunTable(good, 0.25)                              # must not raise
    with pytest.raises(ValueError, match="kilometres"):
        SunTable(good / KM_TO_M, 0.25)


def test_sun_positions_m_returns_metres():
    """Protects: the one conversion in the project.

    Skips without kernels, so a fresh clone with no data still passes.
    """
    sp = pytest.importorskip("spiceypy")
    kernels = ROOT / "data" / "kernels"
    if not kernels.is_dir() or not any(kernels.iterdir()):
        pytest.skip("SPICE kernels are not fetched")
    import examples.plan_pilot as P
    from src.sun import sun_positions_m
    for k in P.KERNELS:
        sp.furnsh(str(kernels / k))
    try:
        et = sp.str2et("2028-02-15T00:00:00")
        raw_km = np.asarray(sp.spkpos("SUN", et, P.FRAME, "LT+S", "MOON")[0], float)
        got = sun_positions_m(et, P.FRAME)
    finally:
        sp.kclear()
    assert got.shape == (1, 3)
    assert np.allclose(got[0], raw_km * KM_TO_M, rtol=0, atol=1e-6)
    d = np.linalg.norm(got[0])
    assert SUN_DISTANCE_MIN_M < d < SUN_DISTANCE_MAX_M


def test_no_spkpos_call_survives_outside_the_one_module():
    """Protects: a producer quietly going back to raw spkpos.

    The guards catch a kilometre table at runtime, but nothing failed when a
    call site reverted, because no test exercised that call site. This is the
    test for the claim src/sun.py makes in its own docstring, that there is one
    conversion in the project: any new spkpos outside it is a second place the
    units can be got wrong.
    """
    import subprocess
    out = subprocess.run(["git", "ls-files", "*.py"], cwd=ROOT,
                         capture_output=True, text=True, check=True)
    offenders = []
    for rel in out.stdout.split():
        if rel in ("src/sun.py", "tests/test_sun_units.py"):
            continue
        text = (ROOT / rel).read_text()
        for i, line in enumerate(text.splitlines(), 1):
            if "spkpos" in line and not line.lstrip().startswith("#"):
                offenders.append(f"{rel}:{i}: {line.strip()}")
    assert not offenders, (
        "spkpos must only be called in src/sun.py, which converts km to m:\n  "
        + "\n  ".join(offenders))


def test_a_producer_entry_point_hands_back_metres():
    """Protects: the conversion reaching a real caller, not just the helper.

    sun_table_at is the producer every interval table in the project descends
    from. Skips without kernels so a fresh clone still passes.
    """
    pytest.importorskip("spiceypy")
    if not (ROOT / "data" / "kernels").is_dir():
        pytest.skip("SPICE kernels are not fetched")
    from examples.build_intervals import sun_table_at
    table = sun_table_at("2028-02-15T00:00:00", 0.0, cadence_h=6.0)
    d = np.linalg.norm(np.asarray(table.p, float), axis=-1)
    assert (d > SUN_DISTANCE_MIN_M).all() and (d < SUN_DISTANCE_MAX_M).all(), \
        f"sun_table_at returned magnitudes {d.min():.3e} to {d.max():.3e}"
