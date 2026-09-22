"""Round trip for the epoch scan in src/illumination.py.

Protects the scan that determines which epoch a published shadow mask was
rendered at. That result is used as a registered target for P6, so the scan
has to be shown to recover an epoch it is given, rather than merely to return
a number. The test builds a mask from our own model at a known epoch, scans a
window around it, and requires the known epoch to be the unique argmax.

Everything here is synthetic: real map coordinates and the real local frame,
which rotate strongly this close to the pole, but synthetic horizon profiles
and a synthetic sun track. So it needs no data and runs on a fresh clone.
"""
import numpy as np
import pytest

from src.frames import MOON_RADIUS_M, cell_position, local_frame
from src.illumination import RegionCells, scan_epochs, shadow_mask

CENTRE = (1000.0, 6000.0)
# With 300 cells and a 0.5 deg step no cell flipped between adjacent epochs,
# so neighbouring masks were identical and the argmax was a tie. A cell flips
# when its horizon crosses the sun elevation, about twice per 360 deg sweep,
# so the expected flips per step is roughly n_cells * 2 * step / 360: 2000
# cells at 1 deg gives about 11, which makes an accidental tie negligible.
N_CELLS = 2000
N_EPOCHS = 120
KNOWN = 47                      # not at either end, and not the midpoint
SUN_RANGE_M = 1.5e11
AZ0, AZ_STEP, SUN_EL = 300.0, 1.0, 1.0


@pytest.fixture(scope="module")
def scene():
    rng = np.random.default_rng(20260922)
    x = CENTRE[0] + rng.uniform(-5000.0, 5000.0, N_CELLS)
    y = CENTRE[1] + rng.uniform(-5000.0, 5000.0, N_CELLS)
    height = rng.uniform(-800.0, 1400.0, N_CELLS)
    cells = RegionCells(x, y, height)

    # Per-cell horizon profiles that straddle the sun elevation, so cells flip
    # between lit and shadowed as the sun sweeps. A flat or shared profile
    # could not distinguish one epoch from another.
    a = np.arange(360.0)
    amp = rng.uniform(0.8, 2.5, N_CELLS)[:, None]
    phase = rng.uniform(0.0, 360.0, N_CELLS)[:, None]
    offset = rng.uniform(-0.6, 0.6, N_CELLS)[:, None]
    profiles = amp * np.cos(np.radians(a[None, :] - phase)) + offset

    up, east, north = local_frame(*CENTRE)
    origin = cell_position(*CENTRE, 0.0)
    e = np.radians(SUN_EL)
    suns = []
    for k in range(N_EPOCHS):
        az = np.radians(AZ0 + k * AZ_STEP)
        d = np.cos(e) * (np.cos(az) * north + np.sin(az) * east) + np.sin(e) * up
        suns.append(origin + d * SUN_RANGE_M)
    return cells, profiles, np.array(suns)


def reference_mask(x, y, height, sun_position, profiles, radius=MOON_RADIUS_M):
    """An independent shadow mask, written from the convention rather than by
    calling the production path.

    If the observed mask were built with shadow_mask, a fault in shadow_mask
    would appear in both the mask and the scan and cancel out of the round
    trip, leaving the test unable to fail. The local frame here is in closed
    form and the horizon interpolation is written out, so nothing is shared.
    """
    r = np.hypot(x, y)
    t = r / (2.0 * radius)
    den = 1.0 + t * t
    st, ct = 2.0 * t / den, (1.0 - t * t) / den
    sl, cl = x / r, y / r
    up = np.stack([st * cl, st * sl, -ct], axis=-1)
    east = np.stack([-sl, cl, np.zeros_like(cl)], axis=-1)
    north = np.stack([ct * cl, ct * sl, st], axis=-1)
    d = np.asarray(sun_position, float) - up * (radius + height)[:, None]
    d = d / np.linalg.norm(d, axis=-1, keepdims=True)
    azimuth = np.degrees(np.arctan2(np.sum(d * east, -1), np.sum(d * north, -1))) % 360.0
    elevation = np.degrees(np.arcsin(np.sum(d * up, -1)))
    i0 = np.floor(azimuth).astype(int) % 360
    frac = azimuth - np.floor(azimuth)
    idx = np.arange(profiles.shape[0])
    horizon = (1.0 - frac) * profiles[idx, i0] + frac * profiles[idx, (i0 + 1) % 360]
    return elevation < horizon


@pytest.fixture(scope="module")
def observed(scene):
    cells, profiles, suns = scene
    return reference_mask(cells.x, cells.y, cells.height, suns[KNOWN], profiles)


def test_production_mask_matches_the_independent_one(scene, observed):
    """The round trip is only meaningful if the production mask agrees with a
    mask this test built without it."""
    cells, profiles, suns = scene
    assert np.array_equal(shadow_mask(cells, suns[KNOWN], profiles), observed)


def test_the_fixture_can_distinguish_epochs(scene, observed):
    """A mask that is all lit, all dark, or unchanged between neighbouring
    epochs would make the argmax meaningless and the test unfailable."""
    cells, profiles, suns = scene
    assert 0.05 < observed.mean() < 0.95, f"mask is {observed.mean():.2%} shadowed"
    for neighbour in (KNOWN - 1, KNOWN + 1):
        other = shadow_mask(cells, suns[neighbour], profiles)
        assert not np.array_equal(other, observed), (
            f"epoch {neighbour} gives the same mask as {KNOWN}; the sun step is too small")


def test_the_fixture_can_see_a_negated_azimuth(scene, observed):
    """Guards the mutation check below: with a symmetric profile set, negating
    the azimuth could map the mask onto itself and prove nothing."""
    cells, profiles, suns = scene
    from src.frames import horizon_lookup
    from src.illumination import sun_angles
    azimuth, elevation = sun_angles(cells, suns[KNOWN])
    negated = (elevation < horizon_lookup(profiles, -azimuth))
    assert not np.array_equal(negated, observed), (
        "negating the azimuth leaves the mask unchanged on this fixture")


def test_scan_recovers_the_epoch_the_mask_was_built_at(scene, observed):
    cells, profiles, suns = scene
    agreement, baseline = scan_epochs(cells, suns, profiles, observed)
    assert int(np.argmax(agreement)) == KNOWN, (
        f"scan picked epoch {int(np.argmax(agreement))}, mask was built at {KNOWN}")
    assert agreement[KNOWN] == pytest.approx(1.0), (
        f"agreement at the true epoch is {agreement[KNOWN]:.6f}, not 1")
    others = np.delete(agreement, KNOWN)
    assert others.max() < 1.0, "another epoch ties the true one; the argmax is not unique"


def test_baseline_is_the_always_lit_predictor(scene, observed):
    """The baseline must be what an empty model scores, or a result that beats
    nothing would look like a result."""
    cells, profiles, suns = scene
    _, baseline = scan_epochs(cells, suns[:3], profiles, observed)
    assert baseline == pytest.approx(1.0 - observed.mean())
    assert baseline < 1.0


def test_scan_rejects_a_mismatched_mask(scene):
    cells, profiles, suns = scene
    with pytest.raises(ValueError):
        scan_epochs(cells, suns[:2], profiles, np.zeros(len(cells) + 1, dtype=bool))
