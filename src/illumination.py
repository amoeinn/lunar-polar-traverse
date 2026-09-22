"""Illumination products built on the horizon model.

Everything here goes through the (azimuth, elevation) decomposition and the
horizon lookup in src/frames.py rather than restating either, so the Python
side has exactly one definition of the sun convention. The frame is hoisted
out of the epoch loop by calling azimuth_elevation_from_frame, which is the
same code path azimuth_elevation takes, not a second copy of it.
"""
import numpy as np

from .frames import (MOON_RADIUS_M, azimuth_elevation_from_frame, cell_position,
                     horizon_lookup, local_frame)


class RegionCells:
    """Per-cell geometry for a fixed set of cells, computed once.

    Building the local frame is the expensive part of an epoch scan and does
    not depend on the epoch, so it happens here and nowhere else.
    """

    def __init__(self, x, y, height, radius=MOON_RADIUS_M, observer_height=0.0):
        self.x = np.asarray(x, dtype=float)
        self.y = np.asarray(y, dtype=float)
        self.height = np.asarray(height, dtype=float)
        self.radius = radius
        self.observer_height = observer_height
        self.up, self.east, self.north = local_frame(self.x, self.y, radius)
        self.origin = cell_position(self.x, self.y, self.height, radius, observer_height)

    def __len__(self):
        return self.x.size


def sun_angles(cells, sun_position):
    """Azimuth and elevation of one Moon-fixed sun position at every cell."""
    return azimuth_elevation_from_frame(sun_position, cells.origin,
                                        cells.up, cells.east, cells.north)


def shadow_mask(cells, sun_position, profiles, edge_deg=0.0):
    """True where the sun is below the horizon, for one Moon-fixed sun position.

    edge_deg selects the solar edge: +s is first light (upper limb clearing),
    0 is the point source, -s is full sun (lower limb clearing).
    """
    azimuth, elevation = sun_angles(cells, sun_position)
    return (elevation + edge_deg) < horizon_lookup(profiles, azimuth)


def scan_epochs(cells, sun_positions, profiles, dark_observed, edge_deg=0.0):
    """Agreement between our shadow mask and an observed dark mask, per epoch.

    sun_positions is (n_epochs, 3) of Moon-fixed positions, one per epoch, in
    the same order as the epochs they came from. Returns the agreement for
    each, and the agreement of the always-lit predictor, which marks nothing
    shadowed and is the baseline any real result has to beat.
    """
    sun_positions = np.atleast_2d(np.asarray(sun_positions, dtype=float))
    dark_observed = np.asarray(dark_observed, dtype=bool)
    if dark_observed.shape != (len(cells),):
        raise ValueError(f"dark_observed has shape {dark_observed.shape}, "
                         f"expected ({len(cells)},)")
    if profiles.shape[0] != len(cells):
        raise ValueError(f"profiles has {profiles.shape[0]} rows for {len(cells)} cells")

    agreement = np.empty(sun_positions.shape[0])
    for i in range(sun_positions.shape[0]):
        agreement[i] = np.mean(
            shadow_mask(cells, sun_positions[i], profiles, edge_deg) == dark_observed)
    baseline_always_lit = float(np.mean(~dark_observed))
    return agreement, baseline_always_lit
