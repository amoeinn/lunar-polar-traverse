"""The one illumination-derived quantity phase 2 produces.

The mean-map baseline is defined in PLAN.md as A* with cost weighted by each
cell's lit fraction over the planning horizon. That fraction is a static
property of a cell over the whole horizon, not a property of an arrival time,
so it belongs to phase 2 -- but it belongs to the baseline, not to the
time-dependent planner, whose cost must stay illumination-free or the
comparison between them becomes circular.

It lives in its own module for that reason. src/cost.py does not import this,
and tests/test_cost_model.py asserts so by reading src/cost.py's imports rather
than by trusting the convention.
"""
import numpy as np

from .illumination import shadow_mask


def lit_fraction(cells, sun_positions, profiles, edge_deg=0.0):
    """Fraction of the sampled epochs at which each cell is lit.

    sun_positions is (n_epochs, 3) of Moon-fixed positions. Sampling, not an
    envelope: the mean map wants how often a cell is lit, which every epoch
    contributes to, where never-lit wanted only the extremes.
    """
    sun_positions = np.atleast_2d(np.asarray(sun_positions, dtype=float))
    lit = np.zeros(len(cells), dtype=np.int64)
    for s in sun_positions:
        lit += ~shadow_mask(cells, s, profiles, edge_deg=edge_deg)
    return lit / float(sun_positions.shape[0])


def mean_map_weight(lit, floor=1e-3):
    """Cost multiplier for the mean-map baseline: cheap where lit, dear where not.

    The floor keeps a never-lit cell expensive rather than infinite, so the
    baseline still produces a plan to compare against instead of declaring the
    goal unreachable, which would make it a different experiment.
    """
    return 1.0 / np.maximum(np.asarray(lit, dtype=float), floor)
