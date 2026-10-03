"""How long the world and the planner's illumination model actually disagree.

The executive's monitor has one rule that fires on its own authority: where the
checker's integration says the rover is dark and the planner's interval table
says its cell is lit, the observation wins and the darkness is carried forward.
How far forward is the question this measures, so that the horizon is set from
the disagreements this terrain produces rather than from whatever makes a bar
pass.

The grid is the checker's own visibility cadence, 60 s, because that is the
resolution at which the world's lit state is known; a finer grid would be
interpolating and a coarser one would be throwing away the data.
"""
import numpy as np

CADENCE_H = 1.0 / 60.0


class FlatIntervals:
    """The planner's per-cell lit/dark table, read from the raw arrays."""

    def __init__(self, itimes_path, ioffsets_path, ilit0_path):
        self.times = np.fromfile(itimes_path, dtype=np.float64)
        self.offsets = np.fromfile(ioffsets_path, dtype=np.int64)
        self.lit0 = np.fromfile(ilit0_path, dtype=np.uint8)

    def is_lit(self, cell, t):
        lo, hi = self.offsets[cell], self.offsets[cell + 1]
        k = int(np.searchsorted(self.times[lo:hi], t, side="right"))
        return bool((self.lit0[cell] != 0) != (k % 2 == 1))


def sample_states(tick_t, tick_lit, tick_index, plan, iv, cadence_h=CADENCE_H):
    """World and model lit states on a fixed grid along one executed trajectory.

    The world's state is piecewise constant between ticks, which is how the
    checker produced it: it re-reads visibility on its cadence and a change
    emits a tick. The model's state is read from the table at the cell the rover
    is on at that moment.
    """
    tick_t = np.asarray(tick_t, dtype=float)
    if tick_t.size < 2:
        return np.zeros(0), np.zeros(0, bool), np.zeros(0, bool)
    grid = np.arange(tick_t[0], tick_t[-1], cadence_h)
    if grid.size == 0:
        return np.zeros(0), np.zeros(0, bool), np.zeros(0, bool)
    seg = np.searchsorted(tick_t, grid, side="right") - 1
    seg = np.clip(seg, 0, tick_t.size - 1)
    world = np.asarray(tick_lit, dtype=bool)[seg]
    idx = np.clip(np.asarray(tick_index, dtype=np.int64)[seg], 0, len(plan) - 1)
    model = np.array([iv.is_lit(int(plan[i]), float(t))
                      for i, t in zip(idx, grid)], dtype=bool)
    return grid, world, model


def episodes(world, model, cadence_h=CADENCE_H, world_dark_only=True):
    """Durations of the maximal runs on which the two disagree.

    world_dark_only keeps the direction the monitor's rule acts on, the world
    dark where the model says lit. The other direction is measured too, because
    a rule justified on one direction should be reported beside the other.
    """
    world = np.asarray(world, dtype=bool)
    model = np.asarray(model, dtype=bool)
    if world.size == 0:
        return np.zeros(0)
    bad = (model & ~world) if world_dark_only else (world & ~model)
    if not bad.any():
        return np.zeros(0)
    # run-length encode the True runs
    d = np.diff(np.concatenate(([0], bad.view(np.int8), [0])))
    starts = np.flatnonzero(d == 1)
    ends = np.flatnonzero(d == -1)
    return (ends - starts) * cadence_h
