"""Lit and dark intervals per cell, and the shadow a traverse accumulates.

The planner needs three things illumination-side: whether a cell is lit at a
time, when it next changes, and how much continuous shadow an edge traversal
accumulates. All three come from the transition times, found once per cell by
sampling g at the registered cadence and bisecting each sign change, which is
what makes the planner's time axis continuous rather than a grid.

The shadow rule is registered in PLAN.md: an edge contributes the measure of
the departure-to-arrival interval during which either endpoint is dark. It is
conservative, because the rover is at one endpoint or the other and never at
both, and the conservatism is for the replay checker to measure.
"""
from pathlib import Path

import numpy as np

from .frames import azimuth_elevation_from_frame, horizon_lookup

CADENCE_H = 0.25          # registered: 15 minutes
BISECTION_STEPS = 40


class SunTable:
    """Moon-fixed sun positions, linearly interpolated between samples.

    The same rule cpp/src/edges.cpp uses, and for the same reason: over 15
    minutes the chord error is 3.5e-5 deg of direction, a thirtieth of a
    thousandth of the ray spacing.
    """

    def __init__(self, positions, dt_hours):
        # Metres, not kilometres. g_at subtracts a cell origin on the 1737400 m
        # sphere, so a km table mixes units and drops the sun 0.67 deg. Seven
        # producers did exactly that until 2026-11-16; this is the gate that
        # makes a repeat loud instead of a 0.67 deg bias nobody sees.
        from src.sun import check_sun_metres
        check_sun_metres(positions, where="SunTable")
        self.p = np.asarray(positions, dtype=float)
        self.dt = float(dt_hours)

    def at(self, t_hours):
        u = np.asarray(t_hours, dtype=float) / self.dt
        k = np.clip(np.floor(u).astype(int), 0, self.p.shape[0] - 2)
        f = (u - k)[..., None]
        return (1.0 - f) * self.p[k] + f * self.p[k + 1]

    @property
    def horizon_h(self):
        return (self.p.shape[0] - 1) * self.dt


def g_at(cells, profiles, sun, t_hours):
    """Sun elevation minus the cell's horizon at the sun's azimuth, in degrees.

    cells is a RegionCells over the same cells as profiles, in the same order.
    t_hours may be an array; the result is (n_cells, n_times) or (n_cells,).
    """
    t = np.asarray(t_hours, dtype=float)
    s = sun.at(t)
    if t.ndim == 0:
        az, el = azimuth_elevation_from_frame(s, cells.origin, cells.up,
                                              cells.east, cells.north)
        return el - horizon_lookup(profiles, az)
    az, el = azimuth_elevation_from_frame(s[None, :, :], cells.origin[:, None, :],
                                          cells.up[:, None, :], cells.east[:, None, :],
                                          cells.north[:, None, :])
    return el - horizon_lookup(profiles[:, None, :], az)


class LitIntervals:
    """Transition times per cell over [0, horizon_h], in hours from the epoch."""

    def __init__(self, times, offsets, lit_at_zero, horizon_h):
        self.times = np.asarray(times, dtype=float)
        self.offsets = np.asarray(offsets, dtype=np.int64)
        self.lit0 = np.asarray(lit_at_zero, dtype=bool)
        self.horizon_h = float(horizon_h)

    @classmethod
    def build(cls, cells, profiles, sun, horizon_h, cadence_h=CADENCE_H, chunk=2000):
        """Transition times for every cell, sampled at the cadence and bisected.

        The bisection is vectorised over every sign change in a chunk at once.
        Doing it one crossing at a time is a hundred times slower and was the
        first thing that made a full-region build impractical.
        """
        grid = np.arange(0.0, horizon_h + 0.5 * cadence_h, cadence_h)
        grid[-1] = min(grid[-1], horizon_h)
        n = len(cells)
        per_cell = [None] * n
        for i in range(0, n, chunk):
            j = min(i + chunk, n)
            sl = slice(i, j)
            sub = _Sub(cells, sl)
            sign = g_at(sub, profiles[sl], sun, grid) > 0.0
            rows, ks = np.nonzero(sign[:, :-1] != sign[:, 1:])
            lo, hi = grid[ks].copy(), grid[ks + 1].copy()
            want = sign[rows, ks + 1]
            orig, up = sub.origin[rows], sub.up[rows]
            east, north = sub.east[rows], sub.north[rows]
            prof = profiles[sl][rows]
            for _ in range(BISECTION_STEPS):
                mid = 0.5 * (lo + hi)
                s = sun.at(mid)
                az, el = azimuth_elevation_from_frame(s, orig, up, east, north)
                pos = (el - horizon_lookup(prof, az)) > 0.0
                hit = pos == want
                hi = np.where(hit, mid, hi)
                lo = np.where(hit, lo, mid)
            cut = 0.5 * (lo + hi)
            for r in range(j - i):
                per_cell[i + r] = np.sort(cut[rows == r])
            if i == 0:
                lit0 = np.empty(n, dtype=bool)
            lit0[sl] = sign[:, 0]
        offsets = np.zeros(n + 1, dtype=np.int64)
        for c in range(n):
            offsets[c + 1] = offsets[c] + per_cell[c].size
        times = np.concatenate(per_cell) if n else np.zeros(0)
        return cls(times, offsets, lit0, horizon_h)

    def save(self, path):
        """Cache to disk. Interval tables are shared by every configuration at
        an epoch, and rebuilding them per configuration costs more than the
        searches they feed."""
        np.savez(str(path), times=self.times, offsets=self.offsets,
                 lit0=self.lit0, horizon_h=np.array([self.horizon_h]))

    @classmethod
    def load(cls, path):
        z = np.load(str(path))
        return cls(z["times"], z["offsets"], z["lit0"], float(z["horizon_h"][0]))

    @classmethod
    def build_or_load(cls, cache, cells, profiles, sun, horizon_h,
                      cadence_h=CADENCE_H, chunk=2000):
        cache = Path(cache)
        if cache.exists():
            iv = cls.load(cache)
            if (iv.offsets.size == len(cells) + 1
                    and abs(iv.horizon_h - horizon_h) < 1e-9):
                return iv, True
        iv = cls.build(cells, profiles, sun, horizon_h, cadence_h=cadence_h, chunk=chunk)
        cache.parent.mkdir(parents=True, exist_ok=True)
        iv.save(cache)
        return iv, False

    def lit_fraction(self):
        """Fraction of the horizon each cell is lit, exactly, from the transitions.

        The mean-map baseline needs this per cell. Sampling it would cost a
        rebuild; the transition times already carry it, so it is a sum of
        interval lengths rather than 2880 shadow masks.
        """
        n = self.offsets.size - 1
        out = np.empty(n, dtype=float)
        for c in range(n):
            tr = self.transitions(c)
            edges = np.concatenate(([0.0], tr, [self.horizon_h]))
            spans = np.diff(edges)
            lit = spans[0::2] if self.lit0[c] else spans[1::2]
            out[c] = lit.sum() / self.horizon_h
        return out

    def transitions(self, cell):
        return self.times[self.offsets[cell]:self.offsets[cell + 1]]

    def is_lit(self, cell, t):
        k = int(np.searchsorted(self.transitions(cell), t, side="right"))
        return bool(self.lit0[cell] ^ (k % 2 == 1))

    def next_change(self, cell, t):
        tr = self.transitions(cell)
        k = int(np.searchsorted(tr, t, side="right"))
        return float(tr[k]) if k < tr.size else self.horizon_h

    def transitions_in(self, cell, t0, t1):
        tr = self.transitions(cell)
        lo = int(np.searchsorted(tr, t0, side="right"))
        hi = int(np.searchsorted(tr, t1, side="left"))
        return tr[lo:hi]

    def dark_spans(self, cells, t0, t1):
        """Maximal sub-intervals of [t0, t1] where any of the given cells is dark."""
        spans = []
        for c in cells:
            t = t0
            while t < t1:
                nxt = min(self.next_change(c, t), t1)
                if not self.is_lit(c, t):
                    spans.append((t, nxt))
                t = nxt if nxt > t else t1
        if not spans:
            return []
        spans.sort()
        merged = [list(spans[0])]
        for a, b in spans[1:]:
            if a <= merged[-1][1]:
                merged[-1][1] = max(merged[-1][1], b)
            else:
                merged.append([a, b])
        return [tuple(m) for m in merged]

    def shadow_along(self, cell_a, cell_b, t_dep, t_arr, s_dep):
        """Continuous shadow time at arrival and its maximum over the traverse.

        The registered rule: the rover is in shadow whenever either endpoint is
        dark. Shadow time resets to zero the moment it is in light.
        """
        spans = self.dark_spans((cell_a, cell_b), t_dep, t_arr)
        s, s_max, s_arr = s_dep, s_dep, 0.0
        for a, b in spans:
            s = (s_dep if a <= t_dep else 0.0) + (b - a)
            s_max = max(s_max, s)
            s_arr = s if b >= t_arr else 0.0
        if not spans:
            s_arr = 0.0
        return s_arr, s_max


class _Sub:
    """A view of a RegionCells over a slice, without copying the frame arrays."""

    def __init__(self, cells, sl):
        self.origin = cells.origin[sl]
        self.up = cells.up[sl]
        self.east = cells.east[sl]
        self.north = cells.north[sl]

    def __len__(self):
        return self.origin.shape[0]
