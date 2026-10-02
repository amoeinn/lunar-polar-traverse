"""Does quantising the shadow resource upward remove the abort censoring?

PLAN.md registers the grid values 1, 60, 300 and 900 s before this ran, and
requires all four to be reported whatever they give: aborts and search time per
tolerance on the same 40-instance draws, and the cost in answers against the
exact engine on the instances both resolve.

Also measured, and labelled as such rather than registered: relaxing the
dominance rule by dropping its reachability clause. Quantisation alone cannot
make dominance fire at a dark cell, because what blocks it there is the
time-reachability clause and not the granularity of s -- measured before either
was implemented. Dropping the clause is pessimistic in the same direction, so
it is offered as an option with its cost, not adopted.
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import examples.plan_pilot as P                                  # noqa: E402
from src.cost import Rover, legal                                # noqa: E402
from src.frames import cell_position                             # noqa: E402
from src.intervals import LitIntervals                           # noqa: E402

TOLERANCES = (0.5, 2.0, 6.0, 24.0)
GRIDS_S = (1.0, 60.0, 300.0, 900.0)        # registered 2026-09-30
SEED = 20260924
CAP_S = 8.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-tolerance", type=int, default=40)
    args = ap.parse_args()
    rover = Rover(v_flat=50.0)
    p, n, px, dem, horizon, cells, gx, gy, edges, graph = P.load(rover)
    xs, ys = gx.ravel(), gy.ravel()
    sun = P.sun_table(P.HORIZON_H)
    cache = (P.DATA / "work" / "intervals"
             / f"{P.EPOCH.replace(':', '')}_{P.HORIZON_H:.0f}h.npz")
    t0 = time.time()
    intervals, cached = LitIntervals.build_or_load(cache, cells,
                                                   horizon.astype(np.float64), sun,
                                                   P.HORIZON_H)
    print(f"intervals {'loaded from cache' if cached else 'built and cached'} in "
          f"{time.time()-t0:.1f} s")

    ok = legal(edges["pitch"], edges["roll"], rover)
    g_all = coo_matrix((np.ones(int(ok.sum())), (edges["a"][ok], edges["b"][ok])),
                       shape=(n * n, n * n))
    _, comp_all = connected_components(g_all, directed=False)
    positions = cell_position(gx, gy, dem, p["radius_m"]).reshape(-1, 3)
    changing = np.diff(intervals.offsets) >= 2
    sel = ok & changing[edges["a"]] & changing[edges["b"]]
    gc = coo_matrix((np.ones(int(sel.sum())), (edges["a"][sel], edges["b"][sel])),
                    shape=(n * n, n * n))
    _, labc = connected_components(gc, directed=False)
    sz = np.bincount(labc)
    pool = np.flatnonzero(labc == np.argsort(sz)[::-1][0])

    def draw(lo, hi, k, label):
        rng = np.random.default_rng([SEED, int(label), int(lo), int(hi)])
        out = []
        while len(out) < k:
            a, b = rng.choice(pool, 2, replace=False)
            d = np.hypot(xs[a] - xs[b], ys[a] - ys[b])
            if lo <= d < hi:
                out.append((int(a), int(b), float(d)))
        return out

    per_band = max(args.per_tolerance // len(P.BANDS), 1)
    pairs_by_tol = {}
    for ti, tol in enumerate(TOLERANCES):
        pairs_by_tol[tol] = [(a, b) for lo, hi in P.BANDS
                             for (a, b, d) in draw(lo, hi, per_band, label=100 + ti)]

    work = P.DATA / "work" / "plan"

    def run(pairs, tol, grid_s=0.0, relax=False):
        return P.run_cpp(graph, intervals, pairs, tol, 0.0, P.HORIZON_H, CAP_S, work,
                         components=comp_all, positions=positions, v_cap=rover.v_cap,
                         shadow_grid_h=grid_s / 3600.0, relax_dominance=relax)

    configs = [("exact", 0.0, False)]
    configs += [(f"grid {g:.0f} s", g, False) for g in GRIDS_S]
    configs += [("relaxed dominance", 0.0, True)]

    print(f"\nall configurations at the same {CAP_S:.0f} s cap, same draws, "
          f"prunings on")
    print(f"  {'configuration':<20}{'tol':>6}{'feas':>6}{'infeas':>8}{'abort':>7}"
          f"{'median s':>10}{'mean s':>9}")
    results = {}
    for name, grid, relax in configs:
        for tol in TOLERANCES:
            r = run(pairs_by_tol[tol], tol, grid, relax)
            results[(name, tol)] = r
            ab = r["aborted"]
            good = ~ab & np.isfinite(r["arrival"])
            bad = ~ab & ~np.isfinite(r["arrival"])
            print(f"  {name:<20}{tol:>6.1f}{int(good.sum()):>6}{int(bad.sum()):>8}"
                  f"{int(ab.sum()):>7}{np.median(r['seconds']):>10.3f}"
                  f"{r['seconds'].mean():>9.3f}")

    print("\ncost in answers, each configuration against exact, on the instances")
    print("both resolve (arrivals changed, paths changed, feasible-to-infeasible flips):")
    print(f"  {'configuration':<20}{'compared':>9}{'arr changed':>12}"
          f"{'path changed':>13}{'feas->infeas':>13}{'infeas->feas':>13}")
    for name, grid, relax in configs[1:]:
        comp, arrc, pathc, lost, gained = 0, 0, 0, 0, 0
        for tol in TOLERANCES:
            e, q = results[("exact", tol)], results[(name, tol)]
            both = ~e["aborted"] & ~q["aborted"]
            for i in np.flatnonzero(both):
                comp += 1
                fe, fq = np.isfinite(e["arrival"][i]), np.isfinite(q["arrival"][i])
                if fe and fq:
                    if e["arrival"][i] != q["arrival"][i]:
                        arrc += 1
                    if e["path_len"][i] != q["path_len"][i]:
                        pathc += 1
                elif fe and not fq:
                    lost += 1
                elif fq and not fe:
                    gained += 1
        print(f"  {name:<20}{comp:>9}{arrc:>12}{pathc:>13}{lost:>13}{gained:>13}")
    print("  a nonzero infeas->feas column would mean the approximation invented a")
    print("  plan, which neither is allowed to do; it must read zero")
    return 0


if __name__ == "__main__":
    sys.exit(main())
