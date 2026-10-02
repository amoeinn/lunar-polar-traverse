"""Strengthened C++/Python cross-check, on instances where a path exists.

The first cross-check met its tolerance on sixteen instances of which fifteen
were agreements that no path exists -- which a bug shared by a port and its
original would also produce. PLAN.md registers the strengthened criterion:
at least ten instances with a path, found by sweeping the shadow tolerance and
not by selecting instances that agree, and on every one identical arrival to
1e-9 h and identical path length.

The tolerance sweep it uses is the registered one, 0.5, 2, 6 and 24 h, and the
feasible fraction and the distance each tolerance buys are reported with it.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import examples.plan_pilot as P                                    # noqa: E402
from src.cost import Rover, edge_time, legal                       # noqa: E402
from src.intervals import LitIntervals                             # noqa: E402
from src.planner import plan                                       # noqa: E402

TOLERANCES = (0.5, 2.0, 6.0, 24.0)        # registered 2026-09-24
RATES = (20.0, 50.0, 100.0, 200.0)
SEED = 20260924
BANDS = P.BANDS
NEED_FEASIBLE = 10
CPP_LIMIT_S = 30.0    # bounds the sweep at 4 x 40 x 30 s worst case
PY_LIMIT_S = 300.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-tolerance", type=int, default=40)
    args = ap.parse_args()
    rover = Rover(v_flat=50.0)
    p, n, px, dem, horizon, cells, gx, gy, edges, graph = P.load(rover)
    xs, ys = gx.ravel(), gy.ravel()
    print(f"region {n}x{n}, commit {p['source_commit']}; graph {graph.indices.size} "
          f"legal directed edges at limits ({rover.pitch_max:.0f}, {rover.roll_max:.0f})")

    t0 = time.time()
    sun = P.sun_table(P.HORIZON_H)
    intervals = LitIntervals.build(cells, horizon.astype(np.float64), sun, P.HORIZON_H)
    print(f"lit intervals over {P.HORIZON_H:.0f} h: {time.time()-t0:.0f} s, "
          f"{np.diff(intervals.offsets).mean():.2f} transitions per cell")

    ok = legal(edges["pitch"], edges["roll"], rover)
    changing = np.diff(intervals.offsets) >= 2
    sel = ok & changing[edges["a"]] & changing[edges["b"]]
    gc = coo_matrix((np.ones(int(sel.sum())), (edges["a"][sel], edges["b"][sel])),
                    shape=(n * n, n * n))
    _, lab = connected_components(gc, directed=False)
    sz = np.bincount(lab)
    pool = np.flatnonzero(lab == np.argsort(sz)[::-1][0])
    print(f"draw pool, the superseded rule: {pool.size} cells ({pool.size/(n*n):.2%})")

    # The effective speed, from the measured edge times rather than from v_flat.
    t_edge = edge_time(edges["length"][ok], edges["pitch"][ok], edges["roll"][ok], rover)
    eff50 = float(np.median(edges["length"][ok] / t_edge))
    print(f"median legal-edge speed at v_flat 50 m/hr: {eff50:.2f} m/hr "
          f"({eff50/50.0:.2f} of nominal)")
    print("\ndistance each tolerance buys, from that median speed:")
    header = "  tolerance " + "".join(f"{v:>10.0f} m/hr" for v in RATES)
    print(header)
    for tol in TOLERANCES:
        row = "".join(f"{tol * eff50 * v / 50.0:>13.0f}" for v in RATES)
        print(f"  {tol:>7.1f} h" + row + "   metres of darkness")

    def draw(lo, hi, k, label):
        rng = np.random.default_rng([SEED, int(label), int(lo), int(hi)])
        out = []
        while len(out) < k:
            a, b = rng.choice(pool, 2, replace=False)
            d = np.hypot(xs[a] - xs[b], ys[a] - ys[b])
            if lo <= d < hi:
                out.append((int(a), int(b), float(d)))
        return out

    per_band = max(args.per_tolerance // len(BANDS), 1)
    print(f"\nfeasible fraction by tolerance, {per_band * len(BANDS)} instances each, "
          f"drawn independently per tolerance:")
    feasible = []
    for ti, tol in enumerate(TOLERANCES):
        pairs, dists = [], []
        for lo, hi in BANDS:
            for (a, b, d) in draw(lo, hi, per_band, label=100 + ti):
                pairs.append((a, b))
                dists.append(d)
        res = P.run_cpp(graph, intervals, pairs, tol, 0.0, P.HORIZON_H, CPP_LIMIT_S,
                        P.DATA / "work" / "plan")
        good = ~res["aborted"] & np.isfinite(res["arrival"])
        resolved = int((~res["aborted"]).sum())
        print(f"  {tol:>5.1f} h: {int(good.sum()):3d} feasible of {resolved} resolved "
              f"({good.sum()/max(resolved,1):6.2%}) and of {len(pairs)} drawn "
              f"({good.mean():6.2%}); {int(res['aborted'].sum())} aborted at "
              f"{CPP_LIMIT_S:.0f} s, so the fraction over drawn is censored from below; "
              f"c++ seconds median {np.median(res['seconds']):.3f} "
              f"max {res['seconds'].max():.1f}")
        for i in np.flatnonzero(good):
            feasible.append(dict(tol=tol, a=pairs[i][0], b=pairs[i][1], d=dists[i],
                                 arrival=float(res["arrival"][i]),
                                 path_len=int(res["path_len"][i]),
                                 seconds=float(res["seconds"][i]),
                                 touched=int(res["stats"][i, 2]),
                                 max_labels=int(res["stats"][i, 3])))

    print(f"\n{len(feasible)} feasible instances found; the cross-check takes them in "
          f"tolerance then draw order,\nwhich is fixed in advance and does not depend on "
          f"whether the two engines agree")
    if len(feasible) < NEED_FEASIBLE:
        print(f"  FEWER THAN {NEED_FEASIBLE}: the criterion cannot be met on this draw")
    use = feasible[:max(NEED_FEASIBLE, min(len(feasible), 20))]

    print(f"\ncross-check on {len(use)} feasible instances, registered criterion: "
          f"identical arrival\nto 1e-9 h and identical path length on every one")
    print(f"  {'tol':>5} {'dist':>7} {'c++ arrival':>13} {'py arrival':>13} "
          f"{'diff':>10} {'len':>5} {'py len':>7} {'py s':>8} {'labels max':>11}")
    worst_t, same_len, aborts = 0.0, 0, 0
    lab_stats = []
    for f in use:
        t1 = time.time()
        arrival, path, st = plan(graph, intervals, f["a"], f["b"], 0.0, f["tol"],
                                 time_limit_s=PY_LIMIT_S)
        secs = time.time() - t1
        if st.timed_out:
            aborts += 1
            print(f"  {f['tol']:>5.1f} {f['d']:>7.0f} {f['arrival']:>13.6f} "
                  f"{'aborted':>13} {'-':>10} {f['path_len']:>5} {'-':>7} {secs:>8.1f}")
            continue
        diff = abs((arrival if arrival is not None else np.nan) - f["arrival"])
        worst_t = max(worst_t, diff)
        same_len += int(len(path) == f["path_len"])
        lpc = np.array(list(st.labels_per_cell.values()))
        lab_stats.append(lpc)
        print(f"  {f['tol']:>5.1f} {f['d']:>7.0f} {f['arrival']:>13.6f} "
              f"{arrival:>13.6f} {diff:>10.2e} {f['path_len']:>5} {len(path):>7} "
              f"{secs:>8.2f} {int(lpc.max()):>11}")
    compared = len(use) - aborts
    print(f"\n  compared {compared} of {len(use)}; {aborts} aborted in Python at "
          f"{PY_LIMIT_S:.0f} s")
    print(f"  worst arrival difference {worst_t:.3e} h; identical path length on "
          f"{same_len} of {compared}")
    verdict = (compared >= NEED_FEASIBLE and worst_t <= 1e-9 and same_len == compared)
    print(f"  -> {'AGREE' if verdict else 'DOES NOT MEET THE CRITERION'} "
          f"(needs at least {NEED_FEASIBLE} feasible, arrival within 1e-9 h, "
          f"identical path length on every one)")

    if lab_stats:
        allv = np.concatenate(lab_stats)
        print(f"\n  labels per cell on the feasible instances, {allv.size} cells over "
              f"{len(lab_stats)} searches:")
        print(f"    median {np.median(allv):.1f}, p90 {np.percentile(allv, 90):.1f}, "
              f"p99 {np.percentile(allv, 99):.1f}, max {allv.max()}")
        print(f"    per-instance worst cell: median {np.median([v.max() for v in lab_stats]):.0f}, "
              f"max {max(v.max() for v in lab_stats)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
