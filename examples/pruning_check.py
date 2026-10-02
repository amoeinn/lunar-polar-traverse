"""Do the two exact prunings change any answer, and what do they buy?

PLAN.md requires both to be exact. The unit tests prove that on fixtures where
each fires; this proves it on the real region, by running the C++ engine with
and without them on the same instances and demanding identical arrivals and
identical path lengths. Then it re-measures the resolved fraction and search
time per tolerance, which is what the prunings were added for.
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
SEED = 20260924
CAP_S = 30.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-tolerance", type=int, default=40)
    ap.add_argument("--exact-only", action="store_true")
    args = ap.parse_args()
    rover = Rover(v_flat=50.0)
    p, n, px, dem, horizon, cells, gx, gy, edges, graph = P.load(rover)
    xs, ys = gx.ravel(), gy.ravel()
    t0 = time.time()
    sun = P.sun_table(P.HORIZON_H)
    intervals = LitIntervals.build(cells, horizon.astype(np.float64), sun, P.HORIZON_H)
    print(f"intervals built in {time.time()-t0:.0f} s")

    ok = legal(edges["pitch"], edges["roll"], rover)
    g_all = coo_matrix((np.ones(int(ok.sum())), (edges["a"][ok], edges["b"][ok])),
                       shape=(n * n, n * n))
    _, comp_all = connected_components(g_all, directed=False)
    positions = cell_position(gx, gy, dem, p["radius_m"]).reshape(-1, 3)
    print(f"static components of the legal graph: {comp_all.max() + 1}; "
          f"v_cap {rover.v_cap:.1f} m/hr")

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
    work = P.DATA / "work" / "plan"

    def run(pairs, tol, pruned):
        kw = dict(components=comp_all, positions=positions, v_cap=rover.v_cap) \
            if pruned else {}
        return P.run_cpp(graph, intervals, pairs, tol, 0.0, P.HORIZON_H, CAP_S,
                         work, **kw)

    # --- exactness on instances that resolve, feasible and infeasible alike ---
    print("\\n--- exactness: same instances, engine with and without the prunings ---")
    same_arrival, same_len, compared, feas, infeas = 0, 0, 0, 0, 0
    worst = 0.0
    for ti, tol in enumerate((0.5, 2.0)):
        pairs = [(a, b) for lo, hi in P.BANDS
                 for (a, b, d) in draw(lo, hi, per_band, label=100 + ti)]
        base = run(pairs, tol, False)
        pru = run(pairs, tol, True)
        both = ~base["aborted"] & ~pru["aborted"]
        for i in np.flatnonzero(both):
            compared += 1
            a1, a2 = base["arrival"][i], pru["arrival"][i]
            if np.isfinite(a1) or np.isfinite(a2):
                feas += 1
                d = abs(a1 - a2)
                worst = max(worst, d)
                same_arrival += int(d == 0.0)
                same_len += int(base["path_len"][i] == pru["path_len"][i])
            else:
                infeas += 1
                same_arrival += 1
                same_len += 1
        print(f"  tolerance {tol:>4.1f} h: {int(both.sum())} instances resolved by both; "
              f"baseline aborts {int(base['aborted'].sum())}, pruned aborts "
              f"{int(pru['aborted'].sum())}")
    print(f"  compared {compared} instances, {feas} with a path and {infeas} without")
    print(f"  worst arrival difference {worst:.3e} h; identical arrival on "
          f"{same_arrival} of {compared}, identical path length on {same_len}")
    print(f"  -> {'EXACT' if (same_arrival == compared and same_len == compared and worst == 0.0) else 'CHANGES ANSWERS'}")
    if args.exact_only:
        return 0

    # --- what the prunings bought, per tolerance ---
    print("\\n--- resolved fraction and search time per tolerance, prunings on ---")
    print(f"  {'tol':>5} {'feasible':>9} {'infeasible':>11} {'aborted':>8} "
          f"{'resolved':>9} {'median s':>9} {'mean s':>8} {'max s':>7}")
    for ti, tol in enumerate(TOLERANCES):
        pairs = [(a, b) for lo, hi in P.BANDS
                 for (a, b, d) in draw(lo, hi, per_band, label=100 + ti)]
        r = run(pairs, tol, True)
        ab = r["aborted"]
        good = ~ab & np.isfinite(r["arrival"])
        bad = ~ab & ~np.isfinite(r["arrival"])
        print(f"  {tol:>5.1f} {int(good.sum()):>9} {int(bad.sum()):>11} "
              f"{int(ab.sum()):>8} {int((~ab).sum()):>9} "
              f"{np.median(r['seconds']):>9.3f} {r['seconds'].mean():>8.3f} "
              f"{r['seconds'].max():>7.1f}")
    print(f"  cap {CAP_S:.0f} s; aborted is a third outcome and is never folded into "
          f"the other two")
    return 0


if __name__ == "__main__":
    sys.exit(main())
