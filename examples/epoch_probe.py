"""Cost of the time-dependent search at one epoch, for re-estimating the sweep.

The pilots all ran at 2028-03-15, which the epoch summary shows is the most
static epoch of the twelve. This measures the search cost at a named epoch so
the sweep estimate rests on a representative one.
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
from examples.build_intervals import HORIZON_H, cache_path       # noqa: E402
from src.cost import Rover, legal                                # noqa: E402
from src.frames import cell_position                             # noqa: E402
from src.intervals import LitIntervals                           # noqa: E402

SEED = 20261001
CAP_S = 30.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--instances", type=int, default=20)
    ap.add_argument("--tolerance", type=float, default=2.0)
    args = ap.parse_args()
    rover = Rover(v_flat=50.0)
    p, n, px, dem, horizon, cells, gx, gy, edges, graph = P.load(rover)
    xs, ys = gx.ravel(), gy.ravel()
    iv = LitIntervals.load(cache_path(args.tag))
    ok = legal(edges["pitch"], edges["roll"], rover)
    g_all = coo_matrix((np.ones(int(ok.sum())), (edges["a"][ok], edges["b"][ok])),
                       shape=(n * n, n * n))
    _, comp_all = connected_components(g_all, directed=False)
    positions = cell_position(gx, gy, dem, p["radius_m"]).reshape(-1, 3)
    changing = np.diff(iv.offsets) >= 2
    sel = ok & changing[edges["a"]] & changing[edges["b"]]
    gc = coo_matrix((np.ones(int(sel.sum())), (edges["a"][sel], edges["b"][sel])),
                    shape=(n * n, n * n))
    _, lab = connected_components(gc, directed=False)
    sz = np.bincount(lab)
    pool = np.flatnonzero(lab == np.argsort(sz)[::-1][0])
    print(f"{args.tag}: pool {pool.size} cells ({pool.size/(n*n):.2%}), "
          f"{np.diff(iv.offsets).mean():.3f} transitions per cell, "
          f"lit at t0 {iv.lit0.mean():.2%}")

    per_band = max(args.instances // len(P.BANDS), 1)
    pairs, dists = [], []
    for bi, (lo, hi) in enumerate(P.BANDS):
        rng = np.random.default_rng([SEED, bi, int(lo), int(hi)])
        got = 0
        while got < per_band:
            a, b = rng.choice(pool, 2, replace=False)
            d = np.hypot(xs[a] - xs[b], ys[a] - ys[b])
            if lo <= d < hi:
                pairs.append((int(a), int(b)))
                dists.append(float(d))
                got += 1
    t0 = time.time()
    r = P.run_cpp(graph, iv, pairs, args.tolerance, 0.0, HORIZON_H, CAP_S,
                  P.DATA / "work" / "plan", components=comp_all, positions=positions,
                  v_cap=rover.v_cap)
    wall = time.time() - t0
    ab = r["aborted"]
    good = ~ab & np.isfinite(r["arrival"])
    bad = ~ab & ~np.isfinite(r["arrival"])
    print(f"  tolerance {args.tolerance:.1f} h, cap {CAP_S:.0f} s, {len(pairs)} instances "
          f"in {wall:.0f} s")
    print(f"  feasible {int(good.sum())}, infeasible {int(bad.sum())}, "
          f"aborted {int(ab.sum())}")
    print(f"  seconds per search: median {np.median(r['seconds']):.3f}, "
          f"mean {r['seconds'].mean():.3f}, max {r['seconds'].max():.1f}")
    if good.any():
        arr = r["arrival"][good]
        print(f"  arrivals, h: median {np.median(arr):.2f}, min {arr.min():.2f}, "
              f"max {arr.max():.2f}")
    print(f"  labels on the worst cell: max {int(r['stats'][:,3].max())}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
