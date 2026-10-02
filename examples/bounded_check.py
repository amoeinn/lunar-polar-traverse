"""The bounded measurement of the relaxed-dominance variant.

PLAN.md registers the sample rule, the agreement criterion and the fallback
before this ran. The sample is deliberately drawn from the instances the exact
engine aborted on at the 8 s cap, because that is the population whose cost is
unmeasured; the exact engine is then given 300 s on them.
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

SEED = 20260924
SCREEN_CAP_S = 8.0        # the cap the registered sample rule refers to
BIG_CAP_S = 300.0
SAMPLES = ((6.0, 2, 5), (24.0, 3, 3))      # tolerance, draw label offset, how many


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
    intervals, cached = LitIntervals.build_or_load(cache, cells,
                                                   horizon.astype(np.float64), sun,
                                                   P.HORIZON_H)
    print(f"intervals {'cached' if cached else 'built'}; horizon {P.HORIZON_H:.0f} h")

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
    work = P.DATA / "work" / "plan"

    def run(pairs, tol, cap, relax):
        return P.run_cpp(graph, intervals, pairs, tol, 0.0, P.HORIZON_H, cap, work,
                         components=comp_all, positions=positions, v_cap=rover.v_cap,
                         relax_dominance=relax)

    verdicts = []
    for tol, label_off, want in SAMPLES:
        drawn = [(a, b, d) for lo, hi in P.BANDS
                 for (a, b, d) in draw(lo, hi, per_band, label=100 + label_off)]
        pairs = [(a, b) for a, b, _ in drawn]
        t0 = time.time()
        screen = run(pairs, tol, SCREEN_CAP_S, False)
        hard = [i for i in range(len(pairs)) if screen["aborted"][i]][:want]
        print(f"\n--- tolerance {tol:.1f} h: screened {len(pairs)} at {SCREEN_CAP_S:.0f} s "
              f"in {time.time()-t0:.0f} s; {int(screen['aborted'].sum())} aborted, "
              f"taking the first {len(hard)} in draw order ---")
        sub = [pairs[i] for i in hard]
        dists = [drawn[i][2] for i in hard]
        t0 = time.time()
        ex = run(sub, tol, BIG_CAP_S, False)
        print(f"  exact at {BIG_CAP_S:.0f} s took {time.time()-t0:.0f} s")
        rx = run(sub, tol, BIG_CAP_S, True)

        print(f"  {'dist':>7} {'exact':>14} {'exact s':>9} {'relaxed':>14} "
              f"{'relax s':>9}  note")
        for k in range(len(sub)):
            def tag(r, i):
                if r["aborted"][i]:
                    return "aborted"
                return "no path" if not np.isfinite(r["arrival"][i]) \
                    else f"{r['arrival'][i]:.6f} h"
            fe = (not ex["aborted"][k]) and np.isfinite(ex["arrival"][k])
            fr = (not rx["aborted"][k]) and np.isfinite(rx["arrival"][k])
            note = ""
            if ex["aborted"][k]:
                note = "exact unresolved: nothing to compare"
            elif fr and not fe:
                note = "INVENTED A PLAN"
            elif fe and not fr:
                note = "cost: feasible plan lost"
            elif fe and fr:
                note = ("identical arrival" if ex["arrival"][k] == rx["arrival"][k]
                        else f"ARRIVAL DIFFERS by {abs(ex['arrival'][k]-rx['arrival'][k]):.3e} h")
            else:
                note = "both agree there is no path"
            print(f"  {dists[k]:>7.0f} {tag(ex, k):>14} {ex['seconds'][k]:>9.2f} "
                  f"{tag(rx, k):>14} {rx['seconds'][k]:>9.3f}  {note}")
            verdicts.append((tol, note))

    comparable = [v for v in verdicts if not v[1].startswith("exact unresolved")]
    invented = [v for v in verdicts if v[1] == "INVENTED A PLAN"]
    differs = [v for v in verdicts if v[1].startswith("ARRIVAL DIFFERS")]
    lost = [v for v in verdicts if v[1].startswith("cost:")]
    print(f"\nsample {len(verdicts)} instances; {len(comparable)} comparable, "
          f"{len(verdicts)-len(comparable)} still unresolved by the exact engine at "
          f"{BIG_CAP_S:.0f} s")
    print(f"  infeasible-to-feasible flips (would invent a plan): {len(invented)}")
    print(f"  arrivals differing where both found a path: {len(differs)}")
    print(f"  feasible plans lost, the approximation's measured cost: {len(lost)}")
    # A criterion evaluated over an empty set is not met, it is vacuous, and
    # saying MET there is the same fault as a PASS on zero instances.
    if not comparable:
        print("  -> agreement criterion: VACUOUS. Zero comparable instances, so "
              "nothing was tested;\n     the registered fallback applies.")
    else:
        agree = (len(invented) == 0 and len(differs) == 0)
        print(f"  -> agreement criterion: {'MET' if agree else 'FAILED'} on the "
              f"{len(comparable)} comparable instances")
    return 0


if __name__ == "__main__":
    sys.exit(main())
