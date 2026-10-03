"""Piece 3 of the sweep: the axes, one parameter varied at a time from the core.

Four epochs chosen before the run and registered in PLAN.md: one per date so the
seasonal axis is spanned, with lit fractions at departure from 0.00% to 86.65%
so both the solvable and the unsolvable phase are represented. 40 pairs per
epoch, 160 instances per point.

Eight points: rates 20, 100, 200 m/hr; tolerances 0.5, 6, 24 h; limit pairs
(15, 10) and (25, 20). Run one axis at a time, which is the registered split.
A ninth, --axis rate_core, is the core configuration on this draw: P7's
intersection rule needs all four rates on one instance set and the core sweep's
50 m/hr point was drawn differently.

Scores nothing. Every plan is judged twice, at the registered 3.206 deg pitch
and roll tolerance and at zero.
"""
import argparse
import subprocess
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
from examples.replay_batch import FINE_SUN_DT_H, fine_sun        # noqa: E402
from src.baselines import frozen_intervals, plan_mean_map        # noqa: E402
from src.cost import Rover, legal                                # noqa: E402
from src.frames import cell_position                             # noqa: E402
from src.intervals import LitIntervals                           # noqa: E402
from src.planner import Graph                                    # noqa: E402
from src.replay import ReplaySettings, replay                    # noqa: E402

SEED = 20260924
CAP_S = 30.0
EPOCHS = ("2028-02-15_p0", "2028-05-15_p1", "2028-08-15_p2", "2028-11-15_p2")
EXPONENTS = (0.0, 0.5, 1.0, 2.0)
PITCH_ROLL_TOL_DEG = 3.206
ARRIVAL_CRITERION = 0.01
PAIRS_PER_BAND = 8
CORE = dict(v_flat=50.0, tol=2.0, pitch_max=20.0, roll_max=15.0)
AXES = {
    "rate": [dict(CORE, v_flat=v, label=f"rate {v:.0f}") for v in (20.0, 100.0, 200.0)],
    "tolerance": [dict(CORE, tol=s, label=f"tol {s:g} h") for s in (0.5, 6.0, 24.0)],
    "limits": [dict(CORE, pitch_max=a, roll_max=b, label=f"limits ({a:.0f},{b:.0f})")
               for a, b in ((15.0, 10.0), (25.0, 20.0))],
    # The core configuration itself, on the axis draw. P7's registered
    # intersection rule needs all four rates on one instance set, and the core
    # sweep's 50 m/hr point is a different draw: the per-epoch random key is the
    # epoch's index in its own list, 0, 1, 2, 3 here against 0, 4, 8, 11 there,
    # so the two coincide at 2028-02-15_p0 and share nothing at the other three.
    "rate_core": [dict(CORE, label="rate 50")],
}


def draw(pool, xs, ys, label):
    out = []
    for bi, (lo, hi) in enumerate(P.BANDS):
        rng = np.random.default_rng([SEED, int(label), bi, int(lo), int(hi)])
        got = 0
        while got < PAIRS_PER_BAND:
            a, b = rng.choice(pool, 2, replace=False)
            d = np.hypot(xs[a] - xs[b], ys[a] - ys[b])
            if lo <= d < hi:
                out.append((int(a), int(b)))
                got += 1
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--axis", required=True, choices=sorted(AXES))
    ap.add_argument("--jobs", type=int, default=12)
    args = ap.parse_args()
    base = Rover(v_flat=50.0)
    p, n, px, dem, horizon, cells, gx, gy, edges, _ = P.load(base)
    xs, ys = gx.ravel(), gy.ravel()
    positions = cell_position(gx, gy, dem, p["radius_m"]).reshape(-1, 3)
    grid_geom = dict(radius_m=p["radius_m"], region_ox=p["region_ox"],
                     region_oy=p["region_oy"], region_px=p["region_px"],
                     region_nx=p["region_nx"], near_path=p["near_path"],
                     near_ox=p["near_ox"], near_oy=p["near_oy"],
                     near_px=p["near_px"], near_width=p["near_width"],
                     near_height=p["near_height"], far_path=p["far_path"],
                     far_ox=p["far_ox"], far_oy=p["far_oy"], far_px=p["far_px"],
                     far_width=p["far_width"], far_height=p["far_height"],
                     reach_m=p["reach_m"], switch_m=p["switch_m"],
                     near_step_m=p["near_step_m"], far_step_m=p["far_step_m"])

    ivs = {tag: LitIntervals.load(cache_path(tag)) for tag in EPOCHS}
    lits = {tag: ivs[tag].lit_fraction() for tag in EPOCHS}
    print(f"axis {args.axis}: {len(AXES[args.axis])} points, {len(EPOCHS)} epochs, "
          f"{PAIRS_PER_BAND*len(P.BANDS)} pairs per epoch, cap {CAP_S:.0f} s")
    for tag in EPOCHS:
        print(f"  {tag:<16} lit at t0 {ivs[tag].lit0.mean():>7.2%}, "
              f"mean lit fraction {lits[tag].mean():.3f}")

    # graphs and draw pools depend on the limits only
    cache_graph = {}

    def graph_for(rover):
        key = (rover.pitch_max, rover.roll_max, rover.v_flat)
        if key not in cache_graph:
            cache_graph[key] = Graph.build(edges, rover, n * n)
        return cache_graph[key]

    cache_pool = {}

    def pool_for(rover, tag):
        key = (rover.pitch_max, rover.roll_max, tag)
        if key not in cache_pool:
            ok = legal(edges["pitch"], edges["roll"], rover)
            changing = np.diff(ivs[tag].offsets) >= 2
            sel = ok & changing[edges["a"]] & changing[edges["b"]]
            gc = coo_matrix((np.ones(int(sel.sum())),
                             (edges["a"][sel], edges["b"][sel])), shape=(n * n,) * 2)
            _, lab = connected_components(gc, directed=False)
            sz = np.bincount(lab)
            pool = np.flatnonzero(lab == np.argsort(sz)[::-1][0])
            g_all = coo_matrix((np.ones(int(ok.sum())),
                                (edges["a"][ok], edges["b"][ok])), shape=(n * n,) * 2)
            _, comp = connected_components(g_all, directed=False)
            cache_pool[key] = (pool, comp)
        return cache_pool[key]

    # --- stage 1: launch every planner job for this axis ---
    jobs, meta = [], {}
    for pi, point in enumerate(AXES[args.axis]):
        rover = Rover(v_flat=point["v_flat"], pitch_max=point["pitch_max"],
                      roll_max=point["roll_max"])
        graph = graph_for(rover)
        for ei, tag in enumerate(EPOCHS):
            pool, comp = pool_for(rover, tag)
            pairs = draw(pool, xs, ys, label=ei)
            froz = frozen_intervals(ivs[tag], 0.0)
            for kind, table in (("td", ivs[tag]), ("snap", froz)):
                work = P.DATA / "work" / "axes" / f"{args.axis}_{pi}_{tag}_{kind}"
                cmd, params = P.run_cpp(graph, table, pairs, point["tol"], 0.0,
                                        HORIZON_H, CAP_S, work, components=comp,
                                        positions=positions, v_cap=rover.v_cap,
                                        want_paths=True, defer=True)
                jobs.append((pi, tag, kind, cmd, params))
            meta[(pi, tag)] = dict(pairs=pairs, rover=rover, graph=graph, point=point)
    # Resume rather than restart: a job whose outputs are already on disk is
    # read back instead of rerun. A sweep piece that outlives its session
    # otherwise loses everything it had finished.
    done = [(pi, tag, kind, cmd, params) for (pi, tag, kind, cmd, params) in jobs
            if Path(params["out_arrival"]).exists()]
    todo = [j for j in jobs if j not in done]
    print(f"\nprepared {len(jobs)} planner jobs; {len(done)} already on disk, "
          f"{len(todo)} to run, launching {args.jobs} at a time")
    t0 = time.time()
    running, results, queue = [], {}, list(todo)
    for pi, tag, kind, cmd, params in done:
        results[(pi, tag, kind)] = P.read_cpp(params, want_paths=True)
    while queue or running:
        while queue and len(running) < args.jobs:
            pi, tag, kind, cmd, params = queue.pop(0)
            running.append((pi, tag, kind, params,
                            subprocess.Popen(cmd, stdout=subprocess.DEVNULL)))
        time.sleep(0.5)
        still = []
        for pi, tag, kind, params, proc in running:
            if proc.poll() is None:
                still.append((pi, tag, kind, params, proc))
            else:
                results[(pi, tag, kind)] = P.read_cpp(params, want_paths=True)
        running = still
    print(f"planner stage done in {time.time()-t0:.0f} s")

    # --- stage 2: mean-map. Its route depends on the graph and the lit fraction
    # only, so on the rate and tolerance axes it is the core's route reused.
    t0 = time.time()
    mm = {}
    for pi, point in enumerate(AXES[args.axis]):
        for tag in EPOCHS:
            m = meta[(pi, tag)]
            # The route is invariant to v_flat by the homogeneity property, but
            # the traverse time this returns is not, so the rate is part of the
            # key. Caching on the limits alone reported the first rate's times
            # for every rate.
            key = (m["rover"].pitch_max, m["rover"].roll_max, m["rover"].v_flat, tag)
            if key not in mm:
                mm[key] = {e: [plan_mean_map(m["graph"], lits[tag], a, b, exponent=e)
                               for (a, b) in m["pairs"]] for e in EXPONENTS}
    print(f"mean-map stage done in {time.time()-t0:.0f} s, {len(mm)} distinct "
          f"(graph, epoch) combinations")

    # --- stage 3: replay, judged twice, per point because timing and verdict
    # both depend on the point ---
    t0 = time.time()
    rows = []
    for pi, point in enumerate(AXES[args.axis]):
        for tag in EPOCHS:
            m = meta[(pi, tag)]
            rover, graph = m["rover"], m["graph"]
            suns = fine_sun(tag)
            work_base = P.DATA / "work" / "axes" / f"{args.axis}_{pi}_{tag}_replay"
            key = (rover.pitch_max, rover.roll_max, rover.v_flat, tag)
            sources = {}
            for kind in ("td", "snap"):
                r = results[(pi, tag, kind)]
                off, pc, pt = r["path_offsets"], r["path_cells"], r["path_times"]
                plans, starts, waits, idx = [], [], [], []
                for q in range(len(m["pairs"])):
                    lo, hi = off[q], off[q + 1]
                    if hi - lo < 2:
                        continue
                    cpath, tpath = pc[lo:hi].astype(np.int64), pt[lo:hi]
                    w = np.zeros(cpath.size)
                    for i in range(cpath.size - 1):
                        u, v = int(cpath[i]), int(cpath[i + 1])
                        s0, s1 = graph.indptr[u], graph.indptr[u + 1]
                        kk = np.flatnonzero(graph.indices[s0:s1] == v)
                        w[i] = max(tpath[i + 1]
                                   - float(graph.travel[s0:s1][kk[0]]) - tpath[i], 0.0)
                    plans.append(cpath); starts.append(float(tpath[0]))
                    waits.append(w); idx.append(q)
                sources[kind] = (plans, starts, waits, idx,
                                 np.array([r["arrival"][q] for q in idx]))
            for e in EXPONENTS:
                plans, starts, waits, idx, claimed = [], [], [], [], []
                for q, (tt, path) in enumerate(mm[key][e]):
                    if not path or len(path) < 2:
                        continue
                    plans.append(np.asarray(path, dtype=np.int64)); starts.append(0.0)
                    waits.append(np.zeros(len(path))); idx.append(q); claimed.append(tt)
                sources[f"mm{e}"] = (plans, starts, waits, idx, np.array(claimed))
            for kind, (plans, starts, waits, idx, claimed) in sources.items():
                if not plans:
                    rows.append((point["label"], tag, kind, 0, 0, 0, 0, 0))
                    continue
                out = {}
                for lbl, tol in (("tol", PITCH_ROLL_TOL_DEG), ("zero", 0.0)):
                    out[lbl] = replay(plans, starts, grid_geom, rover, suns,
                                      FINE_SUN_DT_H, point["tol"],
                                      Path(f"{work_base}_{kind}_{lbl}"),
                                      ReplaySettings(limit_tolerance_deg=tol),
                                      waits=waits, reuse=True)
                a, z = out["tol"], out["zero"]
                near = 0
                for i in range(len(plans)):
                    lo, hi = a["step_offsets"][i], a["step_offsets"][i + 1]
                    wp = np.abs(a["step_pitch"][lo:hi].astype(float))
                    wr = np.abs(a["step_roll"][lo:hi].astype(float))
                    near += int(np.sum((wp > rover.pitch_max - PITCH_ROLL_TOL_DEG)
                                       | (wr > rover.roll_max - PITCH_ROLL_TOL_DEG)))
                rel = np.abs(a["arrival"] - claimed) / np.maximum(claimed, 1e-9)
                rows.append((point["label"], tag, kind, len(plans),
                             int(a["verdict"].sum()), int(z["verdict"].sum()), near,
                             int((rel <= ARRIVAL_CRITERION).sum())))
    print(f"replay stage done in {time.time()-t0:.0f} s")

    print(f"\n=== {args.axis} axis: planner outcomes per epoch "
          f"(feasible / infeasible / aborted), cap {CAP_S:.0f} s ===")
    print(f"  {'point':<16}{'epoch':<16}{'time-dependent':>18}{'snapshot':>18}")
    for pi, point in enumerate(AXES[args.axis]):
        for tag in EPOCHS:
            cols = []
            for kind in ("td", "snap"):
                r = results[(pi, tag, kind)]
                ab = r["aborted"]
                cols.append(f"{int((~ab & np.isfinite(r['arrival'])).sum()):4d} /"
                            f"{int((~ab & ~np.isfinite(r['arrival'])).sum()):4d} /"
                            f"{int(ab.sum()):4d}")
            print(f"  {point['label']:<16}{tag:<16}{cols[0]:>18}{cols[1]:>18}")
        tot = []
        for kind in ("td", "snap"):
            f = sum(int((~results[(pi, tg, kind)]["aborted"]
                         & np.isfinite(results[(pi, tg, kind)]["arrival"])).sum())
                    for tg in EPOCHS)
            i = sum(int((~results[(pi, tg, kind)]["aborted"]
                         & ~np.isfinite(results[(pi, tg, kind)]["arrival"])).sum())
                    for tg in EPOCHS)
            a = sum(int(results[(pi, tg, kind)]["aborted"].sum()) for tg in EPOCHS)
            tot.append(f"{f:4d} /{i:4d} /{a:4d}")
        print(f"  {point['label']:<16}{'point total':<16}{tot[0]:>18}{tot[1]:>18}")

    print(f"\n=== {args.axis} axis: plans and replay verdicts ===")
    print(f"  {'point':<16}{'planner':<8}{'plans':>6}{'pass@tol':>9}{'pass@0':>7}"
          f"{'cost':>6}{'nearlim':>8}{'arr<=1%':>9}")
    for point in AXES[args.axis]:
        for kind in ["td", "snap"] + [f"mm{e}" for e in EXPONENTS]:
            r = [x for x in rows if x[0] == point["label"] and x[2] == kind]
            if not r:
                continue
            pl = sum(x[3] for x in r)
            print(f"  {point['label']:<16}{kind:<8}{pl:>6}{sum(x[4] for x in r):>9}"
                  f"{sum(x[5] for x in r):>7}"
                  f"{sum(x[4] for x in r)-sum(x[5] for x in r):>6}"
                  f"{sum(x[6] for x in r):>8}{sum(x[7] for x in r):>9}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
