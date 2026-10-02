"""Piece 2 of the sweep: the core, default configuration, all three planners.

120 pairs by the registered draw at each of the twelve epochs, the
time-dependent planner, the snapshot baseline on illumination frozen at
departure, and the mean-map baseline swept over its weighting exponent and
reported at its best setting by replay feasibility. Every plan is replayed and
judged twice, at the registered 3.206 deg tolerance and at zero, as registered.

Scores nothing. P2, P3, P4 and P7 are not touched here.
"""
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
from scipy.sparse import coo_matrix, csr_matrix
from scipy.sparse.csgraph import connected_components, dijkstra

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import examples.plan_pilot as P                                  # noqa: E402
from examples.build_intervals import HORIZON_H, cache_path, epochs  # noqa: E402
from examples.replay_batch import FINE_SUN_DT_H, fine_sun        # noqa: E402
from src.baselines import frozen_intervals, plan_mean_map        # noqa: E402
from src.cost import Rover, legal                                # noqa: E402
from src.frames import cell_position                            # noqa: E402
from src.intervals import LitIntervals                          # noqa: E402
from src.replay import ReplaySettings, replay                   # noqa: E402

SEED = 20260924
CAP_S = 30.0
TOLERANCE_H = 2.0
EXPONENTS = (0.0, 0.5, 1.0, 2.0)
PITCH_ROLL_TOL_DEG = 3.206      # registered 2026-10-01 from the first batch
ARRIVAL_CRITERION = 0.01        # registered 2026-10-02
PAIRS_PER_BAND = 24


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
    ap.add_argument("--jobs", type=int, default=12)
    ap.add_argument("--tags", default="all")
    args = ap.parse_args()
    rover = Rover(v_flat=50.0)
    p, n, px, dem, horizon, cells, gx, gy, edges, graph = P.load(rover)
    xs, ys = gx.ravel(), gy.ravel()
    ok = legal(edges["pitch"], edges["roll"], rover)
    g_all = coo_matrix((np.ones(int(ok.sum())), (edges["a"][ok], edges["b"][ok])),
                       shape=(n * n, n * n))
    _, comp_all = connected_components(g_all, directed=False)
    positions = cell_position(gx, gy, dem, p["radius_m"]).reshape(-1, 3)
    tags = ([t for t, _, _ in epochs()] if args.tags == "all" else args.tags.split(","))
    grid = dict(radius_m=p["radius_m"], region_ox=p["region_ox"],
                region_oy=p["region_oy"], region_px=p["region_px"],
                region_nx=p["region_nx"], near_path=p["near_path"],
                near_ox=p["near_ox"], near_oy=p["near_oy"], near_px=p["near_px"],
                near_width=p["near_width"], near_height=p["near_height"],
                far_path=p["far_path"], far_ox=p["far_ox"], far_oy=p["far_oy"],
                far_px=p["far_px"], far_width=p["far_width"],
                far_height=p["far_height"], reach_m=p["reach_m"],
                switch_m=p["switch_m"], near_step_m=p["near_step_m"],
                far_step_m=p["far_step_m"])
    mat_travel = csr_matrix((graph.travel, graph.indices, graph.indptr),
                            shape=(graph.n_cells,) * 2)

    print(f"core: default configuration, v_flat {rover.v_flat:.0f} m/hr, tolerance "
          f"{TOLERANCE_H:.1f} h, limits ({rover.pitch_max:.0f}, {rover.roll_max:.0f}), "
          f"cap {CAP_S:.0f} s")
    print(f"{PAIRS_PER_BAND} pairs per band over {len(P.BANDS)} bands = "
          f"{PAIRS_PER_BAND*len(P.BANDS)} pairs per epoch, {len(tags)} epochs")

    # --- stage 1: prepare and launch every planner job in parallel ---
    jobs, meta = [], {}
    t0 = time.time()
    for ei, tag in enumerate(tags):
        iv = LitIntervals.load(cache_path(tag))
        changing = np.diff(iv.offsets) >= 2
        sel = ok & changing[edges["a"]] & changing[edges["b"]]
        gc = coo_matrix((np.ones(int(sel.sum())), (edges["a"][sel], edges["b"][sel])),
                        shape=(n * n, n * n))
        _, lab = connected_components(gc, directed=False)
        sz = np.bincount(lab)
        pool = np.flatnonzero(lab == np.argsort(sz)[::-1][0])
        pairs = draw(pool, xs, ys, label=ei)
        froz = frozen_intervals(iv, 0.0)
        for kind, table in (("td", iv), ("snap", froz)):
            work = P.DATA / "work" / "sweep" / f"{tag}_{kind}"
            cmd, params = P.run_cpp(graph, table, pairs, TOLERANCE_H, 0.0, HORIZON_H,
                                    CAP_S, work, components=comp_all,
                                    positions=positions, v_cap=rover.v_cap,
                                    want_paths=True, defer=True)
            jobs.append((tag, kind, cmd, params))
        meta[tag] = dict(pairs=pairs, iv=iv, pool=pool.size,
                         lit0=float(iv.lit0.mean()),
                         trans=float(np.diff(iv.offsets).mean()))
    print(f"prepared {len(jobs)} planner jobs in {time.time()-t0:.0f} s; launching "
          f"{args.jobs} at a time")
    t0 = time.time()
    running, results = [], {}
    queue = list(jobs)
    while queue or running:
        while queue and len(running) < args.jobs:
            tag, kind, cmd, params = queue.pop(0)
            running.append((tag, kind, params,
                            subprocess.Popen(cmd, stdout=subprocess.DEVNULL)))
        time.sleep(0.5)
        still = []
        for tag, kind, params, proc in running:
            if proc.poll() is None:
                still.append((tag, kind, params, proc))
            else:
                results[(tag, kind)] = P.read_cpp(params, want_paths=True)
        running = still
    print(f"planner stage done in {time.time()-t0:.0f} s")

    # --- stage 2: mean-map, swept over the weighting exponent ---
    t0 = time.time()
    for tag in tags:
        iv = meta[tag]["iv"]
        lf = iv.lit_fraction()
        meta[tag]["lit_fraction_mean"] = float(lf.mean())
        mm = {}
        for e in EXPONENTS:
            paths = []
            for (a, b) in meta[tag]["pairs"]:
                tt, path = plan_mean_map(graph, lf, a, b, exponent=e)
                paths.append((tt, path))
            mm[e] = paths
        meta[tag]["meanmap"] = mm
    print(f"mean-map stage done in {time.time()-t0:.0f} s "
          f"({len(tags)} epochs x {len(EXPONENTS)} exponents)")

    # --- stage 3: replay everything, judged twice ---
    t0 = time.time()
    report = {}
    for tag in tags:
        suns = fine_sun(tag)
        work = P.DATA / "work" / "sweep" / f"{tag}_replay"
        per_planner = {}
        for kind in ("td", "snap"):
            r = results[(tag, kind)]
            off, pc, pt = r["path_offsets"], r["path_cells"], r["path_times"]
            plans, starts, waits, idx = [], [], [], []
            for q in range(len(meta[tag]["pairs"])):
                lo, hi = off[q], off[q + 1]
                if hi - lo < 2:
                    continue
                cpath, tpath = pc[lo:hi].astype(np.int64), pt[lo:hi]
                w = np.zeros(cpath.size)
                for i in range(cpath.size - 1):
                    u, v = int(cpath[i]), int(cpath[i + 1])
                    s0, s1 = graph.indptr[u], graph.indptr[u + 1]
                    k = np.flatnonzero(graph.indices[s0:s1] == v)
                    w[i] = max(tpath[i + 1] - float(graph.travel[s0:s1][k[0]])
                               - tpath[i], 0.0)
                plans.append(cpath); starts.append(float(tpath[0]))
                waits.append(w); idx.append(q)
            per_planner[kind] = dict(plans=plans, starts=starts, waits=waits, idx=idx,
                                     arrival=r["arrival"], aborted=r["aborted"])
        for e in EXPONENTS:
            plans, starts, waits, idx = [], [], [], []
            for q, (tt, path) in enumerate(meta[tag]["meanmap"][e]):
                if not path or len(path) < 2:
                    continue
                plans.append(np.asarray(path, dtype=np.int64))
                starts.append(0.0)
                waits.append(np.zeros(len(path)))
                idx.append(q)
            per_planner[f"mm{e}"] = dict(plans=plans, starts=starts, waits=waits,
                                         idx=idx, arrival=None, aborted=None)
        judged = {}
        for kind, d in per_planner.items():
            if not d["plans"]:
                judged[kind] = None
                continue
            out = {}
            for label, tol in (("tol", PITCH_ROLL_TOL_DEG), ("zero", 0.0)):
                out[label] = replay(d["plans"], d["starts"], grid, rover, suns,
                                    FINE_SUN_DT_H, TOLERANCE_H, work,
                                    ReplaySettings(limit_tolerance_deg=tol),
                                    waits=d["waits"])
            judged[kind] = out
        report[tag] = dict(per_planner=per_planner, judged=judged)
    print(f"replay stage done in {time.time()-t0:.0f} s")

    # --- tables ---
    print(f"\n=== per epoch, planner outcomes (feasible / infeasible / aborted), "
          f"cap {CAP_S:.0f} s ===")
    print(f"  {'epoch':<16}{'lit t0':>8}{'litfrac':>8}"
          f"{'time-dependent':>18}{'snapshot':>18}")
    for tag in tags:
        row = []
        for kind in ("td", "snap"):
            r = results[(tag, kind)]
            ab = r["aborted"]
            f = int((~ab & np.isfinite(r["arrival"])).sum())
            i = int((~ab & ~np.isfinite(r["arrival"])).sum())
            row.append(f"{f:5d} /{i:5d} /{int(ab.sum()):4d}")
        print(f"  {tag:<16}{meta[tag]['lit0']:>7.2%}"
              f"{meta[tag]['lit_fraction_mean']:>8.3f}{row[0]:>18}{row[1]:>18}")

    print(f"\n=== per epoch, plans produced and replay verdicts ===")
    print(f"  {'epoch':<16}{'planner':<8}{'plans':>6}{'pass@tol':>9}{'pass@0':>7}"
          f"{'cost':>6}{'nearlim':>8}{'arr<=1%':>9}")
    for tag in tags:
        for kind in ["td", "snap"] + [f"mm{e}" for e in EXPONENTS]:
            d = report[tag]["per_planner"][kind]
            j = report[tag]["judged"][kind]
            if j is None:
                continue
            a, z = j["tol"], j["zero"]
            near = 0
            for i in range(len(d["plans"])):
                lo, hi = a["step_offsets"][i], a["step_offsets"][i + 1]
                wp = np.abs(a["step_pitch"][lo:hi].astype(float))
                wr = np.abs(a["step_roll"][lo:hi].astype(float))
                near += int(np.sum((wp > rover.pitch_max - PITCH_ROLL_TOL_DEG)
                                   | (wr > rover.roll_max - PITCH_ROLL_TOL_DEG)))
            if d["arrival"] is not None:
                claimed = np.array([d["arrival"][q] for q in d["idx"]])
            else:
                claimed = np.array([meta[tag]["meanmap"][float(kind[2:])][q][0]
                                    for q in d["idx"]])
            rel = np.abs(a["arrival"] - claimed) / np.maximum(claimed, 1e-9)
            print(f"  {tag:<16}{kind:<8}{len(d['plans']):>6}"
                  f"{int(a['verdict'].sum()):>9}{int(z['verdict'].sum()):>7}"
                  f"{int(a['verdict'].sum())-int(z['verdict'].sum()):>6}{near:>8}"
                  f"{int((rel <= ARRIVAL_CRITERION).sum()):>9}")

    print(f"\n=== mean-map weighting, best exponent per epoch by replay feasibility ===")
    print(f"  {'epoch':<16}" + "".join(f"{f'e={e}':>9}" for e in EXPONENTS) + "   best")
    for tag in tags:
        counts = []
        for e in EXPONENTS:
            j = report[tag]["judged"][f"mm{e}"]
            counts.append(0 if j is None else int(j["tol"]["verdict"].sum()))
        best = EXPONENTS[int(np.argmax(counts))] if max(counts) > 0 else None
        print(f"  {tag:<16}" + "".join(f"{c:>9}" for c in counts)
              + f"   {best if best is not None else 'none feasible'}")
    tot = {e: 0 for e in EXPONENTS}
    for tag in tags:
        for e in EXPONENTS:
            j = report[tag]["judged"][f"mm{e}"]
            tot[e] += 0 if j is None else int(j["tol"]["verdict"].sum())
    print(f"  {'core total':<16}" + "".join(f"{tot[e]:>9}" for e in EXPONENTS)
          + f"   {max(tot, key=tot.get)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
