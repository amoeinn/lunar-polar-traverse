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
# The epoch order the draw is keyed on. It is the registered twelve, and the
# key is an epoch's own place in that list rather than its place in --tags.
# Keying it on the --tags position made the pairs an epoch got depend on which
# other epochs shared the run: the original sweep passed --tags all, so the two
# agreed and nothing showed, and the sun-units regeneration then ran the same
# twelve epochs in five batches and drew a different instance set in each.
# 2028-05-15_p1 and 2028-05-15_p2 ended with their planner stages scored on one
# draw and their mean-map stages on another.
STAGES = ("planner", "mean-map", "replay")
REGISTERED_EPOCHS = tuple(t for t, _, _ in epochs())
_DRAW_LABEL = {t: i for i, t in enumerate(REGISTERED_EPOCHS)}
TOLERANCE_H = 2.0
EXPONENTS = (0.0, 0.5, 1.0, 2.0)
PITCH_ROLL_TOL_DEG = 3.206      # registered 2026-10-01 from the first batch
ARRIVAL_CRITERION = 0.01        # registered 2026-10-02
PAIRS_PER_BAND = 24


def draw_label(tag):
    """The rng key for an epoch's pair draw: its index among the registered
    twelve, independent of how the run was batched."""
    if tag not in _DRAW_LABEL:
        raise KeyError(f"{tag} is not one of the registered epochs "
                       f"{list(REGISTERED_EPOCHS)}")
    return _DRAW_LABEL[tag]


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
    ap.add_argument("--stages", default="all",
                    help="which stages to run: all, or a comma-separated "
                         "subset of planner, mean-map, replay. Splitting a "
                         "long epoch into --stages planner and then "
                         "--stages mean-map,replay is two runs that each fit a "
                         "timeout, instead of one that is killed partway. A "
                         "run without the planner stage reads its results back "
                         "from disk and refuses if any are missing.")
    ap.add_argument("--resume", action="store_true",
                    help="read back planner jobs whose outputs are already "
                         "complete on disk, and replay verdicts already "
                         "computed, instead of recomputing them. The mean-map "
                         "stage keeps nothing on disk, so it is always redone; "
                         "at 17 s an epoch that is not worth caching.")
    args = ap.parse_args()
    # Validated before anything is loaded or written: a mistyped tag should not
    # cost a region load, and must never reach the draw, where an unregistered
    # epoch has no key.
    tags = (list(REGISTERED_EPOCHS) if args.tags == "all" else args.tags.split(","))
    unknown = [t for t in tags if t not in _DRAW_LABEL]
    if unknown:
        print(f"not registered epochs: {unknown}; known: "
              f"{list(REGISTERED_EPOCHS)}")
        return 2
    if args.stages == "all":
        stages = set(STAGES)
    else:
        stages = {t.strip() for t in args.stages.split(",") if t.strip()}
        bad = sorted(stages - set(STAGES))
        if bad or not stages:
            print(f"unknown stages {bad or ['(none given)']}; choose all, or a "
                  f"comma-separated subset of {list(STAGES)}")
            return 2
        if "replay" in stages and "mean-map" not in stages:
            print("--stages replay needs mean-map in the same run: the mean-map "
                  "stage keeps nothing on disk, so its plans have to be "
                  "recomputed in the process that replays them")
            return 2
    rover = Rover(v_flat=50.0)
    p, n, px, dem, horizon, cells, gx, gy, edges, graph = P.load(rover)
    xs, ys = gx.ravel(), gy.ravel()
    ok = legal(edges["pitch"], edges["roll"], rover)
    g_all = coo_matrix((np.ones(int(ok.sum())), (edges["a"][ok], edges["b"][ok])),
                       shape=(n * n, n * n))
    _, comp_all = connected_components(g_all, directed=False)
    positions = cell_position(gx, gy, dem, p["radius_m"]).reshape(-1, 3)
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
    jobs, meta, missing = [], {}, []
    t0 = time.time()
    for tag in tags:
        iv = LitIntervals.load(cache_path(tag))
        changing = np.diff(iv.offsets) >= 2
        sel = ok & changing[edges["a"]] & changing[edges["b"]]
        gc = coo_matrix((np.ones(int(sel.sum())), (edges["a"][sel], edges["b"][sel])),
                        shape=(n * n, n * n))
        _, lab = connected_components(gc, directed=False)
        sz = np.bincount(lab)
        pool = np.flatnonzero(lab == np.argsort(sz)[::-1][0])
        pairs = draw(pool, xs, ys, label=draw_label(tag))
        froz = frozen_intervals(iv, 0.0)
        for kind, table in (("td", iv), ("snap", froz)):
            work = P.DATA / "work" / "sweep" / f"{tag}_{kind}"
            have = (P.finished_job(work, want_paths=True)
                    if (args.resume or "planner" not in stages) else None)
            if have is not None:
                # Read back, and do not touch the directory: run_cpp would
                # rewrite the inputs and the params even when deferring.
                jobs.append((tag, kind, None, have))
                continue
            if "planner" not in stages:
                # Nothing is written for a job this run will not plan: run_cpp
                # writes 41 MB of inputs and a params file before it defers,
                # which would leave a directory that looks started and is not.
                missing.append((tag, kind))
                continue
            cmd, params = P.run_cpp(graph, table, pairs, TOLERANCE_H, 0.0, HORIZON_H,
                                    CAP_S, work, components=comp_all,
                                    positions=positions, v_cap=rover.v_cap,
                                    want_paths=True, defer=True)
            jobs.append((tag, kind, cmd, params))
        meta[tag] = dict(pairs=pairs, iv=iv, pool=pool.size,
                         lit0=float(iv.lit0.mean()),
                         trans=float(np.diff(iv.offsets).mean()))
    if missing:
        print(f"\n--stages {args.stages} does not run the planner, and "
              f"{len(missing)} job(s) have no complete output on disk:")
        for tag, kind in missing:
            print(f"  {tag} {kind}")
        print("run --stages planner for those epochs first; nothing was "
              "written for them here")
        return 2
    done = [j for j in jobs if j[2] is None]
    todo = [j for j in jobs if j[2] is not None]
    print(f"prepared {len(jobs)} planner jobs in {time.time()-t0:.0f} s; "
          f"{len(done)} already complete on disk, {len(todo)} to run, launching "
          f"{args.jobs} at a time")
    t0 = time.time()
    running, results = [], {}
    for tag, kind, _, params in done:
        results[(tag, kind)] = P.read_cpp(params, want_paths=True)
    queue = list(todo)
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

    if "mean-map" not in stages:
        print(f"\n=== per epoch, planner outcomes (feasible / infeasible / "
              f"aborted), cap {CAP_S:.0f} s ===")
        print(f"  {'epoch':<16}{'lit t0':>8}{'time-dependent':>18}{'snapshot':>18}")
        for tag in tags:
            row = []
            for kind in ("td", "snap"):
                r = results[(tag, kind)]
                ab = r["aborted"]
                row.append(f"{int((~ab & np.isfinite(r['arrival'])).sum()):5d} /"
                           f"{int((~ab & ~np.isfinite(r['arrival'])).sum()):5d} /"
                           f"{int(ab.sum()):4d}")
            print(f"  {tag:<16}{meta[tag]['lit0']:>7.2%}{row[0]:>18}{row[1]:>18}")
        print(f"\nstages run: {sorted(stages)}. The mean-map and replay stages "
              f"were not run, so no replay verdict or mean-map exponent is "
              f"reported here. Finish with --stages mean-map,replay.")
        return 0

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

    if "replay" not in stages:
        print(f"\nstages run: {sorted(stages)}. Nothing was replayed, so no "
              f"verdict is reported. The mean-map plans are not written to "
              f"disk, so finishing needs --stages mean-map,replay rather than "
              f"--stages replay.")
        return 0

    # --- stage 3: replay everything, judged twice ---
    t0 = time.time()
    report = {}
    for tag in tags:
        suns = fine_sun(tag)
        # One directory per planner and tolerance label, the convention
        # examples/sweep_axes.py and examples/score_phase3.py already use. The
        # single shared directory this used to pass kept only the last of the
        # twelve calls, which is why phase 3 step 12 had to rebuild the mean-map
        # replay from scratch, and it made resuming unsafe: reuse checks that
        # the outputs exist, so the first call of a resumed run would have read
        # back whichever call happened to write there last.
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
                                    FINE_SUN_DT_H, TOLERANCE_H,
                                    Path(f"{work}_{kind}_{label}"),
                                    ReplaySettings(limit_tolerance_deg=tol),
                                    waits=d["waits"], reuse=args.resume)
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
