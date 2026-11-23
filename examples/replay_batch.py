"""The first batch of plans, replayed; the pitch/roll tolerance measured from it.

PLAN.md registers the sequence: replay a first batch, record the distribution of
the checker-minus-planner difference in pitch and roll, register the tolerance
as its p99 before any plan is judged, then rejudge the batch under it. Also
reports how much the planner's edge-level shadow rule overstates shadow against
the checker's continuous integration.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import spiceypy as sp
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import examples.plan_pilot as P                                  # noqa: E402
from examples.build_intervals import (DATES, HORIZON_H, PHASE_OFFSET_DAYS,  # noqa: E402
                                      cache_path)
from src.arrival import arrival_gap, describe, summarise         # noqa: E402
from src.cost import Rover, build_edges, legal                   # noqa: E402
from src.frames import cell_position                             # noqa: E402
from src.intervals import LitIntervals                           # noqa: E402
from src.replay import ReplaySettings, replay                    # noqa: E402
from src.sun import sun_positions_m  # noqa: E402

SEED = 20261001
CAP_S = 30.0
FINE_SUN_DT_H = 1.0 / 60.0        # one minute, against the planner's fifteen


def fine_sun(tag):
    date, phase = tag.rsplit("_p", 1)
    for k in P.KERNELS:
        sp.furnsh(str(P.DATA / "kernels" / k))
    t0 = sp.str2et(f"{date}T00:00:00") + PHASE_OFFSET_DAYS[int(phase)] * 86400.0
    ets = np.arange(t0, t0 + (HORIZON_H + FINE_SUN_DT_H) * 3600.0,
                    FINE_SUN_DT_H * 3600.0)
    pos = sun_positions_m(ets, P.FRAME)
    sp.kclear()
    return pos


def arrival_report(checker, planner):
    """The batch's arrival agreement, signed.

    This line used to print the median of abs(checker - planner) / planner,
    which is how "+1.169%, checker slower" entered the record: a magnitude with
    a predicted sign written beside it. It now reports the signed gap and the
    count in each direction, and the magnitude only as a figure labelled as
    having no sign.
    """
    s = summarise(arrival_gap(checker, planner))
    return f"  arrival, (checker - planner) / planner: {describe(s)}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tags", default="all",
                    help="comma-separated epoch tags, or all twelve")
    ap.add_argument("--instances", type=int, default=10,
                    help="instances drawn per epoch")
    ap.add_argument("--tolerance", type=float, default=2.0)
    args = ap.parse_args()
    from examples.build_intervals import epochs as all_epochs
    tags = ([t for t, _, _ in all_epochs()] if args.tags == "all"
            else args.tags.split(","))
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
                region_nx=p["region_nx"],
                near_path=p["near_path"], near_ox=p["near_ox"], near_oy=p["near_oy"],
                near_px=p["near_px"], near_width=p["near_width"],
                near_height=p["near_height"],
                far_path=p["far_path"], far_ox=p["far_ox"], far_oy=p["far_oy"],
                far_px=p["far_px"], far_width=p["far_width"],
                far_height=p["far_height"],
                reach_m=p["reach_m"], switch_m=p["switch_m"],
                near_step_m=p["near_step_m"], far_step_m=p["far_step_m"])
    work = P.DATA / "work" / "plan"
    per_band = max(args.instances // len(P.BANDS), 1)

    # Every epoch contributes the same number of drawn instances. Plans arrive
    # disproportionately from the epochs that are lit at departure, which is a
    # fact about the site and not a selection: no plan is kept or dropped on
    # account of what the checker later says about it.
    batch = []
    print(f"\n  {'epoch':<16}{'feas':>6}{'infeas':>8}{'abort':>7}{'plans':>7}"
          f"{'planner s':>11}")
    for tag in tags:
        iv = LitIntervals.load(cache_path(tag))
        changing = np.diff(iv.offsets) >= 2
        sel = ok & changing[edges["a"]] & changing[edges["b"]]
        gc = coo_matrix((np.ones(int(sel.sum())), (edges["a"][sel], edges["b"][sel])),
                        shape=(n * n, n * n))
        _, lab = connected_components(gc, directed=False)
        sz = np.bincount(lab)
        pool = np.flatnonzero(lab == np.argsort(sz)[::-1][0])
        pairs = []
        for bi, (lo, hi) in enumerate(P.BANDS):
            rng = np.random.default_rng([SEED, bi, int(lo), int(hi)])
            got = 0
            while got < per_band:
                a, b = rng.choice(pool, 2, replace=False)
                d = np.hypot(xs[a] - xs[b], ys[a] - ys[b])
                if lo <= d < hi:
                    pairs.append((int(a), int(b)))
                    got += 1
        t0 = time.time()
        res = P.run_cpp(graph, iv, pairs, args.tolerance, 0.0, HORIZON_H, CAP_S, work,
                        components=comp_all, positions=positions, v_cap=rover.v_cap,
                        want_paths=True)
        wall = time.time() - t0
        off, pcells, ptimes = res["path_offsets"], res["path_cells"], res["path_times"]
        got_plans = 0
        for q in range(len(pairs)):
            lo, hi = off[q], off[q + 1]
            if hi - lo < 2:
                continue
            cpath = pcells[lo:hi].astype(np.int64)
            tpath = ptimes[lo:hi]
            w = np.zeros(cpath.size)
            for i in range(cpath.size - 1):
                u, v = int(cpath[i]), int(cpath[i + 1])
                s0, s1 = graph.indptr[u], graph.indptr[u + 1]
                k = np.flatnonzero(graph.indices[s0:s1] == v)
                tau = float(graph.travel[s0:s1][k[0]])
                w[i] = max(tpath[i + 1] - tau - tpath[i], 0.0)
            batch.append(dict(tag=tag, cells=cpath, times=tpath, wait=w,
                              arrival=float(res["arrival"][q])))
            got_plans += 1
        ab = res["aborted"]
        print(f"  {tag:<16}{int((~ab & np.isfinite(res['arrival'])).sum()):>6}"
              f"{int((~ab & ~np.isfinite(res['arrival'])).sum()):>8}"
              f"{int(ab.sum()):>7}{got_plans:>7}{wall:>11.0f}")

    if not batch:
        print("no plans in the batch; nothing to register")
        return 1
    print(f"\n{len(batch)} plans in the batch, from "
          f"{len(set(b['tag'] for b in batch))} epochs; waiting per plan median "
          f"{np.median([b['wait'].sum() for b in batch]):.2f} h, max "
          f"{max(b['wait'].sum() for b in batch):.2f} h")

    # replay, grouped by epoch because the sun table is per epoch
    results = []
    for tag in sorted(set(b["tag"] for b in batch)):
        grp = [b for b in batch if b["tag"] == tag]
        suns = fine_sun(tag)
        r = replay([b["cells"] for b in grp], [float(b["times"][0]) for b in grp],
                   grid, rover, suns, FINE_SUN_DT_H, args.tolerance, work,
                   ReplaySettings(limit_tolerance_deg=1.0e9),
                   waits=[b["wait"] for b in grp])
        results.append((tag, grp, r, suns))
    print(f"replayed {len(batch)} plans across {len(results)} epochs, "
          f"limits not yet judged")

    # --- the pitch and roll difference distribution, before anything is judged ---
    edge_lookup = {}
    for i in range(edges["a"].size):
        edge_lookup[(int(edges["a"][i]), int(edges["b"][i]))] = i
    dp, dr = [], []
    for tag, grp, r, _ in results:
        for i, b in enumerate(grp):
            lo, hi = r["step_offsets"][i], r["step_offsets"][i + 1]
            cp = r["step_pitch_mean"][lo:hi].astype(float)
            cr = r["step_roll_mean"][lo:hi].astype(float)
            cpath = b["cells"]
            for j in range(min(len(cpath) - 1, cp.size)):
                k = edge_lookup.get((int(cpath[j]), int(cpath[j + 1])))
                if k is None:
                    continue
                dp.append(cp[j] - float(edges["pitch"][k]))
                dr.append(cr[j] - float(edges["roll"][k]))
    dp, dr = np.abs(np.array(dp)), np.abs(np.array(dr))
    print(f"\nchecker minus planner, {dp.size} steps over {len(batch)} plans, "
          f"absolute degrees:")
    for name, v in (("pitch", dp), ("roll", dr)):
        print(f"  {name:<6} median {np.median(v):6.3f}  p90 {np.percentile(v,90):6.3f}  "
              f"p99 {np.percentile(v,99):6.3f}  max {v.max():6.3f}")
    tol_deg = float(max(np.percentile(dp, 99), np.percentile(dr, 99)))
    print(f"  registered tolerance, the larger of the two p99 values: "
          f"{tol_deg:.3f} deg")

    # --- rejudge the batch under the registered tolerance ---
    tot_ok = tot = illegal_rej = shadow_rej = near_limit = 0
    chk_arr, pln_arr, over_all, pl_sh, ch_sh = [], [], [], [], []
    for tag, grp, _, suns in results:
        r2 = replay([b["cells"] for b in grp], [float(b["times"][0]) for b in grp],
                    grid, rover, suns, FINE_SUN_DT_H, args.tolerance, work,
                    ReplaySettings(limit_tolerance_deg=tol_deg),
                    waits=[b["wait"] for b in grp])
        iv = LitIntervals.load(cache_path(tag))
        for i, b in enumerate(grp):
            tot += 1
            tot_ok += int(r2["verdict"][i])
            if r2["illegal"][i] > 0:
                illegal_rej += 1
            elif r2["shadow"][i] > args.tolerance:
                shadow_rej += 1
            lo, hi = r2["step_offsets"][i], r2["step_offsets"][i + 1]
            wp = np.abs(r2["step_pitch"][lo:hi].astype(float))
            wr = np.abs(r2["step_roll"][lo:hi].astype(float))
            near_limit += int(np.sum((wp > rover.pitch_max - tol_deg)
                                     | (wr > rover.roll_max - tol_deg)))
            chk_arr.append(float(r2["arrival"][i]))
            pln_arr.append(float(b["arrival"]))
            # the planner's own shadow rule along its own schedule
            s, s_max = 0.0, 0.0
            cpath, tpath = b["cells"], b["times"]
            for j in range(len(cpath) - 1):
                u, v = int(cpath[j]), int(cpath[j + 1])
                s0, s1 = graph.indptr[u], graph.indptr[u + 1]
                k = np.flatnonzero(graph.indices[s0:s1] == v)
                tau = float(graph.travel[s0:s1][k[0]])
                d = tpath[j + 1] - tau
                s, m = iv.shadow_along(u, v, d, d + tau, s)
                s_max = max(s_max, m)
            pl_sh.append(s_max)
            ch_sh.append(float(r2["shadow"][i]))
            over_all.append(s_max - float(r2["shadow"][i]))
    print(f"\nbatch judged under that tolerance: {tot_ok} of {tot} plans replay "
          f"feasible, {tot - tot_ok} rejected")
    print(f"  rejected for illegal steps {illegal_rej}, for shadow {shadow_rej}")
    print(f"  steps within the tolerance of a limit, which the tolerance decides: "
          f"{near_limit}")
    print(arrival_report(chk_arr, pln_arr))

    pl_sh, ch_sh, over = np.array(pl_sh), np.array(ch_sh), np.array(over_all)
    print(f"\nshadow conservatism, the planner's either-endpoint edge rule against "
          f"the checker's\ncontinuous integration of the rover's own position:")
    print(f"  planner max shadow per plan: median {np.median(pl_sh):.3f} h, "
          f"max {pl_sh.max():.3f} h")
    print(f"  checker max shadow per plan: median {np.median(ch_sh):.3f} h, "
          f"max {ch_sh.max():.3f} h")
    print(f"  overstatement: median {np.median(over):+.3f} h, p10 "
          f"{np.percentile(over,10):+.3f}, p90 {np.percentile(over,90):+.3f}, "
          f"min {over.min():+.3f}, max {over.max():+.3f}")
    print(f"  plans where the planner understated: {int((over < -1e-9).sum())} of "
          f"{over.size}")
    print(f"  visibility cadence {ReplaySettings().visibility_cadence_s:.0f} s, "
          f"integration step {ReplaySettings().integration_step_m:.0f} m, "
          f"disc radius {ReplaySettings().disc_radius_m:.0f} m")
    return 0


if __name__ == "__main__":
    sys.exit(main())
