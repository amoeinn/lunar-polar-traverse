"""How long world-and-model illumination disagreements last on the executed plans.

The horizon in the executive's replacement contradiction rule is set from this
and from nothing else. Run before that rule exists and before P8 was written,
so the number comes from the terrain rather than from what makes a bar pass.

Two sources, and which one sets the horizon matters. --source held-out measures
the plans of all nine axis points, HELD_OUT_STEMS, pooled; the comment there
says why pooling is sound. They come from the axis draw, 8 pairs per band
against the core's 24, so they are not the sample the executive is scored on
and a horizon taken from them is not fitted to the data it will be tested
against. The separation is not complete. At 2028-02-15_p0 both draws are keyed
on label 0, so the axis's 40 pairs are a prefix of the core's 120; at the other
three epochs the keys differ and no pair is shared. On the corrected sun and
checker, 8 of the 70 held-out plans that enter shadow sit on a shared pair, and
leaving them out does not move the p99.

The pool also contains tolerance_0, the 0.5 h point that hosts the paired
design. Leaving it out does not move the p99 either; leaving out every plan, at
any point, on a pair the host also plans lowers it by 0.0534 h, on 33 episodes
against 83.
--source sample measures the monitored set itself and is reported beside it as
a check, never as the source.
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.disagree import CADENCE_H, FlatIntervals, episodes, sample_states  # noqa: E402
from src.executive import plan_waits, simulate                   # noqa: E402

RESULTS = ROOT / "data" / "work" / "executive" / "results.json"
AXES = ROOT / "data" / "work" / "axes"
HELD_OUT_EPOCHS = ("2028-02-15_p0", "2028-05-15_p1", "2028-08-15_p2",
                   "2028-11-15_p2")
# Every axis point, all of them on the axis draw and so disjoint from the core
# sample P8 is scored on. Pooling them is not pooling unlike things: what is
# being measured is how long the planner's cell-centre, 15-minute-sun
# illumination model disagrees with the checker's continuous ray march, which is
# a property of that model against this terrain. The rover's rate, budget and
# slope limits change which ground gets walked, not how long a disagreement
# lasts once the rover is standing on it. The 50 m/hr point alone gave 7
# episodes, far too few for a p99.
HELD_OUT_STEMS = ("rate_core_0_", "rate_0_", "rate_1_", "rate_2_",
                  "tolerance_0_", "tolerance_1_", "tolerance_2_",
                  "limits_0_", "limits_1_")


def from_sample():
    """The monitored set's own trajectories, already dumped by the run.

    results.json holds one record per plan per contradiction rule, and leg 0's
    world is the checker on the unmodified plan, identical under both rules
    (measured: 51 of 51 byte-identical on the corrected core). Each plan is
    yielded once, or every episode would be counted twice.
    """
    seen = set()
    for r in json.loads(RESULTS.read_text()):
        if not r["monitored"]:
            continue
        key = (r["tag"], r["plan_index"])
        if key in seen:
            continue
        seen.add(key)
        w = Path(r["work"]) / "leg0_world"
        yield (r["tag"], r["graph"], r["plan"],
               np.fromfile(w / "out_traj_t.bin", np.float64),
               np.fromfile(w / "out_traj_lit.bin", np.uint8),
               np.fromfile(w / "out_traj_index.bin", np.int32))


def from_held_out(work_root):
    """Every axis point's plans, re-simulated so they carry a trajectory dump.

    Their replays predate the trajectory output, so the checker is run again on
    each plan. Nothing about a plan changes: the same cells, the same schedule,
    the same geometry. The replay template is each axis point's own, not the
    core's, because the rover parameters differ between points and integrating a
    20 m/hr plan at 50 m/hr would distort the very durations being measured.
    """
    for stem in HELD_OUT_STEMS:
        for tag in HELD_OUT_EPOCHS:
            pt = AXES / f"{stem}{tag}_td" / "planner_params.json"
            rt = AXES / f"{stem}{tag}_replay_td_tol" / "replay_params.json"
            if not (pt.exists() and rt.exists()):
                continue
            tp = json.loads(pt.read_text())
            graph = {k: tp[k] for k in ("indptr_path", "indices_path",
                                        "travel_path", "itimes_path",
                                        "ioffsets_path", "ilit0_path")}
            indptr = np.fromfile(graph["indptr_path"], np.int64)
            indices = np.fromfile(graph["indices_path"], np.int32)
            travel = np.fromfile(graph["travel_path"], np.float64)
            off = np.fromfile(tp["out_path_offsets"], np.int64)
            pc = np.fromfile(tp["out_path_cells"], np.int32)
            ptm = np.fromfile(tp["out_path_times"], np.float64)
            for q in range(off.size - 1):
                lo, hi = off[q], off[q + 1]
                if hi - lo < 2:
                    continue
                plan, times = pc[lo:hi].astype(np.int32), ptm[lo:hi]
                holds = plan_waits(plan, times, indptr, indices, travel)
                out = simulate(rt, plan, float(times[0]), holds,
                               Path(work_root) / f"{stem}{tag}_{q:03d}")
                tj = out["traj"]
                if tj["t"].size < 2 or out["shadow"] <= 0.0:
                    continue
                yield (tag, graph, plan.tolist(), tj["t"], tj["lit"], tj["index"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=("held-out", "sample"), default="held-out")
    args = ap.parse_args()
    src = (from_sample() if args.source == "sample"
           else from_held_out(ROOT / "data" / "work" / "disagree"))
    cache, dark, lit = {}, [], []
    plans = ticks = 0
    for tag, graph, plan, t, li, ix in src:
        if tag not in cache:
            cache[tag] = FlatIntervals(graph["itimes_path"], graph["ioffsets_path"],
                                       graph["ilit0_path"])
        grid, w, m = sample_states(t, li, ix, plan, cache[tag])
        if grid.size == 0:
            continue
        plans += 1
        ticks += grid.size
        dark.append(episodes(w, m, world_dark_only=True))
        lit.append(episodes(w, m, world_dark_only=False))
    dark = np.concatenate(dark) if dark else np.zeros(0)
    lit = np.concatenate(lit) if lit else np.zeros(0)

    print(f"source {args.source}: {plans} plans that enter shadow, sampled at the "
          f"checker's own {CADENCE_H*3600:.0f} s visibility cadence, {ticks} samples")
    for name, e in (("world dark, model says lit  (the rule's direction)", dark),
                    ("world lit, model says dark  (the other direction)", lit)):
        print(f"\n  {name}: {e.size} episodes")
        if e.size == 0:
            continue
        for q in (0.5, 0.75, 0.9, 0.95, 0.99):
            print(f"    p{int(q*100):<3} {np.quantile(e, q):8.4f} h")
        print(f"    max  {e.max():8.4f} h   mean {e.mean():8.4f} h   "
              f"total {e.sum():8.4f} h")
    if dark.size:
        print(f"\n  the horizon the replacement rule will use is the p99 of the "
              f"first distribution: {np.quantile(dark, 0.99):.4f} h "
              f"({np.quantile(dark, 0.99)*60:.1f} min)")
        print(f"  for comparison, the old rule's projection reached the model's "
              f"next transition, median 139.825 h on the unnecessary firings")
    return 0


if __name__ == "__main__":
    sys.exit(main())
