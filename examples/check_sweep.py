"""Verify core-sweep planner outputs, and that they came from the current tables.

    .venv/bin/python examples/check_sweep.py [<tag> ...]

With no tags it checks the twelve registered epochs. Run it after each batch of
examples/sweep_core.py --tags ...

The invariant that matters here is not that the files exist. It is that each
planner job was run against the interval table now on disk. sweep_core writes
n_transitions into planner_params.json, and the cached interval table carries
the same count, so the two can be compared; a job run against a superseded
table, which is exactly what happened across the whole pipeline when the sun was
passed in kilometres, then shows up as a mismatch instead of as a plausible set
of numbers. The snapshot job needs a second witness, because frozen_intervals
leaves it with no transitions at all and so with nothing to compare: for it the
test is that its outputs are not older than the interval table on disk. The registered configuration is checked the same way and for the
same reason: a run made at a different cap, tolerance or horizon leaves outputs
that are indistinguishable from the right ones by inspection.

The per-epoch outcome triple is printed beside the pre-fix run where a copy
exists under data/work/pre_sunfix/sweep, since the direction of the change is
what D132-3 is scored on.

The replay stage is checked by presence only, per planner and tolerance label.
sweep_core now gives each of the twelve calls its own directory, so a count of
how many carry a verdict says how far the stage got; a tag with none is a run
whose replay stage did not start. A planner with no feasible plan at an epoch
produces no directory, which is why a partial count is reported rather than
failed.

Read only; writes nothing.
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import examples.plan_pilot as P                                 # noqa: E402
from examples.build_intervals import cache_path                 # noqa: E402
from examples.sweep_core import EXPONENTS                       # noqa: E402
from src.intervals import LitIntervals                          # noqa: E402

SWEEP = ROOT / "data" / "work" / "sweep"
OLD = ROOT / "data" / "work" / "pre_sunfix" / "sweep"
DATES = ("2028-02-15", "2028-05-15", "2028-08-15", "2028-11-15")
REGISTERED = tuple(f"{d}_p{k}" for d in DATES for k in range(3))
KINDS = ("td", "snap")
# The registered core configuration, PLAN.md phase 3 step 10. A run at any
# other setting is a different experiment and must not be read as this one.
EXPECTED = {"n_cells": 250000, "n_instances": 120, "shadow_tol_h": 2.0,
            "t_start_h": 0.0, "horizon_h": 720.0, "time_limit_s": 30.0,
            "v_cap": 75.0, "shadow_grid_h": 0.0, "relax_dominance": 0.0}
# Every planner and tolerance label sweep_core replays into its own directory.
REPLAY_KEYS = tuple(f"{k}_{lbl}"
                    for k in ("td", "snap") + tuple(f"mm{e}" for e in EXPONENTS)
                    for lbl in ("tol", "zero"))


def transitions_now(tag):
    """Transition count and mtime of the interval table now on disk, or None."""
    p = Path(cache_path(tag))
    if not p.exists():
        return None, None
    iv = LitIntervals.load(p)
    return int(np.diff(iv.offsets).sum()), p.stat().st_mtime


def check_job(d, tag, kind, expect_trans, table_mtime):
    """Problems with one planner job directory, and its outcome triple."""
    probs, triple, secs = [], None, float("nan")
    pf = d / "planner_params.json"
    if not pf.exists():
        return [f"no planner_params.json in {d.name}"], None, secs
    params = json.loads(pf.read_text())
    for k, v in EXPECTED.items():
        if k in params and params[k] != v:
            probs.append(f"{k} {params[k]} != registered {v}")
    # The time-dependent job carries the table's own transition count, so it can
    # be compared directly. The snapshot job is built by frozen_intervals and
    # has none by construction, so for it the count only confirms that it really
    # is the frozen baseline, and the witness that it came from the current
    # table is the mtime check below.
    got = params.get("n_transitions")
    if kind == "snap":
        if got != 0:
            probs.append(f"n_transitions {got} != 0, so this is not the frozen "
                         f"snapshot baseline")
    elif expect_trans is not None and got != expect_trans:
        probs.append(f"n_transitions {got} != {expect_trans} in the interval "
                     f"table now on disk -- this job used a superseded table")
    probs += P.output_problems(params, want_paths=True)
    for name in [f"out_{k}.bin" for k in P.OUT_WIDTH]:
        f = d / name
        if not f.exists():
            continue
        if f.stat().st_mtime < pf.stat().st_mtime:
            probs.append(f"{name} is older than its inputs, so it is stale")
        elif table_mtime is not None and f.stat().st_mtime < table_mtime:
            probs.append(f"{name} predates the interval table it should have "
                         f"been planned against")
    if not probs:
        arr = np.fromfile(d / "out_arrival.bin", dtype=np.float64)
        ab = np.fromfile(d / "out_flags.bin", dtype=np.uint8).astype(bool)
        secs = float(np.fromfile(d / "out_seconds.bin", dtype=np.float64).sum())
        triple = (int((~ab & np.isfinite(arr)).sum()),
                  int((~ab & ~np.isfinite(arr)).sum()), int(ab.sum()))
    return probs, triple, secs


def replayable_plans(d):
    """How many of a planner job's instances carry a path long enough to
    replay, by the driver's own rule that a path of fewer than two cells is
    not a plan. A planner that found nothing has no replay directory, and that
    is a recorded outcome rather than a missing artifact."""
    f = Path(d) / "out_path_offsets.bin"
    if not f.exists():
        return None
    return int((np.diff(np.fromfile(f, dtype=np.int64)) >= 2).sum())


def old_triple(tag, kind):
    d = OLD / f"{tag}_{kind}"
    try:
        arr = np.fromfile(d / "out_arrival.bin", dtype=np.float64)
        ab = np.fromfile(d / "out_flags.bin", dtype=np.uint8).astype(bool)
        return (int((~ab & np.isfinite(arr)).sum()),
                int((~ab & ~np.isfinite(arr)).sum()), int(ab.sum()))
    except OSError:
        return None


def fmt(t):
    return "    -  /  -  /  - " if t is None else f"{t[0]:5d} /{t[1]:4d} /{t[2]:4d}"


def main():
    tags = sys.argv[1:] or list(REGISTERED)
    unknown = [t for t in tags if t not in REGISTERED]
    if unknown:
        print(f"not registered epochs: {unknown}")
        return 2
    if not SWEEP.is_dir():
        print(f"{SWEEP} does not exist; nothing has been swept yet")
        return 1

    print(f"  {'epoch':<16}{'planner':<7}{'feas/inf/abort':>17}"
          f"{'  was (pre-fix)':>17}{'search s':>10}  status")
    bad, bad_replay = 0, 0
    for tag in tags:
        et, tm = transitions_now(tag)
        if et is None:
            print(f"  {tag:<16}  no interval table on disk; build it first")
            bad += 2
            continue
        for kind in KINDS:
            d = SWEEP / f"{tag}_{kind}"
            if not d.is_dir():
                print(f"  {tag:<16}{kind:<7}  NOT RUN ({d.name} absent)")
                bad += 1
                continue
            probs, triple, secs = check_job(d, tag, kind, et, tm)
            if probs:
                bad += 1
            print(f"  {tag:<16}{kind:<7}{fmt(triple):>17}"
                  f"{fmt(old_triple(tag, kind)):>17}{secs:>10.0f}  "
                  f"{'ok' if not probs else 'BAD'}")
            for p in probs:
                print(f"      {p}")
        keyed = [k for k in REPLAY_KEYS
                 if (SWEEP / f"{tag}_replay_{k}" / "out_verdict.bin").exists()]
        counts = {k: replayable_plans(SWEEP / f"{tag}_{k}") for k in KINDS}
        empty = sorted(k for k, c in counts.items() if c == 0)
        miss = [k for k in REPLAY_KEYS if k not in keyed]
        # Saying "absent" of a planner that produced no plan reads as a
        # missing artifact when it is the outcome: nothing to replay.
        unexplained = [k for k in miss if k.rsplit("_", 1)[0] not in empty]
        if not keyed and unexplained:
            print(f"  {tag:<16}  REPLAY STAGE DID NOT RUN: none of the "
                  f"{len(REPLAY_KEYS)} {tag}_replay_<planner>_<label> "
                  f"directories holds a verdict")
            bad_replay += 1
        else:
            note = ("; " + " and ".join(empty) + " produced 0 plans, "
                    "nothing to replay") if empty else ""
            print(f"  {tag:<16}  replay: {len(keyed)} of {len(REPLAY_KEYS)} "
                  f"planner-label directories"
                  + note
                  + (f"; UNEXPLAINED absent {unexplained}" if unexplained
                     else ("" if note else ", all present")))
        orphan = SWEEP / f"{tag}_replay"
        if orphan.is_dir():
            print(f"  {tag:<16}  note: {orphan.name} is a leftover from the "
                  f"single-shared-directory layout and is no longer read; "
                  f"remove it so it cannot be mistaken for a result")
    print()
    print(f"{2 * len(tags) - bad} of {2 * len(tags)} planner jobs verified "
          f"over {len(tags)} epochs; "
          + (f"{bad_replay} epoch(s) have no replay stage"
             if bad_replay else "every epoch has a replay stage"))
    done = sorted({p.name.rsplit('_', 1)[0] for p in SWEEP.glob("*_td")})
    print(f"{len(done)} of {len(REGISTERED)} registered epochs have planner "
          f"output; absent: {[t for t in REGISTERED if t not in done] or 'none'}")
    return 1 if (bad or bad_replay) else 0


if __name__ == "__main__":
    sys.exit(main())
