"""Verify one axis sweep's planner jobs, and say how far the stages got.

    .venv/bin/python examples/check_axes.py <axis> [<work dir>]

The axis is one of sweep_axes' own names. The optional work directory lets the
pre-fix copy under data/work/pre_sunfix/axes be checked the same way.

Completeness is plan_pilot.output_problems, the same predicate the drivers use
to decide whether a job can be read back, so there is one definition of a
finished planner job. Each point and epoch is expected to produce two planner
jobs and, once replayed, twelve keyed replay directories; a planner that found
no plan produces none for itself, which is reported as such rather than as a
gap, as examples/check_sweep.py does for the core.

Read only; writes nothing.
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import examples.plan_pilot as P                                   # noqa: E402
from examples.sweep_axes import AXES, EPOCHS, EXPONENTS           # noqa: E402

KINDS = ("td", "snap")
REPLAY_KEYS = tuple(f"{k}_{lbl}"
                    for k in KINDS + tuple(f"mm{e}" for e in EXPONENTS)
                    for lbl in ("tol", "zero"))


def localised(d):
    """A job's params with every output path pointed at the directory being
    inspected. run_cpp writes absolute paths, so a job directory that has been
    moved, as the pre-fix sweeps were, records outputs at a location that no
    longer exists and would be judged incomplete when every file is present.
    A directory is judged by what is in it."""
    pf = Path(d) / "planner_params.json"
    if not pf.exists():
        return None
    try:
        params = json.loads(pf.read_text())
    except (ValueError, OSError):
        return None
    for k in list(P.OUT_WIDTH) + ["path_cells", "path_times", "path_offsets"]:
        if f"out_{k}" in params:
            params[f"out_{k}"] = str(Path(d) / f"out_{k}.bin")
    return params


def replayable_plans(d):
    f = Path(d) / "out_path_offsets.bin"
    if not f.exists():
        return None
    return int((np.diff(np.fromfile(f, dtype=np.int64)) >= 2).sum())


def main():
    if len(sys.argv) < 2 or sys.argv[1] not in AXES:
        print(f"usage: check_axes.py <axis> [<work dir>]; axes: {sorted(AXES)}")
        return 2
    axis = sys.argv[1]
    work = Path(sys.argv[2]) if len(sys.argv) > 2 else ROOT / "data" / "work" / "axes"
    if not work.is_dir():
        print(f"{work} does not exist; nothing has been swept on this axis")
        return 1
    print(f"axis {axis}: {len(AXES[axis])} points x {len(EPOCHS)} epochs = "
          f"{2 * len(AXES[axis]) * len(EPOCHS)} planner jobs, under {work}")
    print(f"  {'point':<16}{'epoch':<16}{'kind':<6}{'plans':>7}{'search s':>10}"
          f"  status")
    bad = bad_replay = 0
    for pi, point in enumerate(AXES[axis]):
        for tag in EPOCHS:
            for kind in KINDS:
                d = work / f"{axis}_{pi}_{tag}_{kind}"
                params = localised(d)
                probs = (P.output_problems(params, want_paths=True)
                         if params is not None else None)
                if params is None or probs:
                    why = ("NOT RUN" if params is None
                           else "INCOMPLETE: " + "; ".join(probs[:2]))
                    print(f"  {point['label']:<16}{tag:<16}{kind:<6}{'-':>7}"
                          f"{'-':>10}  {why}")
                    bad += 1
                    continue
                s = np.fromfile(d / "out_seconds.bin", dtype=np.float64).sum()
                print(f"  {point['label']:<16}{tag:<16}{kind:<6}"
                      f"{replayable_plans(d):>7}{s:>10.0f}  ok")
            counts = {k: replayable_plans(work / f"{axis}_{pi}_{tag}_{k}")
                      for k in KINDS}
            empty = sorted(k for k, c in counts.items() if c == 0)
            keyed = [k for k in REPLAY_KEYS
                     if (work / f"{axis}_{pi}_{tag}_replay_{k}"
                         / "out_verdict.bin").exists()]
            miss = [k for k in REPLAY_KEYS if k not in keyed]
            unexplained = [k for k in miss if k.rsplit("_", 1)[0] not in empty]
            if not keyed and unexplained:
                print(f"  {point['label']:<16}{tag:<16}  REPLAY STAGE DID NOT RUN")
                bad_replay += 1
            else:
                note = ("; " + " and ".join(empty) + " produced 0 plans, "
                        "nothing to replay") if empty else ""
                print(f"  {point['label']:<16}{tag:<16}  replay: {len(keyed)} of "
                      f"{len(REPLAY_KEYS)}" + note
                      + (f"; UNEXPLAINED absent {unexplained}" if unexplained
                         else ("" if note else ", all present")))
    n = 2 * len(AXES[axis]) * len(EPOCHS)
    print()
    print(f"{n - bad} of {n} planner jobs verified; "
          + (f"{bad_replay} point-epoch(s) have no replay stage"
             if bad_replay else "every point-epoch has a replay stage"))
    return 1 if (bad or bad_replay) else 0


if __name__ == "__main__":
    sys.exit(main())
