"""The registered arrival criterion, scored signed, for every planner on the core.

Registered 2026-10-02, before any sweep plan existed: the checker's integrated
arrival must agree with the planner's summed arrival to within 1%. It was first
scored on 2026-10-03 as met on 43.3% of time-dependent plans, and re-scored
after the sun units fix, both times on a checker whose step length was short by
cos(pitch) (fixed in 6bb44a9). This scores it on the corrected replays.

Nothing is replanned or replayed. The plans and both planners' claimed arrivals
come from the sweep's planner outputs; the mean-map routes are rebuilt exactly
as score_phase3 rebuilds them, and its reconstruction check refuses any rebuilt
route that is not the plan the sweep replayed; every checker arrival is read
back from the sweep's own replay directory. The modification time of every
replay out_arrival.bin is compared before and after, so a run that recomputed a
replay would say so.

Per plan: (checker - planner) / planner, signed, from src/arrival.py. Reported
per planner: the count within 1% either way, the counts later and earlier, and
the signed median. The registration states no aggregation over plans. The
2026-10-03 scoring recorded FAIL with "on the majority of plans the checker's
continuous integration and the planner's per-edge summation differ by more than
1%", which describes the result without stating a rule, so no verdict is
derived here: the counts are written and the reading is left to the record.

    .venv/bin/python examples/score_arrival.py

writes data/work/arrival_scores.json.
"""
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import examples.plan_pilot as P                                   # noqa: E402
import examples.score_phase3 as S3                                # noqa: E402
import examples.sweep_core as CO                                  # noqa: E402
from examples.build_intervals import epochs                        # noqa: E402
from src.arrival import ARRIVAL_CRITERION, arrival_gap, describe, summarise  # noqa: E402
from src.cost import Rover                                         # noqa: E402

OUT = ROOT / "data" / "work" / "arrival_scores.json"
SWEEP = P.DATA / "work" / "sweep"


def gaps(rows):
    """The signed gap per plan, checker against the planner that made it."""
    return arrival_gap([r["checked"] for r in rows], [r["claimed"] for r in rows])


def score_kind(rows):
    return summarise(gaps(rows), ARRIVAL_CRITERION)


def replay_mtimes():
    return {str(p): p.stat().st_mtime_ns
            for p in sorted(SWEEP.glob("*_replay_*/out_arrival.bin"))}


def head_commit():
    return subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"],
                          capture_output=True, text=True).stdout.strip()


def main():
    before = replay_mtimes()
    site = S3.Site()
    rover = Rover(v_flat=50.0)
    exps = CO.EXPONENTS
    recs = {}
    for ei, tag in enumerate(t for t, _, _ in epochs()):
        recs[tag] = S3.collect(site, SWEEP, "", tag, rover, CO.TOLERANCE_H, ei,
                               CO.draw, exps)
        print(f"  {tag} collected", flush=True)
    print(S3.recon_report(recs, exps))
    after = replay_mtimes()
    changed = [p for p in before if after.get(p) != before[p]]
    if changed or set(after) != set(before):
        raise SystemExit(f"replay outputs changed during scoring: {changed[:5]}; "
                         f"the scores would not be read from the sweep's replays")
    print(f"  {len(before)} replay out_arrival.bin files, none modified")

    kinds = ["td", "snap"] + [f"mm{e}" for e in exps]
    scores = {}
    print(f"\nthe arrival criterion, |(checker - planner) / planner| <= "
          f"{ARRIVAL_CRITERION:.0%}, per planner, every replayed plan, core:")
    for kind in kinds:
        s = score_kind(S3.all_rows(recs, kind))
        scores[kind] = s
        print(f"  {kind:<7} {describe(s)}")
    out = dict(criterion=ARRIVAL_CRITERION,
               quantity="(checker - planner) / planner, signed; positive is "
                        "checker later",
               population="every plan the sweep replayed on the core, 12 epochs, "
                          "checker at the registered limit tolerance",
               verdict_rule="none registered; counts only",
               planners=scores, head_commit=head_commit(),
               written_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
    OUT.write_text(json.dumps(out, indent=1) + "\n")
    print(f"\nwritten to {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
