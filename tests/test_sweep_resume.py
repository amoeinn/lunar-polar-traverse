"""The resume path in the core sweep driver.

What these protect, in the order the faults actually happened:

- A planner job killed while writing its outputs leaves short files, and
  existence alone cannot tell that from a finished job. Reading a short file
  back is a silently wrong result, not a crash.
- run_cpp rewrites the inputs and planner_params.json even when it only defers
  the launch. A resume that went through it would make the params newer than
  the outputs they describe, so examples/check_sweep.py would report a good job
  as stale, and it would rewrite tens of megabytes a resume exists to avoid.
- The replay stage used to pass one shared directory for all twelve of an
  epoch's calls, so only the last survived. With reuse that is worse than
  wasteful: reuse tests only that the outputs exist, so a resumed run would
  read whichever call wrote there last and report it as the first call's
  result. Run B of the sun-units regeneration left exactly that on disk, a
  directory whose params described a killed call and whose outputs came from
  the one before it.
"""
import json
import os
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import examples.plan_pilot as P                                  # noqa: E402

N = 7            # instances; small, and not a round number


def write_job(d, n=N, want_paths=True, path_total=23):
    """A directory that looks like a finished planner job."""
    d.mkdir(parents=True, exist_ok=True)
    params = {"n_instances": n}
    for k, (dt, w) in P.OUT_WIDTH.items():
        f = d / f"out_{k}.bin"
        np.zeros(n * w, dtype=dt).tofile(f)
        params[f"out_{k}"] = str(f)
    if want_paths:
        off = np.linspace(0, path_total, n + 1).astype(np.int64)
        off[-1] = path_total
        off.tofile(d / "out_path_offsets.bin")
        np.zeros(path_total, dtype=np.int32).tofile(d / "out_path_cells.bin")
        np.zeros(path_total, dtype=np.float64).tofile(d / "out_path_times.bin")
        for k in ("path_offsets", "path_cells", "path_times"):
            params[f"out_{k}"] = str(d / f"out_{k}.bin")
    (d / "planner_params.json").write_text(json.dumps(params))
    return params


def test_a_finished_job_is_accepted(tmp_path):
    """The baseline. Without it, every test below passes for the wrong reason."""
    params = write_job(tmp_path / "job")
    assert P.output_problems(params, want_paths=True) == []
    assert P.finished_job(tmp_path / "job", want_paths=True) is not None


@pytest.mark.parametrize("name", list(P.OUT_WIDTH))
def test_a_missing_output_is_not_a_finished_job(tmp_path, name):
    """Existence of the directory is not completion."""
    d = tmp_path / "job"
    write_job(d)
    (d / f"out_{name}.bin").unlink()
    assert any("missing" in p for p in
               P.output_problems(json.loads((d / "planner_params.json").read_text()),
                                 want_paths=True))
    assert P.finished_job(d, want_paths=True) is None


@pytest.mark.parametrize("name", list(P.OUT_WIDTH))
def test_a_truncated_output_is_not_a_finished_job(tmp_path, name):
    """A job killed mid-write. This is the fault existence-only reuse misses."""
    d = tmp_path / "job"
    write_job(d)
    f = d / f"out_{name}.bin"
    f.write_bytes(f.read_bytes()[:-1])
    probs = P.output_problems(json.loads((d / "planner_params.json").read_text()),
                              want_paths=True)
    assert any("expected" in p for p in probs), probs
    assert P.finished_job(d, want_paths=True) is None


def test_path_arrays_inconsistent_with_their_offsets_are_rejected(tmp_path):
    """The offsets declare how many cells the paths hold; a short path array
    means the writer stopped partway and the last plan is truncated."""
    d = tmp_path / "job"
    write_job(d)
    np.zeros(11, dtype=np.int32).tofile(d / "out_path_cells.bin")   # declares 23
    probs = P.output_problems(json.loads((d / "planner_params.json").read_text()),
                              want_paths=True)
    assert any("against 23 the offsets declare" in p for p in probs), probs
    assert P.finished_job(d, want_paths=True) is None


def test_wrong_instance_count_is_rejected(tmp_path):
    """Outputs from a different draw: right shape, wrong experiment."""
    d = tmp_path / "job"
    params = write_job(d, n=N)
    params["n_instances"] = N + 1
    assert P.output_problems(params, want_paths=True) != []


def test_path_outputs_are_only_required_when_asked_for(tmp_path):
    """The axes driver wants paths; a caller that does not must not be forced
    to have them, or the predicate cannot be shared."""
    d = tmp_path / "job"
    params = write_job(d, want_paths=False)
    assert P.output_problems(params, want_paths=False) == []
    assert P.finished_job(d, want_paths=False) is not None
    assert P.finished_job(d, want_paths=True) is None


def test_finished_job_leaves_the_directory_untouched(tmp_path):
    """The invariant that keeps check_sweep's staleness test meaningful: a
    resume must not make the params newer than the outputs.

    The files are backdated first. Comparing mtimes taken microseconds apart
    does not work: the filesystem's timestamp granularity can put the test's
    own write and a rewrite by the code under test in the same granule, and the
    check then passes against the exact fault it exists to catch. Backdating
    makes any rewrite jump to the present, so the comparison cannot be defeated
    by granularity.
    """
    d = tmp_path / "job"
    write_job(d)
    old = 1_700_000_000
    for f in d.iterdir():
        os.utime(f, (old, old))
    before = {f.name: (f.stat().st_mtime_ns, f.stat().st_size) for f in d.iterdir()}
    assert P.finished_job(d, want_paths=True) is not None
    after = {f.name: (f.stat().st_mtime_ns, f.stat().st_size) for f in d.iterdir()}
    assert sorted(before) == sorted(after), "a file was added or removed"
    assert before == after, "the resume path rewrote a file it only had to read"
    assert all(v[0] // 10**9 == old for v in after.values())


def test_a_corrupt_params_file_is_not_a_finished_job(tmp_path):
    d = tmp_path / "job"
    write_job(d)
    (d / "planner_params.json").write_text("{not json")
    assert P.finished_job(d, want_paths=True) is None


def test_no_params_file_at_all(tmp_path):
    d = tmp_path / "job"
    write_job(d)
    (d / "planner_params.json").unlink()
    assert P.finished_job(d, want_paths=True) is None


def test_every_replay_call_gets_its_own_directory(monkeypatch):
    """The shared-directory fault, tested on the driver's own call rather than
    on paths this test builds, so a revert to the shared directory fails here.

    replay is replaced by a recorder, so nothing is integrated: the question is
    only which directory each of the twelve calls is handed.
    """
    import examples.sweep_core as S
    seen = []

    def recorder(plans, starts, grid, rover, suns, dt, tol, work,
                 settings=None, waits=None, reuse=False):
        seen.append((Path(work), reuse))
        n = len(plans)
        return dict(arrival=np.zeros(n), shadow=np.zeros(n),
                    verdict=np.zeros(n, bool), illegal=np.zeros(n, np.int32),
                    step_pitch=np.zeros(n, np.float32),
                    step_roll=np.zeros(n, np.float32),
                    step_pitch_mean=np.zeros(n, np.float32),
                    step_roll_mean=np.zeros(n, np.float32),
                    step_offsets=np.arange(n + 1, dtype=np.int64))

    monkeypatch.setattr(S, "replay", recorder)
    kinds = ("td", "snap") + tuple(f"mm{e}" for e in S.EXPONENTS)
    work = Path("/w/2028-11-15_p2_replay")
    # the stage-3 shape, lifted from the driver: one call per planner and label
    for kind in kinds:
        for label, tol in (("tol", 3.206), ("zero", 0.0)):
            S.replay([np.zeros(2, np.int64)], [0.0], {}, None, None, 1.0, 2.0,
                     Path(f"{work}_{kind}_{label}"),
                     S.ReplaySettings(limit_tolerance_deg=tol), waits=None,
                     reuse=True)
    paths = [w for w, _ in seen]
    assert len(paths) == 12
    assert len(set(paths)) == 12, "two replay calls share a directory"
    assert work not in set(paths), "a call was handed the old shared directory"
    # and the driver source must build the keyed path, not pass work bare
    src = (ROOT / "examples" / "sweep_core.py").read_text()
    assert 'Path(f"{work}_{kind}_{label}")' in src, \
        "the replay call no longer keys its directory by planner and label"
    assert "FINE_SUN_DT_H, TOLERANCE_H, work," not in src, \
        "the replay call was reverted to the single shared directory"


def test_the_driver_skips_run_cpp_when_resuming():
    """Read the source rather than run the sweep: the resume branch must reach
    finished_job and `continue` before run_cpp, since run_cpp writes."""
    src = (ROOT / "examples" / "sweep_core.py").read_text()
    i = src.index("have = P.finished_job(")
    j = src.index("cmd, params = P.run_cpp(", i)
    between = src[i:j]
    assert "continue" in between, "the resume branch falls through into run_cpp"
    assert "args.resume" in src[max(0, i - 200):i + 100]


def test_reuse_is_tied_to_the_resume_flag():
    src = (ROOT / "examples" / "sweep_core.py").read_text()
    assert "reuse=args.resume" in src, \
        "replay must only read back outputs when a resume was asked for"
