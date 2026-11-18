"""check_sweep's account of an epoch whose planners produced no plans.

The fault this protects against is a cosmetic change that hides a real gap.
A planner that finds no feasible traverse at an epoch produces no replay
directory, because the driver skips a planner with no plans, and reporting
that as "absent" reads as a missing artifact when it is the recorded outcome.
Explaining those absences must not also explain away a directory that is
genuinely missing, which is the only way this reporting can do harm.
"""
import importlib.util
import io
import sys
from contextlib import redirect_stdout
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

_spec = importlib.util.spec_from_file_location(
    "cs_report", ROOT / "examples" / "check_sweep.py")
CS = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(CS)


def planner_dir(d, n_plans, n_inst=120):
    """A planner job directory whose first n_plans instances carry a path."""
    d.mkdir(parents=True, exist_ok=True)
    lens = np.array([3] * n_plans + [1] * (n_inst - n_plans), dtype=np.int64)
    off = np.concatenate(([0], np.cumsum(lens)))
    off.tofile(d / "out_path_offsets.bin")
    return off


def test_a_planner_with_no_plans_is_counted_as_zero(tmp_path):
    planner_dir(tmp_path / "job", 0)
    assert CS.replayable_plans(tmp_path / "job") == 0


def test_a_planner_with_plans_is_counted(tmp_path):
    """Discriminating: without this, returning 0 always would pass above."""
    planner_dir(tmp_path / "job", 7)
    assert CS.replayable_plans(tmp_path / "job") == 7


def test_a_single_cell_path_is_not_a_plan(tmp_path):
    """The driver's rule is hi - lo < 2, so a one-cell path is not replayed;
    counting it would make the report claim a plan that was never replayed."""
    d = tmp_path / "job"
    d.mkdir()
    np.array([0, 1, 2, 3], dtype=np.int64).tofile(d / "out_path_offsets.bin")
    assert CS.replayable_plans(d) == 0


def test_no_planner_output_is_unknown_not_zero(tmp_path):
    """None and 0 must not be conflated: an absent job is not a job that
    found nothing, and only the second explains a missing replay directory."""
    assert CS.replayable_plans(tmp_path / "nothing") is None


def _report(tmp_path, monkeypatch, td_plans, snap_plans, keyed):
    """Run main() over one epoch with a synthetic sweep directory."""
    tag = CS.REGISTERED[0]
    sweep = tmp_path / "sweep"
    for kind, n in (("td", td_plans), ("snap", snap_plans)):
        d = sweep / f"{tag}_{kind}"
        planner_dir(d, n)
    for k in keyed:
        kd = sweep / f"{tag}_replay_{k}"
        kd.mkdir(parents=True)
        (kd / "out_verdict.bin").write_bytes(b"\x01")
    monkeypatch.setattr(CS, "SWEEP", sweep)
    # the planner-job half of the check is not what these cases are about
    monkeypatch.setattr(CS, "check_job",
                        lambda *a, **k: ([], (1, 0, 0), 0.0))
    monkeypatch.setattr(CS, "transitions_now", lambda t: (1, 0.0))
    monkeypatch.setattr(sys, "argv", ["check_sweep.py", tag])
    buf = io.StringIO()
    with redirect_stdout(buf):
        rc = CS.main()
    return rc, buf.getvalue()


MM = [f"mm{e}_{l}" for e in (0.0, 0.5, 1.0, 2.0) for l in ("tol", "zero")]


def test_zero_plan_planners_are_explained_and_not_failed(tmp_path, monkeypatch):
    """Both planners found nothing; the eight mean-map directories are there.
    That epoch is complete and must not be reported as absent."""
    rc, out = _report(tmp_path, monkeypatch, 0, 0, MM)
    line = next(l for l in out.splitlines() if "replay:" in l)
    assert "produced 0 plans, nothing to replay" in line
    assert "absent" not in line, line
    assert rc == 0, out


def test_a_genuinely_missing_directory_is_still_reported(tmp_path, monkeypatch):
    """The hazard. Both planners DID produce plans, so nothing explains the
    missing mean-map directory, and the explanation must not absorb it."""
    rc, out = _report(tmp_path, monkeypatch, 5, 5, MM[:-1] + ["td_tol", "td_zero",
                                                             "snap_tol", "snap_zero"])
    assert "UNEXPLAINED absent" in out
    assert "mm2.0_zero" in out


def test_an_unexplained_absence_is_not_hidden_by_an_explained_one(
        tmp_path, monkeypatch):
    """Mixed case: td found nothing, which explains its two directories, while
    a mean-map directory is missing for no reason. Both facts must appear."""
    rc, out = _report(tmp_path, monkeypatch, 0, 5, MM[:-1] + ["snap_tol", "snap_zero"])
    assert "td produced 0 plans" in out
    assert "UNEXPLAINED absent" in out and "mm2.0_zero" in out
    assert "td_tol" not in out.split("UNEXPLAINED")[1]


def test_a_planner_with_plans_whose_directory_is_missing_is_unexplained(
        tmp_path, monkeypatch):
    """The case that separates "explained by having no plans" from "explained
    by being a planner at all". Both planners produced plans and the snapshot
    directories are gone, so its absence has no explanation and the report
    must not claim it produced nothing."""
    rc, out = _report(tmp_path, monkeypatch, 5, 5, MM + ["td_tol", "td_zero"])
    line = next(l for l in out.splitlines() if "replay:" in l)
    assert "UNEXPLAINED absent" in line, line
    assert "snap_tol" in line and "snap_zero" in line
    assert "produced 0 plans" not in line, line


def test_a_replay_stage_that_never_ran_still_fails(tmp_path, monkeypatch):
    """With plans produced and no replay directory at all, this is the run
    that died before replaying, and it must not be explained away."""
    rc, out = _report(tmp_path, monkeypatch, 5, 5, [])
    assert "REPLAY STAGE DID NOT RUN" in out
    assert rc != 0
