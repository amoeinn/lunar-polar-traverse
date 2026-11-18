"""check_axes judges a job directory by what is in it.

The fault this protects against was found by running the tool against the
pre-fix axis sweep rather than against freshly written output. run_cpp records
absolute output paths in planner_params.json, so a directory that has been
moved, which every pre-fix sweep in this project has been, records its outputs
at a location that no longer exists. The checker followed those paths and
called seven of eight complete jobs incomplete, with every file present. A
checker that is wrong about intact data is worse than none, because its failure
looks like a data failure.
"""
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import examples.plan_pilot as P                                   # noqa: E402

_spec = importlib.util.spec_from_file_location(
    "check_axes", ROOT / "examples" / "check_axes.py")
CA = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(CA)

N = 6


def job(d, elsewhere, n=N, total=13):
    """A complete job whose params point at `elsewhere`, not at `d`."""
    d.mkdir(parents=True, exist_ok=True)
    params = {"n_instances": n}
    for k, (dt, w) in P.OUT_WIDTH.items():
        np.zeros(n * w, dtype=dt).tofile(d / f"out_{k}.bin")
        params[f"out_{k}"] = str(Path(elsewhere) / f"out_{k}.bin")
    off = np.linspace(0, total, n + 1).astype(np.int64)
    off[-1] = total
    off.tofile(d / "out_path_offsets.bin")
    np.zeros(total, dtype=np.int32).tofile(d / "out_path_cells.bin")
    np.zeros(total, dtype=np.float64).tofile(d / "out_path_times.bin")
    for k in ("path_offsets", "path_cells", "path_times"):
        params[f"out_{k}"] = str(Path(elsewhere) / f"out_{k}.bin")
    (d / "planner_params.json").write_text(json.dumps(params))
    return params


def test_a_moved_but_complete_job_verifies(tmp_path):
    """The fault. Params name a directory that does not exist; every output is
    present where the checker is looking."""
    d = tmp_path / "moved"
    job(d, tmp_path / "gone")
    assert not (tmp_path / "gone").exists()
    assert P.output_problems(CA.localised(d), want_paths=True) == [], \
        "a complete job in a moved directory was called incomplete"


def test_the_unlocalised_params_really_are_broken(tmp_path):
    """Discriminating: without this, localised() could be a no-op and the test
    above would still pass."""
    d = tmp_path / "moved"
    params = job(d, tmp_path / "gone")
    assert P.output_problems(params, want_paths=True) != [], \
        "the fixture does not reproduce the fault it was built for"


def test_localisation_does_not_invent_completeness(tmp_path):
    """It must not paper over a genuinely incomplete job: redirecting the paths
    is only about where to look, never about what must be there."""
    d = tmp_path / "moved"
    job(d, tmp_path / "gone")
    (d / "out_stats.bin").unlink()
    probs = P.output_problems(CA.localised(d), want_paths=True)
    assert probs and "out_stats.bin missing" in probs[0]


def test_a_truncated_output_is_still_caught_after_localisation(tmp_path):
    d = tmp_path / "moved"
    job(d, tmp_path / "gone")
    f = d / "out_arrival.bin"
    f.write_bytes(f.read_bytes()[:-8])
    assert P.output_problems(CA.localised(d), want_paths=True) != []


def test_no_params_file_is_not_a_job(tmp_path):
    d = tmp_path / "bare"
    d.mkdir()
    assert CA.localised(d) is None


def test_corrupt_params_is_not_a_job(tmp_path):
    d = tmp_path / "bad"
    d.mkdir()
    (d / "planner_params.json").write_text("{not json")
    assert CA.localised(d) is None


def test_localisation_leaves_the_directory_untouched(tmp_path):
    """It reads; it must not rewrite the params to the new location, which
    would make them newer than the outputs they describe."""
    import os
    d = tmp_path / "moved"
    job(d, tmp_path / "gone")
    old = 1_700_000_000
    for f in d.iterdir():
        os.utime(f, (old, old))
    before = {f.name: (f.stat().st_mtime_ns, f.stat().st_size) for f in d.iterdir()}
    CA.localised(d)
    after = {f.name: (f.stat().st_mtime_ns, f.stat().st_size) for f in d.iterdir()}
    assert before == after


def test_an_unknown_axis_is_refused(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["check_axes.py", "not_an_axis"])
    assert CA.main() == 2


def test_the_registered_axes_are_the_drivers_own(monkeypatch):
    import examples.sweep_axes as X
    assert set(CA.AXES) == set(X.AXES)
