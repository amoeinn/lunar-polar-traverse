"""--stages, and the replay reuse check.

What these protect:

- A long epoch had to be run into its timeout on purpose so that the planner
  stage would land on disk and a second pass could finish the rest. That is a
  kill dressed up as a plan: the log is lost to buffering, the replay stage is
  left half-written, and nothing distinguishes it from a real failure.
  --stages makes it two runs that are each expected to succeed.
- replay(reuse=True) judged a directory complete because nine files existed in
  it. cpp/build/replay writes at the end, so a killed call leaves short files,
  and a directory can hold a different call's outputs entirely: the run killed
  on 2026-11-16 left params describing a call that never wrote beside outputs
  from the call before it, 120 mean-map verdicts where a 64-plan time-dependent
  call would have read them back as its own.

Every validation case asserts that the refusal happens before the region is
loaded, by making the load raise. A check that runs after a 343 MB load and 82
MB of writes is not a check, it is a cleanup problem.
"""
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import examples.plan_pilot as P                                   # noqa: E402
import examples.sweep_core as S                                   # noqa: E402
from src.replay import PER_PLAN, PER_STEP, reuse_problems          # noqa: E402

TAG = "2028-02-15_p0"


@pytest.fixture
def no_load(monkeypatch):
    """Any path that reaches the region load has failed to refuse early."""
    def boom(*a, **k):
        raise AssertionError("P.load was reached; the argument was not refused")
    monkeypatch.setattr(P, "load", boom)


def run_main(monkeypatch, argv):
    monkeypatch.setattr(sys, "argv", ["sweep_core.py"] + argv)
    return S.main()


@pytest.mark.parametrize("arg,why", [
    ("bogus", "an unknown stage name"),
    ("planner,bogus", "one good name and one bad"),
    ("", "an empty selection"),
    (",", "only separators"),
    ("replay", "replay without mean-map, whose plans are never on disk"),
    ("planner,replay", "replay without mean-map, with the planner"),
])
def test_a_bad_stage_selection_is_refused_before_anything_loads(
        monkeypatch, no_load, arg, why):
    assert run_main(monkeypatch, ["--tags", TAG, "--stages", arg]) == 2, why


@pytest.mark.parametrize("arg", ["all", "planner", "mean-map",
                                 "planner,mean-map", "mean-map,replay",
                                 "planner,mean-map,replay", " planner , replay , mean-map "])
def test_a_good_stage_selection_is_accepted(monkeypatch, arg):
    """It must get past validation, which here means reaching the load."""
    reached = []
    monkeypatch.setattr(P, "load", lambda *a, **k: reached.append(1) or (_ for _ in ()).throw(SystemExit(99)))
    with pytest.raises(SystemExit) as e:
        run_main(monkeypatch, ["--tags", TAG, "--stages", arg])
    assert e.value.code == 99 and reached, f"{arg} was refused"


def test_replay_needs_mean_map_in_the_same_run():
    """The reason, stated as a fact about the driver rather than a message:
    the mean-map stage writes nothing, so its plans cannot be read back."""
    src = (ROOT / "examples" / "sweep_core.py").read_text()
    assert '"replay" in stages and "mean-map" not in stages' in src
    # and stage 2 really does not persist its plans
    i = src.index("# --- stage 2: mean-map")
    j = src.index("# --- stage 3: replay")
    assert ".save(" not in src[i:j] and "tofile" not in src[i:j]


def test_each_stage_is_actually_guarded():
    """Each guard is looked for where it has to be, not anywhere in the file.
    The validation block mentions mean-map too, so a search over the whole
    source passes even with the stage-2 guard removed."""
    src = (ROOT / "examples" / "sweep_core.py").read_text()
    s1 = src.index("# --- stage 1:")
    s2 = src.index("# --- stage 2: mean-map")
    s3 = src.index("# --- stage 3: replay")
    # the planner guard sits inside stage 1 and before run_cpp writes anything
    i = src.index('if "planner" not in stages:', src.index("for kind, table in"))
    assert s1 < i < src.index("cmd, params = P.run_cpp(")
    # the mean-map guard sits between the planner stage and stage 2
    assert '"mean-map" not in stages' in src[i:s2], \
        "nothing stops the mean-map stage from running when it was not asked for"
    # and the replay guard between stage 2 and stage 3
    assert '"replay" not in stages' in src[s2:s3], \
        "nothing stops the replay stage from running when it was not asked for"


def test_the_stage_names_are_the_three_stages():
    assert S.STAGES == ("planner", "mean-map", "replay")


# ---------------------------------------------------------------- replay reuse

def make_call(d, n_plans=5, steps_per=4):
    """A directory holding a finished replay call for n_plans plans."""
    d.mkdir(parents=True, exist_ok=True)
    cells_per = steps_per + 1
    off = np.arange(0, (n_plans + 1) * cells_per, cells_per, dtype=np.int64)
    files = {
        "plan_offsets_path": off,
        "plan_cells_path": np.arange(off[-1], dtype=np.int32),
        "plan_start_path": np.arange(n_plans, dtype=np.float64),
        "plan_wait_path": np.zeros(off[-1], dtype=np.float64),
    }
    for k, v in files.items():
        v.tofile(d / f"{k}.bin")
    sout = np.arange(0, (n_plans + 1) * steps_per, steps_per, dtype=np.int64)
    sout.tofile(d / "out_step_offsets.bin")
    for k, dt in PER_PLAN.items():
        np.zeros(n_plans, dtype=dt).tofile(d / f"out_{k}.bin")
    for k in PER_STEP:
        np.zeros(int(sout[-1]), dtype=np.float32).tofile(d / f"out_{k}.bin")
    return files


def test_a_finished_call_is_reusable(tmp_path):
    """Baseline, without which nothing below means anything."""
    f = make_call(tmp_path / "c")
    assert reuse_problems(tmp_path / "c", f, 5) == []


def test_a_call_for_a_different_number_of_plans_is_not_reusable(tmp_path):
    """The run B fault: 120 mean-map verdicts offered to a 64-plan call."""
    f = make_call(tmp_path / "c", n_plans=5)
    f2 = make_call(tmp_path / "other", n_plans=3)
    probs = reuse_problems(tmp_path / "c", f2, 3)
    assert probs and "values against" in probs[0]


def test_the_same_shape_but_different_plans_is_not_reusable(tmp_path):
    """Two mean-map exponents give 120 plans each with different routes, so
    size alone cannot tell them apart; the inputs must match too."""
    d = tmp_path / "c"
    f = make_call(d)
    f = dict(f)
    f["plan_cells_path"] = f["plan_cells_path"][::-1].copy()
    probs = reuse_problems(d, f, 5)
    assert probs and "different call" in probs[0]


@pytest.mark.parametrize("name", list(PER_PLAN) + list(PER_STEP)
                         + ["step_offsets"])
def test_a_missing_output_is_not_reusable(tmp_path, name):
    d = tmp_path / "c"
    f = make_call(d)
    (d / f"out_{name}.bin").unlink()
    assert reuse_problems(d, f, 5) != []


@pytest.mark.parametrize("name", list(PER_PLAN) + list(PER_STEP))
def test_a_truncated_output_is_not_reusable(tmp_path, name):
    """A call killed while writing. This is what existence could not see."""
    d = tmp_path / "c"
    f = make_call(d)
    p = d / f"out_{name}.bin"
    p.write_bytes(p.read_bytes()[:-4])
    probs = reuse_problems(d, f, 5)
    assert probs and "bytes against" in " ".join(probs)


def test_missing_plan_inputs_make_it_unreusable(tmp_path):
    """Without the inputs there is nothing tying the outputs to these plans."""
    d = tmp_path / "c"
    f = make_call(d)
    (d / "plan_cells_path.bin").unlink()
    probs = reuse_problems(d, f, 5)
    assert probs and "cannot be tied" in probs[0]


def test_step_offsets_of_the_wrong_length_is_not_reusable(tmp_path):
    """The one fault only the length check can see. An extra offsets entry that
    repeats the final value leaves the step total unchanged, so every per-step
    size still matches and the inputs still match; only the n+1 length is
    wrong."""
    d = tmp_path / "c"
    f = make_call(d, n_plans=5)
    off = np.fromfile(d / "out_step_offsets.bin", dtype=np.int64)
    np.append(off, off[-1]).tofile(d / "out_step_offsets.bin")
    probs = reuse_problems(d, f, 5)
    assert probs and "out_step_offsets.bin is" in probs[0], probs


def test_waits_are_compared_when_given(tmp_path):
    d = tmp_path / "c"
    f = dict(make_call(d))
    f["plan_wait_path"] = f["plan_wait_path"] + 1.0
    assert reuse_problems(d, f, 5) != []


def test_the_sun_is_not_compared(tmp_path):
    """Deliberate: it is identical for every call at an epoch, so it
    discriminates nothing and is the largest file in the directory."""
    from src.replay import IDENTIFYING
    assert "sun_path" not in IDENTIFYING


def test_reuse_is_consulted_before_the_binary_runs():
    src = (ROOT / "src" / "replay.py").read_text()
    assert "if reuse and not reuse_problems(" in src
    assert src.index("if reuse and not reuse_problems(") < src.index("subprocess.run(")
    # and the old existence-only test is gone
    assert "if all(Path(v).exists() for v in found.values())" not in src
