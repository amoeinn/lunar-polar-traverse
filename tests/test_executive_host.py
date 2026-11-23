"""The executive's host: which plans it runs on and at what shadow budget.

The primary branch runs the executive on the tolerance axis at 0.5 h, where the
drivers had only ever run the core at 2 h. A bust is a plan whose shadow exceeds
the budget, so a host scored at the wrong budget does not fail loudly: at 2 h the
0.5 h point has no busts at all, every miss condition is met vacuously, and the
run reports success. These tests protect the budget's one route, from the files
the plans were made and replayed under, through the record, to the executive.
"""
import json
from pathlib import Path

import numpy as np
import pytest

import examples.run_executive as R
import examples.score_correction as C

ROOT = Path(__file__).resolve().parents[1]


def tree(tmp_path, shadows, made=0.5, judged=0.5, tag="2028-08-15_p2", stem="tolerance_0_"):
    """A host directory with one epoch whose plans have the given checker shadows."""
    pd = tmp_path / f"{stem}{tag}_td"
    rd = tmp_path / f"{stem}{tag}_replay_td_tol"
    pd.mkdir(parents=True)
    rd.mkdir(parents=True)
    n = len(shadows)
    off = np.arange(0, 2 * n + 1, 2, dtype=np.int64)
    files = dict(out_path_offsets=off,
                 out_path_cells=np.arange(2 * n, dtype=np.int32),
                 out_path_times=np.tile([0.0, 1.0], n))
    p = dict(shadow_tol_h=made)
    for k, a in files.items():
        (pd / f"{k}.bin").write_bytes(a.tobytes())
        p[k] = str(pd / f"{k}.bin")
    for k in ("indptr_path", "indices_path", "travel_path", "itimes_path",
              "ioffsets_path", "ilit0_path"):
        p[k] = str(pd / k)
    (pd / "planner_params.json").write_text(json.dumps(p))
    (rd / "replay_params.json").write_text(json.dumps(dict(shadow_tol_h=judged)))
    (rd / "out_shadow.bin").write_bytes(np.asarray(shadows, np.float64).tobytes())
    return dict(name="test", work=tmp_path, stem=stem, tags=(tag,), budget_h=0.5,
                out=tmp_path / "out")


def test_a_bust_is_shadow_over_the_hosts_budget_not_the_cores(tmp_path):
    """Protects the bust label from the core's 2 h. At 2 h these five plans hold
    one bust; at the host's 0.5 h they hold three, and only the second is the
    sample the primary branch registered."""
    recs = R.sample(tree(tmp_path, [0.0, 0.3, 0.6, 1.5, 2.5]))
    assert [r["monitored"] for r in recs] == [False, True, True, True, True]
    assert [r["busted"] for r in recs] == [False, False, True, True, True]
    assert all(r["budget_h"] == 0.5 for r in recs)


@pytest.mark.parametrize("made, judged", [(2.0, 0.5), (0.5, 2.0), (2.0, 2.0)])
def test_a_host_refuses_plans_made_or_replayed_at_another_budget(tmp_path, made, judged):
    """Protects against scoring a host at a budget its plans were not made or
    replayed under, which is how a 0.5 h point read at 2 h shows no busts."""
    with pytest.raises(ValueError, match="budget"):
        R.sample(tree(tmp_path, [0.0, 0.6], made=made, judged=judged))


def test_the_executive_and_the_correction_arm_run_at_the_records_budget(tmp_path, monkeypatch):
    """Protects the budget's last step: the drivers pass the record's budget to
    the executive, not a module constant. Both drivers, since each calls
    run_instance itself."""
    rec = R.sample(tree(tmp_path, [0.6]))[0]
    seen = []

    def fake(planner, replay, graph, plan, times, work, budget_h, *a, **k):
        seen.append(budget_h)
        return []
    monkeypatch.setattr(R, "run_instance", fake)
    monkeypatch.setattr(C, "run_instance", fake)
    monkeypatch.setitem(C.OUT, "test", tmp_path / "corr")
    R.one((rec, "bounded", True))
    C.one((rec, "none"))
    assert seen == [0.5, 0.5], seen
    assert Path(R.HOSTS["core"]["out"]) != Path(R.HOSTS["tol05"]["out"])
    assert C.OUT["core"] != C.OUT["tol05"]


TOL05_DATA = all((ROOT / "data" / "work" / "axes" / f"tolerance_0_{t}_replay_td_tol"
                  / "out_shadow.bin").exists() for t in ("2028-08-15_p2", "2028-11-15_p2"))


@pytest.mark.skipif(not TOL05_DATA, reason="the tolerance axis replay outputs are not "
                    "in data/work/axes; run examples/sweep_axes.py --axis tolerance")
def test_the_primary_branch_host_holds_the_two_registered_cases():
    """Protects the sample the registered bars name: 66 plans, 8 monitored, 58
    controls, and exactly the two busts found on the corrected checker. A change
    to either the host definition or the replay output shows up here first."""
    recs = R.sample(R.HOSTS["tol05"])
    mon = [r for r in recs if r["monitored"]]
    assert (len(recs), len(mon), len(recs) - len(mon)) == (66, 8, 58)
    assert sorted((r["tag"], r["plan_index"]) for r in mon if r["busted"]) == [
        ("2028-08-15_p2", 32), ("2028-11-15_p2", 24)]
    assert {r["budget_h"] for r in recs} == {0.5}
