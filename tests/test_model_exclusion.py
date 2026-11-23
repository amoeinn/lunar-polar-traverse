"""The tol05 model arm's training set: no host pair in it, and a registered floor.

PLAN.md, D132 step 13. The model the tol05 arm uses must be trained without any
plan whose start/goal pair the tol05 host also plans, at any axis point and any
epoch, since the axis points walk the same pairs with other rovers and budgets
and a model fit on the host's own corridors would be scored on them. It is fit
only if the region split's training fold keeps at least MIN_EPISODES episodes
in each direction; below that the arm is NOT RUN and nothing is trained or
saved. These tests protect the exclusion reaching the fit, the episode count
being of episodes and not samples, the floor's boundary, and the arm refusing a
model trained for the other host.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pytest

import examples.run_executive as R
import examples.score_correction as C
import examples.traversability_model as M
from src.disagree import episodes

ROOT = Path(__file__).resolve().parents[1]
HOST_TAG, OTHER_TAG = "2028-02-15_p0", "2028-05-15_p1"
HOST_PAIRS = [(1000, 2000), (1001, 2001)]


def block_cell(b):
    """A cell in region block b of the 5 x 5 grid region_blocks uses."""
    bi, bj = divmod(b, 5)
    return (bi * 100 + 50) * 500 + bj * 100 + 50


def labels(plans):
    """A pooled label frame. plans: (source, tag, start, goal, cell, codes), with
    code 0 agree, 1 table lit and checker dark, 2 table dark and checker lit.
    hor_dist carries the plan id, so a fitted row says which plan it came from."""
    tags = sorted({p[1] for p in plans})
    cols = {k: [] for k in ("source", "tag_id", "plan_id", "start", "goal", "cell",
                            "t", "world", "model", "nxt_lit", "margin", "sun_el",
                            "hor_el", "hor_dist")}
    for pid, (src, tag, s, g, cell, codes) in enumerate(plans):
        c = np.asarray(codes)
        n = c.size
        cols["source"].append(np.full(n, src))
        cols["tag_id"].append(np.full(n, tags.index(tag)))
        cols["plan_id"].append(np.full(n, pid))
        cols["start"].append(np.full(n, s))
        cols["goal"].append(np.full(n, g))
        cols["cell"].append(np.full(n, cell))
        cols["t"].append(np.arange(n) / 60.0)
        cols["world"].append(c != 1)
        cols["model"].append(c != 2)
        cols["nxt_lit"].append(np.ones(n, bool))
        for k in ("margin", "sun_el", "hor_el"):
            cols[k].append(np.zeros(n))
        cols["hor_dist"].append(np.full(n, float(pid)))
    d = {k: np.concatenate(v) for k, v in cols.items()}
    d["tag_names"] = np.array(tags)
    return d


class Recorder:
    """Stands in for the classifier and keeps every row it was fit on."""
    fitted = []

    def fit(self, X, y):
        Recorder.fitted.append(np.asarray(X).copy())
        return self

    def predict(self, X):
        return np.zeros(len(X), np.int64)


def fake_host(monkeypatch):
    recs = [dict(tag=HOST_TAG, plan=[s, 5, g]) for s, g in HOST_PAIRS]
    monkeypatch.setattr(R, "sample", lambda host: recs)


def run_main(monkeypatch, tmp_path, d, minimum):
    fake_host(monkeypatch)
    monkeypatch.setattr(M, "pooled", lambda paths: d)
    monkeypatch.setattr(M, "MIN_EPISODES", minimum)
    monkeypatch.setattr(M, "make_classifier", lambda args: Recorder())
    Recorder.fitted = []
    save = tmp_path / "m.joblib"
    monkeypatch.setattr(sys, "argv", ["x", "--exclude-host", "tol05", "--save", str(save)])
    rc = M.main()
    return rc, save


CODES = [0, 1, 1, 0, 2, 2, 0]          # one episode each way


def fixture_with_host_pairs():
    plans, excluded = [], set()
    for b in range(10):
        cell = block_cell(b)
        plans.append((0, HOST_TAG, *HOST_PAIRS[0], cell, CODES))     # host pair, host epoch
        excluded.add(len(plans) - 1)
        plans.append((1, OTHER_TAG, *HOST_PAIRS[1], cell, CODES))    # host pair, other epoch
        excluded.add(len(plans) - 1)
        plans.append((0, OTHER_TAG, 3000 + b, 4000 + b, cell, CODES))
        plans.append((1, HOST_TAG, 2000, 1000, cell, CODES))         # reversed: not a host pair
    return labels(plans), excluded


def test_no_host_pair_reaches_the_fitted_training_set(monkeypatch, tmp_path):
    """Protects the exclusion reaching the fit, at the host's epoch and at any
    other, and from both label sources. The fixture is asserted to contain host
    pairs first, or an empty exclusion would pass."""
    d, excluded = fixture_with_host_pairs()
    assert M.on_pairs(d, set(HOST_PAIRS)).any()
    rc, save = run_main(monkeypatch, tmp_path, d, minimum=1)
    assert rc == 0 and save.exists()
    reached = {int(v) for X in Recorder.fitted for v in X[:, -1]}
    assert reached, "nothing was fit, so the check would be vacuous"
    assert not reached & excluded, sorted(reached & excluded)
    import joblib
    b = joblib.load(save)
    assert b["exclude_host"] == "tol05" and b["episodes"] is not None


def test_episode_counts_are_episodes_not_samples():
    """Protects counting the registered unit. Against src/disagree.py episodes()
    plan by plan, with runs that touch across a plan boundary, which must count
    as two episodes."""
    rng = np.random.default_rng(3)
    plans = [(0, HOST_TAG, i, i, block_cell(i % 10),
              rng.choice(3, size=40, p=[0.6, 0.2, 0.2])) for i in range(12)]
    plans.append((0, HOST_TAG, 90, 90, block_cell(0), [0, 1, 1]))
    plans.append((0, HOST_TAG, 91, 91, block_cell(0), [1, 1, 0]))
    d = labels(plans)
    want = {1: 0, 2: 0}
    for pid in np.unique(d["plan_id"]):
        s = d["plan_id"] == pid
        want[1] += episodes(d["world"][s], d["model"][s], world_dark_only=True).size
        want[2] += episodes(d["world"][s], d["model"][s], world_dark_only=False).size
    got = M.episode_counts(d, np.ones(d["world"].size, bool))
    assert got == want, (got, want)
    n_samples = int((d["model"] & ~d["world"]).sum())
    assert got[1] < n_samples, "the fixture cannot tell episodes from samples"


def test_an_episode_counts_in_the_fold_of_its_first_sample():
    """Protects each episode being counted once, where it starts."""
    d = labels([(0, HOST_TAG, 1, 1, block_cell(0), [1, 1, 1, 0, 2, 2])])
    fold = np.array([False, True, True, True, True, False])
    assert M.episode_counts(d, fold) == {1: 0, 2: 1}


def test_the_floor_is_sixty_in_each_direction():
    """Protects the registered minimum and its boundary: at least 60 of each
    code, both required."""
    assert M.MIN_EPISODES == 60
    assert M.gate({1: 60, 2: 60})[0]
    assert not M.gate({1: 59, 2: 60})[0]
    assert not M.gate({1: 60, 2: 59})[0]


def test_below_the_floor_nothing_is_trained_or_saved(monkeypatch, tmp_path, capsys):
    """Protects "reported as not run, not trained on what remains". The fixture
    keeps 1 episode of each code per surviving plan, well under the floor."""
    d, _ = fixture_with_host_pairs()
    rc, save = run_main(monkeypatch, tmp_path, d, minimum=60)
    out = capsys.readouterr().out
    assert rc == 0 and "tol05 model arm: NOT RUN" in out and "minimum is 60" in out, out
    assert Recorder.fitted == [] and not save.exists()


def test_a_not_run_refuses_a_model_left_from_an_earlier_run(monkeypatch, tmp_path):
    """Protects the arm from reading a stale model as if this run had passed."""
    d, _ = fixture_with_host_pairs()
    (tmp_path / "m.joblib").write_bytes(b"stale")
    with pytest.raises(SystemExit):
        run_main(monkeypatch, tmp_path, d, minimum=60)


def test_each_arm_reads_only_its_own_hosts_model(monkeypatch, tmp_path):
    """Protects the tol05 arm from running on the core's model, which was
    trained on the host's own pairs, and the core from the excluded one."""
    import joblib
    assert M.MODEL_PATH["core"] != M.MODEL_PATH["tol05"], M.MODEL_PATH
    paths = {}
    for name, ex in (("core", None), ("tol05", "tol05")):
        p = tmp_path / f"{name}.joblib"
        joblib.dump(dict(model=None, features=M.FEATURES, exclude_host=ex), p)
        paths[name] = p
    monkeypatch.setattr(M, "MODEL_PATH", {"core": str(paths["tol05"]),
                                          "tol05": str(paths["core"])})
    for host in ("core", "tol05"):
        with pytest.raises(SystemExit):
            C.load_model(host)
    monkeypatch.setattr(M, "MODEL_PATH", {k: str(v) for k, v in paths.items()})
    assert C.load_model("tol05")["exclude_host"] == "tol05"
    assert C.load_model("core")["exclude_host"] is None
    monkeypatch.setattr(M, "MODEL_PATH", {"core": str(paths["core"]),
                                          "tol05": str(tmp_path / "absent.joblib")})
    with pytest.raises(SystemExit, match="not run"):
        C.load_model("tol05")


def test_the_correction_asks_for_the_records_host(monkeypatch):
    """Protects the worker passing the record's host through to the model load."""
    asked = []

    def refuse(host):
        asked.append(host)
        raise RuntimeError("stop")
    monkeypatch.setattr(C, "load_model", refuse)
    monkeypatch.setattr(C, "_CTX", {})
    with pytest.raises(RuntimeError):
        C.model_correction({}, [1, 2], [0.0, 1.0], HOST_TAG, "tol05")
    assert asked == ["tol05"]


AXES = ROOT / "data" / "work" / "axes"


@pytest.mark.skipif(not (AXES / f"tolerance_0_{HOST_TAG}_td").exists(),
                    reason="the tolerance axis planner output is not in data/work/axes")
def test_a_host_pair_is_the_planners_start_goal_pair():
    """Protects the key: a plan's first and last cells are the pair the planner
    was asked for, on every tol05 plan, so the exclusion removes the pairs the
    host plans and not some other endpoint."""
    recs = R.sample(R.HOSTS["tol05"])
    assert len(recs) == 66
    for r in recs:
        p = json.loads(Path(r["planner_params"]).read_text())
        pr = np.fromfile(p["pairs_path"], np.int32).reshape(-1, 2)[r["instance"]]
        assert (int(pr[0]), int(pr[1])) == (r["plan"][0], r["plan"][-1]), r["tag"]
    assert M.host_pairs("tol05") == {(r["plan"][0], r["plan"][-1]) for r in recs}


@pytest.mark.skipif(not all(p.exists() for p in M.LABELS),
                    reason="the label files are not generated yet "
                           "(examples/traversability_labels.py, both sources)")
def test_no_host_pair_survives_the_exclusion_on_the_real_labels():
    """Protects the same property on the label files themselves, once they exist."""
    from examples.traversability_baselines import pooled
    d = pooled(list(M.LABELS))
    pairs = M.host_pairs("tol05")
    drop = M.on_pairs(d, pairs)
    assert drop.any(), "no label plan sits on a host pair, so the check is vacuous"
    kept = M.without(d, drop)
    assert not {(int(a), int(b)) for a, b in zip(kept["start"], kept["goal"])} & pairs


def test_the_worker_passes_its_records_host(monkeypatch):
    """Protects the host carried from the tol05 record through the worker to
    the model load; the correction function alone cannot see the worker drop it."""
    asked = []

    def refuse(host):
        asked.append(host)
        raise RuntimeError("stop")

    def drive(*args):
        args[-1](np.array([1, 2], np.int32), np.array([0.0, 1.0]))
    monkeypatch.setattr(C, "load_model", refuse)
    monkeypatch.setattr(C, "_CTX", {})
    monkeypatch.setattr(C, "run_instance", drive)
    rec = dict(host="tol05", tag=HOST_TAG, plan_index=0, plan=[1, 2], times=[0.0, 1.0],
               graph={}, planner_params="", replay_params="", budget_h=0.5)
    with pytest.raises(RuntimeError):
        C.one((rec, "model"))
    assert asked == ["tol05"]
