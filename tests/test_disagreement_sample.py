"""The sample-source disagreement check reads each executed plan once.

The executive's results.json carries one record per plan per contradiction
rule. Read naively it yields every monitored plan twice, which doubles every
episode count beside H and, in the labels that once read the same file, put
the same plan on both sides of a split.
"""
import json

import numpy as np

import examples.disagreement as D


def test_each_monitored_plan_is_yielded_once(tmp_path, monkeypatch):
    """Protects against the duplicate: two rules over two monitored plans and a
    control must give two plans, not four."""
    recs = []
    for rule in ("unbounded", "bounded"):
        for i, monitored in enumerate((True, True, False)):
            w = tmp_path / rule / f"plan{i}"
            (w / "leg0_world").mkdir(parents=True)
            (w / "leg0_world" / "out_traj_t.bin").write_bytes(np.array([0.0, 1.0]).tobytes())
            (w / "leg0_world" / "out_traj_lit.bin").write_bytes(np.array([1, 0], np.uint8).tobytes())
            (w / "leg0_world" / "out_traj_index.bin").write_bytes(np.array([0, 1], np.int32).tobytes())
            recs.append(dict(rule=rule, tag="2028-08-15_p2", plan_index=i, graph={},
                             plan=[0, 1], monitored=monitored, work=str(w)))
    f = tmp_path / "results.json"
    f.write_text(json.dumps(recs))
    monkeypatch.setattr(D, "RESULTS", f)
    got = [(tag, plan) for tag, _, plan, *_ in D.from_sample()]
    assert len(got) == 2, got
