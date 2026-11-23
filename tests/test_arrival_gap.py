"""The checker-against-planner arrival gap keeps its sign.

examples/replay_batch.py reported the median of abs(checker - planner) /
planner, 1.169%, and the record carried it as "+1.169%, checker slower,
confirmed in direction". No direction had been measured. A signed measurement
found the checker earlier on 156 of 187 plans, and that hid a checker defect
for seven weeks. These tests hold the sign in place at every site that reports
the gap, and hold the registered criterion two sided.
"""
import json

import numpy as np
import pytest

import examples.replay_batch as RB
import examples.score_arrival as SA
from src.arrival import ARRIVAL_CRITERION, arrival_gap, summarise, within_criterion

# Checker earlier on three plans and later on one: the case the abs() hid.
PLANNER = np.array([10.0, 20.0, 40.0, 8.0])
CHECKER = np.array([9.8, 19.7, 39.5, 8.1])


def test_the_fixture_can_tell_a_signed_gap_from_a_magnitude():
    """Discriminating property: the gaps carry both signs and the signed median
    has the opposite sign to the magnitude's."""
    g = CHECKER - PLANNER
    assert (g < 0).any() and (g > 0).any()
    assert np.median(g / PLANNER) < 0 < np.median(np.abs(g) / PLANNER)


def test_the_gap_is_signed_checker_minus_planner():
    """Protects the sign convention: positive means the checker arrives later."""
    g = arrival_gap(CHECKER, PLANNER)
    assert np.allclose(g, (CHECKER - PLANNER) / PLANNER)
    assert list(np.sign(g)) == [-1, -1, -1, 1]


def test_the_summary_counts_each_direction_and_keeps_the_median_signed():
    """Protects summarise() against reporting a magnitude as the median."""
    s = summarise(arrival_gap(CHECKER, PLANNER))
    assert (s["n"], s["later"], s["earlier"], s["equal"]) == (4, 1, 3, 0)
    assert s["median"] < 0
    assert s["median_magnitude"] == pytest.approx(
        np.median(np.abs(CHECKER - PLANNER) / PLANNER))
    assert summarise([])["n"] == 0


def test_the_registered_criterion_is_two_sided():
    """Protects the 1% criterion against being applied to one direction only.
    A gap 0.9% either way is met; 1.1% either way is not."""
    g = np.array([+0.009, -0.009, +0.011, -0.011])
    assert ARRIVAL_CRITERION == 0.01
    assert list(within_criterion(g)) == [True, True, False, False]
    assert summarise(g)["within"] == 2


def test_replay_batch_reports_the_signed_gap():
    """Protects the line that produced "+1.169%": replay_batch's arrival report
    must state the signed median and the count in each direction."""
    text = RB.arrival_report(CHECKER, PLANNER)
    s = summarise(arrival_gap(CHECKER, PLANNER))
    assert f"signed median {s['median']:+.4%}" in text
    assert "checker later on 1, earlier on 3" in text
    assert "median magnitude, no sign" in text


def test_the_scorer_pairs_the_checker_against_the_planner_that_made_the_plan():
    """Protects score_arrival against swapping claimed and checked, which flips
    every sign and leaves every magnitude, and every within-1% count, unchanged."""
    rows = [dict(claimed=float(p), checked=float(c)) for p, c in zip(PLANNER, CHECKER)]
    s = SA.score_kind(rows)
    assert (s["later"], s["earlier"]) == (1, 3)
    assert s["median"] < 0


@pytest.mark.skipif(not SA.OUT.exists(),
                    reason="data/work/arrival_scores.json is not on disk; run "
                           "examples/score_arrival.py")
def test_the_recorded_scores_match_the_replays_read_directly():
    """Protects the recorded time-dependent and snapshot scores against a
    pairing fault in the score_phase3 path they were collected through: here
    the planner's arrival and the checker's are read straight from the binaries,
    paired by the replay's own plan-to-instance index."""
    import examples.demo_traverse as D
    from examples.build_intervals import epochs
    rec = json.loads(SA.OUT.read_text())["planners"]
    for kind in ("td", "snap"):
        chk, pln = [], []
        for tag, _, _ in epochs():
            if not (D.SWEEP / f"{tag}_replay_{kind}_tol").exists():
                continue
            rep = D.replay_out(tag, kind)
            pl = D.planner_out(tag, kind)
            chk.extend(rep["arrival"].tolist())
            pln.extend(pl["arrival"][pl["index"]].tolist())
        s = summarise(arrival_gap(chk, pln))
        for k in ("n", "later", "earlier", "equal", "within"):
            assert s[k] == rec[kind][k], (kind, k, s[k], rec[kind][k])
        assert s["median"] == pytest.approx(rec[kind]["median"], abs=1e-12)
