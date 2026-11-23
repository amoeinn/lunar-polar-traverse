"""Verdicts on bars that are about catching busts, when there are none to catch.

The corrected core has no bust at the 2 h budget, and the executive scorer still
printed "P8: FAIL" for it: a verdict on four registered busts, delivered on a
sample with zero. The miss bar is met there by every rule, never-fire included,
and Matthews has no positive class, so the bars test nothing and the honest
verdict is VOID. These tests protect that, on the printed output itself, for
both scorers.
"""
import examples.run_executive as R
import examples.score_correction as C


def records(n_bust, n_quiet_fired, n_quiet, n_controls):
    recs = [dict(monitored=True, busted=True, fired=True) for _ in range(n_bust)]
    recs += [dict(monitored=True, busted=False, fired=i < n_quiet_fired) for i in range(n_quiet)]
    recs += [dict(monitored=False, busted=False, fired=False) for _ in range(n_controls)]
    return {rule: recs for rule, _ in R.RULES}


def test_p8_is_void_on_a_sample_with_no_busts(capsys):
    """Protects against a FAIL (or a PASS) printed for P8 when no bust exists.
    The fixture is the corrected core's own shape: 51 monitored, 34 fired on,
    175 controls, no busts."""
    R.side_by_side(records(0, 34, 51, 175), R.HOSTS["core"])
    out = capsys.readouterr().out
    assert "P8: VOID" in out, out
    assert "P8: FAIL" not in out and "P8: PASS" not in out
    assert "false-alarm cost only: 34 of 51 monitored plans fired, 0 of 175 controls" in out, out


def test_p8_still_passes_and_fails_when_busts_exist(capsys):
    """Protects the other side: VOID only when the positive class is empty. A
    rule that catches both busts with no false alarm passes; with every quiet
    plan fired on it fails the unnecessary-rate bar."""
    R.side_by_side(records(2, 0, 6, 58), R.HOSTS["core"])
    assert "P8: PASS" in capsys.readouterr().out
    R.side_by_side(records(2, 6, 6, 58), R.HOSTS["core"])
    assert "P8: FAIL" in capsys.readouterr().out


def test_the_per_rule_report_is_void_on_no_busts(capsys, monkeypatch):
    """Protects the per-rule miss line, which printed PASS for 0 misses of 0."""
    recs = records(0, 34, 51, 175)["bounded"]
    for r in recs:
        r.update(legs=[dict(outcome="continued", peak_gap=0.0)], tag="t",
                 plan_index=0, drift_max=0.0, ticks=0)
    monkeypatch.setattr(R, "diagnose", lambda results: None)   # reads run directories
    R.report(recs)
    out = capsys.readouterr().out
    assert "-> VOID" in out and "misses 0  (bar: 0) -> PASS" not in out, out


def test_the_paired_test_is_void_on_no_busts_and_a_comparison_off_the_core():
    """Protects the correction arm's verdict column: VOID with no bust, and on
    a host where the 2026-10-05 test is not the registered one, labelled as a
    comparison rather than printed as that host's verdict."""
    assert C.paired_verdict(0, 0, 0.01, 0, "core") == "VOID"
    assert C.paired_verdict(0, 0, 0.01, 4, "core") == "PASS"
    assert C.paired_verdict(0, 0, 0.25, 4, "core") == "FAIL"
    v = C.paired_verdict(0, 0, 0.25, 2, "tol05")
    assert v.startswith("would fail") and "not registered on tol05" in v
