"""P3's mean-map baseline is scored at the exponent the registered criterion
selects over every swept exponent, from the files.

On 2026-11-22 the scorer applied the criterion to one candidate, the pre-fix
run's 2.0 carried as a constant, and printed "registered criterion selects 2.0"
with 227 passes when the four exponents give 216, 227, 230 and 227 and the
criterion selects 1.0. Clause 1 then read FAIL, 227 against 229, where the
registered selection gives 230. It also compared the rebuilt mean-map against a
pre-fix recorded pass count, 197. These tests protect every exponent reaching
the criterion, the selection being the maximum over them, the printed verdict
using that selection, and the reconstruction being checked against the sweep's
own replay directories instead of a recorded number.
"""
import sys

import numpy as np
import pytest

import examples.score_phase3 as S
import examples.sweep_core as CO

COUNTS = {0.0: 216, 0.5: 227, 1.0: 230, 2.0: 227}
SNAP_PASS = 229


def fake_rows(monkeypatch, counts=COUNTS, snap_pass=SNAP_PASS):
    """all_rows stand-in: per exponent, that many passing plans of 300."""
    def rows(recs, kind):
        if kind.startswith("mm"):
            n = counts[float(kind[2:])]
        elif kind == "snap":
            n = snap_pass
        else:
            n = 0
        return [dict(tag="t", inst=i, pass_tol=i < n, pass_zero=i < n,
                     claimed=10.0, checked=10.0, cells=5) for i in range(300)]
    monkeypatch.setattr(S, "all_rows", rows)
    rec = {"td": dict(n=1440)}
    rec.update({f"mm{e}": dict(judged=object(), recon="identical") for e in counts})
    return {"t": rec}


def test_the_criterion_selects_the_maximum_over_every_exponent(monkeypatch):
    """Protects the selection itself: 1.0 at 230, not the first or the last."""
    recs = fake_rows(monkeypatch)
    best, counts = S.best_exponent(recs, CO.EXPONENTS)
    assert counts == COUNTS and best == 1.0


def test_a_criterion_over_one_exponent_is_refused(monkeypatch):
    """Protects against the defect's shape: one candidate always wins."""
    recs = fake_rows(monkeypatch)
    with pytest.raises(ValueError):
        S.best_exponent(recs, (2.0,))


def test_p3_clause_1_is_scored_at_the_selected_exponent(monkeypatch, capsys):
    """Protects the printed score: every exponent's count, the selection, the
    clause-1 comparison at 230 against 229 with its margin, and no pre-fix
    recorded figure."""
    recs = fake_rows(monkeypatch)
    S.score_p3_p4(recs, CO.EXPONENTS)
    out = capsys.readouterr().out
    assert ("e=0.0: 216, e=0.5: 227, e=1.0: 230, e=2.0: 227; "
            "registered criterion selects 1.0") in out, out
    assert "mean-map 230 feasible" in out and "snapshot 229 feasible" in out, out
    assert "margin: mean-map minus snapshot +1 at the registered tolerance" in out, out
    assert "clause 1: PASS at the registered tolerance" in out, out
    assert "197" not in out and "recorded" not in out, out
    assert "the sweep replayed in 1 of 4 calls" not in out
    assert "the sweep replayed in 4 of 4 calls" in out, out


def test_the_core_hands_every_exponent_to_collect_and_to_the_score(monkeypatch):
    """Protects main: the core's collect and its P3 score both receive all four
    exponents the sweep ran, not a constant."""
    seen = {"collect": set(), "score": None}

    class Site:
        pass
    monkeypatch.setattr(S, "Site", Site)
    monkeypatch.setattr(S, "epochs", lambda: [("a", 0, 0), ("b", 0, 0)])

    def collect(site, work, stem, tag, rover, tol, ei, draw, exponents):
        seen["collect"].add(tuple(exponents))
        return {"td": dict(aborted=np.zeros(1, bool), arrival=np.ones(1)),
                "snap": dict(aborted=np.zeros(1, bool), arrival=np.ones(1))}
    monkeypatch.setattr(S, "collect", collect)
    monkeypatch.setattr(S, "score_p2", lambda *a, **k: None)
    monkeypatch.setattr(S, "score_p3_p4",
                        lambda recs, exps: seen.__setitem__("score", tuple(exps)))
    monkeypatch.setattr(sys, "argv", ["x", "--parts", "core"])
    S.main()
    assert seen["collect"] == {tuple(CO.EXPONENTS)}, seen
    assert seen["score"] == tuple(CO.EXPONENTS), seen


def write_call(d, plans, starts, waits):
    d.mkdir(parents=True)
    off = np.zeros(len(plans) + 1, np.int64)
    for i, p in enumerate(plans):
        off[i + 1] = off[i] + len(p)
    off.tofile(d / "plan_offsets_path.bin")
    np.concatenate(plans).astype(np.int32).tofile(d / "plan_cells_path.bin")
    np.asarray(starts, np.float64).tofile(d / "plan_start_path.bin")
    np.concatenate(waits).astype(np.float64).tofile(d / "plan_wait_path.bin")


def test_the_reconstruction_is_checked_against_the_sweeps_own_call(tmp_path):
    """Protects the check that replaced the recorded 197: rebuilt routes equal
    to the sweep's replayed plans are identical, a directory holding a different
    call is different (and collect refuses to overwrite it), and a missing
    directory is reported absent."""
    plans = [np.array([1, 2, 3]), np.array([4, 5])]
    starts, waits = [0.0, 0.0], [np.zeros(3), np.zeros(2)]
    write_call(tmp_path / "same", plans, starts, waits)
    assert S.sweep_call_status(tmp_path / "same", plans, starts, waits) == "identical"
    other = [np.array([1, 2, 3]), np.array([4, 6])]
    write_call(tmp_path / "other", other, starts, waits)
    assert S.sweep_call_status(tmp_path / "other", plans, starts, waits) == "different"
    assert S.sweep_call_status(tmp_path / "none", plans, starts, waits) == "absent"
    write_call(tmp_path / "p_tol", plans, starts, waits)
    write_call(tmp_path / "p_zero", other, starts, waits)
    with pytest.raises(SystemExit, match="refusing to overwrite"):
        S.rebuilt_call_status(str(tmp_path / "p"), plans, starts, waits)
    write_call(tmp_path / "q_tol", plans, starts, waits)
    write_call(tmp_path / "q_zero", plans, starts, waits)
    assert S.rebuilt_call_status(str(tmp_path / "q"), plans, starts, waits) == \
        dict(tol="identical", zero="identical")


def test_the_reconstruction_report_counts_only_identical_calls():
    """Protects the printed reconstruction line from counting a call the scorer
    had to replay itself as one the sweep's record confirmed."""
    rec = {"mm0.0": dict(judged=1, recon="identical"),
           "mm0.5": dict(judged=1, recon="absent"),
           "mm1.0": dict(judged=None, recon="absent"),
           "mm2.0": dict(judged=1, recon="identical")}
    line = S.recon_report({"t": rec}, CO.EXPONENTS)
    assert "the sweep replayed in 2 of 3 calls" in line, line
    assert "1 had no sweep replay on disk" in line, line
