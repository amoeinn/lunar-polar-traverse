"""Every number a demo caption draws comes from its source.

On 2026-11-22 the demo's captions, text panel, R4 footnote and Gazebo captions,
and verify_gazebo's C3 base rate, were found carrying pre-fix figures as string
literals: 153188 of 168873 lit cells, P3's +0.51% over 182 cases and its FAIL,
178 of 181 paired pairs, 165 of 187 instances, C3's 2.6039%, a 62.9084% constant
predictor. A rerun would have drawn them over corrected data. They are now
computed from the run or read from a recorded result (phase3_scores.json, the
C3 checkpoint, verify_gazebo's measured constants).

The test that protects this: build each caption from sources holding sentinel
values no literal could match, then require every number in the text to equal
one of those sources to the precision it is printed at. A literal reintroduced
anywhere in a sourced line shows up as a number with no source.
"""
import json
import re
from pathlib import Path

import pytest

import examples.demo_traverse as D
import examples.verify_gazebo as V
from src.gzmap import HEIGHTMAP_N, PX, REGION_N
from src.intervals import CADENCE_H

ROOT = Path(__file__).resolve().parents[1]
NUM = re.compile(r"(?<![\w.])[-+]?\d+(?:\.\d+)?%?(?![\w])")

P3 = dict(margin_required=0.13, share_required=0.29, exponent=0.7,
          clause2=dict(verdict="PASS", at_margin=37, n=711, median=0.0347))
PC = dict(both=dict(aa=311, ar=2, ra=1, rr=7), only_td=13, only_snap=171)
PC0 = dict(both=dict(aa=311, ar=0, ra=0, rr=7), only_td=13, only_snap=171)


def sourced(text, values):
    """Numbers in text that match no source value at their printed precision."""
    bad = []
    # "clause 2" names a clause; it is a label and not a measured number.
    for tok in NUM.findall(re.sub(r"\bclause \d\b", "clause", text)):
        pct = tok.endswith("%")
        body = tok[:-1] if pct else tok
        x = float(body)
        d = len(body.split(".")[1]) if "." in body else 0
        tol = 0.5 * 10 ** -d + 1e-12
        if not any(abs(x - (100 * v if pct else v)) <= tol for v in values):
            bad.append(tok)
    return bad


def p3_values(p3):
    c = p3["clause2"]
    return [p3["margin_required"], p3["share_required"], c["at_margin"], c["n"],
            c["median"]]


def test_the_map_caption_numbers_are_the_run_and_the_record(monkeypatch):
    """Protects the map frames' caption: the clip count, P3 read from its record
    with its verdict, the cadence, the floor and the subject's shadow. The
    constants are replaced by sentinels too, so a literal that happens to equal
    today's value is still caught."""
    monkeypatch.setattr(D, "LIT_FLOOR", 0.37)
    monkeypatch.setattr(D, "CADENCE_H", 0.6)
    lines = D.caption_lines(4321, 8765, 47, P3, 0.4321, same_snap=True)
    text = " ".join(lines)
    vals = [4321, 8765, 47, 0.37, 0.6 * 60, 0.4321] + p3_values(P3)
    assert sourced(text, vals) == [], sourced(text, vals)
    assert "P3 clause 2 is scored PASS" in text, text
    assert "+3.47%" in text and "37 of the 711" in text, text
    assert "0.43 h" in text and "4321 of 8765" in text, text


def test_the_caption_does_not_claim_what_the_run_does_not_show():
    """Protects the conditional wording: the same-route sentence only when the
    routes match, and 'cannot resolve' only for shadow shorter than a frame."""
    long_ = " ".join(D.caption_lines(10, 100, 5, P3, 0.9975, same_snap=False))
    assert "same route" not in long_ and "cannot resolve" not in long_, long_
    short = " ".join(D.caption_lines(10, 100, 5, P3, 0.10, same_snap=True))
    assert "same route" in short and "cannot resolve the 0.10 h" in short, short


def test_the_text_panel_numbers_are_the_contingency_and_the_record(monkeypatch):
    """Protects the text panel's pairing, mean-map and P3 lines. The mean-map
    row read "mean-map e=2" after P3's corrected criterion selected 1.0, and
    this test did not see it: the budget, 2.0, was among the sources, so the
    literal 2 matched it. The budget and the exponent are sentinels now, and
    the replay the row reads is recorded."""
    kinds = []

    def replay(tag, kind):
        kinds.append(kind)
        return dict(verdict=[True], arrival=[1.5], shadow=[0.25])
    monkeypatch.setattr(D, "replay_out", replay)
    monkeypatch.setattr(D, "BUDGET_H", 2.35)
    rows = dict(td=0, snap=0, mm=0)
    lines = D.panel_lines("t", 3, rows, {}, ("snap",), PC, P3)
    text = " ".join(lines[2:])           # past the epoch header
    vals = [311, 2 + 1, 7, 311 + 2 + 1 + 7, 13, 171, 1.5, 0.25, 2.35, 0.7] \
        + p3_values(P3)
    assert sourced(text, vals) == [], sourced(text, vals)
    assert "mean-map e=0.7" in text and "mm0.7" in kinds, (text, kinds)
    assert not any(k.startswith("mm") and k != "mm0.7" for k in kinds), kinds
    assert "on the 321 instances both planned: 311 both accepted," in text, text
    assert "3 accepted by one planner and rejected by the other." in text, text
    assert "not paired: 13 with only a time-dependent plan, 171 with only a " \
        "snapshot plan." in text, text
    assert "one only" not in text and "never disagree" not in text, text
    assert "P3 clause 2 is scored PASS:" in text, text


def test_no_disagreement_is_stated_over_shared_pairs_only(monkeypatch):
    """Protects the wording that replaced "0 one only", which read as "no
    instance has one planner's plan only" while 5 had only a time-dependent
    plan and 122 only a snapshot plan."""
    monkeypatch.setattr(D, "replay_out", lambda tag, kind: dict(
        verdict=[True], arrival=[1.5], shadow=[0.25]))
    text = " ".join(D.panel_lines("t", 3, dict(td=0, snap=0, mm=0), {}, (), PC0, P3))
    assert "7 both rejected; no pair accepted by one planner and rejected by " \
        "the other." in text, text
    assert "not paired: 13 with only a time-dependent plan" in text, text
    assert "0 one only" not in text and "never disagree" not in text, text


def test_the_mean_map_row_refuses_rows_from_another_exponent(monkeypatch):
    """Protects the rows and the record agreeing: a route read at one exponent
    and labelled with another is refused."""
    monkeypatch.setattr(D, "replay_out", lambda tag, kind: dict(
        verdict=[True], arrival=[1.5], shadow=[0.25]))
    with pytest.raises(ValueError, match="mm2.0"):
        D.panel_lines("t", 3, dict(td=0, snap=0, mm=0, mm_kind="mm2.0"), {}, (), PC, P3)


def test_the_r4_footnote_counts_its_own_rows():
    """Protects the R4 panel's footnote from a fixed instance count."""
    rows = [dict(td_plans=v) for v in (91, 4, 57, 0, 23, 11)]
    note = D.epoch_note(rows)
    assert sourced(note, [91 + 57 + 23, sum((91, 4, 57, 0, 23, 11)), 6]) == [], note
    assert "171 of the 186" in note and "these 6 epochs" in note, note


def test_the_gazebo_caption_reads_c3_and_the_measured_constants(monkeypatch):
    """Protects the Gazebo captions: C3 from its checkpoint, the surface term
    from SLOPE_BANDS, the offset from TEXTURE_OFFSET_NADIR_M, the margin from
    the grid. The two lines recorded only in PLAN.md are excluded and named.
    Every constant is a sentinel, for the same reason as above."""
    monkeypatch.setattr(V, "SLOPE_BANDS", ((0.0, 1.7, 0.0431), (1.7, 37.5, 0.5),
                                           (37.5, 90.0, 4.2718)))
    monkeypatch.setattr(V, "TEXTURE_OFFSET_NADIR_M", dict(a=3.21, b=6.54))
    monkeypatch.setattr(D, "HEIGHTMAP_N", 611)
    monkeypatch.setattr(D, "REGION_N", 590)
    monkeypatch.setattr(D, "PX", 17.0)
    lines = D.gz_caption_lines(dict(positive_disagreement=0.031415))
    sourced_lines = [l for l in lines if l not in (D.GZ_ROVER_LINE, D.GZ_DISPLAY_LINE)]
    assert len(sourced_lines) == 4
    vals = [0.031415, 1.7, 0.0431, 37.5, 4.2718, 3.21, 6.54, 17.0,
            611, 590, 21, 21 * 17.0]
    text = " ".join(sourced_lines)
    assert sourced(text, vals) == [], sourced(text, vals)
    assert "3.1415% of texture pixels" in text, text
    assert "+0.0431 m below 1.7 deg" in text and "3.2 to 6.5 m" in text, text


def test_verify_gazebo_states_its_base_rate_from_the_mask():
    """Protects C3's printed constant predictor, which carried 62.9084%."""
    line, region, patch = V.constant_predictor_line(0.3141, 1000, 4000)
    assert region == pytest.approx(0.6859) and patch == pytest.approx(0.75)
    assert sourced(line, [region, patch, 4000]) == [], line
    assert "68.5900%" in line and "75.0000%" in line, line


def test_the_checker_finds_a_planted_literal():
    """The discriminating property, asserted before the others are believed:
    a number with no source is reported."""
    assert sourced("median +0.51% over 182 cases", [0.0347, 711]) == ["+0.51%", "182"]
    assert sourced("median +3.47% over 711 cases", [0.0347, 711]) == []


@pytest.mark.skipif(not (D.SCORES.exists() and D.C3_RECORD.exists()),
                    reason="phase3_scores.json or the C3 checkpoint is not on disk")
def test_the_recorded_sources_are_what_the_captions_read():
    """Protects the readers against the files the scorers wrote."""
    p3 = D.p3_record()
    assert p3 == json.loads(D.SCORES.read_text())["P3"]
    assert p3["clause2"]["verdict"] in ("PASS", "FAIL")
    c3 = D.c3_record()
    text = " ".join(D.gz_caption_lines(c3))
    assert f"{c3['positive_disagreement']:.4%}" in text
    text = D.p3_sentence(p3)
    assert sourced(text, p3_values(p3)) == [], text
