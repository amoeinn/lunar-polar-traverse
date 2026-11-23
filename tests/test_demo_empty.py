"""A demo rule that selects nothing is reported as an empty set.

The registration of R1 to R4 (PLAN.md, 2026-10-05) says an empty set is
reported as an empty set and nothing is relaxed or substituted. On the
corrected core R2 is empty, 0 busts among 51 monitored plans, and the panels
stage crashed handing None to the executive panel. These tests protect the
renderer completing with an empty R2 or R3, the panel stating the emptiness
with its counts, nothing being drawn in its place, a panel from an earlier run
not surviving beside it, and the stages that need a traverse refusing when R3
is empty rather than inventing one.
"""
import json
import sys
from pathlib import Path

import matplotlib
import matplotlib.figure
import pytest

import examples.demo_traverse as D

ROOT = Path(__file__).resolve().parents[1]

R2_EMPTY = dict(monitored=51, busts=0)
R3_COUNTS = dict(monitored_nonbust=51, false_alarms=34, removed_by_lit=17)
SUBJECT = dict(tag="2028-02-15_p0", plan_index=32)


def selection(r2=None, r3=SUBJECT, c3=R3_COUNTS):
    return dict(r1=dict(counts={}, n=0, choice=None),
                r2=dict(counts=R2_EMPTY if r2 is None else dict(monitored=51, busts=1),
                        n=0 if r2 is None else 1, choice=r2),
                r3=dict(counts=c3, n=c3["removed_by_lit"], choice=r3))


@pytest.fixture
def drawn(monkeypatch):
    """Every saved figure's text, by output file name, from the figure itself."""
    seen = {}
    real = matplotlib.figure.Figure.savefig

    def save(fig, fname, *a, **k):
        texts = [t.get_text() for t in fig.texts]
        texts += [t.get_text() for ax in fig.axes for t in ax.texts + [ax.title]]
        if fig._suptitle is not None:
            texts.append(fig._suptitle.get_text())
        seen[Path(fname).name] = "\n".join(texts)
        return real(fig, fname, *a, **k)
    monkeypatch.setattr(matplotlib.figure.Figure, "savefig", save)
    return seen


@pytest.fixture
def stubs(monkeypatch):
    """The executive and epoch panels read run directories; record them instead."""
    calls = []

    def exec_panel(tag, plan_index, arms, title, out, dpi=120):
        calls.append(dict(tag=tag, plan_index=plan_index, title=title, out=Path(out).name))
        Path(out).write_bytes(b"exec")
    monkeypatch.setattr(D, "render_exec_panel", exec_panel)
    monkeypatch.setattr(D, "select_r4", lambda: [])
    monkeypatch.setattr(D, "render_epoch_panel",
                        lambda rows, out, dpi=120: Path(out).write_bytes(b"r4"))
    return calls


def test_an_empty_r2_is_rendered_as_an_empty_set(tmp_path, stubs, drawn):
    """Protects the crash fixed on 2026-11-22 and what replaced it: the panels
    complete, R2's panel states it is empty with its counts and why, no plan is
    drawn for R2, and an R2 panel left from an earlier run is overwritten."""
    stale = tmp_path / "panel_r2_bust.png"
    stale.write_bytes(b"a bust panel from a pre-fix run")
    D.render_panels(selection(), SUBJECT, tmp_path)
    text = drawn["panel_r2_bust.png"]
    assert "R2 is EMPTY: 0 busts among the 51 monitored core plans" in text, text
    assert "No core plan busts its 2.0 h budget after the corrections" in text, text
    assert "Nothing is substituted" in text, text
    assert stale.read_bytes()[:4] == b"\x89PNG", "the earlier panel survived"
    assert [c["out"] for c in stubs] == ["panel_r3_false_alarm.png"], stubs


def test_the_r3_title_carries_the_counts_not_a_literal(tmp_path, stubs, drawn):
    """Protects against the pre-fix title, "1 of the 11 ... at p = 0.0005", which
    was written as a literal and would have been drawn over corrected data on
    which the core's paired test is VOID."""
    D.render_panels(selection(), SUBJECT, tmp_path)
    (r3,) = stubs
    assert "1 of the 17 of 34 false alarms" in r3["title"], r3["title"]
    assert "p =" not in r3["title"] and "11" not in r3["title"], r3["title"]


def test_an_empty_r3_renders_the_panels_and_refuses_the_traverse(
        tmp_path, stubs, drawn, monkeypatch, capsys):
    """Protects the same handling for R3, the subject: the panels stage still
    completes with R3 stated empty, and every stage that needs a traverse
    refuses rather than rendering one nobody selected."""
    c3 = dict(monitored_nonbust=51, false_alarms=34, removed_by_lit=0)
    monkeypatch.setattr(D, "report_selection", lambda: selection(r3=None, c3=c3))
    monkeypatch.setattr(D, "OUT", tmp_path)
    monkeypatch.setattr(sys, "argv", ["x", "--stage", "panels"])
    assert D.main() == 0
    text = drawn["panel_r3_false_alarm.png"]
    assert "R3 is EMPTY: 0 of 34 false alarms cleared" in text, text
    assert "clears none of the bounded rule's false alarms" in text, text
    assert "R2 is EMPTY" in drawn["panel_r2_bust.png"]
    assert stubs == [], "an executive panel was drawn for an empty rule"
    assert json.loads((tmp_path / "sidecar.json").read_text())["subject"] is None
    for stage in ("fields", "frames", "video", "gazebo"):
        monkeypatch.setattr(sys, "argv", ["x", "--stage", f"panels,{stage}"])
        with pytest.raises(SystemExit, match="R3 is EMPTY"):
            D.main()


def test_select_r2_returns_none_rather_than_a_stand_in(monkeypatch):
    """Protects the selection itself: no bust means no choice, and the counts
    say so, whatever else the sample holds."""
    rec = lambda m, b: dict(monitored=m, busted=b, fired=True)
    arms = {"none": {("2028-02-15_p0", 0): rec(True, False),
                     ("2028-02-15_p0", 1): rec(False, False)}}
    monkeypatch.setattr(D, "_corr_records", lambda: (arms, sorted(arms["none"])))
    choice, counts, cands = D.select_r2()
    assert choice is None and cands == [] and counts == dict(monitored=1, busts=0)


def test_the_selection_report_states_an_empty_r2(monkeypatch, capsys):
    """Protects the select stage's text, which is the whole record when only
    selection is run."""
    monkeypatch.setattr(D, "select_r1", lambda: (None, {}, []))
    monkeypatch.setattr(D, "select_r2", lambda: (None, R2_EMPTY, []))
    monkeypatch.setattr(D, "select_r3", lambda: (SUBJECT, R3_COUNTS, [SUBJECT]))
    monkeypatch.setattr(D, "pair_contingency", lambda: {})
    D.report_selection()
    out = capsys.readouterr().out
    assert "R2 is EMPTY: 0 busts among the 51 monitored core plans" in out, out


CORR = ROOT / "data" / "work" / "executive_correction" / "results.json"


@pytest.mark.skipif(not CORR.exists(),
                    reason="the core correction arm has not been run "
                           "(data/work/executive_correction/results.json)")
def test_the_panels_complete_on_the_record_as_it_stands(tmp_path, drawn):
    """Protects the crash on the data that produced it: the selection read from
    the committed run, and the real executive and epoch panels drawn."""
    r2, c2, _ = D.select_r2()
    r3, c3, _ = D.select_r3()
    sel = dict(r2=dict(counts=c2, choice=r2), r3=dict(counts=c3, choice=r3))
    D.render_panels(sel, r3, tmp_path)
    for n in ("panel_r2_bust.png", "panel_r3_false_alarm.png", "panel_r4_epochs.png"):
        assert (tmp_path / n).read_bytes()[:4] == b"\x89PNG", n
    if r2 is None:
        assert "R2 is EMPTY" in drawn["panel_r2_bust.png"]
