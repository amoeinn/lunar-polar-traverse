"""The core sweep's pair draw must be keyed on the epoch, not on --tags order.

The fault this protects, found on 2026-11-17. draw() seeded its generator with
the epoch's position in the --tags list. The original sweep passed --tags all,
so that position equalled the epoch's registered index and nothing showed. The
sun-units regeneration ran the same twelve epochs in five batches, and each
batch renumbered its epochs from zero, so an epoch drew a different instance
set depending on which others shared its run. 2028-05-15_p1 and 2028-05-15_p2
finished with their planner stages scored on one draw of 120 pairs and their
mean-map stages on another, disjoint one, and the mean-map pass counts moved
from 0,0,0,0 to 2,2,2,2 and from 3,3,3,3 to 0,0,0,0 with no planner rerun
between them.

A test that only checked one tags list could not see this: the bug is a
relation between two runs, so every case here compares batchings.
"""
import itertools
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import examples.plan_pilot as P                                   # noqa: E402
import examples.sweep_core as S                                   # noqa: E402
from examples.build_intervals import epochs                       # noqa: E402

REG = [t for t, _, _ in epochs()]


def test_the_label_is_the_epochs_registered_index():
    """The registered draw used --tags all, so the index among the registered
    twelve is the key that reproduces it."""
    assert [S.draw_label(t) for t in REG] == list(range(12))


@pytest.mark.parametrize("batching", [
    [REG],                                            # one run, as registered
    [REG[:3], REG[3:]],                               # two runs
    [REG[0:1], REG[1:2], REG[2:]],                    # a single epoch alone
    [["2028-11-15_p2"], ["2028-02-15_p0"], ["2028-08-15_p2"],
     ["2028-02-15_p2", "2028-08-15_p1", "2028-11-15_p1", "2028-08-15_p0",
      "2028-11-15_p0", "2028-05-15_p0", "2028-05-15_p1", "2028-05-15_p2"]],
    [list(reversed(REG))],                            # reordered
])
def test_batching_cannot_change_any_epochs_label(batching):
    """The invariant. Whatever the runs are, each epoch keeps its own key."""
    seen = {}
    for run in batching:
        for pos, tag in enumerate(run):
            seen.setdefault(tag, []).append(S.draw_label(tag))
            # and the key must not be the position it had in this run
            assert S.draw_label(tag) == REG.index(tag)
    assert set(seen) == set(REG) or all(len(v) >= 1 for v in seen.values())
    for tag, labels in seen.items():
        assert len(set(labels)) == 1, f"{tag} drew under more than one key"


def test_splitting_the_tags_cannot_change_the_pairs(monkeypatch):
    """End to end on draw() itself: the same epoch in different batchings must
    produce the identical pair list, pool and geometry held fixed."""
    monkeypatch.setattr(S, "PAIRS_PER_BAND", 3)
    monkeypatch.setattr(P, "BANDS", [(0.0, 4000.0), (4000.0, 9000.0)])
    rng = np.random.default_rng(7)
    pool = rng.choice(40000, 900, replace=False)
    xs = (np.arange(40000) % 200) * 20.0
    ys = (np.arange(40000) // 200) * 20.0

    def pairs_for(run):
        # exactly what the driver's loop does, for each epoch of one run
        return {tag: S.draw(pool, xs, ys, label=S.draw_label(tag)) for tag in run}

    as_registered = pairs_for(REG)
    for batching in ([REG[:4], REG[4:8], REG[8:]],
                     [[t] for t in REG],
                     [list(reversed(REG))]):
        got = {}
        for run in batching:
            got.update(pairs_for(run))
        assert got == as_registered, "a batching changed some epoch's pairs"


def test_the_label_actually_reaches_the_pairs(monkeypatch):
    """Discriminating check. If draw ignored its label, every test above would
    pass for the wrong reason, so confirm two labels give different pairs."""
    monkeypatch.setattr(S, "PAIRS_PER_BAND", 3)
    monkeypatch.setattr(P, "BANDS", [(0.0, 4000.0)])
    rng = np.random.default_rng(7)
    pool = rng.choice(40000, 900, replace=False)
    xs = (np.arange(40000) % 200) * 20.0
    ys = (np.arange(40000) // 200) * 20.0
    a = S.draw(pool, xs, ys, label=0)
    b = S.draw(pool, xs, ys, label=1)
    assert a != b, "draw ignores its label, so keying it on anything is moot"
    assert S.draw(pool, xs, ys, label=0) == a, "draw is not deterministic"


def test_every_pair_of_registered_epochs_gets_a_distinct_key():
    assert len({S.draw_label(t) for t in REG}) == len(REG)


def test_an_unregistered_tag_is_refused_rather_than_silently_keyed():
    with pytest.raises(KeyError):
        S.draw_label("2029-01-01_p0")


def test_the_driver_does_not_key_the_draw_on_list_position():
    """Source-level, so a revert fails here. The loop must not enumerate tags
    for a label, and the draw call must take the epoch's own key."""
    src = (ROOT / "examples" / "sweep_core.py").read_text()
    assert "label=draw_label(tag)" in src, \
        "the draw is no longer keyed on the epoch"
    assert "label=ei" not in src, \
        "the draw was reverted to the --tags position"
    assert "for ei, tag in enumerate(tags)" not in src


def test_unknown_tags_are_rejected_before_any_work():
    """An unregistered tag must stop the run, not KeyError partway through
    after the planner inputs have been written."""
    src = (ROOT / "examples" / "sweep_core.py").read_text()
    assert src.index("unknown = [t for t in tags") < src.index("P.load("), \
        "the tags are validated only after the region is loaded"
