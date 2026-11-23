"""The README's stills are chosen from the run, and the sidecar's size and pace
figures are the ones the frames draw.

docs/make_figures.py carried frame0028 and gz0029 as literals with captions
saying the executive fires there and the rover is on a dark cell. After the
subject changed, the bounded rule fired at 0.653 h, and frame 28 showed nothing
the caption claimed. The README's 1260 m window and 560 m path were likewise
the previous subject's. These tests hold the selection rules and the sidecar's
geometry against their definitions, each with geometry written here.
"""
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

import examples.demo_traverse as D

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("make_figures",
                                               ROOT / "docs" / "make_figures.py")
MF = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(MF)


def test_the_dashboard_is_the_first_frame_after_the_firing_not_the_nearest():
    """Protects the dashboard still against showing a frame before the firing.
    Discriminating property first: the nearest frame precedes the firing."""
    tl = dict(frame_times_h=[0.0, 0.25, 0.5, 0.75, 1.0],
              executive_firings_h=[1.1, 0.55])
    t = np.array(tl["frame_times_h"])
    assert t[np.argmin(np.abs(t - 0.55))] < 0.55
    k, tk, f = MF.dashboard_frame(tl)
    assert (k, tk, f) == (3, 0.75, 0.55)
    # a frame exactly at the firing has not yet shown it
    assert MF.dashboard_frame(dict(frame_times_h=[0.0, 0.5, 0.75],
                                   executive_firings_h=[0.5]))[0] == 2


def test_the_gazebo_still_is_inside_the_longest_dark_run_with_the_rover_dark():
    """Protects the Gazebo still's caption, "inside the dark run, rover on a cell
    the model calls dark". Discriminating property first: without the rover
    filter, and without picking the longest run, the choice would differ."""
    t = list(np.arange(0.0, 5.01, 0.25))
    lit = [True] * len(t)
    for k in (13, 14, 16):         # 3.25, 3.5, 4.0 dark; 3.75 lit
        lit[k] = False
    lit[3] = False                 # 0.75, inside the shorter run
    tl = dict(frame_times_h=t, rover_cell_lit=lit,
              world_dark_runs_h=[[0.6, 1.1, 0.5], [3.1, 4.1, 1.0]])
    tt = np.array(t)
    inside = np.flatnonzero((tt > 3.1) & (tt < 4.1))
    assert inside[inside.size // 2] == 15 and lit[15]
    k, tk, run = MF.gazebo_frame(tl)
    assert (k, tk) == (14, 3.5) and run == (3.1, 4.1, 1.0)
    with pytest.raises(SystemExit):
        MF.gazebo_frame(dict(tl, rover_cell_lit=[True] * len(t)))


def test_the_hero_uses_every_frame_unless_that_is_over_the_limit():
    """Protects the hero's stride rule: every frame under 3 MB, else every second."""
    assert MF.HERO_MAX_BYTES == 3 * 1024 * 1024
    big = {1: 4 << 20, 2: 2 << 20}
    small = {1: 1 << 20, 2: 1 << 19}
    assert MF.hero_stride(big.__getitem__) == (2, 2 << 20)
    assert MF.hero_stride(small.__getitem__) == (1, 1 << 20)


def test_a_dark_run_starts_when_the_darkness_began_not_at_its_first_sample():
    """Protects dark_runs: the accumulator at a run's first sample says how long
    the rover had already been dark."""
    t = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]
    s = [0.0, 0.5, 1.5, 0.0, 0.0, 0.3]
    assert D.dark_runs(t, s) == [(0.5, 2.0, 1.5), (4.7, 5.0, 0.3)]
    assert D.dark_runs(t, [0.0] * 6) == []


def test_the_rover_cell_lit_state_is_read_at_the_step_the_rover_is_on():
    """Protects the per-frame rover state against reading the next step's cell."""
    plan = np.array([5, 6])
    tsteps = np.array([0.0, 1.0])
    litm = np.ones((2, 3, 3), bool)
    litm[0].reshape(-1)[6] = False   # the next cell, dark at frame 0
    litm[1].reshape(-1)[6] = False   # the rover's cell at frame 1
    assert D.rover_cell_lit(plan, tsteps, [0.5, 1.5], litm) == [True, False]


def test_the_zoom_window_is_square_and_holds_every_route_with_its_margin():
    """Protects the window the sidecar reports and the zoom panel draws."""
    routes = {"td": (np.array([1.00, 1.06]), np.array([2.0, 2.4])),
              "mm": (np.array([1.02]), np.array([2.1]))}
    zx, zy = D.zoom_window(routes)
    assert zx[1] - zx[0] == pytest.approx(zy[1] - zy[0])
    margin = max(0.35, 0.25 * 0.4)
    for x, y in routes.values():
        assert (x - zx[0] >= margin - 1e-12).all() and (zx[1] - x >= margin - 1e-12).all()
        assert (y - zy[0] >= margin - 1e-12).all() and (zy[1] - y >= margin - 1e-12).all()
    assert zx[1] - zx[0] == pytest.approx(0.4 + 2 * margin)


SIDECAR = D.OUT / "sidecar.json"


@pytest.mark.skipif(not SIDECAR.exists(), reason="data/work/demo/sidecar.json is "
                    "not on disk; run examples/demo_traverse.py")
def test_the_sidecar_geometry_is_the_subjects_own():
    """Protects the size and pace figures a caption will quote: start to goal and
    extent from cell centres computed here, the total wait from the replay's
    own wait file, and the window from the routes the zoom panel draws."""
    side = json.loads(SIDECAR.read_text())
    sub, rows = side["subject"], side["rows"]
    rep = D.replay_out(sub["tag"], "td")
    lo, hi = int(rep["offsets"][rows["td"]]), int(rep["offsets"][rows["td"] + 1])
    cells = rep["cells"][lo:hi].astype(np.int64)
    i, j = cells // D.REGION_N, cells % D.REGION_N
    x = D.REGION_OX + (j + 0.5) * D.PX
    y = D.REGION_OY - (i + 0.5) * D.PX
    assert sub["start_to_goal_m"] == pytest.approx(np.hypot(x[-1] - x[0], y[-1] - y[0]))
    assert sub["route_extent_m"] == pytest.approx([np.ptp(x), np.ptp(y)])
    assert sub["total_wait_h"] == pytest.approx(float(rep["waits"][lo:hi].sum()), abs=1e-12)
    _, routes, _ = D.subject_routes(sub["tag"], rows, rows["mm_kind"])
    zx, _ = D.zoom_window(routes)
    assert sub["zoom_window_m"] == pytest.approx((zx[1] - zx[0]) * 1000.0)
    assert len(side["timeline"]["frame_times_h"]) == side["configuration"]["frames"]


def test_the_total_wait_sums_every_hold_on_the_plan(monkeypatch):
    """Protects total_wait_h and holds_over_1s. The demo's subject holds for
    no time at all, so the real-data check cannot see a fault in the sum; this
    plan holds four times, once at the start cell and once for under a
    second."""
    waits = np.array([0.1, 0.5, 0.25, 0.0002, 0.0])
    rep = dict(offsets=np.array([0, 5]), cells=np.array([0, 1, 2, 502, 1002]),
               waits=waits, starts=np.array([0.0]))
    monkeypatch.setattr(D, "replay_out", lambda tag, kind, label="tol": rep)
    g = D.subject_geometry("t", dict(td=0, snap=None, mm=None), "mm1.0")
    assert g["total_wait_h"] == pytest.approx(0.8502)
    assert g["holds_over_1s"] == 3
    assert g["start_to_goal_m"] == pytest.approx(np.hypot(40.0, 40.0))
    assert g["route_extent_m"] == pytest.approx([40.0, 40.0])


@pytest.mark.skipif(not SIDECAR.exists(), reason="data/work/demo/sidecar.json is "
                    "not on disk; run examples/demo_traverse.py")
def test_the_sidecar_timeline_is_the_runs_own():
    """Protects the facts the stills are chosen by: every firing of the bounded
    rule over every leg, read here straight from the log files; the longest
    dark run's shadow equal to the checker's stored maximum for the plan."""
    side = json.loads(SIDECAR.read_text())
    sub, tl = side["subject"], side["timeline"]
    base = D.CORR / "none" / sub["tag"] / f"plan{side['rows']['exec_plan_index']:03d}"
    fires, leg = [], 0
    while (base / f"leg{leg}_exec").exists():
        d = base / f"leg{leg}_exec"
        t = np.fromfile(d / "out_log_t.bin", np.float64)
        f = np.fromfile(d / "out_log_fired.bin", np.uint8).astype(bool)
        fires.extend(t[f].tolist())
        leg += 1
    assert leg >= 2, "the discriminating case is a replanned run with two legs"
    assert tl["executive_firings_h"] == pytest.approx(fires)
    longest = max(tl["world_dark_runs_h"], key=lambda r: r[2])
    assert longest[2] == pytest.approx(sub["checker_shadow_h"], abs=1e-9)
    assert longest[1] - longest[0] == pytest.approx(longest[2])


def test_the_subjects_window_spans_every_route_drawn_not_only_its_own(monkeypatch):
    """Protects subject_window against sizing the panel on the time-dependent
    route alone. On the demo's subject all three routes share one span, so the
    real-data check cannot see this; here the mean-map route runs wider."""
    routes = {"td": (np.array([1.0, 1.0]), np.array([2.0, 2.4])),
              "snap": (np.array([1.0, 1.0]), np.array([2.0, 2.4])),
              "mm": (np.array([1.0, 2.2]), np.array([2.0, 2.4]))}
    monkeypatch.setattr(D, "subject_routes", lambda tag, rows, mm: (None, routes, []))
    assert D.zoom_window({"td": routes["td"]}) != D.zoom_window(routes)
    zx, zy = D.subject_window("t", {}, "mm1.0")
    assert zx[0] <= 1.0 - 0.35 and zx[1] >= 2.2 + 0.35


def test_the_hero_crop_refuses_to_take_in_the_next_panel():
    """Protects the hero crop against the 26 px right margin that took in the
    zoom panel's y-axis label, at its measured place, column 478 of 1760."""
    a = np.zeros((990, 1760, 3))
    a[300:400, 478:490] = 0.8                    # the next panel's label
    a[300:600, 459] = 0.5                        # the region panel's frame line
    box = MF.region_crop_box(a.shape)
    assert box[3] < 478 and MF.spill(a, box) == []
    wide = (box[0], box[1], box[2], box[3] + 14)  # the old 26 px margin
    assert MF.spill(a, wide)[:1] == [478]
