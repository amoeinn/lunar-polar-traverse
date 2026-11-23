"""The demo's map frames, rendered from the committed data, show what each frame
is: its own dark cells, the recorded exponent's mean-map route, and notes that
no plotted line crosses.

A review on 2026-11-22 asked three things of the frames. Each is checked here
on rendered figures, with geometry written in this file, not taken from the
code under test:

- The zoom panel's dark cells at a frame equal that frame's lit mask over the
  panel's extent. It was suspected of drawing the union of ever-dark cells, or
  of not updating. Measured, it does neither; the changes are small (5, 7 and
  29 cells by frames 4, 8 and 30), and 56 of the 76-cell cluster's cells are
  dark in every frame. This keeps it so.
- The mean-map route drawn is the one at the exponent P3's record selected. The
  demo carried e = 2.0 as a constant after the corrected criterion selected
  1.0. On this subject the e = 1.0 and e = 2.0 routes are identical cell for
  cell, so the frames are rendered here with the record's exponent replaced by
  0.0, a swept exponent whose route differs, and the drawn route must follow.
- No plotted line, the time cursor included, crosses the executive panel's or
  the shadow panel's note.

Frames 0 to 8 are rendered once, into a temporary directory.
"""
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.figure  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402
from scipy import ndimage  # noqa: E402

import examples.demo_traverse as D  # noqa: E402

N_FRAMES = 9
NEEDED = [D.OUT / "fields.json", D.SCORES]
pytestmark = pytest.mark.skipif(
    not all(p.exists() for p in NEEDED),
    reason="the demo field stack (data/work/demo/fields.json) or "
           "phase3_scores.json is not on disk")


def drawn_hits(ax, bb):
    """Every line on ax whose drawn path, sampled every half pixel, or whose
    marker centre, lies inside bb."""
    hits = []
    for line in ax.get_lines():
        v = line.get_transform().transform(line.get_path().vertices)
        v = v[np.isfinite(v).all(axis=1)]
        if len(v) == 0:
            continue
        pts = [v]
        if line.get_linestyle() not in ("None", "", " "):
            for a, b in zip(v[:-1], v[1:]):
                n = int(np.hypot(*(b - a)) * 2) + 2
                pts.append(a + np.linspace(0, 1, n)[:, None] * (b - a))
        p = np.vstack(pts)
        if np.any((p[:, 0] > bb.x0) & (p[:, 0] < bb.x1) &
                  (p[:, 1] > bb.y0) & (p[:, 1] < bb.y1)):
            hits.append(line.get_gid() or line.get_label())
    return hits


# Where the two notes were fixed before 2026-11-22, for the discriminating check.
OLD_NOTES = {"the world never": (0.985, 0.13, 7.5), "budget is": (0.985, 0.04, 7.0)}


@pytest.fixture(scope="module")
def frames(tmp_path_factory):
    out = tmp_path_factory.mktemp("frames")
    sel = D.report_selection()
    s = sel["r3"]["choice"]
    tag, pi = s["tag"], s["plan_index"]
    rep = D.replay_out(tag, "td")
    _, _, dep = D.plan_at(rep, pi)
    instance = D.planner_out(tag, "td")["index"][pi]
    times = D.frame_times(dep, float(rep["arrival"][pi]))
    p3 = dict(D.p3_record(), exponent=0.0)
    rows = D.route_rows(tag, instance, D.meanmap_arm(p3))
    curves = {k: D.shadow_curve(D.replay_out(tag, k), rows[k], out / "prefix" / k)
              for k in ("td", "snap") if rows[k] is not None}
    sp = D.OUT / f"fields_{tag}_{pi:03d}.npy"
    meta = json.loads((D.OUT / "fields.json").read_text())
    p_, n_, _px, dem_, *_ = D.P.load(D.Rover(v_flat=D.V_FLAT))
    stack = dict(path=str(sp), lit_path=str(sp.with_name(sp.stem + "_lit.npy")),
                 peak=meta["peak_cos_incidence"],
                 lit_fraction=np.asarray(meta["lit_fraction"]),
                 dem=dem_, n=n_, radius_m=p_["radius_m"])
    got = []
    real = matplotlib.figure.Figure.savefig

    def save(fig, fname, *a, **k):
        r = fig.canvas.get_renderer()
        zoom = fig.axes[1]
        im = zoom.images[0]
        mm = [l for l in zoom.get_lines() if l.get_label().startswith("mean-map")]
        notes, old = [], []
        for ax in fig.axes:
            for t in ax.texts:
                if t.get_gid() != "note":
                    continue
                notes.append((t.get_text(), drawn_hits(ax, t.get_window_extent(r))))
                key = next(k_ for k_ in OLD_NOTES if k_ in t.get_text())
                x, y, fs = OLD_NOTES[key]
                probe = ax.text(x, y, t.get_text(), ha="right", fontsize=fs,
                                transform=ax.transAxes)
                old.append((t.get_text(), drawn_hits(ax, probe.get_window_extent(r))))
                probe.remove()
        got.append(dict(
            frame=int(str(fname)[-8:-4]), arr=np.asarray(im.get_array()).copy(),
            extent=im.get_extent(), xlim=zoom.get_xlim(), ylim=zoom.get_ylim(),
            mm=[(l.get_label(), np.asarray(l.get_xdata()), np.asarray(l.get_ydata()))
                for l in mm],
            notes=notes, old=old,
            panel=fig.axes[-1].texts[0].get_text()))
        return real(fig, fname, *a, **k)
    mp = pytest.MonkeyPatch()
    mp.setattr(matplotlib.figure.Figure, "savefig", save)
    try:
        D.render_frames(tag, instance, times[:N_FRAMES], stack, rows, curves,
                        out / "map", pc=sel["pair_contingency"], p3=p3)
    finally:
        mp.undo()
    return dict(got=got, tag=tag, instance=instance, rows=rows,
                litm=np.load(stack["lit_path"], mmap_mode="r"))


def window(c):
    """The cells whose centres lie inside the zoom axes' limits, from the
    image's own extent and shape."""
    x0, x1, y0, y1 = c["extent"]
    nr, nc = c["arr"].shape
    xc = x0 + (np.arange(nc) + 0.5) * (x1 - x0) / nc
    yc = y1 - (np.arange(nr) + 0.5) * (y1 - y0) / nr
    cols = np.flatnonzero((xc > min(c["xlim"])) & (xc < max(c["xlim"])))
    rows = np.flatnonzero((yc > min(c["ylim"])) & (yc < max(c["ylim"])))
    return np.ix_(rows, cols)


def test_the_zoom_panel_draws_each_frames_own_dark_cells(frames):
    """Protects the zoom panel against drawing the union of ever-dark cells or a
    frame other than its own. Discriminating property first: inside the panel
    a cluster of cells dark in at least two frames changes state between two
    rendered frames, and frame 0's dark cells are not the union."""
    got, litm = frames["got"], frames["litm"]
    w = window(got[0])
    dark = {c["frame"]: ~np.asarray(litm[c["frame"]])[w].astype(bool) for c in got}
    ever = (~np.asarray(litm).astype(bool)).sum(axis=0)[w]
    lab, n = ndimage.label(ever >= 2)
    assert n > 0
    changes = [c["frame"] for c in got[1:]
               if ((dark[c["frame"]] != dark[0]) & (lab > 0)).any()]
    assert changes, "no cluster changes state between the rendered frames"
    assert ((ever > 0) != dark[0]).any(), "frame 0's dark cells are the union"
    for c in got:
        shown = c["arr"][window(c)] == 0
        assert shown.shape == dark[c["frame"]].shape
        assert np.array_equal(shown, dark[c["frame"]]), (
            c["frame"], int((shown != dark[c["frame"]]).sum()))


def test_the_zoom_panel_draws_the_recorded_exponents_route(frames):
    """Protects the mean-map route following P3's record, not a constant.
    Discriminating property first: at e = 0.0 the route differs from e = 1.0."""
    tag, inst = frames["tag"], frames["instance"]
    p0 = D.plan_at(D.replay_out(tag, "mm0.0"), inst)[0]
    p1 = D.plan_at(D.replay_out(tag, "mm1.0"), inst)[0]
    assert p0.size != p1.size or not np.array_equal(p0, p1), "routes coincide"
    x, y = D.cells_to_map(p0)
    for c in frames["got"]:
        (lab, xs, ys), = c["mm"]
        assert lab == "mean-map e=0 plan", lab
        assert np.allclose(xs, x / 1000.0) and np.allclose(ys, y / 1000.0)
        assert "mean-map e=0 " in c["panel"]


def test_no_plotted_line_crosses_a_note_in_any_frame(frames):
    """Protects the notes, the cursor included, on every rendered frame.
    Discriminating property first: at their old fixed positions, some line
    crossed each note in some frame."""
    got = frames["got"]
    for key in OLD_NOTES:
        assert any(hits for c in got for text, hits in c["old"] if key in text), key
    for c in got:
        assert len(c["notes"]) == 2, c["notes"]
        for text, hits in c["notes"]:
            assert hits == [], (c["frame"], text, hits)


def test_the_zoom_panel_draws_the_window_the_sidecar_reports(frames):
    """Protects the window width a caption quotes from the sidecar against the
    width actually drawn: subject_geometry and every rendered frame's zoom axes
    must agree, at the exponent the frames were rendered with."""
    rows = frames["rows"]
    g = D.subject_geometry(frames["tag"], rows, rows["mm_kind"])
    for c in frames["got"]:
        (x0, x1), (y0, y1) = c["xlim"], c["ylim"]
        assert (x1 - x0) * 1000.0 == pytest.approx(g["zoom_window_m"])
        assert (y1 - y0) * 1000.0 == pytest.approx(g["zoom_window_m"])
