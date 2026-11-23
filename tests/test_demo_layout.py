"""The demo's executive traces and dark-cluster labels are drawn where they can
be seen.

On 2026-11-22 a smoke frame of R3's subject showed the bounded rule firing at
3.31 h with no orange trace. The rule fired at the first tick of each of its two
legs, so each leg's trace was a single point. Leg 0's point was drawn 3 px wide
under a 12 px firing cross, and leg 1 was never read, because only leg 0 was.
These tests protect every leg of every arm being plotted, every logged tick
carrying a visible marker above a hollow firing mark, and a firing being marked
on every leg that fired.

The same frame's zoom panel stacked its 26 cluster labels on one another. They
were placed at a fixed offset from each cluster. A first replacement, two edge
columns, kept the text apart but laid labels over the clusters they named and
ran leaders through other labels. These tests protect what the placement now
guarantees, checked on the drawn text by the test itself: no two labels overlap;
none sits on a dark cell, a route, a cluster's point or the legend; no leader
runs through a label, a route or the legend, or across another leader; and the
legend covers no cluster.
"""
import itertools

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.text as mtext  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402

import examples.demo_traverse as D  # noqa: E402

TAG, PLAN = "2028-02-15_p0", 32


def write_leg(root, arm, leg, t, s_pred, fired):
    d = root / arm / TAG / f"plan{PLAN:03d}" / f"leg{leg}_exec"
    d.mkdir(parents=True)
    n = len(t)
    cols = dict(t=np.asarray(t, float), s_pred=np.asarray(s_pred, float),
                fired=np.asarray(fired))
    for k, dt in D.LOG_DT.items():
        np.asarray(cols.get(k, np.zeros(n)), dt).tofile(d / f"out_log_{k}.bin")


@pytest.fixture
def legs(tmp_path, monkeypatch):
    """R3's shape: the bounded rule fires at the first tick of two legs, the lit
    arm logs several ticks and never fires."""
    monkeypatch.setattr(D, "CORR", tmp_path)
    write_leg(tmp_path, "none", 0, [0.653], [3.31], [1])
    write_leg(tmp_path, "none", 1, [1.292], [3.213], [1])
    write_leg(tmp_path, "lit", 0, [0.65, 1.3, 1.9, 2.5, 3.1],
              [0.98, 0.95, 0.96, 1.74, 0.0], [0, 0, 0, 0, 0])
    return {a: D.exec_legs(a, TAG, PLAN) for a in ("none", "lit")}


def test_every_leg_is_read(legs):
    """Protects the omission: a firing that replans continues in leg 1, and
    reading leg 0 alone dropped it."""
    assert [len(legs["none"]), len(legs["lit"])] == [2, 1]
    assert [float(g["t"][0]) for g in legs["none"]] == [0.653, 1.292]


def test_every_plotted_rule_has_a_visible_trace_on_every_tick(legs):
    """Protects the trace: every rule's plotted points are exactly its logged
    ticks over all legs, every point carries a marker at least 5 px wide, and
    the markers are drawn above the firing marks, which are hollow, one per
    leg that fired."""
    fig, ax = plt.subplots()
    fires = D.draw_exec_traces(ax, [("none", "C1", "bounded", legs["none"]),
                                    ("lit", "C2", "lit", legs["lit"])])
    for arm in ("none", "lit"):
        lines = [l for l in ax.get_lines() if l.get_gid() == arm]
        assert lines, f"{arm}: no trace plotted"
        got = np.concatenate([l.get_xdata() for l in lines])
        want = np.concatenate([g["t"] for g in legs[arm]])
        assert np.array_equal(np.sort(got), np.sort(want)), (arm, got, want)
        ys = np.concatenate([l.get_ydata() for l in lines])
        assert np.array_equal(np.sort(ys),
                              np.sort(np.concatenate([g["s_pred"] for g in legs[arm]])))
        for l in lines:
            assert l.get_marker() not in (None, "", " ", "None"), arm
            assert l.get_markersize() >= 5, (arm, l.get_markersize())
    marks = [l for l in ax.get_lines() if str(l.get_gid()).startswith("fire:")]
    assert [(m.get_xdata()[0], m.get_ydata()[0]) for m in marks] == \
        [(0.653, 3.31), (1.292, 3.213)]
    assert len(fires) == 2
    trace_z = min(l.get_zorder() for l in ax.get_lines() if l.get_gid() == "none")
    for m in marks:
        assert m.get_markerfacecolor() == "none", "a filled cross hides the tick"
        assert m.get_zorder() < trace_z, "the cross is drawn over the tick"
    plt.close(fig)


# --- cluster labels -------------------------------------------------------------

def dense_fixture(seed=3, n=26):
    """A 1 km window with n small clusters, half crowded along the top edge and
    one in the lower left corner where the legend would go, plus a route."""
    rng = np.random.default_rng(seed)
    top = np.column_stack([rng.uniform(0.03, 0.97, n // 2), rng.uniform(0.82, 0.98, n // 2)])
    rest = np.column_stack([rng.uniform(0.05, 0.95, n - n // 2 - 1),
                            rng.uniform(0.25, 0.75, n - n // 2 - 1)])
    pts = np.vstack([top, rest, [[0.08, 0.06]]])
    clusters = [dict(cells=int(rng.integers(1, 40)), x=float(x), y=float(y),
                     lo=int(rng.integers(2, 50)), hi=50, frames=50, always=0)
                for x, y in pts]
    cells = np.vstack([pts + d for d in ((0, 0), (0.02, 0), (0, 0.02))])
    route = (np.array([0.45, 0.47, 0.55, 0.6]), np.array([0.2, 0.45, 0.55, 0.7]))
    return clusters, cells, route


def draw_fixture(clusters, cells, route):
    fig = plt.figure(figsize=(16, 9), dpi=110)
    ax = fig.add_axes(D.ZOOM_RECT)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.plot(*route, "-", label="route")
    ax.plot([], [], " ", label=D.cluster_key(50))
    leg = ax.legend(loc="lower left", fontsize=6.5, borderaxespad=0.0)
    bare = D.label_obstacles(ax, cells_km=cells, cell_m=20.0, lines_km=[route])
    leg.set_bbox_to_anchor(D.legend_anchor(ax, leg, bare, clusters),
                           transform=ax.transAxes)
    obst = D.label_obstacles(ax, cells_km=cells, cell_m=20.0, lines_km=[route],
                             avoid=[leg])
    anns = D.place_cluster_labels(ax, clusters, obst)
    fig.canvas.draw()
    return fig, ax, leg, anns


def frac(ax, bb):
    a = ax.get_window_extent()
    return ((bb.x0 - a.x0) / a.width, (bb.y0 - a.y0) / a.height,
            (bb.x1 - a.x0) / a.width, (bb.y1 - a.y0) / a.height)


def overlap(a, b):
    return a[0] < b[2] and b[0] < a[2] and a[1] < b[3] and b[1] < a[3]


# The test's own geometry, written apart from the placer's, so a fault in the
# placer's helpers cannot also blind the check.
def segs_cross(p, q, a, b):
    def side(u, v, w):
        return np.sign((v[0] - u[0]) * (w[1] - u[1]) - (v[1] - u[1]) * (w[0] - u[0]))
    return side(a, b, p) * side(a, b, q) < 0 and side(p, q, a) * side(p, q, b) < 0


def seg_through_box(p, q, b):
    """Sampled: any of 400 points along p-q strictly inside b."""
    t = np.linspace(0, 1, 400)
    x, y = p[0] + t * (q[0] - p[0]), p[1] + t * (q[1] - p[1])
    return bool(np.any((b[0] < x) & (x < b[2]) & (b[1] < y) & (y < b[3])))


def test_the_fixture_overlaps_under_a_fixed_offset():
    """The discriminating property, asserted before the others are believed: on
    this fixture the placement being replaced does overlap."""
    clusters, _, _ = dense_fixture()
    fig = plt.figure(figsize=(16, 9), dpi=110)
    ax = fig.add_axes(D.ZOOM_RECT)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ts = [ax.annotate(D.cluster_text(c), xy=(c["x"], c["y"]),
                      xytext=(c["x"] - 0.40, c["y"] + (0.17 if i % 2 == 0 else -0.17)),
                      fontsize=D.LABEL_FONT) for i, c in enumerate(clusters)]
    fig.canvas.draw()
    boxes = [frac(ax, mtext.Text.get_window_extent(t)) for t in ts]
    assert any(overlap(a, b) for a, b in itertools.combinations(boxes, 2))
    plt.close(fig)


def test_no_two_cluster_labels_overlap_and_none_hides_what_it_names():
    """Protects the placement, checked on the drawn figure by this test and
    not only by the placer's own check."""
    clusters, cells, route = dense_fixture()
    fig, ax, leg, anns = draw_fixture(clusters, cells, route)
    assert len(anns) == len(clusters)
    boxes = [frac(ax, mtext.Text.get_window_extent(a)) for a in anns]
    for (i, a), (j, b) in itertools.combinations(enumerate(boxes), 2):
        assert not overlap(a, b), f"labels {i} and {j} overlap: {a} {b}"
    lb = frac(ax, leg.get_window_extent())
    tr = ax.transData + ax.transAxes.inverted()
    pts = [tuple(tr.transform((c["x"], c["y"]))) for c in clusters]
    cell_pts = [tuple(tr.transform(p)) for p in cells]
    rx, ry = route
    route_pts = [tuple(tr.transform((x0 + f * (x1 - x0), y0 + f * (y1 - y0))))
                 for x0, y0, x1, y1 in zip(rx[:-1], ry[:-1], rx[1:], ry[1:])
                 for f in np.linspace(0, 1, 100)]

    def inside(p, b):
        return b[0] < p[0] < b[2] and b[1] < p[1] < b[3]
    for i, b in enumerate(boxes):
        assert not overlap(b, lb), f"label {i} on the legend"
        assert not any(inside(p, b) for p in pts), f"label {i} over a cluster point"
        assert not any(inside(p, b) for p in cell_pts), f"label {i} over a dark cell"
        assert not any(inside(p, b) for p in route_pts), f"label {i} over the route"
    assert not any(inside(p, lb) for p in pts), "the legend covers a cluster"
    rseg = [tuple(tr.transform(p)) for p in zip(rx, ry)]
    for i, a in enumerate(anns):
        c = ((boxes[i][0] + boxes[i][2]) / 2, (boxes[i][1] + boxes[i][3]) / 2)
        for j, b in enumerate(boxes):
            if i != j:
                assert not seg_through_box(pts[i], c, b), f"leader {i} runs through label {j}"
        assert not seg_through_box(pts[i], c, lb), f"leader {i} runs through the legend"
        for u, v in zip(rseg[:-1], rseg[1:]):
            assert not segs_cross(pts[i], c, u, v), f"leader {i} crosses the route"
        for j in range(i + 1, len(anns)):
            cj = ((boxes[j][0] + boxes[j][2]) / 2, (boxes[j][1] + boxes[j][3]) / 2)
            assert not segs_cross(pts[i], c, pts[j], cj), f"leaders {i} and {j} cross"
    plt.close(fig)


def test_the_fault_check_finds_each_planted_fault():
    """Protects label_faults, which both the placer and the drawn-figure check
    rest on: each kind of fault is found, and a clean layout has none."""
    g = D.LABEL_GRID
    occ, lead = np.zeros((g, g), bool), np.zeros((g, g), bool)
    occ[200:210, 200:210] = True
    lead[100:110, :] = True
    occ |= lead
    pts = [(0.1, 0.9), (0.9, 0.9)]
    clean = [(0.15, 0.88, 0.35, 0.92), (0.6, 0.88, 0.85, 0.92)]
    assert D.label_faults(clean, pts, (occ, lead)) == []
    kinds = lambda b, p=pts: {f[0] for f in D.label_faults(b, p, (occ, lead))}
    assert "overlap" in kinds([(0.15, 0.88, 0.35, 0.92), (0.3, 0.88, 0.5, 0.92)])
    assert "obstacle" in kinds([clean[0], (0.76, 0.76, 0.84, 0.84)])
    assert "covers" in kinds([(0.05, 0.88, 0.35, 0.92), clean[1]])
    assert "leader" in kinds([(0.6, 0.88, 0.7, 0.92), (0.3, 0.88, 0.5, 0.92)],
                             [(0.9, 0.9), (0.95, 0.9)])
    assert "leader on route or legend" in kinds([(0.15, 0.2, 0.3, 0.25), clean[1]],
                                               [(0.2, 0.6), (0.9, 0.9)])
    assert "leaders cross" in kinds([(0.6, 0.2, 0.7, 0.25), (0.6, 0.6, 0.7, 0.65)],
                                    [(0.1, 0.65), (0.1, 0.2)])


def test_an_impossible_layout_raises_rather_than_overlapping():
    """Protects the refusal: a panel with no room left for a label raises."""
    fig = plt.figure(figsize=(16, 9), dpi=110)
    ax = fig.add_axes(D.ZOOM_RECT)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    g = D.LABEL_GRID
    full = (np.ones((g, g), bool), np.zeros((g, g), bool))
    with pytest.raises(RuntimeError, match="do not fit"):
        D.plan_cluster_labels(ax, dense_fixture()[0][:3], full)
    plt.close(fig)


def test_no_leader_crosses_the_route():
    """Protects leaders against the route: a cluster beside the route with dark
    cells on its other side has its nearest clear spot across the route. The
    discriminating property is asserted first: with the route removed from the
    leader obstacles, the planned leader does cross it."""
    fig = plt.figure(figsize=(16, 9), dpi=110)
    ax = fig.add_axes(D.ZOOM_RECT)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    route = (np.array([0.45, 0.45]), np.array([0.0, 1.0]))
    gx, gy = np.meshgrid(np.arange(0.02, 0.40, 0.02), np.arange(0.36, 0.66, 0.02))
    cells = np.column_stack([gx.ravel(), gy.ravel()])
    cl = [dict(cells=3, x=0.41, y=0.5, lo=5, hi=50, frames=50, always=0)]
    occ, lead = D.label_obstacles(ax, cells_km=cells, cell_m=20.0, lines_km=[route])
    tr = ax.transData + ax.transAxes.inverted()
    p = tuple(tr.transform((0.41, 0.5)))
    u, v = tuple(tr.transform((0.45, 0.0))), tuple(tr.transform((0.45, 1.0)))

    def leader_crosses(obst):
        fs, (box,) = D.plan_cluster_labels(ax, cl, obst)
        c = ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2)
        return segs_cross(p, c, u, v)
    assert leader_crosses((occ, np.zeros_like(lead))), "the fixture cannot express it"
    assert not leader_crosses((occ, lead))
    plt.close(fig)


# --- notes on the executive and shadow panels -------------------------------------
#
# On 2026-11-22 the executive panel's false-alarm note and the shadow panel's
# budget note sat at fixed positions, and the lit arm's trace and the world's
# shadow trace ran through them. These protect a note being placed where no
# plotted line crosses it, the time cursor being broken around a note, and the
# check that refuses a crossed note finding every kind of crossing.

def drawn_crossings(ax, note):
    """The test's own geometry: every line's drawn path (matplotlib's, steps
    included), transformed to the screen and sampled every half pixel, and its
    marker centres, against the note's text box. Names each line that enters."""
    r = ax.figure.canvas.get_renderer()
    bb = note.get_window_extent(r)
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


def busy_axes():
    """A panel shaped like the executive one: a step trace along the bottom that
    rises through the lower right, a budget line, a marker-only series."""
    fig = plt.figure(figsize=(16, 9), dpi=110)
    ax = fig.add_axes(D.EXEC_RECT)
    ax.set_xlim(-0.6, 12.6)
    ax.set_ylim(-0.2, 4.0)
    t = np.linspace(0, 12, 49)
    ax.step(t, np.where((t > 8) & (t < 11), 1.2, 0.3), where="post", gid="step")
    ax.axhline(2.0, ls="--", gid="budget")
    ax.plot(t[::4], 0.1 + 0.02 * t[::4], "o", gid="markers")
    leg = ax.legend(["a", "b"], loc="upper left")
    return fig, ax, leg


TEXT = "the world never exceeds 1.00 h, so the firing is a false alarm"


def test_a_note_is_moved_off_the_lines_through_its_preferred_spot():
    """Protects the placement. The discriminating property first: at the
    preferred spot, where the note used to be fixed, a line crosses it."""
    fig, ax, leg = busy_axes()
    fixed = ax.text(0.985, 0.13, TEXT, ha="right", va="bottom",
                    transform=ax.transAxes, fontsize=7.5)
    assert drawn_crossings(ax, fixed), "the fixture cannot express a crossing"
    fixed.remove()
    note = D.place_note(ax, TEXT, (0.985, 0.13), ha="right", avoid=[leg], fontsize=7.5)
    assert drawn_crossings(ax, note) == []
    lb, nb = leg.get_window_extent(), note.get_window_extent()
    assert not overlap(tuple(lb.extents), tuple(nb.extents)), "note on the legend"
    plt.close(fig)


def test_the_cursor_is_broken_around_a_note():
    """Protects the cursor: it sweeps every time, so it would cross any note in
    some frame. Discriminating property first: an unbroken cursor at that
    time crosses the note."""
    fig, ax, leg = busy_axes()
    note = D.place_note(ax, TEXT, (0.985, 0.13), ha="right", avoid=[leg], fontsize=7.5)
    bb = note.get_window_extent()
    t = float(ax.transData.inverted().transform(((bb.x0 + bb.x1) / 2, 0))[0])
    full = ax.axvline(t)
    assert drawn_crossings(ax, note) == [full.get_label()]
    full.remove()
    pieces = D.draw_cursor(ax, t, [note])
    assert len(pieces) == 2 and drawn_crossings(ax, note) == []
    assert D.note_faults(ax, [note]) == []
    plt.close(fig)


def test_the_note_check_finds_each_kind_of_crossing():
    """Protects check_notes and note_faults, which refuse a crossed note: a
    plain segment, a step, an axhline, a marker, and the cursor are each found,
    and a clear note is not refused."""
    fig = plt.figure(figsize=(16, 9), dpi=110)
    ax = fig.add_axes(D.SHADOW_RECT)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 1)
    note = ax.text(0.5, 0.5, "a note in the middle", ha="center", va="center",
                   transform=ax.transAxes, gid="note")
    assert D.note_faults(ax, [note]) == []
    D.check_notes(ax, [note])
    for gid, draw in (("segment", lambda: ax.plot([0, 10], [0.2, 0.8], gid="segment")[0]),
                      # Only the drawn step enters the note; the straight
                      # line between the same points passes below it.
                      ("step", lambda: ax.step([0, 5, 10], [0.5, 0.0, 0.0],
                                               where="post", gid="step")[0]),
                      ("budget", lambda: ax.axhline(0.5, gid="budget")),
                      ("marker", lambda: ax.plot([5], [0.5], "o", gid="marker")[0]),
                      ("cursor", lambda: ax.axvline(5.0, gid="cursor"))):
        line = draw()
        assert [b[1] for b in D.note_faults(ax, [note])] == [gid], gid
        with pytest.raises(RuntimeError, match="cross"):
            D.check_notes(ax, [note])
        line.remove()
    plt.close(fig)


def test_a_note_with_no_room_is_refused():
    """Protects the refusal: no clear place raises rather than draw it crossed."""
    fig = plt.figure(figsize=(16, 9), dpi=110)
    ax = fig.add_axes(D.SHADOW_RECT)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    for y in np.linspace(0, 1, 60):
        ax.axhline(y, lw=1.0)
    with pytest.raises(RuntimeError, match="no place"):
        D.place_note(ax, TEXT, (0.985, 0.04), ha="right", fontsize=7)
    plt.close(fig)


def test_the_text_panel_may_not_run_into_the_captions():
    """Protects check_text_clear: the longer pairing statement ran the text
    panel's last line into the first caption."""
    fig = plt.figure(figsize=(16, 9), dpi=110)
    panel = fig.text(0.7, 0.20, "\n".join(["a line"] * 12), va="top", fontsize=7.6)
    cap = fig.text(0.025, 0.068, "x" * 300, fontsize=7)
    with pytest.raises(RuntimeError, match="runs into"):
        D.check_text_clear(fig, panel, [cap])
    panel.set_y(0.40)
    D.check_text_clear(fig, panel, [cap])
    plt.close(fig)
