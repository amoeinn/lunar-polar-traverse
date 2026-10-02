"""The time-dependent planner and check 3's reference, each test naming its target.

PLAN.md, "Planners", registers the dominance rule with its reachability clause,
the shadow-integration rule, and check 3's inequality and convergence criterion.
"""
import numpy as np
import pytest

from src.cost import MOON_RADIUS_M
from src.frames import cell_position, local_frame
from src.intervals import LitIntervals, SunTable, g_at
from src.planner import Graph, _departure_candidates, plan
from src.reference import plan_time_expanded

R = MOON_RADIUS_M
AU = 1.495978707e11


def world(transitions, lit0, horizon=6.0):
    times, off = [], [0]
    for c in range(len(lit0)):
        arr = np.array(transitions.get(c, []), dtype=float)
        times.append(arr)
        off.append(off[-1] + arr.size)
    return LitIntervals(np.concatenate(times) if times else np.zeros(0),
                        np.array(off), np.array(lit0, dtype=bool), horizon)


def chain(travel):
    """A path graph 0 -> 1 -> ... with the given travel times."""
    n = len(travel) + 1
    indptr = np.arange(n + 1, dtype=np.int64)
    indptr[-1] = len(travel)
    indptr = np.clip(np.arange(n + 1, dtype=np.int64), 0, len(travel))
    return Graph(indptr, np.arange(1, n, dtype=np.int64), np.array(travel, float), n)


# The reachability counterexample, used by several tests below. Cell 2 is lit
# [0.3, 0.8), dark [0.8, 1.2), lit after; cell 3 is dark until 2.0.
COUNTER_TRANS = {2: [0.3, 0.8, 1.2], 3: [2.0]}
COUNTER_LIT0 = [True, True, False, False]
COUNTER_GRAPH = Graph(np.array([0, 2, 3, 4, 4], dtype=np.int64),
                      np.array([2, 1, 2, 3], dtype=np.int64),
                      np.array([0.5, 0.45, 0.85, 0.6]), 4)
COUNTER_TOL = 0.5


def test_shadow_along_follows_the_registered_union_rule():
    """Protects the shadow rule: either endpoint dark, reset on light, max kept."""
    iv = world({0: [1.0, 3.0, 6.0]}, [True], horizon=10.0)
    assert iv.shadow_along(0, 0, 0.0, 10.0, 0.0) == (4.0, 4.0)
    assert iv.shadow_along(0, 0, 1.5, 2.5, 0.5) == (1.5, 1.5)
    assert iv.shadow_along(0, 0, 3.5, 5.5, 0.0) == (0.0, 0.0)
    # The maximum is what the tolerance constrains, and it can exceed the value
    # at arrival: dark for two hours, then lit, then arrive.
    s_arr, s_max = iv.shadow_along(0, 0, 1.0, 4.0, 0.0)
    assert (s_arr, s_max) == (0.0, 2.0)
    # Two cells: the union is dark wherever either is.
    iv2 = world({0: [1.0, 2.0], 1: [3.0, 4.0]}, [True, True], horizon=6.0)
    assert iv2.shadow_along(0, 1, 0.0, 5.0, 0.0) == (0.0, 1.0)


def test_intervals_bisect_every_sign_change():
    """Protects interval construction against an analytic sun: no missed crossing."""
    x = np.array([0.0]); y = np.array([1000.0])
    up, east, north = local_frame(x, y, R)
    origin = cell_position(x, y, np.array([0.0]), R)

    class Cells:
        pass
    cells = Cells()
    cells.origin, cells.up, cells.east, cells.north = origin, up, east, north
    cells.__class__.__len__ = lambda self: 1

    dt = 0.05
    ts = np.arange(0.0, 12.0 + dt, dt)
    el = np.radians(2.0 * np.sin(2.0 * np.pi * ts / 5.0))       # peaks at 2 deg
    sun = SunTable(AU * (np.cos(el)[:, None] * north + np.sin(el)[:, None] * up), dt)
    profiles = np.full((1, 360), 1.0)                            # flat 1 deg horizon
    iv = LitIntervals.build(cells, profiles, sun, 12.0, cadence_h=0.25)

    fine = np.arange(0.0, 12.0, 0.002)
    g = g_at(cells, profiles, sun, fine)[0]
    crossings = fine[:-1][np.sign(g[:-1]) != np.sign(g[1:])]
    found = iv.transitions(0)
    assert found.size == crossings.size, (found.size, crossings.size)
    assert np.allclose(found, crossings, atol=0.003)
    for t in fine[::37]:
        assert iv.is_lit(0, float(t)) == bool(np.interp(t, fine, g) > 0.0)


def test_dominance_keeps_labels_the_naive_rule_would_prune():
    """Protects the reachability clause. Without it this instance is unreachable.

    Two labels reach cell 2 with equal shadow and different times, the later one
    only because the rover waited at the start. The naive rule, earlier time and
    no more shadow dominates, prunes the later one -- and the later one carries
    the only feasible continuation, because cell 2's lit window at the earlier
    time closes before any departure that clears the tolerance.
    """
    iv = world(COUNTER_TRANS, COUNTER_LIT0)
    arrival, path, st = plan(COUNTER_GRAPH, iv, 0, 3, 0.0, COUNTER_TOL)
    assert arrival == pytest.approx(2.1, abs=1e-9), arrival
    assert path == [0, 2, 3]
    labels = st.labels_per_cell[2]
    assert labels >= 2, labels
    # The discriminating property, asserted rather than assumed: the naive rule
    # really would have pruned here.
    assert not iv.is_lit(2, 1.0), "cell 2 must go dark between the two labels"
    assert iv.is_lit(2, 0.5) and iv.is_lit(2, 1.3)


def test_departure_candidates_include_the_constraint_crossing():
    """Protects the closed-form candidate set: breakpoints alone are not enough."""
    iv = world(COUNTER_TRANS, COUNTER_LIT0)
    cands = _departure_candidates(iv, 2, 3, 1.3, iv.horizon_h, 0.6, COUNTER_TOL)
    assert any(abs(c - 1.5) < 1e-12 for c in cands), cands
    structural = {1.3} | {2.0, 2.0 - 0.6}
    assert not any(abs(c - 1.5) < 1e-12 for c in structural), \
        "1.5 must not be a structural breakpoint, or the test proves nothing"


def test_waiting_is_used_when_leaving_at_once_fails():
    """Protects that waiting is generated at all, not only allowed in principle."""
    iv = world(COUNTER_TRANS, COUNTER_LIT0)
    arrival, _, _ = plan(COUNTER_GRAPH, iv, 0, 3, 0.0, COUNTER_TOL)
    assert arrival > 0.5 + 0.6, "leaving at once would arrive at 1.1"


def test_waiting_is_not_allowed_in_a_dark_cell():
    """Protects the registered restriction: no waiting out a shadow in place.

    Every cell on the route is dark when the rover reaches it, so there is
    nowhere legal to wait. Waiting at cell 1 until both it and cell 2 are lit
    would arrive comfortably; the instance is infeasible precisely because that
    is forbidden, which is the only way this restriction can be tested.
    """
    g = Graph(np.array([0, 1, 2, 2], dtype=np.int64), np.array([1, 2], dtype=np.int64),
              np.array([0.3, 0.3]), 3)
    iv = world({1: [1.0], 2: [1.1]}, [False, False, False], horizon=8.0)
    assert plan(g, iv, 0, 2, 0.0, 0.5)[0] is None
    # and the waiting route it is denied really would have worked
    assert iv.is_lit(1, 1.1) and iv.is_lit(2, 1.4)


def test_tolerance_constrains_the_maximum_not_the_value_at_arrival():
    """Protects using s_max: a traverse that dips into light at the end still fails."""
    g = Graph(np.array([0, 1, 1], dtype=np.int64), np.array([1], dtype=np.int64),
              np.array([2.0]), 2)
    iv = world({0: [1.9], 1: [1.9]}, [False, False], horizon=8.0)
    # Over [0, 2] the union is dark until 1.9 then lit, so s_arr is 0 but the
    # maximum is 1.9. A tolerance of 1.0 must reject it; 2.0 must accept it.
    assert plan(g, iv, 0, 1, 0.0, 1.0)[0] is None
    assert plan(g, iv, 0, 1, 0.0, 2.0)[0] == pytest.approx(2.0)


def test_unreachable_goal_is_a_result_not_an_error():
    """Protects the phase 2 boundary note: the region is never fully connected."""
    g = Graph(np.array([0, 1, 1, 1], dtype=np.int64), np.array([1], dtype=np.int64),
              np.array([0.1]), 3)
    iv = world({}, [True, True, True], horizon=5.0)
    arrival, path, st = plan(g, iv, 0, 2, 0.0, 1.0)
    assert arrival is None and path == [] and st.popped > 0


def test_travel_time_does_not_depend_on_departure_time():
    """Protects the FIFO property the dominance rule rests on."""
    g = Graph(np.array([0, 1, 1], dtype=np.int64), np.array([1], dtype=np.int64),
              np.array([0.3]), 2)
    iv = world({}, [True, True], horizon=9.0)
    for t0 in (0.0, 1.0, 2.5, 4.25):
        arrival, _, _ = plan(g, iv, 0, 1, t0, 1.0)
        assert arrival == pytest.approx(t0 + 0.3, abs=1e-12)


def dark_oracle(iv, dt):
    """dark(cell, k): is the cell dark anywhere in step k. Conservative on purpose."""
    def dark(cell, k):
        return bool(iv.dark_spans((cell,), k * dt, (k + 1) * dt))
    return dark


def test_reference_never_beats_the_planner():
    """Check 3's registered inequality: T_interval <= T_dt + 1e-9 h, every dt."""
    iv = world(COUNTER_TRANS, COUNTER_LIT0)
    t_int, _, _ = plan(COUNTER_GRAPH, iv, 0, 3, 0.0, COUNTER_TOL)
    for dt in (0.25, 0.125, 0.0625, 0.03125):
        t_dt = plan_time_expanded(COUNTER_GRAPH, dark_oracle(iv, dt),
                                  0, 3, 0.0, COUNTER_TOL, dt, iv.horizon_h)
        assert t_dt is not None, dt
        assert t_int <= t_dt + 1e-9, (dt, t_int, t_dt)


def test_reference_gap_shrinks_as_dt_shrinks():
    """Check 3's convergence: the gap must fall monotonically as dt halves."""
    iv = world(COUNTER_TRANS, COUNTER_LIT0)
    t_int, _, _ = plan(COUNTER_GRAPH, iv, 0, 3, 0.0, COUNTER_TOL)
    gaps = []
    for dt in (0.2, 0.1, 0.05, 0.025):
        t_dt = plan_time_expanded(COUNTER_GRAPH, dark_oracle(iv, dt),
                                  0, 3, 0.0, COUNTER_TOL, dt, iv.horizon_h)
        gaps.append(t_dt - t_int)
    assert all(b <= a + 1e-12 for a, b in zip(gaps, gaps[1:])), gaps
    assert gaps[-1] < 0.01 * t_int, gaps
    assert gaps[0] > gaps[-1], "the fixture must have a gap to shrink"


def test_reachability_precheck_is_exact():
    """Protects pruning (a): it may answer faster, never differently.

    A goal in another static component is unreachable with the shadow
    constraint too, since adding a constraint only removes solutions. The test
    asserts the precheck agrees with the search on both sides: the cross
    component pair, where both say no path, and the connected pair, where both
    must give the same arrival.
    """
    g = Graph(np.array([0, 1, 1, 1], dtype=np.int64), np.array([1], dtype=np.int64),
              np.array([0.3]), 3)
    iv = world({}, [True, True, True], horizon=9.0)
    comp = np.array([0, 0, 1])

    plain = plan(g, iv, 0, 2, 0.0, 1.0)
    pruned = plan(g, iv, 0, 2, 0.0, 1.0, components=comp)
    assert plain[0] is None and pruned[0] is None
    assert pruned[2].pruned_component == 1, "the precheck must be what answered"
    assert plain[2].popped > 0 and pruned[2].popped == 0

    a, b = plan(g, iv, 0, 1, 0.0, 1.0), plan(g, iv, 0, 1, 0.0, 1.0, components=comp)
    assert a[0] == b[0] == pytest.approx(0.3) and a[1] == b[1], (a, b)


def test_arrival_bound_is_exact_and_fires():
    """Protects pruning (b): a label that cannot arrive in time is dropped.

    The bound is the straight-line distance to the goal over the top speed,
    which is a lower bound on any continuation, so dropping a label whose time
    plus that bound exceeds the horizon cannot remove a solution. The fixture
    contains a dead end far from the goal so the pruning actually fires, which
    is asserted: a pruning that never fires proves nothing.
    """
    # 0 -> 1 -> 2 (goal), and 0 -> 3, a dead end 1000 m away from the goal.
    g = Graph(np.array([0, 2, 3, 3, 3], dtype=np.int64),
              np.array([1, 3, 2], dtype=np.int64), np.array([0.5, 0.2, 0.5]), 4)
    iv = world({}, [True, True, True, True], horizon=1.2)
    v_cap = 75.0
    pos = np.array([[60.0, 0.0, 0.0], [45.0, 0.0, 0.0], [0.0, 0.0, 0.0],
                    [1000.0, 0.0, 0.0]])

    plain = plan(g, iv, 0, 2, 0.0, 1.0)
    pruned = plan(g, iv, 0, 2, 0.0, 1.0, positions=pos, v_cap=v_cap)
    assert plain[0] == pytest.approx(1.0) and pruned[0] == pytest.approx(1.0)
    assert plain[1] == pruned[1]
    assert pruned[2].pruned_bound > 0, "the bound must fire, or the test is empty"
    assert plain[2].pruned_bound == 0

    # The discriminating property: the label on the optimal path survives the
    # bound at v_cap and would not survive it at v_flat, so a slower speed in
    # the bound is an unsound pruning this fixture can catch.
    assert 0.5 + 45.0 / v_cap <= iv.horizon_h
    assert 0.5 + 45.0 / (v_cap / 1.5) > iv.horizon_h


def test_shadow_quantisation_is_upward_and_pessimistic():
    """Protects the direction of the approximation: it may lose plans, not invent.

    Quantisation cannot bite on a single edge when the tolerance is a multiple
    of the grid, which every registered tolerance is of every registered grid:
    ceil(s/g)*g <= tol is then the same condition as s <= tol. It bites by
    accumulating, because the overstated shadow at arrival is carried into the
    next traverse. So the fixture is two dark edges, each accruing 0.9 h.

    Exact accrues 1.8 h. A 900 s grid rounds the first arrival to 1.0 and the
    second to 2.0. At a 1.9 h tolerance that loses a feasible plan, which is
    the cost. At a 1.7 h tolerance nothing is feasible either way, and rounding
    down would make it feasible, which is what makes downward rounding
    catchable here.
    """
    g = Graph(np.array([0, 1, 2, 2], dtype=np.int64),
              np.array([1, 2], dtype=np.int64), np.array([0.9, 0.9]), 3)
    iv = world({}, [False, False, False], horizon=8.0)
    fine, coarse = 1.0 / 3600.0, 900.0 / 3600.0

    assert plan(g, iv, 0, 2, 0.0, 1.9)[0] == pytest.approx(1.8)
    assert plan(g, iv, 0, 2, 0.0, 1.9, shadow_grid_h=fine)[0] == pytest.approx(1.8), \
        "a 1 s grid must reproduce the exact answer"
    assert plan(g, iv, 0, 2, 0.0, 1.9, shadow_grid_h=coarse)[0] is None, \
        "a 900 s grid must lose this plan, or the fixture does not bite"

    # The safe direction, on a tolerance neither engine can satisfy: an upward
    # grid must not invent a plan. Rounding down would.
    assert plan(g, iv, 0, 2, 0.0, 1.7)[0] is None
    for grid in (1.0, 60.0, 300.0, 900.0):
        assert plan(g, iv, 0, 2, 0.0, 1.7, shadow_grid_h=grid / 3600.0)[0] is None, grid

    # The maximum is quantised as well as the value at arrival, and it has to
    # be: on a traverse that dips into light at the end the maximum exceeds the
    # arrival value, and rounding only the arrival value down would accept a
    # plan whose worst moment breaks the tolerance.
    g2 = Graph(np.array([0, 1, 1], dtype=np.int64), np.array([1], dtype=np.int64),
               np.array([2.0]), 2)
    iv2 = world({0: [1.9], 1: [1.9]}, [False, False], horizon=8.0)
    assert plan(g2, iv2, 0, 1, 0.0, 1.8)[0] is None, "exact: the maximum is 1.9"
    for grid in (1.0, 60.0, 300.0, 900.0):
        assert plan(g2, iv2, 0, 1, 0.0, 1.8, shadow_grid_h=grid / 3600.0)[0] is None, grid

    # And generally: quantised feasible implies exact feasible.
    for tol in (1.5, 1.7, 1.8, 1.9, 2.0, 2.5):
        e = plan(g, iv, 0, 2, 0.0, tol)[0]
        for grid in (60.0, 900.0):
            q = plan(g, iv, 0, 2, 0.0, tol, shadow_grid_h=grid / 3600.0)[0]
            if q is not None:
                assert e is not None, (tol, grid)


def test_relaxed_dominance_only_loses_plans():
    """Protects the direction of the other approximation, on the case that breaks it.

    The reachability counterexample exists because an earlier label with no
    more shadow can be worse. Dropping the clause prunes the label carrying the
    only feasible continuation, so the relaxed engine reports no path where the
    exact one arrives at 2.1 h. That is the cost, and the test asserts it goes
    that way round and not the other.
    """
    iv = world(COUNTER_TRANS, COUNTER_LIT0)
    exact = plan(COUNTER_GRAPH, iv, 0, 3, 0.0, COUNTER_TOL)[0]
    relaxed = plan(COUNTER_GRAPH, iv, 0, 3, 0.0, COUNTER_TOL, relax_dominance=True)[0]
    assert exact == pytest.approx(2.1, abs=1e-9)
    assert relaxed is None, relaxed
    # never the other way: relaxed feasible implies exact feasible, and no
    # earlier arrival
    for tol in (0.5, 1.0, 2.0, 4.0):
        e = plan(COUNTER_GRAPH, iv, 0, 3, 0.0, tol)[0]
        r = plan(COUNTER_GRAPH, iv, 0, 3, 0.0, tol, relax_dominance=True)[0]
        if r is not None:
            assert e is not None and e <= r + 1e-12, (tol, e, r)


def test_frozen_intervals_never_change_state():
    """Protects the snapshot planner's world: frozen means frozen, not sampled."""
    from src.baselines import frozen_intervals
    iv = world({0: [1.0, 3.0], 1: [2.0]}, [True, False], horizon=9.0)
    fz = frozen_intervals(iv, 0.0)
    assert fz.times.size == 0
    assert fz.is_lit(0, 0.0) and not fz.is_lit(1, 0.0)
    for t in (0.0, 1.5, 2.5, 8.0):
        assert fz.is_lit(0, t) is True and fz.is_lit(1, t) is False
    # frozen at a different instant freezes a different world, which is the
    # whole content of "frozen at departure"
    fz2 = frozen_intervals(iv, 2.5)
    assert not fz2.is_lit(0, 0.0) and fz2.is_lit(1, 0.0)


def test_snapshot_cannot_wait_for_light_that_it_cannot_see():
    """Protects the baseline being the right kind of wrong, not simply crippled.

    The time-dependent planner waits at the start until the route lights up and
    arrives. The snapshot planner runs the same search on a world where nothing
    ever lights up, so no amount of waiting helps it and it finds no plan. Both
    use the same engine and the same constraint; only the illumination differs.
    """
    from src.baselines import plan_snapshot
    g = Graph(np.array([0, 1, 1], dtype=np.int64), np.array([1], dtype=np.int64),
              np.array([0.3]), 2)
    iv = world({1: [2.0]}, [True, False], horizon=9.0)
    assert plan(g, iv, 0, 1, 0.0, 0.1)[0] == pytest.approx(2.3), "waits for 2.0"
    assert plan_snapshot(g, iv, 0, 1, 0.0, 0.1)[0] is None
    # and on a world that never changes, the two must agree exactly
    static = world({}, [True, True], horizon=9.0)
    assert plan(g, static, 0, 1, 0.0, 0.1)[0] == \
        plan_snapshot(g, static, 0, 1, 0.0, 0.1)[0] == pytest.approx(0.3)


def test_mean_map_routes_around_cells_that_are_dark_more_often():
    """Protects the baseline's one illumination input actually steering it.

    Two routes to the goal: a short one through a cell lit 5% of the horizon and
    a long one through cells lit 95%. At exponent 0 the planner is a pure
    fastest-path planner and takes the short route; as the exponent rises it
    must switch. A baseline whose weight does not change its route is a
    strawman, and this asserts it is not one.
    """
    from src.baselines import plan_mean_map
    # 0 -> 1 -> 3 is short; 0 -> 2 -> 3 is long. Cell 1 is rarely lit.
    indptr = np.array([0, 2, 3, 4, 4], dtype=np.int64)
    indices = np.array([1, 2, 3, 3], dtype=np.int64)
    travel = np.array([1.0, 2.0, 1.0, 2.0])
    g = Graph(indptr, indices, travel, 4)
    lit = np.array([1.0, 0.05, 0.95, 1.0])
    t0, p0 = plan_mean_map(g, lit, 0, 3, exponent=0.0)
    assert p0 == [0, 1, 3] and t0 == pytest.approx(2.0)
    t2, p2 = plan_mean_map(g, lit, 0, 3, exponent=1.0)
    assert p2 == [0, 2, 3], p2
    assert t2 == pytest.approx(4.0), "the reported time is the real traverse time"
    assert plan_mean_map(g, lit, 3, 0, exponent=1.0) == (None, [])
