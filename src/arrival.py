"""The checker's arrival against the planner's, with its sign.

The quantity is (checker - planner) / planner per plan: positive when the
checker's continuous integration arrives later than the planner's summed edge
times, negative when it arrives earlier. Every plan starts at t = 0 here, so
relative to arrival and relative to traverse duration are the same number.

The first batch computed abs(checker - planner) / planner and reported its
median, 1.169%. The record then carried "+1.169%, checker slower" as though the
sign had been measured; it was the convexity prediction written beside a
magnitude. A signed measurement later found the checker earlier on most plans,
and the cause was a checker defect. So the sign is what this module returns,
and a magnitude, where one is wanted, is taken from it explicitly and labelled
as carrying no direction.

The registered arrival criterion (2026-10-02) is two sided: the checker's
arrival must agree with the planner's to within 1%, in either direction.
"""
import numpy as np

ARRIVAL_CRITERION = 0.01        # registered 2026-10-02


def arrival_gap(checker, planner):
    """(checker - planner) / planner, signed. Positive: the checker is later."""
    c = np.asarray(checker, dtype=float)
    p = np.asarray(planner, dtype=float)
    return (c - p) / np.maximum(p, 1e-9)


def within_criterion(gap, tol=ARRIVAL_CRITERION):
    """Plans whose checker arrival is within tol of the planner's, either way."""
    return np.abs(np.asarray(gap, dtype=float)) <= tol


def summarise(gap, tol=ARRIVAL_CRITERION):
    """Counts by direction, the signed median and spread, and the criterion.

    median_magnitude is the median of |gap| and is the only figure here without
    a direction; it is what the first batch reported.
    """
    g = np.asarray(gap, dtype=float)
    if g.size == 0:
        return dict(n=0, later=0, earlier=0, equal=0, within=0, median=None,
                    p10=None, p90=None, median_magnitude=None)
    return dict(n=int(g.size),
                later=int((g > 0).sum()),
                earlier=int((g < 0).sum()),
                equal=int((g == 0).sum()),
                within=int(within_criterion(g, tol).sum()),
                median=float(np.median(g)),
                p10=float(np.percentile(g, 10)),
                p90=float(np.percentile(g, 90)),
                median_magnitude=float(np.median(np.abs(g))))


def describe(s):
    """One line for a printed report, the sign stated in words."""
    if s["n"] == 0:
        return "no plans"
    return (f"n {s['n']}: checker later on {s['later']}, earlier on {s['earlier']}, "
            f"equal on {s['equal']}; signed median {s['median']:+.4%} "
            f"(p10 {s['p10']:+.4%}, p90 {s['p90']:+.4%}); within 1% on "
            f"{s['within']} of {s['n']}; median magnitude, no sign, "
            f"{s['median_magnitude']:.4%}")
