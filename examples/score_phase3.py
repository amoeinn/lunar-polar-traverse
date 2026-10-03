"""Score P2, P3, P4 and P7 from the sweep's own outputs.

Runs no planner. Every planner result here is read back from the core sweep's
and the axes' work directories, and every replay verdict is either read from
disk or recomputed by the same committed checker into a directory keyed on the
point, the epoch, the planner and the tolerance label.

What it recomputes rather than reads: the core sweep replayed all four mean-map
exponents into one shared directory, so only the last of them survived on disk.
The mean-map is scored at the exponent the registered criterion selected on that
run, 2.0 on the core, and its routes and replay are rebuilt here. Its pass count
reproducing the recorded 197 is the check that the reconstruction is the same
run.

Every case count is printed beside every median, aborts are a third outcome
everywhere, and the monotonicity bound on the tolerance axis is applied rather
than assumed: feasibility is monotone in the shadow tolerance, so a point whose
resolved feasible count is below a smaller tolerance's is reported as censored
and is not scored.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import examples.plan_pilot as P                                  # noqa: E402
import examples.sweep_axes as AX                                 # noqa: E402
import examples.sweep_core as CO                                 # noqa: E402
from examples.build_intervals import cache_path, epochs          # noqa: E402
from examples.replay_batch import FINE_SUN_DT_H, fine_sun        # noqa: E402
from src.baselines import plan_mean_map                          # noqa: E402
from src.cost import Rover, legal                                # noqa: E402
from src.frames import cell_position                             # noqa: E402
from src.intervals import LitIntervals                           # noqa: E402
from src.planner import Graph                                    # noqa: E402
from src.replay import ReplaySettings, replay                    # noqa: E402

TOL_DEG = 3.206                 # registered pitch and roll tolerance
CORE_BEST_EXPONENT = 2.0        # selected by the registered criterion on the core
CORE_RECORDED_MM_PASS = 197     # what that run reported, as a reconstruction check
P2_MIN_HOURS = 24.0             # P2's own threshold
P2_MIN_INFEASIBLE = 0.30
P3_MIN_MARGIN = 0.10
P3_MIN_SHARE = 0.25
P4_MAX_HOURS = 6.0
P4_MAX_DIFF = 0.02


def q(x):
    return "n/a" if not len(x) else f"{np.median(x):.4g}"


def plans_from_planner(result, pairs, graph):
    """The same reconstruction the sweep drivers do: path, start, per-step waits."""
    off, pc, pt = result["path_offsets"], result["path_cells"], result["path_times"]
    plans, starts, waits, idx = [], [], [], []
    for i in range(len(pairs)):
        lo, hi = off[i], off[i + 1]
        if hi - lo < 2:
            continue
        cpath, tpath = pc[lo:hi].astype(np.int64), pt[lo:hi]
        w = np.zeros(cpath.size)
        for j in range(cpath.size - 1):
            u, v = int(cpath[j]), int(cpath[j + 1])
            s0, s1 = graph.indptr[u], graph.indptr[u + 1]
            k = np.flatnonzero(graph.indices[s0:s1] == v)
            w[j] = max(tpath[j + 1] - float(graph.travel[s0:s1][k[0]]) - tpath[j], 0.0)
        plans.append(cpath)
        starts.append(float(tpath[0]))
        waits.append(w)
        idx.append(i)
    return plans, starts, waits, idx


def judge(plans, starts, waits, grid, rover, suns, shadow_tol, work):
    """Both registered verdicts, from disk when the checker has already run."""
    out = {}
    for label, tol in (("tol", TOL_DEG), ("zero", 0.0)):
        out[label] = replay(plans, starts, grid, rover, suns, FINE_SUN_DT_H,
                            shadow_tol, Path(f"{work}_{label}"),
                            ReplaySettings(limit_tolerance_deg=tol), waits=waits,
                            reuse=True)
    return out


class Site:
    """The region, the graph and the geometry every point shares."""

    def __init__(self):
        self.rover0 = Rover(v_flat=50.0)
        (self.p, self.n, self.px, self.dem, self.horizon, self.cells,
         self.gx, self.gy, self.edges, _) = P.load(self.rover0)
        p = self.p
        self.xs, self.ys = self.gx.ravel(), self.gy.ravel()
        self.positions = cell_position(self.gx, self.gy, self.dem,
                                       p["radius_m"]).reshape(-1, 3)
        self.grid = dict(
            radius_m=p["radius_m"], region_ox=p["region_ox"], region_oy=p["region_oy"],
            region_px=p["region_px"], region_nx=p["region_nx"],
            near_path=p["near_path"], near_ox=p["near_ox"], near_oy=p["near_oy"],
            near_px=p["near_px"], near_width=p["near_width"],
            near_height=p["near_height"], far_path=p["far_path"],
            far_ox=p["far_ox"], far_oy=p["far_oy"], far_px=p["far_px"],
            far_width=p["far_width"], far_height=p["far_height"],
            reach_m=p["reach_m"], switch_m=p["switch_m"],
            near_step_m=p["near_step_m"], far_step_m=p["far_step_m"])
        self._iv, self._lf, self._graph, self._pool, self._sun = {}, {}, {}, {}, {}

    def iv(self, tag):
        if tag not in self._iv:
            self._iv[tag] = LitIntervals.load(cache_path(tag))
        return self._iv[tag]

    def lf(self, tag):
        if tag not in self._lf:
            self._lf[tag] = self.iv(tag).lit_fraction()
        return self._lf[tag]

    def graph(self, rover):
        key = (rover.pitch_max, rover.roll_max, rover.v_flat)
        if key not in self._graph:
            self._graph[key] = Graph.build(self.edges, rover, self.n * self.n)
        return self._graph[key]

    def pool(self, rover, tag):
        """The draw pool: the largest component of edges whose cells change state."""
        key = (rover.pitch_max, rover.roll_max, tag)
        if key not in self._pool:
            ok = legal(self.edges["pitch"], self.edges["roll"], rover)
            changing = np.diff(self.iv(tag).offsets) >= 2
            sel = ok & changing[self.edges["a"]] & changing[self.edges["b"]]
            gc = coo_matrix((np.ones(int(sel.sum())),
                             (self.edges["a"][sel], self.edges["b"][sel])),
                            shape=(self.n * self.n,) * 2)
            _, lab = connected_components(gc, directed=False)
            self._pool[key] = np.flatnonzero(lab == np.argsort(np.bincount(lab))[-1])
        return self._pool[key]

    def sun(self, tag):
        if tag not in self._sun:
            self._sun[tag] = fine_sun(tag)
        return self._sun[tag]


def collect(site, work_dir, stem, tag, rover, shadow_tol, pairs_label, draw_fn,
            exponents):
    """Every planner's per-instance record at one configuration and epoch.

    Reads the planner outputs the sweep left on disk, checks the draw it ran on
    against the registered rule, rebuilds the mean-map routes, and judges every
    plan twice.
    """
    graph = site.graph(rover)
    out_prefix = work_dir / f"{stem}{tag}_replay"
    rec = {}
    pairs = None
    for kind in ("td", "snap"):
        d = work_dir / f"{stem}{tag}_{kind}"
        params = json.loads((d / "planner_params.json").read_text())
        on_disk = np.fromfile(d / "pairs.bin", dtype=np.int32).reshape(-1, 2)
        if pairs is None:
            pairs = on_disk
            drawn = np.asarray(draw_fn(site.pool(rover, tag), site.xs, site.ys,
                                       pairs_label), dtype=np.int32)
            assert np.array_equal(drawn, pairs), f"draw mismatch at {tag}"
        else:
            assert np.array_equal(on_disk, pairs), f"pairs differ between planners"
        r = P.read_cpp(params, want_paths=True)
        plans, starts, waits, idx = plans_from_planner(r, pairs, graph)
        j = judge(plans, starts, waits, site.grid, rover, site.sun(tag), shadow_tol,
                  f"{out_prefix}_{kind}") if plans else None
        rec[kind] = dict(n=len(pairs), aborted=r["aborted"],
                         arrival=r["arrival"], idx=idx, plans=plans, judged=j)
    lf = site.lf(tag)
    for e in exponents:
        routes = [plan_mean_map(graph, lf, int(a), int(b), exponent=e) for a, b in pairs]
        plans, starts, waits, idx, claimed = [], [], [], [], []
        for i, (tt, path) in enumerate(routes):
            if not path or len(path) < 2:
                continue
            plans.append(np.asarray(path, dtype=np.int64))
            starts.append(0.0)
            waits.append(np.zeros(len(path)))
            idx.append(i)
            claimed.append(tt)
        j = judge(plans, starts, waits, site.grid, rover, site.sun(tag), shadow_tol,
                  f"{out_prefix}_mm{e}") if plans else None
        rec[f"mm{e}"] = dict(n=len(pairs), aborted=np.zeros(len(pairs), bool),
                             arrival=np.array([np.nan] * len(pairs)), idx=idx,
                             plans=plans, judged=j)
        for i, tt in zip(idx, claimed):
            rec[f"mm{e}"]["arrival"][i] = tt
    return rec


def rows(rec, kind):
    """Flatten one planner's record to per-instance rows."""
    d = rec[kind]
    if d["judged"] is None:
        return []
    a, z = d["judged"]["tol"], d["judged"]["zero"]
    out = []
    for i, inst in enumerate(d["idx"]):
        out.append(dict(inst=int(inst), claimed=float(d["arrival"][inst]),
                        checked=float(a["arrival"][i]), pass_tol=bool(a["verdict"][i]),
                        pass_zero=bool(z["verdict"][i]), shadow=float(a["shadow"][i]),
                        cells=len(d["plans"][i])))
    return out


def outcome(rec, kind):
    d = rec[kind]
    ab = d["aborted"]
    fin = np.isfinite(d["arrival"])
    return (int((~ab & fin).sum()), int((~ab & ~fin).sum()), int(ab.sum()))


def all_rows(recs, kind):
    out = []
    for tag, rec in recs.items():
        for r in rows(rec, kind):
            out.append(dict(r, tag=tag))
    return out


def best_exponent(recs, exponents):
    """The baseline at its best setting by replay feasibility, as registered."""
    counts = {}
    for e in exponents:
        counts[e] = sum(int(r["pass_tol"]) for r in all_rows(recs, f"mm{e}"))
    best = max(counts, key=lambda e: counts[e])
    return best, counts


def score_p2(name, recs, duration_key="claimed"):
    """P2: snapshot plans over 24 h, infeasible share and its trend in duration."""
    snap = all_rows(recs, "snap")
    print(f"\n--- P2 at {name} ---")
    f, i, a = (sum(x) for x in zip(*[outcome(r, "snap") for r in recs.values()]))
    print(f"  snapshot planner: {f} feasible, {i} infeasible, {a} aborted "
          f"of {f+i+a}; {len(snap)} plans replayed")
    elig = [r for r in snap if r[duration_key] > P2_MIN_HOURS]
    print(f"  cases whose traverse exceeds {P2_MIN_HOURS:.0f} h: {len(elig)} "
          f"of {len(snap)} plans (traverse from the planner's own arrival)")
    if not elig:
        print("  clause 1: VACUOUS, no eligible case")
        return
    bad = [r for r in elig if not r["pass_tol"]]
    bad0 = [r for r in elig if not r["pass_zero"]]
    frac = len(bad) / len(elig)
    print(f"  replayed infeasible: {len(bad)}/{len(elig)} = {frac:.1%} at the "
          f"registered tolerance, {len(bad0)}/{len(elig)} = "
          f"{len(bad0)/len(elig):.1%} at zero")
    print(f"  clause 1 ({P2_MIN_INFEASIBLE:.0%} or more): "
          f"{'PASS' if frac >= P2_MIN_INFEASIBLE else 'FAIL'}")
    # The planner's arrival understates the traverse by about 1%, so the 24 h
    # cut moves a little under the checker's integration. How much is reported
    # rather than assumed negligible.
    elig_c = [r for r in snap if r["checked"] > P2_MIN_HOURS]
    bad_c = [r for r in elig_c if not r["pass_tol"]]
    if elig_c:
        print(f"  same cut on the checker's integrated traverse: "
              f"{len(elig_c)} eligible, {len(bad_c)} infeasible = "
              f"{len(bad_c)/len(elig_c):.1%}")
    d = np.array([r[duration_key] for r in elig])
    bad_i = np.array([0.0 if r["pass_tol"] else 1.0 for r in elig])
    cuts = np.quantile(d, [0.25, 0.5, 0.75])
    which = np.searchsorted(cuts, d, side="right")
    print(f"  quartile cuts of traverse duration: "
          + ", ".join(f"{c:.1f} h" for c in cuts))
    fr = []
    for b in range(4):
        m = which == b
        fr.append(float(bad_i[m].mean()) if m.any() else float("nan"))
        lo = d[m].min() if m.any() else float("nan")
        hi = d[m].max() if m.any() else float("nan")
        print(f"    Q{b+1}: {int(m.sum()):3d} cases, {lo:8.1f} to {hi:8.1f} h, "
              f"infeasible {bad_i[m].sum():.0f} = "
              f"{fr[-1] if m.any() else float('nan'):.1%}")
    ok = all(not np.isnan(x) for x in fr) and all(
        fr[k] <= fr[k + 1] + 1e-12 for k in range(3))
    rho, pv = spearmanr(d, bad_i) if len(set(bad_i)) > 1 else (float("nan"),) * 2
    print(f"  rank correlation of duration with infeasibility: rho {rho:.4f}, "
          f"p {pv:.3g} (reported, not part of the criterion)")
    print(f"  clause 2 (non-decreasing across quartiles): "
          f"{'PASS' if ok else 'FAIL'}")
    print(f"  P2 at {name}: {'PASS' if (frac >= P2_MIN_INFEASIBLE and ok) else 'FAIL'}"
          f" (both clauses required)")


def score_p3_p4(recs, exponents):
    best, counts = best_exponent(recs, exponents)
    snap = {(r["tag"], r["inst"]): r for r in all_rows(recs, "snap")}
    td = {(r["tag"], r["inst"]): r for r in all_rows(recs, "td")}
    mm = {(r["tag"], r["inst"]): r for r in all_rows(recs, f"mm{best}")}
    snap_pass = sum(int(r["pass_tol"]) for r in snap.values())
    mm_pass = sum(int(r["pass_tol"]) for r in mm.values())
    print("\n--- P3 ---")
    print(f"  mean-map exponents by replay pass count: "
          + ", ".join(f"e={e}: {counts[e]}" for e in exponents)
          + f"; registered criterion selects {best}")
    if mm_pass != CORE_RECORDED_MM_PASS:
        print(f"  NOTE: reconstruction gives {mm_pass} passing mean-map plans "
              f"against the recorded {CORE_RECORDED_MM_PASS}")
    else:
        print(f"  reconstruction check: {mm_pass} passing mean-map plans, equal to "
              f"the recorded core run")
    n_inst = sum(rec["td"]["n"] for rec in recs.values())
    snap_z = sum(int(r["pass_zero"]) for r in snap.values())
    mm_z = sum(int(r["pass_tol"] and r["pass_zero"]) for r in mm.values())
    print(f"  clause 1, feasible more often than snapshot, over the same "
          f"{n_inst} instances:")
    print(f"    mean-map {mm_pass} feasible of {n_inst} instances "
          f"({len(mm)} plans produced, {mm_pass/len(mm):.1%} of them feasible)")
    print(f"    snapshot {snap_pass} feasible of {n_inst} instances "
          f"({len(snap)} plans produced, {snap_pass/len(snap):.1%} of them feasible)")
    print(f"    at zero tolerance: mean-map {mm_z}, snapshot {snap_z}")
    print(f"    clause 1: {'PASS' if mm_pass > snap_pass else 'FAIL'} at the "
          f"registered tolerance, "
          f"{'PASS' if mm_z > snap_z else 'FAIL'} at zero")
    both = [(k, td[k], mm[k]) for k in td if k in mm
            and td[k]["pass_tol"] and mm[k]["pass_tol"]]
    print(f"  clause 2 set, both the mean-map and the time-dependent plan "
          f"feasible: {len(both)} cases")
    if not both:
        print("  clause 2: VACUOUS on 0 cases")
        p3 = None
    else:
        for tagname, key in (("planner's own arrivals", "claimed"),
                             ("checker's integrated arrivals", "checked")):
            imp = np.array([(m[key] - t[key]) / m[key] for _, t, m in both])
            share = float((imp >= P3_MIN_MARGIN).mean())
            print(f"    on {tagname}: median improvement {np.median(imp):+.2%}, "
                  f"{int((imp >= P3_MIN_MARGIN).sum())}/{len(imp)} = {share:.1%} "
                  f"at or above {P3_MIN_MARGIN:.0%}; "
                  f"{'PASS' if share >= P3_MIN_SHARE else 'FAIL'}")
            if key == "claimed":
                p3 = share >= P3_MIN_SHARE
    print(f"  P3: {'PASS' if (mm_pass > snap_pass and p3) else 'FAIL'}"
          f" (both clauses required)")

    print("\n--- P4 ---")
    short = [(k, t, m) for k, t, m in both if t["claimed"] < P4_MAX_HOURS]
    both_short = [x for x in short if x[2]["claimed"] < P4_MAX_HOURS]
    mm_short = [(k, t, m) for k, t, m in both if m["claimed"] < P4_MAX_HOURS]
    print(f"  of the {len(both)} cases where both planners are feasible, "
          f"traverses under {P4_MAX_HOURS:.0f} h: {len(short)} by the "
          f"time-dependent arrival, {len(mm_short)} by the mean-map's, "
          f"{len(both_short)} by both. Scored on the first, which is the "
          f"method's own output.")
    if not short:
        print(f"  P4: VACUOUS on 0 cases, not scoreable")
        return
    cells = np.array([t["cells"] for _, t, _ in short])
    print(f"  median path length {np.median(cells):.0f} cells "
          f"({np.median(cells)*20:.0f} m), reported as the registered vacuity "
          f"check asks")
    for tagname, key in (("planner's own arrivals", "claimed"),
                         ("checker's integrated arrivals", "checked")):
        diff = np.array([abs(t[key] - m[key]) / m[key] for _, t, m in short])
        med = float(np.median(diff))
        print(f"    on {tagname}: median absolute difference {med:.2%} over "
              f"{len(diff)} cases, max {diff.max():.2%}; "
              f"{'PASS' if med < P4_MAX_DIFF else 'FAIL'}")
        if key == "claimed":
            print(f"  P4: {'PASS' if med < P4_MAX_DIFF else 'FAIL'}")


def feasible_set(recs, kind):
    """Instances the planner resolved as feasible, and the ones it aborted on."""
    feas, ab = set(), set()
    for tag, rec in recs.items():
        d = rec[kind]
        for i in range(d["n"]):
            if d["aborted"][i]:
                ab.add((tag, i))
            elif np.isfinite(d["arrival"][i]):
                feas.add((tag, i))
    return feas, ab


def score_p7(rate_recs, exponents):
    """P7: the advantage over the mean-map falls with rover speed, on the
    intersection of cases feasible at all four rates, as registered."""
    rates = sorted(rate_recs)
    print("\n--- P7 ---")
    print(f"  {'rate':>8}{'mm e*':>7}{'feasible':>10}{'infeas':>8}{'aborted':>9}"
          f"{'pass@tol':>10}{'pass@0':>8}")
    best, pass_sets, plan_sets = {}, {}, {}
    for r in rates:
        recs = rate_recs[r]
        best[r] = best_exponent(recs, exponents)[0]
        f, i, a = (sum(x) for x in zip(*[outcome(rec, "td") for rec in recs.values()]))
        plan_sets[r], _ = feasible_set(recs, "td")
        pass_sets[r] = {(x["tag"], x["inst"]) for x in all_rows(recs, "td")
                        if x["pass_tol"]}
        zero = sum(int(x["pass_zero"]) for x in all_rows(recs, "td"))
        print(f"  {r:>8.0f}{best[r]:>7}{f:>10}{i:>8}{a:>9}"
              f"{len(pass_sets[r]):>10}{zero:>8}")
    inter_pass = set.intersection(*[pass_sets[r] for r in rates])
    inter_plan = set.intersection(*[plan_sets[r] for r in rates])
    print(f"  intersection across all four rates: {len(inter_pass)} cases by replay "
          f"pass, {len(inter_plan)} by planner feasibility")
    print(f"  cost of the intersection: per-rate replay-pass sets of "
          + ", ".join(f"{len(pass_sets[r])}" for r in rates)
          + f" collapse to {len(inter_pass)}")
    for setname, inter in (("replay-pass intersection", inter_pass),
                           ("planner-feasible intersection", inter_plan)):
        print(f"  on the {setname}, {len(inter)} cases:")
        if not inter:
            print("    not scoreable: the intersection is empty")
            continue
        meds = {}
        for r in rates:
            td = {(x["tag"], x["inst"]): x for x in all_rows(rate_recs[r], "td")}
            mm = {(x["tag"], x["inst"]): x
                  for x in all_rows(rate_recs[r], f"mm{best[r]}")}
            imp = np.array([(mm[k]["claimed"] - td[k]["claimed"]) / mm[k]["claimed"]
                            for k in sorted(inter) if k in td and k in mm])
            meds[r] = float(np.median(imp)) if len(imp) else float("nan")
            same = int((np.abs(imp) < 1e-12).sum())
            print(f"    {r:>5.0f} m/hr: {len(imp):3d} cases, median improvement "
                  f"{meds[r]:+.2%} (exactly {meds[r]:+.6e}), quartiles "
                  f"{np.quantile(imp,0.25):+.2%} to {np.quantile(imp,0.75):+.2%}, "
                  f"max {imp.max():+.2%}; {same} of {len(imp)} arrive at the same "
                  f"time to within 1e-12")
        order = [meds[r] for r in rates]
        # Two readings of "monotone decreasing", and the registered word is
        # decreasing. Strict requires every step to fall; non-increasing allows
        # ties. They differ here only because three medians are exactly zero,
        # so the distinction is reported rather than resolved silently.
        strict = all(order[k] > order[k + 1] for k in range(len(order) - 1))
        weak = all(order[k] >= order[k + 1] - 1e-12 for k in range(len(order) - 1))
        ends = order[0] > order[-1]
        print(f"    clause 1, 20 m/hr above 200 m/hr: {'PASS' if ends else 'FAIL'}")
        print(f"    clause 2, monotone decreasing, strict reading: "
              f"{'PASS' if strict else 'FAIL'}; non-increasing reading: "
              f"{'PASS' if weak else 'FAIL'}")
        print(f"    P7 on this set: {'PASS' if (strict and ends) else 'FAIL'} at the "
              f"strict reading, {'PASS' if (weak and ends) else 'FAIL'} at the "
              f"non-increasing reading")
        # The registered criterion gives the baseline its best exponent at each
        # point, and that setting is not the same at every rate here, so part of
        # any rate-to-rate movement could be the baseline changing rather than
        # the rover's speed. Holding the exponent fixed separates the two.
        print(f"    with the baseline's exponent held fixed instead of set per "
              f"point:")
        for e in exponents:
            row = []
            for r in rates:
                td = {(x["tag"], x["inst"]): x for x in all_rows(rate_recs[r], "td")}
                mm = {(x["tag"], x["inst"]): x
                      for x in all_rows(rate_recs[r], f"mm{e}")}
                v = np.array([(mm[k]["claimed"] - td[k]["claimed"]) / mm[k]["claimed"]
                              for k in sorted(inter) if k in td and k in mm])
                row.append(float(np.median(v)) if len(v) else float("nan"))
            st = all(row[k] > row[k + 1] for k in range(len(row) - 1))
            wk = all(row[k] >= row[k + 1] - 1e-12 for k in range(len(row) - 1))
            print(f"      e={e}: " + ", ".join(f"{x:+.4%}" for x in row)
                  + f"  strict {'PASS' if st else 'FAIL'}, "
                  f"non-increasing {'PASS' if wk else 'FAIL'}")


def censoring_report(site, stem, point, base):
    """Why a tolerance point is not scoreable, with the monotonicity bound.

    Feasibility is monotone in the shadow tolerance: a plan whose maximum
    continuous shadow is inside T is inside every larger T on the same instance.
    So a larger tolerance cannot have fewer feasible cases than a smaller one,
    and whatever the resolved set falls short by must be hidden in the aborts.
    """
    print(f"\n--- P2 at the tolerance axis, {point['label']} ---")
    tot = dict(f=0, i=0, a=0)
    for tag in AX.EPOCHS:
        d = P.DATA / "work" / "axes" / f"{stem}{tag}_snap"
        params = json.loads((d / "planner_params.json").read_text())
        r = P.read_cpp(params)
        ab = r["aborted"]
        tot["f"] += int((~ab & np.isfinite(r["arrival"])).sum())
        tot["i"] += int((~ab & ~np.isfinite(r["arrival"])).sum())
        tot["a"] += int(ab.sum())
    n = tot["f"] + tot["i"] + tot["a"]
    print(f"  snapshot planner: {tot['f']} feasible, {tot['i']} infeasible, "
          f"{tot['a']} aborted of {n}")
    if base is not None:
        hidden = max(0, base["f"] - tot["f"])
        print(f"  monotonicity bound from the 0.5 h point's {base['f']} feasible: "
              f"at least {hidden} feasible cases are hidden in the "
              f"{tot['a']} aborts, so the resolved set is not the whole set")
    print(f"  NOT SCOREABLE, as the registered fallback states: a prediction "
          f"about infeasibility rates cannot be scored on a table whose "
          f"infeasible column is {tot['i']} of {n} with {tot['a']} unresolved")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parts", default="core,p2axis,p7")
    args = ap.parse_args()
    parts = args.parts.split(",")
    site = Site()
    core_tags = [t for t, _, _ in epochs()]
    exps = CO.EXPONENTS

    if "core" in parts:
        rover = Rover(v_flat=50.0)
        recs = {}
        for ei, tag in enumerate(core_tags):
            recs[tag] = collect(site, P.DATA / "work" / "sweep", "", tag, rover,
                                CO.TOLERANCE_H, ei, CO.draw, (CORE_BEST_EXPONENT,))
            print(f"  core {tag}: td {outcome(recs[tag], 'td')}, "
                  f"snap {outcome(recs[tag], 'snap')}", flush=True)
        score_p2("the core, tolerance 2 h, 1440 instances", recs)
        score_p3_p4(recs, (CORE_BEST_EXPONENT,))

    if "p2axis" in parts:
        # The registered fallback scores P2 at 0.5 and 2 h only. The 0.5 h point
        # is scored here; 6 and 24 h get the censoring report the fallback asks
        # for, with the monotonicity bound that says how many feasible cases the
        # aborted sets must contain.
        base = None
        for pi, point in enumerate(AX.AXES["tolerance"]):
            rover = Rover(v_flat=point["v_flat"], pitch_max=point["pitch_max"],
                          roll_max=point["roll_max"])
            if point["tol"] <= 0.5:
                recs = {}
                for ei, tag in enumerate(AX.EPOCHS):
                    recs[tag] = collect(site, P.DATA / "work" / "axes",
                                        f"tolerance_{pi}_", tag, rover,
                                        point["tol"], ei, AX.draw, ())
                base = {k: sum(x) for k, x in
                        zip(("f", "i", "a"),
                            zip(*[outcome(r, "snap") for r in recs.values()]))}
                score_p2(f"the tolerance axis, {point['label']}, 160 instances",
                         recs)
            else:
                censoring_report(site, f"tolerance_{pi}_", point, base)

    if "p7" in parts:
        rate_recs = {}
        for axis, pi, rate in (("rate_core", 0, 50.0), ("rate", 0, 20.0),
                               ("rate", 1, 100.0), ("rate", 2, 200.0)):
            rover = Rover(v_flat=rate, pitch_max=20.0, roll_max=15.0)
            recs = {}
            for ei, tag in enumerate(AX.EPOCHS):
                recs[tag] = collect(site, P.DATA / "work" / "axes", f"{axis}_{pi}_",
                                    tag, rover, 2.0, ei, AX.draw, exps)
            rate_recs[rate] = recs
            print(f"  rate {rate:.0f} collected", flush=True)
        score_p7(rate_recs, exps)
    return 0


if __name__ == "__main__":
    sys.exit(main())
