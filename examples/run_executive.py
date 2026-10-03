"""Run the executive over the registered sample and score it against the auditor.

PLAN.md, "The executive", registers the sample, the tick definition, the
replan-at-a-lit-cell constraint, k, the declaration-audit cap d, the three bars
and the five failure conditions, all before this ran.

Monitored set: every core feasible time-dependent plan whose integrated
continuous shadow is above zero, 45 plans. Control set: every one whose shadow
is exactly zero, 142 plans. The executive must fire on the first where
continuing busts and must never fire on the second.
"""
import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from examples.build_intervals import epochs                      # noqa: E402
from src import auditor as A                                     # noqa: E402
from src.executive import run_instance, simulate, plan_waits      # noqa: E402

WORK = ROOT / "data" / "work" / "sweep"
OUT = ROOT / "data" / "work" / "executive"
BUDGET_H = 2.0
REPLAN_LIMIT = 2            # k, registered
REPLAN_CAP_S = 30.0         # the cap the search runs under everywhere else
DECL_CAP = 6                # d, registered
DECL_CAP_S = 300.0          # the bounded measurement's cap, registered, for the audit only
BAR_MISSES = 0
BAR_UNNECESSARY = 0.25
BAR_MCC = 0.5


def sample():
    """Every instance, with the plan, the schedule and the ground truth."""
    out = []
    for tag, _, _ in epochs():
        pt = WORK / f"{tag}_td" / "planner_params.json"
        rt = WORK / f"{tag}_replay_td_tol" / "replay_params.json"
        if not (pt.exists() and rt.exists()):
            continue
        tp = json.loads(pt.read_text())
        off = np.fromfile(tp["out_path_offsets"], np.int64)
        pc = np.fromfile(tp["out_path_cells"], np.int32)
        ptm = np.fromfile(tp["out_path_times"], np.float64)
        sh = np.fromfile(WORK / f"{tag}_replay_td_tol" / "out_shadow.bin", np.float64)
        idx = [q for q in range(off.size - 1) if off[q + 1] - off[q] >= 2]
        assert len(idx) == sh.size, (tag, len(idx), sh.size)
        for i, q in enumerate(idx):
            lo, hi = off[q], off[q + 1]
            out.append(dict(tag=tag, plan_index=i, instance=q,
                            plan=pc[lo:hi].astype(np.int32).tolist(),
                            times=ptm[lo:hi].tolist(),
                            world_shadow=float(sh[i]),
                            monitored=bool(sh[i] > 0.0),
                            busted=bool(sh[i] > BUDGET_H),
                            planner_params=str(pt), replay_params=str(rt),
                            graph={k: tp[k] for k in
                                   ("indptr_path", "indices_path", "travel_path",
                                    "itimes_path", "ioffsets_path", "ilit0_path")}))
    return out


def one(rec):
    """Run every leg of one instance and audit each against the counterfactual."""
    work = OUT / rec["tag"] / f"plan{rec['plan_index']:03d}"
    plan = np.array(rec["plan"], np.int32)
    times = np.array(rec["times"], np.float64)
    legs = run_instance(rec["planner_params"], rec["replay_params"], rec["graph"],
                        plan, times, work, BUDGET_H, REPLAN_LIMIT, REPLAN_CAP_S)
    audits, fired_any, fired_warranted = [], False, False
    drift_max, ticks = 0.0, 0
    for n, leg in enumerate(legs):
        wd = work / f"leg{n}_world"
        traj_sh = np.fromfile(wd / "out_traj_shadow.bin", np.float64)
        w = dict(traj=dict(shadow=traj_sh), shadow=leg.world_shadow_max)
        a = A.audit_leg(leg, w, BUDGET_H)
        fired = bool(a["fired"].any())
        fired_any |= fired
        if fired:
            k = int(np.flatnonzero(a["fired"])[0])
            fired_warranted |= bool(a["warranted"][k])
        drift_max = max(drift_max, float(a["drift"].max()) if a["drift"].size else 0.0)
        ticks += int(a["drift"].size)
        audits.append(dict(leg=n, outcome=leg.outcome, reason=leg.reason,
                           fired=fired, fired_at_t=leg.fired_at_t,
                           fired_at_tick=leg.fired_at_tick,
                           ticks_consumed=leg.ticks_consumed,
                           ticks_total=leg.ticks_total,
                           world_shadow=leg.world_shadow_max,
                           world_verdict=leg.world_verdict,
                           bust_any=a["bust_any"], peak_gap=a["peak_gap"],
                           decl_plan=leg.plan.tolist() if fired else [],
                           decl_index=int(leg.log["index"][-1])
                           if leg.log["index"].size else -1,
                           drift_max=float(a["drift"].max()) if a["drift"].size else 0.0,
                           replans=leg.replans))
    return dict(rec, legs=audits, fired=fired_any,
                fired_warranted=fired_warranted, drift_max=drift_max,
                ticks=ticks, work=str(work))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=12)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    recs = sample()
    if args.limit:
        mon = [r for r in recs if r["monitored"]][: args.limit]
        ctl = [r for r in recs if not r["monitored"]][: args.limit]
        recs = mon + ctl
    mon = [r for r in recs if r["monitored"]]
    ctl = [r for r in recs if not r["monitored"]]
    print(f"sample: {len(mon)} monitored, {len(ctl)} controls, "
          f"{sum(r['busted'] for r in mon)} busting; budget {BUDGET_H} h, "
          f"k={REPLAN_LIMIT}, replan cap {REPLAN_CAP_S:.0f} s, "
          f"declaration audit at {DECL_CAP_S:.0f} s capped at d={DECL_CAP}", flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    with ProcessPoolExecutor(max_workers=args.jobs) as ex:
        results = list(ex.map(one, recs))
    (OUT / "results.json").write_text(json.dumps(results, indent=1, default=str))
    report(results)
    return 0


def report(results):
    mon = [r for r in results if r["monitored"]]
    ctl = [r for r in results if not r["monitored"]]
    fired = np.array([r["fired"] for r in mon], bool)
    busted = np.array([r["busted"] for r in mon], bool)
    tp, fp, fn, tn = A.contingency(fired, busted)
    m, j = A.mcc(tp, fp, fn, tn), A.youden(tp, fp, fn, tn)

    print(f"\n=== the contingency table on the {len(mon)} monitored plans ===")
    print(f"  {'':26}{'would have busted':>20}{'would not':>12}")
    print(f"  {'executive fired':<26}{tp:>20}{fp:>12}")
    print(f"  {'did not fire':<26}{fn:>20}{tn:>12}")
    unnec = fp / (fp + tn) if (fp + tn) else float("nan")
    print(f"\n  misses {fn}  (bar: {BAR_MISSES}) -> "
          f"{'PASS' if fn <= BAR_MISSES else 'FAIL'}")
    print(f"  unnecessary-intervention rate {unnec:.4f} = {fp}/{fp+tn}  "
          f"(bar: at most {BAR_UNNECESSARY:.2f}) -> "
          f"{'PASS' if unnec <= BAR_UNNECESSARY else 'FAIL'}")
    print(f"  Matthews {m:+.4f}  (bar: at least {BAR_MCC:.2f}, both constant "
          f"predictors score 0.0000) -> {'PASS' if m >= BAR_MCC else 'FAIL'}")
    print(f"  Youden J {j:+.4f}, reported beside it")

    cf = sum(r["fired"] for r in ctl)
    print(f"\n  controls: {cf} of {len(ctl)} fired  (bar: 0) -> "
          f"{'PASS' if cf == 0 else 'FAIL'}")
    any_fire = int(fired.sum()) + cf
    print(f"  vacuity clause: {any_fire} firings across monitored and controls -> "
          f"{'reported as UNTESTED' if any_fire == 0 else 'not vacuous'}")

    print(f"\n=== declarations ===")
    decl = [(r, L) for r in results for L in r["legs"]
            if L["outcome"] == "fired_declared"]
    kinds = {}
    for _, L in decl:
        kinds[L["reason"]] = kinds.get(L["reason"], 0) + 1
    print(f"  {len(decl)} declarations: " +
          (", ".join(f"{k} {v}" for k, v in sorted(kinds.items())) or "none"))
    claims = [(r, L) for r, L in decl if L["reason"] in ("infeasible", "unresolved")]
    print(f"  of those, {len(claims)} claim something about feasibility and are "
          f"auditable; cap d={DECL_CAP}")
    false_decl = 0
    for n, (r, L) in enumerate(claims):
        if n >= DECL_CAP:
            print(f"  {len(claims)-DECL_CAP} declarations beyond the cap are "
                  f"reported as UNAUDITED, not skipped silently")
            break
        a = A.audit_declaration(L["decl_plan"], L["decl_index"], L["fired_at_t"],
                                r["planner_params"],
                                OUT / "audit" / f"{r['tag']}_{r['plan_index']}_{L['leg']}",
                                DECL_CAP_S)
        if not a["audited"]:
            print(f"    {r['tag']} plan {r['plan_index']} leg {L['leg']}: not "
                  f"auditable, {a['reason']}")
            continue
        refuted = a["plan_exists"] and L["reason"] == "infeasible"
        false_decl += int(refuted)
        print(f"    {r['tag']} plan {r['plan_index']} leg {L['leg']}: declared "
              f"{L['reason']}; at {a['cap_s']:.0f} s the engine "
              f"{'FOUND a plan' if a['plan_exists'] else ('aborted' if a['aborted'] else 'found none')}"
              f" in {a['seconds']:.2f} s -> {'FALSE DECLARATION' if refuted else 'upheld'}")
    if not claims:
        # A criterion met on no comparable case is not met. This record has
        # made that mistake once already and labels it rather than counting it.
        print(f"  false declarations: VACUOUS, no declaration claimed anything "
              f"about feasibility, so the {DECL_CAP_S:.0f} s audit ran on "
              f"nothing")
    else:
        print(f"  false declarations: {false_decl} of {min(len(claims), DECL_CAP)} "
              f"audited  (bar: 0) -> {'PASS' if false_decl == 0 else 'FAIL'}")

    print(f"\n=== drift, per instance, worst first (tolerance "
          f"{A.DRIFT_TOL_H:.7f} h) ===")
    rows = sorted(results, key=lambda r: -r["drift_max"])[:12]
    print(f"  {'epoch':<16}{'plan':>5}{'ticks':>7}{'max drift h':>14}{'over tol':>10}")
    for r in rows:
        print(f"  {r['tag']:<16}{r['plan_index']:>5}{r['ticks']:>7}"
              f"{r['drift_max']:>14.3e}{'YES' if r['drift_max'] > A.DRIFT_TOL_H else 'no':>10}")
    allmax = max(r["drift_max"] for r in results)
    over = [r for r in results if r["drift_max"] > A.DRIFT_TOL_H]
    print(f"  worst drift anywhere {allmax:.3e} h over "
          f"{sum(r['ticks'] for r in results)} ticks; {len(over)} instances over "
          f"tolerance -> {'PASS' if not over else 'FAIL'}")

    diagnose(results)

    gaps = [L["peak_gap"] for r in results for L in r["legs"]]
    print(f"\n  counterfactual soundness check, the checker's reported maximum "
          f"minus the trajectory's tick peak:\n    max {max(gaps):.3e} h, "
          f"min {min(gaps):.3e} h over {len(gaps)} legs")


class _Intervals:
    """The planner's interval table, read back to localise a firing's cause."""

    def __init__(self, g):
        self.t = np.fromfile(g["itimes_path"], np.float64)
        self.o = np.fromfile(g["ioffsets_path"], np.int64)
        self.l = np.fromfile(g["ilit0_path"], np.uint8)

    def is_lit(self, c, t):
        lo, hi = self.o[c], self.o[c + 1]
        k = int(np.searchsorted(self.t[lo:hi], t, side="right"))
        return bool((self.l[c] != 0) != (k % 2 == 1))


def diagnose(results):
    """Why the executive fired, per firing, rather than in aggregate.

    The monitor has one rule that can fire on its own authority: where the world
    says dark and the planner's model says lit, the observation wins and the
    darkness is projected forward to the model's next transition. This asks how
    many firings took that branch and what it predicted, because a rate of
    unnecessary interventions says nothing about which part produced them.
    """
    cache, rows = {}, []
    for r in results:
        if not r["monitored"] or not r["fired"]:
            continue
        L = next(l for l in r["legs"] if l["fired"])
        w = Path(r["work"]) / f"leg{L['leg']}_exec"
        lit = np.fromfile(w / "out_log_lit.bin", np.uint8)
        fired = np.fromfile(w / "out_log_fired.bin", np.uint8)
        idxs = np.fromfile(w / "out_log_index.bin", np.int32)
        tms = np.fromfile(w / "out_log_t.bin", np.float64)
        spred = np.fromfile(w / "out_log_s_pred.bin", np.float64)
        k = int(np.flatnonzero(fired)[0])
        if r["tag"] not in cache:
            cache[r["tag"]] = _Intervals(r["graph"])
        plan = np.array(r["plan"], np.int32)
        pos = int(min(max(idxs[k], 0), plan.size - 1))
        dark = lit[k] == 0
        model_lit = cache[r["tag"]].is_lit(int(plan[pos]), float(tms[k]))
        rows.append(dict(busted=r["busted"], dark=bool(dark),
                         contradiction=bool(dark and model_lit),
                         s_pred=float(spred[k]), shadow=r["world_shadow"]))
    if not rows:
        print("\n=== why it fired: nothing fired ===")
        return
    print(f"\n=== why it fired, per firing ===")
    print(f"  firings {len(rows)}; rover in darkness at the firing tick "
          f"{sum(r['dark'] for r in rows)}; the model said lit where the world "
          f"said dark on {sum(r['contradiction'] for r in rows)}")
    for lbl, sel in (("warranted", lambda x: x["busted"]),
                     ("unnecessary", lambda x: not x["busted"])):
        g = [x for x in rows if sel(x)]
        if not g:
            continue
        print(f"  {lbl}: {len(g)} firings, {sum(x['contradiction'] for x in g)} "
              f"on the contradicted-model branch")
        print(f"    predicted shadow at firing, median {np.median([x['s_pred'] for x in g]):.3f} h; "
              f"true maximum, median {np.median([x['shadow'] for x in g]):.3f} h")


if __name__ == "__main__":
    sys.exit(main())
