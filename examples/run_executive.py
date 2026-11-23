"""Run the executive over a registered host sample and score it against the auditor.

PLAN.md, "The executive", registers the sample, the tick definition, the
replan-at-a-lit-cell constraint, k, the declaration-audit cap d, the three bars
and the five failure conditions, all before this ran.

Monitored set: every feasible time-dependent plan of the host whose integrated
continuous shadow is above zero. Control set: every one whose shadow is exactly
zero. A plan busts when that shadow exceeds the host's own budget. The executive
must fire on the busts and must never fire on the controls.

Two hosts. "core" is the sample P8 and the paired design were registered on,
every core epoch at the core's 2 h budget. "tol05" is the primary branch the
registered successor condition selected on 2026-11-21: the tolerance axis point
with a 0.5 h budget, over the axis's four epochs. Each host writes to its own
directory, so running one never overwrites the other.
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
from examples.sweep_axes import AXES as AXIS_POINTS, CORE, EPOCHS as AXIS_EPOCHS  # noqa: E402
from src import auditor as A                                     # noqa: E402
from src.executive import (CONTRA_HORIZON_H, run_instance,       # noqa: E402
                           simulate, plan_waits)

# The budget is a property of the host and travels with each record, never a
# module constant: the executive, the auditor and the bust label must all use
# the budget the host's plans were made and replayed under, and sample() checks
# that against both parameter files rather than trusting this table.
HOSTS = {
    "core": dict(name="core", work=ROOT / "data" / "work" / "sweep", stem="",
                 tags=None, budget_h=CORE["tol"],
                 out=ROOT / "data" / "work" / "executive"),
    "tol05": dict(name="tol05", work=ROOT / "data" / "work" / "axes",
                  stem="tolerance_0_", tags=AXIS_EPOCHS,
                  budget_h=AXIS_POINTS["tolerance"][0]["tol"],
                  out=ROOT / "data" / "work" / "executive_tol05"),
}
REPLAN_LIMIT = 2            # k, registered
REPLAN_CAP_S = 30.0         # the cap the search runs under everywhere else
DECL_CAP = 6                # d, registered
DECL_CAP_S = 300.0          # the bounded measurement's cap, registered, for the audit only
BAR_MISSES = 0
BAR_UNNECESSARY = 0.25
BAR_MCC = 0.5
RULES = (("unbounded", False), ("bounded", True))


def verdict(ok, n_busts):
    """A verdict on bars that are about catching busts.

    With no bust in the sample, the miss bar is met by every rule, never-fire
    included, and Matthews has no positive class to be computed on, so the bars
    do not test anything. The verdict is then VOID, not PASS and not FAIL. That
    is the corrected core: P8's bars were registered on four busts and it has
    none.
    """
    if n_busts == 0:
        return "VOID"
    return "PASS" if ok else "FAIL"


def sample(host=HOSTS["core"]):
    """Every instance of the host, with the plan, the schedule and the ground truth."""
    out = []
    work, stem, budget = Path(host["work"]), host["stem"], float(host["budget_h"])
    tags = host["tags"] if host["tags"] is not None else [t for t, _, _ in epochs()]
    for tag in tags:
        pt = work / f"{stem}{tag}_td" / "planner_params.json"
        rd = work / f"{stem}{tag}_replay_td_tol"
        rt = rd / "replay_params.json"
        if not (pt.exists() and rt.exists()):
            continue
        tp = json.loads(pt.read_text())
        made, judged = float(tp["shadow_tol_h"]), float(json.loads(rt.read_text())["shadow_tol_h"])
        if not (made == judged == budget):
            raise ValueError(f"{host['name']} {tag}: budget {budget} h, but the plans "
                             f"were made at {made} h and replayed at {judged} h")
        off = np.fromfile(tp["out_path_offsets"], np.int64)
        pc = np.fromfile(tp["out_path_cells"], np.int32)
        ptm = np.fromfile(tp["out_path_times"], np.float64)
        sh = np.fromfile(rd / "out_shadow.bin", np.float64)
        idx = [q for q in range(off.size - 1) if off[q + 1] - off[q] >= 2]
        assert len(idx) == sh.size, (tag, len(idx), sh.size)
        for i, q in enumerate(idx):
            lo, hi = off[q], off[q + 1]
            out.append(dict(tag=tag, host=host["name"], budget_h=budget,
                            out=str(host["out"]), plan_index=i, instance=q,
                            plan=pc[lo:hi].astype(np.int32).tolist(),
                            times=ptm[lo:hi].tolist(),
                            world_shadow=float(sh[i]),
                            monitored=bool(sh[i] > 0.0),
                            busted=bool(sh[i] > budget),
                            planner_params=str(pt), replay_params=str(rt),
                            graph={k: tp[k] for k in
                                   ("indptr_path", "indices_path", "travel_path",
                                    "itimes_path", "ioffsets_path", "ilit0_path")}))
    return out


def one(arg):
    """Run every leg of one instance under one contradiction rule, and audit it."""
    rec, rule, bounded = arg
    out_dir = Path(rec["out"])
    work = out_dir / rule / rec["tag"] / f"plan{rec['plan_index']:03d}"
    plan = np.array(rec["plan"], np.int32)
    times = np.array(rec["times"], np.float64)
    legs = run_instance(rec["planner_params"], rec["replay_params"], rec["graph"],
                        plan, times, work, rec["budget_h"], REPLAN_LIMIT, REPLAN_CAP_S,
                        bounded, CONTRA_HORIZON_H)
    audits, fired_any, fired_warranted = [], False, False
    drift_max, ticks = 0.0, 0
    for n, leg in enumerate(legs):
        wd = work / f"leg{n}_world"
        traj_sh = np.fromfile(wd / "out_traj_shadow.bin", np.float64)
        w = dict(traj=dict(shadow=traj_sh), shadow=leg.world_shadow_max)
        a = A.audit_leg(leg, w, rec["budget_h"])
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
    return dict(rec, rule=rule, legs=audits, fired=fired_any,
                fired_warranted=fired_warranted, drift_max=drift_max,
                ticks=ticks, work=str(work))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=12)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--host", choices=sorted(HOSTS), default="core")
    args = ap.parse_args()
    host = HOSTS[args.host]
    recs = sample(host)
    if args.limit:
        mon = [r for r in recs if r["monitored"]][: args.limit]
        ctl = [r for r in recs if not r["monitored"]][: args.limit]
        recs = mon + ctl
    mon = [r for r in recs if r["monitored"]]
    ctl = [r for r in recs if not r["monitored"]]
    print(f"sample: {len(mon)} monitored, {len(ctl)} controls, "
          f"{sum(r['busted'] for r in mon)} busting; host {host['name']}, "
          f"budget {host['budget_h']} h, "
          f"k={REPLAN_LIMIT}, replan cap {REPLAN_CAP_S:.0f} s, "
          f"declaration audit at {DECL_CAP_S:.0f} s capped at d={DECL_CAP}")
    print(f"two contradiction rules from one binary and one sample: the first "
          f"rule, projecting to the model's next transition, and its registered "
          f"replacement at a {CONTRA_HORIZON_H:.4f} h horizon", flush=True)
    out = Path(host["out"])
    out.mkdir(parents=True, exist_ok=True)
    jobs = [(r, rule, b) for rule, b in RULES for r in recs]
    with ProcessPoolExecutor(max_workers=args.jobs) as ex:
        allres = list(ex.map(one, jobs))
    (out / "results.json").write_text(json.dumps(allres, indent=1, default=str))
    by_rule = {rule: [r for r in allres if r["rule"] == rule] for rule, _ in RULES}
    for rule, _ in RULES:
        print(f"\n{'='*72}\n{rule.upper()} contradiction rule"
              + (f", horizon {CONTRA_HORIZON_H:.4f} h" if rule == "bounded"
                 else ", projecting to the model's next transition")
              + f"\n{'='*72}")
        report(by_rule[rule])
    side_by_side(by_rule, host)
    return 0


def side_by_side(by_rule, host=HOSTS["core"]):
    """Both verdicts in one table, scored against the identical bars."""
    first = next(iter(by_rule.values()))
    n_mon = sum(r["monitored"] for r in first)
    n_ctl = len(first) - n_mon
    n_bust = sum(r["busted"] for r in first if r["monitored"])
    n_quiet = n_mon - n_bust
    print(f"\n{'='*72}\nBOTH RULES, the same {n_mon} monitored plans and {n_ctl} "
          f"controls, the same bars\n{'='*72}")
    hdr = f"  {'':<34}" + "".join(f"{r:>16}" for r, _ in RULES)
    print(hdr)
    cells = {}
    for rule, _ in RULES:
        res = by_rule[rule]
        mon = [r for r in res if r["monitored"]]
        ctl = [r for r in res if not r["monitored"]]
        fired = np.array([r["fired"] for r in mon], bool)
        busted = np.array([r["busted"] for r in mon], bool)
        tp, fp, fn, tn = A.contingency(fired, busted)
        unnec = fp / (fp + tn) if (fp + tn) else float("nan")
        cells[rule] = dict(tp=tp, fp=fp, fn=fn, tn=tn, unnec=unnec,
                           mcc=A.mcc(tp, fp, fn, tn), j=A.youden(tp, fp, fn, tn),
                           ctl=sum(r["fired"] for r in ctl),
                           fires=int(fired.sum()))
    rows = [
        ("fired and would have busted", lambda c: f"{c['tp']} of {n_bust}"),
        ("missed a bust", lambda c: f"{c['fn']}"),
        ("fired, would not have busted", lambda c: f"{c['fp']} of {n_quiet}"),
        ("correctly quiet", lambda c: f"{c['tn']} of {n_quiet}"),
        ("control firings", lambda c: f"{c['ctl']} of {n_ctl}"),
        ("", lambda c: ""),
        (f"bar 1  misses == {BAR_MISSES}",
         lambda c: f"{c['fn']}  {verdict(c['fn'] <= BAR_MISSES, n_bust)}"),
        (f"bar 2  unnecessary <= {BAR_UNNECESSARY:.2f}",
         lambda c: f"{c['unnec']:.4f}  {verdict(c['unnec'] <= BAR_UNNECESSARY, n_bust)}"),
        (f"bar 3  Matthews >= {BAR_MCC:.2f}",
         lambda c: f"{c['mcc']:+.4f}  {verdict(c['mcc'] >= BAR_MCC, n_bust)}"),
        ("       Youden J, reported", lambda c: f"{c['j']:+.4f}"),
    ]
    for label, f in rows:
        print(f"  {label:<34}" + "".join(f"{f(cells[r]):>16}" for r, _ in RULES))
    for rule, _ in RULES:
        c = cells[rule]
        ok = (c["fn"] <= BAR_MISSES and c["unnec"] <= BAR_UNNECESSARY
              and c["mcc"] >= BAR_MCC and c["ctl"] == 0)
        name = "P8" if rule == "bounded" else "the first rule"
        if host["name"] != "core":
            # P8's bars were registered on the core and nowhere else. On any
            # other host they are printed for comparison and decide nothing.
            print(f"\n  {name}, P8's core bars applied for comparison only, "
                  f"not registered on {host['name']}: "
                  f"{'would pass' if ok else 'would fail'}")
            continue
        if n_bust == 0:
            print(f"\n  {name}: VOID, no busts in the sample to catch; reported as "
                  f"false-alarm cost only: {c['fp']} of {n_quiet} monitored plans "
                  f"fired, {c['ctl']} of {n_ctl} controls")
            continue
        print(f"\n  {name}: {verdict(ok, n_bust)} "
              f"(all three bars required; misses {c['fn']}, unnecessary "
              f"{c['unnec']:.4f}, Matthews {c['mcc']:+.4f})")


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
          f"{verdict(fn <= BAR_MISSES, tp + fn)}")
    print(f"  unnecessary-intervention rate {unnec:.4f} = {fp}/{fp+tn}  "
          f"(bar: at most {BAR_UNNECESSARY:.2f}) -> "
          f"{verdict(unnec <= BAR_UNNECESSARY, tp + fn)}")
    print(f"  Matthews {m:+.4f}  (bar: at least {BAR_MCC:.2f}, both constant "
          f"predictors score 0.0000) -> {verdict(m >= BAR_MCC, tp + fn)}")
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
                                Path(r["out"]) / "audit" / f"{r['tag']}_{r['plan_index']}_{L['leg']}",
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
