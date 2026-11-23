"""Score a correction through the executive on the registered McNemar test.

The test is registered in PLAN.md on 2026-10-05, before this existed: paired
against the bounded rule on the same 45 monitored plans, passing if the
candidate removes enough of the bounded rule's 17 false alarms to reach
one-sided exact p <= 0.05, with zero busts lost and control firings still 0.
That was the core host. --host tol05 runs the same arms on the primary branch's
host, the tolerance axis at 0.5 h, with the bars PLAN.md registers for it.
--arms selects arms; "none", the bounded rule every arm is paired against,
always runs.

B3b as a monitor correction is "the model says the cell the rover is on is lit
but the next cell on the path is dark at that moment". That is the interval
table on both cells, not the exact-time horizon re-evaluation B3b was measured
with, and the substitution is justified by measurement rather than convenience:
B3a, the exact-time re-evaluation, differed from the table on 5 of 91690
samples, so on this terrain it recovers the table. Using the table costs no
SPICE call and no horizon array per worker.
"""
import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from math import comb
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import examples.run_executive as R                               # noqa: E402
from src import auditor as A                                     # noqa: E402
from src.disagree import FlatIntervals                           # noqa: E402
from src.executive import run_instance                           # noqa: E402

# One output directory per host, beside the executive's own, so the core's
# record and the primary branch's never share a results.json.
OUT = {"core": ROOT / "data" / "work" / "executive_correction",
       "tol05": ROOT / "data" / "work" / "executive_correction_tol05"}
ARMS = ("none", "dark", "lit", "both", "model")
SUNDIR = ROOT / "data" / "work" / "sunpos"
_CTX = {}


def feature_ctx():
    """Cells, horizon and horizon distance, built once per process.

    The monitor computes the model's features at execution time, so the
    correction function does too rather than reading a precomputed table: a
    replan produces a plan whose steps were never precomputed for.
    """
    if "cells" not in _CTX:
        import examples.plan_pilot as P
        from src.cost import Rover
        _, n, px, dem, horizon, cells, gx, gy, edges, graph = P.load(
            Rover(v_flat=50.0))
        _CTX["cells"] = cells
        _CTX["horizon"] = horizon.astype(np.float64).reshape(-1, 360)
        _CTX["hdist"] = np.fromfile(
            ROOT / "data" / "work" / "horizon_dist.f32",
            dtype=np.float32).reshape(-1, 360)
    return _CTX["cells"], _CTX["horizon"], _CTX["hdist"]


def load_model(host):
    """The host's model, refused unless it was trained for that host.

    The core's model is trained on everything. tol05's is trained without the
    host's own start/goal pairs (PLAN.md, D132 step 13), and when that set is
    below the registered episode minimum no model exists and the arm is not run.
    A model trained for the other host is never substituted.
    """
    import joblib
    from examples.traversability_model import MODEL_PATH
    path = ROOT / MODEL_PATH[host]
    if not path.exists():
        raise SystemExit(f"the {host} model arm is not run: no model at {path}; "
                         f"see the model log for the reason")
    b = joblib.load(path)
    want = None if host == "core" else host
    if b.get("exclude_host") != want:
        raise SystemExit(f"{path} was trained with exclude_host="
                         f"{b.get('exclude_host')!r}, the {host} arm needs {want!r}")
    return b


def model_correction(graph, plan, times, tag, host):
    """The learned model's ternary code at each plan step."""
    from src.frames import azimuth_elevation_from_frame, horizon_lookup
    if "clf" not in _CTX:
        b = load_model(host)
        _CTX["clf"], _CTX["feat"] = b["model"], b["features"]
    cells, horizon, hdist = feature_ctx()
    if tag not in _CTX.setdefault("sun", {}):
        _CTX["sun"][tag] = np.load(SUNDIR / f"{tag}.npy")
    sp, cad = _CTX["sun"][tag], 1.0 / 60.0
    iv = FlatIntervals(graph["itimes_path"], graph["ioffsets_path"],
                       graph["ilit0_path"])
    plan = np.asarray(plan, np.int64)
    n = plan.size
    cur = plan[:n]
    nxt = plan[np.minimum(np.arange(n) + 1, n - 1)]
    t = np.asarray(times, float)[:n]
    k = np.clip((t / cad).astype(int), 0, sp.shape[0] - 1)
    spos = sp[k]
    az, el = azimuth_elevation_from_frame(
        spos, cells.origin[cur], cells.up[cur], cells.east[cur],
        cells.north[cur])
    he = horizon_lookup(horizon[cur], az)
    hd = hdist[cur, np.round(az).astype(int) % 360].astype(float)
    mo = np.array([iv.is_lit(int(cur[i]), float(t[i])) for i in range(n)], float)
    nl = np.array([iv.is_lit(int(nxt[i]), float(t[i])) for i in range(n)], float)
    X = np.column_stack([mo, nl, el - he, el, he, hd])
    code = _CTX["clf"].predict(X).astype(np.uint8)
    code[n - 1] = 0
    return code


def b3b_correction(graph, plan, times, arm):
    """The ternary stream: 0 believe the model, 1 treat as dark, 2 as lit.

    B3b marks a step dark where the model calls this cell lit and the next dark;
    B3b-lit marks it lit where the model calls this cell dark and the next lit.
    A step satisfies at most one of the two by construction, since they require
    opposite states at the same cell, so no conflict rule is needed.
    """
    iv = FlatIntervals(graph["itimes_path"], graph["ioffsets_path"],
                       graph["ilit0_path"])
    plan = np.asarray(plan, np.int64)
    out = np.zeros(plan.size, np.uint8)
    for i in range(plan.size - 1):
        t = float(times[i])
        here = iv.is_lit(int(plan[i]), t)
        nxt = iv.is_lit(int(plan[i + 1]), t)
        if here and not nxt and arm in ("dark", "both"):
            out[i] = 1
        elif (not here) and nxt and arm in ("lit", "both"):
            out[i] = 2
    return out


def one(arg):
    rec, kind = arg
    work = OUT[rec["host"]] / kind / rec["tag"] / f"plan{rec['plan_index']:03d}"
    plan = np.array(rec["plan"], np.int32)
    times = np.array(rec["times"], np.float64)
    if kind == "none":
        fn = None
    elif kind == "model":
        fn = lambda p, t: model_correction(rec["graph"], p, t, rec["tag"], rec["host"])
    else:
        fn = lambda p, t: b3b_correction(rec["graph"], p, t, kind)
    legs = run_instance(rec["planner_params"], rec["replay_params"],
                        rec["graph"], plan, times, work, rec["budget_h"],
                        R.REPLAN_LIMIT, R.REPLAN_CAP_S, True,
                        R.CONTRA_HORIZON_H, fn)
    fired_any, fired_warranted = False, False
    for n, leg in enumerate(legs):
        wd = work / f"leg{n}_world"
        traj_sh = np.fromfile(wd / "out_traj_shadow.bin", np.float64)
        w = dict(traj=dict(shadow=traj_sh), shadow=leg.world_shadow_max)
        a = A.audit_leg(leg, w, rec["budget_h"])
        if bool(a["fired"].any()):
            fired_any = True
            k = int(np.flatnonzero(a["fired"])[0])
            fired_warranted |= bool(a["warranted"][k])
    return dict(tag=rec["tag"], plan_index=rec["plan_index"], kind=kind,
                monitored=rec["monitored"], busted=rec["busted"],
                fired=fired_any, fired_warranted=fired_warranted)


def paired_verdict(lost, control_fires, p, n_busts, host):
    """The registered paired test's verdict for one arm.

    VOID when the sample has no bust, since "zero busts lost" is then met by
    every arm and the test checks only the controls and the false alarms. On
    any host but the core the test as registered on 2026-10-05 is not the
    registered test, so it is labelled as a comparison and decides nothing.
    """
    ok = lost == 0 and control_fires == 0 and p <= 0.05
    if n_busts == 0:
        return "VOID"
    if host != "core":
        return f"{'would pass' if ok else 'would fail'}, core test, not registered on {host}"
    return "PASS" if ok else "FAIL"


def mcnemar(b, c):
    n = b + c
    return sum(comb(n, k) for k in range(0, c + 1)) / 2 ** n if n else 1.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=6)
    ap.add_argument("--host", choices=sorted(R.HOSTS), default="core")
    # "none" is the bounded rule every arm is paired against, so it always runs.
    # "model" needs the host's model (traversability_model.MODEL_PATH), which
    # only exists once it has been trained on the current labels, and for tol05
    # only if the excluded training set met the registered episode minimum.
    ap.add_argument("--arms", default=",".join(ARMS))
    args = ap.parse_args()
    arms_run = tuple(a for a in ARMS if a in args.arms.split(","))
    unknown = set(args.arms.split(",")) - set(ARMS)
    if unknown or "none" not in arms_run:
        ap.error(f"--arms must name arms from {ARMS} and include none; got {args.arms}")
    recs = R.sample(R.HOSTS[args.host])
    out = OUT[args.host]
    out.mkdir(parents=True, exist_ok=True)
    if "model" in arms_run:
        load_model(args.host)       # refuse before any worker starts
        # Sun positions cached once, so no worker pays 43200 SPICE calls per epoch.
        SUNDIR.mkdir(parents=True, exist_ok=True)
        from examples.build_intervals import epochs, sun_table_at
        for tag, base, off in epochs():
            f = SUNDIR / f"{tag}.npy"
            if not f.exists() and any(r["tag"] == tag for r in recs):
                np.save(f, sun_table_at(base, off, cadence_h=1.0 / 60.0).p)
                print(f"  cached sun positions for {tag}", flush=True)
    jobs = [(r, k) for k in arms_run for r in recs]
    print(f"host {args.host}, budget {recs[0]['budget_h']} h: {len(recs)} instances x "
          f"{len(arms_run)} arms ({', '.join(arms_run)}), {args.jobs} workers", flush=True)
    with ProcessPoolExecutor(max_workers=args.jobs) as ex:
        res = list(ex.map(one, jobs))
    (out / "results.json").write_text(json.dumps(res, indent=1))

    key = lambda r: (r["tag"], r["plan_index"])
    arms = {k: {key(r): r for r in res if r["kind"] == k} for k in arms_run}
    base = arms["none"]
    mon = [k for k in base if base[k]["monitored"]]
    print(f"\nmonitored {len(mon)}, controls {len(base)-len(mon)}")
    for nm in arms_run:
        arm = arms[nm]
        TP = sum(1 for k in mon if arm[k]["busted"] and arm[k]["fired"])
        FN = sum(1 for k in mon if arm[k]["busted"] and not arm[k]["fired"])
        FP = sum(1 for k in mon if not arm[k]["busted"] and arm[k]["fired"])
        TN = sum(1 for k in mon if not arm[k]["busted"] and not arm[k]["fired"])
        cf = sum(1 for k in base if not base[k]["monitored"] and arm[k]["fired"])
        den = np.sqrt(float((TP+FP)*(TP+FN)*(TN+FP)*(TN+FN)))
        label = {"none": "bounded rule, no stream", "dark": "B3b dark marks only",
                 "lit": "B3b-lit lit marks only",
                 "both": "both directions",
                 "model": "learned ternary model"}[nm]
        print(f"  {label:24s} TP {TP} FN {FN} FP {FP} TN {TN}  controls {cf}  "
              f"unnecessary {FP/max(FP+TN,1):.4f}  MCC "
              f"{(TP*TN-FP*FN)/den if den>0 else float('nan'):+.4f}")
    nonbust = [k for k in mon if base[k]["busted"] is False]
    print(f"\nthe registered paired test, against the bounded rule on the same "
          f"plans:")
    print(f"  {'arm':24s}{'b':>4}{'c':>4}{'busts lost':>12}{'ctl fires':>11}"
          f"{'p':>9}  verdict")
    for nm in arms_run[1:]:
        cand = arms[nm]
        b = sum(1 for k in nonbust if base[k]["fired"] and not cand[k]["fired"])
        c = sum(1 for k in nonbust if not base[k]["fired"] and cand[k]["fired"])
        lost = sum(1 for k in mon if base[k]["busted"] and not cand[k]["fired"])
        cf = sum(1 for k in base if not base[k]["monitored"] and cand[k]["fired"])
        p = mcnemar(b, c)
        n_busts = sum(1 for k in mon if base[k]["busted"])
        print(f"  {nm:24s}{b:4d}{c:4d}{lost:12d}{cf:11d}{p:9.4f}  "
              f"{paired_verdict(lost, cf, p, n_busts, args.host)}")
    print(f"\n  zero-busts-lost is now a live guard and not a formality: a lit")
    print(f"  mark lowers the forecast, so it can hide a real bust. Any arm")
    print(f"  above with busts lost > 0 fails on that alone.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
