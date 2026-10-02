"""Check 3 on small instances, and the 20-instance pilot that sizes the sweep.

Check 3 compares the label-setting planner against the time-expanded reference
at a sequence of dt, against the inequality and convergence criterion PLAN.md
registers. The reference's dark oracle is evaluated from the sun and the
horizon directly, sub-sampled within each step, and never reads the interval
tables, so the two share no illumination machinery.

The pilot measures what a design cannot estimate: labels per cell and seconds
per search, and from them the cost of the full sweep.
"""
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import rasterio
import spiceypy as sp
from rasterio.windows import Window, from_bounds
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.cost import Rover, build_edges, legal                      # noqa: E402
from src.illumination import RegionCells                            # noqa: E402
from src.intervals import LitIntervals, SunTable, g_at              # noqa: E402
from src.planner import Graph, plan                                 # noqa: E402
from src.reference import plan_time_expanded                        # noqa: E402

DATA = ROOT / "data"
KERNELS = ["naif0012.tls", "pck00010.tpc", "moon_pa_de421_1900-2050.bpc",
           "moon_080317.tf", "de421.bsp"]
FRAME = "MOON_ME_DE421"
EPOCH = "2028-03-15T00:00:00"
HORIZON_H = 720.0            # 30 days, one lunation: below that, waiting cannot help
CHECK3_HORIZON_H = 12.0      # check 3's instances are short; its state space is not
CHECK3_PAD = 6              # cells of margin around the start/goal box
SHADOW_TOL_H = 2.0           # a pilot value, not a registered sweep point
SUB_STEPS = 8
LIMIT_S = 60.0                # sub-samples per reference step for the dark oracle
BANDS = ((100., 250.), (250., 600.), (600., 1500.), (1500., 4000.), (4000., 9000.))
SEED = 20260923


def load(rover):
    p = json.loads((DATA / "work" / "params.json").read_text())
    n, px, x0, y1, R = (p["region_nx"], p["region_px"], p["region_ox"],
                        p["region_oy"], p["radius_m"])
    side = n * px
    with rasterio.open(DATA / "lola" / "LDEM_80S_20MPP_ADJ.TIF") as ds:
        w = from_bounds(x0, y1 - side, x0 + side, y1, ds.transform)
        dem = ds.read(1, window=Window(round(w.col_off), round(w.row_off), n, n)).astype(float)
    horizon = np.fromfile(p["out_horizon"], dtype=np.float32).reshape(n * n, p["n_azimuth"])
    xs = x0 + (np.arange(n) + 0.5) * px
    ys = y1 - (np.arange(n) + 0.5) * px
    gx, gy = np.meshgrid(xs, ys)
    cells = RegionCells(gx.ravel(), gy.ravel(), dem.ravel(), radius=R,
                        observer_height=p["observer_height_m"])
    edges = build_edges(dem, x0, y1, px, radius=R)
    graph = Graph.build(edges, rover, n * n)
    return p, n, px, dem, horizon, cells, gx, gy, edges, graph


def sun_table(horizon_h, cadence_h=0.25):
    for k in KERNELS:
        sp.furnsh(str(DATA / "kernels" / k))
    t0 = sp.str2et(EPOCH)
    ets = np.arange(t0, t0 + (horizon_h + cadence_h) * 3600.0, cadence_h * 3600.0)
    pos = np.array([sp.spkpos("SUN", e, FRAME, "LT+S", "MOON")[0] for e in ets])
    sp.kclear()
    return SunTable(pos, cadence_h)


def run_cpp(graph, intervals, pairs, tol, t_start, horizon_h, limit_s, work,
            components=None, positions=None, v_cap=None, shadow_grid_h=0.0,
            relax_dominance=False, want_paths=False, defer=False):
    """Write the binary inputs, run cpp/build/planner, read the results back."""
    work.mkdir(parents=True, exist_ok=True)
    files = {
        "indptr": (graph.indptr.astype(np.int64), "indptr_path"),
        "indices": (graph.indices.astype(np.int32), "indices_path"),
        "travel": (graph.travel.astype(np.float64), "travel_path"),
        "itimes": (intervals.times.astype(np.float64), "itimes_path"),
        "ioffsets": (intervals.offsets.astype(np.int64), "ioffsets_path"),
        "ilit0": (intervals.lit0.astype(np.uint8), "ilit0_path"),
        "pairs": (np.asarray(pairs, dtype=np.int32).ravel(), "pairs_path"),
    }
    if components is not None:
        files["component"] = (np.asarray(components, dtype=np.int32), "component_path")
    if positions is not None and v_cap:
        files["position"] = (np.ascontiguousarray(positions, dtype=np.float64).ravel(),
                             "position_path")
    params = {}
    for name, (arr, key) in files.items():
        path = work / f"{name}.bin"
        path.write_bytes(arr.tobytes())
        params[key] = str(path)
    n_inst = len(pairs)
    params.update(n_cells=int(graph.n_cells), n_edges=int(graph.indices.size),
                  n_transitions=int(intervals.times.size), n_instances=n_inst,
                  shadow_tol_h=tol, t_start_h=t_start, horizon_h=horizon_h,
                  time_limit_s=limit_s,
                  v_cap=float(v_cap) if (positions is not None and v_cap) else 0.0,
                  shadow_grid_h=float(shadow_grid_h),
                  relax_dominance=1.0 if relax_dominance else 0.0)
    keys = ["arrival", "seconds", "flags", "pathlen", "stats"]
    if want_paths:
        keys += ["path_cells", "path_times", "path_offsets"]
    for k in keys:
        params[f"out_{k}"] = str(work / f"out_{k}.bin")
    (work / "planner_params.json").write_text(json.dumps(params, indent=2))
    if defer:
        return [str(ROOT / "cpp" / "build" / "planner"),
                str(work / "planner_params.json")], params
    subprocess.run([str(ROOT / "cpp" / "build" / "planner"),
                    str(work / "planner_params.json")], check=True)
    return read_cpp(params, want_paths)


def read_cpp(params, want_paths=False):
    """Read back what the planner binary wrote."""
    return dict(
        arrival=np.fromfile(params["out_arrival"], dtype=np.float64),
        seconds=np.fromfile(params["out_seconds"], dtype=np.float64),
        aborted=np.fromfile(params["out_flags"], dtype=np.uint8).astype(bool),
        path_len=np.fromfile(params["out_pathlen"], dtype=np.int32),
        stats=np.fromfile(params["out_stats"], dtype=np.int64).reshape(-1, 4),
        **(dict(
            path_cells=np.fromfile(params["out_path_cells"], dtype=np.int32),
            path_times=np.fromfile(params["out_path_times"], dtype=np.float64),
            path_offsets=np.fromfile(params["out_path_offsets"], dtype=np.int64))
           if want_paths else {}))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--instances", type=int, default=20)
    ap.add_argument("--check3", type=int, default=6)
    ap.add_argument("--draw", choices=("registered", "lit", "changing"),
                    default="changing",
                    help="changing: the superseded rule, the largest component of cells "
                         "with at least two transitions in the horizon. registered: the "
                         "original, the largest legal component. lit: cells lit at "
                         "departure, used once for sizing and not a registered rule.")
    ap.add_argument("--engine", choices=("python", "cpp", "both"), default="python")
    ap.add_argument("--skip-check3", action="store_true")
    args = ap.parse_args()
    rover = Rover(v_flat=50.0)
    p, n, px, dem, horizon, cells, gx, gy, edges, graph = load(rover)
    print(f"region {n}x{n}, commit {p['source_commit']}; graph {graph.indices.size} legal "
          f"directed edges at limits ({rover.pitch_max:.0f}, {rover.roll_max:.0f})")

    t0 = time.time()
    sun = sun_table(HORIZON_H)
    cache = DATA / "work" / "intervals" / f"{EPOCH.replace(':', '')}_{HORIZON_H:.0f}h.npz"
    intervals, cached = LitIntervals.build_or_load(cache, cells,
                                                   horizon.astype(np.float64), sun,
                                                   HORIZON_H)
    build_s = time.time() - t0
    print(f"intervals {'loaded from cache' if cached else 'built and cached'}: "
          f"{build_s:.1f} s")
    counts = np.diff(intervals.offsets)
    print(f"lit intervals over {HORIZON_H:.0f} h at 15 min: {build_s:.1f} s, "
          f"{int(counts.sum())} transitions, {counts.mean():.2f} per cell, "
          f"max {counts.max()}; {intervals.lit0.mean():.2%} of cells lit at the epoch")

    ok = legal(edges["pitch"], edges["roll"], rover)
    g = coo_matrix((np.ones(int(ok.sum())), (edges["a"][ok], edges["b"][ok])),
                   shape=(n * n, n * n))
    _, lab = connected_components(g, directed=False)
    big = np.argmax(np.bincount(lab))
    comp = np.flatnonzero(lab == big)
    print(f"largest legal component {comp.size} cells ({comp.size/(n*n):.2%})")
    sel = ok & intervals.lit0[edges["a"]] & intervals.lit0[edges["b"]]
    gl = coo_matrix((np.ones(int(sel.sum())), (edges["a"][sel], edges["b"][sel])),
                    shape=(n * n, n * n))
    _, labl = connected_components(gl, directed=False)
    sizes = np.bincount(labl)
    order = np.argsort(sizes)[::-1]
    lit_comp = np.flatnonzero(labl == order[0]) if sizes[order[0]] > 1 else np.zeros(0, int)
    print(f"lit at the epoch {intervals.lit0.mean():.2%}; largest component of the lit "
          f"subgraph {lit_comp.size} cells ({lit_comp.size/(n*n):.2%})")
    counts_all = np.diff(intervals.offsets)
    changing = counts_all >= 2
    selc = ok & changing[edges["a"]] & changing[edges["b"]]
    gc = coo_matrix((np.ones(int(selc.sum())), (edges["a"][selc], edges["b"][selc])),
                    shape=(n * n, n * n))
    _, labc = connected_components(gc, directed=False)
    sz = np.bincount(labc)
    top = np.argsort(sz)[::-1][0]
    chg_comp = np.flatnonzero(labc == top) if sz[top] > 1 else np.zeros(0, int)
    print(f"cells with at least two transitions {changing.mean():.2%}; largest component "
          f"of that subgraph {chg_comp.size} cells ({chg_comp.size/(n*n):.2%})")
    pool = {"registered": comp, "lit": lit_comp, "changing": chg_comp}[args.draw]
    tag = {"changing": "  [the superseded rule, PLAN.md 2026-09-23]",
           "registered": "  [the original rule, superseded]",
           "lit": "  [not a registered rule; sizing only]"}[args.draw]
    print(f"drawing from the {args.draw} pool, {pool.size} cells" + tag)

    xs, ys = gx.ravel(), gy.ravel()

    def draw(lo, hi, k, label=0):
        """Each draw gets its own stream, so a set does not depend on what ran first."""
        rng = np.random.default_rng([SEED, int(label), int(lo), int(hi)])
        out = []
        while len(out) < k:
            a, b = rng.choice(pool, 2, replace=False)
            d = np.hypot(xs[a] - xs[b], ys[a] - ys[b])
            if lo <= d < hi:
                out.append((int(a), int(b), float(d)))
        return out

    if not args.skip_check3:
      print(f"\n--- check 3, {args.check3} small instances, shadow tolerance "
            f"{SHADOW_TOL_H} h ---")
      print(f"  small means a small subgraph, not only a short distance: the reference "
            f"explores\n  (cell, step, shadow) without a heuristic, so it is run on a box "
            f"around the pair\n  padded by {CHECK3_PAD} cells, over {CHECK3_HORIZON_H:.0f} h, "
            f"and the planner is run on the same box")
      prof64 = horizon.astype(np.float64)

      def dark_oracle(dt):
          def dark(cell, k):
              ts = np.linspace(k * dt, (k + 1) * dt, SUB_STEPS)
              sub = RegionCells(np.array([xs[cell]]), np.array([ys[cell]]),
                                np.array([dem.ravel()[cell]]), radius=p["radius_m"])
              return bool(np.any(g_at(sub, prof64[cell:cell + 1], sun, ts) <= 0.0))
          return dark

      dts = (0.5, 0.25, 0.125, 0.0625)
      rows = []
      for (a, b, d) in draw(100.0, 400.0, args.check3, label=1):
          r0, c0, r1, c1 = a // n, a % n, b // n, b % n
          rlo, rhi = max(0, min(r0, r1) - CHECK3_PAD), min(n, max(r0, r1) + CHECK3_PAD + 1)
          clo, chi = max(0, min(c0, c1) - CHECK3_PAD), min(n, max(c0, c1) + CHECK3_PAD + 1)
          box = np.zeros(n * n, dtype=bool)
          box.reshape(n, n)[rlo:rhi, clo:chi] = True
          keep = box[edges["a"]] & box[edges["b"]] & legal(edges["pitch"], edges["roll"], rover)
          sub = Graph.build({k: (v[keep] if isinstance(v, np.ndarray) else v)
                             for k, v in edges.items()}, rover, n * n)
          t_int, path, st = plan(sub, intervals, a, b, 0.0, SHADOW_TOL_H)
          if t_int is None or t_int > CHECK3_HORIZON_H:
              continue
          graph_c3 = sub
          gaps = []
          for dt in dts:
              cache = {}
              base = dark_oracle(dt)

              def dark(cell, k, _c=cache, _b=base):
                  key = (cell, k)
                  if key not in _c:
                      _c[key] = _b(cell, k)
                  return _c[key]
              t_dt = plan_time_expanded(graph_c3, dark, a, b, 0.0, SHADOW_TOL_H, dt,
                                        CHECK3_HORIZON_H)
              gaps.append(None if t_dt is None else t_dt - t_int)
          rows.append((a, b, d, t_int, len(path), gaps))
          bound_ok = all(g is None or g <= (len(path) - 1) * dd + 1e-12
                         for g, dd in zip(gaps, dts))
          print(f"    gap <= edges x dt on every dt: {'yes' if bound_ok else 'NO'} "
                f"(bounds " + " ".join(f"{(len(path)-1)*dd:.3f}" for dd in dts) + ")")
          print(f"  {a:6d}->{b:6d} {d:7.0f} m  box {rhi-rlo}x{chi-clo}  "
                f"T_int {t_int:7.4f} h, {len(path):3d} cells; "
                f"gaps " + " ".join("none" if x is None else f"{x:7.4f}" for x in gaps))
      print(f"  instances that produced a comparison: {len(rows)} of {args.check3} drawn")
      if not rows:
          print("  VACUOUS: no instance was feasible, so check 3 asserted nothing")
      ok_ineq = all(x is None or x >= -1e-9 for _, _, _, _, _, gs in rows for x in gs)
      mono = all(all(b <= a + 1e-12 for a, b in zip(gs, gs[1:])
                     if a is not None and b is not None) for *_, gs in rows)
      last = [gs[-1] / ti for _, _, _, ti, _, gs in rows if gs[-1] is not None]
      verdict = (lambda ok: "PASS" if ok else "FAIL") if rows else (lambda ok: "VACUOUS")
      print(f"  T_interval <= T_dt + 1e-9 on every instance and dt: {verdict(ok_ineq)}")
      bounded = all(g is None or g <= (pl - 1) * dd + 1e-12
                    for _, _, _, _, pl, gs in rows for g, dd in zip(gs, dts))
      print(f"  replacement clause, gap <= edges x dt at every instance and dt: "
            f"{verdict(bounded)}")
      if last:
          print(f"  gap monotone in dt: {verdict(mono)}; gap at the smallest dt median "
                f"{np.median(last):.4%} of T_interval "
                f"({verdict(np.median(last) < 0.01)} against 1%)")
      else:
          print("  gap at the smallest dt: no instance produced a finite gap")

    print(f"\n--- pilot, {args.instances} instances across the five distance bands ---")
    per = max(args.instances // len(BANDS), 1)
    pairs, dists = [], []
    for lo, hi in BANDS:
        for (a, b, d) in draw(lo, hi, per, label=2):
            pairs.append((a, b))
            dists.append(d)
    dists = np.array(dists)

    py = None
    if args.engine in ("python", "both"):
        rows_py = []
        for (a, b) in pairs:
            t0 = time.time()
            arrival, path, st = plan(graph, intervals, a, b, 0.0, SHADOW_TOL_H,
                                     time_limit_s=LIMIT_S)
            lpc = (np.array(list(st.labels_per_cell.values()))
                   if st.labels_per_cell else np.zeros(1))
            rows_py.append(dict(arrival=np.nan if arrival is None else arrival,
                                aborted=st.timed_out, seconds=time.time() - t0,
                                touched=len(st.labels_per_cell), max_labels=int(lpc.max()),
                                mean_labels=float(lpc.mean()), path_len=len(path)))
        py = rows_py

    cpp = None
    if args.engine in ("cpp", "both"):
        t0 = time.time()
        cpp = run_cpp(graph, intervals, pairs, SHADOW_TOL_H, 0.0, HORIZON_H, LIMIT_S,
                      DATA / "work" / "plan")
        print(f"  C++ wall for all {len(pairs)} instances: {time.time()-t0:.1f} s")

    def report(name, arrival, aborted, seconds, touched, max_labels):
        ok = ~aborted & np.isfinite(arrival)
        unreach = ~aborted & ~np.isfinite(arrival)
        print(f"\n  {name}: {int(ok.sum())} reachable, {int(unreach.sum())} unreachable, "
              f"{int(aborted.sum())} aborted at the {LIMIT_S:.0f} s cap")
        live = seconds[~aborted]
        print(f"    seconds per search, aborts excluded: median {np.median(live):.3f}, "
              f"mean {live.mean():.3f}, max {live.max():.3f}")
        print(f"    seconds per search, aborts at the cap: median "
              f"{np.median(seconds):.3f}, mean {seconds.mean():.3f}")
        print(f"    cells touched median {int(np.median(touched))}, max {int(touched.max())}; "
              f"labels on the worst cell {int(max_labels.max())}")
        return live.mean(), seconds.mean()

    if py is not None:
        arr = np.array([r["arrival"] for r in py])
        ab = np.array([r["aborted"] for r in py])
        sec = np.array([r["seconds"] for r in py])
        tch = np.array([r["touched"] for r in py])
        ml = np.array([r["max_labels"] for r in py])
        for d, r in zip(dists, py):
            tag = ("aborted" if r["aborted"] else "unreachable"
                   if not np.isfinite(r["arrival"]) else f"{r['arrival']:8.3f} h")
            print(f"    {d:7.0f} m  {tag:>11}  {r['seconds']:7.2f} s  touched "
                  f"{r['touched']:6d}  labels/cell mean {r['mean_labels']:6.2f} "
                  f"max {r['max_labels']:5d}")
        mean_live_py, mean_all_py = report("python", arr, ab, sec, tch, ml)

    if cpp is not None:
        for d, a, ab_, s, st in zip(dists, cpp["arrival"], cpp["aborted"],
                                    cpp["seconds"], cpp["stats"]):
            tag = ("aborted" if ab_ else "unreachable" if not np.isfinite(a)
                   else f"{a:8.3f} h")
            print(f"    {d:7.0f} m  {tag:>11}  {s:7.3f} s  touched {st[2]:6d}  "
                  f"labels max {st[3]:5d}  pushed {st[0]}")
        mean_live_cpp, mean_all_cpp = report("c++", cpp["arrival"], cpp["aborted"],
                                             cpp["seconds"], cpp["stats"][:, 2],
                                             cpp["stats"][:, 3])

    if py is not None and cpp is not None:
        arr_py = np.array([r["arrival"] for r in py])
        ab_py = np.array([r["aborted"] for r in py])
        both = ~ab_py & ~cpp["aborted"]
        finite = both & (np.isfinite(arr_py) | np.isfinite(cpp["arrival"]))
        agree_none = both & ~np.isfinite(arr_py) & ~np.isfinite(cpp["arrival"])
        d = np.abs(arr_py[finite] - cpp["arrival"][finite])
        print(f"\n  cross-check, registered tolerance 1e-9 h: {int(both.sum())} of "
              f"{both.size} instances comparable, {int(agree_none.sum())} agree that "
              f"there is no path,\n    {int(finite.sum())} have an arrival on at least "
              f"one side; excluded because either side aborted: "
              f"{int((~both).sum())}")
        if d.size:
            print(f"    worst arrival difference {np.nanmax(d):.3e} h, median "
                  f"{np.nanmedian(d):.3e} h -> "
                  f"{'AGREE' if np.nanmax(d) <= 1e-9 else 'DISAGREE'}")
            pl_py = np.array([r["path_len"] for r in py])[finite]
            print(f"    same path length on {int((pl_py == cpp['path_len'][finite]).sum())} "
                  f"of {int(finite.sum())} (reported, not part of the criterion)")
        # Comparing mean seconds when both engines are censored at the same
        # wall-clock cap measures the cap, not the speed. The ratio is taken
        # per instance, on instances neither engine aborted and where both did
        # enough work for the timer to mean anything.
        sec_py = np.array([r["seconds"] for r in py])
        matched = both & (sec_py > 0.01)
        ratios = sec_py[matched] / np.maximum(cpp["seconds"][matched], 1e-9)
        print(f"    per-instance speedup on {int(matched.sum())} instances neither "
              f"aborted and slower than 10 ms in Python:")
        print(f"      median {np.median(ratios):.0f}x, min {ratios.min():.0f}x, "
              f"max {ratios.max():.0f}x")
        print(f"    (the ratio of mean seconds, {mean_all_py / max(mean_all_cpp, 1e-12):.1f}x, "
              f"is an artefact of the shared cap and is not the speedup)")

    ref = cpp if cpp is not None else None
    mean_s = mean_all_cpp if cpp is not None else mean_all_py
    mean_live = mean_live_cpp if cpp is not None else mean_live_py
    n_td = 4 * 3 * 120 * 12
    cpu_h = (n_td * mean_s + n_td * mean_s * 0.3 + 3 * 120 * mean_s * 0.3) / 3600.0
    print(f"\n  sweep estimate at {mean_s:.3f} s per search "
          f"({'c++' if cpp is not None else 'python'}):")
    print(f"    time-dependent {n_td}, snapshot {n_td}, mean-map {3*120}")
    print(f"    {cpu_h:.1f} CPU-hours, {cpu_h*60/16:.0f} minutes on 16 threads, "
          f"plus one interval build of {build_s:.0f} s per epoch")
    cpu_live = (n_td * mean_live + n_td * mean_live * 0.3
                + 3 * 120 * mean_live * 0.3) / 3600.0
    print(f"    aborted searches are censored at the cap, so this is a lower bound; "
          f"counting only\n    the searches that finished it would be "
          f"{cpu_live:.1f} CPU-hours, {cpu_live*60/16:.0f} minutes, which is the "
          f"optimistic end")
    return 0


if __name__ == "__main__":
    sys.exit(main())
