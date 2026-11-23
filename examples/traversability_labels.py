"""Labels and features for learned traversability: where the planner's
illumination model disagrees with the checker's ray march.

The label is the checker's ray-marched visibility against the planner's
interval table, on the checker's own 60 s cadence, which is the same
comparison src/disagree.py made to set the executive's contradiction horizon.
Nothing here recomputes either side: the world's state comes from the
checker's trajectory dump and the model's from the planner's raw interval
arrays.

Two things stated once, because every number downstream inherits them. The
label is the CHECKER's visibility and not the world's: the checker re-reads
visibility every 60 s, so a lit window shorter than that is counted as
shadow, which makes it strict in a known direction. And the checker shares
the DEM, the projection constants, the sphere radius, the rover parameters
and the sun ephemeris with the planner, so these labels are independent of
the planner's derivation and not of its input data.

Features are restricted to quantities the monitor can get cheaply at
execution time: the current time's exact sun angles, which are one ephemeris
call, and the precomputed 360-azimuth horizon profile and horizon distance,
which are already on disk. The expensive thing, the ray march at a continuous
position, is deliberately absent; that is what the model is a surrogate for.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import examples.disagreement as D                                # noqa: E402
import examples.plan_pilot as P                                  # noqa: E402
from examples.build_intervals import PHASE_OFFSET_DAYS, epochs, sun_table_at  # noqa: E402
from src.cost import Rover                                       # noqa: E402
from src.disagree import CADENCE_H, FlatIntervals, sample_states  # noqa: E402
from src.frames import azimuth_elevation_from_frame, horizon_lookup         # noqa: E402

N_AZIMUTH = 360
FINE_CADENCE_H = 1.0 / 60.0      # the checker's sun table, not the planner's


def step_index(tick_t, tick_index, plan, cadence_h=CADENCE_H):
    """The grid and the plan-step index at each grid point.

    Duplicates the indexing inside sample_states, which returns the states but
    not the index. The duplication is asserted against that function rather
    than trusted: _check_against_sample_states below requires the world and
    model states recomputed from this index to match it exactly.
    """
    tick_t = np.asarray(tick_t, dtype=float)
    if tick_t.size < 2:
        return np.zeros(0), np.zeros(0, np.int64)
    grid = np.arange(tick_t[0], tick_t[-1], cadence_h)
    if grid.size == 0:
        return np.zeros(0), np.zeros(0, np.int64)
    seg = np.clip(np.searchsorted(tick_t, grid, side="right") - 1,
                  0, tick_t.size - 1)
    idx = np.clip(np.asarray(tick_index, dtype=np.int64)[seg], 0, len(plan) - 1)
    return grid, idx


def from_sample_once(rule="bounded"):
    """The monitored plans, once each.

    D.from_sample iterates every monitored record in results.json, and that
    file holds one record per plan PER RULE, so it yields all 45 plans twice.
    Taking labels from it unfiltered gave 90 plans and would have put the same
    plan on both sides of every split. The leg-0 world trajectory does not
    depend on the monitor rule, so one rule's records carry all of it.
    """
    import json as _json
    for r in _json.loads(D.RESULTS.read_text()):
        if not r["monitored"] or r["rule"] != rule:
            continue
        w = Path(r["work"]) / "leg0_world"
        yield (r["tag"], r["graph"], r["plan"],
               np.fromfile(w / "out_traj_t.bin", np.float64),
               np.fromfile(w / "out_traj_lit.bin", np.uint8),
               np.fromfile(w / "out_traj_index.bin", np.int32))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=("held-out", "sample"), default="sample")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    rover = Rover(v_flat=50.0)
    p, n, px, dem, horizon, cells, gx, gy, edges, graph = P.load(rover)
    horizon = horizon.astype(np.float64).reshape(-1, N_AZIMUTH)
    hdist = np.fromfile(ROOT / "data" / "work" / "horizon_dist.f32",
                        dtype=np.float32).reshape(-1, N_AZIMUTH)
    base_off = {tag: (base, off) for tag, base, off in epochs()}

    src = (from_sample_once() if args.source == "sample"
           else D.from_held_out(ROOT / "data" / "work" / "disagree_labels"))

    suns, ivs = {}, {}
    cols = {k: [] for k in ("tag_id", "cell", "t", "world", "model",
                            "sun_el", "sun_az", "hor_el", "hor_dist",
                            "margin", "b3a", "b3b", "nxt_lit", "nxt_cell",
                            "plan_id", "frac", "start", "goal")}
    tag_ids, plans_done, bad = {}, 0, 0
    for tag, gp, plan, tt, tl, ti in src:
        if tag not in ivs:
            ivs[tag] = FlatIntervals(gp["itimes_path"], gp["ioffsets_path"],
                                     gp["ilit0_path"])
            b, o = base_off[tag]
            suns[tag] = sun_table_at(b, o, cadence_h=FINE_CADENCE_H)
            tag_ids.setdefault(tag, len(tag_ids))
        iv, sun = ivs[tag], suns[tag]
        grid, world, model = sample_states(tt, tl, ti, plan, iv)
        if grid.size == 0:
            continue
        g2, idx = step_index(tt, ti, plan)
        assert g2.size == grid.size and np.allclose(g2, grid)
        # the duplication self-check: states rebuilt from idx must match
        reb = np.array([iv.is_lit(int(plan[i]), float(t))
                        for i, t in zip(idx, grid)], bool)
        if not np.array_equal(reb, model):
            bad += 1
            continue
        plan = np.asarray(plan, dtype=np.int64)
        cell = plan[idx]
        nxt = plan[np.minimum(idx + 1, len(plan) - 1)]
        pid = plans_done
        # Vectorised per plan. azimuth_elevation_from_frame sums on the last
        # axis, so the whole trajectory goes through in one call with the
        # cells' own hoisted frames, which is also the only implementation of
        # the convention in this repository.
        spos = np.array([sun.at(float(t)) for t in grid], dtype=float)
        az, el = azimuth_elevation_from_frame(
            spos, cells.origin[cell], cells.up[cell],
            cells.east[cell], cells.north[cell])
        he = horizon_lookup(horizon[cell], az)
        hd = hdist[cell, np.round(az).astype(int) % N_AZIMUTH]
        # B3a: the same horizon criterion the planner's table was built from,
        # evaluated at the exact time instead of read off a 15-minute grid.
        b3a = el > he
        # B3b: B3a also required at the next cell on the path, the cheapest
        # stand-in for the rover's continuous sub-cell position.
        az2, el2 = azimuth_elevation_from_frame(
            spos, cells.origin[nxt], cells.up[nxt],
            cells.east[nxt], cells.north[nxt])
        # The next cell's own state, kept separately from the conjunction.
        # B3b stored only b3a AND next-lit, from which the next cell's state is
        # unrecoverable when the current cell is dark, which is precisely the
        # population the lit direction is about.
        nxt_lit = el2 > horizon_lookup(horizon[nxt], az2)
        b3b = b3a & nxt_lit
        cols["sun_el"].append(el); cols["sun_az"].append(az)
        cols["hor_el"].append(he); cols["hor_dist"].append(hd.astype(float))
        cols["margin"].append(el - he)
        cols["b3a"].append(b3a); cols["b3b"].append(b3b)
        cols["nxt_lit"].append(nxt_lit); cols["nxt_cell"].append(nxt)
        cols["cell"].append(cell); cols["t"].append(grid)
        cols["tag_id"].append(np.full(grid.size, tag_ids[tag], np.int64))
        cols["plan_id"].append(np.full(grid.size, pid, np.int64))
        cols["frac"].append(np.zeros(grid.size))
        # The plan's start/goal pair, so a model can be trained without the
        # pairs a host plans (traversability_model.py --exclude-host).
        cols["start"].append(np.full(grid.size, plan[0], np.int64))
        cols["goal"].append(np.full(grid.size, plan[-1], np.int64))
        cols["world"].append(world)
        cols["model"].append(model)
        plans_done += 1
        print(f"  {tag} plan {pid}: {grid.size} samples", flush=True)

    out = Path(args.out) if args.out else (
        ROOT / "data" / "work" / f"traversability_{args.source}.npz")
    out.parent.mkdir(parents=True, exist_ok=True)
    arrs = {k: (np.concatenate(v) if v else np.zeros(0))
            for k, v in cols.items()}
    arrs["tag_names"] = np.array(sorted(tag_ids, key=tag_ids.get))
    np.savez_compressed(out, **arrs)
    w = arrs["world"].astype(bool); m = arrs["model"].astype(bool)
    print(f"\nsource {args.source}: {plans_done} plans, {w.size} samples, "
          f"{len(tag_ids)} epochs, {bad} plans dropped by the self-check")
    print(f"  world dark & model lit : {int((~w & m).sum())} "
          f"({100*(~w & m).mean():.3f}%)")
    print(f"  world lit & model dark : {int((w & ~m).sum())} "
          f"({100*(w & ~m).mean():.3f}%)")
    print(f"  any disagreement       : {int((w != m).sum())} "
          f"({100*(w != m).mean():.3f}%)")
    print(f"  wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
