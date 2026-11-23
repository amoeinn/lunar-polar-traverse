"""The demo traverse: a 2D map animation and a Gazebo animation of one plan.

Reads and never recomputes. Every plan, verdict, arrival and accumulated shadow
comes from the stored outputs of the committed sweep and executive runs, so this
cannot disagree with the record by re-deriving it. The one thing it does run is
the committed replay checker, and only over prefixes of a plan it already has,
to ask the checker where a plan it already rejected first went over budget. That
question has no stored answer: the sweep records the per-plan maximum and not
the step it occurred at.

The instance selection rules R1 to R4 are registered in PLAN.md on 2026-10-05,
before any candidate had been examined, and are implemented here exactly. Every
filter's candidate count is printed and written to the sidecar, so a reader can
check that the filters and not a preference chose the instance.

What the artifacts may and may not be read as is in PLAN.md under "Where it
could mislead", and the same nine items are drawn on the figures themselves.
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import examples.plan_pilot as P                                      # noqa: E402
from examples.build_intervals import cache_path, epochs              # noqa: E402
from examples.illumination_texture import LIT_FLOOR, irradiance      # noqa: E402
from src.cost import Rover                                           # noqa: E402
from src.gzmap import HEIGHTMAP_N, PX, REGION_N, REGION_OX, REGION_OY  # noqa: E402
from src.gzmap import bilinear, cell_to_map, map_to_world             # noqa: E402
from src.intervals import CADENCE_H, LitIntervals                     # noqa: E402
from src.replay import ReplaySettings, replay                         # noqa: E402

SWEEP = ROOT / "data" / "work" / "sweep"
# Recorded results the captions quote. Read, never restated: a caption that
# carried its numbers as literals drew pre-fix figures over corrected data.
SCORES = ROOT / "data" / "work" / "phase3_scores.json"
C3_RECORD = ROOT / "data" / "work" / "gazebo" / "verify" / "c3.json"
CORR = ROOT / "data" / "work" / "executive_correction"
OUT = ROOT / "data" / "work" / "demo"

BUDGET_H = 2.0              # the monitored configuration's shadow budget
ARRIVAL_MIN_H = 24.0        # R1's clause, which is P2's own registered regime
MEANMAP_ARM = "mm2.0"       # the mean-map baseline at its reported best exponent
# Nearest-neighbour replication of the illumination texture, measured 2026-10-06
# and recorded in PLAN.md. Each 20 m cell becomes an 8 x 8 block of identical
# texels, so the renderer's filter can only blend at block edges. Costs 0.37 s
# per frame to write and 0.11 MB per PNG against 0.01 s and 0.02 MB. Defined
# here rather than beside the other render constants because texture_png takes
# it as a default argument, which Python binds at definition time.
UPSAMPLE = 8
V_FLAT = 50.0               # the monitored configuration's rate

# The keys examples/sweep_core.py puts in the replay checker's `grid`. Read back
# from the stored replay_params.json rather than rebuilt, so a prefix replay runs
# against the same geometry the published verdicts were produced with.
GRID_KEYS = ("radius_m", "region_ox", "region_oy", "region_px", "region_nx",
             "near_path", "near_ox", "near_oy", "near_px", "near_width",
             "near_height", "far_path", "far_ox", "far_oy", "far_px",
             "far_width", "far_height", "reach_m", "switch_m", "near_step_m",
             "far_step_m")


# --------------------------------------------------------------------------
# read-only accessors over the stored sweep and executive outputs
# --------------------------------------------------------------------------

def epoch_tags():
    """The registered enumeration order: epochs() as build_intervals returns it."""
    return [t for t, _, _ in epochs()]


def planner_out(tag, kind):
    """What the planner binary wrote for one epoch and planner, plus its pairs."""
    d = SWEEP / f"{tag}_{kind}"
    pj = json.loads((d / "planner_params.json").read_text())
    off = np.fromfile(pj["out_path_offsets"], np.int64)
    return dict(
        params=pj,
        pairs=np.fromfile(d / "pairs.bin", np.int32).reshape(-1, 2),
        offsets=off,
        cells=np.fromfile(pj["out_path_cells"], np.int32),
        times=np.fromfile(pj["out_path_times"], np.float64),
        arrival=np.fromfile(pj["out_arrival"], np.float64),
        aborted=np.fromfile(pj["out_flags"], np.uint8).astype(bool),
        # The replay stage kept only plans of two cells or more, in pair order.
        # That list is the map from a replay row back to its instance, and it is
        # not stored, so it is rebuilt here the same way sweep_core built it.
        index=[q for q in range(off.size - 1) if off[q + 1] - off[q] >= 2])


def has_replay(tag, kind, label="tol"):
    """Whether the sweep judged any plan here.

    An epoch where a planner produced nothing was never handed to the checker,
    so the directory does not exist. 2028-05-15_p0 is one: the time-dependent
    planner returned 0 feasible plans of 120 instances there.
    """
    return (SWEEP / f"{tag}_replay_{kind}_{label}"
            / "plan_offsets_path.bin").exists()


def replay_out(tag, kind, label="tol"):
    """What the replay checker wrote, with the plans it was given."""
    d = SWEEP / f"{tag}_replay_{kind}_{label}"
    off = np.fromfile(d / "plan_offsets_path.bin", np.int64)
    return dict(
        dir=d,
        params=json.loads((d / "replay_params.json").read_text()),
        offsets=off,
        cells=np.fromfile(d / "plan_cells_path.bin", np.int32),
        starts=np.fromfile(d / "plan_start_path.bin", np.float64),
        waits=np.fromfile(d / "plan_wait_path.bin", np.float64),
        verdict=np.fromfile(d / "out_verdict.bin", np.uint8).astype(bool),
        arrival=np.fromfile(d / "out_arrival.bin", np.float64),
        shadow=np.fromfile(d / "out_shadow.bin", np.float64),
        illegal=np.fromfile(d / "out_illegal.bin", np.int32))


def assert_drivable(tag, plan_index):
    """C5: only a plan the checker accepted may be played back as a drive.

    Asserted from the checker's own out_verdict.bin, not from a filename, which
    is what C5 asks for in those words. A rejected plan has no feasible schedule
    to animate, so the refusal is the honest outcome and inventing a trajectory
    for it would be the exact substitution C5 exists to forbid.
    """
    rep = replay_out(tag, "td")
    if plan_index >= rep["verdict"].size:
        raise SystemExit(f"{tag} plan{plan_index:03d}: no such replay row")
    if not bool(rep["verdict"][plan_index]):
        raise SystemExit(
            f"{tag} plan{plan_index:03d}: the replay checker rejected this plan "
            f"(shadow {rep['shadow'][plan_index]:.4f} h against a "
            f"{BUDGET_H:.1f} h budget, illegal step "
            f"{int(rep['illegal'][plan_index])}); C5 forbids driving it")
    return True


def plan_at(rep, row):
    """One plan, its waits and its departure, out of a replay directory."""
    lo, hi = int(rep["offsets"][row]), int(rep["offsets"][row + 1])
    return (rep["cells"][lo:hi].astype(np.int64),
            rep["waits"][lo:hi].astype(np.float64),
            float(rep["starts"][row]))


# --------------------------------------------------------------------------
# the registered rules R1 to R4, implemented exactly as written
# --------------------------------------------------------------------------

def select_r1():
    """R1: snapshot rejected, time-dependent accepted, arrival over 24 h.

    Ordering is positional and nothing else enters: epochs in epochs() order,
    then instance index ascending, first match taken. The arrival used is the
    checker's, since the checker is the authority on a plan it accepted; the
    planner's is reported beside it.
    """
    counts = dict(epochs=0, epochs_judged=0, snap_plans=0, snap_rejected=0,
                  td_accepted_same_pair=0, over_24h=0)
    cands = []
    for tag in epoch_tags():
        if not (SWEEP / f"{tag}_td" / "planner_params.json").exists():
            continue
        counts["epochs"] += 1
        if not (has_replay(tag, "td") and has_replay(tag, "snap")):
            continue
        counts["epochs_judged"] += 1
        td_p, sn_p = planner_out(tag, "td"), planner_out(tag, "snap")
        td_r, sn_r = replay_out(tag, "td"), replay_out(tag, "snap")
        assert len(td_p["index"]) == td_r["verdict"].size, (tag, "td rows")
        assert len(sn_p["index"]) == sn_r["verdict"].size, (tag, "snap rows")
        td_row = {q: i for i, q in enumerate(td_p["index"])}
        counts["snap_plans"] += len(sn_p["index"])
        for s_row, q in enumerate(sn_p["index"]):
            if sn_r["verdict"][s_row]:
                continue
            counts["snap_rejected"] += 1
            t_row = td_row.get(q)
            if t_row is None or not td_r["verdict"][t_row]:
                continue
            counts["td_accepted_same_pair"] += 1
            arr = float(td_r["arrival"][t_row])
            if arr <= ARRIVAL_MIN_H:
                continue
            counts["over_24h"] += 1
            cands.append(dict(tag=tag, instance=int(q), td_row=t_row,
                              snap_row=s_row, arrival_checker=arr,
                              arrival_planner=float(td_p["arrival"][q])))
    return (cands[0] if cands else None), counts, cands


def pair_contingency():
    """The two planners' replay verdicts, paired per instance.

    R1 reports only the count it needs, "snapshot rejected and time-dependent
    accepted", which is zero. That number alone does not say whether the two
    planners disagree in the OTHER direction either, nor what became of the
    instances where only one of them produced a plan. This prints the whole
    table, because the aggregate survival rates, 13.7% against 59.3%, read as a
    feasibility advantage and the paired table says they are not one.
    """
    both = dict(aa=0, ar=0, ra=0, rr=0)
    only_td = only_snap = 0
    unsolved = dict(aborted=0, infeasible=0)
    for tag in epoch_tags():
        if not (SWEEP / f"{tag}_td" / "planner_params.json").exists():
            continue
        if not (has_replay(tag, "td") and has_replay(tag, "snap")):
            continue
        td_p, sn_p = planner_out(tag, "td"), planner_out(tag, "snap")
        td_r, sn_r = replay_out(tag, "td"), replay_out(tag, "snap")
        tdrow = {q: i for i, q in enumerate(td_p["index"])}
        snrow = {q: i for i, q in enumerate(sn_p["index"])}
        only_td += sum(1 for q in tdrow if q not in snrow)
        only_snap += sum(1 for q in snrow if q not in tdrow)
        for q in set(tdrow) & set(snrow):
            a = bool(sn_r["verdict"][snrow[q]])
            b = bool(td_r["verdict"][tdrow[q]])
            both["aa" if (a and b) else "ar" if a else "ra" if b else "rr"] += 1
        for q in snrow:
            if sn_r["verdict"][snrow[q]] or q in tdrow:
                continue
            # The cast to bool matters: out_flags is uint8 on disk.
            unsolved["aborted" if bool(td_p["aborted"][q]) else "infeasible"] += 1
    n = sum(both.values())
    print(f"\nthe two planners' verdicts, paired on the {n} instances where both "
          f"produced a plan:")
    print(f"  snapshot accepted, time-dependent accepted   {both['aa']:4d}")
    print(f"  snapshot accepted, time-dependent REJECTED   {both['ar']:4d}")
    print(f"  snapshot REJECTED, time-dependent accepted   {both['ra']:4d}")
    print(f"  snapshot REJECTED, time-dependent REJECTED   {both['rr']:4d}")
    print(f"  instances with only a time-dependent plan    {only_td:4d}")
    print(f"  instances with only a snapshot plan          {only_snap:4d}")
    print(f"\n  of the rejected snapshot plans with no time-dependent plan:")
    print(f"    the planner declared no feasible path      "
          f"{unsolved['infeasible']:4d}")
    print(f"    the planner hit its 30 s cap               "
          f"{unsolved['aborted']:4d}")
    return dict(both=both, only_td=only_td, only_snap=only_snap,
                unsolved=unsolved)


def _corr_records():
    """The executive's five-arm records, keyed by arm then (tag, plan index)."""
    recs = json.loads((CORR / "results.json").read_text())
    order = {t: k for k, t in enumerate(epoch_tags())}
    arms = {}
    for r in recs:
        arms.setdefault(r["kind"], {})[(r["tag"], r["plan_index"])] = r
    keys = sorted(arms["none"], key=lambda k: (order[k[0]], k[1]))
    return arms, keys


def select_r2():
    """R2: the busts among the monitored core plans, first in enumeration order.

    Can be empty, and is on the corrected core: an empty R2 is reported as an
    empty set by render_panels and nothing is substituted for it.
    """
    arms, keys = _corr_records()
    cands = [dict(tag=t, plan_index=i)
             for (t, i) in keys
             if arms["none"][(t, i)]["monitored"]
             and arms["none"][(t, i)]["busted"]]
    counts = dict(monitored=sum(1 for k in keys if arms["none"][k]["monitored"]),
                  busts=len(cands))
    return (cands[0] if cands else None), counts, cands


def select_r3():
    """R3: a monitored non-bust the bounded rule fires on and B3b-lit clears."""
    arms, keys = _corr_records()
    cands = []
    for k in keys:
        base = arms["none"][k]
        if not base["monitored"] or base["busted"]:
            continue
        if base["fired"] and not arms["lit"][k]["fired"]:
            cands.append(dict(tag=k[0], plan_index=k[1]))
    nonbust = [k for k in keys
               if arms["none"][k]["monitored"] and not arms["none"][k]["busted"]]
    counts = dict(monitored_nonbust=len(nonbust),
                  false_alarms=sum(1 for k in nonbust if arms["none"][k]["fired"]),
                  removed_by_lit=len(cands))
    return (cands[0] if cands else None), counts, cands


def select_r4():
    """R4: no selection. Every epoch, with what the record already reports."""
    rows = []
    for tag in epoch_tags():
        if not (SWEEP / f"{tag}_td" / "planner_params.json").exists():
            rows.append(dict(tag=tag, present=False))
            continue
        td_p, sn_p = planner_out(tag, "td"), planner_out(tag, "snap")
        iv = LitIntervals.load(cache_path(tag))
        if not (has_replay(tag, "td") and has_replay(tag, "snap")):
            rows.append(dict(tag=tag, present=True, judged=False,
                             lit_t0=float(iv.lit0.mean()),
                             td_plans=len(td_p["index"]),
                             snap_plans=len(sn_p["index"]),
                             aborted=int(td_p["aborted"].sum())))
            continue
        td_r, sn_r = replay_out(tag, "td"), replay_out(tag, "snap")
        rows.append(dict(tag=tag, present=True, judged=True,
                         lit_t0=float(iv.lit0.mean()),
                         td_plans=len(td_p["index"]),
                         td_pass=int(td_r["verdict"].sum()),
                         snap_plans=len(sn_p["index"]),
                         snap_pass=int(sn_r["verdict"].sum()),
                         aborted=int(td_p["aborted"].sum())))
    return rows


# --------------------------------------------------------------------------
# the checker, asked where a plan first went over budget
# --------------------------------------------------------------------------

def shadow_curve(rep, row, work):
    """Accumulated continuous shadow against prefix length, from the checker.

    The sweep stores the per-plan maximum and not where it occurred, so the
    curve is obtained by handing the committed checker every prefix of the plan
    in one batch. The quantity is a maximum over the traverse, so it is monotone
    in prefix length, and the last point must reproduce the stored value exactly.
    That equality is asserted: it checks the whole read-back chain at once, the
    plan, the waits, the departure and the sun table.
    """
    plan, wait, start = plan_at(rep, row)
    if plan.size < 2:
        raise SystemExit("a plan of under two cells has no prefix curve")
    grid = {k: rep["params"][k] for k in GRID_KEYS}
    sun = np.fromfile(rep["dir"] / "sun_path.bin", np.float64).reshape(-1, 3)
    rover = Rover(v_flat=V_FLAT)
    for k, v in (("v_flat", rover.v_flat), ("pitch_max", rover.pitch_max),
                 ("roll_max", rover.roll_max)):
        assert abs(float(rep["params"][k]) - v) < 1e-9, (k, rep["params"][k], v)
    lengths = list(range(2, plan.size + 1))
    out = replay([plan[:L] for L in lengths], [start] * len(lengths), grid, rover,
                 sun, float(rep["params"]["sun_dt_hours"]), BUDGET_H, Path(work),
                 ReplaySettings(
                     limit_tolerance_deg=float(rep["params"]["limit_tolerance_deg"])),
                 waits=[wait[:L] for L in lengths])
    curve = out["shadow"]
    stored = float(rep["shadow"][row])
    assert abs(float(curve[-1]) - stored) < 1e-9, (float(curve[-1]), stored)
    over = np.flatnonzero(curve > BUDGET_H)
    return dict(lengths=np.asarray(lengths), shadow=curve,
                arrival=out["arrival"], stored_shadow=stored,
                first_over=int(lengths[int(over[0])]) if over.size else None,
                illegal=int(rep["illegal"][row]))


# --------------------------------------------------------------------------
# illumination fields, one per frame, on one fixed normalisation
# --------------------------------------------------------------------------

def frame_times(t_dep, t_arr):
    """One frame per sun-table sample, so no frame shows an unevaluated state."""
    k0 = int(np.floor(t_dep / CADENCE_H))
    k1 = int(np.ceil(t_arr / CADENCE_H))
    return np.arange(k0, k1 + 1) * CADENCE_H


def field_stack(tag, times, path):
    """Our model's irradiance per cell at every frame time, as a memmap.

    One fixed normalisation constant for the whole sequence, which is the point
    of building the stack before writing any image: the committed texture writer
    normalises each frame by that frame's own peak, correct frame by frame and
    wrong in a sequence, because the sun elevation moves over a traverse and the
    picture would hold its brightness while the irradiance fell.
    """
    p, n, px, dem, horizon, cells, gx, gy, edges, _ = P.load(Rover(v_flat=V_FLAT))
    iv = LitIntervals.load(cache_path(tag))
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    mpath = path.with_name(path.stem + "_lit.npy")
    mm = np.lib.format.open_memmap(path, mode="w+", dtype=np.float32,
                                   shape=(times.size, n, n))
    ml = np.lib.format.open_memmap(mpath, mode="w+", dtype=np.uint8,
                                   shape=(times.size, n, n))
    lit_frac, peak = np.zeros(times.size), 0.0
    for k, t in enumerate(times):
        lit = np.asarray([iv.is_lit(c, float(t)) for c in range(n * n)], dtype=bool)
        cosi = irradiance(tag, float(t), dem, gx, gy, p["radius_m"])
        f = np.where(lit, cosi, 0.0).reshape(n, n)
        mm[k] = f.astype(np.float32)
        # The mask is stored, not recovered from the field being positive. At
        # this epoch the sun is about a degree up and cos(incidence) clips to
        # exactly zero on most lit cells (the map caption counts them per run),
        # so "field > 0" would call most of the lit region dark and the picture
        # would be the inverse of the model. The committed texture writer floors by the mask for the
        # same reason and this keeps that convention rather than re-deriving it.
        ml[k] = lit.reshape(n, n).astype(np.uint8)
        lit_frac[k] = float(lit.mean())
        peak = max(peak, float(f.max()))
    mm.flush()
    ml.flush()
    return dict(path=str(path), lit_path=str(mpath), peak=peak,
                lit_fraction=lit_frac, dem=dem, n=n, radius_m=p["radius_m"])


def texture_png(field, litmask, peak, out, upsample=UPSAMPLE):
    """One unlit texture, on the sequence's fixed normalisation.

    Identical in convention to examples/illumination_texture.py, including the
    floor that keeps a lit cell from quantising to zero and the 13 extra rows
    and columns written dark rather than extrapolated, and differing from it in
    two ways: peak is the sequence's rather than the frame's, and each cell is
    replicated into an upsample x upsample block of identical texels.

    The replication is what makes the chase view readable. sdformat exposes no
    filtering option, but a filter blends only between NEIGHBOURING texels, so
    replicating a cell into a block leaves nothing to blend except at the block
    edges and the blend shrinks from one 20 m texel to one 20/upsample m texel.
    Nothing is interpolated and no value is invented: every texel in a block is
    the cell's own value, so the texture carries exactly what the 513 x 513 one
    did at eight times the sampling.

    The footprint does not change with it. The world's <texture><size> element
    is 10240 m, computed from the span and not from the pixel count, so a larger
    image maps to the same ground.
    """
    import matplotlib.image as mpimg
    tex = np.zeros((HEIGHTMAP_N, HEIGHTMAP_N), dtype=np.float64)
    tex[:REGION_N, :REGION_N] = field
    img = np.zeros_like(tex) if peak <= 0 else tex / peak
    litmap = np.zeros_like(tex, dtype=bool)
    litmap[:REGION_N, :REGION_N] = np.asarray(litmask, bool)
    img = np.where(litmap, np.maximum(img, LIT_FLOOR), 0.0)
    arr = (np.clip(img, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)
    if upsample > 1:
        arr = np.repeat(np.repeat(arr, upsample, axis=0), upsample, axis=1)
    mpimg.imsave(out, arr, cmap="gray", vmin=0, vmax=255)
    return out


# --------------------------------------------------------------------------
# geometry helpers shared by both artifacts
# --------------------------------------------------------------------------

def cells_to_map(cells, n=REGION_N):
    i, j = np.asarray(cells) // n, np.asarray(cells) % n
    return cell_to_map(i, j)


def step_at(times, t):
    """The plan step the rover is on at time t: the last one already departed."""
    k = int(np.searchsorted(np.asarray(times, float), float(t), side="right") - 1)
    return max(0, min(k, len(times) - 1))


# --------------------------------------------------------------------------
# the 2D map view, which is the load-bearing artifact
# --------------------------------------------------------------------------

def p3_record(path=None):
    """P3 as examples/score_phase3.py recorded it."""
    path = SCORES if path is None else Path(path)
    if not path.exists():
        raise SystemExit(f"{path} is missing; the captions quote P3 from it. Run "
                         f"examples/score_phase3.py --parts core,p2axis,p7")
    return json.loads(path.read_text())["P3"]


def c3_record(path=None):
    """C3's measured figures as examples/verify_gazebo.py checkpointed them."""
    path = C3_RECORD if path is None else Path(path)
    d = json.loads(path.read_text()) if path.exists() else {}
    if "C3" not in d.get("figures", {}):
        raise SystemExit(f"{path} holds no C3 figures; run examples/verify_gazebo.py "
                         f"--parts c3, which checkpoints them")
    return d["figures"]["C3"]


def p3_sentence(p3):
    """P3 clause 2, its verdict read from the record and every number with it."""
    c = p3["clause2"]
    return (f"P3 clause 2 is scored {c['verdict']}: {c['at_margin']} of the {c['n']} "
            f"cases where both plans are feasible arrive at least "
            f"{p3['margin_required']:.0%} earlier, against "
            f"{p3['share_required']:.0%} required; median improvement "
            f"{c['median']:+.2%}.")


def caption_lines(clip_zero, lit_cells, frames, p3, shadow_h, same_snap):
    """The map frames' caption, every figure from the run or the record."""
    most = "most" if 2 * clip_zero > lit_cells else "some"
    if shadow_h < CADENCE_H:
        res = (f"the frame cadence cannot resolve the {shadow_h:.2f} h of "
               f"continuous shadow the checker integrated.")
    else:
        res = (f"the {shadow_h:.2f} h of continuous shadow the checker integrated "
               f"spans several frames, and darkness shorter than a frame is not "
               f"shown.")
    return (
        "Shadows are our illumination model's, not Gazebo's. The greyscale reads as "
        f"lit or dark rather than as brightness: over the {frames} frames "
        f"cos(incidence) clips to zero on {clip_zero} of {lit_cells} lit cell-frames, "
        f"so {most} lit cells sit exactly on the {LIT_FLOOR:.2f} floor and carry no "
        "shading.",
        p3_sentence(p3) + (" At this instance the snapshot planner returned the "
                           "same route, cell for cell." if same_snap else ""),
        f"One epoch is one geometry. Frames are {CADENCE_H * 60:.0f} min apart; a "
        f"cell lit at two consecutive samples can be dark between them, and " + res,
    )

C_TD, C_SNAP, C_MM, C_ROVER = "#36d7e0", "#ff9436", "#d86bff", "#ffe14d"

# Figure geometry. The region map is square data, so its box is square in
# inches: a 16 x 9 figure and a height of 0.78 give a width of 0.78 * 9 / 16.
# Two rows. The two maps are square boxes because their data windows are square,
# so neither is stretched and neither is letterboxed into a band of empty figure.
REGION_RECT = [0.025, 0.495, 0.236, 0.420]
ZOOM_RECT = [0.300, 0.495, 0.236, 0.420]
EXEC_RECT = [0.600, 0.495, 0.375, 0.420]
# Row 2 sits higher than row 1's mirror would put it, because the epoch strip's
# rotated tick labels hang below its box and ran into the caption.
EPOCH_RECT = [0.025, 0.170, 0.330, 0.245]
SHADOW_RECT = [0.420, 0.170, 0.235, 0.245]
TEXT_RECT = [0.700, 0.150, 0.275, 0.265]
C_LIT_ARM, C_BUDGET, C_FIRE = "#7ee787", "#ff5c5c", "#ff5c5c"


def route_rows(tag, instance):
    """The three planners' rows for one instance, where each produced a plan."""
    out = {}
    td = planner_out(tag, "td")
    out["td"] = td["index"].index(instance) if instance in td["index"] else None
    sn = planner_out(tag, "snap")
    out["snap"] = sn["index"].index(instance) if instance in sn["index"] else None
    mm = replay_out(tag, MEANMAP_ARM)
    # The mean-map planner produced a plan for every instance, so its replay row
    # is the instance index. Asserted rather than assumed, since a short plan
    # would have been dropped and shifted every row after it.
    assert mm["offsets"].size - 1 == 120, (tag, mm["offsets"].size - 1)
    out["mm"] = instance
    # The executive enumerates the same feasible list the td replay rows come
    # from, so its plan_index is the td row. Asserted against the committed
    # five-arm run rather than assumed, because the two indices being equal is a
    # property of how both were built and not a guarantee.
    out["exec_plan_index"] = out["td"]
    if out["td"] is not None and (CORR / "results.json").exists():
        recs = json.loads((CORR / "results.json").read_text())
        hit = [r for r in recs
               if r["tag"] == tag and r["plan_index"] == out["td"]
               and r["kind"] == "none"]
        assert len(hit) == 1, (tag, out["td"], len(hit))
    return out


def panel_lines(tag, instance, rows, curves, same=(), pc=None, p3=None):
    """The checker's own output for every planner, verbatim, as panel text.

    The pairing and P3 lines are read from the selection's contingency table and
    the recorded P3 score, not written in.
    """
    lines = [f"epoch {tag}   instance {instance}", ""]
    lines.append(f"{'planner':<14}{'verdict':>9}{'arrive':>9}{'shadow':>8}")
    for key, label, kind in (("td", "time-dependent", "td"),
                             ("snap", "snapshot", "snap"),
                             ("mm", "mean-map e=2", MEANMAP_ARM)):
        r = rows[key]
        if r is None:
            lines.append(f"{label:<14}{'no plan':>9}{'-':>9}{'-':>8}")
            continue
        rep = replay_out(tag, kind)
        v = "ACCEPTED" if rep["verdict"][r] else "REJECTED"
        lines.append(f"{label:<14}{v:>9}{rep['arrival'][r]:>9.2f}"
                     f"{rep['shadow'][r]:>8.3f}")
    lines += ["", f"shadow budget{BUDGET_H:>8.2f} h"]
    c = curves.get("td")
    if c and c["first_over"] is not None:
        lines.append(f"over budget from step {c['first_over']}")
    if "snap" in same:
        both_ok = all(rows[k] is not None and bool(replay_out(tag, kd)["verdict"][rows[k]])
                      for k, kd in (("td", "td"), ("snap", "snap")))
        lines += ["", "the snapshot planner returned the SAME route,",
                  "cell for cell" + (", and the checker accepted both."
                                     if both_ok else ".")]
    if pc is not None:
        b = pc["both"]
        n, split = sum(b.values()), b["ar"] + b["ra"]
        lines += ["", f"paired on {n} shared pairs: {b['aa']} both",
                  f"accepted, {b['rr']} both rejected, {split} one only."]
        if split == 0:
            lines += ["The planners never disagree on feasibility."]
    if p3 is not None:
        c = p3["clause2"]
        lines += ["", f"P3 clause 2 is scored {c['verdict']}:",
                  f"{c['at_margin']} of {c['n']} cases at least "
                  f"{p3['margin_required']:.0%} earlier,",
                  f"{p3['share_required']:.0%} required; median {c['median']:+.2%}."]
    return lines


def dark_clusters(litm, i0, i1, j0, j1, min_frames=1):
    """Connected groups of cells inside the zoom that are dark in some frame.

    Named on the figure because a black or grey square reads as a hole or a
    missing tile and is neither. None of them is called a permanent shadow:
    27.66 h is not a lunation, and all that is established is how many frames of
    THIS traverse each was dark for.

    The mask is cast to bool first. litm is uint8 on disk and "~" on a uint8
    array is a bitwise NOT, so ~0 is 255 rather than True, and a count taken
    that way comes out in the tens of thousands against 112 frames. That is a
    mistake this file made once in a scratch measurement and it is worth the
    line: the wrong answer was 250 times too large and still looked like a
    count.
    """
    from scipy import ndimage
    sub = np.asarray(litm[:, i0:i1 + 1, j0:j1 + 1]).astype(bool)
    n = sub.shape[0]
    ndark = (~sub).sum(axis=0)
    lab, k = ndimage.label(ndark >= min_frames)
    out = []
    for c in range(1, k + 1):
        ii, jj = np.nonzero(lab == c)
        gx, gy = cell_to_map(ii + i0, jj + j0)
        nd = ndark[ii, jj]
        out.append(dict(cells=int(ii.size), x=float(gx.mean()) / 1000.0,
                        y=float(gy.mean()) / 1000.0, lo=int(nd.min()),
                        hi=int(nd.max()), frames=int(n),
                        always=int((nd == n).sum())))
    out.sort(key=lambda d: (-d["hi"], -d["cells"]))
    return out


def render_frames(tag, instance, times, stack, rows, curves, outdir, dpi=110,
                  pc=None, p3=None):
    """One PNG per sun-table sample. Six panels, each carrying a measured thing.

    Two map axes, because one cannot carry both facts: the region is 10.24 km and
    this traverse is 560 m of it, so a single view either shows the shadows
    sweeping and loses the route to four pixels, or shows the route and loses the
    sweep. Then the executive panel, which is the reason R3 selected this
    instance at all; the world's own continuous shadow; all twelve epochs, so one
    traverse is not read as typical; and the checker's verdicts as text.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    fields = np.load(stack["path"], mmap_mode="r")
    litm = np.load(stack["lit_path"], mmap_mode="r")
    peak = stack["peak"]
    ext_km = (REGION_OX / 1000.0, (REGION_OX + REGION_N * PX) / 1000.0,
              (REGION_OY - REGION_N * PX) / 1000.0, REGION_OY / 1000.0)
    rep_td = replay_out(tag, "td")
    plan, _, dep = plan_at(rep_td, rows["td"])
    pt = planner_out(tag, "td")
    off = pt["offsets"]
    lo, hi = int(off[instance]), int(off[instance + 1])
    tsteps = pt["times"][lo:hi]
    routes, same = {}, []
    for key, kind in (("td", "td"), ("snap", "snap"), ("mm", MEANMAP_ARM)):
        if rows[key] is None:
            continue
        pl, _, _ = plan_at(replay_out(tag, kind), rows[key])
        if key != "td" and pl.size == plan.size and bool((pl == plan).all()):
            same.append(key)
        x, y = cells_to_map(pl)
        routes[key] = (x / 1000.0, y / 1000.0)
    rx, ry = cells_to_map(plan)

    # A square zoom window, fixed for the whole sequence, so it cannot be read as
    # a camera following anything interesting.
    allx = np.concatenate([routes[k][0] for k in routes])
    ally = np.concatenate([routes[k][1] for k in routes])
    mx = max(0.35, 0.25 * max(np.ptp(allx), np.ptp(ally)))
    cx, cy = (allx.min() + allx.max()) / 2, (ally.min() + ally.max()) / 2
    half = max(np.ptp(allx), np.ptp(ally)) / 2 + mx
    zx, zy = (cx - half, cx + half), (cy - half, cy + half)

    dem = stack["dem"]
    i0 = int(np.clip((REGION_OY / 1000.0 - zy[1]) * 1000.0 / PX, 0, REGION_N - 1))
    i1 = int(np.clip((REGION_OY / 1000.0 - zy[0]) * 1000.0 / PX, 0, REGION_N - 1))
    j0 = int(np.clip((zx[0] * 1000.0 - REGION_OX) / PX, 0, REGION_N - 1))
    j1 = int(np.clip((zx[1] * 1000.0 - REGION_OX) / PX, 0, REGION_N - 1))
    czz = np.asarray(dem, float)[i0:i1 + 1, j0:j1 + 1]
    cgx = cell_to_map(0, np.arange(j0, j1 + 1))[0] / 1000.0
    cgy = cell_to_map(np.arange(i0, i1 + 1), 0)[1] / 1000.0
    clev = np.arange(np.floor(czz.min() / 10.0) * 10.0, czz.max() + 10.0, 10.0)
    # Labelled only if dark in at least 5% of the frames. Below that a cell is
    # lit in almost every frame and renders lit, so a label would point at
    # nothing a viewer can see.
    cl_min = max(2, int(round(0.05 * litm.shape[0])))
    clusters = dark_clusters(litm, i0, i1, j0, j1, min_frames=cl_min)

    # The executive, which is why R3 chose this instance. Read from the committed
    # five-arm run rather than recomputed.
    ex = {a: exec_log(a, tag, rows["exec_plan_index"]) for a in ("none", "lit")}
    wtr = world_traj("none", tag, rows["exec_plan_index"])
    fire_t = fire_s = None
    if ex["none"] is not None and ex["none"]["fired"].any():
        k = int(np.flatnonzero(ex["none"]["fired"].astype(bool))[0])
        fire_t, fire_s = float(ex["none"]["t"][k]), float(ex["none"]["s_pred"][k])
    wsh = wtr["shadow"]
    wrun = np.maximum.accumulate(wsh)
    # The time axes span the traverse and not the frames being rendered, so a
    # short run for a look shows the same trace the full sequence does.
    t_end = max(float(wtr["t"][-1]),
                float(ex["none"]["t"][-1]) if ex["none"] is not None else 0.0,
                float(ex["lit"]["t"][-1]) if ex["lit"] is not None else 0.0)
    epochs_rows = select_r4()

    p3 = p3_record() if p3 is None else p3
    lines = panel_lines(tag, instance, rows, curves, same, pc, p3)
    clip_zero = lit_cells = 0
    for q in range(litm.shape[0]):
        m = np.asarray(litm[q], bool)
        lit_cells += int(m.sum())
        clip_zero += int((m & (np.asarray(fields[q]) == 0)).sum())
    caption = caption_lines(clip_zero, lit_cells, int(litm.shape[0]), p3,
                            float(rep_td["shadow"][rows["td"]]), "snap" in same)
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    written = []
    for k, t in enumerate(times):
        img = np.asarray(fields[k], float)
        mask = np.asarray(litm[k], bool)
        shown = np.where(mask, np.maximum(img / max(peak, 1e-12), LIT_FLOOR), 0.0)
        fig = plt.figure(figsize=(16, 9), dpi=dpi, facecolor="#07070a")
        step = step_at(tsteps, t)

        axes = []
        for q, (rect, lim, title) in enumerate((
                (REGION_RECT, None, "the planning region, 10.24 km"),
                (ZOOM_RECT, (zx, zy),
                 f"the traverse, {2000 * half:.0f} m across"))):
            ax = fig.add_axes(rect)
            ax.set_facecolor("#07070a")
            ax.imshow(shown, cmap="gray", vmin=0.0, vmax=1.0, extent=ext_km,
                      origin="upper", interpolation="nearest")
            for key, (style, col, lw, lab) in (
                    ("snap", ("-", C_SNAP, 5.0, "snapshot plan (same route)")),
                    ("mm", (":", C_MM, 1.8, "mean-map plan")),
                    ("td", ("-", C_TD, 1.8, "time-dependent, driven"))):
                if key in routes:
                    ax.plot(*routes[key], style, color=col, lw=lw,
                            solid_capstyle="round",
                            alpha=0.55 if key == "snap" else 1.0, label=lab)
            ax.plot([rx[step] / 1000.0], [ry[step] / 1000.0], "o", color=C_ROVER,
                    ms=10 if lim else 6, mec="#07070a", mew=1.2, label="rover")
            ax.plot([rx[0] / 1000.0], [ry[0] / 1000.0], "s", color="#ffffff", ms=4)
            ax.plot([rx[-1] / 1000.0], [ry[-1] / 1000.0], "*", color="#ffffff",
                    ms=10)
            if lim is None:
                ax.add_patch(Rectangle((zx[0], zy[0]), zx[1] - zx[0],
                                       zy[1] - zy[0], fill=False,
                                       edgecolor="#ffffff", lw=0.9, ls="--"))
            else:
                ax.set_xlim(*lim[0])
                ax.set_ylim(*lim[1])
                ax.contour(cgx, cgy, czz, levels=clev, colors="#4b5563",
                           linewidths=0.6, alpha=0.9)
                # Every dark cluster is named, not only the darkest, so the
                # grey levels in this panel have a stated meaning everywhere.
                for ci, cl in enumerate(clusters):
                    cells = f"{cl['cells']} cell" + ("s" if cl["cells"] > 1 else "")
                    txt = (f"{cells}, dark in "
                           + (f"all {cl['frames']}" if cl["lo"] == cl["frames"]
                              else f"{cl['lo']}-{cl['hi']} of {cl['frames']}")
                           + " frames")
                    dy = 0.17 if ci % 2 == 0 else -0.17
                    # Clamped inside the axes: an unclamped label on a cluster
                    # near the edge ran over the panel title.
                    tx = float(np.clip(cl["x"] - 0.40, zx[0] + 0.03,
                                       zx[1] - 0.46))
                    ty = float(np.clip(cl["y"] + dy, zy[0] + 0.05,
                                       zy[1] - 0.05))
                    ax.annotate(txt, xy=(cl["x"], cl["y"]), xytext=(tx, ty),
                                color="#e6e8ec", fontsize=6.2,
                                arrowprops=dict(arrowstyle="->", color="#e6e8ec",
                                                lw=0.7, alpha=0.85))
            ax.set_xlabel("map x (km)", color="#9aa0aa", fontsize=7)
            ax.set_ylabel("map y (km)", color="#9aa0aa", fontsize=7)
            ax.tick_params(colors="#9aa0aa", labelsize=6.5)
            for sp in ax.spines.values():
                sp.set_color("#30343c")
            ax.set_title(title, color="#e6e8ec", fontsize=9)
            axes.append(ax)
        leg = axes[1].legend(loc="lower left", fontsize=6.5, facecolor="#101218",
                             edgecolor="#30343c", labelcolor="#d6d9de")
        leg.get_frame().set_alpha(0.92)

        # --- the executive: the forecast against the budget, both arms ---
        axe = fig.add_axes(EXEC_RECT)
        axe.set_facecolor("#101218")
        for arm, col, lab in (("none", C_SNAP, "bounded rule, no correction"),
                              ("lit", C_LIT_ARM, "with the B3b-lit correction")):
            g = ex[arm]
            if g is None:
                continue
            axe.plot(g["t"], g["s_pred"], "-o", color=col, lw=1.6, ms=3, label=lab)
        axe.axhline(BUDGET_H, color=C_BUDGET, ls="--", lw=1.3,
                    label=f"budget {BUDGET_H:.1f} h")
        if fire_t is not None:
            axe.plot([fire_t], [fire_s], "X", color=C_FIRE, ms=12, mec="#07070a",
                     label=f"bounded rule FIRES, t={fire_t:.2f} h")
            axe.annotate(f"forecast {fire_s:.2f} h", xy=(fire_t, fire_s),
                         xytext=(fire_t + 2.0, fire_s + 0.12), color=C_FIRE,
                         fontsize=7.5)
        axe.axvline(t, color=C_ROVER, lw=1.2, alpha=0.8)
        axe.set_xlim(-0.6, t_end + 0.6)
        # Explicit, because the firing marker sits at the trace's maximum and an
        # auto-scaled axis clipped both it and its label off the top.
        top = max(BUDGET_H, *[float(g["s_pred"].max()) for g in ex.values()
                              if g is not None]) * 1.22
        axe.set_ylim(-0.06 * top, top)
        axe.set_xlabel("time since departure (h)", color="#9aa0aa", fontsize=7.5)
        axe.set_ylabel("executive's forecast maximum (h)", color="#9aa0aa",
                       fontsize=7.5)
        axe.set_title("why this instance: the correction removes a false alarm",
                      color="#e6e8ec", fontsize=9)
        axe.tick_params(colors="#9aa0aa", labelsize=7)
        for sp in axe.spines.values():
            sp.set_color("#30343c")
        le = axe.legend(loc="upper left", fontsize=6.8, facecolor="#101218",
                        edgecolor="#30343c", labelcolor="#d6d9de", ncol=2)
        le.get_frame().set_alpha(0.92)
        axe.text(0.985, 0.13, f"the world never exceeds {wrun.max():.2f} h, so "
                 f"the firing is a false alarm", color="#7e848e", fontsize=7.5,
                 ha="right", transform=axe.transAxes)

        # --- all twelve epochs, no selection ---
        axp = fig.add_axes(EPOCH_RECT)
        axp.set_facecolor("#101218")
        labs = [r["tag"][5:10].replace("-", "") + r["tag"][-3:]
                for r in epochs_rows]
        xs = np.arange(len(epochs_rows))
        lit0 = [100.0 * r.get("lit_t0", 0.0) for r in epochs_rows]
        cols = [C_ROVER if r["tag"] == tag else "#5a6472" for r in epochs_rows]
        axp.bar(xs, lit0, color=cols, width=0.62)
        axp.set_ylabel("region lit at departure (%)", color="#9aa0aa", fontsize=7)
        axp.set_xticks(xs)
        axp.set_xticklabels(labs, rotation=45, ha="right", fontsize=6.5)
        axp.tick_params(colors="#9aa0aa", labelsize=6.5)
        ax2 = axp.twinx()
        ax2.plot(xs, [r.get("td_plans", 0) for r in epochs_rows], "o",
                 color=C_TD, ms=4)
        ax2.set_ylabel("feasible plans of 120", color=C_TD, fontsize=7)
        ax2.tick_params(colors=C_TD, labelsize=6.5)
        for a in (axp, ax2):
            for sp in a.spines.values():
                sp.set_color("#30343c")
        axp.set_title("all twelve epochs, no selection; this one in yellow",
                      color="#e6e8ec", fontsize=9)

        # --- the world's own continuous shadow, with its resets ---
        axs = fig.add_axes(SHADOW_RECT)
        axs.set_facecolor("#101218")
        # Markers as well as the step, because the accumulator returns to zero
        # within a single sample here and the reset is one pixel wide without
        # them.
        axs.step(wtr["t"], wsh, where="post", color=C_TD, lw=1.6,
                 label="current continuous shadow (resets when lit)")
        axs.plot(wtr["t"], wsh, ".", color=C_TD, ms=4)
        axs.plot(wtr["t"], wrun, "--", color="#d6d9de", lw=1.2,
                 label="its running maximum")
        axs.axvline(t, color=C_ROVER, lw=1.2, alpha=0.8)
        axs.set_xlim(-0.6, t_end + 0.6)
        axs.set_ylim(0.0, float(wrun.max()) * 1.45 + 1e-6)
        axs.set_xlabel("time since departure (h)", color="#9aa0aa", fontsize=7.5)
        axs.set_ylabel("continuous shadow (h)", color="#9aa0aa", fontsize=7.5)
        axs.set_title("the world, as the checker integrated it",
                      color="#e6e8ec", fontsize=9)
        axs.tick_params(colors="#9aa0aa", labelsize=7)
        for sp in axs.spines.values():
            sp.set_color("#30343c")
        ls = axs.legend(loc="upper left", fontsize=6.8, facecolor="#101218",
                        edgecolor="#30343c", labelcolor="#d6d9de")
        ls.get_frame().set_alpha(0.92)
        axs.text(0.985, 0.04, f"the {BUDGET_H:.1f} h budget is "
                 f"{BUDGET_H / max(float(wrun.max()), 1e-9):.0f}x off the top "
                 f"of this axis", color="#7e848e", fontsize=7, ha="right",
                 transform=axs.transAxes)

        axt = fig.add_axes(TEXT_RECT)
        axt.axis("off")
        axt.text(0.0, 1.0, "\n".join(lines), color="#d6d9de", fontsize=7.6,
                 family="monospace", va="top", transform=axt.transAxes)

        fig.text(0.025, 0.955, f"{tag}   instance {instance}   t = {t:+.2f} h   "
                 f"{stack['lit_fraction'][k]:.2%} of the region lit   "
                 f"plan step {step + 1} of {plan.size}", color="#9aa0aa",
                 fontsize=10)
        for q, line in enumerate(caption):
            fig.text(0.025, 0.068 - q * 0.024, line, color="#7e848e", fontsize=7)
        out = outdir / f"frame{k:04d}.png"
        fig.savefig(out, facecolor=fig.get_facecolor())
        plt.close(fig)
        written.append(out)
    return written


# --------------------------------------------------------------------------
# the Gazebo animation, the second artifact
# --------------------------------------------------------------------------

GZ = "/usr/bin/gz"
GZ_ENV_KEY = "GZ_CONFIG_PATH"
GZ_CONFIG = "/usr/share/gz"
ROVER_LIFT_M = 0.5          # a display offset; the pose is commanded, see C5
# Camera placements. The wide view is steep enough that no sky enters the frame:
# at 4000 m over a 2828 m ground offset the line of sight is 54.7 deg down and
# the 47 deg vertical field spans 31 to 78 deg, all of it terrain. The first
# attempt put it at 1100 m over 1697 m, 32.9 deg down, and the top of every frame
# was the horizon at 6.6 km, which is outside the heightmap. The second was steep
# enough but only 2.1 km across, a window in which this epoch's shadows barely
# move; 6 km covers most of the 10.24 km region, which is where the sweep is.
WIDE_OFFSET = (-2000.0, -2000.0, 4000.0)
CHASE_OFFSET = (-18.0, -18.0, 9.0)
VIEW_W, VIEW_H = 800, 600
WIDE_FOV, CHASE_FOV = 1.10, 1.05
# One display curve for the whole sequence, never per frame. Our texture floors
# lit cells at 0.25 and this epoch's cos(incidence) peaks at 0.288, so the scene
# renders dim by construction: measured, lit terrain sits at exactly 34 of 255
# with p99 at 34 to 38 and a maximum of 111. A linear gain was tried first and
# rejected by measurement rather than by taste: x6 puts lit terrain at a good 204
# but saturates everything above 42, and the rover box renders at 73 even at a
# shade of 0.05, so lit and dark rovers both clipped to white and the signal the
# shade exists to carry was destroyed. A cube-root curve is monotone over the
# whole range and clips nothing: 0 stays 0, lit terrain 34 reads 130, the lit
# rover reads 255 and the dark rover 154.
DISPLAY_GAMMA = 1.0 / 3.0
# Brightness cannot carry the lit-dark signal and the measurement says why: the
# box cannot render darker than the terrain at any shade. At 0.01 and below it
# reads 34, exactly the terrain, and the darkest usable shade of 0.03 reads 56
# raw, which is 154 after the display curve against lit terrain's 130. So a
# "dark" rover came out brighter than lit ground and read as lit. Hue is free of
# that constraint, and the terrain is neutral grey at zero saturation, so any
# saturated colour separates from it whatever the brightness.
#
# Measured on the rendered image at the display curve, 2026-10-06:
#   amber box   display RGB (252.5, 250.5, 171.8)  saturation 169.5
#   blue  box   display RGB (186.1, 220.1, 252.5)  saturation 148.3
#   terrain     display RGB (130.3, 130.3, 130.3)  saturation   0.0
#   distance from terrain  amber 176.4, blue 161.6; between the two 108.8
ROVER_RGB_LIT = (1.00, 0.72, 0.06)
ROVER_RGB_DARK = (0.10, 0.30, 1.00)
# The swatches on the figure are the colours as they appear after the display
# curve, so the legend matches the pixels rather than the material.
SWATCH_LIT, SWATCH_DARK = "#fcfaac", "#badcfc"

# Two caption lines that carry measurements made once and recorded in PLAN.md
# rather than in a results file: the chassis box and the display curve's levels.
# They are sun-independent or set by the 0.25 floor, and are not on the step 14
# list of figures the corrections made stale; the rest are computed.
GZ_ROVER_LINE = (
    "The rover's pose is commanded, not driven. Nothing here is a claim about "
    "mobility, slip or traction. Its only visual geometry is the 1.53 x 1.53 x "
    "0.60 m chassis box; the mast and wheels carry sensor frames and no visual.")
GZ_DISPLAY_LINE = (
    "Brightness is a single fixed cube-root display curve for the whole "
    "sequence, never per frame, so frames stay comparable: lit terrain renders "
    "at 34 of 255 before it and 130 after.")


def gz_caption_lines(c3):
    """The Gazebo frames' caption: C3 from its checkpoint, the surface term from
    verify_gazebo.SLOPE_BANDS, the placement offset from
    verify_gazebo.TEXTURE_OFFSET_NADIR_M, the margin from the grid constants."""
    from examples.verify_gazebo import SLOPE_BANDS, TEXTURE_OFFSET_NADIR_M
    lo, hi = SLOPE_BANDS[0], SLOPE_BANDS[-1]
    off = list(TEXTURE_OFFSET_NADIR_M.values())
    margin = HEIGHTMAP_N - REGION_N
    half = "under half" if max(off) < PX / 2 else "more than half"
    return (
        "Rendered by Gazebo; the illumination is OUR model written as an unlit "
        "texture, and the scene's light casts no shadow. C3 measured "
        f"{c3['positive_disagreement']:.4%} of texture pixels disagreeing with our "
        "own lit mask.",
        "The rendered surface is not the DEM: it differs by a slope-dependent "
        f"amount, median {lo[2]:+.4f} m below {lo[1]:g} deg rising to "
        f"{hi[2]:+.4f} m above {hi[0]:g} deg. The mechanism is open and unexplained.",
        GZ_ROVER_LINE,
        GZ_DISPLAY_LINE,
        "The thin dark band along the far edge of the wide view is the texture's "
        f"own margin, not terrain: the heightmap is {HEIGHTMAP_N} samples and the "
        f"planning region is {REGION_N}, so {margin} rows and columns, "
        f"{margin * PX:.0f} m, lie outside it and are written dark rather than "
        "extrapolated.",
        f"A rendered cell can sit {min(off):.1f} to {max(off):.1f} m from where the "
        f"model puts it, {half} a {PX:.0f} m cell. Measured at nadir and obliquely "
        "by three instruments; it is not the slope term and the mechanism is open. "
        "So the rover may appear at the edge of its own dark cell. The 2D map view "
        "is unaffected.",
    )

# Sits beside the chase view rather than in the caption block, because it is
# about that panel alone.
CHASE_NOTE = ("about 11 texels wide. sdformat exposes no filtering option, so\n"
              "each 20 m cell is written as an 8 x 8 block of identical texels\n"
              "and the filter can only blend at block edges. Measured on this\n"
              "view, that narrows the lit-to-dark transition from 4.82 m to\n"
              "1.98 m and cuts part-lit pixels from 55.4% to 8.8%. Nothing is\n"
              "interpolated: every texel in a block is that cell's own value.")


def look_at(cam, tgt):
    """Roll, pitch and yaw that point a Gazebo camera from cam at tgt."""
    dx, dy, dz = (tgt[0] - cam[0], tgt[1] - cam[1], tgt[2] - cam[2])
    yaw = float(np.arctan2(dy, dx))
    pitch = float(-np.arctan2(dz, float(np.hypot(dx, dy))))
    return 0.0, pitch, yaw


def rover_world_pose(cells, step, dem):
    """World x, y, z and yaw for the rover at one plan step.

    z comes from the DEM, not from Gazebo's rendered surface, and the two differ
    by a slope-dependent amount this record carries as open. The pose is
    commanded, so this is a placement and not a contact solution.
    """
    i, j = int(cells[step]) // REGION_N, int(cells[step]) % REGION_N
    mx, my = cell_to_map(i, j)
    wx, wy = map_to_world(mx, my)
    z = float(bilinear(dem, i, j)) + ROVER_LIFT_M
    nxt = min(step + 1, len(cells) - 1)
    i2, j2 = int(cells[nxt]) // REGION_N, int(cells[nxt]) % REGION_N
    mx2, my2 = cell_to_map(i2, j2)
    yaw = (float(np.arctan2(my2 - my, mx2 - mx))
           if (i2, j2) != (i, j) else 0.0)
    return float(wx), float(wy), z, yaw


def gz_grab(sdf, topics, poll_s=0.5, cap_s=40.0):
    """Launch, poll until every topic has a non-blank image, tear down.

    Polled rather than slept. Measured on 2026-10-06: the first image arrives in
    2.6 to 3.0 s and is bit-identical to the image at 30 s, because the scene is
    static. The committed camera path's 30 s settle came from the sphere checks,
    where something had to stop moving, and it is 10 times too long here. A
    non-blank check is kept because an arriving message is not a finished render,
    which is this record's standing lesson about trusting a return value.
    """
    import base64
    import os
    import time

    env = dict(os.environ, **{GZ_ENV_KEY: GZ_CONFIG})
    proc = subprocess.Popen([GZ, "sim", "-s", "-r", "-v", "1", str(sdf)],
                            env=env, stdout=subprocess.DEVNULL,
                            stderr=subprocess.DEVNULL, start_new_session=True)
    got, t0 = {}, time.time()
    try:
        while len(got) < len(topics) and time.time() - t0 < cap_s:
            time.sleep(poll_s)
            for t in topics:
                if t in got:
                    continue
                r = subprocess.run([GZ, "topic", "-e", "-t", t, "-n", "1",
                                    "--json-output"], env=env,
                                   capture_output=True, text=True, timeout=20)
                if not r.stdout.strip():
                    continue
                m = json.loads(r.stdout)
                if not m.get("data"):
                    continue
                w, h = int(m["width"]), int(m["height"])
                a = np.frombuffer(base64.b64decode(m["data"]), np.uint8)
                if a.size < w * h * 3 or not a.any():
                    continue
                got[t] = a[:w * h * 3].reshape(h, w, 3)
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
    missing = [t for t in topics if t not in got]
    if missing:
        raise SystemExit(f"no image on {', '.join(missing)} within {cap_s:.0f} s")
    return got


def render_gazebo(tag, instance, times, stack, rows, texdir, outdir, rover_sdf,
                  dpi=100):
    """One Gazebo frame per sun-table sample, two views, same cadence as the map."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from examples.make_world import build, flat_normal_png

    side = json.loads((ROOT / "data" / "work" / "gazebo"
                       / "region_heightmap.json").read_text())
    normal = flat_normal_png(ROOT / "data" / "work" / "gazebo" / "flat_normal.png")
    gz_caption = gz_caption_lines(c3_record())
    rep_td = replay_out(tag, "td")
    plan, _, _ = plan_at(rep_td, rows["td"])
    pt = planner_out(tag, "td")
    lo, hi = int(pt["offsets"][instance]), int(pt["offsets"][instance + 1])
    tsteps = pt["times"][lo:hi]
    litm = np.load(stack["lit_path"], mmap_mode="r")
    dem, outdir, texdir = stack["dem"], Path(outdir), Path(texdir)
    outdir.mkdir(parents=True, exist_ok=True)
    work = outdir / "worlds"
    work.mkdir(parents=True, exist_ok=True)
    written = []
    for k, t in enumerate(times):
        step = step_at(tsteps, t)
        wx, wy, wz, yaw = rover_world_pose(plan, step, dem)
        tgt = (wx, wy, wz + 1.2)
        views = []
        for name, off, fov in (("wide", WIDE_OFFSET, WIDE_FOV),
                               ("chase", CHASE_OFFSET, CHASE_FOV)):
            cam = (wx + off[0], wy + off[1], wz + off[2])
            r, p_, y_ = look_at(cam, tgt)
            views.append((name, cam[0], cam[1], cam[2], r, p_, y_, fov,
                          VIEW_W, VIEW_H))
        cell_lit = bool(litm[k].reshape(-1)[int(plan[step])])
        shade = ROVER_RGB_LIT if cell_lit else ROVER_RGB_DARK
        sdf = build(side["path"], "ours", texdir / f"tex{k:04d}.png", normal,
                    side["pos_z"], [0.0, 0.0, -1.0], [], None, "vertices",
                    side["h_range"], rover_sdf, (), (), (wx, wy, wz, yaw),
                    False, views, shade)
        wf = work / f"frame{k:04d}.sdf"
        wf.write_text(sdf)
        img = gz_grab(wf, ["view/wide/image", "view/chase/image"])
        wide_m = 2.0 * float(np.linalg.norm(
            [WIDE_OFFSET[0], WIDE_OFFSET[1], WIDE_OFFSET[2]])) * np.tan(WIDE_FOV / 2)
        chase_m = float(np.linalg.norm(
            [CHASE_OFFSET[0], CHASE_OFFSET[1], CHASE_OFFSET[2]]))
        fig = plt.figure(figsize=(16, 8), dpi=dpi, facecolor="#07070a")
        for q, (topic, label) in enumerate((
                ("view/wide/image",
                 f"wide view, about {wide_m:.0f} m across, camera "
                 f"{WIDE_OFFSET[2]:.0f} m up"),
                ("view/chase/image",
                 f"chase view, {chase_m:.0f} m range: the chassis box at "
                 f"true scale"))):
            ax = fig.add_axes([0.012 + q * 0.494, 0.258, 0.482, 0.637])
            # One fixed curve, applied identically to every frame of both views.
            ax.imshow((255.0 * (img[topic].astype(float) / 255.0)
                       ** DISPLAY_GAMMA).astype(np.uint8))
            ax.set_xticks([])
            ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_color("#30343c")
            ax.set_title(label, color="#e6e8ec", fontsize=10)
            if topic.endswith("chase/image"):
                # Inside the panel it is about, rather than in the caption block
                # at the foot of the figure, where it overlapped the full-width
                # caption lines.
                ax.text(0.015, 0.025, CHASE_NOTE, transform=ax.transAxes,
                        color="#d6d9de", fontsize=7.0, va="bottom",
                        bbox=dict(facecolor="#07070a", edgecolor="#30343c",
                                  alpha=0.78, pad=4))
        fig.text(0.012, 0.955, f"{tag}   instance {instance}   t = {t:+.2f} h   "
                 f"plan step {step + 1} of {plan.size}   "
                 f"{stack['lit_fraction'][k]:.2%} of the region lit   "
                 f"this cell: {'LIT' if cell_lit else 'DARK'}",
                 color="#9aa0aa", fontsize=10)
        fig.text(0.012, 0.925, f"display curve: brightness to the power "
                 f"{DISPLAY_GAMMA:.3f}, the same for every frame",
                 color="#7e848e", fontsize=8)
        for q, line in enumerate(gz_caption):
            fig.text(0.012, 0.204 - q * 0.0285, line, color="#7e848e",
                     fontsize=7.0)
        # The legend is the point of the colours, so it states them rather than
        # leaving a viewer to infer a code from two frames.
        for q, (sw, lab) in enumerate((
                (SWATCH_LIT, "rover on a cell our model says is LIT"),
                (SWATCH_DARK, "rover on a cell our model says is DARK"))):
            fig.text(0.512 + q * 0.235, 0.925, "    ", fontsize=8,
                     bbox=dict(facecolor=sw, edgecolor="#30343c", pad=2.5))
            fig.text(0.532 + q * 0.235, 0.925, lab, color="#9aa0aa", fontsize=7.5)
        out = outdir / f"gz{k:04d}.png"
        fig.savefig(out, facecolor=fig.get_facecolor())
        plt.close(fig)
        written.append(out)
    return written


# --------------------------------------------------------------------------
# encoding and the sidecar
# --------------------------------------------------------------------------

def encode(pattern, out, fps):
    """PNG sequence to MP4 with the committed ffmpeg, at the stated rate."""
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(fps),
                    "-i", str(pattern), "-c:v", "libx264", "-pix_fmt", "yuv420p",
                    "-crf", "20", str(out)], check=True)
    return out


def head_commit():
    r = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"],
                       capture_output=True, text=True)
    return r.stdout.strip() or "unknown"


def file_record(paths):
    import hashlib
    out = {}
    for p in paths:
        p = Path(p)
        if not p.exists():
            out[str(p)] = None
            continue
        h = hashlib.sha256(p.read_bytes()).hexdigest()[:16] if p.stat().st_size < 2 << 28 \
            else "too large to hash"
        out[str(p)] = dict(bytes=p.stat().st_size, sha256_16=h)
    return out


# --------------------------------------------------------------------------
# the static panels: the executive on two instances, and all twelve epochs
# --------------------------------------------------------------------------

LOG_DT = dict(tick=np.int64, t=np.float64, s_exec=np.float64,
              s_world=np.float64, s_pred=np.float64, index=np.int32,
              kind=np.uint8, lit=np.uint8, fired=np.uint8)


def exec_log(arm, tag, plan_index, leg=0):
    """One leg's executive log, as the committed run left it on disk."""
    d = CORR / arm / tag / f"plan{plan_index:03d}" / f"leg{leg}_exec"
    if not d.exists():
        return None
    return {k: np.fromfile(d / f"out_log_{k}.bin", dt) for k, dt in LOG_DT.items()}


def world_traj(arm, tag, plan_index, leg=0):
    """The world's own accumulated-shadow trajectory for the same leg."""
    d = CORR / arm / tag / f"plan{plan_index:03d}" / f"leg{leg}_world"
    return dict(t=np.fromfile(d / "out_traj_t.bin", np.float64),
                shadow=np.fromfile(d / "out_traj_shadow.bin", np.float64))


def render_exec_panel(tag, plan_index, arms, title, out, dpi=120):
    """Forecast against the world, per arm, with the budget and the firing."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(12, 4.6 * len(arms)), dpi=dpi, facecolor="#07070a")
    for q, (arm, label) in enumerate(arms):
        log, w = exec_log(arm, tag, plan_index), world_traj(arm, tag, plan_index)
        ax = fig.add_subplot(len(arms), 1, q + 1)
        ax.set_facecolor("#101218")
        ax.plot(w["t"], w["shadow"], "-", color="#6f7681", lw=1.2,
                label="the world's accumulated shadow")
        if log is not None:
            ax.plot(log["t"], log["s_pred"], "-", color=C_TD, lw=1.6,
                    label="the executive's forecast maximum")
            fired = log["fired"].astype(bool)
            if fired.any():
                k = int(np.flatnonzero(fired)[0])
                ax.axvline(log["t"][k], color="#ff5c5c", lw=1.4)
                ax.plot([log["t"][k]], [log["s_pred"][k]], "o", color="#ff5c5c",
                        ms=8, label=f"control fires at t = {log['t'][k]:.2f} h")
            else:
                ax.plot([], [], " ", label="no control fires")
        ax.axhline(BUDGET_H, color="#ffa94d", ls="--", lw=1.2,
                   label=f"budget {BUDGET_H:.1f} h")
        ax.set_title(f"{label}", color="#e6e8ec", fontsize=10, loc="left")
        ax.set_xlabel("time since departure (h)", color="#9aa0aa")
        ax.set_ylabel("continuous shadow (h)", color="#9aa0aa")
        ax.tick_params(colors="#9aa0aa", labelsize=8)
        for sp in ax.spines.values():
            sp.set_color("#30343c")
        leg = ax.legend(loc="upper left", fontsize=8, facecolor="#101218",
                        edgecolor="#30343c", labelcolor="#d6d9de")
        leg.get_frame().set_alpha(0.92)
    fig.suptitle(title, color="#e6e8ec", fontsize=12)
    fig.tight_layout(rect=(0, 0.0, 1, 0.97))
    fig.savefig(out, facecolor=fig.get_facecolor())
    plt.close(fig)
    return out


def epoch_note(rows):
    """R4's footnote, counted from the rows it draws."""
    per = sorted((r.get("td_plans", 0) for r in rows), reverse=True)
    return (f"Every epoch is drawn: R4 applies no selection. {sum(per[:3])} of the "
            f"{sum(per)} feasible time-dependent instances come from three of these "
            f"{len(rows)} epochs, which is why one traverse is not typical.")


def render_epoch_panel(rows, out, dpi=120):
    """R4: all twelve epochs, no selection, from the stored sweep outputs."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    tags = [r["tag"] for r in rows]
    lit = [100.0 * r.get("lit_t0", 0.0) for r in rows]
    tdp = [r.get("td_plans", 0) for r in rows]
    tdf = [r.get("td_pass", 0) for r in rows]
    snp = [r.get("snap_plans", 0) for r in rows]
    x = np.arange(len(rows))
    fig = plt.figure(figsize=(13, 7), dpi=dpi, facecolor="#07070a")
    ax = fig.add_subplot(2, 1, 1)
    ax.set_facecolor("#101218")
    ax.bar(x, lit, color="#d6d9de", width=0.6)
    ax.set_ylabel("region lit at departure (%)", color="#9aa0aa")
    ax.set_title("solvability is set by the lunation phase, not by the date",
                 color="#e6e8ec", fontsize=11, loc="left")
    ax2 = fig.add_subplot(2, 1, 2)
    ax2.set_facecolor("#101218")
    ax2.bar(x - 0.2, snp, width=0.36, color=C_SNAP, label="snapshot plans produced")
    ax2.bar(x + 0.2, tdp, width=0.36, color=C_TD, label="time-dependent produced")
    ax2.plot(x + 0.2, tdf, "o", color="#ffffff", ms=5,
             label="time-dependent, checker accepted")
    ax2.set_ylabel("plans of 120 instances", color="#9aa0aa")
    ax2.legend(fontsize=8, facecolor="#101218", edgecolor="#30343c",
               labelcolor="#d6d9de")
    for a in (ax, ax2):
        a.set_xticks(x)
        a.set_xticklabels(tags, rotation=35, ha="right", fontsize=8)
        a.tick_params(colors="#9aa0aa", labelsize=8)
        for sp in a.spines.values():
            sp.set_color("#30343c")
    fig.text(0.01, 0.015, epoch_note(rows), color="#7e848e", fontsize=8)
    fig.tight_layout(rect=(0, 0.035, 1, 1))
    fig.savefig(out, facecolor=fig.get_facecolor())
    plt.close(fig)
    return out


# --------------------------------------------------------------------------
# a rule that selects nothing: reported as an empty set, never substituted
# --------------------------------------------------------------------------

EMPTY_RULE = ("Nothing is substituted: the registration says an empty set is "
              "reported as an empty set.")


def empty_rule_lines(rule, counts):
    """What an empty R2 or R3 says, from its own counts.

    R1 has no panel and is reported as empty in the selection text; R4 applies
    no selection and cannot be empty. R2 and R3 each own a panel, and each can
    come up empty on data the registration did not foresee.
    """
    if rule == "R2":
        return [f"R2 is EMPTY: {counts['busts']} busts among the "
                f"{counts['monitored']} monitored core plans.",
                "R2 selects the busts among the monitored core plans, first in "
                "enumeration order (registered 2026-10-05).",
                f"No core plan busts its {BUDGET_H:.1f} h budget after the "
                f"corrections, so there is no bust for the control to catch.",
                EMPTY_RULE]
    if rule == "R3":
        why = ("the bounded rule raised no false alarm" if counts["false_alarms"] == 0
               else "the B3b-lit correction clears none of the bounded rule's "
                    "false alarms")
        return [f"R3 is EMPTY: {counts['removed_by_lit']} of "
                f"{counts['false_alarms']} false alarms cleared, among "
                f"{counts['monitored_nonbust']} monitored core plans that do not bust.",
                "R3 selects a monitored non-bust the bounded rule fires on and the "
                "B3b-lit correction clears (registered 2026-10-05).",
                f"It is empty because {why}. With no R3 there is no traverse subject.",
                EMPTY_RULE]
    raise ValueError(f"{rule} has no empty-set panel")


def render_empty_panel(title, lines, out, dpi=120):
    """A panel that states a rule is empty, in the panels' own style."""
    import textwrap
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(12, 4.6), dpi=dpi, facecolor="#07070a")
    body = "\n\n".join(textwrap.fill(x, 100) for x in lines)
    fig.text(0.04, 0.86, title, color="#e6e8ec", fontsize=13, va="top")
    fig.text(0.04, 0.72, body, color="#d6d9de", fontsize=10, va="top",
             family="monospace")
    fig.savefig(out, facecolor=fig.get_facecolor())
    plt.close(fig)
    return out


def render_panels(sel, subject, out=None):
    """The three static panels. An empty R2 or R3 gets a panel saying so.

    The file names are kept when a rule is empty, so a panel left from an
    earlier run, when the rule did select something, is overwritten rather than
    shipped beside the statement that the set is now empty.
    """
    out = OUT if out is None else Path(out)
    b, c2 = sel["r2"]["choice"], sel["r2"]["counts"]
    if b is None:
        render_empty_panel("R2, a bust: EMPTY", empty_rule_lines("R2", c2),
                           out / "panel_r2_bust.png")
    else:
        render_exec_panel(b["tag"], b["plan_index"],
                          [("none", "bounded rule, no correction stream")],
                          f"R2, a bust: {b['tag']} plan{b['plan_index']:03d}. "
                          f"The control fires and the correction must not stop it.",
                          out / "panel_r2_bust.png")
    c3 = sel["r3"]["counts"]
    if subject is None:
        render_empty_panel("R3, a false alarm removed: EMPTY",
                           empty_rule_lines("R3", c3),
                           out / "panel_r3_false_alarm.png")
    else:
        tag, plan_index = subject["tag"], subject["plan_index"]
        render_exec_panel(tag, plan_index,
                          [("none", "bounded rule, no correction stream: fires"),
                           ("lit", "with the B3b-lit correction: does not fire")],
                          f"R3, a false alarm removed: {tag} plan{plan_index:03d}, "
                          f"1 of the {c3['removed_by_lit']} of "
                          f"{c3['false_alarms']} false alarms the lit direction "
                          f"removes on the core",
                          out / "panel_r3_false_alarm.png")
    render_epoch_panel(select_r4(), out / "panel_r4_epochs.png")
    names = ("panel_r2_bust.png", "panel_r3_false_alarm.png", "panel_r4_epochs.png")
    print(f"  wrote {', '.join(names)}"
          + ("; R2 is EMPTY" if b is None else "")
          + ("; R3 is EMPTY" if subject is None else ""))
    return [out / n for n in names]


# --------------------------------------------------------------------------
# the run
# --------------------------------------------------------------------------

def report_selection():
    """Every filter's counts and choice, printed and returned for the sidecar."""
    r1, c1, a1 = select_r1()
    r2, c2, a2 = select_r2()
    r3, c3, a3 = select_r3()
    print("the registered rules, applied (PLAN.md 2026-10-05, before looking):")
    print(f"  R1 counts {c1}")
    print(f"  R1 candidates {len(a1)} -> {r1}")
    if r1 is None:
        print("     R1 is EMPTY. Nothing is relaxed: the registration says an empty")
        print("     set is reported as an empty set. The traverse subject falls to")
        print("     R3, itself registered on the same date before any looking.")
    print(f"  R2 counts {c2}")
    print(f"  R2 candidates {len(a2)} -> {r2}")
    if r2 is None:
        for line in empty_rule_lines("R2", c2):
            print(f"     {line}")
    print(f"  R3 counts {c3}")
    print(f"  R3 candidates {len(a3)} -> {r3}")
    if r3 is None:
        for line in empty_rule_lines("R3", c3):
            print(f"     {line}")
    pc = pair_contingency()
    return dict(pair_contingency=pc,r1=dict(counts=c1, n=len(a1), choice=r1),
                r2=dict(counts=c2, n=len(a2), choice=r2),
                r3=dict(counts=c3, n=len(a3), choice=r3))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="select",
                    help="select,fields,frames,panels,video,gazebo or all")
    ap.add_argument("--fps", type=int, default=12)
    ap.add_argument("--limit", type=int, default=0,
                    help="render only the first N frames, for a look")
    ap.add_argument("--rover", default=str(ROOT / "data" / "work" / "gazebo"
                                          / "rover.sdf"))
    args = ap.parse_args()
    want = ["select", "fields", "frames", "panels", "video", "gazebo"] \
        if args.stage == "all" else [t.strip() for t in args.stage.split(",")]
    unknown = [t for t in want if t not in
               ("select", "fields", "frames", "panels", "video", "gazebo")]
    if unknown:
        raise SystemExit(f"unknown stages: {', '.join(unknown)}")
    OUT.mkdir(parents=True, exist_ok=True)

    sel = report_selection()
    subject = sel["r3"]["choice"]
    if subject is None:
        need = [t for t in ("fields", "frames", "video", "gazebo") if t in want]
        if need:
            raise SystemExit(f"R3 is EMPTY, so there is no traverse subject and the "
                             f"stages {', '.join(need)} have nothing to render. "
                             f"--stage panels renders the panels, R3's stating that "
                             f"it is empty.")
        if "panels" in want:
            print("\npanels: the executive on two instances, and all twelve epochs")
            render_panels(sel, None)
        import datetime as _dt
        (OUT / "sidecar.json").write_text(json.dumps(dict(
            head_commit=head_commit(),
            generated_utc=_dt.datetime.now(_dt.timezone.utc).isoformat(),
            subject=None, selection=sel), indent=1))
        print(f"\nsidecar {OUT / 'sidecar.json'}")
        return 0
    tag, plan_index = subject["tag"], subject["plan_index"]
    if "frames" in want:
        p3_record()
    if "gazebo" in want:
        c3_record()
    rep = replay_out(tag, "td")
    plan, _, dep = plan_at(rep, plan_index)
    instance = planner_out(tag, "td")["index"][plan_index]
    arr = float(rep["arrival"][plan_index])
    times = frame_times(dep, arr)
    if args.limit:
        times = times[:args.limit]
    print(f"\nsubject: {tag} plan{plan_index:03d}, instance {instance}")
    print(f"  departs {dep:.4f} h, checker arrival {arr:.4f} h, "
          f"{plan.size} cells, {times.size} frames at {CADENCE_H} h")

    rows = route_rows(tag, instance)
    curves = {}
    if {"fields", "frames", "video", "gazebo"} & set(want):
        for key, kind in (("td", "td"), ("snap", "snap")):
            if rows[key] is None:
                continue
            curves[key] = shadow_curve(replay_out(tag, kind), rows[key],
                                       OUT / "prefix" / kind)
        c = curves["td"]
        print(f"  checker prefix curve: max {c['shadow'].max():.4f} h, stored "
              f"{c['stored_shadow']:.4f} h, agree")

    assert_drivable(tag, plan_index)
    print("  C5: the checker's own verdict on this plan is ACCEPTED")

    stack_path = OUT / f"fields_{tag}_{plan_index:03d}.npy"
    stale = (stack_path.exists()
             and np.load(stack_path, mmap_mode="r").shape[0] != times.size)
    if stale:
        print(f"  the stored field stack has "
              f"{np.load(stack_path, mmap_mode='r').shape[0]} frames and this "
              f"run wants {times.size}; rebuilding it")
    stack = None
    if "fields" in want or stale or not stack_path.exists():
        print("\nfields: our illumination at every frame time, one normalisation")
        stack = field_stack(tag, times, stack_path)
        (OUT / "fields.json").write_text(json.dumps(
            dict(tag=tag, plan_index=plan_index, frames=int(times.size),
                 cadence_h=CADENCE_H, peak_cos_incidence=stack["peak"],
                 lit_fraction=stack["lit_fraction"].tolist()), indent=1))
        print(f"  peak cos(incidence) over the sequence {stack['peak']:.6f}")
        print(f"  lit fraction {stack['lit_fraction'].min():.4%} to "
              f"{stack['lit_fraction'].max():.4%}")
    else:
        meta = json.loads((OUT / "fields.json").read_text())
        p_, n_, _px, dem_, *_ = P.load(Rover(v_flat=V_FLAT))
        stack = dict(path=str(stack_path),
                     lit_path=str(stack_path.with_name(stack_path.stem + "_lit.npy")),
                     peak=meta["peak_cos_incidence"],
                     lit_fraction=np.asarray(meta["lit_fraction"]),
                     dem=dem_, n=n_, radius_m=p_["radius_m"])

    if "frames" in want:
        print("\nframes: the 2D map view")
        texdir = OUT / "textures"
        texdir.mkdir(parents=True, exist_ok=True)
        fields = np.load(stack["path"], mmap_mode="r")
        litm = np.load(stack["lit_path"], mmap_mode="r")
        for k in range(times.size):
            texture_png(np.asarray(fields[k], float),
                        np.asarray(litm[k], bool), stack["peak"],
                        texdir / f"tex{k:04d}.png")
        got = render_frames(tag, instance, times, stack, rows, curves,
                            OUT / "map", pc=sel["pair_contingency"])
        print(f"  {len(got)} map frames and {times.size} textures written")

    if "panels" in want:
        print("\npanels: the executive on two instances, and all twelve epochs")
        render_panels(sel, subject)

    if "video" in want:
        print("\nvideo")
        encode(OUT / "map" / "frame%04d.png", OUT / "demo_map.mp4", args.fps)
        print(f"  demo_map.mp4 at {args.fps} fps = "
              f"{CADENCE_H * args.fps:.2f} h of simulated time per second")

    if "gazebo" in want:
        print("\ngazebo: the second artifact, same subject and same cadence")
        got = render_gazebo(tag, instance, times, stack, rows, OUT / "textures",
                            OUT / "gz", args.rover)
        encode(OUT / "gz" / "gz%04d.png", OUT / "demo_gazebo.mp4", args.fps)
        print(f"  {len(got)} Gazebo frames, demo_gazebo.mp4")

    side = dict(
        head_commit=head_commit(), generated_utc=None,
        subject=dict(tag=tag, plan_index=plan_index, instance=instance,
                     departure_h=dep, checker_arrival_h=arr,
                     checker_shadow_h=float(rep["shadow"][plan_index]),
                     checker_verdict=bool(rep["verdict"][plan_index]),
                     plan_cells=int(plan.size)),
        selection=sel, rows=rows,
        configuration=dict(v_flat=V_FLAT, budget_h=BUDGET_H,
                           cadence_h=CADENCE_H, frames=int(times.size),
                           fps=args.fps,
                           hours_per_second=CADENCE_H * args.fps,
                           normalisation_peak_cos_incidence=stack["peak"],
                           lit_floor=LIT_FLOOR, rover_lift_m=ROVER_LIFT_M,
                           display_gamma=DISPLAY_GAMMA, upsample=UPSAMPLE,
                           rover_rgb_lit=list(ROVER_RGB_LIT),
                           rover_rgb_dark=list(ROVER_RGB_DARK),
                           wide_offset_m=list(WIDE_OFFSET),
                           chase_offset_m=list(CHASE_OFFSET)),
        inputs=file_record([
            SWEEP / f"{tag}_td" / "planner_params.json",
            SWEEP / f"{tag}_replay_td_tol" / "out_verdict.bin",
            SWEEP / f"{tag}_replay_td_tol" / "out_shadow.bin",
            CORR / "results.json",
            cache_path(tag)]),
        versions=dict(numpy=np.__version__))
    import datetime as _dt
    import matplotlib
    side["generated_utc"] = _dt.datetime.now(_dt.timezone.utc).isoformat()
    side["versions"]["matplotlib"] = matplotlib.__version__
    r = subprocess.run([GZ, "sim", "--versions"],
                       env={**__import__("os").environ, GZ_ENV_KEY: GZ_CONFIG},
                       capture_output=True, text=True)
    side["versions"]["gazebo"] = r.stdout.strip().splitlines()[0] if r.stdout else "?"
    (OUT / "sidecar.json").write_text(json.dumps(side, indent=1))
    print(f"\nsidecar {OUT / 'sidecar.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
