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


def meanmap_arm(p3):
    """The mean-map replay the demo draws: the exponent P3's registered
    criterion selected, read from its record. The demo carried "mm2.0" as a
    constant, the pre-fix selection, after the corrected scorer selected 1.0;
    there is deliberately no default."""
    return f"mm{float(p3['exponent'])}"


def meanmap_label(p3):
    return f"mean-map e={float(p3['exponent']):g}"


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


def route_rows(tag, instance, mm_kind):
    """The three planners' rows for one instance, where each produced a plan.
    mm_kind is meanmap_arm(p3_record()), carried in the rows as "mm_kind"."""
    out = {}
    td = planner_out(tag, "td")
    out["td"] = td["index"].index(instance) if instance in td["index"] else None
    sn = planner_out(tag, "snap")
    out["snap"] = sn["index"].index(instance) if instance in sn["index"] else None
    mm = replay_out(tag, mm_kind)
    out["mm_kind"] = mm_kind
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
    if p3 is None:
        raise ValueError("panel_lines needs the P3 record: the mean-map row is "
                         "the exponent it selected")
    mm_kind = meanmap_arm(p3)
    if rows.get("mm_kind", mm_kind) != mm_kind:
        raise ValueError(f"rows hold {rows['mm_kind']} but P3 selected {mm_kind}")
    for key, label, kind in (("td", "time-dependent", "td"),
                             ("snap", "snapshot", "snap"),
                             ("mm", meanmap_label(p3), mm_kind)):
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
        # "0 one only" read as "no instance has one planner's plan only", which
        # is false: 5 have only a time-dependent plan and 122 only a snapshot
        # plan. The disagreement count is over shared pairs, so it says so, and
        # the unpaired instances are stated beside it.
        b = pc["both"]
        n, split = sum(b.values()), b["ar"] + b["ra"]
        lines += ["", f"on the {n} instances both planned: {b['aa']} both accepted,",
                  f"{b['rr']} both rejected; "
                  + ("no pair accepted by one planner" if split == 0
                     else f"{split} accepted by one planner"),
                  "and rejected by the other.",
                  f"not paired: {pc['only_td']} with only a time-dependent plan,",
                  f"{pc['only_snap']} with only a snapshot plan."]
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


LABEL_FONT = 6.2


def cluster_text(cl):
    """'76 cells: 8-50': the cluster's size and how many frames its cells are
    dark in, lowest to highest. The format is stated once, in the legend
    (CLUSTER_KEY), not in every label: written out in full, 26 labels on R3's
    subject needed 37% of the panel at 6.2 pt and fitted only at 4.7 pt with
    leaders across the route."""
    cells = f"{cl['cells']} cell" + ("s" if cl["cells"] > 1 else "")
    if cl["lo"] == cl["frames"]:
        span = f"all {cl['frames']}"
    elif cl["lo"] == cl["hi"]:
        span = f"{cl['lo']}"
    else:
        span = f"{cl['lo']}-{cl['hi']}"
    return f"{cells}: {span}"


def cluster_key(frames):
    # No wider than the legend's widest entry: a longer key widened the legend
    # until no spot on the panel was clear of cells and routes.
    return f"cells: frames dark, of {frames}"


def _overlap(a, b):
    return a[0] < b[2] and b[0] < a[2] and a[1] < b[3] and b[1] < a[3]


def _crosses(p, q, box):
    """Whether segment p-q passes through the open rectangle box (Liang-Barsky)."""
    (x0, y0), (dx, dy) = p, (q[0] - p[0], q[1] - p[1])
    lo, hi = 0.0, 1.0
    for pk, qk in ((-dx, x0 - box[0]), (dx, box[2] - x0),
                   (-dy, y0 - box[1]), (dy, box[3] - y0)):
        if pk == 0:
            if qk <= 0:
                return False
            continue
        t = qk / pk
        if pk < 0:
            lo = max(lo, t)
        else:
            hi = min(hi, t)
        if lo >= hi:
            return False
    return True


def _segments_cross(p, q, a, b):
    """Proper intersection of segments p-q and a-b (shared endpoints excluded)."""
    def orient(u, v, w):
        return (v[0] - u[0]) * (w[1] - u[1]) - (v[1] - u[1]) * (w[0] - u[0])
    d1, d2 = orient(a, b, p), orient(a, b, q)
    d3, d4 = orient(p, q, a), orient(p, q, b)
    return d1 * d2 < 0 and d3 * d4 < 0


def _segment_hits(grid, p, q, skip=0.0):
    """Whether segment p-q passes over a marked cell of grid, ignoring the first
    `skip` (axes fraction) from p."""
    g = grid.shape[0]
    L = float(np.hypot(q[0] - p[0], q[1] - p[1]))
    if L <= skip:
        return False
    t = np.linspace(skip / L, 1.0, int(L * g * 2) + 2)
    j = np.clip((p[0] + t * (q[0] - p[0])) * g, 0, g - 1).astype(int)
    i = np.clip((p[1] + t * (q[1] - p[1])) * g, 0, g - 1).astype(int)
    return bool(grid[i, j].any())


LABEL_GRID = 256


def label_obstacles(ax, cells_km=(), cell_m=PX, lines_km=(), avoid=(), grid=LABEL_GRID,
                    line_px=6.0):
    """Where on the axes a label may not sit, and where a leader may not run,
    as two grids over axes fraction: (box, lead).

    A label may not sit on a labelled dark cell (it would hide what it names), on
    a plotted route sample widened by line_px points (enough to clear the
    rover's marker as it moves), or on any artist in `avoid` (the legend). A
    leader may cross dark cells, but not a route or the legend.
    """
    r = ax.figure.canvas.get_renderer()
    axb = ax.get_window_extent(r)
    occ = np.zeros((grid, grid), bool)
    lead = np.zeros((grid, grid), bool)

    def mark(fx0, fy0, fx1, fy1, also_lead=False):
        j0, j1 = int(np.floor(fx0 * grid)), int(np.ceil(fx1 * grid))
        i0, i1 = int(np.floor(fy0 * grid)), int(np.ceil(fy1 * grid))
        sl = (slice(max(i0, 0), max(min(i1, grid), 0)),
              slice(max(j0, 0), max(min(j1, grid), 0)))
        occ[sl] = True
        if also_lead:
            lead[sl] = True

    def to_frac(x, y):
        p = ax.transData.transform(np.column_stack([x, y]))
        return (p[:, 0] - axb.x0) / axb.width, (p[:, 1] - axb.y0) / axb.height
    if len(cells_km):
        c = np.asarray(cells_km, float)
        h = cell_m / 2000.0
        fx0, fy0 = to_frac(c[:, 0] - h, c[:, 1] - h)
        fx1, fy1 = to_frac(c[:, 0] + h, c[:, 1] + h)
        for v in zip(fx0, fy0, fx1, fy1):
            mark(*v)
    wpx = line_px * ax.figure.dpi / 72.0
    hx, hy = wpx / axb.width, wpx / axb.height
    for xs, ys in lines_km:
        xs, ys = np.asarray(xs, float), np.asarray(ys, float)
        # Resampled so a long straight segment is marked along its length too.
        d = np.hypot(np.diff(xs), np.diff(ys))
        u = np.concatenate([[0.0], np.cumsum(d)])
        n = max(2, int(u[-1] / max(1e-9, (ax.get_xlim()[1] - ax.get_xlim()[0]))
                       * grid * 2) + 2)
        uu = np.linspace(0.0, u[-1], n)
        fx, fy = to_frac(np.interp(uu, u, xs), np.interp(uu, u, ys))
        for x_, y_ in zip(fx, fy):
            mark(x_ - hx, y_ - hy, x_ + hx, y_ + hy, also_lead=True)
    for art in avoid:
        bb = art.get_window_extent(r)
        mark((bb.x0 - axb.x0) / axb.width, (bb.y0 - axb.y0) / axb.height,
             (bb.x1 - axb.x0) / axb.width, (bb.y1 - axb.y0) / axb.height,
             also_lead=True)
    return occ, lead


def _free(occ, box):
    g = occ.shape[0]
    j0, j1 = int(np.floor(box[0] * g)), int(np.ceil(box[2] * g))
    i0, i1 = int(np.floor(box[1] * g)), int(np.ceil(box[3] * g))
    if j0 < 0 or i0 < 0 or j1 > g or i1 > g:
        return False
    return not occ[i0:i1, j0:j1].any()


def _mark_box(occ, box):
    g = occ.shape[0]
    j0, j1 = max(int(np.floor(box[0] * g)), 0), min(int(np.ceil(box[2] * g)), g)
    i0, i1 = max(int(np.floor(box[1] * g)), 0), min(int(np.ceil(box[3] * g)), g)
    occ[i0:i1, j0:j1] = True


def _mark_segment(occ, p, q):
    g = occ.shape[0]
    n = int(np.hypot(q[0] - p[0], q[1] - p[1]) * g * 2) + 2
    t = np.linspace(0.0, 1.0, n)
    j = np.clip((p[0] + t * (q[0] - p[0])) * g, 0, g - 1).astype(int)
    i = np.clip((p[1] + t * (q[1] - p[1])) * g, 0, g - 1).astype(int)
    occ[i, j] = True


def _clear_boxes(occ, w, h):
    """Boolean grid over box lower-left corners: True where a w x h box (axes
    fraction) at that corner holds no occupied cell. Summed-area table, so every
    position is tested at once."""
    g = occ.shape[0]
    sat = np.zeros((g + 1, g + 1), np.int32)
    sat[1:, 1:] = occ.cumsum(0).cumsum(1)
    bw, bh = int(np.ceil(w * g)) + 1, int(np.ceil(h * g)) + 1
    if bw > g or bh > g:
        return np.zeros((0, 0), bool), bw, bh
    tot = sat[bh:, bw:] - sat[:-bh, bw:] - sat[bh:, :-bw] + sat[:-bh, :-bw]
    return tot == 0, bw, bh


def _leaders_clear(p, cx, cy, lead, boxes, leaders, samples=64):
    """For candidate leaders from p to each (cx, cy): True where the leader
    passes over no marked cell of lead (sampled), through no box in boxes, and
    across no leader in leaders. A filter, vectorised over every candidate; the
    exact checks run after it on what survives."""
    g = lead.shape[0]
    ok = np.ones(cx.shape, bool)
    t = np.linspace(0.0, 1.0, samples)[None, :]
    xs = p[0] + t * (cx[:, None] - p[0])
    ys = p[1] + t * (cy[:, None] - p[1])
    ok &= ~lead[np.clip((ys * g).astype(int), 0, g - 1),
                np.clip((xs * g).astype(int), 0, g - 1)].any(axis=1)
    dx, dy = cx - p[0], cy - p[1]
    for b in boxes:
        lo, hi = np.zeros(cx.shape), np.ones(cx.shape)
        dead = np.zeros(cx.shape, bool)
        for pk, qk in ((-dx, p[0] - b[0]), (dx, b[2] - p[0]),
                       (-dy, p[1] - b[1]), (dy, b[3] - p[1])):
            with np.errstate(divide="ignore", invalid="ignore"):
                tt = qk / pk
            zero = pk == 0
            dead |= zero & (qk <= 0)
            lo = np.where(~zero & (pk < 0), np.maximum(lo, tt), lo)
            hi = np.where(~zero & (pk > 0), np.minimum(hi, tt), hi)
        ok &= ~(~dead & (lo < hi))
    for a, b in leaders:
        def orient(ux, uy, vx, vy, wx, wy):
            return (vx - ux) * (wy - uy) - (vy - uy) * (wx - ux)
        d1 = orient(a[0], a[1], b[0], b[1], p[0], p[1])
        d2 = orient(a[0], a[1], b[0], b[1], cx, cy)
        d3 = orient(p[0], p[1], cx, cy, a[0], a[1])
        d4 = orient(p[0], p[1], cx, cy, b[0], b[1])
        ok &= ~((d1 * d2 < 0) & (d3 * d4 < 0))
    return ok


def plan_cluster_labels(ax, clusters, obst=None, fontsize=LABEL_FONT, pad=0.006,
                        margin=0.008, max_leader=0.35):
    """Where every cluster's label goes: (fontsize, [box per cluster]) in axes
    fraction, none overlapping, none on an obstacle, no leader through a label.

    The first placement offset each label a fixed distance from its cluster,
    which on R3's subject (26 clusters) stacked labels on one another; a second,
    two columns at the panel edges, kept the text apart but laid the left column
    over the clusters it named and ran leader arrows through other labels. Here
    each label goes to the clear position nearest its cluster: clear of the obstacles
    (dark cells, routes, legend, from label_obstacles), of every label already
    placed, padded, and of every leader already drawn, and with its own leader
    through no placed label, no route, not the legend and across no other
    leader. A label is kept within max_leader (axes fraction) of its cluster,
    since a longer leader is a poor label, and the leader conditions are
    applied to every candidate at once before the nearest survivor is checked
    exactly: looping over candidates one by one took two minutes on a crowded
    panel, and capping that loop at the nearest 600 cut off every legal spot
    for a cluster whose near side lay across the route. Every grid position is considered, not a fixed set
    of offsets. Two orders are tried, largest cluster first and most crowded
    first, each with any label that found no place moved to the front, then
    smaller text; if nothing fits it raises rather than overlap.
    The obstacles do not change between frames, so this runs once per render.
    """
    if not clusters:
        return fontsize, []
    r = ax.figure.canvas.get_renderer()
    axb = ax.get_window_extent(r)
    g = LABEL_GRID
    occ, lead = _obstacles(obst)
    edge = np.ones((g, g), bool)
    m = int(np.ceil(margin * g))
    edge[m:g - m, m:g - m] = False
    texts = [cluster_text(cl) for cl in clusters]
    pos = []
    for cl in clusters:
        px_, py_ = ax.transData.transform((cl["x"], cl["y"]))
        pos.append(((px_ - axb.x0) / axb.width, (py_ - axb.y0) / axb.height))
    P_ = np.asarray(pos)
    near = [(np.hypot(*(P_ - P_[q]).T) < 0.25).sum() for q in range(len(pos))]
    orders = [sorted(range(len(pos)), key=lambda q: -clusters[q]["cells"]),
              sorted(range(len(pos)), key=lambda q: (-near[q], -clusters[q]["cells"]))]

    def attempt(fs, order):
        wh = []
        for txt in texts:
            t = ax.text(0, 0, txt, fontsize=fs, transform=ax.transAxes)
            bb = t.get_window_extent(r)
            t.remove()
            wh.append((bb.width / axb.width + 2 * pad, bb.height / axb.height + 2 * pad))
        taken = occ | edge          # obstacles, placed labels (padded), leaders
        # Each cluster's own point too: a centroid need not lie on a dark cell,
        # and a label over one leaves that cluster no leader that misses it.
        for p in pos:
            _mark_box(taken, (p[0] - pad, p[1] - pad, p[0] + pad, p[1] + pad))
        boxes, placed, leaders = [None] * len(pos), [], []
        for q in order:
            w, h = wh[q]
            ok, bw, bh = _clear_boxes(taken, w, h)
            ii, jj = np.nonzero(ok)
            if ii.size == 0:
                return q
            x0, y0 = jj / g, ii / g
            # distance from the cluster to the nearest point of each box
            dx = np.maximum(np.maximum(x0 - pos[q][0], pos[q][0] - (x0 + w)), 0)
            dy = np.maximum(np.maximum(y0 - pos[q][1], pos[q][1] - (y0 + h)), 0)
            found = None
            dist = np.hypot(dx, dy)
            near_ = np.flatnonzero(dist <= max_leader)
            cx_ = x0[near_] + w / 2
            cy_ = y0[near_] + h / 2
            keep = _leaders_clear(pos[q], cx_, cy_, lead, placed, leaders)
            near_ = near_[keep]
            for k in near_[np.argsort(dist[near_], kind="stable")]:
                box = (x0[k] + pad, y0[k] + pad, x0[k] + w - pad, y0[k] + h - pad)
                c = ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2)
                if any(_crosses(pos[q], c, o) for o in placed):
                    continue
                if _segment_hits(lead, pos[q], c):
                    continue
                if any(_segments_cross(pos[q], c, a_, b_) for a_, b_ in leaders):
                    continue
                found = (box, c)
                break
            if found is None:
                return q
            box, c = found
            boxes[q] = box
            placed.append(box)
            leaders.append((pos[q], c))
            _mark_box(taken, (box[0] - pad, box[1] - pad, box[2] + pad, box[3] + pad))
            _mark_segment(taken, pos[q], c)
        return boxes

    for fs in np.arange(fontsize, 4.3, -0.3):
        for order in orders:
            # A label that finds no place goes to the front and the layout is
            # retried: clusters on the panel's edge have room on one side only,
            # and labels placed earlier below them caught every leader.
            for _ in range(3 * len(pos)):
                got = attempt(float(fs), order)
                if not isinstance(got, list):
                    order = [got] + [q for q in order if q != got]
                    continue
                if not label_faults(got, pos, obst):
                    return float(fs), got
                break
    raise RuntimeError(f"{len(clusters)} cluster labels do not fit the panel "
                       f"without overlapping at any size down to 4.4 pt")


def legend_anchor(ax, leg, obst, clusters, margin=0.012, pad=0.01):
    """Lower-left corner, in axes fraction, for a legend that covers no dark
    cell, route or cluster point: the clear position nearest any corner of the
    axes. The legend must have loc='lower left' and borderaxespad=0 so the
    anchor is its corner. Raises if nowhere is clear."""
    r = ax.figure.canvas.get_renderer()
    axb, bb = ax.get_window_extent(r), leg.get_window_extent(r)
    w, h = bb.width / axb.width, bb.height / axb.height
    occ = obst[0].copy()
    g = occ.shape[0]
    m = int(np.ceil(margin * g))
    occ[:m, :] = occ[-m:, :] = occ[:, :m] = occ[:, -m:] = True
    for cl in clusters:
        px_, py_ = ax.transData.transform((cl["x"], cl["y"]))
        fx, fy = (px_ - axb.x0) / axb.width, (py_ - axb.y0) / axb.height
        _mark_box(occ, (fx - pad, fy - pad, fx + pad, fy + pad))
    ok, _, _ = _clear_boxes(occ, w + 2 * pad, h + 2 * pad)
    ii, jj = np.nonzero(ok)
    if ii.size == 0:
        raise RuntimeError("no position for the legend clear of cells and routes")
    x0, y0 = jj / g + pad, ii / g + pad
    d = np.min([np.hypot(x0 - cx, y0 - cy) for cx, cy in
                ((0, 0), (1 - w, 0), (0, 1 - h), (1 - w, 1 - h))], axis=0)
    k = int(np.argmin(d))
    return float(x0[k]), float(y0[k])


def _obstacles(obst):
    g = LABEL_GRID
    if obst is None:
        return np.zeros((g, g), bool), np.zeros((g, g), bool)
    return obst


def draw_cluster_labels(ax, clusters, layout, obst=None):
    """Draw a planned layout, then check what was drawn: the measured text
    boxes go through label_faults again, and a fault raises."""
    import matplotlib.text as mtext
    fs, boxes = layout
    if not clusters:
        return []
    r = ax.figure.canvas.get_renderer()
    axb = ax.get_window_extent(r)
    anns, pos = [], []
    for cl, bx in zip(clusters, boxes):
        anns.append(ax.annotate(
            cluster_text(cl), xy=(cl["x"], cl["y"]),
            xytext=((bx[0] + bx[2]) / 2, (bx[1] + bx[3]) / 2),
            textcoords="axes fraction", ha="center", va="center",
            color="#e6e8ec", fontsize=fs, zorder=6,
            bbox=dict(boxstyle="square,pad=0.15", fc="#07070a", ec="none",
                      alpha=0.55),
            arrowprops=dict(arrowstyle="->", color="#e6e8ec", lw=0.6, alpha=0.8,
                            shrinkA=0, shrinkB=1)))
        px_, py_ = ax.transData.transform((cl["x"], cl["y"]))
        pos.append(((px_ - axb.x0) / axb.width, (py_ - axb.y0) / axb.height))
    # An annotation's text transform is applied at draw time, so it is updated
    # first: measured before that, every label read as one box off the axes.
    drawn = []
    for a in anns:
        a.update_positions(r)
        bb = mtext.Text.get_window_extent(a, r)
        drawn.append(((bb.x0 - axb.x0) / axb.width, (bb.y0 - axb.y0) / axb.height,
                      (bb.x1 - axb.x0) / axb.width, (bb.y1 - axb.y0) / axb.height))
    bad = label_faults(drawn, pos, obst)
    if bad:
        raise RuntimeError(f"cluster labels drawn with faults: {bad[:5]}")
    return anns


def place_cluster_labels(ax, clusters, obst=None, **kw):
    """Plan and draw in one call, for a single figure."""
    return draw_cluster_labels(ax, clusters, plan_cluster_labels(ax, clusters, obst, **kw),
                               obst)


def label_faults(boxes, points, obst=None):
    """Every way a set of drawn label boxes fails: two overlapping, one on an
    obstacle, one over a cluster's point, one label's leader (its cluster to its
    centre) through another label, across a route or the legend, or across
    another leader. Empty when the layout is clean. Shared by the placer and its test."""
    occ, lead = _obstacles(obst)
    cen = [((b[0] + b[2]) / 2, (b[1] + b[3]) / 2) for b in boxes]
    bad = []
    for a in range(len(boxes)):
        if _segment_hits(lead, points[a], cen[a]):
            bad.append(("leader on route or legend", a))
        if not _free(occ, boxes[a]):
            bad.append(("obstacle", a))
        for b, p in enumerate(points):
            if boxes[a][0] < p[0] < boxes[a][2] and boxes[a][1] < p[1] < boxes[a][3]:
                bad.append(("covers", a, b))
        for b in range(len(boxes)):
            if a < b and _overlap(boxes[a], boxes[b]):
                bad.append(("overlap", a, b))
            if a != b and _crosses(points[a], cen[a], boxes[b]):
                bad.append(("leader", a, b))
            if a < b and _segments_cross(points[a], cen[a], points[b], cen[b]):
                bad.append(("leaders cross", a, b))
    return bad


def _line_geometry(line):
    """A plotted line as drawn, in display coordinates: (segments, markers,
    marker radius, half line width), both in pixels. Step draw styles are
    expanded to their steps, and axhline/axvline go through their own blended
    transforms, so what is checked is what is drawn."""
    from matplotlib.cbook import STEP_LOOKUP_MAP
    xy = np.asarray(line.get_xydata(), float)
    if xy.size == 0:
        return np.zeros((0, 2, 2)), np.zeros((0, 2)), 0.0, 0.0
    ds = line.get_drawstyle()
    if ds in STEP_LOOKUP_MAP and ds != "default":
        sx, sy = STEP_LOOKUP_MAP[ds](xy[:, 0], xy[:, 1])
        path = np.column_stack([sx, sy])
    else:
        path = xy
    disp = line.get_transform().transform(path)
    dpi = line.figure.dpi
    seg = np.zeros((0, 2, 2))
    if line.get_linestyle() not in ("None", "", " ") and len(disp) > 1:
        seg = np.stack([disp[:-1], disp[1:]], axis=1)
        seg = seg[np.isfinite(seg).all(axis=(1, 2))]
    mk = np.zeros((0, 2))
    if line.get_marker() not in (None, "None", "", " "):
        mk = line.get_transform().transform(xy)
        mk = mk[np.isfinite(mk).all(axis=1)]
    return (seg, mk, line.get_markersize() * dpi / 72.0 / 2.0,
            line.get_linewidth() * dpi / 72.0 / 2.0)


def note_faults(ax, notes, pad_px=1.0):
    """Every plotted line that crosses a note: (note text, line label). Segments
    are tested exactly against the note's drawn text box widened by the line's
    half width and pad_px, and markers by their radius. Every visible line
    counts, the time cursor included."""
    r = ax.figure.canvas.get_renderer()
    bad = []
    for note in notes:
        bb = note.get_window_extent(r)
        for line in ax.get_lines():
            if not line.get_visible():
                continue
            seg, mk, mr, hw = _line_geometry(line)
            e = hw + pad_px
            box = (bb.x0 - e, bb.y0 - e, bb.x1 + e, bb.y1 + e)
            hit = any(_crosses(tuple(a), tuple(b), box) or
                      (box[0] < a[0] < box[2] and box[1] < a[1] < box[3])
                      for a, b in seg)
            hit = hit or any(bb.x0 - mr - pad_px < p[0] < bb.x1 + mr + pad_px and
                             bb.y0 - mr - pad_px < p[1] < bb.y1 + mr + pad_px
                             for p in mk)
            if hit:
                bad.append((note.get_text(), line.get_gid() or line.get_label()))
    return bad


NOTE_GRID = 200


def place_note(ax, text, prefer, ha="right", avoid=(), margin=0.015, pad=0.012,
               **kw):
    """A note on a plot, at the clear position nearest `prefer` (axes fraction,
    the anchor for `ha` at the text's baseline row), where no plotted line,
    marker, legend or other text in `avoid` or on the axes lies. Lines with gid
    "cursor" are not avoided here, because the cursor sweeps every time; it is
    drawn by draw_cursor with a gap at each note instead. Checked after drawing
    with note_faults; raises rather than draw a crossed note.

    The executive panel's false-alarm note and the shadow panel's budget note
    were fixed positions, and the traces ran through both."""
    fig = ax.figure
    r = fig.canvas.get_renderer()
    axb = ax.get_window_extent(r)
    g = NOTE_GRID
    occ = np.zeros((g, g), bool)
    m = int(np.ceil(margin * g))
    occ[:m, :] = occ[-m:, :] = occ[:, :m] = occ[:, -m:] = True

    def to_frac(p):
        return np.column_stack([(p[:, 0] - axb.x0) / axb.width,
                                (p[:, 1] - axb.y0) / axb.height])

    def mark_points(fp, rad):
        rj = int(np.ceil(rad / axb.width * g)) + 1
        ri = int(np.ceil(rad / axb.height * g)) + 1
        j = np.floor(fp[:, 0] * g).astype(int)
        i = np.floor(fp[:, 1] * g).astype(int)
        ok = (j >= -rj) & (j < g + rj) & (i >= -ri) & (i < g + ri)
        for di in range(-ri, ri + 1):
            for dj in range(-rj, rj + 1):
                ii, jj = i[ok] + di, j[ok] + dj
                inb = (ii >= 0) & (ii < g) & (jj >= 0) & (jj < g)
                occ[ii[inb], jj[inb]] = True
    for line in ax.get_lines():
        if not line.get_visible() or line.get_gid() == "cursor":
            continue
        seg, mk, mr, hw = _line_geometry(line)
        if len(seg):
            p, q = to_frac(seg[:, 0]), to_frac(seg[:, 1])
            L = np.hypot(*(q - p).T)
            n = np.ceil(L * g * 2).astype(int) + 2
            idx = np.repeat(np.arange(len(p)), n)
            start = np.repeat(np.cumsum(n) - n, n)
            t = (np.arange(n.sum()) - start) / np.repeat(n - 1, n)
            fp = p[idx] + t[:, None] * (q[idx] - p[idx])
            mark_points(fp, hw + 1.0)
        if len(mk):
            mark_points(to_frac(mk), mr + 1.0)
    others = [a for a in list(ax.texts) + list(avoid)]
    for a in others:
        bb = a.get_window_extent(r)
        _mark_box(occ, ((bb.x0 - axb.x0) / axb.width - pad, (bb.y0 - axb.y0) / axb.height - pad,
                        (bb.x1 - axb.x0) / axb.width + pad, (bb.y1 - axb.y0) / axb.height + pad))
    probe = ax.text(0, 0, text, transform=ax.transAxes, **kw)
    tb = probe.get_window_extent(r)
    probe.remove()
    w, h = tb.width / axb.width, tb.height / axb.height
    ok, _, _ = _clear_boxes(occ, w + 2 * pad, h + 2 * pad)
    ii, jj = np.nonzero(ok)
    if ii.size == 0:
        raise RuntimeError(f"no place on the axes where no line crosses {text!r}")
    x0, y0 = jj / g + pad, ii / g + pad
    ax_ = {"right": x0 + w, "left": x0, "center": x0 + w / 2}[ha]
    k = int(np.argmin(np.hypot(ax_ - prefer[0], y0 - prefer[1])))
    note = ax.text(float(ax_[k]), float(y0[k]), text, ha=ha, va="bottom",
                   transform=ax.transAxes, gid="note", **kw)
    bad = [b for b in note_faults(ax, [note]) if b[1] != "cursor"]
    if bad:
        raise RuntimeError(f"note placed across plotted lines: {bad}")
    return note


def draw_cursor(ax, t, notes, gap_px=3.0, **kw):
    """The current-time line, broken where it would cross a note."""
    import matplotlib.transforms as mtrans
    r = ax.figure.canvas.get_renderer()
    axb = ax.get_window_extent(r)
    tr = mtrans.blended_transform_factory(ax.transData, ax.transAxes)
    xd = ax.transData.transform((t, 0.0))[0]
    gaps = []
    for n in notes:
        bb = n.get_window_extent(r)
        if bb.x0 - gap_px <= xd <= bb.x1 + gap_px:
            gaps.append(((bb.y0 - gap_px - axb.y0) / axb.height,
                         (bb.y1 + gap_px - axb.y0) / axb.height))
    lo, pieces = 0.0, []
    for a, b in sorted(gaps):
        if a > lo:
            pieces.append((lo, a))
        lo = max(lo, b)
    if lo < 1.0:
        pieces.append((lo, 1.0))
    return [ax.plot([t, t], [a, b], transform=tr, gid="cursor", label="_nolegend_",
                    **kw)[0] for a, b in pieces]


def check_text_clear(fig, panel, others):
    """Refuse a frame whose text panel overlaps any of `others` (the captions)
    or runs past the figure's right edge."""
    r = fig.canvas.get_renderer()
    pb = panel.get_window_extent(r)
    hit = [o.get_text()[:40] for o in others
           if _overlap((pb.x0, pb.y0, pb.x1, pb.y1),
                       tuple(o.get_window_extent(r).extents))]
    if hit or pb.x1 > fig.bbox.x1:
        raise RuntimeError(f"the text panel runs into {hit or 'the right edge'}")


def check_notes(ax, notes):
    """Refuse a frame on which any plotted line, the cursor included, crosses a
    note."""
    bad = note_faults(ax, notes)
    if bad:
        raise RuntimeError(f"plotted lines cross notes: {bad}")


TRACE_Z, FIRE_Z = 4, 3


def exec_legs(arm, tag, plan_index):
    """Every leg's executive log for one arm, in order. A firing that replans
    starts a new leg (leg1_exec, ...); reading leg 0 alone drops them."""
    out = []
    while True:
        g = exec_log(arm, tag, plan_index, len(out))
        if g is None:
            return out
        out.append(g)


def draw_exec_traces(ax, traces):
    """The forecast for every rule, every leg, every tick, all visible.

    traces is [(arm, colour, label, legs)]. A leg that fires at its first tick
    is one point, which a 3 px marker under a 12 px firing cross hid entirely:
    the bounded rule on R3's subject fired at the first tick of both its legs.
    So every tick carries a 5 px marker drawn above the firing marks, and the
    firing marks are hollow so the tick shows through them. Returns the
    firings, (arm, label, leg, t, forecast), in order.
    """
    fires = []
    for arm, col, lab, legs in traces:
        n = len(legs)
        for q, g in enumerate(legs):
            name = lab if n == 1 else f"{lab}: {n} legs, replanned after each firing"
            ax.plot(g["t"], g["s_pred"], "-o", color=col, lw=1.6, ms=5,
                    zorder=TRACE_Z, gid=arm,
                    label=name if q == 0 else "_nolegend_")
            hit = np.flatnonzero(np.asarray(g["fired"]).astype(bool))
            if hit.size:
                k = int(hit[0])
                fires.append((arm, lab, q, float(g["t"][k]), float(g["s_pred"][k])))
    for i, (arm, lab, q, ft, fs) in enumerate(fires):
        ax.plot([ft], [fs], "X", mfc="none", mec=C_FIRE, mew=1.8, ms=14,
                zorder=FIRE_Z, gid=f"fire:{arm}",
                label=(f"control fires ({len(fires)}x), first at t={ft:.2f} h"
                       if i == 0 else "_nolegend_"))
    return fires


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
    p3 = p3_record() if p3 is None else p3
    mm_kind = meanmap_arm(p3)
    if rows.get("mm_kind", mm_kind) != mm_kind:
        raise ValueError(f"rows hold {rows['mm_kind']} but P3 selected {mm_kind}")
    ext_km = (REGION_OX / 1000.0, (REGION_OX + REGION_N * PX) / 1000.0,
              (REGION_OY - REGION_N * PX) / 1000.0, REGION_OY / 1000.0)
    rep_td = replay_out(tag, "td")
    plan, _, dep = plan_at(rep_td, rows["td"])
    pt = planner_out(tag, "td")
    off = pt["offsets"]
    lo, hi = int(off[instance]), int(off[instance + 1])
    tsteps = pt["times"][lo:hi]
    routes, same = {}, []
    for key, kind in (("td", "td"), ("snap", "snap"), ("mm", mm_kind)):
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
    nd = (~np.asarray(litm[:, i0:i1 + 1, j0:j1 + 1]).astype(bool)).sum(axis=0)
    di, dj = np.nonzero(nd >= cl_min)
    label_cells = np.column_stack(cell_to_map(di + i0, dj + j0)) / 1000.0

    # The executive, which is why R3 chose this instance. Read from the committed
    # five-arm run rather than recomputed.
    ex = {a: exec_legs(a, tag, rows["exec_plan_index"]) for a in ("none", "lit")}
    wtr = world_traj("none", tag, rows["exec_plan_index"])
    wsh = wtr["shadow"]
    wrun = np.maximum.accumulate(wsh)
    # The time axes span the traverse and not the frames being rendered, so a
    # short run for a look shows the same trace the full sequence does.
    t_end = max([float(wtr["t"][-1])] + [float(g["t"][-1]) for legs in ex.values()
                                         for g in legs])
    epochs_rows = select_r4()

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
    layout = obst = anchor = None
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
                    ("mm", (":", C_MM, 1.8, f"{meanmap_label(p3)} plan")),
                    ("td", ("-", C_TD, 1.8, "time-dependent, driven"))):
                if key in routes:
                    ax.plot(*routes[key], style, color=col, lw=lw,
                            solid_capstyle="round",
                            alpha=0.55 if key == "snap" else 1.0, label=lab)
            ax.plot([rx[step] / 1000.0], [ry[step] / 1000.0], "o", color=C_ROVER,
                    ms=10 if lim else 6, mec="#07070a", mew=1.2, label="rover")
            if lim is not None and clusters:
                ax.plot([], [], " ", label=cluster_key(int(litm.shape[0])))
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
            ax.set_xlabel("map x (km)", color="#9aa0aa", fontsize=7)
            ax.set_ylabel("map y (km)", color="#9aa0aa", fontsize=7)
            ax.tick_params(colors="#9aa0aa", labelsize=6.5)
            for sp in ax.spines.values():
                sp.set_color("#30343c")
            ax.set_title(title, color="#e6e8ec", fontsize=9)
            axes.append(ax)
        leg = axes[1].legend(loc="lower left", fontsize=6.5, facecolor="#101218",
                             edgecolor="#30343c", labelcolor="#d6d9de",
                             borderaxespad=0.0)
        leg.get_frame().set_alpha(0.92)
        # Every dark cluster is named, not only the darkest, so the grey levels
        # in this panel have a stated meaning everywhere. The legend goes first,
        # where it hides no dark cell, route or cluster (in the lower left
        # corner it covered a 10-cell cluster, which no leader could then
        # reach), and the labels are kept clear of it and of what they would
        # otherwise hide. Every frame has the same axes, legend, cells and
        # routes, so both are planned on the first frame and drawn, and
        # checked, on all of them.
        if layout is None:
            bare = label_obstacles(axes[1], cells_km=label_cells,
                                   lines_km=list(routes.values()))
            anchor = legend_anchor(axes[1], leg, bare, clusters)
        leg.set_bbox_to_anchor(anchor, transform=axes[1].transAxes)
        if layout is None:
            obst = label_obstacles(axes[1], cells_km=label_cells,
                                   lines_km=list(routes.values()), avoid=[leg])
            layout = plan_cluster_labels(axes[1], clusters, obst)
        draw_cluster_labels(axes[1], clusters, layout, obst)

        # --- the executive: the forecast against the budget, both arms ---
        axe = fig.add_axes(EXEC_RECT)
        axe.set_facecolor("#101218")
        fires = draw_exec_traces(axe, [
            ("none", C_SNAP, "bounded rule, no correction", ex["none"]),
            ("lit", C_LIT_ARM, "with the B3b-lit correction", ex["lit"])])
        axe.axhline(BUDGET_H, color=C_BUDGET, ls="--", lw=1.3,
                    label=f"budget {BUDGET_H:.1f} h")
        if fires:
            fire_t, fire_s = fires[0][3], fires[0][4]
            axe.annotate(f"forecast {fire_s:.2f} h", xy=(fire_t, fire_s),
                         xytext=(fire_t + 2.0, fire_s + 0.12), color=C_FIRE,
                         fontsize=7.5)
        axe.set_xlim(-0.6, t_end + 0.6)
        # Explicit, because the firing marker sits at the trace's maximum and an
        # auto-scaled axis clipped both it and its label off the top.
        top = max([BUDGET_H] + [float(g["s_pred"].max()) for legs in ex.values()
                                for g in legs]) * 1.22
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
        # Placed where no trace runs, and the cursor broken around it; the
        # fixed position it had sat on the lit arm's trace.
        exec_note = place_note(axe, f"the world never exceeds {wrun.max():.2f} h, "
                               f"so the firing is a false alarm", (0.985, 0.13),
                               ha="right", avoid=[le], color="#7e848e",
                               fontsize=7.5)
        draw_cursor(axe, t, [exec_note], color=C_ROVER, lw=1.2, alpha=0.8)
        check_notes(axe, [exec_note])

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
        # Placed where no trace runs: at the fixed position it had, the
        # running maximum's dashed line ran through it.
        shadow_note = place_note(axs, f"the {BUDGET_H:.1f} h budget is "
                                 f"{BUDGET_H / max(float(wrun.max()), 1e-9):.0f}x "
                                 f"off the top of this axis", (0.985, 0.04),
                                 ha="right", avoid=[ls], color="#7e848e",
                                 fontsize=7)
        draw_cursor(axs, t, [shadow_note], color=C_ROVER, lw=1.2, alpha=0.8)
        check_notes(axs, [shadow_note])

        axt = fig.add_axes(TEXT_RECT)
        axt.axis("off")
        panel = axt.text(0.0, 1.0, "\n".join(lines), color="#d6d9de", fontsize=7.6,
                         family="monospace", va="top", transform=axt.transAxes)

        fig.text(0.025, 0.955, f"{tag}   instance {instance}   t = {t:+.2f} h   "
                 f"{stack['lit_fraction'][k]:.2%} of the region lit   "
                 f"plan step {step + 1} of {plan.size}", color="#9aa0aa",
                 fontsize=10)
        caps = [fig.text(0.025, 0.068 - q * 0.024, line, color="#7e848e", fontsize=7)
                for q, line in enumerate(caption)]
        # The text panel grows with what it reports; a longer pairing statement
        # ran its last line into the first caption. Refused rather than drawn.
        check_text_clear(fig, panel, caps)
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
    """Forecast against the world, per arm, with the budget and every firing.

    Every leg is drawn, the world's and the executive's: a firing that replans
    continues in the next leg, and drawing leg 0 alone showed one point.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(12, 4.6 * len(arms)), dpi=dpi, facecolor="#07070a")
    for q, (arm, label) in enumerate(arms):
        legs = exec_legs(arm, tag, plan_index)
        ax = fig.add_subplot(len(arms), 1, q + 1)
        ax.set_facecolor("#101218")
        for n in range(max(1, len(legs))):
            w = world_traj(arm, tag, plan_index, n)
            ax.plot(w["t"], w["shadow"], "-", color="#6f7681", lw=1.2,
                    label="the world's accumulated shadow" if n == 0 else "_nolegend_")
        fires = draw_exec_traces(ax, [(arm, C_TD, "the executive's forecast maximum",
                                       legs)])
        if not fires:
            ax.plot([], [], " ", label="no control fires")
        ax.axhline(BUDGET_H, color="#ffa94d", ls="--", lw=1.2,
                   label=f"budget {BUDGET_H:.1f} h")
        ax.set_title(f"{label}", color="#e6e8ec", fontsize=10, loc="left")
        ax.set_xlabel("time since departure (h)", color="#9aa0aa")
        ax.set_ylabel("continuous shadow (h)", color="#9aa0aa")
        ax.tick_params(colors="#9aa0aa", labelsize=8)
        for sp in ax.spines.values():
            sp.set_color("#30343c")
        # Placed where it covers the least plotted data: in the upper left it
        # sat over R3's two firings, at the first tick of each leg.
        leg = ax.legend(loc="best", fontsize=8, facecolor="#101218",
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

    # The mean-map route is the exponent P3 selected, so the record is read
    # for every stage that draws or replays a route.
    rows = route_rows(tag, instance, meanmap_arm(p3_record()))
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
