"""Driver for the C++ replay checker.

Writes the binary inputs, runs cpp/build/replay, reads the results back. It
holds no geometry of its own: everything the checker computes, it computes. The
only thing this module decides is what to hand it.
"""
import json
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
BINARY = ROOT / "cpp" / "build" / "replay"


OUT_KEYS = ("arrival", "shadow", "verdict", "illegal", "step_pitch",
            "step_roll", "step_pitch_mean", "step_roll_mean", "step_offsets")
# Outputs sized one per plan, and the four sized one per integration step.
PER_PLAN = {"arrival": "f8", "shadow": "f8", "verdict": "u1", "illegal": "i4"}
PER_STEP = ("step_pitch", "step_roll", "step_pitch_mean", "step_roll_mean")
# The inputs that identify which call a directory holds. sun_path is excluded on
# purpose: it is the same for every call at an epoch, so it discriminates
# nothing, and it is the largest file here.
IDENTIFYING = {"plan_offsets_path": np.int64, "plan_cells_path": np.int32,
               "plan_start_path": np.float64, "plan_wait_path": np.float64}


def reuse_problems(work, files, n_plans):
    """Why the outputs in `work` cannot be read back as this call's result.

    Existence is not completeness. cpp/build/replay writes its outputs at the
    end, so a call killed partway leaves either nothing or short files, and a
    short file read back is a wrong answer rather than a crash. Worse, a
    directory can hold a *different* call's outputs: the sweep driver used to
    pass one directory for all twelve of an epoch's calls, and the run killed on
    2026-11-16 left params describing a call that never wrote beside outputs
    from the call before it. So the inputs on disk must also be the ones this
    call would write, which is what ties the outputs to these plans.
    """
    work = Path(work)
    probs = []
    for key, dt in IDENTIFYING.items():
        if key not in files:
            continue
        f = work / f"{key}.bin"
        want = np.asarray(files[key], dtype=dt)
        if not f.exists():
            probs.append(f"{f.name} missing, so the outputs cannot be tied to "
                         f"these plans")
            continue
        got = np.fromfile(f, dtype=dt)
        if got.size != want.size:
            probs.append(f"{f.name} holds {got.size} values against "
                         f"{want.size} this call would write")
        elif not np.array_equal(got, want):
            probs.append(f"{f.name} is not the input this call would write, so "
                         f"the outputs belong to a different call")
    if probs:
        return probs
    off = work / "out_step_offsets.bin"
    if not off.exists():
        return ["out_step_offsets.bin missing"]
    if off.stat().st_size != (n_plans + 1) * 8:
        return [f"out_step_offsets.bin is {off.stat().st_size} bytes against "
                f"{(n_plans + 1) * 8} for {n_plans} plans"]
    steps = int(np.fromfile(off, dtype=np.int64)[-1])
    for k, dt in PER_PLAN.items():
        f = work / f"out_{k}.bin"
        want = n_plans * np.dtype(dt).itemsize
        if not f.exists():
            probs.append(f"out_{k}.bin missing")
        elif f.stat().st_size != want:
            probs.append(f"out_{k}.bin is {f.stat().st_size} bytes against "
                         f"{want} for {n_plans} plans")
    for k in PER_STEP:
        f = work / f"out_{k}.bin"
        if not f.exists():
            probs.append(f"out_{k}.bin missing")
        elif f.stat().st_size != steps * 4:
            probs.append(f"out_{k}.bin is {f.stat().st_size} bytes against "
                         f"{steps * 4} for {steps} steps")
    return probs


@dataclass
class ReplaySettings:
    integration_step_m: float = 1.0
    visibility_cadence_s: float = 60.0
    disc_radius_m: float = 30.0
    limit_tolerance_deg: float = 1.0e9      # until the registered value is measured


def replay(plans, starts, grid, rover, sun_positions, sun_dt_h, shadow_tol_h,
           work, settings=ReplaySettings(), waits=None, reuse=False):
    """Integrate every plan and report what the checker found.

    plans is a sequence of cell-index sequences; starts their departure times in
    hours from the sun table's own zero. grid carries the region and raster
    geometry. Returns arrays over plans, plus the checker's own per-step pitch
    and roll with offsets, which is what the tolerance measurement needs.
    """
    # Metres. cpp/src/replay.cpp subtracts an eye position built from R + h,
    # so the units have to match before the array is written to disk.
    from src.sun import check_sun_metres
    check_sun_metres(sun_positions, where="replay")
    work = Path(work)
    work.mkdir(parents=True, exist_ok=True)
    offsets = np.zeros(len(plans) + 1, dtype=np.int64)
    for i, pl in enumerate(plans):
        offsets[i + 1] = offsets[i] + len(pl)
    cells = np.concatenate([np.asarray(pl, dtype=np.int32) for pl in plans]) \
        if plans else np.zeros(0, np.int32)

    files = {
        "plan_offsets_path": offsets,
        "plan_cells_path": cells,
        "plan_start_path": np.asarray(starts, dtype=np.float64),
        "sun_path": np.ascontiguousarray(sun_positions, dtype=np.float64).ravel(),
    }
    if waits is not None:
        files["plan_wait_path"] = np.concatenate(
            [np.asarray(w, dtype=np.float64) for w in waits])

    if reuse and not reuse_problems(work, files, len(plans)):
        # A finished call left its outputs here, for these plans, at the sizes
        # they must have. Reading them back rather than repeating the
        # integration is what lets a long sweep resume after an interruption
        # without losing what it had already done.
        return _read_outputs({f"out_{k}": str(work / f"out_{k}.bin")
                              for k in OUT_KEYS})
    params = dict(grid)
    for key, arr in files.items():
        path = work / f"{key}.bin"
        path.write_bytes(arr.tobytes())
        params[key] = str(path)
    params.update(n_plans=len(plans), n_sun=int(np.asarray(sun_positions).shape[0]),
                  sun_dt_hours=float(sun_dt_h), shadow_tol_h=float(shadow_tol_h),
                  **{k: float(v) for k, v in asdict(settings).items()})
    params.update(v_flat=rover.v_flat, mu=rover.mu, cap_ratio=rover.cap_ratio,
                  roll_ref=rover.roll_ref, roll_exponent=rover.roll_exponent,
                  pitch_max=rover.pitch_max, roll_max=rover.roll_max)
    for k in OUT_KEYS:
        params[f"out_{k}"] = str(work / f"out_{k}.bin")
    (work / "replay_params.json").write_text(json.dumps(params, indent=2))
    subprocess.run([str(BINARY), str(work / "replay_params.json")], check=True,
                   capture_output=True)
    return _read_outputs(params)


def _read_outputs(params):
    """Read the checker's outputs, whether just written or found from before."""
    return dict(
        arrival=np.fromfile(params["out_arrival"], dtype=np.float64),
        shadow=np.fromfile(params["out_shadow"], dtype=np.float64),
        verdict=np.fromfile(params["out_verdict"], dtype=np.uint8).astype(bool),
        illegal=np.fromfile(params["out_illegal"], dtype=np.int32),
        step_pitch=np.fromfile(params["out_step_pitch"], dtype=np.float32),
        step_roll=np.fromfile(params["out_step_roll"], dtype=np.float32),
        step_pitch_mean=np.fromfile(params["out_step_pitch_mean"], dtype=np.float32),
        step_roll_mean=np.fromfile(params["out_step_roll_mean"], dtype=np.float32),
        step_offsets=np.fromfile(params["out_step_offsets"], dtype=np.int64))
