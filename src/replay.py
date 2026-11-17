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
    out_keys = ("arrival", "shadow", "verdict", "illegal", "step_pitch",
                "step_roll", "step_pitch_mean", "step_roll_mean", "step_offsets")
    if reuse:
        # A finished run left its outputs here. Reading them back rather than
        # repeating the integration is what lets a long sweep resume after an
        # interruption without losing what it had already done.
        found = {f"out_{k}": str(work / f"out_{k}.bin") for k in out_keys}
        if all(Path(v).exists() for v in found.values()):
            return _read_outputs(found)
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
    for k in ("arrival", "shadow", "verdict", "illegal", "step_pitch", "step_roll",
              "step_pitch_mean", "step_roll_mean", "step_offsets"):
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
