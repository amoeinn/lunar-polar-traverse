"""Build and cache the lit-interval table for one epoch.

Interval tables are shared by every configuration at an epoch, so they are
built once and cached. PLAN.md lists the twelve epochs and the 720 h horizon.
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import spiceypy as sp

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import examples.plan_pilot as P                                  # noqa: E402
from src.cost import Rover                                       # noqa: E402
from src.intervals import LitIntervals, SunTable                 # noqa: E402

DATES = ("2028-02-15", "2028-05-15", "2028-08-15", "2028-11-15")
PHASE_OFFSET_DAYS = (0.0, 9.84, 19.69)          # a third of a synodic month apart
HORIZON_H = 720.0


def epochs():
    out = []
    for d in DATES:
        for k, off in enumerate(PHASE_OFFSET_DAYS):
            out.append((f"{d}_p{k}", f"{d}T00:00:00", off))
    return out


def cache_path(tag):
    return P.DATA / "work" / "intervals" / f"{tag}_{HORIZON_H:.0f}h.npz"


def sun_table_at(base_utc, offset_days, cadence_h=0.25):
    for k in P.KERNELS:
        sp.furnsh(str(P.DATA / "kernels" / k))
    t0 = sp.str2et(base_utc) + offset_days * 86400.0
    ets = np.arange(t0, t0 + (HORIZON_H + cadence_h) * 3600.0, cadence_h * 3600.0)
    pos = np.array([sp.spkpos("SUN", e, P.FRAME, "LT+S", "MOON")[0] for e in ets])
    sp.kclear()
    return SunTable(pos, cadence_h)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    args = ap.parse_args()
    table = {tag: (base, off) for tag, base, off in epochs()}
    if args.tag not in table:
        print(f"unknown tag {args.tag}; known: {sorted(table)}")
        return 2
    base, off = table[args.tag]
    rover = Rover(v_flat=50.0)
    p, n, px, dem, horizon, cells, gx, gy, edges, graph = P.load(rover)
    sun = sun_table_at(base, off)
    t0 = time.time()
    iv, cached = LitIntervals.build_or_load(cache_path(args.tag), cells,
                                            horizon.astype(np.float64), sun, HORIZON_H)
    counts = np.diff(iv.offsets)
    print(f"{args.tag}: {'cached' if cached else 'built'} in {time.time()-t0:.0f} s; "
          f"{int(counts.sum())} transitions, {counts.mean():.3f} per cell, "
          f"max {counts.max()}; lit at the epoch {iv.lit0.mean():.4%}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
