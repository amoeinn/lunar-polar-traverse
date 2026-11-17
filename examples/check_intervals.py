"""Verify cached lit-interval tables against the region and the registered set.

    .venv/bin/python examples/check_intervals.py [<tag> ...]

With no tags it checks the twelve registered epochs. This exists because a
rebuild of the interval tables is the one step in the pipeline whose failure is
silent: LitIntervals.build_or_load returns a cached table whenever the cell
count and the horizon match, so a table written by an earlier run, or a partial
write from an interrupted batch, is reused without complaint and every
downstream stage then reports success on it. The invariant protected here is
that each cached table covers exactly the region's cells over exactly the
registered horizon, and that the directory holds the registered set and nothing
else.

Read only; writes nothing. Where a pre-fix copy exists under
data/work/pre_sunfix/intervals, each figure is printed beside it, since the
direction of the change is what the D132 predictions are scored on.
"""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.intervals import LitIntervals          # noqa: E402

NEW = ROOT / "data" / "work" / "intervals"
OLD = ROOT / "data" / "work" / "pre_sunfix" / "intervals"
CELLS = 250000
HORIZON_H = 720.0
DATES = ("2028-02-15", "2028-05-15", "2028-08-15", "2028-11-15")
REGISTERED = tuple(f"{d}_p{k}" for d in DATES for k in range(3))


def figures(path):
    """Transitions, lit fraction at the epoch, and mean lit fraction over the
    horizon. The last is the committed LitIntervals.lit_fraction rather than a
    faster equivalent, so there is one implementation of the quantity."""
    iv = LitIntervals.load(path)
    return iv, int(np.diff(iv.offsets).sum()), float(iv.lit0.mean()), \
        float(iv.lit_fraction().mean())


def _shown(path):
    """Repo-relative where possible, absolute otherwise, so a redirected
    directory prints rather than raising."""
    try:
        return path.relative_to(ROOT)
    except ValueError:
        return path


def problems_with(iv):
    out = []
    if iv.offsets.size != CELLS + 1:
        out.append(f"offsets {iv.offsets.size} != {CELLS + 1}")
    if iv.lit0.size != CELLS:
        out.append(f"lit0 {iv.lit0.size} != {CELLS}")
    if abs(iv.horizon_h - HORIZON_H) > 1e-9:
        out.append(f"horizon {iv.horizon_h} != {HORIZON_H}")
    if not np.all(np.diff(iv.offsets) >= 0):
        out.append("offsets not monotone")
    if iv.offsets.size and int(iv.offsets[-1]) != iv.times.size:
        out.append(f"offsets end {int(iv.offsets[-1])} != times {iv.times.size}")
    if iv.times.size and not np.isfinite(iv.times).all():
        out.append("non-finite transition times")
    if iv.times.size and (iv.times.min() < 0.0 or iv.times.max() > HORIZON_H):
        out.append(f"times outside [0, {HORIZON_H}]")
    return out


def main():
    tags = sys.argv[1:] or list(REGISTERED)
    if not NEW.is_dir():
        print(f"{NEW} does not exist; run examples/fetch_data.sh and the "
              f"interval build first")
        return 2

    print(f"{'tag':18s}{'MB':>8}{'was':>8}{'transitions':>13}{'was':>12}"
          f"{'lit at t0':>11}{'was':>9}{'mean lit':>10}{'was':>9}  status")
    bad = 0
    for t in tags:
        f = NEW / f"{t}_720h.npz"
        if not f.exists():
            print(f"{t:18s}{'-':>8}  MISSING {f}")
            bad += 1
            continue
        mb = f.stat().st_size / 2**20
        o = OLD / f"{t}_720h.npz"
        omb = o.stat().st_size / 2**20 if o.exists() else float("nan")
        try:
            iv, n, lit, mean = figures(f)
        except Exception as e:                   # noqa: BLE001
            print(f"{t:18s}{mb:8.2f}  FAILED TO LOAD: {e}")
            bad += 1
            continue
        on, olit, omean = float("nan"), float("nan"), float("nan")
        if o.exists():
            try:
                _, on, olit, omean = figures(o)
            except Exception as e:               # noqa: BLE001
                print(f"{t:18s}  pre-fix copy failed to load: {e}")
        probs = problems_with(iv)
        if probs:
            bad += 1
        print(f"{t:18s}{mb:8.2f}{omb:8.2f}{n:13d}{on:12.0f}"
              f"{lit:11.4%}{olit:9.4%}{mean:10.4%}{omean:9.4%}  "
              f"{'ok' if not probs else 'BAD: ' + '; '.join(probs)}")

    print()
    print(f"{len(tags) - bad} of {len(tags)} tags verified")
    have = sorted(p.name.replace("_720h.npz", "") for p in NEW.glob("*_720h.npz"))
    extra = [h for h in have if h not in REGISTERED]
    missing = [r for r in REGISTERED if r not in have]
    print(f"{_shown(NEW)} holds {len(have)} tables; "
          f"the registered set is {len(REGISTERED)}")
    if extra:
        print(f"  NOT REGISTERED, and will be picked up by a cache hit: {extra}")
        bad += 1
    if missing:
        print(f"  registered but absent: {missing}")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
