"""Per-epoch illumination statistics and draw pool, over the twelve cached epochs.

The pilots all ran at one epoch, 2028-03-15, and its figures turned out not to
be representative. This reports every epoch the sweep will use.
"""
import sys
from pathlib import Path

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import examples.plan_pilot as P                                  # noqa: E402
from examples.build_intervals import HORIZON_H, cache_path, epochs  # noqa: E402
from src.cost import Rover, legal                                # noqa: E402
from src.intervals import LitIntervals                           # noqa: E402


def main():
    rover = Rover(v_flat=50.0)
    p, n, px, dem, horizon, cells, gx, gy, edges, graph = P.load(rover)
    ok = legal(edges["pitch"], edges["roll"], rover)
    ncell = n * n
    print(f"region {ncell} cells, {int(ok.sum())} legal directed edges, "
          f"horizon {HORIZON_H:.0f} h")
    print(f"  {'epoch':<16}{'trans/cell':>11}{'lit at t0':>11}{'>=2 trans':>11}"
          f"{'draw pool':>11}{'pool %':>9}")
    rows = []
    for tag, _, _ in epochs():
        path = cache_path(tag)
        if not path.exists():
            print(f"  {tag:<16}  MISSING")
            continue
        iv = LitIntervals.load(path)
        counts = np.diff(iv.offsets)
        changing = counts >= 2
        sel = ok & changing[edges["a"]] & changing[edges["b"]]
        gc = coo_matrix((np.ones(int(sel.sum())), (edges["a"][sel], edges["b"][sel])),
                        shape=(ncell, ncell))
        _, lab = connected_components(gc, directed=False)
        sz = np.bincount(lab)
        pool = int(sz.max()) if sz.size else 0
        if pool <= 1:
            pool = 0
        print(f"  {tag:<16}{counts.mean():>11.3f}{iv.lit0.mean():>10.2%}"
              f"{changing.mean():>11.2%}{pool:>11}{pool/ncell:>8.2%}")
        rows.append((tag, counts.mean(), iv.lit0.mean(), changing.mean(), pool))
    if rows:
        a = np.array([[r[1], r[2], r[3], r[4]] for r in rows])
        print(f"  {'across the twelve':<16}{a[:,0].mean():>11.3f}{a[:,1].mean():>10.2%}"
              f"{a[:,2].mean():>11.2%}{int(a[:,3].mean()):>11}{a[:,3].mean()/ncell:>8.2%}")
        print(f"  {'min':<16}{a[:,0].min():>11.3f}{a[:,1].min():>10.2%}"
              f"{a[:,2].min():>11.2%}{int(a[:,3].min()):>11}{a[:,3].min()/ncell:>8.2%}")
        print(f"  {'max':<16}{a[:,0].max():>11.3f}{a[:,1].max():>10.2%}"
              f"{a[:,2].max():>11.2%}{int(a[:,3].max()):>11}{a[:,3].max()/ncell:>8.2%}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
