"""Check 3's time-expanded reference. Independent of the planner by construction.

Nodes are (cell, time step, shadow step) at step dt. A move takes ceil(tau/dt)
steps, rounded up so the reference is pessimistic and cannot beat the true
optimum by a discretisation artefact. A wait takes one step and is allowed only
where the cell is lit throughout it.

It evaluates lit and dark directly from the sun and the horizon and reads no
interval table, so it shares no illumination machinery with src/planner.py.
See PLAN.md, "Planners", for the registered inequality.

Its oracle is dark(cell, k): is the cell dark anywhere within step k. Anywhere,
not at the sample time, and that is the difference between a reference and a
coincidence. Sampling lit/dark at step boundaries can miss a dark span inside a
step, which relaxes the shadow constraint and lets the reference beat the true
optimum -- the registered inequality then fails for a reason that has nothing
to do with the planner. With the conservative oracle the reference is a
restriction of the true problem on both counts: travel rounded up is a slower
traverse, which is always allowed, and shadow counted over whole steps is never
less than the truth.
"""
import heapq
import math

import numpy as np


def plan_time_expanded(graph, dark, start, goal, t_start, shadow_tol, dt, horizon_h):
    """Earliest arrival on the time-expanded graph.

    dark(cell, k) must be True if the cell is dark anywhere in step k, that is
    anywhere in [t_start + k dt, t_start + (k+1) dt]. Returns hours, or None.
    """
    n_steps = int(math.floor((horizon_h - t_start) / dt))
    shadow_cap = int(math.floor(shadow_tol / dt))
    seen = set()
    heap = [(0, start, 0)]                   # (time step, cell, shadow steps)
    while heap:
        k, cell, j = heapq.heappop(heap)
        if cell == goal:
            return t_start + k * dt
        if (cell, k, j) in seen:
            continue
        seen.add((cell, k, j))
        if k + 1 <= n_steps and not dark(cell, k) and j == 0:
            if (cell, k + 1, 0) not in seen:
                heapq.heappush(heap, (k + 1, cell, 0))
        nb, tau = graph.neighbours(cell)
        for c2, tt in zip(nb, tau):
            c2 = int(c2)
            steps = max(1, int(math.ceil(tt / dt)))
            k2 = k + steps
            if k2 > n_steps:
                continue
            j2, jmax = j, j
            for m in range(k, k2):
                if dark(cell, m) or dark(c2, m):
                    j2 += 1
                    jmax = max(jmax, j2)
                else:
                    j2 = 0
            if jmax > shadow_cap:
                continue
            if (c2, k2, j2) not in seen:
                heapq.heappush(heap, (k2, c2, j2))
    return None
