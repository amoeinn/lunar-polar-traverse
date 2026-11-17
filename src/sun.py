"""The one place a SPICE sun position enters this project, and its unit guard.

SPICE spkpos returns KILOMETRES. Every frame decomposition here subtracts a cell
origin in METRES, on the 1737400 m sphere, so a km value mixes units and inflates
the parallax term by a thousand. Measured on 2026-11-16: the sun came out
0.67218 deg too LOW, at every cell and every epoch, while the azimuth was exact
to 5.7e-14 deg.

The azimuth being exact is structural rather than lucky. cell_position is
radial, so the origin is parallel to the frame's up and subtracting it perturbs
only the up component. That is why the fault survived every azimuth check in
this record and showed up only in elevation.

Seven producers each called spkpos and passed the result on unscaled. They now
all call sun_positions_m, so there is one conversion in the project and a test
can hold it.
"""
import numpy as np

KM_TO_M = 1000.0

# The Moon to Sun distance over 2009 to 2030, sampled every 6 h, runs
# 1.466997e11 to 1.524942e11 m. The band below clears that by about 1.7e10 m on
# each side, and is tight enough to reject both ways of getting it wrong:
# kilometres come in at 1.47e8 and a double conversion at 1.47e14.
SUN_DISTANCE_MIN_M = 1.3e11
SUN_DISTANCE_MAX_M = 1.7e11


def check_sun_metres(positions, where=""):
    """Raise unless every sun position looks like metres.

    Checks the magnitude rather than the units, because units are not carried in
    the array. A km array fails low by three orders and a doubly converted one
    fails high by three, so the band catches the two mistakes that are actually
    available here rather than guarding against the arbitrary.
    """
    p = np.atleast_2d(np.asarray(positions, dtype=float))
    if p.ndim != 2 or p.shape[-1] != 3:
        raise ValueError(f"sun positions must be (n, 3){_at(where)}, got {p.shape}")
    d = np.linalg.norm(p, axis=-1)
    bad = (d < SUN_DISTANCE_MIN_M) | (d > SUN_DISTANCE_MAX_M) | ~np.isfinite(d)
    if bad.any():
        i = int(np.flatnonzero(bad)[0])
        raise ValueError(
            f"sun position {i} has magnitude {d[i]:.6e} m{_at(where)}, outside "
            f"[{SUN_DISTANCE_MIN_M:.3e}, {SUN_DISTANCE_MAX_M:.3e}] m. SPICE "
            f"spkpos returns kilometres; this project works in metres.")
    return p


def _at(where):
    return f" at {where}" if where else ""


def sun_positions_m(ets, frame, target="SUN", observer="MOON", abcorr="LT+S"):
    """Moon-fixed sun positions in METRES, for one or many ephemeris times.

    Kernels must already be furnished; this does not manage them, because the
    callers differ in which set they load and when they clear it.
    """
    import spiceypy as sp
    ets = np.atleast_1d(np.asarray(ets, dtype=float))
    pos = np.array([sp.spkpos(target, float(e), frame, abcorr, observer)[0]
                    for e in ets], dtype=float) * KM_TO_M
    return check_sun_metres(pos, where="sun_positions_m")
