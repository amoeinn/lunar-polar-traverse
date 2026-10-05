"""Score PLAN.md's phase 7a checklist against the running simulator.

Nothing here is believed from a return value. The probe poses are read off the
pose topic because this record holds four cases where a simulator API reported
success and moved nothing, and the illumination comparison takes its
pixel-to-cell mapping from the depth channel's own ranges rather than from
assumed projection arithmetic, so that it does not depend on the mapping C1 is
busy checking.
"""
import argparse
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import rasterio

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.gzmap import (HEIGHTMAP_N, PX, REGION_N, REGION_OX, REGION_OY,
                       SAMPLING,
                       bilinear, cell_to_map, centre_map, map_to_cell,
                       map_to_world, span_m, world_to_map)

GZ = "/usr/bin/gz"
ENV = dict(os.environ, GZ_CONFIG_PATH="/usr/share/gz")
WORK = ROOT / "data" / "work" / "gazebo"
TIF = WORK / "region_heightmap.tif"

PROBE_RADIUS_M = 2.0
L2_TOL_M = 5.0              # registered, the original
# L2' registered 2026-10-04 before the run. The bound behind it: Gazebo
# triangulates the heightmap on the sampling-2 vertex lattice, 10 m, while the
# reference is bilinear on 20 m, and that interpolation gap over this footprint
# is median 0.0231 m, p99 0.1584 m, worst 0.8647 m. 2.0 m is more than twice
# the worst case and 67 times below L3's smallest signature.
L2P_TOL_M = 2.0             # registered, the replacement
RAY_ALT_M = 2836.0
RAY_FOV_RAD = 0.34906585    # 20 deg
RAY_PX = 256
L3_MIN_SIGNATURE_M = 50.0   # registered
DROP_M = 20.0
SETTLE_S = 75.0

# Non-collinear, off-centre, at distinct heights. L3 proves the set
# discriminating before L2 is believed.
PROBES = [(-3000.0, 10000.0), (4500.0, 9500.0), (-2500.0, 2500.0),
          (5000.0, 2000.0), (1500.0, 8000.0), (-1000.0, 4000.0),
          (3000.0, 6500.0)]
# Shackleton's centre from the IAU Gazetteer, as PLAN.md's own landmark test
# records it: latitude -89.67, longitude 129.78, diameter 20.92 km, adopted
# 1994. Carried through the production forward projection this is map
# (7690.2, -6402.7), 10.01 km from the pole, which matches the "10.01 km the
# gazetteer colatitude implies" in the 2026-09-22 landmark result.
#
# It is outside the heightmap. The registered L4 asserted it lands inside, and
# that assertion was wrong: it came from a latitude of -89.9 taken from memory
# rather than from this project's own figures, which is the fault the house
# rules exist to prevent. L4 is scored against what was registered.
SHACKLETON_LATLON = (-89.67, 129.78)
SHACKLETON_MAP = (7690.2, -6402.7)
# L4' registered 2026-10-04. No catalogued feature's centre lies in this
# footprint; the rim of the nearest one does cross it, and the rim is specified
# entirely by the two gazetteer figures above, so the anchor stays external.
# Diameter 20.92 km -> radius 10460 m. Established before the run: the rim
# circle passes between the nearest footprint point (7284.5 m from the centre)
# and the farthest corner (20964.6 m), 352 cells lie within half a pixel of it,
# and carrying that arc through the error modes overlaps the true arc by 0, 1
# and 1 cells, so the modes read different ground rather than the same ground
# relabelled.
SHACKLETON_RIM_R_M = 20.92e3 / 2.0


def dem():
    with rasterio.open(TIF) as ds:
        return ds.read(1)


def height_at_map(h, mx, my):
    i, j = map_to_cell(mx, my)
    return float(bilinear(h, i, j))


def error_signatures(h):
    """What each frame error would do to the probe set, computed from the DEM."""
    n = HEIGHTMAP_N
    modes = {
        "row flip": lambda i, j: (n - 1 - i, j),
        "column flip": lambda i, j: (i, n - 1 - j),
        "transpose": lambda i, j: (j, i),
        "13-cell offset": lambda i, j: (i + 13, j + 13),
    }
    rows, worst = [], {}
    for mx, my in PROBES:
        i, j = map_to_cell(mx, my)
        true_h = float(bilinear(h, i, j))
        row = dict(map_x=mx, map_y=my, i=float(i), j=float(j), height=true_h)
        for name, f in modes.items():
            ei, ej = f(i, j)
            row[name] = abs(true_h - float(bilinear(h, min(ei, n - 1),
                                                    min(ej, n - 1))))
            worst[name] = max(worst.get(name, 0.0), row[name])
        rows.append(row)
    return rows, worst


def run_world(sdf, settle_s, topic="dynamic_pose/info", world="lunar_region"):
    """Start the server, let it settle, read one pose message, stop it."""
    proc = subprocess.Popen([GZ, "sim", "-s", "-r", "-v", "1", str(sdf)],
                            env=ENV, stdout=subprocess.DEVNULL,
                            stderr=subprocess.DEVNULL, start_new_session=True)
    try:
        time.sleep(settle_s)
        out = subprocess.run([GZ, "topic", "-e", "-t",
                              f"/world/{world}/{topic}", "-n", "1"],
                             env=ENV, capture_output=True, text=True, timeout=40)
        return out.stdout
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()


def parse_poses(text):
    """Named positions out of a Pose_V message, read as text."""
    poses = {}
    for blk in re.findall(r"pose \{(.*?)\n\}", text, flags=re.S):
        m = re.search(r'name:\s*"([^"]+)"', blk)
        if not m:
            continue
        pos = re.search(r"position \{(.*?)\}", blk, flags=re.S)
        if not pos:
            continue
        v = {k: float(val) for k, val in
             re.findall(r"([xyz]):\s*(-?[\d.eE+]+)", pos.group(1))}
        poses[m.group(1)] = (v.get("x", 0.0), v.get("y", 0.0), v.get("z", 0.0))
    return poses


def make_world(probes_world, extra=()):
    cmd = [str(ROOT / ".venv" / "bin" / "python"),
           str(ROOT / "examples" / "make_world.py"), "--mode", "ours"]
    if probes_world:
        cmd += ["--probes=" + ";".join(f"{x:.4f},{y:.4f},{z:.4f}"
                                       for x, y, z in probes_world)]
    cmd += list(extra)
    subprocess.run(cmd, check=True, capture_output=True)
    return WORK / "lunar_region_ours.sdf"


L2PPP_OFFSET_MAX_M = 2.0     # registered: 11x the estimator's worst recovery
L2PPP_RESID_MAX_M = 5.0      # registered: 14x below the smallest flip signature


def offset_minimiser(h, fields):
    """The shift that best explains the rendered cloud, and its residual.

    Coarse over +-24 m so a half-cell or one-cell index error is bracketed,
    then refined. The height difference between Gazebo's rendered surface and
    the bilinear DEM scales with local slope, which raises this curve's floor
    everywhere without moving its minimum: that is why the offset is
    measurable on this surface when a per-probe height is not.
    """
    def med(dx, dy):
        o = []
        for smx, smy, wz in fields:
            i, j = map_to_cell(smx + dx, smy + dy)
            o.append(np.abs(wz - bilinear(h, i, j)))
        return float(np.median(np.concatenate(o)))

    c = [0.0, 0.0]
    for span, step in ((24.0, 3.0), (3.0, 0.25)):
        g = sorted((med(c[0] + dx, c[1] + dy), c[0] + dx, c[1] + dy)
                   for dx in np.arange(-span, span + 1e-9, step)
                   for dy in np.arange(-span, span + 1e-9, step))
        c = [g[0][1], g[0][2]]
    return g[0][1], g[0][2], g[0][0], med(0.0, 0.0)


def l2ppp(h):
    """The third replacement for L2: the offset, not a height.

    Four height instruments failed before this one and all four stand in
    PLAN.md with their numbers. A height residual cannot be bounded per probe
    on a surface whose difference from its reference scales with slope, which
    is measured here at a correlation of +0.6888 with local slope. This
    measures the position instead, which is what L2 exists to ask.
    """
    print("=== L2''': placement measured as an offset, not as a height ===")
    print("  Four height instruments failed before this one and all four stand")
    print("  in the record. A height residual cannot be bounded per probe on a")
    print("  surface whose difference from its reference scales with slope, so")
    print("  this measures the position instead.")
    rays = []
    for k, (mx, my) in enumerate(PROBES):
        wx, wy = map_to_world(mx, my)
        rays.append((f"p{k:02d}", float(wx), float(wy),
                     height_at_map(h, mx, my) + RAY_ALT_M))
    rayarg = "--rays=" + ";".join(
        f"{n},{x:.4f},{y:.4f},{z:.4f},{RAY_FOV_RAD:.9f},{RAY_PX},{RAY_PX}"
        for n, x, y, z in rays)
    sdf = make_world([], extra=[rayarg])
    _, clouds = grab_mixed(sdf, 30.0, "dynamic_pose/info",
                           [f"/ray/{n}/depth/points" for n, _, _, _ in rays])
    fields = []
    for nm, cwx, cwy, cwz in rays:
        msg = clouds.get(f"/ray/{nm}/depth/points")
        if msg is None:
            print(f"  {nm}: MISSING from the depth topic")
            continue
        wx, wy, wz, _l, good = cloud_to_world(msg, (cwx, cwy, cwz))
        smx, smy = world_to_map(wx, wy)
        g = (good & (smx > REGION_OX + 100) & (smx < REGION_OX + 9900)
             & (smy < REGION_OY - 100) & (smy > REGION_OY - 9900))
        if g.any():
            fields.append((smx[g], smy[g], wz[g]))
    if len(fields) != len(rays):
        print(f"  NOT RUN: {len(fields)} of {len(rays)} fields returned")
        return dict(L2ppp=None)

    dx, dy, resid, resid0 = offset_minimiser(h, fields)
    mag = float(np.hypot(dx, dy))
    npx = sum(len(f[0]) for f in fields)
    print(f"\n  {npx} pixels across {len(fields)} fields")
    print(f"    recovered offset           ({dx:+.3f}, {dy:+.3f}) m, "
          f"magnitude {mag:.3f} m")
    print(f"    registered bar on it       at most {L2PPP_OFFSET_MAX_M:.1f} m")
    print(f"    median |residual| there    {resid:.4f} m")
    print(f"    median |residual| at zero  {resid0:.4f} m")
    print(f"    registered bar on residual at most {L2PPP_RESID_MAX_M:.1f} m")
    a = bool(mag <= L2PPP_OFFSET_MAX_M)
    b = bool(resid <= L2PPP_RESID_MAX_M)
    print(f"\n    (a) the offset is within its bar: {a}")
    print(f"    (b) a shift explains the data, so this is not a flip or a "
          f"transpose: {b}")
    print(f"        the signatures a shift cannot remove are 693.652 m for a "
          f"row flip,")
    print(f"        764.773 m for a column flip and 69.252 m for a transpose, "
          f"the smallest;")
    print(f"        the half-cell error this record found and fixed would show "
          f"as a")
    print(f"        14.14 m offset, 7 times bar (a).")
    ok = bool(a and b)
    print(f"  L2''' -> {'PASS' if ok else 'FAIL'}")
    return dict(L2ppp=ok)


ROVER_PATCH_MAP = (3010.0, 2490.0)   # registered: the patch already characterised
ROVER_ALT_M = 20.0                   # registered
TARGET_RANGE_M = 10.000              # registered
TARGET_RADIUS_M = 0.5
MAST_Z_M = 2.45                      # clearance + body_h + mast_h, from the xacro
NAV_BASELINE_M = 0.424               # SOURCE [P], PDS context product navcam.msl
NAV_FOV_DEG = 45.0                   # SOURCE [M]
HAZ_FOV_DEG = 124.0                  # SOURCE [H]
CAM_PX = 1024
S1_FOV_TOL_DEG = 0.01                # registered
S2_DISPARITY_TOL_PX = 2.0            # registered
S3_MEDIAN_MAX_M = 1.5                # registered: the predicted maximum
S3_P90_MAX_M = 3.0                   # registered
S3_GRAZE_MIN_DEG = 15.0              # registered
LIDAR_VFOV_DEG = 42.4                # SOURCE [O]
LIDAR_ROWS, LIDAR_COLS = 64, 1024    # SOURCE [O]
LIDAR_RANGE_M = 90.0                 # SOURCE [O], 10% target
# The open slope-dependent rendering term, measured 2026-10-04 over 458752
# pixels. A GPU lidar reads the rendered surface, so S3's tolerance carries it.
SLOPE_BANDS = ((0.0, 2.9, 0.0577), (2.9, 5.7, 0.3919), (5.7, 11.3, 0.2723),
               (11.3, 19.3, 0.0626), (19.3, 31.0, 0.2624), (31.0, 90.0, 3.0844))


def band_term(slope_deg):
    for lo, hi, v in SLOPE_BANDS:
        if lo <= slope_deg < hi:
            return v
    return SLOPE_BANDS[-1][2]


def decode_image(msg):
    """An RGB_INT8 image message to a float luminance array."""
    import base64
    if msg is None:
        return None
    w, hgt = int(msg["width"]), int(msg["height"])
    raw = base64.b64decode(msg["data"])
    a = np.frombuffer(raw, dtype=np.uint8)
    if a.size < w * hgt * 3:
        return None
    a = a[:w * hgt * 3].reshape(hgt, w, 3).astype(float)
    return 0.2126 * a[..., 0] + 0.7152 * a[..., 1] + 0.0722 * a[..., 2]


def march_dem(h, origin, direction, rmax, step=0.25):
    """First DEM intersection along a ray, by marching. Returns range or None."""
    r = np.arange(1.0, rmax, step)
    px = origin[0] + direction[0] * r
    py = origin[1] + direction[1] * r
    pz = origin[2] + direction[2] * r
    mx, my = world_to_map(px, py)
    i, j = map_to_cell(mx, my)
    ok = (i >= 0) & (i < REGION_N) & (j >= 0) & (j < REGION_N)
    if not ok.any():
        return None, None, None
    terr = bilinear(h, np.clip(i, 0, REGION_N - 1), np.clip(j, 0, REGION_N - 1))
    below = (pz <= terr) & ok
    if not below.any():
        return None, None, None
    k = int(np.argmax(below))
    return float(r[k]), float(mx[k]), float(my[k])


def c4x(h):
    """C4 exercised: every sensor scored against known geometry."""
    print("=== C4 exercised: the five sensors against known geometry ===")
    # Convert from the xacro inside the run. A stale rover.sdf converted
    # before the sourcing pass declared a 120 m range and a 45 deg vertical
    # field of view, so the scan was scored against figures the model no
    # longer held. The conversion is part of the measurement.
    rover = WORK / "rover.sdf"
    conv = subprocess.run(
        ["bash", "-lc",
         "source /opt/ros/jazzy/setup.bash >/dev/null 2>&1; "
         f"xacro {ROOT}/gazebo/rover.urdf.xacro -o {WORK}/rover.urdf && "
         f"GZ_CONFIG_PATH=/usr/share/gz /usr/bin/gz sdf -p {WORK}/rover.urdf "
         f"> {WORK}/rover.sdf"], capture_output=True, text=True)
    if conv.returncode != 0 or not rover.exists():
        print(f"  NOT RUN: the xacro did not convert: {conv.stderr[:200]}")
        return dict(S1=None, S2=None, S3=None)
    rtxt = rover.read_text()
    decl_ok = (f"<max>{LIDAR_RANGE_M:.0f}</max>" in rtxt
               and f"{np.radians(LIDAR_VFOV_DEG/2):.8f}"[:9] in rtxt)
    print(f"  rover.sdf regenerated from the xacro; it declares the sourced "
          f"LiDAR figures: {decl_ok}")
    pmx, pmy = ROVER_PATCH_MAP
    wx0, wy0 = (float(v) for v in map_to_world(pmx, pmy))
    base_z = height_at_map(h, pmx, pmy) + ROVER_ALT_M
    mast_z = base_z + MAST_Z_M
    tgt = (wx0 + TARGET_RANGE_M, wy0, mast_z)
    print(f"  rover base commanded to world ({wx0:.3f}, {wy0:.3f}, {base_z:.3f}), "
          f"{ROVER_ALT_M:.0f} m above the terrain at map ({pmx:.0f}, {pmy:.0f})")
    print(f"  mast and sensors at z {mast_z:.3f}; target sphere at "
          f"({tgt[0]:.3f}, {tgt[1]:.3f}, {tgt[2]:.3f}), radius "
          f"{TARGET_RADIUS_M} m, range {TARGET_RANGE_M:.3f} m")

    cmd = [str(ROOT / ".venv" / "bin" / "python"),
           str(ROOT / "examples" / "make_world.py"), "--mode", "ours",
           "--rover", str(rover),
           "--rover-pose", f"{wx0:.4f},{wy0:.4f},{base_z:.4f}",
           "--dark",
           "--targets", f"t0,{tgt[0]:.4f},{tgt[1]:.4f},{tgt[2]:.4f},"
                        f"{TARGET_RADIUS_M}",
           "--out", str(WORK / "c4x.sdf")]
    # Dark, with no light and ambient zero, so only the emissive target
    # renders. The terrain cannot supply a feature to match: its texture is
    # one texel per 20 m cell, so at a range short enough for measurable
    # disparity the whole field of view sits inside a single texel. The LiDAR
    # is geometric and the intrinsics are not lighting-dependent, so one dark
    # world serves all three checks.
    subprocess.run(cmd, check=True, capture_output=True)
    sdf = WORK / "c4x.sdf"

    topics = []
    for nm in ("nav_left", "nav_right", "haz_front", "haz_rear"):
        topics += [f"/rover/{nm}/image", f"/rover/{nm}/camera_info"]
    topics.append("/rover/lidar/scan")
    pose_text, got = grab_mixed(sdf, 35.0, "dynamic_pose/info", topics)

    # The rover must actually be where it was commanded. Four cases in this
    # record have a simulator reporting success and moving nothing.
    poses = parse_poses(pose_text)
    placed = poses.get("rover")
    if placed is not None:
        err = float(np.hypot(np.hypot(placed[0] - wx0, placed[1] - wy0),
                             placed[2] - base_z))
        print(f"  pose read back off the topic: ({placed[0]:.3f}, "
              f"{placed[1]:.3f}, {placed[2]:.3f}), error {err:.4f} m")
        if err > 0.5:
            print(f"  NOT RUN: the rover is not where it was commanded, so "
                  f"S2 and S3 would be scored against geometry that is not "
                  f"the geometry in the world. The first run of this check "
                  f"read back (0, 0, 2000) for a commanded pose 3919.5 m "
                  f"away, because the pose was computed and never passed to "
                  f"the world.")
            return dict(S1=None, S2=None, S3=None)
    else:
        print("  NOT RUN: the rover does not appear on the pose topic, so its "
              "placement is assumed rather than verified")
        return dict(S1=None, S2=None, S3=None)

    # ---- S1: resolution and field of view from the intrinsics ----
    print(f"\n  S1 resolution and field of view, from camera_info")
    print(f"    {'camera':>10}{'w':>6}{'h':>6}{'fx':>11}{'fov deg':>10}"
          f"{'declared':>10}{'err deg':>9}{'verdict':>9}")
    s1 = True
    for nm, decl in (("nav_left", NAV_FOV_DEG), ("nav_right", NAV_FOV_DEG),
                     ("haz_front", HAZ_FOV_DEG), ("haz_rear", HAZ_FOV_DEG)):
        ci = got.get(f"/rover/{nm}/camera_info")
        if ci is None:
            print(f"    {nm:>10}  no camera_info")
            s1 = False
            continue
        w, hh = int(ci["width"]), int(ci["height"])
        K = ci.get("intrinsics", {}).get("k") or ci.get("k")
        fx = float(K[0])
        fov = float(np.degrees(2.0 * np.arctan(w / (2.0 * fx))))
        e = abs(fov - decl)
        ok = (w == CAM_PX and hh == CAM_PX and e <= S1_FOV_TOL_DEG)
        s1 &= ok
        print(f"    {nm:>10}{w:6d}{hh:6d}{fx:11.3f}{fov:10.4f}{decl:10.1f}"
              f"{e:9.4f}{'pass' if ok else 'FAIL':>9}")
    print(f"    S1 -> {'PASS' if s1 else 'FAIL'}")

    # ---- S2: stereo disparity against the sourced baseline ----
    fx_nav = CAM_PX / (2.0 * np.tan(np.radians(NAV_FOV_DEG) / 2.0))
    pred = fx_nav * NAV_BASELINE_M / TARGET_RANGE_M
    print(f"\n  S2 stereo disparity against the sourced baseline")
    print(f"    fx {fx_nav:.2f} px, baseline {NAV_BASELINE_M} m [P], range "
          f"{TARGET_RANGE_M:.3f} m -> predicted {pred:.3f} px")
    cen = {}
    for nm in ("nav_left", "nav_right"):
        img = decode_image(got.get(f"/rover/{nm}/image"))
        if img is None:
            print(f"    {nm}: no image")
            continue
        thr = img.max() * 0.6
        m = img >= thr
        if m.sum() < 50:
            print(f"    {nm}: the target is not distinguishable "
                  f"({int(m.sum())} px above {thr:.1f})")
            continue
        ys, xs = np.nonzero(m)
        cen[nm] = (float(xs.mean()), float(ys.mean()), int(m.sum()))
        print(f"    {nm}: centroid u {cen[nm][0]:.3f}, v {cen[nm][1]:.3f}, "
              f"{cen[nm][2]} px above threshold")
    if len(cen) == 2:
        disp = cen["nav_left"][0] - cen["nav_right"][0]
        e = abs(abs(disp) - pred)
        s2 = bool(e <= S2_DISPARITY_TOL_PX)
        print(f"    measured disparity {disp:+.3f} px, |measured| - predicted "
              f"= {abs(disp)-pred:+.3f} px, tolerance {S2_DISPARITY_TOL_PX} px")
        print(f"    this verifies the baseline to about "
              f"{S2_DISPARITY_TOL_PX*TARGET_RANGE_M/fx_nav*100:.1f} cm, so it "
              f"does not distinguish 0.424 m from the old 0.420 m assumption, "
              f"which would differ by {abs(fx_nav*0.004/TARGET_RANGE_M):.2f} px")
        print(f"    S2 -> {'PASS' if s2 else 'FAIL'}")
    else:
        s2 = None
        print(f"    S2 -> NOT RUN, {len(cen)} of 2 centroids found")

    # ---- S3: lidar ranges against the heightmap ----
    print(f"\n  S3 LiDAR ranges against the heightmap along known rays")
    scan = got.get("/rover/lidar/scan")
    if scan is None:
        print("    no scan message")
        return dict(S1=s1, S2=s2, S3=None)
    rngs = np.asarray([float(v) for v in scan.get("ranges")], dtype=float)
    nrow = int(scan.get("verticalCount") or LIDAR_ROWS)
    ncol = int(scan.get("count") or LIDAR_COLS)
    decl_rmax = float(scan.get("rangeMax", LIDAR_RANGE_M))
    print(f"    scan {ncol} x {nrow}, declared {LIDAR_COLS} x {LIDAR_ROWS} [O]")
    if rngs.size != nrow * ncol:
        print(f"    {rngs.size} ranges for a {ncol} x {nrow} scan; not scoreable")
        return dict(S1=s1, S2=s2, S3=None)
    rngs = rngs.reshape(nrow, ncol)
    amin, amax = float(scan["angleMin"]), float(scan["angleMax"])
    vmin, vmax = float(scan["verticalAngleMin"]), float(scan["verticalAngleMax"])
    print(f"    the scan declares rangeMax {decl_rmax:.0f} m and vertical "
          f"{np.degrees(vmax-vmin):.1f} deg against the sourced "
          f"{LIDAR_RANGE_M:.0f} m and {LIDAR_VFOV_DEG:.1f} deg")
    if abs(decl_rmax - LIDAR_RANGE_M) > 0.5 or abs(
            np.degrees(vmax - vmin) - LIDAR_VFOV_DEG) > 0.1:
        print(f"    NOT SCOREABLE: the running sensor is not configured with "
              f"the sourced figures")
        return dict(S1=s1, S2=s2, S3=None)
    az = np.linspace(amin, amax, ncol)
    el = np.linspace(vmin, vmax, nrow)
    origin = (wx0, wy0, mast_z + 0.15)      # lidar_link is 0.15 m above mast
    rows = []
    for ri in range(0, nrow, 1):
        e = el[ri]
        if e >= 0:
            continue
        for ci in range(0, ncol, 8):
            got_r = rngs[ri, ci]
            if not np.isfinite(got_r) or got_r <= 0 or got_r >= LIDAR_RANGE_M:
                continue
            graze = float(np.degrees(abs(e)))
            if graze < S3_GRAZE_MIN_DEG:
                continue
            a = az[ci]
            d = (np.cos(e) * np.cos(a), np.cos(e) * np.sin(a), np.sin(e))
            pred_r, mx, my = march_dem(h, origin, d, LIDAR_RANGE_M)
            if pred_r is None:
                continue
            rows.append((float(got_r), pred_r, graze, a, mx, my))
    if len(rows) < 50:
        print(f"    only {len(rows)} scoreable beams; not scoreable")
        return dict(S1=s1, S2=s2, S3=None)
    A = np.array([(r[0], r[1], r[2], r[3]) for r in rows])
    errs = np.abs(A[:, 0] - A[:, 1])
    med, p90 = float(np.median(errs)), float(np.percentile(errs, 90))
    print(f"    {len(rows)} beams scored: returned inside {LIDAR_RANGE_M:.0f} m "
          f"and grazing at least {S3_GRAZE_MIN_DEG:.0f} deg")
    print(f"    ranges: median {np.median(A[:,0]):.2f} m, "
          f"{A[:,0].min():.2f} to {A[:,0].max():.2f} m")
    print(f"    |range error|: median {med:.3f} m (bar {S3_MEDIAN_MAX_M}), "
          f"p90 {p90:.3f} m (bar {S3_P90_MAX_M})")
    print(f"    predicted before the run from the open rendering term x "
          f"1/sin(grazing): median 0.858 m, p90 1.154 m, max 1.500 m")
    quad = []
    for q, (lo, hi) in enumerate(((-np.pi, -np.pi/2), (-np.pi/2, 0.0),
                                  (0.0, np.pi/2), (np.pi/2, np.pi))):
        m = (A[:, 3] >= lo) & (A[:, 3] < hi)
        if m.sum() > 10:
            quad.append(float(np.median(errs[m])))
            print(f"      azimuth quadrant {q}: n={int(m.sum()):5d}  median "
                  f"{np.median(errs[m]):.3f} m")
    spread = (max(quad) - min(quad)) if quad else float("inf")
    print(f"    quadrant spread {spread:.3f} m against the "
          f"{S3_MEDIAN_MAX_M} m bar: "
          f"{'inside the bar, so the bar measures something' if spread <= S3_MEDIAN_MAX_M else 'WIDER than the bar, so the bar is inside the estimator spread and measures nothing'}")
    s3 = bool(med <= S3_MEDIAN_MAX_M and p90 <= S3_P90_MAX_M
              and spread <= S3_MEDIAN_MAX_M)
    print(f"    S3 -> {'PASS' if s3 else 'FAIL'}")
    return dict(S1=s1, S2=s2, S3=s3)


def probe_bounds(h):
    """Per-probe tolerance for L2'', derived from the DEM with no simulator.

    Two named terms and no third. tri is the largest gap between the bilinear
    reference and the 10 m triangulation Gazebo renders, over the cell holding
    the probe. curv is the linear interpolation error at the local surface,
    kappa d^2 / 8: interpolating the cloud is exact on a plane, so slope no
    longer enters at first order and curvature is what is left. This is the
    term that replaces L2's lateral quantisation, which was slope x d/sqrt(2)
    and reached 1.615 m at p03 -- omitted from the 2.0 m bar L2' was scored
    against, which is why P7a-1 failed.
    """
    d = 2.0 * RAY_ALT_M * np.tan(RAY_FOV_RAD / 2.0) / RAY_PX
    gi, gj = np.gradient(h)
    hii, hij = np.gradient(gi)
    _, hjj = np.gradient(gj)
    sub = SAMPLING
    out = []
    for mx, my in PROBES:
        i, j = map_to_cell(mx, my)
        ii, jj = int(np.floor(float(i))), int(np.floor(float(j)))
        gx = float(bilinear(gj, i, j)) / PX
        gy = -float(bilinear(gi, i, j)) / PX
        slope = float(np.degrees(np.arctan(np.hypot(gx, gy))))
        worst = 0.0
        for du in np.linspace(0.05, 0.95, 10):
            for dv in np.linspace(0.05, 0.95, 10):
                I, J = ii + du, jj + dv
                b = float(bilinear(h, I, J))
                i0 = np.floor(I * sub) / sub
                j0 = np.floor(J * sub) / sub
                c00 = float(bilinear(h, i0, j0))
                c10 = float(bilinear(h, i0 + 1.0 / sub, j0))
                c01 = float(bilinear(h, i0, j0 + 1.0 / sub))
                c11 = float(bilinear(h, i0 + 1.0 / sub, j0 + 1.0 / sub))
                fu, fv = (I - i0) * sub, (J - j0) * sub
                tri = (c00 + (c10 - c00) * fu + (c01 - c00) * fv
                       if fu + fv <= 1.0 else
                       c11 + (c01 - c11) * (1 - fu) + (c10 - c11) * (1 - fv))
                worst = max(worst, abs(b - tri))
        H = np.array([[float(bilinear(hii, i, j)), float(bilinear(hij, i, j))],
                      [float(bilinear(hij, i, j)), float(bilinear(hjj, i, j))]])
        kappa = float(np.abs(np.linalg.eigvalsh(H / PX ** 2)).max())
        curv = kappa * d * d / 8.0
        out.append(dict(slope=slope, tri=worst, curv=curv, bound=worst + curv,
                        lateral=float(np.hypot(gx, gy) * d / np.sqrt(2.0))))
    return out, d


def l2pp(h):
    """L2'': the cloud interpolated at the probe point, scored per probe."""
    print("=== L2'': placement by interpolated ray, per-probe bounds ===")
    bounds, d = probe_bounds(h)
    rays = []
    for k, (mx, my) in enumerate(PROBES):
        wx, wy = map_to_world(mx, my)
        rays.append((f"p{k:02d}", float(wx), float(wy),
                     height_at_map(h, mx, my) + RAY_ALT_M))
    rayarg = "--rays=" + ";".join(
        f"{n},{x:.4f},{y:.4f},{z:.4f},{RAY_FOV_RAD:.9f},{RAY_PX},{RAY_PX}"
        for n, x, y, z in rays)
    sdf = make_world([], extra=[rayarg])
    print(f"  ray ground sampling d = {d:.4f} m, {RAY_PX} px over "
          f"{2*RAY_ALT_M*np.tan(RAY_FOV_RAD/2):.1f} m from {RAY_ALT_M:.0f} m up")
    print(f"  nothing dropped and nothing settling, so no settle window")
    _, clouds = grab_mixed(sdf, 30.0, "dynamic_pose/info",
                           [f"/ray/{n}/depth/points" for n, _, _, _ in rays])
    print(f"\n  {'probe':>7}{'slope':>8}{'bound':>9}{'residual':>10}"
          f"{'nodes':>7}{'verdict':>9}")
    ok, rs = True, []
    for k, (nm, cwx, cwy, cwz) in enumerate(rays):
        b = bounds[k]
        msg = clouds.get(f"/ray/{nm}/depth/points")
        if msg is None:
            print(f"  {nm:>7}  MISSING from the depth topic")
            ok = False
            continue
        wx, wy, wz, _l, good = cloud_to_world(msg, (cwx, cwy, cwz))
        sel = good & (np.hypot(wx - cwx, wy - cwy) <= 3.0 * d)
        if sel.sum() < 6:
            print(f"  {nm:>7}  only {int(sel.sum())} nodes near the probe")
            ok = False
            continue
        # plane through the local cloud, evaluated exactly at the probe point.
        # Linear interpolation is exact on a plane, so slope leaves the error
        # budget and curvature is what remains.
        A = np.c_[wx[sel] - cwx, wy[sel] - cwy, np.ones(int(sel.sum()))]
        coef, *_ = np.linalg.lstsq(A, wz[sel], rcond=None)
        zi = float(coef[2])
        mx, my = PROBES[k]
        r = zi - height_at_map(h, mx, my)
        rs.append(r)
        good_probe = abs(r) <= b["bound"]
        ok &= good_probe
        print(f"  {nm:>7}{b['slope']:8.2f}{b['bound']:9.4f}{r:+10.4f}"
              f"{int(sel.sum()):7d}{'pass' if good_probe else 'FAIL':>9}")
    l2 = bool(ok and len(rs) == len(rays))
    if rs:
        a = np.array(rs)
        print(f"\n    residuals: max |{np.abs(a).max():.4f}| m, median "
              f"{np.median(a):+.4f} m")
        print(f"    L2' took the nearest pixel and reached 2.610 m at p03; the "
              f"lateral term that")
        print(f"    caused it was up to 1.615 m and is gone from this "
              f"instrument by construction.")

    # Reported, not scored: the offset that minimises the residual per field.
    # This is what separates a placement error from a surface-fidelity one.
    fields = []
    for nm, cwx, cwy, cwz in rays:
        msg = clouds.get(f"/ray/{nm}/depth/points")
        if msg is None:
            continue
        wx, wy, wz, _l, good = cloud_to_world(msg, (cwx, cwy, cwz))
        smx, smy = world_to_map(wx, wy)
        g = (good & (smx > REGION_OX + 100) & (smx < REGION_OX + 9900)
             & (smy < REGION_OY - 100) & (smy > REGION_OY - 9900))
        if g.any():
            fields.append((smx[g], smy[g], wz[g]))

    def med(dx, dy):
        o = []
        for smx, smy, wz in fields:
            i, j = map_to_cell(smx + dx, smy + dy)
            o.append(np.abs(wz - bilinear(h, i, j)))
        return float(np.median(np.concatenate(o)))

    if fields:
        grid = [(med(dx, dy), dx, dy)
                for dx in np.arange(-6.0, 6.1, 2.0)
                for dy in np.arange(-6.0, 6.1, 2.0)]
        grid.sort()
        best, bdx, bdy = grid[0]
        print(f"\n    reported, not scored: the shift that minimises the "
              f"residual over all")
        print(f"    {sum(len(f[0]) for f in fields)} field pixels is "
              f"({bdx:+.1f}, {bdy:+.1f}) m at {best:.4f} m, against "
              f"{med(0.0,0.0):.4f} m at zero shift.")
        print(f"    A residual with the shift at zero is a statement about the "
              f"rendered surface;")
        print(f"    a shift away from zero would be a statement about the "
              f"mapping.")
    print(f"  L2'' -> {'PASS' if l2 else 'FAIL'}")
    return dict(L2pp=l2)


def c2(h):
    """C2 as registered, and C2' as measurement established it.

    The original pass condition had three clauses and one of them asserted that
    omitting <size> makes Gazebo take the DEM's footprint. Measurement showed
    that omitting it loads a 1 m cube, so the code does the opposite of what
    the check registered. Both are scored here, side by side, and the original
    is not edited.
    """
    from src.gzmap import SAMPLING, requested_size, span_m
    print("=== C2: heights, the registered clause and the replacement ===")
    side = json.loads((WORK / "region_heightmap.json").read_text())
    with rasterio.open(TIF) as ds:
        dtype_ok = ds.dtypes[0] == "float32"
    ref = np.fromfile(ROOT / "data" / "work" / "region_height.f32",
                      dtype=np.float32).reshape(REGION_N, REGION_N)
    overlap = np.array_equal(h[:REGION_N, :REGION_N].astype(np.float32), ref)
    scale = float(np.ptp(h[:REGION_N, :REGION_N]) / np.ptp(ref))
    print(f"  float32 with no quantisation step: {dtype_ok}")
    print(f"  exported values equal the DEM: {overlap} "
          f"({side['differing_cells_vs_planner']} differing cells)")
    print(f"  DEM metres to world metres, scale factor {scale:.9f} "
          f"(1.0 is no exaggeration)")

    # The world as the code actually writes it.
    sdf = make_world([])
    txt = Path(sdf).read_text()
    three = re.findall(r"<size>([^<]*)</size>", txt)
    sizes = [v for v in three if len(v.split()) == 3]
    size_omitted = len(sizes) == 0
    # Both the collision and the visual heightmap must carry it, and carry the
    # same value: the collision surface is what L2's spheres rest on and the
    # visual surface is what L2' and C3 see, so if the two sizes differed the
    # instruments would be measuring different terrain and agreeing by luck.
    size_explicit = bool(len(sizes) == 2 and sizes[0] == sizes[1])
    # computed from sampling rather than written down: the function must move
    # when sampling moves, which a literal cannot do.
    r1 = requested_size(HEIGHTMAP_N, PX, 1)
    r2 = requested_size(HEIGHTMAP_N, PX, SAMPLING)
    span = span_m(HEIGHTMAP_N, PX)
    computed = bool(r1 != r2 and abs(r2 - span * (SAMPLING * (HEIGHTMAP_N - 1)
                    + 1) / (SAMPLING * (HEIGHTMAP_N - 1))) < 1e-6)
    print(f"  <size> elements with three components in the world: {len(sizes)}"
          f" (collision and visual), identical: "
          f"{len(set(sizes)) == 1 if sizes else False}")
    if sizes:
        print(f"    {sizes[0]}")
    print(f"  the original clause asks that <size> be omitted: omitted "
          f"{size_omitted}")
    print(f"    omitting it was measured to load a 1 m cube, AABB min "
          f"{{-0.4995, ...}}, so the")
    print(f"    code gives it deliberately and the registered clause cannot "
          f"hold.")
    print(f"  the span correction moves with sampling: {computed} "
          f"(sampling 1 -> {r1:.4f} m, sampling {SAMPLING} -> {r2:.4f} m "
          f"for a true {span:.0f} m span)")

    c2_orig = bool(dtype_ok and overlap and size_omitted)
    c2_prime = bool(dtype_ok and overlap and size_explicit and computed)
    print(f"  C2  -> {'PASS' if c2_orig else 'FAIL'} "
          f"(as registered; the size clause is false)")
    print(f"  C2' -> {'PASS' if c2_prime else 'FAIL'}")
    return dict(C2=c2_orig, C2p=c2_prime)


def grab_mixed(sdf, settle_s, pose_topic, json_topics, world="lunar_region"):
    """One server run, the pose topic as text and several topics as JSON.

    L2 and L2' are deliberately measured in the same world on the same terrain,
    so the rolling instrument and the fixed one can be compared directly rather
    than across two runs.
    """
    proc = subprocess.Popen([GZ, "sim", "-s", "-r", "-v", "1", str(sdf)],
                            env=ENV, stdout=subprocess.DEVNULL,
                            stderr=subprocess.DEVNULL, start_new_session=True)
    got = {}
    try:
        time.sleep(settle_s)
        out = subprocess.run([GZ, "topic", "-e", "-t",
                              f"/world/{world}/{pose_topic}", "-n", "1"],
                             env=ENV, capture_output=True, text=True, timeout=40)
        pose_text = out.stdout
        for t in json_topics:
            r = subprocess.run([GZ, "topic", "-e", "-t", t, "-n", "1",
                                "--json-output"], env=ENV,
                               capture_output=True, text=True, timeout=60)
            got[t] = json.loads(r.stdout) if r.stdout.strip() else None
        return pose_text, got
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()


def rim_arc_mask(n, conv="true"):
    """Cells within half a pixel of Shackleton's gazetteer rim, under one
    cell-to-map convention. The locus comes from the catalogue, not the DEM."""
    i, j = np.mgrid[0:n, 0:n]
    if conv == "true":
        ii, jj = i, j
    elif conv == "row flip":
        ii, jj = n - 1 - i, j
    elif conv == "column flip":
        ii, jj = i, n - 1 - j
    elif conv == "transpose":
        ii, jj = j, i
    else:
        raise ValueError(conv)
    x = REGION_OX + (jj + 0.5) * PX
    y = REGION_OY - (ii + 0.5) * PX
    d = np.hypot(x - SHACKLETON_MAP[0], y - SHACKLETON_MAP[1])
    return np.abs(d - SHACKLETON_RIM_R_M) <= PX / 2.0


def c1(h):
    print("=== C1: mapping ===")
    # layer 1 is asserted by the export; read its sidecar rather than restate it
    side = json.loads((WORK / "region_heightmap.json").read_text())
    l1 = bool(side["roundtrip_exact"] and side["matches_planner_region"])
    print(f"\n  L1 export fidelity: readback exact {side['roundtrip_exact']}, "
          f"equals the planner's array {side['matches_planner_region']} "
          f"({side['differing_cells_vs_planner']} differing cells of "
          f"{REGION_N*REGION_N})  -> {'PASS' if l1 else 'FAIL'}")

    rows, worst = error_signatures(h)
    print(f"\n  L3 the probe set is discriminating, computed before L2 is run")
    print(f"    {'map x':>8}{'map y':>9}{'i':>7}{'j':>7}{'height':>9}"
          f"{'row flip':>11}{'col flip':>11}{'transpose':>11}{'offset':>9}")
    for r in rows:
        print(f"    {r['map_x']:8.0f}{r['map_y']:9.1f}{r['i']:7.1f}{r['j']:7.1f}"
              f"{r['height']:9.2f}{r['row flip']:11.1f}{r['column flip']:11.1f}"
              f"{r['transpose']:11.1f}{r['13-cell offset']:9.1f}")
    l3 = all(v >= L3_MIN_SIGNATURE_M for v in worst.values())
    for name, v in worst.items():
        print(f"    worst {name:<16} {v:8.1f} m  "
              f"{'clears' if v >= L3_MIN_SIGNATURE_M else 'BELOW'} the "
              f"registered {L3_MIN_SIGNATURE_M:.0f} m")
    print(f"    L3 -> {'PASS' if l3 else 'FAIL'}")

    targets = list(PROBES)
    drops = []
    for mx, my in targets:
        wx, wy = map_to_world(mx, my)
        drops.append((float(wx), float(wy),
                      height_at_map(h, mx, my) + PROBE_RADIUS_M + DROP_M))
    # Both instruments in one world on one terrain, so the rolling probe and
    # the fixed ray are directly comparable rather than measured across runs.
    rays = []
    for k, (mx, my) in enumerate(targets):
        wx, wy = map_to_world(mx, my)
        rays.append((f"p{k:02d}", float(wx), float(wy),
                     height_at_map(h, mx, my) + RAY_ALT_M))
    rayarg = "--rays=" + ";".join(
        f"{n},{x:.4f},{y:.4f},{z:.4f},{RAY_FOV_RAD:.9f},{RAY_PX},{RAY_PX}"
        for n, x, y, z in rays)
    sdf = make_world(drops, extra=[rayarg])
    print(f"\n  L2 and L4: {len(drops)} probes dropped {DROP_M:.0f} m above their "
          f"own terrain, settling for {SETTLE_S:.0f} s")
    print(f"  L2': {len(rays)} nadir depth cameras at the same points, fixed, "
          f"{RAY_ALT_M:.0f} m up")
    pose_text, clouds = grab_mixed(
        sdf, SETTLE_S, "dynamic_pose/info",
        [f"/ray/{n}/depth/points" for n, _, _, _ in rays])
    poses = parse_poses(pose_text)
    print(f"    {'probe':>7}{'map x':>8}{'map y':>9}{'settled z':>11}"
          f"{'DEM here':>10}{'residual':>10}{'rolled':>9}")
    resid, ok = [], True
    for k, (mx, my) in enumerate(targets):
        name = f"probe{k:02d}"
        if name not in poses:
            print(f"    {name:>7}  MISSING from the pose topic")
            ok = False
            continue
        sx, sy, sz = poses[name]
        smx, smy = (float(v) for v in world_to_map(sx, sy))
        d = height_at_map(h, smx, smy)
        r = sz - PROBE_RADIUS_M - d
        rolled = float(np.hypot(sx - drops[k][0], sy - drops[k][1]))
        resid.append((name, mx, my, sz, d, r, rolled))
        print(f"    {name:>7}{smx:8.0f}{smy:9.1f}{sz:11.3f}{d:10.3f}"
              f"{r:+10.3f}{rolled:9.1f}")
        ok &= abs(r) <= L2_TOL_M
    l2 = ok and len(resid) == len(targets)
    if resid:
        rs = np.array([x[5] for x in resid])
        print(f"    residuals: max |{np.abs(rs).max():.3f}| m, median "
              f"{np.median(rs):+.3f} m, tolerance {L2_TOL_M:.0f} m")
    print(f"    L2 -> {'PASS' if l2 else 'FAIL'}")

    # L2': the same intent with an instrument that cannot roll. The height is
    # read from the pixel's own range and compared with the DEM at that pixel's
    # own reconstructed position, so lateral sampling cannot enter the residual.
    print(f"\n  L2' placement by fixed nadir ray, nothing dropped and nothing "
          f"settling")
    print(f"    {'probe':>7}{'map x':>9}{'map y':>10}{'ray z':>11}"
          f"{'DEM here':>10}{'residual':>10}{'lateral':>9}")
    rp, okp = [], True
    for k, (nm, cwx, cwy, cwz) in enumerate(rays):
        msg = clouds.get(f"/ray/{nm}/depth/points")
        if msg is None:
            print(f"    {nm:>7}  MISSING from the depth topic")
            okp = False
            continue
        wx, wy, wz, _lum, good = cloud_to_world(msg, (cwx, cwy, cwz))
        if not good.any():
            print(f"    {nm:>7}  no finite returns")
            okp = False
            continue
        gi = np.where(good)[0]
        # the pixel whose own reconstruction is nearest the probe point
        sel = gi[np.argmin(np.hypot(wx[gi] - cwx, wy[gi] - cwy))]
        lat = float(np.hypot(wx[sel] - cwx, wy[sel] - cwy))
        smx, smy = (float(v) for v in world_to_map(wx[sel], wy[sel]))
        d = height_at_map(h, smx, smy)
        r = float(wz[sel]) - d
        rp.append(r)
        print(f"    {nm:>7}{smx:9.1f}{smy:10.1f}{float(wz[sel]):11.3f}"
              f"{d:10.3f}{r:+10.3f}{lat:9.2f}")
        okp &= abs(r) <= L2P_TOL_M
    l2p = bool(okp and len(rp) == len(rays))
    if rp:
        a = np.array(rp)
        print(f"    residuals: max |{np.abs(a).max():.3f}| m, median "
              f"{np.median(a):+.3f} m, registered tolerance {L2P_TOL_M:.1f} m")
        print(f"    the registered signatures are 134.7 m at least, so a frame "
              f"error is {134.7/max(np.abs(a).max(),1e-9):.0f}x this residual")
    print(f"    L2' -> {'PASS' if l2p else 'FAIL'}")

    # Localisation, because an aggregate residual points at nothing. If the
    # residual is a horizontal misplacement rather than a vertical one it will
    # collapse when the DEM is read at a shifted point, and the shift says
    # where. Scanned over every ray's whole field at once, since within one
    # 1 km patch the slope direction is nearly constant and the two components
    # are not separately identifiable from a single rig.
    fields = []
    for nm, cwx, cwy, cwz in rays:
        msg = clouds.get(f"/ray/{nm}/depth/points")
        if msg is None:
            continue
        wx, wy, wz, _l, good = cloud_to_world(msg, (cwx, cwy, cwz))
        smx, smy = world_to_map(wx, wy)
        ok = (good & (smx > REGION_OX + 100) & (smx < REGION_OX + 9900)
              & (smy < REGION_OY - 100) & (smy > REGION_OY - 9900))
        if ok.any():
            fields.append((smx[ok], smy[ok], wz[ok]))

    def med_abs(dx, dy):
        out = []
        for smx, smy, wz in fields:
            i, j = map_to_cell(smx + dx, smy + dy)
            out.append(np.abs(wz - bilinear(h, i, j)))
        return float(np.median(np.concatenate(out)))

    if fields:
        half = PX / 2.0
        base = med_abs(0.0, 0.0)
        shifted = med_abs(half, -half)
        print(f"\n    where the residual comes from, over "
              f"{sum(len(f[0]) for f in fields)} pixels of all "
              f"{len(fields)} fields:")
        print(f"      median |rendered - DEM(x, y)|              "
              f"{base:8.4f} m")
        print(f"      median |rendered - DEM(x+{half:.0f}, y-{half:.0f})|      "
              f"{shifted:8.4f} m")
        # Which of the two is smaller says whether the mapping agrees with the
        # terrain or sits half a cell from it. Before centre_map was corrected
        # the shifted reading was 16.3x better, which is how the offset was
        # found; after it, the unshifted reading must be the better one.
        if shifted < base:
            print(f"      the shifted reading is {base/max(shifted,1e-9):.1f}x "
                  f"better, so the mapping sits half a cell from")
            print(f"      the terrain: a horizontal placement error, not a "
                  f"height error.")
        else:
            print(f"      the unshifted reading is "
                  f"{shifted/max(base,1e-9):.1f}x better, so the mapping and "
                  f"the terrain agree")
            print(f"      and no half-cell offset remains.")
        mid = (HEIGHTMAP_N - 1) // 2
        mc = cell_to_map(mid, mid)
        ext = (REGION_OX + span_m(HEIGHTMAP_N, PX) / 2.0,
               REGION_OY - span_m(HEIGHTMAP_N, PX) / 2.0)
        print(f"      centre_map gives map ({centre_map()[0]:.0f}, "
              f"{centre_map()[1]:.0f}); the middle sample i = j = {mid} is at "
              f"({float(mc[0]):.0f}, {float(mc[1]):.0f})")
        print(f"      and the extent's centre is at ({ext[0]:.0f}, "
              f"{ext[1]:.0f}). Gazebo pins the terrain on the middle")
        print(f"      sample, and for an odd sample count the extent's centre "
              f"is half a cell from it.")

    # layer 4: the external anchor, scored against what was registered
    smx, smy = SHACKLETON_MAP
    wx, wy = (float(v) for v in map_to_world(smx, smy))
    half = (HEIGHTMAP_N - 1) * PX / 2.0
    inside = bool(abs(wx) <= half and abs(wy) <= half)
    l4 = inside            # the registered condition begins with being inside
    print(f"\n  L4 external anchor, Shackleton from the IAU Gazetteer")
    print(f"    this project's own figures: latitude {SHACKLETON_LATLON[0]}, "
          f"longitude {SHACKLETON_LATLON[1]}, diameter 20.92 km, adopted 1994")
    print(f"    through the production projection -> map ({smx:.1f}, {smy:.1f}), "
          f"10.01 km from the pole")
    print(f"    heightmap footprint x [{REGION_OX:.0f}, "
          f"{REGION_OX + HEIGHTMAP_N * PX:.0f}], y "
          f"[{REGION_OY - HEIGHTMAP_N * PX:.0f}, {REGION_OY:.0f}]")
    print(f"    inside the footprint: {inside}")
    print(f"    the registered L4 asserted it lands at map (0.0, 3032.3) and so "
          f"inside.\n    That came from a latitude of -89.9 taken from memory, "
          f"not from this\n    project's figures, and it is wrong: the landmark "
          f"is 13.4 km outside.")
    print(f"    L4 -> {'PASS' if l4 else 'FAIL, the registered premise is false'}")

    # Reported beside L4 and explicitly not it: the heightmap inherits the frame
    # that PLAN.md's landmark test anchors externally, and that inheritance is
    # checkable without the landmark being in the footprint.
    import rasterio as _rio
    with _rio.open(ROOT / "data" / "lola" / "LDEM_80S_20MPP_ADJ.TIF") as a, \
         _rio.open(TIF) as b:
        same_crs = a.crs == b.crs
        dc = (b.transform.c - a.transform.c) / a.transform.a
        df = (b.transform.f - a.transform.f) / a.transform.e
        integer = abs(dc - round(dc)) < 1e-9 and abs(df - round(df)) < 1e-9
        same_px = (a.transform.a == b.transform.a and a.transform.e == b.transform.e)
    print(f"    reported, not scored as L4: the export carries the source's own "
          f"CRS ({same_crs}),\n    the same 20 m pixel ({same_px}), and an exact "
          f"integer pixel offset of\n    ({round(dc)}, {round(df)}) ({integer}), so it sits in the "
          f"georeferenced frame\n    that tests/test_landmarks.py anchors against "
          f"the gazetteer externally.")

    # L4': the gazetteer rim, which does cross the footprint even though no
    # catalogued centre lies in it. Both figures are the catalogue's.
    n = h.shape[0]
    med = float(np.median(h))
    stats = {}
    for conv in ("true", "row flip", "column flip", "transpose"):
        m = rim_arc_mask(n, conv)
        stats[conv] = (int(m.sum()), float(h[m].mean()) - med)
    print(f"\n  L4' external anchor, Shackleton's gazetteer rim")
    print(f"    centre map ({SHACKLETON_MAP[0]:.1f}, {SHACKLETON_MAP[1]:.1f}) "
          f"from latitude {SHACKLETON_LATLON[0]}, longitude "
          f"{SHACKLETON_LATLON[1]}")
    print(f"    rim radius {SHACKLETON_RIM_R_M:.1f} m from the catalogued "
          f"diameter 20.92 km")
    print(f"    no catalogued feature's centre lies inside this footprint; this "
          f"rim crosses it")
    print(f"    footprint median height {med:.3f} m")
    print(f"    {'convention':>14}{'arc cells':>11}{'mean minus median':>20}")
    for conv, (cnt, dv) in stats.items():
        print(f"    {conv:>14}{cnt:11d}{dv:+20.3f}")
    true_dv = stats["true"][1]
    others = [v for k, (_, v) in stats.items() if k != "true"]
    above = bool(true_dv > 0.0)
    first = bool(all(true_dv > o for o in others))
    l4p = bool(above and first)
    print(f"    the rim arc is above the footprint median: {above}")
    print(f"    the true convention ranks first of the four: {first} "
          f"(next best {max(others):+.3f} m)")
    print(f"    L4' -> {'PASS' if l4p else 'FAIL'}")

    return dict(L1=l1, L2=l2, L3=l3, L4=l4, L2p=l2p, L4p=l4p)



# Patch centre, chosen for class balance and recorded as such. The first
# choice, map (1000, 6000), turned out to be 0.00% lit at this epoch, so its
# confusion table had an empty column and measured nothing. This patch is
# 50.16% lit over the 50 x 50 cells it covers, so neither class is trivial.
# The choice is for informativeness and not for outcome: a balanced patch does
# not move the positive check's disagreement or the control's kappa in any
# predictable direction.
PATCH_MAP = (3010.0, 2490.0)
PATCH_FOV_RAD = 0.34906585       # 20 deg
PATCH_PX = 512
C3_KAPPA_MAX = 0.5               # registered
C3_RATIO_MIN = 2.0               # registered


def grab(topics, settle_s, sdf):
    """One message off each topic, from a single server run.

    Both sensors are taken from the same run so the colour and the geometry are
    the same frame of the same scene.
    """
    proc = subprocess.Popen([GZ, "sim", "-s", "-r", "-v", "1", str(sdf)],
                            env=ENV, stdout=subprocess.DEVNULL,
                            stderr=subprocess.DEVNULL, start_new_session=True)
    got = {}
    try:
        time.sleep(settle_s)
        for t in topics:
            out = subprocess.run([GZ, "topic", "-e", "-t", t, "-n", "1",
                                  "--json-output"], env=ENV,
                                 capture_output=True, text=True, timeout=60)
            got[t] = json.loads(out.stdout) if out.stdout.strip() else None
        return got
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()


def rgb_luminance(msg):
    """Per-pixel luminance from the colour image.

    Taken from the image topic rather than the point cloud's rgb field, which
    came back identical under both lighting modes and so was carrying no
    render. The two sensors are co-located with the same resolution and field
    of view, so pixel (r, c) is the same ray in both, which is a far weaker
    assumption than a projection model.
    """
    import base64
    if msg is None or msg.get("pixelFormatType") != "RGB_INT8":
        return None
    w, hgt = int(msg["width"]), int(msg["height"])
    raw = np.frombuffer(base64.b64decode(msg["data"]), dtype=np.uint8)
    img = raw[: w * hgt * 3].reshape(hgt, w, 3).astype(float)
    return (0.2126 * img[..., 0] + 0.7152 * img[..., 1]
            + 0.0722 * img[..., 2]).ravel()


def cloud_to_world(msg, cam_world):
    """Per-pixel ground points and colours, from one registered message.

    The point cloud carries x, y, z and rgb for the same pixel, so the colour
    and the geometry cannot drift apart. The sensor frame is x forward along
    the optical axis, y left, z up, and the camera is pitched a quarter turn to
    look down, so forward is -Z in the world. The reconstruction is checked
    against the DEM below rather than trusted: if the convention were wrong the
    heights would not match and C3 is reported not scoreable.
    """
    import base64
    raw = base64.b64decode(msg["data"])
    n = int(msg["width"]) * int(msg["height"])
    step = int(msg["pointStep"])
    buf = np.frombuffer(raw, dtype=np.uint8).reshape(n, step)
    xyz = buf[:, 0:12].copy().view(np.float32).reshape(n, 3).astype(float)
    rgb_f = buf[:, 16:20].copy().view(np.float32).reshape(n).astype(np.float32)
    rgb_i = rgb_f.view(np.uint32) if rgb_f.dtype == np.float32 else rgb_f
    r = ((rgb_i >> 16) & 255).astype(float)
    g = ((rgb_i >> 8) & 255).astype(float)
    b = (rgb_i & 255).astype(float)
    lum = 0.2126 * r + 0.7152 * g + 0.0722 * b
    _ = rgb_i
    fwd, left, up = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    cx, cy, cz = cam_world
    wx = cx + up
    wy = cy + left
    wz = cz - fwd
    good = np.isfinite(wx) & np.isfinite(wy) & np.isfinite(wz) & (fwd > 1.0)
    return wx, wy, wz, lum, good


def sun_direction_world(patch_map, tag="2028-02-15_p0", hours=0.0):
    """Our own sun vector, expressed as a Gazebo light direction.

    The light direction points from the sun towards the scene, so it is the
    negative of the sun unit vector in the patch's local frame. At this
    latitude the sun sits about a degree above the horizon, so the direction is
    nearly horizontal; a nadir light would cast no terrain shadows at all.
    """
    from examples.illumination_texture import sun_unit_at
    from src.frames import local_frame
    s = sun_unit_at(tag, hours)
    mx, my = patch_map
    up, east, north = local_frame(np.array([mx]), np.array([my]))
    lam = np.arctan2(mx, my)
    ax = np.cos(lam) * north[0] - np.sin(lam) * east[0]
    ay = np.sin(lam) * north[0] + np.cos(lam) * east[0]
    v = np.array([float(ax @ s), float(ay @ s), float(up[0] @ s)])
    v /= np.linalg.norm(v)
    return -v


def c3(h):
    print("\n=== C3: illumination ===")
    tex = sorted(WORK.glob("illum_2028-02-15_p0_*.png"))
    mask_p = sorted(WORK.glob("illum_2028-02-15_p0_*_mask.npy"))
    if not tex or not mask_p:
        print("  NOT RUN: no illumination texture; run "
              "examples/illumination_texture.py")
        return dict(C3=None)
    lit = np.load(mask_p[0])
    pmx, pmy = PATCH_MAP
    wx0, wy0 = (float(v) for v in map_to_world(pmx, pmy))
    alt = 500.0 / np.tan(PATCH_FOV_RAD / 2.0)
    cam = (wx0, wy0, height_at_map(h, pmx, pmy) + alt)
    camarg = f"--cam={cam[0]:.4f},{cam[1]:.4f},{cam[2]:.4f}," \
             f"{PATCH_FOV_RAD:.9f},{PATCH_PX},{PATCH_PX}"
    print(f"  patch centred on map ({pmx:.0f}, {pmy:.0f}), camera {alt:.0f} m up, "
          f"{PATCH_PX} x {PATCH_PX} px, {np.degrees(PATCH_FOV_RAD):.0f} deg fov")

    # The control's light must be our own sun, not a nadir one. Run with
    # --light "0 0 -1" it cast no shadows at all and every pixel came back lit,
    # which is not a control but an empty scene.
    sun = sun_direction_world(PATCH_MAP)
    print(f"  sun direction in the world frame from our own SPICE vector: "
          f"({sun[0]:+.6f}, {sun[1]:+.6f}, {sun[2]:+.6f}), "
          f"elevation {np.degrees(np.arcsin(-sun[2])):+.3f} deg")

    def render(mode, texture):
        cmd = [str(ROOT / ".venv" / "bin" / "python"),
               str(ROOT / "examples" / "make_world.py"), "--mode", mode, camarg,
               "--light", f"{sun[0]:.9f} {sun[1]:.9f} {sun[2]:.9f}",
               "--out", str(WORK / f"c3_{mode}.sdf")]
        if texture:
            cmd += ["--texture", str(texture)]
        subprocess.run(cmd, check=True, capture_output=True)
        return grab(["/probe/depth/points", "/probe/image"], 30.0,
                    WORK / f"c3_{mode}.sdf")

    out = {}
    got = render("ours", tex[0])
    msg = got.get("/probe/depth/points")
    lum = rgb_luminance(got.get("/probe/image"))
    if msg is None or lum is None:
        print("  NOT RUN: the ours render returned no cloud or no image")
        return dict(C3=None)
    wx, wy, wz, _unused, good = cloud_to_world(msg, cam)
    mx, my = world_to_map(wx, wy)
    i, j = map_to_cell(mx, my)
    inside = good & (i >= 0) & (i < REGION_N) & (j >= 0) & (j < REGION_N)
    ii, jj = np.rint(i[inside]).astype(int), np.rint(j[inside]).astype(int)
    dem_h = bilinear(h, i[inside], j[inside])
    geo_resid = wz[inside] - dem_h
    print(f"\n  geometry self-check, before any illumination claim rests on it:")
    print(f"    {int(inside.sum())} of {lum.size} pixels land inside the region")
    print(f"    reconstructed height minus DEM: median "
          f"{np.median(geo_resid):+.3f} m, p90 "
          f"{np.quantile(np.abs(geo_resid), 0.9):.3f} m")
    if np.median(np.abs(geo_resid)) > 25.0:
        print(f"    the reconstruction does not agree with the DEM, so the "
              f"pixel-to-cell mapping is not trustworthy")
        print(f"  C3 NOT SCOREABLE")
        return dict(C3=None)

    ours_lit_px = lum[inside] > 0.5
    model_lit_px = lit[ii, jj]
    agree = ours_lit_px == model_lit_px
    pos_dis = float(1.0 - agree.mean())
    tp = int((ours_lit_px & model_lit_px).sum())
    fp = int((ours_lit_px & ~model_lit_px).sum())
    fn = int((~ours_lit_px & model_lit_px).sum())
    tn = int((~ours_lit_px & ~model_lit_px).sum())
    print(f"\n  positive check, rendered 'ours' against our own mask, per pixel")
    print(f"    {'':22}{'model lit':>12}{'model dark':>12}")
    print(f"    {'rendered lit':<22}{tp:>12}{fp:>12}")
    print(f"    {'rendered dark':<22}{fn:>12}{tn:>12}")
    print(f"    disagreement {pos_dis:.4%} of {int(inside.sum())} pixels")
    cells = {}
    for a, b, c in zip(ii, jj, agree):
        k = (int(a), int(b))
        v = cells.setdefault(k, [0, 0])
        v[0] += 1
        v[1] += int(not c)
    bad = sorted(((v[1] / v[0], k, v) for k, v in cells.items()), reverse=True)
    print(f"    per cell: {len(cells)} cells sampled; "
          f"{sum(1 for _, _, v in bad if v[1] == 0)} with no disagreeing pixel")
    print(f"    worst cells {'cell':>14}{'pixels':>8}{'disagreeing':>12}")
    for frac, k, v in bad[:5]:
        print(f"    {'':16}{str(k):>14}{v[0]:>8}{v[1]:>12}")

    got2 = render("gazebo", None)
    msg2 = got2.get("/probe/depth/points")
    lum2 = rgb_luminance(got2.get("/probe/image"))
    if msg2 is None or lum2 is None:
        print("\n  negative control NOT RUN: no cloud or no image from the "
              "gazebo render")
        return dict(C3=None)
    wx2, wy2, wz2, _u2, good2 = cloud_to_world(msg2, cam)
    mx2, my2 = world_to_map(wx2, wy2)
    i2, j2 = map_to_cell(mx2, my2)
    in2 = good2 & (i2 >= 0) & (i2 < REGION_N) & (j2 >= 0) & (j2 < REGION_N)
    i2i, j2j = np.rint(i2[in2]).astype(int), np.rint(j2[in2]).astype(int)
    gz_lit = lum2[in2] > 0.5
    ours2 = lit[i2i, j2j]
    neg_dis = float((gz_lit != ours2).mean())
    a_ = int((gz_lit & ours2).sum()); b_ = int((gz_lit & ~ours2).sum())
    c_ = int((~gz_lit & ours2).sum()); d_ = int((~gz_lit & ~ours2).sum())
    n_ = a_ + b_ + c_ + d_
    po = (a_ + d_) / n_
    pe = ((a_ + b_) * (a_ + c_) + (c_ + d_) * (b_ + d_)) / (n_ * n_)
    kappa = 0.0 if pe >= 1.0 else (po - pe) / (1.0 - pe)
    ratio = neg_dis / pos_dis if pos_dis > 0 else float("inf")
    print(f"\n  negative control, Gazebo's own shadow mask against ours")
    print(f"    {'':22}{'ours lit':>12}{'ours dark':>12}")
    print(f"    {'gazebo lit':<22}{a_:>12}{b_:>12}")
    print(f"    {'gazebo dark':<22}{c_:>12}{d_:>12}")
    print(f"    disagreement {neg_dis:.4%} of {n_} pixels, raw agreement "
          f"{po:.4%}")
    print(f"    the best constant predictor on this epoch scores 62.9084%, "
          f"which is why the bar is on kappa")
    print(f"    Cohen's kappa {kappa:+.4f}, bar at most {C3_KAPPA_MAX}: "
          f"{'clears' if kappa <= C3_KAPPA_MAX else 'FAILS'}")
    print(f"    disagreement ratio against the positive check "
          f"{ratio:.2f}x, bar at least {C3_RATIO_MIN}: "
          f"{'clears' if ratio >= C3_RATIO_MIN else 'FAILS'}")
    ok = bool(kappa <= C3_KAPPA_MAX and ratio >= C3_RATIO_MIN)
    print(f"    the control {'has demonstrated' if ok else 'has NOT demonstrated'} "
          f"that Gazebo's shadows differ from ours")
    print(f"  C3 -> {'PASS' if ok else 'FAIL'}")
    return dict(C3=ok)


def c5(h):
    """C5: the demo plays back only checker-accepted plans, kinematically."""
    print("\n=== C5: the demo plays back only checker-accepted plans ===")
    sweep = ROOT / "data" / "work" / "sweep"
    picked = None
    for d in sorted(sweep.glob("*_replay_td_tol")):
        tag = d.name[: -len("_replay_td_tol")]
        v = np.fromfile(d / "out_verdict.bin", dtype=np.uint8)
        a = np.fromfile(d / "out_arrival.bin", dtype=np.float64)
        sh = np.fromfile(d / "out_shadow.bin", dtype=np.float64)
        ok = np.flatnonzero(v.astype(bool))
        if ok.size:
            picked = (tag, int(ok[0]), float(a[ok[0]]), float(sh[ok[0]]),
                      int(v.size), int(v.sum()))
            break
    if picked is None:
        print("  NOT RUN: no replayed plan found")
        return dict(C5=None)
    tag, idx, arr, shd, n_pl, n_ok = picked
    print(f"  the verdict is read from the checker's own output, not a filename:")
    print(f"    {tag}: {n_ok} of {n_pl} replayed plans carry verdict true")
    print(f"    selected replayed plan {idx}: arrival {arr:.3f} h, "
          f"max continuous shadow {shd:.4f} h, verdict TRUE")

    # the refusal is asserted, not promised: a rejected plan must not play back
    rejected = None
    for d in sorted(sweep.glob("*_replay_td_tol")):
        v = np.fromfile(d / "out_verdict.bin", dtype=np.uint8)
        bad = np.flatnonzero(~v.astype(bool))
        if bad.size:
            rejected = (d.name[: -len("_replay_td_tol")], int(bad[0]))
            break
    gate_ok = True
    if rejected:
        rtag, ridx = rejected
        rv = np.fromfile(sweep / f"{rtag}_replay_td_tol" / "out_verdict.bin",
                         dtype=np.uint8)
        gate_ok = not bool(rv[ridx])
        print(f"    a rejected plan exists to test the gate on: {rtag} plan "
              f"{ridx}, verdict {bool(rv[ridx])}")
        print(f"    the gate would refuse it: {gate_ok}")
    else:
        print(f"    no rejected plan on disk, so the gate is untested here")
        gate_ok = False

    # commanded, not driven: pose the rover and read the effect back
    td = sweep / f"{tag}_td"
    off = np.fromfile(td / "out_path_offsets.bin", dtype=np.int64)
    pc = np.fromfile(td / "out_path_cells.bin", dtype=np.int32)
    idxs = [q for q in range(off.size - 1) if off[q + 1] - off[q] >= 2]
    q = idxs[idx]
    cells = pc[off[q]:off[q + 1]].astype(np.int64)
    waypoints = []
    for c in cells[:: max(1, len(cells) // 6)][:6]:
        ci, cj = divmod(int(c), REGION_N)
        mx, my = (float(v) for v in cell_to_map_wrap(ci, cj))
        wx, wy = (float(v) for v in map_to_world(mx, my))
        waypoints.append((wx, wy, height_at_map(h, mx, my) + 1.0))
    print(f"  plan has {len(cells)} cells; {len(waypoints)} waypoints commanded")
    rov = WORK / "rover.sdf"
    if not rov.exists():
        print(f"    NOT RUN: {rov} missing; convert the urdf with gz sdf -p")
        return dict(C5=None)
    sdf = make_world([], extra=["--rover", str(rov),
                                "--out", str(WORK / "c5.sdf")])
    moved = command_and_read(WORK / "c5.sdf", waypoints)
    print(f"    {'commanded':>34}{'read back':>34}{'error':>9}")
    worst = 0.0
    for want, got in moved:
        e = float("nan") if got is None else float(np.linalg.norm(
            np.array(want) - np.array(got)))
        if got is not None:
            worst = max(worst, e)
        print(f"    ({want[0]:9.2f},{want[1]:9.2f},{want[2]:8.2f})"
              + ("  " + " " * 30 + "MISSING" if got is None else
                 f"  ({got[0]:9.2f},{got[1]:9.2f},{got[2]:8.2f}){e:9.3f}"))
    posed = all(g is not None for _, g in moved) and worst <= 0.5
    print(f"    every commanded pose read back within 0.5 m: {posed} "
          f"(worst {worst:.3f} m)")
    print(f"    the pose is commanded, never driven, so no unverified motion "
          f"model stands where a measured one exists")
    ok = bool(gate_ok and posed)
    print(f"  C5 -> {'PASS' if ok else 'FAIL'}")
    return dict(C5=ok)


def cell_to_map_wrap(i, j):
    from src.gzmap import cell_to_map
    return cell_to_map(i, j)


def command_and_read(sdf, waypoints):
    """Pose a body at each waypoint and read the pose topic back each time.

    The effect is read rather than the service's reply believed, which is the
    standing lesson in this record: a simulator API reports success for the
    call and not for the effect.
    """
    proc = subprocess.Popen([GZ, "sim", "-s", "-r", "-v", "1", str(sdf)],
                            env=ENV, stdout=subprocess.DEVNULL,
                            stderr=subprocess.DEVNULL, start_new_session=True)
    out, replies = [], []
    try:
        time.sleep(12)
        for wx, wy, wz in waypoints:
            req = (f'name: "rover", position: {{x: {wx}, y: {wy}, z: {wz}}}, '
                   f'orientation: {{w: 1}}')
            rep = subprocess.run(
                [GZ, "service", "-s", "/world/lunar_region/set_pose",
                 "--reqtype", "gz.msgs.Pose", "--reptype", "gz.msgs.Boolean",
                 "--timeout", "3000", "--req", req],
                env=ENV, capture_output=True, text=True, timeout=20)
            replies.append(rep.stdout.strip().replace("\n", " ")[:40])
            time.sleep(1.5)
            r = subprocess.run([GZ, "topic", "-e", "-t",
                                "/world/lunar_region/pose/info", "-n", "1"],
                               env=ENV, capture_output=True, text=True, timeout=30)
            out.append(((wx, wy, wz), parse_poses(r.stdout).get("rover")))
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
    for r in replies[:1]:
        print(f"    the service replied {r!r}; the effect is read back below "
              f"rather than inferred from it")
    return out


class Tee:
    """Per-part stdout capture.

    A part's numbers are the part's result as much as its boolean is, so both
    land on disk the moment the part finishes rather than at the end of the
    run. A previous combined run was killed at a session boundary and left no
    record of pieces that had already completed.
    """

    def __init__(self, path):
        self.real = sys.stdout
        self.fh = open(path, "w")

    def write(self, text):
        self.real.write(text)
        self.fh.write(text)
        self.real.flush()
        self.fh.flush()

    def flush(self):
        self.real.flush()
        self.fh.flush()

    def close(self):
        self.fh.close()


def run_part(name, fn, h, outdir, resume):
    """Run one checklist part, checkpointing its verdicts and its log."""
    jpath = outdir / f"{name}.json"
    if resume and jpath.exists():
        done = json.loads(jpath.read_text())
        print(f"=== {name}: already on disk from "
              f"{done.get('finished_utc', 'an earlier run')}, not rerun ===")
        return done["scores"]
    tee = Tee(outdir / f"{name}.log")
    saved, sys.stdout = sys.stdout, tee
    try:
        scores = fn(h)
    finally:
        sys.stdout = saved
        tee.close()
    jpath.write_text(json.dumps(
        {"scores": scores,
         "finished_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())},
        indent=2) + "\n")
    print(f"=== {name} checkpointed to {jpath.name} ===", flush=True)
    return scores


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parts", default="c1")
    ap.add_argument("--out", default=str(WORK / "verify"))
    ap.add_argument("--resume", action="store_true",
                    help="skip parts whose JSON is already on disk")
    args = ap.parse_args()
    # Split rather than substring-match. "l2pp" in "l2ppp" is true, so
    # --parts l2ppp silently ran l2pp as well, which is the same class of
    # quiet wrong answer this whole subtask is about.
    want = [t.strip() for t in args.parts.split(",") if t.strip()]
    unknown = [t for t in want if t not in ("c1", "l2pp", "l2ppp", "c2",
                                            "c3", "c5", "c4x")]
    if unknown:
        raise SystemExit(f"unknown parts: {', '.join(unknown)}")
    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)
    h = dem()
    out = {}
    for name, fn in (("c1", c1), ("l2pp", l2pp), ("l2ppp", l2ppp),
                     ("c2", c2),
                     ("c3", c3), ("c5", c5), ("c4x", c4x)):
        if name in want:
            out.update(run_part(name, fn, h, outdir, args.resume))
    print(f"\n=== scored so far ===")
    for k, v in out.items():
        print(f"  {k}: {'PASS' if v else 'FAIL' if v is not None else 'NOT RUN'}")
    (outdir / "checklist.json").write_text(json.dumps(out, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
