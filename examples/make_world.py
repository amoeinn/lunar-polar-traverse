"""Generate the Gazebo world for the planning region.

Generated rather than hand-written so the heightmap's own measured constants,
the height range and the pos offset, are single-sourced from the export's
sidecar. Copying 294.7623 into an SDF by hand is how a demo drifts away from
the terrain it claims to show.

Two lighting modes, and the reason the second exists is PLAN.md's record that
Gazebo's shadows are never a source of truth here.

  --mode ours    scene ambient 1, no directional contribution. Ambient light
                 has no direction, so Gazebo applies no normal-dot-light term
                 and the rendered brightness is our illumination texture and
                 nothing else. This is how the design's "unlit map" is actually
                 achieved: sdformat's heightmap texture element has no emissive
                 slot, so the terrain material cannot be made unlit directly.
  --mode gazebo  scene ambient 0, one directional light from our own sun vector
                 with cast_shadows true, and a uniform grey texture. The
                 rendered darkness is then Gazebo's own shading and shadow map.
                 This is the negative control, not the demo.
"""
import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.gzmap import HEIGHTMAP_N, PX, SAMPLING, requested_size, span_m  # noqa: E402

OUT = ROOT / "data" / "work" / "gazebo"

PROBE_RADIUS_M = 2.0        # small against a 20 m cell, large against contact slop


def flat_normal_png(path):
    """A flat normal map, required by the heightmap texture element."""
    import matplotlib.image as mpimg
    a = np.zeros((4, 4, 3), dtype=np.uint8)
    a[..., 0] = 128
    a[..., 1] = 128
    a[..., 2] = 255
    mpimg.imsave(path, a)
    return path


def grey_png(path, value=128):
    import matplotlib.image as mpimg
    a = np.full((4, 4), value, dtype=np.uint8)
    mpimg.imsave(path, a, cmap="gray", vmin=0, vmax=255)
    return path


def rover_block(path, name="rover", pose=(0.0, 0.0, 2000.0)):
    """The rover, converted from its URDF by sdformat and renamed.

    Converted rather than hand-written in SDF so the xacro stays the single
    place the sourced figures live: gz sdf -p turns the expanded URDF into SDF
    with all five sensors preserved.
    """
    txt = Path(path).read_text()
    m = re.search(r"(<model name=.*?</model>)", txt, flags=re.S)
    if not m:
        raise SystemExit(f"no model block in {path}")
    block = re.sub(r"<model name='[^']*'", f"<model name='{name}'",
                   m.group(1), count=1)
    x, y, z = pose
    # Dynamic with gravity off, which took two measured attempts to get right
    # and is worth recording. Left dynamic with gravity, the physics engine
    # overrode every commanded pose and the rover free-fell. Made static, the
    # pose command had no effect at all: gz-sim does not move static entities,
    # so the service accepted the call and moved nothing, which is the standing
    # lesson in this record about trusting a return value. Dynamic with no
    # gravity is the combination where a commanded pose both applies and stays.
    block = block.replace(">", f">\n      <pose>{x} {y} {z} 0 0 0</pose>", 1)
    # <gravity> is a link element, not a model element. Placed before the
    # <link> tag it becomes a child of <model>, where sdformat drops it with a
    # warning and still reports the world Valid, so the rover fell with the
    # element sitting right there in the file. The first diagnosis of this was
    # that the edit had reached only one of two links; the rover has exactly
    # one link, because sdformat merges the fixed joints into base_link, and
    # the position was the fault rather than the count. Inserted inside every
    # link so adding a non-fixed joint later cannot reintroduce it.
    block = re.sub(r"(<link name=[^>]*>)", r"\1\n        <gravity>false</gravity>",
                   block)
    return "\n    " + block


def probe_model(name, wx, wy, wz, radius=PROBE_RADIUS_M):
    """A sphere dropped at a known world position, to be read back once settled."""
    return f"""
    <model name="{name}">
      <pose>{wx:.4f} {wy:.4f} {wz:.4f} 0 0 0</pose>
      <link name="link">
        <inertial><mass>10.0</mass>
          <inertia><ixx>16.0</ixx><iyy>16.0</iyy><izz>16.0</izz>
                   <ixy>0</ixy><ixz>0</ixz><iyz>0</iyz></inertia></inertial>
        <collision name="c"><geometry><sphere><radius>{radius}</radius></sphere>
          </geometry></collision>
        <visual name="v"><geometry><sphere><radius>{radius}</radius></sphere>
          </geometry></visual>
      </link>
    </model>"""


def camera_rig(wx, wy, wz, fov_rad, width, height):
    """A co-located camera and depth camera looking straight down.

    The depth channel is what makes the illumination check independent of the
    heightmap mapping: each pixel's own range says where that pixel actually
    hit, so pixel-to-cell comes from the simulator's geometry rather than from
    assumed projection arithmetic.
    """
    return f"""
    <model name="probe_cam">
      <static>true</static>
      <pose>{wx:.4f} {wy:.4f} {wz:.4f} 0 1.5707963 0</pose>
      <link name="link">
        <sensor name="rgb" type="camera">
          <always_on>1</always_on><update_rate>5</update_rate>
          <topic>probe/image</topic>
          <camera>
            <horizontal_fov>{fov_rad:.9f}</horizontal_fov>
            <image><width>{width}</width><height>{height}</height></image>
            <clip><near>1.0</near><far>200000.0</far></clip>
          </camera>
        </sensor>
        <sensor name="depth" type="depth_camera">
          <always_on>1</always_on><update_rate>5</update_rate>
          <topic>probe/depth</topic>
          <camera>
            <horizontal_fov>{fov_rad:.9f}</horizontal_fov>
            <image><width>{width}</width><height>{height}</height></image>
            <clip><near>1.0</near><far>200000.0</far></clip>
          </camera>
        </sensor>
      </link>
    </model>"""


def ray_rig(name, wx, wy, wz, fov_rad, width, height):
    """A nadir depth camera at a fixed pose: L2's instrument, which cannot roll.

    L2 originally dropped spheres and measured them after settling, and what it
    measured was the spheres. They rolled 316 to 1230 m down the slopes they
    landed on and were still creeping when the settling window closed, and six
    of seven residuals came back positive, which is a 2 m sphere sitting above
    a slope rather than a displaced heightmap.

    Nothing here moves. The camera is static, and each pixel's own returned
    range gives the terrain point it hit, so the height comes from the
    simulator's own geometry at a horizontal position that is read back from
    the same return rather than assumed.
    """
    return f"""
    <model name="ray_{name}">
      <static>true</static>
      <pose>{wx:.4f} {wy:.4f} {wz:.4f} 0 1.5707963 0</pose>
      <link name="link">
        <sensor name="depth" type="depth_camera">
          <always_on>1</always_on><update_rate>5</update_rate>
          <topic>ray/{name}/depth</topic>
          <camera>
            <horizontal_fov>{fov_rad:.9f}</horizontal_fov>
            <image><width>{width}</width><height>{height}</height></image>
            <clip><near>1.0</near><far>200000.0</far></clip>
          </camera>
        </sensor>
      </link>
    </model>"""


def build(side, mode, texture, normal, pos_z, light_dir, probes, cam, convention,
          h_range, rover=None, rays=()):
    span = span_m(HEIGHTMAP_N, PX, convention)
    size_xy = requested_size(HEIGHTMAP_N, PX, SAMPLING, convention)
    ambient = "1 1 1 1" if mode == "ours" else "0 0 0 1"
    light = ""
    if mode == "gazebo":
        dx, dy, dz = light_dir
        light = f"""
    <light type="directional" name="sun">
      <cast_shadows>true</cast_shadows>
      <diffuse>1 1 1 1</diffuse><specular>0 0 0 1</specular>
      <direction>{dx:.9f} {dy:.9f} {dz:.9f}</direction>
    </light>"""
    cam_sdf = camera_rig(*cam) if cam else ""
    cam_sdf += "".join(ray_rig(*r) for r in rays)
    rover_sdf = rover_block(rover) if rover else ""
    probe_sdf = "".join(probe_model(n, x, y, z) for n, x, y, z in probes)
    return f"""<?xml version="1.0" ?>
<!-- Generated by examples/make_world.py; edit that, not this. -->
<sdf version="1.9">
  <world name="lunar_region">
    <spherical_coordinates><surface_model>MOON_SCS</surface_model></spherical_coordinates>
    <gravity>0 0 -1.62</gravity>
    <physics name="default" type="ignored">
      <max_step_size>0.004</max_step_size>
      <dart>
        <!-- the installed dem_moon.sdf example: heightmaps behave better with
             the bullet collision detector -->
        <collision_detector>bullet</collision_detector>
      </dart>
    </physics>
    <plugin filename="gz-sim-physics-system" name="gz::sim::systems::Physics"/>
    <plugin filename="gz-sim-sensors-system" name="gz::sim::systems::Sensors">
      <render_engine>ogre2</render_engine>
    </plugin>
    <plugin filename="gz-sim-scene-broadcaster-system"
            name="gz::sim::systems::SceneBroadcaster"/>
    <!-- UserCommands provides set_pose. Declaring plugins explicitly replaces
         Gazebo's defaults, so omitting this one leaves the service with no
         provider at all: listing the running world's services showed no
         set_pose, no create and no remove, and every call came back with an
         empty reply. That empty reply was the service not existing, and it was
         visible from the first attempt before it was diagnosed. -->
    <plugin filename="gz-sim-user-commands-system"
            name="gz::sim::systems::UserCommands"/>
    <scene>
      <ambient>{ambient}</ambient>
      <background>0 0 0 1</background>
      <grid>false</grid>
    </scene>{light}
    <model name="region">
      <static>true</static>
      <link name="link">
        <collision name="collision">
          <geometry><heightmap>
            <uri>file://{side}</uri>
            <size>{size_xy:.4f} {size_xy:.4f} {h_range:.6f}</size>
            <pos>0 0 {pos_z:.6f}</pos>
            <sampling>{SAMPLING}</sampling>
          </heightmap></geometry>
        </collision>
        <visual name="visual">
          <geometry><heightmap>
            <uri>file://{side}</uri>
            <size>{size_xy:.4f} {size_xy:.4f} {h_range:.6f}</size>
            <pos>0 0 {pos_z:.6f}</pos>
            <sampling>{SAMPLING}</sampling>
            <texture>
              <size>{span:.1f}</size>
              <diffuse>file://{texture}</diffuse>
              <normal>file://{normal}</normal>
            </texture>
          </heightmap></geometry>
        </visual>
      </link>
    </model>{probe_sdf}{cam_sdf}{rover_sdf}
  </world>
</sdf>
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=("ours", "gazebo"), default="ours")
    ap.add_argument("--texture", default="")
    ap.add_argument("--convention", default="vertices")
    ap.add_argument("--light", default="0 0 -1")
    ap.add_argument("--probes", default="")
    ap.add_argument("--cam", default="")
    ap.add_argument("--z-offset", type=float, default=None,
                    help="heightmap pos z; the calibration measures it")
    ap.add_argument("--rays", default="",
                    help="L2' nadir depth cameras: name,x,y,z,fov,w,h;...")
    ap.add_argument("--rover", default="",
                    help="path to the sdformat-converted rover")
    ap.add_argument("--out", default="")
    args = ap.parse_args()
    side = json.loads((OUT / "region_heightmap.json").read_text())
    if args.z_offset is None:
        args.z_offset = side["pos_z"]
    normal = flat_normal_png(OUT / "flat_normal.png")
    texture = Path(args.texture) if args.texture else grey_png(OUT / "grey.png")
    probes = []
    if args.probes:
        for k, item in enumerate(args.probes.split(";")):
            x, y, z = (float(v) for v in item.split(","))
            probes.append((f"probe{k:02d}", x, y, z))
    rays = []
    if args.rays:
        for item in args.rays.split(";"):
            f = item.split(",")
            rays.append((f[0], float(f[1]), float(f[2]), float(f[3]),
                         float(f[4]), int(f[5]), int(f[6])))
    cam = tuple(float(v) for v in args.cam.split(",")) if args.cam else None
    if cam:
        cam = (cam[0], cam[1], cam[2], cam[3], int(cam[4]), int(cam[5]))
    sdf = build(side["path"], args.mode, texture, normal, args.z_offset,
                [float(v) for v in args.light.split()], probes, cam,
                args.convention, side["h_range"],
                args.rover or None, rays)
    out = Path(args.out) if args.out else OUT / f"lunar_region_{args.mode}.sdf"
    out.write_text(sdf)
    print(f"wrote {out}")
    print(f"  mode {args.mode}, ambient {'1 (our texture, unshaded)' if args.mode=='ours' else '0 with a shadow-casting sun'}")
    print(f"  heightmap {side['path']}, span {span_m(HEIGHTMAP_N, PX, args.convention):.0f} m, "
          f"pos z {args.z_offset:.4f}, size requested "
          f"{requested_size(HEIGHTMAP_N, PX, SAMPLING, args.convention):.4f} "
          f"for a true {span_m(HEIGHTMAP_N, PX, args.convention):.0f} m span")
    print(f"  {len(probes)} probes, {len(rays)} ray rigs, "
          f"camera rig: {'yes' if cam else 'no'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
