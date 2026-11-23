"""Regenerate every figure the README embeds, from the run outputs in data/work.

Nothing here draws a new claim. Each figure is a crop or a downscale of
something a committed script already produced, so the full sequences stay out of
the repository and these stand in for them. Which frames become stills, and what
their captions say, is read from the demo's sidecar (data/work/demo/sidecar.json)
rather than typed in: a frame index written here as a literal kept pointing at
the previous subject's firing after the subject changed. Run after the
pipeline, from the repository root:

    .venv/bin/python docs/make_figures.py

Inputs are listed per figure below and the script names the missing ones rather
than leaving a stale file in place.
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
DEMO = ROOT / "data" / "work" / "demo"
SIDECAR = DEMO / "sidecar.json"
DOCS = Path(__file__).resolve().parent

# The hero: the region panel alone, animated over the traverse. Cropped from the
# 2D sequence rather than rendered again, so it is the same pixels the demo
# produced. The crop is derived from the figure geometry in demo_traverse rather
# than typed in, with a margin for the panel title and the axis labels, and
# stopping short of the figure header, which is wider than this crop. Every
# frame is used if the file stays under HERO_MAX_BYTES, otherwise every second.
HERO_STRIDES = (1, 2)
HERO_MAX_BYTES = 3 * 1024 * 1024
HERO_WIDTH = 460           # readable at README width without a click
HERO_MS = 130              # per frame
HERO_MARGIN = (24, 40, 44, 26)   # top, bottom, left, right, in source pixels


def dashboard_frame(tl):
    """The first frame strictly after the bounded rule's first firing.

    A firing falls between frames; the first frame after it is the first one
    that can show it. Returns (frame index, frame time, firing time)."""
    fires = tl["executive_firings_h"]
    if not fires:
        raise SystemExit("the sidecar records no firing; the dashboard still has "
                         "no frame to show one")
    f = min(fires)
    t = np.asarray(tl["frame_times_h"], float)
    after = np.flatnonzero(t > f)
    if after.size == 0:
        raise SystemExit(f"no frame after the firing at {f:.3f} h")
    k = int(after[0])
    return k, float(t[k]), float(f)


def gazebo_frame(tl):
    """A frame inside the world's longest dark run with the rover on a cell our
    model calls dark: the middle such frame. Returns (frame index, frame time,
    (run start, run end, run shadow))."""
    runs = tl["world_dark_runs_h"]
    if not runs:
        raise SystemExit("the sidecar records no dark run on the world's trajectory")
    a, b, s = max(runs, key=lambda r: r[2])
    t = np.asarray(tl["frame_times_h"], float)
    dark = ~np.asarray(tl["rover_cell_lit"], bool)
    inside = np.flatnonzero((t > a) & (t < b) & dark)
    if inside.size == 0:
        raise SystemExit(f"no frame inside the dark run {a:.3f} to {b:.3f} h has "
                         f"the rover on a cell the model calls dark")
    k = int(inside[inside.size // 2])
    return k, float(t[k]), (float(a), float(b), float(s))


def figures(side):
    """(name, source, downscale, caption) for every still, from the sidecar."""
    tl = side["timeline"]
    kd, td, fire = dashboard_frame(tl)
    kg, tg, (a, b, s) = gazebo_frame(tl)
    return (
        ("dashboard.png", DEMO / "map" / f"frame{kd:04d}.png", 1,
         f"the full 2D view at t = +{td:.2f} h, the first frame after the bounded "
         f"rule fires at {fire:.3f} h"),
        ("epochs.png", DEMO / "panel_r4_epochs.png", 2,
         "all twelve epochs, no selection"),
        ("executive.png", DEMO / "panel_r3_false_alarm.png", 2,
         "the executive on R3, with and without the B3b-lit correction"),
        ("gazebo.png", DEMO / "gz" / f"gz{kg:04d}.png", 2,
         f"the Gazebo view at t = +{tg:.2f} h, inside the world's {s:.2f} h dark "
         f"run from {a:.2f} to {b:.2f} h, rover on a cell the model calls dark"),
    )


def box_downscale(a, k):
    """Integer box mean, which cannot invent a value the source did not have.

    That matters for the map panels: they encode a lit or dark mask, and a
    smooth resample would put intermediate greys where the model has none.
    """
    if k <= 1:
        return a
    h, w = a.shape[:2]
    h2, w2 = (h // k) * k, (w // k) * k
    return a[:h2, :w2].reshape(h2 // k, k, w2 // k, k, -1).mean(axis=(1, 3))


def read_rgb(p):
    import matplotlib.image as mpimg
    a = mpimg.imread(p)
    return a[..., :3] if a.ndim == 3 and a.shape[2] == 4 else a


def region_crop_box(shape):
    """The region panel's pixel box, from demo_traverse's own figure geometry."""
    import examples.demo_traverse as D
    h, w = shape[:2]
    x0, y0, dx, dy = D.REGION_RECT
    top, bot, left, right = HERO_MARGIN
    return (int(round((1.0 - (y0 + dy)) * h)) - top,
            int(round((1.0 - y0) * h)) + bot,
            int(round(x0 * w)) - left,
            int(round((x0 + dx) * w)) + right)


def write_hero(frames, dst, stride):
    from PIL import Image
    r0, r1, c0, c1 = region_crop_box(read_rgb(frames[0]).shape)
    out = []
    for f in frames[::stride]:
        a = read_rgb(f)[r0:r1, c0:c1]
        k = max(1, int(round(a.shape[1] / HERO_WIDTH)))
        a = box_downscale(a, k)
        im = Image.fromarray((np.clip(a, 0, 1) * 255 + 0.5).astype(np.uint8))
        # A fixed adaptive palette per frame keeps the lit and dark levels
        # distinct; the panel carries few colours, so 64 is more than enough and
        # keeps the file small.
        out.append(im.convert("P", palette=Image.ADAPTIVE, colors=64))
    out[0].save(dst, save_all=True, append_images=out[1:], loop=0,
                duration=HERO_MS, optimize=True)
    return len(out), out[0].size


def hero_stride(size_at, strides=HERO_STRIDES, limit=HERO_MAX_BYTES):
    """The first stride whose file is under the limit, else the last tried.
    size_at(stride) writes the file and returns its size in bytes."""
    for s in strides:
        b = size_at(s)
        if b <= limit:
            return s, b
    return s, b


def main():
    if not SIDECAR.exists():
        print(f"missing {SIDECAR}; run examples/demo_traverse.py first")
        return 1
    side = json.loads(SIDECAR.read_text())
    frames = sorted((DEMO / "map").glob("frame*.png"))
    n_side = len(side["timeline"]["frame_times_h"])
    if len(frames) != n_side:
        print(f"demo/map holds {len(frames)} frames and the sidecar {n_side}; "
              f"they are not one run")
        return 1
    figs = figures(side)
    missing = [str(s) for _, s, _, _ in figs if not s.exists()]
    if missing:
        print("missing inputs; run the pipeline first:")
        for m in missing:
            print(f"  {m}")
        return 1

    import matplotlib.image as mpimg
    dst = DOCS / "hero.gif"
    shape = {}

    def size_at(s):
        shape["n"], shape["size"] = write_hero(frames, dst, s)
        return dst.stat().st_size

    stride, b = hero_stride(size_at)
    cad = side["configuration"]["cadence_h"] * stride
    every = "every frame" if stride == 1 else f"every {stride} of the sequence"
    print(f"  docs/hero.gif        {shape['size'][0]:5d} x {shape['size'][1]:4d}  "
          f"{b / 1024.0:7.1f} KB  {shape['n']} frames at {HERO_MS} ms, {every}, "
          f"one per {cad:.2f} h of the traverse")

    for name, src, k, what in figs:
        a = box_downscale(read_rgb(src), k)
        mpimg.imsave(DOCS / name,
                     (np.clip(a, 0, 1) * 255 + 0.5).astype(np.uint8))
        kb = (DOCS / name).stat().st_size / 1024.0
        print(f"  docs/{name:16s} {a.shape[1]:5d} x {a.shape[0]:4d}  {kb:7.1f} KB"
              f"  (1/{k} of source, {src.name})  {what}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
