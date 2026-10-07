"""Regenerate every figure the README embeds, from the run outputs in data/work.

Nothing here draws a new claim. Each figure is a crop or a downscale of
something a committed script already produced, so the full sequences stay out of
the repository and these stand in for them. Run after the pipeline, from the
repository root:

    .venv/bin/python docs/make_figures.py

Inputs are listed per figure below and the script names the missing ones rather
than leaving a stale file in place.
"""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
DEMO = ROOT / "data" / "work" / "demo"
DOCS = Path(__file__).resolve().parent

# The hero: the region panel alone, animated over the traverse. Cropped from the
# 2D sequence rather than rendered again, so it is the same pixels the demo
# produced. The crop is derived from the figure geometry in demo_traverse rather
# than typed in, with a margin for the panel title and the axis labels, and
# stopping short of the figure header, which is wider than this crop.
HERO_STRIDE = 2            # 112 frames at 0.25 h -> 56 at 0.5 h
HERO_WIDTH = 460           # readable at README width without a click
HERO_MS = 130              # per frame
HERO_MARGIN = (24, 40, 44, 26)   # top, bottom, left, right, in source pixels

FIGURES = (
    ("dashboard.png", DEMO / "map" / "frame0028.png", 1,
     "the full 2D view at t = +7.00 h, the frame where the bounded rule fires"),
    ("epochs.png", DEMO / "panel_r4_epochs.png", 2,
     "all twelve epochs, no selection"),
    ("executive.png", DEMO / "panel_r3_false_alarm.png", 2,
     "the executive on R3, with and without the B3b-lit correction"),
    ("gazebo.png", DEMO / "gz" / "gz0029.png", 2,
     "the Gazebo view at t = +7.25 h, rover on a dark cell"),
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


def make_hero(dst):
    from PIL import Image
    frames = sorted((DEMO / "map").glob("frame*.png"))
    if not frames:
        return None
    r0, r1, c0, c1 = region_crop_box(read_rgb(frames[0]).shape)
    out = []
    for f in frames[::HERO_STRIDE]:
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


def main():
    missing = [str(s) for _, s, _, _ in FIGURES if not s.exists()]
    if not sorted((DEMO / "map").glob("frame*.png")):
        missing.append(str(DEMO / "map" / "frame*.png"))
    if missing:
        print("missing inputs; run the pipeline first:")
        for m in missing:
            print(f"  {m}")
        return 1

    import matplotlib.image as mpimg
    n, size = make_hero(DOCS / "hero.gif")
    kb = (DOCS / "hero.gif").stat().st_size / 1024.0
    print(f"  docs/hero.gif        {size[0]:5d} x {size[1]:4d}  {kb:7.1f} KB  "
          f"{n} frames at {HERO_MS} ms, every {HERO_STRIDE} of the sequence")

    for name, src, k, what in FIGURES:
        a = box_downscale(read_rgb(src), k)
        mpimg.imsave(DOCS / name,
                     (np.clip(a, 0, 1) * 255 + 0.5).astype(np.uint8))
        kb = (DOCS / name).stat().st_size / 1024.0
        print(f"  docs/{name:16s} {a.shape[1]:5d} x {a.shape[0]:4d}  {kb:7.1f} KB"
              f"  (1/{k} of source)  {what}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
