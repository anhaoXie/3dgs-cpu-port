"""
Build the "tile seams are real, and K_MAX causes them" evidence figure.

Takes one view rendered at two K_MAX settings, auto-locates the region where the
K=256 render has the strongest tile-boundary discontinuity, and crops that same
region out of both.  A 16-pixel tile grid is drawn over both crops in red, so the
reader can see directly whether the visible block edges line up with the tile
grid -- and whether raising K_MAX removes them.

Usage:
  python seam_figure.py --a <render_dir_K256> --b <render_dir_K1024>
                        [--view 00051.png] [--out <png>] [--scale 5] [--tiles 8 6]
"""

import argparse
import os

import numpy as np
from PIL import Image, ImageDraw

TILE = 16


def load(d, view):
    return np.asarray(Image.open(os.path.join(d, view)).convert("RGB"), np.float32)


def worst_boundary_x(a, tile=TILE):
    """Column (a multiple of tile) with the largest mean horizontal jump in `a`."""
    g = np.abs(np.diff(a, axis=1)).mean(axis=(0, 2))     # g[i] spans x=i..i+1
    best, bx = -1.0, tile
    for x in range(tile, g.shape[0], tile):
        if g[x - 1] > best:                              # jump across x-1 -> x
            best, bx = g[x - 1], x
    return bx


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True, help="render dir with the seams (K=256)")
    ap.add_argument("--b", required=True, help="render dir without them (K=1024)")
    ap.add_argument("--view", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--scale", type=int, default=5)
    ap.add_argument("--tiles", type=int, nargs=2, default=[8, 6], metavar=("W", "H"))
    args = ap.parse_args()

    names = sorted(n for n in os.listdir(args.a) if n.lower().endswith(".png"))
    view = args.view or names[len(names) // 2]

    A = load(args.a, view)
    B = load(args.b, view)
    H, W, _ = A.shape

    bx = worst_boundary_x(A)
    # crop spanning the worst boundary, snapped to the tile grid
    cw, ch = args.tiles[0] * TILE, args.tiles[1] * TILE
    x0 = int(np.clip(bx - cw // 2, 0, max(0, W - cw)))
    x0 -= x0 % TILE
    y0 = min(64, max(0, H - ch))
    y0 -= y0 % TILE

    s = args.scale
    panels = []
    for arr, tag in ((A, "K=256"), (B, "K=1024")):
        crop = arr[y0:y0 + ch, x0:x0 + cw]
        img = Image.fromarray(crop.astype(np.uint8)).resize((cw * s, ch * s), Image.NEAREST)
        d = ImageDraw.Draw(img)
        # red grid exactly on the 16-px tile boundaries
        for gx in range(0, cw + 1, TILE):
            d.line([(gx * s, 0), (gx * s, ch * s)], fill=(255, 0, 0), width=1)
        for gy in range(0, ch + 1, TILE):
            d.line([(0, gy * s), (cw * s, gy * s)], fill=(255, 0, 0), width=1)
        panels.append((img, tag))

    gap = 14
    sheet = Image.new("RGB", (panels[0][0].width * 2 + gap, panels[0][0].height), (20, 20, 20))
    sheet.paste(panels[0][0], (0, 0))
    sheet.paste(panels[1][0], (panels[0][0].width + gap, 0))

    out = args.out or os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "notes", "probe", f"seam_compare_{os.path.splitext(view)[0]}.png")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    sheet.save(out)

    print(f"view            : {view}   ({W}x{H})")
    print(f"worst boundary x: {bx}   (multiple of {TILE}")
    print(f"crop            : x {x0}..{x0 + cw}, y {y0}..{y0 + ch}   scale {s}x NEAREST")
    print(f"layout          : LEFT = K=256 (seams), RIGHT = K=1024")
    print(f"red lines       : the 16-px rasterizer tile grid")
    print(f"wrote           : {out}  ({sheet.width}x{sheet.height})")


if __name__ == "__main__":
    main()
