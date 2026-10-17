"""
Quantify the 16x16 tile seams in a directory of renders.

The rasterizer bins work into 16x16 pixel tiles.  If the per-tile candidate cap
(K_MAX) truncates the gaussian list, each tile is synthesized from a slightly
different set of gaussians, and the image shows a discontinuity at every tile
boundary -- i.e. at every x or y that is a multiple of 16.

This measures that directly.  For each column x it computes the mean absolute
horizontal gradient |I[:, x] - I[:, x-1]|, averaged over every image in a
directory.  Columns where x % 16 == 0 sit exactly on a tile boundary; all other
columns are tile-interior.  The reported "seam ratio" is

    mean gradient on tile boundaries / mean gradient in tile interior

A seam-free render still has *some* structure at arbitrary columns, so the ratio
does not have to be 1.0 -- what matters is comparing the same scene at two K_MAX
values.  If raising K_MAX removes seams, the ratio drops toward 1.

Usage:
  python seam_metric.py <dir> [<dir> ...] [--tile 16]
"""

import argparse
import os

import numpy as np
from PIL import Image


def gradients(d):
    """Return (mean horizontal gradient per column boundary, mean vertical per row)."""
    names = sorted(n for n in os.listdir(d) if n.lower().endswith(".png"))
    if not names:
        raise SystemExit(f"no PNGs in {d}")

    hacc = vacc = None
    for n in names:
        a = np.asarray(Image.open(os.path.join(d, n)).convert("RGB"), np.float32)
        # horizontal: |I[:, x] - I[:, x-1]|, averaged over rows and channels
        h = np.abs(np.diff(a, axis=1)).mean(axis=(0, 2))     # length W-1
        v = np.abs(np.diff(a, axis=0)).mean(axis=(1, 2))     # length H-1
        hacc = h if hacc is None else hacc + h
        vacc = v if vacc is None else vacc + v
    return hacc / len(names), vacc / len(names), len(names)


def ratio(grad, tile):
    """grad[i] is the gradient across the boundary between index i and i+1."""
    idx = np.arange(len(grad)) + 1          # boundary position in pixel coords
    on = idx % tile == 0
    off = ~on
    if on.sum() == 0 or off.sum() == 0:
        return float("nan"), float("nan"), float("nan")
    return float(grad[on].mean()), float(grad[off].mean()), float(grad[on].mean() / grad[off].mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dirs", nargs="+")
    ap.add_argument("--tile", type=int, default=16)
    args = ap.parse_args()

    print(f"tile size = {args.tile}; seam ratio = mean gradient on tile boundaries / in tile interior")
    print("(a seam-free render has ratio near 1; higher means visible tile edges)\n")
    print(f"{'render dir':<46} {'views':>5} {'H-bound':>9} {'H-in':>9} {'H ratio':>8} "
          f"{'V-bound':>9} {'V-in':>9} {'V ratio':>8}")

    for d in args.dirs:
        h, v, n = gradients(d)
        hb, hi, hr = ratio(h, args.tile)
        vb, vi, vr = ratio(v, args.tile)
        label = os.path.basename(d.rstrip("\\/"))
        parent = os.path.basename(os.path.dirname(d.rstrip("\\/")))
        label = f"{parent}/{label}"
        print(f"{label:<46} {n:>5} {hb:>9.4f} {hi:>9.4f} {hr:>8.3f} "
              f"{vb:>9.4f} {vi:>9.4f} {vr:>8.3f}")


if __name__ == "__main__":
    main()
