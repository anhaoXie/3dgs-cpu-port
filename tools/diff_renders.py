"""
Pixel-level diff between two render directories.

Compares decoded RGB pixels (not PNG bytes -- two encodings of the same image
can differ in length while being pixel-identical, and the reverse).

Usage:
  python diff_renders.py <dir_a> <dir_b> [--top N]
"""

import argparse
import os
import sys

import numpy as np
from PIL import Image


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("a")
    ap.add_argument("b")
    ap.add_argument("--top", type=int, default=8, help="how many worst files to list")
    args = ap.parse_args()

    names = sorted(n for n in os.listdir(args.a) if n.lower().endswith(".png"))
    rows = []
    for n in names:
        pb = os.path.join(args.b, n)
        if not os.path.isfile(pb):
            continue
        A = np.asarray(Image.open(os.path.join(args.a, n)).convert("RGB"), np.int16)
        B = np.asarray(Image.open(pb).convert("RGB"), np.int16)
        if A.shape != B.shape:
            rows.append((n, -1.0, 1.0))
            continue
        d = np.abs(A - B)
        rows.append((n, float(d.mean()), float(d.max())))

    if not rows:
        raise SystemExit("no overlapping files")

    means = np.array([r[1] for r in rows])
    maxes = np.array([r[2] for r in rows])
    ident = int((maxes == 0).sum())

    print(f"compared {len(rows)} paired renders")
    print(f"  pixel-identical files : {ident}")
    print(f"  differing files       : {len(rows) - ident}")
    print(f"  mean |delta| per pixel, averaged over files : {means.mean():.4f} / 255")
    print(f"  worst single-pixel delta                    : {int(maxes.max())} / 255")
    print(f"  worst file by mean delta                    : {maxes.argmax()} -> {means.max():.4f}")

    print(f"\n  top {args.top} files by mean |delta|:")
    for n, m, mx in sorted(rows, key=lambda r: -r[1])[: args.top]:
        print(f"    {n}  mean={m:7.4f}  max={int(mx)}")


if __name__ == "__main__":
    main()
