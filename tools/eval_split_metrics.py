"""
Compute PSNR / SSIM / LPIPS over an already-rendered split, using the official
upstream metric functions.

Why this exists: metrics.py only ever walks `test/`.  That makes the train/test
gap -- the quantity that separates "the model is underfitting" from "the model
is overfitting" -- impossible to read off a single training run, which is
exactly the question this reproduction is trying to answer.  This walks
whichever split you name, so both halves of the gap come from one run and one
rendering pass.

Because it re-derives the same numbers metrics.py reports, running it on `test`
also doubles as a cross-check that the two agree.

It computes metrics only -- it never writes an image -- so it cannot clobber
the montages under notes/.

Usage:
  python tools/eval_split_metrics.py --model <run dir> --iteration 7000 --split test
  python tools/eval_split_metrics.py --model <run dir> --iteration 7000 --split train
  python tools/eval_split_metrics.py --model <run dir> --iteration 7000 --split test --json out.json
"""

import argparse
import json
import os
import sys

import numpy as np
import torch
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
WORK = os.path.dirname(HERE)
REPO = os.path.join(WORK, "gaussian-splatting")
sys.path.insert(0, REPO)

from utils.loss_utils import ssim                                  # noqa: E402
from utils.image_utils import psnr                                 # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("--model", required=True, help="trained model directory")
    ap.add_argument("--iteration", type=int, required=True)
    ap.add_argument("--split", default="test", choices=["train", "test"])
    ap.add_argument("--json", default=None, help="also write the numbers here")
    args = ap.parse_args()

    base = os.path.join(args.model, args.split, f"ours_{args.iteration}")
    rdir, gdir = os.path.join(base, "renders"), os.path.join(base, "gt")
    if not os.path.isdir(rdir):
        raise SystemExit(f"no renders at {rdir}")
    if not os.path.isdir(gdir):
        raise SystemExit(f"no ground truth at {gdir}")

    # lpipsPyTorch needs its VGG weights; skip the metric rather than the run.
    try:
        from lpipsPyTorch import lpips
        have_lpips = True
    except Exception as exc:                                        # noqa: BLE001
        print("LPIPS unavailable, skipping:", exc)
        have_lpips = False

    names = sorted(n for n in os.listdir(rdir) if n.lower().endswith(".png"))
    if not names:
        raise SystemExit(f"no PNGs in {rdir}")

    gt_names = set(os.listdir(gdir))
    unpaired = [n for n in names if n not in gt_names]
    if unpaired:
        raise SystemExit(f"{len(unpaired)} render(s) have no ground truth, e.g. {unpaired[:3]}")

    psnrs, ssims, lpipss = [], [], []
    for n in names:
        r = np.asarray(Image.open(os.path.join(rdir, n)).convert("RGB"), np.float32) / 255.0
        g = np.asarray(Image.open(os.path.join(gdir, n)).convert("RGB"), np.float32) / 255.0
        if r.shape != g.shape:
            raise SystemExit(f"{n}: render {r.shape} and gt {g.shape} differ")
        rt = torch.from_numpy(r).permute(2, 0, 1)[None].contiguous()
        gt = torch.from_numpy(g).permute(2, 0, 1)[None].contiguous()
        psnrs.append(float(psnr(rt, gt)))
        ssims.append(float(ssim(rt, gt)))
        if have_lpips:
            lpipss.append(float(lpips(rt, gt, net_type="vgg")))

    h, w = r.shape[:2]
    print(f"\n=== {args.model} | split={args.split} | iteration {args.iteration} ===")
    print(f"  views {len(names)}   resolution {w}x{h} (render pixels)")
    print(f"  PSNR  mean={np.mean(psnrs):.4f}   (min {np.min(psnrs):.4f}, max {np.max(psnrs):.4f})")
    print(f"  SSIM  mean={np.mean(ssims):.4f}   (min {np.min(ssims):.4f}, max {np.max(ssims):.4f})")
    if lpipss:
        print(f"  LPIPS mean={np.mean(lpipss):.4f}   (min {np.min(lpipss):.4f}, max {np.max(lpipss):.4f})")

    if args.json:
        payload = {
            "model": os.path.abspath(args.model),
            "split": args.split,
            "iteration": args.iteration,
            "views": len(names),
            "resolution": [w, h],
            "PSNR": float(np.mean(psnrs)),
            "SSIM": float(np.mean(ssims)),
            "LPIPS": float(np.mean(lpipss)) if lpipss else None,
        }
        with open(args.json, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2)
        print(f"\nwrote {args.json}")


if __name__ == "__main__":
    main()
