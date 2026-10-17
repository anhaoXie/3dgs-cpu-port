"""
Build a side-by-side (render | ground truth) montage for a few views, upscaled
so the images are actually inspectable, and print the per-view PSNR/SSIM/LPIPS.

Usage:
  python make_montage.py --model <model_path> --iteration N [--views 0 60 120 180]
                         [--out <png>] [--scale 4]
"""

import argparse
import os
import sys

import numpy as np
import torch
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
WORK = os.path.dirname(HERE)
sys.path.insert(0, os.path.dirname(WORK))          # allow `utils.*` if cwd is repo


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--iteration", type=int, required=True)
    ap.add_argument("--split", default="train")
    ap.add_argument("--views", type=int, nargs="*", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--single-out", default=None,
                    help="where to write the upscaled single view (default: notes/render_<split>_<iter>_view00000.png). "
                         "Set this when probing an ablation, so the delivery image is not overwritten.")
    ap.add_argument("--scale", type=int, default=4)
    args = ap.parse_args()

    base = os.path.join(args.model, args.split,
                        f"ours_{args.iteration}")
    rdir = os.path.join(base, "renders")
    gdir = os.path.join(base, "gt")
    if not os.path.isdir(rdir):
        raise SystemExit(f"no renders at {rdir}")

    names = sorted(os.listdir(rdir))
    idxs = args.views if args.views else [0, len(names) // 4, len(names) // 2,
                                          3 * len(names) // 4]
    idxs = [i for i in idxs if 0 <= i < len(names)]

    # official metric functions from the unmodified upstream checkout
    sys.path.insert(0, os.path.join(os.path.dirname(HERE), "gaussian-splatting"))
    from utils.loss_utils import ssim                       # noqa: E402
    from utils.image_utils import psnr                      # noqa: E402

    psnrs, ssims, lpipss = [], [], []
    try:
        from lpipsPyTorch import lpips
        have_lpips = True
    except Exception as exc:                                # noqa: BLE001
        print("LPIPS unavailable:", exc)
        have_lpips = False

    # evaluate every view (metrics), but only build a montage for a few
    for n in names:
        r = np.asarray(Image.open(os.path.join(rdir, n)).convert("RGB"), np.float32) / 255.0
        g = np.asarray(Image.open(os.path.join(gdir, n)).convert("RGB"), np.float32) / 255.0
        rt = torch.from_numpy(r).permute(2, 0, 1)[None].contiguous()
        gt = torch.from_numpy(g).permute(2, 0, 1)[None].contiguous()
        psnrs.append(float(psnr(rt, gt)))
        ssims.append(float(ssim(rt, gt)))
        if have_lpips:
            lpipss.append(float(lpips(rt, gt, net_type="vgg")))

    print(f"\n=== {args.split} split, iteration {args.iteration}, {len(names)} views ===")
    print(f"  PSNR  mean={np.mean(psnrs):.4f}   (min {np.min(psnrs):.4f}, max {np.max(psnrs):.4f})")
    print(f"  SSIM  mean={np.mean(ssims):.4f}   (min {np.min(ssims):.4f}, max {np.max(ssims):.4f})")
    if lpipss:
        print(f"  LPIPS mean={np.mean(lpipss):.4f}   (min {np.min(lpipss):.4f}, max {np.max(lpipss):.4f})")

    # montage: rows = views, cols = {render, gt}
    tiles = []
    for i in idxs:
        n = names[i]
        r = Image.open(os.path.join(rdir, n)).convert("RGB")
        g = Image.open(os.path.join(gdir, n)).convert("RGB")
        w, h = r.size
        s = args.scale
        r = r.resize((w * s, h * s), Image.LANCZOS)
        g = g.resize((w * s, h * s), Image.LANCZOS)
        row = Image.new("RGB", (r.width + g.width + 12, r.height), (30, 30, 30))
        row.paste(r, (0, 0))
        row.paste(g, (r.width + 12, 0))
        tiles.append(row)

    W = max(t.width for t in tiles)
    H = sum(t.height for t in tiles) + 10 * (len(tiles) - 1)
    sheet = Image.new("RGB", (W, H), (30, 30, 30))
    y = 0
    for t in tiles:
        sheet.paste(t, (0, y))
        y += t.height + 10

    out = args.out or os.path.join(WORK, "notes",
                                   f"montage_{args.split}_{args.iteration}.png")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    sheet.save(out)
    print(f"\nwrote montage {out}  ({sheet.width}x{sheet.height})")
    print("(left column = 3DGS render, right column = ground truth)")

    # also save one full-resolution single view for the deliverable
    single = args.single_out or os.path.join(
        WORK, "notes", f"render_{args.split}_{args.iteration}_view{idxs[0]:05d}.png")
    os.makedirs(os.path.dirname(single), exist_ok=True)
    Image.open(os.path.join(rdir, names[idxs[0]])).convert("RGB").resize(
        (Image.open(os.path.join(rdir, names[idxs[0]])).width * args.scale,
         Image.open(os.path.join(rdir, names[idxs[0]])).height * args.scale),
        Image.LANCZOS).save(single)
    print("wrote single view", single)


if __name__ == "__main__":
    main()
