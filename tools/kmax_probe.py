"""
Probe how the per-tile candidate cap (K_MAX) affects render quality.

Renders the same view of an already-trained model at several K_MAX values and
writes a labelled comparison strip, so we can tell whether the rectangular tile
artifacts come from candidate truncation or are simply an under-trained model.

Usage:
  python kmax_probe.py --model <model_path> --iteration N [--view 0] [--kmax 32 64 128 256 1024]
"""

import argparse
import os
import sys

import numpy as np
import torch
from PIL import Image, ImageDraw

import diff_gaussian_rasterization as dgr

HERE = os.path.dirname(os.path.abspath(__file__))
WORK = os.path.dirname(HERE)
sys.path.insert(0, os.path.dirname(WORK))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--iteration", type=int, required=True)
    ap.add_argument("--view", type=int, default=0)
    ap.add_argument("--kmax", type=int, nargs="*", default=[32, 64, 128, 256, 1024])
    ap.add_argument("--scale", type=int, default=5)
    args = ap.parse_args()

    repo = os.path.join(os.path.dirname(HERE), "gaussian-splatting")
    sys.path.insert(0, repo)
    from argparse import ArgumentParser                           # noqa: E402
    from arguments import ModelParams, PipelineParams             # noqa: E402
    from scene import Scene, GaussianModel                        # noqa: E402
    from gaussian_renderer import render                          # noqa: E402

    ap2 = ArgumentParser()
    mp = ModelParams(ap2)
    pp = PipelineParams(ap2)
    ns = ap2.parse_args(["-s", os.path.join(WORK, "data", "truck"),
                         "-m", args.model, "--data_device", "cpu", "-r", "8"])
    ds = mp.extract(ns)
    pipe = pp.extract(ns)

    g = GaussianModel(ds.sh_degree)
    scene = Scene(ds, g, load_iteration=args.iteration, shuffle=False)
    cams = scene.getTrainCameras()
    cam = cams[args.view]
    print(f"model={args.model} iter={args.iteration} view={args.view} "
          f"({cam.image_name} {cam.image_width}x{cam.image_height})")
    print(f"gaussians={g.get_xyz.shape[0]}")

    bg = torch.zeros(3)
    tiles = []
    for k in args.kmax:
        dgr._KMAX_DEFAULT = k
        with torch.no_grad():
            out = render(cam, g, pipe, bg)
        img = out["render"].clamp(0, 1)
        vis = (img.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        print(f"  K_MAX={k:5d}  radii>0={(out['radii'] > 0).sum().item():6d}  "
              f"mean={float(img.mean()):.4f}")
        s = args.scale
        pil = Image.fromarray(vis).resize(
            (cam.image_width * s, cam.image_height * s), Image.NEAREST)
        canvas = Image.new("RGB", (pil.width, pil.height + 26), (20, 20, 20))
        canvas.paste(pil, (0, 26))
        ImageDraw.Draw(canvas).text((6, 6), f"K_MAX = {k}", fill=(255, 255, 0))
        tiles.append(canvas)

    # ground truth for reference
    gt = (cam.original_image[:, :, :].clamp(0, 1).permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    s = args.scale
    gt_pil = Image.fromarray(gt).resize((cam.image_width * s, cam.image_height * s),
                                       Image.NEAREST)
    canvas = Image.new("RGB", (gt_pil.width, gt_pil.height + 26), (20, 20, 20))
    canvas.paste(gt_pil, (0, 26))
    ImageDraw.Draw(canvas).text((6, 6), "GROUND TRUTH", fill=(0, 255, 0))
    tiles.append(canvas)

    W = sum(t.width for t in tiles) + 8 * (len(tiles) - 1)
    H = max(t.height for t in tiles)
    sheet = Image.new("RGB", (W, H), (20, 20, 20))
    x = 0
    for t in tiles:
        sheet.paste(t, (x, 0))
        x += t.width + 8

    out_png = os.path.join(WORK, "notes", f"kmax_probe_{args.iteration}.png")
    sheet.save(out_png)
    print("wrote", out_png, f"({sheet.width}x{sheet.height})")


if __name__ == "__main__":
    main()
