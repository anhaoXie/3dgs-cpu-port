"""
Report the per-tile candidate statistics the rasterizer actually sees, and the
per-tile cap it derives from them, for one checkpoint.

Why this matters: the rasterizer's effective cap is

    kmax = min(K_MAX, counts_max, BUDGET // (n_tiles * 256))

`counts_max` is the largest number of candidate gaussians landing in any single
16x16 tile.  That number is a property of the *scene*, not of the cap -- so if it
is below K_MAX the cap never binds and the render is identical to an uncapped one.
This is exactly what decides whether raising K_MAX can change anything.

It is also what explains why the old K_MAX=256 conclusion was wrong: `counts_max`
grows as training splits gaussians and the scene gets denser, so a cap tuned on
an early checkpoint quietly starts truncating later.  This probe measures it
instead of inferring it.

Note: counts_max does NOT depend on DSH_3DGS_KMAX, so a single run reports the
full picture for every candidate cap.

Usage:
  python tools/raster_cap_probe.py --model output/truck_full --iteration 15000
  python tools/raster_cap_probe.py --model output/truck_full --iteration 7000 --split test
"""

import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
WORK = os.path.dirname(HERE)
REPO = os.path.join(WORK, "gaussian-splatting")

sys.path.insert(0, os.path.join(WORK, "cpu_port"))
sys.path.insert(0, REPO)

# The rasterizer reads this once at import time, so it has to be set before
# anything pulls diff_gaussian_rasterization in.
os.environ.setdefault("DSH_3DGS_DEBUG", "1")

try:
    import sitecustomize               # noqa: F401  CUDA->CPU shim
except ImportError:
    pass

import torch                                                    # noqa: E402
from arguments import ModelParams, PipelineParams, get_combined_args  # noqa: E402
from scene import Scene, GaussianModel                          # noqa: E402
from gaussian_renderer import render                            # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    mp = ModelParams(ap, sentinel=True)
    pp = PipelineParams(ap)            # register once; re-registering conflicts
    ap.add_argument("--iteration", type=int, required=True)
    ap.add_argument("--split", choices=["train", "test"], default="train")
    ap.add_argument("--view", type=int, default=0)
    ap.add_argument("--save-image", default=None,
                    help="also write the rendered view to this PNG, for seam_metric.py")
    args = get_combined_args(ap)

    dataset = mp.extract(args)
    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians, load_iteration=args.iteration, shuffle=False)

    cams = scene.getTrainCameras() if args.split == "train" else scene.getTestCameras()
    if not cams:
        raise SystemExit(f"no {args.split} cameras in {args.model_path}")
    cam = cams[min(args.view, len(cams) - 1)]

    bg = torch.tensor([1.0, 1.0, 1.0] if dataset.white_background else [0.0, 0.0, 0.0],
                      dtype=torch.float32)

    kmax_raw = os.environ.get("DSH_3DGS_KMAX", "128").strip()
    budget = int(os.environ.get("DSH_3DGS_BUDGET", "6000000"))

    print(f"\n--- checkpoint {args.iteration}, {args.split} camera #{args.view} ---")
    print(f"    gaussians in model      : {gaussians.get_xyz.shape[0]:,}")
    print(f"    DSH_3DGS_KMAX           : {kmax_raw}")
    if kmax_raw.lower() == "auto":
        print(f"    auto memory budget      : {os.environ.get('DSH_3DGS_KMAX_MEM_MB', '2048')} MiB"
              f" x{os.environ.get('DSH_3DGS_KMAX_TENSORS', '8')} live tensors,"
              f" ceiling {os.environ.get('DSH_3DGS_KMAX_CEILING', '16384')}")
    print(f"    DSH_3DGS_BUDGET         : {budget:,}")
    # The rasterizer's own line follows; it prints counts_max and the derived kmax.
    with torch.no_grad():
        out = render(cam, gaussians, pp.extract(args), bg)

    if args.save_image:
        import numpy as np
        from PIL import Image
        # render() clamps to [0, 1] (gaussian_renderer/__init__.py:119)
        arr = (out["render"].clamp(0.0, 1.0).mul(255.0)
               .byte().permute(1, 2, 0).cpu().numpy())
        os.makedirs(os.path.dirname(os.path.abspath(args.save_image)), exist_ok=True)
        Image.fromarray(arr).save(args.save_image)
        print(f"    wrote                   : {args.save_image}")


if __name__ == "__main__":
    main()
