"""
Integration smoke test: drive the OFFICIAL upstream scene loader and renderer
with the CPU rasterizer, on the real Tanks & Temples `truck` COLMAP scene.

This exercises, end to end and without touching upstream code:
  scene/dataset_readers.py  -> COLMAP .bin parsing
  scene/gaussian_model.py   -> create_from_pcd + distCUDA2 (our CPU simple_knn)
  gaussian_renderer/__init__.py -> the real render() call, incl. `device="cuda"`
  utils/loss_utils.py       -> L1 + (pure-torch) SSIM loss

Run from inside the gaussian-splatting checkout with PYTHONPATH=<cpu_port>.
"""

import os
import sys
import time

import torch

WORK = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def find_repo():
    for cand in (os.path.dirname(WORK),
                 os.path.join(os.path.dirname(WORK), "gaussian-splatting"),
                 os.getcwd()):
        if os.path.exists(os.path.join(cand, "train.py")):
            return cand
    raise SystemExit("cannot locate the upstream checkout")


REPO = find_repo()
sys.path.insert(0, REPO)

from argparse import ArgumentParser                     # noqa: E402
from arguments import ModelParams, PipelineParams       # noqa: E402
from scene import Scene, GaussianModel                  # noqa: E402
from gaussian_renderer import render                    # noqa: E402
from utils.loss_utils import l1_loss, ssim              # noqa: E402


def main():
    ap = ArgumentParser()
    mp = ModelParams(ap)
    pp = PipelineParams(ap)
    ap.add_argument("--out", default=os.path.join(WORK, "output", "_scenetest"))
    args = ap.parse_args([
        "-s", os.path.join(WORK, "data", "truck"),
        "-m", os.path.join(WORK, "output", "_scenetest"),
        "--data_device", "cpu",
        "-r", "8",
    ])
    ds = mp.extract(args)
    pipe = pp.extract(args)
    # train.py normally creates this in prepare_output_and_logger(); Scene()
    # writes input.ply / cameras.json into it.
    os.makedirs(ds.model_path, exist_ok=True)

    print("=" * 70)
    print("torch", torch.__version__, "| cuda_available", torch.cuda.is_available())
    print("source_path", ds.source_path)
    print("resolution factor", ds.resolution, "| data_device", ds.data_device)
    print("=" * 70)

    t0 = time.time()
    gaussians = GaussianModel(ds.sh_degree)
    scene = Scene(ds, gaussians)
    print(f"scene loaded in {time.time()-t0:.1f}s")

    train = scene.getTrainCameras()
    test = scene.getTestCameras()
    print(f"train cameras: {len(train)}   test cameras: {len(test)}")
    if not train:
        raise SystemExit("FAIL: no training cameras -- COLMAP parsing problem")

    cam = train[0]
    print(f"camera image  : {cam.image_name}  {cam.image_width}x{cam.image_height}")
    print(f"FoVx/FoVy     : {cam.FoVx:.4f} / {cam.FoVy:.4f}")
    print(f"gaussians     : {gaussians.get_xyz.shape[0]}  (from SfM points3D)")
    print(f"cameras_extent: {scene.cameras_extent:.4f}")
    scales = gaussians.get_scaling.detach()
    print(f"init scales   : min {scales.min():.3e}  max {scales.max():.3e}")

    bg = torch.zeros(3)
    t0 = time.time()
    out = render(cam, gaussians, pipe, bg)
    dt = time.time() - t0
    img = out["render"]
    radii = out["radii"]
    print(f"\nrender ok in {dt*1000:.0f} ms")
    print(f"  image   {tuple(img.shape)}  range [{img.min():.4f}, {img.max():.4f}]  "
          f"mean {img.mean():.4f}")
    print(f"  radii>0 : {(radii > 0).sum().item()}/{radii.numel()}")
    print(f"  vsp grad: {out['viewspace_points'].grad}")

    gt = cam.original_image
    print(f"  gt      {tuple(gt.shape)}  mean {gt.mean():.4f}")
    Ll1 = l1_loss(img, gt)
    sv = ssim(img, gt)
    loss = 0.8 * Ll1 + 0.2 * (1.0 - sv)
    print(f"  L1={Ll1.item():.4f}  SSIM={sv.item():.4f}  loss={loss.item():.4f}")

    # backward must reach every parameter group that training optimises
    t0 = time.time()
    loss.backward()
    print(f"\nbackward ok in {time.time()-t0:.1f}s")
    for name in ("_xyz", "_features_dc", "_features_rest", "_opacity",
                 "_scaling", "_rotation"):
        p = getattr(gaussians, name)
        g = p.grad
        print(f"  grad {name:15s} shape {tuple(p.shape)}  "
              f"{'None' if g is None else f'|g|sum={float(g.abs().sum()):.4e}'}")
    vg = out["viewspace_points"].grad
    print(f"  grad viewspace_points (densification) "
          f"{'None' if vg is None else f'|g[:, :2]|sum={float(vg[:, :2].abs().sum()):.4e}'}")

    print("\nSCENE + RENDER + BACKWARD OK")


if __name__ == "__main__":
    main()
