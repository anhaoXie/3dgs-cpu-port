"""Profile the CPU rasterizer's phases vs Gaussian count and image size."""
import math
import os
import sys
import time

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
WORK = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(WORK, "cpu_port"))

import diff_gaussian_rasterization as dgr          # noqa: E402
from diff_gaussian_rasterization import (          # noqa: E402
    GaussianRasterizationSettings, GaussianRasterizer)

dgr._PROFILE = True
dgr._KMAX_DEFAULT = int(os.environ.get("DSH_3DGS_KMAX", "256"))
dgr._BUDGET_DEFAULT = int(os.environ.get("DSH_3DGS_BUDGET", "16000000"))


def bench(N, W, H, extent=8.0, iters=6):
    torch.manual_seed(0)
    means3D = ((torch.rand(N, 3) - 0.5) * extent).requires_grad_(True)
    means3D = torch.nn.Parameter(means3D + torch.tensor([0.0, 0.0, 12.0]))
    scales = torch.nn.Parameter(torch.rand(N, 3) * 0.06 + 0.01)
    rotations = torch.nn.Parameter(torch.randn(N, 4) / 4.0)
    opacities = torch.nn.Parameter(torch.rand(N, 1) * 0.6 + 0.1)
    shs = torch.nn.Parameter(torch.randn(N, 16, 3) * 0.2)

    tan = math.tan(0.5 * 1.0)
    pm = torch.zeros(4, 4)
    pm[0, 0] = 1.0 / tan
    pm[1, 1] = 1.0 / tan
    pm[2, 2] = 1.0
    pm[2, 3] = 1.0
    rs = GaussianRasterizationSettings(H, W, tan, tan, torch.zeros(3), 1.0,
                                      torch.eye(4), pm, 3, torch.zeros(3),
                                      False, False, False)
    rast = GaussianRasterizer(raster_settings=rs)

    for k in dgr._PROF:
        dgr._PROF[k] = 0.0

    # warmup
    m2 = torch.zeros_like(means3D)
    img, radii, _ = rast(means3D=means3D, means2D=m2, shs=shs, opacities=opacities,
                         scales=scales, rotations=rotations)
    img.sum().backward()
    for p in (means3D, scales, rotations, opacities, shs):
        p.grad = None
    for k in dgr._PROF:
        dgr._PROF[k] = 0.0

    t0 = time.perf_counter()
    for _ in range(iters):
        m2 = torch.zeros_like(means3D)
        img, radii, _ = rast(means3D=means3D, means2D=m2, shs=shs,
                             opacities=opacities, scales=scales, rotations=rotations)
        img.sum().backward()
        for p in (means3D, scales, rotations, opacities, shs):
            p.grad = None
    fwd_bwd = (time.perf_counter() - t0) / iters

    # forward only
    for k in dgr._PROF:
        dgr._PROF[k] = 0.0
    t0 = time.perf_counter()
    with torch.no_grad():
        for _ in range(iters):
            m2 = torch.zeros_like(means3D)
            rast(means3D=means3D, means2D=m2, shs=shs, opacities=opacities,
                 scales=scales, rotations=rotations)
    fwd = (time.perf_counter() - t0) / iters

    n_calls = max(1.0, dgr._PROF["calls"])
    vis = int((radii > 0).sum())
    print(f"N={N:7d} {W}x{H}  visible={vis:7d}  fwd={fwd*1000:7.1f} ms  "
          f"fwd+bwd={fwd_bwd*1000:7.1f} ms  bwd={(fwd_bwd-fwd)*1000:7.1f} ms")
    print(f"           fwd phases x{iters}/calls: "
          f"preprocess={dgr._PROF['preprocess']/n_calls*1000:6.1f} ms  "
          f"colour={dgr._PROF['colour']/n_calls*1000:6.1f} ms  "
          f"binning={dgr._PROF['binning']/n_calls*1000:6.1f} ms  "
          f"composite={dgr._PROF['composite']/n_calls*1000:6.1f} ms")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--ns", type=int, nargs="*", default=[136029, 300000, 600000])
    ap.add_argument("--w", type=int, default=244)
    ap.add_argument("--h", type=int, default=136)
    a = ap.parse_args()
    print(f"torch {torch.__version__}  threads={torch.get_num_threads()}  "
          f"K_MAX={dgr._KMAX_DEFAULT}  BUDGET={dgr._BUDGET_DEFAULT}")
    for n in a.ns:
        bench(n, a.w, a.h)
