"""Diagnose gradcheck failures: report where autograd and finite differences disagree."""
import math
import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
CPU_PORT = os.path.dirname(HERE)
REPO = None
for cand in (os.path.dirname(CPU_PORT),
             os.path.join(os.path.dirname(CPU_PORT), "gaussian-splatting"),
             os.getcwd()):
    if os.path.exists(os.path.join(cand, "utils", "sh_utils.py")):
        REPO = cand
        break
sys.path.insert(0, CPU_PORT)
sys.path.insert(0, REPO or "")

from diff_gaussian_rasterization import (          # noqa: E402
    GaussianRasterizationSettings, GaussianRasterizer)

dtype = torch.float64
H = W = 32
N = 6
torch.manual_seed(3)

means3D = ((torch.rand(N, 3, dtype=dtype) - 0.5) * 2.0) + torch.tensor([0.0, 0.0, 5.0], dtype=dtype)
means3D.requires_grad_(True)
scales = (torch.rand(N, 3, dtype=dtype) * 0.3 + 0.1).requires_grad_(True)
rotations = (torch.randn(N, 4, dtype=dtype) / 2.0).requires_grad_(True)
opacities = (torch.rand(N, 1, dtype=dtype) * 0.7 + 0.2).requires_grad_(True)
shs = (torch.randn(N, 16, 3, dtype=dtype) * 0.2).requires_grad_(True)

tanfovx = tanfovy = math.tan(0.5)
viewmatrix = torch.eye(4, dtype=dtype)
pm = torch.zeros(4, 4, dtype=dtype)
pm[0, 0] = 1.0 / tanfovx
pm[1, 1] = 1.0 / tanfovy
pm[2, 2] = 1.0
pm[2, 3] = 1.0
bg = torch.tensor([0.5, 0.5, 0.5], dtype=dtype)
rs = GaussianRasterizationSettings(H, W, tanfovx, tanfovy, bg, 1.0,
                                   viewmatrix, pm, 3, torch.zeros(3, dtype=dtype),
                                   False, False, False)


def fn(means, sc, rot, opa, sh):
    m2 = torch.zeros_like(means)
    img, _, _ = GaussianRasterizer(raster_settings=rs)(
        means3D=means, means2D=m2, shs=sh, opacities=opa, scales=sc, rotations=rot)
    return img


inputs = [means3D, scales, rotations, opacities, shs]
names = ["means3D", "scales", "rotations", "opacities", "shs"]

print("=== gradcheck detail (raise_exception) ===")
try:
    torch.autograd.gradcheck(fn, inputs, eps=1e-6, atol=1e-4, rtol=1e-2,
                             raise_exception=True, check_batched_grad=False)
    print("gradcheck PASSED")
except Exception as exc:
    print(str(exc)[:3000])

print("\n=== finite-difference vs autograd, per input ===")
out = fn(*inputs)
out.sum().backward()
for name, inp in zip(names, inputs):
    ana = inp.grad.detach().clone()
    num = torch.zeros_like(ana)
    eps = 1e-6
    flat = inp.detach().reshape(-1)
    numflat = num.reshape(-1)
    for i in range(flat.numel()):
        orig = flat[i].item()
        flat[i] = orig + eps
        lp = fn(*inputs).sum().item()
        flat[i] = orig - eps
        lm = fn(*inputs).sum().item()
        flat[i] = orig
        numflat[i] = (lp - lm) / (2 * eps)
    for p in inputs:
        p.grad = None
    out = fn(*inputs)
    out.sum().backward()
    diff = (ana - num).abs()
    denom = num.abs().clamp_min(1e-8)
    rel = diff / denom
    print(f"{name:10s} max|ana-num|={diff.max().item():.3e}  "
          f"max rel={rel.max().item():.3e}  mean|num|={num.abs().mean().item():.3e}  "
          f"n_bad(atol1e-4,rtol1e-2)={(diff > 1e-4 + 1e-2*num.abs()).sum().item()}/{num.numel()}")
