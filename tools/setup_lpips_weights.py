"""
Make upstream metrics.py's LPIPS path work on a network where
download.pytorch.org is unreachable.

metrics.py calls lpipsPyTorch.lpips(..., net_type='vgg'), which needs two files
in the torch hub cache:

  1. vgg16-397923af.pth  -- torchvision VGG16(IMAGENET1K_V1) *features* weights.
     Normally fetched from download.pytorch.org (unreachable here).  We rebuild
     it from timm's vgg16.tv_in1k checkpoint via hf-mirror; both are the original
     Simonyan & Zisserman VGG16 weights, and timm keeps torchvision's
     `features.<idx>` key naming, so the features tensors map 1:1.
  2. vgg.pth -- the LPIPS linear calibration layers, from raw.githubusercontent.
     Fetched directly; torch.hub caches it under the same directory.

This is data preparation only: no repository file is modified.
"""

import os
import shutil
import sys
import urllib.request

import torch

UA = {"User-Agent": "Mozilla/5.0 (dsh-3dgs-cpu-port)"}
CKPT = os.path.join(torch.hub.get_dir(), "checkpoints")
os.makedirs(CKPT, exist_ok=True)
print("hub checkpoints dir:", CKPT)

TIMM_BIN = "https://hf-mirror.com/timm/vgg16.tv_in1k/resolve/main/pytorch_model.bin"
LPIPS_VGG = ("https://raw.githubusercontent.com/richzhang/PerceptualSimilarity/"
             "master/lpips/weights/v0.1/vgg.pth")


def fetch(url, dest, timeout=180):
    print(f"  GET {url}")
    tmp = dest + ".part"
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=timeout) as r, open(tmp, "wb") as f:
        shutil.copyfileobj(r, f, 1 << 16)
    os.replace(tmp, dest)
    print(f"  -> {dest}  ({os.path.getsize(dest)/1e6:.1f} MB)")


def build_vgg16():
    dest = os.path.join(CKPT, "vgg16-397923af.pth")
    if os.path.exists(dest):
        print("vgg16 already present:", dest)
        return True
    src = os.path.join(CKPT, "_timm_vgg16.bin")
    try:
        fetch(TIMM_BIN, src)
    except Exception as exc:                                   # noqa: BLE001
        print("  timm download failed:", exc)
        return False
    try:
        obj = torch.load(src, map_location="cpu", weights_only=False)
    except Exception as exc:                                   # noqa: BLE001
        print("  torch.load failed:", exc)
        return False

    for key in ("state_dict", "model", "model_state_dict"):
        if isinstance(obj, dict) and key in obj and isinstance(obj[key], dict):
            obj = obj[key]
            break

    feats = {k: v for k, v in obj.items() if k.startswith("features.")}
    if not feats:
        print("  no `features.*` tensors found; keys sample:", list(obj)[:8])
        return False

    torch.save(feats, dest)
    os.remove(src)
    print(f"  built {dest} with {len(feats)} feature tensors")

    # sanity check: it must load into torchvision's VGG16 features
    from torchvision import models
    net = models.vgg16(weights=None).features
    missing, unexpected = net.load_state_dict(feats, strict=False)
    print(f"  load check: missing={len(missing)} unexpected={len(unexpected)}")
    return len(unexpected) == 0


def build_lpips_linear():
    dest = os.path.join(CKPT, "vgg.pth")
    if os.path.exists(dest):
        print("lpips vgg.pth already present:", dest)
        return True
    try:
        fetch(LPIPS_VGG, dest)
        return True
    except Exception as exc:                                   # noqa: BLE001
        print("  lpips vgg.pth download failed:", exc)
        return False


ok1 = build_vgg16()
ok2 = build_lpips_linear()
print(f"\nRESULT vgg16_features={ok1} lpips_linear={ok2}")
sys.exit(0 if (ok1 and ok2) else 1)
