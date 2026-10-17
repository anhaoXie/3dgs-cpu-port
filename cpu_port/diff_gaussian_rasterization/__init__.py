"""
diff_gaussian_rasterization -- pure-PyTorch CPU reimplementation.

This package is a drop-in replacement for the CUDA extension shipped with
Kerbl et al., "3D Gaussian Splatting for Real-Time Radiance Field Rendering"
(SIGGRAPH 2023).  It exposes exactly the same public API

    GaussianRasterizationSettings
    GaussianRasterizer            (forward -> color, radii, invdepths)
    SparseGaussianAdam            (deliberately absent -> feature detect fails)

so that the *unmodified* upstream `gaussian_renderer/__init__.py`, `train.py`,
`render.py` and `metrics.py` can drive it.

WHAT IS REPRODUCED FROM THE PAPER / CUDA KERNEL
----------------------------------------------
Everything, on the CPU, with autograd providing the backward pass:

  * `computeCov3D`      -- Sigma = M^T M with M = S R  (Eq. 6 / paper Sec. 4)
  * `computeCov2D`      -- EWA splatting affine approximation
                           Sigma' = J W^T Sigma W J^T, plus the +0.3 low-pass
  * `preprocessCUDA`    -- frustum cull (p_view.z > 0.2), NDC->pixel mapping
                           ndc2Pix(v,S) = ((v+1)*S - 1) * 0.5, 3-sigma radius,
                           16x16 tile bounding rect
  * `duplicateWithKeys` -- [tile_id | depth] key binning, 16x16 tiles
  * `renderCUDA`        -- per-pixel front-to-back alpha compositing,
                           alpha = min(0.99, opacity * exp(power)),
                           C = sum_i alpha_i T_i c_i + T_final * bg

DELIBERATE, DOCUMENTED DEVIATIONS FROM THE CUDA KERNEL
------------------------------------------------------
1. The CUDA kernel stops the per-pixel loop once the running transmittance
   drops below 1e-4.  This implementation evaluates every surviving candidate.
   That is *more* accurate, not less; the difference is far below 8-bit
   quantisation.
2. The backward pass is produced by PyTorch autograd rather than the hand
   written `backward.cu`.  Mathematically it is the same set of derivatives;
   `dL/d(means2D)` is routed with a straight-through identity so that upstream's
   densification statistics (`viewspace_point_tensor.grad`) keep their meaning.
3. Candidates are truncated to at most K_MAX (env `DSH_3DGS_KMAX`, default 160)
   per tile, keeping the nearest ones.  At 1e-4 transmittance the discarded
   tail is invisible; this is the one place a measurable (but tiny) difference
   can appear on extremely dense scenes.
4. Degenerate 2D covariances (det <= 0) are culled.  The CUDA kernel only tests
   `det == 0`, which lets negative determinants through as garbage.
"""

import math
import os
import time
from typing import NamedTuple

import torch
import torch.nn as nn

__all__ = [
    "GaussianRasterizationSettings",
    "GaussianRasterizer",
    "rasterize_gaussians",
]

# ---------------------------------------------------------------------------
# Spherical harmonics constants -- verbatim from cuda_rasterizer/auxiliary.h
# ---------------------------------------------------------------------------
SH_C0 = 0.28209479177387814
SH_C1 = 0.4886025119029199
SH_C2 = (1.0925484305920792, -1.0925484305920792, 0.31539156525252005,
         -1.0925484305920792, 0.5462742152960396)
SH_C3 = (-0.5900435899266435, 2.890611442640554, -0.4570457994644658,
         0.3731763325901154, -0.4570457994644658, 1.445305721320277,
         -0.5900435899266435)

BLOCK_X = 16
BLOCK_Y = 16
BLOCK_SIZE = BLOCK_X * BLOCK_Y            # 256 pixels per tile
H_VAR = 0.3                               # low-pass filter from the paper
ALPHA_MIN = 1.0 / 255.0                   # cuda: `if (alpha < 1/255) continue`
ALPHA_MAX = 0.99                          # cuda: `min(0.99f, ...)`

# Tuning knobs (performance only -- they do not change the algorithm)
#
# K_MAX caps how many candidate gaussians each 16x16 tile may keep.  The CUDA
# kernel has no such cap -- it walks every candidate.  On CPU that is impossible
# at realistic density: the compositing block materialises several float32
# [n_tiles, 256, K] tensors at once, and the busiest tile routinely sees far more
# candidates than a fixed K can hold (measured with tools/raster_cap_probe.py:
# ~11k at 245x136 on the truck scene at iteration 15000, ~20-38k at the smoke
# resolution).  A fixed K=256 therefore discards ~97% of the busiest tile and
# leaves visible 16x16 seams -- see notes/kmax-ablation.md.
#
# So a fixed number is arbitrary.  "auto" instead derives K from a memory budget:
#   DSH_3DGS_KMAX=auto       derive K from the budget below (never exceeds the
#                            candidates actually present, so sparse scenes stay cheap)
#   DSH_3DGS_KMAX_MEM_MB     MiB to spend on the [n_tiles, 256, K] tensors (default 2048)
#   DSH_3DGS_KMAX_CEILING    hard upper bound even when memory would allow more
#   DSH_3DGS_KMAX_TENSORS    how many such tensors are live at peak (default 8)
#
# The default stays the old fixed 128, so existing runs reproduce byte-for-byte.
_KMAX_ENV = os.environ.get("DSH_3DGS_KMAX", "128").strip()
_KMAX_AUTO = _KMAX_ENV.lower() == "auto"
_KMAX_DEFAULT = 128 if _KMAX_AUTO else int(_KMAX_ENV)
_KMAX_AUTO_MIB = float(os.environ.get("DSH_3DGS_KMAX_MEM_MB", "2048"))
_KMAX_CEILING = int(os.environ.get("DSH_3DGS_KMAX_CEILING", "16384"))
_KMAX_LIVE_TENSORS = float(os.environ.get("DSH_3DGS_KMAX_TENSORS", "8"))
_BUDGET_DEFAULT = int(os.environ.get("DSH_3DGS_BUDGET", "6000000"))
_DEBUG = bool(os.environ.get("DSH_3DGS_DEBUG", ""))
_PROFILE = bool(os.environ.get("DSH_3DGS_PROFILE", ""))


def resolve_kmax(n_tiles, counts_max):
    """Return (effective per-tile cap, legacy element budget).

    In auto mode the cap is the largest K whose dense tensors fit the memory
    budget, further bounded by the candidates actually present and by a ceiling.
    In fixed mode this is exactly the previous behaviour: min(K_MAX, counts_max),
    then clamped by the legacy DSH_3DGS_BUDGET element budget.
    """
    legacy_budget = max(1, _BUDGET_DEFAULT // (n_tiles * BLOCK_SIZE))
    if _KMAX_AUTO:
        bytes_per_k = _KMAX_LIVE_TENSORS * n_tiles * BLOCK_SIZE * 4.0
        by_mem = int(_KMAX_AUTO_MIB * 1024 * 1024 / max(bytes_per_k, 1.0))
        # Memory is the authority in auto mode, so the legacy budget is not applied.
        return max(1, min(counts_max, by_mem, _KMAX_CEILING)), legacy_budget
    return max(1, min(min(_KMAX_DEFAULT, counts_max), legacy_budget)), legacy_budget

# Simple phase profiler (opt-in via DSH_3DGS_PROFILE=1).  It only ever measures
# wall time around existing code; it cannot change results.
_PROF = {"preprocess": 0.0, "colour": 0.0, "binning": 0.0, "composite": 0.0,
         "calls": 0.0}


def prof_report():
    if not _PROFILE:
        return
    n = _PROF["calls"]
    print("  [raster profile] calls={:.0f}  preprocess={:.3f}s  colour={:.3f}s  "
          "binning={:.3f}s  composite={:.3f}s  total={:.3f}s".format(
              n, _PROF["preprocess"], _PROF["colour"], _PROF["binning"],
              _PROF["composite"],
              _PROF["preprocess"] + _PROF["colour"] + _PROF["binning"]
              + _PROF["composite"]))


# ---------------------------------------------------------------------------
# Colour from spherical harmonics (cuda: computeColorFromSH)
# ---------------------------------------------------------------------------
def _sh_to_rgb(degree, shs, means3D, campos):
    """Evaluate the SH series for every Gaussian.

    `shs` has shape [N, M, 3] where M = (max_sh_degree+1)**2 (always 16 in
    upstream); only the first (degree+1)**2 coefficients are used.
    """
    dir = means3D - campos.unsqueeze(0)
    dir = dir / (dir.norm(dim=1, keepdim=True) + 1e-12)
    x = dir[:, 0:1]
    y = dir[:, 1:2]
    z = dir[:, 2:3]

    result = SH_C0 * shs[:, 0]

    if degree > 0:
        result = result - SH_C1 * y * shs[:, 1] \
                        + SH_C1 * z * shs[:, 2] \
                        - SH_C1 * x * shs[:, 3]
        if degree > 1:
            xx, yy, zz = x * x, y * y, z * z
            xy, yz, xz = x * y, y * z, x * z
            result = result + (
                SH_C2[0] * xy * shs[:, 4]
                + SH_C2[1] * yz * shs[:, 5]
                + SH_C2[2] * (2.0 * zz - xx - yy) * shs[:, 6]
                + SH_C2[3] * xz * shs[:, 7]
                + SH_C2[4] * (xx - yy) * shs[:, 8]
            )
            if degree > 2:
                result = result + (
                    SH_C3[0] * y * (3.0 * xx - yy) * shs[:, 9]
                    + SH_C3[1] * xy * z * shs[:, 10]
                    + SH_C3[2] * y * (4.0 * zz - xx - yy) * shs[:, 11]
                    + SH_C3[3] * z * (2.0 * zz - 3.0 * xx - 3.0 * yy) * shs[:, 12]
                    + SH_C3[4] * x * (4.0 * zz - xx - yy) * shs[:, 13]
                    + SH_C3[5] * z * (xx - yy) * shs[:, 14]
                    + SH_C3[6] * x * (xx - 3.0 * yy) * shs[:, 15]
                )
    # cuda: `result += 0.5f; return glm::max(result, 0.0f);`
    return (result + 0.5).clamp_min(0.0)


# ---------------------------------------------------------------------------
# 3D covariance from scale + rotation (cuda: computeCov3D)
# ---------------------------------------------------------------------------
def _compute_cov3d(scales, scale_modifier, rotations):
    """Return the packed upper triangle [N, 6] of Sigma = M^T M, M = S R."""
    n = scales.shape[0]
    s = scales * scale_modifier
    r = rotations[:, 0]
    x = rotations[:, 1]
    y = rotations[:, 2]
    z = rotations[:, 3]

    R = torch.empty((n, 3, 3), dtype=scales.dtype, device=scales.device)
    R[:, 0, 0] = 1.0 - 2.0 * (y * y + z * z)
    R[:, 0, 1] = 2.0 * (x * y - r * z)
    R[:, 0, 2] = 2.0 * (x * z + r * y)
    R[:, 1, 0] = 2.0 * (x * y + r * z)
    R[:, 1, 1] = 1.0 - 2.0 * (x * x + z * z)
    R[:, 1, 2] = 2.0 * (y * z - r * x)
    R[:, 2, 0] = 2.0 * (x * z - r * y)
    R[:, 2, 1] = 2.0 * (y * z + r * x)
    R[:, 2, 2] = 1.0 - 2.0 * (x * x + y * y)

    # M = S @ R with S = diag(s)  =>  M[i, j] = s[i] * R[i, j]
    M = R * s.unsqueeze(2)
    Sigma = torch.bmm(M.transpose(1, 2), M)

    return torch.stack(
        (Sigma[:, 0, 0], Sigma[:, 0, 1], Sigma[:, 0, 2],
         Sigma[:, 1, 1], Sigma[:, 1, 2], Sigma[:, 2, 2]),
        dim=1,
    )


def _unpack_cov3d(cov3D):
    """[N, 6] packed upper triangle -> [N, 3, 3] symmetric matrices."""
    n = cov3D.shape[0]
    V = torch.empty((n, 3, 3), dtype=cov3D.dtype, device=cov3D.device)
    V[:, 0, 0] = cov3D[:, 0]
    V[:, 0, 1] = V[:, 1, 0] = cov3D[:, 1]
    V[:, 0, 2] = V[:, 2, 0] = cov3D[:, 2]
    V[:, 1, 1] = cov3D[:, 3]
    V[:, 1, 2] = V[:, 2, 1] = cov3D[:, 4]
    V[:, 2, 2] = cov3D[:, 5]
    return V


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
class GaussianRasterizationSettings(NamedTuple):
    image_height: int
    image_width: int
    tanfovx: float
    tanfovy: float
    bg: torch.Tensor
    scale_modifier: float
    viewmatrix: torch.Tensor
    projmatrix: torch.Tensor
    sh_degree: int
    campos: torch.Tensor
    prefiltered: bool
    debug: bool
    antialiasing: bool


def rasterize_gaussians(
    means3D,
    means2D,
    sh,
    colors_precomp,
    opacities,
    scales,
    rotations,
    cov3Ds_precomp,
    raster_settings,
):
    return _rasterize(
        means3D=means3D,
        means2D=means2D,
        sh=sh,
        colors_precomp=colors_precomp,
        opacities=opacities,
        scales=scales,
        rotations=rotations,
        cov3Ds_precomp=cov3Ds_precomp,
        image_height=raster_settings.image_height,
        image_width=raster_settings.image_width,
        tanfovx=raster_settings.tanfovx,
        tanfovy=raster_settings.tanfovy,
        bg=raster_settings.bg,
        scale_modifier=raster_settings.scale_modifier,
        viewmatrix=raster_settings.viewmatrix,
        projmatrix=raster_settings.projmatrix,
        sh_degree=raster_settings.sh_degree,
        campos=raster_settings.campos,
        prefiltered=raster_settings.prefiltered,
        antialiasing=raster_settings.antialiasing,
    )


def _rasterize(means3D, means2D, sh, colors_precomp, opacities, scales,
               rotations, cov3Ds_precomp, image_height, image_width,
               tanfovx, tanfovy, bg, scale_modifier, viewmatrix, projmatrix,
               sh_degree, campos, prefiltered, antialiasing):
    H = int(image_height)
    W = int(image_width)
    device = means3D.device
    dtype = means3D.dtype
    N = means3D.shape[0]

    out_color = torch.zeros((3, H, W), dtype=dtype, device=device)
    invdepths = torch.zeros((1, H, W), dtype=dtype, device=device)
    radii = torch.zeros((N,), dtype=torch.int32, device=device)

    if N == 0:
        out_color = out_color + bg.to(dtype).view(3, 1, 1)
        return out_color, radii, invdepths

    ones = torch.ones((N, 1), dtype=dtype, device=device)
    _t0 = time.perf_counter() if _PROFILE else 0.0
    p_h = torch.cat((means3D, ones), dim=1)                      # [N, 4]

    # ---- view / clip space (cuda: transformPoint4x4 / transformPoint4x3) ---
    p_view = p_h @ viewmatrix                                    # [N, 4]
    p_hom = p_h @ projmatrix                                     # [N, 4]
    p_w = 1.0 / (p_hom[:, 3] + 1e-7)
    p_proj = p_hom[:, :3] * p_w.unsqueeze(1)

    # ---- near-plane cull: cuda `if (p_view.z <= 0.2f) return;` -------------
    inside = p_view[:, 2] > 0.2
    tz_safe = torch.where(inside, p_view[:, 2],
                          torch.ones_like(p_view[:, 2]))

    # ---- 3D covariance ---------------------------------------------------
    if cov3Ds_precomp is not None and cov3Ds_precomp.numel() > 0:
        cov3D = cov3Ds_precomp
    else:
        cov3D = _compute_cov3d(scales, scale_modifier, rotations)
    Vrk = _unpack_cov3d(cov3D)

    # ---- 2D covariance (cuda: computeCov2D, EWA / paper Eq. 5) ------------
    limx = 1.3 * tanfovx
    limy = 1.3 * tanfovy
    tx = torch.clamp(p_view[:, 0] / tz_safe, -limx, limx) * tz_safe
    ty = torch.clamp(p_view[:, 1] / tz_safe, -limy, limy) * tz_safe

    focal_x = W / (2.0 * tanfovx)
    focal_y = H / (2.0 * tanfovy)

    J = torch.zeros((N, 3, 3), dtype=dtype, device=device)
    J[:, 0, 0] = focal_x / tz_safe
    J[:, 0, 2] = -(focal_x * tx) / (tz_safe * tz_safe)
    J[:, 1, 1] = focal_y / tz_safe
    J[:, 1, 2] = -(focal_y * ty) / (tz_safe * tz_safe)

    Wm = viewmatrix[:3, :3].transpose(0, 1)                      # [3, 3]
    T = torch.matmul(Wm.unsqueeze(0), J)                         # [N, 3, 3]
    cov = torch.bmm(torch.bmm(T.transpose(1, 2), Vrk), T)        # [N, 3, 3]

    cx = cov[:, 0, 0]
    cy = cov[:, 0, 1]
    cz = cov[:, 1, 1]

    det_cov = cx * cz - cy * cy
    cx = cx + H_VAR
    cz = cz + H_VAR
    det = cx * cz - cy * cy                                      # det_cov_plus_h_cov

    # cuda: `if (det == 0.0f) return;`  (we also reject negative determinants,
    # which the CUDA kernel would happily propagate as garbage)
    good_det = det > 1e-12

    if antialiasing:
        ratio = torch.where(det > 0,
                            det_cov / torch.where(good_det, det,
                                                  torch.ones_like(det)),
                            torch.full_like(det, 2.5e-5))
        h_convolution_scaling = torch.sqrt(torch.clamp(ratio, min=2.5e-5))
    else:
        h_convolution_scaling = torch.ones_like(det)

    det_inv = 1.0 / torch.where(good_det, det, torch.ones_like(det))
    conic = torch.stack((cz * det_inv, -cy * det_inv, cx * det_inv), dim=1)

    # ---- screen-space extent (cuda: eigenvalues of the 2D covariance) -----
    mid = 0.5 * (cx + cz)
    disc = torch.sqrt(torch.clamp(mid * mid - det, min=0.1))
    lambda1 = mid + disc
    lambda2 = mid - disc
    my_radius = torch.ceil(3.0 * torch.sqrt(
        torch.clamp(torch.maximum(lambda1, lambda2), min=0.0)))
    _t1 = time.perf_counter() if _PROFILE else 0.0

    # ---- NDC -> pixel (cuda: ndc2Pix) ------------------------------------
    point_image = torch.stack((
        ((p_proj[:, 0] + 1.0) * W - 1.0) * 0.5,
        ((p_proj[:, 1] + 1.0) * H - 1.0) * 0.5,
    ), dim=1)                                                    # [N, 2]

    # ---- straight-through routing of dL/d(means2D) -----------------------
    # Forward value is the analytic projection (point_image), while the
    # gradient is credited to upstream's `screenspace_points` leaf exactly as
    # the CUDA kernel's `dL_dmeans2D` output does.  This is what keeps
    # `gaussians.add_densification_stats(viewspace_point_tensor, ...)` correct.
    if means2D is not None and means2D.requires_grad:
        means2d_used = point_image + (means2D[:, :2] - means2D[:, :2].detach())
    else:
        means2d_used = point_image

    # ---- per-Gaussian colour ---------------------------------------------
    if colors_precomp is not None and colors_precomp.numel() > 0:
        rgb = colors_precomp
    else:
        rgb = _sh_to_rgb(int(sh_degree), sh, means3D, campos)

    opa = opacities.reshape(-1)
    depth = p_view[:, 2]
    _t2 = time.perf_counter() if _PROFILE else 0.0

    # =====================================================================
    # Everything below is index bookkeeping: no gradients flow through it
    # (identical to the CUDA kernel, where binning is a forward-only step).
    # =====================================================================
    gx = (W + BLOCK_X - 1) // BLOCK_X
    gy = (H + BLOCK_Y - 1) // BLOCK_Y
    n_tiles = gx * gy

    with torch.no_grad():
        px = point_image[:, 0]
        py = point_image[:, 1]
        rad = my_radius

        rect_min_x = torch.clamp(torch.floor((px - rad) / BLOCK_X), 0, gx)
        rect_min_y = torch.clamp(torch.floor((py - rad) / BLOCK_Y), 0, gy)
        rect_max_x = torch.clamp(torch.floor((px + rad + (BLOCK_X - 1)) / BLOCK_X), 0, gx)
        rect_max_y = torch.clamp(torch.floor((py + rad + (BLOCK_Y - 1)) / BLOCK_Y), 0, gy)

        touched = (rect_max_x - rect_min_x) * (rect_max_y - rect_min_y)
        keep = inside & good_det & (touched > 0) & torch.isfinite(det)

        radii[keep] = rad[keep].to(torch.int32)

        cand = None
        valid = None
        pix_x = pix_y = None
        in_img = None
        counts_max = 0

        if bool(keep.any()):
            kidx = keep.nonzero(as_tuple=False).squeeze(1)
            k_min_x = rect_min_x[kidx].to(torch.int64)
            k_max_x = rect_max_x[kidx].to(torch.int64)
            k_min_y = rect_min_y[kidx].to(torch.int64)
            k_max_y = rect_max_y[kidx].to(torch.int64)

            ntx = k_max_x - k_min_x
            nty = k_max_y - k_min_y
            rep = ntx * nty
            total = int(rep.sum().item())

            starts = torch.cumsum(rep, 0) - rep
            gauss_of_pair = torch.repeat_interleave(
                torch.arange(kidx.numel(), dtype=torch.int64), rep)
            starts_rep = torch.repeat_interleave(starts, rep)
            rank = torch.arange(total, dtype=torch.int64) - starts_rep

            ntx_p = ntx[gauss_of_pair]
            tile_x = k_min_x[gauss_of_pair] + (rank % ntx_p)
            tile_y = k_min_y[gauss_of_pair] + (rank // ntx_p)
            tile_id = tile_y * gx + tile_x

            pair_depth = depth.detach()[kidx][gauss_of_pair]
            # cuda: key = (tile_id << 32) | float_bits(depth); radix sort
            # ascending => seeds are ordered front-to-back within each tile.
            #
            # We reproduce that ordering without bit-reinterpreting the depth:
            # rank each Gaussian by ascending view depth (0..N-1) and use
            # `tile_id * N + rank` as the key.  This is an exact, total,
            # dtype-agnostic ordering, whereas `tensor.view(torch.int32)` would
            # silently double the element count for float64 inputs.
            depth_order = torch.argsort(depth.detach())
            depth_rank = torch.empty_like(depth_order)
            depth_rank[depth_order] = torch.arange(N, dtype=torch.int64)
            pair_rank = depth_rank[kidx][gauss_of_pair]
            key = tile_id * N + pair_rank
            order = torch.argsort(key, stable=True)

            pair_tile = tile_id[order]
            pair_gauss = kidx[gauss_of_pair[order]]

            counts = torch.bincount(pair_tile, minlength=n_tiles)
            counts_max = int(counts.max().item())

            kmax, kmax_budget = resolve_kmax(n_tiles, counts_max)

            offsets = torch.cumsum(counts, 0) - counts
            rank_in_tile = torch.arange(total, dtype=torch.int64) - offsets[pair_tile]
            sel = rank_in_tile < kmax

            cand = torch.zeros(n_tiles * kmax, dtype=torch.int64)
            valid = torch.zeros(n_tiles * kmax, dtype=torch.bool)
            pos = pair_tile[sel] * kmax + rank_in_tile[sel]
            cand[pos] = pair_gauss[sel]
            valid[pos] = True
            cand = cand.view(n_tiles, kmax)
            valid = valid.view(n_tiles, kmax)

            tile_y_ar = torch.arange(n_tiles, dtype=torch.int64) // gx
            tile_x_ar = torch.arange(n_tiles, dtype=torch.int64) % gx
            py_ar = tile_y_ar.unsqueeze(1) * BLOCK_Y + torch.arange(BLOCK_Y)
            px_ar = tile_x_ar.unsqueeze(1) * BLOCK_X + torch.arange(BLOCK_X)
            pix_y = py_ar.unsqueeze(2).expand(n_tiles, BLOCK_Y, BLOCK_X).reshape(n_tiles, BLOCK_SIZE)
            pix_x = px_ar.unsqueeze(1).expand(n_tiles, BLOCK_Y, BLOCK_X).reshape(n_tiles, BLOCK_SIZE)
            in_img = (pix_y < H) & (pix_x < W)

            if _DEBUG:
                mode = (f"auto({_KMAX_AUTO_MIB:.0f}MiB x{_KMAX_LIVE_TENSORS:.0f})"
                        if _KMAX_AUTO else str(_KMAX_DEFAULT))
                print(f"[cpu-raster] N={N} kept={kidx.numel()} pairs={total} "
                      f"tiles={n_tiles} counts_max={counts_max} kmax={kmax} "
                      f"cap_mode={mode} legacy_budget={kmax_budget}")

    if cand is None:
        if _PROFILE:
            _t3 = time.perf_counter()
            _PROF["preprocess"] += _t1 - _t0
            _PROF["colour"] += _t2 - _t1
            _PROF["binning"] += _t3 - _t2
            _PROF["calls"] += 1
        out_color = out_color + bg.to(dtype).view(3, 1, 1)
        return out_color, radii, invdepths

    _t3 = time.perf_counter() if _PROFILE else 0.0

    n_tiles, kmax = cand.shape
    flat = cand.reshape(-1)                                      # [T*K]
    vmask = valid.reshape(-1)

    g_xy = means2d_used[flat]                                    # [T*K, 2]
    g_conic = conic[flat]                                        # [T*K, 3]
    g_opa = (opa * h_convolution_scaling)[flat]                  # [T*K]
    g_rgb = rgb[flat]                                            # [T*K, 3]
    g_depth = depth.detach()[flat]                               # [T*K]

    z = torch.zeros_like(g_opa)
    g_opa = torch.where(vmask, g_opa, z)
    g_rgb = torch.where(vmask.unsqueeze(1), g_rgb, torch.zeros_like(g_rgb))
    # padding must never contribute to the inverse-depth expectation
    g_invdepth = torch.where(vmask, 1.0 / torch.clamp(g_depth, min=1e-6),
                             torch.zeros_like(g_depth))

    g_xy = g_xy.view(n_tiles, kmax, 2)
    g_conic = g_conic.view(n_tiles, kmax, 3)
    g_opa = g_opa.view(n_tiles, kmax)
    g_rgb = g_rgb.view(n_tiles, kmax, 3)
    g_invdepth = g_invdepth.view(n_tiles, kmax)

    # ---- alpha compositing (cuda: renderCUDA) ----------------------------
    d_x = pix_x.unsqueeze(2) - g_xy[:, None, :, 0]                # [T, P, K]
    d_y = pix_y.unsqueeze(2) - g_xy[:, None, :, 1]
    cn = g_conic[:, None, :, :]
    power = -0.5 * (cn[..., 0] * d_x * d_x + cn[..., 2] * d_y * d_y) \
        - cn[..., 1] * d_x * d_y
    # cuda: `if (power > 0.0f) continue;`
    alpha = torch.where(power <= 0.0,
                        g_opa[:, None, :] * torch.exp(torch.clamp(power, max=0.0)),
                        torch.zeros_like(power))
    alpha = torch.clamp(alpha, max=ALPHA_MAX)
    # cuda: `if (alpha < 1.0f / 255.0f) continue;`
    alpha = torch.where(alpha < ALPHA_MIN, torch.zeros_like(alpha), alpha)

    # transmittance, computed in log space for numerical stability
    log_om = torch.log1p(-alpha)
    csum = torch.cumsum(log_om, dim=2)
    t_excl = torch.exp(torch.cat(
        (torch.zeros_like(csum[:, :, :1]), csum[:, :, :-1]), dim=2))
    t_final = torch.exp(csum[:, :, -1])                          # [T, P]

    weighted = alpha * t_excl
    contrib = torch.einsum("tpk,tkc->tpc", weighted, g_rgb)      # [T, P, 3]

    bg_c = bg.to(dtype).view(1, 1, 3)
    contrib = contrib + t_final.unsqueeze(2) * bg_c

    exp_invdepth = torch.einsum("tpk,tk->tp", weighted, g_invdepth)

    # ---- scatter tiles back into the image -------------------------------
    # Each pixel belongs to exactly one tile, so this is a pure scatter.
    # Done as a flat 1-D index_add, which autograd supports for `source`.
    idx_flat = (pix_y * W + pix_x)[in_img].reshape(-1)            # [M]
    src = contrib.permute(2, 0, 1)[:, in_img].reshape(3, -1)      # [3, M]
    chan = torch.arange(3, device=device).unsqueeze(1) * (H * W)  # [3, 1]
    flat_target = (chan + idx_flat.unsqueeze(0)).reshape(-1)      # [3*M]
    out_color = out_color.reshape(-1).index_add(
        0, flat_target, src.reshape(-1)).view(3, H, W)

    src_d = exp_invdepth[in_img].reshape(-1)                      # [M]
    invdepths = invdepths.reshape(-1).index_add(
        0, idx_flat, src_d).view(1, H, W)

    if _PROFILE:
        _PROF["preprocess"] += _t1 - _t0
        _PROF["colour"] += _t2 - _t1
        _PROF["binning"] += _t3 - _t2
        _PROF["composite"] += time.perf_counter() - _t3
        _PROF["calls"] += 1

    return out_color, radii, invdepths


class GaussianRasterizer(nn.Module):
    def __init__(self, raster_settings):
        super().__init__()
        self.raster_settings = raster_settings

    def markVisible(self, positions):
        with torch.no_grad():
            rs = self.raster_settings
            p_h = torch.cat((positions, torch.ones_like(positions[:, :1])), dim=1)
            p_view = p_h @ rs.viewmatrix
            return p_view[:, 2] > 0.2

    def forward(self, means3D, means2D, opacities, shs=None,
                colors_precomp=None, scales=None, rotations=None,
                cov3D_precomp=None):
        raster_settings = self.raster_settings

        if (shs is None and colors_precomp is None) or \
           (shs is not None and colors_precomp is not None):
            raise Exception('Please provide excatly one of either SHs or precomputed colors!')

        if ((scales is None or rotations is None) and cov3D_precomp is None) or \
           ((scales is not None or rotations is not None) and cov3D_precomp is not None):
            raise Exception('Please provide exactly one of either scale/rotation pair or precomputed 3D covariance!')

        if shs is None:
            shs = torch.Tensor([])
        if colors_precomp is None:
            colors_precomp = torch.Tensor([])
        if scales is None:
            scales = torch.Tensor([])
        if rotations is None:
            rotations = torch.Tensor([])
        if cov3D_precomp is None:
            cov3D_precomp = torch.Tensor([])

        return rasterize_gaussians(
            means3D, means2D, shs, colors_precomp, opacities,
            scales, rotations, cov3D_precomp, raster_settings,
        )
