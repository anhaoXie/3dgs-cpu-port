"""
simple_knn._C -- pure-Python/CPU replacement for the CUDA `simple-knn` extension.

Upstream `scene/gaussian_model.py` initialises per-Gaussian scales with

    dist2 = torch.clamp_min(distCUDA2(points), 0.0000001)
    scales = torch.log(torch.sqrt(dist2))[..., None].repeat(1, 3)

`distCUDA2` (see submodules/simple-knn/simple_knn.cu :: boxMeanDist) returns, for
every point, the **mean of the squared distances to its 3 nearest neighbours**:

    dists[i] = (d1^2 + d2^2 + d3^2) / 3

The CUDA implementation is an exact brute-force search accelerated by a Morton-
ordered bounding-box rejection test -- it is not an approximation.  A k-d tree
therefore reproduces it bit-for-bit up to floating-point associativity.
"""

import numpy as np
import torch

try:
    from scipy.spatial import cKDTree
    _HAVE_SCIPY = True
except ImportError:                                        # pragma: no cover
    _HAVE_SCIPY = False

__all__ = ["distCUDA2"]


def _brute_force_mean3(pts):
    """Exact O(n^2) fallback (also used for very small inputs)."""
    n = pts.shape[0]
    d2 = np.zeros(n, dtype=np.float64)
    for i in range(n):
        diff = pts - pts[i]
        sq = np.einsum("ij,ij->i", diff, diff)
        sq[i] = np.inf
        k = min(3, n - 1)
        if k <= 0:
            d2[i] = 0.0
        else:
            d2[i] = np.sort(sq)[:k].mean()
    return d2


def distCUDA2(points):
    """Mean squared distance to the 3 nearest neighbours, per point.

    `points` is expected to be a float tensor of shape [N, 3].  The returned
    tensor matches upstream's contract: shape [N], float32, on the same device.
    """
    if not isinstance(points, torch.Tensor):
        points = torch.as_tensor(points)

    device = points.device
    n = points.shape[0]
    if n == 0:
        return torch.zeros((0,), dtype=torch.float32, device=device)

    pts = points.detach().to("cpu", torch.float64).numpy()
    pts = np.ascontiguousarray(pts)

    if n <= 4 or not _HAVE_SCIPY:
        d2 = _brute_force_mean3(pts)
    else:
        tree = cKDTree(pts)
        k = min(4, n)                     # column 0 is the point itself
        dist, _ = tree.query(pts, k=k, workers=-1)
        # dist[:, 0] == 0 (self); the 3 nearest neighbours follow.
        sq = dist[:, 1:k] ** 2
        if sq.shape[1] < 3:
            # Degenerate (heavily duplicated) clouds: pad with the largest
            # available squared distance so the mean stays well defined.
            pad = np.repeat(sq[:, -1:], 3 - sq.shape[1], axis=1)
            sq = np.concatenate([sq, pad], axis=1)
        d2 = sq.mean(axis=1)

    d2 = np.maximum(d2, 0.0)
    return torch.from_numpy(d2).to(dtype=torch.float32, device=device)
