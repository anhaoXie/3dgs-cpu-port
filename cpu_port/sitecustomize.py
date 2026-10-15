"""
sitecustomize.py -- CUDA -> CPU device shim for the 3D Gaussian Splatting CPU port.

WHY THIS EXISTS
---------------
The upstream graphdeco-inria/gaussian-splatting code hard-codes CUDA in a handful
of places (`device="cuda"`, `.cuda()`, `.to("cuda")`, `torch.cuda.Event`,
`torch.cuda.empty_cache`).  Those calls are *incidental* -- they say nothing about
the algorithm.  This shim transparently redirects them to the CPU so that the
official, byte-for-byte unmodified `train.py`, `render.py` and `metrics.py` can
run on a machine with no NVIDIA GPU.

The *algorithmically* meaningful CUDA code (the differentiable rasterizer and the
k-nearest-neighbour initialiser) is NOT shimmed: it is reimplemented in pure
PyTorch in the sibling `diff_gaussian_rasterization/` and `simple_knn/` packages.

This file is picked up automatically by CPython because its directory is placed
on PYTHONPATH (the `site` module imports `sitecustomize` at interpreter startup).

Nothing here is imported by the upstream repo; the repo is never modified.
"""

import os
import time as _time

import torch

# --------------------------------------------------------------------------
# 1. Device remapping
# --------------------------------------------------------------------------

def _cpuify(dev):
    """Map any CUDA device specifier to CPU; leave everything else untouched."""
    if dev is None:
        return None
    if isinstance(dev, str):
        if dev.split(":")[0] == "cuda":
            return "cpu"
        return dev
    if isinstance(dev, torch.device) and dev.type == "cuda":
        return torch.device("cpu")
    return dev


# Tensor factory functions that accept a `device=` keyword in the upstream code.
_FACTORIES = (
    "zeros", "ones", "empty", "full", "tensor", "as_tensor", "arange",
    "linspace", "logspace", "rand", "randn", "randint", "randperm", "eye",
    "zeros_like", "ones_like", "empty_like", "full_like", "scalar_tensor",
    "normal", "from_numpy", "asarray",
)


def _wrap_factory(fn):
    def wrapper(*args, **kwargs):
        if "device" in kwargs:
            kwargs["device"] = _cpuify(kwargs["device"])
        return fn(*args, **kwargs)

    wrapper.__name__ = getattr(fn, "__name__", "factory")
    wrapper.__doc__ = getattr(fn, "__doc__", None)
    return wrapper


for _name in _FACTORIES:
    _orig = getattr(torch, _name, None)
    if _orig is not None:
        setattr(torch, _name, _wrap_factory(_orig))
del _name, _orig


# --- Tensor.cuda() ---------------------------------------------------------
def _tensor_cuda(self, device=None, non_blocking=False,
                 memory_format=torch.preserve_format):
    """`.cuda()` is a no-op: this build of PyTorch has no CUDA device."""
    return self


torch.Tensor.cuda = _tensor_cuda


# --- Tensor.to(device) ----------------------------------------------------
_ORIG_TENSOR_TO = torch.Tensor.to


def _tensor_to(self, *args, **kwargs):
    if args:
        first = args[0]
        # .to(other_tensor) and .to(dtype) must keep their normal meaning.
        if not isinstance(first, (torch.Tensor, torch.dtype)):
            args = (_cpuify(first),) + tuple(args[1:])
    if "device" in kwargs:
        kwargs["device"] = _cpuify(kwargs["device"])
    return _ORIG_TENSOR_TO(self, *args, **kwargs)


torch.Tensor.to = _tensor_to


# --- nn.Module.cuda() -----------------------------------------------------
def _module_cuda(self, device=None):
    return self


torch.nn.Module.cuda = _module_cuda


# --------------------------------------------------------------------------
# 2. A CPU implementation of the handful of torch.cuda APIs the repo touches
# --------------------------------------------------------------------------

class _CPUEvent:
    """Stand-in for torch.cuda.Event backed by time.perf_counter()."""

    __slots__ = ("_t", "enable_timing", "blocking", "interprocess")

    def __init__(self, enable_timing=False, blocking=False, interprocess=False):
        self._t = None
        self.enable_timing = enable_timing
        self.blocking = blocking
        self.interprocess = interprocess

    def record(self, stream=None):
        self._t = _time.perf_counter()

    def elapsed_time(self, end_event):
        if self._t is None or end_event._t is None:
            return 0.0
        return float((end_event._t - self._t) * 1000.0)  # milliseconds, like CUDA

    def synchronize(self):
        pass

    def query(self):
        return True

    def wait(self, stream=None):
        pass


def _noop(*_args, **_kwargs):
    return None


torch.cuda.Event = _CPUEvent
torch.cuda.empty_cache = _noop
torch.cuda.set_device = _noop
torch.cuda.synchronize = _noop
torch.cuda.reset_peak_memory_stats = _noop
torch.cuda.reset_accumulated_memory_stats = _noop
torch.cuda.current_device = lambda: 0
torch.cuda.device_count = lambda: 0
torch.cuda.memory_allocated = lambda *a, **k: 0
torch.cuda.max_memory_allocated = lambda *a, **k: 0
torch.cuda.memory_reserved = lambda *a, **k: 0
torch.cuda.max_memory_reserved = lambda *a, **k: 0
torch.cuda.get_device_name = lambda *a, **k: "CPU (emulated CUDA device)"
torch.cuda.is_available = lambda: False

# `with torch.cuda.device(i):` is used by some helper libraries.
from contextlib import contextmanager as _contextmanager


@_contextmanager
def _cpu_device_ctx(device=None):
    yield


torch.cuda.device = _cpu_device_ctx


# --------------------------------------------------------------------------
# 3. Intra-op thread count
# --------------------------------------------------------------------------
# PyTorch's default intra-op parallelism is actively harmful on this machine:
# the i5-13420H is a *hybrid* part (4 P-cores + 4 E-cores).  Spreading OpenMP
# workers across heterogeneous cores makes the synchronising elementwise
# kernels slower, not faster.  Measured, clean (no other load), 400k elements:
#
#     op                  1 thread   12 threads   speedup(12T vs 1T)
#     elementwise [N]        0.4 ms      1.1 ms        0.33x
#     elementwise [T,P,K]   11.3 ms     35.7 ms        0.32x
#     log1p   [T,P,K]       18.6 ms     39.1 ms        0.47x
#     argsort int64 [N]     29.0 ms     63.5 ms        0.46x
#     index_add [T,P,K]     45.9 ms    141.5 ms        0.32x
#     gather  [T,P,K]       26.5 ms     28.1 ms        0.94x
#
# Note also that OMP_NUM_THREADS is *not* honoured by this torch build
# (torch.get_num_threads() stayed at 8 even with OMP_NUM_THREADS=12), so the
# count has to be set through the API -- which is exactly what this shim is for.
#
# Default to 1 thread (the fastest measured setting); override with
# DSH_3DGS_THREADS if that ever stops being true on other hardware.
_threads = os.environ.get("DSH_3DGS_THREADS")
if _threads is not None:
    try:
        torch.set_num_threads(max(1, int(_threads)))
        torch.set_num_interop_threads(1)
    except Exception:                                       # pragma: no cover
        pass

