"""
Report how a COLMAP scene is partitioned into train/test, using the very same
reader that train.py and render.py use.

Why this exists: metrics.py only ever walks the `test/` split, and train.py
creates one only when --eval is passed -- so a wrong or missing split fails
*quietly* and shows up as a slightly odd PSNR rather than as an error.  This
prints the counts straight from scene.dataset_readers.readColmapSceneInfo and
cross-checks them against the LLFF hold-out rule, so the split quoted in
notes/results.md can be re-verified at any time without retraining.

Usage:
  python tools/check_split.py --model_path <run dir>          # reads its cfg_args
  python tools/check_split.py --source_path <scene> --eval    # or state it here

Exit code is 0 when the observed counts match the rule, 1 otherwise.
"""

import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
WORK = os.path.dirname(HERE)
REPO = os.path.join(WORK, "gaussian-splatting")
# cpu_port supplies the pure-PyTorch stand-ins for the CUDA extensions that
# `scene.gaussian_model` imports (simple_knn._C) plus the CUDA->CPU device shim.
# This script is meant to be runnable standalone, so put it on the path here
# instead of requiring the caller to export PYTHONPATH.
sys.path.insert(0, os.path.join(WORK, "cpu_port"))
sys.path.insert(0, REPO)
try:
    import sitecustomize               # noqa: E402,F401  applies the CUDA->CPU shim
except ImportError:
    pass

from arguments import ModelParams, get_combined_args           # noqa: E402
from scene.dataset_readers import readColmapSceneInfo          # noqa: E402

LLFF_HOLD = 8                       # upstream default in readColmapSceneInfo


def main():
    parser = argparse.ArgumentParser(
        description="Report the train/test split of a COLMAP scene")
    ModelParams(parser, sentinel=True)      # registers --source_path/--model_path/--eval/...
    args = get_combined_args(parser)        # command line wins over <model_path>/cfg_args

    if not args.source_path:
        raise SystemExit("need --source_path, or --model_path pointing at a dir with cfg_args")

    print(f"source_path : {args.source_path}")
    print(f"model_path  : {args.model_path}")
    print(f"eval        : {args.eval}")
    print(f"images      : {args.images!r}   depths: {args.depths!r}")

    info = readColmapSceneInfo(args.source_path, args.images, args.depths,
                               args.eval, args.train_test_exp)
    n_train = len(info.train_cameras)
    n_test = len(info.test_cameras)
    total = n_train + n_test

    print(f"\ntrain cameras : {n_train}")
    print(f"test  cameras : {n_test}")
    print(f"total         : {total}")

    if not args.eval:
        print("\n(eval is off, so every camera is train -- pass --eval for a held-out split)")
        return 0

    expected_test = len([i for i in range(total) if i % LLFF_HOLD == 0])
    expected_train = total - expected_test
    print(f"\nLLFF rule: every {LLFF_HOLD}th camera by sorted name (index % {LLFF_HOLD} == 0) is test")
    print(f"  test  expected {expected_test:4d}   observed {n_test:4d}   "
          f"{'OK' if expected_test == n_test else 'MISMATCH'}")
    print(f"  train expected {expected_train:4d}   observed {n_train:4d}   "
          f"{'OK' if expected_train == n_train else 'MISMATCH'}")

    return 0 if (expected_test == n_test and expected_train == n_train) else 1


if __name__ == "__main__":
    sys.exit(main())
