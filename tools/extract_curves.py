"""
Extract the training loss curve and resource usage for the 3DGS CPU run.

Two independent sources are combined:
  1. the official TensorBoard event file written by upstream train.py
     (`train_loss_patches/total_loss`, `train_loss_patches/l1_loss`, `iter_time`,
      `total_points`) -- this is upstream's own logging path, unmodified;
  2. the tqdm progress stream, whose `Loss=` postfix carries upstream's EMA loss.

Outputs CSV files plus a PNG plot, all written next to the notes.
"""

import argparse
import ast
import csv
import os
import re
import sys

TQDM_RE = re.compile(
    r"(\d+)/(\d+)\s*\[[^\]]*?Loss=([0-9.eE+-]+),\s*Depth Loss=([0-9.eE+-]+)")
ITER_RE = re.compile(r"\[ITER (\d+)\] Evaluating (\w+): L1 ([0-9.eE+-]+) PSNR ([0-9.eE+-]+)")


def from_tensorboard(logdir):
    try:
        from tensorboard.backend.event_processing.event_accumulator import (
            EventAccumulator)
    except Exception as exc:                                  # noqa: BLE001
        print("tensorboard import failed:", exc)
        return {}
    if not any(f.startswith("events.out.tfevents") for f in os.listdir(logdir)):
        print("no tfevents file in", logdir)
        return {}
    acc = EventAccumulator(logdir)
    acc.Reload()
    tags = acc.Tags().get("scalars", [])
    print("tensorboard scalar tags:", tags)
    out = {}
    for t in tags:
        out[t] = [(e.step, e.value) for e in acc.Scalars(t)]
    return out


def from_log(path):
    if not os.path.exists(path):
        return [], []
    raw = open(path, "r", encoding="utf-8", errors="replace").read()
    prog = [(int(m.group(1)), float(m.group(3)), float(m.group(4)))
            for m in TQDM_RE.finditer(raw)]
    evals = [(int(m.group(1)), m.group(2), float(m.group(3)), float(m.group(4)))
             for m in ITER_RE.finditer(raw)]
    return prog, evals


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="model_path (has tfevents)")
    ap.add_argument("--log", required=True, help="captured stdout/stderr log")
    ap.add_argument("--outdir", required=True)
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    tb = from_tensorboard(args.model)
    prog, evals = from_log(args.log)

    # --- CSV: tensorboard losses ---
    csv_tb = os.path.join(args.outdir, "loss_tensorboard.csv")
    keys = [k for k in ("train_loss_patches/total_loss",
                        "train_loss_patches/l1_loss",
                        "iter_time", "total_points") if k in tb]
    if keys:
        cols = {k: dict(tb[k]) for k in keys}
        steps = sorted({s for k in keys for s, _ in tb[k]})
        with open(csv_tb, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["iteration"] + keys)
            for s in steps:
                w.writerow([s] + [cols[k].get(s, "") for k in keys])
        print(f"wrote {csv_tb}  ({len(steps)} rows)")

    # --- CSV: tqdm EMA loss ---
    csv_p = os.path.join(args.outdir, "loss_progress.csv")
    if prog:
        with open(csv_p, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["iteration", "ema_loss", "depth_loss"])
            # tqdm repeats the same (iter, loss) pair; keep one per iteration
            seen = {}
            for it, loss, dl in prog:
                seen[it] = (loss, dl)
            for it in sorted(seen):
                w.writerow([it, seen[it][0], seen[it][1]])
        print(f"wrote {csv_p}  ({len(seen)} rows)")

    if evals:
        print("evaluation checkpoints:")
        for it, split, l1, psnr in evals:
            print(f"  ITER {it:6d} {split:5s} L1={l1:.5f} PSNR={psnr:.3f}")

    # --- plot ---
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
        ax = axes[0]
        if prog:
            seen = {}
            for it, loss, _ in prog:
                seen[it] = loss
            xs = sorted(seen)
            ax.plot(xs, [seen[x] for x in xs], lw=1.0, color="#1f77b4",
                    label="tqdm EMA loss (upstream)")
        if "train_loss_patches/l1_loss" in tb:
            xs = [s for s, _ in tb["train_loss_patches/l1_loss"]]
            ys = [v for _, v in tb["train_loss_patches/l1_loss"]]
            ax.plot(xs, ys, lw=0.8, alpha=0.65, color="#d62728", label="L1 (tensorboard)")
        if "train_loss_patches/total_loss" in tb:
            xs = [s for s, _ in tb["train_loss_patches/total_loss"]]
            ys = [v for _, v in tb["train_loss_patches/total_loss"]]
            ax.plot(xs, ys, lw=0.8, alpha=0.8, color="#2ca02c",
                    label="total loss = 0.8*L1 + 0.2*(1-SSIM)")
        ax.set_xlabel("iteration")
        ax.set_ylabel("loss")
        ax.set_title("3DGS training loss (pure-PyTorch CPU rasterizer)")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)

        ax2 = axes[1]
        # `total_points` is only logged at testing_iterations, so early in a run
        # there may be a single sample -- fall back to iteration time then.
        if "total_points" in tb and len(tb["total_points"]) >= 2:
            xs = [s for s, _ in tb["total_points"]]
            ys = [v for _, v in tb["total_points"]]
            ax2.plot(xs, ys, lw=1.4, marker="o", ms=3, color="#9467bd")
            ax2.set_xlabel("iteration")
            ax2.set_ylabel("number of Gaussians")
            ax2.set_title("Gaussian count (adaptive densification)")
        elif "iter_time" in tb:
            xs = [s for s, _ in tb["iter_time"]]
            ys = [v for _, v in tb["iter_time"]]
            ax2.plot(xs, ys, lw=0.6, alpha=0.8, color="#ff7f0e")
            ax2.set_xlabel("iteration")
            ax2.set_ylabel("ms / iteration")
            ax2.set_title("iteration time (per-iteration wall clock)")
            if tb["total_points"]:
                s0 = tb["total_points"][-1]
                ax2.text(0.03, 0.95,
                         f"Gaussians @ iter {s0[0]}: {int(s0[1]):,}",
                         transform=ax2.transAxes, fontsize=9, va="top",
                         bbox=dict(fc="white", ec="#9467bd", alpha=0.9))
        ax2.grid(alpha=0.3)

        fig.tight_layout()
        png = os.path.join(args.outdir, "loss_curve.png")
        fig.savefig(png, dpi=140)
        print("wrote", png)
    except Exception as exc:                                  # noqa: BLE001
        print("plot failed:", exc)


if __name__ == "__main__":
    sys.exit(main())
