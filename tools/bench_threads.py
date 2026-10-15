"""Which torch CPU ops are the serial bottleneck? Compare 1-thread vs N-thread timing."""
import os
import time

import torch


def timeit(fn, reps=5):
    fn()
    t0 = time.perf_counter()
    for _ in range(reps):
        fn()
    return (time.perf_counter() - t0) / reps * 1000.0


def main():
    print(f"torch {torch.__version__}  default threads={torch.get_num_threads()}  "
          f"OMP_NUM_THREADS={os.environ.get('OMP_NUM_THREADS')}")

    N = 400_000            # gaussians
    TPK = 144 * 256 * 256  # the dense [T,P,K] tensor (244x136, K=256)

    results = {}
    for nt in (1, 4, 8, 12):
        torch.set_num_threads(nt)
        e = {}

        a = torch.rand(N)
        b = torch.rand(N, 3)
        e["elementwise N"] = timeit(lambda: (a * 0.5 + 0.25).exp().clamp(max=1.0))
        e["sh strided (30 ops)"] = timeit(
            lambda: sum((b * 0.5).sum() for _ in range(10)))

        shs = torch.rand(N, 16, 3)
        e["shs[:,k] strided x4"] = timeit(
            lambda: sum(shs[:, k] * 0.5 for k in range(4)))

        key = torch.randint(0, 10_000_000, (N,), dtype=torch.int64)
        e["argsort int64 N"] = timeit(lambda: torch.argsort(key, stable=True))
        key32 = key.to(torch.int32)
        e["argsort int32 N"] = timeit(lambda: torch.argsort(key32, stable=True))
        e["bincount N"] = timeit(lambda: torch.bincount(key32 % 144, minlength=144))
        e["repeat_interleave N"] = timeit(
            lambda: torch.repeat_interleave(torch.arange(N, dtype=torch.int32),
                                            torch.ones(N, dtype=torch.int64)))
        e["nonzero N"] = timeit(lambda: (a > 0.5).nonzero())
        e["cumsum N"] = timeit(lambda: torch.cumsum(a, 0))

        big = torch.rand(TPK)
        e["elementwise TPK"] = timeit(lambda: (big * 0.5 + 0.25).exp())
        e["cumsum TPK[K]"] = timeit(lambda: torch.cumsum(big.view(144, 256, 256), 2))
        e["log1p TPK"] = timeit(lambda: torch.log1p(-big * 0.5))

        idx = torch.randint(0, N, (TPK,))
        src = torch.rand(N)
        e["gather TPK"] = timeit(lambda: src[idx])
        e["index_add TPK"] = timeit(
            lambda: torch.zeros(N).index_add(0, idx, torch.rand(TPK)))

        results[nt] = e

    keys = list(results[1].keys())
    print(f"\n{'op':26s} " + "".join(f"{f'{nt}T':>10s}" for nt in results)
          + f"{'12T/1T':>10s}")
    for k in keys:
        row = "".join(f"{results[nt][k]:10.1f}" for nt in results)
        sp = results[1][k] / max(results[12][k], 1e-9)
        print(f"{k:26s} {row}{sp:9.2f}x")


if __name__ == "__main__":
    main()
