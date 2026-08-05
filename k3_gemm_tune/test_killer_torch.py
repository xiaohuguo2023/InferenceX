#!/usr/bin/env python3
"""Quick torch bf16 GEMM sanity check for killer shapes."""
import gc
import os
import sys
import time

import torch

SHAPES = [(4096, 1024, 4096), (32768, 1024, 4096)]
DEV = int(os.environ.get("TEST_GPU", "0"))
ITERS = int(os.environ.get("TEST_ITERS", "5"))


def test_shape(M: int, N: int, K: int, device: int) -> None:
    torch.cuda.set_device(device)
    print(f"\n=== M={M} N={N} K={K} on cuda:{device} ===", flush=True)
    a = torch.randn(M, K, device=f"cuda:{device}", dtype=torch.bfloat16)
    b = torch.randn(K, N, device=f"cuda:{device}", dtype=torch.bfloat16)
    for _ in range(2):
        c = torch.mm(a, b)
    torch.cuda.synchronize(device)
    t0 = time.perf_counter()
    for _ in range(ITERS):
        c = torch.mm(a, b)
    torch.cuda.synchronize(device)
    ms = (time.perf_counter() - t0) / ITERS * 1000
    tflops = 2 * M * N * K / (ms / 1000) / 1e12
    print(
        f"torch.mm ok out={tuple(c.shape)} avg={ms:.2f}ms ~{tflops:.2f} TFLOPS",
        flush=True,
    )
    del a, b, c
    gc.collect()
    torch.cuda.empty_cache()


def main() -> int:
    hip = getattr(torch.version, "hip", None)
    n = torch.cuda.device_count()
    print(f"torch={torch.__version__} hip={hip} devices={n}", flush=True)
    if not torch.cuda.is_available():
        print("ERROR: no cuda/hip devices", flush=True)
        return 1
    for M, N, K in SHAPES:
        test_shape(M, N, K, DEV)
    print("\nALL DONE - no torch fault", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
