#!/usr/bin/env python3
"""Skinny dead-zone head-to-head: wvSplitK vs the tuned GEMM vs torch, at n=5..10.

vllm/model_executor/layers/utils.py routes the unquantized bf16 GEMM as:
    n == 1            -> LLMM1
    2 <= n <= 5       -> wvSplitK
    6 <= n <= 9       -> NOTHING (falls through to aiter tgemm / asm hgemm_bf16)
    10 <= n <= 128    -> wvSplitKrc
Conc-1 with K=7 gives n = 1 + 7 = 8, i.e. dead centre, so 375 hgemm_bf16 calls
per decode step run 16-row tiles on 8-row problems.

The aiter tuner cannot answer whether wvSplitK would be better there: it
enumerated ZERO skinny candidates across all 275 shapes we just tuned, so its
"winner" is best-of{asm,opus,hipBLASLt}. This probe supplies the missing number,
and checks wvSplitK is numerically correct outside its current gate before we
consider widening it.

Point AITER_CONFIG_GEMM_BF16 at the tuned CSV to give tgemm its best shot.
"""
import statistics
import sys

import torch

from vllm import _custom_ops as ops
from aiter.tuned_gemm import tgemm

try:
    from vllm.model_executor.layers.utils import num_compute_units
    CU = num_compute_units()
except Exception:
    CU = 256

# (N, K) pairs actually issued by K3 at TP8, harvested from the serve logs.
SHAPES = [(2880, 7168), (6288, 7168), (8448, 7168), (3584, 7168),
          (2112, 7168), (1536, 7168), (7168, 4224), (7168, 1536)]
NS = [5, 6, 7, 8, 9, 10]
ITERS, WARM = 200, 40
DEV, DT = "cuda", torch.bfloat16


def bench(fn):
    for _ in range(WARM):
        fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(ITERS):
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record(); fn(); e.record(); torch.cuda.synchronize()
        ts.append(s.elapsed_time(e) * 1000.0)
    ts.sort()
    return ts[len(ts) // 2]


def main():
    print("CU=%d  (2<=n<=5 wvSplitK | 6<=n<=9 DEAD ZONE | n>=10 wvSplitKrc)\n" % CU)
    hdr = "%-14s %3s | %10s %10s %10s | %-10s %8s | %s" % (
        "N,K", "n", "wvSplitK", "tuned", "torch", "winner", "vs tuned", "wv err")
    print(hdr)
    print("-" * len(hdr))
    wins = {"wvSplitK": 0, "tuned": 0, "torch": 0}
    for (N, K) in SHAPES:
        w = torch.randn(N, K, device=DEV, dtype=DT).contiguous()
        for n in NS:
            x = torch.randn(n, K, device=DEV, dtype=DT).contiguous()
            ref = torch.nn.functional.linear(x, w)
            # wvSplitK is unsupported for n>=6 (kernel limit, not a gate) -- keep
            # measuring tuned vs torch there, since that IS the dead-zone choice.
            try:
                out_wv = ops.wvSplitK(w, x, CU, None)
                err = (out_wv.float() - ref.float()).abs().max().item()
                rel = err / max(ref.float().abs().max().item(), 1e-9)
                t_wv = bench(lambda: ops.wvSplitK(w, x, CU, None))
            except Exception:
                t_wv, rel = float("nan"), float("nan")
            t_tg = bench(lambda: tgemm.mm(x, w, None))
            t_th = bench(lambda: torch.nn.functional.linear(x, w))
            cand = [c for c in [("wvSplitK", t_wv), ("tuned", t_tg), ("torch", t_th)]
                    if c[1] == c[1]]
            best = min(cand, key=lambda t: t[1])
            wins[best[0]] += 1
            print("%-14s %3d | %10.2f %10.2f %10.2f | %-10s %+7.1f%% | %.1e"
                  % ("%d,%d" % (N, K), n, t_wv, t_tg, t_th, best[0],
                     (t_wv / t_tg - 1) * 100, rel))
        print()
    print("winners:", wins)


if __name__ == "__main__":
    main()
