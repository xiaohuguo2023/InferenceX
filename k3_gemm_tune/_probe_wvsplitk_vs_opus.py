#!/usr/bin/env python3
# Head-to-head isolated microbench for the conc-1 decode dense GEMM (N=2880,K=7168):
#   wvSplitK (skinny, current default)  vs  opus (tuner winner, via tgemm+CSV)  vs  torch ref.
# The aiter tuner never enumerated a `skinny` candidate for this shape, so it could not
# tell us whether opus actually beats wvSplitK. This probe supplies that missing number.
# Run with AITER_CONFIG_GEMM_BF16 pointed at a CSV that contains the opus rows for this shape.
import os, torch, statistics

N, K = 2880, 7168
NS = [2, 3, 4]           # draft M=2, target-verify M=3 (conc-1 nspec-2), +4
ITERS, WARM = 300, 50

dev = "cuda"
dt = torch.bfloat16
weight = torch.randn(N, K, device=dev, dtype=dt).contiguous()   # [N,K]

from vllm import _custom_ops as ops
from aiter.tuned_gemm import tgemm
try:
    from vllm.model_executor.layers.utils import num_compute_units
    cu = num_compute_units()
except Exception:
    cu = 256

def bench(fn):
    for _ in range(WARM): fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(ITERS):
        s = torch.cuda.Event(enable_timing=True); e = torch.cuda.Event(enable_timing=True)
        s.record(); fn(); e.record(); torch.cuda.synchronize()
        ts.append(s.elapsed_time(e) * 1000.0)   # us
    ts.sort()
    return ts[len(ts)//2], statistics.mean(ts)

print(f"shape N={N} K={K}  cu={cu}  dtype={dt}  (us, median / mean over {ITERS})")
print(f"{'n':>3} | {'wvSplitK':>18} | {'opus(tgemm)':>18} | {'torch':>18} | winner")
for n in NS:
    x = torch.randn(n, K, device=dev, dtype=dt).contiguous()    # [n,K]
    xv = x.reshape(-1, x.size(-1)).contiguous()

    def f_wv():  return ops.wvSplitK(weight, xv, cu, None)
    def f_op():  return tgemm.mm(x, weight, None)
    def f_th():  return torch.nn.functional.linear(x, weight, None)

    # correctness sanity vs torch
    ref = f_th()
    for name, f in [("wvSplitK", f_wv), ("opus", f_op)]:
        try:
            o = f().reshape(ref.shape)
            err = (o.float() - ref.float()).abs().max().item() / (ref.float().abs().max().item() + 1e-6)
            if err > 0.05: print(f"   !! {name} n={n} rel-err {err:.3f}")
        except Exception as ex:
            print(f"   !! {name} n={n} FAILED: {ex}")

    wv = bench(f_wv); op = bench(f_op); th = bench(f_th)
    best = min([("wvSplitK", wv[0]), ("opus", op[0]), ("torch", th[0])], key=lambda t: t[1])
    print(f"{n:>3} | {wv[0]:8.2f}/{wv[1]:7.2f} | {op[0]:8.2f}/{op[1]:7.2f} | {th[0]:8.2f}/{th[1]:7.2f} | {best[0]} ({best[1]:.2f}us)")
