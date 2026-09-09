# vLLM csrc patches (require a `_rocm_C` rebuild)

Unlike everything under `../vllm/`, these touch `.cu` sources that are **compiled into the
wheel**, so applying them means rebuilding the `_rocm_C` extension. The benchmark recipe does
NOT apply them; the container carries a rebuilt `_rocm_C.abi3.so` instead.

## `skinny_gemms_n6to9.patch`

Lets `wvSplitK` serve token counts 6-9. Two independent problems:

1. **Missing instantiations.** `switch (N_in)` enumerated only `case 1..5` and threw
   `"Unsupported N value"` otherwise. The kernel is templated on `int N` and uses it purely as
   a loop bound / array extent, so it was never a capability limit -- just five cases.
2. **`YTILE` did not shrink as `N` grew.** Accumulators are `sum[N][YTILE]`,
   `sum4[N][YTILE]`, `bigA[N][UNRL]`, so register cost scales with `N * YTILE`. With
   `M_in=2880` the tile selector picks `YTILE=3`, which past N=5 blows the budget. Measured
   with `-Rpass-analysis=kernel-resource-usage` on gfx950:

   | N | VGPR | scratch B/lane | VGPR spills |
   |---:|---:|---:|---:|
   | 1-4 | 51-122 | 0 | 0 |
   | 5 | 98-144 | 0 | 0 |
   | 6 | 110-164 | 160 | 39 |
   | 7 | 122-186 | 320 | 103 |
   | 8 | 128-206 | **364** | **141** |
   | 9 | 128-226 | **516** | **212** |

   At N=8, `YTILE=1, UNRL=2` needs **zero** scratch. The patch clamps `YTILE` to 1 for N>=6.

Effect (gfx950, idle box, bf16, K=7168):

| N,K | N=8 before | N=8 after | vs torch | vs tuned aiter |
|---|---:|---:|---:|---:|
| 2880,7168 | 44.7 us | **15.2** | -26% | -56% |
| 1536,7168 | 33.3 | **11.2** | -44% | -64% |
| 3584,7168 | 48.6 | **17.3** | -24% | -49% |
| 7168,1536 | 35.0 | **11.6** | -29% | -46% |

Across 48 (shape, n) points at n=5..9 the winner is `wvSplitK` 35 times, `torch` 13, and the
tuned aiter path **0**. The clamp also speeds up the already-supported n<=5 (2880x7168 at n=5:
19.6 -> 11.4 us), because `YTILE=3` was already marginal there.

Pairs with `../vllm/utils.patch`, which widens the Python gate from `0 < n <= 5` to `<= 9`.
Conc-1 with K=7 gives `n = 1 + 7 = 8`, which previously fell through to the aiter/asm path on
16-row tiles for ~12% of the decode step.

**Upstream candidate.** Nothing here is K3-specific: any speculative-decode configuration with
`1 + num_speculative_tokens` in 6-9 hits the same gap, and the YTILE clamp improves n<=5 too.

## Rebuilding

Fetch vLLM at the pinned commit, apply the patch, then:

    mkdir build && cd build
    PYTORCH_ROCM_ARCH=gfx950 cmake .. -G Ninja -DVLLM_TARGET_DEVICE=rocm \
      -DCMAKE_PREFIX_PATH="$(python3 -c "import torch;print(torch.utils.cmake_prefix_path)")" \
      -DVLLM_PYTHON_EXECUTABLE="$(which python3)" -DPython_EXECUTABLE="$(which python3)"
    ninja _rocm_C
    cp _rocm_C.abi3.so <site-packages>/vllm/

**Set `PYTORCH_ROCM_ARCH=gfx950`.** The container env lists nine architectures, and without
this the build compiles every kernel nine times -- 4 minutes becomes 40+, almost all of it on
`attention.hip`, which this patch does not even touch.
