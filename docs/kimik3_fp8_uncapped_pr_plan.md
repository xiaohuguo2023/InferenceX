# Kimi-K3 MI355X fp8-ASM — PR plan (aiter + vLLM)

Plan to upstream the local patches that make K3 **fp8 KV + ASM MLA** run **uncapped**
(full ~1M context, no `--max-context-length`) and fast on MI355X (gfx950, TP8), with
explicit reuse of existing upstream PRs so we don't duplicate or conflict.

## What we changed locally (validated in `xguo-k3asm`)

| # | area | file | change | status upstream |
|---|---|---|---|---|
| L1 | vLLM MLA **decode** | `rocm_aiter_mla.py` | append/tile-pad 12→16 heads, route single-token decode to asm persistent kernel | **already = PR #50578** |
| L2 | vLLM skinny GEMM | `model_executor/layers/utils.py` | force activation contiguous before `wvSplitK`/`wvSplitKrc` | **already = PR #50618** |
| L3 | vLLM MLA **prefill** | `rocm_aiter_mla.py` | **NEW:** pad Q/K/V 12→16 for the fp8 PS asm prefill (gate relax + slice output) | **none — our PR** |
| L4 | vLLM MLA prefill meta | `rocm_aiter_mla.py` | **NEW:** PS `num_head_k` 12→16 in `_init_fp8_prefill_ps_buffers` + `_build_fp8_prefill_ps_metadata` (matches padded tensors; 4032→960 partial tiles, −6 GiB) | **none — our PR** |
| L5 | aiter MLA HSACO | `hsa/gfx950/mla/*.co` + `asm_mla.cu` | 64-bit paged-KV offsets for large page_id | **= PR #4452 (cherry-picked)** |
| L6 | aiter bf16 GEMM cfg | `configs/model_configs/kimik3_bf16_tuned_gemm.csv` | tuned a16w16 GEMM config for K3 shapes | **new config (after tuning)** |

**L3 + L4 are the novel, uncapped-fp8 enablers.** Everything else is adopt-existing.

## Upstream PR landscape

### vLLM (vllm-project/vllm)
| PR | title | state | relation |
|---|---|---|---|
| **#50578** | [ROCm][MLA] Use asm decode for non-divisor small head counts | OPEN (vanshbhatia-amd) | **== L1.** Adopt; our decode patch is byte-equivalent for K3. Our prefill PR (L3/L4) **depends on/stacks on** this. |
| **#50618** | [Bugfix][ROCm] wvSplitK: fix OOB read on strided activations | OPEN | **== L2.** Adopt; unblocks full cudagraph capture. |
| **#48712** | [Bugfix][ROCm] Only run FP8 AITER MLA prefill when using FP8 KV | OPEN | Gate PS metadata/workspace on fp8 KV. **Compose:** our L3/L4 should gate PS workspace the same way (avoid reserving it for bf16). |
| **#44544** | [ROCm][MLA] AITER FP8 ASM prefill backend | OPEN | Long-term: an asm path for the **chunked-context** prefill. Would eliminate the BF16 FMHA fallback that forces the KV-vs-activation trade-off. **Coordinate**, don't conflict. |
| #50579 | [ROCm][MoE] Fix Kimi-K3 a8w4 MoE decode garbage (pin fp8 stage-1) | OPEN (vanshbhatia-amd) | Same K3 a8w4 MoE we serve (`AITER_SITUV2_A8W4`/`BF16_FP8_MOE_BOUND=0`). Adopt as the MoE-correctness dep. |
| #50619 | [ROCm][Perf] Fix Kimi-K3 DSpark FP8 MLA verification | OPEN (JohnQinAMD) | Spec-decode verify path; not needed for our sweeps but same family. |
| #50181 | Fix fp8 KV prefill query quant selection for Kimi-K3 | OPEN | **NVIDIA** prefill-backend selection (FlashInfer/TRTLLM) — not the ROCm asm path. Not ours. |

### aiter (ROCm/aiter)
| PR | title | state | relation |
|---|---|---|---|
| **#4341** | fix(mla): refresh qh16 fp8 persistent decode HSACO for large page_id | **MERGED** | Already in our build; fixes the **decode** kernel offsets. |
| **#4452** | fix(mla): refresh gfx950 MLA HSACO for large page_id KV addressing | OPEN | **== L5.** 64-bit offsets for the a8w8/a16w16 **prefill** kernels + `asm_mla.cu`. We cherry-picked & validated on K3. **Push to merge.** |
| #4351 | fix(mla): refresh gfx950 MLA HSACO batch for large page_id | OPEN | Earlier/subset of #4452 (incl. `CKV_mem_va_upd`). Superseded by #4452. |
| #4474 | [Bugfix][Triton] Fix int32 KV-offset overflow in _mla_gluon >2GB | OPEN | Gluon path (we use asm), same bug class. Adopt if any gluon fallback is used. |
| #4480 | Enable fp8 KV cache for small-head MLA decode (gluon) | OPEN | Gluon fp8; we bypass gluon via asm. Not on our path. |
| #4488 | [Test] mla_gluon: regression test for >2GB KV path | OPEN | Test only. |

## The PRs to file

### PR-A (vLLM, primary) — "Enable fp8 ASM MLA **prefill** for non-divisor small head counts"
- **Content:** L3 + L4.
  - Relax `_fp8_prefill_enabled` gate: allow `0 < num_heads < 16` (not just `% 16 == 0`).
  - In `_mla_fp8_prefill_attn`: replicate-pad Q/K/V to 16 (reuse `get_mla_padded_q`), run PS asm prefill + `mla_reduce_v1` at 16 heads, slice output back to `num_heads`.
  - Set `num_head_k = max(16, num_heads)` in **both** `_init_fp8_prefill_ps_buffers` and `_build_fp8_prefill_ps_metadata` (correctness *and* −6 GiB workspace: gcd(16,256)=16 → ~960 partial tiles vs gcd(12,256)=4 → ~4032).
- **Depends on:** #50578 (decode pad helpers `get_mla_padded_q`/`get_mla_unpadded_o`). Open as a **stacked PR on #50578** (or note the dependency).
- **Compose with #48712:** gate the PS buffer reservation on fp8 KV so bf16 doesn't pay for it.
- **Why:** removes the memory-pathological BF16-FMHA reservation mismatch; lets K3 fp8 run **uncapped** (validated: 197k-crash → 470k/590k OK, gpu-mem 0.95).
- **Test plan:** `vllm bench serve` random ISL 32k/64k/128k + a fresh 512k single request (no OOM); agentic `inferencex-agentx-mvp` conc 1–24 uncapped completes.

### PR-B (aiter) — land #4452
- Not a new PR: **review/validate & push #4452** (64-bit paged-KV offsets). Add our K3 evidence (fresh 470k/590k prefill correct on gfx950). Note #4351 is superseded.

### PR-C (aiter, config) — "Add `kimik3_bf16_tuned_gemm.csv`"
- Tuned a16w16 GEMM config from `kimik3_bf16_tuning_gemm.csv` (424 shapes, 21 (N,K)); removes the 667k `not found tuned config … using torch` fallbacks. File after running the tuner (see `k3_gemm_tune/`). Mixed libtype (flydsl at decode-M, hipBLASLt/asm at prefill-M; 6288/16160 stay hipBLASLt).

### PR-D (vLLM, longer-term, optional) — chunked-context asm prefill
- Coordinate with **#44544**: route the `has_context` (chunked-context) prefill through an asm path that reads paged fp8 KV directly, eliminating the BF16 FMHA decompress fallback. This removes the KV-pool-vs-activation trade-off entirely (today mitigated by max-num-seqs 64 + the L4 workspace reclaim). Track, don't block PR-A on it.

## Ordering & dependencies
```
#50578 (decode pad)  ──►  PR-A (prefill pad + PS meta)         [vLLM]
#50618 (wvSplitK)    ──►  (already needed for capture)         [vLLM, adopt]
#48712 (gate on fp8) ──►  compose into PR-A                     [vLLM, adopt]
#4452  (64-bit off)  ──►  PR-B land                             [aiter]
tuner  ──►  PR-C kimik3_bf16_tuned_gemm.csv                     [aiter config]
#44544 ──►  PR-D chunked-context asm prefill                    [vLLM, later]
```

## Validation evidence (this session, MI355X gfx950 TP8, xguo-k3asm)
- Decode (fp8 `mla_a8w8_qh16`): correct output, ~26 ms/tok captured.
- Full cudagraph capture (FULL_AND_PIECEWISE) clean after #50618.
- **Uncapped fp8 prefill:** 470k (68 s) and 590k (28.6 s) fresh contexts, no OOM, gpu-mem 0.95, after PR-A (L3+L4).
- Agentic sweep (bf16-ASM) conc 1/4/8/16/24 completed; fp8-ASM uncapped sweep running post-fix.
- Config: `AITER_SITUV2_A8W4=1`, `AITER_BF16_FP8_MOE_BOUND=0`, `--moe-backend auto`,
  `--max-num-seqs 64`, `--max-model-len 1048576`, `--kv-cache-dtype fp8`,
  `--attention-backend ROCM_AITER_MLA`.
