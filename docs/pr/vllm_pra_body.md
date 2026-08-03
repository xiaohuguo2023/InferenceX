# [ROCm][MLA] FP8 ASM MLA prefill for non-divisor small head counts (Kimi-K3)

> Draft PR body for **PR-A** (vLLM), independent branch off `main`:
> `xguo/rocm-mla-fp8-prefill-independent`. Single file, +77/−25.

## Purpose

The `ROCM_AITER_MLA` FP8 **prefill** path (`mla_prefill_ps_asm_fwd` + `mla_reduce_v1`, the
"PS" persistent-scheduling kernels) is gated on `num_heads % 16 == 0`. Kimi-K3 has 96
attention heads over `kv_lora_rank=512`, i.e. **12 heads/rank at TP8** — not a multiple of
16 — so its FP8 prefill falls back to the BF16 FMHA decompress path
(`super().forward_mha` → `fmha_fwd_hd192_hd128_bf16`).

That fallback decompresses paged KV into a bf16 working set whose size is not covered by the
FP8 KV-pool accounting, so a large fresh context exhausts the activation arena (0 MB free)
even though the KV pool is <4% used. K3 FP8 crashed at ~197k tokens.

This enables the FP8 asm prefill for non-divisor small head counts by padding the query heads
up to 16 and slicing back. MLA attention is independent per query head over the shared latent
KV, so the padding heads cannot affect the real ones — exact, the same reasoning the asm
*decode* path uses.

## What this changes (single file, `rocm_aiter_mla.py`)

**Non-divisor head padding in `AiterMLAHelper`** (same approach as the asm decode work in
PR #50578, included here so this PR is self-contained):
- `get_mla_padded_q`: tile the query heads and slice to 16 for non-divisor counts
  (`repeat_interleave` only handles divisors).
- `get_mla_unpadded_o`: slice the first `num_heads` back off.
- `use_gluon_decode`: route non-divisor padded decodes to the asm persistent kernel;
  divisors and `max_qo_len > 1` verify still use Gluon.
- docstring / `check_num_heads_validity` message updates.

**FP8 prefill enablement (the new part):**
- Relax the `_fp8_prefill_enabled` gate to also allow `0 < num_heads < 16`.
- In `_mla_fp8_prefill_attn`: replicate-pad Q/K/V to 16 via `get_mla_padded_q`, run the PS
  asm prefill + `mla_reduce_v1` at 16 heads, then copy the first `num_heads` of the result
  back into the caller's output buffer (padded output uses a scratch buffer since it can't
  alias the real-head `out` storage).
- Build the PS metadata for the padded head count — `num_head_k = max(16, num_heads)` in both
  `_init_fp8_prefill_ps_buffers` and `_build_fp8_prefill_ps_metadata` — so the work/reduce
  maps match the padded tensors. This also lowers the partial-tile count:
  `gcd(16, cu_num=256) = 16` → ~960 tiles vs `gcd(12, 256) = 4` → ~4032, reclaiming **~6 GiB**
  of workspace, which is what lets the long context fit.

Divisor head counts and the existing `% 16 == 0` path are unchanged.

## Relationship to other PRs

- **#50578** ([ROCm][MLA] asm decode for non-divisor small head counts) — the non-divisor
  `AiterMLAHelper` padding here is the same change; credited to that PR. Kept in this PR so it
  stands alone; will rebase to drop the overlap if #50578 lands first.
- **#48712** ("Only run FP8 AITER MLA prefill when using FP8 KV") — the PS buffer reservation
  should be gated on FP8 KV so a bf16 serve doesn't pay for it; composes cleanly.
- **ROCm/aiter#4452** — required for correct >4 GB paged-KV addressing (64-bit byte offsets)
  at long context; without it the offsets truncate at 32-bit.

## Test plan

MI355X (gfx950) ×8, TP8, Kimi-K3 mxfp4, `--kv-cache-dtype fp8
--attention-backend ROCM_AITER_MLA --gpu-memory-utilization 0.95` (no
`--max-context-length`):

```bash
vllm bench serve --model <kimi-k3> --dataset-name random --ignore-eos \
  --random-input-len 131072 --random-output-len 128 --max-concurrency 1 --num-prompts 2
```

## Test result

- FP8 prefill previously crashed at ~197k tokens (BF16 FMHA fallback, activation OOM).
- After this change: fresh **470k (68 s)** and **590k (28.6 s)** single-request prefills
  complete, no OOM, gpu-mem util 0.95, KV pool <4%.
- PS workspace reservation drops ~6 GiB (4032 → 960 partial tiles).
- Output unchanged vs the BF16 FMHA path on short contexts (padded heads sliced off).
