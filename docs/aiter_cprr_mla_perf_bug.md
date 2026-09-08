# `mla_*_cprr_ps` decode cost is linear in `max_qo_len`

**Component:** aiter asm MLA decode, `hsa/gfx950/mla/mla_a8w8_qh32_qseqlen4_gqaratio32_lse_cprr_ps.co`
**Severity:** perf. 6.1x slower than the non-CP kernel at identical shape; accounts for the entire
decode-context-parallel regression at low concurrency on MI355X.
**Status:** reproduced standalone on one GPU, no server required.

## Summary

Under decode context parallelism (DCP), every MLA layer calls the round-robin context-parallel
kernel `mla_*_cprr_ps`. Its cost scales **linearly with the multi-token query length**
(`max_qo_len`), which means it re-walks the whole KV shard once per query token instead of reading
the KV once and reusing it across the query dimension. The non-CP kernel `mla_*_ps`, given the same
work, does amortize and is 6.1x cheaper.

Because speculative decoding sets `max_qo_len = 1 + 2*num_speculative_tokens`, this charges a
deeper draft twice under DCP: once to produce the draft, and again in every verify step.

## Environment

| | |
|---|---|
| GPU | MI355X, gfx950, 256 CUs |
| ROCm / torch | ROCm 10, torch 2.12.0+rocm10.0.0 |
| aiter | `ROCm/aiter` @ `55dbc4f47` |
| vLLM | 0.28.1rc1.dev516+g9ea8f3ffc |
| model | Kimi-K3, MLA, fp8 KV cache, TP8, DCP8 |
| profiler | rocprofv3 1.3.5, `--kernel-trace` |

## The measurement that isolates it

Same shape on both sides — 96 attention heads, 27,316 KV tokens, `max_qo_len` 15, same launch grid
(256 workgroups), same LDS (163,840 B), same VGPR (512). The *only* difference is which kernel runs.

| kernel | us/call (p50) | p90 | max |
|---|---:|---:|---:|
| `mla_a8w8_qh32_qseqlen4_gqaratio32_ps` (non-CP) | **73.9** | 75.1 | 78.0 |
| `mla_a8w8_qh32_qseqlen4_gqaratio32_lse_cprr_ps` (CP) | **450.6** | 454.1 | 456.5 |

**6.1x**, at identical work.

## Evidence that the cause is the query dimension

Cost is very nearly proportional to `max_qo_len` (DCP8, 218,529 total ctx = 27,316 per rank,
96 -> 128 padded heads):

| `max_qo_len` | num_speculative_tokens | us/call |
|---:|---:|---:|
| 7 | 3 | 217.8 |
| 15 | 7 | 450.6 |

2.14x the query length produces 2.07x the time. For MLA decode the KV is read once and shared by
every query token, so cost should be `KV_read + qlen * compute`, not `qlen * KV_read`. The non-CP
kernel behaves correctly: 73.9 us at qlen 15 against ~44 us at qlen 1 — nowhere near 15x.

## Corroborating: effective bandwidth is pinned

Per-rank KV volume is `ctx_per_rank * 576 B` (kv_lora_rank 512 + qk_rope_head_dim 64, fp8).

| total ctx | per-rank KV | non-CP us | CP us | ratio | non-CP eff. BW | CP eff. BW |
|---:|---:|---:|---:|---:|---:|---:|
| 65,536 | 4.7 MB | 43.0 | 148.0 | 3.4x | 0.88 TB/s | 32 GB/s |
| 218,529 | 15.7 MB | 90.5 | 450.6 | 5.0x | 1.39 TB/s | 33 GB/s |

The non-CP kernel amortizes as context grows (0.88 -> 1.39 TB/s). The CP kernel sits at a flat
~33 GB/s regardless of size — the signature of repeated traversal rather than a sizing problem.

## Why this is a bug and not an inherent cost of DCP

At DCP8 each rank computes **all** heads over **1/8** of the KV, so the arithmetic is identical to
the non-CP case and the memory traffic is 8x *smaller*:

```
DCP8 : 96 heads x  27,316 tokens = 2.62M head-tokens,  15.7 MB KV read
DCP1 : 12 heads x 218,529 tokens = 2.62M head-tokens, 125.9 MB KV read
```

Same FLOPs, 8x less KV to read. The CP kernel should be **faster** than the non-CP kernel here, not
5x slower.

## Reproducer

Single GPU, no weights, no server, no collectives — ~2 minutes per point.

```bash
# CP kernel (mla_*_lse_cprr_ps)
python3 _dcp_folded_mla_standalone.py --dcp 8 --num-spec 7 --reqs 1 \
        --ctx 218529 --max-model-len 262144 --iters 100

# non-CP kernel at the SAME per-rank shape (96 heads, 27,316 ctx, qlen 15)
python3 _dcp_folded_mla_standalone.py --dcp 1 --num-spec 7 --reqs 1 --heads 768 \
        --ctx 27316 --max-model-len 32768 --iters 100

# query-length scaling: same command, --num-spec 3 (qlen 7) vs 7 (qlen 15)
```

Wrap either in `rocprofv3 --kernel-trace --output-format csv -d <dir> --` and read
`us/call` for the `mla_*` kernel from the per-PID `*_kernel_trace.csv`.

## Impact

In a real serve (Kimi-K3, TP8 + DCP8, conc 1, `num_speculative_tokens=7`), rocprofv3 over the
steady-state decode window shows the CP kernel at **18.6%** of filtered decode GPU time, called
exactly once per MLA layer per step (2,987 dispatches / 24 MLA layers = 124 steps).

Modelled from the isolated numbers, replacing the CP kernel's cost with the non-CP kernel's would
save `450.6 - 73.9 = 376.7 us` per MLA layer, i.e. `376.7 us x 24 layers = 9.0 ms` per decode step,
or **~2.25 ms per output token** at acceptance length 4.

Measured end-to-end, the DCP8 arm loses **1.77 ms/token** of ITL p90 against the otherwise-identical
DCP1 arm (7.66 -> 9.43 ms), which this kernel fully accounts for. Since the benchmark's
interactivity metric is the reciprocal of ITL p90, that is a ~19% interactivity loss attributable
to this one kernel.

## There is no route around it in-tree

| alternative | result |
|---|---|
| `VLLM_ROCM_AITER_MLA_DCP_VERIFY=segmented` (Triton) | `_mla_decode_fwd_kernel_...` at **118,892 us/call** — 264x worse |
| opus / OpFoundry `hk_mla_v32_decode_fwd` | gated on `nhead * max_seqlen_q == 64` (ours is 1920); experimental; DSA v3.2; no context-parallel support |
| flydsl | no MLA decode kernel exists (`aiter/ops/flydsl/kernels/` has only `mla_reduce.py`) |
| `cp_kv_cache_interleave_size != 1` | **incorrect results** — the kernel's global-position causal window assumes interleave 1 |
| a `cprr_v3` build for this shape | not shipped; the only fp8 `cprr_v3` is `qh16/gqaratio16`, and `qh` is set by head count, not query length |
| `hk_mla_v40` (HIP C++, has the right KV pipeline) | `nhead * max_seqlen_q` must be exactly 64 or 128; ours is 672-1440. Also no CP support. See "Suggested direction" |

### Reducing `max_qo_len` does NOT work — measured, not assumed

The obvious mitigation is a shallower draft, since cost is linear in `max_qo_len`. We ran it
end-to-end at concurrency 1 (Kimi-K3 agentic trace, 3600 s, DCP8 + DRAM offload, retention 1536):

| `num_speculative_tokens` | acceptance length | ITL p50 | ITL p90 | interactivity p90 |
|---:|---:|---:|---:|---:|
| 7 (`max_qo_len` 15) | 3.84 | 7.49 | 9.43 | 106.04 |
| 3 (`max_qo_len` 7) | 3.00 | 8.74 | 9.77 | 102.35 |

Halving the kernel (450.6 -> 217.8 us/layer, ~5.6 ms/step saved) is more than cancelled by the
acceptance loss: AL 3.84 -> 3.00 is ~28% more decode steps per output token. Net result is
**worse** — ITL p50 +16.7%, interactivity -3.5%.

**There is therefore no user-side mitigation.** Every ROCm DCP deployment pays this in full.

## Suggested direction

The fix is the same in either implementation: hold a KV tile resident and accumulate it against
*all* query tokens, instead of re-traversing the KV per query token.

**Preferred — port CP support onto `hk_mla_v40` rather than patching the asm.**
`csrc/kernels/mla/hk/` already contains a source-available HIP C++ MLA decode for gfx950
(`mi35x_v40_fwd_decode_m16x8_fp8bf16_fp8bf16_gen1.cuh`, spec in
`doc/hk_mla_v40_gen1_spec.md`) whose `KvManager8to16bitsV2` is exactly the double-buffered,
LDS-resident KV pipeline this bug is about — i.e. it already does the amortization correctly, and
treats multi-token prediction (`mtp`) as a first-class shape parameter. It also already emits
per-split partials + `attn_lse` and combines through `mla_reduce_v1`, which is the structure DCP
needs.

Two gaps to close:

1. **`W = nhead * max_seqlen_q` is capped at 64 or 128** (`aiter/mla.py::mla_v40_decode_fwd`
   raises `NotImplementedError` otherwise). This excludes *every* speculative-decode shape we run,
   DCP or not — DCP8+K=7 needs W=1440, DCP8+K=3 needs 672, and even DCP1+K=7 needs 180. The design
   pins all of Q in VGPRs at occupancy 1, so lifting the cap means iterating Q chunks inside the KV
   tile residency. Note this is **not DCP-specific**: any speculative decode with a realistic head
   count exceeds 128 head-tokens.
2. **CP global-position causal window.** V40 is currently "causal-free per-token"; DCP needs the
   causal mask evaluated on global positions reconstructed from `g_kv_indptr`, because each rank
   holds only every `cp_world_size`-th token.

**Fallback — fix the asm in place.** Amortize the KV read across `max_qo_len` in
`mla_*_cprr_ps` directly. Both kernels already use the same tiling (`qh32`, `qseqlen4`), the same
LDS budget and the same grid, so the query-batching structure of `mla_*_ps` should carry over; the
CP-specific part (the `g_kv_indptr` global-position window) is independent of how the KV read is
batched. Note the shipped tile is `qseqlen4` while real `max_qo_len` is 7 or 15, so a fix likely
also wants tiles covering the query lengths speculative decoding actually uses.
