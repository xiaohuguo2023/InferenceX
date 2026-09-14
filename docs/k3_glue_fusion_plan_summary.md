# Glue / elementwise fusion at conc-1 — where we are

Scope: the non-GEMM, non-attention "Other" cost in the conc-1 decode step —
elementwise kernels, copies, norms/quant, and the metadata build that precedes
the model forward.

## 1. The size of the prize

From `docs/k3_vs_b300_conc1_isl100k.md` (ISL-100k, conc-1, kernel-sum, KDA-clock
normalised):

| component | MI355X | B300 | ratio | calls (MI355X / B300) | % of gap |
|---|---:|---:|---:|---:|---:|
| Glue/elementwise/misc | 997 us | 187 us | **5.34x** | 232 / 87 | 12% |
| Norm/quant | 863 us | 263 us | **3.28x** | 199 / 102 | 9% |
| Memory/copy | 347 us | 59 us | **5.84x** | 85 / 28 | 4% |

**These are the three worst ratios in the whole table.** Together 2,207 us/step,
25% of the kernel-sum gap, on ~2.4x B300's launch count. By contrast the big-us
components are already at parity — Dense GEMM 1.05x, MoE expert GEMM 1.08x, MoE
routing/sort 1.27x. The headroom is in launch-heavy glue, not in the big kernels.

## 2. The finding that decides the approach

Every glue kernel averages **3.8–4.2 us**, and the fastest kernel *anywhere* in
the trace is **2.25 us**. At conc-1 these tensors are a handful of elements, so
the cost is the **dispatch floor, not bandwidth**.

**Consequence: grid/tile tuning cannot help.** It changes workgroups *within* one
launch. This closes the standing "can we pick a better tile size for the fused
elementwise kernel?" question with a measured **no**. The only lever is
**launching fewer kernels**.

The top glue kernel is a single `elementwise_kernel_manual_unroll` at 365 us over
77 calls — 4.7 us/call for work that is almost nothing. That shape repeats all the
way down the table.

## 3. Where the launches come from

Attribution of every glue kernel to its innermost vLLM Python frame:

| bucket | launches/iter | us/iter | in cudagraph? |
|---|---:|---:|---|
| A1 `gdn_attn.build` x14 | 154 | 533 | no |
| A2 `mamba_get_block_table_tensor` x14 | 98 | 381 | no |
| A3 `rocm_aiter_mla._build_decode` x5 | 50 | 482 | no |
| A4 `async_tensor_h2d` | 14 | 47 | no |
| **A — metadata build** | **316** | **1,443** | **no** |
| B1 bf16 `direct_copy` | 139 | 572 | yes |
| B2 fp8 `float8_copy_kernel_cuda` | 48 | 236 | yes |
| **B — copies** | **187** | **808** | **yes** |

**Why x14 and x5.** Not a target/draft double pass — it is the **KV-cache group
count**. K3 is 93 layers = 69 KDA + 24 MLA; the hybrid allocator forces equal
layers per group and picked group_size 5, giving 14 KDA groups + 5 MLA groups.

**The redundancy.** `build_attn_metadata` loops over groups, but only
`block_table_tensor` and `slot_mapping` are genuinely per-group.
`query_start_loc`, `seq_lens`, `num_accepted_tokens` are shared by all 14. Per
call, `torch.arange(1 + num_speculative_blocks)` is a **compile-time constant**
re-materialised on GPU, and `((seq_lens-1)//block_size).clamp_(min=0)` is
identical for every group.

Bucket A is essentially *all* of the eager remainder of the step (the step is
92.9% cudagraph-replayed), so it costs host launch latency on top of GPU time.

## 4. What is already closed

| lever | verdict |
|---|---|
| tile/grid tuning of glue kernels | **dead** — dispatch floor, see §2 |
| `#52968` elementwise fusion | **won**, −1.1% ITL p50 at conc-1 and conc-12 — this is the calibration for what one increment is worth |
| `+situ_and_mul` custom op | **won** — 43% faster than inductor's decomposition at the conc-1 shape |
| MoE tail AR+RMS fusion | **regression** — measured |
| aiter fused all-reduce + RMSNorm | **neutral** at conc-1 |
| attn_res launch tuning / ATOM configs | **dead** — 1.00x |
| KDA out of `splitting_ops` | **dead** — zero effect |

## 5. What is staged now

**The torch.compile / Inductor-fusion A/B** (`_ab_compile_run.sh`, two arms).

The reason this is live: **K3's fusion passes are inert today.** Reading
`vllm/config/vllm.py::enable_rope_kvcache_mla_fusion`:

```
use_inductor_graph_partition OR not splitting_ops_contain_kv_cache_update()
```

At `-O3` with `splitting_ops=None` and `fuse_attn_quant=IS_QUANTIZED`
(hard-coded `False`, vLLM #25689), `set_splitting_ops_for_v1` appends
`unified_kv_cache_update` + `unified_mla_kv_cache_update`, so the predicate is
True and `fuse_rope_kvcache_cat_mla` resolves to **False**. That is the
transitive veto this plan predicted, confirmed in code.

| arm | change | why |
|---|---|---|
| 1 | `use_inductor_graph_partition: true` — **config only, zero code** | unlocks the pass; isolates config from decorator, which upstream PR #56664 could not do |
| 2 | + #56664's `@support_torch_compile` on `K3DSparkModel` | compiles the draft graph so the passes have something to act on |

Upstream #56664 measured **+6.7% intvty p90 / −6.4% ITL p90** at conc-1 on
MI355X, but at depth 3 without DCP, over 1200 s (n≈68), and it moved the
decorator and the config together so the delta cannot be attributed. Treat +6.7%
as an **upper bound**, not a forecast.

**Important scope limit: this cannot touch bucket A.** torch.compile acts on the
model forward; the metadata build runs in the model runner, *before* it. So arms
1 and 2 address bucket B (808 us) at best, never the 1,443 us of bucket A.

## 6. What is untouched — the real remaining work

**Stage 2: hoist the group-invariant metadata work.** Value-preserving by
construction (identical inputs → identical outputs), so no numerical risk:

1. **Memoize the constant** — `torch.arange(1 + num_speculative_blocks)` never
   changes. −13 launches/iter for free.
2. **Compute `start_indices` once per pass** rather than per group. Turns
   7 launches x N groups into 6 + N.
3. **Share the group-invariant cudagraph-padding buffers** — of 6 `copy_` +
   5 `fill_` in `gdn_attn.build`, only `spec_state_indices_tensor` is
   group-dependent.
4. **Same for `_build_decode`** — the `cumsum`/`cat` `paged_kv_indptr`,
   `qo_indptr` and `paged_kv_last_page_len` are shared across MLA groups.

Rough ceiling: **316 → ~60 launches/iter, ~1.0–1.1 ms/iter, 3.2–3.5% of the
step** — and it is eager time, so it also removes host launch latency.

**Not taken:** reducing the KV-cache group count in `kv_cache_utils.py` would
divide bucket A by ~5x, but it trades KV memory and is deep hybrid-allocator
surgery. Recorded as an option, not a plan.

## 7. Order and verification

1. Arm 1 (config only) — cheapest, zero code, and isolates the config.
2. Arm 2 (+ decorator) — also serves as the compiled-draft x cprr composition test.
3. Stage 2 (metadata hoisting) — the largest untouched item, and the only one
   that reaches bucket A.

Standing rules for every arm: **e2e only, no microbenchmark conclusions**;
conc-1 at the full **3600 s**; report ITL p50/p90 and intvty p90 computed as
`frITL = full_decode_duration / OSL` (not aiperf's `inter_token_latency` — they
differ ~9%); print `request_count` and OSL beside every delta or the comparison
is contaminated; GSM8K in BLOCK mode after any fusion change.
