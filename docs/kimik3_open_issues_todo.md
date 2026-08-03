# Kimi-K3 MI355X — open issues & todo list

Consolidated from the K3 bring-up docs. Check items off as they land; link PRs / runs in the **Notes** column.

**Last synced:** 2026-07-31

---

## P0 — blockers (upstream / cap)

| Status | Item | Source | Notes |
|:---:|---|---|---|
| [ ] | **Fix K3 dense-MLA long-context cudagraph capture GPU fault (≥128K).** `TRITON_MLA` fp8 and bf16 gluon both fault during FULL/PIECEWISE capture at ≥128K; fine ≤64K. `--enforce-eager` avoids fault but tanks TPOT on 93-layer 2.8T model. | [baseline](kimik3_mi355x_agentic_baseline.md), [ref vs ours](kimik3_ref_config_vs_ours.md) | vLLM/AITER. Until fixed, agentic recipe stays **64K-capped**. |
| [ ] | **Lift the 64K `max-model-len` cap** once capture bug is fixed. | [baseline](kimik3_mi355x_agentic_baseline.md) | Unblocks full cc-trace context (~131K avg on B300). |
| [ ] | **AITER gluon fp8 MLA: batched decode (`batch>1`).** `ROCM_AITER_MLA` / gluon fp8 asserts `batch_size==1` on `mla_gluon[bh16bn128]`. Workaround today: force **`TRITON_MLA`** for fp8 agentic. | [baseline](kimik3_mi355x_agentic_baseline.md), [tutorial §Part 2](kimik3_beginner_tutorial.md) | Upstream AITER/vLLM; or stay on TRITON_MLA. |

---

## P1 — measurement & apples-to-apples benchmarking

| Status | Item | Source | Notes |
|:---:|---|---|---|
| [ ] | **Add `--enable-prompt-tokens-details`** to serve recipe. Reference bf16 run reported `cached_tokens absent` — can't confirm 63K prefix is actually cached; TTFT 8.5–10.2 s may be full re-prefill. | [ref vs ours](kimik3_ref_config_vs_ours.md) | Re-check prefix-cache hit + TTFT after adding flag. |
| [ ] | **Full conc 1–24 sweep** (B300 range) once 64K cap is lifted. | [baseline](kimik3_mi355x_agentic_baseline.md) | Partial sweep exists (`k3_sweep_c{1,4,8,16,24}/`); need post-cap rerun for fair MI355X vs B300 curve. |
| [ ] | **Document bf16 vs fp8 KV trade-off** in serve recipe: bf16 = native context, slow decode (~100–110 ms TPOT @68K); fp8 = ~2× faster KV read/decode but ≤64K until capture fix. | [ref vs ours](kimik3_ref_config_vs_ours.md) | Recipe flag: `KV_CACHE_DTYPE=auto` → bf16, `=fp8` → TRITON_MLA + 64K cap. |

---

## P2 — performance optimization (our stack)

| Status | Item | Source | Notes |
|:---:|---|---|---|
| [ ] | **Dense / MLA-proj GEMM tuning (~22% GPU time).** Server logged **8,512** hipBLASLt `"not found tuned config … using torch/default"` fallbacks. Target: tuned + weight-preshuffled path (CK `b_preshuffle` / AITER `fp8gemm…BpreShuffle`) — same pattern as DSV4 Target-#7. | [profile](kimik3_fp4_mi355x_conc32_vllm_profile.md) | bf16 `Cijk_` today; not preshuffled. |
| [ ] | **Investigate R0/R1 vs R2–R7 elementwise + norm skew (~13% + ~11%).** Decode end-to-end is balanced (0.3%), but AR spin-wait masks a 2-vs-6 rank-group pattern on elementwise/norm/memcpy — possible NUMA/XCD placement asymmetry. | [tp8 rank imbalance](kimik3_tp8_rank_imbalance.md) | Fixing skew lowers effective AR wait, not AR kernel itself. |
| [ ] | **KDA linear-attention kernel path (~5–6%).** All Triton (fla) today; candidate for AITER/CK fusion if decode share grows. | [profile](kimik3_fp4_mi355x_conc32_vllm_profile.md) | K3-specific; not present in DSV4. |
| [ ] | **Mixed-batch / piecewise graph CPU bubbles.** `splitting_ops` (`unified_mla_attention_with_output`, `kda_attention`, KV updates) force eager CPU launch → ~17 µs idle before each AR; ~35–60 ms stalls before MLA prefill. Mostly prefill/mixed-batch; FULL graph may cover pure decode. | [cpu bubble analysis](kimik3_cpu_bubble_graphcapture_analysis.md) | See graph-capture improvement items below. |
| [ ] | **Add `@eager_break_during_capture` to `vllm::kda_attention`.** MLA op already has it; K3's KDA op (`kimi_gdn_linear_attn.py:43`, ~69 of 93 layers) does not, which blocks `VLLM_USE_BREAKABLE_CUDAGRAPH=1`. Precondition already met (`mutates_args=["core_attn_out"]`). Then A/B breakable vs piecewise. | [cpu bubble analysis §Lever 1](kimik3_cpu_bubble_graphcapture_analysis.md) | ~1-line upstream vLLM PR. **Validate correctness first** — KDA holds recurrent state. |
| [ ] | **Try `use_inductor_graph_partition: true`.** Stops vLLM appending `unified_kv_cache_update` + `unified_mla_kv_cache_update` to `splitting_ops` (2 fewer breaks × 93 layers) and unlocks `fuse_rope_kvcache` / `fuse_qk_norm_rope_kvcache`, currently auto-disabled with a warning. | [cpu bubble analysis §Lever 2](kimik3_cpu_bubble_graphcapture_analysis.md) | Needs torch ≥ 2.9.0.dev. Grep server log for the "Disabling fuse_rope_kvcache" warning to confirm. |
| [ ] | **Check `TRITON_MLA` graph-coverage cap before DSpark.** `TRITON_MLA` declares `UNIFORM_SINGLE_TOKEN_DECODE` (level 1) and vLLM takes the min across backends → FULL graphs only at query_len==1. Spec decode (query_len = 1+num_spec) would drop to PIECEWISE. | [cpu bubble analysis §Lever 3](kimik3_cpu_bubble_graphcapture_analysis.md) | Tied to P0 gluon fp8 batch=1 assert; `ROCM_AITER_MLA` is level 2. |
| [ ] | **Verify cudagraph capture-size coverage.** `max_cudagraph_capture_size` = `min(max_num_seqs*2, 512)`; our `MAX_NUM_SEQS=2×CONC` → at CONC=4 capture sizes only ≤16 (FULL ≤8). Confirm agentic batches aren't dispatching to eager `NONE`. | [cpu bubble analysis §Lever 4](kimik3_cpu_bubble_graphcapture_analysis.md) | Compare with gpt-oss `widegraph` win on MI355X. |

---

## P3 — profiling & analysis follow-ups

| Status | Item | Source | Notes |
|:---:|---|---|---|
| [ ] | **Decode-only segmented profile.** Current conc32 8k/64 window is prefill-dominated → comm (34%) and dense GEMM (22%) are upper bounds vs steady decode. | [profile](kimik3_fp4_mi355x_conc32_vllm_profile.md) | Separate pass with no prefill in window. |
| [ ] | **Confirm FULL CUDA graph removes per-AR bubble on pure decode** (batch ≤32). Hypothesis: piecewise/mixed-batch bubbles don't apply to agentic steady decode. | [cpu bubble analysis](kimik3_cpu_bubble_graphcapture_analysis.md) | Compare gap→kernel on decode-only trace. |
| [ ] | **Fix / work around trace export crash at full OSL.** conc32×8k/1024 crashes worker during trace serialization (`Executor failed` on `/stop_profile`); profiles limited to OSL=64. | [profile](kimik3_fp4_mi355x_conc32_vllm_profile.md) | Kernel names repeat per decode step; export still needed for full-window timing. |
| [ ] | **JIT / warmup before profiling.** `hipModuleLoadDataEx` cold-start inflates eager-MLA prefill stalls (~55 ms in bubble windows). | [cpu bubble analysis](kimik3_cpu_bubble_graphcapture_analysis.md) | Warmup run before capture. |
| [ ] | **Re-measure comm share at agentic serving point** (conc4, ~40K in, variable out) — not just synthetic 8k/1k profile. | [profile](kimik3_fp4_mi355x_conc32_vllm_profile.md), [baseline](kimik3_mi355x_agentic_baseline.md) | Agentic TPOT p50 22 ms vs B300 13.1 ms gap driver. |

---

## Resolved / documented workarounds (keep for reference)

These were issues during bring-up; recipe already encodes the fix. No open action unless regressions appear.

| Item | Workaround in recipe |
|---|---|
| Hybrid KV manager required (MLA + KDA) | `--no-disable-hybrid-kv-cache-manager` |
| AITER fp8 KV layout | `VLLM_ROCM_SHUFFLE_KV_CACHE_LAYOUT=1` |
| Model name mismatch → 404 / warmup abort | Serve name must match AIPerf `--model` |
| Inter-turn socket close → warmup abort | `VLLM_HTTP_TIMEOUT_KEEP_ALIVE=900`, `AIPERF_HTTP_TCP_USER_TIMEOUT=900000` |
| Capture stability | `VLLM_USE_BREAKABLE_CUDAGRAPH=0`, `FULL_AND_PIECEWISE` |
| 64K cap (until upstream fix) | `max-model-len` capped in `benchmarks/single_node/agentic/kimik3_fp4_mi355x_vllm.sh` |

---

## Doc index (where each issue was first raised)

1. [kimik3_mi355x_agentic_baseline.md](kimik3_mi355x_agentic_baseline.md) — enablement findings, open items
2. [kimik3_ref_config_vs_ours.md](kimik3_ref_config_vs_ours.md) — bf16 vs fp8, prefix-cache measurement gap
3. [kimik3_fp4_mi355x_conc32_vllm_profile.md](kimik3_fp4_mi355x_conc32_vllm_profile.md) — kernel breakdown, optimization targets
4. [kimik3_cpu_bubble_graphcapture_analysis.md](kimik3_cpu_bubble_graphcapture_analysis.md) — graph capture / CPU bubbles
5. [kimik3_tp8_rank_imbalance.md](kimik3_tp8_rank_imbalance.md) — rank skew / AR spin-wait
6. [kimik3_beginner_tutorial.md](kimik3_beginner_tutorial.md) — algorithm context for the above
