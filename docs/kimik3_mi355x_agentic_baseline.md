# Kimi-K3 MI355X — first agentic (AIPerf) baseline + fp8/MLA enablement notes

## Result — K3 MI355X vLLM agentic, conc=4, fp8 KV, captured @64K
AIPerf `inferencex-agentx-mvp` replay (cc-traces-weka-062126), fast mode (1200 s), 107 requests.

| Metric | MI355X (this run) | B300 (dashboard, conc4 agentic, offload-on) |
|---|---|---|
| Prefix-cache hit (theoretical) | **93.4%** | 95.7% |
| Request throughput | 0.089 req/s | — |
| Output-token throughput | 44.9 tok/s (aggregate) | — |
| **Interactivity** (out tok/s/user) | **47.7** (p50 45.5) | — |
| **TTFT** | avg **721 ms**, p50 569, p90 1060 | 830 ms |
| **Inter-token latency (TPOT)** | avg **22.4 ms**, p50 22.0, p90 25.9 | 13.1 ms |
| E2E request latency | avg 12.3 s, p50 5.1 s | — |
| Input seq len | avg 40.1K, p50 45.5K (**64K cap**) | ~131K (full) |
| Output seq len | avg 503, p50 190 | — |

**Caveat:** MI355X is capped at **64K context** (see below), so it processes ~3× shorter context than B300's full run — not fully apples-to-apples. TTFT is comparable; per-token B300 is ~1.7× faster here.

## Enablement findings — fp8 KV + MLA on K3 (dense MLA) MI355X
Getting the agentic path to run at all required working through several ROCm/K3 issues:

1. **fp8 KV decode backend.** K3's attention is **dense MLA** (unlike DSV4's *sparse* MLA):
   - `ROCM_AITER_MLA` / `ROCM_AITER_TRITON_MLA` (gluon) fp8 = `mla_gluon[bh16bn128]` → asserts **batch_size==1** (unusable batched).
   - **`TRITON_MLA`** is the only backend giving **batched fp8** decode. → recipe forces it for fp8.
2. **Hybrid KV manager.** K3 = MLA + KDA (linear) layers. `--no-disable-hybrid-kv-cache-manager` (as DSV4) allocates layer types natively (else "Add N padding layers").
3. **Long-context capture GPU fault (the wall).** `TRITON_MLA` fp8 (and bf16 gluon) **GPU-memory-fault during cudagraph capture at ≥128K context** (both FULL and PIECEWISE) — a K3 long-ctx capture bug, independent of KV dtype / hybrid manager. Captures fine ≤64K.
   - `--enforce-eager` avoids it but tanks TPOT on this 93-layer 2.8T model → rejected.
   - **Fix taken: cap served context at 64K** → stays **captured/fast** (FULL_AND_PIECEWISE).
4. **`VLLM_ROCM_SHUFFLE_KV_CACHE_LAYOUT=1`** — AITER fp8 KV layout (as MiniMax-M3).
5. **Model-name match** — agentic AIPerf uses `--model` for both API field and tokenizer; served name must match (path or HF id consistently), else 404 → warmup abort.
6. **Keep-alive** — `VLLM_HTTP_TIMEOUT_KEEP_ALIVE=900` + `AIPERF_HTTP_TCP_USER_TIMEOUT=900000` (avoid the inter-turn socket-close race → warmup abort).

### How other MI355X models compare
- **DSV4:** `ROCM_AITER_MLA_SPARSE` (sparse MLA) → fp8 batched works natively; not available to K3 (dense MLA).
- **MiniMax-M3** (hybrid): `TRITON_ATTN` (not MLA) + `--linear-backend emulation` + `SHUFFLE_KV_CACHE_LAYOUT` — runs fp8 captured; different attention type, so no long-ctx MLA-capture fault.

## Open items

→ Full checklist: [kimik3_open_issues_todo.md](kimik3_open_issues_todo.md)

- **Lift the 64K cap** — needs the K3 dense-MLA long-context cudagraph-capture GPU fault fixed (vLLM/AITER). Until then, agentic runs at ≤64K context (truncates the longest traces).
- Sweep conc 1–24 (B300 range) once the context cap is lifted, for a full apples-to-apples curve.

## Recipe / artifacts
- Recipe: `benchmarks/single_node/agentic/kimik3_fp4_mi355x_vllm.sh` (KV_CACHE_DTYPE=fp8 default → TRITON_MLA + shuffle-KV + 64K cap; `=auto` → bf16).
- Result: `kimik3_agentic_baseline_c4_fp8_64k.json` (+ `aiperf_artifacts/`).
- Run: `MODEL=moonshotai/Kimi-K3 TP=8 CONC=4 KV_OFFLOADING=none TOTAL_CPU_DRAM_GB=0 DURATION=1200 AIPERF_EXPERIMENTAL_FAST=1 RESULT_DIR=... bash benchmarks/single_node/agentic/kimik3_fp4_mi355x_vllm.sh`
