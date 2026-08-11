# Kimi-K3 Attention Benchmark — BASELINE (no speculative decoding)

**Date:** 2026-08-08
**Machine:** MI355X ×8 (gfx950), TP8
**Container:** `xguo-k3nightly` — `vllm/vllm-openai-rocm:nightly-cb8104839c...`, vLLM 0.26.1rc1.dev306, ROCm 7.2.3
**Model:** `moonshotai/Kimi-K3` FP4 (a8w4 MoE), snapshot `9f62e4e9...`, staged in `/dev/shm/hf-cache`
**Instructions followed:** `K3_Attention_Benchmark_Instructions.md` (config adapted to the nightly's stable envelope — see Deltas).

This is the **baseline** run (no DSpark spec decoding). Spec-2 / spec-7 are deferred to a follow-up per the "baseline only first" decision.

---

## 1. Correctness — GSM8K (accuracy gate)

Full GSM8K test set via lm-eval-harness `local-chat-completions` against the served endpoint
(1319 questions, num_fewshot=5, concurrency=64, chat template applied, temperature=0, max_tokens=3072).

| Filter | n-shot | exact_match | stderr |
|---|---|---|---|
| flexible-extract | 5 | **0.9682** | ±0.0048 |
| strict-match | 5 | **0.9666** | ±0.0049 |

**PASS.** K3 is a thinking model (emits `<think>…</think>` via the `kimi_k3` reasoning parser);
flexible-extract is the headline. Both filters agree, confirming the nightly-adapted config —
in particular the **a8w4 MoE on the correct `afp8` FlyDSL path** (not the silent bf16 fallback) —
is numerically correct. (Smoke pre-check LIMIT=20 gave 0.90.)

---

## 2. Performance — long-context concurrency sweep

**Workload:** ~68K ISL = 63,911-token shared prefix + 4,089-token synthetic suffix, **OSL=350 exactly** (`ignore_eos`, min=max=350). aiperf 0.12.0, `--sweep-type zip`, warmup 16/cell, streaming, `--use-server-token-count`. Realized ISL≈68,089 / OSL=350 on every cell (verified).

| conc | req | TTFT ms (P50/P90/Mean) | ITL ms (P50/P90/Mean) | out tok/s | out tok/s/GPU | total tok/s | total tok/s/GPU | reqLat P50 (s) | prefix-cache hit % |
|---|---|---|---|---|---|---|---|---|---|
| **48 (warm)** | 240 | 1337.4 / 2017.1 / 1412.2 | 46.7 / 48.7 / 46.7 | **946.9** | **118.4** | 185155.5 | 23144.4 | 17.65 | 99.3 |
| 48 (cold) ‡ | 240 | 2011.9 / 18246.2 / 5204.5 | 99.1 / 102.2 / 94.2 | 438.4 | 54.8 | 85718.2 | 10714.8 | 36.60 | 91.8 |
| 32 | 160 | 1405.2 / 4351.9 / 1831.2 | 38.8 / 40.2 / 38.9 | **726.4** | **90.8** | 142040.2 | 17755.0 | 14.77 | 98.7 |
| 24 | 120 | 1123.3 / 1409.9 / 1022.7 | 35.7 / 36.9 / 36.0 | 616.9 | 77.1 | 120633.0 | 15079.1 | 13.60 | 99.3 |
| 16 | 80  | 728.4 / 1032.3 / 824.8   | 32.8 / 33.0 / 32.6 | 458.6 | 57.3 | 89670.2  | 11208.8 | 12.18 | 99.3 |
| 12 | 60  | 698.9 / 891.9 / 722.8    | 32.7 / 32.8 / 32.6 | 346.6 | 43.3 | 67780.2  | 8472.5  | 12.13 | 99.3 |
| 8  | 40  | 613.9 / 695.0 / 585.9    | 29.3 / 30.2 / 29.4 | 257.9 | 32.2 | 50428.0  | 6303.5  | 10.84 | 99.3 |
| 4  | 20  | 443.4 / 523.8 / 410.0    | 25.0 / 25.5 / 25.1 | 152.6 | 19.1 | 29846.4  | 3730.8  | 9.16  | 99.3 |
| 2  | 10  | 357.6 / 441.8 / 352.3    | 24.0 / 24.3 / 24.0 | 80.1  | 10.0 | 15655.7  | 1957.0  | 8.72  | 99.2 |
| 1  | 5   | 257.9 / 274.4 / 257.9    | 22.8 / 22.9 / 22.8 | 42.5  | 5.3  | 8313.5   | 1039.2  | 8.22  | 99.2 |

**‡ conc48 (cold) ran first with a cold prefix cache** — its very first requests paid the full 68K-token prefill
(TTFT max ~30.8s, P90 18.2s), dragging mean TTFT and throughput. The **conc48 (warm)** row is the re-run
against the now-warm server and is the value to use; the cold row is retained only to show the cold-start penalty.

### Reading the numbers
- **Peak decode throughput: conc48 (warm) → 946.9 tok/s (118.4 tok/s/GPU).** Warming the shared prefix moved the peak from conc32 (726 tok/s) up to conc48; throughput is still climbing at the seqs=64 ceiling, so the true knee is at/above 48 (bounded by max-num-seqs).
- **`total tok/s` is inflated by prefix caching** — it counts the 68K cached input tokens as "throughput" even though 99% are cache hits and never recomputed. **Use `out tok/s` as the honest work-done metric.** `total`/GPU is reported only because the doc asks for it.
- **ITL is flat and low (23–40 ms P50)** across the warm cells and rises only at conc48 (99 ms) — MLA decode scales cleanly up to the seqs=64 workspace ceiling.
- **TTFT scales as expected** with concurrency on warm cache: 258 ms (conc1) → 1.4 s P50 (conc32).
- **Prefix-cache hit ≥98.7%** on all warm cells — the shared-prefix workload is doing what it should.

---

## 3. Config — deltas vs the doc's Reference Server Command

All deltas forced by the nightly's stable envelope (documented in `docs/k3-nightly-migration`); everything else carried through verbatim.

| Knob | Doc | This run | Why |
|---|---|---|---|
| gpu-memory-utilization | 0.93 | **0.88** | nightly reserves ~10.7 GB/GPU CUDA-graph pool not subtracted from KV sizing |
| max-num-seqs | 128 | **64** | MLA chunked-prefill workspace floor (seqs×block 1536) overrides the 64K cap above this |
| max-num-batched-tokens | 16384 | **4096** | paired with seqs=64 to stay under `HSA_STATUS_ERROR_OUT_OF_RESOURCES` |
| cudagraph_mode | FULL_DECODE_ONLY | **FULL_AND_PIECEWISE** | proven-up nightly mode |
| a8w4 flags | `AITER_SITUV2_A8W4` | **+ `VLLM_ROCM_USE_AITER_MOE_SITUV2_A8W4`** | BOTH required or MoE silently falls back to bf16 |
| kv-cache-dtype | (default) | **fp8** | MI355X production path |
| attention-backend | (default) | **ROCM_AITER_MLA** | MI355X MLA path |
| — | — | **VLLM_USE_BREAKABLE_CUDAGRAPH=0** | nightly stability |

Carried through: TP8, `--max-model-len 1048576`, `--async-scheduling`, `--enable-prefix-caching`,
`--enable-prompt-tokens-details`, `--mm-encoder-tp-mode data`, `+fused_rms_norm_gated`,
`kimi_k3` reasoning+tool parsers, `--enable-auto-tool-choice`.

**aiperf flag fix:** the doc's `--synthetic-input-tokens-mean/-stddev` do not exist in aiperf 0.12.0 —
replaced with `--prompt-input-tokens-mean/-stddev` (same semantics).

---

## 4. Artifacts

- Serve: `_serve_k3_bench_baseline.sh` → `/workspace/serve_k3_bench_baseline.log`
- GSM8K: `_gsm8k_k3.sh` → `/workspace/gsm8k_k3_baseline/` (accuracy above)
- Sweep: `_sweep_k3_longctx_350.sh` → `/workspace/k3_longctx_sweep_baseline/conc*/` (per-cell aiperf export + prefix-cache before/after)
- Parser: `_parse_k3_sweep.py`

## 5. Follow-ups (deferred / recommended)
1. **DSpark spec-2 and spec-7** (`Inferact/Kimi-K3-DSpark` draft) — the deferred half of the benchmark; adds "average acceptance length" to the report.
2. **Warm re-run of conc48** so the top of the sweep is comparable (its cold-cache TTFT/throughput is an artifact of running first).
3. **Raise max-num-batched-tokens above 4096** toward the doc's 16384 intent (OOM was driven by max-num-seqs, not mnbt) to pull TTFT down at high concurrency — after confirming stability.
