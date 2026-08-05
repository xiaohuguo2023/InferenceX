# Kimi-K3 fp8-ASM prefix-heavy synthetic benchmark (MI355X, TP8)

Prefix-heavy stress test that isolates **long shared-prefix prefill + short decode**
throughput on the fp8 ASM-MLA path — distinct from the IX-CI agentic scenario
(`inferencex-agentx-mvp`, warm-per-session + cache-bust first turn). Used to
reproduce the ~9K in-tok/s/GPU headline run and to A/B serve knobs on this shape.

## Workload

aiperf **v1.0.1** (`/workspace/.aiperf_v1_0_1`), `--sweep-type zip`:

| param | value |
|---|---|
| shared prefix | 63,240 tokens × 8 distinct prefixes (`--num-prefix-prompts 8 --prompt-prefix-length 63240`) |
| unique input | 4,760 tokens (`--prompt-input-tokens-mean 4760 --stddev 0`) |
| output | 350 tokens fixed (`--output-tokens-mean 350`, `min/max_tokens 350`, `ignore_eos`) |
| concurrency | **16, 24** (`--concurrency 16,24 --request-count 80,120`) |
| warmup | `--warmup-request-count 3` · `--random-seed 42` · `--use-server-token-count` |

## Serve config (the "9K serve")

`vllm serve`, TP8, gpu-mem **0.95**, `--max-num-seqs 24`, `--moe-backend aiter`,
`--kv-cache-dtype fp8`, `--attention-backend ROCM_AITER_MLA`,
`--enable-prefix-caching --no-disable-hybrid-kv-cache-manager`,
`--compilation-config {mode:3, cudagraph_mode:FULL_AND_PIECEWISE}`.

**Critical:** use the **DEFAULT** `--max-num-batched-tokens` (do **not** set 4096 —
4096 chops the 63k prefill into ~16 chunks, roughly halving throughput and blowing
up TTFT P90). This is the one serve knob that differs from the agentic recipe.

Image deps: vLLM #50578 + PR-A (fp8 ASM MLA non-divisor 12-head) + aiter #4452
(64-bit paged-KV offsets). See `benchmarks/single_node/agentic/kimik3_fp4_mi355x_vllm.sh`.

## ⚠️ Cold vs warm cache — read before comparing

This benchmark swings **~2×** on cold vs warm prefix cache. `--warmup-request-count 3`
cannot prime 8 cold 63k-token prefixes, so a fresh serve measures **cold**
(~4.8K/6.2K in-tok/s/GPU); a serve whose prefixes are already resident measures
**warm** (~9K/12K). Same server, same config. **Always compare cold-to-cold or
warm-to-warm.** The stored runs below are cold (fresh serve each).

## Results — in-tok/s/GPU (÷8), profiling phase

### Baseline: fused OFF

| run | gpu-mem | conc16 in/GPU | conc16 TTFT / ITL | conc24 in/GPU | conc24 TTFT / ITL |
|---|---|--:|--|--:|--|
| `k3_prefixbench_fp8asm_v101_gm095` | 0.95 | **4758** | 7228 ms / 60.7 ms | **6182** | 8308 ms / 70.4 ms |
| `k3_prefixbench_fp8asm_v101`       | 0.95 | 4761 | 7533 ms / 59.8 ms | 6164 | 8274 ms / 70.7 ms |
| `k3_prefixbench_fp8asm` (earliest) | 0.80 | 4431 | 8308 ms / 63.1 ms | 5759 | 9811 ms / 72.4 ms |

*gpu-mem 0.80 ≈ 0.95 here (KV usage is only ~15–28%); the 0.95 rows are the reference.*

**Warm reference:** the original headline run measured **~9K (conc16) / ~12K (conc24)**
in-tok/s/GPU on the same serve with the prefixes already resident.

### Latest update: fused ON (`+fused_rms_norm_gated`, default)

Isolation A/B — only change vs the baseline serve is `custom_ops:["+fused_rms_norm_gated"]`
(KDA gated-RMSNorm fused custom op, now default-ON in the recipe). Everything else
held identical (moe aiter, ms24, default mnbt, gpu 0.95). Run via
`_run_prefixbench_fused.sh` → `k3_prefixbench_fp8asm_v101_fused`.

| run | conc16 in/GPU | conc16 TTFT / ITL | conc24 in/GPU | conc24 TTFT / ITL |
|---|--:|--|--:|--|
| `k3_prefixbench_fp8asm_v101_fused` | _pending_ | _pending_ | _pending_ | _pending_ |

> Chained behind the 1–24 agentic sweep (waits for GPU-free, then runs). Fill this
> row from `k3_prefixbench_fp8asm_v101_fused/concurrency_*/phases/profiling/profile_export_aiperf.json`
> once complete; compute Δ% vs the `_gm095` baseline (cold-vs-cold).

## Reproduce

```bash
# baseline (fused OFF)
OUT=/workspace/k3_prefixbench_fp8asm_v101_gm095 bash /workspace/_run_prefixbench.sh
# latest update (fused ON) — waits for any in-flight serve to free the GPU first
bash /workspace/_run_prefixbench_fused.sh
```

Both free the GPUs on exit (`_freegpu.sh`). Extract numbers with:
`profile_export_aiperf.json` → `input_token_throughput.avg / 8`,
`time_to_first_token.avg`, `inter_token_latency.avg`.
