# conc-1 interactivity gap: MI355X vs B300 (agentic, Kimi-K3 FP4, TP8)

Sources
- **B300**: IX CI run `31509160691` attempt 2 (`agent/refresh-kimik3-b300-dspark-agentx`, 2026-08-11),
  point `conc1_kvnone`. Downloaded to `_ixci_run31509160691_a2/`.
- **MI355X**: `k3_dspark_synth251_ixci_ixci_c1/aiperf_artifacts/` (2026-08-12).

## The two arms are directly comparable

B300's `vllm serve` args (from `server.log`) carry the **same** DSpark spec config we run:

```
speculative_config: {method: dspark, model: Kimi-K3-DSpark, num_speculative_tokens: 2,
                     rejection_sample_method: synthetic, synthetic_acceptance_length: 2.51}
```

measured `Mean acceptance length: 2.50`. Same dataset (`semianalysisai/cc-traces-weka-062126`),
same TP8, same conc-1, both FP4 weights. So the apples-to-apples MI355X arm is the
**synthetic-2.51** one, *not* `k3_dspark_ns2_ixci_ixci_c1` (real verify, fr-ITL p90 19.17 ms) —
using the ns2 arm would overstate the gap as 3.2x.

## Headline

Interactivity basis = `1000 / full_response_itl_p90` (the IX dashboard's own basis; the
B300 artifact's `full_response_itl` p90 0.00603 matches `k3_nv_agentic_dashboard.json`'s
`p90_itl` exactly).

| | MI355X (synth 2.51) | B300 (synth 2.51) | ratio |
|---|---|---|---|
| full_response_itl p50 | 9.228 ms | 5.60 ms | 1.65x |
| full_response_itl **p90** | **11.965 ms** | **6.03 ms** | **1.98x** |
| interactivity p90 | **83.6** | **165.8** | 1.98x |
| implied decode **step** (ITL x 2.51) | ~23.2 ms | ~14.1 ms | 1.64x |
| measured step (`inter_chunk_latency` p50) | 23.07 ms | — | — |
| tail ratio p90/p50 | 1.30 | 1.08 | — |
| TTFT p50 / p90 | 788 / 1695 ms | 1102 / 2296 ms | **MI355X wins** |
| ISL mean / p50 | 162.6k / 120.8k | 221.0k / 169.4k | B300 longer |
| prefix cache hit | 94.9% (96.5% theor.) | 97.4% theor. | — |

Two components: a **1.65x slower median step** plus a **1.20x wider tail**.

## The gap is a flat per-step cost, not context-dependent

Per-request (B300 `profile_export.jsonl`) / per-timeslice (MI355X `..._timeslices.json`),
bucketed by context length — median `full_response_itl` in ms:

| ISL bucket | MI355X | B300 | ratio |
|---|---|---|---|
| 0–50k | 10.31 | 6.28 | 1.64x |
| 50–100k | 10.35 | 4.98 | 2.08x |
| 100–200k | 9.01 | 5.13 | 1.76x |
| 200–300k | 9.11 | 5.46 | 1.67x |
| 300–500k | 9.22 | 5.76 | 1.60x |

Both curves are essentially **flat from 27k to 390k tokens of context**. That rules out:

- **attention / KV length** — if MLA decode scaled worse on our side the ratio would climb
  with the bucket; it does not (it is highest in the 50–100k bucket and lowest at 300–500k).
- **prefix-cache behaviour** — 94.9% vs 97.4%, and TTFT is *better* on MI355X, so prefill and
  cache are not paying the bill.
- **metric basis** — same `full_response_itl` on both sides (MI355X raw `inter_token_latency`
  p90 is 10.43 vs full-response 11.97, only ~13% apart; the plot script's basis mismatch is
  not the cause, though it does flatter MI355X by ~1.15x and should be fixed).
- **acceptance** — both synthetic 2.51, both measured ~2.50.

What is left is a roughly **constant ~9–10 ms of extra work inside the decode step itself**
(~23 ms vs ~13 ms), independent of sequence length. That is the target-verify model forward.

## Where the ~10 ms lives (from our own conc-1 decomposition)

Consistent with what we already measured on MI355X at conc-1:

- target verify cudagraph = **92%** of the step (`k3-conc1-step-decomp-target-graph-92pct`)
  — the lever is kernel work inside the graph, not orchestration;
- MoE ≈ **33%** of the step, with shared-expert bf16 GEMMs landing in the gfx950
  `6 <= n <= 9` skinny dead zone (`k3-conc1-moe-breakdown-shared-expert-deadzone`, PR-plan P6);
- rocprofv3 has already ruled out rank-imbalance and KDA kernel perf
  (`k3-conc1-decode-bottleneck-kda-laggard`), and the fused all-reduce+RMSNorm lever is
  conc-1 neutral (`k3-aiter-ar-rms-fusion-conc1-neutral`).

## Methodology asymmetry worth recording

B300 tailors the server per concurrency point: `max_num_seqs: 2` and
`cudagraph_capture_sizes: [3, 6]` (`max_cudagraph_capture_size: 6`) — i.e. the ladder is
exactly the multiples of `decode_query_len=3`, so every decode step replays an exact-size FULL
graph and vLLM #53407 can never bite them. We run `MAX_NUM_SEQS=64` with a 45-entry ladder and
a 32 GiB KV pin. At conc-1 this does not change our dispatch (M=3 does get a FULL graph), but
the two rigs are not configured the same way, and this should be stated whenever the Pareto
plot is presented.

Also: B300 is not fully tuned either — its own log warns
`No tuned config covers trtllm_batch_decode_mla input_shapes=((1,1,12,576),(1,16392),...)
This shape is outside the tuning bucket range`.

## Conclusion

The conc-1 interactivity p90 gap is **1.98x**, and it is a **flat per-decode-step compute gap
(~23 ms vs ~14 ms)** in the target verify forward — not attention, not context length, not
prefill, not acceptance, not the metric. The existing P6 shared-expert / skinny-GEMM work is
the right lever; nothing here points at a new one.
