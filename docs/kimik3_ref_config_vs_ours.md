# Kimi-K3 MI355X — reference config vs ours (serve + client)

> **Open issues todo:** [kimik3_open_issues_todo.md](kimik3_open_issues_todo.md)

Both run on the **same box** (xguo-k3, 8× MI355X, TP8, image `vllm/vllm-openai-rocm:kimi-k3`).

## Serve config diff
| | **Reference** | **Ours (agentic recipe)** |
|---|---|---|
| **KV cache** | **bf16** (default) | **fp8** (`--kv-cache-dtype fp8`) |
| attention backend | auto → `ROCM_AITER_MLA` (gluon) | forced `TRITON_MLA` (fp8 batched) |
| max-model-len | **native (~1M)** | **capped 64K** (fp8 capture-fault workaround) |
| cudagraph | default FULL_AND_PIECEWISE | same |
| gpu-mem-util | 0.95 | 0.85 |
| max-num-seqs | 128 | 2×conc |
| model | full multimodal (`--mm-encoder-tp-mode data`) | `--language-model-only` |
| env | AITER + `VLLM_USE_BREAKABLE_CUDAGRAPH=0` | same |

**Key result: the reference bf16 serve captures cleanly at native (~1M) context — no memory fault.**
That's the simpler working path: bf16 sidesteps *both* K3 fp8 problems (the `mla_gluon[bh16bn128]`
batch=1 assert **and** the ≥128K cudagraph-capture GPU fault) that forced our fp8 route onto
TRITON_MLA + a 64K cap.

## Client / workload diff — different benchmarks
| | **Reference (synthetic)** | **Ours (agentic replay)** |
|---|---|---|
| client | public `aiperf` 0.11 | AIPerf `--scenario inferencex-agentx-mvp` |
| workload | 8 shared prefixes × **63,240** tok + **4,760** new + **350** out (fixed) | real cc-traces (variable) |
| input len | **68,089** (fixed) | ~40K mean (64K-capped) |
| output len | **350** (fixed, ignore_eos) | ~503 mean (model-driven) |
| conc | zip 16/24 (80/120 reqs) | 1/4/8/16/24 (duration) |

## Results (this box, TP8)
**Reference config** (bf16 serve + synthetic 63K prefix + 4.76K new + 350 out):
| conc | req/s | out tok/s (/gpu) | TPOT | interactivity | TTFT | in/out |
|---:|---:|---:|---:|---:|---:|---:|
| 16 | 0.357 | 124.9 (15.6) | 102.0 ms | 9.8 | **8,475 ms** | 68089/350 |
| 24 | 0.488 | 170.8 (21.3) | 109.8 ms | 9.1 | **10,194 ms** | 68089/350 |

**Our config** (fp8/64K + cc-traces agentic), for reference (different workload):
| conc | TPOT | interactivity | tput/gpu | in/out |
|---:|---:|---:|---:|---:|
| 16 | 152.1 ms | 6.6 | 896 | ~40K/503 |
| 24 | 60.0 ms | 16.7 | 2385 | ~40K/503 |

## Observations
1. **bf16 at native context works** — the reference's headline advantage; our fp8 path is capture-limited to 64K.
2. **Reference TTFT is very high (8.5–10.2 s)** at 68K input. AIPerf warned `cached_tokens absent` — the
   server isn't *reporting* prefix-cache reads (needs **`--enable-prompt-tokens-details`**), so we can't
   confirm the 63K prefix is being served from cache. If it isn't hitting, every request re-prefills ~68K,
   which explains the TTFT. **Action: add `--enable-prompt-tokens-details` and re-check cache hit / TTFT.**
3. **Reference TPOT ~100–110 ms** — 68K-context **bf16** KV decode is memory-bound (large KV read/token).
   **fp8 KV would roughly halve that read** → faster decode — but fp8 hits the K3 ≥128K capture bug. So
   there's a real tension: **bf16 = full context but slow decode; fp8 = faster decode but ≤64K**.
   **Note:** `--kv-cache-dtype fp8` affects **MLA paged KV only**; KDA keeps conv **bf16** + SSM **fp32**
   (see [tutorial Part 2b](kimik3_beginner_tutorial.md#part-2b--cache-dtypes-what---kv-cache-dtype-fp8-actually-changes)).
4. Numbers are **not apples-to-apples** (fixed synthetic 68K/350 vs variable cc-traces) — use them to
   characterize each config, not to rank.

## Takeaways for our recipe
- For **matching the reference / full-context serving**: use **bf16** (default) — it just works.
- For **decode speed at bounded context**: **fp8** wins per-token but needs the ≥128K capture bug fixed to
  go past 64K.
- Add **`--enable-prompt-tokens-details`** to the serve so prefix-cache hit rate + true TTFT are measured.

## Artifacts
- Reference: `k3_ref_c16.json`, `k3_ref_c24.json` (+ `k3_ref_client/`), serve `k3_ref_serve.log`.
- Ours: `k3_sweep_c{1,4,8,16,24}/`, `docs/kimik3_pareto/`.
