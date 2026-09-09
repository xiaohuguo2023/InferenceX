# Atom DeepSeek-V4 (MI355X) — improvement plan: Atom/AITER + InferenceX changes

Starting point: at conc=64 1k/1k decode, **Atom is the fastest backend per token
(TPOT 23.4 ms, tput 608 tok/s/GPU, TTFT 217 ms)** and already has the best GEMM
(CK `kernel_moe_mxgemm_2lds` MoE + preshuffled dense) and routing. Its headroom is
**not compute** — it is:

| Component | Atom share (decode) | ms/tok | Problem |
|---|---|---|---|
| Communication | 15.7% | 3.68 | **RCCL `ncclDevKernel` 12.3%** + aiter custom-AR 3.3% |
| Memory/elementwise | 20.8% | 4.88 | **`CatArrayBatchedCopy` (concat) ~10%** |
| Attention (MLA/DSA) | 25.4% | 5.95 | `_sparse_attn_ragged_varlen_triton` 13.3% (single biggest kernel) |

(Source: `docs/vllm_sglang_atom_dsv4_conc64_compare.md`; Atom trace is the published CI
`mi355x/atom/dsv4 conc64` runtime trace.)

---

## Part A — changes in Atom / AITER (engine + kernels)

### A1. Fuse the TP all-reduce into RMSNorm for DeepSeek-V4  *(highest ROI, low effort)*
- Atom already has `ATOM_ENABLE_ALLREDUCE_RMSNORM_FUSION` and wires it in
  `atom/models/deepseek_v2.py:125` — **but `atom/models/deepseek_v4.py` does not** (it calls
  plain `tensor_model_parallel_all_reduce`, e.g. the MoE-output AR at `deepseek_v4.py:2338`).
- **Change:** wire `ENABLE_ALLREDUCE_RMSNORM_FUSION` into `deepseek_v4.py` (fuse the
  post-attention and post-MoE all-reduce into the following RMSNorm), mirroring
  `deepseek_v2.py`. Removes the standalone AR + a norm kernel from the decode step.
- **Target:** comm 3.7 → ~2.5 ms/tok and norm ~0.

### A2. Route the decode TP all-reduce through AITER custom AR, not RCCL  *(high ROI)*
- The decode all-reduce (`[conc, 7168]` bf16 ≈ 0.9 MB) shows as **RCCL `ncclDevKernel`
  (12.3%)** with only 3.3% on `aiter::cross_device_reduce_2stage`. The custom AR is
  ~2× cheaper (cf. vLLM comm 2.5 vs Atom 3.7 ms/tok) and is CUDA-graph-capturable.
- **Change (Atom):** raise the `CustomAllreduce` size threshold / enable it for this
  all-reduce so it does not fall back to RCCL; keep it inside the decode CUDA graph.
- **Change (AITER):** confirm `cross_device_reduce_2stage` covers the size/dtype (bf16,
  16B-aligned) — it already does for vLLM/SGLang.
- **Note:** A1 largely subsumes A2 for the fused path; A2 covers any residual AR.
- **Target:** eliminate the 12% RCCL from decode.

### A3. Eliminate the decode concatenation (`CatArrayBatchedCopy` ~10%)  *(med effort)*
- Atom materializes tensor concats every decode step (likely QKV/KV concat or MoE
  token gather). This is ~10% of decode with no compute value.
- **Change:** fuse the concat into its producer/consumer or use views/pre-allocated
  contiguous buffers (as SGLang does — its elementwise is only 6.1%).
- **Target:** elementwise 4.9 → ~3 ms/tok.

### A4. Optimize the DSA sparse-attention kernel  *(higher effort)*
- `_sparse_attn_ragged_varlen_triton` is Atom's single biggest kernel (13.3%); attention
  is Atom's largest component (25.4%).
- **Change (AITER/Atom):** retune the Triton kernel (tiling/num_warps/splits) or move to a
  fused/assembly DSA-decode variant. MLA compression (`aiter::mhc_*`) is already shared and
  fast — leave it.
- **Target:** attention 6.0 → ~5 ms/tok (speculative; measure).

### Keep (Atom's current strengths — do not regress)
CK `kernel_moe_mxgemm_2lds` MoE experts, preshuffled dense GEMM, `ck_tile` MoE sort,
fused `dynamic_per_group_scaled_quant`. These are best-of-breed already.

---

## Part B — changes in InferenceX (benchmark repo)

### B1. Add an Atom profiling recipe  *(do first — unblocks measurement)*
- There is **no** `dsv4_fp4_mi355x_atom_profiling.sh` (only `..._atom.sh` / `..._atom_mtp.sh`),
  so Atom can't be profiled locally the way vLLM/SGLang can — the only Atom traces come from CI.
- Atom serve supports **`--torch-profiler-dir`** (`arg_utils.py:140`) + **`/start_profile`**
  (`api_server.py:1375`) — the same env+endpoint pattern.
- **Change:** add `benchmarks/single_node/dsv4_fp4_mi355x_atom_profiling.sh` mirroring the
  vLLM/SGLang profiling recipes: `PROFILE`-gated `--torch-profiler-dir $ATOM_TORCH_PROFILER_DIR`
  on the `atom.entrypoints.openai_server` command; `run_benchmark_serving` adds `--profile`
  (→ `/start_profile`). Add the trace-dump-wait cleanup as in the vLLM variant.

### B2. Add the tuning env flags to the Atom recipe  *(after A1/A2 validated)*
- `benchmarks/single_node/fixed_seq_len/dsv4_fp4_mi355x_atom.sh` currently exports only
  `ATOM_DISABLE_MMAP`, `AITER_BF16_FP8_MOE_BOUND`, `ATOM_MOE_GU_ITLV`.
- **Change:** add `export ATOM_ENABLE_ALLREDUCE_RMSNORM_FUSION=1` (and any custom-AR-size /
  `ATOM_USE_CUSTOM_ALL_GATHER` knob validated in A2) to the recipe.

### B3. Bump the Atom image pin  *(when Atom ships A1–A4)*
- `configs/amd-master.yaml` → `dsv4-fp4-mi355x-atom: image: rocm/atom-dev:nightly_202606161823`
  — bump to the Atom build containing the DSV4 AR+RMSNorm fusion / custom-AR / concat fixes.

### B4. Trigger the re-benchmark
- Append a `perf-changelog.yaml` entry for the Atom change (bilingual EN + 中文 commit per
  `AGENTS.md`) so the sweep re-runs and ingests the new Atom numbers.

---

## Projected impact (conc64 decode)
| Change | comm | elementwise | attention | TPOT |
|---|---|---|---|---|
| baseline | 3.7 | 4.9 | 6.0 | **23.4** |
| +A1/A2 (AR fusion + custom AR) | ~2.0 | 4.9 | 6.0 | ~21.7 |
| +A3 (concat fusion) | ~2.0 | ~3.0 | 6.0 | ~19.8 |
| +A4 (attn retune) | ~2.0 | ~3.0 | ~5.0 | **~18.8** |

→ TPOT 23.4 → ~19 ms, tput ~608 → **~750 tok/s/GPU at the same low TTFT (217 ms)** — i.e.
match SGLang's peak throughput (781) while keeping Atom's latency lead.

## Sequencing
1. **B1** (atom profiling recipe) — enables local measurement/validation.
2. **A1** (AR+RMSNorm fusion; flag already exists, just wire dsv4) → measure with B1.
3. **A2** (custom AR over RCCL) → measure.
4. **A3 / A4** (concat fusion, attention retune) — kernel work.
5. **B2 → B3 → B4** — land env flags, bump image pin, trigger sweep.

## Caveats
- Atom per-function ms are from the published CI runtime trace anchored to real conc64 TPOT
  (comparable to vLLM). Projections assume clean changes with no new overhead — A/B measure
  each via B1. A1/A2 are highest-confidence (flag exists / direct RCCL-vs-customAR delta);
  A4 is the most speculative.
- Exact `CustomAllreduce` size-threshold logic and the concat source weren't located in the
  image (compiled / deeper module); B1 profiling will pinpoint both precisely.
