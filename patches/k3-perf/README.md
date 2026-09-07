# Kimi-K3 conc-1 decode glue patches (latest image)

Four vLLM source patches that cut kernel launches out of the K3 decode step. Independent of
the DCP8 bundle in `../k3-dcp8/` — these apply to the non-DCP path too, and nothing here
touches DCP.

## Pins

| | |
|---|---|
| vLLM image | `vllm/vllm-openai-rocm:nightly-rocm100-e962733e08d10f7ca65dac4df99e116460b8b174` |
| vLLM version | `0.28.1rc1.dev437+ge962733e0` |

The nightly image's `ENTRYPOINT` is `vllm`, so start the container with `--entrypoint sleep`
or it will try to serve immediately.

## Applying

All four apply with `-p1` from the installed vllm package root:

```bash
cd /usr/local/lib/python3.12/dist-packages/vllm
for f in envs utils kda linear; do
    patch -p1 < /path/to/patches/k3-perf/vllm/$f.patch
done
```

Confirm each landed (a *failing* reverse dry-run means not applied):

```bash
for f in envs utils kda linear; do
    printf "%-8s " "$f"
    patch --dry-run -R -p1 < /path/to/patches/k3-perf/vllm/$f.patch >/dev/null 2>&1 \
        && echo applied || echo NOT-APPLIED
done
```

Verified: applying all four to the pristine image files reproduces the measured container
(`k3-rocm10`) byte-for-byte.

## The patches

| patch | target | size | what it does |
|---|---|---|---|
| `utils.patch` | `model_executor/layers/utils.py` | +18/−4 | the two unquantized-GEMM copy fixes |
| `kda.patch` | `models/kimi_k3/amd/kda.py` | +68/−42 | KDA output-buffer elision, in-place merge, o_norm self-copy elision, and an env-gated fused QKV conv |
| `linear.patch` | `models/kimi_k3/amd/linear.py` | +33/−19 | deferred MLP residual add (the #52968 elementwise fusion) |
| `envs.patch` | `envs.py` | +6/−0 | declares `VLLM_ROCM_USE_AITER_FUSED_QKV_CONV` |

They are **not four independent changes**. Three workstreams cut across the four files:

### 1. Copy elimination — `utils.patch` + part of `kda.patch`

`rocm_unquantized_gemm_impl` hoisted `x_view = x.reshape(-1, x.size(-1)).contiguous()` above
the two skinny branches that consume it. Those branches need `n <= 5`, but at DSpark decode
`n = 1 + num_spec = 8`, so neither can ever match and the copy is pure waste on any strided
activation. Fix one sinks it into each branch.

Fix two gates the aiter route on `x.is_contiguous()`. aiter's tuned bf16 GEMM has no `lda`:
`aiter/ops/flydsl/gemm_kernels.py` calls `a.contiguous()` internally, so a strided activation
pays a *second* copy on top of the GEMM. hipBLASLt takes the leading dimension natively.
Justified by a 13-shape strided crossover sweep — `F.linear` wins 11/13 and never loses by
more than 13%. The contiguous path is untouched.

Net on the `f_b_proj` shape (`[8,128]` × `[1536,128]`, ×69 KDA layers): 3 kernels → 1,
~6.5 → 3.03 us/call.

In `kda.patch`, the o_norm hunk is the same class of bug. The fused triton gated RMSNorm
normalises in place (`layer_norm_gated_fwd` uses `y = x` when no `out_dtype` is requested), so
`core_attn_out.copy_(self.o_norm(...))` was a self-copy — one dead kernel per KDA layer.
`forward_native` *does* return a fresh tensor, so the copy is kept for that path via a
`data_ptr()` check.

### 2. Deferred MLP residual add — `linear.patch` + the signature hunks of `kda.patch`

Layer *i* hands its MLP output back unsummed as `pending_add`; layer *i+1*'s attn-residual
kernel folds it in via the existing `delta=` argument. Also drops the `torch.empty_like`
attention output buffer, which is why `KimiMLAAttention.forward` and `KDA._forward` change
from writing into an `output:` argument to returning a tensor. **Those signature changes span
both files — `kda.patch` and `linear.patch` must be applied together or the model will not
run.**

### 3. Fused QKV conv — `envs.patch` + the prefill hunk of `kda.patch`

`VLLM_ROCM_USE_AITER_FUSED_QKV_CONV` routes the KDA *prefill* conv through aiter's
`causal_conv1d_split_qkv_triton_tile_fn`, one launch instead of three.
**Defaults to `False` and has no e2e number.** It is carried here so the experiment is not
lost; do not enable it in a recipe without gating it first.

## Measured

conc-1 DCP1 K=7, IX agentic, full 3600 s, latest image, run `c1_r10_glue` vs `c1_r10_retune`.
Bundles workstreams 1 and 2 plus the `+situ_and_mul` recipe flag (see below); the individual
shares are not separable.

| metric | baseline | patched | delta |
|---|---:|---:|---|
| ITL p50 | 7.8776 | 7.7173 | **−2.04%** |
| ITL p90 | 8.5353 | 8.3921 | **−1.68%** |
| TTFT p50 | 995.3 | 983.1 | −1.22% |
| TTFT p90 | 4167.2 | 4082.2 | −2.04% |
| request_latency p90 | 38127 | 37314 | −2.13% |
| interactivity p90 | 117.16 | 119.16 | +1.7% |

`request_count` 194 → 196 and `output_sequence_length` avg 1657.9 → 1668.0 both moved, so the
throughput figure (1357 → 1379 tok/s/GPU) is trace-contaminated — **quote ITL, not tok/s**.

Accuracy gate: **GSM8K 0.9636** flexible-extract, full 1319, ±0.0052, `rejection_sample_method:
block`. Identical to baseline, as expected for changes that are value-preserving by
construction.

An earlier conc-1 + conc-12 A/B of workstream 2 alone measured −1.1% ITL p50 at both
concurrencies with GSM8K 0.9682.

## Required alongside these patches

Neither is a source change, but the measured numbers include both:

- `custom_ops` must list `+situ_and_mul`. At `compilation_config.mode:3` every *unlisted*
  custom op defaults off, so `SituAndMul` is decomposed for inductor and runs
  `forward_native`, which upcasts to fp32 with temporaries. The shipped C++ op is **43%
  faster** at the conc-1 shape `[8,1536]` (1.74 vs 3.05 us) across all 92 shared-expert
  layers. Zero code change — it is a recipe flag.
- `AITER_CONFIG_GEMM_BF16` must point at the merged tuned CSV
  (`../k3-dcp8/aiter/merged_bf16_tuned_gemm.csv`). Without it `is_tgemm_enabled()` still
  returns true but every lookup misses, and `utils.patch`'s crossover justification no longer
  holds.

## Verifying

Standing rule for this work: **e2e only, no microbenchmark conclusions.** One conc-1 DCP1 K=7
IX agentic point at the full 3600 s, compared on ITL p50/p90 and interactivity p90 — not
tok/s/chip, which is quantized ~0.5% per request at conc-1. Print `request_count` and
`output_sequence_length` beside every delta; if they differ from the baseline the delta is
contaminated.

Accuracy gate is GSM8K in **block** mode (expect 0.963–0.969). `synthetic` acceptance forces
the acceptance length and measures nothing about correctness.

## Teardown

Graceful SIGTERM to the explicit pid. Never SIGKILL, never `pkill -9 python3` — see
`../k3-dcp8/README.md` for why that wedges the box.
