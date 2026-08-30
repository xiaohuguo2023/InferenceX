# Kimi-K3 DSpark K=7 — nightly ROCm image recipe (MI355X TP8)

Current serving combo: **DSpark K=7 / AL 3.84**, `ROCM_AITER_MLA` + **fp8 KV**,
dense prefix-cache retention, `--kv-cache-memory 47691420128` (~44.4 GiB),
`FULL_AND_PIECEWISE` with a capture ladder that includes `(1+K)*conc`.

This file lives on branch `xiaohuguo/kimik3-fp4-latest-nightly` with the
overlays under `benchmarks/single_node/agentic/k3_patches/`. Clone that branch
on another machine — do not depend on `~/work/k3_pr2585_results`.

The Aug-10 K=2 enablement runbook (`kimik3_dspark_fp8asm_recipe.md`, image
`cb8104839`) is historical. Do not apply those in-container patches on this
nightly.

**Do not `BOOTSTRAP=1` / do not run `apply_k3_fp4_fp8asm_dspark_patches.sh`.**
That rebuilds aiter at `55dbc4f` and throws away the image aiter. Use
`k3_patches/apply_nightly_k7_overlays.sh` instead.

---

## 1. Image

**Hub `nightly` as of 2026-08-30 05:27 UTC:**

```
vllm/vllm-openai-rocm:nightly-1dc464d42681d22f38caf1fdc1eb632dc4421c45
```

| | |
|---|---|
| Tag | `nightly-1dc464d42681d22f38caf1fdc1eb632dc4421c45` (also Hub `nightly` that day) |
| vLLM | `1dc464d426` — `[Bugfix] Bound cache_salt length` (#54353) |
| GPUs | 8× MI355X (gfx950), TP8, `--exclusive` hold |
| Target | `moonshotai/Kimi-K3` FP4 |
| Draft | `Inferact/Kimi-K3-DSpark` (`dflash_config.causal=true`) |

**Last measured numbers in this file are from the previous nightly**,
`nightly-6d4562c59b97b4e35d459ff9389e71b6fe4995de` (`v0.28.1rc1.dev87+g6d4562c59`,
2026-08-29). `1dc464d` is **+21 commits**, including
[#51171](https://github.com/vllm-project/vllm/pull/51171) (`[ROCm][MLA] Reach
FULL cudagraphs for AITER MLA speculative decoding`) and
[#50488](https://github.com/vllm-project/vllm/pull/50488) (widest uniform-decode
capture by default). Treat 1dc464d as unvalidated until a K=7 sweep lands.

---

## 2. Knobs

| Knob | Value |
|---|---|
| `NUM_SPEC_TOKENS` | `7` |
| `SYNTHETIC_ACCEPT_LEN` | `3.84` (golden AL at K=7) |
| `KV_CACHE_MEMORY` | `47691420128` (44.4 GiB; ~2.68M GPU KV tokens on 6d4562c) |
| `gpu-memory-utilization` | `0.95` |
| `max-num-seqs` | `64` |
| `max-num-batched-tokens` | `16384` |
| `max-model-len` | `1048576` |
| `kv-cache-dtype` | `fp8` |
| attention | `ROCM_AITER_MLA` target **and** draft |
| `VLLM_ROCM_AITER_MLA_ASM_PADDING` | `asm` |
| cudagraph | `FULL_AND_PIECEWISE` |
| capture sizes | `1,2,3,4,6,8,12,16,24,30,32,36,40,48,56,60,64,72,80,88,96,104,112,120,128,136,144,152,160,168,176,184,192,200,208,216,224,232,240,248,256,272,288,304,320,336,352,368,384` — must include `(1+K)*conc` or that conc falls to PIECEWISE / eager ITL |
| prefix cache | `--enable-prefix-caching`; dense via overlay (`dense` CLI string still aborts) |
| decode-priority | **off** |
| PTPC | **off** at K=7 |
| mixer / fused-KDA spec | **off** |
| AgentX duration | `3600` (`inferencex-agentx-mvp` rejects `< 900`) |
| DRAM | `SimpleCPUOffloadConnector`, `TOTAL_CPU_DRAM_GB=1799`, from **c8** up |

Serve script in this repo:
`benchmarks/single_node/agentic/kimik3_fp4_mi355x_vllm_mtp.sh` (defaults are
still K=2; pass the env above).

---

## 3. Overlays in this repo

All under `benchmarks/single_node/agentic/k3_patches/`.

| File | When | Why |
|---|---|---|
| `patch_prefix_cache_retention_dense.py` | always | factory still `0` if env unset |
| `patch_aiter_blockn_fp8.py` | until stock K=7 never `KeyError`s | 80/96/112 + `.get(...,64)`; not aiter #4713 |
| `patch_aiter_splitk_cudagraph.py` | until image aiter ≥ #4494 | set `SPLITK_CU` to image `aiter_meta/.../asm_gemm_a16w16.cu` |
| `patch_offload_eagle_prefix_veto.py` | **DRAM** (safe on GPU-only) | full-attn eagle pop zeros CPU→GPU |
| `merged_bf16_tuned_gemm.worktree.csv` | always | 3027 rows; must not contain kernel 1212 |
| draft `dflash_config.causal=true` | always | `kimik3_fp4_mi355x_vllm_mtp.sh` edits the staged draft |

Do **not** overlay: KDA safe-stages, mixer #53487, fused KDA spec, PTPC, decode-priority.

Kill between serves: `pkill -9 python3` (not just `vllm serve`), then wait for
the ~280 GiB/GPU drain. Distinct `RESULT_DIR` / `RUN_TAG` on shared NFS.

---

## 4. Repeat on another machine

```bash
git clone -b xiaohuguo/kimik3-fp4-latest-nightly <this-fork> InferenceX
cd InferenceX
# exclusive GPU hold, then:
IMAGE=vllm/vllm-openai-rocm:nightly-1dc464d42681d22f38caf1fdc1eb632dc4421c45
CTR=k3-k7-nightly
docker pull "$IMAGE"
docker run -d --name "$CTR" \
  --ipc=host --network=host --shm-size=137438953472 \
  --device=/dev/kfd --device=/dev/dri \
  --group-add video --group-add render \
  --security-opt seccomp=unconfined --security-opt label=disable \
  --cap-add=SYS_PTRACE -e GPU_ARCHS=gfx950 \
  -v "$PWD:/workspace:ro" \
  --entrypoint sleep "$IMAGE" infinity

docker cp benchmarks/single_node/agentic "$CTR:/opt/k3-recipe"
docker exec "$CTR" bash /opt/k3-recipe/k3_patches/apply_nightly_k7_overlays.sh
```

Inside the container (or `docker exec` with the recipe env):

```bash
export VLLM_ROCM_AITER_MLA_ASM_PADDING=asm
export AITER_CONFIG_GEMM_BF16=/opt/aiter-local/aiter/configs/merged_bf16_tuned_gemm.csv
export SKIP_K3_BOOTSTRAP=1
export NUM_SPEC_TOKENS=7 SYNTHETIC_ACCEPT_LEN=3.84
export KV_CACHE_MEMORY=47691420128
export CAPTURE_SIZES=1,2,3,4,6,8,12,16,24,30,32,36,40,48,56,60,64,72,80,88,96,104,112,120,128,136,144,152,160,168,176,184,192,200,208,216,224,232,240,248,256,272,288,304,320,336,352,368,384
# plus MODEL TP CONC KV_OFFLOADING TOTAL_CPU_DRAM_GB RESULT_DIR DURATION from the harness
bash /opt/k3-recipe/kimik3_fp4_mi355x_vllm_mtp.sh
```

On this cluster the existing fanout still works if you point overlays at this
tree: `RECIPE=latest-k7 IMAGE=... WORKTREE=<clone>`.

---

## 5. Last measured (6d4562c, 3600s AgentX, 2026-08-29)

Not a 1dc464d result. Node noise floor ~4.3%.

| arm | tot/chip | TTFT p90 | ITL p90 | CPU hit | GPU KV |
|---|---:|---:|---:|---:|---:|
| c1 GPU n091 | 1340 | 1.87 s | 8.60 ms | 0 | 2.68M |
| c2 GPU | 1473 | 1.79 s | 10.63 ms | 0 | 2.68M |
| c4 GPU | 2552 | 1.82 s | 10.89 ms | 0 | 2.68M |
| c8 DRAM n013 | 4139 | 1.96 s | 18.1 ms | 5.5% | 2.68M |
| c16 DRAM | 6811 | 7.86 s | 25.0 ms | 60% | 2.68M |

c1 vs prior aa990 K=7 n176 1402 = **−4.4%**; vs ATOM 1424 = −5.9% (ATOM was
K=6+ptpc, wrong recipe). Eagle-veto is why DRAM CPU hits are non-zero.

---

## 6. Do not

- `BOOTSTRAP=1` or `apply_k3_fp4_fp8asm_dspark_patches.sh` on this nightly.
- `--prefix-cache-retention-interval dense` until the parser PR lands.
- Attach serving to a non-exclusive hold.
- Reuse a result tag across nodes (shared NFS).
- Report EVAL_ONLY / `rejection_sample_method=block` tok/s as a throughput number.
