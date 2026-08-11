# Kimi-K3 MI355X — local patch inventory

Idempotent in-container patches under `/workspace`. Each script guards on anchor text and
writes a `.bak` on first apply. Run inside the benchmark container after `setup_benchmark.sh setup`.

See also: [`kimik3_fp8_uncapped_pr_plan.md`](kimik3_fp8_uncapped_pr_plan.md),
[`kimik3_dspark_pr_plan.md`](kimik3_dspark_pr_plan.md).

## fp8 ASM MLA (Track: uncapped fp8 / DSpark base)

| script | target | upstream | purpose |
|--------|--------|----------|---------|
| `_patch_fp8asm.py` | `rocm_aiter_mla.py` | **#50578** | 12→16 head pad; asm persistent decode |
| `_patch_fp8_prefill.py` | `rocm_aiter_mla.py` | **PR-A / #51040** | fp8 asm prefill pad-to-16 |
| `_patch_ps_metadata16.py` | `rocm_aiter_mla.py` | **PR-A / #51040** | PS `num_head_k = max(16, num_heads)` |
| `_patch_skip_k3_fp8_ps.py` | `rocm_aiter_mla.py` | none | Skip PS workspace when K3 uses fused-FA prefill |
| `_patch_wvsplitk.py` | `model_executor/layers/utils.py` | **#50618** | `contiguous()` before wvSplitK |

Applied by `setup_benchmark.sh setup` and `_k3_dspark_fp8asm_apply_patches.sh` step 0.

## DSpark + fp8-asm (delta on fp8 base)

| script | what |
|--------|------|
| `_k3_dspark_fp8asm_apply_patches.sh` | One-shot: fp8 base + forced causal draft + aiter 80-key + KDA PR#27; steps 4–5 no-op if #51011 present |
| `_serve_k3_bench_spec.sh` | DSpark serve: `ROCM_AITER_MLA`, fp8 KV, `ASM_PADDING=asm`, `MAX_NUM_SEQS=16` |
| `_bench_k3_dspark_fp8asm.sh` | aiperf sweep ISL1024/OSL256 c1/8/16 |

## Agentic memory / headroom (orthogonal to DSpark)

| script | target | upstream PR | purpose |
|--------|--------|-------------|---------|
| `_patch_cgmem.py` | `gpu_model_runner.py`, `gpu_worker.py` | **PR-E** | Honest CUDA graph memory estimate |
| `_patch_cgmem_snap.py` | `gpu_model_runner.py` | diagnostic | Allocator snapshot hooks |
| `_patch_moe_scratch.py` | `aiter/ops/flydsl/moe_kernels.py` | **PR-F** (scratch form) | Reuse MoE stage-1 buffers in graphs (~7 GiB) |
| `_patch_moe_prf.py` | `aiter/ops/flydsl/moe_kernels.py` | **PR-F** (upstream form) | WorkspaceManager-friendly variant |
| `_patch_kv_b_proj_chunk.py` | `mla_attention.py` | experimental | Chunk `kv_b_proj` to cap M=196608 workspace |
| `_patch_warmups.py` | harness | serve tuning | Warmup grace for long agentic primers |

Used by `_agentic_ladder.sh`, `_fixed_arm.sh`, `_ttft_arm.sh`, `_k3_attention_debug_c48*.sh`.

## Verify

```bash
./setup_benchmark.sh verify-patches          # fp8 ASM five-pack
./setup_benchmark.sh verify-dspark-patches   # + aiter 80-key, dspark qlen, KDA
bash _lint_scripts.sh                        # host-side script lint
```
