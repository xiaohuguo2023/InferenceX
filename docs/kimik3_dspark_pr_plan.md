# Kimi-K3 DSpark + fp8-asm KV — PR plan (MI355X, TP8)

Upstream plan for **DSpark speculative decoding on Kimi-K3 with native fp8 KV** on the
**ROCM_AITER_MLA asm path** (not the Gluon/TRITON_MLA stack). Operational reproduction:
[`kimik3_dspark_fp8asm_recipe.md`](kimik3_dspark_fp8asm_recipe.md).

Master fp8-uncapped plan (prefill/decode base): [`kimik3_fp8_uncapped_pr_plan.md`](kimik3_fp8_uncapped_pr_plan.md).

## Architecture we upstream

| Component | Choice |
|-----------|--------|
| Target backend | `ROCM_AITER_MLA` + `--kv-cache-dtype fp8` |
| Draft | `Inferact/Kimi-K3-DSpark` with **`dflash_config.causal=true`** |
| Verify path | fp8 asm q-row-fold (`VLLM_ROCM_AITER_MLA_ASM_PADDING=asm`) |
| Serve cap | `MAX_NUM_SEQS=16` (PIECEWISE capture; do not use agentic seq ladder) |

## Local patches → upstream mapping

| # | area | local patch / script | upstream | action |
|---|------|---------------------|----------|--------|
| D0 | fp8 decode routing, qlen, persistent gate | (none — nightly ships it) | **#51011** merged | **Adopt** — replaces recipe §4.3/§4.4 |
| D1 | fp8 prefill pad-to-16 | `_patch_fp8_prefill.py`, `_patch_ps_metadata16.py` | **#51040** (PR-A) | **Filed** — adopt when merged |
| D2 | decode pad + wvSplitK | `_patch_fp8asm.py`, `_patch_wvsplitk.py` | #50578, #50618 | **Adopt** |
| D3 | skip K3 fp8 PS workspace | `_patch_skip_k3_fp8_ps.py` | none | Serve-side until fused-FA path upstreams |
| D4 | aiter `get_block_n_fp8` key 80 | `_k3_dspark_fp8asm_apply_patches.sh` §3 | none | **File PR-G (aiter)** |
| D5 | forced causal draft | `_k3_dspark_fp8asm_apply_patches.sh` §2 | none | **Ops/doc** — edit draft `config.json` |
| D6 | DSpark + fp8 KV glue | (draft-side quant query) | **#51606** draft | **Adopt** when rebased |
| D7 | KDA stride | `_k3_dspark_fp8asm_apply_patches.sh` §6 | vLLM main | **Adopt** — already on current main |

Agentic/memory patches (`_patch_cgmem.py`, `_patch_moe_scratch.py`, etc.) are **orthogonal** to DSpark;
DSpark serve uses `MAX_NUM_SEQS=16`, not c24 agentic. See [`kimik3_patches.md`](kimik3_patches.md).

## PRs to file / adopt

### PR-G (aiter, primary DSpark delta) — `get_block_n_fp8` key 80

DSpark target verify at `num_spec=2` is qlen=5; padded heads=16 → `16×5=80`. The fp8 block-size
table lacks key 80 → `KeyError` mid-run.

- Add keys `80/96/112 → 64` to `get_block_n_fp8`
- Change lookup to `.get(int(nhead * max_seqlen_q), 64)` (safe: `min_block_n` only bounds splits)

**Evidence:** recipe validation logs; manual run AL **2.39** (natural text), clean PIECEWISE 15/15 +
FULL 8/8 on working container (`xguo-k3nc`).

### Adopt — do not duplicate

| PR | repo | what |
|----|------|------|
| **#51011** | vLLM | fp8 decode routing, `reorder_batch_threshold` qlen, `use_gluon_verify`, persistent gate |
| **#51040** | vLLM | fp8 asm prefill pad-to-16 (PR-A) |
| **#51606** | vLLM | DSpark draft + fp8 KV (`supports_quant_query_input`, amd glue) |
| #50578, #50618 | vLLM | decode pad, wvSplitK |
| #50579 | vLLM | a8w4 MoE correctness |

### Do NOT adopt as-is

| PR | why |
|----|-----|
| **#50619** (JohnQinAMD) | **Gluon** native multi-token verify + TRITON_MLA defaults — conflicts with our **asm q-row-fold** path. Coordinate via #50682 / #51232; cite as alternative, not our stack. |

## Dependency order

```
#50578 + #50618  ──►  #51011 (merged)  ──►  #51040 (PR-A)
                              │
                              ├──►  #51606 (DSpark glue)
                              └──►  PR-G (aiter key 80)     [only novel code we file]
```

## Validation status (Aug 2026)

| check | status |
|-------|--------|
| Manual recipe (`xguo-k3nc`) | **PASS** — AL 2.39, cudagraph 15/15 + 8/8, aiperf c1/8/16 |
| Automated recipe (n193/n249) | **BLOCKED** — serve dies at PIECEWISE 3/15 (M=72); patches verify OK |
| GSM8K (#51606 claim) | 94.3–95.1% DSpark+fp8 asm (external PR) |

**Block filing PR-G perf claims on** cluster repro of clean capture, or ship PR-G as correctness-only
with manual-container evidence.

## InferenceX workflow

```bash
export K3_CTR=k3-dspark-benchmark
./setup_benchmark.sh start-dspark
./setup_benchmark.sh setup-dspark      # fp8-asm patches + DSpark enablement
./setup_benchmark.sh verify-dspark-patches
./setup_benchmark.sh serve-dspark      # NUM_SPEC=2, ASM_PADDING=asm, MAX_NUM_SEQS=16
docker exec k3-dspark-benchmark bash -lc 'PORT=8890 bash _bench_k3_dspark_fp8asm.sh'
```

Recipe test driver: `_launch_k3_dspark_recipe_test.sh` → `_k3_dspark_recipe_test_node.sh`.
