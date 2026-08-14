# Kimi-K3 DSpark + fp8-asm KV — consolidated PR plan (MI355X, TP8)

Upstream plan for **DSpark speculative decoding on Kimi-K3 with native fp8 KV** on the
**ROCM_AITER_MLA asm path** (not the Gluon/TRITON_MLA stack). Operational reproduction:
[`kimik3_dspark_fp8asm_recipe.md`](kimik3_dspark_fp8asm_recipe.md). Patch inventory:
[`kimik3_patches.md`](kimik3_patches.md).

_Last consolidated: 2026-08-12. Supersedes the earlier PR-A..PR-G draft; renumbered to
reflect the two PRs now filed and the fixes discovered during the agentic bring-up._

## Architecture we upstream

| Component | Choice |
|-----------|--------|
| Target backend | `ROCM_AITER_MLA` + `--kv-cache-dtype fp8` |
| Draft | `Inferact/Kimi-K3-DSpark` with **`dflash_config.causal=true`** |
| Verify path | fp8 asm q-row-fold (`VLLM_ROCM_AITER_MLA_ASM_PADDING=asm`) |
| Recipe stance | **File our fp8-asm recipe as a SEPARATE PR**, cite #2508 (TRITON_MLA) as the alternative |

## Status: 2 PRs filed (memory/headroom track, both OPEN)

| PR | internal label | scope | evidence |
|----|----------------|-------|----------|
| **vLLM #51590** | PR-E (`_patch_cgmem.py`) | Measure full CUDA-graph capture footprint for KV budgeting; fixes agentic `HSA_STATUS_ERROR_OUT_OF_RESOURCES` at 3–16% KV usage | est. 1.3→4.7–16.9 GiB/GPU; CPU unit tests |
| **aiter #4647** | PR-F (`_patch_moe_scratch.py`) | Reuse FlyDSL MoE stage-1 scratch across layers/graph captures (~6.7 GiB/GPU) | dedup allocs; stream-isolation tests |

## PRs still to file (ranked)

| # | repo | what | local source | type | priority |
|---|------|------|--------------|------|----------|
| **P1** | aiter | `get_block_n_fp8` key **80** (+96/112→64) + `.get(...,64)` fallback — DSpark verify width `16×5` KeyErrors mid-run. **Confirmed still novel** (upstream `aiter/mla.py` dict has no key 80, uses direct `[...]` indexing) | `_k3_dspark_fp8asm_apply_patches.sh` §3 | correctness | **HIGH** |
| ~~P2~~ | ~~aiter~~ | ~~split-K a16w16 cudagraph-safety~~ — **SUPERSEDED by upstream aiter #4494** (merged 2026-08-12); see Adopt table. Our local `_patch_aiter_splitk_cudagraph.py` (disable-by-default workaround) is **retired**; adopt #4494 and re-enable split-K. | — | — | — |
| **P3** | vLLM | uniform-decode **dispatch pads to next FULL graph** (`v1/worker/gpu/cudagraph_utils.py` `_init_candidates`/`dispatch`/`_is_compatible`) — durable replacement for the config-only `{12,36}` capture mitigation | plan item-1 / this file | perf | MED |
| **P4** | aiter | tuned-GEMM config contributions: FlyDSL→torch decode reroute (conc-24 all-reduce stall) + `6288×7168` / `7168×35840` entries (N%64=16 → torch fallback) | `_patch_flydsl_decode_to_torch.sh`; task #50 | perf/data | MED/LOW |
| **P5** | InferenceX | our **fp8-asm ROCM_AITER_MLA agentic recipe** (`kimik3_fp4_mi355x_vllm_dspark.sh` + `_serve_k3_bench_spec.sh` deltas), synthetic-2.51 AgentX-compliant arm; cite #2508 as the TRITON_MLA alternative | this repo branch | recipe | after P1/P2 |

## Adopt — do not duplicate (already upstream)

| PR | repo | what | retires local patch |
|----|------|------|---------------------|
| **#4494** (merged 2026-08-12) | aiter | ASM split-K semaphore cudagraph-replay fix: `get_semaphore_workspace()` returns a **fresh zero workspace per launch while a graph is capturing** (zero-fill becomes a captured node → counter==0 every replay), kept alive for process lifetime; eager keeps the stream-keyed cache. Validated on OUR stack (8×MI355X, K3 TP8, DSpark, draft in FULL graph): capture 8/8, GSM8K 0.990, batch-1 120.9 vs 114.7 tok/s (draft-eager). Same root cause + rocgdb signature (`splitk_clean` counter=2 waiting for 0) as ours. | `_patch_aiter_splitk_cudagraph.py` (the disable-by-default workaround — **delete once aiter ≥ c6ce60c**; then re-enable split-K, no `AITER_ALLOW_SPLITK`) |
| **#51011** (merged) | vLLM | fp8 decode routing, `reorder_batch_threshold` qlen, `use_gluon_verify`, persistent gate | apply §4/§5 |
| **#51040** | vLLM | fp8 asm prefill pad-to-16 | `_patch_fp8_prefill.py`, `_patch_ps_metadata16.py` |
| **#51606** | vLLM | DSpark draft + fp8 KV (`supports_quant_query_input`) | draft-side |
| #50578 / #50618 / #50579 | vLLM | decode pad / wvSplitK `contiguous()` / a8w4 MoE | `_patch_fp8asm.py`, `_patch_wvsplitk.py` |
| main | vLLM | KDA `state_indices` stride | apply §6 |

### Do NOT adopt as-is

| PR | why |
|----|-----|
| **#50619** (JohnQinAMD) | **Gluon** native multi-token verify + TRITON_MLA defaults — conflicts with our **asm q-row-fold** path. Cite as alternative, not our stack. |

## Serve/recipe-side only — never upstream (ship in the recipe, P5)

forced-causal draft (`dflash_config.causal=true`) · `_patch_skip_k3_fp8_ps.py` (skip K3 fp8 PS
workspace) · `AITER_CONFIG_GEMM_BF16` export (pin patched BF16 catalog) · `{12,36}` capture sizes
(until P3) · **synthetic-acceptance 2.51** (AgentX policy — see
[`k3-agentx-synthetic-acceptance-policy`]) · mandated config 64 / 0.95 / 16384.

**Retired workaround:** `_patch_aiter_splitk_cudagraph.py` (disabled split-K by default) — delete
once the image's aiter is ≥ #4494 (`c6ce60c`); then let split-K run (upstream #4494 makes it
cudagraph-replay-safe and it's ~5% faster at batch-1 than draft-eager).

## Dependency order

```
FILED:  #51590 (KV budget)  ─┐ independent, land on their own
        #4647  (MoE scratch) ┘
ADOPT:  #4494 (aiter split-K cudagraph) — MERGED upstream; bump aiter, drop our workaround

P1 aiter key-80 ──► (+ aiter ≥ #4494) ──► stock-image recipe works ──► P5 recipe PR
P3 vLLM dispatch-pad ──► retires {12,36}
P4 aiter tuned configs (perf; after P1 correctness)
```

**File order:** P1 (only remaining aiter correctness blocker for a stock-image recipe now that
#4494 is upstream) → P3 (vLLM durability) → P5 (recipe) → P4 (perf/data). Split-K correctness is
handled by adopting #4494, not by a PR of ours.

## Validation status (Aug 2026)

| check | status |
|-------|--------|
| Manual recipe (`xguo-k3nc` / `k3-dspark-benchmark`) | **PASS** — AL 2.39, cudagraph 15/15 + 8/8 |
| Agentic real-verify sweep {1,2,4,8,16,24} DURATION=3600 | running (c16; c24 to follow) |
| Agentic synthetic-2.51 sweep (AgentX-comparable) | chained, auto-starts after real-verify |
| GSM8K (#51606 claim) | 94.3–95.1% DSpark+fp8 asm (external PR) |
