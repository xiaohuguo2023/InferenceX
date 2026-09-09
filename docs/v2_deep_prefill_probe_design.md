# V2 `profile_deep_prefill_memory` — verified port design

Status: DESIGN READY (2026-08-15). Implementation deferred until after the agentic
bf16 retune + validation, per user sequencing. This is the fix for the
`AttributeError: 'GPUModelRunner' object has no attribute 'profile_deep_prefill_memory'`
(K3 DSpark forces the **V2** runner `vllm/v1/worker/gpu/model_runner.py`; the probe
was patched into the unused V1 file). See [[k3-dspark-forces-v2-model-runner]] and
`memoryestimationplan.md`.

## Approach (chosen: (a) real chunked-prefill batch)
V2's `_dummy_run` only produces a FRESH-prefill batch (`InputBatch.make_dummy` sets
`seq_len==query_len`) with `prepare_dummy_attn` returning ZEROED block tables — no
seam to inject deep context. So instead of V1's scalar `profile_seq_lens` trick,
build a REAL 1-request chunked-prefill `SchedulerOutput` (num_computed_tokens =
ctx - q, aliased deep block table) and call
`execute_model(sched, dummy_run=False, is_profile=True)`:
- `dummy_run=False` → real add_requests/prepare_inputs/prepare_attn (genuine deep
  MLA metadata, max_seq_len=ctx).
- `is_profile=True` → need_eager → cg_mode=NONE → eager fused-MLA prefill kernel.
This is strictly MORE faithful than V1 and less fragile than extending `_dummy_run`.

## Insertion anchor (V2 model_runner.py)
Prepend the three methods before (currently ~line 561):
```
    def _init_kv_zero_meta(self) -> None:
        """Build KV-block zeroing metadata; invoked from gpu_worker."""
```
Patcher matches that 2-line string. `gpu_worker.py` call-site is UNCHANGED (it calls
`self.model_runner.profile_deep_prefill_memory()` and reads the same dict keys).

## Key V2 symbols (model_runner.py unless noted)
- `initialize_kv_cache(self, kv_cache_config)` @453 — NO is_profiling arg (drop V1's).
- `execute_model(self, scheduler_output, intermediate_tensors=None, dummy_run=False, skip_attn_for_dummy_run=False, is_profile=False)` @1227 — the probe engine; sets `self.execute_model_state` @~1451 (MUST reset to None after).
- `add_requests` @858, `prepare_inputs` @950, `prepare_attn` @1131, `_remove_request` @816.
- `self.req_states` (RequestState) @245; `self.block_tables` (BlockTables) @496 (`append_block_ids`, block_table.py:107); `self.kv_connector`/`NO_OP_KV_CONNECTOR` @283/import 100.
- `get_kv_cache_groups` (kv_cache_utils.py:1747), `get_kv_cache_config_from_groups` (:1327), `may_override_num_blocks` (:942).
- `MemorySnapshot` (mem_utils.py:109); `set_current_vllm_config` (config/vllm.py:2374, import locally — NOT imported in V2); `NewRequestData` (v1/core/sched/output.py:35); `SamplingParams`; `cdiv` (math_utils, imported @57).

## Focus answers
1. Temp KV: `initialize_kv_cache(minimal_config)` rebuilds everything; minimal via
   num_gpu_blocks_override=min_blocks + get_kv_cache_config_from_groups(available_memory=0).
   Table width from max_model_len so ctx-deep row fits with few physical blocks.
2. Deep batch: NewRequestData num_computed_tokens=ctx-q, aliased block_ids (one
   physical block/group repeated cdiv(ctx, block_size*dcp_size) times).
3. Forced attention: attention runs whenever `not (dummy_run and skip_attn)`;
   dummy_run=False → always. is_profile=True → eager fused-MLA. No skip_attn.
4. Speculator: execute_model does NOT call speculator.propose (only _dummy_run does)
   → probe measures the TARGET deep-prefill peak in isolation. Drafter footprint is
   already covered by axis-3b in profile_run. Mixed-batch (1 prefill + spec decode)
   is a deferred Phase-2 shape.
5. Pitfalls: KVConnector set_disabled(True/False); execute_model_state=None after
   (most important V2-only cleanup); _remove_request probe reqs; DCP via bs*dcp_size;
   assert pp_size==1; LoRA None for K3.

## On-box traces still needed (D)
1. `len(self.kv_cache_config.kv_cache_groups)` (1 vs 2) + per-group block_size.
2. Aliasing fidelity: rocprof that aliased single-block row hits SAME fused-MLA
   workspace path as distinct blocks; else set VLLM_PROFILE_MIN_BLOCKS>=cdiv(ctx,bs)
   and use distinct block ids.
3. Confirm `self.pcp_manager is None` on the K3 MI355X recipe.
4. EPLB prepare_forward allocates no memory counted into persistent_runtime.
5. SamplingParams() allocates no per-request logprob buffers inflating the peak.
6. Residual in-kernel scratch invisible to torch_peak/mem_get_info → may need the
   backend_workspace_bytes hook (phase-2 fold already reads that key, defaults 0).

The full method bodies (profile_deep_prefill_memory, _init_minimal_kv_cache_for_profiling,
_cleanup_profiling_kv_cache) are in the session transcript for this date; regenerate
into `_patch_profiler_axes.py` under a V2 target when implementing.
