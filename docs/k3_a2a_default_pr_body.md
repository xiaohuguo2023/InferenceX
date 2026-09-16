## Purpose

Kimi-K3 leaves `dcp_comm_backend` unset, so it gets the `ag_rs` default: `allgather(lse)` + `reduce_scatter(out)`. The `a2a` combine is a single `all_to_all_single` and is faster at every token count measured.

Wall ms per combine call, MI355X (gfx950, 8-rank xGMI):

| tokens | `ag_rs` | `a2a` | speedup |
|---:|---:|---:|---:|
| 5 | 0.107 | 0.095 | 1.13x |
| 48 | 0.111 | 0.097 | 1.15x |
| 144 | 0.136 | 0.101 | **1.35x** |

The margin grows with T, and combine runs **per MLA layer per decode step**, so it multiplies by the layer count. Cosine similarity against `ag_rs` was **>= 0.999994** across the sweep.

`GlmMoeDsaForCausalLM` already does exactly this through the same `set_dcp_defaults` hook, added in #50382. This applies the same treatment to Kimi-K3.

Three deliberate limits on scope:

- **ROCm only.** The hook returns early off ROCm. `a2a` is one `all_to_all_single` against `allgather(lse)` + `reduce_scatter(out)`, so which side wins depends on the interconnect, and the numbers above are MI355X (gfx950, 8-rank xGMI). No other platform's default is moved by this change. Both branches are covered by tests.

- **`q_replicate` is not set.** `GlmMoeDsa` pairs `a2a` with `q_replicate=True`; that changes weight loading and is an independent question we have not measured, so it is left alone.
- **This changes only the default.** `set_dcp_defaults` fills options the user left unset, so an explicit `--dcp-comm-backend` still wins. A test pins that, because a "default" that silently overrode the flag would be worse than no default at all.

### Note on prefill context parallelism

`a2a` has a known caveat under PCP: #56677 pins `--dcp-comm-backend ag_rs` for a GLM-5.2 PCP4+DCP4 eval config, and GLM already defaults to `a2a`. The measurements above are **DCP-only, PCP off**; we have not measured `a2a` under PCP. Because this moves only the default, a K3 + PCP configuration can override it the same way that eval config does.

## Test Plan

```bash
pytest tests/models/test_kimi_k3_dcp_defaults.py

# no regression in the neighbouring model-config hook tests
pytest tests/models/test_qwen3_5_mtp_config.py
```

The new tests are CPU-only — they call the `verify_and_update_config` hook directly with a bare `ParallelConfig`, so no model is loaded. They pin four properties:

1. on ROCm, K3 defaults to the `a2a` combine;
2. **off ROCm the default is untouched**, so no unmeasured platform is affected;
3. an explicit `--dcp-comm-backend ag_rs` is **not** overridden;
4. `q_replicate` is left alone, so K3 does not inherit `GlmMoeDsa`'s pairing by accident;
5. the hook is registered for **both** K3 architectures — `KimiK3ForConditionalGeneration` and `KimiK3MTPModel` share the config class, and if only the main model were mapped, the target and its MTP draft would combine differently inside the same run;
6. **the dispatcher actually reaches the hook**, end to end through `VllmConfig.try_verify_and_update_config`, for both architectures. Points 1-5 prove the hook behaves correctly *when called* and that the map entry exists; this proves the two are connected. It uses the same `object.__new__` + `SimpleNamespace` model-config shape as `tests/model_executor/model_loader/test_modelexpress_loader.py`, so nothing is downloaded and no model config has to resolve.

## Test Result

```
tests/models/test_kimi_k3_dcp_defaults.py   7 passed
tests/models/test_qwen3_5_mtp_config.py    10 passed
```

The platform is patched rather than detected, so both branches are exercised on any runner — these tests are not silently vacuous on non-ROCm CI.

Mutations that turn the tests red:

```
remove the ROCm gate:                  1 failed, 6 passed   (off-ROCm default moved)
remove the set_dcp_defaults call:      1 failed, 6 passed   (on-ROCm default missing)
drop the MTP MODELS_CONFIG_MAP entry:  2 failed, 5 passed   (registration + dispatch)
all present:                           7 passed
```

`pre-commit run --files <changed files>` passes in full, including `mypy` 3.10-3.13, `typos`, `check-spdx-header` and `ruff`.

The hook is added to the existing `KimiK3ForConditionalGenerationConfig` class, which already defines `verify_and_update_model_config` for MXFP4 expert routing. These are two different hooks on the same class, so there is no interaction between them.
