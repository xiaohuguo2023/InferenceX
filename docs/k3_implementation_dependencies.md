# Kimi-K3 MI355X — implementation dependencies

Everything the DCP8 + DSpark + DRAM-offload recipe depends on, in one place:
upstream PRs we **consume**, PRs we **own**, and patches still carried locally.

Verified 2026-09-14 against vLLM `origin/main` and image
`vllm/vllm-openai-rocm:nightly-2671fedfc7ae604761990603fc736c0c4f21de57`
(vLLM `0.29.1rc1.dev9+g2671fedfc`, ROCm 7.2.3).

Supersedes the wave planning in `patches/k3-dcp8/UPSTREAM.md`, which was
surveyed 2026-09-03 and has since gone stale in two places (V8 is dead, V6 is
filed).

---

## 1. Upstream vLLM PRs we consume — all MERGED

These are hard dependencies: the recipe does not work without them. All are
present in the pinned image, so nothing here needs chasing.

| PR | title | why we need it |
|---|---|---|
| **#51705** | [ROCm][MLA][DCP] Support causal multi-token verification | the DCP verify framework our asm route plugs into |
| **#53598** | [ROCm][DSpark][DCP] Serve prefix cache hits under DCP for Kimi-K3 | fixes the DCP prefix-cache cliff; closed the ~20pp agentic cache deficit |
| **#55472** | [Bugfix][Spec Decode] Preserve target parallel config (DCP) for DSpark | boot blocker — without it the draft loses DCP and asserts on `dcp_manager` |
| **#52047** | [Bugfix][AMD] Annotate draft KV cache groups on the hybrid grouping path | required once past #47891 for KV offload + DSpark |
| **#50183** | [Bugfix][Spec Decode] Fix NaN handling in rejection sampler tl.argmax | DSpark `0x1016` crash; resets on any rebuild-from-base |
| **#53388** | [Feature][Spec] Support disabling trailing prefix-cache block dropping | landed after our first survey; supersedes part of our offload work |
| **#53407** | [Bugfix][MRV2] Dispatch uniform decode to a padded FULL cudagraph | **ours, merged** — keeps uniform decode on FULL graphs instead of eager |

## 2. PRs we own

| PR | state | content | notes |
|---|---|---|---|
| *(unopened)* | **pushed to fork** | asm round-robin (cprr) DCP verify + `max_split_per_batch` + `envs` + tests | branch `xguo/rocm-mla-dcp-cprr-verify`; 5 commits, +1077/−9; 540 tests pass vs 505 on base; pre-commit fully green; body in `docs/k3_dcp_cprr_pr_body.md` |
| **#55118** | OPEN / **draft** | [Bugfix][KV Offload] Don't pop a verified full-attn eagle prefix chunk | = `scheduler.patch`. Draft for 11 days; drafts get no review. Mark ready or close |
| **#53487** | OPEN / **draft** | [Kimi-K3] Split the KDA mixer out of piecewise CUDA graphs | draft for 22 days |
| **#53475** | OPEN / **draft** | [ROCm] Extend fused KDA decode to DSpark spec (num_spec<=2) | draft for 22 days. **Our own measurement says it loses on GPU time** — close it unless re-measured |
| **#53407** | MERGED | see above | done |

## 3. Local vLLM patches — status after the 2026-09-14 duplicate audit

| patch | lines | verdict |
|---|---|---|
| `rocm_aiter_mla.patch` | 390 | **filed** — drop when our PR merges |
| `scheduler.patch` | 16 | **filed** as #55118 |
| `speculator.patch` | 10 | **still ours, unfiled.** Quiesce ranks before DCP speculator capture; boot blocker (GPU fault). No upstream equivalent — the pinned image's `capture()` has no barrier. Needs `torch.cuda` → `torch.accelerator` (RFC #30679) before filing |
| `config.patch` | 27 | K3 DCP defaults. Convenience only — we pass `--dcp-comm-backend a2a` explicitly. Not audited for duplicates |
| `speculative_draft_dcp.patch` | 13 | **DEAD** — #55472 merged and is in our image, with a more robust fix. Delete |
| `dcp_a2a_pack_mask.patch` | 69 | **probably dead** — upstream ships `tests/v1/attention/test_dcp_a2a_pack_mask.py`. Confirm, then delete |
| `retention_alignment.patch` | 57 | not load-bearing — our values pass stock |
| `cp_common.patch` | 125 | not load-bearing — DCP dispatches PYNCCL, no symm_mem mesh is built. Genuine kernel-lifecycle bug worth filing **on its own merits**, not as part of the K3 story |

## 4. Non-vLLM dependencies

### ROCm/aiter — `patches/k3-dcp8/aiter/`
| id | change | load-bearing |
|---|---|---|
| A1 | take the tighter split-tile bound when a cap is supplied (`max`→`min`) | **yes** — MLA reduce scratch 9.35 → 2.38 GiB; what let FULL cudagraphs fit under DCP |
| A2 | ASM a16w16: don't auto-select split-K under graph replay | **yes** — boot blocker, all waves spin forever at seqs=64 warmup |
| A3 | fp8 MLA `get_block_n_fp8` fallback + 80/96/112 entries | **yes** — `KeyError` on any unlisted `nhead * max_seqlen_q`. NOT on the cprr path (verified), so it is not a dependency of our vLLM PR |
| A4 | K3 bf16 tuned-GEMM rows (CSV) | **yes** — 371 conc-1 tuned-config misses; absence HSA-faults the launcher |

A1 and vLLM's `max_split_per_batch` are **order-independent** — the aiter change
is gated on `max_split_per_batch > 0` with default `-1`, so it is inert for any
caller that does not pass a cap to the sizing call. Landing either first is safe;
the memory reclaim needs both.

**aiter #4494 was REVERTED upstream** — keep our local split-K guard.

### LMCache — `patches/k3-lmcache/`
`0001-retain-exported-ipc-events.patch`. Callers export an event's IPC handle
then drop their reference, so CPython destroys the event on return; HIP only
defers the real destroy while recorded work is outstanding, so once the copy
completes the handle stops being openable and the importer gets
`hipErrorInvalidValue`. Fix is a bounded retain ring
(`LMCACHE_EVENT_IPC_RETAIN`). **Unfiled — should be a PR to LMCache/LMCache.**

Also: LMCache's `nightly-rocm` index is **not immutable**. dev89, dev105 and
dev134 have all been withdrawn under us. Current pin `0.5.6.dev3+rocm7.2`.

### Model revisions — pin these, `main` has moved
| model | revision | note |
|---|---|---|
| `moonshotai/Kimi-K3` | `a590ce090cb049c93a33dfe8c208ec652aa20503` | `refs/main` is now `f831ab66…` — an unpinned fetch gets **different weights** |
| `Inferact/Kimi-K3-DSpark` | `cf6b8244620e7ea4b0651d214f28e89eac75bed6` | |

Re-stage after any reboot with `~/work/devshm_scripts/_fetch_k3.sh` (`/dev/shm`
is tmpfs; a reboot wipes 1.5 TB and resets the mount to 1.5 T — remount at
`size=2560G`, note `2.5T` is rejected).

## 5. Watch list

| PR | why it matters to us |
|---|---|
| **#56664** (open) | `@support_torch_compile` on `K3DSparkModel`. No `enable_if`, so at our `mode:3` it compiles the draft **by default**. On merge, every conc-1 number we hold becomes a pre-change measurement |
| **#56723** (open, draft, maintainer-authored) | "Replicated DFlash/DSpark drafts were still treated as DCP-sharded under DCP > 1" — directly adjacent to our ATOM-sharded draft; may overlap what we carry |

## 6. Standing lesson

`UPSTREAM.md` was written 2026-09-03 and by 09-14 had two entries that were no
longer true — one fixed upstream, one already filed by us. **Check every patch
against current `origin/main` and the PR search before proposing work on it.**
The check that caught #55472 took two minutes and saved a duplicate PR.
