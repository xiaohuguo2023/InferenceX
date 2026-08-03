# aiter PR notes — PR-B (#4452) and PR-C (bf16 GEMM config)

## PR-B — land ROCm/aiter **#4452** (64-bit paged-KV byte offsets, gfx950 MLA HSACO)

**Not a new PR.** Our local branch (`~/work/aiter @ 6fc5733b7`) is a straight cherry-pick of
#4452, verified byte-identical:

- `csrc/py_itfs_cu/asm_mla.cu`: the same 4-line `s_MQA` fix
  (`index a677ccf16..36da571c1`, `@@ -895,6 +895,10 @@`, guarded by
  `persistent && arch_id == "gfx950" && max_seqlen_q >= 3`).
- 26 refreshed `hsa/gfx950/mla/*.co` HSACO (a16w16 + a8w8 qseqlen4 / prefill / decode
  variants) with 64-bit paged-KV byte offsets.

**Action:** review/validate & push #4452 to merge. Add our K3 evidence to the PR thread:

> Validated on Kimi-K3 (mxfp4), 8×MI355X (gfx950), TP8, `--kv-cache-dtype fp8`,
> `ROCM_AITER_MLA`. Before: long-context MLA crashed once the paged-KV pool crossed the
> 32-bit byte-offset boundary. After #4452: fresh 470k / 590k contexts decode+prefill
> correctly, no offset truncation. Cherry-picked cleanly onto aiter `00cbe979f`.

**Relationships:**
- **#4341 (MERGED)** — the qh16 fp8 *persistent decode* HSACO refresh; already in our build,
  fixes the decode kernel offsets. #4452 extends the same fix to the a8w8/a16w16 qseqlen4 /
  prefill kernels + the `asm_mla.cu` source guard.
- **#4351 (OPEN)** — earlier/subset (includes `CKV_mem_va_upd`); **superseded by #4452**.
- **#4474 (OPEN)** — int32 KV-offset overflow in the Triton `_mla_gluon` path (>2 GB). Same
  bug class, different kernel; we use asm so it's not on our path, but worth landing for the
  gluon fallback.
- **#4480 / #4488 (OPEN)** — gluon fp8 small-head decode + a >2 GB regression test; not on
  our asm path.

## PR-C — add `configs/model_configs/kimik3_bf16_tuned_gemm.csv`

**Blocked on tuner run.** Removes the ~667k `not found tuned config … using torch` fallbacks
for K3's bf16 (a16w16) GEMMs by shipping a per-shape winner config.

- Source shapes: `kimik3_bf16_tuning_gemm.csv` (424 shapes, 21 distinct (N,K)).
- Tuner: `k3_gemm_tune/` image — bakes the exact serving aiter build (`00cbe979f`) so tuned
  `libtype`/`solidx` match at serve time. Benchmarks flydsl / asm / hipBLASLt / skinny / opus
  per (M,N,K) and writes the per-shape winner (mixed: flydsl at decode-M, hipBLASLt/asm at
  prefill-M; ~6288/16160 stay hipBLASLt).
- **Action:** run the tuner on a free MI355X, drop the resulting CSV into
  `aiter/configs/model_configs/`, file as a config-only PR.

**Do not** file PR-C until the tuner has actually produced the CSV on gfx950 with the matching
aiter build — a config tuned on a different build dispatches to wrong kernel indices.
