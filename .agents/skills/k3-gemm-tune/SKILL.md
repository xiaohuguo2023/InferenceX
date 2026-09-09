# Tune K3 bf16 (a16w16) GEMMs on MI355X (`k3-gemm-tune`)

Close `not found tuned config` misses and answer "does any backend beat the current kernel
(e.g. wvSplitK) for shape (N,K) at M?" for Kimi-K3's dense bf16 GEMMs on MI355X (gfx950,
TP8). The aiter tuner benchmarks **flydsl / asm / hipBLASLt / opus** (and sometimes skinny)
per (M,N,K) and writes the **per-shape winner + µs**.

**CRITICAL — the tuner's winner does NOT include wvSplitK for most decode shapes.** The
tuner frequently enumerates **zero `skinny` candidates** for a given (N,K) (and `asm` fails
the 256-tile check when `N % 256 ≠ 0`, e.g. 2880/896). So its "winner" is really the best of
{opus, hipBLASLt, flydsl} — which can be **3× slower than wvSplitK** at the skinny M≤5 decode
regime. To actually answer "does anything beat wvSplitK", you MUST run the head-to-head
`_probe_wvsplitk_vs_opus.py` (below) — the tuner output alone is NOT sufficient. (MEASURED
2026-08-17 for N=2880,K=7168: tuner said opus wins ~17.7µs, but head-to-head wvSplitK 10.4µs
vs opus 32µs vs torch 24µs at n=2/3 — wvSplitK wins by 3×; memory
`k3-conc1-wvsplitk-vs-tuned-csv-regresses`.)

Use this when the user says any of: "tune the missing GEMM shape", "close the untuned-gemm
misses", "does X beat wvSplitK for (N,K)", "retune", "harvest the untuned GEMMs". Do **not**
hand-roll a `gemm_tuner.py` invocation or write a new tuner — the wrapper scripts below are
the canonical tooling; reuse them.

All tooling lives in `k3_gemm_tune/` (+ repo-root `_collect_untuned_gemm.py`,
`_make_tuning_csv.py`). The tuner binary is aiter's `gemm_tuner.py` at
`/opt/aiter-local/csrc/gemm_a16w16/gemm_tuner.py` **inside the container**; the repo is
bind-mounted at `/workspace`, so host edits are live.

## Core mental model — the runtime lookup decides which M you must tune

`aiter/tuned_gemm.py::get_GEMM_A16W16_config` keys on
`(gfx, cu_num, padded_M, N, K, bias, dtype, outdtype, scaleAB, bpreshuffle)` and tries
**three `padded_M` values in order, returning the FIRST hit**:

| order | `gl` | `padded_M` | meaning |
|---|---|---|---|
| 1st | `None` | **exact M** | most specific — a row tuned for this exact M wins |
| 2nd | `0` | `get_padded_m(M,N,K,0)` | a coarse bucket (e.g. **16** for all M≤16 on 2880×7168) |
| 3rd | `1` | `get_padded_m(M,N,K,1)` | next power-of-2 (3→4, 5→8, …) |

**Consequence:** to get a config *purpose-tuned for a decode M* (where `skinny`/wvSplitK
tends to win), the input CSV must contain that **EXACT M**. The generic ladder
`[1,2,4,8,16,…]` in `_make_tuning_csv.py` has no `3`, so an actual **M=3** (conc-1 nspec-2
target verify) would fall through to the M=16 bucket's winner — which may be a *different,
worse-at-M=3* kernel. Verify the buckets for your shape before choosing M:

```bash
docker exec <ctr> bash -lc 'cd /opt/aiter-local && HIP_VISIBLE_DEVICES=0 python3 -c "
from aiter.ops.gemm_op_common import get_padded_m
N,K=2880,7168
for M in [1,2,3,4,5,6,8,16]: print(M, get_padded_m(M,N,K,0), get_padded_m(M,N,K,1))"'
```

Decode M for DSpark = `(1+NUM_SPEC)` per sequence at conc-1 → **target verify M=3, draft
M=2** (nspec-2). Reason in the exact M the serve logs, not conc.

**`outdtype` must match what the serve logs.** bf16 GEMMs on this path log
`otype='torch.bfloat16'` → tune with `outdtype=torch.bfloat16`, or the merged row won't be
found at runtime (a different-outdtype row is a different key). Confirm from the actual
`not found tuned config` / `AITER_LOG_TUNED_CONFIG` line.

Input CSV columns (both collector and tuner): `M,N,K,bias,dtype,outdtype,scaleAB,bpreshuffle`.
Merged output is the 18-col `merged_bf16_tuned_gemm.csv`:
`gfx,cu_num,M,N,K,bias,dtype,outdtype,scaleAB,bpreshuffle,libtype,solidx,splitK,us,kernelName,err_ratio,tflops,bw`.

## The scripts — do not rewrite them

| script | role |
|---|---|
| `_collect_untuned_gemm.py` (repo root) | stdin serve-log → `kimik3_bf16_untuned_gemm.csv` + an (N,K) summary (flydsl-eligible flag, M range, occ). **GOTCHA:** its regex hard-codes the old log path `/tmp/aiter_configs/bf16_tuned_gemm.csv`; the current serve logs `.../configs/merged_bf16_tuned_gemm.csv` → update the regex path (or grep the shapes manually) before piping. |
| `_make_tuning_csv.py` (repo root) | expand each (N,K) in `kimik3_bf16_untuned_gemm.csv` to an M-ladder → `kimik3_bf16_tuning_gemm.csv`. Ladder has no `3`/`5`/`6` — **for targeted decode tuning, hand-add the exact decode M (2,3) instead** (see mental model). |
| `k3_gemm_tune/_run_tail_retune_8gpu_fast.sh` | **the main sweep.** Auto-derives distinct (N,K) from `$IN`, round-robins onto 8 GPUs, runs the tuner (`--libtype $LIBS --with-hipblaslt --batch 10 --shape_grouped`), merges shards, prints winners-by-libtype + per-M µs. Env: `IN OUT MMAX(256) LIBS(asm,opus,skinny)`. **flydsl deliberately excluded** (see Don't). Lets shards exit normally. |
| `k3_gemm_tune/_run_tail_flydsl_m72.sh` | **bounded** flydsl measurement at one decode M (default M=72), flydsl-only, one (N,K)/GPU, `timeout 1200`. Use to settle "does flydsl beat the fast-backend winner *here*?" by measurement — adapt the `NK` map + the `$1==<M>` awk filter to your shape/M. |
| `k3_gemm_tune/merge_tuned_shards.sh` / `_merge_recovered_bf16.py` | dedupe + merge shard outputs into `merged_bf16_tuned_gemm.csv` (keep last winner per shape). |
| `k3_gemm_tune/checkpoint_status.sh` / `checkpoint_compact.sh` | tuned-vs-remaining per shard; compact append-only shard CSVs. Resume-safe: the tuner loads `--tuned_file` and skips already-tuned shapes — keep existing outputs, don't delete. |
| `k3_gemm_tune/_probe_wvsplitk_vs_opus.py` | **the decisive wvSplitK head-to-head.** Isolated same-harness microbench of `ops.wvSplitK` (vLLM `_custom_ops`, NOT `aiter.ops`) vs `tgemm.mm` (opus, via `AITER_CONFIG_GEMM_BF16`→tuned CSV) vs torch, at the exact decode n. Use this to answer "does the tuned winner beat wvSplitK?" — the tuner can't, because it often skips skinny. Adapt N/K/NS for other shapes. |
| `k3_gemm_tune/README.md` | the dedicated portable tuning image (`k3-bf16-gemm-tune:gfx950`), libtype profiles, N=896 patches. |

## Two workflows

### A. Targeted single-shape ("does X beat wvSplitK for (N,K)?")
1. Get the shape metadata from the serve `not found` line (N, K, bias, dtype, outdtype,
   scaleAB, bpreshuffle) and the exact M values.
2. Build a small input CSV with the **EXACT decode M** (e.g. 1,2,3,4,5,6,8,16) — not just
   the ladder. Confirm `outdtype` matches the log.
3. Point `_run_tail_retune_8gpu_fast.sh` at it (one (N,K) → uses 1 GPU, fast):
   ```bash
   docker exec <ctr> bash -lc 'cd /workspace/k3_gemm_tune &&
     IN=/workspace/k3_gemm_tune/<in>.csv OUT=/workspace/k3_gemm_tune/<out>.csv \
     MMAX=64 LIBS=asm,opus,skinny bash _run_tail_retune_8gpu_fast.sh'
   ```
4. Read the printed **winners-by-libtype + per-M µs** — but treat it as "best of
   opus/hipBLASLt/flydsl", NOT as "beats wvSplitK" (skinny is usually absent; grep the shard
   log for `skinny` — if there are no skinny lines, it was never benchmarked).
5. **Decide with the head-to-head, not the tuner.** Point `AITER_CONFIG_GEMM_BF16` at the
   tuned CSV and run `_probe_wvsplitk_vs_opus.py` (adapt N/K/NS). If **wvSplitK** is fastest
   at the decode n → it's optimal, **keep it, do NOT flip `VLLM_ROCM_USE_SKINNY_GEMM`, do NOT
   merge** the tuned rows (they'd only be used with skinny off and are worse). Only if the
   tuned backend beats wvSplitK → merge (workflow B step 4+) and confirm with an in-serve A/B.
6. (Optional) measure flydsl separately at the same M via `_run_tail_flydsl_m72.sh` (adapted).

### B. Harvest-and-close (bulk untuned misses from a serve)
1. Serve with `AITER_LOG_TUNED_CONFIG=1`; run a representative load; grep the serve log for
   `not found tuned config`.
2. Pipe those lines through `_collect_untuned_gemm.py` (fix the path regex first) →
   `kimik3_bf16_untuned_gemm.csv`; run `_make_tuning_csv.py` → `kimik3_bf16_tuning_gemm.csv`.
   Add exact decode M by hand if any decode shape is present.
3. `IN=kimik3_bf16_tuning_gemm.csv OUT=<tuned>.csv bash _run_tail_retune_8gpu_fast.sh`
   (shards across 8 GPUs). For large-N decode shapes, optionally add bounded flydsl.
4. **Merge additively** into `merged_bf16_tuned_gemm.csv` (keep existing rows; dedupe last
   winner per key) via `merge_tuned_shards.sh` / `_merge_recovered_bf16.py`.
5. Re-serve with the merged CSV wired via `AITER_CONFIG_GEMM_BF16` (the
   `_serve_k3_bench_spec.sh` serve does this automatically). Verify the specific lookups now
   log `found padded_M:` / are absent from `not found tuned config` (count the target N,K).

## Hard rules / gotchas

- **NEVER SIGKILL a tuner child.** Under the container's `sleep infinity` init a `-9`'d
  `gemm_tuner.py`/child zombie-pins its GPU context. Let shards **exit normally** (the
  wrappers use `timeout -k 60 <T>` = SIGTERM then SIGKILL only after grace). To stop early:
  `pkill -TERM -f 'gemm_tuner.py|gemm_a16w16_tune'`, wait, then confirm VRAM drains to ~0.3
  GiB/GPU. (memory `aiter-gemm-tuner-single-gpu-shard`, `vram-leak-zombie-init-root-cause`)
- **`--shape_grouped` uses only ONE GPU per invocation** (mp_tuner respawn bug) → shard
  distinct (N,K) across 8 GPUs (what `_run_tail_retune_8gpu_fast.sh` does). A single (N,K)
  targeted run correctly uses 1 GPU — that's fine, it's fast.
- **Keep flydsl OUT of the bulk sweep.** flydsl JIT-compiles every candidate tiling from
  scratch (`flydsl/compiler/jit_function.py`) → 100% CPU / 0% GPU, **>3 min/shape** at large
  N → the sweep stalls / times out (this is why my first hand-rolled `--libtype flydsl,...`
  run crawled). asm/opus/skinny/hipBLASLt pick from prebuilt kernels/solidx (seconds/shape).
  **Measure** flydsl instead of assuming — but only **bounded** to the decode-critical M via
  `_run_tail_flydsl_m72.sh`.
- **Large-M prefill (>2048) GPU-faults some asm/opus kernels** under mp_tuner. Use the shape
  guards `AITER_TUNE_ASM_MAX_M=2048 AITER_TUNE_OPUS_MAX_M=2048` (or
  `AITER_TUNE_DISABLE_ASM=1`/`_OPUS=1`), or run large M with hipBLASLt-only. Decode M (≤256)
  is safe with the full backend set.
- **`N=896, K=7168` MoE shapes** need `_patch_gemm_n896.py` (asm 256-tile rejects
  896%256≠0; hipBLASLt enumerates ~240k solutions) — see README.
- **Tuned kernel indices are hardware- AND aiter-build-specific.** Tune on gfx950 with the
  **same aiter build** that serves (`/opt/aiter-local`), or the `libtype`/`solidx` won't
  match at serve time. The dedicated image bakes the serving aiter for this reason.
- Preflight: box free (`rocm-smi` ~0.3 GiB/GPU) and no live colleague TP8 python
  (`<defunct>`/UNKNOWN ≠ live — don't reclaim a colleague's job). Serialize with any serve.

## Don't
- Don't hand-roll the `gemm_tuner.py` command line — use `_run_tail_retune_8gpu_fast.sh`
  (it encodes sharding, `--with-hipblaslt`, merge, winner-report, and normal-exit discipline).
- Don't include `flydsl` in the bulk `LIBS` (JIT-compile stall) — measure it bounded instead.
- Don't SIGKILL to stop a run (zombie-pins GPU ctx). SIGTERM and wait for VRAM drain.
- Don't tune only the ladder for a decode shape — add the **exact** decode M (2,3) or the
  runtime pads to the coarse M=16 bucket and may pick a worse kernel.
- Don't tune with the wrong `outdtype` — match the serve log (`torch.bfloat16` on this path)
  or the row is a dead key.
- Don't assume a non-skinny winner without checking µs vs `skinny` in the tuner output —
  wvSplitK is purpose-built for the skinny M≤5 regime and often wins there (memory
  `k3-conc1-wvsplitk-vs-tuned-csv-regresses`).
- **Don't merge tuner output without also benchmarking plain `torch`.** MEASURED 2026-09-09:
  across 275 freshly tuned K3 decode shapes `skinny` won **0** times (174 asm, 101 opus,
  because the tuner never enumerates it) — and at n=5..10 those winners lose to
  `torch.nn.functional.linear` by **20–43% on 7 of 8 shapes** (1536×7168 @ n=8: tuned 44.7 µs
  vs torch 25.6 µs). The tuner's field is {asm, opus, hipBLASLt}; it contains neither wvSplitK
  nor torch, so its "winner" can be the *worst* of the four real options. Use
  `_deadzone_probe.py` (repo root) for the three-way wvSplitK / tuned / torch at n=5..10.
- **Know which n actually reaches the tuned path.** `utils.py` routes n=1→LLMM1, 2–5→wvSplitK,
  10–128→wvSplitKrc, and leaves **n∈[6,9] with no skinny path** — conc-1 at K=7 gives n=8, so
  it falls through to the tuned/asm path. That dead zone is a missing `switch` instantiation in
  `csrc/rocm/skinny_gemms.cu` (cases 1..5 only; the kernel body is generic in `N`), not a
  kernel limit — memory `k3-skinny-dead-zone-is-missing-instantiation`.
