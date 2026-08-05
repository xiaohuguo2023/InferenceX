# Kimi-K3 bf16 GEMM tuning image (MI355X / gfx950)

Self-contained image to tune K3's bf16 (a16w16) GEMMs on any MI355X. It bakes the exact
aiter build that serves K3 (commit `00cbe979f`), so the tuned `libtype`/`solidx` match at
serve time. The tuner benchmarks **flydsl / asm / hipBLASLt / skinny / opus** per (M,N,K)
and writes the per-shape winner — not "all flydsl" (see
`../docs/kimik3_bf16_gemm_flydsl_tuning_shapes.md`).

## Why an image
Tuning must run on **MI355X (gfx950)** with the **same aiter build** as the serve, because
tuned kernel indices are hardware- and build-specific. Baking aiter into the image makes it
portable to a dedicated tuning box (no serve contention, GPU free).

## Build (on a box with `~/work/aiter @ 00cbe979f`)
```bash
cd k3_gemm_tune
./build.sh                       # stages aiter (minus .git) + CSV, builds k3-bf16-gemm-tune:gfx950
```
Overrides: `AITER_SRC=/path/to/aiter CSV=/path/to/shapes.csv TAG=myimg ./build.sh`

## Ship to another MI355X
```bash
docker save k3-bf16-gemm-tune:gfx950 | zstd > k3tune.tzst
#   copy k3tune.tzst over, then:
zstd -d k3tune.tzst -c | docker load
```

## Run (produces the tuned CSV in the current dir)
```bash
docker run --rm --device /dev/kfd --device /dev/dri --group-add video \
  --security-opt seccomp=unconfined --ipc host --shm-size 16g \
  -v $PWD:/work k3-bf16-gemm-tune:gfx950
# -> ./kimik3_bf16_tuned_gemm.csv
```
Knobs (env): `TUNE_LIBTYPE_PROFILE=safe|n896|full` (default **safe**),
`LIBTYPE=...` (override), `INPUT_CSV=...`, `OUTPUT_CSV=/work/...`.
First run does a ~10s one-time aiter-core JIT rebuild.

### Libtype profiles (avoid GPU faults during tuning)

| Profile | Backends | When to use |
|---------|----------|-------------|
| **safe** (default) | flydsl, hipblaslt, skinny | Main 481 shapes — no opus/asm |
| **n896** | + opus | MoE `N=896,K=7168` only (asm skipped via N%tileN patch) |
| **full** | + opus, asm | Small-M only: `_patch_gemm_tune_safe.py` skips asm/opus when `M > 2048` |

Opus already has host rules (`kid_rejects_shape`, 4g_safe, K-parity) in aiter; large-M
prefill (4096/32768 × 1024) still GPU-faults some opus/asm kernels under mp_tuner.
Plain `torch.mm` is fine — faults are backend-specific.

Shape guard env (used when profile includes asm/opus): `AITER_TUNE_ASM_MAX_M=2048`,
`AITER_TUNE_OPUS_MAX_M=2048`, `AITER_TUNE_DISABLE_ASM=1`, `AITER_TUNE_DISABLE_OPUS=1`.

### N=896 MoE shapes (gfx950)
MoE expert projection uses `N=896, K=7168`. The asm `256×256` tile rejects `896 % 256 ≠ 0`,
and hipBLASLt can enumerate ~240k solutions per shape. `tune.sh` applies live-mount patches:
- `_patch_gemm_n896.py`: skip asm when `N % tileN != 0`; cap hipBLASLt fast-mode
- `_patch_gemm_tune_safe.py`: skip asm/opus when `M > AITER_TUNE_*_MAX_M` (default 2048)

Legacy fallback: `./setup_and_tune.sh tune-split` (two-phase CSV split).

### Checkpoint / resume

Each shard writes append-only progress to `shards/kimik3_bf16_tuned_main_sN.csv`.
The aiter tuner loads that file as `--tuned_file` and **skips shapes already tuned**.

```bash
./checkpoint_status.sh              # tuned vs remaining per shard
./checkpoint_compact.sh             # dedupe output CSVs (keep last winner per shape)
SKIP_SPLIT=1 CHECKPOINT_COMPACT=1 NUM_SHARDS=4 SHARD=2 ./tune_shard.sh bg
```

On restart, keep existing output CSVs on NFS — do not delete them. Use `SKIP_SPLIT=1`
so shard input CSVs are not reshuffled.

## Install the result on the serving box
```bash
cp kimik3_bf16_tuned_gemm.csv \
   <aiter>/aiter/configs/model_configs/kimik3_bf16_tuned_gemm.csv
#   (or /tmp/aiter_configs/bf16_tuned_gemm.csv), then re-serve.
```
The serve must run the **same aiter build**; then `tgemm.mm` dispatches each shape to its
tuned kernel (flydsl at decode-M, hipBLASLt/asm at prefill-M) instead of the torch
fallback — the 667k `not found tuned config` warnings disappear.

## Files
- `Dockerfile` — bakes aiter + shape CSV + entrypoint.
- `tune.sh` — entrypoint: rebuilds aiter core (first run) then runs the a16w16 tuner.
- `build.sh` — stages a clean context and builds the image.
