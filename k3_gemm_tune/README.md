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
Knobs (env): `LIBTYPE=all|flydsl|asm|...` (default all + hipBLASLt), `INPUT_CSV=...`,
`OUTPUT_CSV=/work/...`. First run does a ~10s one-time aiter-core JIT rebuild.

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
