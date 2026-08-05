---
name: kimik3-mi355x-asm
description: Reusable command lines for Kimi-K3 fp8/bf16 ASM-MLA on MI355X (gfx950, TP8) — container bring-up (xguo-k3asm), env build (aiter #4452 change recipe + aiperf v1.0.1 venv), serve, IX-CI agentic sweep, GEMM tuning of untuned shapes, untuned-shape collection, and pareto build. Base-folder working notes (not part of the IX recipe PR).
---

# Kimi-K3 MI355X (gfx950, TP8) — ASM MLA runbook

Working command lines for the fp8/bf16 **ASM** path (`ROCM_AITER_MLA`). The committed
IX recipe is `benchmarks/single_node/agentic/kimik3_fp4_mi355x_vllm.sh`; the scripts
referenced here live in this base folder (untracked, not in the PR).

**Image prereqs for the ASM path** (else 12 heads/rank break on stock vLLM):
vLLM **#50578** (asm decode pad-to-16) + **PR-A** (fp8 asm prefill pad-to-16 + 16-head PS
metadata) + ROCm/aiter **#4452** (64-bit paged-KV offsets).

Common vars:
```bash
MODEL_PATH=/dev/shm/hf-cache/models--moonshotai--Kimi-K3/snapshots/9f62e4e9fffbd0a83ddd60e1c209d828994b3569
AIPERF=/workspace/.aiperf_v1_0_1/bin/aiperf   # aiperf v1.0.1 (agentx-v1.0.1) — see §0b
export HF_HUB_CACHE=/dev/shm/hf-cache HF_HOME=/dev/shm/hf-cache
```

## 0. Container bring-up (xguo-k3asm)

The serving container is the K3 ROCm image with the repo, aiter, and /dev/shm bind-mounted
(reconstructed from `docker inspect xguo-k3asm`):
```bash
docker run -d --name xguo-k3asm \
  --device /dev/kfd --device /dev/dri --group-add video \
  --ipc host --network host --shm-size 128g \
  --security-opt seccomp=unconfined --security-opt apparmor=unconfined --security-opt label=disable \
  --cap-add CAP_SYS_PTRACE \
  -v /data:/data -v /dev/shm:/dev/shm \
  -v ~/work/InferenceX-dspv4:/workspace \
  -v ~/work/aiter:/aiter-latest \
  -e HF_HUB_CACHE=/dev/shm/hf-cache \
  vllm/vllm-openai-rocm:kimi-k3 sleep infinity
# image env of note: AITER_ROCM_ARCH="gfx942;gfx950", HIP_FORCE_DEV_KERNARG=1, PYTHON=3.12
```
Wiring:
- **aiter** is imported straight from the `/aiter-latest` bind-mount
  (`aiter.__file__ = /aiter-latest/aiter` = host `~/work/aiter`) — apply §0a (#4452) on the
  host and it is live in the container (rebuild the extension for `asm_mla.cu`).
- **vLLM** is the image's site-packages build; the ASM patches are applied **in place** (below).
- Model staged in `/dev/shm/hf-cache` (tmpfs) for fast TP8 load; `/workspace` = this repo.

Stage the model once (into /dev/shm):
```bash
HF_HUB_CACHE=/dev/shm/hf-cache hf download moonshotai/Kimi-K3
```

Apply the vLLM ASM patches in place (until #50578 + PR-A + #50618 ship in the image). The
`_patch_*.py` are committed on this branch (`xguo/kimik3-fp4-mi355x-vllm-recipe`), so they
appear at `/workspace/` only when the mounted repo has this branch checked out — a clean
`main` won't have them. They edit `dist-packages/vllm/...` idempotently by string-anchor:
```bash
python /workspace/_patch_fp8asm.py         # decode pad-to-16            (== vLLM #50578)
python /workspace/_patch_fp8_prefill.py    # fp8 asm prefill pad-to-16   (PR-A L3)
python /workspace/_patch_ps_metadata16.py  # PS metadata num_head_k=16   (PR-A L4)
python /workspace/_patch_wvsplitk.py       # wvSplitK strided contiguity (== vLLM #50618)
# verify both patch sites present:
grep -l "PATCH(fp8-prefill-pad)" /usr/local/lib/python3.12/dist-packages/vllm/v1/attention/backends/mla/rocm_aiter_mla.py
```
Then build the aiperf venv (§0b) and serve (§1).

## 0a. aiter change recipe (required to build the ASM MLA env)

The ASM MLA path needs aiter with the **large-page_id (>4 GB paged-KV) offset fix,
ROCm/aiter #4452**, on top of the K3-serving base build. In the container aiter is
bind-mounted from the host: `~/work/aiter` → `/aiter-latest` (editable-installed, so host
edits are picked up on re-import).

```bash
cd ~/work/aiter
git log --oneline -1                     # base build: 00cbe979f (FHMoE mixed-MoE)
# Apply ROCm/aiter#4452 — 64-bit paged-KV byte offsets for gfx950 MLA HSACO:
#   csrc/py_itfs_cu/asm_mla.cu  s_MQA fix (4 lines, gfx950 && max_seqlen_q>=3)
#   + 26 refreshed hsa/gfx950/mla/*.co  (a16w16 + a8w8 qseqlen4/prefill/decode)
git fetch <aiter-remote> pull/4452/head:pr-4452 && git cherry-pick pr-4452   # -> HEAD 6fc5733b7
# (#4341 qh16 fp8 persistent-decode HSACO for large page_id is already MERGED in the base)
```
Make the change active in the serve:
- The `.co` HSACO are loaded **by path at runtime** → refreshing the files is enough for the
  decode/prefill kernels.
- `asm_mla.cu` is compiled into the aiter extension → **rebuild** it so the s_MQA fix takes
  effect:
  ```bash
  cd /aiter-latest && pip install -e . --no-build-isolation   # or the one-time JIT core rebuild on first import
  ```
- Verify: fresh 470k / 590k single-request prefills complete with no >4 GB paged-KV offset
  truncation (this is exactly what #4452 fixes).

Canonical, portable build of this exact aiter state (for a fresh MI355X): the
`k3_gemm_tune/Dockerfile` bakes aiter @ the serve commit — reuse it as the reference build.

(Optional) install the tuned bf16 GEMM config from §4:
```bash
cp kimik3_bf16_tuned_gemm.csv /aiter-latest/aiter/configs/model_configs/kimik3_bf16_tuned_gemm.csv
```

## 0b. aiperf env build (v1.0.1 — the current IX pin)

The IX agentic harness needs `--trace-idle-gap-cap-seconds` (rejected by the old `be758d62`
build). IX now pins aiperf **v1.0.1** (tag `agentx-v1.0.1`, commit `b7b16cf8`). Build it into
an isolated venv (must not share site-packages with vLLM):
```bash
git submodule update --init utils/aiperf                 # or: git -C utils/aiperf checkout agentx-v1.0.1
uv venv --python 3.11 /workspace/.aiperf_v1_0_1
uv pip install --python /workspace/.aiperf_v1_0_1/bin/python \
  -r /workspace/utils/agentic-benchmark/requirements.txt -e /workspace/utils/aiperf \
  "datasets>=4.7.0" "huggingface_hub[cli]>=0.25.0" urllib3 requests
/workspace/.aiperf_v1_0_1/bin/aiperf --version           # sanity (reports 0.12.0; the tag is the truth)
```
Flag note: v1.0.1 uses `--prompt-input-tokens-mean` (older docs/commands say
`--synthetic-input-tokens-mean` — same meaning). `--sweep-type` is supported.

## 1. Serve (validated uncapped ASM config)
```bash
export VLLM_ROCM_USE_AITER=1 VLLM_ROCM_USE_AITER_MOE=1 SAFETENSORS_FAST_GPU=1 \
       AITER_SITUV2_A8W4=1 AITER_BF16_FP8_MOE_BOUND=0 VLLM_USE_BREAKABLE_CUDAGRAPH=0 \
       GPU_ARCHS=gfx950 VLLM_ENGINE_READY_TIMEOUT_S=3600 \
       VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=3600 VLLM_HTTP_TIMEOUT_KEEP_ALIVE=900
setsid nohup vllm serve "$MODEL_PATH" --served-model-name moonshotai/Kimi-K3 \
  --host 0.0.0.0 --port 8888 --tensor-parallel-size 8 --async-scheduling \
  --distributed-executor-backend mp --gpu-memory-utilization 0.95 \
  --max-num-seqs 64 --max-num-batched-tokens 4096 \
  --trust-remote-code --load-format auto --moe-backend auto \
  --kv-cache-dtype fp8 --attention-backend ROCM_AITER_MLA --mm-encoder-tp-mode data \
  --compilation-config '{"mode":3,"cudagraph_mode":"FULL_AND_PIECEWISE","custom_ops":["+fused_rms_norm_gated"]}' \
  --enable-prefix-caching --no-disable-hybrid-kv-cache-manager \
  --reasoning-parser kimi_k3 --tool-call-parser kimi_k3 --enable-auto-tool-choice \
  --disable-uvicorn-access-log --max-model-len 1048576 > serve.log 2>&1 &
# bf16 KV variant: --kv-cache-dtype auto (same asm backend)
# +fused_rms_norm_gated (KDA gated-RMSNorm) is ON by default above (capture-verified,
#   perf-neutral A/B). To disable: drop custom_ops, or FUSED_RMS_NORM_GATED=0 in the recipe.
# readiness:
for i in $(seq 1 144); do curl -s -m5 http://localhost:8888/health -o /dev/null && break; sleep 5; done
```

## 2. Agentic sweep — IX-CI harness (matches build_replay_cmd; against a live serve)
Use the committed `_sweep_fp8asm_ixci.sh` (needs the §0b aiperf `v1.0.1` venv). It mirrors
`benchmarks/benchmark_lib.sh:build_replay_cmd`:
```bash
TAG=fp8asm CONC_LIST="1 4 8 16 24" OUT_ROOT=/workspace bash _sweep_fp8asm_ixci.sh
```
The per-conc aiperf invocation it runs:
```bash
"$AIPERF" profile --scenario inferencex-agentx-mvp --url http://localhost:8888 \
  --endpoint /v1/chat/completions --endpoint-type chat --streaming --model moonshotai/Kimi-K3 \
  --concurrency "$c" --benchmark-duration 1200 --stats-interval 30 --random-seed "$seed" \
  --failed-request-threshold 0.10 --trajectory-start-min-ratio 0.25 --trajectory-start-max-ratio 0.75 \
  --warmup-requests-per-lane 10 --trace-idle-gap-cap-seconds 300 --warmup-grace-period 1800 \
  --use-server-token-count --no-gpu-telemetry --tokenizer-trust-remote-code \
  --num-dataset-entries 393 --slice-duration 1.0 \
  --output-artifact-dir "$out/aiperf_artifacts" --public-dataset semianalysis_cc_traces_weka_062126
```
Correctness rules (learned the hard way):
- **`--trace-idle-gap-cap-seconds 300` is required** and needs aiperf ≥ `v1.0.1` (§0b);
  without the cap the cc-traces trajectories replay full real-world idle gaps and warmup
  never drains at conc≥16. Do **not** use `--agentic-cache-warmup-duration` (not in
  build_replay_cmd) or `--unsafe-override`.
- **No shell `timeout`** — aiperf self-bounds (warmup-grace 1800 + benchmark-duration 1200
  already = 3000 s before send/export; a tight `timeout` kills a healthy run). The script
  uses `set -euo pipefail` so a config error stops before any GPU-freeing step.

## 3. Collect untuned GEMM shapes from the serve log
Turns the `not found tuned config … bf16_tuned_gemm.csv` warnings into a tuner CSV.
```bash
grep -h 'not found tuned config' serve.log | \
  sed -E "s/.*M:([0-9]+), N:([0-9]+), K:([0-9]+) dtype='([^']+)' otype='([^']+)' bias=([A-Za-z]+), scaleAB=([A-Za-z]+), bpreshuffle=([A-Za-z]+).*/\1,\2,\3,\6,\4,\5,\7,\8/" | \
  sort -t, -k2,2n -k3,3n -k1,1n -u > /tmp/untuned.csv
{ echo "M,N,K,bias,dtype,outdtype,scaleAB,bpreshuffle"; cat /tmp/untuned.csv; } > kimik3_bf16_untuned_current.csv
```
Curated tuning set (adds kv_b_proj large-M ladder): `kimik3_bf16_tuning_gemm_v2.csv`.

## 4. Tune the untuned bf16 GEMMs (k3_gemm_tune image)
Must run on MI355X with the **same aiter build** as the serve (tuned kernel indices are
build-specific). Kit: `k3_gemm_tune/` (Dockerfile + build.sh + tune.sh + README).
```bash
cd k3_gemm_tune && ./build.sh          # bakes aiter @ serve commit -> k3-bf16-gemm-tune:gfx950
docker run --rm --device /dev/kfd --device /dev/dri --group-add video \
  --security-opt seccomp=unconfined --ipc host --shm-size 16g \
  -v $PWD:/work -e INPUT_CSV=/work/kimik3_bf16_tuning_gemm_v2.csv \
  k3-bf16-gemm-tune:gfx950                # -> ./kimik3_bf16_tuned_gemm.csv
# install on the serving box (same aiter build), then re-serve:
cp kimik3_bf16_tuned_gemm.csv <aiter>/aiter/configs/model_configs/kimik3_bf16_tuned_gemm.csv
```
Ship the image to another MI355X: `docker save k3-bf16-gemm-tune:gfx950 | zstd > k3tune.tzst`.

## 5. Build the fp8-vs-bf16 pareto (interactivity vs tput/GPU)
```bash
python3 - <<'PY'
import json
NGPU=8
def val(m):
    if isinstance(m,(int,float)): return float(m)
    return next((float(m[k]) for k in ("avg","mean","value","p50") if isinstance(m,dict) and m.get(k) is not None), 0.0)
for c in (1,4,8,16,24):
    d=json.load(open(f"/workspace/k3_fp8asm_ixci_c{c}/aiperf_artifacts/profile_export_aiperf.json"))
    dur=val(d["benchmark_duration"]); tp=(val(d["total_isl"])+val(d["total_osl"]))/dur/NGPU
    tpot=val(d["inter_token_latency"])
    print(f"conc{c:>3}  tput/gpu={tp:7.0f}  TPOT={tpot:6.1f}ms  interact={1000/tpot:6.2f}")
PY
```

## 6. Free the GPUs (shared machine)
```bash
pkill -9 -f "vllm serve"; pkill -9 -f "aiperf profile"; pkill -9 -f EngineCore
# orphaned VRAM: sudo kill -9 $(rocm-smi --showpids | awk 'NR>...') as needed
# NOTE: never `pkill -f vllm` from an inline shell whose own cmdline contains
# "vllm"/"EngineCore" — it self-matches (exit 137). Use a script file (_freegpu.sh).
```

## 7. Best combinations & key findings (read this first)

**Winning fp8-ASM serve config** (as in §1). Serve knobs that MATTER:
- `--attention-backend ROCM_AITER_MLA` + `--kv-cache-dtype fp8` — the asm path (needs the
  image to carry vLLM #50578 + PR-A + ROCm/aiter #4452; else 12 heads/rank break).
- `--no-disable-hybrid-kv-cache-manager` — K3 MLA+KDA hybrid; **do not omit**.
- **`--gpu-memory-utilization 0.95`** — always. More KV pool / prefix-cache retention; do
  **not** drop to 0.8.
- **`--max-num-batched-tokens` is context-dependent — mind the OOM:**
  - **Agentic serve (ms64, uncapped 1M): keep `4096`.** The default (8192) doubles the prefill
    activation arena and **OOMs during warmup at conc16** on top of ~72 GiB KV at ms64/gpu-0.95
    (`HSA_STATUS_ERROR_OUT_OF_RESOURCES`, 0 MB free — validated 2026-08-05). Don't raise it
    without lowering `--max-num-seqs` or `--gpu-memory-utilization`.
  - **Synthetic prefixbench (ms24): leave at default (8192).** There it does *not* OOM (lower
    ms) and 4096 would chop the 63k prefill into ~16 tiny chunks (~half prefill throughput).
- **`--compilation-config '{"mode":3,"cudagraph_mode":"FULL_AND_PIECEWISE","custom_ops":["+fused_rms_norm_gated"]}'`**
  — `+fused_rms_norm_gated` (KDA gated-RMSNorm) is **on by default now**. Plus `--async-scheduling`.

Lower-impact at conc ≤ ~24: `--max-num-seqs` 24/64/128 (non-binding below the concurrency);
`--moe-backend auto` ≈ `aiter`.

**Two benchmarks — do not confuse them:**

| | **IX perf CI (the metric)** | synthetic 68k-prefix (stress test) |
|---|---|---|
| driver | `_sweep_fp8asm_ixci.sh` (§2) | `_run_prefixbench.sh` |
| scenario | `inferencex-agentx-mvp` (cc-traces) | `--num-prefix-prompts 8 --prompt-prefix-length 63240` |
| cache model | **warm within a session; first-turn prefix cache-busted** (realistic) | **one 63k prefix shared across all requests** (artificial) |
| the number | canonical pareto (e.g. fp8 c8 ≈ 2272 tok/s/GPU) | warm ≈ 9–12K, cold ≈ 4.8–6K in tok/s/GPU |

- **IX CI = the agentic scenario.** It warms each trajectory during warmup then profiles from
  the live warm state (`--trajectory-start-min/max-ratio 0.25/0.75`), and **locks
  `--cache-bust first_turn_prefix`** so sessions don't share a fake prefix. That is the
  dashboard number — realistic warm-per-session, not cold-start, not shared-prefix-warm.
- **Synthetic 68k-prefix** does the opposite (shares a 63k prefix, no cache-bust) to stress
  prefix caching. Its throughput swings **~2× on cold vs warm cache**: `--warmup-request-count
  3` can't prime 8 cold 63k prefixes, so the first measured requests eat cold prefills
  (TTFT-P90 ~33 s, ~half tput). Warm all 8 prefixes first for a stable number. **Not the CI metric.**

**Perf notes (settled this session):**
- `+fused_rms_norm_gated` is **perf-neutral** on a clean same-serve A/B — now enabled by default.
- The apparent "9K → 4.8K regression" was **cold-vs-warm prefix cache**, not fp8, not the
  fused op, not `--max-num-batched-tokens`, not gpu-mem, not the aiperf version — all ruled out.
- Agentic tput/GPU is input-dominated + heavy-tailed → run-average is cold-start-sensitive;
  prefer the full conc 1–24 sweep and/or steady-state (post-ramp) slices.
