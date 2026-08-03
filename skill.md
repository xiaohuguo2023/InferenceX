---
name: kimik3-mi355x-asm
description: Reusable command lines for Kimi-K3 fp8/bf16 ASM-MLA on MI355X (gfx950, TP8) — serve, IX-CI agentic sweep, GEMM tuning of untuned shapes, untuned-shape collection, and pareto build. Base-folder working notes (not part of the IX recipe PR).
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
AIPERF=/workspace/.aiperf_be758d/bin/aiperf
export HF_HUB_CACHE=/dev/shm/hf-cache HF_HOME=/dev/shm/hf-cache
```

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
  --compilation-config '{"mode":3,"cudagraph_mode":"FULL_AND_PIECEWISE"}' \
  --enable-prefix-caching --no-disable-hybrid-kv-cache-manager \
  --reasoning-parser kimi_k3 --tool-call-parser kimi_k3 --enable-auto-tool-choice \
  --disable-uvicorn-access-log --max-model-len 1048576 > serve.log 2>&1 &
# bf16 KV variant: --kv-cache-dtype auto (same asm backend)
# readiness:
for i in $(seq 1 144); do curl -s -m5 http://localhost:8888/health -o /dev/null && break; sleep 5; done
```

## 2. Agentic sweep — EXACT IX-CI harness (against a live serve)
Key CI knob: **`--agentic-cache-warmup-duration 600`** (time-bounded warmup). Do **NOT**
use `--warmup-requests-per-lane` — count-based warmup hangs draining the long-context tail
at conc≥16. Script: `_sweep_fp8asm_ixci.sh` (loops conc 1/4/8/16/24).
```bash
for c in 1 4 8 16 24; do
  seed=42; [ "$c" = 1 ] && seed=0
  out="/workspace/k3_fp8asm_ixci_c$c"; rm -rf "$out"; mkdir -p "$out/aiperf_artifacts"
  timeout 3000 "$AIPERF" profile --scenario inferencex-agentx-mvp --url http://localhost:8888 \
    --endpoint /v1/chat/completions --endpoint-type chat --streaming --model moonshotai/Kimi-K3 \
    --concurrency "$c" --benchmark-duration 1200 --stats-interval 30 --random-seed "$seed" \
    --failed-request-threshold 0.10 --trajectory-start-min-ratio 0.25 --trajectory-start-max-ratio 0.75 \
    --agentic-cache-warmup-duration 600 --warmup-grace-period 1800 \
    --use-server-token-count --no-gpu-telemetry --tokenizer-trust-remote-code \
    --num-dataset-entries 393 --slice-duration 1.0 \
    --output-artifact-dir "$out/aiperf_artifacts" --public-dataset semianalysis_cc_traces_weka_062126 \
    > "/workspace/k3_fp8asm_ixci_c$c.log" 2>&1
  pkill -9 -f "aiperf profile"; sleep 3
done
```

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
```
