#!/usr/bin/env bash
# Kimi-K3 fp8/bf16 ASM-MLA benchmark docker setup (skill.md).
#
# Run on a compute node (job 34891 / crsuse2-m2m-002):
#   cd ~/work/InferenceX
#   ./setup_benchmark.sh start          # pull image + start long-lived container
#   ./setup_benchmark.sh setup          # install live aiter + aiperf venv
#   ./setup_benchmark.sh shell          # interactive shell in container
#   ./setup_benchmark.sh serve-fp8      # start fp8 ASM serve (background)
#   ./setup_benchmark.sh serve-fp8-fused # fp8 ASM + fused_rms_norm_gated custom op
#   ./setup_benchmark.sh serve-fp8-ms64 # _serve_fp8_ms64.sh (fused on, 4096 batched-token cap)
#   ./setup_benchmark.sh serve-bf16     # start bf16 KV ASM serve (background)
#   ./setup_benchmark.sh sweep          # one IX-CI point (CONC_LIST must contain one value)
#   ./setup_benchmark.sh sweep-fused    # one point with TAG=fp8asm_fused
#   ./setup_benchmark.sh run-agentic    # cold container/server per c1,2,4,8,16,24
#   ./setup_benchmark.sh run-agentic-ms64 # same cold-server ladder with the ms64 recipe
#   ./setup_benchmark.sh compare        # table vs B300/B200 (needs /tmp/k3_b300.json)
#   ./setup_benchmark.sh verify-patches # audit all 4 vLLM ASM patches
#   ./perf_debug_agentic.sh             # patch audit + GEMM + sweep summary
#   ./setup_benchmark.sh status         # container + serve health
#
# From login node:
#   spur exec 34891 bash -lc 'export HOME=/home/xiaohugu; cd ~/work/InferenceX && ./setup_benchmark.sh start'

set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
cd "$HERE"

USER_HOME="${SPUR_USER_HOME:-/home/$(id -un)}"
export HOME="$USER_HOME"

IX="$USER_HOME/work/InferenceX"
AITER="${AITER_SRC:-$USER_HOME/work/aiter}"
IMAGE="${K3_IMAGE:-vllm/vllm-openai-rocm:kimi-k3}"
CTR="${K3_CTR:-k3-benchmark}"
MODEL_SRC="${MODEL_SRC:-/shared_nfs/models/Kimi-K3}"
SHM_MODEL="${SHM_MODEL:-/dev/shm/hf-cache/models--moonshotai--Kimi-K3/snapshots/9f62e4e9fffbd0a83ddd60e1c209d828994b3569}"
MODEL_PATH="${MODEL_PATH:-$SHM_MODEL}"
SWEEP_TAG="${SWEEP_TAG:-fp8asm}"
FUSED_RMS_NORM_GATED="${FUSED_RMS_NORM_GATED:-0}"

# Key the aiperf venv to the pinned submodule commit, so bumping the pin can't be
# silently benchmarked against a venv built for the previous one.
AIPERF_REV="${AIPERF_REV:-$(git -C "$IX" ls-tree HEAD utils/aiperf 2>/dev/null | awk '{print substr($3,1,8)}')}"
AIPERF_REV="${AIPERF_REV:-818c3a5a}"
# Keep the interpreter and packages container-local. A /workspace venv is shared
# across nodes but points at a container-local Python, which races and breaks in
# parallel cold-container runs.
AIPERF_VENV="/opt/.aiperf_${AIPERF_REV}"
SERVE_LOG="${SERVE_LOG:-}"

require_compute_node() {
  if [[ "$(hostname)" == crs-m2m-cpu-spur-* ]]; then
    echo "ERROR: run on compute node (~/spur-node attach benchmark)" >&2
    exit 1
  fi
}

docker_common() {
  docker run -d \
    --name "$CTR" \
    --ipc=host --network=host --shm-size=137438953472 \
    --device=/dev/kfd --device=/dev/dri \
    --group-add video --group-add render \
    --security-opt seccomp=unconfined --security-opt label=disable \
    --cap-add=SYS_PTRACE \
    -v "$IX:/workspace" \
    -v "$AITER:/aiter-latest" \
    -v "$USER_HOME:$USER_HOME" \
    -v /shared_nfs:/shared_nfs \
    -v /it-shared:/it-shared \
    -v /dev/shm:/dev/shm \
    -w /workspace \
    -e HF_HUB_CACHE=/dev/shm/hf-cache \
    -e HF_HOME=/dev/shm/hf-cache \
    -e GPU_ARCHS=gfx950 \
    --entrypoint sleep \
    "$IMAGE" infinity
}

do_start() {
  command -v docker >/dev/null || { echo "docker not found" >&2; exit 1; }
  [ -d "$AITER/aiter" ] || { echo "aiter missing at $AITER" >&2; exit 1; }

  if docker ps -a --format '{{.Names}}' | grep -qx "$CTR"; then
    echo "[benchmark] removing old container $CTR"
    docker rm -f "$CTR" >/dev/null
  fi

  if ! docker image inspect "$IMAGE" >/dev/null 2>&1; then
    echo "[benchmark] pulling $IMAGE ..."
    docker pull "$IMAGE"
  fi

  echo "[benchmark] starting container $CTR"
  docker_common
  docker ps --filter "name=$CTR" --format '{{.Names}} {{.Status}}'
  echo
  echo "Next:"
  echo "  ./setup_benchmark.sh setup"
  echo "  ./setup_benchmark.sh shell"
}

do_patch() {
  docker exec "$CTR" bash -lc '
set -euo pipefail
MLA=/usr/local/lib/python3.12/dist-packages/vllm/v1/attention/backends/mla/rocm_aiter_mla.py
UTILS=/usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/utils.py
verify() {
  local ok=1
  grep -q "PATCH(fp8-asm)" "$MLA" || { echo "[patch] MISSING: decode pad-to-16 (#50578)" >&2; ok=0; }
  grep -q "PATCH(fp8-prefill-pad)" "$MLA" || { echo "[patch] MISSING: fp8 prefill pad (PR-A)" >&2; ok=0; }
  grep -q "num_head_k = max(16, self.num_heads)" "$MLA" || { echo "[patch] MISSING: PS metadata16 (PR-A)" >&2; ok=0; }
  grep -q "PATCH(vLLM #50618)" "$UTILS" || { echo "[patch] MISSING: wvSplitK (#50618)" >&2; ok=0; }
  [ "$ok" = 1 ]
}
if verify; then
  echo "[patch] all 4 vLLM ASM patches present — skipping apply"
  exit 0
fi
cd /workspace
for p in _patch_fp8asm.py _patch_fp8_prefill.py _patch_ps_metadata16.py _patch_wvsplitk.py; do
  echo "[patch] $p ..."
  python3 "/workspace/$p"
done
verify && echo "[patch] all 4 vLLM ASM patches OK" || { echo "[patch] FAILED" >&2; exit 1; }
'
}

do_setup() {
  docker exec "$CTR" bash -lc '
set -euo pipefail
LOCAL_AITER=/opt/aiter-local
echo "[setup] staging node-local aiter from /aiter-latest..."
rm -rf "$LOCAL_AITER"
cp -a /aiter-latest "$LOCAL_AITER"
# The final tuner output lives on the shared /workspace mount. Install it into
# the node-local aiter copy explicitly; SPUR_USER_HOME may point /aiter-latest at
# a different checkout than /home/$USER/work/aiter on the login node.
FINAL_K3_GEMM=/workspace/k3_gemm_tune/kimik3_bf16_tuned_gemm.csv
LOCAL_K3_GEMM="$LOCAL_AITER/aiter/configs/model_configs/kimik3_bf16_tuned_gemm.csv"
[ -f "$FINAL_K3_GEMM" ] || { echo "ERROR: missing $FINAL_K3_GEMM" >&2; exit 1; }
cp "$FINAL_K3_GEMM" "$LOCAL_K3_GEMM"
cmp -s "$FINAL_K3_GEMM" "$LOCAL_K3_GEMM" \
  || { echo "ERROR: tuned GEMM CSV copy verification failed" >&2; exit 1; }
echo "[setup] installed final K3 tuned GEMM CSV into node-local aiter"
git config --global --add safe.directory "$LOCAL_AITER"
echo "[setup] aiter HEAD: $(git -C "$LOCAL_AITER" log --oneline -1)"
echo "[setup] editable-install aiter..."
cd "$LOCAL_AITER"
pip uninstall -y aiter amd-aiter >/dev/null 2>&1 || true
pip install -e . --no-build-isolation --no-deps

echo "[setup] merging global and per-model BF16 tuned GEMM CSVs..."
python3 - <<'"'"'PY'"'"'
import os
import shutil
from pathlib import Path

from aiter.jit.core import AITER_CONFIGS

configs = Path("/opt/aiter-local/aiter/configs")
sources = [configs / "bf16_tuned_gemm.csv"]
sources.extend(
    path
    for path in sorted((configs / "model_configs").glob("*bf16_tuned_gemm*.csv"))
    if "untuned" not in path.name
)
source_list = os.pathsep.join(str(path) for path in sources if path.is_file())
if not source_list:
    raise SystemExit("ERROR: no BF16 tuned GEMM CSVs found")

try:
    merged = AITER_CONFIGS.update_config_files(source_list, "bf16_tuned_gemm")
except RuntimeError as exc:
    # Current aiter intentionally raises once after resolving cross-file
    # duplicates in place. The second pass produces the clean merged table.
    if "Auto-resolved by keeping best performing" not in str(exc):
        raise
    merged = AITER_CONFIGS.update_config_files(source_list, "bf16_tuned_gemm")

destination = configs / "merged_bf16_tuned_gemm.csv"
shutil.copyfile(merged, destination)
print(f"[setup] merged BF16 GEMM CSV -> {destination}")
PY

AIPERF_REV="'"$AIPERF_REV"'"
AIPERF_VENV="'"$AIPERF_VENV"'"
echo "[setup] syncing aiperf submodule to pin ${AIPERF_REV}..."
cd /workspace
git config --global --add safe.directory /workspace
git config --global --add safe.directory /workspace/utils/aiperf
have=$(git -C /workspace/utils/aiperf rev-parse --short=8 HEAD 2>/dev/null || echo unknown)
if [ "$have" != "$AIPERF_REV" ]; then
  git submodule update --init --force utils/aiperf 2>&1 | tail -2
  have=$(git -C /workspace/utils/aiperf rev-parse --short=8 HEAD 2>/dev/null || echo unknown)
fi
echo "[setup] aiperf checked out at ${have} (pin ${AIPERF_REV})"
[ "$have" = "$AIPERF_REV" ] || { echo "[setup] aiperf pin mismatch: want ${AIPERF_REV}, have ${have}" >&2; exit 1; }

# The venv lives on the /workspace bind mount but its interpreter does not, so a
# container rebuild leaves the launcher in place with a dangling shebang. Probe by
# running it rather than testing for the file.
if ! "$AIPERF_VENV/bin/aiperf" --version >/dev/null 2>&1; then
  echo "[setup] building aiperf venv @ ${AIPERF_REV}..."
  rm -rf "$AIPERF_VENV"
  uv venv --python 3.11 "$AIPERF_VENV"
  uv pip install --python "$AIPERF_VENV/bin/python" \
    -r /workspace/utils/agentic-benchmark/requirements.txt -e /workspace/utils/aiperf \
    "datasets>=4.7.0" "huggingface_hub[cli]>=0.25.0" urllib3 requests
fi
"$AIPERF_VENV/bin/aiperf" --version

echo "[setup] applying vLLM ASM patches (required for fp8 KV + ROCM_AITER_MLA)..."
cd /workspace
MLA=/usr/local/lib/python3.12/dist-packages/vllm/v1/attention/backends/mla/rocm_aiter_mla.py
UTILS=/usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/utils.py
if grep -q "PATCH(fp8-asm)" "$MLA" && grep -q "PATCH(fp8-prefill-pad)" "$MLA" \
   && grep -q "num_head_k = max(16, self.num_heads)" "$MLA" \
   && grep -q "PATCH(vLLM #50618)" "$UTILS"; then
  echo "[setup] all 4 vLLM ASM patches already present"
else
  for p in _patch_fp8asm.py _patch_fp8_prefill.py _patch_ps_metadata16.py _patch_wvsplitk.py; do
    python3 "/workspace/$p"
  done
fi
grep -q "PATCH(fp8-asm)" "$MLA" && grep -q "PATCH(vLLM #50618)" "$UTILS" \
  || { echo "[setup] patch verify FAILED" >&2; exit 1; }
echo "[setup] ASM patches OK (decode #50578, prefill PR-A, ps_metadata16, wvSplitK #50618)"

MODEL_SRC="'"$MODEL_SRC"'"
SHM_MODEL="'"$SHM_MODEL"'"
MODEL_PATH="'"$MODEL_PATH"'"
mkdir -p "$(dirname "$SHM_MODEL")"
if [ -f "$SHM_MODEL/config.json" ]; then
  echo "[setup] model at $SHM_MODEL"
elif [ -f "$MODEL_SRC/config.json" ]; then
  echo "[setup] linking the NFS model into the HF cache (skip 1.5TB staging)"
  rm -rf "$SHM_MODEL"
  ln -s "$MODEL_SRC" "$SHM_MODEL"
else
  echo "[setup] staging Kimi-K3 weights to $SHM_MODEL ..."
  if command -v rsync >/dev/null 2>&1; then
    rsync -a --info=progress2 "$MODEL_SRC/" "$SHM_MODEL/"
  else
    cp -a "$MODEL_SRC/." "$SHM_MODEL/"
  fi
fi
# AIPerf CI loads the model tokenizer by repo ID. Populate the matching refs entry
# so HF_HUB_OFFLINE=1 resolves moonshotai/Kimi-K3 without network revalidation.
HF_REPO_CACHE="${SHM_MODEL%/snapshots/*}"
mkdir -p "$HF_REPO_CACHE/refs"
printf "%s\n" "${SHM_MODEL##*/}" > "$HF_REPO_CACHE/refs/main"
echo "[setup] DONE"
'
}

do_shell() {
  docker exec -it "$CTR" bash -lc 'cd /workspace && exec bash -l'
}

do_serve() {
  local mode="${1:-fp8}"
  local script fused=0 tag="$SWEEP_TAG"
  case "$mode" in
    fp8) script=_serve_fp8asm_ref.sh; SERVE_LOG="${SERVE_LOG:-serve_fp8asm_ref.log}" ;;
    fp8-fused) script=_serve_fp8asm_ref.sh; fused=1; tag=fp8asm_fused; SERVE_LOG="${SERVE_LOG:-serve_fp8asm_ref.log}" ;;
    fp8-ms64) script=_serve_fp8_ms64.sh; fused=1; tag=fp8asm_ms64; SERVE_LOG="${SERVE_LOG:-serve_fp8_ms64.log}" ;;
    bf16) script=_serve_bf16asm_ref.sh; SERVE_LOG="${SERVE_LOG:-serve_bf16asm_ref.log}" ;;
    *) echo "usage: serve-fp8|serve-fp8-fused|serve-fp8-ms64|serve-bf16" >&2; exit 1 ;;
  esac
  # At gpu-mem 0.95 even ~20GiB of leaked VRAM from a previous run makes engine
  # init fail, so refuse to start rather than spend a load cycle finding out.
  docker exec "$CTR" bash -lc '
    busy=$(rocm-smi --showmeminfo vram 2>/dev/null | awk "/Used/ && \$NF > 21474836480" | wc -l)
    if [ "$busy" -gt 0 ]; then
      echo "ERROR: $busy GPU(s) already hold >20GiB — stale processes from a previous run?" >&2
      rocm-smi --showmeminfo vram 2>/dev/null | awk "/Used/ {printf \"  GPU%d %.1f GiB\n\", NR-1, \$NF/1073741824}" >&2
      exit 1
    fi'
  do_patch
  docker exec "$CTR" bash -lc "
    cd /workspace
    export FUSED_RMS_NORM_GATED=$fused
    export MODEL_SRC='$MODEL_SRC'
    export SERVE_LOG='/workspace/$SERVE_LOG'
    export AITER_CONFIG_GEMM_BF16='/opt/aiter-local/aiter/configs/merged_bf16_tuned_gemm.csv'
    bash $script
  "
  echo "[benchmark] serve started ($mode, TAG=$tag). Log: /workspace/$SERVE_LOG"
  echo "  ./setup_benchmark.sh status"
}

do_sweep() {
  local tag="${1:-$SWEEP_TAG}"
  docker exec "$CTR" bash -lc "cd /workspace && HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TAG=$tag CONC_LIST=\"${CONC_LIST:-1}\" DURATION=\"${DURATION:-3600}\" AIPERF=$AIPERF_VENV/bin/aiperf OUT_ROOT=/workspace SWEEP_LOCK=\"/workspace/.k3_agentic_${tag}_c${CONC_LIST:-1}.lock\" bash _sweep_fp8asm_ixci.sh"
}

wait_for_serve() {
  echo "[benchmark] waiting for vllm health on :8888 (up to 3600s)..."
  docker exec "$CTR" bash -lc '
    for i in $(seq 1 360); do
      curl -sf http://localhost:8888/health >/dev/null && { echo "health OK after ${i}0s"; exit 0; }
      # A dead serve will never become healthy — do not burn the full hour on it.
      if ! pgrep -f "[v]llm serve" >/dev/null; then
        echo "serve process exited before becoming healthy" >&2
        break
      fi
      sleep 10
    done
    echo "health timeout — check serve log" >&2
    tail -30 "/workspace/'"$SERVE_LOG"'" 2>/dev/null || true
    exit 1
  '
}

do_run_agentic() {
  local mode="${1:-fp8-fused}" tag="${2:-fp8asm_fused}"
  local concs="${CONC_LIST:-1 2 4 8 16 24}"
  local c
  for c in $concs; do
    echo "========== IX-CI cold-server point c$c =========="
    # Single-node IX CI assigns each concurrency a fresh job/container. Recreate
    # ours as well so KV, JIT, cudagraph, and process state cannot carry over.
    do_start
    do_setup
    SERVE_LOG="serve_${tag}_c${c}.log"
    do_serve "$mode"
    wait_for_serve
    CONC_LIST="$c" do_sweep "$tag"
    docker rm -f "$CTR" >/dev/null
  done
  echo "[benchmark] canonical cold-server ladder complete. Compare:"
  echo "  ./setup_benchmark.sh compare"
}

do_compare() {
  local tag="${SWEEP_TAG:-fp8asm_fused}"
  python3 "$HERE/compare_agentic_sweep.py" --root "$HERE" --tag "$tag" --nv-json "${NV_JSON:-/tmp/k3_b300.json}"
}

do_verify_patches() {
  docker exec "$CTR" bash -lc '
MLA=/usr/local/lib/python3.12/dist-packages/vllm/v1/attention/backends/mla/rocm_aiter_mla.py
UTILS=/usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/utils.py
ok=1
for m in "PATCH(fp8-asm):decode #50578" "PATCH(fp8-prefill-pad):prefill PR-A" \
         "num_head_k = max(16:PS metadata16" "PATCH(vLLM #50618):wvSplitK"; do
  label="${m%%:*}"; name="${m#*:}"
  file="$MLA"; [[ "$label" == "PATCH(vLLM #50618)" ]] && file="$UTILS"
  grep -q "$label" "$file" && echo "  OK  $name" || { echo "  FAIL $name"; ok=0; }
done
exit $((1-ok))
'
}

do_status() {
  docker ps --filter "name=$CTR" --format 'table {{.Names}}\t{{.Status}}\t{{.Image}}' || true
  docker exec "$CTR" bash -lc '
    echo "--- serve ---"
    pgrep -af "vllm serve" || echo "(no vllm serve)"
    curl -s -m3 http://localhost:8888/health && echo " health OK" || echo " health FAIL"
    echo "--- aiperf ---"
    '"$AIPERF_VENV"'/bin/aiperf --version 2>/dev/null || echo "(aiperf not built — run setup)"
    echo "--- aiter ---"
    git -C /aiter-latest log --oneline -1
  ' 2>/dev/null || echo "container $CTR not running"
}

cmd="${1:-help}"

require_compute_node

case "$cmd" in
  patch) do_patch ;;
  start) do_start ;;
  setup) do_setup ;;
  shell) do_shell ;;
  serve-fp8) do_serve fp8 ;;
  serve-fp8-fused) do_serve fp8-fused ;;
  serve-fp8-ms64) do_serve fp8-ms64 ;;
  serve-bf16) do_serve bf16 ;;
  sweep) do_sweep "$SWEEP_TAG" ;;
  sweep-fused) do_sweep fp8asm_fused ;;
  run-agentic) do_run_agentic fp8-fused fp8asm_fused ;;
  run-agentic-ms64) do_run_agentic fp8-ms64 "${RUN_TAG:-fp8asm_ms64}" ;;
  compare) do_compare ;;
  verify-patches) do_verify_patches ;;
  status) do_status ;;
  stop) docker rm -f "$CTR" ;;
  help|-h|--help)
    sed -n '3,16p' "$0" | sed 's/^# \?//'
    ;;
  *)
    echo "Unknown: $cmd" >&2
    exit 1
    ;;
esac
