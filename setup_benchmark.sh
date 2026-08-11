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
#   ./setup_benchmark.sh verify-patches # audit all 5 vLLM ASM patches
#   ./perf_debug_agentic.sh             # patch audit + GEMM + sweep summary
#   ./setup_benchmark.sh status         # container + serve health
#
# DSpark + fp8-asm (nightly cb8104839 base — see docs/kimik3_dspark_fp8asm_recipe.md):
#   K3_CTR=k3-dspark-benchmark ./setup_benchmark.sh start-dspark
#   K3_CTR=k3-dspark-benchmark ./setup_benchmark.sh setup-dspark
#   K3_CTR=k3-dspark-benchmark ./setup_benchmark.sh verify-dspark-patches
#   K3_CTR=k3-dspark-benchmark ./setup_benchmark.sh serve-dspark
#   docker exec k3-dspark-benchmark bash -lc 'PORT=8890 bash _bench_k3_dspark_fp8asm.sh'
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
# DSpark fp8-asm validated aiter (incl. #4579 + #4575 int-32 K/V offset fixes).
AITER_PIN="${AITER_PIN:-55dbc4f475da26c23cdaf73ce6ed38342a2d7f83}"
IMAGE="${K3_IMAGE:-vllm/vllm-openai-rocm:kimi-k3}"
K3_DSPARK_IMAGE="${K3_DSPARK_IMAGE:-vllm/vllm-openai-rocm:nightly-cb8104839c141609d99f1254459ef3a4f1bd4263}"
CTR="${K3_CTR:-k3-benchmark}"
DRAFT_MODEL="${DRAFT_MODEL:-Inferact/Kimi-K3-DSpark}"
DRAFT_SHM="${DRAFT_SHM:-/dev/shm/hf-cache/models--Inferact--Kimi-K3-DSpark/snapshots}"
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
  local img="${START_IMAGE:-$IMAGE}"
  command -v docker >/dev/null || { echo "docker not found" >&2; exit 1; }
  [ -d "$AITER/aiter" ] || { echo "aiter missing at $AITER" >&2; exit 1; }

  if docker ps -a --format '{{.Names}}' | grep -qx "$CTR"; then
    echo "[benchmark] draining host GPU procs before removing $CTR"
    if [ -x "$IX/_drain_vllm_host.sh" ]; then
      bash "$IX/_drain_vllm_host.sh" || true
    fi
    echo "[benchmark] removing old container $CTR"
    docker rm -f "$CTR" >/dev/null
  fi

  if ! docker image inspect "$img" >/dev/null 2>&1; then
    echo "[benchmark] pulling $img ..."
    docker pull "$img"
  fi

  echo "[benchmark] starting container $CTR (image=$img)"
  IMAGE="$img" docker_common
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
  grep -q "PATCH(skip-k3-fp8-ps)" "$MLA" || { echo "[patch] MISSING: skip K3 fp8 PS workspace" >&2; ok=0; }
  grep -q "PATCH(vLLM #50618)" "$UTILS" || { echo "[patch] MISSING: wvSplitK (#50618)" >&2; ok=0; }
  [ "$ok" = 1 ]
}
if verify; then
  echo "[patch] all 5 vLLM ASM patches present — skipping apply"
  exit 0
fi
cd /workspace
for p in _patch_fp8asm.py _patch_fp8_prefill.py _patch_ps_metadata16.py _patch_skip_k3_fp8_ps.py _patch_wvsplitk.py; do
  echo "[patch] $p ..."
  python3 "/workspace/$p"
done
verify && echo "[patch] all 5 vLLM ASM patches OK" || { echo "[patch] FAILED" >&2; exit 1; }
'
}

do_setup() {
  docker exec \
    -e K3_GEMM_CSV="${K3_GEMM_CSV:-/workspace/k3_gemm_tune/kimik3_bf16_tuned_gemm.csv}" \
    -e AITER_PIN="${AITER_PIN}" \
    "$CTR" bash -lc '
set -euo pipefail
LOCAL_AITER=/opt/aiter-local
echo "[setup] staging node-local aiter from /aiter-latest..."
rm -rf "$LOCAL_AITER"
cp -a /aiter-latest "$LOCAL_AITER"
# aiter JIT-compiles cpp_itfs template ops (e.g. top_k_top_p sampling, hit during
# vLLM memory profiling) against CK headers under 3rdparty. An uninitialized or
# partially copied submodule only surfaces mid-startup as a FileNotFoundError on
# the include dir, so repair it here from the bind mount.
if [ ! -d "$LOCAL_AITER/3rdparty/composable_kernel/include" ]; then
  echo "[setup] composable_kernel missing in staged aiter; re-copying from /aiter-latest..."
  if [ -d /aiter-latest/3rdparty/composable_kernel/include ]; then
    mkdir -p "$LOCAL_AITER/3rdparty/composable_kernel"
    cp -a /aiter-latest/3rdparty/composable_kernel/. "$LOCAL_AITER/3rdparty/composable_kernel/"
  else
    echo "[setup] ERROR: /aiter-latest/3rdparty/composable_kernel is empty; run" >&2
    echo "[setup]        git -C \$AITER submodule update --init 3rdparty/composable_kernel" >&2
    exit 1
  fi
fi
# Do not inherit process-owned JIT batons from the shared checkout. Stale
# top-level and nested build locks block rank 0 until the other TP ranks time
# out in RCCL.
python3 - <<'"'"'PY'"'"'
from pathlib import Path

build = Path("/opt/aiter-local/aiter/jit/build")
for lock in build.rglob("lock*"):
    if lock.is_file() or lock.is_symlink():
        lock.unlink()
PY
# The final tuner output lives on the shared /workspace mount. Install it into
# the node-local aiter copy explicitly; SPUR_USER_HOME may point /aiter-latest at
# a different checkout than /home/$USER/work/aiter on the login node.
FINAL_K3_GEMM="$K3_GEMM_CSV"
LOCAL_K3_GEMM="$LOCAL_AITER/aiter/configs/model_configs/kimik3_bf16_tuned_gemm.csv"
[ -f "$FINAL_K3_GEMM" ] || { echo "ERROR: missing $FINAL_K3_GEMM" >&2; exit 1; }
cp "$FINAL_K3_GEMM" "$LOCAL_K3_GEMM"
cmp -s "$FINAL_K3_GEMM" "$LOCAL_K3_GEMM" \
  || { echo "ERROR: tuned GEMM CSV copy verification failed" >&2; exit 1; }
echo "[setup] installed final K3 tuned GEMM CSV into node-local aiter"
git config --global --add safe.directory "$LOCAL_AITER"
echo "[setup] pinning aiter to ${AITER_PIN} (DSpark fp8-asm: #4579 + #4575 K/V offset)..."
git -C "$LOCAL_AITER" fetch --tags origin 2>/dev/null || true
git -C "$LOCAL_AITER" reset --hard "$AITER_PIN"
git -C "$LOCAL_AITER" merge-base --is-ancestor d3ddaabf9 HEAD \
  || { echo "ERROR: aiter missing #4579 (d3ddaabf9) after checkout $AITER_PIN" >&2; exit 1; }
git -C "$LOCAL_AITER" merge-base --is-ancestor 22beb1caa HEAD \
  || { echo "ERROR: aiter missing #4575 (22beb1caa) after checkout $AITER_PIN" >&2; exit 1; }
echo "[setup] wiping stale aiter JIT (must rebuild after #4579)..."
rm -rf "$LOCAL_AITER/aiter/jit/build"
find "$LOCAL_AITER/aiter/jit" -maxdepth 1 -name "module_*.so" -delete 2>/dev/null || true
echo "[setup] aiter HEAD: $(git -C "$LOCAL_AITER" log --oneline -1)"
echo "[setup] editable-install aiter..."
cd "$LOCAL_AITER"
pip uninstall -y aiter amd-aiter >/dev/null 2>&1 || true
pip install -e . --no-build-isolation --no-deps
rm -rf /root/aiter
ln -s "$LOCAL_AITER" /root/aiter
python3 -c "import aiter; assert \"/root/aiter\" in aiter.__file__ or \"/opt/aiter-local\" in aiter.__file__, aiter.__file__; print(\"aiter:\", aiter.__file__)"

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
   && grep -q "PATCH(skip-k3-fp8-ps)" "$MLA" \
   && grep -q "PATCH(vLLM #50618)" "$UTILS"; then
  echo "[setup] all 5 vLLM ASM patches already present"
else
  for p in _patch_fp8asm.py _patch_fp8_prefill.py _patch_ps_metadata16.py _patch_skip_k3_fp8_ps.py _patch_wvsplitk.py; do
    python3 "/workspace/$p"
  done
fi
grep -q "PATCH(fp8-asm)" "$MLA" && grep -q "PATCH(vLLM #50618)" "$UTILS" \
  || { echo "[setup] patch verify FAILED" >&2; exit 1; }
echo "[setup] ASM patches OK (decode #50578, prefill PR-A, ps_metadata16, skip-k3-fp8-ps, wvSplitK #50618)"

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
  docker exec "$CTR" bash -lc "cd /workspace && TAG=$tag CONC_LIST=\"${CONC_LIST:-1}\" DURATION=\"${DURATION:-3600}\" AIPERF=$AIPERF_VENV/bin/aiperf OUT_ROOT=/workspace SWEEP_LOCK=\"/workspace/.k3_agentic_${tag}_c${CONC_LIST:-1}.lock\" bash _sweep_fp8asm_ixci.sh"
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
         "num_head_k = max(16:PS metadata16" "PATCH(skip-k3-fp8-ps):skip K3 fp8 PS" \
         "PATCH(vLLM #50618):wvSplitK"; do
  label="${m%%:*}"; name="${m#*:}"
  file="$MLA"; [[ "$label" == "PATCH(vLLM #50618)" ]] && file="$UTILS"
  grep -q "$label" "$file" && echo "  OK  $name" || { echo "  FAIL $name"; ok=0; }
done
exit $((1-ok))
'
}

do_start_dspark() {
  START_IMAGE="$K3_DSPARK_IMAGE" do_start
}

do_setup_dspark() {
  do_setup
  docker exec "$CTR" bash -lc '
set -euo pipefail
echo "[setup-dspark] triton 3.7.0 (nightly ships 3.6.0)..."
pip install -q --extra-index-url https://pypi.amd.com/triton/release/rocm-7.2.0/simple/ \
  triton==3.7.0 tabulate

DRAFT_MODEL="'"$DRAFT_MODEL"'"
DRAFT_SHM="'"$DRAFT_SHM"'"
mkdir -p "$(dirname "$DRAFT_SHM")"
draft_has_weights() {
  local snap f
  for snap in "$DRAFT_SHM"/*/; do
    [ -f "${snap}config.json" ] || continue
    compgen -G "${snap}*.safetensors" >/dev/null 2>&1 && return 0
    [ -f "${snap}model.safetensors.index.json" ] && return 0
    compgen -G "${snap}*.bin" >/dev/null 2>&1 && return 0
  done
  return 1
}
if draft_has_weights; then
  echo "[setup-dspark] draft weights already staged under $DRAFT_SHM"
else
  echo "[setup-dspark] staging draft $DRAFT_MODEL -> HF cache..."
  HF=/opt/.aiperf_'"$AIPERF_REV"'/bin/hf
  if [ ! -x "$HF" ]; then HF=hf; fi
  if ! command -v "$HF" >/dev/null 2>&1 && [ ! -x "$HF" ]; then
    echo "[setup-dspark] ERROR: hf CLI not found (need huggingface-hub >= 1.0)" >&2
    exit 1
  fi
  "$HF" download "$DRAFT_MODEL" --cache-dir /dev/shm/hf-cache
  draft_has_weights || { echo "[setup-dspark] ERROR: draft download finished but no weight files under $DRAFT_SHM" >&2; exit 1; }
fi

echo "[setup-dspark] applying DSpark + fp8-asm enablement patches..."
export RESEED_SHIPPED_MLA='"'"'${RESEED_SHIPPED_MLA:-0}'"'"'
bash /workspace/_k3_dspark_fp8asm_apply_patches.sh
'
}

do_verify_dspark_patches() {
  do_verify_patches
  docker exec "$CTR" bash -lc '
MLA=/usr/local/lib/python3.12/dist-packages/vllm/v1/attention/backends/mla/rocm_aiter_mla.py
AITER=/opt/aiter-local/aiter/mla.py
[ -f "$AITER" ] || AITER=/root/aiter/aiter/mla.py
KDA=/usr/local/lib/python3.12/dist-packages/vllm/models/kimi_k3/amd/ops/third_party/kda/fused_recurrent.py
ok=1
check() { grep -q "$2" "$1" && echo "  OK  $3" || { echo "  FAIL $3"; ok=0; }; }
check "$MLA" "_mtp_decode_qlen" "recipe _mtp_decode_qlen present"
check "$MLA" "method == \"dspark\"" "dspark verify qlen branch"
check "$MLA" "uses_asm_decode" "persistent-metadata gate"
check "$AITER" "80: 64" "aiter get_block_n_fp8 key 80"
check "$AITER" "get_block_n_fp8.get(" "aiter get_block_n_fp8.get()"
check "$KDA" "stride_indices_seq" "KDA PR#27 stride fix"
DCACHE=/dev/shm/hf-cache/models--Inferact--Kimi-K3-DSpark/snapshots
CFG="$(ls -d "$DCACHE"/*/config.json 2>/dev/null | head -1)"
if [ -n "$CFG" ]; then
  python3 - "$CFG" <<'"'"'PY'"'"'
import json, sys
c = json.load(open(sys.argv[1]))
if c.get("dflash_config", {}).get("causal") is True:
    print("  OK  draft dflash_config.causal=true")
else:
    print("  FAIL draft dflash_config.causal"); raise SystemExit(1)
PY
else
  echo "  FAIL draft config.json not staged"; ok=0
fi
exit $((1-ok))
'
}

do_serve_dspark() {
  local port="${PORT:-8890}" num_spec="${NUM_SPEC:-2}"
  do_patch
  docker exec "$CTR" bash -lc "
    cd /workspace
    export VLLM_ROCM_AITER_MLA_ASM_PADDING=asm
    export PORT='$port' NUM_SPEC='$num_spec'
    export GPU_MEM='${GPU_MEM:-0.88}' MAX_NUM_SEQS='${MAX_NUM_SEQS:-16}'
    export RESEED_SHIPPED_MLA='${RESEED_SHIPPED_MLA:-0}'
    bash _k3_dspark_fp8asm_apply_patches.sh
    bash _serve_k3_bench_spec.sh
  "
  echo "[benchmark] DSpark serve started (port=$port, NUM_SPEC=$num_spec). Log: /workspace/serve_k3_bench_spec${num_spec}.log"
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
  start-dspark) do_start_dspark ;;
  setup-dspark) do_setup_dspark ;;
  verify-dspark-patches) do_verify_dspark_patches ;;
  serve-dspark) do_serve_dspark ;;
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
