#!/usr/bin/env bash
# AgentX-COMPLIANT synthetic-acceptance variant of _run_agentic_dspark.sh.
#
# WHY THIS EXISTS: both the official MI355X recipe (PR #2508) and the B300
# baseline (kimik3_fp4_b300_vllm_mtp.sh) publish agentic numbers with
# rejection_sample_method=synthetic pinned to the committed golden AL, per the
# AgentX policy (golden_al_distribution/README.md: "may not substitute a
# different acceptance target"). Golden AL for K=2 thinking_on = 2.51
# (kimik3_dspark_probabilistic_sample_method_block_rejection_sample_method.yaml).
# Our earlier sweep used REAL block-verify — honest, but NOT comparable to the
# published synthetic-vs-synthetic numbers (and on the agentic corpus real DSpark
# AL collapses to ~1.16-2.01, per the B300 recipe's own admission). This driver
# runs the same cold-serve ladder with synthetic 2.51 so the Pareto is apples-to-
# apples with the merged recipes.
#
#   CONC_LIST="1 2 4 8 16 24" DURATION=3600 bash _run_agentic_dspark_synth.sh
#
# Everything else (container, mandated config, cold serve per conc) is identical
# to _run_agentic_dspark.sh. Results: /workspace/k3_${TAG}_ixci_c{conc}/
set -euo pipefail

CONTAINER="${CONTAINER:-k3-dspark-benchmark}"
CONC_LIST="${CONC_LIST:-1 2 4 8 16 24}"
PORT="${PORT:-8891}"
DURATION="${DURATION:-3600}"
TAG="${TAG:-dspark_synth251_ixci}"
NUM_SPEC="${NUM_SPEC:-2}"
# AgentX golden AL for this (model, thinking_on, K) — do NOT substitute another
# value; it is the prescribed acceptance target for the comparison.
SYNTHETIC_ACCEPT_LEN="${SYNTHETIC_ACCEPT_LEN:-2.51}"
# Mandated DSpark config (validated with the fixes): gpu_mem 0.95, seqs 64,
# MNBT 16384, KV pin 34 GiB, FULL_AND_PIECEWISE. Do NOT change.
GPU_MEM="${GPU_MEM:-0.95}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-64}"
MNBT="${MNBT:-16384}"
AIPERF="${AIPERF:-/workspace/.aiperf_v1_0_1/bin/aiperf}"

log() { echo "[$(date +%T)] $*"; }

_vram_max_gib() {
  docker exec "$CONTAINER" bash -lc "rocm-smi --showmeminfo vram 2>/dev/null | grep -i 'Used Memory' | awk '{g=\$NF/1073741824; if (g>m) m=g} END{printf \"%.0f\", m}'" 2>/dev/null || echo 999
}
_vllm_procs() {
  docker exec "$CONTAINER" bash -lc 'me=$$; pgrep -f "vllm|EngineCore|VllmWorker|multiprocessing.spawn" 2>/dev/null | grep -vx "$me" | wc -l' 2>/dev/null || echo 0
}
kill_serve() {
  # GRACEFUL drain first (SIGTERM -> wait -> SIGKILL fallback). A hard SIGKILL
  # skips vLLM's shutdown so GPU buffers can be orphaned and the ROCm driver won't
  # reclaim them (~30 GiB/GPU) -> only a reboot clears it. SIGTERM lets it free
  # them. Self-exclude the killing shell via $$. Check MAX VRAM across all 8 GPUs.
  docker exec "$CONTAINER" bash -lc '
    me=$$
    for p in $(pgrep -f "vllm|EngineCore|VllmWorker|multiprocessing.spawn"); do
      [ "$p" = "$me" ] && continue; kill -TERM "$p" 2>/dev/null
    done; exit 0' 2>/dev/null || true
  local used
  for _ in $(seq 1 18); do          # up to ~90s for graceful exit + VRAM drain
    sleep 5
    used=$(_vram_max_gib)
    if [ "$(_vllm_procs)" = "0" ] && [ "${used:-999}" -le 20 ] 2>/dev/null; then
      log "graceful teardown clean (procs=0, max VRAM ${used} GiB)"; return 0
    fi
  done
  log "graceful teardown incomplete (VRAM=$(_vram_max_gib) GiB) -> SIGKILL fallback"
  docker exec "$CONTAINER" bash -lc '
    me=$$
    for p in $(pgrep -f "vllm|EngineCore|VllmWorker|multiprocessing.spawn"); do
      [ "$p" = "$me" ] && continue; kill -9 "$p" 2>/dev/null
    done; exit 0' 2>/dev/null || true
  for _ in $(seq 1 12); do
    sleep 5; used=$(_vram_max_gib)
    [ "${used:-999}" -le 20 ] 2>/dev/null && { log "VRAM drained after SIGKILL (${used} GiB)"; return 0; }
  done
  log "WARN: VRAM did not fully drain"
}

log "SYNTHETIC-ACCEPTANCE sweep: golden AL=$SYNTHETIC_ACCEPT_LEN (K=$NUM_SPEC), TAG=$TAG"
for c in $CONC_LIST; do
  log "########## conc=$c : cold DSpark serve (synthetic AL=$SYNTHETIC_ACCEPT_LEN) ##########"
  kill_serve
  log "starting DSpark serve (ns=$NUM_SPEC, seqs=$MAX_NUM_SEQS, port=$PORT)"
  if ! docker exec \
      -e NUM_SPEC="$NUM_SPEC" -e PORT="$PORT" -e GPU_MEM="$GPU_MEM" \
      -e MAX_NUM_SEQS="$MAX_NUM_SEQS" -e MNBT="$MNBT" \
      -e SYNTHETIC_ACCEPT_LEN="$SYNTHETIC_ACCEPT_LEN" \
      "$CONTAINER" bash /workspace/_serve_k3_bench_spec.sh; then
    log "ERROR: serve failed to come up for conc=$c — aborting"
    docker exec "$CONTAINER" bash -lc "tail -40 /workspace/serve_k3_bench_spec${NUM_SPEC}.log" || true
    exit 1
  fi
  log "serve ready; running agentic point conc=$c (DURATION=$DURATION)"
  docker exec \
      -e MODEL=Kimi-K3 -e TAG="$TAG" -e CONC_LIST="$c" -e PORT="$PORT" \
      -e DURATION="$DURATION" -e AIPERF="$AIPERF" -e OUT_ROOT=/workspace \
      -e UNSAFE_OVERRIDE="${UNSAFE_OVERRIDE:-0}" \
      -e SWEEP_LOCK="/workspace/.k3_agentic_${TAG}_c${c}.lock" \
      "$CONTAINER" bash /workspace/_sweep_fp8asm_ixci.sh || {
        log "WARN: agentic point conc=$c returned non-zero (see k3_${TAG}_ixci_c${c}.log)"; }
  log "conc=$c complete"
done

kill_serve
log "########## synthetic DSpark sweep COMPLETE — TAG=$TAG ##########"
log "Artifacts: k3_${TAG}_ixci_c{${CONC_LIST// /,}}/"
