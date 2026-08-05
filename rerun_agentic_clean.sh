#!/usr/bin/env bash
# Full clean agentic rerun after container restart. Run on compute node in background:
#   nohup ./rerun_agentic_clean.sh >> agentic_rerun_latest.log 2>&1 &
set -euo pipefail
cd "$(dirname "$0")"
export HOME="${HOME:-/home/$(id -un)}"

TAG="${TAG:-fp8asm_ms64_ixci_cold}"
CONC_LIST="${CONC_LIST:-1 2 4 8 16 24}"
STAMP="$(date +%Y%m%d_%H%M%S)"
RUN_ID="${SLURM_JOB_ID:-local}_c${CONC_LIST// /-}"
LOG="$PWD/agentic_rerun_${STAMP}_${RUN_ID}.log"
ln -sf "$(basename "$LOG")" "agentic_rerun_${RUN_ID}_latest.log"

log() { echo "$@" | tee -a "$LOG"; }

log "========== canonical cold-server rerun $(date) TAG=$TAG CONC_LIST=$CONC_LIST =========="

log "Starting one fresh container/server per concurrency"
CONC_LIST="$CONC_LIST" RUN_TAG="$TAG" ./setup_benchmark.sh run-agentic-ms64 \
  2>&1 | tee -a "$LOG"

log "DONE $(date)"
log "Artifacts: k3_${TAG}_ixci_c{1,2,4,8,16,24}/"
