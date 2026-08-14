#!/bin/bash
# Kimi-K3 agentic point — OUR ASM serve config (ROCM_AITER_MLA, kv=fp8, gpu-mem
# 0.95, ms64, uncapped) driven by the IX-CI agentic harness, matching
# benchmarks/benchmark_lib.sh:build_replay_cmd.
#
# Harness knobs use the SAME env names + defaults as build_replay_cmd:
#   --warmup-requests-per-lane    AIPERF_WARMUP_REQUESTS_PER_LANE    (default 10)
#   --trace-idle-gap-cap-seconds  AIPERF_TRACE_IDLE_GAP_CAP_SECONDS  (default 300)
#   --warmup-grace-period         AGENTIC_WARMUP_GRACE_PERIOD        (default 1800)
#   --failed-request-threshold    AIPERF_FAILED_REQUEST_THRESHOLD    (default 0.10)
#   --benchmark-duration          DURATION                           (default 3600)
# trace-idle-gap-cap is REQUIRED: without it the cc-traces trajectories replay
# their full real-world idle gaps, so warmup never drains at high concurrency.
#
# NO shell `timeout`: aiperf self-bounds each run — warmup ends at the grace
# period or when it drains, profiling runs benchmark-duration, and
# failed-request-threshold aborts early. A tight shell timeout would kill a
# healthy run (warmup-grace 1800 + benchmark-duration 3600 already = 5400s before
# warmup-send + export). Hang protection, if needed, belongs at the job level
# with a much larger bound + diagnostic cleanup, as in IX-CI.
#
# Runs exactly one concurrency against an already-alive serve on :PORT. The
# setup_benchmark.sh run-agentic-ms64 driver recreates the container and server
# before every point, matching IX single-node CI isolation. Multi-concurrency
# invocations are rejected so JIT/graph/KV state cannot leak between points.
set -euo pipefail
# REQUIRES aiperf pinned to the IX submodule commit (utils/aiperf @ 818c3a5a or
# newer), which allows --trace-idle-gap-cap-seconds for the inferencex-agentx-mvp
# scenario. The older be758d62 build sets forbid_trace_idle_gap_cap=True and
# rejects the flag. `setup_benchmark.sh setup` builds the venv keyed to the
# pinned rev (/opt/.aiperf_<rev>) and passes it in as AIPERF.
# aiperf: honor $AIPERF if it points to a real binary, else auto-detect across
# machines (this box: /workspace/.aiperf_*; others: /opt/.aiperf_*; or on PATH).
if [ -n "${AIPERF:-}" ] && [ -x "$AIPERF" ]; then
  :
elif command -v aiperf >/dev/null 2>&1; then
  AIPERF="$(command -v aiperf)"
else
  AIPERF=""
  # prefer the v1.0.1 pin (b7b16cf8 == /workspace/.aiperf_v1_0_1) for IX-CI parity,
  # then any aiperf. Avoids grabbing an older/incompatible build (e.g. be758d).
  for c in /opt/.aiperf_b7b16cf8/bin/aiperf /opt/.aiperf_*v1_0_1*/bin/aiperf \
           /workspace/.aiperf_v1_0_1/bin/aiperf /workspace/.aiperf_*b7b16cf8*/bin/aiperf \
           /opt/.aiperf_*/bin/aiperf /workspace/.aiperf_*/bin/aiperf "$HOME"/.aiperf_*/bin/aiperf; do
    [ -x "$c" ] && { AIPERF="$c"; break; }
  done
  [ -n "$AIPERF" ] || { echo "!! no aiperf found — set AIPERF=/path/to/bin/aiperf (looked: \$AIPERF, PATH, /opt/.aiperf_*, /workspace/.aiperf_*, \$HOME/.aiperf_*)"; exit 1; }
fi
echo "using aiperf: $AIPERF"
MODEL="${MODEL:-moonshotai/Kimi-K3}"
# tokenizer: prefer a LOCAL model dir (offline-safe + exact Kimi tokenizer) over
# the hub id, which aiperf would fetch at runtime and fail on an offline box.
# A local path load doesn't touch the hub. Override via TOKENIZER; falls back to
# the hub id only if no local tokenizer is found.
TOKENIZER="${TOKENIZER:-${MODEL_PATH:-}}"
if [ ! -f "${TOKENIZER:-}/tokenizer_config.json" ]; then
  TOKENIZER="$MODEL"
  for d in /dev/shm/hf-cache/models--moonshotai--Kimi-K3/snapshots/* "${MODEL_SRC:-/shared_nfs/models/Kimi-K3}"; do
    [ -f "$d/tokenizer_config.json" ] && { TOKENIZER="$d"; break; }
  done
fi
echo "using tokenizer: $TOKENIZER"
PORT="${PORT:-8888}"
DURATION="${DURATION:-3600}"
WARMUP_REQS="${AIPERF_WARMUP_REQUESTS_PER_LANE:-10}"
IDLE_GAP_CAP="${AIPERF_TRACE_IDLE_GAP_CAP_SECONDS:-300}"
GRACE="${AGENTIC_WARMUP_GRACE_PERIOD:-1800}"
FAIL_THRESH="${AIPERF_FAILED_REQUEST_THRESHOLD:-0.10}"
TAG="${TAG:-fp8asm}"
CONC_LIST="${CONC_LIST:-1}"
OUT_ROOT="${OUT_ROOT:-/workspace}"
SWEEP_LOCK="${SWEEP_LOCK:-$OUT_ROOT/.k3_agentic_sweep.lock}"

prepare_out_dir() {
  local out="$1"
  if [ -d "$out" ]; then
    # NFS may leave .nfs* stub files after rm -rf; rename aside instead of fighting rm.
    local trash="${out}.trash.$$"
    if mv "$out" "$trash" 2>/dev/null; then
      rm -rf "$trash" 2>/dev/null || true &
    else
      rm -rf "$out" 2>/dev/null || true
    fi
  fi
  mkdir -p "$out/aiperf_artifacts"
}

run_conc() {
  local c="$1" seed=42
  # conc-1's single lane is unspawnable at seed 42 (EmptyTracePoolError); the
  # dataset sampler yields a valid conc-1 root at seed 0. Overridable via SEED.
  [ "$c" = "1" ] && seed=0
  seed="${SEED:-$seed}"
  local out="$OUT_ROOT/k3_${TAG}_ixci_c$c"
  prepare_out_dir "$out"
  # UNSAFE_OVERRIDE=1 converts the scenario invariants (e.g. duration>=900) to
  # warnings so a short DURATION works for a plumbing/validation run. Such a run
  # is marked submission_valid=false — do NOT use it for reported numbers.
  local OVERRIDE_ARG=(); [ "${UNSAFE_OVERRIDE:-0}" = "1" ] && OVERRIDE_ARG=(--unsafe-override)
  echo "=== ${TAG}-IXCI conc=$c seed=$seed start $(date +%T) ==="
  "$AIPERF" profile --scenario inferencex-agentx-mvp --url "http://localhost:$PORT" \
    --endpoint /v1/chat/completions --endpoint-type chat --streaming --model "$MODEL" \
    "${OVERRIDE_ARG[@]}" \
    --concurrency "$c" --benchmark-duration "$DURATION" --stats-interval 30 --random-seed "$seed" \
    --failed-request-threshold "$FAIL_THRESH" \
    --trajectory-start-min-ratio 0.25 --trajectory-start-max-ratio 0.75 \
    --warmup-requests-per-lane "$WARMUP_REQS" --trace-idle-gap-cap-seconds "$IDLE_GAP_CAP" \
    --warmup-grace-period "$GRACE" \
    --use-server-token-count --tokenizer "$TOKENIZER" --tokenizer-trust-remote-code --no-gpu-telemetry \
    --num-dataset-entries 393 --slice-duration 1.0 \
    --output-artifact-dir "$out/aiperf_artifacts" --public-dataset semianalysis_cc_traces_weka_062126 \
    > "$OUT_ROOT/k3_${TAG}_ixci_c$c.log" 2>&1
  echo "=== ${TAG}-IXCI conc=$c done $(date +%T) ==="
}

read -r -a CONCS <<< "$CONC_LIST"
if [ "${#CONCS[@]}" -ne 1 ]; then
  echo "ERROR: IX single-node CI requires one fresh server per concurrency; got CONC_LIST='$CONC_LIST'" >&2
  echo "Use setup_benchmark.sh run-agentic-ms64 to run the full cold-server ladder." >&2
  exit 2
fi

export AIPERF_DATASET_CONFIGURATION_TIMEOUT="${AIPERF_DATASET_CONFIGURATION_TIMEOUT:-1800}"
export AIPERF_SERVICE_PROFILE_CONFIGURE_TIMEOUT="${AIPERF_SERVICE_PROFILE_CONFIGURE_TIMEOUT:-1800}"
export AIPERF_UI_REALTIME_METRICS_ENABLED="${AIPERF_UI_REALTIME_METRICS_ENABLED:-true}"
export AIPERF_DATASET_WEKA_LIVE_ASSISTANT_RESPONSES="${AIPERF_DATASET_WEKA_LIVE_ASSISTANT_RESPONSES:-0}"
export AIPERF_HTTP_TCP_USER_TIMEOUT="${AIPERF_HTTP_TCP_USER_TIMEOUT:-900000}"

echo "########## ${TAG}-IXCI point start $(date +%T) ##########"
exec 9>"$SWEEP_LOCK"
if ! flock -n 9; then
  echo "ERROR: another sweep holds $SWEEP_LOCK — refusing to overlap conc runs" >&2
  exit 1
fi
run_conc "${CONCS[0]}"
echo "########## ${TAG}-IXCI point COMPLETE $(date +%T) ##########"
