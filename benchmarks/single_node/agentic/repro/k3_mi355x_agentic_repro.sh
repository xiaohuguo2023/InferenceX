#!/usr/bin/env bash
# Reproduce the Kimi-K3 FP4 MI355X agentic conc-1/2/4 numbers on a second box.
#
#   CONC=1 bash k3_mi355x_agentic_repro.sh
#
# Everything below is the exact environment the measured runs used. The recipe
# itself (../kimik3_fp4_mi355x_mtp.sh) supplies the serve flags; this file only
# supplies the environment the CI workflow would otherwise set, plus the four
# things a manual launch gets wrong by default (see NOTES).
#
# ---------------------------------------------------------------------------
# MEASURED RESULTS (3600 s, synthetic acceptance AL 3.84, nspec 7, TP8, EP_SIZE=1)
#
#   CONC  intvty p90   frITL p90   n     ISL med   OSL med   run
#   1     131.35       7.613       203   161287    654.0     r72_c1_ep1     <- image-correct
#   2     102.13       9.792       342    79067    203.0     m1_c2          <- PRE-correction
#   4      93.82      10.659       455    88812    299.0     m1_c4          <- PRE-correction
#
# Only CONC=1 was measured on the correct ROCm 7.2.3 image. The conc-2 and
# conc-4 numbers predate that fix (which was worth ~9.5% at conc-1), so treat
# them as a floor, not a target. Re-running them here is the point of this file.
# ---------------------------------------------------------------------------
set -euo pipefail

# ---- box-specific: override these two on the new machine ------------------
HF_ROOT="${HF_ROOT:-/dev/shm/hf-cache}"
MODEL_SNAPSHOT="${MODEL_SNAPSHOT:-}"          # auto-resolved below if empty
REPO_ROOT="${REPO_ROOT:-/workspace}"

# ---- the arm ---------------------------------------------------------------
CONC="${CONC:-1}"
TAG="${TAG:-repro_c${CONC}}"

if [ -z "$MODEL_SNAPSHOT" ]; then
    MODEL_SNAPSHOT=$(ls -d "$HF_ROOT"/models--moonshotai--Kimi-K3/snapshots/*/ 2>/dev/null | head -1)
fi
[ -n "$MODEL_SNAPSHOT" ] || { echo "ERROR: Kimi-K3 snapshot not found under $HF_ROOT" >&2; exit 1; }

export MODEL=moonshotai/Kimi-K3
export MODEL_PATH="${MODEL_SNAPSHOT%/}"
export TP=8 CONC DURATION=3600

# EP_SIZE=1 is NOT optional. The recipe is pure TP8; --enable-expert-parallel
# costs 6.1% and makes the run non-comparable to every number above.
export EP_SIZE=1

# NOTES -- the four things a manual (non-workflow) launch gets wrong:
#  1. MODEL_PREFIX unset silently selects the *_256k corpus, a DIFFERENT
#     workload. Every number above is on the kimik3 corpus.
export MODEL_PREFIX=kimik3
#  2. EVAL_ONLY is consumed under `set -u`; unset aborts the run.
#     false = the agentic replay (throughput). true = run_eval / GSM8K gate.
export EVAL_ONLY=false
#  3. RESULT_FILENAME/RESULT_DIR are required; unset aborts late, after the
#     serve has already loaded 1.5 TB of weights.
export RESULT_FILENAME="$TAG"
export RESULT_DIR="$REPO_ROOT/results_ixci/$TAG"
#  4. The aiperf uv cache defaults to PID-scoped, so every run re-downloads
#     ~700 MB of wheels (~25 min of wall clock, no effect on the numbers).
export AIPERF_UV_CACHE_DIR="${AIPERF_UV_CACHE_DIR:-/dev/shm/aiperf-uv-cache}"

# ---- per-concurrency topology ---------------------------------------------
# conc-1 measured on DCP8 + LMCache DRAM; conc-2/4 on DCP1 with no offload,
# which is what configs/amd-master.yaml ships for those points.
case "$CONC" in
  1)
    export DCP_SIZE=8
    export KV_OFFLOADING=dram KV_OFFLOAD_BACKEND=lmcache TOTAL_CPU_DRAM_GB=803
    # The nightly-rocm asset index is not immutable and rotates dev pins; dev105
    # and dev89 are already gone. Re-pin if this stops resolving.
    export LMCACHE_VERSION="${LMCACHE_VERSION:-0.5.5.dev114+rocm7.2}"
    ;;
  2|4)
    export DCP_SIZE=1
    export KV_OFFLOADING=none
    ;;
  *)
    echo "ERROR: CONC must be 1, 2 or 4 (those are the measured points)" >&2; exit 1;;
esac

# ---- offline + loader ------------------------------------------------------
# Everything (target + DSpark draft) must already be in $HF_ROOT. vLLM dies
# before /health if the Hub rate-limits this IP.
export HF_HUB_CACHE="$HF_ROOT"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

# EP-off OOMs in the fastsafetensors GPU-staging producer on this image
# ([PG0] Producer batch 0 failed). expandable_segments is refused by
# LMCacheMPConnector, so fall back to the plain loader: appended last, and
# argparse takes the final --load-format. Load-time only, no steady-state effect.
# TRAP: bare JSON in this variable is brace-expanded by the recipe's eval --
# single-quote any JSON you add here.
export EXTRA_VLLM_ARGS="${EXTRA_VLLM_ARGS:---load-format auto}"

# Tuned bf16 GEMM rows for this image line. Without it the serve logs hundreds
# of `not found tuned config` misses and falls back to untuned kernels.
export AITER_CONFIG_GEMM_BF16="${AITER_CONFIG_GEMM_BF16:-$REPO_ROOT/patches/k3-dcp8/aiter/merged_bf16_tuned_gemm.csv}"

# The a8w4 MoE knobs (SITUV2 etc.) are exported by the recipe itself and are
# already correct -- do not set them here. See the recipe's env block.

# ---- preflight -------------------------------------------------------------
# Every check below fails in seconds. Without them the run dies AFTER loading
# ~1.5 TB of weights, or worse, completes and produces a number that is not
# comparable. Set REPRO_SKIP_PREFLIGHT=1 only if you know why.
preflight() {
    local bad=0

    # 1. aiter MUST be rebuilt from source with patches/k3-dcp8/aiter/ applied.
    #    The recipe patches vLLM in place but CANNOT rebuild aiter (a .cu change
    #    needs a compile), so this is the one silent manual prerequisite.
    #    Without 0001-k3-dcp8-code.patch there is no asm cprr kernel and the
    #    whole DCP decode path falls back or fails. See patches/k3-dcp8/README.md section 2.
    local adir
    adir=$(python3 -c 'import aiter, os; print(os.path.dirname(aiter.__file__))' 2>/dev/null || true)
    if [ -z "$adir" ]; then
        echo "PREFLIGHT FAIL: aiter is not importable" >&2; bad=1
    elif ! grep -q cprr "$adir/ops/attention.py" 2>/dev/null; then
        echo "PREFLIGHT FAIL: aiter lacks the DCP cprr path -- patches/k3-dcp8/aiter/" >&2
        echo "  was not applied, or aiter was installed as a wheel instead of rebuilt." >&2
        echo "  Fix: follow patches/k3-dcp8/README.md section 2 (source rebuild required)." >&2
        bad=1
    else
        echo "  ok  aiter DCP cprr path present ($adir)"
    fi

    # 1b. Image line. The plain nightly-<sha> tags are ROCm 7.2.3; the
    #     nightly-rocm100-* tags are ROCm 10 and measured ~9.5% SLOWER on this
    #     workload (they also break the torch profiler in vLLM workers and make
    #     the aiter GEMM tuner report 0 us). Every published number is 7.2.x.
    local rocmv
    rocmv=$(cat /opt/rocm/.info/version 2>/dev/null | cut -d. -f1,2)
    if [ -z "$rocmv" ]; then
        echo "  ..  ROCm version not detectable (/opt/rocm/.info/version missing)"
    elif [ "$rocmv" != "7.2" ]; then
        echo "PREFLIGHT FAIL: ROCm $rocmv -- the published numbers are ROCm 7.2.3." >&2
        echo "  A rocm100 image is ~9.5% slower here and is NOT comparable." >&2
        echo "  Fix: use the image pinned in patches/k3-dcp8/README.md." >&2
        bad=1
    else
        echo "  ok  ROCm $(cat /opt/rocm/.info/version)"
    fi

    # 2. Target weights.
    [ -d "$MODEL_PATH" ] && echo "  ok  target weights: $MODEL_PATH" \
        || { echo "PREFLIGHT FAIL: MODEL_PATH does not exist: $MODEL_PATH" >&2; bad=1; }

    # 3. DSpark draft, staged AND causal. dflash_config.causal lives in the
    #    CHECKPOINT, not in vLLM, and it does not survive a re-download. The
    #    recipe rewrites it on every launch, but only in caches it can find --
    #    so the draft has to be cached here or the recipe's `hf download` will
    #    fail under HF_HUB_OFFLINE=1.
    local nd
    nd=$(ls -d "$HF_ROOT"/models--Inferact--Kimi-K3-DSpark/snapshots/*/ 2>/dev/null | wc -l)
    if [ "$nd" -eq 0 ]; then
        echo "PREFLIGHT FAIL: DSpark draft not staged under $HF_ROOT" >&2
        echo "  HF_HUB_OFFLINE=1 is set, so the recipe's 'hf download' cannot fetch it." >&2
        echo "  Fix: stage it once with HF_HUB_OFFLINE=0, or point HF_ROOT at the cache that has it." >&2
        bad=1
    else
        echo "  ok  DSpark draft staged ($nd snapshot(s))"
    fi

    # 4. conc-1 only: the DRAM KV tier is sized in GB and is MACHINE-SPECIFIC.
    #    803 is what this box used. On a box with less free RAM -- remember the
    #    weights may also sit in /dev/shm and count against it -- lower it.
    if [ "$CONC" = "1" ]; then
        local availgb
        availgb=$(awk '/MemAvailable/{printf "%d", $2/1048576}' /proc/meminfo 2>/dev/null || echo 0)
        echo "  ..  host MemAvailable=${availgb} GB, TOTAL_CPU_DRAM_GB=${TOTAL_CPU_DRAM_GB}"
        if [ "$availgb" -gt 0 ] && [ "$TOTAL_CPU_DRAM_GB" -gt "$availgb" ]; then
            echo "PREFLIGHT WARN: TOTAL_CPU_DRAM_GB (${TOTAL_CPU_DRAM_GB}) exceeds MemAvailable (${availgb} GB)." >&2
            echo "  LMCache grows lazily (l1-init 256 GB) so this may still run, but set" >&2
            echo "  TOTAL_CPU_DRAM_GB=<something under ${availgb}> if the box starts swapping." >&2
        fi
    fi

    [ "$bad" -eq 0 ] || { echo "PREFLIGHT FAILED -- not launching." >&2; exit 1; }
}
[ "${REPRO_SKIP_PREFLIGHT:-0}" = "1" ] || preflight

echo "=== K3 agentic repro: CONC=$CONC DCP=$DCP_SIZE EP_SIZE=$EP_SIZE tag=$TAG ==="
bash "$REPO_ROOT/benchmarks/single_node/agentic/kimik3_fp4_mi355x_mtp.sh"
rc=$?
echo "REPRO_RC=$rc"

# ---- verification: a run that does not pass these is not comparable --------
cat <<EOF

Verify before comparing any number:

  # 1. a8w4 expert GEMMs actually dispatched (NOT the vLLM backend log line --
  #    'AITER_MXFP4_BF16' is a backend FAMILY name, not the dtype)
  grep -c afp8_wfp4 $RESULT_DIR/server.log        # expect ~80, and 0 is a red flag

  # 2. pure TP8 -- no expert parallel
  grep -c enable-expert-parallel $RESULT_DIR/vllm_command.txt   # expect 0

  # 3. acceptance length matches (the metric is per-token, so AL must match)
  python3 - <<'PY'
import json
d=json.load(open("$RESULT_DIR/aiperf_artifacts/server_metrics_export.json"))["metrics"]
a=d["vllm:spec_decode_num_accepted_tokens"]["series"][0]["stats"]["total"]
n=d["vllm:spec_decode_num_drafts"]["series"][0]["stats"]["total"]
print(f"AL = {a/n+1:.4f}   (expect ~3.84)")
PY

  # 4. workload identity -- n / ISL med / OSL med must match the table above,
  #    or the delta is a workload artifact rather than a config effect.
EOF
