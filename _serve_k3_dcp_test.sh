#!/bin/bash
# K3 Attention Benchmark — SPECULATIVE server (DSpark draft).
# Identical to _serve_k3_bench_baseline.sh EXCEPT it adds --speculative-config
# with the DSpark draft. Parameterized by NUM_SPEC (2 or 7 per the doc).
#
#   NUM_SPEC=2 PORT=8889 bash _serve_k3_bench_spec.sh
#
# Spec config follows K3_Attention_Benchmark_Instructions.md:
#   method=dspark, attention_backend=TRITON_MLA,
#   draft_sample_method=probabilistic, rejection_sample_method=block.
set -uo pipefail

# Preflight: refuse to launch onto a box that will make the load hang.
#
# Measured: H2D here is fixed-cost bound (~0.2 s per copy regardless of size or
# pinning) whenever a second process set is GPU-resident, against ~54 GB/s
# clean. The K3 checkpoint is ~519,000 tensors (96 shards x ~5,400, median
# 1.31 MiB), so a degraded box turns weight loading into hours -- the "580 s per
# shard, ETA 15 h" wedge that reads as a silent hang and has repeatedly been
# misattributed to DCP. 20 s of checking beats 15 h of waiting.
# Set SKIP_PREFLIGHT=1 to bypass.
if [ "${SKIP_PREFLIGHT:-0}" != "1" ]; then
  _pf="$(dirname "$(readlink -f "$0")")/_gpu_preflight.py"
  if [ -f "$_pf" ]; then
    echo "=== GPU preflight ==="
    if ! python3 "$_pf" --min-gibs "${PREFLIGHT_MIN_GIBS:-20}" \
                        --stable-for "${PREFLIGHT_STABLE_FOR:-30}" \
                        --wait "${PREFLIGHT_WAIT:-0}"; then
      echo "ABORT: box is degraded; not launching. SKIP_PREFLIGHT=1 to override."
      exit 1
    fi
    echo
  fi

  # The H2D gate above is single-process, and MEASURED 2026-08-21 it is blind to
  # the failure that actually costs us days: a box where single-process work is
  # perfectly clean (53 GiB/s, 25 us H2D, 17 us kernel+sync) while anything
  # multi-rank collapses. On that box, with NOBODY else resident, 8-rank RCCL
  # init took 206 s against a 20 s reference and weight load ran at 495 s/shard
  # (ETA 12h55m) against a ~1.13 it/s reference. The old gate said HEALTHY and
  # waved the boot straight into the wedge.
  #
  # So also check a collective. This is self-limiting: ~30 s when the box is
  # fine, and when it is not we abort in COLLECTIVE_TIMEOUT_S instead of
  # discovering it 13 hours later. Deliberately run at DEFAULT channel count --
  # NCCL_MAX_NCHANNELS below is a mitigation, and capping it here would mask the
  # very state we are testing for.
  _rp="$(dirname "$(readlink -f "$0")")/_rccl_init_probe.py"
  if [ "${SKIP_COLLECTIVE_CHECK:-0}" != "1" ] && [ -f "$_rp" ]; then
    echo "=== collective preflight (8-rank RCCL; healthy ~20-32 s) ==="
    # Capture to a file rather than piping: a pipe would hand us grep's exit
    # status, and grep still matches the partial per-rank stamps a timed-out
    # run leaves behind -- so the gate would pass exactly when it must fail.
    # env -u (not VAR=) so the probe really runs at the default channel count.
    _rout="$(mktemp)"
    ( cd "$(dirname "$_rp")" && \
      env -u NCCL_MAX_NCHANNELS timeout --signal=TERM --kill-after=20s \
        "${COLLECTIVE_TIMEOUT_S:-90}s" \
        torchrun --standalone --nproc-per-node=8 _rccl_init_probe.py ) \
      > "$_rout" 2>&1
    _rrc=$?
    grep -aE "^\[rank 0\]|PASS" "$_rout"
    if [ "$_rrc" -ne 0 ] || ! grep -aq "PASS" "$_rout"; then
      rm -f "$_rout"
      echo
      echo "ABORT: 8-rank RCCL init did not finish in ${COLLECTIVE_TIMEOUT_S:-90}s."
      echo "  The box is in the degraded state where collectives and weight load"
      echo "  crawl while single-process work looks fine. Booting now would wedge"
      echo "  for hours and any number it produced would be contaminated."
      echo "  Sharper confirmation (~2.5 min, no weights), healthy 0.08 ms/call:"
      echo "    cd k3_dcp_direct_hip && ITERS=2 torchrun --standalone \\"
      echo "        --nproc-per-node=8 _test_a2a_syncfree.py   # grep EVENT"
      echo "  SKIP_COLLECTIVE_CHECK=1 overrides."
      exit 1
    fi
    rm -f "$_rout"
    echo
  fi
fi

cd /tmp
PORT="${PORT:-8889}"
NUM_SPEC="${NUM_SPEC:-2}"
GPU_MEM="${GPU_MEM:-0.88}"; MAX_NUM_SEQS="${MAX_NUM_SEQS:-16}"; MNBT="${MNBT:-4096}"
# Spec-decode TARGET backend.
#
# NOTE (fp8+DSpark is architecturally unrunnable on this nightly):
#   The DSpark draft uses semi-autoregressive parallel drafting, which forces
#   non-causal attention (dspark/utils.py: use_non_causal=dflash_has_any_non_causal).
#   On ROCm, ONLY TRITON_MLA advertises supports_non_causal=True; ROCM_AITER_MLA
#   does not. But TRITON_MLA sets supports_quant_query_input=False, and K3's
#   _decode_concat_cache asserts that flag whenever KV is fp8. So:
#     - draft on ROCM_AITER_MLA -> "non-causal attention not supported" (build fails)
#     - draft on TRITON_MLA + fp8 KV -> K3 fp8-query assert (after capture)
#   => DSpark REQUIRES bf16 KV + TRITON_MLA. (PR #51011 correctly fixes the TARGET's
#      fp8 verify routing, but cannot bridge the draft's non-causal requirement.)
#      The doc's reference KV was default (bf16) anyway; fp8 was our own delta.
# With the draft forced causal (dflash_config.causal=true in the draft config),
# use_non_causal=False, so the draft no longer needs TRITON_MLA and can run on the
# fp8 asm path. Target + draft both on ROCM_AITER_MLA + fp8 KV (the MI355X perf
# path). PR #51011 makes the 12-head fp8 spec verify route to the asm q-row-fold.
ATTN_BACKEND="${ATTN_BACKEND:-ROCM_AITER_MLA}"
KV_CACHE_DTYPE="${KV_CACHE_DTYPE:-fp8}"
# Real perf requires cudagraphs — NO eager. PR #51011 also fixes the persistent
# metadata gate so PIECEWISE/FULL capture completes under spec (production run
# captured 35 piecewise + 19 full graphs). Default ENFORCE_EAGER=0.
ENFORCE_EAGER="${ENFORCE_EAGER:-0}"
EAGER_ARG=(); [ "$ENFORCE_EAGER" = "1" ] && EAGER_ARG=(--enforce-eager)

export HF_HUB_CACHE=/dev/shm/hf-cache HF_HOME=/dev/shm/hf-cache
CACHE=/dev/shm/hf-cache/models--moonshotai--Kimi-K3/snapshots
MODEL_PATH="${MODEL_PATH:-$(ls -d "$CACHE"/*/ 2>/dev/null | head -1)}"; MODEL_PATH="${MODEL_PATH%/}"
[ -n "$MODEL_PATH" ] && [ -f "$MODEL_PATH/config.json" ] || { echo "!! K3 weights not staged at $CACHE"; exit 1; }

# resolve local DSpark draft snapshot (offline/portable)
DCACHE=/dev/shm/hf-cache/models--Inferact--Kimi-K3-DSpark/snapshots
DRAFT_PATH="${DRAFT_PATH:-$(ls -d "$DCACHE"/*/ 2>/dev/null | head -1)}"; DRAFT_PATH="${DRAFT_PATH%/}"
if [ -z "$DRAFT_PATH" ] || [ ! -f "$DRAFT_PATH/config.json" ]; then
  echo "!! DSpark draft not staged at $DCACHE — falling back to HF id Inferact/Kimi-K3-DSpark"
  DRAFT_PATH="Inferact/Kimi-K3-DSpark"
else
  # Refuse to start if the draft is still non-causal (cudagraph OOB source).
  if ! python3 - "$DRAFT_PATH/config.json" <<'PY'
import json, sys
c = json.load(open(sys.argv[1]))
if c.get("dflash_config", {}).get("causal") is not True:
    raise SystemExit("dflash_config.causal is not true — run _k3_dspark_fp8asm_apply_patches.sh")
print("draft causal OK")
PY
  then
    echo "!! draft must be forced causal before serve — run _k3_dspark_fp8asm_apply_patches.sh"
    exit 1
  fi
fi
echo "MODEL_PATH=$MODEL_PATH"
echo "DRAFT_PATH=$DRAFT_PATH  NUM_SPEC=$NUM_SPEC  PORT=$PORT"

# DRAFT attention backend. With the draft forced causal it runs on the fp8 asm
# path like the target. ROCM_AITER_MLA (post PR #51011) accepts fp8 query and
# routes small-head verify to the asm fold.
DRAFT_BACKEND="${DRAFT_BACKEND:-ROCM_AITER_MLA}"
# Rejection sampling method:
#   block (default) = REAL target-vs-draft verify (correct output, real ~2.4 AL,
#                     with the real acceptance tail). Our honest internal number.
#   synthetic       = AgentX-PRESCRIBED methodology. Draft+target still run (same
#                     compute) but accept/reject is pinned to the committed golden
#                     AL, removing draft-quality variance. This is what BOTH the
#                     official MI355X recipe (PR #2508) and the B300 baseline
#                     (kimik3_fp4_b300_vllm_mtp.sh) publish — synthetic-vs-synthetic
#                     is the only apples-to-apples comparison. Golden AL (K=2,
#                     thinking_on) = 2.51 per
#                     golden_al_distribution/kimik3_dspark_probabilistic_sample_method_block_rejection_sample_method.yaml.
#                     Set SYNTHETIC_ACCEPT_LEN to enable (2.51 for NUM_SPEC=2;
#                     3.00 for 3, 3.84 for 7 — see the golden YAML).
if [ -n "${SYNTHETIC_ACCEPT_LEN:-}" ]; then
    SPEC_CFG=$(printf '{"model":"%s","num_speculative_tokens":%s,"method":"dspark","attention_backend":"%s","draft_sample_method":"probabilistic","rejection_sample_method":"synthetic","synthetic_acceptance_length":%s}' "$DRAFT_PATH" "$NUM_SPEC" "$DRAFT_BACKEND" "$SYNTHETIC_ACCEPT_LEN")
    echo "SPEC rejection=SYNTHETIC (AgentX golden AL=$SYNTHETIC_ACCEPT_LEN) — matches PR#2508 / B300 baseline"
else
    SPEC_CFG=$(printf '{"model":"%s","num_speculative_tokens":%s,"method":"dspark","attention_backend":"%s","draft_sample_method":"probabilistic","rejection_sample_method":"block"}' "$DRAFT_PATH" "$NUM_SPEC" "$DRAFT_BACKEND")
    echo "SPEC rejection=block (REAL verify) — honest internal number, NOT AgentX-comparable; set SYNTHETIC_ACCEPT_LEN=2.51 for the published methodology"
fi

# cudagraph mode + optional capture-size cap. fp8-asm DSpark MUST use
# MAX_NUM_SEQS=16 (recipe default above) — not the agentic 2*CONC ladder.
# With NUM_SPEC=2, vLLM derives max_cudagraph_capture_size = 16*(1+N)*2 = 96,
# so PIECEWISE capture still walks M=96..72..8; the fp8 asm verify kernel OOBs
# when max_num_seqs is large (M~304 at 64 seqs). MAX_CG overrides the ceiling
# for experiments (batches above the cap run outside the graph, not eager).
# KV-cache memory pin (bytes). At gpu_memory_utilization=0.95 with TP8 K3-fp4,
# weights alone are ~201 GiB of the 273.6 GiB budget. vLLM's auto-sizer hands the
# leftover to KV using only the *profiled* peak-activation estimate (~5.5 GiB) as
# headroom. That estimate is far below the REAL prefill peak: MNBT=16384 chunks x
# up-to-64 concurrent 68k-token requests + DSpark verify buffers spike much
# higher, so the process OOMs to "0 MB free" — at init if KV is sized to the edge,
# or mid-benchmark once real traffic hits the peak (VllmWorker died at conc-48).
# The fix is NOT to touch the mandated knobs (gpu_mem 0.95 / seqs 64 / MNBT /
# FULL_AND_PIECEWISE) but to shrink KV so a large physical headroom remains for
# those runtime spikes. Prefix caching stores the 63,911-tok prefix ONCE, so this
# workload needs only ~350k KV tokens (~6 GiB); 32 GiB (~2M tokens) is >5x that
# while leaving ~32 GiB physical headroom for the prefill/verify activation peak.
# Set KV_CACHE_MEMORY=none|auto|0 to DROP the pin and let vLLM's memory profiler
# auto-size KV (determine_available_memory). Note: `:-` treats empty-string the
# same as unset, so `KV_CACHE_MEMORY=""` still gets the 32 GiB default — you must
# pass the explicit sentinel to disable pinning.
KV_CACHE_MEMORY="${KV_CACHE_MEMORY:-34359738368}"
case "$KV_CACHE_MEMORY" in
  none|auto|AUTO|0) KV_CACHE_MEMORY="" ;;
esac
KVMEM_ARG=(); [ -n "$KV_CACHE_MEMORY" ] && KVMEM_ARG=(--kv-cache-memory "$KV_CACHE_MEMORY")

# Optional CPU KV-cache offload — the MI355X analogue of the NV B300-MTP
# offload_mode="on" arm. On the long-ISL agentic corpus, at conc>=8 the on-device
# KV pin saturates (peak ~97-100%), the shared prefix gets evicted (prefix-cache
# hit collapses ~75%->15%), prefill recompute explodes and TTFT/throughput fall off
# a cliff. Spilling evicted KV blocks to host keeps the prefix resident so the hit
# rate (and thus throughput) holds. Activates ONLY when KV_OFFLOADING_SIZE (total
# GiB summed across all TP ranks) is set; empty default => no offload, so this is a
# no-op for any run that doesn't opt in (in-flight sweeps are unaffected).
KV_OFFLOADING_SIZE="${KV_OFFLOADING_SIZE:-}"
KVOFF_ARG=()
[ -n "$KV_OFFLOADING_SIZE" ] && KVOFF_ARG=(--kv-offloading-size "$KV_OFFLOADING_SIZE" \
  --kv-offloading-backend "${KV_OFFLOADING_BACKEND:-native}")
[ -n "$KV_OFFLOADING_SIZE" ] && echo "KV OFFLOAD on: ${KV_OFFLOADING_SIZE} GiB total (backend=${KV_OFFLOADING_BACKEND:-native})"

CUDAGRAPH_MODE="${CUDAGRAPH_MODE:-FULL_AND_PIECEWISE}"
MAX_CG="${MAX_CG:-}"
# cudagraph_capture_sizes — pin explicitly so DSpark decode (M = 3*conc tokens/step,
# uniform_decode_query_len = 1+num_spec = 3) lands on FULL decode graphs at every
# benchmark concurrency. vLLM derives the FULL/decode graph set as round_up(size,3)
# over this ladder; the AUTO ladder [1,2,4,8,16,24,32,40,...] leaves gaps at 12 and
# 36 (8->9,16->18 skip 12; 32->33,40->42 skip 36), so conc-4 (12 tok) and conc-12
# (36 tok) silently fall to a PIECEWISE graph -> attention runs eager every step ->
# get_mla_metadata_v1 + eager MLA on the host critical path (~75 ms ITL bubble; GPU
# at ~370 W launch-bound). Adding 12 and 36 gives exact FULL decode graphs (num_reqs
# 4 and 12), where get_mla_metadata_v1 takes the fast uniform branch. Rule to extend:
# for a new concurrency C, ensure a size s with round_up(s,3)==3*C exists (add 3*C).
# The rest of the list reproduces vLLM's auto ladder (setting this disables auto-gen).
CAPTURE_SIZES="${CAPTURE_SIZES:-1,2,4,8,12,16,24,32,36,40,48,56,64,72,80,88,96,104,112,120,128,136,144,152,160,168,176,184,192,200,208,216,224,232,240,248,256,272,288,304,320,336,352,368,384}"
if [ -n "$MAX_CG" ]; then
  COMPILE_CFG=$(printf '{"cudagraph_mode":"%s","custom_ops":["+fused_rms_norm_gated"],"cudagraph_capture_sizes":[%s],"max_cudagraph_capture_size":%s}' "$CUDAGRAPH_MODE" "$CAPTURE_SIZES" "$MAX_CG")
else
  COMPILE_CFG=$(printf '{"cudagraph_mode":"%s","custom_ops":["+fused_rms_norm_gated"],"cudagraph_capture_sizes":[%s]}' "$CUDAGRAPH_MODE" "$CAPTURE_SIZES")
fi

MERGED_GEMM_CSV=/opt/aiter-local/aiter/configs/merged_bf16_tuned_gemm.csv
if [ -z "${AITER_CONFIG_GEMM_BF16:-}" ] && [ -f "$MERGED_GEMM_CSV" ]; then
  export AITER_CONFIG_GEMM_BF16="$MERGED_GEMM_CSV"
fi

export VLLM_ROCM_USE_AITER=1 VLLM_ROCM_USE_AITER_MOE=1 SAFETENSORS_FAST_GPU=1 \
       GPU_ARCHS=gfx950 VLLM_ENGINE_READY_TIMEOUT_S=3600 VLLM_HTTP_TIMEOUT_KEEP_ALIVE=900 \
       AITER_SITUV2_A8W4=1 VLLM_ROCM_USE_AITER_MOE_SITUV2_A8W4=1 \
       VLLM_USE_BREAKABLE_CUDAGRAPH=0 \
       VLLM_ROCM_AITER_MLA_ASM_PADDING="${VLLM_ROCM_AITER_MLA_ASM_PADDING:-asm}"

# Bound RCCL communicator setup so init cannot turn into an open-ended hang.
#
# Measured 2026-08-21 with _rccl_init_probe.py (8 ranks, 1 GiB, no vLLM/weights/
# DCP), all under an identical foreign 8-rank job:
#   112 channels (RCCL's default here)  world all_reduce never finished in 200 s
#     8 channels                        world all_reduce 187.6 s, 2nd comm timed out
#     2 channels                        world 79.3 s, +tp, +dcp -> PASS at 126.9 s
# RCCL enumerates 112 channels on this node and each one's P2P/IPC + proxy
# handshake costs ~1-2 s when another process set is GPU-resident, so setup work
# alone runs to tens of minutes and reads as a frozen boot. That is what parked
# all 8 workers in ncclCommInitRank at the TP group -- before DCP's group, before
# any weight, before any kernel. Capping the channel count makes init bounded
# instead of unbounded.
#
# 16 is far above what 8 GPUs on one node need for collective bandwidth while
# cutting setup ~7x; drop to 2 if you must boot alongside another job. This is
# the DCP *test* launcher only -- _serve_k3_bench_spec.sh is deliberately left
# alone so published benchmark numbers keep their current collective config
# until this is A/B'd for throughput.
export NCCL_MAX_NCHANNELS="${NCCL_MAX_NCHANNELS:-16}"

# Optional torch profiler. Set PROFILE_DIR=/path to enable; then drive it with
# curl -X POST /start_profile ... load ... /stop_profile (per-rank .pt.trace.json.gz
# flush to PROFILE_DIR). This build honors --profiler-config, NOT the
# VLLM_TORCH_PROFILER_DIR env. Analyze with analyze_dsv4_trace.py / backend_breakdown.py.
PROFILE_DIR="${PROFILE_DIR:-}"
PROFILE_ARG=(); [ -n "$PROFILE_DIR" ] && { mkdir -p "$PROFILE_DIR"; PROFILE_ARG=(--profiler-config.profiler=torch --profiler-config.torch_profiler_dir="$PROFILE_DIR"); }

# Optional rocprofv3 kernel-trace (profiler-free, low-overhead — does NOT eager-fy
# cudagraph replay, unlike the torch profiler). Set ROCPROF_DIR=/path to enable; the
# whole serve run is traced (child TP workers included: one <pid>_kernel_trace.csv
# per rank). rocprofv3 flushes only on process EXIT, so drive a load then STOP the
# serve cleanly to get the CSVs; analyze with _rocprof_gaps.py (use --last-seconds N
# to isolate the steady-decode tail you drove right before stopping). Mutually
# exclusive with PROFILE_DIR — don't set both.
# ROCPROF_TRACE selects the rocprofv3 mode: "kernel" (default, per-kernel dispatch
# CSV for _rocprof_gaps.py) or "hip" (HIP-API-only trace, MUCH lower host overhead —
# does NOT record kernels inside a graph; use it to count hipGraphLaunch vs direct
# hip*LaunchKernel per step to prove FULL vs PIECEWISE runtime dispatch).
# PREFIX_CACHING=0 disables prefix caching. Used to isolate the DCP draft-group
# cache-hit path: on a hit the engine skips recomputing KV, so if the hit length
# is derived from the target's DCP-scaled accounting the replicated draft never
# writes 7/8 of its own KV while its block table still looks fully populated.
PREFIX_CACHE_ARG=(--enable-prefix-caching)
if [ "${PREFIX_CACHING:-1}" = "0" ]; then
  PREFIX_CACHE_ARG=(--no-enable-prefix-caching)
  echo "PREFIX CACHING DISABLED (draft-KV isolation run)"
fi

ROCPROF_DIR="${ROCPROF_DIR:-}"
ROCPROF_TRACE="${ROCPROF_TRACE:-kernel}"
ROCPROF_PREFIX=()
if [ -n "$ROCPROF_DIR" ]; then
  mkdir -p "$ROCPROF_DIR"
  case "$ROCPROF_TRACE" in
    hip)    ROCPROF_PREFIX=(rocprofv3 --hip-trace --output-format csv -d "$ROCPROF_DIR" --) ;;
    *)      ROCPROF_PREFIX=(rocprofv3 --kernel-trace --output-format csv -d "$ROCPROF_DIR" --) ;;
  esac
fi

LOG=/tmp/serve_k3_dcp${NUM_SPEC}.log
setsid nohup "${ROCPROF_PREFIX[@]}" vllm serve "$MODEL_PATH" --served-model-name Kimi-K3 \
  --host 0.0.0.0 --port "$PORT" --tensor-parallel-size 8 \
  --decode-context-parallel-size "${DCP_SIZE:-8}" --dcp-comm-backend "${DCP_BACKEND:-a2a}" --cp-kv-cache-interleave-size "${CP_INTERLEAVE:-1}" \
  --distributed-executor-backend mp --gpu-memory-utilization "$GPU_MEM" \
  --max-num-seqs "$MAX_NUM_SEQS" --max-model-len "${MAX_MODEL_LEN:-1048576}" --max-num-batched-tokens "$MNBT" \
  --trust-remote-code --load-format "${LOAD_FORMAT:-auto}" --moe-backend aiter \
  --kv-cache-dtype "$KV_CACHE_DTYPE" --attention-backend "$ATTN_BACKEND" --mm-encoder-tp-mode data \
  "${KVMEM_ARG[@]}" \
  "${KVOFF_ARG[@]}" \
  --compilation-config "$COMPILE_CFG" \
  "${PROFILE_ARG[@]}" \
  "${EAGER_ARG[@]}" \
  --speculative-config "$SPEC_CFG" \
  "${PREFIX_CACHE_ARG[@]}" --enable-prompt-tokens-details --no-disable-hybrid-kv-cache-manager \
  --reasoning-parser kimi_k3 --tool-call-parser kimi_k3 --enable-auto-tool-choice \
  --disable-uvicorn-access-log > "$LOG" 2>&1 &

echo "serving K3 spec-$NUM_SPEC (port=$PORT); log: $LOG"

# Watchdog. The preflight gate cannot win a race: measured 2026-08-21, it
# passed a clean box and a colleague's 8-rank job started 17 s after we
# launched, parking all 8 of our workers in librccl's bootstrap recv at VRAM
# 6 GiB -- never a weight loaded, never a DCP kernel run. Waiting the full
# 30 min for that teaches nothing, so notice the stall and say why.
#
# A healthy boot writes to the log continuously (shard progress, capture
# progress). Silence for STALL_S with no /health means wedged.
STALL_S="${STALL_S:-240}"
_last_mtime=0; _last_change=$(date +%s)
for i in $(seq 1 360); do
  curl -sf -m5 "http://localhost:$PORT/health" >/dev/null 2>&1 && { echo "ready $(date +%T)"; exit 0; }
  pgrep -f "vllm serve" >/dev/null 2>&1 || { echo "serve died; tail:"; tail -50 "$LOG"; exit 1; }

  _now=$(date +%s)
  _mtime=$(stat -c %Y "$LOG" 2>/dev/null || echo 0)
  [ "$_mtime" != "$_last_mtime" ] && { _last_mtime=$_mtime; _last_change=$_now; }

  if [ $((_now - _last_change)) -ge "$STALL_S" ]; then
    echo
    echo "!! WEDGED: no log output for $((_now - _last_change))s and /health is down."
    echo "-- VRAM (a stall under ~20 GiB/GPU means it never reached weight load):"
    rocm-smi --showmeminfo vram 2>/dev/null |
      awk '/Used Memory/{printf "%.0f ", $NF/1073741824}'; echo
    echo "-- KFD holders (a second job here is the usual cause; do NOT reclaim it):"
    rocm-smi --showpids 2>/dev/null | sed -n '5,20p'
    if command -v py-spy >/dev/null 2>&1; then
      _w=$(pgrep -f "multiprocessing-fork" | head -1)
      [ -n "$_w" ] && { echo "-- where worker $_w is parked:"
                        timeout 60 py-spy dump --pid "$_w" --nonblocking 2>&1 | sed -n '4,12p'; }
    fi
    echo "-- log tail:"; tail -20 "$LOG"
    echo
    echo "Aborting early instead of hanging. Re-run when the box is free;"
    echo "PREFLIGHT_WAIT=7200 queues the launch until it frees."
    exit 1
  fi
  sleep 5
done
echo "!! not ready after 30min"; tail -50 "$LOG"; exit 1
