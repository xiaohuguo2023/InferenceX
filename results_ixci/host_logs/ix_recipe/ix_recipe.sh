#!/bin/bash
# Kimi-K3 conc-1 IX recipe, one 60-min scored run. Run this ON a GPU node.
#
#   bash ix_recipe.sh old                    # nightly-29468dde + v4 overlap (172.89)
#   bash ix_recipe.sh latest                 # nightly-1a001d58 + armB + v4 overlap
#   bash ix_recipe.sh old mytag              # optional tag to keep result dirs apart
#   OVERLAP=0 bash ix_recipe.sh old          # without the overlap stack (163.94-165.23)
#   LOCAL_OUT=1 bash ix_recipe.sh old        # write run output to container-local disk
#
# LOCAL_OUT moves the continuously-written output (gpu_metrics.csv ~36 MB and
# server.log ~37k lines) off /shared_nfs and onto the container overlay, then
# copies the artifacts back for scoring. On n207 that measured 167.64 against
# 164.78 for the identical recipe writing to NFS, same node, same hour. That
# is +2.9 but our spread on a fixed recipe is ~4 points, so it needs repeating
# before anyone believes it.
#
# To get onto a node from the login host (docker is not usable on slog-006):
#   srun --jobid=<JOBID> --overlap --pty bash -lc 'bash /shared_nfs/xiaohugu/k3_crossover/ab/ix_recipe.sh old'
# or, so it survives the session dropping:
#   srun --jobid=<JOBID> --overlap bash -lc 'setsid nohup bash /shared_nfs/xiaohugu/k3_crossover/ab/ix_recipe.sh old > /shared_nfs/xiaohugu/ix_old.out 2>&1 &'
#
# Bars: 172.97-175.43 historical (Sept, 9+ nodes) | 165.23 best old today | ~161 latest today.
# Our run-to-run spread on a fixed recipe is ~4 points, so one run only settles
# the question if it lands at 171+.
set -uo pipefail

WHICH=${1:-old}
case "$WHICH" in
  old)          IMG=vllm/vllm-openai-rocm:nightly-rocm100-29468dde8b515031dc6d4d9d06bf0a2fa0442098 ;;
  latest|new)   WHICH=new
                IMG=vllm/vllm-openai-rocm:nightly-rocm100-1a001d584244bd21829a5e5bd4a376ea141c0ce3 ;;
  *) echo "usage: $0 {old|latest} [tag]"; exit 2 ;;
esac
TAG=${2:-}

HOSTN=$(hostname); NODE=${HOSTN##*-}
case "$HOSTN" in
  *slog*) echo "refusing to run on the login node ($HOSTN): docker is not usable here."; exit 2 ;;
esac

C=k3-repro-385dce-n$NODE
BASE=/shared_nfs/xiaohugu/k3_crossover
RC=/inferencex_dcp/results_ixci
# OUT_BASE picks the filesystem the run writes to. The September 171-175 runs
# wrote into the repo's results_ixci on /home; everything we ran on Oct 5 wrote
# to /shared_nfs and topped out at 165. Both are NFS but they are different
# volumes, so this is worth holding as a variable:
#   OUT_BASE=/home/xiaohugu/work/InferenceX_dcp/results_ixci   (historical)
OUT_BASE=${OUT_BASE:-$BASE/ab/n$NODE}
D=$OUT_BASE/ix_${WHICH}${TAG:+_$TAG}
mkdir -p "$D" "$BASE/cache/n${NODE}_$WHICH"
log() { echo "[$(date -u +%H:%M:%S)] n$NODE $WHICH $*"; }

log "setup $IMG"
IMAGE=$IMG SKIP_REAP=1 bash /home/xiaohugu/work/InferenceX_dcp/_k3_ixci_repro_setup.sh "$NODE" \
  > "$D/setup.log" 2>&1
grep -aq SETUP_OK "$D/setup.log" || { log "SETUP FAILED"; tail -20 "$D/setup.log"; exit 1; }
docker exec "$C" bash -lc 'python3 -c "import vllm;print(\"vllm\",vllm.__version__)"'
docker exec "$C" bash -lc 'pip install -q tabulate' || true

if [ "${LOCAL_OUT:-0}" = 1 ]; then
  SERVE_OUT=/tmp/ixlocal_n$NODE/serve; IX_OUT=/tmp/ixlocal_n$NODE/ix
  docker exec "$C" bash -lc "mkdir -p $SERVE_OUT $IX_OUT && df -PT $IX_OUT | tail -1"
else
  SERVE_OUT=$D; IX_OUT=$D
fi

if [ "$WHICH" = new ]; then
  log "runtime PRs (WITH4) + h12 + rebased MLA graph schedule + armB table"
  bash $BASE/ab/setup_ab_node.sh > "$D/stack.log" 2>&1
  grep -aq AB_SETUP_OK "$D/stack.log" || { log "stack FAILED"; tail -20 "$D/stack.log"; exit 1; }
fi

# Both arms need this: h12 runs 12-head MLA and stock aiter on the old image
# asserts num_head_qo % 8 == 0. Removing it is not an option -- a run without
# the coef port dies on that assert.
log "aiter coef port (nhead12)"
timeout 2700 docker exec -e TEST_GPU=0 -e BUNDLE=$RC/host_logs/aiter_port_coef \
  "$C" bash $RC/host_logs/k3_aiter_port_install2.sh > "$D/aiter_port.log" 2>&1
docker exec "$C" python3 -c "
import inspect, aiter.ops.attention as a
print('nhead12 port active:', 'num_head_qo % 8' not in inspect.getsource(a.get_mla_metadata_info_v1))"

# The v4 overlap stack is what separates 173 from 164. It runs the MoE router
# down-projection and the MLA gate projection on a side stream concurrently with
# attention, and because it only engages at <=128 tokens per step it affects
# decode and leaves prefill alone -- which is exactly the shape of the deficit
# measured without it (uniform -4% per step at every context and output length).
# Verified on n217: 163.94 without, 172.89 with, same node and image.
#
# Install it before serve, the way k3_install_v4.sh did: no EXTRA_PRS patch
# touches envs.py, linear.py or mla.py, so the launcher's later patching does
# not collide. mla_overlap_gate_v4.py imports sigmoid_gate_mul, so prgate has to
# supply ops/sigmoid_gate.py first -- but gate must then come OUT of EXTRA_PRS,
# or the launcher tries to re-apply it to the replaced mla.py and aborts.
if [ "${OVERLAP:-1}" = 1 ]; then
  log "v4 overlap stack"
  docker exec "$C" bash -lc '
    set -e; S=/usr/local/lib/python3.12/dist-packages; H=/inferencex_dcp/results_ixci/host_logs
    test -f $S/vllm/models/kimi_k3/amd/ops/sigmoid_gate.py \
      || patch -s -p1 -d $S < $H/reverts/prgate.patch
    if ! grep -q "_ROUTED_DOWN_PROJ_STREAM_TOKEN_THRESHOLD = 128" $S/vllm/models/kimi_k3/amd/linear.py; then
      for p in prk3overlap_core_v3 prk3overlap_core_v2 prk3overlap_core; do
        if patch --dry-run -s -R -p1 -d $S < $H/reverts/$p.patch >/dev/null 2>&1; then
          patch -s -R -p1 -d $S < $H/reverts/$p.patch; break
        fi
      done
      patch -s -p1 -d $S < $H/reverts/prk3overlap_core_v4.patch
    fi
    cp $H/mla_overlap_gate_v4.py $S/vllm/models/kimi_k3/amd/mla.py
    cd /tmp && python3 -c "
import vllm.models.kimi_k3.amd.linear as l
assert l._ROUTED_DOWN_PROJ_STREAM_TOKEN_THRESHOLD == 128
from vllm.models.kimi_k3.amd.mla import KimiK3MultiHeadLatentAttentionWrapper
print(\"v4 installed\")"' > "$D/overlap.log" 2>&1
  grep -aq 'v4 installed' "$D/overlap.log" \
    || { log "v4 overlap FAILED"; tail -15 "$D/overlap.log"; exit 1; }
  OVERLAP_ENV=(-e VLLM_KIMI_K3_AMD_MOE_ROUTER_DOWN_PROJ_STREAM=1
               -e VLLM_KIMI_K3_AMD_MLA_GATE_STREAM=1)
  GATE_PR=
else
  OVERLAP_ENV=(); GATE_PR=" gate"
fi

if [ "$WHICH" = old ]; then
  ARMENV=(-e PR58814=1 -e EXTRA_PRS="57978 58633 58381 58732 58737 h12${GATE_PR} 58861 router"
          -e AITER_PR5709_DIR=.aiter_pr5709)
else
  ARMENV=(-e PR58814=0 -e EXTRA_PRS=
          -e SKIP_EXTRA=utils\ kda\ kda_triton_spec_decode\ linear\ rocm_aiter_mla\ rocm_aiter_mla_graph_schedule
          -e AITER_PR5709_DIR=.aiter_pr5709_head)
fi
COMMON=( -e ARM=dcp1gs -e TAG_PREFIX=ix$WHICH "${ARMENV[@]}" "${OVERLAP_ENV[@]}"
  -e DCP_SIZE_OVERRIDE=1 -e KV_OFFLOADING_OVERRIDE=none
  -e CUDAGRAPH_MODE=FULL_DECODE_ONLY -e AITER_MLA_DEFAULT_MAX_SPLIT=128
  -e VLLM_DISABLE_SHARED_EXPERTS_STREAM=0 -e SPEC_REAL_ACCEPT=false
  -e XDG_CACHE_HOME=$BASE/cache/n${NODE}_$WHICH
  -e TORCHINDUCTOR_CACHE_DIR=$BASE/cache/n${NODE}_$WHICH/inductor
  -e TRITON_CACHE_DIR=$BASE/cache/n${NODE}_$WHICH/triton
  -e VLLM_CACHE_ROOT=$BASE/cache/n${NODE}_$WHICH/vllm )

log "serve"
docker exec -d -e SERVE_ONLY=true -e RESULT_DIR=$SERVE_OUT "${COMMON[@]}" "$C" \
  bash -lc "bash $BASE/ab/k3_ix_dur.sh $NODE > $D/launch.log 2>&1"
for i in $(seq 1 150); do
  grep -aq "SERVE_ONLY=true: ready" "$D/launch.log" 2>/dev/null && { log ready; break; }
  grep -aqE "Traceback|Process died|CUDA out of memory" "$D/launch.log" 2>/dev/null \
    && { log "serve FAILED"; grep -aE "Error|Traceback|assert" "$D/launch.log" | tail -12; exit 1; }
  sleep 20
done
grep -aq "SERVE_ONLY=true: ready" "$D/launch.log" || { log "serve timeout"; exit 1; }

log "batch probe (bars: old 6.456 | latest+armB 6.770)"
for pc in 1 2; do
  docker exec -e WORKDIR=$D/pc$pc -e SEED=7 -e ISLS=88000 -e WEIGHTS=1.0 -e OSL=580 \
    -e REQ_COUNT=12 -e PCONC=$pc "$C" bash $BASE/ab/proxy_conc.sh > "$D/pc$pc.log" 2>&1
  grep -a '^ISL' "$D/pc$pc.log" | tail -1
done

log "full 60-min IX"
docker exec -e CLIENT_ONLY=true -e DURATION=3600 -e RESULT_DIR=$IX_OUT "${COMMON[@]}" "$C" \
  bash -lc "bash $BASE/ab/k3_ix_dur.sh $NODE" > "$D/ix.log" 2>&1

log "score"
docker exec "$C" bash -lc "python3 $RC/host_logs/ix_frITL_sum.py $IX_OUT"
if [ "$IX_OUT" != "$D" ]; then
  docker exec "$C" bash -lc "cp -r $IX_OUT $D/ix_artifacts; cp $SERVE_OUT/server.log $D/ 2>/dev/null; chmod -R a+rX $D" || true
fi
# validity: need >=200 scored requests, <1% errors, and exactly one engine init
printf "  errors=%s  engine_inits=%s\n" \
  "$(grep -aoE 'errors=[0-9]+' "$D/ix.log" | tail -1)" \
  "$(grep -acE 'Engine core|init engine' "$D/server.log" 2>/dev/null)"
docker exec "$C" python3 $RC/host_logs/k3_stop_serve.py 2>/dev/null || true
echo "IX_RECIPE_DONE n$NODE $WHICH -> $D"
