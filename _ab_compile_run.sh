#!/usr/bin/env bash
# A/B the Kimi-K3 DSpark torch.compile / Inductor-fusion path at conc-1.
#
#   ARM=1 bash _ab_compile_run.sh     # config only, ZERO code change
#   ARM=2 bash _ab_compile_run.sh     # config + upstream PR #56664 decorator
#   ARM=3 bash _ab_compile_run.sh     # upstream scheduler PRs #54627 + #54625
#
# Run from the host. Needs the GPUs free.
#
# ---------------------------------------------------------------------------
# WHY TWO ARMS
#
# Upstream PR #56664 changed the decorator AND the compilation config together
# and says so explicitly, so its +6.7% cannot be attributed. Arm 1 isolates the
# config: it is a pure serve-flag change with no patch at all. If arm 1 already
# captures the win, we do not need the decorator (and do not need to carry an
# unmerged upstream patch). If arm 1 is flat, the win is the compiled draft
# graph and arm 2 shows it.
#
# WHAT THE CONFIG DOES
#   vllm/config/vllm.py::enable_rope_kvcache_mla_fusion:
#     use_inductor_graph_partition OR not splitting_ops_contain_kv_cache_update()
#   At -O3 with splitting_ops=None and fuse_attn_quant=IS_QUANTIZED (hard-coded
#   False, vLLM #25689), set_splitting_ops_for_v1 appends unified_kv_cache_update
#   + unified_mla_kv_cache_update, so the predicate is True and the MLA
#   rope+kvcache fusion resolves to FALSE. Flipping use_inductor_graph_partition
#   is what unlocks it. Requires torch>=2.9.0.dev; the image has 2.12.0.
#
# BASELINE: do NOT re-run it. results_ixci/abroute_asm_c1 is the same recipe,
# same container, same corpus: frITL p90 7.516 ms, intvty p90 133.04, n=204.
# ---------------------------------------------------------------------------
set -euo pipefail

CTR="${CTR:-k3-2671}"
ARM="${ARM:?set ARM=1 (config only), 2 (config + decorator) or 3 (scheduler PRs)}"
IMG="${IMG:-vllm/vllm-openai-rocm:nightly-2671fedfc7ae604761990603fc736c0c4f21de57}"
VLLM_PKG=/usr/local/lib/python3.12/dist-packages/vllm

case "$ARM" in
  1) TAG=abcompile_cfg_c1;   APPLY_DECORATOR=0; SCHED_PRS="" ;;
  2) TAG=abcompile_deco_c1;  APPLY_DECORATOR=1; SCHED_PRS="" ;;
  # Arm 3: the two upstream scheduler PRs, both off by default upstream, so the
  # arm is (patch) + (flags). Split from 1/2 because it targets a different
  # cost: prefill/decode interleaving and admission order, not graph fusion.
  3) TAG=absched_c1;         APPLY_DECORATOR=0; SCHED_PRS="54627 54625" ;;
  *) echo "ARM must be 1, 2 or 3" >&2; exit 2 ;;
esac

# --- preflight: the GPUs must be free, and they must be free of OTHER people's
# work. A run started on a busy box produces a number that is not comparable,
# which is worse than no number. This is exactly how the first segmented arm
# died.
used=$(rocm-smi --showmeminfo vram 2>/dev/null |
       awk '/Used Memory/{s+=$NF} END{printf "%.0f", s/1073741824}')
if [ "${used:-9999}" -gt 16 ]; then
    echo "REFUSING: GPUs hold ${used} GiB -- someone is using the box." >&2
    exit 1
fi

# --- the container survives reboots as Exited; start it if needed.
docker start "$CTR" >/dev/null 2>&1 || true

# --- ALWAYS reset vLLM to pristine first. The recipe patches in place, and an
# aborted run leaves a half-patched tree that fails on the next launch with
# "Hunk #N FAILED". /dev/shm is bind-mounted, so staging there needs no copy.
if [ ! -d /dev/shm/vllm_pristine ]; then
    docker create --name _pristine_tmp "$IMG" true >/dev/null
    docker cp _pristine_tmp:$VLLM_PKG /dev/shm/vllm_pristine >/dev/null
    docker rm _pristine_tmp >/dev/null
fi
docker exec "$CTR" bash -lc "rm -rf $VLLM_PKG && cp -a /dev/shm/vllm_pristine $VLLM_PKG"

# The recipe applies the whole k3-perf + k3-dcp8 stack itself; only the
# unmerged upstream decorator is ours to add, and only for arm 2.
if [ "$APPLY_DECORATOR" = 1 ]; then
    docker exec "$CTR" bash -lc \
      "patch -p1 -d $VLLM_PKG < /workspace/patches/k3-perf/vllm/dspark_torch_compile.patch"
fi

# --- Arm 3: fetch the two scheduler PRs as patches and apply them. They are not
# in any image, so they have to come from GitHub. Both features are off by
# default, so the flags below are what actually turns them on.
SCHED_ARGS=""
for pr in $SCHED_PRS; do
    raw=/dev/shm/_pr_${pr}.patch
    f=/dev/shm/_pr_${pr}_vllm.patch
    [ -s "$raw" ] || curl -fsSL "https://github.com/vllm-project/vllm/pull/${pr}.diff" -o "$raw"
    # A PR .diff carries tests/ too, but the installed package has no tests/
    # tree -- -p2 would strip a/tests/ to a path under vllm/ and fail with
    # "can't find file to patch". Keep only the vllm/ hunks.
    python3 /workspace/_filter_vllm_diff.py "$raw" "$f" >/dev/null
    docker cp "$f" "$CTR":/tmp/ >/dev/null
    docker exec "$CTR" bash -lc "cd $VLLM_PKG && patch -p2 --forward < /tmp/_pr_${pr}_vllm.patch" \
      || { echo "PR $pr did not apply -- it has drifted from the image; rebase or drop it" >&2; exit 1; }
done
if [ -n "$SCHED_PRS" ]; then
    # #54627: defer new prefills to every Nth step so decode runs in longer
    #   uninterrupted stretches. Silently a no-op before that PR: the base
    #   EngineCore._should_throttle_prefills() returns False unconditionally.
    # #54625: admit a request whose prefix is already resident ahead of a cold
    #   one, so the cold allocation cannot evict blocks the queued request needs.
    #   Requires fcfs + prefix caching; threshold gates it to when KV is tight.
    SCHED_ARGS="--prefill-schedule-interval ${PREFILL_SCHEDULE_INTERVAL:-4}"
    SCHED_ARGS="$SCHED_ARGS --cache-aware-admission-window ${CACHE_AWARE_WINDOW:-4}"
    SCHED_ARGS="$SCHED_ARGS --cache-aware-admission-threshold ${CACHE_AWARE_THRESHOLD:-0.9}"
fi

# --- LMCache's nightly-rocm index is NOT immutable and rotates its dev pins
# (dev89, dev105, dev134 are all gone). A stale pin aborts the run ~8 min in,
# AFTER patching, which is what creates the half-patched tree above.
export LMCACHE_VERSION="${LMCACHE_VERSION:-0.5.6.dev3+rocm7.2}"

# Arms 1/2 test graph fusion; arm 3 tests the scheduler. Keep them disjoint --
# an arm that moved both would measure neither.
if [ -n "$SCHED_PRS" ]; then
    COMPILE_JSON=""
else
    COMPILE_JSON=',\"use_inductor_graph_partition\":true'
fi

echo "=== ARM $ARM -> $TAG (decorator=$APPLY_DECORATOR, sched='$SCHED_ARGS') ==="
docker exec "$CTR" bash -lc "cd /workspace && \
  CONC=1 TAG=$TAG \
  LMCACHE_VERSION=$LMCACHE_VERSION \
  COMPILATION_EXTRA_JSON='$COMPILE_JSON' \
  EXTRA_VLLM_ARGS='--load-format auto $SCHED_ARGS' \
  setsid nohup bash benchmarks/single_node/agentic/repro/k3_mi355x_agentic_repro.sh \
  > /workspace/_${TAG}.log 2>&1 & echo launched"

cat <<EOF

Launched. ~10 min to serve up, then 3600 s of aiperf.

Watch:        tail -f _${TAG}.log
Did it take?  arms 1-2: grep -E 'rope_kvcache_cat_mla|graph partition' _${TAG}.log
              arm 3:    grep -E 'prefill.schedule.interval|cache.aware.admission' \
                          results_ixci/$TAG/vllm_command.txt
              ^ if 'rope_kvcache_cat_mla' is absent, the pass did NOT engage and
                the arm is void -- do not report its number.
Compare:      python3 _ab_route_report.py abroute_asm_c1 $TAG

Do NOT poll with 'pgrep -f <script>': the pattern matches pgrep's own cmdline,
so it reports RUNNING forever. Check the log mtime instead.
EOF
