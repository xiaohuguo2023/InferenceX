#!/bin/bash
# Drive _dcp_phase_bisect.py one phase at a time under an external timeout.
#
# The timeout is the whole point of running it from a script: a wedged DCP phase
# leaves 8 ranks parked with GPU queues held, and the only thing that reliably
# ends that is a signal from outside the process group. TERM first so ranks get
# a chance to unwind, then KILL -- and never a GPU-side trap, which faults the
# queue and poisons the box driver-wide.
#
#   bash _dcp_phase_bisect.sh                 # groups only (default)
#   bash _dcp_phase_bisect.sh groups symm combine
#   PHASE_TIMEOUT_S=600 bash _dcp_phase_bisect.sh groups
#
# Each phase is a FRESH 8-rank job. That is deliberate: it keeps a phase's
# failure from being contaminated by state the previous phase left behind, and
# it means the first phase that fails is the one to fix.
#
# SECOND MODE -- the leak matrix:
#
#   bash _dcp_phase_bisect.sh leak                    # all arms, clean + killed
#   ARMS="nccl symmonly" REPS=5 bash _dcp_phase_bisect.sh leak
#
# That answers a different question from the phases: not "does the chain work"
# but "does running it leave residue in the kernel driver". Each arm runs twice
# -- exiting cleanly, and SIGKILLed inside its --hold window -- and the residue
# (stale kfd_process nodes, dma-bufs, evicted_ms) is differenced across the pair
# and reported as a slope over REPS. A vs B is the decisive comparison: if a
# killed `nccl` leaves nothing and a killed `symmonly` leaves stale KFD nodes,
# the leak is torch symmetric memory's cross-process dma-buf mesh, with no
# model, no MLA and no server anywhere near it.
#
# Run the leak mode from a PRIVILEGED exec -- it wants dmesg and debugfs:
#   docker exec --privileged <ctr> bash -lc 'cd /workspace/k3_dcp_direct_hip && \
#       bash _dcp_phase_bisect.sh leak'
set -uo pipefail

cd "$(dirname "$(readlink -f "$0")")"

PHASES=("$@")
[ ${#PHASES[@]} -eq 0 ] && PHASES=(groups)
TMO="${PHASE_TIMEOUT_S:-300}"
NPROC="${NPROC:-8}"

# ------------------------------------------------------------------ leak mode

ARMS="${ARMS:-nccl symmonly probe dcp dcpstall}"
REPS="${REPS:-3}"
ARM_ITERS="${ARM_ITERS:-200}"
HOLD="${HOLD:-25}"          # kill window the ranks park in
KILL_AT="${KILL_AT:-8}"     # seconds into that window before SIGKILL
COOLDOWN="${COOLDOWN:-20}"  # let the driver's restore worker settle between runs
DMESG_LINES="${DMESG_LINES:-4000}"

# Driver state without a GPU job: the module's RANK/WORLD default to 0/8, so it
# imports fine outside torchrun.
driver_state() {
  python3 -c 'import _dcp_phase_bisect as b; print(b.fmt_state(b.driver_state()))' 2>/dev/null \
    || echo "(driver_state unavailable)"
}

# dmesg is the only place amdgpu_bo_release_notify, VM faults and queue-preemption
# timeouts appear, and it was never captured on 08-21 -- the biggest tooling miss.
dmesg_snap() { dmesg -T 2>/dev/null | tail -n "$DMESG_LINES" > "$1"; }
dmesg_delta() {
  if ! diff -u "$1" "$2" > /dev/null 2>&1; then
    diff --new-line-format='%L' --old-line-format='' --unchanged-line-format='' \
      "$1" "$2" 2>/dev/null | grep -aE 'amdgpu|kfd|dma|fault|timeout|evict' | tail -25
  fi
}

# One arm, one disposition. Clean = the process group is destroyed and the ranks
# exit; killed = SIGKILL lands while they are parked in the hold window, which is
# the case the leak claim is actually about.
run_one() {
  local arm="$1" disp="$2" rep="$3"
  local tag="${arm}_${disp}_r${rep}"
  local out="_leak_${tag}.log"
  local pre_dm="/tmp/_leak_dm_pre_$$" post_dm="/tmp/_leak_dm_post_$$"

  dmesg_snap "$pre_dm"
  printf '  %-9s %-6s rep%-2s PRE  %s\n' "$arm" "$disp" "$rep" "$(driver_state)"

  local extra=()
  [ "$disp" = "clean" ] && extra+=(--clean-exit)

  # setsid gives the job its own process group so SIGKILL reaches every rank and
  # not this shell. Never `pkill -f` here -- the pattern matches our own cmdline.
  # kill-after MUST be generous for DCP. Teardown is a synchronize -> barrier ->
  # free across 8 ranks and then the KFD restore work has to drain; 30s
  # guaranteed a SIGKILL mid-teardown, which is precisely what strands dma-bufs
  # at refcount 7. Note the killed arm below SIGKILLs deliberately -- that is the
  # experiment; this is about not killing the arms that were meant to exit clean.
  setsid timeout --signal=TERM --kill-after=600s "${TMO}s" \
    torchrun --standalone --nproc-per-node="$NPROC" \
      _dcp_phase_bisect.py --arm "$arm" --arm-iters "$ARM_ITERS" \
        --hold "$HOLD" "${extra[@]}" \
    > "$out" 2>&1 &
  local pid=$! pgid
  pgid=$(ps -o pgid= "$pid" 2>/dev/null | tr -d ' ')

  if [ "$disp" = "killed" ]; then
    # Wait for the ranks to reach the hold window, then kill them inside it.
    local waited=0
    while [ "$waited" -lt "$TMO" ]; do
      grep -qa 'HOLDING' "$out" && break
      kill -0 "$pid" 2>/dev/null || break
      sleep 1; waited=$((waited + 1))
    done
    if grep -qa 'HOLDING' "$out"; then
      sleep "$KILL_AT"
      [ -n "$pgid" ] && kill -9 "-$pgid" 2>/dev/null
      echo "  ${arm} killed: SIGKILL sent to pgid ${pgid} inside the hold window"
    else
      echo "  ${arm} killed: never reached HOLDING -- arm did not run (see $out)"
    fi
  fi
  wait "$pid" 2>/dev/null
  local rc=$?

  grep -aE '^\[r0\]|ABORT|WITHHOLDING|Error|Traceback' "$out" | tail -12
  sleep "$COOLDOWN"
  dmesg_snap "$post_dm"
  printf '  %-9s %-6s rep%-2s POST %s  rc=%s\n' "$arm" "$disp" "$rep" "$(driver_state)" "$rc"
  local dm; dm=$(dmesg_delta "$pre_dm" "$post_dm")
  [ -n "$dm" ] && { echo "  --- dmesg delta ---"; echo "$dm" | sed 's/^/  /'; }
  rm -f "$pre_dm" "$post_dm"
}

leak_mode() {
  echo "=== DCP leak matrix: arms=[$ARMS] reps=$REPS iters=$ARM_ITERS ==="
  echo "    Residue is read as a SLOPE across reps, not from one reading."

  if [ ! -d /sys/kernel/debug/dma_buf ]; then
    mount -t debugfs none /sys/kernel/debug 2>/dev/null \
      && echo "    debugfs mounted (dma_buf/bufinfo now readable)" \
      || echo "    WARNING: no debugfs -- dmabuf counts will be self-fd only." \
              "Re-run from a privileged exec for the global view."
  fi
  dmesg -T >/dev/null 2>&1 || echo "    WARNING: dmesg unreadable -- run privileged."

  # Refuse to measure on a poisoned box: a slope means nothing from a dirty start.
  echo "    baseline: $(driver_state)"
  if ! python3 -c '
import sys, _dcp_phase_bisect as b
s = b.driver_state()
if s["kfd_stale"] is None:
    print("    WARNING: nested pid namespace -- kfd_stale unmeasurable."
          " Run this container with --pid=host, or drive it from the host,"
          " or the decisive A-vs-B signal is lost.")
bad = s["evicted_ms"] or s["stat_dirs"] > 8 or (s["kfd_stale"] or 0)
sys.exit(1 if bad else 0)'; then
    echo ">>> ABORT: box is not clean. Only a reboot clears this state, and a"
    echo "    reboot is not available here -- wait, or measure nothing."
    return 2
  fi

  for arm in $ARMS; do
    for rep in $(seq 1 "$REPS"); do
      echo "############ arm=$arm rep=$rep ############"
      run_one "$arm" clean  "$rep"
      run_one "$arm" killed "$rep"
    done
  done

  echo
  echo "=== leak matrix done.  final: $(driver_state) ==="
  echo "    Compare killed-nccl vs killed-symmonly: that difference IS the result."
  return 0
}

if [ "${PHASES[0]}" = "leak" ]; then
  leak_mode
  exit $?
fi

echo "=== DCP phase bisect: ${PHASES[*]} (timeout ${TMO}s each, ${NPROC} ranks) ==="
echo

rc_all=0
# Pass-through for _dcp_phase_bisect.py flags, e.g. BISECT_ARGS="--max-nt 64".
# The workspace is sized at the `symm` phase but the shape that has to fit is
# args.reqs*qlen, decided later in `fold` -- so the graph phase's full
# producer+combine chain silently skips unless max_nt is raised from here.
read -r -a EXTRA_ARGS <<< "${BISECT_ARGS:-}"
for ph in "${PHASES[@]}"; do
  echo "############ phase: $ph ############"
  out="_dcp_phase_${ph}.log"
  # 600s, not 30s: see run_one() above. A phase that wedges is exactly the case
  # where the SIGKILL would land mid-teardown and strand the symm_mem mesh, i.e.
  # the debugging run would damage the box it is debugging.
  timeout --signal=TERM --kill-after=600s "${TMO}s" \
    torchrun --standalone --nproc-per-node="$NPROC" \
      _dcp_phase_bisect.py --stop-after "$ph" "${EXTRA_ARGS[@]}" \
    > "$out" 2>&1
  rc=$?

  # Rank 0's stamps plus anything that named a problem. Full log stays in $out.
  grep -aE "^\[r0\]|^\[r0 |PASS|STOP|SLOW|Error|Traceback|not implemented" "$out" | tail -40

  if [ "$rc" -eq 124 ] || [ "$rc" -eq 137 ]; then
    echo
    echo ">>> WEDGED in phase '$ph' (no completion in ${TMO}s)."
    echo "    This is the phase to fix. Last per-rank progress:"
    # Which ranks got how far matters more than the tail: a phase where 7 ranks
    # passed a collective and 1 did not is a different bug from one where none did.
    for r in $(seq 0 $((NPROC - 1))); do
      printf '      r%s: %s\n' "$r" "$(grep -a "^\[r$r " "$out" | tail -1)"
    done
    rc_all=1
    break
  elif [ "$rc" -ne 0 ]; then
    echo
    echo ">>> phase '$ph' FAILED rc=$rc (not a timeout -- see $out)"
    rc_all=1
    break
  fi
  echo ">>> phase '$ph' OK"
  echo
done

echo
if [ "$rc_all" -eq 0 ]; then
  echo "ALL REQUESTED PHASES PASSED: ${PHASES[*]}"
else
  echo "BISECT STOPPED -- see the phase marked above."
fi
exit "$rc_all"
