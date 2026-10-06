#!/bin/bash
# srun against spur can fail with "CreateJobStep failed: raft propose failed";
# retry the step instead of silently losing a launch.
retry() {
  local i
  for i in 1 2 3 4 5 6; do
    "$@" && return 0
    echo "[$(date +%T)] step failed (attempt $i), retrying: $*" >&2
    sleep 15
  done
  return 1
}
