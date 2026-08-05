#!/usr/bin/env bash
# Print GEMM tuning checkpoint status (tuned vs remaining per shard).
#
# Usage:
#   ./checkpoint_status.sh
#   NUM_SHARDS=4 ./checkpoint_status.sh
#   ./checkpoint_compact.sh          # dedupe output CSVs before restart
#   ./checkpoint_compact.sh --dry-run
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
NUM_SHARDS="${NUM_SHARDS:-4}"
python3 "$HERE/checkpoint.py" --here "$HERE" --num-shards "$NUM_SHARDS" status "$@"
