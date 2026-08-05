#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
NUM_SHARDS="${NUM_SHARDS:-4}"
python3 "$HERE/checkpoint.py" --here "$HERE" --num-shards "$NUM_SHARDS" compact "$@"
