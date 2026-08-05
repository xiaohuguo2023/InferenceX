#!/usr/bin/env python3
"""Pre-download the IX agentic cc-traces dataset into a persistent HF cache.

aiperf resolves the dataset during "Configure Profiling" and aborts the whole run
if that exceeds its internal timeout. Unauthenticated HF requests get 429'd with a
~292s backoff, which blows past that timeout, so a fresh node fails every conc in
the sweep. Fetching the repo once into a cache that outlives the node (NFS home
rather than /dev/shm) makes later sweeps resolve it locally.

Usage:
  HF_HUB_CACHE=/home/$USER/hf-cache python3 _prefetch_agentic_dataset.py
"""

import os
import sys
import time

from huggingface_hub import snapshot_download

REPO = os.environ.get("DATASET_REPO", "semianalysisai/cc-traces-weka-062126")
CACHE = os.environ.get("HF_HUB_CACHE") or os.environ.get("HF_HOME")
ATTEMPTS = int(os.environ.get("ATTEMPTS", "60"))


def main() -> int:
    if not CACHE:
        print("set HF_HUB_CACHE (or HF_HOME) to a persistent path", file=sys.stderr)
        return 2
    os.makedirs(CACHE, exist_ok=True)
    print(f"repo={REPO} cache={CACHE} attempts={ATTEMPTS}", flush=True)

    for i in range(1, ATTEMPTS + 1):
        try:
            path = snapshot_download(repo_id=REPO, repo_type="dataset", cache_dir=CACHE)
            print(f"OK {path}", flush=True)
            return 0
        except Exception as exc:  # noqa: BLE001 - retry on anything, incl. 429/timeouts
            # Unlike aiperf we have no deadline here, so keep waiting out the limit.
            wait = min(60 * i, 600)
            print(
                f"[{i}/{ATTEMPTS}] {type(exc).__name__}: {str(exc)[:180]} "
                f"-> retry in {wait}s",
                flush=True,
            )
            if i < ATTEMPTS:
                time.sleep(wait)
    print("FAILED: exhausted attempts", file=sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(main())
