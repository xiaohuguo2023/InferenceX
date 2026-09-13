#!/usr/bin/env python3
"""Compare IX agentic runs on the metric the IX table actually uses.

    python3 _ab_route_report.py <tag> [<tag> ...]

Reads results_ixci/<tag>/aiperf_artifacts/profile_export.jsonl.

WHY THIS EXISTS -- the ITL trap
-------------------------------
There are two different per-token latencies in an aiperf run and they differ by
~9%, which is larger than most effects being measured:

  * aiperf's reported ``inter_token_latency`` == (request_latency - TTFT)/(OSL-1)
  * the IX table's "frITL"                    == full_decode_duration / OSL

Verified on results_ixci/r72_c1_ep1, whose stored numbers are
``frITL p90 7.613 -> intvty p90 131.35``:

    full_decode_duration / OSL        p90 = 7.629  -> 131.07   <-- matches
    full_decode_duration / (OSL-1)    p90 = 7.641  -> 130.88
    (request_latency-TTFT)/(OSL-1)    p90 = 6.996  -> 142.93   <-- aiperf's ITL
    flattened inter_chunk_latency     p90 = 26.99  ->  37.06

So quoting aiperf's ``inter_token_latency`` against a stored IX number compares
two different quantities and flatters the new run by ~9%. Always use frITL.

Interactivity is 1/frITL with the percentile taken on the LATENCY before
inverting, so p90 < p50: p90 is the slow tail. Do NOT use aiperf's
``output_token_throughput_per_user`` p90, which percentiles the rate instead and
reports a number ~10x higher.
"""

import json
import sys

import numpy as np


def load(tag):
    path = f"results_ixci/{tag}/aiperf_artifacts/profile_export.jsonl"
    rows = []
    for line in open(path):
        if not line.strip():
            continue
        m = (json.loads(line).get("metrics") or {})
        if "request_latency" not in m:
            continue            # error records carry only error_isl
        rows.append({k: m[k]["value"] for k in m})
    return rows


def main(tags):
    hdr = (f"{'run':24s} {'n':>4s} {'frITL p50':>10s} {'frITL p90':>10s} "
           f"{'intvty p90':>11s} {'TTFT p50':>9s} {'ISL med':>9s} {'OSL med':>8s}")
    print(hdr)
    print("-" * len(hdr))
    base = None
    for tag in tags:
        r = load(tag)
        fr = np.array([x["full_decode_duration"] / x["output_sequence_length"]
                       for x in r if x["output_sequence_length"] > 0])
        ttft = np.array([x["time_to_first_token"] for x in r])
        isl = np.array([x["input_sequence_length"] for x in r])
        osl = np.array([x["output_sequence_length"] for x in r])
        p90 = np.percentile(fr, 90)
        intvty = 1000.0 / p90
        print(f"{tag:24s} {len(r):4d} {np.percentile(fr,50):10.3f} {p90:10.3f} "
              f"{intvty:11.2f} {np.percentile(ttft,50):9.1f} "
              f"{np.median(isl):9.0f} {np.median(osl):8.0f}")
        if base is None:
            base = (p90, intvty, len(r), np.median(isl), np.median(osl))
        else:
            d = (intvty / base[1] - 1) * 100
            # Workload identity: if the arms did not draw the same work, the
            # delta is an artifact of the trace, not of the change under test.
            same = (abs(len(r) - base[2]) <= 3
                    and abs(np.median(isl) / base[3] - 1) < 0.02
                    and abs(np.median(osl) / base[4] - 1) < 0.02)
            print(f"{'':24s} {'':4s} {'':10s} {'':10s} {d:+10.2f}%"
                  f"   workload match: {'YES' if same else 'NO -- delta is contaminated'}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(2)
    main(sys.argv[1:])
