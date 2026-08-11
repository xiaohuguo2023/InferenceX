#!/usr/bin/env python3
"""Per-concurrency DSpark acceptance from before/after /metrics snapshots.
Usage: python3 _extract_accept.py <ROOT>"""
import glob, os, re, sys

root = sys.argv[1] if len(sys.argv) > 1 else "/workspace/k3_dspark_longctx_bench_FULLFIX"

def parse(path):
    d = {}
    if not os.path.exists(path):
        return d
    for ln in open(path):
        if ln.startswith("#"):
            continue
        m = re.match(r"(vllm:spec_decode_[a-z_]+)(\{[^}]*\})?\s+([0-9.e+]+)", ln)
        if not m:
            continue
        name, labels, val = m.group(1), m.group(2) or "", float(m.group(3))
        pos = re.search(r'position="(\d+)"', labels)
        key = f"{name}#pos{pos.group(1)}" if pos else name
        d[key] = val
    return d

print(f"{'conc':>5} | {'drafts':>8} | {'draft_tok':>9} | {'accept_tok':>10} | {'AL(tok/step)':>12} | {'pos0%':>6} | {'pos1%':>6}")
print("-" * 78)
rows = []
for dd in glob.glob(os.path.join(root, "concurrency_*__requests_*")):
    m = re.search(r"concurrency_(\d+)__requests_(\d+)", dd)
    if not m:
        continue
    conc = int(m.group(1))
    b = parse(os.path.join(dd, "metrics_before.txt"))
    a = parse(os.path.join(dd, "metrics_after.txt"))
    if not a:
        continue
    def dl(k): return a.get(k, 0) - b.get(k, 0)
    drafts = dl("vllm:spec_decode_num_drafts_total")
    dtok = dl("vllm:spec_decode_num_draft_tokens_total")
    atok = dl("vllm:spec_decode_num_accepted_tokens_total")
    p0 = dl("vllm:spec_decode_num_accepted_tokens_per_pos_total#pos0")
    p1 = dl("vllm:spec_decode_num_accepted_tokens_per_pos_total#pos1")
    al = 1 + atok / drafts if drafts else 0
    pos0 = 100 * p0 / drafts if drafts else 0
    pos1 = 100 * p1 / drafts if drafts else 0
    rows.append((conc, drafts, dtok, atok, al, pos0, pos1))

for conc, drafts, dtok, atok, al, pos0, pos1 in sorted(rows):
    print(f"{conc:>5} | {drafts:8.0f} | {dtok:9.0f} | {atok:10.0f} | {al:12.3f} | {pos0:6.1f} | {pos1:6.1f}")
