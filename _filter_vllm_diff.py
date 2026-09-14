#!/usr/bin/env python3
"""Keep only the vllm/ hunks of a GitHub PR .diff.

The installed package has no tests/ tree, so a raw PR diff cannot be applied
with -p2: patch strips a/tests/ to v1/core/... and finds nothing.
"""
import sys
keep, out = False, []
for line in open(sys.argv[1]):
    if line.startswith("diff --git "):
        keep = " a/vllm/" in line
    if keep:
        out.append(line)
open(sys.argv[2], "w").writelines(out)
print(f"{sys.argv[2]}: {sum(1 for l in out if l[0] in '+-' and not l.startswith(('+++','---')))} lines kept")
