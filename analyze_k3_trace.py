import gzip, json, sys, collections, re
f=sys.argv[1]
print(f"loading {f} ...", file=sys.stderr)
d=json.load(gzip.open(f))
ev=d["traceEvents"]
# GPU kernel events have cat 'kernel'
kern=[e for e in ev if e.get("cat")=="kernel" and "dur" in e]
tot=sum(e["dur"] for e in kern)
print(f"kernel events: {len(kern)}  total GPU kernel time: {tot/1e3:.1f} ms", file=sys.stderr)
by=collections.defaultdict(float); cnt=collections.Counter()
for e in kern:
    n=e["name"]
    by[n]+=e["dur"]; cnt[n]+=1
print(f"\n{'us_total':>12} {'%':>6} {'count':>7}  kernel")
for n,t in sorted(by.items(),key=lambda x:-x[1])[:45]:
    print(f"{t:12.0f} {100*t/tot:6.2f} {cnt[n]:7d}  {n[:110]}")
