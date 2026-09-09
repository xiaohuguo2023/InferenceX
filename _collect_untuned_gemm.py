import re, sys, glob, csv, os
pat = re.compile(
    r"shape is M:(\d+), N:(\d+), K:(\d+) dtype='([^']+)' otype='([^']+)' "
    r"bias=(\w+), scaleAB=(\w+), bpreshuffle=(\w+).*not found tuned config in "
    r"/tmp/aiter_configs/bf16_tuned_gemm\.csv")
COLS=["M","N","K","bias","dtype","outdtype","scaleAB","bpreshuffle"]
shapes=set()          # full (M,N,K,bias,dtype,outdtype,scaleAB,bpreshuffle)
nk=dict()             # (N,K) -> [set(M), count, dtype]
total=0
# read all log lines from stdin
for line in sys.stdin:
    m=pat.search(line)
    if not m: continue
    total+=1
    M,N,K,dt,ot,bias,sab,bp = m.groups()
    row=(int(M),int(N),int(K),bias,dt,ot,sab,bp)
    shapes.add(row)
    key=(int(N),int(K))
    e=nk.setdefault(key,[set(),0])
    e[0].add(int(M)); e[1]+=1

# merge existing csv
existing="kimik3_bf16_untuned_gemm.csv"
pre=0
if os.path.exists(existing):
    with open(existing) as f:
        r=csv.DictReader(f)
        for d in r:
            try:
                row=(int(d["M"]),int(d["N"]),int(d["K"]),d["bias"],d["dtype"],d["outdtype"],d["scaleAB"],d["bpreshuffle"])
                if row not in shapes: pre+=1
                shapes.add(row)
                key=(row[1],row[2]); e=nk.setdefault(key,[set(),0]); e[0].add(row[0])
            except Exception: pass

# write merged csv
out="kimik3_bf16_untuned_gemm.csv"
rows=sorted(shapes, key=lambda r:(r[1],r[2],r[0]))
with open(out,"w",newline="") as f:
    w=csv.writer(f); w.writerow(COLS)
    for r in rows: w.writerow(r)

print(f"parsed {total} warning lines; new-from-existing merged: {pre}")
print(f"distinct (M,N,K,...) shapes: {len(shapes)} -> wrote {out}")
print(f"distinct (N,K) GEMMs: {len(nk)}")
print("\n(N,K) summary  [flydsl-eligible = N%64==0 and K%64==0]:")
print(f"{'N':>7} {'K':>7} {'#M':>4} {'Mmin':>5} {'Mmax':>6}  flydsl  occ")
for (N,K),(Ms,cnt) in sorted(nk.items(), key=lambda kv:(-kv[1][1])):
    fly = "YES" if (N%64==0 and K%64==0) else f"no(N%64={N%64},K%64={K%64})"
    print(f"{N:>7} {K:>7} {len(Ms):>4} {min(Ms):>5} {max(Ms):>6}  {fly:<6} {cnt}")
