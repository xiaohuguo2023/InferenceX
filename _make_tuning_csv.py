import csv
from collections import defaultdict
obs=defaultdict(set); meta={}
with open("kimik3_bf16_untuned_gemm.csv") as f:
    for d in csv.DictReader(f):
        N,K=int(d["N"]),int(d["K"]); obs[(N,K)].add(int(d["M"]))
        meta[(N,K)]=(d["bias"],d["dtype"],d["outdtype"],d["scaleAB"],d["bpreshuffle"])
LADDER=[1,2,4,8,16,24,32,40,48,56,64,96,128,256,512,1024,2048,4096,8192]
BIG=[16384,24576,32768,49152,65536,98304]
COLS=["M","N","K","bias","dtype","outdtype","scaleAB","bpreshuffle"]
rows=[]; nflydsl=0
for (N,K) in sorted(obs):
    mx=max(obs[(N,K)])
    Ms=set(m for m in LADDER if m<=max(mx,64)) | {m for m in BIG if m<=mx}
    if mx>8192: Ms.add(mx)
    b,dt,ot,sab,bp=meta[(N,K)]
    for M in sorted(Ms): rows.append((M,N,K,b,dt,ot,sab,bp))
    if N%64==0 and K%64==0: nflydsl+=1
with open("kimik3_bf16_tuning_gemm.csv","w",newline="") as f:
    w=csv.writer(f); w.writerow(COLS); [w.writerow(r) for r in rows]
print(f"ALL (N,K): {len(obs)} ({nflydsl} flydsl-eligible + {len(obs)-nflydsl} hipBLASLt/asm-only)")
print(f"-> {len(rows)} tuning rows -> kimik3_bf16_tuning_gemm.csv")
print("tuner will pick per (M,N,K): flydsl (small-M, eligible) / hipBLASLt / asm / skinny")
