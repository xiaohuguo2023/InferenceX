import json,sys,statistics
def P(v,p):
    v=sorted(v); k=(len(v)-1)*p/100; f=int(k); c=min(f+1,len(v)-1); return v[f]+(v[c]-v[f])*(k-f)
for d in sys.argv[1:]:
    fr=[]
    for line in open(f"{d}/aiperf_artifacts/profile_export.jsonl"):
        r=json.loads(line)
        if r["metadata"].get("benchmark_phase")!="profiling" or r.get("error"): continue
        m=r["metrics"]
        if "full_decode_duration" not in m: continue
        fr.append(m["full_decode_duration"]["value"]/m["output_sequence_length"]["value"])
    
    print(f"{d}: n={len(fr)} IX p90={1000/P(fr,90):.2f} p50={1000/P(fr,50):.2f}")
