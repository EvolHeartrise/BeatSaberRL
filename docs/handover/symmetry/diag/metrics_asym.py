import json,glob,sys,math
import numpy as np
pairs=[('accL','accR'),('wrev_L','wrev_R'),('flail_L','flail_R'),('lifts_L','lifts_R'),('ent_armL','ent_armR'),('wreset_L','wreset_R')]
files=sorted(glob.glob('*/runs/*/metrics.jsonl'))
for f in files:
    rows=[]
    for ln in open(f):
        try: rows.append(json.loads(ln))
        except: pass
    if not rows or 'accL' not in rows[-1]: continue
    n=len(rows)
    def seg(a,b,k):
        v=[r.get(k) for r in rows[a:b]]
        v=[x for x in v if isinstance(x,(int,float)) and x==x]
        return np.mean(v) if v else float('nan')
    out=[f.split('/runs/')[1].replace('/metrics.jsonl',''),str(n), str(rows[0].get('upd'))]
    # windows: first 2%, 10%, 50%, last 10%
    wins=[(0,max(5,n//50)),(n//10-n//50,n//10),(n//2-n//20,n//2),(n-n//10,n)]
    for (L,R) in pairs[:1]:
        s=[]
        for a,b in wins:
            l,r=seg(a,b,L),seg(a,b,R)
            s.append(f"{(r-l)*100:+.2f}")
        out.append('dAcc(R-L)pp '+' '.join(s)+f" | last L {seg(n-n//10,n,L):.4f} R {seg(n-n//10,n,R):.4f}")
    for (L,R) in pairs[1:]:
        if L not in rows[-1]: continue
        a,b=wins[-1]; l,r=seg(a,b,L),seg(a,b,R)
        a0,b0=wins[0]; l0,r0=seg(a0,b0,L),seg(a0,b0,R)
        out.append(f"{L[:-1]} R/L {r0/l0 if l0 else float('nan'):.3f}->{r/l if l else float('nan'):.3f}")
    print(' | '.join(out))
