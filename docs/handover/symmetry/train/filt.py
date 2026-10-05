import numpy as np
rng=np.random.default_rng(0)
n=2_000_000
for dist in ("gauss","laplace"):
  for r in (1.05,1.1,1.2,1.5):
    f=(lambda s,m: rng.normal(0,s,m)) if dist=="gauss" else (lambda s,m: rng.laplace(0,s,m))
    # half the rows are "left-hand-event" rows, half right; right has sigma r
    L=f(1.0,n); R=f(r,n)
    a=np.concatenate([L,R]); c=np.quantile(np.abs(a),0.5)
    kl=(np.abs(L)>c).mean(); kr=(np.abs(R)>c).mean()
    # PG weight ~ sum |A| over kept rows (normalized by full-batch sd)
    wl=np.abs(L)[np.abs(L)>c].sum(); wr=np.abs(R)[np.abs(R)>c].sum()
    wl0=np.abs(L).sum(); wr0=np.abs(R).sum()
    print(dist,r,"kept L %.3f R %.3f"%(kl,kr),"grad-weight ratio R/L filt %.3f unfilt %.3f"%(wr/wl,wr0/wl0))
