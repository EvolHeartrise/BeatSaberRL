import sys, pickle, numpy as np
A = pickle.load(open(sys.argv[1], "rb")); Bm = pickle.load(open(sys.argv[2], "rb"))
n_exact = 0; tot_ev = 0; firsts = []; evmis = 0; obs_dev_first = []
for ra, rb in zip(A, Bm):
    first = None
    for t, (x, y) in enumerate(zip(ra, rb)):
        dobs = np.abs(x[0] - y[0]).max()
        if dobs > 1e-4 or abs(x[1] - y[1]) > 1e-6 or x[3] != y[3]:
            first = t; break
        tot_ev += len(x[3])
    firsts.append(first if first is not None else len(ra))
    if first is None and len(ra) == len(rb): n_exact += 1
print(f"{sys.argv[2]}: episodes fully mirror-identical {n_exact}/{len(A)}; ticks until first divergence (obs>1e-4 / reward / events): median {np.median(firsts)}, min {min(firsts)}; events matched before divergence {tot_ev}")
