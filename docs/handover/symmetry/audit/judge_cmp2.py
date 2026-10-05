import sys, pickle, numpy as np
A = pickle.load(open(sys.argv[1], "rb")); Bm = pickle.load(open(sys.argv[2], "rb"))
ok_ev = 0; bad = []
for ra, rb in zip(A, Bm):
    for t, (x, y) in enumerate(zip(ra, rb)):
        dob = np.abs(x[0] - y[0]); dob[624:630] = 0; dob = dob.max()   # ignore tip-vel lanes (finite-diff amplified)
        if dob > 1e-3: break
        if x[3] != y[3] or abs(x[1] - y[1]) > 1e-6:
            bad.append((t, dob, x[3], y[3], x[1], y[1])); break
        ok_ev += len(x[3])
print(f"events matched while state agreed (obs dev<1e-3): {ok_ev}; judge mismatches while state agreed: {len(bad)}")
for b in bad[:5]: print("  ", b)
