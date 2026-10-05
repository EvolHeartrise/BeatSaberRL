import sys, pickle, numpy as np
A = pickle.load(open(sys.argv[1], "rb")); Bm = pickle.load(open(sys.argv[2], "rb"))
from collections import Counter
c = Counter(); mags = []
for ra, rb in zip(A, Bm):
    for t, (x, y) in enumerate(zip(ra, rb)):
        d = np.abs(x[0] - y[0])
        if d.max() > 1e-4 or abs(x[1]-y[1]) > 1e-6 or x[3] != y[3]:
            i = int(d.argmax()); c[(i, "rew" if abs(x[1]-y[1])>1e-6 else "", "ev" if x[3]!=y[3] else "")] += 1
            mags.append((t, i, d.max(), x[1], y[1], x[3], y[3])); break
print(c.most_common(12))
for m in mags[:12]: print(m[:5], m[5][:3], m[6][:3])
