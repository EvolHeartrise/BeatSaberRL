import sys, numpy as np
from test_mirror import *
np.set_printoptions(precision=3, linewidth=200, suppress=False)
def one(act, n=1, label=""):
    A = B.RawBeatSim(max_steps=10000, gears=5); Bm = B.RawBeatSim(max_steps=10000, gears=5)
    A.set_chart([], []); Bm.set_chart([], [])
    for _ in range(n):
        A.step(list(act)); Bm.step(list(m_action(act)))
    qa = m_qpos(A.qpos()); qb = np.asarray(Bm.qpos())
    va = m_qvel(list(A.twist()) + list(A.omega())); vb = np.asarray(list(Bm.twist()) + list(Bm.omega()))
    d = np.abs(qb - qa); dv = np.abs(vb - va)
    print(f"{label:28s} n={n} max dq={d.max():.2e} (idx {d.argmax()}) max dv={dv.max():.2e} (idx {dv.argmax()})")
    return d, dv
z = np.zeros(NQ, np.uint8)
for n in (1, 5, 30, 90):
    one(z, n, "coast")
for j in range(NQ):
    for g in (1, 2):
        a = z.copy(); a[j] = g
        d, dv = one(a, 3, f"joint {j} gear {g}")
