import numpy as np
from test_mirror import *
np.set_printoptions(precision=4, linewidth=220, suppress=True)
FAR = [(200.0, 0.3, 1.4, 1, 3*np.pi/2, 18.0, False, 0, False, False)]
def pair():
    A = B.RawBeatSim(max_steps=100000, gears=5); Bm = B.RawBeatSim(max_steps=100000, gears=5)
    A.set_chart(FAR, []); Bm.set_chart([m_note(t) for t in FAR], [])
    return A, Bm
def dev(A, Bm):
    dq = np.abs(np.asarray(Bm.qpos()) - m_qpos(A.qpos()))
    dv = np.abs(np.asarray(list(Bm.twist()) + list(Bm.omega())) - m_qvel(list(A.twist()) + list(A.omega())))
    return dq, dv
A, Bm = pair()
z = [0]*NQ
print("coast:")
for t in range(1, 301):
    oa, ra, da, _ = A.step(z); ob, rb, db, _ = Bm.step(z)
    if t in (1, 2, 3, 5, 10, 30, 100, 300):
        dq, dv = dev(A, Bm)
        print(f" t={t:3d} max|dq|={dq.max():.2e} at {dq.argmax()}  max|dv|={dv.max():.2e}  rootx A={A.qpos()[0]:+.2e} B={Bm.qpos()[0]:+.2e} done={da},{db} "
              f"obs dev={np.abs(np.asarray(ob)-m_obs(np.asarray(oa))).max():.2e}")
# self-symmetric check: does a single sim drift sideways from a symmetric state with symmetric (coast) actions?
q = np.asarray(A.qpos()); print("A ankle rolls (L, -R):", q[7+22], -q[7+28], " hip rolls (L,-R):", q[7+19], -q[7+25], " root x:", q[0], " yaw(quat z):", q[6])
