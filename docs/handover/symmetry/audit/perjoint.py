import sys, os, numpy as np
import beat_arena_rs as B
from mirror_ops import *
FAR = [(200.0, 0.3, 1.4, 1, 3*np.pi/2, 18.0, False, 0, False, False)]
P = B.RawBeatSim(max_steps=100000, gears=5); Q = B.RawBeatSim(max_steps=100000, gears=5)
q = np.zeros(NQ); q[[3,10]] = -0.7; q[6] = q[13] = 1.0; q[7] = 1.5; q[14] = -1.5; q[17]=q[23]=0.5; q[20]=q[26]=0.6
qp = np.concatenate([[0, 3.0, -0.65], [1, 0, 0, 0], q])
for j in range(NQ):
    out = []
    for g in (1, 2, 3, 4):
        a = np.zeros(NQ, np.uint8); a[j] = g
        P.set_chart(FAR, []); Q.set_chart([m_note(t) for t in FAR], [])
        P.set_joints(list(qp), [0.0]*35); Q.set_joints(list(m_qpos(qp)), [0.0]*35)
        P.step(list(a)); Q.step(list(m_action(a)))
        vp = np.asarray(list(P.twist()) + list(P.omega())); vq = np.asarray(list(Q.twist()) + list(Q.omega()))
        out.append(np.abs(vq - m_qvel(vp)).max())
    print(j, " ".join(f"{e:.1e}" for e in out))
