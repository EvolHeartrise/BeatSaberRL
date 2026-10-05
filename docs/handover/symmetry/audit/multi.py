import sys, os, numpy as np
mod = os.environ.get("MOD")
if mod: sys.path.insert(0, mod)
import beat_arena_rs as B
from mirror_ops import *
FAR = [(200.0, 0.3, 1.4, 1, 3*np.pi/2, 18.0, False, 0, False, False)]
P = B.RawBeatSim(max_steps=100000, gears=5); Q = B.RawBeatSim(max_steps=100000, gears=5)
q = np.zeros(NQ); q[[3,10]] = -0.7; q[6] = q[13] = 1.0; q[7] = 1.5; q[14] = -1.5; q[17]=q[23]=0.5; q[20]=q[26]=0.6
qp = np.concatenate([[0, 3.0, -0.65], [1, 0, 0, 0], q])
def run(joints, g=1):
    a = np.zeros(NQ, np.uint8); a[joints] = g
    P.set_chart(FAR, []); Q.set_chart([m_note(t) for t in FAR], [])
    P.set_joints(list(qp), [0.0]*35); Q.set_joints(list(m_qpos(qp)), [0.0]*35)
    P.step(list(a)); Q.step(list(m_action(a)))
    vp = np.asarray(list(P.twist()) + list(P.omega())); vq = np.asarray(list(Q.twist()) + list(Q.omega()))
    w = np.abs(np.asarray(P.omega()))
    return np.abs(vq - m_qvel(vp)).max(), w
for js in ([3, 10], [8], [8, 15], [7, 14], [7, 8], [8, 9], [7, 8, 9], [7, 8, 9, 14, 15, 16], [3,4,5,6], [3,4,5,6,10,11,12,13]):
    for g in (1, 3):
        e, w = run(js, g)
        print(js, g, f"defect {e:.2e}   max |omega| L-wrist {w[7:10].max():.1f} R-wrist {w[14:17].max():.1f}")
