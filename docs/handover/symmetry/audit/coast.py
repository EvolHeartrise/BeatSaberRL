import sys, os, numpy as np
mod = os.environ.get("MOD")
if mod: sys.path.insert(0, mod)
import beat_arena_rs as B
print("module:", B.__file__)
from mirror_ops import *
FAR = [(200.0, 0.3, 1.4, 1, 3*np.pi/2, 18.0, False, 0, False, False)]
def pair(g=5):
    A = B.RawBeatSim(max_steps=100000, gears=g); Bm = B.RawBeatSim(max_steps=100000, gears=g)
    A.set_chart(FAR, []); Bm.set_chart([m_note(t) for t in FAR], [])
    return A, Bm
def dev(A, Bm):
    dq = np.abs(np.asarray(Bm.qpos()) - m_qpos(A.qpos()))
    dv = np.abs(np.asarray(list(Bm.twist()) + list(Bm.omega())) - m_qvel(list(A.twist()) + list(A.omega())))
    return dq.max(), dv.max()
# 1) coast from symmetric start: self-asymmetry of ONE sim (should be 0 for a mirror-exact env)
A, Bm = pair()
z = [0]*NQ
for t in range(1, 21):
    A.step(z); Bm.step(z)
    if t in (1, 2, 5, 10, 20):
        q = np.asarray(A.qpos())
        print(f" coast t={t:2d}  pair dev dq={dev(A,Bm)[0]:.2e}  root_x={q[0]:+.3e}  ankle_roll L+R={q[7+22]+q[7+28]:+.3e}  hip_roll L+R={q[7+19]+q[7+25]:+.3e}")
# 2) 'stand' policy proxy: a symmetric PD-ish hold using gears on legs/spine toward start pose
def hold_action(sim, g=5):
    q = np.asarray(sim.qpos())[7:]; w = np.asarray(sim.omega())
    tgt = np.zeros(NQ); tgt[[3,10]] = -1.396; tgt[[6,13]] = 1.745; tgt[[8,15]] = 1.2217
    tgt[19] = -0.1396; tgt[22] = 0.1396; tgt[25] = 0.1396; tgt[28] = -0.1396
    u = -(q - tgt) * 8.0 - 0.8 * w
    a = np.zeros(NQ, np.uint8)
    a[u > 0.15] = 3; a[u < -0.15] = 4; a[u > 1.0] = 1; a[u < -1.0] = 2
    return a
A, Bm = pair()
for t in range(1, 301):
    a = hold_action(A); b = hold_action(Bm)
    A.step(list(a)); _, _, d, _ = Bm.step(list(b))
    if t in (1, 10, 30, 100, 300):
        q = np.asarray(A.qpos()); qb = np.asarray(Bm.qpos())
        print(f" hold t={t:3d}  pair dev dq={dev(A,Bm)[0]:.2e}  rootx A={q[0]:+.3e} B={qb[0]:+.3e} yawz A={q[6]:+.3e} B={qb[6]:+.3e} doneB={d}")
