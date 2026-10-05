"""One-tick mirror defect over a distribution of plausible states."""
import sys, os, numpy as np
mod = os.environ.get("MOD")
if mod: sys.path.insert(0, mod)
import beat_arena_rs as B
from mirror_ops import *
FAR = [(200.0, 0.3, 1.4, 1, 3*np.pi/2, 18.0, False, 0, False, False)]
def sim():
    s = B.RawBeatSim(max_steps=100000, gears=5); s.set_chart(FAR, []); return s
def hold(sim_, rng):
    q = np.asarray(sim_.qpos())[7:]; w = np.asarray(sim_.omega())
    tgt = np.zeros(NQ); tgt[19] = -0.1396; tgt[22] = 0.1396; tgt[25] = 0.1396; tgt[28] = -0.1396
    u = -(q - tgt) * 3.0 - 0.3 * w
    a = np.zeros(NQ, np.uint8); a[u > 0.3] = 3; a[u < -0.3] = 4
    a[3:17] = np.where(rng.random(14) < 0.5, rng.integers(1, 5, 14), 0)   # random arm gears
    return a
rng = np.random.default_rng(int(os.environ.get("SEED", 0)))
D = []; BIAS = []
A = sim(); P = sim(); Q = sim()
n_ok = 0
for ep in range(40):
    A.set_chart(FAR, [])
    for k in range(25):
        a = hold(A, rng)
        qp = np.asarray(A.qpos()); v = np.asarray(list(A.twist()) + list(A.omega()))
        if qp[1] < 0.6: break
        P.set_chart(FAR, []); Q.set_chart(FAR, [])
        P.set_joints(list(qp), list(v)); Q.set_joints(list(m_qpos(qp)), list(m_qvel(v)))
        P.step(list(a)); Q.step(list(m_action(a)))
        vp = np.asarray(list(P.twist()) + list(P.omega())); vq = np.asarray(list(Q.twist()) + list(Q.omega()))
        D.append(vq - m_qvel(vp))
        _, _, done, _ = A.step(list(a))
        if done: break
D = np.array(D)
names = ["vx","vy","vz","wx","wy","wz"] + [f"j{j}" for j in range(NQ)]
rms = np.sqrt((D**2).mean(0)); mean = D.mean(0); se = D.std(0) / np.sqrt(len(D))
print(f"samples={len(D)}  overall RMS defect (rad/s, m/s) = {np.sqrt((D**2).mean()):.3e}  max={np.abs(D).max():.3e}")
for i in np.argsort(-rms)[:8]:
    print(f"   {names[i]:4s} rms={rms[i]:.3e} mean={mean[i]:+.3e} (se {se[i]:.1e})")
for i in [0, 4, 5]:
    print(f"   world {names[i]} mean={mean[i]:+.3e} se={se[i]:.1e}")
arm_rms = np.sqrt((D[:, 6+3:6+17]**2).mean()); leg_rms = np.sqrt((D[:, 6+17:]**2).mean())
print(f"   arm-joint RMS {arm_rms:.3e}   leg-joint RMS {leg_rms:.3e}")
