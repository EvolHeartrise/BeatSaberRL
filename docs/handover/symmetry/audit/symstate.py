"""Symmetric state + symmetric action -> does one hand come out faster?"""
import sys, os, numpy as np
mod = os.environ.get("MOD")
if mod: sys.path.insert(0, mod)
import beat_arena_rs as B
from mirror_ops import *
from fkcheck import lim
FAR = [(200.0, 0.0, 1.4, 1, 3*np.pi/2, 18.0, True, 1, False, False)]  # centred bomb: symmetric chart
rng = np.random.default_rng(int(os.environ.get("SEED", 0)))
P = B.RawBeatSim(max_steps=100000, gears=5)
L = list(range(3, 10)) + list(range(17, 23))
res = []; asym_q = []
ground = os.environ.get("GROUND", "1") == "1"
for i in range(int(os.environ.get("N", 1500))):
    q = np.zeros(NQ)
    lo, hi = lim[:, 0], lim[:, 1]
    q[L] = rng.uniform(lo[L], hi[L])
    if ground:  # keep legs near stance so feet are on the ground
        q[17:23] = [0, 0, -0.1396, 0, 0, 0.1396] + rng.normal(size=6) * 0.05
    q[0] = rng.uniform(-0.2, 0.5)
    q = 0.5 * (q + m_joint(q))  # symmetrize (keeps L, writes R = mirror(L); spine yaw/roll -> 0)
    q[L] *= 2; q[[j for j in range(17) if j > 2 and j not in L] + list(range(23, 29))] *= 2  # undo halving on non-self-mirror joints
    q = m_joint(q) * 0.5 + q * 0.5
    v = rng.normal(size=35) * 0.5
    v = 0.5 * (v + m_qvel(v))
    qp = np.concatenate([[0, 0.93 if ground else 3.0, -0.65], [1, 0, 0, 0], q])
    qp2 = 0.5 * (qp + m_qpos(qp)); qp2[3:7] = [1, 0, 0, 0]
    a = np.where(rng.random(NQ) < 0.6, rng.integers(1, 5, NQ), 0).astype(np.uint8)
    a[[1, 2]] = 0
    if not ground: pass
    else: a[17:29] = 0  # legs coast
    a[10:17] = m_action(a)[10:17]; a[23:29] = m_action(a)[23:29]
    assert (m_action(a) == a).all()
    P.set_chart(FAR, [])
    if ground:  # drop to ground: find lowest-corner via one set + reading not available; just place root and let solver handle
        pass
    P.set_joints(list(qp2), list(v))
    assert np.abs(m_qpos(P.qpos()) - np.asarray(P.qpos())).max() < 1e-12
    o, r, d, e = P.step(list(a))
    o = np.asarray(o)
    tl, tr = np.linalg.norm(o[624:627]), np.linalg.norm(o[627:630])
    wl, wr = np.linalg.norm(o[78:81]), np.linalg.norm(o[89:92])
    qq = np.asarray(P.qpos()); 
    res.append((tl - tr, wl - wr))
    asym_q.append(np.abs(np.asarray(P.qpos()) - m_qpos(P.qpos())).max())
res = np.array(res) * 6.0  # undo V_NORM -> m/s
n = len(res)
for k, name in enumerate(["tip speed", "wrist speed"]):
    d = res[:, k]; nz = d[np.abs(d) > 1e-9]
    print(f"GROUND={ground} {name}: mean(L-R)={d.mean():+.4f} m/s (se {d.std()/np.sqrt(n):.4f}), "
          f"RMS={np.sqrt((d**2).mean()):.3f}, L faster in {np.mean(nz>0)*100:.1f}% of {len(nz)} asym cases")
print("median self-asym |q - Mq| after 1 tick:", np.median(asym_q), " p90", np.percentile(asym_q, 90))
