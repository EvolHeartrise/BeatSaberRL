"""usage: cross.py gen|run_orig|run_mirror out.npz ; samples from defect-like distribution"""
import sys, os, numpy as np
mod = os.environ.get("MOD")
if mod: sys.path.insert(0, mod)
import beat_arena_rs as B
from mirror_ops import *
FAR = [(200.0, 0.3, 1.4, 1, 3*np.pi/2, 18.0, False, 0, False, False)]
cmd, path = sys.argv[1], sys.argv[2]
def sim(ch=FAR):
    s = B.RawBeatSim(max_steps=100000, gears=5); s.set_chart(ch, []); return s
if cmd == "gen":
    rng = np.random.default_rng(0)
    A = sim(); S = []; Aa = []
    for ep in range(30):
        A.set_chart(FAR, [])
        for k in range(25):
            q = np.asarray(A.qpos())[7:]; w = np.asarray(A.omega())
            tgt = np.zeros(NQ); tgt[19] = -0.1396; tgt[22] = 0.1396; tgt[25] = 0.1396; tgt[28] = -0.1396
            u = -(q - tgt) * 3.0 - 0.3 * w
            a = np.zeros(NQ, np.uint8); a[u > 0.3] = 3; a[u < -0.3] = 4
            a[3:17] = np.where(rng.random(14) < 0.5, rng.integers(1, 5, 14), 0)
            qp = np.asarray(A.qpos()); v = np.asarray(list(A.twist()) + list(A.omega()))
            if qp[1] < 0.6: break
            S.append(np.concatenate([qp, v])); Aa.append(a)
            _, _, d, _ = A.step(list(a))
            if d: break
    np.savez(path, S=np.array(S), A=np.array(Aa))
else:
    z = np.load(path); out = []
    for s, a in zip(z["S"], z["A"]):
        qp, v = s[:36], s[36:]
        if cmd == "run_mirror": qp, v, a = m_qpos(qp), m_qvel(v), m_action(a)
        P = sim([m_note(t) for t in FAR] if cmd == "run_mirror" else FAR)
        P.set_joints(list(qp), list(v)); P.step(list(a))
        out.append(np.concatenate([P.qpos(), list(P.twist()) + list(P.omega())]))
    np.save(path.replace(".npz", f"_{cmd}_{os.environ.get('TAG','')}.npy"), np.array(out))
