import sys, os, numpy as np
sys.path.insert(0, os.environ.get("MOD", "mod_fix"))
import beat_arena_rs as B
from mirror_ops import *
FAR = [(200.0, 0.3, 1.4, 1, 3*np.pi/2, 18.0, False, 0, False, False)]
z = np.load("s.npz"); a = np.load("s_run_orig_fix.npy"); b = np.load("s_run_mirror_fix.npy")
mv = np.array([np.concatenate([m_qpos(r[:36]), m_qvel(r[36:])]) for r in a])
d = np.abs(b - mv)[:, 36:].max(1)
idx = np.where(d > 1e-6)[0]
rng = np.random.default_rng(0)
def run(qp, v, a):
    P = B.RawBeatSim(max_steps=100000, gears=5); P.set_chart(FAR, [])
    P.set_joints(list(qp), list(v)); P.step(list(a))
    return np.asarray(list(P.twist()) + list(P.omega())), P
sens = []
for i in idx:
    s, act = z["S"][i], z["A"][i]
    base, P = run(s[:36], s[36:], act)
    r = 0
    for k in range(8):
        pert, _ = run(s[:36] + rng.normal(size=36) * 1e-13 * np.r_[np.ones(3), np.zeros(4), np.ones(29)], s[36:] * (1 + rng.normal(size=35) * 1e-14), act)
        r = max(r, np.abs(pert - base).max())
    sens.append(r)
    print(i, f"{d[i]:.1e} {sens[-1]:.1e}", "EXPLAINED" if sens[-1] > 0.01*d[i] else "UNEXPLAINED")
