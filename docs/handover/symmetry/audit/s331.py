import sys, numpy as np
sys.path.insert(0, "mod_fix")
import beat_arena_rs as B
from mirror_ops import *
np.set_printoptions(precision=4, suppress=True, linewidth=200)
FAR = [(200.0, 0.3, 1.4, 1, 3*np.pi/2, 18.0, False, 0, False, False)]
z = np.load("s.npz"); i = 331; s, act = z["S"][i], z["A"][i]
def run(qp, v, a, ch=FAR):
    P = B.RawBeatSim(max_steps=100000, gears=5); P.set_chart(ch, [])
    P.set_joints(list(qp), list(v)); P.step(list(a))
    return np.asarray(list(P.twist()) + list(P.omega()))
base = run(s[:36], s[36:], act)
mir = m_qvel(run(m_qpos(s[:36]), m_qvel(s[36:]), m_action(act), [m_note(t) for t in FAR]))
print("gap vec", mir - base)
rng = np.random.default_rng(1)
for eps in (1e-14, 1e-13, 1e-12, 1e-11, 1e-10):
    r = max(np.abs(run(s[:36] + rng.normal(size=36)*eps*np.r_[np.ones(3),np.zeros(4),np.ones(29)], s[36:]*(1+rng.normal(size=35)*eps), act) - base).max() for _ in range(10))
    print(eps, r)
print("q", s[7:36]); print("v", s[36:]); print("a", act)
mq = m_qpos(s[:36]); mvv = m_qvel(s[36:])
mq2 = mq + 0.0; mv2 = mvv + 0.0   # -0.0 -> +0.0
mir2 = m_qvel(run(mq2, mv2, m_action(act), [m_note(t) for t in FAR]))
print("signed-zero-cleaned mirror gap:", np.abs(mir2 - base).max())
for dq in (1e-15, -1e-15, 1e-12, -1e-12):
    q2 = s[:36].copy(); q2[7+14] += dq
    print("perturb r_roll q by", dq, np.abs(run(q2, s[36:], act) - base).max())
for dv in (1e-12, -1e-12):
    v2 = s[36:].copy(); v2[6+14] += dv
    print("perturb r_roll v by", dv, np.abs(run(s[:36], v2, act) - base).max())
