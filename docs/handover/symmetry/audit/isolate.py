import sys, os, numpy as np
mod = os.environ.get("MOD")
if mod: sys.path.insert(0, mod)
import beat_arena_rs as B
from mirror_ops import *
from fkcheck import lim
FAR = [(200.0, 0.3, 1.4, 1, 3*np.pi/2, 18.0, False, 0, False, False)]
P = B.RawBeatSim(max_steps=100000, gears=5); Q = B.RawBeatSim(max_steps=100000, gears=5)
rng = np.random.default_rng(1)
def trial(interior, vscale, act_p, airborne, n=200):
    errs = []
    for i in range(n):
        lo, hi = lim[:, 0], lim[:, 1]
        c = (lo + hi) / 2; h = (hi - lo) / 2 * interior
        q = rng.uniform(c - h, c + h)
        qp = np.concatenate([[0, 3.0 if airborne else 0.93, -0.65], [1, 0, 0, 0], q])
        v = rng.normal(size=35) * vscale
        a = np.where(rng.random(NQ) < act_p, rng.integers(1, 5, NQ), 0).astype(np.uint8)
        P.set_chart(FAR, []); Q.set_chart([m_note(t) for t in FAR], [])
        P.set_joints(list(qp), list(v)); Q.set_joints(list(m_qpos(qp)), list(m_qvel(v)))
        P.step(list(a)); Q.step(list(m_action(a)))
        vp = np.asarray(list(P.twist()) + list(P.omega())); vq = np.asarray(list(Q.twist()) + list(Q.omega()))
        errs.append(np.abs(vq - m_qvel(vp)).max())
    errs = np.array(errs)
    return np.median(errs), np.percentile(errs, 90), errs.max()
for cfg in [(0.5, 0.0, 0.0, True), (0.5, 0.0, 0.5, True), (0.5, 1.0, 0.5, True), (0.95, 1.0, 0.5, True), (1.0, 5.0, 0.5, True),
            (0.3, 0.0, 0.0, False), (0.3, 0.5, 0.5, False)]:
    print(cfg, "median/p90/max |dv| = %.2e %.2e %.2e" % trial(*cfg))
