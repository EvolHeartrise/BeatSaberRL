import sys, numpy as np
import beat_arena_rs as B
from mirror_ops import *
FAR = [(200.0, 0.3, 1.4, 1, 3*np.pi/2, 18.0, False, 0, False, False)]
lim = np.array([(-0.349,0.785),(-0.698,0.698),(-0.523,0.523),(-1.57,1.13),(-1.74,0.78),(-1.57,1.22),(0,2.61),(0,3.14),(-1.65,1.65),(-0.52,0.26)]
               + [(-1.57,1.13),(-0.78,1.74),(-1.22,1.57),(0,2.61),(-3.14,0),(-1.65,1.65),(-0.26,0.52)] + [(-0.3,2.0),(-0.7,0.6),(-0.7,0.5),(0,2.6),(-0.8,0.6),(-0.3,0.3)]
               + [(-0.3,2.0),(-0.6,0.7),(-0.5,0.7),(0,2.6),(-0.8,0.6),(-0.3,0.3)])
rng = np.random.default_rng(0)
P = B.RawBeatSim(max_steps=100000, gears=5); Q = B.RawBeatSim(max_steps=100000, gears=5)
P.set_chart(FAR, []); Q.set_chart(FAR, [])
worst = 0; worst_o = 0
for i in range(200):
    q = rng.uniform(lim[:, 0], lim[:, 1]); quat = rng.normal(size=4); quat /= np.linalg.norm(quat)
    qp = np.concatenate([[rng.normal()*0.1, 0.93, -0.65 + rng.normal()*0.1], quat, q])
    P.set_joints(list(qp), [0.0]*35); Q.set_joints(list(m_qpos(qp)), [0.0]*35)
    sp = np.asarray(P.sabers()); sq = np.asarray(Q.sabers())
    # mirror sabers: swap hands, negate x of pos and blade
    msp = sp[::-1].copy(); msp[:, 0] *= -1; msp[:, 3] *= -1
    worst = max(worst, np.abs(sq - msp).max())
    worst_o = max(worst_o, np.abs(np.asarray(Q.obs()) - m_obs(np.asarray(P.obs()))).max())
print("FK saber mirror max err:", worst, "  obs mirror max err (static):", worst_o)
errs = np.zeros(OBS_DIM)
for i in range(200):
    q = rng.uniform(lim[:, 0], lim[:, 1]); quat = rng.normal(size=4); quat /= np.linalg.norm(quat)
    qp = np.concatenate([[rng.normal()*0.1, 0.93, -0.65 + rng.normal()*0.1], quat, q])
    P.set_joints(list(qp), list(rng.normal(size=35))); Q.set_joints(list(m_qpos(qp)), list(m_qvel(np.asarray(P.twist()+list(P.omega())))))
    errs = np.maximum(errs, np.abs(np.asarray(Q.obs()) - m_obs(np.asarray(P.obs()))))
print("lanes with err>1e-6:", [(i, round(float(errs[i]),4)) for i in np.where(errs > 1e-6)[0]])
