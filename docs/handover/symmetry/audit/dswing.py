"""Symmetric double swings from the reset stance (both hands, mirrored commands)."""
import sys, os, numpy as np
mod = os.environ.get("MOD")
if mod: sys.path.insert(0, mod)
import beat_arena_rs as B
from mirror_ops import *
FAR = [(200.0, 0.0, 1.4, 1, 3*np.pi/2, 18.0, True, 1, False, False)]
rng = np.random.default_rng(int(os.environ.get("SEED", 0))); T = int(os.environ.get("T", 6))
P = B.RawBeatSim(max_steps=100000, gears=5)
yaw = []; dtip = []; dpk = []; capfrac = []
for trial in range(int(os.environ.get("N", 1500))):
    P.set_chart(FAR, [])
    pkL = pkR = 0; capped = False
    for t in range(T):
        if t % 2 == 0:
            a = np.zeros(NQ, np.uint8)
            a[3:10] = np.where(rng.random(7) < 0.7, rng.integers(1, 3, 7), 0)   # full gears
            a[10:17] = m_action(a)[10:17]
        o, _, d, _ = P.step(list(a)); o = np.asarray(o)
        pkL = max(pkL, np.linalg.norm(o[624:627]) * 6); pkR = max(pkR, np.linalg.norm(o[627:630]) * 6)
        om = np.abs(np.asarray(P.omega())); cap = np.array([16]*7 + [50, 90, 50]*0 + [16]*0)
        if (om[3:7] > 15.9).any() or (om[7] > 49.9) or (om[9] > 49.9) or (om[8] > 89.9): capped = True
    q = np.asarray(P.qpos()); w_, x_, y_, z_ = q[3:7]
    yaw.append(np.degrees(np.arctan2(2*(w_*y_ + x_*z_), 1 - 2*(y_*y_ + x_*x_))))
    dpk.append(pkL - pkR); capfrac.append(capped)
yaw = np.array(yaw); dpk = np.array(dpk); n = len(yaw); c = np.array(capfrac)
print(f"T={T}: end yaw mean {yaw.mean():+.3f} deg (se {yaw.std()/np.sqrt(n):.3f}) RMS {np.sqrt((yaw**2).mean()):.3f} | peak tip L-R mean {dpk.mean():+.4f} m/s (se {dpk.std()/np.sqrt(n):.4f}), median {np.median(dpk):+.4f}, L faster {np.mean(dpk>1e-6)*100:.1f}% R faster {np.mean(dpk<-1e-6)*100:.1f}% | cap engaged in {c.mean()*100:.0f}% of swings")
