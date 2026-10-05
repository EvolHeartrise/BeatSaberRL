"""Symmetric actions from the symmetric reset: any yaw / lateral drift is pure mirror defect; its MEAN is a handedness bias."""
import sys, os, numpy as np
mod = os.environ.get("MOD")
if mod: sys.path.insert(0, mod)
import beat_arena_rs as B
from mirror_ops import *
FAR = [(200.0, 0.0, 1.4, 1, 3*np.pi/2, 18.0, True, 1, False, False)]
rng = np.random.default_rng(int(os.environ.get("SEED", 0)))
T = int(os.environ.get("T", 15)); N = int(os.environ.get("N", 400))
P = B.RawBeatSim(max_steps=100000, gears=5)
yaw = []; rx = []; tipd = []; legs = os.environ.get("LEGS", "hold")
for trial in range(N):
    P.set_chart(FAR, [])
    a = np.zeros(NQ, np.uint8)
    for t in range(T):
        if t % 3 == 0:
            a = np.zeros(NQ, np.uint8)
            a[3:10] = np.where(rng.random(7) < 0.5, rng.integers(1, 5, 7), 0)
            a[0] = rng.integers(0, 5) if rng.random() < 0.3 else 0
        # symmetric leg hold on hip/knee pitch only (self-mirror joints)
        b = a.copy()
        if legs == "hold":
            q = np.asarray(P.qpos())[7:]; w = np.asarray(P.omega())
            for j, tgt in ((17, 0.0), (20, 0.0), (21, 0.0)):
                u = -(q[j] - tgt) * 4 - 0.4 * w[j]
                b[j] = 3 if u > 0.2 else (4 if u < -0.2 else 0)
        b[10:17] = m_action(b)[10:17]; b[23:29] = m_action(b)[23:29]
        b[1] = b[2] = 0
        assert (m_action(b) == b).all()
        _, _, d, _ = P.step(list(b))
        if d: break
    qp = np.asarray(P.qpos())
    # yaw about world y from quaternion (w,x,y,z)
    w_, x_, y_, z_ = qp[3:7]
    yaw.append(np.arctan2(2*(w_*y_ + x_*z_), 1 - 2*(y_*y_ + x_*x_))); rx.append(qp[0])
    s = np.asarray(P.sabers()); tl = s[0, :3] + 1.16*s[0, 3:]; tr = s[1, :3] + 1.16*s[1, 3:]
    tipd.append(np.linalg.norm(tl * [-1, 1, 1] - tr))
yaw = np.degrees(yaw); rx = np.array(rx) * 1000; n = len(yaw)
print(f"T={T} N={n} legs={legs}: yaw mean {yaw.mean():+.3f} deg (se {yaw.std()/np.sqrt(n):.3f}), RMS {np.sqrt((yaw**2).mean()):.3f};  root-x mean {rx.mean():+.3f} mm (se {rx.std()/np.sqrt(n):.3f}), RMS {np.sqrt((rx**2).mean()):.3f};  L/R tip asym median {np.median(tipd)*1000:.2f} mm p90 {np.percentile(tipd,90)*1000:.1f}")
