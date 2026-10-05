import sys, os, numpy as np
mod = os.environ.get("MOD")
if mod: sys.path.insert(0, mod)
import beat_arena_rs as B
from mirror_ops import *
FAR = [(200.0, 0.0, 1.4, 1, 3*np.pi/2, 18.0, True, 1, False, False)]
TGT = np.zeros(NQ); TGT[[3,10]] = -1.396; TGT[[6,13]] = 1.745; TGT[[8,15]] = 1.2217
TGT[19] = -0.1396; TGT[22] = 0.1396; TGT[25] = 0.1396; TGT[28] = -0.1396
TGT[[17,23]] = 0.15; TGT[[20,26]] = 0.3; TGT[[21,27]] = -0.15
def ctl(P, kp, kd, thr):
    q = np.asarray(P.qpos())[7:]; w = np.asarray(P.omega())
    u = -(q - TGT) * kp - kd * w
    a = np.zeros(NQ, np.uint8); a[u > thr] = 3; a[u < -thr] = 4
    return a
best = None
for kp, kd, thr in [(2, 0.2, 0.05), (4, 0.3, 0.05), (1, 0.1, 0.02), (6, 0.5, 0.1), (3, 0.6, 0.03)]:
    P = B.RawBeatSim(max_steps=100000, gears=7); P.set_chart(FAR, [])
    for t in range(1, 241):
        _, _, d, _ = P.step(list(ctl(P, kp, kd, thr)))
        if d: break
    q = np.asarray(P.qpos()); w_, x_, y_, z_ = q[3:7]
    yaw = np.degrees(np.arctan2(2*(w_*y_ + x_*z_), 1 - 2*(y_*y_ + x_*x_)))
    print(f"kp={kp} kd={kd} thr={thr}: survived {t} ticks, root=({q[0]*1000:+.2f} mm, {q[1]:.3f}), yaw {yaw:+.3f} deg, falls={P.falls} tilts={P.tilts}")
