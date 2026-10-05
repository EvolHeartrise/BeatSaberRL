import sys, os, numpy as np
mod = os.environ.get("MOD")
if mod: sys.path.insert(0, mod)
import beat_arena_rs as B
from mirror_ops import *
FAR = [(200.0, 0.0, 1.4, 1, 3*np.pi/2, 18.0, True, 1, False, False)]
rng = np.random.default_rng(0); T = 10; both = os.environ.get("BOTH", "0") == "1"
side = sys.argv[1]; out = sys.argv[2]
P = B.RawBeatSim(max_steps=100000, gears=5)
R = []
for trial in range(300):
    P.set_chart(FAR, [])
    acts = []; a = np.zeros(NQ, np.uint8)
    for t in range(T):
        if t % 3 == 0:
            a = np.zeros(NQ, np.uint8)
            a[10:17] = np.where(rng.random(7) < 0.7, rng.integers(1, 5, 7), 0)
            if both: a[3:10] = np.where(rng.random(7) < 0.7, rng.integers(1, 5, 7), 0)
        acts.append(a.copy())
    traj = []
    for a in acts:
        P.step(list(a if side == "R" else m_action(a)))
        s = np.asarray(P.sabers())
        if side == "L":  # mirror back into the R frame
            s = s[::-1].copy(); s[:, 0] *= -1; s[:, 3] *= -1
        traj.append(np.concatenate([s[1, :3] + 1.16 * s[1, 3:], s[0, :3] + 1.16 * s[0, 3:]]))
    R.append(traj)
np.save(out, np.array(R))
