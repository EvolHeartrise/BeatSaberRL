import sys, os, math, numpy as np
mod = os.environ.get("MOD")
if mod: sys.path.insert(0, mod)
import beat_arena_rs as B
from mirror_ops import *
side, out = sys.argv[1], sys.argv[2]
rng = np.random.default_rng(5)
xs = [-0.9, -0.3, 0.3, 0.9]; ys = [0.85, 1.40, 1.90]
ch = []
for i in range(80):
    t = 0.6 + 0.12 * i
    kind = 1 if rng.random() < 0.15 else (2 if rng.random() < 0.1 else (3 if rng.random() < 0.1 else 0))
    ch.append((t, float(rng.choice(xs)) * 0.5, float(rng.choice(ys)), int(rng.integers(0, 2)),
               float(rng.integers(0, 8)) * math.pi / 4 + 0.1, 12.0, bool(rng.random() < 0.2),
               kind, bool(rng.random() < 0.1), bool(rng.random() < 0.1)))
walls = [(2.0, -0.6, 1.5, 0.6, 2.0, 1.0, 12.0), (4.0, 0.3, 2.2, 1.2, 0.5, 2.0, 12.0)]
if side == "L":
    ch = [m_note(t) for t in ch]; walls = [m_wall(w) for w in walls]
for gears in (5,):
  results = []
  for ep in range(int(os.environ.get("EPS", 40))):
    P = B.RawBeatSim(max_steps=100000, gears=gears)
    P.set_chart(ch, walls)
    rec = []
    a = np.zeros(NQ, np.uint8)
    for t in range(150):
        if t % 3 == 0:
            a = np.zeros(NQ, np.uint8)
            a[3:17] = np.where(rng.random(14) < 0.6, rng.integers(1, 5, 14), 0)
        q = np.asarray(P.qpos())[7:]; w = np.asarray(P.omega())
        if side == "L": q = m_joint(q); w = m_joint(w)   # controller sees canonical (R-frame) state
        b = a.copy()
        for j in (0, 17, 20, 21, 23, 26, 27):
            u = -q[j] * 4 - 0.4 * w[j]; b[j] = 3 if u > 0.2 else (4 if u < -0.2 else 0)
        act = b if side == "R" else m_action(b)
        o, r, d, ev = P.step(list(act))
        o = np.asarray(o)
        if side == "L": o = m_obs(o)
        rec.append((o, r, d, [(e[0], e[1], round(e[2], 3)) for e in ev]))
        if d: break
    results.append(rec)
import pickle; pickle.dump(results, open(out, "wb"))
