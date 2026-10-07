"""Stiffness budget of the passive shoulder term with the model's own M(q).

gen : sample 600 replay arm poses (left-canonical, rest of the body at the
      reset pose) x twist placed at u = +-1, +-1.25, +-1.5 and the two twist
      box stops -> poses_stiff.txt (+ meta)
read: per pose K = Hessian(E) wrt (yaw, pitch, twist) by central
      differences of the analytic torque, omega^2 = max eig((M^-1)_sh K)
      with (M^-1)_sh the shoulder 3x3 block of the FULL inverse; omega*h at
      h = 1/240 and 1/480; edge damping b*h/I_ax (explicit-equivalent).
"""
import os
import sys

import numpy as np

from mass_probe import Q_START, SCR
from srom_model import ShoulderROM, DEG

H = 1.0 / 240.0
BOX = (-90 * DEG, 70 * DEG)   # left twist box
TAGS = ("u=+1", "u=-1", "u=+1.25", "u=-1.25", "u=+1.5", "u=-1.5", "box hi", "box lo")
ALLTAGS = TAGS + ("box-random",)


def model():
    return ShoulderROM()


def gen():
    from load_replays import load
    z = load()
    rng = np.random.default_rng(3)
    idx = rng.choice(len(z["q"]), 600, replace=False)
    m = model()
    rows, meta = [], []
    for i in idx:
        a = z["q"][i]
        y, p = a[1], a[0]
        L = m.limits(np.array(y), np.array(p))
        mid = 0.5 * (L["hi"][0] + L["lo"][0])
        wh = 0.5 * (L["hi"][0] - L["lo"][0])
        for tag in TAGS:
            if tag.startswith("u="):
                uu = float(tag[2:])
                t = y - (mid + uu * wh)
            else:
                t = BOX[1] if tag == "box hi" else BOX[0]
            t = float(np.clip(t, BOX[0] - 0.067, BOX[1] + 0.067))  # <= 1 substep overshoot
            q = np.zeros(29)
            for j, v in Q_START.items():
                q[j] = v
            q[3:10] = a
            q[5] = t
            rows.append(q)
            meta.append((i, tag))
    # uniform over the whole joint box (worst cases: deep non-human corners)
    for i in range(3000):
        a = np.array([rng.uniform(-90, 65), rng.uniform(-100, 45), rng.uniform(-90, 70),
                      rng.choice([0.0, 5.0, rng.uniform(0, 150)]), rng.uniform(0, 180),
                      rng.uniform(-95, 95), rng.uniform(-30, 15)]) * DEG
        q = np.zeros(29)
        for j, v in Q_START.items():
            q[j] = v
        q[3:10] = a
        rows.append(q)
        meta.append((-1, "box-random"))
    np.savetxt(os.path.join(SCR, "poses_stiff.txt"), np.array(rows), fmt="%.17g")
    np.save(os.path.join(SCR, "poses_stiff_meta.npy"), np.array(meta, dtype=object))
    print("wrote", len(rows))


def hess(m, y, p, t, h=1e-6):
    K = np.zeros((3, 3))
    for k in range(3):
        d = np.zeros(3)
        d[k] = h
        tp_ = np.array(m.eval(y + d[0], p + d[1], t + d[2])[1])
        tm_ = np.array(m.eval(y - d[0], p - d[1], t - d[2])[1])
        K[:, k] = -(tp_ - tm_) / (2 * h)
    return 0.5 * (K + K.T)


def read():
    m = model()
    P = np.loadtxt(os.path.join(SCR, "poses_stiff.txt"))
    meta = np.load(os.path.join(SCR, "poses_stiff_meta.npy"), allow_pickle=True)
    Ms = np.loadtxt(os.path.join(SCR, "mass_stiff.txt"))
    J = np.array([1.0, 0.0, -1.0])
    out = {t: [] for t in ALLTAGS}
    for q, (i, tag), mm in zip(P, meta, Ms):
        M = mm.reshape(35, 35)
        Mi = np.linalg.inv(M)
        idx = [6 + 4, 6 + 3, 6 + 5]   # yaw, pitch, twist
        B = Mi[np.ix_(idx, idx)]
        y, p, t = q[4], q[3], q[5]
        K = hess(m, y, p, t)
        ev = np.linalg.eigvals(B @ K).real
        w2 = ev.max()
        _, _, u, _ = m.eval(y, p, t)
        iax = 1.0 / (J @ B @ J)
        b = m.damping(u)
        out[tag].append((np.sqrt(max(w2, 0)) * H, b * H / iax, abs(u), np.linalg.eigvalsh(K).max(),
                         ev.min(), q[6]))
    print(f"{'state':9s} {'omega*h@240 med/p99/max':>26s} {'@480 max':>9s} "
          f"{'b*h/I_ax med/max':>18s} {'|u| med':>8s} {'k_max N.m/rad med/max':>22s}")
    for tag in ALLTAGS:
        a = np.array(out[tag])
        wh = a[:, 0]
        print(f"{tag:9s} {np.median(wh):8.3f} {np.percentile(wh, 99):7.3f} {wh.max():7.3f} "
              f"{wh.max() / 2:9.3f} {np.median(a[:, 1]):9.3f} {a[:, 1].max():8.3f} "
              f"{np.median(a[:, 2]):8.2f} {np.median(a[:, 3]):10.1f} {a[:, 3].max():10.1f}")
    allw = np.concatenate([np.array(out[t])[:, 0] for t in ALLTAGS])
    print("worst omega*h at 1/240:", allw.max())
    a = np.array(out["box-random"])
    print("box-random: frac |u|>1.25 (linear-continuation region)", np.mean(a[:, 2] > 1.25),
          " omega*h there max", a[a[:, 2] > 1.25, 0].max() if (a[:, 2] > 1.25).any() else 0)
    print("box-random: negative eigen (non-convex directions) min", a[:, 4].min())


if __name__ == "__main__":
    gen() if sys.argv[1] == "gen" else read()
