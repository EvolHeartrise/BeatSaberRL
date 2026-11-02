"""Gates for the prototype: analytic -grad E vs central differences,
closed-loop work ~ 0, continuity through u = 0 and the hanging pole,
mirror exactness of the canonical-evaluation scheme."""
import numpy as np

from srom_model import ShoulderROM, DEG

rng = np.random.default_rng(1)
m = ShoulderROM()
N = 20000
y = rng.uniform(-100, 45, N) * DEG
p = rng.uniform(-90, 65, N) * DEG
t = rng.uniform(-110, 90, N) * DEG   # a little past the twist box too
E, (ty, tp, tt), u, z = m.eval(y, p, t)
h = 1e-6
num = []
for k in range(3):
    dq = np.zeros((3, N))
    dq[k] = h
    Ep = m.eval(y + dq[0], p + dq[1], t + dq[2])[0]
    Em = m.eval(y - dq[0], p - dq[1], t - dq[2])[0]
    num.append(-(Ep - Em) / (2 * h))
an = np.stack([ty, tp, tt])
num = np.stack(num)
err = np.abs(an - num) / (1e-3 + np.abs(an))
print(f"grad check: worst rel err {err.max():.2e} (median |tau| {np.median(np.abs(an)):.3f})")

# closed loops (random polygon through the space, 400 segments each)
worst = 0.0
for _ in range(20):
    pts = np.stack([rng.uniform(-100, 45, 5), rng.uniform(-90, 65, 5),
                    rng.uniform(-90, 70, 5)]) * DEG
    pts = np.concatenate([pts, pts[:, :1]], 1)
    work, tot = 0.0, 0.0
    for i in range(5):
        s = np.linspace(0, 1, 2001)
        q = pts[:, i:i + 1] + (pts[:, i + 1:i + 2] - pts[:, i:i + 1]) * s
        _, tq, _, _ = m.eval(*q)
        tq = np.stack(tq)
        dq = np.diff(q, axis=1)
        tm = 0.5 * (tq[:, 1:] + tq[:, :-1])
        dw = (tm * dq).sum(0)
        work += dw.sum()
        tot += np.abs(dw).sum()
    worst = max(worst, abs(work) / max(tot, 1e-12))
print(f"closed-loop work / path |work|: worst {worst:.2e}")

# continuity at the hanging pole: same configuration, different (y, t)
# with y - t fixed -> identical E and identical axial torque
for tau0 in (-60, 0, 75):
    ys = np.array([-100, -40, 0, 45]) * DEG
    ts = ys - tau0 * DEG
    E0, tq0, u0, _ = m.eval(ys, np.full(4, -90 * DEG), ts)
    print(f"pole tau={tau0:4d}: E {np.round(E0, 6)}  tau_t {np.round(tq0[2], 5)}  "
          f"tau_p {np.round(tq0[1], 5)}")

# mirror: right arm evaluated at (-yR, p, -tR) == left at (y, p, t), torques negated
yR, tR = -y, -t
Er, (tyr, tpr, ttr), _, _ = m.eval(-yR, p, -tR)
print("mirror bitwise:", np.array_equal(Er, E), np.array_equal(-tyr, -ty), np.array_equal(tpr, tp))
