"""Near-stance SYMMETRIC states + symmetric actions, 1 tick: world lateral/yaw outputs must be exactly 0 in a mirror-exact env."""
import sys, os, numpy as np
mod = os.environ.get("MOD")
if mod: sys.path.insert(0, mod)
import beat_arena_rs as B
from mirror_ops import *
FAR = [(200.0, 0.0, 1.4, 1, 3*np.pi/2, 18.0, True, 1, False, False)]
rng = np.random.default_rng(0)
P = B.RawBeatSim(max_steps=100000, gears=5)
P.set_chart(FAR, []); q0 = np.asarray(P.qpos())
out = []; armd = []; cases = []
from fkcheck import lim as lim_
for i in range(int(os.environ.get("N", 3000))):
    q = q0[7:].copy()
    q[3:10] += rng.normal(size=7) * 0.3; q[17:23] += rng.normal(size=6) * 0.04; q[0] += rng.normal() * 0.05
    from fkcheck import lim
    q = np.clip(q, lim[:,0] + 0.02, lim[:,1] - 0.02) if os.environ.get('CLIP','0')=='1' else q
    q[10:17] = m_joint(q)[10:17]; q[23:29] = m_joint(q)[23:29]; q[1] = q[2] = 0
    qp = np.concatenate([q0[:3], [1, 0, 0, 0], q])
    # re-drop root so lowest corner touches: approximate by keeping reset height minus small random
    qp[1] = q0[1] + rng.normal() * 0.003
    v = np.zeros(35); v[6:] = rng.normal(size=29) * 0.3; v[1] = rng.normal()*0.05; v[3] = rng.normal()*0.1
    v[6+10:6+17] = m_joint(v[6:])[10:17]; v[6+23:6+29] = m_joint(v[6:])[23:29]; v[6+1] = v[6+2] = 0
    assert np.abs(m_qvel(v) - v).max() < 1e-12 and np.abs(m_qpos(qp) - qp).max() < 1e-12
    a = np.zeros(NQ, np.uint8)
    a[3:10] = np.where(rng.random(7) < 0.5, rng.integers(1, 5, 7), 0)
    a[17:23] = np.where(rng.random(6) < 0.5, rng.integers(1, 5, 6), 0)
    a[0] = rng.integers(0, 5)
    a[10:17] = m_action(a)[10:17]; a[23:29] = m_action(a)[23:29]
    P.set_chart(FAR, []); P.set_joints(list(qp), list(v)); P.step(list(a))
    tw = np.asarray(P.twist()); om = np.asarray(P.omega())
    out.append([tw[0], tw[4], tw[5]])
    vv0 = np.concatenate([tw, om]); dd = np.abs(vv0 - m_qvel(vv0))
    if dd.max() > 1e-6 and len(cases) < 12:
        qa = np.asarray(P.qpos())[7:]
        j = int(np.argmax(dd[6:]))
        cases.append((j, dd.max(), "pre q", np.round(q[[j, PERM[j]]], 4), "lim", lim_[j], "post q", np.round(qa[[j, PERM[j]]],5)))      # vx, wy (yaw), wz (roll) — must be 0
    vv = np.concatenate([tw, om]); armd.append(np.abs(vv - m_qvel(vv)).max())
out = np.array(out); n = len(out)
m = out.mean(0); se = out.std(0) / np.sqrt(n)
print(f"N={n}: mean vx {m[0]*1000:+.3f} mm/s (se {se[0]*1000:.3f}) | mean yaw-rate {m[1]:+.4f} rad/s (se {se[1]:.4f}) | mean roll-rate {m[2]:+.4f} (se {se[2]:.4f}) | median self-defect {np.median(armd):.2e}, p90 {np.percentile(armd,90):.2e}")

for c in cases: print(c)
