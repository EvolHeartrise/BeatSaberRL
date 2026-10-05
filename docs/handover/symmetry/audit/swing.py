"""Stereotyped swings from the reset stance: right-arm swing vs its mirror (left-arm swing)."""
import sys, os, numpy as np
mod = os.environ.get("MOD")
if mod: sys.path.insert(0, mod)
import beat_arena_rs as B
from mirror_ops import *
FAR = [(200.0, 0.0, 1.4, 1, 3*np.pi/2, 18.0, True, 1, False, False)]
rng = np.random.default_rng(int(os.environ.get("SEED", 0)))
T = int(os.environ.get("T", 10)); both = os.environ.get("BOTH", "0") == "1"
P = B.RawBeatSim(max_steps=100000, gears=5); Q = B.RawBeatSim(max_steps=100000, gears=5)
peak = []; pos = []; root = []; yaw = []
for trial in range(300):
    P.set_chart(FAR, []); Q.set_chart(FAR, [])
    # random but smooth swing: pick per-joint gear held for random segments, right arm only
    acts = []
    a = np.zeros(NQ, np.uint8)
    for t in range(T):
        if t % 3 == 0:
            a = np.zeros(NQ, np.uint8)
            a[10:17] = np.where(rng.random(7) < 0.7, rng.integers(1, 5, 7), 0)
            if both:
                a[3:10] = np.where(rng.random(7) < 0.7, rng.integers(1, 5, 7), 0)
        acts.append(a.copy())
    pk_p = pk_q = 0; dmax = 0
    for a in acts:
        op, _, dp, _ = P.step(list(a)); oq, _, dq, _ = Q.step(list(m_action(a)))
        op = np.asarray(op); oq = np.asarray(oq)
        pk_p = max(pk_p, np.linalg.norm(op[627:630]) * 6); pk_q = max(pk_q, np.linalg.norm(oq[624:627]) * 6)
        sp = np.asarray(P.sabers()); sq = np.asarray(Q.sabers())
        # P right saber vs mirrored Q left saber tip
        tp = sp[1, :3] + 1.16 * sp[1, 3:]; tq = sq[0, :3] + 1.16 * sq[0, 3:]; tq[0] *= -1
        dmax = max(dmax, np.linalg.norm(tp - tq))
    peak.append(pk_p - pk_q); pos.append(dmax)
    qa = np.asarray(P.qpos()); qb = m_qpos(Q.qpos())
    root.append(qa[0] - qb[0]); yaw.append(qa[5] - qb[5])
peak = np.array(peak); pos = np.array(pos); root = np.array(root)
n = len(peak)
print(f"T={T} both={both}: peak tip speed R-swing minus mirrored L-swing: mean {peak.mean():+.4f} m/s (se {peak.std()/np.sqrt(n):.4f}), RMS {np.sqrt((peak**2).mean()):.4f}; R faster in {(peak>1e-9).mean()*100:.0f}% / L faster in {(peak<-1e-9).mean()*100:.0f}%")
print(f"   tip path mismatch: median {np.median(pos)*1000:.2f} mm, p90 {np.percentile(pos,90)*1000:.2f} mm, max {pos.max()*1000:.1f} mm;  end root-x mismatch RMS {np.sqrt((root**2).mean())*1000:.2f} mm")
