import sys, math, numpy as np
sys.path.insert(0, "/home/evolheartrise/git projects/BeatSaberRL/python")
import beat_arena_rs as B
from mirror_ops import *

o = np.random.default_rng(0).normal(size=OBS_DIM)
assert np.allclose(m_obs(m_obs(o)), o), "M not involution on obs"
a = np.random.default_rng(1).integers(0, 7, NQ)
assert (m_action(m_action(a)) == a).all()

def chart(rng, n=60):
    xs = [-0.9, -0.3, 0.3, 0.9]; ys = [0.85, 1.40, 1.90]
    out = []
    for i in range(n):
        t = 1.0 + 0.25 * i
        kind = 1 if rng.random() < 0.15 else (2 if rng.random() < 0.1 else 0)
        out.append((t, float(rng.choice(xs)), float(rng.choice(ys)), int(rng.integers(0, 2)),
                    float(rng.integers(0, 8)) * math.pi / 4, 18.0, bool(rng.random() < 0.2),
                    kind, bool(rng.random() < 0.1), bool(rng.random() < 0.1)))
    return out

def run(seed, T=150, p_act=0.3, gears=5, tick_hz=30, verbose=True):
    rng = np.random.default_rng(seed)
    ch = chart(rng); walls = [(3.0, -0.9, 1.5, 0.6, 2.0, 1.0, 18.0)]
    A = B.RawBeatSim(max_steps=10000, gears=gears, tick_hz=tick_hz)
    Bm = B.RawBeatSim(max_steps=10000, gears=gears, tick_hz=tick_hz)
    oa = np.asarray(A.set_chart(ch, walls)); ob = np.asarray(Bm.set_chart([m_note(t) for t in ch], [m_wall(w) for w in walls]))
    res = dict(first_q_dev=None, max_q=[], max_obs=[], rew=[], ev_mismatch=0)
    d0 = np.abs(np.asarray(Bm.qpos()) - m_qpos(A.qpos())).max()
    res["reset_q"] = d0; res["reset_obs"] = np.abs(ob - m_obs(oa)).max()
    for t in range(T):
        act = np.where(rng.random(NQ) < p_act, rng.integers(1, gears, NQ), 0).astype(np.uint8)
        oa, ra, da, ea = A.step(list(act))
        ob, rb, db, eb = Bm.step(list(m_action(act)))
        dq = np.abs(np.asarray(Bm.qpos()) - m_qpos(A.qpos())).max()
        dv = np.abs(np.asarray(Bm.twist() + list(Bm.omega())) - m_qvel(list(A.twist()) + list(A.omega()))).max()
        do = np.abs(np.asarray(ob) - m_obs(np.asarray(oa))).max()
        res["max_q"].append(max(dq, dv)); res["max_obs"].append(do); res["rew"].append(abs(ra - rb))
        if [(e[0], e[1], e[2]) for e in ea] != [(e[0], e[1], e[2]) for e in eb]:
            res["ev_mismatch"] += 1
        if da or db:
            res["done"] = (t, da, db); break
    return res

if __name__ == "__main__":
    for gears in (3, 5):
        for seed in range(4):
            r = run(seed, gears=gears)
            mq = np.array(r["max_q"]); mo = np.array(r["max_obs"])
            first = int(np.argmax(mq > 0)) if (mq > 0).any() else -1
            print(f"gears={gears} seed={seed} reset dq={r['reset_q']:.1e} dobs={r['reset_obs']:.1e} | "
                  f"ticks={len(mq)} first nonzero dq @tick {first} | dq@10={mq[min(9,len(mq)-1)]:.1e} "
                  f"dq@50={mq[min(49,len(mq)-1)]:.1e} dq_max={mq.max():.1e} dobs_max={mo.max():.1e} "
                  f"drew_max={max(r['rew']):.1e} ev_mismatch={r['ev_mismatch']} done={r.get('done')}")
