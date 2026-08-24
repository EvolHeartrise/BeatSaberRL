"""Crouch start table: analytic squat synergy + sim-selected balance.
Pose family: knee flexion K drives ankle dorsiflexion a=c1*K (foot
flat -> shank tilts fwd a), hip flexion h=K-a (level pelvis), spine
lean s=c2*K. Root dropped so the lowest sole corner touches y=0
(reset-machinery replica). Per depth, (c1,c2) selected to maximize
zero-action survival in the real sim; accept if >= 18 ticks (the
standing reset itself survives ~23)."""
import sys, math
import numpy as np
import beat_arena_rs
from beat_arena_env.chart import Note, note_tuple
from beat_arena_env.kinematics import fk_body, F_L_FOOT, F_R_FOOT
from beat_arena_env.constants import Q_START_ARM, Q_START_LEG

CORNERS = [[-0.045, -0.070, -0.075], [-0.045, -0.070, 0.175],
           [0.045, -0.070, -0.075], [0.045, -0.070, 0.175]]
FAR = [Note(hit_time=500.0, col=2, row=1, x=2.5, y=1.4, color=0,
            angle=4.712, speed=4.0)]
ANKLE_HI = 0.6108652381980153

def pose(K, c1, c2, c4=0.0):
    a = min(c1 * K + c4, ANKLE_HI)
    h = K - a
    q = [0.0] * 29
    for j, v in Q_START_ARM: q[j] = v
    for j, v in Q_START_LEG: q[j] = v
    q[0] = c2 * K
    for hip, knee, ank in ((17, 20, 21), (23, 26, 27)):
        q[hip] = h; q[knee] = K; q[ank] = a
    q36 = [0.0, 1.2, -0.65, 1.0, 0.0, 0.0, 0.0] + q
    rots, pos = fk_body(q36)
    min_y = 1e9
    for fr in (F_L_FOOT, F_R_FOOT):
        r, p = rots[fr], pos[fr]
        for c in CORNERS:
            y = p[1] + r[1][0]*c[0] + r[1][1]*c[1] + r[1][2]*c[2]
            min_y = min(min_y, y)
    q36[1] -= min_y
    return q36

def mk(q36):
    sim = beat_arena_rs.RawBeatSim(max_steps=20000, effort_coef=0.0,
                                   wide_acc=False, gears=5,
                                   foot_contact=True)
    sim.set_chart([note_tuple(n) for n in FAR], [])
    sim.set_joints(q36, [0.0] * 35)
    return sim

def coast_survival(q36, max_t=40):
    sim = mk(q36)
    for t in range(max_t):
        _o, _r, d, _e = sim.step([0] * 29)
        if d: return t
    return max_t

def supported_survival(q36, max_t=60):
    """Bang-bang leg support + ankle/spine trim from the settled pose:
    can an acting controller keep this depth? Returns ticks survived
    (and requires staying within 0.15 m of the start depth)."""
    sim = mk(q36)
    y0 = q36[1]
    for t in range(max_t):
        vy = sim.twist()[1]; hz = sim.head_pose()[2]
        a = [0] * 29
        leg = 2 if vy < -0.02 else (3 if vy > 0.06 else 0)
        for j in (17, 23, 20, 26): a[j] = leg
        tr = 4 if hz > -0.50 else (3 if hz < -0.68 else 0)
        a[0] = tr
        for j in (21, 27): a[j] = tr
        _o, _r, d, _e = sim.step(a)
        if d or sim.qpos()[1] < y0 - 0.15: return t
    return max_t

rows = []
for K in [0.3, 0.5, 0.7, 0.9, 1.1, 1.3, 1.5, 1.7, 1.9]:
    best = None
    for c1 in (0.35, 0.45, 0.55):
        for c2 in (-0.1, 0.0, 0.1):
            for c4 in (0.0, 0.06, 0.12, 0.18):
                q36 = pose(K, c1, c2, c4)
                sup = supported_survival(q36)
                co = coast_survival(q36)
                score = sup + 0.5 * co
                if best is None or score > best[0]:
                    best = (score, sup, co, c1, c2, c4, q36)
    _sc, sup, co, c1, c2, c4, q36 = best
    ok = sup >= 55 and co >= 10
    rows.append((K, ok, q36))
    print(f"K={K:.1f} rooty={q36[1]:.3f} supported={sup:2d}/60 coast={co:2d}"
          f" c1={c1} c2={c2} c4={c4}  {'ACCEPT' if ok else 'reject'}")
acc = [(K, q36) for K, ok, q36 in rows if ok]
np.save(sys.argv[1], np.array([q for _K, q in acc]))
print(f"accepted {len(acc)}/{len(rows)}")
