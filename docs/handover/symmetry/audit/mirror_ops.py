"""Mirror operator M for the BeatSaberRL env (x -> -x reflection)."""
import math, numpy as np
NQ = 29
# per-joint sign under mirror (q_mirror[perm[j]] = sign[j] * q[j])
SPINE = [1, -1, -1]
ARM = [1, -1, -1, 1, -1, 1, -1]      # pitch yaw twist elbow roll bend dev
LEG = [1, -1, -1, 1, 1, -1]          # hp hy hr knee ap ar
SIGN = np.array(SPINE + ARM + ARM + LEG + LEG, float)
PERM = np.array(list(range(3)) + list(range(10, 17)) + list(range(3, 10))
                + list(range(23, 29)) + list(range(17, 23)))
assert sorted(PERM) == list(range(NQ)) and all(PERM[PERM] == np.arange(NQ))

def m_joint(q):
    out = np.empty(NQ)
    out[PERM] = SIGN * np.asarray(q)
    return out

def m_qpos(qp):
    qp = np.asarray(qp, float)
    p = [-qp[0], qp[1], qp[2]]
    w, x, y, z = qp[3:7]
    return np.concatenate([p, [w, x, -y, -z], m_joint(qp[7:])])

def m_qvel(v):
    v = np.asarray(v, float)
    return np.concatenate([[-v[0], v[1], v[2], v[3], -v[4], -v[5]], m_joint(v[6:])])

# gear id mirror: 1<->2, 3<->4, 5<->6 on sign-flipped joints
GSWAP = np.array([0, 2, 1, 4, 3, 6, 5])
def m_action(a):
    a = np.asarray(a)
    out = np.empty(NQ, dtype=a.dtype)
    flipped = np.where(SIGN < 0, GSWAP[a], a)
    out[PERM] = flipped
    return out

# ---------------- observation (OBS_DIM 630) ----------------
OBS_DIM = 630
def build_obs_map():
    """Return (src_index, sign) arrays: obs_m[i] = sign[i] * obs[src[i]]."""
    src = np.arange(OBS_DIM); sg = np.ones(OBS_DIM)
    def setb(dst, s_idx, signs):
        for d, s, g in zip(dst, s_idx, signs):
            src[d] = s; sg[d] = g
    # root pos, rot6d, vlin, vang
    setb(range(0, 3), range(0, 3), [-1, 1, 1])
    setb(range(3, 9), range(3, 9), [1, -1, -1, -1, 1, 1])
    setb(range(9, 12), range(9, 12), [-1, 1, 1])
    setb(range(12, 15), range(12, 15), [1, -1, -1])
    # q lanes 15..45: spine 15..17, armL 18..25, armR 26..33, legL 34..39, legR 40..45
    arm_lane_sign = [1, -1, -1, 1, 1, -1, 1, -1]   # pitch yaw twist elbow cos sin bend dev
    setb(range(15, 18), range(15, 18), SPINE)
    setb(range(18, 26), range(26, 34), arm_lane_sign)
    setb(range(26, 34), range(18, 26), arm_lane_sign)
    setb(range(34, 40), range(40, 46), LEG)
    setb(range(40, 46), range(34, 40), LEG)
    # qd 46..74
    for j in range(NQ):
        src[46 + PERM[j]] = 46 + j; sg[46 + PERM[j]] = SIGN[j]
    # hands 75..85 / 86..96: pos(3) vel(3) blade(3) meter post
    hs = [-1, 1, 1, -1, 1, 1, -1, 1, 1, 1, 1]
    setb(range(75, 86), range(86, 97), hs)
    setb(range(86, 97), range(75, 86), hs)
    setb(range(97, 100), range(97, 100), [-1, 1, 1])     # head
    setb([100, 101], [101, 100], [1, 1])                  # contact bits
    # entities 102..533: per slot 18 lanes
    for s in range(24):
        b = 102 + 18 * s
        lane_src = list(range(18)); lane_sg = [1] * 18
        lane_sg[0] = -1                      # x
        lane_src[4], lane_src[5] = 5, 4      # colour one-hot swap
        lane_sg[6] = -1                      # cos(angle)
        setb([b + k for k in range(18)], [b + k for k in lane_src], lane_sg)
    # mask 534..557 unchanged; ground imp 558..561 [Lh, Lt, Rh, Rt]
    setb(range(558, 562), [560, 561, 558, 559], [1, 1, 1, 1])
    # body pts 562..579 [le, re, lk, rk, lf, rf], vel 580..597
    for base in (562, 580):
        for i in range(6):
            j = i ^ 1
            setb(range(base + 3 * i, base + 3 * i + 3),
                 range(base + 3 * j, base + 3 * j + 3), [-1, 1, 1])
    # prev blocks 598..602 / 603..607: x y cos sin valid
    ps = [-1, 1, -1, 1, 1]
    setb(range(598, 603), range(603, 608), ps)
    setb(range(603, 608), range(598, 603), ps)
    # tau lanes 608..623 unchanged (axis-shared)
    # tip vel 624..629
    setb(range(624, 627), range(627, 630), [-1, 1, 1])
    setb(range(627, 630), range(624, 627), [-1, 1, 1])
    return src, sg
OBS_SRC, OBS_SG = build_obs_map()
def m_obs(o):
    o = np.asarray(o)
    return OBS_SG.astype(o.dtype) * o[..., OBS_SRC]

# chart
def m_angle(a):
    return (math.pi - a) % (2 * math.pi)
def m_note(t, bomb_kind=1):
    color = t[3] if t[7] == bomb_kind else 1 - t[3]
    return (t[0], -t[1], t[2], color, m_angle(t[4]), t[5], t[6], t[7], t[8], t[9])
def m_wall(w):
    return (w[0], -w[1]) + tuple(w[2:])
