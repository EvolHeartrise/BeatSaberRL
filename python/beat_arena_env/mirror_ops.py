"""The left/right MIRROR operator M on sim state, actions and observations
(2026-10-04, sim/mirror-exact).

M is the reflection x -> -x of the world (the body's sagittal plane; the
note grid is symmetric about x = 0). The sim is mirror-exact by
construction (lib.rs / solver.rs "MIRROR-EXACT" notes):

    env(M s, M a) = M env(s, a)      (up to floating-point rounding)

and on the chart side `mirror.py` reflects the note / wall tuples
(x -> -x, colour swap, cut angle theta -> pi - theta). This module is the
operator for everything else — the input mirror-equivariant nets, mirror
losses and the regression tests use.

Joints (SPEC-v10 §1.4 actuated order, 29): the left/right blocks swap
(arm_l 3..9 <-> arm_r 10..16, leg_l 17..22 <-> leg_r 23..28) and every
coordinate whose rotation axis flips handedness under the reflection
changes sign:

    spine  [pitch +, yaw -, roll -]
    arm    [pitch +, yaw -, twist -, elbow +, roll -, bend +, dev -]
    leg    [hip pitch +, hip yaw -, hip roll -, knee +, ankle pitch +,
            ankle roll -]

(Each joint axis is a body-frame unit vector a; the reflection P = diag
(-1, 1, 1) maps Rot(a, q) to Rot(-P a, q), so an x-axis joint keeps its
angle and a y- or z-axis joint negates it.)

Root: p_x -> -p_x; quaternion (w, x, y, z) -> (w, x, -y, -z); linear
velocity (-, +, +); angular velocity (a pseudovector) (+, -, -).

Actions: the joint permutation, and on the sign-flipped joints the +/-
gear ids swap (1 <-> 2, 3 <-> 4, 5 <-> 6; 0 = coast) — valid because every
gear table is +/- paired and equal across a mirror pair.

Observations (OBS_DIM 630, constants.OBS_OFF): see `obs_map`. The
policy's hand one-hot is tied to the token slot and is NOT part of the
obs, so it does not move.
"""

from __future__ import annotations

import math

import numpy as np

from . import constants as K

NQ = 29
NV = 35

_SPINE = (1, -1, -1)
_ARM = (1, -1, -1, 1, -1, 1, -1)       # pitch yaw twist elbow roll bend dev
_LEG = (1, -1, -1, 1, 1, -1)           # hip p/y/r, knee, ankle p/r
JOINT_SIGN = np.array(_SPINE + _ARM + _ARM + _LEG + _LEG, dtype=np.float64)
JOINT_PERM = np.array(list(range(3)) + list(range(10, 17)) + list(range(3, 10))
                      + list(range(23, 29)) + list(range(17, 23)))
"""JOINT_PERM[j] = the joint the reflection maps j onto (an involution;
rust/src/dynamics/model.rs HUMANOID_MIRROR)."""

GEAR_SWAP = np.array([0, 2, 1, 4, 3, 6, 5])


def m_joint(x):
    """Mirror a per-joint vector (angles, rates, torques) — last axis 29."""
    x = np.asarray(x)
    out = np.empty_like(x, dtype=np.result_type(x, np.float64))
    out[..., JOINT_PERM] = JOINT_SIGN * x
    return out


def m_qpos(qp):
    """Mirror a 36-coordinate pose [root p (3), root quat wxyz (4), 29 q]
    (RawBeatSim.qpos / set_joints layout)."""
    qp = np.asarray(qp, dtype=np.float64)
    out = qp.copy()
    out[..., 0] = -qp[..., 0]
    out[..., 5] = -qp[..., 5]
    out[..., 6] = -qp[..., 6]
    out[..., 7:] = m_joint(qp[..., 7:])
    return out


def m_qvel(v):
    """Mirror a 35-coordinate velocity [root linear (3, world), root
    angular (3, world), 29 joint rates] (RawBeatSim.twist + omega)."""
    v = np.asarray(v, dtype=np.float64)
    out = v.copy()
    out[..., 0] = -v[..., 0]
    out[..., 4] = -v[..., 4]
    out[..., 5] = -v[..., 5]
    out[..., 6:] = m_joint(v[..., 6:])
    return out


def m_action(a):
    """Mirror gear ids (last axis 29)."""
    a = np.asarray(a)
    flipped = np.where(JOINT_SIGN < 0, GEAR_SWAP[a], a)
    out = np.empty_like(a)
    out[..., JOINT_PERM] = flipped
    return out


def m_angle(theta):
    """Cut angle (0 = +x, CCW): theta -> pi - theta, wrapped to [0, 2 pi)."""
    return (math.pi - theta) % (2.0 * math.pi)


def _obs_map():
    """(src, sign) with obs_M[i] = sign[i] * obs[src[i]]."""
    O = K.OBS_OFF
    src = np.arange(K.OBS_DIM)
    sg = np.ones(K.OBS_DIM)

    def put(dst, s_idx, signs):
        for d, s, g in zip(dst, s_idx, signs):
            src[d] = s
            sg[d] = g

    r0 = O["root_pos"]
    put(range(r0, r0 + 3), range(r0, r0 + 3), (-1, 1, 1))
    r6 = O["root_rot6"]   # [R00,R10,R20, R01,R11,R21] of P R P
    put(range(r6, r6 + 6), range(r6, r6 + 6), (1, -1, -1, -1, 1, 1))
    vl = O["root_vlin"]
    put(range(vl, vl + 3), range(vl, vl + 3), (-1, 1, 1))
    va = O["root_vang"]
    put(range(va, va + 3), range(va, va + 3), (1, -1, -1))
    # q block: one lane per joint, the two wrist rolls as (cos, sin)
    lanes, lane = [], O["q"]
    for j in range(NQ):
        n = 2 if j in K.WRAP_JOINTS else 1
        lanes.append(list(range(lane, lane + n)))
        lane += n
    assert lane == O["qd"]
    for j in range(NQ):
        mj = JOINT_PERM[j]
        if len(lanes[j]) == 2:      # cos even, sin odd in the angle
            put(lanes[mj], lanes[j], (1, JOINT_SIGN[j]))
        else:
            put(lanes[mj], lanes[j], (JOINT_SIGN[j],))
    qd = O["qd"]
    for j in range(NQ):
        put((qd + JOINT_PERM[j],), (qd + j,), (JOINT_SIGN[j],))
    # hand blocks: wrist pos (head-relative x/y, world z), wrist vel,
    # blade dir, meter, post_rem
    hs = (-1, 1, 1, -1, 1, 1, -1, 1, 1, 1, 1)
    hl, hr = O["hand_l"], O["hand_r"]
    put(range(hl, hl + K.HAND_FEAT), range(hr, hr + K.HAND_FEAT), hs)
    put(range(hr, hr + K.HAND_FEAT), range(hl, hl + K.HAND_FEAT), hs)
    hd = O["head"]
    put(range(hd, hd + 3), range(hd, hd + 3), (-1, 1, 1))
    ct = O["contact"]
    put((ct, ct + 1), (ct + 1, ct), (1, 1))
    # entity tokens: x (head-relative) -, colour one-hot swap, cut-dir
    # cos -, everything else (y, z, speed, sin, kind/free flags, wall
    # extents, back face) unchanged; the mask is unchanged
    for s in range(K.N_NOTE_SLOTS):
        b = O["notes"] + K.NOTE_FEAT * s
        ls = list(range(K.NOTE_FEAT))
        lg = [1] * K.NOTE_FEAT
        lg[0] = -1
        ls[4], ls[5] = 5, 4
        lg[6] = -1
        put([b + k for k in range(K.NOTE_FEAT)], [b + k for k in ls], lg)
    gi = O["ground_imp"]   # [L_heel, L_toe, R_heel, R_toe]
    put(range(gi, gi + 4), (gi + 2, gi + 3, gi, gi + 1), (1, 1, 1, 1))
    for base in (O["body_pts"], O["body_pt_vel"]):   # [lE, rE, lK, rK, lF, rF]
        for i in range(6):
            put(range(base + 3 * i, base + 3 * i + 3),
                range(base + 3 * (i ^ 1), base + 3 * (i ^ 1) + 3), (-1, 1, 1))
    ps = (-1, 1, -1, 1, 1)   # x, y, dir cos, dir sin, valid
    pl, pr = O["prev_l"], O["prev_r"]
    put(range(pl, pl + K.PREV_FEAT), range(pr, pr + K.PREV_FEAT), ps)
    put(range(pr, pr + K.PREV_FEAT), range(pl, pl + K.PREV_FEAT), ps)
    # tau_scale lanes: one per axis, shared by both sides — unchanged
    tv = O["tip_vel"]        # [L xyz, R xyz]
    put(range(tv, tv + 3), range(tv + 3, tv + 6), (-1, 1, 1))
    put(range(tv + 3, tv + 6), range(tv, tv + 3), (-1, 1, 1))
    return src, sg


OBS_SRC, OBS_SIGN = _obs_map()


def m_obs(obs):
    """Mirror observation rows (last axis OBS_DIM). Sign flips and lane
    permutations only, so it is exact (and an involution) in any dtype."""
    obs = np.asarray(obs)
    return OBS_SIGN.astype(obs.dtype) * obs[..., OBS_SRC]


def m_obs_torch(obs):
    """`m_obs` for a torch tensor (same device / dtype)."""
    import torch
    src = torch.as_tensor(OBS_SRC, device=obs.device)
    sg = torch.as_tensor(OBS_SIGN, device=obs.device, dtype=obs.dtype)
    return obs.index_select(-1, src) * sg


def m_action_torch(a):
    """`m_action` for a torch tensor of gear ids (last axis 29)."""
    import torch
    sg = torch.as_tensor(JOINT_SIGN < 0, device=a.device)
    swap = torch.as_tensor(GEAR_SWAP, device=a.device, dtype=a.dtype)
    flipped = torch.where(sg, swap[a.long()], a)
    out = torch.empty_like(a)
    out[..., torch.as_tensor(JOINT_PERM, device=a.device)] = flipped
    return out
