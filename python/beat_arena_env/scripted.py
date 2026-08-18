"""Scripted reference bot — v10 (P6): gravity-compensated PD-over-IK.

The blade-level PLAN is v5.2-v9 verbatim: for its color's earliest
pending note, plant the wrist so the blade lies in the hit plane
perpendicular to the required cut direction (two-link IK for the four
arm joints), cock the wrist roll up to 130 degrees behind the contact
roll, then whip through (leftover rotation = follow-through). What
changed at v10 is HOW joints are driven: the body is the SPEC-v10
dynamic humanoid under gravity, actions are 29 gear ids scaling
τ_max, so the bot outputs torque, not velocity:

    τ_j = τ_gravity[j] + kp_j·(q_target[j] − q_j) − kd_j·q̇_j

clamped to ±τ_max and mapped to the NEAREST gear (argmin over the
env's gear table of |gear·τ_max − τ|). τ_gravity = the Python
gravity-only RNEA mirror (kinematics.gravity_comp — exact at zero
velocity/acceleration, validated against the potential gradient).
Gains per SPEC-v10 P6 guidance: kp = τ_max/0.5 rad; kd near-critical
from the §1.5 reference inertias, ceilinged at I_ref/DT for 30 Hz
discrete stability (the torque is held over a full control tick).
Whip ticks override the roll gear to FULL in the swing sense — the
wrist-roll axis is gravity-free by geometry (the hand COM sits on it),
so the v9 bang-bang whip carries over cleanly.

Spine and legs PD-hold neutral (their IK "target" is the reference
pose); shoulder anchors are LIVE (fk_body), so arm IK follows the
body's real lean. Stateless per tick; solvability referee, not
optimal.
"""

from __future__ import annotations

import numpy as np

from .constants import (
    ALPHA_WRIST, DT, KIND_BOMB, KIND_CHAIN_LINK, L_FORE, L_UPPER,
    N_JOINTS, OMEGA_WRIST, SHOULDER, TAU_MAX, A_L_ARM, A_R_ARM,
    gear_table, joint_limits, note_center,
)
from .kinematics import (
    F_L_SHOULDER, F_R_SHOULDER, I_REF, com_of, fk, fk_body, ik_arm,
    stance_comp, wrist_for_blade,
)

CONTACT_S = 0.75          # note passes the blade at this fraction of its length
GATE_LIVE_ARC = True      # whip release on the live remaining arc (iter 12)
POSE_KD_ROLL = True       # bend-dependent wrist-roll kd (iter 12)
ARM_ASSIST = 0.0          # m: whip-phase wrist-goal pull along the cut dir
                          # (swept 0/0.25/0.5 — no gain; off by default)
SWING_ARC = np.radians(130.0)  # ideal wind-up, past the 100 deg full credit
REACH = L_UPPER + L_FORE - 0.02   # usable wrist-reach radius around a shoulder
# env v7 geometry (unchanged): shoulders 0.65 m behind the hit plane —
# plant the wrist at z = -0.25 so the roll-swing sweeps inside the
# elongated good box (note_z - 0.75 .. + 0.25) while the note is near
# the plane.
PLANT_Z = -0.25
# whip THROUGH the contact angle this early: the deep hitbox reaches
# the blade zone ~0.15 s before hit_time (cutting early inside the box
# is what real players do).
EARLY_S = 0.10

# idle pose per hand: arm dropped and out, elbow half bent, blade up
PARK_Q = {
    0: np.array([-0.5, -0.7, 0.0, 0.8]),
    1: np.array([-0.5, 0.7, 0.0, 0.8]),
}

# Stance target (iteration 8): a LOADED CROUCH, not the neutral
# straight-leg pose. Neutral parks knee/hip/ankle on their 0-stops —
# the P2-documented knife edge — and the gear-quantization wobble
# RECTIFIES against the limit rows into a slow topple (measured: the
# neutral hold falls in ~3 s even with correct static feedforward).
# Crouch geometry (pelvis dropped 0.05 m, hip kept above the welded
# ankle, legs in the sagittal plane — the unique consistent closure):
# hip 19.3°, knee 39°, ankle dorsiflexion 19.9°. Every stance joint
# now carries a real load (knee ~40 N·m ≈ 1.3 gear quanta), so the
# sigma-delta dithers around the load instead of ringing around zero.
STANCE_HIP = 0.337
STANCE_KNEE = 0.680
STANCE_ANKLE = 0.347
STANCE_Q = np.zeros(N_JOINTS)
for _base in (17, 23):
    STANCE_Q[_base] = STANCE_HIP        # hip_pitch
    STANCE_Q[_base + 3] = STANCE_KNEE   # knee
    STANCE_Q[_base + 4] = STANCE_ANKLE  # ankle_pitch

# --- PD gains. Tuning iteration 2 (P6): the spec's starting point
# (kp = τ_max/0.5, kd near-critical from I_ref) limit-cycled badly at
# 30 Hz (arm p90 error ~70 deg — τ held a full tick overshoots hard).
# Deadbeat velocity-matching form instead, the v10 analog of the v9
# bot's ω-steering: desired ω = clip(K·err, ±ω_cap), then
#   τ = τ_grav + (I_ref/DT)·(ω_des − ω)
# i.e. kd EXACTLY at the discrete-stability ceiling I_ref/DT and
# kp = K·kd (K = per-tier arrival gain, v9 values: 4/s joints,
# 10/s wrists).
K_JOINT = 4.0             # spine + legs (stance holding)
K_ARM = 12.0              # arm chains incl. wrists (5-seed grid best)
K_GAIN = np.full(N_JOINTS, K_JOINT)
K_GAIN[3:17] = K_ARM
WRIST_J = (7, 8, 9, 14, 15, 16)
# Tuning iterations 3-4 (both measured failures): kd from the §1.5
# I_ref limit-cycled at full gear — I_ref is the FIXED-BASE distal
# subtree inertia, but the live chain recoils (free-base coupling) and
# the legs are the inverted case entirely (the ankle's subtree is the
# 1 kg foot; under the weld it moves the body above). The honest
# number is the APPARENT inertia 1/(M⁻¹+constraints)_jj, measured
# open-loop on the sim itself: τ·DT/Δω of a one-tick quarter-gear
# probe from neutral, min over both directions (conservative — PD-held
# neighbors only raise it). Measured 2026-07-31 (probe in the P6
# session notes): e.g. elbow 0.020 vs I_ref 0.120 (6× recoil), twist
# 0.010 vs 0.122, wrist bend 0.0012 vs 0.0027, hip/ankle roll 26
# (weld-locked lateral), knee 0.23 flexing.
I_APP = np.array([
    0.417, 0.202, 3.200,                                   # spine
    0.164, 0.276, 0.0101, 0.0199, 0.0005, 0.0012, 0.0017,  # arm_l
    0.164, 0.276, 0.0101, 0.0199, 0.0005, 0.0012, 0.0017,  # arm_r
    0.262, 0.259, 26.10, 0.228, 0.975, 26.10,              # leg_l
    0.262, 0.259, 26.10, 0.228, 0.975, 26.10,              # leg_r
])
# Iteration 7: the statics moved into the FEEDFORWARD
# (kinematics.stance_comp — constrained weld statics; the plain
# subtree gravity_comp is blind to the trunk weight, which lives on
# the unactuated root rows, and the bot hip-hinged forward in ~1.5 s).
# With the feedforward carrying the hold, the PD only rejects
# disturbances and keeps the conservative measured-I_app sizing.
KD = 0.7 * I_APP / DT
TAU_MAX_V = np.asarray(TAU_MAX)

# COM-over-ankles balance gains (iteration 9; see scripted_action):
# k ≈ m·g/h·margin for the ~0.9 m inverted pendulum, d for ζ ≈ 0.7.
K_COM = 700.0
D_COM = 300.0
COM_REF_Z = -0.612   # whole-body COM z at the reference pose (probed)
# Pelvis-orientation righting through the hips (iteration 10): COM
# feedback alone balances but lets the pelvis ANGLE drift into a
# backward squat (measured: root pitched −50°, wrists ~0.9 m off
# their plants). stance_comp holds whatever pose the body is in, so
# righting needs no static torque — only gentle, damped PD. Probed
# signs: +hip_pitch τ pitches the pelvis toward +x rotvec;
# +hip_yaw τ yaws it toward −y.
K_PELV = 40.0
D_PELV = 4.0
# Spine and hip PD picks are capped BELOW the full gear (their full
# τ_max — 800/485 N·m held for a tick — slams the light pelvis at
# ~15 rad/s; measured iteration 7). Whip overrides are arm-only and
# unaffected. At gears=3 there is no finer gear, so no mask.
SLAM_J = (0, 1, 2, 17, 18, 19, 23, 24, 25)


def _roll_reps(hand, roll, bend):
    """Both in-range representatives of a blade direction:
    d(roll, bend) = d(roll + pi, -bend), rolls 2*pi-shifted into the
    hand's range. Deterministic, so steering targets never thrash."""
    lo, hi = joint_limits(hand)[4]
    if bend < 0.0:
        roll, bend = roll + np.pi, -bend
    reps = []
    for r0, b0 in ((roll, bend), (roll + np.pi, -bend)):
        for k in (0.0, 2 * np.pi, -2 * np.pi):
            if lo - 1e-9 <= r0 + k <= hi + 1e-9:
                reps.append((float(np.clip(r0 + k, lo, hi)), b0))
                break
    return reps or [(float(np.clip(roll, lo, hi)), bend)]


def _fit_roll(hand, roll, bend):
    """First (preferred) in-range representative — see _roll_reps."""
    return _roll_reps(hand, roll, bend)[0]


def _swing_time(arc):
    """Seconds to sweep `arc` from rest at full wrist accel (+ margin).
    ALPHA_WRIST/OMEGA_WRIST mirror the LIVE engine wrist (fist-only
    hand + cap 50 since 2026-08-18) — see constants.py derivation."""
    if arc <= OMEGA_WRIST ** 2 / (2 * ALPHA_WRIST):
        return float(np.sqrt(2 * arc / ALPHA_WRIST)) + 0.04
    return OMEGA_WRIST / (2 * ALPHA_WRIST) + arc / OMEGA_WRIST + 0.04


def _cut_config(hand, sh, center3, angle):
    """(sense, theta_c, wrist_goal3) — blade perpendicular to the cut
    direction at contact, wrist ON the blade line through the note;
    `sh` is the LIVE shoulder anchor (v10)."""
    plant = np.array([0.0, 0.0, PLANT_Z])
    best = None
    for sense, th_c in ((1.0, angle - np.pi / 2), (-1.0, angle + np.pi / 2)):
        ray = np.array([np.cos(th_c), np.sin(th_c), 0.0])
        s_best, over_best = CONTACT_S, np.inf
        for s in np.linspace(0.3, 0.95, 14):
            over = max(0.0, float(
                np.linalg.norm(center3 - s * ray + plant - sh)) - REACH)
            key = (over, abs(s - CONTACT_S))
            if key < (over_best, abs(s_best - CONTACT_S)):
                s_best, over_best = float(s), over
        if best is None or over_best < best[0]:
            best = (over_best, sense, th_c, center3 - s_best * ray + plant)
    _, sense, th_c, goal = best
    return sense, th_c, goal


def _hand_targets(hand, q7, sh, r_chest, time, target, tgt_idx, plans):
    """(q_target(7), whip_gear or None) for one arm block. q7 =
    [pitch, yaw, twist, elbow, roll, bend, dev] (the v10 arm block IS
    the v9 per-hand order). whip_gear = 1/2 forces full ± roll torque
    this tick (the whip phase).

    All v9 arm math (ik_arm / wrist_for_blade / fk) lives in the
    CHEST frame: world targets are pulled back through the live chest
    rotation `r_chest` and the live shoulder anchor `sh` (iteration
    10 — with the trunk swaying, world-aligned wrist math aimed the
    blade wherever the lean pointed it).

    `plans` (iteration 11): per-hand endgame LATCH. The v9 plan was
    deterministic per note; v10's live-pose inputs made it thrash —
    trunk sway re-picked roll representations every tick and the
    wrist re-cocked BACK AND FORTH through the hitbox while the note
    arrived (measured: angle-fails clustered at ±150-180° = backswing
    catches). Once the endgame starts (t_until inside swing time
    + 0.25 s) the plan freezes for that note."""
    rc = np.asarray(r_chest)
    to_local_dir = lambda d: tuple(rc.T @ np.asarray(d, dtype=float))
    if target is None:
        plans.pop(hand, None)
        r_t, b_t = _fit_roll(hand, *wrist_for_blade(
            hand, list(q7[:4]), to_local_dir((0.0, 1.0, 0.0))))  # blade up
        return np.array([*PARK_Q[hand], r_t, b_t, 0.0]), None

    # endgame latch for this note (discrete choices only — sense, arc
    # and the roll REPRESENTATIVE; IK and the exact roll/bend keep
    # tracking the live pose, or the frozen plan goes stale as the
    # trunk sways)
    plan = plans.get(hand)
    if plan is not None and plan["note"] != tgt_idx:
        plan = None

    center3 = np.array([*note_center(target), 0.0])
    # chain links carry their tangent as `angle`; plain dots default to
    # a down-swing
    if not target.any_dir or target.kind == KIND_CHAIN_LINK:
        angle = target.angle
    else:
        angle = 3 * np.pi / 2
    _, _, goal = _cut_config(hand, sh, center3, angle)
    sh_ref = np.asarray(SHOULDER[hand])
    goal_l = sh_ref + rc.T @ (goal - sh)
    q_arm = ik_arm(hand, goal_l, swivel="back")

    # re-aim off the pose the arm will actually reach: v9 fk() gives
    # the wrist in the chest frame off the reference shoulder; push
    # forward through the live chest pose
    _, w_ref, _ = fk(hand, list(q_arm) + [0.0, 0.0, 0.0])
    w_pred = sh + rc @ (np.asarray(w_ref) - sh_ref)
    th_aim = float(np.arctan2(center3[1] - w_pred[1],
                              center3[0] - w_pred[0]))
    # swing plan (v5.2/v9 verbatim): both wrist representations, both
    # senses; sweep within ~55 deg of the cut direction; rank by roll
    # room (wind-up arc + >= 45 deg follow-through)
    lo, hi = joint_limits(hand)[4]
    err = lambda s: abs(np.angle(np.exp(1j * (angle - th_aim - s * np.pi / 2))))
    reps = _roll_reps(hand, *wrist_for_blade(
        hand, list(q7[:4]),
        to_local_dir((float(np.cos(th_aim)), float(np.sin(th_aim)), 0.0))))
    if plan is not None:
        # latched: keep the frozen sense, follow the rep nearest the
        # frozen roll
        sense = plan["sense"]
        arc = plan["arc"]
        r_c, b_c = min(reps, key=lambda rb: abs(rb[0] - plan["r_c"]))
        plan["r_c"] = r_c
    else:
        best = None
        for r_c, b_c in reps:
            for sense in (1.0, -1.0):
                back = (r_c - lo) if sense > 0 else (hi - r_c)
                fwd = (hi - r_c) if sense > 0 else (r_c - lo)
                full = back >= SWING_ARC and fwd >= np.radians(45.0)
                key = (err(sense) > np.radians(55.0), not full,
                       -min(fwd, np.radians(45.0)), err(sense))
                if best is None or key < best[0]:
                    best = (key, sense, back, r_c, b_c)
        _, sense, back, r_c, b_c = best
        arc = float(np.clip(back - np.radians(5.0), np.radians(20.0),
                            SWING_ARC))

    t_until = target.hit_time - EARLY_S - time
    if plan is None and t_until <= _swing_time(arc) + 0.25:
        plans[hand] = {"note": tgt_idx, "r_c": r_c, "sense": sense,
                       "arc": arc}
    # whip release on the LIVE remaining arc to the contact roll
    # (iteration 12): gating on the PLANNED arc fired fixed-schedule
    # whips while the cock was still ~50-80° incomplete — the sweep
    # crossed the note off-phase (angle fails scattered ±100°+). With
    # the remaining-arc gate an incomplete cock just fires later
    # (shorter, correctly-phased swing).
    if GATE_LIVE_ARC:
        remaining = max(float(sense * (r_c - q7[4])), 0.0)
    else:
        remaining = arc
    p = plans.get(hand)
    whipping = p is not None and p.get("whip")
    if not whipping and t_until > _swing_time(remaining):
        # wind-up: cock the roll `arc` behind the contact roll
        return np.array([*q_arm, r_c - sense * arc, b_c, 0.0]), None
    # whip: FULL roll torque in the swing sense until the note resolves
    # (bend/dev PD-hold their contact values). Once released it stays
    # released (the remaining-arc gate would flicker as the roll
    # closes on r_c). Arm assist: the wrist goal pulls along the cut
    # direction so the arm PD adds sweep speed through the note.
    if p is not None:
        p["whip"] = True
    if ARM_ASSIST:
        assist = goal + ARM_ASSIST * np.array(
            [np.cos(angle), np.sin(angle), 0.0])
        q_arm = ik_arm(hand, sh_ref + rc.T @ (assist - sh), swivel="back")
    return (np.array([*q_arm, r_c, b_c, 0.0]),
            1 if sense > 0 else 2)


def scripted_action(env):
    """29 gear ids (SPEC-v10 §1.4 order) for the current tick."""
    sim = env.sim
    status = sim.status()
    targets = {0: None, 1: None}
    target_idx = {0: None, 1: None}
    for i, n in enumerate(env.chart):
        if n.kind == KIND_BOMB:            # bombs are never targets
            continue
        if status[i] == 0 and targets[n.color] is None:
            targets[n.color] = n
            target_idx[n.color] = i
        if targets[0] is not None and targets[1] is not None:
            break

    q36 = np.asarray(sim.qpos())
    om = np.asarray(sim.omega())
    q = q36[7:]
    time = sim.time
    body = fk_body(q36)
    grav = np.asarray(stance_comp(q36, fk=body))
    _, pos = body

    # target pose: spine neutral, legs in the loaded crouch; arms from
    # IK toward the plan
    q_t = STANCE_Q.copy()
    whips = {}
    from .kinematics import F_CHEST
    r_chest = body[0][F_CHEST]
    plans = getattr(env, "_bot_plans", None)
    if plans is None or sim.tick == 0:
        plans = {}
    env._bot_plans = plans
    for hand, base, fsh in ((0, A_L_ARM, F_L_SHOULDER),
                            (1, A_R_ARM, F_R_SHOULDER)):
        qt7, whip = _hand_targets(hand, q[base:base + 7],
                                  np.asarray(pos[fsh]), r_chest, time,
                                  targets[hand], target_idx[hand], plans)
        q_t[base:base + 7] = qt7
        if whip is not None:
            whips[base + 4] = whip         # the roll joint of this arm

    from .constants import OMEGA_CAP
    om_des = np.clip(K_GAIN * (q_t - q), -np.asarray(OMEGA_CAP),
                     np.asarray(OMEGA_CAP))
    kd = KD.copy()
    # wrist-roll inertia is strongly bend-dependent (hand COM swings
    # off the roll axis): I ≈ 0.00046 + 0.0022·sin²(bend) — 0.0005
    # straight (the probed I_app) to 0.0027 bent. Fixed straight-arm
    # kd under-torqued the wind-up 5× at bend ≈ 90° (measured: cocking
    # ~84° incomplete at whip start). Pose-aware kd for the two rolls.
    if POSE_KD_ROLL:
        for roll_j, bend_j in ((7, 8), (14, 15)):
            i_roll = 0.00046 + 0.0022 * float(np.sin(q[bend_j])) ** 2
            kd[roll_j] = 0.7 * i_roll / DT
    tau = grav + kd * (om_des - om)

    # Balance (iteration 9): COM-over-ankles feedback — the classic
    # ankle strategy. The ankles are the GENTLE actuators (a 1/8-gear
    # kick rings the shank ~0.8 rad/s vs the hips' ~8 — measured), and
    # the COM is mass-weighted so the light-pelvis wobble barely moves
    # it (natural low-pass). Probed signs: +ankle_pitch τ pushes the
    # COM toward +z, +ankle_roll τ toward +x.
    #   τ_ankle −= (k·e + d·ė)/2 per ankle, e = com − reference.
    c = com_of(q36, fk=body)
    prev = getattr(env, "_bot_com", None)
    if prev is None or sim.tick == 0:
        prev = c
    vcom = (c - prev) / DT
    env._bot_com = c
    e_z = c[2] - COM_REF_Z
    e_x = c[0] - 0.0
    f_pitch = -0.5 * (K_COM * e_z + D_COM * vcom[2])
    f_roll = -0.5 * (K_COM * e_x + D_COM * vcom[0])
    for base in (17, 23):
        tau[base + 4] += f_pitch     # ankle_pitch
        tau[base + 5] += f_roll      # ankle_roll

    # pelvis-orientation righting (iteration 10; gains above)
    w, vx, vy, vz = q36[3:7]
    vn = float(np.hypot(np.hypot(vx, vy), vz))
    if vn > 1e-9:
        ang = 2.0 * float(np.arctan2(vn, w))
        if ang > np.pi:
            ang -= 2.0 * np.pi
        theta = np.array([vx, vy, vz]) * (ang / vn)
    else:
        theta = np.zeros(3)
    w_root = np.asarray(sim.twist())[3:]
    r_pitch = -0.5 * (K_PELV * theta[0] + D_PELV * w_root[0])
    r_yaw = +0.5 * (K_PELV * theta[1] + D_PELV * w_root[1])
    for base in (17, 23):
        tau[base] += r_pitch         # hip_pitch
        tau[base + 1] += r_yaw       # hip_yaw
    tau = np.clip(tau, -TAU_MAX_V, TAU_MAX_V)

    # Gear choice with sigma-delta (error-feedback) quantization
    # (iteration 6): nearest-gear alone has a ±0.125·τ_max deadband —
    # sustained sub-quantum torques (gravity holds, small stance
    # corrections) never fire. The accumulator carries the rounding
    # error so they materialize as duty-cycled gears; clamped to one
    # quantum (anti-windup). Deterministic; state lives on the env.
    fracs = np.asarray(gear_table(env.gears))          # (K,)
    cand = fracs[None, :] * TAU_MAX_V[:, None]         # (29, K)
    dist_bias = np.zeros_like(cand)
    if env.gears > 3:
        # spine/hips: full gear excluded (pelvis slam — see SLAM_J)
        full = np.abs(fracs) == 1.0
        for j in SLAM_J:
            dist_bias[j, full] = np.inf
    accum = getattr(env, "_bot_accum", None)
    if accum is None or env.sim.tick == 0:
        accum = np.zeros(N_JOINTS)
    want = tau + accum
    acts = np.argmin(np.abs(cand - want[:, None]) + dist_bias, axis=1)
    applied = cand[np.arange(N_JOINTS), acts]
    quantum = np.abs(fracs[fracs != 0.0]).min() * TAU_MAX_V
    env._bot_accum = np.clip(accum + tau - applied, -quantum, quantum)
    for j, g in whips.items():
        acts[j] = g
    return tuple(int(a) for a in acts)
