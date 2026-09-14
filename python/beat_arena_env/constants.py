"""World and observation constants.

MIRROR of rust/src/lib.rs (+ rust/src/dynamics/model.rs) — the Rust
crate is THE sim implementation. Any rule change lands in lib.rs first
and is mirrored here for tests and tooling. SPEC.md + SPEC-v10.md are
the written contract.

Env v10 (P3, 2026-07-31): the v9 kinematic body is RETIRED — the body
is the SPEC-v10 full humanoid (floating pelvis + 29 actuated revolute
DoF, foot welds, self-collision) integrated at 8 × 1/240 s per 30 Hz
control tick. Actions are 29 gear heads (tau_j = gear_fraction ×
TAU_MAX[j]); per-tick positive mechanical work is TELEMETRY only — the
v10 work price (EFFORT_COEF) and the v7.9 EFFORT_COST table are both
retired; movement is priced by --energy-coef (env v43: Σ|τ|) and --limit-coef
(limit rest). Obs is the SPEC-v10 §5 340-lane
layout. World geometry, judgment, scoring, chart generation are
UNCHANGED from env v7-v9.
"""

import math

DT = 1.0 / 30.0
N_SUB = 8              # physics substeps per control tick (SPEC-v10 §3)
H_SUB = 1.0 / 240.0    # substep length
V_NORM = 6.0           # wrist-velocity obs normalizer (m/s)
V_MIN_CUT = 1.0        # min contact-point speed for a good cut (m/s)
ACC_NORM = 0.3         # accuracy normalizer (m) — real BS value
COS_TOL = 0.5
BLOCK = 0.5            # visual block edge (m) — rendering only since v7
CELL_X = 0.6           # real lane pitch (m); columns at x = -0.9 .. 0.9
# Player height (BS mechanic): the game shifts every object's y by
# (playerHeight - 1.8) / 2 (scoresaber-watch ObjectManager, clamps
# settings 1.2-2.4 / replay -0.2..+0.6), applied to row centers here
# and to real-map wall y in beatsaver.py; convert_scoresaber writes
# the matching initialHeight so the viewer applies the same shift.
# 2026-08-21: briefly 2.4 (env v33 limbo-squat fix, offset +0.3),
# REVERTED to 1.8 (offset 0) same day by user order.
PLAYER_HEIGHT = 1.8
PLAYER_HEIGHT_OFFSET = (PLAYER_HEIGHT - 1.8) / 2          # +0.3
_R = PLAYER_HEIGHT_OFFSET
ROW_Y = (0.85 + _R, 1.40 + _R, 1.90 + _R)   # real rows + height offset
Y_CENTER = 1.40        # obs y normalization: (y - Y_CENTER) / Y_NORM
REWARD_BAD = -0.5      # = miss since 2026-07-20 (real-BS severity parity)
REWARD_MISS = -0.5
REWARD_BOMB = -1.5     # hitting a bomb (passing one is free) — env v7.8

# Movement is priced by --energy-coef (env v43, 2026-09-14, user: the
# applied Σ|τ_j| = Σ|gear fraction_j| × TAU_MAX[j] per tick, N·m — a
# human-like force/energy price; it replaced the v23 fraction
# "activation" price, so a spine full gear costs 600/1.06× a wrist full
# gear; default 7.5e-6 reward per N·m·tick ≈ the old 5e-4 fraction bill
# at ~400 N·m/tick) and --limit-coef (limit rest, N·m·s). Telemetry:
# activation (Σ|fraction|, [total, spine, arms, legs]) and torque
# (Σ|τ|, same split) per tick in the replay ("activation"/"torque"
# fields, record_rev 3/4) and per update in the trainer's `[activation]`
# / `[torque]` lines and dashboard panels.

# --- hitboxes (env v7 — real BS, axis-aligned, NOT the visual cube) ----------
GOOD_HALF_X = 0.4
GOOD_HALF_Y = 0.25
GOOD_HALF_Y_DOT = 0.4
GOOD_HALF_Y_LINK = 0.1
GOOD_HALF_Z = 0.5
GOOD_Z_SHIFT = -0.25          # box center = note_z + GOOD_Z_SHIFT
BAD_HALF = 0.175
BAD_HALF_Y_LINK = 0.05
BOMB_RADIUS = 0.18

SABER_LEN = 1.0       # m (real Beat Saber); blade = hand frame +z
# env v42 (2026-09-14, user-directed) "hilt grip": the hand holds the
# HANDLE, not the blade's base. The 0.16 m handle (the real Saber.fbx
# "Handle", viz/app models.js) spans [0, HANDLE_LEN] along the hand
# frame +z from the wrist origin; the cutting blade spans
# [HANDLE_LEN, HANDLE_LEN + SABER_LEN]. The handle does not cut. Obs
# (wrist pos + blade dir) and actions unchanged; every pre-v42
# checkpoint plays a different game on this geometry.
HANDLE_LEN = 0.16

# --- v10 body (SPEC-v10 §1; mirror of rust/src/dynamics/model.rs) ------------
N_JOINTS = 29          # actuated DoF = action heads (was 14 = 2×7 in v9)
NV = 35                # velocity coords (root 6 + 29)
NQPOS = 36             # position coords (root pos 3 + quat 4 + 29)
ROOT_POS0 = (0.0, 0.94, -0.65)   # reference root (pelvis) position
HEAD0 = (0.0, 1.70, -0.65)       # reference head center (v9 HEAD verbatim)
SHOULDER0 = ((-0.2, 1.47, -0.65), (0.2, 1.47, -0.65))  # reference anchors

# actuated order (SPEC-v10 §1.4) — the action/q/omega index contract
JOINT_NAMES = (
    "spine_pitch", "spine_yaw", "spine_roll",
    "l_arm_pitch", "l_arm_yaw", "l_arm_twist", "l_elbow",
    "l_wrist_roll", "l_wrist_bend", "l_wrist_dev",
    "r_arm_pitch", "r_arm_yaw", "r_arm_twist", "r_elbow",
    "r_wrist_roll", "r_wrist_bend", "r_wrist_dev",
    "l_hip_pitch", "l_hip_yaw", "l_hip_roll", "l_knee",
    "l_ankle_pitch", "l_ankle_roll",
    "r_hip_pitch", "r_hip_yaw", "r_hip_roll", "r_knee",
    "r_ankle_pitch", "r_ankle_roll",
)
A_SPINE = 0            # block starts in the actuated order
A_L_ARM = 3
A_R_ARM = 10
A_L_LEG = 17
A_R_LEG = 23

_D = math.radians      # limits below are the SPEC-v10 §1.4 literals
JOINT_LIMITS = (
    (_D(-20), _D(45)), (_D(-40), _D(40)), (_D(-30), _D(30)),          # spine
    (_D(-90), _D(65)), (_D(-100), _D(45)), (_D(-90), _D(70)),         # l arm
    (0.0, _D(150)), (0.0, math.pi), (_D(-95), _D(95)), (_D(-30), _D(15)),
    (_D(-90), _D(65)), (_D(-45), _D(100)), (_D(-70), _D(90)),         # r arm
    (0.0, _D(150)), (-math.pi, 0.0), (_D(-95), _D(95)), (_D(-15), _D(30)),
    (_D(-20), _D(120)), (_D(-45), _D(40)), (_D(-45), _D(30)),         # l leg
    (0.0, _D(150)), (_D(-50), _D(35)), (_D(-20), _D(20)),
    (_D(-20), _D(120)), (_D(-40), _D(45)), (_D(-30), _D(45)),         # r leg
    (0.0, _D(150)), (_D(-50), _D(35)), (_D(-20), _D(20)),
)
TAU_MAX = (
    600.00, 239.95, 375.00,
    85.38, 85.38, 19.45, 19.13, 1.06, 1.06, 1.06,
    85.38, 85.38, 19.45, 19.13, 1.06, 1.06, 1.06,
    262.50, 75.00, 135.00, 150.00, 135.00, 37.50,
    262.50, 75.00, 135.00, 150.00, 135.00, 37.50,
)
OMEGA_CAP = (16.0,) * 7 + (50.0,) * 3 + (16.0,) * 4 + (50.0,) * 3 + (16.0,) * 12

# §19b root-velocity envelope (dynamics/step.rs ROOT_W_MAX / ROOT_V_MAX,
# mirrored in the .cu): a kinematic consequence of the foot welds + the
# ω caps, applied both BEFORE the step-10 integration (counted, §23) and
# after step 11. Bounds ‖Δp‖ ≤ H_SUB·ROOT_V_MAX per substep.
ROOT_W_MAX = 96.0
ROOT_V_MAX = 48.0

# reference (neutral) root position — dynamics/model.rs base_pos0
BASE_POS0 = (0.0, 0.94, -0.65)
# one-way joints (elbow/knee): obs codes (2q - hi)/hi; wrist rolls wrap:
# obs codes cos/sin (2 lanes); everything else q / max(|lo|, |hi|)
ONE_WAY_JOINTS = (6, 13, 20, 26)
WRAP_JOINTS = (7, 14)

# env v20 (user, 2026-08-16): episode-start ARM pose — the real-player
# ready stance (arms hang at pitch -80 deg, forearms cocked at elbow
# 100 deg, wrists tilted forward at bend 70 deg) replaces the zombie
# arms-forward zero pose at every episode reset. (§1.4 indices; exact
# f64 radians, mirrored verbatim in lib.rs Q_START_ARM and the .cu.)
Q_START_ARM = ((3, -1.3962634015954636), (6, 1.7453292519943295),
               (8, 1.2217304763960306), (10, -1.3962634015954636),
               (13, 1.7453292519943295), (15, 1.2217304763960306))
# env v21 (user, 2026-08-16): wide stance — hip_roll splays the legs
# +-8 deg, ankle_roll counter-rolls so the soles stay flat (verified:
# fk_body foot up-vector exactly (0,1,0)). CONTACT MODE ONLY (weld
# mode's feet are welded at the v10 reference). Exact f64 radians,
# mirrored verbatim in lib.rs Q_START_LEG and the .cu.
Q_START_LEG = ((19, -0.13962634015954636), (22, 0.13962634015954636),
               (25, 0.13962634015954636), (28, -0.13962634015954636))
# The stance's per-side foot X (fk_body of the v21 start pose; the
# root ground-drop does not move X): left foot -> -FOOT_X0, right ->
# +FOOT_X0 — the per-foot targets of the --foot-x-coef penalty.
FOOT_X0 = 0.22108059783525694


def q_start(foot_contact=False):
    """The 29-joint episode-start pose vector: env v20 arm ready
    stance, plus (contact mode only) the env v21 wide leg stance."""
    q = [0.0] * N_JOINTS
    for j, v in Q_START_ARM:
        q[j] = v
    if foot_contact:
        for j, v in Q_START_LEG:
            q[j] = v
    return q


# --- action gears (env v8, unchanged tables — v10 scales TORQUE) -------------
# Per-joint action id -> signed torque fraction of TAU_MAX. Id 0 is
# coast and ids 1/2 are +full/-full at EVERY K (the ternary era is the
# K=3 prefix); finer gears append as +/- pairs in decreasing magnitude.
GEAR_MAGS = {3: (1.0,), 5: (1.0, 0.25), 7: (1.0, 1.0 / 3.0, 0.125)}


def gear_table(n_gears):
    """Signed fraction per action id, length n_gears: (0, +1, -1, ...)."""
    mags = GEAR_MAGS[n_gears]
    out = [0.0]
    for m in mags:
        out += [m, -m]
    return tuple(out)


# --- look-at-player (env v7.6; v10 head anchor is the LIVE FK head) ----------
LOOK_VERT = 0.8             # verticalLookStrength (head.y toward note.y)
LOOK_DIST = 8.0             # full head-facing at z=0, none beyond 8 m

N_BLADE_SAMPLES = 8   # contact-point/velocity samples along the blade
# env v10 live gates (SPEC-v10 §4): the broad-phase reach and the miss
# boundary are POSE FUNCTIONS now —
#   reach_z(t) = max_hand(shoulder_z(t)) + REACH_PAST_SHOULDER
#   miss when note z < head_z(t)
# reproducing v9's REACH_Z 1.75 and -Z_MISS -0.65 at the reference pose.
REACH_PAST_SHOULDER = 0.65 + HANDLE_LEN + 1.0 + 0.75   # v42: + the handle
REACH_Z0 = SHOULDER0[0][2] + REACH_PAST_SHOULDER    # 1.91 at reference (1.75 pre-v42)
Z_MISS0 = -HEAD0[2]                                 # 0.65 at reference

# --- scoring (real Beat Saber 115 points per note) ---------------------------
PRE_FULL = math.radians(100.0)   # full pre-swing arc
POST_FULL = math.radians(60.0)   # full follow-through arc
PTS_PRE, PTS_POST, PTS_ACC = 70, 30, 15   # component maxima (integers)
SCORE_MAX = 115.0
CHAIN_HEAD_MAX = 85.0            # 70 pre + 15 acc, no follow-through
CHAIN_LINK_PTS = 20.0            # flat, any direction, speed still required
SWING_BREAK = 0.005   # rad/tick below which the swing chain breaks
FOLLOW_TIMEOUT = 15   # ticks a follow-through may keep accumulating

# --- object kinds (chart + obs) ----------------------------------------------
KIND_NOTE = 0
KIND_BOMB = 1
KIND_CHAIN_HEAD = 2
KIND_CHAIN_LINK = 3

# --- obs layout v17 (spec §12; offsets DERIVED from the block table) ---------
# env v17: the slots are a SHARED entity window (notes + walls),
# nearest-first by 3D head distance; per-token = the v9 note lanes then
# [is_wall, width, height, length]. Behind-the-head entities are
# excluded. 16 -> 24 in env v18 (2026-08-16, user decision): the
# eviction study measured 3.4%% of scoreable notes crowded out of the
# 16-window during their critical pre-hit second (54%% recovered at 24;
# dense-stream maps like MARENOL ran 76%% blind).
N_NOTE_SLOTS = 24
NOTE_FEAT = 18
# env v17 wall reward/geometry literals (spec §12a/b)
WALL_PEN = -1.5         # ONCE, on the head-entry tick (2026-08-22: -0.5 -> -1.5, bomb parity per user; was -0.05/tick pre-2026-08-20)
WALL_FRONT_OFF = 0.25   # viewer front-face offset behind the note plane (m)
# env v19 (2026-08-16): +1 lane — remaining follow-through credit,
# Σ over the hand's live trackers of (1 − min(acc/POST_FULL, 1)),
# clipped to 2 (fresh single-note tracker = 1.0; idle/broken/capped
# all read 0 = "swinging earns nothing"). Closes the post-swing
# credit-assignment hole behind the flail bug (user diagnosis).
HAND_FEAT = 11        # wrist pos(3), wrist vel(3), blade dir(3), meter, post_rem
Q_LANES = N_JOINTS + len(WRAP_JOINTS)   # 31: wrist rolls code as cos/sin

_OBS_BLOCKS = (
    ("root_pos", 3),          # x/1, (y-0.94)/0.75, (z+0.65)/1
    ("root_rot6", 6),         # [R00,R10,R20, R01,R11,R21] world-from-body
    ("root_vlin", 3),         # /6
    ("root_vang", 3),         # world, /16
    ("q", Q_LANES),           # per-joint normalized angle, §1.4 order
    ("qd", N_JOINTS),         # qdot_j / OMEGA_CAP[j]
    ("hand_l", HAND_FEAT),    # v9 derived-hand semantics on the v10 FK
    ("hand_r", HAND_FEAT),
    ("head", 3),              # x/1, (y-1.70)/0.75, (z+0.65)/1
    ("contact", 2),           # contact bits [L, R]
    ("notes", N_NOTE_SLOTS * NOTE_FEAT),   # v17 shared entity block
    ("note_mask", N_NOTE_SLOTS),
    # env v12 (unilateral-contact spec §7): per-foot ground-contact
    # lanes [L_heel, L_toe, R_heel, R_toe] — tick-summed normal impulse
    # / GROUND_IMP_NORM; all-zero in weld mode
    ("ground_imp", 4),
    # env v13 (contact spec §8): world positions of [l_elbow, r_elbow,
    # l_knee, r_knee, l_foot, r_foot], head-style normalization
    # (elbows centered on Y_CENTER, knees on 0.50, feet on 0.07)
    ("body_pts", 18),
    # env v24: the same 6 points' world VELOCITIES — wrist_vel-style
    # finite difference across the tick, /V_NORM; zero at reset and
    # after a set_state refresh. Foot world velocity is the slip signal.
    ("body_pt_vel", 18),
)
OBS_OFF = {}
_off = 0
for _name, _len in _OBS_BLOCKS:
    OBS_OFF[_name] = _off
    _off += _len
OBS_DIM = _off
OFF_ROOT_POS = OBS_OFF["root_pos"]      # 0
OFF_ROOT_ROT = OBS_OFF["root_rot6"]     # 3
OFF_ROOT_VLIN = OBS_OFF["root_vlin"]    # 9
OFF_ROOT_VANG = OBS_OFF["root_vang"]    # 12
OFF_Q = OBS_OFF["q"]                    # 15
OFF_QD = OBS_OFF["qd"]                  # 46
OFF_HAND_L = OBS_OFF["hand_l"]          # 75
OFF_HAND_R = OBS_OFF["hand_r"]          # 86 (env v19)
OFF_HEAD = OBS_OFF["head"]              # 97 (env v19)
OFF_CONTACT = OBS_OFF["contact"]        # 100 (env v19)
OFF_NOTES = OBS_OFF["notes"]            # 102 (env v19)
OFF_NOTE_MASK = OBS_OFF["note_mask"]    # 534 (env v19)
OFF_GROUND_IMP = OBS_OFF["ground_imp"]  # 558 (env v19)
OFF_BODY_PTS = OBS_OFF["body_pts"]      # 562 (env v19)
OFF_BODY_PT_VEL = OBS_OFF["body_pt_vel"]  # 580 (env v24)
assert OBS_DIM == 598, "spec §12 (env v24 body_pt_vel lanes): OBS_DIM must be 598"
KNEE_Y0 = 0.50   # contact spec §8 normalization centers
FOOT_Y0 = 0.07
# ≈ one tick of full-body weight (M_TOTAL*g*DT = 24.53 N*s), the spec
# literal both engines share (contact spec §7)
GROUND_IMP_NORM = 24.5
# contact spec §7: per-tick survival bonus, contact mode only
ALIVE_BONUS = 0.002

X_NORM, Y_NORM = 1.0, 0.75
Z_SCALE = 8.0
Z_CLIP = (-3.0, 12.0)
# reference body plane (obs centering; contact spec §10: also the zero
# point of the v15 head-anchored note-track offset head_z - PLANE_Z0)
PLANE_Z0 = -0.65
ROOT_VLIN_NORM = 6.0
ROOT_VANG_NORM = 16.0


def q_lane(j):
    """First obs lane of joint j inside the q block (wrist rolls own 2)."""
    lane = OFF_Q
    for k in range(j):
        lane += 2 if k in WRAP_JOINTS else 1
    return lane


def q_normalizer(j):
    """Obs normalizer of joint j per the SPEC-v10 §5 rule (None for the
    cos/sin wrist rolls; one-way joints return ('one_way', hi))."""
    if j in WRAP_JOINTS:
        return None
    lo, hi = JOINT_LIMITS[j]
    if j in ONE_WAY_JOINTS:
        return ("one_way", hi)
    return max(abs(lo), abs(hi))


# event codes returned by the Rust sim's step(); "post" = follow-through
# resolution for an already-counted good cut (carries points only).
# env v17 (spec §12b): "wall" = head-in-wall drain tick (NO combo
# effect), "wall_enter" = the rising edge — a COMBO BREAK; >= 1 tick
# out of walls re-arms it. Both carry note id u32::MAX (no note).
EVENT_REASONS = {0: "ok", 1: "slow", 2: "angle", 3: "wrong_hand", 4: "miss",
                 5: "post", 6: "bomb", 7: "wall", 8: "wall_enter"}
# reasons that consume a scoreable note (not follow-throughs / bombs)
NOTE_OUTCOMES = ("ok", "slow", "angle", "wrong_hand", "miss")


def col_x(col):
    """World x of a column center (0..3 -> -0.9 .. 0.9)."""
    return (col - 1.5) * CELL_X


def note_center(n):
    """In-plane (x, y) of an object; explicit x/y win over col/row."""
    x = n.x if getattr(n, "x", None) is not None else col_x(n.col)
    y = n.y if getattr(n, "y", None) is not None else ROW_Y[n.row]
    return (x, y)


# =============================================================================
# v9-LEGACY SECTION (retired with the kinematic body at P3, 2026-07-31).
# Kept ONLY because kinematics.py (v9 arm FK/IK mirror — P6 rebuilds the
# bot on the dynamic body) and scripted.py (v9 bot, broken against the
# v10 sim API, P6 re-authors) still import these names. Nothing in the
# live sim reads them. DO NOT use in new code.
# =============================================================================
DRAG = 0.02           # v9 per-tick velocity decay (now physical damping b)
EFFORT_COST = (2e-4, 2e-4, 2e-4, 1.3e-4, 0.7e-4, 0.7e-4, 0.7e-4)  # v7.9 table
SHOULDER = SHOULDER0
L_UPPER = 0.35        # m
L_FORE = 0.30         # m
HEAD = HEAD0
N_HAND_JOINTS = 7     # v9 per-hand joint count (pitch..dev)
ALPHA_ARM, OMEGA_ARM = 160.0, 16.0
# scripted-harness wrist model — tracks the TRUE engine values so
# scripted swings land as designed (a too-slow model overshoots):
# alpha = tau_max / (armature + I_hand + m_hand*d^2)
#       = 1.06 / (0.0000266 + 0.000264 + 0.35*0.06^2) ~ 684 rad/s^2
# (2026-08-18 fist-only hand); omega = the wrist cap (50).
ALPHA_WRIST, OMEGA_WRIST = 684.0, 50.0
JOINT_ALPHA = (ALPHA_ARM,) * 4 + (ALPHA_WRIST,) * 3
JOINT_OMEGA = (OMEGA_ARM,) * 4 + (OMEGA_WRIST,) * 3
PITCH_LIM = math.radians(90.0)
PITCH_LO = math.radians(-90.0)
PITCH_HI = math.radians(65.0)
YAW_LO = (math.radians(-100.0), math.radians(-45.0))    # (left, right)
YAW_HI = (math.radians(45.0), math.radians(100.0))
TWIST_LIM = math.radians(90.0)
TWIST_LO = (math.radians(-90.0), math.radians(-70.0))   # (left, right)
TWIST_HI = (math.radians(70.0), math.radians(90.0))
ELBOW_HI = math.radians(150.0)
ROLL_LO = (0.0, math.radians(-180.0))                   # (left, right)
ROLL_HI = (math.radians(180.0), 0.0)
BEND_LIM = math.radians(95.0)
DEV_LIM = math.radians(30.0)
DEV_LO = (math.radians(-30.0), math.radians(-15.0))     # (left, right)
DEV_HI = (math.radians(15.0), math.radians(30.0))
REACH_Z = 1.75        # v9 constant gate (now live: REACH_PAST_SHOULDER)
Z_MISS = 0.65         # v9 constant miss boundary (now live: head_z(t))


def joint_limits(hand):
    """v9-LEGACY: [(lo, hi)] x 7 for one hand, in v9 joint order."""
    return [(PITCH_LO, PITCH_HI), (YAW_LO[hand], YAW_HI[hand]),
            (TWIST_LO[hand], TWIST_HI[hand]), (0.0, ELBOW_HI),
            (ROLL_LO[hand], ROLL_HI[hand]), (-BEND_LIM, BEND_LIM),
            (DEV_LO[hand], DEV_HI[hand])]

# env v38 crouch-rich start ladder (user, 2026-08-24): (spine_p,
# hip_p, knee, ankle_p) f64 radians, mirrored VERBATIM in the .cu
# CROUCH_ROWS (parity-bound literals). Physics-validated squat
# synergy; root y 0.921 -> 0.699 over rows 1-7. Reset lerps adjacent
# rows for continuous depth, jitter applies on top, sole-corner drop
# grounds the pose. 2026-09-06 (user): rows 8-10 (knee 1.7/1.9/2.1)
# extend the ladder to head 1.304/1.227/1.064 m — the deep end clears
# the 1.20 m forced-duck wall bottom (the T=64 duck study found the
# policy never crouches; these spawns put crouched states in the data).
# Rows 8-10 lean the spine (c2 0.25-0.3) with chest tilt capped at 32
# deg (8 deg under the TILT termination); balance under the probe
# controller (sup 9-11 / coast 4-8 of 60/40) equals the shipped knee-1.5
# row (11/9), which is the effective acceptance rule (the 08-24 rows
# measure 11-42 supported today, not the >= 55 the generator once
# claimed). Regenerate: gen_crouch_table.py.
CROUCH_START_ROWS = (
    (-0.03, 0.10499999999999998, 0.3, 0.195),
    (-0.05, 0.325, 0.5, 0.175),
    (0.0, 0.45499999999999996, 0.7, 0.24499999999999997),
    (0.09000000000000001, 0.585, 0.9, 0.315),
    (0.0, 0.605, 1.1, 0.49500000000000005),
    (0.0, 0.6891347618019847, 1.3, 0.6108652381980153),
    (0.15000000000000002, 0.8891347618019847, 1.5, 0.6108652381980153),
    (0.51, 1.0891347618019847, 1.7, 0.6108652381980153),
    (0.475, 1.3399999999999999, 1.9, 0.56),
    (0.525, 1.7850000000000001, 2.1, 0.315),
)
