"""World and observation constants.

MIRROR of rust/src/lib.rs (+ rust/src/dynamics/model.rs) — the Rust
crate is THE sim implementation. Any rule change lands in lib.rs first
and is mirrored here for tests and tooling. SPEC.md + SPEC-v10.md are
the written contract.

Env v10 (P3, 2026-07-31): the v9 kinematic body is RETIRED — the body
is the SPEC-v10 full humanoid (floating pelvis + 29 actuated revolute
DoF, foot welds, self-collision) integrated at 8 × 1/240 s per 30 Hz
control tick. Actions are 29 gear heads (tau_j = gear_fraction ×
TAU_MAX[j]); effort is positive mechanical work × EFFORT_COEF (the
v7.9 EFFORT_COST table is gone). Obs is the SPEC-v10 §5 340-lane
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
ROW_Y = (0.85, 1.40, 1.90)   # real at-player row centers (m above floor)
Y_CENTER = 1.40        # obs y normalization: (y - Y_CENTER) / Y_NORM
REWARD_BAD = -0.5      # = miss since 2026-07-20 (real-BS severity parity)
REWARD_MISS = -0.5
REWARD_BOMB = -1.5     # hitting a bomb (passing one is free) — env v7.8

# env v10 effort (SPEC-v10 §4): reward -= EFFORT_COEF * positive
# mechanical work sum_substeps sum_j max(tau_j*qdot_j, 0)*h (post-solve
# substep velocities; tau constant over the tick). 4e-4 per joule —
# derived calibration against the v9 per-swing bill; flagged for P7
# re-tune (the gravity/work economy is a known behavioral break).
EFFORT_COEF = 4e-4

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
    (0.0, _D(150)), (_D(-50), _D(20)), (_D(-20), _D(20)),
    (_D(-20), _D(120)), (_D(-40), _D(45)), (_D(-30), _D(45)),         # r leg
    (0.0, _D(150)), (_D(-50), _D(20)), (_D(-20), _D(20)),
)
TAU_MAX = (
    800.00, 319.93, 500.00,
    113.84, 113.84, 19.45, 19.13, 1.06, 1.06, 1.06,
    113.84, 113.84, 19.45, 19.13, 1.06, 1.06, 1.06,
    484.68, 100.00, 300.00, 250.00, 180.00, 50.00,
    484.68, 100.00, 300.00, 250.00, 180.00, 50.00,
)
OMEGA_CAP = (16.0,) * 7 + (40.0,) * 3 + (16.0,) * 4 + (40.0,) * 3 + (16.0,) * 12

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
REACH_PAST_SHOULDER = 0.65 + 1.0 + 0.75
REACH_Z0 = SHOULDER0[0][2] + REACH_PAST_SHOULDER    # 1.75 at reference
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

# --- obs layout v10 (SPEC-v10 §5; offsets DERIVED from the block table) ------
N_NOTE_SLOTS = 16
NOTE_FEAT = 14
HAND_FEAT = 10        # wrist pos(3), wrist vel(3), blade dir(3), swing meter
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
    ("notes", N_NOTE_SLOTS * NOTE_FEAT),   # v9 NOTE_FEAT verbatim
    ("note_mask", N_NOTE_SLOTS),
    # env v12 (unilateral-contact spec §7): per-foot ground-contact
    # lanes [L_heel, L_toe, R_heel, R_toe] — tick-summed normal impulse
    # / GROUND_IMP_NORM; all-zero in weld mode
    ("ground_imp", 4),
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
OFF_HAND_R = OBS_OFF["hand_r"]          # 85
OFF_HEAD = OBS_OFF["head"]              # 95
OFF_CONTACT = OBS_OFF["contact"]        # 98
OFF_NOTES = OBS_OFF["notes"]            # 100
OFF_NOTE_MASK = OBS_OFF["note_mask"]    # 324
OFF_GROUND_IMP = OBS_OFF["ground_imp"]  # 340
assert OBS_DIM == 344, "contact spec §7: OBS_DIM must be 344"
# ≈ one tick of full-body weight (M_TOTAL*g*DT = 24.53 N*s), the spec
# literal both engines share (contact spec §7)
GROUND_IMP_NORM = 24.5
# contact spec §7: per-tick survival bonus, contact mode only
ALIVE_BONUS = 0.002

X_NORM, Y_NORM = 1.0, 0.75
Z_SCALE = 8.0
Z_CLIP = (-3.0, 12.0)
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
# resolution for an already-counted good cut (carries points only)
EVENT_REASONS = {0: "ok", 1: "slow", 2: "angle", 3: "wrong_hand", 4: "miss",
                 5: "post", 6: "bomb"}
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
ALPHA_WRIST, OMEGA_WRIST = 400.0, 40.0
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
