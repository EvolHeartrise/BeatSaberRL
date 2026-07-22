"""World and observation constants.

MIRROR of rust/src/lib.rs — the Rust crate is THE sim implementation
(single source of truth since 2026-07-20; the Python sim was deleted to
cut maintenance surface). Any rule change lands in lib.rs first and is
mirrored here for the scripted bot, tests, and tooling. SPEC.md remains
the written contract.
"""

import math

DT = 1.0 / 30.0
DRAG = 0.02           # per-tick multiplicative velocity decay (all joints)
V_NORM = 6.0          # wrist-velocity obs normalizer (m/s)
V_MIN_CUT = 1.0       # min contact-point speed for a good cut (m/s)
CUT_RADIUS = 0.28     # accuracy normalizer (circumscribed circle of a face)
COS_TOL = 0.5
CELL = 0.5            # lane pitch (m)
BLOCK = 0.4           # block edge (m)
# env v5.3: contact = swept blade vs the note's ACTUAL cube, rotated about
# the approach (z) axis by (angle - 270 deg) — a down-swing note is
# axis-aligned, dot notes don't rotate
HALF_BLOCK = BLOCK / 2
POS_LO = (-1.5, -1.25)
POS_HI = (1.5, 1.25)
REWARD_BAD = -0.5   # = miss since 2026-07-20 (real-BS severity parity)
REWARD_MISS = -0.5

# --- upper body (env v6, 2026-07-22, user-designed) --------------------------
# Kinematic chain per hand: fixed shoulder -> upper arm -> elbow (hinge)
# -> forearm -> the v5.2 wrist (roll + bend) -> blade. Joint order per
# hand: (pitch, yaw, twist, elbow, roll, bend); all-zero = arm straight
# forward (+z), blade its extension. See kinematics.py for FK/IK.
SABER_LEN = 1.0       # m (real Beat Saber)
SHOULDER = ((-0.2, 0.45, -0.35), (0.2, 0.45, -0.35))    # (left, right)
L_UPPER = 0.35        # m
L_FORE = 0.30         # m
N_JOINTS = 6
# tiered anatomical dynamics: ponderous shoulder/elbow, snappy wrists
ALPHA_ARM, OMEGA_ARM = 80.0, 8.0        # shoulder pitch/yaw/twist + elbow
ALPHA_WRIST, OMEGA_WRIST = 200.0, 20.0  # roll + bend (v5 values)
JOINT_ALPHA = (ALPHA_ARM,) * 4 + (ALPHA_WRIST,) * 2
JOINT_OMEGA = (OMEGA_ARM,) * 4 + (OMEGA_WRIST,) * 2
PITCH_LIM = math.radians(90.0)
YAW_LO = (math.radians(-100.0), math.radians(-45.0))    # (left, right)
YAW_HI = (math.radians(45.0), math.radians(100.0))
TWIST_LIM = math.radians(90.0)
ELBOW_HI = math.radians(150.0)          # flexion 0..150 deg
ROLL_LO = (math.radians(-90.0), math.radians(-190.0))   # (left, right)
ROLL_HI = (math.radians(190.0), math.radians(90.0))
BEND_LIM = math.radians(95.0)


def joint_limits(hand):
    """[(lo, hi)] x 6 for one hand, in joint order."""
    return [(-PITCH_LIM, PITCH_LIM), (YAW_LO[hand], YAW_HI[hand]),
            (-TWIST_LIM, TWIST_LIM), (0.0, ELBOW_HI),
            (ROLL_LO[hand], ROLL_HI[hand]), (-BEND_LIM, BEND_LIM)]


N_BLADE_SAMPLES = 8
REACH_Z = 1.5         # broad-phase gate: max wrist z + blade + half block
Z_MISS = 0.6          # note consumed as missed past this

# --- scoring (real Beat Saber 115 points per note) ---------------------------
PRE_FULL = math.radians(100.0)   # full pre-swing arc
POST_FULL = math.radians(60.0)   # full follow-through arc
PTS_PRE, PTS_POST, PTS_ACC = 70.0, 30.0, 15.0
SCORE_MAX = 115.0
SWING_BREAK = 0.005   # rad/tick below which the swing chain breaks
FOLLOW_TIMEOUT = 15   # ticks a follow-through may keep accumulating

# obs layout v5 (order is a contract; mirrored in policy.py):
# per hand [pitch_n, yaw_n, twist_n, elbow_n, cos_roll, sin_roll, bend_n,
#           om(6)/JOINT_OMEGA, wrist x/1 y/0.75 z/1, wrist_vel(3)/6,
#           blade dir(3), swing_meter]  -> 23
# per note  [cx, cy, z, vz, is_left, is_right, dir_cos, dir_sin, valid]
N_NOTE_SLOTS = 12
NOTE_FEAT = 9
SABER_FEAT = 23
OFF_NOTES = 2 * SABER_FEAT                             # 46
OFF_NOTE_MASK = OFF_NOTES + N_NOTE_SLOTS * NOTE_FEAT   # 154
OBS_DIM = OFF_NOTE_MASK + N_NOTE_SLOTS                 # 166
X_NORM, Y_NORM = 1.0, 0.75
Z_SCALE = 8.0
Z_CLIP = (-1.0, 12.0)

# event codes returned by the Rust sim's step(); "post" = follow-through
# resolution for an already-counted good cut (carries points only)
EVENT_REASONS = {0: "ok", 1: "slow", 2: "angle", 3: "wrong_hand", 4: "miss",
                 5: "post"}
# reasons that consume a note (everything except follow-through resolutions)
NOTE_OUTCOMES = ("ok", "slow", "angle", "wrong_hand", "miss")


def note_center(n):
    """In-plane (x, y) of a note's cell center; grid centered on origin."""
    return ((n.col - 1.5) * CELL, (n.row - 1.0) * CELL)
