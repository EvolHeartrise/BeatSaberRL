"""World and observation constants.

MIRROR of rust/src/lib.rs — the Rust crate is THE sim implementation
(single source of truth since 2026-07-20; the Python sim was deleted to
cut maintenance surface). Any rule change lands in lib.rs first and is
mirrored here for the scripted bot, tests, and tooling. SPEC.md remains
the written contract.
"""

import math

DT = 1.0 / 30.0
ACCEL = 30.0          # m/s^2
DRAG = 0.02           # per-tick multiplicative velocity decay
V_MAX = 6.0           # m/s
V_MIN_CUT = 1.0       # min contact-point speed for a good cut (m/s)
CUT_RADIUS = 0.28
COS_TOL = 0.5
CELL = 0.5            # lane pitch (m)
BLOCK = 0.4           # block edge (m)
POS_LO = (-1.5, -1.25)
POS_HI = (1.5, 1.25)
REWARD_BAD = -0.5   # = miss since 2026-07-20 (real-BS severity parity)
REWARD_MISS = -0.5

# --- blade sabers (env v5, 2026-07-21) ---------------------------------------
SABER_LEN = 1.0       # m (real Beat Saber)
ALPHA_ROT = 200.0     # rad/s^2, both angular DOFs (azimuth + tilt)
OMEGA_MAX = 20.0      # rad/s
# env v5.1 anatomical wrist limits: azimuth per hand from the palm-down
# neutral (ready stance, blade up, 90 deg): right 90 CCW / 190 CW ->
# [-100, 180] deg; mirrored left [0, 280]. Tilt +-95 deg from middle.
THETA_LO = (0.0, math.radians(-100.0))          # (left, right)
THETA_HI = (math.radians(280.0), math.pi)
PHI_LIM = math.radians(95.0)
N_BLADE_SAMPLES = 8
REACH_Z = 1.28        # cut-test gate: max tip z (sin 95 deg) + CUT_RADIUS
Z_MISS = 0.6          # note consumed as missed past this

# --- scoring (real Beat Saber 115 points per note) ---------------------------
PRE_FULL = math.radians(100.0)   # full pre-swing arc
POST_FULL = math.radians(60.0)   # full follow-through arc
PTS_PRE, PTS_POST, PTS_ACC = 70.0, 30.0, 15.0
SCORE_MAX = 115.0
SWING_BREAK = 0.005   # rad/tick below which the swing chain breaks
FOLLOW_TIMEOUT = 15   # ticks a follow-through may keep accumulating

# obs layout v4 (order is a contract; mirrored in policy.py):
# per saber [x, y, vx, vy, cos_th, sin_th, phi, om_th, om_ph, swing_meter]
# per note  [cx, cy, z, vz, is_left, is_right, dir_cos, dir_sin, valid]
N_NOTE_SLOTS = 12
NOTE_FEAT = 9
SABER_FEAT = 10
OFF_NOTES = 2 * SABER_FEAT                             # 20
OFF_NOTE_MASK = OFF_NOTES + N_NOTE_SLOTS * NOTE_FEAT   # 128
OBS_DIM = OFF_NOTE_MASK + N_NOTE_SLOTS                 # 140
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
