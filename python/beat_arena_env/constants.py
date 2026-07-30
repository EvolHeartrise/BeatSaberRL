"""World and observation constants.

MIRROR of rust/src/lib.rs — the Rust crate is THE sim implementation
(single source of truth since 2026-07-20; the Python sim was deleted to
cut maintenance surface). Any rule change lands in lib.rs first and is
mirrored here for the scripted bot, tests, and tooling. SPEC.md remains
the written contract.

Env v7 (2026-07-22, research-driven): real Beat Saber geometry and
judgment, sourced from the ArcViewer + BeatLeader code inventory
(../research/2026-07-22-bs-mechanics-inventory.md). Floor-based world
(y = 0 is the floor), 0.6 m lanes, real non-uniform row heights, real
axis-aligned hitboxes (good box elongated toward the player, small
wrong-color box), swept-surface saber collision, integer-rounded
115-point scoring with the real 0.3 m accuracy normalizer, bombs,
chains (head 85 / links flat 20), and arc pre/post-swing grants.
"""

import math

DT = 1.0 / 30.0
DRAG = 0.02           # per-tick multiplicative velocity decay (all joints)
V_NORM = 6.0          # wrist-velocity obs normalizer (m/s)
V_MIN_CUT = 1.0       # min contact-point speed for a good cut (m/s)
ACC_NORM = 0.3        # accuracy normalizer (m) — real BS value (was 0.28)
COS_TOL = 0.5
BLOCK = 0.5           # visual block edge (m) — rendering only since v7;
# 0.5 = the real game's cube (matched to the ScoreSaber viewer 2026-07-30)
CELL_X = 0.6          # real lane pitch (m); columns at x = -0.9 .. 0.9
ROW_Y = (0.85, 1.40, 1.90)   # real at-player row centers (m above floor)
Y_CENTER = 1.40       # obs y normalization: (y - Y_CENTER) / Y_NORM
REWARD_BAD = -0.5     # = miss since 2026-07-20 (real-BS severity parity)
REWARD_MISS = -0.5
REWARD_BOMB = -1.5    # hitting a bomb (passing one is free) — env v7.8,
# user-priced: proxy for the unmodeled combo-multiplier break; -0.5 made
# bombs the cheapest gym error (12%-eaten equilibrium at any budget)
# env v7.9 (user-designed): tiered effort cost per non-coast joint action
# per tick (metabolic proxy — proximal joints move more mass), scaled by
# the env's effort_coef. Per-hand joint order (pitch..bend):
EFFORT_COST = (2e-4, 2e-4, 2e-4, 1.3e-4, 0.7e-4, 0.7e-4)

# --- hitboxes (env v7 — real BS, axis-aligned, NOT the visual cube) ----------
# Good-cut box (matching saber): 0.8 wide x 0.5 tall x 1.0 deep, center
# shifted 0.25 m toward the player (box spans z in [note_z - 0.75,
# note_z + 0.25]) — early and off-center swings still connect. Dot notes
# are 0.8 tall (modern "new dots"); chain links 0.2 tall. The wrong-color
# saber only triggers the SMALL box (0.35^3; links 0.35 x 0.1 x 0.35) —
# smaller than the visual cube, so corner grazes with the off hand are
# forgiven. Hitboxes do NOT rotate with the cut direction (v5.3's rotated
# collision cube is retired — the real game is axis-aligned and forgiving).
GOOD_HALF_X = 0.4
GOOD_HALF_Y = 0.25
GOOD_HALF_Y_DOT = 0.4
GOOD_HALF_Y_LINK = 0.1
GOOD_HALF_Z = 0.5
GOOD_Z_SHIFT = -0.25          # box center = note_z + GOOD_Z_SHIFT
BAD_HALF = 0.175
BAD_HALF_Y_LINK = 0.05
BOMB_RADIUS = 0.18

# --- upper body (env v6 body, v7 placement) ----------------------------------
# Kinematic chain per hand: fixed shoulder -> upper arm -> elbow (hinge)
# -> forearm -> the v5.2 wrist (roll + bend) -> blade. Joint order per
# hand: (pitch, yaw, twist, elbow, roll, bend); all-zero = arm straight
# forward (+z), blade its extension. See kinematics.py for FK/IK.
# v7 placement: a 1.8 m player — shoulders 1.47 m above the floor,
# 0.65 m behind the hit plane (head at the real BS cut-plane distance).
SABER_LEN = 1.0       # m (real Beat Saber)
SHOULDER = ((-0.2, 1.47, -0.65), (0.2, 1.47, -0.65))    # (left, right)
L_UPPER = 0.35        # m
L_FORE = 0.30         # m
N_JOINTS = 6
# tiered anatomical dynamics: ponderous shoulder/elbow, snappy wrists
ALPHA_ARM, OMEGA_ARM = 80.0, 8.0        # shoulder pitch/yaw/twist + elbow
ALPHA_WRIST, OMEGA_WRIST = 200.0, 20.0  # roll + bend (v5 values)
JOINT_ALPHA = (ALPHA_ARM,) * 4 + (ALPHA_WRIST,) * 2
JOINT_OMEGA = (OMEGA_ARM,) * 4 + (OMEGA_WRIST,) * 2
# env v7.2 (user-specified): shoulder pitch capped at +65 deg (arm up) —
# straight-overhead arms are gone; -90 (straight down) unchanged.
PITCH_LIM = math.radians(90.0)          # obs normalizer only
PITCH_LO = math.radians(-90.0)
PITCH_HI = math.radians(65.0)
YAW_LO = (math.radians(-100.0), math.radians(-45.0))    # (left, right)
YAW_HI = (math.radians(45.0), math.radians(100.0))
# env v7.1 (user-specified, via the pose tool): the v5-era wrist roll
# range was compensation for missing joints; with the full arm chain it
# let training cheat. Right roll [-180, 0], twist [-70, 90] (outward
# humeral rotation keeps its full range, inward is capped); left
# mirrored. Big swing arcs must now come from the arm, not the wrist.
TWIST_LIM = math.radians(90.0)          # obs normalizer only
TWIST_LO = (math.radians(-90.0), math.radians(-70.0))   # (left, right)
TWIST_HI = (math.radians(70.0), math.radians(90.0))
ELBOW_HI = math.radians(150.0)          # flexion 0..150 deg
ROLL_LO = (0.0, math.radians(-180.0))                   # (left, right)
ROLL_HI = (math.radians(180.0), 0.0)
BEND_LIM = math.radians(95.0)


def joint_limits(hand):
    """[(lo, hi)] x 6 for one hand, in joint order."""
    return [(PITCH_LO, PITCH_HI), (YAW_LO[hand], YAW_HI[hand]),
            (TWIST_LO[hand], TWIST_HI[hand]), (0.0, ELBOW_HI),
            (ROLL_LO[hand], ROLL_HI[hand]), (-BEND_LIM, BEND_LIM)]


# --- look-at-player (env v7.6) -----------------------------------------------
# Plain notes bias their full 3D orientation toward facing the player's
# head as they approach (ArcViewer/BeatLeader look-at): forward blends
# from +z to the (vertically 80%-damped) head->note ray with weight
# lam = clamp(1 - z/LOOK_DIST, 0, 1); the cut-direction z-roll rides as
# the frame's up. Hitboxes AND the direction check live in this frame.
# lam = 0 (heads, links, far notes) = the pure v7.4 z-roll frame.
HEAD = (0.0, 1.70, -0.65)   # fixed head anchor (1.8 m player)
LOOK_VERT = 0.8             # verticalLookStrength (head.y toward note.y)
LOOK_DIST = 8.0             # full head-facing at z=0, none beyond 8 m

N_BLADE_SAMPLES = 8   # contact-point/velocity samples along the blade
# broad-phase gate: max wrist z 0.0 (shoulder -0.65 + full arm 0.65) +
# blade 1.0 >= good-box near face at note_z - 0.75 -> note_z <= 1.75
REACH_Z = 1.75
# note consumed as missed past this — the HEAD plane (0 m behind the
# head at z = -0.65). DELIBERATE DEVIATION from the real game (despawn
# ~2 m behind the head, v7.0-v7.2's 2.65): tightens miss credit
# assignment — the -0.5 lands ~5 ticks after the plane instead of ~20,
# and a passed note is certainly lost, never ambiguously rescuable
# (env v7.3, user-specified 2026-07-22).
Z_MISS = 0.65

# --- scoring (real Beat Saber 115 points per note) ---------------------------
PRE_FULL = math.radians(100.0)   # full pre-swing arc
POST_FULL = math.radians(60.0)   # full follow-through arc
PTS_PRE, PTS_POST, PTS_ACC = 70, 30, 15   # component maxima (integers)
SCORE_MAX = 115.0
CHAIN_HEAD_MAX = 85.0            # 70 pre + 15 acc, no follow-through
CHAIN_LINK_PTS = 20.0            # flat, any direction, speed still required
SWING_BREAK = 0.005   # rad/tick below which the swing chain breaks
FOLLOW_TIMEOUT = 15   # ticks a follow-through may keep accumulating
# every component is floor(x + 0.5)-rounded to an integer (real BS scores
# are integers; explicit floor+0.5 so Rust and Python agree on halves)

# --- object kinds (chart + obs) ----------------------------------------------
KIND_NOTE = 0
KIND_BOMB = 1
KIND_CHAIN_HEAD = 2
KIND_CHAIN_LINK = 3

# obs layout v6 (order is a contract; mirrored in policy.py):
# per hand [pitch_n, yaw_n, twist_n, elbow_n, cos_roll, sin_roll, bend_n,
#           om(6)/JOINT_OMEGA, wrist x/1 (y-1.4)/0.75 z/1, wrist_vel(3)/6,
#           blade dir(3), swing_meter]  -> 23
# per note  [x, y_n, z, vz, is_left, is_right, dir_cos, dir_sin,
#            is_bomb, is_chain_head, is_chain_link, free_pre, free_post,
#            valid]  -> 14   (bombs: colorless — is_left = is_right = 0,
#            dir = zero vector; chain links carry their tangent dir but
#            judge as any-direction)
N_NOTE_SLOTS = 16
NOTE_FEAT = 14
SABER_FEAT = 23
OFF_NOTES = 2 * SABER_FEAT                             # 46
OFF_NOTE_MASK = OFF_NOTES + N_NOTE_SLOTS * NOTE_FEAT   # 270
OBS_DIM = OFF_NOTE_MASK + N_NOTE_SLOTS                 # 286
X_NORM, Y_NORM = 1.0, 0.75
Z_SCALE = 8.0
Z_CLIP = (-3.0, 12.0)

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
