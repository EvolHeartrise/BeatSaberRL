"""World and observation constants.

MIRROR of rust/src/lib.rs — the Rust crate is THE sim implementation
(single source of truth since 2026-07-20; the Python sim was deleted to
cut maintenance surface). Any rule change lands in lib.rs first and is
mirrored here for the scripted bot, tests, and tooling. SPEC.md remains
the written contract.
"""

DT = 1.0 / 30.0
ACCEL = 30.0          # m/s^2
DRAG = 0.02           # per-tick multiplicative velocity decay
V_MAX = 6.0           # m/s
V_MIN_CUT = 1.0       # min saber speed for a good cut (m/s)
CUT_RADIUS = 0.28
COS_TOL = 0.5
Z_WINDOW = 0.6        # cuttable slab half-depth (m)
CELL = 0.5            # lane pitch (m)
BLOCK = 0.4           # block edge (m)
POS_LO = (-1.5, -1.25)
POS_HI = (1.5, 1.25)
REWARD_GOOD = 1.0
REWARD_BAD = -0.25
REWARD_MISS = -0.5

# obs layout (order is a contract; mirrored in policy.py)
N_NOTE_SLOTS = 12
NOTE_FEAT = 15
OFF_NOTES = 8
OFF_NOTE_MASK = OFF_NOTES + N_NOTE_SLOTS * NOTE_FEAT  # 188
OBS_DIM = OFF_NOTE_MASK + N_NOTE_SLOTS                # 200
X_NORM, Y_NORM = 1.0, 0.75
Z_SCALE = 8.0
Z_CLIP = (-1.0, 12.0)

# event codes returned by the Rust sim's step()
EVENT_REASONS = {0: "ok", 1: "slow", 2: "angle", 3: "wrong_hand", 4: "miss"}


def note_center(n):
    """In-plane (x, y) of a note's cell center; grid centered on origin."""
    return ((n.col - 1.5) * CELL, (n.row - 1.0) * CELL)
