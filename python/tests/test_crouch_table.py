"""env v38 crouch-start ladder sanity (CPU; the kernel path is gated
on GPU): rows within joint limits, monotone depth, lerp-closed, and
the .cu literals match constants.py verbatim."""
import re

from beat_arena_env.constants import CROUCH_START_ROWS

LIMS = {  # model.rs HUMANOID_JOINT_LIMITS excerpts
    "spine": (-0.6981317007977318, 0.6981317007977318),  # generous stub
    "hip": (-0.3490658503988659, 2.0943951023931953),
    "knee": (0.0, 2.6179938779914944),
    "ankle": (-0.8726646259971648, 0.6108652381980153),
}


def test_rows_within_limits_and_monotone():
    knees = [r[2] for r in CROUCH_START_ROWS]
    assert knees == sorted(knees) and knees[0] == 0.3 and knees[-1] == 2.1
    for sp, hip, knee, ank in CROUCH_START_ROWS:
        assert LIMS["hip"][0] <= hip <= LIMS["hip"][1]
        assert LIMS["knee"][0] <= knee <= LIMS["knee"][1]
        assert LIMS["ankle"][0] <= ank <= LIMS["ankle"][1]
        assert LIMS["spine"][0] <= sp <= LIMS["spine"][1]
        # lerp of any adjacent pair stays in range (convexity)


def _head_root(row):
    """FK head/root height of a ladder row with the sole corners on
    y=0 (the reset's ground-exact drop), neutral arms."""
    from beat_arena_env.constants import Q_START_ARM, Q_START_LEG
    from beat_arena_env.kinematics import fk_body, F_L_FOOT, F_R_FOOT, F_HEAD
    corners = [[-0.045, -0.070, -0.075], [-0.045, -0.070, 0.175],
               [0.045, -0.070, -0.075], [0.045, -0.070, 0.175]]
    sp, hip, knee, ank = row
    q = [0.0] * 29
    for j, v in Q_START_ARM: q[j] = v
    for j, v in Q_START_LEG: q[j] = v
    q[0] = sp
    for h_, k_, a_ in ((17, 20, 21), (23, 26, 27)):
        q[h_] = hip; q[k_] = knee; q[a_] = ank
    rots, pos = fk_body([0.0, 1.2, -0.65, 1.0, 0.0, 0.0, 0.0] + q)
    low = min(pos[f][1] + rots[f][1][0] * c[0] + rots[f][1][1] * c[1]
              + rots[f][1][2] * c[2]
              for f in (F_L_FOOT, F_R_FOOT) for c in corners)
    return pos[F_HEAD][1] - low, 1.2 - low


def test_deep_end_clears_forced_duck_wall():
    """2026-09-06: the ladder's deepest row must put the head under the
    1.20 m forced-duck wall bottom, and every row must keep the root
    above the 0.25 m FALL threshold with margin (jitter is 0.15 rad)."""
    heads = [_head_root(r)[0] for r in CROUCH_START_ROWS]
    roots = [_head_root(r)[1] for r in CROUCH_START_ROWS]
    assert heads == sorted(heads, reverse=True), heads
    assert heads[0] > 1.6 and heads[-1] < 1.20, heads
    assert min(roots) > 0.25 + 0.10, roots


def test_cu_table_matches_constants():
    import os
    cu = open(os.path.join(os.path.dirname(__file__), "..",
                           "beat_arena_env", "csrc",
                           "beat_arena_cuda.cu")).read()
    block = cu.split("CROUCH_ROWS[N_CROUCH][4] = {")[1].split("};")[0]
    rows = re.findall(r"\{([^}]+)\}", block)
    assert len(rows) == len(CROUCH_START_ROWS)
    for row_txt, expect in zip(rows, CROUCH_START_ROWS):
        vals = [float(v) for v in row_txt.split(",")]
        assert vals == list(expect), (vals, expect)
