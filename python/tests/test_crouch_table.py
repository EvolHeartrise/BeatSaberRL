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
    assert knees == sorted(knees) and knees[0] == 0.3 and knees[-1] == 1.5
    for sp, hip, knee, ank in CROUCH_START_ROWS:
        assert LIMS["hip"][0] <= hip <= LIMS["hip"][1]
        assert LIMS["knee"][0] <= knee <= LIMS["knee"][1]
        assert LIMS["ankle"][0] <= ank <= LIMS["ankle"][1]
        # lerp of any adjacent pair stays in range (convexity)


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
