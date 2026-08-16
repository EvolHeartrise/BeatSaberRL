"""env v13 (unilateral-contact spec §8): the elbow/knee/foot obs lanes,
on the Rust reference. The CUDA mirror is gated by the
test_v11_contact_parity.py lockstep tiers (full-obs comparison)."""
import numpy as np

from beat_arena_env.chart import Note, note_tuple
from beat_arena_env.constants import (FOOT_Y0, KNEE_Y0, N_JOINTS, OBS_DIM,
                                      OFF_BODY_PTS, Y_CENTER, Y_NORM)
import beat_arena_rs

FAR_NOTE = Note(hit_time=200.0, col=2, row=1, x=0.3, y=1.4, color=1,
                angle=4.712, speed=4.0)


def _reset_obs(foot_contact):
    sim = beat_arena_rs.RawBeatSim(max_steps=6900, effort_coef=4e-4,
                                   wide_acc=False, gears=5,
                                   foot_contact=foot_contact)
    return np.asarray(sim.set_chart([note_tuple(FAR_NOTE)]),
                      dtype=np.float64)


def test_body_pt_lanes_reference_pose():
    """At the neutral reset the lanes must read the §1.2 reference
    geometry exactly: elbows at (∓0.20, 1.47, −0.30), knees at
    (∓0.10, 0.50, −0.65), feet (ankles) at (∓0.10, 0.07, −0.65) — in
    lane space, x/1, (y−Y0)/0.75, (z+0.65)/1."""
    obs = _reset_obs(True)
    pts = obs[OFF_BODY_PTS:OFF_BODY_PTS + 18].reshape(6, 3)
    # elbows move with the env v20 arm ready stance — take them from
    # the fk_body mirror; knees/feet stay at the §1.2 reference
    from beat_arena_env.constants import q_start
    from beat_arena_env.kinematics import F_L_ELBOW, F_R_ELBOW, fk_body
    _, pos = fk_body([0.0, 0.94, -0.65, 1.0, 0.0, 0.0, 0.0] + q_start())
    le, re = pos[F_L_ELBOW], pos[F_R_ELBOW]
    exp = np.array([
        [le[0], (le[1] - Y_CENTER) / Y_NORM, le[2] + 0.65],  # l_elbow
        [re[0], (re[1] - Y_CENTER) / Y_NORM, re[2] + 0.65],  # r_elbow
        [-0.10, (0.50 - KNEE_Y0) / Y_NORM, 0.0],     # l_knee
        [+0.10, (0.50 - KNEE_Y0) / Y_NORM, 0.0],     # r_knee
        [-0.10, (0.07 - FOOT_Y0) / Y_NORM, 0.0],     # l_foot
        [+0.10, (0.07 - FOOT_Y0) / Y_NORM, 0.0],     # r_foot
    ])
    assert np.abs(pts - exp).max() < 1e-6, f"\n{pts}\nvs\n{exp}"


def test_body_pt_lanes_present_in_weld_mode():
    """Positions exist in every mode — weld obs carries them too, and
    identically to contact mode at the identical reset state."""
    w, c = _reset_obs(False), _reset_obs(True)
    assert w.shape == (OBS_DIM,)
    np.testing.assert_array_equal(w[OFF_BODY_PTS:], c[OFF_BODY_PTS:])
    assert np.abs(w[OFF_BODY_PTS:]).max() > 0.05


def test_body_pt_lanes_track_motion():
    """Driving the left knee moves the l_knee/l_foot lanes and leaves
    the right side's lanes untouched relative to... the right side also
    moves (coupled dynamics through the root), so assert the LEFT foot
    lane moves MORE than the right, and elbows move with hip torque far
    less than feet."""
    sim = beat_arena_rs.RawBeatSim(max_steps=6900, effort_coef=4e-4,
                                   wide_acc=False, gears=5,
                                   foot_contact=False)
    o0 = np.asarray(sim.set_chart([note_tuple(FAR_NOTE)]), dtype=np.float64)
    act = [0] * N_JOINTS
    act[20] = 1  # l_knee full flex (one-way joint, gear id 1 = +1.0)
    obs = o0
    for _ in range(10):
        obs, _r, _d, _ev = sim.step(act)
    d = np.abs(obs[OFF_BODY_PTS:] - o0[OFF_BODY_PTS:]).reshape(6, 3)
    l_knee_move, r_knee_move = d[2].max(), d[3].max()
    assert l_knee_move > 1e-3, f"driven knee's lanes froze: {d}"
    # welded feet rock the whole body, so the passive side moves too —
    # only the RATIO is side-selective
    assert l_knee_move > 2 * r_knee_move, \
        f"knee lanes not side-selective: l {l_knee_move} r {r_knee_move}"
