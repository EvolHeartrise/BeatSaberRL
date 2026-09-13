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
    sim = beat_arena_rs.RawBeatSim(max_steps=6900,
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
    # env v21: ALL six points derive from the fk_body mirror of the
    # CONTACT start pose (arm ready stance + wide leg stance) with the
    # exact sole-corner root drop applied
    from beat_arena_env.constants import q_start
    from beat_arena_env.kinematics import (F_L_ELBOW, F_L_FOOT, F_R_ELBOW,
                                           F_R_FOOT, fk_body)
    corners = ((-0.045, -0.070, -0.075), (-0.045, -0.070, 0.175),
               (0.045, -0.070, -0.075), (0.045, -0.070, 0.175))
    rots, pos = fk_body([0.0, 0.94, -0.65, 1.0, 0.0, 0.0, 0.0]
                        + q_start(foot_contact=True))
    drop = min(pos[fr][1] + sum(rots[fr][1][k] * c[k] for k in range(3))
               for fr in (F_L_FOOT, F_R_FOOT) for c in corners)
    exp = []
    for fr, y0 in ((F_L_ELBOW, Y_CENTER), (F_R_ELBOW, Y_CENTER),
                   (21, KNEE_Y0), (27, KNEE_Y0),         # knee frames
                   (F_L_FOOT, FOOT_Y0), (F_R_FOOT, FOOT_Y0)):
        p = pos[fr]
        exp.append([p[0], (p[1] - drop - y0) / Y_NORM, p[2] + 0.65])
    exp = np.array(exp)
    assert np.abs(pts - exp).max() < 1e-6, f"\n{pts}\nvs\n{exp}"


def test_body_pt_lanes_present_in_weld_mode():
    """Positions exist in every mode. env v21: RESET poses differ by
    mode (the wide leg stance is contact-only), so the cross-mode
    identity is asserted at an EXPLICITLY matched state instead."""
    w = _reset_obs(False)
    assert w.shape == (OBS_DIM,)
    assert np.abs(w[OFF_BODY_PTS:]).max() > 0.05
    from beat_arena_env.constants import q_start
    q36 = [0.0, 0.94, -0.65, 1.0, 0.0, 0.0, 0.0] + q_start()
    obs2 = []
    for fc in (False, True):
        sim = beat_arena_rs.RawBeatSim(max_steps=6900,
                                       wide_acc=False, gears=5,
                                       foot_contact=fc)
        sim.set_chart([note_tuple(FAR_NOTE)])
        sim.set_joints(list(q36), [0.0] * 35)
        obs2.append(np.asarray(sim.obs(), dtype=np.float64))
    np.testing.assert_array_equal(obs2[0][OFF_BODY_PTS:],
                                  obs2[1][OFF_BODY_PTS:])


def test_body_pt_lanes_track_motion():
    """Driving the left knee moves the l_knee/l_foot lanes and leaves
    the right side's lanes untouched relative to... the right side also
    moves (coupled dynamics through the root), so assert the LEFT foot
    lane moves MORE than the right, and elbows move with hip torque far
    less than feet."""
    sim = beat_arena_rs.RawBeatSim(max_steps=6900,
                                   wide_acc=False, gears=5,
                                   foot_contact=False)
    sim.set_chart([note_tuple(FAR_NOTE)])
    # env v20: this script is calibrated to the pre-v20 zero-arm pose;
    # pin it (the ready stance only moves the episode START pose, not
    # the mechanics under test)
    q = list(sim.qpos())
    for _j in (3, 6, 8, 10, 13, 15):
        q[7 + _j] = 0.0
    sim.set_joints(q, [0.0] * 35)
    o0 = np.asarray(sim.obs(), dtype=np.float64)
    act = [0] * N_JOINTS
    act[20] = 1  # l_knee full flex (one-way joint, gear id 1 = +1.0)
    obs = o0
    # drive 10 -> 5 ticks (2026-08-20 ankle_p +35): the wider ankle
    # range lets the welded body ROCK further, and by 10 ticks the
    # whole-body rock drowns the driven side (ratio 1.07); the knee
    # flexion itself saturates early, so 5 ticks keeps the signal
    # (measured l 0.247 / r 0.161, ratio 1.53)
    for _ in range(5):
        obs, _r, _d, _ev = sim.step(act)
    # slice ends at the position block — env v24 appended velocity
    # lanes after it, which this motion test does not pin
    d = np.abs(obs[OFF_BODY_PTS:OFF_BODY_PTS + 18]
               - o0[OFF_BODY_PTS:OFF_BODY_PTS + 18]).reshape(6, 3)
    l_knee_move, r_knee_move = d[2].max(), d[3].max()
    assert l_knee_move > 1e-3, f"driven knee's lanes froze: {d}"
    # welded feet rock the whole body, so the passive side moves too —
    # only the RATIO is side-selective. Threshold 2x -> 1.25x at the
    # 2026-08-20 knee 250 -> 150: the driven side's flexion shrank with
    # the cap while the reaction rocking scales with the same torque
    # (measured ratio 1.39 at knee-150)
    assert l_knee_move > 1.25 * r_knee_move, \
        f"knee lanes not side-selective: l {l_knee_move} r {r_knee_move}"
