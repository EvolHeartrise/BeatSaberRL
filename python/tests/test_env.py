"""Env tests against the Rust sim — THE implementation (SPEC.md §Tests).

Env v10 (P3 re-author, 2026-07-31): the cut scenarios drive the SPEC-v10
dynamic body via the full-state `set_joints` hook (root pose + 29 joints
+ 35 velocities) and assert on the event stream. The v9 "aligned wrist
frame" pose family carries over VERBATIM as initial conditions — the
arm-chain geometry is v9-identical at the reference root (the §1.3
continuity theorem) — but the body now evolves under gravity/dynamics
during the tick, so point-value asserts are re-tuned where the v9
literals assumed the kinematic integrator. The cut geometry contract
(blade through box at speed/angle => score) is RE-PROVEN here on v10.

Moved tests (re-authored in tests/test_v10_env.py, the P3 gate file):
  - test_obs_layout        -> test_obs_length_and_reset_layout (+ full
                              §5 offset checks, q/qd normalizer table)
  - test_recording_roundtrip -> test_replay_v10_roundtrip (JSONL v10)
  - test_effort_penalty_tiers -> test_effort_is_priced_positive_work
                              (the EFFORT_COST table is retired; effort
                              is positive mechanical work, SPEC-v10 §4)
The scripted reference bot and its solvability gate
(test_scripted_bot_solves_easy) were removed 2026-09-22.
"""

import math

import numpy as np
import pytest

from beat_arena_env import (
    OBS_DIM, OFF_NOTES, OFF_NOTE_MASK, N_NOTE_SLOTS, NOTE_FEAT,
    ChartConfig, Note, RustBeatEnv, generate_chart,
)
from beat_arena_env.constants import (
    A_L_ARM, A_R_ARM, DT, KIND_BOMB, KIND_CHAIN_HEAD, KIND_CHAIN_LINK,
    KIND_NOTE, N_JOINTS, NV, NQPOS, OFF_HAND_L, OFF_HAND_R, OFF_HEAD,
    ROOT_POS0,
    ROW_Y, col_x, joint_limits,
)
from beat_arena_env.kinematics import fk

from beat_arena_rs import RawBeatSim

from tests._helpers import ALIVE, plant_reference


DOWN = 6 * np.pi / 4  # downward swing angle
COAST = (0,) * N_JOINTS
ACTION_NVEC = [3] * N_JOINTS
DELTA = 0.7
FLAT_ARM = [-np.pi / 2 + DELTA, 0.0, 0.0, np.pi / 2]
FLAT_WRIST = (0.2, 1.3955705, -0.1950711)   # right-hand wrist FK of FLAT_ARM


def _note(hit_time, col, row, color=1, angle=DOWN, speed=4.0, any_dir=False,
          kind=0, x=None, y=None, free_pre=False, free_post=False):
    # env v33: ROW_Y carries the +0.3 player-height offset; the scripted
    # fixtures in this file are calibrated to the PRE-offset heights and
    # test cut mechanics at fixed world geometry, so pin y explicitly
    # (the row->y mapping itself is covered by the chart/positions tests)
    if y is None:
        from beat_arena_env.constants import PLAYER_HEIGHT_OFFSET
        y = ROW_Y[row] - PLAYER_HEIGHT_OFFSET
    return Note(hit_time=hit_time, col=col, row=row, color=color,
                angle=angle, speed=speed, any_dir=any_dir, kind=kind,
                x=x, y=y, free_pre=free_pre, free_post=free_post)


def _sim_with_notes(notes, max_steps=690, **kw):
    """A sim at the v10 reference stance (tests._helpers.plant_reference
    — the retired weld reset's state, now on ground contact: the cut
    scenarios below are a few ticks long, well inside the ~0.5 s before
    an unactuated body starts to tip)."""
    from beat_arena_env.chart import note_tuple
    sim = RawBeatSim(max_steps=max_steps, **kw)
    sim.set_chart([note_tuple(n) for n in notes])
    return plant_reference(sim)


def _sim_with_note(note, max_steps=690, **kw):
    return _sim_with_notes([note], max_steps=max_steps, **kw)


def _set_arm(sim, hand, q7, om7):
    """Overwrite ONE arm's 7-joint state, preserving everything else —
    the v10 equivalent of the v9 per-arm set_joints hook (the root stays
    wherever it is; velocities of other coords are preserved)."""
    q = np.asarray(sim.qpos())
    v = np.zeros(NV)
    v[:6] = sim.twist()
    v[6:] = sim.omega()
    base = A_L_ARM if hand == 0 else A_R_ARM
    q[7 + base:7 + base + 7] = q7
    v[6 + base:6 + base + 7] = om7
    sim.set_joints(list(q), list(v))


def _park(sim, hand):
    """Swing the arm down and outward: blade far below/outside the grid."""
    yaw = -1.5 if hand == 0 else 1.5
    _set_arm(sim, hand, [-1.2, yaw, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0] * 7)


def _set_flat(sim, hand, azimuth, om_roll=0.0, om_bend=0.0, delta=DELTA):
    """FLAT pose: blade along `azimuth` in a z = const plane (aligned
    wrist frame R_x(-delta) fixes +-x; primary rep roll = azimuth - 90
    deg, bend = +90 deg, dev = 0). For delta = DELTA the right wrist
    sits at FLAT_WRIST at t = 0; roll sweeps (om_roll) move the blade
    vertically. Under v10 dynamics this is the INITIAL condition — the
    pose sags slightly within the tick."""
    lo, hi = joint_limits(hand)[4]
    roll = (azimuth - np.pi / 2 + np.pi) % (2 * np.pi) - np.pi
    bend = np.pi / 2
    if not (lo <= roll <= hi):
        roll = (azimuth + np.pi / 2 + np.pi) % (2 * np.pi) - np.pi
        bend = -np.pi / 2
        assert lo <= roll <= hi, f"azimuth {azimuth} unreachable for hand {hand}"
    q = FLAT_ARM[:1] if delta == DELTA else None  # (doc aid only)
    _set_arm(sim, hand, [-np.pi / 2 + delta, 0.0, 0.0, np.pi / 2, roll,
                         bend, 0.0],
             [0.0, 0.0, 0.0, 0.0, om_roll, om_bend, 0.0])


def _coast(sim):
    _, r, _, events = sim.step(list(COAST))
    return events, r


def _codes(events):
    return [(i, c) for i, c, *_ in events]


# --- kinematics mirror (v9 continuity at the env surface) --------------------

def test_python_fk_matches_rust():
    """kinematics.fk (the v9 arm-chain mirror) still pins the v10 body's
    ARM FK exactly at the reference root — the §1.3 continuity theorem
    surfaced through the sim's sabers()/blades() getters."""
    sim = _sim_with_note(_note(5.0, 2, 1))
    rng = np.random.default_rng(4)
    for _ in range(20):
        hand = int(rng.integers(0, 2))
        q7 = [float(rng.uniform(lo, hi)) for lo, hi in joint_limits(hand)]
        _set_arm(sim, hand, q7, [0.0] * 7)
        _, w, b = fk(hand, q7)
        s = sim.sabers()[hand]
        assert np.allclose(s[:3], w, atol=1e-12)
        assert np.allclose(s[3:], b, atol=1e-12)
        assert np.allclose(sim.blades()[hand], b, atol=1e-12)


def test_flat_pose_geometry():
    """The FLAT scenario pose parks the right wrist at FLAT_WRIST; the
    wrist frame is R_x(-DELTA), which fixes the world x axis — so the
    blade is exactly in-plane along +-x (the azimuths every contact in
    the scenario family uses), and tilted by the frame elsewhere."""
    for az in (0.0, np.pi / 2, np.pi):
        _, w, b = fk(1, FLAT_ARM + [az - np.pi / 2, np.pi / 2])
        assert np.allclose(w, FLAT_WRIST, atol=1e-6)
        expect = (np.cos(az), np.cos(DELTA) * np.sin(az),
                  -np.sin(DELTA) * np.sin(az))
        assert np.allclose(b, expect, atol=1e-9)
    for az in (0.0, np.pi):                      # in-plane at contact
        _, _, b = fk(1, FLAT_ARM + [az - np.pi / 2, np.pi / 2])
        assert abs(b[2]) < 1e-12


# --- obs layout (full §5 table in tests/test_v10_env.py) ---------------------

def test_obs_reset_hands_and_notes():
    # bombs/dots/extras off: this test pins slot/feature offsets against
    # chart[0], which must be a plain arrow note for the color/dir asserts
    env = RustBeatEnv(seed=3, bomb_prob=0.0, dot_prob=0.0, extra_prob=0.0)
    obs, _ = env.reset()
    assert obs.shape == (OBS_DIM,) == (630,)   # env v49: blade-tip velocity lanes
    assert OFF_NOTES == 102 and OFF_NOTE_MASK == 534   # env v19
    # hand blocks at reset (env v20 ready stance): wrists/blades from
    # the fk_body mirror of the shared start pose
    from beat_arena_env.constants import q_start
    from beat_arena_env.kinematics import (F_L_HAND, F_L_WRIST, F_R_HAND,
                                           F_R_WRIST, fk_body)
    rots, pos = fk_body([0.0, 0.94, -0.65, 1.0, 0.0, 0.0, 0.0] + q_start())
    for base, fw, fh in ((OFF_HAND_L, F_L_WRIST, F_L_HAND),
                         (OFF_HAND_R, F_R_WRIST, F_R_HAND)):
        w, r = pos[fw], rots[fh]
        # env v38: wrist x/y head-relative (reference head (0, 1.70))
        assert obs[base] == pytest.approx(w[0] - 0.0, abs=1e-6)
        assert obs[base + 1] == pytest.approx((w[1] - 1.70) / 0.75,
                                              abs=1e-6)
        assert obs[base + 2] == pytest.approx(w[2], abs=1e-6)
        assert not obs[base + 3:base + 6].any()             # wrist vel
        assert obs[base + 8] == pytest.approx(r[2][2], abs=1e-6)  # blade z
        assert obs[base + 9] == 0.0                         # swing meter
    n_pending = min(len(env.chart), N_NOTE_SLOTS)
    assert obs[OFF_NOTE_MASK:OFF_NOTE_MASK + N_NOTE_SLOTS].sum() == n_pending
    # env v17: slots are DISTANCE-sorted (nearest to the head first),
    # so chart[0] (the SOONEST note) sits wherever its distance ranks —
    # find it by its x/z signature, then pin the lane semantics there
    n = env.chart[0]
    from beat_arena_env.constants import note_center as nc
    from tests._helpers import z_lane
    z = n.hit_time * n.speed
    hits = [s2 for s2 in range(N_NOTE_SLOTS)
            if obs[OFF_NOTE_MASK + s2] == 1.0
            and abs(obs[OFF_NOTES + s2 * NOTE_FEAT]
                    - (nc(n)[0] - obs[OFF_HEAD])) < 1e-6
            and abs(obs[OFF_NOTES + s2 * NOTE_FEAT + 2] - z_lane(z)) < 1e-6]
    assert hits, "soonest note missing from the entity window"
    base = OFF_NOTES + hits[0] * NOTE_FEAT
    assert obs[base + 13] == 0.0        # ex-valid lane, retired 2026-08-22
    head_y = obs[OFF_HEAD + 1] * 0.75 + 1.70
    # env v38: entity y is head-relative
    assert obs[base + 1] == pytest.approx((nc(n)[1] - head_y) / 0.75, abs=1e-6)
    assert obs[base + 3] == pytest.approx(-n.speed / 8.0)
    assert obs[base + 6] == pytest.approx(np.cos(n.angle), abs=1e-6)
    assert obs[base + 7] == pytest.approx(np.sin(n.angle), abs=1e-6)
    assert not obs[base + 8:base + 13].any()                # kind/arc flags
    assert not obs[base + 14:base + 18].any()               # wall lanes
    for slot in range(n_pending, N_NOTE_SLOTS):
        assert not obs[OFF_NOTES + slot * NOTE_FEAT:
                       OFF_NOTES + (slot + 1) * NOTE_FEAT].any()


def test_obs_kind_flags():
    """Bombs are colorless/directionless with is_bomb set; chain heads and
    links carry their flags; links keep their tangent direction; arc
    grants surface as free_pre/free_post."""
    notes = [
        _note(2.0, 1, 1, kind=KIND_BOMB, any_dir=True, color=0),
        _note(3.0, 2, 1, kind=KIND_CHAIN_HEAD),
        _note(3.1, 2, 0, kind=KIND_CHAIN_LINK, any_dir=True,
              angle=0.25, x=0.31, y=1.1),
        _note(4.0, 3, 1, free_pre=True, free_post=True),
    ]
    sim = _sim_with_notes(notes)
    obs = np.asarray(sim.set_chart(
        [__import__("beat_arena_env.chart", fromlist=["note_tuple"])
         .note_tuple(n) for n in notes]))
    b0 = OFF_NOTES
    assert obs[b0 + 8] == 1.0                       # is_bomb
    assert not obs[b0 + 4:b0 + 8].any()             # no color, no dir
    b1 = OFF_NOTES + NOTE_FEAT
    assert obs[b1 + 9] == 1.0                       # is_chain_head
    b2 = OFF_NOTES + 2 * NOTE_FEAT
    assert obs[b2 + 10] == 1.0                      # is_chain_link
    assert obs[b2 + 0] == pytest.approx(0.31)       # explicit x
    assert obs[b2 + 6] == pytest.approx(np.cos(0.25), abs=1e-6)  # tangent
    b3 = OFF_NOTES + 3 * NOTE_FEAT
    assert obs[b3 + 11] == 1.0 and obs[b3 + 12] == 1.0  # free_pre/free_post


def test_swing_meter_tracks_rotation():
    """The pre-swing obs feature rises while swinging, resets on reversal.
    Uses the right BEND joint torque (action index 15, env v10)."""
    env = RustBeatEnv(seed=3)
    env.reset()
    # env v20: script calibrated to the pre-v20 zero-arm pose
    _set_arm(env.sim, 0, [0.0] * 7, [0.0] * 7)
    _set_arm(env.sim, 1, [0.0] * 7, [0.0] * 7)
    b = OFF_HAND_R + 9   # right hand's swing meter
    bend_plus = [0] * N_JOINTS
    bend_plus[15] = 1
    bend_minus = [0] * N_JOINTS
    bend_minus[15] = 2
    obs, _, _, _, _ = env.step(bend_plus)
    m1 = obs[b]
    obs, _, _, _, _ = env.step(bend_plus)
    m2 = obs[b]
    assert 0.0 < m1 < m2
    meters = [m2]
    for _ in range(6):        # reverse hard until omega flips sign
        obs, _, _, _, _ = env.step(bend_minus)
        meters.append(obs[b])
    # v10: gravity keeps the falling arm rotating fast, so the post-reset
    # meter restarts from that tick's (large) delta rather than near zero
    # — the RESET itself shows as a strict drop in the sequence while the
    # blade never stops rotating
    drops = np.diff(np.asarray(meters))
    assert drops.min() < -0.1, f"no reversal reset: {meters}"


def test_z_is_time_times_speed():
    """time_until = (z - z_off) / |vz| at every step (spatial obs
    contract; env v15, contact spec §10: the note z lane carries the
    live head anchor z_off = head_z - PLANE_Z0, recovered here from the
    same obs vector's head lane)."""
    from beat_arena_env.constants import OFF_HEAD, PLANE_Z0, X_NORM
    from beat_arena_env.constants import note_center as nc
    from tests._helpers import z_lane_m
    env = RustBeatEnv(seed=9)
    obs, _ = env.reset()
    checked = 0
    for _ in range(60):
        obs, _, done, _, _ = env.step(COAST)
        if done:
            break
        z_off = float(obs[OFF_HEAD + 2]) * X_NORM  # head_z - PLANE_Z0
        status = env.sim.status()
        pending = [i for i, s in enumerate(status) if s == 0]
        # env v17: slots are DISTANCE-sorted, not chart-ordered, so each
        # slot is matched to its note by the x-lane signature + speed
        # (the chart-order pairing only held while the 12 m clamp hid
        # the far, reorderable notes)
        for slot in range(N_NOTE_SLOTS):
            if obs[OFF_NOTE_MASK + slot] != 1.0:
                continue
            base = OFF_NOTES + slot * NOTE_FEAT
            z_obs = z_lane_m(obs[base + 2])
            vz_obs = obs[base + 3] * 8.0
            # floor = LIVE head z since 2026-08-22; in nominal terms
            # the boundary is exactly t*speed > PLANE_Z0 (z_off
            # cancels), and it coincides with miss/pass resolution
            if z_obs - z_off <= PLANE_Z0 + 1e-4:
                continue
            t_obs = (z_obs - z_off) / -vz_obs
            cands = [env.chart[i].hit_time - env.sim.time for i in pending
                     if abs(obs[base] - (nc(env.chart[i])[0] - obs[OFF_HEAD]))
                     < 1e-5
                     and abs(obs[base + 1] * 0.75 + obs[OFF_HEAD + 1] * 0.75
                             + 1.70
                             - nc(env.chart[i])[1]) < 1e-5
                     and abs(vz_obs + env.chart[i].speed) < 1e-4]
            assert cands, (slot, t_obs)
            assert min(abs(t_obs - t) for t in cands) < 1e-4, (slot, t_obs,
                                                                cands)
            checked += 1
    assert checked > 0, checked


# --- determinism -------------------------------------------------------------

def test_determinism():
    def run():
        env = RustBeatEnv(colors="both", bomb_prob=0.3, extra_prob=0.3,
                          seed=7)
        env.reset(seed=7)
        rng = np.random.default_rng(1)
        out = []
        done = False
        while not done:
            _, r, done, _, info = env.step(rng.integers(0, ACTION_NVEC))
            out.append((r, tuple(info["events"])))
        return out

    assert run() == run()


# --- cut detection scenarios (v10 dynamics via the set_joints hook) ----------
# Standard geometry: note at cell (3,1) = center (0.9, 1.40), hit_time =
# DT (z = 0 at the end of the first step). The right arm STARTS in the
# FLAT pose (wrist ~(0.2, 1.396, -0.195), blade along +x at row-1
# height); CW roll (om_roll < 0) sweeps the contact region DOWNWARD
# through the note's good box. Physics evolves the pose within the tick
# (gravity sag ~cm-scale), so point asserts carry v10-tuned bands.

def test_good_swing_cut_scores_integer_points():
    sim = _sim_with_note(_note(DT, 3, 1))
    _set_flat(sim, 1, 0.0, om_roll=-15.0)
    ev, r = _coast(sim)
    assert _codes(ev) == [(0, 0)]
    assert sim.good_cuts == 1
    pts = ev[0][2]
    # ~28 deg of pre-swing (-> ~20) + a few accuracy points (the flat
    # blade plane sits ~0.2 m behind the note center); v10 measured 26
    assert 15.0 <= pts <= 45.0
    assert pts == int(pts)                    # real BS scores are integers
    # coast tick: zero torque => zero activation price, reward is exactly
    # pts/115 plus the per-tick alive bonus
    assert r == pytest.approx(pts / 115.0 + ALIVE, abs=1e-9)


def test_gentle_swing_is_good():
    """V_MIN_CUT is 1 m/s at the CONTACT POINT: a slow-ish roll whose
    contact point still moves >= 1 m/s scores."""
    sim = _sim_with_note(_note(DT, 3, 1))
    _set_flat(sim, 1, 0.0, om_roll=-3.0)
    ev, _ = _coast(sim)
    assert _codes(ev) == [(0, 0)]


def test_slow_graze_is_bad():
    sim = _sim_with_note(_note(DT, 3, 1))
    _set_flat(sim, 1, 0.0, om_roll=-0.5)
    ev, _ = _coast(sim)
    assert _codes(ev) == [(0, 1)]  # slow
    assert ev[0][2] == 0.0


def test_wrong_direction_swing_is_bad():
    sim = _sim_with_note(_note(DT, 3, 1))
    _set_flat(sim, 1, 0.0, om_roll=15.0)
    ev, _ = _coast(sim)
    assert _codes(ev) == [(0, 2)]  # angle (swept upward through a down note)


def test_wrong_hand_only_triggers_small_box():
    """The wrong-color saber only interacts with the SMALL (0.35^3) box:
    at note_z = 0 the flat blade plane (z ~-0.195) is outside it, but a
    note just past the plane (z ~ -0.13) reaches it -> wrong_hand."""
    # hit_time = 0: after one tick the note sits at z = -0.133
    sim = _sim_with_note(_note(0.0, 1, 1, color=0, x=0.0))
    _park(sim, 0)                                     # matching arm far away
    _set_flat(sim, 1, np.pi)                          # right blade points -x
    ev, _ = _coast(sim)
    assert _codes(ev) == [(0, 3)]
    # same geometry but note still AT the plane: small box (+-0.175 z)
    # doesn't reach the blade plane -> no contact at all
    sim = _sim_with_note(_note(DT, 1, 1, color=0, x=0.0))
    _park(sim, 0)
    _set_flat(sim, 1, np.pi)
    ev, _ = _coast(sim)
    assert ev == []


def test_good_box_is_bigger_than_bad_box():
    """The same blade placement is INSIDE the good box for the matching
    hand but OUTSIDE the small box for the wrong hand: the blade plane
    ~0.195 m behind the note center clears the small box's 0.175 but not
    the good box's 1.0 m depth (the near-static probe's cm-scale sag
    keeps it clear of the small box; x = 0 keeps the v7.6 look-at tilt
    out of the picture)."""
    sim = _sim_with_note(_note(DT, 3, 1, x=0.0))
    _set_flat(sim, 1, 0.0)                     # static matching blade
    ev, _ = _coast(sim)
    assert _codes(ev) == [(0, 1)]              # inside good box -> slow
    sim = _sim_with_note(_note(DT, 3, 1, color=0, x=0.0))
    _park(sim, 0)
    _set_flat(sim, 1, 0.0)                     # static wrong-color blade
    ev, _ = _coast(sim)
    assert ev == []                            # small box out of reach


def test_early_cut_through_the_deep_box():
    """The good box extends 0.75 m toward the player: a fast swing
    connects while the note is still ~0.5 m out (a real-BS early cut),
    but not at ~0.8 m — at the head column (x = 0, no look-at tilt).
    At the EDGE column the v7.6 billboard swings the deep box toward
    the head, so the same ~0.8 m timing DOES connect there."""
    for t_hit, x, hit in ((0.13 + DT, 0.0, True), (0.21 + DT, 0.0, False),
                          (0.21 + DT, 0.9, True)):
        sim = _sim_with_note(_note(t_hit, 3, 1, x=x))  # z after 1 tick: 4*t
        _set_flat(sim, 1, 0.0, om_roll=-15.0)
        ev, _ = _coast(sim)
        assert bool(ev) == hit, f"hit_time {t_hit} x {x}"
        if hit:
            assert _codes(ev) == [(0, 0)]


def test_lookat_direction_check_is_local():
    """Env v7.6: the direction check runs in the note's LOCAL frame.
    The same down+forward swing (FLAT delta = 1.19) is a clean 'ok'
    at the head column but judges 'angle' at the edge column, where
    the billboarded frame folds the forward component into local x.
    Note y = 1.618 sits on the delta = 1.19 blade plane. (v10: the
    billboard anchors on the LIVE head — at these single-tick horizons
    it is within mm of the reference head.)"""
    for x, code in ((0.0, 0), (0.9, 2)):
        sim = _sim_with_note(_note(DT, 3, 1, x=x, y=1.618))
        _park(sim, 0)
        _set_flat(sim, 1, 0.0, om_roll=-15.0, delta=1.19)
        ev, _ = _coast(sim)
        assert _codes(ev) == [(0, code)], f"x {x}"


def test_hitboxes_rotate_with_cut_direction():
    """Env v7.4: hitboxes sit on the rotated note transform. A blade
    passing ~0.38 m above the center clears an axis-aligned (down-swing)
    good box (half-height 0.25) but is caught by the raised corner of a
    57-degree-rotated one — the contact is then judged normally (the
    tilted-frame sweep misses the 60-degree tolerance -> 'angle', which
    is fine: the test pins the COLLISION volume, not the judgment)."""
    for angle, hit in ((DOWN, False), (np.radians(327.0), True)):
        sim = _sim_with_note(_note(0.0, 3, 1, angle=angle))
        _park(sim, 0)
        _set_flat(sim, 1, 0.0, om_roll=-15.0, delta=1.6)   # y ~1.78 vs 1.40
        ev, _ = _coast(sim)
        assert bool(ev) == hit, f"angle {angle}"


def test_dot_note_has_taller_hitbox():
    """Modern dots are 0.8 m tall vs 0.5 for arrow notes: a blade passing
    ~0.38 m above the row-1 center clips the dot box only (delta = 1.6
    parks the wrist at y ~1.78, z ~ -0.31)."""
    for any_dir, hit in ((False, False), (True, True)):
        sim = _sim_with_note(_note(0.0, 3, 1, any_dir=any_dir))
        _park(sim, 0)
        _set_flat(sim, 1, 0.0, om_roll=-15.0, delta=1.6)
        ev, _ = _coast(sim)
        assert bool(ev) == hit, f"any_dir {any_dir}"


def test_arm_swing_cuts_and_accuracy_decreases_off_center():
    """A pure SHOULDER-pitch pendulum (wrist joints untouched, gravity
    now assists the down-swing) sweeps the whole arm+blade through a
    note and scores. Yawing the arm off the note's column keeps the hit
    but costs accuracy points."""
    def pendulum(yaw):
        sim = _sim_with_note(_note(0.24, 2, 2))       # center (0.3, 1.9)
        _park(sim, 0)
        _set_arm(sim, 1, [0.35, yaw, 0.0, 0.0, 0.0, 0.0, 0.0],
                 [-8.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        for _ in range(6):
            ev, _ = _coast(sim)
            if ev:
                return ev
        return []

    aligned = pendulum(0.0485)   # blade plane through the note column
    offset = pendulum(-0.10)     # ~0.15 m off-column: still a hit
    assert _codes(aligned) == [(0, 0)]
    assert _codes(offset) == [(0, 0)]
    assert aligned[0][2] > offset[0][2] > 0.0


def test_accuracy_is_timing_independent():
    """The cut-plane rule (2026-07-29): accuracy measures the offset of
    the INFINITE swing plane from the note center, so where along the
    approach the cut lands (note speed => first-contact depth) must not
    change the score. Same aligned shoulder pendulum, three approach
    speeds: all cuts score the same accuracy to within integer rounding
    (residual 1-2 pt spread is physical: different speeds cut on
    different TICKS of the rotating arm)."""
    def pendulum_acc(speed):
        sim = _sim_with_note(_note(0.24, 2, 2, speed=speed))
        _park(sim, 0)
        _set_arm(sim, 1, [0.35, 0.0485, 0.0, 0.0, 0.0, 0.0, 0.0],
                 [-8.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        for _ in range(12):
            ev, _ = _coast(sim)
            if ev:
                assert ev[0][1] == 0, f"speed {speed}: cut code {ev[0][1]}"
                return ev[0][4]
        raise AssertionError(f"no cut at speed {speed}")

    accs = [pendulum_acc(s) for s in (3.0, 4.0, 6.0)]
    assert max(accs) - min(accs) <= 2.0, f"acc varies with speed: {accs}"
    # early cuts through the deep box score near-full accuracy when the
    # swing plane is centered — THE point of the cut-plane rule
    assert min(accs) >= 12.0, f"aligned early cuts should score high: {accs}"


def _whip_events(kind=0, ticks=60):
    """A TORQUE-DRIVEN full-arm swing (v10 re-author of the v9 free
    pendulum, which gravity + the -90 deg pitch stop now truncate):
    wind the right arm up to the +65 deg pitch stop while the note is
    beyond reach, then whip down at full gear through a note — > 100
    deg of blade arc banks the full 70 pre-swing, and the follow-through
    keeps sweeping for the full 30."""
    # The whip's reaction rocks the trunk and the head-anchored track
    # moves with it, so hit time and windup are a swept pair. 2026-09-27
    # (weld harness removed): on ground contact the whip tips the body
    # over — the episode ends on the tilt envelope ~17 ticks in — so the
    # windup is 13 ticks and both kinds use hit 0.75 (swept: the regular
    # note lands 70 pre + 30 post there, the chain head 70 pre, no post).
    hit = 0.75
    sim = _sim_with_note(_note(hit, 2, 1, kind=kind), max_steps=300)
    # env v20: the whip is hand-tuned against the zero-arm start
    _set_arm(sim, 0, [0.0] * 7, [0.0] * 7)
    _set_arm(sim, 1, [0.0] * 7, [0.0] * 7)
    events = []
    for t in range(ticks):
        a = [0] * N_JOINTS
        # windup 15 -> 17 ticks (2026-08-18 fist-only hand): the
        # lighter arm whips ~10% faster, so the old timing slammed the
        # -90 deg pitch stop mid-follow-through (post accrued 0); two
        # more windup ticks re-land the cut with full sweep room.
        # 17 -> 19 (2026-08-20 ankle_p +35: rock retune), 19 -> 13
        # (2026-09-27, ground contact: see hit above)
        a[A_R_ARM] = 1 if t < 13 else 2
        _, _, done, ev = sim.step(a)
        events.extend(ev)
        if done:
            break
    return sim, events


def test_full_swing_scores_full_pre_and_post():
    """A wound-up ARM swing (>100 deg) earns the full 70 pre-swing and
    the full 30 follow-through (pendulum geometry costs accuracy — by
    design; forward-tilted precise blades are the high-acc technique)."""
    sim, events = _whip_events()
    cuts = [e for e in events if e[1] == 0]
    posts = [e for e in events if e[1] == 5]
    assert len(cuts) == 1 and len(posts) == 1
    assert cuts[0][2] >= 70.0            # full pre-swing banked
    assert posts[0][2] == pytest.approx(30.0)  # full follow-through
    assert sim.score >= 100.0
    # component split: event carries (points, pre, acc); the sim's
    # decomposition getters agree with the events
    assert cuts[0][3] == pytest.approx(70.0)         # pre component
    assert cuts[0][2] == cuts[0][3] + cuts[0][4]     # points = pre + acc
    pre, acc, post = sim.score_components()
    assert pre == pytest.approx(70.0)
    assert post == pytest.approx(30.0)
    assert pre + acc + post == pytest.approx(sim.score)


def test_dot_note_scores_from_any_direction():
    """any_dir (dot) notes: an up-swing (angle-bad for a down note) is good."""
    sim = _sim_with_note(_note(DT, 3, 1, any_dir=True))
    _set_flat(sim, 1, 0.0, om_roll=15.0)
    ev, _ = _coast(sim)
    assert _codes(ev) == [(0, 0)]


def test_dot_note_obs_is_zero_vector():
    from beat_arena_env.chart import note_tuple
    sim = RawBeatSim(max_steps=690)
    obs = sim.set_chart([note_tuple(_note(2.0, 1, 1, any_dir=True))])
    assert obs[OFF_NOTES + 6] == 0.0 and obs[OFF_NOTES + 7] == 0.0
    assert obs[OFF_NOTES + 13] == 0.0  # ex-valid lane, retired 2026-08-22


def test_forward_blade_reaches_early_note():
    """The start pose (arm forward, blade its extension) contacts notes
    far before the plane, while a flat blade can't reach."""
    # note arrives in 0.3 s: z ~ 1.07 m after one tick; the forward blade
    # spans z 0..1 at (0.2, 1.47) — right next to cell (2,1) at (0.3, 1.4)
    sim = _sim_with_note(_note(0.3, 2, 1))
    _set_arm(sim, 1, [0.0] * 7, [0.0] * 7)   # env v20: legacy forward arm
    _park(sim, 0)
    ev, _ = _coast(sim)                    # right arm at the LEGACY pose
    assert _codes(ev) == [(0, 1)]          # touched it (near-static -> slow)
    # same note, blade flat at z ~ -0.195: the box front (z ~ 0.32) is
    # still 0.5 m short of the blade plane -> nothing within reach
    sim = _sim_with_note(_note(0.3, 2, 1))
    _set_arm(sim, 1, [0.0] * 7, [0.0] * 7)
    _park(sim, 0)
    _set_flat(sim, 1, np.pi / 2)
    ev, _ = _coast(sim)
    assert ev == []


def test_far_note_is_untouchable():
    sim = _sim_with_note(_note(1.0, 3, 1))   # z ~ 3.9 after one tick
    _set_flat(sim, 1, 0.0, om_roll=-15.0)
    ev, _ = _coast(sim)
    assert ev == []
    assert sim.status()[0] == 0


# --- v7 object kinds ---------------------------------------------------------

def test_chain_link_pays_flat_20_any_direction():
    """Links: flat 20 points whatever the swing looks like — an up-swing
    through a down-tangent link still pays exactly 20, and no
    follow-through tracker opens."""
    link = _note(DT, 3, 1, kind=KIND_CHAIN_LINK, any_dir=True)
    sim = _sim_with_note(link)
    _set_flat(sim, 1, 0.0, om_roll=15.0)     # "wrong" direction: irrelevant
    ev, r = _coast(sim)
    assert _codes(ev) == [(0, 0)]
    assert ev[0][2] == 20.0
    assert r == pytest.approx(20.0 / 115.0 + ALIVE, abs=1e-9)
    for _ in range(6):                        # no post event ever
        ev, _ = _coast(sim)
        assert all(c != 5 for _, c, *_ in ev)
    assert sim.score == 20.0


def test_chain_link_still_requires_speed():
    link = _note(DT, 3, 1, kind=KIND_CHAIN_LINK, any_dir=True)
    sim = _sim_with_note(link)
    _set_flat(sim, 1, 0.0, om_roll=-0.5)
    ev, _ = _coast(sim)
    assert _codes(ev) == [(0, 1)]             # slow


def test_chain_head_has_no_follow_through():
    """Chain heads score pre + accuracy (max 85) and never open a
    follow-through tracker."""
    sim, events = _whip_events(kind=KIND_CHAIN_HEAD)
    cuts = [e for e in events if e[1] == 0]
    posts = [e for e in events if e[1] == 5]
    assert len(cuts) == 1 and posts == []
    assert 70.0 <= cuts[0][2] <= 85.0
    assert sim.score == cuts[0][2]


def test_arc_grants_free_pre_and_post():
    """free_pre pays the full 70 with no wind-up; free_post pays the full
    30 at the cut tick with no tracker (arc head/tail semantics)."""
    n = _note(DT, 3, 1, free_pre=True, free_post=True)
    sim = _sim_with_note(n)
    _set_flat(sim, 1, 0.0, om_roll=-15.0)     # one tick of swing only
    ev, r = _coast(sim)
    assert _codes(ev) == [(0, 0)]
    assert ev[0][2] >= 100.0                  # 70 + acc + 30, one event
    for _ in range(6):
        ev, _ = _coast(sim)
        assert all(c != 5 for _, c, *_ in ev)


@pytest.mark.parametrize("free_pre,free_post", [(False, True), (True, False), (True, True)])
def test_arc_attached_chain_head_gets_fixed_post(free_pre, free_post):
    """Scoring v4 (2026-10-02): a chain head that is also an arc head or
    tail (the game's ChainHeadArcHead / ChainHeadArcTail types) gets the
    30 follow-through points as a fixed grant at the cut tick (max 115,
    BeatLeader's max score), and opens no tracker; an arc tail also
    grants the full 70 pre-swing."""
    n = _note(DT, 3, 1, kind=KIND_CHAIN_HEAD, free_pre=free_pre,
              free_post=free_post)
    sim = _sim_with_note(n)
    _set_flat(sim, 1, 0.0, om_roll=-15.0)     # one tick of swing only
    ev, r = _coast(sim)
    assert _codes(ev) == [(0, 0)]
    pts, pre = ev[0][2], ev[0][3]
    if free_pre:
        assert pre == 70.0
    assert pts == pytest.approx(pre + ev[0][4] + 30.0)
    for _ in range(6):
        ev, _ = _coast(sim)
        assert all(c != 5 for _, c, *_ in ev)


def test_bomb_hit_and_bomb_pass():
    """Sweeping through a bomb = penalty event (code 6); letting it pass
    is silent and free."""
    bomb = _note(DT, 3, 1, kind=KIND_BOMB, any_dir=True, color=0)
    sim = _sim_with_note(bomb)
    _set_flat(sim, 1, 0.0, om_roll=-15.0)
    ev, r = _coast(sim)
    assert _codes(ev) == [(0, 6)]
    # env v7.8: bombs cost 3x a bad cut/miss (combo-break repricing)
    assert r == pytest.approx(-1.5 + ALIVE)
    assert sim.bombs_hit == 1 and sim.bad_cuts == 0 and sim.n_notes == 0
    # pass: park both arms, let it fly through the miss boundary
    sim = _sim_with_note(bomb)
    _park(sim, 0)
    _park(sim, 1)
    events = []
    done = False
    while not done:
        _, _, done, ev = sim.step(list(COAST))
        events.extend(ev)
    assert events == []
    assert sim.misses == 0 and sim.bombs_hit == 0
    assert sim.status()[0] == 4               # passed clean


def test_bomb_ignores_wide_flat_blade():
    """The bomb hitbox is a small r=0.18 sphere at the center: a blade
    passing ~0.38 m above it doesn't trigger."""
    bomb = _note(0.0, 3, 1, kind=KIND_BOMB, any_dir=True, color=0)
    sim = _sim_with_note(bomb)
    _park(sim, 0)
    _set_flat(sim, 1, 0.0, om_roll=-2.0, delta=1.6)   # y ~1.78 vs 1.40
    ev, _ = _coast(sim)
    assert ev == []


def test_joint_limits_clamp_and_kill_omega():
    """Every joint is bounded; driving one INTO its stop parks it there
    (the limit rows + step-11 clamps). v10: gravity may pull a parked
    joint marginally off the stop within the same tick, so the asserts
    carry a small band instead of the v9 exact equality. Spot-checks
    arm-tier and wrist-tier joints on the right side."""
    sim = _sim_with_note(_note(50.0, 2, 1))   # far note: no interference
    checks = [
        (A_R_ARM + 0, 1.10, 8.0, np.radians(65.0)),    # pitch into +65
        (A_R_ARM + 1, 1.70, 8.0, np.radians(100.0)),   # r yaw into +100
        (A_R_ARM + 2, -1.15, -8.0, np.radians(-70.0)), # r twist into -70
        (A_R_ARM + 3, 0.05, -8.0, 0.0),                # elbow into its floor
        (A_R_ARM + 5, 1.60, 15.0, np.radians(95.0)),   # bend into +95
        (A_R_ARM + 6, 0.50, 15.0, np.radians(30.0)),   # r dev into +30
    ]
    for j, q0, om0, lim in checks:
        joints = np.zeros(N_JOINTS)
        joints[j] = q0
        q = np.zeros(NQPOS)
        q[:3] = ROOT_POS0
        q[3] = 1.0
        q[7:] = joints
        v = np.zeros(NV)
        v[6 + j] = om0
        sim.set_joints(list(q), list(v))
        sim.step(list(COAST))
        qq = np.asarray(sim.qpos())[7:]
        om = np.asarray(sim.omega())
        lo_j, hi_j = (lim, np.inf) if om0 < 0 else (-np.inf, lim)
        # v14: stops are reciprocal-impulse springs with a 2 mrad rest
        # slop (contact spec §9) - transient sub-slop penetration
        # replaces the old reaction-free teleport
        assert lo_j - 2.1e-3 <= qq[j] <= hi_j + 2.1e-3, f"joint {j} crossed"
        assert abs(qq[j] - lim) <= 0.02, f"joint {j} not at its stop"
        assert abs(om[j]) <= 0.5, f"joint {j} omega not killed"


def test_miss_fires_exactly_at_exit():
    """Misses register when the note crosses the LIVE head plane (env
    v10: z < head_z(t), ~-0.65 near the reference pose) — exactly once,
    a few ticks after the hit plane."""
    sim = _sim_with_note(_note(DT, 0, 2))
    _park(sim, 0)
    _park(sim, 1)
    events = []
    for _ in range(40):
        ev, _ = _coast(sim)
        events.append(ev)
    miss_ticks = [i for i, ev in enumerate(events) if (0, 4) in _codes(ev)]
    assert len(miss_ticks) == 1
    # head plane trip at speed 4 from the reference head: 0.65/4 s ~ 4.9
    # ticks after the plane; the sagging body moves the head by < ~0.15 m
    # inside this window (+-1 tick band)
    t_miss = (miss_ticks[0] + 1) * DT
    w = 0.65 / 4.0
    assert w - DT <= t_miss - DT <= w + 2 * DT + 1e-9
    assert sim.misses == 1


def test_note_speed_scales_the_time_window():
    """Faster notes reach the miss line sooner (less time near the plane)."""
    ticks = {}
    for speed in (4.0, 8.0):
        sim = _sim_with_note(_note(DT, 0, 2, speed=speed))
        _park(sim, 0)
        _park(sim, 1)
        t = 0
        while sim.status()[0] == 0:
            sim.step(list(COAST))
            t += 1
        ticks[speed] = t
    assert ticks[8.0] < ticks[4.0]


# --- chart generation --------------------------------------------------------

def test_chart_tick_lattice_and_counts():
    """Generator v2: spawn counts track the per-tick probabilities, the
    chart is time-sorted, dots run ~1 in 10 with parser-convention
    angle 0, and bombs are directionless."""
    cfg = ChartConfig(colors="both", chart_len_s=60.0)
    chart = generate_chart(cfg, np.random.default_rng(11))
    notes = [n for n in chart if n.kind == KIND_NOTE]
    bombs = [n for n in chart if n.kind == KIND_BOMB]
    # tick count follows the speed walk (interval stretches by
    # note_speed_max/speed): ~2.3 notes/s both-hands at the defaults
    assert 90 <= len(notes) <= 250
    assert 15 <= len(bombs) <= 70
    # extras can precede their original by <= EXTRA_DZ / speed_min
    from beat_arena_env.chart import EXTRA_DZ
    assert all(n.hit_time >= cfg.first_note_s - EXTRA_DZ / cfg.note_speed_min
               for n in chart)
    assert all(a.hit_time <= b.hit_time for a, b in zip(chart, chart[1:]))
    dots = [n for n in notes if n.any_dir]
    assert 0.03 <= len(dots) / len(notes) <= 0.25
    assert all(n.angle == 0.0 for n in dots)
    assert all(b.any_dir and b.angle == 0.0 for b in bombs)
    none = generate_chart(ChartConfig(note_prob=0.0, bomb_prob=0.0),
                          np.random.default_rng(11))
    assert none == []
    # speed-paced lattice: slow sections are sparser than fast ones
    long = generate_chart(ChartConfig(colors="both", chart_len_s=120.0),
                          np.random.default_rng(2))
    lnotes = [n for n in long if n.kind == KIND_NOTE]
    ts = np.array([n.hit_time for n in lnotes])

    def local_nps(subset):
        return np.mean([(np.abs(ts - n.hit_time) < 2.0).sum() - 1
                        for n in subset]) / 4.0

    slow = [n for n in lnotes if n.speed < 4.0]
    fast = [n for n in lnotes if n.speed > 6.0]
    assert len(slow) > 10 and len(fast) > 10
    assert local_nps(fast) > 1.5 * local_nps(slow)


def test_chart_parity_flow_angles():
    """Consecutive same-hand originals mostly oppose: flip 180 deg +
    N(0, 45 deg) puts ~68% within 45 deg of perfect parity and makes
    big reversals rare but possible."""
    cfg = ChartConfig(colors="both", chart_len_s=120.0, extra_prob=0.0,
                      dot_prob=0.0, bomb_prob=0.0)
    chart = generate_chart(cfg, np.random.default_rng(13))
    devs = []
    for color in (0, 1):
        angs = [n.angle for n in chart if n.color == color]
        assert len(angs) > 80
        for a, b in zip(angs, angs[1:]):
            d = (b - (a + np.pi)) % (2 * np.pi)
            devs.append(min(d, 2 * np.pi - d))
    devs = np.array(devs)
    assert (devs <= np.pi / 4 + 1e-9).mean() > 0.55     # ~0.68 expected
    assert (devs <= np.pi / 2 + 1e-9).mean() > 0.85     # ~0.955 expected
    assert 0.2 < devs.mean() < 1.0                      # offsets are real
    assert len({round(a, 3) for a in devs}) > 50        # and continuous


def test_chart_speed_walk_shared():
    """Speeds stay inside [note_speed_min, note_speed_max] and step at
    most SPEED_DEV per spawn — across colors AND bombs (one shared
    walk), which keeps the two hands' speeds similar."""
    from beat_arena_env.chart import SPEED_DEV
    cfg = ChartConfig(colors="both", chart_len_s=60.0, extra_prob=0.0,
                      dot_prob=0.0)
    chart = generate_chart(cfg, np.random.default_rng(17))
    assert all(cfg.note_speed_min <= n.speed <= cfg.note_speed_max
               for n in chart)
    # no extras: time-sorted order == spawn order (stable sort on ties)
    speeds = [n.speed for n in chart]
    assert all(abs(b - a) <= SPEED_DEV + 1e-9
               for a, b in zip(speeds, speeds[1:]))
    assert len({round(s, 3) for s in speeds}) > 30      # a real walk
    pinned = generate_chart(
        ChartConfig(note_speed_min=3.0, note_speed_max=3.0),
        np.random.default_rng(3))
    assert all(n.speed == 3.0 for n in pinned)


def test_chart_extra_notes():
    """extra_prob: a replica along the original's swing line — same
    speed/color, near-same time (+-0.1 m of z), <= 45 deg rotation
    offset, <= 0.5 m perpendicular offset, and the along-line point
    inside the spawn rectangle (the discard rule)."""
    from beat_arena_env.chart import (EXTRA_DZ, EXTRA_PATH, EXTRA_PERP,
                                      EXTRA_ROT, _pos_ranges)
    from beat_arena_env.constants import note_center as nc
    cfg = ChartConfig(colors="right", chart_len_s=120.0, extra_prob=1.0,
                      dot_prob=0.0, bomb_prob=0.0)
    chart = generate_chart(cfg, np.random.default_rng(29))
    x_lo, x_hi, y_lo, y_hi = _pos_ranges(cfg)
    # pair members share an exact speed (replica) and sit within the z
    # shift of each other; distinct spawns differ in speed almost surely
    pairs, singles, used = [], [], set()
    for i, n in enumerate(chart):
        if i in used:
            continue
        if (i + 1 < len(chart) and chart[i + 1].speed == n.speed
                and abs(chart[i + 1].hit_time - n.hit_time)
                <= EXTRA_DZ / cfg.note_speed_min + 1e-9):
            pairs.append((n, chart[i + 1]))
            used.add(i + 1)
        else:
            singles.append(n)
    assert len(pairs) >= 20                     # extras actually survive
    assert len(singles) >= 20                   # and discards actually happen

    def is_valid_pair(orig, extra):
        d_ang = abs((extra.angle - orig.angle + np.pi) % (2 * np.pi) - np.pi)
        if d_ang > EXTRA_ROT + 1e-9:
            return False
        ox, oy = nc(orig)
        expos = np.array(nc(extra)) - np.array([ox, oy])
        u = np.array([math.cos(orig.angle), math.sin(orig.angle)])
        along = float(expos @ u)
        perp = float(expos @ np.array([-u[1], u[0]]))
        if not (abs(along) <= EXTRA_PATH + 1e-9
                and abs(perp) <= EXTRA_PERP + 1e-9):
            return False
        return (x_lo - 1e-9 <= ox + along * u[0] <= x_hi + 1e-9
                and y_lo - 1e-9 <= oy + along * u[1] <= y_hi + 1e-9)

    for a, b in pairs:
        assert a.color == b.color and a.kind == b.kind
        assert is_valid_pair(a, b) or is_valid_pair(b, a)


def test_chart_positions_are_continuous():
    """Env v7.5: generated positions are continuous over the allowed
    cols/rows span padded by 0.2 m on every side (outer-edge coverage);
    restricting to a single col narrows x to a +-0.2 m corridor.
    (extra_prob 0: extras may overhang the rectangle by <= 0.5 m.)"""
    from beat_arena_env.constants import note_center as nc
    cfg = ChartConfig(colors="both", chart_len_s=60.0, extra_prob=0.0)
    chart = generate_chart(cfg, np.random.default_rng(23))
    xs = [nc(n)[0] for n in chart]
    ys = [nc(n)[1] for n in chart]
    from beat_arena_env.constants import PLAYER_HEIGHT_OFFSET as _PH
    assert all(-1.1 - 1e-9 <= x <= 1.1 + 1e-9 for x in xs)
    assert all(0.65 + _PH - 1e-9 <= y <= 2.1 + _PH + 1e-9 for y in ys)
    assert len({round(x, 3) for x in xs}) > 20      # actually continuous
    assert len({round(y, 3) for y in ys}) > 20
    assert max(xs) > 0.9 or min(xs) < -0.9          # the margin gets used
    pinned = generate_chart(ChartConfig(chart_len_s=40.0, cols=(2,),
                                        rows=(0, 1, 2), extra_prob=0.0),
                            np.random.default_rng(5))
    assert all(abs(nc(n)[0] - col_x(2)) <= 0.2 + 1e-9 for n in pinned)
    assert len({round(nc(n)[1], 3) for n in pinned}) > 5


def test_chain_links_layout():
    from beat_arena_env.chart import chain_links
    head = _note(1.0, 0, 0, angle=0.0)        # pointing +x from (-0.9, 0.85)
    links = chain_links(head, col_x(2), ROW_Y[0], 1.3, sc=4)
    assert len(links) == 3
    ts = [l.hit_time for l in links]
    assert ts == sorted(ts) and 1.0 < ts[0] and ts[-1] == pytest.approx(1.3)
    xs = [l.x for l in links]
    assert all(a < b for a, b in zip(xs, xs[1:]))         # marches toward tail
    assert xs[-1] == pytest.approx(col_x(2))              # ends at the tail
    assert all(l.kind == KIND_CHAIN_LINK and l.any_dir for l in links)


# --- BeatSaver parser (fixtures cover all three schemas + v7 objects) --------

def _write_info(tmp_path, extra=None):
    import json
    info = {"_beatsPerMinute": 120.0, "_songName": "fixture",
            "_difficultyBeatmapSets": [{
                "_beatmapCharacteristicName": "Standard",
                "_difficultyBeatmaps": [
                    {"_difficulty": "Normal", "_beatmapFilename": "Normal.dat",
                     "_noteJumpMovementSpeed": 10.0}]}]}
    info.update(extra or {})
    (tmp_path / "Info.dat").write_text(json.dumps(info))


def test_beatsaver_parser_v2_bombs_and_dots(tmp_path):
    import json
    from beat_arena_env.beatsaver import load_beatmap
    _write_info(tmp_path)
    v2 = {"_version": "2.0.0", "_notes": [
        {"_time": 4, "_lineIndex": 1, "_lineLayer": 0, "_type": 0, "_cutDirection": 1},
        {"_time": 5, "_lineIndex": 2, "_lineLayer": 1, "_type": 3, "_cutDirection": 0},
        {"_time": 6, "_lineIndex": 3, "_lineLayer": 2, "_type": 1, "_cutDirection": 8},
    ]}
    (tmp_path / "Normal.dat").write_text(json.dumps(v2))
    notes, meta = load_beatmap(str(tmp_path), "Normal", first_note_s=2.0)
    assert len(notes) == 3                       # bomb now PARSED
    assert meta["n_notes"] == 2 and meta["n_bombs"] == 1
    assert notes[0].hit_time == pytest.approx(2.0)          # shifted
    assert notes[1].kind == KIND_BOMB
    assert notes[2].hit_time == pytest.approx(3.0)          # 2 beats @120 = 1s
    assert notes[0].angle == pytest.approx(3 * np.pi / 2)   # down
    assert notes[2].any_dir                                  # dot
    assert notes[0].speed == pytest.approx(10.0)             # real NJS


def test_beatsaver_parser_v3_chains_arcs_bpm(tmp_path):
    import json, math
    from beat_arena_env.beatsaver import load_beatmap
    _write_info(tmp_path)
    v3 = {"version": "3.3.0",
          "colorNotes": [
              {"b": 4, "x": 0, "y": 0, "c": 1, "d": 3, "a": 15},
              {"b": 6, "x": 1, "y": 0, "c": 1, "d": 1},   # chain head
              {"b": 8, "x": 2, "y": 1, "c": 0, "d": 0},   # arc head
              {"b": 9, "x": 3, "y": 1, "c": 0, "d": 1},   # arc tail
          ],
          "bombNotes": [{"b": 5, "x": 2, "y": 0}],
          "burstSliders": [
              {"b": 6, "x": 1, "y": 0, "c": 1, "d": 1,
               "tb": 6.5, "tx": 1, "ty": 2, "sc": 4, "s": 1.0}],
          "sliders": [
              {"b": 8, "x": 2, "y": 1, "c": 0, "d": 0, "mu": 1.0,
               "tb": 9, "tx": 3, "ty": 1, "tc": 0, "tmu": 1.0, "m": 0}],
          "bpmEvents": [{"b": 8, "m": 240.0}]}
    (tmp_path / "Normal.dat").write_text(json.dumps(v3))
    notes, meta = load_beatmap(str(tmp_path), "Normal")
    assert notes[0].angle == pytest.approx(math.radians(15))
    heads = [n for n in notes if n.kind == KIND_CHAIN_HEAD]
    links = [n for n in notes if n.kind == KIND_CHAIN_LINK]
    bombs = [n for n in notes if n.kind == KIND_BOMB]
    assert len(heads) == 1 and len(links) == 3 and len(bombs) == 1
    assert all(heads[0].hit_time < l.hit_time for l in links)
    assert sum(n.free_post for n in notes) == 1     # arc head
    assert sum(n.free_pre for n in notes) == 1      # arc tail
    # BPM change at beat 8: beats 8 -> 9 take 60/240 s, not 60/120
    t8 = [n for n in notes if n.free_post][0].hit_time
    t9 = [n for n in notes if n.free_pre][0].hit_time
    assert t9 - t8 == pytest.approx(60.0 / 240.0)
    assert t8 == pytest.approx(8 * 0.5)             # pre-change beats @120


def test_beatsaver_parser_headless_and_single_slice_chains(tmp_path):
    """2026-10-02: a chain with no colour note at its head has no head in
    the game (links only, as BeatSaver and BeatLeader count it), and the
    slice count includes the head, so sc = 1 adds no links."""
    import json
    from beat_arena_env.beatsaver import load_beatmap
    _write_info(tmp_path)
    v3 = {"version": "3.3.0",
          "colorNotes": [{"b": 6, "x": 1, "y": 0, "c": 1, "d": 1}],
          "burstSliders": [
              # headed, sc = 1: the head only
              {"b": 6, "x": 1, "y": 0, "c": 1, "d": 1,
               "tb": 6.5, "tx": 1, "ty": 2, "sc": 1, "s": 1.0},
              # headless, sc = 4: three links, no head
              {"b": 8, "x": 2, "y": 0, "c": 0, "d": 1,
               "tb": 8.5, "tx": 2, "ty": 2, "sc": 4, "s": 1.0}]}
    (tmp_path / "Normal.dat").write_text(json.dumps(v3))
    notes, _ = load_beatmap(str(tmp_path), "Normal")
    heads = [n for n in notes if n.kind == KIND_CHAIN_HEAD]
    links = [n for n in notes if n.kind == KIND_CHAIN_LINK]
    assert len(heads) == 1 and heads[0].color == 1
    assert len(links) == 3 and all(l.color == 0 for l in links)


def test_beatsaver_parser_v3_arc_tail_ignores_tc(tmp_path):
    """v3 slider "tc" is the tail CUT DIRECTION, not a colour: the arc
    tail must match its note whatever tc is (until 2026-10-02 a tc other
    than the arc colour silently dropped the tail's free pre-swing)."""
    import json
    from beat_arena_env.beatsaver import load_beatmap
    _write_info(tmp_path)
    v3 = {"version": "3.2.0",
          "colorNotes": [{"b": 4, "x": 1, "y": 0, "c": 1, "d": 1},
                         {"b": 6, "x": 2, "y": 0, "c": 1, "d": 3}],
          "sliders": [{"b": 4, "c": 1, "x": 1, "y": 0, "d": 1, "mu": 1,
                       "tb": 6, "tx": 2, "ty": 0, "tc": 3, "tmu": 1, "m": 0}]}
    (tmp_path / "Normal.dat").write_text(json.dumps(v3))
    notes, _ = load_beatmap(str(tmp_path), "Normal")
    assert [n.free_post for n in notes] == [True, False]
    assert [n.free_pre for n in notes] == [False, True]


def test_beatsaver_parser_chain_tail_note_is_arc_tail(tmp_path):
    """A colour note at a chain's tail scores like an arc tail (free
    pre-swing); a chain head there is ChainHeadArcTail (115) — the
    BeatLeader max-score rule, which real scores require."""
    import json
    from beat_arena_env import scoresaber
    from beat_arena_env.beatsaver import load_beatmap
    _write_info(tmp_path)
    v3 = {"version": "3.2.0",
          "colorNotes": [{"b": 4, "x": 1, "y": 0, "c": 1, "d": 1},
                         {"b": 6, "x": 2, "y": 0, "c": 1, "d": 3}],
          "burstSliders": [
              {"b": 4, "c": 1, "x": 1, "y": 0, "d": 1, "tb": 6, "tx": 2,
               "ty": 0, "sc": 3, "s": 1},
              {"b": 6, "c": 1, "x": 2, "y": 0, "d": 3, "tb": 6.5, "tx": 3,
               "ty": 0, "sc": 3, "s": 1}]}
    (tmp_path / "Normal.dat").write_text(json.dumps(v3))
    notes, _ = load_beatmap(str(tmp_path), "Normal")
    heads = [n for n in notes if n.kind == KIND_CHAIN_HEAD]
    assert [h.free_pre for h in heads] == [False, True]
    assert [scoresaber.max_unit_points(h) for h in heads] == [85, 115]


def test_beatsaver_parser_reversed_arc(tmp_path):
    """A slider with tailBeat < beat attaches grants with the EARLIER end
    as the head (game + ScoreSaber-viewer behavior): the early note gets
    free_post (arc head), the late one free_pre (arc tail)."""
    import json
    from beat_arena_env.beatsaver import load_beatmap
    _write_info(tmp_path)
    v3 = {"version": "3.3.0",
          "colorNotes": [
              {"b": 4, "x": 0, "y": 1, "c": 0, "d": 1},
              {"b": 6, "x": 3, "y": 1, "c": 0, "d": 0},
          ],
          "sliders": [                    # written tail-first: tb 4 < b 6
              {"b": 6, "x": 3, "y": 1, "c": 0, "d": 0, "mu": 1.0,
               "tb": 4, "tx": 0, "ty": 1, "tc": 0, "tmu": 1.0, "m": 0}]}
    (tmp_path / "Normal.dat").write_text(json.dumps(v3))
    notes, _ = load_beatmap(str(tmp_path), "Normal")
    early = min(notes, key=lambda n: n.hit_time)
    late = max(notes, key=lambda n: n.hit_time)
    assert early.free_post and not early.free_pre   # true arc head
    assert late.free_pre and not late.free_post     # true arc tail


def test_beatsaver_parser_v4(tmp_path):
    import json
    from beat_arena_env.beatsaver import load_beatmap
    info = {"version": "4.0.0",
            "song": {"title": "v4 fixture"},
            "audio": {"bpm": 120.0},
            "difficultyBeatmaps": [
                {"characteristic": "Standard", "difficulty": "Normal",
                 "beatmapDataFilename": "Normal.dat",
                 "noteJumpMovementSpeed": 10.0}]}
    (tmp_path / "Info.dat").write_text(json.dumps(info))
    v4 = {"version": "4.0.0",
          "colorNotes": [{"b": 4, "i": 0}, {"b": 5, "i": 1}],
          "colorNotesData": [
              {"x": 1, "y": 0, "c": 0, "d": 1, "a": 0},
              {"x": 2, "y": 1, "c": 1, "d": 8, "a": 0}],
          "bombNotes": [{"b": 6, "i": 0}],
          "bombNotesData": [{"x": 3, "y": 0}],
          "chains": [{"hb": 5, "tb": 5.5, "i": 1, "ci": 0}],
          "chainsData": [{"tx": 2, "ty": 2, "c": 3, "s": 1.0}],
          "arcs": []}
    (tmp_path / "Normal.dat").write_text(json.dumps(v4))
    notes, meta = load_beatmap(str(tmp_path), "Normal")
    assert meta["n_notes"] == 4                     # 2 notes + 2 links
    assert meta["n_bombs"] == 1
    kinds = sorted(n.kind for n in notes)
    assert kinds == [0, 1, 2, 3, 3]
    head = [n for n in notes if n.kind == KIND_CHAIN_HEAD][0]
    assert head.col == 2 and head.any_dir           # dot head keeps any_dir


def test_beatsaver_angle_snap(tmp_path):
    import json, math
    from beat_arena_env.beatsaver import load_beatmap
    _write_info(tmp_path)
    # two same-color down notes on one beat, diagonal cells (0,0) and
    # (1,1): the line between them runs 42.6 deg above +x; the nearest
    # sense to a down arrow is the descending one (~-137.4 deg), within
    # the 40-deg gate of straight down (deviation ~47.4... rejected!) —
    # use (0,0)+(1,2): steeper line, deviation from down ~28 deg -> snap.
    v3 = {"version": "3.3.0", "colorNotes": [
        {"b": 4, "x": 0, "y": 0, "c": 1, "d": 1},
        {"b": 4, "x": 1, "y": 2, "c": 1, "d": 1},
        {"b": 6, "x": 0, "y": 0, "c": 1, "d": 1},   # control pair: same col
        {"b": 6, "x": 0, "y": 2, "c": 1, "d": 1},
    ]}
    (tmp_path / "Normal.dat").write_text(json.dumps(v3))
    notes, _ = load_beatmap(str(tmp_path), "Normal")
    a, b = notes[0], notes[1]
    dx, dy = col_x(1) - col_x(0), ROW_Y[2] - ROW_Y[0]
    expect = (math.atan2(dy, dx) + math.pi) % (2 * math.pi)   # descending
    assert a.angle == pytest.approx(expect)
    assert b.angle == pytest.approx(expect)
    # control pair shares a column -> no snap: both still straight down
    c, d = notes[2], notes[3]
    assert c.angle == pytest.approx(3 * np.pi / 2)
    assert d.angle == pytest.approx(3 * np.pi / 2)


def test_beatclock_beat_zero_bpm_event_replaces_base(tmp_path):
    """A beat-0 bpmEvent overrides Info.dat's BPM (maps often restate a
    slightly different value, e.g. 171.79 vs 172) and mapper alignment
    spikes (one beat at ~2199 BPM) must integrate exactly — found via
    watch.scoresaber.com's 1 ms note matching on the top03 export."""
    import json
    from beat_arena_env.beatsaver import load_beatmap
    _write_info(tmp_path)                      # Info says 120 bpm
    v3 = {"version": "3.3.0",
          "bpmEvents": [{"b": 0, "m": 60.0},   # beat 0: actually 60 bpm
                        {"b": 2, "m": 2400.0},  # alignment spike
                        {"b": 3, "m": 60.0}],
          "colorNotes": [{"b": 3, "x": 1, "y": 0, "c": 1, "d": 1}]}
    (tmp_path / "Normal.dat").write_text(json.dumps(v3))
    notes, _ = load_beatmap(str(tmp_path), "Normal", first_note_s=None)
    # 2 beats @60 (2.0s) + 1 beat @2400 (0.025s)
    assert notes[0].hit_time == pytest.approx(2.025, abs=1e-6)
