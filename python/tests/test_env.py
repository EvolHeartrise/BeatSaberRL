"""Env tests against the Rust sim — THE implementation (SPEC.md §Tests).

Env v7 (real geometry/judgment): the cut scenarios drive the Rust sim
via its set_joints test hook (per-hand joint angles + angular
velocities) and assert on the event stream (note_idx, code, points).

Scenario poses use the "aligned wrist frame" family: with
pitch = -elbow + delta, yaw = twist = 0, the wrist frame is a pure
R_x(-delta), which FIXES the world x axis — so a blade along +-x with
roll = azimuth - 90 deg, bend = 90 deg stays exactly in a z = const
plane for ANY delta, and roll sweeps move it vertically. delta = 0.7
parks the right wrist at (0.2, ~1.396, ~-0.195): flat-blade geometry at
row-1 height (y 1.40), just behind the hit plane.
"""

import numpy as np
import pytest

from beat_arena_env import (
    OBS_DIM, OFF_NOTES, OFF_NOTE_MASK, N_NOTE_SLOTS, NOTE_FEAT,
    ChartConfig, Note, RustBeatEnv, generate_chart, scripted_action,
)
from beat_arena_env.constants import (
    DT, KIND_BOMB, KIND_CHAIN_HEAD, KIND_CHAIN_LINK, ROW_Y, SABER_FEAT,
    Z_MISS, col_x, joint_limits,
)
from beat_arena_env.kinematics import fk

from beat_arena_rs import RawBeatSim


DOWN = 6 * np.pi / 4  # downward swing angle
COAST = (0,) * 12
ACTION_NVEC = [3] * 12
DELTA = 0.7
FLAT_ARM = [-np.pi / 2 + DELTA, 0.0, 0.0, np.pi / 2]
FLAT_WRIST = (0.2, 1.3955705, -0.1950711)   # right-hand wrist FK of FLAT_ARM


def _note(hit_time, col, row, color=1, angle=DOWN, speed=4.0, any_dir=False,
          kind=0, x=None, y=None, free_pre=False, free_post=False):
    return Note(hit_time=hit_time, col=col, row=row, color=color,
                angle=angle, speed=speed, any_dir=any_dir, kind=kind,
                x=x, y=y, free_pre=free_pre, free_post=free_post)


def _sim_with_notes(notes, max_steps=690):
    from beat_arena_env.chart import note_tuple
    sim = RawBeatSim(max_steps=max_steps)
    sim.set_chart([note_tuple(n) for n in notes])
    return sim


def _sim_with_note(note, max_steps=690):
    return _sim_with_notes([note], max_steps=max_steps)


def _park(sim, hand):
    """Swing the arm down and outward: blade far below/outside the grid."""
    yaw = -1.5 if hand == 0 else 1.5
    sim.set_joints(hand, [-1.2, yaw, 0.0, 0.0, 0.0, 0.0], [0.0] * 6)


def _set_flat(sim, hand, azimuth, om_roll=0.0, om_bend=0.0, delta=DELTA):
    """FLAT pose: blade along `azimuth` in a z = const plane (aligned
    wrist frame R_x(-delta) fixes +-x; primary rep roll = azimuth - 90
    deg, bend = +90 deg). Since env v7.1 the roll ranges are 180-degree
    arcs, so azimuths whose primary roll is out of range use the
    equivalent representation (roll + 180 deg, bend flipped) — same
    blade direction. For delta = DELTA the right wrist sits at
    FLAT_WRIST; roll sweeps (om_roll) move the blade vertically."""
    from beat_arena_env.constants import joint_limits as jl
    lo, hi = jl(hand)[4]
    roll = (azimuth - np.pi / 2 + np.pi) % (2 * np.pi) - np.pi
    bend = np.pi / 2
    if not (lo <= roll <= hi):
        roll = (azimuth + np.pi / 2 + np.pi) % (2 * np.pi) - np.pi
        bend = -np.pi / 2
        assert lo <= roll <= hi, f"azimuth {azimuth} unreachable for hand {hand}"
    q = [-np.pi / 2 + delta, 0.0, 0.0, np.pi / 2, roll, bend]
    sim.set_joints(hand, q, [0.0, 0.0, 0.0, 0.0, om_roll, om_bend])


def _coast(sim):
    _, r, _, events = sim.step(list(COAST))
    return events, r


def _codes(events):
    return [(i, c) for i, c, _ in events]


# --- kinematics mirror -------------------------------------------------------

def test_python_fk_matches_rust():
    """kinematics.fk is the declared MIRROR of lib.rs fk() — pin it via the
    sim's sabers()/blades() getters on a batch of random poses."""
    sim = _sim_with_note(_note(5.0, 2, 1))
    rng = np.random.default_rng(4)
    for _ in range(20):
        hand = int(rng.integers(0, 2))
        q = [float(rng.uniform(lo, hi)) for lo, hi in joint_limits(hand)]
        sim.set_joints(hand, q, [0.0] * 6)
        _, w, b = fk(hand, q)
        assert np.allclose(sim.sabers()[hand][12:], w, atol=1e-12)
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


# --- obs layout --------------------------------------------------------------

def test_obs_layout():
    env = RustBeatEnv(seed=3)
    obs, _ = env.reset()
    assert obs.shape == (OBS_DIM,) == (286,)
    assert OFF_NOTES == 2 * SABER_FEAT == 46
    assert OFF_NOTE_MASK == OFF_NOTES + N_NOTE_SLOTS * NOTE_FEAT == 270
    # arm block at reset: all joints zero (arm straight forward), so the
    # wrist sits at (+-0.2, 1.47, 0.0) and the blade points +z
    for s, x in ((0, -0.2), (1, 0.2)):
        b = s * SABER_FEAT
        assert obs[b] == 0.0                                # pitch
        assert obs[b + 1] == 0.0                            # yaw
        assert obs[b + 2] == 0.0                            # twist
        assert obs[b + 3] == pytest.approx(-1.0)            # elbow at 0
        assert obs[b + 4] == pytest.approx(1.0)             # cos(roll)
        assert obs[b + 5] == pytest.approx(0.0, abs=1e-6)   # sin(roll)
        assert obs[b + 6] == 0.0                            # bend
        assert not obs[b + 7:b + 13].any()                  # omegas
        assert obs[b + 13] == pytest.approx(x / 1.0)        # wrist x
        assert obs[b + 14] == pytest.approx((1.47 - 1.40) / 0.75)  # wrist y
        assert obs[b + 15] == pytest.approx(0.0, abs=1e-6)  # wrist z
        assert not obs[b + 16:b + 19].any()                 # wrist vel
        assert obs[b + 19] == pytest.approx(0.0, abs=1e-6)  # blade x
        assert obs[b + 20] == pytest.approx(0.0, abs=1e-6)  # blade y
        assert obs[b + 21] == pytest.approx(1.0)            # blade z
        assert obs[b + 22] == 0.0                           # pre-swing meter
    n_pending = min(len(env.chart), N_NOTE_SLOTS)
    assert obs[OFF_NOTE_MASK:].sum() == n_pending
    n = env.chart[0]
    base = OFF_NOTES
    assert obs[base + 13] == 1.0                            # valid
    assert obs[base + 0] == pytest.approx(col_x(n.col))
    assert obs[base + 1] == pytest.approx((ROW_Y[n.row] - 1.40) / 0.75)
    z = min(n.hit_time * n.speed, 12.0)
    assert obs[base + 2] == pytest.approx(z / 8.0)
    assert obs[base + 3] == pytest.approx(-n.speed / 8.0)
    assert obs[base + 6] == pytest.approx(np.cos(n.angle), abs=1e-6)
    assert obs[base + 7] == pytest.approx(np.sin(n.angle), abs=1e-6)
    assert not obs[base + 8:base + 13].any()                # kind/arc flags
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
    Uses the right BEND joint (action index 11)."""
    env = RustBeatEnv(seed=3)
    env.reset()
    b = 1 * SABER_FEAT + 22   # right arm's swing meter
    bend_plus = (0,) * 11 + (1,)
    bend_minus = (0,) * 11 + (2,)
    obs, _, _, _, _ = env.step(bend_plus)
    m1 = obs[b]
    obs, _, _, _, _ = env.step(bend_plus)
    m2 = obs[b]
    assert 0.0 < m1 < m2
    meters = []
    for _ in range(6):        # reverse hard until omega flips sign
        obs, _, _, _, _ = env.step(bend_minus)
        meters.append(obs[b])
    assert min(meters) < m1


def test_z_is_time_times_speed():
    """time_until = z / |vz| at every step (spatial obs contract)."""
    env = RustBeatEnv(seed=9)
    obs, _ = env.reset()
    for _ in range(60):
        obs, _, done, _, _ = env.step(COAST)
        if done:
            break
        status = env.sim.status()
        pending = [i for i, s in enumerate(status) if s == 0][:N_NOTE_SLOTS]
        for slot, i in enumerate(pending):
            z_obs = obs[OFF_NOTES + slot * NOTE_FEAT + 2] * 8.0
            vz_obs = obs[OFF_NOTES + slot * NOTE_FEAT + 3] * 8.0
            t_until = env.chart[i].hit_time - env.sim.time
            if -3.0 < t_until * env.chart[i].speed < 12.0:
                assert z_obs / -vz_obs == pytest.approx(t_until, abs=1e-4)


# --- determinism -------------------------------------------------------------

def test_determinism():
    def run():
        env = RustBeatEnv(colors="both", note_rate=1.5, bomb_rate=0.3,
                          chain_prob=0.3, arc_prob=0.3, seed=7)
        env.reset(seed=7)
        rng = np.random.default_rng(1)
        out = []
        done = False
        while not done:
            _, r, done, _, info = env.step(rng.integers(0, ACTION_NVEC))
            out.append((r, tuple(info["events"])))
        return out

    assert run() == run()


# --- cut detection scenarios (Rust sim, via set_joints hook) -----------------
# Standard geometry: note at cell (3,1) = center (0.9, 1.40), hit_time =
# DT (z = 0 at the end of the first step). The right arm holds the FLAT
# pose (wrist ~(0.2, 1.396, -0.195), blade along +x at row-1 height); CW
# roll (om_roll < 0) sweeps the contact region DOWNWARD through the
# note's good box (0.8 x 0.5 x 1.0, shifted 0.25 toward the player).

def test_good_swing_cut_scores_integer_points():
    sim = _sim_with_note(_note(DT, 3, 1))
    _set_flat(sim, 1, 0.0, om_roll=-15.0)
    ev, r = _coast(sim)
    assert _codes(ev) == [(0, 0)]
    assert sim.good_cuts == 1
    pts = ev[0][2]
    # ~28 deg of pre-swing (-> ~20) + a few accuracy points (the flat
    # blade plane sits 0.195 m behind the note center)
    assert 15.0 <= pts <= 45.0
    assert pts == int(pts)                    # real BS scores are integers
    assert r == pytest.approx(pts / 115.0, abs=1e-5)


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
    at note_z = 0 the flat blade plane (z -0.195) is outside it, but a
    note just past the plane (z ~ -0.13) reaches it -> wrong_hand."""
    # hit_time = 0: after one tick the note sits at z = -0.133
    sim = _sim_with_note(_note(0.0, 1, 1, color=0))   # LEFT note at (-0.3, 1.4)
    _park(sim, 0)                                     # matching arm far away
    _set_flat(sim, 1, np.pi)                          # right blade points -x
    ev, _ = _coast(sim)
    assert _codes(ev) == [(0, 3)]
    # same geometry but note still AT the plane: small box (+-0.175 z)
    # doesn't reach the blade plane -> no contact at all
    sim = _sim_with_note(_note(DT, 1, 1, color=0))
    _park(sim, 0)
    _set_flat(sim, 1, np.pi)
    ev, _ = _coast(sim)
    assert ev == []


def test_good_box_is_bigger_than_bad_box():
    """The same blade placement is INSIDE the good box for the matching
    hand but OUTSIDE the small box for the wrong hand: the blade plane
    0.195 m behind the note center clears the small box's 0.175 but not
    the good box's 1.0 m depth (a static probe keeps the sweep from
    tilting into the small box)."""
    sim = _sim_with_note(_note(DT, 3, 1))
    _set_flat(sim, 1, 0.0)                     # static matching blade
    ev, _ = _coast(sim)
    assert _codes(ev) == [(0, 1)]              # inside good box -> slow
    sim = _sim_with_note(_note(DT, 3, 1, color=0))
    _park(sim, 0)
    _set_flat(sim, 1, 0.0)                     # static wrong-color blade
    ev, _ = _coast(sim)
    assert ev == []                            # small box out of reach


def test_early_cut_through_the_deep_box():
    """The good box extends 0.75 m toward the player: a fast swing
    connects while the note is still ~0.5 m out (a real-BS early cut),
    but not at ~0.8 m."""
    for t_hit, hit in ((0.13 + DT, True), (0.21 + DT, False)):
        sim = _sim_with_note(_note(t_hit, 3, 1))    # z after 1 tick: 4*t
        _set_flat(sim, 1, 0.0, om_roll=-15.0)
        ev, _ = _coast(sim)
        assert bool(ev) == hit, f"hit_time {t_hit}"
        if hit:
            assert _codes(ev) == [(0, 0)]


def test_dot_note_has_taller_hitbox():
    """Modern dots are 0.8 m tall vs 0.5 for arrow notes: a blade passing
    0.38 m above the row-1 center clips the dot box only (delta = 1.6
    parks the wrist at y ~1.78, z ~ -0.31)."""
    for any_dir, hit in ((False, False), (True, True)):
        sim = _sim_with_note(_note(0.0, 3, 1, any_dir=any_dir))
        _park(sim, 0)
        _set_flat(sim, 1, 0.0, om_roll=-15.0, delta=1.6)
        ev, _ = _coast(sim)
        assert bool(ev) == hit, f"any_dir {any_dir}"


def test_arm_swing_cuts_and_accuracy_decreases_off_center():
    """A pure SHOULDER-pitch pendulum (wrist joints untouched) swings the
    whole arm+blade through a note and scores. Yawing the arm off the
    note's column keeps the hit but costs accuracy points."""
    def pendulum(yaw):
        sim = _sim_with_note(_note(0.24, 2, 2))       # center (0.3, 1.9)
        _park(sim, 0)
        sim.set_joints(1, [0.35, yaw, 0.0, 0.0, 0.0, 0.0],
                       [-8.0, 0.0, 0.0, 0.0, 0.0, 0.0])
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


def _pitch_pendulum_events(kind=0, ticks=20):
    """A full-arm swing: right arm straight, pitch falling from the
    +65 deg v7.2 cap at full omega WITH the bend joint sweeping down
    from +95 in the same world-x rotation plane — the swing chain
    accumulates their sum (blade elevation starts at 160 deg), so the
    100 deg pre-swing arc is banked before contact with a row-0 note.
    Since env v7.1/v7.2 neither the wrist roll (180 deg fan) nor pitch
    alone (155 deg range) can do this — combined-joint arm swings are
    the mechanism the restrictions force."""
    sim = _sim_with_note(_note(9 * DT, 2, 0, kind=kind), max_steps=60)
    _park(sim, 0)
    sim.set_joints(1, [np.radians(65.0), 0.0, 0.0, 0.0, 0.0,
                       np.radians(95.0)],
                   [-8.0, 0.0, 0.0, 0.0, 0.0, -14.0])
    events = []
    for _ in range(ticks):
        ev, _ = _coast(sim)
        events.extend(ev)
    return sim, events


def test_full_swing_scores_full_pre_and_post():
    """A wound-up ARM swing (>100 deg) earns the full 70 pre-swing and
    the full 30 follow-through (pendulum geometry costs accuracy — by
    design; forward-tilted precise blades are the high-acc technique)."""
    sim, events = _pitch_pendulum_events()
    cuts = [e for e in events if e[1] == 0]
    posts = [e for e in events if e[1] == 5]
    assert len(cuts) == 1 and len(posts) == 1
    assert cuts[0][2] >= 70.0            # full pre-swing banked
    assert posts[0][2] == pytest.approx(30.0)  # full follow-through
    assert sim.score >= 100.0


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
    assert obs[OFF_NOTES + 13] == 1.0  # still valid


def test_forward_blade_reaches_early_note():
    """The start pose (arm forward, blade its extension) contacts notes
    far before the plane, while a flat blade can't reach."""
    # note arrives in 0.3 s: z ~ 1.07 m after one tick; the forward blade
    # spans z 0..1 at (0.2, 1.47) — right next to cell (2,1) at (0.3, 1.4)
    sim = _sim_with_note(_note(0.3, 2, 1))
    _park(sim, 0)
    ev, _ = _coast(sim)                    # right arm still at start pose
    assert _codes(ev) == [(0, 1)]          # touched it (stationary -> slow)
    # same note, blade flat at z ~ -0.195: the box front (z ~ 0.32) is
    # still 0.5 m short of the blade plane -> nothing within reach
    sim = _sim_with_note(_note(0.3, 2, 1))
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
    assert r == pytest.approx(20.0 / 115.0, abs=1e-5)
    for _ in range(6):                        # no post event ever
        ev, _ = _coast(sim)
        assert all(c != 5 for _, c, _p in ev)
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
    sim, events = _pitch_pendulum_events(kind=KIND_CHAIN_HEAD)
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
        assert all(c != 5 for _, c, _p in ev)


def test_bomb_hit_and_bomb_pass():
    """Sweeping through a bomb = penalty event (code 6); letting it pass
    is silent and free."""
    bomb = _note(DT, 3, 1, kind=KIND_BOMB, any_dir=True, color=0)
    sim = _sim_with_note(bomb)
    _set_flat(sim, 1, 0.0, om_roll=-15.0)
    ev, r = _coast(sim)
    assert _codes(ev) == [(0, 6)]
    assert r == pytest.approx(-0.5)
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
    passing 0.38 m above it doesn't trigger."""
    bomb = _note(0.0, 3, 1, kind=KIND_BOMB, any_dir=True, color=0)
    sim = _sim_with_note(bomb)
    _park(sim, 0)
    _set_flat(sim, 1, 0.0, om_roll=-2.0, delta=1.6)   # y ~1.78 vs 1.40
    ev, _ = _coast(sim)
    assert ev == []


def test_joint_limits_clamp_and_kill_omega():
    """Every joint is bounded; hitting a limit zeroes its omega.
    Spot-check pitch (+90), right yaw (+100), elbow (0 floor), right roll
    (+90) and bend (+95) — arm-tier and wrist-tier joints both."""
    sim = _sim_with_note(_note(5.0, 2, 1))   # far note: no interference
    checks = [
        (0, 1.10, 8.0, np.radians(65.0)),             # pitch into +65 (v7.2)
        (1, 1.70, 8.0, np.radians(100.0)),            # right yaw into +100
        (2, -1.15, -8.0, np.radians(-70.0)),          # right twist into -70
        (3, 0.05, -8.0, 0.0),                         # elbow into its floor
        (4, -0.05, 15.0, 0.0),                        # right roll into 0 (v7.1)
        (5, 1.60, 15.0, np.radians(95.0)),            # bend into +95
    ]
    for j, q0, om0, lim in checks:
        q = [0.0] * 6
        om = [0.0] * 6
        q[j], om[j] = q0, om0
        sim.set_joints(1, q, om)
        sim.step(list(COAST))
        s = sim.sabers()[1]
        assert s[j] == pytest.approx(lim), f"joint {j} not clamped"
        assert s[6 + j] == 0.0, f"joint {j} omega not killed"


def test_miss_fires_exactly_at_exit():
    """Misses register when the note crosses z = -Z_MISS (2.65 m — about
    2 m behind the head), not at the plane."""
    sim = _sim_with_note(_note(DT, 0, 2))
    _park(sim, 0)
    _park(sim, 1)
    events = []
    for _ in range(40):
        ev, _ = _coast(sim)
        events.append(ev)
    miss_ticks = [i for i, ev in enumerate(events) if (0, 4) in _codes(ev)]
    assert len(miss_ticks) == 1
    t_miss = (miss_ticks[0] + 1) * DT
    w = Z_MISS / 4.0
    assert w < t_miss - DT <= w + DT + 1e-9
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

def test_chart_respects_hand_gap_and_dirs():
    cfg = ChartConfig(note_rate=3.0, colors="both", dirs=(2, 6),
                      min_hand_gap_s=0.5, chart_len_s=30.0)
    chart = generate_chart(cfg, np.random.default_rng(11))
    assert len(chart) > 20
    assert all(n.angle in (2 * np.pi / 4, 6 * np.pi / 4) for n in chart)
    for color in (0, 1):
        times = [n.hit_time for n in chart if n.color == color]
        assert all(b - a >= 0.5 - 1e-9 for a, b in zip(times, times[1:]))
    assert all(a.hit_time <= b.hit_time for a, b in zip(chart, chart[1:]))


def test_chart_continuous_dirs_and_speed_range():
    cfg = ChartConfig(note_rate=2.0, colors="both", continuous_dirs=True,
                      note_speed=3.0, note_speed_max=6.0, chart_len_s=30.0)
    chart = generate_chart(cfg, np.random.default_rng(13))
    assert len(chart) > 20
    assert all(0.0 <= n.angle < 2 * np.pi for n in chart)
    assert all(3.0 <= n.speed <= 6.0 for n in chart)
    assert len({round(n.angle, 3) for n in chart}) > 10   # actually continuous
    assert len({round(n.speed, 3) for n in chart}) > 10


def test_chart_v7_object_dials():
    """bomb_rate / chain_prob / arc_prob produce their object kinds;
    chains follow the real link layout (times strictly between head and
    tail, positions on the Bezier); arcs pair head free_post with a later
    same-color free_pre."""
    cfg = ChartConfig(note_rate=2.0, colors="both", dirs=(0, 2, 4, 6),
                      chart_len_s=40.0, bomb_rate=0.5, chain_prob=0.4,
                      arc_prob=0.5)
    chart = generate_chart(cfg, np.random.default_rng(17))
    kinds = [n.kind for n in chart]
    assert kinds.count(KIND_BOMB) >= 5
    assert kinds.count(KIND_CHAIN_HEAD) >= 3
    assert kinds.count(KIND_CHAIN_LINK) > kinds.count(KIND_CHAIN_HEAD)
    assert sum(n.free_pre for n in chart) >= 2
    assert sum(n.free_pre for n in chart) == sum(n.free_post for n in chart)
    for n in chart:
        if n.kind == KIND_CHAIN_LINK:
            assert n.any_dir and n.x is not None and n.y is not None
        if n.kind == KIND_BOMB:
            assert n.any_dir
    # same-hand feasibility still holds for non-link objects
    for color in (0, 1):
        times = [n.hit_time for n in chart
                 if n.color == color and n.kind in (0, 2)]
        assert all(b - a >= 0.25 for a, b in zip(times, times[1:]))


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


# --- scripted bot solvability ------------------------------------------------

def test_scripted_bot_solves_easy():
    good, total, score = 0, 0, 0.0
    env = RustBeatEnv(seed=0)
    for seed in range(5):
        env.reset(seed=seed)
        done = False
        while not done:
            _, _, done, _, info = env.step(scripted_action(env))
        good += info["good_cuts"]
        total += info["n_notes"]
        score += info["score"]
    assert good / total >= 0.8
    # real swings, not stabs: mean points per note well above the 15-point
    # stab ceiling
    assert score / total >= 40.0


# --- recording (JSONL v7 from the Rust event stream) ------------------------

def test_recording_roundtrip(tmp_path):
    import json
    path = str(tmp_path / "ep.jsonl")
    env = RustBeatEnv(seed=5, chart_len_s=6.0, bomb_rate=0.3, chain_prob=0.3,
                      arc_prob=0.3, colors="both", note_rate=1.5)
    env.set_record(path)
    env.reset(seed=5)
    done = False
    while not done:
        _, _, done, _, info = env.step(scripted_action(env))
    lines = [json.loads(l) for l in open(path) if l.strip()]
    head, ticks = lines[0], lines[1:]
    assert head["header"]["version"] == 7
    assert head["header"]["config"]["note_speed"] == 4.0
    assert len(head["header"]["chart"]) == len(env.chart)
    assert all("x" in c and "y" in c and "kind" in c
               for c in head["header"]["chart"])
    goods = sum(1 for t in ticks for e in t["events"]
                if "cut" in e and e["cut"]["good"])
    assert goods == info["good_cuts"]
    assert all(len(t["sabers"]) == 2 and len(t["sabers"][0]) == 15
               for t in ticks)
    assert all(len(t["action"]) == 12 for t in ticks)
    pts = sum(e["cut"].get("points", 0.0) for t in ticks for e in t["events"]
              if "cut" in e)
    pts += sum(e["post"]["points"] for t in ticks for e in t["events"]
               if "post" in e)
    assert pts == pytest.approx(info["score"], abs=0.5)


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
    notes, meta = load_beatmap(str(tmp_path), "Normal", speed_scale=0.3,
                               first_note_s=2.0)
    assert len(notes) == 3                       # bomb now PARSED
    assert meta["n_notes"] == 2 and meta["n_bombs"] == 1
    assert notes[0].hit_time == pytest.approx(2.0)          # shifted
    assert notes[1].kind == KIND_BOMB
    assert notes[2].hit_time == pytest.approx(3.0)          # 2 beats @120 = 1s
    assert notes[0].angle == pytest.approx(3 * np.pi / 2)   # down
    assert notes[2].any_dir                                  # dot
    assert notes[0].speed == pytest.approx(3.0)              # njs 10 * 0.3


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
