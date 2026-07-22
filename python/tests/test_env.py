"""Env tests against the Rust sim — THE implementation (SPEC.md §Tests).

Env v6 (kinematic upper body): the cut scenarios drive the Rust sim via
its set_joints test hook (per-hand joint angles + angular velocities) and
assert on the event stream (note_idx, code, points).

Scenario poses lean on the "aligned wrist frame" family: with
pitch = -elbow + delta, yaw = twist = 0, the forearm frame is a pure
R_x(-delta) — for delta = 0 it's the IDENTITY, so (roll, bend) act
exactly like the v5.2 world-frame wrist and all the flat-blade geometry
carries over with the wrist parked at (+-0.2, 0.1, -0.05).
"""

import numpy as np
import pytest

from beat_arena_env import (
    OBS_DIM, OFF_NOTES, OFF_NOTE_MASK, N_NOTE_SLOTS, NOTE_FEAT,
    ChartConfig, Note, RustBeatEnv, generate_chart, scripted_action,
)
from beat_arena_env.constants import DT, SABER_FEAT, Z_MISS, joint_limits
from beat_arena_env.kinematics import fk

from beat_arena_rs import RawBeatSim


DOWN = 6 * np.pi / 4  # downward swing angle
COAST = (0,) * 12
ACTION_NVEC = [3] * 12
FLAT_ARM = [-np.pi / 2, 0.0, 0.0, np.pi / 2]  # identity wrist frame
FLAT_WRIST = (0.2, 0.1, -0.05)                # right-hand wrist FK of FLAT_ARM


def _note(hit_time, col, row, color=1, angle=DOWN, speed=4.0, any_dir=False):
    return Note(hit_time=hit_time, col=col, row=row, color=color,
                angle=angle, speed=speed, any_dir=any_dir)


def _sim_with_note(note, max_steps=630):
    from beat_arena_env.chart import note_tuple
    sim = RawBeatSim(max_steps=max_steps)
    sim.set_chart([note_tuple(note)])
    return sim


def _park(sim, hand):
    """Swing the arm down and outward: blade far below/outside the grid."""
    yaw = -1.5 if hand == 0 else 1.5
    sim.set_joints(hand, [-1.2, yaw, 0.0, 0.0, 0.0, 0.0], [0.0] * 6)


def _set_flat(sim, hand, azimuth, om_roll=0.0, om_bend=0.0, delta=0.0):
    """FLAT pose: blade in the hit plane at `azimuth` (identity wrist
    frame: roll = azimuth - 90 deg, bend = +90 deg), wrist at
    (+-0.2, 0.1, -0.05) for delta = 0."""
    q = [-np.pi / 2 + delta, 0.0, 0.0, np.pi / 2,
         azimuth - np.pi / 2, np.pi / 2]
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
    """The FLAT scenario pose puts the right wrist at (0.2, 0.1, -0.05)
    with the blade exactly in-plane along the requested azimuth."""
    for az in (0.0, np.pi / 2, np.pi):
        _, w, b = fk(1, FLAT_ARM + [az - np.pi / 2, np.pi / 2])
        assert np.allclose(w, FLAT_WRIST, atol=1e-9)
        assert np.allclose(b, (np.cos(az), np.sin(az), 0.0), atol=1e-9)


# --- obs layout --------------------------------------------------------------

def test_obs_layout():
    env = RustBeatEnv(seed=3)
    obs, _ = env.reset()
    assert obs.shape == (OBS_DIM,) == (166,)
    assert OFF_NOTES == 2 * SABER_FEAT == 46
    assert OFF_NOTE_MASK == OFF_NOTES + N_NOTE_SLOTS * NOTE_FEAT == 154
    # arm block at reset: all joints zero (arm straight forward), so the
    # wrist sits at (+-0.2, 0.45, 0.3) and the blade points +z
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
        assert obs[b + 14] == pytest.approx(0.45 / 0.75)    # wrist y
        assert obs[b + 15] == pytest.approx(0.3)            # wrist z
        assert not obs[b + 16:b + 19].any()                 # wrist vel
        assert obs[b + 19] == pytest.approx(0.0, abs=1e-6)  # blade x
        assert obs[b + 20] == pytest.approx(0.0, abs=1e-6)  # blade y
        assert obs[b + 21] == pytest.approx(1.0)            # blade z
        assert obs[b + 22] == 0.0                           # pre-swing meter
    n_pending = min(len(env.chart), N_NOTE_SLOTS)
    assert obs[OFF_NOTE_MASK:].sum() == n_pending
    n = env.chart[0]
    base = OFF_NOTES
    assert obs[base + 8] == 1.0
    assert obs[base + 0] == pytest.approx((n.col - 1.5) / 2.0)
    assert obs[base + 1] == pytest.approx((n.row - 1.0) / 1.5)
    z = min(n.hit_time * n.speed, 12.0)
    assert obs[base + 2] == pytest.approx(z / 8.0)
    assert obs[base + 3] == pytest.approx(-n.speed / 8.0)
    assert obs[base + 6] == pytest.approx(np.cos(n.angle), abs=1e-6)
    assert obs[base + 7] == pytest.approx(np.sin(n.angle), abs=1e-6)
    for slot in range(n_pending, N_NOTE_SLOTS):
        assert not obs[OFF_NOTES + slot * NOTE_FEAT:
                       OFF_NOTES + (slot + 1) * NOTE_FEAT].any()


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


def test_rolling_a_forward_blade_is_not_a_swing():
    """Pure roll at bend 0 spins the saber about its own axis: the blade
    direction never moves, so no swing arc accumulates (correct physics)."""
    env = RustBeatEnv(seed=3)
    env.reset()
    b = 1 * SABER_FEAT + 22
    roll_plus = (0,) * 10 + (1, 0)
    for _ in range(4):
        obs, _, _, _, _ = env.step(roll_plus)
    assert obs[b] == 0.0                                  # meter never moves
    assert obs[1 * SABER_FEAT + 6] == 0.0                 # bend still 0


def test_z_is_time_times_speed():
    """time_until = z / |vz| at every step (spatial obs contract)."""
    env = RustBeatEnv(seed=9)
    obs, _ = env.reset()
    for _ in range(60):
        obs, _, done, _, _ = env.step(COAST)
        if done:
            break
        status = env.sim.status()
        pending = [i for i, s in enumerate(status) if s == 0][:12]
        for slot, i in enumerate(pending):
            z_obs = obs[OFF_NOTES + slot * NOTE_FEAT + 2] * 8.0
            vz_obs = obs[OFF_NOTES + slot * NOTE_FEAT + 3] * 8.0
            t_until = env.chart[i].hit_time - env.sim.time
            if -1.0 < t_until * env.chart[i].speed < 12.0:
                assert z_obs / -vz_obs == pytest.approx(t_until, abs=1e-4)


# --- determinism -------------------------------------------------------------

def test_determinism():
    def run():
        env = RustBeatEnv(colors="both", note_rate=1.5, seed=7)
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
# Standard geometry: note at cell (3,1) = center (0.75, 0), hit_time = DT
# (z = 0 at the end of the first step), down-swing required. The right
# arm holds the FLAT pose (wrist (0.2, 0.1, -0.05), blade in-plane along
# +x); CW roll (om_roll < 0) sweeps the contact point (sample s = 0.5)
# DOWNWARD through the note — the v5 swept geometry on the v6 body.

def test_good_swing_cut_scores():
    sim = _sim_with_note(_note(DT, 3, 1))
    _set_flat(sim, 1, 0.0, om_roll=-15.0)
    ev, r = _coast(sim)
    assert _codes(ev) == [(0, 0)]
    assert sim.good_cuts == 1
    pts = ev[0][2]
    # one tick of swing (~28 deg) of pre-swing + decent accuracy
    assert 18.0 < pts < 45.0
    assert r == pytest.approx(pts / 115.0, abs=1e-5)


def test_gentle_swing_is_good():
    """V_MIN_CUT is 1 m/s at the CONTACT POINT: a slow-ish roll whose
    contact point (s ~ 0.5) still moves >= 1 m/s scores."""
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


def test_wrong_hand_is_bad():
    sim = _sim_with_note(_note(DT, 1, 1, color=0))  # LEFT note at (-0.25, 0)
    _park(sim, 0)                                    # matching arm far away
    _set_flat(sim, 1, np.pi)                         # right blade points -x
    ev, _ = _coast(sim)
    assert _codes(ev) == [(0, 3)]


def test_arm_swing_cuts_and_accuracy_decreases_off_center():
    """env v6: a pure SHOULDER-pitch pendulum (wrist joints untouched)
    swings the whole arm+blade through a note and scores — swings no
    longer need the wrist. Yawing the arm off the note's column keeps the
    hit but costs accuracy points."""
    def pendulum(yaw):
        sim = _sim_with_note(_note(0.24, 2, 2))       # center (0.25, 0.5)
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


def test_full_swing_scores_near_max():
    """A wound-up swing (>100 deg) + follow-through approaches 115 points.

    speed=10 keeps the note out of blade reach until the contact tick.
    CW roll at 12 rad/s with the FLAT blade: the azimuth arc (start
    ~159 deg, contact at 0, follow-through toward -100 deg) fits the
    right hand's roll range."""
    sim = _sim_with_note(_note(8 * DT, 3, 1, speed=10.0), max_steps=60)
    om, sweep = 12.0, 0.0
    deltas = []
    for _ in range(8):
        om *= 1.0 - 0.02
        deltas.append(om * DT)
        sweep += om * DT
    az0 = sweep - deltas[-1] / 2   # azimuth crosses 0 mid-tick-8
    _set_flat(sim, 1, az0, om_roll=-12.0)
    events = []
    for _ in range(14):
        ev, _ = _coast(sim)
        events.extend(ev)
    cuts = [e for e in events if e[1] == 0]
    posts = [e for e in events if e[1] == 5]
    assert len(cuts) == 1 and len(posts) == 1
    assert cuts[0][2] >= 75.0            # full 70 pre-swing + most accuracy
    assert posts[0][2] == pytest.approx(30.0, abs=0.5)  # full follow-through
    assert sim.score >= 105.0


def test_dot_note_scores_from_any_direction():
    """any_dir (dot) notes: an up-swing (angle-bad for a down note) is good."""
    sim = _sim_with_note(_note(DT, 3, 1, any_dir=True))
    _set_flat(sim, 1, 0.0, om_roll=15.0)
    ev, _ = _coast(sim)
    assert _codes(ev) == [(0, 0)]


def test_dot_note_obs_is_zero_vector():
    from beat_arena_env.chart import note_tuple
    sim = RawBeatSim(max_steps=630)
    obs = sim.set_chart([note_tuple(_note(2.0, 1, 1, any_dir=True))])
    assert obs[OFF_NOTES + 6] == 0.0 and obs[OFF_NOTES + 7] == 0.0
    assert obs[OFF_NOTES + 8] == 1.0  # still valid


def test_forward_blade_reaches_early_note():
    """The start pose (arm forward, blade its extension) contacts notes
    far before the plane, while a flat blade can't reach."""
    # note arrives in 0.3 s: z ~ 1.07 m after one tick; the forward blade
    # spans z 0.3..1.3 at (0.2, 0.45) — right next to cell (2,2)
    sim = _sim_with_note(_note(0.3, 2, 2))
    _park(sim, 0)
    ev, _ = _coast(sim)                    # right arm still at start pose
    assert _codes(ev) == [(0, 1)]          # touched it (stationary -> slow)
    # same note, blade flat in the plane: nothing within reach
    sim = _sim_with_note(_note(0.3, 2, 2))
    _park(sim, 0)
    _set_flat(sim, 1, np.pi / 2)
    ev, _ = _coast(sim)
    assert ev == []


def test_note_rotation_affects_collision():
    """env v5.3 on the v6 body: the cut volume is the note's cube rotated
    by (angle - 270 deg). A flat blade passing 0.22 m above the center
    line misses an axis-aligned (down-swing) note but touches one rotated
    57 deg (angle 327 deg), whose corner reaches it."""
    for angle, touched in ((DOWN, False), (np.radians(327.0), True)):
        sim = _sim_with_note(_note(DT, 3, 1, angle=angle))  # center (0.75, 0)
        _park(sim, 0)
        # delta = 20 deg lifts the wrist to (0.2, 0.224, 0.052), blade
        # still exactly in-plane along +x (wrist frame = R_x(-20 deg))
        _set_flat(sim, 1, 0.0, delta=np.radians(20.0))
        ev, _ = _coast(sim)
        if touched:
            assert _codes(ev) == [(0, 1)]   # stationary touch -> slow
        else:
            assert ev == []


def test_dot_note_box_stays_axis_aligned():
    """Dot notes don't rotate whatever their angle field says: the same
    probe that misses a down note misses a 327-deg dot."""
    sim = _sim_with_note(_note(DT, 3, 1, angle=np.radians(327.0),
                               any_dir=True))
    _park(sim, 0)
    _set_flat(sim, 1, 0.0, delta=np.radians(20.0))
    ev, _ = _coast(sim)
    assert ev == []


def test_far_note_is_untouchable():
    sim = _sim_with_note(_note(1.0, 3, 1))   # z ~ 3.9 after one tick
    _set_flat(sim, 1, 0.0, om_roll=-15.0)
    ev, _ = _coast(sim)
    assert ev == []
    assert sim.status()[0] == 0


def test_joint_limits_clamp_and_kill_omega():
    """env v6: every joint is bounded; hitting a limit zeroes its omega.
    Spot-check pitch (+90), right yaw (+100), elbow (0 floor), right roll
    (+90) and bend (+95) — arm-tier and wrist-tier joints both."""
    sim = _sim_with_note(_note(5.0, 2, 1))   # far note: no interference
    checks = [
        (0, 1.55, 8.0, np.pi / 2),                    # pitch into +90
        (1, 1.70, 8.0, np.radians(100.0)),            # right yaw into +100
        (3, 0.05, -8.0, 0.0),                         # elbow into its floor
        (4, 1.50, 15.0, np.pi / 2),                   # right roll into +90
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
    sim = _sim_with_note(_note(DT, 0, 2))
    _park(sim, 0)
    _park(sim, 1)
    events = []
    for _ in range(10):
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


# --- recording (JSONL v6 from the Rust event stream) ------------------------

def test_recording_roundtrip(tmp_path):
    import json
    path = str(tmp_path / "ep.jsonl")
    env = RustBeatEnv(seed=5, chart_len_s=6.0)
    env.set_record(path)
    env.reset(seed=5)
    done = False
    while not done:
        _, _, done, _, info = env.step(scripted_action(env))
    lines = [json.loads(l) for l in open(path) if l.strip()]
    head, ticks = lines[0], lines[1:]
    assert head["header"]["version"] == 6
    assert head["header"]["config"]["note_speed"] == 4.0
    assert len(head["header"]["chart"]) == info["n_notes"]
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


# --- BeatSaver parser (fixtures cover both real schemas) ---------------------

def test_beatsaver_parser_v2_and_v3(tmp_path):
    import json, math
    from beat_arena_env.beatsaver import load_beatmap
    info = {"_beatsPerMinute": 120.0, "_songName": "fixture",
            "_difficultyBeatmapSets": [{
                "_beatmapCharacteristicName": "Standard",
                "_difficultyBeatmaps": [
                    {"_difficulty": "Normal", "_beatmapFilename": "Normal.dat",
                     "_noteJumpMovementSpeed": 10.0}]}]}
    (tmp_path / "Info.dat").write_text(json.dumps(info))

    # v2 schema: down-cut, bomb (skipped), dot note
    v2 = {"_version": "2.0.0", "_notes": [
        {"_time": 4, "_lineIndex": 1, "_lineLayer": 0, "_type": 0, "_cutDirection": 1},
        {"_time": 5, "_lineIndex": 2, "_lineLayer": 1, "_type": 3, "_cutDirection": 0},
        {"_time": 6, "_lineIndex": 3, "_lineLayer": 2, "_type": 1, "_cutDirection": 8},
    ]}
    (tmp_path / "Normal.dat").write_text(json.dumps(v2))
    notes, meta = load_beatmap(str(tmp_path), "Normal", speed_scale=0.3,
                               first_note_s=2.0)
    assert len(notes) == 2                       # bomb dropped
    assert notes[0].hit_time == pytest.approx(2.0)          # shifted
    assert notes[1].hit_time == pytest.approx(3.0)          # 2 beats @120bpm = 1s
    assert notes[0].angle == pytest.approx(3 * np.pi / 2)   # down
    assert notes[0].color == 0 and notes[1].color == 1
    assert notes[1].any_dir                                  # dot
    assert notes[0].speed == pytest.approx(3.0)              # njs 10 * 0.3

    # v3 schema: right-cut with +15deg precision offset
    v3 = {"version": "3.3.0", "colorNotes": [
        {"b": 4, "x": 0, "y": 0, "c": 1, "d": 3, "a": 15},
    ]}
    (tmp_path / "Normal.dat").write_text(json.dumps(v3))
    notes, meta = load_beatmap(str(tmp_path), "Normal")
    assert len(notes) == 1
    assert notes[0].angle == pytest.approx(math.radians(15))  # 0 + 15deg CCW
    assert not notes[0].any_dir
