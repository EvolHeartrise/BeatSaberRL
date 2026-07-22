"""Env tests against the Rust sim — THE implementation (SPEC.md §Tests).

Env v5 (blade sabers): the cut scenarios drive the Rust sim directly via
its set_saber test hook (full saber state: wrist + blade pose + angular
velocities) and assert on the event stream (note_idx, code, points).
"""

import numpy as np
import pytest

from beat_arena_env import (
    OBS_DIM, OFF_NOTES, OFF_NOTE_MASK, N_NOTE_SLOTS, NOTE_FEAT,
    ChartConfig, Note, RustBeatEnv, generate_chart, scripted_action,
)
from beat_arena_env.constants import DT, SABER_FEAT, Z_MISS

from beat_arena_rs import RawBeatSim


DOWN = 6 * np.pi / 4  # downward swing angle
COAST = (0, 0, 0, 0, 0, 0)
ACTION_NVEC = [9, 3, 3, 9, 3, 3]


def _note(hit_time, col, row, color=1, angle=DOWN, speed=4.0, any_dir=False):
    return Note(hit_time=hit_time, col=col, row=row, color=color,
                angle=angle, speed=speed, any_dir=any_dir)


def _sim_with_note(note, max_steps=630):
    from beat_arena_env.chart import note_tuple
    sim = RawBeatSim(max_steps=max_steps)
    sim.set_chart([note_tuple(note)])
    return sim


def _park(sim, saber, x, y):
    """Move a saber's wrist far away, blade up (roll 0, bend 90), still."""
    sim.set_saber(saber, x, y, 0.0, 0.0, 0.0, np.pi / 2, 0.0, 0.0)


def _step(sim, saber, pos, vel=(0.0, 0.0), roll=0.0, bend=0.0,
          om_roll=0.0, om_bend=0.0):
    """Set one saber's full state, coast one tick, return (events, reward)."""
    sim.set_saber(saber, pos[0], pos[1], vel[0], vel[1], roll, bend,
                  om_roll, om_bend)
    _, r, _, events = sim.step(*COAST)
    return events, r


def _flat(azimuth):
    """Joint pose for a blade FLAT in the hit plane at the given in-plane
    azimuth: bend +90 deg, roll = azimuth - 90 deg (see SPEC — the swept
    geometry is then identical to the v5.1 in-plane scenarios)."""
    return dict(roll=azimuth - np.pi / 2, bend=np.pi / 2)


def _codes(events):
    return [(i, c) for i, c, _ in events]


# --- obs layout --------------------------------------------------------------

def test_obs_layout():
    env = RustBeatEnv(seed=3)
    obs, _ = env.reset()
    assert obs.shape == (OBS_DIM,) == (140,)
    assert OFF_NOTES == 2 * SABER_FEAT == 20
    assert OFF_NOTE_MASK == OFF_NOTES + N_NOTE_SLOTS * NOTE_FEAT == 128
    # saber block at reset: wrists at (-+0.5, 0), palm down, blades
    # pointing AT THE NOTES (env v5.2 neutral), at rest
    for s, x in ((0, -0.5), (1, 0.5)):
        b = s * SABER_FEAT
        assert obs[b] == pytest.approx(x / 1.0)
        assert obs[b + 1] == 0.0
        assert obs[b + 2] == obs[b + 3] == 0.0            # wrist velocity
        assert obs[b + 4] == pytest.approx(1.0)            # cos(roll=0)
        assert obs[b + 5] == pytest.approx(0.0, abs=1e-6)  # sin(roll=0)
        assert obs[b + 6] == 0.0                           # bend (forward)
        assert obs[b + 7] == obs[b + 8] == 0.0             # angular velocity
        assert obs[b + 9] == 0.0                           # pre-swing meter
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
    Uses the BEND joint: bending moves the blade direction from any pose."""
    env = RustBeatEnv(seed=3)
    env.reset()
    b = 1 * SABER_FEAT + 9   # right saber's swing meter
    obs, _, _, _, _ = env.step((0, 0, 0, 0, 0, 1))   # right: bend +
    m1 = obs[b]
    obs, _, _, _, _ = env.step((0, 0, 0, 0, 0, 1))
    m2 = obs[b]
    assert 0.0 < m1 < m2
    meters = []
    for _ in range(6):        # reverse hard until omega flips sign
        obs, _, _, _, _ = env.step((0, 0, 0, 0, 0, 2))
        meters.append(obs[b])
    # the meter must DIP at the reversal (a fresh swing restarts near 0);
    # afterwards the new swing legitimately re-accumulates
    assert min(meters) < m1


def test_rolling_a_forward_blade_is_not_a_swing():
    """Pure roll at bend 0 spins the saber about its own axis: the blade
    direction never moves, so no swing arc accumulates (correct physics)."""
    env = RustBeatEnv(seed=3)
    env.reset()
    b = 1 * SABER_FEAT + 9
    for _ in range(4):
        obs, _, _, _, _ = env.step((0, 0, 0, 0, 1, 0))   # right: roll CCW
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


# --- cut detection scenarios (Rust sim, via set_saber hook) ------------------
# Standard geometry: note at cell (2,1) = center (0.25, 0), hit_time = DT
# (z = 0 at the end of the first step), down-swing required. The right
# saber's wrist sits at (-0.5, 0) with the blade FLAT in the plane
# pointing +x (azimuth ~ 0 via _flat: bend +90, roll -90 — the natural
# right-hand down-cut orientation), the note passing the blade at 75% of
# its length; CW roll (om_roll < 0) sweeps the contact point DOWNWARD
# through the note. Identical swept geometry to the v5.1 scenarios.

def test_good_swing_cut_scores():
    sim = _sim_with_note(_note(DT, 2, 1))
    ev, r = _step(sim, 1, (-0.5, 0.0), **_flat(0.25), om_roll=-15.0)
    assert _codes(ev) == [(0, 0)]
    assert sim.good_cuts == 1
    pts = ev[0][2]
    # one tick of swing (~28 deg) of pre-swing + near-center accuracy
    assert 20.0 < pts < 45.0
    assert r == pytest.approx(pts / 115.0, abs=1e-5)


def test_gentle_swing_is_good():
    """V_MIN_CUT is 1 m/s at the CONTACT POINT: a slow-ish roll whose
    contact point still moves >= 1 m/s scores."""
    sim = _sim_with_note(_note(DT, 2, 1))
    ev, _ = _step(sim, 1, (-0.5, 0.0), **_flat(0.0), om_roll=-2.0)
    assert _codes(ev) == [(0, 0)]


def test_slow_graze_is_bad():
    sim = _sim_with_note(_note(DT, 2, 1))
    ev, _ = _step(sim, 1, (-0.5, 0.0), **_flat(0.0), om_roll=-0.5)
    assert _codes(ev) == [(0, 1)]  # slow
    assert ev[0][2] == 0.0


def test_wrong_direction_swing_is_bad():
    sim = _sim_with_note(_note(DT, 2, 1))
    ev, _ = _step(sim, 1, (-0.5, 0.0), **_flat(-0.25), om_roll=15.0)
    assert _codes(ev) == [(0, 2)]  # angle (swept upward through a down note)


def test_wrong_hand_is_bad():
    sim = _sim_with_note(_note(DT, 0, 2, color=0))   # LEFT note at (-0.75, 0.5)
    _park(sim, 0, 1.4, -1.2)                          # matching saber far away
    ev, _ = _step(sim, 1, (-1.5, 0.5), **_flat(0.25), om_roll=-15.0)
    assert _codes(ev) == [(0, 3)]


def test_stab_scores_only_accuracy():
    """Translation-only cut: valid, but no swing arc -> accuracy points only."""
    sim = _sim_with_note(_note(DT, 2, 1))
    # blade flat pointing DOWN, wrist gliding down: the mid-blade passes
    # dead through the note center with zero blade rotation
    ev, r = _step(sim, 1, (0.25, 0.5), vel=(0.0, -5.0), **_flat(-np.pi / 2))
    assert _codes(ev) == [(0, 0)]
    assert ev[0][2] == pytest.approx(15.0, abs=1.0)   # accuracy only
    for _ in range(5):                                # follow-through window
        sim.step(*COAST)
    assert sim.score <= 16.0                          # no post points either


def test_full_swing_scores_near_max():
    """A wound-up swing (>100 deg) + follow-through approaches 115 points.

    speed=10 keeps the note out of blade reach until the contact tick, so
    the sweeping blade can't graze its edge a tick early. CW roll at
    12 rad/s with the blade flat: the whole azimuth arc (start ~159 deg,
    contact at 0, follow-through toward -100 deg) fits the right hand's
    azimuth window."""
    sim = _sim_with_note(_note(8 * DT, 2, 1, speed=10.0), max_steps=60)
    om, sweep = 12.0, 0.0
    deltas = []
    for _ in range(8):
        om *= 1.0 - 0.02
        deltas.append(om * DT)
        sweep += om * DT
    az0 = sweep - deltas[-1] / 2   # azimuth crosses 0 mid-tick-8
    sim.set_saber(1, -0.5, 0.0, 0.0, 0.0, az0 - np.pi / 2, np.pi / 2,
                  -12.0, 0.0)
    events = []
    for _ in range(14):
        _, _, _, ev = sim.step(*COAST)
        events.extend(ev)
    cuts = [e for e in events if e[1] == 0]
    posts = [e for e in events if e[1] == 5]
    assert len(cuts) == 1 and len(posts) == 1
    assert cuts[0][2] >= 78.0            # full 70 pre-swing + most accuracy
    assert posts[0][2] == pytest.approx(30.0, abs=0.5)  # full follow-through
    assert sim.score >= 105.0


def test_accuracy_decreases_off_center():
    def stab(x):
        sim = _sim_with_note(_note(DT, 2, 1))
        ev, _ = _step(sim, 1, (x, 0.5), vel=(0.0, -5.0), **_flat(-np.pi / 2))
        assert _codes(ev) == [(0, 0)]
        return ev[0][2]

    assert stab(0.25) > stab(0.45) > 0.0


def test_dot_note_scores_from_any_direction():
    """any_dir (dot) notes: an up-swing (angle-bad for a down note) is good."""
    sim = _sim_with_note(_note(DT, 2, 1, any_dir=True))
    ev, _ = _step(sim, 1, (-0.5, 0.0), **_flat(-0.25), om_roll=15.0)
    assert _codes(ev) == [(0, 0)]


def test_dot_note_obs_is_zero_vector():
    from beat_arena_env.chart import note_tuple
    sim = RawBeatSim(max_steps=630)
    obs = sim.set_chart([note_tuple(_note(2.0, 1, 1, any_dir=True))])
    assert obs[OFF_NOTES + 6] == 0.0 and obs[OFF_NOTES + 7] == 0.0
    assert obs[OFF_NOTES + 8] == 1.0  # still valid


def test_forward_blade_reaches_early_note():
    """The neutral (forward) blade contacts notes ~1 m before the plane
    (far beyond the old +-0.6 slab), while a flat blade can't reach."""
    # note arrives in 0.3 s: z ~ 1.07 m after one tick
    sim = _sim_with_note(_note(0.3, 2, 1))
    ev, _ = _step(sim, 1, (0.25, 0.0), roll=0.0, bend=0.0)  # pointing at it
    assert _codes(ev) == [(0, 1)]   # touched it (stationary -> slow)
    # same wrist, blade flat in the plane: nothing within reach
    sim = _sim_with_note(_note(0.3, 2, 1))
    ev, _ = _step(sim, 1, (0.25, 0.0), **_flat(np.pi / 2))
    assert ev == []


def test_blade_cannot_point_back_at_player():
    """env v5.2: d_z >= cos(95 deg) by construction — a note 0.5 m behind
    the plane is untouchable from ANY pose (max back-reach ~0.37 m)."""
    from beat_arena_env.constants import BEND_LIM
    sim = _sim_with_note(_note(DT, 2, 1))
    _park(sim, 0, -1.5, -1.25)
    _park(sim, 1, 1.5, -1.25)
    for _ in range(4):                       # let it fly to z = -0.4
        sim.step(*COAST)
    # max-bend poses right on top of the note's lane, various rolls
    for roll in (-np.pi / 2, 0.0, np.pi / 4):
        ev, _ = _step(sim, 1, (0.25, 0.9), roll=roll, bend=BEND_LIM)
        assert ev == [] or all(c == 4 for _, c, _ in ev)
    # ... and it eventually falls off the miss line uncut
    while sim.status()[0] == 0:
        sim.set_saber(1, 1.5, -1.25, 0.0, 0.0, 0.0, np.pi / 2, 0.0, 0.0)
        sim.step(*COAST)
    assert sim.misses == 1


def test_note_rotation_affects_collision():
    """env v5.3: the cut volume is the note's ACTUAL cube, rotated by
    (angle - 270 deg) about the approach axis. A forward blade parked
    0.24 m beside the center misses an axis-aligned (down-swing) note
    (half-extent 0.2) but touches a 45-deg-rotated one (its corner
    reaches sqrt(2)*0.2 ~ 0.283)."""
    for angle, touched in ((DOWN, False), (7 * np.pi / 4, True)):
        sim = _sim_with_note(_note(DT, 2, 1, angle=angle))  # center (0.25, 0)
        _park(sim, 0, -1.5, -1.25)
        ev, _ = _step(sim, 1, (0.49, 0.0), roll=0.0, bend=0.0)
        if touched:
            assert _codes(ev) == [(0, 1)]   # stationary touch -> slow
        else:
            assert ev == []


def test_dot_note_box_stays_axis_aligned():
    """Dot notes don't rotate whatever their angle field says: the same
    0.24 m-offset probe that misses a down note misses a 45-deg dot."""
    sim = _sim_with_note(_note(DT, 2, 1, angle=7 * np.pi / 4, any_dir=True))
    _park(sim, 0, -1.5, -1.25)
    ev, _ = _step(sim, 1, (0.49, 0.0), roll=0.0, bend=0.0)
    assert ev == []


def test_far_note_is_untouchable():
    sim = _sim_with_note(_note(1.0, 2, 1))   # z ~ 3.9 after one tick
    ev, _ = _step(sim, 1, (-0.5, 0.0), **_flat(0.25), om_roll=-15.0)
    assert ev == []
    assert sim.status()[0] == 0


def test_wrist_limits_clamp_and_kill_roll():
    """env v5.2: roll is bounded per hand ([-90,190] deg left,
    [-190,90] right), bend +-95 deg; hitting a limit zeroes the omega."""
    sim = _sim_with_note(_note(5.0, 2, 1))   # far note: no interference
    # right saber rolling CCW into its +90 deg limit
    sim.set_saber(1, 1.0, -1.0, 0.0, 0.0, 1.5, 0.0, 15.0, 0.0)
    sim.step(*COAST)
    s = sim.sabers()[1]
    assert s[4] == pytest.approx(np.pi / 2)  # clamped at +90 deg
    assert s[6] == 0.0                       # roll velocity killed
    # left saber rolling CW into its -90 deg limit
    sim.set_saber(0, -1.0, -1.0, 0.0, 0.0, -1.5, 0.0, -15.0, 0.0)
    sim.step(*COAST)
    s = sim.sabers()[0]
    assert s[4] == pytest.approx(-np.pi / 2)
    assert s[6] == 0.0
    # bend clamps at +-95 deg
    sim.set_saber(1, 1.0, -1.0, 0.0, 0.0, 0.0, 1.6, 0.0, 15.0)
    sim.step(*COAST)
    s = sim.sabers()[1]
    assert s[5] == pytest.approx(np.radians(95.0))
    assert s[7] == 0.0


def test_miss_fires_exactly_at_exit():
    sim = _sim_with_note(_note(DT, 0, 2))
    _park(sim, 0, -1.5, -1.25)
    _park(sim, 1, 1.5, -1.25)
    events = []
    for _ in range(10):
        _, _, _, ev = sim.step(*COAST)
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
        _park(sim, 0, -1.5, -1.25)
        _park(sim, 1, 1.5, -1.25)
        t = 0
        while sim.status()[0] == 0:
            sim.step(*COAST)
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


# --- recording (JSONL v4 from the Rust event stream) ------------------------

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
    assert head["header"]["version"] == 5
    assert head["header"]["config"]["note_speed"] == 4.0
    assert len(head["header"]["chart"]) == info["n_notes"]
    goods = sum(1 for t in ticks for e in t["events"]
                if "cut" in e and e["cut"]["good"])
    assert goods == info["good_cuts"]
    assert all(len(t["sabers"]) == 2 and len(t["sabers"][0]) == 8 for t in ticks)
    assert all(len(t["action"]) == 6 for t in ticks)
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
