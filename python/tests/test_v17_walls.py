"""env v17 (unilateral-contact spec §12): walls + the distance-sorted
shared entity window, on the parser and the Rust reference engine.

Parser geometry is pinned against hand-computed ScoreSaber-watch
conversions (WallManager.cs); the sim tests pin the head-in-wall
penalty window (front face reaches the head 0.4/speed after the wall's
nominal time) and the nearest-first window admission incl. the
behind-the-head exclusion. The CUDA mirror is gated separately by the
§5 lockstep tiers."""
import json
import math

import pytest

from beat_arena_env.beatsaver import load_beatmap
from beat_arena_env.chart import Note, Wall, note_tuple, wall_tuple
from beat_arena_env.constants import (DT, N_JOINTS, N_NOTE_SLOTS, NOTE_FEAT,
                                      OBS_DIM, OFF_NOTES, OFF_NOTE_MASK,
                                      WALL_FRONT_OFF, WALL_PEN)
import beat_arena_rs

ZERO_ACT = [0] * N_JOINTS
PLANE_Z0 = -0.65


# --- parser geometry (spec §12a, WallManager.cs verbatim) --------------------

def _mk_map(tmp_path, version, notes, obstacles, bpm=60.0, njs=10.0):
    """Minimal one-difficulty map dir in the requested schema. 60 bpm
    makes beats == seconds, so expected times are literal."""
    d = tmp_path / "map"
    d.mkdir(exist_ok=True)
    if version == "v4":
        info = {"version": "4.0.0", "audio": {"bpm": bpm},
                "song": {"title": "t"},
                "difficultyBeatmaps": [
                    {"characteristic": "Standard", "difficulty": "Normal",
                     "beatmapDataFilename": "Normal.dat",
                     "noteJumpMovementSpeed": njs}]}
        diff = {"colorNotes": [{"b": b, "i": i} for i, (b, *_r) in
                               enumerate(notes)],
                "colorNotesData": [{"x": x, "y": y, "c": c, "d": dd}
                                   for (_b, x, y, c, dd) in notes],
                "obstacles": [{"b": b, "i": i} for i, (b, *_r) in
                              enumerate(obstacles)],
                "obstaclesData": [{"x": x, "y": y, "w": w, "h": h, "d": du}
                                  for (_b, x, y, w, h, du) in obstacles]}
    else:
        info = {"_version": "2.0.0", "_beatsPerMinute": bpm,
                "_songName": "t",
                "_difficultyBeatmapSets": [
                    {"_beatmapCharacteristicName": "Standard",
                     "_difficultyBeatmaps": [
                         {"_difficulty": "Normal",
                          "_beatmapFilename": "Normal.dat",
                          "_noteJumpMovementSpeed": njs}]}]}
        if version == "v3":
            diff = {"colorNotes": [{"b": b, "x": x, "y": y, "c": c, "d": dd}
                                   for (b, x, y, c, dd) in notes],
                    "obstacles": [{"b": b, "x": x, "y": y, "w": w, "h": h,
                                   "d": du}
                                  for (b, x, y, w, h, du) in obstacles]}
        else:  # v2: obstacles as (_time, _lineIndex, _type, _duration, _width)
            diff = {"_notes": [{"_time": b, "_lineIndex": x, "_lineLayer": y,
                                "_type": c, "_cutDirection": dd}
                               for (b, x, y, c, dd) in notes],
                    "_obstacles": [{"_time": b, "_lineIndex": x, "_type": t,
                                    "_duration": du, "_width": w}
                                   for (b, x, t, du, w) in obstacles]}
    (d / "Info.dat").write_text(json.dumps(info), encoding="utf-8")
    (d / "Normal.dat").write_text(json.dumps(diff), encoding="utf-8")
    return str(d)


NOTE1 = [(1.0, 0, 0, 0, 1)]      # one note at beat 1 so the chart is nonempty


def test_v2_full_wall_geometry(tmp_path):
    # _type 0 full wall, lane 1, 2 lanes wide, 3 beats: y=0 h=5 ->
    # h capped 5, world h=3.0 (0.6 scale), top clamp inert; width 1.2;
    # cx = -0.9 + 1*0.6 + (1.2-0.6)/2 = 0.0; cy = 0 + 3.0/2 = 1.5
    _n, meta = load_beatmap(_mk_map(tmp_path, "v2", NOTE1,
                                    [(1.0, 1, 0, 3.0, 2)]), "Normal",
                            first_note_s=2.0)
    (w,) = meta["walls"]
    assert w.cx == pytest.approx(0.0)
    assert w.cy == pytest.approx(1.5)
    assert w.width == pytest.approx(1.2)
    assert w.height == pytest.approx(3.0)
    assert w.length == pytest.approx(3.0 * 10.0)   # 3 s * njs 10
    # note shifted from 1.0 to 2.0 -> wall shifts with it
    assert w.start_time == pytest.approx(2.0)
    assert w.speed == pytest.approx(10.0)


def test_v2_crouch_wall_hangs_from_above(tmp_path):
    # _type 1: y=2 h=3 -> h=min(3, 5-2)=3, world h=1.8, cy=1.2+0.9=2.1;
    # span 1.2..3.0 (duck under)
    _n, meta = load_beatmap(_mk_map(tmp_path, "v2", NOTE1,
                                    [(1.0, 0, 1, 1.0, 4)]), "Normal",
                            first_note_s=2.0)
    (w,) = meta["walls"]
    assert w.cy == pytest.approx(2.1)
    assert w.height == pytest.approx(1.8)
    assert w.cy - w.height / 2 == pytest.approx(1.2)
    assert w.cy + w.height / 2 == pytest.approx(3.0)
    assert w.width == pytest.approx(2.4)


def test_v2_mapping_extensions_wall_skipped(tmp_path):
    _n, meta = load_beatmap(_mk_map(tmp_path, "v2", NOTE1,
                                    [(1.0, 0, 1500, 1.0, 1)]), "Normal")
    assert meta["walls"] == []


def test_v3_top_clamp_and_min_sizes(tmp_path):
    # y=2, h=3 -> world h=1.8 centered 2.1, top 3.0: clamp inert.
    # y=0, h=5 (5-0 cap inert) -> world 3.0, top exactly 3.0: inert.
    # zero-duration wall -> length floored at MinWallSize 0.06.
    _n, meta = load_beatmap(_mk_map(tmp_path, "v3", NOTE1,
                                    [(1.0, 1, 0, 1, 5, 0.0)]), "Normal")
    (w,) = meta["walls"]
    assert w.height == pytest.approx(3.0)
    assert w.cy == pytest.approx(1.5)
    assert w.length == pytest.approx(0.06)


def test_v3_negative_duration_flips(tmp_path):
    # d=-2 at beat 3 == a 2-beat wall starting at beat 1
    _n, meta = load_beatmap(_mk_map(tmp_path, "v3", NOTE1,
                                    [(3.0, 0, 0, 1, 3, -2.0)]), "Normal",
                            first_note_s=2.0)
    (w,) = meta["walls"]
    assert w.start_time == pytest.approx(2.0)      # beat 1 == the note beat
    assert w.length == pytest.approx(2.0 * 10.0)


def test_v4_y_clamp_is_wider(tmp_path):
    # v4 walls clamp y to 0..4 (v2/v3 clamp to 2): y=4 h=3 ->
    # h=min(3, 5-4)=1, world h=0.6, cy=2.4+0.3=2.7 (top 3.0, inert)
    _n, meta = load_beatmap(_mk_map(tmp_path, "v4", NOTE1,
                                    [(1.0, 0, 4, 1, 3, 1.0)]), "Normal")
    (w,) = meta["walls"]
    assert w.cy == pytest.approx(2.7)
    assert w.height == pytest.approx(0.6)


# --- sim: head-in-wall penalty (spec §12b) -----------------------------------

def _sim(chart, walls, max_steps=600):
    """Weld-mode sim (feet welded — never falls, head stays ~reference),
    zero effort so the only reward terms are judgments + wall pen."""
    sim = beat_arena_rs.RawBeatSim(max_steps=max_steps, effort_coef=0.0,
                                   wide_acc=False, gears=5,
                                   foot_contact=False)
    sim.set_chart([note_tuple(n) for n in chart],
                  [wall_tuple(w) for w in walls])
    return sim


# a far-future far-off note keeps the episode alive without scoring
FAR_NOTE = [Note(hit_time=100.0, col=2, row=1, x=2.5, y=1.4, color=0,
                 angle=4.712, speed=4.0)]


def test_head_in_wall_penalty_window():
    # Wall spanning far beyond the grid: front crosses the plane at
    # t=1, at speed 4 it reaches the head 0.4/4 = 0.1 s later, and the
    # back face (len 2 m) passes 2/4 = 0.5 s after that. The box is
    # made HUGE in x/y because the unactuated weld-mode body SLUMPS —
    # by ~1.5 s the head is near the floor — so the test isolates the
    # z window (the x/y test has its own dodged-wall case).
    wall = Wall(start_time=1.0, cx=0.0, cy=0.0, width=20.0, height=40.0,
                length=2.0, speed=4.0)
    sim = _sim(FAR_NOTE, [wall])
    ticks_in, reward_in = 0, 0.0
    for k in range(1, 121):
        _obs, r, done, ev = sim.step(ZERO_ACT)
        assert not done
        codes = [e[1] for e in ev]
        t = k * DT
        dzf = (wall.start_time - t) * wall.speed - PLANE_Z0 - WALL_FRONT_OFF
        inside = dzf <= 0.0 <= dzf + wall.length
        if inside:
            ticks_in += 1
            reward_in += r
            assert 7 in codes, (k, t)
        else:
            assert 7 not in codes, (k, t)
    assert ticks_in > 0
    assert sim.walls_in == ticks_in
    # 2026-08-20: WALL_PEN charges ONCE, on the entry edge — one
    # continuous pass through the wall is one charge no matter how
    # long the head stays inside (walls_in still counts dwell ticks,
    # and the per-tick code-7 events above are unchanged)
    assert sim.wall_enters == 1
    assert reward_in == pytest.approx(WALL_PEN)


def test_dodged_wall_charges_nothing():
    # same wall but off to the side: z-span crosses, x never contains
    # the head
    wall = Wall(start_time=1.0, cx=2.5, cy=1.5, width=0.6, height=3.0,
                length=2.0, speed=4.0)
    sim = _sim(FAR_NOTE, [wall])
    total = 0.0
    for _ in range(120):
        _obs, r, _d, ev = sim.step(ZERO_ACT)
        total += r
        assert all(e[1] != 7 for e in ev)
    assert sim.walls_in == 0
    assert total == 0.0


def test_overlapping_walls_charge_once_per_entry():
    walls = [Wall(start_time=1.0, cx=0.0, cy=0.0, width=20.0, height=40.0,
                  length=2.0, speed=4.0),
             Wall(start_time=1.0, cx=0.0, cy=0.0, width=20.0, height=40.0,
                  length=2.0, speed=4.0)]
    sim = _sim(FAR_NOTE, walls)
    total = 0.0
    for _ in range(120):
        _obs, r, _d, _ev = sim.step(ZERO_ACT)
        total += r
        assert r >= WALL_PEN - 1e-12      # never double-charged in a tick
    assert sim.walls_in > 0
    # 2026-08-20 entry-edge pricing: two perfectly overlapping walls,
    # one continuous entry -> exactly one WALL_PEN across the pass
    assert sim.wall_enters == 1
    assert total == pytest.approx(WALL_PEN)


# --- obs: the shared distance-sorted entity window (spec §12c/d) -------------

def _slots(obs):
    """[(slot_dict, mask)] for the entity slots."""
    out = []
    for s in range(N_NOTE_SLOTS):
        base = OFF_NOTES + s * NOTE_FEAT
        out.append((obs[base:base + NOTE_FEAT],
                    obs[OFF_NOTE_MASK + s]))
    return out


def test_obs_dim_and_wall_token_lanes():
    wall = Wall(start_time=5.0, cx=0.3, cy=1.5, width=1.2, height=3.0,
                length=4.0, speed=4.0)
    sim = _sim(FAR_NOTE, [wall])
    obs = sim.obs()
    assert len(obs) == OBS_DIM == 598
    slots = _slots(obs)
    # two entities pending: the wall (nearer: front at 0.4+5*4-... at
    # t=0 dz_front=5*4+0.4=20.4 vs the note's dz=100*4+0.65) -> slot 0
    (tok0, m0), (tok1, m1) = slots[0], slots[1]
    assert m0 == 1.0 and m1 == 1.0
    assert all(m == 0.0 for _t, m in slots[2:])
    assert tok0[14] == 1.0                      # is_wall
    assert tok0[13] == 1.0                      # valid
    assert tok0[15] == pytest.approx(1.2, abs=1e-6)   # width / X_NORM(1)
    assert tok0[16] == pytest.approx(3.0 / 0.75, abs=1e-5)  # height / Y_NORM
    assert tok0[17] == pytest.approx(4.0 / 8.0, abs=1e-6)   # length / Z_SCALE
    assert tok0[3] == pytest.approx(-4.0 / 8.0, abs=1e-6)   # z-vel
    assert tok1[14] == 0.0 and tok1[8] == 0.0   # the note: not wall/bomb
    assert tok1[15] == tok1[16] == tok1[17] == 0.0


def test_window_admits_nearest_not_soonest():
    # 16 SOONER notes far off to the side (x=4 -> dist >= 4) vs one
    # LATER but slower note dead ahead of the head (x=0, y=1.7, speed
    # 2, hit 0.9 -> dz = 0.9*2 + 0.65 = 2.45). By hit_time it is 17th
    # and the v16 soonest-first fill would crowd it out; by distance
    # it is FIRST, so the v17 sort must put it in slot 0.
    side = [Note(hit_time=0.1 + 0.05 * i, col=3, row=1, x=4.0, y=1.4,
                 color=1, angle=0.0, speed=4.0) for i in range(16)]
    ahead = Note(hit_time=0.9, col=1, row=1, x=0.0, y=1.7, color=0,
                 angle=0.0, speed=2.0)
    sim = _sim(sorted(side + [ahead], key=lambda n: n.hit_time), [])
    slots = _slots(sim.obs())
    assert slots[0][1] == 1.0
    assert abs(slots[0][0][0]) < 1e-6, "dead-ahead note not in slot 0"
    xs = [t[0] for t, m in slots if m == 1.0]
    assert sum(1 for x in xs if abs(x) < 1e-6) == 1


def test_behind_head_consumes_no_slot():
    # a note whose miss boundary passed is judged (status!=0) and gone;
    # a wall fully behind the head must also free its slot
    wall = Wall(start_time=0.1, cx=0.0, cy=1.5, width=4.0, height=3.0,
                length=0.5, speed=4.0)
    sim = _sim(FAR_NOTE, [wall])
    # back face passes the head at t = 0.1 + (0.4 + 0.5)/4 = 0.325 s
    for _ in range(12):                          # 0.4 s
        sim.step(ZERO_ACT)
    slots = _slots(sim.obs())
    live = [(t, m) for t, m in slots if m == 1.0]
    assert len(live) == 1                        # only the far note
    assert live[0][0][14] == 0.0


def test_slots_sorted_by_head_distance():
    # three notes straight ahead at different z: slots must come out
    # nearest-first regardless of insertion/chart order
    notes = [Note(hit_time=t, col=1, row=1, x=0.0, y=1.7, color=0,
                  angle=0.0, speed=4.0) for t in (3.0, 1.0, 2.0)]
    sim = _sim(sorted(notes, key=lambda n: n.hit_time), [])
    slots = _slots(sim.obs())
    zs = [t[2] for t, m in slots if m == 1.0]
    assert len(zs) == 3
    assert zs == sorted(zs)


def test_wall_at_head_outranks_everything():
    wall = Wall(start_time=0.0, cx=0.0, cy=1.5, width=4.0, height=3.0,
                length=8.0, speed=4.0)          # head inside from ~0.1 s
    notes = [Note(hit_time=0.5 + 0.2 * i, col=1, row=1, x=0.3, y=1.5,
                  color=0, angle=0.0, speed=4.0) for i in range(5)]
    sim = _sim(notes, [wall])
    for _ in range(6):                           # 0.2 s: inside the wall
        sim.step(ZERO_ACT)
    slots = _slots(sim.obs())
    assert slots[0][1] == 1.0 and slots[0][0][14] == 1.0  # wall in slot 0


# --- combo rule (spec §12b addendum): entry breaks, stay drains ---------------

def test_wall_enter_is_edge_triggered():
    # two crossings separated by >= 1 tick out -> exactly two entries;
    # overlapping walls forming one continuous span -> exactly one
    big = dict(cx=0.0, cy=0.0, width=20.0, height=40.0, speed=4.0)
    two = [Wall(start_time=1.0, length=0.8, **big),
           Wall(start_time=1.5, length=0.8, **big)]
    sim = _sim(FAR_NOTE, two)
    enters = 0
    for _ in range(90):
        _o, _r, _d, ev = sim.step(ZERO_ACT)
        enters += sum(1 for e in ev if e[1] == 8)
    assert enters == 2 == sim.wall_enters
    assert sim.walls_in > 2          # the drain ran for the whole spans

    overlap = [Wall(start_time=1.0, length=2.0, **big),
               Wall(start_time=1.2, length=1.0, **big)]
    sim = _sim(FAR_NOTE, overlap)
    enters = 0
    for _ in range(90):
        _o, _r, _d, ev = sim.step(ZERO_ACT)
        enters += sum(1 for e in ev if e[1] == 8)
    assert enters == 1 == sim.wall_enters


def test_combo_tracker_wall_semantics():
    from beat_arena_env.scoresaber import ComboTracker
    w = ComboTracker()
    w.on_cut(0)
    w.on_points(0, 100)
    w.consume([(0, "wall", 0.0)] * 5)          # drain ticks: NO break
    assert w.breaks == 0 and w.combo == 1
    w.consume([(0, "wall_enter", 0.0)])        # the entry breaks
    assert w.breaks == 1 and w.combo == 0
    assert w.score == 100.0                    # breaks never touch score
