"""env v45: per-hand PREVIOUS-NOTE obs blocks v2 (598..607, OBS_DIM 608 -> 624 since env v46).

Contract (both engines): block s is the most recently RESOLVED scoreable
note OF THAT HAND'S COLOUR (colour 0 -> L, 1 -> R), regardless of which
saber resolved it and regardless of outcome — good cut, slow/angle cut,
wrong-hand cut and miss all record on the note's colour; bombs and
walls never. Lanes: x_rel, y_rel (head-relative, live head), dir_cos,
dir_sin (zero on dot notes), valid. All-zero until a note of that
colour has resolved. No time / outcome / colour / chain / free lanes
(the same-day v44 block's were retired at the user's direction).
Observation-only: rewards/dynamics unchanged.
"""

import math

import numpy as np
import pytest
import torch

import beat_arena_rs
from beat_arena_env.chart import Note, note_tuple
from beat_arena_env.constants import (A_L_ARM, A_R_ARM, DT, EVENT_REASONS,
                                      KIND_BOMB, N_JOINTS, OBS_DIM,
                                      OFF_PREV_L, OFF_PREV_R,
                                      PLAYER_HEIGHT_OFFSET, PREV_FEAT, ROW_Y,
                                      X_NORM, Y_NORM, joint_limits,
                                      note_center)
from beat_arena_env.mirror import mirror_tuples
from tests._helpers import FAR_NOTE, cuda, resync, rust_state

# 2026-09-27 (weld harness removed): the coasting body stands on ground
# contact and tips over after ~1 s (arms parked), so the miss fixtures
# arrive EARLY and FAST — the miss fires ~8 ticks in, while the body
# still stands.
EARLY = 0.2
FAST = 8.0

DOWN = 3 * math.pi / 2
UPLEFT = 3 * math.pi / 4
COAST = [0] * N_JOINTS


def _sim(tuples):
    """Both arms PARKED (down and outward, `_park`): at the early miss
    times the ready-stance arms are still swinging down through the grid
    and would cut the fixtures' notes. Cut scenarios set their own arm."""
    sim = beat_arena_rs.RawBeatSim(max_steps=6900,
                                   gears=5)
    sim.set_chart(tuples)
    _park(sim, 0)
    _park(sim, 1)
    return sim


def _blocks(obs):
    obs = np.asarray(obs)
    return (obs[OFF_PREV_L:OFF_PREV_L + PREV_FEAT],
            obs[OFF_PREV_R:OFF_PREV_R + PREV_FEAT])


def _step_until_event(sim, reason, limit=400):
    for _ in range(limit):
        obs, _r, _d, ev = sim.step(COAST)
        hits = [e for e in ev if e[1] == reason]
        if hits:
            return obs, hits
    raise AssertionError(f"no event with reason {reason} in {limit} ticks")


def _expect_block(sim, note):
    """The block a resolved `note` must produce against the sim's LIVE head."""
    x, y = note_center(note)
    hx, hy = sim.head_pose()[0], sim.head_pose()[1]
    show_dir = not getattr(note, "any_dir", False) or note.kind == 3
    return np.array([(x - hx) / X_NORM, (y - hy) / Y_NORM,
                     math.cos(note.angle) if show_dir else 0.0,
                     math.sin(note.angle) if show_dir else 0.0, 1.0])


# --- layout ------------------------------------------------------------------

def test_layout_contract():
    assert OBS_DIM == 624 and PREV_FEAT == 5
    assert OFF_PREV_L == 598 and OFF_PREV_R == 603
    from policy import (OBS_DIM as P_OBS, SABER_FEAT, _SABER_L_IDX,
                        _SABER_R_IDX)
    assert P_OBS == OBS_DIM and SABER_FEAT == 44   # 37 + 7 tau arm lanes (env v46)
    assert _SABER_L_IDX[-(PREV_FEAT + 7):-7] == tuple(range(598, 603))   # tau arm lanes follow (env v46)
    assert _SABER_R_IDX[-(PREV_FEAT + 7):-7] == tuple(range(603, 608))


def test_reset_all_zero():
    sim = _sim([note_tuple(FAR_NOTE)])
    obs = sim.obs()
    assert len(obs) == OBS_DIM
    l, r = _blocks(obs)
    assert not l.any() and not r.any()
    assert sim.prev_note() == [(0, False), (0, False)]


# --- misses ------------------------------------------------------------------

def test_miss_fills_the_colour_block():
    n = Note(hit_time=EARLY, col=2, row=1, x=0.6, y=1.4, color=1,
             angle=DOWN, speed=FAST)
    sim = _sim([note_tuple(n), note_tuple(FAR_NOTE)])
    obs, hits = _step_until_event(sim, 4)
    assert hits[0][0] == 0
    l, r = _blocks(obs)
    assert not l.any(), "a right-colour miss must not touch the L block"
    np.testing.assert_allclose(r, _expect_block(sim, n), atol=1e-6)
    assert sim.prev_note()[1] == (0, True)


def test_record_persists_through_a_gap_tracking_the_live_head():
    """No time lane: after a coast the block is identical except that
    x/y follow the LIVE head. (The gap is 12 ticks — the v10 weld body
    coasted 200 — inside which the tipping body moves the head by
    centimetres, which is exactly what the x/y lanes must follow.)"""
    n = Note(hit_time=EARLY, col=2, row=1, x=0.6, y=1.4, color=0,
             angle=DOWN, speed=FAST)
    sim = _sim([note_tuple(n), note_tuple(FAR_NOTE)])
    obs0, _ = _step_until_event(sim, 4)
    l0, _ = _blocks(obs0)
    head0 = np.array(sim.head_pose()[:2])
    for _ in range(12):
        obs, _r, done, _ev = sim.step(COAST)
        assert not done
    assert np.abs(np.array(sim.head_pose()[:2]) - head0).max() > 1e-3
    l, r = _blocks(obs)
    assert not r.any()
    np.testing.assert_allclose(l, _expect_block(sim, n), atol=1e-6)
    np.testing.assert_array_equal(l[2:], l0[2:])
    assert sim.prev_note()[0] == (0, True)


def test_dot_note_hides_direction():
    n = Note(hit_time=EARLY, col=1, row=1, x=-0.3, y=1.4, color=0,
             angle=UPLEFT, speed=FAST, any_dir=True)
    sim = _sim([note_tuple(n), note_tuple(FAR_NOTE)])
    obs, _ = _step_until_event(sim, 4)
    l, r = _blocks(obs)
    assert not r.any()
    assert (l[2], l[3]) == (0.0, 0.0) and l[4] == 1.0


def test_bomb_pass_never_updates():
    b = Note(hit_time=EARLY, col=2, row=1, x=0.6, y=1.4, color=0,
             angle=0.0, speed=FAST, kind=KIND_BOMB)
    sim = _sim([note_tuple(b), note_tuple(FAR_NOTE)])
    for _ in range(15):
        obs, _r, done, _ev = sim.step(COAST)
        assert not done
    assert sim.status()[0] == 4, "bomb passed clean"
    l, r = _blocks(obs)
    assert not l.any() and not r.any()
    assert sim.prev_note() == [(0, False), (0, False)]


def test_latest_chart_index_wins_within_a_tick():
    a = Note(hit_time=EARLY, col=0, row=1, x=-0.9, y=1.4, color=1,
             angle=DOWN, speed=FAST)
    b = Note(hit_time=EARLY, col=3, row=1, x=0.9, y=1.4, color=1,
             angle=DOWN, speed=FAST)
    sim = _sim([note_tuple(a), note_tuple(b), note_tuple(FAR_NOTE)])
    obs, hits = _step_until_event(sim, 4)
    assert sorted(h[0] for h in hits) == [0, 1], "both miss on one tick"
    _l, r = _blocks(obs)
    assert sim.prev_note()[1] == (1, True)
    np.testing.assert_allclose(r, _expect_block(sim, b), atol=1e-6)


# --- mirror ------------------------------------------------------------------

def test_mirror_augment_reflects_the_block():
    """The trainer's --mirror-augment mirrors CHARTS (mirror.py), not
    obs; the block must then come out mirrored by construction: L <-> R,
    x and dir_cos negated, y / dir_sin / valid equal."""
    n = Note(hit_time=EARLY, col=3, row=1, x=0.9, y=1.4, color=0,
             angle=UPLEFT, speed=FAST)
    tuples = [note_tuple(n), note_tuple(FAR_NOTE)]
    sim_a = _sim(tuples)
    sim_b = _sim(mirror_tuples(tuples))
    obs_a, _ = _step_until_event(sim_a, 4)
    obs_b, _ = _step_until_event(sim_b, 4)
    assert sim_a.tick == sim_b.tick
    la, ra = _blocks(obs_a)
    lb, rb = _blocks(obs_b)
    assert not ra.any() and not lb.any()
    hxa, hxb = sim_a.head_pose()[0], sim_b.head_pose()[0]
    assert la[0] == pytest.approx((0.9 - hxa) / X_NORM, abs=1e-6)
    assert rb[0] == pytest.approx((-0.9 - hxb) / X_NORM, abs=1e-6)
    assert la[0] + rb[0] == pytest.approx(-(hxa + hxb), abs=1e-5)
    assert la[1] == pytest.approx(rb[1], abs=1e-6)
    assert la[2] == pytest.approx(-rb[2], abs=1e-6)
    assert la[3] == pytest.approx(rb[3], abs=1e-6)
    assert la[4] == rb[4] == 1.0


# --- cuts (deterministic blade placements) ----------------------------------
#
# Bot-free event generator: the test_env.py cut-scenario pose family —
# one arm set to the v9 "aligned wrist frame" FLAT pose (blade in a
# z = const plane, roll velocity sweeping it vertically) against a note
# on the plane, then one coast tick. Same inputs => same events.

_DELTA = 0.7


class _Snap:
    """State captured on the tick an event fired."""
    def __init__(self, obs, e, chart, sim):
        self.obs, self.e, self.chart = obs, e, chart
        self.prev = sim.prev_note()
        self.head = sim.head_pose()

    def expect(self, note):
        x, y = note_center(note)
        show_dir = not getattr(note, "any_dir", False) or note.kind == 3
        return np.array([(x - self.head[0]) / X_NORM,
                         (y - self.head[1]) / Y_NORM,
                         math.cos(note.angle) if show_dir else 0.0,
                         math.sin(note.angle) if show_dir else 0.0, 1.0])


def _set_arm(sim, hand, q7, om7):
    """Overwrite one arm's 7-joint state (test_env.py::_set_arm)."""
    q = np.asarray(sim.qpos())
    v = np.concatenate([sim.twist(), sim.omega()])
    base = A_L_ARM if hand == 0 else A_R_ARM
    q[7 + base:7 + base + 7] = q7
    v[6 + base:6 + base + 7] = om7
    sim.set_joints(list(q), list(v))


def _park(sim, hand):
    """Arm down and outward: blade far below/outside the grid."""
    _set_arm(sim, hand, [-1.2, -1.5 if hand == 0 else 1.5, 0, 0, 0, 0, 0],
             [0.0] * 7)


def _set_flat(sim, hand, azimuth, om_roll=0.0):
    """test_env.py::_set_flat: FLAT pose, blade along `azimuth`."""
    lo, hi = joint_limits(hand)[4]
    roll = (azimuth - np.pi / 2 + np.pi) % (2 * np.pi) - np.pi
    bend = np.pi / 2
    if not (lo <= roll <= hi):
        roll = (azimuth + np.pi / 2 + np.pi) % (2 * np.pi) - np.pi
        bend = -np.pi / 2
        assert lo <= roll <= hi, f"azimuth {azimuth} unreachable for hand {hand}"
    _set_arm(sim, hand, [-np.pi / 2 + _DELTA, 0.0, 0.0, np.pi / 2, roll,
                         bend, 0.0],
             [0.0, 0.0, 0.0, 0.0, om_roll, 0.0, 0.0])


# reason -> (note fields, setup(sim)); geometry = test_env.py's
# test_good_swing_cut_scores_integer_points / test_slow_graze_is_bad /
# test_wrong_direction_swing_is_bad / test_wrong_hand_only_triggers_small_box
_SCENARIOS = {
    "ok": (dict(hit_time=DT, col=3, color=1),
           lambda sim: _set_flat(sim, 1, 0.0, om_roll=-15.0)),
    "slow": (dict(hit_time=DT, col=3, color=1),
             lambda sim: _set_flat(sim, 1, 0.0, om_roll=-0.5)),
    "angle": (dict(hit_time=DT, col=3, color=1),
              lambda sim: _set_flat(sim, 1, 0.0, om_roll=15.0)),
    "wrong_hand": (dict(hit_time=0.0, col=1, color=0, x=0.0),
                   lambda sim: (_park(sim, 0), _set_flat(sim, 1, np.pi))),
}


def _cut_run(reason):
    """Play the fixed scenario for `reason` for one tick; return one
    _Snap per event with that reason (the target note is chart idx 0)."""
    fields, setup = _SCENARIOS[reason]
    # row 1 at the pre-v33 height, as test_env.py pins it
    note = Note(row=1, angle=DOWN, speed=4.0,
                y=ROW_Y[1] - PLAYER_HEIGHT_OFFSET, **fields)
    chart = [note, FAR_NOTE]
    sim = _sim([note_tuple(n) for n in chart])
    setup(sim)
    obs, _r, _d, ev = sim.step(COAST)
    return [_Snap(obs, (i, EVENT_REASONS[c], p), chart, sim)
            for i, c, p, *_ in ev if EVENT_REASONS[c] == reason]


def test_good_cut_records_on_the_note_colour():
    hits = _cut_run("ok")
    assert hits, "fixture produced no good cut"
    s = hits[-1]
    note = s.chart[s.e[0]]
    hand = note.color
    idx, valid = s.prev[hand]
    assert valid and idx >= s.e[0] and s.chart[idx].color == hand
    np.testing.assert_allclose(_blocks(s.obs)[hand], s.expect(s.chart[idx]),
                               atol=1e-6)


def test_bad_and_wrong_hand_cuts_record_on_the_note_colour():
    """Every bad cut — slow, angle AND wrong-hand — lands on the block
    of the NOTE'S colour, never on the other saber's."""
    seen = {}
    for reason in ("slow", "angle", "wrong_hand"):
        for s in _cut_run(reason):
            note = s.chart[s.e[0]]
            hand = note.color
            idx, valid = s.prev[hand]
            # later chart indices resolved on the same tick win, so the
            # record is this note or a later note of the SAME colour
            assert valid and idx >= s.e[0], (reason, s.e, idx)
            assert s.chart[idx].color == hand, (reason, s.e, idx)
            np.testing.assert_allclose(_blocks(s.obs)[hand],
                                       s.expect(s.chart[idx]), atol=1e-6)
            seen[s.e[1]] = seen.get(s.e[1], 0) + 1
    assert set(seen) == {"slow", "angle", "wrong_hand"}, seen


def test_wrong_hand_cut_records_on_the_note_colour_not_the_cutter():
    """A note swept by the OTHER saber's bad box records on the block of
    the note's colour; the cutting saber's block never takes it."""
    found = 0
    for s in _cut_run("wrong_hand"):
        note = s.chart[s.e[0]]
        c = note.color
        idx, valid = s.prev[c]
        assert valid and idx >= s.e[0] and s.chart[idx].color == c
        oidx, ovalid = s.prev[1 - c]
        assert not ovalid or oidx != s.e[0], \
            "the cutting saber's block must not take the note"
        np.testing.assert_allclose(_blocks(s.obs)[c],
                                   s.expect(s.chart[idx]), atol=1e-6)
        found += 1
    assert found, "fixture produced no wrong_hand cut"


# --- policy ------------------------------------------------------------------

def test_policy_arch_marker_forward_and_refusal():
    from policy import NotePolicy, load_policy
    pol = NotePolicy()
    assert pol.arch["obs_dim"] == 624
    sim = _sim([note_tuple(Note(hit_time=EARLY, col=2, row=1, x=0.6, y=1.4,
                                color=1, angle=DOWN, speed=FAST)),
                note_tuple(FAR_NOTE)])
    obs, _ = _step_until_event(sim, 4)
    x = torch.as_tensor(np.asarray(obs)).unsqueeze(0)
    with torch.no_grad():
        logits, value = pol.forward_stacked(x)
    assert logits.shape[:2] == (1, 29) and value.shape == (1,)
    assert torch.isfinite(logits).all() and torch.isfinite(value).all()
    for old_dim in (598, 626):
        old = {"arch": dict(pol.arch, obs_dim=old_dim),
               "state_dict": pol.state_dict()}
        with pytest.raises(SystemExit):
            load_policy(old)


# --- CUDA twins --------------------------------------------------------------

def _cuda_env(charts, max_steps=6900):
    from beat_arena_env.cuda_env import CudaBeatVecEnv
    return CudaBeatVecEnv(len(charts), charts=charts, max_steps=max_steps,
                          gears=5, device="cuda",
                          dtype=torch.float64, auto_reset=False,
                          want_contacts=True)


@cuda
def test_cuda_miss_block_matches_rust():
    n = Note(hit_time=EARLY, col=2, row=1, x=0.6, y=1.4, color=1,
             angle=DOWN, speed=FAST)
    chart = [n, FAR_NOTE]
    env = _cuda_env([chart])
    env.reset()
    sim = _sim([note_tuple(x) for x in chart])
    q, v = rust_state(sim)
    resync(env, [q], [v])          # start from the Rust sim's parked arms
    zero = torch.zeros((1, N_JOINTS), dtype=torch.int64, device="cuda")
    for _ in range(25):            # the miss fires ~8 ticks in (EARLY)
        tobs, *_ = env.step(zero)
        robs, *_ = sim.step(COAST)
        # Discrete lanes bitwise. The head-relative x/y lanes are the
        # same expression on both engines, but the two physics engines
        # drift apart under free-running coast (~1e-2 by tick 130 on
        # the pod gate, 2026-09-15), so — exactly like the base parity
        # gate — the CUDA state is resynced from Rust after every tick
        # and the lanes are compared over one tick of divergence only.
        c = tobs[0, OFF_PREV_L:OBS_DIM].cpu().numpy()
        r = np.asarray(robs)[OFF_PREV_L:OBS_DIM]
        np.testing.assert_array_equal(c[2:5], r[2:5])
        np.testing.assert_array_equal(c[7:10], r[7:10])
        np.testing.assert_allclose(c[[0, 1, 5, 6]], r[[0, 1, 5, 6]],
                                   atol=1e-4)
        q, v = rust_state(sim)
        resync(env, [q], [v])
    assert int(env.prev_valid[0, 1]) == 1 and int(env.prev_idx[0, 1]) == 0
    assert int(env.prev_valid[0, 0]) == 0
    assert sim.prev_note()[1] == (0, True)
