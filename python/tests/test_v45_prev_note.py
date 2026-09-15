"""env v45: per-hand PREVIOUS-NOTE obs blocks v2 (598..607, OBS_DIM 608).

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
from beat_arena_env.constants import (KIND_BOMB, N_JOINTS, OBS_DIM,
                                      OFF_PREV_L, OFF_PREV_R, PREV_FEAT,
                                      X_NORM, Y_NORM, note_center)
from beat_arena_env.mirror import mirror_tuples

cuda = pytest.mark.skipif(not torch.cuda.is_available(),
                          reason="needs CUDA")

DOWN = 3 * math.pi / 2
UPLEFT = 3 * math.pi / 4
FAR_NOTE = Note(hit_time=200.0, col=2, row=1, x=0.3, y=1.4, color=1,
                angle=4.712, speed=4.0)
COAST = [0] * N_JOINTS


def _sim(tuples):
    sim = beat_arena_rs.RawBeatSim(max_steps=6900, wide_acc=False,
                                   gears=5, foot_contact=False)
    sim.set_chart(tuples)
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
    assert OBS_DIM == 608 and PREV_FEAT == 5
    assert OFF_PREV_L == 598 and OFF_PREV_R == 603
    from policy import (OBS_DIM as P_OBS, SABER_FEAT, _SABER_L_IDX,
                        _SABER_R_IDX)
    assert P_OBS == OBS_DIM and SABER_FEAT == 37
    assert _SABER_L_IDX[-PREV_FEAT:] == tuple(range(598, 603))
    assert _SABER_R_IDX[-PREV_FEAT:] == tuple(range(603, 608))


def test_reset_all_zero():
    sim = _sim([note_tuple(FAR_NOTE)])
    obs = sim.obs()
    assert len(obs) == OBS_DIM
    l, r = _blocks(obs)
    assert not l.any() and not r.any()
    assert sim.prev_note() == [(0, False), (0, False)]


# --- misses ------------------------------------------------------------------

def test_miss_fills_the_colour_block():
    n = Note(hit_time=1.0, col=2, row=1, x=0.6, y=1.4, color=1,
             angle=DOWN, speed=4.0)
    sim = _sim([note_tuple(n), note_tuple(FAR_NOTE)])
    obs, hits = _step_until_event(sim, 4)
    assert hits[0][0] == 0
    l, r = _blocks(obs)
    assert not l.any(), "a right-colour miss must not touch the L block"
    np.testing.assert_allclose(r, _expect_block(sim, n), atol=1e-6)
    assert sim.prev_note()[1] == (0, True)


def test_record_persists_through_a_gap_tracking_the_live_head():
    """No time lane: after a long coast the block is identical except
    that x/y follow the LIVE head."""
    n = Note(hit_time=1.0, col=2, row=1, x=0.6, y=1.4, color=0,
             angle=DOWN, speed=4.0)
    sim = _sim([note_tuple(n), note_tuple(FAR_NOTE)])
    obs0, _ = _step_until_event(sim, 4)
    l0, _ = _blocks(obs0)
    for _ in range(200):
        obs, *_ = sim.step(COAST)
    l, r = _blocks(obs)
    assert not r.any()
    np.testing.assert_allclose(l, _expect_block(sim, n), atol=1e-6)
    np.testing.assert_array_equal(l[2:], l0[2:])
    assert sim.prev_note()[0] == (0, True)


def test_dot_note_hides_direction():
    n = Note(hit_time=1.0, col=1, row=1, x=-0.3, y=1.4, color=0,
             angle=UPLEFT, speed=4.0, any_dir=True)
    sim = _sim([note_tuple(n), note_tuple(FAR_NOTE)])
    obs, _ = _step_until_event(sim, 4)
    l, r = _blocks(obs)
    assert not r.any()
    assert (l[2], l[3]) == (0.0, 0.0) and l[4] == 1.0


def test_bomb_pass_never_updates():
    b = Note(hit_time=1.0, col=2, row=1, x=0.6, y=1.4, color=0,
             angle=0.0, speed=4.0, kind=KIND_BOMB)
    sim = _sim([note_tuple(b), note_tuple(FAR_NOTE)])
    for _ in range(120):
        obs, *_ = sim.step(COAST)
    assert sim.status()[0] == 4, "bomb passed clean"
    l, r = _blocks(obs)
    assert not l.any() and not r.any()
    assert sim.prev_note() == [(0, False), (0, False)]


def test_latest_chart_index_wins_within_a_tick():
    a = Note(hit_time=1.0, col=0, row=1, x=-0.9, y=1.4, color=1,
             angle=DOWN, speed=4.0)
    b = Note(hit_time=1.0, col=3, row=1, x=0.9, y=1.4, color=1,
             angle=DOWN, speed=4.0)
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
    n = Note(hit_time=1.0, col=3, row=1, x=0.9, y=1.4, color=0,
             angle=UPLEFT, speed=4.0)
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


# --- cuts (scripted bot) -----------------------------------------------------

class _Snap:
    """State captured on the tick an event fired (the env moves on)."""
    def __init__(self, obs, e, env):
        self.obs, self.e, self.chart = obs, e, env.chart
        self.prev = env.sim.prev_note()
        self.head = env.sim.head_pose()

    def expect(self, note):
        x, y = note_center(note)
        show_dir = not getattr(note, "any_dir", False) or note.kind == 3
        return np.array([(x - self.head[0]) / X_NORM,
                         (y - self.head[1]) / Y_NORM,
                         math.cos(note.angle) if show_dir else 0.0,
                         math.sin(note.angle) if show_dir else 0.0, 1.0])


def _bot_run(seed, wanted, limit=3000, need=1):
    """Run the scripted bot until `need` events with a reason in `wanted`
    have fired; return one _Snap per matching event."""
    from beat_arena_env import RustBeatEnv
    from beat_arena_env.scripted import scripted_action
    env = RustBeatEnv(seed=seed, gears=5)
    env.reset(seed=seed)
    out = []
    for _ in range(limit):
        obs, _r, done, _t, info = env.step(scripted_action(env))
        for e in info["events"]:
            if e[1] in wanted:
                out.append(_Snap(obs, e, env))
        if len(out) >= need or done:
            break
    return out


def test_good_cut_records_on_the_note_colour():
    hits = _bot_run(0, ("ok",))
    assert hits, "bot produced no good cut"
    s = hits[-1]
    note = s.chart[s.e[0]]
    hand = note.color
    idx, valid = s.prev[hand]
    assert valid and idx >= s.e[0] and s.chart[idx].color == hand
    np.testing.assert_allclose(_blocks(s.obs)[hand], s.expect(s.chart[idx]),
                               atol=1e-6)


def test_bad_and_wrong_hand_cuts_record_on_the_note_colour():
    """Every bad cut the bot produces — slow, angle AND wrong-hand —
    lands on the block of the NOTE'S colour, never on the other saber's."""
    seen = {}
    for seed in (3, 5, 7, 11):
        for s in _bot_run(seed, ("slow", "angle", "wrong_hand"), need=6):
            note = s.chart[s.e[0]]
            hand = note.color
            idx, valid = s.prev[hand]
            # later chart indices resolved on the same tick win, so the
            # record is this note or a later note of the SAME colour
            assert valid and idx >= s.e[0], (seed, s.e, idx)
            assert s.chart[idx].color == hand, (seed, s.e, idx)
            np.testing.assert_allclose(_blocks(s.obs)[hand],
                                       s.expect(s.chart[idx]), atol=1e-6)
            seen[s.e[1]] = seen.get(s.e[1], 0) + 1
    assert seen, "bot produced no bad cut"


def test_wrong_hand_cut_records_on_the_note_colour_not_the_cutter():
    """A note swept by the OTHER saber's bad box records on the block of
    the note's colour; the cutting saber's block never takes it. Hunts
    the scripted bot for wrong_hand events over a few seeds."""
    found = 0
    for seed in range(0, 40):
        for s in _bot_run(seed, ("wrong_hand",), limit=1500, need=1):
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
        if found:
            break
    if not found:
        pytest.skip("scripted bot produced no wrong_hand cut in 40 seeds")


# --- policy ------------------------------------------------------------------

def test_policy_arch_marker_forward_and_refusal():
    from policy import NotePolicy, load_policy
    pol = NotePolicy()
    assert pol.arch["obs_dim"] == 608
    sim = _sim([note_tuple(Note(hit_time=1.0, col=2, row=1, x=0.6, y=1.4,
                                color=1, angle=DOWN, speed=4.0)),
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
                          gears=5, wide_acc=False, device="cuda",
                          dtype=torch.float64, auto_reset=False,
                          want_contacts=True)


@cuda
def test_cuda_miss_block_matches_rust():
    n = Note(hit_time=1.0, col=2, row=1, x=0.6, y=1.4, color=1,
             angle=DOWN, speed=4.0)
    chart = [n, FAR_NOTE]
    env = _cuda_env([chart])
    env.reset()
    sim = _sim([note_tuple(x) for x in chart])
    zero = torch.zeros((1, N_JOINTS), dtype=torch.int64, device="cuda")
    from tests.test_v10_parity_cuda import _rust_state
    for _ in range(130):
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
        q, v = _rust_state(sim)
        wv = env.wrist_vel.clone()
        env.set_state(torch.tensor([q], dtype=torch.float64),
                      torch.tensor([v], dtype=torch.float64))
        env.wrist_vel.copy_(wv)
    assert int(env.prev_valid[0, 1]) == 1 and int(env.prev_idx[0, 1]) == 0
    assert int(env.prev_valid[0, 0]) == 0
    assert sim.prev_note()[1] == (0, True)


@cuda
def test_cuda_lockstep_prev_record_parity():
    """Lockstep-resync full episodes (test_v10_parity_cuda pattern): the
    per-colour record (index, valid) is EXACT wherever per-note status
    is exact; the prev lanes differ on <= 8%% of env-ticks (the base
    gate's knife-edge band)."""
    from tests.test_v10_parity_cuda import (FULL, _charts, _rust_sims,
                                            _rust_state)
    charts = _charts(FULL, range(300, 306))
    max_steps = max(int((ch[-1].hit_time + 0.5) * 30) + 90 for ch in charts)
    env = _cuda_env(charts, max_steps=max_steps)
    env.reset()
    sims = _rust_sims(charts, 5, max_steps)
    rng = np.random.default_rng(44)
    done_r = [False] * len(charts)
    flips = ticks = 0
    for _t in range(max_steps):
        acts = rng.integers(0, 5, size=(len(charts), N_JOINTS))
        tobs, _trew, tdone, _ = env.step(torch.as_tensor(acts))
        q36s, v35s = [], []
        for e, sim in enumerate(sims):
            if not done_r[e]:
                obs_r, _rew, d_r, _ev = sim.step([int(x) for x in acts[e]])
                done_r[e] = d_r
                d = np.abs(np.asarray(obs_r)[OFF_PREV_L:]
                           - tobs[e, OFF_PREV_L:].cpu().numpy())
                flips += int((d > 1e-3).any())
                ticks += 1
            q, v = _rust_state(sim)
            q36s.append(q)
            v35s.append(v)
        wv = env.wrist_vel.clone()
        env.set_state(torch.tensor(q36s, dtype=torch.float64),
                      torch.tensor(v35s, dtype=torch.float64))
        env.wrist_vel.copy_(wv)
        if all(done_r):
            break
    assert all(done_r)
    assert flips <= 0.08 * ticks, f"{flips}/{ticks} prev-lane flip ticks"
    for e, sim in enumerate(sims):
        for h in range(2):
            idx, valid = sim.prev_note()[h]
            assert (int(env.prev_idx[e, h]), bool(env.prev_valid[e, h])) \
                == (idx, valid), f"env{e} hand{h}"
