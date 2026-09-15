"""env v44: per-hand PREVIOUS-NOTE obs blocks (598..625, OBS_DIM 626).

Contract (both engines): per hand, the most recently RESOLVED
scoreable note attributed to that hand — a cut (good or bad) to the
hand that cut it, a miss to the hand of the note's colour, bombs and
walls never. Lanes: x_rel, y_rel (head-relative, live head), is_left,
is_right, dir_cos, dir_sin (zero on dot notes), is_chain_head,
is_chain_link, free_pre, free_post, t_since = min(ticks, 90)/90 (0 on
the resolving step's obs), cut_good, missed, valid. All-zero until the
hand's first resolution. Observation-only: rewards/dynamics unchanged.
"""

import math

import numpy as np
import pytest
import torch

import beat_arena_rs
from beat_arena_env.chart import Note, note_tuple
from beat_arena_env.constants import (KIND_BOMB, N_JOINTS, OBS_DIM,
                                      OFF_PREV_L, OFF_PREV_R, PREV_FEAT,
                                      PREV_T_CLIP_TICKS, X_NORM, Y_NORM,
                                      note_center)
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


# --- layout ------------------------------------------------------------------

def test_layout_contract():
    assert OBS_DIM == 626 and PREV_FEAT == 14
    assert OFF_PREV_L == 598 and OFF_PREV_R == 612
    assert PREV_T_CLIP_TICKS == 90
    from policy import (OBS_DIM as P_OBS, SABER_FEAT, _SABER_L_IDX,
                        _SABER_R_IDX)
    assert P_OBS == OBS_DIM and SABER_FEAT == 46
    assert _SABER_L_IDX[-PREV_FEAT:] == tuple(range(598, 612))
    assert _SABER_R_IDX[-PREV_FEAT:] == tuple(range(612, 626))


def test_reset_all_zero():
    sim = _sim([note_tuple(FAR_NOTE)])
    obs = sim.obs()
    assert len(obs) == OBS_DIM
    l, r = _blocks(obs)
    assert not l.any() and not r.any()
    assert sim.prev_note() == [(0, 0, 0), (0, 0, 0)]


# --- misses ------------------------------------------------------------------

def test_miss_fills_the_colour_hand():
    n = Note(hit_time=1.0, col=2, row=1, x=0.6, y=1.4, color=1,
             angle=DOWN, speed=4.0)
    sim = _sim([note_tuple(n), note_tuple(FAR_NOTE)])
    obs, hits = _step_until_event(sim, 4)
    assert hits[0][0] == 0
    l, r = _blocks(obs)
    assert not l.any(), "a right-colour miss must not touch the L block"
    hx, hy = sim.head_pose()[0], sim.head_pose()[1]
    assert r[0] == pytest.approx((0.6 - hx) / X_NORM, abs=1e-6)
    assert r[1] == pytest.approx((1.4 - hy) / Y_NORM, abs=1e-6)
    assert (r[2], r[3]) == (0.0, 1.0)
    assert r[4] == pytest.approx(math.cos(DOWN), abs=1e-6)
    assert r[5] == pytest.approx(math.sin(DOWN), abs=1e-6)
    assert list(r[6:10]) == [0.0, 0.0, 0.0, 0.0]
    assert r[10] == 0.0, "t_since reads 0 on the resolving step's obs"
    assert (r[11], r[12], r[13]) == (0.0, 1.0, 1.0)
    idx, tick, out = sim.prev_note()[1]
    assert (idx, out) == (0, 3) and tick == sim.tick


def test_t_since_counts_ticks_and_clips():
    n = Note(hit_time=1.0, col=2, row=1, x=0.6, y=1.4, color=0,
             angle=DOWN, speed=4.0)
    sim = _sim([note_tuple(n), note_tuple(FAR_NOTE)])
    _step_until_event(sim, 4)
    t0 = sim.tick
    for k in range(1, 100):
        obs, *_ = sim.step(COAST)
        l, _r = _blocks(obs)
        expect = np.float32(min(k, PREV_T_CLIP_TICKS)) / np.float32(
            PREV_T_CLIP_TICKS)
        assert l[10] == expect, (k, l[10], expect)
        assert sim.tick - t0 == k
        assert l[13] == 1.0 and l[12] == 1.0
    assert l[10] == 1.0


def test_dot_note_hides_direction():
    n = Note(hit_time=1.0, col=1, row=1, x=-0.3, y=1.4, color=0,
             angle=UPLEFT, speed=4.0, any_dir=True)
    sim = _sim([note_tuple(n), note_tuple(FAR_NOTE)])
    obs, _ = _step_until_event(sim, 4)
    l, r = _blocks(obs)
    assert not r.any()
    assert (l[2], l[3]) == (1.0, 0.0)
    assert (l[4], l[5]) == (0.0, 0.0)
    assert (l[12], l[13]) == (1.0, 1.0)


def test_bomb_pass_never_updates():
    b = Note(hit_time=1.0, col=2, row=1, x=0.6, y=1.4, color=0,
             angle=0.0, speed=4.0, kind=KIND_BOMB)
    sim = _sim([note_tuple(b), note_tuple(FAR_NOTE)])
    for _ in range(120):
        obs, *_ = sim.step(COAST)
    assert sim.status()[0] == 4, "bomb passed clean"
    l, r = _blocks(obs)
    assert not l.any() and not r.any()
    assert sim.prev_note() == [(0, 0, 0), (0, 0, 0)]


def test_latest_chart_index_wins_within_a_tick():
    a = Note(hit_time=1.0, col=0, row=1, x=-0.9, y=1.4, color=1,
             angle=DOWN, speed=4.0)
    b = Note(hit_time=1.0, col=3, row=1, x=0.9, y=1.4, color=1,
             angle=DOWN, speed=4.0)
    sim = _sim([note_tuple(a), note_tuple(b), note_tuple(FAR_NOTE)])
    obs, hits = _step_until_event(sim, 4)
    assert sorted(h[0] for h in hits) == [0, 1], "both miss on one tick"
    _l, r = _blocks(obs)
    assert sim.prev_note()[1][0] == 1
    hx = sim.head_pose()[0]
    assert r[0] == pytest.approx((0.9 - hx) / X_NORM, abs=1e-6)


# --- mirror ------------------------------------------------------------------

def test_mirror_augment_reflects_the_block():
    """The trainer's --mirror-augment mirrors CHARTS (mirror.py), not
    obs; the block must then come out mirrored by construction: L <-> R,
    x and dir_cos negated, colour one-hot swapped, all else equal."""
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
    # x reflects (each block against its OWN live head, which the
    # coasting body leaves a few mm off centre)
    hxa, hxb = sim_a.head_pose()[0], sim_b.head_pose()[0]
    assert la[0] == pytest.approx((0.9 - hxa) / X_NORM, abs=1e-6)
    assert rb[0] == pytest.approx((-0.9 - hxb) / X_NORM, abs=1e-6)
    assert la[0] + rb[0] == pytest.approx(-(hxa + hxb), abs=1e-5)
    assert la[1] == pytest.approx(rb[1], abs=1e-6)
    assert (la[2], la[3]) == (1.0, 0.0) and (rb[2], rb[3]) == (0.0, 1.0)
    assert la[4] == pytest.approx(-rb[4], abs=1e-6)
    assert la[5] == pytest.approx(rb[5], abs=1e-6)
    np.testing.assert_array_equal(la[6:], rb[6:])


# --- cuts (scripted bot) -----------------------------------------------------

def test_good_cut_attributes_to_the_cutting_hand():
    from beat_arena_env import RustBeatEnv
    from beat_arena_env.scripted import scripted_action
    env = RustBeatEnv(seed=0, gears=5)
    env.reset(seed=0)
    for _ in range(3000):
        obs, _r, done, _t, info = env.step(scripted_action(env))
        good = [e for e in info["events"] if e[1] == "ok"]
        if good:
            break
        assert not done, "episode ended before any good cut"
    else:
        raise AssertionError("bot produced no good cut")
    idx = good[-1][0]
    note = env.chart[idx]
    hand = note.color
    rec = env.sim.prev_note()[hand]
    assert rec[0] == idx and rec[2] == 1 and rec[1] == env.sim.tick
    blk = _blocks(obs)[hand]
    x, y = note_center(note)
    hx, hy = env.sim.head_pose()[0], env.sim.head_pose()[1]
    assert blk[0] == pytest.approx((x - hx) / X_NORM, abs=1e-6)
    assert blk[1] == pytest.approx((y - hy) / Y_NORM, abs=1e-6)
    assert blk[2 + hand] == 1.0 and blk[3 - hand] == 0.0
    assert (blk[10], blk[11], blk[12], blk[13]) == (0.0, 1.0, 0.0, 1.0)
    assert blk[6] == float(note.kind == 2) and blk[7] == float(note.kind == 3)


def test_bad_cuts_read_neither_good_nor_missed():
    """Any bad cut the bot produces (slow / angle / wrong-hand) leaves
    cut_good = missed = 0 with valid = 1 on the attributed hand."""
    from beat_arena_env import RustBeatEnv
    from beat_arena_env.scripted import scripted_action
    env = RustBeatEnv(seed=3, gears=5)
    env.reset(seed=3)
    seen = 0
    for _ in range(3000):
        obs, _r, done, _t, info = env.step(scripted_action(env))
        bad = [e for e in info["events"]
               if e[1] in ("slow", "angle", "wrong_hand")]
        if bad:
            e = bad[-1]
            note = env.chart[e[0]]
            hand = (1 - note.color) if e[1] == "wrong_hand" else note.color
            rec = env.sim.prev_note()[hand]
            assert rec[0] == e[0] and rec[2] == 2
            blk = _blocks(obs)[hand]
            assert (blk[11], blk[12], blk[13]) == (0.0, 0.0, 1.0)
            seen += 1
            if seen >= 3:
                break
        if done:
            break
    assert seen >= 1, "bot produced no bad cut"


# --- policy ------------------------------------------------------------------

def test_policy_arch_marker_forward_and_refusal():
    from policy import NotePolicy, load_policy
    pol = NotePolicy()
    assert pol.arch["obs_dim"] == 626
    sim = _sim([note_tuple(Note(hit_time=1.0, col=2, row=1, x=0.6, y=1.4,
                                color=1, angle=DOWN, speed=4.0)),
                note_tuple(FAR_NOTE)])
    obs, _ = _step_until_event(sim, 4)
    x = torch.as_tensor(np.asarray(obs)).unsqueeze(0)
    with torch.no_grad():
        logits, value = pol.forward_stacked(x)
    assert logits.shape[:2] == (1, 29) and value.shape == (1,)
    assert torch.isfinite(logits).all() and torch.isfinite(value).all()
    old = {"arch": dict(pol.arch, obs_dim=598), "state_dict": pol.state_dict()}
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
        # Discrete + tick lanes bitwise. The head-relative x/y lanes are
        # the same expression on both engines (note world x/y minus the
        # LIVE head, entity normalizers), but the two physics engines
        # drift apart under free-running coast (~1e-2 by tick 130 on the
        # pod gate, 2026-09-15), so — exactly like the base parity gate
        # — the CUDA state is resynced from Rust after every tick and
        # the lanes are compared over one tick of divergence only.
        c = tobs[0, OFF_PREV_L:OBS_DIM].cpu().numpy()
        r = np.asarray(robs)[OFF_PREV_L:OBS_DIM]
        np.testing.assert_array_equal(c[2:14], r[2:14])
        np.testing.assert_array_equal(c[16:28], r[16:28])
        np.testing.assert_allclose(c[[0, 1, 14, 15]], r[[0, 1, 14, 15]],
                                   atol=1e-4)
        q, v = _rust_state(sim)
        wv = env.wrist_vel.clone()
        env.set_state(torch.tensor([q], dtype=torch.float64),
                      torch.tensor([v], dtype=torch.float64))
        env.wrist_vel.copy_(wv)
    assert int(env.prev_out[0, 1]) == 3 and int(env.prev_idx[0, 1]) == 0
    assert int(env.prev_tick[0, 1]) == sim.prev_note()[1][1]
    assert sim.prev_note()[1][2] == 3


@cuda
def test_cuda_lockstep_prev_record_parity():
    """Lockstep-resync full episodes (test_v10_parity_cuda pattern): the
    per-hand record (index, outcome) is EXACT wherever per-note status
    is exact; the resolution tick agrees to within the same knife-edge
    flips the base gate tolerates; the prev lanes differ on <= 8%% of
    env-ticks (the base gate's band)."""
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
            idx, tick, out = sim.prev_note()[h]
            assert (int(env.prev_idx[e, h]), int(env.prev_out[e, h])) \
                == (idx, out), f"env{e} hand{h}"
            assert abs(int(env.prev_tick[e, h]) - tick) <= 1, f"env{e} hand{h}"
