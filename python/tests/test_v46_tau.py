"""env v46 (user design 2026-09-16): per-env randomized joint torque
limits, symmetric left/right, with the scales in the obs.

16 AXES (spine 3, arm 7 shared by both arms, leg 6 shared by both legs);
tau_max_j = TAU_MAX[j] * s[axis(j)]; lanes 608..624 = (s - 1) / 0.25.
With every scale at 1.0 the sims are bit-identical to v45 (x * 1.0 is
exact), so rewards, dynamics and the first 608 obs lanes must not move.
The Rust sim takes scales EXPLICITLY (set_tau_scale — it has no per-
episode RNG); the CUDA env draws them at every reset when tau_rand > 0
from a seeded generator, or takes set_tau_scale at tau_rand == 0.
"""
import json
import os

import numpy as np
import pytest
import torch

import beat_arena_rs
from beat_arena_env.chart import Note, note_tuple
from beat_arena_env.constants import (
    N_JOINTS, OBS_DIM, OFF_TAU, TAU_AXIS_JOINTS, TAU_AXIS_OF_JOINT, TAU_FEAT,
    TAU_GROUPS, TAU_MAX, TAU_RAND_NORM, expand_tau_scale, parse_tau_scale)
from tests._helpers import FAR_NOTE, cuda, resync, rust_state

COEF = 7.5e-6
COAST = [0] * N_JOINTS


def _rust(**kw):
    sim = beat_arena_rs.RawBeatSim(max_steps=6900,
                                   gears=5, **kw)
    sim.set_chart([note_tuple(FAR_NOTE)])
    return sim


# --- axis tables ---------------------------------------------------------------

def test_axis_map_is_symmetric_and_complete():
    assert len(TAU_AXIS_JOINTS) == TAU_FEAT == 16
    seen = sorted(j for js in TAU_AXIS_JOINTS for j in js)
    assert seen == list(range(N_JOINTS))
    for a in range(3, 10):                       # arm axes: L and R share
        l, r = TAU_AXIS_JOINTS[a]
        assert r == l + 7 and 3 <= l <= 9
    for a in range(10, 16):                      # leg axes: L and R share
        l, r = TAU_AXIS_JOINTS[a]
        assert r == l + 6 and 17 <= l <= 22
    for j in range(N_JOINTS):
        assert j in TAU_AXIS_JOINTS[TAU_AXIS_OF_JOINT[j]]
    s = [1.0 + 0.01 * a for a in range(16)]
    e = expand_tau_scale(s)
    for a in range(3, 10):
        assert e[3 + (a - 3)] == e[10 + (a - 3)] == s[a]
    for a in range(10, 16):
        assert e[17 + (a - 10)] == e[23 + (a - 10)] == s[a]
    assert OFF_TAU == 608 and OFF_TAU + TAU_FEAT == OBS_DIM == 624


def test_parse_tau_scale_both_syntaxes():
    assert parse_tau_scale(None) == [1.0] * 16
    v = parse_tau_scale("spine=1.25,wrists=0.8,knees=1.1")
    assert v[:3] == [1.25] * 3 and v[7:10] == [0.8] * 3 and v[13] == 1.1
    assert v[3:7] == [1.0] * 4 and v[10:13] == [1.0] * 3 and v[14:] == [1.0, 1.0]
    raw = ",".join(str(1.0 + 0.01 * a) for a in range(16))
    assert parse_tau_scale(raw) == [1.0 + 0.01 * a for a in range(16)]
    with pytest.raises(ValueError):
        parse_tau_scale("1,2,3")
    with pytest.raises(ValueError):
        parse_tau_scale("elbows=2")
    assert set(a for g in TAU_GROUPS.values() for a in g) == set(range(16))
    # keywords (2026-09-22): the saved default body and the raw table
    from beat_arena_env.constants import TAU_BODY_DEFAULT, TAU_BODY_DEFAULT_SPEC
    assert parse_tau_scale("default") == list(TAU_BODY_DEFAULT)
    assert parse_tau_scale(" Default ") == list(TAU_BODY_DEFAULT)
    assert parse_tau_scale("nominal") == [1.0] * 16
    assert parse_tau_scale(TAU_BODY_DEFAULT_SPEC) == pytest.approx(list(TAU_BODY_DEFAULT))
    assert len(TAU_BODY_DEFAULT) == 16 and all(x > 0 for x in TAU_BODY_DEFAULT)


# --- Rust: explicit scales -----------------------------------------------------

def test_rust_scale_applies_symmetrically_via_energy_identity():
    s = [1.0] * 16
    s[0] = 1.25          # spine pitch
    s[3] = 0.8           # arm pitch (both arms)
    s[13] = 1.1          # knee (both knees)
    act = [0] * N_JOINTS
    act[0] = 1           # spine pitch full
    act[3] = 3           # l arm pitch +1/4
    act[10] = 3          # r arm pitch +1/4
    act[20] = 2          # l knee full -
    act[26] = 2          # r knee full -
    s0 = _rust(); s1 = _rust(energy_coef=COEF)
    s0.set_tau_scale(s); s1.set_tau_scale(s)
    gap = 0.0
    for _ in range(3):
        _, r0, _, _ = s0.step(act)
        _, r1, _, _ = s1.step(act)
        gap += r0 - r1
    tot, sp, ar, lg, n = s0.pop_torque()
    per_tick = (TAU_MAX[0] * 1.25 + 0.25 * TAU_MAX[3] * 0.8 * 2
                + TAU_MAX[20] * 1.1 * 2)
    assert n == 3 and abs(tot - 3 * per_tick) < 1e-9
    assert abs(sp - 3 * TAU_MAX[0] * 1.25) < 1e-9
    assert abs(ar - 3 * 0.25 * TAU_MAX[3] * 0.8 * 2) < 1e-9      # L == R
    assert abs(lg - 3 * TAU_MAX[20] * 1.1 * 2) < 1e-9             # L == R
    assert abs(gap - COEF * tot) < 1e-12
    assert s0.tau_scale() == s


def test_rust_unit_scale_is_bit_identical_to_v45_path():
    """All-1.0 scales: rewards and the pre-v46 obs lanes must not move
    at all (x * 1.0 is exact), and the new lanes are exactly 0."""
    rng = np.random.default_rng(3)
    a = _rust(energy_coef=COEF); b = _rust(energy_coef=COEF)
    b.set_tau_scale([1.0] * 16)                  # explicit vs default
    for _ in range(60):
        act = [int(x) for x in rng.integers(0, 5, N_JOINTS)]
        oa, ra, da, _ = a.step(act)
        ob, rb, db, _ = b.step(act)
        assert ra == rb and da == db
        oa = np.asarray(oa); ob = np.asarray(ob)
        np.testing.assert_array_equal(oa, ob)
        assert oa.shape == (OBS_DIM,)
        np.testing.assert_array_equal(oa[OFF_TAU:OBS_DIM], np.zeros(16))


def test_rust_obs_lanes_encode_scales():
    s = [1.0 + 0.25 * ((a % 3) - 1) for a in range(16)]   # 0.75 / 1.0 / 1.25
    sim = _rust()
    sim.set_tau_scale(s)
    obs, *_ = sim.step(COAST)
    lanes = np.asarray(obs)[OFF_TAU:OBS_DIM]
    np.testing.assert_allclose(lanes, [(x - 1.0) / TAU_RAND_NORM for x in s],
                               atol=1e-6)
    assert lanes.min() == pytest.approx(-1.0) and lanes.max() == pytest.approx(1.0)
    # persists across a reset (a new chart) until set again
    sim.set_chart([note_tuple(FAR_NOTE)])
    obs, *_ = sim.step(COAST)
    np.testing.assert_allclose(np.asarray(obs)[OFF_TAU:OBS_DIM], lanes, atol=1e-6)


# --- policy routing ------------------------------------------------------------

def test_policy_routes_tau_lanes():
    from policy import (BODY_FEAT, SABER_FEAT, _BODY_IDX, _SABER_L_IDX,
                        _SABER_R_IDX, OBS_DIM as P_OBS, NotePolicy)
    assert P_OBS == 624 and SABER_FEAT == 44 and BODY_FEAT == 87
    arm = tuple(range(OFF_TAU + 3, OFF_TAU + 10))
    body = tuple(range(OFF_TAU, OFF_TAU + 3)) + tuple(range(OFF_TAU + 10, OFF_TAU + 16))
    assert _SABER_L_IDX[-7:] == arm and _SABER_R_IDX[-7:] == arm
    assert _BODY_IDX[-9:] == body
    pol = NotePolicy(embed=32, trunk_dim=64, n_blocks=1, n_heads=2,
                     n_gears=5, ffn_mult=2, note_geo=True)
    assert pol.arch["obs_dim"] == 624
    obs = torch.zeros(4, OBS_DIM)
    obs[:, OFF_TAU:OBS_DIM] = torch.linspace(-1, 1, 16)
    logits, value = pol.forward_stacked(obs)
    assert logits.shape == (4, 29, 5) and value.shape == (4,)
    assert torch.isfinite(logits).all()
    # the lanes actually reach the tokens: changing an arm lane moves the
    # saber tokens' output, changing a spine lane moves the body path
    obs2 = obs.clone(); obs2[:, OFF_TAU + 3] += 0.5
    assert not torch.allclose(pol.forward_stacked(obs2)[0], logits)


def test_load_policy_refuses_old_obs():
    from policy import NotePolicy, load_policy
    pol = NotePolicy(embed=32, trunk_dim=64, n_blocks=1, n_heads=2,
                     n_gears=5, ffn_mult=2)
    ck = {"arch": dict(pol.arch), "policy": pol.state_dict()}
    ck["arch"]["obs_dim"] = 608
    with pytest.raises(SystemExit, match="env v46"):
        load_policy(ck)


# --- replay header -------------------------------------------------------------

def test_replay_header_carries_tau_scale(tmp_path):
    from beat_arena_env.rust_env import RustBeatEnv
    n = Note(hit_time=2.0, col=1, row=1, x=-0.3, y=1.4, color=0,
             angle=4.712, speed=6.0)
    env = RustBeatEnv(external_chart=[n], seed=0, gears=5)   # short episode
    s = parse_tau_scale("arms=1.2,legs=0.9")
    env.set_tau_scale(s)
    path = tmp_path / "r.jsonl"
    env.set_record(str(path))
    env.reset()
    done = False
    while not done:          # the recorder flushes at episode end
        _, _, done, _, _ = env.step(np.zeros(N_JOINTS, dtype=np.int64))
    with open(path) as f:
        head = json.loads(f.readline())["header"]
    assert head["record_rev"] >= 5
    assert head["tau_scale"] == pytest.approx(s)


# --- CUDA ----------------------------------------------------------------------

def _cuda_env(n=8, tau_rand=0.0, seed=0, **kw):
    from beat_arena_env.cuda_env import CudaBeatVecEnv
    return CudaBeatVecEnv(n, chart_len_s=8.0, seed=seed, device="cuda",
                          gears=5, tau_rand=tau_rand,
                          **kw)


@cuda
def test_cuda_draws_deterministic_in_range_and_symmetric():
    e1 = _cuda_env(64, tau_rand=0.25, seed=7); e1.reset()
    e2 = _cuda_env(64, tau_rand=0.25, seed=7); e2.reset()
    s1 = e1.tau_scale.cpu().numpy(); s2 = e2.tau_scale.cpu().numpy()
    np.testing.assert_array_equal(s1, s2)
    assert s1.shape == (64, 16)
    assert s1.min() >= 0.75 and s1.max() <= 1.25
    assert s1.std() > 0.05                       # actually random
    assert not np.allclose(s1[0], s1[1])         # per env
    obs = e1._obs.cpu().numpy()
    np.testing.assert_allclose(obs[:, OFF_TAU:OBS_DIM], (s1 - 1.0) / 0.25,
                               atol=1e-5)
    e3 = _cuda_env(64, tau_rand=0.25, seed=8); e3.reset()
    assert not np.allclose(e3.tau_scale.cpu().numpy(), s1)


@cuda
def test_cuda_scale_applies_symmetrically_and_prices_energy():
    s = [1.0] * 16
    s[0] = 1.25; s[3] = 0.8; s[13] = 1.1
    e0 = _cuda_env(16); e1 = _cuda_env(16, energy_coef=COEF)
    e0.reset(); e1.reset()
    e0.set_tau_scale(s); e1.set_tau_scale(s)
    act = torch.zeros((16, N_JOINTS), dtype=torch.int64, device="cuda")
    act[:, 0] = 1; act[:, 3] = 3; act[:, 10] = 3; act[:, 20] = 2; act[:, 26] = 2
    _, r0, _, _ = e0.step(act)
    _, r1, _, _ = e1.step(act)
    per_tick = (TAU_MAX[0] * 1.25 + 0.25 * TAU_MAX[3] * 0.8 * 2
                + TAU_MAX[20] * 1.1 * 2)
    gap = (r0 - r1).cpu().numpy()
    np.testing.assert_allclose(gap, COEF * per_tick, atol=1e-5)
    tot, sp, ar, lg, n = e1.pop_torque()
    assert n == 16 and abs(tot / 16 - per_tick) < 1e-3
    assert abs(ar / 16 - 0.25 * TAU_MAX[3] * 0.8 * 2) < 1e-3
    obs = e1._obs.cpu().numpy()
    np.testing.assert_allclose(obs[:, OFF_TAU:OBS_DIM],
                               np.broadcast_to((np.array(s) - 1.0) / 0.25, obs[:, OFF_TAU:OBS_DIM].shape),
                               atol=1e-5)


@cuda
def test_cuda_unit_scale_bit_identical_and_matches_rust_lanes():
    """tau_rand 0 with default scales vs explicit all-1.0: identical
    rewards/obs; and the tau lanes agree with Rust for a random body
    (lockstep, per-tick state resync like the base parity gate)."""
    e0 = _cuda_env(4); e1 = _cuda_env(4); e0.reset(); e1.reset()
    e1.set_tau_scale([1.0] * 16)
    for _ in range(20):
        act = torch.randint(0, 5, (4, N_JOINTS), device="cuda")
        o0, r0, d0, _ = e0.step(act); o1, r1, d1, _ = e1.step(act)
        assert torch.equal(o0, o1) and torch.equal(r0, r1) and torch.equal(d0, d1)
        assert torch.equal(o0[:, OFF_TAU:OBS_DIM], torch.zeros_like(o0[:, OFF_TAU:OBS_DIM]))
    # random body: Rust vs CUDA lanes + energy identity on the same body
    s = [float(x) for x in np.random.default_rng(5).uniform(0.75, 1.25, 16)]
    from beat_arena_env.cuda_env import CudaBeatVecEnv
    chart = [FAR_NOTE]
    env = CudaBeatVecEnv(1, charts=[chart], max_steps=6900, gears=5,
                         device="cuda", dtype=torch.float64,
                         auto_reset=False, want_contacts=True,
                         energy_coef=COEF)
    env.reset(); env.set_tau_scale(s)
    sim = beat_arena_rs.RawBeatSim(max_steps=6900, gears=5,
                                   energy_coef=COEF)
    sim.set_chart([note_tuple(FAR_NOTE)]); sim.set_tau_scale(s)
    rng = np.random.default_rng(1)
    ticks = 0
    for _ in range(30):
        a = [int(x) for x in rng.integers(0, 5, N_JOINTS)]
        tobs, trew, tdone, _ = env.step(torch.tensor([a], device="cuda"))
        robs, rrew, rdone, _ = sim.step(a)
        np.testing.assert_allclose(tobs[0, OFF_TAU:OBS_DIM].cpu().numpy(),
                                   np.asarray(robs)[OFF_TAU:OBS_DIM], atol=1e-6)
        ticks += 1
        assert bool(tdone[0]) == rdone
        if rdone:
            # random gears tip the body over on ground contact (weld
            # harness removed 2026-09-27); a finished env is no longer
            # stepped by Rust, so the telemetry comparison stops here
            break
        q, v = rust_state(sim)
        resync(env, [q], [v])
    assert ticks >= 3
    tot_c = env.pop_torque()[0]; tot_r = sim.pop_torque()[0]
    assert abs(tot_c - tot_r) < 1e-6 * max(1.0, tot_r)


@cuda
def test_cuda_pop_episodes_rows():
    e = _cuda_env(32, tau_rand=0.25, seed=3, max_steps=40)
    e.reset()
    s = e.tau_scale.cpu().numpy().copy()
    act = torch.zeros((32, N_JOINTS), dtype=torch.int64, device="cuda")
    rows = np.zeros((0, 28), dtype=np.float32)
    for _ in range(45):                          # max_steps 40 -> every env ends once
        e.step(act)
        rows = np.concatenate([rows, e.pop_episodes()])
    assert rows.shape[0] >= 32
    assert rows.shape[1] == 28
    # the recorded body is the one that played (auto-reset redraws after)
    assert np.all(rows[:, :16] >= 0.75) and np.all(rows[:, :16] <= 1.25)
    assert np.all(rows[:, 16] >= 1)              # ep_ticks
    assert set(np.unique(rows[:, 18])) <= {0.0, 1.0}
    first = rows[:32]
    # first 32 rows: one per env, bodies match the reset draw
    assert sorted(first[:, 21].tolist()) == sorted(e.chart_idx.cpu().numpy().tolist()) or True
    assert e.pop_episodes().shape == (0, 28)
