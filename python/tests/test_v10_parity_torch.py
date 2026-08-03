"""P4 gate — torch mirror parity vs the Rust sim (SPEC-v10 §7 P4).

Tier 1 (float64, CPU): EXACT — bitwise — agreement. The torch engine
transcribes the Rust expression trees literally (torch_dynamics /
torch_env docstrings), routes transcendentals through the CRT libm on
the f64 CPU path, and orders every reward/score accumulation the way
step_inner does, so the comparisons below use strict equality on
everything: obs (f32 lanes), rewards (f64), done timing, per-note
status, counters, score decomposition, replay-relevant state (qpos /
twist / omega / sabers), tick-summed contact impulses, and effort.
`post` events lose their note ids in the torch tracker ring (aggregated
slots — torch_env docstring); their points are pinned bitwise through
reward + score_post instead.

Tier 2 (float32): well-formedness + statistical bounds — the v10 stiff
dynamics amplify f32 seeds through knife-edge branch flips within
ticks, so the v9-era per-lane tolerances do not transfer (see
test_parity_f32_bounded's docstring for the measured profile).

Runtime note: the torch mirror costs ~1 s per 30 Hz vector tick on CPU
(Python-dispatch bound, batch-size independent) — this module runs
minutes, not seconds; trajectories are batched wide to buy coverage per
tick.
"""

import numpy as np
import pytest
import torch

from beat_arena_env.chart import ChartConfig, Note, generate_chart, note_tuple
from beat_arena_env.constants import JOINT_LIMITS, N_JOINTS, NV, OBS_DIM
from beat_arena_env.torch_env import TorchBeatVecEnv
import beat_arena_rs

FULL = dict(colors="both", chart_len_s=6.0, bomb_prob=0.25, extra_prob=0.3,
            dot_prob=0.15)
# a note far beyond every horizon: pure body dynamics
FAR_NOTE = Note(hit_time=200.0, col=2, row=1, x=0.3, y=1.4, color=1,
                angle=4.712, speed=4.0)


def _charts(cfg_kw, seeds):
    return [generate_chart(ChartConfig(**cfg_kw), np.random.default_rng(s))
            for s in seeds]


def _rust_sims(charts, gears, max_steps, wide_acc=False, effort_coef=4e-4):
    sims = []
    for ch in charts:
        sim = beat_arena_rs.RawBeatSim(max_steps=max_steps,
                                       effort_coef=effort_coef,
                                       wide_acc=wide_acc, gears=gears)
        sim.set_chart([note_tuple(n) for n in ch])
        sims.append(sim)
    return sims


def _tenv(charts, gears, max_steps, dtype=torch.float64, wide_acc=False,
          effort_coef=4e-4):
    return TorchBeatVecEnv(len(charts), charts=charts, max_steps=max_steps,
                           gears=gears, wide_acc=wide_acc,
                           effort_coef=effort_coef, device="cpu",
                           dtype=dtype, auto_reset=False)


def _rand_states(n, seed, spread=1.0):
    rng = np.random.default_rng(seed)
    q36s, v35s = [], []
    for _ in range(n):
        q = [float(rng.uniform(lo * spread, hi * spread))
             for lo, hi in JOINT_LIMITS]
        q36 = [float(rng.uniform(-0.2, 0.2)),
               0.94 + float(rng.uniform(-0.15, 0.1)),
               -0.65 + float(rng.uniform(-0.15, 0.15))]
        quat = rng.normal(size=4)
        quat /= np.linalg.norm(quat)
        q36 += [float(x) for x in quat] + q
        v35s.append([float(x) for x in rng.uniform(-4.0, 4.0, NV)])
        q36s.append(q36)
    return q36s, v35s


def _assert_state_exact(sim, env, e, tag):
    np.testing.assert_array_equal(np.array(sim.qpos()),
                                  env.qpos()[e].numpy(), err_msg=f"{tag} qpos")
    np.testing.assert_array_equal(np.array(sim.twist()),
                                  env.twist()[e].numpy(),
                                  err_msg=f"{tag} twist")
    np.testing.assert_array_equal(np.array(sim.omega()),
                                  env.omega()[e].numpy(),
                                  err_msg=f"{tag} omega")
    np.testing.assert_array_equal(np.array(sim.sabers()),
                                  env.sabers()[e].numpy(),
                                  err_msg=f"{tag} sabers")
    assert sim.last_effort() == float(env.last_effort()[e]), f"{tag} effort"
    imp = env.contacts()[0][e].numpy()
    rust_imp = np.zeros(55)
    for p, _x, _y, _z, j in sim.contacts():
        rust_imp[p] = j
    np.testing.assert_array_equal(rust_imp, imp, err_msg=f"{tag} contacts")


def _assert_counters_exact(sim, env, e, tag):
    g, b, m, sc, bo = (sim.good_cuts, sim.bad_cuts, sim.misses, sim.score,
                       sim.bombs_hit)
    cnt = env.counters()
    assert (g, b, m, bo) == (int(cnt[0][e]), int(cnt[1][e]), int(cnt[2][e]),
                             int(cnt[5][e])), f"{tag} counters"
    assert sc == float(cnt[4][e]), f"{tag} score"
    pre, acc, post = sim.score_components()
    tpre, tacc, tpost = [float(x[e]) for x in env.components()]
    assert (pre, acc, post) == (tpre, tacc, tpost), f"{tag} components"
    np.testing.assert_array_equal(
        np.array(sim.status(), dtype=np.uint8),
        env.status[e, :len(sim.status())].numpy(), err_msg=f"{tag} status")


# --- (a) single-tick parity: 200 seeded random full states, random gears ----

@pytest.mark.parametrize("gears,n_states,seed", [(3, 66, 100), (5, 67, 200),
                                                 (7, 67, 300)])
def test_single_tick_parity_random_states(gears, n_states, seed):
    """One 30 Hz tick (8 constrained substeps) from random full states:
    state/obs/effort/reward bitwise vs Rust, all gear tables.

    These states are deliberately extreme (root orientation uniform, the
    foot welds violated by up to 0.5 m, every DoF at ±4 rad/s), so most
    of them trip the §23 root-escape backstop on the FIRST tick and the
    episode ends there. The tiers agree exactly on WHICH envs and WHEN —
    that agreement is asserted — but an env is not comparable after its
    terminal tick with `auto_reset=False`: Rust's B4 `finished` latch
    freezes a done env (it re-reports `(0.0, true)` without running
    physics) while the batched mirror, which has no latch, keeps
    integrating. So each env is compared up to and including the tick it
    reports done, and dropped after."""
    q36s, v35s = _rand_states(n_states, seed)
    charts = [[FAR_NOTE]] * n_states
    env = _tenv(charts, gears, 6900)
    env.reset()
    env.set_state(torch.tensor(q36s, dtype=torch.float64),
                  torch.tensor(v35s, dtype=torch.float64))
    sims = _rust_sims(charts, gears, 6900)
    rng = np.random.default_rng(seed + 1)
    acts = rng.integers(0, gears, size=(2, n_states, N_JOINTS))
    for e, sim in enumerate(sims):
        sim.set_joints(q36s[e], v35s[e])
    live = [True] * n_states
    compared = 0
    for t in range(2):
        tobs, trew, td, _ = env.step(torch.as_tensor(acts[t]))
        for e, sim in enumerate(sims):
            if not live[e]:
                continue
            obs_r, rew_r, d_r, _ev = sim.step([int(x) for x in acts[t, e]])
            assert d_r == bool(td[e]), f"tick{t} env{e} done"
            assert rew_r == float(trew[e]), f"tick{t} env{e} reward"
            np.testing.assert_array_equal(np.asarray(obs_r), tobs[e].numpy(),
                                          err_msg=f"tick{t} env{e} obs")
            _assert_state_exact(sim, env, e, f"tick{t} env{e}")
            compared += 1
            if d_r:
                live[e] = False
    assert compared >= n_states, "every env must be compared at least once"


# --- (b) multi-tick trajectory parity ----------------------------------------

@pytest.mark.parametrize("gears,n_envs,n_ticks,seed",
                         [(5, 32, 100, 40), (3, 12, 60, 41), (7, 12, 60, 42)])
def test_trajectory_parity(gears, n_envs, n_ticks, seed):
    """Random charts + random actions: obs, rewards, dones, per-note
    status, counters, score decomposition, and replay-relevant state all
    EXACT at every tick."""
    charts = _charts(FULL, range(seed, seed + n_envs))
    max_steps = max(int((ch[-1].hit_time + 0.5) * 30) + 90 for ch in charts)
    sims = _rust_sims(charts, gears, max_steps)
    env = _tenv(charts, gears, max_steps)
    tobs = env.reset()
    for e, sim in enumerate(sims):
        np.testing.assert_array_equal(np.asarray(sim.obs()), tobs[e].numpy(),
                                      err_msg=f"env{e} reset obs")
    rng = np.random.default_rng(seed + 1000)
    done_r = [False] * n_envs
    for t in range(n_ticks):
        acts = rng.integers(0, gears, size=(n_envs, N_JOINTS))
        tobs, trew, tdone, _ = env.step(torch.as_tensor(acts))
        for e, sim in enumerate(sims):
            if done_r[e]:
                continue
            obs_r, rew_r, d_r, _ev = sim.step([int(x) for x in acts[e]])
            assert rew_r == float(trew[e]), f"t{t} env{e} reward"
            assert d_r == bool(tdone[e]), f"t{t} env{e} done"
            np.testing.assert_array_equal(np.asarray(obs_r), tobs[e].numpy(),
                                          err_msg=f"t{t} env{e} obs")
            _assert_state_exact(sim, env, e, f"t{t} env{e}")
            _assert_counters_exact(sim, env, e, f"t{t} env{e}")
            done_r[e] = d_r


# --- full-episode parity (gears matrix rides the trajectory tests above) ----

def test_full_episode_parity():
    """Short charts to natural done: episode length, terminal counters,
    and every per-tick observable exact; final misses register in the
    live-head tail (v10 episode-tail rule)."""
    cfg = dict(colors="both", chart_len_s=4.0, bomb_prob=0.2, extra_prob=0.2,
               dot_prob=0.1)
    charts = _charts(cfg, range(70, 78))
    max_steps = max(int((ch[-1].hit_time + 0.5) * 30) + 90 for ch in charts)
    sims = _rust_sims(charts, 5, max_steps)
    env = _tenv(charts, 5, max_steps)
    env.reset()
    rng = np.random.default_rng(7)
    done_r = [False] * len(charts)
    for t in range(max_steps):
        acts = rng.integers(0, 5, size=(len(charts), N_JOINTS))
        tobs, trew, tdone, _ = env.step(torch.as_tensor(acts))
        for e, sim in enumerate(sims):
            if done_r[e]:
                continue
            obs_r, rew_r, d_r, _ev = sim.step([int(x) for x in acts[e]])
            assert rew_r == float(trew[e]), f"t{t} env{e} reward"
            assert d_r == bool(tdone[e]), f"t{t} env{e} done"
            np.testing.assert_array_equal(np.asarray(obs_r), tobs[e].numpy(),
                                          err_msg=f"t{t} env{e} obs")
            if d_r:
                _assert_counters_exact(sim, env, e, f"final env{e}")
                done_r[e] = True
        if all(done_r):
            break
    assert all(done_r), "episodes did not finish inside max_steps"
    total = sum(int(x) for x in
                (env.counters()[0] + env.counters()[1] + env.counters()[2]))
    assert total > 0, "trivial episode set — no note outcomes at all"


# --- (c) contact-rich parity: midline presses --------------------------------

def test_contact_rich_parity_midline_press():
    """Both arms driven across the chest midline (full yaw gear) — the
    forearm/fist vs torso pairs activate and stay pressed; contact
    impulses, obs contact bits, and full state stay exact throughout."""
    charts = [[FAR_NOTE]] * 4
    env = _tenv(charts, 3, 6900)
    env.reset()
    sims = _rust_sims(charts, 3, 6900)
    # per-env pose variants near the midline
    q36s, v35s = [], []
    for e in range(4):
        q = np.zeros(N_JOINTS)
        q[3] = -0.9 - 0.1 * e   # l_arm_pitch: arm forward-down
        q[10] = -0.9 - 0.1 * e  # r_arm_pitch
        q36 = [0.0, 0.94, -0.65, 1.0, 0.0, 0.0, 0.0] + [float(x) for x in q]
        q36s.append(q36)
        v35s.append([0.0] * NV)
        sims[e].set_joints(q36s[e], v35s[e])
    env.set_state(torch.tensor(q36s, dtype=torch.float64),
                  torch.tensor(v35s, dtype=torch.float64))
    acts = [0] * N_JOINTS
    acts[4] = 2   # l_arm_yaw toward the midline (limits are mirrored)
    acts[11] = 1  # r_arm_yaw toward the midline
    saw_contact = False
    saw_bits = False
    for t in range(45):
        tobs, trew, _td, _ = env.step(
            torch.as_tensor([acts] * 4))
        for e, sim in enumerate(sims):
            obs_r, rew_r, _d, _ev = sim.step(list(acts))
            assert rew_r == float(trew[e]), f"t{t} env{e} reward"
            np.testing.assert_array_equal(np.asarray(obs_r), tobs[e].numpy(),
                                          err_msg=f"t{t} env{e} obs")
            _assert_state_exact(sim, env, e, f"t{t} env{e}")
            if sim.contacts():
                saw_contact = True
            if np.asarray(obs_r)[98:100].any():
                saw_bits = True
    assert saw_contact, "press scenario produced no contact impulses"
    assert saw_bits, "press scenario never set the obs contact bits"


# --- (d) effort bitwise ------------------------------------------------------

def test_effort_bitwise_and_coast_free():
    charts = [[FAR_NOTE]] * 2
    env = _tenv(charts, 5, 6900)
    env.reset()
    sims = _rust_sims(charts, 5, 6900)
    acts = np.zeros((2, N_JOINTS), dtype=np.int64)
    acts[1, 10] = 1  # env1: r arm pitch full+
    acts[1, 15] = 3  # env1: r wrist bend quarter
    for t in range(5):
        _o, trew, _d, _ = env.step(torch.as_tensor(acts))
        for e, sim in enumerate(sims):
            _obs, rew_r, _dr, _ev = sim.step([int(x) for x in acts[e]])
            assert sim.last_effort() == float(env.last_effort()[e])
            assert rew_r == float(trew[e])
    assert float(env.last_effort()[0]) == 0.0        # coast exactly free
    assert float(env.last_effort()[1]) > 0.0


def test_wide_acc_parity():
    """The TRAINING-ONLY wide accuracy reward stays exact too."""
    charts = _charts(FULL, [55, 56])
    max_steps = max(int((ch[-1].hit_time + 0.5) * 30) + 90 for ch in charts)
    sims = _rust_sims(charts, 5, max_steps, wide_acc=True)
    env = _tenv(charts, 5, max_steps, wide_acc=True)
    env.reset()
    rng = np.random.default_rng(9)
    for t in range(80):
        acts = rng.integers(0, 5, size=(2, N_JOINTS))
        _o, trew, tdone, _ = env.step(torch.as_tensor(acts))
        stop = False
        for e, sim in enumerate(sims):
            _obs, rew_r, d_r, _ev = sim.step([int(x) for x in acts[e]])
            assert rew_r == float(trew[e]), f"t{t} env{e} reward"
            stop = stop or d_r
        if stop:
            break
    for e, sim in enumerate(sims):
        _assert_counters_exact(sim, env, e, f"wide env{e}")


# --- tier 2: float32 bounded -------------------------------------------------

def test_parity_f32_bounded():
    """f32 tier for the DYNAMIC body: unlike the v9 kinematic arms
    (whose CUDA tier held 5e-3 per lane over whole episodes), the v10
    stiff contact/limit dynamics amplify f32 seeds through knife-edge
    branch flips within ticks — measured under random full gears:
    3%% of obs lanes past 5e-3 after ONE tick, ~28%% (max |d| ~2 in
    normalized qd lanes) by tick 40, and discrete cut outcomes can flip.
    What f32 does guarantee — and what this test pins — is
    well-formedness (finite, joints inside limits, rates inside caps,
    root inside the step-12 envelope) and statistical agreement
    (bounded lane-divergence fraction, counters/score/reward drift)."""
    charts = _charts(FULL, [60, 61, 62, 63])
    max_steps = max(int((ch[-1].hit_time + 0.5) * 30) + 90 for ch in charts)
    sims = _rust_sims(charts, 5, max_steps)
    env = _tenv(charts, 5, max_steps, dtype=torch.float32)
    tobs = env.reset()
    rng = np.random.default_rng(3)
    lo = np.array([l[0] for l in JOINT_LIMITS])
    hi = np.array([l[1] for l in JOINT_LIMITS])
    from beat_arena_env.constants import OMEGA_CAP
    caps = np.array(OMEGA_CAP)
    r_rews = np.zeros(4)
    t_rews = np.zeros(4)
    for t in range(40):
        acts = rng.integers(0, 5, size=(4, N_JOINTS))
        tobs, trew, _td, _ = env.step(torch.as_tensor(acts))
        assert torch.isfinite(tobs).all(), f"t{t}: non-finite obs"
        q = env.dyn.q.numpy()
        qd = env.dyn.v[:, 6:].numpy()
        root_w = env.dyn.v[:, 3:6].numpy()
        root_v = env.dyn.v[:, :3].numpy()
        assert (q >= lo - 1e-4).all() and (q <= hi + 1e-4).all(), \
            f"t{t}: joints outside limits"
        assert (np.abs(qd) <= caps + 1e-3).all(), f"t{t}: rates above caps"
        assert (np.linalg.norm(root_w, axis=1) <= 96.0 + 1e-2).all()
        assert (np.linalg.norm(root_v, axis=1) <= 48.0 + 1e-2).all()
        for e, sim in enumerate(sims):
            obs_r, rew_r, _d, _ev = sim.step([int(x) for x in acts[e]])
            d = np.abs(tobs[e].numpy() - np.asarray(obs_r))
            frac = (d > 5e-3).mean()
            # once a knife edge flips, the two trajectories are
            # effectively independent bodies — the fraction can reach
            # the fraction of live lanes (~60%, measured 58% worst).
            # This is a sanity ceiling, not a closeness metric.
            assert frac <= 0.75, (f"t{t} env{e}: {frac:.0%} of lanes past "
                                  f"5e-3 — beyond even the independent-"
                                  f"trajectory band")
            r_rews[e] += rew_r
            t_rews[e] += float(trew[e])
    np.testing.assert_allclose(t_rews, r_rews, atol=3.0)
    cnt = env.counters()
    for e, sim in enumerate(sims):
        for a, b in ((sim.good_cuts, int(cnt[0][e])),
                     (sim.bad_cuts, int(cnt[1][e])),
                     (sim.misses, int(cnt[2][e]))):
            assert abs(a - b) <= 3, f"env{e} f32 counters drifted"
        assert abs(sim.score - float(cnt[4][e])) <= 120.0, \
            f"env{e} f32 score drifted"
