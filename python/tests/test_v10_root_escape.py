"""SPEC-v10 Divergences §23 — the root-escape regression (A1 + the
envelope-hit counter + the C backstop), both live tiers (Rust
reference + CUDA mirror; the torch mirror was removed 2026-08-03).

The defect (2026-08-03 investigation, research/2026-08-03-root-escape-
design.md): step 8's ω-cap/limit stage is an OBLIQUE projector under the
§15c weld-null-space projection (‖Q_j‖_M measured 90–116 for ankle joints
just above COMPOSITE_EPS), so its 12-iteration loop can diverge
geometrically. Step 10 then integrated the resulting 1e5-rad/s root
velocity and teleported the body hundreds of metres, whose Baumgarte weld
bias (β/h·Δp) fed the divergence on the next substep.

The fix landed here is A1: the §19b root-velocity envelope is applied a
SECOND time, immediately BEFORE the step-10 integration (the step-12
application stays — step 11's joint impulses have non-zero root rows).
That bounds ‖Δp‖ ≤ h·ROOT_V_MAX = 0.2 m per substep unconditionally, and
because the envelope engages only on artifact states it is bitwise
transparent on healthy trajectories (measured: 0 engagements and a
bitwise-identical 204,800-env-tick converged-policy rollout).

Engagements of the PRE-integration application are counted
(pop_envelope_hits, mirroring pop_pd_hits) and terminate the episode with
a ZEROED reward — an escaped trajectory is bounded but not physical, and
feeding its observations/rewards to the policy is the actual harm.

Fixture: escape #7 of the 2026-08-03 corpus (CUDA f32 env, N=4096 seed 1
gears 3, env 48 @ tick 17) — the state at the START of a control tick
during which the root moved 642.7 m without an auto-reset. It looks
ordinary: |v_root| 0.15 m/s, |ω_root| 6.4 rad/s, |q̇|max at the wrist cap.
Pre-fix the Rust f64 tier moved it 31.1 m in the FIRST substep alone.
"""

import numpy as np
import pytest
import torch

import beat_arena_rs
from beat_arena_env.chart import note_tuple
from beat_arena_env.constants import H_SUB, N_JOINTS, N_SUB, NV, ROOT_V_MAX
# FAR_NOTE is beyond every horizon: the tick is pure body dynamics, so
# `done` can only come from the C backstop
from tests._helpers import FAR_NOTE

ESC_Q36 = [
    -0.18843580782413483, 0.8035035133361816, -1.077133059501648,
    0.8241789937019348, 0.5093252062797546, 0.09989655017852783,
    -0.2265780121088028, 0.7853981852531433, -0.6981030106544495,
    -0.47437214851379395, 0.9689304828643799, -0.1498527228832245,
    0.566002607345581, 0.8901060819625854, 0.07968739420175552,
    1.0873289108276367, 0.2617993950843811, 0.14637738466262817,
    1.3688384294509888, 0.45331719517707825, 0.8741688132286072,
    -0.5112427473068237, -0.5565317869186401, 0.33905500173568726,
    1.5271868705749512, -0.5984416604042053, 0.294033020734787,
    0.005854796152561903, -0.4577142298221588, -0.22766704857349396,
    1.6427476406097412, -0.6469371318817139, 0.26761528849601746, 0.0,
    -0.5667280554771423, -0.2737051248550415,
]
ESC_V35 = [
    0.012548212893307209, 0.0441480427980423, -0.1442444622516632,
    -5.172290325164795, -1.5389772653579712, 3.447293281555176, 0.0,
    -10.52071762084961, -4.945155620574951, -10.435583114624023,
    14.81192398071289, 6.78498649597168, 15.845629692077637, -40.0,
    25.100996017456055, 0.0, -16.0, 10.470308303833008, -15.397571563720703,
    4.1626200675964355, -5.062375545501709, -18.275327682495117,
    39.9994010925293, -14.463310241699219, 5.370003700256348,
    5.604532241821289, -15.959772109985352, -8.515240669250488,
    -0.1936049610376358, -6.113821506500244, 3.4726436138153076,
    0.6403346657752991, 0.0, 0.18586382269859314, 0.2472548931837082,
]
ESC_ACT = [0, 2, 0, 1, 1, 0, 2, 1, 2, 0, 1, 0, 0, 0, 1, 1, 1, 0, 1, 1, 1, 1,
           1, 2, 0, 1, 1, 0, 0]

# one tick = 8 substeps, each bounded by h·ROOT_V_MAX by construction
TICK_DP_BOUND = N_SUB * H_SUB * ROOT_V_MAX      # 1.6 m
NEUTRAL_V = [0.0] * NV


def _rust_sim(gears=3):
    sim = beat_arena_rs.RawBeatSim(max_steps=6900,
                                   gears=gears)
    sim.set_chart([note_tuple(FAR_NOTE)])
    return sim


# --- (a) Rust reference ------------------------------------------------------

def test_rust_escape_terminates_bounded_and_counted():
    sim = _rust_sim()
    assert sim.pop_envelope_hits() == 0
    sim.set_joints(ESC_Q36, ESC_V35)
    p0 = np.array(sim.qpos()[:3])
    obs, reward, done, _ev = sim.step(ESC_ACT)

    hits = sim.pop_envelope_hits()
    assert hits > 0, "the escape state must engage the pre-integration envelope"
    assert sim.pop_envelope_hits() == 0, "pop must zero the counter"
    assert sim.escaped, "the per-tick escape flag must be set"
    assert done, "an escaped env must terminate (C backstop)"
    assert reward == 0.0, "an escaped tick's reward must be zeroed"

    p1 = np.array(sim.qpos()[:3])
    dp = float(np.linalg.norm(p1 - p0))
    assert dp <= TICK_DP_BOUND + 1e-9, (
        f"tick |dp| = {dp} m — the pre-fix teleport (31.1 m in substep 0 "
        f"alone, 642.7 m over the tick in the capturing f32 env) survived")
    assert np.isfinite(obs).all(), "escaped obs must stay finite"
    assert np.isfinite(np.array(sim.twist())).all()
    assert np.isfinite(np.array(sim.omega())).all()


def test_rust_healthy_tick_never_escapes():
    """The counter is a signal, not noise: ordinary motion from neutral."""
    sim = _rust_sim(gears=5)
    acts = [0] * N_JOINTS
    acts[10] = 1
    acts[3] = 2
    acts[15] = 1
    for _ in range(120):
        _obs, _r, done, _ev = sim.step(acts)
        assert not sim.escaped
        if done:
            break
    assert sim.pop_envelope_hits() == 0


def test_vec_sim_pop_envelope_hits_zero_in_ordinary_motion():
    """VecBeatSim carries the same popped counter (it has no set_joints,
    so the escape path is pinned on RawBeatSim above); here it must stay
    silent across a healthy multi-env rollout."""
    vec = beat_arena_rs.VecBeatSim(4, max_steps=6900,
                                   gears=5)
    for i in range(4):
        vec.set_chart(i, [note_tuple(FAR_NOTE)])
    assert vec.pop_envelope_hits() == 0
    acts = np.zeros((4, N_JOINTS), dtype=np.int64)
    acts[:, 10] = 1
    acts[:, 3] = 2
    for _ in range(60):
        vec.step(acts)
    assert vec.pop_envelope_hits() == 0


# --- (b) CUDA tier ----------------------------------------------------------
# (The torch mirror carried a THIRD tier here — a bitwise Rust<->torch f64
#  comparison on the escape tick — until the mirror was removed 2026-08-03.
#  Everything structural it pinned is pinned below against Rust directly:
#  done, zeroed reward, the engagement count, and the h*ROOT_V_MAX bound.)

@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
# f32 (production) only: f64 structure is pinned against Rust below
@pytest.mark.parametrize("dtype", [torch.float32])
def test_cuda_escape_terminates_bounded_and_counted(dtype):
    from beat_arena_env.cuda_env import CudaBeatVecEnv
    env = CudaBeatVecEnv(2, charts=[[FAR_NOTE], [FAR_NOTE]], max_steps=6900,
                         gears=3,
                         dtype=dtype, auto_reset=False)
    env.reset()
    assert env.pop_envelope_hits() == 0
    # env 0 escapes; env 1 sits at the neutral pose and must not
    env.set_state(torch.tensor([ESC_Q36, list(env.qpos()[1].cpu().numpy())],
                               dtype=dtype),
                  torch.tensor([ESC_V35, NEUTRAL_V], dtype=dtype))
    p0 = env.p.clone()
    acts = torch.tensor([ESC_ACT, [0] * N_JOINTS], dtype=torch.int64,
                        device=env.device)
    _obs, rew, done, _ = env.step(acts)
    dp = (env.p - p0).norm(dim=-1)

    assert bool(done[0]), "escaped env must terminate"
    assert not bool(done[1]), "the healthy env must not"
    assert float(rew[0]) == 0.0, "escaped reward must be zeroed"
    hits = env.pop_envelope_hits()
    assert hits > 0
    assert env.pop_envelope_hits() == 0, "pop must zero the counter"
    assert float(dp[0]) <= TICK_DP_BOUND + 1e-6, f"tick |dp| = {float(dp[0])}"
    assert torch.isfinite(env.v).all() and torch.isfinite(env.p).all()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_cuda_f64_escape_matches_rust_closely():
    """P5 tier: the CUDA f64 mirror is 1-ulp-per-op, not bitwise (§21), so
    the escape tick is compared at the single-tick tolerance the rest of
    test_v10_parity_cuda uses — plus the exact structural facts (done,
    zeroed reward, engagement) which must match Rust exactly."""
    from beat_arena_env.cuda_env import CudaBeatVecEnv
    env = CudaBeatVecEnv(1, charts=[[FAR_NOTE]], max_steps=6900, gears=3,
                         dtype=torch.float64, auto_reset=False)
    env.reset()
    env.set_state(torch.tensor([ESC_Q36], dtype=torch.float64),
                  torch.tensor([ESC_V35], dtype=torch.float64))
    sim = _rust_sim()
    sim.set_joints(ESC_Q36, ESC_V35)
    _o, rew, done, _ = env.step(
        torch.tensor([ESC_ACT], dtype=torch.int64, device=env.device))
    _or, rr, dr, _ = sim.step(ESC_ACT)
    assert bool(done[0]) == dr is True
    assert float(rew[0]) == rr == 0.0
    assert env.pop_envelope_hits() == sim.pop_envelope_hits()
    # the escaped state is a chaotic one; the STRUCTURAL guarantee is what
    # is pinned tightly, the state only to the single-tick f64 band
    np.testing.assert_allclose(env.qpos()[0].cpu().numpy(),
                               np.array(sim.qpos()), rtol=0, atol=1e-6)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_cuda_healthy_population_never_engages():
    """The gate the amendment rests on: no engagement on healthy play, so
    the counter and the C backstop are silent there."""
    from beat_arena_env.cuda_env import CudaBeatVecEnv
    env = CudaBeatVecEnv(64, seed=0, chart_pool=64, dtype=torch.float32,
                         gears=5)
    env.reset()
    acts = torch.zeros(64, N_JOINTS, dtype=torch.int64, device=env.device)
    acts[:, 10] = 1
    acts[:, 3] = 2
    for _ in range(60):
        env.step(acts)
    assert env.pop_envelope_hits() == 0
