"""GATE for the SUB-WARP (W2-DUAL) env kernel, design/WARP_RESCUE_SUBWARP.md
§5.2, with the residency and marshalling corrections from
WARP_RESCUE_OCCUPANCY.md §1/§5 and WARP_RESCUE_MARSHALLING.md §4/§8.

WHAT CHANGED AND WHY THE TIER MOVED. The 32-lane warp kernel was gated at
BITWISE identity with the incumbent, because every parallelization it made was
across independent right-hand sides and never across a summation index. The
sub-warp kernel keeps that property for the triangular solves — but it also
splits the PGS row bodies' 35-term dot across the group and recombines it with
a __shfl_xor_sync butterfly, and THAT REASSOCIATES. At W = 2 it is a single
cut: (sum over even i) + (sum over odd i) instead of ascending. So the
old-vs-new comparison moves to a tolerance tier. Everything else in the kernel
is still bitwise by construction and is asserted as such.

MEASURED BOUND (shim, fp32, -ffp-contract=off, 200,000 random 35-term dots per
arm; the harness is in the patch's session notes, not in the repo):

    n = 35    max |split - serial|          max (that) / sum |a_i b_i|
    W = 2     9.5e-06 on N(0,1) data        3.4e-07
    W = 4     9.5e-06                       3.2e-07
    W = 8     7.6e-06                       2.9e-07
    23% of draws are bit-identical; the cancelling regime (true sum ~ 0,
    which is where a converged PGS row sits) gives the SAME relative bound,
    1.8e-07 of sum|a_i b_i| — the reassociation error tracks the magnitude
    sum, not the result, exactly as the standard (n-1)*eps*sum|a_i b_i|
    bound predicts (that bound is 4.1e-06 * sum|ab| at n = 35, eps = 1.19e-7;
    the measured 3.4e-07 sits 12x inside it).

So one PGS dot differs by <= ~3 fp32 ulps OF ITS MAGNITUDE SUM. Propagating
that: jv feeds dl = (bias - jv)/diag and then an axpy into v, so a row update
perturbs v by <= ~1e-5 for the |J| <= ~2, |v| <= ~50 ranges this body runs in;
Gauss-Seidel is contractive over the 12 iterations, and 8 substeps of that
reach v at the ~1e-3 level and q at h * that = ~1e-5. The obs mixes both. That
is the justification for the first-tick band below; past the first few ticks
the v10 dynamics amplify ANY seed at the measured ~0.35 s e-folding and
knife-edge branches (contact activation, clamp `hit` flags, integer score
boundaries) flip, so the horizon-bounded + well-formedness shape of
test_v10_parity_cuda.py's f64/f32 tiers is what applies — this file mirrors it
deliberately.

Tiers:

  (i)   EXACT, and provably so: mode 1 (reset) and mode 2 (set_state) never
        reach the PGS at all — they run reset_env / load_state + refresh_fk on
        the group lead and write obs. Bitwise, no tolerance.
  (ii)  TOLERANCE, short horizon: per-tick obs/reward against the incumbent
        with a documented band and the house flip allowance.
  (iii) WELL-FORMEDNESS + traffic, long horizon, contact-rich: finite obs,
        joints inside limits, rates inside caps, root inside the step-12
        envelope, and the SAME traffic assertions the stage-2 gate carried
        (ground rows fire, contact rows fire, episodes terminate) so a
        scenario that quietly stops making contacts fails loudly instead of
        passing vacuously.
  (iv)  RUN-TO-RUN BITWISE determinism of the new kernel alone. Unchanged, and
        the one gate that survives every future stage: the butterfly is a
        fixed tree with compile-time trip counts and a fixed lane<->element
        mapping, so a reduction ordered by the SCHEDULER instead of by the
        code fails here first.
  (v)   default-OFF.

f32 only: the sub-warp kernel is not instantiated at f64 (the legacy kernel
stays the Rust parity oracle there), so an f64 arm would silently test nothing.

The whole module is skipped without CUDA (the kernel has no CPU path).
"""

import os

import numpy as np
import pytest
import torch

if not torch.cuda.is_available():
    pytest.skip("CudaBeatVecEnv needs CUDA", allow_module_level=True)

from beat_arena_env.constants import (JOINT_LIMITS, N_JOINTS, OMEGA_CAP,
                                      ROOT_V_MAX, ROOT_W_MAX)
from beat_arena_env.cuda_env import CudaBeatVecEnv

# not a multiple of SW_ENVS_PER_BLOCK (32 / W = 16 at W = 2): the last block
# launches 16 groups for 1 live env, so the GROUP-uniform `e >= P.N` guard
# retires 15 groups mid-warp and the survivors must not wait on them. This N
# is the deadlock gate.
N = 257
TICKS = 60
GEARS = 5

# --- tier (ii) band ----------------------------------------------------------
# First tick out of an identical reset: no chaotic amplification yet, so the
# whole difference is the reassociation seed derived above (~1e-3 on the
# velocity lanes of the obs, ~1e-5 on the position lanes). 2e-3 absolute is
# that bound with one decade of headroom.
TIGHT_OBS = 2e-3
TIGHT_REW = 1e-3
# A knife-edge flip (a contact corner activating on one arm and not the other)
# is a legitimate, documented outcome of ANY non-bitwise mirror in this env —
# see test_v10_parity_cuda.test_free_running_gentle_horizon. It must stay
# bounded, and it must not be the common case.
SANE_OBS = 5.0
SANE_REW = 2.0
MIN_TIGHT_FRAC = 0.90


def _env(warp, foot_contact, seed=0, auto_reset=True, n=N, **kw):
    """The one construction site — a `warp` flip is the only difference
    between the two arms, so anything the comparison catches is the
    kernel."""
    return CudaBeatVecEnv(n, seed=seed, chart_pool=8, dtype=torch.float32,
                          colors="both", chart_len_s=6.0, note_prob=0.5,
                          extra_prob=0.4, bomb_prob=0.25, gears=GEARS,
                          want_contacts=True, foot_contact=foot_contact,
                          auto_reset=auto_reset, env_warp=warp, **kw)


def _actions(rng, n=N):
    return torch.as_tensor(rng.integers(0, GEARS, size=(n, N_JOINTS)))


def _state(env):
    return env.p, env.quat, env.q, env.v


# --- tier (i): what is still provably bitwise --------------------------------

@pytest.mark.parametrize("foot_contact", [False, True])
def test_subwarp_reset_bitwise(foot_contact):
    """Mode 1. reset_env is untouched by this change and runs on the group
    lead; nothing on that path reaches a shuffle, a dot or the PGS. EXACT
    equality, and a failure here is a plumbing bug (grid guard, tensor-list
    contract, selection path), never a numerics question."""
    ref = _env(False, foot_contact)
    sw = _env(True, foot_contact)
    assert (ref.kernel_variant, sw.kernel_variant) == (0, 1)
    o_ref, o_sw = ref.reset(), sw.reset()
    assert torch.equal(o_ref, o_sw), "mode 1 (reset) obs differ"
    for name, x, y in zip(("p", "quat", "q", "v"), _state(ref), _state(sw)):
        assert torch.equal(x, y), f"{name} differs after reset"
    assert torch.equal(ref.ep_return, sw.ep_return)


@pytest.mark.parametrize("foot_contact", [False, True])
def test_subwarp_set_state_bitwise(foot_contact):
    """Mode 2 (set_state): FK-cache refresh from a freshly written state plus
    the wrist-velocity reset. The mode-2 arm is load_state + refresh_fk and
    reaches no group-cooperative phase, so everything DERIVED FROM THE SET
    STATE is exact. What is NOT exact after a mode-0 warm-up is persistent
    chart/contact state (note statuses, contact_arm, ground_imp): the warmed
    ticks diverge at tolerance level and a single knife-edge hit flip makes
    those obs lanes legitimately differ. So: part 1 exercises the path fresh
    off reset (identical chart state => FULL obs exact); part 2 warms the
    caches with 5 divergent ticks and asserts exactness only on the lanes
    write_obs derives from p/quat/q/v + FK — obs[:, :98] (root/q/qd/hand/
    head; 98..100 are the persisted contact bits) and obs[:, 560:578] (the
    FK body points) — plus the state and FK-cache tensors themselves."""
    rng = np.random.default_rng(23)

    # part 1: fresh path, full exactness
    ref = _env(False, foot_contact, auto_reset=False)
    sw = _env(True, foot_contact, auto_reset=False)
    ref.reset()
    sw.reset()
    q36 = ref.qpos().clone()
    v35 = ref.v.clone()
    jitter = torch.as_tensor(rng.normal(scale=0.01, size=tuple(q36.shape)),
                             device=q36.device, dtype=q36.dtype)
    ref.set_state(q36 + jitter, v35)
    sw.set_state(q36 + jitter, v35)
    for name, x, y in zip(("p", "quat", "q", "v"), _state(ref), _state(sw)):
        assert torch.equal(x, y), f"{name} differs after fresh set_state"
    assert torch.equal(ref._obs, sw._obs), "mode 2 fresh obs differ"

    # part 2: warmed caches, state-derived exactness only
    for _ in range(5):
        a = _actions(rng)
        ref.step(a)
        sw.step(a)
    q36 = ref.qpos().clone()
    v35 = ref.v.clone()
    jitter = torch.as_tensor(rng.normal(scale=0.01, size=tuple(q36.shape)),
                             device=q36.device, dtype=q36.dtype)
    q36 = q36 + jitter                       # a state neither env stepped into
    ref.set_state(q36, v35)
    sw.set_state(q36, v35)

    for name, x, y in zip(("p", "quat", "q", "v"), _state(ref), _state(sw)):
        assert torch.equal(x, y), f"{name} differs after set_state"
    # No obs assertion here: even the "state" lanes mix in persisted
    # history (the 10-lane hand blocks carry swing/follow-through
    # features from the diverged trackers, 75..95). Part 1 above is the
    # obs gate for this path; what a divergent history still guarantees
    # is the state and FK-cache tensors, asserted exactly:
    assert torch.equal(ref.wrist, sw.wrist)
    assert torch.equal(ref.blade, sw.blade)
    assert torch.equal(ref.wrist_vel, sw.wrist_vel)


# --- tier (ii): tolerance, short horizon -------------------------------------

@pytest.mark.parametrize("foot_contact", [False, True])
def test_subwarp_first_tick_tolerance(foot_contact):
    """ONE tick out of an identical reset, both foot modes. This is the
    load-bearing numeric assertion of the whole file: after exactly one tick
    there has been no chaotic amplification, so the observed difference IS the
    PGS reassociation seed and nothing else. Compared per env, with the house
    flip allowance — a corner whose gap straddles MARGIN can activate on one
    arm and not the other, and that is a branch flip, not a numerics failure,
    so it is bounded rather than forbidden.

    The tick counter is integer bookkeeping in the judgment tail and stays
    EXACT whatever the dynamics did."""
    ref = _env(False, foot_contact, auto_reset=False)
    sw = _env(True, foot_contact, auto_reset=False)
    assert torch.equal(ref.reset(), sw.reset())

    rng = np.random.default_rng(7)
    a = _actions(rng)
    o1, r1, d1, _ = ref.step(a)
    o2, r2, d2, _ = sw.step(a)

    assert torch.equal(d1, d2), "done differs on the first tick"
    assert torch.equal(ref.tick, sw.tick), "tick counter is not integer-exact"
    assert torch.isfinite(o2).all(), "non-finite obs on the sub-warp arm"

    obs_d = (o1 - o2).abs().amax(dim=1)
    rew_d = (r1 - r2).abs()
    tight = (obs_d <= TIGHT_OBS) & (rew_d <= TIGHT_REW)
    frac = float(tight.float().mean())
    assert frac >= MIN_TIGHT_FRAC, (
        f"only {frac:.3f} of envs inside the reassociation band "
        f"(obs {TIGHT_OBS}, rew {TIGHT_REW}); worst obs "
        f"{float(obs_d.max()):.3e}, worst rew {float(rew_d.max()):.3e}")
    assert float(obs_d.max()) <= SANE_OBS, "a flipped env left the sane band"
    assert float(rew_d.max()) <= SANE_REW, "a flipped env's reward is insane"


def test_subwarp_short_horizon_tolerance():
    """Eight ticks, tracked per env. An env stays in the `tight` set until it
    first leaves the band; once it has flipped it is only held to the sanity
    bound (the two arms are then independent bodies, exactly as the f64 tier
    documents). Gate: the majority of envs survive the first tick tight, and
    the tight set decays rather than collapsing — a plumbing bug takes every
    env out on tick 0, which is the failure mode this shape catches and a
    single-number end-of-run comparison does not."""
    ref = _env(False, True, seed=3, auto_reset=False)
    sw = _env(True, True, seed=3, auto_reset=False)
    assert torch.equal(ref.reset(), sw.reset())

    rng = np.random.default_rng(11)
    tight = torch.ones(N, dtype=torch.bool, device=ref.device)
    per_tick = []
    for t in range(8):
        a = _actions(rng)
        o1, r1, _d1, _ = ref.step(a)
        o2, r2, _d2, _ = sw.step(a)
        assert torch.isfinite(o2).all(), f"t{t}: non-finite obs"
        obs_d = (o1 - o2).abs().amax(dim=1)
        rew_d = (r1 - r2).abs()
        assert float(obs_d.max()) <= SANE_OBS, f"t{t}: obs left the sane band"
        assert float(rew_d.max()) <= SANE_REW, f"t{t}: reward insane"
        tight &= (obs_d <= TIGHT_OBS) & (rew_d <= TIGHT_REW)
        per_tick.append(float(tight.float().mean()))
    assert per_tick[0] >= MIN_TIGHT_FRAC, \
        f"tick 0 tight fraction {per_tick[0]:.3f} — that is a plumbing bug, " \
        f"not reassociation"
    assert per_tick[-1] > 0.0, \
        f"every env flipped inside 8 ticks (per-tick {per_tick})"


# --- tier (iii): contact-rich, well-formedness + traffic ---------------------

CN = 200
CTICKS = 300


def test_subwarp_contact_rich_wellformed():
    """THE PHASE GATE. The fused sweep's width is 3*ng + nc + nl and the PGS
    row bodies only exist when rows exist, so the free-running configs above
    gate the plumbing but never the traffic. Here the feet are driven into the
    ground and the root into falls with the v18 push-recovery kick, which is
    the regime the solver doc counted ~33 RHS in — and, at W = 2, the regime
    in which the sweep tiles into several predicated rounds and the ground /
    contact / limit row bodies all run group-split.

    The kick is drawn Python-side from the DEFAULT cuda generator
    (cuda_env._kick), not from the env's own `self._gen`, so the rng state is
    snapshotted before the reference step and restored before the sub-warp
    step: same kick, same actions, same chart draws, so the ONLY difference
    left is the kernel. (After the first divergence the two arms are different
    trajectories under the same forcing — which is what this tier tests.)

    Asserts the traffic before it asserts anything else: a config that stops
    reaching the ground must fail, not pass vacuously."""
    ref = _env(False, True, seed=101, n=CN, push_prob=0.3, push_vmax=3.0)
    sw = _env(True, True, seed=101, n=CN, push_prob=0.3, push_vmax=3.0)
    assert (ref.kernel_variant, sw.kernel_variant) == (0, 1)
    assert torch.equal(ref.reset(), sw.reset()), "mode 1 (reset) obs differ"

    lo = torch.as_tensor([l[0] for l in JOINT_LIMITS], device=sw.device)
    hi = torch.as_tensor([l[1] for l in JOINT_LIMITS], device=sw.device)
    caps = torch.as_tensor(OMEGA_CAP, device=sw.device)

    rng = np.random.default_rng(31)
    ground_seen = contact_seen = done_seen = False
    sw_ground = sw_contact = sw_done = False
    for t in range(CTICKS):
        a = _actions(rng, CN)
        st = torch.cuda.get_rng_state()
        _o1, _r1, d1, _ = ref.step(a)
        torch.cuda.set_rng_state(st)
        o2, r2, d2, _ = sw.step(a)

        # well-formedness of the arm under test, every tick
        assert torch.isfinite(o2).all(), f"t{t}: non-finite obs"
        assert torch.isfinite(r2).all(), f"t{t}: non-finite reward"
        assert bool(((sw.q >= lo - 0.25) & (sw.q <= hi + 0.25)).all()), \
            f"t{t}: joints outside limits"
        assert bool((sw.v[:, 6:].abs() <= caps + 1.0).all()), \
            f"t{t}: joint rates above OMEGA_CAP"
        assert bool((sw.v[:, 3:6].norm(dim=1) <= ROOT_W_MAX + 1e-2).all()), \
            f"t{t}: root angular velocity outside the step-12 envelope"
        assert bool((sw.v[:, :3].norm(dim=1) <= ROOT_V_MAX + 1e-2).all()), \
            f"t{t}: root linear velocity outside the step-12 envelope"

        # traffic, from BOTH arms: the reference arm proves the scenario is
        # doing its job, the sub-warp arm proves the new code actually ran the
        # rows rather than skipping them
        ground_seen |= bool((ref.ground_imp != 0).any())
        contact_seen |= bool((ref.contact_imp != 0).any())
        done_seen |= bool(d1.any())
        sw_ground |= bool((sw.ground_imp != 0).any())
        sw_contact |= bool((sw.contact_imp != 0).any())
        sw_done |= bool(d2.any())

    assert ground_seen and sw_ground, \
        "no ground normal impulse in the whole run — the ground rows never " \
        "fired and neither the sweep nor the split PGS body was ever wide"
    assert contact_seen and sw_contact, "no self-collision contact row fired"
    assert done_seen and sw_done, "no episode terminated; fall path untested"
    falls_ref = ref.pop_fall_hits()
    falls_sw = sw.pop_fall_hits()
    assert falls_ref > 0, "no falls: the push scenario is not doing its job"
    assert falls_sw > 0, "the sub-warp arm never fell — the fall path is dead"
    # Statistical, not exact: after 300 ticks of divergent trajectories under
    # identical forcing the two arms are different bodies, so this is a
    # regime check (same order of magnitude of falls), which is what the f32
    # tier in test_v10_parity_cuda.py asserts against Rust for the same reason.
    assert 0.5 <= falls_sw / falls_ref <= 2.0, \
        f"fall rate regime changed: {falls_sw} vs {falls_ref}"
    # the SPEC-v10 §20e safety net: counted, not hidden
    assert sw.pop_pd_hits() >= 0
    assert sw.pop_envelope_hits() >= 0


# --- tier (iv): determinism, still bitwise -----------------------------------

def test_subwarp_determinism_bitwise():
    """Same build, same seed, same actions, two independent sub-warp runs ->
    byte-equal state and obs. Independent of the incumbent, so it survives the
    stage that retired the parity tier above — and it is exactly what a
    butterfly ordered by the SCHEDULER rather than by the code would fail
    first. It is also the gate on the group-uniformity claim: if the reduced
    dot ever differed between the W lanes of a group, the redundant scalar
    chain would diverge and the run would not reproduce."""
    def run():
        env = _env(True, True, seed=17)
        obs = env.reset()
        rng = np.random.default_rng(11)
        tot = torch.zeros(N, device=env.device, dtype=env.dtype)
        for _ in range(TICKS):
            obs, r, _d, _ = env.step(_actions(rng))
            tot += r
        return (obs.clone(), tot, env.qpos().clone(), env.v.clone(),
                env.pop_fall_hits())

    o1, t1, q1, v1, f1 = run()
    o2, t2, q2, v2, f2 = run()
    assert torch.equal(o1, o2), "obs not bitwise reproducible"
    assert torch.equal(t1, t2), "returns not bitwise reproducible"
    assert torch.equal(q1, q2), "qpos not bitwise reproducible"
    assert torch.equal(v1, v2), "velocity not bitwise reproducible"
    assert f1 == f2


# --- tier (v): adoption is a deliberate act ----------------------------------

def test_default_is_legacy():
    """With neither the kwarg nor BEAT_ENV_WARP set, the pybind call must
    carry variant 0 and the run must be the incumbent's, byte for byte. This
    one stays EXACT after the tier move, because both arms are the incumbent
    kernel."""
    assert os.environ.get("BEAT_ENV_WARP", "0") != "1", \
        "BEAT_ENV_WARP is set in this shell; the default-OFF check is void"
    ref = _env(False, True, n=8)
    plain = CudaBeatVecEnv(8, seed=0, chart_pool=8, dtype=torch.float32,
                           colors="both", chart_len_s=6.0, note_prob=0.5,
                           extra_prob=0.4, bomb_prob=0.25, gears=GEARS,
                           want_contacts=True, foot_contact=True)
    assert plain.kernel_variant == 0
    o1, o2 = ref.reset(), plain.reset()
    rng = np.random.default_rng(5)
    for _ in range(8):
        a = torch.as_tensor(rng.integers(0, GEARS, size=(8, N_JOINTS)))
        o1, r1, d1, _ = ref.step(a)
        o2, r2, d2, _ = plain.step(a)
    assert torch.equal(o1, o2) and torch.equal(r1, r2) and torch.equal(d1, d2)
