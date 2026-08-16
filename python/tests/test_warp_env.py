"""STAGE-1 GATE for the warp-per-env kernel (design/WARP_ENV_SHELL.md §6,
WARP_ENV_RECONCILIATION.md S1).

Stage 1 maps one env to one WARP but still runs the entire body serially
on lane 0, through the same device functions, in the same order. It is
therefore not an approximation of the incumbent — it is the SAME
arithmetic, and the only gate strong enough to be worth running is
BITWISE identity. Anything that differs is a plumbing bug (grid guard,
tensor-list contract, selection path), never a numerics question. That
is the whole point of paying for a scaffold stage that is expected to be
SLOWER: it validates the shell before a line of solver math moves.

Two tiers, both from the shell doc's validation plan:

  (i)  old-vs-new bitwise identity at f32, BOTH foot_contact modes, over
       mode 0 (step), mode 1 (reset) and mode 2 (set_state), with
       N = 257 — deliberately not a multiple of ENVS_PER_BLOCK = 4, so
       the last block runs a partial warp complement and the
       warp-uniform `e >= P.N` guard is exercised rather than assumed;
  (ii) run-to-run bitwise determinism of the new kernel alone (§5b) —
       the cheapest possible variant gate, and the one that stays
       meaningful after stage 3 retires tier (i).

f32 only: the warp kernel is not instantiated at f64 (the legacy kernel
stays the Rust parity oracle there), so an f64 arm would silently test
nothing.

The whole module is skipped without CUDA (the kernel has no CPU path).
"""

import os

import numpy as np
import pytest
import torch

if not torch.cuda.is_available():
    pytest.skip("CudaBeatVecEnv needs CUDA", allow_module_level=True)

from beat_arena_env.constants import N_JOINTS
from beat_arena_env.cuda_env import CudaBeatVecEnv

# not a multiple of ENVS_PER_BLOCK (4): the last block launches 4 warps
# for 1 live env, so lanes that must retire together do
N = 257
TICKS = 60
GEARS = 5


def _env(warp, foot_contact, seed=0, auto_reset=True, n=N, **kw):
    """The one construction site — a `warp` flip is the only difference
    between the two arms, so anything the comparison catches is the
    kernel."""
    return CudaBeatVecEnv(n, seed=seed, chart_pool=8, dtype=torch.float32,
                          colors="both", chart_len_s=6.0, note_prob=0.5,
                          extra_prob=0.4, bomb_prob=0.25, gears=GEARS,
                          want_contacts=True, foot_contact=foot_contact,
                          auto_reset=auto_reset, env_warp=warp, **kw)


def _actions(rng):
    return torch.as_tensor(rng.integers(0, GEARS, size=(N, N_JOINTS)))


def _state(env):
    return env.p, env.quat, env.q, env.v


@pytest.mark.parametrize("foot_contact", [False, True])
def test_warp_bitwise_identical_to_legacy(foot_contact):
    """Modes 0 and 1: same seed, same config, same actions, one env on
    each kernel -> byte-equal obs/reward/done EVERY tick and byte-equal
    dynamics state at the end. Compared per tick, not just at the end:
    v10 amplifies a 1-ulp seed at ~0.35 s, so an end-only comparison
    would report a plumbing bug as a chaotic divergence and lose the
    tick it happened on. auto_reset=True keeps the in-kernel reset and
    the chart draw inside the comparison (both envs draw from
    identically seeded generators)."""
    ref = _env(False, foot_contact)
    warp = _env(True, foot_contact)
    assert (ref.kernel_variant, warp.kernel_variant) == (0, 1)

    o_ref, o_warp = ref.reset(), warp.reset()          # mode 1
    assert torch.equal(o_ref, o_warp), "mode 1 (reset) obs differ"

    rng = np.random.default_rng(7)
    for t in range(TICKS):                             # mode 0
        a = _actions(rng)
        o1, r1, d1, _ = ref.step(a)
        o2, r2, d2, _ = warp.step(a)
        assert torch.equal(o1, o2), f"obs differ at tick {t}"
        assert torch.equal(r1, r2), f"reward differs at tick {t}"
        assert torch.equal(d1, d2), f"done differs at tick {t}"

    for name, x, y in zip(("p", "quat", "q", "v"),
                          _state(ref), _state(warp)):
        assert torch.equal(x, y), f"{name} differs after {TICKS} ticks"
    assert torch.equal(ref.ep_return, warp.ep_return)
    assert torch.equal(ref.contact_imp, warp.contact_imp)
    # counters are per-launch atomics; equal totals mean the same events
    # fired the same number of times, and (with 31 lanes retired in stage
    # 1) that no atomic fired 32x
    assert ref.pop_pd_hits() == warp.pop_pd_hits()
    assert ref.pop_envelope_hits() == warp.pop_envelope_hits()
    assert ref.pop_fall_hits() == warp.pop_fall_hits()


@pytest.mark.parametrize("foot_contact", [False, True])
def test_warp_bitwise_set_state(foot_contact):
    """Mode 2 (set_state): FK-cache refresh from a freshly written state
    plus the wrist-velocity reset, on a state the envs did not reach
    themselves. Hit last so the caches it overwrites are warm."""
    ref = _env(False, foot_contact, auto_reset=False)
    warp = _env(True, foot_contact, auto_reset=False)
    ref.reset()
    warp.reset()
    rng = np.random.default_rng(23)
    for _ in range(5):
        a = _actions(rng)
        ref.step(a)
        warp.step(a)

    q36 = ref.qpos().clone()
    v35 = ref.v.clone()
    jitter = torch.as_tensor(rng.normal(scale=0.01, size=tuple(q36.shape)),
                             device=q36.device, dtype=q36.dtype)
    q36 = q36 + jitter                       # a state neither env stepped into
    ref.set_state(q36, v35)
    warp.set_state(q36, v35)

    for name, x, y in zip(("p", "quat", "q", "v"),
                          _state(ref), _state(warp)):
        assert torch.equal(x, y), f"{name} differs after set_state"
    assert torch.equal(ref._obs, warp._obs), "mode 2 obs differ"
    assert torch.equal(ref.wrist, warp.wrist)
    assert torch.equal(ref.blade, warp.blade)
    assert torch.equal(ref.wrist_vel, warp.wrist_vel)

    # one more tick from the refreshed state: catches an FK cache that
    # agrees on the obs but not on what the next substep reads
    a = _actions(rng)
    o1, r1, _d1, _ = ref.step(a)
    o2, r2, _d2, _ = warp.step(a)
    assert torch.equal(o1, o2) and torch.equal(r1, r2)


def test_warp_determinism_bitwise():
    """Same build, same seed, same actions, two independent warp-kernel
    runs -> byte-equal state and obs. Independent of the incumbent, so it
    survives the stage at which reassociation retires the parity tier
    above — and it is what a shuffle tree ordered by the SCHEDULER rather
    than by the code would fail first."""
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


def test_default_is_legacy():
    """Adoption is a deliberate act: with neither the kwarg nor
    BEAT_ENV_WARP set, the pybind call must carry variant 0 and the run
    must be the incumbent's, byte for byte."""
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
