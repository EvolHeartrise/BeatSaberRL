"""env v18 pins (2026-08-15): constant fall pricing + perturbation kicks.

Constant pricing is the SOLE fall-pricing mode since 2026-08-15 (the
legacy discounted remaining-notes charge is removed): falls in
note-sparse states (segment tails, map intros) were near-free under
the old charge and the policy learned fall-indifference exactly
there (fpm replay analysis; the 3-arm probe measured falls/update
8-10k -> ~35 and bench zeros 35 -> 2 after the fix). Default
fall_const = 5.0 = the probe winner; pushes default ON in
train_batch but OFF at the env layer (parity/tests).

Contact-mode zero-action episodes fall on their own (the unactuated
body slumps; the wall tests lean on the same physics), which gives a
deterministic fall without crafting actions.
"""

import pytest

import beat_arena_rs
from beat_arena_env.chart import Note, note_tuple
from tests._helpers import ZERO_ACT

SPARSE = [Note(hit_time=100.0, col=2, row=1, x=2.5, y=1.4, color=0,
               angle=4.712, speed=4.0)]
DENSE = [Note(hit_time=5.0 + 0.5 * i, col=2, row=1, x=0.3, y=1.4,
              color=i % 2, angle=4.712, speed=4.0) for i in range(50)]


def _fall_reward(chart, fall_const=None):
    kw = {} if fall_const is None else {"fall_const": fall_const}
    sim = beat_arena_rs.RawBeatSim(max_steps=600,
                                   gears=5,
                                   foot_contact=True, **kw)
    sim.set_chart([note_tuple(n) for n in chart], [])
    for _ in range(600):
        _obs, r, done, _ev = sim.step(ZERO_ACT)
        if done:
            return r
    raise AssertionError("episode never terminated")


def test_const_pricing_is_flat():
    r_sparse = _fall_reward(SPARSE, fall_const=5.0)
    r_dense = _fall_reward(DENSE, fall_const=5.0)
    assert r_sparse == pytest.approx(-5.0)
    assert r_dense == pytest.approx(-5.0)


def test_default_is_five():
    # constructor default = the probe winner; no kwarg needed
    assert _fall_reward(DENSE) == pytest.approx(-5.0)
    assert _fall_reward(SPARSE) == pytest.approx(-5.0)


def test_magnitude_is_respected():
    assert _fall_reward(DENSE, fall_const=2.5) == pytest.approx(-2.5)


@pytest.mark.skipif(not __import__("torch").cuda.is_available(),
                    reason="CUDA env")
def test_cuda_const_pricing_and_push_smoke():
    import torch
    from beat_arena_env.cuda_env import CudaBeatVecEnv

    charts = [DENSE, SPARSE, DENSE, SPARSE]
    env = CudaBeatVecEnv(4, charts=charts, gears=5, foot_contact=True,
                         fall_const=5.0, auto_reset=False,
                         device="cuda", dtype=torch.float64)
    env.reset()
    acts = torch.zeros(4, 29, dtype=torch.long, device="cuda")
    done_r = torch.full((4,), float("nan"), dtype=torch.float64,
                        device="cuda")
    for _ in range(600):
        _obs, r, done, _ = env.step(acts)
        new = done & torch.isnan(done_r)
        done_r[new] = r[new]
        if bool(done.all()):
            break
    assert bool(done.all())
    assert torch.allclose(done_r, torch.full_like(done_r, -5.0))

    # push smoke: prob 1, vmax 1 -> root linear velocity must move
    env2 = CudaBeatVecEnv(4, charts=charts, gears=5, foot_contact=True,
                          push_prob=1.0, push_vmax=1.0,
                          device="cuda", dtype=torch.float64)
    env2.reset()
    v0 = env2.v[:, :3].clone()
    env2.step(acts)
    assert not torch.allclose(env2.v[:, :3], v0)
