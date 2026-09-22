"""R12 warp-coherence env permutation (2026-08-21): the step grid walks
env_order (thread gid serves env env_order[gid]). The claim under test
is INVARIANCE — per-env results are bitwise-identical under ANY
bijection, because the permutation only reassigns threads to envs and
every P.*[e] address is unchanged. Gate: full state + obs/reward/done
equality against the identity ordering, for (a) a fixed scrambled
permutation and (b) the live sorted mode (BEAT_ENV_ORDER path), plus
run-to-run determinism of the sorted arm."""
import numpy as np
import torch

from tests._helpers import cuda

N = 64
TICKS = 90


def _env(**kw):
    from beat_arena_env.cuda_env import CudaBeatVecEnv
    return CudaBeatVecEnv(N, chart_len_s=8.0, seed=0, device="cuda",
                          foot_contact=True, gears=5, **kw)


def _acts(rng):
    return torch.as_tensor(rng.integers(0, 3, size=(N, 29)),
                           device="cuda")


_STATE = ("p", "quat", "q", "v", "score", "status", "first_pending",
          "tick", "ep_return", "swing_acc", "flail_rad", "acc_pts")


def _run(env, order=None, sort=False):
    torch.manual_seed(0)
    env.reset()
    if order is not None:
        env.env_order.copy_(order)
    env._env_order_on = sort
    rng = np.random.default_rng(7)
    outs = []
    for _ in range(TICKS):
        obs, rew, done, _ = env.step(_acts(rng))
        outs.append((obs.clone(), rew.clone(), done.clone()))
    state = [getattr(env, k).clone() for k in _STATE]
    return outs, state


def _assert_same(a, b, tag):
    for t, ((oa, ra, da), (ob, rb, db)) in enumerate(zip(a[0], b[0])):
        assert torch.equal(oa, ob), f"{tag}: obs diverged at tick {t}"
        assert torch.equal(ra, rb), f"{tag}: reward diverged at tick {t}"
        assert torch.equal(da, db), f"{tag}: done diverged at tick {t}"
    for k, sa, sb in zip(_STATE, a[1], b[1]):
        assert torch.equal(sa, sb), f"{tag}: state {k} diverged"


@cuda
def test_any_permutation_is_bitwise_identity():
    ref = _run(_env())
    # fixed scramble — the strongest form of the claim (any bijection)
    perm = torch.as_tensor(
        np.random.default_rng(3).permutation(N).astype(np.int32),
        device="cuda")
    scrambled = _run(_env(), order=perm)
    _assert_same(ref, scrambled, "scrambled")


@cuda
def test_sorted_mode_is_bitwise_identity_and_deterministic():
    ref = _run(_env())
    s1 = _run(_env(), sort=True)
    s2 = _run(_env(), sort=True)
    _assert_same(ref, s1, "sorted")
    _assert_same(s1, s2, "sorted-rerun")
