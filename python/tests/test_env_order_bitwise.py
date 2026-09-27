"""R12 warp-coherence env permutation (2026-08-21): the step grid walks
env_order (thread gid serves env env_order[gid]). The claim under test
is INVARIANCE — per-env results are bitwise-identical under the
permutation, because it only reassigns threads to envs and every P.*[e]
address is unchanged. Gate: full state + obs/reward/done equality of the
live sorted mode (CudaBeatVecEnv(env_order=True), what the trainer runs)
against the identity ordering, with the sort shown to have actually
permuted the envs. Sorted-mode run-to-run determinism follows: sorted
== identity bitwise, and the identity order's determinism is pinned by
test_v11_contact_parity.py::test_contact_determinism_bitwise."""
import numpy as np
import torch

from tests._helpers import cuda

N = 16
TICKS = 30


def _env(**kw):
    from beat_arena_env.cuda_env import CudaBeatVecEnv
    return CudaBeatVecEnv(N, chart_len_s=8.0, seed=0, device="cuda",
                          gears=5, **kw)


def _acts(rng):
    return torch.as_tensor(rng.integers(0, 3, size=(N, 29)),
                           device="cuda")


_STATE = ("p", "quat", "q", "v", "score", "status", "first_pending",
          "tick", "ep_return", "swing_acc", "flail_rad", "acc_pts")


def _run(env):
    """-> (per-tick outputs, final state, #ticks stepped under a
    non-identity env_order)."""
    torch.manual_seed(0)
    env.reset()
    ident = torch.arange(N, dtype=torch.int32, device="cuda")
    rng = np.random.default_rng(7)
    outs, permuted = [], 0
    for _ in range(TICKS):
        obs, rew, done, _ = env.step(_acts(rng))
        outs.append((obs.clone(), rew.clone(), done.clone()))
        permuted += int(not torch.equal(env.env_order, ident))
    state = [getattr(env, k).clone() for k in _STATE]
    return outs, state, permuted


def _assert_same(a, b, tag):
    for t, ((oa, ra, da), (ob, rb, db)) in enumerate(zip(a[0], b[0])):
        assert torch.equal(oa, ob), f"{tag}: obs diverged at tick {t}"
        assert torch.equal(ra, rb), f"{tag}: reward diverged at tick {t}"
        assert torch.equal(da, db), f"{tag}: done diverged at tick {t}"
    for k, sa, sb in zip(_STATE, a[1], b[1]):
        assert torch.equal(sa, sb), f"{tag}: state {k} diverged"


@cuda
def test_sorted_mode_is_bitwise_identity():
    ref_env = _env()
    assert ref_env._env_order_on is False     # constructor default: off
    ref = _run(ref_env)
    assert ref[2] == 0                        # off: identity throughout
    e1 = _env(env_order=True)
    assert e1._env_order_on is True
    s1 = _run(e1)
    # the sort must have actually reassigned threads, or the equality
    # below is vacuous
    assert s1[2] > 0, "env_order never left the identity permutation"
    _assert_same(ref, s1, "sorted")
