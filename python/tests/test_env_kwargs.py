"""Every train_batch env backend receives the same reward/body kwargs
(2026-09-30): the procedural --env cuda branch used to drop limit_coef /
limit_arms / wall_despawn / tau_rand / tau_center, and --env rust could
not take them at all. Construction-time only (recorder stand-ins for
the CUDA/flux classes; the real Rust BeatVecEnv for the tau draws)."""
import numpy as np
import pytest
import torch

import train_batch as tb
from beat_arena_env import constants as C


class _Rec:
    """Records its ctor kwargs; quacks enough for make_env."""
    calls = []

    def __init__(self, *a, **kw):
        type(self).calls.append(kw)
        self.kw = kw

    def set_tau_scale(self, s16):
        self.kw["_set_tau_scale"] = list(s16)


@pytest.fixture
def rec(monkeypatch):
    import beat_arena_env.cuda_env as ce
    import beat_arena_env.flux_env as fe
    import beat_arena_env.ranked_pool as rp
    _Rec.calls = []
    monkeypatch.setattr(ce, "CudaBeatVecEnv", _Rec)
    monkeypatch.setattr(fe, "FluxVecEnv", _Rec)
    monkeypatch.setattr(tb, "BeatVecEnv", _Rec)
    monkeypatch.setattr(rp, "load_ranked_pool",
                        lambda **kw: {"n_notes": torch.zeros(3)})
    monkeypatch.setattr(rp, "pool_max_steps", lambda pool: 900)
    return _Rec


# the knobs set off-default so a dropped kwarg cannot hide behind a
# matching ctor default
KNOBS = ["--limit-coef", "0.007", "--no-limit-arms", "--no-wall-despawn",
         "--tau-rand", "0.2", "--tau-center", "spine=1.1",
         "--energy-coef", "1e-5", "--self-contact-coef", "0.05",
         "--fall-pen", "4", "--foot-shape-coef", "0.02", "--gears", "5"]


def _args(*extra):
    a = tb.parse_args([*KNOBS, *extra])
    a.z_asinh = 8.0              # main() resolves this before make_env
    return a


def test_every_cuda_branch_gets_the_full_reward_body_set(rec):
    want = tb.env_kwargs(_args())
    assert want["limit_coef"] == 0.007 and want["limit_arms"] is False
    assert want["wall_despawn"] is False and want["tau_rand"] == 0.2
    assert want["tau_center"][0] == pytest.approx(1.1)
    for extra in ([], ["--no-ranked-maps"],
                  ["--no-ranked-maps", "--game", "flux"]):
        a = _args("--env", "cuda", *extra)
        if "flux" in extra:
            a.ranked_maps = a.ep_stats = False      # main()'s flux switch
        tb.make_env(a, 4, "cpu")
        got = rec.calls[-1]
        for k, v in want.items():
            assert got[k] == v, (extra, k)
        for k, v in tb.cuda_env_kwargs(a, "cpu").items():
            assert got[k] == v, (extra, k)


def test_rust_branch_gets_the_same_reward_body_set(rec, capsys):
    a = _args("--env", "rust", "--no-ranked-maps")
    tb.make_env(a, 4, "cpu")
    got = rec.calls[-1]
    for k, v in tb.env_kwargs(a).items():
        assert got[k] == v, k
    # the kernel-only terms are not passed, and their defaults are
    # announced as not simulated
    assert not set(tb.CUDA_ONLY_TERMS) & set(got)
    assert "NOT simulated" in capsys.readouterr().out


def test_rust_refuses_a_requested_cuda_only_term(rec):
    a = _args("--env", "rust", "--no-ranked-maps", "--push-prob", "0.01")
    with pytest.raises(SystemExit, match="--push-prob"):
        tb.make_env(a, 4, "cpu")
    a = _args("--env", "rust", "--no-ranked-maps", "--push-prob", "0",
              "--start-jitter", "0", "--start-pose-jitter", "0",
              "--crouch-start-prob", "0", "--push-vmax", "0")
    tb.make_env(a, 4, "cpu")                    # off = honored, silent


def test_flux_rust_refuses_terms_it_cannot_simulate(rec):
    a = _args("--env", "rust", "--no-ranked-maps", "--game", "flux")
    with pytest.raises(SystemExit, match="--game flux"):
        tb.make_env(a, 4, "cpu")                # --limit-coef 0.007 etc.
    a = tb.parse_args(["--env", "rust", "--no-ranked-maps", "--game",
                       "flux", "--tau-center", "spine=1.1"])
    tb.make_env(a, 4, "cpu")                    # defaults: notice only
    got = rec.calls[-1]
    assert "limit_coef" not in got and "tau_rand" not in got
    assert got["_set_tau_scale"][0] == pytest.approx(1.1)


# --- the real Rust BeatVecEnv ---------------------------------------------

def _tau_lanes(obs):
    return obs[:, C.OFF_TAU:C.OFF_TAU + 16] * C.TAU_RAND_NORM + 1.0


def test_rust_vec_env_draws_per_env_bodies():
    from beat_arena_env.vec_env import BeatVecEnv
    center = [1.0] * 16
    center[0] = 1.1
    env = BeatVecEnv(6, gears=5, seed=3, tau_rand=0.2, tau_center=center,
                     limit_coef=5e-3, limit_arms=True, wall_despawn=True)
    s = _tau_lanes(env.reset().copy())
    assert np.all(np.abs(s / np.asarray(center) - 1.0) <= 0.2 + 1e-5)
    assert np.unique(s[:, 3].round(5)).size == 6       # per-env draws
    assert s[:, 0].mean() > 1.0                         # around center
    # the chart stream is unchanged by the draws (own numpy stream)
    env0 = BeatVecEnv(6, gears=5, seed=3)
    o0 = env0.reset().copy()
    o = env.reset().copy()
    e = BeatVecEnv(6, gears=5, seed=3, tau_rand=0.2, tau_center=center)
    o1 = e.reset().copy()
    assert np.array_equal(o1[:, :C.OFF_TAU], o0[:, :C.OFF_TAU])
    del o


def test_rust_vec_env_fixed_center_body():
    from beat_arena_env.vec_env import BeatVecEnv
    center = list(C.TAU_BODY_DEFAULT)
    env = BeatVecEnv(3, gears=5, seed=0, tau_rand=0.0, tau_center=center)
    s = _tau_lanes(env.reset().copy())
    assert np.allclose(s, np.asarray(center)[None], atol=1e-5)
    env.step(np.zeros((3, 29), dtype=np.int64))
    with pytest.raises(AssertionError):
        BeatVecEnv(2, tau_rand=0.2).set_tau_scale([1.0] * 16)
