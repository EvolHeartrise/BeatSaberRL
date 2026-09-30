"""Review B10 (2026-09-30): the start-jitter kick lands BEFORE the reset
obs is written.

Before the fix CudaBeatVecEnv.step() added the episode-start root-
velocity kick (|dv| ~ U(0, start_jitter), horizontal, direction uniform)
to v AFTER the kernel had already reset the terminated envs and written
their obs ("obs catches up next tick"), so the first obs of every
episode read root linear velocity 0 while the body moved at up to
start_jitter m/s — a wrong state for V and pi on the balance-critical
first tick; reset() applied no kick at all. Now the wrapper precomputes
the kick into _start_kick before each launch (same default-RNG draws,
same order, same arithmetic as before) and the kernel's reset_env
installs it before store_state / write_obs, on every reset path
(auto-reset, reset(), bench refill). With every joint rate and the
root's angular velocity zero the kick is a rigid translation, so the
finite-difference velocity lanes (wrists, palms, the §8 body points)
start at the same (dvx, 0, dvz).

The Rust sim has no per-episode RNG (no kick, pose jitter, crouch or tau
draws — lib.rs OFF_TAU comment), so there is no Rust twin to fix; with
start_jitter = 0 the kernel stores the old zeros, so every parity suite
is unchanged.

The CPU test pins the tensor-list contract (the new slot sits at the
same index in cuda_env's _tensors and the kernel's enum); the GPU tests
skip without CUDA (run them on the pod).
"""

import ast
import math
import os
import re

import numpy as np
import pytest

from beat_arena_env.constants import (N_JOINTS, OFF_BODY_PT_VEL, OFF_HAND_L,
                                      OFF_HAND_R, OFF_ROOT_VANG,
                                      OFF_ROOT_VLIN, ROOT_VLIN_NORM, V_NORM)
from tests._helpers import cuda

_PKG = os.path.join(os.path.dirname(__file__), "..", "beat_arena_env")
N_BODY_PTS = 6
KICK = 0.15


# -- CPU: the tensor-list contract -------------------------------------------
def _enum_slots():
    src = open(os.path.join(_PKG, "csrc", "beat_arena_cuda.cu")).read()
    body = re.search(r"enum\s*\{\s*(TP = 0.*?)\bT_COUNT\b", src, re.S).group(1)
    body = re.sub(r"//[^\n]*", "", body)
    return [t.split("=")[0].strip() for t in body.split(",") if t.strip()]


def _py_slots():
    tree = ast.parse(open(os.path.join(_PKG, "cuda_env.py")).read())
    for node in ast.walk(tree):
        if (isinstance(node, ast.Assign) and len(node.targets) == 1
                and isinstance(node.targets[0], ast.Attribute)
                and node.targets[0].attr == "_tensors"):
            return [ast.unparse(e) for e in node.value.elts]
    raise AssertionError("self._tensors = [...] not found")


def test_start_kick_tensor_slot_contract():
    enum, py = _enum_slots(), _py_slots()
    assert len(enum) == len(py), (len(enum), len(py))
    i = enum.index("TSTARTKICK")
    assert py[i] == "self._start_kick"
    assert py[enum.index("TPALMV")] == "self.palm_vel"
    assert py[enum.index("TPOSEJIT")] == "self._pose_jit"


# -- GPU ----------------------------------------------------------------------
def _charts(n):
    from beat_arena_env.chart import ChartConfig, generate_chart
    cfg = ChartConfig(chart_len_s=4.0, colors="both")
    rng = np.random.default_rng(7)
    return [generate_chart(cfg, rng) for _ in range(n)]


def _env(dtype, n=64, **kw):
    import torch
    from beat_arena_env.cuda_env import CudaBeatVecEnv
    return CudaBeatVecEnv(n, charts=_charts(n), max_steps=24, gears=5,
                          seed=0, device="cuda",
                          dtype=getattr(torch, dtype), **kw)


def _vel_lane_checks(env, ids, kx, kz, tol):
    """obs of envs `ids` == the kicked state (root vlin, zero vang, the
    finite-diff hand / body-point velocity lanes)."""
    import torch
    obs = env._obs[ids].double()
    v = env.v[ids].double()
    # the kick IS the root's velocity: horizontal, nothing else moving
    assert torch.equal(v[:, 0], kx) and torch.equal(v[:, 2], kz)
    assert torch.count_nonzero(v[:, 1]) == 0
    assert torch.count_nonzero(v[:, 3:]) == 0
    # the tentpole: obs root vlin lanes == the body's actual velocity
    torch.testing.assert_close(obs[:, OFF_ROOT_VLIN:OFF_ROOT_VLIN + 3],
                               v[:, :3] / ROOT_VLIN_NORM,
                               rtol=1e-6, atol=1e-9)
    assert torch.count_nonzero(obs[:, OFF_ROOT_VANG:OFF_ROOT_VANG + 3]) == 0
    lin = torch.stack([kx, torch.zeros_like(kx), kz], -1)  # (M, 3)
    for base in (OFF_HAND_L, OFF_HAND_R):
        torch.testing.assert_close(obs[:, base + 3:base + 6], lin / V_NORM,
                                   rtol=1e-6, atol=tol)
    for i in range(N_BODY_PTS):
        b = OFF_BODY_PT_VEL + 3 * i
        torch.testing.assert_close(obs[:, b:b + 3], lin / V_NORM,
                                   rtol=1e-6, atol=tol)
    torch.testing.assert_close(env.wrist_vel[ids].double(),
                               lin[:, None].expand(-1, 2, -1),
                               rtol=0, atol=0)


@cuda
@pytest.mark.parametrize("dtype", ["float64", "float32"])
def test_auto_reset_obs_shows_start_kick(dtype):
    import torch
    env = _env(dtype, start_jitter=KICK)
    env.reset()
    gen = torch.Generator(device="cuda").manual_seed(1)
    n_reset = 0
    for _ in range(60):
        acts = torch.randint(5, (env.num_envs, N_JOINTS), device="cuda",
                             generator=gen)
        env.step(acts)
        ids = env._done.nonzero().squeeze(1)
        if ids.numel() == 0:
            continue
        n_reset += ids.numel()
        # _start_kick holds the draws of THIS launch (refilled first)
        kx = env._start_kick[ids, 0].double()
        kz = env._start_kick[ids, 1].double()
        mag = torch.sqrt(kx * kx + kz * kz)
        assert (mag <= KICK * (1 + 1e-6)).all()
        _vel_lane_checks(env, ids, kx, kz,
                         tol=1e-12 if dtype == "float64" else 1e-7)
        # fresh episodes really start at tick 0 on this very obs
        assert torch.count_nonzero(env.tick[ids]) == 0
    assert n_reset >= env.num_envs  # max_steps=24: every env reset twice
    # the kicks are real (not all zero) and varied
    assert env._start_kick.abs().max().item() > 0.02


@cuda
def test_explicit_reset_applies_start_kick():
    import torch
    env = _env("float64", start_jitter=KICK)
    env.reset()
    ids = torch.arange(env.num_envs, device="cuda")
    kx = env._start_kick[:, 0].clone()
    kz = env._start_kick[:, 1].clone()
    assert kx.abs().max().item() > 0.02
    _vel_lane_checks(env, ids, kx, kz, tol=1e-12)

    # start_jitter = 0: the old exact-zero reset (parity configs)
    env0 = _env("float64")
    env0.reset()
    assert torch.count_nonzero(env0.v) == 0
    assert torch.count_nonzero(env0._start_kick) == 0
    assert torch.count_nonzero(
        env0._obs[:, OFF_ROOT_VLIN:OFF_ROOT_VLIN + 3]) == 0


@cuda
def test_auto_reset_equals_fresh_reset():
    """An in-step auto-reset and a mode-1 reset() fed the SAME start
    draws (pose jitter + crouch gate/depth, tau body, start kick) and
    chart produce the same start state and the same obs row: nothing
    per-episode is applied after the obs write on either path."""
    import torch
    kw = dict(start_jitter=KICK, start_pose_jitter=0.1,
              crouch_start_prob=0.3, tau_rand=0.2)
    a = _env("float64", **kw)
    b = _env("float64", **kw)
    a.reset()
    gen = torch.Generator(device="cuda").manual_seed(2)
    ids = None
    for _ in range(60):
        acts = torch.randint(5, (a.num_envs, N_JOINTS), device="cuda",
                             generator=gen)
        a.step(acts)
        if int(a._done.sum()) >= 8:
            ids = a._done.nonzero().squeeze(1)
            break
    assert ids is not None, "no auto-resets"
    # replay A's last-launch draws through B's raw mode-1 launch
    b._pose_jit.copy_(a._pose_jit)
    b._tau_jit.copy_(a._tau_jit)
    b._start_kick.copy_(a._start_kick)
    b._ext_step(b._zero_act, a.chart_idx.clone(), 1)
    # reset_env is inlined at two sites (step_env / mode 1): allow an
    # FMA-contraction ulp on the FK-derived fields, demand exactness on
    # the pure copies (velocities, kick, tau)
    for name in ("v", "tau_scale", "wrist_vel", "palm_vel", "_bp_vel",
                 "chart_idx", "tick"):
        assert torch.equal(getattr(a, name)[ids], getattr(b, name)[ids]), name
    for name in ("p", "quat", "q", "wrist", "blade", "head", "_bp_pos"):
        torch.testing.assert_close(getattr(a, name)[ids],
                                   getattr(b, name)[ids],
                                   rtol=0, atol=1e-12, msg=name)
    torch.testing.assert_close(a._obs[ids], b._obs[ids], rtol=0, atol=1e-6)
    # and the draws actually did something on these envs
    assert a._start_kick[ids].abs().max().item() > 0.0
    assert not torch.equal(a.tau_scale[ids],
                           torch.ones_like(a.tau_scale[ids]))
    assert math.isfinite(float(a._obs[ids].abs().max()))
