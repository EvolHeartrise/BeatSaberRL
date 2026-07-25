"""CUDA-kernel vectorized Beat Arena sim — one kernel launch per step.

The THIRD implementation of env v7.9 (user's explicit call, reversing the
single-implementation policy): rust/src/lib.rs remains THE reference,
torch_env.py the tensor-form port, and this backend runs the whole step
(joint integration, swept-surface judgment, auto-reset, obs build) inside
one custom CUDA kernel (csrc/beat_arena_cuda.cu) for rollout throughput.
Parity is pinned by tests/test_cuda_env.py: f64 exact vs Rust on identical
charts + actions, f32 bounded drift — the same two tiers as the torch env.

State is a structure-of-arrays of torch tensors mutated in place by the
kernel; charts live in a padded (P, C) device pool (built by torch_env's
_charts_to_tensors — single source for chart tensorization) and each env
holds only a pool index, so reset never copies chart data. The step loop
performs ZERO host<->device syncs; episode returns go to a device ring
buffer drained by pop_stats() once per PPO update.

The extension is built on first use via torch.utils.cpp_extension.load
(ninja-cached under csrc/build; -arch=sm_61). Windows toolchain quirks
(CUDA_HOME, cl.exe, ninja discovery) are handled in _toolchain_env().

Limits vs BeatVecEnv: no per-step info dicts, no event stream, CUDA only.
"""

from __future__ import annotations

import glob
import os
import shutil
import sys

import numpy as np
import torch

from .chart import ChartConfig, generate_chart
from .constants import FOLLOW_TIMEOUT, N_JOINTS, OBS_DIM
from .torch_env import _charts_to_tensors

N_TRACKERS = FOLLOW_TIMEOUT + 1  # aggregated tracker ring (see torch_env)

_ext = None

# CUDA toolkits with a full nvcc known on this machine / typical installs
_CUDA_CANDIDATES = (
    r"D:\Cuda-12-6",
    r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6",
)


def _toolchain_env():
    """Make torch.utils.cpp_extension work from a bare venv on Windows:
    locate nvcc (CUDA_HOME), cl.exe (newest VS2022 Hostx64/x64), and ninja
    (venv Scripts) if they are not already reachable."""
    if os.name != "nt":
        return
    scripts = os.path.dirname(sys.executable)
    if shutil.which("ninja") is None:
        os.environ["PATH"] = scripts + os.pathsep + os.environ["PATH"]
    if shutil.which("cl") is None:
        pats = (r"C:\Program Files\Microsoft Visual Studio\2022\*"
                r"\VC\Tools\MSVC\*\bin\Hostx64\x64",
                r"C:\Program Files (x86)\Microsoft Visual Studio\2019\*"
                r"\VC\Tools\MSVC\*\bin\Hostx64\x64")
        hits = sorted(h for p in pats for h in glob.glob(p))
        if hits:
            os.environ["PATH"] = hits[-1] + os.pathsep + os.environ["PATH"]
    ch = os.environ.get("CUDA_HOME")
    if not (ch and os.path.exists(os.path.join(ch, "bin", "nvcc.exe"))):
        for cand in _CUDA_CANDIDATES:
            if os.path.exists(os.path.join(cand, "bin", "nvcc.exe")):
                os.environ["CUDA_HOME"] = ch = cand
                break
    if ch:
        os.environ["PATH"] = (os.path.join(ch, "bin") + os.pathsep
                              + os.environ["PATH"])


def _load_ext():
    """Compile-or-load the cached extension (ninja rebuilds only on source
    change; a warm cache loads in ~1 s)."""
    global _ext
    if _ext is None:
        _toolchain_env()
        os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "6.1")
        from torch.utils.cpp_extension import load
        src = os.path.join(os.path.dirname(os.path.abspath(__file__)), "csrc")
        bdir = os.path.join(src, "build")
        os.makedirs(bdir, exist_ok=True)
        _ext = load(name="beat_arena_cuda",
                    sources=[os.path.join(src, "beat_arena_cuda.cu")],
                    build_directory=bdir,
                    extra_cuda_cflags=["-arch=sm_61"])
    return _ext


class CudaBeatVecEnv:
    """TorchBeatVecEnv-alike backed by the CUDA kernel. step() takes an
    (N, 12) long action tensor (per hand: pitch, yaw, twist, elbow, roll,
    bend; left first) and returns (obs, reward, done, []) — all tensors on
    the CUDA device, infos always empty (use pop_stats() once per update).
    """

    def __init__(self, num_envs, colors="right", chart_len_s=20.0,
                 cols=(0, 1, 2, 3), rows=(0, 1, 2),
                 note_speed_min=2.0, note_speed_max=8.0,
                 note_prob=0.25, extra_prob=0.125, dot_prob=0.1,
                 bomb_prob=0.125, seed=0, effort_coef=1.0,
                 charts=None, chart_pool=1024, max_steps=None,
                 device="cuda", dtype=torch.float32, auto_reset=True):
        self.num_envs = num_envs
        self.device = torch.device(device)
        if self.device.type != "cuda":
            raise ValueError("CudaBeatVecEnv requires a CUDA device")
        self.dtype = dtype
        self.auto_reset = auto_reset  # False: terminal envs freeze (tests)
        self.effort_coef = float(effort_coef)
        self._test_charts = charts is not None
        if charts is None:
            cfg = ChartConfig(
                chart_len_s=chart_len_s, colors=colors,
                cols=tuple(cols), rows=tuple(rows),
                note_speed_min=note_speed_min,
                note_speed_max=note_speed_max,
                note_prob=note_prob, extra_prob=extra_prob,
                dot_prob=dot_prob, bomb_prob=bomb_prob)
            rng = np.random.default_rng(seed)
            charts = [generate_chart(cfg, rng) for _ in range(chart_pool)]
        self.pool = _charts_to_tensors(charts, self.device, dtype)
        self.pool_size, self.C = self.pool["hit"].shape
        if max_steps is None:
            max_steps = int(chart_len_s * 30) + 90
        self.max_steps = int(max_steps)
        self._gen = torch.Generator(device=self.device)
        self._gen.manual_seed(seed)
        self._episode = 0  # test-mode sequential chart assignment counter
        self._ext = _load_ext()

        N, C = num_envs, self.C
        dv = self.device
        z = lambda *s, dt=dtype: torch.zeros(*s, device=dv, dtype=dt)
        # arm chains + FK caches
        self.q = z(N, 2, N_JOINTS); self.om = z(N, 2, N_JOINTS)
        self.wrist = z(N, 2, 3); self.blade = z(N, 2, 3)
        self.wrist_vel = z(N, 2, 3)
        # swing chain
        self.swing_acc = z(N, 2)
        self.axis = z(N, 2, 3)
        self.axis_valid = z(N, 2, dt=torch.bool)
        # follow-through tracker ring (aggregated: `cnt` notes per slot)
        self.tr_acc = z(N, 2, N_TRACKERS)
        self.tr_age = z(N, 2, N_TRACKERS, dt=torch.int32)
        self.tr_cnt = z(N, 2, N_TRACKERS, dt=torch.int32)
        self.tr_active = z(N, 2, N_TRACKERS, dt=torch.bool)
        self.tr_ptr = z(N, 2, dt=torch.int32)
        self.tick = z(N, dt=torch.int32)
        # 0 pending, 1 good, 2 bad/bomb-hit, 3 missed, 4 bomb passed clean
        self.status = z(N, C, dt=torch.uint8)
        self.ep_return = z(N)
        self.good = z(N, dt=torch.int32); self.bad = z(N, dt=torch.int32)
        self.miss = z(N, dt=torch.int32); self.bombs = z(N, dt=torch.int32)
        self.score = z(N)
        # score decomposition (2026-07-25): pre/acc/post point totals
        self.score_pre = z(N)
        self.score_acc = z(N)
        self.score_post = z(N)
        self.chart_idx = z(N, dt=torch.int64)
        # episode-return ring buffer (device-side)
        self._cap = 4096
        self._ret_buf = z(self._cap)
        self._ret_total = z(1, dt=torch.int32)
        self._popped = 0
        self._obs = z(N, OBS_DIM, dt=torch.float32)
        self._reward = z(N)
        self._done = z(N, dt=torch.bool)
        self._zero_act = z(N, 12, dt=torch.int64)
        self._zero_cand = z(N, dt=torch.int64)
        p = self.pool
        # order is a contract with the kernel's tensor-list enum
        self._tensors = [
            self.q, self.om, self.wrist, self.blade, self.wrist_vel,
            self.swing_acc, self.axis, self.axis_valid,
            self.tr_acc, self.tr_age, self.tr_cnt, self.tr_active,
            self.tr_ptr, self.tick, self.status, self.ep_return,
            self.good, self.bad, self.miss, self.bombs, self.score,
            self.score_pre, self.score_acc, self.score_post,
            self.chart_idx,
            p["hit"], p["speed"], p["center"], p["dvec"], p["rot"],
            p["good_hy"], p["bad_hy"],
            p["valid"], p["is_right"], p["any_dir"], p["is_bomb"],
            p["is_head"], p["is_link"], p["show_dir"], p["free_pre"],
            p["free_post"],
            p["n_notes"], p["last_hit"], p["tail"],
            self._ret_buf, self._ret_total, self._obs, self._reward,
            self._done,
        ]

    def _cand(self):
        """Fresh chart index per env (used only for envs that reset)."""
        N = self.num_envs
        if self._test_charts:  # deterministic: env i -> pool[(i + ep*N) % P]
            return (torch.arange(N, device=self.device)
                    + self._episode * N) % self.pool_size
        return torch.randint(self.pool_size, (N,), device=self.device,
                             generator=self._gen)

    def reset(self):
        self._ext.step(self._tensors, self._zero_act, self._cand(), 1,
                       self.max_steps, self.effort_coef, True)
        self._episode += 1
        return self._obs

    def step(self, actions):
        if not torch.is_tensor(actions):
            actions = torch.as_tensor(np.asarray(actions))
        actions = actions.to(self.device).long().reshape(
            self.num_envs, 2 * N_JOINTS).contiguous()
        cand = self._cand() if self.auto_reset else self._zero_cand
        self._ext.step(self._tensors, actions, cand, 0,
                       self.max_steps, self.effort_coef, self.auto_reset)
        if self.auto_reset:
            self._episode += 1
        return self._obs, self._reward, self._done, []

    def pop_stats(self):
        """Episode returns finished since the last pop (order approximate
        under ring overflow; logging-grade, one sync per call)."""
        total = int(self._ret_total.item())
        new = total - self._popped
        if new <= 0:
            return []
        take = min(new, self._cap)
        idx = (torch.arange(total - take, total, device=self.device)
               % self._cap)
        vals = self._ret_buf[idx].tolist()
        self._popped = total
        return vals
