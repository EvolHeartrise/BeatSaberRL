"""CUDA-kernel vectorized Beat Arena sim — one kernel launch per step.

rust/src/lib.rs remains THE reference implementation; this backend runs
the whole step (joint integration, swept-surface judgment, auto-reset,
obs build) inside one custom CUDA kernel (csrc/beat_arena_cuda.cu) for
rollout throughput. Parity is pinned by tests/test_cuda_env.py: f64 exact
vs Rust on identical charts + actions, f32 bounded drift. (The
intermediate tensor-op port, torch_env.py, was REMOVED 2026-07-25 —
surface-area reduction once this kernel became the production backend;
its chart tensorization and aggregated tracker-ring design live on
here.)

State is a structure-of-arrays of torch tensors mutated in place by the
kernel; charts live in a padded (P, C) device pool (built by
_charts_to_tensors below — single source for chart tensorization) and
each env holds only a pool index, so reset never copies chart data. The
step loop performs ZERO host<->device syncs; episode returns go to a
device ring buffer drained by pop_stats() once per PPO update.

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
from .constants import (
    BAD_HALF, BAD_HALF_Y_LINK, FOLLOW_TIMEOUT, GOOD_HALF_Y,
    GOOD_HALF_Y_DOT, GOOD_HALF_Y_LINK, KIND_BOMB, KIND_CHAIN_HEAD,
    KIND_CHAIN_LINK, N_JOINTS, OBS_DIM, Z_MISS, note_center,
)

# Follow-through tracker ring: trackers spawned by the same saber in the
# same tick share identical (arc, age) forever, so they aggregate into
# ONE slot with a note COUNT. A surviving tracker lives <= FOLLOW_TIMEOUT
# ticks and the ring has FOLLOW_TIMEOUT + 1 slots, so slot reuse is
# provably safe: the ring is EXACT, not an approximation.
N_TRACKERS = FOLLOW_TIMEOUT + 1

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


def _charts_to_tensors(charts, device, dtype):
    """Pad a list[list[Note]] into per-field (P, C) tensors (env v7:
    explicit world x/y via note_center, object kinds, arc grants, per-note
    hitbox half-heights, per-chart episode tail)."""
    P = len(charts)
    C = max(1, max(len(ch) for ch in charts))
    hit = np.zeros((P, C)); x = np.zeros((P, C)); y = np.zeros((P, C))
    color = np.zeros((P, C)); angle = np.zeros((P, C)); speed = np.ones((P, C))
    any_dir = np.zeros((P, C), dtype=bool); kind = np.zeros((P, C), np.int64)
    fpre = np.zeros((P, C), dtype=bool); fpost = np.zeros((P, C), dtype=bool)
    valid = np.zeros((P, C), dtype=bool)
    for p, ch in enumerate(charts):
        for j, n in enumerate(ch):
            hit[p, j] = n.hit_time
            x[p, j], y[p, j] = note_center(n)
            color[p, j] = n.color; angle[p, j] = n.angle; speed[p, j] = n.speed
            any_dir[p, j] = n.any_dir; kind[p, j] = n.kind
            fpre[p, j] = n.free_pre; fpost[p, j] = n.free_post
            valid[p, j] = True
    is_bomb = valid & (kind == KIND_BOMB)
    is_head = valid & (kind == KIND_CHAIN_HEAD)
    is_link = valid & (kind == KIND_CHAIN_LINK)
    # dot notes encode obs direction as the ZERO vector; chain links carry
    # their tangent despite judging any-direction (lib.rs build_obs)
    show_dir = ~is_bomb & (~any_dir | is_link)
    good_hy = np.where(is_link, GOOD_HALF_Y_LINK,
                       np.where(any_dir, GOOD_HALF_Y_DOT, GOOD_HALF_Y))
    bad_hy = np.where(is_link, BAD_HALF_Y_LINK, BAD_HALF)
    # env v7.4: hitboxes rotate about z by rot = angle - 270 deg —
    # (cos_r, sin_r) = (-sin a, cos a), from the angle's own sin/cos
    # (lib.rs note_rot). Arrow notes/heads and chain links (tangent)
    # rotate; plain dots stay axis-aligned; bombs are spheres (unused).
    rotates = is_link | ~any_dir
    rot_c = np.where(rotates, -np.sin(angle), 1.0)
    rot_s = np.where(rotates, np.cos(angle), 0.0)
    # episode tail: 0.5 + Z_MISS / min speed (lib.rs set_chart_notes);
    # min_speed is also exported per chart — the kernel's high-water
    # bound ((hit - t) * min_speed > REACH_Z => note not yet reachable)
    # needs it, mirroring the Rust-side sorted-hit_time break
    min_speed = np.where(valid, speed, np.inf).min(1)
    finite = np.isfinite(min_speed)
    min_speed = np.where(finite, min_speed, 1.0)
    tail = np.where(finite, 0.5 + Z_MISS / min_speed, 0.5)
    t = lambda a, dt=dtype: torch.as_tensor(a, device=device, dtype=dt)
    return {
        "hit": t(hit), "speed": t(speed),
        "center": torch.stack([t(x), t(y)], -1),
        "dvec": torch.stack([t(np.cos(angle)), t(np.sin(angle))], -1),
        "rot": torch.stack([t(rot_c), t(rot_s)], -1),
        "is_right": t(color, torch.bool),
        "any_dir": t(any_dir, torch.bool), "valid": t(valid, torch.bool),
        "is_bomb": t(is_bomb, torch.bool), "is_head": t(is_head, torch.bool),
        "is_link": t(is_link, torch.bool),
        "show_dir": t(show_dir, torch.bool),
        "free_pre": t(fpre, torch.bool), "free_post": t(fpost, torch.bool),
        "good_hy": t(good_hy), "bad_hy": t(bad_hy),
        "n_notes": t(valid.sum(1), torch.long),
        "n_scoreable": t((valid & ~is_bomb).sum(1), torch.long),
        "last_hit": t(np.where(valid.any(1), (hit * valid).max(1), 0.0)),
        "tail": t(tail),
        "min_speed": t(min_speed),
    }


def _pool_cache_path(cfg, seed, pool_size, dtype):
    """Cache file for a procedural chart pool. Keyed on the FULL chart
    config + seed + pool size + dtype + a hash of chart.py (the
    generator algorithm) — any generator edit invalidates every cached
    pool, mirroring the bench's parser_version() contract."""
    import hashlib
    src = open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "chart.py"), "rb").read()
    blob = repr((repr(cfg), seed, pool_size, str(dtype))).encode()
    key = hashlib.sha1(src + blob).hexdigest()[:16]
    cdir = os.path.join(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))), "pool_cache")
    os.makedirs(cdir, exist_ok=True)
    return os.path.join(cdir, f"pool_{pool_size}_{key}.npz")


def _load_pool_cache(path, device):
    if not os.path.exists(path):
        return None
    data = np.load(path)
    return {k: torch.as_tensor(data[k], device=device) for k in data.files}


class CudaBeatVecEnv:
    """BeatVecEnv-alike backed by the CUDA kernel. step() takes an
    (N, 12) long action tensor (per hand: pitch, yaw, twist, elbow, roll,
    bend; left first) and returns (obs, reward, done, []) — all tensors on
    the CUDA device, infos always empty (use pop_stats() once per update).
    """

    def __init__(self, num_envs, colors="both", chart_len_s=20.0,
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
            # pool cache (2026-07-26): serialized (P, C) tensors, keyed
            # on config+seed+size+generator hash — chart gen is serial
            # Python at ~1.5 s per 1024 charts, so a 65536 pool costs
            # ~96 s fresh but ~1 s from cache. Bit-identical round trip
            # (validated: cached load == fresh gen, torch.equal).
            cpath = _pool_cache_path(cfg, seed, chart_pool, dtype)
            self.pool = _load_pool_cache(cpath, self.device)
            if self.pool is None:
                rng = np.random.default_rng(seed)
                charts = [generate_chart(cfg, rng)
                          for _ in range(chart_pool)]
                self.pool = _charts_to_tensors(charts, self.device, dtype)
                np.savez(cpath, **{k: v.cpu().numpy()
                                   for k, v in self.pool.items()})
                print(f"[pool-cache] generated {chart_pool} charts -> "
                      f"{os.path.basename(cpath)}", flush=True)
            else:
                print(f"[pool-cache] hit: {os.path.basename(cpath)}",
                      flush=True)
        else:
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
        # low-water cursor: smallest pending chart index per env (the
        # kernel's chart scans start here instead of 0)
        self.first_pending = z(N, dt=torch.int32)
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
            self.tr_ptr, self.tick, self.status, self.first_pending,
            self.ep_return,
            self.good, self.bad, self.miss, self.bombs, self.score,
            self.score_pre, self.score_acc, self.score_post,
            self.chart_idx,
            p["hit"], p["speed"], p["center"], p["dvec"], p["rot"],
            p["good_hy"], p["bad_hy"],
            p["valid"], p["is_right"], p["any_dir"], p["is_bomb"],
            p["is_head"], p["is_link"], p["show_dir"], p["free_pre"],
            p["free_post"],
            p["n_notes"], p["last_hit"], p["tail"], p["min_speed"],
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
