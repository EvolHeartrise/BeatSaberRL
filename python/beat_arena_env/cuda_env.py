"""CUDA-kernel vectorized Beat Arena sim, env v10 — one kernel launch per
step (P5).

rust/src/lib.rs remains THE reference implementation; this backend runs
the whole v10 step — 8 constrained dynamics substeps (SPEC-v10 §3),
gear->torque actuation, work-based effort, swept-surface judgment,
auto-reset, obs 340 — inside one custom CUDA kernel
(csrc/beat_arena_cuda.cu), one THREAD per env (P5 parallelization
decision: the Rust expression trees transcribe serially per thread, so
every accumulation keeps Rust's exact order; occupancy comes from batch
width — production runs N = 1024..8192).

Parity is pinned by tests/test_v10_parity_cuda.py: the f64 tier is
HORIZON-BOUNDED tight-tolerance (CUDA f64 sin/cos/acos differ from the
CRT libm by 1 ulp on a few %% of inputs and nvcc contracts FMAs — both
measured there; SPEC-v10 Divergences §21), f32 is the bounded
well-formedness tier. Both dtypes carry a Cholesky pivot-floor safety
net with engagements counted (`pop_pd_hits`): f32 floors at 1e-8
(§20e), f64 at 1e-17 (§22 — the same floor Rust/torch f64 now carry).

State is a structure-of-arrays of torch tensors mutated in place by the
kernel; charts live in a padded (P, C) device pool and each env holds
only a pool index, so reset never copies chart data. The step loop
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
from .constants import (
    BAD_HALF, BAD_HALF_Y_LINK, EFFORT_COEF, FOLLOW_TIMEOUT, GOOD_HALF_Y,
    GOOD_HALF_Y_DOT, GOOD_HALF_Y_LINK, KIND_BOMB, KIND_CHAIN_HEAD,
    KIND_CHAIN_LINK, N_JOINTS, NV, OBS_DIM, note_center,
)

NUM_PAIRS = 55  # SPEC-v10 §2 live pair table (torch_dynamics.NUM_PAIRS)

# Follow-through tracker ring (v7 design, see the kernel header):
# same-tick same-saber trackers share (arc, age) forever, so they
# aggregate into ONE slot with a note COUNT — exact, not approximate.
N_TRACKERS = FOLLOW_TIMEOUT + 1

# lib.rs Z_MISS_TAIL: the episode tail covers the slowest object's trip
# from the hit plane to the WORST-CASE live head plane (env v10).
Z_MISS_TAIL = 1.5

_ext = None

# CUDA toolkits with a full nvcc known on this machine / typical installs
_CUDA_CANDIDATES = (
    r"D:\Cuda-12-6",
    r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6",
)


_toolchain_done = False


def _toolchain_env():
    """Make torch.utils.cpp_extension work from a bare venv on Windows:
    locate nvcc (CUDA_HOME), cl.exe (newest VS2022 Hostx64/x64), and ninja
    (venv Scripts) if they are not already reachable. Memoized."""
    global _toolchain_done
    if os.name != "nt" or _toolchain_done:
        return
    _toolchain_done = True
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
        from beat_arena_env.jit_cache import CUDA_FLAGS, fast_load
        src = os.path.join(os.path.dirname(os.path.abspath(__file__)), "csrc")
        bdir = os.path.join(src, "build")
        os.makedirs(bdir, exist_ok=True)
        srcs = [os.path.join(src, "beat_arena_cuda.cu")]

        def _full():
            _toolchain_env()
            os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "6.1")
            from torch.utils.cpp_extension import load
            return load(name="beat_arena_cuda", sources=srcs,
                        build_directory=bdir,
                        extra_cuda_cflags=CUDA_FLAGS, verbose=True)

        _ext = fast_load("beat_arena_cuda", srcs, bdir, CUDA_FLAGS, _full)
    return _ext


def _charts_to_tensors(charts, device, dtype):
    """Pad a list[list[Note]] into per-field (P, C) tensors — env v10:
    episode tail = 0.5 + 1.5/min_speed (live head-plane miss boundary).
    Chart sin/cos are computed by numpy (the CRT libm — bitwise what
    Rust's set_chart_notes computes)."""
    P = len(charts)
    C = max(1, max((len(ch) for ch in charts), default=1))
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
    show_dir = ~is_bomb & (~any_dir | is_link)
    good_hy = np.where(is_link, GOOD_HALF_Y_LINK,
                       np.where(any_dir, GOOD_HALF_Y_DOT, GOOD_HALF_Y))
    bad_hy = np.where(is_link, BAD_HALF_Y_LINK, BAD_HALF)
    rotates = is_link | ~any_dir
    sin_a = np.sin(angle)
    cos_a = np.cos(angle)
    rot_c = np.where(rotates, -sin_a, 1.0)
    rot_s = np.where(rotates, cos_a, 0.0)
    min_speed = np.where(valid, speed, np.inf).min(1)
    finite = np.isfinite(min_speed)
    # the kernel's high-water bound needs a FINITE min_speed; 1.0 for
    # empty charts is inert (no valid notes to scan)
    min_speed = np.where(finite, min_speed, 1.0)
    tail = np.where(finite, 0.5 + Z_MISS_TAIL / min_speed, 0.5)
    t = lambda a, dt=dtype: torch.as_tensor(a, device=device, dtype=dt)
    return {
        "hit": t(hit), "speed": t(speed),
        "center": torch.stack([t(x), t(y)], -1),
        "dvec": torch.stack([t(cos_a), t(sin_a)], -1),
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
    config + seed + pool size + dtype + a hash of chart.py + a pool
    LAYOUT version (v10: the tail rule changed to 0.5 + 1.5/min_speed —
    bumping the key retires every v9-tail pool)."""
    import hashlib
    src = open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "chart.py"), "rb").read()
    blob = repr(("v10", repr(cfg), seed, pool_size, str(dtype))).encode()
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


def _fix_pool_tail(pool):
    """Recompute the v10 episode tail from min_speed IN PLACE — covers
    pools cached before the v10 tail rule (ranked pools are keyed on the
    parser version only). Empty-chart pools keep tail 0.5 via last_hit=0
    + the no-notes done branch, so the blanket rule is safe."""
    ms = pool["min_speed"]
    pool["tail"] = torch.where(
        torch.isfinite(ms) & (ms > 0),
        0.5 + Z_MISS_TAIL / ms, torch.full_like(ms, 0.5))
    return pool


class CudaBeatVecEnv:
    """BeatVecEnv-alike backed by the v10 CUDA kernel. step() takes an
    (N, 29) long gear-id action tensor (SPEC-v10 §1.4 joint order) and
    returns (obs, reward, done, []) — all tensors on the CUDA device,
    infos always empty (use pop_stats() once per update)."""

    def __init__(self, num_envs, colors="both", chart_len_s=20.0,
                 cols=(0, 1, 2, 3), rows=(0, 1, 2),
                 note_speed_min=2.0, note_speed_max=8.0,
                 note_prob=0.25, extra_prob=0.125, dot_prob=0.1,
                 bomb_prob=0.125, seed=0, effort_coef=EFFORT_COEF,
                 charts=None, chart_pool=1024, max_steps=None,
                 pool=None, wide_acc=False, gears=3,
                 device="cuda", dtype=torch.float32, auto_reset=True,
                 want_contacts=False):
        self.num_envs = num_envs
        self.device = torch.device(device)
        if self.device.type != "cuda":
            raise ValueError("CudaBeatVecEnv requires a CUDA device")
        self.dtype = dtype
        self.auto_reset = auto_reset  # False: terminal envs freeze (tests)
        # contacts() introspection (per-pair impulses + points) costs a
        # 55-wide zeroing plus uncoalesced RMWs per active contact row every
        # tick and feeds NOTHING in the kernel (the obs contact bits come
        # from contact_arm, which is always maintained). Off for training;
        # the parity harnesses pass True. Baked here, never mutated =>
        # constant per instance => safe under CUDA-graph capture.
        self.want_contacts = bool(want_contacts)
        self.effort_coef = float(effort_coef)
        self.wide_acc = bool(wide_acc)
        assert gears in (3, 5, 7), f"gears must be 3, 5, or 7, got {gears}"
        self.gears = int(gears)
        # pool=: prebuilt (P, C) tensor dict (ranked_pool.py) — unlike
        # charts=, resets keep the normal RANDOM chart draw (charts= is
        # the deterministic-sequential test mode)
        self._test_charts = charts is not None and pool is None
        if pool is not None:
            self.pool = _fix_pool_tail(
                {k: v.to(self.device, dtype if v.is_floating_point() else
                         v.dtype) for k, v in pool.items()})
        elif charts is None:
            cfg = ChartConfig(
                chart_len_s=chart_len_s, colors=colors,
                cols=tuple(cols), rows=tuple(rows),
                note_speed_min=note_speed_min,
                note_speed_max=note_speed_max,
                note_prob=note_prob, extra_prob=extra_prob,
                dot_prob=dot_prob, bomb_prob=bomb_prob)
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
        # v10 dynamics state
        self.p = z(N, 3); self.quat = z(N, 4)
        self.q = z(N, N_JOINTS); self.v = z(N, NV)
        # FK caches
        self.wrist = z(N, 2, 3); self.blade = z(N, 2, 3)
        self.wrist_vel = z(N, 2, 3)
        self.head = z(N, 3); self.shoulder_z = z(N, 2)
        # swing chain
        self.swing_acc = z(N, 2)
        self.axis = z(N, 2, 3)
        self.axis_valid = z(N, 2, dt=torch.bool)
        # follow-through tracker ring (aggregated)
        self.tr_acc = z(N, 2, N_TRACKERS)
        self.tr_age = z(N, 2, N_TRACKERS, dt=torch.int32)
        self.tr_cnt = z(N, 2, N_TRACKERS, dt=torch.int32)
        self.tr_active = z(N, 2, N_TRACKERS, dt=torch.bool)
        self.tr_ptr = z(N, 2, dt=torch.int32)
        self.tick = z(N, dt=torch.int32)
        # 0 pending, 1 good, 2 bad/bomb-hit, 3 missed, 4 bomb passed clean
        self.status = z(N, C, dt=torch.uint8)
        self.first_pending = z(N, dt=torch.int32)
        self.ep_return = z(N)
        self.good = z(N, dt=torch.int32); self.bad = z(N, dt=torch.int32)
        self.miss = z(N, dt=torch.int32); self.bombs = z(N, dt=torch.int32)
        self.score = z(N)
        self.score_pre = z(N)
        self.score_acc = z(N)
        self.score_post = z(N)
        # contact aggregation + effort (obs bits / parity introspection)
        self.contact_imp = z(N, NUM_PAIRS)
        self.contact_pt = z(N, NUM_PAIRS, 3)
        self.contact_arm = z(N, 2, dt=torch.bool)
        self.effort_j = z(N)
        # Cholesky pivot-floor engagements (§20e f32 / §22 f64)
        self._pd_hits = z(1, dt=torch.int32)
        self._pd_popped = 0
        self.chart_idx = z(N, dt=torch.int64)
        # episode-return ring buffer (device-side)
        self._cap = 4096
        self._ret_buf = z(self._cap)
        self._ret_total = z(1, dt=torch.int32)
        self._popped = 0
        self._obs = z(N, OBS_DIM, dt=torch.float32)
        self._reward = z(N)
        self._done = z(N, dt=torch.bool)
        self._zero_act = z(N, N_JOINTS, dt=torch.int64)
        self._zero_cand = z(N, dt=torch.int64)
        p = self.pool
        # order is a contract with the kernel's tensor-list enum
        self._tensors = [
            self.p, self.quat, self.q, self.v,
            self.wrist, self.blade, self.wrist_vel, self.head,
            self.shoulder_z,
            self.swing_acc, self.axis, self.axis_valid,
            self.tr_acc, self.tr_age, self.tr_cnt, self.tr_active,
            self.tr_ptr,
            self.tick, self.status, self.first_pending, self.ep_return,
            self.good, self.bad, self.miss, self.bombs, self.score,
            self.score_pre, self.score_acc, self.score_post,
            self.contact_imp, self.contact_pt, self.contact_arm,
            self.effort_j, self._pd_hits, self.chart_idx,
            p["hit"], p["speed"], p["center"], p["dvec"], p["rot"],
            p["good_hy"], p["bad_hy"],
            p["valid"], p["is_right"], p["any_dir"], p["is_bomb"],
            p["is_head"], p["is_link"], p["show_dir"], p["free_pre"],
            p["free_post"],
            p["n_notes"], p["last_hit"], p["tail"], p["min_speed"],
            self._ret_buf, self._ret_total, self._obs, self._reward,
            self._done,
        ]

    def _launch(self, actions, cand, mode):
        self._ext.step(self._tensors, actions, cand, mode,
                       self.max_steps, self.effort_coef, self.wide_acc,
                       self.auto_reset, self.gears, self.want_contacts)

    def _cand(self):
        """Fresh chart index per env (used only for envs that reset)."""
        N = self.num_envs
        if self._test_charts:  # deterministic: env i -> pool[(i + ep*N) % P]
            return (torch.arange(N, device=self.device)
                    + self._episode * N) % self.pool_size
        return torch.randint(self.pool_size, (N,), device=self.device,
                             generator=self._gen)

    def reset(self):
        self._launch(self._zero_act, self._cand(), 1)
        self._episode += 1
        return self._obs

    def step(self, actions):
        if not torch.is_tensor(actions):
            actions = torch.as_tensor(np.asarray(actions))
        actions = actions.to(self.device).long().reshape(
            self.num_envs, N_JOINTS).contiguous()
        cand = self._cand() if self.auto_reset else self._zero_cand
        self._launch(actions, cand, 0)
        if self.auto_reset:
            self._episode += 1
        return self._obs, self._reward, self._done, []

    # -- parity / tooling hooks (tests/test_v10_parity_cuda.py) ---------------
    def set_state(self, q36, v35):
        """RawBeatSim.set_joints mirror: q36 = root pos + quat wxyz + 29
        joints (quat normalized with the same expression), v35 full
        velocity. Broadcasts a single state or takes (N, ·) batches;
        FK caches refresh in-kernel; finite-diff wrist velocity resets."""
        q36 = torch.as_tensor(q36, device=self.device, dtype=self.dtype)
        v35 = torch.as_tensor(v35, device=self.device, dtype=self.dtype)
        if q36.dim() == 1:
            q36 = q36.unsqueeze(0).expand(self.num_envs, -1)
        if v35.dim() == 1:
            v35 = v35.unsqueeze(0).expand(self.num_envs, -1)
        self.p.copy_(q36[:, :3])
        qn = torch.sqrt((q36[:, 3:7] * q36[:, 3:7]).sum(-1, keepdim=True))
        self.quat.copy_(q36[:, 3:7] / qn)
        self.q.copy_(q36[:, 7:])
        self.v.copy_(v35)
        self._launch(self._zero_act, self._zero_cand, 2)

    def qpos(self):
        return torch.cat([self.p, self.quat, self.q], -1)

    def twist(self):
        return self.v[:, :6]

    def omega(self):
        return self.v[:, 6:]

    def sabers(self):
        return torch.cat([self.wrist, self.blade], -1)  # (N, 2, 6)

    def counters(self):
        return (self.good, self.bad, self.miss,
                self.pool["n_scoreable"][self.chart_idx], self.score,
                self.bombs)

    def components(self):
        return (self.score_pre, self.score_acc, self.score_post)

    def last_effort(self):
        return self.effort_j

    def contacts(self):
        """(imp (N,55) tick-summed lambda, point (N,55,3) last substep).
        Requires want_contacts=True at construction — the kernel skips
        these writes otherwise (they feed nothing; obs uses contact_arm)."""
        if not self.want_contacts:
            raise RuntimeError(
                "contacts() needs CudaBeatVecEnv(..., want_contacts=True); "
                "this env was built with it off, so contact_imp/contact_pt "
                "are stale (never written by the kernel)")
        return self.contact_imp, self.contact_pt

    def pop_pd_hits(self):
        """Cholesky pivot-floor engagements since the last pop
        (Divergences §20e f32 / §22 f64 — reported, not hidden).
        One sync."""
        total = int(self._pd_hits.item())
        new = total - self._pd_popped
        self._pd_popped = total
        return new

    def pop_stats(self):
        """Episode returns finished since the last pop (order approximate
        under ring overflow; logging-grade, two syncs per call — the
        counter .item() and the ring-read .tolist())."""
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
