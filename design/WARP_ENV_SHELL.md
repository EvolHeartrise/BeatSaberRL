# Warp-per-env rewrite — SHELL AND INTEGRATION design

Scope: everything around the solver math — launch config, per-env storage, the
peripheral phases, dual-kernel coexistence, validation, staging, risks. A
separate agent designs CRBA / Cholesky / solves / PGS lane parallelism; §7 is
the contract it must meet.

Paths are relative to `projects/beat-arena/`. Bare `:NNNN` citations are
`python/beat_arena_env/csrc/beat_arena_cuda.cu` (5,338 lines, 2026-08-15).
**Design only — no source file is modified by this document.**

---

## 0. Starting point (measured, not assumed)

| fact | source |
|---|---|
| `constexpr int THREADS = 32;` — one warp/block, **one thread per env** | `:928` |
| `blocks = (N + THREADS - 1) / THREADS`, 1-D grid | `:5285` |
| `__global__ __launch_bounds__(THREADS,1) step_kernel(Sim<T>)`, `e = blockIdx.x*blockDim.x + threadIdx.x` | `:5137-5139` |
| The `, 1` is a **register pin at 255**, not an occupancy hint | `:915-926` |
| mode branch (1 reset / 2 refresh / else step), then `write_obs` in all modes | `:5145-5161` |
| `step_kernel` = 90% of rollout ≈ **64% of total wall** at N=16384 | `PERF_AUDIT_2026-08-16.md:16-18` |
| 254 regs (write) / 168 (read-only), 42 KB frame, **zero spills**, sm_89 | audit `:19-21` |
| **4 warps/SM** realized at N=16384; register ceiling 8 | audit `:24, 79-81` |
| Per-env cost **falls** 13.7 → 5.9 us/env, N=4k → 32k ⇒ latency-bound, parallelism-starved | audit `:77-81` |
| Realized `nc <= 7` production / `8` at 4x stress; 92.7% of samples `nc=0` | audit `:82-85` |
| `NC_CAP=16`: frame 42,224 → 30,288 B but **+3.1% SLOWER** (ptxas 254 → 168) | audit `:86-91` |
| ncu locked (`ERR_NVGPUCTRPERM`) ⇒ measure with torch-profiler + A/B, not counters | audit `:22-25` |
| Warp-coherence env permutation measured **NULL** on sm_89 | audit `:44-52` |

Two facts drive everything below.

**(a) The grid is the binder, not the footprint.** 512 blocks / 128 SMs = 4
warps/SM; the register ceiling permits 8; residency probes are flat across
three orders of magnitude of pool size (audit `:59-66`). The lever is *more
resident warps with a shorter serial chain per env*, not fewer bytes.

**(b) sm_61-era verdicts did not transfer, and neither will these.** The
provenance warning at `:97-107` is the house rule. Every number here not in the
table above is a **prediction**, and every stage in §6 carries the measurement
that falsifies it.

### Per-env frame accounting (f32, from `Dyn` `:1554-1600`, `Solver` `:1680-1712`, `WeldRows` `:1613-1657`, `GroundRows` `:1667-1676`)

| block | bytes |
|---|---:|
| `Dyn`: p/quat/q/v (284), r/pos/axw (1,800), r0 (36), m/c/rhs (5,180), CRBA composites (1,612), iw (1,116), RNEA (1,800) | 11,836 |
| `Solver` base: `c_J[55][35]`+`c_minv[55][35]` (15,400), c_bias/diag/lam/pt (1,320), `ecol[29][35]` (4,060), + tables | 21,791 |
| `WeldRows` (FC=0): `mw` (1,680), jwc/jwr/bias_w (384), aww/rw (624), `pcol[29][35]` (4,060) | 6,748 |
| `GroundRows` (FC=1): `g_J`+`g_minv` (6,720), bias/diag/lam (288) | 7,016 |
| **contact mode (production)** | **~40.6 KB** |

Reconciles with the measured 42,224 B once padding and temporaries are added.
`c_J`+`c_minv` alone are **15.4 KB — 38% of the frame — for a realized
`nc <= 8`**: 85% of that block is never written. §2.4 says what to do, and why
not yet.

---

## 1. Launch configuration

### 1.1 Mapping

```
warp = threadIdx.x >> 5;   lane = threadIdx.x & 31;
e    = blockIdx.x * ENVS_PER_BLOCK + warp;    // WARP-UNIFORM
if (e >= P.N) return;                          // whole warp returns together
```

`e` warp-uniform is load-bearing three times: the guard never splits a warp;
every per-env branch (`P.mode`, `nc`, `nl`, `ng`, `done`) becomes warp-uniform;
`__syncwarp` suffices for all intra-env communication.

**Hard invariant: zero `__syncthreads()` in the warp kernel.** Envs in a block
never communicate; shared memory is partitioned per warp. Not style — with a
uniform early return some warps in the last block exit while others continue,
so any block barrier deadlocks. Enforcement in §8.2.

### 1.2 sm_89 (AD102) envelope and the config ladder

128 SMs · 65,536 regs/SM · 48 warps/SM · 100 KB shared/SM (128 KB unified
L1+shared) · **48 KB static shared per block** · 24 blocks/SM.
`regs = 65536 / (threads_per_block * min_blocks_per_sm)`.

| config | thr/blk | envs/blk | `__launch_bounds__` | regs | blk/SM | **warps/SM** | smem/env |
|---|---:|---:|---|---:|---:|---:|---:|
| today | 32 | 32 | `(32, 1)` | 255 | 4 (grid-limited) | **4** | 0 |
| **W1 (start)** | 128 | 4 | `(128, 4)` | 128 | 4 | **16** | 6.2 KB |
| W2 | 256 | 8 | `(256, 2)` | 128 | 2 | 16 | 6.2 KB |
| W3 | 128 | 4 | `(128, 6)` | 85 | 6 | 24 | 4.1 KB |
| W4 | 128 | 4 | `(128, 8)` | 64 | 8 | 32 | 3.1 KB |

Grid supply stops being a constraint: at N=16384 with 4 envs/block the grid is
4,096 blocks / 128 SMs = 32 blocks/SM *available* against a residency of 4–8.
That predicts the audit's N-scaling curve (`:77-81`) should **flatten** — equal
us/env at N=16384 and N=32768. Cheap, falsifiable; measure at stage 6.

**Recommendation: W1, `__launch_bounds__(128, 4)`, 4 warps = 4 envs/block,
16 warps/SM.**

- 16 warps/SM is **4x** today's measured 4 and 2x today's *ceiling* of 8.
- 4 × 6,144 B = **24 KB static shared/block**, well under the 48 KB static
  limit ⇒ no `cudaFuncSetAttribute` dance and no dynamic-shared moving part
  inside graph capture (§8.4).
- Small blocks give finer grid granularity and a shorter tail at any N.
- The budget is **PINNED**, per `:915-926` ("left free, ptxas picks the target
  from a heuristic that is unstable under unrelated frame changes"). Under
  `(128,4)` ptxas has no free choice — which is also why the warp kernel should
  be *less* exposed to the 254 → 168 flip that killed the NC_CAP probe. Still
  gate every stage on the census (§5c-4); "less exposed" is a hypothesis.

W3/W4 are the follow-on ladder, reachable only if smem/env drops (§2.2). Probe
at stage 6, never earlier — an occupancy change confounds every prior A/B.

### 1.3 Shared carveout

96 KB shared/SM (W1 at 4 blocks) leaves only 32 KB of L1 out of 128 KB — a real
cost the kernel does not pay today, and where the residual local frame and the
arena (§2.3) live. Call `cudaFuncSetAttribute(step_kernel_warp<float,FC>,
cudaFuncAttributePreferredSharedMemoryCarveout, 100)` **once at extension load,
outside capture**, and A/B the 100-vs-64 KB carveout at stage 6.

---

## 2. Storage strategy

Today's frame is **42 KB per thread**. The same env over 32 lanes is **~1.3 KB
per thread** before padding. That ~30x per-thread reduction buys the register
budget that buys the occupancy. It does *not* reduce aggregate bytes — §2.5.

### 2.1 Three tiers

| tier | contents | why |
|---|---|---|
| **Shared** (per-warp partition) | `d.m[NV*NV]` (mass matrix, then its in-place factor), `s.aww[12][12]`, small warp-uniform control block | read by *every* lane at *runtime-varying* indices, ~7,000x per factorization; guaranteed residency beats an L1 gamble |
| **Per-lane regs/local** | frame-indexed `Dyn` arrays (NF=30 ≤ 32 ⇒ **one frame per lane**), coordinate-distributed q/v/c/rhs/tau | disjoint slices, no cross-lane index |
| **Per-env global arena** | `c_J`, `c_minv`, `ecol`, `pcol`, `g_J`, `g_minv` | too big for shared at useful occupancy; laid out so lane-parallel access is *contiguous*, which per-thread local can never be for this pattern |

### 2.2 Shared budget per env (f32)

`m[NV*NV]` 4,900 · `aww[144]` 576 (weld only) · control (`nc`/`nl`/`ng`/flags/
32-word reduction scratch) ~200 · `c_pair[55]` int 220 · `l_j[29]` int 116 ·
`sel_idx[24]`+`sel_wall[24]` 120 (§3.7 handoff) = **~6,132 B** → round to
`SMEM_WORDS_PER_ENV = 1536` = **6,144 B**.

```cpp
__shared__ T smem[ENVS_PER_BLOCK * SMEM_WORDS_PER_ENV];
T* const sm = smem + warp * SMEM_WORDS_PER_ENV;
```

**Bank conflicts — do NOT pad the stride.** 32 banks × 4 B. Column-major
`m[c*35 + r]`: a column walk (`r = lane`) is consecutive words, conflict-free;
a row walk (`c = lane`) is stride 35 and `gcd(35,32) = 1`, so **also
conflict-free**. Padding to 36 gives `gcd(36,32) = 4` and 4-way conflicts.
NV=35 being odd is a gift; keep it. (Shared-memory analogue of K11/K12,
`:311-370`.)

**Packed-triangle probe (stage 6, solver scope).** `crba` already writes only
the lower triangle + diagonal (`:1885-1892`, K4c), so a packed `NV(NV+1)/2 =
630`-float factor is available: 2,520 B instead of 4,900. *Different economics*
from the rejected NC_CAP frame cut — the audit rejected frame capacity because
it "shrinks bytes the kernel never waits on", whereas shared bytes trade **1:1
against resident warps**, which the kernel demonstrably does wait on. Caveat:
packed addressing puts integer ops on the dependent path, K17's sign-flipping
term (`:520-546`).

### 2.3 Per-env global arena (replaces per-thread local)

Per-thread local is interleaved so that *all 32 threads at the same array
index* coalesce — optimal for thread-per-env, pessimal for warp-per-env where
32 lanes touch *different indices of the same env's array*. So:

```cpp
// new Sim<T> field; a torch tensor of shape (N, ARENA_WORDS) allocated once in
// CudaBeatVecEnv.__init__ => fixed shape => CUDA-graph safe.
T* __restrict__ arena;      // env e at arena + (size_t)e * ARENA_WORDS
```

Layout rule: every NV-indexed row block is stored **NV-minor**, so a warp
reading row `ci` across columns issues one contiguous 128 B transaction (plus a
3-lane tail, §2.5). Same "one base, immediate offsets" property K11 (+7.06%)
and K12 (+3.95%) found to be what actually pays here (`:296-370`).

Stage-0 size (capacity unchanged at 55): `c_J`+`c_minv` 15,400 + `ecol` 4,060 +
`pcol` 4,060 + `g_J`/`g_minv` 6,720 + margin ≈ **31 KB/env** ⇒ 508 MB at
N=16384, **1.0 GB at N=32768**.

Not new memory *in kind* — the driver already reserves 42 KB × resident-threads
of local backing — but newly **explicit and proportional to N rather than to
residency**, so it must be checked against pod VRAM alongside the 861k-param
policy and the ranked pool. **Largest capacity risk in the design**, and what
makes §2.4 mandatory rather than optional at N=32768.

### 2.4 Capacity: `c_J`/`c_minv` at nc <= 8

`Solver::c_J[NUM_PAIRS][NV]` / `c_minv` (`:1690`) are sized 55; measured max is
7 over 39.3M (env,substep) samples, 8 at 4x stress.

**Decision: keep capacity 55 through stage 5; probe at stage 6, gated.**

- The NC_CAP=16 loss (−3.1%, ptxas 254 → 168) is a failure mode of a *free*
  allocation phase under `__launch_bounds__(32,1)`, which a **pinned** 128-reg
  budget forecloses. But "cannot recur in that form" is not evidence *for*
  shrinking, and K4/K5 order-dependence (`:132-152`) says a verdict measured
  under different pressure is not a verdict.
- Fixed capacity through stage 5 makes each stage's A/B measure the **mapping**,
  not a confounded mapping+capacity change.
- At stage 6 the arena makes capacity a **direct linear VRAM cost** (55 → 12
  takes 31 KB/env → ~12 KB/env, 508 MB → 197 MB at N=16384) — a motivation the
  frame version never had.

**Overflow policy for any reduced capacity.** `nc` is bounded by 55 in
principle; a silent clamp is a semantics change. Follow the repo's "report,
don't hide" idiom (`pd_hits`/`env_hits`/`fall_hits`, `:1600-1603, 4296-4300`):
add an `nc_overflow` int counter, `atomicAdd` on clamp (lane 0), pop next to
`pop_pd_hits` (`cuda_env.py:659-666`). Mechanically one enum slot before `RBUF`
(`:5164-5180`), one `_tensors` entry (`cuda_env.py:524-550`), `T_COUNT` moves —
which `TORCH_CHECK(ts.size() == T_COUNT)` (`:5186`) turns into a loud failure.
Adoption requires the counter to read **exactly 0** across the parity suite
plus a 4x-push stress config.

### 2.5 Honest accounting: what gets worse

- **Padding.** NV=35 does not divide 32. A column-distributed NV array gives
  lane `l` indices `l` and `l+32`, so lanes 0–2 hold two and 3–31 hold one —
  but allocation is uniform, so every NV-indexed array pays `64/35 = 1.83x`.
  Per-env aggregate bytes rise even as per-thread bytes fall ~30x. NQ=29 /
  NF=30 fit one round (7% / 9% idle); NUM_PAIRS=55 is 2 rounds at 86%.
- **Lane efficiency.** `35/64 = 55%` NV-wide, `30/32 = 94%` frame-wide,
  `29/32 = 91%` joint-wide. Nothing recovers the NV tail (§9.2).
- **Net claim, a prediction to be falsified.** Trade ~1.8x aggregate bytes and
  ~45% idle lanes on the widest phase for (a) 4x resident warps, (b) a per-env
  chain shortened by the lane-parallel factor on every NV-wide dot and NF-wide
  FK pass, (c) ~8x fewer L1 transactions on the global phases (§3.1). On a
  kernel at ~0.1% FLOP utilisation and explicitly "latency-bound,
  parallelism-starved" (audit `:80-81`) that is the right side of the trade.
  Still a prediction.

---

## 3. Phase-by-phase shell treatment

`SW` below = `__syncwarp(0xffffffff)`.

### 3.1 `load_state` / `store_state` — the honest coalescing analysis

`load_state` `:4375-4381`, `store_state` `:4383-4390` (called `:4757`).
71 words/env: `P.p[e*3+i]`, `P.quat[e*4+i]`, `P.q[e*NQ+j]`, `P.v[e*NV+i]`.

**Across what dimension does lane-parallel coalesce? Across the FEATURE
dimension X within one env's row — never across envs.** State tensors are
`(N, X)` row-per-env, so env `e`'s row is contiguous and lane `l` → feature `l`
makes the access contiguous *within that one row*.

What it is worth. **Today**, at fixed `j`, the 32 lanes read
`P.q[(e0+l)*NQ + j]` at stride 116 B: 32 distinct 32-B sectors per instruction,
1,024 B moved for 128 B used. But over the whole `j` loop those sectors *are*
all consumed, so **DRAM bytes are identical either way**; the cost is 29 load
instructions each generating 32 L1 transactions. **Warp-per-env**: one
instruction, 29 consecutive floats = 4 sectors, 91% lane efficiency. The win is
therefore **L1/LSU transaction count and MLP, not DRAM traffic** — ~8x fewer
transactions for the same bytes, on a kernel whose L1 is about to be squeezed
to 32 KB by the carveout (§1.3). Do not oversell: 71 words against a ~145 us
kernel.

Lane map (both directions), flat over the 71-word vector — lane `l` covers `l`,
`l+32`, `l+64`: `0..2` p, `3..6` quat, `7..35` q, `36..70` v. One `SW` after
`load_state`, none inside. Both **bitwise identical** to legacy — a copy is a
copy regardless of which lane performs it.

### 3.2 `refresh_fk` and the FK chain

`refresh_fk` `:4359-4372` = `fk_all(d)` + 17 scalar cache writes.

**The writes** (`P.wrist`, `P.blade`, `P.head`, `P.shoulder_z`): today each is
a 32-env-strided warp store = 32 sectors; under warp-per-env each is a 1–6-lane
store in one sector (instructions/array 6 → 1). Cleanest win in the kernel,
smallest absolute payoff. Lanes 0–5 `wrist`/`blade`, 6–8 `head`, 9–10
`shoulder_z`; predicated, one `SW` after.

**`fk_all`** (`:1817-1845`) walks `k = 0..NF-1` with `par = F_PARENT[k]`, so
frame `k` depends on its parent — not embarrassingly parallel, but **NF=30 ≤ 32
⇒ exactly one frame per lane**. Add `constexpr int F_DEPTH[NF]` (derived from
`F_PARENT`, `static_assert(F_DEPTH[k] > F_DEPTH[F_PARENT[k]])`) and loop:

```cpp
for (int dpt = 0; dpt <= MAX_DEPTH; ++dpt) {
    if (lane < NF && F_DEPTH[lane] == dpt) { /* verbatim body for k = lane */ }
    __syncwarp();                            // parent results now visible
}
```

Equal-depth frames are independent, so the chain collapses from 30 frame-bodies
to `MAX_DEPTH+1` (~7–8) — a ~4x cut on a pass that runs 8x/tick inside
`substep` (`:3943`) plus once in `refresh_fk`.

**Bitwise**: each body computes the same expression from the same inputs at a
fixed inline site, parents read after a barrier. The one hazard is nvcc
re-contracting FMAs if the body lands in a differently-shaped enclosing
expression — the K10e2 trap at `:4470-4482`, where relocating `quat_to_rot`
moved obs lanes 3/5/8 by 1 ulp. Keep the body textually intact; if 1-ulp lanes
appear, record them as an expected diff, not a bug.

Cross-lane reads: frame `k` reads `d.r[par]`, `d.pos[par]`, `d.p` — 12
`__shfl_sync(mask, x, par)` per frame (9 `r`, 3 `pos`); `par` is a per-lane
runtime value, handled natively. `d.r0` computed redundantly on all lanes.

### 3.3 Collision compaction — deterministic warp ballot

Broad + narrow phase `:3455-3512`. 55 independent pair tests; the serial code
appends in `pid` order (`ci = s.nc; ... s.nc = ci + 1;`, `:3500-3511`) and
**that order is a semantic contract** — `c_pair` feeds PGS row order, the
contact aggregation loop (`:4731`), and the introspection writes.

Capsule endpoints (`ca`/`cb`/`mid`/`hlr`, `:3457-3475`) are per-collider,
NUM_COLLIDERS=12 ⇒ lane `c < 12` computes collider `c`'s 10 values, `SW`, then
broadcast via shared (480 B from the control block's scratch) or recompute
redundantly; measure both. Compaction, two rounds (55 = 32 + 23):

```cpp
int nc = 0;
for (int base = 0; base < NUM_PAIRS; base += 32) {
    const int pid = base + lane;
    bool hit = false;  T gap, nrm[3], pt[3];
    if (pid < NUM_PAIRS) hit = narrow_phase(pid, &gap, nrm, pt);  // verbatim :3476-3499
    const unsigned m    = __ballot_sync(0xffffffffu, hit);
    const int      rank = __popc(m & ((1u << lane) - 1u));        // set bits BELOW me
    if (hit) { const int ci = nc + rank; sm_c_pair[ci] = pid; /* stash gap/nrm/pt */ }
    nc += __popc(m);                                              // warp-uniform
    __syncwarp();
}
```

**Why this is exactly the serial order.** `__ballot_sync` sets bit `l` iff lane
`l`'s predicate is true, so bit index = lane index = `pid - base`;
`__popc(m & ((1<<lane)-1))` counts *lower-pid* hits in this round; rounds run
in ascending `base`. Slot index is therefore a strictly increasing function of
`pid` — the same permutation the serial append produces, with no tie-breaking
and no atomics.

**Determinism**: `__ballot_sync` (full mask) and `__popc` are pure functions of
the predicate vector. **Bitwise vs legacy**: the per-`pid` broad-phase reject
(`:3479-3485`), `seg_seg_closest`, `gap`, `n[]`, `c_pt` are functions of `pid`
alone with no accumulation across pids, so every stored value and the ordering
are bit-identical. **Collision detection is bitwise-preserving under this
rewrite** — the highest-value early stage: real lane parallelism (55 serial
pair tests → 2 rounds) at zero parity cost.

**Row assembly** (`:3503-3508`) is deliberately *not* fused into the
compaction: at `nc <= 8`, 8 divergent lanes each writing 35 floats is worse
than a warp-uniform `for (ci = 0; ci < nc; ++ci)` with the 35 columns over
lanes — and `nc` is now warp-uniform, so that loop has no divergence tax.
Solver-agent scope; the shell hands over `nc`, `c_pair`, `c_pt`, `c_bias`.

**The K13 tax is gone here.** `:600-660` records that a width ladder selected by
a *per-lane* data-dependent count costs the union of its widths (−2.18%),
because 32 envs shared a warp. Under warp-per-env `nc`/`nl`/`ng`/`na` are
warp-uniform, so that class of loss evaporates and the 4/2/1 ladder idiom is
free again. Counterweight: the sm_89 warp-coherence probe measured **null**
(audit `:44-52`) — budget the parallelism win, take divergence removal as a
bonus.

### 3.4 gear → torque, effort accumulation

**Torque** (`:4690-4696`): `tau[j] = gr[a] * TAU_MAX[j]`,
`a = P.actions[e*NQ+j]` (int64). Lane `j < 29` does its own joint: one
coalesced 232 B int64 load, one table lookup, one multiply. `tau` then stays
**joint-distributed in registers** for the whole tick and is handed to the
solver in that layout (§7.7). Bitwise identical.

**Effort** (`:4726-4730`) is the shell's one genuine reduction —
`for j: pw = tau[j]*v[VOFF+j]; if (pw > 0) effort += pw*H_SUB;` over 8
substeps, ascending `j` then ascending substep. Two options:

- **(a) Bitwise-preserving**: lane 0 accumulates 29 values in ascending `j` via
  a sequential `__shfl_sync` chain per substep — 232 shuffle-latencies/tick,
  keeps stage 4's bitwise gate.
- **(b) Reassociated**: each lane keeps its joint's partial across all 8
  substeps (29 independent chains, no cross-lane traffic during substeps), then
  **one** fixed-shape `__shfl_down` butterfly at end of tick — ~8x less shuffle
  traffic and 8x shorter chain, but moves the stage to the tolerance tier.

Build (a) first; A/B (b) at stage 6. Either way the tree shape must be **fixed
in code**: determinism is a hard gate (§5b), so no atomics and no
participation-dependent reduction.

### 3.5 Fall / ground-impulse / pd / envelope aggregation

- **`fell`** (`:4718-4719`): `d.p[1] < FALL_ROOT_Y` on the owning lane, `__shfl`
  broadcast, OR across substeps in a warp-uniform register.
  `atomicAdd(P.fall_hits,1)` (`:4746`) **lane-0 guarded**.
- **`gimp[4]`** (`:4720-4723`): accumulates `s.g_lam[gi*3]` into a bucket from
  `(s.g_bucket >> 2*gi) & 3`. Two lanes can target one bucket, so a parallel
  version needs conflict resolution for no gain at `ng <= 8`. **Serial on lane
  0** — bitwise, trivial, warp-uniform.
- **`pd_bad` / `env_bad` / `env_n`** (`:4744-4745`): set in the solver, which
  leaves them warp-uniform on every lane (§7.8). Both `atomicAdd`s lane-0.
- **`contact_arm`** (`:4737-4739`, stored `:4755-4756`): OR over active
  contacts; lane 0 or a two-bit ballot. Either is bitwise.

### 3.6 `contact_imp` introspection and the `ret_buf` ring

**`contact_imp`** (`:4731-4736`, gated on `want_contacts`, baked per instance
and graph-safe `:4348-4353`): `P.contact_imp[e*NUM_PAIRS+pid] += s.c_lam[ci]`
plus 3 floats of `contact_pt`. The `Sim` comment calls these "uncoalesced
per-contact read-modify-writes". Assign **lane `ci` (`ci < nc`) to its own
`pid`** — pids are distinct by construction, so no race, no atomic, and all
`nc <= 8` addresses fall in one env's 220-B row = 2 sectors instead of today's
32-env scatter. Bitwise: each `+=` touches a distinct location once.

**`ret_buf` ring** (`:5124-5127`): `pos = atomicAdd(P.ret_total,1);
P.ret_buf[pos % P.cap] = ep;`. Lane-0 guard mandatory (32 lanes would consume
32 slots/env). Note a pre-existing fact the validation plan must accommodate:
**the slot assignment is already non-deterministic across runs in the
incumbent** — `pos` depends on block scheduling order. So the determinism gate
(§5b) must compare `ret_buf` as a **sorted multiset over the first
`min(ret_total, cap)` entries**, not element-wise. Not a regression introduced
here; a deterministic per-env slot is a separate item.

### 3.7 `write_obs` — the 578-lane write

`:4489-4673`. `OBS_DIM = 578` (`:161`); `P.obs` is always `float*` regardless of
env dtype (`cuda_env.py:517`); env `e`'s row is 2,312 contiguous bytes.

| segment | lanes | note |
|---|---|---|
| zero fill `[OFF_NOTES=100, 578)` (`:4491`) | 15 rounds of 32 | 478 floats, fully coalesced. K10e3 (start at `OFF_NOTES`) survives — `:4483-4488` |
| re-`load_state` (`:4492`) | §3.1 map | **keep it**: K10e1 measured deletion at −0.11% and it grew the f64 frame; it is a rematerialization point (`:4451-4462`) |
| root pos / 6D rot / vlin / vang (`:4497-4507`) | 0–14 | `quat_to_rot` recomputed redundantly per lane. **K10e2 hazard** (`:4463-4475`): a new inline site re-contracts FMAs and previously moved obs lanes 3/5/8 by 1 ulp. Expect it, baseline it, do not debug it |
| q block (`:4508-4526`) | lane = joint j | j→lane map is non-uniform (j=7,14 emit two lanes). Add `constexpr int Q_OBS_LANE[NQ]`; lane j writes `obs[Q_OBS_LANE[j]]` (+1 for j∈{7,14}) |
| qd (`:4527-4528`) | lane = j, 29 | contiguous |
| hand blocks ×2 (`:4530-4545`) | 20 | reads `P.wrist`/`wrist_vel`/`blade`/`swing_acc` |
| head + contact bits (`:4546-4551`) | 5 | |
| **entity-window selection** (`:4553-4611`) | **lane 0** | ordered insertion sort into 24 slots with an early exit; inherently serial. Publish `sel_idx`/`sel_wall` to the shared control block, `SW` |
| entity feature emit (`:4612-4658`) | 432 floats, 14 rounds | fully lane-parallel once `sel_*` is published |
| ground_imp (`:4660-4664`) | 4 | |
| body points (`:4666-4673`) | 6 | reads `d.pos[F_*]`, which under frame-per-lane live on 6 specific lanes; each writes its own 3 contiguous floats |

**Instruction-count honesty.** For a pure elementwise write, warp-per-env does
**not** reduce instructions per unit work: today 578 store instructions cover
32 envs; warp-per-env needs ~19/env × 32 = 608. What changes is sector
efficiency per instruction (32 → 4) and, for segments reading lane-distributed
`Dyn` state, removal of the strided gather. Budget the obs writer as
**wash-to-modest**, not a lever — consistent with K19 (`:556-628`), which found
its traffic already hidden behind the solve and priced extraction at −0.28%.
The `insert` lambda's `sel_d`/`sel_wall`/`sel_idx` (`:4568-4583`) stay in lane
0's registers; only the two output arrays are published.

### 3.8 mode 1 (reset-all) and mode 2 (set_state refresh)

Neither is hot: mode 1 is one launch per `reset()` (`cuda_env.py:567-570`),
mode 2 one per `set_state()` (`cuda_env.py:623`), against ~7,000 mode-0 launches
per episode.

**Simplest correct treatment: run both entirely on lane 0**, reusing
`reset_env`/`load_state`/`refresh_fk` verbatim against the arena, lanes 1–31
parked at a `SW`, with only `write_obs` (§3.7) lane-parallel. `P.mode` is a
launch constant, so the branch at `:5145-5159` is uniform across the whole grid
— zero divergence cost. Two free refinements:

- `for (int i = 0; i < P.C; i++) P.status[e*C+i] = 0;` in `reset_env` (`:4470`)
  is the "512-byte reset memset" the audit flags at ~3% sector efficiency
  (`:151-153`, Tier 3.2). One warp writing `C` contiguous bytes with
  lane-parallel `uint32_t` stores fixes it as a side effect of the mapping.
  Guard on `(e*C) % 4 == 0 && C % 4 == 0`, else fall back to the byte loop.
- **`auto_reset` inside mode 0** (`:5127`) is hot-path but rare (`done` ~1/240
  ticks) and now **warp-uniform**. Lane 0 is fine; by the K13 mechanism a rare
  warp-uniform branch costs only when taken.

---

## 4. Dual-kernel coexistence

**Non-negotiable: `step_kernel<T,FC>` (`:5137`) is not touched.** It stays the
f64 parity oracle against Rust and the instant escape hatch. The warp kernel is
a **new f32-only** `__global__` selected at launch time.

### 4.1 pybind signature

Current: `void step(std::vector<at::Tensor> ts, at::Tensor actions,
at::Tensor cand, int64_t mode, int64_t max_steps,
bool auto_reset, int64_t gears, bool want_contacts,
bool foot_contact, double fall_const)` (`:5181-5185`). Add **one trailing
parameter with a pybind default** so every existing caller stays
source-compatible: `int64_t kernel_variant = 0` (0 = legacy, 1 = warp-per-env),
declared with `py::arg("kernel_variant") = 0`.

### 4.2 The `AT_DISPATCH` structure accommodates this cleanly — confirmed

`AT_DISPATCH_FLOATING_TYPES(q.scalar_type(), "beat_arena_step_v10", [&]{...})`
(`:5194`) instantiates the lambda for **both** `float` and `double`. Unguarded,
`kernel_variant == 1` would force `step_kernel_warp<double>` to compile —
doubling the shared budget past the limit for a kernel that can never run. The
guard is `if constexpr`:

```cpp
    // hoisted before AT_DISPATCH, for a readable error:
    TORCH_CHECK(kernel_variant == 0 || q.scalar_type() == at::kFloat,
                "kernel_variant=1 (warp-per-env) is f32-only; got ", q.scalar_type());

    // inside the lambda, replacing :5285-5294:
    if (kernel_variant == 1) {
      if constexpr (std::is_same<scalar_t, float>::value) {
        const unsigned blocks = (unsigned)((N + ENVS_PER_BLOCK - 1) / ENVS_PER_BLOCK);
        if (foot_contact)
          step_kernel_warp<scalar_t, true><<<blocks, WARP_BLOCK_THREADS, 0,
              at::cuda::getCurrentCUDAStream()>>>(P);
        else
          step_kernel_warp<scalar_t, false><<<blocks, WARP_BLOCK_THREADS, 0,
              at::cuda::getCurrentCUDAStream()>>>(P);
      }
    } else {
      const unsigned blocks = (unsigned)((N + THREADS - 1) / THREADS);
      /* ... existing :5289-5294 verbatim ... */
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
```

`if constexpr` discards the `double` branch at compile time, so only
`step_kernel_warp<float,true>` and `<float,false>` are instantiated — two extra
kernels in the fatbin. The existing `foot_contact` compile-time split
(`:5285-5292`) is mirrored unchanged. `Sim<T>` gains one field
(`T* __restrict__ arena;`), harmless to the legacy kernel, which never reads it.

### 4.3 `cuda_env.py` plumbing — default OFF

Follow the repo's escape-hatch precedent (`BEAT_ATTN`, `BEAT_QKV_PACKED`,
`BEAT_ENV_PERM` — audit `:38-40, 45`) and the `want_contacts`/`foot_contact`
baking pattern (`cuda_env.py:369-381`):

```python
# in __init__, alongside foot_contact (cuda_env.py:382). Baked here, never
# mutated => constant per instance => CUDA-graph safe, exactly like
# want_contacts (369-374) and foot_contact (376-381).
self.kernel_variant = int(kernel_variant if kernel_variant is not None
                          else os.environ.get("BEAT_ENV_WARP", "0"))
if self.kernel_variant and self.dtype is not torch.float32:
    raise ValueError("kernel_variant=1 (warp-per-env) requires dtype=float32")
self.arena = torch.zeros(N if self.kernel_variant else 1,
                         ARENA_WORDS if self.kernel_variant else 1,
                         device=dv, dtype=dtype)
```

with `kernel_variant=None` appended to the constructor signature
(`cuda_env.py:349-359`, after `start_jitter`), `self.arena` appended to
`_tensors` (`cuda_env.py:524-550`, matching the new enum slot), and `_launch`
(`cuda_env.py:552-556`) gaining one trailing argument:

```python
    def _launch(self, actions, cand, mode):
        self._ext.step(self._tensors, actions, cand, mode,
                       self.max_steps,
                       self.auto_reset, self.gears, self.want_contacts,
                       self.foot_contact, self.fall_const,
                       self.kernel_variant)          # <-- new, default 0
```

The 1-element dummy arena when variant 0 keeps `_tensors` a fixed length so
`T_COUNT` matches in both modes — otherwise `TORCH_CHECK` at `:5186` fires.
**Default OFF** in both the env var and the pybind default; adoption = flipping
the default in `cuda_env.py` only, after §5d.

### 4.4 Graph-capture compatibility

`GraphedRollout` (`rollout_graph.py:36-65`) captures `venv.step(a)` inside
`torch.cuda.graph`; `cuda_graphs.copy_static` (`cuda_graphs.py:19-29`)
hard-enforces fixed shape and dtype. Variant selection is a **host-side Python
int baked at construction** ⇒ constant across launches ⇒ recorded once as a
fixed kernel node, identical in kind to `foot_contact`/`want_contacts`
(`cuda_env.py:369-381`, `:4348-4353`). Grid and block shapes derive only from
fixed `N`; the arena is a preallocated static tensor; no new host syncs.

The one thing that must **not** happen inside capture is the
`cudaFuncSetAttribute` carveout call (§1.3): expose it as
`m.def("configure_warp_kernel", ...)` and call it from `_load_ext()`
(`cuda_env.py:121-141`) at construction, before any warmup.

### 4.5 Build / census plumbing (already present)

`jit_cache.py:54` emits `-arch=sm_{maj}{mnr}` and `-Xptxas -v`; `_build_log`
(`jit_cache.py:71-99`) tees ptxas output to
`csrc/build/beat_arena_cuda.build.log`. The census gate is therefore free — the
same log the K-series rule reads (`:205-212`). `BEAT_LINEINFO=1`
(`jit_cache.py:58-59`) only for a SASS census, and note the header's verdict
that the SASS LDL census is a **weak instrument here** (`:319-325`) — gate on
"counters not worse", not "counters moved". The source hash covers the flags
(`jit_cache.py:110-119`), so adding the warp kernel forces one rebuild.

---

## 5. Validation plan

### 5a. Old-vs-new f32 tolerance harness

Past stage 2 the warp kernel changes accumulation order, so it is **not**
bitwise vs the incumbent. The f64-vs-Rust tier is unaffected (warp kernel is
f32-only, legacy kernel untouched), so **the bitwise-CRT parity claim survives
intact** — the main reason for keeping both kernels.

Follow the discipline the header set for the CUDA f64 tier (`:47-59`):
**horizon-bounded tolerance**, because v10 amplifies a 1-ulp seed at a measured
e-folding of ~0.35 s (≈10.5 ticks at 30 Hz). A 240-tick trajectory comparison
is not a test; it is a chaos demonstration.

Ladder — **calibrate on the stage-3 build, then freeze**; a tolerance chosen
after seeing the numbers is not a gate:

| horizon | quantities | gate |
|---|---|---|
| 1 tick, 32 fixed seeded states | q, v, p, quat | max abs diff ≤ 1e-5 |
| 1 tick | obs (578 lanes) | ≤ 1e-4; lanes 3/5/8 exempted to 1 ulp for the K10e2 `quat_to_rot` re-contraction (§3.7) |
| 8 ticks | q, v | max rel diff ≤ 1e-3 |
| 30 ticks (≈3 e-foldings) | — | **no per-trajectory gate**; well-formedness only: no NaN/Inf, root in envelope, `pd_hits`/`env_hits` within 2x |
| 240 ticks, N ≥ 4096 | ep_return, effort, fall_hits, score | **distributional**: mean within 2σ of incumbent over the same pool |

**Tests to clone/parameterize** (each funnels every `CudaBeatVecEnv(...)`
through one helper, so a `variant=` kwarg is a one-line change):

1. `python/tests/test_v11_contact_parity.py::_cenv` (line 69) — one
   construction site, already takes `dtype`; parameterizes all 7 tests and
   exercises the **contact** instantiation, the production one. **Primary.**
2. `python/tests/test_v10_parity_cuda.py::_cenv` (line 68) — 12 tests, already
   has the f32/f64 split; `test_parity_f32_bounded` (line 466) is the natural
   home for the ladder above.
3. `python/tests/test_v10_root_escape.py` (lines 151-178) — already
   `@parametrize("dtype", ...)`; a second axis is one line, tests are
   short-horizon by construction.
4. `python/tests/test_v17_wall_parity.py` (single test, line 62) — smallest
   CUDA file; cheap smoke.

New `python/tests/test_warp_env_equiv.py` holds the ladder itself (old vs new
kernel, same seeded states, both f32) plus §5b. The Rust-only suites
(`test_env.py`, `test_v10_env.py`, `test_v12_contact_obs.py`, …) have no CUDA
construction site and are **not** touched.

### 5b. Run-to-run bitwise determinism of the NEW kernel — REQUIRED

Model on `test_v11_contact_parity.py::test_contact_determinism_bitwise` (line
416) and `test_v10_parity_cuda.py::test_same_seed_reproduces` (line 544): same
build, same seed, same inputs, two runs, byte-equal on every state tensor and
`obs`. Threat model:

| source | verdict |
|---|---|
| `atomicAdd(pd_hits/env_hits/fall_hits, 1)` | **int** sums — order-independent |
| `P.contact_imp[...] += ...` | one env, distinct pids, no atomic — deterministic |
| `atomicAdd(ret_total,1)` + `ret_buf[pos % cap]` | **already non-deterministic in the incumbent** (§3.6) — compare as a sorted multiset over the first `min(ret_total, cap)` entries |
| warp reductions (effort, solver dots) | deterministic **iff** the tree shape is fixed in code. **No float atomics anywhere in the warp kernel** — hard rule, §7.3 |
| `__ballot_sync` / `__popc` compaction | pure functions of the predicate vector |

This is the **cheapest possible variant gate** (no Rust side needed) and runs at
every stage including stage 0.

### 5c. Pod A/B measurement protocol

Reuse the existing instruments; do not build new ones.

1. **Kernel timing** — `python perf_profile.py --phase env --num-envs 16384`
   and `--num-envs 32768`. It already profiles with `torch.profiler`
   (`perf_profile.py:23`), builds with `foot_contact=True, gears=5, seed=0`
   (`:42-44`), warms 20 ticks (`:54-55`), and prints
   `key_averages().table(sort_by="self_cuda_time_total")` (`:28-30`), where the
   kernels appear under their mangled names. Add a `--kernel-variant`
   passthrough into the `CudaBeatVecEnv(...)` call at `:42-44`; report ms/tick
   for the step-kernel line only.
2. **Production throughput** — `python train_batch.py --bench`.
   `--bench` (`train_batch.py:165-166`) forces the **exec-true** synchronised
   split (`:755`, `:905`) and prints `THROUGHPUT: … | rollout X% update Y%
   (exec-true)` (`:1018-1022`). Anything else is enqueue-attributed and not a
   measurement.
3. **Protocol** (house standard, `:184-197, 700-716`): **fresh process per
   arm**, ≥5 interleaved rounds, **order-balanced** batches, medians AND
   ranges, accept only **non-overlapping** ranges. Both weld and contact
   instantiations.
4. **ptxas census gate, EVERY stage, before reading any timing.** From
   `csrc/build/beat_arena_cuda.build.log`: registers must equal the
   `__launch_bounds__` pin (128 for W1); **spills must be 0**; shared/block
   must equal `ENVS_PER_BLOCK * SMEM_WORDS_PER_ENV * 4`. A spilling build is a
   revert regardless of timing — and a spilling build at 128 regs means the §2
   split is wrong, not that the budget should be relaxed.
5. **Occupancy verification** — `cudaOccupancyMaxActiveBlocksPerMultiprocessor`
   at load, printed once, asserted `>= MIN_BLOCKS_PER_SM`. With ncu locked
   (audit `:22-23`) this is the only direct instrument available; without it
   the "16 warps/SM" claim is unverified.

### 5d. Learning gate

The ruling metric is **leaf-end `ep_return`**, not bench upp (user decision
2026-08-14).

- **3k-update probe arm**, matched seed and config, warp kernel vs incumbent,
  compared on leaf-end `ep_return` before any campaign adopts it.
- **Read it as a smoke test, not an adoption gate.** The warp kernel changes
  bits ⇒ changes the chaotic trajectory ⇒ is behaviourally a **new seed**, not
  a modified recipe. The session history is explicit that a 6k reading can be a
  phase trough and that the return-slope branch rule needs a longer arm. So: a
  3k probe that *collapses* is a rejection; flat or noisy is **not**, and
  adoption waits on a matched pair at 9k+ or a two-seed distributional
  comparison.
- `pd_hits`/`env_hits`/`fall_hits`/`nc_overflow` must sit inside the
  incumbent's range. A perf win bought by silently degrading the solve shows up
  here first.

---

## 6. Incremental staging (shell side)

Every stage: build → **ptxas census** → determinism test → correctness gate →
A/B. A stage that fails its census is reverted before its timing is read.

**Stage 0 — shell scaffold, all existing code on lane 0.** New
`step_kernel_warp<T,FC>` mapping warp→env. Lane 0 runs the **existing**
`step_env`/`reset_env`/`refresh_fk`/`write_obs` verbatim against the per-env
arena; lanes 1–31 park at a `SW`. Selection plumbing (§4) complete.
`ENVS_PER_BLOCK = 4`, `__launch_bounds__(128,4)`.
*Gate: **BITWISE identical to the legacy kernel*** — stage 0 runs the same code
on the same values in the same order, so any diff is a plumbing or arena-layout
bug. Strongest gate available anywhere in the project; it validates the shell,
the tensor-list contract, the arena, the selection path, and the harness itself
in one shot. **Expect a large SLOWDOWN**, not "no speedup" (31/32 lanes idle,
cold arena) — record it as the floor. Census note: the reported stack frame
should collapse from ~42 KB to a few hundred bytes; if it does not, the arena
indirection did not take and the stage is broken.

**Stage 1 — state I/O + FK cache writes lane-parallel.** §3.1 and §3.2 (writes).
Pure copies, no reductions. *Gate: still bitwise vs legacy.* Measure.

**Stage 2 — collision broad+narrow phase + ballot compaction.** §3.3.
*Gate: still bitwise*, on the standard configs **plus** the contact-heavy
config the header describes (128-env random-gear traffic reaching `nc = 7`,
touching 48 of 55 pair ids, `:434-444`) — the 8-env chart configs never reach
`nc >= 4` and would not exercise the second ballot round at all. First stage
that can plausibly show a gain.

**Stage 3 — `fk_all` depth-scheduled, frame-per-lane.** §3.2. Introduces
`F_DEPTH` + `static_assert`. *Gate: bitwise, with an explicit allowance for the
K10e2 1-ulp class if `quat_to_rot`'s inline site moves* — freeze affected lanes
into the baseline rather than chasing them. First stage that shortens a chain
inside `substep`, so the first credible speedup.

**Stage 4 — torque, effort, `write_obs`.** §3.4 (option (a)) and §3.7. *Gate:
bitwise* except the documented obs lanes. If the obs half measures *negative*,
that is the K19 mechanism repeating (`:556-628`) and the obs writer stays on
lane 0.

**Stage 5 — aggregation, atomics discipline, modes 1/2.** §3.5, §3.6, §3.8. The
lane-0 guard audit (§8.2) lands here in full — the single largest bug-class
checkpoint in the plan. *Gate: bitwise*, plus a test that every counter
(`pd_hits`, `env_hits`, `fall_hits`, `ret_total`) increments **exactly once per
env-tick**: N=64, `auto_reset=False`, compared against the legacy counters
exactly.

**Stage 6 — tuning.** One variable at a time, each with its own census + A/B:
`NC_CAP ∈ {55,16,12}` with the `nc_overflow` counter (§2.4); W1 → W3 → W4
(§1.2); dense vs packed-triangle shared `m` (§2.2); effort (a) vs (b) (§3.4);
carveout 100 vs 64 KB (§1.3); the N-scaling flattening prediction (§1.2).
Anything that changes bits moves to the tolerance tier and re-runs §5a.

**Stage S — solver-core landings, interleaved from stage 3 onward.** Each
carries its own census + tolerance + determinism + A/B on the same protocol.
The first solver stage that reassociates a reduction permanently retires the
bitwise gate and promotes §5a's ladder to the correctness contract — **that
transition must be a single, deliberate, recorded commit**, not something that
drifts in.

**7 shell stages (0–6)** plus the interleaved solver track.

---

## 7. Interface assumptions for the solver-core agent

1. **Mapping.** `warp = threadIdx.x >> 5`, `lane = threadIdx.x & 31`,
   `e = blockIdx.x * ENVS_PER_BLOCK + warp`, `e` warp-uniform, one env per warp
   always. `ENVS_PER_BLOCK = 4`, `blockDim.x = 128` (W1) — compile-time
   constants, never runtime values.
2. **Sync.** `__syncwarp(0xffffffff)` **only**. **No `__syncthreads()`, ever** —
   a uniform early return leaves some warps in the block exited, so a block
   barrier deadlocks. No cooperative groups beyond a 32-lane tile. Document
   every sync point in the device function's header comment.
3. **No floating-point atomics anywhere in the warp kernel.** Every cross-lane
   reduction is a **fixed-shape shuffle tree** ordered by code, not by
   scheduling. Run-to-run bitwise determinism is a hard gate (§5b) and a float
   atomic breaks it unconditionally.
4. **Shared memory.** The shell provides `T* sm = smem + warp *
   SMEM_WORDS_PER_ENV`, `SMEM_WORDS_PER_ENV = 1536` (6,144 B). The shell
   reserves ~660 B for control; the solver owns ~5,480 B — enough for
   `m[NV*NV]` (4,900) + `aww[144]` (576) in weld mode. **Every extra KB/env
   costs resident warps 1:1** (§1.2), so more shared is an occupancy decision.
5. **Bank conflicts.** Store `m` column-major at stride NV=35 — coprime with
   32, so both row and column walks are conflict-free. **Do not pad to 36.**
6. **Distribution conventions.** NV-indexed: lane `l` owns `l` and `l+32` (2
   predicated rounds, lanes 0–2 active in round 2). NF-indexed:
   frame-per-lane, `k = lane`, lanes 30–31 idle. NQ-indexed: joint-per-lane,
   lanes 29–31 idle. The shell owns the arena layout and publishes an accessor
   header; row blocks (`c_J`, `c_minv`, `ecol`, `pcol`, `g_J`, `g_minv`) are
   **NV-minor** so a lane-parallel column sweep is contiguous.
7. **State at entry (per substep).** `d.q`, `d.v`, `tau` coordinate-distributed;
   `d.r`, `d.pos`, `d.axw` frame-per-lane; `s.nc`/`s.nl`/`s.ng` **warp-uniform
   in registers**; `s.c_pair` (shared, **pid-ascending**), `s.c_pt`, `s.c_bias`
   filled; `s.l_j`, `s.l_sign`, `s.l_bias` in joint order; `d.m` in shared,
   column-major.
8. **State at exit.** `d.v` updated in the same distributed layout;
   `s.c_lam`/`s.g_lam` readable by the shell's aggregation lanes;
   `d.pd_bad`/`d.env_bad`/`d.env_n` **warp-uniform on every lane** (the shell
   guards the `atomicAdd`s on lane 0 but may read from any lane).
9. **PGS row order is semantic, not just deterministic.** Rows sequential in
   the existing order (welds, contacts in pid order, limits in joint order);
   lane parallelism goes *inside* a row (over NV), not across rows. Same for
   Cholesky column order and the CRBA leaf→root merge.
10. **The K13 divergence tax is gone.** `nc`/`nl`/`ng`/`na` are warp-uniform, so
    the compile-time width-ladder idiom (`:600-660`) is free again — its
    selector is uniform by construction. Re-probe anything K13 killed.
11. **f32 only.** No fast-math, no ftz, no `__fmaf_rn` rewrites that change
    contraction vs the legacy kernel without a recorded tolerance-baseline
    entry.
12. **Every stage carries a ptxas census** (regs = the pin, spills = 0, shared =
    the budget) and is reverted on a census failure regardless of timing.

---

## 8. Risks and mitigations

**8.1 Shared capacity vs envs/block.** 6,144 B × 4 = 24 KB/block is
comfortable, but the budget is fully consumed by `m` + `aww` with **no
headroom**; a solver request for 8 KB/env silently takes W1 from 16 to 12
warps/SM. *Mitigation*: `static_assert(SMEM_WORDS_PER_ENV * ENVS_PER_BLOCK *
sizeof(T) <= 48*1024)`; the shared-per-block census every stage; the
packed-triangle probe (§2.2) held in reserve as the release valve.

**8.2 `__syncwarp` vs `__syncthreads`, and the lane-0 guard audit** — the two
highest-frequency bug classes here. *(A)* a CI grep asserting `__syncthreads`
appears **zero** times in `step_kernel_warp`'s call graph. *(B)* an explicit
lane-0-guard checklist; each of these fires 32x if unguarded: `P.pd_hits`,
`P.env_hits`, `P.fall_hits`, `P.ret_total`/`P.ret_buf` (`:5124-5127`),
`P.effort[e]` (`:4753`), `P.tick[e]` (`:5052`), `P.first_pending[e]` (`:5057`),
`P.first_wall[e]` (`:5049`), `P.walls_in[e]` (`:5044`),
`P.ep_return`/`P.reward`/`P.done` (`:5118-5120`), `P.score*[e]`,
`P.good/bad/miss/bombs[e]`, `P.contact_arm` (`:4755-4756`), `P.chart_idx[e]`
(`:4466`), `P.swing_acc`/`P.axis`/`P.axis_valid`/`P.tr_*`. *(C)* the stage-5
counter test catches the whole class empirically.

**8.3 Divergence across envs sharing a block.** Different `nc`/`nl`/`ng`/`done`
mean warps take different time, but with no block barrier they schedule
independently — divergence becomes *inter-warp scheduling*, which the SM
handles by design. A genuine structural improvement. *But budget nothing for
it*: the sm_89 permutation probe measured **null** (audit `:44-52`).

**8.4 CUDA-graph capture.** *Satisfied*: variant selection is a host-side
Python int baked at construction (§4.3), identical in kind to
`foot_contact`/`want_contacts`; grid/block derive from fixed `N`; the arena is
preallocated; `cuda_graphs.copy_static` (`cuda_graphs.py:19-29`) enforces
shape/dtype at the boundary. *Residual risk*: the `cudaFuncSetAttribute`
carveout must fire **once at load, outside capture** — a dedicated
`configure_warp_kernel` pybind called from `_load_ext()`
(`cuda_env.py:121-141`), never from `_launch`.

**8.5 ptxas phase flips** — the failure mode that killed NC_CAP (254 → 168,
+3.1%, audit `:86-91`), the K6 spill cliff (`:200-215`), and K12's "moved the
allocator by EXACTLY ZERO" (`:355-370`). *Mitigation*: pin the budget with
`__launch_bounds__(128,4)` so ptxas has no free choice; census
regs/frame/spills/shared every stage; honour the K11/K12 corollary — **gate on
the counters not getting WORSE, not on them moving**, because the largest
measured wins here were invisible to every static counter.

**8.6 Arena VRAM at N=32768** — ~31 KB/env = 1.0 GB (§2.3) on top of the policy
and the ranked pool. *Mitigation*: check free VRAM at construction and raise a
clear error rather than OOMing mid-capture; make the stage-6 capacity reduction
(55 → 12, ~390 MB) a prerequisite for any N=32768 campaign.

**8.7 The rewrite may simply not pay.** The audit priced it as "weeks of work …
Needs an explicit go/no-go as a project" (`:94-98`), and three of that round's
largest predicted wins were refuted by measurement. *Mitigation*: the staging
lets the project be abandoned cheaply at any stage boundary with the legacy
kernel untouched and the default still `0`. Stages 0–2 are bitwise and answer
"does the mapping cost anything on its own?" before a line of solver math is
rewritten. **If stage 3 — the first real chain cut — shows no measurable gain
on the §5c protocol, stop.**

---

## 9. Open questions

1. **Shared `m` (Path S) or shuffle-distributed `m` (Path R)?** Path S caps
   occupancy at ~16–18 warps/SM and costs 96 KB of the 128 KB unified cache;
   Path R could reach W3/W4 (24–32 warps/SM) with L1 intact, at ~1,225
   `__shfl_sync` per factorization. Solver's call; the shell assumes S and has
   a clean fallback either way.
2. **The NV=35 tail.** Two predicated rounds (lanes 0–2 double duty), redundant
   compute on 3 lanes, or a 35-lane logical vector spilling 3 values to shared?
   Affects every NV-wide dot in the solver and every NV-indexed arena array.
3. **`CLAMP_PASSES = 64` Gauss-Seidel over NQ=29 with an early-out `changed`
   flag** (`:4044-4060`) — the early-out becomes a `__ballot_sync` warp-uniform
   test. Shell or solver? Currently unassigned.
4. **Is the arena actually faster than per-thread local for the residual
   arrays?** The §2.3 argument is structural (contiguity vs thread-interleaving)
   but unmeasured. Stage 0's A/B is the first data point and is confounded by
   the 31 idle lanes.
5. **Does the N-scaling curve flatten?** §1.2 predicts equal us/env at N=16384
   and 32768. If it does *not*, the "parallelism-starved" diagnosis was wrong
   and the whole thesis needs revisiting.
6. **Make the `ret_buf` ring deterministic while we are here?** A pre-existing
   non-determinism (§3.6) that forces a sorted-multiset comparison in the
   determinism gate. Out of scope, cheap, and it would strengthen every future
   test.
7. **Effort reduction: bitwise-serial (a) or reassociated (b)?** §3.4. Deferred
   to a stage-6 A/B so stages 0–5 keep a bitwise gate.
8. **Does the entity-window top-k (`:4553-4611`) ever get hot enough to warrant
   a warp-parallel selection?** The only serial tail left in the shell after
   stage 4. Measure before designing.
