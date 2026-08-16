# Warp-per-env rescue — OCCUPANCY / RESIDENCY ARITHMETIC + REGISTER FORENSICS

Lens: how many independent **env chains** the SM can hold, what the register
allocator actually did, and whether any launch configuration closes the gap.
Analysis only — no source touched.

Inputs: `python/beat_arena_env/csrc/beat_arena_cuda.cu` (6,953 lines),
`design/WARP_ENV_SHELL.md` §1, `design/WARP_ENV_SOLVER_CORE.md` §4.6/§6,
`design/WARP_ENV_RECONCILIATION.md`, `PERF_AUDIT_2026-08-16.md`,
and the 2026-08-15 RTX 4090 measurements.

---

## 0. The one-line refutation

`step_kernel` at 255 regs/thread lets the SM hold exactly **256 threads**.
Legacy spends **1 thread per env** → 256 env-chains/SM (128 at N=16384, where
the grid runs out first). `step_kernel_warp` spends **32 threads per env**
→ **8** env-chains/SM. The design docs compare *warps/SM* (4 → 16, "4× more")
and never convert to *envs/SM* (128 → 16, **8× fewer**). That dropped factor
of 32 is the whole regression.

**SHELL §1.2's table is arithmetically correct and structurally misleading:**
every register/block/warp number in it checks out (§1.1 below), but the table
has no `envs/SM` column, and the prose headline — *"16 warps/SM is 4× today's
measured 4"* (`WARP_ENV_SHELL.md:107`) — compares a quantity that changed
meaning between the two rows.

---

## 1. sm_89 residency arithmetic, exact

AD102 / CC 8.9 per SM: 65,536 registers · max 255 regs/thread · 48 warps ·
24 blocks · 100 KB shared (configurable carveout) · **48 KB static shared per
block**. Register allocation granularity on CC 8.x: 256 regs per warp-unit,
i.e. regs/thread rounds **up to a multiple of 8**, then

```
blocks/SM = min( 65536 / (threads_per_block * roundup8(R)),
                 48 / warps_per_block, 24, carveout / smem_per_block )
```

### 1.1 SHELL §1.2 verified against that formula

| SHELL row | thr/blk | R | blk/SM claimed | blk/SM computed | warps/SM | **envs/SM** | smem/SM |
|---|---:|---:|---:|---:|---:|---:|---:|
| today (`:5138`, `THREADS=32` `:929`) | 32 | 255 | 4 (grid-limited) | 8 (reg ceiling) | 8 | **256** | 0 |
| W1 `(128,4)` | 128 | 128 | 4 | **4** ✓ | 16 | **16** | 78,400 B |
| W2 `(256,2)` | 256 | 128 | 2 | **2** ✓ | 16 | **16** | 98,304 B |
| W3 `(128,6)` | 128 | 85→80 | 6 | **6** ✓ | 24 | **24** | 98,400 B |
| W4 `(128,8)` | 128 | 64 | 8 | **8** ✓ | 32 | **32** | 99,200 B |

Three corrections, none of them arithmetic:

1. **The missing column is `envs/SM`.** Adding it inverts the table's verdict:
   today is the *best* row by 8–16×, not the worst.
2. **W2 is not buildable as written.** 8 envs/block × 6,144 B = **49,152 B =
   exactly the 48 KB static-shared limit**, zero slack for anything else the
   block declares. It fails `ptxas` or forces dynamic shared.
3. **W1/W3/W4 all need an explicit carveout call.** 78–99 KB/SM of shared only
   exists at the 100 KB carveout; sm_89's rungs are {0,8,16,32,64,100} KB.
   Without `cudaFuncSetAttribute(…, PreferredSharedMemoryCarveout, 100)`
   (SHELL §1.3 mentions it; nothing in `beat_arena_cuda.cu` calls it) W1 gets
   **3 blocks/SM = 12 envs/SM**, not 4/16 — a silent 25% residency loss.

### 1.2 Measured configurations, resident env-chains per SM

| config | thr/blk | lanes/env | envs/blk | R | blk/SM | warps/SM | **envs/SM** | binder |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| legacy, N=16384 | 32 | 1 | 32 | 255 | 4 | 4 | **128** | grid (512 blk / 128 SM) |
| legacy, N=32768 | 32 | 1 | 32 | 255 | 8 | 8 | **256** | registers |
| warp S1 (`:6702`) | 128 | 32 | 4 | 255 | 2 | 8 | **8** | registers |
| warp S2, FC=false | 128 | 32 | 4 | 255 | 2 | 8 | **8** | registers |
| warp S2, FC=true | 128 | 32 | 4 | **210** | **2** | 8 | **8** | registers |

**The 255 → 210 drop bought exactly nothing.** At 128 threads/block the
residency rungs are:

| regs/thread | roundup8 | blk/SM | warps/SM | envs/SM (32 lanes) |
|---:|---:|---:|---:|---:|
| 169 – 255 | 176 – 256 | **2** | 8 | 8 |
| 129 – 168 | 136 – 168 | **3** | 12 | 12 |
| 105 – 128 | 112 – 128 | **4** | 16 | 16 |
| 97 – 104 | 104 | 4 | 16 | 16 |
| 81 – 96 | 88 – 96 | 5 | 20 | 20 |
| 65 – 80 | 72 – 80 | 6 | 24 | 24 |
| 41 – 64 | 48 – 64 | 8 | 32 | 32 |
| ≤ 40 | 40 | 12 (48-warp cap) | 48 | 48 |

210 lands **mid-rung**. The next rung down is 168. FC=true gave back 45
registers and crossed no boundary — which is exactly consistent with S2
measuring *slower* (634.6 vs 614.0 ms/tick): all of the sweep's cost, none of
its occupancy.

### 1.3 The latency/concurrency decomposition of the measured numbers

At N=16384 legacy resident concurrency is 128 envs/SM × 128 SM = 16,384 =
**the whole batch in one wave**, so tick time *is* the per-env chain latency.
The warp kernel holds 1,024 envs → 16 waves.

| build | envs/SM | concurrent envs | waves | ms/tick | **chain latency** | env rate (envs/ms/SM) |
|---|---:|---:|---:|---:|---:|---:|
| legacy N=4096 | 32 | 4,096 | 1 | 56.1 | 56.1 ms | 0.570 |
| legacy N=8192 | 64 | 8,192 | 1 | 79.6 | 79.6 ms | 0.804 |
| **legacy N=16384** | 128 | 16,384 | 1 | 124.0 | 124.0 ms | **1.032** |
| legacy N=32768 | 256 | 32,768 | 1 | 194.0 | 194.0 ms | 1.320 |
| warp S1 | 8 | 1,024 | 16 | 614.0 | **38.4 ms** | 0.209 |
| warp S2 | 8 | 1,024 | 16 | 634.6 | 39.7 ms | 0.202 |

Two facts fall out that neither design doc states:

- **The uncontended serial chain is ~38 ms/env-tick.** S1 does zero parallel
  work, so its 38.4 ms *is* the legacy chain with 16× less contention.
  Legacy's 124 ms is the same chain inflated **3.2×** by contention: it buys
  16× concurrency for 3.2× latency, net 4.95×. That is the entire S1
  slowdown — a *residency* result, not a *mapping* result.
- **Legacy is not bandwidth-saturated.** Per-env local-line traffic scales as
  `32 / envs_per_warp` (one 128 B local line per warp per access whatever the
  active-lane count: legacy uses 32/32 words of it, warp-per-env 1/32). S1 ran
  **32× legacy's per-env line traffic at 1/4.95 the env rate = 6.4× legacy's
  line-traffic rate** and did not fall over, so there is ≥6.4× headroom above
  legacy's current local demand. This retires the audit's "bound by its 42 KB
  local frame" (`PERF_AUDIT:60-66`) as a *bandwidth* claim: the binder is
  latency × concurrency, and the frame binds through the **register file**.

---

## 2. Break-even chain-shortening factors

Break-even condition, per SM: `E / L ≥ E_legacy / L_legacy`, i.e. a config
holding `E` env-chains needs chain latency `L ≤ 38.4 ms / κ` where

```
κ = 38.4 ms × (E_legacy / L_legacy) / E
```

κ is measured-anchored (it uses S1's real 38.4 ms chain, not a model).

| config | R | envs/SM | **κ vs legacy@16384** | **κ vs legacy@32768** |
|---|---:|---:|---:|---:|
| warp S1/S2 **as shipped** | 255 / 210 | 8 | **4.95×** | 6.33× |
| warp, one rung down | 168 | 12 | 3.30× | 4.22× |
| **W1 as designed** | 128 | 16 | **2.48×** | 3.16× |
| W1 without the carveout call | 128 | 12 | 3.30× | 4.22× |
| W3 | 80 | 24 | 1.65× | 2.11× |
| W4 | 64 | 32 | 1.24× | 1.58× |
| hardware warp ceiling, 32 lanes | ≤40 | 48 | 0.83× | 1.05× |

Against this, the design's own parallelism ceiling from `WARP_ENV_SOLVER_CORE.md:494-510`:
**91,000 dependent steps → 7,100 = 12.8×**, and its own *grounded* estimate at
`:634-644` after a 10× discount against the file's prediction history:
**2–4× total, of which 2–4× is the "chain restructure" component and 1.6–2.0×
was the (miscounted) occupancy component**.

Read together, the current launch config is dead: **required κ = 4.95×
(6.33× vs legacy's best N); grounded band 2–4×.** The full S3–S6 rewrite at
its own *optimistic* model ceiling of 12.8× clears 4.95 but sits outside its
own honest estimate.

Second-order trap: SOLVER_CORE's occupancy component (1.6–2.0×, `:637`) is
*negative* in reality. It was derived from 4 → 16 warps/SM; the same change is
128 → 16 **envs**/SM. Removing it drops the grounded estimate to **1.25–2.5×**
— below break-even for every rung above W4.

### 2.1 The 5-cycle model is off by ~24×

SOLVER_CORE §6 prices the serial chain at `91,000 dep steps × 5 cyc × 8
substeps ≈ 3.6M cycles ≈ 1.5 ms/env-tick`. Measured (S1, uncontended):
**38.4 ms ≈ 88M cycles at 2.3 GHz — 24× the model.** Each "dependent step" is
a *local-memory round trip*, not an ALU latency: `Dyn`/`Solver` are kernel
locals (`:5145-5147`) and the frame is 42 KB. Consequence: **lane-parallelism
removes issue count, not memory latency.** Only phases whose operands become
register- or shared-resident actually shorten — which is the lens the S2
post-mortem needs (§3.3).

Also `WARP_ENV_SOLVER_CORE.md:592`: *"at 16 warps/SM, N=16384 (128 envs/SM, 8
rounds)"* — 16 warps/SM at one env per warp is **16** envs/SM. The 8-round
figure is right (16384 / (128 SM × 16)); "128 envs/SM" is the same
warps-vs-envs conflation, inside the model that justified the project.

---

## 3. Register forensics — why FC=true fell to 210 and FC=false did not

### 3.1 `__launch_bounds__(128,1)` imposes no cap at all

`65536 / (128 × 1) = 512`, clamped to the hardware max **255**. So
`__launch_bounds__(WARP_BLOCK_THREADS, 1)` (`:6702`) sets the ceiling to
exactly where it already was. Same for legacy: `65536/(32×1) = 2048 → 255`
(`:5138`). Neither bound constrains the allocator; both merely tell ptxas
*"optimize for 1 block/SM"*, which sets a 4-warp occupancy target so low that
ptxas will take whatever the live ranges demand.

Therefore:

- **255 is a saturation reading, not a choice.** It means the allocator wanted
  ≥255 and was clamped. Legacy's zero-spill 255 (`PERF_AUDIT:19-21`) means it
  wanted ~255 exactly.
- **210 is a genuine demand reading.** FC=true's peak live-range pressure
  actually *fell below* the ceiling.
- The kernel header's claim at `:915-926` that the `,1` "stops ptxas's
  free-choice heuristic from flipping to 64 regs" is refuted by the repo's own
  evidence: the `NC_CAP=16` build flipped **254 → 168 under the identical
  `__launch_bounds__(THREADS,1)`** (`PERF_AUDIT:86-91`). The pin does not pin.
  What it does is bias the *target*; the allocator still re-solves on any
  frame change.

### 3.2 What the fused sweep removed, per FC path

The stage-2 sweep (`:5637-5711`) replaces the effective-mass ladder. Compare
what each path lost:

| path | removed by the sweep | live-range weight of the removed block |
|---|---|---|
| FC=true | ground ladder `:3555-3586`: `chol_solve_multi_n<4>` / `<2>` over `s.g_minv` | `T s0[W], s1[W], s2[W]` at W=4 (`:2446`) = **12 live accumulators** + 3 factor values + 4 `xk`, fully `#pragma unroll`ed, over 35 columns — the ground path's high-water mark |
| FC=false | `chol_solve_multi_tn<N_WELD=12>` (`:3548`) | wider still (12 RHS), but **not the path's high-water mark** |

What FC=false **keeps**, all inside the copied Phase C (`:5739-6001`): the
fully-unrolled 12×12 `A_ww` assembly (`:5762-5786`, `#pragma unroll` on both
`r` and `c` plus a 6-term inner unroll = 78 entries in flight);
`chol_factor_n(N_WELD, s.aww, …)` (`:5786`); `ensure_pcol_multi<4>`/`<2>`
(`:5804-5812`) — the block the legacy header itself calls *"a doubled
live-`x[8][12]` projection loop"*; and the PGS weld block with `T jv[N_WELD]`
fully unrolled (`:5829+`).

**Mechanism, stated plainly:** the allocator's number is the *maximum* over
program points, not a sum. Removing the ground ladder lowered FC=true's
maximum because that ladder *was* the maximum. Removing the weld solve did not
lower FC=false's maximum because the peak simply relocated to the weld block,
which the sweep does not touch. It is a max-over-regions effect, and the
sweep only deleted one region.

Frame: 42,432 → 41,504 B (−928 B) on FC=true, despite the sweep *adding*
`T x[NV]` (140 B, `:5660`). Consistent: the ladder's staged `g_minv` copies
and the multi-RHS temporaries were worth ~1 KB.

### 3.3 The actual S2 regression: `T x[NV]` is a local array

`:5660` declares `T x[NV]` (35 floats). Two things force it out of registers:

- `:5684` `if (lane == w) x[m] = T(1);` — **runtime index** `m`;
- `:5692` `chol_solve_n(NV, sm, x)` — takes `T* __restrict__ x`, and
  `chol_solve_n` (`:2363`) loops on a **runtime** `n`, so it cannot be
  unrolled to constant indices.

So every one of the 32 lanes now runs ~1,190 dependent **local-memory**
load/stores for its solve, where the serial version ran them on one lane.
That is a **32× local-traffic amplification on the sweep's hot loop**, added on
top of `warp_stage_factor`'s serial 630 loads + 630 shared stores per substep
(`:5336-5346`, all 31 other lanes parked at `__syncwarp`) and 70 shuffles per
RHS in `warp_rhs_in`/`warp_rhs_out` (`:5347-5375`), issued one RHS at a time
in a `for w` loop. +3.4% is a mild outcome for that.

### 3.4 Mechanisms to restore a fat allocation for the serial body

Ranked by cost/benefit, given that §2 says **register width is not the
problem** — the serial body already gets its 255 and still loses on residency:

| mechanism | does it work? | verdict |
|---|---|---|
| `__noinline__` on `step_env_warp`/`substep_warp` so the serial body gets its own allocation | ABI-called functions do get their own allocation, but the **kernel's** occupancy register count is `max(caller, callee)` — CUDA has no per-function register windows. It also pushes `Dyn`/`Solver` across an ABI boundary (more local traffic). | **No.** Costs traffic, buys no occupancy. |
| Per-function `__launch_bounds__` | Not a thing: `__launch_bounds__` is a `__global__` attribute; `__maxnreg__` (12.x) is likewise per-kernel. | **No.** |
| Forced spilling structure (spill the serial body so ptxas reports a low R) | Backwards — but *deliberately* so. Spilling lowers R and raises local traffic, i.e. it trades the resource with ≥6.4× headroom (§1.3) for the one we are short of. That is the right direction of trade; it is just a blunt instrument. | **Marginal.** Only as a way to force the 64-reg rung. |
| **Two kernels per tick** (serial thread-per-env at 255 regs + parallel sweep) | Changes the residency arithmetic, but see below. | **Viable, and bad here.** |

**Costing the split honestly.** The split point is inside the substep and
there are `N_SUB=8` per tick (`:965`), so it is **≥16 launches/tick**, not 2.

- Launch overhead: ~2 µs graphed × 16 = 32 µs against 124,000 µs — **0.03%,
  negligible.**
- **Hand-off is the killer.** Live state at the split is the `Solver` working
  set: `c_J`+`c_minv` (15,400 B) + `ecol` (4,060 B) + row/diag ≈ 25 KB/env
  that today never leaves the thread's local frame. At N=16384: 410 MB out +
  410 MB in per substep × 8 = **6.5 GB/tick** ≈ **6.5 ms** at ~1.0 TB/s
  (`PERF_AUDIT:32`) — 5% of budget, and only if coalesced, which needs the
  SHELL §2.3 arena that does not exist.
- Decisive objection: the serial kernel still carries 100% of the dependent
  chain. The sweep is 42% of *ops* but ~1,190 of 91,000 *dependent steps* once
  fused. **You cannot Amdahl your way out by parallelizing the cheap half.**

**Conclusion of §3: there is no allocation trick.** The serial body already
has the fat allocation it wants. Restoring it is not the problem; the problem
is that 255 regs × 32 threads/env = 8 envs/SM.

---

## 4. The config ladder, quantified

Envs/SM under the warp family reduces to one identity:

```
envs/SM = threads/SM / lanes_per_env ,   threads/SM = 128 × blocks/SM
```

and at 255 regs `threads/SM = 256` on **any** block shape. Legacy is simply
the `lanes_per_env = 1` endpoint of the same ladder.

| # | config | lanes/env | envs/blk (128 thr) | R | blk/SM | **envs/SM** | packed smem/blk | smem/SM | **κ vs 16384** | **κ vs 32768** |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| a | **warp-per-env today** | 32 | 4 | 255 | 2 | **8** | 10,080 B | 20 KB | **4.95×** | 6.33× |
| b | **W1** | 32 | 4 | 128 | 4 | **16** | 10,080 B | 40 KB | **2.48×** | 3.16× |
| c | **sub-warp 8 lanes** | 8 | 16 | **255** | 2 | **32** | 40,320 B | 79 KB | **1.24×** | 1.58× |
| c′ | sub-warp 8 lanes, W1 regs | 8 | 16 | 128 | 4 | **64**† | 40,320 B | 158 KB ✗ | 0.62× | 0.79× |
| d | **2 envs/warp (16 lanes)** | 16 | 8 | 255 | 2 | **16** | 20,160 B | 40 KB | 2.48× | 3.16× |
| d′ | 16 lanes, W1 regs | 16 | 8 | 128 | 4 | **32** | 20,160 B | 79 KB | 1.24× | 1.58× |
| e | 4 lanes/env | 4 | 32 | 255 | 2 | **64** | 80,640 B ✗ | — | 0.62× | 0.79× |

† `c′` is shared-limited to 3 blocks/SM = 48 envs/SM (κ = 0.83× / 1.05×).
Packed factor = lower triangle, `NV(NV+1)/2 = 630` words = 2,520 B/env; the
shipped tile is **dense `NV*NV` = 4,900 B/env** (`:5325`, `:6719`), which
alone rules out rows c, c′, e.

**Two hard ceilings the docs never state:**

1. **The shared factor caps residency at 39 envs/SM.** 100 KB / 2,520 B = 39.8
   with a packed tile; 20 envs/SM with the shipped dense tile. Legacy has no
   such cap because its factor lives in the per-thread local frame, whose
   backing store is device memory (effectively unbounded), not 100 KB of SRAM.
   *Moving the factor to shared is what trades legacy's 256-env residency for a
   39-env one.* This is the single most under-priced decision in
   `WARP_ENV_RECONCILIATION.md:16-22`.
2. **32-lane mapping cannot exceed 48 envs/SM at any register count** (48
   warps/SM hardware cap), i.e. it is 2.67× below legacy@16384 and 5.33× below
   legacy@32768 *before any code runs*.

### 4.1 Which rows can win, against the solver's own phase analysis

Discount SOLVER_CORE §4.6's 32-lane 12.8× by lane width — the width-limited
phases (CRBA column fill at 35, fused solves at 33 RHS, row assembly at 35)
tile in `ceil(width/lanes)` rounds; the order-bound phases (PGS 2,900, RNEA
1,800, CRBA merge 360) do not shrink at all:

| lanes/env | modelled dep steps/substep | model κ | grounded κ (÷ ~3, per the doc's own discount rule at `:620-632`) |
|---:|---:|---:|---:|
| 32 | ~7,100 | 12.8× | 2.0 – 4.3× |
| 16 | ~10,300 | 8.8× | 1.7 – 2.9× |
| 8 | ~15,000 | 6.1× | 1.4 – 2.0× |

Cross-referencing κ-required against κ-available:

| config | κ required | κ available (grounded) | **possible?** |
|---|---:|---|---|
| a — warp today, 8 envs/SM | 4.95 / 6.33 | 2.0 – 4.3 | **No.** Fails even the model against N=32768's 6.33 by a hair, fails the grounded band outright. |
| b — W1, 16 envs/SM | 2.48 / 3.16 | 2.0 – 4.3 | **Marginal.** Straddles. And 128 regs will spill the undistributed 42 KB frame (the kernel header at `:6693-6701` says so itself), which lengthens the chain — so the two halves of the plan fight each other. |
| d — 16 lanes, 16 envs/SM | 2.48 / 3.16 | 1.7 – 2.9 | **Marginal-negative.** |
| d′ — 16 lanes @128 regs, 32 envs/SM | 1.24 / 1.58 | 1.7 – 2.9 | **Yes**, with margin, *if* 128 regs is reachable without spills. |
| **c — 8 lanes @255 regs, 32 envs/SM** | **1.24 / 1.58** | 1.4 – 2.0 | **Yes — and it needs no register surgery at all.** |
| c′ — 8 lanes @128 regs, 48 envs/SM | 0.83 / 1.05 | 1.4 – 2.0 | **Yes**, largest margin; needs both the packed tile and 128 regs. |

**One correction to the optimism.** Rows c/c′ raise per-env local-line traffic
8× over legacy (`32/envs_per_warp = 32/4`). At break-even env rate that is 8×
legacy's line-traffic rate against the ≥6.4× headroom S1 demonstrated (§1.3),
so row c sits *right at the measured bandwidth wall* — it lands, with no room,
unless the frame moves to the SHELL §2.3 arena. Rows b/d are worse on this
axis (32×/16×): a second, independent reason 32 lanes is the wrong end.

**This invalidates the project STOP RULE.** `WARP_ENV_RECONCILIATION.md:66-68`
says *"if S2+S3 do not beat the legacy kernel end-to-end, stop."* At 8 envs/SM
S2+S3 **cannot** beat legacy — they would need 4.95× from two stages that are
70% of ops but only ~40% of dependent steps. As written the rule kills the
project for a reason unrelated to S3's merit. Re-specify: *stop if the mapping
does not beat legacy **after** the residency config is fixed.*

---

## 5. Verdict and the single next experiment

**Yes, a winning configuration exists — but it is not warp-per-env.** It is
**sub-warp-per-env at 8 lanes** (`c` / `c′`), which is the only branch of the
ladder that reaches ≥32 envs/SM without needing a register allocation the
42 KB frame cannot support. 32 lanes/env is structurally refuted: it is capped
at 48 envs/SM by the warp limit and at 39 by the shared factor, both of which
sit below legacy's 128, so it must buy back the whole deficit from
parallelism that its own phase analysis prices at 2–4×.

### The experiment: a pure residency probe — S1 body, 8 lanes/env

Change the mapping only. **No sweep, no shared tile, no register change.** One
number decides the project.

```diff
--- a/python/beat_arena_env/csrc/beat_arena_cuda.cu   (~:6673)
-constexpr int ENVS_PER_BLOCK = 4;
-constexpr int WARP_BLOCK_THREADS = ENVS_PER_BLOCK * 32;  // 128
+constexpr int LANES_PER_ENV      = 8;    // ladder rung: 32 | 16 | 8
+constexpr int WARP_BLOCK_THREADS = 128;  // unchanged: 2 blocks/SM at 255 regs
+constexpr int ENVS_PER_BLOCK     = WARP_BLOCK_THREADS / LANES_PER_ENV;  // 16
+constexpr unsigned SUB_MASK      = ((1u << LANES_PER_ENV) - 1u);

--- a/python/beat_arena_env/csrc/beat_arena_cuda.cu   (~:6704)
 step_kernel_warp(Sim<T> P) {
-  const int warp = threadIdx.x >> 5;
-  const int lane = threadIdx.x & 31;
-  const int e = blockIdx.x * ENVS_PER_BLOCK + warp;
+  const int sub  = threadIdx.x / LANES_PER_ENV;          // env slot in block
+  const int lane = threadIdx.x % LANES_PER_ENV;          // 0 .. LANES-1
+  const int e    = blockIdx.x * ENVS_PER_BLOCK + sub;
+  // group mask replaces WARP_FULL everywhere in the warp call graph
+  const unsigned gm = SUB_MASK << ((threadIdx.x & 31) - lane);
   if (e >= P.N) return;
-  __shared__ T smem[ENVS_PER_BLOCK * WARP_SM_WORDS_PER_ENV];
-  T* const sm = smem + warp * WARP_SM_WORDS_PER_ENV;
+  T* const sm = nullptr;   // S1 probe: no tile. (16 x 4,900 B = 78 KB > 48 KB
+                           //  static limit — the dense tile at :5325 must be
+                           //  packed to 630 words before ANY sub-warp S2.)

# and, mechanically, through warp_stage_factor/warp_rhs_in/warp_rhs_out
# (:5336-5375) and constraint_solve_warp (:5646-5737):
-  __syncwarp(WARP_FULL)            ->  __syncwarp(gm)
-  __shfl_sync(WARP_FULL, v, 0)     ->  __shfl_sync(gm, v, 0, LANES_PER_ENV)
```

Gate the probe behind `BEAT_ENV_WARP_LANES` so the 32-lane build stays
byte-identical, and keep the S2 sweep compiled out (`if constexpr (LANES_PER_ENV
== 32)`) so the measurement isolates residency.

**Prediction (falsifiable, sharp).** 32 envs/SM = 4× S1's residency, same
serial chain, ~8× per-env line traffic instead of 32×:

| rung | envs/SM | predicted ms/tick | vs legacy 124.0 |
|---|---:|---:|---:|
| 32 lanes (S1, measured) | 8 | 614.0 | 4.95× slower |
| 16 lanes (`d`) | 16 | **300 – 340** | 2.4 – 2.7× slower |
| **8 lanes (`c`)** | 32 | **190 – 250** | **1.5 – 2.0× slower** |

If 8 lanes lands in 190–250 ms, the residency model is confirmed and the
remaining 1.24–1.58× is exactly what S3 (PGS row bodies, 28% of ops) plus a
*correctly written* sweep is scoped to deliver — with the register budget
untouched. If it lands near 614/4 = 153 ms, better than predicted. **If it
lands above 400 ms, the mapping is refuted independent of occupancy and the
project should stop.**

**Two prerequisites before any sub-warp S2 re-measure** (both are S2 bugs, not
config issues, and both are cheap):

1. Pack the shared factor to `NV(NV+1)/2 = 630` words (`:5325`) — the dense
   `NV*NV` tile does not fit 16 envs/block under the 48 KB static limit, and
   it halves the 39-envs/SM shared ceiling to 20.
2. Template `chol_solve_n` on a compile-time `N` and rebuild the unit-RHS seed
   at `:5684` as an unrolled predicated write, so `T x[NV]` (`:5660`) lives in
   registers. As written the sweep runs 32 lanes × ~1,190 **local**-memory
   dependent accesses where the serial code ran one lane's — a 32× traffic
   amplification that alone plausibly accounts for S2's +3.4%.

### What not to do

- Do not chase 255 → 210 → lower on the 32-lane mapping. The rungs are at
  168 / 128 / 80 / 64 (§1.2) and even the 40-reg floor caps at 48 envs/SM.
- Do not split the tick into serial + parallel kernels. It is ≥16 launches
  and ~6.5 GB/tick of hand-off, and it leaves the dependent chain — the
  actual binder — untouched (§3.4).
- Do not add `__noinline__` boundaries for allocation control. CUDA has no
  per-function register windows; the kernel's occupancy takes the max.
