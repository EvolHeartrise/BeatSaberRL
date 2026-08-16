# Warp rescue — the legacy side (devil's advocate)

**Question asked:** the warp-per-env rewrite stalled (in fact its stop rule *fired*
— `sessions/2026-08-14-v17-lr-ladder.md:382-399`, commits `1aa0989` / `064fd3d`).
What do the effort's *own measurements* say about making the LEGACY thread-per-env
kernel faster than 124.0 ms/tick?

**Answer in one line:** the effort measured an occupancy curve that has not
flattened, then concluded "occupancy is not the lever" from experiments that were
all run in regimes where occupancy could not vary. Three of the four cheapest
items below have never been measured on sm_89 at all.

Scope note: this doc reads code and evidence only. No source file was modified.

---

## 0. The two walls, and why they coincide at exactly one point

Legacy `step_kernel` is one thread per env, 32-thread blocks
(`beat_arena_cuda.cu:929`, `:5138-5139`, launch at `:6899-6905`), so on a 128-SM
RTX 4090 **resident envs/SM = threads/SM** and **blocks requested per SM =
(N/32)/128 = N/4096**. sm_89 envelope (`WARP_ENV_SHELL.md:82-83`): 65,536 regs/SM,
48 warps/SM, 24 blocks/SM, 128 KB unified L1+shared. Register allocation
granularity is 8 regs/thread, so warps/SM = floor(65536 / (ceil(regs/8)·8·32)).

### Table A — the two ceilings

| N | blocks | blocks/SM requested | envs/SM | warps/SM | warps per **scheduler** (4/SM) | binder |
|---:|---:|---:|---:|---:|---:|---|
| 4,096 | 128 | 1 | 32 | 1 | 0.25 | grid |
| 8,192 | 256 | 2 | 64 | 2 | 0.5 | grid |
| **16,384 (production)** | 512 | **4** | 128 | **4** | **1.0** | **grid** |
| 32,768 | 1,024 | **8** | 256 | 8 | 2.0 | **grid = registers** |
| 49,152 | 1,536 | 12 | 384 | 12 | 3.0 | registers (255) |
| 65,536 | 2,048 | 16 | 512 | 16 | 4.0 | registers (255) |
| 98,304 | 3,072 | 24 | 768 | 24 | 6.0 | registers + 24-block cap |

At production N the SM's four warp schedulers hold **exactly one warp each**.
That is the cleanest statement of "latency-bound, parallelism-starved"
(`PERF_AUDIT_2026-08-16.md:77-81`): there is literally no second warp to switch
to on a stall.

Two independent walls sit at N=32,768 — the grid runs out of blocks at the same
point the 255-reg pin runs out of register file. **Below that point the register
target cannot buy occupancy; above it, N alone cannot.**

### Table B — measured N-ladder and its exponent

`PERF_AUDIT_2026-08-16.md:77-81` (stock kernel, ms/tick):

| N | envs/SM `k` | ms/tick | µs/env | env-ticks/ms | Δ vs prev | fitted `a` in `T ∝ k^a` |
|---:|---:|---:|---:|---:|---:|---:|
| 4,096 | 32 | 56.1 | 13.70 | 73.0 | — | — |
| 8,192 | 64 | 79.6 | 9.72 | 102.9 | **+41.0%** | 0.495 |
| 16,384 | 128 | 124.0 | 7.57 | 132.1 | **+28.4%** | 0.361 |
| 32,768 | 256 | 194.0 | 5.92 | 168.9 | **+27.9%** | **0.355** |

The exponent is *stable* across the last two rungs (0.361 → 0.355). The curve is
still climbing at the point where the hardware stops it. Extrapolating `a`:

| next rung | envs/SM | `a = 0.355` | `a = 0.25` (pessimistic) | needs |
|---|---:|---:|---:|---|
| 12 blocks/SM | 384 | +15.5% | +10.7% | N≥49,152 **and** ≤168 regs |
| 16 blocks/SM | 512 | +27.9% | +18.9% | N≥65,536 **and** ≤128 regs |
| 24 blocks/SM | 768 | +47.7% | +31.6% | N≥98,304 **and** ≤80 regs |

For comparison, the warp rewrite's single most defensible number
(`WARP_ENV_SOLVER_CORE.md:617-620`) was **1.6–2.0×** attributed to "occupancy
4 → 16 warps/SM", extrapolated from this same slope. That is the same commodity
priced in Table B — and `__launch_bounds__(32, 16)` is a one-token purchase of it.

---

## 1. The occupancy ceiling — and the premises of the do-not-retry entries

### 1.1 What the register targets buy

| `__launch_bounds__(32, K)` | ptxas target (65536/(32K)) | regs/warp | warps/SM | envs/SM | N for full residency | local-mem reservation (128 SM × thr/SM × 42,224 B) |
|---|---:|---:|---:|---:|---:|---:|
| `(32, 1)` **today** | ≤255 (per-thread max) | 8,192 | 8 | 256 | 32,768 | 1.38 GB |
| `(32, 8)` | 256 → 255 | 8,192 | 8 | 256 | 32,768 | 1.38 GB |
| `(32, 12)` | 170 → **168** | 5,376 | 12 | 384 | 49,152 | 2.08 GB |
| `(32, 16)` | **128** | 4,096 | 16 | 512 | 65,536 | 2.77 GB |
| `(32, 24)` | 85 → **80** | 2,560 | 24 (block cap) | 768 | 98,304 | 4.15 GB |

Note `(32, 8)` is *arithmetically identical to today's pin* — both resolve to 255.
That makes it a free NULL control rung: it must measure zero, and if it doesn't,
the harness is broken.

**What N makes each rung meaningful (the question the ladder must answer first):**
a rung is a no-op-or-worse unless the grid actually supplies its blocks. At
N=16,384 *every* rung K>4 is strictly downside — the register cut cannot add a
single resident warp, so only its ILP/spill cost is observable. At N=32,768,
every rung K>8 is strictly downside for the same reason. The K=12 rung first has
upside at N=49,152; K=16 at N=65,536.

### 1.2 The `maxrregcount` do-not-retry entry, and its premise

`PERF_AUDIT_2026-08-16.md:169-171` lists `maxrregcount` on the do-not-retry list.
The evidence is `beat_arena_cuda.cu:913-925`:

> The register budget is NOT insensitive, correcting the earlier
> "-maxrregcount=64 vs the natural 128: within noise" note (measured 2026-08-03):
> **left free**, ptxas picks the target from a heuristic that is unstable under
> unrelated frame changes … step_kernel therefore PINS the budget via
> `__launch_bounds__(THREADS, 1)`: minBlocksPerMultiprocessor = 1 is the honest
> statement of this kernel's design (**occupancy is not the lever**) …

Three things about that entry:

1. **Era.** sm_61, 20-SM GTX 1080, per the PROVENANCE WARNING at
   `beat_arena_cuda.cu:98-107`. The GTX 1080 had 2 MB of L2 for 20 SMs
   (100 KB/SM); the 4090 has 72 MB for 128 SMs (576 KB/SM) — **5.8× more L2 per
   SM**, which is the cache that makes spills cheap.
2. **The measurement N.** The K-series micro-bench standard was N=4096
   (`:135-141`). On 20 SMs that is 128 blocks / 20 = **6.4 blocks/SM against a
   residency ceiling of 8** — grid-starved; fewer registers could not add a
   resident warp there either. The other cited occupancy datum, `:109-112`
   (N=1024), is 1.6 blocks/SM. **Every experiment behind "occupancy is not the
   lever" was run where occupancy was pinned by grid supply.**
3. **What it actually argues.** Read literally, `:913-925` is an argument *for
   pinning* and against ptxas free choice. `__launch_bounds__(32, K)` pins just
   as hard as `(32, 1)` — same mechanism, different constant — so the stated
   failure mode ("ptxas's free-choice heuristic flips to 64 regs") **cannot occur
   under any rung of the ladder.** The entry does not cover this experiment.

### 1.3 The NC_CAP result is not evidence against the ladder

`PERF_AUDIT_2026-08-16.md:86-91`: `NC_CAP=16` cut the frame 42,224 → 30,288 B and
measured **+3.1% slower** (124.0 → 127.8), because "ptxas allocation flipped
254 → 168 regs".

That build had 168 regs — i.e. **12 blocks/SM of residency capacity** — and was
timed at **N=16,384, where only 4 blocks/SM are requested.** The entire upside of
a 168-reg build was structurally invisible to the measurement. What was measured
is the pure downside: fewer registers, worse ILP, at unchanged occupancy. The
verdict as written ("DO NOT RETRY frame-capacity or packed-triangle cuts —
they shrink bytes the kernel never waits on") is correct *for capacity cuts* and
says nothing about a deliberate occupancy purchase.

Two further caveats on that 3.1%: **it is inside the campaign's own baseline
drift** — the same kernel at the same N is reported as 146 (`:17`), 136.9 / 138.3
/ 135.7 (`:60-61`), 134.07 / 133.75 (`:47`), 124.0 (`:77`) and 122.5
(`WARP_ENV_RECONCILIATION.md:66`) ms/tick, so a 3.8 ms delta sits inside a ±20 ms
band; and the sm_89 census already reports a **168-reg, zero-spill instantiation**
of essentially this kernel — "254 regs (write) / 168 (read-only), 42 KB stack
frame, ZERO spills" (`:19-21`). Whatever "read-only" denotes, a 42 KB frame
plainly does not *force* 255 registers.

### 1.4 The honest counter-argument

Two, and they are serious.

* **Spills.** K6 (`beat_arena_cuda.cu:203-215`) records a **27.62%** swing from a
  spill-cliff crossing (255 regs + 4.6 KB spill ↔ 249 regs + zero spill) — sm_61,
  100 KB L2/SM. On sm_89 spilled local goes to a 128 KB L1 and a 576 KB/SM share
  of L2, but nobody has measured the cliff height here, and 128 regs against a
  42 KB frame with no lane-splitting will spill *a lot*. This is why the ladder
  must gate on the ptxas census before it gates on time.
* **The bandwidth hypothesis.** `WARP_ENV_SOLVER_CORE.md:779-791` (open question
  4) estimates ~70 GB/tick at N=16,384 = ~565 GB/s, "~58% of the 4090's
  roofline", and notes the pool-residency probe "only cleared the *global pool*,
  not the local frame". If so, the same traffic at N=32,768 is 140 GB / 194 ms =
  **722 GB/s**, and the 0.355 exponent is not latency saturation at all — it is a
  bus filling up, and every rung above 8 blocks/SM is a mirage plus spill traffic
  on a saturated bus. ncu is locked (`ERR_NVGPUCTRPERM`, `:22-25`), so counters
  cannot settle it. **The ladder is simultaneously the cheapest available
  discriminator between the two hypotheses** — that is most of its value.

---

## 2. L1 / shared carveout

The legacy kernel uses **zero shared memory** (`WARP_ENV_SHELL.md:88`, "today"
row). `cudaFuncSetAttribute` appears nowhere in `beat_arena_cuda.cu`; the only
carveout discussion in the corpus (`WARP_ENV_SHELL.md:116-122`) is a stage-6 plan
for the *warp* kernel that was never reached; build flags are `-arch=sm_89
-Xptxas -v` and nothing else (`jit_cache.py:54`). The legacy kernel's carveout has
never been set, read back, or measured.

### Table C — hot working set vs L1, from the struct layout

Frame arithmetic from `Dyn` (`:1555-1588`), `WeldRows` (`:1614-1657`) and
`Solver` (`:1681-1707`), f32, weld mode:

| block | bytes | touched every substep? |
|---|---:|---|
| `d.m[35·35]` (M, then its in-place Cholesky L) | 4,900 | **yes — the hottest object** |
| FK `r`/`pos`/`axw`/`r0` + `iw`/`iw0` | 2,952 | yes |
| CRBA `cm`/`cc`/`ci`/`b*` + RNEA `om`/`al`/`ao`/`fb`/`nb` | 3,412 | yes |
| state `p`/`quat`/`q`/`v` + `c`/`rhs` | 564 | yes |
| weld `mw`/`jwc`/`jwr`/`bias_w`/`aww`/`rw` | 2,688 | yes (weld mode) |
| **hot subtotal** | **≈ 14.5 KB** | |
| `ecol[29][35]` + `pcol[29][35]` | 8,120 | lazily, `nl`+cap columns only |
| `c_J[55][35]` + `c_minv[55][35]` | 15,400 | **only `nc` rows** — measured `nc ≤ 8`, 92.7% zero (`:82-85`) |
| other `c_*`, `l_*`, flags | ~1,500 | partially |
| struct total (computed) | 40,379 | vs 42,224 B measured — 95.6% accounted |

This independently reproduces the audit's finding that capacity is free: ~21–25 KB
of the 42 KB frame is allocated and never touched.

### Table D — can L1 hold anything?

| envs/SM | hot set/SM (14.5 KB each) | vs 128 KB L1 | `d.m` alone/SM | vs 128 KB L1 |
|---:|---:|---:|---:|---:|
| 32 | 464 KB | 3.6× over | 157 KB | 1.2× over |
| 128 (production) | 1.86 MB | 14.5× over | 627 KB | 4.9× over |
| 256 | 3.71 MB | 29× over | 1.25 MB | 9.8× over |
| 512 | 7.43 MB | 58× over | 2.51 MB | 19.6× over |

Concurring with `WARP_ENV_SOLVER_CORE.md:76-80` ("`d.m` alone is 4,900 B/thread =
157 KB/warp … cannot stay resident"). **A single warp's mass matrices already
exceed the entire L1.** Maxing the carveout cannot make the hot set resident at
any occupancy the legacy kernel can reach.

**Verdict on angle 2: expected ≈ null on its own, but it is a precondition, not a
lever.** Two lines, one build. Its value is (a) *verifying* rather than assuming
that a zero-shared kernel got the 128 KB default — `cudaSharedmemCarveoutDefault`
is a driver heuristic, not a contract, and nobody has read it back; and (b) spill
working sets are small, short-lived and reuse-dense, the one class of local
traffic L1 serves well — ~512 B live spill × 128 threads = 64 KB fits a 128 KB L1
and does **not** fit the 32 KB minimum carveout. Running the register ladder
without first pinning the carveout risks measuring a handicapped ladder and
calling it a refutation.

---

## 3. Is N separable from the PPO batch? — **Yes, completely.**

This is the angle where the project has been more pessimistic than its own code
warrants.

* Batch is defined *once*, as a product: `train_batch.py:384-386` —
  `N, T = args.num_envs, args.horizon` / `batch = N * T` /
  `mb_size = batch // args.minibatches`. The recipe constant is a **batch**, not
  an N: `RECIPE_BATCH = 1024 * 128` (`:61`), and the guard at `:439-442` checks
  only the product. Both are plain CLI flags with buffers shaped `(T, N)` at
  runtime (`:93-94`, `:576-588`).
* The policy is **purely feedforward** — no recurrent state (`policy.py:380`,
  `:521`; `B = obs.shape[0]` at `:387`). T has no meaning to the forward pass.
  Advantage normalisation is per-minibatch over the shuffled flat batch
  (`:783-787`, `:798`, `:805-806`), so it never sees the T/N split.
* **Precedent already in-repo**: `v10_shape_ladder.py:37-40` runs
  `(4096,32) / (8192,16) / (16384,8)` — "batch 131,072 each … ONLY the rollout
  shape changes".

So `--num-envs 32768 --horizon 4` gives batch 131,072 with **zero code changes**,
identical `num_updates`, identical `mb_size`, identical checkpoint indices.

**The end-to-end numbers are already measured** (`v10_shape_ladder.py:3-6`): 116k
steps/s at 4096/32 → 171k at 8192/16 → **218k at 16384/8**, i.e. +47% then +27%
*end-to-end wall* at constant batch. Extrapolating the env-phase +28% through the
71/29 rollout/update split (`PERF_AUDIT:15`) with step_kernel at 64% of wall gives
`1/(0.64/1.28 + 0.36)` = **+16.3% wall** for 32768/4. (The audit contradicts
itself here: `:99-102` says +28% env-phase, `:158` says "measured +10% only …
Skip." The `:158` figure is unreconciled with the ladder's own +27% rung.)

**The real cost, quantified** — with γλ = 0.89669, trace weight past the horizon
is `(γλ)^T`:

| shape | T | trace weight past horizon | wall throughput |
|---|---:|---:|---|
| 4096 × 32 | 32 | 3.05% | 116k |
| 8192 × 16 | 16 | 17.4% | 171k |
| **16384 × 8 (today)** | 8 | **41.8%** | 218k |
| 32768 × 4 | 4 | **64.7%** | ~253k (est.) |
| 65536 × 2 | 2 | **80.4%** | — |

The marginal wall win is shrinking (+47%, +27%, ~+16%) while the bootstrap
reliance is growing steeply. **32768/4 is a materially worse trade than the rung
the project already took, and 65536/2 is off the table for training.**

**But:** `--bench` bypasses the recipe guard entirely (`:439`, `if not args.bench
and batch != RECIPE_BATCH`). Throughput at N=49,152 and N=65,536 can be measured
today with no recipe implication whatsoever. The *measurement* needs no gate; only
adoption does. This is what makes the register ladder cheap.

**Two env pools stepped alternately: genuinely inseparable.** Alternating pools
against the same frozen weights advance T/2 steps each over their own N columns —
which is `(T/2, 2N)`, arithmetically identical to N'=2N, T'=T/2. Same batch, same
GAE truncation, same buffers: it is the halve-the-horizon scheme wearing a hat.
The only thing it adds is *pipelining* (pool B's env kernel overlapping pool A's
policy forward), a scheduling win with a small ceiling — rollout is 71% of wall
and step_kernel is 90% of that, so only ~7% of wall is available to hide, and the
29% update phase cannot overlap at all since it depends on the completed rollout.
Listed under Tier 4 as "two-chain rollout graph (RNG-stream shift only — accepted
class)" (`PERF_AUDIT:163-165`).

---

## 4. Micro back-ports from the (terminated) warp work

### 4.1 The one that matters: right-looking Cholesky is warp-independent

`WARP_ENV_SOLVER_CORE.md:300-323` argues that switching the factorization from
left-looking to **right-looking** is **bitwise identical** — same terms, same
ascending-j order, same seed, same FFMA contraction shape as `:2292-2295` —
and cuts the critical path from **595 dependent FMAs** (Σⱼ j over 35 columns)
plus 35 sqrt/divides, to **~140 dependent steps**: "5.7× shorter".

**Nothing in that argument uses the lane mapping** — the chain shortening is a
property of the algorithm. And it is the legacy kernel's single best-attested win
class: `beat_arena_cuda.cu:129-131` — "A4: the unit-RHS forward substitution,
which cuts the dependent chain **without cutting the factor's traffic**, paid
**+21%**" — versus `:120-128`, where two pure traffic cuts (A1/1c widening, A1/1f
register blocking) measured **zero**.

The caveat the design doc does not state: serially, right-looking writes the whole
trailing submatrix per column (~6,545 RMWs into the 42 KB local frame) where
left-looking accumulates in a register. It trades chain length for local traffic —
which on this kernel's own evidence ("traffic … is simply already hidden behind
the chains", `:132-133`) is the right direction of trade. A gated, single-function
change to `chol_factor_n` (`:2260`), and the highest-expected-value *kernel* item
in the corpus that does not require a rewrite.

### 4.2 The g++ shim equivalences: already exhausted for legacy

`beat_arena_cuda.cu:5204-5217` (200 random SPD f32 factors, zero mismatching
words) verifies three solve equivalences — all between functions the **legacy
kernel already calls** — and the one back-portable class they license, widening
RHS batches, is already refuted (`:118-123`: 4→8 "won ZERO"). Its live warning is
load-bearing though: they hold under `--fmad=true` but **break under
`-ffp-contract=fast`**, so the no-fast-math discipline protects the bitwise gate.

### 4.3 Block size: null for overhead, but it gates the top of the ladder

`beat_arena_cuda.cu:109-112` ("128 vs 32: 216 vs 221 ms at N=1024") is sm_61 at
1.6 blocks/SM — and 128 threads/block was in fact **2.3% faster**. Expected effect
on sm_89: **null for launch overhead**, since K19's null-launch floor (`:565-572`)
measured an entire extra kernel launch per step as free. But at 32 threads/block
the 24-blocks/SM cap stops residency at 768 threads/SM where 64 reaches the full
1,536 — not binding for K ≤ 16. Keep `THREADS = 32` and vary only the second
`__launch_bounds__` argument ("Changing THREADS means re-measuring this", `:924`).

### 4.4 The ILP ladder — the only fixed-N lever

At N=16,384 the register target is provably irrelevant (§0), so the *only*
remaining axis is shortening or multiplying the per-thread dependency chains —
the axis that produced every large legacy win (K5 +8.92%, A1 +33%, A4 +21%, K6
package +33.9%). The build sits at 254 regs / **zero spills**: the ladder was
stopped exactly at the register wall, on sm_61 evidence that a spill costs ~20%.
Widening the legal transforms further (forward-substitution row blocking —
bitwise; the back-substitution prohibition at `:216-231` remains absolute) now
trades registers for spills, and the sm_89 spill price has never been measured.
Same trade-off curve as §1, traversed in the other direction because the operating
point is grid-limited rather than residency-limited.

---

## 5. Verdict

Ranked by (expected % × confidence) / cost-to-measure:

| # | item | expected | confidence | eng. cost | gate cost | score |
|---|---|---|---|---|---|---|
| 1 | **`__launch_bounds__(32,K)` ladder, measured at N=65,536 `--bench`** | 0 today; decides a +16–28% path and re-prices the whole rewrite | med | **3 builds, 1 bench** | **none** (`--bench`) | **highest** |
| 2 | **Right-looking `chol_factor_n`** (§4.1) | 5–20% of env kernel | med-low | 1 function, bitwise-gated | none | high |
| 3 | **N=32768 / T=4** | **+16% wall (already measured)** | **high** | **zero — two CLI flags** | learning gate; 41.8% → 64.7% bootstrap reliance | high, gate-blocked |
| 4 | L1 carveout pin + read-back | ~0 alone | high (that it's ~0) | 2 lines | none | precondition for #1 |
| 5 | ILP ladder past the reg wall (§4.4) | unknown sign, historically the big-win axis | low-med | days | none | speculative |
| 6 | Two-chain rollout overlap | ≤7% wall | med | 2 graph captures | none | small |
| 7 | Block size 64 | ~0 | high | 1 token | re-pins the budget | skip |

**Rank #1 is the ladder, not because it wins wall today — at the production recipe
it wins exactly zero — but because it is the only cheap experiment that
discriminates the latency hypothesis from the bandwidth hypothesis
(`WARP_ENV_SOLVER_CORE.md:779-791`), and that discrimination governs #3, #5, and
any future rewrite.** If the 16-warp arm wins, the legacy kernel has a documented
+28% sitting behind a batch decision and the rewrite is redundant. If it loses,
the 0.355 exponent is a bus filling up, N=32,768 is the last rung, and the project
should stop pricing occupancy entirely.

### Top experiment — diff sketch

```diff
--- a/python/beat_arena_env/csrc/beat_arena_cuda.cu
+++ b/python/beat_arena_env/csrc/beat_arena_cuda.cu
@@ -929
 constexpr int THREADS = 32;
+// LADDER PROBE ONLY (do not commit): K selects the pinned register target.
+//   K=8 -> 255 regs (identical to the incumbent (32,1) pin: NULL control)
+//   K=12 -> 168 regs, 12 blocks/SM     K=16 -> 128 regs, 16 blocks/SM
+#ifndef BEAT_MIN_BLOCKS
+#define BEAT_MIN_BLOCKS 1
+#endif
@@ -5138
-__global__ void __launch_bounds__(THREADS, 1)
+__global__ void __launch_bounds__(THREADS, BEAT_MIN_BLOCKS)
 step_kernel(Sim<T> P) {
```

```diff
--- a/python/beat_arena_env/jit_cache.py
+++ b/python/beat_arena_env/jit_cache.py
@@ -54
-    flags = [f"-arch=sm_{maj}{mnr}", "-Xptxas", "-v"]
+    flags = [f"-arch=sm_{maj}{mnr}", "-Xptxas", "-v"]
+    if os.environ.get("BEAT_MIN_BLOCKS"):          # in the srchash digest
+        flags.append(f"-DBEAT_MIN_BLOCKS={os.environ['BEAT_MIN_BLOCKS']}")
```

Plus, once at extension load (angle 2 precondition, `cuda_env.py:_load_ext`), a
pybind entry calling
`cudaFuncSetAttribute(step_kernel<float,FC>, cudaFuncAttributePreferredSharedMemoryCarveout, 0)`
(0 = prefer shared *minimum* ⇒ maximum L1) and printing the read-back plus
`cudaOccupancyMaxActiveBlocksPerMultiprocessor` — with ncu locked, that API is the
**only** direct occupancy instrument available (`WARP_ENV_SHELL.md:682-685`).

### Pod measurement protocol (isolated 4090, ~90 min)

1. **Census gate first, time second.** For each `BEAT_MIN_BLOCKS ∈ {1, 8, 12, 16}`:
   forced rebuild, capture `<name>.build.log`, record `regs / stack frame /
   spill stores / spill loads` for `step_kernel<float,false>`. Assert
   `K=1` and `K=8` are **byte-identical** — if not, stop, the harness is wrong.
   Record spill bytes for K=12/16; this is the number the whole verdict turns on.
2. **Occupancy read-back** at each K: assert
   `cudaOccupancyMaxActiveBlocksPerMultiprocessor == K`. A rung that does not
   achieve its blocks/SM is not a rung.
3. **Timing**, order-balanced interleaved A/B, `train_batch.py --bench`,
   median-of-5 with ranges reported (the ±20 ms baseline drift makes single runs
   worthless):
   * **N=16,384** and **N=32,768** — controls. Every K above the grid supply must
     measure ≤ baseline; this reproduces the NC_CAP direction and proves the
     ladder is not being read at a grid-limited N.
   * **N=65,536** — **the experiment.** K=8 (255 regs, 8 blocks/SM, 2 waves) vs
     K=16 (128 regs, 16 blocks/SM, 1 wave): same N, same total work, no wave-
     quantization confound. Latency hypothesis predicts K=16 wins by ~2^0.355 =
     **1.28× minus spill cost**; bandwidth hypothesis predicts a wash or a loss.
   * **N=49,152** — optional K=12 rung. Report it but *do not* fit the exponent to
     it: 255 regs gives 1.5 waves there and 168 regs gives 1.0, so wave
     quantization inflates the apparent gain.
4. **VRAM check first**: the K=16 build reserves 128 SM × 512 thr × 42,224 B =
   **2.77 GB** of local memory alone, plus env state and the ~242 MB/leg chart
   pool (`train_batch.py:183`).
5. **No correctness gate needed** — the ladder changes no code path, only ptxas
   allocation. Run `test_v10_parity_cuda.py` at K=16 anyway; it should be bitwise,
   and if it is not, that is a compiler bug worth knowing.

**Falsification condition, stated in advance:** if K=16 at N=65,536 is not at
least **+10%** over K=8 at the same N, the occupancy story is dead for the legacy
kernel, N=32,768 is the terminal rung, and every remaining lever is chain-length
(§4.1, §4.4) or recipe.
