# Warp rescue — marshalling elimination and data residency on the serial chain

Lens: every word that crosses a lane boundary, and every byte still
lane-0-resident, priced against the serial critical path. Sources:
`beat_arena_cuda.cu` (S2 at `1aa0989`),
`design/WARP_ENV_{SHELL,SOLVER_CORE,RECONCILIATION}.md`, `PERF_AUDIT`.

**Headline: the marshalling arithmetic does NOT explain the regression, and
fixing the marshalling would not have rescued S2.** Marshalling is 5.9% of the
substep's instructions and ~16% of the miss. The dominant terms are (a) the
`__shared__` tile buying a 64 KB L1 carveout on an SM whose binding resource
is L1 residency of the lane-0 frame, and (b) a 45-reg allocator drop on a
still-serial body. Both follow from one fact the staging plan inverted:
**`d.m` was already the most cache-resident object in the frame — copying it
to shared spent cache capacity relocating data that was not the problem.**

---

## 1. The measurement frame

Occupancy and wave counts, needed before any per-substep number means
anything.

| | legacy `step_kernel` | warp S1 | warp S2 |
|---|---|---|---|
| mapping / threads per block | 1 env / thread, 32 | 1 env / warp, 128 (`:6674`) | same |
| launch bounds | `(32, 1)` | `(128, 1)` (`:6702`) | `(128, 1)` |
| regs | 254/255 (`PERF_AUDIT:19`) | 255 | **210** (`1aa0989`) |
| static shared/block | 0 | **0** (`2a962d0`: no `__shared__`) | **19,600 B** (`:6719`) |
| blocks/SM (65,536 regs) | 4 (grid-limited) | 2 | 2 |
| **warps/SM — envs resident/SM** | 4 — **128** | 8 — 8 | 8 — 8 |
| waves at N=16384, 128 SM | 1 | 16 | 16 |
| ms/tick ⇒ per-env-tick latency | 124.0 ⇒ 124.0 (per 32 envs) | 614.0 ⇒ 38.4 | 634.6 ⇒ 39.7 |

Two facts fall straight out and both matter more than anything in §2:

1. **The warp kernel's per-env chain is already 3.2× faster than legacy's**
   (124.0 → 38.4 ms). S1 changed no instruction; it changed the per-warp
   working set from 32 × 42 KB = 1.34 MB to 1 × 42 KB. The 5× loss is purely
   the 16 waves.
2. **Legacy's local accesses are perfectly coalesced; the warp kernel's are
   not.** Local memory is warp-interleaved, so legacy's 32 lanes executing
   `d.m[c*NV+r]` at the same `(c,r)` issue **one 128 B transaction serving 32
   envs**. S1/S2 issue **one 32 B sector serving 4 useful bytes for one env**.
   Nothing in S1 or S2 recovers any of that. It is recovered only when the
   *frame* distributes across lanes (§7), not when the *factor* moves to
   shared.

Per-substep budget used throughout, rescaled from `SOLVER_CORE §4.6` (nominal
ng=8, nl=8, W=33) to the **measured** workload — ng mean 2.14
(`RECONCILIATION:60-64`), nc mean ~0.15 / zero 92.7% (`PERF_AUDIT:80-84`), nl
mean 7.8 range 3-14 (`:3646`) ⇒ **W = 14** sweep RHS + 1 step-6 RHS:

| phase | nominal ops | realized ops | share | scales with |
|---|---:|---:|---:|---|
| CRBA (seeds + fill + merge) | 7,690 | 7,690 | 13.4% | — |
| Cholesky factor | 7,175 | 7,175 | 12.5% | — |
| Triangular solves (15 RHS) | 39,270 | 17,850 | 31.0% | W |
| RNEA (both passes) | 5,400 | 5,400 | 9.4% | — |
| Row assembly + diag dots | 7,000 | 3,063 | 5.3% | rows |
| PGS (12 iters) | 26,000 | 11,375 | 19.8% | rows |
| step-11 clamps | 5,000 | 5,000 | 8.7% | — |
| **total** | **92,700** | **57,553** | | |

Dependent-step budget at realized workload ≈ **39,108** (solves counted as 5
chains, §3). This is the currency: the kernel is latency-bound at ~2% issue
utilization, so wall time tracks the dependent chain, not the op count.

---

## 2. Marshalling census — exact counts from the S2 code

Per substep, FC (contact) mode, realized ng=2 / nc=0 / nl=8 ⇒ `nwr=0,
ngr=6, ncr=0, nlr=8, wtot=14`, one round of the tiling loop (`:5661`).

| site | code | per-substep count | lane utilization |
|---|---|---:|---|
| factor staging, local loads | `:5339-5341` `src[c*NV+r]` | **630** | 1/32 |
| factor staging, shared stores | `:5339-5341` `sm[c*NV+r]` | **630** | 1/32 |
| RHS publish, `__shfl_sync` | `:5356` × 6 ground rows | **210** | 32/32 |
| RHS publish, limit unit cols | `:5682` 1 shfl each × 8 | **8** | 32/32 |
| RHS publish, lane-0 local loads | `:5354` `src[i*stride]` | **210** | 1/32 |
| RHS publish, `x[i]` stores | `:5357` `if (lane==w)` | **218** | 1/32 |
| RHS zero fill | `:5669` | **35** | 32/32 |
| collect, `x[i]` loads | `:5369` (all lanes load) | **490** | 32/32 |
| collect, `__shfl_sync` | `:5369` × 14 RHS | **490** | 32/32 |
| collect, lane-0 `dst` stores | `:5370` | **490** | 1/32 |
| count broadcasts | `:5651-5654` | **3** | 32/32 |
| `__syncwarp` | `:5342,5646,5691,5693,5711,5737,6185` | **7** | — |
| **total added instructions** | | **≈ 3,414** | |

`3,414 / 57,553 = 5.9%` of the substep's instruction count. The ones that can
stall on the frame are the **1,548 single-lane local accesses** (630 staging
loads + 210 publish loads + 218 `x` stores + 490 collect stores); the other
~1,866 are shuffles (register-file latency, pipelined — every `warp_rhs_*`
loop body is independent across `i`) and shared stores.

Against the 39,108-step dependent chain the single-lane class is **4.0% of
steps**. Weighting shuffles at ~1/20 of an average step's stall and the
single-lane accesses at ~1× gives a marshalling band of **1.5 – 3.0% of S1 =
9 – 18 ms/tick**. Take **+12 ms** as the point estimate.

Note the shape: the publish and collect loops are `for (w = 0; w < wend; w++)`
at `:5671` / `:5695` — **serial in W on lane 0's side**. Marshalling grows
linearly in the sweep width while the solve it feeds is O(1) in W. That is the
structural defect of the hand-off, and it is why the design's own cost note
(`:5296-5305`, "~4,600 hand-off instructions against ~120,000 removed") is
right about the ratio at W=33 and wrong about the workload: at the *measured*
W=14 the hand-off is 3,414 against ~3,570 removed (§3).

---

## 3. What the sweep actually removed — the batching correction

The S2 header (`:5173-5177`) prices the removal as "~33 × (1,190 dependent
FMAs + ~2,450 local loads)". **That over-counts by ~4× because S1 was already
batched.** The legacy ladder S1 ran (`:3556-3676`) groups the RHS 4/2/1 and
each group is a loop interchange sharing one chain and one set of factor
loads.

At realized ng=2 / nc=0 / nl=8, S1 issued exactly **five** solve chains:

| chain | entry point | line | RHS |
|---|---|---|---:|
| ground 4-wide | `chol_solve_multi_n<4>` | `:3561` | 4 |
| ground 2-wide | `chol_solve_multi_n<2>` | `:3572` | 2 |
| limits 4-wide (×2) | `chol_fwd_unit4_n` + `chol_back4_n` | `:3654-3656` | 4+4 |
| step-6 `M⁻¹(τ−C)` | `chol_solve_n` | `:3954` | 1 |

S2 replaces the first four with **one** sweep chain and leaves step-6 serial
(`:6037`, excluded by the dependency audit at `:5245-5252`). So:

| quantity | S1 | S2 | delta |
|---|---:|---:|---:|
| solve chains on the serial path | 5 | 2 | **−3** |
| dependent FMA levels (≈1,190/chain) | 5,950 | 2,380 | **−3,570** |
| factor loads on the chain | 5,950 (lane-0 local) | 630 staging + 1,190 shared broadcast | **−4,130 local** |
| total FLOPs | unchanged (14 RHS either way) | unchanged | 0 |

**Expected saving:** `3,570 / 39,108 = 9.1%` of the dependent chain ⇒
**−56 ms** off 614.0, plus a second-order credit for moving 4,130 single-lane
local factor loads to broadcast shared reads.

**Measured: +20.6 ms.** The miss is therefore **≈ 77 ms = 12.5% of S1**, not
20.6 ms. Any explanation that only accounts for +20.6 is under-explaining by
4×.

---

## 4. Attribution of the 77 ms miss

| # | term | mechanism | estimate | basis |
|---|---|---|---:|---|
| 1 | **shared-tile L1 carveout** | 19,600 B/block × 2 blocks/SM = 39,200 B ⇒ driver carveout 64 KB ⇒ **L1 128 → 64 KB** | **+45 to +65 ms** | residual; direction & magnitude checked below |
| 2 | **register drop 255 → 210** | 45 fewer regs on a still-serial body ⇒ more frame round-trips (0 spills reported, so this is lost promotion, not spill code) | **+10 ms** | measured elasticity `PERF_AUDIT:87-91`: 254→168 (−86 regs) = +3.1%; linear ⇒ −45 regs ≈ +1.6% |
| 3 | **marshalling** | §2 | **+12 ms** (band 9–18) | §2 |
| 4 | whole-warp execution of the solve | 31 extra lanes now run `chol_solve_n` on their own `x[35]`; +4,480 B/warp hot local, ~2,400 warp-wide accesses/round | **+2 ms** | warp-wide accesses are coalesced 128 B; mostly offset by the removed single-lane loads |
| | **sum** | | **~77 ms** | |

### 4.1 Why term 1 is the dominant one

`step_kernel_warp` declares `__shared__ T smem[4 * 1225]` = 19,600 B/block
(`:6719`). At 2 blocks/SM the SM reserves 39,200 B of the 128 KB unified
L1/shared array; Ada's carveout granularity is {0, 8, 16, 32, 64, 100} KB, so
the driver picks **64 KB shared / 64 KB L1**. S1 requested zero shared and ran
with the full **128 KB L1**. `SHELL §1.3` already flagged this ("96 KB
shared/SM leaves only 32 KB of L1 out of 128 KB — a real cost the kernel does
not pay today") and deferred the carveout A/B to stage 6. **It arrived at
stage 2 instead.** Per-SM hot local working set:

| | S1 | S2 |
|---|---:|---:|
| lane-0 frames (8 warps × 42 KB) | 336 KB | 336 KB |
| sweep `x[35]` × 32 lanes × 8 warps | 0 | +36 KB |
| **hot local** | **336 KB** | **372 KB** |
| **L1 capacity** | **128 KB** | **64 KB** |
| capacity / working set | 0.38 | 0.17 |

The phases that lose are those with a large per-phase hot subset: CRBA
touches `d.pos`/`d.r`/`d.ci`/`d.cc`/`d.m` ≈ 10 KB per env — 8 envs = 80 KB,
which fits 128 KB L1 and does **not** fit 64 KB. CRBA + RNEA + assembly =
16,153 ops = **28% of the realized substep**. If their local accesses go from
~50% L1 hit (≈170 cy blended) to ~20% (≈248 cy), that group slows ~46%, i.e.
**+12.9% of the substep = +79 ms** — more than sufficient to cover the 45–65
ms residual, with the solve phase's move to guaranteed-hit shared reads
clawing part of it back. Counters are locked on this pod
(`PERF_AUDIT:22-24`), so this is a bounded model, not a measurement; §8 is
the experiment that falsifies it.

**The finding of this lens, exactly:** `d.m` is 4,900 B of a 42 KB frame read
~7,000× per substep by `chol_factor_n` alone — by a wide margin the most
L1-resident object the kernel had. S2 spent 64 KB of L1 to give it a
*guaranteed* home and evicted the CRBA/RNEA working set, which had no other
home, to pay for it. **A residency regression dressed as a residency
improvement.**

### 4.2 What this does NOT say

Not that marshalling is free — that eliminating 100% of it (§5, §7) recovers
~12 ms against a 490 ms gap to legacy. **Marshalling elimination is necessary
hygiene, not a rescue.**

---

## 5. Build-in-place: can the factor be born in shared?

### 5.1 What the producers need

| producer | lines | change needed |
|---|---|---|
| `crba` — 6 write sites: zero fill `:1907`, diagonal `:1955`, ancestor pair `:1975`, root rows `:1988-1990`, armature `:2019-2021`, root 6×6 `:2030-2040` | `:1904-2040` | fork to take `T* m`; textual `d.m` → `m` |
| `chol_factor_n` | `:6025` | **none** — signature is already `(int n, T* a, bool* bad)`; pass `sm` |

`crba` needs a mechanical fork (`crba_into(Dyn<T>&, T* m)`), textually a
`d.m` → `m` substitution — the same "diff is empty" instrument the S2 forks
already use (`:5379-5382`).

**Aliasing / `__restrict__`:** no hazard. `sm` is `__shared__`, `d` is local —
distinct address spaces, so `__restrict__` on both is trivially true, and the
in-place aliasing `chol_factor_n` relies on (`:2255-2258`) is preserved (`a`
and `bad` stay distinct objects).

**Initialization:** `crba`'s zero fill covers exactly lower+diagonal
(`:1907`) — exactly the set `warp_stage_factor` copies (`:5339-5341`) and
exactly the set every reader touches (`:1888-1893`). The tile is fully
written every substep; no stale-data hazard across substeps or across the
envs a warp visits sequentially.

### 5.2 What breaks — the later serial readers

After the sweep, exactly **two** sites still read the factor out of lane 0's
local `d.m`, and both are inside `if (lane == 0)` blocks:

| site | line | path | frequency |
|---|---|---|---|
| `ensure_pcol` → `ensure_ecol` → `chol_solve_unit_n(NV, d.m, ...)` | `:5986` → `:3051` | step-8 ω-cap | "0..29, usually ~0" (`:5254`) |
| `ensure_ecol` → same | `:6149` → `:3051` | step-11 clamp | **0.12% miss rate** (K16, `:891-900`) |

Both are cold. Both need only a 3-line fork that threads the factor pointer
(`chol_solve_unit_n` already takes `const T* __restrict__ l`). No other reader
exists — I grepped the whole warp fork region (`:5378-6260`): the only other
factor consumers are `:5692` and `:6037`, and **both already read `sm`**.

Follow-on: `d.m` becomes write-dead in the warp path and could leave `Dyn`
(frame 42 KB → 37.1 KB, −11.7%). **Do not do this in the same change.**
`PERF_AUDIT:87-91` is an explicit measurement that a 28% frame cut flipped the
allocator 254 → 168 and cost +3.1%. Leave `d.m` declared and unused; change
one thing at a time.

### 5.3 Serial-chain words saved

| item | per substep | per env-tick |
|---|---:|---:|
| staging local loads removed (`:5339`) | 630 | 5,040 |
| staging shared stores removed (`:5340`) | 630 | 5,040 |
| **direct marshalling removed** | **1,260** | **10,080** |
| `crba` output stores relocated local → shared | 630 | 5,040 |
| `chol_factor_n` accesses relocated local → shared | **~7,175** | ~57,400 |

The 1,260 is 37% of the §2 census — worth ~4 ms. **The 7,175 is the real prize
and it is not a marshalling saving at all:** it moves 12.5% of the substep's
ops from an L1/L2 gamble (~170 cy blended) to a guaranteed ~30 cy shared hit,
on a chain with no latency hiding — and it is available *without* any
lane-parallelism.

### 5.4 RHS assembled directly into shared sweep slots

Possible but currently self-defeating. `s.g_J[gi]` is filled by
`point_jacobian_acc` accumulating into `jr` (`:5598-5602`) and is **also**
read by the diag dots (`:5721`) and the PGS row bodies — so the row would have
to *live* in shared, not visit it. That removes 218 publish shuffles + 210
lane-0 loads, and if `g_minv` joins it, the 490 collect shuffles + 490 loads +
490 stores too: **~1,900 of 3,414 marshalling instructions gone**. Cost: 14
rows × 35 words × 4 B = **1,960 B/env** on top of 4,900 ⇒ 54,880 B/SM. Under
§4.1 that is the wrong direction — more carveout, less L1 — unless the factor
tile is packed first (§7).

---

## 6. Stage ledger under lane-0 residency, at the realized workload

Transfer cost is counted as words that must cross the lane boundary per
substep; ops removed are from the §1 realized table.

| stage | ops removed | share | transfer if state stays lane-0-resident | words / op | verdict |
|---|---:|---:|---|---:|---|
| **S3 PGS** | 11,375 | 19.8% | per iteration: publish `v` (35) + per row publish `J`,`M⁻¹Jᵀ` (70) + collect `v` (35). 12 × [35 + 14×105] = **18,060** | **1.59** | **NET NEGATIVE.** Transfer exceeds the work removed. S3 is unbuildable under lane-0 residency. |
| **S4 Cholesky** | 7,175 | 12.5% | **0** with build-in-place (§5); 630 without | **0.00** | **NET POSITIVE, transfer-free.** The only such stage. |
| **S5 CRBA** | 7,690 | 13.4% | FK output: `pos[30][3]` 90 + `r[30][3][3]` 270 + `axw[30][3]` 90 = **450** publish | 0.059 | **Net positive, marginal on the chain** (450 serial shuffles + 450 single-lane loads). Transfer-free only if S6 lands with it. |
| **S6 RNEA + FK** | 5,400 | 9.4% | `q` 29 + `v` 35 + `p` 3 + `quat` 4 = **71** in; output consumed in-warp by S5 | 0.013 | **Net positive.** |
| assembly + diag | 3,063 | 5.3% | reads `d.pos`/`d.r`/`d.p` — same residency as S5/S6 | — | rides S5/S6 |
| step-11 clamps | 5,000 | 8.7% | needs `ecol` + `v` resident — same as S3 | — | rides S3 |
| **{S4, S5, S6}** | **20,265** | **35.2%** | | | |

**The stop rule was computed on the wrong profile.** `RECONCILIATION:56`
states "S4-S6 cover only ~14% of today's ops" — true at the nominal ng=8 /
W=33 profile where solves are 42%. At the **measured** ng≈2 / nc≈0 profile
the workload-independent stages (CRBA, Cholesky, RNEA) are **35.2%**, and
they are precisely the three that pay little or no transfer. The realized
workload made the transfer-free stages *more* valuable and the transfer-heavy
one (S3/PGS) *less* — the opposite of the staging order the plan chose. The
correct order under lane-0 residency is **S4 → S6 → S5 → S3**: each stage
moves the state its successor needs, so nobody pays a publish/collect for
data a later stage will own anyway.

---

## 7. The residency milestone

The smallest set of changes after which the marshalling tax is structurally
zero for the substep inner loop.

**R1 — the factor never lives in lane 0.** `crba_into(d, sm)` fork (§5.1);
`chol_factor_n(NV, sm, &d.pd_bad)` (no signature change); `ensure_ecol_sm` /
`ensure_pcol_sm` forks for the two cold readers (§5.2). **Plus: pack the tile
to the 630 live words** — dense 4,900 B/env vs packed 2,520 B/env ⇒ 20,160
B/SM ⇒ **32 KB carveout, 96 KB L1** instead of 64/64. Under §4.1 that is the
load-bearing half of R1, not a "stage 6/7 probe" (`:5266-5269`); it costs one
address-map change in three readers. Removes 1,260 marshalling instrs/substep,
relocates 7,805 accesses to shared, frees 4,900 B/env of frame (don't cash it
yet).

**R2 — `v` and the constraint rows never live in lane 0.** `v` distributed one
word per lane (32 + 3-word tail, `SOLVER_CORE §4.5`);
`g_J`/`g_minv`/`c_J`/`c_minv`/`ecol` one word per lane per row. Removes **all**
publish/collect — 708 shuffles, 700 lane-0 accesses, 708 stores = **2,116
instrs/substep, 100% of the remaining marshalling**. Unlocks S3 and the
step-11 clamps.

**R3 — frame state distributed (`pos`/`r`/`axw` lane ≡ frame).** Unlocks
S5/S6, removes the 450-word FK publish, and is where the **coalescing** win of
§1.2 finally lands: 32 lanes at one frame offset = one 128 B transaction
instead of 32 single-lane 32 B sectors.

**Minimal milestone = R1 + R2.** After it the only lane-0-resident state the
substep's inner loop touches is `q`/`quat`/`p` and the judgment state, which
cross the boundary once per *tick*, not once per row.

### Serial-chain arithmetic at R1 + R2

| | S2 today | after R1+R2 |
|---|---:|---:|
| marshalling instrs / substep | 3,414 | **0** |
| L1 carveout | 64 KB shared / 64 KB L1 | 32 KB / 96 KB |
| CRBA (serial) | 7,690 | 7,690 |
| Cholesky | 7,175 serial | ~1,300 warp (S4) |
| solves | 2,380 dep levels | 1,190 |
| RNEA (serial) | 5,400 | 5,400 |
| assembly + diag | 3,063 serial | ~700 |
| PGS | 11,375 serial | ~2,900 dep steps |
| clamps | 5,000 serial | ~900 |
| **dependent steps / substep** | **39,108** | **~18,680** |
| **implied ms/tick** (614 × ratio, chain-linear) | 634.6 | **~293** |

Still 2.4× off legacy's 124.0 — because R1+R2 do not touch the 16-wave
deficit. That deficit is an occupancy question, not a residency one (the code
ships `__launch_bounds__(128, 1)` at `:6702` where SHELL recommended
`(128, 4)`, `SHELL:86-95`); it belongs to the occupancy analysis, and the two
compose multiplicatively. Noted here only because **R1 + R2 alone cannot
reach parity and should not be sold as if they could** — the residency
milestone is worth ~2.2× on the chain, and the remaining ~2.4× has to come
from waves.

---

## 8. Verdict and the one experiment

**Refutation.** Lane-0-residency *marshalling* is not the dominant term. It
is 5.9% of the substep's instructions, ~4% of its dependent steps, and I
price it at **+12 ms of a 77 ms miss (16%)**. The dominant term is the
`__shared__` tile's **64 KB L1 carveout** (+45–65 ms), with the 255→210
register drop second (+10 ms). Both are *residency* failures, but of the
cache, not of the lanes: S2 relocated the one object that was already
resident and evicted the objects that were not.

**The experiment: strip the shared tile, keep the sweep.** The only change
that moves term 1 and term 3 in *opposite* directions, so one A/B separates
them. The factor stays in lane 0's local frame and is broadcast by shuffle
inside the solve — legal because every index in `chol_solve_n` is a loop
counter and therefore warp-uniform, bitwise-safe because a shuffle moves a bit
pattern (the argument `:5195-5199` already makes for the copy). If it lands
**≤ 614**, the carveout was the cost and S2's marshalling was noise — proceed
to R1 with a packed tile. If it lands **~634**, marshalling + regs were the
cost and this attribution is wrong.

```diff
--- a/beat_arena_cuda.cu
+++ b/beat_arena_cuda.cu
+// STAGE 2b — the factor is NOT staged. It stays in lane 0's local frame and
+// every element is BROADCAST at the point of use. Legal: n, i and k below are
+// loop counters, hence warp-uniform, hence every lane reaches every shuffle
+// with the full mask. Bitwise: __shfl_sync moves a bit pattern; the arithmetic
+// is textually chol_solve_n's, on the same values, in the same order.
+#define WBC(e) __shfl_sync(WARP_FULL, lane == 0 ? l[e] : T(0), 0)
+template <typename T>
+__device__ void chol_solve_n_bcast(int n, const T* l, T* __restrict__ x,
+                                   int lane) {
+  /* body of chol_solve_n (:2418-2460) verbatim, every `l[E]` -> `WBC(E)` */
+}

 // ---- substep_warp -----------------------------------------------------
   chol_factor_n(NV, d.m, &d.pd_bad);
   }
-  warp_stage_factor<T>(d.m, sm, lane);          // 630 loads + 630 smem stores
+  __syncwarp(WARP_FULL);   // factor final in lane 0's d.m; nothing to publish
   if (lane == 0) {
-    chol_solve_n(NV, sm, d.rhs);
+    chol_solve_n(NV, d.m, d.rhs);               // back to local, as in S1

 // ---- constraint_solve_warp --------------------------------------------
-    chol_solve_n(NV, sm, x);
+    chol_solve_n_bcast<T>(NV, d.m, x, lane);

 // ---- step_kernel_warp -------------------------------------------------
-  __shared__ T smem[ENVS_PER_BLOCK * WARP_SM_WORDS_PER_ENV];
-  T* const sm = smem + warp * WARP_SM_WORDS_PER_ENV;
+  T* const sm = nullptr;      // stage 2b: zero static shared => 128 KB L1
```

Instruction ledger for the arm (per substep): **removes** 630 staging loads +
630 shared stores; **adds** 1,190 lane-0 local factor loads + 1,190 shuffles
inside the single sweep solve — versus S1's **5,950** such loads across five
chains, so the factor-traffic win of S2 is kept at 5×. Net instruction delta
≈ +1,120, net shared-memory delta **−19,600 B/block**, net L1 delta
**+64 KB/SM**. Census gate first, per the standing rule: if the arm does not
restore ≥ 240 regs the register term is confounded and the result is
uninterpretable.

**Second arm, if the first wins:** R1 with a packed 630-word tile (§7) —
build-in-place, 32 KB carveout, and `chol_factor_n`'s 7,175 accesses moved to
guaranteed-hit shared. Then R2. The occupancy pin (`:6702`) is a separate,
orthogonal lever and composes multiplicatively with both.
