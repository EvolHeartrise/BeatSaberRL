# Warp-per-env rescue — SUB-WARP CONFIGURATIONS

Lens: fewer lanes per env (W ∈ {2,4,8,16}) to recover environment residency
while keeping useful intra-env parallelism; plus the thread-per-env
2-envs-per-thread ILP hybrid. No source file was modified.

Inputs: `python/beat_arena_env/csrc/beat_arena_cuda.cu` (6,953 lines),
`design/WARP_ENV_{SOLVER_CORE,SHELL,RECONCILIATION}.md`,
`PERF_AUDIT_2026-08-16.md`, `sessions/2026-08-14-v17-lr-ladder.md`.

---

## 0. Verdict in one paragraph

The warp rewrite did not measure NULL because of staging. It measured NULL
because **`envs/SM = resident_threads_per_SM / W`**, and sm_89 caps
`resident_threads_per_SM` at `min(1536, 65536/R)`. At W=32 that is a hard
architectural ceiling of **48 envs/SM at any register budget whatsoever**,
against the **128 envs/SM the legacy kernel already has at production N** and
the **256 it has at N=32768**. W=16 caps at 96. Neither can reach parity; both
must buy their way back with chain shortening before earning a single percent.
Sub-warp at W ≤ 8 removes that ceiling (192 / 384 / 768 envs/SM at W = 8/4/2),
fits the problem's dimensions far better (NV=35 over 8 lanes is 87.5%
lane-efficient vs 54.7% over 32; the measured 8–15-RHS fused sweep is a
1–2-round fit at W=8 vs 47% idle lanes at W=32), and **reduces** rather than
reintroduces the K13 divergence tax relative to legacy. But only **W=2 is
residency-free at production N**, so only W=2 admits an incremental stage that
can show a positive number. The ILP hybrid is residency-neutral at N=16384
*and* N=32768 and doubles only at N=65,536 — refuted at production shape.

---

## 1. The residency identity, and where the design docs lost 32×

```
envs/SM = min( 1536 , 24·threads_per_block , 65536/R , grid_supply ) / W
```

`R` = registers per **thread** (allocated uniformly across the block — an idle
lane costs the same register file as a working one), `W` = lanes per env.
sm_89: 65,536 regs/SM, 48 warps = 1,536 threads/SM, 24 blocks/SM, ≤100 KB
shared/SM. Block size drops out except through the 24-block cap.

| kernel | W | R | threads/SM | **envs/SM** | source |
|---|---:|---:|---:|---:|---|
| legacy, N=16384 | 1 | 255 | 128 (grid: 512 blk / 128 SM) | **128** | `:929`, `:5138`, SHELL `:34-37` |
| legacy, N=32768 | 1 | 255 | 256 (reg ceiling) | **256** | PERF_AUDIT `:77-81` |
| warp S1/S2 | 32 | 255 (S2: 210) | 256 | **8** | `:6673-6674`, `:6702`, RECON `:66` |

`65536/(255·128) = 2 blocks/SM × 4 envs/blk = 8`. That is the measured
614.0 ms/tick, exactly 1/16 of legacy residency.

**The doc-level error.** SOLVER_CORE §3.2 (`:204-210`), §6 (`:618-622`) and
SHELL §1.2 (`:86-100`) all price the rewrite as "4 warps/SM → 16 warps/SM =
4×". That is unit-inconsistent: under thread-per-env one warp carries **32
envs**, under warp-per-env it carries **1**. In env units the same table reads
128 → 16, an **8× loss**; the shipped `min_blocks=1` landed at 16×. Every
downstream speedup estimate inherits the 32× slip. This is why a
bitwise-perfect, sanitizer-clean, budget-exact S2 timed 634.6.

### 1.1 Ceilings independent of registers and shared memory

`min(1536, 24·TPB)/W`, TPB ≥ 64:

| W | max envs/SM | vs legacy prod (128) | vs legacy ceiling (256) |
|---:|---:|---:|---:|
| 1 legacy | 1,536 | 12.0× | 6.0× |
| 2 | 768 | 6.0× | 3.0× |
| 4 | 384 | 3.0× | 1.5× |
| 8 | 192 | 1.5× | 0.75× |
| **16** | **96** | **0.75×** | 0.375× |
| **32** | **48** | **0.375×** | **0.188×** |

**W=16 and W=32 cannot reach production residency on sm_89 at any register or
shared budget.** SOLVER_CORE §2.2's "Do not go below 16" (`:153`) is backwards:
16 is the first architecturally excluded rung.

---

## 2. The sub-warp ladder, quantified

### 2.1 Registers per lane — the binding variable

**(a) Lane-0-serial (what S1/S2 built: `:6721-6725`, `:5390`, `:5740`).** The
group's serial body runs on lane 0 and needs the full thread-per-env working
set; registers are allocated uniformly, so `R = 255` for every lane and
`envs/SM = 256/W` — strictly worse than legacy at every W > 1. **The entire
lane-0 family is refuted by arithmetic, not by measurement.**

**(b) Group-distributed.** `R(W) ≈ R₀ + R_dist/W`, `R₀` = irreducible per-lane
overhead (addresses, predicates, loop counters, the PGS scalar chain
`dl/newl/dd` computed redundantly in all lanes, SOLVER_CORE `:459-461`).
Anchors: legacy 255 regs / zero spills on sm_89 (PERF_AUDIT `:19-21`);
read-only variant 168; SOLVER_CORE §3.2 config C estimated **80 regs/lane at
W=32** (`:210`). Envelope: optimistic `24 + 231/W`, doc-anchored `74 + 181/W`
(rounded up to the 8-reg granularity). Then `E = min(128, 65536/(R·W))` — grid
supply caps E at 128 because an SM has only 128 envs to run:

| W | R opt | **E opt** | warps/SM | R doc | **E doc** | regs/**env** (opt) |
|---:|---:|---:|---:|---:|---:|---:|
| 1 legacy | 255 | **128** | 4 | 255 | **128** | 255 |
| 2 | 144 | **128** | 8 | 168 | **128** | 288 |
| 4 | 88 | **128** | 16 | 120 | **128** | 352 |
| **8** | 56 | **128** | 32 | 96 | **85** | 448 |
| 16 | 40 | **96**¹ | 48¹ | 88 | **46** | 640 |
| 32 | 32 | **48**¹ | 48¹ | 80 | **25** | 1,024 |
| 32 as built | 255 | **8** | 8 | 255 | **8** | 8,160 |

¹ the 48-warp cap binds before registers.

Registers **per env** is the honest metric and it rises monotonically with W.
W=2/W=4 hold full production residency under both models; W=8 holds it under
the optimistic one and drops to 0.66× under the pessimistic one.

### 2.2 Shared memory: the factor cannot live there at any sub-warp W

The kernel allocates `NV·NV` floats = **4,900 B/env** (`:5325`,
`:6719-6720`). The 48 KB static per-block limit is not the binder — the per-SM
budget is:

| smem/env | envs/SM cap (100 KB/SM) | reachable targets |
|---:|---:|---|
| 6,144 B (SHELL §2.2, `:140-145`) | **16** | none |
| 4,900 B (as built) | **20** | none |
| 2,520 B (packed 630-triangle, SHELL `:159-166`) | **40** | none |
| **256 B (diag + control + reduction scratch)** | **400** | all W |

Block size cannot rescue this: 16 envs × 4.9 KB = 78.4 KB busts the 48 KB
static limit and, even as opt-in dynamic shared (99 KB/block), fits one block =
16 envs/SM; 4-env 32-thread blocks give 19.6 KB and 5 blocks/SM = 20 envs/SM.
**The same 20-env wall, because 100 KB / 4.9 KB = 20 is a per-SM fact.**
SOLVER_CORE §3.2 config C's "16 warps/SM (shared-capped at 17)" (`:210`) is
that number read in the wrong unit.

**Resolution: the factor stays out of shared.** Put it in the per-env arena
(SHELL §2.3 `:168-193`), NV-minor, read as a group broadcast — at W=8 a
level-`k` column read is 8 consecutive floats per group = one 32 B sector, 4
sectors per warp instruction. Register-distributing it instead needs 630/W
slots per lane: 79 at W=8 (fatal), 39 at W=16, 20 at W=32 — viable only where
the warp cap already excludes W. Shared then holds ~256 B/env: 35 reciprocal
diagonals, the group control block (`nc`/`ng`/`nl`/flags), a W-word reduction
scratch. That 400-env cap never binds.

### 2.3 Break-even chain shortening

Empirical residency law fit to the four measured legacy points (PERF_AUDIT
`:77-79`; at each N every env on the SM is resident, so `T(N) = C(E)`,
`E = N/128`): `C(E) = 6.911·E^0.597` — 56.1/79.6/124.0/194.0 measured vs
54.6/82.7/125.2/189.5 fit (±4%).

At N=16384 an SM retires 128 envs in `⌈128/E⌉` waves:
`T = (128/E)·C(E)·κ = 884.6·E^-0.403·κ`, with κ the per-env chain inflation
versus the legacy scalar body (κ<1 = shorter).

**Validation — κ is the only free parameter:**

| build | E | predicted at κ=1 | measured | implied κ |
|---|---:|---:|---:|---:|
| legacy | 128 | 126.6 | 124.0 | 0.98 |
| warp S1 (lane-0 serial, identical code) | 8 | 385.8 | **614.0** | **1.59** |
| warp S2 (fused sweep) | 8 | 385.8 | **634.6** | **1.645** |

The model reproduces 614.0 to 0.3% at κ=1.59, and κ=1.59 is what a
lane-0-serial form should cost: an `LDL` that served 32 envs in a legacy warp
now serves one. **S2 made the chain 3.4% *longer*** — its win (33 × 1,190
dependent FMAs → 1 × 1,190) was smaller than the marshalling it added
(`warp_rhs_in`/`warp_rhs_out` `:5352-5372`, 35 full-mask shuffles per RHS in
*and* out ≈ 840 at wtot=12; plus lane 0's 630 loads + 630 shared stores in
`warp_stage_factor` `:5336-5343`), all on the serial critical path. Amdahl:
the lane-0 residue is still ~95% of the chain.

Required shortening `Σ* = 7.133·E^-0.403`:

| E | **Σ\*** | reachable by |
|---:|---:|---|
| 8 | **3.11×** | W=32 as built |
| 25 / 46 | 2.03× / 1.53× | W=32 / W=16, doc-anchored regs |
| 48 / 96 | 1.50× / 1.13× | W=32 / W=16, arch ceiling |
| 85 | 1.19× | W=8, doc-anchored regs |
| **128** | **1.02×** | **W=2, W=4, W=8 (optimistic regs)** |

### 2.4 Modelled chain shortening, and the ladder resolved

Dependent steps per env-substep, extending SOLVER_CORE §4.6 (`:496-509`) down
the ladder. Divisors: NV work `⌈35/W⌉`; reductions `log2 W`; frame work bounded
by tree depth (10 levels, ~3 frames/level, so W≥8 is saturated); Cholesky's 35
sqrt/divide levels irreducible; PGS row body `⌈35/W⌉+log2W+⌈35/W⌉` vs 70.

| phase | today | W=2 | W=4 | W=8 | W=16 | W=32 |
|---|---:|---:|---:|---:|---:|---:|
| CRBA seeds / fill / merge | 7,690 | 3,845 | 1,923 | 962 | 639 | 595 |
| Cholesky factor | 630 | 333 | 184 | 109 | 70 | 70 |
| Fused solves | 39,270 | 1,260 | 630 | 350 | 210 | 140 |
| RNEA (both passes) | 5,400 | 3,000 | 2,200 | 1,800 | 1,800 | 1,800 |
| Row assembly + diags | 7,000 | 3,500 | 1,800 | 1,000 | 800 | 700 |
| PGS (12 iters) | 26,000 | 11,929 | 6,448 | 4,191 | 3,224 | 2,900 |
| step-11 clamps | 5,000 | 2,500 | 1,400 | 800 | 600 | 500 |
| **total** | **90,990** | **26,367** | **14,585** | **9,212** | **7,343** | **6,705** |
| **Σ** | 1.00× | **3.45×** | **6.24×** | **9.88×** | **12.4×** | **13.6×** |

The W=32 column reproduces SOLVER_CORE's own 7,100 / 12.8× to within 6% —
that is the model's calibration. **W=8 captures 73% of W=32's chain shortening
at 4× the env residency; W=4 captures 46% at 8×.**

`net = Σ / (τ·Σ*)`, τ = warp-form tax (shuffles, `__syncwarp`, group masks,
arena latency, inter-env divergence inside the warp). The one measured value
is τ=1.59 for the pathological lane-0 form; a distributed form keeps its `LDL`
coalescing, but the group-divergence term rises as W falls (§3.3). I use
τ = 1.2 / 1.4 / 1.7 / 2.0 / 2.0 for W = 2/4/8/16/32.

| W | E opt | Σ\* | Σ | τ | **net kernel** | wall (64% Amdahl) | **net at doc regs** |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 2 | 128 | 1.02 | 3.45 | 1.2 | **2.82×** | 1.67× | 2.82× |
| 4 | 128 | 1.02 | 6.24 | 1.4 | **4.37×** | 1.95× | 4.37× |
| **8** | 128 | 1.02 | 9.88 | 1.7 | **5.70×** | **2.09×** | 4.88× (E=85) |
| 16 | 96 | 1.13 | 12.4 | 2.0 | 5.49× | 2.07× | 4.00× (E=46) |
| 32 | 48 | 1.50 | 13.6 | 2.0 | 4.53× | 1.98× | 3.35× (E=25) |
| 32 as built | 8 | 3.11 | 1.0 | 1.59 | **0.20×** | 0.75× | **0.20×** |

The last row is the 614/634.6 measurement, reproduced. The ladder is nearly
flat from W=4 to W=32 in the best case and **strongly favours W=4–8 under the
pessimistic register model** — chain shortening saturates (Σ rises only 1.4×
from W=8 to W=32) while residency keeps compounding (E falls 5×).

---

## 3. What parallelizes at W=8

### 3.1 Geometric fit — W=8 is the best match to these dimensions

Lane utilization = `dim / (W·⌈dim/W⌉)`:

| dimension | W=2 | W=4 | **W=8** | W=16 | W=32 |
|---|---:|---:|---:|---:|---:|
| NV = 35 (dots, axpys, solves — dominant) | 97.2% | 97.2% | **87.5%** | 72.9% | **54.7%** |
| NQ = 29 / NF = 30 | 96.7/100% | 90.6/93.8% | 90.6/93.8% | 90.6/93.8% | 90.6/93.8% |
| NUM_PAIRS = 55 (collision compaction) | 98.2% | 98.2% | **98.2%** | 85.9% | 85.9% |
| fused-RHS width 8–15 (RECON `:59-65`) | — | — | **94%** (2 rds) | 62% | **47%** (17 idle) |

SOLVER_CORE §2.1 (`:96-101`) already shows W=8 at 88% NV utilization vs W=32's
55% and calls sub-warp "cheaper in issue slots" (`:120-123`). Its three
rejection grounds have expired: (1) "the factor will not fit in registers" —
the factor is in **shared** now (`:5325`, RECON `:16-21`), and §2.2 shows
shared is the worse cage; (2) "warps, not lanes, are the latency-hiding unit" —
true, and it argues *for* sub-warp: at W=8 an SM runs 32 warps against W=32's
8; (3) divergence — §3.3 shows the tax *falls* with W.

### 3.2 The fused solve — a near-perfect fit at W=8

Measured typical width `1 + 3·ng + nc + nl = 8–15` (RECON `:59-65`). The
shipped sweep (`:5661-5710`) is lane ≡ RHS in rounds of 32, so at wtot=15
**17 of 32 lanes solve the zero vector** — deliberately, to avoid divergence
(`:5663-5668`), but 53% waste.

| form | regs/lane | rounds at wtot=15 | dep. steps |
|---|---:|---:|---:|
| W=8, lane=RHS (S2 shape) | 35 (`x[NV]`, `:5660`) | 2 | 2,380 |
| **W=8, lane=row, tile B=4** | **20** | 4 | **1,400** |
| W=8, lane=row, B=8 | 40 | 2 | 700 |
| W=32, lane=row, B=16 | 32 | 1 | 140 |

Bitwise status is unchanged (SOLVER_CORE `:381-388`): lane ≡ row parallelizes
over **rows**, never over the summation index `k`, so each row's accumulator
still consumes `k` ascending exactly as `chol_solve_n` (`:2363`) does — the
max-diff-0.0 gate survives. And it deletes what killed S2: with `v`, `rhs`,
`g_J`, `c_J`, `ecol` group-distributed in the arena there is no
`warp_rhs_in`/`warp_rhs_out` and no `warp_stage_factor` at all.

### 3.3 PGS row bodies, and the K13 tax correctly signed

Per row (`:3841`, `:3856`, `:3881`, `:3912`, `:3928`): 35-dot → 3-op scalar
chain → 35-axpy, strictly Gauss-Seidel over rows.

| W | dot | reduce | axpy | dep. steps/row | vs 70 |
|---:|---:|---:|---:|---:|---:|
| 2 | 18 | 1 | 18 | 37 | 1.9× |
| 4 | 9 | 2 | 9 | 20 | 3.5× |
| **8** | **5** | **3** | **5** | **13** | **5.4×** |
| 16 | 3 | 4 | 3 | 10 | 7.0× |
| 32 | 2 | 5 | 2 | 9 | 7.8× |

**Saturation at W=8**: 69% of W=32's win, and the last 31% costs 4× the
register file per env. PGS is 28% of today's ops and becomes the dominant
post-rewrite residue (SOLVER_CORE §6: 40–50%), so this row matters most.

**Divergence, from the measured histograms.** `ng` per (env,substep),
production, 39.3M samples (RECON `:61-63`):
`{0:4.4%, 1:17.7%, 2:45.7%, 3:24.9%, 4:6.8%, ≥5:0.5%}`, mean 2.135.
`P(4 envs in a warp share ng) = Σpᵢ⁴ = 4.8%` — 95% of W=8 warps diverge. That
sounds fatal until compared against the kernel that ships today, which puts
**32** envs in a warp:

| envs/warp | W | E[max ng] | ng excess (mean 2.135) | E[max nc] | **nc excess** (mean 0.118) |
|---:|---:|---:|---:|---:|---:|
| 1 | 32 | 2.135 | 1.00× | 0.118 | 1.00× |
| 2 | 16 | 2.640 | 1.24× | 0.228 | 1.94× |
| **4** | **8** | **3.068** | **1.44×** | **0.433** | **3.68×** |
| 8 | 4 | 3.449 | 1.62× | 0.784 | 6.67× |
| 16 | 2 | 3.818 | 1.79× | 1.311 | 11.1× |
| **32** | **1 (legacy)** | **4.14** | **1.94×** | **1.965** | **16.7×** |

(`nc` from the measured 92.7%-zero / max-7 histogram, PERF_AUDIT `:82-85`,
as `{0:.927, 1:.045, 2:.017, 3:.007, 4:.003, 5+:.001}`, mean 0.118.)

**The K13 tax is not reborn at W=8; it is cut 1.35× on ground rows and 3.7× on
contact rows relative to what the legacy kernel pays today.** Legacy runs the
contact block on 91.2% of warps (`1 − .927³²`) to serve envs that need it 7.3%
of the time; a W=8 warp runs it on 26.2%. Per-env dependent cost of the contact
PGS block: legacy `1.965 × 70 = 138`, W=8 `0.433 × 13 = 5.6` — **24× on the
kernel's most divergent phase.** This is what K13 measured at 3.5× on
warp-uniform traffic (`:706-714`) and what Tier 3.3's permutation failed to
harvest by reordering (PERF_AUDIT `:44-52`); sub-warp harvests it
*structurally*, by shrinking the group — no permutation can do that.

### 3.4 Mask and sync discipline for 8-lane tiles

`g = (threadIdx.x & 31) >> 3`, `gl = threadIdx.x & 7`, `GMASK = 0xffu << 8*g`.

- **`__shfl_xor_sync(mask, v, b, 8)`** — the `width` argument partitions the
  warp into 8-lane segments with segment-relative lane ids; legal for any
  power-of-two width ≤ 32. A 35-dot reduce is 3 steps.
- **Which mask.** Warp-uniform regions (compile-time trip counts, `P.mode`,
  `N_SUB`, `PGS_ITERS`, the ω-cap pass over all `NQ`) take `0xffffffff` — all
  32 lanes arrive and `width=8` already confines the exchange. Regions entered
  under a per-env count (`s.ng`, `s.nc`, `s.nl`, the `if (dd != T(0))` guards
  at `:3846`/`:3867`/`:3886`/`:3906`) take `GMASK`; same for `__syncwarp`.
  `GMASK` is safe because a group's 8 lanes are convergent by construction —
  they run one env, so every per-env count is group-uniform, exactly the
  property `e` warp-uniform buys at W=32 (SHELL `:71-73`), scaled down. Using
  `0xffffffff` inside a group-divergent region is the hang.
- **Deterministic ballot compaction** (SHELL §3.3) survives:
  `b = __ballot_sync(0xffffffff,pred); gb = (b>>8*g)&0xff;
  slot = __popc(gb & ((1u<<gl)-1u));` — 3 instructions instead of 2, fixed
  tree, deterministic. NUM_PAIRS=55 takes 7 rounds at W=8 vs 2 at W=32; small
  phase, 98.2% lane-efficient.
- **What gets clumsier**: (i) the 3-DOF NV tail (SOLVER_CORE `:157-169`) is
  3-of-40 instead of 3-of-64 — *cheaper*; (ii) NF=30 over 8 lanes is 4 rounds,
  but the FK/RNEA tree has ~3 frames per level over 10 levels, so W=8 costs
  what W=32 costs (both level-bound); (iii) lane-varying `__shfl` source tables
  (`PARENT_LANE`, SOLVER_CORE `:412`) need group-relative indices. The residual
  cost is issue slots, not correctness: submasks make it legal, SIMT still runs
  the union of four groups' paths — that is τ, and §3.3 prices it *below* what
  ships today.

---

## 4. The hybrid: 2 envs per thread, software-interleaved ILP

Keep `step_kernel`'s shape (`:5138-5162`), `THREADS = 32` (`:929`), one thread
→ two independent `Dyn`/`Solver` frames, statements interleaved. Lowest
rewrite cost on the table. It does not survive the residency identity.

`envs/SM = 2 × min(1536, 65536/R, grid)`:

| N | blocks (legacy → hybrid) | legacy envs/SM | **hybrid envs/SM** |
|---:|---|---:|---:|
| 16,384 | 512 → 256 | 128 (grid-bound, 4 of 8 blk) | **128** — neutral |
| 32,768 | 1,024 → 512 | 256 (reg-bound) | **256** — neutral |
| 65,536 | 2,048 → 1,024 | 256 (reg cap) | **512** — 2× |

**At production N the hybrid is exactly residency-neutral and halves the warp
count (4 → 2 warps/SM).** Its mechanism — amortizing one 255-register thread
budget over two envs to defeat the 255-reg/thread ceiling — only engages once
the *register* ceiling binds, and at N=16384 the binder is the **grid** (SHELL
`:34-37`). The legacy kernel uses 32,640 of 65,536 regs/SM at production shape;
the hybrid uses 16,320. It makes the waste worse. Its only winning regime is
N ≥ 65,536, a PPO batch/recipe change gated on learning (PERF_AUDIT `:99-102`)
— and at that N the simpler intervention (raise N) already banks +28%.

**The ILP dilemma is structural.** *Fine interleave* (statement level): both
live sets coexist — sm_89 measures 254 regs with **zero spills** for one chain
(PERF_AUDIT `:19-21`), two want ~508 against a hard 255-reg/thread cap → ~250
registers permanently spilled. This file's history on that class is
unambiguous: the 64-reg build went 8,220/6,820 → 12,672/13,708 spills for 2.9%
wall (`:913-925`), and NC_CAP=16 cut the frame 28% and measured **+3.1%
slower** on a 254→168 ptxas phase flip alone (PERF_AUDIT `:86-91`). *Coarse
interleave* (phase/function level): one chain live, no spills — and no ILP; it
is a `for (k<2)` over envs, arithmetically what the grid already does minus
half the warps. *The narrow middle*: interleave **only** the dependent
reduction loops — `for (i) { jvA += JA[i]*vA[i]; jvB += JB[i]*vB[i]; }` costs
~8 registers and halves the PGS dot's 35-deep FMA dependency **with no
reassociation**, since each env keeps ascending order. No sub-warp form has
that property (the butterfly reassociates, SOLVER_CORE `:475-479`). But its
ceiling is ~1.9× on dot latency only; the two envs' `nc`/`ng`/`nl` must agree
or the loops split anyway (K13 at ILP level — per §3.3 two random envs agree
on `nc` 86% of the time and on `ng` only 30%); and it buys zero residency.
Amdahl with PGS at 28%: `1/(0.72 + 0.28/1.9) = 1.16×` at best, against a
doubled 84 KB frame and half the warps. And the compiler will not do it: ptxas
is spill-averse at 42 KB frames, so a `for k<2` over `step_env` serializes;
manual interleave means duplicating hot loop bodies, every one of which this
file says not to touch (K8's load-bearing `#pragma unroll` `:3755-3761`; K14's
deliberately dense `c_J` dot `:3873-3878`; K13's deliberately unfused limit
axpys `:3891-3898`).

**Refuted at production shape** — not because ILP is wrong (it is the right
idea for a latency-bound chain) but because the 255-reg/thread cap is not
currently the binder, so it pays spill risk for residency it cannot collect.

One cheap residue is worth keeping: **the register ladder probe.**
`__launch_bounds__(THREADS, k)` for k ∈ {1,2,3,4} pins 255/128/85/64 regs; run
`ptxas -v` plus the A/B at N=16384 **and** N=32768. It measures `R` — the one
parameter on which the whole sub-warp ladder pivots (§2.1) — for one line and
one afternoon. If the legacy body fits 128 regs with near-zero spills, W=4 at
88–120 regs/lane (a *quarter* of the state) is comfortably safe and §2.4's
optimistic column is live. If 128 regs already spills, the ladder is dead
above W=2.

---

## 5. Verdict and the experiment

### 5.1 Verdict

1. **W=16/W=32 refuted architecturally**, not by staging: 1,536 threads/SM caps
   them at 96 and 48 envs/SM against the 128 legacy already has. 614.0/634.6 is
   the floor of a mapping that cannot get there.
2. **Shared-resident factor refuted for every W**: 100 KB/SM ÷ 4.9 KB = 20
   envs/SM; the packed triangle reaches only 40.
3. **The lane-0-serial family refuted for every W > 1**: uniform register
   allocation forces `R = 255`, so `envs/SM = 256/W`. Every incremental stage in
   that family pays the full residency loss up front and collects chain
   shortening later — exactly why a bitwise-perfect S2 timed 634.6.
4. **W = 4–8 is the sweet spot**: full production residency under the optimistic
   register model, 46–73% of W=32's chain shortening, the best geometric fit to
   NV=35 / the 8–15-RHS sweep / NUM_PAIRS=55, and a divergence tax **1.35–3.7×
   lower than the shipping kernel's**.
5. **W=2 is the only rung on which a partial rewrite can show a positive
   number** — the only one with `Σ* = 1.02`.
6. **The hybrid is refuted at production N**; salvage its register-ladder probe
   as a gate on the sub-warp ladder.

### 5.2 The experiment: `W2-DUAL` — 2 lanes/env, PGS + fused solve

Chosen because it is the only sub-warp configuration whose break-even is 1.02×
— the only one buildable in one stage and measurable honestly. Template the
lane count from day one so W=4 and W=8 are a recompile.

```cpp
// __launch_bounds__(32, 8): ptxas pins 65536/(32*8) = 256 -> 255 regs, the SAME
// budget the legacy body already fits with ZERO spills (PERF_AUDIT :19-21).
// Grid at N=16384: 16384/(32/W) = 1024 blocks / 128 SM = 8 blocks/SM.
// Resident envs/SM = 8 * 16 = 128 -- IDENTICAL to legacy. Register file goes
// 32,640/65,536 (50%) -> 65,280/65,536 (99.6%); warps/SM 4 -> 8.
// NO __shared__ AT ALL: delete WARP_SM_WORDS_PER_ENV (:5325), the 19.6 KB tile
// (:6719), warp_stage_factor (:5336) and both static_asserts (:6679-6683).
template <int W, typename T, bool FC>
__global__ void __launch_bounds__(32, 8) step_kernel_sw(Sim<T> P) {
  const int gl = threadIdx.x & (W - 1);              // lane within group
  const int g  = threadIdx.x / W;                    // group within warp
  const int e  = blockIdx.x * (32 / W) + g;          // GROUP-uniform
  const unsigned GMASK = ((1u << W) - 1u) << (W * g);// 0x3 << 2g at W=2
  if (e >= P.N) return;
```

`e` is group-uniform, not warp-uniform, so `__syncwarp(GMASK)` replaces
`__syncwarp(WARP_FULL)` (`:5342`, `:5646`, `:5691`). With 32-thread blocks and
`N` a multiple of `32/W` the guard never fires mid-block.

PGS row body — replaces `:3841-3847`, identically `:3856-3868`, `:3881-3887`,
`:3912`, `:3928`:

```diff
-  T jv = T(0);
-  for (int i = 0; i < NV; i++) jv += s.g_J[nrow][i] * v[i];
+  T jv = T(0);                       // lane gl owns rows i = gl, gl+W, gl+2W...
+#pragma unroll
+  for (int r = 0; r < (NV + W - 1) / W; r++) {
+    const int i = r * W + gl;
+    if (i < NV) jv += s.g_J[nrow][i] * v[i];       // predicated, not branched
+  }
+#pragma unroll
+  for (int b = 1; b < W; b <<= 1)                  // 1 step at W=2, 3 at W=8
+    jv += __shfl_xor_sync(0xffffffffu, jv, b, W);  // width=W: segment-local
   const T dl   = (s.g_bias[nrow] - jv) / s.g_diag[nrow];
-  const T newl = tmax(s.g_lam[nrow] + dl, T(0));
+  const T newl = tmax(s.g_lam[nrow] + dl, T(0));   // redundant in all W lanes
   const T dd   = newl - s.g_lam[nrow];
   s.g_lam[nrow] = newl;
   if (dd != T(0))
-    for (int i = 0; i < NV; i++) v[i] += s.g_minv[nrow][i] * dd;
+#pragma unroll
+    for (int r = 0; r < (NV + W - 1) / W; r++) {   // elementwise: BITWISE
+      const int i = r * W + gl;
+      if (i < NV) v[i] += s.g_minv[nrow][i] * dd;
+    }
```

Full mask is correct on that shuffle only if every lane reaches it: hoist
`const int ng_w = group_max(s.ng)` once, run the loop to `ng_w`, predicate the
body on `gi < s.ng`. That pays §3.3's issue-slot tax (1.44× at W=8, **below**
legacy's 1.94×) for full-mask simplicity; `GMASK` is then needed only inside
`if (dd != T(0))` bodies, which contain no collective.
**Fused solve**: keep `:5661-5710`'s structure, delete
`warp_rhs_in`/`warp_rhs_out` (`:5352-5372`) — nothing left to marshal — and run
`chol_solve_n` (`:2363`) lane ≡ row over `⌈35/W⌉` slots against the arena
factor. **Numerics**: only the dots reassociate; at W=2 that is a single cut,
`(Σ even i) + (Σ odd i)` instead of ascending — the mildest entry into the S3
tolerance tier. Axpys, the solve and every scalar chain stay bitwise.

**Predictions, all falsifiable in one A/B:**

| quantity | prediction |
|---|---|
| ptxas census | ≤ 255 regs, **zero spills** (the body holds half an env) — hard gate |
| resident envs/SM at N=16384 | **128**, identical to legacy (8 blk/SM × 16 envs) |
| warps/SM | **8** (2× legacy) |
| break-even chain shortening | **1.02×** |
| Σ, PGS + solve only | 1.52× = `1/(0.30 + 0.28/1.9 + 0.42/2)` |
| τ (no shared, no marshalling, `LDL` still coalesced) | 1.1 – 1.4 |
| **step_kernel** | **1.10 – 1.38×** (124.0 → 90–113 ms/tick) |
| wall | 1.06 – 1.19× |
| W=2 full rewrite (all phases) | 2.82× kernel / 1.67× wall |
| **W=4 recompile, full rewrite** | **4.37× kernel / 1.95× wall** |
| **W=8 recompile, full rewrite** | 4.9 – 5.7× kernel / **2.0 – 2.1× wall** |

**Kill criterion.** If `W2-DUAL` on PGS + solve does not beat 124.0 ms at
N=16384 with the census green and 128 envs/SM confirmed, then residency is not
the explanation for the warp NULL, τ is larger than any of these models allow,
and the sub-warp lens is refuted. That is a single-number falsification the
W=32 staging never offered, because W=32 could never separate "the mapping is
wrong" from "the residency is gone".

### 5.3 What I would not do

- **Resume W=32 staging (S3–S6).** Ceiling 48 envs/SM, shipped at 8; S3's PGS
  win (7.8× on row dep-steps) cannot cover `Σ* = 3.11` **plus** the measured
  1.59 form tax. The stop rule that fired (session `2026-08-14`: "beating legacy
  from 634 needs 5.1× from S3 alone") is correct, and structurally so.
- **Packed-triangle shared factor.** 40 envs/SM is still 0.31× of production
  residency, and packed addressing puts integer ops on the dependent path
  (K17, `:520-546`).
- **Build the ILP hybrid** before the register-ladder probe (§4) shows the
  legacy body fits 128 regs without spilling. That probe gates the sub-warp
  ladder too, and should run first regardless.
