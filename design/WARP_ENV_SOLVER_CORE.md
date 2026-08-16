# Warp-per-env rewrite — SOLVER CORE design

Scope: CRBA, dense Cholesky factor, every triangular-solve variant, RNEA bias,
and the PGS iteration loop. The kernel shell (launch config, Dyn/Solver storage,
FK, collide, obs write, reset, integrator) is a separate agent's; the contract
between us is in **§8 Interface assumptions**.

Target file today: `python/beat_arena_env/csrc/beat_arena_cuda.cu` (5,338 lines).
Reference oracle: `rust/src/lib.rs`. This document changes nothing; it is input
to a later implementation phase.

Production config is **`--foot-contact`** (`cut_leaf.sh:99`), i.e. the
`FC = true` instantiation: the 12 weld rows are dropped
(`beat_arena_cuda.cu:1658`) and up to 24 ground rows exist
(`beat_arena_cuda.cu:1666-1675`, `N_GROUND_ROWS == 24` at `:1443`). All op
counts below are for that mode; the weld mode is called out where it differs,
because it must keep working (it is the v10 parity path).

---

## 0. The one-paragraph recommendation

**Full warp (32 lanes) per env, lane ≡ velocity-DOF row `i`**, with the three
overflow DOFs (`i = 32,33,34`, since `NV = 35`) carried in a 3-lane predicated
tail backed by a small shared tile. The Cholesky factor lives **in registers,
one matrix row per lane** (32 f32/lane for rows 0–31; rows 32–34 in shared);
the multi-row blocks (`g_J`, `g_minv`, `c_J`, `c_minv`, `ecol`) live in
**per-lane local arrays** (dynamically indexable, and now L1-sized because the
per-warp footprint drops 32×); shared memory carries only tails, the
cross-lane staging tile, and phase-boundary transposes. Target 128 regs/lane →
**16 warps/SM against today's 4**. The factorization switches from
left-looking to **right-looking** (bitwise-equivalent, 5.7× shorter critical
path), and all of a substep's 1 + 3·ng + nc + nl right-hand sides collapse into
**one 70-level fused triangular sweep** instead of ~33 independent 1,190-FMA
passes. Realistic band: **3–8× on `step_kernel`, 1.75–2.27× on total training
wall** (kernel = 64% of wall; hard ceiling 2.78×).

---

## 1. Dimensions and the measured baseline

| symbol | value | source |
|---|---|---|
| `NQ` | 29 | `:82` |
| `NV` | 35 | `:83` |
| `VOFF` | 6 | `:84` |
| `NF` | **30** frames | `:85` |
| `NUM_PAIRS` | 55 | `:87` |
| `N_GROUND_ROWS` | 24 (8 corners × 3) | `:1440-1444` |
| `N_SUB` | 8 substeps/tick | `:964` |
| `PGS_ITERS` | 12 | `:976` |
| `CLAMP_PASSES` | 64 (early-out) | `:968`, `:4043-4074` |
| `THREADS` | 32 (1 warp/block today) | `:928`, `:5285` |

Tree shape (`F_PARENT` at `:999-1001`): three roots — frame 0 (spine/arms),
18 (left leg), 24 (right leg). Deepest chain `0→1→2→4→5→6→7→8→9→10`, so
**tree depth = 10 levels over 30 frames**. `F_COORD` (`:1002-1004`) is a
bijection from the 29 coord-bearing frames onto joint columns `0..28`
(static_asserted at `:1023-1026`) — this is what makes a clean
lane↔row↔frame mapping possible.

Measured (2026-08-15, RTX 4090 sm_89, `PERF_AUDIT_2026-08-16.md:10-103`),
taken as given:

- `step_kernel` = 90% of rollout = **64% of total wall**, ~124 ms/tick at
  N=16384.
- Latency-bound / parallelism-starved: 13.7 / 9.7 / 7.6 / 5.9 µs/env at
  N = 4k/8k/16k/32k. **4 warps/SM at production N**, ceiling 8 (254 regs).
- 42 KB/thread local frame, zero spills, ~0.1% FLOP utilization.
- Frame-capacity cuts measured NEGATIVE (NC_CAP=16: 30,288 B frame but +3.1%
  slower via a 254→168 reg ptxas phase flip).
- ncu counters permission-locked (`ERR_NVGPUCTRPERM`).

Two derived facts I will lean on, both consistent with the above:

1. **Per-warp working set is the whole story.** Today one warp carries
   32 envs × 42 KB = **1.34 MB** of local frame; L1 is 128 KB, so the hot
   objects (`d.m` alone is 4,900 B/thread = 157 KB/warp) cannot stay resident.
   Warp-per-env makes the per-warp working set *one env's*, and the design
   below drives that to ~14 KB — i.e. inside L1 at useful occupancy.
2. **Occupancy extrapolation from measured data.** The measured 4→8 warps/SM
   step (N=16k→32k) bought 1.29×. Getting to 16 warps/SM plausibly buys
   1.6–2.0× *on its own*, before any restructuring. That is the most
   trustworthy single number in this document because it is an extrapolation
   of a measured slope rather than a model.

---

## 2. Q1 — Lane mapping

### 2.1 The candidate mappings

`NV = 35` is the awkward number: just above 32. Every NV-length vector
distributed over W lanes costs `ceil(35/W)` slots per lane.

| lanes/env `W` | rounds for NV=35 | lane util (NV op) | envs/warp | inter-env divergence | per-env latency of an NV dot |
|---|---|---|---|---|---|
| 32 | 2 (32 + 3-lane tail) | 35/64 = **55%** | 1 | **none** | 2 FMA + 5-step butterfly ≈ 7 dep. steps |
| 16 | 3 (16+16+3) | 35/48 = 73% | 2 | 2-way | 3 FMA + 4-step ≈ 7 |
| 8 | 5 | 35/40 = 88% | 4 | 4-way | 5 FMA + 3-step ≈ 8 |
| 4 | 9 | 35/36 = 97% | 8 | 8-way | 9 FMA + 2-step ≈ 11 |

Per **phase**, at W=32 (util = active lanes / 32):

| phase | active lanes | util | note |
|---|---|---|---|
| CRBA composite seeds (`:1908-1922`) | 30 frames | 94% | lane = frame |
| CRBA column fill (`:1932-1981`) | `ANC_N[k]` + 6 root rows, mean 11 | 34% | the ancestor rows of one column |
| CRBA composite merge (`:1983-2011`) | 12 (3-vec + 3×3) | 37% | tree-serial, irreducible |
| Cholesky column *j* (`:2261-2340`) | 34−*j*, mean 17.5 | 55% | trailing rows only |
| Triangular solve level *k* (`:2362-2402`) | rows > *k*, mean 17.5 | 55% | ×W_RHS instruction reuse |
| RNEA fwd/bwd (`:2048-2172`) | 30 frames | 94% | lane = frame, scalar 3-vec math per lane |
| PGS dot / axpy (`:3878-3916`) | 35 of 64 | 55% | the dominant phase |
| PGS scalar chain (`dl/newl/dd`) | 1 (replicated) | 3% | computed redundantly in all lanes — deliberate, avoids a broadcast |

Weighted by phase share (§6), mean utilization at W=32 ≈ **50%**.

### 2.2 Why 32 wins anyway

Sub-warp groups look better on the utilization column and are *cheaper in issue
slots* (the reduction tree is shared across the sub-groups in the same
instructions — at W=8 a warp retires 4 envs' dots in ~11 instructions vs
W=32's 12 instructions for 1). Three things kill them:

1. **The factor will not fit.** The Cholesky factor is 1,225 values (630 in
   the occupied triangle). Register-resident, statically indexed, it needs
   `ceil(35/W)` rows per lane at up to 35 slots each: 32 regs/lane at W=32,
   ~50 at W=16, ~119 at W=8, ~230 at W=4. Below W=16 the factor either
   evicts everything else or goes to shared, and shared at 4,900 B/env caps
   occupancy at 20 envs/SM *before* anything else is allocated.
2. **Warps, not lanes, are the latency-hiding unit.** The measured binder is
   stalls, not issue slots (0.1% FLOP utilization = ~200× of issue headroom).
   At W=16 the same env count yields half the warps, i.e. half the independent
   instruction streams per SM. W=32 buys the maximum number of schedulable
   streams for a given register budget.
3. **Divergence.** W=32 makes *every* data-dependent branch in the solver
   warp-uniform: `s.nc`, `s.ng`, `s.nl`, the 4/2/1 ladders (`:3558-3628`),
   the `if (dd != T(0))` axpy guards (`:3885`, `:3905`), the `ensure_ecol`
   cache hit (`:3038`), and the `CLAMP_PASSES` early-out (`:4073`) — which
   today costs the **max over 32 envs**, not each env's own count. The file's
   own K13 experiment (`:666-713`) measured a per-lane-count width ladder at
   −2.18% on production traffic and **+2.26% on warp-uniform traffic with the
   same binary**, and the same probe measured the whole kernel at 100,021 vs
   28,490 env-steps/s (**3.5×**) uniform-vs-production. That 3.5× is an sm_61
   number and Tier 3.3's partial fix measured null on sm_89
   (`PERF_AUDIT:44-52`), so I do **not** bank it — but W=32 captures it
   structurally if any of it survives, and W<32 gives part of it back.

**Recommendation: W = 32 (full warp per env).**
**Fallback: W = 16**, taken only if the S7 occupancy ladder shows the W=32
register budget stuck below ~10 warps/SM. W=16 packs the factor ~25% denser
(800 vs 1,024 floats/env) and gives ~25% more envs in flight, at 2× per-env
latency and 2-way divergence. Do not go below 16.

### 2.3 The 3-DOF overflow

Lane `l` owns DOF row `l` for `l ∈ [0,32)`. Rows 32–34 are the **tail**:

- vectors (`v`, `rhs`, `c`, every `g_J`/`g_minv`/`c_J`/`c_minv`/`ecol` row):
  one register/lane for rows 0–31, plus a 3-float shared slot per vector for
  rows 32–34, owned by lanes 0–2.
- the factor: rows 0–31 in registers (32 slots/lane), rows 32–34 in a shared
  tile `Lhi[3][NV]` = 105 floats = 420 B/env.

Every NV-wide primitive is therefore
`op(lane_value); if (lane < 3) op(tail_value);` — a 3-lane predicated tail,
one extra instruction, **not** a branch. The tail must be folded into a
reduction *before* the butterfly so the reduction tree stays a fixed shape
(§5).

---

## 3. Q2 — Data placement for the hot matrices

### 3.1 `d.m` is column-major and lower-triangle-only — and that is exactly right for a warp

`d.m` holds M from `crba` and then, in place, its Cholesky factor L
(`:1560-1561`, `:2209-2231`). Element (row R, col C) lives at `m[C*NV + R]`
(K12, `:1898-1902` and `:2188-2207`), and **only R ≥ C is ever touched** —
verified: the K4c invariant at `:1887-1896` enumerates every consumer
(`chol_factor_n` in place; `chol_solve_n` (i,k) k<i, (k,i) k>i, (i,i);
`chol_fwd_unit{,2,4}_n`; `chol_back{,2,4}_n`), and `crba`'s zero fill at
`:1905-1906` writes only `j <= i`, the ancestor pair writes one element
(`:1968`), and the root-block mirrors are gone (`:1976-1981`, `:2036-2039`).

Two consequences for the warp design:

1. **Bank conflicts: stride `NV = 35` is the *good* stride, do not pad.**
   For any shared tile of NV-stride rows, a lane-per-row access to a fixed
   column C reads `C*35 + lane` — 32 consecutive words, all 32 banks, zero
   conflicts. A lane-per-column access to a fixed row R reads `R + 35*C` —
   stride 35, and `gcd(35,32) = 1`, so again all 32 banks, zero conflicts.
   **Padding to 36 would be actively harmful** (`gcd(36,32) = 4` → 4-way
   conflicts on the row-wise access). This is the single most useful concrete
   answer in this section, and it is free: it is the layout the kernel already
   has.
2. Only 630 of 1,225 entries carry data, so a register-resident row `i`
   needs slots `0..i` — max 32 for rows 0–31. **32 registers/lane** covers the
   whole in-register part of the factor.

### 3.2 The three placements, budgeted

Register file per SM = 65,536 regs (256 KB); shared = up to 100 KB/SM on
sm_89; L1 = 128 KB/SM. Warps/SM ≈ `65536 / (regs_per_lane × 32)`.

| config | factor | row blocks (`g_*`, `c_*`, `ecol`) | regs/lane | shared/env | local/env | warps/SM |
|---|---|---|---|---|---|---|
| **A** all-registers | registers (32) | registers (≈ 24+24+12+12+29 with tails ≈ 160) | ~200 | 0.5 KB | ~0 | **10** |
| **B (recommended)** | registers (32) | per-lane local arrays | ~122 → pin 128 | ~1.0 KB | ~14 KB | **16** |
| **C** shared factor | shared `[35][35]` | per-lane local arrays | ~80 | 5.9 KB | ~14 KB | **16** (shared-capped at 17) |

Config B register accounting (f32, per lane):

| item | regs |
|---|---|
| factor row (rows 0–31, slots 0..i) | 32 |
| `v`, `rhs`, `c`, `q` slice + tails | 8 |
| owned frame FK (`pos[3]`, `axw[3]`, `r[3][3]`) | 15 |
| CRBA composite of owned frame (`cm`,`cc[3]`,`ci[3][3]`) | 13 |
| RNEA of owned frame (`om`,`al`,`ao`,`fb`,`nb`) | 15 |
| working multi-RHS tile (16 wide) | 16 |
| addresses, predicates, temps | ~23 |
| **total** | **~122** |

Config B local accounting (per lane, f32): `g_J[24]` + `g_minv[24]` +
`c_J[NC_CAP=12]` + `c_minv[12]` + `ecol[29]` + row scalars ≈ 111 floats =
444 B/lane = **14.2 KB/env**, against today's 42 KB/env. At 16 warps/SM that
is 227 KB of local footprint against a 128 KB L1 — roughly 55% resident, and
*all* of it coalesced (per-thread interleaved local memory: 32 lanes reading
`g_J[r]` for the same `r` is one 128 B transaction). The hottest object of
all, the factor, does **zero** memory traffic.

**Recommendation: config B.** Rationale: it removes the single largest traffic
source (the factor: today ~52 KB of local load per factorization, ~13,000 LDL
per env-substep in the factor+solve inner loops), keeps the row blocks
dynamically indexable (registers cannot be indexed by a runtime row id — this
is a hard constraint, not a preference), and lands at 16 warps/SM = 4× today.
Config C is the fallback if the register-resident factor makes ptxas spill;
it costs 5.9 KB/env of shared but frees 32 regs/lane and keeps the same
occupancy, so it is a cheap escape hatch. Config A is listed only to record
that it loses: 10 warps/SM is not enough better than today's 4–8 to justify
the pressure.

**Occupancy knob.** `__launch_bounds__(THREADS, 1)` at `:5137` deliberately
pins 255 registers — `:912-924` explains why (occupancy was not the lever and
ptxas's free choice was unstable). For the warp kernel that reasoning
**inverts**: occupancy *is* the lever. The warp kernel must use
`__launch_bounds__(BLOCK, MIN_BLOCKS)` with `MIN_BLOCKS` chosen so ptxas is
told to fit 16 warps/SM, and the S7 ladder sweeps it over {8, 12, 16, 20, 24}
warps/SM. The old kernel's `__launch_bounds__` is untouched.

---

## 4. Q3 — Parallelization per phase, with honest dependency structure

### 4.1 CRBA (`crba`, `:1904-2040`)

Three separable parts:

**(a) Composite seeds (`:1908-1929`) — fully parallel.** 30 frames × ~50 flops
(`inertia_world2` is 6 symmetric entries, `:1866-1877`). Lane = frame, one
step. **30× fewer dependent steps.**

**(b) Column fill (`:1932-1981`) — parallelizes completely, and this is the
non-obvious win.** For coord-bearing frame `k` with column `j = VOFF + coord`,
the code writes exactly: the diagonal `m[j*NV+j]` (`:1950`), one entry per
strict ancestor at `m[min*NV+max]` (`:1968`), and the six root rows
`m[t*NV+j]`, `m[(3+t)*NV+j]` (`:1978-1981`). **Every one of those writes is in
column `j`** — the loop emits one matrix column per frame. Under lane ≡ row,
a column is spread one element per lane, so *all* of frame `k`'s ancestor
entries are produced in **one parallel step**: lane `i` tests membership of
its own row in `k`'s ancestor set (a 64-bit constexpr mask per frame, exactly
the `wj_chain_mask` idiom already at `:1390-1399`), and if in, computes
`dot3(axw[cf(i)], ha + cross(cc[k]-pos[cf(i)], hl))` where `cf(i)` is the
frame whose `F_COORD` is `i-VOFF` (the bijection at `:1023-1026`). `hl`, `ha`,
`cvec` are warp-uniform scalars for the frame.
The 117 serial ancestor steps (counted at `:1038`; = Σ`ANC_N` over coord
frames − 29) collapse to **29 parallel column emissions**. The `ANC`
tables at `:1167-1180` become the mask table's provenance, not a walk.

**(c) Composite merge (`:1983-2011`) — genuinely tree-serial.** 30 sequential
steps, each ~80 flops (2 × `add_shift`, `:1880-1885`, plus a 3×3 accumulate).
Inside a step, 12 lanes do the 3-vector + 3×3 work → ~12 instructions/step at
37% utilization. **Level-scheduling is available but NOT recommended for the
first landing**: frames at the same depth merge into different parents so they
are independent (depth 10 vs 30 steps), *except* siblings sharing a parent
(frame 2 has children 3, 4, 11), and reordering the merges changes the
floating-point accumulation order into the parent. Keep the `k = NF-1 … 0`
order of `:1932`; revisit under the tolerance tier later.

Honest verdict: after (a) and (b), **(c) becomes the dominant CRBA residue** —
~30 sequential steps at 37% lane utilization, ~15–20% of the post-rewrite
substep critical path.

### 4.2 Cholesky factor, n = 35 (`chol_factor_n`, `:2259-2341`)

Today: left-looking, in-place, unrolled-and-jammed by 4 (`:2285-2333`), with
the next pivot software-pipelined into the first row block (`:2243-2257`).
Op count: Σ_j *j*·(34−*j*) = **6,545 FMA**, plus 595 divisions
(Σ_j (34−*j*)) and 35 `sqrt`. Dependent chain: each accumulator `s_t` is a
serial chain of length *j*, so the critical path is Σ_j *j* = **595 dependent
FMAs** plus 35 (sqrt + divide) — the 4-way jam buys ILP, not chain length.

**Recommended warp form: switch to RIGHT-looking.** Lane ≡ row, row in
registers. For j = 0 … 34 (unrolled):

```
dj   = sqrt(max(A[j][j], PD_FLOOR));      // lane j computes, broadcast (1 shfl)
if (i > j) L[i][j] = A[i][j] / dj;        // one divide, all trailing lanes at once
for k = j+1 … 34:                          // 34-j shuffles
    Lkj = __shfl_sync(FULL, myrow[j], k);  // lane k's L[k][j] to everyone
    if (i >= k) myrow[k] -= myrow[j]*Lkj;  // independent FMA per k
```

- **Instructions**: 595 shuffles + 630 FMA + 35 sqrt + 35 divide ≈ **1,300**,
  against today's 6,545 FMA + ~13,000 local loads.
- **Critical path**: within a column the FMAs target *different* registers and
  are mutually independent, so the chain is `sqrt → broadcast → divide → 1 FMA`
  ≈ 4 dependent ops per column → **~140 dependent steps** vs 595+.
  **5.7× shorter, ~14× fewer instructions, and zero memory traffic.**
- **Bitwise**: right-looking accumulates `A[i][k] -= L[i][j]·L[k][j]` for
  *j ascending* — the same terms in the same order, from the same seed
  `M[i][k]`, with the same `mul+sub → FFMA` contraction shape as `:2292-2295`.
  The pivot `A[j][j]` after all updates equals `M[j][j] − Σ_{k<j} L[j][k]²`
  accumulated ascending, i.e. the `np` chain of `:2296`/`:2311`. The divides
  are element-for-element the same `s / dj`. **Gate this stage at max diff
  0.0**; the only escape is an FMA-contraction shift (the K10e2 hazard,
  `:161-163` and `:582-592`), for which a 1-ulp fallback is acceptable.
- The `PD_FLOOR_F32` guard (`:2265-2268`, `:986`) is unchanged; `d.pd_bad`
  becomes a warp-uniform flag set by lane `j` and broadcast (or written by all
  lanes — it is a monotone latch, so a redundant write is safe).

Do **not** carry the K6 unroll-and-jam or the pivot software-pipelining into
the warp form: both exist to manufacture ILP for one serial thread, and the
warp form has 32-way ILP by construction.

### 4.3 Triangular solves — the biggest structural win

Entry points today: `chol_solve_n` (`:2362`), `chol_solve_multi_n<W>`
(`:2441`), `chol_solve_multi_tn<W>` (`:2532`), `chol_fwd_unit_n` (`:2630`),
`chol_fwd_unit4_n` (`:2687`), `chol_fwd_unit2_n` (`:2794`), `chol_back_n`
(`:2868`), `chol_solve_unit_n` (`:2879`), `chol_back4_n` (`:2900`),
`chol_back2_n` (`:2922`).

**RHS traffic per substep, counted from the code (FC production mode):**

| site | line | RHS count | note |
|---|---|---|---|
| step-6 `v* = v + h M⁻¹(τ−C)` | `:3955` | 1 | `chol_solve_n(NV, d.m, d.rhs)` |
| ground rows `g_minv` | `:3555-3588` | `3·ng` ≤ 24 | 4/2/1 ladder over `s.ng*GROUND_ROWS_PER_CORNER` |
| contact rows `c_minv` | `:3597-3629` | `nc` ≤ 8 | measured max 7 prod / 8 stress, 92.7% zero |
| limit-row unit columns `ecol` | `:3647-3676` | `nl` | mean 7.8 (random-gear, `:3643`), 15.7 on bench traffic (`:700`), max 23 (stopjam, `:653`) |
| ω-cap lazy `ensure_pcol→ensure_ecol` inside PGS | `:3923` | 0..29, usually ~0 | **mid-PGS**, see risk R6 |
| step-11 clamp `ensure_ecol` | `:4065` | ~0 | K16 measured 0.12% miss rate (`:890-910`) |
| *weld mode only:* `chol_solve_multi_tn<N_WELD>` | `:3548` | 12 | plus 12×12 `aww` solves at `:3715`, `:3809`, `:3121` |

Nominal production width **W_tot = 1 + 3·ng + nc + nl ≈ 33** (ng = 8, nc = 0,
nl = 8), worst case 1 + 24 + 8 + 29 = 62. Today that is
**33 × 1,190 = 39,270 dependent FMAs** — the single largest phase.

**All of these RHS are known before any of them is solved.** Verified from
the ordering in `constraint_solve`: ground assembly `:3394-3442`, contact
narrow phase + `c_J` `:3474-3515`, limit rows `:3519-3539` — all complete
before the first solve at `:3548`. So they fuse.

Warp form: lane ≡ row; RHS *w* held as one register per lane (+ shared tail).

```
forward: for k = 0 … 34:                    // 35 levels, irreducible
             xk_w = __shfl_sync(FULL, x_w, k)  (one per RHS in the tile)
             if (i > k) x_w -= myrow[k] * xk_w
             if (i == k) x_w *= rdiag[k]
back:    for i = 34 … 0: symmetric
```

- **Critical path: 70 levels, independent of W.** vs 39,270 serial FMAs →
  **~560× on dependent steps** for this phase.
- Instructions: 70 × (W shuffles + W FMA + overhead) ≈ 150·W ≈ 4,900 at
  W=33, vs 39,270 FMA + ~78,000 local loads today → **~8.5× fewer
  instructions and ~100% of the memory traffic removed.**
- **Bitwise.** Row *i*'s accumulation still runs `k` ascending, one term at a
  time, in lane *i*'s own register — exactly the scalar order of `:2367-2374`
  and `:2392`. Nothing is reduced across lanes here; the parallelism is over
  *rows*, not over the summation index. K6's absolute prohibition on
  row-blocking the back substitution (`:215-229`) is **satisfied by
  construction**: the warp form does not block rows, it assigns them to lanes,
  and every row still consumes `k` in ascending order from values already
  produced.
- Register tiling: the tile width is bounded by registers. Run W_tot in tiles
  of 16 → `ceil(33/16) = 3` sweeps = 210 levels; still ~190× better than
  today. The tile width is an S7 ladder parameter.
- The unit-RHS specialization (`chol_fwd_unit_n`, start at `m = VOFF+j`,
  `:2605-2628`) can be **dropped**: it saves instructions, not levels, and its
  own header proves the skipped rows reproduce `+0.0`, so starting at 0 is
  bitwise identical. Keeping it would fragment the fused sweep for no
  critical-path gain.
- `s.mw`'s RHS-minor layout (K11, `:1622-1651`) and `s.c_minv`'s row-major
  layout (K18(i), `:1684-1687`) both become **moot**: with lane ≡ row there is
  exactly one address per RHS per lane and the "how many live addresses does
  the inner loop need" instrument that decided K11/K12/K18 no longer has a
  degree of freedom to decide.

### 4.4 RNEA bias (`rnea_bias`, `:2044-2179`)

Forward pass (`:2048-2120`) is parent→child chain-serial over 30 frames;
backward pass (`:2149-2172`) is child→parent. Per frame ~80–100 flops of
3-vector and 3×3 math.

Warp form: **lane ≡ frame** (30 of 32 lanes), each lane running its own
frame's scalar 3-vector math, with the chain resolved by **level
synchronization**: 10 levels (§1), and at each level the child lanes pull
`om/al/ao` (9 floats) from the parent lane via `__shfl_sync` with a
*lane-varying source* (`PARENT_LANE[k]`, a constexpr table — `__shfl_sync`
allows each lane a different source lane). Cost ≈ 10 levels × (9 shuffles +
~85 scalar ops) ≈ **900 instructions per pass**, vs 30 × ~90 = 2,700 serial
ops; both passes ≈ 1,800 vs 5,400. **~3× on instructions, ~3× on the chain.**

This is the weakest compression in the design and it is inherent: 3-vector
math on 32 lanes with lane = frame gives 30/32 lane occupancy but *no*
intra-frame parallelism, and the level count cannot go below the tree depth.
RNEA is therefore a **growing relative residue** — roughly 10–15% of the
post-rewrite substep.

The `d.c[VOFF + F_COORD[k]]` write at `:2151` is produced by the frame-lane
and consumed as a *row* by the solver phase: one shuffle permutation
`c_i = __shfl_sync(FULL, val, SRC_LANE[i])` (constexpr table, lane-varying
source). Same trick moves `axw`/`pos` from frame-lanes to row-lanes for CRBA:
6 shuffles.

### 4.5 PGS (`:3760-3935`) — strictly sequential rows, parallel row bodies

Structure per iteration, in fixed order: (a) weld block (weld mode only,
`:3766-3827`); (a2) ground corners, normal then two box-clamped friction rows
(`:3835-3870`); (b) contacts (`:3878-3887`); (c) limit rows (`:3898-3916`);
(d) ω-cap pass over all `NQ` joints (`:3918-3934`).

Each row is: `jv = Σ_i J[r][i]·v[i]` (NV dot) → scalar chain
`dl = (bias − jv)/diag; newl = clamp(...); dd = newl − lam` → `v[i] +=
Minv[r][i]·dd` (NV axpy). Gauss-Seidel: row `r+1` reads the `v` that row `r`
just wrote. **This ordering is load-bearing and is preserved exactly.**

Warp form per row (lane ≡ row-of-v):

```
p  = J_r[lane] * v[lane];
if (lane < 3) p += Jtail[r][lane] * vtail[lane];      // fixed 3-lane tail fold
#pragma unroll
for (b = 0; b < 5; b++) p += __shfl_xor_sync(0xffffffff, p, 1<<b);
// p is now the full dot, identical in every lane — no broadcast needed
dl = (bias[r] - p) / diag[r]; newl = ...; dd = newl - lam[r]; lam[r] = newl;
if (dd != 0.0f) { v[lane] += Minv_r[lane]*dd;
                  if (lane < 3) vtail[lane] += Minvtail[r][lane]*dd; __syncwarp(); }
```

- ~15 warp instructions per row against 70 scalar flops + ~140 local accesses;
  ~7 dependent steps against a 35-long dependent FMA chain twice over.
  **~5× on instructions, ~10× on the chain, and the operands move from local
  memory to registers.**
- The scalar chain is computed **redundantly in all lanes**, which is free
  (the butterfly already leaves the reduction in every lane) and avoids a
  broadcast on the critical path.
- The `if (dd != 0)` guard (`:3845`, `:3866`, `:3885`, `:3905`) is now
  **warp-uniform** — this is precisely the K13 divergence tax
  (`:666-704`) removed rather than paid.

**Determinism of the reduction.** `__shfl_xor_sync` butterfly with a
compile-time width is a fixed tree: every lane performs the same five adds on
the same partials in the same order every run. Different lanes associate
differently (lane 0 computes `(a0+a1)+(a2+a3)`, lane 1 `(a1+a0)+(a3+a2)`) but
IEEE addition is exactly commutative, so **all 32 lanes hold bitwise-identical
results** and the value is reproducible run to run. No atomics, no
`cg::reduce`, no runtime-width reduction. This is the requirement in Q3/Q4 and
it is met by construction.

**What this changes vs the serial code:** the dot's summation order. The
serial loop at `:3880` sums `i = 0 … 34` ascending; the butterfly sums a
pairwise tree. This is a genuine reassociation and is **the first deliberate
non-bitwise step** (S3 in §7). It is accepted under the parity waiver, and it
is worth noting that the tree form is *better* conditioned, not worse.

**Serial residue.** The row order is irreducible. At ng = 8, nl = 8, nc = 0:
12 iterations × (24 + 8 + ~2 cap engagements) ≈ **~410 sequential row updates
per substep**, ~7 dependent steps each ≈ 2,900 dependent steps. After the
solves and factor collapse, **PGS becomes the dominant serial residue of the
whole substep** (~40–50%).

Two later, out-of-scope-for-v1 levers, recorded so they are not lost:
*(i)* form the Delassus matrix `A_rc = J_r·M⁻¹J_c` once (~28k flops at 40
rows) and iterate on residuals instead of re-dotting `v` — halves the per-row
work but does not shorten the row chain; *(ii)* the ω-cap pass (`:3918`) walks
all 29 joints every iteration and can be narrowed to joints whose `v` actually
moved. Both are tolerance-tier and belong after the first landing.

### 4.6 Phase summary

| phase | scalar ops / env-substep | dep. steps today | warp instrs | warp dep. steps | share of today |
|---|---:|---:|---:|---:|---:|
| CRBA seeds | 1,500 | ~1,500 | ~60 | ~60 | 2% |
| CRBA column fill | 3,790 | ~3,790 | ~450 | ~450 | 4% |
| CRBA merge | 2,400 | 2,400 | ~360 | ~360 | 3% |
| Cholesky factor | 7,175 | ~595 + 35 div | ~1,300 | ~140 | 8% |
| Fused solves (33 RHS) | 39,270 | 39,270 | ~4,900 | ~210 | 42% |
| RNEA (both passes) | 5,400 | 5,400 | ~1,800 | ~1,800 | 3% |
| Row assembly + diag dots | ~7,000 | ~7,000 | ~1,400 | ~700 | 8% |
| PGS (12 iters) | 26,000 | 26,000 | ~6,200 | ~2,900 | 28% |
| step-11 clamps | ~5,000 | ~5,000 | ~900 | ~500 | 2% |
| **total** | **~92,700** | **~91,000** | **~17,400** | **~7,100** | 100% |

(ng = 8, nc = 0, nl = 8 nominal. `ng` is the biggest unknown — see §10.)

---

## 5. Q4 — Determinism and the parity/dtype split

**Required: run-to-run bitwise determinism.** Delivered by:
- fixed-width `__shfl_xor_sync` butterflies with compile-time trip counts —
  no `cg::reduce`, no runtime-width trees, no `atomicAdd` anywhere in the
  solver core;
- the 3-lane tail folded at a *fixed* position in the tree (before the
  butterfly), so the tail's contribution is not order-dependent;
- fixed lane↔row and lane↔frame mappings (compile-time constexpr tables);
- no `-use_fast_math` and no `--ftz` in the new translation unit — the two
  numerics relaxations that *are* proposed (reciprocal-multiply for the 595
  divides; multi-accumulator dots) are individually flagged and each is
  deterministic;
- no dependence on scheduling order: `__syncwarp()` at every phase boundary
  and after every lane-predicated shared write consumed by another lane.

**Bitwise parity vs Rust is explicitly waived for the new f32 path** — and
note the tiering that already exists: the CUDA f64 tier is already only
*horizon-bounded tight-tolerance*, not bitwise, because libdevice
`sin/cos/acos` differ from the CRT by 1 ulp and nvcc contracts FMAs
(`:47-57`). The bitwise gate that matters (Rust ↔ torch-CPU-f64, P4) does not
run through this kernel at all.

**The split is workable, and here is exactly how.** Dispatch today is
`AT_DISPATCH_FLOATING_TYPES(q.scalar_type(), …)` at `:5194`, which instantiates
`scalar_t ∈ {float, double}` and launches `step_kernel<scalar_t, FC>` at
`:5289-5294`. The warp kernel is **float-only** and must never instantiate for
double (it will not fit the register budget and there is no reason to try):

```cpp
if constexpr (std::is_same<scalar_t, float>::value) {
  if (warp_solver) { warp_step_kernel<FC><<<gridW, blockW, shmem, stream>>>(P); return; }
}
step_kernel<scalar_t, FC><<<blocks, THREADS, 0, stream>>>(P);   // unchanged
```

`if constexpr` on the dispatch's `scalar_t` compiles the double branch away, so
the f64 parity path (`tests/test_v10_parity_cuda.py`,
`tests/test_v11_contact_parity.py`) keeps calling a **byte-identical**
`step_kernel<double, FC>`. The old f32 kernel also survives untouched as the
A/B reference and the escape hatch.

Selector: `BEAT_WARP_ENV=0|1` read once in `cuda_env.py` and passed as a new
`bool warp_solver` argument to `step()` — same shape as the adopted A7 escape
hatches (`BEAT_ATTN`, `BEAT_QKV_PACKED`, …, `PERF_AUDIT:40`). Default **off**
until the learning gate passes, then default on with `BEAT_WARP_ENV=0` as the
documented revert. The shell agent owns the signature change and the pybind
docstring at `:5329-5334`.

**Validation ladder** (what replaces "max diff 0.0" where it no longer applies):

1. *Determinism*: two runs of the same build, same seed, byte-equal on all 10
   dumped arrays. Non-negotiable, every stage.
2. *Bitwise where claimed*: stages S1, S2, S4, S5 are gated at max diff
   **exactly 0.0** vs the old f32 kernel, on the five configs this file's
   history standardized on — free-running with `want_contacts`,
   `max_steps=60`, the 128-env random-gear traffic, and K13's `stopjam` /
   `altstop` limit-heavy configs (`:641-658`). Add a **ground-heavy** config
   (feet planted, high `ng`) — the existing five under-exercise the FC ground
   block, exactly as the random-gear micro under-hit the limit rows.
3. *Tolerance tier* (S3 onward): single-substep max relative error vs the old
   f32 kernel < ~1e-5 on `v`, `q`, `p`, `quat`; invariants checked
   (`λ_n ≥ 0`, friction inside the box, `|v_j| ≤ OMEGA_CAP`, `pd_hits` /
   `env_hits` / `fall_hits` rates unchanged). Trajectory equality past ~10
   ticks is *not* a valid gate — the measured e-folding is ~0.35 s (`:47-57`).
4. *Statistical*: 256 envs × 240 ticks under a trained checkpoint — `ep_return`,
   fall rate, and the `nc`/`ng`/`nl` histograms match the old kernel within
   noise.
5. *Learning gate*: a leaf run at the production recipe (`cut_leaf.sh`,
   N=16384, `--foot-contact --gears 5`) compared against the banked leaf
   `ep_return` ladder in `sessions/` (3k/6k/9k/12k/15k). This is the repo's own
   instrument and it is the one that decides adoption.

---

## 6. Q5 — Expected speedup model

**Instruction budget.** Warp-per-env *spends* issue slots to buy latency. Per
env-substep: today ~92,700 scalar ops ≈ ~200,000 machine instructions (with
loads), amortized 32-way by SIMT → **~6,250 warp-instructions per env**. The
warp form issues ~17,400 warp-instructions per env (§4.6 table) → **~2.8×
more issue slots per env**. That is affordable: measured issue utilization is
~0.5% of the SM's 4 instr/cycle, i.e. ~200× of headroom.

**Two independent bounds at 16 warps/SM, N=16384 (128 envs/SM, 8 rounds):**

- *Issue-bound floor*: 128 envs × 8 substeps × 17,400 = 17.8M warp-instructions
  per SM per tick; at 4 IPC = 4.5M cycles ≈ **1.8 ms** (vs 124 ms).
- *Critical-path floor*: ~7,100 dependent steps/substep × ~5 cycles × 8
  substeps ≈ 284k cycles/env-tick ≈ 113 µs; 8 rounds of 16 resident warps →
  ≈ **0.9 ms**.

Both floors land near 1–2 ms, i.e. a nominal 60–100×. **I do not believe that
number and neither should the implementation phase.** This file's own history
is a list of models that did not survive contact: K12 was picked because it
strictly removes live address registers and moved ptxas by *exactly zero*
(`:352-364`); NC_CAP=16 cut the frame by 28% and measured **+3.1% slower**
(`PERF_AUDIT:87-91`); Tier 3.3 was priced at 10–25% of the env kernel and
measured **null** (`PERF_AUDIT:44-52`). The models miss stalls, shuffle
throughput, L1 miss traffic on the row blocks, `__syncwarp` cost, and — above
all — the ptxas allocation phase.

**Grounded estimate instead.** Two multiplicative components, each anchored in
a measurement:

| component | basis | factor |
|---|---|---|
| occupancy 4 → 16 warps/SM | extrapolation of the measured 4→8 step (1.29×) along a saturating curve | **1.6 – 2.0×** |
| per-env traffic + chain restructure (factor to registers; 39,270-FMA solve chain → 210 levels; per-warp working set 1.34 MB → 14 KB, i.e. L1-resident) | model, discounted ~10× against this file's prediction history | **2 – 4×** |
| **product** | | **3 – 8×** |

**Amdahl composition** (env kernel = 64% of total training wall,
`PERF_AUDIT:20-22`):

| kernel speedup S | wall time | wall speedup |
|---|---|---|
| 2× | 0.36 + 0.320 = 0.680 | 1.47× |
| **3×** | 0.36 + 0.213 = 0.573 | **1.75×** |
| 4× | 0.36 + 0.160 = 0.520 | 1.92× |
| **6×** | 0.36 + 0.107 = 0.467 | **2.14×** |
| 8× | 0.36 + 0.080 = 0.440 | 2.27× |
| ∞ | 0.360 | 2.78× (hard ceiling) |

**Headline: 3–8× on `step_kernel`, 1.75–2.27× on total training wall.**

**Honest serial residue.** Even at the model floor, four things do not
parallelize and set the shape of the post-rewrite profile:

| residue | why | post-rewrite share |
|---|---|---|
| PGS row order | Gauss-Seidel; row `r+1` reads row `r`'s `v` | **40–50%** |
| RNEA chain | 10 tree levels × 2 passes, 3-vector math at ≤ 30/32 lanes | 10–15% |
| CRBA composite merge | 30 leaf→root steps at 12/32 lanes | 15–20% |
| Cholesky outer loop | 35 columns × (sqrt → broadcast → divide) | 5–8% |

Second-order: if the env kernel really lands at 5×, the rollout phase shrinks
below the update phase and Tier 2.1 / 2.2 (`PERF_AUDIT:136-143`) re-price
upward — the next audit should be re-run after adoption, not before.

---

## 7. Q6 — Implementation staging

Every stage is independently compilable and independently A/B-testable. The
enabling trick is a **per-phase adapter**: a `gather_to_lane0()` /
`scatter_from_lane0()` pair (35 shuffles or a shared staging tile) so any
subset of phases can run in warp form while the rest runs the existing serial
device functions on lane 0. That is what makes S1 a real correctness baseline
rather than a big-bang rewrite.

Every stage's measurement gate includes, without exception: **`ptxas -v`
regs / spill stores / spill loads / frame recorded before any timing is
trusted** (the K6 spill-cliff rule, `:199-213`), and a re-spilled or
phase-flipped build is a revert regardless of the arithmetic.

| # | stage | files / functions | test gate | measurement gate |
|---|---|---|---|---|
| **S0** | Measurement + plumbing. Instrument the realized `ng` histogram and the `W_tot = 1+3ng+nc+nl` distribution (the same throwaway-counter method K16 used, `:890-899`). Add `warp_solver` arg through `step()` / `cuda_env.py` / `vec_env.py` as a no-op. | `beat_arena_cuda.cu:5181-5297`, `cuda_env.py`, `vec_env.py` | pytest green; existing kernel byte-identical | histograms recorded; ptxas unchanged |
| **S1** | Warp shell, all math on lane 0. New `warp_step_kernel<FC>` (float only) + the adapters; lanes 1–31 idle at `__syncwarp`. Existing `crba`/`rnea_bias`/`chol_*`/`constraint_solve` called unmodified. | new section of `beat_arena_cuda.cu` (or a new `.cu` added to the `srcs` list in `cuda_env.py:126-140`) | **max diff exactly 0.0** vs old f32, 6 configs; determinism (2 runs byte-equal) | expect a large regression (~32×); record it as the harness baseline. ptxas regs/spills recorded |
| **S2** | NV-vector primitive layer (`wdot`, `waxpy`, tail handling, transposes) + **fused multi-RHS triangular sweep** replacing `chol_solve_n` / `chol_solve_multi_n` / `chol_solve_multi_tn` / `chol_fwd_unit*` / `chol_back*` at `:3548`, `:3555-3588`, `:3597-3629`, `:3647-3676`, `:3955`. Row blocks move to per-lane local arrays. | `constraint_solve`, `substep`, `ensure_ecol`/`ensure_pcol` | **max diff exactly 0.0** (row-parallel, not sum-parallel — §4.3) | first real number; expect the harness to go from ~32× slow to within ~2–4× of the old kernel |
| **S3** | **PGS** in warp form: butterfly dots + register axpys for ground (`:3835-3870`), contacts (`:3878-3887`), limits (`:3898-3916`), ω-cap (`:3918-3934`), and the step-11 clamp loop (`:4032-4074`). **First non-bitwise stage.** | `constraint_solve`, `substep` | determinism; tolerance tier (§5.3); statistical rollout gate | biggest single expected win; A/B vs S2, 5 interleaved rounds, fresh process per arm |
| **S4** | **Cholesky factor**: right-looking, factor in registers. `chol_factor_n` at `:2259-2341` gets a warp twin. | `chol_factor_n`, `substep:3953`, `crba` output path | **max diff exactly 0.0** vs S3 (1-ulp fallback if FMA contraction shifts) | ptxas: this is the stage most likely to flip the allocator — 32 regs/lane arrive at once |
| **S5** | **CRBA**: parallel column fill (ancestor-mask form), lane-parallel seeds, 12-lane composite merge in the existing `k = NF-1…0` order. | `crba:1904-2040`, new `ANC_MASK[NF]` constexpr table beside `ANC` at `:1167-1180` | **max diff exactly 0.0** vs S4 | A/B; watch the mask table's cmem/global footprint (K17's lesson at `:511-531`) |
| **S6** | **RNEA**: lane ≡ frame, 10-level shuffle-synchronized forward/backward. | `rnea_bias:2044-2179` | tolerance tier (frame-lane math is the same expression tree; expect 0.0, accept 1 ulp) | A/B; smallest expected win of the math stages |
| **S7** | Tuning ladder + adoption. `__launch_bounds__` sweep {8,12,16,20,24} warps/SM; register-vs-local placement of `g_*`/`c_*`/`ecol`; solve tile width {8,12,16,24}; `NC_CAP` {8,12,16} with an overflow counter; optional numerics flags (reciprocal-multiply for the 595 divides; multi-accumulator dots) each behind its own switch. Then the learning gate and the default flip. | launch config, `Solver` capacities, `cuda_env.py` flag default | full ladder re-gated at each rung; learning gate (§5.5) | interleaved A/B per rung, fresh process per arm, medians of ≥5, non-overlapping ranges required — the standing method of this file |

Weld mode (`FC = false`) is carried through S2–S6 as a second instantiation:
the 12-RHS `mw` solve (`:3548`), the 12×12 `aww` factor/solve
(`:3715`/`:3809`/`:3121`), and the `pcol` projection (`:3739-3749`) all map
onto the same primitives (`N_WELD = 12` fits one lane-round trivially). It is
lower-value (not production) but must not rot; gate it at the same 0.0 in
S2/S4/S5.

---

## 8. Interface assumptions for the shell agent

These are the contract points. Where the shell chooses differently, the
solver core absorbs it with an adapter at the cost noted.

1. **Launch geometry.** One env per warp. `e = (blockIdx.x*blockDim.x +
   threadIdx.x) >> 5`, `lane = threadIdx.x & 31`. Grid = `N` warps. Block: 128
   threads (4 envs) preferred for launch efficiency; 32 also fine. **No
   `__syncthreads` anywhere** — warps in a block share nothing, so inter-env
   divergence inside a block costs zero.
2. **The bounds guard must be whole-warp.** `if (e >= P.N) return;` at `:5140`
   becomes a per-warp predicate; a partially-exited warp breaks every
   `__shfl_sync(0xffffffff, …)` in the solver core. Pad `N` to a multiple of
   the warps-per-block instead, or keep the whole warp alive and predicate the
   stores.
3. **Register budget.** The solver core needs ~122 regs/lane in config B and
   asks the shell to pin `__launch_bounds__(BLOCK, MIN_BLOCKS)` for **≥ 16
   warps/SM**. The old kernel's `__launch_bounds__(THREADS, 1)` at `:5137`
   (which deliberately takes all 255 regs, `:912-924`) must **not** be copied
   into the warp kernel — that reasoning is inverted here.
4. **Shared memory contract.** Per env: `Lhi[3][NV]` (420 B) + NV-vector tails
   (3 floats per live vector; ~200 B) + a 32-word staging tile for phase
   transposes (128 B) ≈ **≤ 1 KB/env**. Allocated by the shell as
   `extern __shared__` with per-warp offset `warp_id_in_block * sizeof(WarpShared)`.
   **Any tile of NV-stride rows must use stride 35 exactly — do not pad to 36**
   (§3.1).
5. **Lane identity.** Canonical mapping is **lane ≡ v-row `i`** for the solver
   core. A second mapping, **lane ≡ frame `k`**, is used inside FK / RNEA /
   CRBA seeds; conversion is a single `__shfl_sync` with a lane-varying source
   from a constexpr permutation table (`F_COORD` is a bijection, `:1023-1026`).
6. **FK delivery (the main ask).** Preferred: the shell's warp FK leaves, in
   registers on the frame-lane, `pos[3]`, `axw[3]`, `r[3][3]` for frame
   `lane`, plus warp-uniform `p[3]` and `r0[3][3]`. If instead FK output lands
   in shared or local, the solver core pays a gather of 30 frames × 15 floats
   = **450 shuffles or 1.8 KB/env of shared** per substep — that is a real
   cost and worth coordinating on.
7. **Distributed state.** `v`, `rhs`, `c`: lane `i` holds element `i`, rows
   32–34 in the shared tail. `q`: lane `j` holds `q[j]` for `j < NQ = 29`.
   The shell's integrator (`:3999-4025`) and clamp block (`:4032-4074`)
   operate in that layout; the clamp block is in my scope (S3) but reads/writes
   `d.q`, so ownership is shared.
8. **Warp-uniform counters.** `s.nc`, `s.ng`, `s.nl`, `s.c_pair[]`, and every
   loop bound derived from them must be **identical in all 32 lanes**. The
   collide phase must produce compacted rows in the same pair-id order as
   `:3474-3515` and broadcast `nc`.
9. **Contact capacity.** The solver core needs a compile-time `NC_CAP` (12
   recommended; measured max 7 production / 8 at 4× stress,
   `PERF_AUDIT:82-85`). The shell owns the clamp and an **overflow counter
   tensor** (same pattern as `pd_hits`, `:1229`/`:5229`) so a capacity breach
   is reported, never silent. Note this is *not* the refuted NC_CAP frame cut
   (`PERF_AUDIT:87-91`): there capacity bought nothing because it shrank bytes
   the kernel never waited on; here capacity is registers and local footprint
   that directly set occupancy.
10. **Sync discipline.** `__syncwarp()` at every phase boundary and after every
    lane-predicated shared write consumed by another lane. All `__shfl_*_sync`
    use the full `0xffffffff` mask and every lane must reach every shuffle —
    no `return`/`continue`/`break` out of a divergent branch that skips one.
11. **Dtype.** Warp kernel is float-only, guarded by
    `if constexpr (std::is_same<scalar_t,float>::value)` inside the existing
    `AT_DISPATCH_FLOATING_TYPES` at `:5194`. `step_kernel<double, FC>` stays
    byte-identical.
12. **No atomics** anywhere upstream of the solver core's inputs.

---

## 9. Q7 — Risks and how each is measured

| # | risk | mechanism | mitigation / measurement |
|---|---|---|---|
| **R1** | **Register pressure per lane** blows the 16-warp target | 122 regs/lane is a design estimate; ptxas will disagree | S7 ladder over `__launch_bounds__`; if the factor alone is the problem, config C (factor in shared, 5.9 KB/env, still 16 warps/SM by shared) is a one-flag fallback. Gate: `ptxas -v` regs + **0 spill stores / 0 spill loads**, recorded per stage |
| **R2** | **ptxas allocation phase change** — the failure mode this file has hit five times (K6's +27.6% spill cliff `:199-213`; K12's exactly-zero `:352-364`; NC_CAP's 254→168 flip at +3.1% slower) | the allocator's target is unstable under unrelated frame changes (`:912-924`) | Never trust a micro-bench delta without the build log. Every stage records regs/spills/frame *first*. A stage that regresses the counters is reverted even if it times faster, and vice versa (the K12 rule: gate on the counters not getting **worse**, then measure) |
| **R3** | **Shared bank conflicts** on the column-major factor / tiles | stride choice | Solved analytically: stride `NV = 35` is coprime with 32 → conflict-free in *both* directions; padding to 36 creates 4-way conflicts (§3.1). Measurement proxy without ncu: A/B stride 35 vs 36 vs 37 in S7 — if 36 is not measurably worse, the tile is not on the critical path and the question is moot |
| **R4** | **`__syncwarp` discipline** — Volta+ independent thread scheduling makes a missing sync a *silent* wrong answer, not a hang | divergent shuffles, lane-predicated shared writes | Full-mask `__shfl_*_sync` everywhere; `__syncwarp()` after every predicated shared write; the whole-warp bounds guard (§8.2). Detection: `compute-sanitizer --tool racecheck` / `synccheck` on a small N in CI — these do **not** need the locked ncu counters |
| **R5** | **Divergence between envs in the same block** | none by construction at W=32 (1 env/warp, no `__syncthreads`) | Non-risk at W=32. Becomes real at the W=16 fallback (2-way), where the K13 union-of-widths tax (`:666-704`) returns for the 4/2/1 ladders — so at W=16, drop the ladders and always run the full width |
| **R6** | **Mid-PGS lazy solves.** `ensure_pcol` at `:3923` can trigger a full unit solve *inside* the PGS loop when a joint exceeds `OMEGA_CAP` and its column is not cached | in warp form that is a 70-level sweep injected into the PGS chain, i.e. a latency spike | S0 must instrument the engagement + miss rate (K16's counter method). If the miss rate is non-trivial, hoist: precompute `ecol` for all `NQ` joints in the fused S2 sweep (29 more RHS — the sweep is level-bound, not width-bound, so the marginal cost is instructions, not levels) |
| **R7** | **`ng` is unmeasured** and it sets both the fused-solve width and the PGS row count — the largest single term in §4.6 | production may sit anywhere from ng=2 to ng=8 | S0 blocker. If ng is routinely 8, the solves/PGS shares rise and the win grows; if ng ≈ 2, the CRBA/RNEA residues dominate sooner and the estimate moves to the low end of 3–8× |
| **R8** | **Code size** from fully unrolling the 595-step factor nest and the fused sweep | the existing kernel is already ~38k SASS instructions (`:290-294`) | Keep the *outer* loops rolled where indices need not be static; only the register-indexed dimension must unroll. Measure SASS size per stage — but remember this file's rule that static counts are not dynamic traffic (`:290-294`) |
| **R9** | **A tolerance-tier regression that only shows up in learning**, not in a 60-step diff | chaotic amplification (~0.35 s e-folding, `:47-57`) | The learning gate is mandatory before the default flips, and `BEAT_WARP_ENV=0` stays as a permanent escape hatch |
| **R10** | The whole thing lands at 1.5× instead of 4× | model risk, per §6 | Staging is ordered so the two biggest structural wins (S2 fused solves, S3 PGS) land **first** and are individually measurable. If S2+S3 together do not beat the old kernel, stop — S4–S6 cannot rescue it (they are 14% of today's ops combined) |

---

## 10. Open questions

1. **The realized `ng` histogram** (active ground corners per env-substep,
   trained policy, production pool). Never measured; the `nc` histogram was
   (max 7, 92.7% zero, `PERF_AUDIT:82-85`) but `ng` is the bigger term. **S0
   blocker.**
2. **The `W_tot = 1 + 3·ng + nc + nl` distribution**, which sets the fused
   solve's tile count. Also S0.
3. **The ω-cap `ensure_pcol` miss rate inside PGS** (`:3923`) — decides R6.
4. **Is the 42 KB/thread frame actually costing DRAM bandwidth?** My estimate
   from the op counts is ~70 GB/tick at N=16384, i.e. ~58% of the 4090's
   roofline, which would make the kernel substantially bandwidth-bound and
   makes the register-resident factor worth more than the latency argument
   alone. The audit calls the kernel latency-bound and the residency probe
   (`PERF_AUDIT:59-66`) only cleared the *global pool*, not the local frame.
   ncu is locked, so the discriminator has to be a synthetic: run the existing
   kernel at N=1024 (43 MB of local frame, L2-resident) and at N=16384
   (688 MB) and compare per-env cost *after* correcting for the occupancy
   difference — or simpler, an arm that shrinks only `c_J`/`c_minv` capacity
   (55 → 12) which cuts frame bytes without touching arithmetic. Note the
   NC_CAP probe already did something close to this and measured *slower* via
   a ptxas flip, so the experiment must pin the register target.
5. **Does any of the K13 3.5× divergence figure survive on sm_89?** The header
   number is sm_61 (`:705-713`) and Tier 3.3's partial fix measured null
   (`PERF_AUDIT:44-52`). W=32 captures it for free if it exists; nothing in
   the plan depends on it.
6. **Block shape** — 1 warp/block (today's `THREADS = 32`) vs 4 warps/block.
   Interacts with shared-memory allocation granularity and with the blocks/SM
   cap. Shell's call; the solver core is indifferent.
7. **`ecol` cache capacity.** 29 slots costs 29 floats/lane of local. A 12–16
   slot compacted cache (warp-uniform slot table, now cheap because misses are
   warp-uniform) would save ~half of it; K16 measured a 0.12% miss rate at
   full capacity (`:890-910`) but says nothing about a 12-slot cache.
8. **Delassus-form PGS** (§4.5) — halves the per-row work but not the row
   chain. Worth a probe after the first landing, not before.
9. **Whether `CLAMP_PASSES = 64` (`:968`) can be raised** now that the
   early-out is warp-uniform: today a warp pays the max over 32 envs, so the
   pass count is effectively pinned by the worst env. This is a *fidelity*
   opportunity created by the rewrite, not a perf one, and it is a physics
   decision for the user.
10. **Weld-mode priority.** It is not production (`cut_leaf.sh:99` uses
    `--foot-contact`) but it is the v10 semantics path. Confirm whether it
    needs the warp treatment at all, or whether `BEAT_WARP_ENV` should simply
    fall back to the old kernel when `foot_contact = false`. That would remove
    a whole instantiation from S2–S6 and is my recommendation absent an
    objection.
