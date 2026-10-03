// Beat Arena env v10 sim step as ONE CUDA kernel launch (P5).
//
// Full SPEC-v10 pipeline per env: 8 x substep (FK -> CRBA -> RNEA ->
// tree-sparse Cholesky (audit #10, TreeTabs) -> constraint solve [12 x
// ground block / contacts / limit rows / omega-cap pass, COMPOSITE_EPS
// 1e-4] -> integrate -> step-11 clamp passes -> step-12 root envelope)
// + gear->tau, activation + torque telemetry, blade sweep/cut judgment, scoring,
// obs write (OBS_DIM below) — a semantics-exact transcription of rust/src/lib.rs +
// rust/src/dynamics/{model,fk,algo,collide,solver,step}.rs. The Rust
// crate remains THE reference implementation.
//
// ENV v11 (2026-08-08, "Unilateral foot contact" spec): the 8 sole-box
// corners get unilateral frictional GROUND rows against the plane
// y = 0 (1 normal + 2 box-clamped friction per ACTIVE corner), plus a
// fall termination at root y < FALL_ROOT_Y. This is the ONLY foot mode:
// the v10 welded-feet mode (12 bilateral foot-weld rows, the former
// FC = false instantiation) was deleted on 2026-09-27 (issue #2). The
// K-series perf notes below predate that and mention the weld block
// (s.mw / s.aww / A_ww, ensure_pcol's projection) as provenance only.
//
// Parallelization (P5 decision, profiled — see the session note):
// ONE THREAD PER ENV. The whole substep is a serial transcription of
// Rust's loops in thread-local (local-memory) scratch — every
// accumulation runs in Rust's exact order by construction, so there is
// no reduction-order risk anywhere (candidate B's intra-env parallelism
// was rejected: PGS is strictly ordered and the CRBA/RNEA tree passes
// are sequential). Occupancy comes from batch width: the production
// configs run N = 1024..8192 envs; per-thread scratch (45,296 B f32,
// 91,504 B f64 — measured, not estimated) lives in local memory
// (coalesced per-thread interleaving), so residency is register-bound,
// not smem-bound.
//
// CORRECTION (2026-08-03, A1). That rejection note used to end "...and
// the only wide parallelism — multi-RHS triangular solves — would buy
// ~2x at 3x the parity risk". The risk half was WRONG and is retracted.
// Batching across right-hand sides is a pure LOOP INTERCHANGE, not a
// reassociation: the RHS are independent, each keeps its OWN
// accumulator visiting the same terms in the same order, so every value
// is bitwise what the separate solves produced (argued per site below,
// gated at max diff 0.0 across f32 and f64, free-running and
// max_steps=60, want_contacts). It also needs no cross-thread
// parallelism — the batches run inside the one thread, as ILP. A1
// landed them (the since-deleted weld block, contacts, limit-row unit
// columns, plus the weld dot/axpy interchanges) for +33% end to end: train --bench
// N=1024 5,496 -> 7,283 steps/s, env micro-bench 6,212 -> 8,264
// env-steps/s. The "~2x" estimate of the prize was the honest half.
//
// f64 parity tier (SPEC-v10 Divergences §21, measured by a since-
// retired probe; numbers in §21a): CUDA f64 sqrt and division are
// IEEE-correctly-rounded and match Rust's CRT bitwise; CUDA f64
// sin/cos/acos (libdevice) differ from the CRT libm by 1 ulp on a few
// %% of inputs, and nvcc's default FMA contraction re-rounds
// multiply-add chains — both are <=1-ulp-per-op seeds that the v10
// chaotic dynamics amplify at the measured e-folding (~0.35 s), so the
// CUDA f64 tier is HORIZON-BOUNDED tight-tolerance (the v9 CUDA-tier
// house precedent), not bitwise like torch-CPU-f64. The bitwise gate
// remains Rust<->torch (P4); this backend's job is f32 training
// throughput.
//
// f32 training safety net (SPEC-v10 Divergences §20e mirrored):
// Cholesky pivots floored at 1e-8, engagements counted per tick into
// the pd_hits tensor (popped by the wrapper, reported not hidden).
//
// Env-layer judgment (swept-surface SAT, 115-pt scoring, tracker ring,
// live REACH/miss gates, obs writer) carries the v9 kernel's design
// with the v10 inputs: aggregated follow-through ring (same-tick
// trackers share (arc, age) — one slot with a note count is EXACT; the
// ring is walked from tr_ptr so resolution pays in Rust's spawn order),
// chart pool with padded (P, C) tensors, auto-reset via the cand
// tensor, zero host syncs in the step loop.

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_runtime.h>

#include <limits>
#include <vector>

namespace {

// ---- dimensions -------------------------------------------------------------
constexpr int NQ = 29;
constexpr int NV = 35;
constexpr int VOFF = 6;
constexpr int NF = 30;
constexpr int NUM_COLLIDERS = 12;
constexpr int NUM_PAIRS = 55;
constexpr int N_TRACKERS = 16;  // FOLLOW_TIMEOUT + 1 (aggregated ring)
constexpr int N_NOTE_SLOTS = 24;  // env v18: 16 -> 24 (eviction study)
// env v17 (contact spec §12): shared entity window — notes AND walls
// ride the N_NOTE_SLOTS slots, nearest-first by head distance;
// per-token = the v9 note lanes then [is_wall, width, height,
// back-face z] (last lane carried wall length until 2026-08-22;
// retired as redundant with back-face minus front-face lane 2).
constexpr int NOTE_FEAT = 18;
constexpr double WALL_PEN = -1.5;       // ONCE, on the head-entry tick (2026-08-22: -0.5 -> -1.5, bomb parity per user)
constexpr double WALL_FRONT_OFF = 0.25; // viewer front-face offset (m)
// ============================ PROVENANCE WARNING =====================
// (2026-08-16) EVERY measurement, occupancy figure, and perf verdict
// in this header and the K-series notes below was taken on sm_61
// (20-SM GTX 1080). None has been re-measured on the production RTX
// 4090 (128 SMs, sm_89) or at the current batch shapes. Treat them as
// provenance for the do-not-retry list, NOT as current truth. The
// current measurement plan (Nsight run, bandwidth-vs-latency
// disambiguation, layout candidates) lives in
// PERF_AUDIT_2026-08-16.md.
// =====================================================================
//
// One WARP per block: the grid is N/32 blocks, so small batches spread
// over all 20 SMs. Throughput is INSENSITIVE to the block size (P5
// measurement, 128 vs 32: 216 vs 221 ms at N=1024) — consistent with
// the diagnosis below, and the reason occupancy is not the lever here.
//
// The kernel is SERIAL-LATENCY bound (per the sm_61 measurements —
// the 4090's local-memory picture may differ; see the audit). Earlier notes called it "serial
// latency + local-memory traffic"; the 2026-08-03 A4/A1 measurements
// separate the two, and traffic is not the binding constraint:
//   * A1/1c: batching the dense back substitution of the limit-row unit
//     columns 4-wide won +6.5%; widening the SAME batch to 8, which
//     halves the factor-load traffic again on top of that, won ZERO
//     (7,283 -> 7,286 steps/s). The benefit saturates with the number
//     of independent dependency chains, not with the loads removed —
//     a latency signature.
//   * A1/1f: register-blocking the A_ww assembly cut its s.mw loads
//     6.5x (78*NV -> 12*NV) and measured NEUTRAL on train --bench in
//     both a 4-wide and a full-row form (the full-row form also pushed
//     f32 spills 6,948/4,784 -> 8,576/8,100). REVERTED — a pure
//     traffic cut with no new chains does not pay here.
//   * A4 (previous session): the unit-RHS forward substitution, which
//     cuts the dependent chain without cutting the factor's traffic,
//     paid +21%.
// The working set (45 KB stack/thread) is still far beyond L2, so
// traffic is not free — it is simply already hidden behind the chains.
//
// K4/K5 (2026-08-03) REFINES that, and the refinement is the important
// part: per-item verdicts on this kernel are ORDER-DEPENDENT, so a
// single measurement of an item in isolation is not a verdict.
// Measured at N=4096 (env-only micro-bench, median-of-3, budget 600k):
//   K4c dead upper-triangle stores (~970/substep)      +0.24%
//   K4a batched unit forward substitution              +0.67%
//   K4b pivot chain software-pipelined into the rows   +0.37%
//   K4e inertia_world cached crba -> rnea_bias         -0.06%  (!)
//   K4d broad-phase sphere reject                      +0.03%  (!)
//   K5  weld-Jacobian sparsity (35 -> 12 columns)      +8.92%
// K4e and K4d were REVERTED on those numbers, then RE-PROBED after K5
// landed and both flipped decisively positive: K4e +1.15%, K4d +0.80%.
// The reason is register/spill pressure, not arithmetic. K5 cut the
// three weld contractions from NV to 12 terms, which took f32 spills
// 6,984/4,768 -> 4,516/3,328 B; under the old pressure K4e's +1,104 B
// frame and K4d's extra live values each bought back more spill than
// they saved in work, and the same code became a clear win once the
// budget had slack. Whenever an item lands that moves spills, re-probe
// the items that lost by less than ~1%.
//
// K10 (2026-08-07) — the housekeeping bundle, measured the same way
// (N=4096, 4+ interleaved A/B rounds of 35 s; within-arm spread ~0.1%):
//   K10a delete the dead RNEA vo chain                   +0.21%  KEPT
//   K10b skip axw for the one F_COORD < 0 frame          +0.04%  kept
//                                                (neutral, dead work)
//   K10c constant HLR table                          NOT APPLICABLE
//   K10d bound refresh_fk to the upper-body prefix       +0.00%  REVERTED
//   K10e1 drop write_obs's re-load_state                 -0.11%  REVERTED
//   K10e2 reuse d.r0 instead of re-running quat_to_rot  NOT BITWISE
//   K10e3 start the obs zero fill at OFF_NOTES           +0.08%  KEPT
// Three of those are worth carrying forward as rules:
//  * K10c was already done BY THE COMPILER. The hlr[] "per-substep
//    sqrt" is over constexpr COL_A/COL_B, and nvcc folds the whole
//    expression at compile time in BOTH dtype branches — the f32 SASS
//    carries 0x3ebae148 (0.365f) etc. as MOV32I immediates, the f64
//    SASS the matching double halves. Check the SASS before hand-
//    tabulating a "recomputed" constant.
//  * K10d is structurally unmeasurable, not merely small: refresh_fk
//    runs ONCE per control tick against fk_all's EIGHT (one per
//    substep), so dropping its 12 leg frames removes ~4% of one ninth
//    of the FK, and FK is not the bottleneck. Do NOT re-probe it after
//    a spill shift — it moved the ptxas numbers by exactly zero.
//  * K10e1 is a second instance of the A1/1f finding, now with a
//    NEGATIVE sign: cutting local/global traffic without shortening a
//    dependency chain does not pay here, and a redundant reload can be
//    load-bearing as a rematerialization point (deleting it grew the
//    f64 frame 90,976 -> 91,008 B).
//
// K6 (2026-08-07) — ROW-BLOCK (unroll-and-jam) THE FORWARD
// SUBSTITUTIONS. Every triangular solve in this kernel used to walk one
// row at a time: per (i, k) it loaded l[i*n+k] plus the W right-hand
// sides and did W FMAs. Blocking R rows shares the k < i0 sweep — R
// l-loads + W x-loads buy R*W FMAs — and, more importantly on a
// serial-latency-bound kernel, gives R independent accumulator chains
// where there was one. Measured N=4096, 4 interleaved A/B rounds each,
// every rung added ON TOP of the previous one:
//   chol_solve_multi_n fwd R=2 (W=12/4/2)               +1.21%  ->R=3
//   chol_solve_n fwd R=4                                +0.75%  KEPT
//   chol_fwd_unit4_n fwd R=2                            +0.72%  ->R=4
//   chol_fwd_unit2_n + chol_fwd_unit_n fwd R=4          +1.74%  KEPT
//   chol_fwd_unit4_n R=2 -> R=4                        +27.62%  KEPT
//   chol_solve_multi_n R=2 -> R=3                       +0.14%  KEPT
//   chol_solve_multi_n R=3 -> R=4                       +0.03%  REVERTED
// Package: 18,057 -> 24,180 env-steps/s, +33.9%.
//
// THE +27.6% IS A SPILL CLIFF, NOT ARITHMETIC — and it is the single
// most important fact this task produced. Widening chol_fwd_unit4_n's
// block from 2 to 4 changed the f32 kernel from 255 registers with
// 4,588/3,296 B of spill stores/loads to 249 REGISTERS AND ZERO SPILLS
// (frame 45,056 -> 43,024 B); the f64 kernel, which cannot fit either
// way, moved by ~1% and its arm gained ~nothing. So:
//  * The f32 kernel now sits ON the edge of a cliff with ~6 registers
//    of headroom. Any later item that adds live values can push it back
//    over and cost ~20% — check `0 bytes spill stores` in the build log
//    BEFORE trusting a micro-bench delta, and treat a re-spilled build
//    as a revert regardless of what the arithmetic says.
//  * The K4/K5 ORDER-DEPENDENCE rule above now applies with much more
//    force: the whole sub-1% shelf (and anything previously reverted at
//    a small negative) was measured under 4.6 KB of spill traffic that
//    no longer exists. Re-probe it against THIS build.
//
// ABSOLUTE PROHIBITION: the BACK substitutions are NOT row-blockable —
// not "unmeasured", structurally illegal. The back pass runs i
// DESCENDING and accumulates k ASCENDING from i+1, so for a block of
// rows i0 > i1 > ... the region k > i0 that they share is the TAIL of
// each row's required order, not a prefix; hoisting it into a shared
// sweep would make every blocked row consume its high-k terms FIRST,
// which reassociates the sum. And the terms that would have to run
// first instead — the near-diagonal k in (i_r, i0] — are exactly the
// x[k] that the earlier rows of the same block have not produced yet.
// There is therefore no schedule that both preserves each row's
// ascending-k order and shares work across the block, at any R. Only
// the forward direction has the shared PREFIX that makes the blocking
// bitwise. RHS-batching (chol_solve_multi_n, chol_back{2,4}_n) is a
// different transform — a loop interchange over independent right-hand
// sides — and remains legal in both directions, as today.
//
// K7 / K8+K9 / K11 (2026-08-07) — weld-block items (batched
// weld-null projection, refined weld-Jacobian support, RHS-minor s.mw):
// their code went with weld mode on 2026-09-27; the K8 support layout
// survives in the ground block (GroundRows) and K11's lesson in the
// K18(i) note below. Full notes: git history before issue #2.
//
// K12 (2026-08-07) — COLUMN-MAJOR CHOLESKY FACTORS, the same
// live-address transform applied to the OTHER operand. Both matrices
// that are ever passed as a factor (d.m, 35x35, and s.aww, 12x12) now
// store element (row R, col C) at [C*n + R]; the ten chol_* functions
// and the ~10 producer stores in crba / the A_ww assembly swap their two
// index terms and NOTHING else moves. Worth +3.95%: 27,393.8 ->
// 28,475.6 env-steps/s at N=4096 (5 interleaved A/B rounds, medians,
// ranges 27,384.2-27,406.3 vs 28,468.5-28,504.2 — non-overlapping by
// ~50x the within-arm spread). f64 16,728.9 -> 16,824.2, +0.57%.
// MECHANISM: every hot loop walks a COLUMN of the factor while varying
// the row, because the forward substitutions are row-blocked (K6) and
// the back substitutions read l[k*n+i] with k ascending. Row-major made
// each of those R rows its own runtime address at stride n — 5 live in
// chol_factor_n's inner loop (pivot row + 4 blocked rows), 8 in
// chol_fwd_unit4_n. Column-major collapses them onto ONE base with
// immediate offsets +0..+3, and turns the back pass's strided walk into
// a contiguous one. Same argument as K11, now on the l side.
// BITWISE by construction and gated at max diff 0.0 on f32 AND f64,
// free-running and max_steps=60, want_contacts on: a bijective address
// remap leaves every FP operation with identical operands, identical
// order and identical accumulator seeds at UNCHANGED inline sites, so
// the K10e2 FMA-contraction hazard (which needs a MOVED inline site)
// cannot fire. The producers are value-safe because M and A_ww are
// symmetric — relabelling which triangle of the buffer is occupied does
// not change any stored number.
// THE PREDICTION THAT FAILED, and it is the useful part: this item was
// picked as "the one candidate that strictly REMOVES live address
// registers, so it can widen the 255-register cliff margin". It moved
// the register allocator by EXACTLY ZERO. ptxas is byte-identical to the
// K11 build in both dtypes — f32 41,872 B / 0 / 0 / 255 regs, f64
// 82,912 B / 8 / 8 / 255 regs — and so is the LDL/STL census (f32
// 6,548/3,042, f64 6,570/2,901, unchanged to the load); total SASS grew
// trivially (f32 38,070 -> 38,106, f64 43,290 -> 43,350). So a +3.95%
// win is once again INVISIBLE to every static counter this file tracks.
// Together with K11 that is now two of the round's largest gains with a
// flat census, and the rule hardens: on this kernel, ADDRESS-FORM
// changes pay in issue slots and scheduling that neither ptxas -v nor a
// SASS census can see. Do not gate such an item on the counters moving —
// gate it on the counters not getting WORSE, and measure.
//
// K18(i) (2026-08-07) — RHS-MINOR s.c_minv: BUILT, GATED, AND REJECTED
// at -0.33%. DO NOT RE-TRY IT. The round-5 audit ranked this straight
// after K12 as "K11's argument verbatim" on the contact scratch: store
// c_minv as [NV][NUM_PAIRS] instead of [NUM_PAIRS][NV] so the 4-RHS
// contact solve's lane sweep collapses from 4 runtime stride-NV
// addresses onto one base with immediate offsets. Exactly that was
// built — chol_solve_multi_tn generalized to <W, STRIDE> with base
// &c_minv[0][ci] and STRIDE = NUM_PAIRS — in TWO variants, differing
// only in the 1-wide remainder tier (chol_solve_multi_tn<1> vs a
// strided chol_solve_sn<STRIDE> twin of chol_solve_n that keeps its
// R = 4 forward blocking). Every gate passed except the last:
//   ptxas BYTE-IDENTICAL to the incumbent in both dtypes and both
//     variants — f32 41,872 B / 0 / 0 / 255 regs, f64 82,912 B / 8 / 8
//     / 255 (so the counters were not "not worse", they were equal);
//   bitwise old-vs-new max diff EXACTLY 0.0, f32 AND f64, free-running
//     and max_steps=60, want_contacts, plus a 128-env random-gear
//     traffic config added specifically because the 8-env chart configs
//     never reach nc >= 4 and would not have exercised the 4-wide tier;
//   pytest 172 green;
//   A/B, 5 interleaved rounds, N=4096 f32, fresh process per arm:
//     incumbent            28,492.7  (28,464.5 - 28,521.6)
//     multi_tn<1> variant  28,382.1  (28,359.5 - 28,416.7)   -0.34%
//     chol_solve_sn variant 28,400.2 (28,381.1 - 28,404.3)   -0.32%
//   Both ranges non-overlapping in the WRONG direction, and the two
//   variants agree to 0.02%, which rules out the remainder tier's
//   blocking width as the cause: the LAYOUT itself is the loss.
// WHY IT INVERTS K11's SIGN — this is the rule the round was missing.
// RHS-minor pays only when the INNER loop of the hot consumers is the
// LANE loop AND the lane count is reliably > 1. s.mw meets both
// conditions: all four of its consumers sweep the 12 weld rows
// innermost at a fixed column, so 12 live addresses become one base.
// s.c_minv meets NEITHER where it matters.
//   * The lane count is almost always ONE. Contact-count histogram
//     under bench traffic (256 envs x 240 ticks, pairs active per
//     env-tick): 50,816 zero, 7,670 one, 2,238 two, 555 three, 161 four
//     or more. The 4-wide tier that motivated the item is the rare
//     path; the 1-wide tier, where a "lane sweep" is a single address
//     in either layout, is the common one.
//   * Three of the four consumers vary the ROW index i at a FIXED
//     contact: the seed copy, the c_diag dot, and the PGS contact axpy
//     — the last of which runs NV terms per active contact per PGS
//     iteration x 12 iters x 8 substeps and is the hottest c_minv
//     reader in the kernel. Row-major makes that walk contiguous;
//     RHS-minor makes it stride-NUM_PAIRS. So the transform
//     de-optimizes the hot consumers to optimize a batch width that is
//     seldom taken.
// GENERAL FORM, now the third data point after K11 (+7.06%) and K12
// (+3.95%): before transposing a scratch buffer, ask which index each
// consumer's INNER loop varies, weighted by how often that consumer
// actually runs — not whether the shape matches a previous winner.
// "Same transform as the last win" is not a mechanism.
// (K18(ii), the ecol contiguous-scratch variant, was out of this task's
// scope and remains unprobed.)
//
// K14 (2026-08-07) — COMPILE-TIME CONTACT-ROW SPARSITY: BUILT, GATED
// AND REJECTED IN THREE FORMS. DO NOT RE-TRY IT. The derivation is
// sound and is preserved below so nobody re-derives it; what is dead is
// the idea that s.c_J's structural zeros are worth skipping.
// THE STRUCTURE (verified against the model tables, not estimated).
// Contact row ci is two point_jacobian_acc calls on COL_FRAME[PAIR_A]
// and COL_FRAME[PAIR_B], so
//     support(pid) = {0..5} u chain(fa) u chain(fb),
// with chain() the wj_chain_mask walk — 12 to 23 of NV = 35 columns,
// mean 15.95. It is NOT always two runs (pid 28, colliders 3 and 9, is
// {0..8} u {16,17,18} u {29,30,31}); but a two-segment COVER always
// exists — split the support at its widest interior gap, [0..h0] u
// [l1..h1] — because covering a structurally-zero column is harmless
// while missing a support column is not. Mean cover 17.11 of 35, so
// the dense loops would have lost ~51% of their terms AND half of each
// dot's serial FFMA chain. Table packed h0 | l1<<8 | h1<<16, one
// constexpr entry per pair id, static_asserted to be exactly that
// minimal cover of c_pair_mask(pid).
// WHAT WAS MEASURED (N=4096 f32, 5 interleaved A/B rounds each, fresh
// process per arm, incumbent re-measured inside every pair):
//   stage 1, the three c_diag dot sites     28,417.5 -> 28,405.6
//                                           -0.04%, ranges OVERLAP = noise
//   stage 2a, PGS dot, two dynamic loops    28,419.6 -> 28,016.0   -1.42%
//   stage 2b, same + #pragma unroll 4       28,421.5 -> 28,165.5   -0.90%
//   stage 2c, unrolled NV loop + predicate  28,433.8 -> 28,350.1   -0.29%
// (2a/2b/2c ranges all non-overlapping in the wrong direction; 2c also
// FAILED the ptxas gate on its own — f32 frame 41,872 -> 41,936 B.)
// Everything else passed: bitwise old-vs-new max diff EXACTLY 0.0 on
// f32 AND f64 across free-running, max_steps=60 and the 128-env
// random-gear traffic config (which reaches nc = 7 and touches 48 of
// the 55 pair ids, so the table itself was well covered); pytest 172
// green; f32 stayed 255 regs / 0 / 0 / 41,872 B in 2a and went to 254
// regs in 2b, f64 kept its 8-byte slot throughout.
// THE RULE, and it is the useful part. A sparsity narrowing on this
// kernel must not cost a loop its COMPILE-TIME TRIP COUNT. The dense
// `for (i = 0; i < NV; i++) jv += c_J[ci][i] * v[i]` compiles to a
// fully unrolled straight-line block: 70 LDL at immediate offsets off
// two bases, 35 FFMAs, no branches, every load issuable ahead of the
// chain. Keying the bounds on a RUNTIME value (s.c_pair[ci]) turns it
// into a real loop, and the loss tracks exactly how much of that form
// is destroyed — -1.42% (two dynamic loops) -> -0.90% (4x unrolled) ->
// -0.29% (fully unrolled, runtime predicate). That monotone ordering
// is the proof: the saved arithmetic was never the cost, the FORM was,
// and the best case of the series extrapolates to zero.
// WHY K5/K8 PAID AND THIS CANNOT: the weld rows' support is a function
// of the ROW INDEX, which is a compile-time constant inside the
// `#pragma unroll` nests, so those narrowed loops kept immediates and
// full unrolling (the A_ww site's header records that removing the
// pragma there costs -0.12% for the same reason). "Structural sparsity"
// with a runtime-indexed support is a DIFFERENT item and does not
// inherit K5's result.
// Two further reasons it could not have paid even in the best form:
// contact work is absent on ~84% of env-ticks (the K18 histogram, and
// this task's own probe: 25,325 of 30,720 env-ticks at nc = 0), and
// a ~1% swing from a path that runs on 16% of ticks means most of the
// measured loss was a whole-kernel scheduling/allocation phase change
// (2b moved the f32 allocation to 254 registers, 2a/2c moved the f64
// frame), not contact arithmetic at all — the K6/K12 phase-change
// lesson again, now with the sign against a "pure" work reduction.
//
// K17 (2026-08-07) — CONSTEXPR ANCESTOR TABLES. Of the three F_PARENT
// pointer-chases the audit named, ONE landed (crba, +0.21%) and two
// were measured and reverted. Tables at F_COORD; the crba site is the
// only consumer.
// THE SASS PRE-CHECK, run before any edit (nvcc -lineinfo + nvdisasm
// -c -g on the shipped build): none of the three chases was constant-
// folded, so there was no folding to lose. F_PARENT / F_COORD are
// __device__ constexpr, i.e. GLOBAL memory read through the const
// cache, and every walk carries a loop-carried `LDG.E.CI -> LEA ->
// LDG.E.CI` pair — the next frame index is not addressable until the
// previous load returns, and the F_COORD load that produces the write
// column hangs off it. That includes the WELD walk, whose `frame` IS a
// compile-time 23/29: nvcc unrolls the widx loop into two copies and
// still leaves each one a rolled chase over k. (It does fold the
// per-step `F_COORD[k] >= 0` test, into a MOV32I 0x3efbfffe + SHF.R.U64
// bitmask — that half was already free.)
//   ANC_N / ANC.f / ANC.c              28,413.6 -> 28,473.9   +0.21% KEPT
//   + weld walk, unrolled to 6         29,488.0 -> 29,492.7   +0.02% REVERTED
//   + point_jacobian_acc               28,476.5 -> 28,388.1   -0.31% REVERTED
// (crba: 10 interleaved rounds, 9 of 10 paired wins, and the first
// 5-round batch non-overlapping — 28,378.6-28,416.3 vs 28,458.9-
// 28,498.0. f64 16,903.4 -> 16,920.8, +0.06%. The weld and pja rows
// were probed against the crba build, in interleaved batches; the weld
// batch ran in a transient boost clock regime, which is why its
// absolute numbers are ~1,000 higher — the pairing is what carries it,
// and the -0.31% package reproduced exactly at both clock regimes.)
// ptxas BYTE-IDENTICAL to the incumbent in both dtypes: f32 41,872 B /
// 0 / 0 / 255 regs, f64 82,912 B / 8 / 8 / 255 — the tables live where
// F_PARENT lives, in global memory, not in the frame. Bitwise max diff
// EXACTLY 0.0 on f32 AND f64 across free-running, max_steps=60 and the
// 128-env random-gear traffic config; pytest 172 green.
// WHY ONLY +0.21% WHEN THE CHASE IS 117 SERIAL STEPS PER crba CALL: the
// chase latency was already mostly hidden. ptxas hoists the F_PARENT
// load to the top of the walk body and the body is ~50 instructions
// deep, so what the table actually trades is a dependent load for
// ADDRESS ARITHMETIC — and the arithmetic is where the sign lives. Two
// [NF][10] arrays cost a 64-bit XMAD chain for k * 40 against each of
// two bases (~12 integer instructions per unrolled group) and measured
// +0.13%; the same tables at a POWER-OF-TWO [NF][16] stride, which
// makes the row offset k << 6 and puts both rows a compile-time
// distance apart off one base, measured +0.09% ON TOP of that (5 of 5
// paired rounds) and is what ships.
// PACKING IS THE WRONG DIRECTION, and this is the transferable half.
// Folding frame and column into ONE word (frame | col << 8, [NF][16])
// removes a load and a base register — the K11/K12 instrument says it
// should win — and it measured NEUTRAL-to-NEGATIVE against the
// incumbent (28,414.9 -> 28,413.9) and clearly worse than the two-array
// form. The unpack `p & 0xff` sits ON the dependent path between the
// load and the d.pos[anc] address, which is the one place this kernel
// cannot absorb an extra op; the two arrays' loads are INDEPENDENT of
// each other and issue together. So: on a latency-bound site, count ops
// on the dependent path, not loads.
// WHY point_jacobian_acc INVERTS THE SIGN — the K18(i) frequency rule
// again, now on a table instead of a layout. The edit is identical to
// crba's and the site is structurally the same walk, but it runs in the
// CONTACT block, which is idle on ~84% of env-ticks (the K14/K18
// histograms). Turning the rolled chase into a table loop let ptxas
// unroll it: the two inlined copies went from ~66 instructions and 2
// LDG.E.CI to ~216 instructions and 22, plus a 10-BRA unroll-remainder
// ladder, all of it in a rarely-taken block. A chase costs its latency
// only when it runs; a table costs its code and its loads every time
// the block is scheduled around.
// THE WELD WALK, by contrast, DID fold exactly as designed — with widx
// unrolled and the length static_asserted at 6, every k and col became
// an immediate and both LDG.E.CI disappeared (cmem[2] 876 -> 900 B) —
// and it measured +0.02%, i.e. nothing. 96 chase steps per tick against
// crba's 960: the site was never worth the code. It is REVERTED rather
// than kept-as-neutral because it adds a forced unroll and a
// compile-time coupling for no measured return (the weld walk itself
// was deleted with weld mode on 2026-09-27).
//
// K19 (2026-08-07) — SPLIT write_obs INTO ITS OWN KERNEL: BUILT, FULLY
// GATED, AND REJECTED at -0.28% / -0.33%. DO NOT RE-TRY IT. The round-5
// audit ranked this at +0.5-2% on the premise that step_kernel would get
// structurally smaller once the obs writer's live ranges left it. That
// premise is FALSE, and the reason it is false is the finding.
// WHAT WAS BUILT: a second `__global__ obs_kernel<T>(Sim<T> P)` over the
// same envs, carrying an `ObsState<T>` of exactly the 71 words
// load_state reads (p/quat/q/v) instead of a Dyn/Solver frame;
// write_obs's tail deleted from step_kernel for ALL THREE modes; the new
// kernel launched right after step_kernel on the same stream from the
// host `step()` wrapper (so the CUDA-graph capture in rollout_graph.py
// just records a second node — no Python change). 128 threads, no
// __launch_bounds__, chosen once and not tuned.
// STEP 1, THE NULL-LAUNCH FLOOR, measured first and it is the one number
// that came out as hoped: an EMPTY second kernel with the same Sim<T>
// parameter block costs NOTHING. 5 interleaved rounds, N=4096 f32:
// incumbent 28,456.0 (28,448.5-28,492.8) vs empty-second-launch
// 28,475.4 (28,461.7-28,480.0), i.e. +0.07% with fully overlapping
// ranges. A second launch per step is FREE on this workload — the
// enqueue hides behind the ~145 us physics kernel. So nothing below is
// launch overhead.
// PTXAS, both dtypes, and this is where the premise died:
//   step_kernel<float>  41,872 B / 0 / 0 / 255 regs -> BYTE-IDENTICAL
//     (only cmem[2] 340 -> 316 B, the obs normalisation constants
//     leaving). Removing a ~340-store tail with its own 71-word reload
//     moved the f32 register allocator by EXACTLY ZERO.
//   step_kernel<double> 82,912 B / 8 / 8 -> 82,944 B / 0 / 0 (255 regs):
//     the frame GREW 32 B and the 8-byte spill slot K8 introduced
//     finally cleared — and it bought nothing measurable (below).
//   obs_kernel<float>  312 B frame / 0 spills / 107 regs
//   obs_kernel<double> 616 B frame / 0 spills / 128 regs
// THE BITWISE HAZARD DID NOT FIRE, which is worth recording on its own:
// quat_to_rot acquires a new enclosing __global__ here, exactly the
// K10e2 setup that moved obs lanes 3/5/8 by 1 ulp last time — and nvcc
// contracted it identically. Max diff EXACTLY 0.0 on f32 AND f64 over
// 54 arrays, byte-equal: free-running, max_steps=60, want_contacts,
// 128-env random-gear traffic, plus explicit mode-1 (reset) and mode-2
// (set_state) obs captures with their state/FK tensors and the steps
// that follow each. The difference from K10e2 is that K10e2 moved ONE
// call to a DIFFERENT surrounding expression tree (fk_all's), whereas
// K19 moves the WHOLE function body intact; contraction is decided
// within the tree, so relocating the tree wholesale is safe.
// A/B, N=4096 f32, fresh process per arm, two independent batches:
//   batch 1  incumbent 28,476.5 (28,471.8-28,485.8)
//            split     28,397.5 (28,384.8-28,435.8)   -0.28%
//   batch 2  incumbent 28,489.2 (28,460.6-28,503.5)
//            split     28,394.4 (28,386.1-28,414.3)   -0.33%
//   10 of 10 paired rounds lost; both batches non-overlapping in the
//   WRONG direction. f64 (3 rounds): 16,866.4 -> 16,847.4, -0.11% with
//   OVERLAPPING ranges = noise, so clearing f64's spill slot was worth
//   nothing either.
// THE MECHANISM, and it is the transferable half. The obs writer moves
// ~5.6 MB per step at N=4096 (340 floats x 4,096 envs of stores, plus
// the 71-word reload and the note-block gather). Inside step_kernel
// those stores were issued by warps whose neighbours were still deep in
// the serial-latency-bound solve, so the traffic was HIDDEN. As its own
// kernel it has nothing to overlap with: the grid launches after the
// physics grid has drained and is pure exposed bandwidth. The split
// therefore converts free traffic into charged traffic and buys nothing
// back, because the register frame it was supposed to relieve does not
// move. Same family as A1/1f and K10e1 — a traffic/pressure argument
// with no dependency-chain shortening — now with a THIRD sign
// confirmation, and the general rule for this kernel hardens to:
//   Work that is memory-bound and currently INTERLEAVED with the serial
//   solve is running for free. Extracting it into its own launch prices
//   it, and only a real register-allocation win can pay that price —
//   check the ptxas frame BEFORE assuming the extraction relieves one.
// (The null-launch floor above does survive as a reusable fact: a
// second kernel node per step is free, so future items may split work
// out whenever the SPLIT-OFF WORK ITSELF is what shrinks.)
//
// K13 (2026-08-07) — FUSE THE PGS LIMIT-ROW AXPYS: BUILT, FULLY GATED
// AND REJECTED at -2.18% (4-wide) and -0.70% (2-wide). DO NOT RE-TRY IT
// IN THIS FORM. The arithmetic argument was right and the code is not
// what lost; what lost is a SIMT effect this file had not yet met, and
// it retires an idiom the round had been treating as free.
// WHAT WAS BUILT, exactly as the audit specified. A limit row's coupling
// into v is ONE component (J = sign * e_j) but its update is a full
// NV-wide axpy, so for a group of <=4 consecutive rows: phase 1 runs the
// scalar l_lam/newl/dd chain on 4 LOCAL copies of v[VOFF + j_t],
// cross-updating only those 4 indices as each dd becomes known; phase 2
// makes ONE sweep over i applying the group's columns in ascending row
// order. 280 -> 175 loads and 140 -> 35 stores per 4 rows per PGS
// iteration, and row t+1's read of v[VOFF + j_{t+1}] stops being a
// local store -> load round trip through a dynamic address. Active rows
// were packed with a `k == na` test inside an unrolled loop (so pc[]/
// sd[] stay STATICALLY indexed) and dispatched into compile-time-width
// 4/3/2/1 sweeps — the K14 trip-count rule honoured throughout — with a
// dd == 0 row GENUINELY SKIPPED (never entered into the pack; no *0.0,
// which is not the identity when v[i] is the opposite zero).
// EVERY GATE PASSED EXCEPT THE LAST, in both variants:
//   ptxas f32 41,872 B / 0 / 0 and 254 regs (frame byte-identical, one
//     register BELOW the incumbent's 255 — the cliff did not move);
//     f64 82,912 B / 8 / 8 / 255 byte-identical (4-wide), 82,896 B in
//     the 2-wide. So the spill warning the task was written around
//     never fired, in either dtype.
//   bitwise old-vs-new max diff EXACTLY 0.0, f32 AND f64, all arrays
//     byte-equal, over FIVE configs: free-running, max_steps=60, the
//     128-env random-gear traffic, and TWO NEW LIMIT-HEAVY configs
//     added because the random-gear micro under-hits this exact block
//     (measured nl per env-tick: random gear mean 7.54 / max 17;
//     "stopjam", every joint driven onto its upper stop, mean 20.12 /
//     median 21 / max 23; "altstop", half the joints jammed each way
//     and flipped every 40 ticks, mean 14.29). stopjam fires the 4-wide
//     tier ~5x per PGS iteration and covers every remainder tier. Both
//     were proven deterministic (two dumps of the same build, byte
//     equal) before being used as an A/B instrument.
//   pytest 172 green.
//   A/B, N=4096, fresh process per arm, 5 interleaved rounds each:
//     incumbent  28,490.3 (28,460.1 - 28,503.2)
//     4-wide     27,870.1 (27,859.5 - 27,902.9)   -2.18%   5/5 losses
//     incumbent  28,472.0 (28,453.3 - 28,495.1)
//     2-wide     28,272.5 (28,255.2 - 28,288.8)   -0.70%   5/5 losses
//     f64 (2 rounds, 4-wide) 16,908.2 -> 16,698.6 -1.24%
//   All ranges non-overlapping in the WRONG direction.
// THE MECHANISM — A WIDTH LADDER SELECTED BY A PER-LANE DATA-DEPENDENT
// COUNT COSTS THE UNION OF ITS WIDTHS, NOT THE ONE IT PICKS. This is
// one thread per env, so 32 envs share a warp and `na` (how many of the
// group's rows had dd != 0) is a per-LANE value. The warp must execute
// every arm any lane needs: with na = 1..4 all present across 32 envs —
// which is the norm — the 4-wide ladder runs sweep<4> AND <3> AND <2>
// AND <1>, i.e. 4+3+2+1 = 10 column-sweeps of 35 where the scalar form
// ran 4. The scalar form's `if (dd != 0)` guard has ONE arm, so
// divergence there costs its single 35-wide block no matter how the
// lanes disagree. Predicted excess per row: 6 sweeps / 4 rows = 1.5 for
// the 4-wide vs 1 sweep / 2 rows = 0.5 for the 2-wide, a ratio of 3.0;
// MEASURED ratio of the losses 2.18 / 0.70 = 3.1. That quantitative
// agreement is the first evidence, and the width ordering (deeper
// ladder = bigger loss) is the same monotone-form signature K14 found.
// THE CONFIRMING EXPERIMENT, and it is decisive: rerun the SAME 4-wide
// .pyd against WARP-UNIFORM traffic (chart_pool = 1 and one action row
// broadcast to all envs, verified all-lanes-identical q and v), so that
// na is warp-uniform and the ladder cannot diverge. 4 interleaved
// rounds, N=4096 f32: incumbent 100,020.7 (99,883.6 - 100,277.4) vs
// 4-wide 102,280.4 (102,239.5 - 102,461.2) = +2.26%, 4/4 paired WINS,
// non-overlapping. The transform's latency/traffic gain is REAL and
// roughly the size the audit predicted; on production traffic the
// divergence tax simply exceeds it. Same code, opposite sign, and the
// only thing changed was whether the lanes agree.
// THE RULE, which now qualifies the 4/2/1 ladder idiom K6/K7 made
// standard here: a compile-time-width ladder is free only when its
// SELECTOR IS WARP-UNIFORM (or when the widths dispatch work that is
// large compared with the union's excess). ensure_pcol's ladder is
// keyed on s.nl and runs a handful of times per substep against a 12x12
// solve; this one was keyed on a data-dependent per-lane count and ran
// 96 times per tick around 35 FMAs. Before packing a variable count
// into width tiers, ask what fraction of a warp shares the count.
// TRAFFIC WEIGHTING MAKES IT WORSE, NOT BETTER (R10's rule, R26's
// numbers): the random-gear micro under-represents limit work about 2x
// (nl 7.54 here vs 15.7 on bench traffic, since trained policies park
// joints on stops), so the micro is a LOWER BOUND on the magnitude of a
// limit-row effect — and this effect is negative, i.e. the bench verdict
// is worse than -2.18%. The K18(i)/K17-pja frequency rule with the sign
// reversed: a hot block amplifies a bad transform too.
// THE OTHER FACT THE UNIFORM PROBE TURNED UP, unrelated to K13 and worth
// its own line: the SAME kernel runs 100,021 env-steps/s on identical
// envs versus 28,490 on the production random-gear traffic — 3.5x. That
// is not a K13 result (it is the whole kernel, all blocks), but it puts
// a number on how much of this kernel's cost is warp divergence, and it
// is the largest single lever anyone has measured on this file. No
// production traffic is uniform, so it is not directly harvestable, but
// items that REDUCE divergence (rather than adding a ladder) now have a
// measured ceiling to aim at.
//
// K15 / K15B (2026-08-07) — folding the step-6 solve into the weld
// multi-solve as a 13th RHS: REVERTED at -0.06% f32 (+0.90% f64). The
// weld multi-solve no longer exists (weld mode deleted 2026-09-27); the
// transferable finding is that traffic cuts can pay on f64 while being
// flat on f32 — measure both dtypes before calling one dead.
//
// K20 + K16 (2026-08-07) — THE HOUSEKEEPING BUNDLE, and the round's
// biggest surprise-to-effort ratio. Two of the three K20 items landed
// for +2.08% together; the third is DEAD ON ITS PREMISE; K16 is dead on
// its measurement. Package, 5 interleaved A/B rounds N=4096 f32, fresh
// process per arm: 28,932.0 (28,923.7 - 28,945.3) -> 29,532.5
// (29,524.2 - 29,535.3), +2.08%, 5/5 paired wins, non-overlapping by
// ~25x the within-arm spread.
//
// K20-1 ensure_pcol COPY -> RE-READ ELIMINATION: +1.96% (weld-null
// projection, deleted with weld mode 2026-09-27). RULE it left: a
// removed load that was feeding on a value this thread just STORED is
// not a traffic cut, it is a chain cut — price it like A4/K6, not K19.
//
// K20-2 rnea_bias DEAD ZERO FILL: +0.13% f32 / +0.11% f64, KEPT at the
// edge of noise. `for (i < NV) d.c[i] = 0` ran immediately before a
// backward pass that ASSIGNS every element: the joint columns via
// d.c[VOFF + F_COORD[k]] (F_COORD is a bijection onto 0..NV-VOFF-1 —
// now static_asserted at the table by count + surjectivity, which pins
// injectivity by pigeonhole) and d.c[0..5] from f0/n0, with no read of
// d.c in between. 35 dead stores per rnea_bias call.
//   ptxas BYTE-IDENTICAL both dtypes; SASS f32 STL 2,768 -> 2,733
//   (exactly the 35), LDL UNCHANGED at 6,447 — which is the K10e1 gate
//   this item was written around: a removed store can lose by shifting a
//   rematerialization point, and here nothing was rematerialized.
//   bitwise max diff EXACTLY 0.0 f32 AND f64, all five configs, against
//   BOTH the K20-1 build and the pre-task incumbent; pytest 172 green.
//   A/B, two order-balanced batches of 5: 29,072.6 -> 29,109.6, +0.13%,
//   8/10 paired wins, ranges OVERLAPPING (one 29,316 boost-clock outlier
//   in the incumbent arm). f64 3/3, +0.11%.
//   KEPT rather than declared noise because the sign is consistent
//   across 13 paired rounds and both dtypes and the edit only REMOVES
//   verified-dead work — but it is honestly a sub-noise item; do not
//   re-derive a mechanism from +0.13%.
//
// K20-3 BROAD-PHASE PAIR RE-READS: NOT BUILT — THE PREMISE IS FALSE.
// The audit costed "440 loads for 48 distinct values" over the 55 pairs
// and proposed a collider-outer nest to halve them. The line-attributed
// SASS census (nvcc -lineinfo + nvdisasm -c -g) says the pid loop is
// NOT unrolled: it is ONE ROLLED BODY carrying 8 LDL total for the
// mid/hlr reads (2 per source line at the four broad-phase lines, f32
// and f64 alike). So 440 is the DYNAMIC count, and hoisting the ia side
// saves 4 loads on 43 of the 55 iterations = 172 local loads per
// substep. K15 removed 1,140 loads per substep — 6.6x more — from the
// same kernel and measured −0.06% on f32. The prize here is a sixth of
// a measured zero; there is no version of this that pays.
// The structural objection stands too, and it is why the item cannot be
// rescued: keeping the K14 compile-time trip counts through a
// collider-outer nest needs the pid loop template-specialized (55 copies
// of a narrow-phase body that today is one rolled block) in a block idle
// on ~84% of env-ticks — precisely the K17 point_jacobian_acc negative.
// The runtime-bounded alternative is the K14 negative. Do not re-open
// this without a NEW mechanism; the load count is settled.
//
// K16 STEP-11 CLAMP ensure_ecol BATCHING: DEAD — THE PREDICTABLE SET IS
// EMPTY, measured, not argued. A temporary counter build (three
// __device__ counters, atomicAdd at the step-11 clamp site, read through
// two throwaway pybind entry points — NOT committed) over 245,760
// substeps per config, 128 envs x 240 ticks:
//   traffic        calls/substep   MISSES/substep   miss rate
//   random-gear        10.34          0.0123          0.12%
//   stopjam            56.53          0.0021          0.00%
//   altstop            32.43          0.0021          0.01%
// i.e. the clamp passes call ensure_ecol 10-57 times per substep and
// essentially ALL of them are cache HITS: about one miss per 80
// substeps, and on the limit-heavy configs the misses are a fixed 512
// over the whole run (a reset transient), not a rate. There is no group
// of uncomputed columns at step 11 to batch, at any traffic. The
// comment at the step-8 batched unit-solve block claiming that block is
// "the only place in the kernel where a group of unit solves is known up
// front" is CONFIRMED, not corrected.
// (The mechanism, for the record: step 8 computes the whole s.l_j group
// plus the cap rows into the same per-substep s.ecomp cache, and by the
// time step 11 clamps a joint that joint's column is almost always
// already there. K16 is closed.)
//
// The register budget is NOT insensitive, correcting the earlier
// "-maxrregcount=64 vs the natural 128: within noise" note (measured
// 2026-08-03): left free, ptxas picks the target from a heuristic that
// is unstable under unrelated frame changes — the in-place Cholesky
// below flipped it 168 -> 64 regs with spills 8,220/6,820 ->
// 12,672/13,708 and cost 2.9% wall. step_kernel therefore PINS the
// budget via __launch_bounds__(THREADS, 1): minBlocksPerMultiprocessor
// = 1 is the honest statement of this kernel's design (occupancy is not
// the lever) and it lets ptxas take all 255 registers, which is where
// the spills are lowest — f32 8,220/6,820 -> 6,948/4,744 and f64
// 18,996/19,084 -> 16,160/13,676 vs the free-choice build, worth
// +0.5% on its own and the difference between a 2.9% loss and a wash
// for the in-place factor. Changing THREADS means re-measuring this.
//
// The identified next lever is candidate-B intra-env parallelism
// (warp-per-env with the factor in shared memory) — out of P5 scope.
constexpr int THREADS = 32;

// (The BEAT_MIN_BLOCKS occupancy-ladder pin and the BEAT_CHOL_RL
// right-looking Cholesky arm — the 2026-08-15 warp-rescue back-ports,
// last measured 2026-08-14 (git 91eac64,
// design/WARP_RESCUE_LEGACY_SIDE.md) — were
// removed 2026-09-22; the shipped kernel is the (THREADS, 1) pin and
// the left-looking factor, bit for bit as before.)

// ---- obs layout (SPEC-v10 §5; mirrors constants.py) -------------------------
constexpr int OFF_ROOT_POS = 0;
constexpr int OFF_ROOT_ROT = 3;
constexpr int OFF_ROOT_VLIN = 9;
constexpr int OFF_ROOT_VANG = 12;
constexpr int OFF_Q = 15;
constexpr int OFF_QD = 46;
// env v19: hand blocks grow 10 -> 11 lanes (+ remaining follow-through
// credit, mirroring lib.rs HAND_FEAT = 11) — everything after shifts +2.
constexpr int OFF_HAND_L = 75;
constexpr int OFF_HAND_R = 86;
constexpr int OFF_HEAD = 97;
constexpr int OFF_CONTACT = 100;
constexpr int OFF_NOTES = 102;
constexpr int OFF_NOTE_MASK = OFF_NOTES + N_NOTE_SLOTS * NOTE_FEAT;  // 534
// env v12 (unilateral-contact spec §7): per-foot ground-contact lanes
// [L_heel, L_toe, R_heel, R_toe] — tick-summed normal impulse /
// GROUND_IMP_NORM.
constexpr int OFF_GROUND_IMP = OFF_NOTE_MASK + N_NOTE_SLOTS;         // 558
constexpr int N_GROUND_IMP_LANES = 4;
// env v13 (contact spec §8): [l_elbow, r_elbow, l_knee, r_knee,
// l_foot, r_foot] world positions, head-style normalization.
constexpr int OFF_BODY_PTS = OFF_GROUND_IMP + N_GROUND_IMP_LANES;    // 562
constexpr int N_BODY_PTS = 6;
// env v24: world velocities of the §8 points (lib.rs mirror)
constexpr int OFF_BODY_PT_VEL = OFF_BODY_PTS + 3 * N_BODY_PTS;       // 580
// env v45 (2026-09-15, user order; replaces v44): per-hand PREVIOUS-
// NOTE blocks appended after the v24 block, indexed by NOTE COLOUR
// (lib.rs OFF_PREV_L comment = the lane contract: x_rel, y_rel,
// dir_cos, dir_sin, valid).
constexpr int PREV_FEAT = 5;
constexpr int OFF_PREV_L = OFF_BODY_PT_VEL + 3 * N_BODY_PTS;          // 598
constexpr int OFF_PREV_R = OFF_PREV_L + PREV_FEAT;                    // 603
// env v46 (user design 2026-09-16): per-env joint torque-limit scales,
// one per AXIS (16: spine 3, arm 7 shared L/R, leg 6 shared L/R), as
// (s - 1) / TAU_RAND_NORM lanes (lib.rs OFF_TAU mirror). Drawn at every
// reset when tau_rand > 0 from the wrapper-supplied tau_jit (the full draw)
// buffer (pose_jit precedent); otherwise whatever the wrapper wrote
// (set_tau_scale; default 1.0 => lanes 0, torque bit-identical).
constexpr int TAU_FEAT = 16;
constexpr double TAU_RAND_NORM = 0.25;
constexpr int OFF_TAU = OFF_PREV_R + PREV_FEAT;                       // 608
// env v49 (review item I07, 2026-09-30): per-hand blade-TIP world
// velocity [L xyz, R xyz] / V_NORM (lib.rs OFF_TIP_VEL mirror — the
// finite difference of wrist + (HANDLE_LEN + SABER_LEN)·blade, i.e. the
// judge's tip_move / DT). Zero at reset / mode-2 refresh and in Flux.
constexpr int TIP_VEL_FEAT = 6;
constexpr int OFF_TIP_VEL = OFF_TAU + TAU_FEAT;                       // 624
constexpr int OBS_DIM = OFF_TIP_VEL + TIP_VEL_FEAT;                   // 630
static_assert(OBS_DIM == 630, "spec §12 (env v49 blade-tip velocity lanes): OBS_DIM must be 630");
__device__ constexpr int TAU_AXIS[NQ] = {
    0, 1, 2,
    3, 4, 5, 6, 7, 8, 9,
    3, 4, 5, 6, 7, 8, 9,
    10, 11, 12, 13, 14, 15,
    10, 11, 12, 13, 14, 15};
constexpr double KNEE_Y0 = 0.50;  // hip 0.94 − thigh 0.44
constexpr double FOOT_Y0 = 0.07;  // ankle height at reference (spec §2)
// ≈ one tick of full-body weight (M_TOTAL·g·DT = 24.53 N·s), rounded to
// the spec literal lib.rs shares.
constexpr double GROUND_IMP_NORM = 24.5;

// ---- solver constants (step.rs / solver.rs / collide.rs) --------------------
constexpr double DT = 1.0 / 30.0;
constexpr double H_SUB = 1.0 / 240.0;
constexpr int N_SUB = 8;
// env v14 (2026-08-10): 8 -> 64, early-out keeps the typical substep
// unchanged (mirrors step.rs — see its step-11 comment for the
// phantom-force audit that killed the old guarantee pass).
constexpr int CLAMP_PASSES = 64;
// v14 joint-stop treatment (mirrors step.rs BETA_LIMIT / SLOP_LIMIT /
// LIMIT_TELEPORT_BACKSTOP): stops are reciprocal-impulse springs with
// a 2 mrad rest slop; the position clamp survives only as a deep-
// violation backstop.
constexpr double BETA_LIMIT = 0.2;
constexpr double SLOP_LIMIT = 0.002;
constexpr double LIMIT_TELEPORT_BACKSTOP = 0.35;
constexpr int PGS_ITERS = 12;
// Baumgarte gain of the joint-limit ROWS in the constraint solve
// (solver.rs; named BETA_WELD until the 2026-09-27 weld removal — same
// constant, the value the weld rows shared).
constexpr double BETA_LIMIT_ROW = 0.2;
constexpr double BETA_CONTACT = 0.2;
constexpr double COMPOSITE_EPS = 1e-4;
constexpr double ROOT_W_MAX = 96.0;
constexpr double ROOT_V_MAX = 48.0;
constexpr double MARGIN = 0.01;
constexpr double SLOP = 0.005;
constexpr double SEG_EPS = 1e-12;
constexpr double GRAV = 9.81;
constexpr float PD_FLOOR_F32 = 1e-8f;  // Divergences §20e (f32 only)
// f64 pivot floor (Divergences §22): algo.rs PD_FLOOR_F64 mirrored —
// same constant, same max clamp (pinned in tests/test_v10_pd_floor.py).
constexpr double PD_FLOOR_F64 = 1e-17;

// ---- model tables (model.rs, verbatim) --------------------------------------
constexpr double BASE_MASS = 8.40;
__device__ constexpr double BASE_INERTIA[3] = {0.038640, 0.101640, 0.101640};
__device__ constexpr double BASE_POS0[3] = {0.0, 0.94, -0.65};
// base_com is exactly zero — the base COM terms simplify per Rust's
// mat_vec with a zero vector (still computed generically below).
__device__ constexpr double BASE_COM[3] = {0.0, 0.0, 0.0};

__device__ constexpr int F_PARENT[NF] = {
    -1, 0, 1, 2, 2, 4, 5, 6, 7, 8, 9, 2, 11, 12, 13, 14, 15, 16,
    -1, 18, 19, 20, 21, 22, -1, 24, 25, 26, 27, 28};
__device__ constexpr int F_COORD[NF] = {
    0, 1, 2, -1, 4, 3, 5, 6, 7, 9, 8, 11, 10, 12, 13, 14, 16, 15,
    17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28};
// K20-2: F_COORD is a BIJECTION from the coord-bearing frames onto
// 0..NV-VOFF-1. Count + surjectivity together pin injectivity
// (pigeonhole), which is what licenses rnea_bias to drop its d.c zero
// fill: its backward pass ASSIGNS d.c[VOFF + F_COORD[k]] for every such
// frame, i.e. every one of d.c[VOFF..NV-1], and d.c[0..5] are assigned
// from f0/n0. See rnea_bias.
constexpr unsigned long long f_coord_mask() {
  unsigned long long m = 0;
  for (int k = 0; k < NF; k++)
    if (F_COORD[k] >= 0) m |= 1ull << F_COORD[k];
  return m;
}
constexpr int f_coord_count() {
  int n = 0;
  for (int k = 0; k < NF; k++)
    if (F_COORD[k] >= 0) n++;
  return n;
}
static_assert(f_coord_count() == NV - VOFF
                  && f_coord_mask() == ((1ull << (NV - VOFF)) - 1),
              "K20-2: F_COORD must hit every joint column 0..NV-VOFF-1 "
              "exactly once (rnea_bias's d.c fill is dead only if so)");
// K17 (2026-08-07) — PRECOMPUTED ANCESTOR TABLES, killing the three
// F_PARENT pointer-chases. Each site used to walk `k = F_PARENT[k]` and
// test `F_COORD[k] >= 0` per step; F_PARENT / F_COORD are __device__
// constexpr arrays, i.e. GLOBAL memory read through the const cache, so
// the SASS (verified before the edit, nvcc -lineinfo + nvdisasm -c -g)
// carries a loop-carried `LDG.E.CI -> LEA -> LDG.E.CI` pair per step:
// the next frame index cannot be addressed until the previous load
// returns, and the F_COORD load that produces the write column hangs
// off it. NOT ONE OF THE THREE was constant-folded by nvcc, not even
// the weld walk whose `frame` IS a compile-time 23/29 (the widx loop is
// unrolled into two copies, yet each keeps a rolled chase over k) — so
// there is no folding to lose. Per crba call the walks take 117 chase
// steps; the tables turn each into an INDEPENDENT indexed read off one
// base, which is the same instrument K11/K12 measured: live addresses
// and dependence length, not load count.
//
// ANC_F[f] lists, in exact walk order, the coord-bearing frames on f's
// INCLUSIVE ancestor chain (f itself first when F_COORD[f] >= 0);
// ANC_C[f] the matching v-columns VOFF + F_COORD[.]; ANC_N[f] the
// length, rows padded with -1. crba's walk is the STRICT-ancestor
// suffix, so it runs a = 1 .. ANC_N[k]-1 over the same row — legal only
// because its enclosing `if (coord >= 0)` guard excludes the one
// F_COORD < 0 frame (k = 3, whose row starts at 2, not 3). Both facts
// are pinned by anc_tables_ok() below.
constexpr int ANC_MAXD = 10;
__device__ constexpr int ANC_N[NF] = {1, 2, 3, 3, 4, 5, 6, 7, 8, 9, 10, 4,
                                      5, 6, 7, 8, 9, 10, 1, 2, 3, 4, 5, 6,
                                      1, 2, 3, 4, 5, 6};
// ANC_F / ANC_C are the readable form and exist only at compile time —
// the DEVICE table is ANC below, the same two rows restrided to 16.
constexpr int ANC_F[NF][ANC_MAXD] = {
    { 0, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    { 1,  0, -1, -1, -1, -1, -1, -1, -1, -1},
    { 2,  1,  0, -1, -1, -1, -1, -1, -1, -1},
    { 2,  1,  0, -1, -1, -1, -1, -1, -1, -1},
    { 4,  2,  1,  0, -1, -1, -1, -1, -1, -1},
    { 5,  4,  2,  1,  0, -1, -1, -1, -1, -1},
    { 6,  5,  4,  2,  1,  0, -1, -1, -1, -1},
    { 7,  6,  5,  4,  2,  1,  0, -1, -1, -1},
    { 8,  7,  6,  5,  4,  2,  1,  0, -1, -1},
    { 9,  8,  7,  6,  5,  4,  2,  1,  0, -1},
    {10,  9,  8,  7,  6,  5,  4,  2,  1,  0},
    {11,  2,  1,  0, -1, -1, -1, -1, -1, -1},
    {12, 11,  2,  1,  0, -1, -1, -1, -1, -1},
    {13, 12, 11,  2,  1,  0, -1, -1, -1, -1},
    {14, 13, 12, 11,  2,  1,  0, -1, -1, -1},
    {15, 14, 13, 12, 11,  2,  1,  0, -1, -1},
    {16, 15, 14, 13, 12, 11,  2,  1,  0, -1},
    {17, 16, 15, 14, 13, 12, 11,  2,  1,  0},
    {18, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {19, 18, -1, -1, -1, -1, -1, -1, -1, -1},
    {20, 19, 18, -1, -1, -1, -1, -1, -1, -1},
    {21, 20, 19, 18, -1, -1, -1, -1, -1, -1},
    {22, 21, 20, 19, 18, -1, -1, -1, -1, -1},
    {23, 22, 21, 20, 19, 18, -1, -1, -1, -1},
    {24, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {25, 24, -1, -1, -1, -1, -1, -1, -1, -1},
    {26, 25, 24, -1, -1, -1, -1, -1, -1, -1},
    {27, 26, 25, 24, -1, -1, -1, -1, -1, -1},
    {28, 27, 26, 25, 24, -1, -1, -1, -1, -1},
    {29, 28, 27, 26, 25, 24, -1, -1, -1, -1}};
constexpr int ANC_C[NF][ANC_MAXD] = {
    { 6, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    { 7,  6, -1, -1, -1, -1, -1, -1, -1, -1},
    { 8,  7,  6, -1, -1, -1, -1, -1, -1, -1},
    { 8,  7,  6, -1, -1, -1, -1, -1, -1, -1},
    {10,  8,  7,  6, -1, -1, -1, -1, -1, -1},
    { 9, 10,  8,  7,  6, -1, -1, -1, -1, -1},
    {11,  9, 10,  8,  7,  6, -1, -1, -1, -1},
    {12, 11,  9, 10,  8,  7,  6, -1, -1, -1},
    {13, 12, 11,  9, 10,  8,  7,  6, -1, -1},
    {15, 13, 12, 11,  9, 10,  8,  7,  6, -1},
    {14, 15, 13, 12, 11,  9, 10,  8,  7,  6},
    {17,  8,  7,  6, -1, -1, -1, -1, -1, -1},
    {16, 17,  8,  7,  6, -1, -1, -1, -1, -1},
    {18, 16, 17,  8,  7,  6, -1, -1, -1, -1},
    {19, 18, 16, 17,  8,  7,  6, -1, -1, -1},
    {20, 19, 18, 16, 17,  8,  7,  6, -1, -1},
    {22, 20, 19, 18, 16, 17,  8,  7,  6, -1},
    {21, 22, 20, 19, 18, 16, 17,  8,  7,  6},
    {23, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {24, 23, -1, -1, -1, -1, -1, -1, -1, -1},
    {25, 24, 23, -1, -1, -1, -1, -1, -1, -1},
    {26, 25, 24, 23, -1, -1, -1, -1, -1, -1},
    {27, 26, 25, 24, 23, -1, -1, -1, -1, -1},
    {28, 27, 26, 25, 24, 23, -1, -1, -1, -1},
    {29, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {30, 29, -1, -1, -1, -1, -1, -1, -1, -1},
    {31, 30, 29, -1, -1, -1, -1, -1, -1, -1},
    {32, 31, 30, 29, -1, -1, -1, -1, -1, -1},
    {33, 32, 31, 30, 29, -1, -1, -1, -1, -1},
    {34, 33, 32, 31, 30, 29, -1, -1, -1, -1}};
// Re-derivation of every entry straight off F_PARENT / F_COORD, in the
// wj_chain_mask idiom: the tables are a claim about the tree, and this
// is the tree walking itself.
constexpr int anc_len(int f) {
  int n = 0;
  for (int k = f; k >= 0; k = F_PARENT[k])
    if (F_COORD[k] >= 0) n++;
  return n;
}
constexpr int anc_at(int f, int i) {
  int n = 0;
  for (int k = f; k >= 0; k = F_PARENT[k])
    if (F_COORD[k] >= 0) {
      if (n == i) return k;
      n++;
    }
  return -1;
}
constexpr bool anc_tables_ok() {
  for (int f = 0; f < NF; f++) {
    if (ANC_N[f] != anc_len(f) || ANC_N[f] > ANC_MAXD) return false;
    for (int i = 0; i < ANC_MAXD; i++) {
      const int e = i < ANC_N[f] ? anc_at(f, i) : -1;
      if (ANC_F[f][i] != e) return false;
      if (ANC_C[f][i] != (e >= 0 ? VOFF + F_COORD[e] : -1)) return false;
    }
    // the property crba's a = 1.. suffix rests on: for a coord-bearing
    // frame the row opens with the frame itself and continues with the
    // walk from its parent, entry for entry.
    if (F_COORD[f] >= 0) {
      if (ANC_F[f][0] != f) return false;
      if (ANC_N[f] != anc_len(F_PARENT[f]) + 1) return false;
      for (int i = 1; i < ANC_N[f]; i++)
        if (ANC_F[f][i] != anc_at(F_PARENT[f], i - 1)) return false;
    }
  }
  return true;
}
static_assert(anc_tables_ok(),
              "K17: ANC_N/ANC_F/ANC_C must be the F_PARENT walk, "
              "coord-filtered, in visit order");
// The device-side table is the same two rows on a POWER-OF-TWO stride.
// That detail is addressing, and addressing is what this kernel pays in
// (K11/K12): at [NF][10] ptxas emitted, per unrolled group of the crba
// walk, a 64-bit XMAD chain for k * 40 against each of two bases (~12
// integer instructions); at [NF][16] the row offset is k << 6 and the
// two rows sit a compile-time distance apart off one base.
constexpr int ANC_STRIDE = 16;
struct AncRows {
  int f[NF][ANC_STRIDE];
  int c[NF][ANC_STRIDE];
};
constexpr AncRows anc_rows() {
  AncRows t{};
  for (int i = 0; i < NF; i++)
    for (int j = 0; j < ANC_STRIDE; j++) {
      t.f[i][j] = j < ANC_N[i] ? ANC_F[i][j] : -1;
      t.c[i][j] = j < ANC_N[i] ? ANC_C[i][j] : -1;
    }
  return t;
}
__device__ constexpr AncRows ANC = anc_rows();
__device__ constexpr double F_ANCHOR[NF][3] = {
    {0.0, 0.10, 0.0}, {0, 0, 0}, {0, 0, 0}, {0.0, 0.66, 0.0},
    {-0.20, 0.43, 0.0}, {0, 0, 0}, {0, 0, 0}, {0.0, 0.0, 0.35},
    {0.0, 0.0, 0.30}, {0, 0, 0}, {0, 0, 0},
    {0.20, 0.43, 0.0}, {0, 0, 0}, {0, 0, 0}, {0.0, 0.0, 0.35},
    {0.0, 0.0, 0.30}, {0, 0, 0}, {0, 0, 0},
    {-0.10, 0.0, 0.0}, {0, 0, 0}, {0, 0, 0}, {0.0, -0.44, 0.0},
    {0.0, -0.43, 0.0}, {0, 0, 0},
    {0.10, 0.0, 0.0}, {0, 0, 0}, {0, 0, 0}, {0.0, -0.44, 0.0},
    {0.0, -0.43, 0.0}, {0, 0, 0}};
__device__ constexpr double F_AXIS[NF][3] = {
    {1, 0, 0}, {0, 1, 0}, {0, 0, 1}, {0, 0, 1},
    {0, 1, 0}, {-1, 0, 0}, {0, 0, 1}, {-1, 0, 0},
    {0, 0, 1}, {0, 1, 0}, {-1, 0, 0},
    {0, 1, 0}, {-1, 0, 0}, {0, 0, 1}, {-1, 0, 0},
    {0, 0, 1}, {0, 1, 0}, {-1, 0, 0},
    {-1, 0, 0}, {0, 1, 0}, {0, 0, 1}, {1, 0, 0}, {-1, 0, 0}, {0, 0, 1},
    {-1, 0, 0}, {0, 1, 0}, {0, 0, 1}, {1, 0, 0}, {-1, 0, 0}, {0, 0, 1}};
// hand 0.61 -> 0.35 (user 2026-08-18): fist only, no controller mass
// — real players exploit controller motion prediction and never truly
// swing the composite mass; inertia rescaled with it (model.rs mirror).
// 0.35 -> 0.50 (main's v63x2 body since 2026-10-03; 0.35 on branch
// body/original), inertia rescaled again (0.000460 x 0.50/0.61).
__device__ constexpr double F_MASS[NF] = {
    0, 0, 24.20, 5.20, 0, 0, 2.00, 1.20, 0, 0, 0.50,
    0, 0, 2.00, 1.20, 0, 0, 0.50,
    0, 0, 10.60, 3.20, 0, 1.00,
    0, 0, 10.60, 3.20, 0, 1.00};
__device__ constexpr double F_COM[NF][3] = {
    {0, 0, 0}, {0, 0, 0}, {0.0, 0.215, 0.0}, {0, 0, 0},
    {0, 0, 0}, {0, 0, 0}, {0.0, 0.0, 0.175}, {0.0, 0.0, 0.15},
    {0, 0, 0}, {0, 0, 0}, {0.0, 0.0, 0.06},
    {0, 0, 0}, {0, 0, 0}, {0.0, 0.0, 0.175}, {0.0, 0.0, 0.15},
    {0, 0, 0}, {0, 0, 0}, {0.0, 0.0, 0.06},
    {0, 0, 0}, {0, 0, 0}, {0.0, -0.22, 0.0}, {0.0, -0.215, 0.0},
    {0, 0, 0}, {0.0, -0.035, 0.05},
    {0, 0, 0}, {0, 0, 0}, {0.0, -0.22, 0.0}, {0.0, -0.215, 0.0},
    {0, 0, 0}, {0.0, -0.035, 0.05}};
__device__ constexpr double F_INERTIA[NF][3] = {
    {0, 0, 0}, {0, 0, 0}, {0.957507, 0.254964, 0.957507},
    {0.016848, 0.016848, 0.016848},
    {0, 0, 0}, {0, 0, 0}, {0.029222, 0.029222, 0.001966},
    {0.013056, 0.013056, 0.000931},
    {0, 0, 0}, {0, 0, 0}, {0.000377, 0.000377, 0.000377},
    {0, 0, 0}, {0, 0, 0}, {0.029222, 0.029222, 0.001966},
    {0.013056, 0.013056, 0.000931},
    {0, 0, 0}, {0, 0, 0}, {0.000377, 0.000377, 0.000377},
    {0, 0, 0}, {0, 0, 0}, {0.250198, 0.018493, 0.250198},
    {0.065964, 0.003161, 0.065964}, {0, 0, 0},
    {0.005617, 0.005883, 0.001083},
    {0, 0, 0}, {0, 0, 0}, {0.250198, 0.018493, 0.250198},
    {0.065964, 0.003161, 0.065964}, {0, 0, 0},
    {0.005617, 0.005883, 0.001083}};

// frame indices (model.rs)
constexpr int F_HEAD = 3;
constexpr int F_L_SHOULDER = 4, F_L_WRIST = 8, F_L_HAND = 10;
constexpr int F_R_SHOULDER = 11, F_R_WRIST = 15, F_R_HAND = 17;
constexpr int F_L_FOOT = 23, F_R_FOOT = 29;
// env v13 (contact spec §8): obs body points
constexpr int F_L_ELBOW = 7, F_R_ELBOW = 14;
constexpr int F_L_SHANK = 21, F_R_SHANK = 27;
// The six §8 body points' frames, in obs order. An accessor over a
// function-local table rather than a file-scope __device__ constexpr
// array: callers index it inside fully unrolled loops, and a local
// table folds to immediates exactly like the four per-site copies it
// replaces, whereas a __device__ constexpr array is GLOBAL memory that
// nvcc does not reliably constant-fold (see the K17 note at F_COORD).
__device__ __forceinline__ int body_pt_frame(int i) {
  const int f[N_BODY_PTS] = {F_L_ELBOW, F_R_ELBOW, F_L_SHANK,
                             F_R_SHANK, F_L_FOOT, F_R_FOOT};
  return f[i];
}

__device__ constexpr double LIMIT_LO[NQ] = {
    -0.3490658503988659, -0.6981317007977318, -0.5235987755982988,
    -1.5707963267948966, -1.7453292519943295, -1.5707963267948966,
    0.0, 0.0, -1.6580627893946132, -0.5235987755982988,
    -1.5707963267948966, -0.7853981633974483, -1.2217304763960306,
    0.0, -3.141592653589793, -1.6580627893946132, -0.2617993877991494,
    -0.3490658503988659, -0.7853981633974483, -0.7853981633974483,
    0.0, -0.8726646259971648, -0.3490658503988659,
    -0.3490658503988659, -0.6981317007977318, -0.5235987755982988,
    0.0, -0.8726646259971648, -0.3490658503988659};
__device__ constexpr double LIMIT_HI[NQ] = {
    0.7853981633974483, 0.6981317007977318, 0.5235987755982988,
    1.1344640137963142, 0.7853981633974483, 1.2217304763960306,
    2.6179938779914944, 3.141592653589793, 1.6580627893946132,
    0.2617993877991494,
    1.1344640137963142, 1.7453292519943295, 1.5707963267948966,
    2.6179938779914944, 0.0, 1.6580627893946132, 0.5235987755982988,
    2.0943951023931953, 0.6981317007977318, 0.5235987755982988,
    2.6179938779914944, 0.6108652381980153, 0.3490658503988659,
    2.0943951023931953, 0.7853981633974483, 0.7853981633974483,
    2.6179938779914944, 0.6108652381980153, 0.3490658503988659};
// wrist bend 1.06 -> 1.431 (+35%; main's v63x2 body, model.rs mirror)
__device__ constexpr double TAU_MAX[NQ] = {
    600.00, 239.95, 375.00,
    85.38, 85.38, 19.45, 19.13, 1.06, 1.431, 1.06,
    85.38, 85.38, 19.45, 19.13, 1.06, 1.431, 1.06,
    262.50, 75.00, 135.00, 150.00, 135.00, 37.50,
    262.50, 75.00, 135.00, 150.00, 135.00, 37.50};
// wrist bend 50 -> 65 -> 90 rad/s (main's v63x2 body since 2026-10-03)
__device__ constexpr double OMEGA_CAP[NQ] = {
    16.0, 16.0, 16.0,
    16.0, 16.0, 16.0, 16.0, 50.0, 90.0, 50.0,
    16.0, 16.0, 16.0, 16.0, 50.0, 90.0, 50.0,
    16.0, 16.0, 16.0, 16.0, 16.0, 16.0,
    16.0, 16.0, 16.0, 16.0, 16.0, 16.0};
__device__ constexpr double DAMPING[NQ] = {
    4.3140, 1.1998, 3.6471,
    0.42689, 0.42689, 0.07292, 0.07174, 0.00159, 0.00159, 0.00159,
    0.42689, 0.42689, 0.07292, 0.07174, 0.00159, 0.00159, 0.00159,
    1.8175, 0.28120, 1.8133, 0.26294, 0.00561, 0.00138,
    1.8175, 0.28120, 1.8133, 0.26294, 0.00561, 0.00138};
__device__ constexpr double ARMATURE[NQ] = {
    0.071900, 0.019996, 0.060786,
    0.0071149, 0.0071149, 0.0012154, 0.0011957, 0.000026556, 0.000026556,
    0.000026556,
    0.0071149, 0.0071149, 0.0012154, 0.0011957, 0.000026556, 0.000026556,
    0.000026556,
    0.030292, 0.0046866, 0.030222, 0.0043823, 0.000093419, 0.000023083,
    0.030292, 0.0046866, 0.030222, 0.0043823, 0.000093419, 0.000023083};

// ---- collision spec (collide.rs, verbatim) ----------------------------------
__device__ constexpr int COL_FRAME[NUM_COLLIDERS] = {
    2, 3, 6, 13, 7, 14, 10, 17, 20, 26, 21, 27};
__device__ constexpr double COL_A[NUM_COLLIDERS][3] = {
    {0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0},
    {0.0, 0.0, 0.06}, {0.0, 0.0, 0.06}, {0, 0, 0}, {0, 0, 0},
    {0, 0, 0}, {0, 0, 0}};
__device__ constexpr double COL_B[NUM_COLLIDERS][3] = {
    {0.0, 0.43, 0.0}, {0, 0, 0}, {0.0, 0.0, 0.35}, {0.0, 0.0, 0.35},
    {0.0, 0.0, 0.30}, {0.0, 0.0, 0.30}, {0.0, 0.0, 0.06}, {0.0, 0.0, 0.06},
    {0.0, -0.44, 0.0}, {0.0, -0.44, 0.0}, {0.0, -0.43, 0.0},
    {0.0, -0.43, 0.0}};
__device__ constexpr double COL_R[NUM_COLLIDERS] = {
    0.15, 0.09, 0.045, 0.045, 0.04, 0.04, 0.05, 0.05,
    0.06, 0.06, 0.045, 0.045};
__device__ constexpr int PAIR_A[NUM_PAIRS] = {
    0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3,
    4, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 6, 6, 6, 6, 6,
    7, 7, 7, 7, 8, 8, 9, 10};
__device__ constexpr int PAIR_B[NUM_PAIRS] = {
    4, 5, 6, 7, 10, 11, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11,
    3, 5, 6, 7, 8, 9, 10, 11, 4, 6, 7, 8, 9, 10, 11,
    5, 7, 8, 9, 10, 11, 6, 8, 9, 10, 11, 7, 8, 9, 10, 11,
    8, 9, 10, 11, 9, 11, 10, 11};
// obs contact bits: collider ids 2/4/6 -> left arm, 3/5/7 -> right
__device__ __forceinline__ bool is_arm_l(int id) {
  return id == 2 || id == 4 || id == 6;
}
__device__ __forceinline__ bool is_arm_r(int id) {
  return id == 3 || id == 5 || id == 7;
}

// FOOT SPATIAL-ROW SUPPORT (the former K5/K8 weld-row derivation, now
// serving the ground block — GroundRows / foot_row_dot). A foot's six
// spatial rows (3 linear at the foot origin, 3 world angular) are built
// by one ancestor walk touching row[VOFF + F_COORD[k]] for k on the
// foot's chain plus root columns: walking F_PARENT from F_L_FOOT = 23
// gives v-columns 23..28, from F_R_FOOT = 29 v-columns 29..34.
//   ANGULAR row t: the chain plus ONE root column 3 + t holding exactly
//   +0.0 + 1.0 = 1.0.
//   LINEAR row t (n = e_t): the chain, row[t] = 1.0, and the two root
//   columns {3,4,5} \ {3+t} (cross(rel0, e_t)); column 3+t is a
//   structural +-0.0 absorbed by the zero fill's +0.0 (Divergences 20c).
// Every consumer keeps each accumulator's ASCENDING-i term order; the
// dropped products are (+-0.0)*x into +0.0-seeded accumulators, so the
// -0.0 channel is unreachable, and the exactly-1.0 entries are
// contracted as a bare add (1.0*x == x unconditionally).
constexpr int WJ_L0 = 23;
constexpr int WJ_R0 = 29;
static_assert(WJ_R0 == WJ_L0 + 6 && WJ_R0 + 6 == NV,
              "foot support layout is {0..5} u {L0..L0+5} / {R0..NV}");
// a LINEAR row t's two other root columns, {3,4,5} \ {3+t}, ASCENDING
__device__ constexpr int WJ_R2[3][2] = {{4, 5}, {3, 5}, {3, 4}};
static_assert(WJ_R2[0][0] == 4 && WJ_R2[0][1] == 5 && WJ_R2[1][0] == 3
                  && WJ_R2[1][1] == 5 && WJ_R2[2][0] == 3
                  && WJ_R2[2][1] == 4,
              "linear-row root extras are {3,4,5}\\{3+t}, ascending");
// re-derivation of the ancestor chains straight off F_PARENT/F_COORD:
// the bit mask of v-columns the walk touches must be exactly the
// 6-column segment [WJ_L0 / WJ_R0, +6). Equality pins count, contiguity,
// distinctness AND disjointness from the root columns 0..5 at once.
constexpr unsigned long long wj_chain_mask(int f) {
  unsigned long long m = 0;
  for (int k = f; k >= 0; k = F_PARENT[k])
    if (F_COORD[k] >= 0) m |= 1ull << (VOFF + F_COORD[k]);
  return m;
}
static_assert(wj_chain_mask(F_L_FOOT) == (0x3full << WJ_L0)
                  && wj_chain_mask(F_R_FOOT) == (0x3full << WJ_R0),
              "foot ancestor chains are exactly v-columns 23..28 "
              "and 29..34 -- 6 distinct columns each, none in 0..5");

// ---- audit #10 (2026-09-26): TREE ELIMINATION LAYOUT of the mass matrix ----
// algo.rs `TreeChol::tree`, mirrored as compile-time tables (read the
// algo.rs header for the whole argument). The dense factor eliminated the
// root-first coordinates in index order, so every L row filled; eliminating
// LEAVES FIRST, the floating base LAST, gives a fill-free factor with
// exactly the tree's ancestor sparsity. The order ("e-order") is the
// REVERSED DFS pre-order of the coordinate tree — the frames are already
// in DFS pre-order, so the joints are simply the coord frames in reversed
// frame order (e 0..28: r leg, l leg, r arm, l arm, spine) and the base
// chain v5 -> ... -> v0 reversed puts v0..v5 at e 29..34. Reversed
// pre-order is a POST-order: desc(e) is the contiguous range [lo[e], e),
// so row e of L is non-zero exactly on [lo[e], e] and column e exactly on
// {e} u anc(e), anc ascending (the chain to the root, <= 15 entries).
//
// STORAGE (d.m, TC_NNZ = 341 words, was NV*NV = 1,225): COLUMN-MAJOR
// PROFILE — column e at cb[e] .. cb[e] + d[e]: slot 0 the diagonal, slot t
// the entry of its t-th ancestor. crba writes M straight into these
// slots (each joint's CRBA column IS its chain: diagonal, the coord-bearing
// ancestors nearest first, then the six base rows), the factor overwrites
// them in place, and every solve reads them. Two addressing facts carry
// all the loops:
//   * column k's ancestors are [..., e, anc(e)] for any e in anc(k), so
//     L(e, k) sits at slot d[k] - d[e] of column k: kb[k] - d[e] with
//     kb[k] = cb[k] + d[k] (forward rows, factor inner products);
//   * vslot[s] / eslot[s] name the v / e index of the row slot s holds
//     (back pass, unit chains).
// Right-hand sides and solutions stay in v-order; the loops translate
// through e2v (warp-uniform tables — the loop indices never depend on
// data except in the per-column unit chains, which read the __device__
// copy).
// MEASURED (RTX 5090, N = 32768, 2 interleaved rounds vs 88573c0): flux
// 23.40 -> 19.30 ms/tick (-17.5 %), standard 142.1 -> 137.1 (-3.5 %);
// float contact kernel stack 37,664 -> 34,128 B, spills 48/48 unchanged.
// Of that, interleaving the limit group's four unit chains
// (tree_fwd_unit_b) is -1.1 % on flux by itself.
constexpr int TC_NNZ = 341;
constexpr int TC_BASE_E = NQ;  // e of base coordinate v0 (v_t at NQ + t)
struct TreeTabs {
  int e2v[NV], v2e[NV], par[NV], lo[NV], d[NV], cb[NV], kb[NV];
  int fcb[NF];  // column base of coord frame f (-1: welded frame)
  // per profile slot: the v / e index of the row it holds, and the
  // column base of that row's own column (the unit chains' L(a, k))
  int vslot[TC_NNZ], eslot[TC_NNZ], cbslot[TC_NNZ];
};
constexpr int tc_up(int f) {  // nearest coord-bearing strict ancestor
  for (int k = F_PARENT[f]; k >= 0; k = F_PARENT[k])
    if (F_COORD[k] >= 0) return k;
  return -1;
}
constexpr TreeTabs tree_tabs() {
  TreeTabs t{};
  int ef[NF] = {};
  int r = 0;
  for (int f = 0; f < NF; f++) {
    ef[f] = -1;
    if (F_COORD[f] >= 0) {
      ef[f] = NQ - 1 - r;  // reversed pre-order rank
      t.e2v[ef[f]] = VOFF + F_COORD[f];
      r++;
    }
  }
  for (int f = 0; f < NF; f++)
    if (F_COORD[f] >= 0) {
      const int u = tc_up(f);
      t.par[ef[f]] = u < 0 ? TC_BASE_E : ef[u];
    }
  for (int b = 0; b < VOFF; b++) {
    t.e2v[TC_BASE_E + b] = b;
    t.par[TC_BASE_E + b] = b + 1 < VOFF ? TC_BASE_E + b + 1 : -1;
  }
  for (int e = 0; e < NV; e++) t.v2e[t.e2v[e]] = e;
  int c = 0;
  for (int e = 0; e < NV; e++) {
    int n = 0;
    for (int a = t.par[e]; a >= 0; a = t.par[a]) n++;
    t.d[e] = n;
    int nd = 0;
    for (int k = 0; k < e; k++)
      for (int a = t.par[k]; a >= 0; a = t.par[a])
        if (a == e) nd++;
    t.lo[e] = e - nd;
    t.cb[e] = c;
    t.kb[e] = c + n;
    t.vslot[c] = t.e2v[e];
    t.eslot[c] = e;
    int s = 1;
    for (int a = t.par[e]; a >= 0; a = t.par[a], s++) {
      t.vslot[c + s] = t.e2v[a];
      t.eslot[c + s] = a;
    }
    c += n + 1;
  }
  for (int f = 0; f < NF; f++)
    t.fcb[f] = F_COORD[f] >= 0 ? t.cb[t.v2e[VOFF + F_COORD[f]]] : -1;
  for (int q = 0; q < TC_NNZ; q++) t.cbslot[q] = t.cb[t.eslot[q]];
  return t;
}
__device__ constexpr TreeTabs TREE = tree_tabs();
// the warp-uniform loops read this copy (constant bank: one broadcast
// per warp-uniform index); the per-column unit chains read TREE
__constant__ TreeTabs TREE_C = tree_tabs();
constexpr bool tree_tabs_ok() {
  const TreeTabs t = tree_tabs();
  int nnz = 0;
  for (int e = 0; e < NV; e++) {
    if (t.v2e[t.e2v[e]] != e) return false;
    if (t.par[e] >= 0 && t.par[e] <= e) return false;  // children first
    // post-order: every k in [lo, e) has e on its ancestor chain
    for (int k = t.lo[e]; k < e; k++) {
      bool anc = false;
      for (int a = t.par[k]; a >= 0; a = t.par[a]) anc = anc || a == e;
      if (!anc) return false;
    }
    nnz += t.d[e] + 1;
  }
  if (nnz != TC_NNZ) return false;
  // crba's slot map: a coord frame's column is [itself, its ANC.f row
  // a = 1.. (coord-bearing ancestors, nearest first), base v0..v5]
  for (int f = 0; f < NF; f++) {
    if (F_COORD[f] < 0) continue;
    const int cb = t.fcb[f];
    if (t.vslot[cb] != VOFF + F_COORD[f]) return false;
    for (int a = 1; a < ANC_N[f]; a++)
      if (t.vslot[cb + a] != ANC_C[f][a]) return false;
    for (int b = 0; b < VOFF; b++)
      if (t.vslot[cb + ANC_N[f] + b] != b) return false;
    if (t.d[t.v2e[VOFF + F_COORD[f]]] != ANC_N[f] - 1 + VOFF) return false;
  }
  // the feet's forward closure (tree_fwd_feet_tn): r leg = e 0..5 with
  // the right foot at e 0, l leg = e 6..11 with the left foot at e 6,
  // both legs' profiles closed (lo = 0 / 6), the base at e 29..34
  if (t.v2e[VOFF + F_COORD[F_R_FOOT]] != 0 || t.v2e[VOFF + F_COORD[F_L_FOOT]] != 6)
    return false;
  for (int e = 0; e < 12; e++)
    if (t.lo[e] != (e < 6 ? 0 : 6) || t.v2e[t.e2v[e]] != e) return false;
  for (int e = 0; e < 6; e++)
    if (t.e2v[e] < WJ_R0 || t.e2v[6 + e] < WJ_L0 || t.e2v[6 + e] >= WJ_R0)
      return false;
  for (int b = 0; b < VOFF; b++)
    if (t.lo[TC_BASE_E + b] != 0) return false;
  return true;
}
static_assert(tree_tabs_ok(),
              "audit #10: the tree layout must be a post-order with the "
              "algo.rs TreeChol::tree permutation, 341 profile slots, and "
              "crba's column-slot map");
// crba writes every profile slot exactly once (no zero fill): a joint
// column's slots are written iff the frame's composite mass is > 0, which
// holds whenever the frame's subtree carries mass
constexpr bool tc_all_subtrees_massive() {
  for (int f = 0; f < NF; f++) {
    if (F_COORD[f] < 0) continue;
    double m = 0;
    for (int k = f; k < NF; k++) {
      bool in = k == f;
      for (int a = F_PARENT[k]; a >= 0 && !in; a = F_PARENT[a]) in = a == f;
      if (in) m += F_MASS[k];
    }
    if (!(m > 0)) return false;
  }
  return true;
}
static_assert(tc_all_subtrees_massive(),
              "audit #10: crba's slot writes cover the whole profile only "
              "if every coord frame's composite mass is > 0");

// ---- env v11 GROUND CONTACT (unilateral-contact spec §2/§3) -----------------
// The 8 sole-box corners against the static plane y = 0 — one
// unilateral normal row plus two BOX-clamped friction rows per ACTIVE
// corner, assembled BEFORE the self-collision rows (so the §2 pair-id
// row order downstream is untouched). MIRROR sources: collide.rs
// `ground_phase`, solver.rs `assemble_ground_rows` / `prep_ground_rows`
// / `pgs_ground_sweep_foot`.
constexpr int CORNERS_PER_FOOT = 4;
constexpr int NUM_GROUND_CORNERS = 2 * CORNERS_PER_FOOT;  // foot_l, foot_r
constexpr int GROUND_ROWS_PER_CORNER = 3;                 // N, T1, T2
constexpr int N_GROUND_ROWS = NUM_GROUND_CORNERS * GROUND_ROWS_PER_CORNER;
static_assert(NUM_GROUND_CORNERS == 8 && N_GROUND_ROWS == 24,
              "contact spec §2/§3: 8 sole corners, 3 rows each");
// Coulomb friction coefficient of the ground (contact spec §3).
constexpr double GROUND_MU = 0.9;  // lib.rs MU mirror — DO NOT raise
// past 0.9: the box-clamped PGS loses stick above it (see solver.rs MU)
// Corner-carrying frames in the NORMATIVE order: foot_l then foot_r.
__device__ constexpr int GROUND_FRAMES[2] = {F_L_FOOT, F_R_FOOT};
// Sole-box corners in the FOOT frame, NORMATIVE order within a foot:
// (x-,z-), (x-,z+), (x+,z-), (x+,z+) — collide.rs GROUND_CORNERS,
// derived from the SPEC-v10 §1.2 foot box (sole plane y = -0.070).
__device__ constexpr double GROUND_CORNERS[CORNERS_PER_FOOT][3] = {
    {-0.045, -0.070, -0.075}, {-0.045, -0.070, 0.175},
    {0.045, -0.070, -0.075}, {0.045, -0.070, 0.175}};
// A corner's three row directions in world, NORMATIVE order: normal y,
// then friction x and z (solver.rs GROUND_DIRS). The ground is static,
// so a row is the corner's OWN point Jacobian projected on the direction
// — no second chain to subtract (contrast the §2 self-collision rows).
__device__ constexpr double GROUND_DIRS[GROUND_ROWS_PER_CORNER][3] = {
    {0.0, 1.0, 0.0}, {1.0, 0.0, 0.0}, {0.0, 0.0, 1.0}};
// G1: the world axis of each row direction (GROUND_DIRS[t] = e_axis), so
// row t is a foot LINEAR-shaped row whose 1.0 entry sits at column axis
// (see GroundRows)
__device__ constexpr int GROUND_AXIS[GROUND_ROWS_PER_CORNER] = {1, 0, 2};
static_assert(GROUND_DIRS[0][GROUND_AXIS[0]] == 1.0
                  && GROUND_DIRS[1][GROUND_AXIS[1]] == 1.0
                  && GROUND_DIRS[2][GROUND_AXIS[2]] == 1.0
                  && GROUND_DIRS[0][0] == 0.0 && GROUND_DIRS[0][2] == 0.0
                  && GROUND_DIRS[1][1] == 0.0 && GROUND_DIRS[1][2] == 0.0
                  && GROUND_DIRS[2][0] == 0.0 && GROUND_DIRS[2][1] == 0.0,
              "G1: ground row t is the unit direction e_{GROUND_AXIS[t]}");
// audit #9 (2026-09-26) — GROUND ROWS FROM EACH FOOT'S 6-D SPATIAL
// JACOBIAN (solver.rs GROUND_SPATIAL, the reference; SPEC-v10
// Divergences §24). A corner row is e_a . (v_o + w x r) = e_a . v_o +
// (r x e_a) . w with r = p_corner - p_foot, i.e. the combination
// c = [e_a; r x e_a] of the foot's six spatial rows J_f (3 linear rows at
// the foot origin + 3 world angular rows). Per
// row t: its linear axis a and the two angular slots b1 < b2 where
// r x e_a is non-zero:
//   r x e_1 = (-r2, 0, r0)   r x e_0 = (0, r2, -r1)   r x e_2 = (r1, -r0, 0)
// The kernel solves M^-1 J_f^T (<= 12 columns) instead of M^-1 J^T for
// 24 corner rows and runs the per-row PGS on 6-D foot quantities —
// same rows, order, clamps and iteration count; only rounding moves.
constexpr int FOOT_ROWS = 6;
__device__ constexpr int GROUND_SPATIAL[GROUND_ROWS_PER_CORNER][3] = {
    {1, 0, 2}, {0, 1, 2}, {2, 0, 1}};
static_assert(GROUND_SPATIAL[0][0] == GROUND_AXIS[0]
                  && GROUND_SPATIAL[1][0] == GROUND_AXIS[1]
                  && GROUND_SPATIAL[2][0] == GROUND_AXIS[2],
              "audit #9: GROUND_SPATIAL[t][0] is row t's world axis");

// ---- env constants (lib.rs, verbatim) ---------------------------------------
constexpr double V_NORM = 6.0;
constexpr double V_MIN_CUT = 1.0;
constexpr double ACC_NORM = 0.3;
constexpr double COS_TOL = 0.5;
constexpr double Y_CENTER = 1.40;
constexpr double REWARD_BAD = -0.5;
constexpr double REWARD_MISS = -0.5;
constexpr double REWARD_BOMB = -1.5;
// env v12 (contact spec §7): per-tick survival bonus
// — see lib.rs ALIVE_BONUS for the sizing rationale. Overridden by the
// fall/escape terminal pricing.
constexpr double ALIVE_BONUS = 0.002;
constexpr double GOOD_HALF_X = 0.4;
constexpr double GOOD_HALF_Z = 0.5;
constexpr double GOOD_Z_SHIFT = -0.25;
constexpr double BAD_HALF = 0.175;
constexpr double BOMB_RADIUS = 0.18;
constexpr double SABER_LEN = 1.0;
// env v42 hilt grip: blade spans [HANDLE_LEN, HANDLE_LEN + SABER_LEN] from the wrist
constexpr double HANDLE_LEN = 0.16;
constexpr int N_BLADE_SAMPLES = 8;
constexpr double REACH_PAST_SHOULDER = 0.65 + HANDLE_LEN + 1.0 + 0.75;
constexpr double PRE_FULL = 1.7453292519943295;
constexpr double POST_FULL = 1.0471975511965976;
constexpr double PTS_PRE = 70.0;
constexpr double PTS_POST = 30.0;
constexpr double PTS_ACC = 15.0;
constexpr double SCORE_MAX = 115.0;
constexpr double CHAIN_LINK_PTS = 20.0;
constexpr double SWING_BREAK = 0.005;
// Flux game mode (lib.rs flux.rs verbatim; SPEC.md §Flux): no sabers —
// each hand is its palm centre PALM_OFF along the hand axis; circles
// (colour, disc radius CIRCLE_R) score on the matching palm's touch,
// bars (colourless, 2*BAR_HALF_W wide) on the swept Flux segment
// between the palms. Hitbox depth slab: 5 cm either side of the flat
// visual, swept in the object's frame (flux.rs judge).
constexpr double PALM_OFF = 0.06;
constexpr double CIRCLE_R = 0.175;
constexpr double CIRCLE_PTS = 50.0;
constexpr double CIRCLE_ACC_PTS = 10.0;
constexpr double BAR_PTS = 100.0;
constexpr double BAR_ACC_PTS = 15.0;
constexpr double BAR_HALF_W = 0.025;
constexpr double FLUX_SLAB_HALF = 0.05;
constexpr double FLUX_SLAB_LO = -FLUX_SLAB_HALF;
constexpr double FLUX_SLAB_HI = FLUX_SLAB_HALF;
constexpr double FLUX_REACH_PAST_SHOULDER = 0.65 + PALM_OFF - FLUX_SLAB_LO;
// env v19 flail telemetry: a tick is "note-free" when the next
// scoreable note is further out than this (s). Telemetry only — no
// reward or obs effect.
constexpr double FLAIL_LEAD_S = 1.0;
// wrist-bend reversal telemetry (2026-09-22, user): per hand, a reversal
// is counted when the wrist BEND axis (joints 8 / 15) turns back by
// WREV_HYST from its running extreme — position-based peak detection
// with hysteresis, so tremor under 20 deg never counts; the first
// commitment out of rest is not a reversal, a flick-and-return counts
// 1 (the turn at the peak), steady oscillation 2 per cycle. Every
// tick, no note gating. Mirrored in lib.rs (same constant, same state
// machine).
constexpr double WREV_HYST = 0.34906585;   // 20 deg
// env v21 (lib.rs mirrors, exact shared f64 literals): wide stance —
// hip_roll splays the legs +-8 deg, ankle_roll counter-rolls so the
// soles stay flat. FOOT_X0 = the stance's per-side foot x (fk of
// the start pose; the root ground-drop does not move x).
constexpr double Q_START_HIP_ROLL = 0.13962634015954636;   // 8 deg
constexpr double FOOT_X0 = 0.22108059783525694;
constexpr int FOLLOW_TIMEOUT = 15;
constexpr double LOOK_VERT = 0.8;
constexpr double LOOK_DIST = 8.0;
constexpr double X_NORM = 1.0;
constexpr double Y_NORM = 0.75;
constexpr double Z_SCALE = 8.0;
constexpr double ROOT_Y0 = 0.94;
constexpr double HEAD_Y0 = 1.70;
constexpr double PLANE_Z0 = -0.65;
constexpr double ROOT_VLIN_NORM = 6.0;
constexpr double ROOT_VANG_NORM = 16.0;
// env v11 FALL termination (unilateral-contact spec §4) — lib.rs
// FALL_ROOT_Y. Root below this at the END of a tick
// (per-TICK since 2026-08-20, user: consistent with the tilt check;
// the per-substep dip test is retired) ends the episode with a ZEROED
// reward, reported exactly like the root-escape envelope (SPEC-v10
// §20e "reported, never hidden").
constexpr double FALL_ROOT_Y = 0.25;  // 2026-08-21: 0.5 -> 0.25 (user; follows the player-height revert to 1.8 — full crouch headroom restored)
// TILT termination (user 2026-08-20) — lib.rs
// mirror: CHEST up-axis tilted more than 40 deg FORWARD, 20 deg
// BACKWARD (asymmetric sagittal, user 2026-08-20 — the tight back
// bound targets the v30 backbend exploit; forward stays duck-sized)
// or 40 deg to the side (lateral) at the END of a tick ends the
// episode, priced like a fall (flat -fall_const). Checked per TICK,
// not per substep. Tangent-squared form on the chest up-axis;
// forward = up_z > 0 (body faces +z); also trips whenever
// up_y <= 0 (past horizontal).
constexpr double TILT_TAN2_FWD = 0.7040881910418472;   // tan^2(40 deg)
constexpr double TILT_TAN2_BACK = 0.1324743314317942;  // tan^2(20 deg) — 2026-08-21: briefly 10 deg, reverted to 20 same day (user)
constexpr double TILT_TAN2_ROLL = 0.7040881910418472;  // tan^2(40 deg)
// FOOT-LIFT penalty (user 2026-08-20) — lib.rs
// mirror: flat charge on the TICK a foot leaves the ground (edge:
// that foot's tick-summed ground normal impulse hits zero after a
// grounded tick), per foot. Grazing contact can flicker the edge;
// the pop counter makes that visible if it ever matters.
constexpr double FOOT_LIFT_PEN = -0.1;

// env v8 gears (lib.rs gear_signed)
__device__ constexpr double GEAR_TAB[3][7] = {
    {0.0, 1.0, -1.0, 0.0, 0.0, 0.0, 0.0},                     // K=3
    {0.0, 1.0, -1.0, 0.25, -0.25, 0.0, 0.0},                  // K=5
    {0.0, 1.0, -1.0, 1.0 / 3.0, -1.0 / 3.0, 0.125, -0.125}};  // K=7
__device__ __forceinline__ const double* gear_row(int gears) {
  return GEAR_TAB[gears == 5 ? 1 : (gears == 7 ? 2 : 0)];
}
// main's v63x2 body (since 2026-10-03): the wrist-bend joints (8, 15) take
// their fine gears (ids >= 3) at 2x — K=5's 1/4 gear becomes 1/2 there
// (lib.rs FINE_GEAR_MULT mirror; exact in both dtypes)
__device__ constexpr double FINE_GEAR_MULT[NQ] = {
    1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 2, 1,
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1};

// note kinds (constants.py)
// KIND_NOTE=0, KIND_BOMB=1, KIND_CHAIN_HEAD=2, KIND_CHAIN_LINK=3 —
// pool tensors carry the derived booleans directly.

// ---- small device math ------------------------------------------------------
template <typename T>
__device__ __forceinline__ T tmin(T a, T b) { return a < b ? a : b; }
template <typename T>
__device__ __forceinline__ T tmax(T a, T b) { return a > b ? a : b; }
// env v48: the obs z lane for a travelling object (lib.rs z_lane mirror);
// z_asinh > 0 always (host-validated)
template <typename T>
__device__ __forceinline__ T z_lane(T zu, T z_lo, T z_asinh) {
  return asinh((zu > z_lo ? zu : z_lo) / z_asinh);
}
template <typename T>
__device__ __forceinline__ T tclamp(T x, T lo, T hi) {
  return x < lo ? lo : (x > hi ? hi : x);
}
template <typename T>
__device__ __forceinline__ T round_pts(T x) { return floor(x + T(0.5)); }
template <typename T>
__device__ __forceinline__ T dot3(const T a[3], const T b[3]) {
  return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}
template <typename T>
__device__ __forceinline__ void cross3(const T a[3], const T b[3], T o[3]) {
  o[0] = a[1] * b[2] - a[2] * b[1];
  o[1] = a[2] * b[0] - a[0] * b[2];
  o[2] = a[0] * b[1] - a[1] * b[0];
}
template <typename T>
__device__ __forceinline__ void sub3v(const T a[3], const T b[3], T o[3]) {
  o[0] = a[0] - b[0];
  o[1] = a[1] - b[1];
  o[2] = a[2] - b[2];
}

// ---- dynamics scratch (thread-local) ----------------------------------------
template <typename T>
struct Dyn {
  // state
  T p[3], quat[4], q[NQ], v[NV];
  // FK
  T r[NF][3][3], pos[NF][3], axw[NF][3];
  T r0[3][3];
  // audit #10: m holds M from crba, then its IN-PLACE Cholesky factor L,
  // in the TREE PROFILE layout (TreeTabs: column-major, 341 slots — the
  // dense form was NV*NV = 1,225)
  T m[TC_NNZ], c[NV], rhs[NV];
  // CRBA composites
  T cm[NF], cc[NF][3], ci[NF][3][3];
  T bm, bc[3], bi[3][3];
  // R diag(i) R^T per frame, computed by crba and re-read by rnea_bias
  // (K4e) instead of recomputed there from the same d.r / F_INERTIA.
  // ci/bi hold the same values on entry but are MUTATED by the leaf->root
  // composite merge, so the raw world inertia needs its own slot; it is
  // written and read under the identical F_MASS[k] > 0 predicate, and
  // crba runs immediately before rnea_bias in substep with d.r untouched
  // between them (crba writes only m / cm / cc / ci / b* / iw*).
  T iw[NF][3][3], iw0[3][3];
  // RNEA scratch. NO `vo` (K10a): the linear-velocity forward chain
  // vo[k] = vo[par] + om[par] x (pos[k] - pos[par]) is a fossil of the
  // general RNEA — this call site is RNEA(q, v, 0) and the bias it
  // returns is assembled from om / al / ao / fb / nb only. vo was
  // written by the forward pass and read by nothing but its own
  // recurrence, so the whole chain was dead (om[par] x d IS still
  // needed, as the seed of the centrifugal term om x (om x d)).
  T om[NF][3], al[NF][3], ao[NF][3], fb[NF][3], nb[NF][3];
  bool pd_bad;
  // PRE-integration root-envelope engagements (Divergences §23):
  // env_bad = "engaged anywhere in THIS control tick" (the C backstop
  // trigger), env_n = the per-(env, substep) count the wrapper pops
  bool env_bad;
  int env_n;
};

// GROUND-BLOCK scratch (unilateral-contact spec §3): 3 rows per ACTIVE
// sole corner, corners in the §2 normative order. Same field shape as
// the self-collision contact rows — bias / diag / lambda — because the
// normal row's PGS update IS the contact path's and the friction rows
// differ from it only in the projection. A base of Solver (as it has
// been since v11), so the frame layout is unchanged.
//
// G1 (2026-09-26) — FIXED SLOTS + SUPPORT-ONLY STORAGE. Row slot 3*c + t
// belongs to corner c = 4*foot + k (spec §2 order) whatever the active
// set is; bit c of g_act says whether the corner is in contact. Row t of
// a corner is point_jacobian_acc with n = e_{GROUND_AXIS[t]}, i.e. a foot
// LINEAR row (the support derivation at WJ_L0): one entry of 1.0 at
// column GROUND_AXIS[t], two root values at WJ_R2[axis], six chain values
// at WJ_L0 / WJ_R0 + 0..5; every other column is the zero fill's +0.0.
//
// audit #9 (2026-09-26) — the Jacobian object is no longer the corner
// rows but each foot's SIX SPATIAL ROWS (GROUND_SPATIAL above), stored
// support-only: g_fjc[f][m] the six chain values of spatial row m,
// g_fjr[f][t] linear row t's two root values at WJ_R2[t] (the 1.0 entries
// implicit). g_minv is M^-1 J_f^T for BOTH feet RHS-MINOR ([NV][12], foot
// f at columns 6f..6f+5): ONE tree_solve_feet_tn per substep with any
// corner down. Per corner row: g_coef its two angular coefficients of c,
// g_u = Phi_f c (its image in foot-velocity space, Phi_f = J_f M^-1
// J_f^T), g_diag = c . g_u.
template <typename T>
struct GroundRows {
  uint32_t g_act;  // bit c: corner c in contact this substep (0..7)
  T g_fjc[2][FOOT_ROWS][6], g_fjr[2][3][2];
  T g_minv[NV][2 * FOOT_ROWS];
  T g_coef[N_GROUND_ROWS][2], g_u[N_GROUND_ROWS][FOOT_ROWS];
  T g_bias[N_GROUND_ROWS], g_diag[N_GROUND_ROWS], g_lam[N_GROUND_ROWS];
};
template <typename T>
struct Solver : GroundRows<T> {
  // contacts (compacted, pair-id order)
  int nc;
  int c_pair[NUM_PAIRS];
  // c_minv stays ROW-MAJOR. Transposing it to [NV][NUM_PAIRS] (K11's
  // shape) was built and measured at -0.33% — see the K18(i) entry in
  // the file header for the numbers and for why the mw result does not
  // transfer. Do not re-try it.
  T c_J[NUM_PAIRS][NV], c_minv[NUM_PAIRS][NV];
  T c_bias[NUM_PAIRS], c_diag[NUM_PAIRS], c_lam[NUM_PAIRS];
  T c_pt[NUM_PAIRS][3];
  // limit rows (joint order); l_j[nl..] carries the #6e predicted
  // column list transiently (read by nothing past the column block)
  int nl;
  int l_j[NQ];
  T l_sign[NQ], l_bias[NQ], l_lam[NQ], l_diag[NQ];
  // shared per-joint columns: ecol = M^-1 e_j (limit rows, step-8 lazy
  // cap, step-11 clamps), pdiag = its mobility e_j^T M^-1 e_j with the
  // COMPOSITE_EPS verdict (INF = degenerate, row/cap inert); pcomp
  // caches pdiag. Bitwise identical to Rust's separately-computed
  // minvjt/capcol/solcache columns (linear solves commute with the +-1
  // row sign exactly; recomputation of the same expression tree is
  // deterministic).
  //
  // COLUMN SLOTS (perf/kernel-limits, audit-2 #6a). A column is stored
  // in SLOT eslot[j], not row j: local memory interleaves words across
  // the warp, so only a SAME-word access coalesces, and a limit row's
  // joint differs per lane — s.ecol[l_j[li]][i] put every lane of the
  // seed / back-substitution / PGS axpy on its own 128-byte line. Slots
  // are handed out in computation order: the limit-row group (always the
  // substep's first columns, see constraint_solve) takes slots 0..nl-1 in
  // row order, so row li's column is s.ecol[li], a warp-uniform word; a
  // lazy cap / clamp column takes the next free slot (nslot). At most NQ
  // distinct joints per substep, so NQ slots suffice. Storage only: every
  // value and the order anything is computed or applied in are unchanged.
  // Measured on the RTX 5090 at N=32768 (trained standard, sampled):
  // 69.5 -> 60.9 ms/tick alone, before the register passes (#6c).
  T ecol[NQ][NV], pdiag[NQ];
  int nslot;
  uint8_t eslot[NQ];
  // #6e column prediction: emask = joints whose column this substep
  // built, eprev = the previous substep's emask (0 at the tick's start)
  uint32_t emask, eprev;
  bool ecomp[NQ], pcomp[NQ];
};

// ---- float twins of the double constant tables (audit #5c) -----------------
// Every hot read of a model table is `T(TABLE[...])`: under T = float
// that was a double load plus a runtime f64->f32 conversion (reduced
// rate on GeForce parts) inside the rolled 30-frame FK/CRBA/RNEA loops
// and the per-substep limit/cap passes. The twins hold the SAME values
// converted at compile time — round-to-nearest either way, so every
// float result is bit-identical — and CT(X, [i]...) picks the twin for
// T = float and the original (converted as before) for T = double.
// Reads that do double ARITHMETIC before converting keep their double
// form (or fold it at compile time exactly: PAIR_RSUM below).
template <int A>
struct FTab1 { float v[A]; };
template <int A, int B>
struct FTab2 { float v[A][B]; };
template <int A>
constexpr FTab1<A> f32_tab(const double (&x)[A]) {
  FTab1<A> o{};
  for (int i = 0; i < A; i++) o.v[i] = (float)x[i];
  return o;
}
template <int A, int B>
constexpr FTab2<A, B> f32_tab(const double (&x)[A][B]) {
  FTab2<A, B> o{};
  for (int i = 0; i < A; i++)
    for (int j = 0; j < B; j++) o.v[i][j] = (float)x[i][j];
  return o;
}
__device__ constexpr auto F_ANCHOR_F = f32_tab(F_ANCHOR);
__device__ constexpr auto F_AXIS_F = f32_tab(F_AXIS);
__device__ constexpr auto F_MASS_F = f32_tab(F_MASS);
__device__ constexpr auto F_COM_F = f32_tab(F_COM);
__device__ constexpr auto F_INERTIA_F = f32_tab(F_INERTIA);
__device__ constexpr auto BASE_COM_F = f32_tab(BASE_COM);
__device__ constexpr auto ARMATURE_F = f32_tab(ARMATURE);
__device__ constexpr auto DAMPING_F = f32_tab(DAMPING);
__device__ constexpr auto LIMIT_LO_F = f32_tab(LIMIT_LO);
__device__ constexpr auto LIMIT_HI_F = f32_tab(LIMIT_HI);
__device__ constexpr auto OMEGA_CAP_F = f32_tab(OMEGA_CAP);
__device__ constexpr auto TAU_MAX_F = f32_tab(TAU_MAX);
__device__ constexpr auto COL_A_F = f32_tab(COL_A);
__device__ constexpr auto COL_B_F = f32_tab(COL_B);
__device__ constexpr auto COL_R_F = f32_tab(COL_R);
__device__ constexpr auto GROUND_CORNERS_F = f32_tab(GROUND_CORNERS);
__device__ constexpr auto GROUND_DIRS_F = f32_tab(GROUND_DIRS);
// T(COL_R[a] + COL_R[b]) per pair: the double sum folded at compile time
// (an IEEE double add either way), then converted like any table
struct PairRSum { double v[NUM_PAIRS]; };
constexpr PairRSum pair_rsum() {
  PairRSum t{};
  for (int p = 0; p < NUM_PAIRS; p++) t.v[p] = COL_R[PAIR_A[p]] + COL_R[PAIR_B[p]];
  return t;
}
__device__ constexpr PairRSum PAIR_RSUM = pair_rsum();
__device__ constexpr auto PAIR_RSUM_F = f32_tab(PAIR_RSUM.v);
#define CT(X, IDX) \
  (std::is_same<T, float>::value ? T(X##_F.v IDX) : T(X IDX))

// ---- fk.rs ------------------------------------------------------------------
template <typename T, typename A>
__device__ void rot_axis(const A axis_d[3], T angle, T out[3][3]) {
  T s = sin(angle), cth = cos(angle);
  T k = T(1) - cth;
  T x = T(axis_d[0]), y = T(axis_d[1]), z = T(axis_d[2]);
  out[0][0] = cth + k * x * x;
  out[0][1] = k * x * y - s * z;
  out[0][2] = k * x * z + s * y;
  out[1][0] = k * x * y + s * z;
  out[1][1] = cth + k * y * y;
  out[1][2] = k * y * z - s * x;
  out[2][0] = k * x * z - s * y;
  out[2][1] = k * y * z + s * x;
  out[2][2] = cth + k * z * z;
}

template <typename T>
__device__ void quat_to_rot(const T q[4], T r[3][3]) {
  T w = q[0], x = q[1], y = q[2], z = q[3];
  r[0][0] = T(1) - T(2) * (y * y + z * z);
  r[0][1] = T(2) * (x * y - w * z);
  r[0][2] = T(2) * (x * z + w * y);
  r[1][0] = T(2) * (x * y + w * z);
  r[1][1] = T(1) - T(2) * (x * x + z * z);
  r[1][2] = T(2) * (y * z - w * x);
  r[2][0] = T(2) * (x * z - w * y);
  r[2][1] = T(2) * (y * z + w * x);
  r[2][2] = T(1) - T(2) * (x * x + y * y);
}

template <typename T>
__device__ void quat_mul(const T a[4], const T b[4], T o[4]) {
  o[0] = a[0] * b[0] - a[1] * b[1] - a[2] * b[2] - a[3] * b[3];
  o[1] = a[0] * b[1] + a[1] * b[0] + a[2] * b[3] - a[3] * b[2];
  o[2] = a[0] * b[2] - a[1] * b[3] + a[2] * b[0] + a[3] * b[1];
  o[3] = a[0] * b[3] + a[1] * b[2] - a[2] * b[1] + a[3] * b[0];
}

// Aliasing contract for the pass-by-reference scratch (2026-08-02):
// `d` and `s` are two DISTINCT stack objects, declared side by side in
// step_kernel, and no other pointer into either escapes — so every
// `Dyn<T>& __restrict__ d` / `Solver<T>& __restrict__ s` parameter below
// is legal and true. It matters because the kernel is local-memory-
// traffic bound: without it, a store to any Solver field (s.g_lam,
// s.c_lam …) makes the compiler re-load the Dyn fields it was holding
// — above all d.m, which holds the Cholesky factor read by every solve.
// Read-only aliases are still permitted under restrict (only MODIFIED
// objects must be reached through the one pointer), which is what keeps
// `constraint_solve(d, s, d.q, v)` — qpre pointing into d — well
// defined: nothing writes d.q in that scope.
// Measured (GTX 1080, sm_61): f32 SASS 32,880 -> 32,682 instructions,
// LDL 7,091 -> 6,972, STL 3,682 -> 3,640; ptxas f32 frame 51,056 B
// unchanged with spills 8,244/7,164 -> 8,260/7,176, f64 frame
// 101,920 -> 101,936 B with spills 19,476/19,516 -> 18,888/18,992.
// Bit-for-bit identical trajectories in both dtypes; wall time
// neutral within noise (the kernel is serial-latency bound, so the
// ~200 removed local-memory instructions do not show up as steps/s) —
// kept for the codegen headroom the later algorithmic passes need.
//
// NO `#pragma unroll` ANYWHERE, deliberately (measured 2026-08-02): a
// pass adding 55 pragmas to every constant-trip-3/4 loop in fk_all /
// crba / rnea_bias / the jacobians / constraint_solve / substep
// produced BYTE-IDENTICAL f32 SASS (and identical instruction counts +
// spill bytes for f64) — nvcc -O3 already fully unrolls them, and the
// f32 kernel has only 85 backward branches left in 32,682
// instructions. Everything still rolled is a big loop (NF frames,
// NV=35 solves, the 12 PGS iterations); unrolling those is the
// opposite trade in a kernel already at 128 regs + 8 KB of spills, so
// it belongs with the algorithmic Cholesky work, not with a
// mechanical pragma pass. That Cholesky work has since landed (see
// chol_factor_n): a hand-written unroll-and-JAM by 4 of the factor's
// row loop — worth +2.0% because it cuts dependency-chain latency (4
// independent accumulators) rather than branch overhead, which is
// exactly what a bare `#pragma unroll` on the same loop cannot do.

// K10b — d.axw[k] is DEAD for every frame with F_COORD[k] < 0, which is
// frame 3 (the head) alone. Every reader is guarded by the same
// predicate: crba's column fill (`coord >= 0 && m > 0`) and its ancestor
// walk (`F_COORD[anc] >= 0`), rnea_bias's forward joint term and its
// backward `d.c[VOFF + F_COORD[k]]` write, and both Jacobian
// accumulators (`if (F_COORD[k] >= 0)`) — so the axis is only ever read
// through a coordinate index that frame 3 does not have. fk_all
// therefore computes it inside the same branch that builds the joint
// rotation; d.axw[3] is simply never written (nor read).
constexpr int coordless_frames() {
  int n = 0;
  for (int k = 0; k < NF; k++) n += F_COORD[k] < 0 ? 1 : 0;
  return n;
}
static_assert(coordless_frames() == 1 && F_COORD[3] < 0,
              "K10b: frame 3 must be the ONLY F_COORD < 0 frame");

// fk_all: world rotation/origin per frame + world revolute axes.
template <typename T>
__device__ void fk_all(Dyn<T>& __restrict__ d) {
  quat_to_rot(d.quat, d.r0);
  for (int k = 0; k < NF; k++) {
    const int par = F_PARENT[k];
    const T (*rl)[3] = par < 0 ? d.r0 : d.r[par];
    const T* pl = par < 0 ? d.p : d.pos[par];
    T av[3];
    for (int i = 0; i < 3; i++) {
      av[i] = rl[i][0] * CT(F_ANCHOR, [k][0]) + rl[i][1] * CT(F_ANCHOR, [k][1])
              + rl[i][2] * CT(F_ANCHOR, [k][2]);
      d.pos[k][i] = pl[i] + av[i];
    }
    if (F_COORD[k] >= 0) {
      T rj[3][3];
      if constexpr (std::is_same<T, float>::value)
        rot_axis(F_AXIS_F.v[k], d.q[F_COORD[k]], rj);
      else
        rot_axis(F_AXIS[k], d.q[F_COORD[k]], rj);
      for (int i = 0; i < 3; i++)
        for (int j = 0; j < 3; j++)
          d.r[k][i][j] = rl[i][0] * rj[0][j] + rl[i][1] * rj[1][j]
                         + rl[i][2] * rj[2][j];
      for (int i = 0; i < 3; i++)
        d.axw[k][i] = d.r[k][i][0] * CT(F_AXIS, [k][0])
                      + d.r[k][i][1] * CT(F_AXIS, [k][1])
                      + d.r[k][i][2] * CT(F_AXIS, [k][2]);
    } else {
      for (int i = 0; i < 3; i++)
        for (int j = 0; j < 3; j++) d.r[k][i][j] = rl[i][j];
    }
  }
}

// ---- algo.rs ---------------------------------------------------------------
// I_world = R diag(i) R^T — upper computed, lower mirrored (bitwise
// symmetric, algo.rs inertia_world), teed into two destinations from
// the one register value v (K4e): crba needs a MUTABLE composite seed
// and a pristine copy for rnea_bias, and both get identical bits with
// no read-back.
template <typename T, typename A>
__device__ void inertia_world2(const T r[3][3], const A idiag[3],
                               T out[3][3], T keep[3][3]) {
  for (int a = 0; a < 3; a++)
    for (int b = a; b < 3; b++) {
      T v = r[a][0] * T(idiag[0]) * r[b][0] + r[a][1] * T(idiag[1]) * r[b][1]
            + r[a][2] * T(idiag[2]) * r[b][2];
      out[a][b] = v;
      out[b][a] = v;
      keep[a][b] = v;
      keep[b][a] = v;
    }
}

template <typename T>
__device__ void add_shift(T i[3][3], T m, const T d[3]) {
  T dd = dot3(d, d);
  for (int a = 0; a < 3; a++)
    for (int b = 0; b < 3; b++)
      i[a][b] += m * ((a == b ? dd : T(0)) - d[a] * d[b]);
}

// audit #10 (2026-09-26): crba writes M STRAIGHT INTO THE TREE PROFILE
// (d.m[TC_NNZ], column-major, see TreeTabs): each joint's column is its
// chain — d.m[fcb[k]] the diagonal, d.m[fcb[k] + a] its a-th coord-bearing
// ancestor (the ANC.f row, nearest first — exactly the column's slot
// order, tree_tabs_ok() pins it), d.m[fcb[k] + ANC_N[k] + t] the six base
// rows — and the 6x6 base block fills columns TC_BASE_E.. (compile-time
// slots). Every stored VALUE is the one the dense form stored; only the
// address moved. That retires three things at once:
//   * the zero fill (K4c had narrowed it to the 630 lower+diagonal
//     entries): every profile slot is written exactly once below
//     (tc_all_subtrees_massive() pins the one precondition), and nothing
//     outside the profile exists any more;
//   * the runtime min/max select on the ancestor pair (coord order is not
//     frame order in the shoulder/wrist stacks): the slot is simply
//     fcb[k] + a;
//   * 884 words of frame (1,225 -> 341).
// The dense-era history (K4c lower-triangle-only, K12 column-major) is in
// git; both of its invariants survive in profile form (only the lower
// triangle exists; columns are contiguous).
template <typename T>
__device__ void crba(Dyn<T>& __restrict__ d) {
  // composite seeds
  for (int k = 0; k < NF; k++) {
    const T mk = CT(F_MASS, [k]);
    d.cm[k] = mk;
    if (CT(F_MASS, [k]) > T(0)) {
      for (int i = 0; i < 3; i++)
        d.cc[k][i] = d.pos[k][i] + (d.r[k][i][0] * CT(F_COM, [k][0])
                                    + d.r[k][i][1] * CT(F_COM, [k][1])
                                    + d.r[k][i][2] * CT(F_COM, [k][2]));
      if constexpr (std::is_same<T, float>::value)
        inertia_world2(d.r[k], F_INERTIA_F.v[k], d.ci[k], d.iw[k]);
      else
        inertia_world2(d.r[k], F_INERTIA[k], d.ci[k], d.iw[k]);
    } else {
      for (int i = 0; i < 3; i++) d.cc[k][i] = d.pos[k][i];
      for (int i = 0; i < 3; i++)
        for (int j = 0; j < 3; j++) d.ci[k][i][j] = T(0);
    }
  }
  // base composite (pelvis)
  d.bm = T(BASE_MASS);
  for (int i = 0; i < 3; i++)
    d.bc[i] = d.p[i] + (d.r0[i][0] * CT(BASE_COM, [0])
                        + d.r0[i][1] * CT(BASE_COM, [1])
                        + d.r0[i][2] * CT(BASE_COM, [2]));
  inertia_world2(d.r0, BASE_INERTIA, d.bi, d.iw0);

  // leaf->root accumulation with column fill before parent merge
  for (int k = NF - 1; k >= 0; k--) {
    const int coord = F_COORD[k];
    const T m = d.cm[k];
    T cvec[3] = {d.cc[k][0], d.cc[k][1], d.cc[k][2]};
    if (coord >= 0 && m > T(0)) {
      const int cbk = TREE_C.fcb[k];  // this coordinate's profile column
      const T* a = d.axw[k];
      T rel[3], hl[3], ha[3];
      sub3v(cvec, d.pos[k], rel);
      T cr[3];
      cross3(a, rel, cr);
      for (int i = 0; i < 3; i++) hl[i] = cr[i] * m;
      for (int i = 0; i < 3; i++)
        ha[i] = d.ci[k][i][0] * a[0] + d.ci[k][i][1] * a[1]
                + d.ci[k][i][2] * a[2];
      T relxhl[3];
      cross3(rel, hl, relxhl);
      T tot[3] = {ha[0] + relxhl[0], ha[1] + relxhl[1], ha[2] + relxhl[2]};
      d.m[cbk] = dot3(a, tot);
      // K17 — the same coord-bearing ancestors, in the same order, read
      // out of ANC.f / ANC.c instead of chased through F_PARENT. Entry 0
      // is k itself (F_COORD[k] >= 0 holds here), so the strict
      // ancestors are the a = 1.. suffix; the per-step
      // `if (F_COORD[anc] >= 0)` filter is baked into the row.
      const int na = ANC_N[k];
      for (int a = 1; a < na; a++) {
        const int anc = ANC.f[k][a];
        T rel2[3];
        sub3v(cvec, d.pos[anc], rel2);
        T r2xhl[3];
        cross3(rel2, hl, r2xhl);
        T tot2[3] = {ha[0] + r2xhl[0], ha[1] + r2xhl[1], ha[2] + r2xhl[2]};
        // audit #10: slot a of the column IS ancestor a (tree_tabs_ok)
        d.m[cbk + a] = dot3(d.axw[anc], tot2);
      }
      // root rows
      T relr[3];
      sub3v(cvec, d.p, relr);
      T rrxhl[3];
      cross3(relr, hl, rrxhl);
      T lr[3] = {ha[0] + rrxhl[0], ha[1] + rrxhl[1], ha[2] + rrxhl[2]};
      // the base rows close the column: v0..v5 at slots na .. na + 5
      for (int t = 0; t < 3; t++) {
        d.m[cbk + na + t] = hl[t];
        d.m[cbk + na + 3 + t] = lr[t];
      }
    }
    // merge composite k into parent (or base)
    const int par = F_PARENT[k];
    T* pm;
    T* pc;
    T (*pi)[3];
    if (par < 0) {
      pm = &d.bm;
      pc = d.bc;
      pi = d.bi;
    } else {
      pm = &d.cm[par];
      pc = d.cc[par];
      pi = d.ci[par];
    }
    const T tm = *pm + m;
    if (tm > T(0)) {
      const T inv = T(1) / tm;
      T tc[3];
      for (int i = 0; i < 3; i++) tc[i] = (pc[i] * *pm + cvec[i] * m) * inv;
      T dp[3], dc[3];
      sub3v(pc, tc, dp);
      add_shift(pi, *pm, dp);
      sub3v(cvec, tc, dc);
      add_shift(pi, m, dc);
      for (int i = 0; i < 3; i++)
        for (int j = 0; j < 3; j++) pi[i][j] += d.ci[k][i][j];
      *pm = tm;
      for (int i = 0; i < 3; i++) pc[i] = tc[i];
    }
  }

  // armature diagonal (Divergences §12)
  for (int k = 0; k < NF; k++)
    if (F_COORD[k] >= 0) d.m[TREE_C.fcb[k]] += CT(ARMATURE, [F_COORD[k]]);

  // root 6x6 from the total composite
  T dd[3];
  sub3v(d.bc, d.p, dd);
  T rr[3][3];
  for (int i = 0; i < 3; i++)
    for (int j = 0; j < 3; j++) rr[i][j] = d.bi[i][j];
  add_shift(rr, d.bm, dd);
  // base block: element (row R, col C), R >= C, lives at slot R - C of
  // base column C (e = TC_BASE_E + C) — compile-time addresses
  constexpr int bcb[VOFF] = {TREE.cb[TC_BASE_E], TREE.cb[TC_BASE_E + 1],
                             TREE.cb[TC_BASE_E + 2], TREE.cb[TC_BASE_E + 3],
                             TREE.cb[TC_BASE_E + 4], TREE.cb[TC_BASE_E + 5]};
  for (int i = 0; i < 3; i++)
    for (int j = 0; j <= i; j++) {
      d.m[bcb[j] + (i - j)] = i == j ? d.bm : T(0);
      d.m[bcb[3 + j] + (i - j)] = rr[i][j];
    }
  const T sk[3][3] = {{T(0), -dd[2], dd[1]},
                      {dd[2], T(0), -dd[0]},
                      {-dd[1], dd[0], T(0)}};
  // (3+i, j) is strictly lower for every i, j in 0..2
  for (int i = 0; i < 3; i++)
    for (int j = 0; j < 3; j++) d.m[bcb[j] + (3 + i - j)] = d.bm * sk[i][j];
}

// RNEA(q, v, 0) -> bias C (gravity included), result in d.c.
template <typename T>
__device__ void rnea_bias(Dyn<T>& __restrict__ d) {
  const T g[3] = {T(0), T(-GRAV), T(0)};
  T om0[3] = {d.v[3], d.v[4], d.v[5]};
  // forward pass (acc = 0 everywhere; al0/ao0 = 0)
  for (int k = 0; k < NF; k++) {
    const int par = F_PARENT[k];
    const T* oml = par < 0 ? om0 : d.om[par];
    const T* all = par < 0 ? nullptr : d.al[par];
    const T* aol = par < 0 ? nullptr : d.ao[par];
    const T* ol = par < 0 ? d.p : d.pos[par];
    T dv[3];
    sub3v(d.pos[k], ol, dv);
    T oxd[3];
    cross3(oml, dv, oxd);
    T alxd[3] = {T(0), T(0), T(0)};
    if (all != nullptr) cross3(all, dv, alxd);
    T oxoxd[3];
    cross3(oml, oxd, oxoxd);
    T aok[3];
    for (int i = 0; i < 3; i++)
      aok[i] = ((aol != nullptr ? aol[i] : T(0)) + alxd[i]) + oxoxd[i];
    T omk[3], alk[3];
    if (F_COORD[k] >= 0) {
      const T qd = d.v[VOFF + F_COORD[k]];
      const T* aw = d.axw[k];
      for (int i = 0; i < 3; i++) omk[i] = oml[i] + aw[i] * qd;
      T oxa[3];
      cross3(oml, aw, oxa);
      for (int i = 0; i < 3; i++)
        alk[i] = (all != nullptr ? all[i] : T(0)) + oxa[i] * qd;
    } else {
      for (int i = 0; i < 3; i++) {
        omk[i] = oml[i];
        alk[i] = all != nullptr ? all[i] : T(0);
      }
    }
    for (int i = 0; i < 3; i++) {
      d.om[k][i] = omk[i];
      d.al[k][i] = alk[i];
      d.ao[k][i] = aok[i];
    }
    if (CT(F_MASS, [k]) > T(0)) {
      T e[3];
      for (int i = 0; i < 3; i++)
        e[i] = d.r[k][i][0] * CT(F_COM, [k][0]) + d.r[k][i][1] * CT(F_COM, [k][1])
               + d.r[k][i][2] * CT(F_COM, [k][2]);
      T alxe[3], omxe[3], omxomxe[3];
      cross3(alk, e, alxe);
      cross3(omk, e, omxe);
      cross3(omk, omxe, omxomxe);
      T ac[3];
      for (int i = 0; i < 3; i++) ac[i] = (aok[i] + alxe[i]) + omxomxe[i];
      const T(&iw)[3][3] = d.iw[k];  // K4e
      T fbk[3];
      for (int i = 0; i < 3; i++) fbk[i] = (ac[i] - g[i]) * CT(F_MASS, [k]);
      T iwal[3], iwom[3];
      for (int i = 0; i < 3; i++) {
        iwal[i] = iw[i][0] * alk[0] + iw[i][1] * alk[1] + iw[i][2] * alk[2];
        iwom[i] = iw[i][0] * omk[0] + iw[i][1] * omk[1] + iw[i][2] * omk[2];
      }
      T omxiw[3];
      cross3(omk, iwom, omxiw);
      T nbk[3];
      for (int i = 0; i < 3; i++) nbk[i] = iwal[i] + omxiw[i];
      T exf[3];
      cross3(e, fbk, exf);
      for (int i = 0; i < 3; i++) {
        d.fb[k][i] = fbk[i];
        d.nb[k][i] = nbk[i] + exf[i];
      }
    } else {
      for (int i = 0; i < 3; i++) {
        d.fb[k][i] = T(0);
        d.nb[k][i] = T(0);
      }
    }
  }
  // base body inertial force/moment (al0 = ao0 = 0)
  T f0[3], n0[3];
  {
    T e[3];
    for (int i = 0; i < 3; i++)
      e[i] = d.r0[i][0] * CT(BASE_COM, [0]) + d.r0[i][1] * CT(BASE_COM, [1])
             + d.r0[i][2] * CT(BASE_COM, [2]);
    T omxe[3], omxomxe[3];
    cross3(om0, e, omxe);
    cross3(om0, omxe, omxomxe);
    T ac[3] = {omxomxe[0], omxomxe[1], omxomxe[2]};
    const T(&iw)[3][3] = d.iw0;  // K4e
    for (int i = 0; i < 3; i++) f0[i] = (ac[i] - g[i]) * T(BASE_MASS);
    T iwom[3];
    for (int i = 0; i < 3; i++)
      iwom[i] = iw[i][0] * om0[0] + iw[i][1] * om0[1] + iw[i][2] * om0[2];
    T omxiw[3];
    cross3(om0, iwom, omxiw);
    T exf[3];
    cross3(e, f0, exf);
    for (int i = 0; i < 3; i++) n0[i] = omxiw[i] + exf[i];
  }
  // backward pass
  // K20-2: NO ZERO FILL. Every d.c[i] is ASSIGNED below — the joint
  // columns by `d.c[VOFF + F_COORD[k]] = ...` (F_COORD is a bijection
  // onto 0..NV-VOFF-1, static_asserted at the table) and 0..5 from
  // f0/n0 at the end — and nothing reads d.c in between, so the 35
  // stores the fill used to issue were all dead.
  for (int k = NF - 1; k >= 0; k--) {
    if (F_COORD[k] >= 0)
      d.c[VOFF + F_COORD[k]] = dot3(d.axw[k], d.nb[k]);
    const int par = F_PARENT[k];
    if (par < 0) {
      T lever[3];
      sub3v(d.pos[k], d.p, lever);
      T lxf[3];
      cross3(lever, d.fb[k], lxf);
      for (int i = 0; i < 3; i++) {
        f0[i] += d.fb[k][i];
        n0[i] += d.nb[k][i] + lxf[i];
      }
    } else {
      T lever[3];
      sub3v(d.pos[k], d.pos[par], lever);
      T lxf[3];
      cross3(lever, d.fb[k], lxf);
      for (int i = 0; i < 3; i++) {
        d.fb[par][i] += d.fb[k][i];
        d.nb[par][i] += d.nb[k][i] + lxf[i];
      }
    }
  }
  d.c[0] = f0[0];
  d.c[1] = f0[1];
  d.c[2] = f0[2];
  d.c[3] = n0[0];
  d.c[4] = n0[1];
  d.c[5] = n0[2];
}

// (The dense Cholesky chol_factor_n / chol_solve_n / chol_solve_multi_n
// — after audit #10 they served only the 12x12 weld block A_ww — were
// deleted with weld mode on 2026-09-27. Their loop-shape notes (K4b
// pivot pipelining, K6 forward row blocking and the back-pass
// PROHIBITION, K12 column-major storage, the K10e2 FMA discipline) are
// inherited by the tree routines below; see git history for the text.)

// ---- audit #10: TREE-SPARSE FACTOR AND SOLVES (algo.rs tc_*) -----------------
// The mass matrix's factor and every M^-1 solve now run in the fill-free
// tree elimination order (TreeTabs above; algo.rs header for the argument
// and SPEC-v10 Divergences §25). CANONICAL ARITHMETIC, identical to the
// Rust tc_* routines term for term (each sum in ascending index order, e
// the elimination position):
//   factor, column j (pivot first):
//     d_j  = M_jj - sum_{k in [lo_j, j)} L_jk * L_jk ; floor ; L_jj = sqrt
//     L_aj = (M_aj - sum_{k in [lo_j, j)} L_ak * L_jk) / L_jj, a in anc(j)
//   forward  y_e = (b_e - sum_{k in [lo_e, e)} L_ek * y_k) / L_ee, e up
//   back     x_e = (y_e - sum_{a in anc(e)} L_ae * x_a) / L_ee, e down
// = the dense Cholesky of P M P^T with its structural zeros (exact +0.0)
// skipped. What each kernel loop below adds on top is the usual set of
// bitwise-neutral transforms: RHS batching (a loop interchange, A1), row
// blocking of FORWARD rows that share a k-prefix (K6 — the base rows, all
// of which span every joint), and dropping terms whose y_k is a structural
// +0.0 (the chol_fwd_unit induction: the unit chains and the feet batch).
// The back pass is never row-blocked (K6 ABSOLUTE PROHIBITION, unchanged:
// the shared region is each row's TAIL).
//
// Loop indices are warp-uniform (they depend only on the tables), so the
// tables come from the constant bank (TREE_C); only the unit chains index
// by a per-lane joint and read the __device__ copy (TREE).
//
// FMA-contraction discipline (K10e2): every update stays ONE `X -= Y*Z`
// statement, contracted to one FFMA exactly like the dense solves were.

// Factor helper: the k-loop of column j feeding the pivot (PIV) and R
// consecutive ancestor rows t0..t0+R-1. Column k in [lo, j) holds L_jk at
// slot kb[k] - d[j] and L_{anc_t(j), k} at the next slots (its chain runs
// on through j's), so every k costs ONE table read and loads off ONE base
// with immediate offsets — K12's single-base column form.
template <int R, bool PIV, typename T>
__device__ __forceinline__ void tf_rows(const T* __restrict__ a, int lo, int j,
                                        int dcol, int cj, int t0, T& dpiv,
                                        T (&s)[4]) {
#pragma unroll
  for (int r = 0; r < R; r++) s[r] = a[cj + 1 + t0 + r];
  for (int k = lo; k < j; k++) {
    const int base = TREE_C.kb[k] - dcol;
    const T ljk = a[base];
    if (PIV) dpiv -= ljk * ljk;
#pragma unroll
    for (int r = 0; r < R; r++) s[r] -= a[base + 1 + t0 + r] * ljk;
  }
}

template <int R, typename T>
__device__ __forceinline__ void tf_store(T* __restrict__ a, int cj, int t0,
                                         T dj, const T (&s)[4]) {
#pragma unroll
  for (int r = 0; r < R; r++) a[cj + 1 + t0 + r] = s[r] / dj;
}

// The tree factor, IN PLACE on the profile d.m (crba wrote M there), with
// the §22 / §20e pivot floor verbatim on the new pivots (algo.rs header:
// what a pivot IS moves with the order; a healthy M never floors in
// either). Column j's ancestor rows go in blocks of 4 / 2 / 1 — d[j] is
// warp-uniform, so the ladder cannot diverge (K13's condition) — with the
// pivot riding the first block's k-loop (it needs the same L_jk loads).
// IN PLACE is legal for the dense form's reason: column j reads only
// columns k < j (final) and its own seeds, and writes only column j.
// MEASURED AND REJECTED (2026-09-26): a COMPILE-TIME form (column j a
// template parameter, every k-loop unrolled, every slot an immediate) —
// both with all of a column's rows jammed into one k-loop and with these
// same 4 / 2 / 1 blocks — took the float contact kernel from 48/48 to
// 332/356 B of spills (the unrolled straight-line factor, not the
// accumulator count: the two jam widths spilled identically) for a flux
// gain of −0.5 % (19.28 → 19.19 ms/tick, 2 interleaved rounds). A
// re-spilled build is a revert (K6); the factor runs once per substep.
template <typename T>
__device__ void tree_factor(T* __restrict__ a, bool* __restrict__ bad) {
  for (int j = 0; j < NV; j++) {
    const int lo = TREE_C.lo[j], dn = TREE_C.d[j], cj = TREE_C.cb[j];
    T dpiv = a[cj];
    T s[4];
    const int w = dn >= 4 ? 4 : dn >= 2 ? 2 : dn;
    if (w == 4)
      tf_rows<4, true>(a, lo, j, dn, cj, 0, dpiv, s);
    else if (w == 2)
      tf_rows<2, true>(a, lo, j, dn, cj, 0, dpiv, s);
    else if (w == 1)
      tf_rows<1, true>(a, lo, j, dn, cj, 0, dpiv, s);
    else
      tf_rows<0, true>(a, lo, j, dn, cj, 0, dpiv, s);
    if (sizeof(T) == 8) {
      if (dpiv < T(PD_FLOOR_F64)) *bad = true;
      dpiv = tmax(dpiv, T(PD_FLOOR_F64));
    } else {
      if (dpiv <= T(0)) *bad = true;
      dpiv = tmax(dpiv, T(PD_FLOOR_F32));
    }
    const T dj = sqrt(dpiv);
    a[cj] = dj;
    if (w == 4)
      tf_store<4>(a, cj, 0, dj, s);
    else if (w == 2)
      tf_store<2>(a, cj, 0, dj, s);
    else if (w == 1)
      tf_store<1>(a, cj, 0, dj, s);
    int t = w;
    for (; t + 4 <= dn; t += 4) {
      tf_rows<4, false>(a, lo, j, dn, cj, t, dpiv, s);
      tf_store<4>(a, cj, t, dj, s);
    }
    if (t + 2 <= dn) {
      tf_rows<2, false>(a, lo, j, dn, cj, t, dpiv, s);
      tf_store<2>(a, cj, t, dj, s);
      t += 2;
    }
    if (t < dn) {
      tf_rows<1, false>(a, lo, j, dn, cj, t, dpiv, s);
      tf_store<1>(a, cj, t, dj, s);
    }
  }
}

// FORWARD pass on W right-hand sides in v-order at x[w*SW + v*SV]
// (row-major batches: SW = NV, SV = 1; RHS-minor: SW = 1, SV = W).
// Joint rows are short profile sums (|desc| <= 16); the six base rows each
// span EVERY joint (lo = 0, tree_tabs_ok), so they run as row blocks of
// RB sharing the k in [0, NQ) prefix (K6), then the base-internal
// triangle in ascending k — each row's canonical order.
template <int W, int SW, int SV, int RB, typename T>
__device__ void tree_fwd_w(const T* __restrict__ l, T* __restrict__ x) {
  static_assert(VOFF % RB == 0, "base row blocks tile the base");
  for (int e = 0; e < TC_BASE_E; e++) {
    const int de = TREE_C.d[e], ve = TREE_C.e2v[e] * SV;
    T s[W];
#pragma unroll
    for (int w = 0; w < W; w++) s[w] = x[w * SW + ve];
    for (int k = TREE_C.lo[e]; k < e; k++) {
      const T lk = l[TREE_C.kb[k] - de];
      const int vk = TREE_C.e2v[k] * SV;
#pragma unroll
      for (int w = 0; w < W; w++) s[w] -= lk * x[w * SW + vk];
    }
    const T dg = l[TREE_C.cb[e]];
#pragma unroll
    for (int w = 0; w < W; w++) x[w * SW + ve] = s[w] / dg;
  }
#pragma unroll
  for (int b0 = 0; b0 < VOFF; b0 += RB) {
    // rows e = TC_BASE_E + b0 + r hold v = b0 + r; d[e] = VOFF - 1 - b
    T s[RB][W];
#pragma unroll
    for (int r = 0; r < RB; r++)
#pragma unroll
      for (int w = 0; w < W; w++) s[r][w] = x[w * SW + (b0 + r) * SV];
    for (int k = 0; k < TC_BASE_E; k++) {  // shared prefix: every joint
      const int kb = TREE_C.kb[k], vk = TREE_C.e2v[k] * SV;
#pragma unroll
      for (int r = 0; r < RB; r++) {
        const T lk = l[kb - (VOFF - 1 - (b0 + r))];
#pragma unroll
        for (int w = 0; w < W; w++) s[r][w] -= lk * x[w * SW + vk];
      }
    }
#pragma unroll
    for (int bk = 0; bk < b0; bk++) {  // earlier base rows, shared
      const int cbk = TREE.cb[TC_BASE_E + bk];
#pragma unroll
      for (int r = 0; r < RB; r++) {
        const T lk = l[cbk + (b0 + r - bk)];
#pragma unroll
        for (int w = 0; w < W; w++) s[r][w] -= lk * x[w * SW + bk * SV];
      }
    }
#pragma unroll
    for (int r = 0; r < RB; r++) {  // the block's own triangle
#pragma unroll
      for (int rr = 0; rr < r; rr++) {
        const T lk = l[TREE.cb[TC_BASE_E + b0 + rr] + (r - rr)];
#pragma unroll
        for (int w = 0; w < W; w++)
          s[r][w] -= lk * x[w * SW + (b0 + rr) * SV];
      }
      const T dg = l[TREE.cb[TC_BASE_E + b0 + r]];
#pragma unroll
      for (int w = 0; w < W; w++) x[w * SW + (b0 + r) * SV] = s[r][w] / dg;
    }
  }
}

// BACK pass on the same layouts: e descending, each column's chain
// ascending (slots 1..d[e] of column e, row v = vslot). Not row-blocked.
template <int W, int SW, int SV, typename T>
__device__ void tree_back_w(const T* __restrict__ l, T* __restrict__ x) {
  for (int e = NV - 1; e >= 0; e--) {
    const int c = TREE_C.cb[e], de = TREE_C.d[e], ve = TREE_C.e2v[e] * SV;
    T s[W];
#pragma unroll
    for (int w = 0; w < W; w++) s[w] = x[w * SW + ve];
    for (int t = 1; t <= de; t++) {
      const T la = l[c + t];
      const int va = TREE_C.vslot[c + t] * SV;
#pragma unroll
      for (int w = 0; w < W; w++) s[w] -= la * x[w * SW + va];
    }
    const T dg = l[c];
#pragma unroll
    for (int w = 0; w < W; w++) x[w * SW + ve] = s[w] / dg;
  }
}

// Dense-RHS solves: one vector (the step-6 rhs, a lone contact column)
// and W row-major columns (the contact batches).
template <typename T>
__device__ __forceinline__ void tree_solve_n(const T* __restrict__ l,
                                             T* __restrict__ x) {
  tree_fwd_w<1, 0, 1, 3>(l, x);
  tree_back_w<1, 0, 1>(l, x);
}
template <int W, typename T>
__device__ __forceinline__ void tree_solve_multi_n(const T* __restrict__ l,
                                                   T* __restrict__ x) {
  tree_fwd_w<W, NV, 1, (W >= 4 ? 2 : 3)>(l, x);
  tree_back_w<W, NV, 1>(l, x);
}

// The 12 FOOT columns, RHS-minor x[v*12 + m] (s.g_minv): column m < 6 is a left-foot
// row (support: base + v 23..28 = e 6..11), m >= 6 a right-foot row (base +
// v 29..34 = e 0..5). The forward pass visits only that support's upward
// closure — each leg's rows on its own six lanes, then the base rows with
// each lane group's own leg prefix — so every dropped term multiplies a
// structural +0.0 (the other leg's rows and the arms/spine rows of a foot
// column stay at their +0.0 seed: nothing below them in the tree is
// non-zero). algo.rs `tc_fwd_mask_t` computes the same rows with the
// union mask; the extra terms it keeps are (+-0.0)-products, exact no-ops.
// Every index here is a compile-time constant.
template <typename T>
__device__ __forceinline__ void tree_fwd_leg_tn(const T* __restrict__ l,
                                                T* __restrict__ x, int e0,
                                                int m0) {
  constexpr int W = 2 * FOOT_ROWS;
#pragma unroll
  for (int e = e0; e < e0 + 6; e++) {
    const int de = TREE.d[e], ve = TREE.e2v[e] * W + m0;
    T s[6];
#pragma unroll
    for (int w = 0; w < 6; w++) s[w] = x[ve + w];
#pragma unroll
    for (int k = e0; k < e; k++) {  // lo[e] = e0 on a leg (tree_tabs_ok)
      const T lk = l[TREE.kb[k] - de];
      const int vk = TREE.e2v[k] * W + m0;
#pragma unroll
      for (int w = 0; w < 6; w++) s[w] -= lk * x[vk + w];
    }
    const T dg = l[TREE.cb[e]];
#pragma unroll
    for (int w = 0; w < 6; w++) x[ve + w] = s[w] / dg;
  }
}
template <typename T>
__device__ void tree_solve_feet_tn(const T* __restrict__ l, T* __restrict__ x) {
  constexpr int W = 2 * FOOT_ROWS;
  static_assert(W == 2 * 6, "two feet x six spatial rows");
  tree_fwd_leg_tn(l, x, 6, 0);  // left leg rows, left-foot lanes 0..5
  tree_fwd_leg_tn(l, x, 0, 6);  // right leg rows, right-foot lanes 6..11
#pragma unroll
  for (int b = 0; b < VOFF; b++) {
    const int e = TC_BASE_E + b, de = VOFF - 1 - b;
    T s[W];
#pragma unroll
    for (int w = 0; w < W; w++) s[w] = x[b * W + w];
#pragma unroll
    for (int k = 0; k < 6; k++) {  // right leg (e 0..5): lanes 6..11
      const T lk = l[TREE.kb[k] - de];
      const int vk = TREE.e2v[k] * W;
#pragma unroll
      for (int w = 6; w < W; w++) s[w] -= lk * x[vk + w];
    }
#pragma unroll
    for (int k = 6; k < 12; k++) {  // left leg (e 6..11): lanes 0..5
      const T lk = l[TREE.kb[k] - de];
      const int vk = TREE.e2v[k] * W;
#pragma unroll
      for (int w = 0; w < 6; w++) s[w] -= lk * x[vk + w];
    }
#pragma unroll
    for (int bk = 0; bk < b; bk++) {  // earlier base rows, all lanes
      const T lk = l[TREE.cb[TC_BASE_E + bk] + (b - bk)];
#pragma unroll
      for (int w = 0; w < W; w++) s[w] -= lk * x[bk * W + w];
    }
    const T dg = l[TREE.cb[e]];
#pragma unroll
    for (int w = 0; w < W; w++) x[b * W + w] = s[w] / dg;
  }
  tree_back_w<W, 1, W>(l, x);
}

// UNIT-RHS forward pass for M^-1 e_vm (x: +0.0 everywhere, as ecol_seed
// leaves it): only vm's elimination chain {e0} u anc(e0) — at most 16
// rows (a wrist: 7 arm + 3 spine + 6 base) — can become non-zero, and
// each chain row sums exactly the chain rows below it, ascending (algo.rs
// tc_fwd_unit_lane, term for term: the +0.0 seed, the e0 term, then the
// ancestors). L(a, k) for the chain's r-th and s-th members sits at slot
// s - r of column k, whose base is cbslot[] of the chain slot — every
// table read is independent (no F_PARENT-style chase, K17). The joint is
// per LANE here, so the reads go to the __device__ copy.
template <typename T>
__device__ void tree_fwd_unit_n(const T* __restrict__ l, T* __restrict__ x,
                                int vm) {
  const int e0 = TREE.v2e[vm];
  const int c0 = TREE.cb[e0], len = TREE.d[e0];
  const T y0 = T(1) / l[c0];
  x[vm] = y0;
  for (int s = 1; s <= len; s++) {
    T acc = T(0);
    acc -= l[c0 + s] * y0;
    for (int r = 1; r < s; r++)
      acc -= l[TREE.cbslot[c0 + r] + (s - r)] * x[TREE.vslot[c0 + r]];
    x[TREE.vslot[c0 + s]] = acc / l[TREE.cbslot[c0 + s]];
  }
}

// The same unit forward pass for B columns at once (the limit-row group),
// INTERLEAVED so the B chains are independent accumulators in one loop —
// the ILP chol_fwd_unit4_n bought for the dense form (K4a/K6). Chain row s
// of every column sums r = 0..s-1 (a bound shared by all B), so the only
// per-column difference is the chain length: a column whose chain ended is
// predicated off (loads included — its slots past the chain are another
// column's). Each column's values are bitwise tree_fwd_unit_n's.
template <int B, typename T>
__device__ void tree_fwd_unit_b(const T* __restrict__ l, T* const (&x)[B],
                                const int (&vm)[B]) {
  int c0[B], len[B];
  T y0[B];
  int lmax = 0;
#pragma unroll
  for (int b = 0; b < B; b++) {
    const int e0 = TREE.v2e[vm[b]];
    c0[b] = TREE.cb[e0];
    len[b] = TREE.d[e0];
    y0[b] = T(1) / l[c0[b]];
    x[b][vm[b]] = y0[b];
    lmax = len[b] > lmax ? len[b] : lmax;
  }
  for (int s = 1; s <= lmax; s++) {
    T acc[B];
#pragma unroll
    for (int b = 0; b < B; b++) {
      acc[b] = T(0);
      if (s <= len[b]) acc[b] -= l[c0[b] + s] * y0[b];
    }
    for (int r = 1; r < s; r++) {
#pragma unroll
      for (int b = 0; b < B; b++)
        if (s <= len[b])
          acc[b] -= l[TREE.cbslot[c0[b] + r] + (s - r)]
                    * x[b][TREE.vslot[c0[b] + r]];
    }
#pragma unroll
    for (int b = 0; b < B; b++)
      if (s <= len[b])
        x[b][TREE.vslot[c0[b] + s]] = acc[b] / l[TREE.cbslot[c0[b] + s]];
  }
}

// BACK pass for 4 / 2 columns that are not contiguous (s.ecol rows of
// non-consecutive joints): the tree_back_w loop over explicit pointers —
// the RHS-batching interchange, bitwise per column.
template <typename T>
__device__ void tree_back4_n(const T* __restrict__ l, T* __restrict__ x0,
                             T* __restrict__ x1, T* __restrict__ x2,
                             T* __restrict__ x3) {
  for (int e = NV - 1; e >= 0; e--) {
    const int c = TREE_C.cb[e], de = TREE_C.d[e], ve = TREE_C.e2v[e];
    T s0 = x0[ve], s1 = x1[ve], s2 = x2[ve], s3 = x3[ve];
    for (int t = 1; t <= de; t++) {
      const T la = l[c + t];
      const int va = TREE_C.vslot[c + t];
      s0 -= la * x0[va];
      s1 -= la * x1[va];
      s2 -= la * x2[va];
      s3 -= la * x3[va];
    }
    const T dg = l[c];
    x0[ve] = s0 / dg;
    x1[ve] = s1 / dg;
    x2[ve] = s2 / dg;
    x3[ve] = s3 / dg;
  }
}
template <typename T>
__device__ void tree_back2_n(const T* __restrict__ l, T* __restrict__ x0,
                             T* __restrict__ x1) {
  for (int e = NV - 1; e >= 0; e--) {
    const int c = TREE_C.cb[e], de = TREE_C.d[e], ve = TREE_C.e2v[e];
    T s0 = x0[ve], s1 = x1[ve];
    for (int t = 1; t <= de; t++) {
      const T la = l[c + t];
      const int va = TREE_C.vslot[c + t];
      s0 -= la * x0[va];
      s1 -= la * x1[va];
    }
    const T dg = l[c];
    x0[ve] = s0 / dg;
    x1[ve] = s1 / dg;
  }
}
template <typename T>
__device__ __forceinline__ void tree_solve_unit_n(const T* __restrict__ l,
                                                  T* __restrict__ x, int vm) {
  tree_fwd_unit_n(l, x, vm);
  tree_back_w<1, 0, 1>(l, x);
}

// ---- collide.rs seg_seg_closest (fixed branch order) ------------------------
template <typename T>
__device__ void seg_seg_closest(const T p1[3], const T q1[3], const T p2[3],
                                const T q2[3], T c1[3], T c2[3], T* d2out) {
  T d1[3], d2[3], r[3];
  sub3v(q1, p1, d1);
  sub3v(q2, p2, d2);
  sub3v(p1, p2, r);
  const T a = dot3(d1, d1);
  const T e = dot3(d2, d2);
  const T f = dot3(d2, r);
  T s, t;
  if (a <= T(SEG_EPS) && e <= T(SEG_EPS)) {
    s = T(0);
    t = T(0);
  } else if (a <= T(SEG_EPS)) {
    s = T(0);
    t = tclamp(f / e, T(0), T(1));
  } else {
    const T c = dot3(d1, r);
    if (e <= T(SEG_EPS)) {
      t = T(0);
      s = tclamp(-c / a, T(0), T(1));
    } else {
      const T b = dot3(d1, d2);
      const T denom = a * e - b * b;
      const T s0 = denom > T(SEG_EPS)
                       ? tclamp((b * f - c * e) / denom, T(0), T(1))
                       : T(0);
      const T t0 = b * s0 + f;
      if (t0 < T(0)) {
        t = T(0);
        s = tclamp(-c / a, T(0), T(1));
      } else if (t0 > e) {
        t = T(1);
        s = tclamp((b - c) / a, T(0), T(1));
      } else {
        t = t0 / e;
        s = s0;
      }
    }
  }
  for (int i = 0; i < 3; i++) {
    c1[i] = p1[i] + d1[i] * s;
    c2[i] = p2[i] + d2[i] * t;
  }
  T dd[3];
  sub3v(c1, c2, dd);
  *d2out = dot3(dd, dd);
}

// ---- solver.rs jacobians ----------------------------------------------------
template <typename T>
// row is the only written pointer and never overlaps d/p/n (rows live
// in s.jw / s.c_J, p in d.pos or s.c_pt); d and p are left unqualified
// on purpose — one call site passes p = d.pos[frame], a read-only
// alias.
__device__ void point_jacobian_acc(const Dyn<T>& d, int frame, const T p[3],
                                   const T n[3], T sign, T* __restrict__ row) {
  int k = frame;
  while (k >= 0) {
    if (F_COORD[k] >= 0) {
      const int col = VOFF + F_COORD[k];
      T rel[3], w[3];
      sub3v(p, d.pos[k], rel);
      cross3(d.axw[k], rel, w);
      row[col] += sign * dot3(n, w);
    }
    k = F_PARENT[k];
  }
  for (int t = 0; t < 3; t++) row[t] += sign * n[t];
  T rel0[3], rxn[3];
  sub3v(p, d.p, rel0);
  cross3(rel0, n, rxn);
  for (int t = 0; t < 3; t++) row[3 + t] += sign * rxn[t];
}

// (angular_jacobian_acc lived here; its body — one ancestor walk
// writing row[VOFF + F_COORD[k]] += sign*axw[k][t] plus row[3+t] +=
// sign — is folded into the foot-row walk of the ground block in
// constraint_solve. point_jacobian_acc survives for contacts.)

// RHS seed: the full-length zero fill is load-bearing for
// chol_fwd_unit_n / chol_back_n — see chol_solve_unit_n's header (the
// skipped forward rows keep these +0.0 for the dense back substitution
// to read).
template <typename T>
__device__ __forceinline__ void ecol_seed(T* __restrict__ col, int j) {
  for (int i = 0; i < NV; i++) col[i] = T(0);
  col[VOFF + j] = T(1);
}

// joint j's computed plain column (its slot — see the Solver note)
template <typename T>
__device__ __forceinline__ T* ecol_of(Solver<T>& s, int j) {
  return s.ecol[s.eslot[j]];
}

// Lazily compute the plain M^-1 e_j column (shared by step-8 cap/limit
// rows and the step-11 clamps — same expression tree as Rust's three
// separate caches) into the next free slot.
template <typename T>
__device__ void ensure_ecol(Dyn<T>& __restrict__ d,
                            Solver<T>& __restrict__ s, int j) {
  if (s.ecomp[j]) return;
  s.eslot[j] = uint8_t(s.nslot++);
  s.emask |= 1u << j;
  T* const col = ecol_of(s, j);
  ecol_seed(col, j);
  tree_solve_unit_n(d.m, col, VOFF + j);
  s.ecomp[j] = true;
}

// Lazily compute joint j's plain column and its mobility verdict
// (solver.rs capcol path; also the limit-row diagonal). The direction
// is the plain column s.ecol[j] and the mobility is `plain`; the
// COMPOSITE_EPS verdict is kept verbatim rather than assumed: Rust
// still evaluates `dnew > COMPOSITE_EPS * plain` with dnew == plain,
// which is true for every positive mobility and false for a degenerate
// one (0, NaN, +inf).
template <typename T>
__device__ void ensure_pdiag(Dyn<T>& __restrict__ d,
                             Solver<T>& __restrict__ s, int j) {
  if (s.pcomp[j]) return;
  ensure_ecol(d, s, j);
  const T plain = ecol_of(s, j)[VOFF + j];
  s.pdiag[j] = plain > T(COMPOSITE_EPS) * plain
                   ? plain
                   : std::numeric_limits<T>::infinity();
  s.pcomp[j] = true;
}

// OUT-OF-LINE lazy column builds for the register-resident cap pass and
// step-11 clamps (#6c): those loops are unrolled over the 29 joints, and
// an inlined tree solve per joint would be 29 copies of it. The ecomp /
// pcomp early-out stays inline at the call site, so the call (and the
// register save / restore around it) is paid only on a genuine miss — a
// few per substep against dozens of firings. Measured alternatives
// (trained standard, sampled, N=32768): dropping to the old
// local-memory loop at the first miss, 32.0 ms/tick vs 29.9 here; a
// "restart" form (leave the unrolled sweep at the miss, build inline
// once, resume at that joint) wins without the #6e column prediction
// (25.7) but loses with it (23.8 vs 23.5), and in step 11 either way.
// ptxas: this costs f32 +60 B / +124 B of spill stores (standard /
// Flux), f64 about +1.2 KB — static counts; the timings above are net.
template <typename T>
__device__ __noinline__ void ensure_ecol_ool(Dyn<T>& __restrict__ d,
                                             Solver<T>& __restrict__ s,
                                             int j) {
  ensure_ecol(d, s, j);
}
template <typename T>
__device__ __noinline__ void ensure_pdiag_ool(Dyn<T>& __restrict__ d,
                                              Solver<T>& __restrict__ s,
                                              int j) {
  ensure_pdiag(d, s, j);
}

// audit #9 — support dot of foot spatial row m (0..2 linear, 3..5
// angular; the support shapes at WJ_L0) against a dense NV-vector read
// through g(i): the exactly-1.0 column m as a bare add, a linear row's two
// root values at WJ_R2[m], then the six chain columns ch0..ch0+5 —
// ascending column order, which is solver.rs `ground_support_dot` over
// the dense row. Callers pass compile-time m / ch0 (unrolled loops) so
// every column index folds to an immediate.
template <typename T, typename G>
__device__ __forceinline__ T foot_row_dot(int m, int ch0, const T jr[3][2],
                                          const T jc[6], G g) {
  T sum = T(0);
  sum += g(m);
  if (m < 3) {
    sum += jr[m][0] * g(WJ_R2[m][0]);
    sum += jr[m][1] * g(WJ_R2[m][1]);
  }
#pragma unroll
  for (int e = 0; e < 6; e++) sum += jc[e] * g(ch0 + e);
  return sum;
}

// ---- solver.rs ConstraintSet::solve (steps 7-9) -----------------------------
template <typename T>
__device__ void constraint_solve(Dyn<T>& __restrict__ d,
                                 Solver<T>& __restrict__ s, const T* qpre,
                                 T* __restrict__ v) {
  const T hinv = T(1) / T(H_SUB);

  // -- env v11 GROUND BLOCK (rows ahead of the pair rows) ---------------------
  // MIRROR of collide.rs `ground_phase` + solver.rs `assemble_ground_rows`.
  // The corner order is the spec §2 normative one (foot_l then foot_r,
  // GROUND_CORNERS order within a foot), the row order the §3 one (normal,
  // friction x, friction z). Rust packs the ACTIVE corners' rows from 0;
  // G1 keeps every corner in its FIXED slot (3*c + t) with an active bit
  // instead — the consumers walk the slots in ascending order and skip
  // inactive ones, which visits exactly Rust's rows in Rust's order.
  //
  // audit #9: with any corner down, BOTH feet's six spatial rows are
  // built (Rust builds only the contacting feet; the extra foot's columns
  // are independent RHS of the same solve — A1 — so no active value
  // moves) straight into g_minv as the RHS-minor seed: ONE ancestor walk
  // per foot emits all six rows (rel = pf - pos[k] and w = axw[k] x rel
  // shared; dot3(e_t, w) is w[t] exactly), every column a single
  // `T(0) + x` store — `T(0) + x` is NOT the identity when x is -0.0,
  // and that is the point: it reproduces the zero fill's +0.0
  // absorption bit for bit (F_COORD is injective on the chain and the
  // root columns are disjoint from it — wj_chain_mask), so the stores
  // are what `fill(+0.0); row[col] += x` wrote. The root columns use
  // cross(rel0, e_t) in closed form; the support is teed off into g_fjc
  // / g_fjr. Every corner slot also gets its coefficient pair g_coef.
  {
    s.g_act = 0;
    T pt[NUM_GROUND_CORNERS][3];
#pragma unroll
    for (int f = 0; f < 2; f++) {
      const int frame = GROUND_FRAMES[f];
      uint32_t fact = 0;
#pragma unroll
      for (int k = 0; k < CORNERS_PER_FOOT; k++) {
        const int c = CORNERS_PER_FOOT * f + k;
        // world corner = p[frame] + R[frame] * corner_k  (fk.rs
        // add3(ws.p, mat_vec(ws.r, .)), same term order as the capsule
        // endpoints hoisted below)
        for (int i = 0; i < 3; i++)
          pt[c][i] = d.pos[frame][i]
                     + (d.r[frame][i][0] * CT(GROUND_CORNERS, [k][0])
                        + d.r[frame][i][1] * CT(GROUND_CORNERS, [k][1])
                        + d.r[frame][i][2] * CT(GROUND_CORNERS, [k][2]));
        // the plane is y = 0 and its normal is +y, so the signed gap is
        // the corner's world y outright
        const T gap = pt[c][1];
        const int r0 = 3 * c;
        // NORMAL row rhs: the §3-step-8 contact pipeline verbatim —
        // speculative approach limit while the gap is open, Baumgarte
        // push-out beyond SLOP once penetrating
        s.g_bias[r0] = gap >= T(0)
                           ? -gap * hinv
                           : (T(BETA_CONTACT) * hinv)
                                 * tmax(-gap - T(SLOP), T(0));
        // FRICTION rows rhs = 0: friction stabilizes nothing, so it
        // gets no Baumgarte term (contact spec §3)
        s.g_bias[r0 + 1] = T(0);
        s.g_bias[r0 + 2] = T(0);
        if (gap < T(MARGIN)) fact |= 1u << k;
      }
      s.g_act |= fact << (CORNERS_PER_FOOT * f);
    }
    if (s.g_act != 0) {
      T(*x)[2 * FOOT_ROWS] = s.g_minv;
      for (int i = 0; i < NV; i++)
#pragma unroll
        for (int m = 0; m < 2 * FOOT_ROWS; m++) x[i][m] = T(0);
#pragma unroll
      for (int f = 0; f < 2; f++) {
        const int frame = GROUND_FRAMES[f];
        const T* pf = d.pos[frame];
        const int r0 = FOOT_ROWS * f;
        for (int kk = frame; kk >= 0; kk = F_PARENT[kk]) {
          if (F_COORD[kk] < 0) continue;
          const int col = VOFF + F_COORD[kk];
          const int e = col - (WJ_L0 + 6 * f);
          T rel[3], w[3];
          sub3v(pf, d.pos[kk], rel);
          cross3(d.axw[kk], rel, w);
#pragma unroll
          for (int t = 0; t < 3; t++) {
            const T lv = T(0) + w[t];
            x[col][r0 + t] = lv;
            s.g_fjc[f][t][e] = lv;
            const T av = T(0) + d.axw[kk][t];
            x[col][r0 + 3 + t] = av;
            s.g_fjc[f][3 + t][e] = av;
          }
        }
        T rel0[3];
        sub3v(pf, d.p, rel0);
        const T v04 = T(0) + rel0[2], v05 = T(0) - rel0[1];
        const T v13 = T(0) - rel0[2], v15 = T(0) + rel0[0];
        const T v23 = T(0) + rel0[1], v24 = T(0) - rel0[0];
        x[0][r0 + 0] = T(1);
        x[4][r0 + 0] = v04;
        x[5][r0 + 0] = v05;
        x[1][r0 + 1] = T(1);
        x[3][r0 + 1] = v13;
        x[5][r0 + 1] = v15;
        x[2][r0 + 2] = T(1);
        x[3][r0 + 2] = v23;
        x[4][r0 + 2] = v24;
        s.g_fjr[f][0][0] = v04;  // cols WJ_R2[0] = {4, 5}
        s.g_fjr[f][0][1] = v05;
        s.g_fjr[f][1][0] = v13;  // cols WJ_R2[1] = {3, 5}
        s.g_fjr[f][1][1] = v15;
        s.g_fjr[f][2][0] = v23;  // cols WJ_R2[2] = {3, 4}
        s.g_fjr[f][2][1] = v24;
#pragma unroll
        for (int t = 0; t < 3; t++) x[3 + t][r0 + 3 + t] = T(1);
        // corner levers r = p_corner - p_foot and the GROUND_SPATIAL
        // coefficient pairs (solver.rs assemble_ground_rows, verbatim)
#pragma unroll
        for (int k = 0; k < CORNERS_PER_FOOT; k++) {
          const int r = 3 * (CORNERS_PER_FOOT * f + k);
          T lv[3];
          sub3v(pt[CORNERS_PER_FOOT * f + k], pf, lv);
          s.g_coef[r][0] = -lv[2];  // normal y: w_x, w_z
          s.g_coef[r][1] = lv[0];
          s.g_coef[r + 1][0] = lv[2];  // friction x: w_y, w_z
          s.g_coef[r + 1][1] = -lv[1];
          s.g_coef[r + 2][0] = lv[1];  // friction z: w_x, w_y
          s.g_coef[r + 2][1] = -lv[0];
        }
      }
    }
  }

  // -- step 7: narrow phase + contact rows (pair-id order) --------------------
  // The 55 pairs reference only NUM_COLLIDERS distinct capsules, so the
  // world-space endpoints are hoisted here: 24 transforms instead of 110.
  // Expression structure is identical to the old inline form, so every
  // endpoint is bit-identical to what the pair loop used to compute.
  // BROAD PHASE (K4d): mid[c] and HLR[c] = half_len + radius bound each
  // capsule by a sphere; seg_dist >= |mid_a-mid_b| - half_a - half_b, so
  // the `gap < MARGIN` branch provably cannot fire once
  // |mid_a-mid_b| >= HLR_a + HLR_b + MARGIN. 2% slack on the squared
  // threshold (~1% linear) vs ~1e-6 f32 error. Retained pairs run
  // byte-identical code.
  T ca[NUM_COLLIDERS][3], cb[NUM_COLLIDERS][3], mid[NUM_COLLIDERS][3];
  T hlr[NUM_COLLIDERS];
  for (int c = 0; c < NUM_COLLIDERS; c++) {
    const int f = COL_FRAME[c];
    for (int i = 0; i < 3; i++) {
      ca[c][i] = d.pos[f][i] + (d.r[f][i][0] * CT(COL_A, [c][0])
                                + d.r[f][i][1] * CT(COL_A, [c][1])
                                + d.r[f][i][2] * CT(COL_A, [c][2]));
      cb[c][i] = d.pos[f][i] + (d.r[f][i][0] * CT(COL_B, [c][0])
                                + d.r[f][i][1] * CT(COL_B, [c][1])
                                + d.r[f][i][2] * CT(COL_B, [c][2]));
      mid[c][i] = (ca[c][i] + cb[c][i]) * T(0.5);
    }
    const double ex = COL_B[c][0] - COL_A[c][0];
    const double ey = COL_B[c][1] - COL_A[c][1];
    const double ez = COL_B[c][2] - COL_A[c][2];
    hlr[c] = T(0.5 * sqrt(ex * ex + ey * ey + ez * ez) + COL_R[c]);
  }
  s.nc = 0;
  for (int pid = 0; pid < NUM_PAIRS; pid++) {
    const int ia = PAIR_A[pid], ib = PAIR_B[pid];
    {
      const T mx = mid[ia][0] - mid[ib][0];
      const T my = mid[ia][1] - mid[ib][1];
      const T mz = mid[ia][2] - mid[ib][2];
      const T lim = hlr[ia] + hlr[ib] + T(MARGIN);
      if (mx * mx + my * my + mz * mz > lim * lim * T(1.02)) continue;
    }
    const int fa = COL_FRAME[ia], fbf = COL_FRAME[ib];
    T c1[3], c2[3], dist2;
    seg_seg_closest(ca[ia], cb[ia], ca[ib], cb[ib], c1, c2, &dist2);
    const T dist = sqrt(dist2);
    const T rsum = std::is_same<T, float>::value ? T(PAIR_RSUM_F.v[pid])
                                                 : T(PAIR_RSUM.v[pid]);
    const T gap = dist - rsum;
    if (gap < T(MARGIN)) {
      T n[3];
      if (dist > T(1e-9)) {
        const T inv = T(1) / dist;
        for (int i = 0; i < 3; i++) n[i] = (c2[i] - c1[i]) * inv;
      } else {
        n[0] = T(0);
        n[1] = T(1);
        n[2] = T(0);
      }
      const int ci = s.nc;
      s.c_pair[ci] = pid;
      for (int i = 0; i < 3; i++) {
        const T s1 = c1[i] + n[i] * CT(COL_R, [ia]);
        const T s2 = c2[i] - n[i] * CT(COL_R, [ib]);
        s.c_pt[ci][i] = (s1 + s2) * T(0.5);
      }
      T* jr = s.c_J[ci];
      for (int i = 0; i < NV; i++) jr[i] = T(0);
      point_jacobian_acc(d, fa, s.c_pt[ci], n, T(-1), jr);
      point_jacobian_acc(d, fbf, s.c_pt[ci], n, T(1), jr);
      s.c_bias[ci] = gap >= T(0)
                         ? -gap * hinv
                         : (T(BETA_CONTACT) * hinv) * tmax(-gap - T(SLOP),
                                                           T(0));
      s.nc = ci + 1;
    }
  }

  // -- assembly: joint-limit rows, joint order (Divergences §15) --------------
  s.nl = 0;
  for (int j = 0; j < NQ; j++) {
    const T gl = qpre[j] - CT(LIMIT_LO, [j]);
    const T gh = CT(LIMIT_HI, [j]) - qpre[j];
    const T reach = T(H_SUB) * CT(OMEGA_CAP, [j]);
    T sign, gap;
    if (gl <= gh) {
      sign = T(1);
      gap = gl;
    } else {
      sign = T(-1);
      gap = gh;
    }
    if (gap < reach) {
      const int li = s.nl;
      s.l_j[li] = j;
      s.l_sign[li] = sign;
      s.l_bias[li] =
          gap >= T(0) ? -gap * hinv : (T(BETA_LIMIT_ROW) * hinv) * -gap;
      s.nl = li + 1;
    }
  }

  // -- effective masses (row order: ground, contacts, limits) -----------------
  // Ground rows, audit #9: ONE tree_solve_feet_tn (12 RHS-minor columns)
  // over both feet's spatial rows (the RHS-minor seed the assembly
  // wrote), then per foot
  //   Phi[r][c] = Phi[c][r] = J_f[c] . G_f[:, r]   (c >= r, foot_row_dot)
  //   u[i]      = (Phi[i][a] + c1 * Phi[i][3+b1]) + c2 * Phi[i][3+b2]
  //   diag      = (u[a] + c1 * u[3+b1]) + c2 * u[3+b2]
  // for every corner slot of the foot (solver.rs `prep_ground_foot`; an
  // inactive slot's values are never read). Every batched / blocked form
  // of the solve is a pure loop interchange with each RHS keeping its own
  // ascending accumulator chains (A1), so each G_f column is bit-for-bit
  // what Rust's per-foot batch produces.
  {
    if (s.g_act != 0) {
      tree_solve_feet_tn(d.m, &s.g_minv[0][0]);
      const T(*x)[2 * FOOT_ROWS] = s.g_minv;
#pragma unroll
      for (int f = 0; f < 2; f++) {
        const int ch0 = WJ_L0 + 6 * f;
        T phi[FOOT_ROWS][FOOT_ROWS];
#pragma unroll
        for (int r = 0; r < FOOT_ROWS; r++)
#pragma unroll
          for (int c = r; c < FOOT_ROWS; c++) {
            const T sum = foot_row_dot(
                c, ch0, s.g_fjr[f], s.g_fjc[f][c],
                [&](int i) { return x[i][FOOT_ROWS * f + r]; });
            phi[r][c] = sum;
            phi[c][r] = sum;
          }
#pragma unroll
        for (int k = 0; k < CORNERS_PER_FOOT; k++) {
#pragma unroll
          for (int t = 0; t < GROUND_ROWS_PER_CORNER; t++) {
            const int g = 3 * (CORNERS_PER_FOOT * f + k) + t;
            const int a = GROUND_SPATIAL[t][0];
            const int b1 = 3 + GROUND_SPATIAL[t][1];
            const int b2 = 3 + GROUND_SPATIAL[t][2];
            const T c1 = s.g_coef[g][0], c2 = s.g_coef[g][1];
            T u[FOOT_ROWS];
#pragma unroll
            for (int i = 0; i < FOOT_ROWS; i++) {
              u[i] = phi[i][a] + c1 * phi[i][b1] + c2 * phi[i][b2];
              s.g_u[g][i] = u[i];
            }
            s.g_diag[g] = u[a] + c1 * u[b1] + c2 * u[b2];
          }
        }
      }
    }
  }
  // Contact rows: s.nc is RUNTIME (0..55, typically a handful), so the
  // batch runs as a 4 / 2 / 1 ladder — the 2-wide tier is worth its own
  // branch here (+0.4% on train --bench over 4-with-scalar-remainder),
  // which says the common nc is small. Rows are mutually
  // independent (c_minv[ci] depends only on c_J[ci] and the factor,
  // c_diag[ci] only on those two), so grouping the copies / solves /
  // diagonals changes no value; the solve itself is a bitwise
  // multi-RHS loop interchange (A1).
  {
    int ci = 0;
    for (; ci + 3 < s.nc; ci += 4) {
      for (int t = 0; t < 4; t++)
        for (int i = 0; i < NV; i++) s.c_minv[ci + t][i] = s.c_J[ci + t][i];
      tree_solve_multi_n<4>(d.m, &s.c_minv[ci][0]);
      for (int t = 0; t < 4; t++) {
        T diag = T(0);
        for (int i = 0; i < NV; i++)
          diag += s.c_J[ci + t][i] * s.c_minv[ci + t][i];
        s.c_diag[ci + t] = diag;
      }
    }
    if (ci + 1 < s.nc) {
      for (int t = 0; t < 2; t++)
        for (int i = 0; i < NV; i++) s.c_minv[ci + t][i] = s.c_J[ci + t][i];
      tree_solve_multi_n<2>(d.m, &s.c_minv[ci][0]);
      for (int t = 0; t < 2; t++) {
        T diag = T(0);
        for (int i = 0; i < NV; i++)
          diag += s.c_J[ci + t][i] * s.c_minv[ci + t][i];
        s.c_diag[ci + t] = diag;
      }
      ci += 2;
    }
    for (; ci < s.nc; ci++) {
      for (int i = 0; i < NV; i++) s.c_minv[ci][i] = s.c_J[ci][i];
      tree_solve_n(d.m, s.c_minv[ci]);
      T diag = T(0);
      for (int i = 0; i < NV; i++) diag += s.c_J[ci][i] * s.c_minv[ci][i];
      s.c_diag[ci] = diag;
    }
  }
  // limit rows' plain columns = M^-1 e_j (the +-1 row sign commutes
  // exactly through the solve/projection — computed unsigned, applied
  // signed; bitwise identical to Rust's signed-row solves)
  // BATCHED: this is the first ensure_ecol of the substep (substep
  // clears ecomp right before calling in), so all s.nl columns are
  // genuinely uncomputed and adjacent — the only place in the kernel
  // where a group of unit solves is known up front. Forward halves stay
  // per-column (each has its own m = VOFF + j); the dense back halves
  // run 4 / 2 at a time. s.l_j is strictly ascending and its entries are
  // distinct joints, so the batched pointers never alias.
  // Width: s.nl measures mean 7.8, range 3-14 (f32 trace, 256 envs), so
  // an 8-wide tier was tried too and won NOTHING over 4 (7,283 -> 7,286
  // steps/s) for ~30 more lines — dropped. See the block-size note in
  // the header: the gain saturates at four independent chains.
  // #6a: row li's column goes to SLOT li (see the Solver note), so the
  // seed stores and every back-pass access are warp-uniform words; the
  // scalar tail's ensure_ecol hands out the same slot li (nslot == li).
  //
  // #6e COLUMN PREDICTION (perf/kernel-limits). The cap pass and step 11
  // build their columns lazily, one at a time — each a serial 35-row
  // back substitution through local memory, run once per distinct
  // (joint, sweep position) any lane of the warp misses at. Columns
  // likely to be needed join this batched block instead, appended to
  // the limit group in s.l_j[nl..nx-1] (ascending; slots nl..nx-1, so
  // the limit rows keep slots 0..nl-1): every joint whose column the
  // previous substep built (eprev — caps and clamps persist across
  // substeps) and every joint already within 10% of its speed cap in
  // the unconstrained v*. A column is a pure function of this substep's
  // factor, so building it early, or building one that is never used,
  // changes no value — only when and where it is computed. The alpha
  // was swept: 0.9 / 0.95 best, 0.8 / 0.7 progressively worse; eprev
  // alone recovers little, the v* test alone about two thirds.
  {
    uint32_t lmask = 0;
    for (int li = 0; li < s.nl; li++) lmask |= 1u << s.l_j[li];
    uint32_t want = s.eprev;
#pragma unroll
    for (int j = 0; j < NQ; j++)
      if (fabs(v[VOFF + j]) >= T(0.9) * CT(OMEGA_CAP, [j])) want |= 1u << j;
    want &= ~lmask;
    int nx = s.nl;
    while (want) {
      s.l_j[nx++] = __ffs(want) - 1;
      want &= want - 1;
    }
    int li = 0;
    for (; li + 3 < nx; li += 4) {
      const int j0 = s.l_j[li], j1 = s.l_j[li + 1];
      const int j2 = s.l_j[li + 2], j3 = s.l_j[li + 3];
      s.eslot[j0] = uint8_t(li);
      s.eslot[j1] = uint8_t(li + 1);
      s.eslot[j2] = uint8_t(li + 2);
      s.eslot[j3] = uint8_t(li + 3);
      T* const x0 = s.ecol[li];
      T* const x1 = s.ecol[li + 1];
      T* const x2 = s.ecol[li + 2];
      T* const x3 = s.ecol[li + 3];
      ecol_seed(x0, j0);
      ecol_seed(x1, j1);
      ecol_seed(x2, j2);
      ecol_seed(x3, j3);
      // audit #10: each column's forward pass is its own ancestor
      // chain, the four interleaved (tree_fwd_unit_b); the dense back
      // pass is batched
      {
        T* const xs[4] = {x0, x1, x2, x3};
        const int vs[4] = {VOFF + j0, VOFF + j1, VOFF + j2, VOFF + j3};
        tree_fwd_unit_b<4>(d.m, xs, vs);
      }
      tree_back4_n(d.m, x0, x1, x2, x3);
      s.ecomp[j0] = true;
      s.ecomp[j1] = true;
      s.ecomp[j2] = true;
      s.ecomp[j3] = true;
    }
    if (li + 1 < nx) {
      const int j0 = s.l_j[li], j1 = s.l_j[li + 1];
      s.eslot[j0] = uint8_t(li);
      s.eslot[j1] = uint8_t(li + 1);
      T* const x0 = s.ecol[li];
      T* const x1 = s.ecol[li + 1];
      ecol_seed(x0, j0);
      ecol_seed(x1, j1);
      {
        T* const xs[2] = {x0, x1};
        const int vs[2] = {VOFF + j0, VOFF + j1};
        tree_fwd_unit_b<2>(d.m, xs, vs);
      }
      tree_back2_n(d.m, x0, x1);
      s.ecomp[j0] = true;
      s.ecomp[j1] = true;
      li += 2;
    }
    s.nslot = li;
    s.emask = 0;
    for (int k = 0; k < li; k++) s.emask |= 1u << s.l_j[k];
    for (; li < nx; li++) ensure_ecol(d, s, s.l_j[li]);  // sets its bit
  }

  // -- limit rows: mobility verdict of each plain column ----------------------
  for (int li = 0; li < s.nl; li++) ensure_pdiag(d, s, s.l_j[li]);
  for (int li = 0; li < s.nl; li++) {
    s.l_diag[li] = s.pdiag[s.l_j[li]];
    s.l_lam[li] = T(0);
  }
  for (int ci = 0; ci < s.nc; ci++) s.c_lam[ci] = T(0);
  for (int gi = 0; gi < N_GROUND_ROWS; gi++) s.g_lam[gi] = T(0);

  // -- step 8: exactly 12 iterations, fixed order -----------------------------
  // G2/G5: the whole PGS loop runs on vr, a register copy of the
  // local-memory `v` (see the G2 note at the ground block), taken here and
  // written back after the last iteration.
  T vr[NV];
  {
#pragma unroll
    for (int i = 0; i < NV; i++) vr[i] = v[i];
  }
  for (int iter = 0; iter < PGS_ITERS; iter++) {
    // (a2) env v11 GROUND BLOCK (contact spec §3), corner order: the
    // unilateral normal row first (the §2 contact path verbatim), then
    // its two friction rows BOX-clamped against the lambda_n THIS sweep
    // just produced — Gauss-Seidel coupling, re-clamped on every one of
    // the PGS_ITERS iterations. MIRROR of solver.rs `pgs_ground_sweep` /
    // `pgs_ground_sweep_foot`. It runs first in the sweep, with the same
    // fixed iteration count and no early exit.
    // G1: fixed slots walked in ascending corner order, inactive corners
    // skipped (= Rust's packed order).
    // audit #9 (solver.rs `pgs_ground_sweep_foot`, term for term): per
    // foot with a corner down, V = J_f v (six support dots), then each
    // active corner's rows with residual c . V and the per-row lambda step
    // verbatim; a non-zero step is carried as Lambda += c * dd,
    // V += u * dd, and v += G_f Lambda runs ONCE at the end of the foot
    // (was one dense NV axpy per row). Exact arithmetic gives every row
    // the residual and v the value the per-row form produced.
    // G2 (audit #5b): `v` is a local-memory array (the limit / cap passes
    // index it at runtime), so every term above was an LDL and every
    // axpy element an LDL + STL. The sweep runs on a register copy vr
    // instead — every index into it is compile-time (i unrolled, f / t
    // compile-time) — and writes it back once. A copy is the identity
    // on every value; the arithmetic is unchanged operation for
    // operation.
    // G4: the register copy also carries the (b) contact sweep below.
    {
      const T(*x)[2 * FOOT_ROWS] = s.g_minv;
#pragma unroll
      for (int f = 0; f < 2; f++) {
        const uint32_t fbits = (s.g_act >> (CORNERS_PER_FOOT * f)) & 0xFu;
        if (!fbits) continue;
        const int ch0 = WJ_L0 + 6 * f;
        T vf[FOOT_ROWS], lf[FOOT_ROWS];
#pragma unroll
        for (int m = 0; m < FOOT_ROWS; m++) {
          vf[m] = foot_row_dot(m, ch0, s.g_fjr[f], s.g_fjc[f][m],
                               [&](int i) { return vr[i]; });
          lf[m] = T(0);
        }
        bool any = false;
#pragma unroll 1
        for (int k = 0; k < CORNERS_PER_FOOT; k++) {
          if (!((fbits >> k) & 1u)) continue;
          const int nrow = GROUND_ROWS_PER_CORNER * (CORNERS_PER_FOOT * f + k);
          T bound = T(0);
#pragma unroll
          for (int t = 0; t < GROUND_ROWS_PER_CORNER; t++) {
            const int row = nrow + t;
            const int a = GROUND_SPATIAL[t][0];
            const int b1 = 3 + GROUND_SPATIAL[t][1];
            const int b2 = 3 + GROUND_SPATIAL[t][2];
            const T c1 = s.g_coef[row][0], c2 = s.g_coef[row][1];
            const T jv = vf[a] + c1 * vf[b1] + c2 * vf[b2];
            const T dl = (s.g_bias[row] - jv) / s.g_diag[row];
            const T lam = s.g_lam[row];
            // |lambda_t| <= MU * lambda_n for the friction rows. Rust
            // writes `(lambda + dl).min(bound).max(-bound)`; tmin/tmax
            // reproduce f64::min/max including the NaN rule (both `a < b`
            // and `a > b` are false on NaN, so the OTHER operand is
            // returned, exactly as minnum/maxnum do) — an artifact lambda
            // collapses onto the bound instead of propagating.
            const T newl = t == 0 ? tmax(lam + dl, T(0))  // lambda_n >= 0
                                  : tmax(tmin(lam + dl, bound), -bound);
            const T dd = newl - lam;
            s.g_lam[row] = newl;
            if (dd != T(0)) {
              lf[a] += dd;
              lf[b1] += c1 * dd;
              lf[b2] += c2 * dd;
#pragma unroll
              for (int i = 0; i < FOOT_ROWS; i++) vf[i] += s.g_u[row][i] * dd;
              any = true;
            }
            // the friction cone is read AFTER the normal update, so a
            // corner that just lost its normal impulse loses its friction
            // in the same sweep
            if (t == 0) bound = T(GROUND_MU) * s.g_lam[row];
          }
        }
        if (any) {
#pragma unroll
          for (int i = 0; i < NV; i++) {
            T vi = vr[i];
#pragma unroll
            for (int m = 0; m < FOOT_ROWS; m++)
              vi += x[i][FOOT_ROWS * f + m] * lf[m];
            vr[i] = vi;
          }
        }
      }
      // (b) contacts, pair-id order: projected GS, lambda >= 0 — G4: on
      // vr. This dot is DENSE ON PURPOSE. c_J[ci] has 12-23 structurally
      // nonzero columns of 35 and K14 narrowed it to a two-segment cover
      // of the pair's compile-time support: -1.42% / -0.90% / -0.29% in
      // three forms, all reverted. The bound would be runtime, which
      // costs this loop its full unrolling — see the K14 block in the
      // header before touching it.
      for (int ci = 0; ci < s.nc; ci++) {
        T jv = T(0);
#pragma unroll
        for (int i = 0; i < NV; i++) jv += s.c_J[ci][i] * vr[i];
        const T dl = (s.c_bias[ci] - jv) / s.c_diag[ci];
        const T newl = tmax(s.c_lam[ci] + dl, T(0));
        const T dd = newl - s.c_lam[ci];
        s.c_lam[ci] = newl;
        if (dd != T(0))
#pragma unroll
          for (int i = 0; i < NV; i++) vr[i] += s.c_minv[ci][i] * dd;
      }
    }
    // (c) limit rows, joint order (J = sign * e_j: exact-zero terms
    // dropped — Divergences §20c sign-of-zero class)
    // These axpys are DELIBERATELY NOT FUSED across the limit-row group.
    // K13 built exactly that (phase-1 scalar chain on 4 local v
    // components, one shared sweep over i, active rows packed into
    // compile-time 4/3/2/1 widths): bitwise 0.0, no spill, 254 regs —
    // and -2.18% / -0.70%, because `na` is a per-LANE count and the warp
    // pays the UNION of the ladder's widths. The same .pyd is +2.26% on
    // warp-uniform traffic. Read the K13 block in the header before
    // touching this loop.
    // G5: the rows run on vr. The runtime-indexed read v[VOFF + j]
    // becomes a select chain over compile-time slots (a copy of the one
    // selected value — exact), and the axpy has compile-time i.
    for (int li = 0; li < s.nl; li++) {
      const int j = s.l_j[li];
      T vj = vr[VOFF];
#pragma unroll
      for (int jj = 1; jj < NQ; jj++)
        if (j == jj) vj = vr[VOFF + jj];
      const T jv = s.l_sign[li] * vj;
      const T dl = (s.l_bias[li] - jv) / s.l_diag[li];
      const T newl = tmax(s.l_lam[li] + dl, T(0));
      const T dd = newl - s.l_lam[li];
      s.l_lam[li] = newl;
      if (dd != T(0)) {
        const T sd = s.l_sign[li] * dd;
#pragma unroll
        for (int i = 0; i < NV; i++) vr[i] += s.ecol[li][i] * sd;  // #6a
      }
    }
    // (d) omega-cap pass, joint order: momentum-consistent impulses
    // along the plain M^-1 e_j columns.
    // #6c (perf/kernel-limits; was G5's scan + local-memory fallback
    // loop): the pass runs on vr, unrolled over the 29 joints — per
    // joint the same test, impulse and hard write, in the same ascending
    // order, so every value is unchanged (the old scan only skipped the
    // joints before the first firing one, which change nothing). v never
    // takes the round trip through local memory, every vr index is
    // compile-time, and a column miss builds out of line. Trained
    // standard traffic caps ~21 times per env-substep (warp worst lane
    // ~54), in every one of the 12 iterations.
#pragma unroll
    for (int j = 0; j < NQ; j++) {
      const int vj = VOFF + j;
      const T cap = CT(OMEGA_CAP, [j]);
      const T target = tclamp(vr[vj], -cap, cap);
      if (target != vr[vj]) {
        if (!s.pcomp[j]) ensure_pdiag_ool(d, s, j);
        if (!isinf(s.pdiag[j])) {
          const T lam = (target - vr[vj]) / s.pdiag[j];
          const T* const col = ecol_of(s, j);
#pragma unroll
          for (int i = 0; i < NV; i++) vr[i] += col[i] * lam;
          vr[vj] = target;  // exact — the cap is a hard bound
        }
      }
    }
  }
  {
#pragma unroll
    for (int i = 0; i < NV; i++) v[i] = vr[i];
  }
}

// ---- step.rs substep (constrained) ------------------------------------------
template <typename T>
__device__ void substep(Dyn<T>& __restrict__ d, Solver<T>& __restrict__ s,
                        const T* __restrict__ tau) {
  // 1-3. FK, CRBA, RNEA bias
  fk_all(d);
  crba(d);
  rnea_bias(d);
  // 4. rhs = tau - b*qd - C (root rows 0 - C)
  for (int i = 0; i < VOFF; i++) d.rhs[i] = T(0) - d.c[i];
  for (int j = 0; j < NQ; j++)
    d.rhs[VOFF + j] =
        tau[j] - CT(DAMPING, [j]) * d.v[VOFF + j] - d.c[VOFF + j];
  // 5. Cholesky — IN PLACE: d.m holds L from here to the end of the
  // substep (crba rewrites the whole profile next substep). audit #10: in
  // the fill-free tree order (tree_factor; SPEC-v10 Divergences §25).
  tree_factor(d.m, &d.pd_bad);
  // 6. v* = v + h M^-1 (tau - C)
  tree_solve_n(d.m, d.rhs);
  T v[NV];
  for (int i = 0; i < NV; i++) v[i] = d.v[i] + T(H_SUB) * d.rhs[i];
  // shared per-joint column caches reset per substep (this factor)
  for (int j = 0; j < NQ; j++) {
    s.ecomp[j] = false;
    s.pcomp[j] = false;
  }
  // 7-9. constraint stage
  constraint_solve(d, s, d.q, v);
  for (int i = 0; i < NV; i++) d.v[i] = v[i];
  // 12-PRE. root velocity envelope, FIRST application (SPEC-v10 §3
  //     step 12 as amended 2026-08-03 — Divergences §23). Identical
  //     expression tree to the post-step-11 application at the bottom of
  //     this function; run HERE it is the one that bounds what the
  //     integrator consumes: ‖Δp‖ ≤ h·ROOT_V_MAX and |Δθ| ≤ h·ROOT_W_MAX
  //     per substep, UNCONDITIONALLY, and hence the next substep's
  //     Baumgarte contact / limit bias. Never engages on a healthy trajectory (a
  //     converged v10 policy: 0 engagements in 204,800 env-ticks), so it
  //     is bitwise transparent there. Engagements are counted and the env
  //     layer terminates the episode on them (C backstop).
  {
    bool engaged = false;
    const T w2 = d.v[3] * d.v[3] + d.v[4] * d.v[4] + d.v[5] * d.v[5];
    if (w2 > T(ROOT_W_MAX) * T(ROOT_W_MAX)) {
      const T sc = T(ROOT_W_MAX) / sqrt(w2);
      d.v[3] *= sc;
      d.v[4] *= sc;
      d.v[5] *= sc;
      engaged = true;
    }
    const T l2 = d.v[0] * d.v[0] + d.v[1] * d.v[1] + d.v[2] * d.v[2];
    if (l2 > T(ROOT_V_MAX) * T(ROOT_V_MAX)) {
      const T sc = T(ROOT_V_MAX) / sqrt(l2);
      d.v[0] *= sc;
      d.v[1] *= sc;
      d.v[2] *= sc;
      engaged = true;
    }
    if (engaged) {
      d.env_bad = true;
      d.env_n += 1;
    }
  }
  // 10. semi-implicit Euler positions + root exp-map quaternion
  for (int j = 0; j < NQ; j++) d.q[j] += T(H_SUB) * d.v[VOFF + j];
  for (int i = 0; i < 3; i++) d.p[i] += T(H_SUB) * d.v[i];
  {
    const T u0 = d.v[3] * T(H_SUB) * T(0.5);
    const T u1 = d.v[4] * T(H_SUB) * T(0.5);
    const T u2 = d.v[5] * T(H_SUB) * T(0.5);
    const T un = sqrt(u0 * u0 + u1 * u1 + u2 * u2);
    T qe[4];
    if (un < T(1e-12)) {
      qe[0] = T(1);
      qe[1] = T(0);
      qe[2] = T(0);
      qe[3] = T(0);
    } else {
      const T su = sin(un), cu = cos(un);
      qe[0] = cu;
      qe[1] = su * u0 / un;
      qe[2] = su * u1 / un;
      qe[3] = su * u2 / un;
    }
    T qq[4];
    quat_mul(qe, d.quat, qq);
    const T qn =
        sqrt(qq[0] * qq[0] + qq[1] * qq[1] + qq[2] * qq[2] + qq[3] * qq[3]);
    for (int i = 0; i < 4; i++) d.quat[i] = qq[i] / qn;
  }
  // 11. post-solve clamps as momentum-consistent impulses (joint order,
  //     CLAMP_PASSES Gauss-Seidel passes with Rust's early-out). env
  //     v14: the direct-write guarantee pass is DELETED in both
  //     engines — its reaction-free writes manufactured up to ~335 N
  //     of external force when many joints pinned (step.rs step-11
  //     comment has the audit).
  // hitm bit j: joint j is past a stop (was bool s.hit[NQ] in local
  // memory, re-read by every one of up to 64 unrolled passes below)
  uint32_t hitm = 0;
  for (int j = 0; j < NQ; j++) {
    const T lo = CT(LIMIT_LO, [j]), hi = CT(LIMIT_HI, [j]);
    const bool hit = d.q[j] < lo || d.q[j] > hi;
    hitm |= uint32_t(hit) << j;
    // env v14: no routine position teleport (reaction-free q writes
    // are the phantom-force class — step.rs comment); deep-violation
    // backstop only
    if (hit) {
      const T over = d.q[j] < lo ? lo - d.q[j] : d.q[j] - hi;
      if (over > T(LIMIT_TELEPORT_BACKSTOP)) d.q[j] = tclamp(d.q[j], lo, hi);
    }
  }
  // #6c (perf/kernel-limits): the passes run on a register copy of d.v,
  // unrolled over the 29 joints — the same targets, impulses and hard
  // writes in the same order (a copy is the identity), with compile-time
  // indices and no store -> load chain through local d.v per firing; a
  // column miss builds out of line. On trained standard traffic the
  // warp's worst lane runs ~57 of the 64 passes (~27 firings per
  // env-substep, worst lane ~210), so this loop was the kernel's
  // dominant cost: 69.5 -> 29.9 ms/tick together with the cap pass
  // above (RTX 5090, N=32768, sampled actions).
  T vr[NV];
#pragma unroll
  for (int i = 0; i < NV; i++) vr[i] = d.v[i];
  for (int pass = 0; pass < CLAMP_PASSES; pass++) {
    bool changed = false;
#pragma unroll
    for (int j = 0; j < NQ; j++) {
      const int vj = VOFF + j;
      const T cap = CT(OMEGA_CAP, [j]);
      // v14: at a stop the target is a Baumgarte recovery toward the
      // bound with a SLOP_LIMIT dead-band (mirrors step.rs) — the
      // impulse restores position reciprocally; a parked joint inside
      // the slop fires nothing
      T target;
      if ((hitm >> j) & 1u) {
        const T lo = CT(LIMIT_LO, [j]), hi = CT(LIMIT_HI, [j]);
        const T over = d.q[j] < lo ? lo - d.q[j] : hi - d.q[j];
        const T rec = fabs(over) - T(SLOP_LIMIT);
        target = rec <= T(0)
                     ? T(0)
                     : tclamp(T(BETA_LIMIT) * (over < T(0) ? -rec : rec) / T(H_SUB),
                              -cap, cap);
      } else {
        target = tclamp(vr[vj], -cap, cap);
      }
      if (target != vr[vj]) {
        if (!s.ecomp[j]) ensure_ecol_ool(d, s, j);
        const T* const col = ecol_of(s, j);
        const T lam = (target - vr[vj]) / col[vj];
#pragma unroll
        for (int i = 0; i < NV; i++) vr[i] += col[i] * lam;
        vr[vj] = target;  // exact — the clamp is a hard bound
        changed = true;
      }
    }
    if (!changed) break;
  }
#pragma unroll
  for (int i = 0; i < NV; i++) d.v[i] = vr[i];
  // (env v14: no direct-write guarantee pass — mirror of step.rs.)
  s.eprev = s.emask;  // #6e: this substep's columns predict the next's
  // 12. root velocity envelope guarantee, SECOND application
  //     (Divergences §19b; the first is the 12-PRE block above, §23).
  //     Bounds the STORED velocity — step 11's joint impulses have
  //     non-zero root rows. Deliberately NOT counted: the pre-integration
  //     application is the escape signal.
  {
    const T w2 = d.v[3] * d.v[3] + d.v[4] * d.v[4] + d.v[5] * d.v[5];
    if (w2 > T(ROOT_W_MAX) * T(ROOT_W_MAX)) {
      const T sc = T(ROOT_W_MAX) / sqrt(w2);
      d.v[3] *= sc;
      d.v[4] *= sc;
      d.v[5] *= sc;
    }
    const T l2 = d.v[0] * d.v[0] + d.v[1] * d.v[1] + d.v[2] * d.v[2];
    if (l2 > T(ROOT_V_MAX) * T(ROOT_V_MAX)) {
      const T sc = T(ROOT_V_MAX) / sqrt(l2);
      d.v[0] *= sc;
      d.v[1] *= sc;
      d.v[2] *= sc;
    }
  }
}

// ---- env-layer geometry (lib.rs; carried from the v9 kernel) ----------------
template <typename T>
__device__ bool tri_box_overlap(const T v[3][3], const T h[3]) {
  for (int k = 0; k < 3; k++) {
    T lo = tmin(tmin(v[0][k], v[1][k]), v[2][k]);
    T hi = tmax(tmax(v[0][k], v[1][k]), v[2][k]);
    if (lo > h[k] || hi < -h[k]) return false;
  }
  T e[3][3];
  for (int k = 0; k < 3; k++) {
    e[0][k] = v[1][k] - v[0][k];
    e[1][k] = v[2][k] - v[1][k];
    e[2][k] = v[0][k] - v[2][k];
  }
  for (int j = 0; j < 3; j++) {
    T ax[3][3] = {{T(0), -e[j][2], e[j][1]},
                  {e[j][2], T(0), -e[j][0]},
                  {-e[j][1], e[j][0], T(0)}};
    for (int k = 0; k < 3; k++) {
      T p0 = dot3(v[0], ax[k]);
      T p1 = dot3(v[1], ax[k]);
      T p2 = dot3(v[2], ax[k]);
      T r = h[0] * fabs(ax[k][0]) + h[1] * fabs(ax[k][1])
            + h[2] * fabs(ax[k][2]);
      if (tmin(tmin(p0, p1), p2) > r || tmax(tmax(p0, p1), p2) < -r)
        return false;
    }
  }
  T n[3];
  cross3(e[0], e[1], n);
  T r = h[0] * fabs(n[0]) + h[1] * fabs(n[1]) + h[2] * fabs(n[2]);
  return fabs(dot3(n, v[0])) <= r;
}

template <typename T>
__device__ T tri_point_dist(const T a[3], const T b[3], const T c[3],
                            const T p[3]) {
  T ab[3], ac[3], ap[3];
  for (int k = 0; k < 3; k++) {
    ab[k] = b[k] - a[k];
    ac[k] = c[k] - a[k];
    ap[k] = p[k] - a[k];
  }
  T d1 = dot3(ab, ap);
  T d2 = dot3(ac, ap);
  T q[3];
  if (d1 <= T(0) && d2 <= T(0)) {
    for (int k = 0; k < 3; k++) q[k] = p[k] - a[k];
    return sqrt(dot3(q, q));
  }
  T bp[3];
  for (int k = 0; k < 3; k++) bp[k] = p[k] - b[k];
  T d3 = dot3(ab, bp);
  T d4 = dot3(ac, bp);
  if (d3 >= T(0) && d4 <= d3) {
    for (int k = 0; k < 3; k++) q[k] = p[k] - b[k];
    return sqrt(dot3(q, q));
  }
  T vc = d1 * d4 - d3 * d2;
  if (vc <= T(0) && d1 >= T(0) && d3 <= T(0)) {
    T denom = d1 - d3;
    T v = fabs(denom) < T(1e-15) ? T(0) : d1 / denom;
    for (int k = 0; k < 3; k++) q[k] = p[k] - (a[k] + v * ab[k]);
    return sqrt(dot3(q, q));
  }
  T cp[3];
  for (int k = 0; k < 3; k++) cp[k] = p[k] - c[k];
  T d5 = dot3(ab, cp);
  T d6 = dot3(ac, cp);
  if (d6 >= T(0) && d5 <= d6) {
    for (int k = 0; k < 3; k++) q[k] = p[k] - c[k];
    return sqrt(dot3(q, q));
  }
  T vb = d5 * d2 - d1 * d6;
  if (vb <= T(0) && d2 >= T(0) && d6 <= T(0)) {
    T denom = d2 - d6;
    T w = fabs(denom) < T(1e-15) ? T(0) : d2 / denom;
    for (int k = 0; k < 3; k++) q[k] = p[k] - (a[k] + w * ac[k]);
    return sqrt(dot3(q, q));
  }
  T va = d3 * d6 - d5 * d4;
  if (va <= T(0) && (d4 - d3) >= T(0) && (d5 - d6) >= T(0)) {
    T denom = (d4 - d3) + (d5 - d6);
    T w = fabs(denom) < T(1e-15) ? T(0) : (d4 - d3) / denom;
    for (int k = 0; k < 3; k++) q[k] = p[k] - (b[k] + w * (c[k] - b[k]));
    return sqrt(dot3(q, q));
  }
  T denom = T(1) / (va + vb + vc);
  T v = vb * denom;
  T w = vc * denom;
  for (int k = 0; k < 3; k++) q[k] = p[k] - (a[k] + ab[k] * v + ac[k] * w);
  return sqrt(dot3(q, q));
}

template <typename T>
__device__ T seg_point_dist3(const T a[3], const T b[3], const T c[3]) {
  T ab[3], ac[3];
  for (int k = 0; k < 3; k++) {
    ab[k] = b[k] - a[k];
    ac[k] = c[k] - a[k];
  }
  T denom = dot3(ab, ab);
  T t = denom < T(1e-12) ? T(0) : tclamp(dot3(ac, ab) / denom, T(0), T(1));
  T px = a[0] + t * ab[0] - c[0];
  T py = a[1] + t * ab[1] - c[1];
  T pz = a[2] + t * ab[2] - c[2];
  return sqrt(px * px + py * py + pz * pz);
}

// note_frame — env v10: `head` is the LIVE FK head position.
template <typename T>
__device__ void note_frame(const T p[3], T cos_r, T sin_r, T lam,
                           const T head[3], T f[3][3]) {
  T up[3] = {-sin_r, cos_r, T(0)};
  T hp1 = head[1] + T(LOOK_VERT) * (p[1] - head[1]);
  T dv[3] = {p[0] - head[0], p[1] - hp1, p[2] - head[2]};
  T dn = tmax(sqrt(dot3(dv, dv)), T(1e-9));
  T g[3] = {lam * dv[0] / dn, lam * dv[1] / dn,
            (T(1) - lam) + lam * dv[2] / dn};
  T gn = tmax(sqrt(dot3(g, g)), T(1e-9));
  for (int k = 0; k < 3; k++) f[2][k] = g[k] / gn;
  T cx[3];
  cross3(up, f[2], cx);
  T cn = tmax(sqrt(dot3(cx, cx)), T(1e-9));
  for (int k = 0; k < 3; k++) f[0][k] = cx[k] / cn;
  cross3(f[2], f[0], f[1]);
}

template <typename T>
__device__ void to_note_frame(const T p[3], const T c[3], const T f[3][3],
                              T out[3]) {
  T q[3] = {p[0] - c[0], p[1] - c[1], p[2] - c[2]};
  out[0] = dot3(q, f[0]);
  out[1] = dot3(q, f[1]);
  out[2] = dot3(q, f[2]);
}

template <typename T>
__device__ bool sweep_hits_box(const T w0[3], const T t0[3], const T t1[3],
                               const T w1[3], const T c[3], const T h[3],
                               const T f[3][3]) {
  T rel[3][3];
  to_note_frame(w0, c, f, rel[0]);
  to_note_frame(t0, c, f, rel[1]);
  to_note_frame(t1, c, f, rel[2]);
  if (tri_box_overlap(rel, h)) return true;
  to_note_frame(t1, c, f, rel[1]);
  to_note_frame(w1, c, f, rel[2]);
  return tri_box_overlap(rel, h);
}

template <typename T>
__device__ T sweep_dist(const T w0[3], const T t0[3], const T t1[3],
                        const T w1[3], const T p[3]) {
  return tmin(tri_point_dist(w0, t0, t1, p), tri_point_dist(w0, t1, w1, p));
}

// ---- Flux geometry (flux.rs circle_touch / bar_sweet_dist verbatim) ----
// Palm path a_w -> b_w vs the circle centred at c: clip to the depth
// slab, closest in-plane approach. Returns the distance when it is
// within CIRCLE_R, else -1 (Rust: Option<Real>).
template <typename T>
__device__ T flux_circle_touch(const T a_w[3], const T b_w[3], const T c[3]) {
  const T a[3] = {a_w[0] - c[0], a_w[1] - c[1], a_w[2] - c[2]};
  const T b[3] = {b_w[0] - c[0], b_w[1] - c[1], b_w[2] - c[2]};
  const T dz = b[2] - a[2];
  T t0 = T(0), t1 = T(1);
  if (fabs(dz) < T(1e-12)) {
    if (a[2] < T(FLUX_SLAB_LO) || a[2] > T(FLUX_SLAB_HI)) return T(-1);
  } else {
    const T ta = (T(FLUX_SLAB_LO) - a[2]) / dz;
    const T tb = (T(FLUX_SLAB_HI) - a[2]) / dz;
    t0 = tmax(t0, tmin(ta, tb));
    t1 = tmin(t1, tmax(ta, tb));
    if (t0 > t1) return T(-1);
  }
  const T ex = b[0] - a[0];
  const T ey = b[1] - a[1];
  const T ee = ex * ex + ey * ey;
  const T t = ee < T(1e-12) ? t0
                            : tclamp(-(a[0] * ex + a[1] * ey) / ee, t0, t1);
  const T qx = a[0] + t * ex;
  const T qy = a[1] + t * ey;
  const T d = sqrt(qx * qx + qy * qy);
  return d <= T(CIRCLE_R) ? d : T(-1);
}

// Distance along the bar axis u from where the CURRENT Flux line (palms
// l, r projected onto the bar plane) crosses the axis c + s*u (clamped
// onto the bar) to the sweet spot at sweet*len/2; a parallel or
// collapsed Flux takes its projected span's point nearest the sweet spot.
template <typename T>
__device__ T flux_bar_sweet_dist(T nx, T ny, T len, T sweet, T ux, T uy,
                                 const T l[3], const T r[3]) {
  const T half = T(0.5) * len;
  const T s_sweet = sweet * half;
  const T fx = r[0] - l[0];
  const T fy = r[1] - l[1];
  const T lx = l[0] - nx;
  const T ly = l[1] - ny;
  const T cr = ux * fy - uy * fx;
  const T ff = fx * fx + fy * fy;
  T sv;
  if (ff > T(1e-12) && cr * cr > T(1e-12) * ff) {
    sv = (lx * fy - ly * fx) / cr;
  } else {
    const T sa = lx * ux + ly * uy;
    const T sb = (r[0] - nx) * ux + (r[1] - ny) * uy;
    sv = tclamp(s_sweet, tmin(sa, sb), tmax(sa, sb));
  }
  return fabs(tclamp(sv, -half, half) - s_sweet);
}

// Bar crossing-angle check (flux.rs bar_angle_ok verbatim): the CURRENT
// Flux line may deviate at most 60 deg from the bar's perpendicular
// n = (cos θ, sin θ) = p_rot (2D, undirected): (f·n)^2 >= COS_TOL^2 |f|^2.
// A collapsed Flux (|f|^2 <= 1e-12) fails.
template <typename T>
__device__ bool flux_bar_angle_ok(T cos_r, T sin_r, const T l[3],
                                  const T r[3]) {
  const T fx = r[0] - l[0];
  const T fy = r[1] - l[1];
  const T ff = fx * fx + fy * fy;
  if (ff <= T(1e-12)) return false;
  const T dn = fx * cos_r + fy * sin_r;
  return dn * dn >= T(COS_TOL) * T(COS_TOL) * ff;
}

// ---- kernel parameters ------------------------------------------------------
// Every pointer below is __restrict__: the 58 slots are 58 DISTINCT
// torch tensors, each its own allocation (cuda_env.py CudaBeatVecEnv
// __init__ + the pool dict + the caller's actions/cand), so no two of
// them can overlap. Without the qualifier a store through any one of
// them (e.g. P.status[..] in the note loop) forces the compiler to
// reload everything else it had cached — the kernel is local/global
// traffic bound, so that reload traffic is the cost. The const pool
// pointers additionally become read-only (LDG) loads on sm_61.
// INVARIANT: if a future change ever aliases two slots to the same
// tensor, the restrict qualifiers here must go.
template <typename T>
struct Sim {
  // dynamics state (mutated in place)
  T *__restrict__ p, *__restrict__ quat, *__restrict__ q, *__restrict__ v;
  // FK caches
  T *__restrict__ wrist, *__restrict__ blade, *__restrict__ wrist_vel,
      *__restrict__ head, *__restrict__ shoulder_z;
  // env v24: §8 body-point positions of the last tick (finite-diff
  // base) and the resulting world velocities (lib.rs body_pt_vel)
  T *__restrict__ body_pt, *__restrict__ body_pt_vel;
  // Flux: palm-centre finite-difference velocity (E, 2, 3) — lib.rs
  // Core::palm_vel; zeroed by the mode-2 refresh, set to the start
  // kick by reset_env (0 without start_jitter), written only by the
  // Flux judgment (read only by the Flux obs; dead in the standard game)
  T* __restrict__ palm_vel;
  // env v49: blade-tip finite-difference velocity (E, 2, 3) — lib.rs
  // Body::tip_vel; written every tick next to wrist_vel (both game
  // modes, like wrist_vel), zeroed by reset_env and the mode-2 refresh
  T* __restrict__ tip_vel;
  // swing chain
  T *__restrict__ swing_acc, *__restrict__ axis;
  bool* __restrict__ axis_valid;
  // follow-through tracker ring (aggregated)
  T* __restrict__ tr_acc;
  int *__restrict__ tr_age, *__restrict__ tr_cnt;
  bool* __restrict__ tr_active;
  int* __restrict__ tr_ptr;
  // episode state
  int* __restrict__ tick;
  uint8_t* __restrict__ status;
  int* __restrict__ first_pending;
  // env v45 per-hand previous-note record [(e,L),(e,R)], indexed by
  // note colour: chart index of the last resolved note of that colour
  // + valid (lib.rs prev_idx / prev_valid mirror)
  int* __restrict__ prev_idx;
  uint8_t* __restrict__ prev_valid;
  // env v46: per-env axis torque-limit scales (N, 16), the wrapper's
  // U(-1,1) draw buffer (N, 16) consumed at reset when tau_rand > 0,
  // the half-range, and the per-episode outcome record (N, 28) +
  // flag written at `done` for the trainer's tau_draws log:
  // [s0..s15, ep_ticks, ep_return, fell, good_cuts, score_pts, chart]
  T* __restrict__ tau_scale;
  const T* __restrict__ tau_jit;
  T tau_rand;
  T* __restrict__ ep_rec;
  uint8_t* __restrict__ ep_flag;
  // bench combo walk (2026-09-16, CUDA bench): the ScoreSaber multiplied-
  // score walk of lib.rs ComboWalk, applied AT the event sites in the
  // same order Rust pushes its events (post -> note loop -> walls ->
  // post). combo_walk=false leaves every field untouched (training).
  bool combo_walk;
  // env v47 (2026-09-19): training-time per-episode score telemetry —
  // combo walk on WITHOUT the bench's done-freeze (auto-reset stays),
  // ep_rec rows written for every episode (cols 22..28 filled)
  bool ep_stats;
  int* __restrict__ cw_mult;          // (N) current multiplier 1/2/4/8
  int* __restrict__ cw_prog;          // (N) progress toward the next tier
  int* __restrict__ cw_combo;         // (N) current combo
  int* __restrict__ cw_maxc;          // (N) highest combo
  int* __restrict__ cw_breaks;        // (N) combo breaks
  T* __restrict__ cw_score;           // (N) multiplied score
  uint8_t* __restrict__ cw_note_mult; // (N, C) pinned cut-time multiplier
  int* __restrict__ tr_msum;          // (N, 2, N_TRACKERS) sum of the pinned
                                      // multipliers of a tracker's notes
  // bench slot refill (2026-09-16): at `done` the finished level's bench
  // counters are snapshotted into bench_rec (N, BENCH_REC) + a flag, its
  // flail accumulators cleared, and the slot resets IN-KERNEL onto
  // cand[e] (>= 0: the host's next level for this slot) or stays frozen
  // (cand < 0: queue exhausted — the combo_walk early-out above keeps it
  // idle at ~zero cost). Row: [good, bad, miss, score, bombs, pre, acc,
  // post, flail_rad(2 hands), flail_ticks, cw_score, cw_maxc, cw_breaks,
  // chart_idx, wrev_L, wrev_R, ticks, wall_enters, term_flags] + col 19,
  // the live wall-entry accumulator (kernel scratch, never host-facing).
  bool bench_refill;
  T* __restrict__ bench_rec;
  uint8_t* __restrict__ bench_flag;
  // R12 warp-coherence permutation: thread gid serves env env_order[gid]
  // (identity unless the host sorts it — see cuda_env.step). A bijection
  // on the thread->env assignment only; all P.*[e] addressing unchanged.
  const int* __restrict__ env_order;
  T* __restrict__ ep_return;
  int *__restrict__ good, *__restrict__ bad, *__restrict__ miss,
      *__restrict__ bombs;
  T *__restrict__ score, *__restrict__ score_pre, *__restrict__ score_acc,
      *__restrict__ score_post;
  // contact aggregation (obs bits + parity introspection)
  T *__restrict__ contact_imp, *__restrict__ contact_pt;
  bool* __restrict__ contact_arm;
  // activation telemetry (2026-09-13, replaces the retired positive-
  // work reading): per env [Σ total, Σ spine, Σ arms, Σ legs, ticks]
  // of the commanded |gear fraction| (the retired v23 iso-price base),
  // accumulated always; popped + zeroed by the
  // wrapper's pop_activation, NOT touched by reset_env (limit_rest
  // pattern). lib.rs Core::act_sum mirror.
  T* __restrict__ act_stat;
  // env v43 torque telemetry (2026-09-14): per env [Σ total, Σ spine,
  // Σ arms, Σ legs, ticks] of the applied |τ| in N·m (total = the
  // energy-priced sum); act_stat pattern (popped by pop_torque, not
  // touched by reset_env). lib.rs Core::tau_sum mirror.
  T* __restrict__ tau_stat;
  int* __restrict__ pd_hits;  // f32 pivot-floor engagements (§20e)
  int* __restrict__ env_hits;  // pre-integration root-envelope hits (§23)
  // env v11 FALL terminations (unilateral-contact spec §4), counted per
  // (env, TICK) exactly as lib.rs `fall_hits` is — the root-escape
  // envelope's twin, popped by the wrapper and reported by the trainer.
  int* __restrict__ fall_hits;
  // (env, TICK) tilt terminations — lib.rs tilt_hits mirror
  int* __restrict__ tilt_hits;
  // foot-lift events (pop counter, PER FOOT [L, R] since 2026-08-20)
  // + per-foot grounded-last-tick state (E, 2) — lib.rs
  // foot_lift_hits / foot_down_prev mirrors
  int* __restrict__ foot_lift_hits;
  bool* __restrict__ foot_down_prev;
  // env v19 flail telemetry (per env, popped + zeroed by the wrapper's
  // pop_flail, NOT touched by reset_env): blade arc (rad) accrued on
  // note-free ticks (next scoreable note > FLAIL_LEAD_S away) by hands
  // with no live follow-through tracker — swinging that cannot earn —
  // and the count of those note-free ticks (the denominator).
  T* __restrict__ flail_rad;
  int* __restrict__ flail_ticks;
  // wrist-bend reversal telemetry (2026-09-22): per env x hand [L, R] —
  // count (flail lifecycle: popped by the wrapper's pop_wrist_rev in
  // training, snapshotted + zeroed per level by the bench refill), the
  // running extremes [lo, hi] (N, 2, 2), and the detector state (0 =
  // fresh episode, extremes invalid; 3 = armed, no direction yet; 1 =
  // moving +; 2 = moving -). reset_env zeroes the state, never the
  // count.
  int* __restrict__ wrev;
  T* __restrict__ wrev_ext;
  int* __restrict__ wrev_dir;
  // env v34 (2026-08-21) limit-rest telemetry (per env, [legs, spine],
  // popped + zeroed by the wrapper's pop_limit_rest, NOT touched by
  // reset_env): tick-summed joint-limit-row impulses (N.m.s) the
  // POSTURAL stops carried — spine joints 0..3 and leg joints 17..29;
  // arms accumulate in bucket [2] — telemetry always, PRICED only
  // when limit_arms widens the scope (env v35; whips legitimately slam arm
  // stops at speed). Step-11 clamp impulses are NOT counted (the
  // solver is limit-aware; they are near-null by design).
  T* __restrict__ limit_rest;
  int* __restrict__ limit_rest_ticks;
  // 2026-08-20 training-accuracy telemetry (user metric decision:
  // average note score over all notes hit replaces leaf-end return as
  // the guiding training metric). acc_pts accrues every point of note
  // score AS IT IS CREDITED (pre+acc at the cut, post at follow-
  // through resolution, x cnt for stacked trackers); acc_cuts counts
  // good cuts once at cut time. Per (env, HAND) [L, R] since
  // 2026-08-20 (user: per-hand acc); popped + zeroed by the
  // wrapper's pop_acc, NOT touched by reset_env — a continuous flow
  // with no episode windowing, so the ratio acc_pts/(115*acc_cuts) is
  // phase-free by construction (no completion waves, no censoring).
  T* __restrict__ acc_pts;
  int* __restrict__ acc_cuts;
  // env v12 (contact spec §7): tick-summed ground NORMAL impulses per
  // (env, bucket) — [L_heel, L_toe, R_heel, R_toe] (N·s), the lib.rs
  // Body::ground_imp mirror. Written by the step tick, zeroed by
  // reset_env, read back by write_obs.
  T* __restrict__ ground_imp;
  int64_t* __restrict__ chart_idx;
  // chart pool, read-only. RAGGED (CSR) since 2026-09-03 (audit 2.7):
  // every segment's objects back to back in one flat array per field,
  // chart cc's object i at p_note_off[cc] + i (was cc * C + i in the
  // padded (P, C) layout). C survives only as the per-env status
  // stride / scan bound (max objects over the pool).
  const T *__restrict__ p_hit, *__restrict__ p_speed, *__restrict__ p_center,
      *__restrict__ p_dvec, *__restrict__ p_rot, *__restrict__ p_good_hy,
      *__restrict__ p_bad_hy;
  const bool *__restrict__ p_valid, *__restrict__ p_is_right,
      *__restrict__ p_any_dir, *__restrict__ p_is_bomb,
      *__restrict__ p_is_head, *__restrict__ p_is_link,
      *__restrict__ p_show_dir, *__restrict__ p_free_pre,
      *__restrict__ p_free_post;
  // Flux bars (0 on every other object): length (m) and sweet-spot
  // offset in [-1, 1]; a Flux object is a BAR iff p_len > 0 (flux
  // charts assert len > 0 on bars, circles carry 0 — lib.rs flux.rs)
  const T *__restrict__ p_len, *__restrict__ p_sweet;
  const int64_t* __restrict__ p_n_notes;
  const T *__restrict__ p_last_hit, *__restrict__ p_tail,
      *__restrict__ p_min_speed;
  const int64_t* __restrict__ p_note_off;  // (P + 1,) exclusive prefix
  // env v17 wall pool, read-only, indexed by the SAME chart_idx; ragged
  // like the notes: wall wi of chart cc at p_wall_off[cc] + wi
  const T *__restrict__ p_wall_t, *__restrict__ p_wall_cx,
      *__restrict__ p_wall_cy, *__restrict__ p_wall_w,
      *__restrict__ p_wall_h, *__restrict__ p_wall_len,
      *__restrict__ p_wall_speed;
  const bool* __restrict__ p_wall_valid;
  const int64_t* __restrict__ p_n_walls;
  const int64_t* __restrict__ p_wall_off;  // (P + 1,)
  // env v17 per-env wall state: passed latches (E, W), passed-prefix
  // cursor, and the head-in-wall tick counter the wrapper reports
  bool* __restrict__ wall_passed;
  int* __restrict__ first_wall;
  int* __restrict__ walls_in;
  // was the head inside a wall LAST tick — entry-edge state for the
  // once-per-entry WALL_PEN (lib.rs was_in_wall mirror)
  bool* __restrict__ in_wall_prev;
  int W;  // per-env wall_passed stride (max walls over the pool)
  // io
  const int64_t *__restrict__ actions, *__restrict__ cand;
  T* __restrict__ ret_buf;
  int* __restrict__ ret_total;
  float* __restrict__ obs;
  T* __restrict__ reward;
  bool* __restrict__ done;
  int N, C, max_steps, cap, mode;  // mode: 0 step, 1 reset-all, 2 refresh
  T fall_const;   // v18 (sole mode): flat -fall_const fall reward (lib.rs mirror)
  // env v48 (2026-09-20, user): asinh z lane — lane 2 (note z / wall
  // front) and lane 17 (wall back) become asinh(max(z, z_lo) / z_asinh)
  // (metres; linear near the plane, log far out, NO upper clamp, so far
  // tokens stay distinguishable). Always > 0 (the only z encoding
  // since 2026-09-22). The speed lane is untouched. Recorded in the
  // checkpoint (arch + args); the rust twin reads BEAT_Z_ASINH.
  T z_asinh;
  // env v20 (2026-08-16): per-tick penalty foot_z_coef * (|lfoot_z -
  // PLANE_Z0| + |rfoot_z - PLANE_Z0|) — linear in each foot's world-z
  // distance from the reference foot z (lib.rs mirror). 0 = off.
  T foot_z_coef;
  // env v21: the X twin — each foot priced against its OWN side's
  // wide-stance x (left -> -FOOT_X0, right -> +FOOT_X0). 0 = off.
  T foot_x_coef;
  // env v22: hinged rent radius (0 = v21 pure-linear verbatim) +
  // potential-based delta-shaping coefficient (lib.rs mirror).
  T foot_hinge_r;
  T foot_shape_coef;
  // env v43 (2026-09-14, user): per-tick ENERGY price on the applied
  // Σ|τ_j| = Σ|frac_j|·τ_max[j] (N·m; lib.rs Core::energy_coef mirror).
  // Replaced the v23 fraction price: a joint pays in proportion to its
  // torque limit (the human muscle-volume weighting). Velocity-blind:
  // force, not work. 0 = off. 7.5e-6 ≈ the old 5e-4 fraction bill.
  T energy_coef;
  // env v34: per-tick price on the postural limit-rest impulse (the
  // limit_rest accumulation above; lib.rs Core::limit_coef mirror).
  // Proportional to the CARRIED impulse: a knee-locked stand (near-
  // zero moment on the stop) is near-free, a crouch parked on its
  // hip/knee stops pays its full gravity moment. 0 = off (bitwise
  // pre-v34 rewards).
  T limit_coef;
  int limit_arms;  // env v35: price arm limit rows too (0 = postural only)
  // env v41 (2026-09-12, user): SELF-CONTACT penalty — per-tick price on
  // the tick-summed self-collision row impulses (s.c_lam over the 55
  // capsule pairs, N.s; the same lambdas the replay "contacts" carry).
  // Impulse-weighted: a graze is near-free, a forearm slammed into the
  // head pays its momentum. Blades collide with nothing, so this is
  // body-on-body only. 0 = off (bitwise pre-v41 rewards). Telemetry
  // sc_stat[e] = [impulse sum, ticks with contact, ticks] accumulates
  // regardless of the coefficient (limit_rest pattern; popped + zeroed
  // by the wrapper's pop_self_contact). lib.rs Core::self_contact_coef
  // mirror.
  T self_contact_coef;
  T* __restrict__ sc_stat;
  // env v37: bomb-like wall despawn on charge (training-only flag)
  int wall_despawn;
  // env v22: previous tick's total hinged foot deviation, per env
  // (zeroed by reset_env — resets start on-target).
  T* __restrict__ foot_pot;
  // env v20 start-pose jitter (TRAINING-ONLY, no lib.rs mirror — the
  // push/start_jitter precedent): per-env U(-1,1) draws for the 12 leg
  // joints, refilled by the wrapper every launch; consumed by reset_env
  // when start_pose_jitter > 0 (radians scale).
  const T* __restrict__ pose_jit;
  T start_pose_jitter;
  // start-jitter velocity kick (review B10, 2026-09-30; TRAINING-ONLY,
  // no lib.rs mirror — the Rust sim has no per-episode RNG): (N, 2)
  // world [dvx, dvz] m/s, precomputed by the wrapper every launch
  // (|dv| ~ U(0, start_jitter), direction uniform; all zeros when
  // start_jitter == 0, so reset_env's stores are then the old zeros).
  // reset_env installs it as the root's horizontal velocity BEFORE
  // store_state / write_obs, so the reset obs shows the kicked state.
  // Before B10 the wrapper added it to v AFTER the launch, so the first
  // obs of every episode read root velocity 0 while the body moved.
  const T* __restrict__ start_kick;
  // env v38 crouch-rich starts (user; TRAINING-ONLY, jitter
  // precedent): with prob crouch_prob a reset samples a continuous
  // depth along CROUCH_ROWS (physics-validated squat synergy:
  // supported-survival >= the standing reset's own 21/60 under the
  // same probe controller; generated 2026-08-24, scratchpad
  // gen_crouch_table.py). Jitter then applies CENTERED on the crouch
  // pose; the sole-corner root drop makes any pose ground-exact.
  T crouch_prob;
  int gears;
  bool auto_reset;
  // contact_imp/contact_pt are pure introspection (parity harness /
  // CudaBeatVecEnv.contacts()); nothing in the kernel reads them back.
  // Off by default so the training path skips the 55-wide zeroing and the
  // uncoalesced per-contact read-modify-writes. Baked per instance at
  // construction => constant across launches => CUDA-graph safe.
  bool want_contacts;
};

// FK caches from the current state (lib.rs Body::refresh_fk): wrist =
// hand-frame origin, blade = hand-frame +z column, live head/shoulders.
template <typename T>
__device__ void refresh_fk(Dyn<T>& __restrict__ d, const Sim<T>& P, int e) {
  fk_all(d);
  const int fw[2] = {F_L_WRIST, F_R_WRIST};
  const int fh[2] = {F_L_HAND, F_R_HAND};
  for (int s = 0; s < 2; s++) {
    for (int k = 0; k < 3; k++) {
      P.wrist[(e * 2 + s) * 3 + k] = d.pos[fw[s]][k];
      P.blade[(e * 2 + s) * 3 + k] = d.r[fh[s]][k][2];
    }
  }
  for (int k = 0; k < 3; k++) P.head[e * 3 + k] = d.pos[F_HEAD][k];
  P.shoulder_z[e * 2 + 0] = d.pos[F_L_SHOULDER][2];
  P.shoulder_z[e * 2 + 1] = d.pos[F_R_SHOULDER][2];
}

template <typename T>
__device__ void load_state(Dyn<T>& __restrict__ d, const Sim<T>& P, int e) {
  for (int i = 0; i < 3; i++) d.p[i] = P.p[e * 3 + i];
  for (int i = 0; i < 4; i++) d.quat[i] = P.quat[e * 4 + i];
  for (int j = 0; j < NQ; j++) d.q[j] = P.q[e * NQ + j];
  for (int i = 0; i < NV; i++) d.v[i] = P.v[e * NV + i];
}

template <typename T>
__device__ void store_state(const Dyn<T>& __restrict__ d, const Sim<T>& P,
                            int e) {
  for (int i = 0; i < 3; i++) P.p[e * 3 + i] = d.p[i];
  for (int i = 0; i < 4; i++) P.quat[e * 4 + i] = d.quat[i];
  for (int j = 0; j < NQ; j++) P.q[e * NQ + j] = d.q[j];
  for (int i = 0; i < NV; i++) P.v[e * NV + i] = d.v[i];
}

// Reset env e to the neutral state with a fresh chart (kernel phase E).

// env v38 crouch ladder: (spine_p, hip_p, knee, ankle_p) f64 radians,
// knee 0.3..2.1; root y 0.921..0.393. Values from the balance search
// (c1/c2/c4 synergy params) — regenerate with gen_crouch_table.py.
// 2026-09-06: rows 8-10 (knee 1.7/1.9/2.1, spine-leaning, chest tilt
// <= 32 deg) take the head to 1.30/1.23/1.06 m so the deep end of the
// ladder clears the 1.20 m forced-duck wall bottom. Mirrors
// constants.CROUCH_START_ROWS verbatim (test_crouch_table pins it).
#define N_CROUCH 10
__device__ __constant__ double CROUCH_ROWS[N_CROUCH][4] = {
    {-0.03, 0.10499999999999998, 0.3, 0.195},
    {-0.05, 0.325, 0.5, 0.175},
    {0.0, 0.45499999999999996, 0.7, 0.24499999999999997},
    {0.09000000000000001, 0.585, 0.9, 0.315},
    {0.0, 0.605, 1.1, 0.49500000000000005},
    {0.0, 0.6891347618019847, 1.3, 0.6108652381980153},
    {0.15000000000000002, 0.8891347618019847, 1.5, 0.6108652381980153},
    {0.51, 1.0891347618019847, 1.7, 0.6108652381980153},
    {0.475, 1.3399999999999999, 1.9, 0.56},
    {0.525, 1.7850000000000001, 2.1, 0.315},
};
// bench combo walk (lib.rs ComboWalk::consume, applied at the event
// sites): break = slow / angle / wrong-hand / miss / bomb hit / wall
// ENTRY; cut = promote-before-commit, pin the multiplier, combo up,
// score += pts x mult; follow-through points commit at the pinned
// multipliers of the tracker's notes (tr_msum). Rust rounds pts to f32
// on the event bus before multiplying — mirrored with the (float) cast.
constexpr int BENCH_REC = 28;   // + [wrev_L, wrev_R, ticks] (2026-09-22)
// review B09 (2026-09-30): + [wall_enters, term_flags] (cols 17/18, the
// lib.rs VecBeatSim::term mirror) and col 19 = this level's running
// wall-ENTRY count (the lib.rs wall_enters counter). The accumulator
// lives in the slot's own bench_rec row instead of a new (N) tensor so
// the kernel argument list is untouched: bumped at the two entry sites,
// copied to col 17 at done, zeroed by reset_env.
// 2026-10-01 (user): + this level's applied-torque totals, cols 19..22 =
// Σ over ticks of [total, spine, arms, legs] |τ| (N·m; the tau_stat /
// lib.rs pop_torque_env quantity), snapshotted at done from the running
// accumulators at cols 24..27 (bumped next to tau_stat each tick,
// zeroed by reset_env). The wall-entry accumulator moved 19 -> 23. The
// host reads cols 0..BENCH_COLS-1 = 0..22 only (cuda_env.BENCH_COLS).
constexpr int BENCH_TAU_OUT = 19;
constexpr int BENCH_WALL_ACC = 23;
constexpr int BENCH_TAU_ACC = 24;
// term_flags bits (lib.rs VecBeatSim::term): 0 = the level itself ended
constexpr int TERM_FELL = 1, TERM_TILT = 2, TERM_ENVELOPE = 4,
              TERM_TICKCAP = 8;

template <typename T>
__device__ __forceinline__ void bench_wall_enter(const Sim<T>& P, int e) {
  if (P.bench_refill) P.bench_rec[(size_t)e * BENCH_REC + BENCH_WALL_ACC] += T(1);
}

template <typename T>
__device__ __forceinline__ void cw_break(const Sim<T>& P, int e) {
  if (!P.combo_walk) return;
  P.cw_combo[e] = 0;
  P.cw_breaks[e] += 1;
  P.cw_mult[e] = max(1, P.cw_mult[e] / 2);
  P.cw_prog[e] = 0;
}

template <typename T>
__device__ __forceinline__ int cw_cut(const Sim<T>& P, int e, int i, T pts) {
  if (!P.combo_walk) return 0;
  int m = P.cw_mult[e];
  const int need = (m == 1) ? 2 : (m == 2) ? 4 : (m == 4) ? 8 : 0;
  if (need) {
    P.cw_prog[e] += 1;
    if (P.cw_prog[e] >= need) {
      m *= 2;
      P.cw_mult[e] = m;
      P.cw_prog[e] = 0;
    }
  }
  P.cw_note_mult[(size_t)e * P.C + i] = (uint8_t)m;
  P.cw_combo[e] += 1;
  if (P.cw_combo[e] > P.cw_maxc[e]) P.cw_maxc[e] = P.cw_combo[e];
  P.cw_score[e] += T((float)pts) * T(m);
  return m;
}

// env v24: seed the body points' finite-diff base at the current FK
// pose (d.pos), velocities zero — reset and set_state share it (lib.rs
// Body reset / set_state mirror). Pure copies and zero stores, no
// floating-point arithmetic, so inlining it cannot change FMA
// contraction or any bit of the result.
template <typename T>
__device__ __forceinline__ void seed_body_pts(Dyn<T>& __restrict__ d,
                                              const Sim<T>& P, int e) {
  for (int i = 0; i < N_BODY_PTS; i++)
    for (int k = 0; k < 3; k++) {
      const size_t idx = ((size_t)e * N_BODY_PTS + i) * 3 + k;
      P.body_pt[idx] = d.pos[body_pt_frame(i)][k];
      P.body_pt_vel[idx] = T(0);
    }
}

template <typename T>
__device__ void reset_env(Dyn<T>& __restrict__ d, const Sim<T>& P, int e,
                          int64_t chart) {
  d.p[0] = T(BASE_POS0[0]);
  d.p[1] = T(BASE_POS0[1]);
  d.p[2] = T(BASE_POS0[2]);
  d.quat[0] = T(1);
  d.quat[1] = T(0);
  d.quat[2] = T(0);
  d.quat[3] = T(0);
  for (int j = 0; j < NQ; j++) d.q[j] = T(0);
  // env v20 (user, 2026-08-16): episode-start ARM pose — real-player
  // ready stance (lib.rs Q_START_ARM mirror, exact same f64 literals)
  d.q[3] = T(-1.3962634015954636);   // l_arm_pitch  -80 deg
  d.q[6] = T(1.7453292519943295);    // l_elbow      100 deg
  d.q[8] = T(1.2217304763960306);    // l_wrist_bend  70 deg
  d.q[10] = T(-1.3962634015954636);  // r_arm_pitch
  d.q[13] = T(1.7453292519943295);   // r_elbow
  d.q[15] = T(1.2217304763960306);   // r_wrist_bend
  for (int i = 0; i < NV; i++) d.v[i] = T(0);
  // env v21: wide stance, then OPTIONAL v20
  // start-pose jitter CENTERED on the stance (12 leg joints, §1.4
  // indices 17..28; one-way knees take |draw|), clamped to the joint
  // ranges. ALWAYS finish with the exact no-clip fix: FK the pose,
  // transform the 8 sole corners (contact spec §2 GROUND_CORNERS),
  // and drop/lift the root so the lowest corner sits exactly on the
  // ground plane (y = 0) — the stance alone lifts the ankles ~8.5 mm.
  {
    d.q[19] = T(-Q_START_HIP_ROLL);  // l_hip_roll
    d.q[22] = T(Q_START_HIP_ROLL);   // l_ankle_roll (counter)
    d.q[25] = T(Q_START_HIP_ROLL);   // r_hip_roll
    d.q[28] = T(-Q_START_HIP_ROLL);  // r_ankle_roll (counter)
    if (P.crouch_prob > T(0)
        && (P.pose_jit[e * 14 + 12] + T(1)) * T(0.5) < P.crouch_prob) {
      const T u = (P.pose_jit[e * 14 + 13] + T(1)) * T(0.5)
          * T(N_CROUCH - 1);
      int i0 = (int)u;
      if (i0 > N_CROUCH - 2) i0 = N_CROUCH - 2;
      const T f = u - T(i0);
      T row[4];
      for (int k = 0; k < 4; k++)
        row[k] = T(CROUCH_ROWS[i0][k]) * (T(1) - f)
            + T(CROUCH_ROWS[i0 + 1][k]) * f;
      d.q[0] = row[0];
      d.q[17] = row[1]; d.q[23] = row[1];
      d.q[20] = row[2]; d.q[26] = row[2];
      d.q[21] = row[3]; d.q[27] = row[3];
    }
    if (P.start_pose_jitter > T(0)) {
      for (int k = 0; k < 12; k++) {
        const int j = 17 + k;
        T jit = P.pose_jit[e * 14 + k] * P.start_pose_jitter;
        if (j == 20 || j == 26) jit = fabs(jit);
        d.q[j] = tclamp(d.q[j] + jit, CT(LIMIT_LO, [j]), CT(LIMIT_HI, [j]));
      }
    }
    fk_all(d);
    T min_y = T(1e9);
    for (int f = 0; f < 2; f++) {
      const int fr = GROUND_FRAMES[f];
      for (int c = 0; c < CORNERS_PER_FOOT; c++) {
        T y = d.pos[fr][1];
        for (int k = 0; k < 3; k++)
          y += d.r[fr][1][k] * CT(GROUND_CORNERS, [c][k]);
        min_y = tmin(min_y, y);
      }
    }
    d.p[1] -= min_y;
  }
  // review B10: the start-jitter kick — the root's horizontal velocity,
  // installed before store_state so P.v and the reset obs (write_obs
  // runs after this) carry it. A pure store over the zeroed d.v (the
  // pose block above reads no velocity), bit-identical to the pre-B10
  // wrapper's post-launch `v += dv` on a zero v. Zeros when the
  // wrapper's start_jitter == 0 (parity / bench configs).
  const T kick_x = P.start_kick[e * 2];
  const T kick_z = P.start_kick[e * 2 + 1];
  d.v[0] = kick_x;
  d.v[2] = kick_z;
  store_state(d, P, e);
  refresh_fk(d, P, e);
  // env v24: seed the finite-diff base with the reset pose's §8
  // points, velocities zero (lib.rs Body reset mirror)
  seed_body_pts(d, P, e);
  // review B10: with every joint rate and the root's angular velocity
  // zero, the kick is a rigid translation, so the TRUE velocity of
  // every body point, wrist and palm is exactly (kick_x, 0, kick_z) —
  // the finite-diff velocity lanes start there instead of 0 (all zero
  // without a kick: the lib.rs Body reset values). The next tick's
  // finite difference overwrites them from the position bases, which
  // the kick does not touch.
  for (int i = 0; i < N_BODY_PTS; i++) {
    P.body_pt_vel[((size_t)e * N_BODY_PTS + i) * 3] = kick_x;
    P.body_pt_vel[((size_t)e * N_BODY_PTS + i) * 3 + 2] = kick_z;
  }
  for (int s = 0; s < 2; s++) {
    P.wrist_vel[(e * 2 + s) * 3] = kick_x;
    P.wrist_vel[(e * 2 + s) * 3 + 1] = T(0);
    P.wrist_vel[(e * 2 + s) * 3 + 2] = kick_z;
    P.palm_vel[(e * 2 + s) * 3] = kick_x;
    P.palm_vel[(e * 2 + s) * 3 + 1] = T(0);
    P.palm_vel[(e * 2 + s) * 3 + 2] = kick_z;
    // env v49 tip lanes: the same rigid translation (0 without a kick)
    P.tip_vel[(e * 2 + s) * 3] = kick_x;
    P.tip_vel[(e * 2 + s) * 3 + 1] = T(0);
    P.tip_vel[(e * 2 + s) * 3 + 2] = kick_z;
    P.swing_acc[e * 2 + s] = T(0);
    P.axis_valid[e * 2 + s] = false;
    P.tr_ptr[e * 2 + s] = 0;
    for (int t = 0; t < N_TRACKERS; t++)
      P.tr_active[(e * 2 + s) * N_TRACKERS + t] = false;
    P.contact_arm[e * 2 + s] = false;
  }
  if (P.want_contacts)
    for (int pr = 0; pr < NUM_PAIRS; pr++)
      P.contact_imp[e * NUM_PAIRS + pr] = T(0);
  for (int k = 0; k < N_GROUND_IMP_LANES; k++)
    P.ground_imp[e * N_GROUND_IMP_LANES + k] = T(0);
  P.foot_pot[e] = T(0);
  P.chart_idx[e] = chart;
  P.tick[e] = 0;
  P.first_pending[e] = 0;
  P.ep_return[e] = T(0);
  P.good[e] = 0;
  P.bad[e] = 0;
  P.miss[e] = 0;
  P.bombs[e] = 0;
  P.score[e] = T(0);
  P.score_pre[e] = T(0);
  P.score_acc[e] = T(0);
  P.score_post[e] = T(0);
  P.wrev_dir[e * 2] = 0;       // wrist-reversal detector re-arms on the
  P.wrev_dir[e * 2 + 1] = 0;   // new episode's first tick (count kept)
  for (int i = 0; i < P.C; i++) P.status[e * P.C + i] = 0;
  if (P.combo_walk) {  // bench combo walk: fresh walk per level
    P.cw_mult[e] = 1;
    P.cw_prog[e] = 0;
    P.cw_combo[e] = 0;
    P.cw_maxc[e] = 0;
    P.cw_breaks[e] = 0;
    P.cw_score[e] = T(0);
    for (int i = 0; i < P.C; i++) P.cw_note_mult[(size_t)e * P.C + i] = 0;
  }
  for (int s = 0; s < 2; s++) {  // env v45
    P.prev_idx[e * 2 + s] = 0;
    P.prev_valid[e * 2 + s] = 0;
  }
  // env v46: fresh body for the new map — one scale per axis, both
  // sides share it; only when randomization is on (set_tau_scale's
  // explicit values persist otherwise)
  // env v47b (2026-09-19, --tau-center): the wrapper pre-computes the
  // whole draw, center[a] * (1 + tau_rand * U(-1,1)), into tau_jit — the
  // kernel just installs it (v46 computed 1 + r*u here; same draws up to
  // the host's separate mul/add rounding vs a possible FMA)
  if (P.tau_rand > T(0))
    for (int a = 0; a < TAU_FEAT; a++)
      P.tau_scale[e * TAU_FEAT + a] = P.tau_jit[e * TAU_FEAT + a];
  // env v17 wall state
  P.first_wall[e] = 0;
  P.walls_in[e] = 0;
  P.in_wall_prev[e] = false;
  if (P.bench_refill) {
    T* r = P.bench_rec + (size_t)e * BENCH_REC;
    r[BENCH_WALL_ACC] = T(0);
    for (int k = 0; k < 4; k++) r[BENCH_TAU_ACC + k] = T(0);
  }
  // arm-after-first-plant (user 2026-08-20): start UNARMED so a foot
  // that begins the episode in the air (start-pose jitter) is never
  // charged — the lift edge only arms once the foot has actually been
  // seen grounded.
  P.foot_down_prev[e * 2] = false;
  P.foot_down_prev[e * 2 + 1] = false;
  for (int i = 0; i < P.W; i++) P.wall_passed[e * P.W + i] = false;
}

// Observation writer (lib.rs write_obs, SPEC-v10 §5) — from the CURRENT
// tensors (post-reset state included).
//
// K10e probed three redundancies here; ONE landed. All three rest on the
// same audit of the three paths that reach here (step_kernel's tail is
// the only call site): mode 1 reset_env -> {writes d, store_state,
// refresh_fk}; mode 2 {load_state, refresh_fk}; mode 0 step_env ->
// {load_state, 8 substeps, store_state, refresh_fk, judgment, optional
// reset_env}.
//  1. REVERTED at -0.11% (4 ABBA rounds, N=4096). The re-load_state
//     below IS redundant — on every path d.p / d.quat / d.q / d.v
//     already ARE the current tensors (modes 1/2 wrote d from or to
//     them immediately before; step_env's store_state copied d OUT, and
//     the judgment block that follows touches no d field but d.env_bad,
//     with a terminal auto-reset re-establishing the invariant through
//     reset_env). Deleting it nonetheless COST time and grew the f64
//     frame 90,976 -> 91,008 B (spills 11,756/9,628 -> 11,812/9,696):
//     the 71-word reload is a rematerialization point, and without it
//     the state has to stay live across the whole judgment block. Same
//     lesson as A1/1f — on this kernel a pure traffic cut with no new
//     dependency chain does not pay, and can lose.
//  2. REVERTED, and this one is a trap worth recording. fk_all's first
//     act is quat_to_rot(d.quat, d.r0) and every path above ends in a
//     refresh_fk, so d.r0 IS semantically the rotation recomputed
//     below — but it is NOT bit-identical in f32: nvcc contracts
//     quat_to_rot's multiply-adds differently when the function is
//     inlined into fk_all than when it is inlined here, and the probe
//     caught obs lanes 3, 5 and 8 (r0[0][0], r0[2][0], r0[2][1]) moving
//     by 1 ulp (5.96e-8) on every f32 suite. f64 was clean, which is
//     exactly why a single-dtype gate would have shipped it. The
//     recompute stays. GENERAL RULE: "the same expression tree is
//     deterministic" is only a bitwise argument at a FIXED inline site.
//  3. KEPT, +0.08%: the zero fill starts at OFF_NOTES. Lanes 0..101 are
//     all UNCONDITIONALLY overwritten below — 0..2 root pos, 3..8 the 6D
//     rotation, 9..14 root vlin/vang, 15..45 the q block (29 joints,
//     j = 7 and 14 emitting two lanes each = 31), 46..74 qd, 75..96 the
//     two 11-lane hand blocks (env v19), 97..99 head, 100..101 the
//     contact bits — so their pre-zeroing was dead stores per env per
//     tick (the static_assert pins the prefix gapless). From OFF_NOTES
//     the fill IS load-bearing: note features and the mask are written
//     only for occupied slots.
static_assert(OFF_ROOT_ROT == 3 && OFF_ROOT_VLIN == 9 && OFF_ROOT_VANG == 12
                  && OFF_Q == 15 && OFF_QD == OFF_Q + NQ + 2
                  && OFF_HAND_L == OFF_QD + NQ && OFF_HAND_R == OFF_HAND_L + 11
                  && OFF_HEAD == OFF_HAND_R + 11 && OFF_CONTACT == OFF_HEAD + 3
                  && OFF_NOTES == OFF_CONTACT + 2,
              "K10e: obs lanes 0..OFF_NOTES-1 must be a gapless "
              "unconditionally-written prefix");
// FX = Flux game mode (compile-time, host-selected per instance): the hand blocks carry the palms and the entity window the Flux
// lane meanings (lib.rs write_flux_hands / write_flux_entities).
template <typename T, bool FX>
__device__ void write_obs(Dyn<T>& __restrict__ d, const Sim<T>& P, int e) {
  float* __restrict__ obs = P.obs + (size_t)e * OBS_DIM;
  for (int i = OFF_NOTES; i < OBS_DIM; i++) obs[i] = 0.f;
  load_state(d, P, e);
  // root position
  obs[OFF_ROOT_POS] = (float)(d.p[0] / T(X_NORM));
  obs[OFF_ROOT_POS + 1] = (float)((d.p[1] - T(ROOT_Y0)) / T(Y_NORM));
  obs[OFF_ROOT_POS + 2] = (float)((d.p[2] - T(PLANE_Z0)) / T(X_NORM));
  // 6D two-column rotation (recomputed on purpose — see K10e note 2)
  T r0[3][3];
  quat_to_rot(d.quat, r0);
  for (int i = 0; i < 3; i++) {
    obs[OFF_ROOT_ROT + i] = (float)r0[i][0];
    obs[OFF_ROOT_ROT + 3 + i] = (float)r0[i][1];
  }
  for (int k = 0; k < 3; k++) {
    obs[OFF_ROOT_VLIN + k] = (float)(d.v[k] / T(ROOT_VLIN_NORM));
    obs[OFF_ROOT_VANG + k] = (float)(d.v[3 + k] / T(ROOT_VANG_NORM));
  }
  // q block (31 lanes; §1.4 order; wrist rolls cos/sin)
  int lane = OFF_Q;
  for (int j = 0; j < NQ; j++) {
    const T qj = d.q[j];
    if (j == 7 || j == 14) {
      obs[lane] = (float)cos(qj);
      obs[lane + 1] = (float)sin(qj);
      lane += 2;
    } else if (j == 6 || j == 13 || j == 20 || j == 26) {
      const T hi = CT(LIMIT_HI, [j]);
      obs[lane] = (float)((T(2) * qj - hi) / hi);
      lane += 1;
    } else {
      const T lo = CT(LIMIT_LO, [j]), hi = CT(LIMIT_HI, [j]);
      obs[lane] = (float)(qj / tmax(fabs(lo), fabs(hi)));
      lane += 1;
    }
  }
  for (int j = 0; j < NQ; j++)
    obs[OFF_QD + j] = (float)(d.v[VOFF + j] / CT(OMEGA_CAP, [j]));
  // hand derived blocks
  const int bases[2] = {OFF_HAND_L, OFF_HAND_R};
  if constexpr (FX) {
    // Flux: palm x/y head-relative, palm z world, palm velocity
    // /V_NORM, hand axis; lanes 9..10 zero — they sit in the
    // unconditionally-written prefix (K10e note 3), so write the zeros
    for (int s = 0; s < 2; s++) {
      const int base = bases[s];
      const int wb = (e * 2 + s) * 3;
      T pm[3];
      for (int k = 0; k < 3; k++)
        pm[k] = P.wrist[wb + k] + T(PALM_OFF) * P.blade[wb + k];
      obs[base] = (float)((pm[0] - P.head[e * 3]) / T(X_NORM));
      obs[base + 1] = (float)((pm[1] - P.head[e * 3 + 1]) / T(Y_NORM));
      obs[base + 2] = (float)pm[2];
      for (int k = 0; k < 3; k++) {
        obs[base + 3 + k] = (float)(P.palm_vel[wb + k] / T(V_NORM));
        obs[base + 6 + k] = (float)P.blade[wb + k];
      }
      obs[base + 9] = 0.f;
      obs[base + 10] = 0.f;
    }
  } else
  for (int s = 0; s < 2; s++) {
    const int base = bases[s];
    const int wb = (e * 2 + s) * 3;
    // env v38: wrist x/y head-relative (lib.rs mirror; z stays world
    // — already consistent with the entity z lanes)
    obs[base] = (float)((P.wrist[wb] - P.head[e * 3]) / T(X_NORM));
    obs[base + 1] = (float)((P.wrist[wb + 1] - P.head[e * 3 + 1])
                            / T(Y_NORM));
    obs[base + 2] = (float)P.wrist[wb + 2];
    for (int k = 0; k < 3; k++) {
      obs[base + 3 + k] = (float)(P.wrist_vel[wb + k] / T(V_NORM));
      obs[base + 6 + k] = (float)P.blade[wb + k];
    }
    obs[base + 9] =
        (float)tmin(P.swing_acc[e * 2 + s] / T(PRE_FULL), T(2));
    // env v19: remaining follow-through credit — Σ over live trackers
    // of cnt · (1 − min(acc/POST_FULL, 1)), clipped to 2. Ring walked
    // from tr_ptr with cnt-many single additions so the f-p sum order
    // matches lib.rs' per-note Vec (spawn order) bitwise.
    T rem = T(0);
    const int rptr = P.tr_ptr[e * 2 + s];
    for (int kk = 0; kk < N_TRACKERS; kk++) {
      const int rb = (e * 2 + s) * N_TRACKERS + (rptr + kk) % N_TRACKERS;
      if (!P.tr_active[rb]) continue;
      const T term = T(1) - tmin(P.tr_acc[rb] / T(POST_FULL), T(1));
      for (int cix = P.tr_cnt[rb]; cix > 0; cix--) rem += term;
    }
    obs[base + 10] = (float)tmin(rem, T(2));
  }
  // head + contact bits
  obs[OFF_HEAD] = (float)(P.head[e * 3] / T(X_NORM));
  obs[OFF_HEAD + 1] = (float)((P.head[e * 3 + 1] - T(HEAD_Y0)) / T(Y_NORM));
  obs[OFF_HEAD + 2] = (float)((P.head[e * 3 + 2] - T(PLANE_Z0)) / T(X_NORM));
  obs[OFF_CONTACT] = P.contact_arm[e * 2] ? 1.f : 0.f;
  obs[OFF_CONTACT + 1] = P.contact_arm[e * 2 + 1] ? 1.f : 0.f;
  // entity window (env v17, spec §12c/d) — lib.rs write_obs verbatim:
  // 16 SHARED slots, notes + walls, nearest-first by 3D head distance;
  // behind-the-head entities excluded; ties keep the first-scanned
  // entity (strict-< displacement), scan = notes in chart order then
  // walls (§5 parity: identical comparison order).
  const int64_t cc = P.chart_idx[e];
  const int C = P.C;
  const int nn = (int)P.p_n_notes[cc];
  const int64_t nb = P.p_note_off[cc];   // ragged pool: chart cc's base
  const int64_t wbase = P.p_wall_off[cc];
  const T time_o = T(P.tick[e]) * T(DT);
  // v15 head-anchored track (contact spec §10)
  const T z_off = P.head[e * 3 + 2] - T(PLANE_Z0);
  // 2026-08-22 (user): the z-lane low clamp is the LIVE head z — the
  // exact existence boundary (miss/pass/wall removal all reduce to
  // lane < head_z), so no live object's z is ever flattened, under
  // any lean. (Constant floors -3.0 / 0.0 / -0.65 all predate this.)
  const T z_lo = P.head[e * 3 + 2];
  const T hx = P.head[e * 3], hy = P.head[e * 3 + 1];
  T sel_d[N_NOTE_SLOTS];
  bool sel_wall[N_NOTE_SLOTS];
  int sel_idx[N_NOTE_SLOTS];
  int n_sel = 0;
  auto insert = [&](T d2, bool is_wall, int idx) {
    if (n_sel == N_NOTE_SLOTS && d2 >= sel_d[N_NOTE_SLOTS - 1]) return;
    int i = n_sel < N_NOTE_SLOTS - 1 ? n_sel : N_NOTE_SLOTS - 1;
    for (; i > 0 && d2 < sel_d[i - 1]; i--) {
      sel_d[i] = sel_d[i - 1];
      sel_wall[i] = sel_wall[i - 1];
      sel_idx[i] = sel_idx[i - 1];
    }
    sel_d[i] = d2;
    sel_wall[i] = is_wall;
    sel_idx[i] = idx;
    if (n_sel < N_NOTE_SLOTS) n_sel++;
  };
  const T msp = P.p_min_speed[cc];
  for (int i = P.first_pending[e]; i < nn; i++) {
    const int64_t cb = nb + i;
    if (!P.p_valid[cb] || P.status[e * C + i] != 0) continue;
    // sorted-chart early exit (lib.rs): the z lower bound alone beats
    // the worst kept distance for this and every later note
    const T dz_lb = (P.p_hit[cb] - time_o) * msp - T(PLANE_Z0);
    if (n_sel == N_NOTE_SLOTS && dz_lb > T(0)
        && dz_lb * dz_lb >= sel_d[N_NOTE_SLOTS - 1])
      break;
    const T dz = (P.p_hit[cb] - time_o) * P.p_speed[cb] - T(PLANE_Z0);
    if (dz < T(0)) continue;  // behind the head (past the miss boundary)
    const T dx = P.p_center[cb * 2] - hx;
    const T dy = P.p_center[cb * 2 + 1] - hy;
    insert(dx * dx + dy * dy + dz * dz, false, i);
  }
  const int nw = (int)P.p_n_walls[cc];
  for (int wi = P.first_wall[e]; wi < nw; wi++) {
    const int64_t wb = wbase + wi;
    if (!P.p_wall_valid[wb] || P.wall_passed[e * P.W + wi]) continue;
    const T dzf = (P.p_wall_t[wb] - time_o) * P.p_wall_speed[wb]
        - T(PLANE_Z0) - T(WALL_FRONT_OFF);
    const T len = P.p_wall_len[wb];
    if (dzf + len < T(0)) continue;  // back face behind the head
    const T sx = tmax(fabs(hx - P.p_wall_cx[wb]) - P.p_wall_w[wb] * T(0.5), T(0));
    const T sy = tmax(fabs(hy - P.p_wall_cy[wb]) - P.p_wall_h[wb] * T(0.5), T(0));
    const T sz = tmax(tmax(dzf, -(dzf + len)), T(0));
    insert(sx * sx + sy * sy + sz * sz, true, wi);
  }
  for (int slot = 0; slot < n_sel; slot++) {
    float* nb2 = obs + OFF_NOTES + slot * NOTE_FEAT;
    if (sel_wall[slot]) {
      const int64_t wb = wbase + sel_idx[slot];
      const T zfu = (P.p_wall_t[wb] - time_o) * P.p_wall_speed[wb]
          - T(WALL_FRONT_OFF) + z_off;
      // env v38 (user): entity x/y are HEAD-RELATIVE — the near-
      // constant standing head height let the encoder learn an
      // absolute-coordinate shortcut that mis-generalizes the moment
      // the head moves (posture-invariance / duck-amnesia fix).
      // Mechanics untouched; head pose itself stays in obs. z was
      // already head-anchored via z_off.
      nb2[0] = (float)((P.p_wall_cx[wb] - P.head[e * 3]) / T(X_NORM));
      nb2[1] = (float)((P.p_wall_cy[wb] - P.head[e * 3 + 1]) / T(Y_NORM));
      nb2[2] = (float)z_lane(zfu, z_lo, P.z_asinh);  // env v48
      nb2[3] = (float)(-P.p_wall_speed[wb] / T(Z_SCALE));
      // lanes 4..13 (color/dir/kind/free/ex-valid) stay zero
      nb2[14] = 1.f;  // is_wall
      nb2[15] = (float)(P.p_wall_w[wb] / T(X_NORM));
      nb2[16] = (float)(P.p_wall_h[wb] / T(Y_NORM));
      // env v36: lane 17 = BACK-face z (wall end), encoded like lane
      // 2 (floored at the live head z) — exact for any wall length. An
      // observable wall's back face is ahead of the head, so the floor
      // clips nothing live — it only bounds the lane on the removal tick. (Lived in
      // lane 4 for one commit; moved here 2026-08-22 so no lane is
      // binary-for-notes / continuous-for-walls under the shared
      // token encoder. Lane 17 held wall length until the same day.)
      nb2[17] = (float)z_lane(zfu + P.p_wall_len[wb], z_lo, P.z_asinh);  // env v48
    } else if constexpr (FX) {
      // Flux entity lanes (lib.rs write_flux_entities; Flux pools carry
      // no walls, so the selection above is exactly Rust's notes-only
      // scan): 0-3 as standard, 4/5 circle colour, 8 is_circle; bars
      // 6/7 axis u, 9 is_bar, 10 length, 11 sweet offset, 12/13 the
      // sweet spot head-relative; 14-17 zero
      const int64_t cb = nb + sel_idx[slot];
      const T zu = (P.p_hit[cb] - time_o) * P.p_speed[cb] + z_off;
      const T nx = P.p_center[cb * 2], ny = P.p_center[cb * 2 + 1];
      nb2[0] = (float)((nx - P.head[e * 3]) / T(X_NORM));
      nb2[1] = (float)((ny - P.head[e * 3 + 1]) / T(Y_NORM));
      nb2[2] = (float)z_lane(zu, z_lo, P.z_asinh);
      nb2[3] = (float)(-P.p_speed[cb] / T(Z_SCALE));
      const T blen = P.p_len[cb];
      if (!(blen > T(0))) {  // circle
        nb2[P.p_is_right[cb] ? 5 : 4] = 1.f;
        nb2[8] = 1.f;
      } else {               // bar: p_rot = (cos θ, sin θ)
        const T ux = -P.p_rot[cb * 2 + 1], uy = P.p_rot[cb * 2];
        const T sw = P.p_sweet[cb];
        const T s_sweet = sw * T(0.5) * blen;
        nb2[6] = (float)ux;
        nb2[7] = (float)uy;
        nb2[9] = 1.f;
        nb2[10] = (float)blen;
        nb2[11] = (float)sw;
        nb2[12] = (float)((nx + s_sweet * ux - P.head[e * 3]) / T(X_NORM));
        nb2[13] = (float)((ny + s_sweet * uy - P.head[e * 3 + 1])
                          / T(Y_NORM));
      }
    } else {
      const int64_t cb = nb + sel_idx[slot];
      const T zu = (P.p_hit[cb] - time_o) * P.p_speed[cb] + z_off;
      const bool bomb = P.p_is_bomb[cb];
      // env v38: head-relative x/y (see wall branch note)
      nb2[0] = (float)((P.p_center[cb * 2] - P.head[e * 3]) / T(X_NORM));
      nb2[1] = (float)((P.p_center[cb * 2 + 1] - P.head[e * 3 + 1])
                       / T(Y_NORM));
      nb2[2] = (float)z_lane(zu, z_lo, P.z_asinh);  // env v48
      nb2[3] = (float)(-P.p_speed[cb] / T(Z_SCALE));
      nb2[4] = (!bomb && !P.p_is_right[cb]) ? 1.f : 0.f;
      nb2[5] = (!bomb && P.p_is_right[cb]) ? 1.f : 0.f;
      nb2[6] = P.p_show_dir[cb] ? (float)P.p_dvec[cb * 2] : 0.f;
      nb2[7] = P.p_show_dir[cb] ? (float)P.p_dvec[cb * 2 + 1] : 0.f;
      nb2[8] = bomb ? 1.f : 0.f;
      nb2[9] = P.p_is_head[cb] ? 1.f : 0.f;
      nb2[10] = P.p_is_link[cb] ? 1.f : 0.f;
      nb2[11] = P.p_free_pre[cb] ? 1.f : 0.f;
      nb2[12] = P.p_free_post[cb] ? 1.f : 0.f;
      // lanes 13..17 stay zero. Lane 13 (in-token valid) retired
      // 2026-08-22: it exactly duplicated the OFF_NOTE_MASK block,
      // which gates attention — so it read constant 1 on every token
      // the policy could see. The mask block stays.
    }
    obs[OFF_NOTE_MASK + slot] = 1.f;
  }

  // contact spec §7: per-foot ground-contact lanes [L_heel, L_toe,
  // R_heel, R_toe]
  for (int k = 0; k < N_GROUND_IMP_LANES; k++)
    obs[OFF_GROUND_IMP + k] = (float)(
        P.ground_imp[(size_t)e * N_GROUND_IMP_LANES + k] /
        T(GROUND_IMP_NORM));

  // contact spec §8 (env v13): elbow/knee/foot world positions from
  // d.pos — refresh_fk ran fk_all in every mode before this call
  {
    const double bp_y0[N_BODY_PTS] = {Y_CENTER, Y_CENTER, KNEE_Y0,
                                      KNEE_Y0, FOOT_Y0, FOOT_Y0};
    for (int i = 0; i < N_BODY_PTS; i++) {
      const int base = OFF_BODY_PTS + 3 * i;
      const int f = body_pt_frame(i);
      obs[base] = (float)(d.pos[f][0] / T(X_NORM));
      obs[base + 1] = (float)((d.pos[f][1] - T(bp_y0[i])) / T(Y_NORM));
      obs[base + 2] = (float)((d.pos[f][2] - T(PLANE_Z0)) / T(X_NORM));
    }
  }
  // env v24: the same points' world velocities (lib.rs mirror —
  // finite diff across the last tick, /V_NORM; the tensor is zeroed
  // at reset and after a mode-2 refresh)
  for (int i = 0; i < N_BODY_PTS; i++) {
    const int base = OFF_BODY_PT_VEL + 3 * i;
    for (int k = 0; k < 3; k++)
      obs[base + k] = (float)(
          P.body_pt_vel[((size_t)e * N_BODY_PTS + i) * 3 + k] / T(V_NORM));
  }
  // env v45: per-hand previous-note blocks (by note colour) — lib.rs
  // write_obs loop transcribed (zero until a note of that colour has
  // resolved; x/y against the LIVE head)
  {
    const int pbases[2] = {OFF_PREV_L, OFF_PREV_R};
    for (int s = 0; s < 2; s++) {
      const int ps = e * 2 + s;
      if (!P.prev_valid[ps]) continue;
      const int base = pbases[s];
      const int64_t cb = nb + P.prev_idx[ps];
      obs[base] = (float)((P.p_center[cb * 2] - P.head[e * 3]) / T(X_NORM));
      obs[base + 1] = (float)((P.p_center[cb * 2 + 1] - P.head[e * 3 + 1])
                              / T(Y_NORM));
      if (P.p_show_dir[cb]) {
        obs[base + 2] = (float)P.p_dvec[cb * 2];
        obs[base + 3] = (float)P.p_dvec[cb * 2 + 1];
      }
      obs[base + 4] = 1.f;
    }
  }
  // env v46: torque-limit scale lanes (lib.rs write_obs mirror)
  for (int a = 0; a < TAU_FEAT; a++)
    obs[OFF_TAU + a] = (float)(((double)P.tau_scale[e * TAU_FEAT + a] - 1.0)
                               / TAU_RAND_NORM);
  // env v49: blade-tip world velocity [L xyz, R xyz] / V_NORM (lib.rs
  // write_obs mirror; standard game only — Flux keeps the fill's zeros)
  if constexpr (!FX) {
    for (int s = 0; s < 2; s++)
      for (int k = 0; k < 3; k++)
        obs[OFF_TIP_VEL + 3 * s + k] =
            (float)(P.tip_vel[(e * 2 + s) * 3 + k] / T(V_NORM));
  }
}

// One 30 Hz control tick + judgment + bookkeeping (lib.rs step_inner),
// entirely per-thread.
template <typename T, bool FX>
__device__ void step_env(Dyn<T>& __restrict__ d, Solver<T>& __restrict__ s,
                         const Sim<T>& P, int e) {
  const int C = P.C;
  const int64_t c0 = P.chart_idx[e];

  // previous FK caches
  T prev_w[2][3], prev_d[2][3];
  for (int h = 0; h < 2; h++)
    for (int k = 0; k < 3; k++) {
      prev_w[h][k] = P.wrist[(e * 2 + h) * 3 + k];
      prev_d[h][k] = P.blade[(e * 2 + h) * 3 + k];
    }

  // gear ids -> torques (lib.rs Body::tick); act accumulates the v23
  // commanded activation Σ|frac| (telemetry) in double, matching lib.rs;
  // act_grp splits the same terms [spine, arms, legs] for telemetry
  T tau[NQ];
  const double* gr = gear_row(P.gears);
  double act = 0.0;
  double act_grp[3] = {0.0, 0.0, 0.0};
  // env v43 energy base: Σ|frac·τ_max| in double, joint order, matching
  // lib.rs Body::tick's f64 sum exactly (abs commutes with the product)
  double tau_abs = 0.0;
  double tau_grp[3] = {0.0, 0.0, 0.0};
  for (int j = 0; j < NQ; j++) {
    const int a = (int)P.actions[e * NQ + j];
    const double frac =
        a < 7 ? (a >= 3 ? gr[a] * FINE_GEAR_MULT[j] : gr[a]) : 0.0;
    // env v46: per-env axis scale (1.0 exact when off; lib.rs mirror)
    const double ts = (double)P.tau_scale[e * TAU_FEAT + TAU_AXIS[j]];
    tau[j] = T(frac) * CT(TAU_MAX, [j]) * T(ts);
    const double f = fabs(frac);
    act += f;
    const int g = j < 3 ? 0 : (j < 17 ? 1 : 2);
    act_grp[g] += f;
    const double t = fabs(frac * TAU_MAX[j] * ts);
    tau_abs += t;
    tau_grp[g] += t;
  }

  // 8 substeps + contact aggregation
  load_state(d, P, e);
  d.pd_bad = false;
  d.env_bad = false;
  d.env_n = 0;
  bool arm_l = false, arm_r = false;
  if (P.want_contacts)
    for (int pr = 0; pr < NUM_PAIRS; pr++)
      P.contact_imp[e * NUM_PAIRS + pr] = T(0);
  // env v11 FALL (contact spec §4): per-TICK on the end-of-tick root
  // height since 2026-08-20 (user — consistent with the tilt check; a
  // mid-tick dip that recovers within the tick no longer terminates).
  bool fell = false;
  bool tilted = false;
  // contact spec §7: tick-summed ground normal impulses per obs bucket
  // [L_heel, L_toe, R_heel, R_toe] (lib.rs Body::ground_imp).
  T gimp[4] = {T(0), T(0), T(0), T(0)};
  // env v34 limit-rest: this tick's limit-row impulse buckets
  T lrest_legs = T(0), lrest_spine = T(0), lrest_arms = T(0);
  // env v41: this tick's self-collision row impulse (all pairs)
  T sc_imp = T(0);
  for (int sub = 0; sub < N_SUB; sub++) {
    substep(d, s, tau);
    for (int ci = 0; ci < s.nc; ci++) sc_imp += tmax(s.c_lam[ci], T(0));
    for (int c = 0; c < NUM_GROUND_CORNERS; c++)  // G1 fixed slots
      if ((s.g_act >> c) & 1u)  // ascending = the packed order
        gimp[(c / CORNERS_PER_FOOT) * 2 + (c & 1)] +=
            s.g_lam[c * GROUND_ROWS_PER_CORNER];
    // env v34: joint-limit-row lambda this substep, postural scope
    // ([legs, spine]; arms skipped). l_lam is already >= 0 (PGS
    // unilateral clamp) — tmax is belt-and-suspenders (lib.rs mirror).
    for (int li = 0; li < s.nl; li++) {
      const int lj = s.l_j[li];
      const T ll = tmax(s.l_lam[li], T(0));
      if (lj < 3) lrest_spine += ll;
      else if (lj >= 17) lrest_legs += ll;
      else lrest_arms += ll;   // env v35 bucket (priced iff limit_arms)
    }
    // contact aggregation: obs bits see every ACTIVE row; replay-grade
    // impulses sum the row lambdas; point = last substep's (introspection
    // only — gated on want_contacts; the arm bits below always run)
    for (int ci = 0; ci < s.nc; ci++) {
      const int pid = s.c_pair[ci];
      if (P.want_contacts) {
        P.contact_imp[e * NUM_PAIRS + pid] += s.c_lam[ci];
        for (int k = 0; k < 3; k++)
          P.contact_pt[(e * NUM_PAIRS + pid) * 3 + k] = s.c_pt[ci][k];
      }
      const int ia = PAIR_A[pid], ib = PAIR_B[pid];
      if (is_arm_l(ia) || is_arm_l(ib)) arm_l = true;
      if (is_arm_r(ia) || is_arm_r(ib)) arm_r = true;
    }
  }
  if (d.pd_bad) atomicAdd(P.pd_hits, 1);
  if (d.env_n) atomicAdd(P.env_hits, d.env_n);
  // FALL check at end of tick; one count per fallen tick (lib.rs mirror)
  {
    if (d.p[1] < T(FALL_ROOT_Y)) fell = true;
    if (fell) atomicAdd(P.fall_hits, 1);
  }
  // TILT envelope — ONCE per tick on the end-of-tick state (user
  // 2026-08-20: substep-rate checking is cost without benefit; a tilt
  // is not a transient the way a floor dip is). Measured on the CHEST
  // (user 2026-08-20, v30 night-of-nights replay: the pelvis-only
  // check was gamed — pelvis surfed 28 deg while the spine pinned its
  // -20 deg limit for a sustained ~70 deg backbend). Chest up-axis =
  // R_root * Rx(spine_pitch) * Ry(spine_yaw) * Rz(spine_roll) * yhat;
  // the spine-local up v is composed first, then quat-rotated.
  {
    const T cp = cos(d.q[0]), sp = sin(d.q[0]);
    const T cy = cos(d.q[1]), sy = sin(d.q[1]);
    const T cr = cos(d.q[2]), sr = sin(d.q[2]);
    const T v0 = -cy * sr;
    const T v1 = cp * cr - sp * sy * sr;
    const T v2 = sp * cr + cp * sy * sr;
    const T qw = d.quat[0], qx = d.quat[1], qy = d.quat[2],
            qz = d.quat[3];
    const T upx = (T(1) - T(2) * (qy * qy + qz * qz)) * v0
        + T(2) * (qx * qy - qw * qz) * v1
        + T(2) * (qx * qz + qw * qy) * v2;
    const T upy = T(2) * (qx * qy + qw * qz) * v0
        + (T(1) - T(2) * (qx * qx + qz * qz)) * v1
        + T(2) * (qy * qz - qw * qx) * v2;
    const T upz = T(2) * (qx * qz - qw * qy) * v0
        + T(2) * (qy * qz + qw * qx) * v1
        + (T(1) - T(2) * (qx * qx + qy * qy)) * v2;
    const T sag2 = upz > T(0) ? T(TILT_TAN2_FWD) : T(TILT_TAN2_BACK);
    if (upy <= T(0) || upz * upz > sag2 * upy * upy
        || upx * upx > T(TILT_TAN2_ROLL) * upy * upy)
      tilted = true;
    if (tilted) atomicAdd(P.tilt_hits, 1);
  }
  for (int k = 0; k < N_GROUND_IMP_LANES; k++)
    P.ground_imp[e * N_GROUND_IMP_LANES + k] = gimp[k];
  P.contact_arm[e * 2] = arm_l;
  P.contact_arm[e * 2 + 1] = arm_r;
  store_state(d, P, e);
  refresh_fk(d, P, e);
  T new_w[2][3], new_d[2][3];
  for (int h = 0; h < 2; h++)
    for (int k = 0; k < 3; k++) {
      new_w[h][k] = P.wrist[(e * 2 + h) * 3 + k];
      new_d[h][k] = P.blade[(e * 2 + h) * 3 + k];
      P.wrist_vel[(e * 2 + h) * 3 + k] =
          (new_w[h][k] - prev_w[h][k]) / T(DT);
      // env v49: blade-tip velocity — lib.rs Body::tick's expression
      // tree (= the judge's tip_move below, / DT)
      P.tip_vel[(e * 2 + h) * 3 + k] =
          (new_w[h][k] + T(HANDLE_LEN + SABER_LEN) * new_d[h][k]
           - prev_w[h][k] - T(HANDLE_LEN + SABER_LEN) * prev_d[h][k])
          / T(DT);
    }
  // env v24: body-point world velocities — finite diff against the
  // last tick's stored positions, then advance the base (lib.rs
  // Body::tick mirror; d.pos holds this tick's refreshed FK)
  {
    for (int i = 0; i < N_BODY_PTS; i++)
      for (int k = 0; k < 3; k++) {
        const size_t idx = ((size_t)e * N_BODY_PTS + i) * 3 + k;
        const T np = d.pos[body_pt_frame(i)][k];
        P.body_pt_vel[idx] = (np - P.body_pt[idx]) / T(DT);
        P.body_pt[idx] = np;
      }
  }

  T reward = T(0);
  // env v43 energy price on the applied Σ|τ| (the v23 fraction price
  // retired 2026-09-14) + always-on activation and torque telemetry
  // (lib.rs mirror; both accumulate at coef 0)
  reward -= P.energy_coef * T(tau_abs);
  P.act_stat[e * 5 + 0] += T(act);
  P.act_stat[e * 5 + 1] += T(act_grp[0]);
  P.act_stat[e * 5 + 2] += T(act_grp[1]);
  P.act_stat[e * 5 + 3] += T(act_grp[2]);
  P.act_stat[e * 5 + 4] += T(1);
  P.tau_stat[e * 5 + 0] += T(tau_abs);
  P.tau_stat[e * 5 + 1] += T(tau_grp[0]);
  P.tau_stat[e * 5 + 2] += T(tau_grp[1]);
  P.tau_stat[e * 5 + 3] += T(tau_grp[2]);
  P.tau_stat[e * 5 + 4] += T(1);
  if (P.bench_refill) {   // this level's torque (bench_rec cols 24..27)
    T* rb = P.bench_rec + (size_t)e * BENCH_REC + BENCH_TAU_ACC;
    rb[0] += T(tau_abs);
    rb[1] += T(tau_grp[0]);
    rb[2] += T(tau_grp[1]);
    rb[3] += T(tau_grp[2]);
  }
  // env v34: limit-rest penalty + always-on telemetry (accumulates at
  // coef 0 too, so stop-resting is observable before it is priced)
  reward -= P.limit_coef * (lrest_legs + lrest_spine
                            + (P.limit_arms ? lrest_arms : T(0)));
  P.limit_rest[e * 3 + 0] += lrest_legs;
  P.limit_rest[e * 3 + 1] += lrest_spine;
  P.limit_rest[e * 3 + 2] += lrest_arms;
  P.limit_rest_ticks[e] += 1;
  // env v41: self-contact penalty + always-on telemetry (lib.rs mirror)
  reward -= P.self_contact_coef * sc_imp;
  P.sc_stat[e * 3 + 0] += sc_imp;
  P.sc_stat[e * 3 + 1] += sc_imp > T(0) ? T(1) : T(0);
  P.sc_stat[e * 3 + 2] += T(1);
  // FOOT-LIFT penalty (2026-08-20): flat FOOT_LIFT_PEN the tick a
  // foot leaves the ground — edge on the tick-summed ground normal
  // impulse (gimp buckets [L_heel, L_toe, R_heel, R_toe]), per foot.
  // lib.rs mirror.
  {
    for (int f = 0; f < 2; f++) {
      const bool fdown = gimp[f * 2] + gimp[f * 2 + 1] > T(0);
      if (!fdown && P.foot_down_prev[e * 2 + f]) {
        reward += T(FOOT_LIFT_PEN);
        atomicAdd(&P.foot_lift_hits[f], 1);
      }
      P.foot_down_prev[e * 2 + f] = fdown;
    }
  }
  // env v20/v21/v22 foot terms (lib.rs mirror): hinged rent (free
  // inside foot_hinge_r; r = 0 is the v21 pure-linear behavior) +
  // optional potential-based delta shaping. d.pos holds this tick's
  // FK — the same source the §8 body_pts lanes read.
  if (P.foot_z_coef > T(0) || P.foot_x_coef > T(0)
      || P.foot_shape_coef > T(0)) {
    const T hxl = tmax(fabs(d.pos[F_L_FOOT][0] + T(FOOT_X0))
                       - P.foot_hinge_r, T(0));
    const T hxr = tmax(fabs(d.pos[F_R_FOOT][0] - T(FOOT_X0))
                       - P.foot_hinge_r, T(0));
    const T hzl = tmax(fabs(d.pos[F_L_FOOT][2] - T(PLANE_Z0))
                       - P.foot_hinge_r, T(0));
    const T hzr = tmax(fabs(d.pos[F_R_FOOT][2] - T(PLANE_Z0))
                       - P.foot_hinge_r, T(0));
    const T hx = hxl + hxr;
    const T hz = hzl + hzr;
    reward -= P.foot_x_coef * hx + P.foot_z_coef * hz;
    if (P.foot_shape_coef > T(0)) {
      const T pot = hx + hz;
      reward += P.foot_shape_coef * (P.foot_pot[e] - pot);
      P.foot_pot[e] = pot;
    }
  }

  // swing chain update — Flux has no sabers: no swing chain, no
  // trackers, deltas stay 0 (lib.rs `if !self.flux`; the flail
  // telemetry below then accrues nothing, exactly as Rust)
  T deltas[2] = {T(0), T(0)};
  bool continued[2] = {false, false};
  if constexpr (!FX) {
  for (int h = 0; h < 2; h++) {
    const T dp = tclamp(dot3(prev_d[h], new_d[h]), T(-1), T(1));
    const T delta = acos(dp);
    deltas[h] = delta;
    continued[h] = false;
    T sacc = P.swing_acc[e * 2 + h];
    if (delta < T(SWING_BREAK)) {
      sacc = T(0);
      P.axis_valid[e * 2 + h] = false;
    } else {
      T cx[3];
      cross3(prev_d[h], new_d[h], cx);
      const T cn = sqrt(cx[0] * cx[0] + cx[1] * cx[1] + cx[2] * cx[2]);
      const T ax[3] = {cx[0] / cn, cx[1] / cn, cx[2] / cn};
      const int ab = (e * 2 + h) * 3;
      const bool rev =
          P.axis_valid[e * 2 + h]
          && (P.axis[ab] * ax[0] + P.axis[ab + 1] * ax[1]
              + P.axis[ab + 2] * ax[2]) < T(0);
      if (rev) {
        sacc = delta;
      } else {
        sacc += delta;
        continued[h] = true;
      }
      for (int k = 0; k < 3; k++) P.axis[ab + k] = ax[k];
      P.axis_valid[e * 2 + h] = true;
    }
    P.swing_acc[e * 2 + h] = sacc;
  }

  // follow-through trackers (existing, BEFORE this tick's cuts) — ring
  // walked from tr_ptr so resolutions pay in Rust's spawn order
  for (int h = 0; h < 2; h++) {
    const int ptr = P.tr_ptr[e * 2 + h];
    for (int kk = 0; kk < N_TRACKERS; kk++) {
      const int slot = (ptr + kk) % N_TRACKERS;
      const int b = (e * 2 + h) * N_TRACKERS + slot;
      if (!P.tr_active[b]) continue;
      T acc = P.tr_acc[b];
      bool resolve;
      if (!continued[h]) {
        resolve = true;
      } else {
        acc += deltas[h];
        const int age = P.tr_age[b] + 1;
        P.tr_acc[b] = acc;
        P.tr_age[b] = age;
        resolve = acc >= T(POST_FULL) || age >= FOLLOW_TIMEOUT;
      }
      if (resolve) {
        const T pts =
            round_pts(T(PTS_POST) * tmin(acc / T(POST_FULL), T(1)));
        const int cnt = P.tr_cnt[b];
        for (int cix = 0; cix < cnt; cix++) reward += pts / T(SCORE_MAX);
        P.score[e] += pts * T(cnt);
        P.score_post[e] += pts * T(cnt);
        P.acc_pts[e * 2 + h] += pts * T(cnt);
        if (P.combo_walk) P.cw_score[e] += T((float)pts) * T(P.tr_msum[b]);
        P.tr_active[b] = false;
      }
    }
  }
  }  // !FX

  const T time_new = T(P.tick[e] + 1) * T(DT);

  // both sabers' swept surfaces this tick: blade BASE (wrist + HANDLE_LEN
  // * dir, env v42 hilt grip) to tip (wrist + (HANDLE_LEN + SABER_LEN) *
  // dir); the handle does not cut
  T b0s[2][3], b1s[2][3], t0s[2][3], t1s[2][3];
  for (int h = 0; h < 2; h++)
    for (int k = 0; k < 3; k++) {
      b0s[h][k] = prev_w[h][k] + T(HANDLE_LEN) * prev_d[h][k];
      b1s[h][k] = new_w[h][k] + T(HANDLE_LEN) * new_d[h][k];
      t0s[h][k] = prev_w[h][k] + T(HANDLE_LEN + SABER_LEN) * prev_d[h][k];
      t1s[h][k] = new_w[h][k] + T(HANDLE_LEN + SABER_LEN) * new_d[h][k];
    }

  // live gates (SPEC-v10 §4)
  const T reach_z = tmax(P.shoulder_z[e * 2], P.shoulder_z[e * 2 + 1])
                    + T(REACH_PAST_SHOULDER);
  const T head_pos[3] = {P.head[e * 3], P.head[e * 3 + 1],
                         P.head[e * 3 + 2]};
  const T head_z = head_pos[2];
  const T z_off = head_z - T(PLANE_Z0); // v15 head-anchored note track
  const T min_speed = P.p_min_speed[c0];
  const int nn = (int)P.p_n_notes[c0];
  // ragged pool: chart c0's base. Loaded HERE, next to nn (the old
  // layout's first pool touch), not at the top of step_env — holding
  // it live across the dynamics substeps cost 12 B of spill in a
  // 255-register kernel (measured 2026-09-03: -0.5% steps/s).
  const int64_t nb0 = P.p_note_off[c0];
  int n_new[2] = {0, 0};
  int msum_new[2] = {0, 0};  // bench combo walk: pinned-multiplier sums

  // env v19 flail telemetry: on ticks whose next scoreable note is
  // > FLAIL_LEAD_S out, blade arc from hands with no live
  // follow-through is swinging that cannot earn. Runs BEFORE this
  // tick's cuts (a cut this tick implies a note inside the lead
  // window anyway). Pure accumulation — no reward/obs/termination
  // effect; popped by the wrapper.
  {
    T next_gap = T(2) * T(FLAIL_LEAD_S);
    for (int i = P.first_pending[e]; i < nn; i++) {
      if (P.status[e * C + i] != 0 || !P.p_valid[nb0 + i]) continue;
      next_gap = P.p_hit[nb0 + i] - time_new;
      break;
    }
    if (next_gap > T(FLAIL_LEAD_S)) {
      // per-HAND accumulation since 2026-08-20 (user: separate L/R
      // flail in the training telemetry) — flail_rad is (E, 2)
      for (int h = 0; h < 2; h++) {
        bool live = false;
        for (int kk = 0; kk < N_TRACKERS; kk++)
          if (P.tr_active[(e * 2 + h) * N_TRACKERS + kk]) {
            live = true;
            break;
          }
        if (!live) P.flail_rad[e * 2 + h] += deltas[h];
      }
      P.flail_ticks[e] += 1;
    }
  }

  // wrist-bend reversal telemetry (2026-09-22): peak detection with
  // WREV_HYST hysteresis on this tick's integrated bend angle. Pure
  // accumulation — no reward/obs/termination effect.
  for (int h = 0; h < 2; h++) {
    const int idx = e * 2 + h;
    const T q = d.q[h ? 15 : 8];
    int dir = P.wrev_dir[idx];
    T lo = P.wrev_ext[idx * 2], hi = P.wrev_ext[idx * 2 + 1];
    if (dir == 0) {
      lo = hi = q; dir = 3;
    } else if (dir == 3) {
      // undirected: track both extremes; a WREV_HYST rise from the low
      // (or fall from the high) commits the first direction — not a
      // reversal
      if (q < lo) lo = q;
      if (q > hi) hi = q;
      if (q >= lo + T(WREV_HYST)) { dir = 1; hi = q; }
      else if (q <= hi - T(WREV_HYST)) { dir = 2; lo = q; }
    } else if (dir == 1) {
      if (q > hi) hi = q;
      else if (q <= hi - T(WREV_HYST)) { P.wrev[idx] += 1; dir = 2; lo = q; }
    } else {
      if (q < lo) lo = q;
      else if (q >= lo + T(WREV_HYST)) { P.wrev[idx] += 1; dir = 1; hi = q; }
    }
    P.wrev_dir[idx] = dir;
    P.wrev_ext[idx * 2] = lo;
    P.wrev_ext[idx * 2 + 1] = hi;
  }

  if constexpr (FX) {
    // ---- Flux judgment (flux_sim.rs flux_judge + flux.rs judge verbatim)
    T p0[2][3], p1[2][3];
    for (int h = 0; h < 2; h++)
      for (int k = 0; k < 3; k++) {
        p0[h][k] = prev_w[h][k] + T(PALM_OFF) * prev_d[h][k];
        p1[h][k] = new_w[h][k] + T(PALM_OFF) * new_d[h][k];
      }
    for (int h = 0; h < 2; h++)
      for (int k = 0; k < 3; k++)
        P.palm_vel[(e * 2 + h) * 3 + k] = (p1[h][k] - p0[h][k]) / T(DT);
    const T freach = tmax(P.shoulder_z[e * 2], P.shoulder_z[e * 2 + 1])
                     + T(FLUX_REACH_PAST_SHOULDER);
    for (int i = P.first_pending[e]; i < nn; i++) {
      if (P.status[e * C + i] != 0) continue;
      const int64_t cb = nb0 + i;
      if (!P.p_valid[cb]) continue;
      if ((P.p_hit[cb] - time_new) * min_speed + z_off > freach) break;
      const T z = (P.p_hit[cb] - time_new) * P.p_speed[cb] + z_off;
      const T blen = P.p_len[cb];
      const bool is_bar = blen > T(0);
      if (z > freach) continue;
      const T center[3] = {P.p_center[cb * 2], P.p_center[cb * 2 + 1], z};
      // last tick's palms in the object's frame: it was speed*DT further
      // out then (flux.rs judge's q0)
      const T travel = P.p_speed[cb] * T(DT);
      const T q0[2][3] = {{p0[0][0], p0[0][1], p0[0][2] - travel},
                          {p0[1][0], p0[1][1], p0[1][2] - travel}};
      // the sweep ends where the object crosses the head plane (flux.rs
      // judge's e1): past it the object is a miss, never a touch
      T e1[2][3];
      if (z < head_z) {
        const T sc = tclamp((z + travel - head_z) / travel, T(0), T(1));
        for (int h = 0; h < 2; h++)
          for (int k = 0; k < 3; k++)
            e1[h][k] = q0[h][k] + sc * (p1[h][k] - q0[h][k]);
      } else {
        for (int h = 0; h < 2; h++)
          for (int k = 0; k < 3; k++) e1[h][k] = p1[h][k];
      }
      if (!is_bar) {
        const int m = P.p_is_right[cb] ? 1 : 0;
        const T dm = flux_circle_touch(q0[m], e1[m], center);
        if (dm >= T(0)) {
          const T acc = round_pts(
              T(CIRCLE_ACC_PTS)
              * (T(1) - tclamp(dm / T(ACC_NORM), T(0), T(1))));
          const T pts = T(CIRCLE_PTS) + acc;
          P.status[e * C + i] = 1;
          P.good[e] += 1;
          reward += pts / T(SCORE_MAX);
          P.score[e] += pts;
          P.score_pre[e] += T(CIRCLE_PTS);
          P.score_acc[e] += acc;
          P.acc_pts[e * 2 + m] += pts;
          P.acc_cuts[e * 2 + m] += 1;
          cw_cut(P, e, i, pts);
          P.prev_idx[e * 2 + m] = i;
          P.prev_valid[e * 2 + m] = 1;
        } else if (flux_circle_touch(q0[1 - m], e1[1 - m], center)
                   >= T(0)) {
          P.status[e * C + i] = 2;
          P.bad[e] += 1;
          reward += T(REWARD_BAD);
          cw_break(P, e);
          P.prev_idx[e * 2 + m] = i;
          P.prev_valid[e * 2 + m] = 1;
        }
      } else {
        // bar: lam = 0 z-roll frame, p_rot = (cos θ, sin θ), local y' = u
        T frame[3][3];
        note_frame(center, P.p_rot[cb * 2], P.p_rot[cb * 2 + 1], T(0),
                   head_pos, frame);
        const T bh[3] = {T(BAR_HALF_W), T(0.5) * blen, T(FLUX_SLAB_HALF)};
        // swept Flux quad (L0 -> R0) to (L1 -> R1), relative: triangles
        // (L0, R0, R1) and (L0, R1, L1) — sweep_hits_box's (w0, t0, t1)
        // + (w0, t1, w1) split with w0 = L0, t0 = R0, t1 = R1, w1 = L1
        if (sweep_hits_box(q0[0], q0[1], e1[1], e1[0], center, bh, frame)) {
          if (!flux_bar_angle_ok(P.p_rot[cb * 2], P.p_rot[cb * 2 + 1],
                                 p1[0], p1[1])) {
            // crossed > 60 deg off square: bad cut (event 2 "angle")
            P.status[e * C + i] = 2;
            P.bad[e] += 1;
            reward += T(REWARD_BAD);
            cw_break(P, e);
            continue;
          }
          const T ux = -P.p_rot[cb * 2 + 1], uy = P.p_rot[cb * 2];
          const T dm = flux_bar_sweet_dist(center[0], center[1], blen,
                                           P.p_sweet[cb], ux, uy, p1[0],
                                           p1[1]);
          const T acc = round_pts(
              T(BAR_ACC_PTS)
              * (T(1) - tclamp(dm / T(ACC_NORM), T(0), T(1))));
          const T pts = T(BAR_PTS) + acc;
          P.status[e * C + i] = 1;
          P.good[e] += 1;
          reward += pts / T(SCORE_MAX);
          P.score[e] += pts;
          P.score_pre[e] += T(BAR_PTS);
          P.score_acc[e] += acc;
          // training-acc telemetry (CUDA-only): a bar is both hands' —
          // credited to each hand's flow (no Rust twin)
          for (int h = 0; h < 2; h++) {
            P.acc_pts[e * 2 + h] += pts;
            P.acc_cuts[e * 2 + h] += 1;
          }
          cw_cut(P, e, i, pts);
        }
      }
      // untouched and past the head plane: a miss (contact first — an
      // object can pass a palm and the head within one tick)
      if (P.status[e * C + i] == 0 && z < head_z) {
        P.status[e * C + i] = 3;
        P.miss[e] += 1;
        cw_break(P, e);
        reward += T(REWARD_MISS);
        if (!is_bar) {  // circles record on their colour; bars never
          const int h = P.p_is_right[cb] ? 1 : 0;
          P.prev_idx[e * 2 + h] = i;
          P.prev_valid[e * 2 + h] = 1;
        }
      }
    }
  } else {
  for (int i = P.first_pending[e]; i < nn; i++) {
    if (P.status[e * C + i] != 0) continue;
    const int64_t cb = nb0 + i;
    if (!P.p_valid[cb]) continue;
    // sorted hit_time + min_speed: beyond this bound so is every later
    if ((P.p_hit[cb] - time_new) * min_speed + z_off > reach_z) break;
    const T z = (P.p_hit[cb] - time_new) * P.p_speed[cb] + z_off;
    const bool bomb = P.p_is_bomb[cb];
    if (z < head_z) {
      if (bomb) {
        P.status[e * C + i] = 4;
      } else {
        P.status[e * C + i] = 3;
        P.miss[e] += 1;
        cw_break(P, e);
        reward += T(REWARD_MISS);
        // env v45: a miss records on the note's colour
        const int h = P.p_is_right[cb] ? 1 : 0;
        P.prev_idx[e * 2 + h] = i;
        P.prev_valid[e * 2 + h] = 1;
      }
      continue;
    }
    if (z > reach_z) continue;
    const T center[3] = {P.p_center[cb * 2], P.p_center[cb * 2 + 1], z};

    if (bomb) {
      for (int h = 0; h < 2; h++) {
        if (sweep_dist(b0s[h], t0s[h], t1s[h], b1s[h], center)
            <= T(BOMB_RADIUS)) {
          P.status[e * C + i] = 2;
          P.bombs[e] += 1;
          cw_break(P, e);
          reward += T(REWARD_BOMB);
          break;
        }
      }
      continue;
    }

    const int m = P.p_is_right[cb] ? 1 : 0;
    const bool is_head = P.p_is_head[cb];
    const bool is_link = P.p_is_link[cb];
    const bool any_dir = P.p_any_dir[cb];
    const T lam = (!is_head && !is_link)
                      ? tclamp(T(1) - z / T(LOOK_DIST), T(0), T(1))
                      : T(0);
    T frame[3][3];
    note_frame(center, P.p_rot[cb * 2], P.p_rot[cb * 2 + 1], lam, head_pos,
               frame);
    const T gc[3] = {center[0] + T(GOOD_Z_SHIFT) * frame[2][0],
                     center[1] + T(GOOD_Z_SHIFT) * frame[2][1],
                     center[2] + T(GOOD_Z_SHIFT) * frame[2][2]};
    const T good_h[3] = {T(GOOD_HALF_X), P.p_good_hy[cb], T(GOOD_HALF_Z)};
    if (sweep_hits_box(b0s[m], t0s[m], t1s[m], b1s[m], gc, good_h,
                       frame)) {
      // contact point: closest-passing blade sample; strict < like Rust
      T d_seg = std::numeric_limits<T>::infinity();
      T s_star = T(0);
      for (int jj = 1; jj <= N_BLADE_SAMPLES; jj++) {
        const T rr = T(HANDLE_LEN) + (T(jj) / T(N_BLADE_SAMPLES)) * T(SABER_LEN);
        T p0[3], p1[3];
        for (int k = 0; k < 3; k++) {
          p0[k] = prev_w[m][k] + rr * prev_d[m][k];
          p1[k] = new_w[m][k] + rr * new_d[m][k];
        }
        const T dd = seg_point_dist3(p0, p1, center);
        if (dd < d_seg) {
          d_seg = dd;
          s_star = rr;
        }
      }
      T vc[3];
      for (int k = 0; k < 3; k++)
        vc[k] = (new_w[m][k] + s_star * new_d[m][k] - prev_w[m][k]
                 - s_star * prev_d[m][k]) / T(DT);
      const T vc_mag = sqrt(vc[0] * vc[0] + vc[1] * vc[1] + vc[2] * vc[2]);
      // cut-plane accuracy (2026-07-29 rule)
      T tip_move[3], cw[3], pn[3];
      for (int k = 0; k < 3; k++) {
        tip_move[k] = new_w[m][k] + T(HANDLE_LEN + SABER_LEN) * new_d[m][k]
                      - prev_w[m][k] - T(HANDLE_LEN + SABER_LEN) * prev_d[m][k];
        cw[k] = center[k] - new_w[m][k];
      }
      cross3(new_d[m], tip_move, pn);
      const T pn_mag = sqrt(dot3(pn, pn));
      T d_min;
      if (pn_mag > T(1e-9)) {
        d_min = fabs(dot3(cw, pn) / pn_mag);
      } else {
        const T along = dot3(cw, new_d[m]);
        d_min = sqrt(tmax(dot3(cw, cw) - along * along, T(0)));
      }
      int code;
      if (vc_mag < T(V_MIN_CUT)) {
        code = 1;
      } else if (any_dir) {
        code = 0;
      } else {
        const T vlx = dot3(vc, frame[0]);
        const T vly = dot3(vc, frame[1]);
        const T v2 = sqrt(vlx * vlx + vly * vly);
        const T cosv = -vly / tmax(v2, T(1e-9));
        code = cosv >= T(COS_TOL) ? 0 : 2;
      }
      // env v45: cut by the matching saber (good or bad) — recorded
      // on the note's colour (m == is_right)
      P.prev_idx[e * 2 + m] = i;
      P.prev_valid[e * 2 + m] = 1;
      if (code == 0) {
        P.status[e * C + i] = 1;
        P.good[e] += 1;
        T pts, pre_pts = T(0), acc_pts = T(0);
        bool spawn_tr = false;
        if (is_link) {
          pts = T(CHAIN_LINK_PTS);
        } else {
          const T pre =
              P.p_free_pre[cb]
                  ? T(PTS_PRE)
                  : round_pts(T(PTS_PRE)
                              * tmin(P.swing_acc[e * 2 + m] / T(PRE_FULL),
                                     T(1)));
          const T acc =
              round_pts(T(PTS_ACC)
                        * (T(1) - tclamp(d_min / T(ACC_NORM), T(0), T(1))));
          pts = pre + acc;
          pre_pts = pre;
          acc_pts = acc;
          if (is_head) {
            // chain heads: no follow-through (max 85), except one that is
            // also an arc head or tail (ChainHeadArcHead/ArcTail): the 30
            // post points are a fixed grant (scoring v4, lib.rs mirror)
            if (P.p_free_pre[cb] || P.p_free_post[cb]) {
              pts += T(PTS_POST);
              P.score_post[e] += T(PTS_POST);
            }
          } else if (P.p_free_post[cb]) {
            pts += T(PTS_POST);
            P.score_post[e] += T(PTS_POST);
          } else {
            n_new[m] += 1;  // spawn a follow-through tracker
            spawn_tr = true;
          }
        }
        reward += pts / T(SCORE_MAX);
        P.score[e] += pts;
        P.score_pre[e] += pre_pts;
        P.score_acc[e] += acc_pts;
        P.acc_pts[e * 2 + m] += pts;
        P.acc_cuts[e * 2 + m] += 1;
        {
          const int mult = cw_cut(P, e, i, pts);
          if (spawn_tr) msum_new[m] += mult;
        }
      } else {
        P.status[e * C + i] = 2;
        P.bad[e] += 1;
        reward += T(REWARD_BAD);
        cw_break(P, e);
      }
    } else {
      const T bad_h[3] = {T(BAD_HALF), P.p_bad_hy[cb], T(BAD_HALF)};
      if (sweep_hits_box(b0s[1 - m], t0s[1 - m], t1s[1 - m],
                         b1s[1 - m], center, bad_h, frame)) {
        P.status[e * C + i] = 2;
        P.bad[e] += 1;
        reward += T(REWARD_BAD);
        cw_break(P, e);
        // env v45: wrong-hand cut — still recorded on the NOTE'S
        // colour (m), never on the saber that cut it
        P.prev_idx[e * 2 + m] = i;
        P.prev_valid[e * 2 + m] = 1;
      }
    }
  }
  }  // FX / standard judgment

  // spawn aggregated trackers (same-tick spawns share (arc, age))
  for (int h = 0; h < 2; h++) {
    if (n_new[h] > 0) {
      const int ptr = P.tr_ptr[e * 2 + h];
      const int b = (e * 2 + h) * N_TRACKERS + ptr;
      P.tr_acc[b] = T(0);
      P.tr_age[b] = 0;
      P.tr_cnt[b] = n_new[h];
      P.tr_msum[b] = msum_new[h];
      P.tr_active[b] = true;
      P.tr_ptr[e * 2 + h] = (ptr + 1) % N_TRACKERS;
    }
  }

  // env v17 walls (spec §12b) — lib.rs step_inner verbatim, same
  // reward summation position (after all judgment terms). Head center
  // vs AABB; WALL_PEN charged ONCE on the entry edge (head outside
  // last tick, inside now — lib.rs was_in_wall/wall_enters rule),
  // regardless of overlapping walls. walls_in still counts every
  // tick inside; >= 1 tick fully outside re-arms the charge.
  {
    const int nw = (int)P.p_n_walls[c0];
    const int64_t wb0 = P.p_wall_off[c0];
    const T hx2 = P.head[e * 3], hy2 = P.head[e * 3 + 1];
    bool in_wall = false;
    for (int wi = P.first_wall[e]; wi < nw; wi++) {
      const int64_t wb = wb0 + wi;
      if (!P.p_wall_valid[wb] || P.wall_passed[e * P.W + wi]) continue;
      const T dzf = (P.p_wall_t[wb] - time_new) * P.p_wall_speed[wb]
          - T(PLANE_Z0) - T(WALL_FRONT_OFF);
      const T len = P.p_wall_len[wb];
      // env v36 INTERVAL SWEEP (user): the wall occupies the head
      // plane during a time window; charge if that window intersects
      // (time_old, time_new] instead of point-sampling at time_new —
      // sub-tick (thin/fast) walls can no longer phase through
      // between ticks. In dzf terms the tick's swept span is
      // [dzf, dzf + speed*DT]; overlap with the wall's [-len, 0] iff
      // dzf <= 0 && dzf + len + speed*DT >= 0. x/y use the end-of-
      // tick head (moves mm-scale within one tick — documented
      // approximation). The passed latch still fires on the same
      // tick (disjoint per-tick windows: no double charge).
      const bool z_hit = dzf <= T(0)
          && dzf + len + P.p_wall_speed[wb] * T(DT) >= T(0);
      if (dzf + len < T(0))
        P.wall_passed[e * P.W + wi] = true;  // back face behind the head
      if (z_hit
          && fabs(hx2 - P.p_wall_cx[wb]) <= P.p_wall_w[wb] * T(0.5)
          && fabs(hy2 - P.p_wall_cy[wb]) <= P.p_wall_h[wb] * T(0.5)) {
        if (P.wall_despawn) {
          // env v37 (lib.rs mirror): charge and remove THIS wall —
          // one fee per wall object, no lingering in-wall ticks.
          reward += T(WALL_PEN);
          cw_break(P, e);
          bench_wall_enter(P, e);
          P.wall_passed[e * P.W + wi] = true;
          P.walls_in[e] += 1;  // counts entries in despawn mode
        } else {
          in_wall = true;
        }
      }
    }
    if (in_wall) {
      if (!P.in_wall_prev[e]) {
        reward += T(WALL_PEN);
        cw_break(P, e);
        bench_wall_enter(P, e);
      }
      P.walls_in[e] += 1;
    }
    P.in_wall_prev[e] = in_wall;
    int fw = P.first_wall[e];
    while (fw < nw && P.wall_passed[e * P.W + fw]) fw++;
    P.first_wall[e] = fw;
  }

  const int tick_new = P.tick[e] + 1;
  P.tick[e] = tick_new;
  // advance the low-water cursor
  int fp = P.first_pending[e];
  while (fp < nn && P.status[e * C + fp] != 0) fp++;
  P.first_pending[e] = fp;
  const bool pending = fp < nn;
  bool no_trk = true;
  for (int b = e * 2 * N_TRACKERS; b < (e + 1) * 2 * N_TRACKERS; b++)
    if (P.tr_active[b]) {
      no_trk = false;
      break;
    }
  const T last_hit = P.p_last_hit[c0];
  // C backstop (Divergences §23): an env whose root left the physical
  // envelope this tick terminates — folded into `done` BEFORE the
  // tracker resolution (identical drain to any other terminal tick) and
  // its reward zeroed AFTER.
  // env v11 FALL (contact spec §4) rides the SAME seam, exactly as
  // lib.rs folds `self.body.fell` in beside `self.body.escaped`: same
  // place in the disjunction (so pending follow-throughs still drain),
  // same post-hoc reward zeroing.
  const bool done =
      (!pending && no_trk && (nn == 0 || time_new > last_hit))
      || time_new > last_hit + P.p_tail[c0] || tick_new >= P.max_steps
      || d.env_bad || fell || tilted;

  if (done) {  // resolve pending follow-throughs with partial arcs
    for (int h = 0; h < 2; h++) {
      const int ptr = P.tr_ptr[e * 2 + h];
      for (int kk = 0; kk < N_TRACKERS; kk++) {
        const int slot = (ptr + kk) % N_TRACKERS;
        const int b = (e * 2 + h) * N_TRACKERS + slot;
        if (!P.tr_active[b]) continue;
        const T pts = round_pts(
            T(PTS_POST) * tmin(P.tr_acc[b] / T(POST_FULL), T(1)));
        const int cnt = P.tr_cnt[b];
        for (int cix = 0; cix < cnt; cix++) reward += pts / T(SCORE_MAX);
        P.score[e] += pts * T(cnt);
        P.score_post[e] += pts * T(cnt);
        P.acc_pts[e * 2 + h] += pts * T(cnt);
        if (P.combo_walk) P.cw_score[e] += T((float)pts) * T(P.tr_msum[b]);
        P.tr_active[b] = false;
      }
    }
  }

  // contact spec §7 survival gradient — MIRROR of lib.rs: the alive
  // bonus lands BEFORE the terminal overrides, so a fall/escape tick
  // never collects it.
  reward += T(ALIVE_BONUS);
  if (d.env_bad) reward = T(0);
  // v11.1 fall pricing (contact spec §4, 2026-08-09) — MIRROR of
  // lib.rs's fell block: REWARD_MISS x pending scoreable notes, full
  // 0..nn scan in chart order, statuses untouched. Replaces the
  // reward-0 zeroing whose bounded penalty made fast death the 8h
  // run's converged optimum. The p_valid gate is not load-bearing
  // (slots 0..nn are exactly the valid ones by pool construction) —
  // it mirrors the note loop's own i<nn convention so the two scans
  // can never diverge if that invariant ever changes.
  {
    if (fell || tilted) {
      // env v18 (sole mode) — MIRROR of lib.rs: flat terminal reward;
      // tilt terminations (2026-08-20) priced identically
      reward = -P.fall_const;
    }
  }
  const T ep = P.ep_return[e] + reward;
  P.ep_return[e] = ep;
  P.reward[e] = reward;
  P.done[e] = done;
  if (done) {
    const int pos = atomicAdd(P.ret_total, 1);
    P.ret_buf[pos % P.cap] = ep;
    // env v46 tau_draws record (only while randomizing): the body this
    // episode played and how it went; the wrapper pops + clears flags
    if (P.tau_rand > T(0) || P.ep_stats) {
      T* r = P.ep_rec + (size_t)e * 28;
      for (int a = 0; a < TAU_FEAT; a++) r[a] = P.tau_scale[e * TAU_FEAT + a];
      r[16] = T(tick_new);
      r[17] = ep;
      r[18] = (fell || tilted) ? T(1) : T(0);
      r[19] = T(P.good[e]);
      r[20] = P.score[e];
      r[21] = T(P.chart_idx[e]);
      // env v47 cols: zero unless the combo walk ran this episode
      r[22] = P.ep_stats ? P.cw_score[e] : T(0);
      r[23] = P.ep_stats ? T(P.cw_maxc[e]) : T(0);
      r[24] = P.ep_stats ? T(P.cw_breaks[e]) : T(0);
      // completed = the level itself ended (notes drained / tail), not a
      // fall, tilt, envelope escape or the tick cap
      r[25] = (P.ep_stats && !(fell || tilted || d.env_bad)
               && tick_new < P.max_steps) ? T(1) : T(0);
      r[26] = P.ep_stats ? T(P.bad[e]) : T(0);
      r[27] = P.ep_stats ? T(P.miss[e]) : T(0);
      P.ep_flag[e] = 1;
    }
  }
  if (P.bench_refill && done) {
    // bench slot refill: snapshot this level's result, then either
    // start the slot's next level now (same tick, fresh reset state)
    // or leave it frozen. done stays latched on a frozen slot; a
    // refilled slot must NOT read as done next tick (the combo_walk
    // early-out keys on it).
    T* r = P.bench_rec + (size_t)e * BENCH_REC;
    r[0] = T(P.good[e]);
    r[1] = T(P.bad[e]);
    r[2] = T(P.miss[e]);
    r[3] = P.score[e];
    r[4] = T(P.bombs[e]);
    r[5] = P.score_pre[e];
    r[6] = P.score_acc[e];
    r[7] = P.score_post[e];
    r[8] = P.flail_rad[e * 2] + P.flail_rad[e * 2 + 1];
    r[9] = T(P.flail_ticks[e]);
    r[10] = P.cw_score[e];
    r[11] = T(P.cw_maxc[e]);
    r[12] = T(P.cw_breaks[e]);
    r[13] = T(P.chart_idx[e]);
    r[14] = T(P.wrev[e * 2]);        // wrist-bend reversals L / R this
    r[15] = T(P.wrev[e * 2 + 1]);    // level + the level's ticks
    r[16] = T(tick_new);
    r[17] = r[BENCH_WALL_ACC];       // wall entries this level (B09)
    for (int k = 0; k < 4; k++)      // this level's Σ|τ| [total, spine,
      r[BENCH_TAU_OUT + k] = r[BENCH_TAU_ACC + k];   // arms, legs] N·m
    r[18] = T((fell ? TERM_FELL : 0) | (tilted ? TERM_TILT : 0)
              | (d.env_bad ? TERM_ENVELOPE : 0)
              | (tick_new >= P.max_steps ? TERM_TICKCAP : 0));
    P.bench_flag[e] = 1;
    P.flail_rad[e * 2] = T(0);
    P.flail_rad[e * 2 + 1] = T(0);
    P.flail_ticks[e] = 0;
    P.wrev[e * 2] = 0;
    P.wrev[e * 2 + 1] = 0;
    if (P.cand[e] >= 0) {
      reset_env(d, P, e, P.cand[e]);
      P.done[e] = 0;
    }
  } else if (done && P.auto_reset) {
    reset_env(d, P, e, P.cand[e]);
  }
}

// FX = Flux game mode (lib.rs Core::flux, fixed per instance): a
// compile-time switch, so every Flux branch is discarded from the
// standard FX = false kernels (two instantiations per dtype, selected
// once on the host; the former foot-mode switch FC went with weld mode
// on 2026-09-27). Their arithmetic is unchanged; the
// only additions there are the palm_vel zero stores in reset_env and
// the mode-2 refresh (shared device functions, not FX-templated), so
// the register/spill census may move slightly — re-check on the pod.
template <typename T, bool FX>
// The `, 1` PINS the register budget at 255 — see the THREADS note: it
// is not an occupancy hint, it is what stops ptxas's free-choice
// heuristic from flipping to 64 regs + double the spills.
__global__ void __launch_bounds__(THREADS, 1)
step_kernel(Sim<T> P) {
  // R12 (2026-08-21): warp-coherence env permutation. The grid walks
  // env_order — identity by default; under env_order=True the host
  // sorts it by (chart, first_pending) each tick so warp-mates share
  // note windows (audit 3.3: identical-envs header ran 3.5x production
  // traffic). Per-env work and every P.*[e] address are unchanged —
  // a bijection on thread->env assignment, bitwise-identical per env.
  const int gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid >= P.N) return;
  const int e = P.env_order[gid];
  // A finished episode stays finished — MIRROR of lib.rs's B4
  // `finished` latch (step_inner early-out): reward exactly 0, done
  // stays true, no physics, no event scan, no telemetry, obs row and
  // tick frozen. Applies whenever nothing re-arms the slot in-kernel:
  // auto_reset=False (a latched done is only cleared by reset(), the
  // chart-install funnel), and the bench combo walk (without it a fallen
  // body would keep collecting misses after its terminal tick). Under
  // auto_reset a done flag means the slot was ALREADY reset on its
  // terminal tick, so it must step. Before 2026-09-27 the auto_reset=
  // False case kept stepping the finished body and re-reported its
  // terminal reward (fall_const again) every tick.
  if (P.mode == 0 && P.done[e]
      && (!P.auto_reset || (P.combo_walk && !P.ep_stats))) {
    P.reward[e] = T(0);
    return;
  }
  // thread-local scratch (local memory; per-thread interleaved =
  // coalesced across the warp for same-index access)
  Dyn<T> d;
  Solver<T> s;
  s.eprev = 0;  // #6e: no column history at the tick's first substep
  if (P.mode == 1) {
    reset_env(d, P, e, P.cand[e]);
  } else if (P.mode == 2) {
    // set_state refresh (RawBeatSim.set_joints semantics): FK caches
    // from the freshly written state; finite-diff wrist velocity reset;
    // swing accumulators untouched
    load_state(d, P, e);
    refresh_fk(d, P, e);
    for (int k = 0; k < 6; k++) P.wrist_vel[e * 6 + k] = T(0);
    for (int k = 0; k < 6; k++) P.palm_vel[e * 6 + k] = T(0);  // Flux
    for (int k = 0; k < 6; k++) P.tip_vel[e * 6 + k] = T(0);   // env v49
    // env v24: set_state semantics — re-seed the finite-diff base at
    // the refreshed pose, velocities zero (lib.rs set_state mirror)
    seed_body_pts(d, P, e);
  } else {
    d.pd_bad = false;
    d.env_bad = false;
    d.env_n = 0;
    step_env<T, FX>(d, s, P, e);
  }
  write_obs<T, FX>(d, P, e);
}

// TOMBSTONE (2026-08-15) — the warp / sub-warp-per-env kernel family
// lived here (step_kernel_warp, step_kernel_sw, the *_warp / *_sw device
// forks, SubWarpGeom + the sw_* shuffle helpers, chol_solve_cn, and the
// kernel_variant selection branch in step() below). DELETED: measured
// slower at EVERY lane width W — 169 (W=2) / 251 (W=4) / 338 (W=8) ms
// per tick at N=16,384 against the legacy kernel's 121.8. Redundant
// per-lane execution and its local traffic (plus spills at W>=4) beat
// the residency arithmetic at every rung, so there is no rung to climb.
// Full record (docs deleted 2026-09-22, in git history at 91eac64):
// design/WARP_ENV_SOLVER_CORE.md, WARP_RESCUE_LEGACY_SIDE.md,
// WARP_ENV_RECONCILIATION.md; code in git history at 7dab600 (the
// right-looking Cholesky back-port lived on as a compile-time option
// until 2026-09-22).

// tensor-list slots (order is a contract with cuda_env.py _tensors)
enum {
  TP = 0, TQUAT, TQ, TV,
  TWRIST, TBLADE, TWVEL, THEAD, TSHZ,
  TSWING, TAXIS, TAXV,
  TTRACC, TTRAGE, TTRCNT, TTRACT, TTRPTR,
  TTICK, TSTATUS, TFIRSTP, TEPRET,
  TGOOD, TBAD, TMISS, TBOMBS, TSCORE, TSPRE, TSACC, TSPOST,
  TCIMP, TCPT, TCARM, TPD, TENV, TFALL, TTILT, TLIFT, TFOOTDOWN,
  TFLAILR, TFLAILT, TWREV, TWREVEXT, TWREVDIR, // wrist-bend reversals: count (N,2), extremes (N,2,2), state (N,2) — 2026-09-22
  TACCP, TACCC, TGIMP, TCIDX, TLIMREST, TLIMTICKS,
  TPOSEJIT, TFOOTPOT, TBPPOS, TBPVEL,
  TSCSTAT,  // env v41 self-contact telemetry (N, 3)
  TACTSTAT, // activation telemetry (N, 5) — 2026-09-13
  TTAUSTAT, // env v43 torque telemetry (N, 5) — 2026-09-14
  TPREVIDX, TPREVVALID, // env v45 prev-note record (N, 2) — 2026-09-15
  TTAUSCALE, TTAUJIT, TEPREC, TEPFLAG, // env v46 tau scales (N,16), draws (N,16), episode record (N,28: v47 cols 22..28), flag (N) — 2026-09-16/19
  TCWMULT, TCWPROG, TCWCOMBO, TCWMAXC, TCWBREAKS, TCWSCORE, TCWNOTEMULT, TTRMSUM, // bench combo walk — 2026-09-16
  TBENCHREC, TBENCHFLAG, // bench slot refill: level record (N,BENCH_REC) + flag (N) — 2026-09-16
  TPALMV,   // Flux palm velocity (N, 2, 3)
  TSTARTKICK, // start-jitter kick [dvx, dvz] (N, 2) — review B10, 2026-09-30
  TTIPV,    // env v49 blade-tip velocity (N, 2, 3) — 2026-09-30
  PHIT, PSPEED, PCENTER, PDVEC, PROT, PGHY, PBHY,
  PVALID, PISR, PANY, PBOMB, PHEAD, PLINK, PSHOW, PFPRE, PFPOST,
  PLEN, PSWEET,  // Flux bar length / sweet offset (0 elsewhere)
  PNNOTES, PLASTHIT, PTAIL, PMINSP, PNOFF,
  // env v17 wall pool (ragged) + per-env wall state
  PWALLT, PWALLCX, PWALLCY, PWALLW, PWALLH, PWALLLEN, PWALLSP,
  PWALLVALID, PNWALLS, PWOFF, TWALLPASS, TFIRSTW, TWALLSIN, TINWALLPREV,
  TENVORD,
  RBUF, RTOT, OOBS, OREW, ODONE, T_COUNT
};


void step(std::vector<at::Tensor> ts, at::Tensor actions, at::Tensor cand,
          int64_t mode, int64_t max_steps,
          bool auto_reset, int64_t gears,
          bool want_contacts, double fall_const,
          double foot_z_coef, double foot_x_coef, double foot_hinge_r,
          double foot_shape_coef, double energy_coef, double limit_coef,
          int64_t limit_arms,
          int64_t wall_despawn,
          double start_pose_jitter, double crouch_prob,
          double z_asinh, double self_contact_coef,
          double tau_rand, int64_t combo_walk, int64_t bench_refill,
          int64_t ep_stats, bool flux) {
  TORCH_CHECK((int)ts.size() == T_COUNT, "expected ", (int)T_COUNT,
              " state tensors, got ", ts.size());
  TORCH_CHECK(z_asinh > 0.0, "z_asinh must be > 0 (asinh z lane, metres)");
  const auto& q = ts[TQ];
  TORCH_CHECK(q.is_cuda(), "state must live on CUDA");
  TORCH_CHECK(actions.scalar_type() == at::kLong && actions.is_contiguous());
  TORCH_CHECK(cand.scalar_type() == at::kLong && cand.is_contiguous());
  const int64_t N = q.size(0);
  const int64_t C = ts[TSTATUS].size(1);
  const int64_t cap = ts[RBUF].size(0);
  // ragged pool contract: (P + 1,) int64 offsets, per-object fields flat
  TORCH_CHECK(ts[PNOFF].scalar_type() == at::kLong
              && ts[PNOFF].size(0) == ts[PNNOTES].size(0) + 1
              && ts[PHIT].dim() == 1
              && ts[PWOFF].scalar_type() == at::kLong
              && ts[PWOFF].size(0) == ts[PNWALLS].size(0) + 1
              && ts[PWALLT].dim() == 1,
              "chart pool must be ragged (note_off/wall_off, flat fields)");
  AT_DISPATCH_FLOATING_TYPES(q.scalar_type(), "beat_arena_step_v10", [&] {
    Sim<scalar_t> P;
    P.p = ts[TP].data_ptr<scalar_t>();
    P.quat = ts[TQUAT].data_ptr<scalar_t>();
    P.q = ts[TQ].data_ptr<scalar_t>();
    P.v = ts[TV].data_ptr<scalar_t>();
    P.wrist = ts[TWRIST].data_ptr<scalar_t>();
    P.blade = ts[TBLADE].data_ptr<scalar_t>();
    P.wrist_vel = ts[TWVEL].data_ptr<scalar_t>();
    P.head = ts[THEAD].data_ptr<scalar_t>();
    P.shoulder_z = ts[TSHZ].data_ptr<scalar_t>();
    P.swing_acc = ts[TSWING].data_ptr<scalar_t>();
    P.axis = ts[TAXIS].data_ptr<scalar_t>();
    P.axis_valid = ts[TAXV].data_ptr<bool>();
    P.tr_acc = ts[TTRACC].data_ptr<scalar_t>();
    P.tr_age = ts[TTRAGE].data_ptr<int>();
    P.tr_cnt = ts[TTRCNT].data_ptr<int>();
    P.tr_active = ts[TTRACT].data_ptr<bool>();
    P.tr_ptr = ts[TTRPTR].data_ptr<int>();
    P.tick = ts[TTICK].data_ptr<int>();
    P.status = ts[TSTATUS].data_ptr<uint8_t>();
    P.first_pending = ts[TFIRSTP].data_ptr<int>();
    P.ep_return = ts[TEPRET].data_ptr<scalar_t>();
    P.good = ts[TGOOD].data_ptr<int>();
    P.bad = ts[TBAD].data_ptr<int>();
    P.miss = ts[TMISS].data_ptr<int>();
    P.bombs = ts[TBOMBS].data_ptr<int>();
    P.score = ts[TSCORE].data_ptr<scalar_t>();
    P.score_pre = ts[TSPRE].data_ptr<scalar_t>();
    P.score_acc = ts[TSACC].data_ptr<scalar_t>();
    P.score_post = ts[TSPOST].data_ptr<scalar_t>();
    P.contact_imp = ts[TCIMP].data_ptr<scalar_t>();
    P.contact_pt = ts[TCPT].data_ptr<scalar_t>();
    P.contact_arm = ts[TCARM].data_ptr<bool>();
    P.pd_hits = ts[TPD].data_ptr<int>();
    P.env_hits = ts[TENV].data_ptr<int>();
    P.fall_hits = ts[TFALL].data_ptr<int>();
    P.tilt_hits = ts[TTILT].data_ptr<int>();
    P.foot_lift_hits = ts[TLIFT].data_ptr<int>();
    P.foot_down_prev = ts[TFOOTDOWN].data_ptr<bool>();
    P.flail_rad = ts[TFLAILR].data_ptr<scalar_t>();
    P.flail_ticks = ts[TFLAILT].data_ptr<int>();
    P.wrev = ts[TWREV].data_ptr<int>();
    P.wrev_ext = ts[TWREVEXT].data_ptr<scalar_t>();
    P.wrev_dir = ts[TWREVDIR].data_ptr<int>();
    P.acc_pts = ts[TACCP].data_ptr<scalar_t>();
    P.acc_cuts = ts[TACCC].data_ptr<int>();
    P.ground_imp = ts[TGIMP].data_ptr<scalar_t>();
    P.limit_rest = ts[TLIMREST].data_ptr<scalar_t>();
    P.limit_rest_ticks = ts[TLIMTICKS].data_ptr<int>();
    P.sc_stat = ts[TSCSTAT].data_ptr<scalar_t>();
    P.act_stat = ts[TACTSTAT].data_ptr<scalar_t>();
    P.tau_stat = ts[TTAUSTAT].data_ptr<scalar_t>();
    P.prev_idx = ts[TPREVIDX].data_ptr<int>();
    P.prev_valid = ts[TPREVVALID].data_ptr<uint8_t>();
    P.tau_scale = ts[TTAUSCALE].data_ptr<scalar_t>();
    P.tau_jit = ts[TTAUJIT].data_ptr<scalar_t>();
    P.ep_rec = ts[TEPREC].data_ptr<scalar_t>();
    P.ep_flag = ts[TEPFLAG].data_ptr<uint8_t>();
    P.cw_mult = ts[TCWMULT].data_ptr<int>();
    P.cw_prog = ts[TCWPROG].data_ptr<int>();
    P.cw_combo = ts[TCWCOMBO].data_ptr<int>();
    P.cw_maxc = ts[TCWMAXC].data_ptr<int>();
    P.cw_breaks = ts[TCWBREAKS].data_ptr<int>();
    P.cw_score = ts[TCWSCORE].data_ptr<scalar_t>();
    P.cw_note_mult = ts[TCWNOTEMULT].data_ptr<uint8_t>();
    P.tr_msum = ts[TTRMSUM].data_ptr<int>();
    P.bench_rec = ts[TBENCHREC].data_ptr<scalar_t>();
    P.bench_flag = ts[TBENCHFLAG].data_ptr<uint8_t>();
    P.palm_vel = ts[TPALMV].data_ptr<scalar_t>();
    P.tip_vel = ts[TTIPV].data_ptr<scalar_t>();
    P.bench_refill = bench_refill != 0;
    P.tau_rand = (scalar_t)tau_rand;
    P.combo_walk = combo_walk != 0;
    P.ep_stats = ep_stats != 0;
    P.chart_idx = ts[TCIDX].data_ptr<int64_t>();
    P.p_hit = ts[PHIT].data_ptr<scalar_t>();
    P.p_speed = ts[PSPEED].data_ptr<scalar_t>();
    P.p_center = ts[PCENTER].data_ptr<scalar_t>();
    P.p_dvec = ts[PDVEC].data_ptr<scalar_t>();
    P.p_rot = ts[PROT].data_ptr<scalar_t>();
    P.p_good_hy = ts[PGHY].data_ptr<scalar_t>();
    P.p_bad_hy = ts[PBHY].data_ptr<scalar_t>();
    P.p_valid = ts[PVALID].data_ptr<bool>();
    P.p_is_right = ts[PISR].data_ptr<bool>();
    P.p_any_dir = ts[PANY].data_ptr<bool>();
    P.p_is_bomb = ts[PBOMB].data_ptr<bool>();
    P.p_is_head = ts[PHEAD].data_ptr<bool>();
    P.p_is_link = ts[PLINK].data_ptr<bool>();
    P.p_show_dir = ts[PSHOW].data_ptr<bool>();
    P.p_free_pre = ts[PFPRE].data_ptr<bool>();
    P.p_free_post = ts[PFPOST].data_ptr<bool>();
    P.p_len = ts[PLEN].data_ptr<scalar_t>();
    P.p_sweet = ts[PSWEET].data_ptr<scalar_t>();
    P.p_n_notes = ts[PNNOTES].data_ptr<int64_t>();
    P.p_last_hit = ts[PLASTHIT].data_ptr<scalar_t>();
    P.p_tail = ts[PTAIL].data_ptr<scalar_t>();
    P.p_min_speed = ts[PMINSP].data_ptr<scalar_t>();
    P.p_note_off = ts[PNOFF].data_ptr<int64_t>();
    P.p_wall_t = ts[PWALLT].data_ptr<scalar_t>();
    P.p_wall_cx = ts[PWALLCX].data_ptr<scalar_t>();
    P.p_wall_cy = ts[PWALLCY].data_ptr<scalar_t>();
    P.p_wall_w = ts[PWALLW].data_ptr<scalar_t>();
    P.p_wall_h = ts[PWALLH].data_ptr<scalar_t>();
    P.p_wall_len = ts[PWALLLEN].data_ptr<scalar_t>();
    P.p_wall_speed = ts[PWALLSP].data_ptr<scalar_t>();
    P.p_wall_valid = ts[PWALLVALID].data_ptr<bool>();
    P.p_n_walls = ts[PNWALLS].data_ptr<int64_t>();
    P.p_wall_off = ts[PWOFF].data_ptr<int64_t>();
    P.wall_passed = ts[TWALLPASS].data_ptr<bool>();
    P.first_wall = ts[TFIRSTW].data_ptr<int>();
    P.walls_in = ts[TWALLSIN].data_ptr<int>();
    P.in_wall_prev = ts[TINWALLPREV].data_ptr<bool>();
    P.env_order = ts[TENVORD].data_ptr<int>();
    P.W = (int)ts[TWALLPASS].size(1);
    P.actions = actions.data_ptr<int64_t>();
    P.cand = cand.data_ptr<int64_t>();
    P.ret_buf = ts[RBUF].data_ptr<scalar_t>();
    P.ret_total = ts[RTOT].data_ptr<int>();
    P.obs = ts[OOBS].data_ptr<float>();
    P.reward = ts[OREW].data_ptr<scalar_t>();
    P.done = ts[ODONE].data_ptr<bool>();
    P.N = (int)N;
    P.C = (int)C;
    P.max_steps = (int)max_steps;
    P.cap = (int)cap;
    P.mode = (int)mode;
    P.fall_const = (scalar_t)fall_const;
    P.foot_z_coef = (scalar_t)foot_z_coef;
    P.foot_x_coef = (scalar_t)foot_x_coef;
    P.foot_hinge_r = (scalar_t)foot_hinge_r;
    P.foot_shape_coef = (scalar_t)foot_shape_coef;
    P.energy_coef = (scalar_t)energy_coef;
    P.limit_coef = (scalar_t)limit_coef;
    P.limit_arms = (int)limit_arms;
    P.self_contact_coef = (scalar_t)self_contact_coef;
    P.wall_despawn = (int)wall_despawn;
    P.z_asinh = (scalar_t)z_asinh;
    P.foot_pot = ts[TFOOTPOT].data_ptr<scalar_t>();
    P.body_pt = ts[TBPPOS].data_ptr<scalar_t>();
    P.body_pt_vel = ts[TBPVEL].data_ptr<scalar_t>();
    P.pose_jit = ts[TPOSEJIT].data_ptr<scalar_t>();
    TORCH_CHECK(ts[TSTARTKICK].numel() == 2 * N
                && ts[TSTARTKICK].is_contiguous(),
                "start_kick must be a contiguous (N, 2) tensor");
    P.start_kick = ts[TSTARTKICK].data_ptr<scalar_t>();
    P.start_pose_jitter = (scalar_t)start_pose_jitter;
    P.crouch_prob = (scalar_t)crouch_prob;
    P.gears = (int)gears;
    P.auto_reset = auto_reset;
    P.want_contacts = want_contacts;
    const unsigned blocks = (unsigned)((N + THREADS - 1) / THREADS);
    // Flux (FX) is a compile-time switch (2 instantiations per dtype;
    // the foot-mode switch and its weld instantiations were deleted
    // 2026-09-27); the standard game's arithmetic is untouched. (Audit #5a, 2026-09-26: forcing the max-L1 carveout via
    // cudaFuncSetAttribute measured null on the RTX 5090 — the driver
    // already picks it for this 0 B-shared kernel — so it is not set.)
    auto stream = at::cuda::getCurrentCUDAStream();
    if (flux)
      step_kernel<scalar_t, true><<<blocks, THREADS, 0, stream>>>(P);
    else
      step_kernel<scalar_t, false><<<blocks, THREADS, 0, stream>>>(P);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  });
}

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("step", &step,
        "One v10 sim step (mode 0), reset-all (mode 1) or FK-cache "
        "refresh after a state write (mode 2) for every env; state "
        "tensors mutated in place, obs/reward/done written into their "
        "output slots. Feet are always in env v11 unilateral ground "
        "contact; flux selects the Flux game mode (circles + bars, "
        "palms)");
}
