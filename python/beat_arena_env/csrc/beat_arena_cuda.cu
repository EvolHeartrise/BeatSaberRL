// Beat Arena env v10 sim step as ONE CUDA kernel launch (P5).
//
// Full SPEC-v10 pipeline per env: 8 x substep (FK -> CRBA -> RNEA ->
// dense Cholesky -> constraint solve [12 x weld-block / contacts /
// limit rows / omega-cap pass, weld-null-space projection, COMPOSITE_EPS
// 1e-4] -> integrate -> step-11 clamp passes -> step-12 root envelope)
// + gear->tau, work-based effort, blade sweep/cut judgment, scoring,
// obs write (OBS_DIM below) — a semantics-exact transcription of rust/src/lib.rs +
// rust/src/dynamics/{model,fk,algo,collide,solver,step}.rs. The Rust
// crate remains THE reference implementation.
//
// ENV v11 (2026-08-08, "Unilateral foot contact" spec), foot_mode = 1:
// the 12 weld rows are DROPPED and the 8 sole-box corners get unilateral
// frictional GROUND rows against the plane y = 0 (1 normal + 2
// box-clamped friction per ACTIVE corner), plus a fall termination at
// root y < FALL_ROOT_Y. It is a COMPILE-TIME mode — step_kernel<T, FC>,
// selected on the host — so foot_mode = 0 is the v10 kernel unchanged;
// see the rationale block at GROUND_CORNERS.
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
// landed them (weld block, contacts, limit-row unit columns, plus the
// weld dot/axpy interchanges) for +33% end to end: train --bench
// N=1024 5,496 -> 7,283 steps/s, env micro-bench 6,212 -> 8,264
// env-steps/s. The "~2x" estimate of the prize was the honest half.
//
// f64 parity tier (SPEC-v10 Divergences §21, measured in
// tests/test_v10_parity_cuda.py): CUDA f64 sqrt and division are
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
constexpr int N_WELD = 12;
constexpr int N_TRACKERS = 16;  // FOLLOW_TIMEOUT + 1 (aggregated ring)
constexpr int N_NOTE_SLOTS = 24;  // env v18: 16 -> 24 (eviction study)
// env v17 (contact spec §12): shared entity window — notes AND walls
// ride the N_NOTE_SLOTS slots, nearest-first by head distance;
// per-token = the v9 note lanes then [is_wall, width, height, length].
constexpr int NOTE_FEAT = 18;
constexpr double WALL_PEN = -0.5;       // ONCE, on the tick the head ENTERS a wall (2026-08-20; was -0.05/tick)
constexpr double WALL_FRONT_OFF = 0.25; // viewer front-face offset (m)
// ============================ PROVENANCE WARNING =====================
// (2026-08-16) EVERY measurement, occupancy figure, and perf verdict
// in this header and the K-series notes below was taken on sm_61
// (20-SM GTX 1080). None has been re-measured on the production RTX
// 4090 (128 SMs, sm_89) or at the current batch shapes. Treat them as
// provenance for the do-not-retry list, NOT as current truth. The
// current measurement plan (Nsight run, bandwidth-vs-latency
// disambiguation, layout candidates) lives in
// projects/beat-arena/PERF_AUDIT_2026-08-16.md.
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
// K7 (2026-08-07) — BATCH ensure_pcol OVER THE LIMIT-ROW GROUP, the same
// RHS-batching transform applied to the last serial per-column pass in
// the constraint stage. The limit rows' projected columns were computed
// one joint at a time, each re-reading all of s.jw (144 loads), its own
// 12x12 solve, and all of s.mw (420 loads). Batched 4-wide those reads
// are shared, so a group of four costs one traversal instead of four:
// 24,224 -> 25,370 env-steps/s at N=4096, +4.73% (5 interleaved rounds,
// non-overlapping ranges). Width ladder and the two rejected widths are
// tabulated at the call site.
//
// SPILL-CLIFF STATUS AFTER K7 — the headroom is GONE. f32 goes 249 ->
// 255 registers, still 0 spill stores / 0 spill loads, frame 43,024 ->
// 43,216 B (the +192 is exactly the rw[4][12] batch). So K7 spent all
// six registers K6 bought and the kernel is now pinned AT the 255 cap:
// the NEXT item that adds a live value falls off the cliff, and the
// "check the build log before reading a micro-bench number" rule is now
// the first thing K8/K9/K11 must do, not the last.
// f64 crossed its OWN cliff on the same edit, in the opposite direction
// from what its history predicted: step_kernel<double> went 90,992 B /
// 12,308 / 10,300 spills to 85,536 B / ZERO spills at the same 255
// registers, and its SASS shrank 56,130 -> 38,302 instructions (LDL
// 12,493 -> 6,624, STL 4,744 -> 2,972). Nothing about the f64 arithmetic
// got smaller; ptxas simply found a schedule that fits once the
// per-column pcol pass stopped being four separate live regions. So the
// K6 header's "the f64 kernel cannot fit either way" is now FALSE — f64
// is on the cliff edge too, and K8/K9/K11 must gate BOTH dtypes' spill
// counters, not just f32's.
//
// K8+K9 (2026-08-07) — REFINE the K5 weld support and stop materialising
// the weld Jacobian twice. Three pieces, each probed against the build
// below it, 5 interleaved A/B rounds at N=4096 apiece:
//   K8   support 12 cols -> 9 (linear rows) / 7 (angular)
//                                 25,363.7 -> 25,690.3   +1.29%   KEPT
//   K9a+b one ancestor walk per foot, single-write stores
//                                 25,706.3 -> 25,719.4   +0.10%   KEPT
//   K9c  build J_w straight into s.mw; compact 84-value support store
//                                 25,706.3 -> 25,821.3   +0.45%   KEPT
//   PACKAGE                       25,359.7 -> 25,828.1   +1.85%
// (all four medians of 5; every pair had non-overlapping ranges.)
// The derivation is at WELD_FRAMES: K5 took the six root columns
// wholesale, but a linear row's column 3+t is a STRUCTURAL +0.0 (the
// t-th component of cross(rel0, e_t)) and an angular row touches only
// 3+t of them; each row also has exactly one entry that is exactly 1.0,
// contracted as a bare add (1.0*x == x unconditionally). nnz 144 -> 96,
// of which 12 need no multiply. K9c then drops s.jw entirely: the rows
// are built in s.mw (which the 12-RHS solve overwrote in place anyway),
// the 420-load/420-store seed copy is gone, and the raw Jacobian
// survives as s.jwc[12][6] + s.jwr[6][2] = 84 numbers instead of 420.
// f32 frame 43,216 -> 41,872 B, still 255 registers and ZERO spills.
// MEASURED AND REJECTED (recorded at the site): dropping the forced
// `#pragma unroll` on the A_ww r/c nest costs -0.12% on f32
// (25,838.9 -> 25,809.0) even though it clears f64's 8-byte spill and
// shrinks the SASS — the unrolled nest folds WJ_ONE/WJ_ANG/WJ_R2 to
// immediates, and f32 is the production dtype.
// SPILL STATUS: f32 41,872 B / 0 / 0 / 255 regs; f64 82,912 B / 8 B
// spill stores / 8 B spill loads / 255 regs. f64 picked up that 8-byte
// spill at the K8 step and never lost it; it is one 64-bit value and
// f64 still measures ~1% off f32's shape, so it was not traded for the
// f32 point above. f32 has NO headroom: still pinned at 255.
// SASS f32 33,407 -> 38,070 instrs (LDL 6,595 -> 6,548, STL 2,882 ->
// 3,042); f64 38,302 -> 43,290 (LDL 6,624 -> 6,570, STL 2,972 ->
// 2,901). Static size GREW while the kernel got faster — the pragmas
// above trade code for immediates; static counts are not dynamic
// traffic (the same caveat K7 recorded).
//
// K11 (2026-08-07) — RHS-MINOR s.mw, and the death of the LDL-width
// theory. The task was posed as a SASS census: count vector vs scalar
// local loads in the mw consumers and transpose [N_WELD][NV] ->
// [NV][N_WELD] only if the scalar loads could be fused. The census
// (nvcc -lineinfo + nvdisasm -c -g, per-source-line attribution) said
// they were 100% scalar, so the transpose was built — and it is worth
// +7.06%, 25,769.5 -> 27,588.8 env-steps/s at N=4096 (6 interleaved A/B
// rounds, medians, ranges 25,750.7-25,778.3 vs 27,559.6-27,615.0, i.e.
// non-overlapping by 30x the spread). But NOT ONE LOAD WAS FUSED: f32
// LDL 6,548 -> 6,548, STL 3,042 -> 3,042, f64 6,570 / 2,901 likewise
// unchanged; total instructions 38,070 -> 38,088 and 43,290 -> 43,308.
// The whole gain is ADDRESS ARITHMETIC AND SCHEDULING — 12 consecutive
// words off one base with immediate offsets, instead of 12 independent
// runtime-stride addresses live at once — and that is the general
// lesson for this kernel: it is register- and issue-bound, so a layout
// change should be judged by how many ADDRESSES its inner loop must
// keep live, not by LDL width. The corollary is that the SASS LDL
// census, which K6/K7/K8 all leaned on, is a WEAK instrument here: it
// was flat across the single largest win of the round.
// Why ptxas would not fuse: on sm_61 it vectorizes local accesses only
// in pure copy patterns (the one site in this kernel is
// `x[b][w] = rw[b][w]`, 18 LDL.128 and zero scalars). Feed the same 12
// consecutive words into FFMAs and it emits 12 scalar LDL regardless of
// layout or alignment — __align__(16) on mw was measured at +0.07% =
// noise and cost 16 B of f64 frame (rejected, recorded at the member).
// SPILL STATUS UNCHANGED, both dtypes: f32 41,872 B / 0 / 0 / 255 regs,
// f64 82,912 B / 8 / 8 / 255 regs — byte for byte what K8+K9 left.
// s.c_minv and the ensure_pcol batches stay row-major on
// chol_solve_multi_n; only s.mw uses the _tn twin.
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
// compile-time coupling for no measured return; the derivation stands
// at ANC_WELD_D if anyone wants it back.
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
// K15 (2026-08-07) — FOLD THE STEP-6 SOLVE INTO THE WELD MULTI-SOLVE AS
// A 13th RHS: BUILT, FULLY GATED, AND REVERTED at −0.06% on f32 (noise)
// even though it is +0.90% on f64 and PASSED EVERY OTHER GATE, including
// the spill gate the round had it flagged as most likely to fail. DO NOT
// RE-TRY IT IN THIS FORM. This is the round's cleanest example of the
// A1/1f -> K10e1 -> K19 family: a pure traffic-and-pass removal on a
// serial-latency-bound kernel buys nothing on the dtype that matters.
// WHAT WAS BUILT, exactly as the audit specified. s.mw widened to
// [NV][13] (RHS-minor, K11 layout — lane 12 is one more immediate offset
// off the same base); step 4's rhs assembly left AT ITS INLINE SITE in
// substep() with only its destination changed (d.rhs[i] -> s.mw[i][12]),
// which is legal because the weld assembly writes only lanes 0..11 and
// nothing between the seed and the solve touches lane 12; the standalone
// `chol_solve_n(NV, d.m, d.rhs)` DELETED; chol_solve_multi_tn<12> ->
// <13>; and `v[i] = d.v[i] + H_SUB * s.mw[i][12]` moved into
// constraint_solve immediately after the widened solve (v is first read
// ~160 lines later, at the PGS weld dot, so the move is safe). d.rhs
// removed from Dyn. Net −1,140 dynamic local loads per substep and one
// fewer serially-scheduled solve pass.
// PTXAS — BYTE-IDENTICAL TO THE INCUMBENT IN BOTH DTYPES, which is the
// first surprise: f32 41,872 B / 0 / 0 / 255 regs, f64 82,912 B / 8 / 8
// / 255. The audit ranked this the HIGHEST spill risk of the round on
// the grounds that the forward block's accumulators go 36 -> 39 live
// (s0/s1/s2 gain a lane each) at zero margin. They do, and ptxas
// absorbed all three at the same 255-register allocation. The frame did
// not move either, because mw's +140 B (f32) is exactly d.rhs's −140 B.
// So the 255-register pin is NOT a hard ceiling on live values — it is
// where the allocator parks; the K13 note that "the allocator found a
// 254-register allocation for a different edit" now has a companion in
// the other direction. Stop treating the cliff as one added value away.
// BITWISE, and the argument is worth keeping. Max diff EXACTLY 0.0 on
// f32 AND f64, all 10 dumped arrays byte-equal, over five configs:
// free-running with want_contacts, max_steps=60, the 128-env random-gear
// traffic, and K13's stopjam / altstop limit-heavy configs. The solve
// lane 12 rides is chol_solve_multi_tn (forward row-blocked R=3) where
// it used to be chol_solve_n (R=4) — DIFFERENT blocking, same values,
// because the K6 forward argument makes every R produce each row's terms
// as k = 0..i−1 ascending from the same seed x[i]; the divides and the
// back half are identical. The K10e2 hazard did not fire: `s −= l*x` is
// one mul+add with exactly one contraction available and no freedom, and
// the two expression trees that DID move (the rhs assembly's destination,
// the v* axpy's enclosing function) were relocated whole, which K19
// already showed is safe. pytest 172 green.
// A/B, N=4096, fresh process per arm, TWO order-balanced batches:
//   batch 1 (incumbent first)  incumbent 28,517.4 (28,497.8 - 28,532.6)
//                              K15       28,495.4 (28,478.9 - 28,510.9)
//                              −0.077%, 5/5 paired losses
//   batch 2 (K15 first)        K15       28,497.8 (28,493.0 - 28,512.0)
//                              incumbent 28,499.5 (28,487.0 - 28,532.5)
//                              −0.006%, 2/5 paired wins
//   pooled (10 pairs)          incumbent 28,514.1  K15 28,496.4
//                              −0.062%, 8/10 paired losses, ranges
//                              overlapping over most of their extent
// i.e. NOISE-to-marginal-negative, and reverted on the round's standing
// rule that a non-positive f32 verdict is a revert.
// THE f64 SPLIT IS THE FINDING, and it is new. The SAME .pyd is
// +0.90% on f64 — 16,869.3 (16,868.9 - 16,939.0) -> 17,021.7 (17,017.1
// - 17,042.5), 3/3 paired wins, non-overlapping — while f32 is flat.
// Both dtypes lost the identical pass and the identical ~1,140 loads;
// the difference is that an f64 load moves twice the bytes against the
// same 45/91 KB local working set, so on f64 the removed traffic was NOT
// already hidden behind the serial chains and on f32 it was. That is the
// A1/1f rule with a MEASURED BOUNDARY on it for the first time: "traffic
// cuts do not pay here" is a statement about f32's arithmetic-to-traffic
// ratio, not about the transform. f32 is the production dtype, so the
// verdict follows f32 (the K8 no-unroll precedent) — but an item that
// looks traffic-shaped should now be measured on BOTH dtypes before it is
// called dead, and if the f64 tier ever becomes load-bearing this edit is
// worth exactly +0.9%.
// K15B, THE PADDED-STRIDE VARIANT, MEASURED AND REJECTED HARDER at
// −0.46%. Hypothesis after the flat f32 result: the gain was eaten by
// address arithmetic, because the mw row stride went 12 (two shifts) to
// 13 (three, or an XMAD chain) at every one of the four hot mw
// consumers — K17's power-of-two [NF][16] finding says that is a real
// tax. So mw was padded to [NV][16] with chol_solve_multi_tn
// generalized to <W, T, STRIDE> (the K18(i) generalization, re-derived)
// and W = 13 riding a 16-word row, making every mw address a single
// shift — CHEAPER than the incumbent's 12. It measured 28,525.9
// (28,477.7 - 28,549.8) -> 28,395.7 (28,386.3 - 28,428.5), −0.457%,
// 5/5 paired losses, non-overlapping in the wrong direction, at f32
// 42,288 B / 0 / 0 / 255 regs (frame +416 B, still no spills). So the
// stride hypothesis is FALSIFIED: 105 dead words per thread of frame
// cost more than the address arithmetic saved, and the 12 -> 13 stride
// was never the tax. The transferable half: on this kernel a
// power-of-two stride is worth its padding only when the padding is
// SMALL relative to the frame (K17's tables live in global memory and
// cost the frame nothing, which is why that one paid).
//
// K20 + K16 (2026-08-07) — THE HOUSEKEEPING BUNDLE, and the round's
// biggest surprise-to-effort ratio. Two of the three K20 items landed
// for +2.08% together; the third is DEAD ON ITS PREMISE; K16 is dead on
// its measurement. Package, 5 interleaved A/B rounds N=4096 f32, fresh
// process per arm: 28,932.0 (28,923.7 - 28,945.3) -> 29,532.5
// (29,524.2 - 29,535.3), +2.08%, 5/5 paired wins, non-overlapping by
// ~25x the within-arm spread.
//
// K20-1 ensure_pcol COPY -> RE-READ ELIMINATION: +1.96%, and the audit
// priced it at 0.2%. ensure_pcol used to COPY the plain column into the
// projected one (col[i] = s.ecol[j][i], fused into the dot pass plus a
// copy-only range over VOFF..WJ_L0) and then the projection loop re-read
// col[i] as its seed. The projection now reads s.ecol[j][i] DIRECTLY and
// writes col[i] only as its OUTPUT, so the copy disappears entirely —
// both in the scalar body and in K7's ensure_pcol_multi<B>.
// BITWISE by inspection and gated at max diff EXACTLY 0.0 on f32 AND f64
// across all five configs (free-running want_contacts, max_steps=60,
// 128-env random-gear, stopjam, altstop), all 10 arrays byte-equal:
// col[i] was set to EXACTLY s.ecol[j][i] over the whole 0..NV-1 range
// (the four segments tile it, by the WJ_R0 + 6 == NV static_assert), so
// seeding from the source is the same bytes; nothing between the two
// passes touches either buffer, and s.ecol / s.pcol are distinct members.
//   A/B N=4096 f32, fresh process per arm, TWO order-balanced batches:
//     batch 1 (incumbent first) 28,512.5 (28,507.5-28,550.5) ->
//                               29,084.0 (29,056.8-29,091.1)  +2.00%
//     batch 2 (K20-1 first)     28,504.2 (28,497.2-28,543.4) ->
//                               29,071.0 (29,034.3-29,110.3)  +1.99%
//     pooled (10 pairs)         28,517.3 -> 29,075.7  +1.96%, 10/10
//     paired WINS, ranges non-overlapping by ~15x the spread.
//   f64 16,941.2 -> 17,165.5, +1.32%, 3/3.
//   ptxas BYTE-IDENTICAL both dtypes: f32 41,872 B / 0 / 0 / 255 regs,
//   f64 82,912 B / 8 / 8 / 255. pytest 172 green.
// WHY IT IS 10x THE ESTIMATE, and this is the transferable half. The
// audit costed it as traffic (35 stores + 35 loads per column), and this
// file's standing rule — A1/1f -> K10e1 -> K19 -> K15 — says pure
// traffic cuts do not pay on f32. They still don't: what this removes is
// a STORE -> LOAD FORWARDING DEPENDENCY. The projection's seed for row i
// was a local store issued a few instructions earlier in the same
// function, so every one of the 35 rows had to wait for the copy's store
// to land before its own FFMA chain could start; reading s.ecol instead
// makes all 35 seeds independent loads that ptxas can hoist ahead of the
// chain. The SASS census agrees on the traffic and understates the win,
// as usual here (f32 LDL 6,584 -> 6,447, STL 3,048 -> 2,768, instrs
// 38,334 -> 37,932). RULE: a removed load that was feeding on a value
// this thread just STORED is not a traffic cut, it is a chain cut —
// price it like A4/K6, not like K19.
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

// Occupancy-ladder knob (WARP_RESCUE_LEGACY_SIDE.md §1, 2026-08-15):
// every historical "occupancy is not the lever" verdict — including the
// paragraph above — was measured GRID-STARVED (<= 8 blocks/SM even
// requested). BEAT_MIN_BLOCKS pins minBlocksPerMultiprocessor for the
// legacy kernel exactly as hard as the (THREADS, 1) pin does, so the
// free-choice instability the paragraph above documents cannot occur on
// any rung; the register CAP that comes with K > 8 (65,536/32/K) is the
// experiment. Default 1 == the shipped kernel, bit for bit. The ladder
// is only meaningful at N >= 32,768 (below that the grid supplies fewer
// blocks than the pin asks for). Set via jit_cache BEAT_MIN_BLOCKS=K
// (part of the flag digest => forces the rebuild that re-censuses).
#ifndef BEAT_MIN_BLOCKS
#define BEAT_MIN_BLOCKS 1
#endif

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
// GROUND_IMP_NORM. All-zero in weld mode, so lanes 0..340 are the v10
// obs verbatim.
constexpr int OFF_GROUND_IMP = OFF_NOTE_MASK + N_NOTE_SLOTS;         // 558
constexpr int N_GROUND_IMP_LANES = 4;
// env v13 (contact spec §8): [l_elbow, r_elbow, l_knee, r_knee,
// l_foot, r_foot] world positions, head-style normalization.
constexpr int OFF_BODY_PTS = OFF_GROUND_IMP + N_GROUND_IMP_LANES;    // 562
constexpr int N_BODY_PTS = 6;
// env v24: world velocities of the §8 points (lib.rs mirror)
constexpr int OFF_BODY_PT_VEL = OFF_BODY_PTS + 3 * N_BODY_PTS;       // 580
constexpr int OBS_DIM = OFF_BODY_PT_VEL + 3 * N_BODY_PTS;            // 598
static_assert(OBS_DIM == 598, "spec §12 (env v24 body_pt_vel lanes): OBS_DIM must be 598");
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
constexpr double BETA_WELD = 0.2;
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
// swing the composite mass; inertia rescaled with it (model.rs mirror)
__device__ constexpr double F_MASS[NF] = {
    0, 0, 24.20, 5.20, 0, 0, 2.00, 1.20, 0, 0, 0.35,
    0, 0, 2.00, 1.20, 0, 0, 0.35,
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
    {0, 0, 0}, {0, 0, 0}, {0.000264, 0.000264, 0.000264},
    {0, 0, 0}, {0, 0, 0}, {0.029222, 0.029222, 0.001966},
    {0.013056, 0.013056, 0.000931},
    {0, 0, 0}, {0, 0, 0}, {0.000264, 0.000264, 0.000264},
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
__device__ constexpr double TAU_MAX[NQ] = {
    600.00, 239.95, 375.00,
    85.38, 85.38, 19.45, 19.13, 1.06, 1.06, 1.06,
    85.38, 85.38, 19.45, 19.13, 1.06, 1.06, 1.06,
    262.50, 75.00, 135.00, 150.00, 135.00, 37.50,
    262.50, 75.00, 135.00, 150.00, 135.00, 37.50};
__device__ constexpr double OMEGA_CAP[NQ] = {
    16.0, 16.0, 16.0,
    16.0, 16.0, 16.0, 16.0, 50.0, 50.0, 50.0,
    16.0, 16.0, 16.0, 16.0, 50.0, 50.0, 50.0,
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

// feet weld (solver.rs)
__device__ constexpr int WELD_FRAMES[2] = {F_L_FOOT, F_R_FOOT};
// K5 — STRUCTURAL SPARSITY of the 12 weld Jacobian rows. A weld row is
// built by point_jacobian_acc / angular_jacobian_acc, which touch
// row[VOFF + F_COORD[k]] for k on the frame's ancestor chain plus
// row[0..5]; every other column is left at the +0.0 of the row's zero
// fill. Walking F_PARENT from F_L_FOOT = 23 gives frames
// 23,22,21,20,19,18 -> F_COORD 22,21,20,19,18,17 -> v-columns 23..28,
// and from F_R_FOOT = 29 frames 29,28,27,26,25,24 -> v-columns 29..34.
// So rows 0..5 have support {0..5} u {23..28} and rows 6..11 have
// {0..5} u {29..34}: 12 of NV = 35 columns.
constexpr int WJ_L0 = 23;
constexpr int WJ_R0 = 29;
static_assert(WJ_R0 == WJ_L0 + 6 && WJ_R0 + 6 == NV && N_WELD == 12,
              "K5: weld support layout is {0..5} u {L0..L0+5} / {R0..NV}");
//
// K8 (2026-08-07) — the SAME derivation carried into the six root
// columns, which K5 took wholesale. Row r of the weld block is
// r = 6*widx + m; t = m % 3 and the row is ANGULAR iff m >= 3.
//
//   ANGULAR row (angular_jacobian_acc, sign = +1): the ancestor walk
//   writes only row[VOFF + F_COORD[k]], then `row[3 + axis_t] += sign`
//   -- ONE root column, holding exactly +0.0 + 1.0 = 1.0. Columns 0..2
//   and the other two of 3..5 are never touched.
//        support = {3 + t} u chain6  ->  7 columns, one of them 1.0.
//
//   LINEAR row (point_jacobian_acc with n = e_t, sign = +1): the walk
//   writes row[VOFF + F_COORD[k]] += dot3(e_t, w) = w[t]; then
//   `row[tt] += n[tt]` gives row[t] = +0.0 + 1.0 = 1.0 and leaves the
//   other two of 0..2 at +0.0; then rxn = cross(rel0, e_t) is added
//   into row[3..5]. cross(a, e_t)[t] is structurally a[j]*0 - a[k]*0,
//   i.e. +-0.0, and row[3+t] = +0.0 + (+-0.0) = +0.0 EXACTLY (the zero
//   fill's +0.0 absorbs the sign -- Divergences 20c). The other two
//   components are +-a[.] and generally nonzero.
//        support = {t} u ({3,4,5} \ {3+t}) u chain6  ->  9 columns,
//        one of them 1.0.
//
// nnz 6*9 + 6*7 = 96 vs K5's 144. Per-column contributor counts (used
// by the two i-outer dots): i in 0..2 -> the 2 linear rows with t == i;
// i in 3..5 -> the 2 angular rows with 3+t == i plus the 4 linear rows
// with t != i-3; chainL -> rows 0..5; chainR -> rows 6..11. That is
// 3*2 + 3*6 + 6*6 + 6*6 = 96, which checks.
//
// Every consumer keeps each accumulator's ASCENDING-i term order; the
// dropped products are (+-0.0)*x and the accumulators are +0.0-seeded,
// so the -0.0 channel is unreachable (K5's argument verbatim). The
// entries proven to be exactly 1.0 are contracted as a bare add:
// 1.0*x == x unconditionally, so that drop is exact even for +-0.0.
constexpr bool WJ_ANG(int r) { return (r % 6) >= 3; }
constexpr int WJ_T(int r) { return r % 3; }
constexpr int WJ_CH0(int r) { return WJ_L0 + 6 * (r / 6); }
// the ONE column of row r whose entry is exactly 1.0
constexpr int WJ_ONE(int r) { return WJ_ANG(r) ? 3 + WJ_T(r) : WJ_T(r); }
// a LINEAR row t's two other root columns, {3,4,5} \ {3+t}, ASCENDING
__device__ constexpr int WJ_R2[3][2] = {{4, 5}, {3, 5}, {3, 4}};
// K9c compact-storage indices: linear rows 0,1,2,6,7,8 -> jwr slots
// 0..5; column 3+u of linear row t -> which of that row's two slots.
constexpr int WJ_LIN(int r) { return r < N_WELD / 2 ? r : r - 3; }
constexpr int WJ_RIDX(int t, int u) { return u > t ? u - 1 : u; }
static_assert(WJ_LIN(0) == 0 && WJ_LIN(2) == 2 && WJ_LIN(6) == 3
                  && WJ_LIN(8) == 5,
              "K9c: linear rows 0,1,2,6,7,8 map onto jwr slots 0..5");
static_assert(WJ_R2[0][WJ_RIDX(0, 1)] == 4 && WJ_R2[0][WJ_RIDX(0, 2)] == 5
                  && WJ_R2[1][WJ_RIDX(1, 0)] == 3
                  && WJ_R2[1][WJ_RIDX(1, 2)] == 5
                  && WJ_R2[2][WJ_RIDX(2, 0)] == 3
                  && WJ_R2[2][WJ_RIDX(2, 1)] == 4,
              "K9c: WJ_RIDX(t,u) is the slot holding column 3+u");
// re-derivation of the ancestor chains straight off F_PARENT/F_COORD:
// the bit mask of v-columns the walk touches must be exactly the
// 6-column segment [WJ_CH0, WJ_CH0+6). Equality pins count, contiguity,
// distinctness AND disjointness from the root columns 0..5 at once.
constexpr unsigned long long wj_chain_mask(int f) {
  unsigned long long m = 0;
  for (int k = f; k >= 0; k = F_PARENT[k])
    if (F_COORD[k] >= 0) m |= 1ull << (VOFF + F_COORD[k]);
  return m;
}
static_assert(wj_chain_mask(F_L_FOOT) == (0x3full << WJ_L0)
                  && wj_chain_mask(F_R_FOOT) == (0x3full << WJ_R0),
              "K8: weld ancestor chains are exactly v-columns 23..28 "
              "and 29..34 -- 6 distinct columns each, none in 0..5");
// K17, REVERTED ROW: the weld assembly's chase folds to immediates if
// its walk is unrolled at a compile-time length, and this is that
// length — kept as a compile-time claim (it costs nothing) because the
// item measured +0.02% and the derivation is the only expensive part.
// See the K17 header block before re-trying it.
constexpr int ANC_WELD_D = 6;
static_assert(ANC_N[F_L_FOOT] == ANC_WELD_D && ANC_N[F_R_FOOT] == ANC_WELD_D,
              "K17: each foot's coord-bearing chain is exactly 6 frames");
static_assert(WJ_ONE(0) == 0 && WJ_ONE(2) == 2 && WJ_ONE(3) == 3
                  && WJ_ONE(5) == 5 && WJ_ONE(6) == 0 && WJ_ONE(11) == 5
                  && WJ_CH0(5) == WJ_L0 && WJ_CH0(6) == WJ_R0,
              "K8: row -> (one-column, chain) map");
static_assert(WJ_R2[0][0] == 4 && WJ_R2[0][1] == 5 && WJ_R2[1][0] == 3
                  && WJ_R2[1][1] == 5 && WJ_R2[2][0] == 3
                  && WJ_R2[2][1] == 4,
              "K8: linear-row root extras are {3,4,5}\\{3+t}, ascending");
__device__ constexpr double WELD_TARGET_P[2][3] = {
    {-0.10, 0.07, -0.65}, {0.10, 0.07, -0.65}};

// ---- env v11 GROUND CONTACT (unilateral-contact spec §2/§3) -----------------
// The contact-mode replacement for the 12 weld rows: the 8 sole-box
// corners against the static plane y = 0 — one unilateral normal row
// plus two BOX-clamped friction rows per ACTIVE corner, assembled where
// the weld rows used to sit (so the §2 pair-id row order downstream is
// untouched). MIRROR sources: collide.rs `ground_phase`, solver.rs
// `assemble_ground_rows` / `pgs_ground_sweep` / `pgs_row_update_box`.
//
// WHY A TEMPLATE PARAMETER AND NOT A RUNTIME FLAG. `foot_mode` is baked
// per CudaBeatVecEnv instance and constant across launches, so it is
// warp-uniform AND launch-uniform: nothing is gained by branching on it
// at runtime and two things are lost. (1) The K13 rule — a data-dependent
// selector makes the warp pay the union of the arms; a compile-time one
// cannot. (2) The weld build must stay the v10 kernel EXACTLY: with
// `if constexpr` the ground block is discarded before ptxas sees it, the
// ground scratch collapses to a stub row (see Solver), and the f32/f64
// weld census stays byte-identical to the pre-v11 baseline — which is
// checkable, and was checked. The cost is 4 step_kernel instantiations
// instead of 2 (~2x compile time, still ~2.5 min on the pod).
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

// ---- env constants (lib.rs, verbatim) ---------------------------------------
constexpr double V_NORM = 6.0;
constexpr double V_MIN_CUT = 1.0;
constexpr double ACC_NORM = 0.3;
constexpr double COS_TOL = 0.5;
constexpr double Y_CENTER = 1.40;
constexpr double REWARD_BAD = -0.5;
constexpr double REWARD_MISS = -0.5;
constexpr double REWARD_BOMB = -1.5;
// env v12 (contact spec §7): per-tick survival bonus, contact mode only
// — see lib.rs ALIVE_BONUS for the sizing rationale. Overridden by the
// fall/escape terminal pricing.
constexpr double ALIVE_BONUS = 0.002;
constexpr double GOOD_HALF_X = 0.4;
constexpr double GOOD_HALF_Z = 0.5;
constexpr double GOOD_Z_SHIFT = -0.25;
constexpr double BAD_HALF = 0.175;
constexpr double BOMB_RADIUS = 0.18;
constexpr double SABER_LEN = 1.0;
constexpr int N_BLADE_SAMPLES = 8;
constexpr double REACH_PAST_SHOULDER = 0.65 + 1.0 + 0.75;
constexpr double PRE_FULL = 1.7453292519943295;
constexpr double POST_FULL = 1.0471975511965976;
constexpr double PTS_PRE = 70.0;
constexpr double PTS_POST = 30.0;
constexpr double PTS_ACC = 15.0;
constexpr double SCORE_MAX = 115.0;
constexpr double CHAIN_LINK_PTS = 20.0;
constexpr double SWING_BREAK = 0.005;
// env v19 flail telemetry: a tick is "note-free" when the next
// scoreable note is further out than this (s). Telemetry only — no
// reward or obs effect.
constexpr double FLAIL_LEAD_S = 1.0;
// env v21 (lib.rs mirrors, exact shared f64 literals): wide stance —
// hip_roll splays the legs +-8 deg, ankle_roll counter-rolls so the
// soles stay flat. Contact mode only (weld mode's feet are welded at
// the v10 reference). FOOT_X0 = the stance's per-side foot x (fk of
// the start pose; the root ground-drop does not move x).
constexpr double Q_START_HIP_ROLL = 0.13962634015954636;   // 8 deg
constexpr double FOOT_X0 = 0.22108059783525694;
constexpr int FOLLOW_TIMEOUT = 15;
constexpr double LOOK_VERT = 0.8;
constexpr double LOOK_DIST = 8.0;
constexpr double X_NORM = 1.0;
constexpr double Y_NORM = 0.75;
constexpr double Z_SCALE = 8.0;
constexpr double Z_CLIP_LO = -3.0;
constexpr double Z_CLIP_HI = 12.0;
constexpr double ROOT_Y0 = 0.94;
constexpr double HEAD_Y0 = 1.70;
constexpr double PLANE_Z0 = -0.65;
constexpr double ROOT_VLIN_NORM = 6.0;
constexpr double ROOT_VANG_NORM = 16.0;
// env v11 FALL termination (unilateral-contact spec §4), CONTACT MODE
// ONLY — lib.rs FALL_ROOT_Y. Root below this at the END of a tick
// (per-TICK since 2026-08-20, user: consistent with the tilt check;
// the per-substep dip test is retired) ends the episode with a ZEROED
// reward, reported exactly like the root-escape envelope (SPEC-v10
// §20e "reported, never hidden").
constexpr double FALL_ROOT_Y = 0.25;  // 2026-08-21: 0.5 -> 0.25 (user; follows the player-height revert to 1.8 — full crouch headroom restored)
// TILT termination (user 2026-08-20), CONTACT MODE ONLY — lib.rs
// mirror: CHEST up-axis tilted more than 40 deg FORWARD, 20 deg
// BACKWARD (asymmetric sagittal, user 2026-08-20 — the tight back
// bound targets the v30 backbend exploit; forward stays duck-sized)
// or 40 deg to the side (lateral) at the END of a tick ends the
// episode, priced like a fall (flat -fall_const). Checked per TICK,
// not per substep. Tangent-squared form on the chest up-axis;
// forward = up_z > 0 (body faces +z); also trips whenever
// up_y <= 0 (past horizontal).
constexpr double TILT_TAN2_FWD = 0.7040881910418472;   // tan^2(40 deg)
constexpr double TILT_TAN2_BACK = 0.1324743314317942;  // tan^2(20 deg)
constexpr double TILT_TAN2_ROLL = 0.7040881910418472;  // tan^2(40 deg)
// FOOT-LIFT penalty (user 2026-08-20), CONTACT MODE ONLY — lib.rs
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

// note kinds (constants.py)
// KIND_NOTE=0, KIND_BOMB=1, KIND_CHAIN_HEAD=2, KIND_CHAIN_LINK=3 —
// pool tensors carry the derived booleans directly.

// ---- small device math ------------------------------------------------------
template <typename T>
__device__ __forceinline__ T tmin(T a, T b) { return a < b ? a : b; }
template <typename T>
__device__ __forceinline__ T tmax(T a, T b) { return a > b ? a : b; }
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
  // dense — m holds M from crba, then its IN-PLACE Cholesky factor L
  T m[NV * NV], c[NV], rhs[NV];
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

// FC = env v11 `foot_mode` (unilateral-contact spec §1): false = weld
// (v10, the DEFAULT), true = unilateral frictional ground contact. It is
// a TEMPLATE parameter, not a field — see the rationale at
// GROUND_CORNERS.
//
// The two modes are MUTUALLY EXCLUSIVE (contact mode DROPS the 12 weld
// rows), and each block's scratch is therefore an EMPTY BASE in the
// other instantiation rather than a stub array. That is not tidiness:
// this kernel's whole history says frame size buys spills, and the
// measurement here says it twice. Carrying the other mode's arrays as
// [1]-row stubs cost the weld build 304 B (f32) / 608 B (f64) of frame
// it never touches, and carrying the weld arrays into the CONTACT build
// cost f64 1,382/1,736 B of SPILL that vanished when they went away.
// Empty bases with EBO cost exactly zero, and every reader of either
// block sits under an `if constexpr` on FC, so the names are not even
// looked up in the instantiation that lacks them.
//
// WELD-MODE scratch (foot_mode = 0). `pcol` belongs here too: with no
// weld block there is no null space to project into, so contact mode
// reads `ecol` wherever weld mode reads `pcol` (solver.rs skips the
// entire `if nweld > 0` block, leaving the plain M^-1 e_j column and
// its plain mobility). `pdiag` stays in Solver — it is written in BOTH
// modes, holding that plain mobility in contact mode.
template <typename T, bool FC>
struct WeldRows {
  // weld rows. K9c: the dense J_w is built STRAIGHT INTO mw (which the
  // 12-RHS solve then overwrites in place with M^-1 J_w^T), so the old
  // full jw[12][35] copy source is gone -- 1,680 B of f32 frame and a
  // 420-load/420-store seed pass. All four consumers contract a weld
  // row against its K8 support only, so the raw Jacobian survives as
  // 84 numbers: jwc[r][e] is chain column WJ_CH0(r)+e, jwr[WJ_LIN(r)][.]
  // the two extra root columns WJ_R2[t][.] of a linear row. The 12
  // exactly-1.0 entries are not stored at all (WJ_ONE(r) is implicit).
  // K11 — mw is stored RHS-MINOR, [NV][N_WELD], not [N_WELD][NV], worth
  // +7.06% (25,769 -> 27,589 env-steps/s at N=4096, 6 interleaved A/B
  // rounds, non-overlapping ranges). All four consumers sweep the 12
  // weld rows in the INNER loop at a fixed column i — the 12-RHS solve
  // (chol_solve_multi_tn), both ensure_pcol projections, the PGS weld
  // update — so RHS-minor makes each of those sweeps 12 CONSECUTIVE
  // per-thread words off ONE base register with immediate offsets;
  // row-major made them stride-NV, i.e. 12 independent addresses.
  //
  // WHY, and what it is NOT: the K11 premise was that contiguity would
  // let ptxas fuse the sweep into 3 LDL.128. IT DOES NOT. The SASS
  // census before and after is IDENTICAL to the instruction — f32 LDL
  // 6,548 -> 6,548 and STL 3,042 -> 3,042 (f64 6,570 / 2,901 likewise
  // unchanged), total instructions 38,070 -> 38,088 — and EVERY mw
  // sweep is still 12 scalar LDL (PGS weld update 420 -> 420, A_ww
  // chain 180 -> 180, batched projection 48 -> 48, 12-RHS solve 12 per
  // (i, k) either way). ptxas on sm_61 only vectorizes local accesses
  // in pure copy patterns: the one place it fires in this kernel is
  // `x[b][w] = rw[b][w]` (18 LDL.128, zero scalars) where the loaded
  // value feeds a store and nothing else. Feed the same 12 words into
  // FFMAs and it emits 12 scalar LDL no matter how they are laid out.
  // So the +7% is ADDRESSING AND SCHEDULING, not load width, and
  // forcing alignment buys nothing: __align__(16) on this member (which
  // would make all 35 rows 16-byte aligned) was MEASURED at +0.07% =
  // NOISE (27,609.5 vs 27,589.7 medians, 3 interleaved rounds,
  // overlapping ranges) and cost 16 B of f64 frame — REVERTED, do not
  // re-add it. Local memory is per-thread interleaved, so stride never
  // affected transactions either. LESSON: on this kernel, judge a
  // layout change by how many ADDRESS registers the inner loop needs
  // live, not by LDL width.
  T mw[NV][N_WELD];
  T jwc[N_WELD][6], jwr[N_WELD / 2][2], bias_w[N_WELD];
  T aww[N_WELD * N_WELD], rw[N_WELD];  // aww: A_ww, then in-place L
  T pcol[NQ][NV];  // weld-null-space-projected M^-1 e_j (limit + cap)
};
template <typename T>
struct WeldRows<T, true> {};  // contact mode: the weld block is dropped

// CONTACT-MODE scratch (foot_mode = 1, unilateral-contact spec §3): 3
// rows per ACTIVE sole corner, corners in the §2 normative order. Same
// field shape as the self-collision contact rows — dense J, its
// M^-1 J^T column, bias / diag / lambda — because the normal row's PGS
// update IS the contact path's and the friction rows differ from it only
// in the projection.
template <typename T, bool FC>
struct GroundRows {
  int ng;  // ACTIVE corners this substep, 0..NUM_GROUND_CORNERS
  T g_J[N_GROUND_ROWS][NV], g_minv[N_GROUND_ROWS][NV];
  T g_bias[N_GROUND_ROWS], g_diag[N_GROUND_ROWS], g_lam[N_GROUND_ROWS];
  // contact spec §7: 2-bit obs bucket (foot*2 + toe) of each ACTIVE
  // slot, packed 2 bits/slot — the compaction drops corner ids, and the
  // per-tick ground_imp aggregation needs them back after the solve.
  uint32_t g_bucket;
};
template <typename T>
struct GroundRows<T, false> {};  // weld mode: no ground rows exist

template <typename T, bool FC>
struct Solver : WeldRows<T, FC>, GroundRows<T, FC> {
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
  // limit rows (joint order)
  int nl;
  int l_j[NQ];
  T l_sign[NQ], l_bias[NQ], l_lam[NQ], l_diag[NQ];
  // shared per-joint columns: ecol = M^-1 e_j (plain; step-8 lazy cap +
  // step-11 clamps), pcol = weld-null-space-projected ecol (limit rows
  // + cap pass), pdiag = composite mobility (INF = immobile given the
  // welds). Bitwise identical to Rust's separately-computed
  // minvjt/capcol/solcache columns (linear solves commute with the +-1
  // row sign exactly; recomputation of the same expression tree is
  // deterministic).
  // (`pcol` lives in WeldRows — see the note there.)
  T ecol[NQ][NV], pdiag[NQ];
  bool ecomp[NQ], pcomp[NQ];
  bool hit[NQ];
};

// ---- fk.rs ------------------------------------------------------------------
template <typename T>
__device__ void rot_axis(const double axis_d[3], T angle, T out[3][3]) {
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
__device__ void rot_log(const T r[3][3], T out[3]) {
  T w0 = r[2][1] - r[1][2];
  T w1 = r[0][2] - r[2][0];
  T w2 = r[1][0] - r[0][1];
  T tr = r[0][0] + r[1][1] + r[2][2];
  T cth = tclamp((tr - T(1)) * T(0.5), T(-1), T(1));
  T theta = acos(cth);
  T factor = theta < T(1e-9) ? T(0.5) : theta / (T(2) * sin(theta));
  out[0] = w0 * factor;
  out[1] = w1 * factor;
  out[2] = w2 * factor;
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
// traffic bound: without it, a store to any Solver field (s.jw, s.rw,
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
      av[i] = rl[i][0] * T(F_ANCHOR[k][0]) + rl[i][1] * T(F_ANCHOR[k][1])
              + rl[i][2] * T(F_ANCHOR[k][2]);
      d.pos[k][i] = pl[i] + av[i];
    }
    if (F_COORD[k] >= 0) {
      T rj[3][3];
      rot_axis(F_AXIS[k], d.q[F_COORD[k]], rj);
      for (int i = 0; i < 3; i++)
        for (int j = 0; j < 3; j++)
          d.r[k][i][j] = rl[i][0] * rj[0][j] + rl[i][1] * rj[1][j]
                         + rl[i][2] * rj[2][j];
      for (int i = 0; i < 3; i++)
        d.axw[k][i] = d.r[k][i][0] * T(F_AXIS[k][0])
                      + d.r[k][i][1] * T(F_AXIS[k][1])
                      + d.r[k][i][2] * T(F_AXIS[k][2]);
    } else {
      for (int i = 0; i < 3; i++)
        for (int j = 0; j < 3; j++) d.r[k][i][j] = rl[i][j];
    }
  }
}

// ---- algo.rs ---------------------------------------------------------------
// I_world = R diag(i) R^T — upper computed, lower mirrored (bitwise
// symmetric, algo.rs inertia_world).
template <typename T>
__device__ void inertia_world(const T r[3][3], const double idiag[3],
                              T out[3][3]) {
  for (int a = 0; a < 3; a++)
    for (int b = a; b < 3; b++) {
      T v = r[a][0] * T(idiag[0]) * r[b][0] + r[a][1] * T(idiag[1]) * r[b][1]
            + r[a][2] * T(idiag[2]) * r[b][2];
      out[a][b] = v;
      out[b][a] = v;
    }
}

// Same, teed into two destinations from the one register value v (K4e):
// crba needs a MUTABLE composite seed and a pristine copy for rnea_bias,
// and both get identical bits with no read-back.
template <typename T>
__device__ void inertia_world2(const T r[3][3], const double idiag[3],
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

// LOWER TRIANGLE + DIAGONAL ONLY (2026-08-03, K4c). Every consumer of
// d.m — chol_factor_n (in place), chol_solve_n / chol_solve_multi_n{,_tn}
// ((i,k) k<i; (k,i) k>i; (i,i)), chol_fwd_unit{,2,4}_n ((i,k) m<=k<i;
// diag) and chol_back{,2,4}_n ((k,i) k>i; diag) — touches row >= col
// ONLY. Nothing in the kernel ever reads d.m above the diagonal, so this
// function no longer writes there: the NV*NV zero fill is narrowed to
// the 630 lower+diagonal entries (595 stores saved — and ~286 of the
// kept ones are genuinely load-bearing, entries no producer below
// touches), the symmetric ancestor pair writes one element, and the
// root-block mirror stores are gone. ~970 dead stores per substep.
//
// COLUMN-MAJOR ADDRESSES (K12, 2026-08-07): element (row R, col C) is at
// d.m[C*NV + R] — see chol_factor_n's header for why. Every store below
// is the SAME value at the transposed address; the set of entries
// written is unchanged (still exactly row >= col), and M's symmetry
// means the values were never orientation-dependent in the first place.
template <typename T>
__device__ void crba(Dyn<T>& __restrict__ d) {
  for (int i = 0; i < NV; i++)
    for (int j = 0; j <= i; j++) d.m[j * NV + i] = T(0);
  // composite seeds
  for (int k = 0; k < NF; k++) {
    const T mk = T(F_MASS[k]);
    d.cm[k] = mk;
    if (F_MASS[k] > 0.0) {
      for (int i = 0; i < 3; i++)
        d.cc[k][i] = d.pos[k][i] + (d.r[k][i][0] * T(F_COM[k][0])
                                    + d.r[k][i][1] * T(F_COM[k][1])
                                    + d.r[k][i][2] * T(F_COM[k][2]));
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
    d.bc[i] = d.p[i] + (d.r0[i][0] * T(BASE_COM[0])
                        + d.r0[i][1] * T(BASE_COM[1])
                        + d.r0[i][2] * T(BASE_COM[2]));
  inertia_world2(d.r0, BASE_INERTIA, d.bi, d.iw0);

  // leaf->root accumulation with column fill before parent merge
  for (int k = NF - 1; k >= 0; k--) {
    const int coord = F_COORD[k];
    const T m = d.cm[k];
    T cvec[3] = {d.cc[k][0], d.cc[k][1], d.cc[k][2]};
    if (coord >= 0 && m > T(0)) {
      const int j = VOFF + coord;
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
      d.m[j * NV + j] = dot3(a, tot);
      // K17 — the same coord-bearing ancestors, in the same order, read
      // out of ANC.f / ANC.c instead of chased through F_PARENT. Entry 0
      // is k itself (F_COORD[k] >= 0 holds here), so the strict
      // ancestors are the a = 1.. suffix; the per-step
      // `if (F_COORD[anc] >= 0)` filter is baked into the row.
      const int na = ANC_N[k];
      for (int a = 1; a < na; a++) {
        const int anc = ANC.f[k][a];
        const int ia = ANC.c[k][a];
        T rel2[3];
        sub3v(cvec, d.pos[anc], rel2);
        T r2xhl[3];
        cross3(rel2, hl, r2xhl);
        T tot2[3] = {ha[0] + r2xhl[0], ha[1] + r2xhl[1], ha[2] + r2xhl[2]};
        const T val = dot3(d.axw[anc], tot2);
        // symmetric pair -> lower entry only (F_COORD is not monotonic
        // in the frame index, so the order of ia vs j is a runtime fact)
        d.m[(ia > j ? j : ia) * NV + (ia > j ? ia : j)] = val;
      }
      // root rows
      T relr[3];
      sub3v(cvec, d.p, relr);
      T rrxhl[3];
      cross3(relr, hl, rrxhl);
      T lr[3] = {ha[0] + rrxhl[0], ha[1] + rrxhl[1], ha[2] + rrxhl[2]};
      // j >= VOFF > 3 + t, so only the [j][*] halves are in the lower
      // triangle; the [t][j] / [3+t][j] mirrors were dead.
      for (int t = 0; t < 3; t++) {
        d.m[t * NV + j] = hl[t];
        d.m[(3 + t) * NV + j] = lr[t];
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
    if (F_COORD[k] >= 0) {
      const int j = VOFF + F_COORD[k];
      d.m[j * NV + j] += T(ARMATURE[F_COORD[k]]);
    }

  // root 6x6 from the total composite
  T dd[3];
  sub3v(d.bc, d.p, dd);
  T rr[3][3];
  for (int i = 0; i < 3; i++)
    for (int j = 0; j < 3; j++) rr[i][j] = d.bi[i][j];
  add_shift(rr, d.bm, dd);
  for (int i = 0; i < 3; i++)
    for (int j = 0; j <= i; j++) {
      d.m[j * NV + i] = i == j ? d.bm : T(0);
      d.m[(3 + j) * NV + (3 + i)] = rr[i][j];
    }
  const T sk[3][3] = {{T(0), -dd[2], dd[1]},
                      {dd[2], T(0), -dd[0]},
                      {-dd[1], dd[0], T(0)}};
  // (3+i, j) is strictly lower for every i, j in 0..2; the (j, 3+i)
  // mirror was dead.
  for (int i = 0; i < 3; i++)
    for (int j = 0; j < 3; j++) d.m[j * NV + (3 + i)] = d.bm * sk[i][j];
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
    if (F_MASS[k] > 0.0) {
      T e[3];
      for (int i = 0; i < 3; i++)
        e[i] = d.r[k][i][0] * T(F_COM[k][0]) + d.r[k][i][1] * T(F_COM[k][1])
               + d.r[k][i][2] * T(F_COM[k][2]);
      T alxe[3], omxe[3], omxomxe[3];
      cross3(alk, e, alxe);
      cross3(omk, e, omxe);
      cross3(omk, omxe, omxomxe);
      T ac[3];
      for (int i = 0; i < 3; i++) ac[i] = (aok[i] + alxe[i]) + omxomxe[i];
      const T(&iw)[3][3] = d.iw[k];  // K4e
      T fbk[3];
      for (int i = 0; i < 3; i++) fbk[i] = (ac[i] - g[i]) * T(F_MASS[k]);
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
      e[i] = d.r0[i][0] * T(BASE_COM[0]) + d.r0[i][1] * T(BASE_COM[1])
             + d.r0[i][2] * T(BASE_COM[2]);
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

// Dense Cholesky (lower, no pivoting; algo.rs chol_factor_floored)
// with the same pivot-floor guards: f64
// floors pivots at PD_FLOOR_F64 = 1e-17 (Divergences §22 — same
// constant and max clamp as Rust; Rust counts into ws.pd_hits
// instead of panicking now), f32 floors at 1e-8 and latches bad only
// for d <= 0 (Divergences §20e).
//
// COLUMN-MAJOR STORAGE (K12, 2026-08-07). Element (row R, col C) of both
// M and L lives at a[C*n + R] — the transpose of the address map every
// earlier note here used. NOTHING else changed: same reads, same writes,
// same order, same expression trees, at the same inline sites. It is a
// pure bijective remap of addresses, so it is BITWISE by construction
// (both dtypes gated at max diff 0.0); the producers (crba, the A_ww
// assembly) just swap their two index terms, which is value-preserving
// because M and A_ww are symmetric and only one triangle is ever stored.
// WHY: every hot loop in this file walks a COLUMN of the factor while
// varying the row. Row-major made that R separate runtime addresses at
// stride n (5 live in chol_factor_n's inner loop, 8 in chol_fwd_unit4_n);
// column-major collapses them to ONE base (k*n) with immediate offsets
// +0..+3 — exactly the live-address transform that won K11 its +7.06%,
// and the back substitutions' l[k*n+i] sweep becomes contiguous too.
// This is the one direction that strictly REMOVES live address registers,
// which matters at the 255-register pin (see the SPILL-CLIFF notes).
// The triangle actually occupied is therefore a[C*n+R] with R >= C, i.e.
// the UPPER triangle of the buffer read as a row-major array — the
// K4c "lower triangle + diagonal only" invariant is unchanged in matrix
// terms, only its address form moved.
//
// IN PLACE, one buffer (2026-08-03; supersedes the 2026-08-02 note that
// deliberately withheld __restrict__ here pending exactly this change).
// `a` enters as M and leaves as its lower Cholesky factor L. Legal by
// inspection of the recurrence, and bitwise identical to the old
// two-buffer form because it issues the SAME reads, the SAME writes, in
// the SAME order — only the destination address changes:
//   * column j reads a[j*n+j] (the M diagonal) as its FIRST statement
//     and writes a[j*n+j] = dj only after the k-loop — read-before-
//     write, and no later column ever re-reads a diagonal;
//   * the row loop reads a[j*n+i] (element (i, j)) as the first statement
//     of iteration i and writes exactly that one element as its last —
//     read once, immediately before its own overwrite;
//   * every other read is a[k*n+i] / a[k*n+j] with k < j, i.e. columns
//     already converted to L by earlier j, and no write at column j
//     touches a column < j. Under column-major those columns are the
//     DISJOINT address blocks [k*n, (k+1)*n), which makes the in-place
//     aliasing argument a statement about whole blocks rather than
//     strides: column j writes only inside block j.
// Callers: d.m (written only by crba) and s.aww (rewritten before each
// factor) — neither is read as M after its factor, so nothing needs the
// original matrix. Both are supplied LOWER TRIANGLE + DIAGONAL ONLY
// since K4c; this function reads and writes nothing above the diagonal.
// Frees Dyn::l (NV*NV) and Solver::lww (N_WELD*N_WELD) from the frame.
//
// The dead upper-triangle zeroing that used to run here (element (j, i),
// i > j — 595 stores per NV=35 factor + 66 per weld factor, per
// substep) is GONE: the only readers of a factor are this function
// (elements (i, k), (j, k) with k < j <= i) and chol_solve_n ((i, k)
// k<i; (k, i) k>i; (i, i)) — strictly lower triangle plus diagonal.
// Nothing reads above it, so it needs no value.
// `a` and `bad` are distinct objects (the matrix buffer vs d.pd_bad),
// so __restrict__ on both is true and lets the L row stay in registers
// across the (rare) pivot-floor latch.
//
// PIVOT CHAIN SOFTWARE-PIPELINED (2026-08-03, K4b/a). The column-j pivot
// used to be its own serial reduction over k < j, run back to back with
// the row loop that immediately follows it. Column j+1's pivot,
//   dpiv_{j+1} = M[j+1][j+1] - sum_{k<=j} L[j+1][k]^2,
// depends only on values that are FINAL by the time column j's row loop
// touches row j+1: L[j+1][k] for k < j was written by earlier columns,
// and L[j+1][j] is the very first value that loop produces. So it is
// accumulated INSIDE column j's first row block — the same terms, the
// same ascending k, one extra scalar carried in a register, zero extra
// storage and zero extra loads (element (j+1, k) is already being loaded
// there as s0's operand) — and column j+1 then starts with its pivot in
// hand. Bitwise: identical expression, identical order, identical
// inputs; a[(j+1)*n+(j+1)] is the untouched M diagonal (column j writes
// only inside block j, and the diagonal of column j+1 lives in block
// j+1).
//
// K21 (2026-08-15) — RIGHT-LOOKING ALTERNATIVE, COMPILE-TIME OPTION,
// DEFAULT OFF (`#if BEAT_CHOL_RL` below; jit_cache BEAT_CHOL_RL=1 appends
// -DBEAT_CHOL_RL=1 and is part of the flag digest). The one micro
// back-port that survived the four-way warp rescue analysis
// (WARP_ENV_SOLVER_CORE.md §4.2, WARP_RESCUE_LEGACY_SIDE.md §4.1) after
// the warp/sub-warp kernel family itself was refuted; it is
// warp-independent, so it lands on this LEGACY serial kernel as written.
//
// MEASURED VERDICT (RTX 4090, step ms/tick), which is why it is an option
// and not the body:
//   * ALONE, at production N = 16,384: 124.8 vs 121.8 = +2.5% SLOWER.
//     The ~6,545 read-modify-writes into the 42 KB local frame cost more
//     than the chain cut wins at that occupancy. Bitwise-correct, just
//     slower — so shipping it by default would be a pure regression.
//   * ITS REAL VALUE IS AN ENABLER. It cuts register demand 254 -> 168,
//     which is what lets BEAT_MIN_BLOCKS=16 build at 128 registers with
//     ZERO spills. At N = 65,536: 350.8 (RL + K=16) vs 386.1 (RL alone,
//     K=1) vs 365.6 (default left-looking baseline) — -4.1% against the
//     baseline and -9.2% against its own K=1 arm. The first real
//     occupancy win in this kernel's history.
//
// FLAG CONTRACT: BEAT_CHOL_RL=1 is expected to pay ONLY together with
// BEAT_MIN_BLOCKS=16 AND N >= 65,536. On its own, at any N, or at K = 1,
// it is a measured loss; below N = 65,536 the grid cannot supply the
// blocks the K = 16 pin asks for, so the rung it unlocks is unreachable
// and only the +2.5% RMW cost remains. Set both or set neither.
// (K = 12 is a no-op on this arm: demand 168 < the 170 cap it imposes.)
//
// WHAT MOVES. Left-looking (the default body) holds each output element's
// accumulator in a register and sweeps k = 0..j-1 to gather its terms
// just before writing it. Right-looking turns that inside out: as soon as
// column j's pivot resolves and its column is scaled, the rank-1 update
//   A[i][c] -= L[i][j] * L[c][j],   c = j+1..n-1,  i = c..n-1
// is pushed into the whole trailing submatrix immediately. Every trailing
// element therefore receives its subtractions ONE j AT A TIME, j
// ASCENDING — the same terms, in the same order, from the same seed
// M[i][c] — instead of all at once at the end. Only the interleaving in
// time changed. The pivot also stops needing the K4b/a pipelining above:
// column j's diagonal is finished, in place, by the rank-1 update of
// column j-1, and column j simply LOADS it.
//
// WHY (the chain arithmetic, n = NV = 35). Op counts are unchanged:
// 6,545 FMA (sum_j j*(34-j)), 595 divides (sum_j (34-j)), 35 sqrt. What
// changes is the DEPENDENT chain:
//   * left-looking: accumulator s_t is a serial chain of length j, so
//     the critical path is sum_j j = 595 dependent FMAs, plus 35
//     (sqrt + divide). The K6 jam bought ILP, NOT chain length.
//   * right-looking: within a column the rank-1 FMAs target DIFFERENT
//     elements and are mutually independent, so a column contributes
//     max/sqrt -> divide -> ONE FMA before the next column's diagonal is
//     final. 35 columns * ~4 dependent ops = ~140 dependent steps.
// 595 -> ~140 is 4.3x on the FMA chain (5.7x counting the sqrt/divide
// tail the old form also pays). That predicted a win by the kernel's
// best-attested class (`:129-131`) and did not get one at 16k — the
// counter-traffic it trades for (`:2309-2315` below) is what ate it.
//
// THE COST. Serially, right-looking writes the whole trailing submatrix
// per column: ~6,545 read-modify-writes into the 42 KB local frame, where
// left-looking accumulates in a register and stores once. It TRADES chain
// length for local traffic, and at N = 16,384 the trade loses by 2.5%.
//
// K6's UNROLL-AND-JAM IS DELIBERATELY ABSENT from this arm, and so is the
// peeled pivot row. Both exist in the default body for one reason: to
// manufacture ILP for the left-looking form, whose single accumulator
// chain has none. The right-looking form has natural ILP by construction
// — the rank-1 update's subtractions are independent ACROSS ELEMENTS, so
// the inner loop is already n-c independent FMAs and the scale loop
// n-1-j independent divides. Jamming it would add live registers and
// address pressure to buy ILP that is already there, and register demand
// is the entire point of this arm. Plain nests; no #pragma unroll,
// because every trip count here is a function of the runtime `n` (35 for
// d.m, 12 for s.aww) and there is no compile-time count to unroll.
//
// PRESERVED EXACTLY across both arms: the `chol_factor_floored` pivot-
// FLOOR semantics and both thresholds (f64 latches `bad` on
// dpiv < PD_FLOOR_F64 and clamps there; f32 latches only on dpiv <= 0 and
// clamps at PD_FLOOR_F32; the tmax form still maps a NaN pivot onto the
// floor, which chol_fwd_unit*_n's headers depend on); the `bad` monotone
// latch; the column-major a[C*n+R] contract; fp32/fp64 genericity; and
// the per-element arithmetic ORDER. The lower triangle + diagonal the
// right-looking arm leaves is byte-for-byte what the left-looking arm
// leaves, so every consumer — chol_solve_n / _multi_n / _multi_tn /
// _fwd_unit*_n / _back*_n, which read (i,k) k<i, (k,i) k>i and the
// diagonal — is untouched by construction, and the two arms are
// INTERCHANGEABLE at every call site.
//
// BITWISE ARGUMENT, element by element:
//   * off-diagonal (i, c): left-looking computes
//       s = M[i][c]; for k = 0..c-1: s -= L[i][k]*L[c][k]; L[i][c] = s/dc.
//     Right-looking seeds the SAME M[i][c] in place and applies the SAME
//     terms for k = 0..c-1 in the SAME ascending order (k is now the
//     outer column loop), then divides the same value by the same dc.
//   * diagonal (c, c): left-looking accumulates `np` from M[c][c] over
//     `np -= v0*v0` with v0 = L[c][k], k ascending. Right-looking's
//     update at i == c is `a[cc+c] -= a[cj+c] * ljc` where both operands
//     ARE that same L[c][j] — the identical `v*v` term, same ascending
//     order. The accumulator now lives in memory instead of a register,
//     which is value-preserving: f32/f64 stores round to the declared
//     type and there is no extended-precision register on this target.
//   * OPERAND ORDER inside each multiply is kept as (row element) *
//     (pivot-row element), matching `:2292-2295`, and each subtraction
//     is still ONE STATEMENT of the shape `X -= Y*Z`, so it contracts to
//     the same single FFMA under --fmad=true. No mul/sub was split, no
//     fma chain was restructured — this is the K10e2 hazard's only
//     escape and it is closed by keeping the expression shape identical.
//
// HARNESS (g++ shim, the stage-2 technique; source kept as
// b1_harness.cpp): the two arms compiled side by side under g++ -O2 and
// run on the SAME buffers, comparing the FULL n*n buffer by memcmp plus
// the `bad` flag. 1,754 cases / 1,409,410 words per arm, 688 of them
// tripping the pivot floor: 200 well-conditioned random SPD (n=35), 150
// rank-deficient/near-singular, 100 ill-scaled (1e-8..1e8 row scaling),
// 100 adversarial non-PD (negative diagonals, +0/-0 diagonals,
// denormals, zeroed sub-blocks), the degenerate fixed set (all-zero,
// all-negative-zero, identity, negative identity, alternating-zero
// diagonal, rank-1 all-ones, NaN/+Inf/-Inf entries), 120 at N_WELD = 12,
// and 200 at n = 1..8 to cover every remainder tier of the default form's
// 4-way peel. Buffer slots ABOVE the diagonal are poisoned with sNaN
// patterns so a stray write is caught too.
//   -ffp-contract=off                   -> 0 mismatching words, 0 flags
//   forced fma() at every SUBMUL site   -> 0 mismatching words, 0 flags
// Those two arms BRACKET nvcc: no contraction anywhere, and full
// statement-level contraction everywhere. GCC 13's -ffp-contract=on is
// a no-op alias for `off` (verified: zero vfmadd in the binary), so the
// explicit-fma arm is what stands in for --fmad=true. Mutation-tested
// live: replacing the divide with a reciprocal multiply in the new form
// produced 400,335 mismatching words, and weakening the f32 `bad`
// predicate produced 9 flag mismatches.
//   -ffp-contract=fast FAILS (336,619 words, 1 ulp). That is the
// PRE-EXISTING hazard recorded in WARP_RESCUE_LEGACY_SIDE.md §4.2, not
// a property of this transform: under `fast` GCC contracts each site by
// its own scheduling whim and is not even self-consistent WITHIN the
// default function — it emits vfnmadd231ss for the peeled scalar row and
// separate vmulss + vsubss for the 4-wide jammed block. The kernel's
// no-fast-math discipline is what protects the gate.
template <typename T>
__device__ void chol_factor_n(int n, T* a, bool* bad) {
#if BEAT_CHOL_RL
  for (int j = 0; j < n; j++) {
    const int cj = j * n;  // COLUMN-MAJOR (K12): column j is contiguous
    // The diagonal was finished in place by column j-1's rank-1 update
    // (or is the untouched M[0][0] at j = 0), so the pivot is a LOAD.
    T dpiv = a[cj + j];
    if (sizeof(T) == 8) {
      if (dpiv < T(PD_FLOOR_F64)) *bad = true;
      dpiv = tmax(dpiv, T(PD_FLOOR_F64));
    } else {
      if (dpiv <= T(0)) *bad = true;
      dpiv = tmax(dpiv, T(PD_FLOOR_F32));
    }
    const T dj = sqrt(dpiv);
    a[cj + j] = dj;  // read-before-write, and no later column re-reads it
    // SCALE: L[i][j] = A[i][j] / dj, i > j. The n-1-j divides are
    // mutually independent — one divide of chain, not n-1-j.
    for (int i = j + 1; i < n; i++) {
      const T s = a[cj + i];
      a[cj + i] = s / dj;
    }
    // RANK-1 UPDATE of the trailing submatrix, columns c > j, rows
    // i >= c (diagonal included). Reads live in column j, writes in
    // column c > j: DISJOINT address blocks under column-major, so the
    // in-place aliasing argument is a statement about whole blocks.
    // Every (i, c) here is independent of every other, which is the
    // whole ILP story — no jam required.
    for (int c = j + 1; c < n; c++) {
      const int cc = c * n;
      const T ljc = a[cj + c];  // L[c][j], already scaled above
      for (int i = c; i < n; i++) a[cc + i] -= a[cj + i] * ljc;
    }
  }
#else
  T dpiv = a[0];  // j = 0: the pivot is M[0][0], no terms
  for (int j = 0; j < n; j++) {
    if (sizeof(T) == 8) {
      if (dpiv < T(PD_FLOOR_F64)) *bad = true;
      dpiv = tmax(dpiv, T(PD_FLOOR_F64));
    } else {
      if (dpiv <= T(0)) *bad = true;
      dpiv = tmax(dpiv, T(PD_FLOOR_F32));
    }
    const T dj = sqrt(dpiv);
    a[j * n + j] = dj;
    // Row loop, UNROLLED AND JAMMED BY 4: the pivot row a[j*n+k] (k<j)
    // used to be re-read once per i; jammed it is read once per FOUR i,
    // and the four s_t carry independent dependency chains instead of
    // one. Bitwise: each s_t accumulates over k in the same ascending
    // order, over the same values, as its scalar counterpart — no
    // reassociation, only interleaving. Deferring the four stores to
    // the end of the block is safe under the in-place aliasing too: the
    // inner loop only ever reads columns k < j, and the four deferred
    // stores are all in column j.
    const int cj = j * n;  // COLUMN-MAJOR (K12): column j is contiguous
    int i = j + 1;
    // Row j+1 is peeled (as the head of a 4-block when there is room,
    // scalar otherwise) so the next pivot rides along on its loads.
    T np = i < n ? a[i * n + i] : T(0);
    if (i + 3 < n) {
      T s0 = a[cj + i], s1 = a[cj + i + 1];
      T s2 = a[cj + i + 2], s3 = a[cj + i + 3];
      for (int k = 0; k < j; k++) {
        const int ck = k * n;
        const T ljk = a[ck + j];
        const T v0 = a[ck + i];
        s0 -= v0 * ljk;
        s1 -= a[ck + i + 1] * ljk;
        s2 -= a[ck + i + 2] * ljk;
        s3 -= a[ck + i + 3] * ljk;
        np -= v0 * v0;
      }
      const T l0 = s0 / dj;
      a[cj + i] = l0;
      a[cj + i + 1] = s1 / dj;
      a[cj + i + 2] = s2 / dj;
      a[cj + i + 3] = s3 / dj;
      np -= l0 * l0;
      i += 4;
    } else if (i < n) {
      T s = a[cj + i];
      for (int k = 0; k < j; k++) {
        const int ck = k * n;
        const T v0 = a[ck + i];
        s -= v0 * a[ck + j];
        np -= v0 * v0;
      }
      const T l0 = s / dj;
      a[cj + i] = l0;
      np -= l0 * l0;
      i += 1;
    }
    for (; i + 3 < n; i += 4) {
      T s0 = a[cj + i], s1 = a[cj + i + 1];
      T s2 = a[cj + i + 2], s3 = a[cj + i + 3];
      for (int k = 0; k < j; k++) {
        const int ck = k * n;
        const T ljk = a[ck + j];
        s0 -= a[ck + i] * ljk;
        s1 -= a[ck + i + 1] * ljk;
        s2 -= a[ck + i + 2] * ljk;
        s3 -= a[ck + i + 3] * ljk;
      }
      a[cj + i] = s0 / dj;
      a[cj + i + 1] = s1 / dj;
      a[cj + i + 2] = s2 / dj;
      a[cj + i + 3] = s3 / dj;
    }
    for (; i < n; i++) {
      T s = a[cj + i];
      for (int k = 0; k < j; k++) s -= a[k * n + i] * a[k * n + j];
      a[cj + i] = s / dj;
    }
    dpiv = np;
  }
#endif
}

// Triangular solve. l and x are always distinct objects at every call
// site (d.m, holding the in-place factor, vs {d.rhs, s.mw,
// s.c_minv[ci], s.ecol[j]}; s.aww vs s.rw), and x is the only thing
// written — so restrict here is the
// difference between keeping l's row in registers across the x[i]
// store and re-loading it every iteration.
template <typename T>
//
// The FORWARD half is ROW-BLOCKED BY 4 (K6). Unblocked it is ONE
// dependent chain for the whole triangle: row i's dot is a chain of
// length i and row i+1 cannot start until x[i] is stored. Blocked, the
// four rows of a block share the k < i0 sweep as FOUR INDEPENDENT
// accumulators (and share the x[k] load: 4 + 1 loads per k for 4 FMAs
// instead of 8), and only the short triangular tail k in [i0, i) — at
// most three terms, taken from the values just stored — stays serial.
// BITWISE (forward only): k = 0..i0-1 ascending is the exact PREFIX of
// every blocked row's required order, and the tail k = i0..i-1 follows
// it ascending, so each accumulator meets the same terms in the same
// sequence from the same seed x[i].
__device__ void chol_solve_n(int n, const T* __restrict__ l,
                             T* __restrict__ x) {
  int i = 0;
  for (; i + 3 < n; i += 4) {
    T s0 = x[i], s1 = x[i + 1], s2 = x[i + 2], s3 = x[i + 3];
    for (int k = 0; k < i; k++) {
      const int ck = k * n;
      const T xk = x[k];
      s0 -= l[ck + i] * xk;
      s1 -= l[ck + i + 1] * xk;
      s2 -= l[ck + i + 2] * xk;
      s3 -= l[ck + i + 3] * xk;
    }
    const int c0 = i * n, c1 = c0 + n, c2 = c1 + n, c3 = c2 + n;
    const T v0 = s0 / l[c0 + i];
    x[i] = v0;
    s1 -= l[c0 + i + 1] * v0;
    const T v1 = s1 / l[c1 + i + 1];
    x[i + 1] = v1;
    s2 -= l[c0 + i + 2] * v0;
    s2 -= l[c1 + i + 2] * v1;
    const T v2 = s2 / l[c2 + i + 2];
    x[i + 2] = v2;
    s3 -= l[c0 + i + 3] * v0;
    s3 -= l[c1 + i + 3] * v1;
    s3 -= l[c2 + i + 3] * v2;
    x[i + 3] = s3 / l[c3 + i + 3];
  }
  for (; i < n; i++) {
    T s = x[i];
    for (int k = 0; k < i; k++) s -= l[k * n + i] * x[k];
    x[i] = s / l[i * n + i];
  }
  // BACK half: NOT row-blockable (K6 prohibition, file header).
  for (int i = n - 1; i >= 0; i--) {
    const int ci = i * n;
    T s = x[i];
    for (int k = i + 1; k < n; k++) s -= l[ci + k] * x[k];
    x[i] = s / l[ci + i];
  }
}

// BATCHED multi-RHS solve: W independent right-hand sides, laid out as
// W consecutive rows of stride `n` (s.c_minv and the ensure_pcol
// batches are exactly that; s.mw is RHS-minor and uses the _tn twin),
// sharing ONE factor. This is a pure LOOP INTERCHANGE of W calls to
// chol_solve_n: the (i, k) iteration space is walked in the same
// ascending order, and each RHS w keeps its OWN accumulator visiting the
// same terms in the same sequence from the same seed x[w*n+i], so every
// value is BITWISE what the W separate calls produced. Nothing is
// reassociated; only the w loop is hoisted inside.
//
// Why it pays on this kernel (serial-latency bound, measured): the
// scalar solve's inner k-loop is ONE dependent FMA chain of length i,
// and the W calls run those chains back to back. Interleaved, the W
// chains are mutually independent, so the chain latency is amortized
// W-fold; the factor element (i, k) is also loaded once per (i, k)
// instead of once per (i, k, w). The accumulators are a fixed-size local
// array indexed only by fully unrolled compile-time w, so they live in
// registers (W f32 = W*4 B).
//
// The FORWARD half is additionally ROW-BLOCKED BY 3 (K6): rows i0..i0+2
// share the k < i0 sweep, so the W values x[w*n+k] are loaded ONCE for
// 3*W FMAs instead of once per (row, w) — per k the block costs 3 + W
// loads for 3*W FMAs where the unblocked triple cost 3*(1 + W), i.e.
// 15/36 vs 39/36 at W = 12. The terms k in [i0, i) that a later row of
// the block needs and the earlier ones do not are applied after those
// rows' divides, from the registers just stored.
// R = 4 was measured too and was NOISE over R = 3 (+0.03%, overlapping
// ranges, f64 spills 12,308/10,300 -> 12,380/10,564) — REVERTED, the
// x-load saving is already down to a sixth by R = 3.
// BITWISE (forward only, see the back half's note): row i0's terms are
// k = 0..i0-1 ascending, unchanged; a later row's are k = 0..i0-1
// ascending (the SHARED PREFIX of its required order) followed by the
// tail k = i0.. in ascending order, i.e. exactly its scalar sequence.
// Every accumulator starts from the same seed x[w*n+i] and meets the
// same terms in the same order; the x[i0] values it multiplies are the
// same values the stores just wrote.
template <int W, typename T>
__device__ void chol_solve_multi_n(int n, const T* __restrict__ l,
                                   T* __restrict__ x) {
  int i = 0;
  for (; i + 2 < n; i += 3) {
    T s0[W], s1[W], s2[W];
#pragma unroll
    for (int w = 0; w < W; w++) {
      s0[w] = x[w * n + i];
      s1[w] = x[w * n + i + 1];
      s2[w] = x[w * n + i + 2];
    }
    for (int k = 0; k < i; k++) {
      const int ck = k * n;
      const T l0 = l[ck + i], l1 = l[ck + i + 1], l2 = l[ck + i + 2];
#pragma unroll
      for (int w = 0; w < W; w++) {
        const T xk = x[w * n + k];
        s0[w] -= l0 * xk;
        s1[w] -= l1 * xk;
        s2[w] -= l2 * xk;
      }
    }
    const int c0 = i * n, c1 = c0 + n, c2 = c1 + n;
    const T d0 = l[c0 + i], l1i = l[c0 + i + 1], d1 = l[c1 + i + 1];
    const T l2i = l[c0 + i + 2], l2j = l[c1 + i + 2], d2 = l[c2 + i + 2];
#pragma unroll
    for (int w = 0; w < W; w++) {
      const T v0 = s0[w] / d0;
      x[w * n + i] = v0;
      s1[w] -= l1i * v0;
      const T v1 = s1[w] / d1;
      x[w * n + i + 1] = v1;
      s2[w] -= l2i * v0;
      s2[w] -= l2j * v1;
      x[w * n + i + 2] = s2[w] / d2;
    }
  }
  for (; i < n; i++) {  // up to two rows left over (NV = 35)
    T s[W];
#pragma unroll
    for (int w = 0; w < W; w++) s[w] = x[w * n + i];
    for (int k = 0; k < i; k++) {
      const T lik = l[k * n + i];
#pragma unroll
      for (int w = 0; w < W; w++) s[w] -= lik * x[w * n + k];
    }
    const T dg = l[i * n + i];
#pragma unroll
    for (int w = 0; w < W; w++) x[w * n + i] = s[w] / dg;
  }
  // BACK half: row-blocking here is FORBIDDEN, see the K6 prohibition in
  // the file header (the shared high-k region is each row's TAIL, not a
  // prefix, and the near-diagonal terms are not available yet). The W
  // right-hand sides stay batched — that is a different transform.
  for (int i = n - 1; i >= 0; i--) {
    const int ci = i * n;
    T s[W];
#pragma unroll
    for (int w = 0; w < W; w++) s[w] = x[w * n + i];
    for (int k = i + 1; k < n; k++) {
      const T lki = l[ci + k];
#pragma unroll
      for (int w = 0; w < W; w++) s[w] -= lki * x[w * n + k];
    }
    const T dg = l[ci + i];
#pragma unroll
    for (int w = 0; w < W; w++) x[w * n + i] = s[w] / dg;
  }
}

// K11 — the SAME batched solve on a RHS-MINOR right-hand side: x[k*W+w]
// instead of x[w*n+k]. Only s.mw is stored this way (see Solver::mw);
// s.c_minv and the ensure_pcol batches keep the row-major entry point
// above, so both survive.
//
// TRIVIALLY BITWISE: this is chol_solve_multi_n with one address map
// applied to x and nothing else. Same i order, same k order, same w
// order, same seeds, same divisors, same accumulator chains — every
// floating-point operation is the identical operation on the identical
// operands, only fetched from a different offset.
//
// Why it is worth a second entry point: the W-loop is the innermost
// loop of all three halves, so RHS-minor makes its W accesses W
// CONSECUTIVE per-thread words reachable as immediate offsets from ONE
// base (`bk = k * W`, W a compile-time constant), where the row-major
// form needed W separate addresses at runtime stride n. The load COUNT
// is unchanged — ptxas emits 12 scalar LDL either way, see Solver::mw
// for why the LDL.128 hypothesis was falsified — but the back half
// alone runs n(n-1)/2 = 595 (i, k) pairs at W = 12, and cutting each
// one's address arithmetic to a single IADD is most of K11's +7.06%.
template <int W, typename T>
__device__ void chol_solve_multi_tn(int n, const T* __restrict__ l,
                                    T* __restrict__ x) {
  int i = 0;
  for (; i + 2 < n; i += 3) {  // ROW-BLOCKED BY 3, as above (K6)
    const int b0 = i * W, b1 = b0 + W, b2 = b1 + W;
    T s0[W], s1[W], s2[W];
#pragma unroll
    for (int w = 0; w < W; w++) {
      s0[w] = x[b0 + w];
      s1[w] = x[b1 + w];
      s2[w] = x[b2 + w];
    }
    for (int k = 0; k < i; k++) {
      const int ck = k * n;
      const T l0 = l[ck + i], l1 = l[ck + i + 1], l2 = l[ck + i + 2];
      const int bk = k * W;
#pragma unroll
      for (int w = 0; w < W; w++) {
        const T xk = x[bk + w];
        s0[w] -= l0 * xk;
        s1[w] -= l1 * xk;
        s2[w] -= l2 * xk;
      }
    }
    const int c0 = i * n, c1 = c0 + n, c2 = c1 + n;
    const T d0 = l[c0 + i], l1i = l[c0 + i + 1], d1 = l[c1 + i + 1];
    const T l2i = l[c0 + i + 2], l2j = l[c1 + i + 2], d2 = l[c2 + i + 2];
#pragma unroll
    for (int w = 0; w < W; w++) {
      const T v0 = s0[w] / d0;
      x[b0 + w] = v0;
      s1[w] -= l1i * v0;
      const T v1 = s1[w] / d1;
      x[b1 + w] = v1;
      s2[w] -= l2i * v0;
      s2[w] -= l2j * v1;
      x[b2 + w] = s2[w] / d2;
    }
  }
  for (; i < n; i++) {  // up to two rows left over (NV = 35)
    const int bi = i * W;
    T s[W];
#pragma unroll
    for (int w = 0; w < W; w++) s[w] = x[bi + w];
    for (int k = 0; k < i; k++) {
      const T lik = l[k * n + i];
      const int bk = k * W;
#pragma unroll
      for (int w = 0; w < W; w++) s[w] -= lik * x[bk + w];
    }
    const T dg = l[i * n + i];
#pragma unroll
    for (int w = 0; w < W; w++) x[bi + w] = s[w] / dg;
  }
  // BACK half: not row-blocked (K6 prohibition, file header).
  for (int i = n - 1; i >= 0; i--) {
    const int bi = i * W;
    const int ci = i * n;
    T s[W];
#pragma unroll
    for (int w = 0; w < W; w++) s[w] = x[bi + w];
    for (int k = i + 1; k < n; k++) {
      const T lki = l[ci + k];
      const int bk = k * W;
#pragma unroll
      for (int w = 0; w < W; w++) s[w] -= lki * x[bk + w];
    }
    const T dg = l[ci + i];
#pragma unroll
    for (int w = 0; w < W; w++) x[bi + w] = s[w] / dg;
  }
}

// Same solve, specialized for a UNIT right-hand side e_m (x enters as
// all +0.0 except x[m] = 1). BITWISE identical to chol_solve_n on that
// input, not an approximation — the forward substitution provably
// reproduces the incoming +0.0 for every row i < m, and every k < m
// term of every later row is an exact no-op:
//   * the factor diagonal is always strictly positive and finite
//     (chol_factor_n floors the pivot at PD_FLOOR_* through tmax, whose
//     `a > b ? a : b` form also maps a NaN pivot onto the floor), so
//     +0.0 / L[i][i] = +0.0 — the value already in the buffer;
//   * by induction on i < m: s starts at x[i] = +0.0 and every term
//     L[i][k] * x[k] has x[k] = +0.0 (k < i < m, established by the
//     induction), so the product is +-0.0 and s stays exactly +0.0
//     (+0.0 - +0.0 = +0.0 and +0.0 - -0.0 = +0.0 under RN; FMA
//     contraction does not change either);
//   * at i == m the same k < m terms leave s = x[m] = 1 untouched, so
//     the k-loop may start at k = m (zero iterations here) and the
//     divide is the identical 1 / L[m][m];
//   * at i > m s starts at +0.0, the k < m terms leave it at +0.0, and
//     the surviving k >= m terms then run in the SAME ascending order
//     over the SAME values from the SAME accumulator start.
// The BACK substitution stays dense: it reads x[k] for k > i, which for
// k < m are the skipped rows — they must still hold the +0.0 the
// forward pass would have written, so the caller's full-length zero
// fill CANNOT shrink. (Rows > m need it too: it is their `s` seed.)
template <typename T>
__device__ void chol_fwd_unit_n(int n, const T* __restrict__ l,
                                T* __restrict__ x, int m) {
  // ROW-BLOCKED BY 4 (K6), same forward-only proof as chol_solve_n's.
  int i = m;
  for (; i + 3 < n; i += 4) {
    T s0 = x[i], s1 = x[i + 1], s2 = x[i + 2], s3 = x[i + 3];
    for (int k = m; k < i; k++) {
      const int ck = k * n;
      const T xk = x[k];
      s0 -= l[ck + i] * xk;
      s1 -= l[ck + i + 1] * xk;
      s2 -= l[ck + i + 2] * xk;
      s3 -= l[ck + i + 3] * xk;
    }
    const int c0 = i * n, c1 = c0 + n, c2 = c1 + n, c3 = c2 + n;
    const T v0 = s0 / l[c0 + i];
    x[i] = v0;
    s1 -= l[c0 + i + 1] * v0;
    const T v1 = s1 / l[c1 + i + 1];
    x[i + 1] = v1;
    s2 -= l[c0 + i + 2] * v0;
    s2 -= l[c1 + i + 2] * v1;
    const T v2 = s2 / l[c2 + i + 2];
    x[i + 2] = v2;
    s3 -= l[c0 + i + 3] * v0;
    s3 -= l[c1 + i + 3] * v1;
    s3 -= l[c2 + i + 3] * v2;
    x[i + 3] = s3 / l[c3 + i + 3];
  }
  for (; i < n; i++) {
    T s = x[i];
    for (int k = m; k < i; k++) s -= l[k * n + i] * x[k];
    x[i] = s / l[i * n + i];
  }
}

// BATCHED unit forward substitution, W columns sharing one factor, run
// DENSE from the group's smallest m0 = min(m_t) (K4a). Every column
// keeps its own accumulator over the same ascending k, so this is the
// same loop interchange as chol_solve_multi_n on the rows i >= m_t; the
// rows and terms it ADDS for a column with m_t > m0 are exact no-ops by
// chol_fwd_unit_n's own proof, quoted here for the batched shape:
//   * rows m0 <= i < m_t: x_t[i] enters as the seeded +0.0 and every
//     term L[i][k] * x_t[k] has k < i < m_t, so x_t[k] is +0.0 by
//     induction — s stays exactly +0.0 (+0.0 - (+-0.0) = +0.0 under RN)
//     and the store is +0.0 / diag = +0.0, the value already there
//     (the diagonal is strictly positive and finite: chol_factor_n's
//     tmax floor maps even a NaN pivot onto PD_FLOOR_*);
//   * row i == m_t: s enters as 1 and the k < m_t terms subtract +-0.0
//     from it, leaving 1 — then the surviving k >= m_t terms and the
//     divide are the scalar pass's, bit for bit;
//   * rows i > m_t: s enters +0.0, the k < m_t terms leave it +0.0, and
//     the k >= m_t terms then run in the SAME ascending order over the
//     SAME values from the SAME accumulator start.
// So the batch is bitwise the W separate chol_fwd_unit_n calls, and it
// replaces W back-to-back serial chains with W independent ones.
template <typename T>
__device__ void chol_fwd_unit4_n(int n, const T* __restrict__ l,
                                 T* __restrict__ x0, T* __restrict__ x1,
                                 T* __restrict__ x2, T* __restrict__ x3,
                                 int m0) {
  // ROW-BLOCKED BY 4 as well (K6): the four x_t[k] are loaded once for
  // 16 FMAs. Same forward-only proof — the four rows share the ascending
  // PREFIX k in [m0, i0), and each row's remaining terms k in [i0, i)
  // follow it ascending, taken from the values the earlier rows of the
  // block just stored.
  int i = m0;
  for (; i + 3 < n; i += 4) {
    T a0 = x0[i], a1 = x1[i], a2 = x2[i], a3 = x3[i];
    T b0 = x0[i + 1], b1 = x1[i + 1], b2 = x2[i + 1], b3 = x3[i + 1];
    T c0 = x0[i + 2], c1 = x1[i + 2], c2 = x2[i + 2], c3 = x3[i + 2];
    T e0 = x0[i + 3], e1 = x1[i + 3], e2 = x2[i + 3], e3 = x3[i + 3];
    for (int k = m0; k < i; k++) {
      const int ck = k * n;
      const T la = l[ck + i], lb = l[ck + i + 1];
      const T lc = l[ck + i + 2], le = l[ck + i + 3];
      const T y0 = x0[k], y1 = x1[k], y2 = x2[k], y3 = x3[k];
      a0 -= la * y0;
      a1 -= la * y1;
      a2 -= la * y2;
      a3 -= la * y3;
      b0 -= lb * y0;
      b1 -= lb * y1;
      b2 -= lb * y2;
      b3 -= lb * y3;
      c0 -= lc * y0;
      c1 -= lc * y1;
      c2 -= lc * y2;
      c3 -= lc * y3;
      e0 -= le * y0;
      e1 -= le * y1;
      e2 -= le * y2;
      e3 -= le * y3;
    }
    const int c0b = i * n, c1b = c0b + n, c2b = c1b + n, c3b = c2b + n;
    const T d0 = l[c0b + i];
    const T v0 = a0 / d0, v1 = a1 / d0, v2 = a2 / d0, v3 = a3 / d0;
    x0[i] = v0;
    x1[i] = v1;
    x2[i] = v2;
    x3[i] = v3;
    const T p1 = l[c0b + i + 1];
    b0 -= p1 * v0;
    b1 -= p1 * v1;
    b2 -= p1 * v2;
    b3 -= p1 * v3;
    const T d1 = l[c1b + i + 1];
    const T w0 = b0 / d1, w1 = b1 / d1, w2 = b2 / d1, w3 = b3 / d1;
    x0[i + 1] = w0;
    x1[i + 1] = w1;
    x2[i + 1] = w2;
    x3[i + 1] = w3;
    const T p2 = l[c0b + i + 2], q2 = l[c1b + i + 2];
    c0 -= p2 * v0;
    c1 -= p2 * v1;
    c2 -= p2 * v2;
    c3 -= p2 * v3;
    c0 -= q2 * w0;
    c1 -= q2 * w1;
    c2 -= q2 * w2;
    c3 -= q2 * w3;
    const T d2 = l[c2b + i + 2];
    const T z0 = c0 / d2, z1 = c1 / d2, z2 = c2 / d2, z3 = c3 / d2;
    x0[i + 2] = z0;
    x1[i + 2] = z1;
    x2[i + 2] = z2;
    x3[i + 2] = z3;
    const T p3 = l[c0b + i + 3], q3 = l[c1b + i + 3], u3 = l[c2b + i + 3];
    e0 -= p3 * v0;
    e1 -= p3 * v1;
    e2 -= p3 * v2;
    e3 -= p3 * v3;
    e0 -= q3 * w0;
    e1 -= q3 * w1;
    e2 -= q3 * w2;
    e3 -= q3 * w3;
    e0 -= u3 * z0;
    e1 -= u3 * z1;
    e2 -= u3 * z2;
    e3 -= u3 * z3;
    const T d3 = l[c3b + i + 3];
    x0[i + 3] = e0 / d3;
    x1[i + 3] = e1 / d3;
    x2[i + 3] = e2 / d3;
    x3[i + 3] = e3 / d3;
  }
  for (; i < n; i++) {  // up to three rows left over
    T s0 = x0[i], s1 = x1[i], s2 = x2[i], s3 = x3[i];
    for (int k = m0; k < i; k++) {
      const T lik = l[k * n + i];
      s0 -= lik * x0[k];
      s1 -= lik * x1[k];
      s2 -= lik * x2[k];
      s3 -= lik * x3[k];
    }
    const T dg = l[i * n + i];
    x0[i] = s0 / dg;
    x1[i] = s1 / dg;
    x2[i] = s2 / dg;
    x3[i] = s3 / dg;
  }
}

template <typename T>
__device__ void chol_fwd_unit2_n(int n, const T* __restrict__ l,
                                 T* __restrict__ x0, T* __restrict__ x1,
                                 int m0) {
  // ROW-BLOCKED BY 4 (K6), same forward-only proof.
  int i = m0;
  for (; i + 3 < n; i += 4) {
    T a0 = x0[i], a1 = x1[i];
    T b0 = x0[i + 1], b1 = x1[i + 1];
    T c0 = x0[i + 2], c1 = x1[i + 2];
    T e0 = x0[i + 3], e1 = x1[i + 3];
    for (int k = m0; k < i; k++) {
      const int ck = k * n;
      const T la = l[ck + i], lb = l[ck + i + 1];
      const T lc = l[ck + i + 2], le = l[ck + i + 3];
      const T y0 = x0[k], y1 = x1[k];
      a0 -= la * y0;
      a1 -= la * y1;
      b0 -= lb * y0;
      b1 -= lb * y1;
      c0 -= lc * y0;
      c1 -= lc * y1;
      e0 -= le * y0;
      e1 -= le * y1;
    }
    const int c0b = i * n, c1b = c0b + n, c2b = c1b + n, c3b = c2b + n;
    const T d0 = l[c0b + i];
    const T v00 = a0 / d0, v01 = a1 / d0;
    x0[i] = v00;
    x1[i] = v01;
    const T p1 = l[c0b + i + 1];
    b0 -= p1 * v00;
    b1 -= p1 * v01;
    const T d1 = l[c1b + i + 1];
    const T v10 = b0 / d1, v11 = b1 / d1;
    x0[i + 1] = v10;
    x1[i + 1] = v11;
    const T p2 = l[c0b + i + 2], q2 = l[c1b + i + 2];
    c0 -= p2 * v00;
    c1 -= p2 * v01;
    c0 -= q2 * v10;
    c1 -= q2 * v11;
    const T d2 = l[c2b + i + 2];
    const T v20 = c0 / d2, v21 = c1 / d2;
    x0[i + 2] = v20;
    x1[i + 2] = v21;
    const T p3 = l[c0b + i + 3], q3 = l[c1b + i + 3], u3 = l[c2b + i + 3];
    e0 -= p3 * v00;
    e1 -= p3 * v01;
    e0 -= q3 * v10;
    e1 -= q3 * v11;
    e0 -= u3 * v20;
    e1 -= u3 * v21;
    const T d3 = l[c3b + i + 3];
    x0[i + 3] = e0 / d3;
    x1[i + 3] = e1 / d3;
  }
  for (; i < n; i++) {
    T s0 = x0[i], s1 = x1[i];
    for (int k = m0; k < i; k++) {
      const T lik = l[k * n + i];
      s0 -= lik * x0[k];
      s1 -= lik * x1[k];
    }
    const T dg = l[i * n + i];
    x0[i] = s0 / dg;
    x1[i] = s1 / dg;
  }
}

// The back substitutions (this one and chol_back{2,4}_n below) are the
// one shape in the kernel that must stay row-serial: see the K6
// ABSOLUTE PROHIBITION in the file header. Do not row-block them, at any
// R — batching more right-hand sides in is the legal direction.
template <typename T>
__device__ void chol_back_n(int n, const T* __restrict__ l,
                            T* __restrict__ x) {
  for (int i = n - 1; i >= 0; i--) {
    const int ci = i * n;
    T s = x[i];
    for (int k = i + 1; k < n; k++) s -= l[ci + k] * x[k];
    x[i] = s / l[ci + i];
  }
}

template <typename T>
__device__ void chol_solve_unit_n(int n, const T* __restrict__ l,
                                  T* __restrict__ x, int m) {
  chol_fwd_unit_n(n, l, x, m);
  chol_back_n(n, l, x);
}

// BACK substitution for 4 / 2 independent columns at once. The back pass
// is DENSE and m-independent — identical code for every right-hand side —
// so batching it needs no per-column bookkeeping at all: it is the same
// loop interchange as chol_solve_multi_n (each column keeps its own
// accumulator over the same k in the same descending-i order, so every
// value is bitwise the scalar pass's), just over columns that are not
// contiguous in memory (s.ecol rows for non-consecutive joints), hence
// explicit pointers rather than a stride. Accumulators are named scalars
// so they cannot land in local memory.
//
// The forward half is batched too since K4a (chol_fwd_unit{2,4}_n): it
// runs dense from the group's smallest m, which does re-add some of the
// rows A4 removed, but they are exact arithmetic no-ops and the four
// chains it buys pay for them — measured, see the session note.
template <typename T>
__device__ void chol_back4_n(int n, const T* __restrict__ l,
                             T* __restrict__ x0, T* __restrict__ x1,
                             T* __restrict__ x2, T* __restrict__ x3) {
  for (int i = n - 1; i >= 0; i--) {
    const int ci = i * n;
    T s0 = x0[i], s1 = x1[i], s2 = x2[i], s3 = x3[i];
    for (int k = i + 1; k < n; k++) {
      const T lki = l[ci + k];
      s0 -= lki * x0[k];
      s1 -= lki * x1[k];
      s2 -= lki * x2[k];
      s3 -= lki * x3[k];
    }
    const T dg = l[ci + i];
    x0[i] = s0 / dg;
    x1[i] = s1 / dg;
    x2[i] = s2 / dg;
    x3[i] = s3 / dg;
  }
}

template <typename T>
__device__ void chol_back2_n(int n, const T* __restrict__ l,
                             T* __restrict__ x0, T* __restrict__ x1) {
  for (int i = n - 1; i >= 0; i--) {
    const int ci = i * n;
    T s0 = x0[i], s1 = x1[i];
    for (int k = i + 1; k < n; k++) {
      const T lki = l[ci + k];
      s0 -= lki * x0[k];
      s1 -= lki * x1[k];
    }
    const T dg = l[ci + i];
    x0[i] = s0 / dg;
    x1[i] = s1 / dg;
  }
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

// (angular_jacobian_acc lived here; K9a folded its body — one ancestor
// walk writing row[VOFF + F_COORD[k]] += sign*axw[k][t] plus
// row[3+t] += sign — into the shared weld walk in constraint_solve,
// which was its only caller. point_jacobian_acc survives for contacts.)

// RHS seed: the full-length zero fill is load-bearing for
// chol_fwd_unit_n / chol_back_n — see chol_solve_unit_n's header (the
// skipped forward rows keep these +0.0 for the dense back substitution
// to read).
template <typename T, bool FC>
__device__ __forceinline__ void ecol_seed(Solver<T, FC>& __restrict__ s,
                                          int j) {
  T* col = s.ecol[j];
  for (int i = 0; i < NV; i++) col[i] = T(0);
  col[VOFF + j] = T(1);
}

// Lazily compute the plain M^-1 e_j column (shared by step-8 cap/limit
// rows and the step-11 clamps — same expression tree as Rust's three
// separate caches).
template <typename T, bool FC>
__device__ void ensure_ecol(Dyn<T>& __restrict__ d,
                            Solver<T, FC>& __restrict__ s, int j) {
  if (s.ecomp[j]) return;
  ecol_seed(s, j);
  chol_solve_unit_n(NV, d.m, s.ecol[j], VOFF + j);
  s.ecomp[j] = true;
}

// Lazily compute the weld-null-space-projected column + composite
// mobility (solver.rs capcol path; also the limit-row direction).
template <typename T, bool FC>
__device__ void ensure_pcol(Dyn<T>& __restrict__ d,
                            Solver<T, FC>& __restrict__ s, int j) {
  if (s.pcomp[j]) return;
  ensure_ecol(d, s, j);
  const T plain = s.ecol[j][VOFF + j];
  if constexpr (FC) {
    // env v11 CONTACT mode: there is no weld block, so solver.rs's
    // `if nweld > 0` projection never runs — the projected column IS the
    // plain column (read through s.ecol; see the Solver note) and the
    // composite mobility IS `plain`. The COMPOSITE_EPS verdict is kept
    // verbatim rather than assumed: Rust still evaluates
    // `dnew > COMPOSITE_EPS * plain` with dnew == plain, which is true
    // for every positive mobility and false for a degenerate one.
    s.pdiag[j] = plain > T(COMPOSITE_EPS) * plain
                     ? plain
                     : std::numeric_limits<T>::infinity();
    s.pcomp[j] = true;
  } else {
    T* col = s.pcol[j];
    // K20-1: the 12 dots, in one pass over i. There used to be a COPY
    // half fused into this pass (col[i] = s.ecol[j][i] over all NV,
    // including a copy-only range) whose only consumer was the projection
    // loop below, which re-read col[i] as its seed. The projection now
    // reads s.ecol[j][i] directly and writes col[i] as its OUTPUT only:
    // 35 stores and 35 dependent re-loads per column disappear, and the
    // store->load forwarding edge between the two passes is cut.
    // BITWISE: col[i] was set EXACTLY = s.ecol[j][i] (a plain copy, whole
    // range covered — 0..5, VOFF..WJ_L0, WJ_L0..+6, WJ_R0..+6 = 0..NV-1
    // by the WJ_R0 + 6 == NV static_assert), so seeding from the source
    // is the same bytes. Each sum[w] still accumulates over i in
    // ascending order over the same terms.
    {
      // The dot half is segmented onto the K8 weld support: each sum[w]
      // still visits its own terms in ascending i, only the structural
      // +-0.0 products are gone and the exactly-1.0 entries became bare
      // adds.
      T sum[N_WELD];
#pragma unroll
      for (int w = 0; w < N_WELD; w++) sum[w] = T(0);
#pragma unroll
      for (int t = 0; t < 3; t++) {  // i = 0,1,2: linear rows t, 6+t (1.0)
        const T ci = s.ecol[j][t];
        sum[t] += ci;
        sum[6 + t] += ci;
      }
#pragma unroll
      for (int u = 0; u < 3; u++) {  // i = 3,4,5
        const T ci = s.ecol[j][3 + u];
        sum[3 + u] += ci;  // angular rows 3+u, 9+u (1.0)
        sum[9 + u] += ci;
#pragma unroll
        for (int t = 0; t < 3; t++)
          if (t != u) {
            sum[t] += s.jwr[t][WJ_RIDX(t, u)] * ci;
            sum[6 + t] += s.jwr[3 + t][WJ_RIDX(t, u)] * ci;
          }
      }
      // (rows VOFF..WJ_L0 are structurally outside the weld support and,
      // since K20-1, need no copy either — nothing to do here)
#pragma unroll
      for (int e = 0; e < 6; e++) {  // left-foot chain: rows 0..5
        const T ci = s.ecol[j][WJ_L0 + e];
#pragma unroll
        for (int w = 0; w < N_WELD / 2; w++) sum[w] += s.jwc[w][e] * ci;
      }
#pragma unroll
      for (int e = 0; e < 6; e++) {  // right-foot chain: rows 6..11
        const T ci = s.ecol[j][WJ_R0 + e];
#pragma unroll
        for (int w = N_WELD / 2; w < N_WELD; w++) sum[w] += s.jwc[w][e] * ci;
      }
#pragma unroll
      for (int w = 0; w < N_WELD; w++) s.rw[w] = sum[w];
    }
    chol_solve_n(N_WELD, s.aww, s.rw);
    {
      // Same order-preservation argument as the PGS weld update: all 12
      // multipliers are known before the loop (chol_solve_n has finished
      // with s.rw), and each col[i] still sees w = 0..11 applied in that
      // exact sequence — only its 12 memory round trips are gone.
      T x[N_WELD];
#pragma unroll
      for (int w = 0; w < N_WELD; w++) x[w] = s.rw[w];
      for (int i = 0; i < NV; i++) {
        T ci = s.ecol[j][i];  // K20-1: seed from the source, not the copy
#pragma unroll
        for (int w = 0; w < N_WELD; w++) ci -= s.mw[i][w] * x[w];
        col[i] = ci;
      }
    }
    const T dnew = col[VOFF + j];
    s.pdiag[j] = dnew > T(COMPOSITE_EPS) * plain
                     ? dnew
                     : std::numeric_limits<T>::infinity();
    s.pcomp[j] = true;
  }
}

// K7: BATCHED ensure_pcol. B columns of the limit-row group are computed
// together so they share the three big reads:
//   (a) the dot pass reads each support value ONCE for all
//       B columns;
//   (b) the 12x12 weld solve becomes chol_solve_multi_n<B> — the proven
//       multi-RHS loop interchange, bitwise per right-hand side;
//   (c) the projection pass reads s.mw[i][w] ONCE for all B columns.
// BITWISE: the columns are mutually independent (distinct ecol/pcol
// rows), every sum[b][w] still runs over the same terms in ascending i
// from the same +0.0 seed, and every col[b][i] still applies w = 0..11
// in that exact sequence — only the b loop is hoisted inside, which
// reassociates nothing. NO row blocking is involved anywhere here, so
// K6's back-substitution prohibition does not apply.
//
// PRECONDITIONS at the one call site (constraint_solve's limit
// projection), all three guaranteed by the substep that calls in:
//   - s.ecomp[j] is already true for every j in the group — the batched
//     unit-solve block computes the WHOLE s.l_j group a few lines above
//     — so ensure_ecol would be a no-op here; it is left out rather than
//     drag B copies of chol_solve_unit_n into this instantiation.
//   - s.pcomp[j] is false for every j (substep clears it right before
//     constraint_solve and this is the substep's first ensure_pcol), so
//     the lazy-cache early-out cannot fire.
//   - s.l_j is strictly ascending with distinct joints, so the B
//     ecol/pcol row pairs never alias.
// pdiag / pcomp are written exactly as the scalar path writes them.
template <int B, typename T, bool FC>
__device__ void ensure_pcol_multi(Solver<T, FC>& __restrict__ s,
                                  const int* js) {
  static_assert(!FC, "the batched weld-null projection is weld mode only");
  int j[B];
  T plain[B];
#pragma unroll
  for (int b = 0; b < B; b++) {
    j[b] = js[b];
    plain[b] = s.ecol[j[b]][VOFF + j[b]];
  }
  T rw[B][N_WELD];
  {
    // Same segmentation onto the K8 weld support as the scalar body: the
    // copy half stays full-length, the dot half visits only the support.
    T sum[B][N_WELD];
#pragma unroll
    for (int b = 0; b < B; b++) {
#pragma unroll
      for (int w = 0; w < N_WELD; w++) sum[b][w] = T(0);
    }
#pragma unroll
    for (int t = 0; t < 3; t++) {  // i = 0,1,2: linear rows t, 6+t (1.0)
      T ci[B];
#pragma unroll
      for (int b = 0; b < B; b++) ci[b] = s.ecol[j[b]][t];
#pragma unroll
      for (int b = 0; b < B; b++) {
        sum[b][t] += ci[b];
        sum[b][6 + t] += ci[b];
      }
    }
#pragma unroll
    for (int u = 0; u < 3; u++) {  // i = 3,4,5
      T ci[B];
#pragma unroll
      for (int b = 0; b < B; b++) ci[b] = s.ecol[j[b]][3 + u];
#pragma unroll
      for (int b = 0; b < B; b++) {  // angular rows 3+u, 9+u (1.0)
        sum[b][3 + u] += ci[b];
        sum[b][9 + u] += ci[b];
      }
#pragma unroll
      for (int t = 0; t < 3; t++)
        if (t != u) {
          const T jl = s.jwr[t][WJ_RIDX(t, u)];
          const T jr = s.jwr[3 + t][WJ_RIDX(t, u)];
#pragma unroll
          for (int b = 0; b < B; b++) {
            sum[b][t] += jl * ci[b];
            sum[b][6 + t] += jr * ci[b];
          }
        }
    }
    // (K20-1: the copy-only range VOFF..WJ_L0 is gone with the rest of
    // the copy half — the projection below seeds from s.ecol directly)
#pragma unroll
    for (int e = 0; e < 6; e++) {  // left-foot chain: rows 0..5
      T ci[B];
#pragma unroll
      for (int b = 0; b < B; b++) ci[b] = s.ecol[j[b]][WJ_L0 + e];
#pragma unroll
      for (int w = 0; w < N_WELD / 2; w++) {
        const T jwi = s.jwc[w][e];
#pragma unroll
        for (int b = 0; b < B; b++) sum[b][w] += jwi * ci[b];
      }
    }
#pragma unroll
    for (int e = 0; e < 6; e++) {  // right-foot chain: rows 6..11
      T ci[B];
#pragma unroll
      for (int b = 0; b < B; b++) ci[b] = s.ecol[j[b]][WJ_R0 + e];
#pragma unroll
      for (int w = N_WELD / 2; w < N_WELD; w++) {
        const T jwi = s.jwc[w][e];
#pragma unroll
        for (int b = 0; b < B; b++) sum[b][w] += jwi * ci[b];
      }
    }
#pragma unroll
    for (int b = 0; b < B; b++) {
#pragma unroll
      for (int w = 0; w < N_WELD; w++) rw[b][w] = sum[b][w];
    }
  }
  chol_solve_multi_n<B>(N_WELD, s.aww, &rw[0][0]);
  {
    T x[B][N_WELD];
#pragma unroll
    for (int b = 0; b < B; b++) {
#pragma unroll
      for (int w = 0; w < N_WELD; w++) x[b][w] = rw[b][w];
    }
    for (int i = 0; i < NV; i++) {
      T ci[B];
#pragma unroll
      for (int b = 0; b < B; b++) ci[b] = s.ecol[j[b]][i];  // K20-1
#pragma unroll
      for (int w = 0; w < N_WELD; w++) {
        const T mwi = s.mw[i][w];
#pragma unroll
        for (int b = 0; b < B; b++) ci[b] -= mwi * x[b][w];
      }
#pragma unroll
      for (int b = 0; b < B; b++) s.pcol[j[b]][i] = ci[b];
    }
  }
#pragma unroll
  for (int b = 0; b < B; b++) {
    const T dnew = s.pcol[j[b]][VOFF + j[b]];
    s.pdiag[j[b]] = dnew > T(COMPOSITE_EPS) * plain[b]
                        ? dnew
                        : std::numeric_limits<T>::infinity();
    s.pcomp[j[b]] = true;
  }
}

// ---- solver.rs ConstraintSet::solve (steps 7-9) -----------------------------
template <typename T, bool FC>
__device__ void constraint_solve(Dyn<T>& __restrict__ d,
                                 Solver<T, FC>& __restrict__ s, const T* qpre,
                                 T* __restrict__ v) {
  const T hinv = T(1) / T(H_SUB);

  // -- assembly: weld rows 0-11 (foot_mode = 0 only) --------------------------
  // K9a — ONE ancestor walk per foot emits all six rows. The three
  // point_jacobian_acc calls differed only in n = e_t, so each of them
  // re-derived the SAME rel = pf - pos[k] and w = axw[k] x rel and then
  // dotted it with a basis vector; dot3(e_t, w) is w[t] (1.0*w[t] is
  // exact, and the two +-0.0 terms are absorbed by the row's +0.0 zero
  // fill -- Divergences 20c). The three angular_jacobian_acc calls
  // walked the same chain again for axw[k][t]. Likewise rel0 = pf - d.p
  // and rxn = cross(rel0, e_t) are shared, in closed form:
  //   cross(a, e_0) = (  0, a[2], -a[1])
  //   cross(a, e_1) = (-a[2],  0,  a[0])
  //   cross(a, e_2) = ( a[1], -a[0],  0)
  // whose t-th component is exactly K8's structural +-0.0 (it lands in
  // column 3+t, which is OUTSIDE the refined support).
  // K9b — every entry below is a SINGLE `T(0) + value` store into a
  // column the walk touches at most once (F_COORD injective on the
  // chain, root columns disjoint from it -- see wj_chain_mask's
  // static_assert). `T(0) + x` is NOT the identity when x is -0.0, and
  // that is the point: it reproduces the zero fill's +0.0 absorption
  // bit for bit, so the stores are what `fill(+0.0); row[col] += x`
  // wrote. `T(0) - x` does the same for the negated components.
  // K9c — the rows are built DIRECTLY IN s.mw (same zero fill, same
  // values, same single writes: s.mw was a verbatim copy of s.jw, so
  // writing one instead of both is the identity on every element), and
  // the 84 support values are teed off into s.jwc / s.jwr as they are
  // stored -- no reload, and the exactly-1.0 entries are not stored.
  if constexpr (!FC)
  for (int widx = 0; widx < 2; widx++) {
    const int frame = WELD_FRAMES[widx];
    const T* pf = d.pos[frame];
    const int r0 = 6 * widx;
    // K11 layout: same 6 * NV stores of +0.0, i-outer so the six of them
    // at a fixed column are consecutive words.
    for (int i = 0; i < NV; i++)
#pragma unroll
      for (int m = 0; m < 6; m++) s.mw[i][r0 + m] = T(0);
    for (int k = frame; k >= 0; k = F_PARENT[k]) {
      if (F_COORD[k] < 0) continue;
      const int col = VOFF + F_COORD[k];
      const int e = col - (WJ_L0 + 6 * widx);
      T rel[3], w[3];
      sub3v(pf, d.pos[k], rel);
      cross3(d.axw[k], rel, w);
#pragma unroll
      for (int t = 0; t < 3; t++) {
        const T lv = T(0) + w[t];
        s.mw[col][r0 + t] = lv;
        s.jwc[r0 + t][e] = lv;
        const T av = T(0) + d.axw[k][t];
        s.mw[col][r0 + 3 + t] = av;
        s.jwc[r0 + 3 + t][e] = av;
      }
    }
    T rel0[3];
    sub3v(pf, d.p, rel0);
    const T v04 = T(0) + rel0[2], v05 = T(0) - rel0[1];
    const T v13 = T(0) - rel0[2], v15 = T(0) + rel0[0];
    const T v23 = T(0) + rel0[1], v24 = T(0) - rel0[0];
    s.mw[0][r0 + 0] = T(1);
    s.mw[4][r0 + 0] = v04;
    s.mw[5][r0 + 0] = v05;
    s.mw[1][r0 + 1] = T(1);
    s.mw[3][r0 + 1] = v13;
    s.mw[5][r0 + 1] = v15;
    s.mw[2][r0 + 2] = T(1);
    s.mw[3][r0 + 2] = v23;
    s.mw[4][r0 + 2] = v24;
    s.jwr[3 * widx + 0][0] = v04;  // cols WJ_R2[0] = {4, 5}
    s.jwr[3 * widx + 0][1] = v05;
    s.jwr[3 * widx + 1][0] = v13;  // cols WJ_R2[1] = {3, 5}
    s.jwr[3 * widx + 1][1] = v15;
    s.jwr[3 * widx + 2][0] = v23;  // cols WJ_R2[2] = {3, 4}
    s.jwr[3 * widx + 2][1] = v24;
#pragma unroll
    for (int t = 0; t < 3; t++) {
      s.mw[3 + t][r0 + 3 + t] = T(1);
      s.bias_w[r0 + t] =
          -(T(BETA_WELD) * hinv) * (pf[t] - T(WELD_TARGET_P[widx][t]));
    }
    T phi[3];
    rot_log(d.r[frame], phi);
#pragma unroll
    for (int t = 0; t < 3; t++)
      s.bias_w[r0 + 3 + t] = -(T(BETA_WELD) * hinv) * phi[t];
  }

  // -- env v11 GROUND BLOCK, where the weld rows used to be -------------------
  // MIRROR of collide.rs `ground_phase` + solver.rs `assemble_ground_rows`,
  // transcribed SERIALLY and without any of the K5/K8/K9 weld tricks: the
  // corner order is the spec §2 normative one (foot_l then foot_r,
  // GROUND_CORNERS order within a foot), the row order the §3 one (normal,
  // friction x, friction z), and each row is the plain
  // `zero-fill + point_jacobian_acc` the Rust reference writes. The active
  // set is data-dependent exactly as the 55-pair contact set is, and it
  // uses the SAME compaction convention (s.ng counts ACTIVE corners, rows
  // packed from 0), so a thread assembles the rows Rust assembles, in
  // Rust's order. Trip count over the block is fixed by s.ng, with no
  // early exit inside it — the §2 contact loop's contract.
  if constexpr (FC) {
    s.ng = 0;
    s.g_bucket = 0;
    for (int f = 0; f < 2; f++) {
      const int frame = GROUND_FRAMES[f];
      for (int k = 0; k < CORNERS_PER_FOOT; k++) {
        // world corner = p[frame] + R[frame] * corner_k  (fk.rs
        // add3(ws.p, mat_vec(ws.r, .)), same term order as the capsule
        // endpoints hoisted below)
        T pt[3];
        for (int i = 0; i < 3; i++)
          pt[i] = d.pos[frame][i]
                  + (d.r[frame][i][0] * T(GROUND_CORNERS[k][0])
                     + d.r[frame][i][1] * T(GROUND_CORNERS[k][1])
                     + d.r[frame][i][2] * T(GROUND_CORNERS[k][2]));
        // the plane is y = 0 and its normal is +y, so the signed gap is
        // the corner's world y outright
        const T gap = pt[1];
        if (gap < T(MARGIN)) {
          const int nrow = s.ng * GROUND_ROWS_PER_CORNER;
          for (int t = 0; t < GROUND_ROWS_PER_CORNER; t++) {
            T* jr = s.g_J[nrow + t];
            for (int i = 0; i < NV; i++) jr[i] = T(0);
            const T n[3] = {T(GROUND_DIRS[t][0]), T(GROUND_DIRS[t][1]),
                            T(GROUND_DIRS[t][2])};
            // static ground: only the corner's OWN chain contributes, so
            // this is one signed point_jacobian_acc, not the contact
            // rows' subtract-a-then-add-b pair
            point_jacobian_acc(d, frame, pt, n, T(1), jr);
          }
          // NORMAL row rhs: the §3-step-8 contact pipeline verbatim —
          // speculative approach limit while the gap is open, Baumgarte
          // push-out beyond SLOP once penetrating
          s.g_bias[nrow] = gap >= T(0)
                               ? -gap * hinv
                               : (T(BETA_CONTACT) * hinv)
                                     * tmax(-gap - T(SLOP), T(0));
          // FRICTION rows rhs = 0: friction stabilizes nothing, so it
          // gets no Baumgarte term (contact spec §3)
          s.g_bias[nrow + 1] = T(0);
          s.g_bias[nrow + 2] = T(0);
          // contact spec §7: GROUND_CORNERS order puts z+ (toe) at odd
          // k, so the obs bucket is foot*2 + (k & 1)
          s.g_bucket |= (uint32_t)(f * 2 + (k & 1)) << (2 * s.ng);
          s.ng = s.ng + 1;
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
      ca[c][i] = d.pos[f][i] + (d.r[f][i][0] * T(COL_A[c][0])
                                + d.r[f][i][1] * T(COL_A[c][1])
                                + d.r[f][i][2] * T(COL_A[c][2]));
      cb[c][i] = d.pos[f][i] + (d.r[f][i][0] * T(COL_B[c][0])
                                + d.r[f][i][1] * T(COL_B[c][1])
                                + d.r[f][i][2] * T(COL_B[c][2]));
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
    const T gap = dist - T(COL_R[ia] + COL_R[ib]);
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
        const T s1 = c1[i] + n[i] * T(COL_R[ia]);
        const T s2 = c2[i] - n[i] * T(COL_R[ib]);
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
    const T gl = qpre[j] - T(LIMIT_LO[j]);
    const T gh = T(LIMIT_HI[j]) - qpre[j];
    const T reach = T(H_SUB) * T(OMEGA_CAP[j]);
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
          gap >= T(0) ? -gap * hinv : (T(BETA_WELD) * hinv) * -gap;
      s.nl = li + 1;
    }
  }

  // -- effective masses (row order: welds, contacts, limits) ------------------
  // 12 independent solves against ONE factor: batched (see
  // chol_solve_multi_n — a loop interchange, bitwise). K11: s.mw is
  // T[NV][N_WELD], so the RHS-MINOR entry point takes it (same solve,
  // one address map; see chol_solve_multi_tn).
  // K9c: the assembly above already wrote J_w here, so the 420-element
  // seed copy that used to stand at this point is gone.
  if constexpr (!FC) chol_solve_multi_tn<N_WELD>(NV, d.m, &s.mw[0][0]);
  // Ground rows (contact mode): the SAME 4 / 2 / 1 ladder the contact
  // rows take below. Rust runs ONE ladder over ground++contacts, so one
  // of its groups can straddle the block boundary; the rows are mutually
  // independent and the batch is a pure loop interchange (A1), so every
  // M^-1 J^T column and every diag is bit-for-bit what any grouping
  // produces — running the two blocks as two ladders changes no value.
  if constexpr (FC) {
    const int ngr = s.ng * GROUND_ROWS_PER_CORNER;
    int gi = 0;
    for (; gi + 3 < ngr; gi += 4) {
      for (int t = 0; t < 4; t++)
        for (int i = 0; i < NV; i++) s.g_minv[gi + t][i] = s.g_J[gi + t][i];
      chol_solve_multi_n<4>(NV, d.m, &s.g_minv[gi][0]);
      for (int t = 0; t < 4; t++) {
        T diag = T(0);
        for (int i = 0; i < NV; i++)
          diag += s.g_J[gi + t][i] * s.g_minv[gi + t][i];
        s.g_diag[gi + t] = diag;
      }
    }
    if (gi + 1 < ngr) {
      for (int t = 0; t < 2; t++)
        for (int i = 0; i < NV; i++) s.g_minv[gi + t][i] = s.g_J[gi + t][i];
      chol_solve_multi_n<2>(NV, d.m, &s.g_minv[gi][0]);
      for (int t = 0; t < 2; t++) {
        T diag = T(0);
        for (int i = 0; i < NV; i++)
          diag += s.g_J[gi + t][i] * s.g_minv[gi + t][i];
        s.g_diag[gi + t] = diag;
      }
      gi += 2;
    }
    for (; gi < ngr; gi++) {
      for (int i = 0; i < NV; i++) s.g_minv[gi][i] = s.g_J[gi][i];
      chol_solve_n(NV, d.m, s.g_minv[gi]);
      T diag = T(0);
      for (int i = 0; i < NV; i++) diag += s.g_J[gi][i] * s.g_minv[gi][i];
      s.g_diag[gi] = diag;
    }
  }
  // Contact rows: s.nc is RUNTIME (0..55, typically a handful), so the
  // batch runs as a 4 / 2 / 1 ladder — the 2-wide tier is worth its own
  // branch here (+0.4% on train --bench over 4-with-scalar-remainder),
  // which says the common nc is small. Rows are mutually
  // independent (c_minv[ci] depends only on c_J[ci] and the factor,
  // c_diag[ci] only on those two), so grouping the copies / solves /
  // diagonals changes no value; the solve itself is the same bitwise
  // interchange as the weld batch.
  {
    int ci = 0;
    for (; ci + 3 < s.nc; ci += 4) {
      for (int t = 0; t < 4; t++)
        for (int i = 0; i < NV; i++) s.c_minv[ci + t][i] = s.c_J[ci + t][i];
      chol_solve_multi_n<4>(NV, d.m, &s.c_minv[ci][0]);
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
      chol_solve_multi_n<2>(NV, d.m, &s.c_minv[ci][0]);
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
      chol_solve_n(NV, d.m, s.c_minv[ci]);
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
  // LOAD-BEARING for K7: this block must keep covering the WHOLE s.l_j
  // group — ensure_pcol_multi below takes s.ecomp[j] as a precondition.
  //
  // Width: s.nl measures mean 7.8, range 3-14 (f32 trace, 256 envs), so
  // an 8-wide tier was tried too and won NOTHING over 4 (7,283 -> 7,286
  // steps/s) for ~30 more lines — dropped. See the block-size note in
  // the header: the gain saturates at four independent chains.
  {
    int li = 0;
    for (; li + 3 < s.nl; li += 4) {
      const int j0 = s.l_j[li], j1 = s.l_j[li + 1];
      const int j2 = s.l_j[li + 2], j3 = s.l_j[li + 3];
      ecol_seed(s, j0);
      ecol_seed(s, j1);
      ecol_seed(s, j2);
      ecol_seed(s, j3);
      // s.l_j ascending => VOFF + j0 is the group's smallest m (K4a)
      chol_fwd_unit4_n(NV, d.m, s.ecol[j0], s.ecol[j1], s.ecol[j2],
                       s.ecol[j3], VOFF + j0);
      chol_back4_n(NV, d.m, s.ecol[j0], s.ecol[j1], s.ecol[j2], s.ecol[j3]);
      s.ecomp[j0] = true;
      s.ecomp[j1] = true;
      s.ecomp[j2] = true;
      s.ecomp[j3] = true;
    }
    if (li + 1 < s.nl) {
      const int j0 = s.l_j[li], j1 = s.l_j[li + 1];
      ecol_seed(s, j0);
      ecol_seed(s, j1);
      chol_fwd_unit2_n(NV, d.m, s.ecol[j0], s.ecol[j1], VOFF + j0);
      chol_back2_n(NV, d.m, s.ecol[j0], s.ecol[j1]);
      s.ecomp[j0] = true;
      s.ecomp[j1] = true;
      li += 2;
    }
    for (; li < s.nl; li++) ensure_ecol(d, s, s.l_j[li]);
  }

  // -- weld block A_ww = J_w M^-1 J_w^T, factored once ------------------------
  // entry (r, c>=r) = sum_i J_row_c[i] * minvjt_row_r[i]. A_ww is
  // symmetric, and its only consumers are chol_factor_n (in place) and
  // chol_solve_n, both of which read row >= col only (see crba's K4c
  // note), so ONLY the (row c, col r) entry (lower, c >= r) is stored —
  // the (r, c) mirror was dead. c == r writes the diagonal once. K12:
  // that entry now lives at aww[r*N_WELD + c] (column-major); same
  // value, transposed address, and A_ww's symmetry makes the choice of
  // which triangle to fill immaterial to the values.
  // Only row c's support contributes (s.mw is dense, but it would be
  // multiplied by row c's structural zeros). K8 narrows that
  // from K5's 12 columns to 9 (linear c) / 7 (angular c) and contracts
  // the one exactly-1.0 entry as a bare add. Ascending column order is
  // preserved: WJ_ONE(c) < WJ_R2[.] < chain for both row classes.
  // The two `#pragma unroll` below are LOAD-BEARING, not decoration:
  // they make c a compile-time constant so WJ_ONE / WJ_ANG / WJ_CH0 /
  // WJ_R2 fold to immediates instead of arithmetic plus a cmem load.
  // MEASURED without them (K8, 5 interleaved rounds, N=4096): 25,838.9
  // -> 25,809.0 = -0.12% on f32, and that is despite the un-unrolled
  // build being SMALLER (SASS f32 38,070 -> ~33k) and clearing f64's
  // 8-byte spill. Rejected — f32 is the production dtype.
  if constexpr (!FC) {
#pragma unroll
  for (int r = 0; r < N_WELD; r++)
#pragma unroll
    for (int c = r; c < N_WELD; c++) {
      T sum = T(0);
      sum += s.mw[WJ_ONE(c)][r];  // the row's entry there is exactly 1.0
      if (!WJ_ANG(c)) {
        sum += s.jwr[WJ_LIN(c)][0] * s.mw[WJ_R2[WJ_T(c)][0]][r];
        sum += s.jwr[WJ_LIN(c)][1] * s.mw[WJ_R2[WJ_T(c)][1]][r];
      }
      const int lo = WJ_CH0(c);
#pragma unroll
      for (int e = 0; e < 6; e++) sum += s.jwc[c][e] * s.mw[lo + e][r];
      s.aww[r * N_WELD + c] = sum;
    }
  chol_factor_n(N_WELD, s.aww, &d.pd_bad);  // in place: aww -> L_ww
  }

  // -- limit rows: project into the weld null space ---------------------------
  // BATCHED (K7) on the same 4 / 2 / 1 ladder as the unit solves above:
  // the whole group is uncomputed and its columns are independent, so B
  // of them share the jw read, the 12x12 solve and the mw read. See
  // ensure_pcol_multi's header for the preconditions this relies on
  // (they are exactly what the ecol block above establishes).
  // WIDTH probed (5 / 3 / 3 interleaved rounds vs this build, N = 4096):
  //   4 / 2 / 1 (this)          24,224 -> 25,370  +4.73%   KEPT
  //   2 / 1 only                24,240 -> 24,603  +1.50%   rejected
  //   8 / 4 / 2 / 1             25,366 -> 25,192  -0.69%   REVERTED
  // (the two rejected rows were measured against the 4-wide arm in its
  // own pairs, hence the slightly different A-arm readings)
  // Same saturation the ecol ladder found: four independent chains is
  // where the gain tops out. The 8-tier fits (f32 still 0 spills, 255
  // regs, frame 43,472) but nl means 7.8, so it fires on about half the
  // substeps and pays a doubled live-x[8][12] projection loop for it.
  //
  // CONTACT MODE takes the scalar arm only: with no weld block the
  // projection is the identity (solver.rs's `if nweld > 0` block does not
  // run), so ensure_pcol just records the plain mobility and there is
  // nothing to batch.
  {
    int li = 0;
    if constexpr (!FC) {
      for (; li + 3 < s.nl; li += 4) ensure_pcol_multi<4>(s, &s.l_j[li]);
      if (li + 1 < s.nl) {
        ensure_pcol_multi<2>(s, &s.l_j[li]);
        li += 2;
      }
    }
    for (; li < s.nl; li++) ensure_pcol(d, s, s.l_j[li]);
  }
  for (int li = 0; li < s.nl; li++) {
    s.l_diag[li] = s.pdiag[s.l_j[li]];
    s.l_lam[li] = T(0);
  }
  for (int ci = 0; ci < s.nc; ci++) s.c_lam[ci] = T(0);
  if constexpr (FC)
    for (int gi = 0; gi < s.ng * GROUND_ROWS_PER_CORNER; gi++)
      s.g_lam[gi] = T(0);

  // -- step 8: exactly 12 iterations, fixed order -----------------------------
  for (int iter = 0; iter < PGS_ITERS; iter++) {
    // (a) weld block: exact solve of the current residual.
    // Both halves are LOOP INTERCHANGES of the 12 x NV nests (bitwise —
    // see the argument below); v[i] is touched once per i instead of 12
    // times, and the 12 dots / the 12 terms of each v[i] become
    // independent work instead of 12 back-to-back serial passes.
    if constexpr (!FC) {
    {
      // 12 dots against the SAME v: each jv[r] accumulates over i in the
      // identical ascending order over identical terms (v is read-only
      // here), so every jv[r] is bitwise the scalar loop's.
      // segmented onto the K8 weld support via the per-i contributing-row
      // table (same ascending i per jv[r]; 96 terms, 12 of them 1.0)
      T jv[N_WELD];
#pragma unroll
      for (int r = 0; r < N_WELD; r++) jv[r] = T(0);
#pragma unroll
      for (int t = 0; t < 3; t++) {  // i = 0,1,2: linear rows t, 6+t (1.0)
        const T vi = v[t];
        jv[t] += vi;
        jv[6 + t] += vi;
      }
#pragma unroll
      for (int u = 0; u < 3; u++) {  // i = 3,4,5
        const T vi = v[3 + u];
        jv[3 + u] += vi;  // angular rows 3+u, 9+u (1.0)
        jv[9 + u] += vi;
#pragma unroll
        for (int t = 0; t < 3; t++)
          if (t != u) {  // linear rows with t != u carry rxn[u]
            jv[t] += s.jwr[t][WJ_RIDX(t, u)] * vi;
            jv[6 + t] += s.jwr[3 + t][WJ_RIDX(t, u)] * vi;
          }
      }
#pragma unroll
      for (int e = 0; e < 6; e++) {
        const T vi = v[WJ_L0 + e];
#pragma unroll
        for (int r = 0; r < N_WELD / 2; r++) jv[r] += s.jwc[r][e] * vi;
      }
#pragma unroll
      for (int e = 0; e < 6; e++) {
        const T vi = v[WJ_R0 + e];
#pragma unroll
        for (int r = N_WELD / 2; r < N_WELD; r++) jv[r] += s.jwc[r][e] * vi;
      }
#pragma unroll
      for (int r = 0; r < N_WELD; r++) s.rw[r] = s.bias_w[r] - jv[r];
    }
    chol_solve_n(N_WELD, s.aww, s.rw);
    {
      // All 12 dl are known before the update (chol_solve_n has already
      // finished writing s.rw and nothing below writes it), so the r
      // loop can move inside. ORDER-PRESERVING per element: the r-outer
      // form applied r = 0..11 to v[i] in that sequence, and so does
      // this one — the only thing removed is v[i]'s 12 round trips
      // through memory, which are the identity on the value.
      T dl[N_WELD];
#pragma unroll
      for (int r = 0; r < N_WELD; r++) dl[r] = s.rw[r];
      for (int i = 0; i < NV; i++) {
        T vi = v[i];
#pragma unroll
        for (int r = 0; r < N_WELD; r++) vi += s.mw[i][r] * dl[r];
        v[i] = vi;
      }
    }
    }
    // (a2) env v11 GROUND BLOCK (contact spec §3), corner order: the
    // unilateral normal row first (the §2 contact path verbatim), then
    // its two friction rows BOX-clamped against the lambda_n THIS sweep
    // just produced — Gauss-Seidel coupling, re-clamped on every one of
    // the PGS_ITERS iterations. MIRROR of solver.rs `pgs_ground_sweep` +
    // `pgs_row_update_box`. It runs where the weld block would, with the
    // same fixed iteration count and no early exit.
    if constexpr (FC) {
      for (int gi = 0; gi < s.ng; gi++) {
        const int nrow = gi * GROUND_ROWS_PER_CORNER;
        {
          T jv = T(0);
          for (int i = 0; i < NV; i++) jv += s.g_J[nrow][i] * v[i];
          const T dl = (s.g_bias[nrow] - jv) / s.g_diag[nrow];
          const T newl = tmax(s.g_lam[nrow] + dl, T(0));  // lambda_n >= 0
          const T dd = newl - s.g_lam[nrow];
          s.g_lam[nrow] = newl;
          if (dd != T(0))
            for (int i = 0; i < NV; i++) v[i] += s.g_minv[nrow][i] * dd;
        }
        // the friction cone is read AFTER the normal update, so a corner
        // that just lost its normal impulse loses its friction in the
        // same sweep
        const T bound = T(GROUND_MU) * s.g_lam[nrow];
        for (int t = 1; t < GROUND_ROWS_PER_CORNER; t++) {
          const int row = nrow + t;
          T jv = T(0);
          for (int i = 0; i < NV; i++) jv += s.g_J[row][i] * v[i];
          const T dl = (s.g_bias[row] - jv) / s.g_diag[row];
          // |lambda_t| <= MU * lambda_n. Rust writes
          // `(lambda + dl).min(bound).max(-bound)`; tmin/tmax reproduce
          // f64::min/max including the NaN rule (both `a < b` and
          // `a > b` are false on NaN, so the OTHER operand is returned,
          // exactly as minnum/maxnum do) — an artifact lambda collapses
          // onto the bound instead of propagating.
          const T newl = tmax(tmin(s.g_lam[row] + dl, bound), -bound);
          const T dd = newl - s.g_lam[row];
          s.g_lam[row] = newl;
          if (dd != T(0))
            for (int i = 0; i < NV; i++) v[i] += s.g_minv[row][i] * dd;
        }
      }
    }
    // (b) contacts, pair-id order: projected GS, lambda >= 0
    // This dot is DENSE ON PURPOSE. c_J[ci] has 12-23 structurally
    // nonzero columns of 35 and K14 narrowed it to a two-segment cover
    // of the pair's compile-time support: -1.42% / -0.90% / -0.29% in
    // three forms, all reverted. The bound would be runtime, which
    // costs this loop its full unrolling — see the K14 block in the
    // header before touching it.
    for (int ci = 0; ci < s.nc; ci++) {
      T jv = T(0);
      for (int i = 0; i < NV; i++) jv += s.c_J[ci][i] * v[i];
      const T dl = (s.c_bias[ci] - jv) / s.c_diag[ci];
      const T newl = tmax(s.c_lam[ci] + dl, T(0));
      const T dd = newl - s.c_lam[ci];
      s.c_lam[ci] = newl;
      if (dd != T(0))
        for (int i = 0; i < NV; i++) v[i] += s.c_minv[ci][i] * dd;
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
    for (int li = 0; li < s.nl; li++) {
      const int j = s.l_j[li];
      const T jv = s.l_sign[li] * v[VOFF + j];
      const T dl = (s.l_bias[li] - jv) / s.l_diag[li];
      const T newl = tmax(s.l_lam[li] + dl, T(0));
      const T dd = newl - s.l_lam[li];
      s.l_lam[li] = newl;
      if (dd != T(0)) {
        const T sd = s.l_sign[li] * dd;
        // contact mode reads the PLAIN column (no weld null space to
        // project into — see ensure_pcol); the weld arm is left textually
        // untouched so its codegen cannot move.
        if constexpr (FC) {
          for (int i = 0; i < NV; i++) v[i] += s.ecol[j][i] * sd;
        } else {
          for (int i = 0; i < NV; i++) v[i] += s.pcol[j][i] * sd;
        }
      }
    }
    // (d) omega-cap pass, joint order: weld-consistent impulses
    for (int j = 0; j < NQ; j++) {
      const int vj = VOFF + j;
      const T cap = T(OMEGA_CAP[j]);
      const T target = tclamp(v[vj], -cap, cap);
      if (target != v[vj]) {
        ensure_pcol(d, s, j);
        if (!isinf(s.pdiag[j])) {
          const T lam = (target - v[vj]) / s.pdiag[j];
          if constexpr (FC) {
            for (int i = 0; i < NV; i++) v[i] += s.ecol[j][i] * lam;
          } else {
            for (int i = 0; i < NV; i++) v[i] += s.pcol[j][i] * lam;
          }
          v[vj] = target;  // exact — the cap is a hard bound
        }
      }
    }
  }
}

// ---- step.rs substep (constrained) ------------------------------------------
template <typename T, bool FC>
__device__ void substep(Dyn<T>& __restrict__ d, Solver<T, FC>& __restrict__ s,
                        const T* __restrict__ tau) {
  // 1-3. FK, CRBA, RNEA bias
  fk_all(d);
  crba(d);
  rnea_bias(d);
  // 4. rhs = tau - b*qd - C (root rows 0 - C)
  for (int i = 0; i < VOFF; i++) d.rhs[i] = T(0) - d.c[i];
  for (int j = 0; j < NQ; j++)
    d.rhs[VOFF + j] =
        tau[j] - T(DAMPING[j]) * d.v[VOFF + j] - d.c[VOFF + j];
  // 5. Cholesky — IN PLACE: d.m holds L from here to the end of the
  // substep (crba rewrites all NV*NV of it next substep)
  chol_factor_n(NV, d.m, &d.pd_bad);
  // 6. v* = v + h M^-1 (tau - C)
  chol_solve_n(NV, d.m, d.rhs);
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
  //     Baumgarte weld bias. Never engages on a healthy trajectory (a
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
  for (int j = 0; j < NQ; j++) {
    const T lo = T(LIMIT_LO[j]), hi = T(LIMIT_HI[j]);
    s.hit[j] = d.q[j] < lo || d.q[j] > hi;
    // env v14: no routine position teleport (reaction-free q writes
    // are the phantom-force class — step.rs comment); deep-violation
    // backstop only
    if (s.hit[j]) {
      const T over = d.q[j] < lo ? lo - d.q[j] : d.q[j] - hi;
      if (over > T(LIMIT_TELEPORT_BACKSTOP)) d.q[j] = tclamp(d.q[j], lo, hi);
    }
  }
  for (int pass = 0; pass < CLAMP_PASSES; pass++) {
    bool changed = false;
    for (int j = 0; j < NQ; j++) {
      const int vj = VOFF + j;
      const T cap = T(OMEGA_CAP[j]);
      // v14: at a stop the target is a Baumgarte recovery toward the
      // bound with a SLOP_LIMIT dead-band (mirrors step.rs) — the
      // impulse restores position reciprocally; a parked joint inside
      // the slop fires nothing
      T target;
      if (s.hit[j]) {
        const T lo = T(LIMIT_LO[j]), hi = T(LIMIT_HI[j]);
        const T over = d.q[j] < lo ? lo - d.q[j] : hi - d.q[j];
        const T rec = fabs(over) - T(SLOP_LIMIT);
        target = rec <= T(0)
                     ? T(0)
                     : tclamp(T(BETA_LIMIT) * (over < T(0) ? -rec : rec) / T(H_SUB),
                              -cap, cap);
      } else {
        target = tclamp(d.v[vj], -cap, cap);
      }
      if (target != d.v[vj]) {
        ensure_ecol(d, s, j);
        const T* col = s.ecol[j];
        const T lam = (target - d.v[vj]) / col[vj];
        for (int i = 0; i < NV; i++) d.v[i] += col[i] * lam;
        d.v[vj] = target;  // exact — the clamp is a hard bound
        changed = true;
      }
    }
    if (!changed) break;
  }
  // (env v14: no direct-write guarantee pass — mirror of step.rs.)
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
  T* __restrict__ effort;
  int* __restrict__ pd_hits;  // f32 pivot-floor engagements (§20e)
  int* __restrict__ env_hits;  // pre-integration root-envelope hits (§23)
  // env v11 FALL terminations (unilateral-contact spec §4), counted per
  // (env, TICK) exactly as lib.rs `fall_hits` is — the root-escape
  // envelope's twin, popped by the wrapper and reported by the trainer.
  // Written in contact mode only; stays 0 in weld mode, where the feet
  // cannot let go.
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
  // Body::ground_imp mirror. Written by the step tick in contact mode,
  // zeroed by reset_env, read back by write_obs in EVERY mode (weld
  // mode never writes, so the reset zeros stand — lanes stay 0).
  T* __restrict__ ground_imp;
  int64_t* __restrict__ chart_idx;
  // chart pool (P, C), read-only
  const T *__restrict__ p_hit, *__restrict__ p_speed, *__restrict__ p_center,
      *__restrict__ p_dvec, *__restrict__ p_rot, *__restrict__ p_good_hy,
      *__restrict__ p_bad_hy;
  const bool *__restrict__ p_valid, *__restrict__ p_is_right,
      *__restrict__ p_any_dir, *__restrict__ p_is_bomb,
      *__restrict__ p_is_head, *__restrict__ p_is_link,
      *__restrict__ p_show_dir, *__restrict__ p_free_pre,
      *__restrict__ p_free_post;
  const int64_t* __restrict__ p_n_notes;
  const T *__restrict__ p_last_hit, *__restrict__ p_tail,
      *__restrict__ p_min_speed;
  // env v17 wall pool (P, W), read-only, indexed by the SAME chart_idx
  const T *__restrict__ p_wall_t, *__restrict__ p_wall_cx,
      *__restrict__ p_wall_cy, *__restrict__ p_wall_w,
      *__restrict__ p_wall_h, *__restrict__ p_wall_len,
      *__restrict__ p_wall_speed;
  const bool* __restrict__ p_wall_valid;
  const int64_t* __restrict__ p_n_walls;
  // env v17 per-env wall state: passed latches (E, W), passed-prefix
  // cursor, and the head-in-wall tick counter the wrapper reports
  bool* __restrict__ wall_passed;
  int* __restrict__ first_wall;
  int* __restrict__ walls_in;
  // was the head inside a wall LAST tick — entry-edge state for the
  // once-per-entry WALL_PEN (lib.rs was_in_wall mirror)
  bool* __restrict__ in_wall_prev;
  int W;  // wall pool second dim
  // io
  const int64_t *__restrict__ actions, *__restrict__ cand;
  T* __restrict__ ret_buf;
  int* __restrict__ ret_total;
  float* __restrict__ obs;
  T* __restrict__ reward;
  bool* __restrict__ done;
  int N, C, max_steps, cap, mode;  // mode: 0 step, 1 reset-all, 2 refresh
  T effort_coef;
  T fall_const;   // v18 (sole mode): flat -fall_const fall reward (lib.rs mirror)
  // v28 probe (2026-08-20): obs z-lane upper clamp (m), RUNTIME so one
  // binary serves every era (old ckpts bench at 12; the rust twin
  // reads BEAT_Z_CLIP_HI). Z_CLIP_HI stays as the default.
  T z_clip_hi;
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
  // env v23: per-tick activation price on the commanded Σ|gear frac|
  // over the 29 joints (lib.rs mirror). Velocity-blind by design:
  // effort_coef prices speed (work), this prices force. 0 = off.
  T iso_coef;
  // env v22: previous tick's total hinged foot deviation, per env
  // (zeroed by reset_env — resets start on-target).
  T* __restrict__ foot_pot;
  // env v21: reset_env needs the runtime mode flag (the kernel's
  // FOOT_CONTACT template parameter is not visible there) to gate the
  // wide stance + ground fit.
  bool foot_contact;
  // env v20 start-pose jitter (TRAINING-ONLY, no lib.rs mirror — the
  // push/start_jitter precedent): per-env U(-1,1) draws for the 12 leg
  // joints, refilled by the wrapper every launch; consumed by reset_env
  // when start_pose_jitter > 0 (radians scale).
  const T* __restrict__ pose_jit;
  T start_pose_jitter;
  int gears;
  bool wide_acc;
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
  // env v21 (contact mode only): wide stance, then OPTIONAL v20
  // start-pose jitter CENTERED on the stance (12 leg joints, §1.4
  // indices 17..28; one-way knees take |draw|), clamped to the joint
  // ranges. ALWAYS finish with the exact no-clip fix: FK the pose,
  // transform the 8 sole corners (contact spec §2 GROUND_CORNERS),
  // and drop/lift the root so the lowest corner sits exactly on the
  // ground plane (y = 0) — the stance alone lifts the ankles ~8.5 mm.
  if (P.foot_contact) {
    d.q[19] = T(-Q_START_HIP_ROLL);  // l_hip_roll
    d.q[22] = T(Q_START_HIP_ROLL);   // l_ankle_roll (counter)
    d.q[25] = T(Q_START_HIP_ROLL);   // r_hip_roll
    d.q[28] = T(-Q_START_HIP_ROLL);  // r_ankle_roll (counter)
    if (P.start_pose_jitter > T(0)) {
      for (int k = 0; k < 12; k++) {
        const int j = 17 + k;
        T jit = P.pose_jit[e * 12 + k] * P.start_pose_jitter;
        if (j == 20 || j == 26) jit = fabs(jit);
        d.q[j] = tclamp(d.q[j] + jit, T(LIMIT_LO[j]), T(LIMIT_HI[j]));
      }
    }
    fk_all(d);
    T min_y = T(1e9);
    for (int f = 0; f < 2; f++) {
      const int fr = GROUND_FRAMES[f];
      for (int c = 0; c < CORNERS_PER_FOOT; c++) {
        T y = d.pos[fr][1];
        for (int k = 0; k < 3; k++)
          y += d.r[fr][1][k] * T(GROUND_CORNERS[c][k]);
        min_y = tmin(min_y, y);
      }
    }
    d.p[1] -= min_y;
  }
  store_state(d, P, e);
  refresh_fk(d, P, e);
  // env v24: seed the finite-diff base with the reset pose's §8
  // points, velocities zero (lib.rs Body reset mirror)
  {
    const int bp_f[N_BODY_PTS] = {F_L_ELBOW, F_R_ELBOW, F_L_SHANK,
                                  F_R_SHANK, F_L_FOOT, F_R_FOOT};
    for (int i = 0; i < N_BODY_PTS; i++)
      for (int k = 0; k < 3; k++) {
        const size_t idx = ((size_t)e * N_BODY_PTS + i) * 3 + k;
        P.body_pt[idx] = d.pos[bp_f[i]][k];
        P.body_pt_vel[idx] = T(0);
      }
  }
  for (int s = 0; s < 2; s++) {
    for (int k = 0; k < 3; k++) P.wrist_vel[(e * 2 + s) * 3 + k] = T(0);
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
  P.effort[e] = T(0);
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
  for (int i = 0; i < P.C; i++) P.status[e * P.C + i] = 0;
  // env v17 wall state
  P.first_wall[e] = 0;
  P.walls_in[e] = 0;
  P.in_wall_prev[e] = false;
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
template <typename T>
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
      const T hi = T(LIMIT_HI[j]);
      obs[lane] = (float)((T(2) * qj - hi) / hi);
      lane += 1;
    } else {
      const T lo = T(LIMIT_LO[j]), hi = T(LIMIT_HI[j]);
      obs[lane] = (float)(qj / tmax(fabs(lo), fabs(hi)));
      lane += 1;
    }
  }
  for (int j = 0; j < NQ; j++)
    obs[OFF_QD + j] = (float)(d.v[VOFF + j] / T(OMEGA_CAP[j]));
  // hand derived blocks
  const int bases[2] = {OFF_HAND_L, OFF_HAND_R};
  for (int s = 0; s < 2; s++) {
    const int base = bases[s];
    const int wb = (e * 2 + s) * 3;
    obs[base] = (float)(P.wrist[wb] / T(X_NORM));
    obs[base + 1] = (float)((P.wrist[wb + 1] - T(Y_CENTER)) / T(Y_NORM));
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
  const T time_o = T(P.tick[e]) * T(DT);
  // v15 head-anchored track (contact spec §10)
  const T z_off = P.head[e * 3 + 2] - T(PLANE_Z0);
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
    const int64_t cb = cc * C + i;
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
    const int64_t wb = cc * P.W + wi;
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
      const int64_t wb = cc * P.W + sel_idx[slot];
      const T zf = tclamp(
          (P.p_wall_t[wb] - time_o) * P.p_wall_speed[wb]
              - T(WALL_FRONT_OFF) + z_off,
          T(Z_CLIP_LO), P.z_clip_hi);
      nb2[0] = (float)(P.p_wall_cx[wb] / T(X_NORM));
      nb2[1] = (float)((P.p_wall_cy[wb] - T(Y_CENTER)) / T(Y_NORM));
      nb2[2] = (float)(zf / T(Z_SCALE));
      nb2[3] = (float)(-P.p_wall_speed[wb] / T(Z_SCALE));
      // lanes 4..13 (color/dir/kind/free) stay zero
      nb2[13] = 1.f;  // valid
      nb2[14] = 1.f;  // is_wall
      nb2[15] = (float)(P.p_wall_w[wb] / T(X_NORM));
      nb2[16] = (float)(P.p_wall_h[wb] / T(Y_NORM));
      nb2[17] = (float)(P.p_wall_len[wb] / T(Z_SCALE));
    } else {
      const int64_t cb = cc * C + sel_idx[slot];
      const T z = tclamp((P.p_hit[cb] - time_o) * P.p_speed[cb] + z_off,
                         T(Z_CLIP_LO), P.z_clip_hi);
      const bool bomb = P.p_is_bomb[cb];
      nb2[0] = (float)(P.p_center[cb * 2] / T(X_NORM));
      nb2[1] = (float)((P.p_center[cb * 2 + 1] - T(Y_CENTER)) / T(Y_NORM));
      nb2[2] = (float)(z / T(Z_SCALE));
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
      nb2[13] = 1.f;
      // lanes 14..17 (is_wall, w/h/l) stay zero
    }
    obs[OFF_NOTE_MASK + slot] = 1.f;
  }

  // contact spec §7: per-foot ground-contact lanes [L_heel, L_toe,
  // R_heel, R_toe]; the buffer is all-zero under welds
  for (int k = 0; k < N_GROUND_IMP_LANES; k++)
    obs[OFF_GROUND_IMP + k] = (float)(
        P.ground_imp[(size_t)e * N_GROUND_IMP_LANES + k] /
        T(GROUND_IMP_NORM));

  // contact spec §8 (env v13): elbow/knee/foot world positions from
  // d.pos — refresh_fk ran fk_all in every mode before this call
  {
    const int bp_f[N_BODY_PTS] = {F_L_ELBOW, F_R_ELBOW, F_L_SHANK,
                                  F_R_SHANK, F_L_FOOT, F_R_FOOT};
    const double bp_y0[N_BODY_PTS] = {Y_CENTER, Y_CENTER, KNEE_Y0,
                                      KNEE_Y0, FOOT_Y0, FOOT_Y0};
    for (int i = 0; i < N_BODY_PTS; i++) {
      const int base = OFF_BODY_PTS + 3 * i;
      obs[base] = (float)(d.pos[bp_f[i]][0] / T(X_NORM));
      obs[base + 1] = (float)((d.pos[bp_f[i]][1] - T(bp_y0[i])) / T(Y_NORM));
      obs[base + 2] = (float)((d.pos[bp_f[i]][2] - T(PLANE_Z0)) / T(X_NORM));
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
}

// One 30 Hz control tick + judgment + bookkeeping (lib.rs step_inner),
// entirely per-thread.
template <typename T, bool FC>
__device__ void step_env(Dyn<T>& __restrict__ d, Solver<T, FC>& __restrict__ s,
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
  // commanded activation Σ|frac| in double, matching lib.rs' f64 sum
  T tau[NQ];
  const double* gr = gear_row(P.gears);
  double act = 0.0;
  for (int j = 0; j < NQ; j++) {
    const int a = (int)P.actions[e * NQ + j];
    const double frac = a < 7 ? gr[a] : 0.0;
    tau[j] = T(frac) * T(TAU_MAX[j]);
    act += fabs(frac);
  }

  // 8 substeps, accumulating positive work + contact aggregation
  load_state(d, P, e);
  d.pd_bad = false;
  d.env_bad = false;
  d.env_n = 0;
  bool arm_l = false, arm_r = false;
  if (P.want_contacts)
    for (int pr = 0; pr < NUM_PAIRS; pr++)
      P.contact_imp[e * NUM_PAIRS + pr] = T(0);
  T effort = T(0);
  // env v11 FALL (contact spec §4): per-TICK on the end-of-tick root
  // height since 2026-08-20 (user — consistent with the tilt check; a
  // mid-tick dip that recovers within the tick no longer terminates).
  // Structurally false (and folded away) in weld mode.
  bool fell = false;
  bool tilted = false;
  // contact spec §7: tick-summed ground normal impulses per obs bucket
  // [L_heel, L_toe, R_heel, R_toe] (lib.rs Body::ground_imp). Dead code
  // in weld mode — no reader, so it folds away.
  T gimp[4] = {T(0), T(0), T(0), T(0)};
  for (int sub = 0; sub < N_SUB; sub++) {
    substep(d, s, tau);
    if constexpr (FC)
      for (int gi = 0; gi < s.ng; gi++)
        gimp[(s.g_bucket >> (2 * gi)) & 3u] +=
            s.g_lam[gi * GROUND_ROWS_PER_CORNER];
    // SPEC-v10 §4 effort: positive work at the END-of-substep velocity
    for (int j = 0; j < NQ; j++) {
      const T pw = tau[j] * d.v[VOFF + j];
      if (pw > T(0)) effort += pw * T(H_SUB);
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
  if constexpr (FC) {
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
  if constexpr (FC) {
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
  if constexpr (FC)
    for (int k = 0; k < N_GROUND_IMP_LANES; k++)
      P.ground_imp[e * N_GROUND_IMP_LANES + k] = gimp[k];
  P.effort[e] = effort;
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
    }
  // env v24: body-point world velocities — finite diff against the
  // last tick's stored positions, then advance the base (lib.rs
  // Body::tick mirror; d.pos holds this tick's refreshed FK)
  {
    const int bp_f[N_BODY_PTS] = {F_L_ELBOW, F_R_ELBOW, F_L_SHANK,
                                  F_R_SHANK, F_L_FOOT, F_R_FOOT};
    for (int i = 0; i < N_BODY_PTS; i++)
      for (int k = 0; k < 3; k++) {
        const size_t idx = ((size_t)e * N_BODY_PTS + i) * 3 + k;
        const T np = d.pos[bp_f[i]][k];
        P.body_pt_vel[idx] = (np - P.body_pt[idx]) / T(DT);
        P.body_pt[idx] = np;
      }
  }

  T reward = T(0);
  reward -= P.effort_coef * effort;
  reward -= P.iso_coef * T(act);
  // FOOT-LIFT penalty (2026-08-20): flat FOOT_LIFT_PEN the tick a
  // foot leaves the ground — edge on the tick-summed ground normal
  // impulse (gimp buckets [L_heel, L_toe, R_heel, R_toe]), per foot.
  // lib.rs mirror; folds away in weld mode (feet cannot lift).
  if constexpr (FC) {
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

  // swing chain update
  T deltas[2];
  bool continued[2];
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
        P.tr_active[b] = false;
      }
    }
  }

  const T time_new = T(P.tick[e] + 1) * T(DT);

  // both sabers' swept surfaces this tick
  T t0s[2][3], t1s[2][3];
  for (int h = 0; h < 2; h++)
    for (int k = 0; k < 3; k++) {
      t0s[h][k] = prev_w[h][k] + T(SABER_LEN) * prev_d[h][k];
      t1s[h][k] = new_w[h][k] + T(SABER_LEN) * new_d[h][k];
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
  int n_new[2] = {0, 0};

  // env v19 flail telemetry: on ticks whose next scoreable note is
  // > FLAIL_LEAD_S out, blade arc from hands with no live
  // follow-through is swinging that cannot earn. Runs BEFORE this
  // tick's cuts (a cut this tick implies a note inside the lead
  // window anyway). Pure accumulation — no reward/obs/termination
  // effect; popped by the wrapper.
  {
    T next_gap = T(2) * T(FLAIL_LEAD_S);
    for (int i = P.first_pending[e]; i < nn; i++) {
      if (P.status[e * C + i] != 0 || !P.p_valid[c0 * C + i]) continue;
      next_gap = P.p_hit[c0 * C + i] - time_new;
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

  for (int i = P.first_pending[e]; i < nn; i++) {
    if (P.status[e * C + i] != 0) continue;
    const int64_t cb = c0 * C + i;
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
        reward += T(REWARD_MISS);
      }
      continue;
    }
    if (z > reach_z) continue;
    const T center[3] = {P.p_center[cb * 2], P.p_center[cb * 2 + 1], z};

    if (bomb) {
      for (int h = 0; h < 2; h++) {
        if (sweep_dist(prev_w[h], t0s[h], t1s[h], new_w[h], center)
            <= T(BOMB_RADIUS)) {
          P.status[e * C + i] = 2;
          P.bombs[e] += 1;
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
    if (sweep_hits_box(prev_w[m], t0s[m], t1s[m], new_w[m], gc, good_h,
                       frame)) {
      // contact point: closest-passing blade sample; strict < like Rust
      T d_seg = std::numeric_limits<T>::infinity();
      T s_star = T(0);
      for (int jj = 1; jj <= N_BLADE_SAMPLES; jj++) {
        const T rr = (T(jj) / T(N_BLADE_SAMPLES)) * T(SABER_LEN);
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
        tip_move[k] = new_w[m][k] + T(SABER_LEN) * new_d[m][k]
                      - prev_w[m][k] - T(SABER_LEN) * prev_d[m][k];
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
      if (code == 0) {
        P.status[e * C + i] = 1;
        P.good[e] += 1;
        T pts, pre_pts = T(0), acc_pts = T(0);
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
              P.wide_acc
                  ? T(2) * T(PTS_ACC)
                        * (T(1) - tclamp(d_min / (T(2) * T(ACC_NORM)),
                                         T(0), T(1)))
                  : round_pts(T(PTS_ACC)
                              * (T(1)
                                 - tclamp(d_min / T(ACC_NORM), T(0),
                                          T(1))));
          pts = pre + acc;
          pre_pts = pre;
          acc_pts = acc;
          if (is_head) {
            // chain heads: no follow-through (max 85)
          } else if (P.p_free_post[cb]) {
            pts += T(PTS_POST);
            P.score_post[e] += T(PTS_POST);
          } else {
            n_new[m] += 1;  // spawn a follow-through tracker
          }
        }
        reward += pts / T(SCORE_MAX);
        P.score[e] += pts;
        P.score_pre[e] += pre_pts;
        P.score_acc[e] += acc_pts;
        P.acc_pts[e * 2 + m] += pts;
        P.acc_cuts[e * 2 + m] += 1;
      } else {
        P.status[e * C + i] = 2;
        P.bad[e] += 1;
        reward += T(REWARD_BAD);
      }
    } else {
      const T bad_h[3] = {T(BAD_HALF), P.p_bad_hy[cb], T(BAD_HALF)};
      if (sweep_hits_box(prev_w[1 - m], t0s[1 - m], t1s[1 - m],
                         new_w[1 - m], center, bad_h, frame)) {
        P.status[e * C + i] = 2;
        P.bad[e] += 1;
        reward += T(REWARD_BAD);
      }
    }
  }

  // spawn aggregated trackers (same-tick spawns share (arc, age))
  for (int h = 0; h < 2; h++) {
    if (n_new[h] > 0) {
      const int ptr = P.tr_ptr[e * 2 + h];
      const int b = (e * 2 + h) * N_TRACKERS + ptr;
      P.tr_acc[b] = T(0);
      P.tr_age[b] = 0;
      P.tr_cnt[b] = n_new[h];
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
    const T hx2 = P.head[e * 3], hy2 = P.head[e * 3 + 1];
    bool in_wall = false;
    for (int wi = P.first_wall[e]; wi < nw; wi++) {
      const int64_t wb = c0 * P.W + wi;
      if (!P.p_wall_valid[wb] || P.wall_passed[e * P.W + wi]) continue;
      const T dzf = (P.p_wall_t[wb] - time_new) * P.p_wall_speed[wb]
          - T(PLANE_Z0) - T(WALL_FRONT_OFF);
      const T len = P.p_wall_len[wb];
      if (dzf + len < T(0)) {
        P.wall_passed[e * P.W + wi] = true;  // back face behind the head
        continue;
      }
      if (dzf <= T(0)
          && fabs(hx2 - P.p_wall_cx[wb]) <= P.p_wall_w[wb] * T(0.5)
          && fabs(hy2 - P.p_wall_cy[wb]) <= P.p_wall_h[wb] * T(0.5))
        in_wall = true;
    }
    if (in_wall) {
      if (!P.in_wall_prev[e]) reward += T(WALL_PEN);
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
        P.tr_active[b] = false;
      }
    }
  }

  // contact spec §7 survival gradient — MIRROR of lib.rs: the alive
  // bonus lands BEFORE the terminal overrides, so a fall/escape tick
  // never collects it.
  if constexpr (FC) reward += T(ALIVE_BONUS);
  if (d.env_bad) reward = T(0);
  // v11.1 fall pricing (contact spec §4, 2026-08-09) — MIRROR of
  // lib.rs's fell block: REWARD_MISS x pending scoreable notes, full
  // 0..nn scan in chart order, statuses untouched. Replaces the
  // reward-0 zeroing whose bounded penalty made fast death the 8h
  // run's converged optimum. The p_valid gate is not load-bearing
  // (slots 0..nn are exactly the valid ones by pool construction) —
  // it mirrors the note loop's own i<nn convention so the two scans
  // can never diverge if that invariant ever changes.
  if constexpr (FC) {
    if (fell || tilted) {
      // env v18 (sole mode) — MIRROR of lib.rs: flat terminal reward;
      // tilt terminations (2026-08-20) priced identically
      reward = -P.fall_const;
    }
  } else {
    if (fell) reward = T(0);  // fell/tilted constant-false here; folded
  }
  const T ep = P.ep_return[e] + reward;
  P.ep_return[e] = ep;
  P.reward[e] = reward;
  P.done[e] = done;
  if (done) {
    const int pos = atomicAdd(P.ret_total, 1);
    P.ret_buf[pos % P.cap] = ep;
  }
  if (done && P.auto_reset) reset_env(d, P, e, P.cand[e]);
}

// FC = env v11 foot_mode (see GROUND_CORNERS): FALSE is the v10 kernel,
// unchanged; TRUE swaps the 12 weld rows for the ground block and arms
// the fall envelope. Two instantiations per dtype, selected once on the
// host — the flag is baked per env instance, so it is launch-constant.
template <typename T, bool FC>
// The `, 1` PINS the register budget at 255 — see the THREADS note: it
// is not an occupancy hint, it is what stops ptxas's free-choice
// heuristic from flipping to 64 regs + double the spills.
__global__ void __launch_bounds__(THREADS, BEAT_MIN_BLOCKS)
step_kernel(Sim<T> P) {
  // R12 (2026-08-21): warp-coherence env permutation. The grid walks
  // env_order — identity by default; under BEAT_ENV_ORDER=1 the host
  // sorts it by (chart, first_pending) each tick so warp-mates share
  // note windows (audit 3.3: identical-envs header ran 3.5x production
  // traffic). Per-env work and every P.*[e] address are unchanged —
  // a bijection on thread->env assignment, bitwise-identical per env.
  const int gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid >= P.N) return;
  const int e = P.env_order[gid];
  // thread-local scratch (local memory; per-thread interleaved =
  // coalesced across the warp for same-index access)
  Dyn<T> d;
  Solver<T, FC> s;
  if (P.mode == 1) {
    reset_env(d, P, e, P.cand[e]);
  } else if (P.mode == 2) {
    // set_state refresh (RawBeatSim.set_joints semantics): FK caches
    // from the freshly written state; finite-diff wrist velocity reset;
    // swing accumulators untouched
    load_state(d, P, e);
    refresh_fk(d, P, e);
    for (int k = 0; k < 6; k++) P.wrist_vel[e * 6 + k] = T(0);
    // env v24: set_state semantics — re-seed the finite-diff base at
    // the refreshed pose, velocities zero (lib.rs set_state mirror)
    {
      const int bp_f[N_BODY_PTS] = {F_L_ELBOW, F_R_ELBOW, F_L_SHANK,
                                    F_R_SHANK, F_L_FOOT, F_R_FOOT};
      for (int i = 0; i < N_BODY_PTS; i++)
        for (int k = 0; k < 3; k++) {
          const size_t idx = ((size_t)e * N_BODY_PTS + i) * 3 + k;
          P.body_pt[idx] = d.pos[bp_f[i]][k];
          P.body_pt_vel[idx] = T(0);
        }
    }
  } else {
    d.pd_bad = false;
    d.env_bad = false;
    d.env_n = 0;
    step_env(d, s, P, e);
  }
  write_obs(d, P, e);
}

// TOMBSTONE (2026-08-15) — the warp / sub-warp-per-env kernel family
// lived here (step_kernel_warp, step_kernel_sw, the *_warp / *_sw device
// forks, SubWarpGeom + the sw_* shuffle helpers, chol_solve_cn, and the
// kernel_variant selection branch in step() below). DELETED: measured
// slower at EVERY lane width W — 169 (W=2) / 251 (W=4) / 338 (W=8) ms
// per tick at N=16,384 against the legacy kernel's 121.8. Redundant
// per-lane execution and its local traffic (plus spills at W>=4) beat
// the residency arithmetic at every rung, so there is no rung to climb.
// Full record: design/WARP_ENV_SOLVER_CORE.md, WARP_RESCUE_LEGACY_SIDE.md,
// WARP_ENV_RECONCILIATION.md; code in git history at b02e7ce. The one
// back-port that survived is the right-looking Cholesky, kept above as
// the BEAT_CHOL_RL compile-time option (default OFF).

// tensor-list slots (order is a contract with cuda_env.py _tensors)
enum {
  TP = 0, TQUAT, TQ, TV,
  TWRIST, TBLADE, TWVEL, THEAD, TSHZ,
  TSWING, TAXIS, TAXV,
  TTRACC, TTRAGE, TTRCNT, TTRACT, TTRPTR,
  TTICK, TSTATUS, TFIRSTP, TEPRET,
  TGOOD, TBAD, TMISS, TBOMBS, TSCORE, TSPRE, TSACC, TSPOST,
  TCIMP, TCPT, TCARM, TEFF, TPD, TENV, TFALL, TTILT, TLIFT, TFOOTDOWN,
  TFLAILR, TFLAILT,
  TACCP, TACCC, TGIMP, TCIDX,
  TPOSEJIT, TFOOTPOT, TBPPOS, TBPVEL,
  PHIT, PSPEED, PCENTER, PDVEC, PROT, PGHY, PBHY,
  PVALID, PISR, PANY, PBOMB, PHEAD, PLINK, PSHOW, PFPRE, PFPOST,
  PNNOTES, PLASTHIT, PTAIL, PMINSP,
  // env v17 wall pool (P, W) + per-env wall state
  PWALLT, PWALLCX, PWALLCY, PWALLW, PWALLH, PWALLLEN, PWALLSP,
  PWALLVALID, PNWALLS, TWALLPASS, TFIRSTW, TWALLSIN, TINWALLPREV,
  TENVORD,
  RBUF, RTOT, OOBS, OREW, ODONE, T_COUNT
};

void step(std::vector<at::Tensor> ts, at::Tensor actions, at::Tensor cand,
          int64_t mode, int64_t max_steps, double effort_coef,
          bool wide_acc, bool auto_reset, int64_t gears,
          bool want_contacts, bool foot_contact, double fall_const,
          double foot_z_coef, double foot_x_coef, double foot_hinge_r,
          double foot_shape_coef, double iso_coef,
          double start_pose_jitter, double z_clip_hi) {
  TORCH_CHECK((int)ts.size() == T_COUNT, "expected ", (int)T_COUNT,
              " state tensors, got ", ts.size());
  const auto& q = ts[TQ];
  TORCH_CHECK(q.is_cuda(), "state must live on CUDA");
  TORCH_CHECK(actions.scalar_type() == at::kLong && actions.is_contiguous());
  TORCH_CHECK(cand.scalar_type() == at::kLong && cand.is_contiguous());
  const int64_t N = q.size(0);
  const int64_t C = ts[TSTATUS].size(1);
  const int64_t cap = ts[RBUF].size(0);
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
    P.effort = ts[TEFF].data_ptr<scalar_t>();
    P.pd_hits = ts[TPD].data_ptr<int>();
    P.env_hits = ts[TENV].data_ptr<int>();
    P.fall_hits = ts[TFALL].data_ptr<int>();
    P.tilt_hits = ts[TTILT].data_ptr<int>();
    P.foot_lift_hits = ts[TLIFT].data_ptr<int>();
    P.foot_down_prev = ts[TFOOTDOWN].data_ptr<bool>();
    P.flail_rad = ts[TFLAILR].data_ptr<scalar_t>();
    P.flail_ticks = ts[TFLAILT].data_ptr<int>();
    P.acc_pts = ts[TACCP].data_ptr<scalar_t>();
    P.acc_cuts = ts[TACCC].data_ptr<int>();
    P.ground_imp = ts[TGIMP].data_ptr<scalar_t>();
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
    P.p_n_notes = ts[PNNOTES].data_ptr<int64_t>();
    P.p_last_hit = ts[PLASTHIT].data_ptr<scalar_t>();
    P.p_tail = ts[PTAIL].data_ptr<scalar_t>();
    P.p_min_speed = ts[PMINSP].data_ptr<scalar_t>();
    P.p_wall_t = ts[PWALLT].data_ptr<scalar_t>();
    P.p_wall_cx = ts[PWALLCX].data_ptr<scalar_t>();
    P.p_wall_cy = ts[PWALLCY].data_ptr<scalar_t>();
    P.p_wall_w = ts[PWALLW].data_ptr<scalar_t>();
    P.p_wall_h = ts[PWALLH].data_ptr<scalar_t>();
    P.p_wall_len = ts[PWALLLEN].data_ptr<scalar_t>();
    P.p_wall_speed = ts[PWALLSP].data_ptr<scalar_t>();
    P.p_wall_valid = ts[PWALLVALID].data_ptr<bool>();
    P.p_n_walls = ts[PNWALLS].data_ptr<int64_t>();
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
    P.effort_coef = (scalar_t)effort_coef;
    P.fall_const = (scalar_t)fall_const;
    P.foot_z_coef = (scalar_t)foot_z_coef;
    P.foot_x_coef = (scalar_t)foot_x_coef;
    P.foot_hinge_r = (scalar_t)foot_hinge_r;
    P.foot_shape_coef = (scalar_t)foot_shape_coef;
    P.iso_coef = (scalar_t)iso_coef;
    P.z_clip_hi = (scalar_t)z_clip_hi;
    P.foot_pot = ts[TFOOTPOT].data_ptr<scalar_t>();
    P.body_pt = ts[TBPPOS].data_ptr<scalar_t>();
    P.body_pt_vel = ts[TBPVEL].data_ptr<scalar_t>();
    P.foot_contact = foot_contact;
    P.pose_jit = ts[TPOSEJIT].data_ptr<scalar_t>();
    P.start_pose_jitter = (scalar_t)start_pose_jitter;
    P.gears = (int)gears;
    P.wide_acc = wide_acc;
    P.auto_reset = auto_reset;
    P.want_contacts = want_contacts;
    const unsigned blocks = (unsigned)((N + THREADS - 1) / THREADS);
    // env v11 foot_mode: a COMPILE-TIME switch (see GROUND_CORNERS), so
    // the choice is made here, once per launch, and the weld kernel is
    // the v10 one to the byte.
    if (foot_contact)
      step_kernel<scalar_t, true><<<blocks, THREADS, 0,
                                    at::cuda::getCurrentCUDAStream()>>>(P);
    else
      step_kernel<scalar_t, false><<<blocks, THREADS, 0,
                                     at::cuda::getCurrentCUDAStream()>>>(P);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  });
}

// Transcendental-agreement probe (Divergences §21 measurement): apply
// the named f64 op elementwise on the device so the parity suite can
// compare bitwise against the CRT libm (numpy/Rust). op: 0 sin, 1 cos,
// 2 acos, 3 sqrt, 4 recip.
__global__ void transcend_kernel(const double* x, double* y, int n, int op) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  const double v = x[i];
  y[i] = op == 0 ? sin(v)
         : op == 1 ? cos(v)
         : op == 2 ? acos(v)
         : op == 3 ? sqrt(v)
                   : 1.0 / v;
}

void transcend(at::Tensor x, at::Tensor y, int64_t op) {
  TORCH_CHECK(x.is_cuda() && x.scalar_type() == at::kDouble
              && x.is_contiguous());
  TORCH_CHECK(y.is_cuda() && y.scalar_type() == at::kDouble
              && y.is_contiguous() && y.numel() == x.numel());
  const int n = (int)x.numel();
  transcend_kernel<<<(unsigned)((n + 255) / 256), 256, 0,
                     at::cuda::getCurrentCUDAStream()>>>(
      x.data_ptr<double>(), y.data_ptr<double>(), n, (int)op);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("step", &step,
        "One v10 sim step (mode 0), reset-all (mode 1) or FK-cache "
        "refresh after a state write (mode 2) for every env; state "
        "tensors mutated in place, obs/reward/done written into their "
        "output slots. foot_contact selects env v11 foot_mode = 1 "
        "(unilateral ground contact) instead of the v10 foot welds");
  m.def("transcend", &transcend,
        "f64 transcendental probe: y = op(x) on the device (0 sin, "
        "1 cos, 2 acos, 3 sqrt, 4 recip)");
}
