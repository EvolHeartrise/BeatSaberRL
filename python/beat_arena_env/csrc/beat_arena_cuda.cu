// Beat Arena env v10 sim step as ONE CUDA kernel launch (P5).
//
// Full SPEC-v10 pipeline per env: 8 x substep (FK -> CRBA -> RNEA ->
// dense Cholesky -> constraint solve [12 x weld-block / contacts /
// limit rows / omega-cap pass, weld-null-space projection, COMPOSITE_EPS
// 1e-4] -> integrate -> step-11 clamp passes -> step-12 root envelope)
// + gear->tau, work-based effort, blade sweep/cut judgment, scoring,
// obs 340 — a semantics-exact transcription of rust/src/lib.rs +
// rust/src/dynamics/{model,fk,algo,collide,solver,step}.rs. The Rust
// crate remains THE reference implementation.
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
constexpr int N_NOTE_SLOTS = 16;
constexpr int NOTE_FEAT = 14;
// One WARP per block: the grid is N/32 blocks, so small batches spread
// over all 20 SMs. Throughput is INSENSITIVE to the block size (P5
// measurement, 128 vs 32: 216 vs 221 ms at N=1024) — consistent with
// the diagnosis below, and the reason occupancy is not the lever here.
//
// The kernel is SERIAL-LATENCY bound. Earlier notes called it "serial
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

// ---- obs layout (SPEC-v10 §5; mirrors constants.py) -------------------------
constexpr int OFF_ROOT_POS = 0;
constexpr int OFF_ROOT_ROT = 3;
constexpr int OFF_ROOT_VLIN = 9;
constexpr int OFF_ROOT_VANG = 12;
constexpr int OFF_Q = 15;
constexpr int OFF_QD = 46;
constexpr int OFF_HAND_L = 75;
constexpr int OFF_HAND_R = 85;
constexpr int OFF_HEAD = 95;
constexpr int OFF_CONTACT = 98;
constexpr int OFF_NOTES = 100;
constexpr int OFF_NOTE_MASK = OFF_NOTES + N_NOTE_SLOTS * NOTE_FEAT;  // 324
constexpr int OBS_DIM = OFF_NOTE_MASK + N_NOTE_SLOTS;                // 340
static_assert(OBS_DIM == 340, "SPEC-v10 §5: OBS_DIM must be 340");

// ---- solver constants (step.rs / solver.rs / collide.rs) --------------------
constexpr double DT = 1.0 / 30.0;
constexpr double H_SUB = 1.0 / 240.0;
constexpr int N_SUB = 8;
constexpr int CLAMP_PASSES = 8;
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
__device__ constexpr double F_MASS[NF] = {
    0, 0, 24.20, 5.20, 0, 0, 2.00, 1.20, 0, 0, 0.61,
    0, 0, 2.00, 1.20, 0, 0, 0.61,
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
    {0, 0, 0}, {0, 0, 0}, {0.000460, 0.000460, 0.000460},
    {0, 0, 0}, {0, 0, 0}, {0.029222, 0.029222, 0.001966},
    {0.013056, 0.013056, 0.000931},
    {0, 0, 0}, {0, 0, 0}, {0.000460, 0.000460, 0.000460},
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
    2.6179938779914944, 0.3490658503988659, 0.3490658503988659,
    2.0943951023931953, 0.7853981633974483, 0.7853981633974483,
    2.6179938779914944, 0.3490658503988659, 0.3490658503988659};
__device__ constexpr double TAU_MAX[NQ] = {
    800.00, 319.93, 500.00,
    113.84, 113.84, 19.45, 19.13, 1.06, 1.06, 1.06,
    113.84, 113.84, 19.45, 19.13, 1.06, 1.06, 1.06,
    484.68, 100.00, 300.00, 250.00, 180.00, 50.00,
    484.68, 100.00, 300.00, 250.00, 180.00, 50.00};
__device__ constexpr double OMEGA_CAP[NQ] = {
    16.0, 16.0, 16.0,
    16.0, 16.0, 16.0, 16.0, 40.0, 40.0, 40.0,
    16.0, 16.0, 16.0, 16.0, 40.0, 40.0, 40.0,
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
// {0..5} u {29..34}: 12 of NV = 35 columns. The three consumers that
// contract a weld row (A_ww assembly, the PGS weld dot, ensure_pcol's
// fused dot) iterate only those ranges; s.jw's zero fill and the
// full-length s.mw seed copy are UNCHANGED, since s.mw is dense.
constexpr int WJ_L0 = 23;
constexpr int WJ_R0 = 29;
static_assert(WJ_R0 == WJ_L0 + 6 && WJ_R0 + 6 == NV && N_WELD == 12,
              "K5: weld support layout is {0..5} u {L0..L0+5} / {R0..NV}");
__device__ constexpr double WELD_TARGET_P[2][3] = {
    {-0.10, 0.07, -0.65}, {0.10, 0.07, -0.65}};

// ---- env constants (lib.rs, verbatim) ---------------------------------------
constexpr double V_NORM = 6.0;
constexpr double V_MIN_CUT = 1.0;
constexpr double ACC_NORM = 0.3;
constexpr double COS_TOL = 0.5;
constexpr double Y_CENTER = 1.40;
constexpr double REWARD_BAD = -0.5;
constexpr double REWARD_MISS = -0.5;
constexpr double REWARD_BOMB = -1.5;
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

template <typename T>
struct Solver {
  // weld rows
  T jw[N_WELD][NV], mw[N_WELD][NV], bias_w[N_WELD];
  T aww[N_WELD * N_WELD], rw[N_WELD];  // aww: A_ww, then in-place L
  // contacts (compacted, pair-id order)
  int nc;
  int c_pair[NUM_PAIRS];
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
  T ecol[NQ][NV], pcol[NQ][NV], pdiag[NQ];
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
// d.m — chol_factor_n (in place: reads a[j*n+j], a[j*n+k] k<j,
// a[i*n+k] k<j<i, a[i*n+j] i>j; writes a[j*n+j], a[i*n+j] i>j),
// chol_solve_n / chol_solve_multi_n (l[i*n+k] k<i; l[k*n+i] k>i, i.e.
// row k > col i; l[i*n+i]), chol_fwd_unit_n (l[i*n+k] m<=k<i; diag) and
// chol_back{,2,4}_n (l[k*n+i] k>i; diag) — touches row >= col ONLY.
// Nothing in the kernel ever reads d.m above the diagonal, so this
// function no longer writes there: the NV*NV zero fill is narrowed to
// the 630 lower+diagonal entries (595 stores saved — and ~286 of the
// kept ones are genuinely load-bearing, entries no producer below
// touches), the symmetric ancestor pair writes one element at
// [max*NV+min], and the root-block mirror stores are gone. ~970 dead
// stores per substep.
template <typename T>
__device__ void crba(Dyn<T>& __restrict__ d) {
  for (int i = 0; i < NV; i++)
    for (int j = 0; j <= i; j++) d.m[i * NV + j] = T(0);
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
      int anc = F_PARENT[k];
      while (anc >= 0) {
        if (F_COORD[anc] >= 0) {
          const int ia = VOFF + F_COORD[anc];
          T rel2[3];
          sub3v(cvec, d.pos[anc], rel2);
          T r2xhl[3];
          cross3(rel2, hl, r2xhl);
          T tot2[3] = {ha[0] + r2xhl[0], ha[1] + r2xhl[1], ha[2] + r2xhl[2]};
          const T val = dot3(d.axw[anc], tot2);
          // symmetric pair -> lower entry only (F_COORD is not monotonic
          // in the frame index, so the order of ia vs j is a runtime fact)
          d.m[(ia > j ? ia : j) * NV + (ia > j ? j : ia)] = val;
        }
        anc = F_PARENT[anc];
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
        d.m[j * NV + t] = hl[t];
        d.m[j * NV + (3 + t)] = lr[t];
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
      d.m[i * NV + j] = i == j ? d.bm : T(0);
      d.m[(3 + i) * NV + (3 + j)] = rr[i][j];
    }
  const T sk[3][3] = {{T(0), -dd[2], dd[1]},
                      {dd[2], T(0), -dd[0]},
                      {-dd[1], dd[0], T(0)}};
  // (3+i, j) is strictly lower for every i, j in 0..2; the (j, 3+i)
  // mirror was dead.
  for (int i = 0; i < 3; i++)
    for (int j = 0; j < 3; j++) d.m[(3 + i) * NV + j] = d.bm * sk[i][j];
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
  for (int i = 0; i < NV; i++) d.c[i] = T(0);
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
// IN PLACE, one buffer (2026-08-03; supersedes the 2026-08-02 note that
// deliberately withheld __restrict__ here pending exactly this change).
// `a` enters as M and leaves as its lower Cholesky factor L. Legal by
// inspection of the recurrence, and bitwise identical to the old
// two-buffer form because it issues the SAME reads, the SAME writes, in
// the SAME order — only the destination address changes:
//   * column j reads a[j*n+j] (the M diagonal) as its FIRST statement
//     and writes a[j*n+j] = dj only after the k-loop — read-before-
//     write, and no later column ever re-reads a diagonal;
//   * the row loop reads a[i*n+j] as the first statement of iteration i
//     and writes exactly that one element as its last — read once,
//     immediately before its own overwrite;
//   * every other read is a[i*n+k] / a[j*n+k] with k < j, i.e. columns
//     already converted to L by earlier j, and no write at column j
//     touches a column < j.
// Callers: d.m (written only by crba) and s.aww (rewritten before each
// factor) — neither is read as M after its factor, so nothing needs the
// original matrix. Both are supplied LOWER TRIANGLE + DIAGONAL ONLY
// since K4c; this function reads and writes nothing above the diagonal.
// Frees Dyn::l (NV*NV) and Solver::lww (N_WELD*N_WELD) from the frame.
//
// The dead upper-triangle zeroing that used to run here (l[j*n+i],
// i > j — 595 stores per NV=35 factor + 66 per weld factor, per
// substep) is GONE: the only readers of a factor are this function
// (a[i*n+k], a[j*n+k] with k < j <= i) and chol_solve_n (a[i*n+k] k<i;
// a[k*n+i] k>i = row k, col i with k>i; a[i*n+i]) — strictly lower
// triangle plus diagonal. Nothing reads above it, so it needs no value.
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
// storage and zero extra loads (a[(j+1)*n+k] is already being loaded
// there as s0's operand) — and column j+1 then starts with its pivot in
// hand. Bitwise: identical expression, identical order, identical
// inputs; a[(j+1)*n+(j+1)] is the untouched M diagonal (column j writes
// only column j).
template <typename T>
__device__ void chol_factor_n(int n, T* a, bool* bad) {
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
    int i = j + 1;
    // Row j+1 is peeled (as the head of a 4-block when there is room,
    // scalar otherwise) so the next pivot rides along on its loads.
    T np = i < n ? a[i * n + i] : T(0);
    if (i + 3 < n) {
      const int r0 = i * n, r1 = r0 + n, r2 = r1 + n, r3 = r2 + n;
      T s0 = a[r0 + j], s1 = a[r1 + j], s2 = a[r2 + j], s3 = a[r3 + j];
      for (int k = 0; k < j; k++) {
        const T ljk = a[j * n + k];
        const T v0 = a[r0 + k];
        s0 -= v0 * ljk;
        s1 -= a[r1 + k] * ljk;
        s2 -= a[r2 + k] * ljk;
        s3 -= a[r3 + k] * ljk;
        np -= v0 * v0;
      }
      const T l0 = s0 / dj;
      a[r0 + j] = l0;
      a[r1 + j] = s1 / dj;
      a[r2 + j] = s2 / dj;
      a[r3 + j] = s3 / dj;
      np -= l0 * l0;
      i += 4;
    } else if (i < n) {
      const int r0 = i * n;
      T s = a[r0 + j];
      for (int k = 0; k < j; k++) {
        const T v0 = a[r0 + k];
        s -= v0 * a[j * n + k];
        np -= v0 * v0;
      }
      const T l0 = s / dj;
      a[r0 + j] = l0;
      np -= l0 * l0;
      i += 1;
    }
    for (; i + 3 < n; i += 4) {
      const int r0 = i * n, r1 = r0 + n, r2 = r1 + n, r3 = r2 + n;
      T s0 = a[r0 + j], s1 = a[r1 + j], s2 = a[r2 + j], s3 = a[r3 + j];
      for (int k = 0; k < j; k++) {
        const T ljk = a[j * n + k];
        s0 -= a[r0 + k] * ljk;
        s1 -= a[r1 + k] * ljk;
        s2 -= a[r2 + k] * ljk;
        s3 -= a[r3 + k] * ljk;
      }
      a[r0 + j] = s0 / dj;
      a[r1 + j] = s1 / dj;
      a[r2 + j] = s2 / dj;
      a[r3 + j] = s3 / dj;
    }
    for (; i < n; i++) {
      T s = a[i * n + j];
      for (int k = 0; k < j; k++) s -= a[i * n + k] * a[j * n + k];
      a[i * n + j] = s / dj;
    }
    dpiv = np;
  }
}

// Triangular solve. l and x are always distinct objects at every call
// site (d.m, holding the in-place factor, vs {d.rhs, s.mw[w],
// s.c_minv[ci], s.ecol[j]}; s.aww vs s.rw), and x is the only thing
// written — so restrict here is the
// difference between keeping l's row in registers across the x[i]
// store and re-loading it every iteration.
template <typename T>
__device__ void chol_solve_n(int n, const T* __restrict__ l,
                             T* __restrict__ x) {
  for (int i = 0; i < n; i++) {
    T s = x[i];
    for (int k = 0; k < i; k++) s -= l[i * n + k] * x[k];
    x[i] = s / l[i * n + i];
  }
  for (int i = n - 1; i >= 0; i--) {
    T s = x[i];
    for (int k = i + 1; k < n; k++) s -= l[k * n + i] * x[k];
    x[i] = s / l[i * n + i];
  }
}

// BATCHED multi-RHS solve: W independent right-hand sides, laid out as
// W consecutive rows of stride `n` (s.mw / s.c_minv are exactly that),
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
// W-fold; the factor element l[i*n+k] is also loaded once per (i, k)
// instead of once per (i, k, w). The accumulators are a fixed-size local
// array indexed only by fully unrolled compile-time w, so they live in
// registers (W f32 = W*4 B).
template <int W, typename T>
__device__ void chol_solve_multi_n(int n, const T* __restrict__ l,
                                   T* __restrict__ x) {
  for (int i = 0; i < n; i++) {
    T s[W];
#pragma unroll
    for (int w = 0; w < W; w++) s[w] = x[w * n + i];
    for (int k = 0; k < i; k++) {
      const T lik = l[i * n + k];
#pragma unroll
      for (int w = 0; w < W; w++) s[w] -= lik * x[w * n + k];
    }
    const T dg = l[i * n + i];
#pragma unroll
    for (int w = 0; w < W; w++) x[w * n + i] = s[w] / dg;
  }
  for (int i = n - 1; i >= 0; i--) {
    T s[W];
#pragma unroll
    for (int w = 0; w < W; w++) s[w] = x[w * n + i];
    for (int k = i + 1; k < n; k++) {
      const T lki = l[k * n + i];
#pragma unroll
      for (int w = 0; w < W; w++) s[w] -= lki * x[w * n + k];
    }
    const T dg = l[i * n + i];
#pragma unroll
    for (int w = 0; w < W; w++) x[w * n + i] = s[w] / dg;
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
//     +0.0 / l[i*n+i] = +0.0 — the value already in the buffer;
//   * by induction on i < m: s starts at x[i] = +0.0 and every term
//     l[i*n+k] * x[k] has x[k] = +0.0 (k < i < m, established by the
//     induction), so the product is +-0.0 and s stays exactly +0.0
//     (+0.0 - +0.0 = +0.0 and +0.0 - -0.0 = +0.0 under RN; FMA
//     contraction does not change either);
//   * at i == m the same k < m terms leave s = x[m] = 1 untouched, so
//     the k-loop may start at k = m (zero iterations here) and the
//     divide is the identical 1 / l[m*n+m];
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
  for (int i = m; i < n; i++) {
    T s = x[i];
    for (int k = m; k < i; k++) s -= l[i * n + k] * x[k];
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
//     term l[i*n+k] * x_t[k] has k < i < m_t, so x_t[k] is +0.0 by
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
  for (int i = m0; i < n; i++) {
    T s0 = x0[i], s1 = x1[i], s2 = x2[i], s3 = x3[i];
    for (int k = m0; k < i; k++) {
      const T lik = l[i * n + k];
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
  for (int i = m0; i < n; i++) {
    T s0 = x0[i], s1 = x1[i];
    for (int k = m0; k < i; k++) {
      const T lik = l[i * n + k];
      s0 -= lik * x0[k];
      s1 -= lik * x1[k];
    }
    const T dg = l[i * n + i];
    x0[i] = s0 / dg;
    x1[i] = s1 / dg;
  }
}

template <typename T>
__device__ void chol_back_n(int n, const T* __restrict__ l,
                            T* __restrict__ x) {
  for (int i = n - 1; i >= 0; i--) {
    T s = x[i];
    for (int k = i + 1; k < n; k++) s -= l[k * n + i] * x[k];
    x[i] = s / l[i * n + i];
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
    T s0 = x0[i], s1 = x1[i], s2 = x2[i], s3 = x3[i];
    for (int k = i + 1; k < n; k++) {
      const T lki = l[k * n + i];
      s0 -= lki * x0[k];
      s1 -= lki * x1[k];
      s2 -= lki * x2[k];
      s3 -= lki * x3[k];
    }
    const T dg = l[i * n + i];
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
    T s0 = x0[i], s1 = x1[i];
    for (int k = i + 1; k < n; k++) {
      const T lki = l[k * n + i];
      s0 -= lki * x0[k];
      s1 -= lki * x1[k];
    }
    const T dg = l[i * n + i];
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

template <typename T>
__device__ void angular_jacobian_acc(const Dyn<T>& d, int frame, int axis_t,
                                     T sign, T* __restrict__ row) {
  int k = frame;
  while (k >= 0) {
    if (F_COORD[k] >= 0) row[VOFF + F_COORD[k]] += sign * d.axw[k][axis_t];
    k = F_PARENT[k];
  }
  row[3 + axis_t] += sign;
}

// RHS seed: the full-length zero fill is load-bearing for
// chol_fwd_unit_n / chol_back_n — see chol_solve_unit_n's header (the
// skipped forward rows keep these +0.0 for the dense back substitution
// to read).
template <typename T>
__device__ __forceinline__ void ecol_seed(Solver<T>& __restrict__ s, int j) {
  T* col = s.ecol[j];
  for (int i = 0; i < NV; i++) col[i] = T(0);
  col[VOFF + j] = T(1);
}

// Lazily compute the plain M^-1 e_j column (shared by step-8 cap/limit
// rows and the step-11 clamps — same expression tree as Rust's three
// separate caches).
template <typename T>
__device__ void ensure_ecol(Dyn<T>& __restrict__ d, Solver<T>& __restrict__ s,
                            int j) {
  if (s.ecomp[j]) return;
  ecol_seed(s, j);
  chol_solve_unit_n(NV, d.m, s.ecol[j], VOFF + j);
  s.ecomp[j] = true;
}

// Lazily compute the weld-null-space-projected column + composite
// mobility (solver.rs capcol path; also the limit-row direction).
template <typename T>
__device__ void ensure_pcol(Dyn<T>& __restrict__ d, Solver<T>& __restrict__ s,
                            int j) {
  if (s.pcomp[j]) return;
  ensure_ecol(d, s, j);
  const T plain = s.ecol[j][VOFF + j];
  T* col = s.pcol[j];
  // Copy + 12 dots, INTERCHANGED and fused into one pass over i. Each
  // sum[w] still accumulates over i in ascending order over the same
  // terms, and the copy is per-element independent, so col and every
  // sum[w] are bitwise what the three separate passes produced.
  {
    // The dot half is segmented onto the K5 weld support: each sum[w]
    // still visits its own terms in ascending i, only the structural
    // +-0.0 products are gone. The COPY stays full-length (col is
    // consumed densely).
    T sum[N_WELD];
#pragma unroll
    for (int w = 0; w < N_WELD; w++) sum[w] = T(0);
    for (int i = 0; i < VOFF; i++) {  // root columns: all 12 rows
      const T ci = s.ecol[j][i];
      col[i] = ci;
#pragma unroll
      for (int w = 0; w < N_WELD; w++) sum[w] += s.jw[w][i] * ci;
    }
    for (int i = VOFF; i < WJ_L0; i++) col[i] = s.ecol[j][i];  // copy only
    for (int i = WJ_L0; i < WJ_R0; i++) {  // left-foot chain: rows 0..5
      const T ci = s.ecol[j][i];
      col[i] = ci;
#pragma unroll
      for (int w = 0; w < N_WELD / 2; w++) sum[w] += s.jw[w][i] * ci;
    }
    for (int i = WJ_R0; i < NV; i++) {  // right-foot chain: rows 6..11
      const T ci = s.ecol[j][i];
      col[i] = ci;
#pragma unroll
      for (int w = N_WELD / 2; w < N_WELD; w++) sum[w] += s.jw[w][i] * ci;
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
      T ci = col[i];
#pragma unroll
      for (int w = 0; w < N_WELD; w++) ci -= s.mw[w][i] * x[w];
      col[i] = ci;
    }
  }
  const T dnew = col[VOFF + j];
  s.pdiag[j] = dnew > T(COMPOSITE_EPS) * plain
                   ? dnew
                   : std::numeric_limits<T>::infinity();
  s.pcomp[j] = true;
}

// ---- solver.rs ConstraintSet::solve (steps 7-9) -----------------------------
template <typename T>
__device__ void constraint_solve(Dyn<T>& __restrict__ d,
                                 Solver<T>& __restrict__ s, const T* qpre,
                                 T* __restrict__ v) {
  const T hinv = T(1) / T(H_SUB);

  // -- assembly: weld rows 0-11 ----------------------------------------------
  for (int widx = 0; widx < 2; widx++) {
    const int frame = WELD_FRAMES[widx];
    const T* pf = d.pos[frame];
    for (int t = 0; t < 3; t++) {
      T* jr = s.jw[6 * widx + t];
      for (int i = 0; i < NV; i++) jr[i] = T(0);
      T n[3] = {T(0), T(0), T(0)};
      n[t] = T(1);
      point_jacobian_acc(d, frame, pf, n, T(1), jr);
      s.bias_w[6 * widx + t] =
          -(T(BETA_WELD) * hinv) * (pf[t] - T(WELD_TARGET_P[widx][t]));
    }
    T phi[3];
    rot_log(d.r[frame], phi);
    for (int t = 0; t < 3; t++) {
      T* jr = s.jw[6 * widx + 3 + t];
      for (int i = 0; i < NV; i++) jr[i] = T(0);
      angular_jacobian_acc(d, frame, t, T(1), jr);
      s.bias_w[6 * widx + 3 + t] = -(T(BETA_WELD) * hinv) * phi[t];
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
  // chol_solve_multi_n — a loop interchange, bitwise). s.mw is
  // T[N_WELD][NV], i.e. exactly the W-rows-of-stride-NV layout it wants.
  for (int w = 0; w < N_WELD; w++)
    for (int i = 0; i < NV; i++) s.mw[w][i] = s.jw[w][i];
  chol_solve_multi_n<N_WELD>(NV, d.m, &s.mw[0][0]);
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
  // note), so ONLY the [c][r] (lower, c >= r) entry is stored — the
  // [r][c] mirror was dead. c == r writes the diagonal once.
  // Only the 12 columns of s.jw[c]'s K5 support contribute (s.mw is
  // dense, but it is multiplied by the structural zeros of s.jw[c]).
  for (int r = 0; r < N_WELD; r++)
    for (int c = r; c < N_WELD; c++) {
      const int lo = c < N_WELD / 2 ? WJ_L0 : WJ_R0;
      T sum = T(0);
      for (int i = 0; i < VOFF; i++) sum += s.jw[c][i] * s.mw[r][i];
      for (int i = lo; i < lo + 6; i++) sum += s.jw[c][i] * s.mw[r][i];
      s.aww[c * N_WELD + r] = sum;
    }
  chol_factor_n(N_WELD, s.aww, &d.pd_bad);  // in place: aww -> L_ww

  // -- limit rows: project into the weld null space ---------------------------
  for (int li = 0; li < s.nl; li++) {
    ensure_pcol(d, s, s.l_j[li]);
    s.l_diag[li] = s.pdiag[s.l_j[li]];
    s.l_lam[li] = T(0);
  }
  for (int ci = 0; ci < s.nc; ci++) s.c_lam[ci] = T(0);

  // -- step 8: exactly 12 iterations, fixed order -----------------------------
  for (int iter = 0; iter < PGS_ITERS; iter++) {
    // (a) weld block: exact solve of the current residual.
    // Both halves are LOOP INTERCHANGES of the 12 x NV nests (bitwise —
    // see the argument below); v[i] is touched once per i instead of 12
    // times, and the 12 dots / the 12 terms of each v[i] become
    // independent work instead of 12 back-to-back serial passes.
    {
      // 12 dots against the SAME v: each jv[r] accumulates over i in the
      // identical ascending order over identical terms (v is read-only
      // here), so every jv[r] is bitwise the scalar loop's.
      // segmented onto the K5 weld support (same ascending i per jv[r])
      T jv[N_WELD];
#pragma unroll
      for (int r = 0; r < N_WELD; r++) jv[r] = T(0);
      for (int i = 0; i < VOFF; i++) {
        const T vi = v[i];
#pragma unroll
        for (int r = 0; r < N_WELD; r++) jv[r] += s.jw[r][i] * vi;
      }
      for (int i = WJ_L0; i < WJ_R0; i++) {
        const T vi = v[i];
#pragma unroll
        for (int r = 0; r < N_WELD / 2; r++) jv[r] += s.jw[r][i] * vi;
      }
      for (int i = WJ_R0; i < NV; i++) {
        const T vi = v[i];
#pragma unroll
        for (int r = N_WELD / 2; r < N_WELD; r++) jv[r] += s.jw[r][i] * vi;
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
        for (int r = 0; r < N_WELD; r++) vi += s.mw[r][i] * dl[r];
        v[i] = vi;
      }
    }
    // (b) contacts, pair-id order: projected GS, lambda >= 0
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
    for (int li = 0; li < s.nl; li++) {
      const int j = s.l_j[li];
      const T jv = s.l_sign[li] * v[VOFF + j];
      const T dl = (s.l_bias[li] - jv) / s.l_diag[li];
      const T newl = tmax(s.l_lam[li] + dl, T(0));
      const T dd = newl - s.l_lam[li];
      s.l_lam[li] = newl;
      if (dd != T(0)) {
        const T sd = s.l_sign[li] * dd;
        for (int i = 0; i < NV; i++) v[i] += s.pcol[j][i] * sd;
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
          for (int i = 0; i < NV; i++) v[i] += s.pcol[j][i] * lam;
          v[vj] = target;  // exact — the cap is a hard bound
        }
      }
    }
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
  //     CLAMP_PASSES Gauss-Seidel passes with Rust's early-out, then
  //     the guarantee pass)
  for (int j = 0; j < NQ; j++) {
    const T lo = T(LIMIT_LO[j]), hi = T(LIMIT_HI[j]);
    s.hit[j] = d.q[j] < lo || d.q[j] > hi;
    if (s.hit[j]) d.q[j] = tclamp(d.q[j], lo, hi);
  }
  for (int pass = 0; pass < CLAMP_PASSES; pass++) {
    bool changed = false;
    for (int j = 0; j < NQ; j++) {
      const int vj = VOFF + j;
      const T cap = T(OMEGA_CAP[j]);
      const T target = s.hit[j] ? T(0) : tclamp(d.v[vj], -cap, cap);
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
  for (int j = 0; j < NQ; j++) {
    const int vj = VOFF + j;
    const T cap = T(OMEGA_CAP[j]);
    d.v[vj] = tclamp(d.v[vj], -cap, cap);
    if (s.hit[j]) d.v[vj] = T(0);
  }
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
  // io
  const int64_t *__restrict__ actions, *__restrict__ cand;
  T* __restrict__ ret_buf;
  int* __restrict__ ret_total;
  float* __restrict__ obs;
  T* __restrict__ reward;
  bool* __restrict__ done;
  int N, C, max_steps, cap, mode;  // mode: 0 step, 1 reset-all, 2 refresh
  T effort_coef;
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
  for (int i = 0; i < NV; i++) d.v[i] = T(0);
  store_state(d, P, e);
  refresh_fk(d, P, e);
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
  P.effort[e] = T(0);
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
//  3. KEPT, +0.08%: the zero fill starts at OFF_NOTES. Lanes 0..99 are
//     all UNCONDITIONALLY overwritten below — 0..2 root pos, 3..8 the 6D
//     rotation, 9..14 root vlin/vang, 15..45 the q block (29 joints,
//     j = 7 and 14 emitting two lanes each = 31), 46..74 qd, 75..94 the
//     two 10-lane hand blocks, 95..97 head, 98..99 the contact bits —
//     so their pre-zeroing was 100 dead stores per env per tick (the
//     static_assert pins the prefix gapless). From OFF_NOTES the fill IS
//     load-bearing: note features and the mask are written only for
//     occupied slots.
static_assert(OFF_ROOT_ROT == 3 && OFF_ROOT_VLIN == 9 && OFF_ROOT_VANG == 12
                  && OFF_Q == 15 && OFF_QD == OFF_Q + NQ + 2
                  && OFF_HAND_L == OFF_QD + NQ && OFF_HAND_R == OFF_HAND_L + 10
                  && OFF_HEAD == OFF_HAND_R + 10 && OFF_CONTACT == OFF_HEAD + 3
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
  }
  // head + contact bits
  obs[OFF_HEAD] = (float)(P.head[e * 3] / T(X_NORM));
  obs[OFF_HEAD + 1] = (float)((P.head[e * 3 + 1] - T(HEAD_Y0)) / T(Y_NORM));
  obs[OFF_HEAD + 2] = (float)((P.head[e * 3 + 2] - T(PLANE_Z0)) / T(X_NORM));
  obs[OFF_CONTACT] = P.contact_arm[e * 2] ? 1.f : 0.f;
  obs[OFF_CONTACT + 1] = P.contact_arm[e * 2 + 1] ? 1.f : 0.f;
  // note block — v9 NOTE_FEAT verbatim
  const int64_t cc = P.chart_idx[e];
  const int C = P.C;
  const int nn = (int)P.p_n_notes[cc];
  const T time_o = T(P.tick[e]) * T(DT);
  int slot = 0;
  for (int i = P.first_pending[e]; i < nn && slot < N_NOTE_SLOTS; i++) {
    const int64_t cb = cc * C + i;
    if (!P.p_valid[cb] || P.status[e * C + i] != 0) continue;
    float* nb2 = obs + OFF_NOTES + slot * NOTE_FEAT;
    const T z = tclamp((P.p_hit[cb] - time_o) * P.p_speed[cb], T(Z_CLIP_LO),
                       T(Z_CLIP_HI));
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
    obs[OFF_NOTE_MASK + slot] = 1.f;
    slot++;
  }
}

// One 30 Hz control tick + judgment + bookkeeping (lib.rs step_inner),
// entirely per-thread.
template <typename T>
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

  // gear ids -> torques (lib.rs Body::tick)
  T tau[NQ];
  const double* gr = gear_row(P.gears);
  for (int j = 0; j < NQ; j++) {
    const int a = (int)P.actions[e * NQ + j];
    const double frac = a < 7 ? gr[a] : 0.0;
    tau[j] = T(frac) * T(TAU_MAX[j]);
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
  for (int sub = 0; sub < N_SUB; sub++) {
    substep(d, s, tau);
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

  T reward = T(0);
  reward -= P.effort_coef * effort;

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
  const T min_speed = P.p_min_speed[c0];
  const int nn = (int)P.p_n_notes[c0];
  int n_new[2] = {0, 0};

  for (int i = P.first_pending[e]; i < nn; i++) {
    if (P.status[e * C + i] != 0) continue;
    const int64_t cb = c0 * C + i;
    if (!P.p_valid[cb]) continue;
    // sorted hit_time + min_speed: beyond this bound so is every later
    if ((P.p_hit[cb] - time_new) * min_speed > reach_z) break;
    const T z = (P.p_hit[cb] - time_new) * P.p_speed[cb];
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
  const bool done =
      (!pending && no_trk && (nn == 0 || time_new > last_hit))
      || time_new > last_hit + P.p_tail[c0] || tick_new >= P.max_steps
      || d.env_bad;

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
        P.tr_active[b] = false;
      }
    }
  }

  if (d.env_bad) reward = T(0);
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

template <typename T>
// The `, 1` PINS the register budget at 255 — see the THREADS note: it
// is not an occupancy hint, it is what stops ptxas's free-choice
// heuristic from flipping to 64 regs + double the spills.
__global__ void __launch_bounds__(THREADS, 1)
step_kernel(Sim<T> P) {
  const int e = blockIdx.x * blockDim.x + threadIdx.x;
  if (e >= P.N) return;
  // thread-local scratch (local memory; per-thread interleaved =
  // coalesced across the warp for same-index access)
  Dyn<T> d;
  Solver<T> s;
  if (P.mode == 1) {
    reset_env(d, P, e, P.cand[e]);
  } else if (P.mode == 2) {
    // set_state refresh (RawBeatSim.set_joints semantics): FK caches
    // from the freshly written state; finite-diff wrist velocity reset;
    // swing accumulators untouched
    load_state(d, P, e);
    refresh_fk(d, P, e);
    for (int k = 0; k < 6; k++) P.wrist_vel[e * 6 + k] = T(0);
  } else {
    d.pd_bad = false;
    d.env_bad = false;
    d.env_n = 0;
    step_env(d, s, P, e);
  }
  write_obs(d, P, e);
}

// tensor-list slots (order is a contract with cuda_env.py _tensors)
enum {
  TP = 0, TQUAT, TQ, TV,
  TWRIST, TBLADE, TWVEL, THEAD, TSHZ,
  TSWING, TAXIS, TAXV,
  TTRACC, TTRAGE, TTRCNT, TTRACT, TTRPTR,
  TTICK, TSTATUS, TFIRSTP, TEPRET,
  TGOOD, TBAD, TMISS, TBOMBS, TSCORE, TSPRE, TSACC, TSPOST,
  TCIMP, TCPT, TCARM, TEFF, TPD, TENV, TCIDX,
  PHIT, PSPEED, PCENTER, PDVEC, PROT, PGHY, PBHY,
  PVALID, PISR, PANY, PBOMB, PHEAD, PLINK, PSHOW, PFPRE, PFPOST,
  PNNOTES, PLASTHIT, PTAIL, PMINSP,
  RBUF, RTOT, OOBS, OREW, ODONE, T_COUNT
};

void step(std::vector<at::Tensor> ts, at::Tensor actions, at::Tensor cand,
          int64_t mode, int64_t max_steps, double effort_coef,
          bool wide_acc, bool auto_reset, int64_t gears,
          bool want_contacts) {
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
    P.gears = (int)gears;
    P.wide_acc = wide_acc;
    P.auto_reset = auto_reset;
    P.want_contacts = want_contacts;
    const unsigned blocks = (unsigned)((N + THREADS - 1) / THREADS);
    step_kernel<scalar_t><<<blocks, THREADS, 0,
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
        "output slots");
  m.def("transcend", &transcend,
        "f64 transcendental probe: y = op(x) on the device (0 sin, "
        "1 cos, 2 acos, 3 sqrt, 4 recip)");
}
