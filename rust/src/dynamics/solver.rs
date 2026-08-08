//! SPEC-v10 §3 steps 7–9: constraint assembly + the mixed
//! block/projected-Gauss-Seidel solve (P2 revision — Divergences §15).
//!
//! Row layout (NORMATIVE):
//!   rows 0–5    foot_l weld — 3 linear then 3 angular,
//!   rows 6–11   foot_r weld — same layout,
//!   then one normal row per ACTIVE contact, in pair-id order (§2),
//!   then one unilateral JOINT-LIMIT row per joint whose nearer stop is
//!   within h·ω_cap, joint order 0..28 (folded into the main solve —
//!   the §3 anticipation "P2 may fold these rows"; see below).
//!
//! Solve: exactly 12 iterations, no warm starting (λ = 0 each substep);
//! each iteration = (a) the 12 bilateral weld rows solved EXACTLY as a
//! dense block (Δλ_w = A_ww⁻¹(b_w − J_w v), A_ww = J_w M⁻¹ J_wᵀ
//! Cholesky-factored once per substep), then (b) contact + limit rows
//! projected Gauss-Seidel to λ ≥ 0, in row order. Effective masses and
//! impulse propagation via M⁻¹Jᵀ columns solved with the substep's
//! Cholesky factor (ws.m, factored in place). Why not per-row GS: the two
//! feet's vertical rows are near-parallel through the root at the
//! straight-knee stance — 12 sweeps leave a 0.14 m/s weld-row residual
//! (measured), a ~3 mm steady weld violation. Why limit rows in the
//! solve: the stance parks knee/elbow/wrist-roll exactly ON a 0-stop;
//! a post-hoc clamp fights the weld (solve assigns flexion rate, clamp
//! teleports q back — measured 5.8e-4 m foot error per substep).
//!
//! Right-hand sides (J·v = b welds / J·v ≥ b, λ ≥ 0 unilaterals):
//!   welds:    b = −(β_weld/h)·C_pose   (position error; orientation
//!             error via the SO(3) log map), β_weld = 0.2
//!   contacts: b = (β_c/h)·max(penetration − slop, 0) when penetrating,
//!             β_c = 0.2, slop 0.005; for a still-positive gap inside
//!             the margin, b = −gap/h (speculative activation: approach
//!             is allowed at exactly the rate that closes the remaining
//!             gap this substep — Divergences §16; without this term
//!             the 1 cm margin would act as a hard force field).
//!   limits:   same shape as contacts with slop 0 and β_weld pushback
//!             (row ±e_j toward the interior).
//!
//! Step 11 (ω caps + limit clamps) REMAINS as the outer guarantee —
//! with the solver limit-aware its impulses are near-null in ordinary
//! motion. Full normative substep order: steps 1–6, 7–9 (this module:
//! weld block + contacts pair-id order + limits joint order, 12
//! iterations), 10 integrate, 11 clamp passes joint order 0..28.
//!
//! Loop bounds and evaluation order are fixed throughout — the
//! torch/CUDA mirrors replicate these expression trees literally.
//!
//! ORDER-DEPENDENCE (2026-08-05, R9 — the rule the CUDA kernel header
//! already carries): the four weld contractions now visit the weld
//! Jacobian's SUPPORT instead of all 35 columns, which changes this
//! module's register/inner-loop economics. Jam widths and any change
//! previously measured at under ~1% must be RE-PROBED against the
//! post-R9 build before being called dead.
//! The rule PAID (2026-08-06, R15): the by-4 axpy jams that R1 had
//! established as the winners were beaten by width 12 post-R9. It also
//! bites WITHIN a task — the PGS axpy's own gain fell from +0.29% to
//! +0.17% once the weld-null axpys were widened, so probe increments
//! against the build you will actually ship, not against the baseline.
//!
//! R23 (2026-08-07) — DEAD-STORE SWEEP. Net **+2.10%** at N = 256
//! (22,326 → 22,795 env-steps/s, two ORDER-BALANCED interleaved ABBA
//! batches, 10/10 paired wins, ranges non-overlapping by ~6x the
//! within-arm spread) and **−1.6% of the stars-4-5 bench's step_into
//! wall** (74.3 → 73.1 s over 28,291 group-steps at --limit 300;
//! 1,470-level record EXACT, wpp 4,483.24 / upp 190,452.80, 0 field
//! mismatches, 0 pd-floor and 0 envelope hits). Bitwise: max diff
//! exactly 0.0 over 4.78M recorded words (f64 state + f32 obs, four
//! trajectory suites). Five audited sites, each gated on its own
//! measurement — three landed, two are recorded negatives:
//!   · **R23a** the limit rows' dense Jacobian, cfg-gated (this file):
//!     **+1.87%** alone (22,051 → 22,464, 6/6 paired wins);
//!   · **R23c** `jwt`'s permanently-+0.0 column block, NO cfg gate
//!     because the bytes are identical (this file): **+1.13%** alone
//!     (22,327 → 22,579, 6/6);
//!   · **R23e** `rnea`'s `ws.vo` chain (algo.rs): **+0.45%** alone
//!     (22,330 → 22,430, 6/6);
//!   · **R23b** the mass matrix's upper triangle (algo.rs) — the
//!     free-base half ONLY: **+0.28%** on top of the package
//!     (12 pairs, two order-balanced batches, 10/12 wins);
//!   · **R23d** (step.rs `ws.sol`) **−0.84%** and **R23f** (`rw`,
//!     below) **−2.71%**: both REVERTED, both bitwise-clean, both
//!     documented at their sites.
//!
//! TWO NEW RULES, both earned the expensive way (the six-item package
//! measured **−1.92%** before the per-item pass found the losers):
//!   1. **A `chol_*` callee wants ONE dedicated, loop-invariant scratch
//!      buffer.** Deleting a staging copy by handing the solve a
//!      runtime-offset window into a bigger array (R23d) or a fresh
//!      stack `[f64; 12]` (R23f) costs 1–3% — an order of magnitude
//!      more than the 12–35 word copies it removes. What is being
//!      traded is not traffic but the callee's addressing/aliasing
//!      regime across a 12×12 or 35×35 solve.
//!   2. **A dead-store deletion pays only when the address is a
//!      COMPILE-TIME constant of the loop.** Buying one with a runtime
//!      min/max select is a wash: R23b's ancestor-loop half, whose
//!      lower/upper roles are data-dependent, measured −0.18% and was
//!      dropped while its statically-typed sibling landed.
//! And R15's ORDER-DEPENDENCE rule earned its keep twice: R23d was
//! −0.03% (indistinguishable from zero) measured ALONE and −0.84%
//! against the build that would ship it, while R23a/c/e summed to
//! +3.45% individually but delivered +1.85% together.

use super::algo::{
    chol_back, chol_back_t, chol_back_x2, chol_factor_floored, chol_fwd_unit, chol_solve,
    chol_solve_t, chol_solve_unit, chol_solve_x2, chol_solve_x4,
};
use super::collide::{narrow_phase, Contact, NUM_PAIRS, SLOP};
use super::fk::{cross3, dot3, rot_log, sub3};
use super::model::{Model, F_L_FOOT, F_R_FOOT};
use super::step::{substep_wrench, ZERO_WRENCH};
use super::{State, Workspace};

/// PGS iteration count (§3 step 8, normative).
pub const PGS_ITERS: usize = 12;
/// Baumgarte gains (§3 step 8, normative).
pub const BETA_WELD: f64 = 0.2;
pub const BETA_CONTACT: f64 = 0.2;

/// Feet weld definition (§3): foot body frames held at their
/// reference-pose world poses, identity orientation. The anchor is each
/// foot's world position — NOT the world origin (the oracle's original
/// world-origin-anchor bug put a 0.66 m lever on the rotation rows).
pub const WELD_FRAMES: [usize; 2] = [F_L_FOOT, F_R_FOOT];
pub const WELD_TARGET_P: [[f64; 3]; 2] = [[-0.10, 0.07, -0.65], [0.10, 0.07, -0.65]];

pub const N_WELD_ROWS: usize = 12;
/// Max rows: 12 weld + 55 contact + ≤ 1 joint-limit row per joint
/// (Divergences §15 — the nearer stop only, when within reach).
pub const MAX_ROWS: usize = N_WELD_ROWS + NUM_PAIRS + 29; // 96

// ── R9 (2026-08-05): WELD-JACOBIAN SUPPORT — the port of the CUDA
//    kernel's K5 sparsity (beat_arena_cuda.cu §20c) to the Rust sim.
//
// A weld row is assembled as `jr.fill(0.0)` followed by
// `point_jacobian_acc` / `angular_jacobian_acc`, and those two write
// ONLY (a) the free-base columns `row[0..3]` / `row[3 + t]` and (b)
// `row[voff + fr.coord]` for each frame on the welded foot's ancestor
// chain. Nothing else is ever touched, so every other column keeps the
// memset's +0.0 EXACTLY.
//
//   F_L_FOOT = 23 → chain 23,22,21,20,19,18 → coords 22..17
//                 → v-columns voff+17 .. voff+22 = 23..28
//   F_R_FOOT = 29 → chain 29,28,27,26,25,24 → coords 28..23
//                 → v-columns 29..34
//
// so, with voff = 6 and nv = 35:
//   rows 0..6  (left foot)  support = {0..6} ∪ {23..29}
//   rows 6..12 (right foot) support = {0..6} ∪ {29..35}
// and columns 6..23 are +0.0 in ALL TWELVE rows.
//
// BITWISE: every consumer is a set of 12 dots `s[w] += J_w[w][i]·x[i]`
// with `s` seeded +0.0. A skipped term contributes (+0.0)·x = ±0.0, and
// `y + (±0.0) == y` for every y reachable here: an accumulator seeded
// +0.0 can never become −0.0 (x + y = −0.0 under RN needs x = y = −0.0),
// so the −0.0 channel is unreachable and the accumulator is unchanged.
// The KEPT terms keep their original ascending-i order, interleaved
// skips and all — no reassociation. (Holds on finite trajectories; the
// same acceptance as CUDA K5, since 0·NaN = 0·∞ = NaN. Work-order §10.)
const W_BASE_HI: usize = 6; // free-base columns 0..6, all 12 rows
const W_L_LO: usize = 23; // left-foot chain, rows 0..6
const W_L_HI: usize = 29;
const W_R_LO: usize = 29; // right-foot chain, rows 6..12
const W_R_HI: usize = 35;
// Pin the layout the segments are derived from (mirrors the
// `static_assert` at beat_arena_cuda.cu:349).
const _: () = {
    assert!(F_L_FOOT == 23 && F_R_FOOT == 29);
    assert!(super::model::NQ == 29 && super::model::NV == W_R_HI);
    assert!(N_WELD_ROWS == 12);
    assert!(W_BASE_HI == 6 && W_L_LO == 23 && W_L_HI == 29 && W_R_LO == 29);
};

/// `s[w] += J_w[w][i] · src[i]` for all 12 weld rows, over the SUPPORT
/// of the weld Jacobian only (R9). `jwt` is the transposed weld
/// Jacobian (`jwt[i*12 + w]`), `s` a `[f64; N_WELD_ROWS]` seeded +0.0.
///
/// Only the all-zero column block 6..23 is skipped; all 12 contiguous
/// lanes are kept, so the inner loop stays the unit-stride 12-lane AVX2
/// shape the R1 transpose bought. That is worth +3.4% on its own
/// (19,505 → 20,162 env-steps/s at N = 256, pooled median of six
/// interleaved A/B pairs) and −3.6% of bench wall (472 → 455 s).
///
/// The FULL tier — additionally restricting the left segment {23..29}
/// to lanes 0..6 and the right {29..35} to lanes 6..12, which is the
/// exact 96-of-420 support — was implemented and MEASURED TWICE and is
/// NOT shipped: 6-lane halves +0.15%, halves padded to 8 lanes
/// (0..8 / 4..12, the padding lanes reading the +0.0 entries, which is
/// bitwise-neutral by the same argument) +0.25%, both against a cheap
/// tier whose own run-to-run spread reached 1.5% on this box. Two more
/// loop bodies and a padding argument for a gain under the noise floor
/// is exactly the "neutral with added complexity" the revert rule kills.
/// The remaining halves are 1.5 AVX2 registers wide — the shape, not the
/// term count, is what stops paying here.
macro_rules! weld_support_dot {
    ($s:expr, $jwt:expr, $src:expr) => {{
        for i in 0..W_BASE_HI {
            let x = $src[i];
            let jr = &$jwt[i * N_WELD_ROWS..(i + 1) * N_WELD_ROWS];
            for w in 0..N_WELD_ROWS {
                $s[w] += jr[w] * x;
            }
        }
        for i in W_L_LO..W_R_HI {
            let x = $src[i];
            let jr = &$jwt[i * N_WELD_ROWS..(i + 1) * N_WELD_ROWS];
            for w in 0..N_WELD_ROWS {
                $s[w] += jr[w] * x;
            }
        }
    }};
}

/// `weld_support_dot!` for TWO independent sources at once (R16): the
/// 12-lane `jwt` row is loaded ONCE and feeds both 12-accumulator sets.
/// BITWISE: `$s0` and `$s1` are separate accumulator sets, each still
/// visiting its own source over the same ascending i — interleaving two
/// independent chains moves no term of either.
macro_rules! weld_support_dot2 {
    ($s0:expr, $s1:expr, $jwt:expr, $a:expr, $b:expr) => {{
        for i in 0..W_BASE_HI {
            let (xa, xb) = ($a[i], $b[i]);
            let jr = &$jwt[i * N_WELD_ROWS..(i + 1) * N_WELD_ROWS];
            for w in 0..N_WELD_ROWS {
                $s0[w] += jr[w] * xa;
                $s1[w] += jr[w] * xb;
            }
        }
        for i in W_L_LO..W_R_HI {
            let (xa, xb) = ($a[i], $b[i]);
            let jr = &$jwt[i * N_WELD_ROWS..(i + 1) * N_WELD_ROWS];
            for w in 0..N_WELD_ROWS {
                $s0[w] += jr[w] * xa;
                $s1[w] += jr[w] * xb;
            }
        }
    }};
}

/// `dst[i] <op>= Σ_w  minvjt_w[i] · d[w]` over ALL TWELVE weld rows in
/// ONE pass over `dst` (R15). `$mj` is the head of `minvjt` (the 12 weld
/// rows, row-major, `nv` apart), `$d` a `[f64; N_WELD_ROWS]`, `$op` the
/// compound assignment (`+=` for the PGS impulse propagation, `-=` for
/// the two weld-null projections).
///
/// BITWISE at ANY jam width, including this one: `dst[i]` takes its 12
/// updates in w = 0..11 order exactly as the row-at-a-time (and the
/// by-4) form did — the jam only holds `dst[i]` in a register across all
/// twelve instead of four, so no term moves and nothing reassociates.
/// The twelve `m*` slices are disjoint splits of one 12·nv block, so
/// each stays UNIT-STRIDE in i and the i loop still vectorizes.
///
/// WIDTH LADDER, re-probed post-R9 as the module header's
/// ORDER-DEPENDENCE rule demands (2026-08-06, R15; N = 256, 4 interleaved
/// ABBA pairs each). What is bought is `dst` ROUND TRIPS — the by-4 form
/// read and wrote all of `dst` three times per application:
///   · the two WELD-NULL projections (limit-row + ω-cap column), 4 → 12:
///     **+0.67%** (22,090 → 22,238) — the winner of this task;
///   · the PGS impulse propagation, 4 → 12: +0.29% measured alone
///     (22,080 → 22,145), but only **+0.17%** once the projections were
///     already at 12 (22,235 → 22,274) — the same registers, counted
///     twice. Kept for the shared macro (it deletes the third copy of
///     the by-4 body), NOT for the number;
///   · the PGS propagation at width 6 (two groups, i.e. 2 round trips
///     instead of 3): **+0.04%** — noise. The round trip you do not save
///     is not worth the registers you spend, so it is 12 or nothing.
macro_rules! weld_axpy12 {
    ($dst:expr, $mj:expr, $d:expr, $nv:expr, $op:tt) => {{
        let mj = &$mj[..N_WELD_ROWS * $nv];
        let (m0, t) = mj.split_at($nv);
        let (m1, t) = t.split_at($nv);
        let (m2, t) = t.split_at($nv);
        let (m3, t) = t.split_at($nv);
        let (m4, t) = t.split_at($nv);
        let (m5, t) = t.split_at($nv);
        let (m6, t) = t.split_at($nv);
        let (m7, t) = t.split_at($nv);
        let (m8, t) = t.split_at($nv);
        let (m9, t) = t.split_at($nv);
        let (m10, m11) = t.split_at($nv);
        for i in 0..$nv {
            let mut x = $dst[i];
            x $op m0[i] * $d[0];
            x $op m1[i] * $d[1];
            x $op m2[i] * $d[2];
            x $op m3[i] * $d[3];
            x $op m4[i] * $d[4];
            x $op m5[i] * $d[5];
            x $op m6[i] * $d[6];
            x $op m7[i] * $d[7];
            x $op m8[i] * $d[8];
            x $op m9[i] * $d[9];
            x $op m10[i] * $d[10];
            x $op m11[i] * $d[11];
            $dst[i] = x;
        }
    }};
}

// R16 NEGATIVE (2026-08-06) — do not re-add without a register-economy
// change. Sharing the twelve `minvjt_w[i]` LOADS between two projection
// targets (a `weld_axpy12_x2!` doing `a -= mw[i]·x0[w]; b -= mw[i]·x1[w]`
// for w = 0..11 in one i pass) was written, bitwise-clean, and measured
// **−1.07%** against the same build with two separate `weld_axpy12!`
// calls (22,280 → 22,042 env-steps/s, N = 256, 4 interleaved ABBA pairs,
// every pair negative). The dot side of the same jam WINS (+0.42%) —
// what differs is the loop-invariant budget: the dot carries 12 shared
// `jr[w]` lanes and its accumulators live in memory anyway, while the
// axpy already spends its whole register file on 12 base pointers + 12
// broadcast deltas, so a second column's 12 deltas buy 12 spilled
// reloads per i to save 12 L1 hits. WIDTH 4 (`weld_support_dot4!` +
// `chol_solve_x4` + four `weld_axpy12!`s) was also built and measured:
// **−2.70%** (22,405 → 21,800) — 4×12 dot accumulators no longer fit,
// and the spill traffic swamps the halved `jwt` traversal. Two is the
// width that pays.

// ── R10 (2026-08-06): LIMIT-ROW CLOSED FORMS — the rank-1 sibling of
//    R9's sparsity argument (the CUDA kernel has carried it since the
//    unit-RHS forward pass: beat_arena_cuda.cu's PGS reads
//    `jv = s.l_sign[li] * v[VOFF + j]` instead of a 35-wide dot).
//
// A joint-limit row is assembled as `jr.fill(0.0)` plus ONE store
// `jr[voff + j] = sign` (±1.0). So every contraction `Σᵢ jr[i]·x[i]`
// over such a row is rank-1 and has the closed form below. Three
// consumers collapse: the pre-solve effective mass `Σ jr·mr`, the
// post-weld-null-projection `dnew`, and — 12 times per substep, the hot
// one — the PGS residual `Σ jr·v`.
//
// BITWISE, with the ZERO-SIGN GUARD (the leading `0.0 +` is LOAD-BEARING
// and must not be dropped):
//   · the accumulator is seeded +0.0 and the terms below the pivot are
//     (+0.0)·x[i] = ±0.0; `+0.0 + (±0.0) = +0.0` under RN, so the running
//     sum entering i = voff+j is EXACTLY +0.0 — i.e. the dense sum
//     computes `0.0 + sign·x[vj]`, not `sign·x[vj]`;
//   · those differ: with x[vj] = +0.0 and sign = −1 the bare product is
//     −0.0 while `0.0 + (−0.0)` is +0.0. That case is not exotic — step
//     11 parks a clamped joint's rate at exactly 0.0 and upper-stop rows
//     carry sign = −1 — and it is observable: `bias − jv` turns
//     `−0.0 − (+0.0) = −0.0` into `−0.0 − (−0.0) = +0.0`;
//   · terms above the pivot then add ±0.0 to that value, which leaves it
//     unchanged (including when it is itself +0.0).
// LLVM cannot fold `0.0 + y` away without `nsz`, which this crate never
// enables. (Finite trajectories, work-order §10: 0·NaN = 0·∞ = NaN.)
//
// R23a (2026-08-07): with all three contractions closed-form, the dense
// limit ROW ITSELF has no reader left in the shipping build, so it is no
// longer assembled there. Every consumer of `self.j` at a row ≥ ndense
// was re-checked at land time:
//   · `chol_fwd_unit(nv, m, voff+jt, sign, …)` takes the pivot INDEX and
//     the sign as arguments — it never looks at `j`;
//   · the effective mass and the post-projection `dnew` are the R10
//     closed forms `0.0 + sign·minvjt[row·nv + voff+jt]`;
//   · the PGS residual is `0.0 + sign·v[voff+jt]`;
//   · `jv_pre` is `cfg(any(test, feature = "introspect"))`;
//   · `row_j` / `limit_dot_ref_vs_closed` are `cfg(test)`;
//   · `pub fn jv` has exactly one caller, tests_constraints.rs.
// The `self.minvjt[nweld·nv..ndense·nv] ← self.j` seed and the two
// `for row in 0..ndense` loops stop at ndense by construction, so no
// dense-row consumer can reach a limit row. `LIMIT_J_ROWS` keeps the
// assembly for `cargo test`, `--features introspect` and debug builds —
// the same idiom (and the same three cfgs) as `DIAG_ALL_ROWS` below.
// Worth `nlim × (nv + 1)` stores per substep, i.e. ~270 under the
// random-gear micro and ~565 under bench traffic (R26: nlim mean 15.73).
const LIMIT_J_ROWS: bool = cfg!(any(test, feature = "introspect", debug_assertions));

// MEASURED +5.4% at N = 256 (20,123 → 21,215 env-steps/s, pooled median
// of six interleaved A/B pairs) and −9.8% of bench wall (440.5 → 397.4 s
// on stars 4–5, two runs each, spread under 1 s on both sides). The gap
// between the two is REAL and worth remembering when sizing later work:
// the micro-bench drives random gears, which throw the body into
// contorted poses, while the benched policy parks joints ON their stops
// (knee/elbow/wrist-roll at the 0-limit) — so the bench assembles more
// limit rows per substep than the micro does, and a limit-row saving is
// worth strictly more there. Micro is a LOWER BOUND for this class.

/// Constraint stage state + preallocated buffers (no allocation inside
/// a substep — contact Vec capacity is reserved up front). Construct
/// once per sim instance; plug into `substep` via `substep_constrained`.
pub struct ConstraintSet {
    pub welds: bool,
    pub contacts_enabled: bool,
    /// Substep length h — set by the substep wrappers before solving.
    pub h: f64,
    /// Active contacts of the last solve, pair-id order.
    pub contacts: Vec<Contact>,
    /// Assembled row count of the last solve (welds + contacts +
    /// limit rows).
    pub nrows: usize,
    /// Per-row impulses λ (N·s) of the last solve. Rows 0..12 welds
    /// (when enabled), then contacts in pair-id order, then limit
    /// rows in joint order.
    pub lambda: Vec<f64>,
    /// Per-row RHS target velocity b of the last solve.
    pub bias: Vec<f64>,
    /// Per-row J·v* BEFORE the solve (introspection: complementarity /
    /// no-sticking checks). INTROSPECTION-ONLY — nothing in the solve
    /// reads it, so both the field and its per-row nv-dot are compiled
    /// out of the shipping extension (2026-08-03 dead-work pass);
    /// `cargo test` and `--features introspect` keep it.
    #[cfg(any(test, feature = "introspect"))]
    pub jv_pre: Vec<f64>,
    /// TEST HOOK (R23a): when set, every assembled LIMIT row of `j` is
    /// overwritten with NaN as soon as the assembly loop finishes —
    /// which is exactly the state the SHIPPING build leaves them in
    /// (never written at all, so holding whatever the buffer held).
    /// A trajectory that is bit-identical with this on proves the
    /// shipping build has no reader for them.
    #[cfg(test)]
    pub poison_limit_j: bool,
    nv: usize,
    j: Vec<f64>,      // MAX_ROWS × nv dense Jacobian rows
    minvjt: Vec<f64>, // MAX_ROWS × nv  M⁻¹Jᵀ columns
    diag: Vec<f64>,   // per-row effective mass  J M⁻¹ Jᵀ (diagonal)
    // weld-block exact solve (Divergences §15): A_ww = J_w M⁻¹ J_wᵀ
    aww: Vec<f64>, // 12×12
    lww: Vec<f64>, // its Cholesky factor
    // block residual / solve scratch (12).
    //
    // R23f REVERTED (2026-08-07, the round's largest measured negative).
    // All three A_ww solves round-trip a 12-vector through this heap
    // buffer (`s` → `rw` → `chol_solve` → `x`); replacing it with a
    // `[f64; 12]` STACK array at every site — compile-time length, two
    // 12-word copies fewer per application, bitwise-identical — measured
    // **−2.71%** at N = 256 (22,323 → 21,717 env-steps/s, 3 interleaved
    // ABBA rounds, 6/6 paired losses, ranges completely non-overlapping).
    //
    // THE RULE, and it also killed R23d: on this CPU a `chol_*` callee
    // wants ONE dedicated, loop-invariant scratch buffer. `&mut self.rw`
    // hands it a slice whose base pointer is fixed for the whole substep
    // and whose allocation nothing else in the solve touches; `&mut x`
    // on a fresh stack array forces the address to escape into the
    // callee on every one of the 12 PGS iterations, so the frame slot is
    // re-established and the solve's accesses can no longer be hoisted
    // the same way. The copies the transform deletes are 24 L1 hits; the
    // aliasing/addressing regime it disturbs is the whole 12×12 solve.
    // Do not retry without an aliasing argument (e.g. `#[inline]`ing the
    // solve so the array never escapes).
    rw: Vec<f64>,
    // ω-cap pass caches: weld-consistent M⁻¹e_j columns, lazily solved
    capcol: Vec<f64>,     // nq × nv
    capmass: Vec<f64>,    // nq: e_jᵀ·capcol_j (composite effective mobility)
    capcached: Vec<bool>, // nq
    /// (joint, sign) per assembled limit row, assembly order.
    limset: Vec<(usize, f64)>,
    /// Row index of joint j's LOWER-stop (sign +1) limit row this
    /// solve, or −1. The ω-cap pass reuses that row's already-solved,
    /// already-projected column instead of redoing the work (R1 1f).
    limpos: Vec<i32>,
    /// nv × 12 RHS-minor scratch for the weld batch (R1 1a).
    wt: Vec<f64>,
    /// TRANSPOSED weld Jacobian, nv rows of 12 (`jwt[i*12 + w]` =
    /// J_w[w][i]), rebuilt once per solve. Every consumer of the weld
    /// rows is a set of 12 INDEPENDENT dots against one shared vector
    /// — the PGS residual and both weld-null projections — and in this
    /// layout those 12 accumulators are 12 SIMD lanes, each keeping
    /// its own ascending-i chain (so the vectorization is legal under
    /// strict IEEE, unlike vectorizing a single dot's reduction).
    jwt: Vec<f64>,
}

/// A composite (weld-consistent) direction with mobility below this
/// fraction of the joint's plain (M⁻¹)_jj is treated as immobile: the
/// welds already forbid the motion, so the row/cap is skipped.
///
/// REVISED at P3 (1e-10 → 1e-4, SPEC-v10 Divergences §18): the
/// projected direction c_j = S·e_j with S = M⁻¹ − M⁻¹J_wᵀA_ww⁻¹J_wM⁻¹
/// (symmetric PSD), so enforcing the cap/limit on joint j moves any
/// other coordinate by Δv_i = S_ij·Δv_j/S_jj ≤ sqrt(S_ii/S_jj)·Δv_j.
/// At 1e-10 a joint in the near-immobile band (S_jj ∈ (1e-10, 1e-4)
/// of plain mobility — contorted random-gear poses reach it) passed as
/// "mobile" and its cap assignment amplified impulses into the root by
/// up to ~1e5× per application — measured blow-up from |ω_root| ≈ 15
/// to 5e5 rad/s within ONE control tick (512-env random-gear soak,
/// env 391), ending in a Cholesky PD panic. At 1e-4 the worst-case
/// amplification is ~sqrt(1/1e-4) = 100; joints below it are owned by
/// the welds and their bounds fall to the step-11 clamp (plain
/// momentum-consistent impulses, always well-conditioned).
const COMPOSITE_EPS: f64 = 1e-4;

impl ConstraintSet {
    pub fn new(model: &Model, welds: bool, contacts_enabled: bool) -> ConstraintSet {
        let nv = model.nv();
        ConstraintSet {
            welds,
            contacts_enabled,
            h: super::step::H_SUB,
            contacts: Vec::with_capacity(NUM_PAIRS),
            nrows: 0,
            lambda: vec![0.0; MAX_ROWS],
            bias: vec![0.0; MAX_ROWS],
            #[cfg(any(test, feature = "introspect"))]
            jv_pre: vec![0.0; MAX_ROWS],
            #[cfg(test)]
            poison_limit_j: false,
            nv,
            j: vec![0.0; MAX_ROWS * nv],
            minvjt: vec![0.0; MAX_ROWS * nv],
            diag: vec![0.0; MAX_ROWS],
            aww: vec![0.0; N_WELD_ROWS * N_WELD_ROWS],
            lww: vec![0.0; N_WELD_ROWS * N_WELD_ROWS],
            rw: vec![0.0; N_WELD_ROWS],
            capcol: vec![0.0; model.nq * nv],
            capmass: vec![0.0; model.nq],
            capcached: vec![false; model.nq],
            limset: Vec::with_capacity(model.nq),
            limpos: vec![-1; model.nq],
            wt: vec![0.0; N_WELD_ROWS * nv],
            jwt: vec![0.0; N_WELD_ROWS * nv],
        }
    }

    /// Back-substitute W consecutive `minvjt` rows through the shared
    /// factor in the RHS-MINOR layout: transpose into `wt`, run
    /// `chol_back_t`, transpose back. Both copies are exact, so this is
    /// bitwise `chol_back` applied to each row on its own.
    fn back_t<const W: usize>(
        nv: usize,
        l: &[f64],
        minvjt: &mut [f64],
        row: usize,
        wt: &mut [f64],
    ) {
        for r in 0..W {
            let mr = &minvjt[(row + r) * nv..(row + r + 1) * nv];
            for i in 0..nv {
                wt[i * W + r] = mr[i];
            }
        }
        chol_back_t::<W>(nv, l, &mut wt[..W * nv]);
        for r in 0..W {
            let mr = &mut minvjt[(row + r) * nv..(row + r + 1) * nv];
            for i in 0..nv {
                mr[i] = wt[i * W + r];
            }
        }
    }

    /// One projected-Gauss-Seidel row update given the row's already
    /// computed residual `jv`: λ ← max(λ + (b − J·v)/diag, 0), with the
    /// change propagated through the row's `M⁻¹Jᵀ` column. Split out by
    /// R10 so the contact half (dense dot) and the limit half (rank-1
    /// closed form) share ONE copy of the update — `#[inline(always)]`,
    /// so the two call sites compile to what the single fused loop did.
    #[inline(always)]
    #[allow(clippy::too_many_arguments)]
    fn pgs_row_update(
        row: usize,
        nv: usize,
        jv: f64,
        bias: &[f64],
        diag: &[f64],
        lambda: &mut [f64],
        minvjt: &[f64],
        v: &mut [f64],
    ) {
        let dl = (bias[row] - jv) / diag[row];
        let newl = (lambda[row] + dl).max(0.0); // λ ≥ 0
        let d = newl - lambda[row];
        lambda[row] = newl;
        if d != 0.0 {
            let mr = &minvjt[row * nv..(row + 1) * nv];
            for i in 0..nv {
                v[i] += mr[i] * d;
            }
        }
    }

    /// J·v of an assembled row (post-solve introspection).
    ///
    /// R23a: valid for a LIMIT row only when `LIMIT_J_ROWS` holds
    /// (test / `introspect` / debug builds) — the shipping build no
    /// longer assembles those rows, so `self.j` holds stale words there.
    /// The only caller is `tests_constraints`.
    pub fn jv(&self, row: usize, v: &[f64]) -> f64 {
        let jr = &self.j[row * self.nv..(row + 1) * self.nv];
        let mut s = 0.0;
        for i in 0..self.nv {
            s += jr[i] * v[i];
        }
        s
    }

    /// TEST HOOK (R9): the assembled dense weld Jacobian, `N_WELD_ROWS`
    /// rows of `nv` — the object whose support the segmented loops rest
    /// on.
    #[cfg(test)]
    pub fn weld_j(&self) -> &[f64] {
        &self.j[..N_WELD_ROWS * self.nv]
    }

    /// TEST HOOK (R9): the transposed weld Jacobian (`jwt[i*12 + w]`),
    /// the layout every segmented consumer reads.
    #[cfg(test)]
    pub fn weld_jwt(&self) -> &[f64] {
        &self.jwt[..N_WELD_ROWS * self.nv]
    }

    /// TEST HOOK (R23c): the half-open column range `W_BASE_HI..W_L_LO`
    /// that R9's support argument makes permanently +0.0 in all twelve
    /// weld rows — the block the transpose loop no longer stores into
    /// `jwt`.
    #[cfg(test)]
    pub const WELD_DEAD_COLS: (usize, usize) = (W_BASE_HI, W_L_LO);

    /// TEST HOOK (R23c): that block of the LIVE `jwt`, as stored.
    #[cfg(test)]
    pub fn weld_jwt_dead_block(&self) -> &[f64] {
        &self.jwt[W_BASE_HI * N_WELD_ROWS..W_L_LO * N_WELD_ROWS]
    }

    /// TEST HOOK (R9): the dense-vs-segmented 12-lane dot, run on the
    /// LIVE `jwt` of the last solve. Returns (dense, segmented).
    #[cfg(test)]
    pub fn weld_dot_ref_vs_segmented(
        &self,
        src: &[f64],
    ) -> ([f64; N_WELD_ROWS], [f64; N_WELD_ROWS]) {
        let nv = self.nv;
        let mut dense = [0.0f64; N_WELD_ROWS];
        for i in 0..nv {
            let x = src[i];
            let jr = &self.jwt[i * N_WELD_ROWS..(i + 1) * N_WELD_ROWS];
            for w in 0..N_WELD_ROWS {
                dense[w] += jr[w] * x;
            }
        }
        let mut seg = [0.0f64; N_WELD_ROWS];
        weld_support_dot!(seg, self.jwt, src);
        (dense, seg)
    }

    /// TEST HOOK (R13): fill the two scratch regions the R13 dead-store
    /// deletions rest on — `minvjt`'s weld rows and the `wt` staging
    /// buffer — with NaN. The next solve must reproduce the un-poisoned
    /// trajectory BIT FOR BIT, which is exactly the claim R13 rests on:
    /// the weld block's RHS-MINOR path writes every one of those words
    /// before anything reads them, so seeding `minvjt` rows 0..12 from
    /// `j` (and copying `jwt -> wt` as a separate pass) were both dead.
    #[cfg(test)]
    pub fn poison_weld_scratch(&mut self) {
        let n = N_WELD_ROWS * self.nv;
        self.minvjt[..n].fill(f64::NAN);
        self.wt[..n].fill(f64::NAN);
    }

    /// TEST HOOK (R10): the `(joint, sign)` pairs of the limit rows
    /// assembled by the last solve, assembly order.
    #[cfg(test)]
    pub fn limit_set(&self) -> &[(usize, f64)] {
        &self.limset
    }

    /// TEST HOOK (R10): an assembled Jacobian row of the last solve.
    #[cfg(test)]
    pub fn row_j(&self, row: usize) -> &[f64] {
        &self.j[row * self.nv..(row + 1) * self.nv]
    }

    /// TEST HOOK (R10): an `M⁻¹Jᵀ` column of the last solve (post
    /// weld-null projection for the limit rows).
    #[cfg(test)]
    pub fn row_minvjt(&self, row: usize) -> &[f64] {
        &self.minvjt[row * self.nv..(row + 1) * self.nv]
    }

    /// TEST HOOK (R10): dense-vs-closed-form contraction of every
    /// assembled LIMIT row against `src`, on the LIVE rows of the last
    /// solve. Returns `(dense, closed)` per limit row — the single
    /// expression all three R10 sites collapse.
    #[cfg(test)]
    pub fn limit_dot_ref_vs_closed(&self, voff: usize, src: &[f64]) -> Vec<(f64, f64)> {
        let nv = self.nv;
        let ndense = self.weld_rows() + self.contacts.len();
        (0..self.limset.len())
            .map(|li| {
                let (jt, sign) = self.limset[li];
                let jr = &self.j[(ndense + li) * nv..(ndense + li + 1) * nv];
                let mut d = 0.0;
                for i in 0..nv {
                    d += jr[i] * src[i];
                }
                (d, 0.0 + sign * src[voff + jt])
            })
            .collect()
    }

    /// Weld row count actually assembled (0 or 12).
    pub fn weld_rows(&self) -> usize {
        if self.welds {
            N_WELD_ROWS
        } else {
            0
        }
    }

    /// Number of joint-limit rows assembled by the last solve.
    pub fn limit_rows(&self) -> usize {
        self.nrows - self.weld_rows() - self.contacts.len()
    }

    /// Effective mass (J M⁻¹ Jᵀ diagonal) of an assembled row —
    /// post-solve introspection (tests).
    pub fn row_diag(&self, row: usize) -> f64 {
        self.diag[row]
    }

    /// §3 steps 7–9 against the FK + Cholesky factor already in `ws`
    /// (valid at the seam: computed from the pre-integration q this
    /// substep). `q` is the pre-integration actuated position vector
    /// (joint-limit rows). Projects `v` (= v*) in place.
    pub fn solve(&mut self, model: &Model, ws: &mut Workspace, q: &[f64], v: &mut [f64]) {
        let nv = self.nv;
        debug_assert_eq!(nv, model.nv());
        // (Reborrowing `v` to exactly nv here to hoist the PGS bounds
        // checks was tried 2026-08-03 and measured 2.6% SLOWER — the
        // extra reborrow apparently costs more analysis than the checks
        // it removes, which the row-slice forms already dominate.
        // Same for slicing st.v in step.rs's clamp axpy. Left alone.)
        let voff = model.voff();
        let hinv = 1.0 / self.h;
        let mut nrows = 0usize;

        // ── assembly: weld rows 0–11 ──────────────────────────────────
        if self.welds {
            debug_assert!(model.free_base, "welds assume the humanoid tree");
            // R9: the weld-support column segments are compile-time
            // constants derived from the humanoid layout — which is the
            // only model WELD_FRAMES can index anyway. One predictable
            // branch per solve keeps that implicit assumption honest.
            assert_eq!(nv, W_R_HI, "the R9 weld-support segments pin nv = 35");
            for (widx, (&frame, target)) in
                WELD_FRAMES.iter().zip(WELD_TARGET_P.iter()).enumerate()
            {
                let pf = ws.p[frame];
                // 3 linear rows: world velocity of the foot-frame origin
                // (anchored at the foot's own position — zero lever)
                for t in 0..3 {
                    let row = 6 * widx + t;
                    let jr = &mut self.j[row * nv..(row + 1) * nv];
                    jr.fill(0.0);
                    let mut n = [0.0; 3];
                    n[t] = 1.0;
                    point_jacobian_acc(model, ws, frame, pf, n, 1.0, jr);
                    self.bias[row] = -(BETA_WELD * hinv) * (pf[t] - target[t]);
                }
                // 3 angular rows: world angular velocity of the foot
                // body; orientation error = log(R_foot) (target is the
                // identity)
                let phi = rot_log(&ws.r[frame]);
                for t in 0..3 {
                    let row = 6 * widx + 3 + t;
                    let jr = &mut self.j[row * nv..(row + 1) * nv];
                    jr.fill(0.0);
                    angular_jacobian_acc(model, ws, frame, t, 1.0, jr);
                    self.bias[row] = -(BETA_WELD * hinv) * phi[t];
                }
            }
            nrows = N_WELD_ROWS;
        }

        // ── step 7: collision detection (pair-id order) ───────────────
        if self.contacts_enabled {
            narrow_phase(ws, &mut self.contacts);
        } else {
            self.contacts.clear();
        }

        // ── assembly: one normal row per active contact ───────────────
        for c in self.contacts.iter() {
            let row = nrows;
            let jr = &mut self.j[row * nv..(row + 1) * nv];
            jr.fill(0.0);
            let (ia, ib) = super::collide::PAIRS[c.pair];
            let fa = super::collide::COLLIDERS[ia].frame;
            let fb = super::collide::COLLIDERS[ib].frame;
            // J·v = n·(v_point-on-b − v_point-on-a): subtract collider
            // a's chain FIRST, then add collider b's (fixed order;
            // shared-ancestor and root columns cancel exactly)
            point_jacobian_acc(model, ws, fa, c.point, c.normal, -1.0, jr);
            point_jacobian_acc(model, ws, fb, c.point, c.normal, 1.0, jr);
            // RHS: speculative approach limit while the gap is open;
            // Baumgarte push-out beyond the slop once penetrating
            self.bias[row] = if c.gap >= 0.0 {
                -c.gap * hinv
            } else {
                (BETA_CONTACT * hinv) * (-c.gap - SLOP).max(0.0)
            };
            nrows += 1;
        }

        // ── assembly: joint-limit rows, joint order 0..28 (Divergences
        //    §15). One unilateral row per joint whose NEARER stop is
        //    within reach this substep (gap < h·ω_cap; ties → lower
        //    stop). Row +e_j (lower) / −e_j (upper): q̇ toward the stop
        //    is capped speculatively at gap/h (never crosses), and a
        //    missed stop is pushed back with the weld Baumgarte β.
        //    Without these rows the step-11 position clamp FIGHTS the
        //    weld at the stance stops (knee/elbow/wrist-roll all sit at
        //    a 0-limit at neutral): the weld solve assigns flexion
        //    rate, the clamp teleports q back — measured 5.8e-4 m of
        //    foot error per substep, a 3.7 mm steady weld violation.
        self.limset.clear();
        for p in self.limpos.iter_mut() {
            *p = -1;
        }
        for j in 0..model.nq {
            let (lo, hi) = model.limits[j];
            let gl = q[j] - lo;
            let gh = hi - q[j];
            let reach = self.h * model.omega_cap[j];
            // nearer stop only; both within reach cannot happen (every
            // limit range exceeds 2·h·ω_cap)
            let (sign, gap) = if gl <= gh {
                (1.0, gl)
            } else {
                (-1.0, gh)
            };
            if gap < reach {
                let row = nrows;
                // R23a: the dense row is introspection-only — see the
                // `LIMIT_J_ROWS` note above. `cfg!` is a compile-time
                // constant, so the shipping build drops the fill+store
                // outright.
                if LIMIT_J_ROWS {
                    let jr = &mut self.j[row * nv..(row + 1) * nv];
                    jr.fill(0.0);
                    jr[voff + j] = sign;
                }
                self.bias[row] = if gap >= 0.0 {
                    -gap * hinv
                } else {
                    (BETA_WELD * hinv) * -gap
                };
                if sign == 1.0 {
                    self.limpos[j] = row as i32;
                }
                self.limset.push((j, sign));
                nrows += 1;
            }
        }
        self.nrows = nrows;

        // R23a TEST HOOK: emulate the shipping build, where the limit
        // rows of `j` are never assembled and hold arbitrary words.
        #[cfg(test)]
        if self.poison_limit_j {
            let nd = self.weld_rows() + self.contacts.len();
            self.j[nd * nv..nrows * nv].fill(f64::NAN);
        }

        // R26 (feature "stats"): the two assembled row counts this
        // substep — measured HERE, after both assembly loops, so the
        // numbers are the ones the effective-mass and PGS stages
        // actually pay for. Compiled out by default.
        #[cfg(feature = "stats")]
        {
            let nlim = self.limset.len() as u64;
            let ncon = self.contacts.len() as u64;
            ws.stats.nlim_sum += nlim;
            ws.stats.nlim_max = ws.stats.nlim_max.max(nlim);
            ws.stats.ncon_sum += ncon;
            ws.stats.ncon_max = ws.stats.ncon_max.max(ncon);
        }

        // ── effective masses: M⁻¹Jᵀ via the substep's Cholesky factor ─
        // The 12 WELD rows are solved as a dense A_ww block, never by
        // the per-row `(b − Jv)/diag` update, so their diag is dead in
        // the shipping build (the only readers are the tests_soak debug
        // prints and the debug_assert) — cfg! is a compile-time constant
        // so release drops the branch and the 12 nv-dots with it.
        let nweld = self.weld_rows();
        const DIAG_ALL_ROWS: bool =
            cfg!(any(test, feature = "introspect", debug_assertions));
        // BATCHED multi-RHS solve (2026-08-03, R1): every row's RHS is
        // the same shared factor's, so instead of `nrows` serial dense
        // chol_solves the rows go 4 at a time through `chol_solve_x4`,
        // which loads the factor ONCE per group of 4 and keeps each
        // RHS on its own unit-stride ascending-k accumulator chain —
        // bit-for-bit the serial result (see the algo.rs contract).
        // The 12 weld rows are exactly 3 groups of 4; the contact rows
        // continue the same ladder (4 → 2 → 1) from there.
        let ndense = nweld + self.contacts.len();
        // R13: the weld block (rows 0..nweld) takes the RHS-MINOR path
        // below, which OVERWRITES those `minvjt` rows in full from `wt`
        // — seeding them from `j` here was a dead 12×35 store. `nweld`
        // is 12 or 0 (`weld_rows`), and it is exactly the same
        // `nweld == N_WELD_ROWS` test that selects the path, so this
        // skips precisely the rows that get rewritten.
        self.minvjt[nweld * nv..ndense * nv]
            .copy_from_slice(&self.j[nweld * nv..ndense * nv]);
        {
            let mut row = 0usize;
            if nweld == N_WELD_ROWS {
                // the 12 weld rows go through the RHS-MINOR form
                // (transpose in, solve, transpose out — both copies
                // exact, so the whole thing is bitwise `chol_solve`
                // twelve times). The transposed Jacobian is KEPT in
                // `jwt` afterwards: the PGS residual and the two
                // weld-null projections all want exactly this layout.
                // R13: `wt` is only the solve's scratch copy of `jwt`,
                // so both land from ONE load of `jr[i]` — the separate
                // 420-word `jwt -> wt` copy is gone.
                // R23c: `jwt` rows W_BASE_HI..W_L_LO (columns 6..23) are
                // +0.0 in ALL TWELVE weld rows — that is the R9 support
                // argument, and it holds for every solve — so the 204
                // words they occupy are already the exact `+0.0` bytes
                // this loop would re-store. `jwt` is written NOWHERE else
                // and is zero-initialised at construction, so skipping
                // them leaves identical BYTES, not merely identical
                // values: no cfg gate is needed and the `weld_jwt` test
                // hook still sees the true Jacobian.
                // `wt` is NOT skippable — `back_t` reuses it as scratch
                // at widths 8/4 and leaves a differently-shaped payload
                // in the first 8·nv words — but its skipped block is one
                // contiguous 204-word run, so it becomes a memset.
                for r in 0..N_WELD_ROWS {
                    let jr = &self.j[r * nv..(r + 1) * nv];
                    for i in 0..W_BASE_HI {
                        let x = jr[i];
                        self.jwt[i * N_WELD_ROWS + r] = x;
                        self.wt[i * N_WELD_ROWS + r] = x;
                    }
                    for i in W_L_LO..W_R_HI {
                        let x = jr[i];
                        self.jwt[i * N_WELD_ROWS + r] = x;
                        self.wt[i * N_WELD_ROWS + r] = x;
                    }
                }
                self.wt[W_BASE_HI * N_WELD_ROWS..W_L_LO * N_WELD_ROWS]
                    .fill(0.0);
                chol_solve_t::<N_WELD_ROWS>(nv, &ws.m, &mut self.wt[..N_WELD_ROWS * nv]);
                for r in 0..N_WELD_ROWS {
                    let mr = &mut self.minvjt[r * nv..(r + 1) * nv];
                    for i in 0..nv {
                        mr[i] = self.wt[i * N_WELD_ROWS + r];
                    }
                }
                row = N_WELD_ROWS;
            }
            while row + 4 <= ndense {
                chol_solve_x4(nv, &ws.m, &mut self.minvjt[row * nv..(row + 4) * nv]);
                row += 4;
            }
            if row + 2 <= ndense {
                chol_solve_x2(nv, &ws.m, &mut self.minvjt[row * nv..(row + 2) * nv]);
                row += 2;
            }
            while row < ndense {
                chol_solve(nv, &ws.m, &mut self.minvjt[row * nv..(row + 1) * nv]);
                row += 1;
            }
        }
        // LIMIT rows have the unit right-hand side sign·e_(voff+j) — the
        // assembly is `jr.fill(0.0)` plus one `jr[voff+j] = sign` store,
        // so the FORWARD pass has a closed form: rows below the pivot
        // are exactly the memset's +0.0 (see `chol_fwd_unit`'s bitwise
        // argument, which rests on §22's negated-comparison pivot floor
        // making the diagonal positive and finite unconditionally), the
        // pivot row is one divide, and the rows above it start their
        // accumulation at k = m. The BACK pass has no such structure —
        // it is dense — so it is jammed 4 columns at a time instead.
        for li in 0..self.limset.len() {
            let (jt, sign) = self.limset[li];
            let row = ndense + li;
            chol_fwd_unit(
                nv,
                &ws.m,
                voff + jt,
                sign,
                &mut self.minvjt[row * nv..(row + 1) * nv],
            );
        }
        {
            // RHS-MINOR ladder, same reasoning as the weld batch: the
            // transposes are exact copies, so the group result is
            // bitwise `chol_back` once per column.
            let mut row = ndense;
            while row + 8 <= nrows {
                Self::back_t::<8>(nv, &ws.m, &mut self.minvjt, row, &mut self.wt);
                row += 8;
            }
            if row + 4 <= nrows {
                Self::back_t::<4>(nv, &ws.m, &mut self.minvjt, row, &mut self.wt);
                row += 4;
            }
            if row + 2 <= nrows {
                chol_back_x2(nv, &ws.m, &mut self.minvjt[row * nv..(row + 2) * nv]);
                row += 2;
            }
            while row < nrows {
                chol_back(nv, &ws.m, &mut self.minvjt[row * nv..(row + 1) * nv]);
                row += 1;
            }
        }
        for row in 0..ndense {
            if DIAG_ALL_ROWS || row >= nweld {
                let jr = &self.j[row * nv..(row + 1) * nv];
                let mr = &self.minvjt[row * nv..(row + 1) * nv];
                let mut d = 0.0;
                for i in 0..nv {
                    d += jr[i] * mr[i];
                }
                debug_assert!(d > 0.0, "constraint row {row} has zero effective mass");
                self.diag[row] = d;
            }
            // pre-solve J·v* (introspection only — compiled out below)
            #[cfg(any(test, feature = "introspect"))]
            {
                let jr = &self.j[row * nv..(row + 1) * nv];
                let mut s = 0.0;
                for i in 0..nv {
                    s += jr[i] * v[i];
                }
                self.jv_pre[row] = s;
            }
        }
        // R10: a limit row is rank-1, so its effective mass is one load
        // instead of a 35-wide dot — with the mandatory `0.0 +` guard
        // (see the R10 note above; the ω-cap dedupe below already stated
        // this identity for the sign = +1 rows it reuses).
        for li in 0..self.limset.len() {
            let (jt, sign) = self.limset[li];
            let row = ndense + li;
            let d = 0.0 + sign * self.minvjt[row * nv + voff + jt];
            debug_assert!(d > 0.0, "constraint row {row} has zero effective mass");
            self.diag[row] = d;
            // pre-solve J·v* stays dense: introspection-only, so it is
            // already compiled out of the shipping build.
            #[cfg(any(test, feature = "introspect"))]
            {
                let jr = &self.j[row * nv..(row + 1) * nv];
                let mut s = 0.0;
                for i in 0..nv {
                    s += jr[i] * v[i];
                }
                self.jv_pre[row] = s;
            }
        }

        // ── weld-block effective mass A_ww = J_w M⁻¹ J_wᵀ, factored
        //    once per substep (Divergences §15: per-row Gauss-Seidel
        //    leaves ~0.14 m/s weld residual after 12 sweeps at the
        //    straight-knee stance — the two feet's vertical rows are
        //    near-parallel through the root — so the 12 bilateral rows
        //    are solved as an exact dense block instead) ─────────────
        if nweld > 0 {
            // UNROLL-AND-JAM by 4 over c (2026-08-03): the 78 dots keep
            // their own ascending-i accumulator, so every partial sum —
            // and hence every A_ww entry — is bit-for-bit what the
            // one-dot-at-a-time form produced; only mr's re-traversal
            // count drops (78 → ~24) and all four j rows stay
            // UNIT-STRIDE, which the r-inner interchange would have
            // destroyed (see the weld PGS note below: strided loads cost
            // more here than the traversals they save).
            for r in 0..nweld {
                let mut s = [0.0f64; N_WELD_ROWS];
                {
                    // R9: only the weld Jacobian's support is visited —
                    // columns 6..23 are +0.0 in all 12 rows.
                    let mr = &self.minvjt[r * nv..(r + 1) * nv];
                    weld_support_dot!(s, self.jwt, mr);
                }
                // The whole 12-lane row is computed and the c < r lanes
                // discarded: one vector FMA per i beats the 12 − r
                // scalar ones the jammed triangle needed, and the lanes
                // that ARE kept carry the same ascending-i chain as
                // before. Both triangles still come from ONE value, so
                // A_ww stays bitwise symmetric by construction.
                for c in r..nweld {
                    self.aww[r * nweld + c] = s[c];
                    self.aww[c * nweld + r] = s[c];
                }
            }
            // §22 pivot floor here too (second former panic site; the
            // .cu chol_factor_n template serves both call sites): a
            // healthy A_ww factors bitwise-identically (0 hits), a
            // degenerate one is clamped + counted instead of panicking.
            ws.pd_hits += chol_factor_floored(nweld, &self.aww, &mut self.lww) as u64;
        }

        // ── limit rows: project their update directions into the WELD
        //    NULL SPACE (Divergences §15): c = M⁻¹Jᵀ − M⁻¹J_wᵀ·A_ww⁻¹·
        //    (J_w M⁻¹ Jᵀ), so limit/cap impulses can never break the
        //    welds (J_w·c = 0 exactly). Plain per-row GS between a
        //    pinned joint and the weld block does NOT converge in 12
        //    iterations — the directions are near-parallel in the M
        //    metric (knee motion IS foot motion on a straight leg) and
        //    the leftover fight was measured at up to 38 mm of foot
        //    drift under full random gears. A direction with composite
        //    mobility ≤ COMPOSITE_EPS × its plain mobility is immobile
        //    given the welds (they already forbid the motion): its
        //    diag is set to +∞, which zeroes every subsequent update
        //    deterministically.
        let ncon = self.contacts.len();
        if nweld > 0 {
            // the 12-wide jams below assume the 12-row weld block
            debug_assert_eq!(nweld, N_WELD_ROWS);
            let split = (nweld + ncon) * nv;
            let (head, tail) = self.minvjt.split_at_mut(split);
            let nlim = self.limset.len();
            // R16 (2026-08-06): limit rows are INDEPENDENT here — row li
            // reads and writes only its own `minvjt` column — so the two
            // SHAREABLE stages go two at a time: ONE `jwt` traversal
            // feeds 2×12 dot accumulators, and the two 12-vectors go
            // through ONE `chol_solve_x2` against the same A_ww factor
            // (the same batch contract the weld/contact ladder uses).
            // Each column still sees its own terms in its own ascending
            // order, so this is bit-for-bit the one-row-at-a-time result.
            // MEASURED +0.42% (22,300 → 22,393 env-steps/s at N = 256,
            // 4 interleaved ABBA pairs) — and this is a LOWER BOUND for
            // this class: the micro drives random gears while the benched
            // policy parks joints ON their stops, so the bench assembles
            // more limit rows per substep (see the R10 note above).
            // The projection axpy is deliberately NOT shared and width 4
            // is deliberately not used — both measured negative, see the
            // R16 NEGATIVE block by `weld_axpy12!`.
            let mut lj = 0usize;
            while lj + 2 <= nlim {
                let (t0, t1) = tail[lj * nv..(lj + 2) * nv].split_at_mut(nv);
                let mut s = [0.0f64; 2 * N_WELD_ROWS];
                {
                    let (s0, s1) = s.split_at_mut(N_WELD_ROWS);
                    weld_support_dot2!(s0, s1, self.jwt, t0, t1);
                }
                chol_solve_x2(nweld, &self.lww, &mut s);
                let (x0, x1) = s.split_at(N_WELD_ROWS);
                weld_axpy12!(t0, head, x0, nv, -=);
                weld_axpy12!(t1, head, x1, nv, -=);
                for (k, mr) in [&*t0, &*t1].into_iter().enumerate() {
                    let (jt, sign) = self.limset[lj + k];
                    let row = nweld + ncon + lj + k;
                    let dnew = 0.0 + sign * mr[voff + jt];
                    self.diag[row] = if dnew > COMPOSITE_EPS * self.diag[row] {
                        dnew
                    } else {
                        f64::INFINITY // immobile given welds — row inert
                    };
                }
                lj += 2;
            }
            // remainder row (nlim is odd), the scalar path verbatim
            for li in lj..nlim {
                let (jt, sign) = self.limset[li];
                let row = nweld + ncon + li;
                let mr = &mut tail[li * nv..(li + 1) * nv];
                // UNROLL-AND-JAM (2026-08-03; width re-probed 2026-08-06,
                // R15): mr is traversed ONCE each way while every
                // accumulator keeps its own ascending-i order and every
                // mr[i] takes its 12 decrements in w = 0..11 order —
                // bit-for-bit the one-row-at-a-time result. (A full w/i
                // interchange was measured SLOWER, see the weld PGS note.)
                let mut s = [0.0f64; N_WELD_ROWS];
                weld_support_dot!(s, self.jwt, mr); // R9 support only
                self.rw[..N_WELD_ROWS].copy_from_slice(&s);
                chol_solve(nweld, &self.lww, &mut self.rw);
                let mut x = [0.0f64; N_WELD_ROWS];
                x.copy_from_slice(&self.rw[..N_WELD_ROWS]);
                weld_axpy12!(mr, head, x, nv, -=);
                // R10: rank-1 row again — the projected mobility is the
                // pivot entry of the projected column (guard included).
                let dnew = 0.0 + sign * mr[voff + jt];
                self.diag[row] = if dnew > COMPOSITE_EPS * self.diag[row] {
                    dnew
                } else {
                    f64::INFINITY // immobile given welds — row inert
                };
            }
        }

        // ── step 8: exactly 12 iterations, fixed order, no warm start;
        //    per iteration: (a) weld block solved EXACTLY for the
        //    current residual, (b) contact rows projected Gauss-Seidel
        //    in pair-id order. Step 9 (v ← v* + M⁻¹Jᵀλ) is realized by
        //    the incremental impulse propagation. ────────────────────
        for r in self.lambda[..nrows].iter_mut() {
            *r = 0.0;
        }
        for c in self.capcached.iter_mut() {
            *c = false;
        }
        // R26 (feature "stats"): ω-cap column solves for THIS substep.
        // The cap pass runs inside every one of the 12 PGS iterations
        // but `capcached` is only cleared here, so the per-substep total
        // is at most nq — the counter has to live outside the iteration
        // loop to see that, and it is flushed once below.
        #[cfg(feature = "stats")]
        let mut cap_solves = 0u64;
        for _iter in 0..PGS_ITERS {
            // (a) welds: Δλ_w = A_ww⁻¹ (b_w − J_w v), applied together.
            // (An r/i loop INTERCHANGE was tried here 2026-08-03 —
            // bitwise-clean, but 3.9% SLOWER: at nv = 35 the whole
            // working set is already L1-resident, so there is no
            // streaming win to buy, while the r-inner form's nv-strided
            // loads destroy the unit-stride AVX2 vectorization of these
            // dots/axpys. Reverted.)
            //
            // What DOES win is UNROLL-AND-JAM (2026-08-03, R1; widths
            // re-probed 2026-08-06, R15): i stays inner and unit-stride,
            // one accumulator per weld row, so `v` is traversed ONCE per
            // half instead of 12× — the dot through `weld_support_dot!`,
            // the axpy through `weld_axpy12!` (see there for the width
            // ladder and its measurements). BITWISE: each dot keeps its
            // own ascending-i chain, and in the axpy each v[i] still
            // takes its 12 increments in r = 0..11 order (the jam only
            // holds v[i] in a register across them) — bit-for-bit the
            // row-at-a-time result. This is the hottest loop in the
            // substep: 12 iterations × 12 rows × two nv-wide passes.
            if nweld > 0 {
                debug_assert_eq!(nweld, N_WELD_ROWS);
                // 12 dots against one shared v, in the TRANSPOSED weld
                // layout: 12 lanes, each accumulating its own row's
                // terms over ascending i exactly as the scalar dot did.
                let mut s = [0.0f64; N_WELD_ROWS];
                weld_support_dot!(s, self.jwt, v); // R9 support only
                for r in 0..nweld {
                    self.rw[r] = self.bias[r] - s[r];
                }
                chol_solve(nweld, &self.lww, &mut self.rw);
                let mut d = [0.0f64; N_WELD_ROWS];
                d.copy_from_slice(&self.rw[..N_WELD_ROWS]);
                for r in 0..N_WELD_ROWS {
                    self.lambda[r] += d[r];
                }
                weld_axpy12!(v, self.minvjt, d, nv, +=);
            }
            // (b)+(c) contacts (pair-id order) then joint-limit rows
            // (joint order): projected GS, λ ≥ 0.
            // R10: the one loop is SPLIT at `ndense` — same rows, same
            // order (contacts before limits, each in its own assembly
            // order), but the limit half reads its residual straight off
            // `v` instead of running a 35-wide dot against a row that is
            // one ±1.0 in a field of +0.0. This is the hot site: 12 PGS
            // iterations × ~6 limit rows per substep. (The `jv` argument
            // is the only thing that differs — the λ update and the
            // impulse propagation are the contact path's, verbatim.)
            for row in nweld..ndense {
                let jr = &self.j[row * nv..(row + 1) * nv];
                let mut jv = 0.0;
                for i in 0..nv {
                    jv += jr[i] * v[i];
                }
                Self::pgs_row_update(
                    row,
                    nv,
                    jv,
                    &self.bias,
                    &self.diag,
                    &mut self.lambda,
                    &self.minvjt,
                    v,
                );
            }
            for li in 0..self.limset.len() {
                let (jt, sign) = self.limset[li];
                let jv = 0.0 + sign * v[voff + jt]; // guard: see R10 note
                Self::pgs_row_update(
                    ndense + li,
                    nv,
                    jv,
                    &self.bias,
                    &self.diag,
                    &mut self.lambda,
                    &self.minvjt,
                    v,
                );
            }
            // (d) ω-cap pass, joint order 0..28 (step-11's cap rows
            // folded in — Divergence §13's anticipation): the weld
            // block may assign super-cap joint rates whose h²·q̇²
            // integration error blows the weld (measured 38 mm under
            // full random gears); projecting the caps INSIDE the
            // iteration lets the solve redistribute while integration
            // only ever sees capped rates. Momentum-consistent
            // internal impulses along the WELD-CONSISTENT M⁻¹e_j
            // (weld component eliminated like the limit rows), lazily
            // solved and cached per substep. A joint immobile given
            // the welds (composite mobility ≤ COMPOSITE_EPS × plain)
            // is skipped — the welds own its rate.
            for jt in 0..model.nq {
                let vj = voff + jt;
                let cap = model.omega_cap[jt];
                let target = v[vj].clamp(-cap, cap);
                if target != v[vj] {
                    if !self.capcached[jt] && self.limpos[jt] >= 0 {
                        // DEDUPE (2026-08-03, R1 item 1f): joint jt has
                        // a LOWER-stop limit row this substep, whose
                        // right-hand side is +e_vj — the very column
                        // this pass is about to solve — and which has
                        // already been through the IDENTICAL weld-null
                        // projection (same expression tree, same weld
                        // rows, same A_ww factor). So that row's
                        // `minvjt` IS this joint's `capcol`, bit for
                        // bit, and its `diag` IS `capmass`:
                        //   · pre-projection the two solves are the
                        //     same call on the same RHS;
                        //   · `diag[row] = Σᵢ jr[i]·mr[i]` with jr =
                        //     +e_vj collapses to `1.0 · mr[vj]` (the
                        //     other terms add ±0.0 to a non-zero
                        //     partial sum, which is exact) = the cap
                        //     pass's `plain`, so the COMPOSITE_EPS test
                        //     and its ∞ verdict are the same test;
                        //   · the projection is applied to `diag[row]`
                        //     by the limit block above exactly as the
                        //     cap pass would apply it to `capmass`.
                        // ONLY the +1 sign is reused: a −e_vj row would
                        // need a negation, and −(+0.0) = −0.0 ≠ +0.0
                        // makes that NOT bitwise (an exactly-cancelling
                        // entry flips its zero sign), which is why
                        // `limpos` records lower stops alone.
                        let lr = self.limpos[jt] as usize;
                        self.capcol[jt * nv..(jt + 1) * nv]
                            .copy_from_slice(&self.minvjt[lr * nv..(lr + 1) * nv]);
                        self.capmass[jt] = self.diag[lr];
                        self.capcached[jt] = true;
                    }
                    if !self.capcached[jt] {
                        {
                            // unit RHS +e_vj: the sparse forward pass
                            // (rows below vj are exactly the memset's
                            // +0.0) then the dense back substitution —
                            // bitwise the fill/store/chol_solve it
                            // replaces, see `chol_fwd_unit`.
                            let col = &mut self.capcol[jt * nv..(jt + 1) * nv];
                            chol_solve_unit(nv, &ws.m, vj, 1.0, col);
                        }
                        // R26: counted INSIDE the engage branch — only
                        // a real chol_solve_unit call above bumps it
                        // (the `limpos` dedupe path and every cached
                        // re-entry are deliberately not counted).
                        #[cfg(feature = "stats")]
                        {
                            cap_solves += 1;
                        }
                        let plain = self.capcol[jt * nv + vj];
                        if nweld > 0 {
                            debug_assert_eq!(nweld, N_WELD_ROWS);
                            // same jam as the limit-row projection: one
                            // accumulator per weld row, each still
                            // summing over i ASCENDING, and each col[i]
                            // taking its 12 decrements in w = 0..11
                            // order — bitwise the per-row form.
                            let mut s = [0.0f64; N_WELD_ROWS];
                            {
                                let col = &self.capcol[jt * nv..(jt + 1) * nv];
                                weld_support_dot!(s, self.jwt, col); // R9
                            }
                            self.rw[..N_WELD_ROWS].copy_from_slice(&s);
                            chol_solve(nweld, &self.lww, &mut self.rw);
                            // weld minvjt rows live at the head of the
                            // buffer — disjoint from capcol, so the two
                            // fields can be borrowed at once
                            let (capcol, minvjt) = (&mut self.capcol, &self.minvjt);
                            let col = &mut capcol[jt * nv..(jt + 1) * nv];
                            let mut x = [0.0f64; N_WELD_ROWS];
                            x.copy_from_slice(&self.rw[..N_WELD_ROWS]);
                            weld_axpy12!(col, minvjt, x, nv, -=);
                        }
                        let dnew = self.capcol[jt * nv + vj];
                        self.capmass[jt] = if dnew > COMPOSITE_EPS * plain {
                            dnew
                        } else {
                            f64::INFINITY // immobile given welds
                        };
                        self.capcached[jt] = true;
                    }
                    if self.capmass[jt].is_finite() {
                        let col = &self.capcol[jt * nv..(jt + 1) * nv];
                        let lam = (target - v[vj]) / self.capmass[jt];
                        for i in 0..nv {
                            v[i] += col[i] * lam;
                        }
                        v[vj] = target; // exact — the cap is a hard bound
                    }
                }
            }
        }
        // R26 (feature "stats"): one flush per substep, after all 12
        // PGS iterations — sum for the mean, max for the worst case.
        #[cfg(feature = "stats")]
        {
            ws.stats.cap_solves_sum += cap_solves;
            ws.stats.cap_solves_max = ws.stats.cap_solves_max.max(cap_solves);
        }
    }
}

/// Accumulate `sign · (n · J_point)` into a dense row: the Jacobian of
/// the world velocity of point `p` rigidly attached to body `frame`,
/// projected on direction `n`. Joint columns first (self → root walk),
/// then the free-base columns (fixed order).
fn point_jacobian_acc(
    model: &Model,
    ws: &Workspace,
    frame: usize,
    p: [f64; 3],
    n: [f64; 3],
    sign: f64,
    row: &mut [f64],
) {
    let voff = model.voff();
    let mut k = frame as i32;
    while k >= 0 {
        let fr = &model.frames[k as usize];
        if fr.coord >= 0 {
            let col = voff + fr.coord as usize;
            // unit q̇ of joint k moves p by axw × (p − p_k)
            let w = cross3(ws.axw[k as usize], sub3(p, ws.p[k as usize]));
            row[col] += sign * dot3(n, w);
        }
        k = fr.parent;
    }
    if model.free_base {
        // root linear: v_point = v_lin → n directly
        for t in 0..3 {
            row[t] += sign * n[t];
        }
        // root angular: n·(e_t × r) = (r × n)_t with r = p − p_root
        let rxn = cross3(sub3(p, ws.p0), n);
        for t in 0..3 {
            row[3 + t] += sign * rxn[t];
        }
    }
}

/// Accumulate `sign · (world angular velocity component `axis_t`)` of
/// body `frame` into a dense row (weld angular rows).
fn angular_jacobian_acc(
    model: &Model,
    ws: &Workspace,
    frame: usize,
    axis_t: usize,
    sign: f64,
    row: &mut [f64],
) {
    let voff = model.voff();
    let mut k = frame as i32;
    while k >= 0 {
        let fr = &model.frames[k as usize];
        if fr.coord >= 0 {
            let col = voff + fr.coord as usize;
            row[col] += sign * ws.axw[k as usize][axis_t];
        }
        k = fr.parent;
    }
    if model.free_base {
        row[3 + axis_t] += sign;
    }
}

/// Full v10 substep: P1 pipeline + the constraint stage plugged into
/// the §3 seam (steps 7–9 between the unconstrained-velocity solve and
/// position integration).
pub fn substep_constrained(
    model: &Model,
    st: &mut State,
    tau_act: &[f64],
    h: f64,
    ws: &mut Workspace,
    cs: &mut ConstraintSet,
) {
    substep_constrained_wrench(model, st, tau_act, &ZERO_WRENCH, h, ws, cs);
}

/// `substep_constrained` with an applied root wrench (test harnesses:
/// the P2 static-stance protocol's full 35-DoF RNEA compensation).
pub fn substep_constrained_wrench(
    model: &Model,
    st: &mut State,
    tau_act: &[f64],
    root_wrench: &[f64; 6],
    h: f64,
    ws: &mut Workspace,
    cs: &mut ConstraintSet,
) {
    cs.h = h;
    let mut hook =
        |m: &Model, w: &mut Workspace, q: &[f64], v: &mut [f64]| cs.solve(m, w, q, v);
    substep_wrench(model, st, tau_act, root_wrench, h, ws, Some(&mut hook));
}

/// Harness variant with per-joint IMPLICIT damping (oracle muscle-tone
/// kd — see step.rs `substep_opts` for why explicit application is
/// unstable). The env never uses this.
pub fn substep_constrained_damped(
    model: &Model,
    st: &mut State,
    tau_act: &[f64],
    implicit_damp: &[f64],
    h: f64,
    ws: &mut Workspace,
    cs: &mut ConstraintSet,
) {
    cs.h = h;
    let mut hook =
        |m: &Model, w: &mut Workspace, q: &[f64], v: &mut [f64]| cs.solve(m, w, q, v);
    super::step::substep_opts(
        model,
        st,
        tau_act,
        &ZERO_WRENCH,
        Some(implicit_damp),
        h,
        ws,
        Some(&mut hook),
    );
}
