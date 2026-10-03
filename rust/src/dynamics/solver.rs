//! SPEC-v10 §3 steps 7–9: constraint assembly + the mixed
//! block/projected-Gauss-Seidel solve (P2 revision — Divergences §15).
//!
//! Row layout (NORMATIVE; env v11 contact mode, the only mode since the
//! 2026-09-27 weld removal — the v10 12-row foot-weld block is gone):
//!   3 rows per ACTIVE sole corner, corners in the contact spec §2
//!   normative order (foot_l then foot_r, (x−,z−) (x−,z+) (x+,z−)
//!   (x+,z+) within a foot), rows normal / friction x̂ / friction ẑ,
//!   then one normal row per ACTIVE contact, in pair-id order (§2),
//!   then one unilateral JOINT-LIMIT row per joint whose nearer stop is
//!   within h·ω_cap, joint order 0..28 (folded into the main solve —
//!   the §3 anticipation "P2 may fold these rows"; see below).
//!
//! Solve: exactly 12 iterations, no warm starting (λ = 0 each substep);
//! each iteration = (a) the ground block (per contacting foot, its
//! corners' rows in the §3 order, projected GS with the box friction
//! clamp), then (b) contact + limit rows projected Gauss-Seidel to
//! λ ≥ 0, in row order, then (c) the ω-cap pass. Effective masses and
//! impulse propagation via M⁻¹Jᵀ columns solved with the substep's
//! Cholesky factor (ws.l, audit-#10 tree order). Why limit rows in the
//! solve: the stance parks knee/elbow/wrist-roll exactly ON a 0-stop,
//! and a post-hoc clamp fights the rest of the solve (it assigns
//! flexion rate, the clamp teleports q back).
//!
//! Right-hand sides (J·v ≥ b, λ ≥ 0 unilaterals):
//!   contacts: b = (β_c/h)·max(penetration − slop, 0) when penetrating,
//!             β_c = 0.2, slop 0.005; for a still-positive gap inside
//!             the margin, b = −gap/h (speculative activation: approach
//!             is allowed at exactly the rate that closes the remaining
//!             gap this substep — Divergences §16; without this term
//!             the 1 cm margin would act as a hard force field).
//!   limits:   same shape as contacts with slop 0 and β_limit = 0.2
//!             pushback (row ±e_j toward the interior).
//!   ground:   normal rows take the contact shape verbatim (β_c, slop);
//!             friction rows b = 0 and a BOX projection |λ_t| ≤ MU·λ_n
//!             against the same sweep's λ_n (contact spec §3).
//!
//! Step 11 (ω caps + limit clamps) REMAINS as the outer guarantee —
//! with the solver limit-aware its impulses are near-null in ordinary
//! motion. Full normative substep order: steps 1–6, 7–9 (this module:
//! ground block + contacts pair-id order + limits joint order, 12
//! iterations), 10 integrate, 11 clamp passes joint order 0..28.
//!
//! Loop bounds and evaluation order are fixed throughout — the
//! torch/CUDA mirrors replicate these expression trees literally.
//!
//! ORDER-DEPENDENCE (2026-08-05, R9 — the rule the CUDA kernel header
//! already carries): jam widths and any change previously measured at
//! under ~1% must be RE-PROBED against the build that would ship them
//! before being called dead — a gain measured alone can vanish (or turn
//! negative) once its neighbours change the loop's register economics.
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
//!   · **R23c** a permanently-+0.0 column block of the (since
//!     removed, 2026-09-27) weld Jacobian: **+1.13%** alone;
//!   · **R23e** `rnea`'s `ws.vo` chain (algo.rs): **+0.45%** alone
//!     (22,330 → 22,430, 6/6);
//!   · **R23b** the mass matrix's upper triangle (algo.rs) — the
//!     free-base half ONLY: **+0.28%** on top of the package
//!     (12 pairs, two order-balanced batches, 10/12 wins);
//!   · **R23d** (step.rs `ws.sol`) **−0.84%** and **R23f** (a stack
//!     copy of the since-removed weld block's solve scratch) **−2.71%**:
//!     both REVERTED, both bitwise-clean.
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

use crate::Real;
use super::algo::{
    tc_back_t, tc_fwd_mask_t, tc_fwd_t, tc_fwd_unit_lane, tc_solve_unit_v, TreeChol,
};
use super::collide::{
    ground_phase, narrow_phase, Contact, GroundContact, GROUND_FRAMES, NUM_GROUND_CORNERS,
    NUM_PAIRS, SLOP,
};
use super::fk::{cross3, dot3, sub3};
use super::model::{Model, F_L_FOOT, F_R_FOOT};
use super::step::{substep_wrench, ZERO_WRENCH};
use super::{State, Workspace};

/// PGS iteration count (§3 step 8, normative).
pub const PGS_ITERS: usize = 12;
/// Baumgarte gains (§3 step 8, normative). `BETA_LIMIT_ROW` is the joint-
/// limit rows' push-back gain (the v10 weld gain, which the limit rows
/// shared; the value is unchanged).
pub const BETA_LIMIT_ROW: Real = 0.2;
pub const BETA_CONTACT: Real = 0.2;

// ── env v11 GROUND CONTACT (unilateral-contact spec §3) ──────────────
// Per ACTIVE sole corner one unilateral normal row plus two BOX-clamped
// friction rows. The rows come first (before the self-collision
// normals), so the §2 pair-id row order is untouched.

/// Coulomb friction coefficient of the ground (contact spec §3,
/// normative). No restitution — this is a velocity-level PGS at rest
/// contact.
///
/// DO NOT RAISE above 0.9 without replacing the friction projection:
/// a bump to 1.1 was attempted (2026-08-17, sneaker-rubber realism)
/// and the per-row box-clamped Gauss-Seidel LOSES STICK above
/// μ ≈ 0.9 on the standing body — the sweep limit-cycles instead of
/// converging, and loaded corners creep at 2-3 cm per 0.1 s under a
/// 100 N push (measured; iteration count does not fix it: 12/24/48/96
/// iters all creep, while μ = 0.9 sticks at < 1 mm). Raising μ past
/// the cliff makes friction WEAKER in practice, not stronger. A
/// coupled cone projection per corner would be needed first.
pub const MU: Real = 0.9;

/// Rows per active corner: 1 normal + 2 friction (contact spec §3).
pub const GROUND_ROWS_PER_CORNER: usize = 3;

/// World row directions of a corner's three rows, in the NORMATIVE
/// order (contact spec §3): normal ŷ, then friction x̂ and ẑ. The ground
/// is static, so a row is the corner's own point Jacobian projected on
/// the direction — no second body to subtract (contrast the §2
/// self-collision rows, which contract two chains).
#[rustfmt::skip]
const GROUND_DIRS: [[Real; 3]; GROUND_ROWS_PER_CORNER] = [
    /* N  */ [0.0, 1.0, 0.0],
    /* T1 */ [1.0, 0.0, 0.0],
    /* T2 */ [0.0, 0.0, 1.0],
];

// ── audit #9 (2026-09-26): GROUND ROWS FROM EACH FOOT'S 6-D SPATIAL
//    JACOBIAN. A corner row is `point_jacobian_acc(foot, p_c, e_a)`,
//    i.e. the world velocity of the corner point projected on e_a:
//        j·v = e_a·(v_o + ω × r) = e_a·v_o + (r × e_a)·ω,  r = p_c − p_f,
//    an EXACT linear combination c = [e_a; r × e_a] of the foot's six
//    spatial rows J_f (3 linear rows at the foot origin p_f, then 3 world
//    angular rows — the v10 weld rows of that foot). So instead of
//    M⁻¹Jᵀ for up to 24 corner rows the solve takes M⁻¹J_fᵀ once per
//    CONTACTING foot (6 columns each, ≤ 12), and every row quantity comes
//    from the 6-D foot objects:
//      · Φ_f = J_f M⁻¹ J_fᵀ (6×6; entry (r, c ≥ r) = J_f[c]·G_f[r],
//        mirrored to the other triangle);
//      · u = Φ_f c (row's 6-D mobility image), diag = c·u;
//      · PGS per foot per sweep: V = J_f v (6 support dots), then each
//        active corner's rows in the §3 order with residual c·V, the
//        SAME per-row clamps (λ_n ≥ 0 then the box against this sweep's
//        λ_n), and on a non-zero Δλ: Λ += c·Δλ, V += u·Δλ; finally
//        v += G_f Λ (6 axpys) before the next foot / the pair rows.
//    Exact arithmetic: J_f(v + G_f Λ) = V + Φ_f Λ, so every row sees the
//    residual the per-row form produced, and v leaves the foot block with
//    the per-row form's value — ROW-BY-ROW GAUSS-SEIDEL, ORDER, CLAMPS
//    AND ITERATION COUNT ARE UNCHANGED; only rounding moves (SPEC-v10
//    Divergences §24). The CUDA kernel mirrors this term for term.
/// Per ground row t (normal ŷ, friction x̂, friction ẑ): its linear axis
/// a and the two angular indices b1 < b2 where r × e_a is non-zero —
///   r × e_1 = (−r2, 0,  r0)   r × e_0 = (0,  r2, −r1)
///   r × e_2 = ( r1, −r0, 0)
/// (the coefficient values are set in `assemble_ground_rows`).
const GROUND_SPATIAL: [(usize, usize, usize); GROUND_ROWS_PER_CORNER] =
    [(1, 0, 2), (0, 1, 2), (2, 0, 1)];
/// Spatial rows per foot (3 linear + 3 angular).
const FOOT_ROWS: usize = 6;
/// audit #9: the dense per-corner ground rows have no reader left in the
/// shipping build (every consumer is 6-D) — assembled for `cargo test` /
/// debug introspection only (the `LIMIT_J_ROWS` idiom).
const GROUND_J_ROWS: bool = cfg!(any(test, debug_assertions));

/// item 3 (2026-09-16): widths of the RHS-MINOR batched solves for the
/// dense rows (ground + pair normals) and the limit columns;
/// 0 = off (the pre-item-3 row-major `chol_solve_x4` / `chol_fwd_unit`
/// ladders). `wt` is sized for the widest user.
/// MEASURED on the pod (EPYC 7573X, replay f32_256x2000, 14 threads, vs
/// the item-2 build, ABBA medians 8.79–8.86 s): dense alone W1 = 8
/// −12%, 12 −14%, 16 −12% (the 8-row remainder falls to x4), 24 −18%;
/// 24 then 12 −19.8%, 24 then 8 −19.9%. Limits alone: W = 4 −0.6%, but
/// 8 +0.7%, 12 +0.9%, 16 +1.3% SLOWER — the shared forward from the
/// group's first pivot wastes the sparse per-column starts, and the
/// back pass was already batched at 8. Do not widen LIMIT_T_W without a
/// new argument. Shipped 24/12/4: −21.0% (8.81 → 6.96–6.99 s).
/// audit #10 (2026-09-26) supplied that argument: the limit columns'
/// forward passes are now per-lane ancestor chains (nothing is shared
/// from a group pivot any more) and only the dense back pass is batched
/// — the shape the pre-item-3 back ladder already ran at 8 — so the
/// limit ladder is LIMIT_T_W = 8 then 4 / 2 / 1; the pair-row remainder
/// ladder below DENSE_T_W2 is 4 / 2 / 1, RHS-minor as well.
const DENSE_T_W1: usize = 24;
const DENSE_T_W2: usize = 12;
const LIMIT_T_W: usize = 8;
const WT_ROWS: usize = {
    // the two feet's 12 spatial columns share the scratch too
    let a = if DENSE_T_W1 > 2 * FOOT_ROWS { DENSE_T_W1 } else { 2 * FOOT_ROWS };
    let a = if DENSE_T_W2 > a { DENSE_T_W2 } else { a };
    if LIMIT_T_W > a { LIMIT_T_W } else { a }
};

/// 24 = all 8 corners active, 3 rows each.
pub const N_GROUND_ROWS: usize = NUM_GROUND_CORNERS * GROUND_ROWS_PER_CORNER;

/// Max rows: 24 ground + 55 contact + ≤ 1 joint-limit row per joint
/// (Divergences §15 — the nearer stop only, when within reach). A buffer
/// bound, not a row count.
pub const MAX_ROWS: usize = N_GROUND_ROWS + NUM_PAIRS + 29; // 108

// ── R9 (2026-08-05): FOOT-JACOBIAN SUPPORT — the port of the CUDA
//    kernel's K5 sparsity (first derived for the v10 weld rows, removed
//    2026-09-27; the ground block below inherits the same support).
//
// A foot row is assembled as `jr.fill(0.0)` followed by
// `point_jacobian_acc` / `angular_jacobian_acc`, and those two write
// ONLY (a) the free-base columns `row[0..3]` / `row[3 + t]` and (b)
// `row[voff + fr.coord]` for each frame on the foot's ancestor chain.
// Nothing else is ever touched, so every other column keeps the
// memset's +0.0 EXACTLY.
//
//   F_L_FOOT = 23 → chain 23,22,21,20,19,18 → coords 22..17
//                 → v-columns voff+17 .. voff+22 = 23..28
//   F_R_FOOT = 29 → chain 29,28,27,26,25,24 → coords 28..23
//                 → v-columns 29..34
//
// so, with voff = 6 and nv = 35:
//   left-foot rows  support = {0..6} ∪ {23..29}
//   right-foot rows support = {0..6} ∪ {29..35}
const W_BASE_HI: usize = 6; // free-base columns 0..6
const W_L_LO: usize = 23; // left-foot chain
const W_L_HI: usize = 29;
const W_R_LO: usize = 29; // right-foot chain
const W_R_HI: usize = 35;
// Pin the layout the segments are derived from (mirrors the
// `static_assert` in beat_arena_cuda.cu).
const _: () = {
    assert!(F_L_FOOT == 23 && F_R_FOOT == 29);
    assert!(super::model::NQ == 29 && super::model::NV == W_R_HI);
    assert!(W_BASE_HI == 6 && W_L_LO == 23 && W_L_HI == 29 && W_R_LO == 29);
};

// ── bench-opt item 2 (2026-09-16): GROUND-ROW SUPPORT — R9's argument
//    carried over to the contact-mode ground block.
//
// A ground row is assembled like a foot linear row: `jr.fill(0.0)`
// then ONE `point_jacobian_acc` on the corner's own foot frame (the ground
// is static — no second chain). `ground_phase` only ever emits corners on
// `GROUND_FRAMES` = [F_L_FOOT, F_R_FOOT], foot_l corners FIRST, so the
// block is a left-foot prefix (support {0..6} ∪ {23..29}) followed by a
// right-foot suffix (support {0..6} ∪ {29..35}); every other column holds
// the memset's +0.0. Consumers in the shipping build: the per-substep
// effective mass `Σ jr·mr` and the PGS residual `Σ jr·v` (3 rows per
// active corner × PGS_ITERS). Both are single scalar dots seeded +0.0.
// BITWISE: a skipped term contributes (+0.0)·x = ±0.0, and `y + (±0.0)
// == y` for every y reachable here: an accumulator seeded +0.0 can never
// become −0.0 (x + y = −0.0 under RN needs x = y = −0.0), so the −0.0
// channel is unreachable and the accumulator is unchanged. The kept
// terms keep their ascending-i order — no reassociation. (Finite
// trajectories, work-order §10: 0·NaN = 0·∞ = NaN.)
// The column segment is a compile-time constant of each loop (the
// dead-store rule 2 above): the prefix/suffix split is found once per
// solve, not selected per row. Pinned by `ground_support_dot_bitwise`.
// MEASURED on the pod (EPYC 7573X, f32 native build, replay trace
// f32_256x2000 = 256 bench envs × 2,000 leaf-policy ticks, ABBA ×2):
// step_into **−15.1…−16.2%** at 14 threads (10.40–10.54 → 8.81–8.85 s)
// and **−16.8…−17.3%** at 1 thread (33.30 → 27.54–27.70 s over 500
// ticks), replay BITWISE IDENTICAL at every tick. The ground dots are
// strict-FP scalar reductions (not vectorizable), so 35 → 12 terms is
// a straight 3× on the hottest scalar chain of contact mode.
const _: () = {
    assert!(GROUND_FRAMES[0] == F_L_FOOT && GROUND_FRAMES[1] == F_R_FOOT);
};

/// `Σᵢ jr[i]·x[i]` over the ground-row support {0..W_BASE_HI} ∪ {LO..HI},
/// ascending i, accumulator seeded +0.0 — bitwise the 35-wide dot for a
/// ground row of the foot whose chain columns are LO..HI (see above).
#[inline(always)]
fn ground_support_dot<const LO: usize, const HI: usize>(jr: &[Real], x: &[Real]) -> Real {
    let (jr, x) = (&jr[..HI], &x[..HI]);
    let mut s = 0.0;
    for i in 0..W_BASE_HI {
        s += jr[i] * x[i];
    }
    for i in LO..HI {
        s += jr[i] * x[i];
    }
    s
}

// ── R10 (2026-08-06): LIMIT-ROW CLOSED FORMS — the rank-1 sibling of
//    R9's sparsity argument (the CUDA kernel has carried it since the
//    unit-RHS forward pass: beat_arena_cuda.cu's PGS reads
//    `jv = s.l_sign[li] * v[VOFF + j]` instead of a 35-wide dot).
//
// A joint-limit row is assembled as `jr.fill(0.0)` plus ONE store
// `jr[voff + j] = sign` (±1.0). So every contraction `Σᵢ jr[i]·x[i]`
// over such a row is rank-1 and has the closed form below. Two
// consumers collapse: the pre-solve effective mass `Σ jr·mr` and — 12
// times per substep, the hot one — the PGS residual `Σ jr·v`.
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
// R23a (2026-08-07): with every contraction closed-form, the dense
// limit ROW ITSELF has no reader left in the shipping build, so it is no
// longer assembled there. Every consumer of `self.j` at a row ≥ ndense
// was re-checked at land time:
//   · `chol_fwd_unit(nv, m, voff+jt, sign, …)` takes the pivot INDEX and
//     the sign as arguments — it never looks at `j`;
//   · the effective mass is the R10 closed form
//     `0.0 + sign·minvjt[row·nv + voff+jt]`;
//   · the PGS residual is `0.0 + sign·v[voff+jt]`;
//   · `jv_pre` is `cfg(test)`;
//   · `row_j` / `limit_dot_ref_vs_closed` are `cfg(test)`;
//   · `pub fn jv` has exactly one caller, tests_constraints.rs.
// The dense-row solves and the
// `for row in 0..ndense` loops stop at ndense by construction, so no
// dense-row consumer can reach a limit row. `LIMIT_J_ROWS` keeps the
// assembly for `cargo test` and debug builds — the same idiom (and the
// same cfgs) as `GROUND_J_ROWS`.
// Worth `nlim × (nv + 1)` stores per substep, i.e. ~270 under the
// random-gear micro and ~565 under bench traffic (R26: nlim mean 15.73).
const LIMIT_J_ROWS: bool = cfg!(any(test, debug_assertions));

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
    pub contacts_enabled: bool,
    /// Unilateral frictional GROUND rows on the 8 sole corners (contact
    /// spec §1/§3 — the env's only foot mode). False only in harnesses
    /// that want a free-floating body (the MuJoCo oracle, pendulum-style
    /// dynamics gates).
    pub ground: bool,
    /// Substep length h — set by the substep wrappers before solving.
    pub h: Real,
    /// Active contacts of the last solve, pair-id order.
    pub contacts: Vec<Contact>,
    /// Active sole corners of the last solve, corner order (contact
    /// spec §2). Three rows each.
    pub ground_contacts: Vec<GroundContact>,
    /// Item 2: number of LEFT-foot corners at the head of
    /// `ground_contacts` (the rest are right-foot), found once per solve.
    ground_nleft: usize,
    /// Assembled row count of the last solve (ground + contacts + limit
    /// rows).
    pub nrows: usize,
    /// Per-row impulses λ (N·s) of the last solve: 3 rows per active
    /// sole corner in corner order, then contacts in pair-id order, then
    /// limit rows in joint order.
    pub lambda: Vec<Real>,
    /// Per-row RHS target velocity b of the last solve.
    pub bias: Vec<Real>,
    /// Per-row J·v* BEFORE the solve (introspection: complementarity /
    /// no-sticking checks). INTROSPECTION-ONLY — nothing in the solve
    /// reads it, so both the field and its per-row nv-dot are compiled
    /// out of the shipping extension (2026-08-03 dead-work pass);
    /// `cargo test` keeps it.
    #[cfg(test)]
    pub jv_pre: Vec<Real>,
    /// TEST HOOK (R23a): when set, every assembled LIMIT row of `j` is
    /// overwritten with NaN as soon as the assembly loop finishes —
    /// which is exactly the state the SHIPPING build leaves them in
    /// (never written at all, so holding whatever the buffer held).
    /// A trajectory that is bit-identical with this on proves the
    /// shipping build has no reader for them.
    #[cfg(test)]
    pub poison_limit_j: bool,
    /// TEST HOOK (audit #9): run the ground block in the PRE-audit-#9
    /// per-row form — M⁻¹jᵀ solved for every dense corner row, diag
    /// = j·M⁻¹jᵀ, and the PGS residual / impulse through those nv-wide
    /// rows — so a test can hold the 6-D form against the formulation
    /// it replaced, substep by substep.
    #[cfg(test)]
    pub ground_rowwise_ref: bool,
    nv: usize,
    j: Vec<Real>,      // MAX_ROWS × nv dense Jacobian rows
    minvjt: Vec<Real>, // MAX_ROWS × nv  M⁻¹Jᵀ columns
    diag: Vec<Real>,   // per-row effective mass  J M⁻¹ Jᵀ (diagonal)
    // ω-cap pass caches: M⁻¹e_j columns, lazily solved
    capcol: Vec<Real>,     // nq × nv
    capmass: Vec<Real>,    // nq: e_jᵀ·capcol_j (composite effective mobility)
    capcached: Vec<bool>, // nq
    /// (joint, sign) per assembled limit row, assembly order.
    limset: Vec<(usize, Real)>,
    /// Row index of joint j's LOWER-stop (sign +1) limit row this
    /// solve, or −1. The ω-cap pass reuses that row's already-solved,
    /// already-projected column instead of redoing the work (R1 1f).
    limpos: Vec<i32>,
    /// nv × WT_ROWS RHS-minor scratch for the batched solves (R1 1a).
    wt: Vec<Real>,
    /// audit #9: foot f's six spatial Jacobian rows (3 linear at the
    /// foot origin, 3 world angular), dense nv-wide, at rows 6f..6f+6.
    /// Assembled only for a foot with an active corner.
    gfoot_j: Vec<Real>,
    /// audit #9: `M⁻¹ J_fᵀ`, row-major like `minvjt` (same slots).
    gfoot_minv: Vec<Real>,
    /// audit #9, per ground row g (corner-packed, 3 per active corner):
    /// the two non-zero angular coefficients of c = [e_a; r × e_a]
    /// (at angular indices `GROUND_SPATIAL[t].1/.2`) ...
    gcoef: [[Real; 2]; N_GROUND_ROWS],
    /// ... and u = Φ_f c, the row's image in foot-velocity space.
    gu: [[Real; FOOT_ROWS]; N_GROUND_ROWS],
}

/// ω-cap column guard: a column whose mobility `(M⁻¹)_jj` fails
/// `plain > COMPOSITE_EPS · plain` (i.e. is zero, NaN or infinite — a
/// degenerate factor) gets capmass +∞, which makes the cap a no-op for
/// that joint. The threshold itself dates from the v10 weld-null
/// projection (SPEC-v10 Divergences §18: projected vs plain mobility);
/// without the projection both sides are the same number, and the
/// comparison is kept verbatim so the contact path is bitwise unchanged.
const COMPOSITE_EPS: Real = 1e-4;

impl ConstraintSet {
    /// `contacts_enabled`: the §2 self-collision normals. `ground`: the
    /// env v11 ground block (contact spec §1/§3; the env always sets it —
    /// the v10 foot welds were removed 2026-09-27). With `ground` off the
    /// body floats free (harnesses only).
    pub fn new(model: &Model, contacts_enabled: bool, ground: bool) -> ConstraintSet {
        let nv = model.nv();
        ConstraintSet {
            contacts_enabled,
            ground,
            h: super::step::H_SUB,
            contacts: Vec::with_capacity(NUM_PAIRS),
            ground_contacts: Vec::with_capacity(NUM_GROUND_CORNERS),
            ground_nleft: 0,
            nrows: 0,
            lambda: vec![0.0; MAX_ROWS],
            bias: vec![0.0; MAX_ROWS],
            #[cfg(test)]
            jv_pre: vec![0.0; MAX_ROWS],
            #[cfg(test)]
            poison_limit_j: false,
            #[cfg(test)]
            ground_rowwise_ref: false,
            nv,
            j: vec![0.0; MAX_ROWS * nv],
            minvjt: vec![0.0; MAX_ROWS * nv],
            diag: vec![0.0; MAX_ROWS],
            capcol: vec![0.0; model.nq * nv],
            capmass: vec![0.0; model.nq],
            capcached: vec![false; model.nq],
            limset: Vec::with_capacity(model.nq),
            limpos: vec![-1; model.nq],
            wt: vec![0.0; WT_ROWS * nv],
            gfoot_j: vec![0.0; 2 * FOOT_ROWS * nv],
            gfoot_minv: vec![0.0; 2 * FOOT_ROWS * nv],
            gcoef: [[0.0; 2]; N_GROUND_ROWS],
            gu: [[0.0; FOOT_ROWS]; N_GROUND_ROWS],
        }
    }

    /// audit #10: solve W consecutive rows of `src` (v-order, `nv`
    /// apart) into the same rows of `dst` through the tree factor:
    /// transpose-PERMUTE into the RHS-minor e-order scratch `wt` (exact
    /// copies), forward pass (restricted to `mask` when given — the
    /// upward closure of the rows' structural support, bitwise the full
    /// forward, see `tc_fwd_mask_t`), back pass, transpose-permute out.
    /// Lanes are independent accumulators, so the batch is bitwise W
    /// single-column solves (item 3's argument).
    #[allow(clippy::too_many_arguments)]
    fn solve_rows_tc<const W: usize>(
        tc: &TreeChol,
        l: &[Real],
        src: &[Real],
        dst: &mut [Real],
        row: usize,
        wt: &mut [Real],
        mask: Option<u64>,
    ) {
        let nv = tc.n;
        for r in 0..W {
            let jr = &src[(row + r) * nv..(row + r + 1) * nv];
            for i in 0..nv {
                wt[tc.v2e[i] * W + r] = jr[i];
            }
        }
        match mask {
            Some(m) => tc_fwd_mask_t::<W>(tc, l, m, &mut wt[..W * nv]),
            None => tc_fwd_t::<W>(tc, l, &mut wt[..W * nv]),
        }
        tc_back_t::<W>(tc, l, &mut wt[..W * nv]);
        for r in 0..W {
            let mr = &mut dst[(row + r) * nv..(row + r + 1) * nv];
            for i in 0..nv {
                mr[i] = wt[tc.v2e[i] * W + r];
            }
        }
    }

    /// audit #10: the unit columns `sign·e_(voff+j)` of W limit rows
    /// (`limset[li..li+W]`) into `minvjt` rows `row..row+W`: each lane's
    /// forward runs only its own ancestor chain (`tc_fwd_unit_lane`),
    /// the dense back pass is shared, permuted out like `solve_rows_tc`.
    #[allow(clippy::too_many_arguments)]
    fn solve_limit_cols<const W: usize>(
        tc: &TreeChol,
        l: &[Real],
        limset: &[(usize, Real)],
        voff: usize,
        minvjt: &mut [Real],
        row: usize,
        wt: &mut [Real],
    ) {
        let nv = tc.n;
        wt[..W * nv].fill(0.0);
        for r in 0..W {
            let (jt, sign) = limset[r];
            tc_fwd_unit_lane::<W>(tc, l, tc.v2e[voff + jt], sign, r, &mut wt[..W * nv]);
        }
        tc_back_t::<W>(tc, l, &mut wt[..W * nv]);
        for r in 0..W {
            let mr = &mut minvjt[(row + r) * nv..(row + r + 1) * nv];
            for i in 0..nv {
                mr[i] = wt[tc.v2e[i] * W + r];
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
        jv: Real,
        bias: &[Real],
        diag: &[Real],
        lambda: &mut [Real],
        minvjt: &[Real],
        v: &mut [Real],
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
    /// (test / debug builds) — the shipping build no
    /// longer assembles those rows, so `self.j` holds stale words there.
    /// The only caller is `tests_constraints`.
    pub fn jv(&self, row: usize, v: &[Real]) -> Real {
        let jr = &self.j[row * self.nv..(row + 1) * self.nv];
        let mut s = 0.0;
        for i in 0..self.nv {
            s += jr[i] * v[i];
        }
        s
    }

    /// TEST HOOK (R13): fill the solve's staging scratch — every
    /// `minvjt` row and the `wt` RHS-minor buffer — with NaN. The next
    /// solve must reproduce the un-poisoned trajectory BIT FOR BIT: every
    /// word of those regions a solve reads, it wrote first.
    #[cfg(test)]
    pub fn poison_solve_scratch(&mut self) {
        self.minvjt.fill(Real::NAN);
        self.wt.fill(Real::NAN);
    }

    /// TEST HOOK (R10): the `(joint, sign)` pairs of the limit rows
    /// assembled by the last solve, assembly order.
    #[cfg(test)]
    pub fn limit_set(&self) -> &[(usize, Real)] {
        &self.limset
    }

    /// TEST HOOK (R10): an assembled Jacobian row of the last solve.
    #[cfg(test)]
    pub fn row_j(&self, row: usize) -> &[Real] {
        &self.j[row * self.nv..(row + 1) * self.nv]
    }

    /// TEST HOOK (R10): an `M⁻¹Jᵀ` column of the last solve.
    #[cfg(test)]
    pub fn row_minvjt(&self, row: usize) -> &[Real] {
        &self.minvjt[row * self.nv..(row + 1) * self.nv]
    }

    /// TEST HOOK (R10): dense-vs-closed-form contraction of every
    /// assembled LIMIT row against `src`, on the LIVE rows of the last
    /// solve. Returns `(dense, closed)` per limit row — the single
    /// expression all three R10 sites collapse.
    #[cfg(test)]
    pub fn limit_dot_ref_vs_closed(&self, voff: usize, src: &[Real]) -> Vec<(Real, Real)> {
        let nv = self.nv;
        let ndense = self.ground_rows() + self.contacts.len();
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

    /// Ground row count of the last solve: 3 per ACTIVE sole corner.
    pub fn ground_rows(&self) -> usize {
        self.ground_contacts.len() * GROUND_ROWS_PER_CORNER
    }

    /// Row index of active-corner `gi`'s NORMAL row in the last solve;
    /// its two friction rows are the next two. Introspection (the fall /
    /// no-pull / friction gates).
    pub fn ground_row(&self, gi: usize) -> usize {
        gi * GROUND_ROWS_PER_CORNER
    }

    /// Number of joint-limit rows assembled by the last solve.
    pub fn limit_rows(&self) -> usize {
        self.nrows - self.ground_rows() - self.contacts.len()
    }

    /// `(joint, λ)` of every limit row of the last solve, assembly
    /// order — the impulse (N·m·s) each stop actually carried this
    /// substep. λ is PGS-clamped ≥ 0 (unilateral rows); a row is
    /// assembled whenever the joint is within reach of a stop, so a
    /// hovering-but-not-resting joint reports λ ≈ 0. Feeds the env v34
    /// limit-rest telemetry/penalty (lib.rs Body::limit_imp).
    pub fn limit_row_lambdas(&self) -> impl Iterator<Item = (usize, Real)> + '_ {
        let base = self.ground_rows() + self.contacts.len();
        self.limset
            .iter()
            .enumerate()
            .map(move |(li, &(j, _sign))| (j, self.lambda[base + li]))
    }

    /// Effective mass (J M⁻¹ Jᵀ diagonal) of an assembled row —
    /// post-solve introspection (tests).
    pub fn row_diag(&self, row: usize) -> Real {
        self.diag[row]
    }

    /// MIRROR: assemble the ground block — 3 rows per ACTIVE sole
    /// corner, corners in the contact spec §2 normative order, rows in
    /// the §3 normative order (normal, then friction x̂, then ẑ).
    /// Returns the new row count. Straight-line scalar form on purpose:
    /// the CUDA mirror transcribes it serially.
    ///
    /// The rows come first (the caller passes `nrows` = 0), i.e. BEFORE
    /// the self-collision normals — so the §2 pair-id row order
    /// downstream is untouched.
    ///
    /// audit #9: what is assembled per row is its 6-D foot coefficient
    /// (`gcoef`, see `GROUND_SPATIAL`) plus its RHS; the Jacobian object
    /// is each CONTACTING foot's six spatial rows (`gfoot_j`, from
    /// `point_jacobian_acc` / `angular_jacobian_acc`). The dense corner rows are introspection-only
    /// (`GROUND_J_ROWS`).
    fn assemble_ground_rows(
        &mut self,
        model: &Model,
        ws: &Workspace,
        hinv: Real,
        mut nrows: usize,
    ) -> usize {
        let nv = self.nv;
        let (nl, n) = (self.ground_nleft, self.ground_contacts.len());
        for f in 0..2 {
            let active = if f == 0 { nl > 0 } else { n > nl };
            if !active {
                continue;
            }
            let frame = GROUND_FRAMES[f];
            let pf = ws.p[frame];
            for t in 0..3 {
                let r = FOOT_ROWS * f + t;
                let jr = &mut self.gfoot_j[r * nv..(r + 1) * nv];
                jr.fill(0.0);
                let mut e = [0.0; 3];
                e[t] = 1.0;
                point_jacobian_acc(model, ws, frame, pf, e, 1.0, jr);
            }
            for t in 0..3 {
                let r = FOOT_ROWS * f + 3 + t;
                let jr = &mut self.gfoot_j[r * nv..(r + 1) * nv];
                jr.fill(0.0);
                angular_jacobian_acc(model, ws, frame, t, 1.0, jr);
            }
        }
        for gi in 0..n {
            let gc = self.ground_contacts[gi];
            if GROUND_J_ROWS {
                for t in 0..GROUND_ROWS_PER_CORNER {
                    let row = nrows + t;
                    let jr = &mut self.j[row * nv..(row + 1) * nv];
                    jr.fill(0.0);
                    // static ground: only the corner's own chain contributes
                    point_jacobian_acc(model, ws, gc.frame, gc.point, GROUND_DIRS[t], 1.0, jr);
                }
            }
            // lever from the foot origin; the angular coefficients of
            // c = [e_a; r × e_a] in `GROUND_SPATIAL` order
            let r = sub3(gc.point, ws.p[gc.frame]);
            let g = gi * GROUND_ROWS_PER_CORNER;
            self.gcoef[g] = [-r[2], r[0]]; // normal ŷ: ω_x, ω_z
            self.gcoef[g + 1] = [r[2], -r[1]]; // friction x̂: ω_y, ω_z
            self.gcoef[g + 2] = [r[1], -r[0]]; // friction ẑ: ω_x, ω_y
            // NORMAL row rhs: the §3-step-8 contact pipeline verbatim —
            // speculative approach limit while the gap is open, Baumgarte
            // push-out beyond SLOP once penetrating (contact spec §3:
            // "same β pipeline as the self-collision normals").
            self.bias[nrows] = if gc.gap >= 0.0 {
                -gc.gap * hinv
            } else {
                (BETA_CONTACT * hinv) * (-gc.gap - SLOP).max(0.0)
            };
            // FRICTION rows rhs = 0: friction stabilizes nothing, so it
            // gets no Baumgarte term (contact spec §3).
            self.bias[nrows + 1] = 0.0;
            self.bias[nrows + 2] = 0.0;
            nrows += GROUND_ROWS_PER_CORNER;
        }
        nrows
    }

    /// audit #9: M⁻¹J_fᵀ for every contacting foot (one RHS-MINOR batch:
    /// 12 columns with both feet down, 6 with one — the columns are
    /// independent and the batch is bitwise the per-column solve, item 3's
    /// argument), then per foot Φ_f and every active row's u / diag.
    fn prep_ground_rows(&mut self, ws: &Workspace) {
        let (nl, n) = (self.ground_nleft, self.ground_contacts.len());
        let (a0, a1) = (nl > 0, n > nl);
        // audit #10: a foot's six spatial rows are supported on the base
        // columns plus that foot's chain, so their forward pass runs only
        // the upward closure of that support (the leg chain and the base
        // in the tree order — bitwise the full forward, `tc_fwd_mask_t`)
        let tc = &ws.tc;
        let base = 0..W_BASE_HI;
        if a0 && a1 {
            let mask = tc.closure(base.chain(W_L_LO..W_R_HI));
            Self::solve_rows_tc::<{ 2 * FOOT_ROWS }>(
                tc, &ws.l, &self.gfoot_j, &mut self.gfoot_minv, 0, &mut self.wt, Some(mask),
            );
        } else if a0 || a1 {
            let (r0, lo, hi) = if a0 { (0, W_L_LO, W_L_HI) } else { (FOOT_ROWS, W_R_LO, W_R_HI) };
            let mask = tc.closure(base.chain(lo..hi));
            Self::solve_rows_tc::<FOOT_ROWS>(
                tc, &ws.l, &self.gfoot_j, &mut self.gfoot_minv, r0, &mut self.wt, Some(mask),
            );
        }
        if a0 {
            self.prep_ground_foot::<W_L_LO, W_L_HI>(0, 0, nl);
        }
        if a1 {
            self.prep_ground_foot::<W_R_LO, W_R_HI>(1, nl, n);
        }
    }

    /// `prep_ground_rows` for foot `f` (chain columns LO..HI), whose
    /// active corners are `g0..g1`.
    ///   Φ[r][c] = Φ[c][r] = Σᵢ J_f[c][i]·G_f[r][i]  (c ≥ r; support dot,
    ///                                               ascending i)
    ///   u[i]    = (Φ[i][a] + c1·Φ[i][3+b1]) + c2·Φ[i][3+b2]
    ///   diag    = (u[a] + c1·u[3+b1]) + c2·u[3+b2]
    #[inline(always)]
    fn prep_ground_foot<const LO: usize, const HI: usize>(
        &mut self,
        f: usize,
        g0: usize,
        g1: usize,
    ) {
        let nv = self.nv;
        let mut phi = [[0.0 as Real; FOOT_ROWS]; FOOT_ROWS];
        for r in 0..FOOT_ROWS {
            let rr = FOOT_ROWS * f + r;
            let mr = &self.gfoot_minv[rr * nv..(rr + 1) * nv];
            for c in r..FOOT_ROWS {
                let cc = FOOT_ROWS * f + c;
                let jr = &self.gfoot_j[cc * nv..(cc + 1) * nv];
                let s = ground_support_dot::<LO, HI>(jr, mr);
                phi[r][c] = s;
                phi[c][r] = s;
            }
        }
        for g in g0 * GROUND_ROWS_PER_CORNER..g1 * GROUND_ROWS_PER_CORNER {
            let (a, b1, b2) = GROUND_SPATIAL[g % GROUND_ROWS_PER_CORNER];
            let [c1, c2] = self.gcoef[g];
            let mut u = [0.0 as Real; FOOT_ROWS];
            for i in 0..FOOT_ROWS {
                u[i] = phi[i][a] + c1 * phi[i][3 + b1] + c2 * phi[i][3 + b2];
            }
            let d = u[a] + c1 * u[3 + b1] + c2 * u[3 + b2];
            debug_assert!(d > 0.0, "ground row {g} has zero effective mass");
            self.gu[g] = u;
            self.diag[g] = d;
        }
    }

    /// MIRROR: one PGS sweep over the ground block, corner order. Per
    /// corner: the unilateral normal row first (λ_n ≥ 0, the §2 contact
    /// path verbatim), then its two friction rows box-clamped against
    /// the λ_n THIS sweep just produced — Gauss-Seidel coupling,
    /// re-clamped every iteration (contact spec §3).
    ///
    /// Fixed trip count over the active corners, no early exit and no
    /// data-dependent branch inside the sweep: the §2 contact loop's
    /// contract, which parity and the K13 warp-divergence rule both
    /// require.
    fn pgs_ground_sweep(&mut self, v: &mut [Real]) {
        // item 2: left-foot prefix then right-foot suffix — the same
        // corners in the same order, each loop with its foot's
        // compile-time support segment
        let (nl, n) = (self.ground_nleft, self.ground_contacts.len());
        if nl > 0 {
            self.pgs_ground_sweep_foot::<W_L_LO, W_L_HI>(0, 0, nl, v);
        }
        if n > nl {
            self.pgs_ground_sweep_foot::<W_R_LO, W_R_HI>(1, nl, n, v);
        }
    }

    /// `pgs_ground_sweep` over active corners `g0..g1` (≥ 1), all on foot
    /// `f` whose chain columns are `LO..HI` — audit #9's 6-D form of the
    /// per-row sweep (see `GROUND_SPATIAL`): V = J_f v once, each row's
    /// residual c·V and its `pgs_row_update(_box)` λ step verbatim, the
    /// impulse carried as Λ += c·Δλ / V += u·Δλ, then v += G_f Λ once.
    #[inline(always)]
    fn pgs_ground_sweep_foot<const LO: usize, const HI: usize>(
        &mut self,
        f: usize,
        g0: usize,
        g1: usize,
        v: &mut [Real],
    ) {
        let nv = self.nv;
        let mut vf = [0.0 as Real; FOOT_ROWS];
        for a in 0..FOOT_ROWS {
            let r = FOOT_ROWS * f + a;
            vf[a] = ground_support_dot::<LO, HI>(&self.gfoot_j[r * nv..(r + 1) * nv], v);
        }
        let mut lf = [0.0 as Real; FOOT_ROWS];
        let mut any = false;
        for gi in g0..g1 {
            let nrow = gi * GROUND_ROWS_PER_CORNER;
            let mut bound = 0.0;
            for t in 0..GROUND_ROWS_PER_CORNER {
                let row = nrow + t;
                let g = gi * GROUND_ROWS_PER_CORNER + t;
                let (a, b1, b2) = GROUND_SPATIAL[t];
                let [c1, c2] = self.gcoef[g];
                let jv = vf[a] + c1 * vf[3 + b1] + c2 * vf[3 + b2];
                let dl = (self.bias[row] - jv) / self.diag[row];
                let lam = self.lambda[row];
                // the friction rows' SYMMETRIC BOX |λ_t| ≤ MU·λ_n (contact
                // spec §3; the former `pgs_row_update_box`, verbatim):
                // `.min(bound).max(-bound)` mirrors the normal row's
                // `.max(0.0)` idiom — f64::min/max return the OTHER
                // operand on NaN, so an artifact λ collapses onto the
                // bound; at bound == 0.0 a negative candidate lands on
                // −0.0 and `d != 0.0` is false for a λ already at zero.
                let newl = if t == 0 {
                    (lam + dl).max(0.0) // λ_n ≥ 0
                } else {
                    (lam + dl).min(bound).max(-bound) // |λ_t| ≤ MU·λ_n
                };
                let d = newl - lam;
                self.lambda[row] = newl;
                if d != 0.0 {
                    lf[a] += d;
                    lf[3 + b1] += c1 * d;
                    lf[3 + b2] += c2 * d;
                    let u = &self.gu[g];
                    for i in 0..FOOT_ROWS {
                        vf[i] += u[i] * d;
                    }
                    any = true;
                }
                if t == 0 {
                    // the friction cone is read AFTER the normal update,
                    // so a corner that just lost its normal impulse loses
                    // its friction in the same sweep
                    bound = MU * self.lambda[row];
                }
            }
        }
        if any {
            let r0 = FOOT_ROWS * f;
            let gm = &self.gfoot_minv[r0 * nv..(r0 + FOOT_ROWS) * nv];
            let (m0, t) = gm.split_at(nv);
            let (m1, t) = t.split_at(nv);
            let (m2, t) = t.split_at(nv);
            let (m3, t) = t.split_at(nv);
            let (m4, m5) = t.split_at(nv);
            for i in 0..nv {
                let mut x = v[i];
                x += m0[i] * lf[0];
                x += m1[i] * lf[1];
                x += m2[i] * lf[2];
                x += m3[i] * lf[3];
                x += m4[i] * lf[4];
                x += m5[i] * lf[5];
                v[i] = x;
            }
        }
    }

    /// TEST HOOK (audit #9): the PRE-audit per-row ground sweep, verbatim
    /// in semantics (dense dot residual, per-row clamp, nv-wide impulse
    /// propagation through the row's own M⁻¹jᵀ) — the reference the 6-D
    /// sweep is held against (`ground_rowwise_ref`).
    #[cfg(test)]
    fn pgs_ground_sweep_rowwise(&mut self, v: &mut [Real]) {
        let nv = self.nv;
        for gi in 0..self.ground_contacts.len() {
            let nrow = gi * GROUND_ROWS_PER_CORNER;
            let mut bound = 0.0;
            for t in 0..GROUND_ROWS_PER_CORNER {
                let row = nrow + t;
                let jr = &self.j[row * nv..(row + 1) * nv];
                let mut jv = 0.0;
                for i in 0..nv {
                    jv += jr[i] * v[i];
                }
                let dl = (self.bias[row] - jv) / self.diag[row];
                let lam = self.lambda[row];
                let newl = if t == 0 {
                    (lam + dl).max(0.0)
                } else {
                    (lam + dl).min(bound).max(-bound)
                };
                let d = newl - lam;
                self.lambda[row] = newl;
                if d != 0.0 {
                    let mr = &self.minvjt[row * nv..(row + 1) * nv];
                    for i in 0..nv {
                        v[i] += mr[i] * d;
                    }
                }
                if t == 0 {
                    bound = MU * self.lambda[row];
                }
            }
        }
    }

    /// TEST HOOK (audit #9): foot f's six spatial rows and their M⁻¹J_fᵀ
    /// columns of the last solve (valid when the foot had a contact).
    #[cfg(test)]
    pub fn foot_rows(&self, f: usize) -> (&[Real], &[Real]) {
        let (a, b) = (FOOT_ROWS * f * self.nv, FOOT_ROWS * (f + 1) * self.nv);
        (&self.gfoot_j[a..b], &self.gfoot_minv[a..b])
    }

    /// TEST HOOK (audit #9): ground row g's 6-D coefficient c (full
    /// 6-vector) and u = Φ_f c of the last solve.
    #[cfg(test)]
    pub fn ground_row_6d(&self, g: usize) -> ([Real; FOOT_ROWS], [Real; FOOT_ROWS]) {
        let (a, b1, b2) = GROUND_SPATIAL[g % GROUND_ROWS_PER_CORNER];
        let mut c = [0.0 as Real; FOOT_ROWS];
        c[a] = 1.0;
        c[3 + b1] = self.gcoef[g][0];
        c[3 + b2] = self.gcoef[g][1];
        (c, self.gu[g])
    }

    /// TEST HOOK (item 2 / audit #9): every foot spatial row of the last
    /// solve's CONTACTING feet: (dense 35-wide dot, support dot) against
    /// `src`, plus a flag that every column outside the row's support is
    /// the +0.0 bit pattern.
    #[cfg(test)]
    pub fn ground_dot_ref_vs_support(&self, src: &[Real]) -> Vec<(Real, Real, bool)> {
        let nv = self.nv;
        let (nl, n) = (self.ground_nleft, self.ground_contacts.len());
        let mut out = Vec::new();
        for f in 0..2 {
            if !(if f == 0 { nl > 0 } else { n > nl }) {
                continue;
            }
            for a in 0..FOOT_ROWS {
                let r = FOOT_ROWS * f + a;
                let jr = &self.gfoot_j[r * nv..(r + 1) * nv];
                let mut dense = 0.0;
                for i in 0..nv {
                    dense += jr[i] * src[i];
                }
                let (lo, hi) = if f == 0 { (W_L_LO, W_L_HI) } else { (W_R_LO, W_R_HI) };
                let sup = if f == 0 {
                    ground_support_dot::<W_L_LO, W_L_HI>(jr, src)
                } else {
                    ground_support_dot::<W_R_LO, W_R_HI>(jr, src)
                };
                let zeros = (0..nv)
                    .filter(|&i| !(i < W_BASE_HI || (lo..hi).contains(&i)))
                    .all(|i| jr[i].to_bits() == (0.0 as Real).to_bits());
                out.push((dense, sup, zeros));
            }
        }
        out
    }

    /// §3 steps 7–9 against the FK + Cholesky factor already in `ws`
    /// (valid at the seam: computed from the pre-integration q this
    /// substep). `q` is the pre-integration actuated position vector
    /// (joint-limit rows). Projects `v` (= v*) in place.
    pub fn solve(&mut self, model: &Model, ws: &mut Workspace, q: &[Real], v: &mut [Real]) {
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

        // ── step 7: collision detection ───────────────────────────────
        // Ground corners FIRST (contact spec §2 corner order), then the
        // 55-pair self-collision table (pair-id order). The two sets are
        // disjoint — sole corners touch the GROUND only (contact spec
        // §6) — and the ground block's rows precede the pair rows, so
        // the §2 pair-id row order is untouched.
        if self.ground {
            ground_phase(ws, &mut self.ground_contacts);
            // item 2: the ground support segments pin the humanoid layout
            // and a left-foot-prefix order
            assert_eq!(nv, W_R_HI, "the item-2 ground-support segments pin nv = 35");
            self.ground_nleft = self
                .ground_contacts
                .partition_point(|g| g.frame == F_L_FOOT);
            debug_assert!(self.ground_contacts[self.ground_nleft..]
                .iter()
                .all(|g| g.frame == F_R_FOOT));
        } else {
            self.ground_contacts.clear();
            self.ground_nleft = 0;
        }
        if self.contacts_enabled {
            narrow_phase(ws, &mut self.contacts);
        } else {
            self.contacts.clear();
        }

        // ── assembly: the ground block (rows 0..) ─────────────────────
        nrows = self.assemble_ground_rows(model, ws, hinv, nrows);
        let nground = self.ground_rows();

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
        //    missed stop is pushed back with β_limit. Without these
        //    rows the step-11 position clamp FIGHTS the rest of the solve
        //    at the stance stops (knee/elbow/wrist-roll all sit at a
        //    0-limit at neutral): the solve assigns flexion rate, the
        //    clamp teleports q back (measured under the v10 welds as
        //    5.8e-4 m of foot error per substep).
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
                    (BETA_LIMIT_ROW * hinv) * -gap
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
            // v14 audit fix: include the ground block, or in contact
            // mode this would NaN the GROUND rows instead of the
            // limit rows it means to poison
            let nd = self.ground_rows() + self.contacts.len();
            self.j[nd * nv..nrows * nv].fill(Real::NAN);
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
        // BATCHED multi-RHS solve (2026-08-03, R1): every row's RHS is
        // the same shared factor's, so instead of `nrows` serial dense
        // chol_solves the rows go 4 at a time through `chol_solve_x4`,
        // which loads the factor ONCE per group of 4 and keeps each
        // RHS on its own unit-stride ascending-k accumulator chain —
        // bit-for-bit the serial result (see the algo.rs contract).
        // "dense" = every row whose Jacobian is assembled in full and
        // whose M⁻¹Jᵀ column goes through the batched Cholesky ladder:
        // the ground block, then the self-collision normals.
        // (The limit rows below are rank-1 and take the closed forms.)
        // audit #9: the ground rows are NOT solved per row any more —
        // `prep_ground_rows` solves each contacting foot's six spatial
        // columns instead — so the batched ladder starts at `pbase`.
        let ndense = nground + self.contacts.len();
        let pbase = nground;
        // audit #10: every M⁻¹-column below goes through the TREE factor
        // (`ws.l`, e-order) — RHS-MINOR batches whose transposes also
        // apply the v → e permutation (exact copies), so the permutation
        // costs nothing beyond the copies the batches already made.
        let tc = &ws.tc;
        let lf = &ws.l;
        {
            // audit #9: the ground block's foot columns, Φ_f, u and diag
            if nground > 0 {
                self.prep_ground_rows(ws);
            }
            // item 3: dense pair rows in RHS-MINOR batches straight from
            // `j` — DENSE_T_W1 at a time, one DENSE_T_W2 rung, then a
            // 4 / 2 / 1 remainder ladder (audit #10: the remainder rungs
            // are RHS-minor too, so every row takes the permuting
            // transpose; the old row-major x4/x2/x1 rungs and their
            // `j → minvjt` seed copy are gone). Full forward pass: a pair
            // row's support is two arbitrary chains.
            let mut row = pbase;
            while row + DENSE_T_W1 <= ndense {
                Self::solve_rows_tc::<DENSE_T_W1>(tc, lf, &self.j, &mut self.minvjt, row, &mut self.wt, None);
                row += DENSE_T_W1;
            }
            if row + DENSE_T_W2 <= ndense {
                Self::solve_rows_tc::<DENSE_T_W2>(tc, lf, &self.j, &mut self.minvjt, row, &mut self.wt, None);
                row += DENSE_T_W2;
            }
            while row + 4 <= ndense {
                Self::solve_rows_tc::<4>(tc, lf, &self.j, &mut self.minvjt, row, &mut self.wt, None);
                row += 4;
            }
            if row + 2 <= ndense {
                Self::solve_rows_tc::<2>(tc, lf, &self.j, &mut self.minvjt, row, &mut self.wt, None);
                row += 2;
            }
            if row < ndense {
                Self::solve_rows_tc::<1>(tc, lf, &self.j, &mut self.minvjt, row, &mut self.wt, None);
            }
        }
        // LIMIT rows have the unit right-hand side sign·e_(voff+j) (the
        // assembly is `jr.fill(0.0)` plus one `jr[voff+j] = sign` store —
        // and under R23a not even that in the shipping build; the solve
        // takes the pivot index and the sign, never the row). audit #10:
        // each column's forward pass is its own ancestor chain in the
        // tree order (`tc_fwd_unit_lane`, ≤ 16 rows, bitwise the full
        // forward), and the dense back pass runs LIMIT_T_W / 4 / 2 / 1
        // columns at a time in the RHS-minor layout.
        let nlim = self.limset.len();
        {
            let mut li = 0;
            while li + LIMIT_T_W <= nlim {
                Self::solve_limit_cols::<LIMIT_T_W>(
                    tc, lf, &self.limset[li..], voff, &mut self.minvjt, ndense + li, &mut self.wt,
                );
                li += LIMIT_T_W;
            }
            if li + 4 <= nlim {
                Self::solve_limit_cols::<4>(
                    tc, lf, &self.limset[li..], voff, &mut self.minvjt, ndense + li, &mut self.wt,
                );
                li += 4;
            }
            if li + 2 <= nlim {
                Self::solve_limit_cols::<2>(
                    tc, lf, &self.limset[li..], voff, &mut self.minvjt, ndense + li, &mut self.wt,
                );
                li += 2;
            }
            if li < nlim {
                Self::solve_limit_cols::<1>(
                    tc, lf, &self.limset[li..], voff, &mut self.minvjt, ndense + li, &mut self.wt,
                );
            }
        }
        for row in 0..ndense {
            let ground_row = row < nground;
            if !ground_row {
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
            #[cfg(test)]
            {
                let jr = &self.j[row * nv..(row + 1) * nv];
                let mut s = 0.0;
                for i in 0..nv {
                    s += jr[i] * v[i];
                }
                self.jv_pre[row] = s;
            }
        }
        // audit #9 TEST HOOK: the pre-audit per-row ground quantities
        #[cfg(test)]
        if self.ground_rowwise_ref {
            for row in 0..nground {
                let (j, m) = (&self.j, &mut self.minvjt);
                let mr = &mut m[row * nv..(row + 1) * nv];
                mr.copy_from_slice(&j[row * nv..(row + 1) * nv]);
                super::algo::tc_solve_v(&ws.tc, &ws.l, mr, &mut ws.esc);
                let jr = &self.j[row * nv..(row + 1) * nv];
                let mr = &self.minvjt[row * nv..(row + 1) * nv];
                let mut d = 0.0;
                for i in 0..nv {
                    d += jr[i] * mr[i];
                }
                self.diag[row] = d;
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
            #[cfg(test)]
            {
                let jr = &self.j[row * nv..(row + 1) * nv];
                let mut s = 0.0;
                for i in 0..nv {
                    s += jr[i] * v[i];
                }
                self.jv_pre[row] = s;
            }
        }

        // ── step 8: exactly 12 iterations, fixed order, no warm start;
        //    per iteration: (a) the ground block, (b) contact rows
        //    projected Gauss-Seidel in pair-id order, (c) limit rows,
        //    (d) the ω-cap pass. Step 9 (v ← v* + M⁻¹Jᵀλ) is realized by
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
            // (a) ground block (contact spec §3), corner order: the
            // normal row then its two box-clamped friction rows. Same
            // iteration count and the same fixed sweep order as the rows
            // around it — it is simply another projected-GS block. Empty
            // (and free) with `ground` off: `ground_contacts` is cleared
            // at step 7.
            #[cfg(test)]
            let rowwise = self.ground_rowwise_ref;
            #[cfg(not(test))]
            let rowwise = false;
            if rowwise {
                #[cfg(test)]
                self.pgs_ground_sweep_rowwise(v);
            } else {
                self.pgs_ground_sweep(v);
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
            for row in nground..ndense {
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
            // folded in — Divergence §13's anticipation): the solve may
            // assign super-cap joint rates (under the v10 welds their
            // h²·q̇² integration error measured 38 mm of foot drift);
            // projecting the caps INSIDE the iteration lets the solve
            // redistribute while integration only ever sees capped
            // rates. Momentum-consistent internal impulses along
            // M⁻¹e_j, lazily solved and cached per substep; a degenerate
            // column (see COMPOSITE_EPS) is skipped.
            for jt in 0..model.nq {
                let vj = voff + jt;
                let cap = model.omega_cap[jt];
                let target = v[vj].clamp(-cap, cap);
                if target != v[vj] {
                    if !self.capcached[jt] && self.limpos[jt] >= 0 {
                        // DEDUPE (2026-08-03, R1 item 1f): joint jt has
                        // a LOWER-stop limit row this substep, whose
                        // right-hand side is +e_vj — the very column
                        // this pass is about to solve. So that row's
                        // `minvjt` IS this joint's `capcol`, bit for
                        // bit, and its `diag` IS `capmass`:
                        //   · the two solves are the same call on the
                        //     same RHS;
                        //   · `diag[row] = Σᵢ jr[i]·mr[i]` with jr =
                        //     +e_vj collapses to `1.0 · mr[vj]` (the
                        //     other terms add ±0.0 to a non-zero
                        //     partial sum, which is exact) = the cap
                        //     pass's `plain`, and `plain > COMPOSITE_EPS ·
                        //     plain` holds for any positive finite
                        //     mobility (a limit row's diag is asserted
                        //     > 0), so the cap pass's verdict is the
                        //     row's diag itself.
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
                            // unit RHS +e_vj: audit #10 chain forward
                            // (only vj's ancestors in the tree order —
                            // bitwise the full forward, see
                            // `tc_fwd_unit_lane`) then the dense back
                            // substitution.
                            let col = &mut self.capcol[jt * nv..(jt + 1) * nv];
                            tc_solve_unit_v(&ws.tc, &ws.l, vj, 1.0, col, &mut ws.esc);
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
                        self.capmass[jt] = if plain > COMPOSITE_EPS * plain {
                            plain
                        } else {
                            Real::INFINITY // degenerate column: cap inert
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
    p: [Real; 3],
    n: [Real; 3],
    sign: Real,
    row: &mut [Real],
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
/// body `frame` into a dense row (foot angular rows).
fn angular_jacobian_acc(
    model: &Model,
    ws: &Workspace,
    frame: usize,
    axis_t: usize,
    sign: Real,
    row: &mut [Real],
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
    tau_act: &[Real],
    h: Real,
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
    tau_act: &[Real],
    root_wrench: &[Real; 6],
    h: Real,
    ws: &mut Workspace,
    cs: &mut ConstraintSet,
) {
    cs.h = h;
    let mut hook =
        |m: &Model, w: &mut Workspace, q: &[Real], v: &mut [Real]| cs.solve(m, w, q, v);
    substep_wrench(model, st, tau_act, root_wrench, h, ws, Some(&mut hook));
}

/// Harness variant with per-joint IMPLICIT damping (oracle muscle-tone
/// kd — see step.rs `substep_opts` for why explicit application is
/// unstable). The env never uses this.
pub fn substep_constrained_damped(
    model: &Model,
    st: &mut State,
    tau_act: &[Real],
    implicit_damp: &[Real],
    h: Real,
    ws: &mut Workspace,
    cs: &mut ConstraintSet,
) {
    cs.h = h;
    let mut hook =
        |m: &Model, w: &mut Workspace, q: &[Real], v: &mut [Real]| cs.solve(m, w, q, v);
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

/// `substep_constrained_damped` with an applied root wrench (the §5 (c)
/// static-friction push on the damped muscle-tone stance). The env
/// never uses this.
#[allow(clippy::too_many_arguments)]
pub fn substep_constrained_damped_wrench(
    model: &Model,
    st: &mut State,
    tau_act: &[Real],
    root_wrench: &[Real; 6],
    implicit_damp: &[Real],
    h: Real,
    ws: &mut Workspace,
    cs: &mut ConstraintSet,
) {
    cs.h = h;
    let mut hook =
        |m: &Model, w: &mut Workspace, q: &[Real], v: &mut [Real]| cs.solve(m, w, q, v);
    super::step::substep_opts(
        model,
        st,
        tau_act,
        root_wrench,
        Some(implicit_damp),
        h,
        ws,
        Some(&mut hook),
    );
}
