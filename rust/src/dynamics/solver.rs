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
    nv: usize,
    j: Vec<f64>,      // MAX_ROWS × nv dense Jacobian rows
    minvjt: Vec<f64>, // MAX_ROWS × nv  M⁻¹Jᵀ columns
    diag: Vec<f64>,   // per-row effective mass  J M⁻¹ Jᵀ (diagonal)
    // weld-block exact solve (Divergences §15): A_ww = J_w M⁻¹ J_wᵀ
    aww: Vec<f64>, // 12×12
    lww: Vec<f64>, // its Cholesky factor
    rw: Vec<f64>,  // block residual / solve scratch (12)
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

    /// J·v of an assembled row (post-solve introspection).
    pub fn jv(&self, row: usize, v: &[f64]) -> f64 {
        let jr = &self.j[row * self.nv..(row + 1) * self.nv];
        let mut s = 0.0;
        for i in 0..self.nv {
            s += jr[i] * v[i];
        }
        s
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
                let jr = &mut self.j[row * nv..(row + 1) * nv];
                jr.fill(0.0);
                jr[voff + j] = sign;
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
        self.minvjt[..ndense * nv].copy_from_slice(&self.j[..ndense * nv]);
        {
            let mut row = 0usize;
            if nweld == N_WELD_ROWS {
                // the 12 weld rows go through the RHS-MINOR form
                // (transpose in, solve, transpose out — both copies
                // exact, so the whole thing is bitwise `chol_solve`
                // twelve times). The transposed Jacobian is KEPT in
                // `jwt` afterwards: the PGS residual and the two
                // weld-null projections all want exactly this layout.
                for r in 0..N_WELD_ROWS {
                    let jr = &self.j[r * nv..(r + 1) * nv];
                    for i in 0..nv {
                        self.jwt[i * N_WELD_ROWS + r] = jr[i];
                    }
                }
                self.wt[..N_WELD_ROWS * nv]
                    .copy_from_slice(&self.jwt[..N_WELD_ROWS * nv]);
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
        for row in 0..nrows {
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
                    let mr = &self.minvjt[r * nv..(r + 1) * nv];
                    for i in 0..nv {
                        let m = mr[i];
                        let jr = &self.jwt[i * N_WELD_ROWS..(i + 1) * N_WELD_ROWS];
                        for c in 0..N_WELD_ROWS {
                            s[c] += jr[c] * m;
                        }
                    }
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
            // the by-4 jams below assume the 12-row weld block
            debug_assert_eq!(nweld, N_WELD_ROWS);
            let split = (nweld + ncon) * nv;
            let (head, tail) = self.minvjt.split_at_mut(split);
            for (li, _lim) in self.limset.iter().enumerate() {
                let row = nweld + ncon + li;
                let mr = &mut tail[li * nv..(li + 1) * nv];
                // UNROLL-AND-JAM by 4 (2026-08-03), both here and in the
                // back-substitution below: mr is traversed 3× instead of
                // 12× each way while every accumulator keeps its own
                // ascending-i order and every mr[i] takes its 12
                // decrements in w = 0..11 order — bit-for-bit the
                // one-row-at-a-time result. (A full w/i interchange was
                // measured SLOWER, see the weld PGS note.)
                let mut s = [0.0f64; N_WELD_ROWS];
                for i in 0..nv {
                    let m = mr[i];
                    let jr = &self.jwt[i * N_WELD_ROWS..(i + 1) * N_WELD_ROWS];
                    for w in 0..N_WELD_ROWS {
                        s[w] += jr[w] * m;
                    }
                }
                self.rw[..N_WELD_ROWS].copy_from_slice(&s);
                chol_solve(nweld, &self.lww, &mut self.rw);
                for w in (0..nweld).step_by(4) {
                    let (x0, x1) = (self.rw[w], self.rw[w + 1]);
                    let (x2, x3) = (self.rw[w + 2], self.rw[w + 3]);
                    let m0 = &head[w * nv..(w + 1) * nv];
                    let m1 = &head[(w + 1) * nv..(w + 2) * nv];
                    let m2 = &head[(w + 2) * nv..(w + 3) * nv];
                    let m3 = &head[(w + 3) * nv..(w + 4) * nv];
                    for i in 0..nv {
                        let mut m = mr[i];
                        m -= m0[i] * x0;
                        m -= m1[i] * x1;
                        m -= m2[i] * x2;
                        m -= m3[i] * x3;
                        mr[i] = m;
                    }
                }
                let jr = &self.j[row * nv..(row + 1) * nv];
                let mut dnew = 0.0;
                for i in 0..nv {
                    dnew += jr[i] * mr[i];
                }
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
        for _iter in 0..PGS_ITERS {
            // (a) welds: Δλ_w = A_ww⁻¹ (b_w − J_w v), applied together.
            // (An r/i loop INTERCHANGE was tried here 2026-08-03 —
            // bitwise-clean, but 3.9% SLOWER: at nv = 35 the whole
            // working set is already L1-resident, so there is no
            // streaming win to buy, while the r-inner form's nv-strided
            // loads destroy the unit-stride AVX2 vectorization of these
            // dots/axpys. Reverted.)
            //
            // What DOES win is the same UNROLL-AND-JAM by 4 the A_ww
            // assembly and the weld-null projections use (2026-08-03,
            // R1): i stays inner and unit-stride, one accumulator per
            // weld row, so `v` is traversed 3× per half instead of 12×.
            // BITWISE: each dot keeps its own ascending-i chain, and in
            // the axpy each v[i] still takes its 12 increments in
            // r = 0..11 order (the jam only holds v[i] in a register
            // across four of them) — bit-for-bit the row-at-a-time
            // result. This is the hottest loop in the substep: 12
            // iterations × 12 rows × two nv-wide passes.
            if nweld > 0 {
                debug_assert_eq!(nweld, N_WELD_ROWS);
                // 12 dots against one shared v, in the TRANSPOSED weld
                // layout: 12 lanes, each accumulating its own row's
                // terms over ascending i exactly as the scalar dot did.
                let mut s = [0.0f64; N_WELD_ROWS];
                for i in 0..nv {
                    let vi = v[i];
                    let jr = &self.jwt[i * N_WELD_ROWS..(i + 1) * N_WELD_ROWS];
                    for w in 0..N_WELD_ROWS {
                        s[w] += jr[w] * vi;
                    }
                }
                for r in 0..nweld {
                    self.rw[r] = self.bias[r] - s[r];
                }
                chol_solve(nweld, &self.lww, &mut self.rw);
                for r in (0..nweld).step_by(4) {
                    let (d0, d1) = (self.rw[r], self.rw[r + 1]);
                    let (d2, d3) = (self.rw[r + 2], self.rw[r + 3]);
                    self.lambda[r] += d0;
                    self.lambda[r + 1] += d1;
                    self.lambda[r + 2] += d2;
                    self.lambda[r + 3] += d3;
                    let m0 = &self.minvjt[r * nv..(r + 1) * nv];
                    let m1 = &self.minvjt[(r + 1) * nv..(r + 2) * nv];
                    let m2 = &self.minvjt[(r + 2) * nv..(r + 3) * nv];
                    let m3 = &self.minvjt[(r + 3) * nv..(r + 4) * nv];
                    for i in 0..nv {
                        let mut vi = v[i];
                        vi += m0[i] * d0;
                        vi += m1[i] * d1;
                        vi += m2[i] * d2;
                        vi += m3[i] * d3;
                        v[i] = vi;
                    }
                }
            }
            // (b)+(c) contacts (pair-id order) then joint-limit rows
            // (joint order): projected GS, λ ≥ 0
            for row in nweld..nrows {
                let jr = &self.j[row * nv..(row + 1) * nv];
                let mut jv = 0.0;
                for i in 0..nv {
                    jv += jr[i] * v[i];
                }
                let dl = (self.bias[row] - jv) / self.diag[row];
                let newl = (self.lambda[row] + dl).max(0.0); // λ ≥ 0
                let d = newl - self.lambda[row];
                self.lambda[row] = newl;
                if d != 0.0 {
                    let mr = &self.minvjt[row * nv..(row + 1) * nv];
                    for i in 0..nv {
                        v[i] += mr[i] * d;
                    }
                }
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
                        let plain = self.capcol[jt * nv + vj];
                        if nweld > 0 {
                            debug_assert_eq!(nweld, N_WELD_ROWS);
                            // same by-4 jam as the limit-row projection:
                            // one accumulator per weld row, each still
                            // summing over i ASCENDING, and each col[i]
                            // taking its 12 decrements in w = 0..11
                            // order — bitwise the per-row form.
                            let mut s = [0.0f64; N_WELD_ROWS];
                            {
                                let col = &self.capcol[jt * nv..(jt + 1) * nv];
                                for i in 0..nv {
                                    let c = col[i];
                                    let jr =
                                        &self.jwt[i * N_WELD_ROWS..(i + 1) * N_WELD_ROWS];
                                    for w in 0..N_WELD_ROWS {
                                        s[w] += jr[w] * c;
                                    }
                                }
                            }
                            self.rw[..N_WELD_ROWS].copy_from_slice(&s);
                            chol_solve(nweld, &self.lww, &mut self.rw);
                            // weld minvjt rows live at the head of the
                            // buffer — disjoint from capcol, so the two
                            // fields can be borrowed at once
                            let (capcol, minvjt) = (&mut self.capcol, &self.minvjt);
                            let col = &mut capcol[jt * nv..(jt + 1) * nv];
                            for w in (0..nweld).step_by(4) {
                                let (x0, x1) = (self.rw[w], self.rw[w + 1]);
                                let (x2, x3) = (self.rw[w + 2], self.rw[w + 3]);
                                let m0 = &minvjt[w * nv..(w + 1) * nv];
                                let m1 = &minvjt[(w + 1) * nv..(w + 2) * nv];
                                let m2 = &minvjt[(w + 2) * nv..(w + 3) * nv];
                                let m3 = &minvjt[(w + 3) * nv..(w + 4) * nv];
                                for i in 0..nv {
                                    let mut c = col[i];
                                    c -= m0[i] * x0;
                                    c -= m1[i] * x1;
                                    c -= m2[i] * x2;
                                    c -= m3[i] * x3;
                                    col[i] = c;
                                }
                            }
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
