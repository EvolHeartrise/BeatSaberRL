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
//! Cholesky factor (ws.l). Why not per-row GS for the welds: the two
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

use super::algo::{chol_factor, chol_solve};
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
    /// no-sticking checks).
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
                self.limset.push((j, sign));
                nrows += 1;
            }
        }
        self.nrows = nrows;

        // ── effective masses: M⁻¹Jᵀ via the substep's Cholesky factor ─
        for row in 0..nrows {
            let (jrows, mrows) = (&self.j, &mut self.minvjt);
            let jr = &jrows[row * nv..(row + 1) * nv];
            let mr = &mut mrows[row * nv..(row + 1) * nv];
            mr.copy_from_slice(jr);
            chol_solve(nv, &ws.l, mr);
            let mut d = 0.0;
            for i in 0..nv {
                d += jr[i] * mr[i];
            }
            debug_assert!(d > 0.0, "constraint row {row} has zero effective mass");
            self.diag[row] = d;
            // pre-solve J·v* (introspection only)
            let mut s = 0.0;
            for i in 0..nv {
                s += jr[i] * v[i];
            }
            self.jv_pre[row] = s;
        }

        // ── weld-block effective mass A_ww = J_w M⁻¹ J_wᵀ, factored
        //    once per substep (Divergences §15: per-row Gauss-Seidel
        //    leaves ~0.14 m/s weld residual after 12 sweeps at the
        //    straight-knee stance — the two feet's vertical rows are
        //    near-parallel through the root — so the 12 bilateral rows
        //    are solved as an exact dense block instead) ─────────────
        let nweld = self.weld_rows();
        if nweld > 0 {
            for r in 0..nweld {
                let mr = &self.minvjt[r * nv..(r + 1) * nv];
                for c in r..nweld {
                    let jc = &self.j[c * nv..(c + 1) * nv];
                    let mut s = 0.0;
                    for i in 0..nv {
                        s += jc[i] * mr[i];
                    }
                    // both triangles from one value (bitwise symmetric)
                    self.aww[r * nweld + c] = s;
                    self.aww[c * nweld + r] = s;
                }
            }
            assert!(
                chol_factor(nweld, &self.aww, &mut self.lww),
                "weld block J M^-1 J^T not positive definite"
            );
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
            let split = (nweld + ncon) * nv;
            let (head, tail) = self.minvjt.split_at_mut(split);
            for (li, _lim) in self.limset.iter().enumerate() {
                let row = nweld + ncon + li;
                let mr = &mut tail[li * nv..(li + 1) * nv];
                for w in 0..nweld {
                    let jw = &self.j[w * nv..(w + 1) * nv];
                    let mut s = 0.0;
                    for i in 0..nv {
                        s += jw[i] * mr[i];
                    }
                    self.rw[w] = s;
                }
                chol_solve(nweld, &self.lww, &mut self.rw);
                for w in 0..nweld {
                    let x = self.rw[w];
                    let mw = &head[w * nv..(w + 1) * nv];
                    for i in 0..nv {
                        mr[i] -= mw[i] * x;
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
            // (a) welds: Δλ_w = A_ww⁻¹ (b_w − J_w v), applied together
            if nweld > 0 {
                for r in 0..nweld {
                    let jr = &self.j[r * nv..(r + 1) * nv];
                    let mut jv = 0.0;
                    for i in 0..nv {
                        jv += jr[i] * v[i];
                    }
                    self.rw[r] = self.bias[r] - jv;
                }
                chol_solve(nweld, &self.lww, &mut self.rw);
                for r in 0..nweld {
                    let d = self.rw[r];
                    self.lambda[r] += d;
                    let mr = &self.minvjt[r * nv..(r + 1) * nv];
                    for i in 0..nv {
                        v[i] += mr[i] * d;
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
                    if !self.capcached[jt] {
                        {
                            let col = &mut self.capcol[jt * nv..(jt + 1) * nv];
                            col.fill(0.0);
                            col[vj] = 1.0;
                            chol_solve(nv, &ws.l, col);
                        }
                        let plain = self.capcol[jt * nv + vj];
                        if nweld > 0 {
                            for w in 0..nweld {
                                let jw = &self.j[w * nv..(w + 1) * nv];
                                let col = &self.capcol[jt * nv..(jt + 1) * nv];
                                let mut s = 0.0;
                                for i in 0..nv {
                                    s += jw[i] * col[i];
                                }
                                self.rw[w] = s;
                            }
                            chol_solve(nweld, &self.lww, &mut self.rw);
                            for w in 0..nweld {
                                let x = self.rw[w];
                                // weld minvjt rows live at the head of
                                // the buffer — disjoint from capcol
                                for i in 0..nv {
                                    self.capcol[jt * nv + i] -= self.minvjt[w * nv + i] * x;
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
