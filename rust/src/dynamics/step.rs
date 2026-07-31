//! One physics substep — SPEC-v10 §3, steps 1–6 and 10–11. The
//! ordering below is NORMATIVE; f64 cross-implementation parity depends
//! on every step happening in exactly this order.
//!
//! Steps 7–9 (collision detection + PGS constraint solve) plug in via
//! the `constraints` hook between the unconstrained-velocity solve and
//! position integration — solver.rs `ConstraintSet::solve` is the P2
//! implementation (use `solver::substep_constrained` for the full v10
//! substep). The hook receives the Workspace (ws.l holds the Cholesky
//! factor for the J·M⁻¹·Jᵀ effective-mass solves) plus the mutable
//! velocity to project. Unconstrained callers (P1 tests) pass None.

use super::algo::{chol_factor, chol_solve, crba, rnea};
use super::fk::{fk_all, quat_mul};
use super::model::Model;
use super::{State, Workspace};

/// SPEC-v10 §3 substep h = 1/240 s (8 per 1/30 control tick).
pub const H_SUB: f64 = 1.0 / 240.0;
pub const N_SUB: usize = 8;
/// Gauss-Seidel passes of the step-11 clamp solve (see below).
pub const CLAMP_PASSES: usize = 8;

/// P2 constraint-stage hook (§3 steps 7–9): (model, workspace, q₂₉,
/// v*) → projected velocity written in place. q (pre-integration) is
/// needed for the joint-limit rows folded into the main solver
/// (SPEC-v10 Divergences §15).
pub type ConstraintHook<'a> = &'a mut dyn FnMut(&Model, &mut Workspace, &[f64], &mut [f64]);

/// Zero root wrench — `substep`'s §3 step 4 ("root rows 0").
pub const ZERO_WRENCH: [f64; 6] = [0.0; 6];

/// One substep. `tau_act`: applied actuated torques (nq, §1.4 order;
/// constant across a control tick — the caller holds them). Joint
/// damping −b∘q̇ is added here (§3 step 4); root rows carry no applied
/// force. Panics if M(q) loses positive definiteness — that is a
/// modeling bug, not a runtime condition.
pub fn substep(
    model: &Model,
    st: &mut State,
    tau_act: &[f64],
    h: f64,
    ws: &mut Workspace,
    constraints: Option<ConstraintHook>,
) {
    substep_wrench(model, st, tau_act, &ZERO_WRENCH, h, ws, constraints);
}

/// `substep` with an applied 6-DoF root wrench [force (world), moment
/// about the root origin] added on the free-base rows in step 4. The
/// env never applies one (§3: root rows 0) — this exists for the P2
/// static-stance protocol (full 35-DoF RNEA compensation including the
/// root rows) and oracle harnesses.
pub fn substep_wrench(
    model: &Model,
    st: &mut State,
    tau_act: &[f64],
    root_wrench: &[f64; 6],
    h: f64,
    ws: &mut Workspace,
    constraints: Option<ConstraintHook>,
) {
    substep_opts(model, st, tau_act, root_wrench, None, h, ws, constraints);
}

/// Test-harness variant with OPTIONAL per-joint implicit damping
/// (MuJoCo-Euler semantics: v⁺ = v + h·(M + h·diag(d))⁻¹(τ − C − d∘q̇);
/// d enters M's diagonal before the Cholesky, so the constraint stage
/// works against the same damped operator). The env NEVER uses this —
/// the §3 pipeline's own damping b is explicit (b·h/I ≈ 0.0025, stable)
/// and `substep` passes None, leaving the normative expression tree
/// untouched. It exists because the ORACLE-protocol muscle-tone
/// stabilizer (kd near-critical against the DIAGONAL M_jj) is not
/// explicitly integrable at h = 1/240: at the straight-arm neutral
/// pose shoulder-twist/wrist-roll are collinear, (M⁻¹)_jj ~ 1/armature,
/// and h·kd·(M⁻¹)_jj ≫ 2 — the wrists rattle at the ±40 rad/s cap
/// (measured) exactly as the oracle README describes for the cap-less
/// MuJoCo run. MuJoCo's Euler integrator makes dof damping implicit for
/// the same reason; band comparisons must match that semantics.
pub fn substep_opts(
    model: &Model,
    st: &mut State,
    tau_act: &[f64],
    root_wrench: &[f64; 6],
    implicit_damp: Option<&[f64]>,
    h: f64,
    ws: &mut Workspace,
    constraints: Option<ConstraintHook>,
) {
    let nv = model.nv();
    let voff = model.voff();
    debug_assert_eq!(tau_act.len(), model.nq);

    // 1. kinematics  2. CRBA  3. RNEA(q, v, 0) → bias C
    fk_all(model, st, ws);
    crba(model, ws);
    rnea(model, ws, &st.v, None);

    // 2b (harness only): implicit joint damping onto M's diagonal
    if let Some(d) = implicit_damp {
        for j in 0..model.nq {
            ws.m[(voff + j) * nv + (voff + j)] += h * d[j];
        }
    }

    // 4. applied generalized force τ = τ_gear − b∘q̇ (root rows: the
    //    applied wrench — 0 in the env), assembled directly as the
    //    solve RHS τ − C
    for i in 0..voff {
        ws.rhs[i] = root_wrench[i] - ws.c[i];
    }
    for j in 0..model.nq {
        ws.rhs[voff + j] = tau_act[j] - model.damping[j] * st.v[voff + j] - ws.c[voff + j];
    }
    // 4b (harness only): current-velocity part of the implicit damping
    if let Some(d) = implicit_damp {
        for j in 0..model.nq {
            ws.rhs[voff + j] -= d[j] * st.v[voff + j];
        }
    }

    // 5. dense Cholesky (lower, no pivoting)
    assert!(
        chol_factor(nv, &ws.m, &mut ws.l),
        "mass matrix not positive definite"
    );

    // 6. unconstrained velocity v* = v + h·M⁻¹(τ − C)
    chol_solve(nv, &ws.l, &mut ws.rhs);
    for i in 0..nv {
        ws.vstar[i] = st.v[i] + h * ws.rhs[i];
    }

    // ── constraint stage (§3 steps 7–9) via the hook ───────────────
    // 7. collision detection (55-pair table, margin 0.01)
    // 8. PGS, 12 iterations, fixed row order (welds then contacts)
    // 9. v ← v* + M⁻¹Jᵀλ
    // Implemented by solver::ConstraintSet::solve (P2); None = the
    // unconstrained pipeline.
    if let Some(hook) = constraints {
        let mut v = std::mem::take(&mut ws.vstar);
        hook(model, ws, &st.q, &mut v);
        ws.vstar = v;
    }
    st.v.copy_from_slice(&ws.vstar);
    // ───────────────────────────────────────────────────────────────

    // 10. semi-implicit Euler positions; root quaternion via exp map of
    //     the WORLD angular velocity (Taylor guard |u| < 1e-12)
    for j in 0..model.nq {
        st.q[j] += h * st.v[voff + j];
    }
    if model.free_base {
        for i in 0..3 {
            st.p[i] += h * st.v[i];
        }
        let u = [
            st.v[3] * h * 0.5,
            st.v[4] * h * 0.5,
            st.v[5] * h * 0.5,
        ];
        let un = (u[0] * u[0] + u[1] * u[1] + u[2] * u[2]).sqrt();
        let qe = if un < 1e-12 {
            [1.0, 0.0, 0.0, 0.0]
        } else {
            let (s, c) = un.sin_cos();
            [c, s * u[0] / un, s * u[1] / un, s * u[2] / un]
        };
        let q = quat_mul(qe, st.quat);
        let qn = (q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3]).sqrt();
        st.quat = [q[0] / qn, q[1] / qn, q[2] / qn, q[3] / qn];
    }

    // 11. post-solve per-joint clamps, joint-index order 0..nq, as
    //     MOMENTUM-CONSISTENT internal impulses (P1 divergence from
    //     the spec'd v9-style direct coordinate write — SPEC-v10
    //     Divergences §13): zeroing q̇_j alone discards the joint's
    //     momentum share while the solve's reaction stays in the root,
    //     a pump measured at ~0.5 rad/s of root spin per substep under
    //     1 N·m on every joint — free-fall blow-up past the explicit-
    //     Coriolis stability bound (h·ω ≈ 1) in < 1 s. A physical stop
    //     is an internal impulse: Δv = M⁻¹e_j·λ with λ = (target −
    //     q̇_j)/(M⁻¹)_jj — e_j has zero root rows, so the momentum map
    //     is untouched. CLAMP_PASSES deterministic Gauss-Seidel
    //     passes over the joints in index order (a mini-PGS over the
    //     active clamp rows — one pass is NOT enough: near-collinear
    //     axis pairs (straight arm: twist/roll, (M⁻¹) coupling → ±1)
    //     ping-pong the violation, and the leaked momentum measurably
    //     spins the root up to the explicit-Coriolis stability bound
    //     h·ω ≈ 1). M⁻¹e_j columns come from this substep's factor
    //     (M at the pre-integration q), solved lazily once per joint
    //     and cached. A final direct GUARANTEE pass re-imposes every
    //     bound exactly; by then the residuals are near-null-mode
    //     noise with negligible momentum content. The root is never
    //     clamped. Dissipative as before. (P2 may fold these rows
    //     into the main PGS; the row ORDER joint 0..28 is the
    //     contract either way.)
    for j in 0..model.nq {
        let (lo, hi) = model.limits[j];
        ws.hit[j] = st.q[j] < lo || st.q[j] > hi;
        if ws.hit[j] {
            st.q[j] = st.q[j].clamp(lo, hi);
        }
        ws.cached[j] = false;
    }
    for _pass in 0..CLAMP_PASSES {
        let mut changed = false;
        for j in 0..model.nq {
            let vj = voff + j;
            let cap = model.omega_cap[j];
            let target = if ws.hit[j] { 0.0 } else { st.v[vj].clamp(-cap, cap) };
            if target != st.v[vj] {
                if !ws.cached[j] {
                    ws.sol[..nv].fill(0.0);
                    ws.sol[vj] = 1.0;
                    chol_solve(nv, &ws.l, &mut ws.sol);
                    ws.solcache[j * nv..(j + 1) * nv].copy_from_slice(&ws.sol[..nv]);
                    ws.cached[j] = true;
                }
                let col = &ws.solcache[j * nv..(j + 1) * nv];
                let lam = (target - st.v[vj]) / col[vj];
                for i in 0..nv {
                    st.v[i] += col[i] * lam;
                }
                st.v[vj] = target; // exact — the clamp is a hard bound
                changed = true;
            }
        }
        if !changed {
            break;
        }
    }
    for j in 0..model.nq {
        let vj = voff + j;
        let cap = model.omega_cap[j];
        st.v[vj] = st.v[vj].clamp(-cap, cap);
        if ws.hit[j] {
            st.v[vj] = 0.0;
        }
    }
}
