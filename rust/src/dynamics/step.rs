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

use crate::Real;
use super::algo::{crba, rnea, tc_factor, tc_solve_unit_v, tc_solve_v};
use super::fk::{fk_all, quat_mul};
use super::model::Model;
use super::{State, Workspace};
use super::solver::{pair_box_targets, single_box_target, single_hard_target};

/// SPEC-v10 §3 substep h = 1/240 s (8 per 1/30 control tick).
pub const H_SUB: Real = 1.0 / 240.0;
pub const N_SUB: usize = 8;
/// Gauss-Seidel passes of the step-11 clamp solve (see below). env v14
/// (2026-08-10, phantom-force audit): 8 → 64. The passes early-exit on
/// a clean sweep, so the typical substep is unchanged; the budget only
/// matters for near-collinear axis pairs and many-joints-pinned poses,
/// which previously fell through to a non-reciprocal direct-write
/// guarantee pass (deleted — see the step-11 comment).
pub const CLAMP_PASSES: usize = 64;
/// v14 joint-stop Baumgarte factor (the ground rows' β, reused): the
/// per-substep fraction of a stop violation converted into a recovery
/// rate by the step-11 impulse target.
pub const BETA_LIMIT: Real = 0.2;
/// v14 joint-stop slop (rad past a stop tolerated at rest — the ground
/// rows' SLOP pattern). Inside the slop the rate target is 0 and a
/// parked joint fires NO impulses, so steady-state stop-resting leaks
/// nothing; recovery engages only beyond it. 2 mrad ≈ 0.1°.
pub const SLOP_LIMIT: Real = 0.002;
/// v14 deep-violation position backstop (rad past a stop). Routine
/// stop traffic never reaches it — one-substep overshoot is bounded by
/// h·ω_cap ≈ 0.17 rad — so its rare reaction-free clamp is a bounded
/// safety net of the root-envelope class, not part of the physics.
pub const LIMIT_TELEPORT_BACKSTOP: Real = 0.35;
/// Step-12 root velocity envelope (P3 — Divergences §19; derived on the
/// v10 welded-feet body, and it still bounds a standing body — a falling
/// one terminates on the env's FALL_ROOT_Y): the humanoid's root
/// reaches the ground through a 6-revolute leg chain
/// whose rates are capped at 16 rad/s each, so |ω_root| ≤ 96 rad/s and
/// the linear velocity is bounded by the same rates × the ≤ ~0.5 m
/// root–foot levers (48 m/s). Beyond = solver artifact (see step 12).
pub const ROOT_W_MAX: Real = 96.0;
pub const ROOT_V_MAX: Real = 48.0;

/// P2 constraint-stage hook (§3 steps 7–9): (model, workspace, q₂₉,
/// v*) → projected velocity written in place. q (pre-integration) is
/// needed for the joint-limit rows folded into the main solver
/// (SPEC-v10 Divergences §15).
pub type ConstraintHook<'a> = &'a mut dyn FnMut(&Model, &mut Workspace, &[Real], &mut [Real]);

/// Zero root wrench — `substep`'s §3 step 4 ("root rows 0").
pub const ZERO_WRENCH: [Real; 6] = [0.0; 6];

/// One substep. `tau_act`: applied actuated torques (nq, §1.4 order;
/// constant across a control tick — the caller holds them). Joint
/// damping −b∘q̇ is added here (§3 step 4); root rows carry no applied
/// force. If M(q) loses positive definiteness the Cholesky pivot floor
/// engages (SPEC-v10 Divergences §22): pivots are clamped at
/// PD_FLOOR_F64 and counted into ws.pd_hits — the sim degrades
/// gracefully instead of panicking (greedy contorted policies reach
/// gimbal-adjacent states in long benches; §21's caps + clamps +
/// envelope bound the resulting velocities).
pub fn substep(
    model: &Model,
    st: &mut State,
    tau_act: &[Real],
    h: Real,
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
    tau_act: &[Real],
    root_wrench: &[Real; 6],
    h: Real,
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
    tau_act: &[Real],
    root_wrench: &[Real; 6],
    implicit_damp: Option<&[Real]>,
    h: Real,
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
    // 4c. exp/wrist-tendon: the passive wrist torque τ_p(q) (model.rs
    //     `PassiveWrist`) joins the APPLIED side — the four wrist rows
    //     are re-assembled as (τ_gear + τ_p) − b·q̇ − C. State-dependent:
    //     evaluated at this substep's pre-integration q. Explicit, like
    //     the damping (stability: tests_tendon::stiffness_*). The hard
    //     joint limits are untouched. Not muscle effort: the env never
    //     adds it to Body::tau (energy price / torque telemetry).
    if let Some(pw) = &model.passive_wrist {
        let tp = pw.eval(model.limits, &st.q);
        for (s, &(jb, jd)) in pw.hands.iter().enumerate() {
            let (tb, td, r) = tp[s];
            ws.rhs[voff + jb] = (tau_act[jb] + tb) - model.damping[jb] * st.v[voff + jb]
                - ws.c[voff + jb];
            ws.rhs[voff + jd] = (tau_act[jd] + td) - model.damping[jd] * st.v[voff + jd]
                - ws.c[voff + jd];
            ws.tendon[s] = [tb, td, r];
        }
    }
    // 4b (harness only): current-velocity part of the implicit damping
    if let Some(d) = implicit_damp {
        for j in 0..model.nq {
            ws.rhs[voff + j] -= d[j] * st.v[voff + j];
        }
    }

    // 5. Cholesky (lower, no pivoting), pivots floored at PD_FLOOR_F64
    //    with engagements counted (Divergences §22 — the f64 analogue of
    //    the CUDA f32 §20e net). audit #10 (Divergences §25): factored in
    //    the fill-free LEAVES-FIRST elimination order of the kinematic
    //    tree (algo.rs `TreeChol`), from `ws.m` (M, v-order) into `ws.l`
    //    (L, e-order profile) — the same M and the same no-pivot
    //    Cholesky, only a different data-independent elimination order,
    //    so L carries exactly the tree's ancestor sparsity (factor ~1.7k
    //    MACs, was ~7.1k; a dense solve ~612, was ~1,225). `ws.m` keeps
    //    M from here to the end of the substep; nothing reads it.
    ws.pd_hits += tc_factor(&ws.tc, &ws.m, &mut ws.l) as u64;

    // 6. unconstrained velocity v* = v + h·M⁻¹(τ − C)
    tc_solve_v(&ws.tc, &ws.l, &mut ws.rhs, &mut ws.esc);
    for i in 0..nv {
        ws.vstar[i] = st.v[i] + h * ws.rhs[i];
    }

    // ── constraint stage (§3 steps 7–9) via the hook ───────────────
    // 7. collision detection (55-pair table, margin 0.01)
    // 8. PGS, 12 iterations, fixed row order (the ground block, then
    //    contacts, then limits, then the ω-cap pass)
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

    // 12-PRE. root velocity envelope, FIRST application (SPEC-v10 §3
    //     step 12 as amended 2026-08-03 — Divergences §23). Identical
    //     expression tree to the post-step-11 application at the bottom
    //     of this function; run HERE it is the one that bounds what the
    //     integrator consumes: ‖Δp‖ ≤ h·ROOT_V_MAX and |Δθ| ≤ h·ROOT_W_MAX
    //     per substep, UNCONDITIONALLY, and hence the next substep's
    //     Baumgarte weld bias (β/h·Δp ≤ 9.6 m/s instead of a measured
    //     1.9e3 m/s). Without it a divergent step-8 solve (the §23
    //     oblique-projector mechanism) teleports the body hundreds of
    //     metres in one substep and the weld bias then feeds the
    //     divergence. Engages ONLY on artifact states — a converged v10
    //     policy measured 0 engagements in 204,800 env-ticks — so it is
    //     bitwise transparent on healthy trajectories. Engagements are
    //     counted (ws.env_hits): the env layer terminates the episode on
    //     a nonzero per-tick count (the trajectory is no longer physics).
    if model.free_base {
        let mut engaged = false;
        let w2 = st.v[3] * st.v[3] + st.v[4] * st.v[4] + st.v[5] * st.v[5];
        if w2 > ROOT_W_MAX * ROOT_W_MAX {
            let s = ROOT_W_MAX / w2.sqrt();
            st.v[3] *= s;
            st.v[4] *= s;
            st.v[5] *= s;
            engaged = true;
        }
        let l2 = st.v[0] * st.v[0] + st.v[1] * st.v[1] + st.v[2] * st.v[2];
        if l2 > ROOT_V_MAX * ROOT_V_MAX {
            let s = ROOT_V_MAX / l2.sqrt();
            st.v[0] *= s;
            st.v[1] *= s;
            st.v[2] *= s;
            engaged = true;
        }
        if engaged {
            ws.env_hits += 1;
        }
    }

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
    //     and cached. The root is never clamped. Dissipative as
    //     before. (P2 may fold these rows into the main PGS; the row
    //     ORDER joint 0..28 is the contract either way.)
    //
    //     env v14 (2026-08-10): the final direct GUARANTEE pass that
    //     used to follow the Gauss-Seidel passes is DELETED. Its
    //     premise ("by then the residuals are near-null-mode noise
    //     with negligible momentum content") is FALSE exactly when
    //     many joints pin simultaneously: a trained policy parked
    //     8-13 joints on stops and the pass's reaction-free writes
    //     (st.v[vj] = 0) summed to a measured +250-335 N of phantom
    //     support — 27-46% of body weight, sustained for 12 s — plus
    //     15.6 m of free-fall deviation over 4 s in a mid-air repro
    //     (three-agent audit, 2026-08-08). Bounds
    //     are now enforced only by momentum-consistent impulses; a
    //     residual bound violation after CLAMP_PASSES leaks POSITION
    //     error into the next substep's clamp (physically a slightly
    //     soft stop), never momentum. Gated by the v14 momentum/CoP
    //     suites (tests_dyn / tests_ground).
    for j in 0..model.nq {
        let (lo, hi) = model.limits[j];
        ws.hit[j] = st.q[j] < lo || st.q[j] > hi;
        // env v14: NO routine position teleport — the parked-on-stops
        // gate measures the reaction-free q-write at ~26 mm of CoM
        // drift per 2 s (it was the last non-reciprocal writer once
        // the guarantee pass died). Position recovery is carried by
        // the Baumgarte term in the impulse target below. The clamp
        // survives only as a deep-violation backstop that no sane
        // trajectory reaches (one-substep overshoot is ≤ h·cap =
        // 0.17 rad; the backstop sits at 0.35 rad).
        if ws.hit[j] {
            let over = if st.q[j] < lo { lo - st.q[j] } else { st.q[j] - hi };
            if over > LIMIT_TELEPORT_BACKSTOP {
                st.q[j] = st.q[j].clamp(lo, hi);
            }
        }
        ws.cached[j] = false;
    }
    // R26 (feature "stats"): step-11 clamp column solves for THIS
    // substep. `ws.cached` is cleared in the loop just above, so the
    // per-substep total is at most nq however many Gauss-Seidel passes
    // run; the counter therefore lives outside the pass loop and is
    // flushed once after it. Compiled out by default.
    #[cfg(feature = "stats")]
    let mut s11_solves = 0u64;
    // v14: at a stop the rate target is a Baumgarte recovery toward the
    // violated bound (β = BETA_LIMIT, the ground rows' push-out pattern)
    // instead of dead zero — the impulse then restores POSITION too,
    // reciprocally. Dead-band: inside SLOP_LIMIT a parked joint targets
    // exactly 0 — once its rate dies no further impulse fires, so resting
    // on a stop is leak-free.
    let target_of = |j: usize, q: Real, vj: Real, hit: bool| -> Real {
        let cap = model.omega_cap[j];
        if hit {
            let (lo, hi) = model.limits[j];
            let over = if q < lo { lo - q } else { hi - q };
            let rec = over.abs() - SLOP_LIMIT;
            if rec <= 0.0 {
                0.0
            } else {
                (BETA_LIMIT * over.signum() * rec / h).clamp(-cap, cap)
            }
        } else {
            vj.clamp(-cap, cap)
        }
    };
    // MIRROR-EXACT (sim/mirror-exact, 2026-10-04): each pass visits the
    // joints ORBIT by orbit (`Model::mirror`) — a central joint alone, a
    // mirror pair decided on the same `v` and, when either fires,
    // projected jointly by the exact 2-variable box LCP (solver.rs
    // `pair_box_targets`), so neither side is clamped first.
    for _pass in 0..CLAMP_PASSES {
        let mut changed = false;
        for j0 in 0..model.nq {
            let jm = model.mirror[j0];
            if jm < j0 {
                continue;
            }
            let (va, vb) = (voff + j0, voff + jm);
            let ta = target_of(j0, st.q[j0], st.v[va], ws.hit[j0]);
            let fa = ta != st.v[va];
            if jm == j0 {
                if fa {
                    #[cfg(feature = "stats")]
                    {
                        s11_solves += s11_column(model, ws, j0) as u64;
                    }
                    #[cfg(not(feature = "stats"))]
                    s11_column(model, ws, j0);
                    let col = &ws.solcache[j0 * nv..(j0 + 1) * nv];
                    single_hard_target(&mut st.v, nv, va, ta, col, col[va]);
                    changed = true;
                }
                continue;
            }
            let tb = target_of(jm, st.q[jm], st.v[vb], ws.hit[jm]);
            let fb = tb != st.v[vb];
            if !(fa || fb) {
                continue;
            }
            // one twin off its interval: its single update, unless that
            // pushes the other twin off its own (solver.rs
            // `single_box_target` — needs only the firing twin's column)
            if fa != fb {
                let (js, jo, vs, vo, ts, to) =
                    if fa { (j0, jm, va, vb, ta, tb) } else { (jm, j0, vb, va, tb, ta) };
                #[cfg(feature = "stats")]
                {
                    s11_solves += s11_column(model, ws, js) as u64;
                }
                #[cfg(not(feature = "stats"))]
                s11_column(model, ws, js);
                let col = &ws.solcache[js * nv..(js + 1) * nv];
                let co = model.omega_cap[jo];
                let (lo_o, hi_o) = if ws.hit[jo] { (to, to) } else { (-co, co) };
                if single_box_target(&mut st.v, nv, vs, vo, ts, lo_o, hi_o, col, col[vs]) {
                    changed = true;
                    continue;
                }
            }
            // both columns: the box LCP couples the pair
            for jt in [j0, jm] {
                #[cfg(feature = "stats")]
                {
                    s11_solves += s11_column(model, ws, jt) as u64;
                }
                #[cfg(not(feature = "stats"))]
                s11_column(model, ws, jt);
            }
            let ca = &ws.solcache[j0 * nv..(j0 + 1) * nv];
            let cb = &ws.solcache[jm * nv..(jm + 1) * nv];
            // interval per joint: a joint past a stop has the equality
            // target (lo = hi), a free one its ±cap box
            let (ca_, cb_) = (model.omega_cap[j0], model.omega_cap[jm]);
            let (lo_a, hi_a) = if ws.hit[j0] { (ta, ta) } else { (-ca_, ca_) };
            let (lo_b, hi_b) = if ws.hit[jm] { (tb, tb) } else { (-cb_, cb_) };
            if pair_box_targets(&mut st.v, nv, va, vb, lo_a, hi_a, lo_b, hi_b, ca, cb,
                                ca[va], cb[vb]) {
                changed = true;
            }
        }
        if !changed {
            break;
        }
    }
    // R26 (feature "stats"): flush the step-11 solve count and tick the
    // shared denominator. `nsubsteps` is bumped ONCE per substep here —
    // it is the denominator for the solver-side counters too, which is
    // exact under bench traffic (every substep runs the constraint hook
    // via `substep_constrained`) but would over-count for a P1-style
    // unconstrained caller. Don't mix the two in one collection window.
    #[cfg(feature = "stats")]
    {
        ws.stats.s11_solves_sum += s11_solves;
        ws.stats.s11_solves_max = ws.stats.s11_solves_max.max(s11_solves);
        ws.stats.nsubsteps += 1;
    }
    // (env v14: no direct-write guarantee pass here — see the step-11
    // comment. The Gauss-Seidel impulse passes above are the only
    // writers of joint rates in this block.)

    // 12. root velocity envelope guarantee, SECOND application (ADDED
    //     at P3 — SPEC-v10 Divergences §19b; the first application is
    //     the 12-PRE block above, Divergences §23). This one bounds the
    //     STORED velocity: step 11's momentum-consistent joint impulses
    //     Δv = M⁻¹e_j·λ have non-zero ROOT rows, so the root can be
    //     back over the envelope after step 10 has already run. It is
    //     deliberately NOT counted into ws.env_hits — the pre-integration
    //     application is the escape signal (it is the superset in the
    //     measured adversarial sweeps and the one whose consequences
    //     are unbounded). With the feet welded and every joint rate
    //     capped, the TRUE root velocity is kinematically bounded (the
    //     root connects to the ground through a 6-joint leg chain:
    //     |ω_root| ≤ 6×16 = 96 rad/s, and the linear velocity by the
    //     same rates × the ≤ ~0.5 m root–foot levers). Sustained
    //     random full gears reach contorted cap-riding poses where the
    //     EXPLICIT Coriolis term spikes (h·M⁻¹C-level Δv ≈ 150 in one
    //     substep — the §14 h·ω ≈ 1 instability, which P0 assumed the
    //     welded env could not produce); the solve caps every joint
    //     but the root rows are never clamped, so the root keeps the
    //     artifact velocity, the next substep's Coriolis feeds on it,
    //     and the state diverges to a Cholesky PD panic within a tick
    //     (measured: |ω_root| 15 → 5.5e5 rad/s; 512-env soak). Any
    //     root velocity beyond the envelope is therefore solver/
    //     integrator ARTIFACT, not physics: rescale the root angular /
    //     linear velocity vectors back onto the envelope (deterministic
    //     two-scalar guard, direction-preserving; engages only in
    //     artifact states — ordinary motion never comes near 96).
    if model.free_base {
        let w2 = st.v[3] * st.v[3] + st.v[4] * st.v[4] + st.v[5] * st.v[5];
        if w2 > ROOT_W_MAX * ROOT_W_MAX {
            let s = ROOT_W_MAX / w2.sqrt();
            st.v[3] *= s;
            st.v[4] *= s;
            st.v[5] *= s;
        }
        let l2 = st.v[0] * st.v[0] + st.v[1] * st.v[1] + st.v[2] * st.v[2];
        if l2 > ROOT_V_MAX * ROOT_V_MAX {
            let s = ROOT_V_MAX / l2.sqrt();
            st.v[0] *= s;
            st.v[1] *= s;
            st.v[2] *= s;
        }
    }
}

/// Step 11's lazily solved, per-substep-cached column M⁻¹e_j (unit RHS
/// +e_vj — audit #10: the forward pass runs only e_vj's ancestor chain,
/// ≤ 16 rows, bitwise the full forward — `tc_fwd_unit_lane` — then the
/// dense back pass). Returns whether a real solve ran (R26's event).
///
/// R23d REVERTED (2026-08-07): solving STRAIGHT into `ws.solcache` and
/// deleting `ws.sol` is bitwise-identical but measured −0.84% on top of
/// the landed R23 package — a `chol_*` callee wants a dedicated,
/// loop-invariant scratch slice, so the staging buffer is the FAST form.
fn s11_column(model: &Model, ws: &mut Workspace, j: usize) -> bool {
    if ws.cached[j] {
        return false;
    }
    let nv = model.nv();
    let vj = model.voff() + j;
    tc_solve_unit_v(&ws.tc, &ws.l, vj, 1.0, &mut ws.sol[..nv], &mut ws.esc);
    ws.solcache[j * nv..(j + 1) * nv].copy_from_slice(&ws.sol[..nv]);
    ws.cached[j] = true;
    true
}
