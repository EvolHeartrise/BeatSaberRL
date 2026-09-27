//! P2 constraint gates: static stance (full-RNEA hold on the free-
//! floating body), ground integrity under random full gears and under a
//! scripted two-arm swing, non-penetration + unilaterality of the
//! contact rows, no energy injection, bitwise determinism with contacts,
//! and the solver's structural bitwise claims (R10 / R13 / R23).
//!
//! The v10 versions of the stance / drift / swing gates ran on the
//! welded-feet harness (removed 2026-09-27, issue #2); the ground-block
//! gates that replace their weight-support half live in tests_ground.rs
//! (`no_pull_and_weight_is_carried`, `settle_stands_for_five_seconds`).
//!
//! The muscle-tone scenarios follow the oracle protocol (README): the
//! stabilizer's near-critical kd is applied IMPLICITLY
//! (`substep_constrained_damped`) — MuJoCo-Euler damping semantics; the
//! spring kp stays an explicit applied torque. The production body
//! (spec damping b, ω caps) never needs this.

use super::algo::energy;
use super::fk::fk_all;
use super::model::{HUMANOID, NQ};
use super::solver::{
    substep_constrained, substep_constrained_damped, substep_constrained_wrench, ConstraintSet,
};
use super::testutil::{gravity_comp, muscle_gains, muscle_model, Rng};
use super::{State, Workspace};

const H: f64 = 1.0 / 240.0;
const DEG: f64 = std::f64::consts::PI / 180.0;

/// The env's constraint set: self-collision rows + the ground block.
fn contact_set(m: &super::model::Model) -> ConstraintSet {
    ConstraintSet::new(m, true, true)
}

/// Gate (a), stance protocol (oracle check_static b2) on the FREE-
/// FLOATING body (no ground; the v10 gate held the welded body):
/// constant full 35-DoF RNEA(q_ref, 0, 0) compensation INCLUDING the
/// root wrench, 2 s at h = 1/240. The compensation cancels gravity
/// bitwise, so drift is rounding noise on a knife-edge equilibrium
/// (constant-torque modes are unstable — measured e-fold ~0.35 s from
/// 1e-17 seeds on the welded body), which is why the horizon is the
/// protocol's 2 s and not SPEC §7 P2's 10 s (Divergences §17). Pins the
/// RNEA / CRBA / constraint-stage consistency: joints AND root stay put.
#[test]
fn static_stance_full_rnea_hold() {
    let m = &HUMANOID;
    let mut ws = Workspace::new(m);
    let mut st = State::neutral(m);
    let p0 = st.p;
    let (wrench, tau) = gravity_comp(&[0.0; NQ]);
    let mut cs = ConstraintSet::new(m, true, false);
    let mut max_drift = 0.0f64;
    let mut max_root = 0.0f64;
    for _ in 0..480 {
        substep_constrained_wrench(m, &mut st, &tau, &wrench, H, &mut ws, &mut cs);
        for j in 0..NQ {
            max_drift = max_drift.max(st.q[j].abs());
        }
        for k in 0..3 {
            max_root = max_root.max((st.p[k] - p0[k]).abs());
        }
    }
    println!("static stance 2 s (floating): max joint drift {max_drift:e} rad, root drift {max_root:e} m");
    assert!(max_drift <= 1e-6, "joint drift {max_drift:e}");
    assert!(max_root <= 1e-6, "root drift {max_root:e}");
}

/// Gate (b): 10 s of per-tick random FULL-gear torques on all 29 (fixed
/// seed) on the env constraint set (ground + self-collision), raw
/// production body (spec damping, ω caps — this doubles as the
/// integrability stress test). The body falls and thrashes on the
/// floor; every substep must stay finite and the sole corners must not
/// sink: the ground block's Baumgarte push-out bounds the penetration.
/// (The v10 twin gated weld drift on the welded body.)
#[test]
fn ground_integrity_random_full_gears() {
    let m = &HUMANOID;
    let mut ws = Workspace::new(m);
    let mut st = State::neutral(m);
    let mut cs = contact_set(m);
    let mut rng = Rng(0x5EED_0000_0000_0020);
    let mut tau = [0.0; NQ];
    let mut max_pen = 0.0f64;
    let mut ground_substeps = 0usize;
    for _tick in 0..300 {
        for t in tau.iter_mut() {
            let g = [0.0, 1.0, -1.0, 0.25, -0.25][rng.next_u64() as usize % 5];
            *t = g;
        }
        for j in 0..NQ {
            tau[j] *= m.tau_max[j];
        }
        for _ in 0..8 {
            substep_constrained(m, &mut st, &tau, H, &mut ws, &mut cs);
            for c in cs.ground_contacts.iter() {
                max_pen = max_pen.max(-c.gap);
            }
            ground_substeps += !cs.ground_contacts.is_empty() as usize;
            assert!(st.q.iter().all(|x| x.is_finite()));
            assert!(st.v.iter().all(|x| x.is_finite()));
            assert!(st.p.iter().all(|x| x.is_finite()));
        }
    }
    println!(
        "random full gears 10 s (contact): max sole penetration {max_pen:.4e} m over \
         {ground_substeps} grounded substeps, root y {:.3}, pd_hits {}",
        st.p[1], ws.pd_hits
    );
    assert!(ground_substeps > 1000, "the feet barely touched the ground ({ground_substeps})");
    assert!(max_pen < PEN_RANDOM_GEARS, "sole penetration {max_pen:e} m");
}
/// measured 3.47e-2 m (2026-09-27)
const PEN_RANDOM_GEARS: f64 = 5e-2;

/// SPEC-v10 §7 P2 "stance under load", contact mode: scripted two-arm
/// swinging on the muscle-tone stance (the raw unstabilized body simply
/// falls over — oracle README), 10 s. The body must keep standing on
/// all eight corners and the soles must not creep: foot-frame
/// horizontal drift stays under the static-friction gate's hold bound.
/// (The v10 twin gated weld pose violation ≤ 1e-4 on the welded body.)
#[test]
fn stance_under_two_arm_swing() {
    let m = muscle_model();
    let (kp, kd) = muscle_gains();
    let mut ws = Workspace::new(&m);
    let mut st = State::neutral(&m);
    let mut cs = contact_set(&m);
    fk_all(&m, &st, &mut ws);
    let feet0 = [ws.p[super::model::F_L_FOOT], ws.p[super::model::F_R_FOOT]];
    let mut tau = [0.0; NQ];
    let two_pi = 2.0 * std::f64::consts::PI;
    let mut min_active = usize::MAX;
    let mut max_slip = 0.0f64;
    for k in 0..2400usize {
        let t = k as f64 * H;
        for j in 0..NQ {
            tau[j] = kp[j] * (0.0 - st.q[j]);
        }
        for (a0, ph) in [(3usize, 0.0f64), (10, 0.5 * std::f64::consts::PI)] {
            tau[a0] += 100.0 * (two_pi * 1.5 * t + ph).sin(); // pitch
            tau[a0 + 1] += 75.0 * (two_pi * 1.0 * t + ph).sin(); // yaw
            tau[a0 + 3] += 18.0 * (two_pi * 2.0 * t + ph).sin(); // elbow
            tau[a0 + 5] += 1.0 * (two_pi * 3.0 * t + ph).sin(); // bend
        }
        substep_constrained_damped(&m, &mut st, &tau, &kd, H, &mut ws, &mut cs);
        min_active = min_active.min(cs.ground_contacts.len());
        fk_all(&m, &st, &mut ws);
        for (f, fr) in [super::model::F_L_FOOT, super::model::F_R_FOOT].into_iter().enumerate() {
            let p = ws.p[fr];
            let d = ((p[0] - feet0[f][0]).powi(2) + (p[2] - feet0[f][2]).powi(2)).sqrt();
            max_slip = max_slip.max(d);
        }
        assert!(st.v.iter().all(|x| x.is_finite()));
    }
    println!(
        "two-arm swing 10 s (contact): min active corners {min_active}, max foot slip {max_slip:e} m, root y {:.4}",
        st.p[1]
    );
    assert_eq!(min_active, super::collide::NUM_GROUND_CORNERS, "a corner left the ground");
    assert!(max_slip <= SLIP_SWING, "foot slip {max_slip:e} m");
}
/// measured 4.48e-3 m (2026-09-27)
const SLIP_SWING: f64 = 1e-2;

// ── contact-press scenarios (oracle check_contacts protocol) ─────────

/// Crossed-wrist pose A: forearm–forearm (pair 31) is the live contact.
fn pose_a() -> [f64; NQ] {
    let mut q = [0.0; NQ];
    q[3] = 4.0 * DEG; // l_arm_pitch (slightly up)
    q[4] = 22.0 * DEG; // l_arm_yaw (inward, toward +x)
    q[10] = -4.0 * DEG; // r_arm_pitch (slightly down)
    q[11] = -22.0 * DEG; // r_arm_yaw (inward, toward −x)
    q
}

/// Forearm-on-chest pose B: torso–farm_l (pair 0) is the live contact.
fn pose_b() -> [f64; NQ] {
    let mut q = [0.0; NQ];
    q[3] = -25.0 * DEG; // l_arm_pitch
    q[4] = 38.0 * DEG; // l_arm_yaw
    q[5] = -30.0 * DEG; // l_arm_twist
    q[6] = 120.0 * DEG; // l_elbow
    q
}

struct PressStats {
    max_pen: f64,
    pair_impulse: [f64; super::collide::NUM_PAIRS],
    max_comp: f64, // max λ·max(J·v_post − b, 0) (complementarity)
    min_lambda: f64,
}

/// One press run: spring-hold `pose` (gravity-compensated on the joint
/// rows, kd implicit) plus the constant `extra` torque, `n` substeps.
fn run_press(
    st: &mut State,
    ws: &mut Workspace,
    cs: &mut ConstraintSet,
    pose: &[f64; NQ],
    extra: &[(usize, f64)],
    n: usize,
) -> PressStats {
    let m = muscle_model();
    let (kp, kd) = muscle_gains();
    let (_w, tau_g) = gravity_comp(pose);
    let mut stats = PressStats {
        max_pen: 0.0,
        pair_impulse: [0.0; super::collide::NUM_PAIRS],
        max_comp: 0.0,
        min_lambda: 0.0,
    };
    let mut tau = [0.0; NQ];
    for _ in 0..n {
        for j in 0..NQ {
            tau[j] = tau_g[j] + kp[j] * (pose[j] - st.q[j]);
        }
        for &(j, e) in extra {
            tau[j] += e;
        }
        substep_constrained_damped(&m, st, &tau, &kd, H, ws, cs);
        let nground = cs.ground_rows();
        for (i, c) in cs.contacts.iter().enumerate() {
            let row = nground + i;
            let lam = cs.lambda[row];
            stats.max_pen = stats.max_pen.max(-c.gap);
            stats.pair_impulse[c.pair] += lam;
            stats.min_lambda = stats.min_lambda.min(lam);
            let jv_post = cs.jv(row, &st.v);
            stats.max_comp = stats.max_comp.max(lam * (jv_post - cs.bias[row]).max(0.0));
        }
    }
    stats
}

/// Gate (c): non-penetration + unilaterality. Both oracle contact-probe
/// scenarios: press 1 s (required pair active, max penetration < 2×slop
/// = 0.01 m, λ ≥ 0, complementarity holds — impulse only on rows at
/// their velocity bound), then a strong reversal: once the pair has
/// separated past the margin it must draw exactly zero impulse.
#[test]
fn contact_press_non_penetration() {
    let m = muscle_model();
    for (name, pose, extra, rev, want_pair) in [
        (
            "A crossed wrists",
            pose_a(),
            [(4usize, 5.0f64), (11, -5.0)],
            [(4usize, -20.0f64), (11, 20.0)],
            31usize, // farm_l–farm_r
        ),
        (
            "B forearm on chest",
            pose_b(),
            [(4, 5.0), (6, 3.0)],
            [(4, -20.0), (6, -12.0)],
            0, // torso–farm_l
        ),
    ] {
        let mut ws = Workspace::new(&m);
        let mut st = State::neutral(&m);
        st.q.copy_from_slice(&pose);
        let mut cs = ConstraintSet::new(&m, true, true);
        let s = run_press(&mut st, &mut ws, &mut cs, &pose, &extra, 240);
        println!(
            "{name}: press max pen {:.4} m, pair {} impulse {:.4} N·s, comp {:.2e}",
            s.max_pen, want_pair, s.pair_impulse[want_pair], s.max_comp
        );
        assert!(
            s.pair_impulse[want_pair] > 0.0,
            "{name}: required pair {want_pair} never force-active"
        );
        assert!(s.max_pen < 0.01, "{name}: penetration {} ≥ 2×slop", s.max_pen);
        assert!(s.min_lambda >= 0.0, "{name}: negative contact impulse");
        assert!(s.max_comp < 1e-3, "{name}: complementarity residual {}", s.max_comp);
        // strong reversal: 0.35 s of opening, then a 0.15 s tail in
        // which the separated pair must carry no impulse at all
        let s2 = run_press(&mut st, &mut ws, &mut cs, &pose, &rev, 84);
        let s3 = run_press(&mut st, &mut ws, &mut cs, &pose, &rev, 36);
        println!(
            "{name}: release comp {:.2e}, tail pair impulse {:.3e} N·s",
            s2.max_comp, s3.pair_impulse[want_pair]
        );
        assert!(s2.min_lambda >= 0.0 && s3.min_lambda >= 0.0);
        assert!(s2.max_comp < 1e-3, "{name}: release complementarity");
        assert_eq!(
            s3.pair_impulse[want_pair], 0.0,
            "{name}: impulse on a separated contact"
        );
    }
}

/// Gate (d): no energy injection. Contact-press scenario B, 2 s: total
/// mechanical energy must never rise across a substep by more than the
/// work done by the applied torques (τ·Δq, exact for constant τ over a
/// substep; the implicit kd only dissipates) plus the Baumgarte bound;
/// windowed drift ≤ 1 J over any 1 s window (SPEC-v10 §7 P2). Reports
/// the measured bounds.
///
/// Runs on the FREE-FLOATING body (self-collision rows, no ground): the
/// gate is about the pair rows, and the v10 version's welds were exact
/// bilateral rows that did no work. On the env constraint set the
/// ground block's box-clamped friction sweep is not exactly dissipative
/// per substep at 12 iterations (measured 9e-6 J over one substep of
/// this press, 2026-09-27) — ground energy is gated by tests_ground.rs.
#[test]
fn energy_no_injection_contact_press() {
    let m = muscle_model();
    let (kp, kd) = muscle_gains();
    let pose = pose_b();
    let (_w, tau_g) = gravity_comp(&pose);
    let mut ws = Workspace::new(&m);
    let mut st = State::neutral(&m);
    st.q.copy_from_slice(&pose);
    let mut cs = ConstraintSet::new(&m, true, false);
    let mut tau = [0.0; NQ];
    let n = 480usize;
    let mut excess = vec![0.0f64; n]; // ΔE − W per substep
    let mut max_excess = 0.0f64;
    for k in 0..n {
        for j in 0..NQ {
            tau[j] = tau_g[j] + kp[j] * (pose[j] - st.q[j]);
        }
        tau[4] += 5.0;
        tau[6] += 3.0;
        fk_all(&m, &st, &mut ws);
        let e0 = energy(&m, &mut ws, &st.v);
        let q0: Vec<f64> = st.q.clone();
        substep_constrained_damped(&m, &mut st, &tau, &kd, H, &mut ws, &mut cs);
        fk_all(&m, &st, &mut ws);
        let e1 = energy(&m, &mut ws, &st.v);
        let mut w = 0.0;
        for j in 0..NQ {
            w += tau[j] * (st.q[j] - q0[j]);
        }
        excess[k] = (e1 - e0) - w;
        max_excess = max_excess.max(excess[k]);
        // Baumgarte work bound: the impulses do work only through the
        // bias velocities they are asked to produce — Σ max(λ·b, 0)
        let mut bau = 0.0;
        for r in 0..cs.nrows {
            bau += (cs.lambda[r] * cs.bias[r]).max(0.0);
        }
        assert!(
            excess[k] <= bau + 1e-9,
            "substep {k}: injection {:.3e} J exceeds Baumgarte bound {:.3e} J",
            excess[k],
            bau
        );
    }
    let mut max_window = f64::NEG_INFINITY;
    for k in 0..(n - 240) {
        let s: f64 = excess[k..k + 240].iter().sum();
        max_window = max_window.max(s);
    }
    println!(
        "energy: max per-substep injection {max_excess:e} J (within the per-substep Baumgarte work bound), max 1 s window {max_window:e} J"
    );
    assert!(max_window <= 1.0, "1 s window energy rise {max_window:e} J");
}

/// Gate (f): contact-rich trajectory bitwise-identical across two runs
/// (state AND impulse stream).
#[test]
fn determinism_with_contacts() {
    let m = muscle_model();
    let (kp, kd) = muscle_gains();
    let pose = pose_a();
    let (_w, tau_g) = gravity_comp(&pose);
    let run = || {
        let mut ws = Workspace::new(&m);
        let mut st = State::neutral(&m);
        st.q.copy_from_slice(&pose);
        let mut cs = ConstraintSet::new(&m, true, true);
        let mut tau = [0.0; NQ];
        let mut lam_bits: Vec<u64> = Vec::new();
        for _ in 0..480 {
            for j in 0..NQ {
                tau[j] = tau_g[j] + kp[j] * (pose[j] - st.q[j]);
            }
            tau[4] += 5.0;
            tau[11] -= 5.0;
            substep_constrained_damped(&m, &mut st, &tau, &kd, H, &mut ws, &mut cs);
            for r in 0..cs.nrows {
                lam_bits.push(cs.lambda[r].to_bits());
            }
        }
        (st, lam_bits)
    };
    let (a, la) = run();
    let (b, lb) = run();
    assert_eq!(a.p.map(f64::to_bits), b.p.map(f64::to_bits));
    assert_eq!(a.quat.map(f64::to_bits), b.quat.map(f64::to_bits));
    for j in 0..NQ {
        assert_eq!(a.q[j].to_bits(), b.q[j].to_bits(), "q[{j}]");
    }
    for i in 0..a.v.len() {
        assert_eq!(a.v[i].to_bits(), b.v[i].to_bits(), "v[{i}]");
    }
    assert_eq!(la, lb, "impulse stream differs");
}

/// Gate (R10): the RANK-1 LIMIT-ROW claim the three closed forms rest
/// on, plus a dense-vs-closed bit-equality soak — and a direct pin on
/// the `0.0 +` zero-sign guard.
///
/// Three things are asserted:
///  1. STRUCTURE — a limit row is `fill(0.0)` + one `jr[voff+j] = sign`
///     store, so every other column holds the memset's +0.0 EXACTLY
///     (sign bit clear, not merely `== 0.0`) and the pivot holds ±1.0.
///  2. ARITHMETIC — `0.0 + sign·x[voff+j]` reproduces the dense 35-wide
///     dot bit for bit on the live rows of every solve, over probe
///     vectors including the state, the row's own (projected) `M⁻¹Jᵀ`
///     column, ±1 unit columns, and vectors holding ±0.0 at the pivot.
///  3. THE GUARD IS LOAD-BEARING — for the (common) `x[pivot] = +0.0`,
///     `sign = −1` case the UNGUARDED product is −0.0 while the dense
///     sum is +0.0. The test counts those cases and requires the corpus
///     to actually hit them, so dropping the leading `0.0 +` in the
///     solver fails here rather than silently 500 substeps downstream.
#[test]
fn r10_limit_rows_are_rank1_and_closed_form_is_bitwise() {
    let m = &HUMANOID;
    let nv = m.nv();
    let voff = m.voff();
    let mut rng = Rng(0x0000_000A_5EED_0100);
    let mut solves = 0usize;
    let mut probes = 0usize;
    let mut limrows = 0usize;
    let mut guard_hits = 0usize;
    for _state_i in 0..24 {
        let mut st = super::testutil::random_state(m, &mut rng, true);
        let mut ws = Workspace::new(m);
        let mut cs = ConstraintSet::new(m, true, true);
        let mut tau = [0.0f64; NQ];
        for _k in 0..40 {
            if _k % 8 == 0 {
                for j in 0..NQ {
                    let s = match rng.next_u64() % 3 {
                        0 => 0.0,
                        1 => 1.0,
                        _ => -1.0,
                    };
                    tau[j] = s * m.tau_max[j];
                }
            }
            substep_constrained(m, &mut st, &tau, H, &mut ws, &mut cs);
            if !st.v.iter().all(|x| x.is_finite()) {
                break; // the closed forms are a finite-trajectory claim
            }
            solves += 1;
            let ndense = cs.ground_rows() + cs.contacts.len();
            let limset: Vec<(usize, f64)> = cs.limit_set().to_vec();
            assert_eq!(limset.len(), cs.limit_rows(), "limit row bookkeeping");
            // (1) structure: rank-1, +0.0 everywhere but the pivot
            for (li, &(jt, sign)) in limset.iter().enumerate() {
                let jr = cs.row_j(ndense + li);
                assert!(sign == 1.0 || sign == -1.0, "limit sign {sign}");
                for i in 0..nv {
                    let want = if i == voff + jt { sign } else { 0.0 };
                    assert_eq!(
                        jr[i].to_bits(),
                        want.to_bits(),
                        "limit row {li} (joint {jt}) column {i} is {} — not \
                         the rank-1 ±1.0-in-a-field-of-+0.0 shape",
                        jr[i]
                    );
                }
                limrows += 1;
            }
            // (2)+(3) arithmetic + guard, over adversarial probe vectors
            let mut probe_vecs: Vec<Vec<f64>> = vec![st.v.to_vec()];
            for (li, _) in limset.iter().enumerate() {
                probe_vecs.push(cs.row_minvjt(ndense + li).to_vec());
            }
            probe_vecs.push(vec![0.0f64; nv]); // +0.0 at every pivot
            probe_vecs.push(vec![-0.0f64; nv]); // −0.0 at every pivot
            for c in [0usize, 6, 17, 23, 34] {
                let mut e = vec![0.0f64; nv];
                e[c] = if c % 2 == 0 { 1.0 } else { -1.0 };
                probe_vecs.push(e);
            }
            let mut dense_rand = vec![0.0f64; nv];
            for x in dense_rand.iter_mut() {
                *x = rng.range(-1e3, 1e3);
            }
            probe_vecs.push(dense_rand);
            for pv in probe_vecs.iter() {
                for (li, (dense, closed)) in
                    cs.limit_dot_ref_vs_closed(voff, pv).into_iter().enumerate()
                {
                    assert_eq!(
                        dense.to_bits(),
                        closed.to_bits(),
                        "limit row {li}: closed {closed} != dense {dense}"
                    );
                    let (jt, sign) = limset[li];
                    let bare = sign * pv[voff + jt];
                    if bare.to_bits() != dense.to_bits() {
                        // the ONLY way this can happen is the −0.0 the
                        // leading `0.0 +` exists to absorb
                        assert_eq!(bare.to_bits(), (-0.0f64).to_bits());
                        assert_eq!(dense.to_bits(), 0.0f64.to_bits());
                        guard_hits += 1;
                    }
                    probes += 1;
                }
            }
        }
    }
    println!(
        "R10 rank-1 gate: {solves} solves, {limrows} limit rows, {probes} \
         dense-vs-closed dots, all bit-identical; {guard_hits} of them \
         needed the `0.0 +` zero-sign guard"
    );
    assert!(solves > 500, "corpus too small ({solves} solves)");
    assert!(limrows > 500, "no limit rows exercised ({limrows})");
    assert!(
        guard_hits > 0,
        "the corpus never hit the −0.0 case the guard exists for — the \
         guard is then unpinned, which is worse than a failure"
    );
}

/// Gate (R13): the solve's staging scratch — every `minvjt` row and
/// the `wt` RHS-minor buffer — is WRITE-BEFORE-READ within a solve.
///
/// R13 (2026-08-07) deleted a `j → minvjt` seed copy and a separate
/// `wt` copy pass as dead work on the (since removed) weld block; the
/// batched ground / pair / limit solves that remain rely on the same
/// property. This runs the same random-gear trajectory twice — once
/// clean, once with both regions poisoned to NaN before EVERY substep —
/// and demands bit-identical state and impulse streams. A surviving NaN
/// would contaminate the impulses immediately and the whole state
/// within a substep, so the gate is sharp.
#[test]
fn r13_solve_scratch_is_write_before_read() {
    let m = &HUMANOID;
    let mut rng = Rng(0x0000_000D_5EED_0130);
    // one fixed torque schedule, replayed by both runs
    let mut sched: Vec<[f64; NQ]> = Vec::new();
    for _ in 0..(24 * 5) {
        let mut tau = [0.0f64; NQ];
        for j in 0..NQ {
            let s = match rng.next_u64() % 3 {
                0 => 0.0,
                1 => 1.0,
                _ => -1.0,
            };
            tau[j] = s * m.tau_max[j];
        }
        sched.push(tau);
    }
    let states: Vec<State> = (0..24)
        .map(|_| super::testutil::random_state(m, &mut rng, true))
        .collect();

    let run = |poison: bool| -> Vec<u64> {
        let mut bits: Vec<u64> = Vec::new();
        for (state_i, st0) in states.iter().enumerate() {
            let mut st = st0.clone();
            let mut ws = Workspace::new(m);
            let mut cs = ConstraintSet::new(m, true, true);
            for k in 0..40 {
                let tau = &sched[state_i * 5 + k / 8];
                if poison {
                    cs.poison_solve_scratch();
                }
                substep_constrained(m, &mut st, tau, H, &mut ws, &mut cs);
                if !st.v.iter().all(|x| x.is_finite()) {
                    break; // finite-trajectory claim, same cut in both runs
                }
                for x in st.q.iter().chain(st.v.iter()).chain(st.p.iter()) {
                    bits.push(x.to_bits());
                }
                for r in 0..cs.nrows {
                    bits.push(cs.lambda[r].to_bits());
                }
            }
        }
        bits
    };
    let clean = run(false);
    let poisoned = run(true);
    assert!(clean.len() > 50_000, "corpus too small ({})", clean.len());
    assert_eq!(
        clean, poisoned,
        "poisoning minvjt / wt changed the trajectory — a solve reads \
         a scratch word it did not write first"
    );
    println!(
        "R13 write-before-read gate: {} recorded words, identical with the \
         solve scratch poisoned to NaN before every substep",
        clean.len()
    );
}

// ── R23 (2026-08-07): DEAD-STORE SWEEP GATES ─────────────────────────
//
// Five store sites were verified dead. THREE of them landed (a: the
// dense limit rows of `j`; b: the free-base half of the mass matrix's
// upper triangle; e: `rnea`'s `ws.vo` chain) and two were reverted on
// the measurement, NOT on correctness — R23d (`ws.sol`) and R23f
// (`ConstraintSet::rw`) are both bitwise-clean and both cost ~1–3%; see
// the notes at their call sites. Their poison gates are KEPT here
// anyway, because what they establish (the region really is
// write-before-read / unread) is what makes those reverts pure
// performance findings rather than open correctness questions.
//
// The gates are built exactly the way R13's
// poison gate verified its two — fill the region the shipping
// build stops writing with NaN in a TEST build (where the stores are
// still compiled in, so the poison genuinely replaces live data) and
// require the trajectory back BIT FOR BIT. Any surviving reader turns
// the run into NaN on the first substep, so the gate is sharp rather
// than statistical. The fifth (c, a dead column block of the weld
// Jacobian) left with the weld harness on 2026-09-27.
//
// The recorded corpus is q/v/p plus every row impulse, plus `energy`
// and `momentum` at each tick — the two functions that still own the
// `ws.vo` chain R23e stopped propagating in `rnea`.

/// Which newly-dead region one `r23_run` pass poisons.
#[derive(Clone, Copy, PartialEq)]
enum R23Poison {
    /// reference pass — nothing poisoned
    None,
    /// (a) the dense LIMIT rows of `ConstraintSet::j`
    LimitJ,
    /// (b) the strict UPPER triangle of the CRBA mass matrix
    MUpper,
    /// (d) step 11's `solcache` — the write-before-read property the
    ///     REVERTED R23d rested on, kept as a standing invariant
    Solcache,
    /// (e) `ws.vo` — `rnea` no longer propagates the frame-origin
    ///     velocity chain, so it holds whatever was there
    Vo,
    /// all four at once
    All,
}

impl R23Poison {
    fn is(self, p: R23Poison) -> bool {
        self == p || self == R23Poison::All
    }
}

/// Replay a fixed set of random states under a fixed torque schedule,
/// recording every observable f64 as raw bits.
fn r23_run(p: R23Poison) -> Vec<u64> {
    let m = &HUMANOID;
    let mut rng = Rng(0x0000_0023_5EED_2301);
    let mut sched: Vec<[f64; NQ]> = Vec::new();
    for _ in 0..(24 * 5) {
        let mut tau = [0.0f64; NQ];
        for j in 0..NQ {
            let s = match rng.next_u64() % 3 {
                0 => 0.0,
                1 => 1.0,
                _ => -1.0,
            };
            tau[j] = s * m.tau_max[j];
        }
        sched.push(tau);
    }
    let states: Vec<State> = (0..24)
        .map(|_| super::testutil::random_state(m, &mut rng, true))
        .collect();

    let mut bits: Vec<u64> = Vec::new();
    for (state_i, st0) in states.iter().enumerate() {
        let mut st = st0.clone();
        let mut ws = Workspace::new(m);
        let mut cs = ConstraintSet::new(m, true, true);
        ws.poison_m_upper = p.is(R23Poison::MUpper);
        cs.poison_limit_j = p.is(R23Poison::LimitJ);
        for k in 0..40 {
            let tau = &sched[state_i * 5 + k / 8];
            if p.is(R23Poison::Solcache) {
                ws.solcache.fill(f64::NAN);
            }
            if p.is(R23Poison::Vo) {
                ws.vo.fill([f64::NAN; 3]);
            }
            substep_constrained(m, &mut st, tau, H, &mut ws, &mut cs);
            if !st.v.iter().all(|x| x.is_finite()) {
                break; // finite-trajectory claim, same cut in both runs
            }
            for x in st.q.iter().chain(st.v.iter()).chain(st.p.iter()) {
                bits.push(x.to_bits());
            }
            for r in 0..cs.nrows {
                bits.push(cs.lambda[r].to_bits());
            }
            // the two surviving owners of the `ws.vo` chain — they must
            // rebuild it themselves (R23e), so they are recorded from a
            // possibly-poisoned workspace on purpose
            if p.is(R23Poison::Vo) {
                ws.vo.fill([f64::NAN; 3]);
            }
            bits.push(energy(m, &mut ws, &st.v).to_bits());
            let (pl, am) = super::algo::momentum(m, &mut ws, &st.v);
            for x in pl.iter().chain(am.iter()) {
                bits.push(x.to_bits());
            }
        }
    }
    bits
}

#[test]
fn r23_dead_regions_have_no_reader() {
    let clean = r23_run(R23Poison::None);
    assert!(clean.len() > 50_000, "corpus too small ({})", clean.len());
    for (p, what) in [
        (R23Poison::LimitJ, "R23a: the dense LIMIT rows of `j`"),
        (R23Poison::MUpper, "R23b: the mass matrix's strict UPPER triangle"),
        (R23Poison::Solcache, "R23d (reverted): step 11's `solcache`"),
        (R23Poison::Vo, "R23e: `ws.vo`"),
        (R23Poison::All, "R23a+b+d+e together"),
    ] {
        assert_eq!(
            clean,
            r23_run(p),
            "poisoning {what} changed the trajectory — that region is \
             still READ, so the corresponding R23 store deletion is \
             NOT dead and must be reverted"
        );
    }
    println!(
        "R23 dead-region gate: {} recorded words, identical under all \
         five NaN poisonings",
        clean.len()
    );
}
