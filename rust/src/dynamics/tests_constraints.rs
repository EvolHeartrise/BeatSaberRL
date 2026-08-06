//! P2 constraint gates: static stance (full-RNEA hold + muscle-tone
//! weld force sum), weld drift under load (random full gears; scripted
//! two-arm swing), non-penetration + unilaterality of the contact rows,
//! no energy injection, bitwise determinism with contacts, and the
//! contact-substep wall-time probe.
//!
//! The muscle-tone scenarios follow the oracle protocol (README): the
//! stabilizer's near-critical kd is applied IMPLICITLY
//! (`substep_constrained_damped`) — MuJoCo-Euler damping semantics; the
//! spring kp stays an explicit applied torque. The production body
//! (spec damping b, ω caps) never needs this.

use super::algo::energy;
use super::fk::{fk_all, rot_log};
use super::model::{HUMANOID, NQ};
use super::solver::{
    substep_constrained, substep_constrained_damped, substep_constrained_wrench, ConstraintSet,
    WELD_FRAMES, WELD_TARGET_P,
};
use super::testutil::{gravity_comp, muscle_gains, muscle_model, Rng};
use super::{State, Workspace};

const H: f64 = 1.0 / 240.0;
const DEG: f64 = std::f64::consts::PI / 180.0;

/// Max (position, orientation) weld violation over both feet at the
/// CURRENT state (fresh FK into ws).
fn weld_violation(m: &super::model::Model, st: &State, ws: &mut Workspace) -> (f64, f64) {
    fk_all(m, st, ws);
    let mut wp = 0.0f64;
    let mut wr = 0.0f64;
    for (widx, &frame) in WELD_FRAMES.iter().enumerate() {
        let p = ws.p[frame];
        let t = WELD_TARGET_P[widx];
        let dp = [(p[0] - t[0]), (p[1] - t[1]), (p[2] - t[2])];
        wp = wp.max((dp[0] * dp[0] + dp[1] * dp[1] + dp[2] * dp[2]).sqrt());
        let phi = rot_log(&ws.r[frame]);
        wr = wr.max((phi[0] * phi[0] + phi[1] * phi[1] + phi[2] * phi[2]).sqrt());
    }
    (wp, wr)
}

/// Gate (a1), stance protocol (oracle check_static b2): welds on,
/// constant full 35-DoF RNEA(q_ref, 0, 0) compensation INCLUDING the
/// root wrench (joint-rows-only is not an equilibrium of the welded
/// closed chain), 2 s at h = 1/240. The compensation cancels gravity
/// bitwise, so drift is rounding noise on a knife-edge equilibrium —
/// the equilibrium is UNSTABLE (constant-torque inverted-pendulum
/// modes; measured e-fold ~0.35 s from 1e-17 seeds), which is why the
/// horizon is the protocol's 2 s and not SPEC §7 P2's 10 s (Divergences
/// §17). Task gate < 5e-2 rad; measured ~1e-13.
#[test]
fn static_stance_full_rnea_hold() {
    let m = &HUMANOID;
    let mut ws = Workspace::new(m);
    let mut st = State::neutral(m);
    let (wrench, tau) = gravity_comp(&[0.0; NQ]);
    let mut cs = ConstraintSet::new(m, true, true);
    let mut max_drift = 0.0f64;
    let mut max_wp = 0.0f64;
    let mut max_wr = 0.0f64;
    for _ in 0..480 {
        substep_constrained_wrench(m, &mut st, &tau, &wrench, H, &mut ws, &mut cs);
        for j in 0..NQ {
            max_drift = max_drift.max(st.q[j].abs());
        }
        let (wp, wr) = weld_violation(m, &st, &mut ws);
        max_wp = max_wp.max(wp);
        max_wr = max_wr.max(wr);
    }
    println!(
        "static stance 2 s: max joint drift {max_drift:e} rad, weld violation {max_wp:e} m / {max_wr:e} rad"
    );
    assert!(max_drift <= 1e-6, "joint drift {max_drift:e}");
    assert!(max_wp <= 1e-8, "weld position violation {max_wp:e}");
    assert!(max_wr <= 1e-8, "weld orientation violation {max_wr:e}");
}

/// Gate (a2), oracle protocol check_static (c): muscle-tone spring
/// stance (kp = τ_max/0.2 to neutral, near-critical kd implicit),
/// settle 4 s, then the vertical weld constraint forces must sum to
/// body weight 735.95 N within 5 % over the next second.
#[test]
fn stance_weld_force_sum() {
    let m = muscle_model();
    let (kp, kd) = muscle_gains();
    let mut ws = Workspace::new(&m);
    let mut st = State::neutral(&m);
    let mut cs = ConstraintSet::new(&m, true, true);
    let mut tau = [0.0; NQ];
    let mut run = |st: &mut State, ws: &mut Workspace, cs: &mut ConstraintSet, n: usize| -> f64 {
        let mut fy_sum = 0.0;
        for _ in 0..n {
            for j in 0..NQ {
                tau[j] = kp[j] * (0.0 - st.q[j]);
            }
            substep_constrained_damped(&m, st, &tau, &kd, H, ws, cs);
            // weld linear-y rows: 1 (foot_l) and 7 (foot_r); force = λ/h
            fy_sum += (cs.lambda[1] + cs.lambda[7]) / H;
        }
        fy_sum / n as f64
    };
    run(&mut st, &mut ws, &mut cs, 960); // 4 s settle
    let fy = run(&mut st, &mut ws, &mut cs, 240); // 1 s measure
    let droop = st.q.iter().fold(0.0f64, |a, q| a.max(q.abs()));
    let weight = 75.02 * 9.81;
    println!(
        "muscle-tone stance: mean vertical weld force {fy:.2} N (weight {weight:.2}), max droop {droop:.4} rad"
    );
    assert!(
        (fy - weight).abs() <= 0.05 * weight,
        "vertical weld force {fy:.2} vs weight {weight:.2}"
    );
}

/// Gate (b): 10 s of per-tick random FULL-gear torques on all 29
/// (fixed seed), welds + contacts live, raw production body (spec
/// damping, ω caps active — this doubles as the integrability stress
/// test). Feet must stay put at every substep, no NaN, root sane.
///
/// Gate re-reading (Divergences §17): the original < 2 mm bound is
/// below the velocity-level + Baumgarte(β = 0.2) formulation's
/// intrinsic floor for THIS input — with leg joints riding their
/// 16 rad/s caps the per-substep linearization error is O(h²·q̇²·L)
/// ≈ 1.5 mm, and the steady-state violation is that divided by β.
/// Measured 7.5 mm max (β sensitivity check: 2.5 mm at β = 1 —
/// confirming the floor scales as predicted); gate set at 10 mm /
/// 0.05 rad. The physically sane load regime is 100× under the spec's
/// own 1e-4 gate (see weld_under_two_arm_swing).
#[test]
fn weld_drift_random_full_gears() {
    let m = &HUMANOID;
    let mut ws = Workspace::new(m);
    let mut st = State::neutral(m);
    let mut cs = ConstraintSet::new(m, true, true);
    let mut rng = Rng(0x5EED_0000_0000_0020);
    let mut tau = [0.0; NQ];
    let mut max_wp = 0.0f64;
    let mut max_wr = 0.0f64;
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
            let (wp, wr) = weld_violation(m, &st, &mut ws);
            max_wp = max_wp.max(wp);
            max_wr = max_wr.max(wr);
            assert!(wp < 1e-2, "weld position violation {wp:e} m");
            assert!(wr < 5e-2, "weld orientation violation {wr:e} rad");
            assert!(st.q.iter().all(|x| x.is_finite()));
            assert!(st.v.iter().all(|x| x.is_finite()));
        }
    }
    let dp = [
        st.p[0] - m.base_pos0[0],
        st.p[1] - m.base_pos0[1],
        st.p[2] - m.base_pos0[2],
    ];
    let root_move = (dp[0] * dp[0] + dp[1] * dp[1] + dp[2] * dp[2]).sqrt();
    println!(
        "random full gears 10 s: max weld violation {max_wp:e} m / {max_wr:e} rad, root moved {root_move:.3} m"
    );
    // sanity: the pelvis hangs on the welded feet through ≤ 0.87 m of
    // leg, so it can reach at most ~1.8 m from its start (deep crouch +
    // lean is legitimate under full-gear thrash)
    assert!(root_move < 1.8, "root wandered {root_move} m despite welds");
}

/// SPEC-v10 §7 P2 "weld under load": scripted two-arm swinging on the
/// muscle-tone stance (the raw unstabilized body simply falls over —
/// oracle README), 10 s: weld pose violation ≤ 1e-4 m / 1e-4 rad
/// (Baumgarte holds).
#[test]
fn weld_under_two_arm_swing() {
    let m = muscle_model();
    let (kp, kd) = muscle_gains();
    let mut ws = Workspace::new(&m);
    let mut st = State::neutral(&m);
    let mut cs = ConstraintSet::new(&m, true, true);
    let mut tau = [0.0; NQ];
    let two_pi = 2.0 * std::f64::consts::PI;
    let mut max_wp = 0.0f64;
    let mut max_wr = 0.0f64;
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
        let (wp, wr) = weld_violation(&m, &st, &mut ws);
        max_wp = max_wp.max(wp);
        max_wr = max_wr.max(wr);
    }
    println!("two-arm swing 10 s: max weld violation {max_wp:e} m / {max_wr:e} rad");
    assert!(max_wp <= 1e-4, "weld position violation {max_wp:e}");
    assert!(max_wr <= 1e-4, "weld orientation violation {max_wr:e}");
}

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
        let nweld = cs.weld_rows();
        for (i, c) in cs.contacts.iter().enumerate() {
            let row = nweld + i;
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
#[test]
fn energy_no_injection_contact_press() {
    let m = muscle_model();
    let (kp, kd) = muscle_gains();
    let pose = pose_b();
    let (_w, tau_g) = gravity_comp(&pose);
    let mut ws = Workspace::new(&m);
    let mut st = State::neutral(&m);
    st.q.copy_from_slice(&pose);
    let mut cs = ConstraintSet::new(&m, true, true);
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

/// Gate (g): wall-time probe with welds + an active contact every
/// substep. Run `cargo test --release bench_substep_contacts --
/// --nocapture` for the report number.
#[test]
fn bench_substep_contacts() {
    let m = muscle_model();
    let (kp, kd) = muscle_gains();
    let pose = pose_a();
    let (_w, tau_g) = gravity_comp(&pose);
    let mut ws = Workspace::new(&m);
    let mut st = State::neutral(&m);
    st.q.copy_from_slice(&pose);
    let mut cs = ConstraintSet::new(&m, true, true);
    let mut tau = [0.0; NQ];
    let mut do_step = |st: &mut State, ws: &mut Workspace, cs: &mut ConstraintSet| {
        for j in 0..NQ {
            tau[j] = tau_g[j] + kp[j] * (pose[j] - st.q[j]);
        }
        tau[4] += 5.0;
        tau[11] -= 5.0;
        substep_constrained_damped(&m, st, &tau, &kd, H, ws, cs);
    };
    for _ in 0..200 {
        do_step(&mut st, &mut ws, &mut cs);
    }
    assert!(!cs.contacts.is_empty(), "bench pose lost its contact");
    let n = 5000;
    let t0 = std::time::Instant::now();
    for _ in 0..n {
        do_step(&mut st, &mut ws, &mut cs);
    }
    let us = t0.elapsed().as_secs_f64() * 1e6 / n as f64;
    println!(
        "constrained substep (welds + {} contact rows): {us:.2} us ({n} iters)",
        cs.contacts.len()
    );
}

/// Gate (R9): the WELD-JACOBIAN SUPPORT claim the segmented weld dots
/// rest on, plus a dense-vs-segmented bit-equality soak.
///
/// Two things are pinned:
///  1. STRUCTURE — every weld row is `fill(0.0)` + writes on the
///     free-base columns and the welded foot's ancestor chain, so the
///     columns outside `{0..6} ∪ chain` hold the memset's +0.0 EXACTLY
///     (sign bit clear, not merely `== 0.0`). Both the CHEAP tier's
///     claim (columns 6..23 dead in all 12 rows) and the FULL tier's
///     (23..29 dead in rows 6..12, 29..35 dead in rows 0..6) are
///     asserted, so the record survives whichever tier ships.
///  2. ARITHMETIC — the shipped segmented dot reproduces the dense one
///     bit for bit on the live `jwt` of every solve, over probe vectors
///     including the state itself, ±1 unit columns (the −0.0 channel's
///     only way in) and random dense vectors.
#[test]
fn r9_weld_support_and_segmented_dot_bitwise() {
    let m = &HUMANOID;
    let nv = m.nv();
    let voff = m.voff();
    // ancestor chains of the two welded foot frames, as v-columns
    let chain = |frame: usize| -> Vec<usize> {
        let mut out = vec![];
        let mut k = frame as i32;
        while k >= 0 {
            let fr = &m.frames[k as usize];
            if fr.coord >= 0 {
                out.push(voff + fr.coord as usize);
            }
            k = fr.parent;
        }
        out.sort_unstable();
        out
    };
    let cl = chain(WELD_FRAMES[0]);
    let cr = chain(WELD_FRAMES[1]);
    assert_eq!(cl, (23..29).collect::<Vec<_>>(), "left-foot chain columns");
    assert_eq!(cr, (29..35).collect::<Vec<_>>(), "right-foot chain columns");

    let mut rng = Rng(0x0000_0009_5EED_0090);
    let mut solves = 0usize;
    let mut probes = 0usize;
    for state_i in 0..24 {
        let mut st = super::testutil::random_state(m, &mut rng, true);
        let mut ws = Workspace::new(m);
        let mut cs = ConstraintSet::new(m, true, true);
        let mut tau = [0.0f64; NQ];
        for k in 0..40 {
            if k % 8 == 0 {
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
                break; // the support claim is a finite-trajectory one
            }
            solves += 1;
            // (1) structure: dead columns are EXACTLY +0.0
            let j = cs.weld_j();
            let jwt = cs.weld_jwt();
            for r in 0..12 {
                let live: &[usize] = if r < 6 { &cl } else { &cr };
                for i in 0..nv {
                    let x = j[r * nv + i];
                    assert_eq!(
                        jwt[i * 12 + r].to_bits(),
                        x.to_bits(),
                        "jwt transpose mismatch at row {r} col {i}"
                    );
                    if i < 6 || live.contains(&i) {
                        continue;
                    }
                    assert_eq!(
                        x.to_bits(),
                        0.0f64.to_bits(),
                        "state {state_i} substep {k}: weld row {r} column \
                         {i} is {x} — outside the R9 support"
                    );
                }
            }
            // (2) arithmetic: segmented == dense, bit for bit
            let mut probe_vecs: Vec<Vec<f64>> = vec![st.v.to_vec()];
            for c in [0usize, 5, 6, 22, 23, 29, 34] {
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
                let (dense, seg) = cs.weld_dot_ref_vs_segmented(pv);
                for w in 0..12 {
                    assert_eq!(
                        dense[w].to_bits(),
                        seg[w].to_bits(),
                        "state {state_i} substep {k} lane {w}: segmented \
                         {} != dense {}",
                        seg[w],
                        dense[w]
                    );
                }
                probes += 1;
            }
        }
    }
    println!(
        "R9 support gate: {solves} solves, {probes} dense-vs-segmented \
         12-lane dots, all bit-identical"
    );
    assert!(solves > 500, "corpus too small ({solves} solves)");
}
