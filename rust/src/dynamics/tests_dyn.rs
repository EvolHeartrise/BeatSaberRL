//! P1 dynamics gates: mass-matrix identities (symmetry, PD, CRBA vs
//! RNEA columns), RNEA gravity sanity against hand-computed torques,
//! momentum conservation of the free-floating unconstrained body,
//! bitwise determinism, clamp behavior, and the substep wall-time
//! probe.

use super::algo::{
    chol_factor, chol_factor_floored, chol_solve, crba, momentum, rnea, PD_FLOOR_F64,
};
use super::fk::fk_all;
use super::model::{Model, HUMANOID, NQ, NV};
use super::step::substep;
use super::testutil::{random_state, Rng, ZERO_DAMP29};
use super::{State, Workspace};

#[test]
fn mass_matrix_identities() {
    let m = &HUMANOID;
    let nv = m.nv();
    let mut ws = Workspace::new(m);
    let mut rng = Rng(0x5EED_0000_0000_0002);
    let mut worst_col = 0.0f64;
    for pose in 0..200 {
        let st = random_state(m, &mut rng, false);
        fk_all(m, &st, &mut ws);
        crba(m, &mut ws);
        // exact symmetry: both triangles are written from one value
        for i in 0..nv {
            for j in 0..i {
                assert_eq!(
                    ws.m[i * nv + j].to_bits(),
                    ws.m[j * nv + i].to_bits(),
                    "M not bitwise symmetric at ({i},{j}), pose {pose}"
                );
            }
        }
        // positive definite: Cholesky (no pivoting) succeeds
        let mm = ws.m.clone();
        assert!(
            chol_factor(nv, &mm, &mut ws.l),
            "Cholesky failed at pose {pose}"
        );
        // CRBA column j == RNEA(q, 0, e_j) − RNEA(q, 0, 0): the
        // standard cross-check (run on every 8th pose — 36 RNEA calls
        // per pose)
        if pose % 8 == 0 {
            let v0 = vec![0.0; nv];
            rnea(m, &mut ws, &v0, None);
            let c0 = ws.c.clone();
            let mut ej = vec![0.0; nv];
            for j in 0..nv {
                ej[j] = 1.0;
                rnea(m, &mut ws, &v0, Some(&ej));
                for i in 0..nv {
                    let d = (ws.c[i] - c0[i] - mm[i * nv + j]).abs();
                    worst_col = worst_col.max(d);
                    assert!(d <= 1e-9, "CRBA/RNEA mismatch ({i},{j}) pose {pose}: {d:e}");
                }
                ej[j] = 0.0;
            }
        }
    }
    println!("worst CRBA-vs-RNEA column error: {worst_col:e}");
}

/// RNEA(q, 0, 0) at the neutral pose == hand-computed gravity holding
/// torques. Left shoulder pitch, arm horizontal along +z at neutral:
///   lever arms from the shoulder (z, m): uarm COM 0.175, forearm COM
///   0.35 + 0.15 = 0.50, hand COM 0.35 + 0.30 + 0.06 = 0.71
///   τ = g·Σ mᵢ·zᵢ = 9.81·(2.0·0.175 + 1.2·0.50 + 0.61·0.71)
///     = 9.81·(0.35 + 0.6 + 0.4331) = 9.81·1.3831 = 13.5682110 N·m
/// (+pitch raises the arm — axis (−1,0,0) maps +z toward +y — so the
/// holding torque is positive). Elbow, same reasoning from the elbow
/// point: τ = 9.81·(1.2·0.15 + 0.61·0.36) = 9.81·0.3996 = 3.9200760.
#[test]
fn rnea_gravity_holding_torques() {
    let m = &HUMANOID;
    let mut ws = Workspace::new(m);
    let st = State::neutral(m);
    fk_all(m, &st, &mut ws);
    let v0 = vec![0.0; m.nv()];
    rnea(m, &mut ws, &v0, None);
    let tau_pitch = 9.81 * (2.0 * 0.175 + 1.2 * 0.50 + 0.61 * 0.71);
    let tau_elbow = 9.81 * (1.2 * 0.15 + 0.61 * 0.36);
    assert!(
        (ws.c[6 + 3] - tau_pitch).abs() <= 1e-9,
        "l_arm_pitch: got {} want {tau_pitch}",
        ws.c[6 + 3]
    );
    assert!(
        (ws.c[6 + 6] - tau_elbow).abs() <= 1e-9,
        "l_elbow: got {} want {tau_elbow}",
        ws.c[6 + 6]
    );
    // mirror symmetry at the (symmetric) neutral pose
    assert!((ws.c[6 + 10] - ws.c[6 + 3]).abs() <= 1e-9, "r_arm_pitch mirror");
    assert!((ws.c[6 + 13] - ws.c[6 + 6]).abs() <= 1e-9, "r_elbow mirror");
    // root force rows: required external force = +mg upward (RNEA's
    // F = m(a_c − g) convention — the force that HOLDS the body)
    let w = 75.02 * 9.81;
    assert!((ws.c[0]).abs() <= 1e-9 && (ws.c[2]).abs() <= 1e-9);
    assert!((ws.c[1] - w).abs() <= 1e-9, "root y force {} vs {w}", ws.c[1]);
}

// Unconstrained-conservation variant: gravity off, damping off, clamps
// disabled (limits/caps infinite — a tumbling body would otherwise hit
// them and the clamps are deliberately dissipative).
static FREE_LIMITS: [(f64, f64); NQ] = [(f64::NEG_INFINITY, f64::INFINITY); NQ];
static INF_CAP: [f64; NQ] = [f64::INFINITY; NQ];
static ZERO_DAMP: [f64; NQ] = [0.0; NQ];

fn free_model() -> Model {
    let mut m = HUMANOID;
    m.gravity = [0.0, 0.0, 0.0];
    m.limits = &FREE_LIMITS;
    m.omega_cap = &INF_CAP;
    m.damping = &ZERO_DAMP;
    // armature KEPT: its ½aq̇² term is invariant under global rotation/
    // translation, so the conserved Noether momentum is still the BODY
    // momentum — and without it the limit-free wander goes near the
    // shoulder gimbal singularity and the Cholesky dies.
    m
}

/// Gravity off, no torques, free base: total linear + angular momentum
/// of the tumbling body. Semi-implicit Euler is symplectic, not
/// momentum-exact — per-step error is O(h²), so drift over a fixed
/// horizon is O(h) (SPEC-v10 §7 frames the angular gate exactly that
/// way). Gate: measured drift bound at h = 1/240 over 2 s, plus the
/// O(h) halving check.
#[test]
fn momentum_conservation() {
    let m = free_model();
    let mut rng = Rng(0x5EED_0000_0000_0003);
    let mut st0 = State::neutral(&m);
    for i in 0..3 {
        st0.v[i] = rng.range(-0.5, 0.5); // root linear
        st0.v[3 + i] = rng.range(-1.5, 1.5); // root angular (tumble)
    }
    for j in 0..NQ {
        st0.v[6 + j] = rng.range(-1.0, 1.0);
    }
    let tau = vec![0.0; NQ];

    let mut drift = [0.0f64; 2]; // rel drift at h, h/2
    for (pass, h) in [1.0f64 / 240.0, 1.0 / 480.0].into_iter().enumerate() {
        let n = (2.0 / h).round() as usize;
        let mut st = st0.clone();
        let mut ws = Workspace::new(&m);
        fk_all(&m, &st, &mut ws);
        let (p0, l0) = momentum(&m, &mut ws, &st.v);
        let scale = |v: [f64; 3]| (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
        let (pn, ln) = (scale(p0), scale(l0));
        let mut worst = 0.0f64;
        for _ in 0..n {
            substep(&m, &mut st, &tau, h, &mut ws, None);
            fk_all(&m, &st, &mut ws);
            let (p, l) = momentum(&m, &mut ws, &st.v);
            let dp = scale([p[0] - p0[0], p[1] - p0[1], p[2] - p0[2]]) / pn;
            let dl = scale([l[0] - l0[0], l[1] - l0[1], l[2] - l0[2]]) / ln;
            worst = worst.max(dp).max(dl);
        }
        drift[pass] = worst;
    }
    println!("momentum rel drift: h=1/240 {:e}, h=1/480 {:e}", drift[0], drift[1]);
    // measured: 1.32e-2 / 6.02e-3 (ratio 2.20). SPEC-v10 §7's 1e-10
    // linear gate is unattainable for symplectic Euler in these
    // coordinates (momentum is not a discrete invariant — Divergences
    // §14); the real gate is the O(h) convergence plus this band.
    assert!(drift[0] <= 2e-2, "momentum drift at h=1/240: {:e}", drift[0]);
    let ratio = drift[0] / drift[1];
    assert!(
        ratio > 1.5,
        "momentum drift not O(h): {:e} -> {:e} (ratio {ratio})",
        drift[0],
        drift[1]
    );
}

/// Same inputs → bitwise-identical trajectories (gravity on, clamps
/// on, a seeded gear-like torque schedule held constant per 8-substep
/// tick — the §3 actuation pattern).
#[test]
fn determinism_bitwise() {
    let m = &HUMANOID;
    let h = 1.0 / 240.0;
    let run = || {
        let mut st = State::neutral(m);
        let mut ws = Workspace::new(m);
        let mut rng = Rng(0x5EED_0000_0000_0004);
        let mut tau = vec![0.0; NQ];
        for _ in 0..30 {
            for j in 0..NQ {
                let g = [0.0, 1.0, -1.0, 0.25, -0.25][rng.next_u64() as usize % 5];
                tau[j] = g * m.tau_max[j];
            }
            for _ in 0..8 {
                substep(m, &mut st, &tau, h, &mut ws, None);
            }
        }
        st
    };
    let a = run();
    let b = run();
    assert_eq!(a.p.map(f64::to_bits), b.p.map(f64::to_bits));
    assert_eq!(a.quat.map(f64::to_bits), b.quat.map(f64::to_bits));
    for j in 0..NQ {
        assert_eq!(a.q[j].to_bits(), b.q[j].to_bits(), "q[{j}]");
    }
    for i in 0..NV {
        assert_eq!(a.v[i].to_bits(), b.v[i].to_bits(), "v[{i}]");
    }
}

/// Post-solve clamps (§3 step 11): angles stay inside limits, rates
/// inside caps, and a limit hit zeroes the joint rate.
#[test]
fn clamps_hold_limits() {
    let m = &HUMANOID;
    let h = 1.0 / 240.0;
    let mut st = State::neutral(m);
    let mut ws = Workspace::new(m);
    let mut tau = vec![0.0; NQ];
    tau[7] = m.tau_max[7]; // l_wrist_roll toward its +π stop
    tau[6] = -m.tau_max[6]; // l_elbow against its 0 stop
    for _ in 0..480 {
        substep(m, &mut st, &tau, h, &mut ws, None);
        for j in 0..NQ {
            let (lo, hi) = m.limits[j];
            // v14: stops are reciprocal-impulse springs, not teleports.
            // On the P1 path (no solver hook, so no limit ROWS — this
            // test) a full-torque press reaches the Baumgarte
            // equilibrium h·v_drive/β + slop ≈ 0.06-0.08 rad; the env
            // path's solver rows absorb the torque first and hold the
            // 2 mrad slop band (the parked-on-stops gate pins that).
            // worst excursion = the ARRIVAL substep (integration runs
            // before step-11, so one substep of uncapped approach rate
            // lands ~h·(cap + h·τ/M) deep before recovery bites) —
            // scales with the joint's rate cap; must stay under the
            // 0.35 teleport backstop
            let s = h * m.omega_cap[j] * 1.4 + 0.02;
            if !(st.q[j] >= lo - s && st.q[j] <= hi + s) {
                println!("EXCURSION q[{j}] = {:.4} vs [{lo:.4}, {hi:.4}] v {:.2} bound {s:.3}",
                         st.q[j], st.v[6 + j]);
            }
            assert!(st.q[j] >= lo - s && st.q[j] <= hi + s, "q[{j}] out of limits");
            assert!(st.v[6 + j].abs() <= m.omega_cap[j] + 1e-9, "v[{j}] over cap");
        }
    }
    let (lo6, _) = m.limits[6];
    // v14 P1-path contract: an undamped full-torque press with no
    // solver limit rows is a BOUNDED LIMIT CYCLE around the stop
    // (slam-in, Baumgarte recovery, re-slam) — not a dead park. The
    // env path (solver rows + damping) parks inside the 2 mrad slop;
    // the parked-on-stops momentum gate pins that regime. Here we pin:
    // cycle stays within the arrival band, rate within the cap.
    let band6 = h * m.omega_cap[6] * 1.4 + 0.02;
    assert!(
        (st.q[6] - lo6).abs() <= band6,
        "elbow cycles within its stop band (q {} vs lo {})", st.q[6], lo6
    );
    assert!(
        st.v[6 + 6].abs() <= m.omega_cap[6] + 1e-9,
        "cycle rate within the cap, got {}", st.v[6 + 6]
    );
}

/// Crude wall-time probe for the orchestrator's throughput projection:
/// run `cargo test --release bench_substep -- --nocapture`. Two
/// regimes: free fall with no active clamps (pure pipeline: FK + CRBA
/// + RNEA + Cholesky + integrate), and the worst case with every
/// joint torque-pinned at a limit (clamp solves every substep).
#[test]
fn bench_substep_walltime() {
    let m = &HUMANOID;
    let h = 1.0 / 240.0;
    let n = 5000;
    let mut ws = Workspace::new(m);

    let mut st = State::neutral(m);
    let tau0 = vec![0.0; NQ];
    for _ in 0..100 {
        substep(m, &mut st, &tau0, h, &mut ws, None);
    }
    let t0 = std::time::Instant::now();
    for _ in 0..n {
        substep(m, &mut st, &tau0, h, &mut ws, None);
    }
    let free = t0.elapsed().as_secs_f64() * 1e6 / n as f64;

    let mut st = State::neutral(m);
    let tau1 = vec![1.0; NQ]; // pins every joint at a stop
    for _ in 0..200 {
        substep(m, &mut st, &tau1, h, &mut ws, None);
    }
    let t0 = std::time::Instant::now();
    for _ in 0..n {
        substep(m, &mut st, &tau1, h, &mut ws, None);
    }
    let pinned = t0.elapsed().as_secs_f64() * 1e6 / n as f64;
    println!("35-DoF substep: {free:.2} us free-fall, {pinned:.2} us all-pinned ({n} iters)");
}

/// Regression: the shoulder stack (yaw→pitch→twist) gimbal-locks at
/// pitch ±90°, and l_arm_pitch's LOWER LIMIT is exactly −90° — at the
/// clamp the yaw and twist world axes are anti-parallel through the
/// same point, so the body-inertia mass matrix is exactly singular
/// there (found at P1: pivot −4.3e-18 after 849 substeps of constant
/// torque). The armature term (model.rs, SPEC-v10 Divergences §12)
/// must keep the no-pivot Cholesky alive at that pose.
#[test]
fn pd_at_shoulder_gimbal_lock() {
    let m = &HUMANOID;
    let nv = m.nv();
    let mut ws = Workspace::new(m);
    let mut st = State::neutral(m);
    st.q[3] = m.limits[3].0; // l_arm_pitch = −π/2 exactly
    st.q[10] = m.limits[10].0; // r_arm_pitch = −π/2 exactly
    st.q[4] = 0.3; // arbitrary yaw/twist — the lock is pitch-only
    st.q[5] = -0.2;
    fk_all(m, &st, &mut ws);
    crba(m, &mut ws);
    let mm = ws.m.clone();
    assert!(
        chol_factor(nv, &mm, &mut ws.l),
        "Cholesky must survive the gimbal-lock pose"
    );
}

/// Regression (SPEC-v10 Divergences §22, found benching
/// v10_gears5_400m ckpt_upd0762): a mid-surge flaily policy reached a
/// contorted state whose f64 mass-matrix Cholesky lost a pivot and the
/// old `assert!(chol_factor(..))` PANICKED the whole VecBeatSim bench.
/// The singular family is the §12 gimbal-adjacent one (shoulder stack
/// at pitch −90°, yaw/twist world axes anti-parallel through one
/// point); the armature keeps the HUMANOID's pivots ≥ ~2.7e-5, so to
/// drive a pivot under the 1e-17 floor deterministically the test
/// removes the armature — reproducing the P1-measured exactly-singular
/// matrix (pivot −4.3e-18). The floored factorization must clamp,
/// count, and stay finite — never panic.
#[test]
fn pivot_floor_clamps_singular_gimbal_matrix() {
    let m = &HUMANOID;
    let nv = m.nv();
    let mut ws = Workspace::new(m);
    let mut st = State::neutral(m);
    st.q[3] = m.limits[3].0; // l_arm_pitch = −π/2 exactly (gimbal lock)
    st.q[10] = m.limits[10].0; // r_arm_pitch = −π/2 exactly
    st.q[4] = 0.3;
    st.q[5] = -0.2;
    fk_all(m, &st, &mut ws);
    crba(m, &mut ws);
    let mut mm = ws.m.clone();
    for j in 0..NQ {
        mm[(6 + j) * nv + (6 + j)] -= m.armature[j]; // exactly singular now
    }
    // sanity: the strict factorization DOES reject this matrix
    assert!(
        !chol_factor(nv, &mm, &mut ws.l),
        "armature-less gimbal matrix should not be PD"
    );
    let hits = chol_factor_floored(nv, &mm, &mut ws.l);
    assert!(hits >= 1, "floor must engage on the singular matrix");
    assert!(ws.l.iter().all(|x| x.is_finite()), "factor must stay finite");
    // a solve through the floored factor stays finite too
    let mut rhs: Vec<f64> = (0..nv).map(|i| 0.1 * (i as f64 + 1.0)).collect();
    chol_solve(nv, &ws.l, &mut rhs);
    assert!(rhs.iter().all(|x| x.is_finite()), "solve must stay finite");
    // and a healthy matrix is untouched BITWISE (0 hits, same factor)
    fk_all(m, &st, &mut ws);
    crba(m, &mut ws);
    let mm = ws.m.clone();
    let mut l_strict = vec![0.0; nv * nv];
    assert!(chol_factor(nv, &mm, &mut l_strict));
    assert_eq!(chol_factor_floored(nv, &mm, &mut ws.l), 0);
    // LOWER TRIANGLE + diagonal only: since 2026-08-03 neither factor
    // zeroes the strict upper triangle (dead stores — chol_solve reads
    // only k ≤ i), so those cells hold whatever the buffer had before.
    for i in 0..nv {
        for k in 0..=i {
            assert_eq!(
                ws.l[i * nv + k].to_bits(),
                l_strict[i * nv + k].to_bits(),
                "floored factor must be bitwise at ({i},{k})"
            );
        }
    }
}

/// §22 NaN semantics (pinned 2026-08-03): a NaN pivot must be FLOORED
/// and COUNTED, not passed through. The original `d < PD_FLOOR_F64`
/// test is false for NaN, so a NaN pivot escaped the floor unclamped
/// and poisoned every subsequent column (sqrt(NaN) = NaN, and the
/// divide by it NaNs the whole trailing factor). `!(d >= FLOOR)` is
/// true for NaN and identical for every non-NaN pivot; this matches
/// the CUDA fmax/tmax mirror (NaN → floor).
#[test]
fn nan_pivot_is_floored_and_counted() {
    // 2×2, diagonal, NaN in the second pivot: the first column is
    // healthy so the NaN cannot be attributed to earlier arithmetic.
    let m = [4.0, 0.0, 0.0, f64::NAN];
    let mut l = [0.0f64; 4];
    let hits = chol_factor_floored(2, &m, &mut l);
    assert_eq!(hits, 1, "the NaN pivot must be counted");
    assert!(l.iter().all(|x| x.is_finite()), "factor must stay finite: {l:?}");
    assert_eq!(l[0].to_bits(), 2.0f64.to_bits(), "healthy pivot untouched");
    assert_eq!(
        l[3].to_bits(),
        PD_FLOOR_F64.sqrt().to_bits(),
        "NaN pivot must be clamped to exactly sqrt(PD_FLOOR_F64)"
    );
    // and the solve through it stays finite
    let mut x = [1.0, 1.0];
    chol_solve(2, &l, &mut x);
    assert!(x.iter().all(|v| v.is_finite()), "solve must stay finite");

    // a NaN arriving via the accumulation (m[j][j] finite, an earlier
    // L entry NaN) is floored the same way — the second pivot becomes
    // 1 - NaN = NaN.
    let m2 = [1.0, f64::NAN, f64::NAN, 1.0];
    let mut l2 = [0.0f64; 4];
    assert_eq!(chol_factor_floored(2, &m2, &mut l2), 1);
    assert_eq!(l2[3].to_bits(), PD_FLOOR_F64.sqrt().to_bits());

    // control: a healthy 2×2 is bitwise-identical to the strict factor
    // (0 hits) — the negated comparison changes nothing off the NaN path
    let m3 = [4.0, 1.0, 1.0, 3.0];
    let (mut a, mut b) = ([0.0f64; 4], [0.0f64; 4]);
    assert_eq!(chol_factor_floored(2, &m3, &mut a), 0);
    assert!(chol_factor(2, &m3, &mut b));
    // LOWER TRIANGLE + diagonal only, as in the nv×nv comparison above:
    // neither factor zeroes the strict upper triangle (dead stores —
    // chol_solve reads only k ≤ i), and since 2026-08-03 the two-buffer
    // `chol_factor_floored` is a copy into `l` followed by the in-place
    // factor, so its upper cells hold the INPUT's upper triangle rather
    // than the destination's previous contents. No reader either way.
    for i in 0..2 {
        for k in 0..=i {
            assert_eq!(a[i * 2 + k].to_bits(), b[i * 2 + k].to_bits());
        }
    }
    // and a pivot exactly AT the floor is not counted (>= is inclusive,
    // matching the original `< FLOOR` boundary exactly)
    let m4 = [PD_FLOOR_F64, 0.0, 0.0, 1.0];
    let mut l4 = [0.0f64; 4];
    assert_eq!(chol_factor_floored(2, &m4, &mut l4), 0);
}

/// The same family end-to-end through `substep` (the old panic site):
/// an armature-less model parked at the gimbal-lock pose with joint
/// rates riding the caps must step WITHOUT panicking, increment
/// ws.pd_hits, and keep the state finite. (With the armature the
/// HUMANOID never engages the floor — pd_under_random_full_gears
/// below doubles as the 0-hit guard.)
#[test]
fn substep_survives_pd_loss_and_counts() {
    static ZERO_ARM: [f64; NQ] = [0.0; NQ];
    let mut m = HUMANOID;
    m.armature = &ZERO_ARM;
    m.damping = &ZERO_DAMP29;
    let m = &m;
    let h = 1.0 / 240.0;
    let mut ws = Workspace::new(m);
    let mut st = State::neutral(m);
    st.q[3] = m.limits[3].0; // gimbal-locked shoulders
    st.q[10] = m.limits[10].0;
    st.q[4] = 0.3;
    st.q[5] = -0.2;
    for j in 0..NQ {
        // high joint rates (the flaily-policy signature), inside caps
        st.v[6 + j] = if j % 2 == 0 { 12.0 } else { -12.0 };
    }
    let tau = vec![0.0; NQ];
    assert_eq!(ws.pd_hits, 0);
    for _ in 0..8 {
        substep(m, &mut st, &tau, h, &mut ws, None);
    }
    assert!(ws.pd_hits >= 1, "pivot floor must have engaged");
    assert!(st.q.iter().all(|q| q.is_finite()));
    assert!(st.v.iter().all(|v| v.is_finite()));
    assert!(st.p.iter().all(|p| p.is_finite()));
    assert!(st.quat.iter().all(|q| q.is_finite()));
    // the floor constant itself is part of the §22 contract
    assert_eq!(PD_FLOOR_F64.to_bits(), 1e-17f64.to_bits());
}

/// Long-horizon health under the actuation the trainer will actually
/// emit: seeded random FULL gears on every joint, re-drawn each 30 Hz
/// tick, 10 s free-floating (no welds — the P1 worst case; from P2 on
/// the foot welds ground net moments). The state must stay finite and
/// the §22 pivot floor must never engage (pd_hits stays 0 — the
/// armature keeps every pivot 12 decades above the floor here).
///
/// Deliberately NOT tested: constant tau_max on all 29 joints for
/// many seconds of unsupported free fall. That input sign-sustains a
/// net free moment of O(1000 N·m) on the whole body, which spins the
/// root past the explicit-Coriolis stability bound h·omega ~ 1 (~240
/// rad/s at h = 1/240) — a fixed-step explicit integrator cannot
/// follow it (MuJoCo included), and the welded v10 env can never
/// produce it.
#[test]
fn pd_under_random_full_gears() {
    let m = &HUMANOID;
    let h = 1.0 / 240.0;
    let mut st = State::neutral(m);
    let mut ws = Workspace::new(m);
    let mut rng = Rng(0x5EED_0000_0000_0005);
    let mut tau = vec![0.0; NQ];
    for _tick in 0..300 {
        for j in 0..NQ {
            let g = [0.0, 1.0, -1.0, 0.25, -0.25][rng.next_u64() as usize % 5];
            tau[j] = g * m.tau_max[j];
        }
        for _ in 0..8 {
            substep(m, &mut st, &tau, h, &mut ws, None);
        }
    }
    assert!(st.q.iter().all(|q| q.is_finite()));
    assert!(st.v.iter().all(|v| v.is_finite()));
    assert_eq!(ws.pd_hits, 0, "pivot floor must not engage in-limit");
    let wroot = (st.v[3] * st.v[3] + st.v[4] * st.v[4] + st.v[5] * st.v[5]).sqrt();
    println!("root |omega| after 10 s of random full gears: {wroot:.2} rad/s");
    assert!(wroot < 120.0, "root spin left the stability envelope: {wroot}");
}


/// v14 GATE — the phantom-force class (2026-08-10 three-agent audit):
/// joints PRESSED onto their limit stops in the quasi-static regime —
/// the regime the trained-policy exploit lived in (8-13 joints pinned,
/// rates ~0, +250-335 N of phantom support). Airborne (contact mode,
/// zero ground rows), every joint driven at 0.3·τ_max into a stop:
/// after the 1 s park-in, rates are ~0, so the explicit-Coriolis
/// discretization drift (∝ v², the KNOWN h·ω bound of this integrator
/// — measured ~1.6-3% g when flailing at the 16-40 rad/s caps, and NOT
/// what this gate is about) vanishes, and any CoM deviation from the
/// free-fall parabola is a non-reciprocal constraint write. Pre-v14
/// the direct-write guarantee pass leaked here every substep.
#[test]
fn v14_momentum_conserved_with_joints_parked_on_stops() {
    let m = &HUMANOID;
    let mut ws = Workspace::new(m);
    let mut st = State::neutral(m);
    st.p[1] = 500.0; // airborne: the y=0 plane is unreachable
    let mut cs = super::solver::ConstraintSet::new_modes(m, false, true, true);
    let h = 1.0 / 240.0;
    let g = super::model::G;
    let mut tau = [0.0; NQ];
    for j in 0..NQ {
        tau[j] = -0.3 * m.tau_max[j]; // press the lo stops, gently
    }
    // phase 1: park in (1 s) — rates die on the stops
    for _ in 0..240 {
        super::solver::substep_constrained(m, &mut st, &tau, h, &mut ws, &mut cs);
    }
    // measure the parked joint-rate scale (documents the regime)
    let vmax = (0..NQ).map(|j| st.v[m.voff() + j].abs()).fold(0.0f64, f64::max);
    // phase 2 reference: com + com-velocity by central finite difference
    super::fk::fk_all(m, &st, &mut ws);
    let c0 = super::testutil::body_com(m, &st, &mut ws);
    super::solver::substep_constrained(m, &mut st, &tau, h, &mut ws, &mut cs);
    super::fk::fk_all(m, &st, &mut ws);
    let c1 = super::testutil::body_com(m, &st, &mut ws);
    let v0 = [
        (c1[0] - c0[0]) / h,
        (c1[1] - c0[1]) / h + 0.5 * g * h, // remove the half-step gravity bias
        (c1[2] - c0[2]) / h,
    ];
    // phase 2: 2 s of sustained pressing; CoM must ride the parabola
    let mut worst = 0.0f64;
    for k in 0..480 {
        super::solver::substep_constrained(m, &mut st, &tau, h, &mut ws, &mut cs);
        assert!(cs.ground_contacts.is_empty(), "airborne probe touched ground");
        super::fk::fk_all(m, &st, &mut ws);
        let com = super::testutil::body_com(m, &st, &mut ws);
        let t = (k + 2) as f64 * h; // measured from c0
        let want = [
            c0[0] + v0[0] * t,
            c0[1] + v0[1] * t - 0.5 * g * t * t,
            c0[2] + v0[2] * t,
        ];
        let dev = ((com[0] - want[0]).powi(2)
            + (com[1] - want[1]).powi(2)
            + (com[2] - want[2]).powi(2))
        .sqrt();
        worst = worst.max(dev);
    }
    println!(
        "parked-on-stops free fall: max joint rate {vmax:.2e} rad/s, \
         worst CoM deviation {:.3} mm over 2 s",
        worst * 1e3
    );
    assert!(
        worst < 0.01,
        "phantom force: CoM deviates {:.1} mm from free-fall with joints \
         parked on stops (pre-v14 leaked ~0.3-3 m here)",
        worst * 1e3
    );
}
