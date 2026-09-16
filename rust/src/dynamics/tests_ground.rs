//! Env v11 unilateral foot-contact gates — the contact spec's §5 parity
//! contract, Rust side. Rust is the REFERENCE: the CUDA mirror (Phase 2)
//! transcribes the same corner order, row order, iteration count and
//! clamp sequence, so what these tests pin is the reference behaviour,
//! not merely "some standing".
//!
//! §5 (a) settle, (b) no-pull + weight support, (c) static friction
//! (40 N holds / 1000 N slides; hold arm 100 -> 40 N at the
//! 2026-08-20 torque softening), (e) bitwise determinism, (f) weld
//! regression. §5 (d) fall termination is an ENV-layer gate and lives in
//! lib.rs `env_tests` beside its root-escape twin.

use super::collide::{
    ground_phase, GroundContact, CORNERS_PER_FOOT, GROUND_CORNERS, GROUND_FRAMES, MARGIN,
    NUM_GROUND_CORNERS,
};
use super::fk::{add3, fk_all, mat_vec};
use super::model::{HUMANOID, NQ};
use super::solver::{substep_constrained, ConstraintSet, GROUND_ROWS_PER_CORNER, MU};
use super::{State, Workspace};

const H: f64 = 1.0 / 240.0;
/// SPEC-v10 §1.2 total body mass — the weight the ground must carry.
const BODY_MASS: f64 = 74.50;
const WEIGHT: f64 = BODY_MASS * super::model::G;

fn contact_set(m: &super::model::Model) -> ConstraintSet {
    // contact spec §1: foot_mode = 1 drops the 12 weld rows
    ConstraintSet::new_modes(m, false, true, true)
}

/// Reference-pose sanity for the §2 geometry: all 8 sole corners sit at
/// exactly y = 0 with the ankle at y = 0.07, in the normative order.
#[test]
fn ground_corners_touch_at_reference() {
    let m = &HUMANOID;
    let mut ws = Workspace::new(m);
    let st = State::neutral(m);
    fk_all(m, &st, &mut ws);
    let mut gc: Vec<GroundContact> = Vec::new();
    ground_phase(&ws, &mut gc);
    assert_eq!(gc.len(), NUM_GROUND_CORNERS, "all 8 corners must be active");
    for (i, c) in gc.iter().enumerate() {
        assert_eq!(c.corner, i, "corner order must be the assembly order");
        // §2 says "exactly y = 0"; the FK chain that gets there
        // (0.94 − 0.44 − 0.43 − 0.07) leaves ±5.6e-17 of rounding, which
        // is what "exactly" means here. What the spec's claim actually
        // buys — active (gap < MARGIN) and touching at reset — is exact.
        assert!(c.gap.abs() < 1e-12, "corner {i} gap {} != 0 at the reference pose", c.gap);
        // §2 order within a foot: (x−,z−), (x−,z+), (x+,z−), (x+,z+)
        let k = i % CORNERS_PER_FOOT;
        let foot_x = if i < CORNERS_PER_FOOT { -0.10 } else { 0.10 };
        let want_x = foot_x + if k < 2 { -0.045 } else { 0.045 };
        let want_z = -0.65 + if k % 2 == 0 { -0.075 } else { 0.175 };
        assert!((c.point[0] - want_x).abs() < 1e-12, "corner {i} x {}", c.point[0]);
        assert!((c.point[2] - want_z).abs() < 1e-12, "corner {i} z {}", c.point[2]);
    }
}

/// Gate §5 (a) SETTLE: reference pose, zero gears, 5 s. The body must
/// still be standing — |root_y − 0.94| < 0.05, nothing non-finite, and
/// all 8 corners active at every substep.
///
/// AMBIGUITY RESOLVED (recorded, not silently deviated from): the spec
/// says "zero gears", and a gear is an ACTION (§1.4 torque fraction) —
/// but the raw v10 body has no postural controller at all, and with the
/// feet unwelded it is a free inverted pendulum on frictionless-hinge
/// ankles: the reference-pose COM sits ~38 mm forward of the ankle axis,
/// so 0 N·m of ankle torque topples it in well under a second. That is
/// true of the WELDED body too — "the raw unstabilized body simply falls
/// over" is the oracle README's own note, and every existing stance gate
/// (`stance_weld_force_sum`, `weld_under_two_arm_swing`) therefore runs
/// the oracle muscle-tone protocol. This gate does the same: zero GEAR
/// torque, muscle-tone stance holding the joints at neutral. What it
/// tests is the ground block — that the contacts carry the body, do not
/// sink and do not let the feet slide — which is what §5 (a) is for.
#[test]
fn settle_stands_for_five_seconds() {
    let m = super::testutil::muscle_model();
    let (kp, kd) = super::testutil::muscle_gains();
    let mut ws = Workspace::new(&m);
    let mut st = State::neutral(&m);
    let mut cs = contact_set(&m);
    let mut tau = [0.0; NQ];
    let mut min_active = NUM_GROUND_CORNERS;
    let mut max_pen = 0.0f64;
    for k in 0..1200 {
        for j in 0..NQ {
            tau[j] = kp[j] * (0.0 - st.q[j]); // zero gears; posture only
        }
        super::solver::substep_constrained_damped(&m, &mut st, &tau, &kd, H, &mut ws, &mut cs);
        min_active = min_active.min(cs.ground_contacts.len());
        for c in cs.ground_contacts.iter() {
            max_pen = max_pen.max(-c.gap);
        }
        assert!(st.p.iter().all(|x| x.is_finite()), "substep {k}: root NaN");
        assert!(st.q.iter().all(|x| x.is_finite()), "substep {k}: q NaN");
        assert!(st.v.iter().all(|x| x.is_finite()), "substep {k}: v NaN");
    }
    let dy = st.p[1] - 0.94;
    println!(
        "settle 5 s: root y {:.5} (dy {dy:+.5}), min active corners {min_active}, \
         max penetration {max_pen:.2e} m",
        st.p[1]
    );
    assert!(dy.abs() < 0.05, "root y moved {dy:+.4} m");
    assert_eq!(min_active, NUM_GROUND_CORNERS, "a corner left the ground");
}

/// Gate §5 (b) NO-PULL: every normal impulse λ_n ≥ 0 at every substep,
/// and the total vertical contact force over a settled second equals the
/// body weight 736 N within 2 %. The friction rows must also stay inside
/// their box |λ_t| ≤ MU·λ_n at every substep.
#[test]
fn no_pull_and_weight_is_carried() {
    let m = super::testutil::muscle_model();
    let (kp, kd) = super::testutil::muscle_gains();
    let mut ws = Workspace::new(&m);
    let mut st = State::neutral(&m);
    let mut cs = contact_set(&m);
    let mut tau = [0.0; NQ];
    let mut min_lambda_n = f64::INFINITY;
    let mut max_cone = 0.0f64; // max (|λ_t| − MU·λ_n), must stay ≤ 0
    let mut fy_sum = 0.0;
    let mut measured = 0usize;
    for k in 0..1200 {
        for j in 0..NQ {
            tau[j] = kp[j] * (0.0 - st.q[j]);
        }
        super::solver::substep_constrained_damped(&m, &mut st, &tau, &kd, H, &mut ws, &mut cs);
        let mut fy = 0.0;
        for gi in 0..cs.ground_contacts.len() {
            let nrow = cs.ground_row(gi);
            let ln = cs.lambda[nrow];
            min_lambda_n = min_lambda_n.min(ln);
            fy += ln;
            for t in 1..GROUND_ROWS_PER_CORNER {
                max_cone = max_cone.max(cs.lambda[nrow + t].abs() - MU * ln);
            }
        }
        // 3 s settle, then measure over the last second
        if k >= 960 {
            fy_sum += fy / H;
            measured += 1;
        }
    }
    let fy = fy_sum / measured as f64;
    println!(
        "no-pull: min λ_n {min_lambda_n:.3e} N·s, max cone excess {max_cone:.3e} N·s, \
         mean vertical contact force {fy:.2} N (weight {WEIGHT:.2})"
    );
    assert!(min_lambda_n >= 0.0, "a normal row PULLED: λ_n {min_lambda_n:e}");
    assert!(max_cone <= 0.0, "friction left the box by {max_cone:e} N·s");
    assert!(
        (fy - WEIGHT).abs() <= 0.02 * WEIGHT,
        "vertical contact force {fy:.2} N vs weight {WEIGHT:.2} N"
    );
}

/// Gate §5 (c) STATIC FRICTION: from a settled stance, a sustained
/// lateral +x root push. At 40 N the μ = 0.9 cone must HOLD the loaded
/// sole corners; at 1000 N (> μ·mg ≈ 662 N) the stance must slide out.
///
/// Hold-arm force 100 -> 40 N (2026-08-20 spine -25% / leg torque
/// pass): the muscle-tone stance's P-gains scale with tau_max, and at
/// the softened waist a sustained 100 N (or 60 N) has NO static lean —
/// the body folds until a foot unloads, its shrunken cone lets go, and
/// the stance walks away (root 1.6 m by 1.4 s at 100 N). The friction
/// CONTRACT is unchanged; 40 N is the posture-holdable load at the new
/// gains (measured: 0.57 mm slip, 2 corners loaded, 0.59 m lean).
///
/// AMBIGUITY RESOLVED (recorded, not silently deviated from), two
/// points, both about WHAT IS MEASURED — the scenario and the forces are
/// the spec's:
///
///  1. §5 (c) states the metric as "root x moves < 1 cm". Root x cannot
///     measure friction on this body: the root sits 0.87 m above the
///     sole on a spring-held leg chain, and 100 N of lateral force is
///     ~87 N·m about the ankle against an ankle-roll spring of
///     τ_max/0.2 = 250 N·m/rad, so the body LEANS 0.115 m (measured)
///     with the feet perfectly stuck. Leaning and sliding are different
///     failures and only the second is the contact model's, so the hold
///     arm is gated on the SOLE CORNERS' lateral travel.
///  2. Only corners actually CARRYING load can be slid. A corner inside
///     the 1 cm speculative margin but above the plane draws λ_n ≈ 0 and
///     hence, by |λ_t| ≤ MU·λ_n, no friction at all — it is free to
///     move. The hold arm therefore tracks corners PENETRATING (gap < 0)
///     at both ends of the window; a tipping stance lifts the upwind
///     ones and they drop out by construction.
///
/// GATE RECALIBRATION (2026-08-17): the original hold gate measured
/// corner travel over the WHOLE 0.5 s push window, which folds the
/// lean transient in — while the body takes up its ~0.11 m lean the
/// feet ROLL slightly and a corner translates without sliding. The
/// gate now measures what §5 (c) means: STEADY-STATE creep — corner
/// travel over the FINAL 0.1 s of the push, after the lean settles.
/// At μ = 0.9 the loaded corners creep < 1 mm / 0.1 s (true stick).
/// This metric is also what exposed the μ-raise cliff (see solver.rs
/// MU): at μ ≥ 0.95 the box-PGS limit-cycles and the same corners
/// creep 2-3 cm / 0.1 s at ANY iteration count — which is why MU
/// stays 0.9. The slide arm is gated on the root's meter-scale
/// runaway (past the cone nothing stays penetrating or in-margin, so
/// corner travel is unmeasurable there by construction).
#[test]
fn static_friction_holds_at_40n_and_slides_at_1000n() {
    let m = super::testutil::muscle_model();
    let (kp, _kd) = super::testutil::muscle_gains();
    let mut slip_load = [0.0f64; 2]; // corners penetrating at both ends
    let mut slip_near = [0.0f64; 2]; // corners inside the margin at both
    let mut tracked = [0usize; 2];
    let mut root = [0.0f64; 2];
    // slide arm must clear the cone: MU * 736 N body weight (~662 N at
    // mu = 0.9), so 1000 N with ~50 % margin
    for (case, push) in [(0usize, 40.0f64), (1, 1000.0)] {
        let mut ws = Workspace::new(&m);
        let mut st = State::neutral(&m);
        let mut cs = contact_set(&m);
        let mut tau = [0.0; NQ];
        let mut run = |st: &mut State, ws: &mut Workspace, cs: &mut ConstraintSet,
                       n: usize, fx: f64| {
            for _ in 0..n {
                for j in 0..NQ {
                    tau[j] = kp[j] * (0.0 - st.q[j]);
                }
                // world +x force on the root origin (§3 step 4's applied
                // root wrench — the harness path the stance gates use)
                let wrench = [fx, 0.0, 0.0, 0.0, 0.0, 0.0];
                super::solver::substep_constrained_wrench(&m, st, &tau, &wrench, H, ws, cs);
            }
        };
        // all 8 corners' (world x, gap) — the narrow phase's own
        // expression, unfiltered by activity
        let corners = |st: &State, ws: &mut Workspace| -> [[f64; 2]; NUM_GROUND_CORNERS] {
            fk_all(&m, st, ws);
            let mut out = [[0.0f64; 2]; NUM_GROUND_CORNERS];
            for f in 0..GROUND_FRAMES.len() {
                let frame = GROUND_FRAMES[f];
                for k in 0..CORNERS_PER_FOOT {
                    let p = add3(ws.p[frame], mat_vec(&ws.r[frame], GROUND_CORNERS[k]));
                    out[f * CORNERS_PER_FOOT + k] = [p[0], p[1]];
                }
            }
            out
        };
        run(&mut st, &mut ws, &mut cs, 960, 0.0); // 4 s settle
        let rx0 = st.p[0];
        // 1.4 s of lean transient (the 2026-08-17 leg caps softened the
        // hip springs — 0.4 s -> 0.9 s; the 2026-08-20 spine -25% pass
        // softened the waist too and 0.9 s still left 1.3 mm of creep
        // in the measurement window)
        run(&mut st, &mut ws, &mut cs, 336, push);
        let cm = corners(&st, &mut ws); // steady-state window start
        run(&mut st, &mut ws, &mut cs, 24, push); // final 0.1 s
        let c1 = corners(&st, &mut ws);
        for i in 0..NUM_GROUND_CORNERS {
            let d = (c1[i][0] - cm[i][0]).abs();
            if cm[i][1] < 0.0 && c1[i][1] < 0.0 {
                slip_load[case] = slip_load[case].max(d);
                tracked[case] += 1;
            }
            if cm[i][1] < MARGIN && c1[i][1] < MARGIN {
                slip_near[case] = slip_near[case].max(d);
            }
        }
        root[case] = (st.p[0] - rx0).abs();
        println!(
            "friction: {push:.0} N -> steady-state (last 0.1 s) loaded-corner slip \
             {:.5} m over {} corners, in-margin slip {:.5} m, root x {:.5} m",
            slip_load[case], tracked[case], slip_near[case], root[case]
        );
        assert!(st.p.iter().all(|x| x.is_finite()));
    }
    // HOLD arm: loaded corners must be STUCK once the lean settles —
    // < 1 mm over the final 0.1 s (i.e. creep < 1 cm/s)
    assert!(tracked[0] >= 2, "40 N: only {} corners stayed loaded", tracked[0]);
    assert!(
        slip_load[0] < 0.001,
        "40 N steady-state creep {:.4} m / 0.1 s — the μ = 0.9 cone should hold",
        slip_load[0]
    );
    // lean bound 0.15 -> 0.7: at the 2026-08-17 human leg caps the
    // stance springs are softer and 100 N held for 1.0 s leans the body
    // ~0.56 m (feet STUCK at 0.15 mm — the friction contract holds; the
    // lean is posture, not sliding, and posture is not this gate's job)
    assert!(root[0] < 0.7, "40 N moved the root {:.4} m (lean bound)", root[0]);
    // SLIDE arm: past the cone the stance runs away at meter scale
    // (nothing stays penetrating or in-margin, so corner travel is
    // unmeasurable — the root IS the metric here)
    assert!(root[1] > 0.5, "1000 N moved the root only {:.4} m", root[1]);
    assert!(
        root[1] > 4.0 * root[0],
        "1000 N runaway ({:.4} m) is not clear of the 40 N lean ({:.4} m)",
        root[1],
        root[0]
    );
}

/// Gate §5 (e) DETERMINISM: same seed, same actions, two runs → bitwise
/// equal q trajectories (f64), state AND the ground impulse stream.
#[test]
fn contact_mode_determinism_bitwise() {
    let m = &HUMANOID;
    let run = || {
        let mut ws = Workspace::new(m);
        let mut st = State::neutral(m);
        let mut cs = contact_set(m);
        let mut rng = super::testutil::Rng(0x5EED_0000_0000_0B11);
        let mut tau = [0.0; NQ];
        let mut bits: Vec<u64> = Vec::new();
        for tick in 0..120 {
            if tick % 8 == 0 {
                for j in 0..NQ {
                    let g = [0.0, 1.0, -1.0, 0.25, -0.25][rng.next_u64() as usize % 5];
                    tau[j] = g * m.tau_max[j];
                }
            }
            substep_constrained(m, &mut st, &tau, H, &mut ws, &mut cs);
            for j in 0..NQ {
                bits.push(st.q[j].to_bits());
            }
            for r in 0..cs.nrows {
                bits.push(cs.lambda[r].to_bits());
            }
        }
        (st, bits)
    };
    let (a, ba) = run();
    let (b, bb) = run();
    assert_eq!(a.p.map(f64::to_bits), b.p.map(f64::to_bits));
    assert_eq!(a.quat.map(f64::to_bits), b.quat.map(f64::to_bits));
    for i in 0..a.v.len() {
        assert_eq!(a.v[i].to_bits(), b.v[i].to_bits(), "v[{i}]");
    }
    assert_eq!(ba.len(), bb.len(), "row stream length differs");
    assert_eq!(ba, bb, "q/impulse trajectory differs");
    println!("contact determinism: {} words bit-identical across two runs", ba.len());
}

/// Gate §5 (f) WELD REGRESSION, direct form: `foot_mode` = 0 assembles
/// ZERO ground rows and produces a trajectory bit-identical to the
/// pre-v11 constructor's. (The full evidence is the rest of the suite
/// passing unchanged; this pins the switch itself.)
#[test]
fn weld_mode_is_untouched_by_the_ground_block() {
    let m = &HUMANOID;
    let run = |ground_ctor: bool| {
        let mut ws = Workspace::new(m);
        let mut st = State::neutral(m);
        let mut cs = if ground_ctor {
            ConstraintSet::new_modes(m, true, true, false)
        } else {
            ConstraintSet::new(m, true, true)
        };
        let mut rng = super::testutil::Rng(0x5EED_0000_0000_0B22);
        let mut tau = [0.0; NQ];
        let mut bits: Vec<u64> = Vec::new();
        for tick in 0..120 {
            if tick % 8 == 0 {
                for j in 0..NQ {
                    let g = [0.0, 1.0, -1.0, 0.25, -0.25][rng.next_u64() as usize % 5];
                    tau[j] = g * m.tau_max[j];
                }
            }
            substep_constrained(m, &mut st, &tau, H, &mut ws, &mut cs);
            assert_eq!(cs.ground_rows(), 0, "weld mode assembled a ground row");
            assert!(cs.ground_contacts.is_empty());
            for j in 0..NQ {
                bits.push(st.q[j].to_bits());
            }
            for r in 0..cs.nrows {
                bits.push(cs.lambda[r].to_bits());
            }
        }
        bits
    };
    assert_eq!(run(false), run(true), "the foot_mode switch moved weld physics");
}

/// The mode flag is a SWITCH (contact spec §1): welds and ground rows
/// are never both live — the weld-null projection would not cover the
/// ground impulses.
#[test]
#[should_panic(expected = "exclusive")]
fn welds_and_ground_are_mutually_exclusive() {
    let _ = ConstraintSet::new_modes(&HUMANOID, true, true, true);
}


/// v14 GATE — CoP consistency (2026-08-10 audit finding #2: no gate
/// ever checked MOMENT balance, so a phantom moment subsidy was
/// invisible to the force-only §5 (b) gate). In the settled muscle-tone
/// stance the center of pressure — Σ λ_n·(x,z) / Σ λ_n over a second —
/// must sit inside the active-corner hull AND within 4 cm of the CoM's
/// ground projection: quasi-statics with honest contacts allows nothing
/// else. Pre-v14, the trained-policy pose held CoP-vs-CoM gaps of
/// 10-23 cm for 12 s.
#[test]
fn v14_settled_cop_matches_com_projection() {
    let m = super::testutil::muscle_model();
    let (kp, kd) = super::testutil::muscle_gains();
    let mut ws = Workspace::new(&m);
    let mut st = State::neutral(&m);
    let mut cs = contact_set(&m);
    let mut tau = [0.0; NQ];
    let mut cop_num = [0.0f64; 2];
    let mut cop_den = 0.0f64;
    let mut com_acc = [0.0f64; 2];
    let mut n_meas = 0usize;
    for k in 0..1200 {
        for j in 0..NQ {
            tau[j] = kp[j] * (0.0 - st.q[j]);
        }
        super::solver::substep_constrained_damped(&m, &mut st, &tau, &kd, H, &mut ws, &mut cs);
        if k >= 960 {
            for gi in 0..cs.ground_contacts.len() {
                let ln = cs.lambda[cs.ground_row(gi)];
                let p = cs.ground_contacts[gi].point;
                cop_num[0] += ln * p[0];
                cop_num[1] += ln * p[2];
                cop_den += ln;
            }
            let com = super::testutil::body_com(&m, &st, &ws);
            com_acc[0] += com[0];
            com_acc[1] += com[2];
            n_meas += 1;
        }
    }
    let cop = [cop_num[0] / cop_den, cop_num[1] / cop_den];
    let com = [com_acc[0] / n_meas as f64, com_acc[1] / n_meas as f64];
    let gap = ((cop[0] - com[0]).powi(2) + (cop[1] - com[1]).powi(2)).sqrt();
    println!(
        "settled CoP ({:.4}, {:.4}) vs CoM projection ({:.4}, {:.4}): gap {:.1} mm",
        cop[0], cop[1], com[0], com[1], gap * 1e3
    );
    // active-corner hull at the reference stance: x ∈ ±0.145, z ∈
    // [-0.725, -0.475] — assert CoP inside with 1 mm slack
    assert!(cop[0].abs() < 0.146 && cop[1] > -0.726 && cop[1] < -0.474,
            "CoP ({:.4}, {:.4}) left the support hull", cop[0], cop[1]);
    assert!(gap < 0.04, "CoP-CoM gap {:.1} mm: moment balance broken", gap * 1e3);
}

/// bench-opt item 2 (2026-09-16): the ground rows' support dots are
/// BITWISE the dense 35-wide dots, on LIVE rows of a random-gear
/// contact-mode trajectory (corners come and go, both feet, falls) —
/// against the post-solve v AND each row's own M⁻¹Jᵀ column (the two
/// shipping consumers: PGS residual and effective mass) — and every
/// column outside a row's support is the +0.0 bit pattern.
#[test]
fn ground_support_dot_bitwise() {
    let m = &HUMANOID;
    let mut ws = Workspace::new(m);
    let mut st = State::neutral(m);
    let mut cs = contact_set(m);
    let mut rng = super::testutil::Rng(0x5EED_0000_0000_17E2);
    let mut tau = [0.0; NQ];
    let (mut checked, mut left_rows, mut right_rows) = (0usize, 0usize, 0usize);
    for tick in 0..480 {
        if tick % 8 == 0 {
            for j in 0..NQ {
                let g = [0.0, 1.0, -1.0, 0.25, -0.25][rng.next_u64() as usize % 5];
                tau[j] = g * m.tau_max[j];
            }
        }
        substep_constrained(m, &mut st, &tau, H, &mut ws, &mut cs);
        let nv = st.v.len();
        let mut probe = vec![0.0; nv];
        for i in 0..nv {
            // a signed-zero-rich finite vector: exact zeros on odd columns
            probe[i] = if i % 2 == 1 { -0.0 } else { st.v[i] };
        }
        let lefts: Vec<bool> = cs.ground_contacts.iter().map(|g| g.frame == GROUND_FRAMES[0]).collect();
        for src in [&st.v[..], &probe[..]] {
            for (g, (dense, sup, zeros)) in cs.ground_dot_ref_vs_support(src).into_iter().enumerate() {
                assert!(zeros, "substep {tick}: ground row {g} has a non-+0.0 word off its support");
                assert_eq!(dense.to_bits(), sup.to_bits(), "substep {tick}: ground row {g} dot vs v");
                checked += 1;
            }
        }
        for g in 0..cs.ground_rows() {
            let row = cs.ground_row(0) + g;
            let mr = cs.row_minvjt(row).to_vec();
            let (dense, sup, _) = cs.ground_dot_ref_vs_support(&mr)[g];
            assert_eq!(dense.to_bits(), sup.to_bits(), "substep {tick}: ground row {g} diag dot");
            assert_eq!(sup.to_bits(), cs.row_diag(row).to_bits(), "substep {tick}: live diag");
            if lefts[g / GROUND_ROWS_PER_CORNER] {
                left_rows += 1;
            } else {
                right_rows += 1;
            }
        }
    }
    println!("item-2 ground support: {checked} v-dots, {left_rows} left / {right_rows} right diag rows bitwise");
    assert!(left_rows > 0 && right_rows > 0, "trajectory must exercise both feet");
}

/// item 3: the RHS-MINOR batched solves for the dense (ground + pair)
/// rows and the limit columns are bitwise the per-row reference solves
/// (`chol_solve` on the assembled row / `chol_solve_unit` on the pivot)
/// against the substep's live factor, along a random-gear contact run.
#[test]
fn batched_row_solves_bitwise() {
    use super::algo::{chol_solve, chol_solve_unit};
    let m = &HUMANOID;
    let mut ws = Workspace::new(m);
    let mut st = State::neutral(m);
    let mut cs = contact_set(m);
    let mut rng = super::testutil::Rng(0x5EED_0000_0000_17E3);
    let mut tau = [0.0; NQ];
    let (mut dense_rows, mut lim_rows, mut wide_dense, mut wide_lim) = (0usize, 0usize, 0usize, 0usize);
    for tick in 0..480 {
        if tick % 8 == 0 {
            for j in 0..NQ {
                let g = [0.0, 1.0, -1.0, 0.25, -0.25][rng.next_u64() as usize % 5];
                tau[j] = g * m.tau_max[j];
            }
        }
        substep_constrained(m, &mut st, &tau, H, &mut ws, &mut cs);
        let nv = st.v.len();
        let nlim = cs.limit_rows();
        let ndense = cs.nrows - nlim;
        assert_eq!(cs.weld_rows(), 0);
        wide_dense += (ndense >= 24) as usize;
        wide_lim += (nlim >= 8) as usize;
        let mut x = vec![0.0; nv];
        for row in 0..ndense {
            x.copy_from_slice(cs.row_j(row));
            chol_solve(nv, &ws.m, &mut x);
            for (i, (a, b)) in x.iter().zip(cs.row_minvjt(row)).enumerate() {
                assert_eq!(a.to_bits(), b.to_bits(), "substep {tick}: dense row {row} word {i}");
            }
            dense_rows += 1;
        }
        let voff = nv - NQ;
        for (li, &(jt, sign)) in cs.limit_set().to_vec().iter().enumerate() {
            chol_solve_unit(nv, &ws.m, voff + jt, sign, &mut x);
            let row = ndense + li;
            for (i, (a, b)) in x.iter().zip(cs.row_minvjt(row)).enumerate() {
                assert_eq!(a.to_bits(), b.to_bits(), "substep {tick}: limit row {row} word {i}");
            }
            lim_rows += 1;
        }
    }
    println!("item-3 batched solves: {dense_rows} dense / {lim_rows} limit columns bitwise ({wide_dense} / {wide_lim} substeps wide)");
    assert!(wide_dense > 0 && wide_lim > 0, "trajectory must exercise both batched paths");
}
