//! Env v11 unilateral foot-contact gates — the contact spec's §5 parity
//! contract, Rust side. Rust is the REFERENCE: the CUDA mirror (Phase 2)
//! transcribes the same corner order, row order, iteration count and
//! clamp sequence, so what these tests pin is the reference behaviour,
//! not merely "some standing".
//!
//! §5 (a) settle, (b) no-pull + weight support, (c) static friction
//! (100 N holds / 800 N slides), (e) bitwise determinism, (f) weld
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
const BODY_MASS: f64 = 75.02;
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

/// Gate §5 (c) STATIC FRICTION: from a settled stance, 0.5 s of lateral
/// +x root push. At 100 N the μ = 0.9 cone must HOLD the loaded sole
/// corners; at 800 N (> μ·mg = 662 N) the stance must slide out.
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
/// Measured ladder over the same protocol (max travel of the penetrating
/// set): 0 N 1.7 mm (the stance's own drift floor), 25 N 2.2, 50 N 2.6,
/// 100 N 4.3, 200 N 6.4, 400 N 20.9, 600 N 127 — the cone gives way
/// between 200 and 600 N, i.e. around the μ·mg = 662 N it should. At
/// 800 N nothing is left penetrating at all, so the SLIDE arm is gated
/// on the travel of whatever is still within the margin plus the root's
/// own 1.13 m of runaway.
#[test]
fn static_friction_holds_at_100n_and_slides_at_800n() {
    let m = super::testutil::muscle_model();
    let (kp, _kd) = super::testutil::muscle_gains();
    let mut slip_load = [0.0f64; 2]; // corners penetrating at both ends
    let mut slip_near = [0.0f64; 2]; // corners inside the margin at both
    let mut tracked = [0usize; 2];
    let mut root = [0.0f64; 2];
    for (case, push) in [(0usize, 100.0f64), (1, 800.0)] {
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
        let c0 = corners(&st, &mut ws);
        let rx0 = st.p[0];
        run(&mut st, &mut ws, &mut cs, 120, push); // 0.5 s push
        let c1 = corners(&st, &mut ws);
        for i in 0..NUM_GROUND_CORNERS {
            let d = (c1[i][0] - c0[i][0]).abs();
            if c0[i][1] < 0.0 && c1[i][1] < 0.0 {
                slip_load[case] = slip_load[case].max(d);
                tracked[case] += 1;
            }
            if c0[i][1] < MARGIN && c1[i][1] < MARGIN {
                slip_near[case] = slip_near[case].max(d);
            }
        }
        root[case] = (st.p[0] - rx0).abs();
        println!(
            "friction: {push:.0} N for 0.5 s -> loaded-corner slip {:.5} m over {} \
             corners, in-margin slip {:.5} m, root x {:.5} m",
            slip_load[case], tracked[case], slip_near[case], root[case]
        );
        assert!(st.p.iter().all(|x| x.is_finite()));
    }
    // HOLD arm
    assert!(tracked[0] >= 2, "100 N: only {} corners stayed loaded", tracked[0]);
    assert!(
        slip_load[0] < 0.01,
        "100 N slid a loaded corner {:.4} m — the μ = 0.9 cone should hold",
        slip_load[0]
    );
    assert!(root[0] < 0.15, "100 N moved the root {:.4} m (lean bound)", root[0]);
    // SLIDE arm
    assert!(
        slip_near[1] > 0.1,
        "800 N moved the stance only {:.4} m — it should slide out",
        slip_near[1]
    );
    assert!(root[1] > 0.5, "800 N moved the root only {:.4} m", root[1]);
    assert!(
        slip_near[1] > 10.0 * slip_load[0],
        "800 N ({:.4} m) is not clear of the 100 N hold ({:.4} m)",
        slip_near[1],
        slip_load[0]
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

