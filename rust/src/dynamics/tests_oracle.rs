//! Gate (e): MuJoCo reference-trajectory bands. The 5 refdata scenarios
//! (python/oracle/gen_reference.py) replayed through the Rust engine
//! with the SAME per-substep pattern torque + muscle-tone stabilizer
//! law (τ_total = tau29 + kp·(0 − q) − kd·q̇; kd REPLACES the spec
//! damping b_j), on the FREE-FLOATING body: self-collision rows and
//! joint limits live, no ground and no foot constraint (the refdata
//! were regenerated so 2026-09-27, when the v10 foot welds they used to
//! run under were removed — issue #2). MuJoCo's Newton solver and
//! implicit-damping Euler are NOT our PGS-12 / explicit damping, so the
//! comparison uses tolerance BANDS, never parity: max |Δq| ≤ 0.05 rad
//! on the 4 smooth scenarios, ≤ 0.15 rad on random_all29.
//!
//! refdata qpos rows are MuJoCo tree order (shoulders yaw,pitch,twist /
//! wrists roll,dev,bend) — mapped to §1.4 actuated order by joint name
//! before comparing.

use super::model::NQ;
use super::solver::{substep_constrained_damped, ConstraintSet};
use super::testjson::{parse, Json};
use super::testutil::muscle_model;
use super::{State, Workspace};

const H: f64 = 1.0 / 240.0;

/// §1.4 actuated order — must match each refdata `joint_order`.
const JOINT_ORDER: [&str; NQ] = [
    "spine_pitch",
    "spine_yaw",
    "spine_roll",
    "l_arm_pitch",
    "l_arm_yaw",
    "l_arm_twist",
    "l_elbow",
    "l_wrist_roll",
    "l_wrist_bend",
    "l_wrist_dev",
    "r_arm_pitch",
    "r_arm_yaw",
    "r_arm_twist",
    "r_elbow",
    "r_wrist_roll",
    "r_wrist_bend",
    "r_wrist_dev",
    "l_hip_pitch",
    "l_hip_yaw",
    "l_hip_roll",
    "l_knee",
    "l_ankle_pitch",
    "l_ankle_roll",
    "r_hip_pitch",
    "r_hip_yaw",
    "r_hip_roll",
    "r_knee",
    "r_ankle_pitch",
    "r_ankle_roll",
];

fn load(name: &str) -> Json {
    let path = format!(
        "{}/../python/oracle/refdata/{}.json",
        env!("CARGO_MANIFEST_DIR"),
        name
    );
    let text = std::fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("cannot read {path}: {e} — run gen_reference.py"));
    parse(&text)
}

/// Replay one scenario; returns (max |Δq| over all substeps and joints,
/// max root position error).
fn band(name: &str) -> (f64, f64) {
    let doc = load(name);
    assert_eq!(doc.get("h").num(), H);
    assert_eq!(doc.get("integrator").str(), "Euler");
    let n_sub = doc.get("n_substeps").num() as usize;
    let joint_order = doc.get("joint_order").strs();
    assert_eq!(joint_order, JOINT_ORDER, "{name}: joint_order mismatch");
    let qpos_names = doc.get("qpos_joint_names").strs();
    // qmap[j] = qpos column of actuated joint j (root pos 3 + quat 4
    // lead the row)
    let qmap: Vec<usize> = JOINT_ORDER
        .iter()
        .map(|n| {
            7 + qpos_names
                .iter()
                .position(|q| q == n)
                .unwrap_or_else(|| panic!("{name}: joint {n} missing from qpos_joint_names"))
        })
        .collect();
    let stab = doc.get("stabilizer");
    let kp = stab.get("kp").nums();
    let kd = stab.get("kd").nums();
    let setpoint = stab.get("setpoint").nums();
    let tau29 = doc.get("tau29").rows();
    let qpos = doc.get("qpos").rows();
    assert_eq!(tau29.len(), n_sub);
    assert_eq!(qpos.len(), n_sub + 1);

    let m = muscle_model(); // kd is the TOTAL damping — spec b_j off
    let mut ws = Workspace::new(&m);
    let mut st = State::neutral(&m);
    let mut cs = ConstraintSet::new(&m, true, false); // floating
    // row 0 must be the shared neutral state
    assert!((qpos[0][0]).abs() < 1e-12 && (qpos[0][1] - 0.94).abs() < 1e-12);
    for j in 0..NQ {
        assert!(qpos[0][qmap[j]].abs() < 1e-12, "{name}: non-neutral start");
    }

    let mut tau = [0.0; NQ];
    let mut max_dq = 0.0f64;
    let mut max_droot = 0.0f64;
    for k in 0..n_sub {
        for j in 0..NQ {
            tau[j] = tau29[k][j] + kp[j] * (setpoint[j] - st.q[j]);
        }
        // kd applied implicitly — MuJoCo-Euler damping semantics (see
        // step.rs substep_opts; explicit kd is unstable at h = 1/240
        // for the collinear straight-arm wrist DoFs)
        substep_constrained_damped(&m, &mut st, &tau, &kd, H, &mut ws, &mut cs);
        let row = &qpos[k + 1];
        for j in 0..NQ {
            max_dq = max_dq.max((st.q[j] - row[qmap[j]]).abs());
        }
        for t in 0..3 {
            max_droot = max_droot.max((st.p[t] - row[t]).abs());
        }
    }
    (max_dq, max_droot)
}

#[test]
fn oracle_trajectory_bands() {
    let smooth = [
        "neutral_hold",
        "const_l_arm_pitch",
        "sin_both_arms",
        "leg_spine",
    ];
    let mut fails: Vec<String> = Vec::new();
    for name in smooth {
        let (dq, droot) = band(name);
        println!("oracle band {name}: max |dq| {dq:.5} rad, root {droot:.6} m");
        if dq > 0.05 {
            fails.push(format!("{name}: {dq:.5} > 0.05"));
        }
    }
    let (dq, droot) = band("random_all29");
    println!("oracle band random_all29: max |dq| {dq:.5} rad, root {droot:.6} m");
    if dq > 0.15 {
        fails.push(format!("random_all29: {dq:.5} > 0.15"));
    }
    assert!(fails.is_empty(), "oracle bands blown: {fails:?}");
}
