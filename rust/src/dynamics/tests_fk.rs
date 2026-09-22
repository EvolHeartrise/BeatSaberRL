//! P1 FK gates: model-table sanity and the neutral-stance anchors
//! (SPEC-v10 §1.1).

use super::fk::fk_all;
use super::model::{
    HUMANOID, F_HEAD, F_L_FOOT, F_L_SHOULDER, F_R_FOOT, F_R_SHOULDER, NQ,
};
use super::testutil::Rng;
use super::{State, Workspace};

const TOL: f64 = 1e-12;

fn assert_vec3(got: [f64; 3], want: [f64; 3], what: &str) {
    for k in 0..3 {
        assert!(
            (got[k] - want[k]).abs() <= TOL,
            "{what}[{k}]: got {} want {} (|d| = {:e})",
            got[k],
            want[k],
            (got[k] - want[k]).abs()
        );
    }
}

#[test]
fn model_tables_sanity() {
    let m = &HUMANOID;
    assert_eq!(m.frames.len(), 30);
    assert_eq!(m.nq, NQ);
    assert_eq!(m.nv(), 35);
    // topological order + exactly one frame per actuated coordinate
    let mut seen = [false; NQ];
    let mut welded = 0usize;
    for (k, fr) in m.frames.iter().enumerate() {
        assert!(fr.parent < k as i32, "frame {k} not topologically ordered");
        if fr.coord >= 0 {
            let c = fr.coord as usize;
            assert!(c < NQ && !seen[c], "coord {c} out of range or duplicated");
            seen[c] = true;
            let n = fr.axis[0] * fr.axis[0] + fr.axis[1] * fr.axis[1] + fr.axis[2] * fr.axis[2];
            assert!((n - 1.0).abs() < 1e-15, "frame {k} axis not unit");
        } else {
            welded += 1;
        }
    }
    assert!(seen.iter().all(|&s| s), "coord map is not a permutation");
    assert_eq!(welded, 1, "head is the only welded frame");
    // §1.2 mass closure (2026-08-18 hands fist-only 0.35, no
    // controller mass): 74.50 kg total
    let total: f64 = m.base_mass + m.frames.iter().map(|f| f.mass).sum::<f64>();
    assert!((total - 74.50).abs() < 1e-9, "total mass {total}");
    for j in 0..NQ {
        let (lo, hi) = m.limits[j];
        assert!(lo < hi, "limits[{j}]");
        assert!(m.omega_cap[j] > 0.0 && m.tau_max[j] > 0.0 && m.damping[j] > 0.0);
    }
}

#[test]
fn neutral_stance() {
    let m = &HUMANOID;
    let st = State::neutral(m);
    let mut ws = Workspace::new(m);
    fk_all(m, &st, &mut ws);
    assert_vec3(ws.p[F_L_SHOULDER], [-0.2, 1.47, -0.65], "shoulder_l");
    assert_vec3(ws.p[F_R_SHOULDER], [0.2, 1.47, -0.65], "shoulder_r");
    assert_vec3(ws.p[F_HEAD], [0.0, 1.70, -0.65], "head");
    assert_vec3(ws.p[F_L_FOOT], [-0.10, 0.07, -0.65], "ankle_l");
    assert_vec3(ws.p[F_R_FOOT], [0.10, 0.07, -0.65], "ankle_r");
    // soles flat on the floor: ankle y 0.07, box center −0.035, half 0.035
    for f in [F_L_FOOT, F_R_FOOT] {
        let sole = ws.p[f][1] + (-0.035) - 0.035;
        assert!(sole.abs() <= TOL, "sole y = {sole:e}");
    }
    // world-aligned frames at neutral (pure translations of world)
    for k in 0..m.frames.len() {
        for i in 0..3 {
            for j in 0..3 {
                let want = if i == j { 1.0 } else { 0.0 };
                assert!((ws.r[k][i][j] - want).abs() <= TOL);
            }
        }
    }
}

/// `rot_to_quat` must be the EXACT inverse of `quat_to_rot` — it is the
/// only step between the FK head frame and the replay/.dat head
/// rotation, so a wrong branch would silently mis-aim the exported
/// first-person camera instead of failing loudly.
///
/// Grid: identity, the three 180° axis flips (θ = π — the trace branch
/// is unusable there and a naive √(1+tr) implementation NaNs), and 500
/// random rotations. Checks the rotation round-trips to 1e-14 and the
/// quaternion returns up to sign (q and −q are one rotation).
#[test]
fn rot_to_quat_inverts_quat_to_rot() {
    use super::fk::{quat_to_rot, rot_to_quat};

    let mut quats: Vec<[f64; 4]> = vec![
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0], // π about x — r00 branch
        [0.0, 0.0, 1.0, 0.0], // π about y — r11 branch
        [0.0, 0.0, 0.0, 1.0], // π about z — r22 branch
    ];
    let mut rng = Rng(0x5EED_F0F0_1234_0002);
    while quats.len() < 504 {
        let mut q = [0.0f64; 4];
        for x in q.iter_mut() {
            *x = rng.range(-1.0, 1.0);
        }
        let n = (q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3]).sqrt();
        if n > 0.1 {
            quats.push([q[0] / n, q[1] / n, q[2] / n, q[3] / n]);
        }
    }

    for (qi, q) in quats.iter().enumerate() {
        let r = quat_to_rot(*q);
        let got = rot_to_quat(&r);
        // unit, and equal to ±q componentwise
        let n = (got[0] * got[0] + got[1] * got[1] + got[2] * got[2] + got[3] * got[3]).sqrt();
        assert!((n - 1.0).abs() < 1e-14, "quat {qi} norm {n}");
        let dot = got[0] * q[0] + got[1] * q[1] + got[2] * q[2] + got[3] * q[3];
        let s = if dot < 0.0 { -1.0 } else { 1.0 };
        for k in 0..4 {
            let d = (s * got[k] - q[k]).abs();
            assert!(d < 1e-14, "quat {qi} component {k}: |d| = {d:e}");
        }
        // and the rotation itself is bit-close after the round trip
        let r2 = quat_to_rot(got);
        for i in 0..3 {
            for j in 0..3 {
                let d = (r2[i][j] - r[i][j]).abs();
                assert!(d < 1e-14, "quat {qi} R[{i}][{j}]: |d| = {d:e}");
            }
        }
    }
}

/// The exported head pose (replay v10 `head` / .dat pose frame): the
/// head is WELDED to the chest, so its world rotation IS the spine
/// stack, and its local +z is the gaze direction down the note track —
/// the same local-+z-is-forward convention the blades use. Neutral must
/// give the (0, 1.70, −0.65) anchor with an identity quaternion, since
/// the .dat exporter writes that quaternion straight through as the
/// viewer's first-person camera rotation.
#[test]
fn head_pose_is_the_spine_pose() {
    use super::fk::{quat_to_rot, rot_to_quat};
    const A_SPINE_YAW: usize = 1; // §1.4 order: pitch, yaw, roll

    let m = &HUMANOID;
    let mut ws = Workspace::new(m);
    let mut st = State::neutral(m);
    fk_all(m, &st, &mut ws);
    assert_vec3(ws.p[F_HEAD], [0.0, 1.70, -0.65], "head");
    let q = rot_to_quat(&ws.r[F_HEAD]);
    for (k, want) in [1.0, 0.0, 0.0, 0.0].iter().enumerate() {
        assert!(
            (q[k] - want).abs() <= TOL,
            "neutral head quat[{k}] = {}",
            q[k]
        );
    }

    // yaw the spine +30°: the gaze (local +z) swings toward +x, and the
    // quaternion is the half-angle rotation about +y
    let a = 30f64.to_radians();
    st.q[A_SPINE_YAW] = a;
    fk_all(m, &st, &mut ws);
    let q = rot_to_quat(&ws.r[F_HEAD]);
    let want = [(a / 2.0).cos(), 0.0, (a / 2.0).sin(), 0.0];
    for k in 0..4 {
        assert!(
            (q[k] - want[k]).abs() <= TOL,
            "yawed head quat[{k}] = {}",
            q[k]
        );
    }
    let gaze = [ws.r[F_HEAD][0][2], ws.r[F_HEAD][1][2], ws.r[F_HEAD][2][2]];
    assert_vec3(gaze, [a.sin(), 0.0, a.cos()], "gaze");
    // and the matrix the viewer would rebuild from the quaternion is
    // the FK matrix itself
    let r2 = quat_to_rot(q);
    for i in 0..3 {
        for j in 0..3 {
            assert!((r2[i][j] - ws.r[F_HEAD][i][j]).abs() <= TOL);
        }
    }
}
