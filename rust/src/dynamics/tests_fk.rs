//! P1 FK gates: model-table sanity, the neutral-stance anchors
//! (SPEC-v10 §1.1), and the v9 continuity theorem (§1.3) — v10 tree FK
//! with the root at reference + spine zero must reproduce the v9
//! `crate::fk()` wrist/elbow/blade to ≤ 1e-12 per component. v9
//! behavior is the ground truth; the spec is its description.

use super::fk::fk_all;
use super::model::{
    HUMANOID, A_L_ARM, A_R_ARM, F_HEAD, F_L_ELBOW, F_L_FOOT, F_L_HAND, F_L_SHOULDER,
    F_L_WRIST, F_R_ELBOW, F_R_FOOT, F_R_HAND, F_R_SHOULDER, F_R_WRIST, NQ,
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
    // §1.2 mass closure: 74.72 body + 2 × 0.15 controller = 75.02
    let total: f64 = m.base_mass + m.frames.iter().map(|f| f.mass).sum::<f64>();
    assert!((total - 75.02).abs() < 1e-9, "total mass {total}");
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

/// The load-bearing P1 gate: v10 FK vs the v9 `fk()` on the arm chains.
/// Grid: all-zero, every joint at each limit (per hand), 100 random
/// in-limit poses (fixed seed). If this fails the bug may be in
/// SPEC-v10's stacking-order spec rather than the code — v9 wins.
#[test]
fn v9_fk_continuity() {
    let m = &HUMANOID;
    let mut ws = Workspace::new(m);
    let mut poses: Vec<[f64; 2 * 7]> = Vec::new(); // [L q7, R q7]

    poses.push([0.0; 14]);
    for hand in 0..2 {
        let base = if hand == 0 { A_L_ARM } else { A_R_ARM };
        for j in 0..7 {
            for lim in 0..2 {
                let mut p = [0.0; 14];
                let (lo, hi) = m.limits[base + j];
                p[hand * 7 + j] = if lim == 0 { lo } else { hi };
                poses.push(p);
            }
        }
    }
    let mut rng = Rng(0x5EED_F0F0_1234_0001);
    for _ in 0..100 {
        let mut p = [0.0; 14];
        for hand in 0..2 {
            let base = if hand == 0 { A_L_ARM } else { A_R_ARM };
            for j in 0..7 {
                let (lo, hi) = m.limits[base + j];
                p[hand * 7 + j] = rng.range(lo, hi);
            }
        }
        poses.push(p);
    }

    let mut st = State::neutral(m);
    for (pi, pose) in poses.iter().enumerate() {
        for j in 0..7 {
            st.q[A_L_ARM + j] = pose[j];
            st.q[A_R_ARM + j] = pose[7 + j];
        }
        fk_all(m, &st, &mut ws);
        for hand in 0..2 {
            let q7: [f64; 7] = pose[hand * 7..hand * 7 + 7].try_into().unwrap();
            let (e9, w9, b9, _) = crate::fk(hand, &q7);
            let (fe, fw, fh) = if hand == 0 {
                (F_L_ELBOW, F_L_WRIST, F_L_HAND)
            } else {
                (F_R_ELBOW, F_R_WRIST, F_R_HAND)
            };
            let blade = [ws.r[fh][0][2], ws.r[fh][1][2], ws.r[fh][2][2]];
            let name = format!("pose {pi} hand {hand}");
            assert_vec3(ws.p[fe], e9, &format!("{name} elbow"));
            assert_vec3(ws.p[fw], w9, &format!("{name} wrist"));
            assert_vec3(blade, b9, &format!("{name} blade"));
        }
    }
}
