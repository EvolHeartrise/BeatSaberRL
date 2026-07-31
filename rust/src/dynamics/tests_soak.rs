//! P3 soak-regression states: wild random-gear poses harvested from
//! 512-env soak runs that historically blew the solver up (mass-matrix
//! PD panic within a tick). The gate: one control tick from each baked
//! state stays finite and within the kinematic velocity envelope.

use super::model::HUMANOID;
use super::solver::{substep_constrained, ConstraintSet};
use super::step::{H_SUB, N_SUB};
use super::{State, Workspace};

// env 211 @ step 42 of the 2026-07-31 soak (chart seed stream 0, action
// stream 1): two ticks later |w_root| hit 137 rad/s and the Cholesky
// died. Captured one tick before the jump.
const BLOW_P: [f64; 3] = [-0.012004299910065231, 0.11164618873371661, -0.8973459493032614];
const BLOW_QUAT: [f64; 4] =
    [0.7468246421133534, -0.6121551459028927, -0.2344003380922594, 0.11214059380574189];
const BLOW_Q: [f64; 29] = [
    0.12527992223325116, 0.6981317007977246, 0.18282272557633225, 0.39907507813916576,
    0.7853981633974483, 0.523937750268085, 2.449408207539925, 3.1403960466118406,
    -1.2593424712490064, 0.16655055656950501, -0.3533506471398101, -0.5881858025960125,
    -0.6888599839725571, 0.4097626954660411, -2.4267496006266494, 1.425684023622238,
    0.517268347904222, 1.403136050567315, -0.08751475750639288, 0.4077194266823122,
    2.514191767443639, -0.2948879751694364, -0.12237720999663282, 1.0275427662801087,
    -0.040811592893150395, 0.29038644186219875, 2.575676673776639, 0.13793264074355327,
    -0.2829109393113989,
];
const BLOW_V: [f64; 35] = [
    -0.3941135544771008, 0.018174475083022546, -1.4205220159051266, -8.79035036634662,
    -11.196760573210192, 4.866963812003916, 12.583459533245792, 9.06665421888918e-07,
    10.391745543426033, -10.171829887219628, 0.0, -16.0, -2.64760573262783,
    -0.2871758595946998, 35.105254625990156, 25.362144152101713, -7.5411620828773565,
    6.22077598741561, 15.999999999892028, 15.21598808283985, -7.453848789938344,
    13.735903869617088, -1.0181079202375956, -11.60478586054918, -1.4218738167990397,
    15.921647011511649, -4.920916305112038, -4.834582565636065, 7.535745476332602, -16.0,
    -3.8735701594644842, 3.9297828356157005, -2.559709422984333, 0.422403162600755,
    -7.373038439132941,
];
const BLOW_A: [u8; 29] = [
    0, 1, 1, 0, 0, 0, 0, 2, 1, 2, 1, 1, 0, 1, 2, 2, 0, 1, 0, 1, 0, 0, 2, 1, 2, 0, 1, 1, 1,
];

fn blow_state() -> State {
    let m = &HUMANOID;
    let mut st = State::neutral(m);
    st.p = BLOW_P;
    st.quat = BLOW_QUAT;
    st.q.copy_from_slice(&BLOW_Q);
    st.v.copy_from_slice(&BLOW_V);
    st
}

fn gear_tau(a: &[u8; 29]) -> [f64; 29] {
    let signed = [0.0, 1.0, -1.0];
    let mut tau = [0.0; 29];
    for j in 0..29 {
        tau[j] = signed[a[j] as usize] * HUMANOID.tau_max[j];
    }
    tau
}

#[test]
fn soak_state_stays_bounded() {
    let m = &HUMANOID;
    let mut st = blow_state();
    let mut ws = Workspace::new(m);
    let mut cs = ConstraintSet::new(m, true, true);
    let tau = gear_tau(&BLOW_A);
    for tick in 0..3 {
        for sub in 0..N_SUB {
            substep_constrained(m, &mut st, &tau, H_SUB, &mut ws, &mut cs);
            let w = (st.v[3] * st.v[3] + st.v[4] * st.v[4] + st.v[5] * st.v[5]).sqrt();
            let vmax = st.v.iter().fold(0.0f64, |a, &b| a.max(b.abs()));
            let lmax = cs.lambda[..cs.nrows]
                .iter()
                .fold(0.0f64, |a, &b| a.max(b.abs()));
            println!(
                "tick {tick} sub {sub}: |w_root| {w:9.3} vmax {vmax:9.3} \
                 rows {} (con {} lim {}) lam_max {lmax:9.3}",
                cs.nrows,
                cs.contacts.len(),
                cs.limit_rows()
            );
            let nweld = cs.weld_rows();
            for (ci, c) in cs.contacts.iter().enumerate() {
                let row = nweld + ci;
                println!(
                    "    contact pair {:2} gap {:+.5} diag {:.6e} jv_pre {:+9.3} lam {:9.3}",
                    c.pair, c.gap, cs.row_diag(row), cs.jv_pre[row], cs.lambda[row]
                );
            }
            if lmax > 100.0 {
                for row in 0..cs.nrows {
                    println!(
                        "    row {:2} diag {:.6e} bias {:+9.3} jv_pre {:+9.3} lam {:+10.3}",
                        row, cs.row_diag(row), cs.bias[row], cs.jv_pre[row], cs.lambda[row]
                    );
                }
            }
            assert!(st.v.iter().all(|x| x.is_finite()), "non-finite velocity");
            // kinematic envelope: welded feet + capped joints bound the
            // root rotation by the leg-chain rate sum (~6*16) plus slack
            assert!(w < 200.0, "root spin {w} outside the kinematic envelope");
        }
    }
}
