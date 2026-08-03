//! P3 soak-regression states: wild random-gear poses harvested from
//! 512-env soak runs that historically blew the solver up (mass-matrix
//! PD panic within a tick). The gate: one control tick from each baked
//! state stays finite and within the kinematic velocity envelope.

use super::model::HUMANOID;
use super::solver::{substep_constrained, ConstraintSet};
use super::step::{H_SUB, N_SUB, ROOT_V_MAX, ROOT_W_MAX};
use super::testutil::Rng;
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

// ── §23 root-escape regression (A1: the envelope applied BEFORE the
//    step-10 integration) ───────────────────────────────────────────────
//
// Escape #7 of the 2026-08-03 corpus (`p1_escapes.npz`, CUDA f32 env,
// N=4096 seed 1 gears 3, env 48 @ tick 17): the state at the START of a
// control tick during which the root moved 642.7 m without an auto-reset.
// It is an ORDINARY-LOOKING state — |v_root| 0.15 m/s, |w_root| 6.4 rad/s,
// |qd|max at the 40 rad/s wrist cap — whose step-8 PGS diverges
// geometrically (Divergences §23) and hands step 10 a root velocity of
// order 1e5. Pre-A1 the Rust f64 tier reproduces the teleport: 31.12 m
// in substep 0 ALONE, |p| = 31.40 m. Post-A1 every substep's |dp| is
// bounded by h*ROOT_V_MAX = 0.2 m (substep 0 attains it exactly) and the
// body is back inside 1.5 m of the origin by the end of the tick.
pub(crate) const ESC_P: [f64; 3] = [-0.18843580782413483, 0.8035035133361816, -1.077133059501648];
pub(crate) const ESC_QUAT: [f64; 4] = [
    0.8241789937019348, 0.5093252062797546, 0.09989655017852783, -0.2265780121088028,
];
pub(crate) const ESC_Q: [f64; 29] = [
    0.7853981852531433, -0.6981030106544495, -0.47437214851379395, 0.9689304828643799,
    -0.1498527228832245, 0.566002607345581, 0.8901060819625854, 0.07968739420175552,
    1.0873289108276367, 0.2617993950843811, 0.14637738466262817, 1.3688384294509888,
    0.45331719517707825, 0.8741688132286072, -0.5112427473068237, -0.5565317869186401,
    0.33905500173568726, 1.5271868705749512, -0.5984416604042053, 0.294033020734787,
    0.005854796152561903, -0.4577142298221588, -0.22766704857349396, 1.6427476406097412,
    -0.6469371318817139, 0.26761528849601746, 0.0, -0.5667280554771423,
    -0.2737051248550415,
];
pub(crate) const ESC_V: [f64; 35] = [
    0.012548212893307209, 0.0441480427980423, -0.1442444622516632, -5.172290325164795,
    -1.5389772653579712, 3.447293281555176, 0.0, -10.52071762084961, -4.945155620574951,
    -10.435583114624023, 14.81192398071289, 6.78498649597168, 15.845629692077637, -40.0,
    25.100996017456055, 0.0, -16.0, 10.470308303833008, -15.397571563720703,
    4.1626200675964355, -5.062375545501709, -18.275327682495117, 39.9994010925293,
    -14.463310241699219, 5.370003700256348, 5.604532241821289, -15.959772109985352,
    -8.515240669250488, -0.1936049610376358, -6.113821506500244, 3.4726436138153076,
    0.6403346657752991, 0.0, 0.18586382269859314, 0.2472548931837082,
];
pub(crate) const ESC_A: [u8; 29] = [
    0, 2, 0, 1, 1, 0, 2, 1, 2, 0, 1, 0, 0, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 2, 0, 1, 1, 0, 0,
];

pub(crate) fn escape_state() -> State {
    let m = &HUMANOID;
    let mut st = State::neutral(m);
    st.p = ESC_P;
    st.quat = ESC_QUAT;
    st.q.copy_from_slice(&ESC_Q);
    st.v.copy_from_slice(&ESC_V);
    st
}

/// A1's central guarantee: the root position increment of a substep is
/// bounded by the envelope times h, UNCONDITIONALLY — even when the
/// step-8 solve diverges. Also pins the escape SIGNAL: the
/// pre-integration envelope must engage on this state (which is what the
/// env layer's C backstop terminates on).
#[test]
fn root_step_bounded_on_the_escape_state() {
    let m = &HUMANOID;
    let mut st = escape_state();
    let mut ws = Workspace::new(m);
    let mut cs = ConstraintSet::new(m, true, true);
    let tau = gear_tau(&ESC_A);
    // h * ROOT_V_MAX exactly; the rescale lands the root ON the cap, so
    // the bound is attained, not approached
    let bound = H_SUB * ROOT_V_MAX;
    let mut maxdp = 0.0f64;
    for sub in 0..N_SUB {
        let p0 = st.p;
        substep_constrained(m, &mut st, &tau, H_SUB, &mut ws, &mut cs);
        let dp = ((st.p[0] - p0[0]).powi(2)
            + (st.p[1] - p0[1]).powi(2)
            + (st.p[2] - p0[2]).powi(2))
        .sqrt();
        maxdp = maxdp.max(dp);
        println!(
            "sub {sub}: |dp| {dp:.6} m  |p| {:.4}  env_hits {}",
            (st.p[0] * st.p[0] + st.p[1] * st.p[1] + st.p[2] * st.p[2]).sqrt(),
            ws.env_hits
        );
        assert!(st.p.iter().all(|x| x.is_finite()), "non-finite position");
        assert!(st.v.iter().all(|x| x.is_finite()), "non-finite velocity");
        assert!(
            dp <= bound * (1.0 + 1e-12),
            "substep {sub} integrated |dp| = {dp} m, above the h*ROOT_V_MAX \
             = {bound} m guarantee (A1 regression)"
        );
    }
    assert!(maxdp > 0.19, "this state must actually engage the envelope");
    assert!(
        ws.env_hits > 0,
        "the pre-integration envelope must have engaged on the escape state"
    );
    // and the body stays in the room instead of leaving at 640 m/tick
    let pn = (st.p[0] * st.p[0] + st.p[1] * st.p[1] + st.p[2] * st.p[2]).sqrt();
    assert!(pn < 3.0, "|p| = {pn} m after one tick — the escape survived");
}

/// The counter is a SIGNAL, not noise: healthy motion must never engage
/// the pre-integration envelope, so the env layer's C backstop can never
/// fire on it. Gentle seeded gears (0.15 × tau_max, redrawn per control
/// tick) from the neutral pose — the mirror's "gentle" bitwise suite —
/// over 2 s of sim time.
#[test]
fn healthy_motion_never_engages_the_envelope() {
    let m = &HUMANOID;
    let mut st = State::neutral(m);
    let mut ws = Workspace::new(m);
    let mut cs = ConstraintSet::new(m, true, true);
    let mut rng = Rng(0x51E5_0000);
    let mut tau = [0.0f64; 29];
    for k in 0..(60 * N_SUB) {
        if k % N_SUB == 0 {
            for j in 0..m.nq {
                tau[j] = rng.range(-0.15, 0.15) * m.tau_max[j];
            }
        }
        substep_constrained(m, &mut st, &tau, H_SUB, &mut ws, &mut cs);
    }
    assert_eq!(
        ws.env_hits, 0,
        "gentle healthy motion must not trip the pre-integration envelope"
    );
}

/// The §19b soak state is itself an ESCAPE state — it was harvested one
/// tick before a Cholesky PD panic — so the pre-integration envelope
/// SHOULD engage on it, and A1 must hold the Δp bound while it does.
/// (Pinning this both ways is what makes the counter a usable signal:
/// it fires on the states §19b was added for and on nothing healthy.)
#[test]
fn soak_state_engages_the_envelope_and_stays_bounded() {
    let m = &HUMANOID;
    let mut st = blow_state();
    let mut ws = Workspace::new(m);
    let mut cs = ConstraintSet::new(m, true, true);
    let tau = gear_tau(&BLOW_A);
    let bound = H_SUB * ROOT_V_MAX;
    for _ in 0..3 * N_SUB {
        let p0 = st.p;
        substep_constrained(m, &mut st, &tau, H_SUB, &mut ws, &mut cs);
        let dp = ((st.p[0] - p0[0]).powi(2)
            + (st.p[1] - p0[1]).powi(2)
            + (st.p[2] - p0[2]).powi(2))
        .sqrt();
        assert!(dp <= bound * (1.0 + 1e-12), "|dp| {dp} above the bound");
    }
    println!("§19b soak state: {} envelope engagement(s)", ws.env_hits);
    assert!(ws.env_hits > 0);
}

/// The `p2_adversarial.py` generator, ported: states far worse than the
/// env can reach — 60 % of joints parked exactly ON a stop, every joint
/// rate saturated at its cap, the root at the envelope, a uniformly
/// random root orientation, and the foot welds pre-violated by up to
/// 0.10 m. Pre-A1 this class goes NON-FINITE in f32 and reaches |dp| =
/// 3.0e10 m per substep in f64; post-A1 every substep is finite and
/// inside the h*ROOT_V_MAX bound.
#[test]
fn adversarial_states_stay_finite_and_bounded() {
    let m = &HUMANOID;
    let bound = H_SUB * ROOT_V_MAX;
    let mut worst = 0.0f64;
    let mut engaged = 0u64;
    for state_i in 0..192u64 {
        let mut rng = Rng(0xA0A0_0000 + state_i);
        let mut st = State::neutral(m);
        // root pose: uniform random orientation, welds violated <= 0.10 m
        for i in 0..3 {
            st.p[i] = m.base_pos0[i] + rng.range(-0.10, 0.10);
        }
        loop {
            let q = [
                rng.range(-1.0, 1.0),
                rng.range(-1.0, 1.0),
                rng.range(-1.0, 1.0),
                rng.range(-1.0, 1.0),
            ];
            let n = (q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3]).sqrt();
            if n > 0.1 {
                st.quat = [q[0] / n, q[1] / n, q[2] / n, q[3] / n];
                break;
            }
        }
        // 60 % of joints exactly on a stop, the rest uniform in range;
        // every joint rate saturated at +-omega_cap
        for j in 0..m.nq {
            let (lo, hi) = m.limits[j];
            st.q[j] = if rng.uniform() < 0.6 {
                if rng.uniform() < 0.5 {
                    lo
                } else {
                    hi
                }
            } else {
                rng.range(lo, hi)
            };
            st.v[6 + j] =
                if rng.uniform() < 0.5 { -1.0 } else { 1.0 } * m.omega_cap[j];
        }
        // root at (a random fraction of) the envelope
        let mut w = [
            rng.range(-1.0, 1.0),
            rng.range(-1.0, 1.0),
            rng.range(-1.0, 1.0),
        ];
        let wn = (w[0] * w[0] + w[1] * w[1] + w[2] * w[2]).sqrt().max(1e-9);
        let wsc = ROOT_W_MAX * rng.uniform() / wn;
        for i in 0..3 {
            w[i] *= wsc;
            st.v[3 + i] = w[i];
        }
        let mut l = [
            rng.range(-1.0, 1.0),
            rng.range(-1.0, 1.0),
            rng.range(-1.0, 1.0),
        ];
        let ln = (l[0] * l[0] + l[1] * l[1] + l[2] * l[2]).sqrt().max(1e-9);
        let lsc = ROOT_V_MAX * rng.uniform() / ln;
        for i in 0..3 {
            l[i] *= lsc;
            st.v[i] = l[i];
        }
        let mut tau = [0.0f64; 29];
        let mut ws2 = Workspace::new(m);
        let mut cs = ConstraintSet::new(m, true, true);
        for k in 0..16 {
            if k % N_SUB == 0 {
                for j in 0..m.nq {
                    let s = match rng.next_u64() % 3 {
                        0 => 0.0,
                        1 => 1.0,
                        _ => -1.0,
                    };
                    tau[j] = s * m.tau_max[j];
                }
            }
            let p0 = st.p;
            substep_constrained(m, &mut st, &tau, H_SUB, &mut ws2, &mut cs);
            let dp = ((st.p[0] - p0[0]).powi(2)
                + (st.p[1] - p0[1]).powi(2)
                + (st.p[2] - p0[2]).powi(2))
            .sqrt();
            worst = worst.max(dp);
            assert!(
                st.p.iter().all(|x| x.is_finite())
                    && st.quat.iter().all(|x| x.is_finite())
                    && st.v.iter().all(|x| x.is_finite())
                    && st.q.iter().all(|x| x.is_finite()),
                "state {state_i} substep {k}: non-finite state"
            );
            assert!(
                dp <= bound * (1.0 + 1e-12),
                "state {state_i} substep {k}: |dp| = {dp} m above the \
                 h*ROOT_V_MAX = {bound} m guarantee"
            );
        }
        engaged += ws2.env_hits;
    }
    println!(
        "adversarial: 192 states x 16 substeps, worst |dp| {worst:.6} m \
         (bound {bound:.6}), envelope engagements {engaged}"
    );
    assert!(
        engaged > 0,
        "the adversarial set must actually exercise the envelope"
    );
}
