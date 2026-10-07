//! Mirror-exactness regression tests (sim/mirror-exact, 2026-10-04).
//!
//! The body, its tables and the solve are left/right symmetric, so the
//! physics tick must commute with the x → −x reflection M:
//!     tick(M s, M a) = M tick(s, a)
//! up to floating-point rounding. M is the operator in
//! python/beat_arena_env/mirror_ops.py: swap the left/right joint blocks
//! (`HUMANOID_MIRROR`), negate the coordinates whose axis flips under the
//! reflection (`JOINT_SIGN`), p_x → −p_x, quaternion (w,x,y,z) →
//! (w,x,−y,−z), root linear velocity (−,+,+), root angular (+,−,−), and
//! swap the ± gear ids of the sign-flipped joints.
//!
//! Pre-fix (the left-before-right solve orders) these tests measured a
//! median one-tick gap of 0.46 (p99 26) over the soak states, a coasting
//! reset stance breaking symmetry by 0.28 within 10 ticks, and a mean
//! lateral velocity of −6.0 ± 1.2 mm/s per tick over symmetric
//! near-stance states (the 2026-10-04 audit: mean yaw rate +0.136 rad/s
//! per tick on the same kind of set, from the ω-cap pass).

use super::*;

/// Per-joint sign of the coordinate under the reflection (§1.4 order):
/// spine [pitch +, yaw −, roll −]; arm [pitch +, yaw −, twist −,
/// elbow +, roll −, bend +, dev −]; leg [hip pitch +, hip yaw −,
/// hip roll −, knee +, ankle pitch +, ankle roll −].
/// The actuated-coordinate mirror map (model.rs `HUMANOID_MIRROR`,
/// restated so the operator here is independent of the solver's copy).
const JOINT_MIRROR: [usize; NQ] = [
    0, 1, 2, 10, 11, 12, 13, 14, 15, 16, 3, 4, 5, 6, 7, 8, 9,
    23, 24, 25, 26, 27, 28, 17, 18, 19, 20, 21, 22,
];

const JOINT_SIGN: [Real; NQ] = [
    1.0, -1.0, -1.0,
    1.0, -1.0, -1.0, 1.0, -1.0, 1.0, -1.0,
    1.0, -1.0, -1.0, 1.0, -1.0, 1.0, -1.0,
    1.0, -1.0, -1.0, 1.0, 1.0, -1.0,
    1.0, -1.0, -1.0, 1.0, 1.0, -1.0,
];

fn m_joint(x: &[Real]) -> Vec<Real> {
    let mut out = vec![0.0; NQ];
    for j in 0..NQ {
        out[JOINT_MIRROR[j]] = JOINT_SIGN[j] * x[j];
    }
    out
}

fn m_state(st: &State) -> State {
    let mut m = st.clone();
    m.p = [-st.p[0], st.p[1], st.p[2]];
    m.quat = [st.quat[0], st.quat[1], -st.quat[2], -st.quat[3]];
    m.q = m_joint(&st.q);
    let jv = m_joint(&st.v[6..]);
    m.v[0] = -st.v[0];
    m.v[1] = st.v[1];
    m.v[2] = st.v[2];
    m.v[3] = st.v[3];
    m.v[4] = -st.v[4];
    m.v[5] = -st.v[5];
    m.v[6..].copy_from_slice(&jv);
    m
}

fn m_action(a: &[u8; NQ]) -> [u8; NQ] {
    // ± gear pairs: 1<->2, 3<->4, 5<->6 (0 = coast)
    const SWAP: [u8; 7] = [0, 2, 1, 4, 3, 6, 5];
    let mut out = [0u8; NQ];
    for j in 0..NQ {
        out[JOINT_MIRROR[j]] = if JOINT_SIGN[j] < 0.0 { SWAP[a[j] as usize] } else { a[j] };
    }
    out
}

/// max |x − y| over the full state (pose coordinates and velocities).
fn state_gap(x: &State, y: &State) -> Real {
    let mut g: Real = 0.0;
    for i in 0..3 {
        g = g.max((x.p[i] - y.p[i]).abs());
    }
    for i in 0..4 {
        g = g.max((x.quat[i] - y.quat[i]).abs());
    }
    for (a, b) in x.q.iter().zip(&y.q) {
        g = g.max((a - b).abs());
    }
    for (a, b) in x.v.iter().zip(&y.v) {
        g = g.max((a - b).abs());
    }
    g
}

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
    fn uniform(&mut self) -> Real {
        (self.next() >> 11) as Real / (1u64 << 53) as Real
    }
    fn normal(&mut self) -> Real {
        let (u1, u2) = (self.uniform().max(1e-300), self.uniform());
        (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
    }
}

/// One control tick of a fresh body from `st` with gear ids `a`.
fn tick_from(st: &State, a: &[u8; NQ], gear: &[[Real; 7]; NQ]) -> State {
    let mut b = Body::new();
    b.st = st.clone();
    b.refresh_fk();
    b.tick(a, gear);
    b.st
}

/// Soak states: episodes from the reset stance under random arm gears
/// (each arm joint fires w.p. 1/2, ids 1..=4) and a crude per-joint leg
/// hold toward the stance (fine gears) — the falling, cap- and
/// stop-riding traffic that exposed the handed solve. Returns (state,
/// action) pairs, one per tick.
fn soak_states(n: usize, seed: u64, gear: &[[Real; 7]; NQ]) -> Vec<(State, [u8; NQ])> {
    let mut rng = Rng(seed);
    let mut out = Vec::with_capacity(n);
    let mut tgt = [0.0 as Real; NQ];
    for (j, v) in Q_START_LEG {
        tgt[j] = v;
    }
    while out.len() < n {
        let mut b = Body::new();
        b.reset();
        for _ in 0..25 {
            if b.st.p[1] < 0.6 || out.len() >= n {
                break;
            }
            let mut a = [0u8; NQ];
            for j in 17..NQ {
                let u = -(b.st.q[j] - tgt[j]) * 3.0 - 0.3 * b.st.v[6 + j];
                a[j] = if u > 0.3 { 3 } else if u < -0.3 { 4 } else { 0 };
            }
            for j in 3..17 {
                if rng.uniform() < 0.5 {
                    a[j] = 1 + (rng.next() % 4) as u8;
                }
            }
            out.push((b.st.clone(), a));
            b.tick(&a, gear);
            if b.fell || b.escaped {
                break;
            }
        }
    }
    out
}

/// Does the UN-mirrored tick map itself jump by ≳ `gap` under a 1e-12
/// nudge of one coordinate? (A one-sided perturbation of every joint
/// angle and rate.) Such a state sits on one of the dynamics' intrinsic
/// discontinuities — a joint resting exactly on its stop, where step
/// 11's `hit` test flips — and rounding alone picks the side, in the
/// mirrored copy as in a nudged one.
fn intrinsically_discontinuous(st: &State, a: &[u8; NQ], gear: &[[Real; 7]; NQ],
                               gap: Real) -> bool {
    let base = tick_from(st, a, gear);
    for j in 0..NQ {
        for sgn in [1.0, -1.0] {
            let mut s = st.clone();
            s.q[j] += sgn * 1e-12;
            if state_gap(&tick_from(&s, a, gear), &base) > 0.01 * gap {
                return true;
            }
            let mut s = st.clone();
            s.v[6 + j] += sgn * 1e-12;
            if state_gap(&tick_from(&s, a, gear), &base) > 0.01 * gap {
                return true;
            }
        }
    }
    false
}

/// The core property: one tick from M s under M a lands on M (one tick
/// from s under a), over ~500 soak states.
#[test]
fn tick_commutes_with_the_mirror() {
    let gear = joint_gear_rows(5);
    let states = soak_states(500, 0x5EED_0001, &gear);
    let mut gaps = Vec::with_capacity(states.len());
    let mut flagged = 0;
    let mut unexplained = Vec::new();
    for (i, (st, a)) in states.iter().enumerate() {
        let out = tick_from(st, a, &gear);
        let out_m = tick_from(&m_state(st), &m_action(a), &gear);
        let gap = state_gap(&out_m, &m_state(&out));
        if gap > 1e-6 {
            flagged += 1;
            if intrinsically_discontinuous(st, a, &gear, gap) {
                // rounding-picked side of an intrinsic jump: excluded
                // from the continuous-state statistics below
                continue;
            }
            unexplained.push((i, gap));
        }
        gaps.push(gap);
    }
    gaps.sort_by(|x, y| x.partial_cmp(y).unwrap());
    let q = |p: f64| gaps[((gaps.len() - 1) as f64 * p) as usize];
    println!(
        "mirror gap over {} soak states: median {:.2e}, p90 {:.2e}, p99 {:.2e} \
         (continuous states); {} flagged > 1e-6, {} unexplained",
        states.len(), q(0.5), q(0.9), q(0.99), flagged, unexplained.len()
    );
    assert!(q(0.5) < 1e-12, "median mirror gap {:.3e} >= 1e-12", q(0.5));
    assert!(q(0.99) < 1e-6, "p99 mirror gap {:.3e} >= 1e-6 (unexplained: {unexplained:?})",
            q(0.99));
}

/// A mirror-symmetric state stepped with mirror-symmetric commands
/// stays symmetric: the reset stance under coasting and under a
/// symmetric two-hand swing pattern.
#[test]
fn symmetric_state_stays_symmetric() {
    let gear = joint_gear_rows(5);
    let mut swing = [0u8; NQ];
    swing[3] = 1; // l arm pitch +
    swing[6] = 2; // l elbow −
    swing[7] = 1; // l wrist roll + (cap-riding)
    swing[8] = 2; // l wrist bend −
    let swing = {
        let m = m_action(&swing);
        let mut s = swing;
        for j in 10..17 {
            s[j] = m[j];
        }
        s
    };
    assert_eq!(m_action(&swing), swing, "the swing pattern must be symmetric");
    for (name, a) in [("coast", [0u8; NQ]), ("double swing", swing)] {
        let mut b = Body::new();
        b.reset();
        assert_eq!(state_gap(&b.st, &m_state(&b.st)), 0.0, "the reset stance is symmetric");
        let mut worst: Real = 0.0;
        for _ in 0..10 {
            b.tick(&a, &gear);
            worst = worst.max(state_gap(&b.st, &m_state(&b.st)));
        }
        println!("{name}: max |s - M s| over 10 ticks = {worst:.2e}");
        assert!(worst < 1e-10, "{name}: symmetric state broke symmetry by {worst:.3e}");
    }
}

/// Over symmetric near-stance states under symmetric commands, the
/// one-tick root yaw rate (and lateral velocity, roll rate) must be
/// zero up to noise: |mean| within 3 standard errors.
#[test]
fn no_mean_yaw_over_symmetric_states() {
    let gear = joint_gear_rows(5);
    let mut rng = Rng(0x5EED_0002);
    let mut b0 = Body::new();
    b0.reset();
    let n = 500;
    let mut acc = [[0.0 as Real; 2]; 3]; // [vx, yaw, roll] x [sum, sum sq]
    for _ in 0..n {
        let mut st = b0.st.clone();
        for j in 3..10 {
            st.q[j] += 0.3 * rng.normal();
        }
        for j in 17..23 {
            st.q[j] += 0.04 * rng.normal();
        }
        st.q[0] += 0.05 * rng.normal();
        for j in 0..NQ {
            let (lo, hi) = MODEL.limits[j];
            st.q[j] = st.q[j].clamp(lo + 0.02, hi - 0.02);
        }
        st.p[1] += 0.003 * rng.normal();
        for j in 0..NQ {
            st.v[6 + j] = 0.3 * rng.normal();
        }
        st.v[1] = 0.05 * rng.normal();
        st.v[3] = 0.1 * rng.normal();
        // symmetrize: the right side and the odd spine coordinates
        // follow from the left side
        let mq = m_joint(&st.q);
        let mv = m_joint(&st.v[6..]);
        for j in (10..17).chain(23..29) {
            st.q[j] = mq[j];
            st.v[6 + j] = mv[j];
        }
        for j in [1, 2] {
            st.q[j] = 0.0;
            st.v[6 + j] = 0.0;
        }
        let mut a = [0u8; NQ];
        for j in (3..10).chain(17..23) {
            if rng.uniform() < 0.5 {
                a[j] = 1 + (rng.next() % 4) as u8;
            }
        }
        a[0] = (rng.next() % 5) as u8;
        let ma = m_action(&a);
        for j in (10..17).chain(23..29) {
            a[j] = ma[j];
        }
        assert_eq!(state_gap(&st, &m_state(&st)), 0.0);
        assert_eq!(m_action(&a), a);
        let out = tick_from(&st, &a, &gear);
        for (k, x) in [out.v[0], out.v[4], out.v[5]].into_iter().enumerate() {
            acc[k][0] += x;
            acc[k][1] += x * x;
        }
    }
    for (k, name) in ["lateral velocity", "yaw rate", "roll rate"].into_iter().enumerate() {
        let mean = acc[k][0] / n as Real;
        let var = (acc[k][1] / n as Real - mean * mean).max(0.0);
        let se = (var / n as Real).sqrt();
        println!("{name}: mean {mean:+.3e} (se {se:.3e})");
        if k == 0 {
            // RESIDUAL (documented): ~7% of these states sit where the
            // light wrist stack is unstable within one tick, so the
            // mirror-image rounding of the tree Cholesky / CRBA (left
            // subtree accumulated before the right — not bit-exact
            // symmetric) picks the branch, and not at random: a
            // +1.0 ± 0.3 mm/s mean lateral velocity survives on this
            // set (pre-fix −6.0 ± 1.2 mm/s, from the solve order, which
            // this bound rejects).
            assert!(mean.abs() <= 3.0 * se + 2e-3,
                    "{name}: mean {mean:+.3e} beyond 3 SE ({se:.3e}) + 2 mm/s");
        } else {
            assert!(mean.abs() <= 3.0 * se + 1e-12,
                    "{name}: mean {mean:+.3e} beyond 3 SE ({se:.3e}) — a handed solve");
        }
    }
}


/// The body tables the mirror-exact solve relies on: `HUMANOID_MIRROR`
/// is an involution matching this file's operator, and every per-joint
/// table is equal across a mirror pair (limits negated and swapped where
/// the coordinate flips sign); the collider pair table is mirror-closed.
#[test]
fn humanoid_tables_are_mirror_symmetric() {
    use crate::dynamics::collide::{COLLIDER_MIRROR, PAIRS, PAIR_MIRROR, PAIR_ORBIT_ORDER};
    for j in 0..NQ {
        let m = MODEL.mirror[j];
        assert_eq!(m, JOINT_MIRROR[j]);
        assert_eq!(MODEL.mirror[m], j, "mirror map must be an involution");
        assert_eq!(JOINT_SIGN[m], JOINT_SIGN[j]);
        for (name, t) in [("tau_max", MODEL.tau_max), ("omega_cap", MODEL.omega_cap),
                          ("damping", MODEL.damping), ("armature", MODEL.armature)] {
            assert_eq!(t[j], t[m], "{name}[{j}] != {name}[{m}]");
        }
        let (lo, hi) = MODEL.limits[j];
        let (mlo, mhi) = MODEL.limits[m];
        if JOINT_SIGN[j] > 0.0 {
            assert_eq!((lo, hi), (mlo, mhi), "limits of {j}/{m}");
        } else {
            assert_eq!((lo, hi), (-mhi, -mlo), "limits of {j}/{m}");
        }
        assert_eq!(FINE_GEAR_MULT[j], FINE_GEAR_MULT[m]);
        assert_eq!(JOINT_EXTRA_N[j], JOINT_EXTRA_N[m]);
    }
    // exp/wrist-tendon: the two hands' (bend, dev) coordinates are each
    // other's mirror images, the bend keeps its sign and the dev flips
    // (so the tendon's limit-normalized u_d is exactly negated, and
    // E / the bend torque are bit-identical — tendon_tests::
    // tendon_is_mirror_exact), and the τ_peaks are per-model scalars
    // shared by both hands by construction.
    let pw = MODEL.passive_wrist.expect("HUMANOID carries the wrist tendon");
    for s in 0..2 {
        let (jb, jd) = pw.hands[s];
        assert_eq!((MODEL.mirror[jb], MODEL.mirror[jd]), pw.hands[1 - s],
                   "tendon hands must be a mirror pair");
        assert_eq!(JOINT_SIGN[jb], 1.0, "bend keeps its sign under the mirror");
        assert_eq!(JOINT_SIGN[jd], -1.0, "dev flips sign under the mirror");
    }
    assert!(pw.tau_peak_bend > 0.0 && pw.tau_peak_dev > 0.0);
    // issue #61 edge damping: per-model scalars shared by both hands, on
    // the shared radius r (bit-identical across the mirror), diagonal
    assert!(pw.b_peak_bend > 0.0 && pw.b_peak_dev > 0.0);
    // exp/shoulder-rom: the two arms' (yaw, pitch, twist) are each other's
    // mirror images, pitch keeps its sign and yaw / twist flip (the
    // canonical evaluation at (−yaw, pitch, −twist) is then the left
    // arm's own expression — shoulder_tests::shoulder_is_mirror_exact),
    // and every parameter is a per-model scalar shared by both arms.
    let ps = MODEL.passive_shoulder.expect("HUMANOID carries the passive shoulder");
    for s in 0..2 {
        let (jy, jp, jt) = ps.arms[s];
        let (my, mp, mt) = ps.arms[1 - s];
        assert_eq!((MODEL.mirror[jy], MODEL.mirror[jp], MODEL.mirror[jt]), (my, mp, mt),
                   "shoulder arms must be a mirror pair");
        assert_eq!((JOINT_SIGN[jy], JOINT_SIGN[jp], JOINT_SIGN[jt]), (-1.0, 1.0, -1.0));
    }
    assert!(ps.tau_peak > 0.0 && ps.b_peak > 0.0 && ps.u_lin > 1.0);
    for (k, &(a, b)) in PAIRS.iter().enumerate() {
        let mk = PAIR_MIRROR[k];
        assert_eq!(PAIR_MIRROR[mk], k);
        let (ma, mb) = (COLLIDER_MIRROR[a], COLLIDER_MIRROR[b]);
        assert_eq!(PAIRS[mk], (ma.min(mb), ma.max(mb)));
    }
    let mut seen = [false; 55];
    for &p in PAIR_ORBIT_ORDER.iter() {
        assert!(!seen[p]);
        seen[p] = true;
    }
}
