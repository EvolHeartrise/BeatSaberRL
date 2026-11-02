//! P1 analytic gates on tiny ad-hoc model tables run through the SAME
//! machinery (proving the core is generic over the table): a 1-link
//! and a 2-link point-mass pendulum. References: the closed-form
//! small-oscillation period, and hand-derived ODEs integrated by RK4
//! at tiny step (the engine's semi-implicit Euler must converge O(h)
//! to them — halve h, error halves).

use super::algo::energy;
use super::fk::fk_all;
use super::model::{Frame, Model};
use super::step::substep;
use super::{State, Workspace};

const G: f64 = 9.81;
const INF: f64 = f64::INFINITY;

// 1-link: unit point mass on a massless 1 m rod, revolute about z at
// the origin, hanging along −y at q = 0. I_pivot = m·l² = 1.
static P1_FRAMES: [Frame; 1] = [Frame {
    parent: -1,
    anchor: [0.0, 0.0, 0.0],
    axis: [0.0, 0.0, 1.0],
    coord: 0,
    mass: 1.0,
    com: [0.0, -1.0, 0.0],
    inertia: [0.0, 0.0, 0.0],
}];
static P1_LIMITS: [(f64, f64); 1] = [(-INF, INF)];
static P1_ONE: [f64; 1] = [1.0];
static P1_INF: [f64; 1] = [INF];
static P1_ZERO: [f64; 1] = [0.0];
static PEND1: Model = Model {
    free_base: false,
    base_mass: 0.0,
    base_com: [0.0; 3],
    base_inertia: [0.0; 3],
    base_pos0: [0.0; 3],
    frames: &P1_FRAMES,
    nq: 1,
    limits: &P1_LIMITS,
    omega_cap: &P1_INF,
    tau_max: &P1_ONE,
    damping: &P1_ZERO,
    armature: &P1_ZERO,
    gravity: [0.0, -G, 0.0],
    mirror: &[0],
    passive_wrist: None,
    passive_shoulder: None,
};

// 2-link: two unit point masses, 1 m links, both revolute about z;
// q[1] is the RELATIVE elbow angle (absolute φ2 = q0 + q1).
static P2_FRAMES: [Frame; 2] = [
    Frame {
        parent: -1,
        anchor: [0.0, 0.0, 0.0],
        axis: [0.0, 0.0, 1.0],
        coord: 0,
        mass: 1.0,
        com: [0.0, -1.0, 0.0],
        inertia: [0.0, 0.0, 0.0],
    },
    Frame {
        parent: 0,
        anchor: [0.0, -1.0, 0.0],
        axis: [0.0, 0.0, 1.0],
        coord: 1,
        mass: 1.0,
        com: [0.0, -1.0, 0.0],
        inertia: [0.0, 0.0, 0.0],
    },
];
static P2_LIMITS: [(f64, f64); 2] = [(-INF, INF), (-INF, INF)];
static P2_ONE: [f64; 2] = [1.0, 1.0];
static P2_INF: [f64; 2] = [INF, INF];
static P2_ZERO: [f64; 2] = [0.0, 0.0];
static PEND2: Model = Model {
    free_base: false,
    base_mass: 0.0,
    base_com: [0.0; 3],
    base_inertia: [0.0; 3],
    base_pos0: [0.0; 3],
    frames: &P2_FRAMES,
    nq: 2,
    limits: &P2_LIMITS,
    omega_cap: &P2_INF,
    tau_max: &P2_ONE,
    damping: &P2_ZERO,
    armature: &P2_ZERO,
    gravity: [0.0, -G, 0.0],
    mirror: &[0, 1],
    passive_wrist: None,
    passive_shoulder: None,
};

/// Run the engine for `n` substeps at step `h`, zero torque.
fn run(model: &Model, st: &mut State, ws: &mut Workspace, h: f64, n: usize) {
    let tau = vec![0.0; model.nq];
    for _ in 0..n {
        substep(model, st, &tau, h, ws, None);
    }
}

/// Small oscillation: measured period vs 2π√(l/g)·(1 + θ0²/16).
#[test]
fn pendulum_small_oscillation_period() {
    let m = &PEND1;
    let mut st = State::neutral(m);
    st.q[0] = 0.02;
    let mut ws = Workspace::new(m);
    let h = 1e-4;
    let tau = vec![0.0; 1];
    // collect upward zero crossings (q: − → +), linear interpolation
    let mut crossings: Vec<f64> = Vec::new();
    let mut prev = st.q[0];
    for i in 0..100_000 {
        substep(m, &mut st, &tau, h, &mut ws, None);
        let cur = st.q[0];
        if prev < 0.0 && cur >= 0.0 {
            let t = (i as f64 + prev / (prev - cur)) * h + h;
            crossings.push(t);
        }
        prev = cur;
    }
    assert!(crossings.len() >= 4, "too few oscillations seen");
    let n = crossings.len();
    let period = (crossings[n - 1] - crossings[0]) / (n - 1) as f64;
    let expect = 2.0 * std::f64::consts::PI / G.sqrt() * (1.0 + 0.02f64.powi(2) / 16.0);
    let rel = (period - expect).abs() / expect;
    println!("period {period} vs {expect} (rel {rel:e})");
    assert!(rel <= 5e-4, "period off: rel {rel:e}");
}

/// RK4 at tiny step on θ̈ = −(g/l)·sinθ — the continuous reference.
fn rk4_pend1(theta0: f64, t_end: f64, h: f64) -> (f64, f64) {
    let f = |th: f64, om: f64| (om, -G * th.sin());
    let (mut th, mut om) = (theta0, 0.0);
    let n = (t_end / h).round() as usize;
    for _ in 0..n {
        let (k1t, k1o) = f(th, om);
        let (k2t, k2o) = f(th + 0.5 * h * k1t, om + 0.5 * h * k1o);
        let (k3t, k3o) = f(th + 0.5 * h * k2t, om + 0.5 * h * k2o);
        let (k4t, k4o) = f(th + h * k3t, om + h * k3o);
        th += h / 6.0 * (k1t + 2.0 * k2t + 2.0 * k3t + k4t);
        om += h / 6.0 * (k1o + 2.0 * k2o + 2.0 * k3o + k4o);
    }
    (th, om)
}

/// Semi-implicit Euler converges O(h) to the continuous solution:
/// halve h, error halves (ratio gate [1.7, 2.3]).
#[test]
fn pendulum_o_h_convergence() {
    let theta0 = 1.2;
    let t_end = 2.0;
    let (th_ref, _) = rk4_pend1(theta0, t_end, 1e-5);
    let mut errs = [0.0f64; 3];
    for (i, h) in [1.0 / 240.0, 1.0 / 480.0, 1.0 / 960.0].into_iter().enumerate() {
        let m = &PEND1;
        let mut st = State::neutral(m);
        st.q[0] = theta0;
        let mut ws = Workspace::new(m);
        run(m, &mut st, &mut ws, h, (t_end / h).round() as usize);
        errs[i] = (st.q[0] - th_ref).abs();
    }
    let r1 = errs[0] / errs[1];
    let r2 = errs[1] / errs[2];
    println!("1-link errors {errs:?}; ratios {r1:.3}, {r2:.3}");
    assert!((1.7..=2.3).contains(&r1), "ratio h/(h/2) = {r1}");
    assert!((1.7..=2.3).contains(&r2), "ratio (h/2)/(h/4) = {r2}");
}

/// No damping, no clamps: energy drift over 10 s stays bounded (the
/// symplectic oscillation band) and shrinks with h.
#[test]
fn pendulum_energy_bounded() {
    let mut drift = [0.0f64; 2];
    for (i, h) in [1.0f64 / 240.0, 1.0 / 480.0].into_iter().enumerate() {
        let m = &PEND1;
        let mut st = State::neutral(m);
        st.q[0] = 1.2;
        let mut ws = Workspace::new(m);
        fk_all(m, &st, &mut ws);
        let e0 = energy(m, &mut ws, &st.v);
        let tau = vec![0.0; 1];
        let n = (10.0 / h).round() as usize;
        let mut worst = 0.0f64;
        for _ in 0..n {
            substep(m, &mut st, &tau, h, &mut ws, None);
            fk_all(m, &st, &mut ws);
            worst = worst.max((energy(m, &mut ws, &st.v) - e0).abs());
        }
        drift[i] = worst;
    }
    println!("1-link energy drift: h=1/240 {:e}, h=1/480 {:e}", drift[0], drift[1]);
    // scale: E swing amplitude mgl(1−cos 1.2) ≈ 6.25 J
    assert!(drift[0] <= 0.2, "energy drift {:e}", drift[0]);
    assert!(drift[1] <= 0.75 * drift[0], "drift not shrinking with h");
}

/// Hand-derived double-pendulum ODE in ABSOLUTE angles (m1 = m2 = l1 =
/// l2 = 1), from the Lagrangian (Δ = φ1 − φ2):
///   2φ̈1 + cosΔ·φ̈2 = −sinΔ·ω2² − 2g·sinφ1
///   cosΔ·φ̈1 + φ̈2 =  sinΔ·ω1² − g·sinφ2
fn rk4_pend2(phi0: [f64; 2], t_end: f64, h: f64) -> ([f64; 2], [f64; 2]) {
    let f = |s: [f64; 4]| {
        let [p1, p2, w1, w2] = s;
        let d = p1 - p2;
        let (sd, cd) = d.sin_cos();
        let b1 = -sd * w2 * w2 - 2.0 * G * p1.sin();
        let b2 = sd * w1 * w1 - G * p2.sin();
        let det = 2.0 - cd * cd;
        let a1 = (b1 - cd * b2) / det;
        let a2 = (2.0 * b2 - cd * b1) / det;
        [w1, w2, a1, a2]
    };
    let mut s = [phi0[0], phi0[1], 0.0, 0.0];
    let n = (t_end / h).round() as usize;
    for _ in 0..n {
        let k1 = f(s);
        let mut s2 = s;
        for i in 0..4 {
            s2[i] = s[i] + 0.5 * h * k1[i];
        }
        let k2 = f(s2);
        for i in 0..4 {
            s2[i] = s[i] + 0.5 * h * k2[i];
        }
        let k3 = f(s2);
        for i in 0..4 {
            s2[i] = s[i] + h * k3[i];
        }
        let k4 = f(s2);
        for i in 0..4 {
            s[i] += h / 6.0 * (k1[i] + 2.0 * k2[i] + 2.0 * k3[i] + k4[i]);
        }
    }
    ([s[0], s[1]], [s[2], s[3]])
}

fn pend2_energy(s: [f64; 4]) -> f64 {
    let [p1, p2, w1, w2] = s;
    let t = w1 * w1 + 0.5 * w2 * w2 + w1 * w2 * (p1 - p2).cos();
    let v = -2.0 * G * p1.cos() - G * p2.cos();
    t + v
}

/// 2-link chaotic pendulum vs the RK4 reference over 0.5 s — engine in
/// relative coordinates through the generic tables, reference in
/// absolute angles; compared as absolute (φ1, φ1 + q1).
#[test]
fn double_pendulum_vs_rk4() {
    let q0 = [1.2, 0.7]; // relative
    let phi0 = [1.2, 1.9]; // absolute
    let t_end = 0.5;
    // RK4 self-check: energy conservation pins the hand-derived RHS
    let (pr, wr) = rk4_pend2(phi0, t_end, 1e-5);
    let e0 = pend2_energy([phi0[0], phi0[1], 0.0, 0.0]);
    let e1 = pend2_energy([pr[0], pr[1], wr[0], wr[1]]);
    assert!((e1 - e0).abs() <= 1e-8, "RK4 reference energy drift {:e}", e1 - e0);

    let mut errs = [0.0f64; 2];
    for (i, h) in [1.0 / 960.0, 1.0 / 1920.0].into_iter().enumerate() {
        let m = &PEND2;
        let mut st = State::neutral(m);
        st.q[0] = q0[0];
        st.q[1] = q0[1];
        let mut ws = Workspace::new(m);
        run(m, &mut st, &mut ws, h, (t_end / h).round() as usize);
        let phi = [st.q[0], st.q[0] + st.q[1]];
        errs[i] = (phi[0] - pr[0]).abs().max((phi[1] - pr[1]).abs());
    }
    let ratio = errs[0] / errs[1];
    println!("2-link errors {errs:?}; ratio {ratio:.3}");
    assert!(errs[1] <= 2e-2, "2-link error at h=1/1920: {:e}", errs[1]);
    assert!((1.5..=2.5).contains(&ratio), "2-link not O(h): ratio {ratio}");
}
