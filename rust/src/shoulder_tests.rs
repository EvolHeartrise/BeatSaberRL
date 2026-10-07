//! exp/shoulder-rom (2026-10-07): the passive shoulder axial-rotation
//! limit (model.rs `PassiveShoulder` / `shoulder_rom`, step.rs step 4d) and
//! the widened twist box. Design: docs/design/shoulder-rom.md.
//!
//! Gates: constants (peak, damping, box, anchors), the pure-axis profile,
//! −∇E (finite differences, incl. the u = s / 1 / uc knees and the hanging
//! pole), zero work on closed loops, continuity (u = 0 sign switch, the
//! pole), bitwise mirror-exactness, the explicit-integration stiffness
//! budget with the model's own M⁻¹ (ω·h), the implicit edge damping's
//! dissipation, a coasting release past the limit, hard-driven soaks at
//! h = 1/240 and 1/480, and the env-layer exclusions plus [shoulder]
//! telemetry.

use super::*;
use dynamics::algo::{crba, energy, tc_factor, tc_solve_unit_v};
use dynamics::fk::fk_all;
use dynamics::model::{
    shoulder_limits, shoulder_rom, srom_shape, PassiveShoulder, HUMANOID_LIMITS,
    HUMANOID_PASSIVE_SHOULDER, SROM_B_PEAK, SROM_COEF_HI, SROM_COEF_LO, SROM_TAU_PEAK,
};
use dynamics::solver::{substep_constrained, ConstraintSet};

const PS: PassiveShoulder = HUMANOID_PASSIVE_SHOULDER;
const DEG: Real = std::f64::consts::PI as Real / 180.0;
// left arm coordinates
const LY: usize = 4;
const LP: usize = 3;
const LT: usize = 5;
const RY: usize = 11;
const RP: usize = 10;
const RT: usize = 12;

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
    fn range(&mut self, a: Real, b: Real) -> Real {
        a + (b - a) * self.uniform()
    }
}

fn ev(y: Real, p: Real, t: Real) -> dynamics::model::SromOut {
    shoulder_rom(y, p, t, &PS)
}

/// Twist that puts the left arm at normalized excursion u (canonical).
fn twist_at(y: Real, p: Real, u: Real) -> Real {
    let (lo, hi) = shoulder_limits(y, p, &PS);
    y - (0.5 * (hi + lo) + u * 0.5 * (hi - lo))
}

/// Random left-arm (yaw, pitch, twist) over the joint box, with a margin
/// past the stops (one substep of overshoot and more).
fn rand_pose(rng: &mut Rng) -> (Real, Real, Real) {
    let (ylo, yhi) = HUMANOID_LIMITS[LY];
    let (plo, phi) = HUMANOID_LIMITS[LP];
    let (tlo, thi) = HUMANOID_LIMITS[LT];
    (rng.range(ylo - 0.1, yhi + 0.1), rng.range(plo - 0.05, phi + 0.1),
     rng.range(tlo - 0.1, thi + 0.1))
}

#[test]
fn shoulder_constants_box_and_anchors() {
    let t = 0.30 * 19.45 * 0.8729;
    assert!((SROM_TAU_PEAK - t).abs() < 1e-12, "T {SROM_TAU_PEAK}");
    assert!((SROM_B_PEAK - t / 16.0).abs() < 1e-12);
    assert!((SROM_TAU_PEAK - 5.0933715).abs() < 1e-9);
    // the widened twist box: left [-190, 70] deg, right mirrored
    let (lo, hi) = HUMANOID_LIMITS[LT];
    assert!((lo + 190.0 * DEG).abs() < 1e-15 && (hi - 70.0 * DEG).abs() < 1e-15);
    assert_eq!(HUMANOID_LIMITS[RT], (-hi, -lo));
    // the four anchors (theta, phi, ER, IR) in tau degrees, design §B.2
    for (th, ph, er, ir) in [(0.0, 0.0, -80.0, 77.0), (90.0, 0.0, -100.0, 60.0),
                             (90.0, 90.0, -25.0, 130.0), (155.0, 40.0, -75.0, 20.0)] {
        let (y, p) = ((ph - 90.0) * DEG, (th - 90.0) * DEG);
        let (flo, fhi) = shoulder_limits(y, p, &PS);
        println!("anchor elev {th:5.1} plane {ph:4.1}: ER {:8.4} IR {:8.4} deg",
                 flo / DEG, fhi / DEG);
        assert!((flo / DEG - er).abs() < 1e-9 && (fhi / DEG - ir).abs() < 1e-9,
                "anchor ({th}, {ph}) -> ({}, {})", flo / DEG, fhi / DEG);
    }
    // every human IR limit over the yaw/pitch box is reachable inside the
    // twist box (design: min over the box of yaw - tau_hi = -184.3 deg)
    let mut worst: Real = 0.0;
    for i in 0..=145 {
        for k in 0..=155 {
            let (y, p) = ((-100.0 + i as Real) * DEG, (-90.0 + k as Real) * DEG);
            let (_, fhi) = shoulder_limits(y, p, &PS);
            worst = worst.min(y - fhi);
        }
    }
    println!("twist needed for the human IR limit, worst over the box: {:.2} deg", worst / DEG);
    assert!(worst > lo + 3.0 * DEG, "IR limit {} not reachable in the box", worst / DEG);
    assert_eq!(SROM_COEF_LO.len() + SROM_COEF_HI.len(), 8);
}

/// Pure axial profile: along the twist at fixed direction, τ_t equals
/// sign(u)·T·g(|u|) with g = ((a−s)/(1−s))² + ((a−1)/wb)²₊ to uc, linear
/// past it, and the stated knee values.
#[test]
fn shoulder_pure_axis_profile() {
    for &(yd, pd) in &[(0.0, -80.0), (-90.0, 0.0), (0.0, 0.0), (-50.0, 60.0), (30.0, -30.0)] {
        let (y, p) = (yd * DEG, pd * DEG);
        for i in -80..=80 {
            let u = i as Real * 0.025;
            let o = ev(y, p, twist_at(y, p, u));
            let a = u.abs();
            let z = if a > 0.3 { (a - 0.3) / 0.7 } else { 0.0 };
            let x = if a > 1.0 { (a - 1.0) / 0.25 } else { 0.0 };
            let g = if a <= 1.25 {
                z * z + x * x
            } else {
                let (zc, xc) = (0.95 / 0.7, 1.0);
                (zc * zc + xc * xc) + (2.0 * zc / 0.7 + 2.0 * xc / 0.25) * (a - 1.25)
            };
            let want = u.signum() * SROM_TAU_PEAK * g;
            assert!((o.u - u).abs() < 1e-12, "u {u} vs {}", o.u);
            assert!((o.tau_t - want).abs() < 1e-9 * (1.0 + want.abs()),
                    "({yd}, {pd}) u {u}: tau_t {} want {want}", o.tau_t);
        }
    }
    let knee = |u: Real| SROM_TAU_PEAK * srom_shape(u, &PS).0;
    for (u, want) in [(0.5, 0.4158), (0.92, 4.0), (1.0, 5.0934), (1.1, 7.4713),
                      (1.25, 14.4747), (1.5, 29.5973)] {
        println!("pure axis u {u:4.2}: {:.4} N*m", knee(u));
        assert!((knee(u) - want).abs() < 2e-3 * want, "u {u}: {} vs {want}", knee(u));
    }
    // zero torque (and energy) at mid-range; the reset pose is neutral
    let o = ev(0.0, -80.0 * DEG, 0.0);
    println!("reset pose: u {:.4}  tau ({:.2e}, {:.2e}, {:.2e})", o.u, o.tau_y, o.tau_p, o.tau_t);
    assert!(o.u.abs() < 0.3 && o.tau_t == 0.0 && o.tau_y == 0.0 && o.tau_p == 0.0 && o.e == 0.0);
}

#[test]
fn shoulder_matches_energy_gradient() {
    let mut rng = Rng(0x5A0_0001);
    let h = 1e-6;
    let mut worst: Real = 0.0;
    for _ in 0..60_000 {
        let (y, p, t) = rand_pose(&mut rng);
        let o = ev(y, p, t);
        let fd = [
            -(ev(y + h, p, t).e - ev(y - h, p, t).e) / (2.0 * h),
            -(ev(y, p + h, t).e - ev(y, p - h, t).e) / (2.0 * h),
            -(ev(y, p, t + h).e - ev(y, p, t - h).e) / (2.0 * h),
        ];
        for (an, nu) in [(o.tau_y, fd[0]), (o.tau_p, fd[1]), (o.tau_t, fd[2])] {
            let rel = (an - nu).abs() / (1e-3 + an.abs());
            // FD across a knee (u = s, 1, uc: C1 kinks of tau) is O(h) off
            if rel > worst {
                worst = rel;
            }
        }
    }
    println!("shoulder -grad E vs central differences: worst rel err {worst:.2e}");
    assert!(worst < 1e-5, "torque is not -grad E: {worst:e}");
}

/// ∮τ·dq over closed loops (random polygons through the box, past the
/// stops, crossing u = 0 / ±s / ±1 / ±uc) by composite Simpson on fine
/// segments: zero up to quadrature error.
#[test]
fn shoulder_is_conservative_on_closed_loops() {
    let mut rng = Rng(0x5A0_0002);
    let mut worst: Real = 0.0;
    for _ in 0..24 {
        let pts: Vec<(Real, Real, Real)> = (0..5).map(|_| rand_pose(&mut rng)).collect();
        let (mut work, mut tot) = (0.0, 0.0);
        for i in 0..5 {
            let a = pts[i];
            let b = pts[(i + 1) % 5];
            let n = 20_000;
            let d = (b.0 - a.0, b.1 - a.1, b.2 - a.2);
            let f = |s: Real| {
                let o = ev(a.0 + d.0 * s, a.1 + d.1 * s, a.2 + d.2 * s);
                o.tau_y * d.0 + o.tau_p * d.1 + o.tau_t * d.2
            };
            let mut acc = f(0.0) + f(1.0);
            for k in 1..n {
                let s = k as Real / n as Real;
                acc += f(s) * if k % 2 == 1 { 4.0 } else { 2.0 };
            }
            let wseg = acc / (3.0 * n as Real);
            work += wseg;
            tot += wseg.abs();
        }
        worst = worst.max(work.abs() / tot.max(1e-12));
    }
    println!("closed loops: worst |work| / path |work| {worst:.2e}");
    assert!(worst < 1e-7, "non-conservative: {worst:e}");
}

/// Continuity: across u = 0 (the sign switch) and the knees, and at the
/// hanging pole (pitch −90°) where only yaw − twist is a configuration
/// coordinate: E and τ_t are identical for every (y, t) with the same
/// y − t, τ_yaw = −τ_twist exactly (the two axes are anti-parallel), and
/// τ_pitch stays bounded and continuous approaching the pole.
#[test]
fn shoulder_is_continuous_at_the_pole_and_knees() {
    let pole = -90.0 * DEG;
    for tau0 in [-150.0, -60.0, 0.0, 40.0, 75.0, 120.0] {
        let r = ev(0.0, pole, -tau0 * DEG);
        for yd in [-100.0, -40.0, 20.0, 45.0] {
            let y = yd * DEG;
            let o = ev(y, pole, y - tau0 * DEG);
            assert!((o.e - r.e).abs() <= 1e-12 * (1.0 + r.e), "E differs at the pole");
            assert!((o.tau_t - r.tau_t).abs() <= 1e-10 * (1.0 + r.tau_t.abs()));
            assert!((o.tau_y + o.tau_t).abs() <= 1e-10 * (1.0 + o.tau_t.abs()),
                    "tau_y {} != -tau_t {}", o.tau_y, o.tau_t);
            // approaching the pole from inside the box: no blow-up
            for dp in [1e-3, 1e-6, 1e-9] {
                let q = ev(y, pole + dp, y - tau0 * DEG);
                assert!((q.tau_p - o.tau_p).abs() < 1e-2 * (1.0 + o.tau_p.abs()) + 50.0 * dp,
                        "pitch torque jumps near the pole: {} vs {}", q.tau_p, o.tau_p);
            }
        }
    }
    // the u = 0 / s / 1 / uc knees: torque continuous
    let (y, p) = (-0.3, -0.4);
    for uk in [0.0, 0.3, -0.3, 1.0, -1.0, 1.25, -1.25] {
        let e = 1e-11;
        let a = ev(y, p, twist_at(y, p, uk - e));
        let b = ev(y, p, twist_at(y, p, uk + e));
        for (x, z) in [(a.tau_y, b.tau_y), (a.tau_p, b.tau_p), (a.tau_t, b.tau_t)] {
            assert!((x - z).abs() < 1e-8, "jump at u = {uk}: {x} vs {z}");
        }
    }
}

/// The right arm at the mirrored pose gives bit-identical u, b, E and
/// τ_pitch and bit-negated τ_yaw / τ_twist (canonical evaluation).
#[test]
fn shoulder_is_mirror_exact() {
    let mut rng = Rng(0x5A0_0003);
    let mut q = vec![0.0 as Real; NQ];
    for _ in 0..100_000 {
        let (y, p, t) = rand_pose(&mut rng);
        q[LY] = y;
        q[LP] = p;
        q[LT] = t;
        q[RY] = -y;
        q[RP] = p;
        q[RT] = -t;
        let [l, r] = PS.eval(&q);
        assert_eq!(l.u.to_bits(), r.u.to_bits());
        assert_eq!(l.b.to_bits(), r.b.to_bits());
        assert_eq!(l.e.to_bits(), r.e.to_bits());
        assert_eq!(l.tau_p.to_bits(), r.tau_p.to_bits());
        assert_eq!(l.tau_y.to_bits(), (-r.tau_y).to_bits());
        assert_eq!(l.tau_t.to_bits(), (-r.tau_t).to_bits());
    }
}

// ---- stiffness budget -------------------------------------------------------

/// Shoulder 3×3 block (yaw, pitch, twist) of the FULL M⁻¹ (root and every
/// joint free) at pose `st`, from the model's own CRBA + tree factor.
fn shoulder_minv(st: &State) -> [[Real; 3]; 3] {
    let m = MODEL;
    let mut ws = Workspace::new(m);
    fk_all(m, st, &mut ws);
    crba(m, &mut ws);
    tc_factor(&ws.tc, &ws.m, &mut ws.l);
    let nv = m.nv();
    let idx = [LY, LP, LT];
    let mut out = [[0.0; 3]; 3];
    for (a, &ja) in idx.iter().enumerate() {
        let mut col = vec![0.0; nv];
        let mut esc = vec![0.0; nv];
        tc_solve_unit_v(&ws.tc, &ws.l, 6 + ja, 1.0, &mut col, &mut esc);
        for (b, &jb) in idx.iter().enumerate() {
            out[a][b] = col[6 + jb];
        }
    }
    out
}

/// K = −∂τ/∂q (3×3, symmetrized) by central differences.
fn shoulder_k(y: Real, p: Real, t: Real) -> [[Real; 3]; 3] {
    let h = 1e-6;
    let tv = |y: Real, p: Real, t: Real| {
        let o = ev(y, p, t);
        [o.tau_y, o.tau_p, o.tau_t]
    };
    let mut k = [[0.0; 3]; 3];
    for c in 0..3 {
        let mut d = [0.0; 3];
        d[c] = h;
        let a = tv(y + d[0], p + d[1], t + d[2]);
        let b = tv(y - d[0], p - d[1], t - d[2]);
        for r in 0..3 {
            k[r][c] = -(a[r] - b[r]) / (2.0 * h);
        }
    }
    for r in 0..3 {
        for c in 0..r {
            let s = 0.5 * (k[r][c] + k[c][r]);
            k[r][c] = s;
            k[c][r] = s;
        }
    }
    k
}

/// Eigenvalues of a symmetric 3×3 (cyclic Jacobi).
fn sym_eig3(mut a: [[Real; 3]; 3]) -> [Real; 3] {
    for _ in 0..50 {
        for &(p, q) in &[(0usize, 1usize), (0, 2), (1, 2)] {
            if a[p][q].abs() < 1e-300 {
                continue;
            }
            let th = 0.5 * (2.0 * a[p][q]).atan2(a[q][q] - a[p][p]);
            let (s, c) = th.sin_cos();
            let mut b = a;
            for k in 0..3 {
                b[k][p] = c * a[k][p] - s * a[k][q];
                b[k][q] = s * a[k][p] + c * a[k][q];
            }
            let mut d = b;
            for k in 0..3 {
                d[p][k] = c * b[p][k] - s * b[q][k];
                d[q][k] = s * b[p][k] + c * b[q][k];
            }
            a = d;
        }
    }
    [a[0][0], a[1][1], a[2][2]]
}

/// max eig(B·K) for B SPD: eig(Lᵀ K L), B = L Lᵀ (Cholesky).
fn max_eig_bk(b: [[Real; 3]; 3], k: [[Real; 3]; 3]) -> (Real, Real) {
    let mut l = [[0.0; 3]; 3];
    for i in 0..3 {
        for j in 0..=i {
            let mut s = b[i][j];
            for kk in 0..j {
                s -= l[i][kk] * l[j][kk];
            }
            l[i][j] = if i == j { s.sqrt() } else { s / l[j][j] };
        }
    }
    let mut a = [[0.0; 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            let mut s = 0.0;
            for p in 0..3 {
                for q in 0..3 {
                    s += l[p][i] * k[p][q] * l[q][j];
                }
            }
            a[i][j] = s;
        }
    }
    let e = sym_eig3(a);
    (e.iter().cloned().fold(Real::MIN, Real::max), e.iter().cloned().fold(Real::MAX, Real::min))
}

/// Explicit-integration budget with the model's own M⁻¹: arm poses over the
/// box (straight and bent elbows) × twist at the human limits, past them
/// and at both twist stops (+ one substep of overshoot), plus random box
/// poses. Symplectic Euler on a linear spring is stable for ω·h < 2; the
/// wrist tendon was accepted at 0.71. Also prints the explicit-equivalent
/// edge-damping b·h/I_ax (why the damping is implicit).
#[test]
fn shoulder_stiffness_budget() {
    let mut st = State::neutral(MODEL);
    for (j, v) in Q_START_ARM.iter().chain(Q_START_LEG.iter()) {
        st.q[*j] = *v;
    }
    let (tlo, thi) = HUMANOID_LIMITS[LT];
    let over = H_SUB * MODEL.omega_cap[LT];
    let mut worst = (0.0 as Real, String::new());
    let mut worst_b: Real = 0.0;
    let mut rows = Vec::new();
    let mut rng = Rng(0x5A0_0004);
    let mut poses: Vec<(Real, Real, Real, String)> = Vec::new();
    for &pd in &[-89.0, -60.0, -30.0, 0.0, 30.0, 64.0] {
        for &yd in &[-99.0, -60.0, -20.0, 20.0, 44.0] {
            for &ed in &[0.0, 15.0, 90.0] {
                let (y, p) = (yd * DEG, pd * DEG);
                for (tag, t) in [("u=+1", twist_at(y, p, 1.0)), ("u=-1", twist_at(y, p, -1.0)),
                                 ("u=+1.25", twist_at(y, p, 1.25)),
                                 ("u=-1.25", twist_at(y, p, -1.25)),
                                 ("box lo", tlo - over), ("box hi", thi + over)] {
                    let t = t.clamp(tlo - over, thi + over);
                    poses.push((y, p, t, format!("{tag} p{pd} y{yd} e{ed}")));
                    let _ = ed;
                }
            }
        }
    }
    for _ in 0..1500 {
        let (y, p, t) = rand_pose(&mut rng);
        poses.push((y, p, t, "random".to_string()));
    }
    for (i, (y, p, t, tag)) in poses.iter().enumerate() {
        // elbow cycles 0 / 15 / 90 deg (straight is the lightest axial I)
        st.q[6] = [0.0, 15.0, 90.0][i % 3] * DEG;
        st.q[LY] = *y;
        st.q[LP] = *p;
        st.q[LT] = *t;
        let b = shoulder_minv(&st);
        let k = shoulder_k(*y, *p, *t);
        let (lmax, _lmin) = max_eig_bk(b, k);
        let wh = lmax.max(0.0).sqrt() * H_SUB;
        // axial effective inertia 1/(J B Jᵀ), J = (1, 0, −1)
        let jbj = b[0][0] - 2.0 * b[0][2] + b[2][2];
        let o = ev(*y, *p, *t);
        let bh = o.b * H_SUB * jbj;
        worst_b = worst_b.max(bh);
        rows.push(wh);
        if wh > worst.0 {
            worst = (wh, tag.clone());
        }
    }
    rows.sort_by(|a, b| a.partial_cmp(b).unwrap());
    println!("shoulder stiffness budget over {} poses: omega*h at 1/240 median {:.3} p99 {:.3} \
              max {:.3} ({}); at 1/480 max {:.3}; edge damping b*h/I_ax max {:.3}",
             rows.len(), rows[rows.len() / 2], rows[rows.len() * 99 / 100], worst.0, worst.1,
             worst.0 * 0.5, worst_b);
    assert!(worst.0 < 1.0, "explicit shoulder tendon integration is marginal: {}", worst.0);
}

// ---- dynamics -----------------------------------------------------------------

fn variant(gravity: bool, joint_damping: bool, tau_peak: Real, b_peak: Real) -> Model {
    static ZD: [Real; NQ] = [0.0; NQ];
    let mut m = HUMANOID;
    if !gravity {
        m.gravity = [0.0; 3];
    }
    if !joint_damping {
        m.damping = &ZD;
    }
    let mut ps = PS;
    ps.tau_peak = tau_peak;
    ps.b_peak = b_peak;
    m.passive_shoulder = Some(ps);
    // the wrist tendon's own potential would leak into these energy
    // ledgers (the reset pose's 70° bend is in its zone)
    m.passive_wrist = None;
    m
}

/// Left-arm pose for the dynamics tests: arm forward-raised (pitch −30°,
/// yaw −20°), elbow straight or bent, rest of the body at the reset pose.
fn arm_state(m: &Model, elbow: Real, u: Real) -> State {
    let mut st = State::neutral(m);
    for (j, v) in Q_START_ARM.iter().chain(Q_START_LEG.iter()) {
        st.q[*j] = *v;
    }
    let (y, p) = (-20.0 * DEG, -30.0 * DEG);
    st.q[LY] = y;
    st.q[LP] = p;
    st.q[LT] = twist_at(y, p, u);
    st.q[6] = elbow;
    st.q[RY] = -y;
    st.q[RP] = p;
    st.q[RT] = -st.q[LT];
    st.q[13] = elbow;
    st.q[7] = 1.0;
    st.q[14] = -1.0;
    st
}

fn srom_energy(m: &Model, q: &[Real]) -> Real {
    m.passive_shoulder.map_or(0.0, |ps| {
        let [l, r] = ps.eval(q);
        l.e + r.e
    })
}

/// Implicit edge damping alone (tendon stiffness 0, joint damping 0,
/// gravity 0): an axial spin through the zone loses kinetic energy every
/// substep and strictly faster than with b = 0.
#[test]
fn shoulder_edge_damping_dissipates() {
    let damped = variant(false, false, 0.0, SROM_B_PEAK);
    let free = variant(false, false, 0.0, 0.0);
    let run = |m: &Model| -> Vec<Real> {
        let mut st = arm_state(m, 0.0, 0.9);
        // spin the left axial coordinate outward (u increasing) at 8 rad/s
        st.v[6 + LY] = 0.0;
        st.v[6 + LT] = -8.0;
        st.v[6 + RT] = 8.0;
        let mut ws = Workspace::new(m);
        let mut cs = ConstraintSet::new(m, false, false);
        let tau = [0.0 as Real; NQ];
        let mut out = Vec::new();
        for _ in 0..24 {
            substep_constrained(m, &mut st, &tau, H_SUB, &mut ws, &mut cs);
            fk_all(m, &st, &mut ws);
            out.push(energy(m, &mut ws, &st.v));
        }
        out
    };
    let (a, b) = (run(&damped), run(&free));
    println!("edge damping alone: KE {:.4} -> {:.4} J (b = 0: {:.4} -> {:.4})",
             a[0], a[23], b[0], b[23]);
    for k in 1..24 {
        assert!(a[k] <= a[k - 1] * (1.0 + 1e-12), "damped spin gained energy at {k}");
        assert!(a[k] < b[k], "damping did not dissipate at substep {k}");
    }
}

/// Coasting release from past the ER limit (u = −1.2, gravity off, real
/// joint damping and edge damping, ω caps on): total energy (kinetic +
/// shoulder tendon) never rises above E0 and decays — at h = 1/240 and
/// 1/480, straight and bent elbows.
#[test]
fn shoulder_release_energy() {
    let m = variant(false, true, SROM_TAU_PEAK, SROM_B_PEAK);
    for (h, hl) in [(H_SUB, "1/240"), (H_SUB * 0.5, "1/480")] {
        for elbow in [0.0, 90.0 * DEG] {
            let mut st = arm_state(&m, elbow, -1.2);
            let mut ws = Workspace::new(&m);
            let mut cs = ConstraintSet::new(&m, false, false);
            let tau = [0.0 as Real; NQ];
            let total = |st: &State, ws: &mut Workspace| {
                fk_all(&m, st, ws);
                energy(&m, ws, &st.v) + srom_energy(&m, &st.q)
            };
            let e0 = total(&st, &mut ws);
            let mut emax = e0;
            let mut e1 = e0;
            let n = (0.5 / h) as usize;
            for _ in 0..n {
                substep_constrained(&m, &mut st, &tau, h, &mut ws, &mut cs);
                e1 = total(&st, &mut ws);
                emax = emax.max(e1);
            }
            let u_end = PS.eval(&st.q)[0].u;
            println!("release h={hl} elbow {:3.0}: E0 {e0:.4} J  max {emax:.4}  after 0.5 s {e1:.4}  \
                      u_end {u_end:.3}", elbow / DEG);
            assert!(emax <= e0 * (1.0 + 1e-3), "release gained energy");
            assert!(e1 < 0.5 * e0, "release did not decay");
        }
    }
}

const TAU_BODY_DEFAULT: [Real; TAU_FEAT] = [
    0.9012, 0.9804, 0.9370, 0.9616, 0.8482, 0.8729, 1.1000, 1.1889, 1.4466,
    1.0220, 0.9055, 1.0453, 1.0608, 1.0269, 1.0668, 1.1428,
];

#[derive(Default, Debug)]
struct Soak {
    substeps: u64,
    zone: u64,
    out: u64,
    umax: Real,
    tmax: Real,
    vmax: Real,
    pd: u64,
    env: u64,
}

/// Dynamics-level soak: reset stance, legs and spine posture-held, both
/// arms' yaw / pitch / twist DRIVEN at full default-body τmax with random
/// signs switched every 0.1–0.4 s (slams into the human limit, the twist
/// stops and the corners), elbows random.
fn arm_soak(m: &Model, h: Real, seconds: Real, seed: u64) -> Soak {
    let mut rng = Rng(seed);
    let mut out = Soak::default();
    let mut ws = Workspace::new(m);
    let mut cs = ConstraintSet::new(m, true, true);
    let scale: Vec<Real> = (0..NQ).map(|j| TAU_BODY_DEFAULT[TAU_AXIS[j]]).collect();
    let mut tgt = [0.0 as Real; NQ];
    for (j, v) in Q_START_LEG.iter().chain(Q_START_ARM.iter()) {
        tgt[*j] = *v;
    }
    let n_total = (seconds / h) as usize;
    let mut done = 0;
    while done < n_total {
        let mut st = State::neutral(m);
        for (j, v) in Q_START_ARM.iter().chain(Q_START_LEG.iter()) {
            st.q[*j] = *v;
        }
        st.p[1] -= 0.0085;
        let mut hold = 0usize;
        let mut sgn = [0.0 as Real; NQ];
        for _ in 0..(4.0 / h) as usize {
            if hold == 0 {
                hold = (rng.range(0.1, 0.4) / h) as usize;
                for j in (3..7).chain(10..14) {
                    sgn[j] = if j == 6 || j == 13 {
                        rng.range(-0.5, 0.5)
                    } else if rng.uniform() < 0.5 { 1.0 } else { -1.0 };
                }
            }
            hold -= 1;
            let mut tau = [0.0 as Real; NQ];
            for j in 0..NQ {
                let tm = m.tau_max[j] * scale[j];
                tau[j] = if j < 3 || j >= 17 {
                    (tm * (-(st.q[j] - tgt[j]) * 4.0 - 0.4 * st.v[6 + j])).clamp(-tm, tm)
                } else {
                    sgn[j] * tm
                };
            }
            substep_constrained(m, &mut st, &tau, h, &mut ws, &mut cs);
            done += 1;
            out.substeps += 1;
            for s in 0..2 {
                let [tq, u, _, _] = ws.shoulder[s];
                out.umax = out.umax.max(u.abs());
                out.tmax = out.tmax.max(tq);
                if u.abs() > 0.3 {
                    out.zone += 1;
                }
                if u.abs() > 1.0 {
                    out.out += 1;
                }
            }
            for x in st.q.iter().chain(st.v.iter()).chain(st.p.iter()) {
                assert!(x.is_finite(), "non-finite state");
            }
            for j in [LY, LP, LT, RY, RP, RT] {
                out.vmax = out.vmax.max(st.v[6 + j].abs() / m.omega_cap[j]);
            }
            if st.p[1] < 0.55 || done >= n_total {
                break;
            }
        }
    }
    out.pd = ws.pd_hits;
    out.env = ws.env_hits;
    out
}

/// Soaks at h = 1/240 (production) and 1/480: finite, rates under their
/// caps, no more pivot-floor / envelope engagements than the same traffic
/// without the passive shoulder.
#[test]
fn shoulder_soak() {
    let bare = Model { passive_shoulder: None, ..HUMANOID };
    for (h, label) in [(H_SUB, "1/240"), (H_SUB * 0.5, "1/480")] {
        let with = arm_soak(MODEL, h, 40.0, 0x5A0_0005);
        let without = arm_soak(&bare, h, 40.0, 0x5A0_0005);
        let pct = |x: u64| 100.0 * x as Real / (2 * with.substeps) as Real;
        println!("soak h={label} ({} substeps, 40 s): zone {:.1}%  beyond |u|=1 {:.1}%  \
                  max |u| {:.3}  max sum|tau_p| {:.2} N*m  max |qd|/cap - 1 {:.2e}  pd {}  env {} \
                  | none: max |qd|/cap - 1 {:.2e}  pd {}  env {}",
                 with.substeps, pct(with.zone), pct(with.out), with.umax, with.tmax,
                 with.vmax - 1.0, with.pd, with.env, without.vmax - 1.0, without.pd, without.env);
        assert!(with.out > 0, "the soak must reach past the human limit");
        // the step-11 Gauss-Seidel clamp leaves ~2e-5 relative residue on
        // coupled arm joints with or without the passive shoulder
        assert!(with.vmax <= 1.0 + 1e-4 && with.vmax <= without.vmax + 1e-5,
                "a shoulder rate passed its cap");
        assert!(with.pd <= without.pd && with.env <= without.env,
                "the passive shoulder added pivot-floor / envelope engagements");
    }
}

/// Env layer: not muscle effort (a coasting arm parked past the human ER
/// limit pays no torque / energy), counted by the [shoulder] telemetry,
/// and switched off by the test hook.
#[test]
fn shoulder_excluded_from_effort_and_counted_in_telemetry() {
    let gear = joint_gear_rows(5);
    let mut core = Core::new_core(100, 5);
    core.energy_coef = 1.0;
    core.set_chart_notes(Vec::new());
    let park = |core: &mut Core| {
        core.body.reset();
        let (y, p) = (-20.0 * DEG, -30.0 * DEG);
        let t = twist_at(y, p, -1.15);
        core.body.st.q[LY] = y;
        core.body.st.q[LP] = p;
        core.body.st.q[LT] = t;
        core.body.st.q[RY] = -y;
        core.body.st.q[RP] = p;
        core.body.st.q[RT] = -t;
        core.body.refresh_fk();
    };
    park(&mut core);
    let _ = core.pop_shoulder();
    let _ = core.pop_torque();
    let a = [0u8; NQ];
    core.body.tick(&a, &gear);
    assert_eq!(core.body.tau_abs, 0.0, "passive torque leaked into the energy base");
    assert!(core.body.tau.iter().all(|&t| t == 0.0));
    let tk = core.body.srom_tick;
    println!("one coasting tick past the ER limit: srom_tick {tk:?}");
    assert!(tk[0] > 0.0 && (tk[0] - tk[1]).abs() <= 1e-9 * tk[0], "arms differ: {tk:?}");
    assert_eq!(tk[2], N_SUB as Real);
    assert!(tk[4] >= 1.0 && tk[6] > 0.0 && (tk[6] - tk[7]).abs() <= 1e-9 * tk[6]);
    park(&mut core);
    let (_, _) = core.step_inner(&a);
    let (s, n) = core.pop_shoulder();
    let (tq, _) = core.pop_torque();
    assert_eq!(n, N_SUB as u64);
    assert_eq!(tq[0], 0.0);
    assert!(s[0] > 0.0 && s[2] == N_SUB as Real && s[4] >= 1.0);
    assert_eq!(core.pop_shoulder().1, 0, "pop zeroes");
    core.body.srom_on = false;
    park(&mut core);
    core.body.tick(&a, &gear);
    assert_eq!(core.body.srom_tick, [0.0; 8]);
    assert_eq!(core.body.ws.shoulder, [[0.0; 4]; 2]);
}
