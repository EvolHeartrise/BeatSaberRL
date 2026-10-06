//! exp/wrist-tendon (2026-10-05): the passive wrist tendon torque
//! (model.rs `PassiveWrist` / `wrist_tendon`, step.rs step 4c).
//!
//! Gates: the pure-axis ramp, −∇E (finite differences), zero work on
//! closed loops (incl. ones crossing the q = 0 sign boundaries where
//! the normalizing limit switches), continuity at those boundaries,
//! bitwise mirror-exactness, the corner torques, the explicit-
//! integration stiffness budget (ω·h), energy conservation of a free
//! release from a corner, hard-driven corner soaks at h = 1/240 and
//! 1/480, and the env-layer exclusions (energy price / [torque]) plus
//! the [tendon] telemetry wiring.

use super::*;
use dynamics::algo::{crba, energy, tc_factor, tc_solve_unit_v};
use dynamics::fk::fk_all;
use dynamics::model::{
    tendon_edge_x2, wrist_tendon, wrist_tendon_energy, HUMANOID_LIMITS, HUMANOID_PASSIVE_WRIST,
    TENDON_B_PEAK_BEND, TENDON_B_PEAK_DEV, TENDON_S_BEND, TENDON_S_DEV, TENDON_TAU_PEAK_BEND,
    TENDON_TAU_PEAK_DEV,
};
use dynamics::solver::{substep_constrained, ConstraintSet};

const LB: usize = 8; // l_wrist_bend
const LD: usize = 9; // l_wrist_dev
const RB: usize = 15;
const RD: usize = 16;

fn tl(qb: Real, qd: Real) -> (Real, Real, Real) {
    wrist_tendon(qb, qd, HUMANOID_LIMITS[LB], HUMANOID_LIMITS[LD],
                 TENDON_TAU_PEAK_BEND, TENDON_TAU_PEAK_DEV, TENDON_S_BEND, TENDON_S_DEV)
}
fn tr(qb: Real, qd: Real) -> (Real, Real, Real) {
    wrist_tendon(qb, qd, HUMANOID_LIMITS[RB], HUMANOID_LIMITS[RD],
                 TENDON_TAU_PEAK_BEND, TENDON_TAU_PEAK_DEV, TENDON_S_BEND, TENDON_S_DEV)
}
fn el(qb: Real, qd: Real) -> Real {
    wrist_tendon_energy(qb, qd, HUMANOID_LIMITS[LB], HUMANOID_LIMITS[LD],
                        TENDON_TAU_PEAK_BEND, TENDON_TAU_PEAK_DEV, TENDON_S_BEND,
                        TENDON_S_DEV)
}
fn er(qb: Real, qd: Real) -> Real {
    wrist_tendon_energy(qb, qd, HUMANOID_LIMITS[RB], HUMANOID_LIMITS[RD],
                        TENDON_TAU_PEAK_BEND, TENDON_TAU_PEAK_DEV, TENDON_S_BEND,
                        TENDON_S_DEV)
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
    fn range(&mut self, a: Real, b: Real) -> Real {
        a + (b - a) * self.uniform()
    }
}

#[test]
fn tendon_peaks_are_030_of_the_default_body() {
    // 0.30 x 1.431 x 1.4466 (bend, exp/bend7 default body) and
    // 0.30 x 1.06 x 1.0220 (dev) — constants.TENDON_TAU_PEAK_* agree
    assert!((TENDON_TAU_PEAK_BEND - 0.30 * 1.431 * 1.4466).abs() < 1e-15);
    assert!((TENDON_TAU_PEAK_DEV - 0.30 * 1.06 * 1.0220).abs() < 1e-15);
    println!("tau_peak bend {TENDON_TAU_PEAK_BEND:.5} N*m, dev {TENDON_TAU_PEAK_DEV:.5} N*m");
    let pw = MODEL.passive_wrist.expect("HUMANOID carries the tendon");
    assert_eq!(pw.hands, [(LB, LD), (RB, RD)]);
}

/// u_d = 0: τ_b = −sign·τ_peak·((|u_b| − s_b)/(1 − s_b))² with the bend
/// zone start s_b = 0 (τ_peak_b·u_b² from neutral), τ_peak at the limit,
/// no dev torque; likewise for dev (s_d = 0) on both sides
/// of both hands (asymmetric 30°/15° limits). The elliptical term (onset
/// r = 0.5) plus the per-joint centring term (tendon_centring) give
/// exactly this ramp on a pure axis, past the stops too.
#[test]
fn tendon_pure_axis() {
    let ramp = |u: Real, tp: Real, s: Real| -> Real {
        let a = u.abs();
        if a <= s { 0.0 } else { -u.signum() * tp * ((a - s) / (1.0 - s)).powi(2) }
    };
    let mut worst: Real = 0.0;
    for i in -300..=300 {
        let u = i as Real / 200.0; // -1.5 ..= 1.5 (past the stops too)
        for (f, jb, jd) in [(tl as fn(Real, Real) -> (Real, Real, Real), LB, LD), (tr, RB, RD)] {
            // bend axis
            let (lo, hi) = HUMANOID_LIMITS[jb];
            let q = if u >= 0.0 { u * hi } else { u * -lo };
            let (tb, td, r) = f(q, 0.0);
            assert_eq!(td, 0.0);
            assert!((r - u.abs()).abs() < 1e-15);
            let want = ramp(u, TENDON_TAU_PEAK_BEND, TENDON_S_BEND);
            worst = worst.max((tb - want).abs());
            assert!((tb - want).abs() <= 1e-14 * (1.0 + want.abs()), "bend u={u}: {tb} vs {want}");
            // dev axis
            let (lo, hi) = HUMANOID_LIMITS[jd];
            let q = if u >= 0.0 { u * hi } else { u * -lo };
            let (tb, td, _) = f(0.0, q);
            assert_eq!(tb, 0.0);
            let want = ramp(u, TENDON_TAU_PEAK_DEV, TENDON_S_DEV);
            worst = worst.max((td - want).abs());
            assert!((td - want).abs() <= 1e-14 * (1.0 + want.abs()), "dev u={u}: {td} vs {want}");
        }
    }
    // the limits themselves
    let (b_hi, d_hi, d_lo) = (HUMANOID_LIMITS[LB].1, HUMANOID_LIMITS[LD].1, HUMANOID_LIMITS[LD].0);
    let at = [tl(b_hi, 0.0).0, tl(-b_hi, 0.0).0, tl(0.0, d_hi).1, tl(0.0, d_lo).1];
    println!("pure axis at the limits: bend +{:.5}/-{:.5}  dev(+15deg) {:.5}  dev(-30deg) {:.5} N*m; \
              max |err| {worst:.1e}", -at[0], at[1], -at[2], at[3]);
    assert!((at[0] + TENDON_TAU_PEAK_BEND).abs() < 1e-14);
    assert!((at[1] - TENDON_TAU_PEAK_BEND).abs() < 1e-14);
    assert!((at[2] + TENDON_TAU_PEAK_DEV).abs() < 1e-14);
    assert!((at[3] - TENDON_TAU_PEAK_DEV).abs() < 1e-14);
    // onsets: both joints centre from neutral (zone starts 0)
    assert_eq!((TENDON_S_BEND, TENDON_S_DEV), (0.0, 0.0));
    assert!(tl(1e-6, 0.0).0 < 0.0 && tl(-1e-6, 0.0).0 > 0.0);
    assert!(tl(0.0, 1e-6).1 < 0.0 && tl(0.0, -1e-6).1 > 0.0);
    assert_eq!(tl(0.0, 0.0), (0.0, 0.0, 0.0));
    // pure-axis table (left wrist): bend / dev at fractions of the range
    for f in [0.1, 0.25, 0.5, 0.75, 1.0] {
        let (tb, _, _) = tl(f * b_hi, 0.0);
        let (_, tdu, _) = tl(0.0, f * d_lo);
        let (_, tdr, _) = tl(0.0, f * d_hi);
        println!("  pure axis {:3.0}%: bend {tb:+.4}  dev(-30deg side) {tdu:+.4}  dev(+15deg side) {tdr:+.4}",
                 100.0 * f);
    }
}

/// τ = −∇E by central differences over the box and past the stops.
#[test]
fn tendon_matches_energy_gradient() {
    let mut rng = Rng(0x7E4D_0001);
    let mut worst: Real = 0.0;
    for _ in 0..20_000 {
        for (f, e, jb, jd) in [(tl as fn(Real, Real) -> (Real, Real, Real),
                                el as fn(Real, Real) -> Real, LB, LD), (tr, er, RB, RD)] {
            let (blo, bhi) = HUMANOID_LIMITS[jb];
            let (dlo, dhi) = HUMANOID_LIMITS[jd];
            let qb = rng.range(1.3 * blo, 1.3 * bhi);
            let qd = rng.range(1.6 * dlo, 1.6 * dhi);
            let h = 1e-6;
            let gb = (e(qb + h, qd) - e(qb - h, qd)) / (2.0 * h);
            let gd = (e(qb, qd + h) - e(qb, qd - h)) / (2.0 * h);
            let (tb, td, _) = f(qb, qd);
            let scale = 1e-3 + tb.abs() + td.abs();
            let err = ((tb + gb).abs() + (td + gd).abs()) / scale;
            worst = worst.max(err);
            assert!(err < 1e-5, "q=({qb},{qd}): tau ({tb},{td}) vs -grad ({},{})", -gb, -gd);
        }
    }
    println!("tau vs -grad E: worst relative error {worst:.2e} over 40,000 points");
}

/// Work integral ∮ τ·dq ≈ 0 around closed loops: circles/ellipses about
/// the neutral pose (crossing q_b = 0 and q_d = 0, where the limit
/// switches), loops through the box corners and past the stops, and
/// small loops straddling each sign boundary inside the zone.
#[test]
fn tendon_is_conservative_on_closed_loops() {
    let loops: Vec<(Real, Real, Real, Real)> = vec![
        // (center_b, center_d, radius_b, radius_d)
        (0.0, 0.0, 1.2, 0.35),     // around neutral, crosses both boundaries
        (0.0, 0.0, 1.66, 0.52),    // through/near the corners
        (0.0, 0.0, 2.0, 0.6),      // past the stops
        (1.2, 0.0, 0.3, 0.2),      // straddles q_d = 0 in the bend zone
        (0.0, -0.3, 0.3, 0.15),    // straddles q_b = 0 in the dev zone
        (-1.4, 0.2, 0.5, 0.3),     // corner region, crosses q_d = 0
        (0.5, 0.1, 0.2, 0.05),     // mostly outside the zone
    ];
    for (f, name) in [(tl as fn(Real, Real) -> (Real, Real, Real), "L"), (tr, "R")] {
        for &(cb, cd, rb, rd) in &loops {
            let n = 200_000;
            let (mut work, mut abs_work) = (0.0 as Real, 0.0 as Real);
            let pt = |k: usize| -> (Real, Real) {
                let a = 2.0 * std::f64::consts::PI * k as Real / n as Real;
                (cb + rb * a.cos(), cd + rd * a.sin())
            };
            for k in 0..n {
                let (b0, d0) = pt(k);
                let (b1, d1) = pt(k + 1);
                let (tb0, td0, _) = f(b0, d0);
                let (tb1, td1, _) = f(b1, d1);
                let dw = 0.5 * (tb0 + tb1) * (b1 - b0) + 0.5 * (td0 + td1) * (d1 - d0);
                work += dw;
                abs_work += dw.abs();
            }
            println!("{name} loop c=({cb},{cd}) r=({rb},{rd}): closed-loop work {work:+.2e} J \
                      (path |work| {abs_work:.3e} J)");
            assert!(work.abs() <= 1e-6 * abs_work.max(1e-9) + 1e-12,
                    "{name} loop ({cb},{cd},{rb},{rd}): work {work:e} of {abs_work:e}");
        }
    }
}

/// The torque is continuous across q_b = 0 and q_d = 0 (where L flips
/// between the two limits), anywhere in the zone.
#[test]
fn tendon_is_continuous_across_sign_boundaries() {
    let eps = 1e-12;
    let mut worst: Real = 0.0;
    for (f, jb, jd) in [(tl as fn(Real, Real) -> (Real, Real, Real), LB, LD), (tr, RB, RD)] {
        let (blo, bhi) = HUMANOID_LIMITS[jb];
        let (dlo, dhi) = HUMANOID_LIMITS[jd];
        for i in 0..=100 {
            let s = i as Real / 100.0;
            // across q_d = 0, bend anywhere (incl. past the stops)
            let qb = 1.3 * (blo + (bhi - blo) * s);
            let (a, b) = (f(qb, eps), f(qb, -eps));
            worst = worst.max((a.0 - b.0).abs()).max((a.1 - b.1).abs());
            // across q_b = 0
            let qd = 1.3 * (dlo + (dhi - dlo) * s);
            let (a, b) = (f(eps, qd), f(-eps, qd));
            worst = worst.max((a.0 - b.0).abs()).max((a.1 - b.1).abs());
        }
    }
    println!("max torque jump across the sign boundaries (|dq| = 2e-12): {worst:.2e} N*m");
    assert!(worst < 1e-9, "discontinuity {worst:e}");
    // kink-free (C¹) at the centring onset a = s_b (bend) and at the
    // ellipse onset a = 0.5: one-sided stiffness on both sides agrees
    let bhi = HUMANOID_LIMITS[LB].1;
    let dhi = HUMANOID_LIMITS[LD].1;
    let h = 1e-7;
    for (name, f) in [
        ("bend a=s_b (=0, across q=0)", (|d: Real| tl((TENDON_S_BEND + d) * HUMANOID_LIMITS[LB].1, 0.0).0) as fn(Real) -> Real),
        ("bend a=0.5", |d: Real| tl((0.5 + d) * HUMANOID_LIMITS[LB].1, 0.0).0),
        ("dev a=0.5", |d: Real| tl(0.0, (0.5 + d) * HUMANOID_LIMITS[LD].1).1),
    ] {
        let kl = (f(0.0) - f(-h)) / h;
        let kr = (f(h) - f(0.0)) / h;
        println!("  {name}: one-sided slopes {kl:+.3e} / {kr:+.3e} (per unit u)");
        assert!((kl - kr).abs() < 1e-4, "{name}: kink {kl} vs {kr}");
    }
    let _ = (bhi, dhi);
}

/// Bitwise mirror-exactness: the right wrist at the mirrored pose
/// (bend kept, dev negated) yields the left's bend torque and the
/// negated dev torque, bit for bit.
#[test]
fn tendon_is_mirror_exact() {
    let mut rng = Rng(0x7E4D_0002);
    for _ in 0..100_000 {
        let qb = rng.range(-2.2, 2.2);
        let qd = rng.range(-0.9, 0.9);
        let (lb, ld, lr) = tl(qb, qd);
        let (rb, rd, rr) = tr(qb, -qd);
        // bit for bit, except the sign of an exact zero (r <= R0 returns
        // +0.0 on both hands; −(+0.0) is −0.0, the same value)
        assert_eq!(lb.to_bits(), rb.to_bits(), "bend at ({qb},{qd})");
        if ld != 0.0 {
            assert_eq!(ld.to_bits(), (-rd).to_bits(), "dev at ({qb},{qd})");
        } else {
            assert_eq!(rd, 0.0);
        }
        assert_eq!(lr.to_bits(), rr.to_bits());
        assert_eq!(el(qb, qd).to_bits(), er(qb, -qd).to_bits());
    }
    // the model-level parameters map onto themselves under the mirror
    let pw = HUMANOID_PASSIVE_WRIST;
    for s in 0..2 {
        let (jb, jd) = pw.hands[s];
        assert_eq!((MODEL.mirror[jb], MODEL.mirror[jd]), pw.hands[1 - s]);
    }
}

/// Torques at the box corners (full bend + full dev) and at the 70%/70%
/// point. Expected well above τ_peak at the corners (P keeps rising
/// past r = 1).
#[test]
fn tendon_corner_torques() {
    let (bhi, dlo, dhi) = (HUMANOID_LIMITS[LB].1, HUMANOID_LIMITS[LD].0, HUMANOID_LIMITS[LD].1);
    println!("left wrist (dev + = ulnar-side 15 deg stop, dev - = 30 deg stop):");
    for (name, qb, qd) in [
        ("corner +bend +dev(15)", bhi, dhi),
        ("corner +bend -dev(30)", bhi, dlo),
        ("corner -bend +dev(15)", -bhi, dhi),
        ("corner -bend -dev(30)", -bhi, dlo),
        ("70/70 +bend +dev", 0.7 * bhi, 0.7 * dhi),
        ("70/70 +bend -dev", 0.7 * bhi, 0.7 * dlo),
        ("50/50 +bend -dev", 0.5 * bhi, 0.5 * dlo),
    ] {
        let (tb, td, r) = tl(qb, qd);
        println!("  {name:24} q=({qb:+.4},{qd:+.4}) r={r:.3}  tau_bend {tb:+.4}  tau_dev {td:+.4} N*m  \
                  E {:.4} J", el(qb, qd));
        if name.starts_with("corner") {
            assert!(tb.abs() > TENDON_TAU_PEAK_BEND && td.abs() > TENDON_TAU_PEAK_DEV,
                    "{name}: corner torques must exceed the pure-axis peaks");
        }
        // always restoring
        assert!(tb * qb < 0.0 && td * qd < 0.0);
    }
}

/// M⁻¹ restricted to a wrist's (bend, dev) rows at pose `st`: the
/// effective inverse inertia the explicit tendon spring acts through.
fn wrist_minv(st: &State, jb: usize, jd: usize) -> [[Real; 2]; 2] {
    let m = MODEL;
    let mut ws = Workspace::new(m);
    fk_all(m, st, &mut ws);
    crba(m, &mut ws);
    tc_factor(&ws.tc, &ws.m, &mut ws.l);
    let nv = m.nv();
    let mut out = [[0.0; 2]; 2];
    for (a, &ja) in [jb, jd].iter().enumerate() {
        let mut col = vec![0.0; nv];
        let mut esc = vec![0.0; nv];
        tc_solve_unit_v(&ws.tc, &ws.l, 6 + ja, 1.0, &mut col, &mut esc);
        out[a][0] = col[6 + jb];
        out[a][1] = col[6 + jd];
    }
    out
}

/// Stiffness K = −∂τ/∂q (2×2, symmetric: conservative) by central
/// differences.
fn tendon_k(f: fn(Real, Real) -> (Real, Real, Real), qb: Real, qd: Real) -> [[Real; 2]; 2] {
    let h = 1e-7;
    let (a, b) = (f(qb + h, qd), f(qb - h, qd));
    let (c, d) = (f(qb, qd + h), f(qb, qd - h));
    [[-(a.0 - b.0) / (2.0 * h), -(c.0 - d.0) / (2.0 * h)],
     [-(a.1 - b.1) / (2.0 * h), -(c.1 - d.1) / (2.0 * h)]]
}

fn max_eig2(a: [[Real; 2]; 2]) -> Real {
    // largest real eigenvalue of a 2x2 (A = M⁻¹K is similar to a
    // symmetric matrix when K is SPD; the discriminant is then >= 0)
    let tr = a[0][0] + a[1][1];
    let det = a[0][0] * a[1][1] - a[0][1] * a[1][0];
    let disc = (0.25 * tr * tr - det).max(0.0).sqrt();
    0.5 * tr + disc
}

/// Explicit-integration budget at the stiffest reachable states: the
/// box corners and the deepest overshoot past both stops a single
/// substep can carry (h·ω_cap: 0.4375 rad bend, 0.2083 rad dev), with
/// the hand's effective inverse inertia from the model's own M⁻¹ at the
/// reset arm pose. Symplectic Euler on a linear spring is stable for
/// ω·h < 2.
#[test]
fn tendon_stiffness_budget() {
    let (bhi, dlo, dhi) = (HUMANOID_LIMITS[LB].1, HUMANOID_LIMITS[LD].0, HUMANOID_LIMITS[LD].1);
    let ob = H_SUB * MODEL.omega_cap[LB];
    let od = H_SUB * MODEL.omega_cap[LD];
    let mut st = State::neutral(MODEL);
    for (j, v) in Q_START_ARM {
        st.q[j] = v;
    }
    let mut worst = (0.0 as Real, "");
    for (name, qb, qd) in [
        ("bend stop", bhi, 0.0),
        ("dev stop (15)", 0.0, dhi),
        ("dev stop (30)", 0.0, dlo),
        ("70/70 (+dev)", 0.7 * bhi, 0.7 * dhi),
        ("corner (+dev 15)", bhi, dhi),
        ("corner (-dev 30)", bhi, dlo),
        ("overshoot (+dev)", bhi + ob, dhi + od),
        ("overshoot (-dev)", bhi + ob, dlo - od),
    ] {
        let k = tendon_k(tl, qb, qd);
        st.q[LB] = qb;
        st.q[LD] = qd;
        let mi = wrist_minv(&st, LB, LD);
        // A = M⁻¹_ww K
        let a = [[mi[0][0] * k[0][0] + mi[0][1] * k[1][0], mi[0][0] * k[0][1] + mi[0][1] * k[1][1]],
                 [mi[1][0] * k[0][0] + mi[1][1] * k[1][0], mi[1][0] * k[0][1] + mi[1][1] * k[1][1]]];
        let kmax = max_eig2(k);
        let w2 = max_eig2(a);
        let w = w2.max(0.0).sqrt();
        println!("  {name:18} k_max {kmax:7.3} N*m/rad  I_eff(bend,dev) {:.2e}/{:.2e}  \
                  omega {w:6.1} rad/s  omega*h: 1/240 {:.3}  1/480 {:.3}",
                 1.0 / mi[0][0], 1.0 / mi[1][1], w * H_SUB, w * H_SUB * 0.5);
        if w * H_SUB > worst.0 {
            worst = (w * H_SUB, name);
        }
    }
    println!("worst omega*h at h=1/240: {:.3} ({})", worst.0, worst.1);
    assert!(worst.0 < 1.0, "explicit tendon integration is marginal: omega*h = {}", worst.0);
}

/// Release from a corner-region pose, no gravity, no ground, no gears
/// (roll mid-range so no arm joint starts on a stop); the window ends at
/// the first arm stop contact (inelastic rows). E = kinetic + tendon.
fn release(m: &Model, start: Real, h: Real, seconds: Real) -> (Real, Real, Real, Real, Real) {
    let mut st = State::neutral(m);
    for (j, v) in Q_START_ARM {
        st.q[j] = v;
    }
    let (bhi, dhi) = (HUMANOID_LIMITS[LB].1, HUMANOID_LIMITS[LD].1);
    st.q[7] = 1.0;
    st.q[14] = -1.0;
    st.q[LB] = start * bhi;
    st.q[LD] = start * dhi;
    st.q[RB] = start * bhi;
    st.q[RD] = -start * dhi;
    let mut ws = Workspace::new(m);
    let mut cs = ConstraintSet::new(m, false, false);
    let tau = [0.0 as Real; NQ];
    let total = |st: &State, ws: &mut Workspace| -> Real {
        fk_all(m, st, ws);
        energy(m, ws, &st.v) + el(st.q[LB], st.q[LD]) + er(st.q[RB], st.q[RD])
    };
    let e0 = total(&st, &mut ws);
    let (mut emax, mut e1, mut t) = (e0, e0, 0.0);
    for i in 0..(seconds / h) as usize {
        substep_constrained(m, &mut st, &tau, h, &mut ws, &mut cs);
        if (3..17).any(|j| st.q[j] <= m.limits[j].0 || st.q[j] >= m.limits[j].1) {
            break;
        }
        e1 = total(&st, &mut ws);
        emax = emax.max(e1);
        t = (i + 1) as Real * h;
    }
    (e0, emax, e1, t, (e1 - e0) / t.max(1e-9))
}

/// HUMANOID variants for the energy / vibration experiments.
fn variant(gravity: bool, joint_damping: bool, caps: bool, edge: bool) -> Model {
    static ZD: [Real; NQ] = [0.0; NQ];
    static NOCAP: [Real; NQ] = [1e9; NQ];
    let mut m = HUMANOID;
    if !gravity {
        m.gravity = [0.0; 3];
    }
    if !joint_damping {
        m.damping = &ZD;
    }
    if !caps {
        m.omega_cap = &NOCAP;
    }
    if !edge {
        let mut pw = HUMANOID_PASSIVE_WRIST;
        pw.b_peak_bend = 0.0;
        pw.b_peak_dev = 0.0;
        m.passive_wrist = Some(pw);
    }
    m
}

/// Energy behaviour (stability evidence).
///
/// (1) The real body (joint damping + ω caps + edge damping): released
///     from 70 % of the corner, total energy never rises above E0 and
///     ends near 0 — at h = 1/240 and 1/480.
/// (2) The tendon ALONE (joint damping 0, caps off, edge damping off):
///     semi-implicit Euler with a configuration-dependent M is not
///     symplectic, and the stiff tendon shows its first-order energy
///     drift as a SECULAR gain (from the 55 %-corner start: see the printed rates) that
///     halves with h — integration error, asserted to converge.
/// (3) The same with ONLY the edge damping on (issue #61): net
///     dissipative at every h (ends below half of E0). Since the zone
///     starts moved to 0 (centring torques active everywhere, edge
///     damping still only for r > 0.5), the undamped hand also swings
///     through the central region, where the explicit centring spring's
///     small first-order gain is not damped: a transient excess of < 1 %
///     of E0 at 1/240, shrinking with h (asserted < 2 %). The real body's
///     joint damping removes it ((1): E never exceeds E0).
#[test]
fn tendon_release_energy() {
    for (h, label) in [(H_SUB, "1/240"), (H_SUB * 0.5, "1/480")] {
        let m = variant(false, true, true, true);
        let (e0, emax, e1, t, _) = release(&m, 0.7, h, 2.0);
        println!("real-body release h={label}: {t:.2} s  E0 {e0:.4} J  max {emax:.4} J  end {e1:.4} J");
        assert!(emax <= e0 * (1.0 + 1e-9), "energy rose above its start: {e0} -> {emax}");
        assert!(e1 < 0.5 * e0, "not dissipative: {e0} -> {e1}");
        assert!(t > 0.5, "too short a stop-free window ({t} s)");
    }
    let bare = variant(false, false, false, false);
    let edge = variant(false, false, false, true);
    let mut gains = Vec::new();
    for k in 0..4 {
        let h = H_SUB / (1 << k) as Real;
        let (e0, emax, e1, t, rate) = release(&bare, 0.55, h, 1.0);
        println!("tendon only, h=1/{:.0}: {t:.2} s  E0 {e0:.4} J  max {emax:.4} J  end {e1:.4} J  \
                  drift {rate:+.4} J/s", 1.0 / h);
        assert!(t > 0.9, "the undamped release hit a stop at {t} s");
        gains.push(emax - e0);
        let (e0, emax, e1, t, rate) = release(&edge, 0.55, h, 1.0);
        println!("  + edge damping:    {t:.2} s  E0 {e0:.4} J  max {emax:.4} J  end {e1:.4} J  \
                  drift {rate:+.4} J/s", );
        println!("    transient excess {:+.2e} J ({:.2} % of E0)", emax - e0, 100.0 * (emax - e0) / e0);
        assert!(emax <= e0 * 1.02, "edge damping: energy rose {e0} -> {emax}");
        assert!(e1 < 0.5 * e0, "edge damping does not dissipate: {e0} -> {e1}");
    }
    for k in 1..gains.len() {
        assert!(gains[k] < 0.75 * gains[k - 1], "energy error does not converge with h: {gains:?}");
    }
}

/// Edge damping shape (issue #61): b_j(r) = b_peak_j·x², x = (r − 0.5)/0.5
/// for r > 0.5 — zero inside, continuous at the zone edge (x² → 0) and
/// across q = 0 (r is), uncapped past r = 1; b_peak = τ_peak / 30 rad/s;
/// ≥ 0 everywhere, so τ_d·q̇ = −b·q̇² ≤ 0 (strictly dissipative).
#[test]
fn edge_damping_shape() {
    assert!((TENDON_B_PEAK_BEND - TENDON_TAU_PEAK_BEND / 30.0).abs() < 1e-15);
    assert!((TENDON_B_PEAK_DEV - TENDON_TAU_PEAK_DEV / 30.0).abs() < 1e-15);
    println!("b_peak bend {TENDON_B_PEAK_BEND:.6}  dev {TENDON_B_PEAK_DEV:.6} N*m*s/rad");
    assert_eq!(tendon_edge_x2(0.0), 0.0);
    assert_eq!(tendon_edge_x2(0.5), 0.0);
    assert!(tendon_edge_x2(0.5 + 1e-9) < 1e-16);
    assert_eq!(tendon_edge_x2(1.0), 1.0);
    assert!((tendon_edge_x2(2.0) - 9.0).abs() < 1e-14);
    let mut rng = Rng(0x7E4D_0005);
    for _ in 0..100_000 {
        let (qb, qd) = (rng.range(-2.2, 2.2), rng.range(-0.9, 0.9));
        let (vb, vd) = (rng.range(-120.0, 120.0), rng.range(-60.0, 60.0));
        let (_, _, r) = tl(qb, qd);
        let x2 = tendon_edge_x2(r);
        let (bb, bd) = (TENDON_B_PEAK_BEND * x2, TENDON_B_PEAK_DEV * x2);
        assert!(bb >= 0.0 && bd >= 0.0);
        assert!((-bb * vb) * vb + (-bd * vd) * vd <= 0.0, "power > 0");
        if r <= 0.5 {
            assert_eq!((bb, bd), (0.0, 0.0));
        }
        // mirror: the right wrist at the mirrored pose has the same r,
        // hence bit-identical b
        let (_, _, rr) = tr(qb, -qd);
        assert_eq!(tendon_edge_x2(r).to_bits(), tendon_edge_x2(rr).to_bits());
    }
    // continuity across q = 0 (r continuous) and the zone edge
    let mut worst: Real = 0.0;
    for i in 0..=100 {
        let qb = -2.0 + 4.0 * i as Real / 100.0;
        let a = tendon_edge_x2(tl(qb, 1e-12).2);
        let b = tendon_edge_x2(tl(qb, -1e-12).2);
        worst = worst.max((a - b).abs());
        let qd = -0.6 + 1.2 * i as Real / 100.0;
        let a = tendon_edge_x2(tl(1e-12, qd).2);
        let b = tendon_edge_x2(tl(-1e-12, qd).2);
        worst = worst.max((a - b).abs());
    }
    assert!(worst < 1e-10, "edge damping discontinuous: {worst:e}");
}

/// Step-level dissipativity: a wrist spun inside the zone with the
/// tendon stiffness zeroed (τ_peak = 0, b_peak kept), gravity / joint
/// damping / caps / contacts / ground off, stops never reached: the
/// edge damping alone strictly lowers the energy every substep relative
/// to the same run with b_peak = 0, and never adds energy.
#[test]
fn edge_damping_dissipates_in_the_substep() {
    let mut damped = variant(false, false, false, true);
    let mut pw = HUMANOID_PASSIVE_WRIST;
    pw.tau_peak_bend = 0.0;
    pw.tau_peak_dev = 0.0;
    damped.passive_wrist = Some(pw);
    let mut bare = damped;
    pw.b_peak_bend = 0.0;
    pw.b_peak_dev = 0.0;
    bare.passive_wrist = Some(pw);
    let run = |m: &Model| -> Vec<Real> {
        let mut st = State::neutral(m);
        for (j, v) in Q_START_ARM {
            st.q[j] = v;
        }
        st.q[7] = 1.0;
        st.q[14] = -1.0;
        // in the zone (u_b = 0.9) and moving along the dev axis
        st.q[LB] = 0.9 * HUMANOID_LIMITS[LB].1;
        st.q[RB] = 0.9 * HUMANOID_LIMITS[RB].1;
        st.v[6 + LD] = -3.0;
        st.v[6 + RD] = 3.0;
        let mut ws = Workspace::new(m);
        let mut cs = ConstraintSet::new(m, false, false);
        let tau = [0.0 as Real; NQ];
        let mut es = Vec::new();
        for _ in 0..24 {
            substep_constrained(m, &mut st, &tau, H_SUB, &mut ws, &mut cs);
            fk_all(m, &st, &mut ws);
            es.push(energy(m, &mut ws, &st.v));
            for j in [LB, LD, RB, RD] {
                assert!(st.q[j] > m.limits[j].0 && st.q[j] < m.limits[j].1);
            }
        }
        es
    };
    let (ed, eb) = (run(&damped), run(&bare));
    println!("spin energy after 24 substeps: edge-damped {:.6} J vs bare {:.6} J", ed[23], eb[23]);
    for k in 0..24 {
        assert!(ed[k] < eb[k], "substep {k}: damped {} >= bare {}", ed[k], eb[k]);
    }
    for k in 1..24 {
        assert!(ed[k] <= ed[k - 1] + 1e-12, "edge damping added energy at substep {k}");
    }
}

/// Stability of the edge damping and its damping ratios.
///
/// b·h/I_eff (largest eigenvalue of h·M⁻¹_ww·B, the model's own M⁻¹ at
/// the reset arm pose) across the reachable range — explicit treatment
/// would be fine only well below ~0.5; corners sit at ~0.4 and the
/// one-substep overshoot past both stops above 1, so the damping is
/// IMPLICIT (step.rs 4c). ζ per tendon mode: K φ = ω² M_eff φ (M_eff =
/// (M⁻¹_ww)⁻¹), ζ = φᵀ(B + b_joint)φ / (2 ω φᵀM_eff φ), before (joint
/// damping only) and after (+ edge damping).
#[test]
fn edge_damping_budget_and_zeta() {
    let (bhi, dlo, dhi) = (HUMANOID_LIMITS[LB].1, HUMANOID_LIMITS[LD].0, HUMANOID_LIMITS[LD].1);
    let ob = H_SUB * MODEL.omega_cap[LB];
    let od = H_SUB * MODEL.omega_cap[LD];
    let mut st = State::neutral(MODEL);
    for (j, v) in Q_START_ARM {
        st.q[j] = v;
    }
    let bj = MODEL.damping[LB];
    let mut worst = 0.0 as Real;
    for (name, qb, qd) in [
        ("zone edge r=0.6 (bend)", 0.6 * bhi, 0.0),
        ("zone edge r=0.6 (diag)", 0.6 / 2f64.sqrt() * bhi, 0.6 / 2f64.sqrt() * dhi),
        ("bend stop r=1", bhi, 0.0),
        ("dev stop (15) r=1", 0.0, dhi),
        ("dev stop (30) r=1", 0.0, dlo),
        ("70/70 (+dev)", 0.7 * bhi, 0.7 * dhi),
        ("corner (+dev 15)", bhi, dhi),
        ("corner (-dev 30)", bhi, dlo),
        ("overshoot (+dev)", bhi + ob, dhi + od),
        ("overshoot (-dev)", bhi + ob, dlo - od),
    ] {
        st.q[LB] = qb;
        st.q[LD] = qd;
        let mi = wrist_minv(&st, LB, LD);
        let (_, _, r) = tl(qb, qd);
        let x2 = tendon_edge_x2(r);
        let (bb, bd) = (TENDON_B_PEAK_BEND * x2, TENDON_B_PEAK_DEV * x2);
        // h·M⁻¹B: B diagonal
        let a = [[mi[0][0] * bb, mi[0][1] * bd], [mi[1][0] * bb, mi[1][1] * bd]];
        let bhi_ = max_eig2(a);
        // M_eff = inv(M⁻¹_ww)
        let det = mi[0][0] * mi[1][1] - mi[0][1] * mi[1][0];
        let me = [[mi[1][1] / det, -mi[0][1] / det], [-mi[1][0] / det, mi[0][0] / det]];
        let k = tendon_k(tl, qb, qd);
        // generalized eigenproblem via A = M⁻¹K (2×2), eigenvectors
        let ak = [[mi[0][0] * k[0][0] + mi[0][1] * k[1][0], mi[0][0] * k[0][1] + mi[0][1] * k[1][1]],
                  [mi[1][0] * k[0][0] + mi[1][1] * k[1][0], mi[1][0] * k[0][1] + mi[1][1] * k[1][1]]];
        let tr_ = ak[0][0] + ak[1][1];
        let dt_ = ak[0][0] * ak[1][1] - ak[0][1] * ak[1][0];
        let disc = (0.25 * tr_ * tr_ - dt_).max(0.0).sqrt();
        let mut zs = String::new();
        for lam in [0.5 * tr_ + disc, 0.5 * tr_ - disc] {
            if lam <= 1e-9 {
                zs += "  (mode ω²≤0)";
                continue;
            }
            let w = lam.sqrt();
            // eigenvector of ak for lam
            let phi = if ak[0][1].abs() > 1e-12 { [ak[0][1], lam - ak[0][0]] }
                      else if ak[1][0].abs() > 1e-12 { [lam - ak[1][1], ak[1][0]] }
                      else if (lam - ak[0][0]).abs() < (lam - ak[1][1]).abs() { [1.0, 0.0] }
                      else { [0.0, 1.0] };
            let pmp = phi[0] * (me[0][0] * phi[0] + me[0][1] * phi[1])
                + phi[1] * (me[1][0] * phi[0] + me[1][1] * phi[1]);
            let pb0 = bj * (phi[0] * phi[0] + phi[1] * phi[1]);
            let pb1 = pb0 + bb * phi[0] * phi[0] + bd * phi[1] * phi[1];
            zs += &format!("  ω {:5.1} rad/s ({:4.1} Hz) ζ {:.4} -> {:.4}",
                           w, w / (2.0 * std::f64::consts::PI),
                           pb0 / (2.0 * w * pmp), pb1 / (2.0 * w * pmp));
        }
        println!("  {name:24} r {r:.3}  b_bend {bb:.4} b_dev {bd:.4}  b·h/I: 1/240 {:.3}  1/480 {:.3}{zs}",
                 bhi_ * H_SUB, bhi_ * H_SUB * 0.5);
        worst = worst.max(bhi_ * H_SUB);
    }
    println!("worst explicit b·h/I at 1/240: {worst:.3} (-> implicit edge damping)");
    // the reason the damping is implicit: explicit would not be
    // comfortably stable on reachable states
    assert!(worst > 0.5);
}

/// Corner-release vibration, before (tendon only) and after (+ edge
/// damping): the real body (gravity off, joint damping, caps, stops)
/// released at rest from 98 % of the (+bend, +15° dev) corner (both
/// hands, mirrored). Reports the ringing frequency (zero crossings of
/// the left dev rate in the first 0.5 s), the remaining fraction of the
/// released energy (kinetic + tendon) at fixed times, and the time to
/// shed 90 % / 99 % of it.
#[test]
fn corner_release_vibration() {
    let measure = |m: &Model| -> (Real, Vec<Real>, Real, Real) {
        let mut st = State::neutral(m);
        for (j, v) in Q_START_ARM {
            st.q[j] = v;
        }
        st.q[7] = 1.0;
        st.q[14] = -1.0;
        st.q[LB] = 0.98 * HUMANOID_LIMITS[LB].1;
        st.q[LD] = 0.98 * HUMANOID_LIMITS[LD].1;
        st.q[RB] = 0.98 * HUMANOID_LIMITS[RB].1;
        st.q[RD] = -0.98 * HUMANOID_LIMITS[LD].1;
        let mut ws = Workspace::new(m);
        let mut cs = ConstraintSet::new(m, false, false);
        let tau = [0.0 as Real; NQ];
        let total = |st: &State, ws: &mut Workspace| -> Real {
            fk_all(m, st, ws);
            energy(m, ws, &st.v) + el(st.q[LB], st.q[LD]) + er(st.q[RB], st.q[RD])
        };
        let e0 = total(&st, &mut ws);
        let n = (3.0 / H_SUB) as usize;
        let (mut zc, mut prev) = (0usize, 0.0 as Real);
        let (mut t90, mut t99) = (Real::NAN, Real::NAN);
        let mut at = Vec::new();
        for i in 1..=n {
            substep_constrained(m, &mut st, &tau, H_SUB, &mut ws, &mut cs);
            let t = i as Real * H_SUB;
            let vd = st.v[6 + LD];
            if t <= 0.5 && i > 1 && vd * prev < 0.0 {
                zc += 1;
            }
            prev = vd;
            let f = total(&st, &mut ws) / e0;
            if t90.is_nan() && f < 0.1 {
                t90 = t;
            }
            if t99.is_nan() && f < 0.01 {
                t99 = t;
            }
            for tt in [0.05, 0.1, 0.25, 0.5, 1.0] {
                if i == (tt / H_SUB).round() as usize {
                    at.push(f);
                }
            }
        }
        (zc as Real / 2.0 / 0.5, at, t90, t99)
    };
    let before = measure(&variant(false, true, true, false));
    let after = measure(&variant(false, true, true, true));
    for (name, (hz, at, t90, t99)) in [("tendon only", &before), ("+ edge damping", &after)] {
        println!("corner release, {name:15}: ~{hz:.0} Hz; E/E0 at 0.05/0.1/0.25/0.5/1 s = \
                  {:.3}/{:.3}/{:.3}/{:.3}/{:.4}; 90% shed in {t90:.3} s, 99% in {t99:.3} s",
                 at[0], at[1], at[2], at[3], at[4]);
    }
    assert!(after.2 < before.2, "edge damping must shed the release energy faster");
    assert!(after.1[2] < before.1[2]);
}

/// Default-body τ scales per joint (constants.TAU_BODY_DEFAULT).
const TAU_BODY_DEFAULT: [Real; TAU_FEAT] = [
    0.9012, 0.9804, 0.9370, 0.9616, 0.8482, 0.8729, 1.1000, 1.1889, 1.4466,
    1.0220, 0.9055, 1.0453, 1.0608, 1.0269, 1.0668, 1.1428,
];

#[derive(Default, Debug)]
struct Soak {
    substeps: u64,
    zone: u64,
    out: u64,
    rmax: Real,
    tmax: Real,
    vmax: Real,
    pd: u64,
    env: u64,
}

/// Dynamics-level soak at step size `h`: the reset stance, legs held by
/// a PD toward the stance, arms random, and both wrists DRIVEN at full
/// default-body τmax into a random box corner, switching corner every
/// ~0.1–0.4 s (the hardest wrist traffic: slam, pin, reverse).
fn corner_soak(m: &Model, h: Real, seconds: Real, seed: u64) -> Soak {
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
        // fresh episode
        let mut st = State::neutral(m);
        for (j, v) in Q_START_ARM.iter().chain(Q_START_LEG.iter()) {
            st.q[*j] = *v;
        }
        st.p[1] -= 0.0085;
        let mut corner = [(1.0, 1.0); 2];
        let mut hold = 0usize;
        let mut arm_a = [0.0 as Real; NQ];
        for _ in 0..(4.0 / h) as usize {
            if hold == 0 {
                hold = (rng.range(0.1, 0.4) / h) as usize;
                for s in 0..2 {
                    corner[s] = (if rng.uniform() < 0.5 { 1.0 } else { -1.0 },
                                 if rng.uniform() < 0.5 { 1.0 } else { -1.0 });
                }
                for j in (3..7).chain(10..14) {
                    arm_a[j] = rng.range(-0.3, 0.3);
                }
            }
            hold -= 1;
            let mut tau = [0.0 as Real; NQ];
            for j in 0..NQ {
                let tm = m.tau_max[j] * scale[j];
                tau[j] = if j < 3 || j >= 17 {
                    // posture hold (spine + legs)
                    (tm * (-(st.q[j] - tgt[j]) * 4.0 - 0.4 * st.v[6 + j])).clamp(-tm, tm)
                } else {
                    arm_a[j] * tm
                };
            }
            for (s, &(jb, jd)) in [(LB, LD), (RB, RD)].iter().enumerate() {
                tau[jb] = corner[s].0 * m.tau_max[jb] * scale[jb];
                tau[jd] = corner[s].1 * m.tau_max[jd] * scale[jd];
            }
            substep_constrained(m, &mut st, &tau, h, &mut ws, &mut cs);
            done += 1;
            out.substeps += 1;
            for s in 0..2 {
                let [tb, td, r, _] = ws.tendon[s];
                out.rmax = out.rmax.max(r);
                out.tmax = out.tmax.max(tb.abs()).max(td.abs());
                if r > 0.5 {
                    out.zone += 1;
                }
                if r > 1.0 {
                    out.out += 1;
                }
            }
            for x in st.q.iter().chain(st.v.iter()).chain(st.p.iter()) {
                assert!(x.is_finite(), "non-finite state");
            }
            for j in [LB, LD, RB, RD] {
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

/// Corner soaks at the production step (h = 1/240: 30 Hz × 8) and at
/// 1/480: bounded, finite, no pivot-floor or envelope engagements, and
/// the wrist rates stay under their caps — with the tendon, and no
/// worse than the same traffic without it.
#[test]
fn tendon_corner_soak() {
    static NO_TENDON: std::sync::OnceLock<Model> = std::sync::OnceLock::new();
    let bare: &Model = NO_TENDON.get_or_init(|| {
        let mut m = HUMANOID;
        m.passive_wrist = None;
        m
    });
    for (h, label) in [(H_SUB, "1/240"), (H_SUB * 0.5, "1/480")] {
        let with = corner_soak(MODEL, h, 60.0, 0x7E4D_0003);
        let without = corner_soak(bare, h, 60.0, 0x7E4D_0003);
        let pct = |x: u64| 100.0 * x as Real / (2 * with.substeps) as Real;
        println!("soak h={label} ({} substeps, 60 s): zone {:.1}%  beyond r=1 {:.1}%  \
                  max r {:.3}  max |tau_p| {:.3} N*m  max |qd|/cap {:.3}  pd {}  env {}  \
                  | no-tendon: max |qd|/cap {:.3}  pd {}  env {}",
                 with.substeps, pct(with.zone), pct(with.out), with.rmax, with.tmax,
                 with.vmax, with.pd, with.env, without.vmax, without.pd, without.env);
        assert!(with.out > 0, "the soak must actually reach the corners");
        assert!(with.vmax <= 1.0 + 1e-9, "a wrist rate passed its cap");
        assert!(with.pd <= without.pd && with.env <= without.env,
                "the tendon added pivot-floor / envelope engagements");
        assert!(with.rmax < 2.7, "r beyond the one-substep overshoot bound");
    }
}

/// Env layer: the tendon is not muscle effort — a coasting wrist parked
/// in a corner pays no torque/energy and reads no [torque] — while the
/// [tendon] telemetry counts it.
#[test]
fn tendon_excluded_from_effort_and_counted_in_telemetry() {
    let gear = joint_gear_rows(5);
    let mut core = Core::new_core(100, 5);
    core.energy_coef = 1.0;
    core.set_chart_notes(Vec::new());
    core.body.reset();
    let (bhi, dhi) = (HUMANOID_LIMITS[LB].1, HUMANOID_LIMITS[LD].1);
    core.body.st.q[LB] = 0.99 * bhi;
    core.body.st.q[LD] = 0.99 * dhi;
    core.body.st.q[RB] = 0.99 * bhi;
    core.body.st.q[RD] = -0.99 * dhi;
    core.body.refresh_fk();
    let _ = core.pop_tendon();
    let _ = core.pop_torque();
    let a = [0u8; NQ];
    core.body.tick(&a, &gear);
    assert_eq!(core.body.tau_abs, 0.0, "passive torque leaked into the energy base");
    assert!(core.body.tau.iter().all(|&t| t == 0.0));
    let tk = core.body.tendon_tick;
    println!("one coasting tick from the corners: tendon_tick {tk:?}");
    assert!(tk[0] > 0.0 && tk[1] > 0.0);
    assert_eq!(tk[0].to_bits(), tk[1].to_bits(), "mirror-symmetric hands read equal");
    assert_eq!(tk[2], N_SUB as Real);
    // the Core accumulator: one step through step_inner
    core.body.reset();
    core.body.st.q[LB] = 0.99 * bhi;
    core.body.st.q[LD] = 0.99 * dhi;
    core.body.refresh_fk();
    let (r, _) = core.step_inner(&a);
    let (s, n) = core.pop_tendon();
    let (tq, tn) = core.pop_torque();
    println!("step_inner: reward {r:.4e}  tendon {s:?} over {n} substeps  torque {tq:?}/{tn}");
    assert_eq!(n, N_SUB as u64);
    assert_eq!(tq[0], 0.0);
    assert!(s[0] > 0.0 && s[2] == N_SUB as Real && s[4] > 0.0);
    // the right hand sits at the reset pose's 70 deg bend: in the zone,
    // inside the ellipse
    assert!(s[1] > 0.0 && s[3] == N_SUB as Real && s[5] == 0.0);
    assert_eq!(core.pop_tendon().1, 0, "pop zeroes");
    // the test/tooling switch (RawBeatSim.set_passive_wrist(False)):
    // same pose, tendon off -> nothing applied, nothing counted
    core.body.tendon_on = false;
    core.body.reset();
    core.body.st.q[LB] = 0.99 * bhi;
    core.body.st.q[LD] = 0.99 * dhi;
    core.body.refresh_fk();
    core.body.tick(&a, &gear);
    assert_eq!(core.body.tendon_tick, [0.0; 8]);
    assert_eq!(core.body.ws.tendon, [[0.0; 4]; 2]);
}
