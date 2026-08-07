//! P2 narrow-phase gates: segment–segment distance vs an independent
//! brute-force refinement search (incl. parallel and degenerate cases),
//! pair-table integrity, and the §2 reference-pose roster fact (min gap
//! 0.08 at thigh–thigh, no pair within the margin).

use super::collide::{
    narrow_phase, narrow_phase_dense, seg_seg_closest, Contact, COLLIDERS, EXCLUDED_PAIRS, HLR,
    MARGIN, NUM_COLLIDERS, PAIRS,
};
use super::fk::{add3, dot3, scale3, sub3};
use super::model::HUMANOID;
use super::testutil::{random_state, Rng};
use super::{State, Workspace};

fn dist_at(p1: [f64; 3], d1: [f64; 3], p2: [f64; 3], d2: [f64; 3], s: f64, t: f64) -> f64 {
    let c1 = add3(p1, scale3(d1, s));
    let c2 = add3(p2, scale3(d2, t));
    let d = sub3(c1, c2);
    dot3(d, d).sqrt()
}

/// Independent verification of the closest-distance routine on 10k
/// random pairs (incl. parallel and degenerate). Two parts:
///   1. brute-force sampling UPPER bound: the analytic distance must
///      never exceed any of 33×33 grid samples (catches gross errors /
///      wrong minima; pure grid refinement cannot certify 1e-9 — it
///      stagnates on anisotropic near-parallel valleys — so precision
///      comes from part 2);
///   2. KKT certificate at 1e-9: f(s,t) = |c1 − c2|² is convex, so
///      box-KKT stationarity of the returned (s, t) — computed from
///      independent gradient formulas ∂f/∂s = 2·d1·(c1−c2), ∂f/∂t =
///      −2·d2·(c1−c2) — proves the GLOBAL minimum.
#[test]
fn seg_seg_vs_brute_force() {
    let mut rng = Rng(0x5EED_0000_0000_0010);
    let mut worst_kkt = 0.0f64;
    for case in 0..10_000 {
        let mut p = [[0.0f64; 3]; 4];
        for v in p.iter_mut() {
            for x in v.iter_mut() {
                *x = rng.range(-1.0, 1.0);
            }
        }
        let [p1, mut q1, p2, mut q2] = p;
        match case % 10 {
            // 10%: parallel segments (q2 along d1 from p2)
            7 => {
                let d1 = sub3(q1, p1);
                q2 = add3(p2, scale3(d1, rng.range(0.2, 1.5)));
            }
            // 10%: first degenerate (point vs segment)
            8 => q1 = p1,
            // 10%: both degenerate (point vs point)
            9 => {
                q1 = p1;
                q2 = p2;
            }
            _ => {}
        }
        let (s, t, c1, c2, d2) = seg_seg_closest(p1, q1, p2, q2);
        let analytic = d2.sqrt();
        let d1 = sub3(q1, p1);
        let d2v = sub3(q2, p2);
        // 1. sampling upper bound
        for i in 0..33 {
            for j in 0..33 {
                let ss = i as f64 / 32.0;
                let tt = j as f64 / 32.0;
                let d = dist_at(p1, d1, p2, d2v, ss, tt);
                assert!(
                    analytic <= d + 1e-12,
                    "case {case}: sample ({ss},{tt}) beats the analytic minimum"
                );
            }
        }
        // 2. box-KKT at the returned parameters (scale-normalized)
        let diff = sub3(c1, c2);
        let gs = 2.0 * dot3(d1, diff);
        let gt = -2.0 * dot3(d2v, diff);
        let scale = 1.0 + dot3(d1, d1).max(dot3(d2v, d2v));
        let viol = |g: f64, x: f64| -> f64 {
            // interior: |g| = 0; at 0: g ≥ 0; at 1: g ≤ 0
            if x <= 0.0 {
                (-g).max(0.0)
            } else if x >= 1.0 {
                g.max(0.0)
            } else {
                g.abs()
            }
        };
        let kkt = (viol(gs, s) + viol(gt, t)) / scale;
        worst_kkt = worst_kkt.max(kkt);
        assert!(
            kkt <= 1e-9,
            "case {case}: KKT violation {kkt:e} at (s={s}, t={t})"
        );
    }
    println!("seg-seg worst normalized KKT violation = {worst_kkt:e}");
}

/// R11 part 2 pins the HLR table against the §2 geometry: each entry
/// must be an UPPER bound of half the segment length plus the radius
/// (conservativeness — the reject argument in `mid_rej2` allows the
/// table to overshoot by up to 1e-12), and must not overshoot by more
/// than that (tightness — an inflated entry would silently disable the
/// reject instead of failing loudly).
#[test]
fn hlr_table_bounds_geometry() {
    for (ci, c) in COLLIDERS.iter().enumerate() {
        let d = sub3(c.b, c.a);
        let geom = 0.5 * dot3(d, d).sqrt() + c.radius;
        assert!(
            HLR[ci] >= geom - 1e-12,
            "collider {ci}: HLR {} is below the geometric bound {geom}",
            HLR[ci]
        );
        assert!(
            HLR[ci] <= geom + 1e-12,
            "collider {ci}: HLR {} overshoots the geometric bound {geom}",
            HLR[ci]
        );
    }
}

/// Pose on the straight line between two states (root translation,
/// joint angles; quaternion lerped then renormalized).
fn lerp_state(a: &State, b: &State, u: f64) -> State {
    let mut st = a.clone();
    for i in 0..3 {
        st.p[i] = a.p[i] + u * (b.p[i] - a.p[i]);
    }
    let mut qt = [0.0f64; 4];
    for i in 0..4 {
        qt[i] = a.quat[i] + u * (b.quat[i] - a.quat[i]);
    }
    let n = (qt[0] * qt[0] + qt[1] * qt[1] + qt[2] * qt[2] + qt[3] * qt[3]).sqrt();
    if n > 0.1 {
        st.quat = [qt[0] / n, qt[1] / n, qt[2] / n, qt[3] / n];
    }
    for j in 0..a.q.len() {
        st.q[j] = a.q[j] + u * (b.q[j] - a.q[j]);
    }
    st
}

/// Dense gap of one pair from a posed workspace (the pre-R11 formula).
fn pair_gap(ws: &Workspace, pid: usize) -> f64 {
    let (ia, ib) = PAIRS[pid];
    let (ca, cb) = (&COLLIDERS[ia], &COLLIDERS[ib]);
    let p1 = add3(ws.p[ca.frame], super::fk::mat_vec(&ws.r[ca.frame], ca.a));
    let q1 = add3(ws.p[ca.frame], super::fk::mat_vec(&ws.r[ca.frame], ca.b));
    let p2 = add3(ws.p[cb.frame], super::fk::mat_vec(&ws.r[cb.frame], cb.a));
    let q2 = add3(ws.p[cb.frame], super::fk::mat_vec(&ws.r[cb.frame], cb.b));
    let (_, _, _, _, d2) = seg_seg_closest(p1, q1, p2, q2);
    d2.sqrt() - (ca.radius + cb.radius)
}

/// R11 gate: the broad-phase rejects must be invisible. Byte-compares
/// the whole emitted `Vec<Contact>` (length, pair ids, and the RAW BIT
/// PATTERNS of point/normal/gap) against the pre-R11 dense body.
///
/// Corpus: 200 random in-limit poses joined by 32-step linear sweeps
/// (gaps move continuously along a sweep, so every pair is seen at many
/// separations), PLUS — and this is what gives the test its power — a
/// bisection onto every `gap = MARGIN` crossing found on those sweeps.
/// A crossing pose sits within ~1e-13 of the exact keep/reject boundary,
/// i.e. deep inside the 1e-9 reject slack, so a slack that is too tight
/// (or the wrong sign) drops a contact HERE. Verified: flipping part 1's
/// slack to −1e-6 leaves the plain sweep green and fails this test.
#[test]
fn broad_phase_matches_dense() {
    let m = &HUMANOID;
    let mut ws = Workspace::new(m);
    let mut rng = Rng(0x5EED_0000_0000_0B11);
    let mut fast: Vec<Contact> = Vec::with_capacity(PAIRS.len());
    let mut slow: Vec<Contact> = Vec::with_capacity(PAIRS.len());
    let bits = |c: &Contact| {
        (
            c.pair,
            c.point.map(f64::to_bits),
            c.normal.map(f64::to_bits),
            c.gap.to_bits(),
        )
    };
    let mut states = 0usize;
    let mut emitted = 0usize;
    let mut boundary = 0usize;
    let mut worst_boundary = f64::INFINITY;
    // one byte-compare at the current ws pose
    macro_rules! check {
        ($what:expr) => {{
            narrow_phase(&ws, &mut fast);
            narrow_phase_dense(&ws, &mut slow);
            assert_eq!(
                fast.len(),
                slow.len(),
                "{}: broad phase emitted {:?}, dense {:?}",
                $what,
                fast.iter().map(|c| c.pair).collect::<Vec<_>>(),
                slow.iter().map(|c| c.pair).collect::<Vec<_>>(),
            );
            for (cf, cs) in fast.iter().zip(slow.iter()) {
                assert_eq!(bits(cf), bits(cs), "{}: pair {} differs bitwise", $what, cs.pair);
            }
            states += 1;
            emitted += slow.len();
        }};
    }
    let mut a = random_state(m, &mut rng, false);
    for seg in 0..200 {
        let b = random_state(m, &mut rng, false);
        let mut prev = [f64::NAN; PAIRS.len()];
        for k in 0..=32u32 {
            let u = f64::from(k) / 32.0;
            super::fk::fk_all(m, &lerp_state(&a, &b, u), &mut ws);
            check!(format!("seg {seg} u {u}"));
            let mut cur = [0.0f64; PAIRS.len()];
            for (pid, g) in cur.iter_mut().enumerate() {
                *g = pair_gap(&ws, pid);
            }
            // bisect every margin crossing on this sub-interval
            if k > 0 {
                for pid in 0..PAIRS.len() {
                    let (g0, g1) = (prev[pid] - MARGIN, cur[pid] - MARGIN);
                    if !(g0 < 0.0) == !(g1 < 0.0) || !g0.is_finite() || !g1.is_finite() {
                        continue;
                    }
                    let (mut lo, mut hi) = (f64::from(k - 1) / 32.0, u);
                    let lo_neg = g0 < 0.0;
                    for _ in 0..46 {
                        let mid = 0.5 * (lo + hi);
                        super::fk::fk_all(m, &lerp_state(&a, &b, mid), &mut ws);
                        if (pair_gap(&ws, pid) - MARGIN < 0.0) == lo_neg {
                            lo = mid;
                        } else {
                            hi = mid;
                        }
                    }
                    for uu in [lo, hi] {
                        super::fk::fk_all(m, &lerp_state(&a, &b, uu), &mut ws);
                        worst_boundary = worst_boundary.min((pair_gap(&ws, pid) - MARGIN).abs());
                        check!(format!("seg {seg} pair {pid} boundary u {uu}"));
                    }
                    boundary += 1;
                }
            }
            prev = cur;
        }
        a = b;
    }
    println!(
        "broad-phase soak: {states} poses ({boundary} margin crossings bisected, \
         closest |gap−MARGIN| = {worst_boundary:e}), {emitted} contacts emitted"
    );
    // power checks: the corpus must produce contacts AND land on the
    // keep/reject boundary, otherwise the byte comparison is vacuous
    assert!(emitted > 1000, "soak corpus too tame: only {emitted} contacts");
    assert!(boundary > 200, "too few margin crossings: {boundary}");
    assert!(worst_boundary < 1e-11, "bisection did not reach the boundary");
}

#[test]
fn pair_table_integrity() {
    // 55 live + 11 excluded = C(12,2); i < j everywhere; live table in
    // strictly increasing lexicographic order (= pair-id order)
    assert_eq!(PAIRS.len() + EXCLUDED_PAIRS.len(), 66);
    for w in PAIRS.windows(2) {
        assert!(w[0] < w[1], "pair table not lexicographic: {:?}", w);
    }
    for &(i, j) in PAIRS.iter().chain(EXCLUDED_PAIRS.iter()) {
        assert!(i < j && j < NUM_COLLIDERS);
    }
    let mut all: Vec<(usize, usize)> = PAIRS
        .iter()
        .chain(EXCLUDED_PAIRS.iter())
        .copied()
        .collect();
    all.sort();
    all.dedup();
    assert_eq!(all.len(), 66, "duplicate pair between live and excluded");
    let mut expect = Vec::new();
    for i in 0..NUM_COLLIDERS {
        for j in (i + 1)..NUM_COLLIDERS {
            expect.push((i, j));
        }
    }
    assert_eq!(all, expect, "live ∪ excluded != C(12,2)");
}

#[test]
fn neutral_pose_roster() {
    // §2: at the reference pose no live pair is within the margin;
    // verified min gap 0.08 (thigh–thigh, pair id 51)
    let m = &HUMANOID;
    let mut ws = Workspace::new(m);
    let st = State::neutral(m);
    super::fk::fk_all(m, &st, &mut ws);
    let mut contacts: Vec<Contact> = Vec::with_capacity(PAIRS.len());
    narrow_phase(&ws, &mut contacts);
    assert!(
        contacts.is_empty(),
        "live pairs within margin at neutral: {:?}",
        contacts.iter().map(|c| c.pair).collect::<Vec<_>>()
    );
    // recompute all gaps directly for the min-gap fact
    let mut min_gap = f64::INFINITY;
    let mut min_pid = usize::MAX;
    for (pid, &(ia, ib)) in PAIRS.iter().enumerate() {
        let (ca, cb) = (&COLLIDERS[ia], &COLLIDERS[ib]);
        let p1 = add3(ws.p[ca.frame], super::fk::mat_vec(&ws.r[ca.frame], ca.a));
        let q1 = add3(ws.p[ca.frame], super::fk::mat_vec(&ws.r[ca.frame], ca.b));
        let p2 = add3(ws.p[cb.frame], super::fk::mat_vec(&ws.r[cb.frame], cb.a));
        let q2 = add3(ws.p[cb.frame], super::fk::mat_vec(&ws.r[cb.frame], cb.b));
        let (_, _, _, _, d2) = seg_seg_closest(p1, q1, p2, q2);
        let gap = d2.sqrt() - (ca.radius + cb.radius);
        if gap < min_gap {
            min_gap = gap;
            min_pid = pid;
        }
    }
    println!("neutral min gap {min_gap:.6} at pair {min_pid}");
    assert!(min_gap > MARGIN);
    assert_eq!(min_pid, 51, "min-gap pair should be thigh–thigh (8,9)");
    assert!((min_gap - 0.08).abs() < 1e-9, "min gap {min_gap} != 0.08");
}
