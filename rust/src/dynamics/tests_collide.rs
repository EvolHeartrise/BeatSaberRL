//! P2 narrow-phase gates: segment–segment distance vs an independent
//! brute-force refinement search (incl. parallel and degenerate cases),
//! pair-table integrity, and the §2 reference-pose roster fact (min gap
//! 0.08 at thigh–thigh, no pair within the margin).

use super::collide::{
    narrow_phase, seg_seg_closest, Contact, COLLIDERS, EXCLUDED_PAIRS, MARGIN, NUM_COLLIDERS,
    PAIRS,
};
use super::fk::{add3, dot3, scale3, sub3};
use super::model::HUMANOID;
use super::testutil::Rng;
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
