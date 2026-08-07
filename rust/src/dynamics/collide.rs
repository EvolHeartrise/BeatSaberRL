//! SPEC-v10 §2 collision spec — the 12-collider roster, the 55-live-pair
//! table (row index = NORMATIVE pair id), and the capsule–capsule narrow
//! phase (segment–segment closest points, Ericson §5.1.9; a sphere is a
//! zero-length capsule).
//!
//! Every literal here is normative from SPEC-v10 §1.2 / §2 as written.
//! The branch structure of `seg_seg_closest` is FIXED — the torch/CUDA
//! mirrors (P4/P5) replicate it literally (same comparisons, same order),
//! so do not "simplify" it.

use super::fk::{add3, dot3, mat_vec, scale3, sub3};
use super::model::{
    F_CHEST, F_HEAD, F_L_HAND, F_L_ELBOW, F_L_SHANK, F_L_THIGH, F_L_UARM, F_R_HAND, F_R_ELBOW,
    F_R_SHANK, F_R_THIGH, F_R_UARM,
};
use super::Workspace;

/// Contact margin: a constraint row activates when gap < MARGIN (§2).
pub const MARGIN: f64 = 0.01;
/// Penetration slop: Baumgarte pushes only beyond it (§3 step 8).
pub const SLOP: f64 = 0.005;

/// One collider: a capsule (segment a–b, radius r) rigidly attached to a
/// body frame. Spheres are zero-length capsules (a == b).
pub struct Collider {
    pub frame: usize,  // body frame index (model.rs HUMANOID_FRAMES)
    pub a: [f64; 3],   // segment start, in the body frame
    pub b: [f64; 3],   // segment end, in the body frame
    pub radius: f64,
}

const fn col(frame: usize, a: [f64; 3], b: [f64; 3], radius: f64) -> Collider {
    Collider { frame, a, b, radius }
}

pub const NUM_COLLIDERS: usize = 12;

/// §2 collider roster, fixed normative ids 0–11. Geometry verbatim from
/// the §1.2 body table (segment endpoints in the body frame).
#[rustfmt::skip]
pub const COLLIDERS: [Collider; NUM_COLLIDERS] = [
    /*  0 torso   */ col(F_CHEST,   [0.0, 0.0, 0.0], [0.0, 0.43, 0.0],  0.15),
    /*  1 head    */ col(F_HEAD,    [0.0, 0.0, 0.0], [0.0, 0.0, 0.0],   0.09),
    /*  2 uarm_l  */ col(F_L_UARM,  [0.0, 0.0, 0.0], [0.0, 0.0, 0.35],  0.045),
    /*  3 uarm_r  */ col(F_R_UARM,  [0.0, 0.0, 0.0], [0.0, 0.0, 0.35],  0.045),
    /*  4 farm_l  */ col(F_L_ELBOW, [0.0, 0.0, 0.0], [0.0, 0.0, 0.30],  0.04),
    /*  5 farm_r  */ col(F_R_ELBOW, [0.0, 0.0, 0.0], [0.0, 0.0, 0.30],  0.04),
    /*  6 fist_l  */ col(F_L_HAND,  [0.0, 0.0, 0.06], [0.0, 0.0, 0.06], 0.05),
    /*  7 fist_r  */ col(F_R_HAND,  [0.0, 0.0, 0.06], [0.0, 0.0, 0.06], 0.05),
    /*  8 thigh_l */ col(F_L_THIGH, [0.0, 0.0, 0.0], [0.0, -0.44, 0.0], 0.06),
    /*  9 thigh_r */ col(F_R_THIGH, [0.0, 0.0, 0.0], [0.0, -0.44, 0.0], 0.06),
    /* 10 shank_l */ col(F_L_SHANK, [0.0, 0.0, 0.0], [0.0, -0.43, 0.0], 0.045),
    /* 11 shank_r */ col(F_R_SHANK, [0.0, 0.0, 0.0], [0.0, -0.43, 0.0], 0.045),
];

pub const NUM_PAIRS: usize = 55;

/// §2 live pair table, verbatim. Row index = the normative PAIR ID used
/// by the PGS row ordering and the replay contact events. Lexicographic
/// (i, j), i < j, minus the 11 excluded pairs.
#[rustfmt::skip]
pub const PAIRS: [(usize, usize); NUM_PAIRS] = [
    /*  0 */ (0, 4),  /*  1 */ (0, 5),  /*  2 */ (0, 6),  /*  3 */ (0, 7),
    /*  4 */ (0, 10), /*  5 */ (0, 11),
    /*  6 */ (1, 2),  /*  7 */ (1, 3),  /*  8 */ (1, 4),  /*  9 */ (1, 5),
    /* 10 */ (1, 6),  /* 11 */ (1, 7),  /* 12 */ (1, 8),  /* 13 */ (1, 9),
    /* 14 */ (1, 10), /* 15 */ (1, 11),
    /* 16 */ (2, 3),  /* 17 */ (2, 5),  /* 18 */ (2, 6),  /* 19 */ (2, 7),
    /* 20 */ (2, 8),  /* 21 */ (2, 9),  /* 22 */ (2, 10), /* 23 */ (2, 11),
    /* 24 */ (3, 4),  /* 25 */ (3, 6),  /* 26 */ (3, 7),  /* 27 */ (3, 8),
    /* 28 */ (3, 9),  /* 29 */ (3, 10), /* 30 */ (3, 11),
    /* 31 */ (4, 5),  /* 32 */ (4, 7),  /* 33 */ (4, 8),  /* 34 */ (4, 9),
    /* 35 */ (4, 10), /* 36 */ (4, 11),
    /* 37 */ (5, 6),  /* 38 */ (5, 8),  /* 39 */ (5, 9),  /* 40 */ (5, 10),
    /* 41 */ (5, 11),
    /* 42 */ (6, 7),  /* 43 */ (6, 8),  /* 44 */ (6, 9),  /* 45 */ (6, 10),
    /* 46 */ (6, 11),
    /* 47 */ (7, 8),  /* 48 */ (7, 9),  /* 49 */ (7, 10), /* 50 */ (7, 11),
    /* 51 */ (8, 9),  /* 52 */ (8, 11),
    /* 53 */ (9, 10),
    /* 54 */ (10, 11),
];

/// §2 excluded pairs (11) — kept as a table for the roster tests
/// (live ∪ excluded must equal C(12,2)).
#[rustfmt::skip]
pub const EXCLUDED_PAIRS: [(usize, usize); 11] = [
    (0, 1), (0, 2), (0, 3), (0, 8), (0, 9),
    (2, 4), (3, 5), (4, 6), (5, 7), (8, 10), (9, 11),
];

/// Degeneracy threshold of the segment–segment routine (squared-length
/// and denominator guard — Ericson's EPSILON).
pub const SEG_EPS: f64 = 1e-12;

// ---------------------------------------------------------------------
// R11 broad phase. Mean contact count is 0.11/tick, so ~all 55 pairs are
// computed only to be thrown away. Two CONSERVATIVE reject tests run
// before the expensive work; a pair that survives both runs the original
// code on the original values, so every emitted Contact is byte-identical
// (gated by `broad_phase_matches_dense`, which byte-compares the whole
// emitted Vec<Contact> against the pre-R11 dense body over a pose soak).
// Both predicates are written so that only a TRUE comparison rejects:
// with a NaN coordinate every comparison is false, the pair falls through
// to the dense path, and `gap < MARGIN` is false there exactly as before
// — no NaN behaviour is changed.
// ---------------------------------------------------------------------

/// Part 1 — reject on the SQUARED closest-point distance, before the √.
///
/// The narrow phase keeps a pair iff `fl(fl(√d²) − S) < MARGIN` with
/// `S = fl(r_a + r_b)`, i.e. iff d² lies below a threshold ≈ T², where
/// `T = S + MARGIN`. √ and the subtraction are monotone in d², so the
/// keep set is downward closed and a test on d² is exact up to the
/// rounding of those two operations:
///   keep ⟹ fl(√d²) < T + ulp(0.01)/2 ⟹ √d² < T·(1 + 2⁻⁵²) + 1e-18,
/// and with T ≤ 0.4 here that is √d² < T + 1e-16, i.e. d² < (T + 1e-16)².
/// So rejecting on `d² ≥ (T + 1e-9)²` can never drop a kept pair: the
/// 1e-9 slack adds 2·T·1e-9 ≥ 2e-10 to the squared threshold, six orders
/// above that 1e-16 bound and above the ≤1 ulp (≤1e-17) error of
/// evaluating the table entry itself in a different association order.
const fn dist_rej2() -> [f64; NUM_PAIRS] {
    let mut t = [0.0f64; NUM_PAIRS];
    let mut k = 0;
    while k < NUM_PAIRS {
        let (ia, ib) = PAIRS[k];
        let s = COLLIDERS[ia].radius + COLLIDERS[ib].radius + MARGIN + 1e-9;
        t[k] = s * s;
        k += 1;
    }
    t
}
/// `(r_a + r_b + MARGIN + 1e-9)²` per pair id — see `dist_rej2`.
pub const DIST_REJ2: [f64; NUM_PAIRS] = dist_rej2();

/// Part 2 — bounding-sphere radius per collider (port of CUDA K4d):
/// half the segment length plus the radius, so every point of the
/// collider SURFACE lies within `HLR[c]` of the centerline midpoint.
/// The segment is rigid, so the world half-length equals this
/// body-frame one. Literal table (no const √); `hlr_table_bounds_geometry`
/// pins each entry against the §2 collider geometry.
pub const HLR: [f64; NUM_COLLIDERS] = [
    /*  0 torso   */ 0.365, /*  1 head    */ 0.09,
    /*  2 uarm_l  */ 0.22, /*  3 uarm_r  */ 0.22,
    /*  4 farm_l  */ 0.19, /*  5 farm_r  */ 0.19,
    /*  6 fist_l  */ 0.05, /*  7 fist_r  */ 0.05,
    /*  8 thigh_l */ 0.28, /*  9 thigh_r */ 0.28,
    /* 10 shank_l */ 0.26, /* 11 shank_r */ 0.26,
];

/// Part 2 — bounding-sphere reject threshold per pair.
///
/// With `mid_c = 0.5·(wa_c + wb_c)`, every point of collider `c` is
/// within `HLR[c]` of `mid_c`, so by the triangle inequality
///     gap = seg_dist − r_a − r_b ≥ |mid_a − mid_b| − (HLR_a + HLR_b),
/// hence keep (gap < MARGIN) ⟹ |mid_a − mid_b| < U, where
/// `U = HLR_a + HLR_b + MARGIN`. Rejecting on `|Δmid|² > U²·1.02` is
/// therefore conservative with enormous room to spare: the ×1.02 admits
/// |Δmid| up to U·1.00995, i.e. ≥ 1.09e-3 of ABSOLUTE slack (the tightest
/// pair is fist–fist, U = 0.11), which swamps every error in the bound —
///   · the ≤1e-12 tolerance the HLR table is pinned to,
///   · rounding of mid, of the difference and of the dot (≤ ~1e-16 at
///     these magnitudes),
///   · the ≤1e-15 relative non-orthonormality of the FK rotation, the
///     only thing that can make the world segment longer than the body
///     segment whose half-length HLR was built from.
const fn mid_rej2() -> [f64; NUM_PAIRS] {
    let mut t = [0.0f64; NUM_PAIRS];
    let mut k = 0;
    while k < NUM_PAIRS {
        let (ia, ib) = PAIRS[k];
        let u = HLR[ia] + HLR[ib] + MARGIN;
        t[k] = u * u * 1.02;
        k += 1;
    }
    t
}
/// `(HLR_a + HLR_b + MARGIN)²·1.02` per pair id — see `mid_rej2`.
pub const MID_REJ2: [f64; NUM_PAIRS] = mid_rej2();

/// Closest points between segments p1–q1 and p2–q2 (Ericson, "Real-Time
/// Collision Detection", §5.1.9 `ClosestPtSegmentSegment`), returning
/// (s, t, c1, c2, dist²). The parameter-space objective is convex, so
/// the clamped stationary point is the global minimum. Branch order is
/// FIXED (mirrored literally in torch/CUDA):
///   1. both degenerate  2. first degenerate  3. second degenerate
///   4. general (incl. parallel: denom == 0 → s = 0).
#[allow(clippy::many_single_char_names)]
pub fn seg_seg_closest(
    p1: [f64; 3],
    q1: [f64; 3],
    p2: [f64; 3],
    q2: [f64; 3],
) -> (f64, f64, [f64; 3], [f64; 3], f64) {
    let d1 = sub3(q1, p1);
    let d2 = sub3(q2, p2);
    let r = sub3(p1, p2);
    let a = dot3(d1, d1);
    let e = dot3(d2, d2);
    let f = dot3(d2, r);
    let (s, t);
    if a <= SEG_EPS && e <= SEG_EPS {
        // both segments degenerate to points
        s = 0.0;
        t = 0.0;
    } else if a <= SEG_EPS {
        // first segment degenerates to a point
        s = 0.0;
        t = (f / e).clamp(0.0, 1.0);
    } else {
        let c = dot3(d1, r);
        if e <= SEG_EPS {
            // second segment degenerates to a point
            t = 0.0;
            s = (-c / a).clamp(0.0, 1.0);
        } else {
            // general non-degenerate case
            let b = dot3(d1, d2);
            let denom = a * e - b * b;
            // parallel (denom == 0): pick s = 0, then t from the
            // closest point on segment 2 — deterministic and optimal
            // (the valley is flat along the common direction)
            let s0 = if denom > SEG_EPS {
                ((b * f - c * e) / denom).clamp(0.0, 1.0)
            } else {
                0.0
            };
            let t0 = b * s0 + f;
            if t0 < 0.0 {
                t = 0.0;
                s = (-c / a).clamp(0.0, 1.0);
            } else if t0 > e {
                t = 1.0;
                s = ((b - c) / a).clamp(0.0, 1.0);
            } else {
                t = t0 / e;
                s = s0;
            }
        }
    }
    let c1 = add3(p1, scale3(d1, s));
    let c2 = add3(p2, scale3(d2, t));
    let d = sub3(c1, c2);
    (s, t, c1, c2, dot3(d, d))
}

/// One active contact (gap < MARGIN) — the closest-point pair of a live
/// collider pair, one normal row (§2).
#[derive(Clone, Copy)]
pub struct Contact {
    /// Normative pair id (row index in PAIRS).
    pub pair: usize,
    /// World contact point: midpoint of the two surface points.
    pub point: [f64; 3],
    /// Unit normal, from collider PAIRS[pair].0 toward PAIRS[pair].1
    /// (positive relative velocity along it = separating).
    pub normal: [f64; 3],
    /// Signed gap = centerline distance − radius sum (< 0 penetrating).
    pub gap: f64,
}

/// Narrow phase over the 55-pair table in PAIR-ID ORDER (normative —
/// the PGS row order and the replay contact stream depend on it).
/// Requires fk_all in ws. Appends active contacts to `out` (cleared
/// here; capacity NUM_PAIRS is reserved by the caller — no allocation
/// inside a substep).
pub fn narrow_phase(ws: &Workspace, out: &mut Vec<Contact>) {
    out.clear();
    // Collider segment endpoints to world (body frame pose from FK),
    // ONCE per collider instead of once per pair-and-side: the 55 pairs
    // reference only 12 colliders, so the old inline form recomputed
    // each of the 24 endpoint transforms ~9× per substep. Bitwise: the
    // expression per endpoint is character-identical, only hoisted, so
    // every p1/q1/p2/q2 below is the same f64 it was before.
    let mut wa = [[0.0f64; 3]; NUM_COLLIDERS];
    let mut wb = [[0.0f64; 3]; NUM_COLLIDERS];
    // bounding-sphere centres for the R11 part-2 reject: 36 mul+add for
    // the whole roster, against 55 segment–segment solves
    let mut mid = [[0.0f64; 3]; NUM_COLLIDERS];
    for (ci, c) in COLLIDERS.iter().enumerate() {
        wa[ci] = add3(ws.p[c.frame], mat_vec(&ws.r[c.frame], c.a));
        wb[ci] = add3(ws.p[c.frame], mat_vec(&ws.r[c.frame], c.b));
        for k in 0..3 {
            mid[ci][k] = 0.5 * (wa[ci][k] + wb[ci][k]);
        }
    }
    for (pid, &(ia, ib)) in PAIRS.iter().enumerate() {
        // R11 part 2: conservative bounding-sphere reject (see MID_REJ2)
        let dm = sub3(mid[ia], mid[ib]);
        if dot3(dm, dm) > MID_REJ2[pid] {
            continue;
        }
        let ca = &COLLIDERS[ia];
        let cb = &COLLIDERS[ib];
        let (p1, q1) = (wa[ia], wb[ia]);
        let (p2, q2) = (wa[ib], wb[ib]);
        let (_s, _t, c1, c2, d2) = seg_seg_closest(p1, q1, p2, q2);
        // R11 part 1: conservative reject before the √ (see DIST_REJ2)
        if d2 >= DIST_REJ2[pid] {
            continue;
        }
        let dist = d2.sqrt();
        let gap = dist - (ca.radius + cb.radius);
        if gap < MARGIN {
            // normal from collider ia toward ib; centerline coincidence
            // (dist ~ 0) cannot arise for capsules whose radii already
            // overlap by margin activation — fixed +y fallback keeps the
            // branch deterministic anyway
            let n = if dist > 1e-9 {
                scale3(sub3(c2, c1), 1.0 / dist)
            } else {
                [0.0, 1.0, 0.0]
            };
            // surface points: c1 + n·r_a and c2 − n·r_b; report the
            // midpoint (single contact point per pair, §2)
            let s1 = add3(c1, scale3(n, ca.radius));
            let s2 = sub3(c2, scale3(n, cb.radius));
            let point = scale3(add3(s1, s2), 0.5);
            out.push(Contact { pair: pid, point, normal: n, gap });
        }
    }
}

/// Pre-R11 narrow phase, verbatim: every pair goes through
/// `seg_seg_closest` + √ + the margin test, no broad phase. Kept as the
/// reference the R11 rejects are gated against — `broad_phase_matches_dense`
/// byte-compares the emitted `Vec<Contact>` of the two over a pose soak.
#[cfg(test)]
pub fn narrow_phase_dense(ws: &Workspace, out: &mut Vec<Contact>) {
    out.clear();
    let mut wa = [[0.0f64; 3]; NUM_COLLIDERS];
    let mut wb = [[0.0f64; 3]; NUM_COLLIDERS];
    for (ci, c) in COLLIDERS.iter().enumerate() {
        wa[ci] = add3(ws.p[c.frame], mat_vec(&ws.r[c.frame], c.a));
        wb[ci] = add3(ws.p[c.frame], mat_vec(&ws.r[c.frame], c.b));
    }
    for (pid, &(ia, ib)) in PAIRS.iter().enumerate() {
        let ca = &COLLIDERS[ia];
        let cb = &COLLIDERS[ib];
        let (p1, q1) = (wa[ia], wb[ia]);
        let (p2, q2) = (wa[ib], wb[ib]);
        let (_s, _t, c1, c2, d2) = seg_seg_closest(p1, q1, p2, q2);
        let dist = d2.sqrt();
        let gap = dist - (ca.radius + cb.radius);
        if gap < MARGIN {
            let n = if dist > 1e-9 {
                scale3(sub3(c2, c1), 1.0 / dist)
            } else {
                [0.0, 1.0, 0.0]
            };
            let s1 = add3(c1, scale3(n, ca.radius));
            let s2 = sub3(c2, scale3(n, cb.radius));
            let point = scale3(add3(s1, s2), 0.5);
            out.push(Contact { pair: pid, point, normal: n, gap });
        }
    }
}
