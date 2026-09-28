//! Judge-shared geometry, constants and the chart object record.
//!
//! This file is compiled into TWO crates: the Python env crate (lib.rs,
//! `mod geom`) and the Flux WebXR game's wasm crate
//! (`rust/flux-wasm`, via `#[path]`), together with `flux.rs` and
//! `combo.rs`. Everything the Flux judge touches lives in these three
//! files, so the game in the browser and the env the agent trains on run
//! the SAME source — there is no port to drift. Keep them free of pyo3,
//! the body, and anything else the wasm crate does not have: they may
//! only use `crate::Real` and each other.

use crate::Real;

// --- tick, scoring and judgment constants -----------------------------------
pub(crate) const DT: Real = 1.0 / 30.0;
pub(crate) const ACC_NORM: Real = 0.3; // real BS accuracy normalizer (m)
pub(crate) const COS_TOL: Real = 0.5;
pub(crate) const REWARD_BAD: Real = -0.5;
pub(crate) const REWARD_MISS: Real = -0.5;
pub(crate) const SCORE_MAX: Real = 115.0;

// good-box depth convention (env v7 real BS hitboxes; Flux circles and
// bars reuse it for their slab)
pub(crate) const GOOD_HALF_Z: Real = 0.5;
pub(crate) const GOOD_Z_SHIFT: Real = -0.25; // box center = note_z + shift (toward player)

// miss boundary: note consumed as missed past the LIVE head plane
// z < head_z(t) (v9's -Z_MISS = -0.65 at the reference pose). The
// episode tail must cover the slowest object's trip from the hit plane
// to the WORST-CASE head plane; the head cannot physically reach
// z < -1.5 on the standing body (waist z -0.65, head lever 1.09 m
// fully folded back still > -1.8, and the spine/hip limits stop far
// short; a body that falls further terminates on FALL_ROOT_Y first) —
// P3 divergence note, SPEC-v10 Divergences §18.
pub(crate) const Z_MISS_TAIL: Real = 1.5;

// object kinds (mirrors constants.py)
pub(crate) const KIND_NOTE: u8 = 0;
pub(crate) const KIND_BOMB: u8 = 1;
pub(crate) const KIND_CHAIN_HEAD: u8 = 2;
pub(crate) const KIND_CHAIN_LINK: u8 = 3;
// Flux mode objects (flux.rs; never appear in a standard chart)
pub(crate) const KIND_CIRCLE: u8 = 4;
pub(crate) const KIND_BAR: u8 = 5;

// env v15 (contact spec §10): the note track is head-anchored in z —
// every note's world z carries the offset head_z − PLANE_Z0 (zero at
// the reference pose). Real-BS room semantics: walking toward the
// notes shifts them away by exactly the walked amount, so approach
// distance (and therefore hit/miss timing) is invariant to root z.
// Applied at every note-z evaluation: obs lane, broad-phase bound,
// judgment. x/y stay world-fixed (lanes are body-independent).
pub(crate) const PLANE_Z0: Real = -0.65; // reference body plane (obs centering)

/// Slowest object speed (infinite for an empty chart): with a sorted
/// chart, `(hit_time - t) * min_speed + z_off > reach_z` is an exact
/// "this and every later object is beyond reach" bound.
pub(crate) fn chart_min_speed(chart: &[Note]) -> Real {
    chart.iter().map(|n| n.speed).fold(Real::INFINITY, Real::min)
}

/// The episode tail after the last hit_time for a chart whose slowest
/// object moves at `min_speed` (infinite for an empty chart): it must
/// cover that object's trip from the plane to the WORST-CASE miss
/// boundary (live head plane, env v10), or final misses never register.
pub(crate) fn chart_tail_s(min_speed: Real) -> Real {
    if min_speed.is_finite() {
        0.5 + Z_MISS_TAIL / min_speed
    } else {
        0.5
    }
}

/// Advance the low-water cursor past the resolved prefix of `status`
/// (0 = pending; the scan loops start there).
pub(crate) fn advance_first_pending(status: &[u8], first_pending: &mut usize) {
    while *first_pending < status.len() && status[*first_pending] != 0 {
        *first_pending += 1;
    }
}

/// The chart part of the episode-end rule (the body's fall / tilt /
/// escape disjuncts are the env's own): every object resolved, no
/// follow-through pending and the last hit time passed — or the tail
/// ran out, or the tick budget did.
#[allow(clippy::too_many_arguments)]
pub(crate) fn chart_done(pending: bool, no_trackers: bool, chart_empty: bool,
                         time: Real, last_hit: Real, tail_s: Real,
                         tick: u32, max_steps: u32) -> bool {
    (!pending && no_trackers && (chart_empty || time > last_hit))
        || time > last_hit + tail_s
        || tick >= max_steps
}

/// Real BS scores are integers; explicit floor(x + 0.5) so every
/// implementation (Rust, torch, Python tools) agrees on halves.
pub(crate) fn round_pts(x: Real) -> Real {
    (x + 0.5).floor()
}

// ---- deterministic sin / cos ---------------------------------------------
// `f64::sin` is the PLATFORM libm: glibc for the Linux env, Rust's musl
// port for wasm32 — they disagree in the last bit for ~6% of angles, which
// would let the WebXR game (rust/flux-wasm) and the env install a bar with
// different sin/cos. Every judge-shared angle goes through this pure-Rust
// copy of musl's algorithm instead (medium-range Cody-Waite reduction +
// the __sin/__cos kernels, < 1 ulp): plain f64 arithmetic, no FMA
// contraction in Rust, so every build computes the same bits.

const TOINT: f64 = 1.5 / f64::EPSILON;
const PIO4: f64 = 0.785_398_163_397_448_279; // 0x1.921fb54442d18p-1
const INVPIO2: f64 = 6.36619772367581382433e-01;
const PIO2_1: f64 = 1.57079632673412561417e+00;
const PIO2_1T: f64 = 6.07710050650619224932e-11;
const PIO2_2: f64 = 6.07710050630396597660e-11;
const PIO2_2T: f64 = 2.02226624879595063154e-21;
const PIO2_3: f64 = 2.02226624871116645580e-21;
const PIO2_3T: f64 = 8.47842766036889956997e-32;

fn k_sin(x: f64, y: f64, iy: bool) -> f64 {
    const S1: f64 = -1.66666666666666324348e-01;
    const S2: f64 = 8.33333333332248946124e-03;
    const S3: f64 = -1.98412698298579493134e-04;
    const S4: f64 = 2.75573137070700676789e-06;
    const S5: f64 = -2.50507602534068634195e-08;
    const S6: f64 = 1.58969099521155010221e-10;
    let z = x * x;
    let w = z * z;
    let r = S2 + z * (S3 + z * S4) + z * w * (S5 + z * S6);
    let v = z * x;
    if !iy {
        x + v * (S1 + z * r)
    } else {
        x - ((z * (0.5 * y - v * r) - y) - v * S1)
    }
}

fn k_cos(x: f64, y: f64) -> f64 {
    const C1: f64 = 4.16666666666666019037e-02;
    const C2: f64 = -1.38888888888741095749e-03;
    const C3: f64 = 2.48015872894767294178e-05;
    const C4: f64 = -2.75573143513906633035e-07;
    const C5: f64 = 2.08757232129817482790e-09;
    const C6: f64 = -1.13596475577881948265e-11;
    let z = x * x;
    let w = z * z;
    let r = z * (C1 + z * (C2 + z * C3)) + w * w * (C4 + z * (C5 + z * C6));
    let hz = 0.5 * z;
    let w = 1.0 - hz;
    w + (((1.0 - w) - hz) + (z * r - x * y))
}

/// x = n·π/2 + (y0 + y1), |y0 + y1| <= π/4 — musl's medium-range path,
/// valid for |x| < 2^20·π/2 (callers fold larger |x| first).
fn rem_pio2(x: f64) -> (i32, f64, f64) {
    let ex = ((x.to_bits() >> 52) & 0x7ff) as i32;
    let mut f = x * INVPIO2 + TOINT - TOINT;
    let mut n = f as i32;
    let mut r = x - f * PIO2_1;
    let mut w = f * PIO2_1T;
    if r - w < -PIO4 {
        n -= 1;
        f -= 1.0;
        r = x - f * PIO2_1;
        w = f * PIO2_1T;
    } else if r - w > PIO4 {
        n += 1;
        f += 1.0;
        r = x - f * PIO2_1;
        w = f * PIO2_1T;
    }
    let mut y0 = r - w;
    let ey = ((y0.to_bits() >> 52) & 0x7ff) as i32;
    if ex - ey > 16 {
        let t = r;
        w = f * PIO2_2;
        r = t - w;
        w = f * PIO2_2T - ((t - r) - w);
        y0 = r - w;
        let ey = ((y0.to_bits() >> 52) & 0x7ff) as i32;
        if ex - ey > 49 {
            let t = r;
            w = f * PIO2_3;
            r = t - w;
            w = f * PIO2_3T - ((t - r) - w);
            y0 = r - w;
        }
    }
    (n, y0, (r - y0) - w)
}

/// (sin x, cos x), bit-identical on every target (see above).
#[cfg_attr(feature = "f32", allow(dead_code))]
pub(crate) fn det_sin_cos(x: f64) -> (f64, f64) {
    if !x.is_finite() {
        return (x - x, x - x);
    }
    // fold huge angles with the exact IEEE remainder (never in a real chart)
    let x = if x.abs() >= 1.0e6 { x % std::f64::consts::TAU } else { x };
    if x.abs() <= PIO4 {
        return (k_sin(x, 0.0, false), k_cos(x, 0.0));
    }
    let (n, y0, y1) = rem_pio2(x);
    let (s, c) = (k_sin(y0, y1, true), k_cos(y0, y1));
    match n & 3 {
        0 => (s, c),
        1 => (c, -s),
        2 => (-s, -c),
        _ => (-c, s),
    }
}

// 3-vector ops (dynamics::fk re-exports these for the physics)
pub fn sub3(a: [Real; 3], b: [Real; 3]) -> [Real; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

pub fn dot3(a: [Real; 3], b: [Real; 3]) -> Real {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

pub fn cross3(a: [Real; 3], b: [Real; 3]) -> [Real; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

#[derive(Clone, Copy)]
pub(crate) struct Note {
    pub(crate) hit_time: Real,
    pub(crate) x: Real, // world position at the hit plane (meters)
    pub(crate) y: Real,
    pub(crate) color: u8,
    // required swing direction (0 = +x, CCW), stored as its sin/cos pair
    pub(crate) sin_a: Real,
    pub(crate) cos_a: Real,
    pub(crate) speed: Real,     // approach speed toward the plane (m/s)
    pub(crate) any_dir: bool,  // dot note / chain link: any swing direction scores
    pub(crate) kind: u8,       // KIND_NOTE / _BOMB / _CHAIN_HEAD / _CHAIN_LINK
    pub(crate) free_pre: bool, // arc tail: full 70 pre-swing granted
    pub(crate) free_post: bool, // arc head: full 30 follow-through granted
    // Flux bars only (0 elsewhere): length (m) and sweet-spot offset
    // in [-1, 1] along the bar axis (sin_a/cos_a carry its rotation)
    pub(crate) len: Real,
    pub(crate) sweet: Real,
}

/// Triangle vs axis-aligned box (Akenine-Möller SAT). Verts are given
/// RELATIVE to the box center; h = half extents.
pub(crate) fn tri_box_overlap(v: [[Real; 3]; 3], h: [Real; 3]) -> bool {
    // box-axis tests
    for k in 0..3 {
        let lo = v[0][k].min(v[1][k]).min(v[2][k]);
        let hi = v[0][k].max(v[1][k]).max(v[2][k]);
        if lo > h[k] || hi < -h[k] {
            return false;
        }
    }
    let e = [sub3(v[1], v[0]), sub3(v[2], v[1]), sub3(v[0], v[2])];
    // 9 cross-axis tests: a = u_k x e_j (u_k = box axes)
    for j in 0..3 {
        let ax = [
            [0.0, -e[j][2], e[j][1]],
            [e[j][2], 0.0, -e[j][0]],
            [-e[j][1], e[j][0], 0.0],
        ];
        for a in ax {
            let p0 = dot3(v[0], a);
            let p1 = dot3(v[1], a);
            let p2 = dot3(v[2], a);
            let r = h[0] * a[0].abs() + h[1] * a[1].abs() + h[2] * a[2].abs();
            if p0.min(p1).min(p2) > r || p0.max(p1).max(p2) < -r {
                return false;
            }
        }
    }
    // triangle-plane test
    let n = cross3(e[0], e[1]);
    let r = h[0] * n[0].abs() + h[1] * n[1].abs() + h[2] * n[2].abs();
    dot3(n, v[0]).abs() <= r
}

// Env v7.4: hitboxes ROTATE with the object about the approach (z)
// axis by rot = angle - 270 deg. Env v7.6: plain notes additionally
// billboard toward the player's head as they approach; env v10 the
// head anchor is the LIVE FK head position (SPEC-v10 §4).
pub(crate) const LOOK_VERT: Real = 0.8; // verticalLookStrength: head.y 80% toward note.y
pub(crate) const LOOK_DIST: Real = 8.0; // full head-facing at z=0, none beyond 8 m

/// Orthonormal note frame columns (x', y', z'): z' = the blended
/// forward, up from the z-roll, x' = up x z'. `head` is the live FK
/// head position (env v10).
pub(crate) fn note_frame(
    p: [Real; 3],
    cos_r: Real,
    sin_r: Real,
    lam: Real,
    head: [Real; 3],
) -> [[Real; 3]; 3] {
    let up = [-sin_r, cos_r, 0.0];
    let hp = [head[0], head[1] + LOOK_VERT * (p[1] - head[1]), head[2]];
    let d = sub3(p, hp);
    let dn = dot3(d, d).sqrt().max(1e-9);
    let g = [
        lam * d[0] / dn,
        lam * d[1] / dn,
        (1.0 - lam) + lam * d[2] / dn,
    ];
    let gn = dot3(g, g).sqrt().max(1e-9);
    let fz = [g[0] / gn, g[1] / gn, g[2] / gn];
    let cx = cross3(up, fz);
    let cn = dot3(cx, cx).sqrt().max(1e-9);
    let fx = [cx[0] / cn, cx[1] / cn, cx[2] / cn];
    let fy = cross3(fz, fx);
    [fx, fy, fz]
}

/// p relative to `center`, expressed in the note frame.
pub(crate) fn to_frame(p: [Real; 3], center: [Real; 3], f: &[[Real; 3]; 3]) -> [Real; 3] {
    let q = sub3(p, center);
    [dot3(q, f[0]), dot3(q, f[1]), dot3(q, f[2])]
}

pub(crate) fn sweep_hits_box(
    tris: &[[[Real; 3]; 3]; 2],
    center: [Real; 3],
    h: [Real; 3],
    frame: &[[Real; 3]; 3],
) -> bool {
    for t in tris {
        let rel = [
            to_frame(t[0], center, frame),
            to_frame(t[1], center, frame),
            to_frame(t[2], center, frame),
        ];
        if tri_box_overlap(rel, h) {
            return true;
        }
    }
    false
}
