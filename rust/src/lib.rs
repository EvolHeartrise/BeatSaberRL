//! Beat Arena sim core in Rust — THE physics implementation (SPEC.md is
//! the contract; `python/beat_arena_env/constants.py` mirrors constants).
//!
//! Env v7 (2026-07-22, research-driven): real Beat Saber geometry and
//! judgment on the v6 kinematic upper body. Floor-based world (y = 0 is
//! the floor), 0.6 m lanes, real non-uniform row heights; the good-cut
//! hitbox is the real elongated axis-aligned box (0.8 x 0.5 x 1.0,
//! shifted 0.25 m toward the player; dots 0.8 tall), the wrong-color
//! hitbox the real small box (0.35^3) — v5.3's rotated visual-cube
//! collision is retired. The saber sweep is the full swept surface
//! between the previous and current blade segments (two triangles), not
//! sampled points. Scoring is the real integer 115: floor(x+0.5)-rounded
//! components, accuracy normalized by 0.3 m. New object kinds: bombs
//! (r 0.18 sphere, hit = penalty, pass = free), chain heads (max 85, no
//! follow-through), chain links (flat 20, any direction, speed still
//! required), and arc pre/post-swing grants on anchoring notes.
//!
//! Body: the v6 arm chains, shoulders at (+-0.2, 1.47, -0.65) — a 1.8 m
//! player with the head 0.65 m behind the hit plane. Notes stay cuttable
//! until ~2 m behind the head (Z_MISS 2.65), where misses register.
//!
//! Chart generation stays in Python (single source of truth, once per
//! episode); the chart is injected via `set_chart`.

use numpy::{PyArray1, PyReadonlyArray2, PyReadwriteArray2, PyUntypedArrayMethods};
use pyo3::prelude::*;
use rayon::prelude::*;

const DT: f64 = 1.0 / 30.0;
const DRAG: f64 = 0.02;
const V_NORM: f64 = 6.0; // wrist-velocity obs normalizer (m/s)
const V_MIN_CUT: f64 = 1.0;
const ACC_NORM: f64 = 0.3; // real BS accuracy normalizer (m)
const COS_TOL: f64 = 0.5;
// lanes at x = -0.9 .. 0.9 (0.6 m pitch), rows at y = 0.85/1.40/1.90 —
// positions arrive PRE-RESOLVED in the chart tuple (constants.py owns them)
const Y_CENTER: f64 = 1.40; // obs y normalization
const REWARD_BAD: f64 = -0.5;
const REWARD_MISS: f64 = -0.5;
// env v7.8 (2026-07-24, user-priced): a combo break on ScoreSaber costs
// ~4 notes of multiplier; -0.5 made bombs the CHEAPEST error in the gym
// (miss = -0.5 + ~1.0 forgone points) while the referee prices all
// breaks equally — both the 598M and gen2 policies equilibrated at
// ~12% bombs eaten under -0.5, flat across budget and distribution
const REWARD_BOMB: f64 = -1.5;
// env v7.9 (2026-07-24, user-designed): tiered effort cost per non-coast
// joint action per tick (metabolic proxy — proximal joints move more
// mass), scaled by effort_coef. Calibrated so one accuracy point
// (1/115 ~ 0.0087) dwarfs the effort of earning it: a typical played
// episode's total effort bill is ~0.5-1.0 of a ~+40 return.
// Per-hand joint order (pitch, yaw, twist, elbow, roll, bend):
const EFFORT_COST: [f64; N_JOINTS] = [2e-4, 2e-4, 2e-4, 1.3e-4, 0.7e-4, 0.7e-4];

// --- hitboxes (env v7 — real BS, axis-aligned) -------------------------------
const GOOD_HALF_X: f64 = 0.4;
const GOOD_HALF_Y: f64 = 0.25;
const GOOD_HALF_Y_DOT: f64 = 0.4;
const GOOD_HALF_Y_LINK: f64 = 0.1;
const GOOD_HALF_Z: f64 = 0.5;
const GOOD_Z_SHIFT: f64 = -0.25; // box center = note_z + shift (toward player)
const BAD_HALF: f64 = 0.175;
const BAD_HALF_Y_LINK: f64 = 0.05;
const BOMB_RADIUS: f64 = 0.18;

// --- upper body (env v6 chains, v7 placement) --------------------------------
const SABER_LEN: f64 = 1.0; // m (real Beat Saber)
const SHOULDER: [[f64; 3]; 2] = [[-0.2, 1.47, -0.65], [0.2, 1.47, -0.65]];
const L_UPPER: f64 = 0.35; // m
const L_FORE: f64 = 0.30; // m
// tiered anatomical dynamics: ponderous shoulder/elbow, snappy wrists
const ALPHA_ARM: f64 = 80.0; // rad/s^2 (shoulder pitch/yaw/twist, elbow)
const OMEGA_ARM: f64 = 8.0; // rad/s
const ALPHA_WRIST: f64 = 200.0; // rad/s^2 (roll, bend — v5 values)
const OMEGA_WRIST: f64 = 20.0; // rad/s
// env v7.2 (user-specified): shoulder pitch capped at +65 deg (arm up)
const PITCH_LIM: f64 = 1.5707963267948966; // 90 deg — obs normalizer only
const PITCH_LO: f64 = -1.5707963267948966;
const PITCH_HI: f64 = 1.1344640137963142; // 65 deg
const YAW_LO: [f64; 2] = [-1.7453292519943295, -0.7853981633974483]; // L, R
const YAW_HI: [f64; 2] = [0.7853981633974483, 1.7453292519943295];
// env v7.1 (user-specified): tightened wrist roll + twist — the v5-era
// roll range was compensation for missing joints and let training
// cheat. Right roll [-180, 0] deg, twist [-70, 90]; left mirrored.
const TWIST_LIM: f64 = 1.5707963267948966; // 90 deg — obs normalizer only
const TWIST_LO: [f64; 2] = [-1.5707963267948966, -1.2217304763960306]; // L, R
const TWIST_HI: [f64; 2] = [1.2217304763960306, 1.5707963267948966];
const ELBOW_HI: f64 = 2.6179938779914944; // 150 deg (lo = 0)
const ROLL_LO: [f64; 2] = [0.0, -3.141592653589793]; // L, R
const ROLL_HI: [f64; 2] = [3.141592653589793, 0.0];
const BEND_LIM: f64 = 1.6580627893946132; // 95 deg from forward
const N_JOINTS: usize = 6;
const JOINT_ALPHA: [f64; N_JOINTS] =
    [ALPHA_ARM, ALPHA_ARM, ALPHA_ARM, ALPHA_ARM, ALPHA_WRIST, ALPHA_WRIST];
const JOINT_OMEGA: [f64; N_JOINTS] =
    [OMEGA_ARM, OMEGA_ARM, OMEGA_ARM, OMEGA_ARM, OMEGA_WRIST, OMEGA_WRIST];

fn joint_limits(hand: usize) -> [(f64, f64); N_JOINTS] {
    [
        (PITCH_LO, PITCH_HI),
        (YAW_LO[hand], YAW_HI[hand]),
        (TWIST_LO[hand], TWIST_HI[hand]),
        (0.0, ELBOW_HI),
        (ROLL_LO[hand], ROLL_HI[hand]),
        (-BEND_LIM, BEND_LIM),
    ]
}
const N_BLADE_SAMPLES: usize = 8; // contact-point/velocity samples
// broad-phase gate: max wrist z 0.0 (shoulder -0.65 + full arm) + blade
// 1.0 >= good-box near face at note_z - 0.75 -> note_z <= 1.75
const REACH_Z: f64 = 1.75;
// note consumed as missed past this — the HEAD plane (0 m behind the
// head at z = -0.65). DELIBERATE DEVIATION from the real game (which
// judges misses at despawn ~2 m behind the head, env v7.0-v7.2's 2.65):
// the late boundary paid the -0.5 up to ~20 ticks after the plane and
// kept "lost" notes ambiguous with "rescuable" ones — this tightens
// miss credit assignment (env v7.3, user-specified 2026-07-22).
const Z_MISS: f64 = 0.65;

// --- scoring (real Beat Saber 115, integer components) -----------------------
const PRE_FULL: f64 = 1.7453292519943295; // 100 deg
const POST_FULL: f64 = 1.0471975511965976; // 60 deg
const PTS_PRE: f64 = 70.0;
const PTS_POST: f64 = 30.0;
const PTS_ACC: f64 = 15.0;
const SCORE_MAX: f64 = 115.0;
const CHAIN_LINK_PTS: f64 = 20.0;
const SWING_BREAK: f64 = 0.005; // rad/tick below which the swing chain breaks
const FOLLOW_TIMEOUT: u32 = 15; // ticks a follow-through may keep accumulating

// object kinds (mirrors constants.py)
const KIND_NOTE: u8 = 0;
const KIND_BOMB: u8 = 1;
const KIND_CHAIN_HEAD: u8 = 2;
const KIND_CHAIN_LINK: u8 = 3;

const N_NOTE_SLOTS: usize = 16;
const NOTE_FEAT: usize = 14;
// obs v6: per hand 23 features — 6 joint angles (roll as cos/sin -> 7
// numbers), 6 angular velocities, derived wrist pos (3) + finite-diff
// wrist vel (3) + blade unit dir (3), and the live pre-swing meter
const SABER_FEAT: usize = 23;
const OFF_NOTES: usize = 2 * SABER_FEAT; // 46
const OFF_NOTE_MASK: usize = OFF_NOTES + N_NOTE_SLOTS * NOTE_FEAT; // 270
const OBS_DIM: usize = OFF_NOTE_MASK + N_NOTE_SLOTS; // 286
const X_NORM: f64 = 1.0;
const Y_NORM: f64 = 0.75;
const Z_SCALE: f64 = 8.0;
const Z_CLIP: (f64, f64) = (-3.0, 12.0);

/// Real BS scores are integers; explicit floor(x + 0.5) so every
/// implementation (Rust, torch, Python tools) agrees on halves.
fn round_pts(x: f64) -> f64 {
    (x + 0.5).floor()
}

#[derive(Clone, Copy)]
struct Note {
    hit_time: f64,
    x: f64, // world position at the hit plane (meters)
    y: f64,
    color: u8,
    // required swing direction (0 = +x, CCW), stored as its sin/cos pair
    // (2026-07-27): the angle is episode-constant, yet build_obs + the
    // hitbox roll used to evaluate sin/cos per visible note per tick
    sin_a: f64,
    cos_a: f64,
    speed: f64,     // approach speed toward the plane (m/s)
    any_dir: bool,  // dot note / chain link: any swing direction scores
    kind: u8,       // KIND_NOTE / _BOMB / _CHAIN_HEAD / _CHAIN_LINK
    free_pre: bool, // arc tail: full 70 pre-swing granted
    free_post: bool, // arc head: full 30 follow-through granted
}

/// Forward kinematics: (elbow, wrist, blade unit dir) from the six joint
/// angles q = [pitch, yaw, twist, elbow, roll, bend].
///
/// Arm frame  M = R_y(yaw) · R_x(−pitch) · R_z(twist)  (local +z = upper
/// arm; at zero everything points at the notes). Elbow flexion rotates
/// about the twist-oriented local x: forearm = M · (0, sin e, cos e).
/// The blade is the v5.2 wrist in the FOREARM frame
/// M_f = M · R_x(−elbow):  d(roll, bend) = (−sinψ·sinφ, cosψ·sinφ, cosφ)
/// — the neutral blade is the forearm's extension, exactly v5.2's law
/// generalized to a moving forearm.
/// Returns (elbow, wrist, blade dir, (cos roll, sin roll)) — the roll
/// pair is computed here anyway (2026-07-27) and build_obs needs it, so
/// exporting it saves recomputing the two transcendentals per arm-tick.
fn fk(
    hand: usize,
    q: &[f64; N_JOINTS],
) -> ([f64; 3], [f64; 3], [f64; 3], (f64, f64)) {
    let (sp, cp) = q[0].sin_cos();
    let (sy, cy) = q[1].sin_cos();
    let (st, ct) = q[2].sin_cos();
    let (se, ce) = q[3].sin_cos();
    let m = [
        [cy * ct - sy * sp * st, -cy * st - sy * sp * ct, sy * cp],
        [cp * st, cp * ct, sp],
        [-sy * ct - cy * sp * st, sy * st - cy * sp * ct, cy * cp],
    ];
    let sh = SHOULDER[hand];
    let e = [
        sh[0] + L_UPPER * m[0][2],
        sh[1] + L_UPPER * m[1][2],
        sh[2] + L_UPPER * m[2][2],
    ];
    let f = [
        m[0][1] * se + m[0][2] * ce,
        m[1][1] * se + m[1][2] * ce,
        m[2][1] * se + m[2][2] * ce,
    ];
    let w = [e[0] + L_FORE * f[0], e[1] + L_FORE * f[1], e[2] + L_FORE * f[2]];
    let (sr, cr) = q[4].sin_cos();
    let (sb, cb) = q[5].sin_cos();
    let d = [-sr * sb, cr * sb, cb];
    let b = [
        m[0][0] * d[0] + (m[0][1] * ce - m[0][2] * se) * d[1] + f[0] * d[2],
        m[1][0] * d[0] + (m[1][1] * ce - m[1][2] * se) * d[1] + f[1] * d[2],
        m[2][0] * d[0] + (m[2][1] * ce - m[2][2] * se) * d[1] + f[2] * d[2],
    ];
    (e, w, b, (cr, sr))
}

fn sub(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}
fn dot3(a: [f64; 3], b: [f64; 3]) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}
fn cross3(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

/// Triangle vs axis-aligned box (Akenine-Möller SAT). Verts are given
/// RELATIVE to the box center; h = half extents.
fn tri_box_overlap(v: [[f64; 3]; 3], h: [f64; 3]) -> bool {
    // box-axis tests
    for k in 0..3 {
        let lo = v[0][k].min(v[1][k]).min(v[2][k]);
        let hi = v[0][k].max(v[1][k]).max(v[2][k]);
        if lo > h[k] || hi < -h[k] {
            return false;
        }
    }
    let e = [sub(v[1], v[0]), sub(v[2], v[1]), sub(v[0], v[2])];
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

/// Closest distance from point p to triangle (a, b, c) (Ericson,
/// Real-Time Collision Detection §5.1.5). Handles degenerate triangles
/// via the edge/vertex regions.
fn tri_point_dist(a: [f64; 3], b: [f64; 3], c: [f64; 3], p: [f64; 3]) -> f64 {
    let ab = sub(b, a);
    let ac = sub(c, a);
    let ap = sub(p, a);
    let d1 = dot3(ab, ap);
    let d2 = dot3(ac, ap);
    let dist = |q: [f64; 3]| {
        let d = sub(p, q);
        dot3(d, d).sqrt()
    };
    if d1 <= 0.0 && d2 <= 0.0 {
        return dist(a);
    }
    let bp = sub(p, b);
    let d3 = dot3(ab, bp);
    let d4 = dot3(ac, bp);
    if d3 >= 0.0 && d4 <= d3 {
        return dist(b);
    }
    let vc = d1 * d4 - d3 * d2;
    if vc <= 0.0 && d1 >= 0.0 && d3 <= 0.0 {
        let denom = d1 - d3;
        let v = if denom.abs() < 1e-15 { 0.0 } else { d1 / denom };
        return dist([a[0] + v * ab[0], a[1] + v * ab[1], a[2] + v * ab[2]]);
    }
    let cp = sub(p, c);
    let d5 = dot3(ab, cp);
    let d6 = dot3(ac, cp);
    if d6 >= 0.0 && d5 <= d6 {
        return dist(c);
    }
    let vb = d5 * d2 - d1 * d6;
    if vb <= 0.0 && d2 >= 0.0 && d6 <= 0.0 {
        let denom = d2 - d6;
        let w = if denom.abs() < 1e-15 { 0.0 } else { d2 / denom };
        return dist([a[0] + w * ac[0], a[1] + w * ac[1], a[2] + w * ac[2]]);
    }
    let va = d3 * d6 - d5 * d4;
    if va <= 0.0 && (d4 - d3) >= 0.0 && (d5 - d6) >= 0.0 {
        let denom = (d4 - d3) + (d5 - d6);
        let w = if denom.abs() < 1e-15 { 0.0 } else { (d4 - d3) / denom };
        return dist([
            b[0] + w * (c[0] - b[0]),
            b[1] + w * (c[1] - b[1]),
            b[2] + w * (c[2] - b[2]),
        ]);
    }
    let denom = 1.0 / (va + vb + vc);
    let v = vb * denom;
    let w = vc * denom;
    dist([
        a[0] + ab[0] * v + ac[0] * w,
        a[1] + ab[1] * v + ac[1] * w,
        a[2] + ab[2] * v + ac[2] * w,
    ])
}

fn seg_point_dist3(a: [f64; 3], b: [f64; 3], c: [f64; 3]) -> f64 {
    let ab = sub(b, a);
    let denom = dot3(ab, ab);
    let ac = sub(c, a);
    let t = if denom < 1e-12 {
        0.0
    } else {
        (dot3(ac, ab) / denom).clamp(0.0, 1.0)
    };
    let px = a[0] + t * ab[0] - c[0];
    let py = a[1] + t * ab[1] - c[1];
    let pz = a[2] + t * ab[2] - c[2];
    (px * px + py * py + pz * pz).sqrt()
}

/// The swept blade surface over one tick: quad (w0->t0) to (w1->t1),
/// split into triangles (w0, t0, t1) and (w0, t1, w1).
fn sweep_tris(
    w0: [f64; 3],
    d0: [f64; 3],
    w1: [f64; 3],
    d1: [f64; 3],
) -> [[[f64; 3]; 3]; 2] {
    let t0 = [
        w0[0] + SABER_LEN * d0[0],
        w0[1] + SABER_LEN * d0[1],
        w0[2] + SABER_LEN * d0[2],
    ];
    let t1 = [
        w1[0] + SABER_LEN * d1[0],
        w1[1] + SABER_LEN * d1[1],
        w1[2] + SABER_LEN * d1[2],
    ];
    [[w0, t0, t1], [w0, t1, w1]]
}

// Env v7.4: hitboxes ROTATE with the object about the approach (z)
// axis by rot = angle - 270 deg — the real game's colliders sit on the
// rotated note transform (v7.0's axis-aligned reading was a
// BeatLeader-viewer simplification). Down-swing notes are axis-aligned
// (rot = 0); dot notes never rotate; chain links rotate with their
// tangent. (cos_r, sin_r) derived exactly from the angle's own sin/cos —
// since 2026-07-27 read from the Note's precomputed pair at the use site.

// --- env v7.6: look-at-player note orientation -------------------------------
// Notes bias their FULL 3D orientation toward facing the player's head
// as they approach (ArcViewer ObjectManager.LookAtPlayer / BeatLeader
// beat.js lookAt): the head's y is damped 80% toward the note's y, the
// forward axis blends from +z to the head->note ray with weight
// lam = clamp(1 - z/LOOK_DIST, 0, 1) (our analog of jump progress —
// the real game reaches full head-facing at the play plane), and the
// cut-direction z-roll rides along as the frame's up. lam = 0
// reproduces the v7.4 pure z-roll frame EXACTLY. Applies to plain
// notes only (chain heads are excluded in ArcViewer; links keep their
// tangent roll; bombs are spheres). The game's colliders AND its
// local-frame direction check both live on this transform.
const HEAD: [f64; 3] = [0.0, 1.70, -0.65]; // fixed head anchor (1.8 m player)
const LOOK_VERT: f64 = 0.8; // verticalLookStrength: head.y 80% toward note.y
const LOOK_DIST: f64 = 8.0; // full head-facing at z=0, none beyond 8 m

/// Orthonormal note frame columns (x', y', z'): z' = the blended
/// forward, up from the z-roll, x' = up x z'.
fn note_frame(p: [f64; 3], cos_r: f64, sin_r: f64, lam: f64) -> [[f64; 3]; 3] {
    let up = [-sin_r, cos_r, 0.0];
    let hp = [HEAD[0], HEAD[1] + LOOK_VERT * (p[1] - HEAD[1]), HEAD[2]];
    let d = sub(p, hp);
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
fn to_frame(p: [f64; 3], center: [f64; 3], f: &[[f64; 3]; 3]) -> [f64; 3] {
    let q = sub(p, center);
    [dot3(q, f[0]), dot3(q, f[1]), dot3(q, f[2])]
}

fn sweep_hits_box(
    tris: &[[[f64; 3]; 3]; 2],
    center: [f64; 3],
    h: [f64; 3],
    frame: &[[f64; 3]; 3],
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

fn sweep_dist(tris: &[[[f64; 3]; 3]; 2], p: [f64; 3]) -> f64 {
    tri_point_dist(tris[0][0], tris[0][1], tris[0][2], p)
        .min(tri_point_dist(tris[1][0], tris[1][1], tris[1][2], p))
}

#[derive(Clone, Copy)]
struct Arm {
    q: [f64; N_JOINTS],  // pitch, yaw, twist, elbow, roll, bend
    om: [f64; N_JOINTS], // angular velocities
    wrist: [f64; 3],     // FK cache (updated whenever q changes)
    blade: [f64; 3],     // FK cache, unit blade direction
    roll_cs: (f64, f64), // FK cache: (cos, sin) of q[4] for build_obs
    wrist_vel: [f64; 3], // finite difference across the last tick
    swing_acc: f64,      // arc of the current continuous swing (rad)
    axis: Option<[f64; 3]>, // last tick's rotation axis (unit)
}

impl Arm {
    fn start(hand: usize) -> Self {
        // all joints zero: arm straight forward, blade its extension
        let (_, w, b, rcs) = fk(hand, &[0.0; N_JOINTS]);
        Arm {
            q: [0.0; N_JOINTS],
            om: [0.0; N_JOINTS],
            wrist: w,
            blade: b,
            roll_cs: rcs,
            wrist_vel: [0.0; 3],
            swing_acc: 0.0,
            axis: None,
        }
    }
}

/// A good cut's pending follow-through meter.
#[derive(Clone, Copy)]
struct Tracker {
    note: u32,
    acc: f64,
    age: u32,
}

type ChartTuple = (f64, f64, f64, u8, f64, f64, bool, u8, bool, bool);

struct Core {
    max_steps: u32,
    effort_coef: f64, // scales EFFORT_COST (0 = free movement)
    chart: Vec<Note>,
    tail_s: f64, // episode tail after the last hit_time (covers Z_MISS exit)
    min_speed: f64, // slowest object; (hit_time-t)*min_speed > REACH_Z is a
                    // conservative exact "beyond reach" bound for a sorted chart
    // state
    tick: u32,
    time: f64,
    first_pending: usize, // smallest index with status 0 (monotone per episode)
    arms: [Arm; 2],
    trackers: [Vec<Tracker>; 2],
    status: Vec<u8>, // 0 pending, 1 good, 2 bad/bomb-hit, 3 missed, 4 bomb passed
    good_cuts: u32,
    bad_cuts: u32,
    misses: u32,
    bombs_hit: u32,
    n_scoreable: u32, // chart objects that are not bombs
    score: f64,       // summed points; max = 115*notes + 85*heads + 20*links
    // score decomposition (2026-07-25, user-directed): the three graded
    // components tracked independently — links' flat 20s are in none
    // (score - pre - acc - post = link points)
    score_pre: f64,
    score_acc: f64,
    score_post: f64,
    // reusable event buffer (2026-07-27): step_inner used to allocate a
    // fresh Vec per tick; cleared and refilled in place instead. Read it
    // right after step_inner — the next tick clears it.
    events_buf: Vec<(u32, u8, f32, f32, f32)>,
}

impl Core {
    fn new_core(max_steps: u32, effort_coef: f64) -> Self {
        Core {
            max_steps,
            effort_coef,
            chart: Vec::new(),
            tail_s: 0.5,
            min_speed: f64::INFINITY,
            tick: 0,
            time: 0.0,
            first_pending: 0,
            arms: [Arm::start(0), Arm::start(1)],
            trackers: [Vec::new(), Vec::new()],
            status: Vec::new(),
            good_cuts: 0,
            bad_cuts: 0,
            misses: 0,
            bombs_hit: 0,
            n_scoreable: 0,
            score: 0.0,
            score_pre: 0.0,
            score_acc: 0.0,
            score_post: 0.0,
            events_buf: Vec::new(),
        }
    }

    fn set_chart_notes(&mut self, chart: Vec<ChartTuple>) {
        self.chart = chart
            .into_iter()
            .map(|(t, x, y, color, angle, speed, any_dir, kind, fpre, fpost)| Note {
                hit_time: t,
                x,
                y,
                color,
                sin_a: angle.sin(),
                cos_a: angle.cos(),
                speed,
                any_dir,
                kind,
                free_pre: fpre,
                free_post: fpost,
            })
            .collect();
        // the scan cursors assume hit_time order (chart.py/beatsaver.py
        // both sort); an unsorted chart would silently change behavior
        assert!(
            self.chart.windows(2).all(|w| w[0].hit_time <= w[1].hit_time),
            "chart must be sorted by hit_time"
        );
        // the episode tail must cover the slowest object's trip from the
        // plane to the miss boundary, or final misses never register
        let min_speed = self
            .chart
            .iter()
            .map(|n| n.speed)
            .fold(f64::INFINITY, f64::min);
        self.min_speed = min_speed;
        self.tail_s = if min_speed.is_finite() {
            0.5 + Z_MISS / min_speed
        } else {
            0.5
        };
        self.reset_state();
    }

    fn reset_state(&mut self) {
        self.tick = 0;
        self.time = 0.0;
        self.first_pending = 0;
        self.arms = [Arm::start(0), Arm::start(1)];
        self.trackers = [Vec::new(), Vec::new()];
        self.status = vec![0; self.chart.len()];
        self.good_cuts = 0;
        self.bad_cuts = 0;
        self.misses = 0;
        self.bombs_hit = 0;
        self.n_scoreable = self.chart.iter().filter(|n| n.kind != KIND_BOMB).count() as u32;
        self.score = 0.0;
        self.score_pre = 0.0;
        self.score_acc = 0.0;
        self.score_post = 0.0;
    }

    /// Obs written into a caller-owned 286-lane slice (2026-07-27): the
    /// return-by-value build_obs paid a 1144 B stack array + a second
    /// memcpy at every caller; the vector step writes rows in place.
    fn write_obs(&self, obs: &mut [f32]) {
        debug_assert_eq!(obs.len(), OBS_DIM);
        obs.fill(0.0);
        for (s, arm) in self.arms.iter().enumerate() {
            let b = s * SABER_FEAT;
            // joint pose: bounded scalars normalized to their ranges,
            // except roll (280 deg arc) which keeps its v5 cos/sin coding
            obs[b] = (arm.q[0] / PITCH_LIM) as f32;
            obs[b + 1] = (arm.q[1] / YAW_HI[1]) as f32;
            obs[b + 2] = (arm.q[2] / TWIST_LIM) as f32;
            obs[b + 3] = ((2.0 * arm.q[3] - ELBOW_HI) / ELBOW_HI) as f32;
            obs[b + 4] = arm.roll_cs.0 as f32;
            obs[b + 5] = arm.roll_cs.1 as f32;
            obs[b + 6] = (arm.q[5] / BEND_LIM) as f32;
            for j in 0..N_JOINTS {
                obs[b + 7 + j] = (arm.om[j] / JOINT_OMEGA[j]) as f32;
            }
            // derived hand state: FK wrist, finite-diff velocity, blade dir
            obs[b + 13] = (arm.wrist[0] / X_NORM) as f32;
            obs[b + 14] = ((arm.wrist[1] - Y_CENTER) / Y_NORM) as f32;
            obs[b + 15] = arm.wrist[2] as f32;
            for k in 0..3 {
                obs[b + 16 + k] = (arm.wrist_vel[k] / V_NORM) as f32;
                obs[b + 19 + k] = arm.blade[k] as f32;
            }
            // live pre-swing meter: 1.0 = full pre-swing credit banked
            obs[b + 22] = (arm.swing_acc / PRE_FULL).min(2.0) as f32;
        }
        let mut slot = 0usize;
        // start past the resolved prefix (everything before first_pending
        // has status != 0 by construction) — the loop then typically
        // terminates within ~N_NOTE_SLOTS iterations instead of O(chart)
        for i in self.first_pending..self.chart.len() {
            let n = &self.chart[i];
            if self.status[i] != 0 {
                continue;
            }
            if slot >= N_NOTE_SLOTS {
                break;
            }
            let base = OFF_NOTES + slot * NOTE_FEAT;
            let z = ((n.hit_time - self.time) * n.speed).clamp(Z_CLIP.0, Z_CLIP.1);
            obs[base] = (n.x / X_NORM) as f32;
            obs[base + 1] = ((n.y - Y_CENTER) / Y_NORM) as f32;
            obs[base + 2] = (z / Z_SCALE) as f32;
            obs[base + 3] = (-n.speed / Z_SCALE) as f32;
            if n.kind != KIND_BOMB {
                obs[base + 4 + n.color as usize] = 1.0;
            }
            // dot notes encode direction as the ZERO vector; chain links
            // carry their tangent (path information) despite judging as
            // any-direction — is_chain_link disambiguates
            let show_dir = !n.any_dir || n.kind == KIND_CHAIN_LINK;
            if n.kind != KIND_BOMB && show_dir {
                obs[base + 6] = n.cos_a as f32;
                obs[base + 7] = n.sin_a as f32;
            }
            obs[base + 8] = (n.kind == KIND_BOMB) as u8 as f32;
            obs[base + 9] = (n.kind == KIND_CHAIN_HEAD) as u8 as f32;
            obs[base + 10] = (n.kind == KIND_CHAIN_LINK) as u8 as f32;
            obs[base + 11] = n.free_pre as u8 as f32;
            obs[base + 12] = n.free_post as u8 as f32;
            obs[base + 13] = 1.0; // valid
            obs[OFF_NOTE_MASK + slot] = 1.0;
            slot += 1;
        }
    }

    fn build_obs(&self) -> [f32; OBS_DIM] {
        let mut obs = [0.0f32; OBS_DIM];
        self.write_obs(&mut obs);
        obs
    }

    // event codes: 0 ok, 1 slow, 2 angle, 3 wrong_hand, 4 miss, 5 post
    // (follow-through resolution), 6 bomb hit. The f32 is points awarded.
    // Events land in self.events_buf (cleared here), in the same order the
    // old per-tick Vec produced them — order is the ComboWalk contract.
    fn step_inner(&mut self, actions: [[u8; N_JOINTS]; 2]) -> (f64, bool) {
        let prev_pos = [self.arms[0].wrist, self.arms[1].wrist];
        let prev_dir = [self.arms[0].blade, self.arms[1].blade];

        let mut reward = 0.0;
        self.events_buf.clear();
        // per arm this tick: (swing delta, did the chain continue)
        let mut deltas = [0.0f64; 2];
        let mut continued = [false; 2];

        // env v7.9: tiered effort — every non-coast action costs
        let mut effort = 0.0;
        for s in 0..2 {
            for j in 0..N_JOINTS {
                if actions[s][j] != 0 {
                    effort += EFFORT_COST[j];
                }
            }
        }
        reward -= self.effort_coef * effort;

        for s in 0..2 {
            let arm = &mut self.arms[s];
            // one integrator for every joint: accel, drag, omega cap,
            // angle step, limit clamp (hitting a limit kills the omega)
            let lims = joint_limits(s);
            for j in 0..N_JOINTS {
                arm.om[j] += match actions[s][j] {
                    1 => JOINT_ALPHA[j] * DT,
                    2 => -JOINT_ALPHA[j] * DT,
                    _ => 0.0,
                };
                arm.om[j] =
                    (arm.om[j] * (1.0 - DRAG)).clamp(-JOINT_OMEGA[j], JOINT_OMEGA[j]);
                arm.q[j] += arm.om[j] * DT;
                if arm.q[j] < lims[j].0 || arm.q[j] > lims[j].1 {
                    arm.q[j] = arm.q[j].clamp(lims[j].0, lims[j].1);
                    arm.om[j] = 0.0;
                }
            }
            let (_, w, d_new, rcs) = fk(s, &arm.q);
            arm.wrist = w;
            arm.blade = d_new;
            arm.roll_cs = rcs;
            for k in 0..3 {
                arm.wrist_vel[k] = (w[k] - prev_pos[s][k]) / DT;
            }

            // swing chain update (feeds pre-swing + follow-through scores)
            let d_old = prev_dir[s];
            let dot = (d_old[0] * d_new[0] + d_old[1] * d_new[1] + d_old[2] * d_new[2])
                .clamp(-1.0, 1.0);
            let delta = dot.acos();
            deltas[s] = delta;
            if delta < SWING_BREAK {
                arm.swing_acc = 0.0;
                arm.axis = None;
            } else {
                let cx = cross3(d_old, d_new);
                let cn = (cx[0].powi(2) + cx[1].powi(2) + cx[2].powi(2)).sqrt();
                let ax = [cx[0] / cn, cx[1] / cn, cx[2] / cn];
                match arm.axis {
                    Some(a0) if a0[0] * ax[0] + a0[1] * ax[1] + a0[2] * ax[2] < 0.0 => {
                        arm.swing_acc = delta; // reversal: a new swing starts
                    }
                    _ => {
                        arm.swing_acc += delta;
                        continued[s] = true;
                    }
                }
                arm.axis = Some(ax);
            }
        }

        // follow-through trackers (existing ones, BEFORE this tick's cuts):
        // chain broke/reversed -> resolve; else accumulate and resolve on
        // reaching 60 deg or timing out.
        for s in 0..2 {
            // in-place retain (2026-07-27): the drain-into-fresh-Vec
            // rebuild paid ~2 malloc/free per env-tick whenever a tracker
            // survived; retain_mut visits in the same order, so events
            // and arithmetic are unchanged. (mem::take dodges the twin
            // &mut self borrow; it swaps in an alloc-free empty Vec.)
            let mut trs = std::mem::take(&mut self.trackers[s]);
            trs.retain_mut(|tr| {
                let resolve = if !continued[s] {
                    true
                } else {
                    tr.acc += deltas[s];
                    tr.age += 1;
                    tr.acc >= POST_FULL || tr.age >= FOLLOW_TIMEOUT
                };
                if resolve {
                    let pts = round_pts(PTS_POST * (tr.acc / POST_FULL).min(1.0));
                    reward += pts / SCORE_MAX;
                    self.score += pts;
                    self.score_post += pts;
                    self.events_buf.push((tr.note, 5, pts as f32, 0.0, 0.0));
                }
                !resolve
            });
            self.trackers[s] = trs;
        }

        self.time = (self.tick + 1) as f64 * DT;

        // both sabers' swept surfaces this tick
        let tris = [
            sweep_tris(prev_pos[0], prev_dir[0], self.arms[0].wrist, self.arms[0].blade),
            sweep_tris(prev_pos[1], prev_dir[1], self.arms[1].wrist, self.arms[1].blade),
        ];

        for i in self.first_pending..self.chart.len() {
            if self.status[i] != 0 {
                continue;
            }
            let n = &self.chart[i];   // by ref: Note is 80 B post-sin/cos
            // sorted by hit_time, and speed >= min_speed for every note:
            // once this bound clears REACH_Z, so does every later note's z
            if (n.hit_time - self.time) * self.min_speed > REACH_Z {
                break;
            }
            let z = (n.hit_time - self.time) * n.speed;
            if z < -Z_MISS {
                if n.kind == KIND_BOMB {
                    self.status[i] = 4; // passed clean: no penalty, no event
                } else {
                    self.status[i] = 3;
                    self.misses += 1;
                    reward += REWARD_MISS;
                    self.events_buf.push((i as u32, 4, 0.0, 0.0, 0.0));
                }
                continue;
            }
            if z > REACH_Z {
                continue;
            }
            let center = [n.x, n.y, z];

            if n.kind == KIND_BOMB {
                for s in 0..2 {
                    if sweep_dist(&tris[s], center) <= BOMB_RADIUS {
                        self.status[i] = 2;
                        self.bombs_hit += 1;
                        reward += REWARD_BOMB;
                        self.events_buf.push((i as u32, 6, 0.0, 0.0, 0.0));
                        break;
                    }
                }
                continue;
            }

            let m = n.color as usize; // matching saber
            let good_hy = if n.kind == KIND_CHAIN_LINK {
                GOOD_HALF_Y_LINK
            } else if n.any_dir {
                GOOD_HALF_Y_DOT
            } else {
                GOOD_HALF_Y
            };
            let good_h = [GOOD_HALF_X, good_hy, GOOD_HALF_Z];
            // arrow notes/heads and chain links (tangent) roll their
            // hitboxes; plain dots stay roll-free
            let rotates = n.kind == KIND_CHAIN_LINK || !n.any_dir;
            // rot = angle - 270 deg (see the v7.4 note above note_frame):
            // cos(a - 3pi/2) = -sin a, sin(a - 3pi/2) = cos a — read from
            // the precomputed chart-constant pair
            let (cos_r, sin_r) = if rotates { (-n.sin_a, n.cos_a) } else { (1.0, 0.0) };
            // env v7.6: plain notes additionally billboard toward the
            // head as they approach; heads/links keep the pure z-roll
            let lam = if n.kind == KIND_NOTE {
                (1.0 - z / LOOK_DIST).clamp(0.0, 1.0)
            } else {
                0.0
            };
            let frame = note_frame(center, cos_r, sin_r, lam);
            // the good box's 0.25 m shift is along the note's LOCAL z
            // (it rides the transform, extending toward the head)
            let good_center = [
                center[0] + GOOD_Z_SHIFT * frame[2][0],
                center[1] + GOOD_Z_SHIFT * frame[2][1],
                center[2] + GOOD_Z_SHIFT * frame[2][2],
            ];

            if sweep_hits_box(&tris[m], good_center, good_h, &frame) {
                // contact point: the blade sample whose displacement
                // segment passes closest to the note center; its
                // finite-difference velocity is the contact velocity
                let arm = self.arms[m];
                let mut d_seg = f64::INFINITY;
                let mut s_star = 0.0;
                for j in 1..=N_BLADE_SAMPLES {
                    let r = (j as f64 / N_BLADE_SAMPLES as f64) * SABER_LEN;
                    let p0 = [
                        prev_pos[m][0] + r * prev_dir[m][0],
                        prev_pos[m][1] + r * prev_dir[m][1],
                        prev_pos[m][2] + r * prev_dir[m][2],
                    ];
                    let p1 = [
                        arm.wrist[0] + r * arm.blade[0],
                        arm.wrist[1] + r * arm.blade[1],
                        arm.wrist[2] + r * arm.blade[2],
                    ];
                    let d = seg_point_dist3(p0, p1, center);
                    if d < d_seg {
                        d_seg = d;
                        s_star = r;
                    }
                }
                let vc = [
                    (arm.wrist[0] + s_star * arm.blade[0]
                        - prev_pos[m][0] - s_star * prev_dir[m][0]) / DT,
                    (arm.wrist[1] + s_star * arm.blade[1]
                        - prev_pos[m][1] - s_star * prev_dir[m][1]) / DT,
                    (arm.wrist[2] + s_star * arm.blade[2]
                        - prev_pos[m][2] - s_star * prev_dir[m][2]) / DT,
                ];
                let vc_mag = (vc[0].powi(2) + vc[1].powi(2) + vc[2].powi(2)).sqrt();
                // accuracy: closest approach of the swept surface to the
                // note center, normalized by the real 0.3 m
                let d_min = sweep_dist(&tris[m], center);
                let code: u8 = if vc_mag < V_MIN_CUT {
                    1 // slow
                } else if n.any_dir {
                    0 // dot note / chain link: any direction scores
                } else {
                    // env v7.6: direction check in the NOTE'S LOCAL
                    // frame (the game transforms the cut velocity by
                    // the note rotation) — required direction is local
                    // -y'; at lam = 0 this is exactly the old world
                    // in-plane check
                    let vlx = dot3(vc, frame[0]);
                    let vly = dot3(vc, frame[1]);
                    let v2_mag = (vlx * vlx + vly * vly).sqrt();
                    let cosv = -vly / v2_mag.max(1e-9);
                    if cosv >= COS_TOL { 0 } else { 2 } // ok / angle
                };
                if code == 0 {
                    self.status[i] = 1;
                    self.good_cuts += 1;
                    let (pts, pre_pts, acc_pts) = if n.kind == KIND_CHAIN_LINK {
                        (CHAIN_LINK_PTS, 0.0, 0.0) // flat, regardless of swing
                    } else {
                        let pre = if n.free_pre {
                            PTS_PRE
                        } else {
                            round_pts(PTS_PRE * (self.arms[m].swing_acc / PRE_FULL).min(1.0))
                        };
                        let acc =
                            round_pts(PTS_ACC * (1.0 - (d_min / ACC_NORM).clamp(0.0, 1.0)));
                        let mut p = pre + acc;
                        if n.kind == KIND_CHAIN_HEAD {
                            // chain heads have no follow-through (max 85)
                        } else if n.free_post {
                            p += PTS_POST; // arc head: granted in full
                            self.score_post += PTS_POST;
                        } else {
                            self.trackers[m].push(Tracker {
                                note: i as u32,
                                acc: 0.0,
                                age: 0,
                            });
                        }
                        (p, pre, acc)
                    };
                    reward += pts / SCORE_MAX;
                    self.score += pts;
                    self.score_pre += pre_pts;
                    self.score_acc += acc_pts;
                    self.events_buf.push((i as u32, 0, pts as f32, pre_pts as f32,
                                 acc_pts as f32));
                } else {
                    self.status[i] = 2;
                    self.bad_cuts += 1;
                    reward += REWARD_BAD;
                    self.events_buf.push((i as u32, code, 0.0, 0.0, 0.0));
                }
            } else {
                // wrong-color saber only triggers the SMALL box
                let bad_hy = if n.kind == KIND_CHAIN_LINK {
                    BAD_HALF_Y_LINK
                } else {
                    BAD_HALF
                };
                if sweep_hits_box(&tris[1 - m], center, [BAD_HALF, bad_hy, BAD_HALF],
                                  &frame) {
                    self.status[i] = 2;
                    self.bad_cuts += 1;
                    reward += REWARD_BAD;
                    self.events_buf.push((i as u32, 3, 0.0, 0.0, 0.0)); // wrong_hand
                }
            }
        }

        self.tick += 1;
        // advance the low-water cursor past the newly resolved prefix;
        // "smallest pending index" is monotone within an episode because
        // status only ever goes 0 -> nonzero between resets
        while self.first_pending < self.chart.len()
            && self.status[self.first_pending] != 0
        {
            self.first_pending += 1;
        }
        let last_hit = self.chart.last().map_or(0.0, |n| n.hit_time);
        let pending = self.first_pending < self.chart.len();
        // natural completion waits for pending follow-throughs so the last
        // note's post-swing isn't truncated; the time/tick caps still end
        // the episode hard (trackers force-resolve below).
        let no_trackers = self.trackers[0].is_empty() && self.trackers[1].is_empty();
        let done = (!pending
            && no_trackers
            && (self.chart.is_empty() || self.time > last_hit))
            || self.time > last_hit + self.tail_s
            || self.tick >= self.max_steps;

        if done {
            // resolve any pending follow-throughs with their partial arcs
            for s in 0..2 {
                for tr in self.trackers[s].drain(..) {
                    let pts = round_pts(PTS_POST * (tr.acc / POST_FULL).min(1.0));
                    reward += pts / SCORE_MAX;
                    self.score += pts;
                    self.score_post += pts;
                    self.events_buf.push((tr.note, 5, pts as f32, 0.0, 0.0));
                }
            }
        }

        (reward, done)
    }
}

#[pyclass]
struct RawBeatSim {
    core: Core,
}

#[pymethods]
impl RawBeatSim {
    #[new]
    #[pyo3(signature = (max_steps=690, effort_coef=1.0))]
    fn new(max_steps: u32, effort_coef: f64) -> Self {
        RawBeatSim {
            core: Core::new_core(max_steps, effort_coef),
        }
    }

    /// chart: list of (hit_time, x, y, color, angle_rad, speed, any_dir,
    /// kind, free_pre, free_post), sorted by hit_time; positions in
    /// meters (env v7). Resets state, returns the reset observation.
    fn set_chart<'py>(
        &mut self,
        py: Python<'py>,
        chart: Vec<ChartTuple>,
    ) -> Bound<'py, PyArray1<f32>> {
        self.core.set_chart_notes(chart);
        PyArray1::from_slice_bound(py, &self.core.build_obs())
    }

    /// actions: 12 ints in 0..2 — per hand (pitch, yaw, twist, elbow,
    /// roll, bend), left hand first; 0 = coast, 1 = +alpha, 2 = -alpha.
    /// Returns (obs, reward, done, events); events = [(note_idx, code,
    /// points, pre_pts, acc_pts)] with codes 0 ok, 1 slow, 2 angle,
    /// 3 wrong_hand, 4 miss, 5 post, 6 bomb. The component split
    /// (2026-07-25, user-directed): a good cut's points = pre + acc
    /// (+ 30 for arc-headed free_post grants); post/link/error events
    /// carry (0, 0).
    fn step<'py>(
        &mut self,
        py: Python<'py>,
        actions: Vec<u8>,
    ) -> (Bound<'py, PyArray1<f32>>, f64, bool, Vec<(u32, u8, f32, f32, f32)>) {
        assert_eq!(actions.len(), 2 * N_JOINTS);
        assert!(actions.iter().all(|&a| a <= 2));
        let mut acts = [[0u8; N_JOINTS]; 2];
        for s in 0..2 {
            acts[s].copy_from_slice(&actions[s * N_JOINTS..(s + 1) * N_JOINTS]);
        }
        let (reward, done) = self.core.step_inner(acts);
        (
            PyArray1::from_slice_bound(py, &self.core.build_obs()),
            reward,
            done,
            self.core.events_buf.clone(),
        )
    }

    /// Arm state [[q(6) ++ om(6) ++ wrist(3)]; 2] — joint angles
    /// (pitch, yaw, twist, elbow, roll, bend), angular velocities, and
    /// the FK wrist position.
    fn sabers(&self) -> [[f64; 15]; 2] {
        [0, 1].map(|s| {
            let a = self.core.arms[s];
            let mut out = [0.0; 15];
            out[..6].copy_from_slice(&a.q);
            out[6..12].copy_from_slice(&a.om);
            out[12..].copy_from_slice(&a.wrist);
            out
        })
    }

    /// Blade unit directions [[x, y, z]; 2] (FK cache).
    fn blades(&self) -> [[f64; 3]; 2] {
        [0, 1].map(|s| self.core.arms[s].blade)
    }

    /// Per-note status: 0 pending, 1 good, 2 bad/bomb-hit, 3 missed,
    /// 4 bomb passed clean.
    fn status(&self) -> Vec<u8> {
        self.core.status.clone()
    }

    /// TEST HOOK: overwrite one arm's joint state before a step (FK cache
    /// refreshes immediately; the finite-diff wrist velocity resets).
    fn set_joints(&mut self, s: usize, q: Vec<f64>, om: Vec<f64>) {
        assert_eq!(q.len(), N_JOINTS);
        assert_eq!(om.len(), N_JOINTS);
        let arm = &mut self.core.arms[s];
        arm.q.copy_from_slice(&q);
        arm.om.copy_from_slice(&om);
        let (_, w, b, rcs) = fk(s, &arm.q);
        arm.wrist = w;
        arm.blade = b;
        arm.roll_cs = rcs;
        arm.wrist_vel = [0.0; 3];
    }

    #[getter]
    fn good_cuts(&self) -> u32 {
        self.core.good_cuts
    }
    #[getter]
    fn bad_cuts(&self) -> u32 {
        self.core.bad_cuts
    }
    #[getter]
    fn misses(&self) -> u32 {
        self.core.misses
    }
    #[getter]
    fn bombs_hit(&self) -> u32 {
        self.core.bombs_hit
    }
    /// Scoreable objects (notes + chain heads + links; bombs excluded).
    #[getter]
    fn n_notes(&self) -> u32 {
        self.core.n_scoreable
    }
    #[getter]
    fn tick(&self) -> u32 {
        self.core.tick
    }
    #[getter]
    fn time(&self) -> f64 {
        self.core.time
    }
    /// Summed points across all scored cuts.
    #[getter]
    fn score(&self) -> f64 {
        self.core.score
    }
    /// (pre_swing, accuracy, follow_through) point totals — the score
    /// decomposition; score − sum = chain-link flat points.
    fn score_components(&self) -> (f64, f64, f64) {
        (self.core.score_pre, self.core.score_acc, self.core.score_post)
    }
}

/// ScoreSaber multiplied-score combo walk (bench-only; the training gym
/// stays multiplier-free). Mirrors beat_arena_env/scoresaber.py's
/// ComboTracker exactly — pinned by a cross-implementation test:
/// promotion (1x/2x/4x/8x, bar 2/4/8 hits) applies to the hit that
/// fills the bar; a note's cut AND its later post points commit at its
/// cut-time multiplier; any break (bad cut / miss / bomb) zeroes the
/// combo, halves the multiplier, resets the bar.
struct ComboWalk {
    multiplier: u32,
    progress: u32,
    combo: u32,
    max_combo: u32,
    breaks: u32,
    score: f64,
    note_mult: Vec<u32>, // 0 = not cut yet
}

impl ComboWalk {
    fn new(n_objects: usize) -> Self {
        ComboWalk {
            multiplier: 1,
            progress: 0,
            combo: 0,
            max_combo: 0,
            breaks: 0,
            score: 0.0,
            note_mult: vec![0; n_objects],
        }
    }

    fn progress_needed(m: u32) -> Option<u32> {
        match m {
            1 => Some(2),
            2 => Some(4),
            4 => Some(8),
            _ => None, // 8x cap
        }
    }

    fn consume(&mut self, note: u32, code: u8, pts: f32) {
        let i = note as usize;
        match code {
            0 => {
                // good cut: promote-before-commit, pin the multiplier
                if let Some(need) = Self::progress_needed(self.multiplier) {
                    self.progress += 1;
                    if self.progress >= need {
                        self.multiplier *= 2;
                        self.progress = 0;
                    }
                }
                self.note_mult[i] = self.multiplier;
                self.combo += 1;
                self.max_combo = self.max_combo.max(self.combo);
                self.score += pts as f64 * self.multiplier as f64;
            }
            5 => {
                // follow-through: the note's cut-time multiplier
                let m = if self.note_mult[i] == 0 {
                    self.multiplier
                } else {
                    self.note_mult[i]
                };
                self.score += pts as f64 * m as f64;
            }
            _ => {
                // slow / angle / wrong_hand / miss / bomb: combo break
                self.combo = 0;
                self.breaks += 1;
                self.multiplier = (self.multiplier / 2).max(1);
                self.progress = 0;
            }
        }
    }
}

/// N sims stepped in ONE FFI call per vector step. Chart generation stays
/// in Python: after any done env, call `set_chart(i, chart)` before the
/// next `step` (the wrapper in beat_arena_env/vec_env.py handles this).
/// `combo_walk=True` (bench): each env additionally runs the ScoreSaber
/// multiplied-score walk over its own ordered events; read with
/// `combo(i)` BEFORE recycling the env's chart (set_chart resets it).
#[pyclass]
struct VecBeatSim {
    sims: Vec<Core>,
    walks: Option<Vec<ComboWalk>>,
    return_events: bool,
}

#[pymethods]
impl VecBeatSim {
    #[new]
    #[pyo3(signature = (n, max_steps=690, effort_coef=1.0, combo_walk=false,
                        return_events=true))]
    fn new(n: usize, max_steps: u32, effort_coef: f64, combo_walk: bool,
           return_events: bool) -> Self {
        VecBeatSim {
            sims: (0..n).map(|_| Core::new_core(max_steps, effort_coef)).collect(),
            walks: combo_walk.then(|| (0..n).map(|_| ComboWalk::new(0)).collect()),
            // return_events=false skips materializing the per-step event
            // list into Python objects (~2 events per scoreable object);
            // the in-Rust combo walk still consumes every event. Both
            // vectorized consumers (bench, BeatVecEnv) discard the list.
            return_events,
        }
    }

    fn set_chart(&mut self, i: usize, chart: Vec<ChartTuple>) {
        let n_objects = chart.len();
        self.sims[i].set_chart_notes(chart);
        if let Some(walks) = &mut self.walks {
            walks[i] = ComboWalk::new(n_objects);
        }
    }

    /// (multiplied_score, max_combo, breaks) of env i's combo walk.
    fn combo(&self, i: usize) -> (f64, u32, u32) {
        let w = &self.walks.as_ref().expect("combo_walk not enabled")[i];
        (w.score, w.max_combo, w.breaks)
    }

    /// Current observation of every env, flat (N*286,) — reshape Python-side.
    fn obs<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f32>> {
        let n = self.sims.len();
        let mut flat = vec![0.0f32; n * OBS_DIM];
        for (i, sim) in self.sims.iter().enumerate() {
            sim.write_obs(&mut flat[i * OBS_DIM..(i + 1) * OBS_DIM]);
        }
        PyArray1::from_vec_bound(py, flat)
    }

    /// actions: (N, 12) int64 — per hand (pitch, yaw, twist, elbow, roll,
    /// bend), left first. Returns (obs flat (N*286,) f32 — reshape
    /// Python-side, rewards (N,) f32, dones (N,) bool, events). events =
    /// [(env_idx, note_idx, code, points)] concatenated in env order with
    /// each env's own event ORDER preserved — order matters to consumers
    /// like the ScoreSaber combo/multiplier walk (a cut commits at its
    /// cut-time multiplier). Codes as in RawBeatSim::step. Done envs must
    /// get a fresh set_chart before the next step; their returned obs row
    /// is the terminal obs.
    fn step<'py>(
        &mut self,
        py: Python<'py>,
        actions: PyReadonlyArray2<'py, i64>,
    ) -> (
        Bound<'py, PyArray1<f32>>,
        Bound<'py, PyArray1<f32>>,
        Bound<'py, PyArray1<bool>>,
        Vec<(u32, u32, u8, f32, f32, f32)>,
    ) {
        let n = self.sims.len();
        let mut flat = vec![0.0f32; n * OBS_DIM];
        let (rewards, dones, all_events) = self.step_core(py, actions, &mut flat);
        (
            PyArray1::from_vec_bound(py, flat),
            PyArray1::from_vec_bound(py, rewards),
            PyArray1::from_vec_bound(py, dones),
            all_events,
        )
    }

    /// Like `step`, but writes the obs into a caller-owned C-contiguous
    /// (N, 286) f32 array instead of allocating a fresh 586 KB buffer per
    /// step — the buffer stays cache-resident, and a pinned torch buffer
    /// makes the following H2D copy an async DMA. Returns
    /// (rewards, dones, events).
    fn step_into<'py>(
        &mut self,
        py: Python<'py>,
        actions: PyReadonlyArray2<'py, i64>,
        mut out_obs: PyReadwriteArray2<'py, f32>,
    ) -> (
        Bound<'py, PyArray1<f32>>,
        Bound<'py, PyArray1<bool>>,
        Vec<(u32, u32, u8, f32, f32, f32)>,
    ) {
        let n = self.sims.len();
        assert_eq!(out_obs.shape(), &[n, OBS_DIM], "out_obs must be (N, OBS_DIM)");
        let mut out = out_obs.as_array_mut();
        let flat = out
            .as_slice_mut()
            .expect("out_obs must be C-contiguous");
        let (rewards, dones, all_events) = self.step_core(py, actions, flat);
        (
            PyArray1::from_vec_bound(py, rewards),
            PyArray1::from_vec_bound(py, dones),
            all_events,
        )
    }

    /// Single env's current obs (286,) — used for cheap post-reset reads.
    fn obs_one<'py>(&self, py: Python<'py>, i: usize) -> Bound<'py, PyArray1<f32>> {
        PyArray1::from_slice_bound(py, &self.sims[i].build_obs())
    }

    /// Single env's current obs written into row i of a caller-owned
    /// (N, 286) f32 array (the `step_into` buffer) — post-reset refresh
    /// without a per-call allocation.
    fn obs_one_into<'py>(
        &self,
        i: usize,
        mut out_obs: PyReadwriteArray2<'py, f32>,
    ) {
        assert_eq!(out_obs.shape(), &[self.sims.len(), OBS_DIM]);
        let mut out = out_obs.as_array_mut();
        let flat = out.as_slice_mut().expect("out_obs must be C-contiguous");
        self.sims[i].write_obs(&mut flat[i * OBS_DIM..(i + 1) * OBS_DIM]);
    }

    /// (good, bad, miss, n_scoreable, score, bombs_hit) for env i.
    fn counters(&self, i: usize) -> (u32, u32, u32, u32, f64, u32) {
        let s = &self.sims[i];
        (
            s.good_cuts,
            s.bad_cuts,
            s.misses,
            s.n_scoreable,
            s.score,
            s.bombs_hit,
        )
    }

    /// (pre_swing, accuracy, follow_through) point totals for env i.
    fn components(&self, i: usize) -> (f64, f64, f64) {
        let s = &self.sims[i];
        (s.score_pre, s.score_acc, s.score_post)
    }
}

impl VecBeatSim {
    /// Shared body of `step`/`step_into`: rayon-step every env, write obs
    /// rows into `flat`, and concatenate per-env events in index order.
    fn step_core(
        &mut self,
        py: Python<'_>,
        actions: PyReadonlyArray2<'_, i64>,
        flat: &mut [f32],
    ) -> (Vec<f32>, Vec<bool>, Vec<(u32, u32, u8, f32, f32, f32)>) {
        let acts = actions.as_array();
        let n = self.sims.len();
        assert_eq!(acts.shape(), &[n, 2 * N_JOINTS]);
        assert_eq!(flat.len(), n * OBS_DIM);
        // one flat memcpy (2026-07-27): the per-element ndarray indexing
        // built the per-env action arrays with ~12N bounds-checked reads
        // under the GIL; the u8 unpack now runs inside the parallel span
        let flat_acts: Vec<i64> = match acts.as_slice() {
            Some(s) => s.to_vec(),
            None => acts.iter().copied().collect(),
        };
        let ret_ev = self.return_events;
        let sims = &mut self.sims;
        let walks = &mut self.walks;
        // Rayon over envs (2026-07-25): envs are independent, each env's
        // combo walk is its own, and each env's event ORDER is kept in
        // its own vec — the index-order concatenation below makes the
        // output identical to the old serial loop (pinned by the
        // vec-matches-single and Rust==Python-walk tests). The GIL is
        // released for the span (plain Rust data only), so Python
        // threads — the bench's parse/download producer — truly overlap.
        // with_min_len: per-env work is ~1-3 us; without a floor rayon's
        // adaptive splitter can split toward single items, paying job +
        // atomic overhead and false-sharing obs rows (1144 B, not
        // line-aligned) across threads.
        let per_env: Vec<(f32, bool, Vec<(u32, u8, f32, f32, f32)>)> = py
            .allow_threads(|| {
                let walk_opts: Vec<Option<&mut ComboWalk>> = match walks {
                    Some(ws) => ws.iter_mut().map(Some).collect(),
                    None => (0..n).map(|_| None).collect(),
                };
                sims.par_iter_mut()
                    .zip_eq(walk_opts)
                    .zip_eq(flat.par_chunks_mut(OBS_DIM))
                    .zip_eq(flat_acts.par_chunks(2 * N_JOINTS))
                    .with_min_len(16)
                    .map(|(((sim, mut walk), obs_chunk), a)| {
                        let mut acts_e = [[0u8; N_JOINTS]; 2];
                        for s in 0..2 {
                            for j in 0..N_JOINTS {
                                acts_e[s][j] = a[s * N_JOINTS + j] as u8;
                            }
                        }
                        let (r, d) = sim.step_inner(acts_e);
                        sim.write_obs(obs_chunk);
                        if let Some(w) = walk.as_mut() {
                            for &(note, code, pts, _, _) in &sim.events_buf {
                                w.consume(note, code, pts);
                            }
                        }
                        // return_events=false (the bench): Vec::new() is
                        // alloc-free — the per-tick event Vec is gone
                        let ev = if ret_ev { sim.events_buf.clone() } else { Vec::new() };
                        (r as f32, d, ev)
                    })
                    .collect()
            });
        let mut rewards = vec![0.0f32; n];
        let mut dones = vec![false; n];
        let mut all_events: Vec<(u32, u32, u8, f32, f32, f32)> = Vec::new();
        for (i, (r, d, events)) in per_env.into_iter().enumerate() {
            rewards[i] = r;
            dones[i] = d;
            if self.return_events {
                for (note, code, pts, pre, acc) in events {
                    all_events.push((i as u32, note, code, pts, pre, acc));
                }
            }
        }
        (rewards, dones, all_events)
    }
}

#[pymodule]
fn beat_arena_rs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<RawBeatSim>()?;
    m.add_class::<VecBeatSim>()?;
    Ok(())
}
