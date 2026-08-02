//! Beat Arena sim core in Rust — THE physics implementation (SPEC.md +
//! SPEC-v10.md are the contract; `python/beat_arena_env/constants.py`
//! mirrors constants).
//!
//! Env v10 (P3, 2026-07-31): the v9 per-joint kinematic integrator and
//! 2-arm FK are RETIRED — the body is the SPEC-v10 full humanoid
//! (floating pelvis + 29 actuated revolutes, foot welds, self-collision)
//! stepped by the P1/P2 dynamics engine in `dynamics/`
//! (`solver::substep_constrained`, 8 × h = 1/240 per 30 Hz control
//! tick). Actions are 29 gear heads; τ_j = gear_fraction × τ_max[j],
//! held across the tick's substeps. Effort is positive mechanical work
//! (SPEC-v10 §4), replacing the v7.9 EFFORT_COST table.
//!
//! Blade segments come from the dynamics FK (hand frame +z, 1 m); the
//! swept-segment cut detection, 115-pt scoring, swing accumulator, and
//! miss/bomb logic are UNCHANGED in mechanism — only their inputs moved.
//! REACH/miss gates are live-pose functions (SPEC-v10 §4): reach =
//! max_hand(shoulder_z) + 0.65 + 1.0 + 0.75, miss boundary = head_z(t)
//! (both reproduce the v9 constants 1.75 / −0.65 at the reference pose).
//! Note look-at billboards to the LIVE head position.
//!
//! Obs is the SPEC-v10 §5 340-lane layout (offsets derived from the
//! block table below — no magic numbers). Replay is v10 (q36 + root
//! twist + ω29 + contact events with pair ids).
//!
//! Chart generation stays in Python (single source of truth, once per
//! episode); the chart is injected via `set_chart`.

use numpy::{PyArray1, PyReadonlyArray2, PyReadwriteArray2, PyUntypedArrayMethods};
use pyo3::prelude::*;
use rayon::prelude::*;

// env v10 dynamics core — SPEC-v10.md §1–3 (P1 math, P2 constraints).
pub mod dynamics;

use dynamics::collide::{NUM_PAIRS, PAIRS};
use dynamics::fk::quat_to_rot;
use dynamics::model::{
    Model, F_HEAD, F_L_HAND, F_L_SHOULDER, F_L_WRIST, F_R_HAND, F_R_SHOULDER,
    F_R_WRIST, HUMANOID, NQ, NV,
};
use dynamics::solver::{substep_constrained, ConstraintSet};
use dynamics::step::{H_SUB, N_SUB};
use dynamics::{State, Workspace};

const DT: f64 = 1.0 / 30.0;
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
// env v10 effort (SPEC-v10 §4): positive mechanical work,
//   effort_tick = effort_coef * sum_substeps sum_j max(tau_j*qd_j, 0)*h
// with the POST-SOLVE substep velocity (end of the substep, after the
// clamp passes). Replaces the v7.9 EFFORT_COST table. Default coef
// 4e-4 per joule (SPEC-v10 §4 calibration; flagged for P7 re-tune).
const EFFORT_COEF_DEFAULT: f64 = 4e-4;

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

const SABER_LEN: f64 = 1.0; // m (real Beat Saber)

// env v8 gears: action id -> signed torque fraction of tau_max (v10:
// the gear scales TORQUE, SPEC-v10 §4). Id 0 = coast, 1/2 = +/-full at
// EVERY table size (K=3 is the ternary era's prefix); finer gears
// append as +/- pairs. MIRROR of constants.gear_table — the max table
// (K=7) is stored, a sim's `gears` field bounds the valid ids.
const GEAR_SIGNED: [f64; 7] = [0.0, 1.0, -1.0, 1.0 / 3.0, -1.0 / 3.0,
                               0.125, -0.125];
const GEAR_SIGNED_5: [f64; 7] = [0.0, 1.0, -1.0, 0.25, -0.25, 0.0, 0.0];

fn gear_signed(gears: u8) -> [f64; 7] {
    match gears {
        5 => GEAR_SIGNED_5,
        7 => GEAR_SIGNED,
        _ => [0.0, 1.0, -1.0, 0.0, 0.0, 0.0, 0.0], // ternary (K=3)
    }
}

const N_BLADE_SAMPLES: usize = 8; // contact-point/velocity samples
// live broad-phase reach gate (SPEC-v10 §4): note_z <=
// max_hand(shoulder_z(t)) + 0.65 + 1.0 + 0.75 — the arm span past the
// shoulder (0.65) + blade (1.0) + good-box near-face reach (0.75).
// At the reference pose (shoulder_z = -0.65) this is v9's REACH_Z 1.75.
const REACH_PAST_SHOULDER: f64 = 0.65 + 1.0 + 0.75;
// miss boundary: note consumed as missed past the LIVE head plane
// z < head_z(t) (v9's -Z_MISS = -0.65 at the reference pose). The
// episode tail must cover the slowest object's trip from the hit plane
// to the WORST-CASE head plane; the head cannot physically reach
// z < -1.5 on the welded-feet body (waist z -0.65, head lever 1.09 m
// fully folded back still > -1.8, and the spine/hip limits stop far
// short) — P3 divergence note, SPEC-v10 Divergences §18.
const Z_MISS_TAIL: f64 = 1.5;

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

// --- obs layout v10 (SPEC-v10 §5; offsets DERIVED, order is a contract) ------
const N_NOTE_SLOTS: usize = 16;
const NOTE_FEAT: usize = 14;
const HAND_FEAT: usize = 10; // wrist pos(3) + wrist vel(3) + blade(3) + meter
const Q_LANES: usize = NQ + 2; // 31: the two wrist rolls code as cos/sin
const OFF_ROOT_POS: usize = 0; // x/1, (y-0.94)/0.75, (z+0.65)/1
const OFF_ROOT_ROT: usize = OFF_ROOT_POS + 3; // 6D two-column rotation
const OFF_ROOT_VLIN: usize = OFF_ROOT_ROT + 6; // /6
const OFF_ROOT_VANG: usize = OFF_ROOT_VLIN + 3; // world, /16
const OFF_Q: usize = OFF_ROOT_VANG + 3; // 15
const OFF_QD: usize = OFF_Q + Q_LANES; // 46: qd_j / omega_cap[j]
const OFF_HAND_L: usize = OFF_QD + NQ; // 75
const OFF_HAND_R: usize = OFF_HAND_L + HAND_FEAT; // 85
const OFF_HEAD: usize = OFF_HAND_R + HAND_FEAT; // 95
const OFF_CONTACT: usize = OFF_HEAD + 3; // 98: contact bits [L, R]
const OFF_NOTES: usize = OFF_CONTACT + 2; // 100
const OFF_NOTE_MASK: usize = OFF_NOTES + N_NOTE_SLOTS * NOTE_FEAT; // 324
const OBS_DIM: usize = OFF_NOTE_MASK + N_NOTE_SLOTS; // 340
const _: () = assert!(OBS_DIM == 340, "SPEC-v10 §5: OBS_DIM must be 340");
const _: () = assert!(OFF_QD == 46 && OFF_HAND_L == 75 && OFF_NOTES == 100);

const X_NORM: f64 = 1.0;
const Y_NORM: f64 = 0.75;
const Z_SCALE: f64 = 8.0;
const Z_CLIP: (f64, f64) = (-3.0, 12.0);
const ROOT_Y0: f64 = 0.94; // reference root height (obs centering)
const HEAD_Y0: f64 = 1.70; // reference head height (obs centering)
const PLANE_Z0: f64 = -0.65; // reference body plane (obs centering)
const ROOT_VLIN_NORM: f64 = 6.0;
const ROOT_VANG_NORM: f64 = 16.0;

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
    sin_a: f64,
    cos_a: f64,
    speed: f64,     // approach speed toward the plane (m/s)
    any_dir: bool,  // dot note / chain link: any swing direction scores
    kind: u8,       // KIND_NOTE / _BOMB / _CHAIN_HEAD / _CHAIN_LINK
    free_pre: bool, // arc tail: full 70 pre-swing granted
    free_post: bool, // arc head: full 30 follow-through granted
}

// --- v9 reference arm FK (TEST-ONLY since P3) --------------------------------
// The v10 continuity gate (dynamics/tests_fk.rs `v9_fk_continuity`)
// pins the dynamics tree against the retired v9 arm chains; the v9
// `fk()` survives as a #[cfg(test)] reference implementation for that
// gate (SPEC-v10 Divergences §18). Nothing in the live env calls it.
#[cfg(test)]
const SHOULDER: [[f64; 3]; 2] = [[-0.2, 1.47, -0.65], [0.2, 1.47, -0.65]];
#[cfg(test)]
const L_UPPER: f64 = 0.35;
#[cfg(test)]
const L_FORE: f64 = 0.30;

/// v9 forward kinematics: (elbow, wrist, blade unit dir, (cos, sin) of
/// roll) from q = [pitch, yaw, twist, elbow, roll, bend, dev].
/// Arm frame M = R_y(yaw)·R_x(−pitch)·R_z(twist); forearm =
/// M·(0, sin e, cos e); blade = Rz(roll)·Ry(dev)·Rx(−bend)·e_z in the
/// forearm frame M_f = M·R_x(−elbow).
#[cfg(test)]
pub fn fk(hand: usize, q: &[f64; 7]) -> ([f64; 3], [f64; 3], [f64; 3], (f64, f64)) {
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
    let (sv, cv) = q[6].sin_cos();
    let d = [cr * sv * cb - sr * sb, sr * sv * cb + cr * sb, cv * cb];
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
// axis by rot = angle - 270 deg. Env v7.6: plain notes additionally
// billboard toward the player's head as they approach; env v10 the
// head anchor is the LIVE FK head position (SPEC-v10 §4).
const LOOK_VERT: f64 = 0.8; // verticalLookStrength: head.y 80% toward note.y
const LOOK_DIST: f64 = 8.0; // full head-facing at z=0, none beyond 8 m

/// Orthonormal note frame columns (x', y', z'): z' = the blended
/// forward, up from the z-roll, x' = up x z'. `head` is the live FK
/// head position (env v10).
fn note_frame(
    p: [f64; 3],
    cos_r: f64,
    sin_r: f64,
    lam: f64,
    head: [f64; 3],
) -> [[f64; 3]; 3] {
    let up = [-sin_r, cos_r, 0.0];
    let hp = [head[0], head[1] + LOOK_VERT * (p[1] - head[1]), head[2]];
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

/// A good cut's pending follow-through meter.
#[derive(Clone, Copy)]
struct Tracker {
    note: u32,
    acc: f64,
    age: u32,
}

type ChartTuple = (f64, f64, f64, u8, f64, f64, bool, u8, bool, bool);

/// The v10 physical body: dynamics state + solver buffers + the derived
/// per-tick caches the judgment/obs layers consume. One per env.
struct Body {
    st: State,
    ws: Workspace,
    cs: ConstraintSet,
    tau: [f64; NQ], // this tick's applied torques (constant across substeps)
    // FK caches (refreshed after every tick / state write)
    wrist: [[f64; 3]; 2],
    blade: [[f64; 3]; 2],
    head: [f64; 3],
    shoulder_z: [f64; 2],
    // judgment caches
    wrist_vel: [[f64; 3]; 2], // finite difference across the last tick
    swing_acc: [f64; 2],      // arc of the current continuous swing (rad)
    axis: [Option<[f64; 3]>; 2], // last tick's rotation axis (unit)
    // per-tick contact aggregation (obs bits + replay events)
    contact_arm: [bool; 2],          // any active contact row touching the arm
    contact_imp: [f64; NUM_PAIRS],   // tick-summed normal impulse per pair
    contact_pt: [[f64; 3]; NUM_PAIRS], // last substep's contact point
    effort_j: f64,                   // last tick's positive work (J)
    // P6 (additive introspection): tick-summed foot-weld impulses (N·s),
    // solver rows 0–11 — [foot_l lin xyz, ang xyz, foot_r lin, ang].
    // λ is the constraint impulse ON the body (world frame): the ground
    // reaction; vertical force ≈ weld_imp[1 or 7] / DT.
    weld_imp: [f64; 12],
}

const MODEL: &Model = &HUMANOID;

impl Body {
    fn new() -> Body {
        let st = State::neutral(MODEL);
        let ws = Workspace::new(MODEL);
        let cs = ConstraintSet::new(MODEL, true, true);
        let mut b = Body {
            st,
            ws,
            cs,
            tau: [0.0; NQ],
            wrist: [[0.0; 3]; 2],
            blade: [[0.0; 3]; 2],
            head: [0.0; 3],
            shoulder_z: [0.0; 2],
            wrist_vel: [[0.0; 3]; 2],
            swing_acc: [0.0; 2],
            axis: [None; 2],
            contact_arm: [false; 2],
            contact_imp: [0.0; NUM_PAIRS],
            contact_pt: [[0.0; 3]; NUM_PAIRS],
            effort_j: 0.0,
            weld_imp: [0.0; 12],
        };
        b.refresh_fk();
        b
    }

    fn reset(&mut self) {
        self.st = State::neutral(MODEL);
        self.tau = [0.0; NQ];
        self.wrist_vel = [[0.0; 3]; 2];
        self.swing_acc = [0.0; 2];
        self.axis = [None; 2];
        self.contact_arm = [false; 2];
        self.contact_imp = [0.0; NUM_PAIRS];
        self.effort_j = 0.0;
        self.weld_imp = [0.0; 12];
        self.refresh_fk();
    }

    /// FK caches from the current state: wrist = hand-frame origin,
    /// blade = hand-frame +z column (SPEC-v10 §1.3), live head +
    /// shoulder anchors for the reach/miss gates.
    fn refresh_fk(&mut self) {
        dynamics::fk::fk_all(MODEL, &self.st, &mut self.ws);
        for (s, (fw, fh)) in [(F_L_WRIST, F_L_HAND), (F_R_WRIST, F_R_HAND)]
            .into_iter()
            .enumerate()
        {
            self.wrist[s] = self.ws.p[fw];
            let r = &self.ws.r[fh];
            self.blade[s] = [r[0][2], r[1][2], r[2][2]];
        }
        self.head = self.ws.p[F_HEAD];
        self.shoulder_z = [self.ws.p[F_L_SHOULDER][2], self.ws.p[F_R_SHOULDER][2]];
    }

    /// One 30 Hz control tick: gear ids -> torques, 8 × substep at
    /// h = 1/240 (SPEC-v10 §3/§4), accumulating positive work and
    /// contact impulses, then FK-cache refresh + finite-diff wrist vel.
    fn tick(&mut self, actions: &[u8; NQ], gear: &[f64; 7]) {
        for j in 0..NQ {
            let a = actions[j] as usize;
            let frac = if a < 7 { gear[a] } else { 0.0 };
            self.tau[j] = frac * MODEL.tau_max[j];
        }
        let prev_wrist = self.wrist;
        self.contact_arm = [false; 2];
        self.contact_imp = [0.0; NUM_PAIRS];
        self.weld_imp = [0.0; 12];
        let mut effort = 0.0;
        for _ in 0..N_SUB {
            substep_constrained(MODEL, &mut self.st, &self.tau, H_SUB,
                                &mut self.ws, &mut self.cs);
            // SPEC-v10 §4: positive mechanical work at the post-solve
            // substep velocity (tau constant over the tick)
            for j in 0..NQ {
                let p = self.tau[j] * self.st.v[6 + j];
                if p > 0.0 {
                    effort += p * H_SUB;
                }
            }
            // contact aggregation: obs bits see every ACTIVE row this
            // substep; replay impulses sum the row lambdas (N·s)
            let nweld = self.cs.weld_rows();
            // P6: tick-summed foot-weld impulses (ground reaction)
            for r in 0..nweld {
                self.weld_imp[r] += self.cs.lambda[r];
            }
            for (i, c) in self.cs.contacts.iter().enumerate() {
                let lam = self.cs.lambda[nweld + i];
                self.contact_imp[c.pair] += lam;
                self.contact_pt[c.pair] = c.point;
                let (ia, ib) = PAIRS[c.pair];
                for id in [ia, ib] {
                    match id {
                        2 | 4 | 6 => self.contact_arm[0] = true,
                        3 | 5 | 7 => self.contact_arm[1] = true,
                        _ => {}
                    }
                }
            }
        }
        self.effort_j = effort;
        self.refresh_fk();
        for s in 0..2 {
            for k in 0..3 {
                self.wrist_vel[s][k] = (self.wrist[s][k] - prev_wrist[s][k]) / DT;
            }
        }
    }

    /// Live broad-phase reach gate (SPEC-v10 §4).
    fn reach_z(&self) -> f64 {
        self.shoulder_z[0].max(self.shoulder_z[1]) + REACH_PAST_SHOULDER
    }
}

struct Core {
    max_steps: u32,
    effort_coef: f64, // per-joule work price (0 = free movement)
    gear: [f64; 7],   // env v8: action id -> signed torque fraction
    wide_acc: bool,   // TRAINING-ONLY wide accuracy reward (see cut scoring)
    chart: Vec<Note>,
    tail_s: f64, // episode tail after the last hit_time (covers miss exit)
    min_speed: f64, // slowest object; (hit_time-t)*min_speed > reach_z is a
                    // conservative exact "beyond reach" bound for a sorted chart
    // state
    tick: u32,
    time: f64,
    first_pending: usize, // smallest index with status 0 (monotone per episode)
    body: Body,
    trackers: [Vec<Tracker>; 2],
    status: Vec<u8>, // 0 pending, 1 good, 2 bad/bomb-hit, 3 missed, 4 bomb passed
    good_cuts: u32,
    bad_cuts: u32,
    misses: u32,
    bombs_hit: u32,
    n_scoreable: u32, // chart objects that are not bombs
    score: f64,       // summed points; max = 115*notes + 85*heads + 20*links
    score_pre: f64,
    score_acc: f64,
    score_post: f64,
    // reusable event buffer: cleared and refilled in place each tick.
    events_buf: Vec<(u32, u8, f32, f32, f32)>,
}

impl Core {
    fn new_core(max_steps: u32, effort_coef: f64, wide_acc: bool,
                gears: u8) -> Self {
        Core {
            max_steps,
            effort_coef,
            gear: gear_signed(gears),
            wide_acc,
            chart: Vec::new(),
            tail_s: 0.5,
            min_speed: f64::INFINITY,
            tick: 0,
            time: 0.0,
            first_pending: 0,
            body: Body::new(),
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
        // plane to the WORST-CASE miss boundary (live head plane, env
        // v10), or final misses never register
        let min_speed = self
            .chart
            .iter()
            .map(|n| n.speed)
            .fold(f64::INFINITY, f64::min);
        self.min_speed = min_speed;
        self.tail_s = if min_speed.is_finite() {
            0.5 + Z_MISS_TAIL / min_speed
        } else {
            0.5
        };
        self.reset_state();
    }

    fn reset_state(&mut self) {
        self.tick = 0;
        self.time = 0.0;
        self.first_pending = 0;
        self.body.reset();
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

    /// Obs written into a caller-owned OBS_DIM slice — SPEC-v10 §5
    /// layout, offsets from the derived consts above.
    fn write_obs(&self, obs: &mut [f32]) {
        debug_assert_eq!(obs.len(), OBS_DIM);
        obs.fill(0.0);
        let st = &self.body.st;
        // root position (centered on the reference pose)
        obs[OFF_ROOT_POS] = (st.p[0] / X_NORM) as f32;
        obs[OFF_ROOT_POS + 1] = ((st.p[1] - ROOT_Y0) / Y_NORM) as f32;
        obs[OFF_ROOT_POS + 2] = ((st.p[2] - PLANE_Z0) / X_NORM) as f32;
        // root orientation, 6D two-column: [R00,R10,R20, R01,R11,R21]
        let r = quat_to_rot(st.quat);
        for i in 0..3 {
            obs[OFF_ROOT_ROT + i] = (r[i][0]) as f32;
            obs[OFF_ROOT_ROT + 3 + i] = (r[i][1]) as f32;
        }
        for k in 0..3 {
            obs[OFF_ROOT_VLIN + k] = (st.v[k] / ROOT_VLIN_NORM) as f32;
            obs[OFF_ROOT_VANG + k] = (st.v[3 + k] / ROOT_VANG_NORM) as f32;
        }
        // q block (31 lanes): per-joint normalized angle in §1.4 order;
        // the two wrist rolls keep the v9 cos/sin coding; the one-way
        // elbows/knees keep the elbow-style (2q − hi)/hi; everything
        // else is q / max(|lo|, |hi|) — all derived from model.limits
        let mut lane = OFF_Q;
        for j in 0..NQ {
            let q = st.q[j];
            match j {
                7 | 14 => {
                    // l/r wrist roll: cos/sin (the only wrapping joints)
                    let (s, c) = q.sin_cos();
                    obs[lane] = c as f32;
                    obs[lane + 1] = s as f32;
                    lane += 2;
                }
                6 | 13 | 20 | 26 => {
                    // l/r elbow, l/r knee: one-way (2q − hi)/hi
                    let hi = MODEL.limits[j].1;
                    obs[lane] = ((2.0 * q - hi) / hi) as f32;
                    lane += 1;
                }
                _ => {
                    let (lo, hi) = MODEL.limits[j];
                    obs[lane] = (q / lo.abs().max(hi.abs())) as f32;
                    lane += 1;
                }
            }
        }
        debug_assert_eq!(lane, OFF_QD);
        for j in 0..NQ {
            obs[OFF_QD + j] = (st.v[6 + j] / MODEL.omega_cap[j]) as f32;
        }
        // hand derived blocks (v9 semantics on the v10 FK)
        for (s, base) in [(0usize, OFF_HAND_L), (1usize, OFF_HAND_R)] {
            let b = &self.body;
            obs[base] = (b.wrist[s][0] / X_NORM) as f32;
            obs[base + 1] = ((b.wrist[s][1] - Y_CENTER) / Y_NORM) as f32;
            obs[base + 2] = b.wrist[s][2] as f32;
            for k in 0..3 {
                obs[base + 3 + k] = (b.wrist_vel[s][k] / V_NORM) as f32;
                obs[base + 6 + k] = b.blade[s][k] as f32;
            }
            obs[base + 9] = (b.swing_acc[s] / PRE_FULL).min(2.0) as f32;
        }
        // head position (FK, centered on the reference pose)
        obs[OFF_HEAD] = (self.body.head[0] / X_NORM) as f32;
        obs[OFF_HEAD + 1] = ((self.body.head[1] - HEAD_Y0) / Y_NORM) as f32;
        obs[OFF_HEAD + 2] = ((self.body.head[2] - PLANE_Z0) / X_NORM) as f32;
        // contact bits [L, R]
        obs[OFF_CONTACT] = self.body.contact_arm[0] as u8 as f32;
        obs[OFF_CONTACT + 1] = self.body.contact_arm[1] as u8 as f32;
        // note block — v9 NOTE_FEAT verbatim
        let mut slot = 0usize;
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
    // v9 sim produced them — order is the ComboWalk contract.
    fn step_inner(&mut self, actions: &[u8; NQ]) -> (f64, bool) {
        let prev_pos = self.body.wrist;
        let prev_dir = self.body.blade;

        let mut reward = 0.0;
        self.events_buf.clear();
        // per arm this tick: (swing delta, did the chain continue)
        let mut deltas = [0.0f64; 2];
        let mut continued = [false; 2];

        // ── the physics tick (SPEC-v10 §3/§4) ─────────────────────────
        let gear = self.gear;
        self.body.tick(actions, &gear);
        reward -= self.effort_coef * self.body.effort_j;

        // swing chain update (feeds pre-swing + follow-through scores)
        for s in 0..2 {
            let d_old = prev_dir[s];
            let d_new = self.body.blade[s];
            let dot = (d_old[0] * d_new[0] + d_old[1] * d_new[1]
                + d_old[2] * d_new[2])
                .clamp(-1.0, 1.0);
            let delta = dot.acos();
            deltas[s] = delta;
            if delta < SWING_BREAK {
                self.body.swing_acc[s] = 0.0;
                self.body.axis[s] = None;
            } else {
                let cx = cross3(d_old, d_new);
                let cn = (cx[0].powi(2) + cx[1].powi(2) + cx[2].powi(2)).sqrt();
                let ax = [cx[0] / cn, cx[1] / cn, cx[2] / cn];
                match self.body.axis[s] {
                    Some(a0) if a0[0] * ax[0] + a0[1] * ax[1] + a0[2] * ax[2] < 0.0 => {
                        self.body.swing_acc[s] = delta; // reversal: new swing
                    }
                    _ => {
                        self.body.swing_acc[s] += delta;
                        continued[s] = true;
                    }
                }
                self.body.axis[s] = Some(ax);
            }
        }

        // follow-through trackers (existing ones, BEFORE this tick's cuts)
        for s in 0..2 {
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
            sweep_tris(prev_pos[0], prev_dir[0], self.body.wrist[0], self.body.blade[0]),
            sweep_tris(prev_pos[1], prev_dir[1], self.body.wrist[1], self.body.blade[1]),
        ];

        // live gates (SPEC-v10 §4)
        let reach_z = self.body.reach_z();
        let head_z = self.body.head[2];
        let head = self.body.head;

        for i in self.first_pending..self.chart.len() {
            if self.status[i] != 0 {
                continue;
            }
            let n = &self.chart[i]; // by ref: Note is 80 B post-sin/cos
            // sorted by hit_time, and speed >= min_speed for every note:
            // once this bound clears reach_z, so does every later note's z
            if (n.hit_time - self.time) * self.min_speed > reach_z {
                break;
            }
            let z = (n.hit_time - self.time) * n.speed;
            if z < head_z {
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
            if z > reach_z {
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
            let rotates = n.kind == KIND_CHAIN_LINK || !n.any_dir;
            // rot = angle - 270 deg: cos(a - 3pi/2) = -sin a,
            // sin(a - 3pi/2) = cos a — from the precomputed pair
            let (cos_r, sin_r) = if rotates { (-n.sin_a, n.cos_a) } else { (1.0, 0.0) };
            // env v7.6: plain notes billboard toward the LIVE head as
            // they approach; heads/links keep the pure z-roll
            let lam = if n.kind == KIND_NOTE {
                (1.0 - z / LOOK_DIST).clamp(0.0, 1.0)
            } else {
                0.0
            };
            let frame = note_frame(center, cos_r, sin_r, lam, head);
            // the good box's 0.25 m shift is along the note's LOCAL z
            let good_center = [
                center[0] + GOOD_Z_SHIFT * frame[2][0],
                center[1] + GOOD_Z_SHIFT * frame[2][1],
                center[2] + GOOD_Z_SHIFT * frame[2][2],
            ];

            if sweep_hits_box(&tris[m], good_center, good_h, &frame) {
                // contact point: the blade sample whose displacement
                // segment passes closest to the note center; its
                // finite-difference velocity is the contact velocity
                let wrist_m = self.body.wrist[m];
                let blade_m = self.body.blade[m];
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
                        wrist_m[0] + r * blade_m[0],
                        wrist_m[1] + r * blade_m[1],
                        wrist_m[2] + r * blade_m[2],
                    ];
                    let d = seg_point_dist3(p0, p1, center);
                    if d < d_seg {
                        d_seg = d;
                        s_star = r;
                    }
                }
                let vc = [
                    (wrist_m[0] + s_star * blade_m[0]
                        - prev_pos[m][0] - s_star * prev_dir[m][0]) / DT,
                    (wrist_m[1] + s_star * blade_m[1]
                        - prev_pos[m][1] - s_star * prev_dir[m][1]) / DT,
                    (wrist_m[2] + s_star * blade_m[2]
                        - prev_pos[m][2] - s_star * prev_dir[m][2]) / DT,
                ];
                let vc_mag = (vc[0].powi(2) + vc[1].powi(2) + vc[2].powi(2)).sqrt();
                // accuracy distance: note center to the INFINITE cutting
                // plane (blade dir x tip movement; stab fallback = blade
                // line) — the 2026-07-29 cut-plane rule, unchanged
                let tip_move = [
                    wrist_m[0] + SABER_LEN * blade_m[0]
                        - prev_pos[m][0] - SABER_LEN * prev_dir[m][0],
                    wrist_m[1] + SABER_LEN * blade_m[1]
                        - prev_pos[m][1] - SABER_LEN * prev_dir[m][1],
                    wrist_m[2] + SABER_LEN * blade_m[2]
                        - prev_pos[m][2] - SABER_LEN * prev_dir[m][2],
                ];
                let cw = [
                    center[0] - wrist_m[0],
                    center[1] - wrist_m[1],
                    center[2] - wrist_m[2],
                ];
                let pn = cross3(blade_m, tip_move);
                let pn_mag = dot3(pn, pn).sqrt();
                let d_min = if pn_mag > 1e-9 {
                    (dot3(cw, pn) / pn_mag).abs()
                } else {
                    let along = dot3(cw, blade_m);
                    (dot3(cw, cw) - along * along).max(0.0).sqrt()
                };
                let code: u8 = if vc_mag < V_MIN_CUT {
                    1 // slow
                } else if n.any_dir {
                    0 // dot note / chain link: any direction scores
                } else {
                    // direction check in the NOTE'S LOCAL frame —
                    // required direction is local -y'
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
                            round_pts(PTS_PRE
                                * (self.body.swing_acc[m] / PRE_FULL).min(1.0))
                        };
                        // wide_acc (TRAINING ONLY): 30·(1−d/0.6), unrounded
                        let acc = if self.wide_acc {
                            2.0 * PTS_ACC
                                * (1.0 - (d_min / (2.0 * ACC_NORM)).clamp(0.0, 1.0))
                        } else {
                            round_pts(PTS_ACC * (1.0 - (d_min / ACC_NORM).clamp(0.0, 1.0)))
                        };
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
        // advance the low-water cursor past the newly resolved prefix
        while self.first_pending < self.chart.len()
            && self.status[self.first_pending] != 0
        {
            self.first_pending += 1;
        }
        let last_hit = self.chart.last().map_or(0.0, |n| n.hit_time);
        let pending = self.first_pending < self.chart.len();
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
    #[pyo3(signature = (max_steps=690, effort_coef=EFFORT_COEF_DEFAULT,
                        wide_acc=false, gears=3))]
    fn new(max_steps: u32, effort_coef: f64, wide_acc: bool,
           gears: u8) -> Self {
        RawBeatSim {
            core: Core::new_core(max_steps, effort_coef, wide_acc, gears),
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

    /// actions: 29 gear ids in SPEC-v10 §1.4 joint order — spine(3),
    /// arm_l(7), arm_r(7), leg_l(6), leg_r(6); id 0 = coast, 1/2 =
    /// +/-full torque, finer +/- gear pairs append (constants.gear_table).
    /// Returns (obs, reward, done, events); events = [(note_idx, code,
    /// points, pre_pts, acc_pts)] with codes 0 ok, 1 slow, 2 angle,
    /// 3 wrong_hand, 4 miss, 5 post, 6 bomb.
    fn step<'py>(
        &mut self,
        py: Python<'py>,
        actions: Vec<u8>,
    ) -> (Bound<'py, PyArray1<f32>>, f64, bool, Vec<(u32, u8, f32, f32, f32)>) {
        assert_eq!(actions.len(), NQ);
        assert!(actions.iter().all(|&a| a <= 6)); // gear ids 0..=6
        let mut acts = [0u8; NQ];
        acts.copy_from_slice(&actions);
        let (reward, done) = self.core.step_inner(&acts);
        (
            PyArray1::from_slice_bound(py, &self.core.build_obs()),
            reward,
            done,
            self.core.events_buf.clone(),
        )
    }

    /// Current observation without stepping (state introspection after
    /// `set_joints`; the obs the next step would be built FROM).
    fn obs<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f32>> {
        PyArray1::from_slice_bound(py, &self.core.build_obs())
    }

    /// Replay v10 `q`: the 36 position coordinates — root position (3),
    /// root quaternion wxyz (4), then the 29 joint angles (§1.4 order).
    fn qpos(&self) -> Vec<f64> {
        let st = &self.core.body.st;
        let mut out = Vec::with_capacity(36);
        out.extend_from_slice(&st.p);
        out.extend_from_slice(&st.quat);
        out.extend_from_slice(&st.q);
        out
    }

    /// Replay v10 `twist`: root world-frame [vx, vy, vz, wx, wy, wz].
    fn twist(&self) -> [f64; 6] {
        let v = &self.core.body.st.v;
        [v[0], v[1], v[2], v[3], v[4], v[5]]
    }

    /// Replay v10 `omega`: the 29 joint rates (§1.4 order).
    fn omega(&self) -> Vec<f64> {
        self.core.body.st.v[6..].to_vec()
    }

    /// Replay v10 `sabers`: per hand [L, R] the FK wrist position and
    /// unit blade direction — [wx, wy, wz, bx, by, bz].
    fn sabers(&self) -> [[f64; 6]; 2] {
        [0, 1].map(|s| {
            let w = self.core.body.wrist[s];
            let b = self.core.body.blade[s];
            [w[0], w[1], w[2], b[0], b[1], b[2]]
        })
    }

    /// Blade unit directions [[x, y, z]; 2] (FK cache).
    fn blades(&self) -> [[f64; 3]; 2] {
        [0, 1].map(|s| self.core.body.blade[s])
    }

    /// Replay v10 `contacts` of the LAST tick: (pair_id, px, py, pz, j)
    /// per live pair with nonzero tick-summed normal impulse (N·s);
    /// the point is the last substep's world contact point.
    fn contacts(&self) -> Vec<(u32, f64, f64, f64, f64)> {
        let b = &self.core.body;
        (0..NUM_PAIRS)
            .filter(|&p| b.contact_imp[p] != 0.0)
            .map(|p| {
                let pt = b.contact_pt[p];
                (p as u32, pt[0], pt[1], pt[2], b.contact_imp[p])
            })
            .collect()
    }

    /// P6 (additive): tick-summed foot-weld impulses of the LAST tick,
    /// per foot [L, R]: [jx, jy, jz, jax, jay, jaz] (N·s / N·m·s, world
    /// frame) — the ground reaction through each welded foot. Vertical
    /// reaction force ≈ j[1] / DT.
    fn weld_impulses(&self) -> [[f64; 6]; 2] {
        let w = &self.core.body.weld_imp;
        [
            [w[0], w[1], w[2], w[3], w[4], w[5]],
            [w[6], w[7], w[8], w[9], w[10], w[11]],
        ]
    }

    /// Per-note status: 0 pending, 1 good, 2 bad/bomb-hit, 3 missed,
    /// 4 bomb passed clean.
    fn status(&self) -> Vec<u8> {
        self.core.status.clone()
    }

    /// TEST/TOOLING HOOK (env v10): overwrite the FULL generalized
    /// state — q: 36 position coords (root pos, root quat wxyz, 29
    /// joints), v: 35 velocity coords (root linear world, root angular
    /// world, 29 joint rates). FK caches refresh; the finite-diff wrist
    /// velocity resets; swing accumulators are left untouched.
    fn set_joints(&mut self, q: Vec<f64>, v: Vec<f64>) {
        assert_eq!(q.len(), 3 + 4 + NQ, "q must be 36 (root pose + 29)");
        assert_eq!(v.len(), NV, "v must be 35 (root twist + 29)");
        let body = &mut self.core.body;
        body.st.p = [q[0], q[1], q[2]];
        let qn = (q[3] * q[3] + q[4] * q[4] + q[5] * q[5] + q[6] * q[6]).sqrt();
        assert!(qn > 1e-9, "root quaternion must be nonzero");
        body.st.quat = [q[3] / qn, q[4] / qn, q[5] / qn, q[6] / qn];
        body.st.q.copy_from_slice(&q[7..]);
        body.st.v.copy_from_slice(&v);
        body.refresh_fk();
        body.wrist_vel = [[0.0; 3]; 2];
    }

    /// Last tick's effort in JOULES (positive mechanical work, SPEC-v10
    /// §4) — the reward charged effort_coef × this. Test introspection.
    fn last_effort(&self) -> f64 {
        self.core.body.effort_j
    }

    /// f64 Cholesky pivot-floor engagements since the last pop
    /// (SPEC-v10 Divergences §22) — the CUDA backend's pop_pd_hits
    /// mirror. Returns the count and zeroes it. ADDITIVE getter; 0 in
    /// every state the strict pre-§22 factorization survived.
    fn pop_pd_hits(&mut self) -> u64 {
        let h = self.core.body.ws.pd_hits;
        self.core.body.ws.pd_hits = 0;
        h
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
    #[pyo3(signature = (n, max_steps=690, effort_coef=EFFORT_COEF_DEFAULT,
                        combo_walk=false, return_events=true, wide_acc=false,
                        gears=3))]
    fn new(n: usize, max_steps: u32, effort_coef: f64, combo_walk: bool,
           return_events: bool, wide_acc: bool, gears: u8) -> Self {
        VecBeatSim {
            sims: (0..n)
                .map(|_| Core::new_core(max_steps, effort_coef, wide_acc, gears))
                .collect(),
            walks: combo_walk.then(|| (0..n).map(|_| ComboWalk::new(0)).collect()),
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

    /// Current observation of every env, flat (N*340,) — reshape Python-side.
    fn obs<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f32>> {
        let n = self.sims.len();
        let mut flat = vec![0.0f32; n * OBS_DIM];
        for (i, sim) in self.sims.iter().enumerate() {
            sim.write_obs(&mut flat[i * OBS_DIM..(i + 1) * OBS_DIM]);
        }
        PyArray1::from_vec_bound(py, flat)
    }

    /// actions: (N, 29) int64 gear ids, SPEC-v10 §1.4 joint order.
    /// Returns (obs flat (N*340,) f32 — reshape Python-side, rewards
    /// (N,) f32, dones (N,) bool, events). events = [(env_idx, note_idx,
    /// code, points, pre, acc)] concatenated in env order with each
    /// env's own event ORDER preserved. Done envs must get a fresh
    /// set_chart before the next step; their returned obs row is the
    /// terminal obs.
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
    /// (N, 340) f32 array instead of allocating a fresh buffer per
    /// step. Returns (rewards, dones, events).
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

    /// Single env's current obs (340,) — used for cheap post-reset reads.
    fn obs_one<'py>(&self, py: Python<'py>, i: usize) -> Bound<'py, PyArray1<f32>> {
        PyArray1::from_slice_bound(py, &self.sims[i].build_obs())
    }

    /// Single env's current obs written into row i of a caller-owned
    /// (N, 340) f32 array (the `step_into` buffer).
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

    /// f64 Cholesky pivot-floor engagements summed over all envs since
    /// the last pop (SPEC-v10 Divergences §22) — the CUDA backend's
    /// pop_pd_hits mirror. Returns the count and zeroes the per-env
    /// counters. ADDITIVE getter (no existing FFI shape changed); 0 in
    /// every state the strict pre-§22 factorization survived.
    fn pop_pd_hits(&mut self) -> u64 {
        self.sims
            .iter_mut()
            .map(|s| {
                let h = s.body.ws.pd_hits;
                s.body.ws.pd_hits = 0;
                h
            })
            .sum()
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
        assert_eq!(acts.shape(), &[n, NQ]);
        assert_eq!(flat.len(), n * OBS_DIM);
        let flat_acts: Vec<i64> = match acts.as_slice() {
            Some(s) => s.to_vec(),
            None => acts.iter().copied().collect(),
        };
        let ret_ev = self.return_events;
        let sims = &mut self.sims;
        let walks = &mut self.walks;
        // Rayon over envs: envs are independent; the index-order
        // concatenation keeps the output identical to a serial loop
        // (pinned by the vec-matches-single test). The GIL is released
        // for the span. v10: per-env work is ~0.5 ms (dynamics), so no
        // with_min_len floor — full splitting is wanted now.
        let per_env: Vec<(f32, bool, Vec<(u32, u8, f32, f32, f32)>)> = py
            .allow_threads(|| {
                let walk_opts: Vec<Option<&mut ComboWalk>> = match walks {
                    Some(ws) => ws.iter_mut().map(Some).collect(),
                    None => (0..n).map(|_| None).collect(),
                };
                sims.par_iter_mut()
                    .zip_eq(walk_opts)
                    .zip_eq(flat.par_chunks_mut(OBS_DIM))
                    .zip_eq(flat_acts.par_chunks(NQ))
                    .map(|(((sim, mut walk), obs_chunk), a)| {
                        let mut acts_e = [0u8; NQ];
                        for j in 0..NQ {
                            acts_e[j] = a[j] as u8;
                        }
                        let (r, d) = sim.step_inner(&acts_e);
                        sim.write_obs(obs_chunk);
                        if let Some(w) = walk.as_mut() {
                            for &(note, code, pts, _, _) in &sim.events_buf {
                                w.consume(note, code, pts);
                            }
                        }
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

#[cfg(test)]
mod env_tests {
    use super::*;

    /// SPEC-v10 §4 effort wiring: Body::tick's accumulated positive
    /// work equals an independent loop over the same normative pipeline
    /// BITWISE (identical expression tree), and coasting is exactly
    /// free. (The P3 Python gate checks the reward identity
    /// reward_effort = −coef × last_effort on top of this.)
    #[test]
    fn effort_is_positive_work() {
        let gear = gear_signed(5);
        let mut acts = [0u8; NQ];
        acts[15] = 1; // r wrist bend, full +
        acts[10] = 2; // r arm pitch, full −
        acts[0] = 3; // spine pitch, +1/4
        let mut tau = [0.0; NQ];
        for j in 0..NQ {
            tau[j] = gear[acts[j] as usize] * MODEL.tau_max[j];
        }
        // independent mirror of the tick loop
        let mut st = State::neutral(MODEL);
        let mut ws = Workspace::new(MODEL);
        let mut cs = ConstraintSet::new(MODEL, true, true);
        let mut effort = 0.0;
        for _ in 0..N_SUB {
            substep_constrained(MODEL, &mut st, &tau, H_SUB, &mut ws, &mut cs);
            for j in 0..NQ {
                let p = tau[j] * st.v[6 + j];
                if p > 0.0 {
                    effort += p * H_SUB;
                }
            }
        }
        let mut body = Body::new();
        body.tick(&acts, &gear);
        assert!(effort > 0.0, "driven joints must do positive work");
        assert_eq!(body.effort_j, effort, "effort expression trees diverge");
        // trajectory parity too (the mirror IS the pipeline)
        for j in 0..NQ {
            assert_eq!(body.st.q[j], st.q[j]);
        }
        let mut b2 = Body::new();
        b2.tick(&[0u8; NQ], &gear);
        assert_eq!(b2.effort_j, 0.0, "coast must be exactly free");
    }

    /// Live gates at the reference pose reproduce the v9 constants:
    /// reach 1.75, miss boundary −0.65 (SPEC-v10 §4).
    #[test]
    fn live_gates_reference_values() {
        let body = Body::new();
        assert!((body.reach_z() - 1.75).abs() < 1e-12);
        assert!((body.head[2] - (-0.65)).abs() < 1e-12);
    }
}
