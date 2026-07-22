//! Beat Arena sim core in Rust — THE physics implementation (SPEC.md is
//! the contract; `python/beat_arena_env/constants.py` mirrors constants).
//!
//! Env v6 (2026-07-22, user-designed): kinematic upper body. Each hand
//! is a chain from a FIXED SHOULDER anchor: upper arm -> elbow (hinge)
//! -> forearm -> the v5.2 wrist (roll + bend, in the forearm frame) ->
//! 1 m blade. No force interactions — every joint integrates the same
//! way (accel, drag, omega cap, anatomical limits that zero omega);
//! the wrist is wherever forward kinematics puts it (fully 3D). Tiered
//! dynamics: ponderous shoulder/elbow, snappy wrists. Contact velocity
//! is the contact point's finite-difference displacement over the tick.
//! Cuts are swept-blade-through-block events (the note's actual 0.4 m
//! cube, rotated to its cut direction — env v5.3) scored with the real
//! Beat Saber 115-point system: 70 pre-swing (100 deg), 30
//! follow-through (60 deg), 15 accuracy-to-center.
//!
//! Chart generation stays in Python (single source of truth, once per
//! episode); the chart is injected via `set_chart`.

use numpy::{PyArray1, PyReadonlyArray2};
use pyo3::prelude::*;

const DT: f64 = 1.0 / 30.0;
const DRAG: f64 = 0.02;
const V_NORM: f64 = 6.0; // wrist-velocity obs normalizer (m/s)
const V_MIN_CUT: f64 = 1.0;
// accuracy normalizer only (circumscribed circle of the 0.4 m face) —
// contact itself is the swept blade vs the note's ROTATED box (env v5.3)
const CUT_RADIUS: f64 = 0.28;
const HALF_BLOCK: f64 = 0.2;
const COS_TOL: f64 = 0.5;
const CELL: f64 = 0.5;
const REWARD_BAD: f64 = -0.5;
const REWARD_MISS: f64 = -0.5;

// --- upper body (env v6, 2026-07-22, user-designed) -------------------------
// Kinematic chain per hand: fixed shoulder anchor -> upper arm -> elbow
// (hinge) -> forearm -> the v5.2 wrist (roll + bend) -> 1 m blade. No
// force interactions — every joint integrates the same way (accel, drag,
// omega cap, limits that zero omega). The wrist is wherever forward
// kinematics puts it (fully 3D; the z = 0 plane confinement is gone).
// Realistic fixed proportions: shoulder width 0.4 m, shoulders 0.45 m
// above grid center and 0.35 m behind the hit plane.
const SABER_LEN: f64 = 1.0; // m (real Beat Saber)
const SHOULDER: [[f64; 3]; 2] = [[-0.2, 0.45, -0.35], [0.2, 0.45, -0.35]];
const L_UPPER: f64 = 0.35; // m
const L_FORE: f64 = 0.30; // m
// tiered anatomical dynamics: ponderous shoulder/elbow, snappy wrists
const ALPHA_ARM: f64 = 80.0; // rad/s^2 (shoulder pitch/yaw/twist, elbow)
const OMEGA_ARM: f64 = 8.0; // rad/s
const ALPHA_WRIST: f64 = 200.0; // rad/s^2 (roll, bend — v5 values)
const OMEGA_WRIST: f64 = 20.0; // rad/s
// joint order per hand: pitch, yaw, twist, elbow, roll, bend.
// Neutral (all zeros) = arm straight forward (+z), blade its extension.
// pitch: + = arm up, +-90 deg. yaw: + = toward +x; outward 100 deg /
// cross-body 45 deg per hand (mirrored). twist (humeral rotation):
// +-90 deg, orients the elbow's bend plane. elbow flexion: 0..150 deg,
// curls toward local +y. roll/bend: the unchanged v5.2 wrist.
const PITCH_LIM: f64 = 1.5707963267948966; // 90 deg
const YAW_LO: [f64; 2] = [-1.7453292519943295, -0.7853981633974483]; // L, R
const YAW_HI: [f64; 2] = [0.7853981633974483, 1.7453292519943295];
const TWIST_LIM: f64 = 1.5707963267948966; // 90 deg
const ELBOW_HI: f64 = 2.6179938779914944; // 150 deg (lo = 0)
const ROLL_LO: [f64; 2] = [-1.5707963267948966, -3.3161255787892263]; // L, R
const ROLL_HI: [f64; 2] = [3.3161255787892263, 1.5707963267948966];
const BEND_LIM: f64 = 1.6580627893946132; // 95 deg from forward
const N_JOINTS: usize = 6;
const JOINT_ALPHA: [f64; N_JOINTS] =
    [ALPHA_ARM, ALPHA_ARM, ALPHA_ARM, ALPHA_ARM, ALPHA_WRIST, ALPHA_WRIST];
const JOINT_OMEGA: [f64; N_JOINTS] =
    [OMEGA_ARM, OMEGA_ARM, OMEGA_ARM, OMEGA_ARM, OMEGA_WRIST, OMEGA_WRIST];

fn joint_limits(hand: usize) -> [(f64, f64); N_JOINTS] {
    [
        (-PITCH_LIM, PITCH_LIM),
        (YAW_LO[hand], YAW_HI[hand]),
        (-TWIST_LIM, TWIST_LIM),
        (0.0, ELBOW_HI),
        (ROLL_LO[hand], ROLL_HI[hand]),
        (-BEND_LIM, BEND_LIM),
    ]
}
const N_BLADE_SAMPLES: usize = 8; // sweep test points along the blade
// broad-phase gate: max wrist z (0.3, arm straight forward) + blade + half
// block — beyond this no pose can touch a note
const REACH_Z: f64 = 1.5;
const Z_MISS: f64 = 0.6; // note consumed as missed past this

// --- scoring (real Beat Saber 115) ------------------------------------------
const PRE_FULL: f64 = 1.7453292519943295; // 100 deg
const POST_FULL: f64 = 1.0471975511965976; // 60 deg
const PTS_PRE: f64 = 70.0;
const PTS_POST: f64 = 30.0;
const PTS_ACC: f64 = 15.0;
const SCORE_MAX: f64 = 115.0;
const SWING_BREAK: f64 = 0.005; // rad/tick below which the swing chain breaks
const FOLLOW_TIMEOUT: u32 = 15; // ticks a follow-through may keep accumulating

const N_NOTE_SLOTS: usize = 12;
const NOTE_FEAT: usize = 9;
// obs v5: per hand 23 features — 6 joint angles (roll as cos/sin -> 7
// numbers), 6 angular velocities, derived wrist pos (3) + finite-diff
// wrist vel (3) + blade unit dir (3), and the live pre-swing meter
const SABER_FEAT: usize = 23;
const OFF_NOTES: usize = 2 * SABER_FEAT; // 46
const OFF_NOTE_MASK: usize = OFF_NOTES + N_NOTE_SLOTS * NOTE_FEAT; // 154
const OBS_DIM: usize = OFF_NOTE_MASK + N_NOTE_SLOTS; // 166
const X_NORM: f64 = 1.0;
const Y_NORM: f64 = 0.75;
const Z_SCALE: f64 = 8.0;
const Z_CLIP: (f64, f64) = (-1.0, 12.0);

#[derive(Clone, Copy)]
struct Note {
    hit_time: f64,
    col: u8,
    row: u8,
    color: u8,
    angle: f64,    // required swing direction in radians (0 = +x, CCW)
    speed: f64,    // approach speed toward the plane (m/s)
    any_dir: bool, // dot note: any swing direction scores
}

fn note_center(n: &Note) -> [f64; 2] {
    [(n.col as f64 - 1.5) * CELL, (n.row as f64 - 1.0) * CELL]
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
fn fk(hand: usize, q: &[f64; N_JOINTS]) -> ([f64; 3], [f64; 3], [f64; 3]) {
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
    (e, w, b)
}

/// Env v5.3: a note is its actual 0.4 m cube, rotated about the approach
/// (z) axis so the block's "up" tracks the required cut direction:
/// rot = angle - 270 deg (a down-swing note is axis-aligned, as before;
/// dot notes don't rotate). (cos_rot, sin_rot) derived exactly from the
/// angle's own sin/cos: cos(a - 3pi/2) = -sin a, sin(a - 3pi/2) = cos a.
fn note_rot(n: &Note) -> (f64, f64) {
    if n.any_dir {
        (1.0, 0.0)
    } else {
        let (sa, ca) = n.angle.sin_cos();
        (-sa, ca)
    }
}

/// p relative to the box center, expressed in the box's rotated frame.
fn to_box_frame(p: [f64; 3], center: [f64; 3], cos_r: f64, sin_r: f64) -> [f64; 3] {
    let q = [p[0] - center[0], p[1] - center[1], p[2] - center[2]];
    [
        cos_r * q[0] + sin_r * q[1],
        -sin_r * q[0] + cos_r * q[1],
        q[2],
    ]
}

/// Segment vs origin-centered axis-aligned cube of half-extent `h`
/// (slab clipping of the segment parameter to [0, 1]).
fn seg_box_hit(a: [f64; 3], b: [f64; 3], h: f64) -> bool {
    let mut tmin = 0.0f64;
    let mut tmax = 1.0f64;
    for k in 0..3 {
        let d = b[k] - a[k];
        if d.abs() < 1e-12 {
            if a[k].abs() > h {
                return false;
            }
        } else {
            let t1 = (-h - a[k]) / d;
            let t2 = (h - a[k]) / d;
            let (lo, hi) = if t1 < t2 { (t1, t2) } else { (t2, t1) };
            tmin = tmin.max(lo);
            tmax = tmax.min(hi);
            if tmin > tmax {
                return false;
            }
        }
    }
    true
}

fn seg_point_dist3(a: [f64; 3], b: [f64; 3], c: [f64; 3]) -> f64 {
    let ab = [b[0] - a[0], b[1] - a[1], b[2] - a[2]];
    let denom = ab[0] * ab[0] + ab[1] * ab[1] + ab[2] * ab[2];
    let ac = [c[0] - a[0], c[1] - a[1], c[2] - a[2]];
    let t = if denom < 1e-12 {
        0.0
    } else {
        ((ac[0] * ab[0] + ac[1] * ab[1] + ac[2] * ab[2]) / denom).clamp(0.0, 1.0)
    };
    let px = a[0] + t * ab[0] - c[0];
    let py = a[1] + t * ab[1] - c[1];
    let pz = a[2] + t * ab[2] - c[2];
    (px * px + py * py + pz * pz).sqrt()
}

#[derive(Clone, Copy)]
struct Arm {
    q: [f64; N_JOINTS],  // pitch, yaw, twist, elbow, roll, bend
    om: [f64; N_JOINTS], // angular velocities
    wrist: [f64; 3],     // FK cache (updated whenever q changes)
    blade: [f64; 3],     // FK cache, unit blade direction
    wrist_vel: [f64; 3], // finite difference across the last tick
    swing_acc: f64,      // arc of the current continuous swing (rad)
    axis: Option<[f64; 3]>, // last tick's rotation axis (unit)
}

impl Arm {
    fn start(hand: usize) -> Self {
        // all joints zero: arm straight forward, blade its extension
        let (_, w, b) = fk(hand, &[0.0; N_JOINTS]);
        Arm {
            q: [0.0; N_JOINTS],
            om: [0.0; N_JOINTS],
            wrist: w,
            blade: b,
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

struct Core {
    max_steps: u32,
    chart: Vec<Note>,
    // state
    tick: u32,
    time: f64,
    arms: [Arm; 2],
    trackers: [Vec<Tracker>; 2],
    status: Vec<u8>, // 0 pending, 1 good, 2 bad, 3 missed
    good_cuts: u32,
    bad_cuts: u32,
    misses: u32,
    score: f64, // summed points; max = 115 * n_notes
}

impl Core {
    fn new_core(max_steps: u32) -> Self {
        Core {
            max_steps,
            chart: Vec::new(),
            tick: 0,
            time: 0.0,
            arms: [Arm::start(0), Arm::start(1)],
            trackers: [Vec::new(), Vec::new()],
            status: Vec::new(),
            good_cuts: 0,
            bad_cuts: 0,
            misses: 0,
            score: 0.0,
        }
    }

    fn set_chart_notes(&mut self, chart: Vec<(f64, u8, u8, u8, f64, f64, bool)>) {
        self.chart = chart
            .into_iter()
            .map(|(t, col, row, color, angle, speed, any_dir)| Note {
                hit_time: t,
                col,
                row,
                color,
                angle,
                speed,
                any_dir,
            })
            .collect();
        self.reset_state();
    }

    fn reset_state(&mut self) {
        self.tick = 0;
        self.time = 0.0;
        self.arms = [Arm::start(0), Arm::start(1)];
        self.trackers = [Vec::new(), Vec::new()];
        self.status = vec![0; self.chart.len()];
        self.good_cuts = 0;
        self.bad_cuts = 0;
        self.misses = 0;
        self.score = 0.0;
    }

    fn build_obs(&self) -> [f32; OBS_DIM] {
        let mut obs = [0.0f32; OBS_DIM];
        for (s, arm) in self.arms.iter().enumerate() {
            let b = s * SABER_FEAT;
            // joint pose: bounded scalars normalized to their ranges,
            // except roll (280 deg arc) which keeps its v5 cos/sin coding
            obs[b] = (arm.q[0] / PITCH_LIM) as f32;
            obs[b + 1] = (arm.q[1] / YAW_HI[1]) as f32;
            obs[b + 2] = (arm.q[2] / TWIST_LIM) as f32;
            obs[b + 3] = ((2.0 * arm.q[3] - ELBOW_HI) / ELBOW_HI) as f32;
            obs[b + 4] = arm.q[4].cos() as f32;
            obs[b + 5] = arm.q[4].sin() as f32;
            obs[b + 6] = (arm.q[5] / BEND_LIM) as f32;
            for j in 0..N_JOINTS {
                obs[b + 7 + j] = (arm.om[j] / JOINT_OMEGA[j]) as f32;
            }
            // derived hand state: FK wrist, finite-diff velocity, blade dir
            obs[b + 13] = (arm.wrist[0] / X_NORM) as f32;
            obs[b + 14] = (arm.wrist[1] / Y_NORM) as f32;
            obs[b + 15] = arm.wrist[2] as f32;
            for k in 0..3 {
                obs[b + 16 + k] = (arm.wrist_vel[k] / V_NORM) as f32;
                obs[b + 19 + k] = arm.blade[k] as f32;
            }
            // live pre-swing meter: 1.0 = full pre-swing credit banked
            obs[b + 22] = (arm.swing_acc / PRE_FULL).min(2.0) as f32;
        }
        let mut slot = 0usize;
        for (i, n) in self.chart.iter().enumerate() {
            if self.status[i] != 0 {
                continue;
            }
            if slot >= N_NOTE_SLOTS {
                break;
            }
            let base = OFF_NOTES + slot * NOTE_FEAT;
            let c = note_center(n);
            let z = ((n.hit_time - self.time) * n.speed).clamp(Z_CLIP.0, Z_CLIP.1);
            obs[base] = (c[0] / X_NORM) as f32;
            obs[base + 1] = (c[1] / Y_NORM) as f32;
            obs[base + 2] = (z / Z_SCALE) as f32;
            obs[base + 3] = (-n.speed / Z_SCALE) as f32;
            obs[base + 4 + n.color as usize] = 1.0;
            // dot notes encode direction as the ZERO vector — the one point
            // that is no direction at all (unit-circle points = constraints)
            let d = if n.any_dir { [0.0, 0.0] } else { [n.angle.cos(), n.angle.sin()] };
            obs[base + 6] = d[0] as f32;
            obs[base + 7] = d[1] as f32;
            obs[base + 8] = 1.0; // valid
            obs[OFF_NOTE_MASK + slot] = 1.0;
            slot += 1;
        }
        obs
    }

    // event codes: 0 ok, 1 slow, 2 angle, 3 wrong_hand, 4 miss, 5 post
    // (follow-through resolution; the f32 is the points awarded).
    fn step_inner(&mut self, actions: [[u8; N_JOINTS]; 2]) -> (f64, bool, Vec<(u32, u8, f32)>) {
        let prev_pos = [self.arms[0].wrist, self.arms[1].wrist];
        let prev_dir = [self.arms[0].blade, self.arms[1].blade];

        let mut reward = 0.0;
        let mut events: Vec<(u32, u8, f32)> = Vec::new();
        // per arm this tick: (swing delta, did the chain continue)
        let mut deltas = [0.0f64; 2];
        let mut continued = [false; 2];

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
            let (_, w, d_new) = fk(s, &arm.q);
            arm.wrist = w;
            arm.blade = d_new;
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
                let cx = [
                    d_old[1] * d_new[2] - d_old[2] * d_new[1],
                    d_old[2] * d_new[0] - d_old[0] * d_new[2],
                    d_old[0] * d_new[1] - d_old[1] * d_new[0],
                ];
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
            let mut kept: Vec<Tracker> = Vec::new();
            for mut tr in self.trackers[s].drain(..) {
                let resolve = if !continued[s] {
                    true
                } else {
                    tr.acc += deltas[s];
                    tr.age += 1;
                    tr.acc >= POST_FULL || tr.age >= FOLLOW_TIMEOUT
                };
                if resolve {
                    let pts = PTS_POST * (tr.acc / POST_FULL).min(1.0);
                    reward += pts / SCORE_MAX;
                    self.score += pts;
                    events.push((tr.note, 5, pts as f32));
                } else {
                    kept.push(tr);
                }
            }
            self.trackers[s] = kept;
        }

        self.time = (self.tick + 1) as f64 * DT;

        for i in 0..self.chart.len() {
            if self.status[i] != 0 {
                continue;
            }
            let n = self.chart[i];
            let z = (n.hit_time - self.time) * n.speed;
            if z < -Z_MISS {
                self.status[i] = 3;
                self.misses += 1;
                reward += REWARD_MISS;
                events.push((i as u32, 4, 0.0));
                continue;
            }
            if z > REACH_Z {
                continue;
            }
            let center = [note_center(&n)[0], note_center(&n)[1], z];
            let (cos_r, sin_r) = note_rot(&n);
            for &s in &[n.color as usize, 1 - n.color as usize] {
                let arm = self.arms[s];
                let d_new = arm.blade;
                // swept-blade test: 8 sample points along the blade, each
                // tracing its own displacement segment (full 3D) against
                // the note's ROTATED cube; distance-to-center of the
                // hitting samples still grades accuracy / picks s*
                let mut d_min = f64::INFINITY;
                let mut s_star = 0.0;
                for j in 1..=N_BLADE_SAMPLES {
                    let frac = j as f64 / N_BLADE_SAMPLES as f64;
                    let r = frac * SABER_LEN;
                    let p0 = [
                        prev_pos[s][0] + r * prev_dir[s][0],
                        prev_pos[s][1] + r * prev_dir[s][1],
                        prev_pos[s][2] + r * prev_dir[s][2],
                    ];
                    let p1 = [
                        arm.wrist[0] + r * d_new[0],
                        arm.wrist[1] + r * d_new[1],
                        arm.wrist[2] + r * d_new[2],
                    ];
                    if !seg_box_hit(
                        to_box_frame(p0, center, cos_r, sin_r),
                        to_box_frame(p1, center, cos_r, sin_r),
                        HALF_BLOCK,
                    ) {
                        continue;
                    }
                    let d = seg_point_dist3(p0, p1, center);
                    if d < d_min {
                        d_min = d;
                        s_star = r;
                    }
                }
                if !d_min.is_finite() {
                    continue;
                }
                // contact velocity (env v6): the contact point's actual
                // displacement over the tick / DT — the same segments the
                // collision test sweeps, no chain Jacobian needed
                let vc = [
                    (arm.wrist[0] + s_star * d_new[0]
                        - prev_pos[s][0] - s_star * prev_dir[s][0]) / DT,
                    (arm.wrist[1] + s_star * d_new[1]
                        - prev_pos[s][1] - s_star * prev_dir[s][1]) / DT,
                    (arm.wrist[2] + s_star * d_new[2]
                        - prev_pos[s][2] - s_star * prev_dir[s][2]) / DT,
                ];
                let vc_mag = (vc[0].powi(2) + vc[1].powi(2) + vc[2].powi(2)).sqrt();
                let code: u8 = if s != n.color as usize {
                    3 // wrong_hand
                } else if vc_mag < V_MIN_CUT {
                    1 // slow
                } else if n.any_dir {
                    0 // dot note: any direction scores
                } else {
                    // direction check on the IN-PLANE contact velocity
                    let d2 = [n.angle.cos(), n.angle.sin()];
                    let v2_mag = (vc[0].powi(2) + vc[1].powi(2)).sqrt();
                    let cosv = (vc[0] * d2[0] + vc[1] * d2[1]) / v2_mag.max(1e-9);
                    if cosv >= COS_TOL { 0 } else { 2 } // ok / angle
                };
                if code == 0 {
                    self.status[i] = 1;
                    self.good_cuts += 1;
                    let pre = PTS_PRE * (self.arms[s].swing_acc / PRE_FULL).min(1.0);
                    // corner clips of the rotated box can sit slightly
                    // outside the circumscribed circle -> floor at 0
                    let acc = PTS_ACC * (1.0 - d_min / CUT_RADIUS).max(0.0);
                    let pts = pre + acc;
                    reward += pts / SCORE_MAX;
                    self.score += pts;
                    self.trackers[s].push(Tracker { note: i as u32, acc: 0.0, age: 0 });
                    events.push((i as u32, 0, pts as f32));
                } else {
                    self.status[i] = 2;
                    self.bad_cuts += 1;
                    reward += REWARD_BAD;
                    events.push((i as u32, code, 0.0));
                }
                break;
            }
        }

        self.tick += 1;
        let last_hit = self.chart.last().map_or(0.0, |n| n.hit_time);
        let pending = self.status.iter().any(|&s| s == 0);
        // natural completion waits for pending follow-throughs so the last
        // note's post-swing isn't truncated; the time/tick caps still end
        // the episode hard (trackers force-resolve below).
        let no_trackers = self.trackers[0].is_empty() && self.trackers[1].is_empty();
        let done = (!pending
            && no_trackers
            && (self.chart.is_empty() || self.time > last_hit))
            || self.time > last_hit + 0.5
            || self.tick >= self.max_steps;

        if done {
            // resolve any pending follow-throughs with their partial arcs
            for s in 0..2 {
                for tr in self.trackers[s].drain(..) {
                    let pts = PTS_POST * (tr.acc / POST_FULL).min(1.0);
                    reward += pts / SCORE_MAX;
                    self.score += pts;
                    events.push((tr.note, 5, pts as f32));
                }
            }
        }

        (reward, done, events)
    }
}

#[pyclass]
struct RawBeatSim {
    core: Core,
}

#[pymethods]
impl RawBeatSim {
    #[new]
    #[pyo3(signature = (max_steps=630))]
    fn new(max_steps: u32) -> Self {
        RawBeatSim {
            core: Core::new_core(max_steps),
        }
    }

    /// chart: list of (hit_time, col, row, color, angle_rad, speed, any_dir),
    /// sorted by hit_time. Resets state, returns the reset observation.
    fn set_chart<'py>(
        &mut self,
        py: Python<'py>,
        chart: Vec<(f64, u8, u8, u8, f64, f64, bool)>,
    ) -> Bound<'py, PyArray1<f32>> {
        self.core.set_chart_notes(chart);
        PyArray1::from_slice_bound(py, &self.core.build_obs())
    }

    /// actions: 12 ints in 0..2 — per hand (pitch, yaw, twist, elbow,
    /// roll, bend), left hand first; 0 = coast, 1 = +alpha, 2 = -alpha.
    /// Returns (obs, reward, done, events); events = [(note_idx, code, points)]
    /// with codes 0 ok, 1 slow, 2 angle, 3 wrong_hand, 4 miss, 5 post.
    fn step<'py>(
        &mut self,
        py: Python<'py>,
        actions: Vec<u8>,
    ) -> (Bound<'py, PyArray1<f32>>, f64, bool, Vec<(u32, u8, f32)>) {
        assert_eq!(actions.len(), 2 * N_JOINTS);
        assert!(actions.iter().all(|&a| a <= 2));
        let mut acts = [[0u8; N_JOINTS]; 2];
        for s in 0..2 {
            acts[s].copy_from_slice(&actions[s * N_JOINTS..(s + 1) * N_JOINTS]);
        }
        let (reward, done, events) = self.core.step_inner(acts);
        (
            PyArray1::from_slice_bound(py, &self.core.build_obs()),
            reward,
            done,
            events,
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

    /// Per-note status: 0 pending, 1 good, 2 bad, 3 missed.
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
        let (_, w, b) = fk(s, &arm.q);
        arm.wrist = w;
        arm.blade = b;
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
    fn n_notes(&self) -> usize {
        self.core.chart.len()
    }
    #[getter]
    fn tick(&self) -> u32 {
        self.core.tick
    }
    #[getter]
    fn time(&self) -> f64 {
        self.core.time
    }
    /// Summed points across all scored cuts (max 115 * n_notes).
    #[getter]
    fn score(&self) -> f64 {
        self.core.score
    }
}

/// N sims stepped in ONE FFI call per vector step. Chart generation stays
/// in Python: after any done env, call `set_chart(i, chart)` before the
/// next `step` (the wrapper in beat_arena_env/vec_env.py handles this).
#[pyclass]
struct VecBeatSim {
    sims: Vec<Core>,
}

#[pymethods]
impl VecBeatSim {
    #[new]
    #[pyo3(signature = (n, max_steps=630))]
    fn new(n: usize, max_steps: u32) -> Self {
        VecBeatSim {
            sims: (0..n).map(|_| Core::new_core(max_steps)).collect(),
        }
    }

    fn set_chart(&mut self, i: usize, chart: Vec<(f64, u8, u8, u8, f64, f64, bool)>) {
        self.sims[i].set_chart_notes(chart);
    }

    /// Current observation of every env, flat (N*140,) — reshape Python-side.
    fn obs<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f32>> {
        let n = self.sims.len();
        let mut flat = vec![0.0f32; n * OBS_DIM];
        for (i, sim) in self.sims.iter().enumerate() {
            flat[i * OBS_DIM..(i + 1) * OBS_DIM].copy_from_slice(&sim.build_obs());
        }
        PyArray1::from_vec_bound(py, flat)
    }

    /// actions: (N, 12) int64 — per hand (pitch, yaw, twist, elbow, roll,
    /// bend), left first. Returns (obs flat (N*166,) f32 — reshape
    /// Python-side, rewards (N,) f32, dones (N,) bool). Done envs must get
    /// a fresh set_chart before the next step; their returned obs row is
    /// the terminal obs.
    fn step<'py>(
        &mut self,
        py: Python<'py>,
        actions: PyReadonlyArray2<'py, i64>,
    ) -> (
        Bound<'py, PyArray1<f32>>,
        Bound<'py, PyArray1<f32>>,
        Bound<'py, PyArray1<bool>>,
    ) {
        let acts = actions.as_array();
        let n = self.sims.len();
        assert_eq!(acts.shape(), &[n, 2 * N_JOINTS]);
        let mut flat = vec![0.0f32; n * OBS_DIM];
        let mut rewards = vec![0.0f32; n];
        let mut dones = vec![false; n];
        for (i, sim) in self.sims.iter_mut().enumerate() {
            let mut a = [[0u8; N_JOINTS]; 2];
            for s in 0..2 {
                for j in 0..N_JOINTS {
                    a[s][j] = acts[[i, s * N_JOINTS + j]] as u8;
                }
            }
            let (r, d, _events) = sim.step_inner(a);
            flat[i * OBS_DIM..(i + 1) * OBS_DIM].copy_from_slice(&sim.build_obs());
            rewards[i] = r as f32;
            dones[i] = d;
        }
        (
            PyArray1::from_vec_bound(py, flat),
            PyArray1::from_vec_bound(py, rewards),
            PyArray1::from_vec_bound(py, dones),
        )
    }

    /// Single env's current obs (140,) — used for cheap post-reset reads.
    fn obs_one<'py>(&self, py: Python<'py>, i: usize) -> Bound<'py, PyArray1<f32>> {
        PyArray1::from_slice_bound(py, &self.sims[i].build_obs())
    }

    fn counters(&self, i: usize) -> (u32, u32, u32, usize, f64) {
        let s = &self.sims[i];
        (s.good_cuts, s.bad_cuts, s.misses, s.chart.len(), s.score)
    }
}

#[pymodule]
fn beat_arena_rs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<RawBeatSim>()?;
    m.add_class::<VecBeatSim>()?;
    Ok(())
}
