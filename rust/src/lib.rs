//! Beat Arena sim core in Rust — THE physics implementation (SPEC.md is
//! the contract; `python/beat_arena_env/constants.py` mirrors constants).
//!
//! Env v5.2 (2026-07-21): blade sabers with forearm-anchored joints.
//! Each hand = a wrist (point mass on the hit plane, the old point-saber
//! dynamics) + a rigid 1 m blade whose neutral is the forearm's
//! extension, POINTING AT THE NOTES; two joints drive it: roll (forearm
//! CW/CCW about the forward axis, per-hand 280-degree range) and bend
//! (wrist deflection away from forward, +-95 deg — the blade can never
//! point back at the player). Cuts are swept-blade-through-block events
//! scored with the real Beat Saber 115-point system: 70 pre-swing
//! (100 deg), 30 follow-through (60 deg), 15 accuracy-to-center.
//!
//! Chart generation stays in Python (single source of truth, once per
//! episode); the chart is injected via `set_chart`.

use numpy::{PyArray1, PyArrayMethods, PyReadonlyArray2};
use pyo3::prelude::*;

const DT: f64 = 1.0 / 30.0;
const ACCEL: f64 = 30.0;
const DRAG: f64 = 0.02;
const V_MAX: f64 = 6.0;
const V_MIN_CUT: f64 = 1.0;
const CUT_RADIUS: f64 = 0.28;
const COS_TOL: f64 = 0.5;
const CELL: f64 = 0.5;
const POS_LO: [f64; 2] = [-1.5, -1.25];
const POS_HI: [f64; 2] = [1.5, 1.25];
const REWARD_BAD: f64 = -0.5;
const REWARD_MISS: f64 = -0.5;
const SHAPE_GAMMA: f64 = 0.99;
const SHAPE_DIST_CLIP: f64 = 2.0;

// --- blade (env v5) ---------------------------------------------------------
const SABER_LEN: f64 = 1.0; // m (real Beat Saber)
const ALPHA_ROT: f64 = 200.0; // rad/s^2, both angular DOFs
const OMEGA_MAX: f64 = 20.0; // rad/s
// env v5.2 forearm-anchored joints (2026-07-21, user-specified): the
// neutral blade is the forearm's extension, POINTING AT THE NOTES (+z).
// roll = forearm CW/CCW about the forward axis; from the palm-down
// neutral the right wrist rolls 90 deg CCW / 190 deg CW ->
// [-190, 90] deg, mirrored left [-90, 190]. bend = wrist deflection
// away from forward, +-95 deg (90 = blade flat in the hit plane) — the
// blade can NEVER point back at the player (d_z >= cos 95 ~ -0.087).
const ROLL_LO: [f64; 2] = [-1.5707963267948966, -3.3161255787892263]; // L, R
const ROLL_HI: [f64; 2] = [3.3161255787892263, 1.5707963267948966];
const BEND_LIM: f64 = 1.6580627893946132; // 95 deg from forward
const N_BLADE_SAMPLES: usize = 8; // sweep test points along the blade
const REACH_Z: f64 = 1.28; // straight-forward tip (1 m) + CUT_RADIUS
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
// obs v4: per saber 10 features (wrist + blade pose + pre-swing meter)
const SABER_FEAT: usize = 10;
const OFF_NOTES: usize = 2 * SABER_FEAT; // 20
const OFF_NOTE_MASK: usize = OFF_NOTES + N_NOTE_SLOTS * NOTE_FEAT; // 128
const OBS_DIM: usize = OFF_NOTE_MASK + N_NOTE_SLOTS; // 140
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

fn action_dir(a: u8) -> [f64; 2] {
    let th = a as f64 * std::f64::consts::PI / 4.0;
    [th.cos(), th.sin()]
}

fn note_center(n: &Note) -> [f64; 2] {
    [(n.col as f64 - 1.5) * CELL, (n.row as f64 - 1.0) * CELL]
}

/// Blade direction unit vector from (roll, bend): neutral (+z, at the
/// notes), bent away from forward in the plane the roll orients.
fn blade_dir(roll: f64, bend: f64) -> [f64; 3] {
    let (sr, cr) = roll.sin_cos();
    let (sb, cb) = bend.sin_cos();
    [-sr * sb, cr * sb, cb]
}

/// d/dt of the blade direction, given the angular velocities.
fn blade_dir_dot(roll: f64, bend: f64, om_r: f64, om_b: f64) -> [f64; 3] {
    let (sr, cr) = roll.sin_cos();
    let (sb, cb) = bend.sin_cos();
    [
        -om_r * cr * sb - om_b * sr * cb,
        -om_r * sr * sb + om_b * cr * cb,
        -om_b * sb,
    ]
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
struct Saber {
    pos: [f64; 2],
    vel: [f64; 2],
    roll: f64,
    bend: f64,
    om_roll: f64,
    om_bend: f64,
    swing_acc: f64,        // arc of the current continuous swing (rad)
    axis: Option<[f64; 3]>, // last tick's rotation axis (unit)
}

impl Saber {
    fn start(x: f64) -> Self {
        Saber {
            pos: [x, 0.0],
            vel: [0.0; 2],
            roll: 0.0, // palm down
            bend: 0.0, // blade pointing at the notes
            om_roll: 0.0,
            om_bend: 0.0,
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
    shaping_coef: f64,
    max_steps: u32,
    chart: Vec<Note>,
    // state
    tick: u32,
    time: f64,
    sabers: [Saber; 2],
    trackers: [Vec<Tracker>; 2],
    status: Vec<u8>, // 0 pending, 1 good, 2 bad, 3 missed
    phi_pot: f64,    // shaping potential (NOT the tilt angle)
    good_cuts: u32,
    bad_cuts: u32,
    misses: u32,
    score: f64, // summed points; max = 115 * n_notes
}

impl Core {
    fn new_core(shaping_coef: f64, max_steps: u32) -> Self {
        Core {
            shaping_coef,
            max_steps,
            chart: Vec::new(),
            tick: 0,
            time: 0.0,
            sabers: [Saber::start(-0.5), Saber::start(0.5)],
            trackers: [Vec::new(), Vec::new()],
            status: Vec::new(),
            phi_pot: 0.0,
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

    fn potential(&self) -> f64 {
        if self.shaping_coef == 0.0 {
            return 0.0;
        }
        let mut phi = 0.0;
        let mut seen: u8 = 0;
        for (i, n) in self.chart.iter().enumerate() {
            if self.status[i] != 0 {
                continue;
            }
            if seen & (1 << n.color) != 0 {
                continue;
            }
            seen |= 1 << n.color;
            let c = note_center(n);
            let p = self.sabers[n.color as usize].pos;
            let d = ((p[0] - c[0]).powi(2) + (p[1] - c[1]).powi(2)).sqrt();
            phi -= self.shaping_coef * d.min(SHAPE_DIST_CLIP) / SHAPE_DIST_CLIP;
            if seen == 3 {
                break;
            }
        }
        phi
    }

    fn reset_state(&mut self) {
        self.tick = 0;
        self.time = 0.0;
        self.sabers = [Saber::start(-0.5), Saber::start(0.5)];
        self.trackers = [Vec::new(), Vec::new()];
        self.status = vec![0; self.chart.len()];
        self.good_cuts = 0;
        self.bad_cuts = 0;
        self.misses = 0;
        self.score = 0.0;
        self.phi_pot = self.potential();
    }

    fn build_obs(&self) -> [f32; OBS_DIM] {
        let mut obs = [0.0f32; OBS_DIM];
        for (s, sab) in self.sabers.iter().enumerate() {
            let b = s * SABER_FEAT;
            obs[b] = (sab.pos[0] / X_NORM) as f32;
            obs[b + 1] = (sab.pos[1] / Y_NORM) as f32;
            obs[b + 2] = (sab.vel[0] / V_MAX) as f32;
            obs[b + 3] = (sab.vel[1] / V_MAX) as f32;
            obs[b + 4] = sab.roll.cos() as f32;
            obs[b + 5] = sab.roll.sin() as f32;
            obs[b + 6] = (sab.bend / BEND_LIM) as f32;
            obs[b + 7] = (sab.om_roll / OMEGA_MAX) as f32;
            obs[b + 8] = (sab.om_bend / OMEGA_MAX) as f32;
            // live pre-swing meter: 1.0 = full pre-swing credit banked
            obs[b + 9] = (sab.swing_acc / PRE_FULL).min(2.0) as f32;
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
    fn step_inner(&mut self, actions: [[u8; 3]; 2]) -> (f64, bool, Vec<(u32, u8, f32)>) {
        let prev_pos = [self.sabers[0].pos, self.sabers[1].pos];
        let prev_dir = [
            blade_dir(self.sabers[0].roll, self.sabers[0].bend),
            blade_dir(self.sabers[1].roll, self.sabers[1].bend),
        ];

        let mut reward = 0.0;
        let mut events: Vec<(u32, u8, f32)> = Vec::new();
        // per saber this tick: (swing delta, did the chain continue)
        let mut deltas = [0.0f64; 2];
        let mut continued = [false; 2];

        for s in 0..2 {
            let sab = &mut self.sabers[s];
            // translation (unchanged point-mass wrist dynamics)
            let a = actions[s][0];
            if a > 0 {
                let d = action_dir(a - 1);
                sab.vel[0] += ACCEL * DT * d[0];
                sab.vel[1] += ACCEL * DT * d[1];
            }
            sab.vel[0] *= 1.0 - DRAG;
            sab.vel[1] *= 1.0 - DRAG;
            let speed = (sab.vel[0].powi(2) + sab.vel[1].powi(2)).sqrt();
            if speed > V_MAX {
                let k = V_MAX / speed;
                sab.vel[0] *= k;
                sab.vel[1] *= k;
            }
            sab.pos[0] += sab.vel[0] * DT;
            sab.pos[1] += sab.vel[1] * DT;
            for ax in 0..2 {
                if sab.pos[ax] < POS_LO[ax] || sab.pos[ax] > POS_HI[ax] {
                    sab.pos[ax] = sab.pos[ax].clamp(POS_LO[ax], POS_HI[ax]);
                    sab.vel[ax] = 0.0;
                }
            }
            // rotation: forearm roll (CW/CCW) and wrist bend
            let roll_a = actions[s][1];
            sab.om_roll += match roll_a {
                1 => ALPHA_ROT * DT,
                2 => -ALPHA_ROT * DT,
                _ => 0.0,
            };
            let bend_a = actions[s][2];
            sab.om_bend += match bend_a {
                1 => ALPHA_ROT * DT,
                2 => -ALPHA_ROT * DT,
                _ => 0.0,
            };
            sab.om_roll = (sab.om_roll * (1.0 - DRAG)).clamp(-OMEGA_MAX, OMEGA_MAX);
            sab.om_bend = (sab.om_bend * (1.0 - DRAG)).clamp(-OMEGA_MAX, OMEGA_MAX);
            sab.roll += sab.om_roll * DT;
            // anatomical limits: bounded joints, no wrap — hitting a limit
            // kills that joint's velocity
            if sab.roll < ROLL_LO[s] || sab.roll > ROLL_HI[s] {
                sab.roll = sab.roll.clamp(ROLL_LO[s], ROLL_HI[s]);
                sab.om_roll = 0.0;
            }
            sab.bend += sab.om_bend * DT;
            if sab.bend < -BEND_LIM || sab.bend > BEND_LIM {
                sab.bend = sab.bend.clamp(-BEND_LIM, BEND_LIM);
                sab.om_bend = 0.0;
            }

            // swing chain update (feeds pre-swing + follow-through scores)
            let d_new = blade_dir(sab.roll, sab.bend);
            let d_old = prev_dir[s];
            let dot = (d_old[0] * d_new[0] + d_old[1] * d_new[1] + d_old[2] * d_new[2])
                .clamp(-1.0, 1.0);
            let delta = dot.acos();
            deltas[s] = delta;
            if delta < SWING_BREAK {
                sab.swing_acc = 0.0;
                sab.axis = None;
            } else {
                let cx = [
                    d_old[1] * d_new[2] - d_old[2] * d_new[1],
                    d_old[2] * d_new[0] - d_old[0] * d_new[2],
                    d_old[0] * d_new[1] - d_old[1] * d_new[0],
                ];
                let cn = (cx[0].powi(2) + cx[1].powi(2) + cx[2].powi(2)).sqrt();
                let ax = [cx[0] / cn, cx[1] / cn, cx[2] / cn];
                match sab.axis {
                    Some(a0) if a0[0] * ax[0] + a0[1] * ax[1] + a0[2] * ax[2] < 0.0 => {
                        sab.swing_acc = delta; // reversal: a new swing starts
                    }
                    _ => {
                        sab.swing_acc += delta;
                        continued[s] = true;
                    }
                }
                sab.axis = Some(ax);
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
            for &s in &[n.color as usize, 1 - n.color as usize] {
                let sab = self.sabers[s];
                let d_new = blade_dir(sab.roll, sab.bend);
                // swept-blade test: 8 sample points along the blade, each
                // tracing its own displacement segment (full 3D)
                let mut d_min = f64::INFINITY;
                let mut s_star = 0.0;
                for j in 1..=N_BLADE_SAMPLES {
                    let frac = j as f64 / N_BLADE_SAMPLES as f64;
                    let r = frac * SABER_LEN;
                    let p0 = [
                        prev_pos[s][0] + r * prev_dir[s][0],
                        prev_pos[s][1] + r * prev_dir[s][1],
                        r * prev_dir[s][2],
                    ];
                    let p1 = [
                        sab.pos[0] + r * d_new[0],
                        sab.pos[1] + r * d_new[1],
                        r * d_new[2],
                    ];
                    let d = seg_point_dist3(p0, p1, center);
                    if d < d_min {
                        d_min = d;
                        s_star = r;
                    }
                }
                if d_min > CUT_RADIUS {
                    continue;
                }
                // contact velocity: wrist translation + rotation at s*
                let dd = blade_dir_dot(sab.roll, sab.bend, sab.om_roll, sab.om_bend);
                let vc = [
                    sab.vel[0] + s_star * dd[0],
                    sab.vel[1] + s_star * dd[1],
                    s_star * dd[2],
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
                    let pre = PTS_PRE * (self.sabers[s].swing_acc / PRE_FULL).min(1.0);
                    let acc = PTS_ACC * (1.0 - d_min / CUT_RADIUS);
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

        let new_phi = if done { 0.0 } else { self.potential() };
        reward += SHAPE_GAMMA * new_phi - self.phi_pot;
        self.phi_pot = new_phi;

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
    #[pyo3(signature = (shaping_coef=0.0, max_steps=630))]
    fn new(shaping_coef: f64, max_steps: u32) -> Self {
        RawBeatSim {
            core: Core::new_core(shaping_coef, max_steps),
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

    /// Per-hand action = (move 0..8, roll 0..2, bend 0..2); left then right.
    /// Returns (obs, reward, done, events); events = [(note_idx, code, points)]
    /// with codes 0 ok, 1 slow, 2 angle, 3 wrong_hand, 4 miss, 5 post.
    #[allow(clippy::too_many_arguments)]
    fn step<'py>(
        &mut self,
        py: Python<'py>,
        al_move: u8,
        al_rot: u8,
        al_tilt: u8,
        ar_move: u8,
        ar_rot: u8,
        ar_tilt: u8,
    ) -> (Bound<'py, PyArray1<f32>>, f64, bool, Vec<(u32, u8, f32)>) {
        assert!(al_move <= 8 && ar_move <= 8);
        assert!(al_rot <= 2 && al_tilt <= 2 && ar_rot <= 2 && ar_tilt <= 2);
        let (reward, done, events) = self
            .core
            .step_inner([[al_move, al_rot, al_tilt], [ar_move, ar_rot, ar_tilt]]);
        (
            PyArray1::from_slice_bound(py, &self.core.build_obs()),
            reward,
            done,
            events,
        )
    }

    /// Saber state [[x, y, vx, vy, roll, bend, om_roll, om_bend]; 2].
    fn sabers(&self) -> [[f64; 8]; 2] {
        [0, 1].map(|s| {
            let sab = self.core.sabers[s];
            [
                sab.pos[0], sab.pos[1], sab.vel[0], sab.vel[1],
                sab.roll, sab.bend, sab.om_roll, sab.om_bend,
            ]
        })
    }

    /// Per-note status: 0 pending, 1 good, 2 bad, 3 missed.
    fn status(&self) -> Vec<u8> {
        self.core.status.clone()
    }

    /// TEST HOOK: overwrite one saber's full state before a step.
    #[allow(clippy::too_many_arguments)]
    fn set_saber(
        &mut self,
        s: usize,
        x: f64,
        y: f64,
        vx: f64,
        vy: f64,
        roll: f64,
        bend: f64,
        om_roll: f64,
        om_bend: f64,
    ) {
        let sab = &mut self.core.sabers[s];
        sab.pos = [x, y];
        sab.vel = [vx, vy];
        sab.roll = roll;
        sab.bend = bend;
        sab.om_roll = om_roll;
        sab.om_bend = om_bend;
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
    #[pyo3(signature = (n, shaping_coef=0.0, max_steps=630))]
    fn new(n: usize, shaping_coef: f64, max_steps: u32) -> Self {
        VecBeatSim {
            sims: (0..n)
                .map(|_| Core::new_core(shaping_coef, max_steps))
                .collect(),
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

    /// actions: (N, 6) int64 — per hand (move, rot, tilt), left first.
    /// Returns (obs flat (N*140,) f32 — reshape Python-side, rewards (N,)
    /// f32, dones (N,) bool). Done envs must get a fresh set_chart before
    /// the next step; their returned obs row is the terminal obs.
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
        assert_eq!(acts.shape(), &[n, 6]);
        let mut flat = vec![0.0f32; n * OBS_DIM];
        let mut rewards = vec![0.0f32; n];
        let mut dones = vec![false; n];
        for (i, sim) in self.sims.iter_mut().enumerate() {
            let (r, d, _events) = sim.step_inner([
                [acts[[i, 0]] as u8, acts[[i, 1]] as u8, acts[[i, 2]] as u8],
                [acts[[i, 3]] as u8, acts[[i, 4]] as u8, acts[[i, 5]] as u8],
            ]);
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
