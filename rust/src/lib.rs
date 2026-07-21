//! Beat Arena sim core in Rust — a semantics-exact port of
//! `python/beat_arena_env/sim.py` (step + obs; SPEC.md is the contract).
//!
//! Chart generation stays in Python (single source of truth, once per
//! episode); the chart is injected via `set_chart`. Replay recording is
//! Python-only — this port exists for training/eval throughput.
//! Parity with the Python sim is pinned by `tests/test_rust_parity.py`.

use numpy::{PyArray1, PyArrayMethods, PyReadonlyArray2};
use pyo3::prelude::*;

const DT: f64 = 1.0 / 30.0;
const ACCEL: f64 = 30.0;
const DRAG: f64 = 0.02;
const V_MAX: f64 = 6.0;
const V_MIN_CUT: f64 = 1.0;
const CUT_RADIUS: f64 = 0.28;
const COS_TOL: f64 = 0.5;
const Z_WINDOW: f64 = 0.6;
const CELL: f64 = 0.5;
const POS_LO: [f64; 2] = [-1.5, -1.25];
const POS_HI: [f64; 2] = [1.5, 1.25];
const REWARD_GOOD: f64 = 1.0;
// Equal to a miss since 2026-07-20 (user audit): in real Beat Saber a
// bad cut and a miss are equally terminal (no score, combo break). The
// old -0.25 was a v0 exploration crutch (engagement gradient) that
// under-punished wrong_hand once the policy learned to engage.
const REWARD_BAD: f64 = -0.5;
const REWARD_MISS: f64 = -0.5;
const SHAPE_GAMMA: f64 = 0.99;
const SHAPE_DIST_CLIP: f64 = 2.0;

const N_NOTE_SLOTS: usize = 12;
// obs v3 (2026-07-20): direction as its unit vector (cos, sin) instead of
// an 8-dim one-hot — ready for continuous precision angles, and it hands
// the network the exact object the cut rule uses (cos angle = v . d / |v|).
const NOTE_FEAT: usize = 9;
const OFF_NOTES: usize = 8;
const OFF_NOTE_MASK: usize = OFF_NOTES + N_NOTE_SLOTS * NOTE_FEAT; // 116
const OBS_DIM: usize = OFF_NOTE_MASK + N_NOTE_SLOTS; // 128
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
    dir: u8,
}

fn dir_vec(d: u8) -> [f64; 2] {
    let a = d as f64 * std::f64::consts::PI / 4.0;
    [a.cos(), a.sin()]
}

fn note_center(n: &Note) -> [f64; 2] {
    [
        (n.col as f64 - 1.5) * CELL,
        (n.row as f64 - 1.0) * CELL,
    ]
}

fn seg_point_dist(a: [f64; 2], b: [f64; 2], c: [f64; 2]) -> f64 {
    let ab = [b[0] - a[0], b[1] - a[1]];
    let denom = ab[0] * ab[0] + ab[1] * ab[1];
    if denom < 1e-12 {
        return ((c[0] - a[0]).powi(2) + (c[1] - a[1]).powi(2)).sqrt();
    }
    let t = (((c[0] - a[0]) * ab[0] + (c[1] - a[1]) * ab[1]) / denom).clamp(0.0, 1.0);
    let px = a[0] + t * ab[0] - c[0];
    let py = a[1] + t * ab[1] - c[1];
    (px * px + py * py).sqrt()
}

struct Core {
    note_speed: f64,
    shaping_coef: f64,
    max_steps: u32,
    chart: Vec<Note>,
    // state
    tick: u32,
    time: f64,
    pos: [[f64; 2]; 2],
    vel: [[f64; 2]; 2],
    status: Vec<u8>, // 0 pending, 1 good, 2 bad, 3 missed
    phi: f64,
    good_cuts: u32,
    bad_cuts: u32,
    misses: u32,
}

impl Core {
    fn new_core(note_speed: f64, shaping_coef: f64, max_steps: u32) -> Self {
        Core {
            note_speed,
            shaping_coef,
            max_steps,
            chart: Vec::new(),
            tick: 0,
            time: 0.0,
            pos: [[-0.5, 0.0], [0.5, 0.0]],
            vel: [[0.0; 2]; 2],
            status: Vec::new(),
            phi: 0.0,
            good_cuts: 0,
            bad_cuts: 0,
            misses: 0,
        }
    }

    fn set_chart_notes(&mut self, chart: Vec<(f64, u8, u8, u8, u8)>) {
        self.chart = chart
            .into_iter()
            .map(|(t, col, row, color, dir)| Note {
                hit_time: t,
                col,
                row,
                color,
                dir,
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
            let p = self.pos[n.color as usize];
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
        self.pos = [[-0.5, 0.0], [0.5, 0.0]];
        self.vel = [[0.0; 2]; 2];
        self.status = vec![0; self.chart.len()];
        self.good_cuts = 0;
        self.bad_cuts = 0;
        self.misses = 0;
        self.phi = self.potential();
    }

    fn build_obs(&self) -> [f32; OBS_DIM] {
        let mut obs = [0.0f32; OBS_DIM];
        for s in 0..2 {
            obs[s * 4] = (self.pos[s][0] / X_NORM) as f32;
            obs[s * 4 + 1] = (self.pos[s][1] / Y_NORM) as f32;
            obs[s * 4 + 2] = (self.vel[s][0] / V_MAX) as f32;
            obs[s * 4 + 3] = (self.vel[s][1] / V_MAX) as f32;
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
            let z = ((n.hit_time - self.time) * self.note_speed).clamp(Z_CLIP.0, Z_CLIP.1);
            obs[base] = (c[0] / X_NORM) as f32;
            obs[base + 1] = (c[1] / Y_NORM) as f32;
            obs[base + 2] = (z / Z_SCALE) as f32;
            obs[base + 3] = (-self.note_speed / Z_SCALE) as f32;
            obs[base + 4 + n.color as usize] = 1.0;
            let d = dir_vec(n.dir);
            obs[base + 6] = d[0] as f32;   // cos(theta): swing dir x
            obs[base + 7] = d[1] as f32;   // sin(theta): swing dir y
            obs[base + 8] = 1.0;           // valid
            obs[OFF_NOTE_MASK + slot] = 1.0;
            slot += 1;
        }
        obs
    }

    // event codes: 0 ok, 1 slow, 2 angle, 3 wrong_hand, 4 miss
    fn step_inner(&mut self, actions: [u8; 2]) -> (f64, bool, Vec<(u32, u8)>) {
        let prev_pos = self.pos;
        for s in 0..2 {
            let a = actions[s];
            if a > 0 {
                let d = dir_vec(a - 1);
                self.vel[s][0] += ACCEL * DT * d[0];
                self.vel[s][1] += ACCEL * DT * d[1];
            }
            self.vel[s][0] *= 1.0 - DRAG;
            self.vel[s][1] *= 1.0 - DRAG;
            let speed = (self.vel[s][0].powi(2) + self.vel[s][1].powi(2)).sqrt();
            if speed > V_MAX {
                let k = V_MAX / speed;
                self.vel[s][0] *= k;
                self.vel[s][1] *= k;
            }
            self.pos[s][0] += self.vel[s][0] * DT;
            self.pos[s][1] += self.vel[s][1] * DT;
            for ax in 0..2 {
                if self.pos[s][ax] < POS_LO[ax] || self.pos[s][ax] > POS_HI[ax] {
                    self.pos[s][ax] = self.pos[s][ax].clamp(POS_LO[ax], POS_HI[ax]);
                    self.vel[s][ax] = 0.0;
                }
            }
        }

        self.time = (self.tick + 1) as f64 * DT;
        let mut reward = 0.0;
        let mut events: Vec<(u32, u8)> = Vec::new();

        for i in 0..self.chart.len() {
            if self.status[i] != 0 {
                continue;
            }
            let n = self.chart[i];
            let z = (n.hit_time - self.time) * self.note_speed;
            if z.abs() > Z_WINDOW {
                if z < -Z_WINDOW {
                    self.status[i] = 3;
                    self.misses += 1;
                    reward += REWARD_MISS;
                    events.push((i as u32, 4));
                }
                continue;
            }
            let center = note_center(&n);
            for &s in &[n.color as usize, 1 - n.color as usize] {
                let crossed =
                    seg_point_dist(prev_pos[s], self.pos[s], center) <= CUT_RADIUS;
                if !crossed {
                    continue;
                }
                let code: u8 = if s != n.color as usize {
                    3 // wrong_hand
                } else {
                    let speed =
                        (self.vel[s][0].powi(2) + self.vel[s][1].powi(2)).sqrt();
                    if speed < V_MIN_CUT {
                        1 // slow
                    } else {
                        let d = dir_vec(n.dir);
                        let cosv =
                            (self.vel[s][0] * d[0] + self.vel[s][1] * d[1]) / speed.max(1e-9);
                        if cosv >= COS_TOL { 0 } else { 2 } // ok / angle
                    }
                };
                let good = code == 0;
                self.status[i] = if good { 1 } else { 2 };
                if good {
                    self.good_cuts += 1;
                    reward += REWARD_GOOD;
                } else {
                    self.bad_cuts += 1;
                    reward += REWARD_BAD;
                }
                events.push((i as u32, code));
                break;
            }
        }

        self.tick += 1;
        let last_hit = self.chart.last().map_or(0.0, |n| n.hit_time);
        let pending = self.status.iter().any(|&s| s == 0);
        let done = (!pending && (self.chart.is_empty() || self.time > last_hit))
            || self.time > last_hit + 0.5
            || self.tick >= self.max_steps;

        let new_phi = if done { 0.0 } else { self.potential() };
        reward += SHAPE_GAMMA * new_phi - self.phi;
        self.phi = new_phi;

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
    #[pyo3(signature = (note_speed=4.0, shaping_coef=0.0, max_steps=630))]
    fn new(note_speed: f64, shaping_coef: f64, max_steps: u32) -> Self {
        RawBeatSim {
            core: Core::new_core(note_speed, shaping_coef, max_steps),
        }
    }

    /// chart: list of (hit_time, col, row, color, dir), sorted by hit_time.
    /// Resets all episode state and returns the reset observation.
    fn set_chart<'py>(
        &mut self,
        py: Python<'py>,
        chart: Vec<(f64, u8, u8, u8, u8)>,
    ) -> Bound<'py, PyArray1<f32>> {
        self.core.set_chart_notes(chart);
        PyArray1::from_slice_bound(py, &self.core.build_obs())
    }

    /// Returns (obs, reward, done, events); events = [(note_idx, code)]
    /// with codes 0 ok, 1 slow, 2 angle, 3 wrong_hand, 4 miss.
    fn step<'py>(
        &mut self,
        py: Python<'py>,
        action_left: u8,
        action_right: u8,
    ) -> (Bound<'py, PyArray1<f32>>, f64, bool, Vec<(u32, u8)>) {
        assert!(action_left <= 8 && action_right <= 8);
        let (reward, done, events) = self.core.step_inner([action_left, action_right]);
        (
            PyArray1::from_slice_bound(py, &self.core.build_obs()),
            reward,
            done,
            events,
        )
    }

    /// Saber state [[x, y, vx, vy]; 2] (world units).
    fn sabers(&self) -> [[f64; 4]; 2] {
        [0, 1].map(|s| [self.core.pos[s][0], self.core.pos[s][1],
                        self.core.vel[s][0], self.core.vel[s][1]])
    }

    /// Per-note status: 0 pending, 1 good, 2 bad, 3 missed.
    fn status(&self) -> Vec<u8> {
        self.core.status.clone()
    }

    /// TEST HOOK: overwrite one saber's position/velocity before a step.
    fn set_saber(&mut self, s: usize, x: f64, y: f64, vx: f64, vy: f64) {
        self.core.pos[s] = [x, y];
        self.core.vel[s] = [vx, vy];
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
    #[pyo3(signature = (n, note_speed=4.0, shaping_coef=0.0, max_steps=630))]
    fn new(n: usize, note_speed: f64, shaping_coef: f64, max_steps: u32) -> Self {
        VecBeatSim {
            sims: (0..n)
                .map(|_| Core::new_core(note_speed, shaping_coef, max_steps))
                .collect(),
        }
    }

    fn set_chart(&mut self, i: usize, chart: Vec<(f64, u8, u8, u8, u8)>) {
        self.sims[i].set_chart_notes(chart);
    }

    /// Current observation of every env, flat (N*200,) — reshape Python-side.
    fn obs<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f32>> {
        let n = self.sims.len();
        let mut flat = vec![0.0f32; n * OBS_DIM];
        for (i, sim) in self.sims.iter().enumerate() {
            flat[i * OBS_DIM..(i + 1) * OBS_DIM].copy_from_slice(&sim.build_obs());
        }
        PyArray1::from_vec_bound(py, flat)
    }

    /// actions: (N, 2) int64. Returns (obs flat (N*200,) f32 — reshape
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
        assert_eq!(acts.shape(), &[n, 2]);
        let mut flat = vec![0.0f32; n * OBS_DIM];
        let mut rewards = vec![0.0f32; n];
        let mut dones = vec![false; n];
        for (i, sim) in self.sims.iter_mut().enumerate() {
            let (r, d, _events) =
                sim.step_inner([acts[[i, 0]] as u8, acts[[i, 1]] as u8]);
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

    /// Single env's current obs (200,) — used for cheap post-reset reads.
    fn obs_one<'py>(&self, py: Python<'py>, i: usize) -> Bound<'py, PyArray1<f32>> {
        PyArray1::from_slice_bound(py, &self.sims[i].build_obs())
    }

    fn counters(&self, i: usize) -> (u32, u32, u32, usize) {
        let s = &self.sims[i];
        (s.good_cuts, s.bad_cuts, s.misses, s.chart.len())
    }
}

#[pymodule]
fn beat_arena_rs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<RawBeatSim>()?;
    m.add_class::<VecBeatSim>()?;
    Ok(())
}
