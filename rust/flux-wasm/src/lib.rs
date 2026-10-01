//! The Flux WebXR game's judge (SPEC.md §Flux, §Flux WebXR game).
//!
//! The browser game has no simulated body — the player's tracked palms
//! ARE the palms — but everything that decides an outcome is the env's
//! own code: `geom.rs`, `flux.rs` and `combo.rs` are compiled in here
//! straight from the env crate (`rust/src`), and `FluxGame::step` runs
//! the Flux tick exactly as `Core::step_inner` does (tick end time, the
//! judge on last tick's → this tick's palms, tick count, cursor, the
//! chart end rule), then the bench's ScoreSaber combo walk. The f64
//! build: the env's reference precision.
//!
//! Judge rate: the env judges once per 30 Hz tick (DT). The game may judge
//! at `judge_hz` = 30·k (k judge ticks per env tick: finer palm paths,
//! hits judged closer to when they happen) — the SAME judge on a finer
//! sampling of the player's motion; judge tick j ends at (j + 1)·DT/k.
//! 30 Hz is the env's tick, bit for bit (the parity fixture pins it).
//! (Unrelated to the body sim's physics substeps: the game has no body.)
//!
//! The JS side (viz/app/src/lib/fluxcore.js) talks to a plain C ABI over
//! linear memory — no bindings generator, no dependencies.

pub type Real = f64;

#[allow(dead_code)]
#[path = "../../src/geom.rs"]
mod geom;
#[allow(dead_code)]
#[path = "../../src/flux.rs"]
mod flux;
#[allow(dead_code)]
#[path = "../../src/combo.rs"]
mod combo;

use combo::ComboWalk;
use flux::{Event, FluxTuple};
use geom::*;
// the env's 30 Hz base tick (the env crate itself runs at a runtime rate)
use geom::DT_30 as DT;
use std::cell::RefCell;

/// Floats per chart object in `fg_new`'s input: the set_flux_chart tuple
/// (hit_time, x, y, kind, color, rot, speed, length, sweet).
pub const OBJ_F: usize = 9;
/// Floats per `fg_step` input: palms last tick [L xyz, R xyz], palms
/// this tick [L xyz, R xyz], head xyz, forward-most shoulder z.
pub const STEP_F: usize = 16;
/// Floats per event in `fg_events_ptr`: (index, code, points, base, acc).
pub const EVENT_F: usize = 5;

/// Running tallies, laid out for JS (`fg_stats_ptr`, STAT_* in
/// fluxcore.js — keep the order in step).
#[repr(C)]
#[derive(Default, Clone, Copy)]
pub struct Stats {
    pub tick: f64,
    pub time: f64,
    pub good: f64,
    pub bad: f64,
    pub misses: f64,
    /// unmultiplied points (the env's `score`)
    pub raw_score: f64,
    pub score_base: f64,
    pub score_acc: f64,
    /// the ScoreSaber multiplied score (the bench's combo walk)
    pub score: f64,
    pub combo: f64,
    pub max_combo: f64,
    pub multiplier: f64,
    pub progress: f64,
    pub breaks: f64,
    /// Σ judge reward (good pts/115, bad and miss −0.5) — the env's
    /// Flux reward terms without the body's (energy, alive, falls)
    pub reward: f64,
    pub finished: f64,
    pub n_objects: f64,
    pub max_score: f64,
}
pub const STATS_F: usize = std::mem::size_of::<Stats>() / 8;

pub struct FluxGame {
    chart: Vec<geom::Note>,
    status: Vec<u8>,
    first_pending: usize,
    min_speed: Real,
    tail_s: Real,
    max_steps: u32,
    /// judge tick length: DT / (judge_hz / 30) (DT itself at 30 Hz)
    tick_dt: Real,
    tick: u32,
    time: Real,
    finished: bool,
    events: Vec<Event>,
    walk: ComboWalk,
    stats: Stats,
    ev_out: Vec<f64>,
    /// installed (sin_a, cos_a) per object, flat — the parity record
    trig: Vec<f64>,
}

/// Max objects score: 60 per circle, 115 per bar, before multipliers.
fn max_points(chart: &[geom::Note]) -> Real {
    chart
        .iter()
        .map(|n| if n.kind == KIND_CIRCLE {
            flux::CIRCLE_PTS + flux::CIRCLE_ACC_PTS
        } else {
            flux::BAR_PTS + flux::BAR_ACC_PTS
        })
        .sum()
}

impl FluxGame {
    /// Install a chart exactly as `Core::set_flux_chart` does: the same
    /// per-object checks and the same sorted-order requirement, the same
    /// slowest-object tail. `max_steps` (in ENV ticks) = 0 picks the env's
    /// fixed-chart budget (`RustFluxEnv(external_chart=...)`: (last hit +
    /// 0.5)·30 + 90). `judge_hz`: the judge rate, a multiple of 30 in
    /// 30..=480 (30 = the env's tick).
    pub fn new(chart: &[FluxTuple], max_steps: u32, judge_hz: u32)
               -> Result<Self, &'static str> {
        if judge_hz % 30 != 0 || !(30..=480).contains(&judge_hz) {
            return Err("judge_hz must be a multiple of 30 in 30..=480");
        }
        // judge ticks per env tick
        let per_tick = judge_hz / 30;
        let notes = chart
            .iter()
            .map(|&t| flux::flux_note(t))
            .collect::<Result<Vec<_>, _>>()?;
        if !notes.windows(2).all(|w| w[0].hit_time <= w[1].hit_time) {
            return Err("chart must be sorted by hit_time");
        }
        let min_speed = chart_min_speed(&notes);
        let max_steps = if max_steps > 0 {
            max_steps
        } else {
            let last = notes.last().map_or(0.0, |n| n.hit_time);
            ((last + 0.5) * 30.0) as u32 + 90
        } * per_tick;
        let n = notes.len();
        let stats = Stats {
            n_objects: n as f64,
            max_score: max_points(&notes),
            multiplier: 1.0,
            ..Stats::default()
        };
        Ok(FluxGame {
            status: vec![0; n],
            first_pending: 0,
            min_speed,
            tail_s: chart_tail_s(min_speed),
            max_steps,
            // exactly DT at 30 Hz (x / 1.0 == x)
            tick_dt: DT / per_tick as Real,
            tick: 0,
            time: 0.0,
            finished: false,
            events: Vec::new(),
            walk: ComboWalk::new(n),
            stats,
            ev_out: Vec::new(),
            trig: notes.iter().flat_map(|n| [n.sin_a, n.cos_a]).collect(),
            chart: notes,
        })
    }

    /// One judge tick (at `judge_hz`) in the env's Flux order (lib.rs
    /// `step_inner`):
    /// the tick's end time, `flux::judge` on (p0 → p1), tick += 1, the
    /// low-water cursor, then the chart end rule (`geom::chart_done`;
    /// Flux never has follow-through trackers, and the player has no
    /// fall / tilt / escape terminations). Returns done. A finished game
    /// re-stepped reports done with no events (the env's B4 latch).
    pub fn step(&mut self, p0: &[[Real; 3]; 2], p1: &[[Real; 3]; 2],
                head: [Real; 3], shoulder_z_max: Real) -> bool {
        self.events.clear();
        self.ev_out.clear();
        if self.finished {
            return true;
        }
        self.time = (self.tick + 1) as Real * self.tick_dt;
        let mut reward = 0.0;
        flux::judge(&self.chart, &mut self.status, self.first_pending,
                    self.min_speed, self.time, self.tick_dt, head,
                    flux::reach_z(shoulder_z_max), p0, p1, &mut reward,
                    &mut self.events);
        self.tick += 1;
        advance_first_pending(&self.status, &mut self.first_pending);
        let last_hit = self.chart.last().map_or(0.0, |n| n.hit_time);
        let pending = self.first_pending < self.chart.len();
        let done = chart_done(pending, true, self.chart.is_empty(), self.time,
                              last_hit, self.tail_s, self.tick, self.max_steps);

        let s = &mut self.stats;
        for &(i, code, pts, pre, acc) in &self.events {
            self.walk.consume(i, code, pts);
            match code {
                0 => {
                    s.good += 1.0;
                    s.raw_score += pts as f64;
                    s.score_base += pre as f64;
                    s.score_acc += acc as f64;
                }
                4 => s.misses += 1.0,
                _ => s.bad += 1.0,
            }
            self.ev_out.extend_from_slice(
                &[i as f64, code as f64, pts as f64, pre as f64, acc as f64]);
        }
        s.tick = self.tick as f64;
        s.time = self.time;
        s.reward += reward;
        s.score = self.walk.score;
        s.combo = self.walk.combo as f64;
        s.max_combo = self.walk.max_combo as f64;
        s.multiplier = self.walk.multiplier as f64;
        s.progress = self.walk.progress as f64;
        s.breaks = self.walk.breaks as f64;
        if done {
            self.finished = true;
            s.finished = 1.0;
        }
        done
    }

    pub fn events(&self) -> &[Event] {
        &self.events
    }

    pub fn stats(&self) -> &Stats {
        &self.stats
    }

    pub fn status(&self) -> &[u8] {
        &self.status
    }
}

// ---- C ABI for fluxcore.js ---------------------------------------------------

thread_local! {
    static LAST_ERR: RefCell<&'static str> = const { RefCell::new("") };
}

/// Judge constants the renderer draws with, so the JS side can check its
/// own copies against the real ones (fluxcore.js `CONSTS`).
static CONSTS: [f64; 8] = [
    DT,
    flux::PALM_OFF,
    flux::CIRCLE_R,
    2.0 * flux::BAR_HALF_W,
    flux::SLAB_LO,
    flux::SLAB_HI,
    PLANE_Z0,
    COS_TOL,
];

/// A zeroed f64 buffer of `n` values in linear memory (free with fg_dealloc).
#[no_mangle]
pub extern "C" fn fg_alloc(n: u32) -> *mut f64 {
    let b = vec![0.0f64; n as usize].into_boxed_slice();
    Box::into_raw(b) as *mut f64
}

/// # Safety
/// `p` / `n` must come from one `fg_alloc(n)` call, freed once.
#[no_mangle]
pub unsafe extern "C" fn fg_dealloc(p: *mut f64, n: u32) {
    if !p.is_null() {
        drop(Box::from_raw(std::ptr::slice_from_raw_parts_mut(p, n as usize)));
    }
}

/// New game over `n` objects of OBJ_F floats at `chart`, judged at
/// `judge_hz`; null on an invalid chart or rate (reason via fg_error_ptr /
/// fg_error_len).
///
/// # Safety
/// `chart` must point at `n * OBJ_F` readable f64s.
#[no_mangle]
pub unsafe extern "C" fn fg_new(chart: *const f64, n: u32, max_steps: u32,
                                judge_hz: u32) -> *mut FluxGame {
    let raw = if n == 0 {
        &[][..]
    } else {
        std::slice::from_raw_parts(chart, n as usize * OBJ_F)
    };
    let mut tuples = Vec::with_capacity(n as usize);
    for o in raw.chunks_exact(OBJ_F) {
        // kind / colour arrive as floats; anything not a small whole
        // number becomes an out-of-range code the checks reject
        let small = |v: f64| if v >= 0.0 && v <= 255.0 && v.fract() == 0.0 { v as u8 } else { 255 };
        tuples.push((o[0], o[1], o[2], small(o[3]), small(o[4]), o[5], o[6], o[7], o[8]));
    }
    match FluxGame::new(&tuples, max_steps, judge_hz) {
        Ok(g) => Box::into_raw(Box::new(g)),
        Err(e) => {
            LAST_ERR.with(|m| *m.borrow_mut() = e);
            std::ptr::null_mut()
        }
    }
}

#[no_mangle]
pub extern "C" fn fg_error_ptr() -> *const u8 {
    LAST_ERR.with(|m| m.borrow().as_ptr())
}

#[no_mangle]
pub extern "C" fn fg_error_len() -> u32 {
    LAST_ERR.with(|m| m.borrow().len() as u32)
}

/// # Safety
/// `g` must come from fg_new and not be used afterwards.
#[no_mangle]
pub unsafe extern "C" fn fg_free(g: *mut FluxGame) {
    if !g.is_null() {
        drop(Box::from_raw(g));
    }
}

/// One tick from STEP_F floats at `input`; returns 1 when done.
///
/// # Safety
/// `g` from fg_new; `input` points at STEP_F readable f64s.
#[no_mangle]
pub unsafe extern "C" fn fg_step(g: *mut FluxGame, input: *const f64) -> u32 {
    if g.is_null() || input.is_null() {
        return 1; // no game: nothing left to play
    }
    let g = &mut *g;
    let v = std::slice::from_raw_parts(input, STEP_F);
    let p0 = [[v[0], v[1], v[2]], [v[3], v[4], v[5]]];
    let p1 = [[v[6], v[7], v[8]], [v[9], v[10], v[11]]];
    g.step(&p0, &p1, [v[12], v[13], v[14]], v[15]) as u32
}

/// This tick's events, EVENT_F floats each (valid until the next step).
///
/// # Safety
/// `g` from fg_new.
#[no_mangle]
pub unsafe extern "C" fn fg_events_ptr(g: *const FluxGame) -> *const f64 {
    if g.is_null() {
        return std::ptr::null();
    }
    (*g).ev_out.as_ptr()
}

/// # Safety
/// `g` from fg_new.
#[no_mangle]
pub unsafe extern "C" fn fg_events_len(g: *const FluxGame) -> u32 {
    if g.is_null() {
        return 0;
    }
    (*g).events.len() as u32
}

/// # Safety
/// `g` from fg_new.
#[no_mangle]
pub unsafe extern "C" fn fg_stats_ptr(g: *const FluxGame) -> *const f64 {
    if g.is_null() {
        return std::ptr::null();
    }
    &(*g).stats as *const Stats as *const f64
}

#[no_mangle]
pub extern "C" fn fg_stats_len() -> u32 {
    STATS_F as u32
}

/// Per-object status (0 pending, 1 good, 2 bad, 3 missed), one byte each.
///
/// # Safety
/// `g` from fg_new.
#[no_mangle]
pub unsafe extern "C" fn fg_status_ptr(g: *const FluxGame) -> *const u8 {
    if g.is_null() {
        return std::ptr::null();
    }
    (*g).status.as_ptr()
}

/// The installed rotation record: (sin_a, cos_a) per object, 2 floats
/// each (the env's `RawBeatSim.chart_trig()` twin).
///
/// # Safety
/// `g` from fg_new (or null).
#[no_mangle]
pub unsafe extern "C" fn fg_trig_ptr(g: *const FluxGame) -> *const f64 {
    if g.is_null() {
        return std::ptr::null();
    }
    (*g).trig.as_ptr()
}

#[no_mangle]
pub extern "C" fn fg_consts_ptr() -> *const f64 {
    CONSTS.as_ptr()
}

#[no_mangle]
pub extern "C" fn fg_consts_len() -> u32 {
    CONSTS.len() as u32
}

#[cfg(test)]
mod tests {
    use super::*;

    const DTT: f64 = 1.0 / 30.0;
    const SPEED: f64 = 12.0;
    // a standing player: head at the reference plane, palms 20 cm ahead
    const HEAD: [f64; 3] = [0.0, 1.7, PLANE_Z0];

    fn circle(t: f64, x: f64, y: f64, color: u8) -> FluxTuple {
        (t, x, y, KIND_CIRCLE, color, 0.0, SPEED, 0.0, 0.0)
    }

    /// hit_time that puts an object's centre `lead` m beyond a palm at
    /// z = pz on tick 1 (z_off = 0 for the reference head)
    fn arrive(pz: f64, lead: f64) -> f64 {
        DTT + (pz + lead) / SPEED
    }

    fn run(g: &mut FluxGame, palms: [[f64; 3]; 2], ticks: usize) -> Vec<Event> {
        let mut ev = Vec::new();
        for _ in 0..ticks {
            let done = g.step(&palms, &palms, HEAD, PLANE_Z0);
            ev.extend_from_slice(g.events());
            if done {
                break;
            }
        }
        ev
    }

    #[test]
    fn matching_palm_scores_wrong_palm_is_bad_untouched_misses() {
        let l = [-0.3, 1.2, -0.45];
        let r = [0.3, 1.2, -0.45];
        let chart = [
            circle(arrive(l[2], 0.2), l[0] + 0.03, l[1], 0), // left on left: good
            circle(arrive(r[2], 0.2) + 0.001, r[0], r[1], 0), // left on right: bad
            circle(arrive(0.0, 0.0) + 0.002, 0.9, 1.9, 1),   // nobody: miss
        ];
        let mut g = FluxGame::new(&chart, 0, 30).unwrap();
        let ev = run(&mut g, [l, r], 60);
        let codes: Vec<(u32, u8)> = ev.iter().map(|e| (e.0, e.1)).collect();
        assert_eq!(codes, vec![(0, 0), (1, 3), (2, 4)]);
        // 3 cm off-centre: acc = round(10·(1 − 0.03/0.3)) = 9
        assert_eq!((ev[0].2, ev[0].3, ev[0].4), (59.0, 50.0, 9.0));
        let s = g.stats();
        assert_eq!((s.good, s.bad, s.misses, s.raw_score), (1.0, 1.0, 1.0, 59.0));
        // combo walk: 1 good at 1x, then two breaks
        assert_eq!((s.score, s.combo, s.max_combo, s.breaks), (59.0, 0.0, 1.0, 2.0));
        assert_eq!(s.finished, 1.0);
        // a finished game stays done and silent (the env's B4 latch)
        assert!(g.step(&[l, r], &[l, r], HEAD, PLANE_Z0));
        assert!(g.events().is_empty());
    }

    #[test]
    fn invalid_charts_are_refused_with_the_env_messages() {
        let bad_kind = [(1.0, 0.0, 1.4, 0u8, 0u8, 0.0, SPEED, 0.0, 0.0)];
        assert_eq!(FluxGame::new(&bad_kind, 0, 30).err(),
                   Some("flux chart kinds are 4 (circle) / 5 (bar)"));
        let unsorted = [circle(2.0, 0.0, 1.4, 0), circle(1.0, 0.0, 1.4, 1)];
        assert_eq!(FluxGame::new(&unsorted, 0, 30).err(),
                   Some("chart must be sorted by hit_time"));
        let bar = [(1.0, 0.0, 1.4, KIND_BAR, 0u8, 0.0, SPEED, 0.5, 1.5)];
        assert_eq!(FluxGame::new(&bar, 0, 30).err(),
                   Some("sweet-spot offset must be in [-1, 1]"));
    }

    #[test]
    fn a_faster_judge_rate_runs_the_same_rules_on_a_finer_clock() {
        let l = [-0.3, 1.2, -0.45];
        let r = [0.3, 1.2, -0.45];
        let chart = [
            circle(arrive(l[2], 0.2), l[0] + 0.03, l[1], 0),
            circle(arrive(r[2], 0.2) + 0.001, r[0], r[1], 0),
            circle(arrive(0.0, 0.0) + 0.002, 0.9, 1.9, 1),
        ];
        let mut g1 = FluxGame::new(&chart, 0, 30).unwrap();
        let mut g4 = FluxGame::new(&chart, 0, 120).unwrap();
        let e1 = run(&mut g1, [l, r], 60);
        let e4 = run(&mut g4, [l, r], 240);
        // a static pose: the same outcomes and points at either rate ...
        let k = |e: &Vec<Event>| e.iter().map(|e| (e.0, e.1, e.2)).collect::<Vec<_>>();
        assert_eq!(k(&e1), k(&e4));
        // ... on a clock 4x finer: tick k ends at (k + 1)·DT/4, and the
        // budget is counted in judge ticks
        assert_eq!(g4.tick_dt, DT / 4.0);
        assert_eq!(g1.tick_dt, DT);
        assert_eq!(g4.max_steps, 4 * g1.max_steps);
        // the miss lands no later than at 30 Hz, and within one env tick
        let (t1, t4) = (g1.stats().time, g4.stats().time);
        assert!(t4 <= t1 && t1 - t4 < DT, "{t1} vs {t4}");
        for bad in [0, 45, 100, 510] {
            assert_eq!(FluxGame::new(&chart, 0, bad).err(),
                       Some("judge_hz must be a multiple of 30 in 30..=480"));
        }
    }

    #[test]
    fn a_hit_is_judged_when_it_happens_not_at_the_env_tick_end() {
        // a palm sweeps across a circle between t = 0 and DT; at 120 Hz
        // the touch is judged on the 1/120 s tick it happens in
        // the circle reaches the palms' plane (z = 0) half-way through
        // the env tick, as the palm crosses its centre
        let c = circle(0.5 * DTT, 0.0, 1.2, 0);
        let mut g = FluxGame::new(&[c], 0, 120).unwrap();
        let path = |u: f64| [[-0.4 + 0.8 * u, 1.2, 0.0], [0.6, 1.2, 0.0]];
        let mut hit_tick = None;
        for k in 0..8 {
            let (u0, u1) = (k as f64 / 4.0, (k + 1) as f64 / 4.0);
            g.step(&path(u0.min(1.0)), &path(u1.min(1.0)), HEAD, PLANE_Z0);
            if !g.events().is_empty() {
                hit_tick = Some(k);
                break;
            }
        }
        // the palm crosses x = 0 half-way through the env tick: sub-tick 1
        assert_eq!(hit_tick, Some(1));
        assert_eq!(g.events()[0].1, 0);
    }

    #[test]
    fn empty_chart_ends_on_the_first_tick() {
        let mut g = FluxGame::new(&[], 0, 30).unwrap();
        assert!(g.step(&[[0.0; 3]; 2], &[[0.0; 3]; 2], HEAD, PLANE_Z0));
    }

    #[test]
    fn c_abi_round_trip() {
        unsafe {
            let l = [-0.3, 1.2, -0.45];
            let chart = fg_alloc(OBJ_F as u32);
            let c = circle(arrive(l[2], 0.02), l[0], l[1], 0);
            let v = [c.0, c.1, c.2, 4.0, 0.0, 0.0, c.6, 0.0, 0.0];
            std::ptr::copy_nonoverlapping(v.as_ptr(), chart, OBJ_F);
            let g = fg_new(chart, 1, 0, 30);
            fg_dealloc(chart, OBJ_F as u32);
            assert!(!g.is_null());
            let inp = fg_alloc(STEP_F as u32);
            let x = [l[0], l[1], l[2], 0.3, 1.2, -0.45];
            let mut row = [0.0; STEP_F];
            row[..6].copy_from_slice(&x);
            row[6..12].copy_from_slice(&x);
            row[12..15].copy_from_slice(&HEAD);
            row[15] = PLANE_Z0;
            std::ptr::copy_nonoverlapping(row.as_ptr(), inp, STEP_F);
            assert_eq!(fg_step(g, inp), 1); // the only object resolves: done
            assert_eq!(fg_events_len(g), 1);
            let e = std::slice::from_raw_parts(fg_events_ptr(g), EVENT_F);
            assert_eq!(e, &[0.0, 0.0, 60.0, 50.0, 10.0]);
            let st = std::slice::from_raw_parts(fg_stats_ptr(g), STATS_F);
            assert_eq!(st[8], 60.0); // multiplied score
            assert_eq!(*fg_status_ptr(g), 1);
            fg_dealloc(inp, STEP_F as u32);
            fg_free(g);
            // invalid chart: null + the reason
            let bad = fg_alloc(OBJ_F as u32);
            *bad.add(3) = 7.0;
            *bad.add(6) = 1.0;
            assert!(fg_new(bad, 1, 0, 30).is_null());
            let msg = std::slice::from_raw_parts(fg_error_ptr(), fg_error_len() as usize);
            assert_eq!(msg, b"flux chart kinds are 4 (circle) / 5 (bar)");
            fg_dealloc(bad, OBJ_F as u32);
        }
    }
}
