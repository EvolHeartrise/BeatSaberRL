//! Beat Arena sim core in Rust — THE physics implementation (SPEC.md +
//! SPEC-v10.md are the contract; `python/beat_arena_env/constants.py`
//! mirrors constants).
//!
//! Env v10 (P3, 2026-07-31): the v9 per-joint kinematic integrator and
//! 2-arm FK are RETIRED — the body is the SPEC-v10 full humanoid
//! (floating pelvis + 29 actuated revolutes, self-collision; env v11
//! unilateral ground contact on the soles — the v10 foot welds were
//! removed 2026-09-27)
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

/// Scalar type of the whole sim (physics + env layer). Default f64 is the
/// exact-parity regime every Rust test is pinned to; `--features f32`
/// (the `beat_arena_rs32` build, see ../rust32) mirrors the CUDA training
/// kernel's f32 instantiation — the bench's default engine, because
/// training runs f32.
#[cfg(not(feature = "f32"))]
pub type Real = f64;
#[cfg(feature = "f32")]
pub type Real = f32;

use numpy::{PyArray1, PyReadonlyArray2, PyReadwriteArray2, PyUntypedArrayMethods};
use pyo3::prelude::*;
use rayon::prelude::*;

// env v10 dynamics core — SPEC-v10.md §1–3 (P1 math, P2 constraints).
pub mod dynamics;
// Flux game mode (circles + bars, no sabers) — SPEC.md §Flux.
mod flux;
// judge-shared geometry / constants / combo walk (also compiled into the
// Flux WebXR game's wasm crate, rust/flux-wasm)
mod combo;
mod geom;
mod flux_sim;

use combo::ComboWalk;
use geom::*;

use dynamics::collide::{CORNERS_PER_FOOT, NUM_PAIRS, PAIRS};
use dynamics::fk::{cross3, dot3, quat_to_rot, sub3};
use dynamics::model::{
    Model, TENDON_R0, F_HEAD, F_L_ELBOW, F_L_FOOT, F_L_HAND, F_L_SHANK, F_L_SHOULDER,
    F_L_WRIST, F_R_ELBOW, F_R_FOOT, F_R_HAND, F_R_SHANK, F_R_SHOULDER,
    F_R_WRIST, HUMANOID, NQ, NV,
};
use dynamics::solver::{substep_constrained, ConstraintSet};
use dynamics::step::{H_SUB, N_SUB};
use dynamics::{DbgStats, State, Workspace};
use std::collections::HashMap;

/// R26: turn an aggregated `DbgStats` into the `pop_dbg_stats` return
/// dict. `enabled` is 1 only in a `--features stats` build — in the
/// default build every other value is structurally 0 (nothing counts),
/// so a collection script should assert `d["enabled"] == 1` before
/// trusting a rate.
fn dbg_stats_dict(s: &DbgStats) -> HashMap<String, u64> {
    let mut out: HashMap<String, u64> = s
        .as_pairs()
        .iter()
        .map(|(k, v)| ((*k).to_string(), *v))
        .collect();
    out.insert("enabled".to_string(), cfg!(feature = "stats") as u64);
    out
}

const V_NORM: Real = 6.0; // wrist-velocity obs normalizer (m/s)
const V_MIN_CUT: Real = 1.0;
// lanes at x = -0.9 .. 0.9 (0.6 m pitch), rows at y = 0.85/1.40/1.90 —
// positions arrive PRE-RESOLVED in the chart tuple (constants.py owns them)
const Y_CENTER: Real = 1.40; // obs y normalization
// env v12 (unilateral-contact spec §7, 2026-08-09): per-tick survival
// bonus, CONTACT MODE ONLY. The v11.1 fall pricing left survival-time
// flat BEFORE the first note (every pre-first-note fall prices the full
// chart), so a from-scratch policy gets zero gradient on "die at 0.5 s
// vs 1.9 s" until exploration tunnels to 2.0 s on its own. 0.002/tick
// (0.06/s) keeps a full 3-min chart's worth (~+11) well under the note
// stream's scale (~|70|+), so it cannot out-bid hitting notes; it is
// overridden on fall/escape ticks (the terminal pricing wins).
const ALIVE_BONUS: Real = 0.002;
// env v7.8 (2026-07-24, user-priced): a combo break on ScoreSaber costs
// ~4 notes of multiplier; -0.5 made bombs the CHEAPEST error in the gym
// (miss = -0.5 + ~1.0 forgone points) while the referee prices all
// breaks equally — both the 598M and gen2 policies equilibrated at
// ~12% bombs eaten under -0.5, flat across budget and distribution
const REWARD_BOMB: Real = -1.5;
// Movement is priced by energy_coef (env v43, 2026-09-14: the applied
// Σ|τ_j| = Σ|gear fraction_j| × τ_max[j] over the 29 joints per tick —
// a human-like force/energy price that replaced the v23 fraction
// "activation" price, so a spine gear costs 600/1.06× a wrist gear) and limit_coef (limit
// rest). The v10 positive-work "effort" price AND its telemetry were
// retired 2026-09-13; the energy telemetry is now the activation itself
// — per tick [total, spine, arms, legs] via last_activation() (replay
// "activation" field, viewer meter) and the per-update accumulator
// pop_activation() (trainer `[activation]` line, dashboard).

// --- hitboxes (env v7 — real BS, axis-aligned) -------------------------------
const GOOD_HALF_X: Real = 0.4;
const GOOD_HALF_Y: Real = 0.25;
const GOOD_HALF_Y_DOT: Real = 0.4;
const GOOD_HALF_Y_LINK: Real = 0.1;
const BAD_HALF: Real = 0.175;
const BAD_HALF_Y_LINK: Real = 0.05;
const BOMB_RADIUS: Real = 0.18;

const SABER_LEN: Real = 1.0; // m (real Beat Saber)
// env v42 "hilt grip" (2026-09-14, user-directed): the hand holds the
// 0.16 m handle; the cutting blade spans [HANDLE_LEN, HANDLE_LEN +
// SABER_LEN] along the blade dir from the wrist. The handle does not cut.
const HANDLE_LEN: Real = 0.16;

// env v8 gears: action id -> signed torque fraction of tau_max (v10:
// the gear scales TORQUE, SPEC-v10 §4). Id 0 = coast, 1/2 = +/-full at
// EVERY table size (K=3 is the ternary era's prefix); finer gears
// append as +/- pairs. MIRROR of constants.gear_table — the max table
// (K=7) is stored, a sim's `gears` field bounds the valid ids.
const GEAR_SIGNED: [Real; 7] = [0.0, 1.0, -1.0, 1.0 / 3.0, -1.0 / 3.0,
                               0.125, -0.125];
const GEAR_SIGNED_5: [Real; 7] = [0.0, 1.0, -1.0, 0.25, -0.25, 0.0, 0.0];

// Main's body (v63x2, since 2026-10-03; all 1.0 on branch body/original):
// the wrist-bend joints (8, 15) take their fine gears (ids >= 3) at 2x —
// K=5's 1/4 gear becomes 1/2 there.
// Mirrored by the kernel's FINE_GEAR_MULT and constants.gear_fraction.
const FINE_GEAR_MULT: [Real; NQ] = {
    let mut m = [1.0; NQ];
    m[8] = 2.0;
    m[15] = 2.0;
    m
};

fn gear_signed(gears: u8) -> [Real; 7] {
    match gears {
        5 => GEAR_SIGNED_5,
        7 => GEAR_SIGNED,
        _ => [0.0, 1.0, -1.0, 0.0, 0.0, 0.0, 0.0], // ternary (K=3)
    }
}

// Per-joint EXTRA gears (2026-10-04, per-joint gear counts): signed
// fractions for joint j's action ids gears, gears+1, ... (+/- pairs, not
// scaled by FINE_GEAR_MULT); JOINT_EXTRA_N[j] of them are live. All
// zero / 0 on main — every joint keeps its `gears` ids. MIRROR of
// constants.JOINT_EXTRA_GEARS and the kernel's JOINT_EXTRA_FRAC.
// exp/bend7 (user 2026-10-04, branch only): the wrist bends (8, 15) get
// +/-1/4 at ids gears, gears+1 — 7 gears at --gears 5.
const JOINT_EXTRA_N: [usize; NQ] = {
    let mut n = [0; NQ];
    n[8] = 2;
    n[15] = 2;
    n
};
const JOINT_EXTRA_FRAC: [[Real; 4]; NQ] = {
    let mut f = [[0.0; 4]; NQ];
    f[8] = [0.25, -0.25, 0.0, 0.0];
    f[15] = [0.25, -0.25, 0.0, 0.0];
    f
};

/// Per-joint action id -> signed torque fraction (constants
/// joint_gear_table, padded to 7 ids): gear_signed(gears) with the fine
/// ids (>= 3) at FINE_GEAR_MULT[j], then joint j's extra ids. Ids past a
/// joint's count read 0 (coast), as ids past `gears` always have. The
/// fine products are formed here once instead of per tick — the same
/// IEEE product, so the applied fractions are bit-identical.
fn joint_gear_rows(gears: u8) -> [[Real; 7]; NQ] {
    let base = gear_signed(gears);
    let k = match gears {
        5 => 5,
        7 => 7,
        _ => 3,
    };
    let mut rows = [[0.0 as Real; 7]; NQ];
    for j in 0..NQ {
        assert!(k + JOINT_EXTRA_N[j] <= 7,
                "joint {j}: gears={gears} + {} extra ids exceeds 7", JOINT_EXTRA_N[j]);
        for a in 0..7 {
            rows[j][a] = if a < k {
                if a >= 3 { base[a] * FINE_GEAR_MULT[j] } else { base[a] }
            } else if a - k < JOINT_EXTRA_N[j] {
                JOINT_EXTRA_FRAC[j][a - k]
            } else {
                0.0
            };
        }
    }
    rows
}

const N_BLADE_SAMPLES: usize = 8; // contact-point/velocity samples
// live broad-phase reach gate (SPEC-v10 §4): note_z <=
// max_hand(shoulder_z(t)) + 0.65 + 0.16 + 1.0 + 0.75 — the arm span past
// the shoulder (0.65) + handle (0.16, v42) + blade (1.0) + good-box
// near-face reach (0.75). At the reference pose (shoulder_z = -0.65)
// this is 1.91 (v9's REACH_Z 1.75 before the v42 handle).
const REACH_PAST_SHOULDER: Real = 0.65 + HANDLE_LEN + 1.0 + 0.75;
// env v11 FALL termination (unilateral-contact spec §4): with the feet
// on unilateral ground contact the root can go to the floor, and
// a fallen body's observations/rewards are not the task. Reference-pose
// root y is 0.94; the original 0.55 was a deep crouch the policy cannot
// reach by squatting within the leg limits and holding the sole flat.
// 2026-08-20: lowered to 0.45 then 0.4 to add dynamic headroom below the legal
// squat envelope — the entry-edge WALL_PEN wants ducks to be learnable,
// and a fast hip-fold descent transiently undershoots the static
// minimum. Reported exactly like the root-escape envelope (SPEC-v10
// §20e "reported, never hidden"): per-tick bool, popped cumulative
// count, episode counter.
const FALL_ROOT_Y: Real = 0.25;
// TILT termination (2026-08-20): CHEST up-axis
// tilted more than 40 deg FORWARD, 20 deg BACKWARD (asymmetric
// sagittal, user 2026-08-20 — the tight back bound targets the v30
// backbend exploit; forward stays duck-sized) or 40 deg to the side
// (lateral) at the END of a tick ends the episode, priced like a
// fall (flat -fall_const). Per-TICK check. Tangent-squared form;
// forward = up_z > 0 (body faces +z); also trips whenever
// up_y <= 0 (past horizontal).
const TILT_TAN2_FWD: Real = 0.7040881910418472; // tan^2(40 deg)
const TILT_TAN2_BACK: Real = 0.1324743314317942; // tan^2(20 deg) — 2026-08-21: briefly 10 deg, reverted to 20 same day (user)
const TILT_TAN2_ROLL: Real = 0.7040881910418472; // tan^2(40 deg)
// FOOT-LIFT penalty (2026-08-20): flat charge on
// the TICK a foot leaves the ground (edge: that foot's tick-summed
// ground normal impulse hits zero after a grounded tick), per foot.
const FOOT_LIFT_PEN: Real = -0.1;

// --- scoring (real Beat Saber 115, integer components) -----------------------
const PRE_FULL: Real = 1.7453292519943295; // 100 deg
const POST_FULL: Real = 1.0471975511965976; // 60 deg
const PTS_PRE: Real = 70.0;
const PTS_POST: Real = 30.0;
const PTS_ACC: Real = 15.0;
const CHAIN_LINK_PTS: Real = 20.0;
// rad per 30 Hz tick below which the swing chain breaks (0.15 rad/s);
// applied as SWING_BREAK * TickRate::tscale per sim tick
const SWING_BREAK: Real = 0.005;
// 30 Hz ticks a follow-through may keep accumulating (0.5 s); applied as
// FOLLOW_TIMEOUT * TickRate::ratio sim ticks
const FOLLOW_TIMEOUT: u32 = 15;

// --- control rate (exp/tick-rate, 2026-09-28) --------------------------------
// The physics substep is fixed at H_SUB = 1/240 s; the control tick is
// DT = 1/tick_hz with N_SUB = 240/tick_hz substeps, the action held for
// the whole tick. Every rule is written per 30 Hz tick (the training
// rate) and rescaled so it keeps its REAL-TIME meaning at any rate:
// per-tick thresholds / rewards × tscale, tick counts × ratio,
// tick-summed obs × ratio, "any this tick" obs bits OR'ed over the last
// `ratio` ticks (one 30 Hz tick of real time). At 30 Hz every factor is
// exactly 1 (x * 1.0 == x, n * 1 == n): bit-identical to the fixed-rate
// sim.
const TICK_HZ_BASE: u32 = 30;
const SUBSTEP_HZ: u32 = 240;

#[derive(Clone, Copy, Debug)]
pub(crate) struct TickRate {
    hz: u32,
    dt: Real,       // control tick (s) = 1 / hz
    n_sub: usize,   // physics substeps per tick = 240 / hz
    ratio: u32,     // sim ticks per 30 Hz tick = hz / 30
    ratio_r: Real,  // ratio as Real (tick-summed obs lanes × this)
    tscale: Real,   // 30 / hz: per-30-Hz-tick amounts × this per sim tick
}

impl TickRate {
    pub(crate) fn new(hz: u32) -> Result<TickRate, String> {
        if hz == 0 || hz % TICK_HZ_BASE != 0 || SUBSTEP_HZ % hz != 0 {
            return Err(format!(
                "tick_hz must be a multiple of {TICK_HZ_BASE} that divides \
                 {SUBSTEP_HZ} (30, 60, 120 or 240), got {hz}"));
        }
        let n_sub = (SUBSTEP_HZ / hz) as usize;
        let ratio = hz / TICK_HZ_BASE;
        Ok(TickRate {
            hz,
            // 1.0 / 30.0 at the default: the same correctly rounded
            // constant as geom::DT_30
            dt: 1.0 / hz as Real,
            n_sub,
            ratio,
            ratio_r: ratio as Real,
            tscale: TICK_HZ_BASE as Real / hz as Real,
        })
    }

    #[cfg(test)]
    pub(crate) fn base() -> TickRate {
        TickRate::new(TICK_HZ_BASE).unwrap()
    }
}
const _: () = assert!(N_SUB * TICK_HZ_BASE as usize == SUBSTEP_HZ as usize);
// env v19 flail telemetry: a tick is "note-free" when the next pending
// chart object is further out than this (s). Telemetry only.
const FLAIL_LEAD_S: Real = 1.0;
// wrist-bend reversal telemetry (2026-09-22, user): a reversal is
// counted when the wrist BEND axis (joints 8 / 15) turns back by
// WREV_HYST from its running extreme (peak detection with hysteresis,
// every tick). Mirrors the .cu twin exactly. 20 -> 60 deg (user,
// 2026-10-04): only full swings count, not mid-size corrections.
const WREV_HYST: Real = 1.0471975511965976;   // 60 deg (pi/3)
// wrist-reset telemetry (2026-10-04, user): per hand, a good note cut
// whose bend-velocity sign EQUALS the previous swing's, with a reversal
// of the bend motion in between, is one WRIST RESET (the hand went back
// without a hit). Reversals: the WREV state machine on the bend angle
// with its own WRESET_HYST = 30 deg (user) — a turn-back of 30 deg from
// the running extreme. 60 (WREV_HYST) would miss real resets with a
// short backswing; 30 is above tremor and mid-swing corrections. A cut
// with |bend velocity| < WRESET_VMIN takes the detector's committed
// direction instead of the velocity's sign. Mirrors the .cu twin.
const WRESET_HYST: Real = 0.5235987755982988;   // 30 deg (pi/6)
const WRESET_VMIN: Real = 1.0;                  // rad/s

/// Wrist-reset detector + counters for one hand (2026-10-04). Swings:
/// consecutive good hits of one continuous bend motion (no reversal
/// between them — a stack / slider / window cut in one motion) are ONE
/// swing. At a hit with sign s: no previous swing, or s != the previous
/// swing's sign (alternation), or a reversal since the last hit -> a
/// new swing; same sign WITH a reversal in between -> also a reset.
/// Same sign without one -> the same swing, nothing counted. Any number
/// of reversals between two hits add at most one reset. Only good note
/// cuts (chain heads count, links do not) are hits; bad cuts and misses
/// neither break nor continue a swing.
#[derive(Clone, Copy, Default)]
struct WristReset {
    dir: u8,      // 0 fresh, 3 armed (no direction yet), 1 moving +, 2 moving -
    lo: Real,     // running extremes of the bend angle
    hi: Real,
    rev: bool,    // a reversal committed since the last hit
    last: u8,     // previous swing's sign: 0 none, 1 +, 2 -
    resets: u32,  // this episode
    swings: u32,
}

impl WristReset {
    /// One tick of the bend angle q (after the physics, before the
    /// judge — the WREV detector's slot).
    fn tick(&mut self, q: Real) {
        let (mut lo, mut hi) = (self.lo, self.hi);
        match self.dir {
            0 => {
                lo = q;
                hi = q;
                self.dir = 3;
            }
            3 => {
                if q < lo {
                    lo = q;
                }
                if q > hi {
                    hi = q;
                }
                if q >= lo + WRESET_HYST {
                    self.dir = 1;
                    hi = q;
                } else if q <= hi - WRESET_HYST {
                    self.dir = 2;
                    lo = q;
                }
            }
            1 => {
                if q > hi {
                    hi = q;
                } else if q <= hi - WRESET_HYST {
                    self.rev = true;
                    self.dir = 2;
                    lo = q;
                }
            }
            _ => {
                if q < lo {
                    lo = q;
                } else if q >= lo + WRESET_HYST {
                    self.rev = true;
                    self.dir = 1;
                    hi = q;
                }
            }
        }
        self.lo = lo;
        self.hi = hi;
    }

    /// A good note cut by this hand with bend velocity qd. Returns
    /// (reset, new swing).
    fn hit(&mut self, qd: Real) -> (bool, bool) {
        let s: u8 = if qd.abs() >= WRESET_VMIN {
            if qd > 0.0 { 1 } else { 2 }
        } else if self.dir == 1 || self.dir == 2 {
            self.dir
        } else if qd >= 0.0 {
            1
        } else {
            2
        };
        let reset = self.last != 0 && s == self.last && self.rev;
        let swing = self.last == 0 || s != self.last || self.rev;
        self.resets += reset as u32;
        self.swings += swing as u32;
        self.last = s;
        self.rev = false;
        (reset, swing)
    }
}
// env v20 episode-start arm stance (§1.4 indices): l/r arm_pitch -80
// deg, l/r elbow 100 deg, l/r wrist_bend 70 deg.
// env v21 (user, 2026-08-16): wide stance — hip_roll splays the legs
// +-8 deg, ankle_roll counter-rolls so the soles stay flat (verified
// via the fk_body mirror: foot up-vector exactly (0,1,0)).
const Q_START_LEG: [(usize, Real); 4] = [
    (19, -0.13962634015954636),  // l_hip_roll  -8 deg
    (22, 0.13962634015954636),   // l_ankle_roll +8 deg (counter)
    (25, 0.13962634015954636),   // r_hip_roll  +8 deg
    (28, -0.13962634015954636),  // r_ankle_roll -8 deg (counter)
];
// The wide stance's foot X positions (fk_body of the v21 start pose;
// the root ground-drop does not move X). Per-foot penalty targets:
// left foot -> -FOOT_X0, right foot -> +FOOT_X0.
const FOOT_X0: Real = 0.22108059783525694;

const Q_START_ARM: [(usize, Real); 6] = [
    (3, -1.3962634015954636),
    (6, 1.7453292519943295),
    (8, 1.2217304763960306),
    (10, -1.3962634015954636),
    (13, 1.7453292519943295),
    (15, 1.2217304763960306),
];

// env v17 (contact spec §12): walls. The head CENTER is the collision
// point (no sweep). WALL_PEN is charged ONCE, on the tick the head
// ENTERS a wall (the was_in_wall rising edge, same trigger as the
// combo break; >= 1 tick fully outside re-arms it) — 2026-08-20,
// was -0.05 every tick inside. walls_in still counts ticks inside.
// The viewer builds wall front faces 0.25 m behind the note plane
// ("walls just built like that" — WallManager.cs UpdateVisual), so a
// wall's front reaches the head 0.4/NJS s after its nominal time.
const WALL_PEN: Real = -1.5; // 2026-08-22: -0.5 -> -1.5 (bomb parity, user)
const WALL_FRONT_OFF: Real = 0.25;

// --- obs layout v17 (spec §12d; offsets DERIVED, order is a contract) --------
// The 16 slots are a SHARED entity window (notes + walls), filled
// nearest-first by 3D distance from the head center; entities behind
// the head consume no slots. ENT_FEAT = the v9 note lanes verbatim,
// then [is_wall, width, height, back-face z] (last lane carried wall
// length until 2026-08-22; retired as redundant with back-face minus
// front-face lane 2).
const N_NOTE_SLOTS: usize = 24; // env v18: 16 -> 24 (eviction study)
const NOTE_FEAT: usize = 18;
// env v19: + remaining follow-through credit (1 lane) — Σ over the
// hand's live trackers of (1 − min(acc/POST_FULL, 1)), clipped to 2.
// Fixes the post-swing credit-assignment hole: follow-through reward
// lands after the note leaves the obs, which taught policies that
// swinging at nothing pays (the flail bug, user diagnosis 2026-08-16).
const HAND_FEAT: usize = 11; // wrist pos(3) + vel(3) + blade(3) + meter + post_rem
const Q_LANES: usize = NQ + 2; // 31: the two wrist rolls code as cos/sin
const OFF_ROOT_POS: usize = 0; // x/1, (y-0.94)/0.75, (z+0.65)/1
const OFF_ROOT_ROT: usize = OFF_ROOT_POS + 3; // 6D two-column rotation
const OFF_ROOT_VLIN: usize = OFF_ROOT_ROT + 6; // /6
const OFF_ROOT_VANG: usize = OFF_ROOT_VLIN + 3; // world, /16
const OFF_Q: usize = OFF_ROOT_VANG + 3; // 15
const OFF_QD: usize = OFF_Q + Q_LANES; // 46: qd_j / omega_cap[j]
const OFF_HAND_L: usize = OFF_QD + NQ; // 75
const OFF_HAND_R: usize = OFF_HAND_L + HAND_FEAT; // 86 (env v19)
const OFF_HEAD: usize = OFF_HAND_R + HAND_FEAT; // 97 (env v19)
const OFF_CONTACT: usize = OFF_HEAD + 3; // 100: contact bits [L, R]
const OFF_NOTES: usize = OFF_CONTACT + 2; // 102 (env v19)
const OFF_NOTE_MASK: usize = OFF_NOTES + N_NOTE_SLOTS * NOTE_FEAT; // 532
// env v12 (unilateral-contact spec §7): per-foot ground-contact lanes
// [L_heel, L_toe, R_heel, R_toe] — tick-summed normal impulse (N·s) per
// front/back sole-corner pair, / GROUND_IMP_NORM.
const OFF_GROUND_IMP: usize = OFF_NOTE_MASK + N_NOTE_SLOTS; // 556
const N_GROUND_IMP_LANES: usize = 4;
// env v13 (contact spec §8): explicit FK shortcuts — world positions of
// [l_elbow, r_elbow, l_knee, r_knee, l_foot, r_foot], head-style
// normalization (x/1, (y−Y0)/0.75, (z−PLANE_Z0)/1). Y0 is the point's
// reference-pose height: elbows share the wrists' Y_CENTER (they live
// in the note field), knees/feet get their §1.2 reference literals.
const OFF_BODY_PTS: usize = OFF_GROUND_IMP + N_GROUND_IMP_LANES; // 408
const N_BODY_PTS: usize = 6;
// env v24 (user direction 2026-08-17): world VELOCITIES of the same 6
// §8 body points, wrist_vel-style finite difference across the tick,
// /V_NORM. Point velocity is otherwise a deep FK-Jacobian computation
// to ask of the network — and foot world velocity is exactly the slip
// signal (same legibility lesson as the v19 post_rem lane).
const OFF_BODY_PT_VEL: usize = OFF_BODY_PTS + 3 * N_BODY_PTS; // 580
// env v45 (user order 2026-09-15; replaces the same-day v44 block):
// per-hand PREVIOUS-NOTE block — for hand s the most recently RESOLVED
// scoreable note OF THAT HAND'S COLOUR (colour 0 -> L, 1 -> R),
// regardless of which saber resolved it and regardless of outcome:
// good cut, bad cut (slow/angle), wrong-hand cut and miss all update
// the note's-colour block and never the other hand's; bombs and walls
// never update anything. Purpose (user): parity control, wind-up
// positioning, and holding the saber at the right spot through gaps
// until the next note. Lanes, in the SAME head-relative frame and
// normalizers as the entity block (x/y recomputed from the note's
// world position against the LIVE head each tick):
//   0 x_rel, 1 y_rel, 2 dir_cos, 3 dir_sin (zero on dot notes,
//   tangent on chain links — the entity rule), 4 valid (1 once any
//   note of that colour has resolved; the whole block is 0 before).
// No time / outcome / colour / chain / free lanes (v44's were retired
// the same day at the user's direction). APPENDED after the v24 block
// so every earlier offset is unchanged: L block at 598, R block at
// 603. Within a tick, later chart indices overwrite earlier ones (the
// judge loop walks chart order), so the record is the latest-hit_time
// note of that colour resolved so far.
const PREV_FEAT: usize = 5;
const OFF_PREV_L: usize = OFF_BODY_PT_VEL + 3 * N_BODY_PTS; // 598
const OFF_PREV_R: usize = OFF_PREV_L + PREV_FEAT; // 603
// env v46 (user design 2026-09-16): per-env joint torque-limit SCALES,
// one per AXIS (16: spine 3, arm 7 shared L/R, leg 6 shared L/R), as
// (s − 1) / TAU_RAND_NORM lanes appended after the prev-note blocks.
// The Rust sim has no per-episode RNG (jitter/pushes/crouch starts are
// CUDA-only training features), so here the scales are EXPLICIT
// (set_tau_scale; default all 1.0 → lanes 0, torque bit-identical to
// v45 since x * 1.0 is exact). The CUDA kernel draws them at reset.
const TAU_FEAT: usize = 16;
const TAU_RAND_NORM: Real = 0.25;
const OFF_TAU: usize = OFF_PREV_R + PREV_FEAT; // 608
// env v49 (review item I07, 2026-09-30): per-hand BLADE-TIP world
// velocity, [L xyz, R xyz] / V_NORM, appended after the tau lanes so
// every earlier offset is unchanged. The tip is wrist + (HANDLE_LEN +
// SABER_LEN)·blade; its finite difference across the tick is exactly
// the judge's `tip_move / DT` — the vector the cut-plane accuracy rule
// (d_min = |cw·norm(blade × tip_move)|) and, near the tip, the contact
// velocity of the direction check are built from — which the wrist
// velocity (hand block lanes 3..5) misses whenever the saber rotates
// (a 1.16 m lever). World frame (like wrist_vel), zero at reset and
// after a set_state refresh; zero in the Flux game (no blade).
const TIP_VEL_FEAT: usize = 6;
const OFF_TIP_VEL: usize = OFF_TAU + TAU_FEAT; // 624
const OBS_DIM: usize = OFF_TIP_VEL + TIP_VEL_FEAT; // 630 (env v49)
// joint -> axis (constants.TAU_AXIS_OF_JOINT mirror)
const TAU_AXIS: [usize; NQ] = [
    0, 1, 2,
    3, 4, 5, 6, 7, 8, 9,
    3, 4, 5, 6, 7, 8, 9,
    10, 11, 12, 13, 14, 15,
    10, 11, 12, 13, 14, 15,
];
// §8 body-point frames + their obs y-centering (shared by the position
// lanes, the velocity finite-diff and the foot reward terms' docs)
const BODY_PTS: [(usize, Real); N_BODY_PTS] = [
    (F_L_ELBOW, Y_CENTER), (F_R_ELBOW, Y_CENTER),
    (F_L_SHANK, KNEE_Y0), (F_R_SHANK, KNEE_Y0),
    (F_L_FOOT, FOOT_Y0), (F_R_FOOT, FOOT_Y0),
];
const _: () = assert!(OBS_DIM == 630, "spec §12 (env v49 blade-tip velocity lanes): OBS_DIM must be 630");
const KNEE_Y0: Real = 0.50; // hip 0.94 − thigh 0.44
const FOOT_Y0: Real = 0.07; // ankle height at reference (contact spec §2)
// ≈ one tick of full-body weight: M_TOTAL·g·DT = 74.50 kg · 9.81 · (1/30)
// = 24.53 N·s (24.46 at main's 74.80 kg since 2026-10-03; the literal is
// kept), rounded to a spec literal both engines share. A settled
// two-foot stance sums to ≈ 1.0 across the four lanes.
const GROUND_IMP_NORM: Real = 24.5;
const _: () = assert!(OFF_QD == 46 && OFF_HAND_L == 75 && OFF_NOTES == 102);

const X_NORM: Real = 1.0;
const Y_NORM: Real = 0.75;
const Z_SCALE: Real = 8.0;
// env v48 (2026-09-20, user): asinh z lane — lanes 2/17 are
// asinh(max(z, z_lo) / c) with c = BEAT_Z_ASINH metres (linear near the
// plane, log far out, no upper clamp); the only z encoding since
// 2026-09-22 (the legacy clamp / Z_SCALE path is gone). Unset = 8 m (the
// recipe value); a value <= 0 or unparsable panics. Read once per
// process; the CUDA twin takes it as a step parameter (z_asinh).
fn z_asinh() -> Real {
    static V: std::sync::OnceLock<Real> = std::sync::OnceLock::new();
    *V.get_or_init(|| {
        let c: Real = match std::env::var("BEAT_Z_ASINH") {
            Ok(s) => s
                .parse()
                .unwrap_or_else(|_| panic!("BEAT_Z_ASINH={s:?} is not a number")),
            Err(_) => 8.0,
        };
        assert!(c > 0.0, "BEAT_Z_ASINH must be > 0 (metres), got {c}");
        c
    })
}
/// The obs z lane for a travelling object (.cu z_lane mirror).
fn z_lane(zu: Real, z_lo: Real) -> Real {
    (zu.max(z_lo) / z_asinh()).asinh()
}
const ROOT_Y0: Real = 0.94; // reference root height (obs centering)
const HEAD_Y0: Real = 1.70; // reference head height (obs centering)
const ROOT_VLIN_NORM: Real = 6.0;
const ROOT_VANG_NORM: Real = 16.0;

/// env v17 obstacle (spec §12): a world-space AABB riding the
/// head-anchored track. Geometry arrives FINISHED from the parser
/// (viewer scales/clamps applied); the sim only moves the box and
/// tests the head point. Half-extents precomputed for the AABB test.
#[derive(Clone, Copy)]
struct Wall {
    start_time: Real, // front face crosses the note plane (nominal, s)
    cx: Real,         // box center x (m, world)
    cy: Real,         // box center y (m, world)
    half_w: Real,
    half_h: Real,
    len: Real,        // full z extent (m)
    speed: Real,      // approach speed (m/s)
}

/// Closest distance from point p to triangle (a, b, c) (Ericson,
/// Real-Time Collision Detection §5.1.5). Handles degenerate triangles
/// via the edge/vertex regions.
fn tri_point_dist(a: [Real; 3], b: [Real; 3], c: [Real; 3], p: [Real; 3]) -> Real {
    let ab = sub3(b, a);
    let ac = sub3(c, a);
    let ap = sub3(p, a);
    let d1 = dot3(ab, ap);
    let d2 = dot3(ac, ap);
    let dist = |q: [Real; 3]| {
        let d = sub3(p, q);
        dot3(d, d).sqrt()
    };
    if d1 <= 0.0 && d2 <= 0.0 {
        return dist(a);
    }
    let bp = sub3(p, b);
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
    let cp = sub3(p, c);
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

fn seg_point_dist3(a: [Real; 3], b: [Real; 3], c: [Real; 3]) -> Real {
    let ab = sub3(b, a);
    let denom = dot3(ab, ab);
    let ac = sub3(c, a);
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

/// The swept blade surface over one tick: quad (b0->t0) to (b1->t1),
/// split into triangles (b0, t0, t1) and (b0, t1, b1), where b = the
/// blade BASE (wrist + HANDLE_LEN * dir, env v42) and t = the tip
/// (wrist + (HANDLE_LEN + SABER_LEN) * dir). The handle [wrist, b) is
/// not part of the surface: it does not cut.
fn sweep_tris(
    w0: [Real; 3],
    d0: [Real; 3],
    w1: [Real; 3],
    d1: [Real; 3],
) -> [[[Real; 3]; 3]; 2] {
    let b0 = [
        w0[0] + HANDLE_LEN * d0[0],
        w0[1] + HANDLE_LEN * d0[1],
        w0[2] + HANDLE_LEN * d0[2],
    ];
    let b1 = [
        w1[0] + HANDLE_LEN * d1[0],
        w1[1] + HANDLE_LEN * d1[1],
        w1[2] + HANDLE_LEN * d1[2],
    ];
    let t0 = [
        w0[0] + (HANDLE_LEN + SABER_LEN) * d0[0],
        w0[1] + (HANDLE_LEN + SABER_LEN) * d0[1],
        w0[2] + (HANDLE_LEN + SABER_LEN) * d0[2],
    ];
    let t1 = [
        w1[0] + (HANDLE_LEN + SABER_LEN) * d1[0],
        w1[1] + (HANDLE_LEN + SABER_LEN) * d1[1],
        w1[2] + (HANDLE_LEN + SABER_LEN) * d1[2],
    ];
    [[b0, t0, t1], [b0, t1, b1]]
}

fn sweep_dist(tris: &[[[Real; 3]; 3]; 2], p: [Real; 3]) -> Real {
    tri_point_dist(tris[0][0], tris[0][1], tris[0][2], p)
        .min(tri_point_dist(tris[1][0], tris[1][1], tris[1][2], p))
}

/// Entity-window selection: insert `cand` = (dist², is_wall, index)
/// into the ascending buffer; ties keep the FIRST-inserted entry
/// (strict-< displacement), a full buffer drops anything not closer
/// than its worst entry. Shared by the standard and Flux obs writers.
fn slot_insert(
    sel: &mut [(Real, bool, usize); N_NOTE_SLOTS],
    n_sel: &mut usize,
    cand: (Real, bool, usize),
) {
    if *n_sel == N_NOTE_SLOTS && cand.0 >= sel[N_NOTE_SLOTS - 1].0 {
        return;
    }
    let mut i = (*n_sel).min(N_NOTE_SLOTS - 1);
    while i > 0 && cand.0 < sel[i - 1].0 {
        sel[i] = sel[i - 1];
        i -= 1;
    }
    sel[i] = cand;
    *n_sel = (*n_sel + 1).min(N_NOTE_SLOTS);
}

/// A good cut's pending follow-through meter.
#[derive(Clone, Copy)]
struct Tracker {
    note: u32,
    acc: Real,
    age: u32,
}

type ChartTuple = (Real, Real, Real, u8, Real, Real, bool, u8, bool, bool);
// wall_tuple order (chart.py): (t0, cx, cy, width, height, length, speed)
type WallTuple = (Real, Real, Real, Real, Real, Real, Real);

/// The v10 physical body: dynamics state + solver buffers + the derived
/// per-tick caches the judgment/obs layers consume. One per env.
struct Body {
    // control rate (TickRate): substeps per tick, DT for the finite
    // differences, the contact-bit window. Fixed at construction.
    rate: TickRate,
    st: State,
    ws: Workspace,
    cs: ConstraintSet,
    tau: [Real; NQ], // this tick's applied torques (constant across substeps)
    // env v46: per-axis torque-limit scales (16) + their per-joint
    // expansion; NOT touched by reset() (explicit via set_tau_scale)
    tau_axis: [Real; TAU_FEAT],
    tau_scale: [Real; NQ],
    // FK caches (refreshed after every tick / state write)
    wrist: [[Real; 3]; 2],
    blade: [[Real; 3]; 2],
    head: [Real; 3],
    shoulder_z: [Real; 2],
    // judgment caches
    wrist_vel: [[Real; 3]; 2], // finite difference across the last tick
    // env v49: blade-TIP world velocity, the same finite difference on
    // wrist + (HANDLE_LEN + SABER_LEN)·blade (= the judge's tip_move / DT)
    tip_vel: [[Real; 3]; 2],
    // env v24: §8 body-point world velocities, same finite-diff scheme
    body_pt_vel: [[Real; 3]; N_BODY_PTS],
    swing_acc: [Real; 2],      // arc of the current continuous swing (rad)
    axis: [Option<[Real; 3]>; 2], // last tick's rotation axis (unit)
    // per-tick contact aggregation (obs bits + replay events)
    contact_arm: [bool; 2],          // any active contact row touching the arm
    // sim ticks since the arm last had an active contact row (saturating;
    // u32::MAX = never). The obs bit is `age < rate.ratio`: "any contact
    // in the last 30 Hz tick of real time" (== this tick at 30 Hz).
    contact_arm_age: [u32; 2],
    contact_imp: [Real; NUM_PAIRS],   // tick-summed normal impulse per pair
    contact_pt: [[Real; 3]; NUM_PAIRS], // last substep's contact point
    // env v23 activation (user direction 2026-08-17): this tick's
    // commanded |gear fraction| summed over the 29 joints — the
    // muscle-activation analog (Σ|tau_j|/tau_max_j, constant over the
    // tick since tau is held). Telemetry since env v43 (the price
    // moved to Body::tau_abs / Core::energy_coef); coast reads 0.
    act_frac: Real,
    // the same sum split [spine (joints 0..3), arms (3..17), legs
    // (17..29)] — telemetry only (2026-09-13, replaces the retired
    // positive-work reading); act_frac is the priced total.
    act_group: [Real; 3],
    // env v43 (2026-09-14, user): this tick's applied Σ|τ_j| in N·m
    // (= Σ|frac_j|·τ_max[j], the energy-priced total) and its
    // [spine, arms, legs] split — same joint-order summation as
    // act_frac; CUDA computes the identical double sum.
    tau_abs: Real,
    tau_group: [Real; 3],
    // exp/wrist-tendon telemetry: this tick's passive wrist torque,
    // summed over its substeps, per hand — [Σ(|τ_bend| + |τ_dev|) L, R
    // (N·m), substeps in the zone r > R0 L, R, substeps beyond the
    // ellipse r > 1 L, R, edge-damping dissipated power Σ b_e·q̇² L, R
    // (W, issue #61)]. Neither the tendon nor its damping is muscle
    // effort: they never enter `tau` / `tau_abs` (energy price,
    // [torque], replay torque).
    tendon_tick: [Real; 8],
    // exp/wrist-tendon TEST/TOOLING switch: false steps the substeps with
    // HUMANOID_NO_TENDON (identical tables, passive_wrist = None). Default
    // true; training, bench and replays never touch it (the CUDA kernel
    // has no switch). For kinematic judge fixtures that pose the wrist
    // deep in the tendon zone (RawBeatSim.set_passive_wrist).
    tendon_on: bool,
    // exp/shoulder-rom telemetry: this tick's passive shoulder terms,
    // summed over its substeps, per arm — [Σ(|τ_yaw| + |τ_pitch| +
    // |τ_twist|) L, R (N·m), substeps in the zone |u| > s L, R, substeps
    // beyond the human limit |u| > 1 L, R, edge-damping dissipated power
    // Σ b·τ̇² L, R (W)]. Never muscle effort (not in `tau` / `tau_abs`).
    srom_tick: [Real; 8],
    // exp/shoulder-rom TEST/TOOLING switch (RawBeatSim.set_passive_
    // shoulder): false steps with passive_shoulder = None. Default true;
    // training, bench and replays never touch it (CUDA has no switch).
    srom_on: bool,
    // C backstop (2026-08-03, SPEC-v10 Divergences §23): did the
    // PRE-integration root envelope engage anywhere in THIS tick's 8
    // substeps? A1 makes such a trajectory bounded; it does not make it
    // physics, so the env terminates the episode instead of feeding the
    // policy the garbage observation/reward (the fp16 loss-scale
    // collapse and the §22 bench panic were both downstream of exactly
    // that). Measured incidence on a converged policy: 0 in 204,800
    // env-ticks.
    escaped: bool,
    // Contact spec §4: was the root below FALL_ROOT_Y at the END of
    // this tick (per-TICK since 2026-08-20)? Same
    // envelope pattern as `escaped`: the env terminates the episode
    // and zeroes the reward.
    fell: bool,
    // TILT (2026-08-20): did the CHEST up-axis exceed the sagittal /
    // lateral limits during THIS tick's 8 substeps? Same envelope
    // pattern as `fell`, same flat -fall_const pricing.
    tilted: bool,
    // Cumulative fall terminations until the owner pops them (lib.rs
    // pop_fall_hits) — the `ws.env_hits` / `pd_hits` pattern.
    fall_hits: u64,
    // Cumulative tilt terminations (pop_tilt_hits) — fall_hits pattern.
    tilt_hits: u64,
    // FOOT-LIFT (2026-08-20): per-foot grounded-last-tick edge state,
    // this tick's lift-event count (0..2), and the cumulative pop
    // counters (pop_foot_lifts, PER FOOT [L, R]) — fall_hits pattern.
    foot_down_prev: [bool; 2],
    foot_lifts_tick: u8,
    foot_lift_hits: [u64; 2],
    // env v12 (contact spec §7): tick-summed ground NORMAL impulses per
    // front/back sole-corner pair — [L_heel, L_toe, R_heel, R_toe]
    // (N·s). Feeds the four OFF_GROUND_IMP obs lanes.
    ground_imp: [Real; 4],
    // env v34 (2026-08-21): tick-summed joint-limit-row impulses
    // (N·m·s) the stops carried, bucketed [legs, spine]. Scope is
    // POSTURAL joints only — spine 0..3 and legs 17..29; the 14 arm
    // joints are excluded (whips legitimately slam arm stops at
    // speed). Step-11 clamp impulses are NOT counted: the solver is
    // limit-aware, so those are near-null by design (solver.rs R10
    // notes). Priced by Core::limit_coef; resting a pose on its
    // stops shows up as sustained λ-flow here, an active hold does
    // not.
    limit_imp: [Real; 3],
    // replay fix (2026-08-17): tick-summed ground-reaction impulse per
    // foot [L, R] x [x friction, y normal, z friction] (N·s, world
    // frame) — ALL rows of every active sole corner, not just the §7
    // normal buckets. Feeds the replay "feet" lanes.
    foot_grf: [[Real; 3]; 2],
}

const MODEL: &Model = &HUMANOID;
/// HUMANOID without the passive wrist tendon (Body::tendon_on = false —
/// a test/tooling switch only).
const HUMANOID_NO_TENDON: Model = Model { passive_wrist: None, ..HUMANOID };
/// ... without the passive shoulder (Body::srom_on = false), and without
/// both — test/tooling switches only.
const HUMANOID_NO_SROM: Model = Model { passive_shoulder: None, ..HUMANOID };
const HUMANOID_NO_PASSIVE: Model =
    Model { passive_wrist: None, passive_shoulder: None, ..HUMANOID };

impl Body {
    #[cfg(test)]
    fn new() -> Body {
        Body::with_rate(TickRate::base())
    }

    fn with_rate(rate: TickRate) -> Body {
        let st = State::neutral(MODEL);
        let ws = Workspace::new(MODEL);
        // self-collision rows + the contact spec §3 ground block
        let cs = ConstraintSet::new(MODEL, true, true);
        let mut b = Body {
            rate,
            st,
            ws,
            cs,
            tau: [0.0; NQ],
            wrist: [[0.0; 3]; 2],
            blade: [[0.0; 3]; 2],
            head: [0.0; 3],
            shoulder_z: [0.0; 2],
            wrist_vel: [[0.0; 3]; 2],
            tip_vel: [[0.0; 3]; 2],
            body_pt_vel: [[0.0; 3]; N_BODY_PTS],
            swing_acc: [0.0; 2],
            axis: [None; 2],
            contact_arm: [false; 2],
            contact_arm_age: [u32::MAX; 2],
            contact_imp: [0.0; NUM_PAIRS],
            contact_pt: [[0.0; 3]; NUM_PAIRS],
            act_frac: 0.0,
            act_group: [0.0; 3],
            tau_abs: 0.0,
            tau_group: [0.0; 3],
            tendon_tick: [0.0; 8],
            tendon_on: true,
            srom_tick: [0.0; 8],
            srom_on: true,
            tau_axis: [1.0; TAU_FEAT],
            tau_scale: [1.0; NQ],
            escaped: false,
            fell: false,
            tilted: false,
            fall_hits: 0,
            tilt_hits: 0,
            foot_down_prev: [false, false],
            foot_lifts_tick: 0,
            foot_lift_hits: [0, 0],
            ground_imp: [0.0; 4],
            limit_imp: [0.0; 3],
            foot_grf: [[0.0; 3]; 2],
        };
        b.refresh_fk();
        b
    }

    /// env v46: set the 16 axis scales (both sides of an axis share one
    /// value) and expand them per joint. Persists across resets.
    fn set_tau_axis(&mut self, s: &[Real; TAU_FEAT]) {
        self.tau_axis = *s;
        for j in 0..NQ {
            self.tau_scale[j] = s[TAU_AXIS[j]];
        }
    }

    fn reset(&mut self) {
        self.st = State::neutral(MODEL);
        // env v20 (user, 2026-08-16): episode-start ARM pose — the
        // real-player ready stance (arms hang, forearms cocked, wrists
        // tilted forward) instead of the zombie arms-forward zero pose.
        // Literals are exact f64 radians shared verbatim with the .cu
        // and constants.py mirrors (parity-bound).
        for (j, v) in Q_START_ARM {
            self.st.q[j] = v;
        }
        // env v21 wide stance
        for (j, v) in Q_START_LEG {
            self.st.q[j] = v;
        }
        self.tau = [0.0; NQ];
        self.wrist_vel = [[0.0; 3]; 2];
        self.tip_vel = [[0.0; 3]; 2];
        self.body_pt_vel = [[0.0; 3]; N_BODY_PTS];
        self.swing_acc = [0.0; 2];
        self.axis = [None; 2];
        self.contact_arm = [false; 2];
        self.contact_arm_age = [u32::MAX; 2];
        self.contact_imp = [0.0; NUM_PAIRS];
        self.act_frac = 0.0;
        self.act_group = [0.0; 3];
        self.tau_abs = 0.0;
        self.tau_group = [0.0; 3];
        self.tendon_tick = [0.0; 8];
        self.srom_tick = [0.0; 8];
        self.escaped = false;
        self.fell = false;
        self.tilted = false;
        // arm-after-first-plant (user 2026-08-20): start UNARMED so a
        // foot that begins the episode in the air (start-pose jitter)
        // is never charged — the lift edge only arms once the foot has
        // actually been seen grounded.
        self.foot_down_prev = [false, false];
        self.foot_lifts_tick = 0;
        self.ground_imp = [0.0; 4];
        self.foot_grf = [[0.0; 3]; 2];
        self.refresh_fk();
        // env v21: the wide stance lifts the ankles ~8.5 mm, so drop
        // the root until the lowest sole corner sits exactly on the
        // ground (the v20 jitter's corner arithmetic; both engines
        // share this expression tree).
        use dynamics::collide::{GROUND_CORNERS, GROUND_FRAMES};
        let mut min_y = Real::INFINITY;
        for fr in GROUND_FRAMES {
            let (r, p) = (&self.ws.r[fr], &self.ws.p[fr]);
            for c in &GROUND_CORNERS {
                let y = p[1] + r[1][0] * c[0] + r[1][1] * c[1]
                    + r[1][2] * c[2];
                if y < min_y {
                    min_y = y;
                }
            }
        }
        self.st.p[1] -= min_y;
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

    /// One control tick: gear ids -> torques, rate.n_sub × substep at
    /// h = 1/240 (SPEC-v10 §3/§4; 8 at the 30 Hz default), accumulating
    /// activation and contact impulses, then FK-cache refresh +
    /// finite-diff wrist vel over rate.dt.
    fn tick(&mut self, actions: &[u8; NQ], gear: &[[Real; 7]; NQ]) {
        let mut act = 0.0;
        let mut grp = [0.0 as Real; 3];
        let mut tau_abs = 0.0;
        let mut tgrp = [0.0 as Real; 3];
        for j in 0..NQ {
            let a = actions[j] as usize;
            // per-joint row (joint_gear_rows: FINE_GEAR_MULT + extras)
            let frac = if a >= 7 { 0.0 } else { gear[j][a] };
            // env v46: τ_max scaled per env (s = 1.0 is exact: bit-
            // identical to v45); same expression tree as the kernel
            self.tau[j] = frac * MODEL.tau_max[j] * self.tau_scale[j];
            let f = frac.abs();
            act += f;
            let g = if j < 3 { 0 } else if j < 17 { 1 } else { 2 };
            grp[g] += f;
            // env v43 energy base: |τ_j| = |frac·τ_max·s| (abs commutes
            // with the product exactly; same expression tree as the kernel)
            let t = (frac * MODEL.tau_max[j] * self.tau_scale[j]).abs();
            tau_abs += t;
            tgrp[g] += t;
        }
        self.act_frac = act;
        self.act_group = grp;
        self.tau_abs = tau_abs;
        self.tau_group = tgrp;
        let prev_wrist = self.wrist;
        let prev_blade = self.blade;
        // env v24: last tick's §8 body-point positions (ws.p holds the
        // previous refresh_fk), the finite-diff base for body_pt_vel
        let mut prev_bp = [[0.0 as Real; 3]; N_BODY_PTS];
        for (i, (f, _)) in BODY_PTS.iter().enumerate() {
            prev_bp[i] = self.ws.p[*f];
        }
        let mut arm_now = [false; 2];
        self.contact_imp = [0.0; NUM_PAIRS];
        self.ground_imp = [0.0; 4];
        self.limit_imp = [0.0; 3];
        self.foot_grf = [[0.0; 3]; 2];
        // C backstop: this tick's pre-integration envelope engagements
        // are ws.env_hits' delta over the 8 substeps (the counter itself
        // is cumulative until pop_envelope_hits)
        let env_hits0 = self.ws.env_hits;
        // contact spec §4: the fall test is a PER-SUBSTEP one, like the
        // envelope above — a body that dips under the floor mid-tick and
        // bounces back has still stopped being the standing task.
        let mut fell = false;
        let mut tilted = false;
        self.tendon_tick = [0.0; 8];
        self.srom_tick = [0.0; 8];
        let model: &Model = match (self.tendon_on, self.srom_on) {
            (true, true) => MODEL,
            (false, true) => &HUMANOID_NO_TENDON,
            (true, false) => &HUMANOID_NO_SROM,
            (false, false) => &HUMANOID_NO_PASSIVE,
        };
        if !self.tendon_on {
            self.ws.tendon = [[0.0; 4]; 2]; // step 4c never writes it
        }
        if !self.srom_on {
            self.ws.shoulder = [[0.0; 4]; 2]; // step 4d never writes it
        }
        let srom_s = MODEL.passive_shoulder.map_or(0.0, |p| p.zone_start);
        for _ in 0..self.rate.n_sub {
            substep_constrained(model, &mut self.st, &self.tau, H_SUB,
                                &mut self.ws, &mut self.cs);
            // exp/wrist-tendon telemetry: the passive torque step 4
            // applied this substep (ws.tendon, [τ_b, τ_d, r] per hand)
            for h in 0..2 {
                let [tb, td, r, pd] = self.ws.tendon[h];
                self.tendon_tick[h] += tb.abs() + td.abs();
                self.tendon_tick[6 + h] += pd;
                if r > TENDON_R0 {
                    self.tendon_tick[2 + h] += 1.0;
                }
                if r > 1.0 {
                    self.tendon_tick[4 + h] += 1.0;
                }
            }
            // exp/shoulder-rom telemetry (ws.shoulder, [Σ|τ_p|, u, b,
            // P_diss] per arm)
            for a in 0..2 {
                let [tq, u, _b, pd] = self.ws.shoulder[a];
                self.srom_tick[a] += tq;
                self.srom_tick[6 + a] += pd;
                if u.abs() > srom_s {
                    self.srom_tick[2 + a] += 1.0;
                }
                if u.abs() > 1.0 {
                    self.srom_tick[4 + a] += 1.0;
                }
            }

            // contact aggregation: obs bits see every ACTIVE row this
            // substep; replay impulses sum the row lambdas (N·s)
            // the self-collision rows sit AFTER the ground block —
            // contact spec §3
            let ncbase = self.cs.ground_rows();
            // contact spec §7: tick-summed ground NORMAL impulses,
            // bucketed [L_heel, L_toe, R_heel, R_toe] — corner ids are
            // foot*4 + k with k odd = z+ (toe), so bucket =
            // foot*2 + (k & 1). Obs lanes read these after the tick.
            for (gi, g) in self.cs.ground_contacts.iter().enumerate() {
                let b = (g.corner / CORNERS_PER_FOOT) * 2 + (g.corner & 1);
                let nrow = self.cs.ground_row(gi);
                self.ground_imp[b] += self.cs.lambda[nrow];
                // full ground reaction per foot for the replay
                // "feet" lanes: rows are normal ŷ, friction x̂, ẑ
                let f = g.corner / CORNERS_PER_FOOT;
                self.foot_grf[f][0] += self.cs.lambda[nrow + 1];
                self.foot_grf[f][1] += self.cs.lambda[nrow];
                self.foot_grf[f][2] += self.cs.lambda[nrow + 2];
            }
            // env v34: joint-limit-row λ this substep, postural scope
            // ([legs, spine]; arms skipped). λ is already ≥ 0 (PGS
            // unilateral clamp) — max() is belt-and-suspenders against
            // a future solver change.
            for (j, lam) in self.cs.limit_row_lambdas() {
                let l = lam.max(0.0);
                if j < 3 {
                    self.limit_imp[1] += l;
                } else if j >= 17 {
                    self.limit_imp[0] += l;
                } else {
                    // arms — telemetry always; priced only when
                    // Core::limit_arms widens the scope (env v35)
                    self.limit_imp[2] += l;
                }
            }
            for (i, c) in self.cs.contacts.iter().enumerate() {
                let lam = self.cs.lambda[ncbase + i];
                self.contact_imp[c.pair] += lam;
                self.contact_pt[c.pair] = c.point;
                let (ia, ib) = PAIRS[c.pair];
                for id in [ia, ib] {
                    match id {
                        2 | 4 | 6 => arm_now[0] = true,
                        3 | 5 | 7 => arm_now[1] = true,
                        _ => {}
                    }
                }
            }
        }
        // contact bits: OR over the last 30 Hz tick of real time (the
        // last rate.ratio sim ticks; exactly this tick at 30 Hz)
        for s in 0..2 {
            self.contact_arm_age[s] = if arm_now[s] {
                0
            } else {
                self.contact_arm_age[s].saturating_add(1)
            };
            self.contact_arm[s] = self.contact_arm_age[s] < self.rate.ratio;
        }
        self.escaped = self.ws.env_hits != env_hits0;
        // contact spec §4 (per-TICK since 2026-08-20): root below the
        // floor threshold at the end of the tick
        if self.st.p[1] < FALL_ROOT_Y {
            fell = true;
        }
        self.fell = fell;
        if fell {
            self.fall_hits += 1;
        }
        // TILT envelope (2026-08-20): ONCE per tick on the end-of-tick
        // state (substep-rate checking is cost without benefit — a
        // tilt is not a transient the way a floor dip is). Measured on
        // the CHEST (user 2026-08-20, v30 night-of-nights replay: the
        // pelvis-only check was gamed — pelvis surfed 28 deg while the
        // spine pinned its -20 deg limit for a sustained ~70 deg
        // backbend). Chest up-axis = R_root * Rx(spine_pitch) *
        // Ry(spine_yaw) * Rz(spine_roll) * yhat (§1.3 spine stack);
        // spine-local up composed first, then rotated by the root quat.
        let (sp, sy, sr) = (self.st.q[0], self.st.q[1], self.st.q[2]);
        let (cp, s_p) = (sp.cos(), sp.sin());
        let (cy, s_y) = (sy.cos(), sy.sin());
        let (cr, s_r) = (sr.cos(), sr.sin());
        let v = [
            -cy * s_r,
            cp * cr - s_p * s_y * s_r,
            s_p * cr + cp * s_y * s_r,
        ];
        let r = quat_to_rot(self.st.quat);
        let upx = r[0][0] * v[0] + r[0][1] * v[1] + r[0][2] * v[2];
        let upy = r[1][0] * v[0] + r[1][1] * v[1] + r[1][2] * v[2];
        let upz = r[2][0] * v[0] + r[2][1] * v[1] + r[2][2] * v[2];
        let sag2 = if upz > 0.0 { TILT_TAN2_FWD } else { TILT_TAN2_BACK };
        if upy <= 0.0
            || upz * upz > sag2 * upy * upy
            || upx * upx > TILT_TAN2_ROLL * upy * upy
        {
            tilted = true;
        }
        self.tilted = tilted;
        if tilted {
            self.tilt_hits += 1;
        }
        // FOOT-LIFT (2026-08-20): edge on the tick-summed ground
        // normal impulse per foot (ground_imp buckets
        // [L_heel, L_toe, R_heel, R_toe]); charged in step_inner.
        self.foot_lifts_tick = 0;
        for f in 0..2 {
            let down =
                self.ground_imp[f * 2] + self.ground_imp[f * 2 + 1] > 0.0;
            if !down && self.foot_down_prev[f] {
                self.foot_lifts_tick += 1;
                self.foot_lift_hits[f] += 1;
            }
            self.foot_down_prev[f] = down;
        }
        self.refresh_fk();
        for s in 0..2 {
            for k in 0..3 {
                self.wrist_vel[s][k] =
                    (self.wrist[s][k] - prev_wrist[s][k]) / self.rate.dt;
                // env v49: tip velocity — the judge's tip_move expression
                // tree (step_inner), / the runtime tick length; the .cu
                // transcribes it (at the fixed 30 Hz DT)
                self.tip_vel[s][k] = (self.wrist[s][k]
                    + (HANDLE_LEN + SABER_LEN) * self.blade[s][k]
                    - prev_wrist[s][k]
                    - (HANDLE_LEN + SABER_LEN) * prev_blade[s][k]) / self.rate.dt;
            }
        }
        for (i, (f, _)) in BODY_PTS.iter().enumerate() {
            for k in 0..3 {
                self.body_pt_vel[i][k] =
                    (self.ws.p[*f][k] - prev_bp[i][k]) / self.rate.dt;
            }
        }
    }

    /// Live broad-phase reach gate (SPEC-v10 §4).
    fn reach_z(&self) -> Real {
        self.shoulder_z[0].max(self.shoulder_z[1]) + REACH_PAST_SHOULDER
    }
}

struct Core {
    // tick budget in SIM ticks (the constructor's max_steps is in 30 Hz
    // ticks and is scaled by TickRate::ratio, so it keeps its real time)
    max_steps: u32,
    // env v8: action id -> signed torque fraction, per joint since
    // 2026-10-04 (joint_gear_rows: FINE_GEAR_MULT + per-joint extras)
    gear: [[Real; 7]; NQ],
    chart: Vec<Note>,
    tail_s: Real, // episode tail after the last hit_time (covers miss exit)
    min_speed: Real, // slowest object; (hit_time-t)*min_speed > reach_z is a
                    // conservative exact "beyond reach" bound for a sorted chart
    // state
    tick: u32,
    time: Real,
    first_pending: usize, // smallest index with status 0 (monotone per episode)
    body: Body,
    trackers: [Vec<Tracker>; 2],
    status: Vec<u8>, // 0 pending, 1 good, 2 bad/bomb-hit, 3 missed, 4 bomb passed
    // env v45 per-hand previous-note record (see OFF_PREV_L), indexed
    // by NOTE COLOUR: chart index of the last resolved note of that
    // colour + valid (false until one has resolved).
    prev_idx: [u32; 2],
    prev_valid: [bool; 2],
    good_cuts: u32,
    bad_cuts: u32,
    misses: u32,
    bombs_hit: u32,
    // contact spec §4: fall terminations in THIS episode (0 or 1 — the
    // fall ends the episode). Reported, never hidden: the trainer's
    // `[fall]` line reads it.
    falls: u32,
    // TILT terminations in this episode (2026-08-20) — `falls` pattern.
    tilts: u32,
    // env v18 constant fall pricing (2026-08-15, sole mode): flat
    // -fall_const terminal reward on a fall (see fell block).
    fall_const: Real,
    // env v20 (2026-08-16, user direction): per-tick penalty
    // foot_z_coef * (|lfoot_z - PLANE_Z0| + |rfoot_z - PLANE_Z0|) —
    // linear in each foot's world-z distance from the reference
    // ("standard starting") foot z. Targets the shimmy toward/away
    // from the note plane. 0.0 = off (the pre-v20 reward verbatim).
    foot_z_coef: Real,
    // env v21: the X twin — per-tick foot_x_coef * (|lfoot_x -
    // (-FOOT_X0)| + |rfoot_x - FOOT_X0|), each foot priced against its
    // OWN side's wide-stance x. 0.0 = off.
    foot_x_coef: Real,
    // env v22 (user direction 2026-08-16): the foot penalties become
    // HINGED — deviation inside this radius is free ("target AREAS"),
    // rent applies only to the excess. 0.0 = the v21 pure-linear
    // behavior verbatim.
    foot_hinge_r: Real,
    // env v22: potential-based delta shaping (Ng et al. 1999,
    // undiscounted variant — round trips telescope to exactly zero):
    // each tick, reward += coef * (prev_pot - pot) where pot = the
    // total hinged deviation of both feet on both axes. Sharpens
    // credit assignment onto the tick that moved a foot; the RENT
    // terms remain the anti-parking objective (delta shaping alone
    // would make a parked pose free). 0.0 = off.
    foot_shape_coef: Real,
    // env v43 (2026-09-14, user): per-tick ENERGY price on Body::tau_abs
    // (Σ|τ| in N·m; reward -= energy_coef × Σ|τ|). Replaced the v23
    // fraction price (iso_coef, retired): a joint pays in proportion to
    // its torque limit, so trunk/leg torque is dear and wrist torque
    // near-free — the human muscle-volume weighting. Velocity-blind:
    // force, not work. 0 = off. Calibration: Σ|τ| ≈ 400 N·m/tick on the
    // 2026-09-14 leaves, so 7.5e-6 matches the old 5e-4 fraction bill.
    energy_coef: Real,
    // env v34 (2026-08-21): per-tick price on the joint-limit-row
    // impulse the postural stops carried (Body::limit_imp, [legs,
    // spine] N·m·s). 0 = the pre-v34 reward verbatim. Proportional to
    // the CARRIED impulse, so a knee-locked stand (near-zero moment on
    // the stop) is near-free while a crouch parked on hip/knee stops
    // pays its full gravity moment.
    limit_coef: Real,
    // env v35: widen pricing scope to the 14 arm joints too
    limit_arms: bool,
    // env v37 (user, 2026-08-23): training-only credit-assignment aid —
    // a wall DESPAWNS on the tick its fee is charged (bomb-like: one
    // WALL_PEN per wall object, per-wall not per-edge; stacked walls
    // each charge). Removed walls leave obs immediately, so any
    // visible wall is an unpaid, still-avoidable threat. Default
    // false: bench/replays keep faithful persistent walls.
    wall_despawn: bool,
    // cumulative limit-rest telemetry until popped (pop_limit_rest,
    // fall_hits pattern): λ-flow sums [legs, spine] and the tick count
    // — accumulated REGARDLESS of limit_coef so stop-resting is
    // observable at coef 0.
    limit_rest_sum: [Real; 3],
    limit_rest_ticks: u64,
    // activation telemetry (2026-09-13, replaces the retired work
    // reading): Σ over ticks of [total, spine, arms, legs] commanded
    // |gear fraction| + the tick count, until popped (pop_activation;
    // limit_rest pattern). Total was the v23 iso-priced quantity.
    act_sum: [Real; 4],
    act_ticks: u64,
    // env v43 torque telemetry: Σ over ticks of [total, spine, arms,
    // legs] applied |τ| (N·m) + the tick count, until popped
    // (pop_torque; act_sum pattern). Total is the energy-priced quantity.
    tau_sum: [Real; 4],
    tau_ticks: u64,
    // exp/wrist-tendon telemetry: Σ of Body::tendon_tick over ticks
    // ([|τ_p| L, R, zone L, R, out L, R, P_diss L, R]) + the SUBSTEP
    // count, until popped (pop_tendon; tau_sum pattern). Means are per
    // substep.
    tendon_sum: [Real; 8],
    tendon_n: u64,
    // exp/shoulder-rom telemetry: Σ of Body::srom_tick over ticks + the
    // SUBSTEP count, until popped (pop_shoulder; tendon_sum pattern)
    srom_sum: [Real; 8],
    srom_n: u64,
    // env v41 (2026-09-12, user): self-contact penalty — price on the
    // tick-summed self-collision row impulses (Body::contact_imp over
    // all 55 pairs, N·s). 0 = the pre-v41 reward verbatim. CUDA
    // self_contact_coef mirror.
    self_contact_coef: Real,
    // cumulative self-contact telemetry until popped
    // (pop_self_contact): [impulse sum, ticks with contact, ticks] —
    // accumulated REGARDLESS of the coefficient.
    sc_stat: [Real; 3],
    // the previous tick's pot (reset to 0 — resets start on-target)
    prev_foot_pot: Real,
    n_scoreable: u32, // chart objects that are not bombs
    // env v19 flail telemetry (CUDA-backend mirror, telemetry only):
    // blade arc (rad) accrued on note-free ticks (next pending chart
    // object > FLAIL_LEAD_S out) by hands with no live follow-through
    // tracker, plus the count of those ticks. PER-EPISODE here (zeroed
    // by reset_state, read per level by the bench); the CUDA twin is
    // pop-based instead — each matches its consumer.
    flail_rad: Real,
    flail_ticks: u32,
    // wrist-bend reversal telemetry (2026-09-22): per hand [L, R] count
    // THIS episode, running extremes [lo, hi], detector state (0 fresh, 3
    // armed, 1 moving +, 2 moving -). Zeroed by reset_state.
    wrev: [u32; 2],
    wrev_lo: [Real; 2],
    wrev_hi: [Real; 2],
    wrev_dir: [u8; 2],
    // wrist-reset telemetry (2026-10-04): per hand [L, R], THIS episode
    // (zeroed by reset_state); the CUDA twin keeps the same per episode
    // (ep_rec cols 40..43) plus a popped per-update accumulator.
    wrs: [WristReset; 2],
    // env v17 (spec §12): obstacles, sorted by start_time; wall_passed
    // latches once the back face clears the head (monotone per wall);
    // first_wall advances over the passed prefix; walls_in counts ticks
    // with the head inside any wall this episode.
    walls: Vec<Wall>,
    wall_passed: Vec<bool>,
    first_wall: usize,
    walls_in: u32,
    // spec §12b combo rule (user, 2026-08-13): ENTERING a wall is a
    // combo break (event 8); STAYING inside drains (event 7 per tick)
    // but does not re-break; the head must be OUT for >= 1 tick before
    // the next entry breaks again — i.e. break on the rising edge of
    // the in-wall state. exp/tick-rate: "out for >= 1 tick" is one 30 Hz
    // tick of REAL time, so the edge re-arms after rate.ratio consecutive
    // sim ticks out; this counts them (saturating; u32::MAX = never in).
    // At 30 Hz `wall_out >= 1` is exactly the old `!was_in_wall`.
    wall_out: u32,
    wall_enters: u32,
    // B4 (2026-08-03): sticky "this episode is over" latch. `done` is
    // recomputed from state every tick and each of its disjuncts is
    // MONOTONE (first_pending only advances and, once !pending, the cut
    // loop is empty so no new trackers can appear; time and tick only
    // increase), so a finished env re-stepped without a chart (re)install
    // can only ever report done again — and it was paying a full 8-substep
    // physics tick to say so. Bench idle slots (empty chart, done on every
    // tick) were the whole cost of the "~free" claim in the docstring.
    // Set where done is computed true, CLEARED in reset_state (the single
    // funnel every chart install goes through).
    finished: bool,
    score: Real,       // summed points; max = 115*notes + 85*heads + 20*links
    score_pre: Real,
    score_acc: Real,
    score_post: Real,
    // reusable event buffer: cleared and refilled in place each tick.
    events_buf: Vec<(u32, u8, f32, f32, f32)>,
    // Flux mode (flux.rs): fixed at construction — the chart holds
    // circles/bars, the hands are palms (no sabers), and the obs hand /
    // entity blocks switch to the Flux lane meanings.
    flux: bool,
    // Flux: palm-centre finite-difference velocity across the last tick
    palm_vel: [[Real; 3]; 2],
    // replay recorder (VecBeatSim.set_record_env): one frame per sim
    // tick stepped through step_repeat; None = not recording
    rec: Option<Box<Recorder>>,
}

impl Core {
    #[cfg(test)]
    fn new_core(max_steps: u32, gears: u8) -> Self {
        Core::new_core_rate(max_steps, gears, TickRate::base())
    }

    fn new_core_rate(max_steps: u32, gears: u8, rate: TickRate) -> Self {
        Core {
            max_steps: max_steps.saturating_mul(rate.ratio),
            gear: joint_gear_rows(gears),
            chart: Vec::new(),
            tail_s: 0.5,
            min_speed: Real::INFINITY,
            tick: 0,
            time: 0.0,
            first_pending: 0,
            body: Body::with_rate(rate),
            trackers: [Vec::new(), Vec::new()],
            status: Vec::new(),
            prev_idx: [0, 0],
            prev_valid: [false, false],
            good_cuts: 0,
            bad_cuts: 0,
            misses: 0,
            bombs_hit: 0,
            falls: 0,
            tilts: 0,
            fall_const: 5.0,
            foot_z_coef: 0.0,
            foot_x_coef: 0.0,
            foot_hinge_r: 0.0,
            foot_shape_coef: 0.0,
            energy_coef: 0.0,
            limit_coef: 0.0,
            limit_arms: false,
            wall_despawn: false,
            limit_rest_sum: [0.0; 3],
            limit_rest_ticks: 0,
            act_sum: [0.0; 4],
            act_ticks: 0,
            tau_sum: [0.0; 4],
            tau_ticks: 0,
            tendon_sum: [0.0; 8],
            tendon_n: 0,
            srom_sum: [0.0; 8],
            srom_n: 0,
            self_contact_coef: 0.0,
            sc_stat: [0.0; 3],
            prev_foot_pot: 0.0,
            n_scoreable: 0,
            flail_rad: 0.0,
            flail_ticks: 0,
            wrev: [0, 0],
            wrev_lo: [0.0, 0.0],
            wrev_hi: [0.0, 0.0],
            wrev_dir: [0, 0],
            wrs: [WristReset::default(); 2],
            walls: Vec::new(),
            wall_passed: Vec::new(),
            first_wall: 0,
            walls_in: 0,
            wall_out: u32::MAX,
            wall_enters: 0,
            finished: false,
            score: 0.0,
            score_pre: 0.0,
            score_acc: 0.0,
            score_post: 0.0,
            events_buf: Vec::new(),
            flux: false,
            palm_vel: [[0.0; 3]; 2],
            rec: None,
        }
    }

    fn set_walls(&mut self, walls: Vec<WallTuple>) {
        self.walls = walls
            .into_iter()
            .map(|(t0, cx, cy, w, h, len, speed)| Wall {
                start_time: t0,
                cx,
                cy,
                half_w: w / 2.0,
                half_h: h / 2.0,
                len,
                speed,
            })
            .collect();
        assert!(
            self.walls
                .windows(2)
                .all(|w| w[0].start_time <= w[1].start_time),
            "walls must be sorted by start_time"
        );
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
                len: 0.0,
                sweet: 0.0,
            })
            .collect();
        self.finish_chart_install();
    }

    /// Shared tail of every chart install (standard and Flux): order
    /// check, slowest-object tail, state reset.
    fn finish_chart_install(&mut self) {
        // the scan cursors assume hit_time order (chart.py/beatsaver.py
        // both sort); an unsorted chart would silently change behavior
        assert!(
            self.chart.windows(2).all(|w| w[0].hit_time <= w[1].hit_time),
            "chart must be sorted by hit_time"
        );
        // the episode tail must cover the slowest object's trip from the
        // plane to the WORST-CASE miss boundary (live head plane, env
        // v10), or final misses never register
        self.min_speed = chart_min_speed(&self.chart);
        self.tail_s = chart_tail_s(self.min_speed);
        self.reset_state();
    }

    fn reset_state(&mut self) {
        self.tick = 0;
        self.time = 0.0;
        self.first_pending = 0;
        self.body.reset();
        self.trackers = [Vec::new(), Vec::new()];
        self.status = vec![0; self.chart.len()];
        self.prev_idx = [0, 0];
        self.prev_valid = [false, false];
        self.wall_passed = vec![false; self.walls.len()];
        self.first_wall = 0;
        self.walls_in = 0;
        self.wall_out = u32::MAX;
        self.wall_enters = 0;
        self.good_cuts = 0;
        self.bad_cuts = 0;
        self.misses = 0;
        self.bombs_hit = 0;
        self.falls = 0;
        self.tilts = 0;
        self.n_scoreable = self.chart.iter().filter(|n| n.kind != KIND_BOMB).count() as u32;
        self.flail_rad = 0.0;
        self.flail_ticks = 0;
        self.wrev = [0, 0];
        self.wrev_dir = [0, 0];
        self.wrs = [WristReset::default(); 2];
        self.prev_foot_pot = 0.0;
        self.finished = false;  // B4: every chart (re)install funnels here
        self.score = 0.0;
        self.score_pre = 0.0;
        self.score_acc = 0.0;
        self.score_post = 0.0;
        self.palm_vel = [[0.0; 3]; 2];
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
        // hand derived blocks (v9 semantics on the v10 FK; Flux: palms)
        if self.flux {
            self.write_flux_hands(obs);
        } else {
        for (s, base) in [(0usize, OFF_HAND_L), (1usize, OFF_HAND_R)] {
            let b = &self.body;
            // env v38: wrist x/y join the entities' head-relative
            // frame so note<->saber aiming is a per-axis difference
            // inside one frame (lit: interacting tokens must share a
            // frame; dual-effector rules out saber anchors). z stays
            // world — entity z lanes are world z (z_off cancels), so
            // z was already frame-consistent. Velocities/blades stay
            // world (no near-constant shortcut in velocity; jitter).
            obs[base] = ((b.wrist[s][0] - b.head[0]) / X_NORM) as f32;
            obs[base + 1] = ((b.wrist[s][1] - b.head[1]) / Y_NORM) as f32;
            obs[base + 2] = b.wrist[s][2] as f32;
            for k in 0..3 {
                obs[base + 3 + k] = (b.wrist_vel[s][k] / V_NORM) as f32;
                obs[base + 6 + k] = b.blade[s][k] as f32;
            }
            obs[base + 9] = (b.swing_acc[s] / PRE_FULL).min(2.0) as f32;
            // env v19: remaining follow-through credit. Encoded so 0
            // means "swinging earns nothing now" (idle / broken /
            // capped all collapse there) and a fresh single-note
            // tracker reads 1.0. Summed in spawn order — the CUDA
            // ring walk from tr_ptr must add in this same order for
            // bitwise obs parity.
            let mut rem = 0.0 as Real;
            for tr in &self.trackers[s] {
                rem += 1.0 - (tr.acc / POST_FULL).min(1.0);
            }
            obs[base + 10] = rem.min(2.0) as f32;
        }
        }
        // head position (FK, centered on the reference pose)
        obs[OFF_HEAD] = (self.body.head[0] / X_NORM) as f32;
        obs[OFF_HEAD + 1] = ((self.body.head[1] - HEAD_Y0) / Y_NORM) as f32;
        obs[OFF_HEAD + 2] = ((self.body.head[2] - PLANE_Z0) / X_NORM) as f32;
        // contact bits [L, R]
        obs[OFF_CONTACT] = self.body.contact_arm[0] as u8 as f32;
        obs[OFF_CONTACT + 1] = self.body.contact_arm[1] as u8 as f32;
        // entity window (env v17, spec §12c/d): 16 SHARED slots, notes
        // and walls in one pool, filled nearest-first by 3D distance
        // from the head center. Entities behind the head consume no
        // slots (notes: past the §10b miss boundary; walls: back face
        // past the head). Head-anchoring makes every dz head-z-
        // invariant, so the sort cannot be gamed by walking (v15).
        if self.flux {
            self.write_flux_entities(obs);
        } else {
        let z_off = self.body.head[2] - PLANE_Z0;
        // 2026-08-22: z-lane low clamp = LIVE head z (exact existence
        // boundary; see the .cu mirror's note)
        let z_lo = self.body.head[2];
        let head = &self.body.head;
        // selection buffer, ascending by dist2; ties keep the FIRST-
        // scanned entity (strict-< displacement). Scan order: notes in
        // chart order, then walls — the CUDA mirror transcribes the
        // same comparison order (§5 parity).
        let mut sel = [(Real::INFINITY, false, 0usize); N_NOTE_SLOTS];
        let mut n_sel = 0usize;
        for i in self.first_pending..self.chart.len() {
            if self.status[i] != 0 {
                continue;
            }
            let n = &self.chart[i];
            // sorted-chart early exit: dz >= (hit−t)·min_speed − PLANE_Z0
            // for this and every later note, and dist >= dz — once that
            // lower bound alone beats the current worst kept distance,
            // no later note can enter the window
            let dz_lb = (n.hit_time - self.time) * self.min_speed - PLANE_Z0;
            if n_sel == N_NOTE_SLOTS && dz_lb > 0.0
                && dz_lb * dz_lb >= sel[N_NOTE_SLOTS - 1].0
            {
                break;
            }
            let dz = (n.hit_time - self.time) * n.speed - PLANE_Z0;
            if dz < 0.0 {
                continue; // behind the head (== past the miss boundary)
            }
            let dx = n.x - head[0];
            let dy = n.y - head[1];
            slot_insert(&mut sel, &mut n_sel, (dx * dx + dy * dy + dz * dz, false, i));
        }
        for wi in self.first_wall..self.walls.len() {
            if self.wall_passed[wi] {
                continue;
            }
            let w = &self.walls[wi];
            let dzf = (w.start_time - self.time) * w.speed - PLANE_Z0 - WALL_FRONT_OFF;
            if dzf + w.len < 0.0 {
                continue; // back face behind the head (flag lags a tick)
            }
            // clamp-based nearest point on the AABB: per-axis separation
            // (0 inside; the middle z term is negative whenever the head
            // is within the span, so .max(0.0) closes the interior case)
            let sx = ((head[0] - w.cx).abs() - w.half_w).max(0.0);
            let sy = ((head[1] - w.cy).abs() - w.half_h).max(0.0);
            let sz = dzf.max(-(dzf + w.len)).max(0.0);
            slot_insert(&mut sel, &mut n_sel, (sx * sx + sy * sy + sz * sz, true, wi));
        }
        for slot in 0..n_sel {
            let (_, is_wall, idx) = sel[slot];
            let base = OFF_NOTES + slot * NOTE_FEAT;
            if is_wall {
                let w = &self.walls[idx];
                let zfu = (w.start_time - self.time) * w.speed - WALL_FRONT_OFF
                    + z_off;
                // env v38 (user): entity x/y are HEAD-RELATIVE —
                // kills the absolute-coordinate shortcut the near-
                // constant standing head height taught (posture-
                // invariance / duck-amnesia fix). Mechanics
                // untouched; head pose stays in obs; z was already
                // head-anchored via z_off.
                obs[base] = ((w.cx - self.body.head[0]) / X_NORM) as f32;
                obs[base + 1] =
                    ((w.cy - self.body.head[1]) / Y_NORM) as f32;
                obs[base + 2] = z_lane(zfu, z_lo) as f32; // env v48
                obs[base + 3] = (-w.speed / Z_SCALE) as f32;
                // lanes 4..13 (color/dir/kind/free/ex-valid) stay zero
                obs[base + 14] = 1.0; // is_wall
                obs[base + 15] = (2.0 * w.half_w / X_NORM) as f32;
                obs[base + 16] = (2.0 * w.half_h / Y_NORM) as f32;
                // env v36: lane 17 = BACK-face z (wall end), encoded
                // like lane 2 (floored at the live head z) — exact for
                // any wall length. (Moved from lane 4 2026-08-22 so no
                // lane is binary-for-notes / continuous-for-walls
                // under the shared token encoder; lane 17 held wall
                // length until the same day.)
                obs[base + 17] = z_lane(zfu + w.len, z_lo) as f32; // env v48
            } else {
                let n = &self.chart[idx];
                let zu = (n.hit_time - self.time) * n.speed + z_off;
                // env v38: head-relative x/y (see wall branch note)
                obs[base] = ((n.x - self.body.head[0]) / X_NORM) as f32;
                obs[base + 1] =
                    ((n.y - self.body.head[1]) / Y_NORM) as f32;
                obs[base + 2] = z_lane(zu, z_lo) as f32; // env v48
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
                // lanes 13..17 stay zero. Lane 13 (in-token valid)
                // retired 2026-08-22: it exactly duplicated the
                // OFF_NOTE_MASK block, which gates attention — so it
                // read constant 1 on every token the policy could
                // see. The mask block stays.
            }
            obs[OFF_NOTE_MASK + slot] = 1.0;
        }
        }

        // contact spec §7: per-foot ground-contact lanes
        // [L_heel, L_toe, R_heel, R_toe]
        for k in 0..N_GROUND_IMP_LANES {
            // exp/tick-rate: the lanes are TICK-summed impulses, so scale
            // them to one 30 Hz tick's worth (× ratio; × 1.0 at 30 Hz)
            obs[OFF_GROUND_IMP + k] = (self.body.ground_imp[k]
                * self.body.rate.ratio_r / GROUND_IMP_NORM) as f32;
        }

        // contact spec §8 (env v13): elbow/knee/foot world positions —
        // FK the engines already ran, handed to the policy instead of
        // being re-derived from q through the network
        for (i, (f, y0)) in BODY_PTS.iter().enumerate() {
            let p = &self.body.ws.p[*f];
            let base = OFF_BODY_PTS + 3 * i;
            obs[base] = (p[0] / X_NORM) as f32;
            obs[base + 1] = ((p[1] - y0) / Y_NORM) as f32;
            obs[base + 2] = ((p[2] - PLANE_Z0) / X_NORM) as f32;
        }
        // env v24: the same points' world velocities (finite diff
        // across the last tick, wrist_vel-style, /V_NORM; zero at
        // reset and after a set_state refresh)
        for i in 0..N_BODY_PTS {
            let base = OFF_BODY_PT_VEL + 3 * i;
            for k in 0..3 {
                obs[base + k] =
                    (self.body.body_pt_vel[i][k] / V_NORM) as f32;
            }
        }
        // env v45: per-hand previous-note blocks (lane list at
        // OFF_PREV_L), indexed by note colour. Zero until a note of
        // that colour has resolved. x/y are recomputed against the
        // LIVE head (entity-lane rule). The CUDA write_obs transcribes
        // this loop.
        for (s, base) in [(0usize, OFF_PREV_L), (1usize, OFF_PREV_R)] {
            if !self.prev_valid[s] {
                continue;
            }
            let n = &self.chart[self.prev_idx[s] as usize];
            obs[base] = ((n.x - self.body.head[0]) / X_NORM) as f32;
            obs[base + 1] = ((n.y - self.body.head[1]) / Y_NORM) as f32;
            let show_dir = !n.any_dir || n.kind == KIND_CHAIN_LINK;
            if show_dir {
                obs[base + 2] = n.cos_a as f32;
                obs[base + 3] = n.sin_a as f32;
            }
            obs[base + 4] = 1.0;
        }
        // env v46: torque-limit scale lanes, (s − 1) / TAU_RAND_NORM
        // (exactly 0 at s = 1.0); the CUDA writer transcribes this loop
        for a in 0..TAU_FEAT {
            obs[OFF_TAU + a] = ((self.body.tau_axis[a] - 1.0) / TAU_RAND_NORM) as f32;
        }
        // env v49: blade-tip world velocity [L xyz, R xyz] / V_NORM
        // (standard game only — the Flux hands have no blade, the lanes
        // stay at the fill's zero); the CUDA writer transcribes this loop
        if !self.flux {
            for s in 0..2 {
                for k in 0..3 {
                    obs[OFF_TIP_VEL + 3 * s + k] =
                        (self.body.tip_vel[s][k] / V_NORM) as f32;
                }
            }
        }
    }

    fn build_obs(&self) -> [f32; OBS_DIM] {
        let mut obs = [0.0f32; OBS_DIM];
        self.write_obs(&mut obs);
        obs
    }

    // event codes: 0 ok, 1 slow, 2 angle, 3 wrong_hand, 4 miss, 5 post
    // (follow-through resolution), 6 bomb hit, 7 head-in-wall tick (env
    // v17; note id = u32::MAX). The f32 is points awarded.
    // Events land in self.events_buf (cleared here), in the same order the
    // v9 sim produced them — order is the ComboWalk contract.
    fn step_inner(&mut self, actions: &[u8; NQ]) -> (Real, bool) {
        // B4 early-out (2026-08-03): the episode already ended and no
        // chart was installed since, so `done` is structurally pinned true
        // (see the `finished` field comment). Skip the whole tick —
        // physics, event scan, obs-affecting state — and report the same
        // (reward 0, done) pair with an EMPTY event buffer, so a bench
        // idle slot's ComboWalk cannot re-consume last tick's events.
        // Frozen state also means write_obs() (still called by the
        // vector step) keeps reproducing the terminal obs row verbatim,
        // and `tick` stops climbing so a long-idle slot can never wrap
        // into max_steps or overflow u32.
        if self.finished {
            self.events_buf.clear();
            return (0.0, true);
        }
        let prev_pos = self.body.wrist;
        let prev_dir = self.body.blade;

        let mut reward = 0.0;
        self.events_buf.clear();
        // per arm this tick: (swing delta, did the chain continue)
        let mut deltas = [0.0 as Real; 2];
        let mut continued = [false; 2];

        // ── the physics tick (SPEC-v10 §3/§4) ─────────────────────────
        let rate = self.body.rate;
        self.body.tick(actions, &self.gear);
        // env v43 energy price on the applied Σ|τ| (the v23 fraction
        // price retired 2026-09-14) + always-on activation and torque
        // telemetry (accumulate at coef 0 too)
        // (per-tick price × tscale: the same price per second at any rate)
        reward -= self.energy_coef * self.body.tau_abs * rate.tscale;
        self.act_sum[0] += self.body.act_frac;
        self.act_sum[1] += self.body.act_group[0];
        self.act_sum[2] += self.body.act_group[1];
        self.act_sum[3] += self.body.act_group[2];
        self.act_ticks += 1;
        self.tau_sum[0] += self.body.tau_abs;
        self.tau_sum[1] += self.body.tau_group[0];
        self.tau_sum[2] += self.body.tau_group[1];
        self.tau_sum[3] += self.body.tau_group[2];
        self.tau_ticks += 1;
        // exp/wrist-tendon telemetry (never priced)
        for k in 0..8 {
            self.tendon_sum[k] += self.body.tendon_tick[k];
        }
        self.tendon_n += rate.n_sub as u64;
        // exp/shoulder-rom telemetry (never priced)
        for k in 0..8 {
            self.srom_sum[k] += self.body.srom_tick[k];
        }
        self.srom_n += rate.n_sub as u64;
        // env v34: limit-rest penalty + always-on telemetry (see
        // Body::limit_imp for scope/units)
        reward -= self.limit_coef
            * (self.body.limit_imp[0] + self.body.limit_imp[1]
               + if self.limit_arms { self.body.limit_imp[2] } else { 0.0 });
        self.limit_rest_sum[0] += self.body.limit_imp[0];
        self.limit_rest_sum[1] += self.body.limit_imp[1];
        self.limit_rest_sum[2] += self.body.limit_imp[2];
        self.limit_rest_ticks += 1;
        // env v41: self-contact penalty + always-on telemetry. The
        // solver clamps contact rows to λ ≥ 0, so the pair sum is the
        // tick's total self-collision impulse (CUDA sc_imp mirror).
        let sc_imp: Real = self.body.contact_imp.iter().map(|l| l.max(0.0)).sum();
        reward -= self.self_contact_coef * sc_imp;
        self.sc_stat[0] += sc_imp;
        if sc_imp > 0.0 {
            self.sc_stat[1] += 1.0;
        }
        self.sc_stat[2] += 1.0;
        // FOOT-LIFT penalty (2026-08-20): flat charge per lift event
        // this tick (see Body::tick edge detection)
        reward += FOOT_LIFT_PEN * self.body.foot_lifts_tick as Real;
        // env v20/v21/v22 foot terms: ws.p holds this tick's FK (the
        // same source the §8 body_pts obs lanes read). Hinged rent
        // (free inside foot_hinge_r, the v21 pure-linear behavior at
        // r = 0) + optional potential-based delta shaping.
        if self.foot_z_coef > 0.0 || self.foot_x_coef > 0.0
            || self.foot_shape_coef > 0.0
        {
            let r = self.foot_hinge_r;
            let h = |v: Real| (v.abs() - r).max(0.0);
            let lf = &self.body.ws.p[F_L_FOOT];
            let rf = &self.body.ws.p[F_R_FOOT];
            let hx = h(lf[0] + FOOT_X0) + h(rf[0] - FOOT_X0);
            let hz = h(lf[2] - PLANE_Z0) + h(rf[2] - PLANE_Z0);
            // per-tick rent × tscale (same rent per second at any rate)
            reward -= (self.foot_x_coef * hx + self.foot_z_coef * hz) * rate.tscale;
            if self.foot_shape_coef > 0.0 {
                let pot = hx + hz;
                reward += self.foot_shape_coef * (self.prev_foot_pot - pot);
                self.prev_foot_pot = pot;
            }
        }

        // swing chain update (feeds pre-swing + follow-through scores);
        // Flux has no sabers, so no swing chain and no trackers
        if !self.flux {
        for s in 0..2 {
            let d_old = prev_dir[s];
            let d_new = self.body.blade[s];
            let dot = (d_old[0] * d_new[0] + d_old[1] * d_new[1]
                + d_old[2] * d_new[2])
                .clamp(-1.0, 1.0);
            let delta = dot.acos();
            deltas[s] = delta;
            if delta < SWING_BREAK * rate.tscale {
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
                    tr.acc >= POST_FULL || tr.age >= FOLLOW_TIMEOUT * rate.ratio
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
        }

        self.time = (self.tick + 1) as Real * rate.dt;

        // env v19 flail telemetry (CUDA mirror, telemetry only): on
        // ticks whose next pending chart object is > FLAIL_LEAD_S out,
        // blade arc from hands with no live follow-through tracker is
        // swinging that cannot earn. Runs BEFORE this tick's cuts,
        // exactly like the .cu twin.
        {
            let mut next_gap = 2.0 * FLAIL_LEAD_S;
            for i in self.first_pending..self.chart.len() {
                if self.status[i] != 0 {
                    continue;
                }
                next_gap = self.chart[i].hit_time - self.time;
                break;
            }
            if next_gap > FLAIL_LEAD_S {
                for s in 0..2 {
                    if self.trackers[s].is_empty() {
                        self.flail_rad += deltas[s];
                    }
                }
                self.flail_ticks += 1;
            }
        }

        // wrist-bend reversal telemetry (2026-09-22): peak detection
        // with WREV_HYST hysteresis on this tick's integrated bend
        // angle, per hand. Telemetry only — the .cu twin's state
        // machine, step for step.
        for h in 0..2 {
            let q = self.body.st.q[if h == 0 { 8 } else { 15 }];
            let mut dir = self.wrev_dir[h];
            let (mut lo, mut hi) = (self.wrev_lo[h], self.wrev_hi[h]);
            if dir == 0 {
                lo = q;
                hi = q;
                dir = 3;
            } else if dir == 3 {
                // undirected: track both extremes; a WREV_HYST rise from
                // the low (or fall from the high) commits the first
                // direction — not a reversal
                if q < lo {
                    lo = q;
                }
                if q > hi {
                    hi = q;
                }
                if q >= lo + WREV_HYST {
                    dir = 1;
                    hi = q;
                } else if q <= hi - WREV_HYST {
                    dir = 2;
                    lo = q;
                }
            } else if dir == 1 {
                if q > hi {
                    hi = q;
                } else if q <= hi - WREV_HYST {
                    self.wrev[h] += 1;
                    dir = 2;
                    lo = q;
                }
            } else if q < lo {
                lo = q;
            } else if q >= lo + WREV_HYST {
                self.wrev[h] += 1;
                dir = 1;
                hi = q;
            }
            self.wrev_dir[h] = dir;
            self.wrev_lo[h] = lo;
            self.wrev_hi[h] = hi;
            // wrist-reset detector (30 deg), same slot
            self.wrs[h].tick(q);
        }

        if self.flux {
            self.flux_judge(&mut reward, &prev_pos, &prev_dir);
        } else {
        // both sabers' swept surfaces this tick
        let tris = [
            sweep_tris(prev_pos[0], prev_dir[0], self.body.wrist[0], self.body.blade[0]),
            sweep_tris(prev_pos[1], prev_dir[1], self.body.wrist[1], self.body.blade[1]),
        ];

        // live gates (SPEC-v10 §4)
        let reach_z = self.body.reach_z();
        let head_z = self.body.head[2];
        let head = self.body.head;
        let z_off = head_z - PLANE_Z0; // v15 head-anchored note track

        for i in self.first_pending..self.chart.len() {
            if self.status[i] != 0 {
                continue;
            }
            let n = &self.chart[i]; // by ref: Note is 80 B post-sin/cos
            // sorted by hit_time, and speed >= min_speed for every note:
            // once this bound clears reach_z, so does every later note's z
            // (v15: z_off is shared by every note, so the bound shifts
            // with it and the sorted-break argument is unchanged)
            if (n.hit_time - self.time) * self.min_speed + z_off > reach_z {
                break;
            }
            let z = (n.hit_time - self.time) * n.speed + z_off;
            if z < head_z {
                if n.kind == KIND_BOMB {
                    self.status[i] = 4; // passed clean: no penalty, no event
                } else {
                    self.status[i] = 3;
                    self.misses += 1;
                    reward += REWARD_MISS;
                    self.events_buf.push((i as u32, 4, 0.0, 0.0, 0.0));
                    // env v45: a miss records on the note's colour
                    let h = n.color as usize;
                    self.prev_idx[h] = i as u32;
                    self.prev_valid[h] = true;
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
                let mut d_seg = Real::INFINITY;
                let mut s_star = 0.0;
                for j in 1..=N_BLADE_SAMPLES {
                    // env v42: samples along the BLADE, [HANDLE_LEN, HANDLE_LEN + SABER_LEN]
                    let r = HANDLE_LEN + (j as Real / N_BLADE_SAMPLES as Real) * SABER_LEN;
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
                        - prev_pos[m][0] - s_star * prev_dir[m][0]) / rate.dt,
                    (wrist_m[1] + s_star * blade_m[1]
                        - prev_pos[m][1] - s_star * prev_dir[m][1]) / rate.dt,
                    (wrist_m[2] + s_star * blade_m[2]
                        - prev_pos[m][2] - s_star * prev_dir[m][2]) / rate.dt,
                ];
                let vc_mag = (vc[0].powi(2) + vc[1].powi(2) + vc[2].powi(2)).sqrt();
                // accuracy distance: note center to the INFINITE cutting
                // plane (blade dir x tip movement; stab fallback = blade
                // line) — the 2026-07-29 cut-plane rule, unchanged
                let tip_move = [
                    wrist_m[0] + (HANDLE_LEN + SABER_LEN) * blade_m[0]
                        - prev_pos[m][0] - (HANDLE_LEN + SABER_LEN) * prev_dir[m][0],
                    wrist_m[1] + (HANDLE_LEN + SABER_LEN) * blade_m[1]
                        - prev_pos[m][1] - (HANDLE_LEN + SABER_LEN) * prev_dir[m][1],
                    wrist_m[2] + (HANDLE_LEN + SABER_LEN) * blade_m[2]
                        - prev_pos[m][2] - (HANDLE_LEN + SABER_LEN) * prev_dir[m][2],
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
                // env v45: cut by the matching saber (good or bad) —
                // recorded on the note's colour (m == n.color)
                self.prev_idx[m] = i as u32;
                self.prev_valid[m] = true;
                if code == 0 {
                    self.status[i] = 1;
                    self.good_cuts += 1;
                    if n.kind != KIND_CHAIN_LINK {
                        // wrist-reset telemetry: this hand's bend velocity
                        let qd = self.body.st.v[6 + if m == 0 { 8 } else { 15 }];
                        self.wrs[m].hit(qd);
                    }
                    let (pts, pre_pts, acc_pts) = if n.kind == KIND_CHAIN_LINK {
                        (CHAIN_LINK_PTS, 0.0, 0.0) // flat, regardless of swing
                    } else {
                        let pre = if n.free_pre {
                            PTS_PRE
                        } else {
                            round_pts(PTS_PRE
                                * (self.body.swing_acc[m] / PRE_FULL).min(1.0))
                        };
                        let acc =
                            round_pts(PTS_ACC * (1.0 - (d_min / ACC_NORM).clamp(0.0, 1.0)));
                        let mut p = pre + acc;
                        if n.kind == KIND_CHAIN_HEAD {
                            // chain heads have no follow-through (max 85),
                            // except one that is also an arc head or tail
                            // (ChainHeadArcHead / ChainHeadArcTail): the
                            // 30 post points are a fixed grant (scoring v4,
                            // 2026-10-02; kernel mirror)
                            if n.free_pre || n.free_post {
                                p += PTS_POST;
                                self.score_post += PTS_POST;
                            }
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
                    // env v45: wrong-hand cut — still recorded on the
                    // NOTE'S colour (m), never on the saber that cut it
                    self.prev_idx[m] = i as u32;
                    self.prev_valid[m] = true;
                }
            }
        }
        }

        self.tick += 1;
        // advance the low-water cursor past the newly resolved prefix
        let head = self.body.head; // (bound inside the cut branch above too)
        // ── env v17 walls (spec §12b): head center vs AABB ────────────
        // Head-anchored like notes: dz_front = (t0 − t)·speed + 0.4 is
        // the front face's z relative to the HEAD ((−PLANE_Z0) − 0.25 =
        // 0.4), invariant to root z by construction. Inside iff the
        // head sits within the span and the x/y box. One penalty per
        // tick regardless of how many walls overlap the head.
        let mut in_wall = false;
        for wi in self.first_wall..self.walls.len() {
            if self.wall_passed[wi] {
                continue;
            }
            let w = &self.walls[wi];
            let dzf = (w.start_time - self.time) * w.speed - PLANE_Z0 - WALL_FRONT_OFF;
            // env v36 interval sweep — see the .cu mirror: charge if
            // the wall's head-plane window intersects this tick's
            // (time_old, time_new] span; sub-tick walls can't phase.
            let z_hit = dzf <= 0.0 && dzf + w.len + w.speed * rate.dt >= 0.0;
            if dzf + w.len < 0.0 {
                self.wall_passed[wi] = true; // back face behind the head
            }
            if z_hit
                && (head[0] - w.cx).abs() <= w.half_w
                && (head[1] - w.cy).abs() <= w.half_h
            {
                if self.wall_despawn {
                    // env v37: charge and remove THIS wall; no
                    // lingering in-wall ticks, no event 7 stream.
                    self.wall_enters += 1;
                    self.walls_in += 1; // counts entries in despawn mode
                    self.events_buf.push((u32::MAX, 8, 0.0, 0.0, 0.0));
                    reward += WALL_PEN;
                    self.wall_passed[wi] = true;
                } else {
                    in_wall = true;
                }
            }
        }
        if in_wall {
            if self.wall_out >= rate.ratio {
                // rising edge: COMBO BREAK (event 8). The scoring
                // layers (ComboWalk / ComboTracker / .dat walk) break
                // on 8 and IGNORE 7 — spec §12b combo rule. Since
                // 2026-08-20 WALL_PEN is also charged HERE, once per
                // entry, instead of every tick inside.
                self.wall_enters += 1;
                self.events_buf.push((u32::MAX, 8, 0.0, 0.0, 0.0));
                reward += WALL_PEN;
            }
            self.walls_in += 1;
            // event code 7: head-in-wall tick (note id unused -> u32::MAX)
            self.events_buf.push((u32::MAX, 7, 0.0, 0.0, 0.0));
        }
        self.wall_out = if in_wall { 0 } else { self.wall_out.saturating_add(1) };
        while self.first_wall < self.walls.len() && self.wall_passed[self.first_wall] {
            self.first_wall += 1;
        }

        advance_first_pending(&self.status, &mut self.first_pending);
        let last_hit = self.chart.last().map_or(0.0, |n| n.hit_time);
        let pending = self.first_pending < self.chart.len();
        let no_trackers = self.trackers[0].is_empty() && self.trackers[1].is_empty();
        // C backstop (SPEC-v10 Divergences §23): an env whose root left
        // the physical envelope during this tick terminates. A1 bounds
        // that trajectory but does not make it meaningful, and the
        // observations/rewards it produces are the actual harm. Folded
        // into `done` BEFORE the tracker resolution so the pending
        // follow-throughs are drained exactly as on any other terminal
        // tick; the reward is zeroed AFTER (this tick's physics reward
        // is garbage, so none of it — activation, cuts, follow-through —
        // reaches the policy).
        // env v11 FALL termination (unilateral-contact spec §4): in
        // contact mode a root below FALL_ROOT_Y ends the episode with a
        // ZEROED reward, folded in exactly like the C backstop above —
        // same place in the `done` disjunction (so the pending
        // follow-throughs still drain), same post-hoc reward zeroing.
        if self.body.fell {
            self.falls += 1;
        }
        if self.body.tilted {
            self.tilts += 1;
        }
        let done = chart_done(pending, no_trackers, self.chart.is_empty(),
                              self.time, last_hit, self.tail_s,
                              self.tick, self.max_steps)
            || self.body.escaped
            || self.body.fell
            || self.body.tilted;

        if done {
            self.finished = true;   // B4: latch; cleared only by reset_state
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
        // contact spec §7: the survival gradient — a per-tick alive
        // bonus. Added BEFORE the terminal overrides below so a
        // fall/escape tick never collects it.
        reward += ALIVE_BONUS * rate.tscale;
        if self.body.escaped {
            reward = 0.0;
        }
        if self.body.fell || self.body.tilted {
            // env v18 (2026-08-15, DEFAULT — legacy remaining-notes
            // pricing REMOVED): flat terminal reward; tilt terminations
            // (2026-08-20) priced identically. The discounted
            // remaining-notes charge priced sparse-state falls near
            // zero and taught fall-indifference exactly there (fpm
            // replay analysis); the 3-arm probe measured falls/update
            // 8-10k -> ~35 and bench zeros 35 -> 2 under the flat
            // price. Sized ~ the dense-case discounted charge so the
            // v11.1 too-harsh regime is never re-entered.
            reward = -self.fall_const;
        }

        (reward, done)
    }
}

/// Shared constructor + per-env telemetry pops, used by both
/// `RawBeatSim` (one Core) and `VecBeatSim` (folds over its Cores in
/// index order).
impl Core {
    /// `new_core` plus the per-env coefficients both Python
    /// constructors expose.
    #[allow(clippy::too_many_arguments)]
    fn configured(max_steps: u32, gears: u8,
                  fall_const: Real, foot_z_coef: Real, foot_x_coef: Real,
                  foot_hinge_r: Real, foot_shape_coef: Real,
                  energy_coef: Real, limit_coef: Real, limit_arms: bool,
                  wall_despawn: bool, self_contact_coef: Real,
                  rate: TickRate) -> Self {
        let mut core = Core::new_core_rate(max_steps, gears, rate);
        core.fall_const = fall_const;
        core.foot_z_coef = foot_z_coef;
        core.foot_x_coef = foot_x_coef;
        core.foot_hinge_r = foot_hinge_r;
        core.foot_shape_coef = foot_shape_coef;
        core.energy_coef = energy_coef;
        core.limit_coef = limit_coef;
        core.limit_arms = limit_arms;
        core.wall_despawn = wall_despawn;
        core.self_contact_coef = self_contact_coef;
        core
    }

    /// exp/tick-rate action repeat: `step_inner` with the same actions
    /// for up to `repeat` sim ticks, handing each tick's events to
    /// `on_events` in order. Rewards are summed; stops at the first done
    /// tick (further ticks would be latched no-ops). repeat = 1 is
    /// exactly one step_inner.
    /// A recording env (`rec`) also captures one frame per sim tick
    /// (latched no-op ticks of a finished env excluded).
    fn step_repeat(&mut self, actions: &[u8; NQ], repeat: u32,
                   mut on_events: impl FnMut(&[(u32, u8, f32, f32, f32)]))
                   -> (Real, bool) {
        let (mut reward, mut done) = self.step_recorded(actions);
        on_events(&self.events_buf);
        for _ in 1..repeat {
            if done {
                break;
            }
            let (r, d) = self.step_recorded(actions);
            on_events(&self.events_buf);
            reward += r;
            done = d;
        }
        (reward, done)
    }

    /// `step_inner`, plus the recorder frame when this env records.
    fn step_recorded(&mut self, actions: &[u8; NQ]) -> (Real, bool) {
        let tick0 = self.tick;
        let (r, d) = self.step_inner(actions);
        if self.tick != tick0 {
            if let Some(mut rec) = self.rec.take() {
                rec.push(self, actions, r);
                self.rec = Some(rec);
            }
        }
        (r, d)
    }

    /// Raw self-contact accumulators (impulse N·s, ticks with contact,
    /// env-ticks — the counts stored as Real), zeroed.
    fn pop_self_contact(&mut self) -> [Real; 3] {
        std::mem::replace(&mut self.sc_stat, [0.0; 3])
    }

    /// (Σ activation [total, spine, arms, legs], env-ticks), zeroed.
    fn pop_activation(&mut self) -> ([Real; 4], u64) {
        let s = std::mem::replace(&mut self.act_sum, [0.0; 4]);
        (s, std::mem::replace(&mut self.act_ticks, 0))
    }

    /// (Σ applied |τ| [total, spine, arms, legs], env-ticks), zeroed.
    fn pop_torque(&mut self) -> ([Real; 4], u64) {
        let s = std::mem::replace(&mut self.tau_sum, [0.0; 4]);
        (s, std::mem::replace(&mut self.tau_ticks, 0))
    }

    /// exp/wrist-tendon: (Σ over substeps of [|τ_p| L, |τ_p| R (N·m),
    /// zone L, zone R, out L, out R, P_diss L, P_diss R (W)], substeps),
    /// zeroed.
    fn pop_tendon(&mut self) -> ([Real; 8], u64) {
        let s = std::mem::replace(&mut self.tendon_sum, [0.0; 8]);
        (s, std::mem::replace(&mut self.tendon_n, 0))
    }

    /// exp/shoulder-rom: (Σ over substeps of [|τ_p| L, |τ_p| R (N·m, L1
    /// over yaw/pitch/twist), zone L, zone R (|u| > s), beyond L, beyond R
    /// (|u| > 1), P_diss L, P_diss R (W)], substeps), zeroed.
    fn pop_shoulder(&mut self) -> ([Real; 8], u64) {
        let s = std::mem::replace(&mut self.srom_sum, [0.0; 8]);
        (s, std::mem::replace(&mut self.srom_n, 0))
    }

    fn pop_pd_hits(&mut self) -> u64 {
        std::mem::replace(&mut self.body.ws.pd_hits, 0)
    }

    fn pop_envelope_hits(&mut self) -> u64 {
        std::mem::replace(&mut self.body.ws.env_hits, 0)
    }

    fn pop_fall_hits(&mut self) -> u64 {
        std::mem::replace(&mut self.body.fall_hits, 0)
    }

    #[cfg(feature = "stats")]
    fn pop_dbg_stats(&mut self) -> DbgStats {
        std::mem::take(&mut self.body.ws.stats)
    }
}

/// Replay v10 per-tick getters (SPEC-v10 §6) shared by RawBeatSim's
/// Python getters and the batched Recorder, so both read identical values.
impl Core {
    /// `q`: root position (3), root quaternion wxyz (4), 29 joint angles.
    fn rec_qpos(&self) -> [Real; 7 + NQ] {
        let st = &self.body.st;
        let mut out = [0.0 as Real; 7 + NQ];
        out[..3].copy_from_slice(&st.p);
        out[3..7].copy_from_slice(&st.quat);
        out[7..].copy_from_slice(&st.q);
        out
    }

    /// `twist`: root world-frame [vx, vy, vz, wx, wy, wz].
    fn rec_twist(&self) -> [Real; 6] {
        let v = &self.body.st.v;
        [v[0], v[1], v[2], v[3], v[4], v[5]]
    }

    /// `omega`: the 29 joint rates.
    fn rec_omega(&self) -> &[Real] {
        &self.body.st.v[6..]
    }

    /// `sabers`: per hand [L, R] FK wrist position + unit blade direction.
    fn rec_sabers(&self) -> [[Real; 6]; 2] {
        [0, 1].map(|s| {
            let w = self.body.wrist[s];
            let b = self.body.blade[s];
            [w[0], w[1], w[2], b[0], b[1], b[2]]
        })
    }

    /// `head`: FK head frame [px, py, pz, qw, qx, qy, qz].
    fn rec_head_pose(&self) -> [Real; 7] {
        let ws = &self.body.ws;
        let p = ws.p[F_HEAD];
        let q = dynamics::fk::rot_to_quat(&ws.r[F_HEAD]);
        [p[0], p[1], p[2], q[0], q[1], q[2], q[3]]
    }

    /// `contacts` of the last tick: (pair_id, px, py, pz, j) per pair
    /// with nonzero tick-summed normal impulse.
    fn rec_contacts(&self) -> impl Iterator<Item = (u32, Real, Real, Real, Real)> + '_ {
        let b = &self.body;
        (0..NUM_PAIRS)
            .filter(|&p| b.contact_imp[p] != 0.0)
            .map(|p| {
                let pt = b.contact_pt[p];
                (p as u32, pt[0], pt[1], pt[2], b.contact_imp[p])
            })
    }

    /// `activation`: last tick's [total, spine, arms, legs] |gear fraction|.
    fn rec_activation(&self) -> [Real; 4] {
        let b = &self.body;
        [b.act_frac, b.act_group[0], b.act_group[1], b.act_group[2]]
    }

    /// `torque`: last tick's [total, spine, arms, legs] applied |tau| N.m.
    fn rec_torque(&self) -> [Real; 4] {
        let b = &self.body;
        [b.tau_abs, b.tau_group[0], b.tau_group[1], b.tau_group[2]]
    }
}

// Recorder frame layout (lanes per tick, Real): q 36, twist 6, omega 29,
// sabers 2x6, head 7, feet (ground reaction) 2x3, activation 4, torque 4,
// reward 1. beat_arena_env/replay_writer.py mirrors it (REC_LANES checked).
const REC_Q: usize = 0;
const REC_TWIST: usize = REC_Q + 7 + NQ;
const REC_OMEGA: usize = REC_TWIST + 6;
const REC_SABERS: usize = REC_OMEGA + NQ;
const REC_HEAD: usize = REC_SABERS + 12;
const REC_FEET: usize = REC_HEAD + 7;
const REC_ACT: usize = REC_FEET + 6;
const REC_TORQUE: usize = REC_ACT + 4;
const REC_REWARD: usize = REC_TORQUE + 4;
const REC_LANES: usize = REC_REWARD + 1;
const _: () = assert!(REC_LANES == 105);

/// Fast replay generation (2026-10-02): a per-env columnar capture of the
/// replay v10 per-tick record, filled inside the batched step (one frame
/// per SIM tick, action-repeat ticks included) so a bench-style wave
/// records without a per-tick Python round trip. Values come from the
/// same Core getters RawBeatSim exposes; contacts and events are
/// variable-length, stored flat with per-tick prefix offsets.
#[derive(Default)]
struct Recorder {
    tick: Vec<u32>,           // the tick's index (Core::tick - 1 after it)
    time: Vec<Real>,          // raw Core::time after the tick
    frame: Vec<Real>,         // REC_LANES per tick
    action: Vec<u8>,          // NQ per tick
    contact_off: Vec<u32>,    // ticks + 1 prefix offsets into contact_*
    contact_pair: Vec<u32>,
    contact_val: Vec<Real>,   // px, py, pz, j per contact
    event_off: Vec<u32>,      // ticks + 1 prefix offsets into event_*
    event_note: Vec<u32>,
    event_code: Vec<u8>,
    event_val: Vec<f32>,      // points, pre, acc per event
}

impl Recorder {
    fn new() -> Recorder {
        Recorder { contact_off: vec![0], event_off: vec![0], ..Default::default() }
    }

    fn push(&mut self, core: &Core, actions: &[u8; NQ], reward: Real) {
        self.tick.push(core.tick - 1);
        self.time.push(core.time);
        let base = self.frame.len();
        self.frame.resize(base + REC_LANES, 0.0);
        let f = &mut self.frame[base..];
        f[REC_Q..REC_TWIST].copy_from_slice(&core.rec_qpos());
        f[REC_TWIST..REC_OMEGA].copy_from_slice(&core.rec_twist());
        f[REC_OMEGA..REC_SABERS].copy_from_slice(core.rec_omega());
        let sb = core.rec_sabers();
        f[REC_SABERS..REC_SABERS + 6].copy_from_slice(&sb[0]);
        f[REC_SABERS + 6..REC_HEAD].copy_from_slice(&sb[1]);
        f[REC_HEAD..REC_FEET].copy_from_slice(&core.rec_head_pose());
        let g = core.body.foot_grf;
        f[REC_FEET..REC_FEET + 3].copy_from_slice(&g[0]);
        f[REC_FEET + 3..REC_ACT].copy_from_slice(&g[1]);
        f[REC_ACT..REC_TORQUE].copy_from_slice(&core.rec_activation());
        f[REC_TORQUE..REC_REWARD].copy_from_slice(&core.rec_torque());
        f[REC_REWARD] = reward;
        self.action.extend_from_slice(actions);
        for (p, x, y, z, j) in core.rec_contacts() {
            self.contact_pair.push(p);
            self.contact_val.extend_from_slice(&[x, y, z, j]);
        }
        self.contact_off.push(self.contact_pair.len() as u32);
        for &(note, code, pts, pre, acc) in &core.events_buf {
            self.event_note.push(note);
            self.event_code.push(code);
            self.event_val.extend_from_slice(&[pts, pre, acc]);
        }
        self.event_off.push(self.event_note.len() as u32);
    }
}

#[pyclass]
struct RawBeatSim {
    core: Core,
}

#[pymethods]
impl RawBeatSim {
    /// The body stands on unilateral frictional ground contact (8 sole
    /// corners, unilateral-contact spec) — the only foot mode since the
    /// v10 weld harness was removed (2026-09-27; `foot_contact=` is no
    /// longer accepted).
    #[new]
    #[pyo3(signature = (max_steps=690, gears=3,
                        fall_const=5.0, foot_z_coef=0.0,
                        foot_x_coef=0.0, foot_hinge_r=0.0,
                        foot_shape_coef=0.0, energy_coef=0.0,
                        limit_coef=0.0, limit_arms=false,
                        wall_despawn=false, self_contact_coef=0.0,
                        flux=false, tick_hz=30))]
    #[allow(clippy::too_many_arguments)]
    fn new(max_steps: u32, gears: u8, fall_const: Real,
           foot_z_coef: Real, foot_x_coef: Real, foot_hinge_r: Real,
           foot_shape_coef: Real, energy_coef: Real, limit_coef: Real,
           limit_arms: bool, wall_despawn: bool,
           self_contact_coef: Real, flux: bool, tick_hz: u32) -> PyResult<Self> {
        let rate = TickRate::new(tick_hz)
            .map_err(pyo3::exceptions::PyValueError::new_err)?;
        let mut core = Core::configured(max_steps, gears,
                                        fall_const, foot_z_coef, foot_x_coef,
                                        foot_hinge_r, foot_shape_coef,
                                        energy_coef, limit_coef, limit_arms,
                                        wall_despawn, self_contact_coef, rate);
        core.flux = flux;
        Ok(RawBeatSim { core })
    }

    /// Control rate (Hz): the sim ticks at 1/tick_hz s with 240/tick_hz
    /// physics substeps per tick (exp/tick-rate; default 30). `max_steps`
    /// is given in 30 Hz ticks and scaled to the rate.
    #[getter]
    fn tick_hz(&self) -> u32 {
        self.core.body.rate.hz
    }
    /// Control tick length (s) = 1 / tick_hz.
    #[getter]
    fn dt(&self) -> Real {
        self.core.body.rate.dt
    }
    /// Physics substeps per control tick = 240 / tick_hz.
    #[getter]
    fn n_sub(&self) -> usize {
        self.core.body.rate.n_sub
    }
    /// The effective tick budget in SIM ticks (max_steps × tick_hz / 30).
    #[getter]
    fn max_steps(&self) -> u32 {
        self.core.max_steps
    }

    /// Flux mode: the sim was built with `flux=True`.
    #[getter]
    fn flux(&self) -> bool {
        self.core.flux
    }

    /// Flux chart (flux sims only): list of (hit_time, x, y, kind,
    /// color, rot_rad, speed, length, sweet) sorted by hit_time — kind 4
    /// circle / 5 bar (flux.rs). Resets state, returns the reset obs.
    fn set_flux_chart<'py>(
        &mut self,
        py: Python<'py>,
        chart: Vec<flux::FluxTuple>,
    ) -> Bound<'py, PyArray1<f32>> {
        self.core.set_flux_chart(chart);
        PyArray1::from_slice_bound(py, &self.core.build_obs())
    }

    /// Flux: the two palm-centre points [L, R] (wrist + PALM_OFF along
    /// the hand axis) — the circle contact points and Flux-line ends.
    fn palms(&self) -> [[Real; 3]; 2] {
        [0, 1].map(|s| flux::palm(self.core.body.wrist[s], self.core.body.blade[s]))
    }

    /// Shoulder z [L, R] (FK cache) — the reach gate's input; with
    /// palms() and head_pose() the full input of a Flux judge tick (the
    /// WebXR game parity fixture records all three).
    fn shoulder_z(&self) -> [Real; 2] {
        self.core.body.shoulder_z
    }

    /// The installed chart's rotation record, (sin_a, cos_a) per object —
    /// the WebXR game parity fixture pins the wasm build's install to it
    /// bit for bit.
    fn chart_trig(&self) -> Vec<(Real, Real)> {
        self.core.chart.iter().map(|n| (n.sin_a, n.cos_a)).collect()
    }

    /// env v41 self-contact telemetry since the last pop: (summed
    /// self-collision impulse N·s, ticks with contact, env-ticks). CUDA
    /// `pop_self_contact` mirror.
    fn pop_self_contact(&mut self) -> (Real, u64, u64) {
        let s = self.core.pop_self_contact();
        (s[0], s[1] as u64, s[2] as u64)
    }

    /// chart: list of (hit_time, x, y, color, angle_rad, speed, any_dir,
    /// kind, free_pre, free_post), sorted by hit_time; positions in
    /// meters (env v7). walls (env v17): list of (start_time, cx, cy,
    /// width, height, length, speed), sorted by start_time — installed
    /// BEFORE the chart so reset_state sizes wall_passed. Resets state,
    /// returns the reset observation.
    #[pyo3(signature = (chart, walls = Vec::new()))]
    fn set_chart<'py>(
        &mut self,
        py: Python<'py>,
        chart: Vec<ChartTuple>,
        walls: Vec<WallTuple>,
    ) -> Bound<'py, PyArray1<f32>> {
        assert!(!self.core.flux, "flux sim: use set_flux_chart");
        self.core.set_walls(walls);
        self.core.set_chart_notes(chart);
        PyArray1::from_slice_bound(py, &self.core.build_obs())
    }

    /// actions: 29 gear ids in SPEC-v10 §1.4 joint order — spine(3),
    /// arm_l(7), arm_r(7), leg_l(6), leg_r(6); id 0 = coast, 1/2 =
    /// +/-full torque, finer +/- gear pairs append (constants.gear_table).
    /// Returns (obs, reward, done, events); events = [(note_idx, code,
    /// points, pre_pts, acc_pts)] with codes 0 ok, 1 slow, 2 angle,
    /// 3 wrong_hand, 4 miss, 5 post, 6 bomb, 7 head-in-wall (id u32::MAX).
    fn step<'py>(
        &mut self,
        py: Python<'py>,
        actions: Vec<u8>,
    ) -> (Bound<'py, PyArray1<f32>>, Real, bool, Vec<(u32, u8, f32, f32, f32)>) {
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
    fn qpos(&self) -> Vec<Real> {
        self.core.rec_qpos().to_vec()
    }

    /// Replay v10 `twist`: root world-frame [vx, vy, vz, wx, wy, wz].
    fn twist(&self) -> [Real; 6] {
        self.core.rec_twist()
    }

    /// Replay v10 `omega`: the 29 joint rates (§1.4 order).
    fn omega(&self) -> Vec<Real> {
        self.core.rec_omega().to_vec()
    }

    /// Replay v10 `sabers`: per hand [L, R] the FK wrist position and
    /// unit blade direction — [wx, wy, wz, bx, by, bz].
    fn sabers(&self) -> [[Real; 6]; 2] {
        self.core.rec_sabers()
    }

    /// Replay v10 `head` (record_rev 2, ADDITIVE): the FK head frame's
    /// world pose — origin (3) then the world-from-body rotation as a
    /// scalar-first quaternion, [px, py, pz, qw, qx, qy, qz]. Same
    /// quaternion convention as `qpos`'s root block (wxyz), so the two
    /// read alike.
    ///
    /// WHY a getter and not "derive it from q": the head is WELDED to
    /// the chest (§1.2, no neck DoF), so its rotation is the spine
    /// pitch→yaw→roll stack applied to the root quaternion — cheap
    /// here off the live FK workspace, a full tree walk anywhere else.
    /// `sabers()` exists for exactly the same reason (§6 "viewer
    /// convenience"), and the .dat exporter needs the head pose the way
    /// it needs the wrists: watch.scoresaber.com drives its
    /// first-person camera straight off the recorded head frame
    /// (Previewer/Graphics/Camera/CameraUpdater.UpdateFirstPersonCamera).
    ///
    /// At the neutral stance this is [0, 1.70, -0.65, 1, 0, 0, 0]:
    /// world-aligned frames, head local +z = the gaze direction down
    /// the note track (the same local-+z-is-forward convention `blade`
    /// uses for the hands).
    fn head_pose(&self) -> [Real; 7] {
        self.core.rec_head_pose()
    }

    /// Blade unit directions [[x, y, z]; 2] (FK cache).
    fn blades(&self) -> [[Real; 3]; 2] {
        [0, 1].map(|s| self.core.body.blade[s])
    }

    /// Replay v10 `contacts` of the LAST tick: (pair_id, px, py, pz, j)
    /// per live pair with nonzero tick-summed normal impulse (N·s);
    /// the point is the last substep's world contact point.
    fn contacts(&self) -> Vec<(u32, Real, Real, Real, Real)> {
        self.core.rec_contacts().collect()
    }

    /// Replay fix (2026-08-17): tick-summed GROUND-CONTACT reaction
    /// impulse of the LAST tick per foot [L, R], [jx, jy, jz] (N·s,
    /// world frame) — jy sums the sole corners' normal rows, jx/jz their
    /// friction rows. Vertical reaction force ≈ jy / dt (the sim tick,
    /// 1 / tick_hz; tick-summed, so NOT rescaled to 30 Hz like the obs).
    fn ground_reaction(&self) -> [[Real; 3]; 2] {
        self.core.body.foot_grf
    }

    /// Per-note status: 0 pending, 1 good, 2 bad/bomb-hit, 3 missed,
    /// 4 bomb passed clean.
    fn status(&self) -> Vec<u8> {
        self.core.status.clone()
    }

    /// env v45 introspection: per colour/hand [L, R] the previous-note
    /// record (chart index, valid).
    fn prev_note(&self) -> [(u32, bool); 2] {
        [
            (self.core.prev_idx[0], self.core.prev_valid[0]),
            (self.core.prev_idx[1], self.core.prev_valid[1]),
        ]
    }

    /// TEST/TOOLING HOOK (exp/wrist-tendon): switch the passive wrist
    /// tendon off (False) / back on (True, the default) for this sim.
    /// For kinematic judge fixtures that pose a wrist deep in the tendon
    /// zone (bend ±90°) and want the pre-tendon passive wrist; training,
    /// the bench and replays always run the tendon (the CUDA kernel has
    /// no switch). Persists across resets.
    fn set_passive_wrist(&mut self, on: bool) {
        self.core.body.tendon_on = on;
    }

    /// TEST/TOOLING HOOK (exp/shoulder-rom): switch the passive shoulder
    /// axial limit (and its edge damping) off (False) / back on (True,
    /// the default). For fixtures that pose a shoulder beyond the human
    /// axial range; training, the bench and replays always run it (the
    /// CUDA kernel has no switch). Persists across resets.
    fn set_passive_shoulder(&mut self, on: bool) {
        self.core.body.srom_on = on;
    }

    /// TEST/TOOLING HOOK (env v10): overwrite the FULL generalized
    /// state — q: 36 position coords (root pos, root quat wxyz, 29
    /// joints), v: 35 velocity coords (root linear world, root angular
    /// world, 29 joint rates). FK caches refresh; the finite-diff wrist
    /// velocity resets; swing accumulators are left untouched.
    fn set_joints(&mut self, q: Vec<Real>, v: Vec<Real>) {
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
        body.tip_vel = [[0.0; 3]; 2];
        body.body_pt_vel = [[0.0; 3]; N_BODY_PTS];
        self.core.palm_vel = [[0.0; 3]; 2];
    }

    /// env v46: set the 16 per-axis torque-limit scales (spine 3, arm 7,
    /// leg 6; each arm/leg axis applies to BOTH sides). Persists across
    /// resets; all-1.0 = the v45 body (bit-identical rewards/dynamics).
    fn set_tau_scale(&mut self, s: Vec<Real>) {
        assert_eq!(s.len(), TAU_FEAT, "tau scale must have 16 axis values");
        let mut a = [1.0 as Real; TAU_FEAT];
        a.copy_from_slice(&s);
        self.core.body.set_tau_axis(&a);
    }

    /// env v46: the current 16 axis scales (replay header field).
    fn tau_scale(&self) -> Vec<Real> {
        self.core.body.tau_axis.to_vec()
    }

    /// Last tick's joint activation — commanded Σ|gear fraction| as
    /// (total, spine, arms, legs); the v23 fraction base (telemetry only
    /// since env v43).
    /// Telemetry (replay "activation" field, viewer meter, tests).
    fn last_activation(&self) -> (Real, Real, Real, Real) {
        let a = self.core.rec_activation();
        (a[0], a[1], a[2], a[3])
    }

    /// Activation telemetry since the last pop: (Σ total, Σ spine,
    /// Σ arms, Σ legs, env-ticks). CUDA `pop_activation` mirror.
    fn pop_activation(&mut self) -> (Real, Real, Real, Real, u64) {
        let (s, t) = self.core.pop_activation();
        (s[0], s[1], s[2], s[3], t)
    }

    /// Last tick's applied joint torque — Σ|τ_j| in N·m as (total,
    /// spine, arms, legs); total is the env v43 energy-priced quantity.
    /// Telemetry (replay "torque" field, tests).
    fn last_torque(&self) -> (Real, Real, Real, Real) {
        let t = self.core.rec_torque();
        (t[0], t[1], t[2], t[3])
    }

    /// Torque telemetry since the last pop: (Σ total, Σ spine, Σ arms,
    /// Σ legs, env-ticks) of applied |τ| (N·m). CUDA `pop_torque` mirror.
    fn pop_torque(&mut self) -> (Real, Real, Real, Real, u64) {
        let (s, t) = self.core.pop_torque();
        (s[0], s[1], s[2], s[3], t)
    }

    /// exp/wrist-tendon telemetry since the last pop: (Σ|τ_p| L, Σ|τ_p|
    /// R in N·m, zone substeps L, R, beyond-ellipse substeps L, R,
    /// substeps) — |τ_p| per hand = |τ_bend| + |τ_dev|, zone = r > 0.5,
    /// beyond = r > 1, then the edge-damping dissipated power Σ b_e·q̇²
    /// L, R (W, issue #61), every quantity summed per SUBSTEP — 9-tuple
    /// (…, diss L, diss R, substeps). CUDA `pop_tendon` mirror.
    fn pop_tendon(&mut self) -> (Real, Real, Real, Real, Real, Real, Real, Real, u64) {
        let (s, n) = self.core.pop_tendon();
        (s[0], s[1], s[2], s[3], s[4], s[5], s[6], s[7], n)
    }

    /// exp/shoulder-rom telemetry since the last pop: (Σ|τ_p| L, Σ|τ_p| R
    /// in N·m — L1 over the arm's yaw/pitch/twist rows, zone substeps L,
    /// R (|u| > 0.3), beyond-the-human-limit substeps L, R (|u| > 1),
    /// edge-damping dissipated power Σ b·τ̇² L, R (W), substeps) — every
    /// quantity summed per SUBSTEP. CUDA `pop_shoulder` mirror.
    fn pop_shoulder(&mut self) -> (Real, Real, Real, Real, Real, Real, Real, Real, u64) {
        let (s, n) = self.core.pop_shoulder();
        (s[0], s[1], s[2], s[3], s[4], s[5], s[6], s[7], n)
    }

    /// f64 Cholesky pivot-floor engagements since the last pop
    /// (SPEC-v10 Divergences §22) — the CUDA backend's pop_pd_hits
    /// mirror. Returns the count and zeroes it. ADDITIVE getter; 0 in
    /// every state the strict pre-§22 factorization survived.
    fn pop_pd_hits(&mut self) -> u64 {
        self.core.pop_pd_hits()
    }

    /// PRE-integration root-velocity-envelope engagements since the last
    /// pop (SPEC-v10 Divergences §23), counted per (env, substep) — the
    /// CUDA backend's pop_envelope_hits mirror. Returns the count and
    /// zeroes it. 0 on every healthy trajectory; nonzero means at least
    /// one tick's episode was terminated by the C backstop.
    fn pop_envelope_hits(&mut self) -> u64 {
        self.core.pop_envelope_hits()
    }

    /// FALL terminations since the last pop (unilateral-contact spec
    /// §4) — the CUDA backend's `pop_fall_hits` mirror, same shape as
    /// `pop_envelope_hits`. Returns the count and zeroes it.
    fn pop_fall_hits(&mut self) -> u64 {
        self.core.pop_fall_hits()
    }

    /// TILT terminations since the last pop (2026-08-20): root up-axis
    /// > 40 deg sagittal / > 40 deg lateral at end of tick, priced at
    /// the flat fall_const — the CUDA backend's `pop_tilt_hits` mirror.
    fn pop_tilt_hits(&mut self) -> u64 {
        let h = self.core.body.tilt_hits;
        self.core.body.tilt_hits = 0;
        h
    }

    /// Foot-lift events since the last pop (2026-08-20), (left,
    /// right) — a foot's tick-summed ground normal impulse hit zero
    /// after a grounded tick, each charged the flat FOOT_LIFT_PEN.
    /// The CUDA backend's `pop_foot_lifts` mirror.
    fn pop_foot_lifts(&mut self) -> (u64, u64) {
        let h = self.core.body.foot_lift_hits;
        self.core.body.foot_lift_hits = [0, 0];
        (h[0], h[1])
    }

    /// env v34 limit-rest telemetry since the last pop: (legs λ-flow
    /// N·m·s, spine λ-flow, env-ticks). CUDA `pop_limit_rest` mirror.
    fn pop_limit_rest(&mut self) -> (Real, Real, Real, u64) {
        let s = self.core.limit_rest_sum;
        let t = self.core.limit_rest_ticks;
        self.core.limit_rest_sum = [0.0; 3];
        self.core.limit_rest_ticks = 0;
        (s[0], s[1], s[2], t)
    }

    /// R26 per-substep instrumentation counters since the last pop —
    /// the single-env mirror of `VecBeatSim.pop_dbg_stats`. Returns the
    /// dict and zeroes the counters. In the DEFAULT build nothing counts
    /// and every value is 0 with `enabled` = 0 (see `dbg_stats_dict`).
    fn pop_dbg_stats(&mut self) -> HashMap<String, u64> {
        #[allow(unused_mut)]
        let mut acc = DbgStats::default();
        #[cfg(feature = "stats")]
        {
            acc.merge(&self.core.pop_dbg_stats());
        }
        dbg_stats_dict(&acc)
    }

    /// Did the PRE-integration envelope engage during the LAST tick
    /// (i.e. did the C backstop fire)? Test introspection.
    #[getter]
    fn escaped(&self) -> bool {
        self.core.body.escaped
    }

    /// Did the root drop below FALL_ROOT_Y during the LAST tick
    /// (unilateral-contact spec §4)? Test introspection, `escaped`'s
    /// twin.
    #[getter]
    fn fell(&self) -> bool {
        self.core.body.fell
    }

    /// Fall terminations in THIS episode (contact spec §4) — the `falls`
    /// counter the info dict reports. 0 or 1: a fall ends the episode.
    #[getter]
    fn falls(&self) -> u32 {
        self.core.falls
    }

    /// Tilt terminations in THIS episode (2026-08-20). 0 or 1: a tilt
    /// ends the episode.
    #[getter]
    fn tilts(&self) -> u32 {
        self.core.tilts
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
    /// env v17: ticks this episode with the head inside a wall
    /// (telemetry only since 2026-08-20 — WALL_PEN now lands once per
    /// wall_enters edge, not per tick). ADDITIVE getter (spec §12b).
    #[getter]
    fn walls_in(&self) -> u32 {
        self.core.walls_in
    }
    /// wrist-bend reversal telemetry, THIS episode: (L count, R count,
    /// ticks played) — 2026-09-22. ADDITIVE getter.
    fn wrist_rev(&self) -> (u32, u32, u32) {
        (self.core.wrev[0], self.core.wrev[1], self.core.tick)
    }
    /// wrist-reset telemetry, THIS episode (2026-10-04): (L resets, R
    /// resets, L swings, R swings) — see WristReset. ADDITIVE getter.
    fn wrist_resets(&self) -> (u32, u32, u32, u32) {
        let w = &self.core.wrs;
        (w[0].resets, w[1].resets, w[0].swings, w[1].swings)
    }
    /// env v17: wall ENTRIES this episode — combo breaks under the
    /// §12b combo rule (edge-triggered; >= 1 tick out re-arms).
    #[getter]
    fn wall_enters(&self) -> u32 {
        self.core.wall_enters
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
    fn time(&self) -> Real {
        self.core.time
    }
    /// Summed points across all scored cuts.
    #[getter]
    fn score(&self) -> Real {
        self.core.score
    }
    /// (pre_swing, accuracy, follow_through) point totals — the score
    /// decomposition; score − sum = chain-link flat points.
    fn score_components(&self) -> (Real, Real, Real) {
        (self.core.score_pre, self.core.score_acc, self.core.score_post)
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
    // per-env action repeat overriding the step call's `repeat`
    // (action_repeat_env; None = the call's value for every env)
    repeat_env: Option<Vec<u32>>,
}

#[pymethods]
impl VecBeatSim {
    /// Unilateral ground contact is the only foot mode (see RawBeatSim).
    ///
    /// Fast replay generation (2026-10-02): `tick_hz_env` (length n)
    /// gives each env its own control rate instead of `tick_hz`, and
    /// `action_repeat_env` (length n) its own action repeat, replacing
    /// the step call's `repeat` — so one vector step is one policy
    /// decision for every env, whatever run regime each replays.
    #[new]
    #[pyo3(signature = (n, max_steps=690,
                        combo_walk=false, return_events=true,
                        gears=3, fall_const=5.0,
                        foot_z_coef=0.0, foot_x_coef=0.0,
                        foot_hinge_r=0.0, foot_shape_coef=0.0,
                        energy_coef=0.0, limit_coef=0.0,
                        limit_arms=false, wall_despawn=false,
                        self_contact_coef=0.0, flux=false, tick_hz=30,
                        tick_hz_env=None, action_repeat_env=None))]
    #[allow(clippy::too_many_arguments)]
    fn new(n: usize, max_steps: u32, combo_walk: bool,
           return_events: bool, gears: u8,
           fall_const: Real, foot_z_coef: Real,
           foot_x_coef: Real, foot_hinge_r: Real,
           foot_shape_coef: Real, energy_coef: Real, limit_coef: Real,
           limit_arms: bool, wall_despawn: bool,
           self_contact_coef: Real, flux: bool, tick_hz: u32,
           tick_hz_env: Option<Vec<u32>>,
           action_repeat_env: Option<Vec<u32>>) -> PyResult<Self> {
        let hz = tick_hz_env.unwrap_or_else(|| vec![tick_hz; n]);
        if hz.len() != n {
            return Err(pyo3::exceptions::PyValueError::new_err(
                format!("tick_hz_env has {} rates for {n} envs", hz.len())));
        }
        if let Some(r) = &action_repeat_env {
            if r.len() != n || r.iter().any(|&x| x < 1) {
                return Err(pyo3::exceptions::PyValueError::new_err(
                    "action_repeat_env must hold n values >= 1"));
            }
        }
        let mut sims = Vec::with_capacity(n);
        for &h in &hz {
            let rate = TickRate::new(h)
                .map_err(pyo3::exceptions::PyValueError::new_err)?;
            let mut core = Core::configured(
                max_steps, gears, fall_const,
                foot_z_coef, foot_x_coef, foot_hinge_r,
                foot_shape_coef, energy_coef, limit_coef,
                limit_arms, wall_despawn, self_contact_coef, rate);
            core.flux = flux;
            sims.push(core);
        }
        Ok(VecBeatSim {
            sims,
            walks: combo_walk.then(|| (0..n).map(|_| ComboWalk::new(0)).collect()),
            return_events,
            repeat_env: action_repeat_env,
        })
    }

    /// Control rate (Hz) of every env (see RawBeatSim.tick_hz) — env 0's
    /// under tick_hz_env (see env_tick_hz).
    #[getter]
    fn tick_hz(&self) -> u32 {
        self.sims.first().map_or(TICK_HZ_BASE, |c| c.body.rate.hz)
    }

    /// Env i's control rate (Hz).
    fn env_tick_hz(&self, i: usize) -> u32 {
        self.sims[i].body.rate.hz
    }

    /// Start (`on`, discarding anything captured so far) or stop env i's
    /// replay recorder. A recording env captures one frame per sim tick
    /// it steps (action-repeat ticks included); see take_record_env.
    fn set_record_env(&mut self, i: usize, on: bool) {
        self.sims[i].rec = on.then(|| Box::new(Recorder::new()));
    }

    /// Env i's captured frames since set_record_env / the last take, as
    /// a dict of flat numpy arrays (the recorder keeps running, empty):
    /// `tick` u32 (T,), `time` f64 (T,) raw sim time after each tick,
    /// `frame` f64 (T * lanes,) — q 36, twist 6, omega 29, sabers 2x6,
    /// head 7, feet 2x3 (ground reaction), activation 4, torque 4,
    /// reward 1 —, `action` u8 (T * 29,), `contact_off` / `event_off`
    /// u32 (T + 1,) prefix offsets, `contact_pair` u32, `contact_val` f64
    /// (px, py, pz, j per contact), `event_note` u32, `event_code` u8,
    /// `event_val` f64 (points, pre, acc per event), `lanes` int.
    fn take_record_env<'py>(&mut self, py: Python<'py>, i: usize)
                            -> PyResult<Bound<'py, pyo3::types::PyDict>> {
        let rec = match self.sims[i].rec.as_mut() {
            Some(r) => std::mem::replace(r.as_mut(), Recorder::new()),
            None => return Err(pyo3::exceptions::PyRuntimeError::new_err(
                format!("env {i} is not recording (set_record_env)"))),
        };
        let f64v = |v: Vec<Real>| v.into_iter().map(|x| x as f64).collect::<Vec<f64>>();
        let d = pyo3::types::PyDict::new_bound(py);
        d.set_item("tick", PyArray1::from_vec_bound(py, rec.tick))?;
        d.set_item("time", PyArray1::from_vec_bound(py, f64v(rec.time)))?;
        d.set_item("frame", PyArray1::from_vec_bound(py, f64v(rec.frame)))?;
        d.set_item("action", PyArray1::from_vec_bound(py, rec.action))?;
        d.set_item("contact_off", PyArray1::from_vec_bound(py, rec.contact_off))?;
        d.set_item("contact_pair", PyArray1::from_vec_bound(py, rec.contact_pair))?;
        d.set_item("contact_val", PyArray1::from_vec_bound(py, f64v(rec.contact_val)))?;
        d.set_item("event_off", PyArray1::from_vec_bound(py, rec.event_off))?;
        d.set_item("event_note", PyArray1::from_vec_bound(py, rec.event_note))?;
        d.set_item("event_code", PyArray1::from_vec_bound(py, rec.event_code))?;
        let ev: Vec<f64> = rec.event_val.into_iter().map(|x| x as f64).collect();
        d.set_item("event_val", PyArray1::from_vec_bound(py, ev))?;
        d.set_item("lanes", REC_LANES)?;
        Ok(d)
    }

    #[pyo3(signature = (i, chart, walls = Vec::new()))]
    fn set_chart(&mut self, i: usize, chart: Vec<ChartTuple>, walls: Vec<WallTuple>) {
        assert!(!self.sims[i].flux, "flux sim: use set_flux_chart");
        let n_objects = chart.len();
        self.sims[i].set_walls(walls);
        self.sims[i].set_chart_notes(chart);
        if let Some(walks) = &mut self.walks {
            walks[i] = ComboWalk::new(n_objects);
        }
    }

    /// Flux chart for env i (flux sims only) — tuple layout as
    /// `RawBeatSim.set_flux_chart`.
    fn set_flux_chart(&mut self, i: usize, chart: Vec<flux::FluxTuple>) {
        let n_objects = chart.len();
        self.sims[i].set_flux_chart(chart);
        if let Some(walks) = &mut self.walks {
            walks[i] = ComboWalk::new(n_objects);
        }
    }

    /// (multiplied_score, max_combo, breaks) of env i's combo walk.
    fn combo(&self, i: usize) -> (Real, u32, u32) {
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
    ///
    /// `repeat` (exp/tick-rate action repeat): hold each env's action for
    /// this many sim ticks in the one call — rewards summed, done if any
    /// tick was done (a finished env stops stepping; the latch would make
    /// further ticks no-ops anyway), events concatenated in tick order,
    /// and the obs is the one after the LAST tick (the next decision's).
    /// 1 = the plain single-tick step. A sim built with
    /// `action_repeat_env` uses its per-env values instead.
    #[pyo3(signature = (actions, repeat = 1))]
    fn step<'py>(
        &mut self,
        py: Python<'py>,
        actions: PyReadonlyArray2<'py, i64>,
        repeat: u32,
    ) -> (
        Bound<'py, PyArray1<f32>>,
        Bound<'py, PyArray1<f32>>,
        Bound<'py, PyArray1<bool>>,
        Vec<(u32, u32, u8, f32, f32, f32)>,
    ) {
        let n = self.sims.len();
        let mut flat = vec![0.0f32; n * OBS_DIM];
        let (rewards, dones, all_events) =
            self.step_core(py, actions, &mut flat, repeat);
        (
            PyArray1::from_vec_bound(py, flat),
            PyArray1::from_vec_bound(py, rewards),
            PyArray1::from_vec_bound(py, dones),
            all_events,
        )
    }

    /// Like `step`, but writes the obs into a caller-owned C-contiguous
    /// (N, 340) f32 array instead of allocating a fresh buffer per
    /// step. Returns (rewards, dones, events). `repeat` as in `step`.
    #[pyo3(signature = (actions, out_obs, repeat = 1))]
    fn step_into<'py>(
        &mut self,
        py: Python<'py>,
        actions: PyReadonlyArray2<'py, i64>,
        mut out_obs: PyReadwriteArray2<'py, f32>,
        repeat: u32,
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
        let (rewards, dones, all_events) = self.step_core(py, actions, flat, repeat);
        (
            PyArray1::from_vec_bound(py, rewards),
            PyArray1::from_vec_bound(py, dones),
            all_events,
        )
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
    fn counters(&self, i: usize) -> (u32, u32, u32, u32, Real, u32) {
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
    fn components(&self, i: usize) -> (Real, Real, Real) {
        let s = &self.sims[i];
        (s.score_pre, s.score_acc, s.score_post)
    }

    /// env v19 flail telemetry for env i, THIS episode: (blade arc in
    /// rad accrued by follow-through-idle hands on note-free ticks,
    /// count of note-free ticks). Read at done, before set_chart
    /// resets it — same lifecycle as counters(). ADDITIVE getter.
    fn flail(&self, i: usize) -> (Real, u32) {
        let s = &self.sims[i];
        (s.flail_rad, s.flail_ticks)
    }

    /// wrist-bend reversal telemetry for env i, THIS episode: (L count,
    /// R count, ticks played). Same lifecycle as flail(). ADDITIVE.
    fn wrist_rev(&self, i: usize) -> (u32, u32, u32) {
        let s = &self.sims[i];
        (s.wrev[0], s.wrev[1], s.tick)
    }

    /// wrist-reset telemetry for env i, THIS episode (2026-10-04): (L
    /// resets, R resets, L swings, R swings). Same lifecycle as flail().
    fn wrist_resets(&self, i: usize) -> (u32, u32, u32, u32) {
        let w = &self.sims[i].wrs;
        (w[0].resets, w[1].resets, w[0].swings, w[1].swings)
    }

    /// Bench termination record for env i, THIS episode (review B09,
    /// 2026-09-30): (wall ENTRIES — the §12b combo-breaking edges, the
    /// wall_enters counter; termination flags). Flags, valid once the
    /// env is done (the B4 latch freezes the terminal tick's body
    /// state): 1 = FALL (root below FALL_ROOT_Y), 2 = TILT, 4 = the
    /// §23 root-envelope escape (C backstop), 8 = the max_steps cap;
    /// 0 = the level itself ended (notes drained / tail). The CUDA
    /// bench_rec cols 17/18 mirror this exactly. Same lifecycle as
    /// flail(). ADDITIVE getter.
    fn term(&self, i: usize) -> (u32, u32) {
        let s = &self.sims[i];
        let flags = (s.falls > 0) as u32
            | ((s.tilts > 0) as u32) << 1
            | (s.body.escaped as u32) << 2
            | ((s.tick >= s.max_steps) as u32) << 3;
        (s.wall_enters, flags)
    }

    /// f64 Cholesky pivot-floor engagements summed over all envs since
    /// the last pop (SPEC-v10 Divergences §22) — the CUDA backend's
    /// pop_pd_hits mirror. Returns the count and zeroes the per-env
    /// counters. ADDITIVE getter (no existing FFI shape changed); 0 in
    /// every state the strict pre-§22 factorization survived.
    fn pop_pd_hits(&mut self) -> u64 {
        self.sims.iter_mut().map(Core::pop_pd_hits).sum()
    }

    /// PRE-integration root-velocity-envelope engagements summed over
    /// all envs since the last pop (SPEC-v10 Divergences §23) — the
    /// CUDA backend's pop_envelope_hits mirror. Returns the count and
    /// zeroes the per-env counters. ADDITIVE getter; 0 on every healthy
    /// trajectory.
    fn pop_envelope_hits(&mut self) -> u64 {
        self.sims.iter_mut().map(Core::pop_envelope_hits).sum()
    }

    /// Activation telemetry summed over all envs since the last pop:
    /// (Σ total, Σ spine, Σ arms, Σ legs, env-ticks). CUDA
    /// `pop_activation` mirror.
    fn pop_activation(&mut self) -> (Real, Real, Real, Real, u64) {
        let mut acc = [0.0 as Real; 4];
        let mut tk = 0u64;
        for s in self.sims.iter_mut() {
            let (a, t) = s.pop_activation();
            for k in 0..4 {
                acc[k] += a[k];
            }
            tk += t;
        }
        (acc[0], acc[1], acc[2], acc[3], tk)
    }

    /// env v46: set the 16 per-axis torque-limit scales on EVERY env
    /// (bench / eval at a fixed body). RawBeatSim.set_tau_scale mirror.
    fn set_tau_scale(&mut self, s: Vec<Real>) {
        assert_eq!(s.len(), TAU_FEAT, "tau scale must have 16 axis values");
        let mut a = [1.0 as Real; TAU_FEAT];
        a.copy_from_slice(&s);
        for c in self.sims.iter_mut() {
            c.body.set_tau_axis(&a);
        }
    }

    /// env v46: set env i's 16 per-axis torque-limit scales — the
    /// per-episode --tau-rand draw of the Python wrapper (vec_env.py
    /// draws center * (1 + r U(-1,1)) at each env reset, as the CUDA
    /// kernel does). Like set_tau_scale, untouched by set_chart/reset;
    /// the obs tau lanes read it on the next obs.
    fn set_tau_scale_env(&mut self, i: usize, s: Vec<Real>) {
        assert_eq!(s.len(), TAU_FEAT, "tau scale must have 16 axis values");
        let mut a = [1.0 as Real; TAU_FEAT];
        a.copy_from_slice(&s);
        self.sims[i].body.set_tau_axis(&a);
    }

    /// Per-env torque telemetry since that env's last pop: (Σ total,
    /// Σ spine, Σ arms, Σ legs, ticks) of applied |τ| in N·m (one add
    /// per control tick), zeroed. The bench reads it when a level ends
    /// (before set_chart) and pops it again right after loading the next
    /// level, so each result carries exactly its own level's torque.
    /// CUDA bench_rec cols 19..22 mirror. ADDITIVE getter.
    fn pop_torque_env(&mut self, i: usize) -> (Real, Real, Real, Real, u64) {
        let (a, t) = self.sims[i].pop_torque();
        (a[0], a[1], a[2], a[3], t)
    }

    /// Torque telemetry (env v43) summed over all envs since the last
    /// pop: (Σ total, Σ spine, Σ arms, Σ legs, env-ticks) of applied
    /// |τ| in N·m. CUDA `pop_torque` mirror.
    fn pop_torque(&mut self) -> (Real, Real, Real, Real, u64) {
        let mut acc = [0.0 as Real; 4];
        let mut tk = 0u64;
        for s in self.sims.iter_mut() {
            let (a, t) = s.pop_torque();
            for k in 0..4 {
                acc[k] += a[k];
            }
            tk += t;
        }
        (acc[0], acc[1], acc[2], acc[3], tk)
    }

    /// exp/wrist-tendon telemetry summed over all envs since the last
    /// pop — RawBeatSim.pop_tendon's tuple. CUDA `pop_tendon` mirror.
    fn pop_tendon(&mut self) -> (Real, Real, Real, Real, Real, Real, Real, Real, u64) {
        let mut acc = [0.0 as Real; 8];
        let mut n = 0u64;
        for s in self.sims.iter_mut() {
            let (a, t) = s.pop_tendon();
            for k in 0..8 {
                acc[k] += a[k];
            }
            n += t;
        }
        (acc[0], acc[1], acc[2], acc[3], acc[4], acc[5], acc[6], acc[7], n)
    }

    /// exp/shoulder-rom telemetry summed over all envs since the last pop
    /// — RawBeatSim.pop_shoulder's tuple. CUDA `pop_shoulder` mirror.
    fn pop_shoulder(&mut self) -> (Real, Real, Real, Real, Real, Real, Real, Real, u64) {
        let mut acc = [0.0 as Real; 8];
        let mut n = 0u64;
        for s in self.sims.iter_mut() {
            let (a, t) = s.pop_shoulder();
            for k in 0..8 {
                acc[k] += a[k];
            }
            n += t;
        }
        (acc[0], acc[1], acc[2], acc[3], acc[4], acc[5], acc[6], acc[7], n)
    }

    /// env v41 self-contact telemetry summed over all envs since the
    /// last pop: (impulse N·s, ticks with contact, env-ticks). CUDA
    /// `pop_self_contact` mirror.
    fn pop_self_contact(&mut self) -> (Real, u64, u64) {
        let (mut imp, mut ct, mut tk) = (0.0 as Real, 0.0 as Real, 0.0 as Real);
        for s in self.sims.iter_mut() {
            let a = s.pop_self_contact();
            imp += a[0];
            ct += a[1];
            tk += a[2];
        }
        (imp, ct as u64, tk as u64)
    }

    /// FALL terminations summed over all envs since the last pop
    /// (unilateral-contact spec §4) — the CUDA backend's `pop_fall_hits`
    /// mirror, same shape as `pop_envelope_hits`. Returns the count and
    /// zeroes the per-env counters. ADDITIVE getter.
    fn pop_fall_hits(&mut self) -> u64 {
        self.sims.iter_mut().map(Core::pop_fall_hits).sum()
    }

    /// R26 instrumentation: the per-substep rate counters aggregated
    /// over ALL envs since the last pop, as a dict (sums add, maxes take
    /// the max, per-env counters are zeroed) — pop_pd_hits-style
    /// ADDITIVE getter, no existing FFI shape changed.
    ///
    /// Keys: nsubsteps, {nlim,ncon,cap_solves,s11_solves}_{sum,max},
    /// plus `enabled`. Divide a `_sum` by `nsubsteps` for the mean rate;
    /// `_max` is the worst single substep seen by any env.
    ///
    /// The counters only move in a build with `--features stats`. In the
    /// DEFAULT (shipping) build the counting code does not exist and
    /// this returns all zeros with `enabled` = 0 — so a collection
    /// script can call it unconditionally and just assert `enabled`.
    fn pop_dbg_stats(&mut self) -> HashMap<String, u64> {
        #[allow(unused_mut)]
        let mut acc = DbgStats::default();
        #[cfg(feature = "stats")]
        for s in self.sims.iter_mut() {
            acc.merge(&s.pop_dbg_stats());
        }
        dbg_stats_dict(&acc)
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
        repeat: u32,
    ) -> (Vec<f32>, Vec<bool>, Vec<(u32, u32, u8, f32, f32, f32)>) {
        assert!(repeat >= 1, "repeat must be >= 1");
        let acts = actions.as_array();
        let n = self.sims.len();
        assert_eq!(acts.shape(), &[n, NQ]);
        assert_eq!(flat.len(), n * OBS_DIM);
        let flat_acts: Vec<i64> = match acts.as_slice() {
            Some(s) => s.to_vec(),
            None => acts.iter().copied().collect(),
        };
        let ret_ev = self.return_events;
        let reps: Vec<u32> = match &self.repeat_env {
            Some(r) => r.clone(),
            None => vec![repeat; n],
        };
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
                    .zip_eq(reps.par_iter())
                    .map(|((((sim, mut walk), obs_chunk), a), &rep)| {
                        let mut acts_e = [0u8; NQ];
                        for j in 0..NQ {
                            acts_e[j] = a[j] as u8;
                        }
                        let mut ev = Vec::new();
                        let (r, d) = sim.step_repeat(&acts_e, rep, |events| {
                            if let Some(w) = walk.as_mut() {
                                for &(note, code, pts, _, _) in events {
                                    w.consume(note, code, pts);
                                }
                            }
                            if ret_ev {
                                ev.extend_from_slice(events);
                            }
                        });
                        sim.write_obs(obs_chunk);
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
#[cfg_attr(feature = "f32", pyo3(name = "beat_arena_rs32"))]
fn beat_arena_rs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<RawBeatSim>()?;
    m.add_class::<VecBeatSim>()?;
    Ok(())
}

#[cfg(all(test, not(feature = "f32")))]
mod mirror_tests;

#[cfg(all(test, not(feature = "f32")))]
mod tendon_tests;

#[cfg(all(test, not(feature = "f32")))]
mod shoulder_tests;

#[cfg(test)]
mod env_tests {
    use super::*;

    /// Activation telemetry wiring: Body::tick's act_frac is the
    /// commanded Σ|gear fraction| (the v23 fraction base, telemetry) and the
    /// [spine, arms, legs] groups partition it exactly; coasting reads
    /// 0 everywhere.
    #[test]
    fn activation_is_commanded_gear_fraction() {
        let rows = joint_gear_rows(5);
        let gear = gear_signed(5);
        let mut acts = [0u8; NQ];
        acts[0] = 3;  // spine pitch, +1/4
        acts[10] = 2; // r arm pitch, full −
        acts[15] = 1; // r wrist bend, full +
        acts[20] = 4; // l knee, −1/4
        let mut body = Body::new();
        body.tick(&acts, &rows);
        let expect: Real = acts.iter().map(|&a| gear[a as usize].abs()).sum();
        assert_eq!(body.act_frac, expect, "total is the joint-order sum");
        assert_eq!(body.act_group[0], gear[3].abs(), "spine");
        assert_eq!(body.act_group[1], gear[2].abs() + gear[1].abs(), "arms");
        assert_eq!(body.act_group[2], gear[4].abs(), "legs");
        assert_eq!(body.act_group.iter().sum::<Real>(), body.act_frac);
        // env v43 torque base: Σ|frac|·τ_max, same partition
        let tq: Real = (0..NQ)
            .map(|j| (gear[acts[j] as usize] * MODEL.tau_max[j]).abs())
            .sum();
        assert_eq!(body.tau_abs, tq, "torque total is the joint-order sum");
        assert_eq!(body.tau_group[0], (gear[3] * MODEL.tau_max[0]).abs());
        assert_eq!(body.tau_group[1],
                   (gear[2] * MODEL.tau_max[10]).abs()
                       + (gear[1] * MODEL.tau_max[15]).abs());
        assert_eq!(body.tau_group[2], (gear[4] * MODEL.tau_max[20]).abs());
        // a wrist full gear is priced 1.06/600 of a spine full gear
        let mut w = [0u8; NQ];
        w[7] = 1;
        let mut bw = Body::new();
        bw.tick(&w, &rows);
        let mut sp = [0u8; NQ];
        sp[0] = 1;
        let mut bs = Body::new();
        bs.tick(&sp, &rows);
        assert!((bw.tau_abs / bs.tau_abs - MODEL.tau_max[7] / MODEL.tau_max[0]).abs() < 1e-12);
        let mut b2 = Body::new();
        b2.tick(&[0u8; NQ], &rows);
        assert_eq!(b2.act_frac, 0.0, "coast must read zero activation");
        assert_eq!(b2.act_group, [0.0; 3]);
        assert_eq!(b2.tau_abs, 0.0, "coast must read zero torque");
    }

    /// Scripted wrist-reset sequences (2026-10-04, user spec). `bend`
    /// walks the bend angle from `from` to `to` (deg) in 5 deg ticks.
    fn bend(w: &mut WristReset, from: Real, to: Real) {
        let n = ((to - from).abs() / 5.0).ceil().max(1.0) as usize;
        for k in 1..=n {
            w.tick((from + (to - from) * k as Real / n as Real).to_radians());
        }
    }
    const DOWN: Real = -20.0; // bend velocity at a down cut, rad/s
    const UP: Real = 20.0;

    fn armed(start: Real) -> WristReset {
        let mut w = WristReset::default();
        w.tick(start.to_radians());
        w
    }

    #[test]
    fn wrist_reset_alternation_is_no_reset() {
        // down - up - down, each cut mid-swing: 3 swings, 0 resets
        let mut w = armed(60.0);
        bend(&mut w, 60.0, -60.0);
        assert_eq!(w.hit(DOWN), (false, true));
        bend(&mut w, -60.0, 60.0);
        assert_eq!(w.hit(UP), (false, true));
        bend(&mut w, 60.0, -60.0);
        assert_eq!(w.hit(DOWN), (false, true));
        assert_eq!((w.resets, w.swings), (0, 3));
    }

    #[test]
    fn wrist_reset_windback_is_one_reset() {
        // down, wind back 40 deg (past 30) without a hit, down again
        let mut w = armed(60.0);
        bend(&mut w, 60.0, -60.0);
        w.hit(DOWN);
        bend(&mut w, -60.0, -20.0);
        bend(&mut w, -20.0, -60.0);
        assert_eq!(w.hit(DOWN), (true, true));
        assert_eq!((w.resets, w.swings), (1, 2));
        // the 30 deg boundary: a 25 deg wind-back is no reversal (same
        // swing), a 35 deg one is (a reset)
        let mut w = armed(60.0);
        bend(&mut w, 60.0, -60.0);
        w.hit(DOWN);
        bend(&mut w, -60.0, -35.0);
        bend(&mut w, -35.0, -60.0);
        assert_eq!(w.hit(DOWN), (false, false));
        bend(&mut w, -60.0, -25.0);
        bend(&mut w, -25.0, -70.0);
        assert_eq!(w.hit(DOWN), (true, true));
        assert_eq!((w.resets, w.swings), (1, 2));
    }

    #[test]
    fn wrist_reset_stack_and_slider_are_one_swing() {
        // a stack (two hits on one tick) and a slider (hits along one
        // continuous motion): one swing, no reset
        let mut w = armed(80.0);
        bend(&mut w, 80.0, 0.0);
        assert_eq!(w.hit(DOWN), (false, true));
        assert_eq!(w.hit(DOWN), (false, false));
        for k in 0..4 {
            bend(&mut w, -10.0 * k as Real, -10.0 * (k + 1) as Real);
            assert_eq!(w.hit(DOWN), (false, false));
        }
        assert_eq!((w.resets, w.swings), (0, 1));
    }

    #[test]
    fn wrist_reset_tremor_is_no_reversal() {
        // +-12 deg (24 deg peak to peak < 30) tremor between two down hits
        let mut w = armed(60.0);
        bend(&mut w, 60.0, -40.0);
        w.hit(DOWN);
        for _ in 0..6 {
            bend(&mut w, -40.0, -16.0);
            bend(&mut w, -16.0, -40.0);
        }
        assert!(!w.rev && w.dir == 2);
        assert_eq!(w.hit(DOWN), (false, false));
        assert_eq!((w.resets, w.swings), (0, 1));
    }

    #[test]
    fn wrist_reset_long_break_counts_at_most_one() {
        // a long note-free stretch with several 60 deg back-and-forths:
        // (a) then a same-sign hit -> exactly one reset; (b) then an
        // opposite-sign hit -> none
        for (sign, want) in [(DOWN, (true, true)), (UP, (false, true))] {
            let mut w = armed(60.0);
            bend(&mut w, 60.0, -60.0);
            w.hit(DOWN);
            for _ in 0..5 {
                bend(&mut w, -60.0, 0.0);
                bend(&mut w, 0.0, -60.0);
            }
            if sign > 0.0 {
                bend(&mut w, -60.0, 0.0);
            }
            assert_eq!(w.hit(sign), want, "sign {sign}");
            assert_eq!(w.resets, want.0 as u32);
            assert_eq!(w.swings, 2);
        }
    }

    #[test]
    fn wrist_reset_hands_are_independent() {
        // a reset on one hand never moves the other's counts
        let mut w = [armed(60.0), armed(60.0)];
        bend(&mut w[0], 60.0, -60.0);
        bend(&mut w[1], 60.0, -60.0);
        w[0].hit(DOWN);
        w[1].hit(DOWN);
        bend(&mut w[0], -60.0, 0.0); // L winds back, R holds still
        bend(&mut w[0], 0.0, -60.0);
        for _ in 0..12 {
            w[1].tick((-60.0 as Real).to_radians());
        }
        assert_eq!(w[0].hit(DOWN), (true, true));
        assert_eq!(w[1].hit(DOWN), (false, false));
        assert_eq!((w[0].resets, w[0].swings, w[1].resets, w[1].swings),
                   (1, 2, 0, 1));
    }

    #[test]
    fn wrist_reset_near_zero_velocity_takes_the_committed_direction() {
        let mut w = armed(60.0);
        bend(&mut w, 60.0, -60.0); // committed: moving -
        assert_eq!(w.hit(0.4), (false, true)); // |qd| < 1: sign -
        assert_eq!(w.last, 2);
        bend(&mut w, -60.0, 0.0); // reversal: moving +
        assert_eq!(w.hit(-0.4), (false, true)); // committed +, opposite
        assert_eq!(w.last, 1);
        // no direction committed yet (armed): the velocity's own sign
        let mut f = armed(0.0);
        f.hit(-0.4);
        assert_eq!(f.last, 2);
    }

    /// Per-joint gear rows (2026-10-04): ids below `gears` are exactly
    /// the pre-2026-10-04 per-tick decode (gear_signed, fine ids x
    /// FINE_GEAR_MULT — bitwise), the next JOINT_EXTRA_N[j] ids are the
    /// joint's extra gears, everything after reads 0 (coast).
    #[test]
    fn joint_gear_rows_match_the_per_tick_decode() {
        for gears in [3u8, 5, 7] {
            let k = gears as usize;
            if (0..NQ).any(|j| k + JOINT_EXTRA_N[j] > 7) {
                continue; // this build's extras do not fit on that base
            }
            let g = gear_signed(gears);
            let rows = joint_gear_rows(gears);
            for j in 0..NQ {
                for a in 0..7 {
                    let want = if a < k {
                        if a >= 3 { g[a] * FINE_GEAR_MULT[j] } else { g[a] }
                    } else if a - k < JOINT_EXTRA_N[j] {
                        JOINT_EXTRA_FRAC[j][a - k]
                    } else {
                        0.0
                    };
                    assert_eq!(rows[j][a].to_bits(), want.to_bits(),
                               "gears {gears} joint {j} id {a}");
                }
                // extras come as +/- pairs
                for p in (0..JOINT_EXTRA_N[j]).step_by(2) {
                    assert_eq!(JOINT_EXTRA_FRAC[j][p], -JOINT_EXTRA_FRAC[j][p + 1]);
                }
            }
        }
    }

    /// B4: once an episode is done, re-stepping it WITHOUT a chart
    /// install runs no physics at all — the tick is a pure no-op that
    /// re-reports (0.0, true) with an empty event buffer, and a chart
    /// (re)install clears the latch so the env steps normally again.
    #[test]
    fn finished_env_early_outs_and_reinstall_clears_it() {
        let mut acts = [0u8; NQ];
        acts[10] = 1; // r arm pitch, full + — real torque every tick
        acts[3] = 2;

        // empty chart = the bench's idle slot: done on the FIRST tick
        let mut core = Core::new_core(100_000_000, 5);
        core.set_chart_notes(Vec::new());
        assert!(!core.finished, "install must clear the latch");
        let (_r0, d0) = core.step_inner(&acts);
        assert!(d0, "an empty chart is done on tick 1");
        assert!(core.finished, "done must latch");
        let q_frozen = core.body.st.q.clone();
        let v_frozen = core.body.st.v.clone();
        let obs_frozen = core.build_obs();
        let tick_frozen = core.tick;
        let time_frozen = core.time;

        for _ in 0..25 {
            let (r, d) = core.step_inner(&acts);
            assert_eq!(r, 0.0, "a finished env must earn exactly nothing");
            assert!(d, "a finished env stays done");
            assert!(core.events_buf.is_empty(),
                    "stale events would be re-consumed by the ComboWalk");
        }
        assert_eq!(core.body.st.q, q_frozen, "physics ran on a finished env");
        assert_eq!(core.body.st.v, v_frozen, "physics ran on a finished env");
        assert_eq!(core.tick, tick_frozen, "tick must not climb when idle");
        assert_eq!(core.time, time_frozen, "time must not climb when idle");
        assert_eq!(core.build_obs(), obs_frozen, "obs must stay the terminal row");

        // reinstall: the latch clears and the body moves again
        core.set_chart_notes(Vec::new());
        assert!(!core.finished);
        let (_r, d) = core.step_inner(&acts);
        assert!(d, "empty chart is done again after one real tick");
        assert_ne!(core.body.st.q, State::neutral(MODEL).q,
                   "the post-install tick must have run real physics");
    }

    /// B4 monotonicity, the property the early-out rests on: a natural
    /// (non-empty chart) episode reports done on exactly one tick and
    /// would keep reporting done forever — checked against the raw
    /// recomputation with the latch bypassed.
    #[test]
    fn done_is_monotone_on_a_real_chart() {
        // three notes, all far past reach: they resolve as misses and the
        // episode ends in the v10 tail
        let chart: Vec<ChartTuple> = (0u8..3)
            .map(|i| (1.0 + i as Real * 0.25, 0.0, 1.0, i % 2, 0.0, 6.0,
                      false, KIND_NOTE, false, false))
            .collect();
        let mut core = Core::new_core(690, 5);
        core.set_chart_notes(chart);
        let acts = [0u8; NQ];
        let mut first_done = None;
        for t in 0..690 {
            let (_r, d) = core.step_inner(&acts);
            if d {
                first_done = Some(t);
                break;
            }
        }
        let t_done = first_done.expect("episode never finished");
        assert!(t_done > 0, "a real chart must take more than zero ticks");
        // bypass the latch and recompute `done` from the frozen state the
        // way step_inner would have: every disjunct is still true
        core.finished = false;
        let (_r, d) = core.step_inner(&acts);
        assert!(d, "done must be structurally monotone");
    }

    /// Unilateral-contact spec §5 (d), updated 2026-08-20: in contact
    /// mode a hard lean (full +hip_pitch gears) now trips the TILT
    /// envelope (chest up-axis > 40 deg sagittal) BEFORE the root can
    /// drop below FALL_ROOT_Y — terminating the episode priced flat at
    /// -fall_const and COUNTED, the root-escape envelope's contract.
    /// The fall path shares the same pricing branch (fell || tilted),
    /// so the flat-pricing assert covers both.
    #[test]
    fn fall_terminates_with_zero_reward_and_is_counted() {
        // a real (non-empty) chart far in the future, so `done` cannot
        // come from the empty-chart disjunct. v18 (sole mode): the fall
        // step is priced FLAT at -fall_const (default 5.0), independent
        // of the pending chart — the 3 unjudged notes and the bomb are
        // scenery. (Pre-v18 this asserted REWARD_MISS x pending notes;
        // the assert was stale from 2026-08-13 to 2026-08-17.)
        let mut chart: Vec<ChartTuple> = (0u8..3)
            .map(|i| {
                (8.0 + i as Real * 0.25, 0.0, 1.0, i % 2, 0.0, 6.0, false,
                 KIND_NOTE, false, false)
            })
            .collect();
        chart.push((9.0, 0.0, 1.0, 0, 0.0, 6.0, false, KIND_BOMB,
                    false, false));
        let mut core = Core::new_core(690, 5);
        core.set_chart_notes(chart);
        let mut acts = [0u8; NQ];
        acts[17] = 1; // l_hip_pitch, full +
        acts[23] = 1; // r_hip_pitch, full +
        let mut ended_at = None;
        for t in 0..30 {
            // 1 s
            let (reward, done) = core.step_inner(&acts);
            if core.body.tilted || core.body.fell {
                assert!(done, "a tilted/fallen env must terminate");
                assert_eq!(
                    reward, -core.fall_const,
                    "the terminal step must be priced flat at \
                     -fall_const (v18 sole mode), chart-independent"
                );
                assert!(core.finished, "the terminal latch must be set");
                assert!(
                    core.body.tilted,
                    "a full hip-pitch lean must trip the TILT envelope \
                     before the root reaches FALL_ROOT_Y (root y {})",
                    core.body.st.p[1]
                );
                assert_eq!(core.tilts, 1, "the tilt must be counted in `tilts`");
                assert!(core.body.tilt_hits > 0, "the pop counter must see it");
                ended_at = Some(t);
                break;
            }
            assert!(!done, "tick {t} ended for some other reason");
        }
        let t = ended_at.expect("1 s of full hip flexion did not tip the body");
        println!("tilt: terminated at tick {t} with root y {:.4}", core.body.st.p[1]);
        // popping zeroes, pd_hits-style
        let popped = core.body.tilt_hits;
        core.body.tilt_hits = 0;
        assert!(popped > 0);
        assert_eq!(core.body.tilt_hits, 0);
    }

    /// Replay fix (2026-08-17): in contact mode the per-foot ground-
    /// reaction accumulator (the replay "feet" lanes' new source) must
    /// carry the body weight — the tick-summed normal impulses of both
    /// feet ≈ weight x DT while the body stands. Regression for the
    /// all-zero "feet" bug: the recorder read the retired welds'
    /// impulses in contact mode.
    #[test]
    fn ground_reaction_carries_weight_in_contact_mode() {
        let chart: Vec<ChartTuple> = vec![(8.0, 0.0, 1.0, 0, 0.0, 6.0,
                                           false, KIND_NOTE, false, false)];
        let mut core = Core::new_core(690, 5);
        core.set_chart_notes(chart);
        let acts = [0u8; NQ];
        // ticks 1-3: the first tick still carries the reset's settling
        // transient (the corners load up ~16 % under weight x DT);
        // by tick 3 the stance is in force balance and the zero-torque
        // topple (~0.7 s) has not unloaded it yet
        core.step_inner(&acts);
        core.step_inner(&acts);
        core.step_inner(&acts);
        let jy = core.body.foot_grf[0][1] + core.body.foot_grf[1][1];
        let weight_dt = 74.80 * dynamics::model::G / 30.0; // v63x2 body
        assert!(
            (jy - weight_dt).abs() < 0.15 * weight_dt,
            "both feet's normal impulse {jy:.3} N·s vs weight x DT \
             {weight_dt:.3} N·s"
        );
        // per-foot split is roughly even at the symmetric reset
        for f in 0..2 {
            assert!(
                core.body.foot_grf[f][1] > 0.25 * weight_dt,
                "foot {f} carries only {:.3} N·s",
                core.body.foot_grf[f][1]
            );
        }
    }

    // ---- exp/tick-rate: runtime control rate ---------------------------

    /// DT / N_SUB / scale derivation; the 30 Hz default is exactly the
    /// fixed-rate constants (bitwise), and only rates that are multiples
    /// of 30 dividing 240 are accepted.
    #[test]
    fn tick_rate_derivation() {
        let r30 = TickRate::new(30).unwrap();
        assert_eq!(r30.dt.to_bits(), DT_30.to_bits(), "30 Hz DT must be DT_30 bitwise");
        assert_eq!(r30.n_sub, N_SUB);
        assert_eq!((r30.ratio, r30.ratio_r, r30.tscale), (1, 1.0, 1.0));
        for (hz, n_sub, ratio, tscale) in [(60, 4, 2, 0.5), (120, 2, 4, 0.25), (240, 1, 8, 0.125)] {
            let r = TickRate::new(hz).unwrap();
            assert_eq!(r.n_sub, n_sub, "{hz} Hz");
            assert_eq!(r.ratio, ratio, "{hz} Hz");
            assert_eq!(r.ratio_r, ratio as Real, "{hz} Hz");
            assert_eq!(r.tscale, tscale, "{hz} Hz");
            assert_eq!(r.dt, 1.0 / hz as Real, "{hz} Hz");
            // the substep never changes: n_sub · H_SUB = DT
            assert!((r.n_sub as Real * H_SUB - r.dt).abs() < 1e-6, "{hz} Hz");
        }
        for bad in [0, 15, 45, 90, 480] {
            assert!(TickRate::new(bad).is_err(), "{bad} Hz must be rejected");
        }
    }

    /// Per-tick rules keep their real-time meaning: thresholds per tick
    /// × tscale, tick counts × ratio, the tick budget × ratio.
    #[test]
    fn per_tick_thresholds_scale_with_rate() {
        let r60 = TickRate::new(60).unwrap();
        assert_eq!(SWING_BREAK * r60.tscale, 0.0025);
        assert_eq!(FOLLOW_TIMEOUT * r60.ratio, 30);
        assert_eq!(ALIVE_BONUS * r60.tscale, 0.001);
        let r120 = TickRate::new(120).unwrap();
        assert_eq!(SWING_BREAK * r120.tscale, 0.00125);
        assert_eq!(FOLLOW_TIMEOUT * r120.ratio, 60);
        assert_eq!(Core::new_core(690, 5).max_steps, 690);
        assert_eq!(Core::new_core_rate(690, 5, r60).max_steps, 1380);
        assert_eq!(Core::new_core_rate(690, 5, r120).max_steps, 2760);
        assert_eq!(Core::new_core_rate(u32::MAX, 5, r120).max_steps, u32::MAX);
        // the alive bonus per second is rate-independent: 1 s of standing
        // pays the same at 30 and 60 Hz (far-future chart: nothing judged)
        let chart: Vec<ChartTuple> = vec![(100.0, 0.0, 1.0, 0, 0.0, 6.0,
                                           false, KIND_NOTE, false, false)];
        let mut tot = [0.0 as Real; 2];
        for (k, rate) in [TickRate::base(), r60].into_iter().enumerate() {
            let mut core = Core::new_core_rate(100_000, 5, rate);
            core.set_chart_notes(chart.clone());
            for _ in 0..(15 * rate.ratio) {
                let (r, d) = core.step_inner(&[0u8; NQ]);
                assert!(!d);
                tot[k] += r;
            }
            assert!((core.time - 0.5).abs() < 1e-9, "0.5 s elapsed at {} Hz", rate.hz);
        }
        assert!((tot[0] - tot[1]).abs() < 1e-12, "alive bonus / s: {tot:?}");
    }

    /// EQUAL STRENGTH: the same held action for 1 s is the same 240
    /// substeps with the same torques at every control rate — the body
    /// state at t = 1 s must match BITWISE (30 × 8 = 60 × 4 = 120 × 2
    /// substeps). Nothing in the actuation or the physics runs per tick:
    /// τ = gear · τ_max · tau_scale is recomputed each tick from the same
    /// inputs, limit rows / ω-cap / step-11 clamps / contacts / the root
    /// envelope all live inside `substep_constrained`, which carries no
    /// per-tick state (no warm start). Checked for several gear patterns
    /// incl. full-torque ones and a non-unit tau_scale body.
    #[test]
    fn equal_strength_across_tick_rates() {
        let gear = joint_gear_rows(5);
        let mut patterns: Vec<(&str, [u8; NQ], bool)> = Vec::new();
        patterns.push(("coast", [0u8; NQ], false));
        patterns.push(("all full +", [1u8; NQ], false));
        patterns.push(("all full -", [2u8; NQ], false));
        let mut mixed = [0u8; NQ];
        for j in 0..NQ {
            mixed[j] = [1u8, 2, 3, 4, 0][j % 5];
        }
        patterns.push(("mixed gears", mixed, false));
        let mut arms = [0u8; NQ];
        for j in 3..17 {
            arms[j] = if j % 2 == 0 { 1 } else { 2 }; // slams arm stops
        }
        patterns.push(("arms full (limit rows)", arms, false));
        patterns.push(("mixed, tau_scale != 1", mixed, true));
        let ts: [Real; TAU_FEAT] = [0.9012, 0.9804, 0.937, 0.9616, 0.8482, 0.8729,
                                    1.1, 1.1889, 1.0567, 1.022, 0.9055, 1.0453,
                                    1.0608, 1.0269, 1.0668, 1.1428];
        for (name, acts, scaled) in &patterns {
            let mut states = Vec::new();
            let mut imp = Vec::new();
            for hz in [30u32, 60, 120] {
                let rate = TickRate::new(hz).unwrap();
                let mut b = Body::with_rate(rate);
                if *scaled {
                    b.set_tau_axis(&ts);
                }
                b.reset();
                let mut env_hits = 0;
                let mut gimp = 0.0 as Real;
                for _ in 0..hz {
                    b.tick(acts, &gear);
                    env_hits += b.escaped as u32;
                    gimp += b.ground_imp.iter().sum::<Real>();
                }
                states.push((b.st.p, b.st.quat, b.st.q.clone(), b.st.v.clone(), b.ws.env_hits, env_hits));
                imp.push(gimp);
            }
            for k in 1..3 {
                let (p0, q0, j0, v0, e0, _) = &states[0];
                let (p, q, j, v, e, _) = &states[k];
                let bits = |x: &[Real]| x.iter().map(|v| v.to_bits()).collect::<Vec<_>>();
                assert_eq!(bits(p), bits(p0), "{name}: root pos");
                assert_eq!(bits(q), bits(q0), "{name}: root quat");
                assert_eq!(bits(j), bits(j0), "{name}: joints");
                assert_eq!(bits(v), bits(v0), "{name}: velocities");
                assert_eq!(e, e0, "{name}: envelope engagements");
                // tick-summed ground impulse over the second: the same
                // substep lambdas, only grouped differently (rounding)
                let rel = (imp[k] - imp[0]).abs() / imp[0].abs().max(1e-9);
                assert!(rel < 1e-4, "{name}: 1 s ground impulse {} vs {}", imp[k], imp[0]);
            }
            println!("equal strength ok: {name} (root y {:.4}, envelope hits {})",
                     states[0].0[1], states[0].4);
        }
    }

    /// The ground-impulse obs lanes are rescaled to one 30 Hz tick's
    /// worth: a standing body reads the same lanes at 30 / 60 / 120 Hz.
    #[test]
    fn ground_imp_lanes_rate_invariant() {
        let chart: Vec<ChartTuple> = vec![(100.0, 0.0, 1.0, 0, 0.0, 6.0,
                                           false, KIND_NOTE, false, false)];
        let mut lanes = Vec::new();
        for hz in [30u32, 60, 120] {
            let rate = TickRate::new(hz).unwrap();
            let mut core = Core::new_core_rate(100_000, 5, rate);
            core.set_chart_notes(chart.clone());
            // 0.1 s: past the reset's settling transient, well before the
            // zero-torque topple unloads the stance
            for _ in 0..(3 * rate.ratio) {
                core.step_inner(&[0u8; NQ]);
            }
            let obs = core.build_obs();
            let l: Vec<f32> = obs[OFF_GROUND_IMP..OFF_GROUND_IMP + N_GROUND_IMP_LANES].to_vec();
            println!("{hz} Hz ground lanes {l:?} (sum {:.4})", l.iter().sum::<f32>());
            lanes.push(l);
        }
        let s30: f32 = lanes[0].iter().sum();
        assert!((s30 - 1.0).abs() < 0.1, "a settled stance sums to ~1: {s30}");
        for k in 1..3 {
            for i in 0..N_GROUND_IMP_LANES {
                assert!((lanes[k][i] - lanes[0][i]).abs() < 0.02,
                        "lane {i}: {:?} vs {:?}", lanes[k], lanes[0]);
            }
        }
    }

    /// Action repeat: step_repeat(a, n) is exactly n step_inner(a) calls
    /// (state, summed reward, events in tick order) and stops at done.
    #[test]
    fn action_repeat_matches_repeated_steps() {
        // notes 0.3 -> 0.25 s (exp/wrist-tendon zone starts, 2026-10-05: the
        // flipping-pitch body falls at ~0.6 s, before the 0.5 s note resolved)
        let chart: Vec<ChartTuple> = (0u8..3)
            .map(|i| (0.25 + i as Real * 0.1, 0.0, 1.0, i % 2, 0.0, 6.0,
                      false, KIND_NOTE, false, false))
            .collect();
        let rate = TickRate::new(60).unwrap();
        let mut a = Core::new_core_rate(100_000, 5, rate);
        let mut b = Core::new_core_rate(100_000, 5, rate);
        a.set_chart_notes(chart.clone());
        b.set_chart_notes(chart);
        let mut acts = [0u8; NQ];
        acts[10] = 1;
        acts[15] = 2;
        let mut steps = 0;
        loop {
            let mut ev_a = Vec::new();
            let (ra, da) = a.step_repeat(&acts, 2, |e| ev_a.extend_from_slice(e));
            let mut ev_b = Vec::new();
            let (r1, d1) = b.step_inner(&acts);
            ev_b.extend_from_slice(&b.events_buf);
            let (mut rb, mut db) = (r1, d1);
            if !d1 {
                let (r2, d2) = b.step_inner(&acts);
                ev_b.extend_from_slice(&b.events_buf);
                rb += r2;
                db = d2;
            }
            assert_eq!(ra.to_bits(), rb.to_bits(), "reward at decision {steps}");
            assert_eq!(da, db);
            assert_eq!(ev_a, ev_b);
            assert_eq!(a.tick, b.tick);
            assert_eq!(a.build_obs(), b.build_obs());
            steps += 1;
            if da {
                break;
            }
            // flip the action every decision so holding is actually tested
            acts[10] = if acts[10] == 1 { 2 } else { 1 };
        }
        assert!(a.misses + a.good_cuts + a.bad_cuts == 3, "every note resolved ({} {} {}, tick {}, falls {} tilts {})", a.misses, a.good_cuts, a.bad_cuts, a.tick, a.falls, a.tilts);
        // repeat = 1 is one plain step
        let mut c = Core::new_core_rate(100_000, 5, rate);
        let mut d = Core::new_core_rate(100_000, 5, rate);
        c.set_chart_notes(vec![(5.0, 0.0, 1.0, 0, 0.0, 6.0, false, KIND_NOTE, false, false)]);
        d.set_chart_notes(vec![(5.0, 0.0, 1.0, 0, 0.0, 6.0, false, KIND_NOTE, false, false)]);
        let (rc, _) = c.step_repeat(&acts, 1, |_| {});
        let (rd, _) = d.step_inner(&acts);
        assert_eq!(rc.to_bits(), rd.to_bits());
        assert_eq!(c.tick, 1);
    }

    /// Fast replay generation: the batched Recorder captures, per SIM
    /// tick (inside action repeat too), exactly what RawBeatSim's replay
    /// getters read after that tick — checked bit for bit against a
    /// single-tick RawBeatSim twin at 60 Hz with repeat 2, through cuts,
    /// self-contacts and the terminal (fall) tick; latched ticks after
    /// done add nothing.
    #[test]
    fn recorder_frames_equal_raw_getters() {
        let chart: Vec<ChartTuple> = (0u8..6)
            .map(|i| (0.4 + i as Real * 0.15, -0.3 + 0.2 * (i % 4) as Real,
                      1.0, i % 2, 0.0, 6.0, i % 3 == 0, KIND_NOTE, false, false))
            .collect();
        let rate = TickRate::new(60).unwrap();
        let mut a = Core::new_core_rate(100_000, 5, rate);
        a.set_chart_notes(chart.clone());
        a.rec = Some(Box::new(Recorder::new()));
        let mut core_b = Core::new_core_rate(100_000, 5, rate);
        core_b.set_chart_notes(chart);
        let mut b = RawBeatSim { core: core_b };
        let mut acts = [0u8; NQ];
        acts[3] = 1;
        acts[8] = 2;
        acts[10] = 1;
        acts[13] = 2;
        acts[15] = 1;
        let mut want: Vec<(u32, Real, Vec<Real>, Vec<(u32, Real, Real, Real, Real)>,
                           Vec<(u32, u8, f32, f32, f32)>, [u8; NQ])> = Vec::new();
        let mut n_events = 0;
        for decision in 0..10_000 {
            let (_, da) = a.step_repeat(&acts, 2, |_| {});
            for _ in 0..2 {
                let (r, db) = b.core.step_inner(&acts);
                let mut f: Vec<Real> = b.qpos();
                f.extend_from_slice(&b.twist());
                f.extend(b.omega());
                for s in b.sabers() {
                    f.extend_from_slice(&s);
                }
                f.extend_from_slice(&b.head_pose());
                for g in b.ground_reaction() {
                    f.extend_from_slice(&g);
                }
                let (a0, a1, a2, a3) = b.last_activation();
                f.extend_from_slice(&[a0, a1, a2, a3]);
                let (t0, t1, t2, t3) = b.last_torque();
                f.extend_from_slice(&[t0, t1, t2, t3]);
                f.push(r);
                n_events += b.core.events_buf.len();
                want.push((b.tick() - 1, b.time(), f, b.contacts(),
                           b.core.events_buf.clone(), acts));
                if db {
                    break;
                }
            }
            assert_eq!(da, b.core.finished, "done at decision {decision}");
            if da {
                break;
            }
            // vary the action so the arms keep swinging through the chart
            acts[10] = if decision % 6 < 3 { 1 } else { 2 };
            acts[13] = if decision % 4 < 2 { 2 } else { 1 };
        }
        assert!(a.finished, "the chart never finished");
        assert!(n_events > 0, "the test chart produced no events");
        // latched ticks of a finished env record nothing
        a.step_repeat(&acts, 2, |_| {});
        let rec = a.rec.take().unwrap();
        assert_eq!(rec.tick.len(), want.len());
        assert_eq!(rec.event_off.len(), want.len() + 1);
        for (k, (tick, time, f, contacts, events, act)) in want.iter().enumerate() {
            assert_eq!(rec.tick[k], *tick);
            assert_eq!(rec.time[k].to_bits(), time.to_bits(), "time at {k}");
            let got = &rec.frame[k * REC_LANES..(k + 1) * REC_LANES];
            assert_eq!(got.len(), f.len());
            for (l, (x, y)) in got.iter().zip(f).enumerate() {
                assert_eq!(x.to_bits(), y.to_bits(), "tick {k} lane {l}");
            }
            assert_eq!(&rec.action[k * NQ..(k + 1) * NQ], &act[..], "action at {k}");
            let (c0, c1) = (rec.contact_off[k] as usize, rec.contact_off[k + 1] as usize);
            assert_eq!(c1 - c0, contacts.len(), "contacts at {k}");
            for (m, &(p, x, y, z, j)) in contacts.iter().enumerate() {
                assert_eq!(rec.contact_pair[c0 + m], p);
                assert_eq!(&rec.contact_val[4 * (c0 + m)..4 * (c0 + m) + 4], &[x, y, z, j]);
            }
            let (e0, e1) = (rec.event_off[k] as usize, rec.event_off[k + 1] as usize);
            let got_ev: Vec<(u32, u8, f32, f32, f32)> = (e0..e1)
                .map(|m| (rec.event_note[m], rec.event_code[m], rec.event_val[3 * m],
                          rec.event_val[3 * m + 1], rec.event_val[3 * m + 2]))
                .collect();
            assert_eq!(&got_ev, events, "events at {k}");
        }
        assert_eq!(rec.tick[0], 0);
        assert!(rec.tick.windows(2).all(|w| w[1] == w[0] + 1));    }

    /// The episode clock keeps real time: a chart ends at the same time
    /// (to within one 30 Hz tick) with the same outcomes at every rate.
    #[test]
    fn episode_clock_is_real_time() {
        let chart: Vec<ChartTuple> = (0u8..3)
            .map(|i| (1.0 + i as Real * 0.25, 0.0, 1.0, i % 2, 0.0, 6.0,
                      false, KIND_NOTE, false, false))
            .collect();
        let mut ends = Vec::new();
        for hz in [30u32, 60, 120] {
            let rate = TickRate::new(hz).unwrap();
            let mut core = Core::new_core_rate(690, 5, rate);
            core.set_chart_notes(chart.clone());
            let mut t = 0;
            while !core.step_inner(&[0u8; NQ]).1 {
                t += 1;
                assert!(t < 100_000);
            }
            ends.push((core.time, core.misses, core.tick));
        }
        for k in 1..3 {
            assert!((ends[k].0 - ends[0].0).abs() <= DT_30 + 1e-9, "{ends:?}");
            assert_eq!(ends[k].1, ends[0].1, "{ends:?}");
        }
    }

    /// Live gates at the reference pose reproduce the v9 constants:
    /// reach 1.75 + the env v42 handle (0.16) = 1.91, miss boundary −0.65
    /// (SPEC-v10 §4).
    #[test]
    fn live_gates_reference_values() {
        let body = Body::new();
        assert!((body.reach_z() - (1.75 + HANDLE_LEN)).abs() < 1e-12);
        assert!((body.head[2] - (-0.65)).abs() < 1e-12);
    }

    /// A swinging tick for the env v49 tests: both arms and wrists
    /// under full gears, so the blades ROTATE (tip velocity != wrist
    /// velocity) — returns (wrist, blade) before the tick.
    fn swing_tick(core: &mut Core) -> ([[Real; 3]; 2], [[Real; 3]; 2]) {
        let mut acts = [0u8; NQ];
        acts[3] = 1; // l arm pitch +
        acts[8] = 2; // l wrist bend −
        acts[10] = 1; // r arm pitch +
        acts[13] = 2; // r elbow −
        acts[15] = 1; // r wrist bend +
        let before = (core.body.wrist, core.body.blade);
        core.step_inner(&acts);
        before
    }

    fn far_chart() -> Vec<ChartTuple> {
        (0u8..3)
            .map(|i| (8.0 + i as Real * 0.25, 0.0, 1.0, i % 2, 0.0, 6.0,
                      false, KIND_NOTE, false, false))
            .collect()
    }

    /// env v49 (I07): the blade-tip velocity lanes are the finite
    /// difference of the tip position wrist + (HANDLE_LEN + SABER_LEN)·
    /// blade across the tick, / V_NORM, per hand [L xyz, R xyz]; zero
    /// at reset; different from the wrist velocity on a rotating blade
    /// (the reason they exist).
    /// exp/tick-rate x I07: at 60 Hz the tip lanes divide by the
    /// runtime tick length (1/60 s), not the 30 Hz constant.
    #[test]
    fn tip_vel_lanes_use_the_runtime_tick_at_60hz() {
        let rate = TickRate::new(60).unwrap();
        let mut core = Core::new_core_rate(1380, 5, rate);
        core.set_chart_notes(far_chart());
        let l = HANDLE_LEN + SABER_LEN;
        let dt = 1.0 / 60.0;
        assert!((core.body.rate.dt - dt).abs() < 1e-12);
        for _ in 0..6 {
            let (w0, b0) = swing_tick(&mut core);
            let (w1, b1) = (core.body.wrist, core.body.blade);
            for s in 0..2 {
                for k in 0..3 {
                    let fd = ((w1[s][k] + l * b1[s][k])
                        - (w0[s][k] + l * b0[s][k])) / dt;
                    let tv = core.body.tip_vel[s][k];
                    let tol = 1e3 * Real::EPSILON * (1.0 + fd.abs()) + 1e-9;
                    assert!((tv - fd).abs() <= tol, "s{s} k{k}: {tv} vs {fd}");
                }
            }
        }
    }

    #[test]
    fn tip_vel_lanes_are_the_tip_finite_difference() {
        let mut core = Core::new_core(690, 5);
        core.set_chart_notes(far_chart());
        let obs0 = core.build_obs();
        assert!(obs0[OFF_TIP_VEL..OFF_TIP_VEL + TIP_VEL_FEAT].iter()
                    .all(|&x| x == 0.0), "tip lanes must be 0 at reset");
        assert_eq!(OFF_TIP_VEL, 624);
        assert_eq!(OBS_DIM, OFF_TIP_VEL + TIP_VEL_FEAT);
        let l = HANDLE_LEN + SABER_LEN;
        let mut max_gap: Real = 0.0;
        for _ in 0..6 {
            let (w0, b0) = swing_tick(&mut core);
            let (w1, b1) = (core.body.wrist, core.body.blade);
            let obs = core.build_obs();
            for s in 0..2 {
                for k in 0..3 {
                    let fd = ((w1[s][k] + l * b1[s][k])
                        - (w0[s][k] + l * b0[s][k])) / core.body.rate.dt;
                    let tv = core.body.tip_vel[s][k];
                    let tol = 1e3 * Real::EPSILON * (1.0 + fd.abs());
                    assert!((tv - fd).abs() <= tol,
                            "hand {s} axis {k}: tip_vel {tv} vs FD {fd}");
                    assert_eq!(obs[OFF_TIP_VEL + 3 * s + k],
                               (tv / V_NORM) as f32, "lane encoding");
                    max_gap = max_gap
                        .max((tv - core.body.wrist_vel[s][k]).abs());
                }
            }
        }
        assert!(max_gap > 0.5,
                "a rotating blade's tip must outrun the wrist (gap {max_gap} m/s)");
        // a reset clears it
        core.reset_state();
        assert_eq!(core.body.tip_vel, [[0.0; 3]; 2]);
        let obs = core.build_obs();
        assert!(obs[OFF_TIP_VEL..].iter().all(|&x| x == 0.0));
    }

    /// env v49 (I07) sign convention of the policy's cut-plane lane,
    /// pinned against the judge: from the OBS lanes alone (head-relative
    /// wrist x/y, world z, blade dir, tip velocity), the signed distance
    /// dp·norm(blade × v_tip) of a point p = wrist + a·blade + c·n̂ is
    /// +c, where n̂ = norm(blade × tip_move) is the judge's cut-plane
    /// normal (step_inner) — so positive = the side blade × v_tip points
    /// to, and |signed| is exactly the judge's accuracy distance d_min.
    #[test]
    fn cut_plane_signed_distance_matches_the_judge() {
        let mut core = Core::new_core(690, 5);
        core.set_chart_notes(far_chart());
        let l = HANDLE_LEN + SABER_LEN;
        for _ in 0..4 {
            swing_tick(&mut core);
        }
        let (w0, b0) = swing_tick(&mut core);
        let (w1, b1) = (core.body.wrist, core.body.blade);
        let head = core.body.head;
        let obs = core.build_obs();
        for s in 0..2 {
            // the judge's plane (step_inner's tip_move / pn)
            let mut tip_move = [0.0 as Real; 3];
            for k in 0..3 {
                tip_move[k] = w1[s][k] + l * b1[s][k] - w0[s][k] - l * b0[s][k];
            }
            let pn = cross3(b1[s], tip_move);
            let pn_mag = dot3(pn, pn).sqrt();
            assert!(pn_mag > 1e-6, "hand {s}: degenerate swing");
            let nh = [pn[0] / pn_mag, pn[1] / pn_mag, pn[2] / pn_mag];
            // decode the obs lanes the policy reads
            let hb = if s == 0 { OFF_HAND_L } else { OFF_HAND_R };
            let wx = obs[hb] as Real * X_NORM + head[0];
            let wy = obs[hb + 1] as Real * Y_NORM + head[1];
            let wz = obs[hb + 2] as Real;
            let b = [obs[hb + 6] as Real, obs[hb + 7] as Real, obs[hb + 8] as Real];
            let v = [
                obs[OFF_TIP_VEL + 3 * s] as Real * V_NORM,
                obs[OFF_TIP_VEL + 3 * s + 1] as Real * V_NORM,
                obs[OFF_TIP_VEL + 3 * s + 2] as Real * V_NORM,
            ];
            let bv = cross3(b, v);
            let bv_mag = dot3(bv, bv).sqrt();
            for (a, c) in [(0.6 as Real, 0.2 as Real), (1.0, -0.15), (0.3, 0.0)] {
                let p = [
                    w1[s][0] + a * b1[s][0] + c * nh[0],
                    w1[s][1] + a * b1[s][1] + c * nh[1],
                    w1[s][2] + a * b1[s][2] + c * nh[2],
                ];
                let dp = [p[0] - wx, p[1] - wy, p[2] - wz];
                let signed = dot3(dp, bv) / bv_mag;
                assert!((signed - c).abs() < 2e-4,
                        "hand {s} (a {a}, c {c}): obs-derived signed {signed}");
                // the judge's accuracy distance (cw = centre - wrist)
                let cw = [p[0] - w1[s][0], p[1] - w1[s][1], p[2] - w1[s][2]];
                let d_min = (dot3(cw, pn) / pn_mag).abs();
                assert!((signed.abs() - d_min).abs() < 2e-4,
                        "hand {s}: |signed| {signed} vs judge d_min {d_min}");
            }
        }
    }
}
