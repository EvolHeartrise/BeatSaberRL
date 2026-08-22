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

use dynamics::collide::{CORNERS_PER_FOOT, NUM_PAIRS, PAIRS};
use dynamics::fk::quat_to_rot;
use dynamics::model::{
    Model, F_HEAD, F_L_ELBOW, F_L_FOOT, F_L_HAND, F_L_SHANK, F_L_SHOULDER,
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
// env v12 (unilateral-contact spec §7, 2026-08-09): per-tick survival
// bonus, CONTACT MODE ONLY. The v11.1 fall pricing left survival-time
// flat BEFORE the first note (every pre-first-note fall prices the full
// chart), so a from-scratch policy gets zero gradient on "die at 0.5 s
// vs 1.9 s" until exploration tunnels to 2.0 s on its own. 0.002/tick
// (0.06/s) keeps a full 3-min chart's worth (~+11) well under the note
// stream's scale (~|70|+), so it cannot out-bid hitting notes; it is
// overridden on fall/escape ticks (the terminal pricing wins).
const ALIVE_BONUS: f64 = 0.002;
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
// env v11 FALL termination (unilateral-contact spec §4), CONTACT MODE
// ONLY: with the feet no longer welded the root can go to the floor, and
// a fallen body's observations/rewards are not the task. Reference-pose
// root y is 0.94; the original 0.55 was a deep crouch the policy cannot
// reach by squatting within the leg limits and holding the sole flat.
// 2026-08-20: lowered to 0.45 then 0.4 to add dynamic headroom below the legal
// squat envelope — the entry-edge WALL_PEN wants ducks to be learnable,
// and a fast hip-fold descent transiently undershoots the static
// minimum. Reported exactly like the root-escape envelope (SPEC-v10
// §20e "reported, never hidden"): per-tick bool, popped cumulative
// count, episode counter.
const FALL_ROOT_Y: f64 = 0.25;
// TILT termination (2026-08-20), contact mode only: CHEST up-axis
// tilted more than 40 deg FORWARD, 20 deg BACKWARD (asymmetric
// sagittal, user 2026-08-20 — the tight back bound targets the v30
// backbend exploit; forward stays duck-sized) or 40 deg to the side
// (lateral) at the END of a tick ends the episode, priced like a
// fall (flat -fall_const). Per-TICK check. Tangent-squared form;
// forward = up_z > 0 (body faces +z); also trips whenever
// up_y <= 0 (past horizontal).
const TILT_TAN2_FWD: f64 = 0.7040881910418472; // tan^2(40 deg)
const TILT_TAN2_BACK: f64 = 0.1324743314317942; // tan^2(20 deg) — 2026-08-21: briefly 10 deg, reverted to 20 same day (user)
const TILT_TAN2_ROLL: f64 = 0.7040881910418472; // tan^2(40 deg)
// FOOT-LIFT penalty (2026-08-20), contact mode only: flat charge on
// the TICK a foot leaves the ground (edge: that foot's tick-summed
// ground normal impulse hits zero after a grounded tick), per foot.
const FOOT_LIFT_PEN: f64 = -0.1;

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
// env v19 flail telemetry: a tick is "note-free" when the next pending
// chart object is further out than this (s). Telemetry only.
const FLAIL_LEAD_S: f64 = 1.0;
// env v20 episode-start arm stance (§1.4 indices): l/r arm_pitch -80
// deg, l/r elbow 100 deg, l/r wrist_bend 70 deg.
// env v21 (user, 2026-08-16): wide stance — hip_roll splays the legs
// +-8 deg, ankle_roll counter-rolls so the soles stay flat (verified
// via the fk_body mirror: foot up-vector exactly (0,1,0)). CONTACT
// MODE ONLY: in weld mode the feet are welded at the v10 reference,
// and a leg stance would fight the welds.
const Q_START_LEG: [(usize, f64); 4] = [
    (19, -0.13962634015954636),  // l_hip_roll  -8 deg
    (22, 0.13962634015954636),   // l_ankle_roll +8 deg (counter)
    (25, 0.13962634015954636),   // r_hip_roll  +8 deg
    (28, -0.13962634015954636),  // r_ankle_roll -8 deg (counter)
];
// The wide stance's foot X positions (fk_body of the v21 start pose;
// the root ground-drop does not move X). Per-foot penalty targets:
// left foot -> -FOOT_X0, right foot -> +FOOT_X0.
const FOOT_X0: f64 = 0.22108059783525694;

const Q_START_ARM: [(usize, f64); 6] = [
    (3, -1.3962634015954636),
    (6, 1.7453292519943295),
    (8, 1.2217304763960306),
    (10, -1.3962634015954636),
    (13, 1.7453292519943295),
    (15, 1.2217304763960306),
];

// object kinds (mirrors constants.py)
const KIND_NOTE: u8 = 0;
const KIND_BOMB: u8 = 1;
const KIND_CHAIN_HEAD: u8 = 2;
const KIND_CHAIN_LINK: u8 = 3;

// env v17 (contact spec §12): walls. The head CENTER is the collision
// point (no sweep). WALL_PEN is charged ONCE, on the tick the head
// ENTERS a wall (the was_in_wall rising edge, same trigger as the
// combo break; >= 1 tick fully outside re-arms it) — 2026-08-20,
// was -0.05 every tick inside. walls_in still counts ticks inside.
// The viewer builds wall front faces 0.25 m behind the note plane
// ("walls just built like that" — WallManager.cs UpdateVisual), so a
// wall's front reaches the head 0.4/NJS s after its nominal time.
const WALL_PEN: f64 = -1.5; // 2026-08-22: -0.5 -> -1.5 (bomb parity, user)
const WALL_FRONT_OFF: f64 = 0.25;

// --- obs layout v17 (spec §12d; offsets DERIVED, order is a contract) --------
// The 16 slots are a SHARED entity window (notes + walls), filled
// nearest-first by 3D distance from the head center; entities behind
// the head consume no slots. ENT_FEAT = the v9 note lanes verbatim,
// then [is_wall, width, height, length].
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
// front/back sole-corner pair, / GROUND_IMP_NORM. All-zero in weld mode
// (the ground block does not exist there), so lanes 0..340 are the v10
// obs verbatim.
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
const OBS_DIM: usize = OFF_BODY_PT_VEL + 3 * N_BODY_PTS; // 598 (env v24)
// §8 body-point frames + their obs y-centering (shared by the position
// lanes, the velocity finite-diff and the foot reward terms' docs)
const BODY_PTS: [(usize, f64); N_BODY_PTS] = [
    (F_L_ELBOW, Y_CENTER), (F_R_ELBOW, Y_CENTER),
    (F_L_SHANK, KNEE_Y0), (F_R_SHANK, KNEE_Y0),
    (F_L_FOOT, FOOT_Y0), (F_R_FOOT, FOOT_Y0),
];
const _: () = assert!(OBS_DIM == 598, "spec §12 (env v24 body_pt_vel lanes): OBS_DIM must be 598");
const KNEE_Y0: f64 = 0.50; // hip 0.94 − thigh 0.44
const FOOT_Y0: f64 = 0.07; // ankle height at reference (contact spec §2)
// ≈ one tick of full-body weight: M_TOTAL·g·DT = 74.50 kg · 9.81 · (1/30)
// = 24.53 N·s, rounded to a spec literal both engines share. A settled
// two-foot stance sums to ≈ 1.0 across the four lanes.
const GROUND_IMP_NORM: f64 = 24.5;
const _: () = assert!(OFF_QD == 46 && OFF_HAND_L == 75 && OFF_NOTES == 102);

const X_NORM: f64 = 1.0;
const Y_NORM: f64 = 0.75;
const Z_SCALE: f64 = 8.0;
const Z_CLIP: (f64, f64) = (-0.65, 12.0); // lo = PLANE_Z0 = the head/miss boundary (2026-08-22; was -3.0, briefly 0.0 — see .cu note)
// v28 probe (2026-08-20): the obs z-lane upper clamp is runtime-
// overridable via BEAT_Z_CLIP_HI (meters; default Z_CLIP.1 = the
// pre-v28 era) so one binary serves every checkpoint era — the CUDA
// twin takes it as a step parameter instead. Read once per process.
fn z_clip_hi() -> f64 {
    static V: std::sync::OnceLock<f64> = std::sync::OnceLock::new();
    *V.get_or_init(|| {
        std::env::var("BEAT_Z_CLIP_HI")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(Z_CLIP.1)
    })
}
const ROOT_Y0: f64 = 0.94; // reference root height (obs centering)
const HEAD_Y0: f64 = 1.70; // reference head height (obs centering)
const PLANE_Z0: f64 = -0.65; // reference body plane (obs centering)
// env v15 (contact spec §10): the note track is head-anchored in z —
// every note's world z carries the offset head_z − PLANE_Z0 (zero at
// the reference pose). Real-BS room semantics: walking toward the
// notes shifts them away by exactly the walked amount, so approach
// distance (and therefore hit/miss timing) is invariant to root z.
// Applied at every note-z evaluation: obs lane, broad-phase bound,
// judgment. x/y stay world-fixed (lanes are body-independent).
const ROOT_VLIN_NORM: f64 = 6.0;
const ROOT_VANG_NORM: f64 = 16.0;

/// Real BS scores are integers; explicit floor(x + 0.5) so every
/// implementation (Rust, torch, Python tools) agrees on halves.
fn round_pts(x: f64) -> f64 {
    (x + 0.5).floor()
}

/// env v17 obstacle (spec §12): a world-space AABB riding the
/// head-anchored track. Geometry arrives FINISHED from the parser
/// (viewer scales/clamps applied); the sim only moves the box and
/// tests the head point. Half-extents precomputed for the AABB test.
#[derive(Clone, Copy)]
struct Wall {
    start_time: f64, // front face crosses the note plane (nominal, s)
    cx: f64,         // box center x (m, world)
    cy: f64,         // box center y (m, world)
    half_w: f64,
    half_h: f64,
    len: f64,        // full z extent (m)
    speed: f64,      // approach speed (m/s)
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
// wall_tuple order (chart.py): (t0, cx, cy, width, height, length, speed)
type WallTuple = (f64, f64, f64, f64, f64, f64, f64);

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
    // env v24: §8 body-point world velocities, same finite-diff scheme
    body_pt_vel: [[f64; 3]; N_BODY_PTS],
    swing_acc: [f64; 2],      // arc of the current continuous swing (rad)
    axis: [Option<[f64; 3]>; 2], // last tick's rotation axis (unit)
    // per-tick contact aggregation (obs bits + replay events)
    contact_arm: [bool; 2],          // any active contact row touching the arm
    contact_imp: [f64; NUM_PAIRS],   // tick-summed normal impulse per pair
    contact_pt: [[f64; 3]; NUM_PAIRS], // last substep's contact point
    effort_j: f64,                   // last tick's positive work (J)
    // env v23 activation (user direction 2026-08-17): this tick's
    // commanded |gear fraction| summed over the 29 joints — the
    // muscle-activation analog (Σ|tau_j|/tau_max_j, constant over the
    // tick since tau is held). Priced by Core::iso_coef; isometric
    // holds and limit-jamming pay it, coast is the only free command.
    act_frac: f64,
    // C backstop (2026-08-03, SPEC-v10 Divergences §23): did the
    // PRE-integration root envelope engage anywhere in THIS tick's 8
    // substeps? A1 makes such a trajectory bounded; it does not make it
    // physics, so the env terminates the episode instead of feeding the
    // policy the garbage observation/reward (the fp16 loss-scale
    // collapse and the §22 bench panic were both downstream of exactly
    // that). Measured incidence on a converged policy: 0 in 204,800
    // env-ticks.
    escaped: bool,
    // env v11 (unilateral-contact spec §1): foot_mode. false = weld
    // (v10, the DEFAULT — every existing trajectory is bitwise
    // unchanged), true = unilateral frictional ground contact.
    foot_contact: bool,
    // Contact spec §4: was the root below FALL_ROOT_Y at the END of
    // this tick (per-TICK since 2026-08-20)? Contact mode only. Same
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
    // P6 (additive introspection): tick-summed foot-weld impulses (N·s),
    // solver rows 0–11 — [foot_l lin xyz, ang xyz, foot_r lin, ang].
    // λ is the constraint impulse ON the body (world frame): the ground
    // reaction; vertical force ≈ weld_imp[1 or 7] / DT.
    weld_imp: [f64; 12],
    // env v12 (contact spec §7): tick-summed ground NORMAL impulses per
    // front/back sole-corner pair — [L_heel, L_toe, R_heel, R_toe]
    // (N·s). Contact mode only; stays 0 under welds. Feeds the four
    // OFF_GROUND_IMP obs lanes.
    ground_imp: [f64; 4],
    // env v34 (2026-08-21): tick-summed joint-limit-row impulses
    // (N·m·s) the stops carried, bucketed [legs, spine]. Scope is
    // POSTURAL joints only — spine 0..3 and legs 17..29; the 14 arm
    // joints are excluded (whips legitimately slam arm stops at
    // speed). Step-11 clamp impulses are NOT counted: the solver is
    // limit-aware, so those are near-null by design (solver.rs R10
    // notes). Priced by Core::limit_coef; resting a pose on its
    // stops shows up as sustained λ-flow here, an active hold does
    // not.
    limit_imp: [f64; 3],
    // replay fix (2026-08-17): tick-summed ground-reaction impulse per
    // foot [L, R] x [x friction, y normal, z friction] (N·s, world
    // frame) — ALL rows of every active sole corner, not just the §7
    // normal buckets. Contact mode only; stays 0 under welds. Feeds
    // the replay "feet" lanes, which read all-zero in contact mode
    // while they were wired to the retired welds' impulses.
    foot_grf: [[f64; 3]; 2],
}

const MODEL: &Model = &HUMANOID;

impl Body {
    fn new(foot_contact: bool) -> Body {
        let st = State::neutral(MODEL);
        let ws = Workspace::new(MODEL);
        // contact spec §1: contact mode DROPS the 12 weld rows
        let cs = ConstraintSet::new_modes(MODEL, !foot_contact, true, foot_contact);
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
            body_pt_vel: [[0.0; 3]; N_BODY_PTS],
            swing_acc: [0.0; 2],
            axis: [None; 2],
            contact_arm: [false; 2],
            contact_imp: [0.0; NUM_PAIRS],
            contact_pt: [[0.0; 3]; NUM_PAIRS],
            effort_j: 0.0,
            act_frac: 0.0,
            escaped: false,
            foot_contact,
            fell: false,
            tilted: false,
            fall_hits: 0,
            tilt_hits: 0,
            foot_down_prev: [false, false],
            foot_lifts_tick: 0,
            foot_lift_hits: [0, 0],
            weld_imp: [0.0; 12],
            ground_imp: [0.0; 4],
            limit_imp: [0.0; 3],
            foot_grf: [[0.0; 3]; 2],
        };
        b.refresh_fk();
        b
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
        // env v21 wide stance (contact mode only; weld mode's feet are
        // welded at the v10 reference and a stance would fight them)
        if self.foot_contact {
            for (j, v) in Q_START_LEG {
                self.st.q[j] = v;
            }
        }
        self.tau = [0.0; NQ];
        self.wrist_vel = [[0.0; 3]; 2];
        self.body_pt_vel = [[0.0; 3]; N_BODY_PTS];
        self.swing_acc = [0.0; 2];
        self.axis = [None; 2];
        self.contact_arm = [false; 2];
        self.contact_imp = [0.0; NUM_PAIRS];
        self.effort_j = 0.0;
        self.act_frac = 0.0;
        self.escaped = false;
        self.fell = false;
        self.tilted = false;
        // arm-after-first-plant (user 2026-08-20): start UNARMED so a
        // foot that begins the episode in the air (start-pose jitter)
        // is never charged — the lift edge only arms once the foot has
        // actually been seen grounded.
        self.foot_down_prev = [false, false];
        self.foot_lifts_tick = 0;
        self.weld_imp = [0.0; 12];
        self.ground_imp = [0.0; 4];
        self.foot_grf = [[0.0; 3]; 2];
        self.refresh_fk();
        // env v21: the wide stance lifts the ankles ~8.5 mm, so drop
        // the root until the lowest sole corner sits exactly on the
        // ground (the v20 jitter's corner arithmetic; both engines
        // share this expression tree).
        if self.foot_contact {
            use dynamics::collide::{GROUND_CORNERS, GROUND_FRAMES};
            let mut min_y = f64::INFINITY;
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
        let mut act = 0.0;
        for j in 0..NQ {
            let a = actions[j] as usize;
            let frac = if a < 7 { gear[a] } else { 0.0 };
            self.tau[j] = frac * MODEL.tau_max[j];
            act += frac.abs();
        }
        self.act_frac = act;
        let prev_wrist = self.wrist;
        // env v24: last tick's §8 body-point positions (ws.p holds the
        // previous refresh_fk), the finite-diff base for body_pt_vel
        let mut prev_bp = [[0.0f64; 3]; N_BODY_PTS];
        for (i, (f, _)) in BODY_PTS.iter().enumerate() {
            prev_bp[i] = self.ws.p[*f];
        }
        self.contact_arm = [false; 2];
        self.contact_imp = [0.0; NUM_PAIRS];
        self.weld_imp = [0.0; 12];
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
            // the self-collision rows sit AFTER the ground block (0 rows
            // in weld mode) — contact spec §3
            let ncbase = nweld + self.cs.ground_rows();
            // P6: tick-summed foot-weld impulses (ground reaction)
            for r in 0..nweld {
                self.weld_imp[r] += self.cs.lambda[r];
            }
            // contact spec §7: tick-summed ground NORMAL impulses,
            // bucketed [L_heel, L_toe, R_heel, R_toe] — corner ids are
            // foot*4 + k with k odd = z+ (toe), so bucket =
            // foot*2 + (k & 1). Obs lanes read these after the tick.
            if self.foot_contact {
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
                        2 | 4 | 6 => self.contact_arm[0] = true,
                        3 | 5 | 7 => self.contact_arm[1] = true,
                        _ => {}
                    }
                }
            }
        }
        self.effort_j = effort;
        self.escaped = self.ws.env_hits != env_hits0;
        // contact spec §4 (per-TICK since 2026-08-20): root below the
        // floor threshold at the end of the tick
        if self.foot_contact && self.st.p[1] < FALL_ROOT_Y {
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
        if self.foot_contact {
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
        }
        self.tilted = tilted;
        if tilted {
            self.tilt_hits += 1;
        }
        // FOOT-LIFT (2026-08-20): edge on the tick-summed ground
        // normal impulse per foot (ground_imp buckets
        // [L_heel, L_toe, R_heel, R_toe]); charged in step_inner.
        self.foot_lifts_tick = 0;
        if self.foot_contact {
            for f in 0..2 {
                let down =
                    self.ground_imp[f * 2] + self.ground_imp[f * 2 + 1] > 0.0;
                if !down && self.foot_down_prev[f] {
                    self.foot_lifts_tick += 1;
                    self.foot_lift_hits[f] += 1;
                }
                self.foot_down_prev[f] = down;
            }
        }
        self.refresh_fk();
        for s in 0..2 {
            for k in 0..3 {
                self.wrist_vel[s][k] = (self.wrist[s][k] - prev_wrist[s][k]) / DT;
            }
        }
        for (i, (f, _)) in BODY_PTS.iter().enumerate() {
            for k in 0..3 {
                self.body_pt_vel[i][k] =
                    (self.ws.p[*f][k] - prev_bp[i][k]) / DT;
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
    // contact spec §4: fall terminations in THIS episode (0 or 1 — the
    // fall ends the episode). Reported, never hidden: the trainer's
    // `[fall]` line reads it.
    falls: u32,
    // TILT terminations in this episode (2026-08-20) — `falls` pattern.
    tilts: u32,
    // env v18 constant fall pricing (2026-08-15, sole mode): flat
    // -fall_const terminal reward on a fall (see fell block).
    fall_const: f64,
    // env v20 (2026-08-16, user direction): per-tick penalty
    // foot_z_coef * (|lfoot_z - PLANE_Z0| + |rfoot_z - PLANE_Z0|) —
    // linear in each foot's world-z distance from the reference
    // ("standard starting") foot z. Targets the shimmy toward/away
    // from the note plane. 0.0 = off (the pre-v20 reward verbatim).
    foot_z_coef: f64,
    // env v21: the X twin — per-tick foot_x_coef * (|lfoot_x -
    // (-FOOT_X0)| + |rfoot_x - FOOT_X0|), each foot priced against its
    // OWN side's wide-stance x. 0.0 = off.
    foot_x_coef: f64,
    // env v22 (user direction 2026-08-16): the foot penalties become
    // HINGED — deviation inside this radius is free ("target AREAS"),
    // rent applies only to the excess. 0.0 = the v21 pure-linear
    // behavior verbatim.
    foot_hinge_r: f64,
    // env v22: potential-based delta shaping (Ng et al. 1999,
    // undiscounted variant — round trips telescope to exactly zero):
    // each tick, reward += coef * (prev_pot - pot) where pot = the
    // total hinged deviation of both feet on both axes. Sharpens
    // credit assignment onto the tick that moved a foot; the RENT
    // terms remain the anti-parking objective (delta shaping alone
    // would make a parked pose free). 0.0 = off.
    foot_shape_coef: f64,
    // env v23: per-tick activation price on Body::act_frac (0 = the
    // pre-v23 reward verbatim; velocity-blind by design — the work
    // term effort_coef prices speed, this prices force)
    iso_coef: f64,
    // env v34 (2026-08-21): per-tick price on the joint-limit-row
    // impulse the postural stops carried (Body::limit_imp, [legs,
    // spine] N·m·s). 0 = the pre-v34 reward verbatim. Proportional to
    // the CARRIED impulse, so a knee-locked stand (near-zero moment on
    // the stop) is near-free while a crouch parked on hip/knee stops
    // pays its full gravity moment.
    limit_coef: f64,
    // env v35: widen pricing scope to the 14 arm joints too
    limit_arms: bool,
    // cumulative limit-rest telemetry until popped (pop_limit_rest,
    // fall_hits pattern): λ-flow sums [legs, spine] and the tick count
    // — accumulated REGARDLESS of limit_coef so stop-resting is
    // observable at coef 0.
    limit_rest_sum: [f64; 3],
    limit_rest_ticks: u64,
    // the previous tick's pot (reset to 0 — resets start on-target)
    prev_foot_pot: f64,
    n_scoreable: u32, // chart objects that are not bombs
    // env v19 flail telemetry (CUDA-backend mirror, telemetry only):
    // blade arc (rad) accrued on note-free ticks (next pending chart
    // object > FLAIL_LEAD_S out) by hands with no live follow-through
    // tracker, plus the count of those ticks. PER-EPISODE here (zeroed
    // by reset_state, read per level by the bench); the CUDA twin is
    // pop-based instead — each matches its consumer.
    flail_rad: f64,
    flail_ticks: u32,
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
    // this latch, which holds last tick's in-wall state.
    was_in_wall: bool,
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
    score: f64,       // summed points; max = 115*notes + 85*heads + 20*links
    score_pre: f64,
    score_acc: f64,
    score_post: f64,
    // reusable event buffer: cleared and refilled in place each tick.
    events_buf: Vec<(u32, u8, f32, f32, f32)>,
}

impl Core {
    fn new_core(max_steps: u32, effort_coef: f64, wide_acc: bool,
                gears: u8, foot_contact: bool) -> Self {
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
            body: Body::new(foot_contact),
            trackers: [Vec::new(), Vec::new()],
            status: Vec::new(),
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
            iso_coef: 0.0,
            limit_coef: 0.0,
            limit_arms: false,
            limit_rest_sum: [0.0; 3],
            limit_rest_ticks: 0,
            prev_foot_pot: 0.0,
            n_scoreable: 0,
            flail_rad: 0.0,
            flail_ticks: 0,
            walls: Vec::new(),
            wall_passed: Vec::new(),
            first_wall: 0,
            walls_in: 0,
            was_in_wall: false,
            wall_enters: 0,
            finished: false,
            score: 0.0,
            score_pre: 0.0,
            score_acc: 0.0,
            score_post: 0.0,
            events_buf: Vec::new(),
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
        self.wall_passed = vec![false; self.walls.len()];
        self.first_wall = 0;
        self.walls_in = 0;
        self.was_in_wall = false;
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
        self.prev_foot_pot = 0.0;
        self.finished = false;  // B4: every chart (re)install funnels here
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
            // env v19: remaining follow-through credit. Encoded so 0
            // means "swinging earns nothing now" (idle / broken /
            // capped all collapse there) and a fresh single-note
            // tracker reads 1.0. Summed in spawn order — the CUDA
            // ring walk from tr_ptr must add in this same order for
            // bitwise obs parity.
            let mut rem = 0.0f64;
            for tr in &self.trackers[s] {
                rem += 1.0 - (tr.acc / POST_FULL).min(1.0);
            }
            obs[base + 10] = rem.min(2.0) as f32;
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
        let z_off = self.body.head[2] - PLANE_Z0;
        let head = &self.body.head;
        // selection buffer, ascending by dist2; ties keep the FIRST-
        // scanned entity (strict-< displacement). Scan order: notes in
        // chart order, then walls — the CUDA mirror transcribes the
        // same comparison order (§5 parity).
        let mut sel = [(f64::INFINITY, false, 0usize); N_NOTE_SLOTS];
        let mut n_sel = 0usize;
        fn insert(
            sel: &mut [(f64, bool, usize); N_NOTE_SLOTS],
            n_sel: &mut usize,
            cand: (f64, bool, usize),
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
            insert(&mut sel, &mut n_sel, (dx * dx + dy * dy + dz * dz, false, i));
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
            insert(&mut sel, &mut n_sel, (sx * sx + sy * sy + sz * sz, true, wi));
        }
        for slot in 0..n_sel {
            let (_, is_wall, idx) = sel[slot];
            let base = OFF_NOTES + slot * NOTE_FEAT;
            if is_wall {
                let w = &self.walls[idx];
                let zfu = (w.start_time - self.time) * w.speed - WALL_FRONT_OFF
                    + z_off;
                let zf = zfu.clamp(Z_CLIP.0, z_clip_hi());
                obs[base] = (w.cx / X_NORM) as f32;
                obs[base + 1] = ((w.cy - Y_CENTER) / Y_NORM) as f32;
                obs[base + 2] = (zf / Z_SCALE) as f32;
                obs[base + 3] = (-w.speed / Z_SCALE) as f32;
                // env v36: lane 4 = BACK-face z (wall end), clamped
                // [Z_CLIP.0, hi] (= the head boundary, like lane 2) —
                // exact for any wall length while lane 2's front face
                // saturates inside long walls.
                obs[base + 4] =
                    ((zfu + w.len).clamp(Z_CLIP.0, z_clip_hi()) / Z_SCALE)
                        as f32;
                // lanes 5..13 (color/dir/kind/free) stay zero
                obs[base + 13] = 1.0; // valid
                obs[base + 14] = 1.0; // is_wall
                obs[base + 15] = (2.0 * w.half_w / X_NORM) as f32;
                obs[base + 16] = (2.0 * w.half_h / Y_NORM) as f32;
                obs[base + 17] = (w.len / Z_SCALE) as f32;
            } else {
                let n = &self.chart[idx];
                let z = ((n.hit_time - self.time) * n.speed + z_off)
                    .clamp(Z_CLIP.0, z_clip_hi());
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
                // lanes 14..17 (is_wall, w/h/l) stay zero
            }
            obs[OFF_NOTE_MASK + slot] = 1.0;
        }

        // contact spec §7: per-foot ground-contact lanes
        // [L_heel, L_toe, R_heel, R_toe]; all-zero under welds
        for k in 0..N_GROUND_IMP_LANES {
            obs[OFF_GROUND_IMP + k] =
                (self.body.ground_imp[k] / GROUND_IMP_NORM) as f32;
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
    fn step_inner(&mut self, actions: &[u8; NQ]) -> (f64, bool) {
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
        let mut deltas = [0.0f64; 2];
        let mut continued = [false; 2];

        // ── the physics tick (SPEC-v10 §3/§4) ─────────────────────────
        let gear = self.gear;
        self.body.tick(actions, &gear);
        reward -= self.effort_coef * self.body.effort_j;
        reward -= self.iso_coef * self.body.act_frac;
        // env v34: limit-rest penalty + always-on telemetry (see
        // Body::limit_imp for scope/units)
        reward -= self.limit_coef
            * (self.body.limit_imp[0] + self.body.limit_imp[1]
               + if self.limit_arms { self.body.limit_imp[2] } else { 0.0 });
        self.limit_rest_sum[0] += self.body.limit_imp[0];
        self.limit_rest_sum[1] += self.body.limit_imp[1];
        self.limit_rest_sum[2] += self.body.limit_imp[2];
        self.limit_rest_ticks += 1;
        // FOOT-LIFT penalty (2026-08-20): flat charge per lift event
        // this tick (see Body::tick edge detection)
        reward += FOOT_LIFT_PEN * self.body.foot_lifts_tick as f64;
        // env v20/v21/v22 foot terms: ws.p holds this tick's FK (the
        // same source the §8 body_pts obs lanes read). Hinged rent
        // (free inside foot_hinge_r, the v21 pure-linear behavior at
        // r = 0) + optional potential-based delta shaping.
        if self.foot_z_coef > 0.0 || self.foot_x_coef > 0.0
            || self.foot_shape_coef > 0.0
        {
            let r = self.foot_hinge_r;
            let h = |v: f64| (v.abs() - r).max(0.0);
            let lf = &self.body.ws.p[F_L_FOOT];
            let rf = &self.body.ws.p[F_R_FOOT];
            let hx = h(lf[0] + FOOT_X0) + h(rf[0] - FOOT_X0);
            let hz = h(lf[2] - PLANE_Z0) + h(rf[2] - PLANE_Z0);
            reward -= self.foot_x_coef * hx + self.foot_z_coef * hz;
            if self.foot_shape_coef > 0.0 {
                let pot = hx + hz;
                reward += self.foot_shape_coef * (self.prev_foot_pot - pot);
                self.prev_foot_pot = pot;
            }
        }

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
            let z_hit = dzf <= 0.0 && dzf + w.len + w.speed * DT >= 0.0;
            if dzf + w.len < 0.0 {
                self.wall_passed[wi] = true; // back face behind the head
            }
            if z_hit
                && (head[0] - w.cx).abs() <= w.half_w
                && (head[1] - w.cy).abs() <= w.half_h
            {
                in_wall = true;
            }
        }
        if in_wall {
            if !self.was_in_wall {
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
        self.was_in_wall = in_wall;
        while self.first_wall < self.walls.len() && self.wall_passed[self.first_wall] {
            self.first_wall += 1;
        }

        while self.first_pending < self.chart.len()
            && self.status[self.first_pending] != 0
        {
            self.first_pending += 1;
        }
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
        // is garbage, so none of it — effort, cuts, follow-through —
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
        let done = (!pending
            && no_trackers
            && (self.chart.is_empty() || self.time > last_hit))
            || self.time > last_hit + self.tail_s
            || self.tick >= self.max_steps
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
        // bonus, contact mode only. Added BEFORE the terminal overrides
        // below so a fall/escape tick never collects it.
        if self.body.foot_contact {
            reward += ALIVE_BONUS;
        }
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

#[pyclass]
struct RawBeatSim {
    core: Core,
}

#[pymethods]
impl RawBeatSim {
    /// `foot_contact` (unilateral-contact spec §1 `foot_mode`): false =
    /// weld (v10 — TEST-HARNESS ONLY since 2026-08-16, production is
    /// contact-only), true = unilateral frictional ground contact on
    /// the 8 sole corners.
    #[new]
    #[pyo3(signature = (max_steps=690, effort_coef=EFFORT_COEF_DEFAULT,
                        wide_acc=false, gears=3, foot_contact=false,
                        fall_const=5.0, foot_z_coef=0.0,
                        foot_x_coef=0.0, foot_hinge_r=0.0,
                        foot_shape_coef=0.0, iso_coef=0.0,
                        limit_coef=0.0, limit_arms=false))]
    fn new(max_steps: u32, effort_coef: f64, wide_acc: bool,
           gears: u8, foot_contact: bool, fall_const: f64,
           foot_z_coef: f64, foot_x_coef: f64, foot_hinge_r: f64,
           foot_shape_coef: f64, iso_coef: f64, limit_coef: f64,
           limit_arms: bool) -> Self {
        let mut core = Core::new_core(max_steps, effort_coef, wide_acc,
                                      gears, foot_contact);
        core.fall_const = fall_const;
        core.foot_z_coef = foot_z_coef;
        core.foot_x_coef = foot_x_coef;
        core.foot_hinge_r = foot_hinge_r;
        core.foot_shape_coef = foot_shape_coef;
        core.iso_coef = iso_coef;
        core.limit_coef = limit_coef;
        core.limit_arms = limit_arms;
        RawBeatSim { core }
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

    /// World center of mass from the §1.2 mass model (introspection —
    /// balance forensics). Σ mᵢ (pᵢ + Rᵢ·comᵢ) / M over the base +
    /// every mass-carrying frame, current FK.
    fn com(&self) -> [f64; 3] {
        let b = &self.core.body;
        let mut acc = [0.0f64; 3];
        let mut mass = MODEL.base_mass;
        let r0 = dynamics::fk::quat_to_rot(b.st.quat);
        for k in 0..3 {
            // base com is in the root frame
            acc[k] = MODEL.base_mass
                * (b.st.p[k]
                    + r0[k][0] * MODEL.base_com[0]
                    + r0[k][1] * MODEL.base_com[1]
                    + r0[k][2] * MODEL.base_com[2]);
        }
        for (f, fr) in MODEL.frames.iter().enumerate() {
            if fr.mass == 0.0 {
                continue;
            }
            mass += fr.mass;
            let (p, r) = (&b.ws.p[f], &b.ws.r[f]);
            for k in 0..3 {
                acc[k] += fr.mass
                    * (p[k]
                        + r[k][0] * fr.com[0]
                        + r[k][1] * fr.com[1]
                        + r[k][2] * fr.com[2]);
            }
        }
        [acc[0] / mass, acc[1] / mass, acc[2] / mass]
    }

    /// World positions of the 8 sole corners (contact spec §2 normative
    /// order), current FK. Introspection — support-polygon forensics.
    fn sole_corners(&self) -> Vec<[f64; 3]> {
        use dynamics::collide::{CORNERS_PER_FOOT, GROUND_CORNERS, GROUND_FRAMES};
        let b = &self.core.body;
        let mut out = Vec::with_capacity(8);
        for f in 0..GROUND_FRAMES.len() {
            let frame = GROUND_FRAMES[f];
            let (p, r) = (&b.ws.p[frame], &b.ws.r[frame]);
            for k in 0..CORNERS_PER_FOOT {
                let c = &GROUND_CORNERS[k];
                out.push([
                    p[0] + r[0][0] * c[0] + r[0][1] * c[1] + r[0][2] * c[2],
                    p[1] + r[1][0] * c[0] + r[1][1] * c[1] + r[1][2] * c[2],
                    p[2] + r[2][0] * c[0] + r[2][1] * c[1] + r[2][2] * c[2],
                ]);
            }
        }
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
    fn head_pose(&self) -> [f64; 7] {
        let ws = &self.core.body.ws;
        let p = ws.p[F_HEAD];
        let q = dynamics::fk::rot_to_quat(&ws.r[F_HEAD]);
        [p[0], p[1], p[2], q[0], q[1], q[2], q[3]]
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

    /// Contact-mode twin of `weld_impulses` (replay fix 2026-08-17):
    /// tick-summed GROUND-CONTACT reaction impulse per foot [L, R],
    /// [jx, jy, jz] (N·s, world frame) — jy sums the sole corners'
    /// normal rows, jx/jz their friction rows. Vertical reaction force
    /// ≈ jy / DT. Zero under welds (no ground rows exist there).
    fn ground_reaction(&self) -> [[f64; 3]; 2] {
        self.core.body.foot_grf
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
        body.body_pt_vel = [[0.0; 3]; N_BODY_PTS];
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

    /// PRE-integration root-velocity-envelope engagements since the last
    /// pop (SPEC-v10 Divergences §23), counted per (env, substep) — the
    /// CUDA backend's pop_envelope_hits mirror. Returns the count and
    /// zeroes it. 0 on every healthy trajectory; nonzero means at least
    /// one tick's episode was terminated by the C backstop.
    fn pop_envelope_hits(&mut self) -> u64 {
        let h = self.core.body.ws.env_hits;
        self.core.body.ws.env_hits = 0;
        h
    }

    /// FALL terminations since the last pop (unilateral-contact spec
    /// §4) — the CUDA backend's `pop_fall_hits` mirror, same shape as
    /// `pop_envelope_hits`. Returns the count and zeroes it. Always 0 in
    /// weld mode (`foot_contact=False`), where the feet cannot let go.
    fn pop_fall_hits(&mut self) -> u64 {
        let h = self.core.body.fall_hits;
        self.core.body.fall_hits = 0;
        h
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
    fn pop_limit_rest(&mut self) -> (f64, f64, f64, u64) {
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
            acc.merge(&self.core.body.ws.stats);
            self.core.body.ws.stats = DbgStats::default();
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
    /// twin. Always false in weld mode.
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
            7 => {}
            // 7 = head-in-wall drain tick: NO combo effect (spec §12b —
            // only the ENTRY breaks; without this arm the catch-all
            // below would have broken combo every tick inside a wall)
            _ => {
                // slow / angle / wrong_hand / miss / bomb / wall ENTRY
                // (8): combo break
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
    /// `foot_contact` (unilateral-contact spec §1 `foot_mode`): false =
    /// weld (v10 — TEST-HARNESS ONLY since 2026-08-16, production is
    /// contact-only), true = unilateral frictional ground contact on
    /// the 8 sole corners.
    #[new]
    #[pyo3(signature = (n, max_steps=690, effort_coef=EFFORT_COEF_DEFAULT,
                        combo_walk=false, return_events=true, wide_acc=false,
                        gears=3, foot_contact=false, fall_const=5.0,
                        foot_z_coef=0.0, foot_x_coef=0.0,
                        foot_hinge_r=0.0, foot_shape_coef=0.0,
                        iso_coef=0.0, limit_coef=0.0,
                        limit_arms=false))]
    #[allow(clippy::too_many_arguments)]
    fn new(n: usize, max_steps: u32, effort_coef: f64, combo_walk: bool,
           return_events: bool, wide_acc: bool, gears: u8,
           foot_contact: bool, fall_const: f64, foot_z_coef: f64,
           foot_x_coef: f64, foot_hinge_r: f64,
           foot_shape_coef: f64, iso_coef: f64, limit_coef: f64,
           limit_arms: bool) -> Self {
        VecBeatSim {
            sims: (0..n)
                .map(|_| {
                    let mut c = Core::new_core(max_steps, effort_coef,
                                               wide_acc, gears, foot_contact);
                    c.fall_const = fall_const;
                    c.foot_z_coef = foot_z_coef;
                    c.foot_x_coef = foot_x_coef;
                    c.foot_hinge_r = foot_hinge_r;
                    c.foot_shape_coef = foot_shape_coef;
                    c.iso_coef = iso_coef;
                    c.limit_coef = limit_coef;
                    c.limit_arms = limit_arms;
                    c
                })
                .collect(),
            walks: combo_walk.then(|| (0..n).map(|_| ComboWalk::new(0)).collect()),
            return_events,
        }
    }

    #[pyo3(signature = (i, chart, walls = Vec::new()))]
    fn set_chart(&mut self, i: usize, chart: Vec<ChartTuple>, walls: Vec<WallTuple>) {
        let n_objects = chart.len();
        self.sims[i].set_walls(walls);
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

    /// env v19 flail telemetry for env i, THIS episode: (blade arc in
    /// rad accrued by follow-through-idle hands on note-free ticks,
    /// count of note-free ticks). Read at done, before set_chart
    /// resets it — same lifecycle as counters(). ADDITIVE getter.
    fn flail(&self, i: usize) -> (f64, u32) {
        let s = &self.sims[i];
        (s.flail_rad, s.flail_ticks)
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

    /// PRE-integration root-velocity-envelope engagements summed over
    /// all envs since the last pop (SPEC-v10 Divergences §23) — the
    /// CUDA backend's pop_envelope_hits mirror. Returns the count and
    /// zeroes the per-env counters. ADDITIVE getter; 0 on every healthy
    /// trajectory.
    fn pop_envelope_hits(&mut self) -> u64 {
        self.sims
            .iter_mut()
            .map(|s| {
                let h = s.body.ws.env_hits;
                s.body.ws.env_hits = 0;
                h
            })
            .sum()
    }

    /// env v34 limit-rest telemetry summed over all envs since the last
    /// pop: (legs λ-flow N·m·s, spine λ-flow, env-ticks). CUDA
    /// `pop_limit_rest` mirror.
    fn pop_limit_rest(&mut self) -> (f64, f64, f64, u64) {
        let (mut lg, mut sp, mut ar, mut tk) = (0.0f64, 0.0f64, 0.0f64, 0u64);
        for s in self.sims.iter_mut() {
            lg += s.limit_rest_sum[0];
            sp += s.limit_rest_sum[1];
            ar += s.limit_rest_sum[2];
            tk += s.limit_rest_ticks;
            s.limit_rest_sum = [0.0; 3];
            s.limit_rest_ticks = 0;
        }
        (lg, sp, ar, tk)
    }

    /// FALL terminations summed over all envs since the last pop
    /// (unilateral-contact spec §4) — the CUDA backend's `pop_fall_hits`
    /// mirror, same shape as `pop_envelope_hits`. Returns the count and
    /// zeroes the per-env counters. ADDITIVE getter; always 0 in weld
    /// mode.
    fn pop_fall_hits(&mut self) -> u64 {
        self.sims
            .iter_mut()
            .map(|s| {
                let h = s.body.fall_hits;
                s.body.fall_hits = 0;
                h
            })
            .sum()
    }

    /// TILT terminations summed over all envs since the last pop
    /// (2026-08-20) — the CUDA backend's `pop_tilt_hits` mirror.
    /// ADDITIVE getter; always 0 in weld mode.
    fn pop_tilt_hits(&mut self) -> u64 {
        self.sims
            .iter_mut()
            .map(|s| {
                let h = s.body.tilt_hits;
                s.body.tilt_hits = 0;
                h
            })
            .sum()
    }

    /// Foot-lift events summed over all envs since the last pop
    /// (2026-08-20), (left, right) — the CUDA backend's
    /// `pop_foot_lifts` mirror. ADDITIVE getter; (0, 0) in weld mode.
    fn pop_foot_lifts(&mut self) -> (u64, u64) {
        let (mut l, mut r) = (0u64, 0u64);
        for s in self.sims.iter_mut() {
            let h = s.body.foot_lift_hits;
            s.body.foot_lift_hits = [0, 0];
            l += h[0];
            r += h[1];
        }
        (l, r)
    }

    /// Fall terminations in env i's current episode (contact spec §4).
    fn falls(&self, i: usize) -> u32 {
        self.sims[i].falls
    }

    /// Tilt terminations in env i's current episode (2026-08-20).
    fn tilts(&self, i: usize) -> u32 {
        self.sims[i].tilts
    }

    /// env v17: head-in-wall ticks in env i's current episode (each
    /// charged WALL_PEN). ADDITIVE getter (spec §12b).
    fn walls_in(&self, i: usize) -> u32 {
        self.sims[i].walls_in
    }

    /// env v17: wall entries (combo breaks) in env i's episode.
    fn wall_enters(&self, i: usize) -> u32 {
        self.sims[i].wall_enters
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
            acc.merge(&s.body.ws.stats);
            s.body.ws.stats = DbgStats::default();
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
        let mut body = Body::new(false);
        body.tick(&acts, &gear);
        assert!(effort > 0.0, "driven joints must do positive work");
        assert_eq!(body.effort_j, effort, "effort expression trees diverge");
        // trajectory parity too (the mirror IS the pipeline)
        for j in 0..NQ {
            assert_eq!(body.st.q[j], st.q[j]);
        }
        let mut b2 = Body::new(false);
        b2.tick(&[0u8; NQ], &gear);
        assert_eq!(b2.effort_j, 0.0, "coast must be exactly free");
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
        let mut core = Core::new_core(100_000_000, 4e-4, false, 5, false);
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
            .map(|i| (1.0 + i as f64 * 0.25, 0.0, 1.0, i % 2, 0.0, 6.0,
                      false, KIND_NOTE, false, false))
            .collect();
        let mut core = Core::new_core(690, 4e-4, false, 5, false);
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

    /// C backstop (SPEC-v10 Divergences §23): an env whose root leaves
    /// the physical envelope during a tick terminates with a ZEROED
    /// reward, and the engagement is counted. The fixture is escape #7
    /// of the 2026-08-03 corpus (see dynamics::tests_soak) — a state the
    /// pre-A1 engine teleported 642.7 m from.
    #[test]
    fn escaped_env_terminates_with_zero_reward() {
        use crate::dynamics::tests_soak::{ESC_A, ESC_P, ESC_Q, ESC_QUAT, ESC_V};
        // a real (non-empty) chart so `done` cannot come from the
        // empty-chart disjunct; the notes are far in the future
        let chart: Vec<ChartTuple> = (0u8..3)
            .map(|i| {
                (8.0 + i as f64 * 0.25, 0.0, 1.0, i % 2, 0.0, 6.0, false,
                 KIND_NOTE, false, false)
            })
            .collect();
        // effort_coef > 0 so the physics reward would otherwise be
        // strictly negative — the zeroing is observable
        let mut core = Core::new_core(690, 4e-4, false, 3, false);
        core.set_chart_notes(chart);
        core.body.st.p = ESC_P;
        core.body.st.quat = ESC_QUAT;
        core.body.st.q.copy_from_slice(&ESC_Q);
        core.body.st.v.copy_from_slice(&ESC_V);
        core.body.refresh_fk();
        let p0 = core.body.st.p;

        let (reward, done) = core.step_inner(&ESC_A);
        assert!(core.body.escaped, "the escape state must trip the envelope");
        assert!(done, "an escaped env must terminate");
        assert_eq!(reward, 0.0, "an escaped tick's reward must be zeroed");
        assert!(core.finished, "the terminal latch must be set");
        assert!(core.body.ws.env_hits > 0, "the engagement must be counted");
        // A1's bound, at the env layer: 8 substeps x h*ROOT_V_MAX
        let dp = ((core.body.st.p[0] - p0[0]).powi(2)
            + (core.body.st.p[1] - p0[1]).powi(2)
            + (core.body.st.p[2] - p0[2]).powi(2))
        .sqrt();
        assert!(
            dp <= 8.0 * H_SUB * dynamics::step::ROOT_V_MAX + 1e-9,
            "tick |dp| = {dp} m — the teleport survived"
        );
        assert!(
            core.body.st.p.iter().all(|x| x.is_finite())
                && core.body.st.v.iter().all(|x| x.is_finite()),
            "escaped state must still be finite"
        );
        // popping zeroes, pd_hits-style
        let popped = core.body.ws.env_hits;
        core.body.ws.env_hits = 0;
        assert!(popped > 0);
    }

    /// The other half of the contract: an ordinary episode never
    /// terminates via the C backstop and never counts an engagement.
    #[test]
    fn healthy_env_never_escapes() {
        let chart: Vec<ChartTuple> = (0u8..3)
            .map(|i| {
                (1.0 + i as f64 * 0.25, 0.0, 1.0, i % 2, 0.0, 6.0, false,
                 KIND_NOTE, false, false)
            })
            .collect();
        let mut core = Core::new_core(690, 4e-4, false, 5, false);
        core.set_chart_notes(chart);
        let mut acts = [0u8; NQ];
        acts[10] = 1;
        acts[3] = 2;
        acts[15] = 1;
        for _ in 0..120 {
            let (_r, d) = core.step_inner(&acts);
            assert!(!core.body.escaped, "healthy motion must never escape");
            if d {
                break;
            }
        }
        assert_eq!(core.body.ws.env_hits, 0);
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
                (8.0 + i as f64 * 0.25, 0.0, 1.0, i % 2, 0.0, 6.0, false,
                 KIND_NOTE, false, false)
            })
            .collect();
        chart.push((9.0, 0.0, 1.0, 0, 0.0, 6.0, false, KIND_BOMB,
                    false, false));
        let mut core = Core::new_core(690, 4e-4, false, 5, true);
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
    /// feet ≈ weight x DT while the body stands — and the weld lanes
    /// must stay zero. Regression for the all-zero "feet" bug: the
    /// recorder read the retired welds' impulses in contact mode.
    #[test]
    fn ground_reaction_carries_weight_in_contact_mode() {
        let chart: Vec<ChartTuple> = vec![(8.0, 0.0, 1.0, 0, 0.0, 6.0,
                                           false, KIND_NOTE, false, false)];
        let mut core = Core::new_core(690, 4e-4, false, 5, true);
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
        let weight_dt = 74.50 * dynamics::model::G / 30.0;
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
        assert!(
            core.body.weld_imp.iter().all(|v| *v == 0.0),
            "weld lanes must stay zero in contact mode"
        );
    }

    /// The other half of the §5 (d) / §5 (f) contract: in WELD mode
    /// (`foot_contact = false`, the default) the fall test never fires —
    /// the feet cannot let go, and the flag must not perturb the v10 env.
    #[test]
    fn weld_mode_never_falls() {
        let chart: Vec<ChartTuple> = (0u8..3)
            .map(|i| {
                (8.0 + i as f64 * 0.25, 0.0, 1.0, i % 2, 0.0, 6.0, false,
                 KIND_NOTE, false, false)
            })
            .collect();
        let mut core = Core::new_core(690, 4e-4, false, 5, false);
        core.set_chart_notes(chart);
        let mut acts = [0u8; NQ];
        acts[17] = 1;
        acts[23] = 1;
        for _ in 0..30 {
            let (_r, d) = core.step_inner(&acts);
            assert!(!core.body.fell, "weld mode must never report a fall");
            if d {
                break;
            }
        }
        assert_eq!(core.falls, 0);
        assert_eq!(core.body.fall_hits, 0);
    }

    /// Live gates at the reference pose reproduce the v9 constants:
    /// reach 1.75, miss boundary −0.65 (SPEC-v10 §4).
    #[test]
    fn live_gates_reference_values() {
        let body = Body::new(false);
        assert!((body.reach_z() - 1.75).abs() < 1e-12);
        assert!((body.head[2] - (-0.65)).abs() < 1e-12);
    }
}
