//! SPEC-v10 §1 model tables — the SINGLE source of truth for the v10
//! kinematic tree: parent indices, joint axes, stacking orders, anchors,
//! masses/COMs/inertias, tau_max, omega caps, damping, joint limits.
//! Every literal is normative from SPEC-v10 §1.1–1.5 as written; nothing
//! here is re-derived, and no other module carries its own copy (the v9
//! "6-initializer zero-fill" bug class is the cautionary tale).

use crate::Real;
/// One kinematic frame: a revolute DoF (`coord >= 0`) or a welded body
/// frame (`coord == -1`; head only, in the humanoid). Frames are listed
/// in CHAIN (stacking, topological) order; `coord` maps each frame to
/// its ACTUATED index (§1.4). Stacking order != actuated order for the
/// shoulder (stack yaw→pitch→twist, actuated pitch,yaw,twist) and wrist
/// (stack roll→dev→bend, actuated roll,bend,dev) clusters — both
/// orderings are normative (§1.3).
#[derive(Clone, Copy)]
pub struct Frame {
    pub parent: i32,       // frame index; -1 = base (pelvis, or world)
    pub anchor: [Real; 3],  // frame origin offset in the PARENT frame
    pub axis: [Real; 3],    // revolute axis (unit, local); fixed: unused
    pub coord: i32,        // actuated coordinate index; -1 = welded
    pub mass: Real,         // body mass attached to this frame (0 = none)
    pub com: [Real; 3],     // in this frame
    pub inertia: [Real; 3], // principal diagonal about the COM, frame axes
}

/// A rigid-body model the dynamics core is generic over: the humanoid
/// below, or ad-hoc test tables (pendulums). Per-coordinate arrays are
/// indexed in ACTUATED order (§1.4).
#[derive(Clone, Copy)]
pub struct Model {
    pub free_base: bool, // 6-DoF free joint on the base body
    pub base_mass: Real,  // base body (pelvis); 0 for fixed-base models
    pub base_com: [Real; 3],
    pub base_inertia: [Real; 3],
    pub base_pos0: [Real; 3], // reference (neutral) base position
    pub frames: &'static [Frame],
    pub nq: usize, // actuated coordinate count
    pub limits: &'static [(Real, Real)],
    pub omega_cap: &'static [Real],
    pub tau_max: &'static [Real],
    pub damping: &'static [Real], // b: tau_damp = -b * qdot (§1.5)
    // P1 divergence (SPEC-v10 Divergences §12): per-DoF armature
    // (reflected rotor) inertia, added to M's diagonal AND to RNEA's
    // qdd term (MuJoCo semantics — the MJCF twin mirrors it via the
    // `armature` attribute). Without it M(q) is exactly SINGULAR at
    // reachable poses: the shoulder stack yaw→pitch→twist gimbal-locks
    // at pitch ±90°, and l_arm_pitch's lower limit IS −90° — at the
    // clamp the yaw and twist world axes are anti-parallel through the
    // same point (null vector yaw=twist=1), and the §3 no-pivot
    // Cholesky dies. armature = I_ref/100 (1 % reflected inertia).
    pub armature: &'static [Real],
    pub gravity: [Real; 3],
    /// Left/right MIRROR map of the actuated coordinates (an involution,
    /// actuated order): `mirror[j]` is the coordinate the x → −x
    /// reflection maps j onto — j itself for a central joint. The
    /// solver's sequential passes (joint-limit rows, ω caps, step-11
    /// clamps) visit the coordinates orbit by orbit and update the two
    /// members of a mirror pair JOINTLY from the same pre-state, so no
    /// side is ever solved first (sim/mirror-exact, 2026-10-04).
    /// Identity for models without a mirror symmetry.
    pub mirror: &'static [usize],
    /// Passive wrist "tendon" torque (exp/wrist-tendon, 2026-10-05):
    /// a conservative, state-dependent joint torque on each wrist's
    /// (bend, dev) pair, added to the applied side of the substep RHS
    /// next to τ_gear − b∘q̇ (step.rs step 4). Some on HUMANOID; None on
    /// the test pendulums and other test tables. See `PassiveWrist`.
    pub passive_wrist: Option<PassiveWrist>,
    /// Passive shoulder axial-rotation limit (exp/shoulder-rom,
    /// 2026-10-07): a conservative, arm-direction-dependent torque on
    /// each shoulder's (yaw, pitch, twist) plus implicit edge damping on
    /// the axial rate, added next to the wrist tendon (step.rs step 4d).
    /// Some on HUMANOID; None on the test tables. See `PassiveShoulder`
    /// and docs/design/shoulder-rom.md.
    pub passive_shoulder: Option<PassiveShoulder>,
}

/// Passive wrist tendon parameters (exp/wrist-tendon, 2026-10-05).
///
/// Per hand, from the CURRENT (q_bend, q_dev) of every substep:
///   u_j = q_j / L_j,  L_j = hi_j if q_j ≥ 0 else |lo_j|  (the limit the
///         joint is heading to, read from `Model::limits` — mirror-exact
///         because the right hand's dev limits are the left's negated)
///   r   = sqrt(u_b² + u_d²)                         (elliptical radius)
///   E   = P(r)·(Ē_b·u_b² + Ē_d·u_d²)/r²,  P(r) = ((r − R0)/(1 − R0))³
///         for r > R0 = 0.5, else 0 — UNCAPPED past r = 1 (deliberate:
///         box corners like full bend + full dev, anatomically
///         unreachable, become effectively inaccessible)
///   Ē_j = τ_peak_j·L_j·(1 − R0)/3  (the pure-axis energy stored at the
///         limit: ∫ τ_peak·((u − R0)/(1 − R0))² L du over [R0, 1])
///   τ_j = −∂E/∂q_j  (analytic; `wrist_tendon`)
/// On a pure axis this is exactly τ = −sign(u)·τ_peak·((|u| − R0)/(1 − R0))²:
/// zero inside half the range, τ_peak at the limit. PLUS (2026-10-05)
/// additive per-joint centring terms (`tendon_centring`) that move each
/// joint's pure-axis zone start to s_j (bend 0, dev 0): pure axis
/// τ = −sign·τ_peak·((|u| − s)/(1 − s))²₊, corners unchanged. τ_peak is FIXED in
/// N·m (0.30 × the DEFAULT body's τmax, `TENDON_TAU_PEAK_*`) — not
/// scaled by the per-env tau randomization or a bench body: tendon
/// stiffness is not muscle strength.
#[derive(Clone, Copy, Debug)]
pub struct PassiveWrist {
    /// per hand [L, R]: (bend coordinate, dev coordinate), actuated order
    pub hands: [(usize, usize); 2],
    pub tau_peak_bend: Real,
    pub tau_peak_dev: Real,
    /// Per-joint tendon zone starts (normalized excursion) of the
    /// additive centring terms (`tendon_centring`): bend 0, dev 0.
    pub zone_start_bend: Real,
    pub zone_start_dev: Real,
    /// Edge-zone damping peaks (issue #61, 2026-10-05): per joint
    /// τ_d,j = −b_j·q̇_j, b_j = b_peak_j·[x_ell² + c(u_j)] with x_ell =
    /// (r − R0)/(1 − R0) for r > R0 (the tendon's radius; uncapped past
    /// r = 1), else 0, and the additive per-joint centring c (2026-10-06,
    /// `damp_centring`: pure axis b_peak·u² from neutral, corners
    /// unchanged). Diagonal and ≥ 0,
    /// so strictly dissipative. Applied IMPLICITLY (h·b on M's diagonal
    /// + −b·q̇ on the RHS — step.rs 4c; the explicit b·h/I_eff reaches
    /// ~0.4 at the corners and > 1 on reachable overshoot states).
    pub b_peak_bend: Real,
    pub b_peak_dev: Real,
    /// The per-joint damping centring term (2026-10-06) on — true on
    /// HUMANOID (the CUDA kernel has it unconditionally). false gives the
    /// 2026-10-05 law b = b_peak·x_ell² (tests / A-B diagnostics only).
    pub damp_from_zero: bool,
}

/// Tendon zone onset (normalized radius) and the τ_peak fraction.
pub const TENDON_R0: Real = 0.5;
pub const TENDON_FRAC: f64 = 0.30;
/// The DEFAULT body's wrist τmax scales (python constants.TAU_BODY_DEFAULT
/// axes wrist_bend / wrist_dev — the exp/bend7 default body), restated
/// here because the Rust sim has no default-body table of its own
/// (constants.py asserts the two agree).
pub const TAU_BODY_DEFAULT_WRIST_BEND: f64 = 1.4466;
pub const TAU_BODY_DEFAULT_WRIST_DEV: f64 = 1.0220;
/// τ_peak = 0.30 × default-body τmax, in N·m, computed in f64 and
/// rounded once to Real (the CUDA kernel's `T(double constexpr)`):
/// bend 0.30 × 1.431 × 1.4466 = 0.62103 N·m, dev 0.30 × 1.06 × 1.0220 =
/// 0.32500 N·m.
pub const TENDON_TAU_PEAK_BEND: Real = (TENDON_FRAC * 1.431 * TAU_BODY_DEFAULT_WRIST_BEND) as Real;
pub const TENDON_TAU_PEAK_DEV: Real = (TENDON_FRAC * 1.06 * TAU_BODY_DEFAULT_WRIST_DEV) as Real;
/// Per-joint tendon zone starts (2026-10-05, user): the pure-axis torque
/// starts at |u| = s — bend 0, dev 0 (both always centring; bend was 0.1
/// for a few hours on 2026-10-05) — via additive centring terms on top of
/// the R0 = 0.5 elliptical coupling.
pub const TENDON_S_BEND: Real = 0.0;
pub const TENDON_S_DEV: Real = 0.0;
/// Edge-damping reference speed (issue #61): at r = 1 and |q̇| = 30 rad/s
/// the damping torque equals the joint's tendon τ_peak, i.e.
/// b_peak = τ_peak / 30 — bend 0.020701, dev 0.010833 N·m·s/rad. Derived
/// from the τ_peak expressions in f64, rounded once to Real (the .cu's
/// `T(double constexpr)`); fixed, not scaled by tau draws / tau_fit.
pub const TENDON_DAMP_REF_SPEED: f64 = 30.0;
pub const TENDON_B_PEAK_BEND: Real =
    (TENDON_FRAC * 1.431 * TAU_BODY_DEFAULT_WRIST_BEND / TENDON_DAMP_REF_SPEED) as Real;
pub const TENDON_B_PEAK_DEV: Real =
    (TENDON_FRAC * 1.06 * TAU_BODY_DEFAULT_WRIST_DEV / TENDON_DAMP_REF_SPEED) as Real;

/// Edge-zone shape x² of the radius r: ((r − R0)/(1 − R0))² for r > R0,
/// else 0 (uncapped). The edge damping is b_peak_j·tendon_edge_x2(r).
#[inline]
pub fn tendon_edge_x2(r: Real) -> Real {
    if r > TENDON_R0 {
        let x = (r - TENDON_R0) / (1.0 - TENDON_R0);
        x * x
    } else {
        0.0
    }
}

/// Distance to the limit a wrist coordinate is heading to.
#[inline(always)]
fn tendon_lim(q: Real, lim: (Real, Real)) -> Real {
    if q >= 0.0 { lim.1 } else { -lim.0 }
}

/// Per-joint centring part of the edge damping (2026-10-06, user: the
/// damping starts at 0 too): with a = |u|,
///   c(a) = max(a² − ((a − R0)/(1 − R0))²₊, 0)
/// ADDITIVE to the elliptical x_ell(r)², so a pure axis gets b_peak·a²
/// from neutral (b_peak at the limit) while the box corners keep their
/// 3.34·b_peak. ≥ 0 on [0, 1] (the bracket is (1 − a)(3a − 1) on
/// [R0, 1]); the max(·, 0) only engages on overshoot past a stop
/// (a > 1), so b never goes negative.
#[inline(always)]
fn damp_centring(u: Real) -> Real {
    let a = u.abs();
    let y = if a > TENDON_R0 { (a - TENDON_R0) / (1.0 - TENDON_R0) } else { 0.0 };
    (a * a - y * y).max(0.0)
}

/// Edge-zone damping coefficients (b_bend, b_dev) of one wrist at
/// (q_b, q_d) with tendon radius r: b_j = b_peak_j·[x_ell(r)² + c(u_j)]
/// (`tendon_edge_x2` + `damp_centring`). Diagonal, ≥ 0 — strictly
/// dissipative. The .cu `tendon_edge_b` transcribes this tree.
#[inline]
pub fn tendon_edge_b(qb: Real, qd: Real, lim_b: (Real, Real), lim_d: (Real, Real), r: Real,
                     bpb: Real, bpd: Real) -> (Real, Real) {
    let x2 = tendon_edge_x2(r);
    let cb = damp_centring(qb / tendon_lim(qb, lim_b));
    let cd = damp_centring(qd / tendon_lim(qd, lim_d));
    (bpb * (x2 + cb), bpd * (x2 + cd))
}

/// Passive tendon torque on one wrist: (τ_bend, τ_dev, r) from the
/// current angles, the two joints' limits and the two τ_peaks. Zero
/// torque for r ≤ R0 (E ≡ 0 there, which also keeps r → 0 safe).
/// Gradient of E (S = Ē_b u_b² + Ē_d u_d², ρ = r²):
///   ∂E/∂u_b = u_b·[P'(r)·S/(r·ρ) + 2P·(Ē_b − Ē_d)·u_d²/ρ²]
///   ∂E/∂u_d = u_d·[P'(r)·S/(r·ρ) + 2P·(Ē_d − Ē_b)·u_b²/ρ²]
///   τ_j = −(∂E/∂u_j)/L_j.
/// The CUDA kernel (`wrist_tendon`) transcribes this expression tree.
#[inline]
pub fn wrist_tendon(qb: Real, qd: Real, lim_b: (Real, Real), lim_d: (Real, Real),
                    tpb: Real, tpd: Real, sb: Real, sd: Real) -> (Real, Real, Real) {
    let lb = tendon_lim(qb, lim_b);
    let ld = tendon_lim(qd, lim_d);
    let ub = qb / lb;
    let ud = qd / ld;
    let ub2 = ub * ub;
    let ud2 = ud * ud;
    let rho = ub2 + ud2;
    let r = rho.sqrt();
    let tcb = tendon_centring(qb, ub, tpb, sb);
    let tcd = tendon_centring(qd, ud, tpd, sd);
    if !(r > TENDON_R0) {
        return (tcb, tcd, r);
    }
    let x = (r - TENDON_R0) / (1.0 - TENDON_R0);
    let p = x * x * x;
    let dp = 3.0 * x * x / (1.0 - TENDON_R0);
    let eb = tpb * lb * ((1.0 - TENDON_R0) / 3.0);
    let ed = tpd * ld * ((1.0 - TENDON_R0) / 3.0);
    let s = eb * ub2 + ed * ud2;
    let k = dp * s / (r * rho);
    let w = 2.0 * p / (rho * rho);
    let gb = ub * (k + w * (eb - ed) * ud2);
    let gd = ud * (k + w * (ed - eb) * ub2);
    (-gb / lb + tcb, -gd / ld + tcd, r)
}

/// Per-joint CENTRING torque (2026-10-05, user: per-joint tendon zone
/// starts — "always trying to center the hand"), ADDITIVE to the
/// elliptical term. Joint zone start s (bend 0, dev 0), a = |u|:
///   τ_c = −sign(q)·τ_peak·[((a − s)/(1 − s))²₊ − ((a − R0)/(1 − R0))²₊]
///   E_c = τ_peak·L·[((1 − s)/3)·((a − s)/(1 − s))³₊
///                   − ((1 − R0)/3)·((a − R0)/(1 − R0))³₊]
/// so a pure axis totals exactly τ_peak·((a − s)/(1 − s))²₊ (the
/// ellipse's R0-onset ramp plus this). The bracket is ≥ 0 for a ≤ 1 and
/// ZERO at the limit, so the box-corner torques are unchanged; C¹ at
/// a = s (quadratic onset) and zero gradient at q = 0 for s ≥ 0. Past the
/// stop (a > 1, one substep of overshoot) the same expressions hold
/// (still conservative). Sign by q with an exact-negation tree:
/// mirror-exact. s = 0 makes (a − s)/(1 − s) exactly a.
#[inline(always)]
fn tendon_centring(q: Real, u: Real, tp: Real, s: Real) -> Real {
    let a = u.abs();
    let y0 = if a > s { (a - s) / (1.0 - s) } else { 0.0 };
    let y = if a > TENDON_R0 { (a - TENDON_R0) / (1.0 - TENDON_R0) } else { 0.0 };
    let c = tp * (y0 * y0 - y * y);
    if q >= 0.0 { -c } else { c }
}

/// Centring energy of one joint (tests / diagnostics; see tendon_centring).
fn tendon_centring_energy(u: Real, l: Real, tp: Real, s: Real) -> Real {
    let a = u.abs();
    let y0 = if a > s { (a - s) / (1.0 - s) } else { 0.0 };
    let y = if a > TENDON_R0 { (a - TENDON_R0) / (1.0 - TENDON_R0) } else { 0.0 };
    tp * l * (((1.0 - s) / 3.0) * y0 * y0 * y0 - ((1.0 - TENDON_R0) / 3.0) * y * y * y)
}

/// The tendon energy E(q_b, q_d) (J) — tests / diagnostics only.
pub fn wrist_tendon_energy(qb: Real, qd: Real, lim_b: (Real, Real), lim_d: (Real, Real),
                           tpb: Real, tpd: Real, sb: Real, sd: Real) -> Real {
    let lb = tendon_lim(qb, lim_b);
    let ld = tendon_lim(qd, lim_d);
    let (ub, ud) = (qb / lb, qd / ld);
    let rho = ub * ub + ud * ud;
    let r = rho.sqrt();
    // per-joint centring energies (additive; see tendon_centring)
    let ec = tendon_centring_energy(ub, lb, tpb, sb) + tendon_centring_energy(ud, ld, tpd, sd);
    if !(r > TENDON_R0) {
        return ec;
    }
    let x = (r - TENDON_R0) / (1.0 - TENDON_R0);
    let eb = tpb * lb * ((1.0 - TENDON_R0) / 3.0);
    let ed = tpd * ld * ((1.0 - TENDON_R0) / 3.0);
    x * x * x * (eb * ub * ub + ed * ud * ud) / rho + ec
}

impl PassiveWrist {
    /// Tendon torques of both hands at pose `q` for `model`'s limits:
    /// [(τ_bend, τ_dev, r); 2].
    #[inline]
    pub fn eval(&self, limits: &[(Real, Real)], q: &[Real]) -> [(Real, Real, Real); 2] {
        let mut out = [(0.0, 0.0, 0.0); 2];
        for (s, &(jb, jd)) in self.hands.iter().enumerate() {
            out[s] = wrist_tendon(q[jb], q[jd], limits[jb], limits[jd],
                                  self.tau_peak_bend, self.tau_peak_dev,
                                  self.zone_start_bend, self.zone_start_dev);
        }
        out
    }
}

/// HUMANOID's tendon: wrists (8 bend, 9 dev) and (15 bend, 16 dev).
pub const HUMANOID_PASSIVE_WRIST: PassiveWrist = PassiveWrist {
    hands: [(8, 9), (15, 16)],
    tau_peak_bend: TENDON_TAU_PEAK_BEND,
    tau_peak_dev: TENDON_TAU_PEAK_DEV,
    zone_start_bend: TENDON_S_BEND,
    zone_start_dev: TENDON_S_DEV,
    b_peak_bend: TENDON_B_PEAK_BEND,
    b_peak_dev: TENDON_B_PEAK_DEV,
    damp_from_zero: true,
};

// ---- exp/shoulder-rom (2026-10-07): passive shoulder axial limit -----------
//
// docs/design/shoulder-rom.md. Per arm, every substep, from the CURRENT
// (yaw y, pitch p, twist t) in LEFT-CANONICAL coordinates (the right arm
// is evaluated at (−yaw, pitch, −twist) and its yaw / twist torques are
// negated — the sim's mirror map, so mirror-exactness holds by
// construction, with no reliance on sin(−x) == −sin(x)):
//   d_l = −cos p sin y, d_f = cos p cos y   (humerus lateral / forward)
//   c   = −sin p                            (cos of the elevation θ = p + 90°)
//   φ   = y + π/2                           (plane of elevation)
//   τ   = y − t                             (axial rotation, IR +: the
//         swing-twist angle from the hanging arm = ISB plane + axial)
//   f_k = a0 + c (a1 + c a2) + B (d_l − d_f) + (1 − c) φ,  k = lo (ER), hi (IR)
//   m = (f_hi + f_lo)/2, w = (f_hi − f_lo)/2, u = (τ − m)/w, a = |u|
//   z = ((a − s)/(1 − s))₊, x = ((a − 1)/wb)₊, g = z² + kb x² (a ≤ uc),
//   g = g_c + g'_c (a − uc) past uc (linear torque: bounded stiffness)
//   E = T w G(a), G' = g, G(0) = 0
//   dE = sign(u) T g (dτ − dm) + T (G − a g) dw;  τ_joint = −dE/dq
// Pure axial torque −sign(u)·T·g: 0 inside |u| ≤ s, T at the human limit,
// steep past it. The limits are smooth in q inside the joint box (the
// hanging pole needs no special case; the (1 − c) φ gauge vanishes to
// second order there). Edge damping b = b_peak·z(min(a, uc))² on τ̇ =
// ẏ − ṫ: joint-space B = b·JᵀJ, J = (1, 0, −1) — rank 1, PSD, IMPLICIT
// (h·B onto M's (yaw, twist) block before the factor; step.rs 4d).

/// Passive shoulder parameters (exp/shoulder-rom). Fixed N·m, not scaled
/// by tau draws / tau_fit (capsule stiffness is not muscle strength).
#[derive(Clone, Copy, Debug)]
pub struct PassiveShoulder {
    /// per arm [L, R]: (yaw, pitch, twist) coordinates, actuated order
    pub arms: [(usize, usize, usize); 2],
    /// ER-side (lo) and IR-side (hi) limit coefficients [a0, a1, a2, B]
    /// (rad), left-canonical (see the header comment)
    pub coef_lo: [Real; 4],
    pub coef_hi: [Real; 4],
    pub tau_peak: Real,
    pub zone_start: Real,
    pub beyond_k: Real,
    pub beyond_w: Real,
    pub u_lin: Real,
    pub b_peak: Real,
}

/// One arm's passive-shoulder evaluation (left-canonical unless noted).
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct SromOut {
    pub tau_y: Real,
    pub tau_p: Real,
    pub tau_t: Real,
    /// normalized axial excursion (±1 at the human limits)
    pub u: Real,
    /// edge-damping coefficient on τ̇ (N·m·s/rad)
    pub b: Real,
    /// tendon energy (J) — tests / diagnostics
    pub e: Real,
}

pub const SROM_FRAC: f64 = 0.30;
/// The DEFAULT body's arm-twist τmax scale (python
/// constants.TAU_BODY_DEFAULT axis arm_twist), restated (constants.py
/// asserts the two agree).
pub const TAU_BODY_DEFAULT_ARM_TWIST: f64 = 0.8729;
/// T = 0.30 × 19.45 × 0.8729 = 5.0933715 N·m, f64 then rounded once.
pub const SROM_TAU_PEAK: Real = (SROM_FRAC * 19.45 * TAU_BODY_DEFAULT_ARM_TWIST) as Real;
pub const SROM_S: Real = 0.3;
pub const SROM_KB: Real = 1.0;
pub const SROM_WB: Real = 0.25;
pub const SROM_UC: Real = 1.25;
/// b_peak = T / 16 rad/s (the twist ω cap): at the human limit, moving at
/// the cap, the damping torque equals T. 0.318336 N·m·s/rad.
pub const SROM_DAMP_REF_SPEED: f64 = 16.0;
pub const SROM_B_PEAK: Real =
    (SROM_FRAC * 19.45 * TAU_BODY_DEFAULT_ARM_TWIST / SROM_DAMP_REF_SPEED) as Real;
/// Limit coefficients (rad), the exact 4-anchor fit (design doc §B.2):
/// τ anchors (ER, IR) side (−80°, 77°), 90° abduction (−100°, 60°), 90°
/// flexion (−25°, 130°), top θ 155° / φ 40° (−75°, 20°). In degrees:
/// lo [−107.5, 38.6243, −11.1243, 7.5], hi [50, 74.6372, −47.6372, 10].
pub const SROM_COEF_LO: [Real; 4] =
    [-1.8762289458939043, 0.6741217713310772, -0.19415622703263663, 0.1308996938995747];
pub const SROM_COEF_HI: [Real; 4] =
    [0.8726646259971648, 1.3026652970940116, -0.8314263990555426, 0.17453292519943295];

/// Pure-axis shape at |u| = a (no continuation): (g, G, g').
#[inline(always)]
fn srom_raw(a: Real, s: Real, kb: Real, wb: Real) -> (Real, Real, Real) {
    let z = if a > s { (a - s) / (1.0 - s) } else { 0.0 };
    let x = if a > 1.0 { (a - 1.0) / wb } else { 0.0 };
    let g = z * z + kb * (x * x);
    let gg = ((1.0 - s) / 3.0) * (z * z * z) + (kb * wb / 3.0) * (x * x * x);
    let dg = 2.0 * z / (1.0 - s) + 2.0 * kb * x / wb;
    (g, gg, dg)
}

/// (g, G) = (pure-axis torque / T, energy / (T w)) with the linear-torque
/// continuation past uc.
#[inline(always)]
pub fn srom_shape(a: Real, ps: &PassiveShoulder) -> (Real, Real) {
    if a > ps.u_lin {
        let (gc, ggc, dgc) = srom_raw(ps.u_lin, ps.zone_start, ps.beyond_k, ps.beyond_w);
        let d = a - ps.u_lin;
        (gc + dgc * d, ggc + gc * d + 0.5 * dgc * (d * d))
    } else {
        let (g, gg, _) = srom_raw(a, ps.zone_start, ps.beyond_k, ps.beyond_w);
        (g, gg)
    }
}

/// One limit f_k and its (y, p) partials.
#[inline(always)]
#[allow(clippy::too_many_arguments)]
fn srom_limit(k: &[Real; 4], c: Real, c_p: Real, h: Real, h_y: Real, h_p: Real,
              lam: Real, lam_p: Real, phi: Real) -> (Real, Real, Real) {
    let f = (k[0] + c * (k[1] + c * k[2])) + k[3] * h + lam * phi;
    let f_y = k[3] * h_y + lam;
    let f_p = (k[1] + 2.0 * c * k[2]) * c_p + k[3] * h_p + lam_p * phi;
    (f, f_y, f_p)
}

/// The two human limits (f_lo = ER side, f_hi = IR side; τ radians) at
/// left-canonical (y, p) — tests / diagnostics.
pub fn shoulder_limits(y: Real, p: Real, ps: &PassiveShoulder) -> (Real, Real) {
    let (sy, cy) = y.sin_cos();
    let (sp, cp) = p.sin_cos();
    let c = -sp;
    let h = -cp * sy - cp * cy;
    let phi = y + core::f64::consts::FRAC_PI_2 as Real;
    let lo = srom_limit(&ps.coef_lo, c, -cp, h, 0.0, 0.0, 1.0 - c, cp, phi).0;
    let hi = srom_limit(&ps.coef_hi, c, -cp, h, 0.0, 0.0, 1.0 - c, cp, phi).0;
    (lo, hi)
}

/// Passive shoulder torque of one arm at LEFT-CANONICAL (y, p, t). The
/// CUDA `shoulder_rom` and constants.py `shoulder_rom` transcribe this
/// expression tree.
#[inline]
pub fn shoulder_rom(y: Real, p: Real, t: Real, ps: &PassiveShoulder) -> SromOut {
    let (sy, cy) = y.sin_cos();
    let (sp, cp) = p.sin_cos();
    let c = -sp;
    let c_p = -cp;
    let h = -cp * sy - cp * cy;   // d_l − d_f
    let h_y = cp * sy - cp * cy;  // ∂/∂y
    let h_p = sp * sy + sp * cy;  // ∂/∂p
    let lam = 1.0 - c;
    let lam_p = cp;
    let phi = y + core::f64::consts::FRAC_PI_2 as Real;
    let (flo, flo_y, flo_p) = srom_limit(&ps.coef_lo, c, c_p, h, h_y, h_p, lam, lam_p, phi);
    let (fhi, fhi_y, fhi_p) = srom_limit(&ps.coef_hi, c, c_p, h, h_y, h_p, lam, lam_p, phi);
    let m = 0.5 * (fhi + flo);
    let w = 0.5 * (fhi - flo);
    let m_y = 0.5 * (fhi_y + flo_y);
    let m_p = 0.5 * (fhi_p + flo_p);
    let w_y = 0.5 * (fhi_y - flo_y);
    let w_p = 0.5 * (fhi_p - flo_p);
    let u = ((y - t) - m) / w;
    let a = u.abs();
    let (g, gg) = srom_shape(a, ps);
    let tg = ps.tau_peak * g;
    let e_tau = if u >= 0.0 { tg } else { -tg };
    let e_w = ps.tau_peak * (gg - a * g);
    let ac = if a < ps.u_lin { a } else { ps.u_lin };
    let zc = if ac > ps.zone_start { (ac - ps.zone_start) / (1.0 - ps.zone_start) } else { 0.0 };
    SromOut {
        tau_y: -(e_tau * (1.0 - m_y) + e_w * w_y),
        tau_p: e_tau * m_p - e_w * w_p,
        tau_t: e_tau,
        u,
        b: ps.b_peak * (zc * zc),
        e: ps.tau_peak * w * gg,
    }
}

impl PassiveShoulder {
    /// Both arms at pose `q` (raw actuated angles), torques in RAW
    /// coordinates (the right arm's yaw / twist torques un-mirrored; u,
    /// b, e are mirror-invariant).
    #[inline]
    pub fn eval(&self, q: &[Real]) -> [SromOut; 2] {
        let (jy, jp, jt) = self.arms[0];
        let l = shoulder_rom(q[jy], q[jp], q[jt], self);
        let (jy, jp, jt) = self.arms[1];
        let mut r = shoulder_rom(-q[jy], q[jp], -q[jt], self);
        r.tau_y = -r.tau_y;
        r.tau_t = -r.tau_t;
        [l, r]
    }
}

/// HUMANOID's passive shoulders: (yaw 4, pitch 3, twist 5), (11, 10, 12).
pub const HUMANOID_PASSIVE_SHOULDER: PassiveShoulder = PassiveShoulder {
    arms: [(4, 3, 5), (11, 10, 12)],
    coef_lo: SROM_COEF_LO,
    coef_hi: SROM_COEF_HI,
    tau_peak: SROM_TAU_PEAK,
    zone_start: SROM_S,
    beyond_k: SROM_KB,
    beyond_w: SROM_WB,
    u_lin: SROM_UC,
    b_peak: SROM_B_PEAK,
};

impl Model {
    /// Velocity-coordinate count (root 6 + actuated when free-based).
    pub fn nv(&self) -> usize {
        self.nq + if self.free_base { 6 } else { 0 }
    }
    /// Offset of actuated coordinate j in the velocity vector.
    pub fn voff(&self) -> usize {
        if self.free_base { 6 } else { 0 }
    }
}

pub const G: Real = 9.81; // m/s^2, along -y (SPEC-v10 header)

pub const NQ: usize = 29;
pub const NV: usize = 35;

// Frame indices (chain order; see HUMANOID_FRAMES below).
pub const F_CHEST: usize = 2; // spine_roll — the chest body frame
pub const F_HEAD: usize = 3;
pub const F_L_SHOULDER: usize = 4; // l_sh_yaw — shoulder anchor origin
pub const F_L_ELBOW: usize = 7;
pub const F_L_WRIST: usize = 8; // l_wr_roll — wrist/hand origin
pub const F_L_HAND: usize = 10; // l_wr_bend — the hand BODY frame
pub const F_R_SHOULDER: usize = 11;
pub const F_R_ELBOW: usize = 14;
pub const F_R_WRIST: usize = 15;
pub const F_R_HAND: usize = 17;
pub const F_L_FOOT: usize = 23;
pub const F_R_FOOT: usize = 29;
// collider-carrying body frames (P2, collide.rs roster)
pub const F_L_UARM: usize = 6; // l_sh_twist — upper-arm body frame
pub const F_R_UARM: usize = 13;
pub const F_L_THIGH: usize = 20; // l_hip_roll — thigh body frame
pub const F_R_THIGH: usize = 26;
pub const F_L_SHANK: usize = 21; // l_knee — shank body frame
pub const F_R_SHANK: usize = 27;

// Actuated coordinate indices (§1.4 order).
pub const A_L_ARM_PITCH: usize = 3;
pub const A_L_ELBOW: usize = 6;
pub const A_L_ARM: usize = 3; // l arm block start (7 coords)
pub const A_R_ARM: usize = 10; // r arm block start (7 coords)

// §1.2 bodies (masses kg, COMs m, inertia diagonals kg·m^2).
const M_CHEST: Real = 24.20;
const COM_CHEST: [Real; 3] = [0.0, 0.215, 0.0];
const I_CHEST: [Real; 3] = [0.957507, 0.254964, 0.957507];
const M_HEAD: Real = 5.20;
const I_HEAD: [Real; 3] = [0.016848, 0.016848, 0.016848];
const M_UARM: Real = 2.00;
const COM_UARM: [Real; 3] = [0.0, 0.0, 0.175];
const I_UARM: [Real; 3] = [0.029222, 0.029222, 0.001966];
const M_FARM: Real = 1.20;
const COM_FARM: [Real; 3] = [0.0, 0.0, 0.15];
const I_FARM: [Real; 3] = [0.013056, 0.013056, 0.000931];
// Hand mass (user direction 2026-08-18): 0.35 kg fist, NO controller
// mass (was 0.61 = 0.46 fist + 0.15 controller). Real Beat Saber
// players exploit controller movement/angular prediction to make
// large swings with low effort — they never truly accelerate the
// full fist+controller mass through the whole range at speed — so a
// literal composite mass over-prices swings and was feeding vertical
// body jitter during play. Inertia rescaled with the mass
// (0.000460 x 0.35/0.61).
// 0.35 -> 0.50 (user 2026-10-01; main's body since 2026-10-03, the
// v63x2 body — the 0.35 kg hand lives on branch body/original),
// inertia rescaled again (0.000460 x 0.50/0.61).
const M_HAND: Real = 0.50;
const COM_HAND: [Real; 3] = [0.0, 0.0, 0.06];
const I_HAND: [Real; 3] = [0.000377, 0.000377, 0.000377];
const M_THIGH: Real = 10.60;
const COM_THIGH: [Real; 3] = [0.0, -0.22, 0.0];
const I_THIGH: [Real; 3] = [0.250198, 0.018493, 0.250198];
const M_SHANK: Real = 3.20;
const COM_SHANK: [Real; 3] = [0.0, -0.215, 0.0];
const I_SHANK: [Real; 3] = [0.065964, 0.003161, 0.065964];
const M_FOOT: Real = 1.00;
const COM_FOOT: [Real; 3] = [0.0, -0.035, 0.05];
const I_FOOT: [Real; 3] = [0.005617, 0.005883, 0.001083];

const ZERO3: [Real; 3] = [0.0, 0.0, 0.0];
const AX: [Real; 3] = [1.0, 0.0, 0.0];
const AXN: [Real; 3] = [-1.0, 0.0, 0.0];
const AY: [Real; 3] = [0.0, 1.0, 0.0];
const AZ: [Real; 3] = [0.0, 0.0, 1.0];

/// Massless intermediate frame of a stacked-revolute cluster.
const fn rev(parent: i32, anchor: [Real; 3], axis: [Real; 3], coord: i32) -> Frame {
    Frame { parent, anchor, axis, coord, mass: 0.0, com: ZERO3, inertia: ZERO3 }
}

/// Cluster-terminal revolute frame carrying the body's mass.
const fn revb(
    parent: i32,
    anchor: [Real; 3],
    axis: [Real; 3],
    coord: i32,
    mass: Real,
    com: [Real; 3],
    inertia: [Real; 3],
) -> Frame {
    Frame { parent, anchor, axis, coord, mass, com, inertia }
}

// §1.3 joint semantics as stacked revolutes, chain order. Child body
// orientation = parent × Rot(axis_1, q_1) × Rot(axis_2, q_2) × ... —
// intrinsic composition, leftmost applied first (reproduces the
// retired v9 arm chains):
//   shoulder  M   = R_y(yaw)·R_x(−pitch)·R_z(twist)  — axes (0,1,0),
//             (−1,0,0), (0,0,1); upper arm along local +z (0.35)
//   elbow     M_f = M·R_x(−e)                        — axis (−1,0,0);
//             forearm along local +z (0.30)
//   wrist     R_w = R_z(roll)·R_y(dev)·R_x(−bend)    — axes (0,0,1),
//             (0,1,0), (−1,0,0); blade = hand-frame +z
//   spine     R_x(pitch)·R_y(yaw)·R_z(roll)
//   hip       R_x(−pitch)·R_y(yaw)·R_z(roll)         — axes (−1,0,0),
//             (0,1,0), (0,0,1); thigh along local −y
//   knee      axis (+1,0,0); shank along local −y
//   ankle     R_x(−pitch)·R_z(roll)                  — axes (−1,0,0),
//             (0,0,1); foot box along local +z
#[rustfmt::skip]
pub const HUMANOID_FRAMES: [Frame; 30] = [
    // spine: pelvis → chest, anchor at the waist (0, 0.10, 0)
    /*  0 */ rev(-1, [0.0, 0.10, 0.0], AX, 0),                     // spine_pitch
    /*  1 */ rev(0, ZERO3, AY, 1),                                 // spine_yaw
    /*  2 */ revb(1, ZERO3, AZ, 2, M_CHEST, COM_CHEST, I_CHEST),   // spine_roll + chest
    // head: WELDED to chest (§1.2 — no neck DoF)
    /*  3 */ Frame { parent: 2, anchor: [0.0, 0.66, 0.0], axis: AZ, coord: -1,
                     mass: M_HEAD, com: ZERO3, inertia: I_HEAD },
    // left arm (coords 3..=9): stack yaw(4) → pitch(3) → twist(5)
    /*  4 */ rev(2, [-0.20, 0.43, 0.0], AY, 4),                    // l_sh_yaw
    /*  5 */ rev(4, ZERO3, AXN, 3),                                // l_sh_pitch
    /*  6 */ revb(5, ZERO3, AZ, 5, M_UARM, COM_UARM, I_UARM),      // l_sh_twist + uarm_l
    /*  7 */ revb(6, [0.0, 0.0, 0.35], AXN, 6, M_FARM, COM_FARM, I_FARM), // l_elbow + farm_l
    // left wrist: stack roll(7) → dev(9) → bend(8)
    /*  8 */ rev(7, [0.0, 0.0, 0.30], AZ, 7),                      // l_wr_roll
    /*  9 */ rev(8, ZERO3, AY, 9),                                 // l_wr_dev
    /* 10 */ revb(9, ZERO3, AXN, 8, M_HAND, COM_HAND, I_HAND),     // l_wr_bend + hand_l
    // right arm (coords 10..=16)
    /* 11 */ rev(2, [0.20, 0.43, 0.0], AY, 11),                    // r_sh_yaw
    /* 12 */ rev(11, ZERO3, AXN, 10),                              // r_sh_pitch
    /* 13 */ revb(12, ZERO3, AZ, 12, M_UARM, COM_UARM, I_UARM),    // r_sh_twist + uarm_r
    /* 14 */ revb(13, [0.0, 0.0, 0.35], AXN, 13, M_FARM, COM_FARM, I_FARM), // r_elbow
    /* 15 */ rev(14, [0.0, 0.0, 0.30], AZ, 14),                    // r_wr_roll
    /* 16 */ rev(15, ZERO3, AY, 16),                               // r_wr_dev
    /* 17 */ revb(16, ZERO3, AXN, 15, M_HAND, COM_HAND, I_HAND),   // r_wr_bend + hand_r
    // left leg (coords 17..=22): stack pitch → yaw → roll = actuated
    /* 18 */ rev(-1, [-0.10, 0.0, 0.0], AXN, 17),                  // l_hip_pitch
    /* 19 */ rev(18, ZERO3, AY, 18),                               // l_hip_yaw
    /* 20 */ revb(19, ZERO3, AZ, 19, M_THIGH, COM_THIGH, I_THIGH), // l_hip_roll + thigh_l
    /* 21 */ revb(20, [0.0, -0.44, 0.0], AX, 20, M_SHANK, COM_SHANK, I_SHANK), // l_knee
    /* 22 */ rev(21, [0.0, -0.43, 0.0], AXN, 21),                  // l_ankle_pitch
    /* 23 */ revb(22, ZERO3, AZ, 22, M_FOOT, COM_FOOT, I_FOOT),    // l_ankle_roll + foot_l
    // right leg (coords 23..=28)
    /* 24 */ rev(-1, [0.10, 0.0, 0.0], AXN, 23),                   // r_hip_pitch
    /* 25 */ rev(24, ZERO3, AY, 24),                               // r_hip_yaw
    /* 26 */ revb(25, ZERO3, AZ, 25, M_THIGH, COM_THIGH, I_THIGH), // r_hip_roll + thigh_r
    /* 27 */ revb(26, [0.0, -0.44, 0.0], AX, 26, M_SHANK, COM_SHANK, I_SHANK), // r_knee
    /* 28 */ rev(27, [0.0, -0.43, 0.0], AXN, 27),                  // r_ankle_pitch
    /* 29 */ revb(28, ZERO3, AZ, 28, M_FOOT, COM_FOOT, I_FOOT),    // r_ankle_roll + foot_r
];

// §1.4 actuated table, exact literals, actuated order:
// spine[pitch,yaw,roll], arm_l[pitch,yaw,twist,elbow,roll,bend,dev],
// arm_r[same], leg_l[hp,hy,hr,knee,ap,ar], leg_r[same].
// exp/shoulder-rom (2026-10-07): the arm twist box's INTERNAL side widened
// (l lo −90° → −190°, r hi +90° → +190°). The v10 box was ER 160°…IR 0° in
// ISB axial terms at every arm direction (twist 0 = forearm plane through
// the plane of elevation); the passive shoulder now carries the human
// axial range, and its IR limit needs down to −184.3° of twist (abduction
// plane, ~40° elevation) — all of it inside the box. Reach invariant:
// 2·h·ω_cap = 0.133 rad ≪ 260° range.
#[rustfmt::skip]
pub const HUMANOID_LIMITS: [(Real, Real); NQ] = [
    (-0.3490658503988659, 0.7853981633974483),   //  0 spine_pitch
    (-0.6981317007977318, 0.6981317007977318),   //  1 spine_yaw
    (-0.5235987755982988, 0.5235987755982988),   //  2 spine_roll
    (-1.5707963267948966, 1.1344640137963142),   //  3 l_arm_pitch
    (-1.7453292519943295, 0.7853981633974483),   //  4 l_arm_yaw
    (-3.3161255787892263, 1.2217304763960306),   //  5 l_arm_twist (lo -90 -> -190 deg, exp/shoulder-rom)
    (0.0, 2.6179938779914944),                   //  6 l_elbow
    (0.0, 3.141592653589793),                    //  7 l_wrist_roll
    (-1.6580627893946132, 1.6580627893946132),   //  8 l_wrist_bend
    (-0.5235987755982988, 0.2617993877991494),   //  9 l_wrist_dev
    (-1.5707963267948966, 1.1344640137963142),   // 10 r_arm_pitch
    (-0.7853981633974483, 1.7453292519943295),   // 11 r_arm_yaw
    (-1.2217304763960306, 3.3161255787892263),   // 12 r_arm_twist (hi +90 -> +190 deg, exp/shoulder-rom)
    (0.0, 2.6179938779914944),                   // 13 r_elbow
    (-3.141592653589793, 0.0),                   // 14 r_wrist_roll
    (-1.6580627893946132, 1.6580627893946132),   // 15 r_wrist_bend
    (-0.2617993877991494, 0.5235987755982988),   // 16 r_wrist_dev
    (-0.3490658503988659, 2.0943951023931953),   // 17 l_hip_pitch
    (-0.7853981633974483, 0.6981317007977318),   // 18 l_hip_yaw
    (-0.7853981633974483, 0.5235987755982988),   // 19 l_hip_roll
    (0.0, 2.6179938779914944),                   // 20 l_knee
    (-0.8726646259971648, 0.6108652381980153),   // 21 l_ankle_pitch (hi 20 -> 35 deg, 2026-08-20)
    (-0.3490658503988659, 0.3490658503988659),   // 22 l_ankle_roll
    (-0.3490658503988659, 2.0943951023931953),   // 23 r_hip_pitch
    (-0.6981317007977318, 0.7853981633974483),   // 24 r_hip_yaw
    (-0.5235987755982988, 0.7853981633974483),   // 25 r_hip_roll
    (0.0, 2.6179938779914944),                   // 26 r_knee
    (-0.8726646259971648, 0.6108652381980153),   // 27 r_ankle_pitch (hi 20 -> 35 deg, 2026-08-20)
    (-0.3490658503988659, 0.3490658503988659),   // 28 r_ankle_roll
];

#[rustfmt::skip]
pub const HUMANOID_TAU_MAX: [Real; NQ] = [
    600.00, 239.95, 375.00,                           // spine
    85.38, 85.38, 19.45, 19.13, 1.06, 1.431, 1.06,  // arm_l (bend
    85.38, 85.38, 19.45, 19.13, 1.06, 1.431, 1.06,  // arm_r  +35%, 2026-10-01)
    262.50, 75.00, 135.00, 150.00, 135.00, 37.50,    // leg_l (2026-08-20: knee 250 -> 150,
    262.50, 75.00, 135.00, 150.00, 135.00, 37.50,    // leg_r  all other leg joints -25%)
];

// omega caps: 16 rad/s everywhere except the wrist tier's 50
// (§1.4; 40 -> 50 per user 2026-08-18 — with the fist-only hand the
// full-gear bend swing was cap-clipped mid-arc); bend 50 -> 65 (user
// 2026-10-01); 65 -> 90 (user 2026-10-03) — main's v63x2 body; 50
// everywhere on the wrist tier on branch body/original. 90 -> 105 (user
// 2026-10-04, exp/bend7 only): 2*h*cap = 0.875 rad, far inside the
// bend's +-95 deg range (the limit solver's reach invariant).
#[rustfmt::skip]
pub const HUMANOID_OMEGA_CAP: [Real; NQ] = [
    16.0, 16.0, 16.0,
    16.0, 16.0, 16.0, 16.0, 50.0, 105.0, 50.0,
    16.0, 16.0, 16.0, 16.0, 50.0, 105.0, 50.0,
    16.0, 16.0, 16.0, 16.0, 16.0, 16.0,
    16.0, 16.0, 16.0, 16.0, 16.0, 16.0,
];

// b = 0.6 · I_ref (pre-clamp), §1.5 table values verbatim.
#[rustfmt::skip]
pub const HUMANOID_DAMPING: [Real; NQ] = [
    4.3140, 1.1998, 3.6471,
    0.42689, 0.42689, 0.07292, 0.07174, 0.00159, 0.00159, 0.00159,
    0.42689, 0.42689, 0.07292, 0.07174, 0.00159, 0.00159, 0.00159,
    1.8175, 0.28120, 1.8133, 0.26294, 0.00561, 0.00138,
    1.8175, 0.28120, 1.8133, 0.26294, 0.00561, 0.00138,
];

// armature = I_ref[j] / 100 (§1.5 I_ref column, pre-clamp, verbatim).
#[rustfmt::skip]
pub const HUMANOID_ARMATURE: [Real; NQ] = [
    0.071900, 0.019996, 0.060786,
    0.0071149, 0.0071149, 0.0012154, 0.0011957, 0.000026556, 0.000026556, 0.000026556,
    0.0071149, 0.0071149, 0.0012154, 0.0011957, 0.000026556, 0.000026556, 0.000026556,
    0.030292, 0.0046866, 0.030222, 0.0043823, 0.000093419, 0.000023083,
    0.030292, 0.0046866, 0.030222, 0.0043823, 0.000093419, 0.000023083,
];

/// Actuated-coordinate mirror map: spine (0..3) central; arm_l (3..10)
/// <-> arm_r (10..17); leg_l (17..23) <-> leg_r (23..29). Each pair's two
/// joints share their axis convention up to the reflection (the yaw /
/// twist / roll / dev / hip-yaw / hip-roll / ankle-roll coordinates flip
/// sign, the rest keep it — python/beat_arena_env/mirror_ops.py), and
/// every body table above is equal across a pair (limits negated where
/// the sign flips) — pinned by `humanoid_tables_are_mirror_symmetric`.
#[rustfmt::skip]
pub const HUMANOID_MIRROR: [usize; NQ] = [
    0, 1, 2,
    10, 11, 12, 13, 14, 15, 16,
    3, 4, 5, 6, 7, 8, 9,
    23, 24, 25, 26, 27, 28,
    17, 18, 19, 20, 21, 22,
];

/// The SPEC-v10 humanoid: floating pelvis + 29 actuated revolutes,
/// NV = 35. Reference pose: root (0, 0.94, −0.65), identity, all zero.
pub const HUMANOID: Model = Model {
    free_base: true,
    base_mass: 8.40,
    base_com: ZERO3,
    base_inertia: [0.038640, 0.101640, 0.101640],
    base_pos0: [0.0, 0.94, -0.65],
    frames: &HUMANOID_FRAMES,
    nq: NQ,
    limits: &HUMANOID_LIMITS,
    omega_cap: &HUMANOID_OMEGA_CAP,
    tau_max: &HUMANOID_TAU_MAX,
    damping: &HUMANOID_DAMPING,
    armature: &HUMANOID_ARMATURE,
    gravity: [0.0, -G, 0.0],
    mirror: &HUMANOID_MIRROR,
    passive_wrist: Some(HUMANOID_PASSIVE_WRIST),
    passive_shoulder: Some(HUMANOID_PASSIVE_SHOULDER),
};
