//! SPEC-v10 §1 model tables — the SINGLE source of truth for the v10
//! kinematic tree: parent indices, joint axes, stacking orders, anchors,
//! masses/COMs/inertias, tau_max, omega caps, damping, joint limits.
//! Every literal is normative from SPEC-v10 §1.1–1.5 as written; nothing
//! here is re-derived, and no other module carries its own copy (the v9
//! "6-initializer zero-fill" bug class is the cautionary tale).

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
    pub anchor: [f64; 3],  // frame origin offset in the PARENT frame
    pub axis: [f64; 3],    // revolute axis (unit, local); fixed: unused
    pub coord: i32,        // actuated coordinate index; -1 = welded
    pub mass: f64,         // body mass attached to this frame (0 = none)
    pub com: [f64; 3],     // in this frame
    pub inertia: [f64; 3], // principal diagonal about the COM, frame axes
}

/// A rigid-body model the dynamics core is generic over: the humanoid
/// below, or ad-hoc test tables (pendulums). Per-coordinate arrays are
/// indexed in ACTUATED order (§1.4).
#[derive(Clone, Copy)]
pub struct Model {
    pub free_base: bool, // 6-DoF free joint on the base body
    pub base_mass: f64,  // base body (pelvis); 0 for fixed-base models
    pub base_com: [f64; 3],
    pub base_inertia: [f64; 3],
    pub base_pos0: [f64; 3], // reference (neutral) base position
    pub frames: &'static [Frame],
    pub nq: usize, // actuated coordinate count
    pub limits: &'static [(f64, f64)],
    pub omega_cap: &'static [f64],
    pub tau_max: &'static [f64],
    pub damping: &'static [f64], // b: tau_damp = -b * qdot (§1.5)
    // P1 divergence (SPEC-v10 Divergences §12): per-DoF armature
    // (reflected rotor) inertia, added to M's diagonal AND to RNEA's
    // qdd term (MuJoCo semantics — the MJCF twin mirrors it via the
    // `armature` attribute). Without it M(q) is exactly SINGULAR at
    // reachable poses: the shoulder stack yaw→pitch→twist gimbal-locks
    // at pitch ±90°, and l_arm_pitch's lower limit IS −90° — at the
    // clamp the yaw and twist world axes are anti-parallel through the
    // same point (null vector yaw=twist=1), and the §3 no-pivot
    // Cholesky dies. armature = I_ref/100 (1 % reflected inertia).
    pub armature: &'static [f64],
    pub gravity: [f64; 3],
}

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

pub const G: f64 = 9.81; // m/s^2, along -y (SPEC-v10 header)

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
const M_CHEST: f64 = 24.20;
const COM_CHEST: [f64; 3] = [0.0, 0.215, 0.0];
const I_CHEST: [f64; 3] = [0.957507, 0.254964, 0.957507];
const M_HEAD: f64 = 5.20;
const I_HEAD: [f64; 3] = [0.016848, 0.016848, 0.016848];
const M_UARM: f64 = 2.00;
const COM_UARM: [f64; 3] = [0.0, 0.0, 0.175];
const I_UARM: [f64; 3] = [0.029222, 0.029222, 0.001966];
const M_FARM: f64 = 1.20;
const COM_FARM: [f64; 3] = [0.0, 0.0, 0.15];
const I_FARM: [f64; 3] = [0.013056, 0.013056, 0.000931];
// Hand mass (user direction 2026-08-18): 0.35 kg fist, NO controller
// mass (was 0.61 = 0.46 fist + 0.15 controller). Real Beat Saber
// players exploit controller movement/angular prediction to make
// large swings with low effort — they never truly accelerate the
// full fist+controller mass through the whole range at speed — so a
// literal composite mass over-prices swings and was feeding vertical
// body jitter during play. Inertia rescaled with the mass
// (0.000460 x 0.35/0.61).
const M_HAND: f64 = 0.35;
const COM_HAND: [f64; 3] = [0.0, 0.0, 0.06];
const I_HAND: [f64; 3] = [0.000264, 0.000264, 0.000264];
const M_THIGH: f64 = 10.60;
const COM_THIGH: [f64; 3] = [0.0, -0.22, 0.0];
const I_THIGH: [f64; 3] = [0.250198, 0.018493, 0.250198];
const M_SHANK: f64 = 3.20;
const COM_SHANK: [f64; 3] = [0.0, -0.215, 0.0];
const I_SHANK: [f64; 3] = [0.065964, 0.003161, 0.065964];
const M_FOOT: f64 = 1.00;
const COM_FOOT: [f64; 3] = [0.0, -0.035, 0.05];
const I_FOOT: [f64; 3] = [0.005617, 0.005883, 0.001083];

const ZERO3: [f64; 3] = [0.0, 0.0, 0.0];
const AX: [f64; 3] = [1.0, 0.0, 0.0];
const AXN: [f64; 3] = [-1.0, 0.0, 0.0];
const AY: [f64; 3] = [0.0, 1.0, 0.0];
const AZ: [f64; 3] = [0.0, 0.0, 1.0];

/// Massless intermediate frame of a stacked-revolute cluster.
const fn rev(parent: i32, anchor: [f64; 3], axis: [f64; 3], coord: i32) -> Frame {
    Frame { parent, anchor, axis, coord, mass: 0.0, com: ZERO3, inertia: ZERO3 }
}

/// Cluster-terminal revolute frame carrying the body's mass.
const fn revb(
    parent: i32,
    anchor: [f64; 3],
    axis: [f64; 3],
    coord: i32,
    mass: f64,
    com: [f64; 3],
    inertia: [f64; 3],
) -> Frame {
    Frame { parent, anchor, axis, coord, mass, com, inertia }
}

// §1.3 joint semantics as stacked revolutes, chain order. Child body
// orientation = parent × Rot(axis_1, q_1) × Rot(axis_2, q_2) × ... —
// intrinsic composition, leftmost applied first. Verified against v9
// fk() by the continuity gate (tests_fk.rs):
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
#[rustfmt::skip]
pub const HUMANOID_LIMITS: [(f64, f64); NQ] = [
    (-0.3490658503988659, 0.7853981633974483),   //  0 spine_pitch
    (-0.6981317007977318, 0.6981317007977318),   //  1 spine_yaw
    (-0.5235987755982988, 0.5235987755982988),   //  2 spine_roll
    (-1.5707963267948966, 1.1344640137963142),   //  3 l_arm_pitch
    (-1.7453292519943295, 0.7853981633974483),   //  4 l_arm_yaw
    (-1.5707963267948966, 1.2217304763960306),   //  5 l_arm_twist
    (0.0, 2.6179938779914944),                   //  6 l_elbow
    (0.0, 3.141592653589793),                    //  7 l_wrist_roll
    (-1.6580627893946132, 1.6580627893946132),   //  8 l_wrist_bend
    (-0.5235987755982988, 0.2617993877991494),   //  9 l_wrist_dev
    (-1.5707963267948966, 1.1344640137963142),   // 10 r_arm_pitch
    (-0.7853981633974483, 1.7453292519943295),   // 11 r_arm_yaw
    (-1.2217304763960306, 1.5707963267948966),   // 12 r_arm_twist
    (0.0, 2.6179938779914944),                   // 13 r_elbow
    (-3.141592653589793, 0.0),                   // 14 r_wrist_roll
    (-1.6580627893946132, 1.6580627893946132),   // 15 r_wrist_bend
    (-0.2617993877991494, 0.5235987755982988),   // 16 r_wrist_dev
    (-0.3490658503988659, 2.0943951023931953),   // 17 l_hip_pitch
    (-0.7853981633974483, 0.6981317007977318),   // 18 l_hip_yaw
    (-0.7853981633974483, 0.5235987755982988),   // 19 l_hip_roll
    (0.0, 2.6179938779914944),                   // 20 l_knee
    (-0.8726646259971648, 0.3490658503988659),   // 21 l_ankle_pitch
    (-0.3490658503988659, 0.3490658503988659),   // 22 l_ankle_roll
    (-0.3490658503988659, 2.0943951023931953),   // 23 r_hip_pitch
    (-0.6981317007977318, 0.7853981633974483),   // 24 r_hip_yaw
    (-0.5235987755982988, 0.7853981633974483),   // 25 r_hip_roll
    (0.0, 2.6179938779914944),                   // 26 r_knee
    (-0.8726646259971648, 0.3490658503988659),   // 27 r_ankle_pitch
    (-0.3490658503988659, 0.3490658503988659),   // 28 r_ankle_roll
];

#[rustfmt::skip]
pub const HUMANOID_TAU_MAX: [f64; NQ] = [
    800.00, 319.93, 500.00,                          // spine
    85.38, 85.38, 19.45, 19.13, 1.06, 1.06, 1.06,  // arm_l
    85.38, 85.38, 19.45, 19.13, 1.06, 1.06, 1.06,  // arm_r
    262.50, 75.00, 135.00, 150.00, 135.00, 37.50,    // leg_l (2026-08-20: knee 250 -> 150,
    262.50, 75.00, 135.00, 150.00, 135.00, 37.50,    // leg_r  all other leg joints -25%)
];

// omega caps: 16 rad/s everywhere except the wrist tier's 50
// (§1.4; 40 -> 50 per user 2026-08-18 — with the fist-only hand the
// full-gear bend swing was cap-clipped mid-arc).
#[rustfmt::skip]
pub const HUMANOID_OMEGA_CAP: [f64; NQ] = [
    16.0, 16.0, 16.0,
    16.0, 16.0, 16.0, 16.0, 50.0, 50.0, 50.0,
    16.0, 16.0, 16.0, 16.0, 50.0, 50.0, 50.0,
    16.0, 16.0, 16.0, 16.0, 16.0, 16.0,
    16.0, 16.0, 16.0, 16.0, 16.0, 16.0,
];

// b = 0.6 · I_ref (pre-clamp), §1.5 table values verbatim.
#[rustfmt::skip]
pub const HUMANOID_DAMPING: [f64; NQ] = [
    4.3140, 1.1998, 3.6471,
    0.42689, 0.42689, 0.07292, 0.07174, 0.00159, 0.00159, 0.00159,
    0.42689, 0.42689, 0.07292, 0.07174, 0.00159, 0.00159, 0.00159,
    1.8175, 0.28120, 1.8133, 0.26294, 0.00561, 0.00138,
    1.8175, 0.28120, 1.8133, 0.26294, 0.00561, 0.00138,
];

// armature = I_ref[j] / 100 (§1.5 I_ref column, pre-clamp, verbatim).
#[rustfmt::skip]
pub const HUMANOID_ARMATURE: [f64; NQ] = [
    0.071900, 0.019996, 0.060786,
    0.0071149, 0.0071149, 0.0012154, 0.0011957, 0.000026556, 0.000026556, 0.000026556,
    0.0071149, 0.0071149, 0.0012154, 0.0011957, 0.000026556, 0.000026556, 0.000026556,
    0.030292, 0.0046866, 0.030222, 0.0043823, 0.000093419, 0.000023083,
    0.030292, 0.0046866, 0.030222, 0.0043823, 0.000093419, 0.000023083,
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
};
