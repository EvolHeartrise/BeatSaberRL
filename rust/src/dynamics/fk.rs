//! Generalized tree forward kinematics over the model tables (SPEC-v10
//! §1.3): world rotation + origin per frame. Stacking orders live in the
//! frame table's chain order — this walk implements them by construction
//! (child rotation = parent × Rot(axis, q)).
//!
//! Expression trees are written plainly (fixed evaluation order, no
//! fastmath assumptions) — the torch/CUDA mirrors replicate them
//! literally.

use crate::Real;
use super::model::Model;
use super::{State, Workspace};

// the three judge-shared vector ops live in crate::geom (also compiled
// into the Flux WebXR game's wasm build); re-exported here unchanged
pub use crate::geom::{cross3, dot3, sub3};

pub type Mat3 = [[Real; 3]; 3];

pub const IDENT3: Mat3 = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];

pub fn mat_vec(m: &Mat3, v: [Real; 3]) -> [Real; 3] {
    [
        m[0][0] * v[0] + m[0][1] * v[1] + m[0][2] * v[2],
        m[1][0] * v[0] + m[1][1] * v[1] + m[1][2] * v[2],
        m[2][0] * v[0] + m[2][1] * v[1] + m[2][2] * v[2],
    ]
}

pub fn mat_mul(a: &Mat3, b: &Mat3) -> Mat3 {
    let mut out = [[0.0; 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            out[i][j] = a[i][0] * b[0][j] + a[i][1] * b[1][j] + a[i][2] * b[2][j];
        }
    }
    out
}

pub fn add3(a: [Real; 3], b: [Real; 3]) -> [Real; 3] {
    [a[0] + b[0], a[1] + b[1], a[2] + b[2]]
}

pub fn scale3(a: [Real; 3], s: Real) -> [Real; 3] {
    [a[0] * s, a[1] * s, a[2] * s]
}

/// Rodrigues: R = I + sinθ·[a]x + (1−cosθ)·[a]x² for a UNIT axis.
pub fn rot_axis(axis: [Real; 3], angle: Real) -> Mat3 {
    let (s, c) = angle.sin_cos();
    let k = 1.0 - c;
    let [x, y, z] = axis;
    [
        [c + k * x * x, k * x * y - s * z, k * x * z + s * y],
        [k * x * y + s * z, c + k * y * y, k * y * z - s * x],
        [k * x * z - s * y, k * y * z + s * x, c + k * z * z],
    ]
}

/// World-from-body rotation of a wxyz (scalar-first) quaternion.
pub fn quat_to_rot(q: [Real; 4]) -> Mat3 {
    let [w, x, y, z] = q;
    [
        [
            1.0 - 2.0 * (y * y + z * z),
            2.0 * (x * y - w * z),
            2.0 * (x * z + w * y),
        ],
        [
            2.0 * (x * y + w * z),
            1.0 - 2.0 * (x * x + z * z),
            2.0 * (y * z - w * x),
        ],
        [
            2.0 * (x * z - w * y),
            2.0 * (y * z + w * x),
            1.0 - 2.0 * (x * x + y * y),
        ],
    ]
}

/// SO(3) → wxyz (scalar-first) quaternion: the exact inverse of
/// `quat_to_rot`, for exporting a FK frame's world rotation (replay
/// v10 `head`, §6). Shepperd's branch — take the largest of
/// (trace, r00, r11, r22) so the divisor is never small; the naive
/// w = ½√(1+tr) form loses every digit as θ → π (tr → −1), which a
/// tumbling body REACHES (a fallen head faces backwards).
///
/// Returns an arbitrary sign representative (q and −q are the same
/// rotation). Consumers that INTERPOLATE a sequence of these must fix
/// the sign continuity themselves — see convert_scoresaber.py, which
/// hemisphere-aligns consecutive replay frames because the viewer
/// lerps between them.
pub fn rot_to_quat(r: &Mat3) -> [Real; 4] {
    let (m00, m11, m22) = (r[0][0], r[1][1], r[2][2]);
    let tr = m00 + m11 + m22;
    if tr > 0.0 {
        let s = (tr + 1.0).sqrt() * 2.0; // 4w
        [
            0.25 * s,
            (r[2][1] - r[1][2]) / s,
            (r[0][2] - r[2][0]) / s,
            (r[1][0] - r[0][1]) / s,
        ]
    } else if m00 > m11 && m00 > m22 {
        let s = (1.0 + m00 - m11 - m22).sqrt() * 2.0; // 4x
        [
            (r[2][1] - r[1][2]) / s,
            0.25 * s,
            (r[0][1] + r[1][0]) / s,
            (r[0][2] + r[2][0]) / s,
        ]
    } else if m11 > m22 {
        let s = (1.0 + m11 - m00 - m22).sqrt() * 2.0; // 4y
        [
            (r[0][2] - r[2][0]) / s,
            (r[0][1] + r[1][0]) / s,
            0.25 * s,
            (r[1][2] + r[2][1]) / s,
        ]
    } else {
        let s = (1.0 + m22 - m00 - m11).sqrt() * 2.0; // 4z
        [
            (r[1][0] - r[0][1]) / s,
            (r[0][2] + r[2][0]) / s,
            (r[1][2] + r[2][1]) / s,
            0.25 * s,
        ]
    }
}

pub fn quat_mul(a: [Real; 4], b: [Real; 4]) -> [Real; 4] {
    [
        a[0] * b[0] - a[1] * b[1] - a[2] * b[2] - a[3] * b[3],
        a[0] * b[1] + a[1] * b[0] + a[2] * b[3] - a[3] * b[2],
        a[0] * b[2] - a[1] * b[3] + a[2] * b[0] + a[3] * b[1],
        a[0] * b[3] + a[1] * b[2] - a[2] * b[1] + a[3] * b[0],
    ]
}

/// FK over the whole tree: fills ws.r0/p0 (base) and ws.r/ws.p per
/// frame, plus ws.axw (world revolute axes, R_k · axis).
pub fn fk_all(model: &Model, st: &State, ws: &mut Workspace) {
    ws.r0 = if model.free_base { quat_to_rot(st.quat) } else { IDENT3 };
    ws.p0 = st.p;
    for (k, fr) in model.frames.iter().enumerate() {
        let (rl, pl) = if fr.parent < 0 {
            (&ws.r0, ws.p0)
        } else {
            (&ws.r[fr.parent as usize], ws.p[fr.parent as usize])
        };
        ws.p[k] = add3(pl, mat_vec(rl, fr.anchor));
        ws.r[k] = if fr.coord >= 0 {
            mat_mul(rl, &rot_axis(fr.axis, st.q[fr.coord as usize]))
        } else {
            *rl
        };
        ws.axw[k] = mat_vec(&ws.r[k], fr.axis);
    }
}
