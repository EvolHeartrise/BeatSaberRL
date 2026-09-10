// Replay format v10 (SPEC-v10 §6): env v10 full-body dynamics.
// Pure logic only (no three/svelte imports) — unit-tested in node.
//
// World: floor-based coordinates, y = 0 is the floor. Lanes at
// x = -0.9, -0.3, 0.3, 0.9 (0.6 m pitch); row centers y = 0.85, 1.40,
// 1.90. Hit plane z = 0; objects approach from +z. Per-tick records
// carry the FULL body state: q(36) = root pos + root quat (wxyz) + 29
// joint angles (§1.4 actuated order), root twist(6), omega(29), the
// per-hand FK sabers [wrist(3), blade(3)] (viewer convenience — the
// skeleton is drawn from full-body FK over q), 29 gear-id actions,
// contact events {pair, p, j} (§2 pair ids, tick-summed impulse), and
// the P6 ADDITIVE fields "feet" (per-foot tick-summed weld impulses
// [jx,jy,jz,jax,jay,jaz]; vertical reaction N ≈ jy/DT) and "effort"
// (positive mechanical work, J) — additive: readers must tolerate
// their absence. Applied torque is NOT logged; it is exactly
// gearTable(config.gears)[action[j]] * TAU_MAX29[j].
// The Three.js scene maps world (x, y, z) -> (x, y, -z): the camera sits at
// scene +z looking down -z, preserving the player's view (x right, y up,
// notes approaching).
//
// env v15 (contact spec §10) — HEAD-ANCHORED NOTE TRACK. Every chart
// object's world z carries the live offset z_off(t) = head_z(t) - PLANE_Z0:
//
//     note_z(t) = (hit_time - t) * speed + z_off(t)
//
// so walking/leaning toward the notes pushes the whole track away by
// exactly the distance travelled. x and y are unchanged. The header field
// "track_anchor": "head" is ADDITIVE: recordings that carry it get the
// offset, pre-v15 recordings (field absent) keep the old fixed track.

// BLOCK 0.5: the real game's visual cube (verified against the
// ScoreSaber viewer's note mesh, 2026-07-30)
export const CELL = 0.6, BLOCK = 0.5, HALF = BLOCK / 2;
export const LANE_X = [-0.9, -0.3, 0.3, 0.9];
export const ROW_Y = [0.85, 1.4, 1.9];
// grid frame: lane cell edges; row edges midway between the non-uniform
// row centers (extrapolated at the ends)
export const GX0 = -1.2, GX1 = 1.2;
export const GY0 = 0.575, GY1 = 2.15;
export const ROW_EDGES = [GY0, 1.125, 1.65, GY1];
export const Z_FAR = 12.0;
// env v10: REACH/miss are LIVE-POSE gates in the sim; the viewer keeps
// the reference-pose planes (1.75 / -0.65) for the static scenery only
export const SABER_LEN = 1.0, REACH_Z = 1.75, Z_MISS = 0.65;
// env v15: the reference body plane the head-anchored track is measured
// from (head z at the reference pose). Z_MISS === -PLANE_Z0.
export const PLANE_Z0 = -0.65;
// header value that turns the head-anchored track on (strict gate)
export const TRACK_ANCHOR_HEAD = "head";
// env v17 (contact spec §12a): a wall's front face runs 0.25 m BEHIND
// the note plane (ScoreSaber WallManager build quirk):
//     z_front(t) = (wall.t - t) * speed + WALL_Z_SHIFT + z_off
// and the box extends AWAY from the player to z_front + l.
export const WALL_Z_SHIFT = -0.25;
export const ARC_WINDOW = 2.5; // s: max head->tail gap for an arc pair

// object kinds (chart entry `kind`)
export const KIND_NOTE = 0, KIND_BOMB = 1, KIND_CHAIN_HEAD = 2,
  KIND_CHAIN_LINK = 3;
// Visual sizes follow ScoreSaber's replay viewer models (src/lib/models.js
// is the authority): bomb axis extent 0.324, spikes to r 0.345 (collision radius stays
// 0.18), chain link 0.5 x 0.087 x 0.5 slab, chain head = the note minus
// its swing-side portion (y in [-0.0435, 0.25] of the note frame).
export const BOMB_RADIUS = 0.324;
export const LINK_W = 0.5, LINK_H = 0.087, LINK_D = 0.5;

// real-BS hitbox sizes (for the optional 'h' wireframe overlay). Since
// env v7.4 hitboxes ROTATE with the object about the approach (z) axis
// (see hitboxRot below); half-extents are given in the box's local frame.
export const HB = {
  goodHalf: [0.4, 0.25, 0.5],      // arrow notes / chain heads
  goodHalfDot: [0.4, 0.4, 0.5],    // dot notes
  goodHalfLink: [0.4, 0.1, 0.5],   // chain links
  goodZShift: -0.25,               // box center = note_z - 0.25 (toward player)
  badHalf: [0.175, 0.175, 0.175],  // wrong-color small box
  badHalfLink: [0.175, 0.05, 0.175],
  bombR: 0.18,
};

// note cubes are drawn rotated about the approach axis so the block
// visually tracks its required cut direction (dot notes stay square).
export const noteRot = (n) => (n.any ? 0 : n.angle - 1.5 * Math.PI);

// env v7.4: hitboxes rotate about the approach (z) axis by
// rot = angle - 270° too, mirroring lib.rs note_rot.
export const hitboxRot = (n) =>
  n.kind !== KIND_BOMB && (n.kind === KIND_CHAIN_LINK || !n.any)
    ? n.angle - 1.5 * Math.PI
    : 0;

// --- env v7.6 look-at-player note orientation (mirror of lib.rs
// note_frame). Env v10: the sim billboards to the LIVE head; the viewer
// uses the replayed head position via noteFrame's `head` argument
// (defaults to the reference anchor for callers without a pose).
export const HEAD = [0, 1.7, -0.65]; // reference head anchor
export const LOOK_VERT = 0.8; // head.y damped 80% toward note.y
export const LOOK_DIST = 8.0; // full head-facing at z=0, none beyond 8 m

/** Look-at blend weight for an object at world z. */
export const noteLam = (n, z) =>
  n.kind === KIND_NOTE ? Math.min(1, Math.max(0, 1 - z / LOOK_DIST)) : 0;

/** Orthonormal note frame [x', y', z'] (world-coord columns) at world
 * position p with z-roll psi: z' = the blended forward, up = the rolled
 * (-sin psi, cos psi, 0), x' = up x z'. `head` = live head position. */
export function noteFrame(p, psi, lam, head = HEAD) {
  const up = [-Math.sin(psi), Math.cos(psi), 0];
  const hp = [head[0], head[1] + LOOK_VERT * (p[1] - head[1]), head[2]];
  const d = [p[0] - hp[0], p[1] - hp[1], p[2] - hp[2]];
  const dn = Math.max(Math.hypot(...d), 1e-9);
  const g = [
    (lam * d[0]) / dn,
    (lam * d[1]) / dn,
    1 - lam + (lam * d[2]) / dn,
  ];
  const gn = Math.max(Math.hypot(...g), 1e-9);
  const fz = g.map((v) => v / gn);
  const cx = _cross(up, fz);
  const cn = Math.max(_len(cx), 1e-9);
  const fx = cx.map((v) => v / cn);
  return [fx, _cross(fz, fx), fz];
}

/** Scene quaternion [x, y, z, w] of a world note frame. The scene maps
 * world (x, y, z) -> (x, y, -z), so the orientation is CONJUGATED by
 * S = diag(1, 1, -1): M' = S M S (entries with exactly one z index flip
 * sign), which stays a proper rotation. */
export function frameQuatScene(f) {
  const m00 = f[0][0], m01 = f[1][0], m02 = -f[2][0];
  const m10 = f[0][1], m11 = f[1][1], m12 = -f[2][1];
  const m20 = -f[0][2], m21 = -f[1][2], m22 = f[2][2];
  const tr = m00 + m11 + m22;
  if (tr > 0) {
    const s = 2 * Math.sqrt(tr + 1);
    return [(m21 - m12) / s, (m02 - m20) / s, (m10 - m01) / s, s / 4];
  } else if (m00 > m11 && m00 > m22) {
    const s = 2 * Math.sqrt(1 + m00 - m11 - m22);
    return [s / 4, (m01 + m10) / s, (m02 + m20) / s, (m21 - m12) / s];
  } else if (m11 > m22) {
    const s = 2 * Math.sqrt(1 + m11 - m00 - m22);
    return [(m01 + m10) / s, s / 4, (m12 + m21) / s, (m02 - m20) / s];
  }
  const s = 2 * Math.sqrt(1 + m22 - m00 - m11);
  return [(m02 + m20) / s, (m12 + m21) / s, s / 4, (m10 - m01) / s];
}
export const V_MIN = 1, DT = 1 / 30;

// ============================================================================
// v10 body model — MIRROR of rust/src/dynamics/model.rs (SPEC-v10 §1).
// Actuated order (§1.4): spine[pitch,yaw,roll], arm_l[pitch,yaw,twist,
// elbow,roll,bend,dev], arm_r[same], leg_l[hp,hy,hr,knee,ap,ar], leg_r.
// ============================================================================
const RAD = Math.PI / 180;
export const N_JOINTS = 29;
export const JOINT_NAMES = [
  "sp_pitch", "sp_yaw", "sp_roll",
  "pitch", "yaw", "twist", "elbow", "roll", "bend", "dev",
  "pitch", "yaw", "twist", "elbow", "roll", "bend", "dev",
  "hip_p", "hip_y", "hip_r", "knee", "ank_p", "ank_r",
  "hip_p", "hip_y", "hip_r", "knee", "ank_p", "ank_r",
];
// exact limit literals (model.rs HUMANOID_LIMITS)
export const JOINT_LIMITS = [
  [-20 * RAD, 45 * RAD], [-40 * RAD, 40 * RAD], [-30 * RAD, 30 * RAD],
  [-90 * RAD, 65 * RAD], [-100 * RAD, 45 * RAD], [-90 * RAD, 70 * RAD],
  [0, 150 * RAD], [0, Math.PI], [-95 * RAD, 95 * RAD], [-30 * RAD, 15 * RAD],
  [-90 * RAD, 65 * RAD], [-45 * RAD, 100 * RAD], [-70 * RAD, 90 * RAD],
  [0, 150 * RAD], [-Math.PI, 0], [-95 * RAD, 95 * RAD], [-15 * RAD, 30 * RAD],
  [-20 * RAD, 120 * RAD], [-45 * RAD, 40 * RAD], [-45 * RAD, 30 * RAD],
  [0, 150 * RAD], [-50 * RAD, 35 * RAD], [-20 * RAD, 20 * RAD],
  [-20 * RAD, 120 * RAD], [-40 * RAD, 45 * RAD], [-30 * RAD, 45 * RAD],
  [0, 150 * RAD], [-50 * RAD, 35 * RAD], [-20 * RAD, 20 * RAD],
];
export const TAU_MAX = [
  600.00, 239.95, 375.00,
  85.38, 85.38, 19.45, 19.13, 1.06, 1.06, 1.06,
  85.38, 85.38, 19.45, 19.13, 1.06, 1.06, 1.06,
  262.50, 75.00, 135.00, 150.00, 135.00, 37.50,
  262.50, 75.00, 135.00, 150.00, 135.00, 37.50,
];
export const OMEGA_CAP = JOINT_LIMITS.map((_, j) =>
  j === 7 || j === 8 || j === 9 || j === 14 || j === 15 || j === 16 ? 40 : 16);
// gauge grouping (§1.4 blocks)
export const GROUPS = [
  { name: "spine", joints: [0, 1, 2] },
  { name: "L arm", joints: [3, 4, 5, 6, 7, 8, 9] },
  { name: "R arm", joints: [10, 11, 12, 13, 14, 15, 16] },
  { name: "L leg", joints: [17, 18, 19, 20, 21, 22] },
  { name: "R leg", joints: [23, 24, 25, 26, 27, 28] },
];

// env v8 gear tables (constants.gear_table mirror): id 0 coast,
// 1/2 = ±full at every K; K=5 appends ±1/4; K=7 appends ±1/3, ±1/8.
export function gearTable(n) {
  const mags = { 3: [1], 5: [1, 0.25], 7: [1, 1 / 3, 0.125] }[n] ?? [1];
  const out = [0];
  for (const m of mags) out.push(m, -m);
  return out;
}

/** Applied torque per joint this tick: gear fraction × τ_max. */
export function tauOf(action, gears) {
  const table = gearTable(gears);
  return action.map((a, j) => (table[a] ?? 0) * TAU_MAX[j]);
}

// Kinematic frame table (model.rs HUMANOID_FRAMES): [parent, anchor,
// axis, coord] in chain (stacking) order; parent -1 = the free root
// (pelvis). Stacking order != actuated order for the shoulder
// (yaw→pitch→twist) and wrist (roll→dev→bend) clusters.
const AX = [1, 0, 0], AXN = [-1, 0, 0], AY = [0, 1, 0], AZ = [0, 0, 1];
const Z3 = [0, 0, 0];
export const FRAMES = [
  [-1, [0, 0.10, 0], AX, 0],           //  0 spine_pitch
  [0, Z3, AY, 1],                      //  1 spine_yaw
  [1, Z3, AZ, 2],                      //  2 spine_roll + chest
  [2, [0, 0.66, 0], AZ, -1],           //  3 head (WELD)
  [2, [-0.20, 0.43, 0], AY, 4],        //  4 l_sh_yaw
  [4, Z3, AXN, 3],                     //  5 l_sh_pitch
  [5, Z3, AZ, 5],                      //  6 l_sh_twist + uarm_l
  [6, [0, 0, 0.35], AXN, 6],           //  7 l_elbow + farm_l
  [7, [0, 0, 0.30], AZ, 7],            //  8 l_wr_roll
  [8, Z3, AY, 9],                      //  9 l_wr_dev
  [9, Z3, AXN, 8],                     // 10 l_wr_bend + hand_l
  [2, [0.20, 0.43, 0], AY, 11],        // 11 r_sh_yaw
  [11, Z3, AXN, 10],                   // 12 r_sh_pitch
  [12, Z3, AZ, 12],                    // 13 r_sh_twist + uarm_r
  [13, [0, 0, 0.35], AXN, 13],         // 14 r_elbow + farm_r
  [14, [0, 0, 0.30], AZ, 14],          // 15 r_wr_roll
  [15, Z3, AY, 16],                    // 16 r_wr_dev
  [16, Z3, AXN, 15],                   // 17 r_wr_bend + hand_r
  [-1, [-0.10, 0, 0], AXN, 17],        // 18 l_hip_pitch
  [18, Z3, AY, 18],                    // 19 l_hip_yaw
  [19, Z3, AZ, 19],                    // 20 l_hip_roll + thigh_l
  [20, [0, -0.44, 0], AX, 20],         // 21 l_knee + shank_l
  [21, [0, -0.43, 0], AXN, 21],        // 22 l_ankle_pitch
  [22, Z3, AZ, 22],                    // 23 l_ankle_roll + foot_l
  [-1, [0.10, 0, 0], AXN, 23],         // 24 r_hip_pitch
  [24, Z3, AY, 24],                    // 25 r_hip_yaw
  [25, Z3, AZ, 25],                    // 26 r_hip_roll + thigh_r
  [26, [0, -0.44, 0], AX, 26],         // 27 r_knee + shank_r
  [27, [0, -0.43, 0], AXN, 27],        // 28 r_ankle_pitch
  [28, Z3, AZ, 28],                    // 29 r_ankle_roll + foot_r
];
export const F_CHEST = 2, F_HEAD = 3;
export const F_L_SHOULDER = 4, F_L_ELBOW = 7, F_L_WRIST = 8, F_L_HAND = 10;
export const F_R_SHOULDER = 11, F_R_ELBOW = 14, F_R_WRIST = 15,
  F_R_HAND = 17;
export const F_L_THIGH = 20, F_L_SHANK = 21, F_L_FOOT = 23;
export const F_R_THIGH = 26, F_R_SHANK = 27, F_R_FOOT = 29;
export const ROOT_POS0 = [0, 0.94, -0.65];

// collider-carrying frames per SPEC-v10 §2 pair ids 0-11 (for contact
// markers the point comes from the replay; this maps id -> label)
export const COLLIDER_NAMES = [
  "torso", "head", "uarm_l", "uarm_r", "farm_l", "farm_r",
  "fist_l", "fist_r", "thigh_l", "thigh_r", "shank_l", "shank_r",
];
export const PAIRS = (() => {
  const excluded = new Set(["0,1", "0,2", "0,3", "0,8", "0,9", "2,4",
    "3,5", "4,6", "5,7", "8,10", "9,11"]);
  const out = [];
  for (let i = 0; i < 12; i++)
    for (let j = i + 1; j < 12; j++)
      if (!excluded.has(`${i},${j}`)) out.push([i, j]);
  return out; // 55 pairs, index = normative pair id
})();

function rotAxis(axis, angle) {
  const s = Math.sin(angle), c = Math.cos(angle), k = 1 - c;
  const [x, y, z] = axis;
  return [
    [c + k * x * x, k * x * y - s * z, k * x * z + s * y],
    [k * x * y + s * z, c + k * y * y, k * y * z - s * x],
    [k * x * z - s * y, k * y * z + s * x, c + k * z * z],
  ];
}

/** World-from-body rotation of a wxyz (scalar-first) quaternion. */
export function quatToRot(q) {
  const [w, x, y, z] = q;
  return [
    [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
    [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
    [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
  ];
}

const matMul = (a, b) => {
  const o = [[0, 0, 0], [0, 0, 0], [0, 0, 0]];
  for (let i = 0; i < 3; i++)
    for (let j = 0; j < 3; j++)
      o[i][j] = a[i][0] * b[0][j] + a[i][1] * b[1][j] + a[i][2] * b[2][j];
  return o;
};
const matVec = (m, v) => [
  m[0][0] * v[0] + m[0][1] * v[1] + m[0][2] * v[2],
  m[1][0] * v[0] + m[1][1] * v[1] + m[1][2] * v[2],
  m[2][0] * v[0] + m[2][1] * v[1] + m[2][2] * v[2],
];

/** Full-body FK from a replay q36 = [root pos(3), root quat wxyz(4),
 * 29 joint angles]. Returns { r, p }: per-frame world rotation and
 * origin in FRAMES order (30 entries, welded head included) — the JS
 * mirror of rust/src/dynamics/fk.rs fk_all. */
export function bodyFK(q36) {
  const p0 = [q36[0], q36[1], q36[2]];
  const r0 = quatToRot([q36[3], q36[4], q36[5], q36[6]]);
  const r = [], p = [];
  for (const [parent, anchor, axis, coord] of FRAMES) {
    const rl = parent < 0 ? r0 : r[parent];
    const pl = parent < 0 ? p0 : p[parent];
    const av = matVec(rl, anchor);
    p.push([pl[0] + av[0], pl[1] + av[1], pl[2] + av[2]]);
    r.push(coord >= 0 ? matMul(rl, rotAxis(axis, q36[7 + coord])) : rl);
  }
  return { r, p, r0, p0 };
}

/** World point `local` expressed in frame k of an FK result. */
export function framePoint(fk, k, local) {
  const v = matVec(fk.r[k], local);
  return [fk.p[k][0] + v[0], fk.p[k][1] + v[1], fk.p[k][2] + v[2]];
}

/** Wrist position + unit blade dir per hand from an FK result:
 * [wx, wy, wz, bx, by, bz] — the replay `sabers` layout. */
export function fkSaber(fk, hand) {
  const w = fk.p[hand === 0 ? F_L_WRIST : F_R_WRIST];
  const r = fk.r[hand === 0 ? F_L_HAND : F_R_HAND];
  return [w[0], w[1], w[2], r[0][2], r[1][2], r[2][2]];
}

/** Interpolated q36 at (tickIdx + u): root position and joints lerp,
 * root quaternion nlerp (short arc). */
export function qAt(rep, tickIdx, u) {
  const a = rep.ticks[Math.min(tickIdx, rep.ticks.length - 1)].q;
  const b = rep.ticks[Math.min(tickIdx + 1, rep.ticks.length - 1)].q;
  const out = new Array(36);
  for (let i = 0; i < 3; i++) out[i] = lerp(a[i], b[i], u);
  // nlerp with sign fix (short arc)
  let dot = 0;
  for (let i = 3; i < 7; i++) dot += a[i] * b[i];
  const s = dot < 0 ? -1 : 1;
  let n = 0;
  for (let i = 3; i < 7; i++) {
    out[i] = lerp(a[i], s * b[i], u);
    n += out[i] * out[i];
  }
  n = Math.sqrt(n) || 1;
  for (let i = 3; i < 7; i++) out[i] /= n;
  for (let i = 7; i < 36; i++) out[i] = lerp(a[i], b[i], u);
  return out;
}

// ---- manual pose mode (pure helpers; UI in PosePanel.svelte) ---------------
// Pose state: 29 joint angles (§1.4 order) + root {pos, yaw, pitch,
// roll}. The preview runs entirely through bodyFK (no sim roundtrip).

/** q36 from a pose { q: [29], root: {x, y, z, yaw, pitch, roll} } —
 * root orientation composed yaw(Y)·pitch(X)·roll(Z), intrinsic. */
export function poseQ36(pose) {
  const { x, y, z, yaw, pitch, roll } = pose.root;
  const cy = Math.cos(yaw / 2), sy = Math.sin(yaw / 2);
  const cp = Math.cos(pitch / 2), sp = Math.sin(pitch / 2);
  const cr = Math.cos(roll / 2), sr = Math.sin(roll / 2);
  // q = qy ⊗ qx ⊗ qz (wxyz)
  const qy = [cy, 0, sy, 0], qx = [cp, sp, 0, 0], qz = [cr, 0, 0, sr];
  const mul = (a, b) => [
    a[0] * b[0] - a[1] * b[1] - a[2] * b[2] - a[3] * b[3],
    a[0] * b[1] + a[1] * b[0] + a[2] * b[3] - a[3] * b[2],
    a[0] * b[2] - a[1] * b[3] + a[2] * b[0] + a[3] * b[1],
    a[0] * b[3] + a[1] * b[2] - a[2] * b[1] + a[3] * b[0],
  ];
  const q = mul(mul(qy, qx), qz);
  return [x, y, z, ...q, ...pose.q];
}

/** Clamp a 29-joint pose into the anatomical ranges. */
export function clampQ29(q) {
  return q.map((v, j) =>
    Math.min(JOINT_LIMITS[j][1], Math.max(JOINT_LIMITS[j][0], v)));
}

/** Mirror a full 29-joint pose across the body midplane: arm blocks
 * swap L<->R (yaw/twist/roll/dev flip sign; pitch/elbow/bend keep),
 * leg blocks swap (hip_yaw/hip_roll/ankle_roll flip), spine yaw/roll
 * flip. The per-side ranges are exact mirrors, so legal poses mirror
 * to legal poses. */
export function mirrorQ29(q) {
  const out = new Array(29);
  [out[0], out[1], out[2]] = [q[0], -q[1], -q[2]];
  const arm = (s, d) => {
    out[d] = q[s];           // pitch
    out[d + 1] = -q[s + 1];  // yaw
    out[d + 2] = -q[s + 2];  // twist
    out[d + 3] = q[s + 3];   // elbow
    out[d + 4] = -q[s + 4];  // roll
    out[d + 5] = q[s + 5];   // bend
    out[d + 6] = -q[s + 6];  // dev
  };
  arm(3, 10);
  arm(10, 3);
  const leg = (s, d) => {
    out[d] = q[s];           // hip_pitch
    out[d + 1] = -q[s + 1];  // hip_yaw
    out[d + 2] = -q[s + 2];  // hip_roll
    out[d + 3] = q[s + 3];   // knee
    out[d + 4] = q[s + 4];   // ankle_pitch
    out[d + 5] = -q[s + 5];  // ankle_roll
  };
  leg(17, 23);
  leg(23, 17);
  return out;
}

// UI palette (HUD text, gauges, trails). The 3D objects use ScoreSaber's
// default note colours instead — NOTE_COLORS in models.js.
export const SABER_COLORS = ["#e06c75", "#61afef"]; // left/red, right/blue
export const SABER_DARK = ["#7c3b41", "#33586f"];
export const GOOD_COL = "#7dd3a0", BAD_COL = "#f0a35e", MISS_COL = "#e06c75";

// per-kind max points (real BS: 115 note, 85 chain head, 20 link, bombs 0)
export const KIND_MAX = [115, 0, 85, 20];

export function parseReplay(text) {
  const lines = text.split("\n").filter((l) => l.trim());
  const head = JSON.parse(lines[0]);
  if (!head.header) throw new Error("not a beat-arena replay");
  const v = head.header.version;
  if (v !== "v10")
    throw new Error(
      "replay version " + v + " — this viewer reads v10 (env v10 " +
      "full-body dynamics); re-record from a current checkpoint",
    );
  const ticks = lines.slice(1).map((l) => JSON.parse(l));
  const chart = head.header.chart;
  const outcomes = chart.map(() => null);
  const posts = []; // follow-through resolutions: {note, tick, points}
  ticks.forEach((t) =>
    t.events.forEach((e) => {
      if (e.cut)
        outcomes[e.cut.note] = {
          kind: e.cut.good ? "good" : "bad",
          reason: e.cut.reason,
          tick: t.t,
          points: e.cut.points || 0,
        };
      if (e.post)
        posts.push({ note: e.post.note, tick: t.t, points: e.post.points });
      if (e.miss !== undefined)
        outcomes[e.miss.note] = {
          kind: "miss", reason: "miss", tick: t.t, points: 0,
        };
      if (e.bomb !== undefined)
        outcomes[e.bomb.note] = {
          kind: "bomb", reason: "bomb", tick: t.t, points: 0,
        };
    }),
  );
  return {
    config: head.header.config, chart, ticks, outcomes, posts,
    // env v15 ADDITIVE header field: "head" -> head-anchored note track.
    // Anything else (including absent) means the pre-v15 FIXED track.
    trackAnchor: head.header.track_anchor ?? null,
    // env v17 ADDITIVE header field: world-space wall AABBs, geometry
    // finished by the parser ({t, cx, cy, w, h, l, speed} per wall).
    // Pre-v17 recordings (field absent) render exactly as before.
    walls: head.header.walls ?? [],
    arcs: arcPairs(chart),
  };
}

/** True when this replay's chart objects ride the v15 head-anchored
 * track. STRICT gate: the header field must be present and "head". */
export function isHeadAnchored(rep) {
  return !!rep && rep.trackAnchor === TRACK_ANCHOR_HEAD;
}

// The head-anchored offset needs one full-body FK per query; playback
// asks for the same (tick, u) many times per frame, so memoize the last.
// Integer-tick (u = 0) offsets are also re-asked every frame by the
// aftermath/post effects (their zOff is FROZEN at the spawn tick, each
// at its own k, which would thrash a single-slot memo) — cache those
// per replay so each spawn tick costs one FK ever.
let _offMemo = { rep: null, k: -1, u: -1, v: 0 };
const _offTicks = new WeakMap(); // rep -> Map(k -> z_off at (k, 0))

/** World z of the HEAD frame origin at fractional tick (tickIdx + u) —
 * the same head the viewer draws the skeleton's head from, and the one
 * the sim anchors the track to (lib.rs `self.body.head[2]`). */
export function headZAt(rep, tickIdx, u = 0) {
  return bodyFK(qAt(rep, tickIdx, u)).p[F_HEAD][2];
}

/** v15 track offset z_off = head_z - PLANE_Z0 at fractional tick
 * (tickIdx + u); exactly 0 for pre-v15 replays (no "track_anchor" in the
 * header), so legacy recordings render on the old fixed track. */
export function trackOffset(rep, tickIdx, u = 0) {
  if (!isHeadAnchored(rep) || !rep.ticks || !rep.ticks.length) return 0;
  const k = Math.max(0, Math.min(Math.floor(tickIdx), rep.ticks.length - 1));
  const uu = Math.min(1, Math.max(0, u));
  if (uu === 0) {
    let m = _offTicks.get(rep);
    if (!m) _offTicks.set(rep, (m = new Map()));
    let v = m.get(k);
    if (v === undefined) m.set(k, (v = headZAt(rep, k, 0) - PLANE_Z0));
    return v;
  }
  if (_offMemo.rep === rep && _offMemo.k === k && _offMemo.u === uu)
    return _offMemo.v;
  const v = headZAt(rep, k, uu) - PLANE_Z0;
  _offMemo = { rep, k, u: uu, v };
  return v;
}

/** Same, for a continuous replay time in seconds. */
export function trackOffsetAt(rep, time) {
  const f = time / DT;
  const k = Math.floor(f);
  return trackOffset(rep, k, f - k);
}

/** z_off of the pose/body the viewer is currently DRAWING, from its FK
 * result — used so notes line up with the sabers frame-for-frame. */
export function trackOffsetOfFK(fk) {
  return fk ? fk.p[F_HEAD][2] - PLANE_Z0 : 0;
}

/** THE viewer's track-offset gate, in one place: `rep` = the loaded
 * replay (or null), `poseOn` = manual pose mode, `fk` = FK of the body
 * being drawn. A loaded replay decides STRICTLY by its header — only
 * "track_anchor": "head" shifts anything, so pre-v15 recordings keep the
 * old fixed track. With no replay loaded, pose mode shows the current
 * (v15) env's head-anchored track. */
export function viewerTrackOffset(rep, poseOn, fk) {
  if (!fk) return 0;
  if (rep) return isHeadAnchored(rep) ? trackOffsetOfFK(fk) : 0;
  return poseOn ? trackOffsetOfFK(fk) : 0;
}

/** Arc scoring hints: pair each fpost note (arc head) with the NEXT
 * same-color fpre note (arc tail) within ARC_WINDOW seconds. */
export function arcPairs(chart) {
  const arcs = [];
  chart.forEach((n, i) => {
    if (!n.fpost) return;
    for (let j = i + 1; j < chart.length; j++) {
      const m = chart[j];
      if (m.t - n.t > ARC_WINDOW) break;
      if (m.fpre && m.color === n.color && m.t > n.t) {
        arcs.push({ head: i, tail: j });
        break;
      }
    }
  });
  return arcs;
}

export function maxScore(chart) {
  return chart.reduce((a, n) => a + KIND_MAX[n.kind], 0);
}

export function configSummary(rep) {
  const c = rep.config;
  const spd = c.note_speed_min != null
    ? `${c.note_speed_min}-${c.note_speed_max} m/s`
    : (c.note_speed != null ? `${c.note_speed} m/s` : null);
  const parts = [c.colors, spd];
  if (c.gears != null) parts.push(`${c.gears} gears`);
  if (c.note_prob != null) parts.push(`p ${c.note_prob}`);
  if (c.extra_prob) parts.push(`extras ${c.extra_prob}`);
  if (c.dot_prob) parts.push(`dots ${c.dot_prob}`);
  if (c.bomb_prob) parts.push(`bombs ${c.bomb_prob}`);
  return `${rep.chart.length} objects, ${rep.ticks.length} ticks ` +
    `(${parts.filter(Boolean).join(", ")})`;
}

/** In-plane world (x, y) — explicit in the v7+ chart. */
export function noteCenter(n) {
  return [n.x, n.y];
}

export const lerp = (a, b, u) => a + (b - a) * u;

export function lerpAng(a, b, u) {
  let d = (b - a) % (2 * Math.PI);
  if (d > Math.PI) d -= 2 * Math.PI;
  if (d < -Math.PI) d += 2 * Math.PI;
  return a + d * u;
}

/** Interpolated 6-field saber [wx,wy,wz,bx,by,bz] at (tickIdx + u):
 * wrist lerped, blade lerp-normalized — from the LOGGED sabers (the
 * sim's own FK; the skeleton uses bodyFK(qAt(...)) instead, and the
 * two agree because the logged fields ARE that FK's outputs). */
export function saberAt(rep, tickIdx, u, s) {
  const a = rep.ticks[Math.min(tickIdx, rep.ticks.length - 1)].sabers[s];
  const b = rep.ticks[Math.min(tickIdx + 1, rep.ticks.length - 1)].sabers[s];
  const out = [];
  for (let i = 0; i < 3; i++) out.push(lerp(a[i], b[i], u));
  let bx = lerp(a[3], b[3], u), by = lerp(a[4], b[4], u),
    bz = lerp(a[5], b[5], u);
  const n = Math.hypot(bx, by, bz) || 1;
  out.push(bx / n, by / n, bz / n);
  return out;
}

/** Blade tip in WORLD coords [x, y, z] from a 6-field saber. */
export function tipOf(sab) {
  return [
    sab[0] + SABER_LEN * sab[3],
    sab[1] + SABER_LEN * sab[4],
    sab[2] + SABER_LEN * sab[5],
  ];
}

const _dot = (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
const _sub = (a, b) => [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
const _cross = (a, b) => [
  a[1] * b[2] - a[2] * b[1],
  a[2] * b[0] - a[0] * b[2],
  a[0] * b[1] - a[1] * b[0],
];
const _len = (a) => Math.hypot(a[0], a[1], a[2]);

/** The ACTUAL cut plane of a good cut, reconstructed from the cutting
 * hand's blade sweep at the cut tick, in NOTE-LOCAL coords. Returns
 * { p, n, v, b } or null when the sweep is degenerate (pure stab). */
export function cutPlaneLocal(rep, noteIdx) {
  const n = rep.chart[noteIdx], o = rep.outcomes[noteIdx];
  if (!o || o.kind !== "good") return null;
  const k = Math.min(o.tick, rep.ticks.length - 1);
  const k0 = Math.max(0, k - 1);
  const hand = n.color; // good cuts are matching-hand by rule
  const prev = rep.ticks[k0].sabers[hand];
  const cur = rep.ticks[k].sabers[hand];
  const pw = [prev[0], prev[1], prev[2]], pd = [prev[3], prev[4], prev[5]];
  const cw = [cur[0], cur[1], cur[2]], cd = [cur[3], cur[4], cur[5]];
  // v15: the note really sat at base + z_off(tick k) when the blade met
  // it — the sabers below are that same tick's, so the offset must be
  // taken there for the reconstructed plane to be the sim's plane.
  const zCut = (n.t - (k + 1) * DT) * n.speed + trackOffset(rep, k);
  const [cx, cy] = noteCenter(n);
  const center = [cx, cy, zCut];
  // contact fraction: closest point of the tick-k blade line to center
  const s = Math.min(Math.max(_dot(_sub(center, cw), cd), 0), SABER_LEN);
  const p1 = [0, 1, 2].map((i) => cw[i] + s * cd[i]);
  const p0 = [0, 1, 2].map((i) => pw[i] + s * pd[i]);
  const sweep = _sub(p1, p0);
  const nrm = _cross(cd, sweep);
  const nl = _len(nrm), vl = _len(sweep);
  if (nl < 1e-6 || vl < 1e-9) return null;
  // world -> note-local: R(-rot) about z, origin at the note center
  const rot = noteRot(n);
  const c = Math.cos(rot), sn = Math.sin(rot);
  const loc = (w) => [c * w[0] + sn * w[1], -sn * w[0] + c * w[1], w[2]];
  const p = loc(_sub(p1, center)).map(
    (x) => Math.min(Math.max(x, -HALF * 0.95), HALF * 0.95));
  return {
    p,
    n: loc(nrm.map((x) => x / nl)),
    v: loc(sweep.map((x) => x / vl)),
    b: loc(cd),
  };
}

/** Contact-capability: tip speed via finite difference across one tick
 * (contact-velocity semantics). */
export function tipSpeed(rep, tickIdx, s) {
  const k = Math.min(tickIdx, rep.ticks.length - 1);
  const k0 = Math.max(0, k - 1);
  const t1 = tipOf(rep.ticks[k].sabers[s]);
  const t0 = tipOf(rep.ticks[k0].sabers[s]);
  if (k === k0) return 0;
  return Math.hypot(t1[0] - t0[0], t1[1] - t0[1], t1[2] - t0[2]) / DT;
}

/** Cumulative HUD counters up to and including uptoTick. O(1) per call:
 * the running totals are built once per replay (lazily, cached by rep) by
 * the same loop in the same order the old per-call scan summed them, so
 * every float comes out bit-identical — the HUD asks at every tick of
 * playback, which made the per-call O(ticks) scan quadratic overall. */
const _countsCache = new WeakMap(); // rep -> per-tick cumulative arrays
export function counts(rep, uptoTick) {
  let cum = _countsCache.get(rep);
  if (!cum) {
    const n = rep.ticks.length;
    cum = {
      good: new Int32Array(n), bad: new Int32Array(n),
      miss: new Int32Array(n), bombs: new Int32Array(n),
      reward: new Float64Array(n), score: new Float64Array(n),
    };
    const c = { good: 0, bad: 0, miss: 0, bombs: 0, reward: 0, score: 0 };
    for (let k = 0; k < n; k++) {
      c.reward += rep.ticks[k].reward;
      rep.ticks[k].events.forEach((e) => {
        if (e.cut) {
          c[e.cut.good ? "good" : "bad"]++;
          c.score += e.cut.points || 0;
        }
        if (e.post) c.score += e.post.points;
        if (e.miss !== undefined) c.miss++;
        if (e.bomb !== undefined) c.bombs++;
      });
      cum.good[k] = c.good; cum.bad[k] = c.bad;
      cum.miss[k] = c.miss; cum.bombs[k] = c.bombs;
      cum.reward[k] = c.reward; cum.score[k] = c.score;
    }
    _countsCache.set(rep, cum);
  }
  const k = Math.min(uptoTick, rep.ticks.length - 1);
  if (k < 0) return { good: 0, bad: 0, miss: 0, bombs: 0, reward: 0, score: 0 };
  return {
    good: cum.good[k], bad: cum.bad[k], miss: cum.miss[k],
    bombs: cum.bombs[k], reward: cum.reward[k], score: cum.score[k],
  };
}

/** Self-collision contact markers visible at `time` (0.4 s flash):
 * world point, tick-summed impulse (N·s), age-based alpha, radius
 * scaled by impulse. Reads the replay `contacts` events. */
export const CONTACT_FADE_S = 0.4;
export function activeContacts(rep, time) {
  const out = [];
  const t0 = Math.max(0, Math.floor(time / DT) - Math.ceil(CONTACT_FADE_S / DT));
  const t1 = Math.min(rep.ticks.length - 1, Math.floor(time / DT));
  for (let k = t0; k <= t1; k++) {
    const tk = rep.ticks[k];
    if (!tk.contacts) continue;
    const age = time - (tk.t + 1) * DT;
    if (age < 0 || age > CONTACT_FADE_S) continue;
    for (const c of tk.contacts) {
      out.push({
        pair: c.pair, p: c.p, j: c.j, age,
        alpha: 1 - age / CONTACT_FADE_S,
        // radius: 2 cm at tiny impulses -> 8 cm at ~5 N·s
        radius: 0.02 + 0.06 * Math.min(1, Math.abs(c.j) / 5),
      });
    }
  }
  return out;
}

/** Per-foot vertical ground reaction (N) at a tick, from the additive
 * "feet" weld-impulse field; null when the replay predates it. */
export function footForces(rep, tickIdx) {
  const t = rep.ticks[Math.min(tickIdx, rep.ticks.length - 1)];
  if (!t.feet) return null;
  return [t.feet[0][1] / DT, t.feet[1][1] / DT];
}

/** Effort (J of positive work) at a tick; null when absent. */
export function effortAt(rep, tickIdx) {
  const t = rep.ticks[Math.min(tickIdx, rep.ticks.length - 1)];
  return t.effort ?? null;
}

/** Objects still flying at `time`, with their world z. Bombs that were
 * never hit produce NO event — they despawn at the miss boundary.
 *
 * `zOff` is the v15 head-anchored track offset (0 for pre-v15 replays).
 * Each entry carries BOTH z's: `z` is the rendered world z (base + zOff,
 * what the sim judges against and what must line up with the sabers) and
 * `zBase` is the chart-time z. The spawn/despawn bounds stay on `zBase`
 * because the sim's boundaries move with the track too — `note_z < head_z`
 * reduces to `base z < PLANE_Z0` (spec §10b), and the far bound shifts
 * with z_off since z_off is shared by every note (§10a). */
export function pendingNotes(rep, time, zOff = 0) {
  const out = [];
  rep.chart.forEach((n, i) => {
    const o = rep.outcomes[i];
    if (o && time >= (o.tick + 1) * DT) return;
    const zBase = (n.t - time) * n.speed;
    if (zBase <= Z_FAR && zBase >= -Z_MISS)
      out.push({ n, i, z: zBase + zOff, zBase });
  });
  return out;
}

/** env v17 walls overlapping the visible tunnel at `time`. Walls ride
 * the SAME head-anchored track as notes (spec §12a), so `zOff` is the
 * v15 offset already computed for pendingNotes; like there, visibility
 * bounds stay on the chart-time z (the far bound and the head plane
 * shift with z_off together). Pre-v17 replays have walls = [] and
 * render exactly as before. */
export function pendingWalls(rep, time, zOff = 0) {
  const out = [];
  (rep.walls ?? []).forEach((w, i) => {
    const zFrontBase = (w.t - time) * w.speed + WALL_Z_SHIFT;
    if (zFrontBase <= Z_FAR && zFrontBase + w.l >= -Z_MISS)
      out.push({ w, i, zFront: zFrontBase + zOff, zFrontBase });
  });
  return out;
}

// 0.7 s, matching the reference viewer's indicatorLifetime: the flying hit
// score (Notes.svelte) fades over exactly this window, so a shorter one
// would cut its tail off. The cut debris and fail outlines share it.
const FADE_S = 0.7;

/** Resolved-note effects (0.6 s window): split halves / link pop /
 * bomb flash / fail outline. Each entry carries the v15 track offset
 * FROZEN at the frame the effect spawns on (the viewer draws the body of
 * tick o.tick + 1 then): the debris stays where the note actually was
 * when it resolved, instead of sliding with later head motion. */
export function aftermaths(rep, time) {
  const out = [];
  rep.chart.forEach((n, i) => {
    const o = rep.outcomes[i];
    if (!o) return;
    const age = time - (o.tick + 1) * DT;
    if (age < 0 || age > FADE_S) return;
    out.push({
      n, i, o, age, alpha: 1 - age / FADE_S,
      zOff: trackOffset(rep, o.tick + 1),
    });
  });
  return out;
}

/** Follow-through "+N follow" popups active at `time`. Single pass that
 * only allocates for the (few) active popups — the map/filter/map chain
 * cloned every post on the replay every frame. */
export function activePosts(rep, time) {
  const out = [];
  for (const p of rep.posts) {
    const age = time - (p.tick + 1) * DT;
    if (age < 0 || age > FADE_S) continue;
    out.push({
      ...p, age, alpha: 1 - age / FADE_S, zOff: trackOffset(rep, p.tick + 1),
    });
  }
  return out;
}

/** Arc hint tubes visible at `time`: head/tail world positions of each
 * paired arc whose span overlaps the visible tunnel. `zOff` is the v15
 * track offset (see pendingNotes): the tube rides the track, the
 * visibility bounds stay on the chart-time z. */
export function activeArcs(rep, time, zOff = 0) {
  const out = [];
  rep.arcs.forEach(({ head, tail }, k) => {
    const h = rep.chart[head], t = rep.chart[tail];
    const zh = (h.t - time) * h.speed;
    const zt = (t.t - time) * t.speed;
    if (zt > Z_FAR || zh < -Z_MISS) return;
    out.push({
      k, head, tail, color: h.color,
      a: [...noteCenter(h), zh + zOff],
      b: [...noteCenter(t), zt + zOff],
    });
  });
  return out;
}
