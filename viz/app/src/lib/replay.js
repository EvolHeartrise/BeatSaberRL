// Replay format v7 (SPEC.md §Replay log): env v7 real-BS geometry.
// Pure logic only (no three/svelte imports) — unit-tested in node.
//
// World (env v7): floor-based coordinates, y = 0 is the floor. Lanes at
// x = -0.9, -0.3, 0.3, 0.9 (0.6 m pitch); row centers y = 0.85, 1.40,
// 1.90. Hit plane z = 0; objects approach from +z; misses register at
// z = -0.65 (the head plane). Chart entries carry EXPLICIT world
// (x, y) plus a kind: 0 note, 1 bomb, 2 chain head, 3 chain link, and
// arc grants fpre/fpost. Sabers = 15 numbers per hand:
// [q(6) = pitch, yaw, twist, elbow, roll, bend | om(6) | wrist x, y, z].
// Each hand is a chain from a fixed shoulder (±0.2, 1.47, -0.65):
// upper arm -> elbow -> forearm -> wrist (roll + bend) -> 1 m blade;
// armFK mirrors the sim's forward kinematics exactly.
// The Three.js scene maps world (x, y, z) -> (x, y, -z): the camera sits at
// scene +z looking down -z, preserving the player's view (x right, y up,
// notes approaching).

// BLOCK 0.5: the real game's visual cube (verified against the
// ScoreSaber viewer's note mesh, 2026-07-30 — 0.4 under-drew notes 20%
// and exaggerated the hitbox-vs-visual overhang in the 'h' overlay)
export const CELL = 0.6, BLOCK = 0.5, HALF = BLOCK / 2;
export const LANE_X = [-0.9, -0.3, 0.3, 0.9];
export const ROW_Y = [0.85, 1.4, 1.9];
// grid frame: lane cell edges; row edges midway between the non-uniform
// row centers (extrapolated at the ends)
export const GX0 = -1.2, GX1 = 1.2;
export const GY0 = 0.575, GY1 = 2.15;
export const ROW_EDGES = [GY0, 1.125, 1.65, GY1];
export const Z_FAR = 12.0;
// env v7.3: the miss plane is the HEAD plane (z = -0.65) — a deliberate
// deviation from the real game's despawn (~2 m behind the head) to
// tighten credit assignment; mirrors constants.py / lib.rs
export const SABER_LEN = 1.0, REACH_Z = 1.75, Z_MISS = 0.65;
export const ARC_WINDOW = 2.5; // s: max head->tail gap for an arc pair

// object kinds (chart entry `kind`)
export const KIND_NOTE = 0, KIND_BOMB = 1, KIND_CHAIN_HEAD = 2,
  KIND_CHAIN_LINK = 3;
export const BOMB_RADIUS = 0.2; // visual (collision radius is 0.18)
export const LINK_W = 0.4, LINK_H = 0.13, LINK_D = 0.13; // link slab visual
export const HEAD_SQUASH = 0.5; // chain heads are ~half-height cubes

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
// rot = angle - 270° too, mirroring lib.rs note_rot. (v7.0 drew/judged
// them axis-aligned — a BeatLeader-viewer simplification mistaken for
// game behavior; the real colliders sit on the rotated note transform.)
// Arrow notes and chain heads rotate by their angle; chain links by
// their tangent (their `angle` field) even though links judge
// any-direction; plain dots and bombs stay unrotated. Differs from
// noteRot only for links (any=true but still rotating).
export const hitboxRot = (n) =>
  n.kind !== KIND_BOMB && (n.kind === KIND_CHAIN_LINK || !n.any)
    ? n.angle - 1.5 * Math.PI
    : 0;

// --- env v7.6: look-at-player note orientation (mirror of lib.rs
// note_frame). Plain notes (kind 0, dots included) bias their FULL 3D
// orientation toward facing the player's head as they approach: the
// head's y is damped 80% toward the note's y, the forward axis blends
// from +z to the head->note ray with weight lam = clamp(1 - z/8, 0, 1)
// — our analog of the game's jump progress (full head-facing at the
// plane) — and the cut-direction z-roll rides along as the frame's up.
// lam = 0 reproduces the v7.4 pure z-roll frame EXACTLY (chain heads
// and links stay there; bombs are spheres). Both the visuals and the
// hitboxes live on this transform, exactly as in the sim.
export const HEAD = [0, 1.7, -0.65]; // fixed head anchor (1.8 m player)
export const LOOK_VERT = 0.8; // head.y damped 80% toward note.y
export const LOOK_DIST = 8.0; // full head-facing at z=0, none beyond 8 m

/** Look-at blend weight for an object at world z. */
export const noteLam = (n, z) =>
  n.kind === KIND_NOTE ? Math.min(1, Math.max(0, 1 - z / LOOK_DIST)) : 0;

/** Orthonormal note frame [x', y', z'] (world-coord columns) at world
 * position p with z-roll psi: z' = the blended forward, up = the rolled
 * (-sin psi, cos psi, 0), x' = up x z', y' = z' x x'. */
export function noteFrame(p, psi, lam) {
  const up = [-Math.sin(psi), Math.cos(psi), 0];
  const hp = [HEAD[0], HEAD[1] + LOOK_VERT * (p[1] - HEAD[1]), HEAD[2]];
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
// env v6 body (v7 placement) + anatomical joint limits (mirror the sim)
const RAD = Math.PI / 180;
export const SHOULDER = [[-0.2, 1.47, -0.65], [0.2, 1.47, -0.65]];
export const L_UPPER = 0.35, L_FORE = 0.3;
// env v7.2: pitch capped at +65 deg (up); -90 straight-down unchanged
export const PITCH_LO = -90 * RAD, PITCH_HI = 65 * RAD;
export const YAW_LO = [-100 * RAD, -45 * RAD]; // left, right
export const YAW_HI = [45 * RAD, 100 * RAD];
// env v7.1: twist and roll tightened, per-hand mirrored ranges
export const TWIST_LO = [-90 * RAD, -70 * RAD]; // left, right
export const TWIST_HI = [70 * RAD, 90 * RAD];
export const ELBOW_HI = 150 * RAD;
export const ROLL_LO = [0 * RAD, -180 * RAD];   // left, right
export const ROLL_HI = [180 * RAD, 0 * RAD];
export const BEND_LIM = 95 * RAD;
export const OMEGA_ARM = 8, OMEGA_WRIST = 20;

// ---- manual pose mode (pure helpers; UI in PosePanel.svelte) ---------------
export const JOINT_NAMES = ["pitch", "yaw", "twist", "elbow", "roll", "bend"];

/** [(lo, hi)] x 6 for one hand, in joint order (mirror of constants.py). */
export function jointLimits(hand) {
  return [
    [PITCH_LO, PITCH_HI],
    [YAW_LO[hand], YAW_HI[hand]],
    [TWIST_LO[hand], TWIST_HI[hand]],
    [0, ELBOW_HI],
    [ROLL_LO[hand], ROLL_HI[hand]],
    [-BEND_LIM, BEND_LIM],
  ];
}

/** Clamp a 6-joint pose into the hand's anatomical ranges. */
export function clampQ(hand, q) {
  const lims = jointLimits(hand);
  return q.map((v, j) => Math.min(lims[j][1], Math.max(lims[j][0], v)));
}

/** Mirror a pose across the body midplane (left <-> right): yaw, twist
 * and roll flip sign; pitch, elbow and bend are symmetric. The hands'
 * joint ranges are exact mirrors, so a legal pose mirrors to a legal one. */
export const mirrorQ = (q) => [q[0], -q[1], -q[2], q[3], -q[4], q[5]];

/** Synthetic 15-field arm state from a static pose: q(6) | omegas all
 * zero | FK wrist(3). Drives the same rendering path as replay ticks. */
export function poseSaber(hand, q) {
  const { wrist } = armFK(hand, q);
  return [...q, 0, 0, 0, 0, 0, 0, ...wrist];
}
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
  if (v !== 7)
    throw new Error(
      "replay version " + v + " — this viewer reads v7 (env v7 real Beat " +
      "Saber geometry); re-record from a current checkpoint",
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
    arcs: arcPairs(chart),
  };
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
  // generator v2 headers carry note_speed_min/prob dials; real-map and
  // pre-v2 headers may lack them — show what exists
  const spd = c.note_speed_min != null
    ? `${c.note_speed_min}-${c.note_speed_max} m/s`
    : (c.note_speed != null ? `${c.note_speed} m/s` : null);
  const parts = [c.colors, spd];
  if (c.note_prob != null) parts.push(`p ${c.note_prob}`);
  if (c.extra_prob) parts.push(`extras ${c.extra_prob}`);
  if (c.dot_prob) parts.push(`dots ${c.dot_prob}`);
  if (c.bomb_prob) parts.push(`bombs ${c.bomb_prob}`);
  return `${rep.chart.length} objects, ${rep.ticks.length} ticks ` +
    `(${parts.filter(Boolean).join(", ")})`;
}

/** In-plane world (x, y) — explicit in the v7 chart. */
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

/** Forward kinematics — MIRROR of the sim's fk() (SPEC §Sim rules).
 * q = [pitch, yaw, twist, elbow, roll, bend] (world coords, z toward
 * the notes). Returns {elbow, wrist, blade} — blade is a unit vector. */
export function armFK(hand, q) {
  const sp = Math.sin(q[0]), cp = Math.cos(q[0]);
  const sy = Math.sin(q[1]), cy = Math.cos(q[1]);
  const st = Math.sin(q[2]), ct = Math.cos(q[2]);
  const se = Math.sin(q[3]), ce = Math.cos(q[3]);
  const m = [
    [cy * ct - sy * sp * st, -cy * st - sy * sp * ct, sy * cp],
    [cp * st, cp * ct, sp],
    [-sy * ct - cy * sp * st, sy * st - cy * sp * ct, cy * cp],
  ];
  const sh = SHOULDER[hand];
  const elbow = [0, 1, 2].map((i) => sh[i] + L_UPPER * m[i][2]);
  const f = [0, 1, 2].map((i) => m[i][1] * se + m[i][2] * ce);
  const wrist = [0, 1, 2].map((i) => elbow[i] + L_FORE * f[i]);
  const sr = Math.sin(q[4]), cr = Math.cos(q[4]);
  const sb = Math.sin(q[5]), cb = Math.cos(q[5]);
  const d = [-sr * sb, cr * sb, cb];
  const blade = [0, 1, 2].map(
    (i) => m[i][0] * d[0] + (m[i][1] * ce - m[i][2] * se) * d[1] + f[i] * d[2]);
  return { elbow, wrist, blade };
}

/** Interpolated 15-field arm state at (tickIdx + u): joint angles lerped
 * (all joints are bounded, no wrap), omegas + wrist from the later tick. */
export function saberAt(rep, tickIdx, u, s) {
  const a = rep.ticks[Math.min(tickIdx, rep.ticks.length - 1)].sabers[s];
  const b = rep.ticks[Math.min(tickIdx + 1, rep.ticks.length - 1)].sabers[s];
  const out = [];
  for (let j = 0; j < 6; j++) out.push(lerp(a[j], b[j], u));
  for (let j = 6; j < 15; j++) out.push(b[j]);
  return out;
}

/** Blade tip in WORLD coords [x, y, z] from a 15-field arm state. */
export function tipOf(hand, sab) {
  const { wrist, blade } = armFK(hand, sab);
  return [0, 1, 2].map((i) => wrist[i] + SABER_LEN * blade[i]);
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
 * hand's blade sweep at the cut tick, in NOTE-LOCAL coords (the frame
 * the cube is drawn in: rotated by noteRot, origin at the note center,
 * z toward the notes). Returns { p (point on plane, clamped into the
 * cube), n (unit normal), v (unit sweep dir), b (unit blade dir) } or
 * null when the sweep is degenerate (pure stab: sweep parallel to the
 * blade) — callers fall back to the canonical center split. */
export function cutPlaneLocal(rep, noteIdx) {
  const n = rep.chart[noteIdx], o = rep.outcomes[noteIdx];
  if (!o || o.kind !== "good") return null;
  const k = Math.min(o.tick, rep.ticks.length - 1);
  const k0 = Math.max(0, k - 1);
  const hand = n.color; // good cuts are matching-hand by rule
  const prev = armFK(hand, rep.ticks[k0].sabers[hand]);
  const cur = armFK(hand, rep.ticks[k].sabers[hand]);
  const zCut = (n.t - (k + 1) * DT) * n.speed;
  const [cx, cy] = noteCenter(n);
  const center = [cx, cy, zCut];
  // contact fraction: closest point of the tick-k blade line to center
  const s = Math.min(Math.max(_dot(_sub(center, cur.wrist), cur.blade), 0),
    SABER_LEN);
  const p1 = [0, 1, 2].map((i) => cur.wrist[i] + s * cur.blade[i]);
  const p0 = [0, 1, 2].map((i) => prev.wrist[i] + s * prev.blade[i]);
  const sweep = _sub(p1, p0);
  const nrm = _cross(cur.blade, sweep);
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
    b: loc(cur.blade),
  };
}

/** Contact-capability: tip speed via finite difference across one tick
 * (env v6 contact-velocity semantics). */
export function tipSpeed(rep, tickIdx, s) {
  const k = Math.min(tickIdx, rep.ticks.length - 1);
  const k0 = Math.max(0, k - 1);
  const t1 = tipOf(s, rep.ticks[k].sabers[s]);
  const t0 = tipOf(s, rep.ticks[k0].sabers[s]);
  if (k === k0) return 0;
  return Math.hypot(t1[0] - t0[0], t1[1] - t0[1], t1[2] - t0[2]) / DT;
}

/** Cumulative HUD counters up to and including uptoTick. */
export function counts(rep, uptoTick) {
  const c = { good: 0, bad: 0, miss: 0, bombs: 0, reward: 0, score: 0 };
  for (let k = 0; k <= uptoTick && k < rep.ticks.length; k++) {
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
  }
  return c;
}

/** Objects still flying at `time`, with their world z. Bombs that were
 * never hit produce NO event — they despawn at the miss boundary. */
export function pendingNotes(rep, time) {
  const out = [];
  rep.chart.forEach((n, i) => {
    const o = rep.outcomes[i];
    if (o && time >= (o.tick + 1) * DT) return;
    const z = (n.t - time) * n.speed;
    if (z <= Z_FAR && z >= -Z_MISS) out.push({ n, i, z });
  });
  return out;
}

const FADE_S = 0.6;

/** Resolved-note effects (0.6 s window): split halves / link pop /
 * bomb flash / fail outline. */
export function aftermaths(rep, time) {
  const out = [];
  rep.chart.forEach((n, i) => {
    const o = rep.outcomes[i];
    if (!o) return;
    const age = time - (o.tick + 1) * DT;
    if (age < 0 || age > FADE_S) return;
    out.push({ n, i, o, age, alpha: 1 - age / FADE_S });
  });
  return out;
}

/** Follow-through "+N follow" popups active at `time`. */
export function activePosts(rep, time) {
  return rep.posts
    .map((p) => ({ ...p, age: time - (p.tick + 1) * DT }))
    .filter((p) => p.age >= 0 && p.age <= FADE_S)
    .map((p) => ({ ...p, alpha: 1 - p.age / FADE_S }));
}

/** Arc hint tubes visible at `time`: head/tail world positions of each
 * paired arc whose span overlaps the visible tunnel. */
export function activeArcs(rep, time) {
  const out = [];
  rep.arcs.forEach(({ head, tail }, k) => {
    const h = rep.chart[head], t = rep.chart[tail];
    const zh = (h.t - time) * h.speed;
    const zt = (t.t - time) * t.speed;
    if (zt > Z_FAR || zh < -Z_MISS) return;
    out.push({
      k, head, tail, color: h.color,
      a: [...noteCenter(h), zh],
      b: [...noteCenter(t), zt],
    });
  });
  return out;
}
