// Replay format v6 (SPEC.md §Replay log): env v6 kinematic upper body.
// Pure logic only (no three/svelte imports) — unit-tested in node.
//
// World: hit plane z = 0; grid spans [-1,1] x [-0.75,0.75] (0.5 m lanes);
// 0.4 m note cubes approach from +z. Sabers = 15 numbers per hand:
// [q(6) = pitch, yaw, twist, elbow, roll, bend | om(6) | wrist x, y, z].
// Each hand is a chain from a fixed shoulder: upper arm -> elbow ->
// forearm -> wrist (roll + bend) -> 1 m blade; armFK mirrors the sim's
// forward kinematics exactly.
// The Three.js scene maps world (x, y, z) -> (x, y, -z): the camera sits at
// scene +z looking down -z, preserving the player's view (x right, y up,
// notes approaching).

export const CELL = 0.5, BLOCK = 0.4, HALF = BLOCK / 2;
export const GX = 1.0, GY = 0.75, Z_FAR = 12.0;
export const SABER_LEN = 1.0, REACH_Z = 1.5, Z_MISS = 0.6;

// env v5.3: note cubes are rotated about the approach axis so the block
// tracks its required cut direction (down-swing = axis-aligned); dot
// notes stay square. Mirrors the sims' collision volume.
export const noteRot = (n) => (n.any ? 0 : n.angle - 1.5 * Math.PI);
export const V_MIN = 1, DT = 1 / 30;
// env v6 body + anatomical joint limits (mirror the sim)
const RAD = Math.PI / 180;
export const SHOULDER = [[-0.2, 0.45, -0.35], [0.2, 0.45, -0.35]];
export const L_UPPER = 0.35, L_FORE = 0.3;
export const PITCH_LIM = 90 * RAD;
export const YAW_LO = [-100 * RAD, -45 * RAD]; // left, right
export const YAW_HI = [45 * RAD, 100 * RAD];
export const TWIST_LIM = 90 * RAD;
export const ELBOW_HI = 150 * RAD;
export const ROLL_LO = [-90 * RAD, -190 * RAD]; // left, right
export const ROLL_HI = [190 * RAD, 90 * RAD];
export const BEND_LIM = 95 * RAD;
export const OMEGA_ARM = 8, OMEGA_WRIST = 20;
export const SABER_COLORS = ["#e06c75", "#61afef"]; // left/red, right/blue
export const SABER_DARK = ["#7c3b41", "#33586f"];
export const GOOD_COL = "#7dd3a0", BAD_COL = "#f0a35e", MISS_COL = "#e06c75";

export function parseReplay(text) {
  const lines = text.split("\n").filter((l) => l.trim());
  const head = JSON.parse(lines[0]);
  if (!head.header) throw new Error("not a beat-arena replay");
  const v = head.header.version;
  if (v !== 6)
    throw new Error(
      "replay version " + v + " — this viewer reads v6 (env v6 kinematic " +
      "upper body); older replays use retired saber parameterizations",
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
    }),
  );
  return { config: head.header.config, chart, ticks, outcomes, posts };
}

export function configSummary(rep) {
  const c = rep.config;
  const spd = c.note_speed_max
    ? `${c.note_speed}-${c.note_speed_max} m/s`
    : `${c.note_speed} m/s`;
  const dirs = c.continuous_dirs ? "360°" : `dirs [${c.dirs}]`;
  return `${rep.chart.length} notes, ${rep.ticks.length} ticks ` +
    `(rate ${c.note_rate}, ${c.colors}, ${dirs}, ${spd})`;
}

export function noteCenter(n) {
  return [(n.col - 1.5) * CELL, (n.row - 1.0) * CELL];
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
  const c = { good: 0, bad: 0, miss: 0, reward: 0, score: 0 };
  for (let k = 0; k <= uptoTick && k < rep.ticks.length; k++) {
    c.reward += rep.ticks[k].reward;
    rep.ticks[k].events.forEach((e) => {
      if (e.cut) {
        c[e.cut.good ? "good" : "bad"]++;
        c.score += e.cut.points || 0;
      }
      if (e.post) c.score += e.post.points;
      if (e.miss !== undefined) c.miss++;
    });
  }
  return c;
}

/** Notes still flying at `time`, with their world z. */
export function pendingNotes(rep, time) {
  const out = [];
  rep.chart.forEach((n, i) => {
    const o = rep.outcomes[i];
    if (o && time >= (o.tick + 1) * DT) return;
    const z = (n.t - time) * n.speed;
    if (z <= Z_FAR) out.push({ n, i, z });
  });
  return out;
}

const FADE_S = 0.6;

/** Resolved-note effects (0.6 s window): split halves / fail outline. */
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
