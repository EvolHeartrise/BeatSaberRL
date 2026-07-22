// Replay format v5 (SPEC.md §Replay log): env v5.2 forearm-anchored blades.
// Pure logic only (no three/svelte imports) — unit-tested in node.
//
// World: hit plane z = 0; grid spans [-1,1] x [-0.75,0.75] (0.5 m lanes);
// 0.4 m note cubes approach from +z. Sabers = [x, y, vx, vy, roll, bend,
// om_roll, om_bend]: wrist on the plane + 1 m blade whose neutral points AT
// the notes; roll orients the bend plane (forearm CW/CCW), bend deflects
// away from forward (+-95 deg — it can never point back at the player).
// The Three.js scene maps world (x, y, z) -> (x, y, -z): the camera sits at
// scene +z looking down -z, preserving the player's view (x right, y up,
// notes approaching).

export const CELL = 0.5, BLOCK = 0.4, HALF = BLOCK / 2;
export const GX = 1.0, GY = 0.75, Z_FAR = 12.0;
export const SABER_LEN = 1.0, REACH_Z = 1.28, Z_MISS = 0.6;
export const V_MIN = 1, DT = 1 / 30;
// env v5.2 anatomical joint limits (mirror the sim): forearm roll per
// hand, wrist bend +-95 deg from straight-forward
const RAD = Math.PI / 180;
export const ROLL_LO = [-90 * RAD, -190 * RAD]; // left, right
export const ROLL_HI = [190 * RAD, 90 * RAD];
export const BEND_LIM = 95 * RAD;
export const OMEGA_MAX = 20;
export const SABER_COLORS = ["#e06c75", "#61afef"]; // left/red, right/blue
export const SABER_DARK = ["#7c3b41", "#33586f"];
export const GOOD_COL = "#7dd3a0", BAD_COL = "#f0a35e", MISS_COL = "#e06c75";

export function parseReplay(text) {
  const lines = text.split("\n").filter((l) => l.trim());
  const head = JSON.parse(lines[0]);
  if (!head.header) throw new Error("not a beat-arena replay");
  const v = head.header.version;
  if (v !== 5)
    throw new Error(
      "replay version " + v + " — this viewer reads v5 (env v5.2 forearm-" +
      "anchored blades); older replays use retired saber parameterizations",
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

export function bladeDir(roll, bend) {
  const cr = Math.cos(roll), sr = Math.sin(roll);
  const cb = Math.cos(bend), sb = Math.sin(bend);
  return [-sr * sb, cr * sb, cb];
}

export function bladeDirDot(roll, bend, wr, wb) {
  const cr = Math.cos(roll), sr = Math.sin(roll);
  const cb = Math.cos(bend), sb = Math.sin(bend);
  return [
    -wr * cr * sb - wb * sr * cb,
    -wr * sr * sb + wb * cr * cb,
    -wb * sb,
  ];
}

/** Interpolated 8-field saber state at (tickIdx + u). */
export function saberAt(rep, tickIdx, u, s) {
  const a = rep.ticks[Math.min(tickIdx, rep.ticks.length - 1)].sabers[s];
  const b = rep.ticks[Math.min(tickIdx + 1, rep.ticks.length - 1)].sabers[s];
  return [
    lerp(a[0], b[0], u), lerp(a[1], b[1], u), b[2], b[3],
    lerpAng(a[4], b[4], u), lerp(a[5], b[5], u), b[6], b[7],
  ];
}

/** Blade tip in WORLD coords [x, y, z] from an 8-field saber state. */
export function tipOf(sab) {
  const d = bladeDir(sab[4], sab[5]);
  return [
    sab[0] + SABER_LEN * d[0],
    sab[1] + SABER_LEN * d[1],
    SABER_LEN * d[2],
  ];
}

/** Contact-capability: speed of the blade TIP (wrist velocity + rotation). */
export function tipSpeed(sab) {
  const dd = bladeDirDot(sab[4], sab[5], sab[6], sab[7]);
  return Math.hypot(
    sab[2] + SABER_LEN * dd[0],
    sab[3] + SABER_LEN * dd[1],
    SABER_LEN * dd[2],
  );
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
