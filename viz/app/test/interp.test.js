// The pose between ticks (replay.js qAt): joint-space cubic Hermite through
// the recorded angles with the recorded joint rates as tangents — hits
// every tick exactly, is C1 across ticks, stays inside the joint limits
// (widened to the endpoints), and falls back to the old lerp without ω.
// Plus the shared blade history (lib/bladetrail.js): the ribbon's tip edge
// IS the tip line, and both end on the drawn blade.
// Run: node test/interp.test.js
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import * as R from "../src/lib/replay.js";
import { BASE, RIBBON_S, TIP, createBladeTrail } from "../src/lib/bladetrail.js";
import { FIXTURE, readReplayFile } from "./fixture.js";

const rep = R.parseReplay(readReplayFile(FIXTURE));
const N = rep.ticks.length, hz = R.tickHzOf(rep), dt = 1 / hz;
assert.ok(rep.ticks.every((t) => t.omega?.length === 29), "fixture carries ω");
const near = (a, b, tol, msg) => assert.ok(Math.abs(a - b) <= tol, `${msg}: ${a} vs ${b}`);

// a synthetic replay: joints mid-range, smooth-ish random rates (no clamps)
function synth({ omega = true, twist = false, hzS = 30 } = {}) {
  let seed = 7;
  const rnd = () => ((seed = (seed * 16807) % 2147483647) / 2147483647) * 2 - 1;
  const ticks = [];
  for (let k = 0; k < 40; k++) {
    // a root that turns ~0.08 rad per tick about a wandering axis
    const ax = [Math.sin(0.2 * k), 1, Math.cos(0.3 * k)], an = Math.hypot(...ax);
    const ang = 0.08 * k, sh = Math.sin(ang / 2) / an;
    const q = [0.1 * k, 1, -0.5, Math.cos(ang / 2), ax[0] * sh, ax[1] * sh, ax[2] * sh];
    const w = [];
    for (let j = 0; j < 29; j++) {
      const [lo, hi] = R.JOINT_LIMITS[j];
      q.push((lo + hi) / 2 + 0.15 * (hi - lo) * Math.sin(0.3 * k + j));
      w.push(2 * rnd());
    }
    const t = { q };
    if (omega) t.omega = w;
    if (twist) t.twist = [3, 0.5, -1, 2.4, -1, 1.5].map((s) => s * (1 + 0.5 * rnd()));
    ticks.push(t);
  }
  return { tickHz: hzS, ticks };
}

// ---- exact at the ticks -------------------------------------------------------
for (const r of [rep, synth()]) {
  const n = r.ticks.length;
  for (let k = 0; k < n - 1; k++) {
    const q0 = R.qAt(r, k, 0), q1 = R.qAt(r, k, 1), qn = R.qAt(r, k + 1, 0);
    for (let i = 7; i < 36; i++) {
      assert.equal(q0[i], r.ticks[k].q[i], `u=0 is tick ${k}`);
      near(q1[i], r.ticks[k + 1].q[i], 1e-12, `u=1 is tick ${k + 1}`);
      near(q1[i], qn[i], 1e-12, `continuous at tick ${k + 1}`);
    }
  }
}

// ---- C1: the slope on both sides of a tick is the recorded rate ------------
{
  const r = synth(), d = 1 / r.tickHz, h = 1e-5;
  for (let k = 1; k < r.ticks.length - 1; k++) {
    const right = R.qAt(r, k, h), q = r.ticks[k].q;
    const left = R.qAt(r, k - 1, 1 - h);
    for (let j = R.HERMITE_FIRST; j <= R.HERMITE_LAST; j++) {
      const want = r.ticks[k].omega[j] * d; // dq/du, u in ticks
      near((right[7 + j] - q[7 + j]) / h, want, 1e-3, `slope after tick ${k} joint ${j}`);
      near((q[7 + j] - left[7 + j]) / h, want, 1e-3, `slope before tick ${k} joint ${j}`);
    }
  }
  // and the midpoint is the Hermite's: (qa + qb)/2 + dt/8 (ωa - ωb)
  const m = R.qAt(r, 5, 0.5), a = r.ticks[5], b = r.ticks[6];
  for (let j = R.HERMITE_FIRST; j <= R.HERMITE_LAST; j++)
    near(m[7 + j], (a.q[7 + j] + b.q[7 + j]) / 2 + (d / 8) * (a.omega[j] - b.omega[j]),
      1e-12, `Hermite midpoint joint ${j}`);
  // the tangent scale follows the replay's own rate (60 Hz: half the dt)
  const r60 = synth({ hzS: 60 }), m60 = R.qAt(r60, 5, 0.5);
  near(m60[10] - (r60.ticks[5].q[10] + r60.ticks[6].q[10]) / 2,
    (1 / 480) * (r60.ticks[5].omega[3] - r60.ticks[6].omega[3]), 1e-12, "60 Hz tangent scale");
  // the spine and legs (root-coupled) stay linear even with ω
  assert.equal(R.HERMITE_FIRST, 3);
  assert.equal(R.HERMITE_LAST, 16);
  for (const u of [0.2, 0.5, 0.9]) {
    const q = R.qAt(r, 7, u), qa = r.ticks[7].q, qb = r.ticks[8].q;
    for (let j = 0; j < 29; j++) {
      if (j >= R.HERMITE_FIRST && j <= R.HERMITE_LAST) continue;
      assert.equal(q[7 + j], R.lerp(qa[7 + j], qb[7 + j], u), `joint ${j} is lerped`);
    }
  }
}

// ---- without ω: exactly the old lerp -----------------------------------------
{
  const r = synth({ omega: false });
  for (const u of [0.1, 0.5, 0.77]) {
    const q = R.qAt(r, 3, u), a = r.ticks[3].q, b = r.ticks[4].q;
    for (let i = 7; i < 36; i++) assert.equal(q[i], R.lerp(a[i], b[i], u), "linear fallback");
    for (let i = 0; i < 3; i++) assert.equal(q[i], R.lerp(a[i], b[i], u), "root position lerp");
  }
}

// ---- root: unchanged (position lerp, quaternion nlerp) -----------------------
{
  const r = synth(), q = R.qAt(r, 2, 0.3);
  for (let i = 0; i < 3; i++) assert.equal(q[i], R.lerp(r.ticks[2].q[i], r.ticks[3].q[i], 0.3));
}

// ---- limits: an overshoot is clamped; a recorded angle is never moved ------
{
  const r = synth();
  const j = 6; // L elbow, limits [0, 150°]
  const hi = R.JOINT_LIMITS[j][1];
  r.ticks[10].q[7 + j] = hi - 0.01; r.ticks[10].omega[j] = 30; // rushing at the stop
  r.ticks[11].q[7 + j] = hi - 0.01; r.ticks[11].omega[j] = -30;
  for (const u of [0.25, 0.5, 0.75]) {
    const v = R.qAt(r, 10, u)[7 + j];
    assert.ok(v <= hi + 1e-15, `clamped to the limit at u=${u}: ${v}`);
  }
  assert.equal(R.qAt(r, 10, 0.5)[7 + j], hi, "the overshoot sits on the limit");
  // an endpoint recorded past the limit widens the bound to it
  r.ticks[20].q[7 + j] = hi + 0.002; r.ticks[20].omega[j] = 0;
  r.ticks[21].q[7 + j] = hi + 0.002; r.ticks[21].omega[j] = 0;
  assert.equal(R.qAt(r, 20, 0)[7 + j], hi + 0.002);
  near(R.qAt(r, 20, 0.5)[7 + j], hi + 0.002, 1e-15, "flat between two over-limit ticks");
}

// ---- `out` is filled in place ---------------------------------------------------
{
  const out = new Array(36);
  assert.equal(R.qAt(rep, 10, 0.4, out), out);
  assert.deepEqual(out, R.qAt(rep, 10, 0.4));
}

// ---- the full path: root Hermite with the recorded twist, all 29 joints ----
const qmul = (a, b) => [a[0] * b[0] - a[1] * b[1] - a[2] * b[2] - a[3] * b[3],
  a[0] * b[1] + a[1] * b[0] + a[2] * b[3] - a[3] * b[2],
  a[0] * b[2] - a[1] * b[3] + a[2] * b[0] + a[3] * b[1],
  a[0] * b[3] + a[1] * b[2] - a[2] * b[1] + a[3] * b[0]];
const conj = (q) => [q[0], -q[1], -q[2], -q[3]];
const qlog = (q) => {
  const s = q[0] < 0 ? -1 : 1, v = [s * q[1], s * q[2], s * q[3]];
  const n = Math.hypot(...v), a = 2 * Math.atan2(n, s * q[0]);
  return n < 1e-15 ? v.map((x) => 2 * x) : v.map((x) => (x * a) / n);
};
const rot = (q) => q.slice(3, 7);
{
  const r = synth({ twist: true }), d = 1 / r.tickHz, h = 1e-6, n = r.ticks.length;
  for (let k = 1; k < n - 1; k++) {
    const t = r.ticks[k];
    // exact at the ticks: position and rotation from both sides
    for (const q of [R.qAt(r, k, 0), R.qAt(r, k - 1, 1)]) {
      for (let i = 0; i < 3; i++) near(q[i], t.q[i], 1e-12, `root position at tick ${k}`);
      near(Math.hypot(...qlog(qmul(rot(q), conj(rot(t.q))))), 0, 1e-9, `root rotation at tick ${k}`);
    }
    // the tangents at the ticks are the recorded twist, from both sides:
    // linear velocity (m/s) and WORLD angular velocity (rad/s)
    const after = R.qAt(r, k, h), before = R.qAt(r, k - 1, 1 - h), at0 = R.qAt(r, k, 0);
    for (let i = 0; i < 3; i++) {
      near((after[i] - at0[i]) / (h * d), t.twist[i], 1e-4, `v after tick ${k}`);
      near((at0[i] - before[i]) / (h * d), t.twist[i], 1e-4, `v before tick ${k}`);
    }
    const wA = qlog(qmul(rot(after), conj(rot(at0)))).map((x) => x / (h * d));
    const wB = qlog(qmul(rot(at0), conj(rot(before)))).map((x) => x / (h * d));
    for (let i = 0; i < 3; i++) {
      near(wA[i], t.twist[3 + i], 1e-4, `ω after tick ${k} axis ${i}`);
      near(wB[i], t.twist[3 + i], 1e-4, `ω before tick ${k} axis ${i}`);
    }
    // all 29 joints are Hermite now (spine and legs included), C1
    for (let j = 0; j < 29; j++) {
      const want = t.omega[j] * d;
      near((after[7 + j] - at0[7 + j]) / h, want, 1e-4, `joint ${j} slope after tick ${k}`);
      near((at0[7 + j] - before[7 + j]) / h, want, 1e-4, `joint ${j} slope before tick ${k}`);
    }
  }
  // a unit quaternion everywhere in between
  for (const u of [0.1, 0.5, 0.93]) near(Math.hypot(...rot(R.qAt(r, 9, u))), 1, 1e-12, "unit");
}

// ---- fallback coverage ----------------------------------------------------------
{
  const a = synth({ twist: true }), c = synth({ omega: false, twist: true });
  const b = { ...a, ticks: a.ticks.map(({ twist, ...t }) => t) }; // same ticks, no twist
  // twist on one tick only: that interval takes the arms-only path
  const mixed = synth({ twist: true });
  delete mixed.ticks[6].twist;
  const half = R.qAt(mixed, 5, 0.4), armsOnly = R.qAt(b, 5, 0.4);
  for (let i = 0; i < 36; i++) assert.equal(half[i], armsOnly[i], "one tick without twist: arms-only");
  const ok = R.qAt(mixed, 7, 0.4), full = R.qAt(a, 7, 0.4);
  for (let i = 0; i < 36; i++) assert.equal(ok[i], full[i], "the next interval is full again");
  // twist but no ω: everything lerps
  const lin = R.qAt(c, 3, 0.6);
  for (let i = 0; i < 3; i++) assert.equal(lin[i], R.lerp(c.ticks[3].q[i], c.ticks[4].q[i], 0.6));
  for (let i = 7; i < 36; i++) assert.equal(lin[i], R.lerp(c.ticks[3].q[i], c.ticks[4].q[i], 0.6));
}
// the committed fixtures take the full path: a format-2 replay and a Flux run
{
  const flux = R.parseReplay(readFileSync(new URL("./fixtures/flux_scripted.jsonl", import.meta.url), "utf8"));
  for (const [name, r] of [["format-2 fixture", rep], ["Flux fixture", flux]]) {
    assert.ok(r.ticks.every((t) => t.twist?.length === 6 && t.omega?.length === 29), `${name} carries twist + ω`);
    // a spine joint (lerped on the arms-only path) follows its Hermite
    const k = 40, a = r.ticks[k], b = r.ticks[k + 1], d = 1 / R.tickHzOf(r);
    const want = (a.q[7] + b.q[7]) / 2 + (d / 8) * (a.omega[0] - b.omega[0]);
    const lim = R.JOINT_LIMITS[0];
    const lo = Math.min(lim[0], a.q[7], b.q[7]), hi = Math.max(lim[1], a.q[7], b.q[7]);
    near(R.qAt(r, k, 0.5)[7], Math.min(hi, Math.max(lo, want)), 1e-12, `${name}: spine on the full path`);
  }
  // human (WebXR) runs carry no body: the scene draws them from the palms
  // and head (humanPalmsAt / humanHeadAt) and never calls qAt
  assert.equal(R.isHuman({ ticks: [], player: "human" }), true, "the gate Scene.svelte uses");
}

// ---- the shared blade history on the fixture -----------------------------------
const hl = R.handleLen(rep);
const drawnTip = (cur, s) => { // Scene.svelte + Saber.svelte: the drawn blade
  const k = Math.min(Math.floor(cur), N - 1);
  return R.tipOf(R.fkSaber(R.bodyFK(R.qAt(rep, k, cur - k)), s), hl);
};
const drawnBase = (cur, s) => {
  const k = Math.min(Math.floor(cur), N - 1);
  return R.baseOf(R.fkSaber(R.bodyFK(R.qAt(rep, k, cur - k)), s), hl);
};
const F32 = 2e-6; // float32 storage of world coords (~2 m)
{
  const bt = createBladeTrail();
  let cur = 100.13;
  for (let f = 0; f < 120; f++, cur += 0.5) { // 0.5 tick/frame (2x at 60 fps)
    bt.ensure(rep, cur, 0.2);
    const rib = bt.view(RIBBON_S), line = bt.view(0.2);
    const S = 12, off = line.npts - rib.npts;
    assert.ok(rib.npts >= RIBBON_S * 240, "ribbon on the 240 Hz grid");
    for (let s = 0; s < 2; s++) {
      // the ribbon's tip edge is the tip line, point for point (its own
      // exact tail aside)
      for (let i = 1; i < rib.npts; i++)
        for (let c = 0; c < 3; c++)
          near(rib.pts[i * S + TIP(s) + c], line.pts[(off + i) * S + TIP(s) + c], 0,
            `ribbon edge == tip line (frame ${f}, pt ${i})`);
      // both end on the drawn blade, tip and base
      const tp = drawnTip(cur, s), bp = drawnBase(cur, s), h = (rib.npts - 1) * S;
      const want = [tp[0], tp[1], -tp[2]], wantB = [bp[0], bp[1], -bp[2]];
      for (let c = 0; c < 3; c++) {
        near(rib.pts[h + TIP(s) + c], want[c], F32, "ribbon head = drawn tip");
        near(rib.pts[h + BASE(s) + c], wantB[c], F32, "ribbon head = drawn base");
        near(line.pts[(line.npts - 1) * S + TIP(s) + c], want[c], F32, "line head = drawn tip");
      }
      // the tail: exactly RIBBON_S back
      const tt = drawnTip(cur - RIBBON_S * hz, s);
      for (let c = 0; c < 3; c++)
        near(rib.pts[TIP(s) + c], c === 2 ? -tt[c] : tt[c], F32, "ribbon tail");
    }
  }
  // seek back and forward: the same as a fresh history
  for (const to of [40.5, 41.2, 300.9, 0.4]) {
    bt.ensure(rep, to, 0.2);
    const fresh = createBladeTrail();
    fresh.ensure(rep, to, 0.2);
    for (const w of [RIBBON_S, 0.2]) {
      const a = bt.view(w), b = fresh.view(w);
      assert.equal(a.npts, b.npts, `seek to ${to}: point count`);
      for (let p = 0; p < a.npts * 12; p++) near(a.pts[p], b.pts[p], 0, `seek to ${to}`);
    }
  }
  // paused: no grid samples
  const n0 = bt.trail.ring.sampled;
  bt.ensure(rep, 0.4, 0.2);
  assert.equal(bt.trail.ring.sampled, n0);
  // a new replay resets the history
  const other = R.parseReplay(readReplayFile(FIXTURE));
  const fills = bt.trail.ring.fills;
  bt.ensure(other, 0.4, 0.2);
  assert.equal(bt.trail.ring.fills, fills + 1, "a new replay refills");
}

console.log(`interp OK: root + joint Hermite exact at ticks, tangents = twist / ω, ` +
  `fallbacks (arms-only, linear), clamped; ` +
  `ribbon edge == tip line == drawn blade`);
