// exp/tick-rate (replay record_rev 6): the viewer reads the replay header's
// tick_hz (absent = 30) and every tick <-> seconds conversion follows it.
//
// The 60 Hz replay here is the committed 30 Hz fixture UPSAMPLED: 60 Hz
// tick 2k is 30 Hz tick k, tick 2k+1 the midpoint, and every event of 30
// Hz tick k moves to 60 Hz tick 2k+1 — whose end, (2k+2)/60 s, is the 30
// Hz tick's end (k+1)/30 s. So at any real time the two replays must show
// the same scene: same pose, same flying notes, same outcomes / popups /
// contacts, same track offset. Run: node test/tickrate.test.js
import assert from "node:assert/strict";
import * as R from "../src/lib/replay.js";
import { FOLLOW_SNAP, followClock } from "../src/lib/sync.js";
import { FIXTURE, readReplayFile } from "./fixture.js";

const bytes = readReplayFile(FIXTURE);
const { header, ticks } = R.decodeReplay(bytes);

// ---- header parsing ----------------------------------------------------------
const rep30 = R.parseReplay(bytes);
assert.equal(rep30.tickHz, 30, "absent tick_hz = 30");
assert.equal(R.tickHzOf(rep30), 30);
assert.equal(R.tickHzOf(null), 30, "no replay = 30");
assert.equal(R.tickHzOf({ ticks: [] }), 30, "human/synthetic reps = 30");
assert.throws(() => R.replayFromParts({ ...header, tick_hz: "bogus" }, ticks));

// ---- the 60 Hz upsample ------------------------------------------------------
const mid = (a, b) => a.map((v, i) => Array.isArray(v) ? mid(v, b[i]) : (v + b[i]) / 2);
// The arm joints between ticks are a cubic Hermite through (q, ω) at both
// ends (replay.js qAt). Splitting a cubic at its midpoint gives two cubics
// whose end values and slopes are the midpoint's, so a 60 Hz tick carrying
// the 30 Hz Hermite's midpoint angle AND rate makes the 60 Hz replay trace
// the very same curves. In seconds: q(½) = (qa + qb)/2 + dt/8 (ωa − ωb),
// q'(½) = 1.5 (qb − qa)/dt − (ωa + ωb)/4, dt = 1/30. (The root, spine and
// legs, lerped, stay the linear midpoint.)
const DT30 = 1 / 30;
const isArm = (j) => j >= R.HERMITE_FIRST && j <= R.HERMITE_LAST; // the Hermite joints
const hermiteMid = (a, b) => ({
  q: a.q.map((v, i) => !isArm(i - 7) ? (v + b.q[i]) / 2
    : (v + b.q[i]) / 2 + (DT30 / 8) * (a.omega[i - 7] - b.omega[i - 7])),
  omega: a.omega.map((w, j) =>
    (1.5 * (b.q[7 + j] - a.q[7 + j])) / DT30 - (w + b.omega[j]) / 4),
});
function upsample() {
  const head = { ...header, tick_hz: 60, action_repeat: 1 };
  const t30 = ticks;
  const out = [];
  t30.forEach((t, k) => {
    const nxt = t30[Math.min(k + 1, t30.length - 1)];
    out.push({ ...t, t: 2 * k, time: (2 * k) / 60, events: [], contacts: [] });
    // the odd tick carries the 30 Hz tick's end-of-tick events (its END is
    // (2k+2)/60 = (k+1)/30) and, as pose, the Hermite midpoint of 30 Hz
    // ticks k and k+1 (angle and rate) — exactly what the 30 Hz viewer
    // interpolates at u = 0.5, so the two replays interpolate to the same
    // poses. (The sabers stay the linear midpoint: only tipSpeed reads
    // them, as a one-tick chord.)
    out.push({
      ...t, t: 2 * k + 1, time: (2 * k + 1) / 60,
      ...hermiteMid(t, nxt), sabers: mid(t.sabers, nxt.sabers),
      feet: t.feet, events: t.events, contacts: t.contacts,
    });
  });
  return R.replayFromParts(head, out);
}
const rep60 = upsample();
assert.equal(rep60.tickHz, 60);
assert.equal(rep60.ticks.length, 2 * rep30.ticks.length);
assert.equal(R.tickHzOf(rep60), 60);

// ---- DT follows the active replay -------------------------------------------
R.setTickRate(R.tickHzOf(rep60));
assert.equal(R.DT, 1 / 60);
R.setTickRate(R.tickHzOf(rep30));
assert.equal(R.DT, 1 / 30, "30 Hz is exactly the old constant");
R.setTickRate();
assert.equal(R.DT, 1 / 30);
assert.throws(() => R.setTickRate(0));

// evaluate f with replay `rep` active (DT = its rate)
const at = (rep, f) => { R.setTickRate(R.tickHzOf(rep)); return f(); };

// ---- outcomes sit at (tick + 1) * DT: the same instants at both rates --------
let nOut = 0;
rep30.outcomes.forEach((o, i) => {
  if (!o) return;
  const o60 = rep60.outcomes[i];
  assert.equal(o60.tick, 2 * o.tick + 1);
  const a = at(rep30, () => R.outcomeTime(o));
  const b = at(rep60, () => R.outcomeTime(o60));
  assert.ok(Math.abs(a - b) < 1e-12, `outcome ${i}: ${a} vs ${b}`);
  nOut++;
});
assert.ok(nOut > 5, "fixture has outcomes");

// ---- the scene at real time T: same at both rates ---------------------------
const last = rep30.ticks.length - 1;
let checked = 0, jointsChecked = 0, clampSkipped = 0;
// does the unclamped 30 Hz Hermite of joint j leave its (endpoint-widened)
// limits anywhere in [k, k + 1]?
function clampBinds(k, j) {
  if (!isArm(j)) return false; // lerped: never past its endpoints
  const a = rep30.ticks[k], b = rep30.ticks[k + 1];
  const qa = a.q[7 + j], qb = b.q[7 + j];
  const lo = Math.min(R.JOINT_LIMITS[j][0], qa, qb), hi = Math.max(R.JOINT_LIMITS[j][1], qa, qb);
  for (let i = 0; i <= 64; i++) {
    const u = i / 64, u2 = u * u, u3 = u2 * u;
    const v = (2 * u3 - 3 * u2 + 1) * qa + (3 * u2 - 2 * u3) * qb
      + DT30 * ((u3 - 2 * u2 + u) * a.omega[j] + (u3 - u2) * b.omega[j]);
    if (v < lo - 1e-12 || v > hi + 1e-12) return true;
  }
  return false;
}
for (let k = 0; k < last; k++) {
  for (const frac of [0, 0.25, 0.5, 0.9]) {
    const T = (k + frac) / 30;
    // the replay clock: pb.cur = T / DT, split into tick + fraction
    const pose = (rep) => at(rep, () => {
      const f = T / R.DT, i = Math.floor(f);
      return R.qAt(rep, i, f - i);
    });
    const q30 = pose(rep30), q60 = pose(rep60);
    // root: exact on the 30 Hz grid and at the midpoint; elsewhere both
    // are linear between the same two 30 Hz samples
    for (let j = 0; j < 7; j++)
      assert.ok(Math.abs(q30[j] - q60[j]) < 1e-9, `root q at T=${T}`);
    // joints: the same Hermite curve at both rates — wherever the joint-
    // limit clamp does not bind in this 30 Hz interval (where it does, the
    // clamped curve is no cubic and the split need not reproduce it)
    for (let j = 7; j < 36; j++) {
      if (clampBinds(k, j - 7)) { clampSkipped++; continue; }
      assert.ok(Math.abs(q30[j] - q60[j]) < 1e-9, `joint ${j - 7} at T=${T}: ` +
        `${q30[j]} vs ${q60[j]}`);
      jointsChecked++;
    }
    const ids = (rep) => at(rep, () =>
      R.pendingNotes(rep, T).map((p) => p.i).join(","));
    assert.equal(ids(rep60), ids(rep30), `pending notes at T=${T}`);
    const off = (rep) => at(rep, () => R.trackOffsetAt(rep, T));
    assert.ok(Math.abs(off(rep60) - off(rep30)) < 1e-9, `z_off at T=${T}`);
    const posts = (rep) => at(rep, () =>
      JSON.stringify(R.activePosts(rep, T).map((p) => [p.note, p.points])));
    assert.equal(posts(rep60), posts(rep30), `posts at T=${T}`);
    const aft = (rep) => at(rep, () =>
      R.aftermaths(rep, T).map((a) => a.i ?? a.note).join(","));
    assert.equal(aft(rep60), aft(rep30), `aftermaths at T=${T}`);
    const con = (rep) => at(rep, () => R.activeContacts(rep, T).length);
    assert.equal(con(rep60), con(rep30), `contacts at T=${T}`);
    checked++;
  }
}
assert.ok(checked > 1000);
assert.ok(jointsChecked > 10 * clampSkipped, `joint checks: ${jointsChecked}, clamp-skipped ${clampSkipped}`);

// ---- rates: per-tick quantities scale with the tick length -------------------
{
  // foot force = tick-summed impulse / DT: the same impulse over a 1/60 s
  // tick is twice the force
  const k = 40;
  const f30 = at(rep30, () => R.footForces(rep30, k));
  const f60 = at(rep60, () => R.footForces(rep60, 2 * k));
  assert.ok(Math.abs(f60[0] - 2 * f30[0]) < 1e-9 * Math.max(1, Math.abs(f30[0])));
  // tip speed = one-tick displacement / DT: half the displacement over
  // half the time (tick 2k+1 is the midpoint of 30 Hz ticks k, k+1)
  const v30 = at(rep30, () => R.tipSpeed(rep30, k + 1, 1));
  const v60 = at(rep60, () => R.tipSpeed(rep60, 2 * k + 2, 1));
  assert.ok(Math.abs(v30 - v60) < 1e-6 * Math.max(1, v30), `${v30} vs ${v60}`);
}

// ---- the replay clock (sync.js): ticks per second = the replay's rate --------
{
  const d = 1 / 144;
  assert.equal(followClock({ cur: 0, target: null, delta: d }), d * 30);
  assert.equal(followClock({ cur: 0, target: null, delta: d, tps: 60 }), d * 60);
  // the snap threshold is 0.5 s at any rate: 15 ticks at 30 Hz, 30 at 60
  assert.equal(FOLLOW_SNAP, 15);
  const free = 5 + d * 60;
  assert.notEqual(followClock({ cur: 5, target: free + 20, delta: d, tps: 60 }),
                  free + 20, "20 ticks at 60 Hz (0.33 s) is drift, eased");
  assert.equal(followClock({ cur: 5, target: free + 31, delta: d, tps: 60 }),
               free + 31, "31 ticks at 60 Hz (> 0.5 s) snaps");
  // audio sync: song time of a frame = offset + pb.cur * DT, and the song
  // clock's tick = (currentTime - offset) / DT — inverse at any rate
  for (const rep of [rep30, rep60]) at(rep, () => {
    const offset = 1.234, cur = 417.5;
    const song = offset + cur * R.DT;
    assert.ok(Math.abs((song - offset) / R.DT - cur) < 1e-9);
  });
}

R.setTickRate();
console.log(`tickrate OK: ${nOut} outcomes, ${checked} scene instants, ` +
            `${jointsChecked} joint values (${clampSkipped} at a clamp skipped), ` +
            `60 Hz == 30 Hz upsampled`);
