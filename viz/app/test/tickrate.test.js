// exp/tick-rate (replay record_rev 6): the viewer reads the replay header's
// tick_hz (absent = 30) and every tick <-> seconds conversion follows it.
//
// The 60 Hz replay here is the committed 30 Hz fixture UPSAMPLED: 60 Hz
// tick 2k is 30 Hz tick k, tick 2k+1 the midpoint, and every event of 30
// Hz tick k moves to 60 Hz tick 2k+1 — whose end, (2k+2)/60 s, is the 30
// Hz tick's end (k+1)/30 s. So at any real time the two replays must show
// the same scene: same pose, same flying notes, same outcomes / popups /
// contacts, same track offset. Run: node test/tickrate.test.js
import { readFileSync } from "node:fs";
import assert from "node:assert/strict";
import * as R from "../src/lib/replay.js";
import { FOLLOW_SNAP, followClock } from "../src/lib/sync.js";

const text = readFileSync(
  new URL("./fixtures/v10_scripted_ep00_seed0.jsonl", import.meta.url), "utf8");
const lines = text.split("\n").filter((l) => l.trim());

// ---- header parsing ----------------------------------------------------------
const rep30 = R.parseReplay(text);
assert.equal(rep30.tickHz, 30, "absent tick_hz = 30");
assert.equal(R.tickHzOf(rep30), 30);
assert.equal(R.tickHzOf(null), 30, "no replay = 30");
assert.equal(R.tickHzOf({ ticks: [] }), 30, "human/synthetic reps = 30");
{
  const h = JSON.parse(lines[0]);
  h.header.tick_hz = "bogus";
  assert.throws(() => R.parseReplay([JSON.stringify(h), ...lines.slice(1)].join("\n")));
}

// ---- the 60 Hz upsample ------------------------------------------------------
const mid = (a, b) => a.map((v, i) => Array.isArray(v) ? mid(v, b[i]) : (v + b[i]) / 2);
function upsample() {
  const head = JSON.parse(lines[0]);
  head.header.tick_hz = 60;
  head.header.action_repeat = 1;
  const t30 = lines.slice(1).map((l) => JSON.parse(l));
  const out = [];
  t30.forEach((t, k) => {
    const nxt = t30[Math.min(k + 1, t30.length - 1)];
    out.push({ ...t, t: 2 * k, time: (2 * k) / 60, events: [], contacts: [] });
    // the odd tick carries the 30 Hz tick's end-of-tick events (its END is
    // (2k+2)/60 = (k+1)/30) and, as pose, the linear midpoint of 30 Hz
    // ticks k and k+1 — exactly what the 30 Hz viewer interpolates at
    // u = 0.5, so the two replays interpolate to the same poses.
    out.push({
      ...t, t: 2 * k + 1, time: (2 * k + 1) / 60,
      q: mid(t.q, nxt.q), sabers: mid(t.sabers, nxt.sabers),
      feet: t.feet, events: t.events, contacts: t.contacts,
    });
  });
  return [JSON.stringify(head), ...out.map((t) => JSON.stringify(t))].join("\n");
}
const rep60 = R.parseReplay(upsample());
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
let checked = 0;
for (let k = 0; k < last; k++) {
  for (const frac of [0, 0.25, 0.5, 0.9]) {
    const T = (k + frac) / 30;
    // the replay clock: pb.cur = T / DT, split into tick + fraction
    const pose = (rep) => at(rep, () => {
      const f = T / R.DT, i = Math.floor(f);
      return R.qAt(rep, i, f - i);
    });
    const q30 = pose(rep30), q60 = pose(rep60);
    // exact on the 30 Hz grid and at the midpoint; elsewhere both are
    // linear between the same two 30 Hz samples
    for (let j = 0; j < 7; j++)
      assert.ok(Math.abs(q30[j] - q60[j]) < 1e-9, `root q at T=${T}`);
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
            `60 Hz == 30 Hz upsampled`);
