// Node tests for the Flux WebXR game's session layer (src/lib/fluxgame.js):
// the display-rate pose track -> 30 Hz env ticks plumbing, the live
// recording (a replay the viewer reads), the chart pack, and a scripted
// "player" clearing a real pack chart through the same code path the
// headset uses. The judge itself is pinned to the env by
// flux-parity.test.js.
// Run: node test/flux-game.test.js
import assert from "node:assert/strict";
import PACK from "../src/lib/fluxcharts.data.js";
import { loadFluxCore } from "../src/lib/fluxcore.js";
import { FluxSession, LEAD_IN_S, quatSceneToEnv, rank, sceneToEnv }
  from "../src/lib/fluxgame.js";
import {
  DT, KIND_BAR, KIND_CIRCLE, PLANE_Z0, barAxis, counts, humanHeadAt,
  humanPalmsAt, isFlux, isHuman, parseReplay, pendingNotes, sweetPoint,
  trackOffset,
} from "../src/lib/replay.js";

const core = await loadFluxCore();
const HEAD = [0, 1.7, PLANE_Z0];
const REST = [[-0.25, 1.1, -0.4], [0.25, 1.1, -0.4]];

// ---- the chart pack: training generator output, valid for the env judge ----
assert.ok(PACK.length >= 6);
assert.equal(new Set(PACK.map((c) => c.id)).size, PACK.length, "unique ids");
for (const c of PACK) {
  assert.ok(c.chart.length > 10, c.id);
  assert.ok(c.chart.every((o, i, a) => i === 0 || a[i - 1].t <= o.t), `${c.id} sorted`);
  assert.ok(c.chart.every((o) => o.kind === KIND_CIRCLE || o.kind === KIND_BAR));
  core.newGame(c.chart).free(); // the env's install checks accept it
}
// the "training" entries are the generator's DEFAULT config
const train = PACK.find((c) => c.id === "train-1");
assert.equal(train.config.note_speed_min, 10);
assert.equal(train.config.note_speed_max, 18);
assert.equal(train.config.circle_prob, 0.3);
assert.equal(train.config.bar_prob, 0.15);
assert.equal(train.config.chart_len_s, 20);

// ---- the env refuses what the game refuses (the same Rust) ----------------
assert.throws(() => core.newGame([{ t: 1, x: 0, y: 1.4, kind: 0, speed: 10 }]),
  /flux chart kinds are 4 \(circle\) \/ 5 \(bar\)/);
assert.throws(() => core.newGame([
  { t: 2, x: 0, y: 1.4, kind: 4, color: 0, speed: 10 },
  { t: 1, x: 0, y: 1.4, kind: 4, color: 1, speed: 10 }]), /sorted by hit_time/);

// ---- frames ---------------------------------------------------------------
assert.deepEqual(sceneToEnv([1, 2, 3]), [1, 2, -3]);
// yaw about y flips sign under z-mirroring, rotation about z does not
assert.deepEqual(quatSceneToEnv([0, 0.5, 0, 0.8]), [-0, -0.5, 0, 0.8]);
assert.deepEqual(quatSceneToEnv([0, 0, 0.5, 0.8]), [-0, -0, 0.5, 0.8]);

// ---- tick plumbing: samples ON tick times == stepping the judge directly ----
// at the env's own rate (hz 30: one judge tick per env tick, the env's clock)
{
  const chart = train.chart;
  const s = new FluxSession(core, train, { hz: 30 });
  const g = core.newGame(chart);
  // a wandering palm path, sampled exactly at each tick's end time, plus
  // an off-tick sample in between (which must not change anything)
  const path = (t) => [
    [-0.3 + 0.25 * Math.sin(2.1 * t), 1.25 + 0.3 * Math.sin(1.3 * t), -0.35],
    [0.3 + 0.25 * Math.sin(1.7 * t + 1), 1.3 + 0.3 * Math.cos(1.1 * t), -0.35]];
  s.feed(0, path(0), HEAD);
  let prev = path(0);
  for (let k = 0; k < 120; k++) {
    const T = (k + 1) * DT;
    s.feed(T - 0.4 * DT, path(T - 0.4 * DT), HEAD);
    s.feed(T, path(T), HEAD);
    const want = g.step(prev, path(T), HEAD, HEAD[2]);
    prev = path(T);
    const tk = s.rep.ticks[k];
    assert.equal(tk.t, k);
    assert.deepEqual(tk.events.length, want.events.length, `tick ${k}`);
  }
  assert.deepEqual(s.judge.status(), g.status(), "session == direct judge");
  assert.deepEqual(s.stats, g.stats());
  s.free(); g.free();
}

// ---- interpolation: 90 Hz samples of a linear path land on tick times ------
{
  const s = new FluxSession(core, train, { hz: 30 });
  const lin = (t) => [[-0.4 + 0.5 * t, 1.2, -0.3], [0.4, 1.2 + 0.2 * t, -0.3]];
  for (let f = -LEAD_IN_S * 90; f <= 90; f++) s.feed(f / 90, lin(f / 90), HEAD);
  assert.equal(s.rep.ticks.length, 30, "1 s of 30 Hz ticks");
  s.rep.ticks.forEach((tk, k) => {
    const want = lin((k + 1) * DT);
    for (let h = 0; h < 2; h++)
      for (let i = 0; i < 3; i++)
        assert.ok(Math.abs(tk.palms[h][i] - want[h][i]) < 2e-5, `tick ${k}`);
  });
  s.free();
}

// ---- judging faster than the env's tick (the default 120 Hz) ---------------
{
  assert.equal(new FluxSession(core, train).hz, 120, "default judge rate");
  assert.throws(() => new FluxSession(core, train, { hz: 100 }), /multiple of 30/);
  // plumbing: a 120 Hz session == the judge stepped directly at 120 Hz
  const s = new FluxSession(core, train, { hz: 120 });
  const g = core.newGame(train.chart, 0, 120);
  const path = (t) => [
    [-0.3 + 0.25 * Math.sin(2.1 * t), 1.25 + 0.3 * Math.sin(1.3 * t), -0.35],
    [0.3 + 0.25 * Math.sin(1.7 * t + 1), 1.3 + 0.3 * Math.cos(1.1 * t), -0.35]];
  s.feed(0, path(0), HEAD);
  let prev = path(0);
  for (let j = 0; j < 4 * 300 && !s.done; j++) {
    const T = (j + 1) * (DT / 4);
    s.feed(T, path(T), HEAD);
    g.step(prev, path(T), HEAD, HEAD[2]);
    prev = path(T);
  }
  assert.deepEqual(s.judge.status(), g.status(), "120 Hz session == the judge stepped at 120 Hz");
  assert.deepEqual(s.stats, g.stats());
  // the recording stays in 30 Hz frames (the viewer's clock) ...
  assert.equal(s.rep.ticks.length, Math.ceil(s.stats.tick / 4));
  // ... and every outcome carries its exact judge time, inside its frame
  let n = 0;
  s.rep.outcomes.forEach((o) => {
    if (!o) return;
    n++;
    assert.ok(o.t > o.tick * DT - 1e-12 && o.t <= (o.tick + 1) * DT + 1e-12,
      `outcome at ${o.t} outside frame ${o.tick}`);
    // on the 120 Hz grid
    const q = o.t / (DT / 4);
    assert.ok(Math.abs(q - Math.round(q)) < 1e-9);
  });
  assert.ok(n > 10);
  // round trip: the viewer keeps the exact times
  const back = parseReplay(s.toJSONL());
  assert.equal(back.judgeHz, 120);
  back.outcomes.forEach((o, i) => {
    if (o) assert.equal(o.t, s.rep.outcomes[i].t);
  });
  s.free(); g.free();
}
// a hit is judged (and drawn) when it happens: a palm crossing a circle
// half-way through an env tick resolves it on that 120 Hz sub-tick (t =
// DT/2), where the env's clock only judges it at the tick's end (t = DT).
// The circle reaches the palms' depth (z = 0) at t = DT/2 too.
{
  const c = { t: 0.5 * DT, x: 0, y: 1.2, kind: 4, color: 0, rot: 0,
              speed: 12, len: 0, sweet: 0 };
  const at = (hz) => {
    const s = new FluxSession(core, { id: "x", chart: [c] }, { hz });
    const p = (u) => [[-0.4 + 0.8 * u, 1.2, 0.0], [0.6, 1.2, 0.0]];
    s.feed(0, p(0), HEAD);
    for (let f = 1; f <= 16; f++) s.feed(f * DT / 8, p(Math.min(1, f / 8)), HEAD);
    const o = s.rep.outcomes[0];
    s.free();
    return o;
  };
  const o30 = at(30), o120 = at(120);
  assert.equal(o30.kind, "good"); assert.equal(o120.kind, "good");
  assert.equal(o30.t, DT, "30 Hz: judged at the env tick's end");
  assert.equal(o120.t, 2 * (DT / 4), "120 Hz: judged on the sub-tick it happens");
}

// ---- a scripted player clears a pack chart via the headset code path ------
// The "player" moves each palm onto the next circle of its colour as that
// circle arrives, and holds the Flux square across each bar through its
// sweet spot — what a person does, as an env-world pose track.
// Palms sit at env z = PZ, 0.35 m past the hit plane (z_off = 0 with the
// head at PLANE_Z0), which an object reaches PZ / speed after its hit time;
// each palm holds its object until then.
const PZ = -0.35;
function playerFor(chart) {
  const reaches = (o) => o.t - PZ / o.speed;
  const next = (h, t) => chart.find((o) => reaches(o) >= t - 0.02 &&
    ((o.kind === KIND_CIRCLE && o.color === h) || o.kind === KIND_BAR));
  return (t) => {
    const palms = [0, 1].map((h) => {
      const o = next(h, t);
      if (!o) return REST[h];
      if (o.kind === KIND_CIRCLE) return [o.x, o.y, PZ];
      // bar: palms either side of the sweet spot, along the bar normal
      const [ux, uy] = barAxis(o.rot);
      const [sx, sy] = sweetPoint(o);
      const nx = uy, ny = -ux; // perpendicular to the axis
      const side = h === 0 ? -1 : 1;
      const d = side * 0.3 * (nx < 0 ? -1 : 1);
      return [sx + d * nx, sy + d * ny, PZ];
    });
    return { palms, head: HEAD, quat: [0, 0, 0, 1] };
  };
}
{
  const entry = PACK.find((c) => c.id === "warmup-1");
  const s = new FluxSession(core, entry, { onEvent: () => {} });
  const player = playerFor(entry.chart);
  let t = -LEAD_IN_S;
  while (!s.done && t < 120) {
    const p = player(t);
    s.feed(t, p.palms, p.head, p.quat);
    t += 1 / 72;
  }
  assert.ok(s.done, "the chart ends");
  const st = s.stats;
  assert.equal(st.good + st.bad + st.misses, entry.chart.length, "all resolved");
  assert.ok(st.good >= 0.6 * entry.chart.length,
    `scripted player clears most of the chart (${st.good}/${entry.chart.length})`);
  assert.ok(st.score > st.raw_score, "the combo multiplier applies");
  assert.ok(["SS", "S", "A", "B", "C", "D"].includes(rank(st)));

  // the live recording reads as a replay ...
  const rep = s.rep;
  assert.ok(isFlux(rep) && isHuman(rep));
  assert.equal(rep.outcomes.filter(Boolean).length, entry.chart.length);
  const c = counts(rep, rep.ticks.length - 1);
  assert.equal(c.good, st.good);
  assert.equal(c.miss, st.misses);
  assert.equal(c.score, st.raw_score);
  // ... and so does its export, through the viewer's own parser
  const back = parseReplay(s.toJSONL());
  assert.ok(isFlux(back) && isHuman(back));
  assert.equal(back.ticks.length, rep.ticks.length);
  assert.equal(back.result.score, st.score);
  const c2 = counts(back, back.ticks.length - 1);
  assert.deepEqual([c2.good, c2.bad, c2.miss, c2.score],
    [st.good, st.bad, st.misses, st.raw_score]);
  // human runs: palms / head / head-anchored track straight from the log
  const k = 10;
  assert.deepEqual(humanPalmsAt(back, k, 0), back.ticks[k].palms);
  assert.deepEqual(humanHeadAt(back, k, 0), back.ticks[k].head.slice(0, 3));
  assert.ok(Math.abs(trackOffset(back, k) - (back.ticks[k].head[2] - PLANE_Z0)) < 1e-12);
  assert.ok(pendingNotes(back, back.chart[0].t, 0).length > 0);
  s.free();
}

// ---- rebase: a resumed / regained pose does not sweep the gap -------------
// A left circle sits midway on the path of a left palm that "jumps" 0.8 m
// across a pause. A plain feed sweeps the palm through it (touch); after
// rebase the sweep starts at the new pose (the player never moved there).
{
  const circle = { t: 1.5 * DT, x: 0, y: 1.2, kind: 4, color: 0, rot: 0,
                   speed: 1, len: 0, sweet: 0 };
  // at 1 m/s the circle's thin slab (5 cm either side) is at the palm's
  // depth (z 0, within reach) through tick 1, when the jump happens
  const from = [[-0.4, 1.2, 0], REST[1]];
  const to = [[0.4, 1.2, 0], REST[1]];
  const run = (rebase) => {
    const s = new FluxSession(core, { id: "gap", chart: [circle] });
    s.feed(0, from, HEAD);
    s.feed(DT, from, HEAD);            // tick 0: palm resting left
    if (rebase) s.rebase(1.5 * DT, { palms: to, head: HEAD });
    s.feed(2 * DT, to, HEAD);          // tick 1: palm now right
    const good = s.stats.good;
    s.free();
    return good;
  };
  assert.equal(run(false), 1, "a real 0.8 m sweep through the circle touches it");
  assert.equal(run(true), 0, "a rebased jump sweeps nothing");
}

// ---- a finished session is inert --------------------------------------------
{
  const s = new FluxSession(core, { id: "one", chart: [
    { t: 0.2, x: 0.9, y: 1.9, kind: 4, color: 1, rot: 0, speed: 10, len: 0, sweet: 0 }] });
  for (let t = 0; t < 2; t += 1 / 60) s.feed(t, REST, HEAD);
  assert.ok(s.done);
  const n = s.rep.ticks.length;
  assert.equal(s.feed(3, REST, HEAD), 0);
  assert.equal(s.rep.ticks.length, n);
  assert.equal(s.stats.misses, 1);
  s.free();
}

console.log("flux-game: ok");
