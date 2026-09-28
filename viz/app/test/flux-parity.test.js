// Env <-> WebXR game parity (SPEC.md §Flux WebXR game): the game's judge
// (src/lib/fluxcore.js over rust/flux-wasm) replays episodes the REAL env
// played (python/tools/make_flux_game_fixture.py records every Flux tick's
// full input — both ticks' palms, head, shoulder z — at full precision)
// and must emit the env's event stream exactly, tick for tick, end on the
// same tick, and land on the same tallies.
//
// Run: node test/flux-parity.test.js [fixture.json]
// (python/tests/test_flux_game_parity.py runs it on a freshly recorded
// fixture from the live env.)
import { readFileSync } from "node:fs";
import assert from "node:assert/strict";
import { loadFluxCore } from "../src/lib/fluxcore.js";
import { BAR_W, CIRCLE_R, DT, PALM_OFF, PLANE_Z0 } from "../src/lib/replay.js";

const path = process.argv[2] ?? "test/fixtures/flux_game_parity.json";
const fx = JSON.parse(readFileSync(path, "utf8"));
const core = await loadFluxCore();

// the renderer's copies of the judge geometry are the judge's
assert.equal(core.consts.DT, DT);
assert.equal(core.consts.PALM_OFF, PALM_OFF);
assert.equal(core.consts.CIRCLE_R, CIRCLE_R);
assert.ok(Math.abs(core.consts.BAR_W - BAR_W) < 1e-15);
assert.equal(core.consts.PLANE_Z0, PLANE_Z0);

let nTicks = 0, nEvents = 0;
const seen = new Set();
for (const ep of fx.episodes) {
  const g = core.newGame(ep.chart, ep.max_steps);
  ep.ticks.forEach((tk, k) => {
    const { done, events } = g.stepFlat(tk.in);
    const got = events.map((e) => [e.i, e.code, e.pts, e.pre, e.acc]);
    assert.deepEqual(got, tk.events, `${ep.name}: tick ${k} events`);
    assert.equal(done, tk.done, `${ep.name}: tick ${k} done`);
    events.forEach((e) => seen.add(e.code));
    nEvents += events.length;
  });
  nTicks += ep.ticks.length;
  const s = g.stats();
  const f = ep.final;
  assert.equal(s.good, f.good, `${ep.name}: good`);
  assert.equal(s.bad, f.bad, `${ep.name}: bad`);
  assert.equal(s.misses, f.misses, `${ep.name}: misses`);
  assert.equal(s.raw_score, f.score, `${ep.name}: score`);
  assert.equal(s.score_base, f.score_base, `${ep.name}: base`);
  assert.equal(s.score_acc, f.score_acc, `${ep.name}: acc`);
  // every object resolved, and resolved the way the events say
  const st = g.status();
  assert.equal(st.filter((v) => v === 0).length, 0, `${ep.name}: all resolved`);
  g.free();
}
// the fixture must exercise every Flux outcome
for (const code of [0, 2, 3, 4]) assert.ok(seen.has(code), `event code ${code}`);
console.log(`flux-parity: ${fx.episodes.length} episodes, ${nTicks} ticks, ` +
  `${nEvents} events — wasm judge == env`);
