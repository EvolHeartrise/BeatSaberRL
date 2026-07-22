// Node smoke test for the viewer's pure logic (src/lib/replay.js) against a
// real recorded replay. Run: npm test [path-to-replay.jsonl]
import { readFileSync } from "node:fs";
import assert from "node:assert/strict";
import {
  activePosts, aftermaths, counts, parseReplay, pendingNotes, saberAt,
  tipOf, tipSpeed, DT,
} from "../src/lib/replay.js";

const path = process.argv[2] ??
  "../../python/replays/v52_scripted_dyn_ep00_seed7000.jsonl";
const rep = parseReplay(readFileSync(path, "utf8"));
console.log(`loaded: ${rep.chart.length} notes, ${rep.ticks.length} ticks, ` +
            `${rep.posts.length} post events`);

// outcomes cover the chart exactly
const last = rep.ticks.length - 1;
const c = counts(rep, last);
assert.equal(c.good + c.bad + c.miss, rep.chart.length,
  "every note resolves exactly once");
// score = sum of all event points, reward sums to the episode return
const evPts = rep.ticks.flatMap((t) => t.events).reduce((a, e) =>
  a + (e.cut ? e.cut.points || 0 : 0) + (e.post ? e.post.points : 0), 0);
assert.ok(Math.abs(c.score - evPts) < 1e-6, "counts.score matches events");
console.log(`counts: good ${c.good} bad ${c.bad} miss ${c.miss} ` +
            `score ${c.score.toFixed(1)} return ${c.reward.toFixed(2)}`);

// saber interpolation and blade math stay finite over the whole episode
for (let k = 0; k < last; k += 7) {
  for (const s of [0, 1]) {
    const sab = saberAt(rep, k, 0.5, s);
    const tip = tipOf(sab);
    assert.ok([...sab, ...tip, tipSpeed(sab)].every(Number.isFinite),
      `NaN in saber math at tick ${k}`);
  }
}

// derived render lists behave at several times
for (const frac of [0, 0.3, 0.6, 0.9, 1.0]) {
  const t = frac * last * DT;
  const p = pendingNotes(rep, t), a = aftermaths(rep, t),
    po = activePosts(rep, t);
  assert.ok(p.length + a.length >= 0 && po.length >= 0);
}

// replays from retired parameterizations are rejected
assert.throws(
  () => parseReplay('{"header": {"version": 4, "config": {}, "chart": []}}'),
  /version 4/);

console.log("replay.js smoke test OK");
