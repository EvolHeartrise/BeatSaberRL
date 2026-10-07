// The timeline's torque profile (lib/torquegraph.js): the per-tick series is
// the replay's own "torque" total, null without one; the per-pixel reduction
// covers every tick exactly once, keeps every spike (column max) and the
// level (column mean, weight-preserving); the full scale is the p99; the
// mistake marks are the ticks with a miss / bad cut / bomb.
// Run: node test/torquegraph.test.js
import assert from "node:assert/strict";
import { parseReplay } from "../src/lib/replay.js";
import {
  columnRange, downsample, fracToTick, mistakeTicks, robustMax, tickToFrac, torqueSeries,
} from "../src/lib/torquegraph.js";
import { FIXTURE, readReplayFile } from "./fixture.js";

// ---- the series: the replay's torque total, tick for tick
const rep = parseReplay(readReplayFile(FIXTURE));
const s = torqueSeries(rep);
assert.ok(s instanceof Float32Array && s.length === rep.ticks.length, "one value per tick");
rep.ticks.forEach((t, k) => assert.ok(Math.abs(s[k] - t.torque[0]) < 1e-3, `tick ${k}`));
assert.ok(s.some((v) => v > 0), "the fixture exerts torque");
// no torque data -> no graph (a Flux run), and partial / all-zero data too
const mk = (vals) => ({ ticks: vals.map((v) => (v === undefined ? {} : { torque: [v, 0, 0, 0] })) });
assert.equal(torqueSeries(mk([undefined, undefined])), null);
assert.equal(torqueSeries(mk([1, undefined, 2])), null);
assert.equal(torqueSeries(mk([0, 0, 0])), null);
assert.equal(torqueSeries(null), null);
assert.equal(torqueSeries({ ticks: [] }), null);

// ---- columns: consecutive, disjoint, covering
for (const [n, cols] of [[596, 800], [2625, 800], [2625, 1600], [7, 3], [1, 5], [800, 800]]) {
  let next = 0;
  for (let i = 0; i < cols; i++) {
    const [lo, hi] = columnRange(i, cols, n);
    assert.ok(lo >= 0 && hi <= n && hi > lo, `column ${i}/${cols} of ${n}: [${lo}, ${hi})`);
    if (n >= cols) {
      assert.equal(lo, next, `column ${i}/${cols} of ${n} starts where the last ended`);
      next = hi;
    } else {
      assert.equal(hi - lo, 1, "fewer ticks than columns: one tick per column");
      assert.ok(lo >= next && lo - next <= 1, "the tick under each column, left to right");
      next = lo;
    }
  }
  if (n >= cols) assert.equal(next, n, `all ${n} ticks covered by ${cols} columns`);
}

// ---- downsampling keeps spikes (max) and the level (mean)
for (const cols of [37, 300, 596, 800, 1601]) {
  const { max, mean } = downsample(s, cols);
  assert.equal(max.length, cols);
  assert.ok(Math.abs(Math.max(...max) - Math.max(...s)) < 1e-3, `global max kept at ${cols}`);
  if (cols <= s.length) {
    let sum = 0;
    for (let i = 0; i < cols; i++) {
      const [lo, hi] = columnRange(i, cols, s.length);
      sum += mean[i] * (hi - lo);
      assert.ok(max[i] >= mean[i] - 1e-3, "max >= mean");
    }
    const total = s.reduce((a, v) => a + v, 0);
    assert.ok(Math.abs(sum - total) / total < 1e-5, `mean preserves the sum at ${cols}`);
  }
}
// a one-tick spike survives 30:1 compression, in the column that holds it
const flat = new Float32Array(3000).fill(10);
flat[1777] = 900;
const ds = downsample(flat, 100);
const col = ds.max.findIndex((v) => v === 900);
const [lo, hi] = columnRange(col, 100, 3000);
assert.ok(col >= 0 && lo <= 1777 && 1777 < hi, "the spike is in its own column");
assert.equal(ds.max.filter((v) => v === 900).length, 1, "and only there");
assert.ok(Math.abs(ds.mean[col] - (10 * 29 + 900) / 30) < 1e-3, "its column's mean");

// ---- full scale: the p99, never below 1
const ramp = Float32Array.from({ length: 1000 }, (_, i) => i);
assert.equal(robustMax(ramp), 989);
assert.equal(robustMax(new Float32Array([0, 0.1, 0.2])), 1);
assert.ok(robustMax(s) <= Math.max(...s) && robustMax(s) > 0);

// ---- mistake marks
const evRep = { ticks: [
  { events: [{ cut: { good: true } }] }, { events: [{ miss: { note: 1 } }] }, {},
  { events: [{ cut: { good: false } }] }, { events: [{ bomb: { note: 2 } }] },
  { events: [{ post: { note: 0 } }] },
] };
assert.deepEqual(mistakeTicks(evRep), [1, 3, 4]);

// ---- tick <-> track fraction
assert.equal(tickToFrac(0, 100), 0);
assert.equal(tickToFrac(100, 100), 1);
assert.equal(tickToFrac(5, 0), 0);
assert.equal(fracToTick(0.5, 595), 298);
assert.equal(fracToTick(-1, 595), 0);
assert.equal(fracToTick(2, 595), 595);
for (let k = 0; k <= 595; k++) assert.equal(fracToTick(tickToFrac(k, 595), 595), k);

console.log(`torquegraph OK: ${s.length} ticks, p99 ${robustMax(s).toFixed(1)} N·m, ` +
  `max ${Math.max(...s).toFixed(1)} N·m`);
