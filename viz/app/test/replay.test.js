// Node smoke test for the viewer's pure logic (src/lib/replay.js) against a
// real recorded replay. Run: npm test [path-to-replay.jsonl]
import { readFileSync } from "node:fs";
import assert from "node:assert/strict";
import {
  JOINT_NAMES, KIND_BOMB, activeArcs, activePosts, aftermaths, armFK,
  clampQ, counts, cutPlaneLocal, jointLimits, maxScore, mirrorQ,
  parseReplay, pendingNotes, poseSaber, saberAt, tipOf, tipSpeed, DT,
} from "../src/lib/replay.js";

const path = process.argv[2] ??
  "../../python/replays/v7fix_ep00_seed7000.jsonl";
const rep = parseReplay(readFileSync(path, "utf8"));
console.log(`loaded: ${rep.chart.length} objects, ${rep.ticks.length} ticks, ` +
            `${rep.posts.length} post events, ${rep.arcs.length} arc pairs`);

// v7 chart entries carry explicit world positions and kinds
rep.chart.forEach((n, i) => {
  assert.ok(Number.isFinite(n.x) && Number.isFinite(n.y),
    `chart entry ${i} lacks explicit x/y`);
  assert.ok(n.y > 0 && n.y < 2.5, `implausible object height at ${i}: ${n.y}`);
  assert.ok([0, 1, 2, 3].includes(n.kind), `unknown kind at ${i}`);
});

// every scoreable object resolves exactly once; bombs only resolve if hit
const last = rep.ticks.length - 1;
const c = counts(rep, last);
const scoreable = rep.chart.filter((n) => n.kind !== KIND_BOMB).length;
const nBombs = rep.chart.length - scoreable;
assert.equal(c.good + c.bad + c.miss, scoreable,
  "every scoreable object resolves exactly once");
assert.ok(c.bombs <= nBombs, "more bomb hits than bombs");
// score = sum of all event points, reward sums to the episode return
const evPts = rep.ticks.flatMap((t) => t.events).reduce((a, e) =>
  a + (e.cut ? e.cut.points || 0 : 0) + (e.post ? e.post.points : 0), 0);
assert.ok(Math.abs(c.score - evPts) < 1e-6, "counts.score matches events");
assert.ok(c.score <= maxScore(rep.chart), "score exceeds the per-kind max");
console.log(`counts: good ${c.good} bad ${c.bad} miss ${c.miss} ` +
            `bombs ${c.bombs}/${nBombs} score ${c.score.toFixed(1)}` +
            `/${maxScore(rep.chart)} return ${c.reward.toFixed(2)}`);

// arm interpolation and blade math stay finite over the whole episode,
// and the JS forward kinematics MIRROR matches the sim-logged wrist
for (let k = 0; k < last; k += 7) {
  for (const s of [0, 1]) {
    const sab = saberAt(rep, k, 0.5, s);
    const tip = tipOf(s, sab);
    assert.ok([...sab, ...tip, tipSpeed(rep, k, s)].every(Number.isFinite),
      `NaN in arm math at tick ${k}`);
    const raw = rep.ticks[k].sabers[s];
    const { wrist } = armFK(s, raw);
    for (let i = 0; i < 3; i++)
      assert.ok(Math.abs(wrist[i] - raw[12 + i]) < 2e-4,
        `JS FK wrist diverges from sim at tick ${k} hand ${s} axis ${i}`);
  }
}

// derived render lists behave at several times
for (const frac of [0, 0.3, 0.6, 0.9, 1.0]) {
  const t = frac * last * DT;
  const p = pendingNotes(rep, t), a = aftermaths(rep, t),
    po = activePosts(rep, t), ar = activeArcs(rep, t);
  assert.ok(p.length + a.length >= 0 && po.length >= 0 && ar.length >= 0);
  // objects past the miss boundary never linger in the pending list
  p.forEach(({ z }) => assert.ok(z >= -2.65 - 1e-9 && z <= 12 + 1e-9));
}

// arc pairs are sane: same color, tail after head, within the window
rep.arcs.forEach(({ head, tail }) => {
  const h = rep.chart[head], t = rep.chart[tail];
  assert.ok(h.fpost && t.fpre, "arc pair without fpost/fpre grants");
  assert.equal(h.color, t.color, "arc pair colors differ");
  assert.ok(t.t > h.t && t.t - h.t <= 2.5, "arc pair outside the window");
});

// actual-cut-plane reconstruction: every good cut yields a sane plane
// (unit normal, point inside the cube) or a declared-degenerate null
let planes = 0;
rep.chart.forEach((n, i) => {
  const o = rep.outcomes[i];
  if (!o || o.kind !== "good") return;
  const cp = cutPlaneLocal(rep, i);
  if (!cp) return; // pure stab: canonical fallback
  planes++;
  const nl = Math.hypot(...cp.n);
  assert.ok(Math.abs(nl - 1) < 1e-9, `non-unit cut normal at note ${i}`);
  assert.ok(cp.p.every((x) => Math.abs(x) <= 0.2 && Number.isFinite(x)),
    `cut point outside cube at note ${i}`);
  assert.ok([...cp.v, ...cp.b].every(Number.isFinite));
});
assert.ok(planes > 0, "no cut planes reconstructed at all");
console.log(`cut planes reconstructed for ${planes} good cuts`);

// ---- manual pose mode pure logic -------------------------------------------
assert.equal(JOINT_NAMES.length, 6);
for (const hand of [0, 1]) {
  const lims = jointLimits(hand);
  assert.equal(lims.length, 6);
  lims.forEach(([lo, hi]) => assert.ok(lo < hi));
  // clamping pins out-of-range joints to the anatomical bounds
  clampQ(hand, Array(6).fill(99)).forEach((v, j) =>
    assert.equal(v, lims[j][1], `clamp hi joint ${j} hand ${hand}`));
  clampQ(hand, Array(6).fill(-99)).forEach((v, j) =>
    assert.equal(v, lims[j][0], `clamp lo joint ${j} hand ${hand}`));
  // in-range poses pass through untouched
  const mid = lims.map(([lo, hi]) => (lo + hi) / 2);
  assert.deepEqual(clampQ(hand, mid), mid);
}
// the hands' ranges are exact mirrors, so a legal left pose mirrors to a
// legal right pose and its FK wrist mirrors across the body midplane
const qL = clampQ(0, [0.4, -0.6, 0.3, 0.9, 1.2, -0.5]);
const qR = mirrorQ(qL);
assert.deepEqual(clampQ(1, qR), qR, "mirrored pose escapes right limits");
const wL = armFK(0, qL).wrist, wR = armFK(1, qR).wrist;
assert.ok(Math.abs(wL[0] + wR[0]) < 1e-12 && Math.abs(wL[1] - wR[1]) < 1e-12
  && Math.abs(wL[2] - wR[2]) < 1e-12, "mirrored wrist not mirrored");
// poseSaber: 15-field arm state, omegas zero, wrist = FK
const ps = poseSaber(1, qR);
assert.equal(ps.length, 15);
assert.deepEqual(ps.slice(6, 12), [0, 0, 0, 0, 0, 0]);
assert.deepEqual(ps.slice(12), wR);
// neutral pose: arm straight forward (+z) from the shoulder
const neutral = armFK(0, [0, 0, 0, 0, 0, 0]);
assert.ok(Math.abs(neutral.wrist[0] + 0.2) < 1e-12
  && Math.abs(neutral.wrist[1] - 1.47) < 1e-12
  && Math.abs(neutral.wrist[2] - 0.0) < 1e-12, "neutral wrist off");
assert.deepEqual(neutral.blade, [0, 0, 1]);
console.log("pose-mode helpers OK");

// replays from retired parameterizations are rejected with the friendly
// "re-record" message
assert.throws(
  () => parseReplay('{"header": {"version": 6, "config": {}, "chart": []}}'),
  /version 6.*re-record/s);
assert.throws(
  () => parseReplay('{"header": {"version": 5, "config": {}, "chart": []}}'),
  /version 5/);

console.log("replay.js smoke test OK");
