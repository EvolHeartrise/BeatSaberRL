// Node test for the viewer's Flux-mode logic (SPEC.md §Flux) against the
// scripted Flux fixture (python/tools/make_flux_fixture.py).
// Run: node test/flux.test.js [path-to-flux-replay.jsonl]
import { readFileSync } from "node:fs";
import assert from "node:assert/strict";
import {
  CIRCLE_R, KIND_BAR, KIND_CIRCLE, KIND_MAX, PALM_OFF, barAxis, bodyFK,
  configSummary, counts, fkSaber, isFlux, isFluxKind, maxScore, palmOf,
  parseReplay, pendingNotes, sweetPoint,
} from "../src/lib/replay.js";

const path = process.argv[2] ?? "test/fixtures/flux_scripted.jsonl";
const rep = parseReplay(readFileSync(path, "utf8"));
assert.ok(isFlux(rep), "header game=flux");
assert.equal(rep.arcs.length, 0);
assert.ok(rep.chart.every(isFluxKind));
assert.ok(rep.chart.some((n) => n.kind === KIND_CIRCLE));
assert.ok(rep.chart.some((n) => n.kind === KIND_BAR));
assert.equal(KIND_MAX[KIND_CIRCLE], 60);
assert.equal(KIND_MAX[KIND_BAR], 115);
assert.equal(maxScore(rep.chart), rep.chart.reduce(
  (a, n) => a + (n.kind === KIND_CIRCLE ? 60 : 115), 0));
assert.match(configSummary(rep), /^FLUX: \d+ circles, \d+ bars/);

// palms: the logged "palms" field == wrist + PALM_OFF * hand axis, both
// from the logged sabers and from the viewer's own body FK
for (const t of rep.ticks.slice(0, 40)) {
  for (let h = 0; h < 2; h++) {
    const fromSab = palmOf(t.sabers[h]);
    const fromFk = palmOf(fkSaber(bodyFK(t.q), h));
    for (let k = 0; k < 3; k++) {
      assert.ok(Math.abs(fromSab[k] - t.palms[h][k]) < 2e-4, "palm (sabers)");
      assert.ok(Math.abs(fromFk[k] - t.palms[h][k]) < 2e-3, "palm (FK)");
    }
  }
}
assert.equal(PALM_OFF, 0.06);

// bar geometry: rot 0 points up, sweet +1 = the +u end
{ const [ux, uy] = barAxis(0); assert.ok(Math.abs(ux) < 1e-12 && uy === 1); }
{
  const b = { x: 0.1, y: 1.4, rot: Math.PI / 2, len: 0.6, sweet: 1 };
  const [sx, sy] = sweetPoint(b);
  assert.ok(Math.abs(sx - (0.1 - 0.3)) < 1e-12 && Math.abs(sy - 1.4) < 1e-12);
}

// outcomes: every good touch scores inside its kind's range; circles
// 50 + acc (acc 0..10, >= 4 given the hitbox radius), bars 100 + 0..15
let goods = 0;
rep.chart.forEach((n, i) => {
  const o = rep.outcomes[i];
  assert.ok(o, `object ${i} resolved`);
  if (o.kind !== "good") return;
  goods++;
  if (n.kind === KIND_CIRCLE) assert.ok(o.points >= 54 && o.points <= 60);
  else assert.ok(o.points >= 100 && o.points <= 115);
});
const end = counts(rep, rep.ticks.length - 1);
assert.equal(end.good, goods);
assert.equal(end.good + end.bad + end.miss, rep.chart.length);
assert.ok(end.bad > 0 && end.miss > 0, "fixture covers bad + miss");
assert.ok(CIRCLE_R > 0.17 && CIRCLE_R < 0.18);
// pending objects carry world z and vanish after resolving
const first = rep.chart[0];
assert.ok(pendingNotes(rep, 0).some((p) => p.n === first));
console.log(`flux ok: ${rep.chart.length} objects, ${goods} good, ` +
            `${end.bad} bad, ${end.miss} miss, score ${end.score}`);
