// Node test for the 3D HUD's numbers (src/lib/hud.js): the multiplier /
// combo / score / accuracy / energy walk, and the ScoreSaber-watch text
// formats. Run: npm test
import { readFileSync } from "node:fs";
import assert from "node:assert/strict";
import { parseReplay } from "../src/lib/replay.js";
import {
  ENERGY_FLOOR, ENERGY_START, formatScore, gradeOf, hudAt, pctString,
} from "../src/lib/hud.js";

// A minimal replay in parseReplay's shape: ticks of events, chart kinds,
// posts indexed by note.
function rep(chart, ticks, tickHz = 30) {
  const postsByNote = [];
  ticks.forEach((evs, t) => evs.forEach((e) => {
    if (e.post) (postsByNote[e.post.note] ??= []).push({ tick: t, ...e.post });
  }));
  return {
    chart: chart.map((kind) => ({ kind })),
    ticks: ticks.map((events, t) => ({ t, events })),
    postsByNote, tickHz,
  };
}
const cut = (note, points = 85, good = true) => ({ cut: { note, good, points } });
const post = (note, points = 30) => ({ post: { note, points } });
const close = (a, b, msg) => assert.ok(Math.abs(a - b) < 1e-9, `${msg}: ${a} vs ${b}`);

// ---- multiplier ladder: 1x, then 2x on the 2nd hit, 4x after 4 more, 8x
// after 8 more (the promoted multiplier applies to the hit that fills the
// bar), and ScoreSaber's full-combo max 115 * (8n - 63) for n >= 14 --------
{
  const N = 20;
  const ticks = [];
  for (let i = 0; i < N; i++) ticks.push([cut(i)], [post(i)]);
  const r = rep(Array(N).fill(0), ticks);
  const multAfter = (hits) => hudAt(r, 2 * hits - 2).mult; // tick of hit #hits
  assert.equal(multAfter(1), 1);
  assert.equal(multAfter(2), 2, "2x on the 2nd hit");
  assert.equal(multAfter(5), 2);
  assert.equal(multAfter(6), 4, "4x after 2 + 4 hits");
  assert.equal(multAfter(13), 4);
  assert.equal(multAfter(14), 8, "8x after 2 + 4 + 8 hits");
  // ring fill = progress / needed
  assert.equal(hudAt(r, 2 * 3 - 2).fill, 1 / 4, "1 of 4 toward 4x");
  assert.equal(hudAt(r, 2 * 14 - 2).fill, 0, "empty ring at 8x");
  const end = hudAt(r, ticks.length - 1);
  assert.equal(end.score, 115 * (8 * N - 63), "all-115 run = the SS max formula");
  assert.equal(end.max, 115 * (8 * N - 63));
  assert.equal(end.pct, 100);
  assert.equal(end.combo, N);
  assert.equal(end.maxCombo, N);
  assert.equal(end.misses, 0);
  assert.ok(end.fullCombo);
  // the score (and its max share) lands when the follow-through does: at
  // the cut tick the note is not counted yet, so accuracy never dips
  const atCut = hudAt(r, 2);
  assert.equal(atCut.combo, 2, "combo counts at the cut");
  assert.equal(atCut.score, 115, "note 1 finished, note 2 still pending");
  assert.equal(atCut.max, 115);
  assert.equal(hudAt(r, 3).score, 115 + 2 * 115, "note 2 lands at 2x");
  for (let k = 0; k < ticks.length; k++) assert.equal(hudAt(r, k).pct, 100);
}

// ---- breaks: bad cut / miss / bomb / wall entry drop one tier, empty the
// bar and zero the combo; a tick inside a wall does not break -------------
{
  const ticks = [];
  for (let i = 0; i < 14; i++) ticks.push([cut(i, 85)], [post(i, 30)]);
  const at8 = ticks.length - 1;
  ticks.push([cut(14, 0, false)]);            // bad cut: 8x -> 4x
  const afterBad = ticks.length - 1;
  ticks.push([cut(15, 85)], [post(15, 30)]); // 1 of 4 toward 8x
  ticks.push([{ miss: { note: 16 } }]);        // 4x -> 2x
  const afterMiss = ticks.length - 1;
  ticks.push([{ bomb: { note: 17 } }]);        // 2x -> 1x
  ticks.push([{ bomb: { note: 18 } }]);        // stays 1x
  const afterBombs = ticks.length - 1;
  ticks.push([cut(19, 85)], [post(19, 30)]);
  ticks.push([{ wall_enter: {} }, { wall: {} }], [{ wall: {} }], [{ wall: {} }]);
  const afterWall = ticks.length - 1;
  const chart = [...Array(17).fill(0), 1, 1, 0];
  const r = rep(chart, ticks);
  assert.equal(hudAt(r, at8).mult, 8);
  let s = hudAt(r, afterBad);
  assert.deepEqual([s.mult, s.combo, s.progress, s.misses], [4, 0, 0, 1]);
  assert.ok(!s.fullCombo, "FC bars go at the first break");
  s = hudAt(r, afterBad + 2);
  assert.deepEqual([s.mult, s.combo, s.progress], [4, 1, 1]);
  s = hudAt(r, afterMiss);
  assert.deepEqual([s.mult, s.combo, s.misses], [2, 0, 2]);
  s = hudAt(r, afterBombs);
  assert.deepEqual([s.mult, s.combo, s.misses], [1, 0, 4]);
  s = hudAt(r, afterWall);
  assert.deepEqual([s.combo, s.misses], [0, 5], "wall entry breaks once");
  assert.equal(s.maxCombo, 14);
  // score: 14 notes at the ladder, #15 at 4x, #16 at 1x (after the bombs)
  const ladder = [1, 2, 2, 2, 2, 4, 4, 4, 4, 4, 4, 4, 4, 8];
  const want = 115 * (ladder.reduce((a, b) => a + b) + 4 + 1);
  assert.equal(s.score, want);
  // max: the unbroken walk over the 17 resolved notes (bombs excluded)
  const maxLadder = [...ladder, 8, 8, 8, 8];
  assert.equal(s.max, 115 * maxLadder.reduce((a, b) => a + b));
  close(s.pct, (100 * want) / s.max, "accuracy = score / max so far");
  // energy: 0.5 + 14 heals - bad 0.1 + heal - miss 0.15 - 2 bombs 0.3 +
  // heal, then 3 ticks inside the wall at 1.3 / s run it onto the floor
  close(hudAt(r, afterWall - 3).energy,
    0.5 + 0.14 - 0.1 + 0.01 - 0.15 - 0.3 + 0.01, "energy");
  close(hudAt(r, afterWall - 1).energy,
    0.5 + 0.14 - 0.1 + 0.01 - 0.15 - 0.3 + 0.01 - 2 * 1.3 / 30, "wall drain");
  assert.equal(s.energy, ENERGY_FLOOR, "floored, not failed");
}

// ---- chain links: 20 points, the link heal/damage constants; the floor --
{
  const r = rep([3, 3, 3], [[cut(0, 20)], [cut(1, 0, false)], [{ miss: { note: 2 } }]]);
  close(hudAt(r, 0).energy, ENERGY_START + 0.002, "link heal");
  close(hudAt(r, 1).energy, ENERGY_START + 0.002 - 0.025, "link bad cut");
  close(hudAt(r, 2).energy, ENERGY_START + 0.002 - 0.025 - 0.03, "link miss");
  assert.equal(hudAt(r, 0).score, 20, "a link with no post finishes at the cut");
  const w = rep([0], [Array(200).fill({ wall: {} })]);
  assert.equal(hudAt(w, 0).energy, ENERGY_FLOOR, "no fail: floored");
  // 60 Hz: the drain is per second, not per tick
  const w60 = rep([0], [[{ wall: {} }]], 60);
  close(hudAt(w60, 0).energy, ENERGY_START - 1.3 / 60, "60 Hz wall drain");
}

// ---- before the first tick / empty replay -------------------------------
{
  const s = hudAt(rep([0], [[]]), -1);
  assert.deepEqual([s.score, s.pct, s.grade, s.mult, s.combo, s.energy],
    [0, 100, "SS", 1, 0, ENERGY_START]);
}

// ---- formats (ScoreManager.FormatScore / GetPercentageString /
// GradeFromPercentage) ----------------------------------------------------
assert.equal(formatScore(0), "0");
assert.equal(formatScore(999), "999");
assert.equal(formatScore(1000), "1 000");
assert.equal(formatScore(530686), "530 686");
assert.equal(formatScore(2496090), "2 496 090");
assert.equal(pctString(100), "100.00%");
assert.equal(pctString(82.625), "82.63%");
assert.equal(pctString(3.3), "3.30%");
for (const [p, g] of [[100, "SS"], [90, "SS"], [89.99, "S"], [80, "S"], [79.9, "A"],
  [65, "A"], [64.9, "B"], [50, "B"], [49, "C"], [35, "C"], [34, "D"], [20, "D"],
  [19.99, "E"], [0, "E"]]) assert.equal(gradeOf(p), g, `grade at ${p}%`);

// ---- the recorded fixture, pinned against the python walk ---------------
// python/beat_arena_env/scoresaber.py ComboTracker over the same events
// (publish_replays.walk_events): score 881, max combo 2, final combo 1 at
// 1x, 29 breaks (17 bad cuts + 9 misses + 3 bombs).
{
  const r = parseReplay(readFileSync(
    new URL("./fixtures/v10_scripted_ep00_seed0.jsonl", import.meta.url), "utf8"));
  const s = hudAt(r, r.ticks.length - 1);
  assert.deepEqual([s.score, s.maxCombo, s.combo, s.mult, s.misses],
    [881, 2, 1, 1, 29]);
  for (let k = 0; k < r.ticks.length; k++) {
    const h = hudAt(r, k);
    assert.ok(h.pct <= 100 + 1e-9 && h.pct >= 0, `accuracy in range at ${k}`);
    assert.ok(h.energy >= ENERGY_FLOOR && h.energy <= 1, `energy in range at ${k}`);
  }
}

console.log("hud.js test OK");
