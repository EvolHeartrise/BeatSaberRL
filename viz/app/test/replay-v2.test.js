// Replay format 2 decoder (src/lib/replay.js decodeReplay) against the
// Python decoder (python/beat_arena_env/replay_format.py): the committed
// scripted fixture (sabers / head / activation / torque DERIVED from q and
// action) and its stored-fields twin (all four STORED as blocks, the
// writer's fallback) must decode to what replay_v2_ref.json says
// (python tools/make_replay_fixture.py). Run: node test/replay-v2.test.js
import { readFileSync } from "node:fs";
import assert from "node:assert/strict";
import {
  REPLAY_MAGIC, decodeReplay, effortOf, isReplayBytes, parseReplay,
} from "../src/lib/replay.js";
import { FIXTURE, readReplayFile } from "./fixture.js";

const ref = JSON.parse(readFileSync(
  new URL("./fixtures/replay_v2_ref.json", import.meta.url), "utf8"));
const STORED = new URL("./fixtures/v10_scripted_ep00_seed0_stored.rpl.gz",
  import.meta.url);

// numbers to 1e-9 (FK in numpy vs JS differs in the last bits; every
// stored value is int / scale on both sides, so those agree exactly)
function close(a, b, path) {
  if (Array.isArray(b)) {
    assert.ok(Array.isArray(a) && a.length === b.length, `${path}: length`);
    b.forEach((v, i) => close(a[i], v, `${path}[${i}]`));
  } else if (b !== null && typeof b === "object") {
    assert.deepEqual(Object.keys(a).sort(), Object.keys(b).sort(), `${path}: keys`);
    for (const k of Object.keys(b)) close(a[k], b[k], `${path}.${k}`);
  } else if (typeof b === "number") {
    assert.ok(Math.abs(a - b) <= 1e-9 * Math.max(1, Math.abs(b)),
      `${path}: ${a} vs ${b}`);
  } else {
    assert.equal(a, b, path);
  }
}

for (const [name, file] of [["main", FIXTURE], ["stored", STORED]]) {
  const bytes = readReplayFile(file);
  assert.ok(isReplayBytes(bytes));
  const { header, ticks } = decodeReplay(bytes);
  const r = ref[name];
  assert.equal(header.format, 2);
  assert.equal(ticks.length, r.T);
  for (const [k, want] of Object.entries(r.sample)) close(ticks[+k], want, `${name} tick ${k}`);
  // every tick: the exact fields
  ticks.forEach((t, k) => {
    assert.equal(t.t, k);
    assert.deepEqual(t.events, r.events[k], `${name} events at ${k}`);
    assert.deepEqual(t.contacts, r.contacts[k], `${name} contacts at ${k}`);
    assert.deepEqual(t.action, r.actions[k], `${name} action at ${k}`);
  });
  console.log(`${name}: ${ticks.length} ticks match the python decoder ` +
    `(${Object.keys(r.sample).length} sampled in full)`);
}

// stored blocks are read back, and they equal the derived values to their
// quantization (sabers 1e-4, head 1e-6, activation 1e-4, torque 1e-3)
{
  const a = decodeReplay(readReplayFile(FIXTURE)).ticks;
  const b = decodeReplay(readReplayFile(STORED)).ticks;
  let worst = { sabers: 0, head: 0, activation: 0, torque: 0 };
  b.forEach((t, k) => {
    for (const f of Object.keys(worst)) {
      const x = a[k][f].flat(), y = t[f].flat();
      x.forEach((v, i) => { worst[f] = Math.max(worst[f], Math.abs(v - y[i])); });
    }
  });
  assert.ok(worst.sabers <= 5e-5 + 1e-12 && worst.head <= 5e-7 + 1e-12 &&
    worst.activation <= 5e-5 + 1e-12 && worst.torque <= 5e-4 + 1e-12,
    JSON.stringify(worst));
  console.log("stored == derived within quantization:", JSON.stringify(worst));
}

// derived torque follows the header's body, not the nominal table
{
  const body = { tau_max: Array(29).fill(2), gear_frac: Array(29).fill([0, 1, -1, 0.5, -0.5]) };
  const [act, trq] = effortOf([1, 2, ...Array(27).fill(0)].map((v, j) => (j === 20 ? 3 : v)), body);
  assert.deepEqual(act, [2.5, 2, 0, 0.5]);
  assert.deepEqual(trq, [5, 4, 0, 1]);
}

// malformed input is reported, not half-decoded
{
  const bytes = readReplayFile(FIXTURE);
  assert.throws(() => decodeReplay(bytes.slice(0, bytes.length - 100)), /truncated/);
  const bad = bytes.slice();
  bad[0] = 0x58;
  assert.ok(!isReplayBytes(bad));
  assert.throws(() => decodeReplay(bad), /not a beat-arena replay/);
  assert.equal(REPLAY_MAGIC.length, 8);
  // a rep built from it is the viewer's replay object
  const rep = parseReplay(bytes);
  assert.ok(rep.body && rep.body.tau_max.length === 29);
}
console.log("replay format 2 decoder OK");
