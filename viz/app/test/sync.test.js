// Node test for the live A/V sync policy (src/lib/sync.js) — the rules that
// keep the song from being seeked while it is sounding, which is what used to
// chop it. Pure functions, no DOM. Run: npm test
import assert from "node:assert/strict";
import {
  FOLLOW_SNAP, FOLLOW_TAU, NO_NUDGE, SYNC, followClock, syncAction,
} from "../src/lib/sync.js";

const act = (sim, audio, now = 1000, active = NO_NUDGE) =>
  syncAction({ sim, audio, now, active });

// ---- thresholds are the ones the fix is specified with ----------------------
assert.equal(SYNC.tol, 0.08);   // ignore anything smaller
assert.equal(SYNC.hard, 0.5);   // only THIS may seek a playing element
assert.ok(SYNC.nudge > 0 && SYNC.nudge <= 0.03, "nudge is a couple of percent");

// ---- in tolerance: do nothing ----------------------------------------------
// The regression this whole file exists for: a few tens of ms of drift is
// normal every frame (currentTime is quantized to the audio callback) and
// must NOT produce a seek. The old viewer seeked at 80 ms and the seek itself
// froze the playback position for longer than that, so it never stopped.
for (const d of [0, 0.001, -0.02, 0.05, -0.079, 0.0799]) {
  const r = act(10 + d, 10);
  assert.equal(r.action, "none", `drift ${d} must be ignored`);
  assert.equal(r.rate, 1);
}

// ---- past tolerance: nudge the rate, never seek -----------------------------
{
  const behind = act(10.2, 10); // song is BEHIND the replay -> speed it up
  assert.equal(behind.action, "nudge");
  assert.ok(behind.rate > 1 && behind.rate <= 1.03, behind.rate);
  assert.ok(Math.abs(behind.drift - 0.2) < 1e-9);
  assert.equal(behind.until, 1000 + SYNC.hold);

  const ahead = act(10, 10.2); // song is AHEAD -> slow it down
  assert.equal(ahead.action, "nudge");
  assert.ok(ahead.rate < 1 && ahead.rate >= 0.97, ahead.rate);
}
// the whole band between tol and hard nudges; nothing in it seeks
for (const d of [0.081, 0.2, 0.49, -0.081, -0.3, -0.4999]) {
  assert.equal(act(10 + d, 10).action, "nudge", `drift ${d}`);
}

// ---- hopelessly apart: one seek --------------------------------------------
for (const d of [0.51, 3, -0.6, -30]) {
  const r = act(10 + d, 10);
  assert.equal(r.action, "seek", `drift ${d}`);
  assert.equal(r.rate, 1, "a seek lands on target: no rate left over");
  assert.equal(r.until, 0);
}

// ---- hysteresis: a nudge holds until it has actually closed the gap ---------
{
  const live = { rate: 1 + SYNC.nudge, until: 2000 }; // speeding the song up
  // back inside tolerance but not yet closed, window still open: keep going
  const held = act(10.05, 10, 1500, live);
  assert.equal(held.action, "nudge");
  assert.equal(held.rate, live.rate);
  assert.equal(held.until, live.until, "the window is not extended");
  // closed
  assert.equal(act(10.005, 10, 1500, live).action, "none");
  // window expired
  assert.equal(act(10.05, 10, 2500, live).action, "none");
  // drift flipped sign: a nudge the other way would be fighting itself
  assert.equal(act(9.95, 10, 1500, live).action, "none");
  // a still-large gap re-arms a FRESH window rather than holding the old one
  const re = act(10.3, 10, 1500, live);
  assert.equal(re.action, "nudge");
  assert.equal(re.until, 1500 + SYNC.hold);
}

// ---- garbage in (a media element with no timeline yet) ----------------------
assert.equal(act(NaN, 10).action, "none");
assert.equal(act(10, NaN).action, "none");

// ---- followClock: the replay clock is the slave -----------------------------
{
  const d = 1 / 60; // one 60 Hz frame
  // no song (or a silent one): free-run on the frame delta, 30 ticks/s
  assert.equal(followClock({ cur: 0, target: null, delta: d }), d * 30);
  assert.equal(followClock({ cur: 5, target: null, delta: d, speed: 2 }),
    5 + d * 60);
  assert.equal(followClock({ cur: 5, target: NaN, delta: d }), 5 + d * 30);

  // perfectly in sync: the ease adds nothing
  const free = 5 + d * 30;
  assert.ok(Math.abs(followClock({ cur: 5, target: free, delta: d }) - free)
    < 1e-12);

  // behind the song: move TOWARD it, by a fraction of the gap, never past it
  {
    const next = followClock({ cur: 5, target: free + 1, delta: d });
    assert.ok(next > free && next < free + 1, next);
    const frac = (next - free) / 1;
    assert.ok(Math.abs(frac - d / FOLLOW_TAU) < 1e-12, frac);
  }
  // ahead of the song: eased back the same way
  {
    const next = followClock({ cur: 5, target: free - 1, delta: d });
    assert.ok(next < free && next > free - 1, next);
  }
  // convergence: repeated frames close the gap without overshoot
  {
    let cur = 5, target = 5 + FOLLOW_SNAP / 2; // half a snap out, and static
    for (let i = 0; i < 600; i++)
      cur = followClock({ cur, target, delta: d, speed: 0 });
    assert.ok(Math.abs(cur - target) < 1e-6, `converged to ${cur}`);
  }
  // a gap bigger than the snap is not drift (scrub/load/stall): take it whole
  assert.equal(followClock({ cur: 0, target: 900, delta: d }), 900);
  assert.equal(followClock({ cur: 900, target: 0, delta: d }), 0);
  // a huge frame delta must not overshoot the target either
  {
    const next = followClock({ cur: 5, target: 5.2, delta: 1, speed: 0 });
    assert.ok(Math.abs(next - 5.2) < 1e-12, next);
  }
}

console.log("sync.js policy test OK");
