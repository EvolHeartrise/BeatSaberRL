// The blade-tip trail's ring buffer and sampling (lib/tiptrail.js): the
// polyline is a pure function of (clock, window) — playing forward frame
// by frame, at any speed, gives the same points as a fresh refill; a seek
// leaves no stale samples; the head sits at the clock and the tail exactly
// one window back; nothing is re-sampled while playing forward.
// Run: node test/tiptrail.test.js
import assert from "node:assert/strict";
import {
  TIP_TRAIL_DEFAULT_S, TIP_TRAIL_MAX_S, TIP_TRAIL_MIN_S, TIP_TRAIL_RATE,
  TipTrail, TrailRing, clampTrailS,
} from "../src/lib/tiptrail.js";

// a smooth synthetic "tip": two points (both hands) on known curves
const curve = (t) => [Math.sin(3 * t), Math.cos(2 * t), t, -Math.sin(t), 0.5 * t, 1];
let calls = 0;
const sample = (t, out, off) => {
  calls++;
  const v = curve(t);
  for (let c = 0; c < 6; c++) out[off + c] = v[c];
};
const close = (a, b, msg) => assert.ok(Math.abs(a - b) < 1e-5, `${msg}: ${a} vs ${b}`);

function points(tr) {
  const out = [];
  for (let i = 0; i < tr.npts; i++)
    out.push(Array.from(tr.pts.subarray(i * 6, i * 6 + 6)));
  return out;
}
function checkAgainstFresh(tr, tNow, w, msg) {
  const fresh = new TipTrail();
  fresh.update(tNow, w, sample);
  assert.equal(tr.npts, fresh.npts, msg + ": point count");
  const a = points(tr), b = points(fresh);
  for (let i = 0; i < a.length; i++)
    for (let c = 0; c < 6; c++) close(a[i][c], b[i][c], `${msg}: pt ${i}.${c}`);
  for (let i = 0; i < tr.npts; i++) close(tr.age[i], fresh.age[i], `${msg}: age ${i}`);
}

// ---- head at the clock, tail one window back, ages monotone --------------
{
  const tr = new TipTrail();
  const tNow = 3.1234, w = TIP_TRAIL_DEFAULT_S;
  tr.update(tNow, w, sample);
  const pts = points(tr);
  const head = curve(tNow), tail = curve(tNow - w);
  for (let c = 0; c < 6; c++) {
    close(pts[pts.length - 1][c], head[c], "head = the tip at the clock");
    close(pts[0][c], tail[c], "tail = the tip one window back");
  }
  assert.equal(tr.age[0], 0);
  assert.equal(tr.age[tr.npts - 1], 1);
  for (let i = 1; i < tr.npts; i++) assert.ok(tr.age[i] > tr.age[i - 1], "ages increase");
  // grid density: ~RATE samples per second of window (+ tail + head)
  assert.ok(Math.abs(tr.npts - (w * TIP_TRAIL_RATE + 2)) <= 1, "point count " + tr.npts);
}

// ---- playing forward == a fresh refill, at any speed; no re-sampling ------
for (const speed of [0.25, 1, 4]) {
  const tr = new TipTrail();
  let t = 1.0;
  const dt = speed / 60; // 60 fps
  tr.update(t, TIP_TRAIL_DEFAULT_S, sample);
  const fills0 = tr.ring.fills;
  const sampled0 = tr.ring.sampled;
  const steps = 200;
  for (let i = 0; i < steps; i++) {
    t += dt;
    tr.update(t, TIP_TRAIL_DEFAULT_S, sample);
  }
  assert.equal(tr.ring.fills, fills0, `speed ${speed}: forward play never refills`);
  // each grid sample is taken once: only the ones the clock passed
  const passed = Math.floor(t * TIP_TRAIL_RATE + 1e-9) - Math.floor(1.0 * TIP_TRAIL_RATE + 1e-9);
  assert.equal(tr.ring.sampled - sampled0, passed, `speed ${speed}: grid samples taken once`);
  checkAgainstFresh(tr, t, TIP_TRAIL_DEFAULT_S, `speed ${speed}`);
}

// ---- seeks: back, far forward, into the start --------------------------------
{
  const tr = new TipTrail();
  tr.update(5.0, 0.2, sample);
  for (const t of [2.0, 2.01, 9.5, 0.05, 0.0, 0.3]) {
    tr.update(t, 0.2, sample);
    checkAgainstFresh(tr, t, 0.2, `seek to ${t}`);
    const pts = points(tr);
    // every point lies inside the window: no stray segment from before the seek
    for (let i = 0; i < tr.npts; i++) {
      // the z of hand 0 is the sample time on the synthetic curve
      assert.ok(pts[i][2] >= Math.max(0, t - 0.2) - 1e-5 && pts[i][2] <= t + 1e-5,
        `seek to ${t}: point ${i} at t=${pts[i][2]} is outside the window`);
    }
  }
  // clamped at the replay start: at t = 0 a single point, no segments
  tr.update(0, 0.2, sample);
  assert.equal(tr.npts, 1);
  const pos = new Float32Array(600), col = new Float32Array(600);
  assert.equal(tr.writeSegments(0, pos, col, [1, 1, 1], [0, 0, 0]), 0);
}

// ---- paused: the same clock gives the same line, without sampling ---------
{
  const tr = new TipTrail();
  tr.update(4.2, 0.2, sample);
  const before = points(tr);
  const sampled = tr.ring.sampled;
  tr.update(4.2, 0.2, sample);
  assert.deepEqual(points(tr), before);
  assert.equal(tr.ring.sampled, sampled, "a paused frame takes no grid samples");
}

// ---- window changes: grow refills, shrink trims; both match a refill ------
{
  const tr = new TipTrail();
  tr.update(4.0, 0.2, sample);
  tr.update(4.0, 0.4, sample);
  checkAgainstFresh(tr, 4.0, 0.4, "window grown");
  tr.update(4.0, 0.1, sample);
  checkAgainstFresh(tr, 4.0, 0.1, "window shrunk");
  // a window longer than the ring is clamped, never overflows
  tr.update(6.0, 10, sample);
  assert.ok(tr.npts <= tr.maxPts);
}

// ---- reset forgets everything -------------------------------------------------
{
  const tr = new TipTrail();
  tr.update(4.0, 0.2, sample);
  tr.reset();
  assert.equal(tr.npts, 0);
  assert.equal(tr.ring.n, 0);
}

// ---- writeSegments: LineSegmentsGeometry layout + the colour ramp ----------
{
  const tr = new TipTrail();
  tr.update(2.0, 0.2, sample);
  const pos = new Float32Array(tr.maxPts * 6), col = new Float32Array(tr.maxPts * 6);
  const head = [1, 0.9, 0.8], tail = [0.2, 0.1, 0];
  for (const comp of [0, 3]) {
    const n = tr.writeSegments(comp, pos, col, head, tail);
    assert.equal(n, tr.npts - 1);
    for (let i = 0; i < n; i++) {
      for (let c = 0; c < 3; c++) {
        close(pos[i * 6 + c], tr.pts[i * 6 + comp + c], "segment start");
        close(pos[i * 6 + 3 + c], tr.pts[(i + 1) * 6 + comp + c], "segment end");
      }
      // consecutive segments join
      if (i > 0) for (let c = 0; c < 3; c++) close(pos[i * 6 + c], pos[(i - 1) * 6 + 3 + c], "joined");
    }
    for (let c = 0; c < 3; c++) {
      close(col[c], tail[c], "tail colour");
      close(col[(n - 1) * 6 + 3 + c], head[c], "head colour");
    }
  }
}

// ---- the ring wraps without losing order ---------------------------------------
{
  const ring = new TrailRing(1, 5);
  const put = (g, out, off) => { out[off] = g; };
  ring.cover(0, 4, put);
  for (let g = 5; g < 23; g++) {
    ring.cover(g - 4, g, put);
    for (let x = ring.g0; x < ring.g0 + ring.n; x++)
      assert.equal(ring.buf[ring.offsetOf(x)], x, `wrapped slot for ${x}`);
  }
  assert.equal(ring.fills, 1);
  assert.throws(() => ring.cover(0, 5, put), /exceeds/);
}

// ---- the length setting --------------------------------------------------------
assert.equal(clampTrailS(0.3), 0.3);
assert.equal(clampTrailS(99), TIP_TRAIL_MAX_S);
assert.equal(clampTrailS(-1), TIP_TRAIL_MIN_S);
assert.equal(clampTrailS("x"), TIP_TRAIL_DEFAULT_S);

console.log("tiptrail: ok (" + calls + " samples)");
