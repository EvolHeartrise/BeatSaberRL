// The blade history's ring buffer and windows (lib/tiptrail.js): a window
// is a pure function of (clock, window) — playing forward frame by frame,
// at any speed, gives the same points as a fresh refill; a seek leaves no
// stale samples; the head sits at the clock and the tail exactly one
// window back; nothing is re-sampled while playing forward or paused.
// Run: node test/tiptrail.test.js
import assert from "node:assert/strict";
import {
  PoseTrail, TIP_TRAIL_DEFAULT_S, TIP_TRAIL_MAX_S, TIP_TRAIL_MIN_S, TIP_TRAIL_RATE,
  TrailRing, TrailView, clampTrailS,
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

// one history + one window, the way TipTrail.svelte reads it
function mk() {
  const trail = new PoseTrail({ stride: 6 });
  const view = new TrailView(trail);
  return {
    trail, view, ring: trail.ring,
    update(t, w, cover = w) { trail.update(t, cover, sample); return view.build(w); },
  };
}
function points(view) {
  const out = [];
  for (let i = 0; i < view.npts; i++)
    out.push(Array.from(view.pts.subarray(i * 6, i * 6 + 6)));
  return out;
}
function checkAgainstFresh(tr, tNow, w, msg) {
  const fresh = mk();
  fresh.update(tNow, w);
  assert.equal(tr.view.npts, fresh.view.npts, msg + ": point count");
  const a = points(tr.view), b = points(fresh.view);
  for (let i = 0; i < a.length; i++)
    for (let c = 0; c < 6; c++) close(a[i][c], b[i][c], `${msg}: pt ${i}.${c}`);
  for (let i = 0; i < a.length; i++) close(tr.view.age[i], fresh.view.age[i], `${msg}: age ${i}`);
}

// ---- head at the clock, tail one window back, ages monotone --------------
{
  const tr = mk();
  const tNow = 3.1234, w = TIP_TRAIL_DEFAULT_S;
  tr.update(tNow, w);
  const pts = points(tr.view);
  const head = curve(tNow), tail = curve(tNow - w);
  for (let c = 0; c < 6; c++) {
    close(pts[pts.length - 1][c], head[c], "head = the tip at the clock");
    close(pts[0][c], tail[c], "tail = the tip one window back");
  }
  const { age, npts } = tr.view;
  assert.equal(age[0], 0);
  assert.equal(age[npts - 1], 1);
  for (let i = 1; i < npts; i++) assert.ok(age[i] > age[i - 1], "ages increase");
  // grid density: ~RATE samples per second of window (+ tail + head)
  assert.ok(Math.abs(npts - (w * TIP_TRAIL_RATE + 2)) <= 1, "point count " + npts);
}

// ---- playing forward == a fresh refill, at any speed; no re-sampling ------
for (const speed of [0.25, 1, 4]) {
  const tr = mk();
  let t = 1.0;
  const dt = speed / 60; // 60 fps
  tr.update(t, TIP_TRAIL_DEFAULT_S);
  const fills0 = tr.ring.fills;
  const sampled0 = tr.ring.sampled;
  for (let i = 0; i < 200; i++) {
    t += dt;
    tr.update(t, TIP_TRAIL_DEFAULT_S);
  }
  assert.equal(tr.ring.fills, fills0, `speed ${speed}: forward play never refills`);
  // each grid sample is taken once: only the ones the clock passed
  const passed = Math.floor(t * TIP_TRAIL_RATE + 1e-9) - Math.floor(1.0 * TIP_TRAIL_RATE + 1e-9);
  assert.equal(tr.ring.sampled - sampled0, passed, `speed ${speed}: grid samples taken once`);
  checkAgainstFresh(tr, t, TIP_TRAIL_DEFAULT_S, `speed ${speed}`);
}

// ---- seeks: back, far forward, into the start --------------------------------
{
  const tr = mk();
  tr.update(5.0, 0.2);
  for (const t of [2.0, 2.01, 9.5, 0.05, 0.0, 0.3]) {
    tr.update(t, 0.2);
    checkAgainstFresh(tr, t, 0.2, `seek to ${t}`);
    const pts = points(tr.view);
    // every point lies inside the window: no stray segment from before the seek
    for (let i = 0; i < pts.length; i++) {
      // the z of hand 0 is the sample time on the synthetic curve
      assert.ok(pts[i][2] >= Math.max(0, t - 0.2) - 1e-5 && pts[i][2] <= t + 1e-5,
        `seek to ${t}: point ${i} at t=${pts[i][2]} is outside the window`);
    }
  }
  // clamped at the replay start: at t = 0 a single point, no segments
  tr.update(0, 0.2);
  assert.equal(tr.view.npts, 1);
  const pos = new Float32Array(600), col = new Float32Array(600);
  assert.equal(tr.view.writeSegments(0, pos, col, [1, 1, 1], [0, 0, 0]), 0);
}

// ---- paused: the same clock gives the same line, without sampling ---------
{
  const tr = mk();
  tr.update(4.2, 0.2);
  const before = points(tr.view);
  const sampled = tr.ring.sampled, n0 = calls;
  tr.update(4.2, 0.2);
  assert.deepEqual(points(tr.view), before);
  assert.equal(tr.ring.sampled, sampled, "a paused frame takes no grid samples");
  assert.ok(calls - n0 <= 2, "a paused frame samples at most the head and tail");
}

// ---- window changes: grow refills, shrink trims; both match a refill ------
{
  const tr = mk();
  tr.update(4.0, 0.2);
  tr.update(4.0, 0.4);
  checkAgainstFresh(tr, 4.0, 0.4, "window grown");
  tr.update(4.0, 0.1);
  checkAgainstFresh(tr, 4.0, 0.1, "window shrunk");
  // a window longer than the ring is clamped, never overflows
  tr.update(6.0, 10);
  assert.ok(tr.view.npts <= tr.trail.maxPts);
}

// ---- two windows over one history (the ribbon + the tip trail) ------------
{
  const tr = mk();
  const short = new TrailView(tr.trail);
  let t = 2.0;
  for (let i = 0; i < 50; i++, t += 1 / 60) {
    tr.update(t, 0.3, 0.3);
    short.build(0.1);
    const fresh = mk();
    fresh.update(t, 0.1);
    assert.equal(short.npts, fresh.view.npts);
    for (let p = 0; p < short.npts * 6; p++) close(short.pts[p], fresh.view.pts[p], "short window");
    // the short window is the tail end of the long one, point for point
    const off = tr.view.npts - short.npts;
    for (let p = 6; p < short.npts * 6; p++)
      close(short.pts[p], tr.view.pts[off * 6 + p], "short window = end of the long one");
  }
}

// ---- reset forgets everything -------------------------------------------------
{
  const tr = mk();
  tr.update(4.0, 0.2);
  tr.trail.reset();
  assert.equal(tr.ring.n, 0);
}

// ---- writeSegments: LineSegmentsGeometry layout + the colour ramp ----------
{
  const tr = mk();
  tr.update(2.0, 0.2);
  const v = tr.view;
  const pos = new Float32Array(tr.trail.maxPts * 6), col = new Float32Array(tr.trail.maxPts * 6);
  const head = [1, 0.9, 0.8], tail = [0.2, 0.1, 0];
  for (const comp of [0, 3]) {
    const n = v.writeSegments(comp, pos, col, head, tail);
    assert.equal(n, v.npts - 1);
    for (let i = 0; i < n; i++) {
      for (let c = 0; c < 3; c++) {
        close(pos[i * 6 + c], v.pts[i * 6 + comp + c], "segment start");
        close(pos[i * 6 + 3 + c], v.pts[(i + 1) * 6 + comp + c], "segment end");
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
