// Blade history for the TIP TRAIL (TipTrail.svelte) and the swept RIBBON
// (Saber.svelte): one fixed-size ring of samples on a replay-time grid,
// read through windows of any length up to the ring's. Pure JS (no
// three.js, no replay format) so it can be unit-tested in node; the FK
// sampler that fills it is lib/bladetrail.js.
//
// Sampling: a fixed grid of replay time (RATE samples per replay second),
// never per logged tick and never per rendered frame, so a curve is the
// same at any playback speed, while scrubbing and at any tick rate. The
// newest point of every window is taken at the playback clock itself, so
// it sits on the drawn blade, and the oldest is sampled exactly `window`
// seconds back, so the length does not flicker between grid steps.
//
// Storage: playing forward only samples the grid points the clock passed
// since the last frame and drops the ones that left the window; any jump
// the ring cannot extend (a seek back, a skip past the window) refills it
// from scratch. A grid sample is a pure function of its time, so an
// extended ring and a refilled one hold the same values: nothing stale can
// survive a seek. Nothing here allocates after construction.

export const TIP_TRAIL_RATE = 240;      // grid samples per replay second
export const TIP_TRAIL_DEFAULT_S = 0.2; // tip trail window, replay seconds
export const TIP_TRAIL_MIN_S = 0.05;
export const TIP_TRAIL_MAX_S = 0.5;

const EPS = 1e-9;

export const clampTrailS = (s) => {
  const v = Number(s);
  if (!Number.isFinite(v)) return TIP_TRAIL_DEFAULT_S;
  return Math.min(TIP_TRAIL_MAX_S, Math.max(TIP_TRAIL_MIN_S, v));
};

/** Fixed-capacity ring of samples on the grid t = g / rate, each `stride`
 * floats. Holds the consecutive grid indices [g0, g0 + n). */
export class TrailRing {
  constructor(stride, cap) {
    this.stride = stride;
    this.cap = cap;
    this.buf = new Float32Array(cap * stride);
    this.g0 = 0;   // grid index of the oldest sample
    this.n = 0;    // samples held
    this.head = 0; // ring slot of the oldest sample
    this.fills = 0; // full refills (test / debug counter)
    this.sampled = 0; // grid samples taken in total (test / debug counter)
  }

  reset() {
    this.n = 0;
  }

  /** Buffer offset of grid sample g (g0 <= g < g0 + n). */
  offsetOf(g) {
    return ((this.head + (g - this.g0)) % this.cap) * this.stride;
  }

  /** Make the ring hold exactly the grid samples [gLo, gHi], calling
   * sample(g, buf, off) only for the ones it does not already hold. */
  cover(gLo, gHi, sample) {
    if (gHi < gLo) { this.n = 0; return; }
    if (gHi - gLo + 1 > this.cap) throw new Error("trail window exceeds the ring");
    const gLast = this.g0 + this.n - 1;
    if (this.n === 0 || gLo < this.g0 || gHi < gLast || gLo > gLast + 1) {
      // not a forward extension: refill from scratch
      this.fills++;
      this.g0 = gLo;
      this.head = 0;
      this.n = 0;
    } else {
      const drop = gLo - this.g0; // samples that left the window
      this.head = (this.head + drop) % this.cap;
      this.g0 = gLo;
      this.n -= drop;
    }
    for (let g = this.g0 + this.n; g <= gHi; g++) {
      sample(g, this.buf, ((this.head + this.n) % this.cap) * this.stride);
      this.n++;
      this.sampled++;
    }
  }
}

/** The sampled history: a ring covering [tNow - cover, tNow] plus the
 * sample at tNow itself. `sample(t, out, off)` writes `stride` floats for
 * replay time t (seconds) into out[off ..]. */
export class PoseTrail {
  constructor({ stride = 6, maxS = TIP_TRAIL_MAX_S, rate = TIP_TRAIL_RATE } = {}) {
    this.stride = stride;
    this.rate = rate;
    this.ring = new TrailRing(stride, Math.ceil(maxS * rate) + 2);
    this.maxS = (this.ring.cap - 2) / rate;
    this.maxPts = this.ring.cap + 2; // [tail] + ring + [head]
    this.headBuf = new Float32Array(stride);
    this.headOnGrid = false;
    this.tNow = 0;
    this.stamp = 0; // bumped by every update / reset (views rebuild on it)
    this.sample = null;
    this._sample = null;
    this._gridSample = null;
  }

  /** Forget everything (a new replay). */
  reset() {
    this.ring.reset();
    this.stamp++;
  }

  /** Bring the history to clock `tNow` (s), covering `coverS` seconds. */
  update(tNow, coverS, sample) {
    const rate = this.rate;
    if (this._sample !== sample) { // bind once per sampler, not per frame
      this._sample = sample;
      this._gridSample = (g, out, off) => sample(g / rate, out, off);
    }
    this.sample = sample;
    const w = Math.min(coverS, this.maxS);
    const gLo = Math.ceil(Math.max(0, tNow - w) * rate - EPS);
    const gHi = Math.floor(tNow * rate + EPS);
    this.ring.cover(gLo, gHi, this._gridSample);
    this.headOnGrid = gHi / rate >= tNow - EPS && this.ring.n > 0;
    if (!this.headOnGrid) sample(tNow, this.headBuf, 0);
    this.tNow = tNow;
    this.stamp++;
  }
}

/** The last `windowS` seconds of a PoseTrail as an ordered polyline:
 * pts (stride floats per point) and age (0 = tail .. 1 = head). */
export class TrailView {
  constructor(trail) {
    this.trail = trail;
    this.pts = new Float32Array(trail.maxPts * trail.stride);
    this.age = new Float32Array(trail.maxPts);
    this.npts = 0;
    this.stamp = -1;
    this.windowS = -1;
  }

  /** Rebuild for the trail's current clock (a no-op when nothing moved). */
  build(windowS) {
    const tr = this.trail;
    if (this.stamp === tr.stamp && this.windowS === windowS) return this.npts;
    this.stamp = tr.stamp;
    this.windowS = windowS;
    const S = tr.stride, rate = tr.rate, ring = tr.ring, tNow = tr.tNow;
    const pts = this.pts, age = this.age;
    if (ring.n === 0 && !tr.sample) { this.npts = 0; return 0; }
    const w = Math.min(windowS, tr.maxS);
    const tLo = Math.max(0, tNow - w);
    const span = tNow - tLo;
    const ageOf = (t) => (span > 0 ? (t - tLo) / span : 1);
    let gLo = Math.max(ring.g0, Math.ceil(tLo * rate - EPS));
    let n = 0;
    if (gLo / rate > tLo + EPS) { // the tail, exactly `w` back
      tr.sample(tLo, pts, 0);
      age[n++] = 0;
    }
    for (let g = gLo; g < ring.g0 + ring.n; g++) {
      const src = ring.offsetOf(g), dst = n * S;
      for (let c = 0; c < S; c++) pts[dst + c] = ring.buf[src + c];
      age[n++] = ageOf(g / rate);
    }
    if (!tr.headOnGrid) { // the head, at the clock
      pts.set(tr.headBuf, n * S);
      age[n++] = 1;
    }
    this.npts = n;
    return n;
  }

  /** Write the polyline of the point at component offset `comp` (0, 3, ..)
   * as line SEGMENTS — [x0 y0 z0 x1 y1 z1] per segment, three.js
   * LineSegmentsGeometry's instance layout — into `pos`, and per-endpoint
   * RGB lerped from `tail` (age 0) to `head` (age 1) into `col`. Returns
   * the segment count. */
  writeSegments(comp, pos, col, head, tail) {
    const S = this.trail.stride, pts = this.pts, age = this.age;
    const nseg = Math.max(0, this.npts - 1);
    for (let i = 0; i < nseg; i++) {
      for (let e = 0; e < 2; e++) {
        const p = (i + e) * S + comp, d = i * 6 + e * 3;
        pos[d] = pts[p];
        pos[d + 1] = pts[p + 1];
        pos[d + 2] = pts[p + 2];
        const a = age[i + e];
        col[d] = tail[0] + (head[0] - tail[0]) * a;
        col[d + 1] = tail[1] + (head[1] - tail[1]) * a;
        col[d + 2] = tail[2] + (head[2] - tail[2]) * a;
      }
    }
    return nseg;
  }
}
