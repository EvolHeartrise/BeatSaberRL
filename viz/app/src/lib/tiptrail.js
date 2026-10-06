// The TIP TRAIL (viewer cue, 2026-10-06): a thin, crisp polyline through
// the path each blade tip took over the last fraction of a second, drawn
// over the wide swept ribbon (Saber.svelte) so the tip's motion reads at a
// glance. Pure JS (no three.js) so it can be unit-tested in node; the
// drawing is TipTrail.svelte.
//
// Sampling: the tip is sampled from the INTERPOLATED pose — the same body
// FK of the lerped joint angles that draws the sabers — on a fixed grid of
// replay time (RATE samples per replay second), never per logged tick and
// never per rendered frame. So the line is the same smooth curve at any
// playback speed, while scrubbing and at any tick rate; the newest point
// is taken at the playback clock itself, so the line stays welded to the
// drawn tip, and the oldest is interpolated to exactly `window` seconds
// back, so the length does not flicker between grid steps.
//
// Storage: a fixed-size ring of grid samples. Playing forward only samples
// the grid points the clock passed since the last frame and drops the ones
// that left the window; any jump the ring cannot extend (a seek back, a
// skip past the window, a new replay) refills it from scratch. A grid
// sample is a pure function of its time, so an extended ring and a
// refilled one hold the same values: nothing stale can survive a seek.
// Nothing here allocates after construction.

export const TIP_TRAIL_RATE = 240;      // grid samples per replay second
export const TIP_TRAIL_DEFAULT_S = 0.2; // window, replay seconds
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

/** The tip trail of `stride / 3` points sampled together (both blade tips
 * from one FK per sample time). `sample(t, out, off)` writes the points at
 * replay time t (seconds) into out[off ..]. */
export class TipTrail {
  constructor({ stride = 6, maxS = TIP_TRAIL_MAX_S, rate = TIP_TRAIL_RATE } = {}) {
    this.stride = stride;
    this.rate = rate;
    this.ring = new TrailRing(stride, Math.ceil(maxS * rate) + 2);
    // the ordered polyline: [tail] + ring samples + [head]
    this.maxPts = this.ring.cap + 2;
    this.pts = new Float32Array(this.maxPts * stride);
    this.age = new Float32Array(this.maxPts); // 0 = tail .. 1 = head
    this.npts = 0;
    this._gridSample = null;
    this._sample = null;
  }

  /** Forget everything (a seek, a new replay, the trail switched off). */
  reset() {
    this.ring.reset();
    this.npts = 0;
  }

  /** Rebuild the polyline for clock `tNow` (s) and window `windowS` (s). */
  update(tNow, windowS, sample) {
    const rate = this.rate;
    if (this._sample !== sample) { // bind once per sampler, not per frame
      this._sample = sample;
      this._gridSample = (g, out, off) => sample(g / rate, out, off);
    }
    const w = Math.min(windowS, (this.ring.cap - 2) / rate);
    const tLo = Math.max(0, tNow - w);
    const gLo = Math.ceil(tLo * rate - EPS);
    const gHi = Math.floor(tNow * rate + EPS);
    this.ring.cover(gLo, gHi, this._gridSample);

    const S = this.stride, pts = this.pts, age = this.age;
    const span = tNow - tLo;
    const ageOf = (t) => (span > 0 ? (t - tLo) / span : 1);
    let n = 0;
    if (gLo / rate > tLo + EPS) { // the tail, exactly `w` back
      sample(tLo, pts, 0);
      age[n++] = 0;
    }
    const ring = this.ring;
    for (let g = ring.g0; g < ring.g0 + ring.n; g++) {
      const src = ring.offsetOf(g), dst = n * S;
      for (let c = 0; c < S; c++) pts[dst + c] = ring.buf[src + c];
      age[n++] = ageOf(g / rate);
    }
    if (gHi / rate < tNow - EPS || n === 0) { // the head, at the clock
      sample(tNow, pts, n * S);
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
    const S = this.stride, pts = this.pts, age = this.age;
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
