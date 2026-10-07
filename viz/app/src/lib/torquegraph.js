// The timeline's effort profile (ControlBar.svelte's TorqueSeek): every
// tick's applied torque — the replay's "torque" field, Σ|τ| over the 29
// joints in N·m as [total, spine, arms, legs] (record_rev 3/4; replay.js
// decodes it, or derives it from the action x header.body) — reduced once
// per replay and per track width to one column per device pixel.
//
// Pure functions, no DOM: test/torquegraph.test.js covers them.

/** The replay's per-tick TOTAL torque as a Float32Array (one entry per
 * tick), or null when the replay carries none (a person's Flux run from
 * the WebXR game: no body) — the timeline then falls back to a plain
 * track. */
export function torqueSeries(rep) {
  const ticks = rep?.ticks;
  if (!ticks?.length) return null;
  const out = new Float32Array(ticks.length);
  let any = false;
  for (let k = 0; k < ticks.length; k++) {
    const v = ticks[k].torque?.[0];
    if (!Number.isFinite(v)) return null; // partial data: no graph at all
    out[k] = Math.abs(v);
    if (v !== 0) any = true;
  }
  return any ? out : null;
}

/** Column `i` of `cols` covers ticks [lo, hi) of `n`: consecutive,
 * non-overlapping, every tick in exactly one column when n >= cols; with
 * fewer ticks than columns each column shows the tick under it. */
export function columnRange(i, cols, n) {
  const lo = Math.min(n - 1, Math.floor((i * n) / cols));
  const hi = Math.max(lo + 1, Math.min(n, Math.floor(((i + 1) * n) / cols)));
  return [lo, hi];
}

/** Per-column max and mean of `series` over `cols` columns. The max keeps
 * a one-tick spike visible however many ticks share a pixel; the mean is
 * the column's typical level. */
export function downsample(series, cols) {
  const n = series.length;
  const max = new Float32Array(cols), mean = new Float32Array(cols);
  for (let i = 0; i < cols; i++) {
    const [lo, hi] = columnRange(i, cols, n);
    let m = -Infinity, s = 0;
    for (let k = lo; k < hi; k++) {
      const v = series[k];
      if (v > m) m = v;
      s += v;
    }
    max[i] = m;
    mean[i] = s / (hi - lo);
  }
  return { max, mean };
}

/** The graph's full-scale value: the 99th percentile of the series (a
 * robust max), never below 1 N·m. Linear and zero-based so a column's
 * height reads as its N·m; the rare ticks above p99 clip at the top
 * (and are marked there) instead of flattening the whole profile — a
 * single full-gear spike sits at ~1.5–3x the p99 in real replays. */
export function robustMax(series, q = 0.99) {
  const s = Float32Array.from(series).sort();
  const v = s[Math.min(s.length - 1, Math.floor(q * (s.length - 1)))];
  return Math.max(1, v);
}

/** Ticks with a miss, a bad cut or a bomb hit, for the faint marks along
 * the bottom of the graph. */
export function mistakeTicks(rep) {
  const out = [];
  const ticks = rep?.ticks ?? [];
  for (let k = 0; k < ticks.length; k++) {
    const ev = ticks[k].events;
    if (!ev?.length) continue;
    if (ev.some((e) => e.miss || e.bomb || (e.cut && e.cut.good === false))) out.push(k);
  }
  return out;
}

/** Fraction along the track (0..1) <-> tick (0..last). */
export const tickToFrac = (k, last) => (last > 0 ? k / last : 0);
export const fracToTick = (f, last) => Math.round(Math.min(1, Math.max(0, f)) * last);
