// The Flux judge the WebXR game plays with: rust/flux-wasm, i.e. the env's
// OWN rule files (rust/src/geom.rs, flux.rs, combo.rs) compiled to wasm32.
// Nothing in this file decides an outcome — it only moves numbers across the
// C ABI (see rust/flux-wasm/src/lib.rs). Framework-free, so node tests and
// the game share it.
import WASM_B64 from "./fluxcore.gen.js";

/** Floats per chart object / step input / event — flux-wasm OBJ_F etc. */
const OBJ_F = 9, STEP_F = 16, EVENT_F = 5;

/** Field order of flux-wasm's `Stats` (#[repr(C)], all f64). */
export const STAT_NAMES = [
  "tick", "time", "good", "bad", "misses", "raw_score", "score_base",
  "score_acc", "score", "combo", "max_combo", "multiplier", "progress",
  "breaks", "reward", "finished", "n_objects", "max_score",
];

/** Order of flux-wasm's CONSTS table. */
const CONST_NAMES = [
  "DT", "PALM_OFF", "CIRCLE_R", "BAR_W", "SLAB_LO", "SLAB_HI", "PLANE_Z0",
  "COS_TOL",
];

/** Event codes -> the env's reason names (constants.EVENT_REASONS). */
export const REASONS = { 0: "ok", 2: "angle", 3: "wrong_hand", 4: "miss" };

function decodeB64(s) {
  const bin = atob(s);
  const out = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) out[i] = bin.charCodeAt(i);
  return out;
}

class FluxCore {
  constructor(exports) {
    this.x = exports;
    const n = exports.fg_consts_len();
    const c = this.f64(exports.fg_consts_ptr(), n);
    this.consts = Object.fromEntries(CONST_NAMES.map((k, i) => [k, c[i]]));
    if (exports.fg_stats_len() !== STAT_NAMES.length)
      throw new Error("flux-wasm Stats layout changed: update STAT_NAMES");
  }

  // Views are made per access: a memory.grow detaches every earlier view.
  f64(ptr, n) { return new Float64Array(this.x.memory.buffer, ptr, n); }
  u8(ptr, n) { return new Uint8Array(this.x.memory.buffer, ptr, n); }

  /**
   * A new game over `chart` — viewer/replay-header objects
   * {t, x, y, kind, color, rot, speed, len, sweet} or set_flux_chart tuples.
   * `maxSteps` 0 = the env's fixed-chart budget. Throws with the env's own
   * message on a chart the env would refuse.
   */
  newGame(chart, maxSteps = 0) {
    const n = chart.length;
    const buf = this.x.fg_alloc(Math.max(1, n * OBJ_F));
    const v = this.f64(buf, Math.max(1, n * OBJ_F));
    chart.forEach((o, k) => {
      const row = Array.isArray(o)
        ? o
        : [o.t, o.x, o.y, o.kind, o.color ?? 0, o.rot ?? 0, o.speed,
           o.len ?? 0, o.sweet ?? 0];
      for (let j = 0; j < OBJ_F; j++) v[k * OBJ_F + j] = row[j];
    });
    const g = this.x.fg_new(buf, n, maxSteps);
    this.x.fg_dealloc(buf, Math.max(1, n * OBJ_F));
    if (!g) {
      const msg = new TextDecoder().decode(
        this.u8(this.x.fg_error_ptr(), this.x.fg_error_len()));
      throw new Error("chart refused: " + msg);
    }
    return new FluxJudge(this, g, n);
  }
}

/** One game: feed it one tick at a time. */
export class FluxJudge {
  constructor(core, ptr, n) {
    this.core = core;
    this.ptr = ptr;
    this.n = n;
    this.inp = core.x.fg_alloc(STEP_F);
  }

  /**
   * One 30 Hz tick. `p0` / `p1` = [[lx, ly, lz], [rx, ry, rz]] palms (env
   * world) at the previous and this tick, `head` = [x, y, z], `shoulderZ` =
   * the forward-most shoulder's z (the reach gate). Returns
   * {done, events: [{i, code, reason, pts, pre, acc}]}.
   */
  step(p0, p1, head, shoulderZ) {
    const v = this.core.f64(this.inp, STEP_F);
    v.set(p0[0], 0); v.set(p0[1], 3); v.set(p1[0], 6); v.set(p1[1], 9);
    v.set(head, 12); v[15] = shoulderZ;
    return this.stepRaw();
  }

  /** Same, from a flat STEP_F array (the parity fixture's rows). */
  stepFlat(row) {
    this.core.f64(this.inp, STEP_F).set(row);
    return this.stepRaw();
  }

  stepRaw() {
    const x = this.core.x;
    const done = x.fg_step(this.ptr, this.inp) === 1;
    const ne = x.fg_events_len(this.ptr);
    const events = [];
    if (ne) {
      const e = this.core.f64(x.fg_events_ptr(this.ptr), ne * EVENT_F);
      for (let k = 0; k < ne; k++) {
        const b = k * EVENT_F;
        events.push({ i: e[b], code: e[b + 1], reason: REASONS[e[b + 1]],
                      pts: e[b + 2], pre: e[b + 3], acc: e[b + 4] });
      }
    }
    return { done, events };
  }

  /** Running tallies ({tick, time, good, ..., score, combo, multiplier}). */
  stats() {
    const s = this.core.f64(this.core.x.fg_stats_ptr(this.ptr), STAT_NAMES.length);
    return Object.fromEntries(STAT_NAMES.map((k, i) => [k, s[i]]));
  }

  /** Per-object status: 0 pending, 1 good, 2 bad, 3 missed. */
  status() {
    return Array.from(this.core.u8(this.core.x.fg_status_ptr(this.ptr), this.n));
  }

  free() {
    if (!this.ptr) return;
    this.core.x.fg_free(this.ptr);
    this.core.x.fg_dealloc(this.inp, STEP_F);
    this.ptr = 0;
  }
}

let loading = null;
/** The (singleton) judge module; resolves once the wasm is instantiated. */
export function loadFluxCore() {
  loading ??= WebAssembly.instantiate(decodeB64(WASM_B64), {})
    .then(({ instance }) => new FluxCore(instance.exports));
  return loading;
}
