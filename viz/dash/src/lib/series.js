// Pure helpers over metrics records (the metrics_log.py vocabulary).

export const X_MODES = [
  { key: "samples", label: "samples", unit: "steps" },
  { key: "updates", label: "updates", unit: "upd" },
  { key: "wall", label: "wall-clock", unit: "h" },
];

// x values per record. Wall-clock: hours since the run's first record —
// from the `t` epoch stamps when every record has one (trainer-written
// or backfilled from the log mtime), else from `elapsed` made
// cumulative across resumes (elapsed restarts per process).
export function xValues(recs, mode) {
  const n = recs.length;
  const x = new Array(n);
  if (mode === "samples") {
    for (let i = 0; i < n; i++) x[i] = recs[i].steps;
  } else if (mode === "updates") {
    for (let i = 0; i < n; i++) x[i] = recs[i].upd;
  } else {
    const allT = n > 0 && recs.every((r) => typeof r.t === "number");
    if (allT) {
      const t0 = recs[0].t;
      for (let i = 0; i < n; i++) x[i] = (recs[i].t - t0) / 3600;
    } else {
      let off = 0, prev = 0;
      for (let i = 0; i < n; i++) {
        const e = recs[i].elapsed ?? prev;
        if (e < prev) off += prev; // a resume restarted the process clock
        prev = e;
        x[i] = (off + e) / 3600;
      }
    }
  }
  return x;
}

function per1M(r, k, batch) {
  const v = r[k];
  if (v == null || !batch) return null;
  return (v * 1e6) / batch;
}

// Batch size per update (steps advanced), for the per-1M-sample rates.
export function batchOf(recs, i) {
  if (i > 0) {
    const d = recs[i].steps - recs[i - 1].steps;
    if (d > 0) return d;
  }
  if (recs.length > 1 && i === 0) {
    const d = recs[1].steps - recs[0].steps;
    if (d > 0) return d;
  }
  return recs[i].steps / Math.max(1, recs[i].upd);
}

const DERIVED = {
  falls_per_1m: (r, i, recs) => per1M(r, "falls", batchOf(recs, i)),
  tilts_per_1m: (r, i, recs) => per1M(r, "tilts", batchOf(recs, i)),
  lifts_per_1m: (r, i, recs) => per1M(r, "lifts", batchOf(recs, i)),
  sps_k: (r) => (r.sps == null ? null : r.sps / 1000),
  kl_first_pos: (r) => (r.kl_first > 0 ? r.kl_first : null),
  kl_last_pos: (r) => (r.kl_last > 0 ? r.kl_last : null),
  // log-scale variants: a log axis cannot show 0, so zero-event updates
  // become gaps rather than dragging the axis to -inf (2026-09-03)
  falls_per_1m_pos: (r, i, recs) => { const v = per1M(r, "falls", batchOf(recs, i)); return v > 0 ? v : null; },
  tilts_per_1m_pos: (r, i, recs) => { const v = per1M(r, "tilts", batchOf(recs, i)); return v > 0 ? v : null; },
  // env v41 [contact]: share of env-ticks with body-on-body contact, in %
  contact_pct: (r) => (r.contact_frac == null ? null : r.contact_frac * 100),
};

export function seriesValues(recs, key) {
  const f = DERIVED[key];
  const n = recs.length;
  const y = new Array(n);
  for (let i = 0; i < n; i++) {
    const v = f ? f(recs[i], i, recs) : recs[i][key];
    y[i] = typeof v === "number" && Number.isFinite(v) ? v : null;
  }
  return y;
}

export function hasAny(recs, key) {
  const f = DERIVED[key];
  for (let i = 0; i < recs.length; i++) {
    const v = f ? f(recs[i], i, recs) : recs[i][key];
    if (typeof v === "number" && Number.isFinite(v)) return true;
  }
  return false;
}

// Panels: one chart each; `series` are metric keys drawn together.
export const PANELS = [
  { id: "acc", title: "accuracy (note score / 115·cuts)", series: ["acc", "accL", "accR"] },
  { id: "accparts", title: "accuracy by component (points / max over note cuts · pre-swing /70 · post-swing /30 · precision /15)", series: ["acc_pre", "acc_post", "acc_prec"] },
  { id: "return", title: "return (reward per env this update)", series: ["return"] },
  { id: "ent", title: "entropy (ent: last minibatch · entfull: full batch)", series: ["ent", "entfull"] },
  { id: "ev", title: "explained variance", series: ["ev"] },
  { id: "sps", title: "steps / s (thousands)", series: ["sps_k"] },
  { id: "falls", title: "terminations per 1M samples (log)", series: ["falls_per_1m_pos", "tilts_per_1m_pos"], log: true },
  { id: "clipfrac", title: "clipfrac (first / last minibatch · cfh = per-head, --head-clip)", series: ["clipfrac_first", "clipfrac_last", "clipfrac_head_last"] },
  { id: "kl", title: "KL k3 (first / last minibatch, log)", series: ["kl_first_pos", "kl_last_pos"], log: true },
  { id: "gnorm", title: "grad norm · clip coefficient", series: ["gnorm", "clipc"] },
  { id: "entgroups", title: "entropy per head group (mean per head)", series: ["ent_spine", "ent_armL", "ent_armR", "ent_legL", "ent_legR"] },
  { id: "lifts", title: "foot lifts per 1M samples", series: ["lifts_per_1m"] },
  { id: "flail", title: "flail (rad / idle tick)", series: ["flail", "flail_L", "flail_R"] },
  { id: "wrev", title: "wrist-bend reversals per second (per hand · 60° hysteresis since 2026-10-04, 20° before; every tick)", series: ["wrev_L", "wrev_R"] },
  { id: "limitrest", title: "limit rest (N·m·s / tick)", series: ["limitrest", "limitrest_legs", "limitrest_spine", "limitrest_arms"] },
  { id: "activation", title: "joint activation (Σ|gear fraction| / env-tick · total / spine / arms / legs)", series: ["activation", "activation_spine", "activation_arms", "activation_legs"] },
  { id: "torque", title: "joint torque Σ|τ| (N·m / env-tick · total / spine / arms / legs) — the energy price base (env v43)", series: ["torque", "torque_spine", "torque_arms", "torque_legs"] },
  { id: "contact", title: "self-contact impulse (N·s / env-tick, env v41)", series: ["contact"] },
  { id: "contact_pct", title: "self-contact: % of env-ticks with body-on-body contact", series: ["contact_pct"] },
  { id: "fp16", title: "fp16 ratio diag r1 · max log-ratio", series: ["r1", "maxlr"] },
  { id: "vram", title: "peak VRAM at checkpoints (GiB)", series: ["peak_vram"] },
];

// Game mode of a run (index.json `game`, SPEC.md §Flux; absent = standard).
export const gameOf = (run) => run?.game ?? "standard";

// Panels for a run's game mode. Flux (no sabers): the trainer's acc is
// points per touch / 115 — a perfect circle reads 60/115 = 0.52, a
// perfect bar 1.0, and a bar counts for both hands (both palms make the
// Flux line) — so the panel says so; flail (saber arc on idle ticks) is
// structurally zero and dropped.
export function panelsFor(game) {
  if (game !== "flux") return PANELS;
  return PANELS.filter((p) => p.id !== "flail").map((p) => p.id === "acc"
    ? { ...p, title: "FLUX points per touch / 115 (perfect circle 0.52, perfect bar 1.0; bars count for both hands)" }
    : p);
}

// Compare view: scalar metrics worth a small multiple and a diff panel.
export const COMPARE_KEYS = [
  "acc", "acc_pre", "acc_post", "acc_prec", "return", "ent", "entfull", "ev", "sps_k", "falls_per_1m_pos", "tilts_per_1m_pos",
  "clipfrac_last", "kl_last_pos", "gnorm", "clipc", "limitrest", "flail", "wrev", "lifts_per_1m",
  "activation", "activation_spine", "activation_arms", "activation_legs",
  "torque", "torque_spine", "torque_arms", "torque_legs",
  "contact", "contact_pct",
];
// ep_return dropped from every view 2026-09-12 (user: too noisy; the
// trainer no longer emits it — `return` over the update is the measure)
export const DIFF_KEYS = ["acc", "return", "ent", "ev"];

export const LABELS = {
  acc: "acc", accL: "accL", accR: "accR", return: "return",
  acc_pre: "pre-swing", acc_post: "post-swing", acc_prec: "precision",
  ent: "ent", entfull: "entfull", ev: "ev", sps_k: "steps/s (k)",
  falls_per_1m: "falls / 1M", tilts_per_1m: "tilts / 1M", lifts_per_1m: "lifts / 1M",
  falls_per_1m_pos: "falls / 1M (log)", tilts_per_1m_pos: "tilts / 1M (log)",
  clipfrac_first: "clipfrac mb1", clipfrac_last: "clipfrac mbN", clipfrac_head_last: "cfh mbN",
  kl_first_pos: "kl mb1", kl_last_pos: "kl mbN", gnorm: "gnorm", clipc: "clipc",
  ent_spine: "spine", ent_armL: "armL", ent_armR: "armR", ent_legL: "legL", ent_legR: "legR",
  flail: "flail", flail_L: "flail L", flail_R: "flail R",
  wrev: "wrist rev /s", wrev_L: "wrist rev L /s", wrev_R: "wrist rev R /s",
  limitrest: "total", limitrest_legs: "legs", limitrest_spine: "spine", limitrest_arms: "arms",
  activation: "activation", activation_spine: "act. spine", activation_arms: "act. arms", activation_legs: "act. legs",
  torque: "Σ|τ| N·m", torque_spine: "τ spine", torque_arms: "τ arms", torque_legs: "τ legs",
  contact: "N·s / tick", contact_pct: "% ticks in contact",
  r1: "r1", maxlr: "maxlr", peak_vram: "peak VRAM",
};
export const label = (k) => LABELS[k] ?? k;

export const COLORS = [
  "#4e79a7", "#f28e2b", "#e15759", "#76b7b2", "#59a14f", "#edc948",
  "#b07aa1", "#ff9da7", "#9c755f", "#bab0ac", "#1f77b4", "#d62728",
];
export const color = (i) => COLORS[i % COLORS.length];

// Linear interpolation of (xs, ys) at x; null outside [xs[0], xs[n-1]] or
// when the bracketing values are null. xs ascending.
export function interp(xs, ys, x) {
  const n = xs.length;
  if (!n || x < xs[0] || x > xs[n - 1]) return null;
  let lo = 0, hi = n - 1;
  while (hi - lo > 1) {
    const mid = (lo + hi) >> 1;
    if (xs[mid] <= x) lo = mid;
    else hi = mid;
  }
  if (xs[lo] === x) return ys[lo];
  if (xs[hi] === x) return ys[hi];
  const a = ys[lo], b = ys[hi];
  if (a == null || b == null) return null;
  const t = (x - xs[lo]) / (xs[hi] - xs[lo]);
  return a + t * (b - a);
}

// Trailing rolling mean over a window of w points, nulls skipped.
export function rolling(ys, w) {
  const n = ys.length;
  const out = new Array(n);
  if (w <= 1) return ys.slice();
  let sum = 0, cnt = 0;
  const q = [];
  for (let i = 0; i < n; i++) {
    const v = ys[i];
    q.push(v);
    if (v != null) { sum += v; cnt++; }
    if (q.length > w) {
      const o = q.shift();
      if (o != null) { sum -= o; cnt--; }
    }
    out[i] = cnt ? sum / cnt : null;
  }
  return out;
}

// Last record at or before `samples` (matched-checkpoint lookup).
export function atSamples(recs, samples) {
  let lo = 0, hi = recs.length - 1, ans = -1;
  while (lo <= hi) {
    const mid = (lo + hi) >> 1;
    if (recs[mid].steps <= samples) { ans = mid; lo = mid + 1; } else hi = mid - 1;
  }
  return ans >= 0 ? recs[ans] : null;
}

export function fmt(v, digits = 4) {
  if (v == null || !Number.isFinite(v)) return "—";
  const a = Math.abs(v);
  if (a >= 1e9) return (v / 1e9).toFixed(2) + "G";
  if (a >= 1e6) return (v / 1e6).toFixed(2) + "M";
  if (a >= 1e4) return Math.round(v).toLocaleString();
  if (a >= 1) return v.toFixed(Math.max(0, digits - 1));
  if (a === 0) return "0";
  if (a < 1e-3) return v.toExponential(2);
  return v.toFixed(digits);
}

export function fmtDuration(s) {
  if (s == null || !Number.isFinite(s)) return "—";
  if (s < 90) return `${Math.round(s)} s`;
  if (s < 5400) return `${(s / 60).toFixed(1)} min`;
  if (s < 172800) return `${(s / 3600).toFixed(2)} h`;
  return `${(s / 86400).toFixed(1)} d`;
}

export function fmtSamples(n) {
  if (n == null) return "—";
  if (n >= 1e9) return (n / 1e9).toFixed(3) + "B";
  if (n >= 1e6) return (n / 1e6).toFixed(1) + "M";
  return n.toLocaleString();
}

export function parseSampleList(text) {
  const out = [];
  for (const tok of text.split(/[,\s]+/)) {
    if (!tok) continue;
    const m = tok.match(/^([\d.]+)\s*([kmgb]?)$/i);
    if (!m) continue;
    const mult = { "": 1, k: 1e3, m: 1e6, g: 1e9, b: 1e9 }[m[2].toLowerCase()];
    const v = parseFloat(m[1]) * mult;
    if (Number.isFinite(v)) out.push(Math.round(v));
  }
  return out.sort((a, b) => a - b);
}
