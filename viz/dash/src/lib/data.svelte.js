// Archive access: index.json + per-run metrics.jsonl, fetched over plain
// HTTP (serve.py, or `python -m http.server` from the archive root when
// dist/ is copied there). No backend: everything is a static file.

// One run's metrics as columns: RunData (metricsCore.js), a class
// instance, so the reactive store holds it as is.
import { RunData } from "./metricsCore.js";
export { RunData };

export const store = $state({
  base: null, // URL prefix of the archive root ("./" or "/archive/")
  index: null, // parsed index.json
  error: null,
  metrics: {}, // run id -> RunData (columns; replaced whole when the file changed)
  loading: {}, // run id -> true while fetching
  swingVer: {}, // run id -> bumped when that run's swing.json changed
});

const CANDIDATES = (() => {
  const q = new URLSearchParams(location.search).get("archive");
  const c = [];
  if (q) c.push(q.endsWith("/") ? q : q + "/");
  c.push("./", "/archive/");
  return c;
})();

// Auto-refresh period: 15 s; `?refresh=<seconds>` overrides (the
// headless refresh test runs at 1 s).
export const REFRESH_MS = (() => {
  const v = Number(new URLSearchParams(location.search).get("refresh"));
  return v > 0 ? v * 1000 : 15000;
})();

// All fetching and parsing runs in a worker (fetch.worker.js, 2026-10-06:
// each auto-refresh used to freeze the page — whole metrics files parsed
// on the main thread into deeply reactive records, every panel re-derived
// through the proxies). The page keeps only what the worker says changed.
//
// Change detection (2026-10-02, user request: an idle refresh reset the
// zoom). A refresh must not write to `store` unless the data changed:
// every write re-derives the panels and pokes the charts. The worker
// answers "same" for a 304 (serve.py's mtime+size ETag), for identical
// bytes (index.json compared minus its `generated` stamp, which the
// backfill bumps every pass) and for a growing file whose new bytes hold
// no complete new line. Unchanged run entries keep their object identity
// across index updates, so views on one run do not re-derive when another
// run changes.
let worker = null;
let seq = 0;
const pending = new Map();
function call(op, args) {
  if (!worker) {
    worker = new Worker(new URL("./fetch.worker.js", import.meta.url), { type: "module" });
    worker.onmessage = (e) => {
      const { id, ok, res, err } = e.data;
      const p = pending.get(id);
      if (!p) return;
      pending.delete(id);
      if (ok) p.resolve(res);
      else p.reject(new Error(err));
    };
  }
  return new Promise((resolve, reject) => {
    const id = ++seq;
    pending.set(id, { resolve, reject });
    worker.postMessage({ id, op, args });
  });
}
// test hook: the worker's fetch counters (full / ranged / 304 reads, bytes)
globalThis.__dashStats = () => call("stats", {});

const abs = (rel) => new URL(rel, location.href).href;

// apply the worker's index diff to store.index, writing only what changed
function applyIndex(d) {
  if (d.same) {
    if (store.index && d.generated != null && store.index.generated !== d.generated) store.index.generated = d.generated;
    return;
  }
  const prev = new Map((store.index?.runs ?? []).map((r) => [r.id, r]));
  const runs = d.order.map((id) => d.entries[id] ?? prev.get(id));
  if (!store.index) {
    store.index = { ...d.top, generated: d.generated, runs };
    return;
  }
  for (const k of Object.keys(d.top)) store.index[k] = d.top[k];
  if (store.index.generated !== d.generated) store.index.generated = d.generated;
  const cur = store.index.runs;
  if (cur.length !== runs.length || runs.some((r, i) => r !== cur[i])) store.index.runs = runs;
}

export async function loadIndex() {
  const bases = store.base ? [store.base] : CANDIDATES;
  let lastErr = null;
  for (const b of bases) {
    try {
      const d = await call("index", { url: abs(b + "index.json"), holding: !!store.index && store.base === b });
      if (store.base !== b) store.base = b;
      applyIndex(d);
      if (store.error) store.error = null;
      return store.index;
    } catch (e) {
      lastErr = e;
    }
  }
  store.error = `index.json not reachable (${lastErr?.message ?? "?"}) — run metrics_backfill.py and serve the archive (see README)`;
  return null;
}

export function runById(id) {
  return store.index?.runs.find((r) => r.id === id) ?? null;
}

export async function loadMetrics(run, { force = false } = {}) {
  if (!run) return null;
  const cur = store.metrics[run.id];
  if (cur && !force) return cur;
  if (store.loading[run.id]) return cur ?? null;
  // flag only a first fetch: the flag is reactive, and a background
  // re-check has nothing to show for itself
  if (!cur) store.loading[run.id] = true;
  try {
    let d;
    try {
      d = await call("metrics", {
        key: run.id, url: abs(store.base + run.metrics),
        hurl: run.swing?.hist ? abs(store.base + run.swing.hist) : null, holding: !!cur,
      });
    } catch (e) {
      throw new Error(`${e.message} for ${run.metrics}`);
    }
    if (d.same) return cur;
    const data = new RunData(d.data);
    store.metrics[run.id] = data;
    return data;
  } catch (e) {
    store.error = `metrics for ${run.id}: ${e.message}`;
    return cur ?? null;
  } finally {
    if (store.loading[run.id]) delete store.loading[run.id];
  }
}

export function mostRecentRun() {
  const runs = store.index?.runs ?? [];
  let best = null;
  for (const r of runs) if (!best || (r.t_last ?? 0) > (best.t_last ?? 0)) best = r;
  return best;
}

export function archiveUrl(rel) {
  if (!rel) return null;
  if (rel.startsWith("/")) return null; // absolute repo path, not served
  return store.base + rel;
}

// Per-level swing summaries (swing.json, python/swing_summary.py): a few MB
// of columnar arrays per run, so they live OUTSIDE the reactive store (a
// $state proxy over 7k-element arrays costs on every read); the store only
// carries a version number per run that views depend on. Fetched and
// parsed by the worker; a 304 or identical bytes leave the version alone,
// so an idle refresh re-derives nothing (table sort / scroll stay).
const swingData = new Map(); // run id -> parsed swing.json
const swingBusy = new Set();

export function swingOf(id) {
  return id != null && store.swingVer[id] != null ? swingData.get(id) ?? null : null;
}

export async function loadSwing(run, { force = false } = {}) {
  if (!run?.swing?.path) return null;
  const cur = swingData.get(run.id);
  if (cur && !force) return cur;
  if (swingBusy.has(run.id)) return cur ?? null;
  swingBusy.add(run.id);
  try {
    const d = await call("swing", { url: abs(store.base + run.swing.path), holding: !!cur });
    if (d.same) return cur;
    swingData.set(run.id, d.data);
    store.swingVer[run.id] = (store.swingVer[run.id] ?? 0) + 1;
    return d.data;
  } catch (e) {
    store.error = `swing summary for ${run.id}: ${e.message}`;
    return cur ?? null;
  } finally {
    swingBusy.delete(run.id);
  }
}
