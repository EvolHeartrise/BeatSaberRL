// Archive access: index.json + per-run metrics.jsonl, fetched over plain
// HTTP (serve.py, or `python -m http.server` from the archive root when
// dist/ is copied there). No backend: everything is a static file.

export const store = $state({
  base: null, // URL prefix of the archive root ("./" or "/archive/")
  index: null, // parsed index.json
  error: null,
  metrics: {}, // run id -> {recs, at, bytes}
  loading: {}, // run id -> true while fetching
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

function bust(url) {
  return url + (url.includes("?") ? "&" : "?") + "t=" + Date.now();
}

// Change detection (2026-10-02, user request: an idle refresh reset the
// zoom). A refresh must not write to `store` unless the data changed:
// every write re-derives the panels and pokes the charts. Two layers:
//   1. conditional GET: serve.py answers If-None-Match with 304 when the
//      file's (mtime, size) ETag is unchanged, so there is no body to parse;
//   2. content compare: a 200 whose bytes (metrics.jsonl), or whose JSON
//      minus the `generated` stamp (index.json, which the backfill
//      rewrites every pass), match what we already hold is a no-op. This
//      layer alone also works on a plain `python -m http.server`.
// Unchanged run entries keep their object identity across index updates,
// so views on one run do not re-derive when another run changes.
const etags = new Map(); // url (no cache-buster) -> ETag of what we hold
const bodies = new Map(); // metrics url -> raw text parsed last
let indexSig = null; // JSON of the held index minus `generated`
const runSigs = new Map(); // run id -> JSON of its held index entry

async function fetchIfChanged(url, holding) {
  const headers = {};
  const tag = holding ? etags.get(url) : null;
  if (tag) headers["If-None-Match"] = tag;
  const r = await fetch(bust(url), { cache: "no-store", headers });
  if (r.status === 304) return null;
  if (!r.ok) throw new Error(`${r.status} ${r.statusText}`);
  const text = await r.text();
  const et = r.headers.get("ETag");
  if (et) etags.set(url, et);
  else etags.delete(url);
  return text;
}

// merge a freshly fetched index into store.index, writing only what changed
function applyIndex(idx) {
  const { generated, ...rest } = idx;
  const sig = JSON.stringify(rest);
  if (store.index && sig === indexSig) {
    if (store.index.generated !== generated) store.index.generated = generated;
    return;
  }
  indexSig = sig;
  const prev = new Map((store.index?.runs ?? []).map((r) => [r.id, r]));
  const runs = idx.runs.map((r) => {
    const rs = JSON.stringify(r);
    const old = prev.get(r.id);
    const same = old && runSigs.get(r.id) === rs;
    runSigs.set(r.id, rs);
    return same ? old : r;
  });
  if (!store.index) {
    store.index = { ...idx, runs };
    return;
  }
  for (const k of Object.keys(idx)) {
    if (k === "runs") continue;
    if (JSON.stringify(store.index[k]) !== JSON.stringify(idx[k])) store.index[k] = idx[k];
  }
  const cur = store.index.runs;
  if (cur.length !== runs.length || runs.some((r, i) => r !== cur[i])) store.index.runs = runs;
}

export async function loadIndex() {
  const bases = store.base ? [store.base] : CANDIDATES;
  let lastErr = null;
  for (const b of bases) {
    try {
      const text = await fetchIfChanged(b + "index.json", !!store.index && store.base === b);
      if (text != null) {
        const idx = JSON.parse(text);
        if (!idx || !Array.isArray(idx.runs)) throw new Error("not an index");
        if (store.base !== b) store.base = b;
        applyIndex(idx);
      }
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

export function parseJsonl(text) {
  const out = [];
  for (const line of text.split("\n")) {
    if (!line) continue;
    try {
      out.push(JSON.parse(line));
    } catch {
      /* partial last line of a live file */
    }
  }
  // a trainer --resume re-appends from the checkpoint: last record per
  // update wins, sorted by update index
  const byUpd = new Map();
  for (const r of out) byUpd.set(r.upd, r);
  return [...byUpd.values()].sort((a, b) => a.upd - b.upd);
}

export async function loadMetrics(run, { force = false } = {}) {
  if (!run) return null;
  const cur = store.metrics[run.id];
  if (cur && !force) return cur.recs;
  if (store.loading[run.id]) return cur?.recs ?? null;
  // flag only a first fetch: the flag is reactive, and a background
  // re-check has nothing to show for itself
  if (!cur) store.loading[run.id] = true;
  const url = store.base + run.metrics;
  try {
    let text;
    try {
      text = await fetchIfChanged(url, !!cur);
    } catch (e) {
      throw new Error(`${e.message} for ${run.metrics}`);
    }
    // 304, or the same bytes as last time: leave the store alone
    if (cur && (text == null || bodies.get(url) === text)) return cur.recs;
    const recs = parseJsonl(text);
    bodies.set(url, text);
    store.metrics[run.id] = { recs, at: Date.now(), bytes: text.length };
    return recs;
  } catch (e) {
    store.error = `metrics for ${run.id}: ${e.message}`;
    return cur?.recs ?? null;
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
