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

function bust(url) {
  return url + (url.includes("?") ? "&" : "?") + "t=" + Date.now();
}

export async function loadIndex() {
  const bases = store.base ? [store.base] : CANDIDATES;
  let lastErr = null;
  for (const b of bases) {
    try {
      const r = await fetch(bust(b + "index.json"), { cache: "no-store" });
      if (!r.ok) throw new Error(`${r.status} ${r.statusText}`);
      const idx = await r.json();
      if (!idx || !Array.isArray(idx.runs)) throw new Error("not an index");
      store.base = b;
      store.index = idx;
      store.error = null;
      return idx;
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
  store.loading[run.id] = true;
  try {
    const r = await fetch(bust(store.base + run.metrics), { cache: "no-store" });
    if (!r.ok) throw new Error(`${r.status} for ${run.metrics}`);
    const text = await r.text();
    const recs = parseJsonl(text);
    store.metrics[run.id] = { recs, at: Date.now(), bytes: text.length };
    return recs;
  } catch (e) {
    store.error = `metrics for ${run.id}: ${e.message}`;
    return cur?.recs ?? null;
  } finally {
    delete store.loading[run.id];
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
