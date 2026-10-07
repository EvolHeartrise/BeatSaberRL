// Fetch + parse off the main thread (2026-10-06, user: each auto-refresh
// froze the page: hover, zoom and drag blocked for 0.4 s on Live and for
// seconds on Compare). The page asks; this worker fetches, detects change,
// parses and answers with only what changed:
//   index    index.json -> the run entries that changed (a whole index is
//            1.2 MB; a refresh usually changes one or two entries)
//   metrics  metrics.jsonl -> columns (metricsCore.js), transferred as
//            Float64Arrays. A growing file is read from where the last
//            read stopped (HTTP Range, serve.py), only the new lines are
//            parsed; a server without Range support sends the whole file
//            and the consumed prefix is checked by hash instead.
//            swing_hist.json (the best-possible curves) merges in here.
//   swing    swing.json, parsed here; posted only when its bytes changed.
// Change detection as before: If-None-Match -> 304, then content compare;
// "same" answers let the page leave its store alone.
import { JsonlReader, fnv, sameBytes } from "./metricsCore.js";

const bust = (url) => url + (url.includes("?") ? "&" : "?") + "t=" + Date.now();
const etags = new Map(); // url -> ETag of what we hold
const stats = { full: 0, ranged: 0, notModified: 0, bytes: 0 };

async function get(url, { holding = false, range = null } = {}) {
  const headers = {};
  const tag = holding ? etags.get(url) : null;
  if (tag) headers["If-None-Match"] = tag;
  if (range != null) headers["Range"] = `bytes=${range}-`;
  const r = await fetch(bust(url), { cache: "no-store", headers });
  if (r.status === 304) {
    stats.notModified++;
    return { status: 304 };
  }
  if (r.status === 416) return { status: 416 };
  if (!r.ok) throw new Error(`${r.status} ${r.statusText}`);
  const bytes = new Uint8Array(await r.arrayBuffer());
  stats.bytes += bytes.length;
  if (r.status === 206) stats.ranged++;
  else stats.full++;
  const et = r.headers.get("ETag");
  if (et) etags.set(url, et);
  else etags.delete(url);
  return { status: r.status, bytes };
}

// ------------------------------------------------------------------ index
let indexSig = null; // JSON of the held index minus `generated`
let runSigs = new Map(); // run id -> JSON of its entry as last sent
let topSigs = new Map(); // top-level key -> JSON as last sent

async function index({ url, holding }) {
  if (!holding) {
    indexSig = null;
    runSigs = new Map();
    topSigs = new Map();
  }
  const g = await get(url, { holding });
  if (g.status === 304) return { same: true };
  const text = new TextDecoder().decode(g.bytes);
  const idx = JSON.parse(text);
  if (!idx || !Array.isArray(idx.runs)) throw new Error("not an index");
  const { generated, ...rest } = idx;
  const sig = JSON.stringify(rest);
  if (holding && sig === indexSig) return { same: true, generated };
  indexSig = sig;
  const top = {};
  for (const k of Object.keys(idx)) {
    if (k === "runs") continue;
    const s = JSON.stringify(idx[k]);
    if (topSigs.get(k) !== s) {
      top[k] = idx[k];
      topSigs.set(k, s);
    }
  }
  const entries = {};
  const seen = new Map();
  for (const r of idx.runs) {
    const s = JSON.stringify(r);
    seen.set(r.id, s);
    if (runSigs.get(r.id) !== s) entries[r.id] = r;
  }
  runSigs = seen;
  return { same: false, generated, top, order: idx.runs.map((r) => r.id), entries };
}

// ---------------------------------------------------------------- metrics
const runs = new Map(); // run id -> {reader, hist, histHash, chain}

async function pullMetrics(st, url) {
  const rd = st.reader;
  if (rd.offset > 0) {
    // re-read the last few consumed bytes too: they must still be there
    const k = rd.tail.length;
    const g = await get(url, { holding: true, range: rd.offset - k });
    if (g.status === 304) return false;
    if (g.status === 206) {
      if (g.bytes.length >= k && sameBytes(g.bytes.subarray(0, k), rd.tail)) return rd.feed(g.bytes.subarray(k));
    } else if (g.status === 200) {
      // no Range support: the whole file; parse only past what we hold
      if (rd.extends(g.bytes)) return rd.feed(g.bytes.subarray(rd.offset));
      st.reader = new JsonlReader();
      st.reader.feed(g.bytes);
      return true;
    }
    // 416 (the file shrank) or a different prefix: it was rewritten
  }
  const g = await get(url);
  st.reader = new JsonlReader();
  st.reader.feed(g.bytes);
  return true;
}

async function pullHist(st, hurl) {
  if (!hurl) return false;
  let g;
  try {
    g = await get(hurl, { holding: st.hist != null });
  } catch {
    return false; // the curve is optional: a missing file shows no panel
  }
  if (g.status !== 200) return false;
  const h = fnv(0x811c9dc5, g.bytes);
  if (st.hist && h === st.histHash) return false;
  try {
    st.hist = JSON.parse(new TextDecoder().decode(g.bytes));
    st.histHash = h;
    return true;
  } catch {
    return false; // a half-written file: keep the last curve
  }
}

async function metrics({ key, url, hurl, holding }) {
  let st = runs.get(key);
  if (!st || !holding) {
    st = { reader: new JsonlReader(), hist: null, histHash: 0, chain: Promise.resolve() };
    runs.set(key, st);
  }
  // one pull at a time per run
  const job = st.chain.then(async () => {
    const m = await pullMetrics(st, url);
    const h = await pullHist(st, hurl);
    if (holding && !m && !h) return [{ same: true }];
    const snap = st.reader.log.snapshot(st.hist);
    const { transfer, ...data } = snap;
    return [{ same: false, data }, transfer];
  });
  st.chain = job.catch(() => {});
  return job;
}

// ------------------------------------------------------------------ swing
const swingHash = new Map(); // url -> hash of the bytes held by the page

async function swing({ url, holding }) {
  const g = await get(url, { holding });
  if (g.status === 304) return { same: true };
  const h = fnv(0x811c9dc5, g.bytes);
  if (holding && swingHash.get(url) === h) return { same: true };
  const data = JSON.parse(new TextDecoder().decode(g.bytes));
  swingHash.set(url, h);
  return { same: false, data };
}

const OPS = { index, metrics, swing, stats: async () => ({ ...stats }) };

self.onmessage = async (e) => {
  const { id, op, args } = e.data;
  try {
    const out = await OPS[op](args);
    const [res, transfer] = Array.isArray(out) ? out : [out, []];
    self.postMessage({ id, ok: true, res }, transfer ?? []);
  } catch (err) {
    self.postMessage({ id, ok: false, err: String(err?.message ?? err) });
  }
};
