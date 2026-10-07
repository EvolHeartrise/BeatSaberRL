// Incremental metrics parsing (2026-10-06) == one full parse.
// The fetch worker reads a growing metrics.jsonl in byte chunks (HTTP
// Range from where the last read stopped; rsync --append-verify can leave
// a line cut mid-write) and keeps columns; the page derives the chart
// series from those columns. This checks, without a browser:
//   - JsonlReader fed in random chunks (cuts mid-line, mid-UTF-8 char)
//     ends with the very same columns as one feed of the whole file;
//   - those columns hold exactly the records of the original full parse
//     (parseJsonl: bad lines skipped, last record per update wins after a
//     --resume re-append, rows sorted by update);
//   - every chart series (series.js over RunData) equals the original
//     record-based derivation (copied below), incl. the swing_hist merge;
//   - a truncated last line counts only once it is complete, and a
//     complete unterminated last line counts as in a full parse.
// Also runs over the real archive's biggest live metrics file when one is
// there (read-only). Run: node test/incremental.mjs (npm test runs it).
import { readFileSync, existsSync, readdirSync, statSync } from "node:fs";
import { homedir } from "node:os";
import { resolve } from "node:path";
import { JsonlReader, MetricsLog, parseJsonl, RunData, BEST_KEYS } from "../src/lib/metricsCore.js";
import { xValues, seriesValues, hasAny, X_MODES, PANELS, COMPARE_KEYS } from "../src/lib/series.js";

let failures = 0, checks = 0;
function check(ok, msg) {
  checks++;
  if (!ok) { failures++; console.log(`FAIL ${msg}`); }
}
const enc = new TextEncoder();

// ------------------------------------------- the original record-based code
function refX(recs, mode) {
  const n = recs.length, x = new Array(n);
  if (mode === "samples") for (let i = 0; i < n; i++) x[i] = recs[i].steps;
  else if (mode === "updates") for (let i = 0; i < n; i++) x[i] = recs[i].upd;
  else {
    const allT = n > 0 && recs.every((r) => typeof r.t === "number");
    if (allT) { const t0 = recs[0].t; for (let i = 0; i < n; i++) x[i] = (recs[i].t - t0) / 3600; }
    else {
      let off = 0, prev = 0;
      for (let i = 0; i < n; i++) {
        const e = recs[i].elapsed ?? prev;
        if (e < prev) off += prev;
        prev = e;
        x[i] = (off + e) / 3600;
      }
    }
  }
  return x;
}
function per1M(r, k, batch) { const v = r[k]; if (v == null || !batch) return null; return (v * 1e6) / batch; }
function batchOf(recs, i) {
  if (i > 0) { const d = recs[i].steps - recs[i - 1].steps; if (d > 0) return d; }
  if (recs.length > 1 && i === 0) { const d = recs[1].steps - recs[0].steps; if (d > 0) return d; }
  return recs[i].steps / Math.max(1, recs[i].upd);
}
const DERIVED = {
  falls_per_1m: (r, i, recs) => per1M(r, "falls", batchOf(recs, i)),
  tilts_per_1m: (r, i, recs) => per1M(r, "tilts", batchOf(recs, i)),
  lifts_per_1m: (r, i, recs) => per1M(r, "lifts", batchOf(recs, i)),
  sps_k: (r) => (r.sps == null ? null : r.sps / 1000),
  kl_first_pos: (r) => (r.kl_first > 0 ? r.kl_first : null),
  kl_last_pos: (r) => (r.kl_last > 0 ? r.kl_last : null),
  falls_per_1m_pos: (r, i, recs) => { const v = per1M(r, "falls", batchOf(recs, i)); return v > 0 ? v : null; },
  tilts_per_1m_pos: (r, i, recs) => { const v = per1M(r, "tilts", batchOf(recs, i)); return v > 0 ? v : null; },
  contact_pct: (r) => (r.contact_frac == null ? null : r.contact_frac * 100),
};
function refY(recs, key) {
  const f = DERIVED[key];
  return recs.map((r, i) => { const v = f ? f(r, i, recs) : r[key]; return typeof v === "number" && Number.isFinite(v) ? v : null; });
}
function refMergeBest(recs, hist) {
  if (!hist || !hist.upd?.length) return recs;
  let j = 0, last = -1;
  return recs.map((r) => {
    while (j < hist.upd.length && hist.upd[j] <= r.upd) last = j++;
    if (last < 0) return r;
    const o = { ...r };
    for (const k of BEST_KEYS) o["best_" + k] = hist[k][last];
    return o;
  });
}

// ------------------------------------------------------------- comparisons
const same = (a, b) => (Number.isNaN(a) && Number.isNaN(b)) || Object.is(a, b) || a === b;
function sameArr(a, b) {
  if (a.length !== b.length) return false;
  for (let i = 0; i < a.length; i++) if (!same(a[i], b[i])) return false;
  return true;
}
function sameSnap(a, b) {
  if (a.n !== b.n) return "n " + a.n + " vs " + b.n;
  const ka = Object.keys(a.cols).sort(), kb = Object.keys(b.cols).sort();
  if (ka.join() !== kb.join()) return "column keys differ";
  for (const k of ka) if (!sameArr(a.cols[k], b.cols[k])) return "column " + k;
  const sa = Object.keys(a.strs).sort(), sb = Object.keys(b.strs).sort();
  if (sa.join() !== sb.join()) return "string keys differ";
  for (const k of sa) if (!sameArr(a.strs[k], b.strs[k])) return "strings " + k;
  return null;
}
const KEYS = [...new Set([...PANELS.flatMap((p) => p.series), ...COMPARE_KEYS, "upd", "steps", "t", "elapsed"])];

// the columns hold exactly the full parse's records, and every series
// derived from them equals the record-based derivation
function againstFullParse(text, hist, what) {
  const recs = refMergeBest(parseJsonl(text), hist);
  const rd = new JsonlReader();
  rd.feed(enc.encode(text));
  const snap = rd.log.snapshot(hist);
  const data = new RunData(snap);
  check(data.n === recs.length, `${what}: ${data.n} rows == full parse ${recs.length}`);
  let bad = null;
  for (let i = 0; i < recs.length && !bad; i++) {
    const r = data.rec(i), o = recs[i];
    for (const k of Object.keys(o)) {
      const v = o[k];
      const keep = typeof v === "number" || typeof v === "string" || typeof v === "boolean";
      if (keep && !(k in r) && !(typeof v === "number" && !Number.isFinite(v))) bad = `row ${i} lacks ${k}`;
      else if (keep && k in r && !same(r[k], v)) bad = `row ${i} ${k}: ${r[k]} vs ${v}`;
    }
    for (const k of Object.keys(r)) if (!(k in o)) bad = `row ${i} has extra ${k}`;
  }
  check(!bad, `${what}: rows equal the full parse's records${bad ? " — " + bad : ""}`);
  for (const m of X_MODES) check(sameArr(xValues(data, m.key), refX(recs, m.key)), `${what}: x (${m.key}) equal`);
  const off = KEYS.filter((k) => !sameArr(seriesValues(data, k), refY(recs, k)) || hasAny(data, k) !== refY(recs, k).some((v) => v != null));
  check(off.length === 0, `${what}: ${KEYS.length} series equal the record-based ones${off.length ? " — differ: " + off.join(",") : ""}`);
}

// incremental: chunks cut at random byte positions (the file growing under
// Range reads) vs one feed of the whole file
function chunked(bytes, rng, hist) {
  const rd = new JsonlReader();
  let end = 0, changes = 0;
  while (end < bytes.length) {
    end = Math.min(bytes.length, end + 1 + Math.floor(rng() * rng() * 4000));
    // the worker snapshots after every read that changed something
    if (rd.feed(bytes.subarray(rd.offset, end))) { changes++; rd.log.snapshot(hist); }
    // a re-read that brought nothing new must not count as a change
    if (rd.feed(bytes.subarray(rd.offset, end))) changes = -1e9;
  }
  return { snap: rd.log.snapshot(hist), changes };
}
function rngOf(seed) {
  let s = seed >>> 0;
  return () => ((s = Math.imul(s ^ (s >>> 15), 0x2c1b3c6d) + 0x6d2b79f5 >>> 0) / 4294967296);
}

// ---------------------------------------------------------------- fixtures
const BATCH = 1 << 20;
function rec(i, extra = {}) {
  const r = {
    upd: i, n_upd: 500, steps: i * BATCH, t: 1790000000 + i * 7.25, elapsed: i * 7.25, sps: 150000 + 1000 * Math.sin(i),
    acc: 0.5 + i / 1000, accL: 0.48 + i / 1000, return: -0.1 + i / 3000, ent: 15 - i / 100, ev: 0.9, kl_first: i % 5 ? 1e-6 : 0,
    kl_last: 0.04, clipfrac_last: 0.25, gnorm: 2.8, falls: i % 4, tilts: i % 3, lifts: i % 7, contact_frac: 0.01 * (i % 9),
    diag_groups: { a: 1, b: [1, 2] }, ...extra,
  };
  if (i % 50 === 0) r.ckpt = `trunk_upd${String(i).padStart(5, "0")}.pt`;
  if (i % 13 === 0) delete r.accL;
  if (i % 17 === 0) r.note = "é ✓ 長"; // multi-byte UTF-8: chunk cuts land inside it
  if (i % 19 === 0) r.flag = i % 2 === 0;
  return r;
}
const line = (r) => JSON.stringify(r) + "\n";
let synth = "";
for (let i = 1; i <= 300; i++) synth += line(rec(i));
synth += "this is not json\n";
// a --resume from update 250: 251.. re-appended with new values (last wins)
for (let i = 251; i <= 420; i++) synth += line(rec(i, { acc: 0.9 - i / 5000, resumed: 1 }));
synth += line({ upd: 100, steps: 100 * BATCH, acc: 0.123 }); // an out-of-order replacement
synth += "\n\n";
synth += line(rec(421)).slice(0, 37) + "\n"; // a garbled cut line, then the full one
synth += line(rec(421));
synth += line(rec(421, { acc: 0.777 })); // the last update re-written (a resume from it)
const hist = { upd: [40, 41, 90, 300, 400], ss_wpp: [1, 2, 3, null, 5], ss_upp: [10, 20, 30, 40, 50], bl_wpp: [1, 1, 1, 1, 1], bl_upp: [2, 2, 2, 2, 2] };

const fixtures = [["synthetic", synth, null], ["synthetic + swing_hist", synth, hist]];
// the real archive's biggest metrics file, when there is one (read-only)
const ARCH = resolve(homedir(), "beat-arena-volume-backup");
if (existsSync(resolve(ARCH, "index.json"))) {
  try {
    const idx = JSON.parse(readFileSync(resolve(ARCH, "index.json"), "utf8"));
    const live = idx.runs.filter((r) => r.metrics && existsSync(resolve(ARCH, r.metrics)))
      .sort((a, b) => statSync(resolve(ARCH, b.metrics)).size - statSync(resolve(ARCH, a.metrics)).size)[0];
    if (live) {
      const h = live.swing?.hist && existsSync(resolve(ARCH, live.swing.hist)) ? JSON.parse(readFileSync(resolve(ARCH, live.swing.hist), "utf8")) : null;
      fixtures.push([`archive ${live.id}`, readFileSync(resolve(ARCH, live.metrics), "utf8"), h]);
    }
  } catch (e) {
    console.log(`(archive fixture skipped: ${e.message})`);
  }
}

for (const [what, text, h] of fixtures) {
  againstFullParse(text, h, what);
  const bytes = enc.encode(text);
  const whole = new JsonlReader();
  whole.feed(bytes);
  const ref = whole.log.snapshot(h);
  for (let seed = 1; seed <= (what.startsWith("archive") ? 3 : 25); seed++) {
    const { snap, changes } = chunked(bytes, rngOf(seed * 7919), h);
    const d = sameSnap(snap, ref);
    check(!d && changes >= 0, `${what}: chunked feed #${seed} == one full feed${d ? " — " + d : ""}${changes < 0 ? " — a no-op re-read reported a change" : ""}`);
  }
}

// a line cut mid-write: not counted until complete; then exactly once
{
  const full = line(rec(1)) + line(rec(2));
  const cut = full.length - 25;
  const rd = new JsonlReader();
  rd.feed(enc.encode(full.slice(0, cut)));
  check(rd.log.snapshot().n === 1 && rd.offset === enc.encode(line(rec(1))).length, "cut line: only the complete line counted, offset at its end");
  check(rd.feed(enc.encode(full).subarray(rd.offset)) === true, "cut line completed: reported as a change");
  const s = rd.log.snapshot();
  check(s.n === 2 && s.cols.acc[1] === rec(2).acc, "cut line completed: row 2 present with its values");
  // a complete last line without "\n" counts (as in a full parse), and its
  // later "\n" is not a second change
  const rd2 = new JsonlReader();
  const t2 = line(rec(1)) + JSON.stringify(rec(2));
  rd2.feed(enc.encode(t2));
  check(rd2.log.snapshot().n === parseJsonl(t2).length && rd2.log.snapshot().n === 2, "unterminated complete last line counted like a full parse");
  check(rd2.feed(enc.encode(t2 + "\n").subarray(rd2.offset)) === false, "its newline arriving is not a change");
}
// the hash guard: a full re-read that extends what was consumed
{
  const rd = new JsonlReader();
  const a = enc.encode(line(rec(1)) + line(rec(2)));
  rd.feed(a);
  const b = enc.encode(line(rec(1)) + line(rec(2)) + line(rec(3)));
  check(rd.extends(b), "hash guard: an appended file extends the consumed prefix");
  const c = enc.encode(line(rec(1)) + line(rec(9)) + line(rec(3)));
  check(!rd.extends(c), "hash guard: a rewritten prefix is detected");
}
// MetricsLog: an identical re-read line is not a change
{
  const m = new MetricsLog();
  check(m.addLine(JSON.stringify(rec(5))) && !m.addLine(JSON.stringify(rec(5))), "same line twice: one change");
}

console.log(failures ? `\n${failures} of ${checks} check(s) FAILED` : `\nincremental: all ${checks} checks passed`);
process.exit(failures ? 1 : 0);
