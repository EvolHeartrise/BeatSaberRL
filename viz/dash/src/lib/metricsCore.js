// metrics.jsonl -> columns, incrementally (2026-10-06, user: every
// auto-refresh froze the page). Pure module, no DOM: the fetch worker
// (fetch.worker.js) runs it off the main thread, and test/incremental.mjs
// checks that feeding a file in arbitrary byte chunks (Range reads of a
// growing file, a line cut mid-write by rsync --append-verify) gives the
// very same columns as one full parse.
//
// Semantics are those of the original full parse (parseJsonl below):
// invalid lines are skipped (a partial last line of a live file), a
// trainer --resume re-appends from its checkpoint so the LAST record per
// update wins, rows sorted by update index. Columns: one Float64Array per
// numeric key (NaN = absent), one plain array per string / boolean key
// (null = absent); objects and lists (diag_groups, ...) are not kept.

// the reference: whole text -> records (what data.svelte.js did before)
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
  const byUpd = new Map();
  for (const r of out) byUpd.set(r.upd, r);
  return [...byUpd.values()].sort((a, b) => a.upd - b.upd);
}

const byUpdOrder = (a, b) => a.rec.upd - b.rec.upd;

export class MetricsLog {
  constructor() {
    this.byUpd = new Map(); // upd -> {line, rec}; insertion order = first occurrence
    this.n = 0;
    this.cap = 0;
    this.lastUpd = -Infinity;
    this.dirty = false; // a replaced or out-of-order row: rebuild before reading
    this.cols = new Map(); // key -> Float64Array(cap), NaN = absent
    this.strs = new Map(); // key -> Array(cap), null = absent
  }

  // one line of the file; true when it changed the data
  addLine(line) {
    if (!line) return false;
    let rec;
    try {
      rec = JSON.parse(line);
    } catch {
      return false;
    }
    if (rec === null || typeof rec !== "object") rec = {};
    const upd = rec.upd;
    const old = this.byUpd.get(upd);
    if (old && old.line === line) return false; // same record again (a re-read last line)
    this.byUpd.set(upd, { line, rec });
    if (!old && !this.dirty && (this.n === 0 || upd > this.lastUpd)) {
      this.grow(this.n + 1);
      this.write(this.n, rec);
      this.n++;
      this.lastUpd = upd;
    } else {
      this.dirty = true;
    }
    return true;
  }

  addText(text) {
    let changed = false;
    let i = 0;
    while (i <= text.length) {
      let j = text.indexOf("\n", i);
      if (j < 0) j = text.length;
      if (this.addLine(text.slice(i, j))) changed = true;
      i = j + 1;
    }
    return changed;
  }

  grow(need) {
    if (need <= this.cap) return;
    let cap = Math.max(256, this.cap);
    while (cap < need) cap *= 2;
    for (const [k, c] of this.cols) {
      const d = new Float64Array(cap).fill(NaN);
      d.set(c.subarray(0, this.n));
      this.cols.set(k, d);
    }
    for (const s of this.strs.values()) s.length = cap;
    this.cap = cap;
  }

  write(i, rec) {
    for (const k in rec) {
      const v = rec[k];
      if (typeof v === "number") {
        let c = this.cols.get(k);
        if (!c) this.cols.set(k, (c = new Float64Array(this.cap).fill(NaN)));
        c[i] = v;
      } else if (typeof v === "string" || typeof v === "boolean") {
        let s = this.strs.get(k);
        if (!s) this.strs.set(k, (s = new Array(this.cap).fill(null)));
        s[i] = v;
      }
    }
  }

  rebuild() {
    if (!this.dirty) return;
    const rows = [...this.byUpd.values()].sort(byUpdOrder);
    this.grow(rows.length);
    for (const c of this.cols.values()) c.fill(NaN);
    for (const s of this.strs.values()) s.fill(null);
    for (let i = 0; i < rows.length; i++) this.write(i, rows[i].rec);
    this.n = rows.length;
    this.lastUpd = rows.length ? rows[rows.length - 1].rec.upd : -Infinity;
    this.dirty = false;
  }

  // copies sized to n (the worker transfers them to the page); `hist`
  // (swing_hist.json) adds the best-possible curves as step series:
  // best_<k> = the curve's last value at or before the row's update
  snapshot(hist = null) {
    this.rebuild();
    const n = this.n;
    const cols = {}, strs = {}, transfer = [];
    for (const [k, c] of this.cols) {
      const a = c.slice(0, n);
      cols[k] = a;
      transfer.push(a.buffer);
    }
    for (const [k, s] of this.strs) strs[k] = s.slice(0, n);
    if (hist && hist.upd?.length && n) {
      const upd = this.cols.get("upd");
      const out = BEST_KEYS.map(() => new Float64Array(n).fill(NaN));
      let j = 0, last = -1;
      for (let i = 0; i < n; i++) {
        const u = upd ? upd[i] : NaN;
        while (j < hist.upd.length && hist.upd[j] <= u) last = j++;
        if (last < 0) continue;
        BEST_KEYS.forEach((k, q) => {
          const v = hist[k]?.[last];
          out[q][i] = typeof v === "number" ? v : NaN;
        });
      }
      BEST_KEYS.forEach((k, q) => {
        cols["best_" + k] = out[q];
        transfer.push(out[q].buffer);
        delete strs["best_" + k];
      });
    }
    return { n, cols, strs, transfer };
  }
}

export const BEST_KEYS = ["ss_wpp", "ss_upp", "bl_wpp", "bl_upp"];

// Byte-level reader of a growing file: consumes only complete lines
// (through the last "\n"); an unterminated tail is offered to the log as
// a provisional line (it counts only when it is already valid JSON, as in
// a full parse) and re-read from the same offset next time. `tail` keeps
// the last bytes consumed, so a Range read can check the file was not
// rewritten underneath; `hash` (FNV-1a over all consumed bytes) does the
// same for a full re-read.
const TAIL = 256;
const dec = new TextDecoder();

export class JsonlReader {
  constructor() {
    this.log = new MetricsLog();
    this.offset = 0;
    this.tail = new Uint8Array(0);
    this.hash = 0x811c9dc5;
    this.pending = null; // the provisional line last offered
  }

  // bytes of the file starting at this.offset; true when the data changed
  feed(bytes) {
    let changed = false;
    const nl = bytes.lastIndexOf(10);
    if (nl >= 0) {
      const done = bytes.subarray(0, nl + 1);
      if (this.log.addText(dec.decode(done))) changed = true;
      this.hash = fnv(this.hash, done);
      this.offset += done.length;
      const t = new Uint8Array(Math.min(TAIL, this.tail.length + done.length));
      const fromDone = Math.min(done.length, t.length);
      const fromOld = t.length - fromDone;
      t.set(this.tail.subarray(this.tail.length - fromOld), 0);
      t.set(done.subarray(done.length - fromDone), fromOld);
      this.tail = t;
      this.pending = null;
    }
    const rest = bytes.subarray(nl + 1);
    if (rest.length) {
      const line = dec.decode(rest);
      if (line !== this.pending) {
        this.pending = line;
        if (this.log.addLine(line)) changed = true;
      }
    }
    return changed;
  }

  // a full read of the file: does it extend what was consumed?
  extends(bytes) {
    return bytes.length >= this.offset && fnv(0x811c9dc5, bytes.subarray(0, this.offset)) === this.hash;
  }
}

export function fnv(h, bytes) {
  for (let i = 0; i < bytes.length; i++) {
    h ^= bytes[i];
    h = Math.imul(h, 0x01000193) >>> 0;
  }
  return h;
}

export function sameBytes(a, b) {
  if (a.length !== b.length) return false;
  for (let i = 0; i < a.length; i++) if (a[i] !== b[i]) return false;
  return true;
}

// One run's metrics as columns (metricsCore.js): a class instance, so the
// reactive store holds it as is (Svelte proxies plain objects only) —
// reads cost nothing and a refresh swaps one reference. `cols[key]` is a
// Float64Array (NaN = absent), `strs[key]` a plain array (null = absent).
// The best-possible curves over training (swing_hist.json, written by
// python/swing_summary.py next to swing.json) are merged in by the worker
// as step series best_ss_wpp, best_ss_upp, best_bl_wpp, best_bl_upp.
export class RunData {
  constructor({ n, cols, strs }) {
    this.n = n;
    this.cols = cols;
    this.strs = strs;
  }
  val(i, k) {
    const c = this.cols[k];
    if (c) {
      const v = c[i];
      if (!Number.isNaN(v)) return v;
    }
    const s = this.strs[k];
    return s ? s[i] ?? null : null;
  }
  // row i as a plain record (status line, matched-samples table)
  rec(i) {
    if (!(i >= 0 && i < this.n)) return null;
    const o = {};
    for (const k in this.strs) if (this.strs[k][i] != null) o[k] = this.strs[k][i];
    for (const k in this.cols) {
      const v = this.cols[k][i];
      if (!Number.isNaN(v)) o[k] = v;
    }
    return o;
  }
  get last() {
    return this.rec(this.n - 1);
  }
}
