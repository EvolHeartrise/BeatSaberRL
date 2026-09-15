// Map metadata for the viewer's top-left card: cover art, song/sub name,
// song author, mapper and the BeatSaver key. None of that is in a replay —
// the header only carries `map = {hash, difficulty, song, audio_offset_s}`
// (see lib/audio.js) — so it comes from BeatSaver by hash, the same host
// the song audio already comes from (`access-control-allow-origin: *`, so
// it works from the file:// page the viewer is built as).
//
// No Svelte in here; the card component does the rendering and the
// stale-response handling.
//
// The LOCAL map cache (python/maps/scoresaber/<hash>/) is deliberately NOT
// a fallback: it holds Info.dat and the difficulty .dat files but no cover
// image, and the viewer is a single file:// page, so it cannot read the
// repo anyway without a server and a new build step.

const API = "https://api.beatsaver.com/maps/hash/";
// Bump the version suffix to invalidate everything cached by an older
// shape of `shape()` below.
const STORE = "beat-arena.mapmeta.v1";
const MAX = 200; // cached maps kept in localStorage (oldest dropped first)

// hash (lowercase) -> meta object, or the in-flight promise for it. Also
// caches the null/Error outcome so a missing map is not re-asked every
// time the replay is reloaded.
const mem = new Map();

/** The fields the card needs, out of BeatSaver's map document. */
function shape(json, hash) {
  const h = hash.toLowerCase();
  const vs = Array.isArray(json.versions) ? json.versions : [];
  // the version whose hash is the one the replay was recorded on; the last
  // one is BeatSaver's newest upload and the best guess if none matches
  const v = vs.find((x) => (x.hash ?? "").toLowerCase() === h)
    ?? vs[vs.length - 1] ?? null;
  const m = json.metadata ?? {};
  const id = json.id ?? "";
  return {
    id,
    song: m.songName ?? "",
    sub: m.songSubName ?? "",
    author: m.songAuthorName ?? "",
    mapper: m.levelAuthorName ?? "",
    cover: v?.coverURL ?? "",
    url: id ? "https://beatsaver.com/maps/" + id : "",
  };
}

// ---- localStorage cache --------------------------------------------------
// Every path degrades to "no cache" rather than throwing: a private window,
// or file:// with site data blocked, refuses storage outright. The whole
// table lives under one key — it is a few hundred bytes per map.
function readAll() {
  try {
    const raw = localStorage.getItem(STORE);
    const obj = raw ? JSON.parse(raw) : null;
    return obj && typeof obj === "object" ? obj : {};
  } catch {
    return {};
  }
}

function remember(hash, meta) {
  try {
    const all = readAll();
    all[hash] = { ...meta, at: Date.now() };
    const keys = Object.keys(all);
    if (keys.length > MAX) {
      keys.sort((a, b) => (all[a].at ?? 0) - (all[b].at ?? 0))
        .slice(0, keys.length - MAX)
        .forEach((k) => delete all[k]);
    }
    localStorage.setItem(STORE, JSON.stringify(all));
  } catch {
    /* the cache is an optimization; never let it break the card */
  }
}

/** Metadata already known for this hash, without touching the network —
 * memory first, then localStorage. Null when nothing is cached. */
export function cachedMeta(hash) {
  if (!hash) return null;
  const h = hash.toLowerCase();
  const hit = mem.get(h);
  if (hit && !(hit instanceof Promise)) return hit;
  const saved = readAll()[h];
  if (saved && saved.id !== undefined) {
    const { at, ...meta } = saved;
    mem.set(h, meta);
    return meta;
  }
  return null;
}

/** Fetch (or recall) BeatSaver's metadata for a map hash.
 * Resolves to {id, song, sub, author, mapper, cover, url}; rejects when the
 * map is unknown or the network is unreachable. Concurrent calls for the
 * same hash share one request. */
export function mapMeta(hash) {
  if (!hash) return Promise.reject(new Error("no map hash"));
  const h = hash.toLowerCase();
  const hit = mem.get(h);
  if (hit) return hit instanceof Promise ? hit : Promise.resolve(hit);
  const saved = cachedMeta(h);
  if (saved) return Promise.resolve(saved);

  const p = (async () => {
    const res = await fetch(API + h);
    if (!res.ok) {
      throw new Error(res.status === 404
        ? `BeatSaver has no map ${h.slice(0, 8)}`
        : `BeatSaver returned ${res.status}`);
    }
    const meta = shape(await res.json(), h);
    mem.set(h, meta);
    remember(h, meta);
    return meta;
  })();
  // a failed lookup must not be cached as a result — the next replay load
  // (or the next time the laptop is online) should try again
  mem.set(h, p);
  p.catch(() => { if (mem.get(h) === p) mem.delete(h); });
  return p;
}
