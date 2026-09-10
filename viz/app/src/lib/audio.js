// Song audio for a replay: BeatSaver -> map .zip -> the map's own audio
// file -> a blob URL an <audio> element can play. No Svelte in here; the
// playback wiring lives in App.svelte.
//
// A replay only knows which song it is if it was stamped by
// python/annotate_replay_audio.py, which writes header.map =
// {hash, difficulty, song, audio_offset_s}. audio_offset_s is the song time
// at replay t = 0 — recordings re-base the chart so the first note lands at
// 2 s, which throws the song's clock away, and the observed offsets run from
// -0.04 s to +19.6 s, so it is not a detail that can be guessed.
//
// Both BeatSaver hosts send `access-control-allow-origin: *`, so this works
// from the file:// page the viewer is built as.
import { unzipSync, strFromU8 } from "fflate";

const API = "https://api.beatsaver.com/maps/hash/";
const DB = "beat-arena-audio", STORE = "songs";

// ---- IndexedDB cache (keyed by map hash) ---------------------------------
// A map zip is ~10 MB. Re-viewing a replay, or scrubbing to a replay and
// back, must not re-download it. Every path here degrades to "no cache"
// rather than failing: private windows and file:// with site data blocked
// can refuse to open a database at all.
function idb() {
  return new Promise((res, rej) => {
    let req;
    try {
      req = indexedDB.open(DB, 1);
    } catch (e) {
      rej(e);
      return;
    }
    req.onupgradeneeded = () => req.result.createObjectStore(STORE);
    req.onsuccess = () => res(req.result);
    req.onerror = () => rej(req.error);
  });
}

async function cacheGet(key) {
  try {
    const db = await idb();
    return await new Promise((res, rej) => {
      const r = db.transaction(STORE, "readonly").objectStore(STORE).get(key);
      r.onsuccess = () => res(r.result || null);
      r.onerror = () => rej(r.error);
    });
  } catch {
    return null;
  }
}

async function cachePut(key, val) {
  try {
    const db = await idb();
    await new Promise((res, rej) => {
      const t = db.transaction(STORE, "readwrite");
      t.objectStore(STORE).put(val, key);
      t.oncomplete = () => res();
      t.onerror = () => rej(t.error);
    });
  } catch {
    /* cache is an optimization; a failure here must not break playback */
  }
}

// ---- the map's audio file ------------------------------------------------
// Info.dat names it: `_songFilename` in the v2/v3 schema, `audio.songFilename`
// in v4. Beat Saber ships it as .egg, which is Ogg Vorbis under another
// extension — hence the explicit MIME type, or the browser sniffs nothing
// useful from a blob and refuses to decode it.
const MIME = { egg: "audio/ogg", ogg: "audio/ogg", mp3: "audio/mpeg",
               wav: "audio/wav", m4a: "audio/mp4" };

function audioFromZip(buf) {
  const files = unzipSync(new Uint8Array(buf));
  const find = (want) => Object.keys(files).find(
    (k) => k.toLowerCase() === want || k.toLowerCase().endsWith("/" + want));
  const infoKey = find("info.dat");
  if (!infoKey) throw new Error("map zip has no Info.dat");
  const info = JSON.parse(strFromU8(files[infoKey]));
  const name = info._songFilename ?? info.audio?.songFilename;
  if (!name) throw new Error("Info.dat names no audio file");
  const key = find(name.toLowerCase());
  if (!key) throw new Error(`audio file ${name} missing from the zip`);
  const ext = name.split(".").pop().toLowerCase();
  return { blob: new Blob([files[key]], { type: MIME[ext] ?? "audio/ogg" }),
           name };
}

/** Fetch (or recall) the song audio for a map hash as a Blob.
 * `onStage(text)` reports progress for the UI. */
export async function songBlob(hash, onStage = () => {}) {
  const hit = await cacheGet(hash);
  if (hit) {
    onStage("cached");
    return hit;
  }
  onStage("finding map…");
  const meta = await fetch(API + hash);
  if (!meta.ok) throw new Error(`BeatSaver has no map ${hash.slice(0, 8)}`);
  const url = (await meta.json()).versions?.[0]?.downloadURL;
  if (!url) throw new Error("BeatSaver returned no download URL");

  onStage("downloading…");
  const zip = await fetch(url);
  if (!zip.ok) throw new Error(`map download failed (${zip.status})`);
  const total = +zip.headers.get("content-length") || 0;
  let buf;
  if (zip.body && total) { // report progress on the one slow step
    const reader = zip.body.getReader();
    const parts = [];
    let got = 0;
    for (;;) {
      const { done, value } = await reader.read();
      if (done) break;
      parts.push(value);
      got += value.length;
      onStage(`downloading ${Math.round((100 * got) / total)}%`);
    }
    buf = new Blob(parts).arrayBuffer ? await new Blob(parts).arrayBuffer()
      : parts;
  } else {
    buf = await zip.arrayBuffer();
  }

  onStage("unpacking…");
  const { blob } = audioFromZip(buf);
  await cachePut(hash, blob);
  return blob;
}
