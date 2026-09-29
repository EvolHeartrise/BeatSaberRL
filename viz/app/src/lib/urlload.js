// Open a replay named by the page URL: `?replay=/r/<file>.jsonl[.gz]`.
// This is how the hosted viewer (replays.flee.systems) is linked to from the
// records site. No Svelte in here; App.svelte owns the loading UI.
//
// Only same-origin files under /r/ may be named: the parameter is a path,
// never a URL, so a crafted link cannot make the viewer fetch from anywhere
// else. A 60 Hz JSONL runs ~6-25 MB raw, so the host serves gzip files
// (`.jsonl.gz`, ~5x smaller) and they are inflated here with the browser's
// own DecompressionStream. The bytes decide, not the name: gzip starts with
// 1f 8b, and if something upstream already undid the compression (a server
// sending Content-Encoding: gzip) the body arrives as plain JSONL and is
// read as it is.

export const REPLAY_PARAM = "replay";

// flat file names only: /r/<name>.jsonl or /r/<name>.jsonl.gz, where the name
// is letters, digits, '_' and '-' (no dots, so no '..', no hidden files, no
// second extension), and nothing percent-encoded
const PATH_RE = /^\/r\/[A-Za-z0-9_-]{1,128}\.jsonl(\.gz)?$/;

/** The replay path the page URL names, or null when it names none.
 * Throws with a message fit for the user when the parameter is present but
 * not an allowed path. `search` is location.search. */
export function replayPathFrom(search) {
  const params = new URLSearchParams(search);
  if (!params.has(REPLAY_PARAM)) return null;
  const p = params.get(REPLAY_PARAM) ?? "";
  if (!PATH_RE.test(p))
    throw new Error("the replay link must name a file under /r/ on this site "
      + "(like /r/0123456789abcdef.jsonl.gz)");
  return p;
}

/** The absolute URL to fetch for an allowed path, checked to stay on
 * `origin` (belt and braces: PATH_RE already rules out anything else). */
export function replayUrl(path, origin) {
  const u = new URL(path, origin);
  if (u.origin !== new URL(origin).origin || !u.pathname.startsWith("/r/"))
    throw new Error("the replay must be on this site");
  return u.href;
}

export const isGzip = (b) => b.length >= 2 && b[0] === 0x1f && b[1] === 0x8b;

/** Bytes -> replay text, inflating gzip when the bytes are gzip. */
export async function bytesToText(bytes) {
  if (!isGzip(bytes)) return new TextDecoder().decode(bytes);
  if (typeof DecompressionStream === "undefined")
    throw new Error("this browser cannot unpack .gz files (no DecompressionStream)");
  const stream = new Blob([bytes]).stream()
    .pipeThrough(new DecompressionStream("gzip"));
  try {
    return await new Response(stream).text();
  } catch (e) {
    throw new Error("the replay file is damaged (gzip: " + e.message + ")");
  }
}

/** A File (drag-and-drop / file picker) -> replay text; takes .gz too. */
export async function fileToText(file) {
  return bytesToText(new Uint8Array(await file.arrayBuffer()));
}

/** Download a replay with progress, then return its text.
 * onProgress({stage, loaded, total}): stage is "downloading" (total = 0 when
 * the size is unknown) then "unpacking". `fetchFn` is for tests. */
export async function fetchReplay(url, onProgress = () => {}, fetchFn = fetch) {
  let res;
  try {
    res = await fetchFn(url);
  } catch (e) {
    throw new Error("could not reach the server (" + e.message + ")");
  }
  if (res.status === 404) throw new Error("this replay does not exist (404)");
  if (!res.ok) throw new Error(`the server refused the replay (HTTP ${res.status})`);
  // a static host answers an unknown path with its index page on some
  // configurations; that is not a replay, whatever the status says
  const type = res.headers.get("content-type") || "";
  if (type.includes("text/html")) throw new Error("this replay does not exist");

  // content-length is the size on the wire; with a transfer encoding the
  // byte count below is of decoded data, so only trust it when unencoded
  const encoded = !!res.headers.get("content-encoding");
  const total = encoded ? 0 : +res.headers.get("content-length") || 0;
  let bytes;
  if (res.body && res.body.getReader) {
    const reader = res.body.getReader();
    const parts = [];
    let loaded = 0;
    onProgress({ stage: "downloading", loaded, total });
    for (;;) {
      const { done, value } = await reader.read();
      if (done) break;
      parts.push(value);
      loaded += value.length;
      onProgress({ stage: "downloading", loaded, total });
    }
    bytes = new Uint8Array(loaded);
    let o = 0;
    for (const p of parts) { bytes.set(p, o); o += p.length; }
  } else {
    bytes = new Uint8Array(await res.arrayBuffer());
  }
  onProgress({ stage: "unpacking", loaded: bytes.length, total: bytes.length });
  return bytesToText(bytes);
}

/** Display name for a replay loaded from a URL: the song and difficulty when
 * the header names the map, else the file name without extensions. */
export function nameFor(rep, path) {
  const m = rep?.map;
  const base = m?.song ? (m.difficulty ? `${m.song} [${m.difficulty}]` : m.song)
    : path.split("/").pop().replace(/\.jsonl(\.gz)?$/, "");
  return rep?.model ? `${base} · ${rep.model}` : base;
}
