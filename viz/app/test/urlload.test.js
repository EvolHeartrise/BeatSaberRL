// URL loading (?replay=/r/<file>, the hosted viewer at replays.flee.systems):
// the path allow-list, gzip detection + inflation, download progress and the
// error messages, and a real replay surviving the gzip round trip.
// Run: node test/urlload.test.js
import { readFileSync } from "node:fs";
import { gzipSync } from "node:zlib";
import assert from "node:assert/strict";
import * as U from "../src/lib/urlload.js";
import { parseReplay } from "../src/lib/replay.js";
import { FIXTURE, readReplayFile } from "./fixture.js";

// ---- the allow-list -------------------------------------------------------
assert.equal(U.replayPathFrom(""), null, "no parameter = nothing to load");
assert.equal(U.replayPathFrom("?foo=1"), null);
for (const ok of ["/r/0123456789abcdef.rpl.gz", "/r/a_b-C9.rpl.gz"])
  assert.equal(U.replayPathFrom("?replay=" + encodeURIComponent(ok)), ok, ok);
// URLSearchParams decodes, so an encoded path is the same path
assert.equal(U.replayPathFrom("?replay=%2Fr%2Fabc.rpl.gz"), "/r/abc.rpl.gz");
for (const bad of [
  "", "https://evil.example/r/x.rpl.gz", "//evil.example/r/x.rpl.gz",
  "/r/../index.html", "/r/..%2Fx.rpl.gz", "/r/x/../../y.rpl.gz", "/r/.rpl.gz",
  "r/x.rpl.gz", "/x.rpl.gz", "/r/x.json", "/r/x.txt", "/r/x.rpl.gz.js",
  "/r/sub/x.rpl.gz", "/r/x.rpl.gz?y=1", "/r/x.rpl.gz#y", "\\\\evil\\r\\x.rpl.gz",
  "/r/x y.rpl.gz", "javascript:alert(1)//r/x.rpl.gz", "/r/%2e%2e.rpl.gz",
  "/r/" + "a".repeat(200) + ".rpl.gz", "/r/x.rpl", "/r/x.gz",
  // the retired JSONL replay names
  "/r/0123456789abcdef.jsonl.gz", "/r/0123456789abcdef.jsonl",
]) assert.throws(() => U.replayPathFrom("?replay=" + encodeURIComponent(bad)),
                 /under \/r\//, JSON.stringify(bad));
assert.equal(U.replayUrl("/r/abc.rpl.gz", "https://replays.flee.systems"),
             "https://replays.flee.systems/r/abc.rpl.gz");
assert.throws(() => U.replayUrl("//evil.example/r/x", "https://replays.flee.systems"));

// ---- gzip ------------------------------------------------------------------
const gz = new Uint8Array(readFileSync(FIXTURE));    // the file as served
const raw = readReplayFile(FIXTURE);                  // its inflated payload
assert.ok(U.isGzip(gz) && !U.isGzip(raw) && !U.isGzip(new Uint8Array([0x1f])));
assert.deepEqual(await U.bytesToReplay(gz), raw, "gzip inflates to the replay bytes");
assert.deepEqual(await U.bytesToReplay(raw), raw, "inflated bytes pass through");
await assert.rejects(U.bytesToReplay(gz.slice(0, gz.length >> 1)), /damaged/,
                     "a truncated gzip is reported, not parsed half-way");
// drag-and-drop of a .rpl.gz
assert.deepEqual(await U.fileToReplay(new Blob([gz])), raw);
// the inflated replay is the replay
const a = parseReplay(await U.bytesToReplay(gz)), b = parseReplay(raw);
assert.equal(a.ticks.length, b.ticks.length);
assert.deepEqual(a.outcomes, b.outcomes);
// a Flux run's JSONL (not a replay payload) comes back as text
{
  const flux = readFileSync(new URL("./fixtures/flux_scripted.jsonl", import.meta.url), "utf8");
  const fgz = new Uint8Array(gzipSync(new TextEncoder().encode(flux)));
  assert.equal(await U.bytesToReplay(fgz), flux);
  assert.equal(parseReplay(await U.bytesToReplay(fgz)).game, "flux");
}

// ---- fetchReplay: progress, content types, errors -------------------------
function mockFetch(body, { status = 200, headers = {}, chunk = 4096 } = {}) {
  return async () => {
    let off = 0;
    const stream = new ReadableStream({
      pull(c) {
        if (off >= body.length) { c.close(); return; }
        c.enqueue(body.slice(off, off + chunk));
        off += chunk;
      },
    });
    return new Response(status === 200 ? stream : null,
                        { status, headers: new Headers(headers) });
  };
}
{
  const seen = [];
  const t = await U.fetchReplay("x", (p) => seen.push({ ...p }), mockFetch(gz, {
    headers: { "content-length": String(gz.length),
               "content-type": "application/gzip" } }));
  assert.deepEqual(t, raw);
  const dl = seen.filter((p) => p.stage === "downloading");
  assert.ok(dl.length > 2, "progress reported while downloading");
  assert.equal(dl.at(-1).loaded, gz.length);
  assert.ok(dl.every((p) => p.total === gz.length));
  assert.ok(dl.every((p, i) => i === 0 || p.loaded >= dl[i - 1].loaded), "monotone");
  assert.equal(seen.at(-1).stage, "unpacking");
}
{
  // Content-Encoding: the body arrives already inflated, and content-length
  // (the wire size) must not be used as the total
  const seen = [];
  const t = await U.fetchReplay("x", (p) => seen.push(p), mockFetch(raw, {
    headers: { "content-length": "123", "content-encoding": "gzip" } }));
  assert.deepEqual(t, raw);
  assert.ok(seen.filter((p) => p.stage === "downloading").every((p) => p.total === 0));
}
await assert.rejects(U.fetchReplay("x", () => {}, mockFetch(raw, { status: 404 })),
                     /does not exist \(404\)/);
await assert.rejects(U.fetchReplay("x", () => {}, mockFetch(raw, { status: 500 })),
                     /HTTP 500/);
await assert.rejects(U.fetchReplay("x", () => {}, mockFetch(
  new TextEncoder().encode("<!doctype html>"), { headers: { "content-type": "text/html" } })),
  /does not exist/, "an index-page fallback is not a replay");
await assert.rejects(U.fetchReplay("x", () => {}, async () => { throw new TypeError("offline"); }),
                     /could not reach the server/);

// ---- display name ----------------------------------------------------------
assert.equal(U.nameFor({ map: { song: "Air Raid 2010", difficulty: "ExpertPlus" } },
                       "/r/abc.rpl.gz"), "Air Raid 2010 [ExpertPlus]");
assert.equal(U.nameFor({ map: null }, "/r/abc.rpl.gz"), "abc");
assert.equal(U.nameFor({ map: { song: "S", difficulty: "Hard" }, model: "v60-60hz" },
                       "/r/abc.rpl.gz"), "S [Hard] · v60-60hz");
assert.equal(parseReplay(raw).model, null, "model is an optional header field");

console.log("urlload tests passed");
