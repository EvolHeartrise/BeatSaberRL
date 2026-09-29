// URL loading (?replay=/r/<file>, the hosted viewer at replays.flee.systems):
// the path allow-list, gzip detection + inflation, download progress and the
// error messages, and a real replay surviving the gzip round trip.
// Run: node test/urlload.test.js
import { readFileSync } from "node:fs";
import { gzipSync } from "node:zlib";
import assert from "node:assert/strict";
import * as U from "../src/lib/urlload.js";
import { parseReplay } from "../src/lib/replay.js";

// ---- the allow-list -------------------------------------------------------
assert.equal(U.replayPathFrom(""), null, "no parameter = nothing to load");
assert.equal(U.replayPathFrom("?foo=1"), null);
for (const ok of ["/r/0123456789abcdef.jsonl.gz", "/r/0123456789abcdef.jsonl",
                  "/r/a_b-C9.jsonl.gz"])
  assert.equal(U.replayPathFrom("?replay=" + encodeURIComponent(ok)), ok, ok);
// URLSearchParams decodes, so an encoded path is the same path
assert.equal(U.replayPathFrom("?replay=%2Fr%2Fabc.jsonl.gz"), "/r/abc.jsonl.gz");
for (const bad of [
  "", "https://evil.example/r/x.jsonl.gz", "//evil.example/r/x.jsonl.gz",
  "/r/../index.html", "/r/..%2Fx.jsonl", "/r/x/../../y.jsonl", "/r/.jsonl",
  "r/x.jsonl.gz", "/x.jsonl.gz", "/r/x.json", "/r/x.txt", "/r/x.jsonl.gz.js",
  "/r/sub/x.jsonl.gz", "/r/x.jsonl?y=1", "/r/x.jsonl#y", "\\\\evil\\r\\x.jsonl",
  "/r/x y.jsonl", "javascript:alert(1)//r/x.jsonl", "/r/%2e%2e.jsonl",
  "/r/" + "a".repeat(200) + ".jsonl",
]) assert.throws(() => U.replayPathFrom("?replay=" + encodeURIComponent(bad)),
                 /under \/r\//, JSON.stringify(bad));
assert.equal(U.replayUrl("/r/abc.jsonl.gz", "https://replays.flee.systems"),
             "https://replays.flee.systems/r/abc.jsonl.gz");
assert.throws(() => U.replayUrl("//evil.example/r/x", "https://replays.flee.systems"));

// ---- gzip ------------------------------------------------------------------
const text = readFileSync(
  new URL("./fixtures/v10_scripted_ep00_seed0.jsonl", import.meta.url), "utf8");
const raw = new TextEncoder().encode(text);
const gz = new Uint8Array(gzipSync(raw));
assert.ok(U.isGzip(gz) && !U.isGzip(raw) && !U.isGzip(new Uint8Array([0x1f])));
assert.equal(await U.bytesToText(gz), text, "gzip inflates to the same text");
assert.equal(await U.bytesToText(raw), text, "plain bytes pass through");
await assert.rejects(U.bytesToText(gz.slice(0, gz.length >> 1)), /damaged/,
                     "a truncated gzip is reported, not parsed half-way");
// drag-and-drop of a .jsonl.gz
assert.equal(await U.fileToText(new Blob([gz])), text);
// the inflated replay is the replay
const a = parseReplay(await U.bytesToText(gz)), b = parseReplay(text);
assert.equal(a.ticks.length, b.ticks.length);
assert.deepEqual(a.outcomes, b.outcomes);

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
  assert.equal(t, text);
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
  assert.equal(t, text);
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
                       "/r/abc.jsonl.gz"), "Air Raid 2010 [ExpertPlus]");
assert.equal(U.nameFor({ map: null }, "/r/abc.jsonl.gz"), "abc");
assert.equal(U.nameFor({ map: { song: "S", difficulty: "Hard" }, model: "v60-60hz" },
                       "/r/abc.jsonl.gz"), "S [Hard] · v60-60hz");
assert.equal(parseReplay(text).model, null, "model is an optional header field");

console.log("urlload tests passed");
