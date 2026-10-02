// Headless-browser check of URL loading on a SERVED viewer — the hosted one
// (https://replays.flee.systems) or `wrangler dev` in viz/replays-site:
//   - ?replay=/r/<file>.rpl.gz shows download progress, opens the replay
//     paused behind a "watch" button that names it (song [difficulty] ·
//     model, from the header), at the header's tick rate;
//   - the song is fetched from BeatSaver (the header's map hash);
//   - after the click (a real user gesture: no autoplay flag here) the replay
//     runs at the header's tick rate in real time, with the song sounding
//     and locked to it;
//   - a missing file and a foreign URL each end in a clear error, not a crash.
// Run: node test/urlload-browser.mjs <base-url> </r/file.rpl.gz> [referrer-page]
// (one Chrome, SwiftShader — keep it to one run at a time on the laptop)
import { existsSync, readdirSync } from "node:fs";
import { homedir } from "node:os";
import { resolve } from "node:path";
import puppeteer from "puppeteer-core";

function findBrowser() {
  if (process.env.PUPPETEER_EXECUTABLE_PATH) return process.env.PUPPETEER_EXECUTABLE_PATH;
  const root = resolve(homedir(), ".cache/puppeteer/chrome");
  if (existsSync(root))
    for (const v of readdirSync(root).sort().reverse()) {
      const p = resolve(root, v, "chrome-linux64/chrome");
      if (existsSync(p)) return p;
    }
  throw new Error("no browser found: set PUPPETEER_EXECUTABLE_PATH");
}
const [base, path] = process.argv.slice(2);
if (!base || !path) throw new Error("usage: urlload-browser.mjs <base-url> </r/file.rpl.gz>");
const url = (p) => `${base.replace(/\/$/, "")}/?replay=${encodeURIComponent(p)}`;
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
const fail = (m) => { throw new Error(m); };

const browser = await puppeteer.launch({
  executablePath: findBrowser(),
  headless: true,
  args: ["--use-angle=swiftshader", "--disable-gpu-sandbox", "--no-sandbox"],
});
try {
  const page = await browser.newPage();
  await page.setViewport({ width: 1100, height: 800 });
  const errors = [], hosts = new Set();
  page.on("pageerror", (e) => errors.push("pageerror: " + e.message));
  page.on("console", (m) => { if (m.type() === "error") errors.push("console: " + m.text()); });
  page.on("request", (r) => hosts.add(new URL(r.url()).host));
  page.on("response", (r) => { if (r.status() >= 400) console.log(`  (HTTP ${r.status()} ${r.url().slice(0, 100)})`); });
  // slow the replay download to a visible crawl-ish pace so the progress
  // indicator is observable even from localhost
  const cdp = await page.createCDPSession();
  await cdp.send("Network.emulateNetworkConditions", {
    offline: false, latency: 50, downloadThroughput: 4 * 1024 * 1024, uploadThroughput: -1,
  });

  // ---- load from the URL -----------------------------------------------------
  const t0 = Date.now();
  await page.goto(url(path));
  let sawProgress = null;
  for (let k = 0; k < 600 && !sawProgress; k++) {
    sawProgress = await page.evaluate(() =>
      document.querySelector("#urlload [role=status]")?.textContent.replace(/\s+/g, " ").trim() || null);
    if (!sawProgress && await page.$("#watch")) break;
    await sleep(25);
  }
  console.log(`progress indicator: ${sawProgress ?? "(not seen — download too fast)"}`);
  await page.waitForSelector("#watch", { timeout: 120000 });
  console.log(`replay opened in ${((Date.now() - t0) / 1000).toFixed(1)}s`);
  await cdp.send("Network.emulateNetworkConditions", {
    offline: false, latency: 0, downloadThroughput: -1, uploadThroughput: -1 });
  // the replay's name and tick rate, from the viewer's state (the page no
  // longer prints a title line); the name is what the watch button shows
  const opened = await page.evaluate(() => ({
    ...window.__viewerState?.(),
    watch: document.querySelector("#watch")?.textContent.replace(/\s+/g, " ").trim(),
  }));
  console.log(`name: ${opened.name} · ${opened.tickHz} Hz`);
  if (!opened.loaded || !opened.name) fail("a linked replay must open with a name");
  if (!opened.watch?.includes(opened.name)) fail("the watch button does not name the replay: " + opened.watch);
  if (!(opened.tickHz > 0)) fail("no tick rate in the viewer state");
  const hz = opened.tickHz;
  // the viewer's test hook: the tick on screen and its time
  const hud = () => page.evaluate(() => JSON.stringify(window.__viewerState?.()));
  const hudT = () => page.evaluate(() => {
    const s = window.__viewerState?.();
    return s?.loaded ? { t: s.t, tick: s.tick } : null;
  });
  const still = await hudT();
  if (!still || still.tick !== 0) fail("a linked replay must open paused at tick 0: " + (await hud()));

  // ---- the song, from BeatSaver ----------------------------------------------
  const song = await page.waitForFunction(
    () => /with the song|no audio/.test(document.querySelector("#watch")?.textContent ?? ""),
    { timeout: 120000 }).then(() => page.$eval("#watch", (el) => el.textContent), () => "timeout");
  const bs = [...hosts].filter((h) => h.includes("beatsaver"));
  console.log(`song: ${/with the song/.test(song) ? "ready" : song.trim()} · BeatSaver hosts: ${bs.join(", ") || "none"}`);
  if (!bs.includes("api.beatsaver.com")) fail("the song was not looked up on BeatSaver");
  if (!/with the song/.test(song)) fail("no song: " + song);

  // ---- watch: real time at the header's rate, the song sounding --------------
  await page.click("#watch");
  await sleep(1500);
  const a0 = await hudT(), w0 = Date.now();
  await sleep(4000);
  const a1 = await hudT(), wall = (Date.now() - w0) / 1000;
  const adv = a1.t - a0.t, tps = (a1.tick - a0.tick) / adv;
  console.log(`playback: ${adv.toFixed(2)}s of replay (${a1.tick - a0.tick} ticks) in ` +
              `${wall.toFixed(2)}s wall → ${tps.toFixed(1)} ticks/s`);
  if (adv / wall < 0.75 || adv / wall > 1.25) fail("replay clock is not real time");
  if (Math.abs(tps - hz) > hz * 0.05) fail(`ticks per second ${tps.toFixed(1)} != ${hz}`);
  const s = await page.evaluate(() => {
    const a = document.querySelector("audio");
    return { cur: a.currentTime, paused: a.paused };
  });
  await sleep(1000);
  const s2 = await page.evaluate(() => document.querySelector("audio").currentTime);
  console.log(`song element: paused ${s.paused}, advanced ${(s2 - s.cur).toFixed(2)}s in ~1s`);
  if (s.paused || s2 - s.cur < 0.7) fail("the song is not playing after the click");
  await page.screenshot({ path: "test/out/urlload-playing.png" });

  if (errors.length) fail("page errors:\n" + errors.join("\n"));

  // ---- failures are explained (the 404 itself is logged by the browser) ------
  for (const [p, want] of [["/r/0000000000000000.rpl.gz", /does not exist/],
                           ["https://evil.example/r/x.rpl.gz", /under \/r\//],
                           ["/r/../index.html", /under \/r\//]]) {
    await page.goto(url(p));
    await page.waitForSelector("#urlerror", { timeout: 30000 });
    const msg = await page.$eval("#urlerror", (el) => el.textContent.replace(/\s+/g, " ").trim());
    console.log(`${p} -> ${msg}`);
    if (!want.test(msg)) fail(`${p}: unexpected message`);
  }
  if (hosts.has("evil.example")) fail("fetched a foreign host");

  const bad = errors.filter((e) => !/404/.test(e));
  if (bad.length) fail("page errors:\n" + bad.join("\n"));
  console.log("urlload browser check OK");
} finally {
  await browser.close();
}
