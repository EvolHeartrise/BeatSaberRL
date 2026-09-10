// Guard for the frame-by-frame video export (src/lib/record.js). Runs a real
// export in the built viewer and checks the file is not BLANK.
//   node test/export-probe.mjs [replay.jsonl] [fps]
//
// This exists because of a bug that every structural check passed: the
// exporter called Threlte's advance(), which only sets a "this frame wants a
// render" flag — the render happens inside scheduler.run(), which advance()
// does not call. So every frame was captured off an unrendered canvas. Frame
// count, frame rate, duration, pixel format and the audio track were all
// perfect; the video was 23k frames of pure black. Bytes-per-frame is what
// separates the two: ~0.5 kB blank vs ~13 kB with gameplay in it.
import { resolve } from "node:path";
import { existsSync, readdirSync } from "node:fs";
import { homedir } from "node:os";
import puppeteer from "puppeteer-core";

const MIN_BYTES_PER_FRAME = 2000;

const replay = resolve(process.argv[2]
  ?? "../../python/replays/v15_probe_toppp01_delightful-introduction_1pp.jsonl");
const fps = process.argv[3] ?? "30";
const dist = resolve("dist/index.html");
if (!existsSync(dist)) throw new Error("run `npm run build` first");

const root = resolve(homedir(), ".cache/puppeteer/chrome");
const chrome = readdirSync(root).sort().reverse()
  .map((v) => resolve(root, v, "chrome-linux64/chrome")).find(existsSync);
const browser = await puppeteer.launch({
  executablePath: chrome,
  headless: true,
  args: ["--use-angle=swiftshader", "--no-sandbox",
         "--autoplay-policy=no-user-gesture-required"],
});
const page = await browser.newPage();
await page.setViewport({ width: 480, height: 360 });
const errors = [];
page.on("pageerror", (e) => errors.push(e.message));
page.on("dialog", async (d) => { errors.push("dialog: " + d.message()); await d.dismiss(); });

await page.goto("file://" + dist);
await page.waitForSelector("#stage canvas", { timeout: 15000 });
(await page.$("input[type=file]")).uploadFile(replay);
await page.waitForFunction(() => /tick/i.test(document.body.innerText),
  { timeout: 20000 });
// let the song settle (or report that this replay has none)
await page.waitForFunction(() => {
  const a = document.querySelector("audio");
  return !a || a.readyState >= 1
    || /no song|no audio/.test(document.querySelector("#snd")?.textContent ?? "");
}, { timeout: 120000, polling: 500 });
await page.keyboard.press(" "); // pause playback
const ticks = await page.evaluate(() =>
  +(/tick \d+\/(\d+)/.exec(document.querySelector("#hud").textContent)?.[1] ?? 0));
await page.select("select", fps);

// keep the Blob instead of downloading it
await page.evaluate(() => {
  window.__blob = null;
  const orig = URL.createObjectURL.bind(URL);
  URL.createObjectURL = (b) => {
    if (b instanceof Blob && /webm/.test(b.type)) window.__blob = b;
    return orig(b);
  };
  const click = HTMLAnchorElement.prototype.click;
  HTMLAnchorElement.prototype.click = function () {
    if (this.href?.startsWith("blob:")) return;
    return click.apply(this, arguments);
  };
});
const t0 = Date.now();
await page.evaluate(() => [...document.querySelectorAll("button")]
  .find((b) => /export/i.test(b.textContent)).click());
await page.waitForFunction(() => !!window.__blob,
  { timeout: 600000, polling: 500 });
const size = await page.evaluate(() => window.__blob.size);
await browser.close();

const frames = Math.floor((ticks / 30) * +fps) + 1;
const per = size / frames;
console.log(`${ticks} ticks @ ${fps} fps -> ${frames} frames, ${size} bytes `
  + `(${(per / 1024).toFixed(1)} kB/frame) in ${((Date.now() - t0) / 1000).toFixed(1)}s`);
if (errors.length) throw new Error("page errors: " + errors.join("; "));
if (per < MIN_BYTES_PER_FRAME) {
  throw new Error(`export looks BLANK: ${per.toFixed(0)} bytes/frame is under `
    + `${MIN_BYTES_PER_FRAME} — the frames are probably not being rendered`);
}
console.log("export probe OK");
