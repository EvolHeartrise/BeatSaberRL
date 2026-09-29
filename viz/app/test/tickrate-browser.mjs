// Headless-browser check of the BUILT viewer on a non-30 Hz replay
// (exp/tick-rate, replay record_rev 6): the replay clock runs at the
// header's tick_hz, a scrubbed tick shows tick / tick_hz seconds, playback
// covers real time, the song stays locked to the replay clock, and loading
// a 30 Hz replay afterwards puts the clock back at 30 ticks/s.
// Run: node test/tickrate-browser.mjs <60hz-replay.jsonl> [30hz-replay.jsonl]
// (one Chrome, SwiftShader — keep it to one run at a time on the laptop)
import { existsSync, readFileSync, readdirSync } from "node:fs";
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
const dist = resolve("dist/index.html");
if (!existsSync(dist)) throw new Error("run `npm run build` first");
const fast = process.argv[2] && resolve(process.argv[2]);
if (!fast) throw new Error("usage: tickrate-browser.mjs <replay.jsonl> [30hz.jsonl]");
const slow = resolve(process.argv[3] ?? "test/fixtures/v10_scripted_ep00_seed0.jsonl");
const headerOf = (p) => {
  const fd = readFileSync(p, "utf8");
  return JSON.parse(fd.slice(0, fd.indexOf("\n"))).header;
};
const HZ = headerOf(fast).tick_hz ?? 30;
if (HZ === 30) throw new Error(fast + " is a 30 Hz replay");

const browser = await puppeteer.launch({
  executablePath: findBrowser(),
  headless: true,
  args: ["--use-angle=swiftshader", "--disable-gpu-sandbox", "--no-sandbox",
         "--autoplay-policy=no-user-gesture-required"],
});
const fail = (m) => { throw new Error(m); };
try {
  const page = await browser.newPage();
  await page.setViewport({ width: 1100, height: 900 });
  const errors = [];
  page.on("pageerror", (e) => errors.push("pageerror: " + e.message));
  page.on("console", (m) => { if (m.type() === "error") errors.push("console: " + m.text()); });
  await page.goto("file:///" + dist);
  await page.waitForSelector("#stage canvas", { timeout: 15000 });

  const hud = () => page.$eval("#hud", (el) => el.textContent.replace(/\s+/g, " "));
  const hudT = async () => {
    const m = /t (\d+\.\d+)s · tick (\d+)/.exec(await hud());
    return m ? { t: +m[1], tick: +m[2] } : null;
  };
  const scrub = (tick) => page.evaluate((tick) => {
    const r = document.querySelector("#bottom input[type=range]");
    Object.getOwnPropertyDescriptor(HTMLInputElement.prototype, "value").set.call(r, tick);
    r.dispatchEvent(new Event("input", { bubbles: true }));
  }, tick);
  const load = async (p) => {
    const input = await page.$("input[type=file]");
    await input.uploadFile(p);
    await page.waitForFunction(
      () => document.querySelector("#hud").textContent.includes("tick"), { timeout: 30000 });
    await new Promise((r) => setTimeout(r, 800));
  };

  // ---- the fast replay: scrubbed tick -> tick / HZ seconds
  await load(fast);
  const drop = await page.$eval("body", (el) => el.textContent);
  if (!drop.includes(`${HZ} Hz`)) fail(`the replay summary does not name ${HZ} Hz`);
  await page.keyboard.press(" "); // pause
  for (const secs of [10, 25]) {
    await scrub(secs * HZ);
    await new Promise((r) => setTimeout(r, 300));
    const h = await hudT();
    console.log(`scrub tick ${secs * HZ}: HUD t ${h?.t}s tick ${h?.tick}`);
    if (!h || h.tick !== secs * HZ || Math.abs(h.t - secs) > 0.011)
      fail(`tick ${secs * HZ} should read t ${secs}.00s at ${HZ} Hz: ${await hud()}`);
  }
  await page.screenshot({ path: "test/out/tickrate-60hz.png" });

  // ---- playback covers real time; the song tracks the replay clock
  // the song comes from BeatSaver by the header's map hash: give it a while
  const hasSong = await page.waitForFunction(
    () => !!document.querySelector("#snd input[type=range]"), { timeout: 30000 })
    .then(() => true, () => false);
  await scrub(5 * HZ);
  await page.keyboard.press(" "); // play
  await new Promise((r) => setTimeout(r, 500));
  const a0 = await hudT(), w0 = Date.now();
  await new Promise((r) => setTimeout(r, 3000));
  const a1 = await hudT(), wall = (Date.now() - w0) / 1000;
  const adv = a1.t - a0.t;
  console.log(`playback: HUD advanced ${adv.toFixed(2)}s (${a1.tick - a0.tick} ticks) ` +
              `in ${wall.toFixed(2)}s wall`);
  // SwiftShader may drop frames, but the clock is delta-driven: a 30 Hz
  // clock on a 60 Hz replay would read half speed, a 1/60 DT on 30 Hz
  // ticks double
  if (adv / wall < 0.75 || adv / wall > 1.25) fail("replay clock is not real time");
  if (Math.abs((a1.tick - a0.tick) / adv - HZ) > HZ * 0.05)
    fail(`ticks per second ${((a1.tick - a0.tick) / adv).toFixed(1)} != ${HZ}`);
  if (hasSong) {
    const s = await page.evaluate(() => {
      const a = document.querySelector("audio");
      return { cur: a.currentTime, paused: a.paused };
    });
    const offset = headerOf(fast).map?.audio_offset_s ?? 0;
    const h = await hudT();
    const lag = s.cur - offset - h.t;
    console.log(`song: element at ${s.cur.toFixed(2)}s, replay song time ` +
                `${(offset + h.t).toFixed(2)}s (offset ${offset.toFixed(3)}), ` +
                `diff ${lag.toFixed(3)}s, paused ${s.paused}`);
    if (!s.paused && Math.abs(lag) > 0.2) fail("song and replay clock apart: " + lag);
  } else {
    console.log("song: none loaded (offline?) — sync check skipped");
  }

  // ---- back to a 30 Hz replay: DT returns to 1/30
  await load(slow);
  await page.keyboard.press(" ");
  await scrub(300);
  await new Promise((r) => setTimeout(r, 300));
  const h30 = await hudT();
  console.log(`30 Hz replay, scrub tick 300: HUD t ${h30?.t}s`);
  if (!h30 || h30.tick !== 300 || Math.abs(h30.t - 10) > 0.011)
    fail("30 Hz replay after a 60 Hz one does not read 30 ticks/s: " + (await hud()));

  if (errors.length) fail("page errors:\n" + errors.join("\n"));
  console.log(`tickrate browser check OK (${HZ} Hz)`);
} finally {
  await browser.close();
}
