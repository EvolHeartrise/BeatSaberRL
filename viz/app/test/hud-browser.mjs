// Headless-browser eyeball check of the 3D in-game HUD (src/lib/Hud3D.svelte)
// on the built viewer: loads a replay into dist/index.html, scrubs to a few
// ticks and screenshots them in the orbit and first-person cameras, into
// test/out/hud-*.png. Fails on page errors.
// Run: npm run build && node test/hud-browser.mjs <replay.rpl.gz> [tick ...]
// (one Chrome, SwiftShader — keep it to one run at a time on the laptop)
import { existsSync, mkdirSync, readdirSync } from "node:fs";
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
const replay = resolve(process.argv[2] ?? "test/fixtures/v10_scripted_ep00_seed0.rpl.gz");
const ticks = process.argv.slice(3).map(Number);
if (!ticks.length) ticks.push(60, 200);
if (!existsSync(dist)) throw new Error("run `npm run build` first");
mkdirSync("test/out", { recursive: true });
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

const browser = await puppeteer.launch({
  executablePath: findBrowser(),
  headless: true,
  args: ["--use-angle=swiftshader", "--disable-gpu-sandbox", "--no-sandbox"],
});
try {
  const page = await browser.newPage();
  // HUD_VIEW=WxH picks another window shape (default 1280x720)
  const [vw, vh] = (process.env.HUD_VIEW ?? "1280x720").split("x").map(Number);
  await page.setViewport({ width: vw, height: vh, deviceScaleFactor: 1.5 });
  const errors = [];
  page.on("pageerror", (e) => errors.push("pageerror: " + e.message));
  page.on("console", (m) => { if (m.type() === "error") errors.push("console: " + m.text()); });
  await page.goto("file:///" + dist.replace(/\\/g, "/"));
  await page.waitForSelector("#stage canvas", { timeout: 15000 });
  const input = await page.$("input[type=file]");
  await input.uploadFile(replay);
  await page.waitForFunction(
    () => window.__viewerState?.().loaded,
    { timeout: 60000 });
  // paused, so the scrubbed tick is the one on screen
  await page.evaluate(() => [...document.querySelectorAll("#bar button")]
    .find((b) => b.textContent.includes("pause"))?.click());
  const scrub = (tick) => page.evaluate((tick) => {
    // the timeline is the last slider in the bar (fov / volume come first)
    const r = [...document.querySelectorAll("#bottom input[type=range]")].at(-1);
    Object.getOwnPropertyDescriptor(HTMLInputElement.prototype, "value").set.call(r, tick);
    r.dispatchEvent(new Event("input", { bubbles: true }));
  }, tick);
  const FP_ON = `[...document.querySelectorAll("#bar button")]
    .some((b) => b.textContent.includes("first-person") && b.classList.contains("on"))`;
  for (const view of ["orbit", "fp"]) {
    if (view === "fp") {
      await page.keyboard.press("f");
      await page.waitForFunction(FP_ON, { timeout: 5000 });
    }
    for (const t of ticks) {
      await scrub(t);
      await sleep(900);
      const vs = await page.evaluate(() => window.__viewerState());
      if (vs.tick !== Math.min(t, vs.lastTick)) throw new Error(`scrub to ${t} landed on ${vs.tick}`);
      const hud = `tick ${vs.tick}/${vs.lastTick} t ${vs.t.toFixed(2)}s`;
      const out = `test/out/hud-${view}-${t}${process.env.HUD_VIEW ? "-" + vw + "x" + vh : ""}.png`;
      await page.screenshot({ path: out });
      console.log(`${out}  (${hud})`);
    }
  }
  if (errors.length) throw new Error("page errors:\n" + errors.join("\n"));
  console.log("hud browser check OK");
} finally {
  await browser.close();
}
