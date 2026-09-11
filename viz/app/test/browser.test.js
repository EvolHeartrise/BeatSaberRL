// Headless-browser smoke test of the BUILT single-file viewer: loads
// dist/index.html in Edge, uploads a real v10 replay through the file
// input, plays a second, and asserts the HUD advanced with zero page
// errors. Also drives the v10 manual-pose panel (root + 29 joints) and
// screenshots the articulated skeleton + per-joint gauges/forces, and
// toggles the first-person (head-frame) camera on and back off.
// Screenshots land in test/out/ for eyeball verification.
// Run: node test/browser.test.js [path-to-replay.jsonl]
import { existsSync, mkdirSync } from "node:fs";
import { resolve } from "node:path";
import puppeteer from "puppeteer-core";

// browser binary: $PUPPETEER_EXECUTABLE_PATH, else Edge on the Windows
// box, else the Chrome that `npx puppeteer browsers install chrome` put
// under ~/.cache/puppeteer (Linux laptop)
import { readdirSync } from "node:fs";
import { homedir } from "node:os";
function findBrowser() {
  if (process.env.PUPPETEER_EXECUTABLE_PATH) return process.env.PUPPETEER_EXECUTABLE_PATH;
  const edge = "C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe";
  if (existsSync(edge)) return edge;
  const root = resolve(homedir(), ".cache/puppeteer/chrome");
  if (existsSync(root)) {
    for (const v of readdirSync(root).sort().reverse()) {
      const p = resolve(root, v, "chrome-linux64/chrome");
      if (existsSync(p)) return p;
    }
  }
  throw new Error("no browser found: set PUPPETEER_EXECUTABLE_PATH");
}
const EDGE = findBrowser();
const dist = resolve("dist/index.html");
const replay = resolve(process.argv[2] ??
  "../../python/replays/v10_scripted_ep00_seed0.jsonl");
// env v15: a recording whose header carries "track_anchor": "head", so the
// built bundle is exercised on the head-anchored track as well as on the
// legacy fixed one above (skipped if the probe recording is not present).
const replayV15 = resolve(process.argv[3] ??
  "../../python/replays/v15_probe_toppp01_delightful-introduction_1pp.jsonl");
if (!existsSync(dist)) throw new Error("run `npm run build` first");
mkdirSync("test/out", { recursive: true });

const browser = await puppeteer.launch({
  executablePath: EDGE,
  headless: true,
  // --no-sandbox: Ubuntu 24.04's AppArmor blocks Chrome's user-namespace
  // sandbox for unprivileged users; this only ever loads our own local
  // file, so the sandbox buys nothing here
  args: ["--use-angle=swiftshader", "--disable-gpu-sandbox", "--no-sandbox"],
});
try {
  const page = await browser.newPage();
  await page.setViewport({ width: 1100, height: 900 });
  const errors = [];
  page.on("pageerror", (e) => errors.push("pageerror: " + e.message));
  page.on("console", (m) => {
    if (m.type() === "error") errors.push("console: " + m.text());
  });

  await page.goto("file:///" + dist.replace(/\\/g, "/"));
  await page.waitForSelector("#stage canvas", { timeout: 15000 });
  await page.screenshot({ path: "test/out/empty.png" });

  // ---- manual pose mode works straight from the landing state (no file):
  // toggle 'p', drive root + joint sliders, check the FK readout reacts.
  // Slider DOM order: 0-5 root (x,y,z,yaw,pitch,roll), 6-8 spine,
  // 9-15 L arm, 16-21 L leg, 22-28 R arm, 29-34 R leg.
  await page.keyboard.press("p");
  await page.waitForSelector("#pose", { timeout: 5000 });
  const sliderCount = await page.$$eval("#pose input[type=range]",
    (els) => els.length);
  if (sliderCount !== 35)
    throw new Error(`expected 35 pose sliders (6 root + 29), got ${sliderCount}`);
  const fk0 = await page.$eval("#pose .fk", (el) => el.textContent);
  const setSlider = (idx, frac) => page.evaluate(({ idx, frac }) => {
    const r = document.querySelectorAll("#pose input[type=range]")[idx];
    const v = +r.min + frac * (+r.max - +r.min);
    Object.getOwnPropertyDescriptor(HTMLInputElement.prototype, "value")
      .set.call(r, v);
    r.dispatchEvent(new Event("input", { bubbles: true }));
  }, { idx, frac });
  await setSlider(9, 0.85);   // L arm pitch well up
  await setSlider(25, 0.6);   // R elbow flexed
  await setSlider(23, 0.9);   // R arm yaw swung out
  await setSlider(19, 0.7);   // L knee bent (the legs are real now)
  await setSlider(4, 0.45);   // root pitch leaned
  await new Promise((r) => setTimeout(r, 400));
  const fk1 = await page.$eval("#pose .fk", (el) => el.textContent);
  if (fk0 === fk1) throw new Error("pose sliders did not move the FK readout");
  await page.screenshot({ path: "test/out/pose.png" });
  await page.keyboard.press("p"); // leave pose mode
  await page.waitForFunction(() => !document.querySelector("#pose"),
    { timeout: 5000 });

  const input = await page.$("input[type=file]");
  await input.uploadFile(replay);
  await page.waitForFunction(
    () => document.querySelector("#hud").textContent.includes("tick"),
    { timeout: 10000 });
  await new Promise((r) => setTimeout(r, 1500)); // let it play a bit
  await page.screenshot({ path: "test/out/playing.png" });

  // ---- skeleton + gauges + contact-marker eyeball shots ('h' hitboxes on):
  // scrub to a mid-episode tick (the bot swinging, weight on the welds —
  // gauges show τ bars, foot N and effort) and to a late tick.
  await page.keyboard.press("h");
  const scrub = (tick) => page.evaluate((tick) => {
    const r = document.querySelector("#bottom input[type=range]");
    Object.getOwnPropertyDescriptor(HTMLInputElement.prototype, "value")
      .set.call(r, tick);
    r.dispatchEvent(new Event("input", { bubbles: true }));
  }, tick);
  await scrub(100);
  await new Promise((r) => setTimeout(r, 500));
  await page.screenshot({ path: "test/out/skeleton-gauges.png" });
  await scrub(300);
  await new Promise((r) => setTimeout(r, 500));
  await page.screenshot({ path: "test/out/late.png" });

  const hud = await page.$eval("#hud", (el) => el.textContent);
  console.log("HUD:", hud.replace(/\s+/g, " ").trim());
  if (!/t\s+[1-9]?\d\.\d+s/.test(hud)) throw new Error("time not advancing");

  // ---- gauges button: mirrors the 'g' key (the panels are on at load)
  const GAUGES_ON = `[...document.querySelectorAll("#bar button")]
    .some((b) => b.textContent.includes("gauges") && b.classList.contains("on"))`;
  if (!(await page.evaluate(GAUGES_ON))) throw new Error("gauges button not lit at load");
  await page.evaluate(() => [...document.querySelectorAll("#bar button")]
    .find((b) => b.textContent.includes("gauges")).click());
  await page.waitForFunction(`!(${GAUGES_ON})`, { timeout: 5000 });
  await page.keyboard.press("g"); // the key toggles the same flag back on
  await page.waitForFunction(GAUGES_ON, { timeout: 5000 });

  // ---- first-person ('f'): the camera rides the head frame. The bar
  // button lights up, the frame renders with no page errors, and 'f'
  // again returns to the orbit camera.
  // (a string expression: waitForFunction cannot be handed a closure)
  const FP_ON = `[...document.querySelectorAll("#bar button")]
    .some((b) => b.textContent.includes("first-person")
      && b.classList.contains("on"))`;
  await page.keyboard.press("f");
  await page.waitForFunction(FP_ON, { timeout: 5000 });
  await new Promise((r) => setTimeout(r, 500));
  await page.screenshot({ path: "test/out/first-person.png" });
  // the two first-person sliders (fov, rotation smoothing) live in #fp
  const fpSliders = await page.$$eval("#fp input[type=range]", (els) => els.length);
  if (fpSliders !== 2) throw new Error(`expected 2 first-person sliders, got ${fpSliders}`);
  const setFp = (idx, frac) => page.evaluate(({ idx, frac }) => {
    const r = document.querySelectorAll("#fp input[type=range]")[idx];
    const v = +r.min + frac * (+r.max - +r.min);
    Object.getOwnPropertyDescriptor(HTMLInputElement.prototype, "value")
      .set.call(r, v);
    r.dispatchEvent(new Event("input", { bubbles: true }));
  }, { idx, frac });
  await setFp(0, 1.0);  // fov 110°
  await setFp(1, 0.4);  // smoothing 0.4 s
  await new Promise((r) => setTimeout(r, 500));
  const fpLabels = await page.$eval("#fp", (el) => el.textContent.replace(/\s+/g, " "));
  if (!/110°/.test(fpLabels) || !/0\.40s/.test(fpLabels))
    throw new Error("first-person sliders did not take: " + fpLabels);
  await page.screenshot({ path: "test/out/first-person-wide-smooth.png" });
  await page.keyboard.press("f");
  await page.waitForFunction(`!(${FP_ON})`, { timeout: 5000 });
  await new Promise((r) => setTimeout(r, 300));
  await page.screenshot({ path: "test/out/after-first-person.png" });

  // ---- env v15 head-anchored track: same bundle, a replay whose header
  // carries "track_anchor": "head". The notes and the tunnel/grid/reach
  // scenery ride head_z - PLANE_Z0, so a tick where the bot has moved in z
  // shows the whole track displaced while the floor stays put.
  if (existsSync(replayV15)) {
    await input.uploadFile(replayV15);
    await page.waitForFunction(
      () => document.querySelector("#drop").textContent.includes("v15"),
      { timeout: 10000 });
    await new Promise((r) => setTimeout(r, 800));
    await scrub(60);
    await new Promise((r) => setTimeout(r, 500));
    await page.screenshot({ path: "test/out/v15-head-anchored.png" });
    const hud15 = await page.$eval("#hud", (el) => el.textContent);
    console.log("HUD (v15):", hud15.replace(/\s+/g, " ").trim());
    if (!/tick\s*60\b/.test(hud15))
      throw new Error("v15 replay did not scrub");
  } else {
    console.log("no v15 recording at " + replayV15 + " — skipped");
  }

  const webgl = await page.evaluate(() => {
    const cv = document.querySelector("#stage canvas");
    return !!(cv.getContext("webgl2") || cv.getContext("webgl"));
  });
  if (!webgl) throw new Error("no WebGL context on the stage canvas");
  if (errors.length) throw new Error("browser errors:\n" + errors.join("\n"));
  console.log("browser smoke test OK (screenshots in test/out/)");
} finally {
  await browser.close();
}
