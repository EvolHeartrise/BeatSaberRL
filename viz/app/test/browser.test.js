// Headless-browser smoke test of the BUILT single-file viewer: loads
// dist/index.html in Edge, uploads a real v10 replay through the file
// input, plays a second, and asserts the HUD advanced with zero page
// errors. Also drives the v10 manual-pose panel (root + 29 joints) and
// screenshots the articulated skeleton + per-joint gauges/forces.
// Screenshots land in test/out/ for eyeball verification.
// Run: node test/browser.test.js [path-to-replay.jsonl]
import { existsSync, mkdirSync } from "node:fs";
import { resolve } from "node:path";
import puppeteer from "puppeteer-core";

const EDGE = "C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe";
const dist = resolve("dist/index.html");
const replay = resolve(process.argv[2] ??
  "../../python/replays/v10_scripted_ep00_seed0.jsonl");
if (!existsSync(dist)) throw new Error("run `npm run build` first");
mkdirSync("test/out", { recursive: true });

const browser = await puppeteer.launch({
  executablePath: EDGE,
  headless: true,
  args: ["--use-angle=swiftshader", "--disable-gpu-sandbox"],
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
