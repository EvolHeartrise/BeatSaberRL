// Headless-browser smoke test of the BUILT single-file viewer: loads
// dist/index.html in Edge, uploads a real replay through the file input,
// plays a second, and asserts the HUD advanced with zero page errors.
// Screenshots land in test/out/ for eyeball verification.
// Run: node test/browser.test.js [path-to-replay.jsonl]
import { existsSync, mkdirSync } from "node:fs";
import { resolve } from "node:path";
import puppeteer from "puppeteer-core";

const EDGE = "C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe";
const dist = resolve("dist/index.html");
const replay = resolve(process.argv[2] ??
  "../../python/replays/v7fix_ep00_seed7000.jsonl");
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
  // toggle 'p', drive two sliders, check the FK readout reacts, screenshot
  await page.keyboard.press("p");
  await page.waitForSelector("#pose", { timeout: 5000 });
  const fk0 = await page.$eval("#pose .fk", (el) => el.textContent);
  const setSlider = (idx, frac) => page.evaluate(({ idx, frac }) => {
    const r = document.querySelectorAll("#pose input[type=range]")[idx];
    const v = +r.min + frac * (+r.max - +r.min);
    Object.getOwnPropertyDescriptor(HTMLInputElement.prototype, "value")
      .set.call(r, v);
    r.dispatchEvent(new Event("input", { bubbles: true }));
  }, { idx, frac });
  await setSlider(0, 0.85);  // L pitch well up
  await setSlider(9, 0.6);   // R elbow flexed (index 9 = right hand elbow)
  await setSlider(7, 0.9);   // R yaw swung out
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
