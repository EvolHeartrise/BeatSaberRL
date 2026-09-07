// Close-up screenshots of the models in the built viewer: load a replay,
// scrub to given ticks, place the camera, screenshot. Usage:
//   node closeup.mjs <dist/index.html> <replay.jsonl> <outdir> tick[,tick...]
import { resolve } from "node:path";
import { mkdirSync, existsSync, readdirSync } from "node:fs";
import { homedir } from "node:os";
import puppeteer from "puppeteer-core";
const [dist, replay, outdir, ticksArg] = process.argv.slice(2);
mkdirSync(outdir, { recursive: true });
const root = resolve(homedir(), ".cache/puppeteer/chrome");
const chrome = readdirSync(root).sort().reverse().map((v) => resolve(root, v, "chrome-linux64/chrome")).find(existsSync);
const browser = await puppeteer.launch({ executablePath: chrome, headless: true, args: ["--use-angle=swiftshader", "--disable-gpu-sandbox", "--no-sandbox"] });
const page = await browser.newPage();
await page.setViewport({ width: 1200, height: 900 });
const errors = [];
page.on("pageerror", (e) => errors.push(e.message));
await page.goto("file://" + resolve(dist));
await page.waitForSelector("#stage canvas", { timeout: 15000 });
const input = await page.$("input[type=file]");
await input.uploadFile(resolve(replay));
await page.waitForFunction(() => /tick/i.test(document.body.innerText), { timeout: 15000 });
await page.keyboard.press(" "); // pause if autoplay
await page.keyboard.press("g"); // hide the gauge panels
if (process.env.HITBOX) await page.keyboard.press("h"); // hitbox mode
for (const t of ticksArg.split(",")) {
  const [tick, view] = t.split("@");
  await page.evaluate((tick) => { window.__pb && (window.__pb.cur = +tick); }, tick);
  // fall back: scrub via the range input if the store is not exposed
  await page.evaluate((tick) => {
    const r = document.querySelector("input[type=range]");
    if (r) { Object.getOwnPropertyDescriptor(HTMLInputElement.prototype, "value").set.call(r, tick); r.dispatchEvent(new Event("input", { bubbles: true })); }
  }, tick);
  // zoom in a little (OrbitControls listens for wheel on the canvas)
  const canvas = await page.$("#stage canvas");
  const box = await canvas.boundingBox();
  await page.mouse.move(box.x + box.width / 2, box.y + box.height / 2);
  for (let i = 0; i < (+(view ?? 0)); i++) await page.mouse.wheel({ deltaY: -120 });
  await new Promise((r) => setTimeout(r, 600));
  await page.screenshot({ path: `${outdir}/tick${tick}${view ? "-" + view : ""}.png` });
}
console.log("errors:", errors);
await browser.close();
