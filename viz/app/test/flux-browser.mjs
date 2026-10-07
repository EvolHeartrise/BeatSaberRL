// Headless-browser check of the BUILT viewer on a Flux replay: loads the
// scripted Flux fixture, pauses, scrubs to a few ticks (with and without
// the 'h' hitbox overlay) and screenshots each into test/out/flux-*.png;
// fails on any page error. Run after `npm run build`:
//   node test/flux-browser.mjs [path-to-flux-replay.jsonl]
import { existsSync, mkdirSync, readdirSync } from "node:fs";
import { homedir } from "node:os";
import { resolve } from "node:path";
import puppeteer from "puppeteer-core";
import { clickBar, scrubTo } from "./bar.js";

function findBrowser() {
  if (process.env.PUPPETEER_EXECUTABLE_PATH) return process.env.PUPPETEER_EXECUTABLE_PATH;
  const root = resolve(homedir(), ".cache/puppeteer/chrome");
  for (const v of existsSync(root) ? readdirSync(root).sort().reverse() : []) {
    const p = resolve(root, v, "chrome-linux64/chrome");
    if (existsSync(p)) return p;
  }
  throw new Error("no browser found: set PUPPETEER_EXECUTABLE_PATH");
}
const dist = resolve("dist/index.html");
const replay = resolve(process.argv[2] ?? "test/fixtures/flux_scripted.jsonl");
if (!existsSync(dist)) throw new Error("run `npm run build` first");
mkdirSync("test/out", { recursive: true });

const browser = await puppeteer.launch({
  executablePath: findBrowser(), headless: true,
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
  await page.goto("file:///" + dist);
  await page.waitForSelector("#stage canvas", { timeout: 15000 });
  const input = await page.$("input[type=file]");
  await input.uploadFile(replay);
  await page.waitForFunction(
    () => window.__viewerState?.().loaded,
    { timeout: 10000 });
  const kind = await page.evaluate(() => window.__viewerState());
  if (!kind.flux) throw new Error("not opened as a Flux replay: " + JSON.stringify(kind));
  await clickBar(page, "pause");
  // an agent's Flux run logs its torque: the timeline graphs it
  const graph = await page.$eval("#seekbar", (e) => e.dataset.graph);
  if (graph !== "torque") throw new Error("an agent Flux run's timeline should graph torque: " + graph);
  // the timeline: a pointer press on its track + arrow keys (test/bar.js)
  const scrub = (tick) => scrubTo(page, tick);
  const shots = [[12, "approach"], [48, "mid"], [83, "bar"], [140, "late"]];
  for (const [tick, name] of shots) {
    await scrub(tick);
    await new Promise((r) => setTimeout(r, 600));
    await page.screenshot({ path: `test/out/flux-${name}.png` });
  }
  await page.keyboard.press("h");
  await scrub(83);
  await new Promise((r) => setTimeout(r, 600));
  await page.screenshot({ path: "test/out/flux-hitboxes.png" });
  const vs = await page.evaluate(() => window.__viewerState());
  console.log("viewer:", JSON.stringify(vs));
  if (vs.tick !== 83) throw new Error("did not scrub to tick 83");
  if (errors.length) throw new Error("page errors:\n" + errors.join("\n"));
  console.log("flux browser OK");
} finally {
  await browser.close();
}
