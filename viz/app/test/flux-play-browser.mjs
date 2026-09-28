// Headless-browser check of the Flux WebXR game in the BUILT app, without a
// headset: enter play mode from the viewer bar, pick a chart, start it, and
// let a scripted "player" (window.__fluxPlay.setScript — poses fed exactly
// where the XR read feeds them, FluxPlay.svelte) clear it at 4x clock
// speed. Checks the run ends with every object judged, the in-page results
// panel, "watch run" reopening it in the replay viewer as a human Flux run,
// and no page errors. Screenshots -> test/out/flux-play-*.png.
// Run after `npm run build`: node test/flux-play-browser.mjs
import { existsSync, mkdirSync, readdirSync } from "node:fs";
import { homedir } from "node:os";
import { resolve } from "node:path";
import puppeteer from "puppeteer-core";

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
if (!existsSync(dist)) throw new Error("run `npm run build` first");
mkdirSync("test/out", { recursive: true });

const browser = await puppeteer.launch({
  executablePath: findBrowser(), headless: true,
  args: ["--use-angle=swiftshader", "--disable-gpu-sandbox", "--no-sandbox"],
});
const click = (page, sel, text) => page.evaluate((sel, text) => {
  const b = [...document.querySelectorAll(sel)].find((e) => e.textContent.includes(text));
  if (!b) throw new Error(`no ${sel} with "${text}"`);
  b.click();
}, sel, text);
try {
  const page = await browser.newPage();
  await page.setViewport({ width: 1100, height: 800 });
  const errors = [];
  page.on("pageerror", (e) => errors.push("pageerror: " + e.message));
  page.on("console", (m) => {
    if (m.type() === "error") errors.push("console: " + m.text());
  });
  await page.goto("file:///" + dist);
  await page.waitForSelector("#stage canvas", { timeout: 15000 });

  await click(page, "#bar button", "play Flux");
  await page.waitForSelector("#play select", { timeout: 10000 });
  const options = await page.$$eval("#play select option", (os) => os.map((o) => o.value));
  if (!options.includes("warmup-1") || options.length < 6)
    throw new Error("chart list: " + options.join(","));
  await page.select("#play select", "warmup-1");

  // the scripted player: each palm on the next circle of its colour as it
  // arrives, the Flux square across each bar through its sweet spot
  await page.evaluate(() => {
    const P = window.__fluxPlay;
    P.setClockScale(4);
    const REST = [[-0.25, 1.1, -0.4], [0.25, 1.1, -0.4]];
    P.setScript((t) => {
      const chart = P.session()?.chart ?? [];
      const palms = [0, 1].map((h) => {
        const o = chart.find((o) => o.t >= t - 0.02 &&
          ((o.kind === 4 && o.color === h) || o.kind === 5));
        if (!o) return REST[h];
        if (o.kind === 4) return [o.x, o.y, -0.35];
        const ux = -Math.sin(o.rot), uy = Math.cos(o.rot);
        const s = o.sweet * 0.5 * o.len;
        const sx = o.x + s * ux, sy = o.y + s * uy;
        const nx = uy, ny = -ux;
        const d = (h === 0 ? -1 : 1) * 0.3 * (nx < 0 ? -1 : 1);
        return [sx + d * nx, sy + d * ny, -0.35];
      });
      return { palms, head: [0, 1.7, -0.65], quat: [0, 0, 0, 1] };
    });
  });
  await page.screenshot({ path: "test/out/flux-play-menu.png" });
  await click(page, "#play button", "start");
  await page.waitForFunction(() => window.__fluxPlay.play.phase === "playing",
    { timeout: 20000 });
  await page.waitForFunction(() => window.__fluxPlay.play.time > 4, { timeout: 60000 });
  await page.screenshot({ path: "test/out/flux-play-mid.png" });
  await page.waitForFunction(() => window.__fluxPlay.play.phase === "done",
    { timeout: 120000 });
  await page.screenshot({ path: "test/out/flux-play-done.png" });

  const st = await page.evaluate(() => ({ ...window.__fluxPlay.play.stats }));
  if (st.good + st.bad + st.misses !== st.n_objects)
    throw new Error("not every object judged: " + JSON.stringify(st));
  if (!(st.good > 0.5 * st.n_objects))
    throw new Error("the scripted player should clear most of it: " + JSON.stringify(st));
  const result = await page.$eval("#play .result", (e) => e.textContent);
  if (!result.includes("watch run")) throw new Error("no results panel: " + result);

  // the run reopens in the replay viewer as a human Flux run
  await click(page, "#play button", "watch run");
  await page.waitForFunction(
    () => document.querySelector("#drop")?.textContent.includes("played in VR"),
    { timeout: 10000 });
  await page.evaluate(() => [...document.querySelectorAll("#bar button")]
    .find((b) => b.textContent.includes("pause"))?.click());
  await new Promise((r) => setTimeout(r, 800));
  await page.screenshot({ path: "test/out/flux-play-watch.png" });
  const hud = await page.$eval("#hud", (e) => e.textContent);
  if (!hud.includes(`good ${st.good}`) && !hud.includes("good"))
    throw new Error("viewer HUD: " + hud);

  if (errors.length) throw new Error(errors.join("\n"));
  console.log(`flux-play-browser OK: ${st.n_objects} objects, good ${st.good}, ` +
    `bad ${st.bad}, miss ${st.misses}, score ${st.score}`);
} finally {
  await browser.close();
}
