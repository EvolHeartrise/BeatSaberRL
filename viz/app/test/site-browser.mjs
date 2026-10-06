// Headless-browser check of the SITE build (npm run build:site, deployed at
// https://replays.flee.systems) on a served viewer — the hosted one or
// `wrangler dev` in viz/replays-site:
//   - a ?replay= load shows the map card and NOTHING at the top of the page
//     (no title line), the viewer knows the replay's name (state + the watch
//     button), and there is no file-picker button / file input, no "play
//     Flux", no pose button and no map-info toggle; 'p' opens no pose panel;
//   - a Flux replay (header "game": "flux") is refused with a clear message
//     and nothing renders.
// Playback itself (real time, song) is test/urlload-browser.mjs's job.
// Run: node test/site-browser.mjs <base-url> </r/replay.rpl.gz> [</r/flux.rpl.gz>]
// (the Flux file: a gzipped Flux JSONL served under a .rpl.gz name)
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
const [base, path, fluxPath] = process.argv.slice(2);
if (!base || !path) throw new Error("usage: site-browser.mjs <base-url> </r/file.rpl.gz> [</r/flux.rpl.gz>]");
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
  const errors = [];
  page.on("pageerror", (e) => errors.push("pageerror: " + e.message));
  page.on("console", (m) => { if (m.type() === "error") errors.push("console: " + m.text()); });

  // ---- an agent replay: map card, nothing up top, no picker / Flux / pose ---
  await page.goto(url(path));
  await page.waitForSelector("#watch", { timeout: 120000 });
  const ui = await page.evaluate(() => {
    // accessible names: the bar's icon buttons carry aria-labels
    const buttons = [...document.querySelectorAll("button")]
      .map((b) => (b.getAttribute("aria-label") ?? b.textContent).trim());
    return {
      name: window.__viewerState().name,
      watch: document.querySelector("#watch")?.textContent.replace(/\s+/g, " ").trim(),
      top: document.querySelector("#top")?.textContent.trim() ?? null,
      topKids: document.querySelector("#top")?.children.length ?? null,
      title: !!document.querySelector("#title"),
      drop: !!document.querySelector("#drop"),
      fileInputs: document.querySelectorAll("input[type=file]").length,
      buttons,
    };
  });
  console.log(`name (state): ${ui.name} · watch: ${ui.watch}`);
  if (!ui.name) fail("the viewer state has no replay name");
  if (!ui.watch?.includes(ui.name)) fail("the watch button does not name the replay");
  if (ui.title || ui.top || ui.topKids) fail("the top of the page must be empty: " + JSON.stringify(ui.top));
  if (ui.drop || ui.fileInputs) fail("the site build still has the file picker");
  const bad = ui.buttons.filter((b) => /play Flux|pose|map info|choose/.test(b));
  if (bad.length) fail("unexpected buttons: " + bad.join(" | "));
  console.log(`buttons: ${ui.buttons.join(" | ")}`);
  const card = await page.waitForSelector("#mapcard", { timeout: 30000 })
    .then((el) => el.evaluate((e) => e.textContent.replace(/\s+/g, " ").trim()), () => null);
  console.log(`map card: ${card ?? "(none)"}`);
  if (!card) fail("a replay that names a map must show the map card");
  await page.keyboard.press("p");
  await sleep(300);
  if (await page.$("#pose")) fail("'p' opened the pose panel");
  await page.screenshot({ path: "test/out/site-loaded.png" });

  // ---- a Flux replay is refused -----------------------------------------------
  if (fluxPath) {
    await page.goto(url(fluxPath));
    await page.waitForSelector("#urlerror", { timeout: 60000 });
    const msg = await page.$eval("#urlerror", (el) => el.textContent.replace(/\s+/g, " ").trim());
    const vs = await page.evaluate(() => window.__viewerState());
    console.log(`${fluxPath} -> ${msg} · viewer: ${JSON.stringify(vs)}`);
    if (!/Flux replays aren't supported on this site/.test(msg)) fail("Flux replay: unexpected message");
    if (await page.$("#watch") || await page.$("#title") || vs.loaded)
      fail("a refused Flux replay must not open");
    await page.screenshot({ path: "test/out/site-flux-refused.png" });
  }

  if (errors.length) fail("page errors:\n" + errors.join("\n"));
  console.log("site browser check OK");
} finally {
  await browser.close();
}
