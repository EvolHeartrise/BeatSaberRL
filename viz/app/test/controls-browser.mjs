// Headless-browser check of the replay viewer's control bar (src/lib/
// ControlBar.svelte — shadcn-svelte + Lucide icons) on the BUILT viewer
// (dist/index.html, npm run build):
//   - every bar button has an accessible name; hovering one shows a tooltip
//     with its name and keyboard shortcut;
//   - toggles report on/off (aria-pressed) and follow their keys ('g', 'f',
//     't'); the tip-trail length slider shows only while the trail is on; the
//     trail's switch and length survive a reload (localStorage);
//   - the timeline: Home / End / arrows on its thumb (one tick per press —
//     the window's own arrow-key stepping must not add a second), a pointer
//     drag pauses and seeks, and the thumb follows the clock while playing;
//   - play/pause, restart and the speed buttons do what their keys do;
//   - Tailwind does not leak: no preflight reset reaches the gauges overlay;
//   - at phone width (390 px) nothing overflows horizontally and the view
//     toggles live in a "more controls" popover that still drives them.
// Screenshots land in test/out/controls-*.png.
// Run: node test/controls-browser.mjs [replay.rpl.gz]
// (one Chrome, SwiftShader — keep it to one run at a time on the laptop)
import { existsSync, mkdirSync, readdirSync } from "node:fs";
import { homedir } from "node:os";
import { resolve } from "node:path";
import puppeteer from "puppeteer-core";
import { barButton, barLabels, clickBar, toggleOn } from "./bar.js";

function findBrowser() {
  if (process.env.PUPPETEER_EXECUTABLE_PATH) return process.env.PUPPETEER_EXECUTABLE_PATH;
  const edge = "C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe";
  if (existsSync(edge)) return edge;
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
if (!existsSync(dist)) throw new Error("build first: npm run build");
mkdirSync("test/out", { recursive: true });
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
const fail = (m) => { throw new Error(m); };

const browser = await puppeteer.launch({
  executablePath: findBrowser(), headless: true,
  args: ["--use-angle=swiftshader", "--disable-gpu-sandbox", "--no-sandbox"],
});
try {
  const page = await browser.newPage();
  await page.setViewport({ width: 1100, height: 800 });
  const errors = [];
  page.on("pageerror", (e) => errors.push("pageerror: " + e.message));
  page.on("console", (m) => { if (m.type() === "error") errors.push("console: " + m.text()); });
  page.on("dialog", (d) => { errors.push("dialog: " + d.message()); d.dismiss(); });
  const state = () => page.evaluate(() => window.__viewerState());
  const pressed = (label) => page.evaluate(toggleOn(label));
  const load = async () => {
    await page.goto("file:///" + dist.replace(/\\/g, "/"));
    await page.waitForSelector("#stage canvas", { timeout: 15000 });
    await (await page.$("input[type=file]")).uploadFile(replay);
    await page.waitForFunction(() => window.__viewerState?.().loaded, { timeout: 20000 });
  };
  await load();

  // ---- names + tooltips
  const labels = await barLabels(page);
  console.log("bar:", labels.join(" | "));
  for (const want of ["pause", "restart", "slower", "faster", "play Flux", "pose", "gauges",
    "first-person", "tip trail", "sound"])
    if (!labels.includes(want)) fail(`no "${want}" control in the bar: ${labels.join(" | ")}`);
  if (labels.some((l) => !l)) fail("a bar button has no accessible name");
  await page.hover(barButton("gauges"));
  const tip = await page.waitForSelector("[data-slot=tooltip-content]", { timeout: 5000 })
    .then((el) => el.evaluate((e) => ({
      text: e.textContent.replace(/\s+/g, " ").trim(),
      kbd: [...e.querySelectorAll("kbd")].map((k) => k.textContent.trim()),
    })));
  console.log("tooltip:", JSON.stringify(tip));
  if (!tip.text.startsWith("gauges") || !tip.kbd.includes("G"))
    fail("the gauges tooltip should name it and its key: " + JSON.stringify(tip));
  await page.screenshot({ path: "test/out/controls-tooltip.png" });
  await page.mouse.move(550, 200);

  // ---- toggles: aria-pressed follows the keys
  for (const [label, key] of [["gauges", "g"], ["first-person", "f"], ["tip trail", "t"]]) {
    const before = await pressed(label);
    await page.keyboard.press(key);
    await page.waitForFunction(`(${toggleOn(label)}) === ${!before}`, { timeout: 3000 })
      .catch(() => fail(`'${key}' did not flip the ${label} toggle`));
    await clickBar(page, label);
    await page.waitForFunction(`(${toggleOn(label)}) === ${before}`, { timeout: 3000 })
      .catch(() => fail(`clicking ${label} did not flip it back`));
  }
  // the trail's length slider: only while the trail is on
  if (!(await pressed("tip trail"))) fail("the tip trail should be on by default");
  if (!(await page.$("#tt [role=slider]"))) fail("no trail-length slider while the trail is on");
  if (await page.$("#fp [role=slider]")) fail("a fov slider without first-person");
  await page.focus("#tt [role=slider]");
  await page.keyboard.press("End");
  const tt = await page.$eval("#tt", (e) => e.textContent.replace(/\s+/g, " "));
  if (!/0\.50 s/.test(tt)) fail("the trail length did not take End: " + tt);
  await page.keyboard.press("t"); // keys on the trail's own slider stay there…
  if (!(await pressed("tip trail"))) fail("'t' on the length slider toggled the trail");
  await page.evaluate(() => document.activeElement.blur());
  await page.keyboard.press("t"); // …anywhere else they toggle it
  await page.waitForFunction(`!(${toggleOn("tip trail")})`, { timeout: 3000 });
  if (await page.$("#tt")) fail("the trail-length slider is still shown with the trail off");
  const saved = await page.evaluate(() => localStorage.getItem("beat-arena.tipTrail"));
  if (saved !== JSON.stringify({ on: false, s: 0.5 })) fail("tip trail not remembered: " + saved);
  await load(); // a reload keeps both
  if (await pressed("tip trail")) fail("the trail came back on after a reload");
  await page.keyboard.press("t");
  await page.waitForSelector("#tt", { timeout: 3000 });
  const tt2 = await page.$eval("#tt", (e) => e.textContent.replace(/\s+/g, " "));
  if (!/0\.50 s/.test(tt2)) fail("the trail length was not remembered: " + tt2);

  // ---- play / pause / restart / speed
  if (!(await state()).playing) fail("a dropped replay should play");
  await clickBar(page, "pause");
  if ((await state()).playing) fail("pause did not pause");
  if (!(await page.$(barButton("play")))) fail("the paused button should read play");
  await clickBar(page, "play");
  if (!(await state()).playing) fail("play did not play");
  const speed = () => page.$eval("#speed", (e) => e.textContent.trim());
  await clickBar(page, "faster");
  if ((await speed()) !== "2.0×") fail("faster: " + (await speed()));
  await clickBar(page, "slower"); await clickBar(page, "slower");
  if ((await speed()) !== "0.5×") fail("slower: " + (await speed()));
  await page.keyboard.press("+");
  if ((await speed()) !== "1.0×") fail("'+': " + (await speed()));

  // ---- the timeline: follows playback, keyboard, pointer drag
  const thumb = "#seek [role=slider]";
  const now = () => page.$eval(thumb, (t) => +t.getAttribute("aria-valuenow"));
  await sleep(300);
  const v0 = await now(); await sleep(800); const v1 = await now();
  if (!(v1 > v0)) fail(`the timeline does not follow playback: ${v0} -> ${v1}`);
  const t0 = (await state()).tick;
  if (Math.abs(v1 - t0) > 3) fail(`timeline ${v1} vs viewer tick ${t0}`);
  await page.focus(thumb);
  await page.keyboard.press("Home");
  let s = await state();
  if (s.tick !== 0 || s.playing) fail("Home should pause on tick 0: " + JSON.stringify(s));
  for (let i = 0; i < 3; i++) await page.keyboard.press("ArrowRight");
  s = await state();
  if (s.tick !== 3 || (await now()) !== 3)
    fail(`three arrow presses on the timeline should land on tick 3, got ${s.tick}`);
  await page.keyboard.press("ArrowLeft");
  if ((await state()).tick !== 2) fail("ArrowLeft on the timeline is not one tick back");
  await page.keyboard.press("End");
  s = await state();
  if (s.tick !== s.lastTick) fail("End should land on the last tick: " + JSON.stringify(s));
  await page.keyboard.press(" "); // space still plays/pauses from the timeline
  if (!(await state()).playing) fail("space on the timeline did not toggle playback");
  await page.evaluate(() => document.activeElement.blur());
  const box = await page.$eval("#seek [data-slot=slider]", (e) => {
    const r = e.getBoundingClientRect();
    return { x: r.left, y: r.top + r.height / 2, w: r.width };
  });
  await page.mouse.move(box.x + 0.2 * box.w, box.y);
  await page.mouse.down();
  for (const f of [0.3, 0.4, 0.5]) {
    await page.mouse.move(box.x + f * box.w, box.y, { steps: 4 });
    await sleep(60);
    const mid = await state();
    if (mid.playing) fail("dragging the timeline should pause");
    if (Math.abs(mid.tick / mid.lastTick - f) > 0.03)
      fail(`drag at ${f}: viewer at tick ${mid.tick} of ${mid.lastTick}`);
  }
  await page.mouse.up();
  await page.mouse.move(550, 200);
  await sleep(300);
  await page.screenshot({ path: "test/out/controls-desktop.png" });

  // restart: tick 0 and playing (from the paused drag)
  await clickBar(page, "restart");
  s = await state();
  if (!s.playing || s.tick > 5) fail("restart: " + JSON.stringify(s));

  // ---- no preflight leak: the gauges overlay keeps browser-default styling
  // (preflight would make svgs vertical-align: middle and every box
  // border-box)
  const leak = await page.evaluate(() => {
    const svg = document.querySelector(".gauges svg");
    const panel = document.querySelector(".gauges .panel");
    return svg && { svg: getComputedStyle(svg).verticalAlign,
      box: getComputedStyle(panel).boxSizing };
  });
  if (!leak) fail("no gauges overlay to check");
  if (leak.svg !== "baseline" || leak.box !== "content-box")
    fail("Tailwind preflight reached the gauges overlay: " + JSON.stringify(leak));

  // ---- phone width
  await page.setViewport({ width: 390, height: 844, deviceScaleFactor: 2, isMobile: true,
    hasTouch: true });
  // (switching to touch emulation reloads the page: open the replay again)
  await load();
  await sleep(600);
  const phone = await page.evaluate(() => {
    const r = document.querySelector("#bar").getBoundingClientRect();
    return { scrollW: document.documentElement.scrollWidth, w: innerWidth,
      left: r.left, right: r.right, h: r.height };
  });
  const phoneLabels = await barLabels(page);
  console.log("phone:", JSON.stringify(phone), phoneLabels.join(" | "));
  for (const want of ["restart", "slower", "faster", "sound", "more controls"])
    if (!phoneLabels.includes(want)) fail(`no "${want}" in the phone bar: ${phoneLabels.join(" | ")}`);
  if (phone.scrollW > phone.w || phone.left < 0 || phone.right > phone.w)
    fail("the bar overflows at phone width: " + JSON.stringify(phone));
  if (await page.$(barButton("gauges"))) fail("view toggles still inline at phone width");
  await page.tap(barButton("more controls"));
  await page.waitForSelector("#more", { timeout: 3000 });
  const more = await page.$$eval("#more button", (bs) => bs.map((b) => b.getAttribute("aria-label")));
  for (const want of ["play Flux", "pose", "gauges", "first-person", "tip trail"])
    if (!more.includes(want)) fail(`no ${want} in the overflow menu: ${more.join(" | ")}`);
  const g0 = await page.$eval('#more [aria-label="gauges"]', (b) => b.getAttribute("aria-pressed"));
  await page.tap('#more [aria-label="gauges"]');
  await page.waitForFunction((g0) => document.querySelector('#more [aria-label="gauges"]')
    ?.getAttribute("aria-pressed") !== g0, { timeout: 3000 }, g0)
    .catch(() => fail("the overflow menu's gauges toggle does nothing"));
  if (!!(await page.$(".gauges")) === (g0 === "true")) fail("gauges overlay did not follow the menu");
  const popW = await page.evaluate(() => document.documentElement.scrollWidth);
  if (popW > 390) fail("the open overflow menu overflows: " + popW);
  await page.screenshot({ path: "test/out/controls-phone-more.png" });

  if (errors.length) fail("page errors:\n  " + errors.join("\n  "));
  console.log("PASS — control bar: names, tooltips, toggles + persistence, timeline, phone");
} finally {
  await browser.close();
}
