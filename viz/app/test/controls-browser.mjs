// Headless-browser check of the replay viewer's control bar (src/lib/
// ControlBar.svelte — shadcn-svelte + Lucide icons) on the BUILT viewer
// (dist/index.html, npm run build):
//   - every bar button has an accessible name; hovering one shows a tooltip
//     with its name and keyboard shortcut;
//   - toggles report on/off (aria-pressed) and follow their keys ('g', 'f',
//     't'); the tip-trail length slider shows only while the trail is on; the
//     trail's switch and length survive a reload (localStorage);
//   - the timeline: Home / End / arrows on it (one tick per press — the
//     window's own arrow-key stepping must not add a second), a pointer
//     drag pauses and seeks, and the playhead follows the clock while
//     playing;
//   - the timeline's torque graph: painted from the replay's own per-tick
//     torque (every canvas column's height = that column's max at the p99
//     full scale), once per replay and size — never per frame — and redrawn
//     on resize; hovering names the time and that tick's torque;
//   - play/pause, restart and the speed buttons do what their keys do;
//   - Tailwind does not leak: no preflight reset reaches the gauges overlay;
//   - at phone width (390 px) nothing overflows horizontally and the view
//     toggles live in a "more controls" popover that still drives them; a
//     touch drag on the timeline seeks.
// Screenshots land in test/out/controls-*.png.
// Run: node test/controls-browser.mjs [replay.rpl.gz]
// (one Chrome, SwiftShader — keep it to one run at a time on the laptop)
import { existsSync, mkdirSync, readdirSync } from "node:fs";
import { homedir } from "node:os";
import { resolve } from "node:path";
import puppeteer from "puppeteer-core";
import { SEEK, barButton, barLabels, clickBar, toggleOn } from "./bar.js";
import { parseReplay } from "../src/lib/replay.js";
import { downsample, mistakeTicks, robustMax, torqueSeries } from "../src/lib/torquegraph.js";
import { readReplayFile } from "./fixture.js";

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
  const thumb = SEEK;
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
  // space still plays/pauses from the timeline (mid-replay: at the last tick
  // playback stops again on the next frame)
  await page.keyboard.press(" ");
  if (!(await state()).playing) fail("space on the timeline did not toggle playback");
  await page.keyboard.press(" ");
  await page.keyboard.press("End");
  s = await state();
  if (s.tick !== s.lastTick) fail("End should land on the last tick: " + JSON.stringify(s));
  await page.evaluate(() => document.activeElement.blur());
  const box = await page.$eval(SEEK, (e) => {
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

  // ---- the torque graph
  const repData = parseReplay(readReplayFile(replay));
  const series = torqueSeries(repData);
  if (!series) fail("the fixture should carry torque");
  const graph = () => page.evaluate(() => {
    const bar = document.querySelector("#seekbar");
    const dim = bar.querySelector("canvas");
    const W = dim.width, H = dim.height;
    const px = dim.getContext("2d").getImageData(0, 0, W, H).data;
    const tops = [];
    for (let x = 0; x < W; x++) {
      let y = 0;
      while (y < H && px[(y * W + x) * 4 + 3] === 0) y++;
      tops.push(y);
    }
    return { kind: bar.dataset.graph, draws: +bar.dataset.draws, top: +bar.dataset.top,
      cssW: bar.querySelector("[data-cols]").getBoundingClientRect().width, W, H, tops };
  });
  let g = await graph();
  if (g.kind !== "torque") fail("no torque graph on the timeline");
  const top = robustMax(series);
  if (Math.abs(g.top - top) > 0.06) fail(`graph full scale ${g.top} != p99 ${top}`);
  if (Math.abs(g.W - Math.round(g.cssW)) > 1) fail(`${g.W} columns for a ${g.cssW} px track`);
  const ds = downsample(series, g.W);
  const markX = new Set(mistakeTicks(repData).flatMap((k) => {
    const x = Math.round((k / (series.length - 1)) * (g.W - 1));
    return [x - 1, x, x + 1];
  }));
  let checked = 0, off = 0;
  for (let x = 0; x < g.W; x++) {
    if (markX.has(x)) continue;
    const h = Math.round(Math.min(1, ds.max[x] / top) * g.H);
    const want = ds.max[x] > top ? 0 : g.H - h;
    checked++;
    if (g.tops[x] !== want) off++;
  }
  console.log(`torque graph: ${g.W}x${g.H} px, p99 ${top.toFixed(1)} N·m, ` +
    `${checked - off}/${checked} columns at their max`);
  if (off > checked * 0.01) fail(`${off} of ${checked} graph columns off their data`);
  // playback moves the playhead only: no repaint
  await clickBar(page, "play");
  await sleep(1200);
  await clickBar(page, "pause");
  const g2 = await graph();
  if (g2.draws !== g.draws) fail(`the graph repainted during playback (${g.draws} -> ${g2.draws})`);
  // hover: the time and that tick's torque
  const bb = await page.$eval(SEEK, (e) => {
    const r = e.getBoundingClientRect();
    return { x: r.left, y: r.top + r.height / 2, w: r.width };
  });
  await page.mouse.move(bb.x + 0.25 * bb.w, bb.y);
  const tipText = await page.waitForSelector("#seektip", { timeout: 3000 })
    .then((el) => el.evaluate((e) => e.textContent.replace(/\s+/g, " ").trim()));
  const hk = Math.round(0.25 * (series.length - 1)), hs = Math.floor(hk / 30);
  const hWant = `${Math.floor(hs / 60)}:${String(hs % 60).padStart(2, "0")}`;
  console.log("hover:", tipText);
  if (!tipText.startsWith(hWant) || !tipText.includes(`${series[hk].toFixed(0)} N·m`)
      || !/spine \d+ · arms \d+ · legs \d+/.test(tipText))
    fail(`hover at tick ${hk} should read ${hWant} and ${series[hk].toFixed(0)} N·m: ${tipText}`);
  await page.screenshot({ path: "test/out/controls-seek-hover.png" });
  await page.mouse.move(550, 200);
  if (await page.$("#seektip")) fail("the hover readout outlived the hover");
  // a resize repaints at the new width
  await page.setViewport({ width: 760, height: 800 }); // (the bar caps at 52rem)
  await sleep(400);
  const g3 = await graph();
  if (g3.draws <= g2.draws || g3.W === g2.W) fail("the graph did not redraw on resize");
  await page.setViewport({ width: 1100, height: 800 });

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
  // a touch drag on the (compact) timeline seeks and pauses
  const tb = await page.$eval(SEEK, (e) => {
    const r = e.getBoundingClientRect();
    return { x: r.left, y: r.top + r.height / 2, w: r.width, h: r.height };
  });
  if (tb.h > 32) fail("the phone timeline should be compact: " + tb.h);
  await page.touchscreen.touchStart(tb.x + 0.3 * tb.w, tb.y);
  await page.touchscreen.touchMove(tb.x + 0.45 * tb.w, tb.y);
  await page.touchscreen.touchMove(tb.x + 0.6 * tb.w, tb.y);
  await page.touchscreen.touchEnd();
  s = await state();
  if (s.playing || Math.abs(s.tick / s.lastTick - 0.6) > 0.04)
    fail("a touch drag to 60% of the timeline: " + JSON.stringify(s));
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
