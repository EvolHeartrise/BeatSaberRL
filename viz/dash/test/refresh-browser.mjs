// Headless-browser check that the auto-refresh leaves the charts alone
// when nothing changed, and updates them when rows arrive:
//   - a fake archive (test/.archive, gitignored) with a live (partial) and
//     a finished run, served by serve.py --no-backfill on a free port;
//   - the page polls every 1 s (?refresh=1);
//   - a "backfill pass" that rewrites index.json with only a new
//     `generated` stamp (and rewrites metrics.jsonl byte-identically)
//     must cause zero redraws and keep a manual drag-zoom, on the Live
//     tab (with and without the window lock) and on Compare;
//   - appending rows must reach every chart (zoomed ones keep their
//     window, full-extent ones grow to the new tail) and the status line;
//   - a new run in the index (newer t_last) takes over "most recent";
//     picking a run in the selector switches the charts to it.
// Run: npm test  (= vite build && node test/refresh-browser.mjs)
// One Chrome, no WebGL — still, keep it to one run at a time on the laptop.
import { spawn } from "node:child_process";
import { appendFileSync, existsSync, mkdirSync, readdirSync, rmSync, writeFileSync, readFileSync } from "node:fs";
import { createServer } from "node:net";
import { homedir } from "node:os";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import puppeteer from "puppeteer-core";

const HERE = dirname(fileURLToPath(import.meta.url));
const DASH = resolve(HERE, "..");
const ARCH = resolve(HERE, ".archive");

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
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
let failures = 0;
function check(ok, msg) {
  console.log(`${ok ? "ok  " : "FAIL"} ${msg}`);
  if (!ok) failures++;
}

// ---------------------------------------------------------------- fixture
const BATCH = 1 << 20;
const T0 = 1790000000;
function rec(i) {
  const s = i / 300;
  return {
    upd: i, n_upd: 1000, steps: i * BATCH, t: T0 + i * 7, elapsed: i * 7, sps: 150000 + 1000 * Math.sin(i),
    acc: 0.5 + 0.3 * s, accL: 0.48 + 0.3 * s, accR: 0.52 + 0.3 * s, return: -0.1 + 0.2 * s,
    ent: 15 - s, entfull: 14 - s, ev: 0.9 + 0.05 * s, kl_first: 1e-6, kl_last: 0.04 + 0.01 * Math.cos(i),
    clipfrac_first: 0, clipfrac_last: 0.25, gnorm: 2.8, clipc: 0.23, falls: i % 4, tilts: i % 3,
  };
}
const jsonl = (a, b) => { let t = ""; for (let i = a; i <= b; i++) t += JSON.stringify(rec(i)) + "\n"; return t; };
function runEntry(id, n, status, tLast) {
  const [campaign, run] = id.split("/");
  return { id, campaign, run, run_dir: `${campaign}/runs/${run}`, metrics: `${campaign}/runs/${run}/metrics.jsonl`,
    status, samples: n * BATCH, updates: n, last_update: n, num_envs: 32768, horizon: 32, t_last: tLast, logs: [] };
}
let runs = [];
function writeIndex() {
  writeFileSync(resolve(ARCH, "index.json"), JSON.stringify({ generated: Date.now() / 1000, archive: ARCH, runs }, null, 0));
}
function writeRun(id, a, b) {
  const p = resolve(ARCH, id.split("/")[0], "runs", id.split("/")[1]);
  mkdirSync(p, { recursive: true });
  writeFileSync(resolve(p, "metrics.jsonl"), jsonl(a, b));
}
const metricsPath = (id) => resolve(ARCH, id.split("/")[0], "runs", id.split("/")[1], "metrics.jsonl");

rmSync(ARCH, { recursive: true, force: true });
mkdirSync(ARCH, { recursive: true });
writeRun("camp/live_run", 1, 300);
writeRun("camp/done_run", 1, 250);
runs = [runEntry("camp/live_run", 300, "partial", T0 + 2100), runEntry("camp/done_run", 250, "finished", T0 + 1750)];
writeIndex();

// "backfill pass" with nothing new: index.json gets a fresh `generated`,
// metrics files are rewritten byte-identically (new mtime, same bytes)
function noopPass() {
  writeIndex();
  for (const r of runs) {
    const p = metricsPath(r.id);
    writeFileSync(p, readFileSync(p));
  }
}

// ---------------------------------------------------------------- server
const port = await new Promise((res) => {
  const s = createServer();
  s.listen(0, "127.0.0.1", () => { const p = s.address().port; s.close(() => res(p)); });
});
if (!existsSync(resolve(DASH, "dist/index.html"))) throw new Error("dist/ missing: run `npm run build` (npm test does)");
const srv = spawn("python3", [resolve(DASH, "serve.py"), "--no-backfill", "--archive", ARCH, "--port", String(port)],
  { stdio: ["ignore", "inherit", "inherit"] });
const BASE = `http://127.0.0.1:${port}`;
for (let i = 0; i < 50; i++) {
  try { if ((await fetch(BASE + "/archive/index.json")).ok) break; } catch {}
  await sleep(100);
}

const browser = await puppeteer.launch({ executablePath: findBrowser(), headless: true, args: ["--no-sandbox", "--disable-gpu"] });
try {
  const page = await browser.newPage();
  await page.setViewport({ width: 1400, height: 1000 });
  page.on("pageerror", (e) => check(false, `page error: ${e.message}`));
  let polls = 0;
  page.on("request", (r) => { if (r.url().includes("index.json")) polls++; });

  // every chart's uPlot instance, its x scale, data length and draw count
  const snap = () => page.evaluate(() => [...document.querySelectorAll(".chart")].filter((el) => el.__uplot).map((el) => {
    const u = el.__uplot;
    if (u.__id == null) {
      u.__id = (window.__ids = (window.__ids || 0) + 1);
      u.__draws = 0;
      (u.hooks.draw ??= []).push(() => u.__draws++);
    }
    return { id: u.__id, title: u.root.querySelector(".u-title")?.textContent ?? "", min: u.scales.x.min, max: u.scales.x.max,
      n: u.data[0].length, last: u.data[0][u.data[0].length - 1], draws: u.__draws };
  }));
  const waitLen = (n, t = 8000) => page.waitForFunction((n) => {
    const us = [...document.querySelectorAll(".chart")].map((el) => el.__uplot).filter(Boolean);
    return us.length > 0 && us.every((u) => u.data[0].length === n);
  }, { timeout: t }, n);
  async function dragZoom(i, a, b) {
    const box = await page.evaluate((i) => {
      const el = [...document.querySelectorAll(".chart")].filter((e) => e.__uplot)[i];
      el.scrollIntoView({ block: "center" });
      const r = el.__uplot.over.getBoundingClientRect();
      return { x: r.left, y: r.top, w: r.width, h: r.height };
    }, i);
    const y = box.y + box.h / 2;
    await page.mouse.move(box.x + box.w * a, y);
    await page.mouse.down();
    await page.mouse.move(box.x + box.w * (a + b) / 2, y, { steps: 4 });
    await page.mouse.move(box.x + box.w * b, y, { steps: 4 });
    await page.mouse.up();
    await sleep(100);
  }
  const status = () => page.$eval(".status", (e) => e.textContent.replace(/·\s*[^·]*ago/, "").replace(/\s+/g, " ").trim()).catch(() => "");
  // run N no-op passes and wait for the page to poll after each
  async function idleRefreshes(n = 3) {
    for (let k = 0; k < n; k++) {
      const p0 = polls;
      noopPass();
      const t0 = Date.now();
      while (polls === p0 && Date.now() - t0 < 5000) await sleep(50);
      await sleep(400); // let the fetch + any reaction land
    }
  }
  function sameView(before, after, what) {
    check(before.length === after.length && before.length > 0, `${what}: ${after.length} charts still there`);
    const rebuilt = after.filter((a, i) => a.id !== before[i]?.id).length;
    const redrawn = after.filter((a, i) => a.draws !== before[i]?.draws).length;
    const moved = after.filter((a, i) => a.min !== before[i]?.min || a.max !== before[i]?.max).length;
    check(rebuilt === 0, `${what}: no chart rebuilt (${rebuilt})`);
    check(redrawn === 0, `${what}: no chart redrawn (${redrawn})`);
    check(moved === 0, `${what}: no x range changed (${moved})`);
  }

  // ---------------------------------------------------------- Live tab
  await page.goto(`${BASE}/?refresh=1#live`, { waitUntil: "load" });
  await waitLen(300);
  await snap();
  await dragZoom(0, 0.3, 0.6);
  let s0 = await snap();
  check(s0[0].min > 1 * BATCH && s0[0].max < 300 * BATCH, `live: drag-zoomed chart 0 to [${(s0[0].min / BATCH) | 0}, ${(s0[0].max / BATCH) | 0}] updates`);
  const st0 = await status();
  await idleRefreshes(3);
  check(polls >= 3, `live: page polled index.json (${polls}x)`);
  let s1 = await snap();
  sameView(s0, s1, "live, unchanged data");
  check((await status()) === st0, "live: status line unchanged (except the ago clock)");

  // rows arrive
  appendFileSync(metricsPath("camp/live_run"), jsonl(301, 305));
  runs[0] = runEntry("camp/live_run", 305, "partial", T0 + 2135);
  writeIndex();
  await waitLen(305);
  await sleep(300);
  let s2 = await snap();
  check(s2.every((c, i) => c.id === s1[i].id), "live, new rows: charts updated in place (not rebuilt)");
  check(s2.every((c) => c.n === 305 && c.last === 305 * BATCH), "live, new rows: every chart has the 5 new rows");
  check(s2[0].min === s1[0].min && s2[0].max === s1[0].max, "live, new rows: the zoomed chart kept its window");
  check(s2[0].draws > s1[0].draws, "live, new rows: the zoomed chart redrew (new points visible)");
  // uPlot syncs the x scale across the cursor-sync group: one drag zooms all
  check(s2.every((c, i) => c.min === s1[i].min && c.max === s1[i].max),"live, new rows: the synced charts kept the zoom too");
  check((await status()).includes("last update 305"), "live, new rows: status line shows update 305");

  // window lock: pin the last 100 M samples, then drag-zoom inside it
  await page.$eval(".win input", (e) => { e.value = "100"; e.dispatchEvent(new Event("change", { bubbles: true })); });
  await sleep(300);
  let w0 = await snap();
  check(w0.every((c) => c.max === 305 * BATCH && Math.abs(c.min - (305 * BATCH - 100e6)) < 1), "window lock: every chart shows the last 100 M samples");
  await dragZoom(1, 0.2, 0.5);
  w0 = await snap();
  check(w0[1].max < 305 * BATCH, "window lock: drag-zoomed chart 1 inside the window");
  await idleRefreshes(2);
  sameView(w0, await snap(), "window lock, unchanged data");
  appendFileSync(metricsPath("camp/live_run"), jsonl(306, 310));
  runs[0] = runEntry("camp/live_run", 310, "partial", T0 + 2170);
  writeIndex();
  await waitLen(310);
  await sleep(300);
  const w1 = await snap();
  check(w1.every((c) => c.max === 310 * BATCH), "window lock, new rows: every chart follows the run to the new tail");
  await page.$eval(".win button", (e) => e.click());
  await sleep(300);
  check((await snap()).every((c) => c.min <= 1 * BATCH && c.max >= 310 * BATCH), "window 'all': every chart back to the whole run");

  // run list changes: a newer run takes over "most recent"
  writeRun("camp/new_run", 1, 120);
  runs.unshift(runEntry("camp/new_run", 120, "partial", T0 + 9999));
  writeIndex();
  await waitLen(120);
  check((await status()).includes("camp/new_run"), "run list: a newer run takes over 'most recent'");
  const n0 = await snap();
  check(n0.every((c) => c.min <= 1 * BATCH && c.max >= 120 * BATCH), "run list: the new run's charts show its whole extent (no inherited zoom)");
  appendFileSync(metricsPath("camp/new_run"), jsonl(121, 125));
  runs[0] = runEntry("camp/new_run", 125, "partial", T0 + 9999 + 35);
  writeIndex();
  await waitLen(125);
  await sleep(300);
  const n1 = await snap();
  check(n1.every((c, i) => c.id === n0[i].id && c.max >= 125 * BATCH && c.min <= 1 * BATCH), "full-extent view, new rows: charts grow to the new tail in place");
  // explicit pick in the selector
  await page.select(".bar select", "camp/done_run");
  await waitLen(250);
  check((await status()).includes("camp/done_run"), "selector: picking a run switches the charts to it");
  const d0 = await snap();
  await idleRefreshes(2);
  sameView(d0, await snap(), "picked finished run, unchanged data");

  // -------------------------------------------------------- Compare tab
  await page.$$eval("nav button", (bs) => bs.find((b) => b.textContent.trim() === "compare").click());
  await page.waitForSelector("aside .list label");
  await page.$$eval("aside .list label", (ls) => {
    for (const l of ls) if (/live_run|done_run/.test(l.textContent)) l.querySelector("input").click();
  });
  await page.waitForFunction(() => [...document.querySelectorAll(".chart")].filter((e) => e.__uplot).length > 4, { timeout: 8000 });
  await sleep(500);
  await snap();
  await dragZoom(0, 0.4, 0.7);
  const c0 = await snap();
  const rows0 = await page.$eval("table tbody", (t) => t.textContent);
  await idleRefreshes(3);
  sameView(c0, await snap(), "compare, unchanged data");
  check((await page.$eval("table tbody", (t) => t.textContent)) === rows0, "compare: matched-samples table unchanged");
  appendFileSync(metricsPath("camp/live_run"), jsonl(311, 320));
  runs[1] = runEntry("camp/live_run", 320, "partial", T0 + 2240);
  writeIndex();
  await page.waitForFunction(() => [...document.querySelectorAll(".chart")].some((e) => e.__uplot && e.__uplot.data[0].includes(320 * 1048576)), { timeout: 8000 });
  await sleep(300);
  const c1 = await snap();
  check(c1[0].id === c0[0].id && c1[0].min === c0[0].min && c1[0].max === c0[0].max, "compare, new rows: zoomed chart kept its window");
  check(c1.filter((c) => !c.title.startsWith("Δ")).every((c) => c.last === 320 * BATCH), "compare, new rows: small multiples reach the new tail");
} finally {
  await browser.close();
  srv.kill();
}
console.log(failures ? `\n${failures} check(s) FAILED` : "\nall checks passed");
process.exit(failures ? 1 : 0);
