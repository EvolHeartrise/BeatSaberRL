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
//     picking a run in the selector switches the charts to it;
//   - Swing tab (2026-10-04): the per-level table from swing.json sorts
//     (default: lowest pre-swing first), and an idle refresh keeps the
//     sort, the scroll position and the very same row elements; new data
//     updates the rows in place under the user's sort; a window switch
//     changes the plays; an old (v1) run shows its totals and marks the
//     swing columns unavailable.
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

// swing.json fixtures (python/swing_summary.py format): a v2 run with
// swing columns and an old v1 run without; older t_last than the Live
// runs so "most recent" stays theirs
const NLEV = 400;
const WKEYS = ["n1", "n5", "n20", "k100", "k500", "all"];
function swingDoc(v2, bump = 0) {
  const lv = { key: [], song: [], mapper: [], diff: [], ss_stars: [], bl_stars: [], ss_id: [], bl_id: [], nps: [],
    att: [], done: [], best_acc: [], ss_acc: [], ss_pp: [], bl_acc: [], bl_pp: [], best_upd: [] };
  for (let i = 0; i < NLEV; i++) {
    lv.key.push(`h${i}_Expert_Standard`); lv.song.push(`Song ${String(i).padStart(3, "0")}`); lv.mapper.push("m");
    lv.diff.push(i % 2 ? "Expert" : "ExpertPlus"); lv.ss_stars.push(i % 3 ? 2 + (i % 11) : null);
    lv.bl_stars.push(i % 3 !== 1 ? 3 + (i % 7) : null); lv.ss_id.push(i); lv.bl_id.push(`b${i}`);
    lv.nps.push(2 + (i % 9) / 2); lv.att.push(30); lv.done.push(25); lv.best_acc.push(95);
    lv.ss_acc.push(95); lv.ss_pp.push(100 + i); lv.bl_acc.push(95); lv.bl_pp.push(120 + i); lv.best_upd.push(100);
  }
  const win = {};
  WKEYS.forEach((k, wi) => {
    const w = { plays: [], acc: [] };
    for (const f of ["pre", "post", "prec"]) for (const h of ["", "L", "R"]) w[f + h] = v2 ? [] : null;
    for (let i = 0; i < NLEV; i++) {
      w.plays.push([1, 5, 20, 7, 25, 25][wi]);
      w.acc.push(90 + ((i * 7) % 100) / 10);
      if (v2) {
        const pre = 80 + ((i * 37 + wi) % 200) / 10, post = 70 + ((i * 53 + bump) % 300) / 10, prec = 85 + ((i * 11) % 150) / 10;
        w.pre.push(pre); w.preL.push(pre - 1); w.preR.push(pre + 1);
        w.post.push(post); w.postL.push(post); w.postR.push(post);
        w.prec.push(prec); w.precL.push(prec); w.precR.push(prec);
      }
    }
    win[k] = w;
  });
  return { format: "beat-arena-swing", version: 1, generated: 1, ep_layout: v2 ? 2 : 1, swing: v2, levels_source: "trainer",
    catalog: "both", rows: 12000, completed: 10000, last_update: 600, n_levels: NLEV,
    totals: { ss_wpp: 23456.7 + bump, ss_upp: 3000000, bl_wpp: 31000, bl_upp: 2900000, ss_levels: 267, bl_levels: 267 },
    windows: WKEYS.map((k) => ({ key: k, label: k, available: true })), levels: lv, win };
}
const swingPath = (id) => resolve(ARCH, id.split("/")[0], "runs", id.split("/")[1], "swing.json");
function writeSwing(id, v2, bump = 0) {
  writeFileSync(swingPath(id), JSON.stringify(swingDoc(v2, bump)));
  const r = runs.find((x) => x.id === id);
  r.swing = { path: `${id.split("/")[0]}/runs/${id.split("/")[1]}/swing.json`, swing: v2, totals: swingDoc(v2, bump).totals };
}
writeRun("camp/swing_v2", 1, 50);
writeRun("camp/swing_v1", 1, 40);
runs.push(runEntry("camp/swing_v2", 50, "finished", T0 + 500), runEntry("camp/swing_v1", 40, "finished", T0 + 400));
writeSwing("camp/swing_v2", true);
writeSwing("camp/swing_v1", false);
writeIndex();

// "backfill pass" with nothing new: index.json gets a fresh `generated`,
// metrics files are rewritten byte-identically (new mtime, same bytes)
function noopPass() {
  writeIndex();
  for (const r of runs) {
    const p = metricsPath(r.id);
    writeFileSync(p, readFileSync(p));
    if (r.swing) writeFileSync(swingPath(r.id), readFileSync(swingPath(r.id)));
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

  // hover (2026-10-06): values go to one floating tooltip (the hovered
  // chart's), the title floats on the plot, and no card changes height
  const hb = await page.evaluate(() => {
    const el = [...document.querySelectorAll(".chart")].filter((e) => e.__uplot)[1];
    el.scrollIntoView({ block: "center" });
    const r = el.__uplot.over.getBoundingClientRect();
    return { x: r.left, y: r.top, w: r.width, h: r.height, titleInPlot: !!el.__uplot.over.querySelector(".u-title") };
  });
  check(hb.titleInPlot, "hover: the title floats on the plot area");
  const cellHs = () => page.$$eval(".cell", (cs) => cs.map((c) => c.getBoundingClientRect().height));
  const visTips = () => page.$$eval(".dash-tip", (ts) => ts.filter((t) => t.style.display !== "none").map((t) =>
    [...t.querySelectorAll(".dash-tip-row")].filter((r) => r.style.display !== "none").map((r) => r.querySelector(".dash-tip-val").textContent)));
  const h0 = await cellHs();
  const doc0 = await page.evaluate(() => document.documentElement.scrollHeight);
  let moved = 0, tipCounts = new Set(), rowsOk = true;
  for (let k = 0; k <= 20; k++) {
    await page.mouse.move(hb.x + hb.w * (0.02 + (0.96 * k) / 20), hb.y + hb.h * (0.15 + 0.035 * k));
    await sleep(20);
    const h = await cellHs();
    if (h.some((v, i) => v !== h0[i])) moved++;
    const t = await visTips();
    tipCounts.add(t.length);
    if (t.length === 1 && !(t[0].length === 2 && t[0].every((v) => v && v !== "—"))) rowsOk = false;
  }
  check(moved === 0, `hover: no card changed height over 21 cursor positions (${moved} moved)`);
  check((await page.evaluate(() => document.documentElement.scrollHeight)) === doc0, "hover: page height unchanged");
  check(tipCounts.size === 1 && tipCounts.has(1), `hover: exactly one tooltip shown, the hovered chart's (${[...tipCounts]})`);
  check(rowsOk, "hover: tooltip lists both runs with values");
  await page.mouse.move(2, 2);
  await sleep(50);
  check((await visTips()).length === 0, "hover: tooltip gone when the cursor leaves");

  // ---------------------------------------------------------- Swing tab
  await page.$$eval("nav button", (bs) => bs.find((b) => b.textContent.trim() === "swing").click());
  await page.waitForSelector(".tablewrap tbody tr", { timeout: 8000 });
  const colOf = (label) => page.$$eval(".tablewrap thead th", (ths, label) => ths.findIndex((t) => t.textContent.replace(/[▲▼]/g, "").trim() === label), label);
  const column = async (label, n = 60) => {
    const c = await colOf(label);
    return page.$$eval(".tablewrap tbody tr", (trs, c, n) => trs.slice(0, n).map((tr) => tr.children[c].textContent.trim()), c, n);
  };
  const nums = (a) => a.map((x) => (x === "—" ? null : Number(x.replace(/,/g, ""))));
  const sorted = (a, dir) => a.every((v, i) => i === 0 || (dir > 0 ? a[i - 1] <= v : a[i - 1] >= v));
  check((await page.$eval(".status", (e) => e.textContent)).includes("camp/swing_v2"), "swing: most recent run with a swing summary selected");
  check((await page.$eval(".totals", (e) => e.textContent)).includes("23,457"), "swing: best-possible SS WPP tile shown");
  const sortedHdr = () => page.$eval(".tablewrap thead", (t) => [...t.querySelectorAll("th")].find((h) => /[▲▼]/.test(h.textContent))?.textContent.trim());
  check((await sortedHdr()) === "pre % ▲", `swing: default sort is pre-swing ascending (${await sortedHdr()})`);
  check(sorted(nums(await column("pre %")), 1), "swing: rows ordered by lowest pre-swing first");
  // user sort: post-swing descending
  await page.$$eval(".tablewrap thead th", (ths) => ths.find((t) => t.textContent.trim().startsWith("post %")).click());
  await page.$$eval(".tablewrap thead th", (ths) => ths.find((t) => t.textContent.trim().startsWith("post %")).click());
  await sleep(200);
  check((await sortedHdr()) === "post % ▼" && sorted(nums(await column("post %")), -1), "swing: click sorts by post-swing, again reverses");
  await page.evaluate(() => window.scrollTo(0, 900));
  await sleep(200);
  const mark = () => page.evaluate(() => {
    const trs = [...document.querySelectorAll(".tablewrap tbody tr")];
    trs.forEach((tr, i) => { if (tr.__mark == null) tr.__mark = i; });
    return { y: window.scrollY, marks: trs.slice(0, 50).map((tr) => tr.__mark), text: document.querySelector(".tablewrap tbody").textContent };
  });
  const m0 = await mark();
  check(m0.y > 500, `swing: scrolled the table (scrollY ${m0.y})`);
  await idleRefreshes(3);
  const m1 = await mark();
  check(m1.y === m0.y, `swing, unchanged data: scroll position kept (${m0.y} -> ${m1.y})`);
  check(m1.text === m0.text, "swing, unchanged data: table content unchanged");
  check(m1.marks.every((v, i) => v === m0.marks[i]), "swing, unchanged data: the same row elements (no re-render)");
  check((await sortedHdr()) === "post % ▼", "swing, unchanged data: sort kept");
  // window switch: plays follow
  await page.$$eval(".wins button", (bs) => bs.find((b) => b.textContent.trim() === "n20").click());
  await sleep(200);
  check((await column("plays", 5)).every((v) => v === "20"), "swing: window 'last 20 plays' shows 20 plays");
  await page.$$eval(".wins button", (bs) => bs.find((b) => b.textContent.trim() === "n5").click());
  await sleep(200);
  // new data: post values move; the sort and the scroll stay
  const firstPost0 = (await column("post %", 1))[0];
  writeSwing("camp/swing_v2", true, 7);
  writeIndex();
  await page.waitForFunction(() => document.querySelector(".totals").textContent.includes("23,464"), { timeout: 8000 });
  await sleep(300);
  const m2 = await mark();
  check((await sortedHdr()) === "post % ▼" && sorted(nums(await column("post %")), -1), "swing, new data: still sorted by post-swing descending");
  check(Math.abs(m2.y - m0.y) < 2, `swing, new data: scroll position kept (${m0.y} -> ${m2.y})`);
  check(m2.text !== m0.text, `swing, new data: rows updated (top post ${firstPost0} -> ${(await column("post %", 1))[0]})`);
  // an old v1 run: totals yes, swing columns unavailable
  await page.select(".bar select", "camp/swing_v1");
  await page.waitForFunction(() => document.querySelector(".status")?.textContent.includes("camp/swing_v1"), { timeout: 8000 });
  await sleep(200);
  check((await page.$eval(".status", (e) => e.textContent)).includes("swing columns unavailable"), "swing, v1 run: swing columns marked unavailable");
  check((await column("pre %", 20)).every((v) => v === "—"), "swing, v1 run: pre-swing cells empty");
  check((await page.$eval(".totals", (e) => e.textContent)).includes("23,457"), "swing, v1 run: best-possible totals shown");
} finally {
  await browser.close();
  srv.kill();
}
console.log(failures ? `\n${failures} check(s) FAILED` : "\nall checks passed");
process.exit(failures ? 1 : 0);
