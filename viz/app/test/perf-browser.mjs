// Playback benchmark of the BUILT viewer: plays a real replay in headless
// Chrome (SwiftShader) and reports what a frame costs. Not a pass/fail test
// — it prints numbers to compare before / after a change.
//   - frames: rAF callbacks per second while playing (the song is blocked,
//     so the replay clock runs on the frame delta);
//   - main thread: busy ms per frame from a Chrome trace (every top-level
//     task on the renderer's main thread), split into script, GC and
//     style/layout, plus the garbage the minor GCs swept (MB/s) — the
//     allocation rate;
//   - GL: per frame, the draw calls, texture uploads (count + MB, and the
//     sizes sent) and buffer uploads (count + MB), plus how many buffers /
//     textures were created and never deleted over the window (a steady
//     climb is a leak) — counted by wrapping the WebGL2 entry points before
//     the page loads;
//   - --profile <file>: also writes a CPU profile of the window (open it in
//     DevTools > Performance, or pass --top N to print the hottest self-time
//     functions; build with `npx vite build --minify false` for names);
//   - NAMES=1 in the environment: also prints the main thread's trace events
//     by total time.
// Run: node test/perf-browser.mjs <replay.rpl.gz> [--seconds 12] [--start T]
//        [--gauges on|off] [--speed 1] [--runs 1] [--profile out.cpuprofile]
//        [--top 25] [--size 1100x800] [--dist dist/index.html]
// (one Chrome, SwiftShader — keep it to one run at a time on the laptop)
import { existsSync, readdirSync, writeFileSync } from "node:fs";
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

const args = process.argv.slice(2);
const opt = (name, dflt) => {
  const i = args.indexOf("--" + name);
  return i >= 0 ? args.splice(i, 2)[1] : dflt;
};
const SECONDS = +opt("seconds", 12);
const START = opt("start", null);
const GAUGES = opt("gauges", "on") === "on";
const SPEED = +opt("speed", 1);
const RUNS = +opt("runs", 1);
const PROFILE = opt("profile", null);
const TOP = +opt("top", 0);
const [VW, VH] = opt("size", "1100x800").split("x").map(Number);
const dist = resolve(opt("dist", "dist/index.html"));
const replay = args[0] && resolve(args[0]);
if (!replay) throw new Error("usage: perf-browser.mjs <replay.rpl.gz> [options]");
if (!existsSync(dist)) throw new Error("run `npm run build` first");
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

// GL call counters, installed before any page script runs
const COUNT_GL = () => {
  const c = window.__gl = { draws: 0, texUp: 0, texBytes: 0, bufUp: 0,
    bufBytes: 0, programs: 0 };
  const P = WebGL2RenderingContext.prototype;
  const wrap = (name, f) => {
    const orig = P[name];
    P[name] = function (...a) { f(a); return orig.apply(this, a); };
  };
  for (const n of ["drawElements", "drawArrays", "drawElementsInstanced",
    "drawArraysInstanced"]) wrap(n, () => c.draws++);
  // uploads by size ("w x h"), to tell which texture is being re-sent.
  // texImage2D: (target, level, ifmt, w, h, border, fmt, type, pixels) or
  // (target, level, ifmt, fmt, type, source) for a DOM source;
  // texSubImage2D: (target, level, x, y, w, h, fmt, type, pixels) or
  // (target, level, x, y, fmt, type, source)
  c.sizes = {};
  const upload = (w, h) => {
    c.texUp++;
    c.texBytes += 4 * w * h;
    c.sizes[`${w}x${h}`] = (c.sizes[`${w}x${h}`] ?? 0) + 1;
  };
  wrap("texImage2D", (a) => (a.length === 6
    ? upload(a[5]?.width ?? 0, a[5]?.height ?? 0) : upload(a[3], a[4])));
  wrap("texSubImage2D", (a) => (a.length === 7
    ? upload(a[6]?.width ?? 0, a[6]?.height ?? 0) : upload(a[4], a[5])));
  wrap("bufferData", (a) => { c.bufUp++; c.bufBytes += a[1]?.byteLength ?? 0; });
  wrap("bufferSubData", (a) => { c.bufUp++; c.bufBytes += a[2]?.byteLength ?? 0; });
  wrap("linkProgram", () => c.programs++);
  // GPU objects alive: buffers / textures created minus deleted (a steady
  // climb during playback is a leak)
  c.buffers = 0; c.textures = 0;
  wrap("createBuffer", () => c.buffers++);
  wrap("deleteBuffer", () => c.buffers--);
  wrap("createTexture", () => c.textures++);
  wrap("deleteTexture", () => c.textures--);
  // frames: one per rAF callback batch
  window.__frames = 0;
  const tick = () => { window.__frames++; requestAnimationFrame(tick); };
  requestAnimationFrame(tick);
};

const TRACE_CATS = ["toplevel", "devtools.timeline", "v8", "disabled-by-default-v8.gc",
  "disabled-by-default-devtools.timeline"];

function analyzeTrace(events, frames) {
  // the renderer's main thread (the page) and the GPU process's main thread
  const names = new Map();
  for (const e of events)
    if (e.ph === "M" && e.name === "thread_name") names.set(`${e.pid}:${e.tid}`, e.args.name);
  const busy = { main: 0, gpu: 0 }, by = {}, styleBy = {}, allBy = {};
  let garbage = 0, gcCount = 0;
  // nested events of interest on the main thread (durations in µs)
  const KIND = {
    script: ["FunctionCall", "EvaluateScript", "V8.Execute", "TimerFire",
      "FireAnimationFrame", "EventDispatch", "v8.callFunction", "v8.run",
      "RunMicrotasks"],
    gc: ["MinorGC", "MajorGC", "V8.GC_SCAVENGER", "V8.GC_MARK_COMPACTOR",
      "V8.GC_MC_INCREMENTAL", "V8.GC_MINOR_MARK_SWEEPER", "BlinkGC.AtomicPhase"],
    style: ["UpdateLayoutTree", "RecalculateStyles", "Layout", "PrePaint",
      "Paint", "Layerize", "UpdateLayer", "CompositeLayers", "Commit"],
  };
  const kindOf = {};
  for (const [k, ns] of Object.entries(KIND)) for (const n of ns) kindOf[n] = k;
  // only count each kind's OUTERMOST event (FireAnimationFrame contains
  // FunctionCall, a MinorGC can nest in script, etc.)
  const open = { script: [], gc: [], style: [] };
  const sorted = events.filter((e) => e.ph === "X" && e.dur != null)
    .sort((a, b) => a.ts - b.ts);
  for (const e of sorted) {
    const th = names.get(`${e.pid}:${e.tid}`);
    if (e.name === "ThreadControllerImpl::RunTask" || e.name === "RunTask") {
      if (th === "CrRendererMain") busy.main += e.dur;
      else if (th === "CrGpuMain") busy.gpu += e.dur;
    }
    if (th !== "CrRendererMain") continue;
    if (process.env.NAMES) allBy[e.name] = (allBy[e.name] ?? 0) + e.dur;
    if ((e.name === "MinorGC" || e.name === "V8.GC_SCAVENGER") && e.args?.usedHeapSizeBefore) {
      garbage += e.args.usedHeapSizeBefore - e.args.usedHeapSizeAfter;
    }
    if (e.name === "MinorGC" || e.name === "MajorGC") gcCount++;
    const k = kindOf[e.name];
    if (!k) continue;
    const end = e.ts + e.dur;
    const st = open[k];
    while (st.length && st.at(-1) <= e.ts) st.pop();
    if (!st.length) by[k] = (by[k] ?? 0) + e.dur;
    if (k === "style") styleBy[e.name] = (styleBy[e.name] ?? 0) + e.dur;
    st.push(end);
  }
  const perF = (us) => +(us / 1000 / frames).toFixed(2);
  if (process.env.NAMES)
    console.log(Object.entries(allBy).sort((a, b) => b[1] - a[1]).slice(0, 30)
      .map(([k, v]) => `${perF(v)} ${k}`).join("\n"));
  return {
    mainMsPerFrame: perF(busy.main), gpuMsPerFrame: perF(busy.gpu),
    scriptMsPerFrame: perF(by.script ?? 0), gcMsPerFrame: perF(by.gc ?? 0),
    styleLayoutPaintMsPerFrame: perF(by.style ?? 0),
    gcCount, garbageMB: +(garbage / 1048576).toFixed(1),
    // each style / layout / paint step on its own (they nest, so these
    // do not add up to the total above)
    styleSteps: Object.fromEntries(Object.entries(styleBy).map(([k, v]) => [k, perF(v)])),
  };
}

function topSelf(profile, n) {
  const self = new Map(), byId = new Map();
  for (const node of profile.nodes) byId.set(node.id, node);
  const dt = profile.timeDeltas;
  const counts = new Map();
  profile.samples.forEach((id, i) => counts.set(id, (counts.get(id) ?? 0) + (dt[i] ?? 0)));
  let total = 0;
  for (const [id, us] of counts) {
    const cf = byId.get(id).callFrame;
    const key = `${cf.functionName || "(anon)"} ${cf.url ? cf.url.split("/").at(-1) : ""}:${cf.lineNumber + 1}:${cf.columnNumber + 1}`;
    self.set(key, (self.get(key) ?? 0) + us);
    total += us;
  }
  return [...self].sort((a, b) => b[1] - a[1]).slice(0, n)
    .map(([k, us]) => `${(100 * us / total).toFixed(1).padStart(5)}%  ${(us / 1000).toFixed(0).padStart(6)} ms  ${k}`);
}

const browser = await puppeteer.launch({
  executablePath: findBrowser(),
  headless: true,
  args: ["--use-angle=swiftshader", "--disable-gpu-sandbox", "--no-sandbox"],
});
try {
  const results = [];
  for (let run = 0; run < RUNS; run++) {
    const page = await browser.newPage();
    await page.setViewport({ width: VW, height: VH });
    const errors = [];
    page.on("pageerror", (e) => errors.push("pageerror: " + e.message));
    // (the blocked requests below log "Failed to load resource" — expected)
    page.on("console", (m) => {
      if (m.type() === "error" && !/Failed to load resource/.test(m.text()))
        errors.push("console: " + m.text());
    });
    // no song: block everything that is not the page itself (BeatSaver
    // audio + map metadata), so the clock is the frame delta and a run does
    // not depend on the network
    await page.setRequestInterception(true);
    page.on("request", (r) => (r.url().startsWith("file:") || r.url().startsWith("data:")
      || r.url().startsWith("blob:") ? r.continue() : r.abort()));
    await page.evaluateOnNewDocument(COUNT_GL);
    await page.goto("file:///" + dist);
    await page.waitForSelector("#stage canvas", { timeout: 15000 });
    const input = await page.$("input[type=file]");
    await input.uploadFile(replay);
    await page.waitForFunction(() => window.__viewerState?.().loaded, { timeout: 60000 });
    const st = await page.evaluate(() => window.__viewerState());
    if (!GAUGES) await page.keyboard.press("g");
    // speed: the bar's +/- keys double / halve
    for (let s = 1; s < SPEED; s *= 2) await page.keyboard.press("+");
    if (START != null) {
      await page.evaluate((tick) => {
        const r = [...document.querySelectorAll("#bottom input[type=range]")].at(-1);
        Object.getOwnPropertyDescriptor(HTMLInputElement.prototype, "value").set.call(r, tick);
        r.dispatchEvent(new Event("input", { bubbles: true }));
      }, +START);
      await page.keyboard.press(" "); // the scrub paused it
    }
    await sleep(3000); // warm up: programs compiled, notes on screen

    const cdp = await page.createCDPSession();
    const g0 = await page.evaluate(() => ({ ...window.__gl, frames: window.__frames }));
    const tick0 = (await page.evaluate(() => window.__viewerState())).tick;
    const traceEvents = [];
    cdp.on("Tracing.dataCollected", (d) => traceEvents.push(...d.value));
    const traceDone = new Promise((r) => cdp.once("Tracing.tracingComplete", r));
    await cdp.send("Tracing.start", { traceConfig: { includedCategories: TRACE_CATS },
      transferMode: "ReportEvents" });
    if (PROFILE) {
      await cdp.send("Profiler.enable");
      await cdp.send("Profiler.setSamplingInterval", { interval: 250 });
      await cdp.send("Profiler.start");
    }
    const w0 = Date.now();
    await sleep(SECONDS * 1000);
    const wall = (Date.now() - w0) / 1000;
    let prof = null;
    if (PROFILE) prof = (await cdp.send("Profiler.stop")).profile;
    await cdp.send("Tracing.end");
    await traceDone;
    const g1 = await page.evaluate(() => ({ ...window.__gl, frames: window.__frames }));
    const vs = await page.evaluate(() => window.__viewerState());
    const frames = g1.frames - g0.frames;
    const pf = (k, d = 1) => +((g1[k] - g0[k]) / frames).toFixed(d);
    const mb = (k) => +((g1[k] - g0[k]) / frames / 1048576).toFixed(3);
    const res = {
      replay: st.name, tickHz: st.tickHz, size: `${VW}x${VH}`, gauges: GAUGES, speed: SPEED,
      ticks: [tick0, vs.tick], playing: vs.playing,
      fps: +(frames / wall).toFixed(1), frames,
      ...analyzeTrace(traceEvents, frames),
      drawsPerFrame: pf("draws"), texUploadsPerFrame: pf("texUp", 2),
      texMBPerFrame: mb("texBytes"), bufUploadsPerFrame: pf("bufUp", 1),
      bufMBPerFrame: mb("bufBytes"), programsLinked: g1.programs - g0.programs,
      liveBuffersGrowth: g1.buffers - g0.buffers, liveTexturesGrowth: g1.textures - g0.textures,
      texSizes: Object.fromEntries(Object.entries(g1.sizes)
        .map(([k, v]) => [k, v - (g0.sizes[k] ?? 0)]).filter(([, v]) => v > 0)
        .sort((a, b) => b[1] - a[1]).slice(0, 6)),
    };
    res.garbageMBps = +(res.garbageMB / wall).toFixed(1);
    results.push(res);
    console.log(JSON.stringify(res));
    if (prof) {
      writeFileSync(PROFILE.replace(/(\.cpuprofile)?$/, run ? `.${run}.cpuprofile` : ".cpuprofile"),
        JSON.stringify(prof));
      if (TOP) console.log(topSelf(prof, TOP).join("\n"));
    }
    if (errors.length) throw new Error("page errors:\n" + errors.join("\n"));
    await page.close();
  }
  if (RUNS > 1) {
    const med = (k) => {
      const v = results.map((r) => r[k]).sort((a, b) => a - b);
      return v[Math.floor(v.length / 2)];
    };
    const keys = Object.keys(results[0]).filter((k) => typeof results[0][k] === "number");
    console.log("median: " + JSON.stringify(Object.fromEntries(keys.map((k) => [k, med(k)]))));
  }
} finally {
  await browser.close();
}
