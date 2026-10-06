// The Flux WebXR game through a REAL WebXR session, headless: Meta's
// Immersive Web Emulation Runtime (iwer) emulates a Quest 3 in Chrome, and
// its controllers are moved every frame along a scripted player's path.
// Exercises exactly what a headset does: "enter VR" (a trusted click),
// the reference-space offset (the player's floor origin at the env body's
// plane), grip-space palm reads, the trigger's `select` to start, the
// judged run, and the results — then exits VR. Screenshot ->
// test/out/flux-xr-*.png. Run after `npm run build`:
//   node test/flux-xr-browser.mjs
import { existsSync, mkdirSync, readFileSync, readdirSync } from "node:fs";
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
const IWER = readFileSync(resolve("node_modules/iwer/build/iwer.min.js"), "utf8");

// Page-side harness, installed before the app: the emulated headset (head
// 1.7 m up at the floor origin, mono, small views so software GL keeps up)
// and a per-frame driver that puts each controller's GRIP at the scripted
// player's palm — converted env world -> XR local-floor coords, the
// inverse of FluxPlay's read: env (x, y, z) = scene (x, y, -z), scene =
// local + (0, 0, -PLANE_Z0).
//
// iwer's controller.position is the TARGET-RAY origin; the grip space the
// app reads is that pose times the controller profile's gripOffsetMatrix
// (Touch Plus: (-/+0.004, -0.016, +0.050) m, plus a ~45 deg tilt). The
// controllers keep iwer's identity orientation, so the driver subtracts the
// offset's translation to land the grip — the palm — on the path.
//
// Timing: the driver is a window rAF and can only use the LAST frame's
// play.time; iwer snapshots controller poses at the start of its next XR
// frame, and that frame's game tick advances play.time again before the
// test can look. So play.palms trail play.time by one frame (3x clock on a
// slow software-GL frame: 0.1-0.3 s of chart time, across a palm jump).
// The harness therefore stamps what it drove — {t, palms} — and, in an XR
// frame callback (after iwer's snapshot, in the same frame as the app's
// read), latches the stamp the snapshot used: window.__gripDriven.
const HARNESS = `
  const dev = new IWER.XRDevice(IWER.metaQuest3);
  dev.stereoEnabled = false;
  dev.position.set(0, 1.7, 0);
  dev.installRuntime({ forceInstall: true });
  window.__xr = dev;
  const GRIP = Object.fromEntries(["left", "right"].map((side) => {
    const m = IWER.metaQuest3.controllerConfig.layout[side].gripOffsetMatrix;
    return [side, [m[12], m[13], m[14]]];
  }));
  let driven = null;
  const xrs = navigator.xr, request = xrs.requestSession.bind(xrs);
  xrs.requestSession = async (...args) => {
    const s = await request(...args);
    const latch = () => { window.__gripDriven = driven; s.requestAnimationFrame(latch); };
    s.requestAnimationFrame(latch);
    return s;
  };
  const REST = [[-0.25, 1.1, -0.4], [0.25, 1.1, -0.4]];
  // palms at env z -0.35, reached 0.35 / speed after the hit time: each
  // palm holds its object until then
  window.__palmAt = (t, chart) => [0, 1].map((h) => {
    const o = chart.find((o) => o.t + 0.35 / o.speed >= t - 0.02 &&
      ((o.kind === 4 && o.color === h) || o.kind === 5));
    if (!o) return REST[h];
    if (o.kind === 4) return [o.x, o.y, -0.35];
    const ux = -Math.sin(o.rot), uy = Math.cos(o.rot);
    const s = o.sweet * 0.5 * o.len;
    const sx = o.x + s * ux, sy = o.y + s * uy, nx = uy, ny = -ux;
    const d = (h === 0 ? -1 : 1) * 0.3 * (nx < 0 ? -1 : 1);
    return [sx + d * nx, sy + d * ny, -0.35];
  });
  const drive = () => {
    const P = window.__fluxPlay;
    if (P) {
      const chart = P.session()?.chart ?? [];
      const t = P.play.time;
      const palms = window.__palmAt(t, chart);
      [["left", 0], ["right", 1]].forEach(([side, h]) => {
        const c = dev.controllers[side];
        const [x, y, z] = palms[h], g = GRIP[side];
        if (c) c.position.set(x - g[0], y - g[1], -z - 0.65 - g[2]);
      });
      driven = { t, palms };
    }
    requestAnimationFrame(drive);
  };
  requestAnimationFrame(drive);
`;

const browser = await puppeteer.launch({
  executablePath: findBrowser(), headless: true,
  args: ["--use-angle=swiftshader", "--disable-gpu-sandbox", "--no-sandbox"],
});
// by accessible name: the aria-label of the bar's icon buttons, else the text
const clickText = async (page, sel, text) => {
  const h = await page.evaluateHandle((sel, text) =>
    [...document.querySelectorAll(sel)]
      .find((e) => (e.getAttribute("aria-label") ?? e.textContent).includes(text)),
  sel, text);
  const el = h.asElement();
  if (!el) throw new Error(`no ${sel} with "${text}"`);
  await el.click(); // a trusted click: requestSession needs user activation
};
const waitPlay = (page, fn, arg, timeout) =>
  page.waitForFunction(fn, { timeout, polling: 100 }, arg);
try {
  const page = await browser.newPage();
  await page.setViewport({ width: 900, height: 700 });
  const errors = [];
  page.on("pageerror", (e) => errors.push("pageerror: " + e.message));
  page.on("console", (m) => {
    if (m.type() === "error") errors.push("console: " + m.text());
  });
  await page.evaluateOnNewDocument(IWER + "\n" + HARNESS);
  await page.goto("file:///" + dist);
  await page.waitForSelector("#stage canvas", { timeout: 15000 });

  await clickText(page, "#bar button", "play Flux");
  await page.waitForSelector("#play select", { timeout: 10000 });
  await page.select("#play select", "warmup-2");
  await waitPlay(page, () => window.__fluxPlay.play.xrSupported === true, null, 10000);
  await page.evaluate(() => window.__fluxPlay.setClockScale(3));
  await clickText(page, "#play button", "enter VR");
  await waitPlay(page, () => window.__fluxPlay.play.presenting, null, 15000);

  // the reference-space offset: the emulated head (1.7 m up at the floor
  // origin) must read as the env body's head position, world z = PLANE_Z0
  await waitPlay(page, () => !!window.__fluxPlay.play.head, null, 10000);
  const head = await page.evaluate(() => Array.from(window.__fluxPlay.play.head));
  if (Math.abs(head[0]) > 1e-3 || Math.abs(head[1] - 1.7) > 1e-3 ||
      Math.abs(head[2] + 0.65) > 1e-3)
    throw new Error("head in env coords: " + head + "\n" + errors.join("\n"));

  // the trigger's `select` starts the chart (the in-headset control)
  await page.evaluate(() => {
    window.__xr.controllers.right.updateButtonValue("trigger", 1);
  });
  await new Promise((r) => setTimeout(r, 400));
  await page.evaluate(() => {
    window.__xr.controllers.right.updateButtonValue("trigger", 0);
  });
  await waitPlay(page, () => window.__fluxPlay.play.phase === "countdown" ||
    window.__fluxPlay.play.phase === "playing", null, 10000);

  // the palms are the controllers' grips, in env coords: play.palms (read
  // this frame) against the palms the harness drove into the pose iwer
  // snapshotted for this frame, stamped with the game time they were
  // scripted for — not play.time, which has since advanced a frame
  await waitPlay(page, () => window.__fluxPlay.play.time > 2, null, 90000);
  const pal = await page.evaluate(() => {
    const P = window.__fluxPlay, d = window.__gripDriven;
    return { got: P.play.palms.map((p) => Array.from(p)),
             want: d.palms, at: d.t, now: P.play.time };
  });
  const off = Math.max(...[0, 1].flatMap((h) =>
    [0, 1, 2].map((i) => Math.abs(pal.got[h][i] - pal.want[h][i]))));
  // float32 pose math only: no lag, no grip-offset slack
  if (!(off < 1e-3) || !(pal.at <= pal.now))
    throw new Error("grip palms vs driven: " + JSON.stringify(pal));
  await page.screenshot({ path: "test/out/flux-xr-mid.png" });
  // the viewer's DOM overlays stay out of the game: no joint gauges
  if (await page.$(".gauges")) throw new Error("joint gauges shown in the WebXR game");

  await waitPlay(page, () => window.__fluxPlay.play.phase === "done", null, 240000);
  const st = await page.evaluate(() => ({ ...window.__fluxPlay.play.stats }));
  await page.screenshot({ path: "test/out/flux-xr-done.png" });
  if (st.good + st.bad + st.misses !== st.n_objects)
    throw new Error("not every object judged: " + JSON.stringify(st));
  if (!(st.good > 0.5 * st.n_objects))
    throw new Error("the driven controllers should clear most of it: " + JSON.stringify(st));

  // (a DOM click: iwer's presenting canvas overlays the page, and ending a
  // session needs no user activation)
  await page.evaluate(() => [...document.querySelectorAll("#play button")]
    .find((b) => b.textContent.includes("exit VR")).click());
  await waitPlay(page, () => !window.__fluxPlay.play.presenting, null, 10000);
  if (errors.length) throw new Error(errors.join("\n"));
  console.log(`flux-xr-browser OK: head ${head.map((v) => v.toFixed(3))}, grip ` +
    `offset ${off.toExponential(1)} m (${(pal.now - pal.at).toFixed(3)} s behind ` +
    `play.time); ${st.n_objects} objects, good ${st.good}, bad ${st.bad}, ` +
    `miss ${st.misses}, score ${st.score}`);
} finally {
  await browser.close();
}
