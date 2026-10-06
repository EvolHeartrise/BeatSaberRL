// Headless-browser smoke test of the BUILT single-file viewer: loads
// dist/index.html in Edge, uploads a real v10 replay through the file
// input, plays a second, and asserts playback advanced with zero page
// errors. Also drives the v10 manual-pose panel (root + 29 joints) and
// screenshots the articulated skeleton + per-joint gauges/forces, checks
// the DOM gauges show the replay's own numbers at a known tick (and the
// posed angles in pose mode), and toggles the first-person camera (at the
// head, looking straight down the track) on and back off.
// Screenshots land in test/out/ for eyeball verification.
// Run: node test/browser.test.js [path-to-replay.rpl.gz]
import { existsSync, mkdirSync } from "node:fs";
import { basename, resolve } from "node:path";
import puppeteer from "puppeteer-core";
import {
  JOINT_LIMITS, TAU_MAX, activationAt, footForces, gearTable, parseReplay,
  setTickRate, tickHzOf,
} from "../src/lib/replay.js";
import { readReplayFile } from "./fixture.js";
import { barLabels, clickBar, scrubTo, sliderToMax, toggleOn } from "./bar.js";

// browser binary: $PUPPETEER_EXECUTABLE_PATH, else Edge on the Windows
// box, else the Chrome that `npx puppeteer browsers install chrome` put
// under ~/.cache/puppeteer (Linux laptop)
import { readdirSync } from "node:fs";
import { homedir } from "node:os";
function findBrowser() {
  if (process.env.PUPPETEER_EXECUTABLE_PATH) return process.env.PUPPETEER_EXECUTABLE_PATH;
  const edge = "C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe";
  if (existsSync(edge)) return edge;
  const root = resolve(homedir(), ".cache/puppeteer/chrome");
  if (existsSync(root)) {
    for (const v of readdirSync(root).sort().reverse()) {
      const p = resolve(root, v, "chrome-linux64/chrome");
      if (existsSync(p)) return p;
    }
  }
  throw new Error("no browser found: set PUPPETEER_EXECUTABLE_PATH");
}
const EDGE = findBrowser();
const dist = resolve("dist/index.html");
const replay = resolve(process.argv[2] ??
  "test/fixtures/v10_scripted_ep00_seed0.rpl.gz");
// env v15: a recording whose header carries "track_anchor": "head", so the
// built bundle is exercised on the head-anchored track as well as on the
// legacy fixed one above (skipped if the probe recording is not present).
const replayV15 = resolve(process.argv[3] ??
  "../../python/replays/v15_probe_toppp01_delightful-introduction_1pp.rpl.gz");
if (!existsSync(dist)) throw new Error("run `npm run build` first");
mkdirSync("test/out", { recursive: true });

// ---- the DOM gauges (BodyGauges.svelte) against the replay itself --------
// Paused on a whole tick k, every joint row must read the replay's angle
// (q at k), ω and applied τ (both at k + 1, as the viewer draws them), each
// needle must point at the angle, and the force block must show tick k + 1's
// foot reactions and joint activation. Also: the two panels sit in the
// bottom corners at the sprites' old size (0.62 / (2 tan 27.5°) of the
// stage height, 104 px above the bottom), and let pointer events through.
async function checkGauges(page, file, k) {
  const rep = parseReplay(readReplayFile(file));
  setTickRate(tickHzOf(rep)); // footForces divides by the replay's DT
  const k1 = Math.min(k + 1, rep.ticks.length - 1);
  const tb = rep.ticks[k1];
  const table = rep.body ? null : gearTable(rep.config.gears ?? 3);
  const deg = (r) => (r * 180) / Math.PI;
  const want = JOINT_LIMITS.map(([lo, hi], j) => {
    const q = rep.ticks[k].q[7 + j], om = tb.omega[j], a = tb.action[j];
    const lim = rep.body ? rep.body.tau_max[j] : TAU_MAX[j];
    const tau = rep.body ? (rep.body.gear_frac[j][a] ?? 0) * lim : (table[a] ?? 0) * lim;
    const t = tau === 0 ? 0 : Math.max(-1, Math.min(1, tau / lim));
    return { q, ang: `${deg(q).toFixed(0)}°`, om: `ω ${om.toFixed(1)}`,
      hot: Math.min(q - lo, hi - q) < (10 * Math.PI) / 180, t };
  });
  const got = await page.evaluate(() => {
    const $ = (s) => document.querySelector(".gauges " + s);
    return {
      joints: Array.from({ length: 29 }, (_, j) => {
        const g = $(`g[data-joint="${j}"]`), n = g.querySelector(".needle");
        const tau = g.querySelector(".tau");
        return {
          ang: $(`.ang[data-joint="${j}"]`).textContent,
          om: $(`.om[data-joint="${j}"]`).textContent,
          needle: ["x1", "y1", "x2", "y2"].map((a) => +n.getAttribute(a)),
          hot: n.classList.contains("hot"),
          tauOff: tau.classList.contains("off"), tauNeg: tau.classList.contains("neg"),
          tauX2: +tau.getAttribute("x2"),
        };
      }),
      feet: [0, 1].map((i) => $(`[data-foot="${i}"]`).textContent),
      act: $("[data-act]").textContent,
      panels: [...document.querySelectorAll(".gauges .panel")].map((p) => {
        const r = p.getBoundingClientRect();
        return { left: r.left, right: r.right, bottom: r.bottom, height: r.height };
      }),
      stage: document.querySelector("#stage").getBoundingClientRect().toJSON(),
      events: getComputedStyle($(".panel")).pointerEvents,
    };
  });
  const bad = [];
  want.forEach((w, j) => {
    const g = got.joints[j];
    if (g.ang !== w.ang) bad.push(`joint ${j} angle ${g.ang} != ${w.ang}`);
    if (g.om !== w.om) bad.push(`joint ${j} ${g.om} != ${w.om}`);
    if (g.hot !== w.hot) bad.push(`joint ${j} limit warning ${g.hot} != ${w.hot}`);
    // the needle runs from the dial centre toward the angle (0 = up, clockwise)
    const [x1, y1, x2, y2] = g.needle;
    const a = Math.atan2(x2 - x1, y1 - y2);
    if (Math.abs(a - w.q) > 2e-3) bad.push(`joint ${j} needle at ${deg(a)}° != ${deg(w.q)}°`);
    // τ bar: hidden at zero, else from the middle (189) by t × 49 units
    if (g.tauOff !== (w.t === 0)) bad.push(`joint ${j} τ bar shown ${!g.tauOff}, τ ${w.t}`);
    if (w.t !== 0 && (g.tauNeg !== w.t < 0 || Math.abs(g.tauX2 - (189 + w.t * 49)) > 0.01))
      bad.push(`joint ${j} τ bar to ${g.tauX2}${g.tauNeg ? " (neg)" : ""} != ${189 + w.t * 49}`);
  });
  const feet = footForces(rep, k1), act = activationAt(rep, k1);
  [0, 1].forEach((i) => {
    const s = feet ? `${feet[i].toFixed(0)} N` : "n/a";
    if (got.feet[i] !== s) bad.push(`foot ${i}: ${got.feet[i]} != ${s}`);
  });
  const as = act ? `${act[0].toFixed(1)}/29` : "n/a";
  if (got.act !== as) bad.push(`activation ${got.act} != ${as}`);
  const H = got.stage.height, hWant = (0.62 / (2 * Math.tan((27.5 * Math.PI) / 180))) * H;
  if (got.panels.length !== 2) bad.push(`${got.panels.length} gauge panels`);
  for (const [i, p] of got.panels.entries()) {
    if (Math.abs(p.height - hWant) > 1) bad.push(`panel ${i} height ${p.height} != ${hWant}`);
    if (Math.abs(p.bottom - (H - 104)) > 1) bad.push(`panel ${i} bottom ${p.bottom}`);
  }
  if (!(got.panels[0]?.left < 30) || !(got.panels[1]?.right > got.stage.width - 30))
    bad.push("panels not in the corners: " + JSON.stringify(got.panels));
  if (got.events !== "none") bad.push("the panels catch pointer events");
  if (bad.length) throw new Error(`gauges at tick ${k}:\n  ` + bad.join("\n  "));
  console.log(`gauges at tick ${k}: 29 joints, feet ${got.feet.join(" / ")}, ` +
    `activation ${got.act} — match the replay`);
}

const browser = await puppeteer.launch({
  executablePath: EDGE,
  headless: true,
  // --no-sandbox: Ubuntu 24.04's AppArmor blocks Chrome's user-namespace
  // sandbox for unprivileged users; this only ever loads our own local
  // file, so the sandbox buys nothing here
  // autoplay-policy: the live A/V sync check below needs the song to
  // actually start without a click on the page itself
  args: ["--use-angle=swiftshader", "--disable-gpu-sandbox", "--no-sandbox",
         "--autoplay-policy=no-user-gesture-required"],
});
try {
  const page = await browser.newPage();
  await page.setViewport({ width: 1100, height: 900 });
  const errors = [];
  page.on("pageerror", (e) => errors.push("pageerror: " + e.message));
  page.on("console", (m) => {
    if (m.type() === "error") errors.push("console: " + m.text());
  });
  // an alert() would hang the page: auto-dismiss it and count it as an error
  page.on("dialog", (d) => { errors.push("dialog: " + d.message()); d.dismiss(); });

  await page.goto("file:///" + dist.replace(/\\/g, "/"));
  await page.waitForSelector("#stage canvas", { timeout: 15000 });
  await page.screenshot({ path: "test/out/empty.png" });

  // ---- manual pose mode works straight from the landing state (no file):
  // toggle 'p', drive root + joint sliders, check the FK readout reacts.
  // Slider DOM order: 0-5 root (x,y,z,yaw,pitch,roll), 6-8 spine,
  // 9-15 L arm, 16-21 L leg, 22-28 R arm, 29-34 R leg.
  await page.keyboard.press("p");
  await page.waitForSelector("#pose", { timeout: 5000 });
  const sliderCount = await page.$$eval("#pose input[type=range]",
    (els) => els.length);
  if (sliderCount !== 35)
    throw new Error(`expected 35 pose sliders (6 root + 29), got ${sliderCount}`);
  const fk0 = await page.$eval("#pose .fk", (el) => el.textContent);
  const setSlider = (idx, frac) => page.evaluate(({ idx, frac }) => {
    const r = document.querySelectorAll("#pose input[type=range]")[idx];
    const v = +r.min + frac * (+r.max - +r.min);
    Object.getOwnPropertyDescriptor(HTMLInputElement.prototype, "value")
      .set.call(r, v);
    r.dispatchEvent(new Event("input", { bubbles: true }));
  }, { idx, frac });
  await setSlider(9, 0.85);   // L arm pitch well up
  await setSlider(25, 0.6);   // R elbow flexed
  await setSlider(23, 0.9);   // R arm yaw swung out
  await setSlider(19, 0.7);   // L knee bent (the legs are real now)
  await setSlider(4, 0.45);   // root pitch leaned
  await new Promise((r) => setTimeout(r, 400));
  const fk1 = await page.$eval("#pose .fk", (el) => el.textContent);
  if (fk0 === fk1) throw new Error("pose sliders did not move the FK readout");
  // the gauges show the posed joints (ω = τ = 0): L arm pitch (joint 3,
  // slider 9) reads the slider's angle
  const posed = await page.evaluate(() => ({
    slider: +document.querySelectorAll("#pose input[type=range]")[9].value,
    ang: document.querySelector('.gauges .ang[data-joint="3"]')?.textContent,
    om: document.querySelector('.gauges .om[data-joint="3"]')?.textContent,
    tauOff: document.querySelector('.gauges g[data-joint="3"] .tau')
      ?.classList.contains("off"),
    feet: document.querySelector('.gauges [data-foot="0"]')?.textContent,
  }));
  if (!posed.ang || Math.abs(parseFloat(posed.ang) - posed.slider) > 0.51
      || posed.om !== "ω 0.0" || !posed.tauOff || posed.feet !== "n/a")
    throw new Error("pose-mode gauges do not show the pose: " + JSON.stringify(posed));
  await page.screenshot({ path: "test/out/pose.png" });
  await page.keyboard.press("p"); // leave pose mode
  await page.waitForFunction(() => !document.querySelector("#pose"),
    { timeout: 5000 });

  const input = await page.$("input[type=file]");
  await input.uploadFile(replay);
  await page.waitForFunction(
    () => window.__viewerState?.().loaded,
    { timeout: 10000 });
  await new Promise((r) => setTimeout(r, 1500)); // let it play a bit
  await page.screenshot({ path: "test/out/playing.png" });
  // the top of the page no longer carries the replay's name / summary: the
  // picker reads as a plain "open" action once a replay is loaded, and the
  // name is still known to the viewer
  const loaded = await page.evaluate(() => ({
    drop: document.querySelector("#drop")?.textContent.replace(/\s+/g, " ").trim(),
    name: window.__viewerState().name,
  }));
  loaded.bar = await barLabels(page); // accessible names (icon buttons)
  console.log("loaded:", JSON.stringify(loaded));
  if (loaded.drop !== "📂 open replay…")
    throw new Error("the picker should read as an open action: " + loaded.drop);
  if (loaded.name !== basename(replay).replace(/\.rpl\.gz$|\.[^.]*$/, ""))
    throw new Error("unexpected replay name in state: " + loaded.name);
  if (loaded.bar.some((b) => /export|stop & save|fps/.test(b)))
    throw new Error("video export controls are back in the bar: " + loaded.bar.join(" | "));

  // ---- skeleton + gauges + contact-marker eyeball shots ('h' hitboxes on):
  // scrub to a mid-episode tick (the bot swinging, weight on the welds —
  // gauges show τ bars, foot N and joint activation) and to a late tick.
  await page.keyboard.press("h");
  // the timeline: a pointer press on its track + arrow keys (test/bar.js)
  const scrub = (tick) => scrubTo(page, tick);
  await scrub(100);
  await new Promise((r) => setTimeout(r, 500));
  await page.screenshot({ path: "test/out/skeleton-gauges.png" });
  await checkGauges(page, replay, 100);
  await scrub(300);
  await new Promise((r) => setTimeout(r, 500));
  await page.screenshot({ path: "test/out/late.png" });

  const vs = await page.evaluate(() => window.__viewerState());
  console.log("viewer:", JSON.stringify(vs));
  if (vs.tick !== 300 || !(vs.t > 0)) throw new Error("time not advancing");

  // ---- gauges button: mirrors the 'g' key (the panels are on at load)
  const GAUGES_ON = toggleOn("gauges");
  if (!(await page.evaluate(GAUGES_ON))) throw new Error("gauges button not lit at load");
  if (!(await clickBar(page, "gauges"))) throw new Error("no gauges toggle in the bar");
  await page.waitForFunction(`!(${GAUGES_ON})`, { timeout: 5000 });
  await page.waitForFunction(() => !document.querySelector(".gauges"), { timeout: 5000 });
  await page.keyboard.press("g"); // the key toggles the same flag back on
  await page.waitForFunction(GAUGES_ON, { timeout: 5000 });
  await page.waitForSelector(".gauges .panel", { timeout: 5000 });

  // ---- first-person ('f'): the camera sits at the head, fixed forward. The bar
  // button lights up, the frame renders with no page errors, and 'f'
  // again returns to the orbit camera.
  // (a string expression: waitForFunction cannot be handed a closure)
  const FP_ON = toggleOn("first-person");
  await page.keyboard.press("f");
  await page.waitForFunction(FP_ON, { timeout: 5000 });
  await new Promise((r) => setTimeout(r, 500));
  await page.screenshot({ path: "test/out/first-person.png" });
  // the gauges do not depend on the camera: same panels, same numbers
  await checkGauges(page, replay, 300);
  // the one first-person slider (fov) lives in #fp — no rotation smoothing
  // since the camera no longer turns with the head
  const fpSliders = await page.$$eval("#fp [role=slider]", (els) => els.length);
  if (fpSliders !== 1) throw new Error(`expected 1 first-person slider, got ${fpSliders}`);
  await sliderToMax(page, "#fp [role=slider]");  // fov 110° (the End key)
  await new Promise((r) => setTimeout(r, 500));
  const fpLabels = await page.$eval("#fp", (el) => el.textContent.replace(/\s+/g, " "));
  if (!/110°/.test(fpLabels) || /smooth/.test(fpLabels))
    throw new Error("first-person fov slider did not take: " + fpLabels);
  await page.screenshot({ path: "test/out/first-person-wide.png" });
  await page.keyboard.press("f");
  await page.waitForFunction(`!(${FP_ON})`, { timeout: 5000 });
  await new Promise((r) => setTimeout(r, 300));
  await page.screenshot({ path: "test/out/after-first-person.png" });

  // ---- env v15 head-anchored track: same bundle, a replay whose header
  // carries "track_anchor": "head". The notes and the tunnel/grid/reach
  // scenery ride head_z - PLANE_Z0, so a tick where the bot has moved in z
  // shows the whole track displaced while the floor stays put.
  if (existsSync(replayV15)) {
    await input.uploadFile(replayV15);
    await page.waitForFunction(
      () => window.__viewerState?.().name?.includes("v15"),
      { timeout: 10000 });
    await new Promise((r) => setTimeout(r, 800));
    await scrub(60);
    await new Promise((r) => setTimeout(r, 500));
    await page.screenshot({ path: "test/out/v15-head-anchored.png" });
    const vs15 = await page.evaluate(() => window.__viewerState());
    console.log("viewer (v15):", JSON.stringify(vs15));
    if (vs15.tick !== 60)
      throw new Error("v15 replay did not scrub");

    // ---- live A/V sync: the song must RUN, not be seeked about.
    // This replay names a map, so if BeatSaver is reachable the viewer has a
    // real song here. The 2026-09-14 chop was one el.currentTime write per
    // animation frame; the fix seeks only on a deliberate jump, so a couple
    // of seconds of plain playback must produce ~no seeks at all. Skipped
    // (not failed) when there is no network, or when autoplay never starts.
    const ready = await page.evaluate(() => !!document.querySelector(
      "#snd [role=slider]"));
    if (ready) {
      await page.evaluate(() => {
        const a = document.querySelector("audio");
        window.__seeks = 0;
        a.addEventListener("seeking", () => { window.__seeks++; });
      });
      await page.keyboard.press("r"); // restart: exactly one deliberate seek
      await new Promise((r) => setTimeout(r, 300));
      const t0 = await page.evaluate(() => {
        window.__seeks = 0; // the restart's own seek is not drift
        return document.querySelector("audio").currentTime;
      });
      const wall0 = Date.now();
      await new Promise((r) => setTimeout(r, 2000));
      const wall = (Date.now() - wall0) / 1000;
      const { seeks, t1, rate } = await page.evaluate(() => ({
        seeks: window.__seeks,
        t1: document.querySelector("audio").currentTime,
        rate: document.querySelector("audio").playbackRate,
      }));
      if (t1 - t0 > 0.5) {
        console.log(`live sync: song advanced ${(t1 - t0).toFixed(2)}s of ` +
          `${wall.toFixed(2)}s wall with ${seeks} seek(s), ` +
          `rate ${rate.toFixed(3)}`);
        // a song that runs free covers real time; one being seeked about
        // spends its life re-buffering and falls behind the wall clock
        if ((t1 - t0) / wall < 0.6)
          throw new Error(`the song only advanced ${(t1 - t0).toFixed(2)}s in ` +
            `${wall.toFixed(2)}s — it is stalling, not playing`);
        if (seeks > 2)
          throw new Error(`the song was seeked ${seeks} times during plain ` +
            "playback — that is the chop");
        if (Math.abs(rate - 1) > 0.05)
          throw new Error("playbackRate left far from 1: " + rate);
      } else {
        console.log("live sync: the song never sounded here — check skipped");
      }
    } else {
      console.log("live sync: no song loaded (offline?) — check skipped");
    }
  } else {
    console.log("no v15 recording at " + replayV15 + " — skipped");
  }

  const webgl = await page.evaluate(() => {
    const cv = document.querySelector("#stage canvas");
    return !!(cv.getContext("webgl2") || cv.getContext("webgl"));
  });
  if (!webgl) throw new Error("no WebGL context on the stage canvas");
  if (errors.length) throw new Error("browser errors:\n" + errors.join("\n"));
  console.log("browser smoke test OK (screenshots in test/out/)");
} finally {
  await browser.close();
}
