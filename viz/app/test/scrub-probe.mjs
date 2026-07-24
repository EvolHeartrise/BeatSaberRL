// Full-playthrough probe: scrub through the whole replay in steps,
// reporting the first page error and the tick where it appeared.
// Run: node test/scrub-probe.mjs <replay.jsonl> [step]
import { resolve } from "node:path";
import puppeteer from "puppeteer-core";

const EDGE = "C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe";
const replay = resolve(process.argv[2]);
const step = +(process.argv[3] ?? 50);

const browser = await puppeteer.launch({
  executablePath: EDGE,
  headless: true,
  args: ["--use-angle=swiftshader", "--disable-gpu-sandbox"],
});
try {
  const page = await browser.newPage();
  const errors = [];
  page.on("pageerror", (e) => errors.push("pageerror: " + e.message));
  page.on("console", (m) => {
    if (m.type() === "error") errors.push("console: " + m.text());
  });
  await page.goto("file:///" + resolve("dist/index.html").replace(/\\/g, "/"));
  await page.waitForSelector("#stage canvas", { timeout: 15000 });
  const input = await page.$("input[type=file]");
  await input.uploadFile(replay);
  await page.waitForFunction(
    () => document.querySelector("#hud").textContent.includes("tick"),
    { timeout: 10000 });
  const maxTick = await page.evaluate(() => {
    const r = document.querySelector("#bottom input[type=range]");
    return +r.max;
  });
  const scrub = (t) => page.evaluate((t) => {
    const r = document.querySelector("#bottom input[type=range]");
    Object.getOwnPropertyDescriptor(HTMLInputElement.prototype, "value")
      .set.call(r, t);
    r.dispatchEvent(new Event("input", { bubbles: true }));
  }, t);
  for (let t = 0; t <= maxTick; t += step) {
    await scrub(t);
    await new Promise((r) => setTimeout(r, 25));
    if (errors.length) {
      console.log(`FIRST ERROR at tick <= ${t}:`);
      console.log(errors.slice(0, 4).join("\n"));
      process.exit(1);
    }
  }
  console.log(`full scrub clean (0..${maxTick} step ${step}), no errors`);
} finally {
  await browser.close();
}
