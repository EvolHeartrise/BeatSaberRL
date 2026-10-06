// Puppeteer helpers for the replay viewer's control bar (src/lib/
// ControlBar.svelte: shadcn-svelte / bits-ui, icon buttons with aria-labels,
// toggles with aria-pressed, role="slider" thumbs instead of <input
// type=range>). They drive it the way a person does — pointer and keyboard —
// so a test exercises the real controls, not a value poked into the DOM.

/** CSS selector of a bar control by its accessible name ("pause",
 * "gauges", "tip trail", ...). */
export const barButton = (label) => `#bar button[aria-label="${label}"]`;

/** A string expression (for page.waitForFunction / evaluate): is the bar
 * toggle `label` on? */
export const toggleOn = (label) =>
  `document.querySelector('${barButton(label)}')?.getAttribute("aria-pressed") === "true"`;

/** Click the bar control `label` if it is there (a no-op otherwise, like the
 * old `.find(...)?.click()`). Resolves to whether it was clicked. */
export function clickBar(page, label) {
  return page.evaluate((sel) => {
    const b = document.querySelector(sel);
    b?.click();
    return !!b;
  }, barButton(label));
}

/** Every bar button's accessible name (aria-label, else its text). */
export function barLabels(page) {
  return page.evaluate(() => [...document.querySelectorAll("#bar button, #bar select")]
    .map((b) => (b.getAttribute("aria-label") ?? b.textContent).trim()));
}

/** Seek the timeline to `tick` ("max" = the last tick; past-the-end ticks
 * clamp, as the old range input did) like a user: a pointer press at the
 * tick's spot on the track (which pauses and seeks, as dragging always has),
 * then the arrow keys on the focused thumb — one tick per press — to land
 * on it exactly. Throws unless the slider reads `tick` afterwards. Leaves
 * focus and the pointer off the bar. Resolves to the tick landed on. */
export async function scrubTo(page, tick) {
  const geo = await page.$eval("#seek [data-slot=slider]", (root) => {
    const r = root.getBoundingClientRect();
    const th = root.querySelector("[role=slider]");
    return { x: r.left, y: r.top + r.height / 2, w: r.width,
      min: +th.getAttribute("aria-valuemin"), max: +th.getAttribute("aria-valuemax") };
  });
  const want = tick === "max" ? geo.max : Math.min(Math.max(tick, geo.min), geo.max);
  const frac = (want - geo.min) / Math.max(1, geo.max - geo.min);
  // (kept a pixel inside the track: a press on its very edge misses it)
  await page.mouse.click(Math.min(Math.max(geo.x + frac * geo.w, geo.x + 1), geo.x + geo.w - 1),
    geo.y);
  await page.focus("#seek [role=slider]"); // the press focused it; make sure
  if (want === geo.max) await page.keyboard.press("End");
  else if (want === geo.min) await page.keyboard.press("Home");
  const now = () => page.$eval("#seek [role=slider]", (t) => +t.getAttribute("aria-valuenow"));
  for (let guard = 0; guard < 400; guard++) {
    const v = await now();
    if (v === want) break;
    await page.keyboard.press(v < want ? "ArrowRight" : "ArrowLeft");
  }
  const got = await now();
  if (got !== want) throw new Error(`timeline scrub to ${want} stuck at ${got}`);
  await page.evaluate(() => document.activeElement?.blur?.());
  await page.mouse.move(1, 1);
  return got;
}

/** Push the slider whose thumb matches `thumbSel` to its maximum (End). */
export async function sliderToMax(page, thumbSel) {
  await page.focus(thumbSel);
  await page.keyboard.press("End");
  await page.evaluate(() => document.activeElement?.blur?.());
}
