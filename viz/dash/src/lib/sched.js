// Chart work in small slices (2026-10-06, user: a refresh froze hover,
// zoom and drag). A data change used to rebuild / setData every chart in
// one task — 24 panels on Live, 40 on Compare with 16 runs overlaid —
// while input waited. Charts queue their work here instead (one entry per
// chart: newer work replaces older); the queue runs a few charts at a
// time, ~8 ms per slice, and yields to the browser between slices so
// pending input and frames go first.
const queue = new Map(); // owner -> fn
let running = false;
const BUDGET_MS = 8;

const yieldToMain = () =>
  globalThis.scheduler?.yield ? globalThis.scheduler.yield() : new Promise((r) => setTimeout(r, 0));

export function schedule(owner, fn) {
  queue.delete(owner); // newest work goes to the back: oldest waits least
  queue.set(owner, fn);
  if (!running) {
    running = true;
    setTimeout(pump, 0);
  }
}

export function cancel(owner) {
  queue.delete(owner);
}

async function pump() {
  try {
    while (queue.size) {
      const t0 = performance.now();
      while (queue.size && performance.now() - t0 < BUDGET_MS) {
        const [owner, fn] = queue.entries().next().value;
        queue.delete(owner);
        try {
          fn();
        } catch (e) {
          console.error(e);
        }
        // let the chart's queued draw (uPlot commits in a microtask) run
        // now, inside this task, so the budget counts it
        await null;
      }
      if (queue.size) await yieldToMain();
    }
  } finally {
    running = false;
  }
}

// test hook: resolves once no chart work is queued
export const idle = () => new Promise((res) => {
  const check = () => (queue.size || running ? setTimeout(check, 10) : res());
  check();
});
globalThis.__dashChartsIdle = idle;
