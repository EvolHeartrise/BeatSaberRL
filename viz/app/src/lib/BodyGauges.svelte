<script>
  // Per-joint instrumentation, generalized to all 29 v10 joints and
  // grouped spine / L arm / R arm / L leg / R leg. Each joint row: a
  // mini-dial (allowed arc lo→hi, ω-aware needle, orange within 10° of
  // a limit), the angle + angular velocity, and an applied-τ bar
  // (τ = gear_fraction × τ_max from the tick's logged action — τ is
  // exactly derivable, not separately logged). NEW force block: per-
  // foot vertical ground reaction (N, from the additive "feet" weld
  // impulses) and the joint-activation meter (commanded Σ|gear
  // fraction|, 0..29 — the --iso-coef base; record_rev 3 "activation"
  // field, derived from the action for older replays).
  //
  // A DOM overlay over the WebGL canvas (2026-10-02; until then two
  // canvas-texture sprites hung off the camera, re-drawn and re-uploaded
  // every frame — canvas only so that a video export of the canvas kept
  // them, and the export is gone). Left panel: spine + L arm + L leg.
  // Right panel: R arm + R leg + forces. App.svelte mounts it whenever
  // there is a body to instrument.
  //
  // The markup is built ONCE: the rows, dials and bars never change shape,
  // only the values in them. The frame loop writes those values straight
  // into the elements — a needle's end point, a bar's end, a label's text —
  // and only the ones that changed: no per-frame Svelte re-render of 29
  // rows. Each panel is one SVG (arcs, needles, bars) under HTML text, laid
  // out in the old canvas's pixel units (252 wide) and scaled as a whole to
  // the on-screen size the sprites had.
  import { pb, pose } from "./state.svelte.js";
  import {
    GROUPS, JOINT_LIMITS, JOINT_NAMES, OMEGA_CAP, SABER_COLORS,
    TAU_MAX, activationAt, footForces, gearTable, omegaCapOf, qAt,
  } from "./replay.js";

  const W = 252;
  const ROW_H = 25, HEAD_H = 20, PAD = 10;
  const NEAR = 10 * Math.PI / 180; // limit-proximity warning band
  const deg = (r) => (r * 180 / Math.PI);

  const PANELS = [
    { groups: [0, 1, 3], forces: false },  // spine, L arm, L leg
    { groups: [2, 4], forces: true },      // R arm, R leg + forces
  ];
  const FORCES_H = 112;  // 2 foot rows + activation bar + group line
  const panelH = (p) =>
    2 * PAD +
    p.groups.reduce((a, g) => a + HEAD_H + GROUPS[g].joints.length * ROW_H, 0) +
    (p.forces ? HEAD_H + FORCES_H : 0);

  const groupColor = (g) =>
    g === 1 || g === 3 ? SABER_COLORS[0]
    : g === 2 || g === 4 ? SABER_COLORS[1]
    : "#c5b3f0";

  // ---- layout, in panel units ----------------------------------------------
  // dial: centre (24, row middle), radius 10; 0 points up and positive
  // angles turn clockwise, as on the canvas. Every v10 range fits in ±180°,
  // so the allowed arc is one SVG arc lo→hi.
  const R = 10, DX = 24;
  const at = (cy, a, r) => [DX + r * Math.sin(a), cy - r * Math.cos(a)];
  function arcD(cy, [lo, hi]) {
    const [x0, y0] = at(cy, lo, R), [x1, y1] = at(cy, hi, R);
    return `M${x0} ${y0}A${R} ${R} 0 ${hi - lo > Math.PI ? 1 : 0} 1 ${x1} ${y1}`;
  }
  const radial = (cy, a, r0, r1) => {
    const [x1, y1] = at(cy, a, r0), [x2, y2] = at(cy, a, r1);
    return { x1, y1, x2, y2 };
  };
  // τ bar: centre-anchored, ± τ_max full-scale
  const BX0 = 140, BX1 = W - 14, BMID = (BX0 + BX1) / 2;
  // force bars: per-foot vertical reaction (full scale 1200 N) and the
  // activation meter (full scale 29 = every joint at full gear)
  const FX0 = 60, FX1 = W - 64, FS = 1200;
  const forceY = (y0, row) => y0 + HEAD_H + 4 + 24 * row;

  // each panel's blocks top to bottom: group headings, joint rows (cy =
  // the row's middle) and the force block
  const panels = PANELS.map((p) => {
    const heads = [], joints = [];
    let y = PAD;
    for (const g of p.groups) {
      heads.push({ g, y });
      y += HEAD_H;
      for (const j of GROUPS[g].joints) {
        joints.push({ g, j, cy: y + ROW_H / 2 });
        y += ROW_H;
      }
    }
    return { ...p, heads, joints, forcesY: p.forces ? y : null, h: panelH(p) };
  });

  // ---- the elements the frame loop writes ----------------------------------
  // per joint: needle, ω tick, angle text, ω text, τ bar (and the row's
  // centre); the force block's bars and texts. Each write remembers what it
  // last wrote, so a value that did not change costs a comparison and no
  // DOM write.
  const J = Array.from({ length: 29 }, () => ({}));
  const F = { feet: [{}, {}], act: {} };
  const last = new Map();
  function put(el, key, v, write) {
    if (!el) return;
    let m = last.get(el);
    if (!m) last.set(el, (m = {}));
    if (m[key] === v) return;
    m[key] = v;
    write(el, v);
  }
  // text changes in place (the span keeps its one text node)
  const text = (el, s) => put(el, "t", s, (e, v) => {
    if (e.firstChild) e.firstChild.data = v;
    else e.textContent = v;
  });
  const attr = (el, name, v) => put(el, name, v, (e, s) => e.setAttribute(name, s));
  // (classList, not the class attribute: that would drop Svelte's scoping class)
  const cls = (el, name, on) => put(el, name, on, (e, v) => e.classList.toggle(name, v));
  // a line's end points — plain attributes, not CSS properties, so moving
  // one never costs a style recalc (a `transform` would)
  function ends(el, { x1, y1, x2, y2 }) {
    attr(el, "x1", x1.toFixed(2)); attr(el, "y1", y1.toFixed(2));
    attr(el, "x2", x2.toFixed(2)); attr(el, "y2", y2.toFixed(2));
  }

  /** One joint's dial + numbers + τ bar (τ as the fraction of its limit). */
  function joint(j, q, om, t, cap = OMEGA_CAP[j]) {
    const [lo, hi] = JOINT_LIMITS[j];
    const r = J[j];
    const hot = Math.min(q - lo, hi - q) < NEAR;
    cls(r.needle, "hot", hot);
    cls(r.omTick, "hot", hot);
    ends(r.needle, radial(r.cy, q, 0, R - 1));
    // ω needle: short outer tick at q + ω·(1/8 s), clamped to the range
    ends(r.omTick, radial(r.cy, Math.min(hi, Math.max(lo, q + om / 8)), R + 1, R + 4));
    text(r.ang, `${deg(q).toFixed(0)}°`);
    text(r.om, `ω ${om.toFixed(1)}`);
    cls(r.om, "hot", Math.abs(om) > 0.9 * cap);
    cls(r.tau, "off", t === 0);
    cls(r.tau, "neg", t < 0);
    if (t !== 0) attr(r.tau, "x2", (BMID + t * (BX1 - BX0) / 2).toFixed(2));
  }

  function forces(feet, act) {
    for (const i of [0, 1]) {
      const n = feet ? feet[i] : null, f = F.feet[i];
      cls(f.bar, "off", n === null);
      cls(f.bar, "neg", n !== null && n < 0);
      if (n !== null)
        attr(f.bar, "x2", (FX0 + Math.max(0, Math.min(1, n / FS)) * (FX1 - FX0)).toFixed(2));
      text(f.val, n === null ? "n/a" : `${n.toFixed(0)} N`);
      cls(f.val, "na", n === null);
    }
    const a = F.act;
    cls(a.bar, "off", act === null);
    if (act !== null)
      attr(a.bar, "x2", (FX0 + Math.max(0, Math.min(1, act[0] / 29)) * (FX1 - FX0)).toFixed(2));
    text(a.val, act === null ? "n/a" : `${act[0].toFixed(1)}/29`);
    cls(a.val, "na", act === null);
    // group split on the next line: spine / arms / legs
    text(a.split, act === null ? "" : `spine ${act[1].toFixed(1)}  arms ${act[2].toFixed(1)}  `
      + `legs ${act[3].toFixed(1)}`);
  }

  // The frame loop: Scene.svelte's clock moves pb.cur every frame while
  // playing, which re-runs this; nothing is allocated per joint.
  const qNow = new Array(36); // qAt scratch
  $effect(() => {
    if (pose.on) { // gauges reflect the manually posed joints (ω = τ = 0)
      const q = pose.q;
      for (let j = 0; j < 29; j++) joint(j, q[j], 0, 0);
      forces(null, null);
      return;
    }
    const rep = pb.rep, cur = pb.cur;
    if (!rep) return;
    const n = rep.ticks.length;
    const k = Math.min(Math.floor(cur), n - 1);
    const k1 = Math.min(k + 1, n - 1);
    const u = cur - k;
    // the angles: the drawn pose (qAt, the joint-space Hermite)
    const q = qAt(rep, k, u, qNow), tb = rep.ticks[k1];
    // applied τ (tauOf, one joint at a time) as the fraction of this
    // joint's limit, and the ω cap — the replay body's (replayBody: per
    // joint, so original-body and v63x2-body replays both read right),
    // else the nominal tables
    const body = rep.body, table = body ? null : gearTable(rep.config.gears ?? 3);
    for (let j = 0; j < 29; j++) {
      const a = tb.action[j];
      const tau = body ? (body.gear_frac[j][a] ?? 0) * body.tau_max[j]
        : (table[a] ?? 0) * TAU_MAX[j];
      const t = tau === 0 ? 0
        : Math.max(-1, Math.min(1, tau / (body?.tau_max[j] ?? TAU_MAX[j])));
      joint(j, q[7 + j], tb.omega[j], t, omegaCapOf(rep, j));
    }
    forces(footForces(rep, k1), activationAt(rep, k1));
  });

  // ---- on-screen placement -------------------------------------------------
  // The sprites were sized in camera space: 0.62 units tall at z = -1 under
  // the orbit camera's 55° fov (rescaled with the fov on the first-person
  // camera, so the same on-screen size at any fov), anchored 0.015 units in
  // from the side and just above the floating bottom UI. As fractions of the
  // stage height that is:
  const VIEW_H = 2 * Math.tan((55 / 2) * Math.PI / 180); // camera units
  const SCREEN_H = 0.62 / VIEW_H;  // each panel's height
  const MARGIN = 0.015 / VIEW_H;   // gap to the left / right edge
  // the control bar wrapped to two rows (buttons + timeline) is ~97 px
  // tall with its padding; keep a small gap above it
  const BOTTOM_UI_PX = 104;
  let stageH = $state(0);
</script>

<div class="gauges" bind:clientHeight={stageH}>
  {#each panels as p, i (i)}
    <div class="panel" class:right={i === 1}
      style:height="{p.h}px"
      style:bottom="{BOTTOM_UI_PX}px"
      style:--margin="{MARGIN * stageH}px"
      style:transform="scale({(SCREEN_H * stageH) / p.h})">
      <!-- two SVGs: what never changes (arcs, limit ticks, bar tracks) and
           what the frame loop moves (needles, ω ticks, bars) — the browser
           then keeps the static one's paint and re-records only the other -->
      <svg width={W} height={p.h} viewBox="0 0 {W} {p.h}" aria-hidden="true">
        {#each p.joints as { j, cy } (j)}
          <path class="arc" d={arcD(cy, JOINT_LIMITS[j])} />
          {#each JOINT_LIMITS[j] as b, s (s)}
            <line class="lim" {...radial(cy, b, R - 3, R + 3)} />
          {/each}
          <line class="track" x1={BX0} x2={BX1} y1={cy} y2={cy} />
          <line class="mid" x1={BMID} x2={BMID} y1={cy - 5} y2={cy + 5} />
        {/each}
        {#if p.forces}
          {#each [0, 1, 2] as row (row)}
            {@const y = forceY(p.forcesY, row)}
            <line class="ftrack" x1={FX0} x2={FX1} y1={y} y2={y} />
          {/each}
        {/if}
      </svg>
      <svg width={W} height={p.h} viewBox="0 0 {W} {p.h}" aria-hidden="true">
        {#each p.joints as { g, j, cy } (j)}
          {@const r = Object.assign(J[j], { cy })}
          <g style:--c={groupColor(g)} data-joint={j}>
            <line class="needle" bind:this={r.needle} />
            <line class="omtick" bind:this={r.omTick} />
            <line class="tau off" x1={BMID} x2={BMID} y1={cy} y2={cy}
              bind:this={r.tau} />
          </g>
        {/each}
        {#if p.forces}
          {#each [0, 1, 2] as row (row)}
            {@const y = forceY(p.forcesY, row)}
            {#if row < 2}
              <line class="fill off" x1={FX0} x2={FX0} y1={y} y2={y}
                bind:this={F.feet[row].bar} />
            {:else}
              <line class="fill act off" x1={FX0} x2={FX0} y1={y} y2={y}
                bind:this={F.act.bar} />
            {/if}
          {/each}
        {/if}
      </svg>
      {#each p.heads as { g, y } (g)}
        <span class="head" style:top="{y + 8}px" style:color={groupColor(g)}>
          {GROUPS[g].name}</span>
      {/each}
      {#each p.joints as { j, cy } (j)}
        <span class="name" style:top="{cy - 5}px">{JOINT_NAMES[j]}</span>
        <span class="ang" style:top="{cy + 7}px" bind:this={J[j].ang} data-joint={j}> </span>
        <span class="om" style:top="{cy + 7}px" bind:this={J[j].om} data-joint={j}> </span>
      {/each}
      {#if p.forces}
        {@const y0 = p.forcesY}
        <span class="head fh" style:top="{y0 + 8}px">forces</span>
        {#each ["L foot", "R foot", "activ."] as name, row (row)}
          <span class="fname" style:top="{forceY(y0, row)}px">{name}</span>
          {#if row < 2}
            <span class="val na" style:top="{forceY(y0, row)}px"
              bind:this={F.feet[row].val} data-foot={row}> </span>
          {:else}
            <span class="val na" style:top="{forceY(y0, row)}px"
              bind:this={F.act.val} data-act=""> </span>
          {/if}
        {/each}
        <span class="split" style:top="{forceY(y0, 2) + 20}px"
          bind:this={F.act.split}> </span>
      {/if}
    </div>
  {/each}
</div>

<style>
  /* the stage-sized layer the two corner panels hang in: under every other
     overlay (they are drawn over it, as they were over the sprites) and
     never in the way of the orbit drag */
  .gauges {
    position: absolute; inset: 0; z-index: 1;
    pointer-events: none; overflow: hidden;
    font-family: system-ui, sans-serif; line-height: 1;
    user-select: none;
  }
  .panel {
    position: absolute; left: var(--margin); width: 252px;
    transform-origin: bottom left;
    background: rgba(11, 13, 17, 0.72); border-radius: 14px;
    contain: strict; /* the per-frame writes lay out + paint inside it only */
  }
  .panel.right { left: auto; right: var(--margin); transform-origin: bottom right; }
  svg { position: absolute; left: 0; top: 0; }

  /* text sits on its vertical middle at `top` (half its 1 em box up), as
     the canvas drew it — plus 0.1 em: textBaseline "middle" is the middle
     of the em box, while a CSS line box centres the taller ascent + descent
     box, ~0.1 em lower */
  span {
    position: absolute; height: 1em; margin-top: -0.6em; white-space: pre;
  }
  /* the spans rewritten every frame are sized boxes with strict
     containment, so a new number lays out that box alone */
  .ang, .om, .val, .split { contain: strict; }
  .head { left: 10px; font-size: 12px; font-weight: bold; }
  .fh { color: #e8eaf0; }

  /* the dial: allowed arc, orange limit ticks, needle + ω tick in the
     group colour (orange within 10° of a limit) */
  line, path { fill: none; stroke-linecap: round; }
  .arc { stroke: #3a4050; stroke-width: 3.5; }
  .lim { stroke: #f0a35e; stroke-width: 1.5; }
  .needle { stroke: var(--c); stroke-width: 2.5; }
  .omtick { stroke: var(--c); stroke-width: 1; }
  .needle.hot, .omtick.hot { stroke: #f0a35e; }
  .track { stroke: #3a4050; stroke-width: 4; }
  .mid { stroke: #6b7180; stroke-width: 1; }
  .tau { stroke: #7dd3a0; stroke-width: 5; }
  .tau.neg { stroke: #e06c75; }
  .off { visibility: hidden; }

  .name { left: 40px; font-size: 11px; color: #9aa0ae; }
  .ang, .om { font-size: 10px; color: #6b7180; }
  .ang { left: 40px; width: 40px; }
  .om { left: 84px; width: 54px; }
  .om.hot { color: #f0a35e; }

  .ftrack { stroke: #3a4050; stroke-width: 5; }
  .fill { stroke: #7aa2f7; stroke-width: 6; }
  .fill.neg { stroke: #e06c75; }
  .fill.act { stroke: #c5b3f0; }
  .fname { left: 10px; font-size: 11px; color: #9aa0ae; }
  .val { left: 194px; width: 56px; font-size: 11px; color: #cfd3dc; }
  .val.na { color: #6b7180; }
  .split { left: 60px; width: 190px; font-size: 10px; color: #9aa0ae; }
</style>
