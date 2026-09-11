<script>
  // Per-joint instrumentation, generalized to all 29 v10 joints and
  // grouped spine / L arm / R arm / L leg / R leg. Each joint row: a
  // mini-dial (allowed arc lo→hi, ω-aware needle, orange within 10° of
  // a limit), the angle + angular velocity, and an applied-τ bar
  // (τ = gear_fraction × τ_max from the tick's logged action — τ is
  // exactly derivable, not separately logged). NEW force block: per-
  // foot vertical ground reaction (N, from the additive "feet" weld
  // impulses) and the effort-per-tick meter (J of positive work).
  //
  // Drawn as canvas-texture sprites ATTACHED TO THE CAMERA: they stay
  // screen-fixed under orbiting, show up in exported videos, and need
  // no DOM/fonts (offline + export safe) — same rationale as the score
  // popups. Left sprite: spine + L arm + L leg. Right sprite: R arm +
  // R leg + forces.
  import { T, useThrelte } from "@threlte/core";
  import * as THREE from "three";
  import { pb, pose } from "./state.svelte.js";
  import {
    DT, GROUPS, JOINT_LIMITS, JOINT_NAMES, OMEGA_CAP, SABER_COLORS,
    TAU_MAX, effortAt, footForces, lerp, tauOf,
  } from "./replay.js";

  // the fov of the camera these sprites hang off (the corner placement
  // below is in camera space, so a wider camera needs a wider anchor)
  let { fov = 55 } = $props();

  const W = 252;
  const ROW_H = 25, HEAD_H = 20, PAD = 10;
  const NEAR = 10 * Math.PI / 180; // limit-proximity warning band
  const deg = (r) => (r * 180 / Math.PI);

  const PANELS = [
    { groups: [0, 1, 3], forces: false },  // spine, L arm, L leg
    { groups: [2, 4], forces: true },      // R arm, R leg + forces
  ];
  const FORCES_H = 92;
  const panelH = (p) =>
    2 * PAD +
    p.groups.reduce((a, g) => a + HEAD_H + GROUPS[g].joints.length * ROW_H, 0) +
    (p.forces ? HEAD_H + FORCES_H : 0);

  const panels = PANELS.map((p) => {
    const cv = document.createElement("canvas");
    cv.width = W;
    cv.height = panelH(p);
    const tex = new THREE.CanvasTexture(cv);
    tex.colorSpace = THREE.SRGBColorSpace;
    return { ...p, cv, ctx: cv.getContext("2d"), tex, h: panelH(p) };
  });

  const groupColor = (g) =>
    g === 1 || g === 3 ? SABER_COLORS[0]
    : g === 2 || g === 4 ? SABER_COLORS[1]
    : "#c5b3f0";

  function label(ctx, text, x, y, color = "#9aa0ae", size = 12, bold = false,
                 align = "left") {
    ctx.fillStyle = color;
    ctx.font = `${bold ? "bold " : ""}${size}px system-ui, sans-serif`;
    ctx.textAlign = align;
    ctx.textBaseline = "middle";
    ctx.fillText(text, x, y);
  }

  /** One joint row at baseline y: dial + name + numbers + τ bar. */
  function jointRow(ctx, y, j, q, om, tau, col) {
    const [lo, hi] = JOINT_LIMITS[j];
    const cx = 24, cy = y, r = 10;
    // dial: allowed arc drawn lo→hi with 0 pointing up (canvas angle
    // -PI/2 + angle; every v10 range fits in ±180°)
    const cAng = (p) => -Math.PI / 2 + p;
    ctx.lineWidth = 3.5;
    ctx.strokeStyle = "#3a4050";
    ctx.beginPath();
    ctx.arc(cx, cy, r, cAng(lo), cAng(hi));
    ctx.stroke();
    for (const b of [lo, hi]) { // limit ticks
      ctx.strokeStyle = "#f0a35e";
      ctx.lineWidth = 1.5;
      ctx.beginPath();
      ctx.moveTo(cx + (r - 3) * Math.cos(cAng(b)), cy + (r - 3) * Math.sin(cAng(b)));
      ctx.lineTo(cx + (r + 3) * Math.cos(cAng(b)), cy + (r + 3) * Math.sin(cAng(b)));
      ctx.stroke();
    }
    const hot = Math.min(q - lo, hi - q) < NEAR;
    const nc = hot ? "#f0a35e" : col;
    ctx.strokeStyle = nc;
    ctx.lineWidth = 2.5;
    ctx.lineCap = "round";
    ctx.beginPath();
    ctx.moveTo(cx, cy);
    ctx.lineTo(cx + (r - 1) * Math.cos(cAng(q)), cy + (r - 1) * Math.sin(cAng(q)));
    ctx.stroke();
    // ω needle: short outer tick at the needle + q + ω·(1/8 s) preview
    const qn = Math.min(hi, Math.max(lo, q + om / 8));
    ctx.strokeStyle = nc;
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(cx + (r + 1) * Math.cos(cAng(qn)), cy + (r + 1) * Math.sin(cAng(qn)));
    ctx.lineTo(cx + (r + 4) * Math.cos(cAng(qn)), cy + (r + 4) * Math.sin(cAng(qn)));
    ctx.stroke();
    label(ctx, JOINT_NAMES[j], 40, y - 5, "#9aa0ae", 11);
    label(ctx, `${deg(q).toFixed(0)}°`, 40, y + 7, "#6b7180", 10);
    label(ctx, `ω ${om.toFixed(1)}`, 84, y + 7,
      Math.abs(om) > 0.9 * OMEGA_CAP[j] ? "#f0a35e" : "#6b7180", 10);
    // τ bar: center-anchored, ± τ_max full-scale
    const bx0 = 140, bx1 = W - 14, bmid = (bx0 + bx1) / 2;
    ctx.strokeStyle = "#3a4050";
    ctx.lineWidth = 4;
    ctx.beginPath();
    ctx.moveTo(bx0, y);
    ctx.lineTo(bx1, y);
    ctx.stroke();
    ctx.strokeStyle = "#6b7180";
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(bmid, y - 5);
    ctx.lineTo(bmid, y + 5);
    ctx.stroke();
    if (tau !== 0) {
      const t = Math.max(-1, Math.min(1, tau / TAU_MAX[j]));
      ctx.strokeStyle = t > 0 ? "#7dd3a0" : "#e06c75";
      ctx.lineWidth = 5;
      ctx.beginPath();
      ctx.moveTo(bmid, y);
      ctx.lineTo(bmid + t * (bx1 - bx0) / 2, y);
      ctx.stroke();
    }
  }

  function forcesBlock(ctx, y0, feet, effort) {
    label(ctx, "forces", PAD, y0 + 8, "#e8eaf0", 12, true);
    let y = y0 + HEAD_H + 4;
    // per-foot vertical reaction bars, full scale 1200 N
    const FS = 1200;
    for (const [i, name] of [[0, "L foot"], [1, "R foot"]]) {
      const n = feet ? feet[i] : null;
      label(ctx, name, PAD, y, "#9aa0ae", 11);
      const bx0 = 60, bx1 = W - 64;
      ctx.strokeStyle = "#3a4050";
      ctx.lineWidth = 5;
      ctx.beginPath();
      ctx.moveTo(bx0, y);
      ctx.lineTo(bx1, y);
      ctx.stroke();
      if (n !== null) {
        const t = Math.max(0, Math.min(1, n / FS));
        ctx.strokeStyle = n < 0 ? "#e06c75" : "#7aa2f7";
        ctx.lineWidth = 6;
        ctx.beginPath();
        ctx.moveTo(bx0, y);
        ctx.lineTo(bx0 + t * (bx1 - bx0), y);
        ctx.stroke();
        label(ctx, `${n.toFixed(0)} N`, W - 58, y, "#cfd3dc", 11);
      } else {
        label(ctx, "n/a", W - 58, y, "#6b7180", 11);
      }
      y += 24;
    }
    // effort meter (J of positive work this tick), full scale 40 J
    label(ctx, "effort", PAD, y, "#9aa0ae", 11);
    const bx0 = 60, bx1 = W - 64;
    ctx.strokeStyle = "#3a4050";
    ctx.lineWidth = 5;
    ctx.beginPath();
    ctx.moveTo(bx0, y);
    ctx.lineTo(bx1, y);
    ctx.stroke();
    if (effort !== null) {
      const t = Math.max(0, Math.min(1, effort / 40));
      ctx.strokeStyle = "#c5b3f0";
      ctx.lineWidth = 6;
      ctx.beginPath();
      ctx.moveTo(bx0, y);
      ctx.lineTo(bx0 + t * (bx1 - bx0), y);
      ctx.stroke();
      label(ctx, `${effort.toFixed(1)} J`, W - 58, y, "#cfd3dc", 11);
    } else {
      label(ctx, "n/a", W - 58, y, "#6b7180", 11);
    }
  }

  function draw(panel, q29, om29, tau29, feet, effort) {
    const { ctx, groups, forces, h } = panel;
    ctx.clearRect(0, 0, W, h);
    ctx.fillStyle = "rgba(11,13,17,0.72)";
    ctx.beginPath();
    ctx.roundRect(0, 0, W, h, 14);
    ctx.fill();
    let y = PAD;
    for (const g of groups) {
      const grp = GROUPS[g];
      const col = groupColor(g);
      label(ctx, grp.name, PAD, y + 8, col, 12, true);
      y += HEAD_H;
      for (const j of grp.joints) {
        jointRow(ctx, y + ROW_H / 2, j, q29[j], om29[j], tau29[j], col);
        y += ROW_H;
      }
    }
    if (forces) forcesBlock(ctx, y, feet, effort);
    panel.tex.needsUpdate = true;
  }

  $effect(() => {
    if (pose.on) { // gauges reflect the manually posed joints (ω = τ = 0)
      const z = Array(29).fill(0);
      for (const p of panels) draw(p, pose.q, z, z, null, null);
      return;
    }
    if (!pb.rep) return;
    const k = Math.min(Math.floor(pb.cur), pb.rep.ticks.length - 1);
    const k1 = Math.min(k + 1, pb.rep.ticks.length - 1);
    const u = pb.cur - k;
    const qa = pb.rep.ticks[k].q, qb = pb.rep.ticks[k1].q;
    const q29 = Array.from({ length: 29 },
      (_, j) => lerp(qa[7 + j], qb[7 + j], u));
    const om29 = pb.rep.ticks[k1].omega;
    const tau29 = tauOf(pb.rep.ticks[k1].action, pb.rep.config.gears ?? 3);
    draw(panels[0], q29, om29, tau29, null, null);
    draw(panels[1], q29, om29, tau29, footForces(pb.rep, k1),
      effortAt(pb.rep, k1));
  });

  $effect(() => () => panels.forEach((g) => g.tex.dispose()));

  // camera-space placement: at z = -1 the view half-height is
  // tan(fov / 2) (0.52 for the orbit camera's 55°); anchor the panels to
  // the screen corners, above the floating bottom UI.
  const { size } = useThrelte();
  const HALF_H = $derived(Math.tan((fov / 2) * Math.PI / 180));
  const SCALE_Y = 0.62;
  const MARGIN = 0.015;
  const BOTTOM_UI_PX = 130;
</script>

{#each panels as g, i (i)}
  {@const sc = [SCALE_Y * (W / g.h), SCALE_Y * 1, 1]}
  {@const gx = HALF_H * ($size.width / $size.height) - MARGIN - sc[0] / 2}
  {@const gy = -HALF_H + 2 * HALF_H * (BOTTOM_UI_PX / $size.height) + sc[1] / 2}
  <T.Sprite
    position={[i === 0 ? -gx : gx, gy, -1]}
    scale={sc}
    renderOrder={20}
  >
    <T.SpriteMaterial map={g.tex} transparent depthTest={false} />
  </T.Sprite>
{/each}
