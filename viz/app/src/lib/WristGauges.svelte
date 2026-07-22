<script>
  // Per-hand wrist/joint readouts: a roll dial (needle = blade azimuth,
  // allowed 280-degree arc bright, forbidden zone dark) and a tilt dial
  // (+-95 degrees, right = toward the notes), plus numeric angles and
  // angular velocities. Needles turn orange within 10 degrees of a limit.
  //
  // Drawn as canvas-texture sprites ATTACHED TO THE CAMERA: they stay
  // screen-fixed under orbiting, show up in exported videos, and need no
  // DOM/fonts (offline-safe) — same rationale as the score popups.
  import { T } from "@threlte/core";
  import * as THREE from "three";
  import { pb } from "./state.svelte.js";
  import {
    PHI_LIM, SABER_COLORS, THETA_HI, THETA_LO, saberAt,
  } from "./replay.js";

  const W = 256, H = 310;
  const NEAR = 10 * Math.PI / 180; // limit-proximity warning band
  const deg = (r) => (r * 180 / Math.PI);

  const hands = [0, 1].map((hand) => {
    const cv = document.createElement("canvas");
    cv.width = W;
    cv.height = H;
    const tex = new THREE.CanvasTexture(cv);
    tex.colorSpace = THREE.SRGBColorSpace;
    return { hand, cv, ctx: cv.getContext("2d"), tex };
  });

  function needle(ctx, cx, cy, r, canvasAngle, color) {
    ctx.strokeStyle = color;
    ctx.lineWidth = 5;
    ctx.lineCap = "round";
    ctx.beginPath();
    ctx.moveTo(cx, cy);
    ctx.lineTo(cx + r * Math.cos(canvasAngle), cy + r * Math.sin(canvasAngle));
    ctx.stroke();
    ctx.fillStyle = color;
    ctx.beginPath();
    ctx.arc(cx, cy, 5, 0, 7);
    ctx.fill();
  }

  function label(ctx, text, x, y, color = "#9aa0ae", size = 17, bold = false) {
    ctx.fillStyle = color;
    ctx.font = `${bold ? "bold " : ""}${size}px system-ui, sans-serif`;
    ctx.textAlign = "center";
    ctx.textBaseline = "middle";
    ctx.fillText(text, x, y);
  }

  function draw(g, sab) {
    const { hand, ctx } = g;
    const [lo, hi] = [THETA_LO[hand], THETA_HI[hand]];
    const th = sab[4], ph = sab[5], wt = sab[6], wp = sab[7];
    const col = SABER_COLORS[hand];

    ctx.clearRect(0, 0, W, H);
    ctx.fillStyle = "rgba(11,13,17,0.72)";
    ctx.beginPath();
    ctx.roundRect(0, 0, W, H, 16);
    ctx.fill();
    label(ctx, hand === 0 ? "L wrist" : "R wrist", W / 2, 20, col, 19, true);

    // ---- roll dial: needle = blade azimuth as the player sees it -----------
    // world theta (0 = +x, CCW) -> canvas angle -theta (canvas y is down)
    const cx = W / 2, cy = 116, r = 66;
    ctx.lineWidth = 9;
    ctx.strokeStyle = "#232733"; // forbidden zone: full faint circle first
    ctx.beginPath();
    ctx.arc(cx, cy, r, 0, 2 * Math.PI);
    ctx.stroke();
    ctx.strokeStyle = "#3a4050"; // allowed 280-degree arc
    ctx.beginPath();
    ctx.arc(cx, cy, r, -lo, -hi, true);
    ctx.stroke();
    for (const b of [lo, hi]) { // limit ticks
      ctx.strokeStyle = "#f0a35e";
      ctx.lineWidth = 3;
      ctx.beginPath();
      ctx.moveTo(cx + (r - 8) * Math.cos(-b), cy + (r - 8) * Math.sin(-b));
      ctx.lineTo(cx + (r + 8) * Math.cos(-b), cy + (r + 8) * Math.sin(-b));
      ctx.stroke();
    }
    const rollHot = th - lo < NEAR || hi - th < NEAR;
    needle(ctx, cx, cy, r - 10, -th, rollHot ? "#f0a35e" : col);
    label(ctx, `roll ${deg(th).toFixed(0)}°  ·  ω ${wt.toFixed(1)}`,
      W / 2, 200);

    // ---- tilt dial: up = in-plane, right = toward the incoming notes -------
    const tx = W / 2, ty = 274, tr = 44;
    const cAng = (p) => -Math.PI / 2 + p; // canvas angle for tilt p
    ctx.lineWidth = 9;
    ctx.strokeStyle = "#3a4050";
    ctx.beginPath();
    ctx.arc(tx, ty, tr, cAng(-PHI_LIM), cAng(PHI_LIM));
    ctx.stroke();
    for (const b of [-PHI_LIM, PHI_LIM]) {
      ctx.strokeStyle = "#f0a35e";
      ctx.lineWidth = 3;
      ctx.beginPath();
      ctx.moveTo(tx + (tr - 8) * Math.cos(cAng(b)), ty + (tr - 8) * Math.sin(cAng(b)));
      ctx.lineTo(tx + (tr + 8) * Math.cos(cAng(b)), ty + (tr + 8) * Math.sin(cAng(b)));
      ctx.stroke();
    }
    label(ctx, "notes", tx + tr + 24, ty - 6, "#6b7180", 13);
    label(ctx, "you", tx - tr - 22, ty - 6, "#6b7180", 13);
    const tiltHot = PHI_LIM - Math.abs(ph) < NEAR;
    needle(ctx, tx, ty, tr - 8, cAng(ph), tiltHot ? "#f0a35e" : col);
    label(ctx, `tilt ${deg(ph).toFixed(0)}°  ·  ω ${wp.toFixed(1)}`,
      W / 2, H - 12);

    g.tex.needsUpdate = true;
  }

  $effect(() => {
    if (!pb.rep) return;
    const tickIdx = Math.min(Math.floor(pb.cur), pb.rep.ticks.length - 1);
    const u = pb.cur - tickIdx;
    for (const g of hands) draw(g, saberAt(pb.rep, tickIdx, u, g.hand));
  });

  $effect(() => () => hands.forEach((g) => g.tex.dispose()));

  // camera-space placement: at z = -1 with fov 55 the view half-height is
  // tan(27.5 deg) ~ 0.52; park the panels in the lower corners
  const SCALE = [0.26 * (W / H), 0.26, 1];
</script>

{#each hands as g (g.hand)}
  <T.Sprite
    position={[g.hand === 0 ? -0.56 : 0.56, -0.36, -1]}
    scale={SCALE}
    renderOrder={20}
  >
    <T.SpriteMaterial map={g.tex} transparent depthTest={false} />
  </T.Sprite>
{/each}
