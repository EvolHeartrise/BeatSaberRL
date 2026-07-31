<script>
  // Per-hand wrist/joint readouts: a roll dial (needle = forearm roll,
  // allowed arc bright — 180 degrees since env v7.1 — forbidden zone dark),
  // a bend dial (0 = pointing at the notes, +-90 = blade flat in the hit
  // plane, +-95 = limit), and the env v9 DEV dial (radial/ulnar
  // deviation, +-45 = limit), plus numeric angles and angular
  // velocities. Needles turn orange within 10 degrees of a limit.
  //
  // Drawn as canvas-texture sprites ATTACHED TO THE CAMERA: they stay
  // screen-fixed under orbiting, show up in exported videos, and need no
  // DOM/fonts (offline-safe) — same rationale as the score popups.
  import { T, useThrelte } from "@threlte/core";
  import * as THREE from "three";
  import { pb, pose } from "./state.svelte.js";
  import {
    BEND_LIM, DEV_LIM, ELBOW_HI, PITCH_HI, PITCH_LO, ROLL_HI, ROLL_LO,
    SABER_COLORS, TWIST_HI, TWIST_LO, YAW_HI, YAW_LO, poseSaber, saberAt,
  } from "./replay.js";

  const W = 256, H = 520;
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
    const [lo, hi] = [ROLL_LO[hand], ROLL_HI[hand]];
    // env v9 arm state: q(7) | om(7) | wrist(3)
    const th = sab[4], ph = sab[5], dv = sab[6];
    const wt = sab[11], wp = sab[12], wd = sab[13];
    const col = SABER_COLORS[hand];

    ctx.clearRect(0, 0, W, H);
    ctx.fillStyle = "rgba(11,13,17,0.72)";
    ctx.beginPath();
    ctx.roundRect(0, 0, W, H, 16);
    ctx.fill();
    label(ctx, hand === 0 ? "L arm" : "R arm", W / 2, 20, col, 19, true);

    // ---- roll dial: needle = blade azimuth as the player sees it -----------
    // world theta (0 = +x, CCW) -> canvas angle -theta (canvas y is down)
    const cx = W / 2, cy = 116, r = 66;
    ctx.lineWidth = 9;
    ctx.strokeStyle = "#232733"; // forbidden zone: full faint circle first
    ctx.beginPath();
    ctx.arc(cx, cy, r, 0, 2 * Math.PI);
    ctx.stroke();
    ctx.strokeStyle = "#3a4050"; // allowed roll arc (180 deg since v7.1)
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

    // ---- bend dial: up = at the notes (neutral), +-90 = blade flat in
    // the hit plane, +-95 = the anatomical limit
    const tx = W / 2, ty = 274, tr = 44;
    const cAng = (p) => -Math.PI / 2 + p; // canvas angle for bend p
    ctx.lineWidth = 9;
    ctx.strokeStyle = "#3a4050";
    ctx.beginPath();
    ctx.arc(tx, ty, tr, cAng(-BEND_LIM), cAng(BEND_LIM));
    ctx.stroke();
    for (const b of [-BEND_LIM, BEND_LIM]) {
      ctx.strokeStyle = "#f0a35e";
      ctx.lineWidth = 3;
      ctx.beginPath();
      ctx.moveTo(tx + (tr - 8) * Math.cos(cAng(b)), ty + (tr - 8) * Math.sin(cAng(b)));
      ctx.lineTo(tx + (tr + 8) * Math.cos(cAng(b)), ty + (tr + 8) * Math.sin(cAng(b)));
      ctx.stroke();
    }
    for (const b of [-Math.PI / 2, Math.PI / 2]) { // "blade flat" marks
      ctx.strokeStyle = "#6b7180";
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.moveTo(tx + (tr - 6) * Math.cos(cAng(b)), ty + (tr - 6) * Math.sin(cAng(b)));
      ctx.lineTo(tx + (tr + 6) * Math.cos(cAng(b)), ty + (tr + 6) * Math.sin(cAng(b)));
      ctx.stroke();
    }
    label(ctx, "notes", tx, ty - tr - 14, "#6b7180", 13);
    label(ctx, "flat", tx + tr + 22, ty - 6, "#6b7180", 13);
    label(ctx, "flat", tx - tr - 20, ty - 6, "#6b7180", 13);
    const bendHot = BEND_LIM - Math.abs(ph) < NEAR;
    needle(ctx, tx, ty, tr - 8, cAng(ph), bendHot ? "#f0a35e" : col);
    label(ctx, `bend ${deg(ph).toFixed(0)}°  ·  ω ${wp.toFixed(1)}`,
      W / 2, 344);

    // ---- dev dial (env v9): radial/ulnar deviation — up = neutral,
    // +-45 = the limit; the blade's SECOND bending axis
    const dx = W / 2, dy = 404, dr = 34;
    ctx.lineWidth = 9;
    ctx.strokeStyle = "#3a4050";
    ctx.beginPath();
    ctx.arc(dx, dy, dr, cAng(-DEV_LIM), cAng(DEV_LIM));
    ctx.stroke();
    for (const b of [-DEV_LIM, DEV_LIM]) {
      ctx.strokeStyle = "#f0a35e";
      ctx.lineWidth = 3;
      ctx.beginPath();
      ctx.moveTo(dx + (dr - 8) * Math.cos(cAng(b)), dy + (dr - 8) * Math.sin(cAng(b)));
      ctx.lineTo(dx + (dr + 8) * Math.cos(cAng(b)), dy + (dr + 8) * Math.sin(cAng(b)));
      ctx.stroke();
    }
    const devHot = DEV_LIM - Math.abs(dv) < NEAR;
    needle(ctx, dx, dy, dr - 6, cAng(dv), devHot ? "#f0a35e" : col);
    label(ctx, `dev ${deg(dv).toFixed(0)}°  ·  ω ${wd.toFixed(1)}`,
      W / 2, 456);

    // ---- arm-joint bars (env v6): angle within its anatomical range ------
    const bars = [
      ["pitch", sab[0], PITCH_LO, PITCH_HI],
      ["yaw", sab[1], YAW_LO[hand], YAW_HI[hand]],
      ["twist", sab[2], TWIST_LO[hand], TWIST_HI[hand]],
      ["elbow", sab[3], 0, ELBOW_HI],
    ];
    bars.forEach(([name, v, blo, bhi], i) => {
      const y = 472 + i * 13;
      const x0 = 62, x1 = W - 52;
      ctx.textAlign = "right";
      ctx.fillStyle = "#6b7180";
      ctx.font = "12px system-ui, sans-serif";
      ctx.fillText(name, x0 - 6, y + 4);
      ctx.strokeStyle = "#3a4050";
      ctx.lineWidth = 4;
      ctx.beginPath();
      ctx.moveTo(x0, y);
      ctx.lineTo(x1, y);
      ctx.stroke();
      const t = (v - blo) / (bhi - blo);
      const hot = Math.min(v - blo, bhi - v) < NEAR;
      ctx.strokeStyle = hot ? "#f0a35e" : col;
      ctx.lineWidth = 3;
      ctx.beginPath();
      ctx.moveTo(x0 + t * (x1 - x0), y - 5);
      ctx.lineTo(x0 + t * (x1 - x0), y + 5);
      ctx.stroke();
      ctx.textAlign = "left";
      ctx.fillStyle = "#6b7180";
      ctx.fillText(`${deg(v).toFixed(0)}°`, x1 + 8, y + 4);
    });

    g.tex.needsUpdate = true;
  }

  $effect(() => {
    if (pose.on) { // gauges reflect the manually posed joints (ω = 0)
      for (const g of hands) draw(g, poseSaber(g.hand, pose.q[g.hand]));
      return;
    }
    if (!pb.rep) return;
    const tickIdx = Math.min(Math.floor(pb.cur), pb.rep.ticks.length - 1);
    const u = pb.cur - tickIdx;
    for (const g of hands) draw(g, saberAt(pb.rep, tickIdx, u, g.hand));
  });

  $effect(() => () => hands.forEach((g) => g.tex.dispose()));

  // camera-space placement: at z = -1 with fov 55 (mirrors Scene.svelte)
  // the view half-height is tan(27.5 deg) ~ 0.52 and the half-width is
  // aspect x that. The viewport is now the whole page, so anchor the
  // panels to the actual corners: a margin in from the sides, and far
  // enough up to clear the floating bottom UI (hud/bar/help ~ 130 px).
  const { size } = useThrelte();
  const HALF_H = Math.tan((55 / 2) * Math.PI / 180);
  const SCALE = [0.26 * (W / H), 0.26, 1];
  const MARGIN = 0.015;
  const BOTTOM_UI_PX = 130;
  const gx = $derived(
    HALF_H * ($size.width / $size.height) - MARGIN - SCALE[0] / 2);
  const gy = $derived(
    -HALF_H + 2 * HALF_H * (BOTTOM_UI_PX / $size.height) + SCALE[1] / 2);
</script>

{#each hands as g (g.hand)}
  <T.Sprite
    position={[g.hand === 0 ? -gx : gx, gy, -1]}
    scale={SCALE}
    renderOrder={20}
  >
    <T.SpriteMaterial map={g.tex} transparent depthTest={false} />
  </T.Sprite>
{/each}
