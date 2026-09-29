<script module>
  // Teko Medium, the watch HUD's font, embedded (tekofont.data.js) and
  // registered once per page. The canvases redraw when it arrives.
  import teko from "./tekofont.data.js";
  const FAMILY = "BSHudTeko";
  let fontLoad = null;
  function loadFont() {
    if (!fontLoad) {
      const bin = Uint8Array.from(atob(teko), (c) => c.charCodeAt(0));
      const face = new FontFace(FAMILY, bin.buffer);
      fontLoad = face.load().then((f) => { document.fonts.add(f); });
    }
    return fontLoad;
  }
</script>

<script>
  // The in-game HUD in WORLD space, rebuilt from ScoreSaber's replay viewer
  // (github.com/ScoreSaber/watch @6926860, Previewer.unity "Preview/HUD/
  // Main" and ScoreManager.cs). It is a world-space canvas at localPosition
  // (0, 1.8, 5), scale 0.01 (1 canvas unit = 1 cm), 900 x 350 units. The
  // watch world is this env's world: floor y = 0, the "Notes" parent lifts
  // row 0 to y 0.85, notes are hit at z = 0, and "Player" sits at
  // z = -0.65 (= PLANE_Z0). So the reference's positions carry over
  // unchanged, and world -> scene is (x, y, -z) as everywhere else here.
  // Like the reference's it stays fixed in the room: it does not ride the
  // head-anchored note track.
  //
  //   left column  u in [-450, -300]: Combo block (centre v 115, 150 x 115:
  //     "COMBO" + count, FC bars top and bottom while nothing has broken),
  //     Score block (centre v -50, 150 x 190: score, accuracy, rank letter)
  //   right column u in [300, 450]: multiplier ring (centre v 115, 150 x
  //     150: ComboMeter ring + "x" multiplier), Misses (v -75 .. 15)
  //   EnergyBar: 275 x 6 under the track, top edge at v -245 (world y
  //     -0.65), 90 units BEHIND the canvas (world z 5.9), with the fail
  //     (X) and energy (bolt) icons at its ends
  // The numbers come from lib/hud.js, a pure function of the tick. The
  // text and the ring are drawn into canvas textures IN the WebGL scene,
  // so they show in both cameras and in the exported video. The ring
  // matches ComboMeter.png: outer radius 75 units, 9.375 thick.
  // Skipped: the PP line (only for ScoreSaber-ranked leaderboards, off
  // without one) and "FC : xx.xx%" (off by default in the watch's settings).
  import { T } from "@threlte/core";
  import * as THREE from "three";
  import { pb, play, pose } from "./state.svelte.js";
  import { isFlux } from "./replay.js";
  import { formatScore, hudAt, pctString } from "./hud.js";

  // ---- reference constants (Previewer.unity) --------------------------------
  const UNIT = 0.01;        // Main localScale: metres per canvas unit
  const CY = 1.8, CZ = 5;   // Main anchoredPosition.y / localPosition.z
  const PX = 4;             // texture pixels per canvas unit
  const WHITE = "#ffffff";
  const GRAY = "rgb(153,153,153)";          // (0.6, 0.6, 0.6)
  const RING_BG = "rgba(94,94,94,0.749)";   // (0.368, 0.368, 0.368, 0.749)
  const ICON = "#666666";                   // (0.4, 0.4, 0.4)
  // Teko-Medium SDF face info (point size 50): cap line 32, ascent 37.5,
  // descent -12.5, superscript offset 37.5 at size 0.5; TextMeshPro italic
  // = a 0.35 shear (italicStyle 35), bold = dilate + 7/100 em spacing
  const CAP = 0.64, ASC = 0.75, DESC = -0.25;
  const SUP_RISE = 0.75, SUP_SIZE = 0.5;
  const SHEAR = 0.35, BOLD_SPACING = 0.07, BOLD_STROKE = 0.07;

  // world position of a canvas point (u, v) in units; z behind the canvas
  const at = (u, v, dz = 0) => [u * UNIT, CY + v * UNIT, -(CZ + dz * UNIT)];

  // ---- visibility: standard replays only -----------------------------------
  const shown = $derived(!!pb.rep && !isFlux(pb.rep) && !pose.on && !play.on
    && pb.rep.ticks.length > 0);

  let fontReady = $state(false);
  $effect(() => {
    loadFont().then(() => (fontReady = true), () => (fontReady = true));
  });

  // ---- the numbers at this tick (the DOM HUD's floor(cur) convention) ------
  const h = $derived(shown
    ? hudAt(pb.rep, Math.min(Math.floor(pb.cur), pb.rep.ticks.length - 1))
    : null);
  // strings, so a canvas only redraws when what it shows changes
  const leftKey = $derived(h ? [h.combo, formatScore(h.score),
    pctString(h.pct), h.grade, h.fullCombo].join("|") : "");
  const rightKey = $derived(h ? [h.mult, h.fill, h.misses].join("|") : "");
  const energy = $derived(h ? h.energy : 0);

  // ---- canvas panels ---------------------------------------------------------
  // A panel covers the canvas rect u in [u0, u1], v in [v0, v1].
  function panel(u0, u1, v0, v1) {
    const cv = document.createElement("canvas");
    cv.width = Math.round((u1 - u0) * PX);
    cv.height = Math.round((v1 - v0) * PX);
    const tex = new THREE.CanvasTexture(cv);
    tex.colorSpace = THREE.SRGBColorSpace;
    tex.anisotropy = 4;
    const geo = new THREE.PlaneGeometry((u1 - u0) * UNIT, (v1 - v0) * UNIT);
    return { cv, ctx: cv.getContext("2d"), tex, geo, u0, v1,
      pos: at((u0 + u1) / 2, (v0 + v1) / 2) };
  }
  const L = panel(-450, -300, -145, 172.5);
  const R = panel(300, 450, -75, 190);

  // canvas px of a canvas-unit point in a panel
  const px = (p, u) => (u - p.u0) * PX;
  const py = (p, v) => (p.v1 - v) * PX;

  /** One TextMeshPro line, centred in the rect [u0, u1] x [v0, v1]. runs:
   * [{ t, sup }] (a <sup> run is half size, raised 0.75 em). valign:
   * "middle" centres the ascent..descent box (TMP Middle), "capline"
   * centres the cap height (TMP Capline). */
  function text(p, runs, u0, u1, v0, v1, o) {
    const { size, color = WHITE, italic = false, bold = false,
      spacing = 0, valign = "capline" } = o;
    const c = p.ctx;
    const em = size * PX;
    const mid = (py(p, v0) + py(p, v1)) / 2;
    const base = valign === "middle" ? mid + ((ASC + DESC) / 2) * em
      : mid + (CAP / 2) * em;
    const track = (spacing + (bold ? BOLD_SPACING : 0)) * em;
    const fontOf = (s) => `${s}px "${FAMILY}", sans-serif`;
    const parts = runs.map((r) => {
      const s = r.sup ? em * SUP_SIZE : em;
      c.font = fontOf(s);
      c.letterSpacing = `${track}px`;
      return { ...r, s, w: c.measureText(r.t).width };
    });
    const total = parts.reduce((a, r) => a + r.w, 0) - track;
    let x = (px(p, u0) + px(p, u1)) / 2 - total / 2;
    for (const r of parts) {
      c.save();
      c.font = fontOf(r.s);
      c.letterSpacing = `${track}px`;
      c.fillStyle = color;
      c.translate(x, base - (r.sup ? SUP_RISE * em : 0));
      if (italic) c.transform(1, 0, -SHEAR, 1, 0, 0);
      if (bold) {
        c.strokeStyle = color;
        c.lineJoin = "round";
        c.lineWidth = BOLD_STROKE * r.s;
        c.strokeText(r.t, 0, 0);
      }
      c.fillText(r.t, 0, 0);
      c.restore();
      x += r.w;
    }
  }

  function rect(p, u0, u1, v0, v1, color) {
    p.ctx.fillStyle = color;
    p.ctx.fillRect(px(p, u0), py(p, v1), (u1 - u0) * PX, (v1 - v0) * PX);
  }

  function drawLeft(hh) {
    const c = L.ctx;
    c.clearRect(0, 0, L.cv.width, L.cv.height);
    // Combo (centre v 115, 150 x 115); FCBars: 8-unit lines on its edges
    if (hh.fullCombo) {
      rect(L, -450, -300, 164.5, 172.5, WHITE);
      rect(L, -450, -300, 57.5, 65.5, WHITE);
    }
    // Text child: the block inset 15 top and bottom (v 72.5 .. 157.5)
    text(L, [{ t: "COMBO" }], -450, -300, 120.1, 157.5,
      { size: 50, italic: true, spacing: -0.03, valign: "middle" });
    text(L, [{ t: String(hh.combo) }], -450, -300, 72.5, 122.5, { size: 70 });
    // Score (centre v -50, 150 x 190)
    text(L, [{ t: formatScore(hh.score) }], -450, -300, 8, 45, { size: 50 });
    text(L, [{ t: pctString(hh.pct) }], -450, -300, -29, 8,
      { size: 40, color: GRAY });
    text(L, [{ t: hh.grade }], -450, -300, -108, -29,
      { size: 100, color: GRAY, spacing: -0.05 });
    L.tex.needsUpdate = true;
  }

  function drawRight(hh) {
    const c = R.ctx;
    c.clearRect(0, 0, R.cv.width, R.cv.height);
    // ComboMultiplier (centre v 115, 150 x 150): the ComboMeter ring (a
    // 512 px sprite, 32 px thick) as a background, and the same ring as
    // a Radial360 fill from the top, clockwise, fillAmount = the bar
    const cx = px(R, 375), cy = py(R, 115);
    const thick = 150 * (32 / 512) * PX;
    const rad = 75 * PX - thick / 2;
    c.lineWidth = thick;
    c.lineCap = "butt";
    c.strokeStyle = RING_BG;
    c.beginPath(); c.arc(cx, cy, rad, 0, 2 * Math.PI); c.stroke();
    if (hh.fill > 0) {
      c.strokeStyle = WHITE;
      c.beginPath();
      c.arc(cx, cy, rad, -Math.PI / 2, -Math.PI / 2 + 2 * Math.PI * hh.fill);
      c.stroke();
    }
    // "<sup>x</sup>8", 100, bold italic
    text(R, [{ t: "x", sup: true }, { t: String(hh.mult) }], 300, 450, 40, 190,
      { size: 100, bold: true, italic: true });
    // Misses (bottom-anchored at v -75, 150 x 90)
    text(R, [{ t: "Misses" }], 300, 450, -22.4, 15,
      { size: 40, valign: "middle" });
    text(R, [{ t: String(hh.misses) }], 300, 450, -75, -37.6,
      { size: 60, valign: "middle" });
    R.tex.needsUpdate = true;
  }

  $effect(() => {
    void fontReady;
    if (leftKey) drawLeft(h);
  });
  $effect(() => {
    void fontReady;
    if (rightKey) drawRight(h);
  });

  // ---- the energy bar ------------------------------------------------------
  // EnergyBar: anchored bottom-centre, pivot top, anchoredPosition y -70
  // (top edge v -245), 275 x 6, local z 90. Background (0.078 gray, 0.6
  // alpha) full width; Fill white, left-anchored, width 275 * energy.
  const EB_W = 275, EB_H = 6, EB_V = -248, EB_Z = 90;
  const ebGeo = new THREE.PlaneGeometry(EB_W * UNIT, EB_H * UNIT);
  const ebPos = at(0, EB_V, EB_Z);
  // icons: 20 x 20, FailIcon (X.png) 5 units left of the bar, EnergyIcon
  // (Energy.png, a bolt) 4 units right of it; 0.4 gray
  const iconGeo = new THREE.PlaneGeometry(20 * UNIT, 20 * UNIT);
  function icon(draw) {
    const cv = document.createElement("canvas");
    cv.width = cv.height = 128;
    const c = cv.getContext("2d");
    c.scale(128, 128); // draw in the unit square
    c.fillStyle = c.strokeStyle = ICON;
    draw(c);
    const tex = new THREE.CanvasTexture(cv);
    tex.colorSpace = THREE.SRGBColorSpace;
    return tex;
  }
  // X.png: two rounded strokes corner to corner, inset 1/16
  const failTex = icon((c) => {
    c.lineWidth = 0.2;
    c.lineCap = "round";
    c.beginPath();
    c.moveTo(0.16, 0.16); c.lineTo(0.84, 0.84);
    c.moveTo(0.84, 0.16); c.lineTo(0.16, 0.84);
    c.stroke();
  });
  // Energy.png: the bolt outline traced from the sprite on a 32 grid
  const energyTex = icon((c) => {
    const P = [[18.7, 0.5], [19, 13], [26, 13], [14, 31.5], [13, 19], [6, 19]];
    c.beginPath();
    P.forEach(([x, y], i) => (i ? c.lineTo(x / 32, y / 32) : c.moveTo(x / 32, y / 32)));
    c.closePath();
    c.fill();
  });
  const failPos = at(-EB_W / 2 - 5 - 10, EB_V, EB_Z);
  const energyPos = at(EB_W / 2 + 4 + 10, EB_V, EB_Z);

  $effect(() => () => {
    for (const t of [L.tex, R.tex, failTex, energyTex]) t.dispose();
    for (const g of [L.geo, R.geo, ebGeo, iconGeo]) g.dispose();
  });
</script>

{#if shown}
  <!-- UI / TMP shaders: Cull Off, ZTest LEqual, ZWrite Off -->
  {#each [L, R] as p, i (i)}
    <T.Mesh geometry={p.geo} position={p.pos}>
      <T.MeshBasicMaterial map={p.tex} transparent depthWrite={false}
        side={THREE.DoubleSide} toneMapped={false} />
    </T.Mesh>
  {/each}
  <T.Mesh geometry={ebGeo} position={ebPos} renderOrder={1}>
    <T.MeshBasicMaterial color="#141414" transparent opacity={0.6}
      depthWrite={false} side={THREE.DoubleSide} toneMapped={false} />
  </T.Mesh>
  {#if energy > 0}
    <T.Mesh geometry={ebGeo} renderOrder={2}
      position={[ebPos[0] - (EB_W * UNIT * (1 - energy)) / 2, ebPos[1], ebPos[2]]}
      scale={[energy, 1, 1]}>
      <T.MeshBasicMaterial color={WHITE} transparent depthWrite={false}
        side={THREE.DoubleSide} toneMapped={false} />
    </T.Mesh>
  {/if}
  <T.Mesh geometry={iconGeo} position={failPos}>
    <T.MeshBasicMaterial map={failTex} transparent depthWrite={false}
      side={THREE.DoubleSide} toneMapped={false} />
  </T.Mesh>
  <T.Mesh geometry={iconGeo} position={energyPos}>
    <T.MeshBasicMaterial map={energyTex} transparent depthWrite={false}
      side={THREE.DoubleSide} toneMapped={false} />
  </T.Mesh>
{/if}
