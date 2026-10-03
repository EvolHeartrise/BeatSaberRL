<script module>
  import * as THREE from "three";

  const PX = 64; // texture text height in pixels

  // One texture per distinct (text, colour, weight), shared by every sprite
  // showing it. Each score popup used to draw its own canvas and upload its
  // own texture when it appeared — and again each time its number counted
  // up — though a replay only ever shows a few dozen distinct strings
  // ("115", "MISS", "X", a reason). Sprites hold references; a texture
  // nobody shows waits in `idle` (oldest first) for the next popup that
  // needs it, and only the IDLE_MAX most recent are kept.
  const cache = new Map(); // key -> { key, tex, aspect, refs }
  const idle = new Map();  // key -> entry, unused ones, oldest first
  const IDLE_MAX = 64;

  function draw(text, color, bold) {
    const cv = document.createElement("canvas");
    const ctx = cv.getContext("2d");
    const font = `${bold ? "bold " : ""}${PX}px system-ui, sans-serif`;
    ctx.font = font;
    const w = Math.ceil(ctx.measureText(text).width) + 16;
    cv.width = w;
    cv.height = PX + 16;
    ctx.font = font;
    ctx.textAlign = "center";
    ctx.textBaseline = "middle";
    ctx.lineWidth = 8;
    ctx.strokeStyle = "#0b0d11";
    ctx.strokeText(text, w / 2, cv.height / 2);
    ctx.fillStyle = color;
    ctx.fillText(text, w / 2, cv.height / 2);
    const tex = new THREE.CanvasTexture(cv);
    tex.colorSpace = THREE.SRGBColorSpace;
    return { tex, aspect: w / cv.height };
  }

  /** The shared texture for this text (drawn on first use). A new entry
   * starts idle: it is only held once a sprite's effect takes it. */
  function textTexture(text, color, bold) {
    const key = `${bold ? 1 : 0}${color}|${text}`;
    let e = cache.get(key);
    if (!e) {
      e = { key, refs: 0, ...draw(text, color, bold) };
      cache.set(key, e);
      idle.set(key, e);
    }
    return e;
  }
  function hold(e) {
    e.refs++;
    idle.delete(e.key);
  }
  function release(e) {
    if (--e.refs > 0) return;
    if (cache.get(e.key) !== e) { e.tex.dispose(); return; } // evicted meanwhile
    idle.delete(e.key); // re-insert: now the most recent
    idle.set(e.key, e);
    for (const [k, old] of idle) {
      if (idle.size <= IDLE_MAX) break;
      idle.delete(k);
      cache.delete(k);
      old.tex.dispose();
    }
  }
</script>

<script>
  // Billboard text via a canvas texture — self-contained (system font, no
  // network font fetch) and rendered IN the WebGL canvas, so score popups
  // and fail labels live in the 3D scene with everything else.
  import { T } from "@threlte/core";

  let {
    text,
    color = "#e8eaf0",
    size = 0.12, // world height of the text
    position = [0, 0, 0],
    opacity = 1,
    bold = true,
  } = $props();

  const made = $derived(textTexture(text, color, bold));
  $effect(() => {
    const e = made;
    hold(e);
    return () => release(e);
  });
</script>

<!-- Depth-tested (2026-09-09), so a label is occluded by whatever is in
     front of it instead of floating over the whole scene — the reference
     viewer's TMP shader behaves the same way (ZTest LEqual, ZWrite Off).
     depthWrite stays off: the quad is mostly transparent, and writing depth
     would let those empty pixels hide the geometry behind them. -->
<T.Sprite {position} scale={[size * made.aspect, size, 1]}>
  <T.SpriteMaterial map={made.tex} transparent {opacity} depthWrite={false} />
</T.Sprite>
