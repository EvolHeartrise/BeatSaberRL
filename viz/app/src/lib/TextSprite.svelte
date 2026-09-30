<script>
  // Billboard text via a canvas texture — self-contained (system font, no
  // network font fetch) and rendered IN the WebGL canvas, so score popups
  // and fail labels live in the 3D scene with everything else.
  import { T } from "@threlte/core";
  import * as THREE from "three";

  let {
    text,
    color = "#e8eaf0",
    size = 0.12, // world height of the text
    position = [0, 0, 0],
    opacity = 1,
    bold = true,
  } = $props();

  const PX = 64; // texture text height in pixels

  const made = $derived.by(() => {
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
  });

  $effect(() => {
    const t = made.tex;
    return () => t.dispose();
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
