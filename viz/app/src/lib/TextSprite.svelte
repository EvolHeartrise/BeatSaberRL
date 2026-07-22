<script>
  // Billboard text via a canvas texture — self-contained (system font, no
  // network font fetch) and rendered IN the WebGL canvas, so exported
  // videos keep score popups and fail labels.
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

<T.Sprite {position} scale={[size * made.aspect, size, 1]} renderOrder={10}>
  <T.SpriteMaterial
    map={made.tex}
    transparent
    {opacity}
    depthTest={false}
  />
</T.Sprite>
