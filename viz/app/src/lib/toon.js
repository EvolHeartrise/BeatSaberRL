// Cel shading (2026-09-09): the two pieces every solid in the scene shares
// — a banded ramp for MeshToonMaterial, and the inverted-hull outline the
// solids double into. Kept in one module so the bands and the ink stay
// consistent across the body, the sabers and the notes.
import * as THREE from "three";

// ---- the ramp ------------------------------------------------------------
// MeshToonMaterial samples this 1-D texture at (dot(N, L) * 0.5 + 0.5) with
// NEAREST filtering, so N texels = N flat bands and nothing in between.
// Band 0 sits at 0.18 rather than 0: a true-black terminator turns the
// shadow side of the white body into a silhouette against the dark stage
// and throws away the arm tint exactly where the arms are most readable.
export function toonRamp(bands) {
  const tex = new THREE.DataTexture(
    Uint8Array.from(bands, (b) => Math.round(b * 255)),
    bands.length, 1, THREE.RedFormat);
  tex.minFilter = tex.magFilter = THREE.NearestFilter;
  tex.generateMipmaps = false;
  tex.needsUpdate = true;
  return tex;
}

// three bands: shadow / mid / lit. Two reads as a cheap paper cut-out,
// four starts looking like plain smooth shading at this object size.
export const RAMP = toonRamp([0.18, 0.45, 1.0]);

// ---- the ink -------------------------------------------------------------
// Inverted hull: draw the BACK faces of a copy of the mesh with every
// vertex pushed out along its normal, so the copy survives only where it
// pokes past the front-facing original — i.e. as a band around the
// silhouette AND around every self-overlap (an arm crossing the chest).
// The push is in object space, but every geometry here is baked at its real
// size and no mesh carrying an outline is scaled, so one thickness is one
// constant in metres.
//
// The normal comes from the `normal` attribute directly, not from
// `objectNormal`: three only defines objectNormal in the basic-material
// vertex shader under USE_ENVMAP / USE_SKINNING, neither of which is on
// here.
export function outlineMaterial(thickness, color = "#0a0c10") {
  const m = new THREE.MeshBasicMaterial({ color, side: THREE.BackSide });
  m.onBeforeCompile = (shader) => {
    shader.uniforms.uThickness = { value: thickness };
    shader.vertexShader = "uniform float uThickness;\n" + shader.vertexShader
      .replace("#include <begin_vertex>",
        "#include <begin_vertex>\n\ttransformed += normalize(normal) * uThickness;");
  };
  // three keys the program cache on the material's parameters, which are
  // identical across these instances — without this the first outline
  // compiled would hand its thickness to every other one.
  m.customProgramCacheKey = () => "outline:" + thickness;
  return m;
}

// One shared instance per thickness, sized to what it wraps: bones and
// joints are 4-9 cm across, the 0.5 m note cubes take a heavier line, and
// the ~2 cm saber needs a hairline or the ink swallows the blade.
export const OUTLINE_BODY = outlineMaterial(0.006);
export const OUTLINE_NOTE = outlineMaterial(0.011);
export const OUTLINE_SABER = outlineMaterial(0.0025);
