import { defineConfig } from "vite";
import { fileURLToPath } from "node:url";
import { svelte } from "@sveltejs/vite-plugin-svelte";
import tailwindcss from "@tailwindcss/vite";
import { viteSingleFile } from "vite-plugin-singlefile";
import basicSsl from "@vitejs/plugin-basic-ssl";

// Single-file build: the whole SPA (Svelte + Threlte + three) inlines into
// dist/index.html so the viewer stays a double-clickable file:// page with
// no server and no toolchain, like the hand-written canvas viewer it
// replaces. The build is copied to ../replay.html (the documented path).
//
// `--mode xr` (npm run xr): the Flux WebXR game. WebXR only exists in a
// secure context, and a headset reaches the laptop over the LAN, so the dev
// server listens on every interface over HTTPS with a throwaway
// self-signed certificate (accept the warning once on the headset).
//
// `--mode site` (npm run build:site): the replays.flee.systems viewer, into
// dist-site/ — __SITE__ switches off the Flux game + Flux replays, pose mode
// and the file-picker button (src/lib/build.js), and the gated code is
// tree-shaken out of that bundle.
//
// Tailwind serves the control bar's shadcn-svelte components only and is
// scoped in src/app.css (no preflight; utilities from the bar's own files).
// $lib is the alias the generated components import through.
export default defineConfig(({ mode }) => ({
  base: "./",
  define: { __SITE__: JSON.stringify(mode === "site") },
  resolve: { alias: { $lib: fileURLToPath(new URL("./src/lib", import.meta.url)) } },
  plugins: [tailwindcss(), svelte(), viteSingleFile(), ...(mode === "xr" ? [basicSsl()] : [])],
  server: mode === "xr" ? { host: true } : {},
  build: {
    chunkSizeWarningLimit: 2048,
    ...(mode === "site" ? { outDir: "dist-site" } : {}),
  },
}));
