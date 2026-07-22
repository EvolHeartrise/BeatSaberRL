import { defineConfig } from "vite";
import { svelte } from "@sveltejs/vite-plugin-svelte";
import { viteSingleFile } from "vite-plugin-singlefile";

// Single-file build: the whole SPA (Svelte + Threlte + three) inlines into
// dist/index.html so the viewer stays a double-clickable file:// page with
// no server and no toolchain, like the hand-written canvas viewer it
// replaces. The build is copied to ../replay.html (the documented path).
export default defineConfig({
  base: "./",
  plugins: [svelte(), viteSingleFile()],
  build: { chunkSizeWarningLimit: 2048 },
});
