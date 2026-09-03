import { defineConfig } from "vite";
import { svelte } from "@sveltejs/vite-plugin-svelte";

// Static build (dist/) served by serve.py or any plain http.server; base
// "./" so dist can also be copied INTO the archive root and opened from
// there. In dev, /archive and /replay.html proxy to serve.py on :8765.
export default defineConfig({
  base: "./",
  plugins: [svelte()],
  server: {
    proxy: {
      "/archive": "http://127.0.0.1:8765",
      "/replay.html": "http://127.0.0.1:8765",
    },
  },
});
