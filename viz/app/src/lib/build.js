// Build flavour. `npm run build:site` (vite build --mode site) is the viewer
// deployed at replays.flee.systems: a watch-only page opened from ?replay=
// links, WITHOUT the Flux WebXR game, Flux replay playback, manual pose mode
// or the file-picker button. Every other build — `npm run build`
// (viz/replay.html), `npm run dev|xr` — and node (tests, where nothing is
// defined) is the full viewer. The flags are build-time constants, so the
// site build drops the gated code from its bundle.
/* global __SITE__ */
export const SITE = typeof __SITE__ !== "undefined" && __SITE__;
export const FLUX = !SITE; // the Flux game + Flux replays
export const POSE = !SITE; // manual pose mode ('p', PosePanel)
export const PICKER = !SITE; // the "drop a replay / click to choose" button
