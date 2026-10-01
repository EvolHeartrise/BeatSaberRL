// Worker entry: bundles the game build (the committed viz/replay.html, as a
// Text module — see `rules` in wrangler.jsonc) and hands every request to
// the auth gate in app.js.
import gameHtml from "../../replay.html";
import { handle } from "./app.js";

export default {
  fetch(request, env) {
    return handle(request, env, gameHtml);
  },
};
