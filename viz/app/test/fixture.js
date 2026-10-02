// Replay files for the node tests: read (and inflate) a format-2 replay.
import { readFileSync } from "node:fs";
import { gunzipSync } from "node:zlib";

/** The committed scripted episode (30 Hz, legacy fixed track, contacts,
 * cuts, posts, misses, bombs) as a format-2 replay. */
export const FIXTURE = new URL("./fixtures/v10_scripted_ep00_seed0.rpl.gz",
  import.meta.url);

/** A replay file's bytes, inflated when gzip — parseReplay's input. */
export function readReplayFile(path) {
  const b = readFileSync(path);
  return new Uint8Array(b[0] === 0x1f && b[1] === 0x8b ? gunzipSync(b) : b);
}
