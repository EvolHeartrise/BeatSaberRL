# ScoreSaber watch — vendored source snapshot

- Origin: https://github.com/ScoreSaber/watch ("Web viewer for ScoreSaber
  replays", the Unity/WebGL app behind watch.scoresaber.com; an ArcViewer
  derivative — the same codebase family env v7's geometry research used)
- Commit: `6926860aeb02fd9156a26c08fda738e2a3569962` (2026-07-10,
  "merge scoresaber perf opts"), cloned 2026-07-23
- License: GPL-3.0 (LICENSE included). Fine to archive and use
  internally; anything we DERIVE and distribute must stay GPL-compatible.
- Contents: `Assets/__Scripts` only (all C# app code, ~2 MB) + LICENSE +
  README. Unity assets/models/scenes (~200 MB of binaries) are NOT
  vendored — the FULL working clone (with git history) lives at
  `~/git projects/scoresaber-watch` on the Linux laptop (re-cloned 2026-09-06, same commit; the old Windows-box copy was at `D:\Git\scoresaber-watch`); re-clone or copy from there if
  the upstream ever disappears and a buildable project is needed.

## Why it's here

`python/convert_scoresaber.py` writes ScoreSaber replay `.dat` files by
mirroring this code exactly. Load-bearing files, all under
`__Scripts/`:

- `Loading/Replays/ScoreSaber/ScoreSaberDecoder.cs` — the format:
  28-byte magic ("ScoreSaber Replay 👌🤠\r\n") + LZMA-alone body;
  9-int pointer table (metadata, poses, heights, notes, scores, combo,
  multiplier, energy, fps/extensions); v2 vs v3 field layouts.
- `Loading/Replays/ScoreSaber/Utils/ScoreSaberUtils.cs` — magic bytes +
  primitive readers (little-endian, length-prefixed UTF-8 strings).
- `Loading/Replays/ScoreSaber/Utils/LzmaDecoder.cs` — 7-zip SDK subset;
  unknown-size (-1) LZMA-alone streams decode until the end marker, so
  Python's native `lzma.FORMAT_ALONE` output is compatible as-is.
- `Loading/Replays/Replay.cs` — decoded object model (Frame,
  NoteEvent, NoteCutInfo, eventTypes 1 good / 2 bad / 3 miss / 4 bomb).
- `Previewer/MapControl/Objects/ObjectManager.cs` — replay-to-map note
  matching: ID = scoringType*10000 + lineIndex*1000 + lineLayer*100 +
  colorType*10 + cutDirection, grouped by note time with a 1 ms epsilon
  (`SameTimeEpsilon`); bombs match ignoring the direction digit.
- `Previewer/MapControl/Replays/ScoringEvent.cs` — per-note score is
  RECOMPUTED from cut ratings: round(70·before) + round(30·after) +
  round(15·(1−dist/0.3)); chain links flat 20. `MatchNote` tolerates
  scoring-type mismatches across compatible types.
- `Loading/MapLoader.cs` (~line 1073) — a `.dat` with the magic header
  loads as a replay directly (file open / drag-drop); the map is then
  fetched from BeatSaver via the hash in `levelID`
  (`custom_level_<HASH>`).
