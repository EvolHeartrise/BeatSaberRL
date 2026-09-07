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

## Models (2026-09-06) — embedded in our replay viewer

`viz/app/src/lib/models.data.js` carries these meshes from the full
clone's `Assets/Models` (not vendored here — see the clone path above),
converted with headless Blender 5.0.1 (`viz/app/tools/export.py`, which
dumps the raw mesh coordinates so Blender's FBX unit scale on the object
is ignored exactly as Unity's `useFileScale: 0` ignores it) and composed
by `viz/app/tools/compose.mjs`. Every visual size in our viewer follows
the watch scene, which is the canonical reference. Transform ledger
(Unity metres; then scene-local = Unity with x negated on import and z
mirrored, i.e. a 180° rotation about y, so winding is preserved):

| part | source | import scale | prefab / scene transform | result |
|---|---|---|---|---|
| note | `Objects/Cube.fbx` "Cube" | 1 | Note.prefab root, identity | ±0.25 m beveled cube |
| arrow | `Objects/Note.fbx` "Arrow" | 0.25 | Note.prefab "Arrow": scale 2, rot z +90°, z −0.238 | triangle 0.309 wide, y ∈ [0.055, 0.152], z ∈ [0.238, 0.253] |
| dot | `Objects/Note.fbx` "Dot" | 0.25 | Note.prefab "Dot": scale (0.35, 0.35, 1), z −0.215 | disc ø 0.175, z ∈ [0.215, 0.255] |
| chainHead | `Objects/ChainHead.fbx` | 1 | via NoteHandler.SetMesh | y ∈ [−0.0435, 0.25], x/z ±0.25 |
| chainLink | `Objects/ChainLink.fbx` | 1 | ChainLink.prefab root | 0.5 × 0.087 × 0.5 |
| linkDot | `Objects/Note.fbx` "Dot" | 0.25 | ChainLink.prefab "Dot": scale (0.075, 0.075, 0.5), rot z +90°, z −0.235 | ø 0.0375 |
| bomb | `Objects/Bomb.obj` | 1 | Bomb.prefab root | axis extent 0.32, spikes to r 0.345 |
| blade | `Player/Saber.fbx` "Blade" | 0.5 | Saber.prefab "Blade": z −0.08 | r 0.018, z ∈ [0, 1.0] |
| handle | `Player/Saber.fbx` "Handle" | 0.5 | Saber.prefab "Handle": z −0.08 | z ∈ [−0.16, 0] |
| headset | `Player/Headset.fbx` | 1 | Previewer.unity "Headset": scale 0.35 under Player | 0.291 wide, visor +0.094, straps −0.272 |

Note.prefab's cube is `Cube.fbx`; `Note.fbx`'s "Bloq" object is unused
by the prefab (only its Arrow/Dot children are). Direction convention:
`ObjectManager.DirectionAngles` = {up 180, down 0, left −90, right 90,
…}, rotation applied as `AngleAxis(angle, forward)` — identical to our
note frame's `angle − 270°` roll, so the baked glyphs need no extra
roll. Colours: `ColorManager` defaults left (200, 20, 20) / right
(40, 142, 210); `NoteManager.GetMaterialColors` with the scene's
noteEmission 0.15, arrowSaturation 0.05, arrowBrightness 0.75,
arrowGlowSaturation 0.5, arrowEmission 1.5; Bomb.mat base 0.1415 gray;
Headset.mat base 0.53 gray; Wall.mat (1, 0.184, 0.184) α 0.5.
