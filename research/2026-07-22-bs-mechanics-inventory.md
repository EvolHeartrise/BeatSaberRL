# Beat Saber mechanics inventory — ArcViewer + BeatLeader web-replays (2026-07-22)

Source repos (analyzed at depth-1 clones, 2026-07-22):
- **ArcViewer** (github.com/AllPoland/ArcViewer) — Unity/C# map previewer + BSOR replay renderer. Authoritative for: spawn/jump math, grid geometry, arcs/chains geometry, walls, scoring arithmetic, energy, modifiers, map-format fields.
- **BeatLeader/BeatSaber-Web-Replays** (github.com/BeatLeader/BeatSaber-Web-Replays) — JS/A-Frame replay viewer. Authoritative for: scoring formulas, hitbox dimensions, spawn phases, BSOR replay format.

Caveat both agents confirmed: **neither repo re-simulates cut detection or swing-rating accumulation** — both consume the game-recorded per-cut ratings from BSOR replays. Hitboxes/spawn math ARE modeled (BeatLeader collides them for visualization). Swing-rating internals (100°/60° arc accumulation), the ±60° direction tolerance, min-saber-speed threshold, and the too-soon window exist only in the game binary; the repos corroborate the point values but not the accumulation rules.

---

## A. Values we model but get WRONG (corrections to our sim)

| # | Item | Real value | Ours | Ref |
|---|---|---|---|---|
| A1 | Accuracy denominator | `15 * (1 − clamp(cutDistanceToCenter / **0.3**, 0, 1))`, each score part **rounded to int** and clamped | `/0.28`, continuous | BL `replay-loader.js:689-695`; AV `ScoringEvent.cs` |
| A2 | Lane pitch | **0.6 m** horizontal (lanes at x = −0.9, −0.3, 0.3, 0.9) | 0.5 m | BL `utils.js:2-3,44-46`; AV `ObjectManager.cs:41` |
| A3 | Row heights | **Non-uniform**: at-player row centers y ≈ **0.85 / 1.40 / 1.90 m** (AV vanilla relative heights 0 / 0.55 / 1.05; spawn rows 0.25 / 0.85 / 1.45 — notes rise in flight) | uniform 0.5 m pitch, 3 rows centered on origin | BL `utils.js:53-59`; AV `ObjectManager.cs:48-53` |
| A4 | Good-cut hitbox | **0.8 w × 0.5 h × 1.0 d box, offset +0.25 m toward player** (extends 0.75 m in front of note center); modern dot notes **0.8 tall**; NOT the visual cube | the 0.4 m visual cube, rotated to cut direction | BL `beat.js:645-700` |
| A5 | Bad-cut (wrong color) hitbox | **0.35³** (game; BL viewer deliberately renders 0.4³) — *smaller* than the visual cube | same 0.4 cube as good cuts | BL `beat.js:675-682` (comment at :679) |
| A6 | Hitbox rotation | Viewer collides **axis-aligned** boxes (hitboxes don't track the arrow rotation) | our v5.3 rotates the collision cube by cut direction | BL `beat.js:1104-1160` |
| A7 | Cut plane / player geometry | Note target plane is **0.65 m in front of the head** (AV) / z = −SWORD_OFFSET(0.9) tracking the headset z (BL); floor offset, default player height 1.8 m, height offset `clamp((h−1.8)/2, −0.2, 0.6)` | fixed hit plane z = 0, no player | AV `ObjectManager.cs:35-46`; BL `utils.js:323`, `replay-player.js:91-99` |
| A8 | Miss / despawn geometry | Collidable from z > −2.6 (relative), miss/despawn ~2 m behind player | testable −0.6..1.5, miss at z < −0.6 | BL `beat.js:16,356-377` |
| A9 | Saber sweep | Swept **surface** (triangle: prev tip, cur tip, base → oriented box) per frame | 8 sampled points along blade | BL `saber-controls.js:236-277` |
| A10 | Min saber speed | Not in either repo (game records `speedOK`); BL's *display* stroke heuristic uses 2 m/s tip speed | 1 m/s at contact | BL `saber-controls.js:115-220` — **unverified vs game** |

## B. Object types entirely missing from our sim

- **B1 Bombs**: spawn/move exactly like notes, no direction/color; hitbox **sphere r = 0.18 m**; hitting one = bad-cut event: −0.15 energy, combo break, 0 points. We currently *skip them at parse*. (BL `beat.js:646`, `replay-loader.js`; AV `BombManager.cs:194-221`)
- **B2 Walls / obstacles**: width/height in **0.6 m units**; v2 full wall = 3.0 m tall from floor, crouch wall = 1.8 m tall starting at layer 2; length = `duration_s × NJS` (min 0.06); head-inside-wall = combo break + **energy drain 1.3/s** (AV modeled it; BL only breaks combo). v3 free y/h with version-dependent clamps. (AV `WallManager.cs`, `PlayerPositionManager.cs:348-489`; BL `wall.js`)
- **B3 Arcs (sliders)**: cubic Bézier head→tail; control points = `headPos + dirVec(headAngle)·mu·2.5` (tail mirrored, `tmu`), dot ends → zero offset; mid-anchor mode (m=1/2 CW/CCW) with midpoint offset 2.5, rotation 90°; ends attach to coincident notes (+0.225 m offset along direction). **Scoring: ArcHead ⇒ post-swing granted in full (30); ArcTail ⇒ pre-swing granted in full (70).** (AV `ArcManager.cs:220-328`, `ScoringEvent.cs:107-125`; BL `mapPostprocessor.js:583-667`)
- **B4 Chains (burst sliders)**: head = normal note **capped at 85** (70 pre + 15 acc, post-swing = 0); each link = **flat 20 points** regardless of swing; links = quadratic Bézier (`p1 = head + dirVec(headAngle)·dist/2`), link i at `t = (i/(sc−1))·squish`, angle from curve tangent, cutDirection = Any; link hitboxes 0.8×**0.2**×1.0 good / 0.35×**0.1**×0.35 bad; link energy +0.002 / −0.025 / −0.03. (AV `ChainManager.cs:34-161`; BL `replay-loader.js:666-698`, `beat.js:424-463,689-691`)

## C. Scoring-layer systems missing

- **C1 Combo multiplier**: ×1→×2 (2 hits) →×4 (4 hits) →×8 (8 hits), cap ×8; any bad event resets progress and **halves the multiplier one tier**. Score = Σ multiplier × note score. (BL `MultiplierCounter.js`; AV `ScoreManager.cs:126-217`)
- **C2 Max score / accuracy %**: parallel always-increasing max-multiplier counter over blocks (115/85/20 by type); accuracy = score/maxScore; **FC accuracy** imputes missed notes at prior average. 115-streak stat. (BL `replay-loader.js:462-576`)
- **C3 Energy bar / fail**: start 0.5; good +0.01; bad cut −0.1; miss/bomb −0.15 (chain-link variants 0.002/0.025/0.03); wall 1.3/s; fail at ≤0. Battery (4 lives) / OneLife / NoFail variants. (AV `PlayerPositionManager.cs:348-489`; BL `replay-loader.js:483-527`)
- **C4 Score sanity + grades**: computed good cut > 115 treated as bad cut; grade thresholds SS≥90 … (BL `replay-loader.js:700-706`; AV `ScoreManager.cs:247-274`)

## D. Spawn / movement system missing (we use constant per-note speed)

- **D1 NJS / HJD / JD**: half-jump duration starts at **4 beats**, **halved while jump distance > ~36 m** (BL: `njs·(60/bpm)·hjd > 18`; AV: JD > 35.998), then `+= noteJumpStartBeatOffset`, floor **0.25 beats**; `JD = 2·RT·NJS`; NJS defaults by difficulty 10/10/10/12/16. Reaction time = JD/2/NJS. (BL `beat-generator.js:936-1013`; AV `BeatmapManager.cs:60-64,130-138`)
- **D2 Spawn phases**: far-back "float/move" phase (BL: 200 m/s slide-in, 0.3 s; AV: 300 units over 0.975 s) before the jump proper. (BL `beat-generator.js:4-5`, `beat.js:254-267`; AV `JumpManager.cs:46-62`)
- **D3 Vertical gravity arc**: notes rise from spawn row y (0.25/0.85/1.45) to at-player row y (0.85/1.4/1.9) on a parabola peaking at the cut plane, `g = 2·Δy/HJD²`; stacked same-column notes spawn compacted (`beforeJumpLineLayer`) and spread during the jump. (BL `beat.js:263,353-357,495-497`, `mapPostprocessor.js:447-464`; AV `JumpManager.cs:77-98`)
- **D4 Spawn rotation animation**: arrow rotation eases in over the first ~30% of the jump (AV: Sine.Out over 0.3 of progress; BL: slerp with random ±20° wobble over first half) + **look-at-player** orientation near spawn — cut direction is not fully readable at spawn. (AV `NoteManager.cs:127-141`; BL `beat.js:297-326`)
- **D5 Note flip ("rabbit jump")**: crossed-hand same-beat pairs (red right of blue) swap x over the first quarter of the jump with a y bump (0.45 over / −0.15 under). (BL `mapPostprocessor.js:466-496`, `beat.js:251,293-295`; AV `NoteManager.cs:510-548`)
- **D6 Variable NJS events** (v4 `njsEvents`): per-beat NJS deltas with Beat Saber's easing set (incl. its broken InOutBack/Elastic/Bounce; Sine/Cubic/etc. degrade to step); RT-vs-JD compensation rules. (AV `JumpManager.cs:111-233`, `BeatSaberEasings.cs`; BL `beat-generator.js:283-313`)
- **D7 90/360 rotation events**: v2 types 14/15 (±15/30/45/60 table), v3 `rotationEvents`, v4 `spawnRotations`; BL rotates spawn positions about the y axis through the head. (BL `beat-generator.js:229-237`, `beat.js:277-287`)

## E. Chart/format semantics missing

- **E1 Paired-note angle snapping**: exactly 2 same-color notes on a beat → both snap to the line between them (desired = atan2(−Δx, Δy), 180°-flip if >90° off, rejected if >40° from a non-dot arrow). Changes the *actual* required direction on real maps. (AV `NoteManager.cs:431-507`; BL `mapPostprocessor.js:395-432`)
- **E2 BPM changes**: v2 event type 100, v3 `bpmEvents {b,m}`, v4 sample-accurate `audioData.bpmData` — piecewise beat→time. We read a single Info bpm. (AV `TimeManager.cs:95-202`; BL `mapPostprocessor.js:686-776`)
- **E3 v4 map schema**: split element/data arrays with index indirection (we parse v2/v3 only).
- **E4 Noodle/ME customData**: `coordinates` (precision grid, 0.6 units), per-note `noteJumpMovementSpeed`/`noteJumpStartBeatOffset`, ME ≥1000 encodings (positions, precision directions 1000–1360, wall types), **fake notes** (skipped by viewers), one-saber characteristic, left-hand mirroring transforms. (AV `BeatmapDifficulty.cs:320-349`, `BeatmapDifficultyV2.cs`)

## F. Judgment dimensions the game measures that we don't (BSOR NoteCutInfo)

Per-cut the game records: `speedOK, directionOK, saberTypeOK, **wasCutTooSoon**, saberSpeed, saberDir, timeDeviation, cutDirDeviation, cutPoint, cutNormal, cutDistanceToCenter, cutAngle, beforeCutRating, afterCutRating`. (BL `open-replay-decoder.js:339-373`; AV `Replay.cs:100-117`)
- **F1 "Too soon" bad-cut type** — cutting a note well before its time window is a distinct failure; we have no temporal judgment at all (purely spatial).
- **F2 Time-dependence** = |cutNormal.z| — a tracked skill metric.
- Our slow/angle/wrong_hand map onto speedOK/directionOK/saberTypeOK; tooSoon is the missing fourth.

## G. Modifiers (whole system absent)

Score multipliers: SS slower 0.85× song (×0.70 score), FS 1.2× (×1.08), SF 1.5× (×1.1), GN ghost notes (×1.11), DA disappearing arrows (×1.07), NA no arrows→all dots (×0.7), NB no bombs (×0.9), NO no walls (×0.95); SC small notes (all hitboxes ×0.5), PM pro mode (good hitbox = 0.5³, no offset), SA strict angles, OD old dots (dot hitbox 0.5 tall); NF/BE(4 lives)/IF(one life). (AV `ReplayManager.cs:61-155`; BL `replay-loader.js:592-604`, `beat.js:605-633`)

## H. Not in either repo — must come from the game itself

- Swing-rating **accumulation** internals (how the 100°/60° arcs integrate over frames, plane projection, post-cut continuation window). Both repos consume recorded ratings. Our chain-accumulator + follow-through tracker is a reconstruction; point values (70/30/15, 100°/60°, denominator 0.3) are corroborated.
- Exact direction tolerance (±60°; our cos ≥ 0.5 ≡ 60° matches community consensus), dot-note leniency details.
- Min saber speed threshold and the too-soon time window.
- Saber-vs-saber clash physics.

---

Web cross-checks used (qualitative confirmation of hitbox/scoring facts):
[BSMG Ranking & Scoring Guide](https://bsmg.wiki/ranking-guide.html), [Beat Saber Fandom — Gameplay](https://beatsaber.fandom.com/wiki/Gameplay), [Steam scoring guide](https://steamcommunity.com/sharedfiles/filedetails/?id=2585074407), [official precise-scoring news post](https://store.steampowered.com/news/app/620980/view/4036870438631348235).
