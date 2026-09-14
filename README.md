# Beat Arena

Beat Saber-style toy RL environment + transformer policy experiments.
See `SPEC.md` for the full contract (constants, obs layout, file formats).

Env v7 (real Beat Saber geometry & judgment on the v6 kinematic upper
body): each hand is a chain from a fixed shoulder — upper arm, elbow,
forearm, two-DOF wrist, 1 m blade; independent integrators with
anatomical limits, slow shoulders (Ω ≤ 8 rad/s) vs snappy wrists
(≤ 20) — position with the arm, whip with the wrist. The world is the
REAL one (sourced from the ArcViewer/BeatLeader code inventory in
`research/`): floor-based coords, 0.6 m lanes at x ±0.9/±0.3, rows at
y 0.85/1.40/1.90, shoulders at (±0.2, 1.47, −0.65) — a 1.8 m player.
Collision uses the real axis-aligned hitboxes (good box 0.8×0.5×1.0
shifted 0.25 m toward the player — early cuts are real; wrong-color
small box 0.35³) with full swept-surface saber tests. Scoring is the
real integer 115 — round(70·pre) + round(30·post) + round(15·acc)
with the real 0.3 m accuracy normalizer — paid as reward/115. New
object types: bombs (r 0.18, hit = −1.5 since env v7.8, pass = free),
chains (head
max 85, links flat 20 on the real Bézier layout), and arcs as
pre/post-swing grants. Per-tick discrete control at 30 Hz: 12 ternary
joint heads (per hand: pitch, yaw, twist, elbow, roll, bend).

Env v8 (2026-07-30, user-directed): joint α and ω caps DOUBLED (arm
160 rad/s² / 16 rad/s, wrist 400 / 40 — effective in-game hand motion,
not biomechanics: tracking prediction lets real players beat physics)
with **gear-table actions** restoring fine control: `--gears 5` adds
±¼-torque options per joint head, `--gears 7` adds ±⅓ and ±⅛; id
0/1/2 stay coast/±full at every K. Effort scales by |fraction|.
Pre-v8 checkpoints and references are a different body.

Env v9 (2026-07-30, user-caught modeling gap): 7th joint per hand —
**wrist radial/ulnar deviation** (right [−15°, +30°], left mirrored;
wrist tier). The v5.2 roll+bend
wrist could POINT anywhere but had out-of-plane blade speed
`roll_rate·sin(bend)` — a pole singularity at straight wrists; blade is
now `Rz(roll)·Ry(dev)·Rx(−bend)·e_z` (dev = 0 reproduces v8 exactly).
Obs 286 → 290, actions 12 → 14 heads, replay v8 — ALL v8 checkpoints,
replays, and records retired. Fused policy kernels (bench
`--fused-forward`, `--fused-update`) are v8-baked and guard out until
ported; the production torch paths are shape-driven and unaffected.

Env v10 (2026-07-31, P3 — SPEC-v10.md): the kinematic upper body is
RETIRED. The body is a full **dynamic humanoid** — floating pelvis +
29 actuated revolute DoF (spine 3, arms 2×7 with the v9 arm chains
verbatim, legs 2×6), de Leva masses on the v9 anchor geometry, feet
welded to the floor, 12-collider self-contact — integrated by a
hand-rolled f64 engine (CRBA/RNEA/Cholesky + weld-block/PGS solver,
`rust/src/dynamics/`) at 8 × 1/240 s per 30 Hz control tick. Actions
are **29 gear heads** (τ = gear_fraction × τ_max); movement is priced by
`--iso-coef` (joint activation: the commanded Σ|gear fraction| per tick)
and `--limit-coef --limit-arms` (limit rest) — the v10 positive-work
"effort" price and its telemetry were retired 2026-09-13, and the
activation itself is the energy telemetry (replay `activation`
[total, spine, arms, legs] per tick, the trainer's `[activation]` line,
the dashboard panel); the reach/miss gates and note look-at follow
the LIVE pose; obs is 340 lanes; replay is JSONL "v10" (full q36 +
root twist + ω29 + contact events; since P6 additive per-tick "feet"
weld impulses; record_rev 3 "activation"). Gravity is real and nothing holds the body
up — posture is the policy's problem now. ALL v9 checkpoints,
replays, and records are retired. The viewer and scripted bot are v10
(P6); the CUDA kernel (P5, `csrc/beat_arena_cuda.cu` — the whole
8-substep dynamics tick as ONE launch, one thread per env, f64 branch
horizon-bounded tight-tolerance vs Rust) is the default training
backend (`--env cuda`) since 2026-08-01. The P4 torch mirror
(`--env torch`, f64-bitwise vs Rust) served as the batched parity
witness through P4/P5 and was REMOVED 2026-08-03 (surface-area
reduction — every physics change had been landing tri-tier); Rust is
the reference and `tests/test_v10_parity_cuda.py` is the live gate.

Env v42 (2026-09-14, user-directed): **hilt grip**. The hand now holds
the saber's 0.16 m handle (the real Saber.fbx "Handle" the viewer draws)
instead of the blade's base: the cutting blade spans 0.16–1.16 m along
the blade axis from the wrist, the handle 0–0.16 m does not cut, and the
reach gate grows by the same 0.16 m (1.91 m at the reference pose). Obs,
actions, checkpoints and the replay format are unchanged (recordings
carry `config.handle_len` so the viewer places the saber correctly for
both eras). Every pre-v42 checkpoint plays a different game on this
geometry — their bench numbers are not comparable; retrain.

## Quick start

Uses kiting-arena's venv (`../kiting-arena/python/.venv`), no extra deps.

```bash
cd python
VENV=../../kiting-arena/python/.venv

$VENV/bin/python -m pytest tests/
# the trainer — v10 (P5): cuda env is the default GPU backend;
# rust env = the CPU path AND the parity reference
$VENV/bin/python train_batch.py --num-envs 1024
$VENV/bin/python train_batch.py --env rust --num-envs 512
```

**Optimizer: hybrid Muon is the default since 2026-09-12** (`--optimizer
muon`): Newton-Schulz Muon on the 26 hidden matrices, Adam on the encoder
input layers, the 29 heads, the critic's final layer, biases and LN gains;
updates RMS-matched so `--lr 7e-4` (critic ×3) carries over. v40 probes
(from scratch, 1.12B samples, same pod): 0.9409 vs Adam's 0.9099 training
acc, +10% return, half the falls, no throughput cost, seed-1 replication
within 0.3 pts. Muon at 2× lr lost; AdamW-style `--weight-decay` (0.03 a
wash, 0.1 hurts) and `--head-clip` (ε 0.2 entropy collapse, ε 0.05 too
tight) both lost. `--resume` of a pre-09-12 checkpoint keeps Adam
automatically (exact resume); `--optimizer adam` forces the old path.

**Architecture: embed 192 / trunk 384 is the default since 2026-09-13**
(2,414,930 params, 1.90M actor + 0.52M critic; was 128/256 = 1.09M).
The v40_w192 probe (from scratch under Muon, lr 5.7e-4 = 7e-4·√(128/192),
matched 1.12B samples) gained +0.45 pt training acc, reached 0.91 at
update 406 vs 513, halved falls, and its cd300 leaf benched 2,353,043 upp
/ 16,420 wpp vs 2,275,937 / 16,034 for the narrow model on the same pod
(+3.4%, the gain growing with map difficulty), at +15% wall. **Use
`--lr 5.7e-4` with the wide arch**; scripts that still say 7e-4 carry the
narrow width's value. Checkpoints carry their own arch, so older 128/256
checkpoints still load for benches, replays and leaves. Watch on long
wide trunks: the fp16 diag `r1` runs ~1.8× the narrow model's and rises;
if it passes 1e-2 or `clipfrac` on the first minibatch leaves 0, move the
head GEMM to fp32.

Since 2026-07-20 the Rust crate (`rust/`) is the reference sim
and the transformer the only policy — the Python sim, pool encoder, and
serial trainer were deleted (surface-area reduction). The tensor-op
port (`--env torch`, torch_env.py + torch_dynamics.py) — removed
2026-07-25 in the kinematic era, REBUILT at v10 P4 as a literal batched
mirror of the dynamics engine (f64 CPU bitwise vs Rust) — is **removed
again, 2026-08-03**. It was Python-dispatch bound (~1 s/vector tick
regardless of width — full-body dynamics is ~100k tensor ops/tick),
which is why P5 moved training to the CUDA kernel; after that it was
consumed only by its own parity tests, so every physics change was
paying a third tier for a witness of a backend nothing ran.
`--env cuda` runs the whole tick as one launch (~7.2k steps/s at
N=1024 on the GTX 1080) and is the default; `--env rust` is the CPU
path and the parity reference.

The Rust sim core (63x faster env.step; training 1.3-1.6x on 8 serial
envs) builds once with:

```bash
cd rust && VIRTUAL_ENV=$VENV $VENV/bin/maturin develop --release
```

Difficulty dials (generator v2, SPEC §Chart generation — colors is
always both since 2026-07-26; the `--colors` flag is gone):
doubles the note stream; per-tick probabilities `--note-prob 0.25
--extra-prob 0.125 --dot-prob 0.1 --bomb-prob 0.125` (the defaults) and
the speed-walk clamp `--note-speed-min 2 --note-speed-max 8`. The v7.9
tiered effort cost and its v10 work-price successor (`--effort-coef`) are
RETIRED (2026-09-13; every recipe since v23 ran it at 0) — the live
movement prices are `--iso-coef` (activation) and `--limit-coef
--limit-arms` (limit rest). **Accuracy is judged against the
infinite cutting plane since 2026-07-29** (user-caught v7.4-class fix:
the swept-surface distance used before conflated timing with lateral
error and capped accuracy at 0.870 = 100/115). **Real maps play at
their real NJS since 2026-07-30** — the speed_scale rescale (NJS×0.3)
is removed. Bench results stamp `scoring_version` (2 = cut-plane acc
at 0.3× speeds, 3 = real speeds) and never merge across rules —
pre-v3 numbers are a different physics regime. `--wide-acc` (30·clamp(1 − d/0.6), unrounded, training
only) is opt-in: it loses at 400M but is the strongest takeoff
accelerator measured for short legs.

## ScoreSaber bench (checkpoint evaluation on all ranked maps)

```bash
$VENV/bin/python bench_scoresaber.py --ckpt runs/<run>/final.pt [--stars 2 4]
```

One greedy (argmax) rollout per ranked ScoreSaber leaderboard — ~6.1k
difficulties over ~2.5k songs; deterministic sim + fixed chart means one
rollout is the exact answer. No replays are recorded. Scoring uses the
REAL combo multiplier (1x/2x/4x/8x, halves on a combo break — bench-only;
the training gym stays multiplier-free) and ScoreSaber's official PP
curve: pp = curve(acc) x maxPP, profile total weighted 0.965^rank.
Computed max scores match the API's `maxScore` field exactly (validated).

Results land in `bench/<run>_<ckpt>.json`: checkpoint name + weight
sha256, per-level entries (score, accuracy, pp, misses, bad cuts,
bombs hit, highest combo, FC flag) and the weighted total PP. Reruns
MERGE: already-evaluated levels are skipped (so star ranges accumulate
across runs), previously-skipped ones are retried, `--force`
re-evaluates, and a different model writing to the same file is an
error. The ranked catalog caches at `maps/scoresaber_catalog.json`
(`--refresh-catalog` to refetch) and chart data at `maps/scoresaber/`
(audio stripped, ~100 KB/song). `--limit N` for smoke runs;
`--num-envs` maps play in parallel — one `VecBeatSim` FFI call per
vector step (vectorized per-env events, 2026-07-24). Since 2026-07-27
the defaults ARE the fast path (`--num-envs 512`, fused forward on;
`--no-fused-forward` to opt out) and the results JSON records which
forward path produced it — fused/graphed near-tie argmax flips mean
per-level results only compare within one path, so cross-path merges
are refused. Since 2026-07-27 the loop is a two-group software pipeline on the fused
path (one group's forward+D2H in flight on the GPU while the CPU runs
the other group's Rust sim; a `query()` after each issue flushes WDDM's
submission batch, without which nothing overlaps) — plus the
last-block-pruned, register-blocked, smem-dieted kernel, pinned action
D2H, and a JIT trust-the-cache startup. All bitwise-exact (the fused
per-sample kernel is batch-size-invariant, verified over the full
catalog): **~128 maps/s warm, full catalog ~48s** on the GTX 1080 box
(first run parses + warms the cache under `maps/chart_cache/`, ~380 MB).
Those figures are the KINEMATIC (v7) era and no longer apply: on the
v10 dynamic body a full 6,097-difficulty bench measures **~18 min
(~343 levels/min) on a 5090** (2026-08-31). The 7-10 min/checkpoint
quoted in the 2026-08-08 migration notes is the v10/v11 env on a 4090
and is also stale — walls and MAX_OBJECTS 1536 landed after it. Rate
is very uneven across the catalog (a high-star stretch runs ~134
levels/min), so do NOT extrapolate a total from a short sample.

## Exporting to ScoreSaber's replay viewer

```bash
$VENV/bin/python convert_scoresaber.py --replay replays/<x>.jsonl \
    --hash <song-hash> --difficulty Hard
```

Converts a JSONL v7 replay of a REAL map into a ScoreSaber replay
`.dat` that loads on watch.scoresaber.com (open/drag the file onto the
page; the viewer fetches the map from BeatSaver by the embedded hash).
Format reverse-engineered from the viewer's own source — vendored
snapshot + annotated format notes in
`research/vendor/scoresaber-watch/` (full clone: `~/git projects/scoresaber-watch`).
Pose frames come from the recorded joint state via the FK mirror —
including the HEAD pose (position + rotation), which the viewer uses
for real: first person is its default camera and its note-arrival
plane sits at `head.z + 0.65`, so the exported head both aims the
camera and reproduces env v15's head-anchored note track. Recordings
older than record_rev 2 have no head rotation and still convert, to
the old fixed forward-facing placeholder. The
final score reproduces the ScoreSaber bench's multiplied score exactly
(per-note totals exact; the pre/acc split of a cut is reconstructed
assuming full pre-swing, so per-component popups can differ on
partial-swing cuts). Procedural-chart replays can't be converted — the
viewer needs a real BeatSaver map to render.

## Training dashboard (laptop, over the run archive)

Since 2026-09-03 the trainer also writes one JSON line per update to
`<run_dir>/metrics.jsonl` (`python/metrics_log.py`: the same host
values as the `update` / `[kl]` / event lines, no extra sync, never
fatal). `python/metrics_backfill.py` rebuilds the same file from every
stdout log in `~/beat-arena-volume-backup` (all eras back to v11) and
writes an `index.json` registry; `viz/dash/` is a Vite + Svelte + uPlot
page over them — Live (auto-refreshing, x = samples | updates | wall),
Compare (matched-sample small multiples, difference-vs-reference panel,
matched-checkpoint table) and a sortable Runs table with bench/replay
links. No backend:

```bash
python3 python/metrics_backfill.py           # metrics.jsonl + index.json
(cd viz/dash && npm install && npm run build)
python3 viz/dash/serve.py                    # http://127.0.0.1:8765/
```

See `viz/dash/README.md` for the field vocabulary and index schema.

## Watching replays

```bash
$VENV/bin/python record_replay.py --ckpt runs/pure_tf_batch_4m/final.pt --episodes 3
$VENV/bin/python record_replay.py --scripted
```

then open `viz/replay.html` in a browser and drop a
`python/replays/*.jsonl` file in. Since 2026-07-21 the viewer is a
Svelte + Threlte (three.js) SPA with orbit controls (drag = orbit,
wheel = zoom, right-drag = pan), compiled to that single static file —
source in `viz/app/`, rebuild with `npm install && npm run build` and
copy `dist/index.html` over `replay.html`. Space = play/pause, ←/→ = tick-step,
+/− = speed, `g` (or the 📊 button) = gauge panels, `h` = hitbox mode (real-hitbox wireframes plus the track
datum: the floating tunnel grid, hit-plane cells and the reach and
miss/head planes — hidden otherwise since 2026-09-06; plus each in-reach
note's own bounding box, hidden otherwise since 2026-09-09), `f` (or the
👁 button) = first-person since 2026-09-11: the camera rides the FK head
frame — origin and rotation, gaze = head local +z. Two sliders appear in
the bar while it is on: vertical FOV (40–110°, default 75°) and
head-rotation smoothing (0–1 s, default 0 = the exact head frame; a
per-tick exponential filter over the recorded head track, so it is still
a pure function of the tick — scrubbing and the frame-by-frame export
match playback, and the export records whichever camera is active); the head sphere is hidden
while it is on, the gauges move to that camera, and the orbit camera
keeps its place for when you toggle back — drag the scrub bar. White
ring around a saber = fast enough to cut (≥1 m/s); arrows = required
swing direction; bombs are black spheres (red flash on hit), chain
heads half-height cubes, links thin slabs, arc grants faint tubes; green
split = good cut; orange/red flash = bad cut / miss. Since 2026-09-09 the
flying hit scores follow the reference viewer's ScoreManager: the bare
score (115 scale, white) pops up in the note's lane at the player's plane
and eases away down the tunnel on a quartic ease-out over 0.7 s — good
hits sinking, bad ones rising — fading in over 0.1 s and out over the
last 0.3 s. Misses read "MISS" and bad cuts/bombs an X, both in the
reference's red, and a bad cut keeps a small reason label under its X.
The sim scores a cut in two installments (pre-swing + accuracy at the cut,
follow-through a few ticks later) but the indicator shows ONE number per
note out of the full 115: it spawns with the cut's points and counts up
when the follow-through lands (`landedPost`). Env v10 (P6): the viewer draws the REAL
articulated skeleton (full-body FK from the replayed q36 — capsules at
the SPEC-v10 collider dims, feet on their weld anchors), flashes
yellow contact markers sized by impulse at self-collision events, and
the `g` gauges are per-joint dials for all 29 joints (grouped spine /
arms / legs: limit arcs, ω, near-limit orange, applied-τ bars derived
from the logged gear actions) plus per-foot vertical ground-reaction
N and the joint-activation meter (total/29 with the spine / arms /
legs split; from the replay's `activation` field, derived from the
logged actions on older recordings).
Pose mode (`p`) drives root + all 29 joints with a copy-as-`set_joints`
q36 export. Non-"v10" replays are rejected — re-record from a current
checkpoint.

Since 2026-09-09 the whole scene is cel shaded (`viz/app/src/lib/toon.js`):
every solid is a `MeshToonMaterial` on one shared three-band ramp and is
drawn twice — once as an inverted-hull ink outline, once banded — over a
key/rim/hemisphere rig with tone mapping off, so the bands stay flat.
The body is white (arms washed ~60% toward their saber hue), which is
what the ink and the bands are tuned against. The old full-width floor
plane is gone since 2026-09-09 in favour of the game's player platform,
lifted from the same reference (Previewer.unity "Player/PlayerPlatform"):
a 3 x 0.05 x 2 m slab whose top face IS the y = 0 floor, with four 2 cm
cyan glow strips inset 1 cm from its rim, fixed at world z -0.65 — which
is where our own root sits. Cut debris falls under the sim's own g
(9.81 -y) instead of drifting in a straight line. The saber trail is cut the
same way: a 0.1 s ribbon swept between the tip path and the blade's base
at the grip — the whole plane the blade passed through — in flat saber
colour, three hard alpha steps and an ink border, sampled at `pb.cur` so
it stays welded to the blade instead of lagging a tick behind it.

## Song audio in the replay viewer

Since 2026-09-09 the viewer plays the real song under a replay, pulled from
BeatSaver at view time (both its hosts send `access-control-allow-origin: *`,
so it works from the `file://` page):

```bash
$VENV/bin/python annotate_replay_audio.py 'replays/*.jsonl'
```

A recording only knows which song it is if it carries the ADDITIVE header
field that writes: `"map": {hash, difficulty, song, audio_offset_s}`.
`make_top10_replays.py` stamps new recordings itself. `audio_offset_s` is the
song time at replay t=0 and is NOT optional — recordings re-base the chart so
the first note lands at 2 s, and the real offsets across the replay library
run from −0.04 s to +19.6 s.

The annotator can be run over recordings made before the field existed
because the header carries the whole chart: a candidate (map, difficulty)
from the ScoreSaber catalog is re-parsed at the recording convention and its
note times compared against the header's, so the map, the difficulty and the
time base are all PROVEN before anything is written. A wrong guess can't
silently desync the audio — the failure mode is "not annotated". 604 of the
639 replays on disk annotate; the rest are synthetic charts with no song.

Viewer side (`viz/app/src/lib/audio.js`): BeatSaver hash lookup → map `.zip`
→ the audio file `Info.dat` names (`.egg` is Ogg Vorbis, so it needs an
explicit MIME type) → a blob an `<audio>` element plays, cached in IndexedDB
by map hash so re-viewing costs no download. The tick clock stays
authoritative and the audio follows it — seeks when it drifts past 80 ms,
`playbackRate` tracks the speed control, and it stays silent while the song
time is still negative. Mute + volume sit in the bottom bar.

`export video` renders FRAME BY FRAME (`viz/app/src/lib/record.js`), not in
real time: it steps Threlte's render loop once per replay tick, stamps each
frame with its own timestamp and encodes VP9 + Opus through WebCodecs into a
WebM (`webm-muxer`). So the file is exactly `ticks + 1` frames at 30 fps
however long a frame took to draw, and the song is copied out of the decoded
audio by array slice — with silence padded wherever the recording runs
outside it — rather than recorded off the speakers.

That replaced a `captureStream` + MediaRecorder pass, which cannot be tuned
into correctness on a machine that can't render 30 fps live: MediaRecorder
timestamps frames by ARRIVAL time, so late frames stretch the video instead
of dropping, and capturing a stalling audio graph gives genuinely choppy
audio. The real-time pass survives as the fallback for browsers without
WebCodecs, and there it still muxes the song by tapping the element through
WebAudio and forking it — via a gain node to the speakers, straight to a
MediaStreamDestination for the recorder. That gain node is where mute/volume
live, since the element's own volume sits upstream of the tap and would
otherwise bake a muted viewer into a silent file.

`node test/export-probe.mjs [replay] [fps]` guards it: it runs a real export
and fails if the file is blank. Structural checks cannot catch that — a
broken exporter produced perfect frame counts, frame rate, duration, pixel
format and audio track with 23k frames of pure black — but bytes-per-frame
separates them cleanly (~0.5 kB blank vs ~13 kB with gameplay in it).

Replays with no map export video-only. The frame rate is a dropdown next to
the button (24 / 30 / 60 / 120): output frames are decoupled from the
replay's 30 ticks/s — frame i renders at tick `i * 30 / fps`, which the
viewer interpolates — so 60 fps genuinely smooths the motion rather than
duplicating frames. The export always runs at 1x regardless of the speed
control. Since 2026-09-11 a `■ stop & save` button sits next to the
progress readout (which shows percent and the exported length as m:ss):
it ends the export at the next frame and still writes a complete file of
everything rendered so far, with the song cut to match — so the first
3 minutes of a 6-minute replay cost 3 minutes of rendering, not 6. The
real-time fallback path stops its recorder the same way.

Cut debris lives half the aftermath window (0.35 s of the 0.7 s; the window
itself is the flying score's lifetime and has to stay) and separates at
2.8 m/s per half, up from 1.4 — the same 0.98 m of travel as before, covered
twice as fast rather than half as far.

The scene also paints its own `#0b0d11` background as of 2026-09-09. Threlte
builds its renderer with `alpha: true` and nothing had set `scene.background`,
so the canvas was transparent and the dark stage was the `#stage` CSS behind
it — invisible in the browser, but `captureStream` records the canvas, so
exports came out black-keyed. Verified: `pix_fmt=yuv420p`, not `yuva420p`.

Since 2026-09-06 the objects are the REAL Beat Saber models as
ScoreSaber's replay viewer renders them — the canonical reference for
every visual size: the 0.5 m beveled note cube with the game's triangle
glyph (in the half away from the swing, apex past centre) and dot, the
chain head (the note minus its swing-side portion, flat face toward the
links) and 0.5 × 0.087 m links with their dots, the spiked bomb (~0.65 m
over the spikes), the saber blade (r 0.018, 1 m) with its 0.16 m handle
behind the grip (the watch headset model was tried and removed the
same day — the skeleton's head stays bare), plus ScoreSaber's default
note colours and glyph colour rules. Meshes are
baked from the watch clone's `Assets/Models` (Unity import scale, prefab
child transforms and scene scale applied; Unity's x negation on FBX
import + our z mirror) into `viz/app/src/lib/models.data.js` by
`viz/app/tools/export.py` (headless Blender 5.0) and `compose.mjs`;
`viz/app/src/lib/models.js` builds the geometries and colours and
`test/models.test.js` pins the reference sizes. The transform ledger is
in `research/vendor/scoresaber-watch/SOURCE.md` ("Models"). GPL-3.0
applies to the embedded meshes: `replay.html` is internal tooling; keep
any redistribution GPL-compatible. `test/closeup-probe.mjs` screenshots
the built viewer at chosen ticks for eyeballing.

Reference points (env v10 P6 scripted bot — gravity-compensated
PD-over-IK on the dynamic body; 5 eps each, gears 5, seeds 0-4,
2026-07-31): easy (right-only, default dials) **36.4%** good cuts /
25.7 pts per note; hard (both hands, default dials) **36.9% / 28.3**,
~4 bombs hit/ep; dynamic (both hands, note_prob 0.35) **33.0% /
24.2**. Random actions land 5-7% / ~5 on the same gyms. The retired
v9 kinematic bot's 59.1/66.2% are NOT comparable — the physics jump
(gravity, free-root reaction coupling, ≥¼-τ_max gear quanta) costs
the scripted referee roughly half its rate, the v6-precedent drop;
13 tuning iterations are documented in `beat_arena_env/scripted.py`
(measured apparent inertias, constrained-stance feedforward,
COM-over-ankles balance, pelvis righting, chest-frame IK, endgame
plan latch).

## Status

- v0 (2026-07-20): env + chart gen + scripted bot + policies + PPO
  harness built (pool encoder + Python sim later deleted; see SPEC).
- Raw reward proved too sparse to learn from (flat at random, entropy
  stuck at uniform) → potential-based shaping added (since REMOVED —
  see the pure-reward era below) and takeoff needs patience: learning
  starts ~100–150k steps.
- World v2 (same day): metric units (0.5 m lanes, 40 cm blocks — real
  Beat Saber size), hit plane at z=0 centered on the origin, notes carry
  (x, y, z, vz) instead of a time feature (z/|vz| IS time-until), spatial
  cut slab, 3D perspective viewer. Obs 188→200; pre-v2 checkpoints
  incompatible.
- 500k-step results, easy chart, seed 0, unshaped eval over 30 eps:
  pre-metric **pool 93% good cuts / 17.7, transformer 75% / 12.3**;
  world-v2 **pool 93% / 17.6** (reparametrization-neutral, as predicted),
  **transformer 91% / 16.9** (big single-seed jump — unexplained,
  needs paired seeds before believing it). Scripted ceiling 97% / 18.7.
- Hard config (both hands + 8 dirs, rate 1.0): scripted 96%, but the
  transformer trained from scratch PLATEAUS at 3% good cuts (600k steps,
  seed 0) — 70% misses, and half of all touches use the wrong hand.
  Three skills stacked at once (hand assignment, direction-conditional
  swings, two-hand traffic) broke the shaping staircase. Curriculum
  warm-start from the easy checkpoint is the proposed next step.
- Pure-reward era (2026-07-20, by design): rewards only from sim
  scoring (V_MIN 1 m/s; bad cut = miss = −0.5). The shaping mechanism
  itself was deleted from the code 2026-07-21.
- **Hard config (both hands, 8 dirs) SOLVED at 14M steps: 98% good cuts,
  beating the scripted bot (96%) — pure rewards, no shaping, no
  curriculum, ~65 min total training at ~4.4k steps/s.** Checkpoint arc:
  2.7% (1M) → 19% (4M) → 41% (8M) → 68% (12M) → 98% (14M).
- Dynamic gym (env v4): continuous angles + per-note speeds; 19M-step
  policy reaches 75-80% vs scripted 81%.
- **v3 milestone: real BeatSaver maps parse (v2+v3 schemas, dot notes,
  precision angles) and sight-read** — e.g. 81% on Spectre [Easy] with
  zero training on real charts. Try:
  `python sightread.py --map maps/<map>.zip --difficulty Easy --ckpt <ckpt> --replay out.jsonl`
- GPU training (2026-07-21, Windows box): `--device cuda` on a GTX 1080
  (torch cu126 wheels — the last line with Pascal kernels) sustains
  ~24.7k steps/s vs 8.5k CPU; rollout is now the bottleneck (77%).
  20M-step dynamic-gym run from scratch in 13.5 min, seed 1: **84.7%
  good cuts greedy — first policy above the 81% scripted ceiling.**
- GPU-resident sim (2026-07-21): `--env torch` runs the whole vec env as
  tensor ops on the GPU (`beat_arena_env/torch_env.py`); Rust remains the
  reference — float64 parity is exact (tests). 89k steps/s at N=8192
  (vs 24.7k at N=128), but so does Rust at wide N: the PPO update is the
  ceiling on this GPU, not the env. Scale `--minibatches` with N or WDDM
  memory oversubscription silently tanks throughput.
- Wide-batch tuning (2026-07-21): update DENSITY is the lever — single-pass
  wide arms all fail (1–8%) at any LR/schedule; epochs climb 73.5/81.6/92.2%
  (e2/e3/e4). **`--env torch --num-envs 4096 --minibatches 16 --lr 1e-3
  --epochs 4` beats the narrow default outright: 92.2% @ 20M in ~10 min vs
  84.7% in 13.5 (N=128).** Adding `--lr-decay cosine` → **95.3% @ 20M in
  ~10 min** — the best one-command from-scratch recipe on this card:
  `--env torch --num-envs 4096 --minibatches 16 --lr 1e-3 --epochs 4
  --lr-decay cosine`.
- **Project record (2026-07-21): 98.2% good cuts / +46.3 on the dynamic gym
  — 30M steps, one command, ~15 min from scratch** (`runs/wide_e4cos_30m`):
  the wide-cosine recipe above with `--total-steps 30000000`.
- Next: close the human-pattern distribution gap; recipe seed replication;
  density knee (e5-6).
- **Env v5 (2026-07-21, user-designed): blade sabers** — wrist rotation
  DOFs (azimuth + supination/pronation tilt), real 1 m saber, swept-blade
  cut geometry replacing the ±0.6 m slab, real-BS 115-point graded
  scoring (pre-swing/follow-through/accuracy) as reward. All v4
  checkpoints/replays retired. Same-day: `--env torch` ported (float64
  parity with Rust exact, 29k steps/s in the wide+e4 regime, 4.3 GB) and
  `viz/replay.html` ported to replay v4 (blade rendering, score popups,
  reach region).
- **First v5 record (2026-07-21, night): 98.3% good / 0 miss /
  score 104.1/115 / +42.67 on the dynamic gym — beats the scripted bot
  (94% / 95.8) on both axes.** 50M steps, wide-cosine recipe, one
  command, 28.5 min from scratch (`runs/v5_wide_e4cos_50m`). First v5
  full combo: seed 7000, 48/48 sampled, 105.2/115
  (`replays/v5_50m_ep00_seed7000.jsonl`). Pure rewards climbed the whole
  stab→swing staircase. SUPERSEDED by v5.1: that policy exploited
  unbounded wrist rolls.
- **Env v5.1 (same night, user-specified): anatomical wrist limits** —
  azimuth right [−100°, 180°] / left [0°, 280°] (280° arcs from the
  palm-down neutral), tilt ±95°; limits zero the angular velocity.
  Scripted bot refits arcs into range: easy 100%/95.9, hard 90%/80.0,
  dynamic 92%/83.1. **Fresh 50M run: 98.7% good / 0 miss / 99.1/115 /
  +40.72 (`runs/v51_wide_e4cos_50m`)** — the limits delayed the surge
  ~3–5M steps and taxed score −5, but engagement held completely;
  another sampled full combo (seed 7004, 46/46).
- **Env v5.2 (user-caught model error): forearm-anchored joints.** The
  v5.x gauges revealed the 98.7% policy holding blades tilted BACKWARD
  (impossible). New model: neutral blade = forearm extension pointing AT
  the notes; roll ψ (forearm CW/CCW, right [−190°, 90°] / left mirrored)
  orients the plane the wrist bends in; bend φ (±95° from forward,
  ±90° = flat in the hit plane) — **the blade can never point back at
  the player** (d_z ≥ cos 95°). Obs/replay semantics changed (replay v5;
  v5.0–v5.1 checkpoints and replays retired). Scripted bot swings flat
  (bend 90° + roll): easy 100%/98.0, hard 85%/77.4, dynamic 84%/76.1 —
  this model is meaningfully harder. 41/41 tests green, torch parity
  exact, viewer + gauges updated (bend dial: 0 = notes, ±90 = flat).
- **PROJECT RECORD (v5.2, 2026-07-21 night): 99.1% good / 0 miss /
  score 107.7/115 / +44.37** — 50M wide-cosine from scratch
  (`runs/v52_wide_e4cos_50m`), beating both prior v5.x runs on every
  axis ON the strictest body model. The correct joint configuration
  out-learned the exploit-era models: a better body beats extra
  freedom. User's registered mechanism ("full swinging conceptually
  easier") vindicated.
- **Env v5.3 (2026-07-22): notes are their ACTUAL rotated boxes.** The
  collision volume is the 0.4 m cube rotated about the approach axis by
  (angle − 270°) — down-swing notes stay axis-aligned, dot notes never
  rotate — replacing the rotation-blind 0.28 m contact sphere; accuracy
  still grades by distance-to-center, floored at 0 on corner clips.
  Torch env has exact float64 parity (unedited parity tests); the
  viewer renders the rotated cubes (and rotated split-halves/fail
  outlines). The v5.2 record ckpt re-evaled at **98.0% good / 0 miss /
  106.8 / +43.72** on the stricter geometry (was 99.1/107.7) —
  center-cutting generalizes.
- **v6 PROJECT RECORD (2026-07-22): 99.5% good / 109.7 score / 0 miss /
  +45.28** — 500M-schedule wide-cosine run (stopped externally at
  443M; eval = `runs/v6_wide_e4cos_500m/ckpt_upd0800.pt`, 419M steps,
  ~5.2 h). Beats v5.2's 99.1/107.7 on the far harder body and settles
  the 50M run's open question: 90.6 was budget starvation, not a d64
  ceiling — the same 107k-param net plays the 12-DOF arm chain
  near-perfectly given steps.
- **First v6 result (2026-07-22): 90.6% good / 91.7 score / 1.3% miss /
  +35.74 on the dynamic gym** — 50M wide-cosine from scratch
  (`runs/v6_wide_e4cos_50m`, 37 min, 22.3k steps/s), 30-ep greedy eval.
  Pure rewards learned arm-positioning + wrist-whipping through the
  3^12 action space and beat the 61.2% scripted ceiling by 29 points;
  onset was late (return crossed 0 at ~34M vs v5.2's ~23M) and the run
  ended mid-surge under the cosine tail, so 90.6 is likely not this
  body's ceiling. Residual errors are bad cuts (8.1%), not misses —
  engagement solved; swing-credit under time pressure is the frontier.
- **Env v6 (2026-07-22, user-designed): kinematic upper body.** Wrists
  stop free-floating: fixed shoulders at (±0.2, 0.45, −0.35), upper arm
  0.35 m, forearm 0.30 m, 4-DOF arm (shoulder pitch/yaw/twist + elbow)
  + the v5.2 wrist pair per hand, all with anatomical limits and tiered
  dynamics (shoulder/elbow 80 rad/s² / 8 rad/s; wrists 200 / 20). The
  9-way translation head is gone — action = MultiDiscrete([3]×12); obs
  140 → 166 (joint state + FK wrist/blade); contact velocity is the
  contact point's finite-difference displacement (no Jacobians
  anywhere); replay v6 logs joint state + FK wrist. Closed-form FK/IK
  in `beat_arena_env/kinematics.py` powers the rewritten scripted bot
  (two-link IK with a forearm-forward "back" swivel, re-aimed blade,
  room-ranked swing senses over both wrist representations). Torch env
  ported with exact float64 parity; viewer draws the full body (torso,
  arm segments, joints) with arm-joint bars added to the gauges, FK
  mirror validated against sim-logged wrists. All v5.x checkpoints,
  replays, and reference numbers retired.
- **Env v7 (2026-07-22, research-driven): real Beat Saber geometry &
  judgment.** Built from the ArcViewer + BeatLeader mechanics inventory
  (`research/2026-07-22-bs-mechanics-inventory.md`). Floor-based world,
  0.6 m lanes, real non-uniform row heights (0.85/1.40/1.90), body at
  real 1.8 m-player placement (shoulders ±0.2, 1.47, −0.65); real
  axis-aligned hitboxes replace v5.3's rotated cubes (good 0.8×0.5×1.0
  offset 0.25 m playerward, dots 0.8 tall; wrong-color 0.35³; bomb
  r 0.18); saber collision = full swept-surface (triangle-AABB SAT);
  integer 115 scoring with the real 0.3 m accuracy normalizer; misses
  register ~2 m behind the head (Z_MISS 2.65). New objects end-to-end
  (sim, obs 166 → 286 with 16 slots + kind/arc flags, generator dials
  `--bomb-rate --chain-prob --arc-prob`, parser, replay v7, viewer):
  bombs, chains (head 85 / links 20, real Bézier layout), arcs as
  swing grants. Parser adds v4 schema, BPM changes, and the
  paired-note angle-snap rule; walls/NJS-jump-math/spawn animations/
  combo/energy/modifiers noted in the research doc, deliberately not
  modeled yet. All v6 checkpoints, replays, and references retired.
- **Env v7.1 (2026-07-22, user-specified): tightened wrist limits.**
  Found with the viewer's new manual pose mode: the v5-era wrist roll
  range (280° arcs) was compensation for the then-missing arm joints
  and let training cheat with wrist-only swings. Right roll
  [−190°, 90°] → **[−180°, 0°]**, twist ±90° → **[−70°, 90°]**; left
  mirrored. With bend +90° a wrist roll now covers only a 180° blade
  fan — 100°+ pre-swing arcs must come from the arm. Landed before any
  v7 training run (no checkpoints invalidated); torch parity held with
  zero test edits; scripted bot drops to 83.2 / 72.0 / 64.1 (easy
  gate ≥80% still holds).
- **First v7.1 run (2026-07-22): 68.5% good / 64.3 / 3.3% miss /
  +19.10** — 50M wide-cosine (`runs/v71_wide_e4cos_50m`, 54 min,
  15.4k steps/s), 30-ep greedy. Head-to-head #4: BOTH predictions
  missed HIGH (Claude score 76±10 and 4 more claims all wrong; user
  82±4) — first joint over-prediction after five under-prediction
  rounds. Autopsy: forgiving hitboxes make CONTACT free but convert
  clumsy swings into −0.5 bad cuts (403 bad vs 47 miss at eval);
  return crossed 0 only at ~40M and the run froze mid-surge —
  budget starvation again, v6-50M déjà vu.
- **Env v7.2 (2026-07-22, user-specified): shoulder pitch capped at
  +65°** (was ±90; straight-down unchanged) — overhead arm poses gone.
  Suite green, torch parity zero-edit again; bot 87.3 / 72.0 / 63.5.
  The full-pre-swing test became a combined pitch+bend swing (same
  rotation plane, chain sums the joints) — neither wrist roll (v7.1
  fan) nor pitch alone (155°) can bank 100° now.
- **Env v7.3 (2026-07-22, user-specified): miss plane at the HEAD
  (Z_MISS 2.65 → 0.65)** — a DELIBERATE deviation from real-game
  despawn rules, for credit assignment: the miss penalty lands ~5
  ticks after the play plane instead of ~20 (γλ trace over those 15
  ticks diluted the responsible actions' advantage to ~0.4x), and a
  passed note is certainly lost rather than ambiguously rescuable.
  Suite green, parity zero-edit. Bot: easy 87.3 / hard 69.9 /
  dynamic 67.8 (dynamic UP from 63.5 — its stray late contacts
  behind the head were mostly bad cuts, now silent misses cost less).
- **First v7.3 run (2026-07-22): 97.8% good / 96.6 / 0.35% miss /
  +39.48** — 50M wide-cosine (`runs/v73_wide_e4cos_50m`), onset ~21M
  (vs v7.1's ~40M): the user's miss-plane credit fix (+ pitch cap,
  confounded) transformed learning; best 50M in project history on the
  most restricted body. Claude's prediction 0-for-5 UNDER after going
  0-for-5 OVER in round #4 — ledger 17-for-48.
- **Env v7.4 (2026-07-22, user-caught): hitboxes rotate with the cut
  direction** — v7.0 shipped them axis-aligned, misreading a
  BeatLeader-viewer simplification (THREE.Box3); the game's colliders
  sit on the rotated note transform. Good + small boxes now rotate by
  (angle − 270°); links rotate with their tangent; dots stay square;
  research doc A6 corrected. Suite 60 green, parity zero-edit (8th).
  The v7.3 record ckpt re-evals BETTER on the honest geometry:
  **98.6% / 97.8 / +40.16** — center-cutting generalizes, v5.3 pattern.
  Bot: 87.3 / 74.6 / 68.5.
- **Env v7.5 (2026-07-22, user-directed): continuous generated
  positions with an outer margin** — the procedural generator samples
  x, y ~ U over the allowed cols/rows span padded 0.2 m per side
  (x ±1.1, y 0.65-2.1; user-specified margin so the old edge cells —
  measure-zero under uniform sampling — don't atrophy); chains go
  0.6-1.2 m along the head's direction; real maps keep their cells;
  sim/obs unchanged (explicit positions since v7.0). **The
  cell-trained v7.3 record ckpt evals 98.9% / 98.8 / +40.78 on the
  unpadded continuous gym — zero generalization gap** (spatial
  control, not cell memorization) — and 97.6 / 96.7 with the margin
  (the padded extremes are genuinely new territory to train). Bot on
  the padded gym: 88.6 / 81.3 / 68.0.
- **Env v7.6 (2026-07-22, user-directed after a formula deep-dive):
  look-at-player note orientation.** Plain notes blend their full 3D
  frame toward facing the head (anchor (0, 1.70, −0.65), vertical 80%
  damped, weight λ = 1 − z/8 clamped — full billboard at the plane;
  ArcViewer/BeatLeader formulas in the research addendum). Applied to
  BOTH hitbox orientation (the good box's 0.25 m shift rides local z,
  extending toward the head at the edges) and the DIRECTION CHECK,
  which now runs in the note's local frame (forward-heavy swings at
  edge notes lose direction credit — real-game behavior). Chain
  heads/links keep pure z-roll; freeze-after-pass and the
  spawn-settle info timeline deliberately skipped. λ = 0 reproduces
  v7.4 exactly. Record ckpt on v7.6: **98.3 / 97.3 / +40.02** — still
  rising. Bot: 82.9 / 77.7 / 68.9.
- **Env v7.7 (2026-07-22, user-directed): bombs in the default gym**
  (`bomb_rate` default 0.25/s everywhere — ChartConfig, env wrappers,
  CLI flags; ~5 per 20 s chart, ~1:10 vs dynamic notes). The
  bomb-naive v7.3 record ckpt does NOT absorb this one: it slices
  ~2/3 of bombs (61 hits / 30 eps → 92.7 / 89.5 / +34.44) — the first
  v7.x change requiring retraining, i.e. the reason bombs belong in
  the training distribution. Suite 62 green.
- **Generator v2 (2026-07-24, user-designed, hard replace): parity-flow
  tick-lattice chart generation.** The rate-scheduled uniform stream
  (note_rate/dirs/continuous_dirs/min_hand_gap/chain_prob/arc_prob) is
  gone; notes now spawn on a jittered ~0.15 s lattice from three
  independent paths (left, right, bombs) with 180°-flip parity angles
  (+ N(0, 45°)), a shared ±0.4 m/s speed random walk clamped to
  2-8 m/s, swing-line EXTRA notes (1/8), and a 1/10 dot roll. Chains
  and arcs come only from real maps now. Same day (user-directed): the
  lattice is SPEED-PACED — the tick interval scales by
  note_speed_max/walk_speed (top speed keeps the 0.15 s base; the
  tick-to-tick z-gap is speed-invariant ~1.2 m), so slow sections are
  sparse and fast sections dense like real maps (measured ~1.4 vs
  ~3.2 local notes/s; ~2.3 notes/s + ~0.6 bombs/s overall both-hands).
  Scripted bot 59.1 / 66.2 (right / both) — new baseline, old bot
  references not comparable. Suite 83 green (f64 + f32 parity
  unchanged); viewer config summary updated + rebuilt. Fresh training
  run pending.
