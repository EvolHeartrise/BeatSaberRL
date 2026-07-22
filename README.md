# Beat Arena

Beat Saber-style toy RL environment + transformer policy experiments.
See `SPEC.md` for the full contract (constants, obs layout, file formats).

Env v6 (kinematic upper body): each hand is a chain from a fixed
shoulder anchor — upper arm, elbow (hinge), forearm, a two-DOF wrist
(roll + bend), and a rigid 1 m blade as the forearm's extension. No
force interactions; every joint is an independent integrator with
anatomical limits, and the big joints (shoulder, elbow: Ω ≤ 8 rad/s)
are slower than the wrists (Ω ≤ 20 rad/s) — position with the arm, whip
with the wrist. 40 cm note cubes (real Beat Saber size, 0.5 m lanes,
world in meters, cubes rotated to their cut direction) fly in and must
be swung through with the matching hand, in the right direction, fast
enough at the contact point (≥1 m/s). Good cuts are graded with the
real Beat Saber 115-point system — 70 pre-swing (100°), 30
follow-through (60°), 15 accuracy-to-center — paid as reward/115.
Per-tick discrete control at 30 Hz: 12 ternary joint heads (per hand:
pitch, yaw, twist, elbow, roll, bend).

## Quick start

Uses kiting-arena's venv (`../kiting-arena/python/.venv`), no extra deps.

```bash
cd python
VENV=../../kiting-arena/python/.venv

$VENV/bin/python -m pytest tests/
$VENV/bin/python train_batch.py --num-envs 128 --epochs 1   # the trainer
```

Since 2026-07-20 the Rust crate (`rust/`) is the ONLY sim implementation
and the transformer the only policy — the Python sim, pool encoder, and
serial trainer were deleted (surface-area reduction). Batched training
reaches ~4.5k steps/s (transformer, 8 CPU cores, --epochs 1); the
ceiling is the torch update, not the env (SPEC §Batched rollouts).

The Rust sim core (63x faster env.step; training 1.3-1.6x on 8 serial
envs) builds once with:

```bash
cd rust && VIRTUAL_ENV=$VENV $VENV/bin/maturin develop --release
```

Difficulty dials: `--note-rate 1.5 --colors both --dirs 0 2 4 6 --min-hand-gap 0.4`.

## Curriculum training (real maps, ScoreSaber star ladder)

Alternative to the procedural generator (SPEC §Curriculum training):

```bash
$VENV/bin/python train_batch.py --curriculum --num-envs 128 --epochs 1 \
    --total-steps 50000000 --ckpt-every 50
```

Starts on a random ScoreSaber-ranked ~1.0★ map (fetched automatically
into `maps/curriculum/`). Every `--fc-eval-every` updates the newest map
is played once greedily; on a **full combo** (every note a good cut) a
`fc_<stars>star.pt` checkpoint drops, a new random map 0.2★ harder joins
the mix, and the old maps keep getting half the episodes so they don't
regress. Ladder state lives in `<run-dir>/curriculum.json` — pass it as
`--curriculum-state` to a later run to resume the ladder.

## Watching replays

```bash
$VENV/bin/python record_replay.py --ckpt runs/pure_tf_batch_4m/final.pt --episodes 3
$VENV/bin/python record_replay.py --scripted --note-rate 2.5 --colors both --dirs 0 1 2 3 4 5 6 7
```

then open `viz/replay.html` in a browser and drop a
`python/replays/*.jsonl` file in. Since 2026-07-21 the viewer is a
Svelte + Threlte (three.js) SPA with orbit controls (drag = orbit,
wheel = zoom, right-drag = pan), compiled to that single static file —
source in `viz/app/`, rebuild with `npm install && npm run build` and
copy `dist/index.html` over `replay.html`. Space = play/pause, ←/→ = tick-step,
+/− = speed, drag the scrub bar. White ring around a saber = fast enough
to cut (≥1 m/s); arrows on note cubes = required swing direction; white
outline = inside the cuttable slab; green split = good cut; orange/red
flash = bad cut / miss, labeled with the reason. Pre-metric (v1) replays
are rejected — re-record from a current checkpoint.

Reference points (env v6 scripted bot, 30 eps each): easy (rate 1.0,
right hand only, down-swings) 85.4% good cuts / 75.4 pts per note; hard
(both hands, 8 dirs) 69.5% / 66.0; dynamic gym 61.0% / 59.3. The
realistic body costs the bot dearly vs the v5.3 free-wrist numbers
(98.8 / 82.7 / 84.6) — reach and joint limits are now real constraints,
and pre-v6 references are not comparable.

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
