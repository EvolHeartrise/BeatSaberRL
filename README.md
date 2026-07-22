# Beat Arena

Beat Saber-style toy RL environment + transformer policy experiments.
See `SPEC.md` for the full contract (constants, obs layout, file formats).

Env v5 (blade sabers): each hand is a wrist (point mass on the z=0 hit
plane) holding a rigid 1 m blade whose direction is driven by two wrist
DOFs — azimuth (CW/CCW rotation, the swing) and tilt
(supination/pronation, pitch toward the incoming notes). 40 cm note
cubes (real Beat Saber size, 0.5 m lanes, world in meters) fly in and
must be swung through with the matching hand, in the right direction,
fast enough at the contact point (≥1 m/s). Good cuts are graded with the
real Beat Saber 115-point system — 70 pre-swing (100°), 30
follow-through (60°), 15 accuracy-to-center — paid as reward/115.
Per-tick discrete control at 30 Hz: per hand move(9) × rot(3) × tilt(3).

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

then open `viz/replay.html` (3D, perspective view) in a browser and drop a
`python/replays/*.jsonl` file in. Space = play/pause, ←/→ = tick-step,
+/− = speed, drag the scrub bar. White ring around a saber = fast enough
to cut (≥1 m/s); arrows on note cubes = required swing direction; white
outline = inside the cuttable slab; green split = good cut; orange/red
flash = bad cut / miss, labeled with the reason. Pre-metric (v1) replays
are rejected — re-record from a current checkpoint.

Reference points on the default easy chart (rate 1.0, right hand only,
down-swings, 20 s ≈ 19 notes), env v5 scripted bot: easy 100% good cuts
/ 97.6 pts per note; hard (both hands, 8 dirs) 95% / 96.3; dynamic gym
94% / 95.8 — the 1 m blade RAISED the scripted ceiling vs the point
saber (dynamic was 81%). Pre-v5 reference numbers are point-saber and
not comparable.

## Status

- v0 (2026-07-20): env + chart gen + scripted bot + policies + PPO
  harness built (pool encoder + Python sim later deleted; see SPEC).
- Raw reward proved too sparse to learn from (flat at random, entropy
  stuck at uniform) → potential-based shaping added (`shaping_coef`,
  see SPEC) and takeoff needs patience: learning starts ~100–150k steps.
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
- Pure-reward era (2026-07-20, by design): shaping OFF, rewards only
  from sim scoring (V_MIN 1 m/s; bad cut = miss = −0.5).
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
