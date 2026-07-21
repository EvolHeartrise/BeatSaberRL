# Beat Arena

Beat Saber-style toy RL environment + transformer policy experiments.
See `SPEC.md` for the full contract (constants, obs layout, file formats).

Two sabers (left/red, right/blue) on a hit plane at z=0; 40 cm note
cubes (real Beat Saber size, 0.5 m lanes, world in meters) fly in at
4 m/s and must be swung through fast enough (≥1 m/s), in the right
direction, with the matching hand, while inside the ±0.6 m slab at the
plane. Per-tick discrete acceleration control (9 actions per hand), 30 Hz.

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
down-swings, 20 s ≈ 19 notes), unshaped returns: random ≈ −8.8,
scripted bot ≈ +18.7 (97% good cuts).

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
- Pure-reward era (2026-07-20, by design): shaping default is OFF —
  rewards come only from sim scoring (V_MIN 1 m/s). First pure run on the
  hard config (tf, 600k): 2.2% good cuts vs random 1.2%, shaped-run 5%,
  scripted 97% — slow upward crawl, no takeoff yet.
- Next: env-side curriculum (easy→hard charts), longer pure runs,
  real BeatSaver `.dat` parsing (v3).
