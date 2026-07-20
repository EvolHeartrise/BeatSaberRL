# Beat Arena

Beat Saber-style toy RL environment + transformer policy experiments.
See `SPEC.md` for the full contract (constants, obs layout, file formats).

Two sabers (left/red, right/blue) on a 4×3 cut plane; notes arrive on a
schedule and must be swung through fast enough, in the right direction,
with the matching hand, inside a ±0.15 s window. Per-tick discrete
acceleration control (9 actions per hand), 30 Hz.

## Quick start

Uses kiting-arena's venv (`../kiting-arena/python/.venv`), no extra deps.

```bash
cd python
VENV=../../kiting-arena/python/.venv

$VENV/bin/python -m pytest tests/          # 15 tests
$VENV/bin/python train_smoke.py            # PPO smoke, easy chart, pool encoder
$VENV/bin/python train_smoke.py --encoder transformer
```

Difficulty dials: `--note-rate 1.5 --colors both --dirs 0 2 4 6 --min-hand-gap 0.4`.

Reference points on the default easy chart (rate 1.0, right hand only,
down-swings, 20 s ≈ 19 notes), unshaped returns: random ≈ −8.8,
scripted bot ≈ +18.5 (96% good cuts).

## Status

- v0 (2026-07-20): env + chart gen + scripted bot + pool/transformer
  policies + PPO harness built; 16 tests green.
- Raw reward proved too sparse to learn from (flat at random, entropy
  stuck at uniform) → potential-based shaping added (`shaping_coef`,
  see SPEC) and takeoff needs patience: learning starts ~100–150k steps.
- 500k-step results, easy chart, seed 0, unshaped eval over 30 eps:
  **pool 93% good cuts, return 17.7** (≈ scripted); **transformer 75%,
  return 12.3** in 2.2× the wall-clock. Single seed — no verdicts.
- Next: v1 two-hand/direction-variety charts, difficulty curriculum,
  replay visualizer, real BeatSaver `.dat` parsing (v3).
