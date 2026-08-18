# Robo-Saber (Eurographics 2026) — comparison reference

Investigated 2026-08-18 (Opus agent, repo commit 1718a2b). Repo:
https://github.com/namheegordonkim/robo-saber · paper arXiv 2602.18319
· page https://robo-saber.github.io/

## What's released (and not)

- **Generator-only release.** The physics half (PHC tracker, Isaac Gym
  harness, 16 private mocap sequences, fine-tuned policy) is NOT
  released — zero physics code in the tree. Physics-vs-physics
  head-to-head is impossible against the published artifacts.
- **Kinematic generator weights ARE released**: 261 MB pickle bundle
  (all 4 nets) on Google Drive id `1TqJBxNk7LEBh7EqcU11tON5Couxiid1c`
  (verified live). SECURITY: loaded with `weights_only=False` —
  arbitrary code execution; sandbox the first run.
- **NO LICENSE file** (GitHub license: null) → all rights reserved.
  Email authors before publishing numbers derived from their weights.
  Paper itself is CC BY 4.0. BOXRR-23 is HF `cschell/boxrr-23`,
  "cc"-tagged, per-user tarballs fetched on demand (never need the
  5 TB).

## Running their inference

`uv run snakemake generate` → `robo-saber/generate.py` with a CSV of
(User ID, Song Hash, Difficulty). Any BeatSaver hash downloads on
demand — our ranked maps work. Python 3.11 pinned, torch 2.8, CUDA
required (hardcoded), no Isaac Gym. Patch `MAX_OUTPUT_ROWS = 10`
(generate.py:29). Output: NetCDF of 60 Hz 3-point trajectories
(head + 2 sabers, xyz + 6D rot, Unity coords) + TS metrics.
Per step it samples 32 candidates and argmax-selects BY TorchSaber
score — the metric is inside their inference loop.

## TorchSaber scoring (the r=0.856 proxy)

`vendor/torch_saber/torch_saber.py:141-191`:
`0.7 * clamp(pre_angle,100)/100 + 0.3 * clamp(post_angle,60)/60`,
normalized per opportunity. Matches real pre(70/115) + post(30/115);
**NO accuracy-15 component, NO combo multiplier, no bomb/wall penalty
in the score** (those only enter candidate selection, λ=10). More
forgiving than real scoring on precision; linear (not super-linear)
in misses. 60 Hz fixed, swept-segment collision (5 keypoints/saber),
acausal gaussian velocity smoothing, PLAYER_HEIGHT hardcoded 1.5044,
note grid 1.8 x 1.05 m, objects live to beat+0.1 s. Code (not paper
Appendix D) is the operative spec — params were tuned post-paper
(saber 1.0 m not 1.2; good-cut box 1.0x0.8x0.5).

- r=0.856: TS vs real scores on held-out HUMAN replays (validity).
- r=0.789: generated vs ground-truth player scores (impersonation).

## Their results (for our comparison table)

- Kinematic (N_ref=5): ~60th percentile vs humans (protocol: human
  replays RE-SCORED through TS on the 100 most-played maps per
  difficulty; percentile against those).
- **Physics version: ">40% on average on Normal maps, poor on Expert
  and above ... does not keep up with humans on Expert and Expert+"
  (paper §5).** Our bench is ranked (largely Expert/Expert+), where
  their embodied agent is near-floor by their own account.

## Head-to-head plan (agent's recommendation, endorsed)

Highest value per effort: **lift `beaty_common/eval_utils.py:
evaluate_3p_on_map()` as a standalone scorer** — takes (trajectory
tensor, parsed beatmap) → (normalized_score, opportunities, goods,
hits, misses). Feed OUR policy's head+saber trajectories (resample
30→60 Hz, convert to Unity coords + 6D rotations, note the 1.5044
height renorm) and score both systems on the same metric (~1 day).
Caveats to state in any table: (1) they argmax-select against TS
during inference (flatters them on TS); (2) kinematic vs embodied is
apples-to-oranges — separate rows; (3) their Info.dat parser is
v2-schema-only, check our ranked set for v4 info files.

Related: standalone https://github.com/namheegordonkim/TorchSaber
(2024, older API, has a PyVista visualizer + sample .xror replays);
stale unmodified PHC fork on the same account.
