# Parameter-norm census — v38t32, af50n, t64 campaigns (2026-09-11)

Measured on the laptop (CPU torch 2.13) from `~/beat-arena-volume-backup`,
108 checkpoints across 9 runs. `param_norms.py` reproduces `param_norms.csv`.
(`param_norms.py` was removed 2026-09-22; recover with
`git show 91eac64:research/param-norms-2026-09-11/param_norms.py`.)
All norms are L2 over fp32 master weights. "actor" = enc + blocks + trunk +
heads; "critic" = the adapool critic (231k params). `ln` = mean LayerNorm
gain over the 7 LN layers. `qk` = max over blocks of ‖W_q‖₂·‖W_k‖₂ (a bound
on attention-logit scale, not the logits themselves).

Fresh init (seed 0): total 53.7 — enc 16.1, blocks 43.9, trunk 13.1,
heads 7.0, critic 21.7; ln 1.00; qk 1.2.

## Matched checkpoints

| campaign | run | ckpt | upd | samples | acc | total | actor | critic | enc | blocks | trunk | heads | ln | qk |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| — | init | — | 0 | 0 | — | 53.7 | 49.0 | 21.7 | 16.1 | 43.9 | 13.1 | 7.0 | 1.00 | 1.2 |
| v38t32 | v38t32_trunk | upd125 | 125 | 0.13B | 0.771 | 78.2 | 56.3 | 54.2 | 20.1 | 49.0 | 16.5 | 9.7 | 0.88 | 4.8 |
| v38t32 | v38t32_trunk | upd500 | 500 | 0.52B | 0.871 | 130.0 | 78.3 | 103.7 | 29.0 | 65.8 | 25.9 | 17.2 | 0.68 | 12.2 |
| v38t32 | v38t32_trunk | upd1000 | 1000 | 1.05B | 0.909 | 168.7 | 97.6 | 137.7 | 36.5 | 81.3 | 32.1 | 23.4 | 0.55 | 29.8 |
| v38t32 | v38t32_trunk | upd2000 | 2000 | 2.10B | 0.939 | 217.4 | 125.9 | 177.3 | 47.3 | 105.9 | 38.9 | 29.6 | 0.43 | 70.7 |
| v38t32 | v38t32_trunk | upd3000 | 3000 | 3.15B | 0.951 | 252.4 | 148.8 | 203.9 | 55.6 | 126.5 | 44.2 | 33.1 | 0.37 | 106.2 |
| v38t32 | v38t32_trunk | upd4500 | 4500 | 4.72B | 0.958 | 293.7 | 177.1 | 234.3 | 65.6 | 152.7 | 49.0 | 36.4 | 0.32 | 163.9 |
| v38t32 | cut02250_cd200_s1 | leaf end | 2450 | 2.57B | 0.958 | 229.7 | 134.0 | 186.5 | 50.2 | 112.9 | 41.3 | 31.4 | 0.41 | 82.1 |
| v38t32 | cut04500_cd200_s1 | leaf end | 4700 | 4.93B | 0.968 | 295.4 | 178.5 | 235.3 | 66.1 | 153.9 | 49.4 | 36.8 | 0.32 | 164.3 |
| af50n | af50n_trunk | upd125 | 125 | 0.13B | 0.765 | 63.9 | 52.7 | 36.2 | 18.3 | 46.3 | 15.2 | 8.5 | 0.95 | 3.4 |
| af50n | af50n_trunk | upd500 | 500 | 0.52B | 0.868 | 93.3 | 65.3 | 66.6 | 23.1 | 55.7 | 21.1 | 13.5 | 0.82 | 8.2 |
| af50n | af50n_trunk | upd1000 | 1000 | 1.05B | 0.918 | 122.6 | 78.4 | 94.3 | 28.1 | 66.1 | 25.7 | 18.2 | 0.71 | 24.4 |
| af50n | ctrl_trunk | upd125 | 125 | 0.13B | 0.768 | 77.6 | 56.0 | 53.6 | 20.1 | 48.7 | 16.3 | 9.7 | 0.88 | 4.4 |
| af50n | ctrl_trunk | upd500 | 500 | 0.52B | 0.875 | 126.9 | 77.6 | 100.4 | 29.0 | 65.4 | 25.0 | 16.8 | 0.67 | 12.2 |
| af50n | ctrl_trunk | upd1000 | 1000 | 1.05B | 0.912 | 164.7 | 97.8 | 132.5 | 36.8 | 81.9 | 31.6 | 22.5 | 0.53 | 26.0 |
| af50n | v38t32af50_trunk | upd1000 | 1000 | 1.05B | 0.911 | 121.5 | 78.1 | 93.1 | 28.2 | 65.7 | 26.0 | 17.7 | 0.70 | 11.5 |
| t64 | t64_g965 | upd504 | 504 | 1.06B | 0.902 | 126.5 | 78.1 | 99.5 | 28.5 | 66.0 | 25.0 | 17.8 | 0.69 | 17.4 |
| t64 | t64_g98 | upd504 | 504 | 1.06B | 0.885 | 126.0 | 76.0 | 100.5 | 28.4 | 64.4 | 22.4 | 18.2 | 0.71 | 13.2 |
| t64 | t64_g98 | upd1008 | 1008 | 2.11B | 0.907 | 166.4 | 92.8 | 138.1 | 35.6 | 78.0 | 26.2 | 24.0 | 0.56 | 24.6 |
| t64 | t64_g98 | upd1890 | 1890 | 3.96B | 0.923 | 213.4 | 114.6 | 180.0 | 44.1 | 97.4 | 29.5 | 28.7 | 0.42 | 52.2 |
| t64 | t64_g98_leaf_cd200 | leaf end | 2090 | 4.38B | 0.935 | 216.4 | 116.5 | 182.3 | 45.0 | 98.9 | 30.1 | 29.4 | 0.42 | 54.5 |

Recipes: all lr 7e-4 (critic ×3), N = 32,768, minibatch 16,384, 1 epoch.
v38t32 / ctrl: T = 32, no filter → 64 optimizer steps per update.
af50n / v38t32af50: T = 32, `--adv-filter 0.5` → 32 steps per update.
t64: T = 64 (2.1M samples per update), filter 0.5 → 64 steps per update.

## Findings

1. **Norms grow without bound, at a rate set by optimizer steps, not
   samples.** Log-log slope of total norm vs update index is 0.36–0.39 in
   every trunk. For the actor the norm² increment per 1,000 updates in
   v38t32 settles at ~6,150 from update 1,000 on (7,251 → 6,327 → 6,287 →
   6,150 → 6,141): a constant increment of norm² per step is the signature
   of an unbiased random walk of Adam steps of roughly fixed size. The
   critic's increment decays (its early growth is directed), which is why
   the total slope sits below 0.5.
2. **The advantage filter halves the drift for free.** af50n (32 steps per
   update) vs ctrl (64) at update 1,000: total 122.6 vs 164.7, ratio 1.34
   ≈ √2. Same samples, same accuracy or better, half the parameter travel.
   v38t32af50 reproduces af50n to within 1%; ctrl reproduces v38t32 to
   within 2.5% — the census is deterministic enough to compare across pods.
3. **The critic dominates the norm.** 231k params at RMS 0.49 vs the
   actor's 863k at RMS ~0.19 by 4.7B samples; it grows 10.8× from init vs
   the actor's 3.6×. That is the ×3 lr multiplier: Adam's per-element step
   scales with lr, so the critic random-walks 3× faster.
4. **LayerNorm gains shrink steadily: 1.00 → 0.32 over 4.7B samples**,
   while the matrices they feed grow 3.5×. The product g·W is what the
   network computes, so this is a silent reparametrization — but Adam's
   effective step on a scale-invariant weight goes as lr/‖W‖², so the
   "constant-lr" WSD trunk has had its effective lr on the block matrices
   fall roughly (152.7/43.9)² ≈ 12× since init, ~3.5× between 1B and 4.7B.
   The trunk is not at constant lr in function space.
5. **Head weights grow 5.2× (7.0 → 36.4).** Action logits are
   heads(ReLU(trunk)); growing head norm scales the logits directly, which
   lowers entropy independent of the entropy coefficient. This is a
   candidate mechanism for the long-trunk entropy decline (v16/v17
   signature) — a hypothesis, not shown here.
6. **‖W_q‖₂·‖W_k‖₂ grows ~linearly with updates (slope ~1.0–1.2), 1.2 →
   164.** Actual attention logits are also scaled by the LN gains (0.32),
   so this is a bound, not evidence of logit explosion; but it is the
   quantity MuonClip exists for, and it should be logged as the real
   max-logit before any Muon run.
7. **Leaves barely move the norms** (cd200: +1–3 on total) — the 1−√t
   decay does what it should.

## Implications

- Weight decay is a live question, not a hypothetical: nothing bounds the
  walk, and the effective-lr decay it induces on LN-fed matrices is an
  unplanned schedule inside the "stable" WSD phase. A small decoupled
  decay (AdamW, wd ~1e-4 to 1e-3 at lr 7e-4 → time constant 1.4k–14k
  steps) would hold ‖W‖ near a fixed point and make the trunk lr mean
  what it says. That is a recipe change and needs a matched-samples
  probe like everything else.
- Muon without wd would also walk (fixed-norm orthogonal steps), and
  Moonshot needed wd 0.1 for exactly this reason. Any Muon probe should
  log param norms per group and max attention logit, and probably carry
  a small wd from the start.
- Cheap first step: add `pnorm/*` (per-group norms) and `attn_logit_max`
  to `metrics.jsonl` so the dashboard shows this live; one reduction per
  update at the existing sync.
