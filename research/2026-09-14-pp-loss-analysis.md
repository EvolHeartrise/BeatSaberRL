# Where the remaining pp is lost — data-driven failure analysis (2026-09-14)

Reference model: **v40_w192_cd300** (embed 192 / trunk 384, Muon, cd300 leaf off the
1,068-update trunk; bench 2,353,043 upp / 16,419.9 wpp / mean level accuracy 0.9690,
6,097 ranked levels, 6,476,219 scoreable notes). Cross-checked against v40_muon_cd300,
v40_t64b_cd150, v40_muon_cd200 and v41_sc005_cd200. The per-level feature
table lives in `research/2026-09-14-pp-loss-analysis/` (scripts: see §8). No project code was
modified.

Standing constraints honoured throughout: no change to WHEN or HOW MUCH reward is
paid, no posture/effort costs, T = 32 floor, the policy learns on its own.

## 0. Headline

1. **The 15-point cut-accuracy component is the loss.** Of the 211.1M multiplied
   score points the bench leaves on the table (3.57% of 5,914M), **161.8M = 76.6%
   is the accuracy component** on good cuts (mean 11.87 / 15; the cut plane
   averages ~6.3 cm from the note centre). To first order it is worth **1.13M upp** —
   every other bucket together is under 120k. Falls / early terminations are
   **2,900 notes on 7 levels ≈ 2k upp (0.1%)** — the 2026-09-13 wrap-up's reading
   ("the residual bench loss is falls/terminations on specific maps, not swing
   quality") was true of the three worst-accuracy replays and false of the
   aggregate.
2. **Combo breaks are the second bucket (28.7M pts, 13.6% of the deficit, ~93k upp
   counterfactual)** — and the largest sub-bucket is one the bench never counts:
   **wall head-entries** (est. 2,560 breaks on 314 levels, ~41k upp; TWERKMASTER,
   Drop That Low, BRACE FOR IMPACT, Oyasumi ... play at 0.61–0.83 with zero misses).
   Miss/bad-cut breaks ~36k upp, bomb breaks ~16k upp, the missed notes themselves 9k.
3. **Follow-through** (post-swing) is 11.5M pts / 5.4% / ~24k upp; pre-swing 1.4%.
4. Conversion that every proposal below is priced in: **+0.25 acc-points per good
   cut = +122k upp (+688 wpp); +0.5 = +268k; +1.0 = +667k** (curve convexity, see
   §4). For scale, cd200 → cd300 moved the component +0.20 (11.56 → 11.76) and
   narrow → wide +0.11 — the two best results of the week were both accuracy-
   component moves.
5. Training exposure is uniform over notes; the bench's marginal pp is not. Hard
   content (8★+) is **55% of training notes but 25% of the marginal pp**; 4–8★ maps
   are 44% of the notes and 70% of the marginal value (§5). Hard content is *over*-
   sampled relative to upp, not under-sampled — but that is the upp lens, not the
   "superhuman everywhere" lens.

## 1. Exact score-deficit decomposition (whole bench)

Method (`bench_decomp.py`): per level, max_score − score is split exactly into
(a) notes never played (n_notes − good − bad − miss; the bench's n_notes matches the
catalog's scoreable count, so this is the termination bucket), (b) missed + bad-cut
notes at the level's max points per note, (c) combo loss on the good cuts
(8·U − ramp − score, U = unmultiplied good-cut points; the FC ramp is exact from
max_score), (d) cut-quality loss on good cuts split pre/acc/post by the level's
per-kind maxima. (a)+(b)+(c)+(d) = deficit to the point. Combo loss is attributed
across break causes; the bench does not log wall entries, but the combo walk breaks
on them (lib.rs code 8), so breaks beyond the observed miss+bad+bomb count are
estimated from the loss itself (one isolated break at 8× costs 32 × max-pts/note).
pp columns: "counterfactual" = re-run the official curve with that category zeroed
on each level; "first-order" = local slope × points. Counterfactuals do not add up
(the curve is convex — fixing one thing raises the value of fixing the next).

| category | score pts lost | share | pp if zeroed | pp first-order |
|---|---|---|---|---|
| unplayed (terminations) | 2.7M | 1.3% | 2k | 0k |
| missed + bad-cut notes | 4.6M | 2.2% | 9k | 9k |
| combo loss — miss/bad breaks | 15.2M | 7.2% | 36k | 30k |
| combo loss — bomb breaks | 4.1M | 1.9% | 16k | 12k |
| combo loss — wall breaks (est.) | 9.4M | 4.5% | 41k | 21k |
| cut quality: pre-swing (70) | 3.0M | 1.4% | 8k | 8k |
| **cut quality: accuracy (15)** | **161.8M** | **76.6%** | **4,657k** | **1,129k** |
| cut quality: follow-through (30) | 11.5M | 5.4% | 24k | 22k |
| all | 211.1M | 100% | 7,234k (headroom to 100%) | — |

Counts: 6,468,257 good / 3,256 bad / 1,806 miss / 1,421 bombs / 2,900 unplayed;
3,980 FCs; per good cut pre 69.94 / acc 11.87 / post 29.78.

Same decomposition on the other leaves (share of deficit): muon_cd300 acc 72.7%,
unplayed 5.0% (12,528 notes — two more terminations), combo 13.4%; t64b_cd150 acc
80.6%, unplayed 0.6%, combo 11.1%. The ordering is identical on every checkpoint.

## 2. Where is it lost — by map attributes (v40_w192_cd300)

### 2a. By stars

| stars | levels | notes | mean acc | pp | headroom | acc-pts / good cut | pp if acc-component perfect | pp if wall breaks zeroed |
|---|---|---|---|---|---|---|---|---|
| 2–4 | 357 | 102k | 0.9769 | 90k | 204k | 13.03 | 178k | 3k |
| 4–6 | 2,458 | 1,554k | 0.9763 | 790k | 1,926k | 12.82 | 1,617k | 25k |
| 6–8 | 1,193 | 1,267k | 0.9744 | 498k | 1,382k | 12.39 | 1,167k | 8k |
| 8–10 | 1,113 | 1,639k | 0.9659 | 514k | 1,741k | 11.78 | 1,109k | 3k |
| 10–12 | 794 | 1,544k | 0.9486 | 377k | 1,572k | 10.91 | 532k | 2k |
| 12–14 | 165 | 356k | 0.9257 | 80k | 391k | 10.24 | 52k | 0k |
| 14+ | 6 | 13k | 0.8965 | 3k | 16k | 9.62 | 1k | 0k |

Two readings. (i) The accuracy component degrades smoothly with stars: 87% of
15 on 2★ maps, 64% on 14★ — but **even the easiest maps lose 13% of it**, so it is
a broad precision floor plus a difficulty slope, not a hard-map-only problem.
(ii) Because the curve is convex and easy maps already sit at 0.976, the pp value
of one accuracy point lives in the 4–10★ band (§4).

### 2b. By NPS / peak density / NJS / duration / difficulty label

| NPS bin | levels | mean acc | pp | acc-pts/cut deficit driver |
|---|---|---|---|---|
| 0–3 | 964 | 0.974 | 270k | q_acc 514k |
| 3–5 | 1,777 | 0.976 | 604k | q_acc 1,249k |
| 5–7 | 1,500 | 0.971 | 615k | q_acc 1,330k |
| 7–10 | 1,415 | 0.962 | 658k | q_acc 1,262k |
| 10+ | 441 | 0.946 | 206k | q_acc 303k |

| peak NPS (densest 5 s) | levels | mean acc | | NJS (m/s) | levels | mean acc |
|---|---|---|---|---|---|---|
| < 8 | 2,556 | 0.9757 | | < 14 | 1,008 | 0.973 |
| 8–12 | 1,658 | 0.9715 | | 14–18 | 2,270 | 0.976 |
| 12–14 | 626 | 0.9654 | | 18–20 | 1,359 | 0.968 |
| 14+ | 1,246 | 0.9542 | | 20–22 | 1,043 | 0.959 |
| | | | | 22+ | 417 | 0.953 |

Duration: flat 0.971–0.977 up to 240 s, 0.965/0.961 for 240–400 s (those are the
harder maps; log-duration has no independent effect in the fit). Difficulty label:
Easy 0.978 / Normal 0.977 / Hard 0.975 / Expert 0.969 / ExpertPlus 0.958 — entirely
explained by stars.

### 2c. Linear fit of level accuracy on chart features (n = 6,058, R² = 0.38)

Standardised coefficients (Δacc in points per 1 sd of the feature):

| feature | mean ± sd | coef | | feature | mean ± sd | coef |
|---|---|---|---|---|---|---|
| stars | 6.98 ± 2.51 | **−1.00** | | frac_top-row | 0.23 ± 0.11 | +0.31 |
| peak NPS (5 s) | 9.95 ± 5.20 | **−0.53** | | frac_mid-row | 0.25 ± 0.08 | +0.24 |
| NJS | 16.9 ± 3.4 | **+0.53** (at fixed stars) | | frac_crossover | 0.020 ± 0.030 | −0.22 |
| bomb fraction | 0.07 ± 0.21 | **−0.49** | | frac_double | 0.31 ± 0.16 | −0.08 |
| walls / min | 5.6 ± 11.4 | **−0.45** | | mean NPS | 5.75 ± 2.69 | −0.01 |

Univariate: corr(acc, stars) −0.48, (acc, NPS) −0.36, (acc, NJS) −0.26. Once stars
are in, NJS flips positive (faster notes at equal stars are *easier* for this
policy — consistent with the replay finding that its cuts land at the front of the
hit box, where a fast note spends less time drifting), peak density is what hurts,
and bombs and walls are the two independent penalties.

Worst residuals (acc below the fit by 12–30 pt, all fully played, all with 0–1
misses): Drop That Low ×4 (Uninstaller; −30/−27/−22/−15), TWERKMASTER 2000 (−28),
BRACE FOR FRICKING IMPACT ×3 (Alice), Oyasumi ×2 (Alice), Walk on Water, Last
Kingdom, edamame, 39 Music!, Golden Hour, Crazy Shuffle, Witch Doctor — **wall/duck
maps**; and MARENOL / Fracture Ray (Jabob, Bitz) with 33–70 bombs hit. Mapper
residuals: Uninstaller −2.4 pt, Alice −1.3 pt over 52 / 81 levels; every other
mapper with ≥ 8 levels is within ±0.6.

Largest counterfactual-loss levels (Σ over categories) are simply the 10–12★
Expert maps at 0.93–0.975 — Bad Apple!! (2,190 pp of headroom), Rocket Cider, Voices
From Heaven, Preserved Valkyria, EmbryO ... — 99% of each is the accuracy component.

### 2d. Wall-break estimate

314 levels carry an estimated ≥ 1 wall break; 2,560 breaks total ≈ 41k upp. The
estimate is stable across checkpoints (TWERKMASTER 71 / 76 / 73 on w192 / muon /
t64b) and matches the 2026-09-14 duck census (72% of duckable walls hit, 8–40 real
ducks over 1,085 duckable). Top levels: TWERKMASTER 2000 (430 walls, 314 pp),
BRACE FOR IMPACT Expert (245), Dark Flight Dreamer Easy (238), SswwiinnddlleerR!!
(233), Crazy Shuffle (221), Oyasumi (221), Drop That Low Expert (215).

## 3. Cut-level anatomy from the replays

`replay_cuts.py` over the logged cut events (per note: pre / acc / post points,
reason, tick; chart geometry for row / column / colour / angle / double / stack /
crossover / local density). Top-pp sets are 10–13★ Expert maps.

### 3a. Which component, on the top-pp plays

| set | good cuts | pre | acc | post | loss share pre / acc / post | mean earliness |
|---|---|---|---|---|---|---|
| w192_cd300 top-3 (etude, rock-it, appassionata) | 3,052 | 69.94 | 11.68 | 29.64 | 2% / **89%** / 10% | −3.01 ticks |
| muon_cd300 top-3 | 2,823 | 69.91 | 11.75 | 29.65 | 2% / 88% / 10% | −2.88 |
| t64b_cd150 top-3 + TWERKMASTER | 6,072 | 69.94 | 11.48 | 29.44 | 2% / 85% / 13% | −3.05 |
| w192_cd300 bottom-3 (terminated plays) | 1,844 | 69.55 | 9.10 | 27.64 | 5% / 68% / 27% | −2.23 |

Acc-component histogram (w192 top-3): 15 pts 11.7%, 14 21.5%, 13 17.7%, 12 12.5%,
11 9.7%, 10 8.1%, ≤ 8 13.3%, 0 (≥ 30 cm off or clipped) 1.2%. **38% of the lost
accuracy points come from the 13% of cuts at ≤ 8 (≥ 14 cm off centre); 28% from the
63% of cuts already within 6 cm.** So it is both: a fat tail of badly placed cuts and
a broad 1–3-point shortfall on the rest.

### 3b. By chart cell (w192 top-3; acc pts per good cut, max 15; ≥ 60 cuts)

| split | cell | cuts | acc | pts lost / cut | | split | cell | cuts | acc | pts lost / cut |
|---|---|---|---|---|---|---|---|---|---|---|
| row | bottom | 1,694 | 11.90 | 3.26 | | column | far-left | 627 | 11.17 | 3.96 |
| | mid | 711 | 11.18 | 4.02 | | | inner-left | 873 | 12.16 | 3.01 |
| | top | 649 | 11.65 | 3.51 | | | inner-right | 906 | 12.02 | 3.12 |
| hand | left/red | 1,498 | 11.78 | 3.36 | | | far-right | 648 | 11.04 | 4.18 |
| | right/blue | 1,556 | 11.57 | 3.61 | | crossover | yes (10%) | 299 | **9.60** | **5.67** |
| double | yes | 626 | 11.82 | 3.43 | | | no | 2,755 | 11.90 | 3.25 |
| stack | yes | 157 | 11.53 | 3.59 | | density ±0.5 s | ≤ 7 notes | 428 | ~12.6 | ~2.5 |
| NJS | 17 | 810 | 12.33 | 2.70 | | | ≥ 9 notes | 2,389 | 11.50 | 3.68 |
| | 19 | 2,244 | 11.44 | 3.77 | | direction | down/up (0°/180°) | 82 | 9.6–10.2 | 5.3–6.5 |

Worst (row, col, hand) cells: mid-row far-left with the right hand 5.99 lost/cut,
mid-row far-right with the left hand 5.62, bottom far-left right-hand 5.46, bottom
far-right left-hand 5.15 — i.e. **crossovers to the far column** cost 1.7× a normal
cut; far columns in general 1.3×; mid row 1.2×; dense passages 1.4× sparse ones.
The t64b set reproduces every one of these (crossover 9.69 vs 11.67; far columns
10.76 / 10.98 vs 11.74 / 11.55). Hand asymmetry is small (0.2 pts) — mirror
augmentation is on in the recipe and is doing its job.

### 3c. The 3-tick-early habit — not a bench loss

100% of good cuts land early, mean −3.0 ticks. Acc by earliness (w192 top-3):
[−4,−3) ticks 11.92, [−3,−2) 11.49, [−2,−1) 8.13. **The earliest cuts are the best
cuts**; the poor ones are the late, reach-limited ones (also on the terminated
bottom-3 set: −3 ticks 12.2 vs −1 tick 6.2). Scoring is timing-blind, so the habit
costs nothing on the bench; it is a γ-premium artefact only (2026-09-13 note) and
is out of scope here.

### 3d. Terminations, misses, bombs, walls in the replays

| replay | stars | ended | last note | outcome |
|---|---|---|---|---|
| w192 botacc01 Attractor Dimension | 11.0 | 32 s | 152 s | terminated: 408 / 2,045 notes, 0 misses before |
| w192 botacc02 No Fear, No Despair | 12.8 | 122 s | 171 s | terminated: 1,281 good, 20 bad, 14 miss, 1 bomb |
| w192 botacc03 BPM Easy | 4.6 | 110 s | 160 s | terminated: 3 wall entries, 1 bomb before |
| t64b botacc01/02 Last Wind / Op. Annihilation | 4.9 / 10.2 | 123 / 176 s | 220 / 324 s | terminated, ≤ 1 bad cut before |
| t64b botacc03 TWERKMASTER | 10.4 | full | | survived; 2,052 wall-tick events, acc 0.61 |
| all 12 top-pp replays | 10–13 | full | | 0–1 miss/bad each |

Terminations are catastrophic where they happen (a 450-pp level → 26 pp) but there
are 7 of them on w192, 9 on muon_cd300, 5 on t64b, on *different* maps each time
(Attractor Dimension is the only repeat; JDM, Machinegun, ATTACK FROM MANDRAKE,
Absolute Psalm, Funk Assembly flip between 0.12 and 0.95 across leaves). Each is
a −400 pp event ≈ 0.02% of upp; ten of them are inside the per-cut bench noise
(σ_cut 24,750). They are worth logging, not worth a training change.

### 3e. Limitation

A geometric bias-vs-variance decomposition of the 15-pt component (signed x/y
offset of the cut plane from the note centre by cell) was attempted
(`replay_geometry.py`) and is **not reportable**: the replay's `sabers` record is
the end-of-tick wrist pose + blade direction (SPEC-v10 §6) while cuts are resolved
on the swept blade quad within the tick (lib.rs L533), and a blade moving at
20–40 m/s travels 0.7–1.3 m per tick, so the reconstructed line offset does not
correlate with the paid acc (r = 0.00). Doing this properly needs the kernel to log
the cut-plane offset vector in the cut event (a telemetry addition, not a training
change) — recommended, see §6 P0.

## 4. What an accuracy-point is worth (curve convexity)

`acc_component_value.py`: add +x acc-points to every good cut (capped at 15),
rescale the multiplied score, re-evaluate the curve.

| +x per good cut | Δupp | Δwpp | | +1.0 on one star bin only | Δupp | notes share |
|---|---|---|---|---|---|---|
| +0.25 | +122k | +688 | | 2–4★ | +42k | 1.6% |
| +0.5 | +268k | +1,586 | | 4–6★ | **+319k** | 24.0% |
| +1.0 | +667k | +4,310 | | 6–8★ | +156k | 19.6% |
| +2.0 | +2,455k | +22,940 | | 8–10★ | +105k | 25.3% |
| +3.13 (perfect) | +4,152k | +41,671 | | 10–12★ | +40k | 23.8% |
| post-swing perfect (+0.22) + pre (+0.06) | +138k | | | 12–14★ | +4k | 5.5% |

The MDD on a single bench cut (~100k upp) corresponds to **+0.2 acc-points per cut**
or **+0.17 pt of mean level accuracy**. wpp is a different objective: 97.4% of it
is the top 100 levels (10–13★), where the acc component is 10–11 / 15.

## 5. Training exposure vs bench weight (`train_vs_bench_weight.py`)

The ranked pool holds 49,407 windows of 25 s (98.8k with mirror), drawn **uniformly
per reset** (`cuda_env._cand` → `torch.randint`); every note is in exactly one
window, so training is uniform over notes and a level's share ∝ its duration. At
32,768 × T=32 about 1,400 episodes start per update; each window is played ~30
times in 1,068 updates (~15 with mirror counted separately).

| stars | levels | train share (windows) | train share (notes) | bench share of upp | share of headroom | share of "+0.5 pt" value | upp / notes |
|---|---|---|---|---|---|---|---|
| 2–4 | 357 | 4.8% | 1.6% | 3.8% | 2.8% | 5.7% | 2.43 |
| 4–6 | 2,458 | 37.4% | 24.0% | 33.6% | 26.6% | **45.8%** | 1.40 |
| 6–8 | 1,193 | 19.6% | 19.6% | 21.1% | 19.1% | 23.8% | 1.08 |
| 8–10 | 1,113 | 19.8% | 25.3% | 21.8% | 24.1% | 17.0% | 0.86 |
| 10–12 | 794 | 15.1% | 23.8% | 16.0% | 21.7% | 6.9% | 0.67 |
| 12–14 | 165 | 3.3% | 5.5% | 3.4% | 5.4% | 0.8% | 0.62 |

By NJS: ≥ 20 m/s content is 42% of training notes and 29% of upp / 18% of marginal
value; by NPS: ≥ 8 NPS is 38% of notes, 25% of upp, 14% of marginal value.

Answer to Q3: **high-pp content is not under-sampled; it is over-sampled relative
to its upp weight**, because hard maps are long and dense (more notes per level)
while the curve pays most for the 4–8★ maps that sit at 0.976–0.974. Relative to
the *headroom* (what a superhuman player would take) the training mix is close to
proportional (24/25/24/22 vs 27/19/24/22). So there is no static reweighting that
serves both objectives; the lever that serves both is *dynamic* (where the policy is
currently failing), §6 P1.

## 6. Proposals, ranked

Common probe frame: from scratch, 32,768 × T=32, 1,068 updates = 1.12B samples,
wide arch, Muon, lr 5.7e-4, seed 0, cd300 leaf seed 1, bench on the same pod as a
fresh w192_cd300 re-bench (bench numbers are pod-bound). Readouts: trunk acc
(last-50) vs 0.9466 — **not comparable when the sampling distribution changes** —
so add a fixed uniform-window eval (1 update of greedy play on a frozen 4,096-window
subset, once per 100 updates, ~10 s) as the training-side metric; bench upp vs
2,353,043 (MDD ~100k), acc-points per good cut vs 11.87 (MDD ~0.2), per-star acc,
and the four secondary buckets above (wall-break estimate, combo, post).

### P0 (instrument first, ½ day, no training): cut-offset telemetry + per-window score log

Mechanism: log in the cut event the signed cut-plane offset (dx, dy in the note
frame) and the blade speed / reach at the cut; log per-window mean note score in
the trainer (window index is known at reset). Evidence: the acc component is 77%
of the loss and this report cannot yet say whether it is a bias (systematic
overshoot of the far column, undercut of the mid row) or a spread; the per-window
score is the input P1 needs anyway. Cost: kernel + Rust mirror + parity test
(house pattern). Expected pp: none directly; it decides between P1/P4/P5.

### P1: failure-weighted window sampling (SONIC "bin-based adaptive sampling" / PLR-lite)

Mechanism: keep an EMA per pool window of the episode's note-score fraction s_i
(the trainer already computes it per finished episode; needs the window index per
env at `done`). Draw resets from p_i = ½·(1/P) + ½·w_i/Σw, w_i = clip(1 − s_i,
0.01, 0.15) (a capped failure rate, SONIC's recipe: "weights sampling by capped
failure rates, balancing targeted practice on challenging motions with uniform
coverage"), unseen windows at the prior mean; refresh once per update; replace
`torch.randint` in `_cand` with `torch.multinomial` (P = 98.8k, negligible). No
reward, timing, or cost change — only *which* real maps the policy sees more.
Evidence: (i) the loss is concentrated in identifiable cells the uniform draw
under-trains — crossover-to-far-column cuts lose 1.7×, far columns 1.3×, dense
passages 1.4×, wall windows lose whole combos (TWERKMASTER 71 breaks) and the duck
is never learned at ~30 plays per window; (ii) the same policy family, given 2×
samples (t64b), moved hard-map acc up (+1.0 pt on 12–14★) and easy-map acc down
(−15k upp on 4–6★) — the precision budget *does* re-allocate with exposure, which is
exactly what a weighting exploits; (iii) SONIC (arXiv 2511.07820) scales a 29-DoF G1
PPO tracker on 100M+ frames with this sampler; PHC (ICCV 2023) trains each new
primitive on the clips the previous ones fail (hard-negative mining); Rudin et al.
2022 run a promote/demote terrain curriculum at 4k envs; PLR (Jiang et al., ICML
2021) formalises value-loss-ranked level replay with a staleness/uniform mix.
Risk (from the literature and from t64b): pure failure weighting trades easy-map
precision for hard-map precision, which upp punishes and wpp rewards; the 50%
uniform floor and the 0.15 cap are the guard, and the per-star acc table is the
readout. Prompt-Curriculum-Learning (ICLR 2026) reports the curriculum gain
shrinking as batch grows — at 1M samples/update expect a modest, not dramatic,
effect. Expected: acc-points per cut +0.1 to +0.3 (**+50k to +150k upp**), wall-
break pp −10 to −20k, crossover cells +0.5–1.0 pts; trunk acc *lower* than 0.9466
(harder mix), fixed-subset eval higher. Cost: ~1 day of plumbing, 0 wall per probe.
One probe: P1 vs w192 at matched samples; second arm with the floor at 25% if the
first is positive. Prediction registered: +60k ± 60k upp, P(> MDD) ≈ 0.3.

### P2: longer leaf cooldown (cd600) and a lower entropy floor — schedule only

Mechanism: the leaf is where the accuracy component has moved: cd200 → cd300
+0.20 pts/cut (+80k upp), entropy 14.8 → 13.1 (bench is argmax, so this is real
sharpening, not sampling noise). The knee was called "past 300" on 09-12 and never
located. Evidence: §1 and the 09-12 cd table; the wide model's leaf ended at ent
12.5, still falling. Expected: **+40k to +100k upp** if the knee is past 450,
saturating otherwise; falls unaffected. Cost: +16 min per +300 updates, no code.
Probe: cd600 leaf off `runs/v40_w192/trunk_final.pt`, seed 1, same pod as a cd300
re-cut (seed 2) so the seed band is measured at the same time. Do this first — it
is the cheapest positive-expected-value run on the list and it is the control every
later probe's leaf will be compared to.

### P3: longer trunk (the 4–5B Muon trunk already queued)

Mechanism: the accuracy component is still moving with samples (muon 1.1B → t64b
2.2B: 10–12★ 10.87 → 10.98, 12–14★ 10.21 → 10.38) and the trunk was still climbing
0.3 pt / 100 upd at 1,068. Evidence: §2a per-star components; 09-12 recommendation
(1). Expected: hard-map-weighted (wpp) gain first, upp inside the MDD until the
easy-map component moves; +100–200k upp at 4B if the t64b per-star trend holds.
Cost: 3–4 h. Not a new probe design — the point is to *read it with the §1 tables*
(per-star acc-points per cut, wall-break estimate) rather than upp alone.

### P4: crossover / far-column exposure without changing rewards — mirror is on; add "swap-start" jitter? No: covered by P1

Considered and folded into P1: the two clean cell-level losses (crossover 1.7×,
far column 1.3×) are properties of windows, and a window-level failure weight
already up-weights crossover-rich and wall-rich windows. A note-level weighting is
not available without touching reward (forbidden). Left here so it is not
re-proposed.

### P5: bench-marginal (upp-optimal) static reweighting — offered, not recommended

Mechanism: weight windows by their level's dpp/dacc per note (2–8★ ×1.3–2.4,
10★+ ×0.6). Expected +30–60k upp from moving 4–8★ acc +0.1–0.2, at the cost of
hard-map precision; wpp down. It optimises the deliverable number against the
user's stated goal (superhuman on hard maps too) — listed for completeness.

### P6: more real maps (data diversity beyond the 6,097-level catalog)

SONIC's strongest axis was data volume, but their test set is out-of-distribution
motion; our bench *is* the training catalog. Adding unranked BeatSaver maps or the
other difficulties of ranked songs dilutes bench-relevant samples at fixed budget;
expected bench effect ≤ 0 short-term, positive only for generalisation (a future
"unseen ranked maps" metric). Not a probe for this campaign.

### P7: whole-map episodes / longer segments — not a bench lever

Unplayed notes are 0.1% of upp headroom; only 7 levels terminate; the cold start
every 25 s and the 0.5-s lead gap cost nothing measurable on the bench. The 09-06
analysis stands (curriculum risk from scratch, +0.8 GB). Skip.

### P8: action resolution (`--gears 7`) — hold until P2/P3 saturate

The broad 1–3-point shortfall on 63% of cuts could be a control-resolution floor
(5 gears × 30 Hz), but the component is still moving with schedule and samples, and
11.7% of cuts on 10–13★ maps already score the full 15, so the floor is not yet demonstrated.
Revisit if a cd600 leaf and the long trunk both stall the acc component at ~12.

## 7. Literature notes (what applies, what does not)

- **PLR** (Jiang, Grefenstette, Rocktäschel, ICML 2021, arXiv 2010.03934): sample
  levels by a learning-potential score (L1 value loss / GAE magnitude), rank-based
  with temperature, mixed with a staleness term and a replay probability; induces
  an emergent easy→hard curriculum on Procgen/MiniGrid. Applies as the *mechanism*
  (a scored replay distribution over a fixed level set — our 98.8k windows are
  exactly "levels"); their score is value loss, ours can be the direct note score,
  which is a cleaner signal here because the task reward is dense and bounded.
  Caveats: reported on small-batch PPO in discrete domains; "solely maximising
  replay of high-TD-error levels may lead to overfitting or curriculum collapse"
  — hence the uniform floor.
- **Robust PLR / ACCEL** (Jiang et al. NeurIPS 2021; Parker-Holder et al. ICML 2022,
  arXiv 2203.01302): regret-based UED that *edits* levels toward the frontier.
  Editing real maps is not the goal (bench = real maps); the regret framing does
  not apply. Skip.
- **SONIC** (NVIDIA GEAR, arXiv 2511.07820): 29-DoF G1 PPO tracker scaled 1.2M →
  42M params and 100M+ frames; "bin-based adaptive motion sampling that partitions
  the dataset into fixed-duration bins and weights sampling by capped failure
  rates, balancing targeted practice on challenging motions with uniform coverage";
  larger batches improved stability; gains largest on OOD motions. Direct template
  for P1 (fixed-duration bins = our 25 s windows).
- **PHC** (Luo et al., ICCV 2023, arXiv 2305.06456): progressive multiplicative
  control policy — each new primitive is trained on the clips the previous ones
  fail (hard-negative mining); fail-state recovery added the same way. Supports the
  "train on failures" direction; the capacity-allocation part is not needed at our
  scale.
- **DeepMimic RSI** (Peng et al. 2018): starting episodes at random points of the
  reference clip is what lets the agent learn late/hard phases. Our uniform window
  draw is already RSI over the catalog; `--start-pose-jitter 0.15` and
  `--crouch-start-prob 0.25` cover the pose side. Nothing to add.
- **Rudin et al. 2022** (Learning to walk in minutes): per-env promote/demote terrain
  curriculum at thousands of envs — precedent that a sampler-side curriculum works
  in massively parallel PPO without reward changes.
- **Prompt Curriculum Learning** (ICLR 2026) and the LLM-RL curriculum results in the
  search: the benefit of ordered/prioritised sampling narrows as the batch grows —
  expect P1's effect to be moderate at 1M samples/update.
- Whole-episode vs segment training: no published result bears on it for a
  memoryless policy with a ~1 s credit window; the data (§3d) settle it for the bench.

## 8. Reproduction

`projects/beat-arena/research/2026-09-14-pp-loss-analysis/`:

> The `.py` scripts below were removed on 2026-09-22 (`level_features.json` is kept); recover with
> `git show 4c235a5:projects/beat-arena/research/2026-09-14-pp-loss-analysis/<script>.py`.

- `level_features.py` — builds `level_features.json` from `python/maps/chart_cache`
  (NJS, NPS, peak density, kinds, walls, row/column/crossover/double fractions,
  per-kind maxima). 16 s.
- `ppcurve.py` — the official curve lifted from `beat_arena_env/scoresaber.py`
  (the package `__init__` needs gymnasium + the Rust module).
- `bench_decomp.py <bench.json> [--ref ...]` — §1, §2, §2c/2d, mapper table,
  paired comparisons.
- `replay_cuts.py '<replays>*.jsonl'` — §3a–3d.
- `replay_geometry.py` — the attempted §3e reconstruction (not reportable;
  see limitation).
- `train_vs_bench_weight.py <bench.json>` — §5.
- `acc_component_value.py <bench.json>` — §4.

Data: `~/beat-arena-volume-backup/beat-arena-v40/bench/*.json`,
`python/replays/v40_*.jsonl`, `python/maps/chart_cache/*.pkl`.
