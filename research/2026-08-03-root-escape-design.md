# env v10 — the root-escape instability: mechanism, A-vs-B, and whether the "real fix" exists

**Status:** analysis + mirror prototypes only. Nothing landed, nothing committed.
**Date:** 2026-08-03. **Hardware:** GTX 1080, exclusive. **Tier used for all
prototypes:** the batched torch mirror (`beat_arena_env/torch_dynamics.py`), f64
unless stated.

**Harnesses** (all in `…/scratchpad/`): `p1_capture.py`, `p1_probe.py`,
`p1_solver.py`, `p1_causal.py`, `p1_boundary.py`, `p1_margins.py`, `p1_why.py`,
`p1_norm.py`, `p2_bitwise.py`, `p2_bhole.py`, `p2_recover.py`,
`p2_adversarial.py`, `p3_eps.py`, `p3_variantD.py`, `p3_more.py`,
`mirror_variants.py`, `variants2.py`, `variants3.py`.

Every prototype variant is produced by **text surgery on the real mirror source**
(`inspect.getsource(DynBatch.substep)` → targeted string replacement → `compile`),
so every expression not explicitly edited is bitwise identical to production.
That is what makes the "bitwise on healthy" claims below meaningful rather than
approximate.

Throughout: **PROVED** = measured in this session; **INFERRED** = argued from
code/theory without a direct measurement.

---

## 0. Headline

**The stated premise is wrong.** The §19b root escape is **not** an
explicit-Coriolis instability, and none of the Part-3 candidates (implicit
gyroscopic update, geometric rotation update, adaptive substepping) would fix it.

What actually happens (PROVED):

> The spike is born **inside step 8**, in the **ω-cap / joint-limit stage**, on
> the **ankle joints owned by the foot welds**. The weld-null-space projection
> (Divergence §15c) makes that stage an **oblique — not M-orthogonal —
> projector** whose M-norm reaches **90–116** for joints in the near-immobile
> band. The fixed 12-iteration loop re-applies it, and the iteration
> **diverges geometrically: ×2.2–4.7 per iteration, ×10⁵ over one substep,
> in f64**, from a state whose root is turning at **1.9 rad/s** (h·ω = 0.008 —
> three orders of magnitude below any explicit-integrator bound).

This is the **same failure family as Divergence §19a** (COMPOSITE_EPS
1e-10 → 1e-4). That fix bounded the *per-application* amplification at
≈ √(1/1e-4) = 100 — but the loop applies it **12 times**, and 12 applications of
a gain > 1 is all the instability needs. §19b's envelope was then added as a
symptom guard on top.

Consequences for the three parts:

| Part | Verdict |
|---|---|
| 1 | Mechanism identified and proved. Coriolis is a *bystander at birth*; it becomes a positive-feedback amplifier only *after* the solver has already put ~10⁵ rad/s into the root. |
| 2 | **A as literally specified is not a null-op on healthy trajectories** (it moves the root integration past the step-11 joint clamps, which *do* modify root rows). Corrected form **A1** and form **B** are both bitwise-identical on healthy trajectories, but **B has a measured hole**: it bounds Δp only when the step-12 envelope engages, and the over-integration can happen when it doesn't. **A1 wins.** |
| 3 | The real fix is a **solver** change, not an integrator change: make the step-8 cap/limit stage M-orthogonal (plain `M⁻¹e_j`), or fold the saturated caps into the exact weld block. Both are principled and both **alter every healthy trajectory**, because the affected joints (the four ankles) are engaged on **88 % of healthy trained-policy substeps** — a full physics + recipe re-gate for a defect that occurs on 0 % of converged-policy env-ticks. **Defer to v11.** |

---

## Part 1 — Characterizing the instability

### 1.1 Escape corpus

`p1_capture.py` ran the CUDA f32 env under a random policy over 6
(N, seed, gears) configurations and flagged any control tick where the root
moved further than `ROOT_V_MAX·DT = 1.6 m` without an auto-reset — which is only
possible if step 10 integrated an unclamped root velocity. It saved the state at
the **start** of the escaping tick (the exact mirror-replayable input).

```
N=4096 seed=0 gears=3 ticks=32:  19 teleport env-ticks / 131072 = 0.0145%
N=4096 seed=1 gears=3 ticks=32:   7 / 131072 = 0.0053%
N=2048 seed=2 gears=5 ticks=48:   2 /  98304 = 0.0020%
N=2048 seed=3 gears=5 ticks=48:   5 /  98304 = 0.0051%
N=1024 seed=4 gears=7 ticks=64:   0 /  65536 = 0.0000%
N=1024 seed=5 gears=3 ticks=64:   3 /  65536 = 0.0046%
```

14 distinct sane-precursor states were kept (4 seeds × gears 3 and 5, 9 distinct
envs). Their entry conditions:

```
 #   |v_root|   |w_root|  |qd|max   tick-dp
 0     2.082      21.64     40        14.29 m
 1     2.173      1.928     39.74     16.65 m
 2     0.875      14.69     40        15.60 m
 3     1.104      1.102     39.88     62.36 m
 4     2.794      5.158     40       143.1  m
 5     0.979      5.842     39.96     57.58 m
 7     0.151      6.404     40       642.7  m
 8     2.246      19.86     40         8.46 m
 9     1.777      8.957     40         7.80 m  (gears 5)
10     3.179      5.994     40         1.97 m  (gears 5)
12     1.453      3.662     40        12.60 m  (gears 5)
13     0.234      15.89     27.96     262.9  m
```

Every escape enters with **|ω_root| between 1.1 and 26 rad/s** and **|q̇| pinned
at or near a cap** (40 = the wrist cap; the ankles ride 16). The root is *slow*.
`h·|ω_root| = 0.005 … 0.11`.

### 1.2 The Coriolis term is not the source (PROVED)

`p1_probe.py` decomposes the step-6 update per substep:

`v* = v + h·M⁻¹(τ_damped) + h·M⁻¹(−C_grav) + h·M⁻¹(−C_cor)` with
`C_grav = RNEA(q, 0)` and `C_cor = RNEA(q, v) − C_grav` (exactly the
velocity-product / gyroscopic part), then

```
dv_solve = v_after_step8  − v*
dv_clamp = v_after_step11 − v_after_step8
dv_env   = v_after_step12 − v_after_step11
```

Root-angular magnitudes at the birth substep, f64:

| state | sub | \|ω₀\| | dv_cor | dv_grav | dv_τ | **dv_solve** | dv_clamp | dp (m) | min L pivot |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 0.0 | 1.93 | **0.067** | 4e-16 | 14.9 | **1.372e+05** | 1.096e+05 | 75.88 | 0.0221 |
| 0 | 0.6 | 10.07 | **0.182** | 4e-16 | 12.2 | **2.644e+04** | 2.309e+04 | 17.99 | 0.0261 |
| 2 | 0.2 | 7.89 | **0.291** | 4e-16 | 8.07 | **1.224e+05** | 1.117e+05 | 47.06 | 0.0250 |

The Coriolis contribution to the root's per-substep Δω at the moment of birth is
**0.07 – 0.3 rad/s**. The solver's is **10⁴–10⁵ rad/s**. Four to six orders of
magnitude apart. The Cholesky pivot is healthy (~5e-4, twelve decades above the
§22 floor) — this is not a conditioning failure of M either.

Direct causal test (`p1_causal.py`) — replay the 14 escapes for one control tick
with the **entire Coriolis/velocity-product term deleted** (`nocor`, gravity-only
bias):

```
gears3 set:  prod  maxdp 339.7 m  |  nocor  maxdp 1.424e+08 m
gears5 set:  prod  maxdp  34.07 m |  nocor  maxdp 8.722 m
```

Removing Coriolis entirely does **not** prevent the escape (it perturbs the
chaotic trajectory and on the gears-3 set makes it worse). **Coriolis is
exonerated as the birth mechanism.**

Where Coriolis *does* matter: once the root is already at the envelope (|ω| = 96,
|v| = 48) the measured `dv_cor` per substep rises to **12–49 rad/s** — a ~0.1–0.5
relative per-substep growth. So the SPEC-§19b narrative ("the next substep's
Coriolis feeds on it") is correct as a description of the *sustain* phase; it is
wrong about the *ignition*.

### 1.3 Where the spike is born: inside step 8 (PROVED)

`p1_solver.py` runs an instrumented substep with taps after each PGS stage
(weld block → contacts → limit rows → each ω-cap row). Escape state 1, f64,
entry |ω| = 1.93, |q̇|max = 39.74:

```
  Lww diag min = 0.3313 (A_ww pivot min 0.11)   <- weld block is well conditioned
  limit rows active: 9   immobile joints: 0
  min mobility ratio over MOBILE joints: 1.117e-04   <- just ABOVE COMPOSITE_EPS
  it 0 weld   : |l|=2.22  |w|=12.66  |qd|max=44.38   (weld block assigns over-cap rates)
  it 0 contact: |l|=2.27  |w|=12.85  |qd|max=63.25
  it 0 limit  : |l|=2.35  |w|= 7.35  |qd|max=71.44
  it 0 caprow28: |w|=105.1   lam=266          <== JUMP   (joint 28 = r_ankle_roll)
  it 1 weld   : |w|=107.0
  it 1 caprow28: |w|=144.2   lam=345          <== JUMP
  it 2 caprow28: |w|=464.2   lam=1147
  it 3 caprow28: |w|=486.1
  it 4 caprow28: |w|=1796    lam=4391   (+ caprow2 spine_roll, caprow22 l_ankle_roll)
  it 6 caprow28: |w|=7413
  it 8 caprow28: |w|=3.16e4
  it10 caprow28: |w|=1.363e5
  it11 pgs_end : |l|=1.82e4  |w|=1.372e5  |qd|max=1.63e5
```

The offending joints are **28 (r_ankle_roll), 22 (l_ankle_roll), 2 (spine_roll)** —
the joints whose motion *is* foot motion under the welds. Growth is **geometric,
≈ ×3.9 per full iteration** and it is the ω-cap sub-stage (8d) that supplies it;
the weld block and contact rows merely re-establish the violation each round so
the cap stage fires again.

A healthy solve from the same corpus (state 2, substep 0) converges by iteration
3 and sits at |ω| = 11.404 for the remaining 9 iterations — the loop is perfectly
well-behaved when no joint sits in the band.

### 1.4 WHY the loop diverges: the cap stage is an oblique projector (PROVED)

In exact arithmetic every stage of step 8 *looks* like an M-orthogonal projection
onto a closed convex set (weld affine block; contact/limit half-spaces; cap box),
and a composition of M-orthogonal projections is firmly non-expansive — it
**cannot** grow geometrically. So the observed growth requires one of them not to
be orthogonal. It is the cap/limit stage, and the reason is the weld-null-space
projection added at §15c.

The stage's update is `v ← Q_j v + const` with

```
Q_j = I − (S e_j)(e_jᵀ) / S_jj ,   S = M⁻¹ − M⁻¹J_wᵀ A_ww⁻¹ J_w M⁻¹
```

`Q_j` is idempotent, but it is M-self-adjoint only if `M S e_j e_jᵀ =
e_j e_jᵀ S M`, which fails because `M S = I − J_wᵀ A_ww⁻¹ J_w M⁻¹` is not
symmetric. So `Q_j` is an **oblique projector** and `‖Q_j‖_M` can exceed 1.

`p1_norm.py` computes `‖Q_j‖_M = ‖M^{1/2} Q_j M^{-1/2}‖₂` per joint:

```
escape states, gears 3            n   median ‖Q‖_M   max ‖Q‖_M
  ratio (0    , 1e-04]            7      105.67       113.85
  ratio (1e-04, 1e-03]           12       78.50        94.62
  ratio (1e-03, 1e-02]           21       11.97        24.08
  ratio (1e-02, 1e-01]           14        3.44         4.25
  ratio (1e-01, 1     ]         236        1.0014       3.08
escape states, gears 5:  max ‖Q‖_M = 90.83 at ratio 1.21e-04
neutral (healthy) pose:  max ‖Q‖_M = 116.45 at ratio 7.37e-05  (below EPS -> inert)
```

`‖Q_j‖_M ≈ 1/√(S_jj/(M⁻¹)_jj)` — **exactly the §19a amplification bound**, now
measured as an operator norm rather than argued as a per-application bound. An
M-orthogonal projector would give exactly 1.0; well-conditioned joints do
(1.0014).

So: **COMPOSITE_EPS = 1e-4 admits operators of M-norm up to ~100 into a loop that
applies them 12 times.** The theoretical worst-case loop gain is 100¹² ≈ 1e24; the
realized gain is smaller (the other stages partially undo it) but still ≈ 2–5 per
iteration, which is all that is needed.

Supporting measurement — the "impulse gain into the root per unit cap correction"
`max |c_j[root]| / S_jj` (`p1_why.py`): **5.455** at ratio 1.54e-4 on the escape
states versus **1.0** at the neutral pose. Correcting an ankle rate by Δ rad/s
injects up to 5.5·Δ into the root's 6-vector, every iteration.

Rounding is *not* the driver: the relative weld-orthogonality residual
`‖J_w c_j‖ / (‖J_w‖_F‖c_j‖)` is only 3.7e-14 in f64 even in the worst band
(it is 1e-5…7e-5 in f32 — worth knowing for the production f32 tier, but the
escape reproduces identically in f64).

### 1.5 Decisive confirmation: the artifact scales with the iteration count

`p3_more.py`, escape set gears 3, one control tick, f64 — only `PGS_ITERS`
changes:

```
  PGS_ITERS= 2   max dp/substep =    0.3047 m   max |v| inside PGS =      387
  PGS_ITERS= 4   max dp/substep =    0.5904 m   max |v| inside PGS =     1824
  PGS_ITERS= 6   max dp/substep =    2.786  m   max |v| inside PGS =     8509
  PGS_ITERS=12   max dp/substep =  339.7    m   max |v| inside PGS =  9.32e5   (production)
  PGS_ITERS=24   max dp/substep = 2.97e18 m     max |v| inside PGS =  8.29e21
```

Monotone geometric in the iteration count — a loop gain of ≈ 4.7 per iteration
over 2→6. Nothing about the integrator, the timestep, or the Coriolis term can
produce that signature; only a divergent fixed-point iteration can.

### 1.6 The stability boundary — measured in the right coordinates

The classic explicit-gyroscopic boundary is **not** the operative one:

* Escapes are born at `h·|ω_root| = 0.005 … 0.11`.
* Bare explicit integrator (`p1_boundary.py`: steps 1–6 + 10 only, no solve/clamps)
  from the neutral and two escape poses survives |ω₀| = 1–20 rad/s over 1 s and
  first diverges between 20 and 50 rad/s — but that run has *all 29 joints free
  and uncapped*, so it measures the floppy-articulated-body limit, not a
  gyroscopic one. Root composite inertia ratio I_max/I_min = 3.8–6.9, which puts
  the textbook `h·ω·(I_max/I_min − 1) ≈ 1` estimate at **ω ≈ 40 rad/s** — still
  above every measured escape entry, and irrelevant since the term contributes
  0.07–0.3 rad/s at birth. **PROVED that the gyroscopic boundary is not what is
  being crossed.**

The operative boundary is the **solver** one, and it has two coordinates:

1. **Mobility ratio** `r_j = S_jj / (M⁻¹)_jj` for a joint whose cap/limit engages.
   `‖Q_j‖_M ≈ r_j^{-1/2}`. The escapes all have an engaged joint at
   `r_j ∈ [1.1e-4, 1.6e-4]` — the very bottom of the admitted band.
2. **Iteration count** — the artifact is `gain^PGS_ITERS`, and `PGS_ITERS = 12`
   is fixed.

The instability is therefore triggered when a **weld-owned joint (ankle) rides
its ω cap while its mobility ratio sits just above COMPOSITE_EPS**. That state is
reached by cap-riding contorted poses, which is why random and mid-training
policies find it and converged policies do not.

### 1.7 How close do trained policies get? (PROVED)

`p1_margins.py`, CUDA f32, the run's own ranked-map pool, 512 envs × 400 ticks
= 204,800 env-ticks per row:

| checkpoint | mode | envelope-engaged | teleports | \|ω_root\| p50 / p99.9 / max | \|v_root\| p50 / p99.9 / max | worst-env margin |
|---|---|---|---|---|---|---|
| `final.pt` | sample | **0 (0.00000 %)** | **0** | 2.41 / 16.21 / 21.76 | 0.52 / 3.84 / 16.47 | 4.41× (ω), 2.91× (v) |
| `final.pt` | greedy | **0** | **0** | 0.75 / 9.87 / 16.70 | 0.22 / 2.37 / 7.72 | 5.75× (ω), 6.22× (v) |
| `ckpt_upd0762` (mid-surge) | sample | 3 (0.00146 %) | **1 (0.00049 %)** | 4.02 / 18.77 / **96** | 0.74 / 7.39 / **48** | 1× (saturated) |

Per-env peak |q̇| is 40 (the wrist cap) for the median env in every row — the
policy *does* ride caps constantly; what it does not do is put a *weld-owned
ankle* on its cap in a near-immobile pose.

**Blast radius, honestly sized:** a converged policy never touches the envelope.
The exposure is early/mid training and exploration. At `ckpt_upd0762`'s
0.00049 % and a 400 M-step run, that is order **2 × 10³ teleport events per
400 M run**, each producing NaN/garbage obs and rewards for that env — which is
exactly the mechanism behind the fp16 loss-scale story and the §22 bench panic.
Small but not negligible, and concentrated where the policy is most plastic.

---

## Part 2 — A vs B, deeply

### 2.0 What actually consumes the integrated `p` (traced, PROVED)

Within one substep, after step 10 writes `q`, `p`, `quat`:

* **step 11** reads `q` only (the limit-hit test and the clamp), and uses `ecols`
  = `M⁻¹e_j` computed from the **pre-integration** factor. It never reads `p` or
  `quat`.
* **step 12** reads `v` only.
* the env layer's per-substep work (`torch_env._tick_dynamics`, `lib.rs`) reads
  only `dyn.v` (effort) and the step-8 contact caches. `refresh_fk` runs once
  *after* the 8 substeps.

So the integrated `p`/`quat` are consumed **only by the next substep's `fk_all`**
→ weld Baumgarte bias `−(β/h)·C_pose`, contact geometry, limit reach — and by the
tick-end observation. This is verified identically in `torch_dynamics.py`,
`rust/src/dynamics/step.rs` and `csrc/beat_arena_cuda.cu`.

**Consequence:** the cascade the SPEC describes is real and the bias term is the
carrier — a 39 m position error becomes a `0.2/h × 39 = 1.9e3` m/s weld bias next
substep. Bounding Δp to `h·48 = 0.2 m` bounds that bias at `0.2/h·0.2 = 9.6` m/s,
i.e. **the same order as the envelope itself**. (PROVED arithmetically; the
measured end-of-tick weld error in §2.3 confirms it empirically.)

### 2.1 A as literally specified is NOT a null-op (PROVED)

"Apply the envelope before step-10 integration" has two readings, and the obvious
one is wrong:

* **A (root integration moved to the end, after steps 11 and 12).** Semantically
  the cleanest — `p_{k+1} − p_k = h·v_{k+1}` becomes an identity — but step 11's
  joint-clamp impulses `Δv = M⁻¹e_j·λ` have **non-zero root rows**, so this
  integrates a *different* root velocity on every substep. `p2_bitwise.py`:

  ```
  passive     (τ=0):      max|Δ| 1.8e-26   (near-null impulses)
  gentle      (0.15 τmax) max|Δp| 1.46e-03 m, |Δv| 0.42   over 6 ticks
  full-random (τmax)      max|Δp| 4.53e-02 m, |Δv| 45.3   over 6 ticks
  full-random f32         max|Δp| 1.08e-02 m, |Δv| 20.2
  ```
  This is a **science change on every trajectory** — a full physics + recipe
  re-gate. Rejected as the near-term fix (it is a legitimate v11 candidate on
  correctness grounds; see §3.6).

* **A1 (the correct null-op form): run a second, identical envelope block
  immediately before step 10, and keep the step-12 envelope.** Because it never
  engages on a healthy solve, `torch.where(bad, …, v)` returns `v` bitwise.
  Step 12 must stay, because step 11 runs after step 10 and can still pump root
  rows.

  Two sub-forms exist. **A1a** (prototyped, recommended) clamps the *stored*
  `v` in place, so step 11 also starts from an enveloped root and the invariant
  becomes "after the solve, the stored root velocity is never outside the
  envelope, at any point in the substep". **A1b** would integrate from a clamped
  temporary and leave `v` untouched for step 11 — one more temporary, no benefit,
  and a weaker invariant. Both are bitwise-identical on healthy trajectories.

### 2.2 Bitwise-on-healthy gate (PROVED)

`p3_variantD.py`, four suites, production vs each variant, exact tensor equality
on (p, quat, q, v) after the full run:

```
  passive         (n=8,  6 ticks, f64, τ=0)      A1=BITWISE  B=BITWISE  D=BITWISE  D+A1=BITWISE
  gentle          (n=8,  6 ticks, f64, 0.15τ)    A1=BITWISE  B=BITWISE  D=BITWISE  D+A1=BITWISE
  full-random     (n=32, 8 ticks, f64, τmax)     A1=BITWISE  B=BITWISE  D=BITWISE  D+A1=BITWISE
  full-random f32 (n=32, 8 ticks, f32, τmax)     A1=BITWISE  B=BITWISE  D=BITWISE  D+A1=BITWISE
  envelope-engaged substep-envs in all four suites: 0
```

**Engagement frequency on healthy rollouts** (§1.7): `final.pt` = 0 in 204,800
env-ticks; `ckpt_upd0762` = 3 in 204,800 (0.0015 %). The true blast radius of any
of these variants is that set.

### 2.3 Escape behaviour (PROVED)

`p3_variantD.py`, the captured escape states, one control tick (8 substeps), f64:

```
gears 3, 10 states                max dp/substep    |p|max     weld_end   |v_root|
  prod                              339.7 m         271.5      271.8       107.3
  A1                                  0.2 m           1.381      1.436     104.5
  B                                   0.2 m           1.317      1.399      99.11
  D  (envelope per PGS iteration)     0.2 m           1.334      1.427      98.34
  D+A1                                0.2 m           1.334      1.427      98.34

gears 5, 4 states
  prod                               34.07 m         21.88      21.79      107.3
  A1                                  0.2 m           1.460      1.018      74.74
  B                                   0.2052 m        1.350      1.289      68.75
  D                                   0.2 m           1.473      0.3889     57.36
```

`0.2 m` is exactly `h·ROOT_V_MAX`. All three variants convert a 340 m teleport
into a bounded 0.2 m step and keep the body within 1.5 m of the origin with a
weld error of order 1 m instead of 272 m. **The cascade provably dies**: with
Δp ≤ 0.2 m the next substep's weld bias is ≤ `0.2/h·0.2 ≈ 9.6` m/s instead of
`1.9e3` m/s.

### 2.3b Does the *env* recover? (PROVED — full env layer, rewards included)

`p2_recover.py` injects each captured escape into the **full `TorchBeatVecEnv`**
(obs + reward + done, the env layer the CUDA backend mirrors), holds the recorded
action, and runs **40 control ticks = 320 substeps**, f64:

```
gears 3, 10 states       Δp/substep(avg)  max|p|    weld_end   |v_root|_end  nf_obs  min reward
  prod                       901.8 m       7299 m   0.000875      11.75        0      -19.05
  A1                           0.191        1.465   0.00211        9.163       0       -1.008
  A (root integ. last)         0.191        1.478   0.005689       5.444       0       -0.512
  B (rewind p + quat)          0.191        1.484   0.000758       6.310       0       -1.001
  B-noquat (rewind p only)     0.191        1.521   0.523         91.38        0       -1.007
  eps1e-2                      0.180        1.500   0.0402        12.80        0       -1.013
  eps1e-2 + A1                 0.180        1.486   0.0402        11.22        0       -1.013

gears 5, 4 states
  prod                          2.74        21.88   0.001181       7.629       0       -1.553
  A1                            0.189        1.470   0.001533      3.588       0       -1.018
  B (rewind p + quat)           0.189        1.465   0.000786      5.920       0       -0.538
  eps1e-2                       0.185        1.515   0.129        16.41        0       -1.012
```

Readings:

* **Production does eventually recover** — after flying **7.3 km** away, and it
  pays **reward −19.05** on the way (the normal per-tick floor is ≈ −1). That
  reward, and the observation that produced it, are what the policy learns from.
  A1 keeps the worst reward at −1.008, i.e. inside the normal range.
* **A1 / A / B all settle**: body within 1.5 m, weld error back to millimetres,
  |v_root| back to single digits, episode continues sanely.
* **`B-noquat` (rewind position but not orientation) does NOT settle**: weld error
  0.523 m and |v_root| still **91.4** after 40 ticks. So B must re-integrate the
  quaternion too — which is the extra `sin`/`cos`/`sqrt` on a divergent branch in
  the CUDA kernel counted in §2.5.
* `eps1e-2` recovers but leaves a 40–130 mm weld error — visible confirmation that
  reclassifying those joints degrades constraint satisfaction (§3.3).
* **No non-finite obs or rewards in any f64 40-tick replay, including
  production** — the NaNs are an f32-tier phenomenon (production went non-finite
  in all six f32 adversarial configurations, §2.6). The f64 harm is
  *garbage-but-finite* rewards; the f32 harm is NaN.

### 2.4 B has a real hole — the deciding measurement (PROVED)

B rewinds `p` **only when the step-12 envelope engages** — and that conditioning
is exactly what makes B bitwise on healthy trajectories. But **step 10 integrates
the post-step-9 velocity while step 12 tests the post-step-11 velocity**, and
step 11's impulses change root rows. Whenever the root is over-envelope at step
10 but back under at step 12, B never rewinds and the whole over-integration
survives.

`p2_bhole.py` counts exactly that:

```
                                     over at step 10 | engaged at 12 | B BLIND | worst un-rewound dp
escape set gears 3 (8 substeps)            45              45            0        0 m
escape set gears 5 (8 substeps)            19              18            1        0.1578 m
adversarial weldviol 0.05, 50% on-stop    672             461          221        0.2467 m
adversarial weldviol 0.20, 80% on-stop    752             550          215        0.2526 m
```

**29–33 % of over-envelope integrations are invisible to B.** And in
`p2_adversarial.py` the leak shows up directly as a violated bound:

```
adversarial, f64, per-substep Δp (bound h·48 = 0.2 m)
  weldviol 0.02 / 30% on-stop, gears 3:  prod 3932      A1 0.2   B 0.3334   D 0.2
  weldviol 0.02 / 30% on-stop, gears 5:  prod 7.87e8    A1 0.2   B 0.3379   D 0.2
  weldviol 0.10 / 60% on-stop, gears 3:  prod 2.996e10  A1 0.2   B 0.4737   D 0.2
  weldviol 0.10 / 60% on-stop, gears 5:  prod 4.73e7    A1 0.2   B 0.2562   D 0.2
```

**A1 hits the bound exactly, every configuration. B exceeds it every
configuration.** Making B unconditional would fix the hole but would turn B into
variant A — the healthy-trajectory-changing one.

### 2.5 A1 vs B on the other axes

| axis | A1 | B |
|---|---|---|
| Δp bound | `h·48 = 0.2 m` **unconditionally, by construction** | only when step 12 engages — measured leak to 0.47 m |
| bitwise on healthy | yes (PROVED) | yes (PROVED) |
| what the next substep's solver sees in `v` | identical to today (step 12 is unchanged) | identical to today |
| interaction with the weld `qpre`/bias | bounds the bias by construction (§2.0) | bounds it only on engagement |
| orientation | the quaternion is integrated from the enveloped ω too — `\|u\| ≤ 96h/2 = 0.2 rad` — for free | needs a *second* exp-map evaluation on the rewind path. **Skipping it does not work**: `B-noquat` leaves the body at weld error 0.523 m and \|v_root\| 91.4 after 40 ticks (§2.3b) |
| state carried across step 10→12 | none | `p_pre` (3 floats) + `quat_pre` (4) live across the whole step-10…12 region |
| CUDA cost | ~25 scalar ops + 2 `sqrt` per substep, no new live state | 7 extra live floats in a kernel already at 255 regs / 45 KB frame → spill risk; plus a divergent branch containing `sin`/`cos`/`sqrt` |
| tri-tier transcription | one copy-paste of an existing 12-line block | new state machine + a duplicated exp-map block |
| SPEC semantics | "the envelope is applied before *anything* consumes the root velocity, and again after step 11" — two sites, one rule | "positions are rewound when the envelope engages" — a rewind is a new concept in the pipeline |

**Verdict: A1.** It is strictly better on the property that matters (an
unconditional bound), strictly cheaper in the kernel, and simpler to transcribe
into three tiers. B's only advantage — the `Δp = h·v_final` identity — is not
actually achieved by the bitwise-safe form of B.

### 2.6 Adversarial survival (PROVED, with one caveat)

`p2_adversarial.py` generates states far worse than the env can reach: `q`
uniform in `[lo,hi]` with 30–90 % of joints parked **exactly on a stop**, `q̇`
saturated at ±ω_cap with random signs, root ω up to 96 and v up to 48, uniform
random root orientation, and the feet welds pre-violated by up to 0.02 / 0.10 /
0.30 m. 192 states per configuration, 16 substeps of full random gears.

```
per-substep Δp (bound h·48 = 0.2 m) | max |v| INSIDE the PGS | non-finite
f64  weldviol 0.02 / 30% on-stop  g3: prod 3932      1.12e07  0 | A1 0.2  1.53e06  0 | B 0.3334 1.58e07 0 | D 0.2 6.68e09 0
f64  weldviol 0.02 / 30% on-stop  g5: prod 7.87e08   2.10e12  0 | A1 0.2  7.48e08  0 | B 0.3379 7.58e08 0 | D 0.2 2.11e11 0
f64  weldviol 0.10 / 60% on-stop  g3: prod 3.00e10   3.03e13  0 | A1 0.2  5.53e06  0 | B 0.4737 4.56e07 0 | D 0.2 3.66e11 0
f64  weldviol 0.10 / 60% on-stop  g5: prod 4.73e07   1.45e11  0 | A1 0.2  1.46e07  0 | B 0.2562 1.76e07 0 | D 0.2 1.24e15 0
f64  weldviol 0.30 / 90% on-stop  g3: prod  NON-FINITE          | A1 0.2  1.09e10  0 | B 0.257  1.09e10 0 | D 0.2 2.24e14 0
f64  weldviol 0.30 / 90% on-stop  g5: prod 2.70e05   5.17e08  0 | A1 0.2  6.74e07  0 | B 0.4047 9.87e07 0 | D 0.2 1.62e17 0
f32  weldviol 0.02 / 30% on-stop  g3: prod  NON-FINITE          | A1 0.2  1.13e07  0 | B 0.3232 5.92e07 0 | D 0.2 9.13e09 0
f32  weldviol 0.02 / 30% on-stop  g5: prod  NON-FINITE          | A1 0.2  1.46e08  0 | B 0.3302 2.02e07 0 | D 0.2 7.88e10 0
f32  weldviol 0.10 / 60% on-stop  g3: prod  NON-FINITE          | A1 0.2  1.28e08  0 | B 0.3297 9.60e06 0 | D 0.2 6.77e08 0
f32  weldviol 0.10 / 60% on-stop  g5: prod  NON-FINITE          | A1 0.2  1.44e09  0 | B 0.2962 1.42e08 0 | D 0.2 8.27e15 0
f32  weldviol 0.30 / 90% on-stop  g3: prod  NON-FINITE          | A1 0.2  4.41e09  0 | B 0.2581 3.42e07 0 | D 0.2 2.30e15 0
f32  weldviol 0.30 / 90% on-stop  g5: prod  NON-FINITE          | A1 0.2  3.24e07  0 | B 0.9750 3.09e07 0 | D 0.2 6.54e14 0
```

* **Production goes non-finite in ALL SIX f32 configurations** and at the hardest
  f64 setting. **A1, B and D stayed finite in every one of the 12 configurations**
  (192 states × 16 substeps each). A1 alone therefore removes the NaN in every
  adversarial state found.
* **A1 held Δp = 0.2 m exactly in all 12 configurations. B exceeded the bound in
  all 12** (0.256 – **0.975 m**, i.e. up to ~5× the bound).
* The residual risk A1 does *not* remove: the **intra-substep** velocity the PGS
  reaches. Production hit `max |v| inside the PGS = 3.0e13` (f64); A1's worst was
  4.4e9; **D reached 1.6e17** — clamping the root every iteration removes it as a
  sink and lets the joint rates run further. The *stored* state is always bounded
  (caps + step 11 + step 12), so a substep always starts bounded — but in **f32**
  an intra-PGS magnitude above ~1e19 squares to overflow in the next substep's
  RNEA/CRBA. **No adversarial state found reached that under A1**, so this is a
  theoretical rather than a demonstrated gap; the option-C backstop (§4.3) is
  recommended for the *garbage-trajectory* reason, not because A1 was seen to
  NaN. It also argues against pairing D with A1: A1 alone keeps the intra-PGS
  magnitude five to seven decades lower than D does.

---

## Part 3 — Is there a real fix?

### 3.1 The Part-3 candidates as specified do not address the bug

| candidate | verdict |
|---|---|
| **(i) semi-implicit / implicit gyroscopic update** | **Would not fix it.** Deleting the Coriolis term entirely does not prevent the escapes (§1.2), so making it implicit cannot either. Structurally it is also the most expensive thing on the list: it requires `∂C/∂v` (a 35×35 non-symmetric Jacobian; analytic RNEA derivative ≈ 3–5× RNEA, or 35 finite-difference RNEA calls), and `M + h·∂C/∂v` is **non-symmetric** — which retires the "M symmetric PD, no-pivot Cholesky" invariant that the entire tri-tier bitwise parity method rests on. Partial-pivoting LU has *state-dependent row swaps*, i.e. a knife-edge branch that the CUDA tier would take differently from Rust on a 1-ulp difference. Estimated substep cost +30–60 %, giving back most of the 1.64× the kernel just gained, and it **alters every healthy trajectory** (full physics + recipe re-gate). INFERRED cost, PROVED irrelevance. |
| **(ii) exact/geometric rigid-rotation update for the root** | **Not applicable.** The root's 6 DoF are not an isolated rigid body — its angular velocity is coupled to all 29 joints through a dense `M`, so there is no root angular momentum to rotate independently. The part that *is* geometric (the quaternion update via the exp map) is already implemented that way. Moot in any case given §1.2. |
| **(iii) adaptive substepping on near-boundary states** | **Would not fix it — measured, it makes it WORSE.** `p3_more.py`, escape set, one control tick, f64, same total time: <br>`gears 3:  h → 339.7 m ;  h/2 (16 sub) → 213.9 m ;  h/4 (32 sub) → NON-FINITE`<br>`gears 5:  h →  34.07 m ;  h/2 →   3.75 m ;  h/4 →  13.71 m`<br>The divergence happens *inside one substep's 12-iteration loop*; `h` does not enter the loop gain `‖Q_j‖_M` at all. It only enters the Baumgarte gain `β/h` (which **doubles** when `h` halves) and the limit-row reach `h·ω_cap` (which halves, changing which rows assemble). Determinism/parity would have been acceptable (the trigger is a bitwise-reproducible predicate on the f64 tier and falls inside the existing CUDA tolerance story) and the CUDA cost small (~0.005 % of envs flagged ⇒ ~0.16 % of 32-env warps stall) — but it buys nothing and can cost. |

**The most valuable single result of this investigation is that the "real
underlying fix" the user asked about is aimed at the wrong subsystem.**

### 3.2 The actual candidate space (solver-side)

The loop gain must be brought to ≤ 1. `‖Q_j‖_M ≈ r_j^{-1/2}`, so:

| candidate | what it does | measured / estimated |
|---|---|---|
| **E1 — raise COMPOSITE_EPS** | more joints declared weld-immobile; their bounds fall through to step 11 (plain `M⁻¹e_j`, which **is** M-orthogonal, `‖Q‖_M = 1` exactly) | **Works**: `p1_causal.py`, escape set gears 3, maxdp 339.7 m → **0.2752 m at 1e-2** (and 5.507 m at 1e-3); gears 5, 34.07 → 0.3157. **But**: to get the 12-loop gain under ~10 you need `‖Q‖ ≤ 1.2`, i.e. `r ≥ 0.7` — absurd. And the band is *populated on healthy trajectories*: see §3.3. |
| **E2 — bound the cap impulse λ** | fixed clamp on the step-8d impulse | Mixed: `lam ≤ 1e2` gives 0.3253 m (gears 3) but `lam ≤ 1e3` gives **215 m** on gears 5 — the safe threshold is state-dependent, i.e. a magic number with no invariant behind it. Rejected. |
| **E3 — drop the step-8d ω-cap pass** (caps left to step 11) | removes the oblique operator from the loop entirely | 0.3347 m on gears 3 but **4.472 m on gears 5** — the limit rows (also projected, also oblique) still diverge. Also directly re-opens §15c's 38 mm weld drift. Rejected alone. |
| **E4 — use the PLAIN `M⁻¹e_j` direction for cap/limit rows inside step 8** | makes the stage an M-orthogonal projection, `‖Q‖_M = 1` **by construction** — the loop becomes provably non-expansive | The theoretically correct fix. Cost: re-opens exactly what §15c measured (up to 38 mm foot drift under full random gears), i.e. it trades a stability guarantee for a constraint-accuracy regression. Not prototyped (a fair evaluation needs the weld-drift gate suite, which is a Rust cargo test). **Recommended as the v11 investigation.** |
| **E5 — assemble the saturated ω-cap rows into the exact weld block** | one exact solve instead of alternating between two nearly-parallel constraints; no oblique operator at all | The "right" answer. Block grows from 12×12 to up to 41×41: `A_ww` assembly `O(rows²·nv)` = 5 k → 59 k flops/substep, Cholesky 12³/6 → 41³/6 = 288 → 11 k flops. **≈ +50 % substep cost**, i.e. it would give back all of the recent 1.64×. Variable row count = variable work per thread in a one-thread-per-env kernel (warp divergence). Also a science change everywhere. Rejected on cost for now. |
| **E6 / D — move the §19b envelope INSIDE the PGS loop** | applies the root envelope at the end of each of the 12 iterations | **Bitwise on all four healthy suites (PROVED).** Bounds Δp ≤ 0.2 m by construction (the envelope is the last operation before step 10). On the escapes it recovers *better* than A1/B (gears 5 end-of-tick weld error **0.389 m vs 1.018 / 1.289**, final \|v_root\| 57 vs 75 / 69). **But** it does not stop the joint-rate amplification — it makes it worse: adversarially `max |v| inside the PGS` reached **1.6e17 for D vs 6.7e7 for A1** on the same set (clamping the root removes it as a sink). Cost: 12 × (~25 ops + 2 `sqrt`) per substep ≈ 0.3 %. Not recommended. |
| **E7 — reduce PGS_ITERS** | `artifact = gain^ITERS` | Measured (§1.5): 12 → 6 takes the artifact from 339.7 m to 2.79 m, 12 → 2 to 0.30 m. A real lever, and it would make the kernel *faster*. But it is a wholesale constraint-accuracy change (every weld/contact/limit residual gets worse) — a full physics re-gate. Listed for completeness, not recommended. |

### 3.3 The killer constraint on every E variant: the danger band is populated on HEALTHY states (PROVED)

`p3_eps.py` harvests states from the CUDA env and computes the per-joint mobility
ratio in the mirror:

```
TRAINED final.pt (gears 5, ranked pool), 400 states x 29 joints
  ratio band            all joints          of joints whose limit-row or cap ENGAGES
  (1e-06, 1e-04]     77   ( 0.66 %)              9   ( 0.20 %)
  (1e-04, 1e-03]    194   ( 1.67 %)             57   ( 1.26 %)
  (1e-03, 1e-02]   1329   (11.46 %)            697   (15.43 %)
  (1e-02, 1e-01]    142   ( 1.22 %)              9   ( 0.20 %)
  (1e-01, 1     ]  9858   (84.98 %)           3745   (82.91 %)
  ENGAGED joints in the (1e-4, 1e-2] danger band, by joint id:
      21 (l_ankle_pitch):304   22 (l_ankle_roll):217
      27 (r_ankle_pitch):127   28 (r_ankle_roll):106
  states with >=1 engaged joint in the danger band: 352/400 (88.0 %)

RANDOM gears 3: 66.1 % of states;  RANDOM gears 5: 65.5 % of states — same four joints.
```

**88 % of healthy trained-policy substeps have an engaged ankle in the very band
that any COMPOSITE_EPS raise would switch off.** So E1 (and by extension anything
that reclassifies that band) is *not* a free fix: it changes the ankle's
constraint treatment on almost every substep of every healthy trajectory. That is
a full physics re-gate (P2's weld/contact suites) **plus** a training re-gate
(every recipe and every record was measured on the current body).

Note also what this table proves about the trigger: the band is populated
constantly and healthy trajectories still do not blow up. The escape needs the
ratio at the very *bottom* of the band (`1.1e-4 … 1.6e-4`, i.e. `‖Q‖_M ≈ 80–95`)
**and** a cap violation that the other stages keep regenerating. That is why the
incidence is 1e-4 rather than 1.

### 3.4 Feasibility matrix (tri-tier / determinism / perf / science)

| | Rust f64 | CUDA f32+f64 | torch mirror | bitwise tiers preserved | determinism | perf Δ | alters healthy trajectories |
|---|---|---|---|---|---|---|---|
| **A1** | 12-line copy of the §19b block before step 10 | same, `if` form | same, `where` form | **yes** (never engages on healthy) | trivially (two scalar comparisons) | < 0.1 % | **no** (PROVED bitwise) |
| **B** | + `p_pre`, `quat_pre` state, rewind branch, second exp-map | + 7 live floats in a 255-reg/45 KB kernel, divergent `sin/cos` branch | same | yes | yes | ~0.1 % + spill risk | **no**, but the bound leaks (§2.4) |
| **A** (root integ. last) | reorder | reorder | reorder | yes (structurally) | yes | 0 | **YES — every trajectory** |
| **D** (envelope per PGS iteration) | 12-line block inside the loop | same | same | **yes** (PROVED bitwise) | yes | ~0.3 % | **no** |
| **E1** COMPOSITE_EPS | one constant | one constant | one constant | yes (constant is shared) | yes | 0 | **YES — 88 % of substeps** |
| **E4** plain cap/limit direction | delete the projection for those rows | same | same | yes | yes | **faster** (drops 29 weld-block solves/substep) | **YES**, and re-opens §15c weld drift |
| **E5** caps in the exact block | variable-size block | variable-size block, warp divergence | variable-size block (batched: hard) | yes | yes if the active set is a deterministic predicate | **≈ +50 %** | **YES** |
| **(i)** implicit Coriolis | non-symmetric solve, LU+pivoting | ditto, pivot branch is a parity knife-edge | ditto | **no** — retires no-pivot Cholesky | pivoting is state-dependent | **+30–60 %** | **YES — every trajectory**, and **does not fix the bug** |
| **(iii)** adaptive substep | loop-count branch | warp divergence | batched: masked substeps | yes | yes (bitwise predicate on f64) | ~0.2 % | only on flagged states — **but does not fix the bug** |

### 3.5 Honest weighing

The user's stated preference is the real underlying fix. Taking it seriously:

* The real underlying fix **exists** and is now precisely characterised: make the
  step-8 cap/limit stage an M-orthogonal projection (E4), or fold the saturated
  caps into the exact block (E5). Both are principled; both provably remove the
  geometric gain rather than bounding its consequences.
* Both **alter every healthy trajectory**, because the affected joints (the four
  ankles) are engaged on 88 % of healthy substeps. That means: re-run the P1/P2
  physics gates, re-run the MuJoCo oracle bands, re-establish the Rust↔torch
  bitwise bar, re-gate the CUDA tier, and then **re-measure every v10 record and
  re-tune every recipe**, because the body's low-level behaviour changed.
* Meanwhile the thing being fixed happens on **0 %** of converged-policy env-ticks
  and **0.0005 %** of mid-training ones.

The cost/benefit is therefore lopsided. **A1 now + E4/E5 scoped as a v11 body
change is strictly better than E now.**

### 3.6 A footnote in variant A's favour, for the v11 file

Variant **A** (root integration after steps 11 and 12) is not just a fix — it is
arguably *more correct*: it makes `p_{k+1} − p_k = h·v_{k+1}` an identity, which
is what semi-implicit Euler is supposed to be, and it removes the current
situation where the position is advanced with a velocity the state never stores.
It changes healthy trajectories by ~1.5 mm over 6 gentle ticks. If a v11 body
revision happens anyway (for E4/E5), **A should ride along in the same re-gate**
and A1 can then be retired back to a single envelope site.

---

## 4. Recommendation and landing plan

### 4.1 Land now: A1 + the counter + the C backstop

**A1** — a second, verbatim copy of the §19b root-velocity envelope immediately
before step 10, keeping the existing step-12 envelope.

Rationale, in one line: it is the only prototyped option that bounds `Δp` by
`h·ROOT_V_MAX` **unconditionally**, it is **bitwise identical on healthy
trajectories** (proved on four suites including full-random f64 and f32), it
costs < 0.1 % of a substep, and it is a copy-paste in all three tiers.

**Do not** fold in **D** (the same block at the end of each PGS iteration) despite
its attractions — it is equally bitwise-safe and gives measurably better escape
recovery (end-of-tick weld error 0.39 m vs A1's 1.02 m on the gears-5 escapes),
but the adversarial sweep shows it drives the **intra-PGS** magnitude three to
seven decades *higher* than A1 (1.6e17 vs 6.7e7 on the same set): clamping the
root every iteration removes the root as a sink and lets the joint rates run
further. Since the one remaining NaN path is f32 overflow of intra-PGS
magnitudes, D moves the wrong number. **Land A1 alone.**

### 4.2 Tiers and gates

| tier | change | gate |
|---|---|---|
| `rust/src/dynamics/step.rs` (reference) | 12-line envelope block before the step-10 position/quaternion integration | existing `tests_dyn` / `tests_soak` cargo suites must be **unchanged green**; add `tests_soak::root_step_bounded` asserting `‖Δp‖ ≤ H_SUB·ROOT_V_MAX` on the baked §19b regression state |
| `beat_arena_env/torch_dynamics.py` | same block, `where` form | P4 Rust↔torch **f64 BITWISE** suite must be unchanged green (it is, by construction — the block is a no-op on every state those tests visit) |
| `csrc/beat_arena_cuda.cu` | same block, both `T` instantiations | P5 `test_v10_parity_cuda.py` unchanged green; rebuild the ninja cache |
| new | envelope-hit counter, popped like `pd_hits` | `pop_env_hits()` on RawBeatSim/VecBeatSim + CUDA, reported per update in `train_batch` and in `bench_scoresaber`'s results JSON next to `pd_floor_hits` |

Additional gates worth running before landing:

1. **Bitwise-on-healthy replay**: the existing P4 trajectory suites, plus a
   400-tick `final.pt` rollout under both binaries — obs/reward/done byte-identical.
2. **Escape replay**: the 14 captured states in `p1_escapes.npz` — assert
   `max Δp/substep ≤ h·ROOT_V_MAX` and finite state after 40 ticks.
3. **Adversarial**: `p2_adversarial.py`'s generator ported to a cargo test at the
   0.10 m weld-violation / 60 %-on-stop setting; assert no non-finite and the Δp
   bound.

### 4.3 Option C (terminate-on-escape) — yes, but for a different reason than expected

The adversarial sweep did **not** find a state where A1 still NaNs: production
went non-finite in all six f32 configurations and at the hardest f64 one, while
A1 stayed finite in all twelve. So C is not needed as a *numerical* backstop on
the evidence available.

It is still worth landing, for the reason that actually motivated this hunt: an
env that engaged the envelope is running a physically meaningless trajectory, and
feeding its observations and rewards to the policy is the harm (the fp16
loss-scale collapse and the §22 bench panic were both downstream of exactly
that). A1 makes that trajectory *bounded*; it does not make it *meaningful*.

A1 also does not bound the magnitude the PGS reaches inside a substep (A1's
adversarial worst: 4.4e9; production: 3.0e13). In f32 an intra-PGS magnitude
above ~1e19 would overflow the next substep's velocity products. Unobserved
under A1, but not excluded — C closes it as a side effect.

Recommended C form — cheap, deterministic, and it *reports* rather than hides:

* the envelope-hit counter of §4.2 is per-env;
* an env whose counter is non-zero **for this tick** gets `status = ESCAPED`, its
  observation is emitted from the pre-tick state (or zeroed), its reward is 0 for
  that tick, and `done = true` (auto-reset);
* the count is reported per update alongside `pd_floor_hits`.

This makes a physically meaningless trajectory terminate instead of feeding the
policy garbage, which is the actual harm (the fp16 loss-scale collapse and the
§22 bench panic were both downstream of exactly that). It is a **reward/episode
semantics change**, so it needs its own smoke gate (a 47-update run at the v10
recipe, r1 and steps/s unchanged) — but at 0.0005 % incidence it cannot move any
statistic measurably.

### 4.4 SPEC amendment sketch

Replace §3 step 12 with:

> 12. **Root velocity envelope guarantee (ADDED at P3 — Divergences §19b;
>     AMENDED 2026-08-03 — Divergences §23)**: the envelope below is applied
>     **twice per substep — immediately BEFORE step 10 and again after step 11**:
>     if `|ω_root| > 96` rad/s, rescale the root angular 3-vector onto 96; if
>     `|v_root| > 48` m/s, rescale the linear 3-vector onto 48.
>     The pre-step-10 application is what bounds the INTEGRATED root motion:
>     `‖Δp‖ ≤ h·48` and `|Δθ| ≤ h·96` per substep, unconditionally. The
>     post-step-11 application is what bounds the STORED velocity (step 11's
>     momentum-consistent joint impulses have non-zero root rows and can
>     re-exceed the envelope after step 10 has run). Both are no-ops on any
>     state ordinary motion reaches — a converged v10 policy measured 0
>     engagements in 204,800 env-ticks — so the amendment is BITWISE
>     transparent on healthy trajectories.

And add Divergence §23 (text sketch):

> 23. **§19b root envelope applied before integration; the escape mechanism
>     re-diagnosed (2026-08-03).** §19b attributed the root escape to the
>     explicit Coriolis term at `h·ω ≈ 1`. **That attribution is wrong**
>     (measured): escapes are born at `h·|ω_root| = 0.005…0.11`, the Coriolis
>     contribution to the root's Δω at the birth substep is 0.07–0.3 rad/s
>     against the solver's 1e4–1e5 rad/s, and deleting the velocity-product bias
>     term entirely does not prevent them. The spike is born **inside step 8's
>     ω-cap/limit stage**: the §15c weld-null-space projection makes that stage
>     the operator `Q_j = I − (S e_j)e_jᵀ/S_jj`, which is idempotent but **not
>     M-self-adjoint** (`M S` is not symmetric), so `‖Q_j‖_M ≈ (S_jj/(M⁻¹)_jj)^{-1/2}`
>     — measured **90–116** for the ankle joints just above the §19a
>     COMPOSITE_EPS = 1e-4 threshold. Twelve iterations of a gain > 1 diverge
>     geometrically: measured ×2.2–4.7 per iteration, `|v|` inside the PGS 387
>     (2 iterations) → 8.5e3 (6) → 9.3e5 (12) → 8.3e21 (24) on the same state in
>     f64. §19a bounded the per-application amplification; it did not bound the
>     loop. The **envelope is therefore also applied before step 10**, which
>     bounds the integrated root motion (`‖Δp‖ ≤ h·48`) and hence the next
>     substep's Baumgarte weld bias (`≤ β/h·h·48 = 9.6` m/s instead of the
>     measured 1.9e3 m/s), killing the cascade. The underlying non-orthogonal
>     projection is NOT fixed here: making the cap/limit rows use the plain
>     (M-orthogonal, `‖Q‖_M = 1`) `M⁻¹e_j` direction, or folding the saturated
>     caps into the exact weld block, would remove the gain itself — but the
>     affected joints (the four ankles) are engaged on **88 %** of healthy
>     trained-policy substeps, so either change alters every trajectory and
>     belongs to a v11 body revision with a full physics + recipe re-gate.

### 4.5 What NOT to do

* Do **not** implement implicit/semi-implicit Coriolis. It does not fix this bug,
  it costs 30–60 % of the substep, and it retires the symmetric-PD/no-pivot
  Cholesky invariant that the tri-tier bitwise parity method depends on.
* Do **not** raise COMPOSITE_EPS as a quick fix. It works (339.7 m → 0.275 m) but
  it silently reclassifies the ankles on 88 % of healthy substeps.
* Do **not** ship B. Its bound leaks by construction (measured 0.25–0.47 m against
  a 0.2 m bound, 29–33 % of over-integrations invisible to it), and the
  unconditional form is variant A, which is a science change.

---

## 5. Open items

1. **Why is the realized loop gain only 2–5 when `‖Q_j‖_M` is 90–116?** The other
   stages partially undo each cap application; the composition's spectral radius
   was not measured directly. A per-substep measurement of the full 4-stage
   iteration operator's spectral radius would turn the "engaged ankle at
   `r ∈ [1.1e-4, 1.6e-4]`" trigger into a sharp, checkable predicate — which is
   what a proper stability gate (and any future adaptive guard) would key on.
2. **E4 evaluation** (plain `M⁻¹e_j` for the step-8 cap/limit rows) needs the
   Rust weld-drift gate suite to be a fair test — it is the theoretically correct
   fix and it would make the solver *faster*. Scope it for v11.
3. Whether the four ankle joints should have a *different* constraint treatment
   from the rest (they are the only joints whose motion is directly the welded
   body's motion) is the design question behind both E4 and E5.
