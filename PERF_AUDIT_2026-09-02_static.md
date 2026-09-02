# Static optimization audit — 2026-09-02

Three independent static-analysis passes over the training stack (no
code was run; every number is derived from the source plus the
measurements already on record in `PERF_AUDIT_2026-08-16.md`, the
in-code comment history, and the 2026-08-31 session logs). Production
shape assumed throughout: N=32768 × T=32, minibatch 16384,
`--minibatches 64 --mb-accum 2`, `--fp16 all`, adapool critic, RTX 5090.

Angles: **A** update phase (fwd/bwd/optimizer), **B** rollout phase
(env kernel / inference / buffers / GAE), **C** PPO algorithmics
(estimators, optimization dynamics, architecture levers).

One arithmetic correction that matters for everything below: with
`--minibatches 64`, `opt.step()` runs once per minibatch
(`train_batch.py` minibatch loop), so the unfiltered update is **64
optimizer steps**, not 32; `--mb-accum 2` only splits each backward.
`--adv-filter 0.5` makes it 32.

---

## Synthesis — what to do, in order

### Tier 0: measure or log before anything else (zero science risk)

| # | item | why first | source |
|---|------|-----------|--------|
| 0.1 | Check `BEAT_ENV_ORDER=1` is exported on the production pod; promote to a trainer flag defaulting ON and print it on the banner | +5.4% steps/s measured at exactly this shape (2026-08-21), bitwise identical, gated by an env var no launcher sets | B §1 |
| 0.2 | `train_batch --bench` at production shape, `--mb-accum 1` vs `2` | accum 2 was adopted for graph-pool OOMs that no longer exist; it doubles the per-minibatch enqueue. Decides whether the update is CPU-enqueue-bound, which gates A §4/§6/§7/§10 | A §2 |
| 0.3 | Log clipfrac + approx-KL (first and last minibatch), pre-clip grad norm / `clip_coef`, per-group and min-head entropy, and a full-batch entropy under the filter | The joint 29-head ratio leaves ±0.2 at a per-head change of 0.63%; nothing measures how much of each update is zero-gradient. `total_norm` is computed and discarded. Under the filter the printed `ent` is the kept-row entropy and is not comparable to unfiltered runs | C §1.2, §2.2, §5 |

### Tier 1: confirmed-by-code defects and free wins

| # | item | expected | risk | source |
|---|------|----------|------|--------|
| 1.1 | **The critic runs in fp32 through stock torch.** `policy.py:481` adds the fp32 `ffn.2.bias` Parameter to the fp16 residual stream → `x_res` promotes to fp32 → fp32 LN, stock `F.linear` (non-tensor-core), two hidden 113 MB `reshape` copies in the pooling matmuls, fp32 `critic_mlp`. Fix: reuse the final block's already-resolved fp16 `x` (bit-consistent with what its own k/v saw), SDPA with Tq=1 for the pool, pack critic k/v into one GEMM | ~8–12% of update GPU time (~3–4% wall), ~1.5% of rollout, **−1.5 to −1.8 GiB peak** | value changes at fp16-rounding level (class already accepted for the actor); `r1` unaffected | A §1, §9; B §7 |
| 1.2 | `del p_obs, p_act, p_logp, p_adv, p_ret, perm` after the epoch loop — the 1.25 GB permuted obs copy and 243 MB action copy currently live through the entire next rollout | **−1.5 GiB peak** at filter 1.0 (−0.75 at 0.5 — this exactly explains the −0.70 GiB the filter measured) | none, bit-identical | A §3 |
| 1.3 | `b_act` int64 → uint8 (gears ≤ 7), `.long()` per micro-batch | −213 MB resident, −213 MB on the perm copy | none, integers | A §3, B §5 |
| 1.4 | `--adv-filter` × per-minibatch advantage normalization: the kept set `{|A| ≥ median|A|}` has σ ≈ 1.36σ_full, so normalized advantages shrink 0.73× relative to entropy and value terms — implicitly `ent_coef ×1.36`, value-gradient share into the shared trunk ×2.2. Fix: compute μ/σ once per update over the full `f_adv` and use them when filtering (bit-identical when off) | removes a confound from the 0.5 probe's "parity" reading; 3 lines | none when filter off | C §1.1 |
| 1.5 | `relu_bwd`: `at::zeros` → `at::empty` (`fp16_gemm.cu:821`); every column is written | ~900 dead memsets/update | none | A §8 |

### Tier 2: real levers, each gated on a Tier-0 measurement

| # | item | ceiling | gate | source |
|---|------|---------|------|--------|
| 2.1 | Per-head PPO clip (or wider joint ε) | unknown — potentially the largest hidden loss of effective batch in late minibatches | clipfrac on the last minibatch > ~0.25 (0.3) | C §1.2 |
| 2.2 | Critic on a side stream so it overlaps `step_kernel` (bit-identical, ~30 lines) | ~0.8% wall alone; it is the go/no-go for 2.3 | do after 1.1 | B §2 |
| 2.3 | Two-chain rollout: N split in half on two streams, env(A) ‖ policy(B) then swap; T stays 32, batch unchanged | 14% wall (whole forward hidden); realistic 4–8% | 2.2 measures > 0; needs nsys timeline | B §3 |
| 2.4 | CUDA-graph the whole update minibatch (2 micro fwd+bwd + loss-scale + clip + capturable Adam). Differs from the retired rollout graph: fixed-shape, grad-enabled, and the enqueue/GPU ratio has changed. All custom kernels are capturable; `cpu_opt_state()` already normalizes `capturable`/tensor lr | 0 if GPU-bound; 8–12% wall if 0.2 says CPU-bound | 0.2 | A §4 |
| 2.5 | cuBLASLt fused bias+ReLU epilogue (`torch._addmm_activation`) — the un-executed half of the 08-16 audit's Tier 1.1 | 3–5% of update | sm_120 Lt kernel runs at plain-GEMM speed (sm_61 didn't) | A §5 |
| 2.6 | T=64 + `--adv-filter 0.5`: 64 steps/update on 1M kept rows — same step count and minibatch size as today's unfiltered recipe, mean captured GAE window 10.1 → 13.5 ticks, ≥50%-mass samples 62% → 81% | sample-efficiency lever; rescale `--ent-tau` 62, ckpt cadence, `VALUE_NORM_BETA` 0.02; b_obs 2.5 GB | Tier 1 VRAM items first | C §1.3 |
| 2.7 | Ragged (CSR) chart pool — 10 fp32 + 9 bool lanes padded to C≈1.5k for every one of ~92k segments ≈ 6.9 GB at ~7% occupancy | **~−6 GB VRAM**, 0 wall, bit-identical by construction | P unverified — read the `[ranked-pool] … MB` log line | B §4 |
| 2.8 | Adaptive-KL trunk lr (rl_games-style ×1.5 / ÷1.5 per update on mean KL) — the principled form of the lr-ladder finding | self-tuning trunk lr; must be checkpointed | 0.3 logging, threshold set from measured joint-29-head KL | C §2.3 |
| 2.9 | `wgrad` with beta=1 straight into the flat-grad views (skip 62 `AccumulateGrad` adds per micro; packed-head dW into the (145,256) flat view in one GEMM) | ~1% update GPU, 2–3% if CPU-bound | 0.2; re-measure (split-K kernel choice) | A §7 |
| 2.10 | ~20 bookkeeping launches per minibatch (`v_tgt` normalization per micro is constant per update; duplicate `found_inf` ops; 10-kernel loss-scale adaptation chain) | 0.3–2% of update depending on 0.2 | 0.2 | A §6 |
| 2.11 | `torch.compile` only the pure-torch loss tail (post-`forward_stacked`); Dynamo cannot trace the custom `autograd.Function`s | 0.5–3% depending on 0.2 | 0.2 | A §10 |

### Cross-checks between the three passes

- B's roofline estimate (forward ~65–70% of roofline) was made before A found the fp32 critic; ~3 ms of the 26.7 ms `bench_policy_forward` figure is that promotion, so the actor path is closer to roofline than B credited and the critic is the slack. Do A §1 before B §2/§3, and re-baseline the forward afterwards.
- A and B independently arrived at `b_act` uint8 and at packing critic k/v.
- C's §1.1 (filter normalization) and A's §3 (perm copies) both explain parts of the 2026-08-31 probe numbers: the −0.70 GiB is the half-size perm copy, not an update-activation saving; the "parity" was measured with an implicit +36% entropy pressure.
- Explicitly **not** worth doing, all three agree: fusing/parallelizing the GAE scan (<0.5 ms/update), the sampling path, obs dtype casts, per-tick Python overhead (refuted by the graph measurement), value clipping, per-batch vs per-minibatch normalization without the filter (n=16384 → 0.55% σ error), the env kernel internals (the 08-16 do-not-retry list stands).

### Diagnostics worth adding (all fit in the once-per-update print sync)

Approx-KL and clipfrac (first/last mb); `clip_coef` mean; per-group and min-head entropy; full-batch entropy under the filter; bootstrap calibration (`b_val[T]` from update u vs realized 32-tick return from u+1 — the non-structural critic-quality metric EV cannot give); critic skill vs a note-counter baseline; kept/full advantage σ ratio when filtering; diag on kept rows; per-position advantage variance. (C §5)

### Architecture notes (C §3)

- `trunk = Linear, ReLU, Linear` then Linear heads: the actor's readout ends in two linear layers in a row, so `trunk.2` adds optimization geometry but no capacity, and `dormant` measured on its signed linear output is structurally ~0 — the instrument is placed where it cannot read. Measure dormancy on `trunk.0`'s ReLU output or FFN hiddens; a ReLU after `trunk.2` is a probe, not a patch.
- Action-head init and temperature: not a lever (entropy 42 of 46.7 max at start; CleanRL's 0.01 head gain is irrelevant after ~50 updates).
- Critic head is sound; shared-trunk interference is not the bottleneck (gradcos ≈ 0, |g_v| ≪ |g_pi|) — until the filter's ×2.2 value share (1.4).

---

## Pass A — update phase (full report)

Scope: `python/train_batch.py` main() from the flatten (L1046) through `opt.step()` (L1277), `policy.py` forward/critic, `fp16_linear.py`, `fused_ln.py`, `csrc/layer_norm.cu`, `csrc/fp16_gemm.cu`, `bench_policy_update.py`, `PERF_AUDIT_2026-08-16.md`, and the 2026-08-31 session note. Static only; every number below is derived from bytes/launch counts in the code and labelled as an estimate.

### Baseline census

Per micro-batch (B=8192, fp16 stream, grad enabled), counted from the code paths actually taken:

- Forward ≈ 125 launches: obs split + 3 encoder MLPs ≈ 14; attn_bias 3 (L464–467); block 0/1 ≈ 11 each (ln, packed qkv addmm, SDPA + mask pad, att reshape copy, out_proj, ln_fwd_add, ffn mm + `bias_relu_` + nobias mm); `x_res` 2; final block ≈ 12; trunk 3; packed heads 1 + `.float()` 1; **critic ≈ 20** (see §1); log_softmax/logprob/entropy ≈ 7; loss + diagnostics ≈ 30 (L1177–1205).
- Backward ≈ 220 (autograd ≈ 1.7× fwd, plus **138 `AccumulateGrad` in-place adds** — one per parameter tensor, including 58 for the 29 head weight/bias slices returned by `_Fp16PackedLinear.backward`, L316–324 — plus 14 `pdw/pdb.sum(0)` for the 7 LNs and 7 `part.sum(0)` for relu sites).
- Per minibatch: 2 micro × ~345 + ~35 fixed (zero 1, loss-scale bookkeeping ≈ 24, clip 5, fused Adam 4, replica refresh 1) ≈ **725 launches**; ≈ **46k launches/update** at 64 minibatches.

Saved-activation footprint per micro-batch ≈ 1.85 GB actor (fp16; the (B·27, 512) ffn hidden is 226 MB per block) + ≈ 0.45 GB critic (fp32, see §1) ≈ 2.3 GB.

The prior census (08-16 audit Tier 2.1, 4090, **mb 16384 / accum 1**) measured CPU enqueue 30 ms vs GPU 48 ms per minibatch. Production now runs **accum 2** on a 5090 whose GPU time is ~40 ms/minibatch (39.7 ms measured in `bench_policy_update`, which has no accum). Accum 2 roughly doubles the fwd/bwd enqueue per minibatch (~55 ms if the 4090-pod CPU figure transfers) against ~40 ms GPU → the update may now be **CPU-enqueue-bound**. This can't be settled statically; it changes the payoff of half the items below, so §2 is the first measurement to make.

### A §1. The critic runs in fp32 through stock torch — unintended dtype promotion

**Anchor:** `policy.py:481-482`

```python
x_res = (x_res + delta if delta_b is None
         else x_res + delta + delta_b)
```

`delta_b` is `ffn.2.bias`, an **fp32 (d,) Parameter** (the fold-tail path, `_seq(..., fold_tail=True)` L88–94, default on). A dimensioned fp16 tensor + a dimensioned fp32 tensor promotes to **fp32** (only 0-dim tensors/Python scalars are lower priority). So on the default fp16 stream `x_res` is fp32 (B,27,128), and everything downstream follows:

- `critic_ln(x_res)` (L535) → `FusedLayerNorm.forward` takes the **fp32** `ln_fwd_kernel` (the deliberately non-register-cached, non-specialised one, `layer_norm.cu:54`), output fp32.
- `_lin(critic_q/k/v/out, hv)` (L536–538, 546) → `_lin` dispatches on `x.dtype is torch.float16` (L73) → **stock `nn.Linear` / `F.linear` in fp32**, with stock autograd backward (fp32 dX GEMM, fp32 dW GEMM, `sum(0)` db). No TF32 anywhere in the repo, so these are full-rate FP32 SGEMMs, not tensor-core.
- `torch.cat([a, h])` (L547) promotes `h` (fp16 trunk readout) to fp32 → `critic_mlp` runs the stock fp32 path too (`_run` L106 only takes the `bias_relu` epilogue for fp16).
- `(q*scale) @ k.transpose(-2,-1)` and `softmax @ v` (L543–545): 4-D `matmul` with batch dims (B, 4) whose strides (3456, 32) are not collapsible → `reshape` **copies k and v** (113 MB fp32 each, forward and again in backward). The actor blocks avoid this because SDPA accepts the strided 4-D layout directly.

The comments say "the critic pays it once, one (B,27,d) add" (L475–478) and `val.float()` (L548) expects an fp16 value — this is almost certainly unintended. The critic-design topic never declares an fp32 critic; `git log -S x_res` shows it landed with the adapool commit (8d36d47) untouched since.

**Estimated cost (per micro-batch, B=8192):** forward ≈ 1.2 GB traffic (two adds 340 MB, LN 226 MB, k/v GEMMs 450 MB, two bmm copies 450 MB), backward ≈ 2.5 GB (k/v dX+dW+db ≈ 1.1 GB, grad accumulation of dhv from q/k/v incl. a full-size zero-fill from the `hv[:, 2:3]` slice backward ≈ 450 MB, fp32 ln_bwd 340 MB, the add-backward cast + reductions ≈ 400 MB). At B=16384 (the bench shape) ≈ 7.4 GB → ~5 ms at ~1.5 TB/s effective, + ~1 ms of full-fp32 GEMM compute, + inefficient m=1 batched GEMMs. **≈ 6–7 ms of the measured 39.7 ms/minibatch (~15%).** On the fp16 stream ≈ 2–2.5 ms.

**Expected gain:** ~8–12% of the update phase (≈ 3–4% wall), plus a rollout-side bonus: the same promotion runs in the no-grad rollout forward at N=32768 (≈ 4.8 GB extra traffic ≈ 3 ms of the 26.7 ms `bench_policy_forward` figure, ~1.5% of the rollout phase). **VRAM:** in the rollout at N=32768 the fp32 critic transients (x_res, hv, k, v, k-copy, v-copy ≈ 6 × 453 MB ≈ 2.7 GB live) drop to ≈ 0.9 GB; in the update ≈ −0.25 GB per micro-batch of saved tensors. If the 13.78 GiB peak lands in the rollout (likely — see suspicious list), this is −1.5 to −1.8 GiB.

**Fix (preferred):** don't recompute `x_res` at all. The final block's `ln_attn.forward_add(x, delta, delta_bias)` (L249) already writes the resolved fp16 residual stream `x = round_h(a + (b + bb))` — bit-identical to what its own k/v see — before the `x[:, :n_queries]` slice at L283. Return/stash that `x` from the final block and feed it to `_critic`. This deletes 2 kernels (340 MB), makes the critic's input *more* consistent with the actor, and the whole critic then flows through the existing fp16 machinery: fp16 `ln_fwd_h_kernel<2>`, `fp16_linear` for q/k/v/out (custom `wgrad`), `_run`'s `bias_relu` epilogue for `critic_mlp`, `relu_bwd` for its two ReLUs (d=256 is in `_RELU_BWD_DIMS`). Replace the two `matmul`s with `F.scaled_dot_product_attention(q, k, v, attn_mask=attn_bias.unsqueeze(1))` (Tq=1). Autograd: the critic's gradient on `x` then enters `_LnAddFn.backward` as `gx`, which is already the fused path (`fused_ln.py:563-578`).

**Numerics/repro risk:** the value prediction changes at the fp16-rounding level. `r1` is unaffected (value isn't in the ratio). Gate: the 50-update smoke reading EV (0.98 at upd 776 on the af50 probe is the reference); `tests/test_policy.py:59-63` (atol 1e-5 on `v`) is a same-mode consistency test and should still pass. **Effort:** ~half a day + one pin test asserting `x_res.dtype is torch.float16` on the fp16 stream. **Verify:** `bench_policy_update.py --fp16 --mbs 8192 16384` (ms and peak MB) and `bench_policy_forward.py --ns 32768` before/after; `train_batch --bench` at production shape for the wall number.

### A §2. `--mb-accum 2`: its justification was removed with `--graph-rollout` — re-test accum 1 (zero code)

**Anchor:** `v27_critic_probe.py:6-9` — accum 2 was adopted because "plain N=32k OOMs on **graph pools** + the env's non-torch state", with `expandable_segments`. The graphed rollout and its private pools were deleted 2026-08-31. Every current driver (`v27`–`v30_*_probe.py:51`) still carries `--mb-accum 2` verbatim.

**Mechanism:** accum 2 doubles fwd/bwd launches per minibatch (~690 vs ~345), halves GEMM M (221k → 110k rows), and doubles the CPU enqueue per optimizer step — the ratio that decides whether §4 is worth anything. Accum 1 is never slower on the GPU side; the only price is VRAM: +≈ 2.3 GB of saved activations (13.78 → ~16 GiB on 32 GB — fits, and §1/§3 buy that back).

**Expected gain:** 0–3% of the update if it is GPU-bound; **10–30%** if the accum-2 enqueue exceeds GPU time. `help` text calls accum "same math (…modulo fp association)", so accum 1 vs 2 is the same near-tie class already accepted.

**Verify (this also gates §4, §6, §7):** `train_batch --bench` at production shape with `--mb-accum 1` vs `2` (exec-true split); `bench_policy_update.py --fp16 --mbs 8192 16384` — if 2·t(8192) ≈ t(16384) the GPU side of micro-batching is free and any production gap is CPU. Add a `--mb-accum` knob to `bench_policy_update.py` for the isolated number.

### A §3. VRAM: the permuted copies outlive the update; per-micro gathers; int64 actions

**Anchors:** `train_batch.py:1157-1158` (`p_obs, p_act, p_logp, p_adv, p_ret = f_*[perm]`), L786 (`b_act` int64), L1279–1283 (end of update — no `del`).

- `p_obs` (1,048,576 × 598 fp16 = **1.25 GB**) and `p_act` (× 29 int64 = **243 MB**) are Python locals assigned inside the epoch loop and **stay allocated through the entire next rollout and GAE** until reassigned on the next update. That matches the adv-filter measurement exactly: at 0.5, `train_n` halves → p_obs+p_act shrink by 0.63 + 0.12 GB = 0.75 GB ≈ the −0.70 GiB observed. So the peak is (or is within ~1 GB of) the rollout, and these buffers sit under it doing nothing. `del p_obs, p_act, p_logp, p_adv, p_ret, perm` after the epoch loop: **−1.5 GiB peak at filter 1.0, −0.75 at 0.5**, zero risk, bit-identical.
- Going further: gather per micro-batch instead of per epoch — `torch.index_select(f_obs, 0, perm[sl], out=static_obs)` (and likewise act). Same total bytes moved, identical rows in identical order → bit-identical. Costs +2 launches per micro, removes the 1.5 GB resident copies entirely, and provides the static input buffers §4 needs. Differs from the retired host-side shuffle (device `randperm`, device `index_select`, no H2D).
- `b_act` is int64 for values in [0, 5). Store uint8 and `.long()` per micro: **−213 MB** resident, −213 MB on the perm copy. Bit-identical.

Wall-clock: ~0. VRAM: −1.5 to −2 GiB combined. Effort: 1 hour.

### A §4. CUDA-graph the whole minibatch — gated on §2

How this differs from what was retired: the rollout graph captured a no-grad forward at N=32768 — one large launch stream that was already GPU-bound. The 08-16 audit Tier 2.1 refuted graphing the update at **mb 16384 / accum 1 on a 4090** (CPU 63% busy). The case here rests on two things that changed since: accum 2 doubled the enqueue per step, and the 5090 cut the GPU time ~20%. If §2 shows the update is GPU-bound at the chosen accum, **skip this**.

Feasibility with the custom stack (all checked statically):
- `ln_fwd/ln_fwd_add/ln_bwd`, `bias_relu_`, `relu_bwd`: plain stream launches on `getCurrentCUDAStream()`, capturable.
- `wgrad` uses `at::cuda::getCurrentCUDABlasHandle()` (`fp16_gemm.cu:860`) — torch's own handle with its per-stream workspace (warm up before capture).
- SDPA mem-efficient backward is capturable.
- Fused Adam: needs `capturable=True` and tensor `lr` — the trainer sets `g["lr"]` as a Python float every update (L914–915); make it a per-group 0-dim device tensor updated by `.fill_()`. `cpu_opt_state()` (L432–446) already normalises `capturable`/`lr` for checkpoints.
- Loss-scale state is already fully device-side (L1217–1254); `_amp_foreach_non_finite_check_and_unscale_` is graph-safe.
- `flat_refresh()` (`fp16_linear.py:93-98`) is version-keyed in Python: capture right after a `bump_weights_version()` so the `copy_` is inside the graph.
- `ratio_diag` (L1188–1193) is assigned only on the first minibatch — compute it every replay into a static tensor.
- Inputs: static micro-batch buffers filled by `index_select(out=)` (§3).
- Autograd accumulation into the pre-existing flat-grad views is in-place (the `flat_guard` at L1256–1266 proves it); run the guard before capture.

Launch count per minibatch goes from ~725 to ~3. **Gain: 0 if GPU-bound; plausibly 20–30% of the update (≈ 8–12% wall) in the accum-2 CPU-bound scenario.** Numerics: bit-identical replay. Effort: 2–3 days + tests. Verify: `train_batch --bench` exec-true split; GPU-busy % in the update via `nsys profile --stats=true` or `torch.profiler`.

### A §5. cuBLASLt fused bias+ReLU epilogue — the un-executed half of the 08-16 Tier 1.1

**Anchor:** `fp16_linear.py:416-422` (`torch.mm` bias-free, then `_load().bias_relu_(y, b)` — a second full pass over the output). The split was justified by sm_61 addmm-epilogue slowness. The width fix was done 2026-08-15; the Lt re-probe is not in the A/B ladder.

**Mechanism:** `torch._addmm_activation(b16, x2, wh.t(), use_gelu=False)` routes to cublasLt's `RELU_BIAS` epilogue with contiguous fp16 inputs and a 1-D bias. Removes the `bias_relu_` pass: per micro-batch ≈ 1.5 GB less traffic ≈ **~1 ms/micro ≈ 3–5% of the update** if the Lt kernel runs at plain-GEMM speed on sm_120.

**Numerics:** one rounding instead of two, bias enters as fp16 — fp16 last-rounding class. Backward unchanged. Effort: half a day behind a `BEAT_LT_EPILOGUE=1` toggle. Verify: `bench_policy_update --fp16 --mbs 16384` A/B, then the grad-probe rel-error gate.

### A §6. Per-minibatch bookkeeping: ~20 pure-overhead launches per minibatch

All in the L1159–1278 loop; payoff depends on §2 (≈ 0.3% of the update if GPU-bound, ≈ 1.5–2% if CPU-bound). All bit-identical:

- L1245–1246 recompute `found_inf[0].clamp(max=1.0)` and `1.0 - overflow` that L1223 already computed as `ok` (−2).
- L1217 `inv_scale.copy_(fp16_scale.reciprocal())` → `torch.reciprocal(fp16_scale, out=inv_scale)` (−1).
- L1197 `v_tgt = (p_ret[sl] - ret_mu) / ret_sig` is constant per update: normalise `f_ret` once instead of 2 × 128 (−254/update).
- L1194 `(a_sd + 1e-8)` per micro → per minibatch.
- L1185–1187 `logratio.abs().max()` + `maximum(out=)`: `torch.linalg.vector_norm(logratio, ord=inf)` is 1 launch.
- L1248–1252 scale-adaptation chain (≈ 10 kernels): 3–4 with `torch.where` on a stacked pair.

### A §7. `wgrad` with beta=1 straight into the flat-grad views

**Anchors:** `fp16_gemm.cu:858-859` (`c = at::empty(...)`, `beta = 0`), `fp16_linear.py:313-317, 364, 396, 443`, `train_batch.py:750-757` (every `p.grad` is a flat view).

Every weight grad is produced as a fresh fp32 temporary and then added into the flat view by `AccumulateGrad` — 62 weight tensors per micro. Passing `beta=1` with `c = p.grad` (the head weights are laid out consecutively in `flatten_fp16_replicas`, so the packed head dW can accumulate into the (145, 256) flat view in **one** GEMM) and returning `None` removes 62 add kernels + 62 temporaries per micro ≈ 8k launches/update. **Gain ≈ 1% of the update GPU time; 2–3% if CPU-bound.** Differs from the retired packed stack: autograd still owns the graph; only weight-grad accumulation moves into the GEMM epilogue. Numerics: last-rounding class; cuBLAS may pick a split-K kernel with beta≠0, so re-measure. The diag block's two `retain_graph` backwards rely on accumulation into `flat_grad` — beta=1 preserves that.

### A §8. `relu_bwd`: `at::zeros` → `at::empty`

**Anchor:** `fp16_gemm.cu:821`. Every block writes all `d` columns of its `part` row, so the memset is dead. 7 relu sites × 128 micro ≈ 900 memset launches/update. `ln_bwd` already uses `at::empty` (L583–584).

### A §9. Once the critic is on the fp16 stream: pack critic k/v, SDPA for the pool

`critic_k`/`critic_v` share input `hv` (L537–538) — the qkv-group machinery turns them into one GEMM (N=256) + one `wgrad` + one bias sum. `critic_q` runs on the single body row, so it stays separate. SDPA with Tq=1 replaces the two batched matmuls and their hidden copies. Together ≈ −8 launches and ≈ −0.5 GB traffic per micro on top of §1 — ~1–2% of the update. Only sensible as part of §1.

### A §10. torch.compile: only the pure-torch loss tail is viable

Dynamo cannot trace through `_Fp16Linear*`, `_LnFn`, `_LnAddFn` (custom `autograd.Function`s wrapping extension calls) without custom-op registration. `mode="reduce-overhead"` is CUDA graphs, i.e. §4 with less control. Viable: compile `(logits, action, old_logp, adv, a_mu, a_sd, newval, v_tgt, ent_coef_t, scale_t) → loss, entropy_mean, logr_max`: ~30 small fwd kernels + ~40 bwd on (8192, 29, 5) tensors fuse to ~6–8. Pass `ent_now`/scale as tensors. **Gain ≈ 0.5–1% if GPU-bound, 2–3% if CPU-bound.**

### A — suspicious but not confirmable statically

1. **Update CPU-bound at accum 2?** Every update-phase number in the comments (39.70 ms × 64) is at accum 1; production is accum 2.
2. **Which SDPA backend actually runs** (`policy.py:306-307`): fp16, dh=32, T=27, Tq=3, additive mask (B,1,1,27). A silent fall-through to the math path would materialise (B,4,Tq,27) scores. Check with `sdpa_kernel([SDPBackend.EFFICIENT_ATTENTION])` around one forward.
3. **Hidden copies in the critic's 4-D `matmul`s** — derived from stride collapsibility; `torch.profiler` on one minibatch shows `copy_` adjacent to `bmm`.
4. **Where the 13.78 GiB peak lands.** `max_memory_allocated` excludes the env kernel's ~1.4 GB of driver-managed local memory and cuBLAS/SDPA workspaces. `torch.cuda.memory._record_memory_history()` over one update would settle it.
5. **`torch._foreach_mul_(grads, ok)`** (L1224) with a 0-dim CUDA tensor: on older builds the scalar path would `.item()` — a hidden sync per minibatch.
6. **`x[:, :n_queries].contiguous()` backward** (`policy.py:283`): SliceBackward zero-fills a full (B,27,128) fp16 tensor per micro. ~0.1 ms/micro.
7. The `.half()` at `policy.py:440` is a no-op on fp16 obs, so the comment "the cast IS the copy" is stale — the copy happens later, once, in the Function.

### A — already well-optimised

No host syncs in the minibatch loop (verified line by line); flat grad buffer with single-tensor unscale/check/clip; fused Adam over 2 flat groups; one replica refresh per step, version-keyed; packed qkv and packed 29-head GEMMs; fused residual-add + LN; ffn.2 bias fold; register-cached fp16 LN at d=128; fused `relu_bwd`; custom `wgrad`; final-block query pruning; block-invariant `attn_bias` hoist; device-side hoisted GAE; fp16 obs buffer; adv-filter composed lazily into the shuffle index.

---

## Pass B — rollout phase (full report)

### B §0. Where the rollout tick goes (derived)

Per tick at N=32768 on the 5090, using the repo's own numbers (`train_batch.py:960-965`: env kernel ~43% of a 5.96 s update = 2.56 s / 32 ticks; `bench_policy_forward --ns 32768` = 26.72 ms):

| item | ms/tick | share of tick | share of wall |
|---|---|---|---|
| `step_kernel` (one launch, 8 substeps) `beat_arena_cuda.cu:5983-5990` | ~80 | ~75% | ~43% |
| policy forward (`policy.py:432-520` + `_critic` 522-548) | ~26.7 | ~25% | ~14% |
| sampling: log_softmax, exponential_, log, sub, argmax, gather, sum (`policy.py:622-634`) | ~0.1 | <0.1% | ~0 |
| buffer writes: `b_obs[t]=obs` (78 MB fp32 read + 39 MB fp16 write), `b_act` 7.6 MB, four 131 KB vectors (`train_batch.py:929-938`) | ~0.1 | <0.1% | ~0 |
| `_split` gathers: 2 `index_select` over 64 lanes + `notes.half()` (`policy.py:424-440`) | ~0.15 | <0.2% | ~0 |
| env wrapper: `randint` cand, optional `_kick`, `_pose_jit.uniform_`, optional argsort (`cuda_env.py:660-725`) | <0.1 | — | ~0 |

Total ≈ 107 ms/tick; 32 ticks ≈ 3.43 s ≈ 57% of 5.96 s. Everything outside the two big kernels is well under 1% of the tick. GAE is 64 launches on 131 KB tensors plus ~8 elementwise passes over 4 MB tensors: well under 0.5 ms per update. Roughly ~100 launches per tick; CPU enqueue is ~20-30x faster than GPU execution here, which is exactly why the whole-rollout graph measured 0.99x.

Policy forward roofline: 884k tokens × ~590k MACs/token ≈ 1.0 TFLOP, final block n_queries=3 so ≈ 0.8 TFLOP → ~8 ms of compute at ~105 TFLOPS. Activation traffic ≈ 14-15 GB ≈ 8 ms at 1.8 TB/s. Floor ≈ 16-18 ms vs 26.7 measured: ~65-70% of roofline (before accounting for Pass A's fp32-critic finding, which is ~3 ms of the gap). The only way to make the forward "free" is to run it concurrently with the env kernel (§2, §3).

Obs path: the kernel writes obs directly into `venv._obs` (fp32, `cuda_env.py:615`, `beat_arena_cuda.cu:4903-4905`), NOT into `b_obs[t]`; the trainer copies it with an fp32→fp16 cast (`train_batch.py:929`). The obs buffer is already fp16 under `--fp16 all` (1.25 GB). The critic runs at full width every tick and must: `b_val[t]` feeds GAE. It costs roughly 5-8% of the forward.

### B §1. Confirm `BEAT_ENV_ORDER=1` is actually exported on the production pod

- Anchors: `cuda_env.py:608`, `cuda_env.py:713-724` (dynamics-key argsort), `beat_arena_cuda.cu:5772`.
- The R12 dynamics-key permutation measured +5.4% end-to-end at exactly this shape on the 5090 (08-16 audit addendum, 2026-08-21), is per-env bitwise identical (`tests/test_env_order_bitwise.py`), and is default OFF behind an env var. The v30 campaign launcher (`v30_package_probe.py:50-62`) does not set it; the v34 trunk note says it was exported by hand.
- Change: promote to a trainer flag defaulting ON (or set `os.environ` in the launchers) and print its state on the `=== batched PPO` line.
- Expected gain: 0% if already on; +5.4% steps/s if not. Risk: none. Effort: 10 lines. Verify: `--bench` with/without; 123,081 vs 116,804 from 2026-08-21 is the reference.

### B §2. Run the critic on a side stream so it overlaps the env kernel (bit-identical probe)

- Anchors: `policy.py:519-520` (critic called inline, after heads), `train_batch.py:930-936`, `cuda_env.py:725` (`_launch`).
- Mechanism: only the actor path is on the critical path to `venv.step`. The critic's inputs exist before the heads GEMM; its output is not needed until GAE. On a side stream the critic's GEMMs + LN + softmax run concurrently with `step_kernel`.
- Why concurrency should help here: `step_kernel` is latency-bound and parallelism-starved (08-16 Tier 4: 4-6 warps/SM, ~0.1% FLOP utilization). At N=32768: 1024 blocks × 32 threads, 254 regs → ~6 blocks/SM = 192 threads/SM, tensor cores idle, and a straggler tail (thread-per-env with divergent contact counts). A concurrent kernel fills the tail and idle issue slots. This is a GPU-side concurrency change, not a launch-overhead change, so the `--graph-rollout` 0.99x verdict does not apply.
- Expected gain: ceiling ~1.5-2 ms/tick ≈ 0.8% wall. Its real value is as the cheapest go/no-go for §3. If this measures ~0, do not build §3.
- Risk: bit-identical (same kernels, same batch; actions unchanged). Watch stream semantics (`record_stream` / keep the join inside the tick). Register pressure may prevent GEMM CTAs from co-residing with 6 env blocks/SM — then only the tail overlaps.
- Effort: ~30 lines. Verify: `--bench` A/B; `perf_profile.py --phase env` (kernel time sum unchanged, wall down); gate; r1 stays 0.

### B §3. Two-chain rollout: split N into two N/2 halves on two streams, alternate policy/env

- Anchors: `train_batch.py:928-947`, `cuda_env.py:349-660` (env instance = per-N state; the pool dict is shareable), `beat_arena_cuda.cu:5983-5990`.
- Two `CudaBeatVecEnv(N/2, pool=pool)` instances on two streams; per tick, chain A's env kernel runs while chain B's policy forward runs, then swap. Same `(T,N,...)` buffers written at `[:N/2]` / `[N/2:]`; GAE and the update unchanged; T stays 32 per chain.
- How it differs from retired ideas: not `--graph-rollout`; not stale-policy pipelining (each chain strictly on-policy). `design/WARP_RESCUE_LEGACY_SIDE.md:279-287` capped this at ~7% of wall when step_kernel was 90% of the rollout on the 4090 at N=16k. On the 5090 at N=32768 the forward is ~25% of the rollout (14% of wall) — the hideable slice doubled.
- Expected gain: ceiling 14% wall. Realistic 4-8%, only if both halves' env launches overlap each other (512+512 blocks fit in one 1360-slot wave). If they serialize, the N-scaling curve (2x envs = 1.56x time) makes two half launches cost ~1.28x one full launch and the whole thing loses. Gate on §2.
- Risk: not bit-reproducible vs today (cuBLAS may pick different kernels at batch N/2). Run-to-run deterministic. `pop_*` counters merged across two instances (every pop returns a sum). Effort: ~150 lines, one day plus measurement.
- Verify: `--bench`; an nsys timeline (does not need the locked counters that killed ncu on RunPod); then a 50M learning gate.

### B §4. Ragged (CSR) chart pool: ~6 GB VRAM, zero wall change, bit-identical

- Anchors: `cuda_env.py:161-244` (`_charts_to_tensors`, every field padded to (P, C)), `ranked_pool.py:52` (`MAX_OBJECTS = 1536`), `ranked_pool.py:233-247`, kernel index sites `beat_arena_cuda.cu:5015, 5074` (`cb = cc * C + i`), walls at `5031, 5045, 5633`, status stride at `4840, 5016` (`e * C + i`).
- Sizes: per (chart, slot) 10 fp32 lanes + 9 bools = 49 B. With `--chart-pool 999999 --mirror-augment` P ≈ 2 × 46k ≈ 92k segments and C ≈ 1.5k → ~141M slots ≈ 6.9 GB, of which the ~100-note median segment uses ~7%. Likely the single largest tensor in the peak. P unverified statically; the `[ranked-pool] ... MB` line settles it.
- Change: store notes flat with a per-chart `p_off[P]` and read `cb = p_off[cc] + i`; keep `status`/`wall_passed` at their per-env strides. Values, arithmetic, scan order untouched: bit-identical by construction.
- Expected gain: ~−6 GB VRAM, 0 wall (the residency probe showed step time flat across pool sizes 2 MB..7 GB). Value: headroom for N=65536 or larger pools.
- Risk: low; parity suite + a pool-equality test. Effort: half a day (kernel indexing at ~8 sites, pool builder, npz cache key bump).

### B §5. `b_act` as uint8: −213 MB VRAM, free

`train_batch.py:786`: `b_act` int64 (T,N,29) = 243 MB at T=32. gears ≤ 7 fit in uint8. Bit-exact. 3 lines.

### B §6. Collapse the per-update telemetry syncs (hygiene, ~0.02% wall)

`train_batch.py:1284-1360` → `cuda_env.py:787-901`: ~19 host syncs per update after the update, where the queue is already short. A single `torch.stack([...]).tolist()` is cleaner but will not move steps/s.

### B §7. Pack `critic_k`/`critic_v` into one GEMM (~0.1% wall)

`policy.py:536-538` — three separate `_lin` calls over `hv`; k and v read the full tensor twice. The actor blocks already do this. Fold into §2 / Pass A §1.

### B — not worth doing

GAE fusion/parallel scan (<0.5 ms/update); sampling path (~0.1 ms/tick); obs copies/casts (~0.3 ms/tick); Python/launch overhead in the T-loop (refuted by the graph measurement); value-norm EMA; the value head at full width (required by GAE); bootstrap forward reuse (algorithm change for <0.5%); env kernel internals (08-16 do-not-retry list stands).

### B — suspicious but not confirmable statically

Whether `BEAT_ENV_ORDER=1` is in the pod environment; actual pool VRAM (read the log line); whether the 26.72 ms forward figure was taken with `--fp16`; which SDPA backend runs on sm_120 with an additive (B,1,1,27) mask; the env kernel's straggler tail size (nsys timeline of one tick); sector efficiency of the two 64-lane `index_select` gathers; `allow_fp16_reduced_precision_reduction` left at default True (numerics observation).

### B — already well-optimized

fp16 obs buffer with the exact-rounding argument; `b_val` shifted-view GAE with zero allocations; flat fp16 replica mirror with a single refresh kernel; fused-head GEMM; packed qkv; SDPA; fused LN+residual passes; bias+ReLU epilogues; `n_queries=3`; `need_entropy=False`; in-kernel auto-reset; device-side episode-return ring; no host syncs in the tick; dynamics-key env permutation available; telemetry syncs deferred out of the rollout→GAE boundary.

---

## Pass C — PPO algorithmics (full report)

Scope: `train_batch.py`, `policy.py`, the CUDA env's terminal seam (`beat_arena_cuda.cu:5698-5747`), SPEC/README/PERF_AUDIT, and the session logs. Anchors are `train_batch.py:LINE` unless stated.

### C §1.1 Advantage normalization under `--adv-filter` silently re-weights entropy and value loss

**Where.** `a_mu`/`a_sd` at 1166-1167 are computed over the kept minibatch (`p_adv` is already the `keep_idx`-gathered set); `a_mb` at 1194 divides by that. Entropy (1199) and value loss (1198) are not rescaled.

**Mechanism.** The kept set is `{|A| ≥ median|A|}`. For A ~ N(0, σ²), with c = 0.6745σ:
- E[A² | |A|>c] = 1 + c·φ(c)/(1−Φ(c)) = **1.857σ²** → σ_kept = **1.363σ** (Laplace tails give 1.39; robust at ~1.36-1.4).
- E[|A| | |A|>c] = 1.271σ vs unconditional 0.798σ.

Per kept sample the normalized advantage magnitude is 0.933 vs 0.798 unfiltered (+17% per-sample signal), but the PG gradient's scale relative to the entropy and value terms drops by 1/1.363 = **0.73**. Under Adam this is purely a change of direction — equivalent to running the filtered recipe with `ent_coef × 1.36` and `vf_coef × 1.36`. The value term compounds: the critic now trains on the half with the largest residuals (mean |residual| 1.59× the full-batch mean), so the value-gradient share into the shared trunk rises by 1.59/0.73 ≈ **2.2×**. The `[diag]` block runs on unfiltered rows, so it cannot see this.

**Fix.** Normalize by full-batch statistics computed once per update from `f_adv` and use them at 1194 when `adv_filter_on`. Bit-identical when off.

**Probe.** Resume `trunk_upd01000.pt`, 300 updates, two arms differing only in the σ source. Predicted: full-σ arm runs lower `ent` and slightly higher `maxlr`; acc at matched updates is the verdict.

### C §1.2 PPO clip on the joint 29-head ratio is extremely tight per head — clipfrac is not logged

**Where.** `logprob` is the sum over 29 heads (`policy.py:635`); `pg_loss_fn` clips the joint ratio at 1±0.2.

**Mechanism.** The joint ratio leaves the trust window when the geometric-mean per-head ratio exceeds 1.2^(1/29) = **1.0063**. With independent per-head log-ratio perturbations of RMS s, the joint log-ratio has RMS √29·s = 5.4s, so half the batch is outside ±0.18 once s ≈ 0.034. After 64 Adam steps at 7e-4, mid-to-late minibatches likely have a large fraction of adv>0 samples with r > 1.2 and therefore **zero policy gradient**. `maxlr` (2-4) is the max over 1M samples of a 29-head sum and says nothing about the typical sample.

**Fix candidates (contingent on measurement).** (a) Per-head clipping — apply `pg_loss_fn` per head with the same A; standard in multi-discrete implementations but the surrogate is no longer a bound on the joint policy's change. (b) Widen ε for the joint ratio (0.3-0.4).

**Probe.** Step 1: log clipfrac and mean |log r| on the first and last minibatch, 50 updates. Step 2 only if last-mb clipfrac > ~0.25: 500-update A/B, per-head vs joint.

### C §1.3 GAE at T=32: the average credit window is ~10 ticks, not 18

Captured trace mass for a sample at rollout position t = 1−(γλ)^(T−t) with γλ = 0.9457. Half the mass needs T−t ≥ 12.4 ticks; 90% needs ≥ 41.3.

| T | mean captured window (ticks) | samples with ≥50% mass | ≥90% |
|---|---|---|---|
| 8 | 4.0 | 0% | 0% |
| **32** | **10.1** | **62%** | **0%** |
| 64 | 13.5 | 81% | 36% |

A full pre-swing is ~10-15 ticks, so for over a third of the batch the cut reward lands outside the window. The T=8→32 move recovered only 55% of the nominal 1/(1−γλ) window.

**Proposal.** T=64 at N=32768 (`--minibatches 128`, batch 2.1M) with `--adv-filter 0.5`: 64 steps/update on 1M kept rows — identical step count and minibatch size to today's unfiltered recipe. Rescale `--ent-tau` 62, checkpoint cadence, `VALUE_NORM_BETA` 0.01→0.02. `b_obs` doubles to 2.5 GB.

**Probe.** 300 T=64 updates vs 600 T=32 updates from scratch, acc and ep_return at matched samples, falls/1M, maxlr. The ent anneal must be matched in samples.

### C §1.4 Non-issues, checked

Deltas use unnormalized V (consistent to within β = 1%). Time-limit truncation: `max_steps = seg_len·30 + 90 = 840` exceeds the natural end, so the cap essentially never fires; segment ends are genuinely terminal; rollout truncation at T is bootstrapped through `b_val[T]` — correct. Fall/tilt replaces the tick's reward with −fall_const — intentional. `LOG_RATIO_CLAMP 20` is inert (no gradient flows above r = 3 in any branch). Dual-clip c = 3 engages only in the tail. Entropy summed over heads is the right reduction for factored heads. No value clipping — correct. Reward scale is handled by value + advantage normalization. Per-minibatch normalization at n=16384 → σ estimate error 0.55%, equivalent to per-batch. `--mb-accum 2` is gradient-identical to K=1.

### C §2.1 Effective step size: 64 Adam steps at 7e-4 is a hot regime; the drift instrument is wrong

Max drift per update = 64 × 7e-4 = 0.045 vs default-init weight std 0.026-0.051 — drift ceiling ~1-2× initial weight scale per update; realized ~10-50%. The lr ladder put the optimum at 7e-4 with hotter rungs "trained fine but produced worse policies" — the signature of drift. The instrument that would confirm this (mean approx-KL per update, mean |log r| on the last minibatch) is not logged; `r1` measures fp16 precision at the first minibatch only.

### C §2.2 Grad clip at 0.5: whether it binds is computed and discarded

`total_norm` (1270) is never read. Adam is invariant to a constant rescale, so the clip only acts through intermittent binding. Log `clip_coef` mean per update.

### C §2.3 Adaptive-KL lr on the trunk (rl_games-style)

Multiply lr by 1.5 or 1/1.5 per update depending on whether mean KL is below thr/2 or above 2·thr. Self-tunes the trunk lr per phase, never discards rollout data (unlike KL early-stop). Compatible with WSD if the leaf decay is relative to the trunk's live lr. Must be checkpointed. **Do not deploy before §1.2's logging exists** — the threshold has to be set from measured KL under the joint 29-head ratio. Probe: 500 updates from `trunk_upd01000.pt`; abort if lr leaves [3.5e-4, 1.4e-3].

### C §2.4-2.6

Critic lr 3x, vf 0.5 — keep (|g_v| ≪ |g_pi|, gradcos ≈ 0); re-run the diag on `keep_idx` rows when filtering. Warmup — fine. Step-shape probes the filter result motivates: (a) 2 epochs on the top 50% (64 steps; the clip becomes live on epoch 2); (b) §1.3's T=64 + filter. Run (a) only after §1.1's fix.

### C §3 Architecture-level levers

- **§3.1** `trunk = Linear, ReLU, Linear` (`policy.py:372`) then Linear heads: the actor's last nonlinearity is `trunk.0`; `trunk.2` adds no capacity. `dormant` on this signed linear output is structurally ~0 — every log reads `dormant 0.000`. Measure dormancy on `trunk.0`'s ReLU output or FFN hiddens; a ReLU after `trunk.2` is a probe, not a patch.
- **§3.2** Action-head init — not a lever.
- **§3.3** Critic head sound; the filter is the only thing that changes its gradient share. The 5-gear categorical ignores ordinal structure — noted, not a problem.

### C §4 Already good — don't touch

Dual-clip with the log-ratio clamp; value normalization T1; no value clipping; `done` masking of bootstrap and trace; per-minibatch advantage normalization at n=16384; `--mb-accum` as a pure memory lever; 1 epoch × 64 minibatches (density finding stands); WSD with the entropy curve as a function of absolute update index; critic lr split + vf 0.5; fp16 storage with fp32 masters and fp32 PPO math; two-sided |A| filter over signed.

### C §5 Things the code should log but doesn't

1. Mean approx-KL (k3) and clipfrac on the first and last minibatch.
2. Pre-clip gradient norm / `clip_coef` mean.
3. Per-group entropy (spine / armL / armR / legL / legR) and min-head entropy.
4. Full-batch entropy under the filter (the printed `ent` is kept-row entropy).
5. Bootstrap calibration: `b_val[T]` from update u vs realized discounted 32-tick return from u+1 for the same envs.
6. Critic skill score vs a trivial baseline (per-update least-squares on 2-3 obs lanes).
7. Kept-set/full-set advantage σ ratio when filtering.
8. Diag on kept rows when filtering.
9. Per-position advantage variance (by t within the rollout).
