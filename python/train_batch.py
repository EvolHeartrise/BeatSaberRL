"""Batched-rollout PPO: N parallel envs (default --env cuda, the P5
kernel sim), ONE policy forward per vector step. GAE + dual-clip PPO;
batch = num_envs x horizon.

Throughput study:  python train_batch.py --bench --num-envs 256
Real training:     python train_batch.py --num-envs 128 --total-steps 5000000

NOTE on science: changing num_envs changes the optimization regime
(updates-per-sample, minibatch size). Throughput comparisons are clean;
learning-curve comparisons across different batch shapes are NOT
apples-to-apples without retuning (LR, epochs).

CUDA graphs were REMOVED 2026-07-25: the launch storm they hid (12
separate heads + 12 Categoricals, ~120 tiny kernels per forward) was
eliminated at the source by the fused action head + last-block pruning
in policy.py, after which capture measured as pure overhead (static-
buffer copies + capturable Adam) at every shape tried. Checkpoints
saved by graphs-era runs load unchanged (they were normalized to the
eager Adam form at save time).

The update path is eager torch + the fp16-storage helpers
(fp16_linear.py: flat fp32 masters with contiguous fp16 replicas,
fused Adam over flat views, device-side dynamic loss scale). The old
fused-kernel update stack (recompute backward, packed single-leaf
Adam, d64/1-head only) was DELETED at the 2026-08-13 scale-up — see
PERF_AUDIT_2026-08-16.md for what is and isn't currently measured.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from collections import deque
from datetime import datetime

import numpy as np

# env v47 --ep-stats row layout (float32; v2 since 2026-10-04 = header row
# + the 16 v1 cols, update, 12 per-hand swing cols — see ep_stats.py)
from ep_stats import EP_STATS_COLS, EpStatsWriter, build_rows as ep_stats_rows
import torch
import torch.nn as nn

from beat_arena_env.vec_env import BeatVecEnv
from beat_arena_env.constants import joint_gear_counts, parse_tau_scale
# the ranked pool's recipe (lead gap, mirror) lives with the pool: both
# are in its npz cache key, so prefetch_maps and this trainer must agree
from beat_arena_env.ranked_pool import RECIPE_LEAD_S, RECIPE_MIRROR
from fp16_linear import bump_weights_version
from metrics_log import dumps_record, update_record
from policy import (hl_loss, HEAD_GROUPS, HEAD_SIZES, NotePolicy, OBS_DIM,
                    load_policy)

A_DIM = len(HEAD_SIZES)  # 29 factored joint heads (env v10)

GAMMA = 0.965  # the recipe since v53 (2026-09-21)
GAE_LAMBDA = 0.95
CLIP = 0.3  # 0.3 since 2026-09-15 (user decision after the clip ladder)
DUAL_CLIP_C = 3.0
VF_COEF = 0.5
# value-target normalization EMA rate, per update (2026-08-19 critic
# redesign, T1): the critic predicts
# NORMALIZED values v_hat; V = v_hat*sig+mu everywhere GAE reads it.
# Fixes the silent VF_COEF re-weighting under return-scale shifts
# (v26 gamma 0.99 inflated v_loss ~10x into the shared trunk).
# halflife ~69 updates at 0.01.
VALUE_NORM_BETA = 0.01
LR = 5.7e-4  # 7e-4 * sqrt(128/192), the wide-arch recipe
MAX_GRAD_NORM = 0.5
# log-ratio clamp before exp() (2026-08-02, 400M-run postmortem). NOT a
# science change: dual-clip's selection only lets gradient flow through
# ratio where ratio < max(1 + CLIP, DUAL_CLIP_C), and the loss VALUE is
# selection-invariant for any ratio above that, so clamping at e^20 =
# 4.9e8 changes no finite loss or gradient — it only stops exp()
# overflowing fp32 at logratio > ~88, whose inf then rode 0*inf = NaN
# through the max/min/where backward into every grad (the residual
# scale-1.0 overflow-zeroed steps in the 400M log).
LOG_RATIO_CLAMP = 20.0
# dynamic loss-scale growth: x2 after this many consecutive clean
# gradient steps (torch GradScaler defaults 2000; 500 here — 62 updates
# at mb8 — because the scale also has a hard cap at the initial value)
FP16_GROWTH_INTERVAL = 500.0


def pg_loss_fn(ratio, adv, clip=CLIP, dual_clip_c=DUAL_CLIP_C):
    """Dual-clip PPO policy loss (arXiv:1912.09729). (A frac_active
    diagnostic — how often the dual-clip ceiling binds — was computed
    here per minibatch and never read by any caller; removed
    2026-07-28, 5 kernels/minibatch.)"""
    pg1 = -adv * ratio
    pg2 = -adv * torch.clamp(ratio, 1 - clip, 1 + clip)
    standard = torch.max(pg1, pg2)
    ceiling = -dual_clip_c * adv
    dual = torch.min(standard, ceiling)
    loss = torch.where(adv < 0, dual, standard)
    return loss.mean()


def adv_filter_indices(f_adv, train_n):
    """Top-advantage filter (arXiv:2606.23348 §V-E): indices of the
    train_n transitions to train on, ranked by |GAE advantage| —
    two-sided, so both strong positive and strong negative surprises
    survive and the near-zero-gradient middle drops. (A signed
    one-sided variant was cut before ever training a full run,
    2026-08-31 user decision: its critic only saw returns above its
    own predictions and EV collapsed 0.768 -> 0.015 in the 50-update
    smoke.) Unordered (sorted=False); the caller shuffles."""
    return torch.topk(f_adv.abs(), train_n, sorted=False).indices


def adv_filter_stats(f_adv, keep_idx):
    """Advantage-normalization statistics under --adv-filter (2026-09-02,
    Tier 1.4 of the 09-02 static audit, C §1.1). Returns (mu, sd,
    sd_kept / sd_full), all 0-dim device tensors, no host sync.

    Why FULL-batch: the kept set {|A| >= median|A|} has sd ~1.36x the
    full batch's (Gaussian: sqrt(1.857); Laplace 1.39). Normalizing each
    minibatch by ITS OWN stats — which is what the unfiltered path does
    and what the filtered path did until today — therefore shrank the
    normalized advantages 0.73x relative to the entropy and value terms,
    i.e. an implicit ent_coef x1.36 and, compounded with the kept rows'
    larger value residuals, ~2.2x the value-gradient share into the
    shared trunk. The 2026-08-31 "parity" probe was run with that
    confound. Normalizing by the full batch's mu/sd puts a kept row on
    exactly the scale it would have had unfiltered, so the filter changes
    WHICH rows carry gradient and nothing about the loss balance.

    The ratio is the C §5 item-7 instrument: it is the confound's
    magnitude, measured per update instead of assumed. Computed once per
    update on the flat GAE buffer — one gather of train_n floats plus
    three reductions, against a ~46k-launch update."""
    mu = f_adv.mean()
    sd = f_adv.std()
    ratio = f_adv[keep_idx].std() / (sd + 1e-8)
    return mu, sd, ratio


def entropy_report(ent_head):
    """Per-limb-group entropy, the weakest head, and the 29-head sum.

    `ent_head` is a length-29 sequence of MEAN per-head entropies in the
    env's actuated order (policy.HEAD_GROUPS: spine 3, armL 7, armR 7,
    legL 6, legR 6). Pure host-side arithmetic over values that have
    ALREADY been synced by the per-update print — it adds no device
    work and no sync.

    Group figures are MEANS over the heads in a group, not sums, so
    spine (3 heads) and armL (7 heads) are directly comparable to each
    other; ln(K) is the per-head ceiling (1.0986 at --gears 3, 1.6094
    at --gears 5). The returned total is the 29-head SUM, i.e. the
    full-batch analogue of the printed `ent` (2026-09-02).

    Returns (groups, min_head_index, min_head_entropy, total)."""
    groups = [(nm, sum(ent_head[a:b]) / (b - a))
              for nm, a, b in HEAD_GROUPS]
    lo = min(range(len(ent_head)), key=ent_head.__getitem__)
    return groups, lo, ent_head[lo], sum(ent_head)


def format_kl_line(update, klf, kll, cff, cfl, gnorm, clipc, ent_head,
                   af_ratio=None, kl_stop=None):
    """The `[kl]` diagnostic line (2026-09-02, Tier 0.3 of the 09-02
    static audit). `mb1/mbN` = first/last minibatch of THIS update, so
    the pair brackets the within-update drift over 64 optimizer steps.
    kl is the k3 estimator; clipfrac is the share of rows outside
    1 +/- CLIP, which contribute zero policy gradient. entfull is the
    full-batch 29-head entropy sum from the t=0 rollout forward — it is
    unaffected by --adv-filter (unlike `ent`, which is the last
    minibatch's KEPT rows) and lags `ent` by one update. `af_ratio`
    (1.4, only when filtering) is sd(kept advantages) / sd(all), the
    measured size of the normalization confound C §1.1 predicted at
    ~1.36 — printed as `afsd`."""
    groups, lo, lo_v, total = entropy_report(ent_head)
    gs = "  ".join(f"{nm} {v:.3f}" for nm, v in groups)
    af = "" if af_ratio is None else f"  afsd {af_ratio:.3f}"
    return (f"[kl] upd {update:>3}  kl(mb1/mbN) {klf:.2e}/{kll:.2e}"
            f"  clipfrac {cff:.4f}/{cfl:.4f}"
            f"  gnorm {gnorm:.3f}  clipc {clipc:.3f}"
            f"  ent/head {gs}  min {lo_v:.3f} (h{lo})"
            f"  entfull {total:.3f}{af}"
            + ("" if kl_stop is None else f"  STOP@mb{kl_stop}"))


def build_parser():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    # DEFAULTS = THE PRODUCTION RECIPE (2026-09-22, user): a bare
    # `train_batch.py --run-dir X` launches the v58 trunk recipe
    # (runs/v58_6blk_body3/args.json). Recipe switches are
    # BooleanOptionalAction (--no-ranked-maps, --no-mirror-augment, ...);
    # CPU / --env rust / test runs pass small shapes and --no-ranked-maps.
    p.add_argument("--num-envs", type=int, default=32768)
    p.add_argument("--horizon", type=int, default=32)
    # e1/mb8 was the default from 2026-07-26 (user decision after the paired
    # 50M gate): same 8 gradient steps/update as the old e2/mb4 at HALF
    # the update compute — measured +110.9 wpp (dAcc t=+12.88,
    # score_acc +32.5%) AND 1.61x wall (249.5k steps/s). The 07-21
    # "single-pass fails" study conflated sample reuse with
    # gradient-step density; density was what mattered.
    p.add_argument("--minibatches", type=int, default=64)
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--total-steps", type=int, default=1_000_000)
    p.add_argument("--time-cap", type=float, default=0.0)
    p.add_argument("--lr", type=float, default=LR)
    # WSD is the ONLY schedule since 2026-08-11 (user decision after the
    # 3-round A/B: r3 leaf 166.4 vs cosine 174.6 = parity, falls 149 vs
    # 859, plus extendable trunks). Every run is a TRUNK (warmup then
    # constant lr, open-ended) or a LEAF (--cooldown-updates branch).
    # cosine/linear and the ckpt_upd/final.pt naming removed with it;
    # pre-WSD checkpoints remain usable as --init-from warm-starts.
    p.add_argument("--lr-final", type=float, default=1e-4,
                   help="the leaf cooldown's lr floor")
    p.add_argument("--warmup-updates", type=int, default=34,
                   help="linear lr warmup length in updates (0 = auto, "
                        "1.5%% of the run's updates); stored in the "
                        "checkpoint schedule so resumes agree")
    # 2026-08-19 critic redesign (T0-T4)
    p.add_argument("--vf-coef", type=float, default=VF_COEF,
                   help="value-loss weight; targets are normalized, so "
                        "this is a real relative weight now")
    p.add_argument("--critic-lr-mult", type=float, default=3.0,
                   help="critic param-group lr multiplier (T1, "
                        "ProtoMotions ships 5x; 1.0 disables)")
    p.add_argument("--probe-capture-update", type=int, default=250,
                   help="psrank probe: freeze the [diag] batch of the "
                        "first diag at or after this update (the trained "
                        "policy's own states; earlier diags print no "
                        "psrank). Before 2026-10-01 the probe was the "
                        "update-0 batch, i.e. states of the random "
                        "initial policy")
    p.add_argument("--diag-every", type=int, default=50,
                   help="every N updates run the critic/trunk "
                        "diagnostics minibatch (grad cosine, dormant "
                        "neurons, feature rank; ~2 extra backwards on "
                        "2048 rows + one host sync). 0 disables")
    p.add_argument("--mb-accum", type=int, default=1,
                   help="split each minibatch into K micro-batches with "
                        "gradient accumulation — same math (full-mb adv "
                        "stats, means of equal micro means), ~1/K the "
                        "update-pass activation peak (24 GB fallback)")
    # Top-advantage filtering (arXiv:2606.23348 §V-E, Generals.io):
    # keep only the top FRACTION of each rollout batch ranked by
    # |GAE advantage| and run the whole update (policy + value +
    # entropy) on those alone. Minibatch SIZE is unchanged; the
    # minibatch COUNT shrinks, so optimizer steps per update scale
    # with the fraction — that is where the wall-clock saving comes
    # from. Ranking is by MAGNITUDE (two-sided; see
    # adv_filter_indices for why the signed variant was cut).
    p.add_argument("--adv-filter", type=float, default=0.5,
                   help="fraction of the rollout batch to train on, "
                        "selected by |advantage| rank (1.0 = off). "
                        "Rounded to a whole number of minibatches")
    p.add_argument("--target-kl", type=float, default=1.0,
                   help="KL early stop (2026-09-20 after v52's "
                        "update-561 runaway: last-minibatch KL 0.2 -> 52 "
                        "in three updates, ratio 144, 2,000 falls). Every "
                        "--kl-check-every minibatches the k3 approx KL of "
                        "the current minibatch vs the rollout policy is "
                        "read (one host sync); past this value the rest of "
                        "the update is skipped. Healthy updates run 0.2-0.3 "
                        "at the last minibatch. DEFAULT 1.0 (the recipe "
                        "guard since v55); 0 = off")
    p.add_argument("--kl-check-every", type=int, default=4,
                   help="minibatch stride of the --target-kl probe")
    p.add_argument("--clip", type=float, default=CLIP,
                   help="joint PPO ratio clip epsilon (the CLIP constant; "
                        "0.3 since 2026-09-15 — the ladder: 0.3 beat 0.2 "
                        "by +1.7%% bench, 0.4 lost −8.9%% while training acc "
                        "barely moved; 0.2 was the value from the start). "
                        "Also the clipfrac threshold in the [kl] "
                        "diagnostics")
    # Decoupled weight decay (2026-09-12, AdamW semantics, motivated by
    # the parameter-norm census in research/param-norms-2026-09-11):
    # the 2-D weights random-walk outward with optimizer steps under
    # plain Adam (actor norm^2 +6,150 per 1,000 updates, LN gains
    # 1.0 -> 0.32), which silently decays the effective lr on every
    # LN-fed matrix. Applied as p *= 1 - lr_group * wd right before
    # opt.step(), exactly what torch.optim.AdamW does, on ndim >= 2
    # params only (LN gains and biases undecayed). The critic group
    # decays at its own (x3) lr, as AdamW would. Keeps the 2-group Adam
    # state format, so --resume of Adam checkpoints stays exact.
    p.add_argument("--weight-decay", type=float, default=0.0,
                   help="decoupled (AdamW-style) weight decay on 2-D "
                        "weights; 0 = plain Adam (every run before "
                        "2026-09-12). Time constant 1/(lr*wd) optimizer "
                        "steps: 0.1 at lr 7e-4 = 14k steps")
    # Hybrid Muon (2026-09-12 probe; the 2026-09-10 PufferLib 5
    # evaluation's "Muon deep dive"), the ONLY optimizer since 2026-09-22
    # (pure Adam, every run before 2026-09-12, removed; so is the
    # per-matrix NS loop every run before v48 used — the shape-batched
    # bmm step is the only step): Muon (Jordan 2024) on the HIDDEN
    # 2-D matrices — attention q/k/v/out, FFN, trunk, critic attention +
    # hidden MLP layers — and the existing Adam on everything else (the
    # three encoder input layers, the 29 heads, the critic's final 1-row
    # layer, biases, LN gains). Updates are RMS-matched to Adam's
    # (0.2*sqrt(max(A,B)) scale, Moonshot arXiv 2502.16982) so --lr and
    # --critic-lr-mult keep their meaning. --weight-decay applies to the
    # Muon groups too. Checkpoints carry the Muon momentum under "muon";
    # --resume needs such a checkpoint (Adam-era ones are --init-from
    # only). User decision after the v40 probes: 0.9409 vs Adam 0.9099
    # training acc at matched 1.12B samples, seed 1 tracking seed 0.
    p.add_argument("--muon-enc-hidden", action=argparse.BooleanOptionalAction,
                   default=False,
                   help="also put the encoders' hidden 192x192 layers "
                        "(note/saber/torso/leg_mlp.2) on Muon; only their "
                        "input layers stay on Adam (architecture review "
                        "2026-10-01). Changes the optimizer layout: a "
                        "--resume inherits it from the checkpoint")
    p.add_argument("--depth-init", action=argparse.BooleanOptionalAction,
                   default=False,
                   help="fresh runs: scale every residual write (block "
                        "out_proj, ffn.2, trunk_fc2) by 1/sqrt(2*n_blocks) "
                        "with zero bias, and the 29 action heads by 0.01 "
                        "with zero bias (GPT-2 style depth-safe init; "
                        "architecture review 2026-10-01). Ignored with "
                        "--resume / --init-from")
    p.add_argument("--muon-momentum", type=float, default=0.95,
                   help="Muon Nesterov momentum (Jordan default 0.95; "
                        "raise it under noisier gradients)")
    # env v41 (2026-09-12, user): self-contact penalty. Greedy replays of
    # the v40 policies: 10-13% of ticks carry a body-on-body contact,
    # dominated by hands/forearms hitting the head and torso; summed
    # impulse ~0.18 N.s/tick (Muon) vs 0.58 (Adam). coef 0.05 prices
    # that at ~0.009/tick, ~5% of mean reward/tick.
    p.add_argument("--self-contact-coef", type=float, default=0.0,
                   help="env v41: per-tick price on the summed "
                        "self-collision impulse (N.s) — body-on-body "
                        "only, blades collide with nothing; 0 = off "
                        "(every run before 2026-09-12). Telemetry "
                        "([contact] line, metrics) accumulates always")
    p.add_argument("--z-asinh", type=float, default=None,
                   help="env v48: obs z lanes (note z / wall front, wall "
                        "back) = asinh(max(z, head z) / C) with C in metres "
                        "— linear near the cut plane, log far out, no upper "
                        "clamp. The only z encoding (the legacy clamp / 8 "
                        "path was removed 2026-09-22); C must be > 0. "
                        "DEFAULT 8 for a fresh run (the recipe since "
                        "2026-09-21, v55); a --resume/--init-from run "
                        "inherits the checkpoint's value when the flag is "
                        "not given. Recorded in the checkpoint (arch + "
                        "args); the bench/replay tools read it from there")
    p.add_argument("--cooldown-updates", type=int, default=0,
                   help="wsd LEAF mode: with --resume <trunk ckpt>, branch "
                        "off and decay lr 1-sqrt from --lr to --lr-final "
                        "(and ent linearly from --ent-coef to --ent-final) "
                        "over this many updates. Saves "
                        "leaf_fromNNNNN_*.pt; leaves are terminal "
                        "deliverables — continue training from the TRUNK "
                        "checkpoint, never from a leaf")
    p.add_argument("--ent-coef", type=float, default=0.03)
    p.add_argument("--ent-final", type=float, default=1e-3,
                   help="leaf: the cooldown's entropy target; trunk with "
                        "--ent-tau: the anneal FLOOR. Ignored (with a "
                        "notice) on a constant-ent trunk")
    p.add_argument("--ent-tau", type=int, default=125,
                   help="trunk: exponential entropy-anneal time constant "
                        "in UPDATES — ent(u) = --ent-final + (--ent-coef - "
                        "--ent-final) * exp(-u/tau), a pure function of the "
                        "ABSOLUTE update index, so extending the trunk "
                        "continues the same curve (budget-free). 0 = constant "
                        "trunk entropy. On the trunk --ent-final is the "
                        "anneal FLOOR, not an endpoint. Motivated 2026-08-11: "
                        "round-1 A/B proved constant 0.03 PAYS for "
                        "randomness (41.6 nats at trunk end, leaf 1.5 vs "
                        "cosine 174.6 upp) — the ent bonus is part of the "
                        "objective, unlike lr")
    p.add_argument("--gamma", type=float, default=GAMMA,
                   help="discount (default 0.965, the recipe since v53; "
                        "0.99 ~ 3.3 s effective horizon at 30 Hz was the "
                        "old default; swept 2026-07-27 — user hypothesis: "
                        "Beat Saber decisions live ~1 s ahead, gamma "
                        "~0.967)")
    p.add_argument("--gae-lambda", type=float, default=GAE_LAMBDA,
                   help="GAE trace decay (default 0.95; swept 2026-07-27 "
                        "jointly with --gamma — gamma*lambda sets the "
                        "policy-gradient credit window, gamma alone the "
                        "value-target horizon)")
    p.add_argument("--value-norm-beta", type=float, default=VALUE_NORM_BETA,
                   help="value-target normalization EMA rate per UPDATE "
                        "(default 0.01, halflife ~69 updates). It is "
                        "update-indexed, so a T-rescaled recipe that "
                        "doubles samples per update should double it to "
                        "keep the halflife the same in samples (T=64: "
                        "0.02; audit 2.6). Default is bit-identical to "
                        "the old module constant.")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--ckpt-every", type=int, default=0)
    p.add_argument("--run-dir", type=str, default=None)
    p.add_argument("--init-from", type=str, default=None,
                   help="warm-start WEIGHTS only; fresh optimizer + schedule")
    p.add_argument("--resume", type=str, default=None,
                   help="EXACT resume from a checkpoint of this trainer: "
                        "restores weights, optimizer state, value norm, "
                        "the RNG streams and the original schedule "
                        "position (lr/ent/warmup come from the checkpoint, "
                        "CLI values ignored; a leaf's --lr-final / "
                        "--ent-final are its own). Every other flag NOT "
                        "given on this command line inherits the "
                        "checkpoint's recorded value (args); given flags "
                        "that differ are printed as a diff list. Never "
                        "inherited: --run-dir, --total-steps, --time-cap, "
                        "--ckpt-every, --cooldown-updates, --ent-final, "
                        "--device, --env, --bench. The env/chart stream "
                        "restarts fresh — only the optimization state is "
                        "exact.")
    p.add_argument("--ignore-arch-flags", action="store_true",
                   help="--resume: proceed when an explicitly given arch "
                        "flag (--embed/--n-blocks/--hl-bins/...) differs "
                        "from the checkpoint's; the checkpoint's arch "
                        "wins either way (refused by default so the flag "
                        "is not silently dropped)")
    p.add_argument("--bench", action="store_true",
                   help="short run, prints steps/s split (rollout vs update)")
    p.add_argument("--device", type=str, default=None,
                   help="cuda|cpu (default: cuda if available)")
    p.add_argument("--env", choices=["rust", "cuda"], default=None,
                   help="cuda = the P5 kernel sim (the training backend "
                        "since 2026-08-01; one launch per tick). rust = "
                        "the reference sim over FFI (CPU path + the "
                        "parity reference). Default: cuda on a CUDA "
                        "device, else rust. (The P4 torch mirror, "
                        "--env torch, was removed 2026-08-03.)")
    p.add_argument("--graph-rollout", action="store_true", default=False,
                   help="capture the rollout step (policy fwd + sample + "
                        "env kernel) as ONE CUDA graph and replay it for "
                        "t = 1..T-1 (t = 0 stays eager: full-batch "
                        "entropy). RESTORED 2026-09-19 for slow-single-"
                        "core hosts where python dispatch starves the "
                        "GPU; measured 0.99x on a fast core (2026-08-31) "
                        "and 0.985x on a slow shared Xeon (VastAI, "
                        "2026-09-19: 8.91 vs 8.78 s/update, ABAB) — the "
                        "rollout is GPU-execution-bound at both widths, "
                        "so this has never paid; kept as a knob only. "
                        "Changes the sampling RNG stream vs eager. Needs "
                        "--env cuda and not --fp16 update; auto-downgrades "
                        "with a note otherwise")
    p.add_argument("--chart-pool", type=int, default=999999,
                   help="level subsample size with --ranked-maps "
                        "(default 999999 = the whole pool); pregenerated "
                        "chart pool size with --no-ranked-maps --env cuda, "
                        "where 65536 was the default (2026-07-26, user "
                        "decision "
                        "after two paired 50M gates: 1024->8192 +91.3 "
                        "wpp (t=+8.31), 8192->65536 +80.9 wpp (t=+12.55, "
                        "score_acc +21%%, bad_cuts -31%%, FCs +38%%) — "
                        "chart diversity was the binding constraint. "
                        "Costs ~96 s pool gen + ~242 MB VRAM per leg; "
                        "~1.2 repeats/chart on a 50M leg, ~16 on 700M)")
    p.add_argument("--fp16", choices=["off", "update", "all"],
                   default="all",
                   help="fp16-storage activations (fp32 cuBLAS compute, "
                        "fp32 master weights, fp32 PPO math; 2026-07-25). "
                        "'all' (the DEFAULT since 2026-07-26, user "
                        "decision after the graph gate: +72.4 wpp over "
                        "'update', r1 2.0e-3 -> 6.7e-4 — rollout/update "
                        "consistency beats rollout precision) = rollout "
                        "AND update; 'update' = minibatch fwd+bwd only "
                        "(the pre-gate default); 'off' = full fp32 "
                        "(record-attempt fallback). Checkpoints stay "
                        "fp32 in every mode. Forced off on CPU devices")
    p.add_argument("--fp16-accum", choices=["off", "fwd", "all"],
                   default=None,
                   help="GEMM accumulate mode of the fp16 stream (issue "
                        "#3, fp16_linear.fp16_accum): 'off' = fp32 "
                        "accumulate everywhere (the historical path); "
                        "'fwd' = the torch forward GEMMs accumulate in "
                        "fp16 (CUBLAS_COMPUTE_16F, ~2x tensor-core peak "
                        "on the RTX 5090); 'all' = 'fwd' + the dgrad "
                        "GEMMs. The wgrad / bias-grad reductions and the "
                        "ReLU-epilogue GEMMs stay fp32 in every mode. "
                        "Rollout and update always share the mode (their "
                        "forwards stay bitwise at equal row counts). "
                        "Stamped in the checkpoint arch. Default: 'all' "
                        "for fresh runs and --init-from warm starts; a "
                        "--resume keeps its checkpoint's mode (absent = "
                        "'off')")
    p.add_argument("--fused-ffn", choices=["off", "on", "up16"],
                   default=None,
                   help="fused block-FFN kernel on the fp16 stream (review "
                        "I13, csrc/fused_ffn.cu): FFN + residual + the next "
                        "LayerNorm in one kernel, the 768-wide hidden kept "
                        "on-chip (rollout writes no hidden / ReLU mask). "
                        "'on' = up-projection fp32-accumulated, down-"
                        "projection per --fp16-accum; 'up16' = both fp16-"
                        "accumulated — LIKELY WORSE FOR TRAINING: the "
                        "2026-09-30 A/B (one seed per arm, full recipe "
                        "trunk + cd400 leaf) benched up16 -2.7%% wpp / "
                        "-2.3%% upp at the default body and -2.9%% wpp at "
                        "each leaf's own fit, lower acc in every star band, "
                        "+37%% bad cuts, for a 1.8%% wall-clock saving. Near-"
                        "tie numerics vs "
                        "the unfused path; rollout and update stay bitwise. "
                        "Stamped in the checkpoint arch. Default: 'on' for "
                        "fresh runs and --init-from since 2026-09-30 (pod "
                        "bench: rollout forward 51.7 -> 45.9 ms at N 32768, "
                        "r1 unchanged); a --resume keeps its checkpoint's "
                        "mode (absent = 'off')")
    p.add_argument("--fp16-scale", type=float, default=131072.0,
                   help="initial loss scale for --fp16 (2^17 from the "
                        "2026-07-25 grad probe: max|g| 2.2e-2, 26%% of "
                        "grad elements below fp16 normal range unscaled). "
                        "Dynamic since 2026-08-02: halves per overflow "
                        "MINIBATCH and doubles after 500 clean gradient "
                        "steps (capped at the initial value, floored at "
                        "1) — the 400M postmortem found the old "
                        "halve-per-UPDATE never-grow ratchet collapsed "
                        "131072->1 by update 188 and stayed there for "
                        "94%% of the run, discarding 9.3%% of all "
                        "gradient steps as overflow-zeroed")
    # model arch (ignored with --init-from: the checkpoint's arch wins)
    # v17 scaled arch (2026-08-13, user-approved): 861,202 params at
    # gears 5 (was 138,642). Widened 2026-09-13 (user decision after the
    # v40_w192 probe): embed 192 / trunk 384 = 2,414,930 params, +0.45 pt
    # trunk acc at matched 1.12B samples under Muon, cd300 bench +3.4%
    # upp (2,353,043, pod best), +15% wall; lr 5.7e-4 (7e-4 * sqrt(128/192)).
    # Checkpoints carry their own arch, so 128/256 ckpts still load.
    p.add_argument("--embed", type=int, default=192)
    p.add_argument("--trunk-dim", type=int, default=384)
    p.add_argument("--n-blocks", type=int, default=6)
    p.add_argument("--n-heads", type=int, default=4)
    p.add_argument("--ffn-mult", type=int, default=4)
    # readout trunk: SimBa (Linear projection -> residual block
    # x + fc2(relu(fc1(LN x))) -> LN, SimBa 2410.09754; the plain MLP
    # trunk was removed 2026-09-22) unless --trunk-arch simbav2
    p.add_argument("--trunk-hidden", type=int, default=None,
                   help="hidden width of the SimBa trunk's residual "
                        "block (default = --trunk-dim; 1536 = the ffn x4 "
                        "rung, +885,888 params at 192/384)")
    # --- architecture A/B options (2026-10-03). All OFF by default; off
    # = today's network bit for bit. Each is stamped in the checkpoint
    # arch (load_policy rebuilds it; absent = off) and inherited by a
    # --resume (RESUME_ARCH).
    p.add_argument("--emb-ln", action=argparse.BooleanOptionalAction,
                   default=False,
                   help="LayerNorm (learnable gain/bias) on the 29 input "
                        "tokens before block 0 (block 0 writes 15-27x the "
                        "embeddings' RMS into the stream)")
    p.add_argument("--value-residual", action=argparse.BooleanOptionalAction,
                   default=False,
                   help="ResFormer value residual (arXiv 2410.17897): "
                        "blocks 1..L-1 use v <- a v + b v_0 with block 0's "
                        "attention values v_0; learnable per-block scalars "
                        "init (a, b) = (1, 0) = the baseline")
    p.add_argument("--final-ln", action=argparse.BooleanOptionalAction,
                   default=False,
                   help="LayerNorm on the 5 lead tokens' final-block "
                        "outputs before trunk_proj (they arrive at RMS "
                        "~30 otherwise)")
    p.add_argument("--trunk-arch", choices=["simba", "simbav2"],
                   default="simba",
                   help="readout trunk: simba (the production trunk) or "
                        "simbav2 (SimBaV2 arXiv 2502.15280: hyperspherical "
                        "normalisation, unit-norm weight rows re-projected "
                        "after every optimizer step, learnable scalers and "
                        "LERP residual; hidden width = --trunk-hidden)")
    p.add_argument("--note-geo", action=argparse.BooleanOptionalAction,
                   default=True,
                   help="DEFAULT ON since 2026-09-14 (user decision: part of "
                        "the recipe with env v42; --no-note-geo opts out). "
                        "Hand-relative geometry lanes on the note tokens "
                        "(2026-09-14, research/2026-09-14-architecture-"
                        "brainstorm.md #1): per (hand, note) the wrist->"
                        "note offset, distance, seconds-to-plane and the "
                        "swing/blade . cut-direction dots (14 lanes) are "
                        "computed from the obs inside the policy and "
                        "appended to the 18 note lanes before note_mlp. "
                        "Obs/env untouched; arch marker 'note_geo' is "
                        "stamped so bench/replay tools rebuild it")
    # env config (generator v2 dials, SPEC.md §Chart generation)
    # --colors REMOVED (2026-07-26, user decision): the right-only
    # default silently trained gen3's 5B run one-handed (49.9% of
    # real-map notes uncuttable — the whole transfer regression).
    # Training is always both-hands now; ChartConfig still accepts
    # colors="right" programmatically for tests/curricula.
    p.add_argument("--cols", type=int, nargs="+", default=[0, 1, 2, 3])
    p.add_argument("--rows", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--chart-len", type=float, default=20.0)
    p.add_argument("--note-speed-min", type=float, default=None,
                   help="speed random-walk clamp (m/s); default 2 "
                        "(standard) / 10 (flux, real-NJS range)")
    p.add_argument("--note-speed-max", type=float, default=None,
                   help="default 8 (standard) / 18 (flux)")
    p.add_argument("--game", choices=["standard", "flux"],
                   default="standard",
                   help="game mode (SPEC.md §Flux). flux = circles + bars "
                        "on the palms/Flux line, no sabers: procedural "
                        "charts only (flux_chart.py), so --ranked-maps "
                        "and --ep-stats switch off; the policy is stamped "
                        "arch game='flux' and never loads into the "
                        "standard tools (or vice versa)")
    p.add_argument("--circle-prob", type=float, default=0.3,
                   help="flux: P(circle) per hand path per generation tick")
    p.add_argument("--bar-prob", type=float, default=0.15,
                   help="flux: P(bar) per generation tick")
    p.add_argument("--note-prob", type=float, default=0.25,
                   help="P(note) per hand path per ~0.15 s generation tick")
    p.add_argument("--extra-prob", type=float, default=0.125,
                   help="P(a spawned note brings an extra along its swing line)")
    p.add_argument("--dot-prob", type=float, default=0.1,
                   help="P(a generated note turns into a dot)")
    p.add_argument("--bomb-prob", type=float, default=0.125,
                   help="P(bomb) per generation tick")
    # (the v10 "effort" work price and its --effort-coef flag were RETIRED
    # 2026-09-13: every recipe since v23 ran it at 0. The v23 fraction
    # "activation" price (--iso-coef) was retired 2026-09-14 for the env
    # v43 torque energy price; the live movement prices are --energy-coef
    # (Σ|τ|) and --limit-coef/--limit-arms (limit rest).)
    p.add_argument("--gears", type=int, default=5, choices=[3, 5, 7],
                   help="env v8 action-table size per joint head: 3 = "
                        "ternary coast/+full/-full; 5 adds +/-1/4-torque "
                        "gears; 7 adds +/-1/3 and +/-1/8 (fine control "
                        "under the doubled alpha/omega body). Resuming a "
                        "geared checkpoint needs the matching --gears. "
                        "The BASE table: a build's per-joint extra gears "
                        "(constants.JOINT_EXTRA_GEARS, e.g. exp/bend7's "
                        "7-gear wrist bends) append on top of it")
    p.add_argument("--foot-contact", default=True,
                   action=argparse.BooleanOptionalAction,
                   help="DEPRECATED NO-OP: contact mode (env v11) is the "
                        "only physics — the v10 weld harness was removed "
                        "from the engines 2026-09-27. --foot-contact is "
                        "accepted so older drivers keep running; "
                        "--no-foot-contact is an ERROR")
    p.add_argument("--ranked-maps", action=argparse.BooleanOptionalAction,
                   default=True,
                   help="train on WHOLE LEVELS of the real ranked ScoreSaber "
                        "catalog (beat_arena_env/ranked_pool.py; bench "
                        "parsing protocol, real NJS speeds since "
                        "2026-07-30) instead of the procedural "
                        "generator. --chart-pool becomes the level "
                        "subsample size; generator dials "
                        "(--note-prob/--bomb-prob/...) are ignored. "
                        "Requires --env cuda. (sweep s2, 2026-07-29)")
    p.add_argument("--catalog", type=str, default="scoresaber",
                   choices=("scoresaber", "beatleader", "both"),
                   help="--ranked-maps level catalog: scoresaber (the "
                        "historical pool), beatleader, or both (the union "
                        "keyed like bench_scoresaber --catalog both, each "
                        "shared level once). Non-SS levels carry no SS pp "
                        "in the ep-stats telemetry (level_lb -1, max_pp "
                        "0). (2026-10-01)")
    p.add_argument("--lead-s", type=float, default=RECIPE_LEAD_S,
                   help="ranked-maps front gap (s) — notes before it are "
                        "dropped, walls clipped to it; the earliest "
                        "first note of an episode (env v16, contact spec "
                        "§11: shorter keeps note data in the mix when "
                        "falls spike, damping survival-drift)")
    p.add_argument("--mirror-augment", action=argparse.BooleanOptionalAction,
                   default=RECIPE_MIRROR,
                   help="double the ranked-maps pool with left-right "
                        "mirrored copies of every level (handedness-"
                        "bias regularizer, 2026-08-14; --chart-pool "
                        "subsamples the doubled pool)")
    p.add_argument("--fall-pen", type=float, default=5.0,
                   help="env v18: constant fall penalty (return units, "
                        "positive; terminal reward = -value). DEFAULT is "
                        "the 3-arm-probe winner (2026-08-15: falls/update "
                        "8-10k -> ~35, bench zeros 35 -> 2); the legacy "
                        "remaining-notes charge is REMOVED")
    p.add_argument("--ep-stats", action=argparse.BooleanOptionalAction,
                   default=True,
                   help="env v47 (2026-09-19): run the ScoreSaber combo "
                        "walk in the training envs and log every "
                        "episode's outcome (multiplied score, accuracy, "
                        "PP, max combo, breaks, completed, cuts) to "
                        "<run-dir>/ep_stats.f32 (EP_STATS_COLS float32 "
                        "cols; reader: ep_stats.py); [pp] line every 25 "
                        "updates. Sampled-policy plays at the drawn body "
                        "— tracking-grade, not the bench. ~0 kernel cost; "
                        "+N*C bytes VRAM (note multiplier plane)")
    # critic head: HL-Gauss (Farebrother 2024), the ONLY head since
    # 2026-09-22 (the scalar MSE head of every run through v48 and its
    # --value-head flag were removed): --hl-bins histogram over
    # [-hl-vmax, hl-vmax] of the normalized return, cross-entropy to a
    # Gaussian-smoothed target (--hl-sigma in bin widths, paper 0.75;
    # tails truncated + renormalized), value = expectation
    p.add_argument("--hl-bins", type=int, default=101)
    p.add_argument("--hl-vmax", type=float, default=6.0,
                   help="HL-Gauss support half-width in NORMALIZED return "
                        "units (the critic target is ~unit variance)")
    p.add_argument("--hl-sigma", type=float, default=0.75)
    p.add_argument("--tau-center", type=str, default="default",
                   help="env v47b: center of the --tau-rand draws per axis "
                        "(16 floats or 'spine=0.9,legs=1.1,...'). DEFAULT "
                        "= the saved body TAU_BODY_DEFAULT (constants.py; "
                        "the v56 leaf fit since 2026-09-22); 'nominal' = "
                        "all 1.0 = the TAU_MAX table (every run through "
                        "v50). Draw = center * (1 +- r). Also the fixed "
                        "body when --tau-rand 0. Moved between trunks by "
                        "tau_fit.py on the previous leaf's ep_stats")
    p.add_argument("--tau-rand", type=float, default=0.25,
                   help="env v46 (user design 2026-09-16): per-env "
                        "randomized joint torque limits — at every "
                        "episode reset each of the 16 axes (spine 3, arm "
                        "7 shared L/R, leg 6 shared L/R) draws a scale "
                        "U(1-r, 1+r) applied to tau_max; the scales are "
                        "in the obs (lanes 608..624). 0 = off (all 1.0, "
                        "bit-identical to v45). Episode outcomes per "
                        "body are logged to <run-dir>/tau_draws.f32 "
                        "(tau_regress.py). CUDA env only")
    p.add_argument("--push-prob", type=float, default=0.00333,
                   help="env v18 perturbation training (DEFAULT = probe "
                        "winner, ~1 kick / 10 s / env): per-(env,tick) "
                        "probability of a horizontal root-velocity kick "
                        "(CUDA env only). 0 disables")
    p.add_argument("--push-vmax", type=float, default=0.15,
                   help="max kick magnitude m/s (|dv| ~ U(0, vmax); "
                        "0.5 -> 0.15 2026-08-20)")
    p.add_argument("--start-jitter", type=float, default=0.15,
                   help="episode-start kick cap m/s for freshly reset "
                        "envs: |dv| ~ U(0, s), direction uniform "
                        "(RSI-flavored start variety; 0.2 -> 0.15 "
                        "2026-08-20)")
    p.add_argument("--foot-hinge-r", type=float, default=0.12,
                   help="env v22: free radius (m) for BOTH foot penalties "
                        "— deviation inside it costs nothing (target "
                        "AREAS), rent applies to the excess. 0 = the v21 "
                        "pure-linear behavior")
    p.add_argument("--foot-shape-coef", type=float, default=0.05,
                   help="env v22: delta shaping on the total hinged foot "
                        "deviation — reward/punish each tick's CHANGE in "
                        "deviation, coef * (prev - now) (round trips net "
                        "zero; sharpens credit assignment). Potential-"
                        "STYLE, not the Ng et al. form, so NOT guaranteed "
                        "optimum-preserving: the difference is undiscounted "
                        "(no gamma, while PPO discounts at --gamma), the "
                        "potential starts each episode at 0 rather than at "
                        "the start state's, and is not zeroed at "
                        "termination — an episode's shaping sums to "
                        "-coef * (deviation at its end), a term that "
                        "depends on where (and how, e.g. a fall) the "
                        "episode ends. 0 = off")
    p.add_argument("--limit-coef", type=float, default=5e-3,
                   help="env v34: per-tick price on the joint-limit-row "
                        "impulse (N*m*s) the stops carry — the POSTURAL "
                        "stops (spine + legs), plus the 14 arm joints "
                        "under --limit-arms (ON by default, so the "
                        "default prices the whole body). Resting a pose "
                        "on its stops pays its gravity moment, an active "
                        "hold does not. 0 = off (bitwise pre-v34 rewards)")
    p.add_argument("--limit-arms", action=argparse.BooleanOptionalAction,
                   default=True,
                   help="env v35: widen --limit-coef's scope to the 14 "
                        "arm joints as well (all-body pricing). DEFAULT "
                        "ON; --no-limit-arms restores the v34 postural-"
                        "only scope, which keeps whip stop-slams free")
    p.add_argument("--wall-despawn", action=argparse.BooleanOptionalAction,
                   default=True,
                   help="env v37: walls despawn on the charged hit "
                        "(bomb-like; training-only credit aid). DEFAULT "
                        "ON (trunk recipe); leaves that must match the "
                        "bench pass --no-wall-despawn")
    p.add_argument("--energy-coef", type=float, default=7.5e-6,
                   help="env v43 (2026-09-14, user): per-tick ENERGY cost "
                        "in reward per N*m*tick — energy_coef * sum over "
                        "the 29 joints of the applied |tau_j| (= |gear "
                        "fraction| x TAU_MAX[j]). Replaces the v23 "
                        "fraction price (--iso-coef, retired): a joint pays "
                        "in proportion to its torque limit, the human "
                        "muscle-volume weighting (a spine full gear costs "
                        "600/1.06x a wrist full gear). Isometric holds and "
                        "limit-jamming pay, coast is free; velocity-blind "
                        "(force, not work). DEFAULT 7.5e-6: calibrated so "
                        "the bill matches the old 5e-4 fraction price at "
                        "the 2026-09-14 leaves' ~400 N*m/tick (~0.003 "
                        "reward/tick). 0 = off")
    p.add_argument("--foot-x-coef", type=float, default=0.01,
                   help="env v21: per-tick penalty coefficient on each "
                        "foot's |world_x - its_stance_x| (per-side "
                        "targets +-FOOT_X0; targets lateral wander). "
                        "0 = off")
    p.add_argument("--foot-z-coef", type=float, default=0.01,
                   help="env v20: per-tick penalty coefficient on each "
                        "foot's |world_z - reference_z| (linear; targets "
                        "the toward/away-from-notes shimmy). 0 = off")
    p.add_argument("--crouch-start-prob", type=float, default=0.25,
                   help="env v38: fraction of resets that spawn in a "
                        "physics-validated crouch (continuous depth, "
                        "knee 0.3-1.5 rad; posture-diversity fix)")
    p.add_argument("--start-pose-jitter", type=float, default=0.15,
                   help="env v20 (CUDA training only): episode-start leg/"
                        "foot joint jitter scale in radians — 12 leg "
                        "joints sampled U(-s, s) (knees |U|), clamped to "
                        "limits, root dropped so no sole corner clips the "
                        "ground. Teaches knee bend / balance variety")
    return p


def parse_args(argv=None):
    return build_parser().parse_args(argv)


# Newton-Schulz coefficient schedule, one (a, b, c) per step, applied as
# p(x) = a x + b x^3 + c x^5 to the singular values of the normalised
# update. The modded-nanogpt medium-track record 4 schedule (2025-02-14,
# leloykun / You Jiacheng), the one PufferLib 5.0 ships in src/algo.cu:
# steeper first step than Jordan's single quintic (4.08 vs 3.44) and two
# polishing steps with p(1) -> 0.94, 0.99, so the same five matmul
# triples return singular values in [0.98, 1.03] (input sigma >= 0.01)
# and stay flat past 1 (1.05 -> 1.03) instead of Jordan's [0.68, 1.13]
# and 1.05 -> 0.68. The recipe since v56 (2026-09-22): matched-update
# trunk never behind Jordan's, ahead 0.4-0.7 pt at 200/400/600, gradient
# 10-35% lower, leaf bench 2,946,594 upp (best on record). Jordan's
# (3.4445, -4.7750, 2.0315) x 5 was every Muon run before v56 and is
# gone (user, no back-compat). NOT Polar Express (the Sept-2025 set
# starting 8.16 / -22.5 / 15.9).
MUON_NS_COEFS = ((4.0848, -6.8946, 2.9270),
                 (3.9505, -6.3029, 2.6377),
                 (3.7418, -5.5913, 2.3037),
                 (2.8769, -3.1427, 1.2046),
                 (2.8366, -3.0525, 1.2012))


def newton_schulz5_batched(G, eps=1e-7, scale=None):
    """Orthogonalize a stack (k, m, n) of same-shaped 2-D updates in one
    bmm chain: each matrix -> the nearest orthogonal matrix by the five
    quintic Newton-Schulz steps of MUON_NS_COEFS (singular values land
    in ~(0.98, 1.03)). Tall matrices are transposed so the iteration
    runs on the small Gram matrix; fp32 (the matrices are <= 768 wide).
    Batched since 2026-09-19 (the only path since 2026-09-22): the old
    per-matrix loop was 5 launch triples per matrix per optimizer step
    (~30k launches per update at the 2026-09-19 arch) — 0.6-0.7 s of the
    update phase on a slow-dispatch host;
    bmm vs mm rounding differs at ~3e-8 (max |update| ~1e-3).
    Each step is 3 kernels since 2026-09-27 (#7b): s = x x^T, then
    B = b s + c s s and x = a x + B x as two baddbmm (the scalar
    multiplies and adds ride the GEMM epilogue; fp32 throughout, no
    TF32). `scale` (optional, a list of (start, stop, factor) row runs
    of the stack covering all of it) folds the caller's per-matrix step
    scale into the LAST step's epilogue: x_out = factor * (a x + B x)."""
    x = G / (torch.linalg.vector_norm(G, dim=(1, 2), keepdim=True) + eps)
    tall = x.shape[1] > x.shape[2]
    if tall:
        x = x.transpose(1, 2)
    last = len(MUON_NS_COEFS) - 1
    for j, (a, b, c) in enumerate(MUON_NS_COEFS):
        s = torch.bmm(x, x.transpose(1, 2))
        s2 = torch.baddbmm(s, s, s, beta=b, alpha=c)
        if j == last and scale is not None:
            out = torch.empty_like(x)
            for lo, hi, f in scale:
                torch.baddbmm(x[lo:hi], s2[lo:hi], x[lo:hi], beta=a * f,
                              alpha=f, out=out[lo:hi])
            x = out
        else:
            x = torch.baddbmm(x, s2, x, beta=a)
    return x.transpose(1, 2) if tall else x


class Muon:
    """The Muon half of the hybrid optimizer: Nesterov
    momentum -> Newton-Schulz orthogonalization -> RMS-matched scale
    0.2*sqrt(max(A, B)) -> SGD step. `param_groups` mirrors
    torch.optim's list-of-dicts (params, lr, lr_mult) so the schedule
    loop and --weight-decay treat it like the Adam groups. Gradients are
    read from p.grad (the flat fp32 buffer's views), after the trainer's
    unscale / overflow-zero / global clip. The step is shape-grouped:
    foreach momentum + one bmm Newton-Schulz chain per distinct matrix
    shape instead of one per matrix."""

    def __init__(self, param_groups, momentum=0.95):
        self.param_groups = param_groups
        self.momentum = float(momentum)
        self.state = {p: torch.zeros_like(p)
                      for g in param_groups for p in g["params"]}

    def _params(self):
        return [p for g in self.param_groups for p in g["params"]]

    @torch.no_grad()
    def step(self):
        # Consolidated 2026-09-27 (#7b): ONE Newton-Schulz chain per
        # matrix shape across BOTH lr groups (lr only scales NS's
        # output), with tall matrices transposed into their wide
        # orientation so ffn.0 (768x192) and ffn.2 (192x768) share one
        # stack — 5 chains at the production arch instead of 8 — and
        # the per-matrix step scale folded into the last NS step's
        # epilogue. fp32 throughout (no TF32/bf16: that changes
        # learning). Same math as the per-group loop; fp32 rounding
        # differs at the ~1e-7 relative level (bmm batch composition,
        # fused epilogue adds).
        from collections import defaultdict
        mu = self.momentum
        ps, scl = [], []
        for g in self.param_groups:
            lr = float(g["lr"])
            for p in g["params"]:
                if p.grad is not None:
                    ps.append(p)
                    scl.append(-lr * 0.2 * math.sqrt(max(p.shape)))
        if not ps:
            return
        bufs = [self.state[p] for p in ps]
        gs = [p.grad for p in ps]
        torch._foreach_mul_(bufs, mu)
        torch._foreach_add_(bufs, gs)
        upds = torch._foreach_add(gs, bufs, alpha=mu)          # Nesterov
        byshape = defaultdict(list)
        for i, p in enumerate(ps):
            m, n = p.shape
            byshape[(min(m, n), max(m, n))].append(i)
        outs = [None] * len(ps)
        for shp, idx in byshape.items():
            # order: wide first then tall (one contiguous un-transpose
            # copy), each sorted by scale so equal-scale runs are
            # contiguous slices for the fused last step
            idx = sorted(idx, key=lambda i: (ps[i].shape[0] > ps[i].shape[1],
                                             scl[i]))
            tall = [ps[i].shape[0] > ps[i].shape[1] for i in idx]
            X = torch.stack([upds[i].t() if t else upds[i]
                             for i, t in zip(idx, tall)])
            runs = []
            for k, i in enumerate(idx):
                if runs and runs[-1][2] == scl[i]:
                    runs[-1][1] = k + 1
                else:
                    runs.append([k, k + 1, scl[i]])
            o = newton_schulz5_batched(X, scale=runs)
            nt = sum(tall)
            if nt:
                ot = o[len(idx) - nt:].transpose(1, 2).contiguous()
            for k, i in enumerate(idx):
                outs[i] = ot[k - (len(idx) - nt)] if tall[k] else o[k]
        torch._foreach_add_(ps, outs)

    def zero_grad(self):
        for p in self._params():
            p.grad = None

    def state_dict(self):
        return {"momentum": self.momentum,
                "bufs": [self.state[p].detach().cpu() for p in self._params()]}

    def load_state_dict(self, sd):
        ps = self._params()
        if len(sd["bufs"]) != len(ps):
            raise ValueError(f"muon state has {len(sd['bufs'])} buffers, "
                             f"policy has {len(ps)} muon params")
        for p, src in zip(ps, sd["bufs"]):
            self.state[p].copy_(src.to(p.device))


def muon_param_names(policy, enc_hidden=False):
    """Which params Muon owns: hidden 2-D matrices only. Encoder input
    layers (note/saber/torso/leg_mlp), the 29 heads, the critic's final
    K-bin HL-Gauss layer, biases and LN gains stay on Adam (Jordan's
    rule). Historically the WHOLE encoder modules stayed on Adam,
    including their hidden second layers; enc_hidden=True
    (--muon-enc-hidden) moves those to Muon and keeps only the input
    layers (`*_mlp.0`) on Adam."""
    names = []
    enc = ("note_mlp", "saber_mlp", "torso_mlp", "leg_mlp")
    for n, p in policy.named_parameters():
        if p.ndim != 2:
            continue
        if n.startswith("heads."):
            continue
        if n.startswith(enc) and (not enc_hidden or ".0." in n):
            continue
        if n == f"critic_mlp.{len(policy.critic_mlp) - 1}.weight":
            continue                      # the K-bin HL-Gauss output layer
        names.append(n)
    return names


def cpu_state(policy):
    """Checkpoint state_dict on CPU regardless of the training device."""
    return {k: v.detach().cpu() for k, v in policy.state_dict().items()}


def cpu_opt_state(opt):
    """Adam state_dict with tensors on CPU (load_state_dict moves them
    back to the params' device on resume). The lr/capturable
    normalization survives from the CUDA-graphs era (removed
    2026-07-25) — a no-op for eager runs, and it keeps this trainer
    byte-compatible with graphs-era checkpoints on file."""
    s = opt.state_dict()
    s["state"] = {k: {kk: (vv.detach().cpu() if torch.is_tensor(vv) else vv)
                      for kk, vv in st.items()}
                  for k, st in s["state"].items()}
    s["param_groups"] = [
        {**g, "lr": (float(g["lr"]) if torch.is_tensor(g["lr"]) else g["lr"]),
         "capturable": False}
        for g in s["param_groups"]]
    return s


# --resume flag resolution (2026-09-30, review bug 1b). Until today a
# resume took only lr/ent/schedule/z/arch from the checkpoint and every
# other knob from the CLI DEFAULTS — i.e. from whatever the recipe was
# on the day of the resume, not the day of the trunk: gamma, lambda,
# clip, N/T, the reward coefficients all silently switched at the
# branch point (cut_leaf.sh still passed the v16-era N 16384 / T 8 /
# lambda 0.98 by hand to paper over it). Now a flag the user did NOT
# type inherits the checkpoint's recorded value; a typed flag wins and,
# if it differs, is printed in the diff list. Legit leaf departures
# (--no-wall-despawn, --tau-rand, --ent-final, ...) keep working — they
# are simply typed.
#
# never inherited: where/how long/which host this launch runs, plus the
# leaf mode knobs (--ent-final is the LEAF's target; the checkpoint's
# value is the trunk's anneal FLOOR, which the schedule already carries
# as ent_floor)
RESUME_NEVER_INHERIT = frozenset({
    "run_dir", "resume", "init_from", "total_steps", "time_cap",
    "ckpt_every", "cooldown_updates", "ent_final", "bench", "device",
    "env", "ignore_arch_flags"})
# schedule state: from ckpt["schedule"], as before; a typed value that
# differs is IGNORED (loudly) — except a leaf's own --lr-final floor
RESUME_SCHEDULE = ("lr", "lr_final", "ent_coef", "ent_tau", "warmup_updates")
# the checkpoint's arch always wins (load_policy); a typed value that
# differs would be silently dropped, so it is refused unless
# --ignore-arch-flags. --gears / --game / --z-asinh have their own hard
# guards downstream (n_gears check, load_policy, the z cross-check).
RESUME_ARCH = frozenset({
    "embed", "trunk_dim", "n_blocks", "n_heads", "ffn_mult",
    "trunk_hidden", "note_geo", "hl_bins", "hl_vmax", "hl_sigma",
    "emb_ln", "value_residual", "final_ln", "trunk_arch"})
# the 2026-10-03 architecture A/B options: (args dest, arch key, value
# when the checkpoint arch predates the key). A checkpoint's arch
# decides them on --resume / --init-from (check_arch_options).
ARCH_OPTIONS = (("emb_ln", "emb_ln", False),
                ("value_residual", "value_residual", False),
                ("final_ln", "final_ln", False),
                ("trunk_arch", "trunk_arch", "simba"))


def check_arch_options(args, explicit, ckpt, what):
    """Make args agree with the checkpoint's arch for ARCH_OPTIONS (the
    checkpoint's network is what load_policy builds). A typed flag that
    differs is refused unless --ignore-arch-flags (then dropped, loudly);
    an untyped one is set from the arch — this also covers checkpoints
    whose args snapshot predates the flag (resolve_resume_args reports
    those as 'missing' and keeps the CLI value). Pure host logic; returns
    the [(dest, cli value, checkpoint value)] it changed."""
    arch = ckpt.get("arch", {})
    changed = []
    for dest, key, absent in ARCH_OPTIONS:
        ck = arch.get(key, absent)
        mine = getattr(args, dest)
        if mine == ck:
            continue
        if dest in explicit and not args.ignore_arch_flags:
            raise SystemExit(
                f"{what}: {flag_name(dest)} {mine!r} but the checkpoint's "
                f"arch has {key}={ck!r}; the checkpoint's network is what "
                f"loads — remove the flag, or pass --ignore-arch-flags")
        if dest in explicit:
            print(f"!!! --ignore-arch-flags: {flag_name(dest)} {mine!r} "
                  f"dropped — the checkpoint's arch has {ck!r} !!!",
                  flush=True)
        setattr(args, dest, ck)
        changed.append((dest, mine, ck))
    return changed
# resolved by their own arch-first inheritance in main()
RESUME_OWN_GUARD = frozenset({"z_asinh"})


def explicit_dests(argv):
    """The argparse dests the user actually typed on `argv`: a re-parse
    with every default SUPPRESSed leaves only the given flags in the
    namespace (abbreviations and --no-X forms included, since it is the
    same parser)."""
    p = build_parser()
    for a in p._actions:
        if a.dest != "help":
            a.default = argparse.SUPPRESS
    return set(vars(p.parse_args(argv)))


def resolve_resume_args(args, explicit, ckpt_args, sched, leaf):
    """Apply --resume inheritance to `args` IN PLACE. Pure host logic
    (no torch), unit-tested in tests/test_resume.py.

    ckpt_args = ckpt["args"] (the resolved snapshot the run recorded),
    sched = ckpt["schedule"], leaf = --cooldown-updates > 0. Returns a
    report dict of (dest, value, checkpoint value) lists:
      inherited — untyped flags whose CLI default differed (now = ckpt)
      diffs     — typed flags that differ from the checkpoint (kept)
      ignored   — typed schedule flags that differ (ckpt value used)
      arch      — typed arch flags that differ (the caller refuses)
      missing   — dests the checkpoint's snapshot predates (CLI kept)"""
    rep = {k: [] for k in ("inherited", "diffs", "ignored", "arch",
                           "missing")}
    # the schedule block's own fallbacks, unchanged from before
    sched_v = {"lr": sched.get("lr"), "lr_final": sched.get("lr_final"),
               "ent_coef": sched.get("ent_coef", args.ent_coef),
               "ent_tau": sched.get("ent_tau", 0),
               "warmup_updates": sched.get("warmup_updates",
                                           args.warmup_updates)}
    for dest in sorted(vars(args)):
        mine = getattr(args, dest)
        typed = dest in explicit
        if dest in RESUME_NEVER_INHERIT or dest in RESUME_OWN_GUARD:
            continue
        if dest in RESUME_SCHEDULE:
            ck = sched_v[dest]
            if ck is None:
                continue
            if typed and dest == "lr_final" and leaf:
                if mine != ck:
                    rep["diffs"].append((dest, mine, ck))
                continue
            if typed and mine != ck:
                rep["ignored"].append((dest, mine, ck))
            setattr(args, dest, ck)
            continue
        if dest not in ckpt_args:
            rep["missing"].append((dest, mine, None))
            continue
        ck = ckpt_args[dest]
        if dest == "tau_center" and not typed and "tau_center_body" in ckpt_args:
            # "default" is a NAME for TAU_BODY_DEFAULT, which tau_fit
            # moves between trunks: inherit the body the checkpoint
            # actually played (its resolved 16 floats), not the name
            body = [float(x) for x in ckpt_args["tau_center_body"]]
            if parse_tau_scale(mine) != body:
                rep["inherited"].append((dest, mine, "resolved body"))
            setattr(args, dest, ",".join(repr(x) for x in body))
            continue
        if typed:
            if mine != ck:
                rep["arch" if dest in RESUME_ARCH else "diffs"].append(
                    (dest, mine, ck))
        elif mine != ck:
            rep["inherited"].append((dest, mine, ck))
            setattr(args, dest, ck)
    return rep


def flag_name(dest):
    return "--" + dest.replace("_", "-")


# RNG streams in the checkpoint (2026-09-30, review bug 1a): the
# trainer reseeds from --seed at every launch, so before this an
# extension replayed update 0's sampling / shuffle stream from its
# branch point. Stored as tensors + plain ints so torch.load(...,
# weights_only=True) reads them back (no numpy objects in the pickle).
def rng_state(venv=None):
    import random
    np_s = np.random.get_state()
    st = {"torch": torch.get_rng_state(),
          "numpy": {"key": torch.from_numpy(np_s[1].astype(np.int64)),
                    "pos": int(np_s[2]), "has_gauss": int(np_s[3]),
                    "cached": float(np_s[4])},
          "python": [int(random.getstate()[0]),
                     list(random.getstate()[1])]}
    if torch.cuda.is_available() and torch.cuda.is_initialized():
        st["cuda"] = torch.cuda.get_rng_state_all()
    gen = getattr(venv, "_gen", None)   # the CUDA env's own generator
    if isinstance(gen, torch.Generator):
        st["env_gen"] = gen.get_state()
    return st


def restore_rng(st, venv=None, what=("torch", "cuda", "numpy", "python")):
    """Inverse of rng_state for the streams in `what` ("env_gen" is
    restored separately, before the env's first reset). Returns the
    names restored."""
    import random
    done = []
    if "torch" in what and "torch" in st:
        torch.set_rng_state(st["torch"])
        done.append("torch")
    if "cuda" in what and "cuda" in st and torch.cuda.is_available():
        cs = st["cuda"]
        if len(cs) == torch.cuda.device_count():
            torch.cuda.set_rng_state_all(cs)
        else:
            torch.cuda.set_rng_state(cs[0])
        done.append("cuda")
    if "numpy" in what and "numpy" in st:
        n = st["numpy"]
        np.random.set_state(("MT19937", n["key"].numpy().astype(np.uint32),
                             n["pos"], n["has_gauss"], n["cached"]))
        done.append("numpy")
    if "python" in what and "python" in st:
        v, key = st["python"]
        random.setstate((v, tuple(key), None))
        done.append("python")
    if "env_gen" in what and "env_gen" in st:
        gen = getattr(venv, "_gen", None)
        if isinstance(gen, torch.Generator):
            gen.set_state(st["env_gen"])
            done.append("env_gen")
    return done


def value_loss(policy, ret, ret_mu, ret_sig):
    """THE critic loss, unweighted (callers multiply by --vf-coef): the
    HL-Gauss cross-entropy of the last grad-enabled forward's value
    logits against the NORMALIZED return (T1: the critic learns
    (ret - mu) / sig). One function for the minibatch loss and the
    [diag] probe so the two can never drift apart again (review bug 3:
    the diag differentiated a 0.5*MSE on the HL expectation, a loss
    nothing trains)."""
    return hl_loss(policy._value_logits, (ret - ret_mu) / ret_sig,
                   policy.value_centers, policy.value_sigma)


def diag_index(batch, dn, seed=0x5EED):
    """Fixed row index for the [diag] minibatch over the FLAT T x N
    buffer (row = t*N + n). `f_obs[:dn]` was t=0 of envs 0..dn-1 only —
    one tick per env, and at a resume every env sits at an episode
    start. A seeded permutation spans all ticks and envs, is the same
    rows every diag of a process, and draws from its OWN generator, so
    the training RNG stream is untouched. Sorted for gather locality."""
    g = torch.Generator().manual_seed(seed)
    return torch.randperm(batch, generator=g)[:dn].sort().values


# ONE reward/body kwarg set for every env backend (2026-09-30). The five
# constructor calls in main() each spelled out their own list and
# drifted: procedural --env cuda never passed limit_coef / limit_arms /
# wall_despawn / tau_rand / tau_center, and --env rust could not take
# them (BeatVecEnv lacked the args), so every non-ranked run silently
# trained without the limit-rest price at the nominal body while the
# banner still announced --tau-rand. A backend that cannot simulate a
# term now says so (check_unsupported) instead of dropping it.
def env_kwargs(args):
    """Reward + body terms every backend must honor."""
    return dict(gears=args.gears, fall_const=args.fall_pen,
                foot_z_coef=args.foot_z_coef, foot_x_coef=args.foot_x_coef,
                foot_hinge_r=args.foot_hinge_r,
                foot_shape_coef=args.foot_shape_coef,
                energy_coef=args.energy_coef,
                limit_coef=args.limit_coef, limit_arms=args.limit_arms,
                wall_despawn=args.wall_despawn,
                self_contact_coef=args.self_contact_coef,
                tau_rand=args.tau_rand,
                tau_center=parse_tau_scale(args.tau_center))


# kernel-side episode-start / perturbation terms (help texts: "CUDA env
# only"); the Rust sim resets deterministically and never kicks
CUDA_ONLY_TERMS = ("push_prob", "push_vmax", "start_jitter",
                   "start_pose_jitter", "crouch_start_prob")


def cuda_env_kwargs(args, device):
    return dict(push_prob=args.push_prob, push_vmax=args.push_vmax,
                start_jitter=args.start_jitter,
                start_pose_jitter=args.start_pose_jitter,
                crouch_start_prob=args.crouch_start_prob,
                z_asinh=args.z_asinh, env_order=True, device=device)


def check_unsupported(args, names, backend):
    """Terms `backend` cannot simulate: off (0 / False) is trivially
    honored; the recipe DEFAULT gets a loud notice (these were always
    documented CUDA-only, and a CPU smoke must not need a dozen --X 0
    flags); any other value is refused — it was asked for and would be
    dropped."""
    dflt = vars(parse_args([]))
    loud, bad = [], []
    for n in names:
        v = getattr(args, n)
        if not v:
            continue
        (loud if v == dflt[n] else bad).append(f"{flag_name(n)} {v!r}")
    if bad:
        raise SystemExit(
            f"{backend} cannot simulate {', '.join(bad)} — pass 0 / the "
            f"--no- form, or train on --env cuda")
    if loud:
        print(f"!!! {backend}: NOT simulated (recipe defaults of "
              f"CUDA-only terms): {'  '.join(loud)} !!!", flush=True)


def make_env(args, N, device):
    """Build the training vec-env for the resolved args (the ranked /
    flux / procedural CUDA env, or the Rust twins)."""
    kw = env_kwargs(args)
    gen = dict(cols=tuple(args.cols), rows=tuple(args.rows),
               chart_len_s=args.chart_len,
               note_speed_min=args.note_speed_min,
               note_speed_max=args.note_speed_max)
    if args.env == "cuda":
        from beat_arena_env.cuda_env import CudaBeatVecEnv as VecCls
        ckw = cuda_env_kwargs(args, device)
        if args.ranked_maps:
            from beat_arena_env.ranked_pool import load_ranked_pool
            from beat_arena_env.ranked_pool import pool_max_steps
            from beat_arena_env.ranked_pool import train_catalog_path
            pool = load_ranked_pool(sample=args.chart_pool, seed=args.seed,
                                    device=device, lead_s=args.lead_s,
                                    mirror=args.mirror_augment,
                                    catalog=train_catalog_path(args.catalog))
            # whole-level episodes (the only mode since 2026-09-16): the
            # episode limit comes from the longest level in the pool
            pool_steps = pool_max_steps(pool)
            print(f"[ranked-pool] WHOLE-LEVEL episodes: max_steps "
                  f"{pool_steps} ({pool_steps / 30:.0f} s incl. tail + "
                  f"margin), {int(pool['n_notes'].shape[0])} levels in "
                  f"the pool", flush=True)
            return VecCls(N, pool=pool, max_steps=pool_steps,
                          ep_stats=args.ep_stats, seed=args.seed,
                          **kw, **ckw)
        if args.game == "flux":
            return VecCls(N, game="flux", circle_prob=args.circle_prob,
                          bar_prob=args.bar_prob, seed=args.seed,
                          chart_pool=args.chart_pool, **gen, **kw, **ckw)
        return VecCls(N, note_prob=args.note_prob,
                      extra_prob=args.extra_prob, dot_prob=args.dot_prob,
                      bomb_prob=args.bomb_prob, seed=args.seed,
                      chart_pool=args.chart_pool, **gen, **kw, **ckw)
    check_unsupported(args, CUDA_ONLY_TERMS, "--env rust")
    if args.game == "flux":
        # FluxVecEnv (flux_env.py) has no limit / wall / tau-draw args
        # yet although its VecBeatSim does; a fixed body goes in through
        # its set_tau_scale
        from beat_arena_env.flux_env import FluxVecEnv
        check_unsupported(args, ("limit_coef", "limit_arms",
                                 "wall_despawn", "tau_rand"),
                          "--env rust --game flux")
        fkw = {k: v for k, v in kw.items()
               if k not in ("limit_coef", "limit_arms", "wall_despawn",
                            "tau_rand", "tau_center")}
        venv = FluxVecEnv(N, circle_prob=args.circle_prob,
                          bar_prob=args.bar_prob, seed=args.seed,
                          **gen, **fkw)
        venv.set_tau_scale(kw["tau_center"])
        return venv
    return BeatVecEnv(N, note_prob=args.note_prob,
                      extra_prob=args.extra_prob, dot_prob=args.dot_prob,
                      bomb_prob=args.bomb_prob, seed=args.seed,
                      **gen, **kw)


def main():
    args = parse_args()
    # --resume: load FIRST and resolve the flags before anything below
    # derives from them (the flux/speed defaults, CLIP, --bench's step
    # count, the run dir). See resolve_resume_args.
    resume_ckpt = None
    resume_rep = None
    explicit = explicit_dests(sys.argv[1:])
    if args.resume:
        assert not args.init_from, "--resume and --init-from are exclusive"
        resume_ckpt = torch.load(args.resume, map_location="cpu",
                                 weights_only=True)
        if "optimizer" not in resume_ckpt:
            raise SystemExit(
                "--resume needs a checkpoint carrying optimizer + schedule "
                "state (saved by this trainer version); older checkpoints "
                "can only be --init-from warm-starts")
        sched = resume_ckpt["schedule"]
        if sched.get("lr_decay", "wsd") != "wsd":
            raise SystemExit(
                f"--resume: pre-WSD checkpoint (schedule "
                f"'{sched['lr_decay']}') — the old schedules were removed "
                f"2026-08-11; old-era checkpoints can only be --init-from "
                f"warm-starts")
        ck_args = resume_ckpt.get("args")
        if not ck_args:
            print("!!! resume: checkpoint carries no args snapshot — "
                  "nothing to inherit, every non-schedule flag is CLI "
                  "(the pre-2026-09-30 behavior) !!!", flush=True)
            ck_args = {}
        resume_rep = resolve_resume_args(
            args, explicit, ck_args, sched, leaf=args.cooldown_updates > 0)
        if resume_rep["inherited"]:
            print("resume: inherited from the checkpoint (differs from "
                  "today's CLI default): " + "  ".join(
                      f"{flag_name(d)} {c!r} (default {m!r})"
                      for d, m, c in resume_rep["inherited"]), flush=True)
        if resume_rep["diffs"]:
            print(f"!!! resume: {len(resume_rep['diffs'])} flag(s) given "
                  f"on the command line DIFFER from the checkpoint's run "
                  f"(the command line wins):", flush=True)
            for d, m, c in resume_rep["diffs"]:
                print(f"!!!   {flag_name(d)} {m!r}   (checkpoint {c!r})",
                      flush=True)
        if resume_rep["ignored"]:
            print("!!! resume: schedule state comes from the checkpoint — "
                  "IGNORED: " + "  ".join(
                      f"{flag_name(d)} {m!r} (using {c!r})"
                      for d, m, c in resume_rep["ignored"]) + " !!!",
                  flush=True)
        if resume_rep["missing"] and ck_args:
            # today's default may not be what the trunk trained with
            # (e.g. --tau-rand 0.25, --limit-arms on): say so loudly
            print("!!! resume: checkpoint's args predate "
                  + ", ".join(f"{flag_name(d)} (using today's {m!r})"
                              for d, m, _ in resume_rep["missing"])
                  + " — type the trunk's value if it trained without "
                    "these !!!", flush=True)
        if (ck_args and "tau_center" not in explicit
                and "tau_center_body" not in ck_args):
            print(f"!!! resume: checkpoint predates tau_center_body — "
                  f"--tau-center {args.tau_center!r} resolves to TODAY's "
                  f"body, which may not be the one the trunk played !!!",
                  flush=True)
        if resume_rep["arch"]:
            _a = "  ".join(f"{flag_name(d)} {m!r} (checkpoint {c!r})"
                           for d, m, c in resume_rep["arch"])
            if not args.ignore_arch_flags:
                raise SystemExit(
                    f"--resume: arch flag(s) differ from the checkpoint: "
                    f"{_a}. The checkpoint's arch always wins on a resume, "
                    f"so these would be silently dropped — remove them, "
                    f"or pass --ignore-arch-flags to continue with the "
                    f"checkpoint's arch")
            print(f"!!! --ignore-arch-flags: {_a} dropped — the "
                  f"checkpoint's arch wins !!!", flush=True)
            for d, _, c in resume_rep["arch"]:
                setattr(args, d, c)
        # a trunk at (or past) its planned length with no --total-steps
        # extension would run zero updates and then crash at the final
        # save on an undefined `update` — refuse up front instead
        if args.cooldown_updates <= 0:
            _at = int(resume_ckpt["update"])
            _goal = int(sched["num_updates"])
            if "total_steps" in explicit:     # see the WSD TRUNK EXTENDED note
                _goal = max(_goal, args.total_steps
                            // (args.num_envs * args.horizon))
            if _at >= _goal:
                raise SystemExit(
                    f"--resume: trunk checkpoint is at update {_at:,} of "
                    f"{int(sched['num_updates']):,} — nothing to run. "
                    f"Extend with --total-steps > "
                    f"{_at * args.num_envs * args.horizon:,}, or branch a "
                    f"leaf with --cooldown-updates N")
    global CLIP
    CLIP = float(args.clip)
    if CLIP != 0.3:
        print(f"--clip {CLIP}: joint PPO clip epsilon (default 0.3)")
    # env v21 (2026-08-16) made contact the only training mode; the weld
    # harness left the engines 2026-09-27 (issue #2), so there is no
    # other physics to select.
    if not args.foot_contact:
        sys.exit("--no-foot-contact: weld mode was removed (2026-09-27); "
                 "contact physics is the only mode")
    # game mode (SPEC.md §Flux): per-mode generator speed defaults; Flux
    # has no real maps, so the ranked pool and its ep-stats walk are off
    _speeds = (10.0, 18.0) if args.game == "flux" else (2.0, 8.0)
    if args.note_speed_min is None:
        args.note_speed_min = _speeds[0]
    if args.note_speed_max is None:
        args.note_speed_max = _speeds[1]
    if args.game == "flux":
        for _flag in ("ranked_maps", "ep_stats"):
            if getattr(args, _flag):
                print(f"[flux] --{_flag.replace('_', '-')} does not apply "
                      f"to the flux game (procedural charts only): off",
                      flush=True)
                setattr(args, _flag, False)
        if args.chart_pool == 999999:
            # the ranked-pool default; the procedural pool's size
            args.chart_pool = 65536
    if args.bench:
        args.total_steps = args.num_envs * args.horizon * 4
    init_ckpt = None
    if args.init_from and not args.resume:
        init_ckpt = torch.load(args.init_from, map_location="cpu",
                               weights_only=True)
    # architecture A/B options: the checkpoint's arch decides (before the
    # args snapshot below, so args.json records the network that runs)
    for _what, _ck in (("--resume", resume_ckpt), ("--init-from", init_ckpt)):
        if _ck is not None:
            check_arch_options(args, explicit, _ck, _what)
    if args.resume:
        # lr / lr_final / ent_coef / ent_tau / warmup_updates were set
        # from ckpt["schedule"] by resolve_resume_args above: the trunk
        # entropy curve is checkpoint state, like lr — an extension must
        # continue the exact curve, and a leaf must start from the
        # trunk's live value (--ent-final stays CLI: the leaf's target;
        # a typed --lr-final is the leaf's floor)
        if args.run_dir is None:      # continue in the checkpoint's run dir
            args.run_dir = os.path.dirname(os.path.abspath(args.resume))
    if args.run_dir is None:
        args.run_dir = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "runs", "batch_" + datetime.now().strftime("%Y%m%d-%H%M%S"))
    os.makedirs(args.run_dir, exist_ok=True)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(
        args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    if args.env is None:
        # env v10 (P5): the CUDA kernel is the training backend again;
        # rust stays the CPU path and the parity reference
        args.env = "cuda" if device.type == "cuda" else "rust"
        print(f"--env not given: defaulting to '{args.env}'", flush=True)

    N, T = args.num_envs, args.horizon
    batch = N * T
    mb_size = batch // args.minibatches
    assert args.mb_accum >= 1 and mb_size % args.mb_accum == 0, (
        f"--mb-accum {args.mb_accum} must divide minibatch {mb_size}")
    assert 0.0 < args.adv_filter <= 1.0, (
        f"--adv-filter {args.adv_filter} must be in (0, 1]")
    # round to whole minibatches so the slice loop needs no tail case
    train_n = max(mb_size,
                  int(round(batch * args.adv_filter / mb_size)) * mb_size)
    train_n = min(train_n, batch)
    adv_filter_on = train_n < batch

    # --fp16 update is the DEFAULT (2026-07-25); downgrade quietly where
    # it cannot apply instead of crashing paths that never asked for it
    if args.fp16 != "off" and device.type != "cuda":
        print("--fp16: CPU device, forcing off", flush=True)
        args.fp16 = "off"
    fp16_update = args.fp16 in ("update", "all")
    # device-side loss-scaling state: the loss-scale machinery adds NO
    # host syncs to the minibatch loop (an overflow zeroes the grads
    # device-side; the audit rides the existing once-per-update print
    # sync). The loop is NOT sync-free overall: the --target-kl probe
    # (default on, 1.0) reads float(kl) every --kl-check-every
    # minibatches — one sync per probe, 7 per update at the recipe's
    # 32 kept minibatches (mb64, --adv-filter 0.5) probed every 4. The scale itself is a
    # device tensor since 2026-08-02 so it can adapt PER MINIBATCH
    # (halve on overflow, x2 after GROWTH_INTERVAL clean steps, capped
    # at the initial value) without ever syncing — the old host-float
    # scale could only move at the once-per-update audit, so all 8
    # minibatches of an update overflowed before it reacted.
    fp16_scale = torch.full((), args.fp16_scale, device=device)
    scale_cap = float(args.fp16_scale)
    clean_steps = torch.zeros((), device=device)
    found_inf = torch.zeros(1, device=device)
    inv_scale = torch.zeros(1, device=device)
    inf_count = torch.zeros(1, device=device)
    ratio_diag = torch.zeros((), device=device)
    logr_max = torch.zeros((), device=device)  # max|log ratio| this update
    # --- Tier 0.3 diagnostics (2026-09-02 static audit, C §1.2 / §2.2 /
    # §5). ALL device-side, accumulated in the minibatch loop with
    # add_/copy_ and read exactly once at the existing per-update print
    # sync: they add no host sync to the minibatch loop (whose only
    # syncs are the --target-kl probes, see above). Cost is
    # ~11 kernels on each of 4 micro-batches (first/last minibatch) plus
    # 2 per optimizer step, against ~46k launches/update.
    #   kl_*  = k3 approx-KL (Schulman, http://joschu.net/blog/kl-approx),
    #           mean of (r - 1) - log r: unbiased, non-negative, far
    #           lower variance than -log r at the ratios PPO lives at.
    #   cf_*  = clipfrac, fraction of rows with |r - 1| > CLIP — the
    #           share of the minibatch contributing ZERO policy gradient.
    #           first vs last minibatch brackets the within-update drift
    #           (64 Adam steps at production shape).
    kl_first = torch.zeros((), device=device)
    kl_last = torch.zeros((), device=device)
    cf_first = torch.zeros((), device=device)
    cf_last = torch.zeros((), device=device)
    # 1.4 / C §5 item 7: sd(kept adv) / sd(full adv) when filtering,
    # written once per update by adv_filter_stats, read at the sync.
    af_ratio = torch.zeros((), device=device)
    # pre-clip global grad norm and the clip coefficient it produces,
    # summed over the update's optimizer steps (C §2.2: total_norm was
    # computed and thrown away, so nothing ever said whether the 0.5
    # clip binds). clipc < 1 means it bound on that step.
    gnorm_sum = torch.zeros((), device=device)
    clipc_sum = torch.zeros((), device=device)
    opt_steps = 0                      # host-side counter, no sync
    # full-batch per-head entropy, (29,). Taken from the t=0 rollout
    # forward (see the rollout loop) so it is over ALL N envs and is
    # therefore comparable across --adv-filter settings, unlike the
    # printed `ent`, which is the last minibatch's KEPT rows (C §5.4).
    ent_head = torch.zeros(A_DIM, device=device)

    tensor_env = args.env == "cuda"
    # (--graph-rollout / rollout_graph.py REMOVED 2026-08-31: measured
    # 0.99x eager for the graphed forward at N=32,768 and an identical
    # end-to-end phase split, so the capture/refresh machinery and its
    # eager-divergent sampling RNG stream bought nothing at production
    # width. The 1.20x it was adopted for in 2026-07-26 was a small-N
    # launch-overhead win. The BENCH keeps its own separate graphed
    # argmax — cuda_graphs.py, bench_scoresaber --no-graphs — which is
    # untouched, and bench JSONs still stamp forward=graphed/eager.)

    # Config provenance (2026-07-26, after the gen3 postmortem: a
    # missing --colors flag survived a 5B run and a 4-arm bisect because
    # nothing recorded what any run was launched with). Written AFTER
    # every downgrade above so it records the RESOLVED config, not the
    # requested one; the same snapshot is embedded in every checkpoint.
    if args.fp16_accum is None:
        # default "all" (2026-09-27, issue #3 A/B: +5.5% e2e, acc at
        # parity at update 500, r1 ~10x lower). A --resume (trunk
        # continuation or cooldown leaf) keeps the mode its checkpoint
        # was trained with (arch; absent = "off", the pre-#3 path) so a
        # leaf never switches numerics mid-run; fresh runs and
        # --init-from warm starts take the default.
        if resume_ckpt is not None:
            args.fp16_accum = resume_ckpt.get("arch", {}).get(
                "fp16_accum", "off")
        else:
            args.fp16_accum = "all"
    if args.fused_ffn is None:
        # like --fp16-accum: a resume never switches numerics mid-run
        # (absent = the pre-I13 unfused path); fresh runs and --init-from
        # take the default, 'on' since 2026-09-30
        args.fused_ffn = (resume_ckpt.get("arch", {}).get("fused_ffn", "off")
                          if resume_ckpt is not None else "on")
    args_snapshot = {k: v for k, v in sorted(vars(args).items())}
    # the body --tau-center resolved to, so a resume inherits the body
    # itself, not a name whose value tau_fit moves (review 2026-09-30)
    args_snapshot["tau_center_body"] = parse_tau_scale(args.tau_center)
    # a resume IN PLACE (the trunk's own run dir) must not overwrite the
    # trunk's original args.json — the launch record harvest_obs /
    # metrics_backfill read. Each resume there gets its own
    # args_resume_NNNNN.json (NNNNN = the update it resumed at; _2, _3
    # for repeats) carrying the inheritance report as well.
    args_path = os.path.join(args.run_dir, "args.json")
    args_rec = {"argv": sys.argv[1:], "resolved": args_snapshot}
    if resume_ckpt is not None:
        args_rec.update(
            resume_from=args.resume,
            explicit=sorted(explicit),
            # {dest: [cli value (default if untyped), checkpoint value]}
            **{k: {d: [m, c] for d, m, c in v}
               for k, v in resume_rep.items()
               if k in ("inherited", "diffs", "ignored")})
        if os.path.exists(args_path):
            _base = os.path.join(
                args.run_dir, f"args_resume_{int(resume_ckpt['update']):05d}")
            args_path, _k = _base + ".json", 2
            while os.path.exists(args_path):
                args_path, _k = f"{_base}_{_k}.json", _k + 1
    with open(args_path, "w", encoding="utf-8") as f:
        json.dump(args_rec, f, indent=1, default=str)
    print(f"argv: {' '.join(sys.argv[1:]) or '(bare)'}", flush=True)
    print(f"config: {args_snapshot}", flush=True)

    # Wide-N hazard (2026-08-03): every run driver hardcodes the SHAPE
    # (--num-envs 1024 --horizon 128) because the tuned hypers and the
    # checkpoint cadences are calibrated to batch 131,072. Raising N
    # alone silently rescales the batch, so say so loudly. A warning,
    # not an error: --bench and the micro-shape tools are legitimately
    # off-recipe (and skip this entirely under --bench).
    if args.ranked_maps and args.env != "cuda":
        raise SystemExit("--ranked-maps requires --env cuda")
    if args.z_asinh is None:
        # recipe default (2026-09-21): C = 8 m for a fresh run; a resumed
        # or warm-started run keeps the checkpoint's encoding
        _ck = resume_ckpt if resume_ckpt is not None else init_ckpt
        if _ck is not None:
            args.z_asinh = float(_ck.get("arch", {}).get(
                "z_asinh", _ck.get("args", {}).get("z_asinh", 0.0)))
            print(f"--z-asinh not given: inherited {args.z_asinh:g} from "
                  f"the checkpoint", flush=True)
        else:
            args.z_asinh = 8.0
    if not args.z_asinh > 0:
        raise SystemExit(
            f"--z-asinh must be > 0 (metres), got {args.z_asinh:g}: the "
            f"legacy clamp z encoding (0, pre-v52 checkpoints) was removed")
    # env v48: the rust twin reads BEAT_Z_ASINH once per process — set it
    # before any sim exists so --env rust and the CUDA env agree
    os.environ["BEAT_Z_ASINH"] = repr(float(args.z_asinh))
    for _name, _ck in (("--resume", resume_ckpt),
                       ("--init-from", init_ckpt)):
        if _ck is None:
            continue
        for _key, _flag, _mine in (("z_asinh", "--z-asinh", args.z_asinh),):
            _prev = _ck.get("arch", {}).get(
                _key, _ck.get("args", {}).get(_key, 0.0))
            if float(_prev) != float(_mine):
                raise SystemExit(
                    f"{_name}: checkpoint was trained with {_flag} {_prev} "
                    f"but this run asks for {_mine}; the obs z lanes "
                    f"would not match (pass the checkpoint's value)")
    venv = make_env(args, N, device)

    # RNG streams (review bug 1a): restored only when this launch keeps
    # the checkpoint's seed — a typed different --seed means "reseed"
    # (e.g. a second leaf off the same trunk). The env's own generator
    # goes back NOW, before its first reset draws the start states; the
    # global torch/CUDA/numpy/python streams go back right before the
    # update loop, after every construction below has consumed its
    # seed-stream draws (policy init, pool shuffles), so update k+1 of a
    # resumed run draws exactly what update k+1 of the uninterrupted run
    # did. (The env STATE still restarts fresh, so trajectories — and
    # hence losses — differ; only the streams are continued.)
    rng_ck = None
    if resume_ckpt is not None:
        if "rng" not in resume_ckpt:
            print("!!! resume: checkpoint predates RNG-state saving "
                  "(2026-09-30) — sampling/shuffle streams restart from "
                  f"--seed {args.seed}, replaying update 0's stream !!!",
                  flush=True)
        elif args.seed != resume_ckpt.get("args", {}).get("seed", args.seed):
            print(f"resume: --seed {args.seed} differs from the "
                  f"checkpoint's — RNG streams reseeded, not restored",
                  flush=True)
        else:
            rng_ck = resume_ckpt["rng"]
            restore_rng(rng_ck, venv, what=("env_gen",))

    if resume_ckpt is not None:
        policy = load_policy(resume_ckpt, game=args.game)
        if policy.upgraded_from:   # I07: its optimizer state is the old shape
            raise SystemExit(f"--resume: {policy.upgraded_from} checkpoint "
                             "(obs 624); warm-start it with --init-from")
    elif args.init_from:
        policy = load_policy(init_ckpt, game=args.game)
        print(f"warm-started from {args.init_from} (arch {policy.arch})",
              flush=True)
    else:
        policy = NotePolicy(embed=args.embed, trunk_dim=args.trunk_dim,
                            n_blocks=args.n_blocks, n_heads=args.n_heads,
                            n_gears=args.gears, ffn_mult=args.ffn_mult,
                            note_geo=args.note_geo,
                            trunk_hidden=args.trunk_hidden,
                            value_bins=args.hl_bins,
                            value_vmax=args.hl_vmax,
                            value_sigma=args.hl_sigma,
                            z_asinh=args.z_asinh, game=args.game,
                            emb_ln=args.emb_ln,
                            value_residual=args.value_residual,
                            final_ln=args.final_ln,
                            trunk_arch=args.trunk_arch,
                            # per-joint gear counts of this build (all
                            # args.gears on main; 2026-10-04)
                            head_gears=joint_gear_counts(args.gears))
        if args.depth_init:
            from policy import apply_depth_init
            print(f"--depth-init: {apply_depth_init(policy)}", flush=True)
    if args.depth_init and (resume_ckpt is not None or args.init_from):
        print("--depth-init: ignored (weights come from the checkpoint)",
              flush=True)
    # accumulate mode: the CLI (resolved above — inherited from the
    # checkpoint when not given) wins, and the arch records it
    policy.fp16_accum = args.fp16_accum
    policy.arch["fp16_accum"] = args.fp16_accum
    policy.fused_ffn = policy.arch["fused_ffn"] = args.fused_ffn
    print(f"HL-Gauss critic: {policy.value_bins} bins over "
          f"[-{policy.arch['value_vmax']:g}, {policy.arch['value_vmax']:g}] "
          f"(normalized return), sigma {policy.value_sigma_bins:g} bins",
          flush=True)
    if policy.n_gears != args.gears:
        raise SystemExit(f"--gears {args.gears} but the checkpoint's heads "
                         f"are n_gears={policy.n_gears} — the env and "
                         f"policy action tables must match")
    # per-joint gear counts (2026-10-04): the heads must be this build's
    # (load_policy checks checkpoints; this covers every path)
    if tuple(policy.head_sizes) != tuple(venv.joint_gears):
        raise SystemExit(f"policy head sizes {list(policy.head_sizes)} != "
                         f"the env's per-joint gear counts "
                         f"{list(venv.joint_gears)}")
    n_params = sum(p.numel() for p in policy.parameters())
    print(f"policy arch {policy.arch} — {n_params:,} params", flush=True)
    n_trunk = sum(p_.numel() for n_, p_ in policy.named_parameters()
                  if n_.startswith("trunk"))
    print(f"readout trunk: {policy.arch['trunk_arch']} "
          f"(hidden {policy.arch.get('trunk_hidden')}) — {n_trunk:,} params",
          flush=True)
    _opts_on = [k for k in ("emb_ln", "value_residual", "final_ln")
                if policy.arch.get(k)]
    if _opts_on or policy.arch["trunk_arch"] != "simba":
        print(f"arch options: {', '.join(_opts_on) or '-'}; trunk "
              f"{policy.arch['trunk_arch']}", flush=True)
    policy = policy.to(device)
    policy.fp16 = args.fp16 == "all"
    if args.fp16 != "off":
        print(f"--fp16 {args.fp16}: fp16-storage activations, fp32 "
              f"masters, dynamic loss scale (init {args.fp16_scale:.0f}, "
              f"halve/overflow-mb, x2/{FP16_GROWTH_INTERVAL:.0f} clean), "
              f"GEMM accumulate: {args.fp16_accum}", flush=True)
        # flat fp32 masters + flat fp16 replica mirror (2026-07-28):
        # every replica refresh (once per update for the graphed
        # rollout, once per minibatch after opt.step) becomes ONE cast
        # kernel, and the fused-head cache becomes a static view. Must
        # precede graph capture; p.data views never move afterward
        # (checkpoints/Adam unchanged — Adam mutates the views in
        # place).
        if device.type == "cuda":
            policy.flatten_fp16()
            print("--fp16: flat master/replica buffers (1-kernel "
                  "refresh)", flush=True)
    graph_roll = None
    if args.graph_rollout:
        if (not tensor_env or device.type != "cuda"
                or args.fp16 == "update"):
            print("--graph-rollout: needs --env cuda on a CUDA device "
                  "and not --fp16 update (the captured kernels cannot "
                  "follow the fp16 mode flip), forcing off", flush=True)
        else:
            try:
                from rollout_graph import GraphedRollout
                graph_roll = GraphedRollout(policy, venv)
                print(f"--graph-rollout: rollout step captured as one "
                      f"CUDA graph ({'fp16' if policy.fp16 else 'fp32'} "
                      f"rollout; t=0 eager, t=1..T-1 replayed; sampling "
                      f"RNG stream differs from eager)", flush=True)
            except Exception as e:
                graph_roll = None
                print(f"--graph-rollout: capture failed, eager rollout "
                      f"[{type(e).__name__}: {e}]", flush=True)
    # (--fused-update was retired 2026-08-13 with the 138k arch: the
    # packed kernels were hard-wired to embed 64 / trunk 128 and the
    # eager fp16 path was the production path throughout v16.)
    # fused=True: one multi-tensor kernel instead of the foreach path's
    # ~10 launches per step (77 elementwise calls/minibatch measured in
    # the 2026-07-25 profile). Same per-element math, not guaranteed
    # bitwise vs foreach.
    fused_adam = device.type == "cuda"
    # T1 critic lr split: the critic params (the adapool critic_*
    # modules) form their own group with an lr multiplier the
    # schedule loop re-applies every update via the "lr_mult" key
    # (extra param_group keys survive state_dict round trips).
    # hybrid Muon: the hidden matrices leave the Adam groups
    muon_names = set(muon_param_names(policy, args.muon_enc_hidden))
    # frozen params (the SimBaV2 trunk's zero biases) join no optimizer
    crit_p = [p for n, p in policy.named_parameters()
              if n.startswith("critic_")
              and n not in muon_names and p.requires_grad]
    main_p = [p for n, p in policy.named_parameters()
              if not n.startswith("critic_")
              and n not in muon_names and p.requires_grad]
    opt = torch.optim.Adam(
        [{"params": main_p, "lr": args.lr, "lr_mult": 1.0},
         {"params": crit_p, "lr": args.lr * args.critic_lr_mult,
          "lr_mult": args.critic_lr_mult}],
        lr=args.lr, fused=fused_adam)
    if resume_ckpt is not None and not resume_ckpt.get("muon"):
        # checked before the Adam load: an Adam-era checkpoint's Adam
        # groups also hold the hidden matrices and would fail there
        raise SystemExit("--resume needs a checkpoint saved by a muon "
                         "run (Adam-era ones: --init-from)")
    if resume_ckpt is not None:
        try:
            # stamp the fused flag into the SAVED groups before loading:
            # load_state_dict places the 'step' state tensors by the
            # saved group's fused/capturable flags, and fused Adam needs
            # them on the param device (a foreach-era checkpoint would
            # otherwise leave them on CPU and crash the first step)
            opt_sd = resume_ckpt["optimizer"]
            for g in opt_sd["param_groups"]:
                g["fused"] = fused_adam
            opt.load_state_dict(opt_sd)
        except ValueError as e:
            raise SystemExit(
                f"--resume optimizer state does not match this policy "
                f"arch (pre-scale checkpoints were retired). [{e}]")
        # load_state_dict REPLACES the group hyperparams with the saved
        # ones, lr_mult included, so --critic-lr-mult was silently the
        # checkpoint's on every resume (review bug 1b2) while the Muon
        # groups (built fresh below) took the CLI's — the critic's Adam
        # and Muon halves could disagree. Re-stamp this run's multipliers
        # (inherited unless typed); the schedule loop sets
        # lr = lr_now * lr_mult from the first update on.
        _ck_mult = [g.get("lr_mult", 1.0) for g in opt.param_groups]
        for g, m_ in zip(opt.param_groups, (1.0, args.critic_lr_mult)):
            g["lr_mult"] = m_
            g["lr"] = args.lr * m_
        if _ck_mult[1] != args.critic_lr_mult:
            print(f"resume: critic lr multiplier {_ck_mult[1]:g} -> "
                  f"{args.critic_lr_mult:g} (Adam and Muon critic "
                  f"groups)", flush=True)

    m_main = [p for n, p in policy.named_parameters()
              if n in muon_names
              and not n.startswith("critic_")]
    m_crit = [p for n, p in policy.named_parameters()
              if n in muon_names
              and n.startswith("critic_")]
    muon = Muon(
        [{"params": m_main, "lr": args.lr, "lr_mult": 1.0},
         {"params": m_crit, "lr": args.lr * args.critic_lr_mult,
          "lr_mult": args.critic_lr_mult}],
        momentum=args.muon_momentum)
    if resume_ckpt is not None:
        muon.load_state_dict(resume_ckpt["muon"])
    n_m = sum(p.numel() for p in m_main + m_crit)
    n_a = sum(p.numel() for g in opt.param_groups for p in g["params"])
    print(f"optimizer: Muon on {len(muon_names)} hidden "
          f"matrices ({n_m:,} params; momentum {args.muon_momentum}, "
          f"5-step tuned NS schedule, RMS-matched "
          f"0.2*sqrt(max(A,B))), Adam on the remaining "
          f"{n_a:,} params", flush=True)
    # every lr-carrying group (Adam's two, plus Muon's two): the
    # schedule loop and --weight-decay walk this list
    all_groups = list(opt.param_groups) + muon.param_groups

    # --weight-decay: per optimizer group, the ndim >= 2 params it will
    # decay at that group's live lr (see the flag's comment)
    wd = float(args.weight_decay)
    wd_groups = []
    # --trunk-arch simbav2: its W_in/W1/W2 are re-projected onto unit
    # rows after every step (policy.project_trunk_weights), so decay
    # would be undone — leave them out of it
    proj_lins = policy.trunk_projected()
    proj_ids = {id(m_.weight) for m_ in proj_lins}
    if wd > 0.0:
        for g in all_groups:
            ps = [p_ for p_ in g["params"] if p_.ndim >= 2
                  and id(p_) not in proj_ids]
            if ps:
                wd_groups.append((g, ps))
        n_wd = sum(p_.numel() for _, ps in wd_groups for p_ in ps)
        print(f"--weight-decay {wd:g}: decoupled decay on "
              f"{n_wd:,} params in {len(wd_groups)} group(s); "
              f"time constant {1.0 / (args.lr * wd):,.0f} optimizer "
              f"steps at lr {args.lr:g} (critic x{args.critic_lr_mult:g})",
              flush=True)
    # parameter norms for the per-update line/metrics (actor vs critic),
    # read at the print sync like everything else
    _pn_actor_params = [p_ for n_, p_ in policy.named_parameters()
                        if not n_.startswith("critic_")]
    _pn_critic_params = [p_ for n_, p_ in policy.named_parameters()
                         if n_.startswith("critic_")]

    @torch.no_grad()
    def param_norms():
        pa = torch.stack(torch._foreach_norm(_pn_actor_params)).norm()
        pc = (torch.stack(torch._foreach_norm(_pn_critic_params)).norm()
              if _pn_critic_params else torch.zeros((), device=device))
        return pa, pc

    # Flat grad buffer (2026-07-28): every p.grad is a view into ONE
    # contiguous buffer, so zero/unscale/overflow-mul/nan_to_num/clip
    # run as single-tensor kernels instead of 66-tensor loops (~880
    # launches/update -> ~30). Autograd accumulates in place into
    # existing grads, so the views persist; a one-time guard below
    # verifies that.
    flat_grad = None
    diag_groups = None
    if device.type == "cuda":
        f_params = list(policy.parameters())
        flat_grad = torch.zeros(sum(p.numel() for p in f_params),
                                device=device)
        off = 0
        name_off = {}
        for (n_, _), p_ in zip(policy.named_parameters(), f_params):
            p_.grad = flat_grad[off:off + p_.numel()].view_as(p_)
            name_off[n_] = off
            off += p_.numel()
        flat_guard = True
        # diag slices over the flat buffer: params are registered
        # encoder -> blocks -> trunk -> heads -> critic, so the SHARED
        # region is everything before the first head. Per-group
        # boundaries feed the [diag] per-layer grad norms.
        shared_end = name_off.get("heads.0.weight", off)
        # the trunk group's first param by NAME PREFIX (trunk_proj.weight)
        # (--final-ln's LayerNorm, registered right before the trunk, is
        # the readout's input norm: it joins the trunk group)
        trunk_start = min((o_ for n_, o_ in name_off.items()
                           if n_.startswith(("trunk", "final_ln"))),
                          default=shared_end)
        diag_groups = [("enc", 0, name_off.get("blocks.0.ln_attn.weight",
                                               shared_end))]
        for i in range(len(policy.blocks)):
            lo = name_off.get(f"blocks.{i}.ln_attn.weight")
            hi = name_off.get(f"blocks.{i + 1}.ln_attn.weight", trunk_start)
            if lo is not None:
                diag_groups.append((f"blk{i}", lo, hi))
        diag_groups.append(("trunk", trunk_start, shared_end))

    obs = torch.as_tensor(venv.reset(), dtype=torch.float32, device=device)

    # --fp16 all stores the rollout obs buffer half: every consumer
    # casts obs to fp16 anyway (policy.py notes.half()/sabers.to(dt)),
    # and fp32->fp16 round-to-nearest at the buffer store produces the
    # SAME BITS as the same cast at use — so this halves the biggest
    # tensor in the pipeline (150 MB at N=4096) for free (2026-07-28).
    # 'update'/'off' keep fp32 (the fp32 rollout forward reads b_obs's
    # source values at full precision there).
    obs_dt = torch.float16 if args.fp16 == "all" else torch.float32
    b_obs = torch.zeros((T, N, OBS_DIM), device=device, dtype=obs_dt)
    # uint8 actions (2026-09-02, Tier 1.3 of the 09-02 audit A §3 /
    # B §5): gear indices live in [0, n_gears) with n_gears <= 7, so
    # the int64 (T, N, 29) buffer (243 MB at production shape, plus the
    # same again on the per-epoch perm copy) was 8x oversized. The
    # per-tick store casts (exact, integers); the consumers cast back
    # to int64 per micro-batch (policy.get_action_and_value — gather
    # needs a long index). Bit-identical.
    b_act = torch.zeros((T, N, A_DIM), dtype=torch.uint8, device=device)
    b_logp = torch.zeros((T, N), device=device)
    b_rew = torch.zeros((T, N), device=device)
    b_done = torch.zeros((T, N), device=device)
    # row T holds the bootstrap value: b_val[1:] IS the shifted value
    # view the GAE deltas need, killing the per-update torch.cat
    # (2026-07-28; bit-identical, values unchanged)
    b_val = torch.zeros((T + 1, N), device=device)
    b_adv = torch.zeros((T, N), device=device)
    b_ret = torch.zeros((T, N), device=device)
    gae_zero = torch.zeros(N, device=device)  # t = T-1 scan seed

    # T1 value-target normalization (2026-08-19): the policy's value
    # output is the NORMALIZED prediction; b_val stores UNNORMALIZED
    # V = v_hat*sig+mu so GAE/EV read real return units. Stats are
    # device tensors (no syncs; floats only at the ckpt-save sync).
    ret_mu = torch.zeros((), device=device)
    ret_var = torch.ones((), device=device)
    ret_sig = torch.ones((), device=device)
    value_norm_init = True
    # --init-from too (review bug 2): the warm-started critic outputs
    # values NORMALIZED by its run's (mu, sig). Scaling them with the
    # fresh mu=0 / sig=1 made the first rollout's b_val garbage, and
    # value_norm_init then copied its stats from returns BUILT on that
    # garbage. The pair is only meaningful with the critic it trained,
    # and the critic head (HL-Gauss bins / support) comes from the same
    # checkpoint's arch (load_policy; the --hl-* flags are ignored), so
    # the stats and the support always agree. A recipe change on the
    # warm start (gamma, reward coefs) moves the return scale; the EMA
    # re-fits at --value-norm-beta from the checkpoint's stats instead
    # of from garbage.
    vn_ck = resume_ckpt if resume_ckpt is not None else init_ckpt
    vn_name = "resume" if resume_ckpt is not None else "init-from"
    if vn_ck is not None:
        vn = vn_ck.get("value_norm")
        if vn is not None:
            ret_mu.fill_(vn["mu"])
            ret_var.fill_(vn["var"])
            torch.sqrt(ret_var.clamp_min(1e-8), out=ret_sig)
            value_norm_init = False
            if init_ckpt is not None:
                _g = init_ckpt.get("args", {}).get("gamma")
                print(f"init-from: value normalizer loaded (mu "
                      f"{float(vn['mu']):.4g}, sig "
                      f"{float(vn['var']) ** 0.5:.4g})"
                      + ("" if _g is None or float(_g) == args.gamma else
                         f" — trained at gamma {_g:g}, this run "
                         f"{args.gamma:g}: the EMA re-fits the return "
                         f"scale"), flush=True)
        else:
            print(f"!!! {vn_name}: checkpoint predates value "
                  "normalization — the critic will RE-FIT to normalized "
                  "targets over the first ~100 updates (EV dips, then "
                  "recovers) !!!", flush=True)

    # bounded (2026-08-02, audit 4.1): only [-100:] is ever read, and
    # the unbounded list leaked ~monotonically on multi-B runs (both
    # gen3 5B attempts died ~3.8B with this among the suspects)
    ep_returns = deque(maxlen=200)
    leaf_from = None          # set in wsd leaf (cooldown) mode
    warmup_updates = 0
    ent_floor = None          # wsd trunk exponential-anneal floor
    ent_start = None          # wsd leaf: trunk's live ent at the branch

    def trunk_ent(u):
        """The trunk entropy coef at absolute update u — a pure function
        of u (never of num_updates), so extension continues the curve."""
        if args.ent_tau <= 0:
            return args.ent_coef
        return ent_floor + (args.ent_coef - ent_floor) * math.exp(
            -u / args.ent_tau)

    if resume_ckpt is not None:
        num_updates = resume_ckpt["schedule"]["num_updates"]
        start_update = resume_ckpt["update"]
        global_step = resume_ckpt["global_step"]
        ent_floor = resume_ckpt["schedule"].get("ent_floor")
        if args.cooldown_updates > 0:
            assert resume_ckpt.get("wsd_role", "trunk") == "trunk", \
                "cooldown must branch from a TRUNK checkpoint, not a leaf"
            leaf_from = start_update
            num_updates = start_update + args.cooldown_updates
            ent_start = trunk_ent(leaf_from)
            print(f"WSD LEAF: branching from trunk update {leaf_from:,}, "
                  f"cooling down over {args.cooldown_updates} updates "
                  f"(lr {args.lr:.3g} -> {args.lr_final:.3g} 1-sqrt"
                  + (f", ent {ent_start:.3g} -> {args.ent_final:.3g}"
                     if args.ent_final is not None
                     else f", ent held {ent_start:.3g}") + ")",
                  flush=True)
        else:
            assert resume_ckpt.get("wsd_role", "trunk") == "trunk", \
                "leaves are terminal — continue from the trunk checkpoint"
            # only a TYPED --total-steps extends: the untyped default
            # (1M steps) is 1 update at recipe shape but 15,625 at a
            # 64-row test shape, where it silently turned "continue the
            # trunk" into a runaway extension (2026-09-30)
            extended = (max(1, args.total_steps // batch)
                        if "total_steps" in explicit else 0)
            if extended > num_updates:
                print(f"WSD TRUNK EXTENDED: {num_updates:,} -> "
                      f"{extended:,} updates (constant lr, absolute-index "
                      f"ent curve — extension rewrites nothing)", flush=True)
                num_updates = extended
        warmup_updates = resume_ckpt["schedule"].get(
            "warmup_updates",
            args.warmup_updates or max(1, round(0.015 * num_updates)))
        print(f"RESUMED {args.resume} at update {start_update}/{num_updates} "
              f"(step {global_step:,}; lr/schedule/Adam state restored; "
              f"env stream fresh)", flush=True)
    else:
        num_updates = max(1, args.total_steps // batch)
        start_update = 0
        global_step = 0
        warmup_updates = (args.warmup_updates
                          or max(1, round(0.015 * num_updates)))
        if args.ent_tau > 0:
            assert args.ent_final is not None, \
                "--ent-tau needs --ent-final as the anneal floor"
            ent_floor = args.ent_final
            ent_desc = (f"ent {args.ent_coef:.3g} -> floor "
                        f"{ent_floor:.3g} exp(tau={args.ent_tau} upd)")
        else:
            ent_desc = f"ent constant {args.ent_coef:.3g}"
        print(f"WSD TRUNK: warmup {warmup_updates} updates -> constant "
              f"lr {args.lr:.3g}, {ent_desc} (open-ended; branch a leaf "
              f"with --resume <trunk ckpt> --cooldown-updates N)",
              flush=True)
    if (leaf_from is None and args.ent_tau <= 0
            and args.ent_final is not None):
        print("trunk (no --ent-tau): --ent-final IGNORED — entropy is "
              "constant on this trunk; the anneal belongs to the leaf "
              "cooldown", flush=True)
    steps_before = global_step
    t_rollout = t_update = 0.0
    # --bench only (2026-09-02, Tier 1.2 of the 09-02 static audit):
    # per-phase peak-VRAM attribution. The audit INFERRED "the peak is
    # in the rollout" from the adv-filter delta; this reads it. Host-
    # side allocator stats taken at the phase syncs --bench already
    # does — no device work, no extra syncs. The global peak on the
    # THROUGHPUT line becomes the max of the two brackets (the resets
    # would otherwise hide the earlier phase from max_memory_allocated).
    # res_pre = resident bytes at the START of a rollout, i.e. what the
    # previous update left behind (the perm copies, before 1.2).
    peak_roll = peak_upd = res_pre = 0
    # --bench only: per-tick CUDA events (fwd / env / book) recorded in
    # the eager rollout loop, read after the post-rollout drain (no sync
    # inside the loop). ev[t] = (pre-fwd, pre-env, post-env, post-book);
    # book = post-env -> next tick's pre-fwd (buffer writes + b_obs copy).
    # Update 1 is warmup and excluded from the averages.
    tick_ev = None
    tick_ms = [0.0, 0.0, 0.0]
    tick_n = 0
    if args.bench and device.type == "cuda" and graph_roll is None:
        tick_ev = [[torch.cuda.Event(enable_timing=True) for _ in range(4)]
                   for _ in range(T)]
    diag_probe_obs = None    # frozen post-warmup diag batch (psrank probe)
    diag_probe_upd = None    # the update it was captured at
    if resume_ckpt is not None:
        # persisted since 2026-09-30 so psrank stays comparable across
        # a resume (see the psrank comment in the diag block). A probe
        # without "diag_probe_upd" predates 2026-10-01 and is the
        # update-0 random-policy batch: dropped, so the resumed run
        # re-captures one from its own states.
        _p = resume_ckpt.get("diag_probe_obs")
        _u = resume_ckpt.get("diag_probe_upd")
        if (_p is not None and _u is not None and _p.ndim == 2
                and _p.shape[1] == OBS_DIM):
            diag_probe_obs = _p.to(device=device, dtype=obs_dt)
            diag_probe_upd = int(_u)
        elif _p is not None:
            print("resume: dropping the checkpoint's pre-2026-10-01 "
                  "psrank probe (update-0 random-policy states); "
                  "re-capturing at the first diag", flush=True)
    diag_idx = None          # fixed [diag] rows over the flat T x N buffer
    start = time.time()

    filt = (f", adv-filter top |A| {train_n}/{batch} "
            f"({train_n // mb_size} minibatches, full-batch adv norm)"
            if adv_filter_on else "")
    # env-order state is on the banner because it was invisible for the
    # 12 days it existed (2026-08-21 -> 09-02): nothing printed it and
    # nothing set it, so no log ever said which ordering a run used.
    # Read off the env object (the trainer always passes env_order=True
    # to the CUDA env; the Rust env has no permutation, prints nothing).
    eo_live = getattr(venv, "_env_order_on", None)
    eo = "" if eo_live is None else \
        f", env-order {'on' if eo_live else 'off'}"
    print(f"=== batched PPO: {N} envs x {T} horizon = {batch} steps/update, "
          f"minibatch {mb_size}, device {device}{filt}{eo} ===",
          flush=True)

    # Structured per-update metrics (2026-09-03, dashboard option B):
    # one JSON line per update appended to <run_dir>/metrics.jsonl,
    # built from the values the per-update print sync already brought
    # to the host (no new sync, no science). Append-mode so a --resume
    # continues the file. Telemetry must never kill training: the first
    # failure disables it for the rest of the run.
    metrics_path = os.path.join(args.run_dir, "metrics.jsonl")
    metrics_f = None

    def write_metrics(*a, **kw):
        nonlocal metrics_f
        if metrics_f is False:
            return
        try:
            if metrics_f is None:
                metrics_f = open(metrics_path, "a", encoding="utf-8")
            metrics_f.write(dumps_record(update_record(*a, **kw)) + "\n")
            metrics_f.flush()
        except Exception as e:  # noqa: BLE001 — never fatal
            print(f"[metrics] disabled after write failure: {e!r}",
                  flush=True)
            metrics_f = False

    # env v46 body-draw log state
    tau_draws_path = os.path.join(args.run_dir, "tau_draws.f32")
    tau_n_episodes = 0
    tau_recent = []
    # env v47 per-episode score/PP log state (--ep-stats)
    ep_stats_path = os.path.join(args.run_dir, "ep_stats.f32")
    ep_stats_n, ep_stats_recent = 0, []
    ep_meta = None
    if args.ep_stats:
        if not (tensor_env and hasattr(venv, "pool")
                and "level_lb" in venv.pool):
            sys.exit("--ep-stats needs --env cuda with --ranked-maps "
                     "(the pool carries the per-level max score / max PP)")
        ep_meta = {k: venv.pool[k].cpu().numpy()
                   for k in ("level_lb", "level_max_pp", "level_max_score")}
        ep_stats_w = EpStatsWriter(args.run_dir)
        print(f"--ep-stats: combo walk in the training envs; per-episode "
              f"score/acc/PP + per-hand swing rows -> {ep_stats_path} "
              f"(v2: header + {EP_STATS_COLS} float32 cols)", flush=True)
        # 2026-10-04: chart_idx -> level sidecar (song, SS / BL ids, stars,
        # max scores, BL ratings, NPS, mirrored) for the dashboard's swing
        # table + best-possible PP totals (python/swing_summary.py).
        # Telemetry only: a failure never stops the run.
        try:
            from beat_arena_env.ranked_pool import write_levels_json
            write_levels_json(
                os.path.join(args.run_dir, "levels.json"), args.catalog,
                lead_s=args.lead_s, mirror=args.mirror_augment,
                sample=args.chart_pool, seed=args.seed,
                expect=ep_meta, log=print)
        except Exception as e:  # noqa: BLE001
            print(f"--ep-stats: levels.json not written: {e!r}", flush=True)
    if args.tau_rand > 0.0 and getattr(venv, "tau_rand", 0.0) > 0.0:
        # printed only when the env actually draws (the flux Rust env
        # does not); the per-episode log needs the CUDA pop_episodes
        _log = (f"episode log -> {tau_draws_path}"
                if hasattr(venv, "pop_episodes")
                else "no per-episode log on this backend")
        print(f"--tau-rand {args.tau_rand}: per-env joint torque limits "
              f"drawn center*(1 +- r) per axis at every reset (16 axes, "
              f"L/R shared); {_log}", flush=True)
    if rng_ck is not None:
        # last, after every construction above (see the rng_ck comment)
        _r = restore_rng(rng_ck, venv)
        print(f"resume: RNG streams restored ({', '.join(_r)})",
              flush=True)
    def save_ckpt(path, upd):
        """One checkpoint schema for the numbered and the final saves
        (they were two hand-kept copies of the same dict). `rng` (every
        stream, captured at the save = the state entering update upd+1)
        and `diag_probe_obs` (the frozen psrank probe, ~2.5 MB fp16 at
        production) added 2026-09-30 for the exact resume. Reads the
        loop's live locals at call time."""
        torch.save({"state_dict": cpu_state(policy), "arch": policy.arch,
                    "encoder": policy.encoder,
                    "update": upd, "global_step": global_step,
                    "recent_return": float(recent),
                    "recent_acc": float(acc_now),
                    "value_norm": {"mu": float(ret_mu),
                                   "var": float(ret_var)},
                    "optimizer": cpu_opt_state(opt),
                    "muon": muon.state_dict(),
                    "rng": rng_state(venv),
                    "diag_probe_obs": (None if diag_probe_obs is None
                                       else diag_probe_obs.cpu()),
                    "diag_probe_upd": diag_probe_upd,
                    "args": args_snapshot,
                    "wsd_role": ("leaf" if leaf_from is not None
                                 else "trunk"),
                    "branched_from_update": leaf_from,
                    "schedule": {"lr": args.lr, "lr_final": args.lr_final,
                                 "lr_decay": "wsd",
                                 "warmup_updates": warmup_updates,
                                 "ent_coef": args.ent_coef,
                                 "ent_tau": args.ent_tau,
                                 "ent_floor": ent_floor,
                                 "num_updates": num_updates}},
                   path)

    update = start_update - 1    # the final save's index if 0 updates run
    recent = float("nan")
    acc_now = float("nan")
    for update in range(start_update, num_updates):
        if leaf_from is not None:
            # leaf cooldown: 1-sqrt decay to the floor (EPFL 2405.18392:
            # 1-sqrt/linear shapes win; ~10-20% of trunk length)
            tau = (update - leaf_from + 1) / max(1, args.cooldown_updates)
            lr_now = args.lr_final + (args.lr - args.lr_final) * (
                1.0 - math.sqrt(tau))
        elif update < warmup_updates:
            lr_now = args.lr * (update + 1) / warmup_updates
        else:
            lr_now = args.lr                # the trunk: constant, forever
        for g in all_groups:
            g["lr"] = lr_now * g.get("lr_mult", 1.0)
        if leaf_from is not None:
            ent_now = ent_start
            if args.ent_final is not None:
                tau = (update - leaf_from + 1) / max(
                    1, args.cooldown_updates)
                ent_now = ent_start + (args.ent_final - ent_start) * tau
        else:
            # trunk: absolute-index exponential to the floor (or constant
            # if --ent-tau 0) — never a function of num_updates, so
            # extension continues the curve
            ent_now = trunk_ent(update)
        if args.bench and device.type == "cuda":
            res_now = torch.cuda.memory_allocated()
            res_pre = max(res_pre, res_now)
            torch.cuda.reset_peak_memory_stats()
        t0 = time.perf_counter()
        if graph_roll is not None:
            graph_roll.refresh()  # in-place fp16 caches after opt.steps
        for t in range(T):
            b_obs[t] = obs
            if graph_roll is not None and t > 0:
                a, logp, v, rew, done = graph_roll.step()
                b_act[t], b_logp[t] = a, logp
                b_rew[t] = rew.to(b_rew.dtype)
                b_done[t].copy_(done)
                b_val[t].copy_(v).mul_(ret_sig).add_(ret_mu)
                obs = venv._obs
                global_step += N
                continue
            if tick_ev is not None:
                tick_ev[t][0].record()
            with torch.no_grad():
                # t == 0 only: pay the entropy reduction (~4 kernels and
                # one (N,29,3) probs buffer, ONCE per update) to get a
                # FULL-BATCH per-head entropy — every env, before the
                # --adv-filter ever selects rows. Chosen over the two
                # alternatives in the audit because it adds NO extra
                # forward at production width: the rollout forward is
                # happening anyway and the rollout is 70% of the wall.
                # It reads the policy at the START of update u (i.e.
                # after update u-1's last opt.step), where the printed
                # `ent` reads it mid-update; that one-update lag is the
                # price of the free measurement and is stated on the
                # [kl] line's `entfull` label. Entropy consumes no RNG,
                # so the sampling stream is bit-identical either way
                # (policy.get_action_and_value docstring, 2026-07-28).
                a, logp, _, v = policy.get_action_and_value(
                    obs, need_entropy=(t == 0))
            if t == 0:
                torch.mean(policy._ent_h, dim=0, out=ent_head)
                policy._ent_h = None
            b_act[t], b_logp[t] = a, logp
            if tick_ev is not None:
                tick_ev[t][1].record()
            if tensor_env:
                nobs, rew, done, infos = venv.step(a)
                if tick_ev is not None:
                    tick_ev[t][2].record()
                b_rew[t] = rew.to(b_rew.dtype)
                b_done[t].copy_(done)
                obs = nobs
            else:
                nobs, rew, done, infos = venv.step(a.cpu().numpy())
                b_rew[t] = torch.as_tensor(rew, device=device)
                b_done[t] = torch.as_tensor(
                    done.astype(np.float32), device=device)
                obs = torch.as_tensor(nobs, dtype=torch.float32, device=device)
            # the b_val[t] write sits after the env step (2026-09-03) so
            # the main stream enqueues the env kernel first; b_val[t] has
            # no other reader until GAE.
            b_val[t].copy_(v).mul_(ret_sig).add_(ret_mu)
            if tick_ev is not None:
                tick_ev[t][3].record()
            for info in infos:
                ep_returns.append(info["episode_return"])
            global_step += N
        # (pop_stats moved to the per-update print block 2026-07-28: its
        # two .item()/.tolist() syncs sat exactly at the rollout->GAE
        # boundary, draining the queue right before the most launch-
        # latency-sensitive region; the print block already syncs)
        # drain before reading the phase timer: kernel launches are
        # async, so without this the section boundary charges queued
        # work to whichever section blocks next. Measured exec-true
        # 2026-08-31 on a 5090, N=32,768/T=32, the 1.09M-param policy:
        # 70% rollout / 30% update (measured identical graphed vs
        # eager, which is why --graph-rollout was removed same day).
        # That --bench figure is warmup-inflated (it caps at 4 updates
        # and the drain serializes), so the steady-state split is better
        # read off the isolated instruments at the same shape:
        # bench_policy_update --mbs 16384 = 39.70 ms x 64 mb = 2.54 s and
        # bench_policy_forward --ns 32768 = 26.72 ms x 32 = 0.86 s
        # against a 5.96 s/update production cycle => roughly
        # 57% rollout / 43% update, the env kernel being ~43% alone.
        # The UPDATE SHARE GREW ~5x from the 93/7 measured 2026-08-07
        # (N=4096/T=32): that was the 138k-param policy, before the
        # 6.2x actor scale-up (2026-08-13) and the adapool critic
        # (2026-08-19). Do not size update-path work off 93/7 — and the
        # 2026-07-25 ~16/84 is older still (retired torch mirror).
        # --bench only: the drain serializes CPU
        # enqueue with GPU execution and costs ~2-8% wall, so
        # production keeps async timers (and says so). Since the
        # --graph-rollout removal the async split is at least
        # indicative — a production T=32 leg now prints ~56/44 against
        # the ~57/43 exec-true estimate, where the graphed rollout used
        # to print a meaningless 0/100 (one launch swallowed the whole
        # rollout). Still not execution time; use --bench to rule. The 2-8% is
        # itself unverified at this width: the 2026-08-31 --bench run
        # came out 34% below production steps/s, but at 4 updates that
        # is mostly warmup, so the two effects are not separable
        # without lifting the cap.
        if args.bench and device.type == "cuda":
            torch.cuda.synchronize()
            pk_r = torch.cuda.max_memory_allocated()
            peak_roll = max(peak_roll, pk_r)
            torch.cuda.reset_peak_memory_stats()
            if (tick_ev is not None and tensor_env
                    and update > start_update):
                for t in range(T):
                    ev = tick_ev[t]
                    nxt = tick_ev[t + 1][0] if t + 1 < T else ev[3]
                    tick_ms[0] += ev[0].elapsed_time(ev[1])
                    tick_ms[1] += ev[1].elapsed_time(ev[2])
                    tick_ms[2] += ev[2].elapsed_time(nxt)
                tick_n += T
        t_rollout += time.perf_counter() - t0

        t0 = time.perf_counter()
        with torch.no_grad():
            # bootstrap value only: no sample, no logprob (2026-07-28;
            # the discarded sample also consumed RNG — dropping it
            # shifts the sampling stream, accepted at the same time as
            # the Gumbel-max switch)
            b_val[T].copy_(policy.forward_stacked(obs)[1]
                           ).mul_(ret_sig).add_(ret_mu)
            # everything non-recursive hoisted out of the T-loop (same
            # elementwise op order as the per-t version, so bit-exact);
            # only the scan itself stays sequential. The scan writes
            # coef[t]*prev then adds deltas[t] in place: IEEE addition
            # is commutative, so prod+delta == delta+prod bitwise —
            # 2 launches/step, zero allocations (was 3 + 2 allocs).
            nonterm = 1.0 - b_done
            deltas = b_rew + args.gamma * b_val[1:] * nonterm - b_val[:T]
            coef = (args.gamma * args.gae_lambda) * nonterm
            adv = b_adv
            torch.mul(coef[T - 1], gae_zero, out=adv[T - 1])
            adv[T - 1].add_(deltas[T - 1])
            for t in reversed(range(T - 1)):
                torch.mul(coef[t], adv[t + 1], out=adv[t])
                adv[t].add_(deltas[t])
            returns = torch.add(adv, b_val[:T], out=b_ret)
            # critic diagnostic (2026-08-19): explained variance of the
            # PRE-update value predictions against their own GAE targets
            # over the full T x N batch. 0-dim device tensor, read at
            # the per-update print sync — no extra host sync.
            # READ THIS BEFORE TRUSTING EV (2026-08-31): returns is
            # adv + b_val[:T], so (returns - b_val[:T]) IS adv exactly
            # and EV = 1 - Var(adv)/Var(adv + V) identically. It is
            # therefore near 1 whenever state-value spread dominates
            # per-action advantage spread — which is always true here
            # (V swings hugely across a song). EV is a CHANGE detector,
            # not an absolute critic-quality measure: a drop means
            # something broke, but a high value is close to structural.
            # It also falls with T for a mechanical reason (longer
            # traces accumulate more deltas, so Var(adv) grows): the
            # T=8 0.995 vs T=32 0.992 gap is not a quality gap.
            # Bootstrap share of the target is gamma^T = 75% at T=8,
            # 32% at T=32.
            ev = 1.0 - (returns - b_val[:T]).var() \
                / returns.var().clamp_min(1e-12)
            # T1 value-norm stats (EMA, device-side, no syncs). Updated
            # BEFORE the targets are formed; b_val was unnormalized
            # with last update's stats — consistent to within one
            # beta = --value-norm-beta step (default VALUE_NORM_BETA).
            r_mu, r_var = returns.mean(), returns.var()
            if value_norm_init:
                ret_mu.copy_(r_mu)
                ret_var.copy_(r_var)
                value_norm_init = False
            else:
                vnb = args.value_norm_beta
                ret_mu.mul_(1.0 - vnb).add_(vnb * r_mu)
                ret_var.mul_(1.0 - vnb).add_(vnb * r_var)
            torch.sqrt(ret_var.clamp_min(1e-8), out=ret_sig)

        f_obs = b_obs.reshape(-1, OBS_DIM)
        f_act = b_act.reshape(-1, A_DIM)
        f_logp = b_logp.reshape(-1)
        f_adv = adv.reshape(-1)
        f_ret = returns.reshape(-1)

        if adv_filter_on:
            # top-advantage filter, LAZILY — only the index tensor
            # lives here; the per-epoch gather composes it with the
            # shuffle so no filtered copy of the flat buffers is ever
            # materialized. Diagnostics and EV above stay full-batch
            # on purpose: they remain comparable across runs.
            keep_idx = adv_filter_indices(f_adv, train_n)
            # 1.4: normalization stats over the FULL batch, once per
            # update, used by every minibatch below (see
            # adv_filter_stats for why; C §1.1). Off -> the minibatch
            # loop keeps its own per-minibatch stats, bit-identical.
            af_mu, af_sd, af_r = adv_filter_stats(f_adv, keep_idx)
            af_ratio.copy_(af_r)

        if fp16_update:
            policy.fp16 = True  # 'update' mode: rollout stayed fp32

        m_diag = None
        if (args.diag_every > 0 and flat_grad is not None
                and update % args.diag_every == 0):
            # T0 diagnostics (2026-08-19 critic redesign): a
            # 2048-row grad-enabled forward + TWO extra backwards,
            # separating the policy-loss and value-loss gradients over
            # the SHARED params (flat buffer [0, shared_end)) — their
            # cosine discriminates actor/critic interference from
            # underfitting; per-group norms + dormant-neuron fraction +
            # trunk-feature effective rank discriminate critic
            # interference from gradient-propagation failure
            # (2506.15544's critic-independent collapse mode). Costs
            # one host sync at its print — diag-cadence only.
            # Rows (2026-09-30, review bug 3): a fixed seeded sample of
            # the whole flat T x N buffer (diag_index), the SAME index
            # for every tensor sliced here — it was f_obs[:dn], i.e. t=0
            # of envs 0..2047 only. Losses: exactly the minibatch loop's
            # terms — the policy loss on advantages normalized the way
            # the training path does (full-batch stats under
            # --adv-filter), and vf_coef * value_loss (HL-Gauss CE on
            # the normalized return) — where the value side used to be
            # 0.5*MSE on the HL expectation without vf_coef, a loss
            # nothing trains, so |g_v| and gradcos measured the wrong
            # gradient.
            dn = min(2048, batch)
            if diag_idx is None:
                diag_idx = diag_index(batch, dn).to(device)
            policy._diag_stash = True
            _, dlogp, _, dval = policy.get_action_and_value(
                f_obs[diag_idx], f_act[diag_idx], need_entropy=False)
            policy._diag_stash = False
            da = f_adv[diag_idx]
            if adv_filter_on:
                da = (da - af_mu) / (af_sd + 1e-8)
            else:
                da = (da - da.mean()) / (da.std() + 1e-8)
            dratio = (dlogp - f_logp[diag_idx]).clamp(
                max=LOG_RATIO_CLAMP).exp()
            d_pg = pg_loss_fn(dratio, da, clip=CLIP)
            d_v = args.vf_coef * value_loss(policy, f_ret[diag_idx],
                                            ret_mu, ret_sig)
            policy._value_logits = None
            dscale = fp16_scale if fp16_update else 1.0
            flat_grad.zero_()
            (d_pg * dscale).backward(retain_graph=True)
            g_pi = flat_grad[:shared_end] / dscale
            flat_grad.zero_()
            (d_v * dscale).backward()
            g_v = flat_grad[:shared_end] / dscale
            flat_grad.zero_()
            torch.nan_to_num_(g_pi, nan=0.0, posinf=0.0, neginf=0.0)
            torch.nan_to_num_(g_v, nan=0.0, posinf=0.0, neginf=0.0)
            n_pi, n_v = g_pi.norm(), g_v.norm()
            gcos = (g_pi @ g_v) / (n_pi * n_v + 1e-12)
            h_d = policy._diag_h.float()
            policy._diag_h = None
            a_mean = h_d.abs().mean(0)
            dorm = (a_mean < 0.025 * a_mean.mean()).float().mean()
            sv = torch.linalg.svdvals(h_d - h_d.mean(0))
            cum = sv.cumsum(0) / sv.sum().clamp_min(1e-12)
            srank = int((cum < 0.99).sum().item()) + 1
            # fixed-probe srank (2026-08-20): the on-policy srank
            # confounds representation contraction with STATE-
            # DISTRIBUTION narrowing (a sharpening policy visits fewer
            # states). Freeze the first diag batch of this process and
            # re-measure rank on it every diag: probe-srank stable
            # while on-policy srank falls = benign narrowing; both
            # falling = genuine capacity contraction (plasticity loss)
            # -> act (weight decay / resets / scale-up).
            # NOT COMPARABLE ACROSS A RESUME (2026-08-31): the probe is
            # process-local, so every --resume re-captures it and psrank
            # steps discontinuously at the branch point. Worse, a resume
            # captures it on the first update after "env stream fresh",
            # when every env is at t=0 — an episode-start-heavy and so
            # unusually NARROW batch, which depresses psrank for reasons
            # that have nothing to do with the network (seen at the T=32
            # 2250->4500 extension: psrank read 51 vs on-policy 56,
            # inverting the healthy ordering purely as an artifact).
            # FIXED 2026-09-30: diag_probe_obs is saved in every
            # checkpoint and restored by --resume, so a resumed run
            # keeps measuring psrank on its trunk's probe; and the probe
            # is captured from the diag_index rows (all ticks), not t=0.
            # (Checkpoints from before carry none; the first diag after
            # such a resume still re-captures.)
            # FIXED 2026-10-01 (architecture review): the probe was the
            # update-0 batch — states of the RANDOM initial policy, off
            # the trained policy's distribution for the whole run (on
            # v61 its value read -2.1 sd vs -1.0 on-policy). It is now
            # captured at the first diag at/after --probe-capture-update
            # (default 250, past warmup and the early collapse of the
            # random-policy states); earlier diags print no psrank.
            psrank = None
            if (diag_probe_obs is None
                    and update >= args.probe_capture_update):
                diag_probe_obs = f_obs[diag_idx].clone()
                diag_probe_upd = update
                print(f"[diag] psrank probe captured at update "
                      f"{update + 1}", flush=True)
            if diag_probe_obs is not None:
                with torch.no_grad():
                    policy._diag_stash = True
                    policy.get_action_and_value(diag_probe_obs,
                                                need_entropy=False)
                    policy._diag_stash = False
                hp = policy._diag_h.float()
                policy._diag_h = None
                svp = torch.linalg.svdvals(hp - hp.mean(0))
                cump = svp.cumsum(0) / svp.sum().clamp_min(1e-12)
                psrank = int((cump < 0.99).sum().item()) + 1
            # same .item() calls as before, hoisted so the metrics line
            # can reuse them without a second sync each
            per_gv = [(nm, g_pi[a:b].norm().item(), g_v[a:b].norm().item())
                      for nm, a, b in diag_groups]
            per_g = "  ".join(f"{nm} {a:.1e}/{b:.1e}" for nm, a, b in per_gv)
            gcos_v, npi_v, nv_v, dorm_v = (gcos.item(), n_pi.item(),
                                           n_v.item(), dorm.item())
            print(f"[diag] upd {update + 1}  gradcos {gcos_v:+.3f}"
                  f"  |g_pi| {npi_v:.2e}  |g_v| {nv_v:.2e}"
                  f"  dormant {dorm_v:.3f}  srank {srank}/"
                  f"{h_d.shape[1]}"
                  + (f"  psrank {psrank}" if psrank is not None else "")
                  + f"  pi/v: {per_g}", flush=True)
            m_diag = {"gradcos": gcos_v, "gpi": npi_v, "gv": nv_v,
                      "dormant": dorm_v, "srank": srank,
                      "srank_dim": int(h_d.shape[1]), "psrank": psrank,
                      "groups": {nm: [a, b] for nm, a, b in per_gv}}
            if psrank is None:           # before the probe is captured
                del m_diag["psrank"]

        kl_stop_mb = None   # --target-kl: minibatch index the update stopped at
        gn_sq = None        # [gnorm] per-module grad norm^2 of the last minibatch
        for ep_i in range(args.epochs):
            if kl_stop_mb is not None:
                break
            # device-side shuffle: a host-side index array costs a
            # synchronizing pageable H2D copy per minibatch (~26 ms/update
            # measured on WDDM). Permute each flat tensor ONCE per epoch
            # and hand minibatches contiguous slices: 5 gathers instead
            # of 40, identical rows in identical order per minibatch, so
            # every downstream op sees the same values (2026-07-28)
            perm = torch.randperm(train_n, device=device)
            if adv_filter_on:
                perm = keep_idx[perm]
            p_obs, p_act = f_obs[perm], f_act[perm]
            p_logp, p_adv, p_ret = f_logp[perm], f_adv[perm], f_ret[perm]
            for s in range(0, train_n, mb_size):
                if kl_stop_mb is not None:
                    break
                # advantage stats over the FULL minibatch so --mb-accum
                # K>1 is the same math as K=1 (each micro loss is a
                # mean over an equal slice; the (1/K)-scaled sum of
                # micro means IS the minibatch mean, modulo fp
                # association); one opt.step per minibatch either way.
                # Under --adv-filter the stats are the full-batch ones
                # computed once above (1.4): a kept minibatch's own sd
                # is ~1.36x the batch's, which used to shrink the PG
                # term 0.73x against entropy and value.
                if adv_filter_on:
                    a_mu, a_sd = af_mu, af_sd
                else:
                    a_full = p_adv[s:s + mb_size]
                    a_mu = a_full.mean()
                    a_sd = a_full.std()
                # 0.3: which minibatch(es) get the KL/clipfrac probe.
                # Host-side ints only — no sync. Accumulated across the
                # accum micros as (1/K)-weighted micro means, which IS
                # the minibatch mean (the same equal-slice argument the
                # loss below relies on). A single-minibatch update
                # (--minibatches 1) is both first and last and fills
                # both slots, so mb1 == mbN there rather than mbN going
                # permanently stale.
                kl_slots = []
                if ep_i == 0 and s == 0:
                    kl_slots.append((kl_first, cf_first))
                if (ep_i == args.epochs - 1 and s + mb_size >= train_n):
                    kl_slots.append((kl_last, cf_last))
                if flat_grad is not None:
                    flat_grad.zero_()
                else:
                    opt.zero_grad(set_to_none=True)
                    muon.zero_grad()
                micro = mb_size // args.mb_accum
                for j in range(args.mb_accum):
                    sl = slice(s + j * micro, s + (j + 1) * micro)
                    _, newlogp, entropy, newval = \
                        policy.get_action_and_value(p_obs[sl], p_act[sl])
                    logratio = newlogp - p_logp[sl]
                    ratio = logratio.clamp(max=LOG_RATIO_CLAMP).exp()
                    if (args.target_kl > 0 and j == 0 and s > 0
                            and (s // mb_size) % args.kl_check_every == 0):
                        # --target-kl probe: k3 approx KL of THIS
                        # minibatch vs the rollout policy, one host sync.
                        # Past the target the update ends here: no
                        # backward has run for this minibatch and grads
                        # were zeroed above, so nothing partial is
                        # applied. kl_last/cf_last take the probe's
                        # values so the [kl] line and metrics show the
                        # drift that tripped it.
                        _r = ratio.detach() - 1.0
                        _kl = (_r - logratio.detach()).mean()
                        if float(_kl) > args.target_kl:
                            kl_stop_mb = s // mb_size
                            kl_last.copy_(_kl)
                            cf_last.copy_((_r.abs() > CLIP).to(_r.dtype).mean())
                            # release this minibatch's forward graph:
                            # no backward will run, and the live
                            # activations of a 16k-row forward are
                            # what the next rollout needs (v52 OOMed
                            # on the update after the first stop)
                            del newlogp, entropy, newval, logratio, ratio
                            del _r, _kl
                            policy._value_logits = None
                            break
                    if fp16_update:
                        # overflow forensics (device-side, read at the
                        # print sync): a max|logratio| near ~88 fingers
                        # exp() fp32 overflow as the non-finite-grad
                        # source, small values finger the fp16
                        # activations instead
                        torch.maximum(logr_max,
                                      logratio.detach().abs().max(),
                                      out=logr_max)
                    if kl_slots:
                        # k3 approx-KL and clipfrac, device-side.
                        # .detach() keeps these out of the graph; the
                        # ratio/logratio tensors are the ones the loss
                        # already built, so nothing is recomputed.
                        r_m1 = ratio.detach() - 1.0
                        kl_m = (r_m1 - logratio.detach()).mean()
                        cf_m = (r_m1.abs() > CLIP).to(r_m1.dtype).mean()
                        for kl_t, cf_t in kl_slots:
                            if j == 0:
                                kl_t.zero_()
                                cf_t.zero_()
                            kl_t.add_(kl_m, alpha=1.0 / args.mb_accum)
                            cf_t.add_(cf_m, alpha=1.0 / args.mb_accum)
                    if fp16_update and ep_i == 0 and s == 0 and j == 0:
                        # "same policy" fixed point: |ratio-1| at epoch
                        # 1 measures rollout/update precision
                        # consistency (device-side; read at the
                        # per-update print sync)
                        ratio_diag = (ratio - 1.0).abs().mean().detach()
                    a_mb = (p_adv[sl] - a_mu) / (a_sd + 1e-8)
                    pg_loss = pg_loss_fn(ratio, a_mb, clip=CLIP)
                    # T1: the critic learns the NORMALIZED target
                    v_loss = value_loss(policy, p_ret[sl], ret_mu, ret_sig)
                    ent_loss = entropy.mean()
                    loss = (pg_loss - ent_now * ent_loss
                            + args.vf_coef * v_loss)
                    if args.mb_accum > 1:
                        loss = loss / args.mb_accum
                    if fp16_update:
                        (loss * fp16_scale).backward()
                    else:
                        loss.backward()
                if kl_stop_mb is not None:
                    break
                if fp16_update:
                    # dynamic loss scale, ZERO host syncs: unscale + flag
                    # non-finites in one kernel over the FLAT grad
                    # buffer, then zero every grad on overflow
                    # device-side. MUST run before the clip (an inf
                    # grad would turn clip's 0.5/inf coef into 0*inf =
                    # NaN everywhere). Overflow steps become
                    # momentum-only Adam steps — bounded by lr, audited
                    # at the print.
                    inv_scale.copy_(fp16_scale.reciprocal())
                    grads = ([flat_grad] if flat_grad is not None else
                             [p.grad for p in policy.parameters()
                              if p.grad is not None])
                    torch._amp_foreach_non_finite_check_and_unscale_(
                        grads, found_inf, inv_scale)
                    ok = 1.0 - found_inf[0].clamp(max=1.0)  # 0-dim
                    torch._foreach_mul_(grads, ok)
                    # mul-by-0 is NOT zeroing for non-finite grads:
                    # inf*0 = NaN, which then rode fused Adam into the
                    # weights and killed BOTH gen3 5B attempts at ~3.8B
                    # (2026-07-26: ent nan -> overflow cascade ->
                    # multinomial device assert -> segfault under the
                    # graphed rollout). nan_to_num_ makes the zeroing
                    # actually zero; no-op on finite grads, no syncs.
                    for g in grads:
                        torch.nan_to_num_(g, nan=0.0, posinf=0.0,
                                          neginf=0.0)
                    # per-MINIBATCH scale adaptation, all device-side
                    # (2026-08-02): halve on overflow, x2 after
                    # FP16_GROWTH_INTERVAL consecutive clean steps,
                    # clamp to [1, initial]. The old host-float scale
                    # halved once per UPDATE and never grew — one early
                    # transient disabled scaling for the rest of a run
                    # (400M: scale 1 from update 188 on, 9.3% of all
                    # gradient steps overflow-zeroed). Stream-ordered
                    # after the backward, so the mul_ never races the
                    # backward's read of the old scale.
                    overflow = found_inf[0].clamp(max=1.0)  # 0-dim, 0/1
                    clean = 1.0 - overflow
                    clean_steps.mul_(clean).add_(clean)
                    grow = (clean_steps >= FP16_GROWTH_INTERVAL).to(
                        torch.float32)
                    fp16_scale.mul_(0.5 * overflow + clean * (1.0 + grow))
                    fp16_scale.clamp_(min=1.0, max=scale_cap)
                    clean_steps.mul_(1.0 - grow)
                    inf_count += found_inf
                    found_inf.zero_()
                if (args.diag_every > 0 and update % args.diag_every == 0
                        and ep_i == args.epochs - 1
                        and s + mb_size >= train_n):
                    # [gnorm] (2026-09-20, v52's 4x gradient norm): the
                    # ACTUAL minibatch gradient (post-unscale, pre-clip)
                    # split by top-level module, all params — the
                    # [diag] line covers the shared params only, so a
                    # head or critic-head gradient is invisible there.
                    gn_sq = {}
                    for n_, p_ in policy.named_parameters():
                        if p_.grad is not None:
                            k_ = n_.split(".")[0]
                            v_ = p_.grad.detach().float().square().sum()
                            gn_sq[k_] = gn_sq[k_] + v_ if k_ in gn_sq else v_
                if flat_grad is not None:
                    if flat_guard:
                        # one-time: autograd must have accumulated INTO
                        # the views (in place), not replaced them
                        bad = [p_ for p_ in f_params
                               if p_.grad is None
                               or p_.grad._base is not flat_grad]
                        if bad:
                            raise RuntimeError(
                                "flat-grad views were detached by "
                                f"autograd ({len(bad)} params)")
                        flat_guard = False
                    # flat global-norm clip: same math as
                    # clip_grad_norm_ (norm of per-tensor norms == norm
                    # of the concatenation), one kernel each
                    total_norm = torch.linalg.vector_norm(flat_grad)
                    clip_coef = (MAX_GRAD_NORM / (total_norm + 1e-6)
                                 ).clamp(max=1.0)
                    flat_grad.mul_(clip_coef)
                    # 0.3 / C §2.2: both were computed and discarded.
                    # 2 device-side adds per optimizer step; the means
                    # are formed at the print sync. clipc < 1 = the 0.5
                    # clip bound on some steps this update. NOTE these
                    # are POST-unscale, POST-overflow-zero norms, so an
                    # overflow step contributes gnorm 0 / clipc 1.
                    gnorm_sum.add_(total_norm)
                    clipc_sum.add_(clip_coef)
                    opt_steps += 1
                else:
                    tn = nn.utils.clip_grad_norm_(policy.parameters(),
                                                  MAX_GRAD_NORM)
                    gnorm_sum.add_(tn.detach())
                    clipc_sum.add_((MAX_GRAD_NORM / (tn.detach() + 1e-6)
                                    ).clamp(max=1.0))
                    opt_steps += 1
                if wd_groups:
                    # decoupled decay, one foreach kernel per group,
                    # scalar from the host (no sync); lr is what the
                    # schedule loop set this update
                    with torch.no_grad():
                        for g_, ps_ in wd_groups:
                            torch._foreach_mul_(ps_, 1.0 - g_["lr"] * wd)
                opt.step()
                muon.step()
                if proj_lins:
                    # SimBaV2: rows back onto the unit sphere after EVERY
                    # step (Muon and Adam both ran), in place on the fp32
                    # masters, before the replica invalidation below
                    policy.project_trunk_weights()
                bump_weights_version()  # invalidate fp16 replica caches
        # Tier 1.2 (2026-09-02 audit A §3): the permuted copies (p_obs
        # 1.25 GB + p_act at production shape) are dead once the epoch
        # loop ends, but as loop-scope Python locals they lived through
        # the ENTIRE next rollout + GAE until the next update's gather
        # reassigned them — sitting under the rollout-phase peak doing
        # nothing. keep_idx (--adv-filter) likewise. Bit-identical: the
        # same rows are re-gathered next update either way. `_` is the
        # last micro's action slice — a VIEW of p_act, which would keep
        # that storage alive through the del (measured: +0.23 GiB
        # resident at the next rollout start without it).
        del p_obs, p_act, p_logp, p_adv, p_ret, perm, _
        if adv_filter_on:
            del keep_idx
        if fp16_update:
            policy.fp16 = args.fp16 == "all"
        if args.bench and device.type == "cuda":
            torch.cuda.synchronize()  # see rollout-timer note
            pk_u = torch.cuda.max_memory_allocated()
            peak_upd = max(peak_upd, pk_u)
            # per-update attribution line (host-side only): the first
            # update's rollout has no update leftovers yet, so the
            # u=1 vs u>=2 rollout peaks are the "what the update left
            # behind" measurement in one run
            print(f"[bench-mem] upd {update + 1}: resident at rollout "
                  f"start {res_now / 2**30:.2f}  rollout peak "
                  f"{pk_r / 2**30:.2f}  update peak {pk_u / 2**30:.2f} "
                  "GiB", flush=True)
        t_update += time.perf_counter() - t0

        m_ev = {}  # event-line values for the metrics line
        if tensor_env:
            # deferred from the rollout->GAE boundary: this is the one
            # place per update the host reads the device anyway
            ep_returns.extend(venv.pop_stats())
            # f32 Cholesky pivot-floor engagements (SPEC-v10 §20e):
            # reported, never hidden
            if hasattr(venv, "pop_pd_hits"):
                pd = venv.pop_pd_hits()
                m_ev["pd_net"] = pd
                if pd:
                    print(f"[pd-net] {pd} f32 pivot-floor engagement(s) "
                          "this update", flush=True)
            # pre-integration root-envelope engagements (SPEC-v10
            # Divergences §23): each one terminated an env's episode with
            # a zeroed reward (the C backstop). Reported, never hidden —
            # a converged policy measures 0.
            if hasattr(venv, "pop_envelope_hits"):
                eh = venv.pop_envelope_hits()
                m_ev["envelope"] = eh
                if eh:
                    print(f"[envelope] {eh} root-escape engagement(s) "
                          "this update (episodes terminated, reward 0)",
                          flush=True)
        # env v11 FALL terminations (unilateral-contact spec §4): the
        # root-escape envelope's twin — the body went to the floor.
        # v11.1: each fall is PRICED at REWARD_MISS x the remaining
        # scoreable notes (the reward-0 version bred the 8h suicide
        # optimum). Reported, never hidden, on BOTH backends (unlike
        # [envelope], a CUDA-only counter).
        if hasattr(venv, "pop_fall_hits"):
            fh = venv.pop_fall_hits()
            m_ev["falls"] = fh
            if fh:
                print(f"[fall] {fh} fall termination(s) this update "
                      "(each priced at the flat fall_const)",
                      flush=True)
        if hasattr(venv, "pop_tilt_hits"):
            th = venv.pop_tilt_hits()
            m_ev["tilts"] = th
            if th:
                print(f"[tilt] {th} tilt termination(s) this update "
                      "(chest up-axis fwd>40 / back>20 / lateral>40 deg; "
                      "flat fall_const)", flush=True)
        if hasattr(venv, "pop_foot_lifts"):
            lh_l, lh_r = venv.pop_foot_lifts()
            m_ev.update(lifts=lh_l + lh_r, lifts_L=lh_l, lifts_R=lh_r)
            if lh_l or lh_r:
                print(f"[lift] {lh_l + lh_r} foot-lift event(s) this "
                      f"update (L {lh_l} / R {lh_r}; each charged the "
                      "flat -0.1)", flush=True)
        # env v19 flail telemetry: mean blade arc (rad, both hands) per
        # note-free env-tick from hands with no live follow-through —
        # the phantom-swing signal the v19 post_rem obs lane targets.
        # CUDA-backend counter; the Rust wrapper reports zeros.
        if hasattr(venv, "pop_flail"):
            fl_l, fl_r, fl_ticks = venv.pop_flail()
            if fl_ticks:
                m_ev.update(flail=(fl_l + fl_r) / fl_ticks,
                            flail_L=fl_l / fl_ticks, flail_R=fl_r / fl_ticks,
                            flail_ticks=fl_ticks)
                print(f"[flail] {(fl_l + fl_r) / fl_ticks:.4f} "
                      f"rad/idle-tick (L {fl_l / fl_ticks:.4f} / "
                      f"R {fl_r / fl_ticks:.4f}; {fl_ticks} note-free "
                      "env-ticks)", flush=True)
        # wrist-bend reversal telemetry (2026-09-22, user): reversals of
        # the bend axis per hand (60 deg hysteresis peak detection in
        # the kernel, every tick) per second of env time; the update
        # stepped num_envs x horizon env-ticks at 30 Hz.
        if hasattr(venv, "pop_wrist_rev"):
            wr_l, wr_r = venv.pop_wrist_rev()
            _wt = float(args.num_envs * args.horizon)
            wr_ls, wr_rs = 30.0 * wr_l / _wt, 30.0 * wr_r / _wt
            m_ev.update(wrev=0.5 * (wr_ls + wr_rs), wrev_L=wr_ls, wrev_R=wr_rs)
            print(f"[wrev] L {wr_ls:.3f} /s  R {wr_rs:.3f} /s wrist-bend "
                  f"reversals per hand (60 deg hysteresis; {int(_wt)} "
                  "env-ticks)", flush=True)
        # limit-rest telemetry: mean limit-row lambda-flow per
        # env-tick (N*m*s), legs / spine / arms split — sustained flow
        # means the pose is resting on its stops. Arms accumulate
        # always; they are PRICED only under --limit-arms (env v35).
        if hasattr(venv, "pop_limit_rest"):
            lr_l, lr_s, lr_a, lr_t = venv.pop_limit_rest()
            if lr_t:
                m_ev.update(limitrest=(lr_l + lr_s + lr_a) / lr_t,
                            limitrest_legs=lr_l / lr_t,
                            limitrest_spine=lr_s / lr_t,
                            limitrest_arms=lr_a / lr_t)
                print(f"[limitrest] {(lr_l + lr_s + lr_a) / lr_t:.4f} "
                      f"N*m*s/tick (legs {lr_l / lr_t:.4f} / spine "
                      f"{lr_s / lr_t:.4f} / arms {lr_a / lr_t:.4f})",
                      flush=True)
        # activation telemetry (2026-09-13, user: energy-conservation
        # stats over training): mean commanded |gear fraction| summed
        # over the 29 joints per env-tick (0..29; the retired v23 base),
        # spine / arms / legs split. Accumulates at coef 0 too.
        if hasattr(venv, "pop_activation"):
            ac_t, ac_s, ac_a, ac_l, ac_n = venv.pop_activation()
            if ac_n:
                m_ev.update(activation=ac_t / ac_n,
                            activation_spine=ac_s / ac_n,
                            activation_arms=ac_a / ac_n,
                            activation_legs=ac_l / ac_n)
                print(f"[activation] {ac_t / ac_n:.4f} /env-tick (spine "
                      f"{ac_s / ac_n:.4f} / arms {ac_a / ac_n:.4f} / legs "
                      f"{ac_l / ac_n:.4f})", flush=True)
        # env v43 torque telemetry (2026-09-14, user): mean applied
        # sum |tau| over the 29 joints per env-tick in N*m — the
        # --energy-coef base — spine / arms / legs split. Accumulates
        # at coef 0 too.
        if hasattr(venv, "pop_torque"):
            tq_t, tq_s, tq_a, tq_l, tq_n = venv.pop_torque()
            if tq_n:
                m_ev.update(torque=tq_t / tq_n,
                            torque_spine=tq_s / tq_n,
                            torque_arms=tq_a / tq_n,
                            torque_legs=tq_l / tq_n)
                print(f"[torque] {tq_t / tq_n:.3f} N*m/env-tick (spine "
                      f"{tq_s / tq_n:.3f} / arms {tq_a / tq_n:.3f} / legs "
                      f"{tq_l / tq_n:.3f})", flush=True)
        # env v46 body-draw log: every episode that ended this update
        # -> (s0..s15, ticks, return, fell, cuts, pts, chart) appended to
        # <run-dir>/tau_draws.f32 (raw float32 rows, 22 cols; reader:
        # tau_regress.py); a [tau] line every 25 updates with the
        # least-squares slope of fall rate on each group's mean scale
        rows = None
        if (args.tau_rand > 0.0 or args.ep_stats) and hasattr(venv, "pop_episodes"):
            rows = venv.pop_episodes()
        if args.tau_rand > 0.0 and rows is not None:
            if rows.shape[0]:
                with open(tau_draws_path, "ab") as f:
                    f.write(np.ascontiguousarray(rows[:, :22],
                                                 dtype=np.float32).tobytes())
                tau_n_episodes += int(rows.shape[0])
                tau_recent.append(rows[:, :22])
                if len(tau_recent) > 400:
                    tau_recent.pop(0)
            m_ev.update(tau_episodes=tau_n_episodes)
        if args.ep_stats and rows is not None and rows.shape[0]:
            # env v47: per-episode score -> accuracy -> PP at the level's
            # max_pp (API maxPP), the bench's own curve (scoresaber.py);
            # v2 (2026-10-04) adds the update index + per-hand swing parts
            er = ep_stats_rows(rows, ep_meta, update + 1)   # = metrics upd
            ep_stats_w.append(er)
            ep_stats_n += int(er.shape[0])
            ep_stats_recent.append(er)
            if len(ep_stats_recent) > 400:
                ep_stats_recent.pop(0)
            m_ev.update(ep_stats_episodes=ep_stats_n)
            if update % 25 == 0:
                R = np.concatenate(ep_stats_recent)
                R = R[R[:, 4] >= 60]                 # skip lead-only episodes
                if R.shape[0] >= 50:
                    done_m = R[:, 6] > 0.5
                    comp = float(done_m.mean())
                    if done_m.any():
                        C = R[done_m]
                        acc_m, pp_m = float(C[:, 8].mean()), float(C[:, 9].mean())
                        fc = float((C[:, 11] == 0).mean())
                        m_ev.update(ep_complete=comp, ep_acc=acc_m, ep_pp=pp_m)
                        print(f"[pp] {ep_stats_n:,} episodes logged | completed "
                              f"{100 * comp:.1f}% of the last {R.shape[0]:,} | "
                              f"completed maps: acc {acc_m:.4f}  pp {pp_m:.1f} "
                              f"(median {float(np.median(C[:, 9])):.1f}, max "
                              f"{float(C[:, 9].max()):.1f})  FC {100 * fc:.1f}%  "
                              f"breaks {float(C[:, 11].mean()):.2f}  "
                              f"maxcombo {float(C[:, 10].mean()):.0f}", flush=True)
        # [tau] fall slope: reads only tau_recent (filled under --tau-rand),
        # so it is independent of --ep-stats and of this update's pop
        if update % 25 == 0 and tau_recent:
            R = np.concatenate(tau_recent)
            R = R[R[:, 16] >= 60]           # skip lead-only episodes
            if R.shape[0] >= 200:
                from beat_arena_env.constants import TAU_GROUPS
                fell = R[:, 18]
                parts = []
                for g, axes in TAU_GROUPS.items():
                    x = (R[:, list(axes)].mean(1) - 1.0) / 0.25
                    xc = x - x.mean()
                    den = float((xc * xc).sum())
                    slope = float((xc * (fell - fell.mean())).sum() / den) if den > 0 else 0.0
                    parts.append(f"{g} {slope:+.4f}")
                print(f"[tau] {tau_n_episodes:,} episodes logged | fall rate "
                      f"{fell.mean():.4f} over the last {R.shape[0]:,} | "
                      f"d(fall)/d(+25%) " + "  ".join(parts), flush=True)
        # env v41 self-contact telemetry: mean self-collision impulse
        # per env-tick (N.s) and the share of env-ticks with any
        # body-on-body contact — accumulates at coef 0 too, so the
        # behavior is visible before it is priced.
        if hasattr(venv, "pop_self_contact"):
            sc_imp, sc_ct, sc_t = venv.pop_self_contact()
            if sc_t:
                m_ev.update(contact=sc_imp / sc_t, contact_frac=sc_ct / sc_t)
                print(f"[contact] {sc_imp / sc_t:.4f} N*s/tick "
                      f"({100.0 * sc_ct / sc_t:.2f}% of env-ticks in "
                      f"self-contact)", flush=True)
        # 2026-08-20 THE metric (user decision): training accuracy =
        # note score credited / (115 * good cuts) since the last pop.
        # Event-flow ratio — no completion window, no wave phase, no
        # censoring; at recipe shape ~tens of thousands of cuts per
        # update, so a single print is already ±~0.1%.
        acc_now = acc_l = acc_r = float("nan")
        if hasattr(venv, "pop_acc"):
            p_l, p_r, c_l, c_r = venv.pop_acc()
            if c_l + c_r:
                acc_now = (p_l + p_r) / (115.0 * (c_l + c_r))
            if c_l:
                acc_l = p_l / (115.0 * c_l)
            if c_r:
                acc_r = p_r / (115.0 * c_r)
        # 2026-10-04 (user): the acc's components over the same window —
        # pre-swing / 70, post-swing / 30, precision / 15 (note cuts;
        # cuda_env.pop_acc_parts)
        acc_pre = acc_post = acc_prec = float("nan")
        if hasattr(venv, "pop_acc_parts"):
            acc_pre, acc_post, acc_prec = venv.pop_acc_parts()
            print(f"[accparts] pre {acc_pre:.4f}  post {acc_post:.4f}  "
                  f"prec {acc_prec:.4f}", flush=True)
        recent = (np.mean(list(ep_returns)[-100:]) if ep_returns
                  else float("nan"))
        elapsed = time.time() - start
        rate = (global_step - steps_before) / elapsed
        pn_a, pn_c = param_norms()
        if fp16_update:
            # ONE host sync (same as the .item() it replaces) reads the
            # loss, the overflow audit, the ratio diagnostic, the
            # log-ratio forensic, and the live scale together
            # "return" (2026-08-24, user): reward accumulated THIS
            # update, mean per env over the T-tick rollout — the
            # population analog of acc, unlike the episode-terminated
            # ep_return(last100) trailing window.
            # ONE cat -> ONE .tolist() -> ONE sync, exactly as before:
            # the 0.3 diagnostics ride the sync that already existed,
            # they do not add one (that is the whole design constraint).
            _d = torch.cat([
                torch.stack(
                    [ent_loss.detach(), ev, inf_count[0], ratio_diag,
                     logr_max, fp16_scale,
                     b_rew.sum() / b_rew.shape[1],
                     kl_first, kl_last, cf_first, cf_last,
                     gnorm_sum / max(opt_steps, 1),
                     clipc_sum / max(opt_steps, 1), af_ratio,
                     pn_a, pn_c]),
                ent_head]).tolist()
            (ent_v, ev_v, inf_v, r1_v, lmax_v, scale_v, ret_v,
             klf_v, kll_v, cff_v, cfl_v, gn_v, cc_v, af_v,
             pna_v, pnc_v) = _d[:16]
            eh_v = _d[16:]
            logr_max.zero_()
            gnorm_sum.zero_()
            clipc_sum.zero_()
            opt_steps = 0
            extra = (f"  r1 {r1_v:.1e}  maxlr {lmax_v:.1f}  "
                     f"pn {pna_v:.1f}/{pnc_v:.1f}")
            if inf_v > 0:
                inf_count.zero_()
                extra += (f"  [fp16] {int(inf_v)} overflow step(s) "
                          f"zeroed; scale {scale_v:.0f}")
            elif scale_v != scale_cap:
                extra += f"  scale {scale_v:.0f}"
            print(f"update {update+1:>3}/{num_updates}  steps "
                  f"{global_step:>9}  acc {acc_now:.4f}  "
                  f"accL {acc_l:.4f}  accR {acc_r:.4f}  "
                  f"return {ret_v:7.3f}  "
                  f"ent {ent_v:.3f}  ev {ev_v:.3f}  {rate:8,.0f} steps/s  "
                  f"{elapsed:6.1f}s{extra}", flush=True)
            print(format_kl_line(update + 1, klf_v, kll_v, cff_v, cfl_v,
                                 gn_v, cc_v, eh_v,
                                 af_v if adv_filter_on else None,
                                 kl_stop=kl_stop_mb),
                  flush=True)
        else:
            # one stacked read = one sync, same as the old ent .item()
            _d = torch.cat([
                torch.stack(
                    [ent_loss.detach(), ev,
                     b_rew.sum() / b_rew.shape[1],
                     kl_first, kl_last, cf_first, cf_last,
                     gnorm_sum / max(opt_steps, 1),
                     clipc_sum / max(opt_steps, 1), af_ratio,
                     pn_a, pn_c]),
                ent_head]).tolist()
            (ent_v, ev_v, ret_v, klf_v, kll_v, cff_v, cfl_v,
             gn_v, cc_v, af_v, pna_v, pnc_v) = _d[:12]
            eh_v = _d[12:]
            gnorm_sum.zero_()
            clipc_sum.zero_()
            opt_steps = 0
            print(f"update {update+1:>3}/{num_updates}  steps {global_step:>9}  "
                  f"acc {acc_now:.4f}  accL {acc_l:.4f}  accR {acc_r:.4f}  "
                  f"return {ret_v:7.3f}  "
                  f"ent {ent_v:.3f}  "
                  f"ev {ev_v:.3f}  {rate:8,.0f} steps/s  {elapsed:6.1f}s  "
                  f"pn {pna_v:.1f}/{pnc_v:.1f}",
                  flush=True)
            print(format_kl_line(update + 1, klf_v, kll_v, cff_v, cfl_v,
                                 gn_v, cc_v, eh_v,
                                 af_v if adv_filter_on else None,
                                 kl_stop=kl_stop_mb),
                  flush=True)
        if gn_sq:
            # (both print branches above lead here)
            gng_v = {k: float(v.sqrt()) for k, v in gn_sq.items()}
            print(f"[gnorm] upd {update + 1}  total "
                  f"{sum(v * v for v in gng_v.values()) ** 0.5:.2f}  "
                  + "  ".join(f"{k} {v:.2f}" for k, v in
                              sorted(gng_v.items(), key=lambda kv: -kv[1])),
                  flush=True)
            m_diag = dict(m_diag or {})
            m_diag["gn"] = gng_v
            # per-INPUT-LANE norm of the note encoder's first-layer
            # gradient: which of the 18 token lanes (+14 note-geo lanes)
            # owns the note_mlp gradient (2026-09-20, v52-v54)
            _w0 = policy.note_mlp[0].weight
            if _w0.grad is not None:
                _col = _w0.grad.detach().float().norm(dim=0).tolist()
                _top = sorted(enumerate(_col), key=lambda kv: -kv[1])[:8]
                print(f"[gnorm-lanes] upd {update + 1}  note_mlp[0] "
                      f"{sum(v * v for v in _col) ** 0.5:.2f}  "
                      + "  ".join(f"lane{i} {v:.2f}" for i, v in _top),
                      flush=True)
                m_diag["gn_lanes"] = [round(v, 4) for v in _col]

        m_ckpt = m_vram = None
        if args.ckpt_every > 0 and (update + 1) % args.ckpt_every == 0:
            name = (f"leaf_from{leaf_from:05d}_upd{update+1:05d}.pt"
                    if leaf_from is not None
                    else f"trunk_upd{update+1:05d}.pt")
            save_ckpt(os.path.join(args.run_dir, name), update + 1)
            m_ckpt = name
            if device.type == "cuda":
                # VRAM audit (GPU-tier decision prereq, 2026-08-19)
                m_vram = torch.cuda.max_memory_allocated() / 2**30
                print(f"[ckpt] {name}  peak-vram "
                      f"{torch.cuda.max_memory_allocated() / 2**30:.2f} "
                      f"GiB", flush=True)
        # the metrics line: the same host values the two prints above
        # consumed (plus this update's event counters, [diag] and
        # [ckpt]) — no extra sync, and write_metrics is never fatal
        write_metrics(
            update + 1, num_updates, global_step, t=time.time(),
            elapsed=elapsed, rate=rate, acc=acc_now, acc_l=acc_l,
            acc_pre=acc_pre, acc_post=acc_post, acc_prec=acc_prec,
            # ep_return(last100) dropped from the line and the record
            # (2026-09-12, user: too noisy; `return` over the update is
            # the stable population measure). Checkpoints still carry
            # recent_return for old tooling.
            acc_r=acc_r, ret=ret_v, ent=ent_v, ev=ev_v,
            r1=r1_v if fp16_update else None,
            maxlr=lmax_v if fp16_update else None,
            fp16_overflow=int(inf_v) if fp16_update else None,
            fp16_scale=scale_v if fp16_update else None,
            kl_first=klf_v, kl_last=kll_v, cf_first=cff_v, cf_last=cfl_v,
            gnorm=gn_v, clipc=cc_v, ent_head=eh_v,
            af_ratio=af_v if adv_filter_on else None,
            events=m_ev,
            pnorm_actor=pna_v, pnorm_critic=pnc_v, kl_stop=kl_stop_mb,
            diag=m_diag, ckpt=m_ckpt, peak_vram=m_vram)
        if args.time_cap > 0 and elapsed > args.time_cap:
            print(f"Reached --time-cap {args.time_cap:.0f}s.", flush=True)
            break

    recent = (np.mean(list(ep_returns)[-100:]) if ep_returns
                  else float("nan"))
    final_name = (f"leaf_from_upd{leaf_from:05d}.pt"
                  if leaf_from is not None else "trunk_final.pt")
    save_ckpt(os.path.join(args.run_dir, final_name), update + 1)
    if tick_n:
        f, e, k = (x / tick_n for x in tick_ms)
        print(f"[tick] fwd {f:.2f} / env {e:.2f} / book {k:.2f} ms "
              f"(sum {f + e + k:.2f} ms/tick, mean over {tick_n} ticks, "
              f"update 1 excluded)", flush=True)
    tot = t_rollout + t_update
    split_kind = "exec-true" if args.bench else \
        "enqueue-attributed, NOT execution time — use --bench"
    vram = ""
    if device.type == "cuda":
        peak = torch.cuda.max_memory_allocated()
        if args.bench:
            peak = max(peak, peak_roll, peak_upd)
        vram = f" | peak-vram {peak / 2**30:.2f} GiB"
        if args.bench:
            vram += (f" (rollout {peak_roll / 2**30:.2f} / update "
                     f"{peak_upd / 2**30:.2f}; resident at rollout start "
                     f"{res_pre / 2**30:.2f})")
    # seconds alongside the percentages (2026-09-02): integer percents
    # cannot resolve an update-phase A/B (a 10% update change is 3% of
    # the wall and rounds away), which is exactly what Tier 0.2 asked
    # this line for.
    print(f"THROUGHPUT: {(global_step - steps_before) / (time.time() - start):,.0f} steps/s total | "
          f"rollout {100*t_rollout/tot:.0f}% update {100*t_update/tot:.0f}% "
          f"({split_kind}) [{t_rollout:.2f}s / {t_update:.2f}s over "
          f"{num_updates - start_update} upd] | N={N} T={T}{vram}",
          flush=True)


if __name__ == "__main__":
    main()
