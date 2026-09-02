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
import torch
import torch.nn as nn

from beat_arena_env.vec_env import BeatVecEnv
from fp16_linear import bump_weights_version
from policy import (HEAD_GROUPS, HEAD_SIZES, NotePolicy, OBS_DIM,
                    load_policy)

A_DIM = len(HEAD_SIZES)  # 29 factored joint heads (env v10)

GAMMA = 0.99
GAE_LAMBDA = 0.95
CLIP = 0.2
DUAL_CLIP_C = 3.0
VF_COEF = 0.5
# value-target normalization EMA rate, per update (2026-08-19 critic
# redesign, topics/critic-design.md T1): the critic predicts
# NORMALIZED values v_hat; V = v_hat*sig+mu everywhere GAE reads it.
# Fixes the silent VF_COEF re-weighting under return-scale shifts
# (v26 gamma 0.99 inflated v_loss ~10x into the shared trunk).
# halflife ~69 updates at 0.01.
VALUE_NORM_BETA = 0.01
LR = 3e-4
MAX_GRAD_NORM = 0.5
# the recipe SHAPE. STALE as of 2026-08-31: no production run has used
# 1024x128 since the 2026-08-08 shape ladder. The live shapes are
# 32,768x8 (batch 262,144, the v27-v38 champion recipe) and 32,768x32
# (batch 1,048,576, the T=32 era), so the warning below now fires on
# every legitimate run and is pure noise. Changing the constant is a
# BEHAVIOUR change (it decides who gets warned) and needs a call on
# whether the guard should carry a set of admissible shapes or be
# retired; left alone deliberately.
RECIPE_BATCH = 1024 * 128
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


def resolve_env_order(flag, environ):
    """R12 env-order resolution (2026-09-02, Tier 0.1). `flag` is
    args.env_order: True/False if --env-order/--no-env-order was given,
    None if neither was. Precedence: explicit flag > a BEAT_ENV_ORDER
    already in the shell > ON.

    The shell variable is kept in the chain (rather than the flag simply
    defaulting True) so the standalone env-var contract other tools and
    the 2026-08-21 A/B harness use still works, and so a pod that
    exports BEAT_ENV_ORDER=0 to quarantine the permutation is not
    silently overridden by the new default."""
    if flag is None:
        return environ.get("BEAT_ENV_ORDER", "1") == "1"
    return bool(flag)


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
                   af_ratio=None):
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
            f"  entfull {total:.3f}{af}")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--num-envs", type=int, default=128)
    p.add_argument("--horizon", type=int, default=128)
    # e1/mb8 DEFAULT since 2026-07-26 (user decision after the paired
    # 50M gate): same 8 gradient steps/update as the old e2/mb4 at HALF
    # the update compute — measured +110.9 wpp (dAcc t=+12.88,
    # score_acc +32.5%) AND 1.61x wall (249.5k steps/s). The 07-21
    # "single-pass fails" study conflated sample reuse with
    # gradient-step density; density was what mattered.
    p.add_argument("--minibatches", type=int, default=8)
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
    p.add_argument("--warmup-updates", type=int, default=0,
                   help="linear lr warmup length in updates (0 = auto, "
                        "1.5%% of the run's updates); stored in the "
                        "checkpoint schedule so resumes agree")
    # 2026-08-19 critic redesign (topics/critic-design.md T0-T4)
    p.add_argument("--vf-coef", type=float, default=VF_COEF,
                   help="value-loss weight; targets are normalized, so "
                        "this is a real relative weight now")
    p.add_argument("--critic-lr-mult", type=float, default=3.0,
                   help="critic param-group lr multiplier (T1, "
                        "ProtoMotions ships 5x; 1.0 disables). Inactive "
                        "when resuming a legacy single-group optimizer")
    p.add_argument("--critic-detach", action="store_true",
                   help="T4: stop value-loss gradients at the shared "
                        "trunk boundary (PPG-style detach)")
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
    p.add_argument("--adv-filter", type=float, default=1.0,
                   help="fraction of the rollout batch to train on, "
                        "selected by |advantage| rank (1.0 = off). "
                        "Rounded to a whole number of minibatches")
    p.add_argument("--z-clip-hi", type=float, default=12.0,
                   help="v28: obs z-lane upper clamp in meters (exact-"
                        "timing lookahead = z_clip_hi / note speed). "
                        "12.0 = every pre-v28 era; CUDA env only")
    p.add_argument("--init-critic-fresh", action="store_true",
                   help="with --init-from and a pre-redesign checkpoint: "
                        "load actor/trunk weights, build a FRESH "
                        "adapool_v1 critic (cross-era warm start)")
    p.add_argument("--cooldown-updates", type=int, default=0,
                   help="wsd LEAF mode: with --resume <trunk ckpt>, branch "
                        "off and decay lr 1-sqrt from --lr to --lr-final "
                        "(and ent linearly from --ent-coef to --ent-final) "
                        "over this many updates. Saves "
                        "leaf_fromNNNNN_*.pt; leaves are terminal "
                        "deliverables — continue training from the TRUNK "
                        "checkpoint, never from a leaf")
    p.add_argument("--ent-coef", type=float, default=0.01)
    p.add_argument("--ent-final", type=float, default=None,
                   help="leaf: the cooldown's entropy target; trunk with "
                        "--ent-tau: the anneal FLOOR. Ignored (with a "
                        "notice) on a constant-ent trunk")
    p.add_argument("--ent-tau", type=int, default=0,
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
                   help="discount (default 0.99 ~ 3.3 s effective horizon "
                        "at 30 Hz; swept 2026-07-27 — user hypothesis: "
                        "Beat Saber decisions live ~1 s ahead, gamma "
                        "~0.967)")
    p.add_argument("--gae-lambda", type=float, default=GAE_LAMBDA,
                   help="GAE trace decay (default 0.95; swept 2026-07-27 "
                        "jointly with --gamma — gamma*lambda sets the "
                        "policy-gradient credit window, gamma alone the "
                        "value-target horizon)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--ckpt-every", type=int, default=0)
    p.add_argument("--run-dir", type=str, default=None)
    p.add_argument("--init-from", type=str, default=None,
                   help="warm-start WEIGHTS only; fresh optimizer + schedule")
    p.add_argument("--resume", type=str, default=None,
                   help="EXACT resume from a checkpoint of this trainer: "
                        "restores weights, Adam state, and the original LR "
                        "schedule position (lr/lr-final/lr-decay/total-steps "
                        "come from the checkpoint, CLI values ignored). The "
                        "env/chart stream restarts fresh — only the "
                        "optimization state is exact.")
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
    # R12 warp-coherence env permutation (cuda_env.py:608, the
    # dynamics-key argsort). Promoted from a bare BEAT_ENV_ORDER env
    # var to a real flag 2026-09-02 (Tier 0.1 of the 09-02 static
    # audit): it was default OFF and no launcher ever set it, so the
    # +5.4% measured on the previous pod (2026-08-21, 123,081 vs
    # 116,804 steps/s) was never actually collected by a campaign.
    # Per-env results are bitwise-identical under ANY thread->env
    # bijection (tests/test_env_order_bitwise.py), so this is a pure
    # throughput knob. Default ON. Resolution order: explicit flag >
    # a pre-existing BEAT_ENV_ORDER in the shell > ON. The trainer
    # re-exports the resolved value into os.environ BEFORE the env is
    # constructed, so cuda_env's reader is untouched and the env var
    # keeps working standalone (the bitwise test drives
    # _env_order_on directly and is unaffected either way).
    p.add_argument("--env-order", dest="env_order", action="store_true",
                   default=None,
                   help="R12 warp-coherence env permutation (--env cuda; "
                        "bitwise identical per env). Default ON; "
                        "overrides BEAT_ENV_ORDER")
    p.add_argument("--no-env-order", dest="env_order", action="store_false",
                   help="disable the R12 permutation (identity ordering)")
    p.add_argument("--chart-pool", type=int, default=65536,
                   help="pregenerated chart pool size for --env cuda "
                        "(default 65536 since 2026-07-26, user decision "
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
    # gears 5 (was 138,642). The flags remain for ablations.
    p.add_argument("--embed", type=int, default=128)
    p.add_argument("--trunk-dim", type=int, default=256)
    p.add_argument("--n-blocks", type=int, default=3)
    p.add_argument("--n-heads", type=int, default=4)
    p.add_argument("--ffn-mult", type=int, default=4)
    # env config (generator v2 dials, SPEC.md §Chart generation)
    # --colors REMOVED (2026-07-26, user decision): the right-only
    # default silently trained gen3's 5B run one-handed (49.9% of
    # real-map notes uncuttable — the whole transfer regression).
    # Training is always both-hands now; ChartConfig still accepts
    # colors="right" programmatically for tests/curricula.
    p.add_argument("--cols", type=int, nargs="+", default=[0, 1, 2, 3])
    p.add_argument("--rows", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--chart-len", type=float, default=20.0)
    p.add_argument("--note-speed-min", type=float, default=2.0,
                   help="speed random-walk clamp (m/s)")
    p.add_argument("--note-speed-max", type=float, default=8.0)
    p.add_argument("--note-prob", type=float, default=0.25,
                   help="P(note) per hand path per ~0.15 s generation tick")
    p.add_argument("--extra-prob", type=float, default=0.125,
                   help="P(a spawned note brings an extra along its swing line)")
    p.add_argument("--dot-prob", type=float, default=0.1,
                   help="P(a generated note turns into a dot)")
    p.add_argument("--bomb-prob", type=float, default=0.125,
                   help="P(bomb) per generation tick")
    p.add_argument("--effort-coef", type=float, default=1.0,
                   help="scales the tiered per-joint effort cost "
                        "(EFFORT_COST; 0 = free movement)")
    # wide-acc training reward (2026-07-29, user-directed): acc_train =
    # 30*clamp(1 - d/0.6), unrounded — effectively DOUBLING the acc
    # component's range and reward vs the real round(15*(1 - d/0.3)).
    # Head-to-head at 400M it LOST to real scoring on every bench metric
    # but was the best takeoff accelerator measured (+46% upp at 50M,
    # crossover ~200M) — default OFF since the cut-plane accuracy fix
    # (same day: the real formula now has reachable lateral gradient);
    # opt in for short-budget legs. Training gym only either way.
    p.add_argument("--wide-acc", default=False,
                   action=argparse.BooleanOptionalAction,
                   help="train with the wide accuracy reward "
                        "(30*clamp(1 - d/0.6), unrounded; best for short "
                        "50-150M legs, loses at 400M)")
    p.add_argument("--gears", type=int, default=3, choices=[3, 5, 7],
                   help="env v8 action-table size per joint head: 3 = "
                        "ternary coast/+full/-full; 5 adds +/-1/4-torque "
                        "gears; 7 adds +/-1/3 and +/-1/8 (fine control "
                        "under the doubled alpha/omega body). Resuming a "
                        "geared checkpoint needs the matching --gears")
    p.add_argument("--foot-contact", default=True,
                   action=argparse.BooleanOptionalAction,
                   help="DEPRECATED NO-OP (2026-08-16, user decision): "
                        "contact mode (env v11 foot_mode = 1) is the ONLY "
                        "training mode — the flag is accepted so older "
                        "drivers keep running, and --no-foot-contact is "
                        "IGNORED with a notice. Weld mode survives solely "
                        "as an internal test harness. Historical text: "
                        "PHYSICS-REGIME BREAK — "
                        "checkpoints and bench numbers do not carry "
                        "across the weld/contact boundary")
    p.add_argument("--ranked-maps", action="store_true",
                   help="train on SEGMENTS of the real ranked ScoreSaber "
                        "catalog (beat_arena_env/ranked_pool.py; bench "
                        "parsing protocol, real NJS speeds since "
                        "2026-07-30) instead of the procedural "
                        "generator. --chart-pool becomes the segment "
                        "subsample size; generator dials "
                        "(--note-prob/--bomb-prob/...) are ignored. "
                        "Requires --env cuda. (sweep s2, 2026-07-29)")
    p.add_argument("--lead-s", type=float, default=1.5,
                   help="ranked-maps segment head gap (s) — the earliest "
                        "first note of an episode (env v16, contact spec "
                        "§11: shorter keeps note data in the mix when "
                        "falls spike, damping survival-drift)")
    p.add_argument("--seg-len", type=float, default=25.0,
                   help="ranked-maps segment length in song seconds")
    p.add_argument("--mirror-augment", action="store_true",
                   help="double the ranked-maps pool with left-right "
                        "mirrored copies of every segment (handedness-"
                        "bias regularizer, 2026-08-14; --chart-pool "
                        "subsamples the doubled pool)")
    p.add_argument("--fall-pen", type=float, default=5.0,
                   help="env v18: constant fall penalty (return units, "
                        "positive; terminal reward = -value). DEFAULT is "
                        "the 3-arm-probe winner (2026-08-15: falls/update "
                        "8-10k -> ~35, bench zeros 35 -> 2); the legacy "
                        "remaining-notes charge is REMOVED")
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
    p.add_argument("--foot-hinge-r", type=float, default=0.0,
                   help="env v22: free radius (m) for BOTH foot penalties "
                        "— deviation inside it costs nothing (target "
                        "AREAS), rent applies to the excess. 0 = the v21 "
                        "pure-linear behavior")
    p.add_argument("--foot-shape-coef", type=float, default=0.0,
                   help="env v22: potential-based delta shaping on the "
                        "total hinged foot deviation — reward/punish each "
                        "tick's CHANGE in deviation (round trips net "
                        "zero; sharpens credit assignment; provably "
                        "optimum-preserving). 0 = off")
    p.add_argument("--limit-coef", type=float, default=0.0,
                   help="env v34: per-tick price on the joint-limit-row "
                        "impulse (N*m*s) the POSTURAL stops carry (spine "
                        "+ legs; arms exempt) — resting a pose on its "
                        "stops pays its gravity moment, an active hold "
                        "does not. 0 = off (bitwise pre-v34 rewards)")
    p.add_argument("--limit-arms", action="store_true",
                   help="env v35: widen --limit-coef's scope to the 14 "
                        "arm joints as well (all-body pricing; default "
                        "postural-only keeps whip stop-slams free)")
    p.add_argument("--wall-despawn", action="store_true",
                   help="env v37: walls despawn on the charged hit "
                        "(bomb-like; training-only credit aid)")
    p.add_argument("--iso-coef", type=float, default=0.0,
                   help="env v23: per-tick activation cost — iso_coef * "
                        "sum of commanded |gear fraction| over the 29 "
                        "joints (muscle-activation analog: isometric "
                        "holds and limit-jamming pay, coast is free; "
                        "velocity-blind, unlike --effort-coef's "
                        "positive-work price). 0 = off")
    p.add_argument("--foot-x-coef", type=float, default=0.0,
                   help="env v21: per-tick penalty coefficient on each "
                        "foot's |world_x - its_stance_x| (per-side "
                        "targets +-FOOT_X0; targets lateral wander). "
                        "0 = off")
    p.add_argument("--foot-z-coef", type=float, default=0.0,
                   help="env v20: per-tick penalty coefficient on each "
                        "foot's |world_z - reference_z| (linear; targets "
                        "the toward/away-from-notes shimmy). 0 = off")
    p.add_argument("--crouch-start-prob", type=float, default=0.0,
                   help="env v38: fraction of resets that spawn in a "
                        "physics-validated crouch (continuous depth, "
                        "knee 0.3-1.5 rad; posture-diversity fix)")
    p.add_argument("--start-pose-jitter", type=float, default=0.0,
                   help="env v20 (CUDA training only): episode-start leg/"
                        "foot joint jitter scale in radians — 12 leg "
                        "joints sampled U(-s, s) (knees |U|), clamped to "
                        "limits, root dropped so no sole corner clips the "
                        "ground. Teaches knee bend / balance variety")
    return p.parse_args()


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


def main():
    args = parse_args()
    # env v21 (2026-08-16, user decision): weld mode is REMOVED from the
    # production surface — contact is the only mode this trainer runs.
    if not args.foot_contact:
        print("[foot-contact] --no-foot-contact is a deprecated no-op: "
              "contact mode is the only training mode (weld = test "
              "harness only)", flush=True)
        args.foot_contact = True
    if args.bench:
        args.total_steps = args.num_envs * args.horizon * 4
    resume_ckpt = None
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
        args.lr, args.lr_final = sched["lr"], sched["lr_final"]
        # trunk entropy curve is checkpoint state, like lr: an extension
        # must continue the exact curve, and a leaf must start from the
        # trunk's live value (--ent-final stays CLI: the leaf's target)
        args.ent_coef = sched.get("ent_coef", args.ent_coef)
        args.ent_tau = sched.get("ent_tau", 0)
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
    # R12 env-order resolution, BEFORE any env is constructed (cuda_env
    # snapshots os.environ in __init__). Explicit flag wins; otherwise
    # a BEAT_ENV_ORDER already in the shell wins; otherwise ON.
    args.env_order = resolve_env_order(args.env_order, os.environ)
    os.environ["BEAT_ENV_ORDER"] = "1" if args.env_order else "0"
    # device-side loss-scaling state: NO host syncs in the minibatch
    # loop (an overflow zeroes the grads device-side; the audit rides
    # the existing once-per-update print sync). The scale itself is a
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
    # sync: the minibatch loop still has ZERO host syncs, which is the
    # invariant the loss-scale machinery above is built around. Cost is
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
    args_snapshot = {k: v for k, v in sorted(vars(args).items())}
    with open(os.path.join(args.run_dir, "args.json"), "w",
              encoding="utf-8") as f:
        json.dump({"argv": sys.argv[1:], "resolved": args_snapshot},
                  f, indent=1, default=str)
    print(f"argv: {' '.join(sys.argv[1:]) or '(bare)'}", flush=True)
    print(f"config: {args_snapshot}", flush=True)

    # Wide-N hazard (2026-08-03): every run driver hardcodes the SHAPE
    # (--num-envs 1024 --horizon 128) because the tuned hypers and the
    # checkpoint cadences are calibrated to batch 131,072. Raising N
    # alone silently rescales the batch, so say so loudly. A warning,
    # not an error: --bench and the micro-shape tools are legitimately
    # off-recipe (and skip this entirely under --bench).
    if not args.bench and batch != RECIPE_BATCH:
        print(f"!!! BATCH {batch:,} = {N} envs x {T} horizon, NOT the "
              f"recipe {RECIPE_BATCH:,} — tuned hypers and checkpoint "
              f"cadences assume {RECIPE_BATCH:,} !!!", flush=True)

    if args.ranked_maps and args.env != "cuda":
        raise SystemExit("--ranked-maps requires --env cuda")
    if args.z_clip_hi != 12.0 and args.env != "cuda":
        raise SystemExit("--z-clip-hi: the rust backend reads "
                         "BEAT_Z_CLIP_HI instead (CUDA env only here)")
    if tensor_env and args.ranked_maps:
        from beat_arena_env.cuda_env import CudaBeatVecEnv as VecCls
        from beat_arena_env.ranked_pool import load_ranked_pool
        pool = load_ranked_pool(seg_len=args.seg_len,
                                sample=args.chart_pool, seed=args.seed,
                                device=device, lead_s=args.lead_s,
                                mirror=args.mirror_augment)
        venv = VecCls(N, pool=pool,
                      max_steps=int(args.seg_len * 30) + 90,
                      effort_coef=args.effort_coef,
                      wide_acc=args.wide_acc, gears=args.gears,
                      foot_contact=args.foot_contact,
                      fall_const=args.fall_pen,
                      push_prob=args.push_prob, push_vmax=args.push_vmax,
                      start_jitter=args.start_jitter,
                      foot_z_coef=args.foot_z_coef,
                      foot_x_coef=args.foot_x_coef,
                      foot_hinge_r=args.foot_hinge_r,
                      foot_shape_coef=args.foot_shape_coef,
                      iso_coef=args.iso_coef,
                      limit_coef=args.limit_coef,
                      limit_arms=args.limit_arms,
                      wall_despawn=args.wall_despawn,
                      start_pose_jitter=args.start_pose_jitter,
                      crouch_start_prob=args.crouch_start_prob,
                      z_clip_hi=args.z_clip_hi,
                      seed=args.seed, device=device)
    elif tensor_env:
        from beat_arena_env.cuda_env import CudaBeatVecEnv as VecCls
        venv = VecCls(
            N, cols=tuple(args.cols),
            rows=tuple(args.rows), chart_len_s=args.chart_len,
            note_speed_min=args.note_speed_min,
            note_speed_max=args.note_speed_max,
            note_prob=args.note_prob, extra_prob=args.extra_prob,
            dot_prob=args.dot_prob, bomb_prob=args.bomb_prob,
            effort_coef=args.effort_coef, wide_acc=args.wide_acc,
            gears=args.gears, foot_contact=args.foot_contact,
            fall_const=args.fall_pen,
            push_prob=args.push_prob, push_vmax=args.push_vmax,
            start_jitter=args.start_jitter,
            foot_z_coef=args.foot_z_coef,
            foot_x_coef=args.foot_x_coef,
            foot_hinge_r=args.foot_hinge_r,
            foot_shape_coef=args.foot_shape_coef,
            iso_coef=args.iso_coef,
            start_pose_jitter=args.start_pose_jitter,
            crouch_start_prob=args.crouch_start_prob,
            z_clip_hi=args.z_clip_hi,
            seed=args.seed, chart_pool=args.chart_pool, device=device)
    else:
        venv = BeatVecEnv(
            N, gears=args.gears, cols=tuple(args.cols),
            rows=tuple(args.rows), chart_len_s=args.chart_len,
            note_speed_min=args.note_speed_min,
            note_speed_max=args.note_speed_max,
            note_prob=args.note_prob, extra_prob=args.extra_prob,
            dot_prob=args.dot_prob, bomb_prob=args.bomb_prob,
            effort_coef=args.effort_coef, wide_acc=args.wide_acc,
            foot_contact=args.foot_contact, fall_const=args.fall_pen,
            foot_z_coef=args.foot_z_coef,
            foot_x_coef=args.foot_x_coef,
            foot_hinge_r=args.foot_hinge_r,
            foot_shape_coef=args.foot_shape_coef,
            iso_coef=args.iso_coef,
            seed=args.seed)

    if resume_ckpt is not None:
        policy = load_policy(resume_ckpt)
    elif args.init_from:
        ckpt = torch.load(args.init_from, map_location="cpu", weights_only=True)
        policy = load_policy(ckpt)
        if args.init_critic_fresh and policy.critic_arch == "linear_v0":
            fresh = NotePolicy(
                embed=policy.arch["embed"],
                trunk_dim=policy.arch["trunk_dim"],
                n_blocks=policy.arch["n_blocks"],
                n_heads=policy.arch["n_heads"],
                n_gears=policy.arch["n_gears"],
                ffn_mult=policy.arch["ffn_mult"], critic="adapool_v1")
            missing, unexpected = fresh.load_state_dict(
                policy.state_dict(), strict=False)
            assert (all(k.startswith("critic_") for k in missing)
                    and all(k.startswith("value_head") for k in
                            unexpected)), (missing, unexpected)
            policy = fresh
            print("--init-critic-fresh: actor/trunk from the "
                  "checkpoint, FRESH adapool_v1 critic", flush=True)
        print(f"warm-started from {args.init_from} (arch {policy.arch})",
              flush=True)
    else:
        policy = NotePolicy(embed=args.embed, trunk_dim=args.trunk_dim,
                            n_blocks=args.n_blocks, n_heads=args.n_heads,
                            n_gears=args.gears, ffn_mult=args.ffn_mult)
    policy.critic_detach = args.critic_detach
    if args.critic_detach:
        print("--critic-detach: value-loss gradients stop at the "
              "shared-trunk boundary (T4)", flush=True)
    if policy.n_gears != args.gears:
        raise SystemExit(f"--gears {args.gears} but the checkpoint's heads "
                         f"are n_gears={policy.n_gears} — the env and "
                         f"policy action tables must match")
    n_params = sum(p.numel() for p in policy.parameters())
    print(f"policy arch {policy.arch} — {n_params:,} params", flush=True)
    policy = policy.to(device)
    policy.fp16 = args.fp16 == "all"
    if args.fp16 != "off":
        print(f"--fp16 {args.fp16}: fp16-storage activations, fp32 "
              f"masters, dynamic loss scale (init {args.fp16_scale:.0f}, "
              f"halve/overflow-mb, x2/{FP16_GROWTH_INTERVAL:.0f} clean)",
              flush=True)
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
    # (--fused-update was retired 2026-08-13 with the 138k arch: the
    # packed kernels were hard-wired to embed 64 / trunk 128 and the
    # eager fp16 path was the production path throughout v16.)
    # fused=True: one multi-tensor kernel instead of the foreach path's
    # ~10 launches per step (77 elementwise calls/minibatch measured in
    # the 2026-07-25 profile). Same per-element math, not guaranteed
    # bitwise vs foreach.
    fused_adam = device.type == "cuda"
    # T1 critic lr split: the critic params (adapool modules, or the
    # legacy value_head) form their own group with an lr multiplier the
    # schedule loop re-applies every update via the "lr_mult" key
    # (extra param_group keys survive state_dict round trips).
    crit_p = [p for n, p in policy.named_parameters()
              if n.startswith(("critic_", "value_head"))]
    main_p = [p for n, p in policy.named_parameters()
              if not n.startswith(("critic_", "value_head"))]
    opt = torch.optim.Adam(
        [{"params": main_p, "lr": args.lr, "lr_mult": 1.0},
         {"params": crit_p, "lr": args.lr * args.critic_lr_mult,
          "lr_mult": args.critic_lr_mult}],
        lr=args.lr, fused=fused_adam)
    if resume_ckpt is not None and len(
            resume_ckpt["optimizer"]["param_groups"]) != 2:
        print("resume: legacy single-group optimizer state — "
              "--critic-lr-mult inactive on this run", flush=True)
        opt = torch.optim.Adam(policy.parameters(), lr=args.lr,
                               fused=fused_adam)
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
        diag_groups = [("enc", 0, name_off.get("blocks.0.ln_attn.weight",
                                               shared_end))]
        for i in range(len(policy.blocks)):
            lo = name_off.get(f"blocks.{i}.ln_attn.weight")
            hi = name_off.get(f"blocks.{i + 1}.ln_attn.weight",
                              name_off.get("trunk.0.weight", shared_end))
            if lo is not None:
                diag_groups.append((f"blk{i}", lo, hi))
        diag_groups.append(("trunk", name_off.get("trunk.0.weight",
                                                  shared_end), shared_end))

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
    if resume_ckpt is not None:
        vn = resume_ckpt.get("value_norm")
        if vn is not None:
            ret_mu.fill_(vn["mu"])
            ret_var.fill_(vn["var"])
            torch.sqrt(ret_var.clamp_min(1e-8), out=ret_sig)
            value_norm_init = False
        else:
            print("!!! resume: checkpoint predates value normalization "
                  "— the critic will RE-FIT to normalized targets over "
                  "the first ~100 updates (EV dips, then recovers) !!!",
                  flush=True)

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
            extended = max(1, args.total_steps // batch)
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
    diag_probe_obs = None    # frozen first-diag batch (psrank probe)
    start = time.time()

    filt = (f", adv-filter top |A| {train_n}/{batch} "
            f"({train_n // mb_size} minibatches, full-batch adv norm)"
            if adv_filter_on else "")
    # env-order state is on the banner because it was invisible for the
    # 12 days it existed (2026-08-21 -> 09-02): nothing printed it and
    # nothing set it, so no log ever said which ordering a run used.
    # Read off the ENV, not off args — a banner that reports the request
    # rather than the fact is how this knob stayed dark in the first
    # place; the assert catches a future env that stops reading the var.
    eo_live = getattr(venv, "_env_order_on", None)
    if eo_live is not None:
        assert eo_live == bool(args.env_order), (
            f"--env-order {args.env_order} did not reach the env "
            f"(_env_order_on={eo_live})")
    eo = "" if eo_live is None else \
        f", env-order {'on' if eo_live else 'off'}"
    print(f"=== batched PPO: {N} envs x {T} horizon = {batch} steps/update, "
          f"minibatch {mb_size}, device {device}{filt}{eo} ===", flush=True)

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
        for g in opt.param_groups:
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
        for t in range(T):
            b_obs[t] = obs
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
            b_val[t].copy_(v).mul_(ret_sig).add_(ret_mu)
            if tensor_env:
                nobs, rew, done, infos = venv.step(a)
                b_rew[t] = rew.to(b_rew.dtype)
                b_done[t].copy_(done)
                obs = nobs
            else:
                nobs, rew, done, infos = venv.step(a.cpu().numpy())
                b_rew[t] = torch.as_tensor(rew, device=device)
                b_done[t] = torch.as_tensor(
                    done.astype(np.float32), device=device)
                obs = torch.as_tensor(nobs, dtype=torch.float32, device=device)
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
            # beta = VALUE_NORM_BETA step.
            r_mu, r_var = returns.mean(), returns.var()
            if value_norm_init:
                ret_mu.copy_(r_mu)
                ret_var.copy_(r_var)
                value_norm_init = False
            else:
                ret_mu.mul_(1.0 - VALUE_NORM_BETA).add_(
                    VALUE_NORM_BETA * r_mu)
                ret_var.mul_(1.0 - VALUE_NORM_BETA).add_(
                    VALUE_NORM_BETA * r_var)
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

        if (args.diag_every > 0 and flat_grad is not None
                and update % args.diag_every == 0):
            # T0 diagnostics (2026-08-19, topics/critic-design.md): a
            # 2048-row grad-enabled forward + TWO extra backwards,
            # separating the policy-loss and value-loss gradients over
            # the SHARED params (flat buffer [0, shared_end)) — their
            # cosine discriminates actor/critic interference from
            # underfitting; per-group norms + dormant-neuron fraction +
            # trunk-feature effective rank discriminate critic
            # interference from gradient-propagation failure
            # (2506.15544's critic-independent collapse mode). Costs
            # one host sync at its print — diag-cadence only.
            dn = min(2048, batch)
            policy._diag_stash = True
            _, dlogp, _, dval = policy.get_action_and_value(
                f_obs[:dn], f_act[:dn], need_entropy=False)
            policy._diag_stash = False
            dratio = (dlogp - f_logp[:dn]).clamp(
                max=LOG_RATIO_CLAMP).exp()
            da = f_adv[:dn]
            da = (da - da.mean()) / (da.std() + 1e-8)
            d_pg = pg_loss_fn(dratio, da)
            d_v = 0.5 * ((dval - (f_ret[:dn] - ret_mu) / ret_sig)
                         ** 2).mean()
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
            # Compare psrank only WITHIN one process; persisting
            # diag_probe_obs in the checkpoint would fix this.
            if diag_probe_obs is None:
                diag_probe_obs = f_obs[:dn].clone()
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
            per_g = "  ".join(
                f"{nm} {g_pi[a:b].norm().item():.1e}/"
                f"{g_v[a:b].norm().item():.1e}"
                for nm, a, b in diag_groups)
            print(f"[diag] upd {update + 1}  gradcos {gcos.item():+.3f}"
                  f"  |g_pi| {n_pi.item():.2e}  |g_v| {n_v.item():.2e}"
                  f"  dormant {dorm.item():.3f}  srank {srank}/"
                  f"{h_d.shape[1]}  psrank {psrank}"
                  f"  pi/v: {per_g}", flush=True)

        for ep_i in range(args.epochs):
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
                micro = mb_size // args.mb_accum
                for j in range(args.mb_accum):
                    sl = slice(s + j * micro, s + (j + 1) * micro)
                    _, newlogp, entropy, newval = \
                        policy.get_action_and_value(p_obs[sl], p_act[sl])
                    logratio = newlogp - p_logp[sl]
                    ratio = logratio.clamp(max=LOG_RATIO_CLAMP).exp()
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
                    pg_loss = pg_loss_fn(ratio, a_mb)
                    # T1: the critic learns the NORMALIZED target
                    v_tgt = (p_ret[sl] - ret_mu) / ret_sig
                    v_loss = 0.5 * ((newval - v_tgt) ** 2).mean()
                    ent_loss = entropy.mean()
                    loss = (pg_loss - ent_now * ent_loss
                            + args.vf_coef * v_loss)
                    if args.mb_accum > 1:
                        loss = loss / args.mb_accum
                    if fp16_update:
                        (loss * fp16_scale).backward()
                    else:
                        loss.backward()
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
                opt.step()
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

        if tensor_env:
            # deferred from the rollout->GAE boundary: this is the one
            # place per update the host reads the device anyway
            ep_returns.extend(venv.pop_stats())
            # f32 Cholesky pivot-floor engagements (SPEC-v10 §20e):
            # reported, never hidden
            if hasattr(venv, "pop_pd_hits"):
                pd = venv.pop_pd_hits()
                if pd:
                    print(f"[pd-net] {pd} f32 pivot-floor engagement(s) "
                          "this update", flush=True)
            # pre-integration root-envelope engagements (SPEC-v10
            # Divergences §23): each one terminated an env's episode with
            # a zeroed reward (the C backstop). Reported, never hidden —
            # a converged policy measures 0.
            if hasattr(venv, "pop_envelope_hits"):
                eh = venv.pop_envelope_hits()
                if eh:
                    print(f"[envelope] {eh} root-escape engagement(s) "
                          "this update (episodes terminated, reward 0)",
                          flush=True)
        # env v11 FALL terminations (unilateral-contact spec §4): the
        # root-escape envelope's twin — the body went to the floor.
        # v11.1: each fall is PRICED at REWARD_MISS x the remaining
        # scoreable notes (the reward-0 version bred the 8h suicide
        # optimum). Reported, never hidden, on BOTH backends (unlike
        # [envelope], a CUDA-only counter): structurally 0 without
        # --foot-contact.
        if hasattr(venv, "pop_fall_hits"):
            fh = venv.pop_fall_hits()
            if fh:
                print(f"[fall] {fh} fall termination(s) this update "
                      "(each priced at the flat fall_const)",
                      flush=True)
        if hasattr(venv, "pop_tilt_hits"):
            th = venv.pop_tilt_hits()
            if th:
                print(f"[tilt] {th} tilt termination(s) this update "
                      "(chest up-axis fwd>40 / back>20 / lateral>40 deg; "
                      "flat fall_const)", flush=True)
        if hasattr(venv, "pop_foot_lifts"):
            lh_l, lh_r = venv.pop_foot_lifts()
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
                print(f"[flail] {(fl_l + fl_r) / fl_ticks:.4f} "
                      f"rad/idle-tick (L {fl_l / fl_ticks:.4f} / "
                      f"R {fl_r / fl_ticks:.4f}; {fl_ticks} note-free "
                      "env-ticks)", flush=True)
        # limit-rest telemetry: mean limit-row lambda-flow per
        # env-tick (N*m*s), legs / spine / arms split — sustained flow
        # means the pose is resting on its stops. Arms accumulate
        # always; they are PRICED only under --limit-arms (env v35).
        if hasattr(venv, "pop_limit_rest"):
            lr_l, lr_s, lr_a, lr_t = venv.pop_limit_rest()
            if lr_t:
                print(f"[limitrest] {(lr_l + lr_s + lr_a) / lr_t:.4f} "
                      f"N*m*s/tick (legs {lr_l / lr_t:.4f} / spine "
                      f"{lr_s / lr_t:.4f} / arms {lr_a / lr_t:.4f})",
                      flush=True)
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
        recent = (np.mean(list(ep_returns)[-100:]) if ep_returns
                  else float("nan"))
        elapsed = time.time() - start
        rate = (global_step - steps_before) / elapsed
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
                     clipc_sum / max(opt_steps, 1), af_ratio]),
                ent_head]).tolist()
            (ent_v, ev_v, inf_v, r1_v, lmax_v, scale_v, ret_v,
             klf_v, kll_v, cff_v, cfl_v, gn_v, cc_v, af_v) = _d[:14]
            eh_v = _d[14:]
            logr_max.zero_()
            gnorm_sum.zero_()
            clipc_sum.zero_()
            opt_steps = 0
            extra = f"  r1 {r1_v:.1e}  maxlr {lmax_v:.1f}"
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
                  f"ep_return(last100) {recent:8.3f}  "
                  f"ent {ent_v:.3f}  ev {ev_v:.3f}  {rate:8,.0f} steps/s  "
                  f"{elapsed:6.1f}s{extra}", flush=True)
            print(format_kl_line(update + 1, klf_v, kll_v, cff_v, cfl_v,
                                 gn_v, cc_v, eh_v,
                                 af_v if adv_filter_on else None),
                  flush=True)
        else:
            # one stacked read = one sync, same as the old ent .item()
            _d = torch.cat([
                torch.stack(
                    [ent_loss.detach(), ev,
                     b_rew.sum() / b_rew.shape[1],
                     kl_first, kl_last, cf_first, cf_last,
                     gnorm_sum / max(opt_steps, 1),
                     clipc_sum / max(opt_steps, 1), af_ratio]),
                ent_head]).tolist()
            (ent_v, ev_v, ret_v, klf_v, kll_v, cff_v, cfl_v,
             gn_v, cc_v, af_v) = _d[:10]
            eh_v = _d[10:]
            gnorm_sum.zero_()
            clipc_sum.zero_()
            opt_steps = 0
            print(f"update {update+1:>3}/{num_updates}  steps {global_step:>9}  "
                  f"acc {acc_now:.4f}  accL {acc_l:.4f}  accR {acc_r:.4f}  "
                  f"return {ret_v:7.3f}  "
                  f"ep_return(last100) {recent:8.3f}  ent {ent_v:.3f}  "
                  f"ev {ev_v:.3f}  {rate:8,.0f} steps/s  {elapsed:6.1f}s",
                  flush=True)
            print(format_kl_line(update + 1, klf_v, kll_v, cff_v, cfl_v,
                                 gn_v, cc_v, eh_v,
                                 af_v if adv_filter_on else None),
                  flush=True)

        if args.ckpt_every > 0 and (update + 1) % args.ckpt_every == 0:
            name = (f"leaf_from{leaf_from:05d}_upd{update+1:05d}.pt"
                    if leaf_from is not None
                    else f"trunk_upd{update+1:05d}.pt")
            torch.save({"state_dict": cpu_state(policy), "arch": policy.arch,
                        "encoder": policy.encoder,
                        "update": update + 1, "global_step": global_step,
                        "recent_return": float(recent),
                        "recent_acc": float(acc_now),
                        "value_norm": {"mu": float(ret_mu),
                                       "var": float(ret_var)},
                        "optimizer": cpu_opt_state(opt),
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
                       os.path.join(args.run_dir, name))
            if device.type == "cuda":
                # VRAM audit (GPU-tier decision prereq, 2026-08-19)
                print(f"[ckpt] {name}  peak-vram "
                      f"{torch.cuda.max_memory_allocated() / 2**30:.2f} "
                      f"GiB", flush=True)
        if args.time_cap > 0 and elapsed > args.time_cap:
            print(f"Reached --time-cap {args.time_cap:.0f}s.", flush=True)
            break

    recent = (np.mean(list(ep_returns)[-100:]) if ep_returns
                  else float("nan"))
    final_name = (f"leaf_from_upd{leaf_from:05d}.pt"
                  if leaf_from is not None else "trunk_final.pt")
    torch.save({"state_dict": cpu_state(policy), "arch": policy.arch,
                "encoder": policy.encoder,
                "update": update + 1, "global_step": global_step,
                "recent_return": float(recent),
                "recent_acc": float(acc_now),
                "value_norm": {"mu": float(ret_mu),
                               "var": float(ret_var)},
                "optimizer": cpu_opt_state(opt),
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
               os.path.join(args.run_dir, final_name))
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
