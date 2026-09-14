"""metrics_log: the per-update JSON schema and its stdout parser
(2026-09-03, dashboard option B).

Pinned here:

1. Formatter and parser share ONE vocabulary: every key the parser can
   emit is in UPDATE_FIELDS, and a record built by `update_record`
   round-trips through `dumps_record` unchanged.
2. The parser reads every archive era verbatim (lines copied from
   ~/beat-arena-volume-backup): v11 (no acc/ev/return), v27 (+ev),
   v30 (+acc), v35 (+accL/R), v38 (+return), af50n (+[kl]), the
   fp16 overflow/scale suffixes, and the run-level banner/pool/
   RESUMED/WSD/THROUGHPUT lines. Events print BEFORE their update
   line, [diag] before the epoch loop, [ckpt] after — the join must
   respect that order.
3. Chain logs split into one segment per `argv:` invocation, and
   `merge_segments` resolves a resume by discarding the stale tail.
4. The entropy group arithmetic matches train_batch.entropy_report
   (torch-free duplicate of HEAD_GROUPS).
"""
import json
import math
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import metrics_log as ml  # noqa: E402

AF50N_HEAD = """\
argv: --num-envs 32768 --horizon 32 --minibatches 64 --mb-accum 2 --foot-contact --gears 5 --ranked-maps --seg-len 25 --chart-pool 999999 --mirror-augment --gamma 0.965 --gae-lambda 0.98 --z-clip-hi 40 --iso-coef 5e-4 --ent-coef 0.03 --ent-tau 125 --lead-s 0.5 --lr 7e-4 --start-pose-jitter 0.15 --foot-z-coef 0.01 --foot-x-coef 0.01 --foot-hinge-r 0.12 --foot-shape-coef 0.05 --limit-coef 5e-3 --limit-arms --crouch-start-prob 0.25 --seed 0 --ent-final 1e-3 --wall-despawn --warmup-updates 34 --total-steps 1119879168 --ckpt-every 125 --adv-filter 0.5 --run-dir runs/af50n_trunk
config: {'adv_filter': 0.5, 'bench': False, 'bomb_prob': 0.125, 'chart_len': 20.0}
--env not given: defaulting to 'cuda'
[ranked-pool] 93472 segments on cuda (C=1489, 7655 MB)
--fp16 all: fp16-storage activations, fp32 masters, dynamic loss scale (init 131072, halve/overflow-mb, x2/500 clean)
--fp16: flat master/replica buffers (1-kernel refresh)
policy arch {'embed': 64, 'trunk_dim': 256, 'n_blocks': 3, 'n_heads': 1} — 1,234,567 params
WSD TRUNK: warmup 34 updates -> constant lr 0.0007, ent 0.03 -> floor 0.001 exp(tau=125 upd) (open-ended; branch a leaf with --resume <trunk ckpt> --cooldown-updates N)
=== batched PPO: 32768 envs x 32 horizon = 1048576 steps/update, minibatch 16384, device cuda, adv-filter top |A| 524288/1048576 (32 minibatches, full-batch adv norm), env-order on ===
[diag] upd 1  gradcos +0.049  |g_pi| 8.14e-01  |g_v| 8.12e-02  dormant 0.000  srank 148/256  psrank 148  pi/v: enc 2.4e-01/3.2e-02  blk0 4.1e-01/5.0e-02  blk1 3.0e-01/3.9e-02  blk2 2.4e-01/1.6e-02  trunk 5.4e-01/3.5e-02
[fall] 22732 fall termination(s) this update (each priced at the flat fall_const)
[tilt] 60710 tilt termination(s) this update (chest up-axis fwd>40 / back>20 / lateral>40 deg; flat fall_const)
[lift] 247239 foot-lift event(s) this update (L 125031 / R 122208; each charged the flat -0.1)
[flail] 1.9150 rad/idle-tick (L 0.9561 / R 0.9588; 193862 note-free env-ticks)
[limitrest] 16.1259 N*m*s/tick (legs 7.6030 / spine 7.1072 / arms 1.4157)
[activation] 9.8123 /env-tick (spine 1.0210 / arms 5.6413 / legs 3.1500)
[torque] 391.125 N*m/env-tick (spine 69.900 / arms 93.600 / legs 227.625)
update   1/1068  steps   1048576  acc 0.6432  accL 0.6512  accR 0.6359  return -15.622  ep_return(last100)   -6.598  ent 46.560  ev 0.000   135,551 steps/s     7.7s  r1 2.7e-06  maxlr 0.6
[kl] upd   1  kl(mb1/mbN) 6.62e-10/7.02e-03  clipfrac 0.0000/0.0906  gnorm 0.441  clipc 0.945  ent/head spine 1.603  armL 1.605  armR 1.605  legL 1.604  legR 1.606  min 1.597 (h20)  entfull 46.548  afsd 0.175
[tilt] 60000 tilt termination(s) this update (chest up-axis fwd>40 / back>20 / lateral>40 deg; flat fall_const)
[lift] 247000 foot-lift event(s) this update (L 125000 / R 122000; each charged the flat -0.1)
[flail] 1.9000 rad/idle-tick (L 0.9500 / R 0.9500; 193000 note-free env-ticks)
[limitrest] 16.0000 N*m*s/tick (legs 7.6000 / spine 7.1000 / arms 1.4000)
[activation] 9.8000 /env-tick (spine 1.0000 / arms 5.6000 / legs 3.2000)
[torque] 390.000 N*m/env-tick (spine 70.000 / arms 93.000 / legs 227.000)
update   2/1068  steps   2097152  acc 0.6534  accL 0.6598  accR 0.6471  return -17.464  ep_return(last100)   -6.583  ent 46.555  ev 0.001   141,518 steps/s    14.8s  r1 2.2e-06  maxlr 0.6
[kl] upd   2  kl(mb1/mbN) 1.00e-09/8.00e-03  clipfrac 0.0000/0.1000  gnorm 0.450  clipc 0.940  ent/head spine 1.600  armL 1.600  armR 1.600  legL 1.600  legR 1.600  min 1.590 (h3)  entfull 46.400  afsd 0.180
[ckpt] trunk_upd00002.pt  peak-vram 12.17 GiB
THROUGHPUT: 203,864 steps/s total | rollout 63% update 37% (enqueue-attributed, NOT execution time — use --bench) [3423.00s / 2039.41s over 1068 upd] | N=32768 T=32 | peak-vram 12.17 GiB
"""


def test_af50n_era_full_join():
    segs = ml.parse_log(AF50N_HEAD, t_end=1000.0)
    assert len(segs) == 1
    s = segs[0]
    assert len(s.updates) == 2
    r1, r2 = s.updates
    # update line
    assert r1["upd"] == 1 and r1["n_upd"] == 1068
    assert r1["steps"] == 1048576
    assert r1["acc"] == 0.6432 and r1["accL"] == 0.6512
    assert r1["accR"] == 0.6359
    assert r1["return"] == -15.622
    assert r1["ep_return"] == -6.598
    assert r1["ent"] == 46.560 and r1["ev"] == 0.0
    assert r1["sps"] == 135551 and r1["elapsed"] == 7.7
    assert r1["r1"] == 2.7e-06 and r1["maxlr"] == 0.6
    # [kl]
    assert r1["kl_first"] == 6.62e-10 and r1["kl_last"] == 7.02e-03
    assert r1["clipfrac_first"] == 0.0 and r1["clipfrac_last"] == 0.0906
    assert r1["gnorm"] == 0.441 and r1["clipc"] == 0.945
    assert r1["ent_spine"] == 1.603 and r1["ent_legR"] == 1.606
    assert r1["ent_min"] == 1.597 and r1["ent_min_head"] == 20
    assert r1["entfull"] == 46.548 and r1["afsd"] == 0.175
    # events printed BEFORE the update line belong to it
    assert r1["falls"] == 22732 and r1["tilts"] == 60710
    assert r1["lifts"] == 247239 and r1["lifts_L"] == 125031
    assert r1["lifts_R"] == 122208
    assert r1["flail"] == 1.9150 and r1["flail_L"] == 0.9561
    assert r1["flail_ticks"] == 193862
    assert r1["limitrest"] == 16.1259 and r1["limitrest_arms"] == 1.4157
    assert r1["activation"] == 9.8123 and r1["activation_spine"] == 1.021
    assert r1["activation_arms"] == 5.6413 and r1["activation_legs"] == 3.15
    assert r2["activation"] == 9.8 and r2["activation_legs"] == 3.2
    # [torque] (env v43): applied sum|tau| N*m per env-tick + groups
    assert r1["torque"] == 391.125 and r1["torque_spine"] == 69.9
    assert r1["torque_arms"] == 93.6 and r1["torque_legs"] == 227.625
    assert r2["torque"] == 390.0 and r2["torque_legs"] == 227.0
    # update 2 had no [fall] line -> structural zero (era prints it)
    assert r2["falls"] == 0 and r2["tilts"] == 60000
    # [diag] attaches by its explicit upd index
    assert r1["diag_gradcos"] == 0.049 and r1["diag_srank"] == 148
    assert r1["diag_srank_dim"] == 256 and r1["diag_psrank"] == 148
    assert r1["diag_groups"]["blk2"] == [0.24, 0.016]
    assert "diag_gradcos" not in r2
    # [ckpt] attaches to the update named in the file
    assert r2["ckpt"] == "trunk_upd00002.pt" and r2["peak_vram"] == 12.17
    assert "ckpt" not in r1
    # wall-clock reconstruction from t_end
    assert r2["t"] == 1000.0
    assert r1["t"] == pytest.approx(1000.0 - (14.8 - 7.7))
    # run-level
    run = s.run
    assert run["run_dir"] == "runs/af50n_trunk"
    assert run["seed"] == 0
    assert run["argv"][0] == "--num-envs"
    assert run["config"]["adv_filter"] == 0.5
    assert run["num_envs"] == 32768 and run["horizon"] == 32
    assert run["batch"] == 1048576 and run["minibatch"] == 16384
    assert run["device"] == "cuda"
    assert run["adv_filter_keep"] == 524288
    assert run["env_order"] is True
    assert run["pool_segments"] == 93472
    assert run["fp16"] == "all"
    assert run["n_params"] == 1234567
    assert run["arch"]["trunk_dim"] == 256
    assert run["role"] == "trunk" and run["warmup_updates"] == 34
    assert run["lr"] == 0.0007
    assert run["status"] == "finished"
    assert run["sps_total"] == 203864
    assert run["rollout_pct"] == 63 and run["update_pct"] == 37
    assert run["t_rollout"] == 3423.0 and run["t_update"] == 2039.41
    assert run["peak_vram"] == 12.17
    assert run["checkpoints"] == ["trunk_upd00002.pt"]
    assert run["samples"] == 2097152 and run["updates"] == 2
    assert run["line_counts"]["kl"] == 2


def test_every_parsed_key_is_in_schema():
    segs = ml.parse_log(AF50N_HEAD)
    keys = set()
    for r in segs[0].updates:
        keys |= set(r)
    assert keys <= set(ml.UPDATE_FIELDS), keys - set(ml.UPDATE_FIELDS)


V11_LINES = """\
argv: --num-envs 16384 --horizon 8 --foot-contact --lr-decay cosine --seed 0 --ranked-maps --seg-len 25 --chart-pool 24576 --gears 5 --lr 0.00296092 --total-steps 3801088000 --ckpt-every 2400 --run-dir runs/v11_contact_8h
[ranked-pool] 24576 segments on cuda (C=512, 589 MB)
=== batched PPO: 16384 envs x 8 horizon = 131072 steps/update, minibatch 16384, device cuda ===
[fall] 1585 fall termination(s) this update (episodes terminated, reward 0)
update   1/29000  steps    131072  ep_return(last100)  -69.843  ent 46.292   131,073 steps/s     1.0s  r1 0.0e+00  maxlr 14.1
update   2/29000  steps    262144  ep_return(last100)      nan  ent 30.985   142,395 steps/s     1.8s  r1 0.0e+00  maxlr 108.0
Traceback (most recent call last):
  File "train_batch.py", line 1, in <module>
KeyboardInterrupt
"""


def test_v11_era_no_acc_ev_return():
    s = ml.parse_log(V11_LINES)[0]
    r1, r2 = s.updates
    assert r1["ep_return"] == -69.843 and r1["ent"] == 46.292
    assert r1["sps"] == 131073 and r1["elapsed"] == 1.0
    assert r1["maxlr"] == 14.1 and r1["falls"] == 1585
    for k in ("acc", "accL", "accR", "return", "ev", "kl_first"):
        assert k not in r1
    # nan is absent, not a string
    assert "ep_return" not in r2
    assert r2["falls"] == 0
    assert s.run["status"] == "crashed"
    assert s.run["adv_filter_keep"] if "adv_filter_keep" in s.run else True
    assert "env_order" not in s.run


V27_V30_V35_LINES = [
    # v27: ev present, no acc
    "update   1/1500  steps    262144  ep_return(last100)   -5.116  ent 46.553  ev -0.000   107,459 steps/s     2.4s  r1 2.0e-04  maxlr 0.5",
    # v30: acc present, no accL/R
    "update   2/4500  steps    524288  acc 0.6299  ep_return(last100)   -5.267  ent 46.550  ev 0.003   117,888 steps/s     4.4s  r1 1.9e-04  maxlr 0.4",
    # v35: accL/accR, no return
    "update   3/4500  steps    786432  acc 0.6548  accL 0.6445  accR 0.6646  ep_return(last100)   -6.983  ent 46.557  ev -0.001   118,689 steps/s     4.4s  r1 1.9e-04  maxlr 0.5",
    # fp16 overflow suffix
    "update 4801/9000  steps 1258553344  acc 0.9386  accL 0.9354  accR 0.9417  ep_return(last100)   93.873  ent 13.978  ev 0.995   177,590 steps/s   444.3s  r1 3.1e-03  maxlr 7.9  [fp16] 1 overflow step(s) zeroed; scale 65536",
    # fp16 scale-below-cap suffix, with return
    "update 9745/9800  steps 2554593280  acc 0.9659  accL 0.9669  accR 0.9649  return   1.404  ep_return(last100)  128.554  ent 7.235  ev 0.997   175,908 steps/s  1110.2s  r1 4.7e-05  maxlr 2.1  scale 65536",
    # non-fp16 path (no r1/maxlr)
    "update   6/10  steps     98304  acc 0.8550  accL 0.8500  accR 0.8600  return   0.100  ep_return(last100) -1699.960  ent 26.982  ev 0.092    24,307 steps/s     2.0s",
]


def test_intermediate_eras():
    recs = ml.parse_log("\n".join(V27_V30_V35_LINES))[0].updates
    assert len(recs) == 6
    assert recs[0]["ev"] == 0.0 and "acc" not in recs[0]
    assert recs[1]["acc"] == 0.6299 and "accL" not in recs[1]
    assert recs[2]["accL"] == 0.6445 and recs[2]["accR"] == 0.6646
    assert "return" not in recs[2]
    assert recs[3]["fp16_overflow"] == 1 and recs[3]["fp16_scale"] == 65536
    assert recs[3]["maxlr"] == 7.9
    assert "fp16_overflow" not in recs[4]
    assert recs[4]["fp16_scale"] == 65536 and recs[4]["return"] == 1.404
    assert recs[4]["ep_return"] == 128.554
    assert recs[5]["ep_return"] == -1699.96 and "r1" not in recs[5]
    assert recs[5]["elapsed"] == 2.0


OLD_EVENT_LINES = """\
[flail] 1.7727 rad/idle-tick (33269 note-free env-ticks)
[tilt] 29734 tilt termination(s) this update (root up-axis >30deg sagittal / >40deg lateral; flat fall_const)
[lift] 92505 foot-lift event(s) this update (each charged the flat -0.1)
[pd-net] 3 f32 pivot-floor engagement(s) this update
[envelope] 2 root-escape engagement(s) this update (episodes terminated, reward 0)
[diag] upd 1  gradcos +0.007  |g_pi| 8.41e-01  |g_v| 2.48e-02  dormant 0.000  srank 168/256  pi/v: enc 2.4e-01/1.0e-02  blk0 4.4e-01/1.6e-02
update   1/3000  steps    131072  ep_return(last100)   -5.686  ent 46.587    81,469 steps/s     1.6s  r1 0.0e+00  maxlr 0.5
"""


def test_old_event_line_variants():
    r = ml.parse_log(OLD_EVENT_LINES)[0].updates[0]
    assert r["flail"] == 1.7727 and r["flail_ticks"] == 33269
    assert "flail_L" not in r
    assert r["tilts"] == 29734
    assert r["lifts"] == 92505 and "lifts_L" not in r
    assert r["pd_net"] == 3 and r["envelope"] == 2
    assert r["diag_gradcos"] == 0.007 and "diag_psrank" not in r
    assert r["diag_groups"]["enc"] == [0.24, 0.01]


CHAIN = """\
argv: --num-envs 8 --horizon 8 --run-dir runs/v38_trunk
=== batched PPO: 8 envs x 8 horizon = 64 steps/update, minibatch 64, device cuda ===
update   1/3  steps        64  ep_return(last100)   -1.000  ent 1.000     1,000 steps/s     1.0s
update   2/3  steps       128  ep_return(last100)   -2.000  ent 1.000     1,000 steps/s     2.0s
update   3/3  steps       192  ep_return(last100)   -3.000  ent 1.000     1,000 steps/s     3.0s
[ckpt] trunk_upd00003.pt  peak-vram 11.66 GiB
THROUGHPUT: 160,470 steps/s total | rollout 0% update 100% (enqueue-attributed, NOT execution time — use --bench) | N=32768 T=8 | peak-vram 11.66 GiB
V38-TRUNK-DONE
argv: --num-envs 8 --horizon 8 --resume runs/v38_trunk/trunk_upd00002.pt --cooldown-updates 2 --run-dir runs/v38_cut2
RESUMED runs/v38_trunk/trunk_upd00002.pt at update 2/4 (step 128; lr/schedule/Adam state restored; env stream fresh)
WSD LEAF: branching from trunk update 2, cooldown 2 updates
=== batched PPO: 8 envs x 8 horizon = 64 steps/update, minibatch 64, device cuda ===
update   3/4  steps       192  ep_return(last100)   -3.500  ent 1.000     1,000 steps/s     1.0s
update   4/4  steps       256  ep_return(last100)   -4.000  ent 1.000     1,000 steps/s     2.0s
=== ScoreSaber bench: v38_cut2/leaf_from_upd00002.pt on cuda
done: 6097 played (0 skipped)
V38-CHAIN-ALL-DONE
"""


def test_chain_log_splits_per_invocation():
    segs = ml.parse_log(CHAIN, t_end=500.0)
    assert [s.run["run_dir"] for s in segs] == ["runs/v38_trunk",
                                                "runs/v38_cut2"]
    a, b = segs
    assert a.run["status"] == "finished" and a.run["sps_total"] == 160470
    assert a.run["peak_vram"] == 11.66
    assert a.run["checkpoints"] == ["trunk_upd00003.pt"]
    assert a.updates[-1]["ckpt"] == "trunk_upd00003.pt"
    assert b.run["resume_from"] == "runs/v38_trunk/trunk_upd00002.pt"
    assert b.run["resume_update"] == 2 and b.run["num_updates"] == 4
    assert b.run["role"] == "leaf" and b.run["leaf_from"] == 2
    assert b.run["status"] == "partial"
    assert [r["upd"] for r in b.updates] == [3, 4]
    # only the last segment gets wall-clock stamps
    assert "t" not in a.updates[0]
    assert b.updates[-1]["t"] == 500.0 and b.updates[0]["t"] == 499.0


def test_merge_segments_resume_discards_stale_tail():
    segs = ml.parse_log(CHAIN)
    recs, runs = ml.merge_segments(segs)
    assert [r["upd"] for r in recs] == [1, 2, 3, 4]
    # upd 3 comes from the resumed segment, not the trunk's
    assert recs[2]["ep_return"] == -3.5
    assert len(runs) == 2


def test_resumed_extension_run_level():
    txt = ("argv: --resume runs/x/trunk_final.pt --run-dir runs/x\n"
           "RESUMED runs/x/trunk_final.pt at update 24000/45000 (step 3,145,728,000; lr/schedule/Adam state restored; env stream fresh)\n"
           "WSD TRUNK EXTENDED: 24,000 -> 45,000 updates (constant lr, absolute-index ent curve — extension rewrites nothing)\n"
           "=== batched PPO: 16384 envs x 8 horizon = 131072 steps/update, minibatch 16384, device cuda ===\n"
           "update 24001/45000  steps 3145859072  ep_return(last100)      nan  ent 0.396   154,233 steps/s     0.8s  r1 0.0e+00  maxlr 3.7\n")
    s = ml.parse_log(txt)[0]
    assert s.run["resume_update"] == 24000
    assert s.run["extended_from"] == 24000 and s.run["num_updates"] == 45000
    assert s.run["role"] == "trunk"
    assert s.updates[0]["upd"] == 24001 and s.updates[0]["steps"] == 3145859072


def test_update_record_roundtrip_and_schema():
    ent_head = [1.6] * 29
    ent_head[20] = 1.5
    rec = ml.update_record(
        7, 1068, 7340032, t=1725000000.123, elapsed=51.2, rate=140000.0,
        acc=0.7, acc_l=0.71, acc_r=0.69, ret=-10.5, recent=-6.0,
        ent=46.5, ev=0.01, r1=2.5e-6, maxlr=0.6, fp16_overflow=0,
        fp16_scale=131072, kl_first=1e-9, kl_last=7e-3, cf_first=0.0,
        cf_last=0.09, gnorm=0.44, clipc=0.95, ent_head=ent_head,
        af_ratio=0.175,
        events={"falls": 3, "tilts": 0, "lifts": 10, "lifts_L": 4,
                "lifts_R": 6, "flail": 1.9, "flail_L": 0.95,
                "flail_R": 0.95, "flail_ticks": 1000, "limitrest": 16.0,
                "limitrest_legs": 7.6, "limitrest_spine": 7.1,
                "limitrest_arms": 1.4, "activation": 9.8,
                "activation_spine": 1.0, "activation_arms": 5.6,
                "activation_legs": 3.2, "torque": 390.0,
                "torque_spine": 70.0, "torque_arms": 93.0,
                "torque_legs": 227.0},
        ckpt="trunk_upd00007.pt", peak_vram=12.17)
    assert set(rec) <= set(ml.UPDATE_FIELDS)
    assert rec["upd"] == 7 and rec["steps"] == 7340032
    assert rec["ent_spine"] == pytest.approx(1.6)
    assert rec["ent_armR"] == pytest.approx(1.6)
    assert rec["ent_min"] == 1.5 and rec["ent_min_head"] == 20
    assert rec["entfull"] == pytest.approx(1.6 * 28 + 1.5)
    assert rec["tilts"] == 0 and "fp16_overflow" not in rec
    assert rec["fp16_scale"] == 131072
    line = ml.dumps_record(rec)
    assert "\n" not in line
    assert json.loads(line) == rec


def test_update_record_drops_nan_and_none():
    rec = ml.update_record(1, 10, 100, acc=float("nan"), acc_l=None,
                           ent=1.0, recent=float("nan"))
    assert "acc" not in rec and "accL" not in rec
    assert "ep_return" not in rec and rec["ent"] == 1.0
    assert json.loads(ml.dumps_record(rec)) == rec
    # a nan that sneaks in at dump time is dropped, not emitted as NaN
    rec["ev"] = math.nan
    assert "ev" not in json.loads(ml.dumps_record(rec))


def test_formatter_parser_agree_on_fields():
    """A record the trainer would write, printed the way the trainer
    prints, parses back to the same numbers (to print precision)."""
    ent_head = [1.603] * 3 + [1.605] * 14 + [1.604] * 6 + [1.606] * 6
    ent_head[20] = 1.597
    rec = ml.update_record(
        1, 1068, 1048576, elapsed=7.7, rate=135551, acc=0.6432,
        acc_l=0.6512, acc_r=0.6359, ret=-15.622, recent=-6.598,
        ent=46.560, ev=0.0, r1=2.7e-06, maxlr=0.6, kl_first=6.62e-10,
        kl_last=7.02e-03, cf_first=0.0, cf_last=0.0906, gnorm=0.441,
        clipc=0.945, ent_head=ent_head, af_ratio=0.175,
        events={"falls": 22732, "tilts": 60710, "lifts": 247239,
                "lifts_L": 125031, "lifts_R": 122208, "flail": 1.915,
                "flail_L": 0.9561, "flail_R": 0.9588,
                "flail_ticks": 193862, "limitrest": 16.1259,
                "limitrest_legs": 7.603, "limitrest_spine": 7.1072,
                "limitrest_arms": 1.4157, "activation": 9.8123,
                "activation_spine": 1.021, "activation_arms": 5.6413,
                "activation_legs": 3.15, "torque": 391.125,
                "torque_spine": 69.9, "torque_arms": 93.6,
                "torque_legs": 227.625})
    parsed = ml.parse_log(AF50N_HEAD)[0].updates[0]
    for k in ("upd", "n_upd", "steps", "elapsed", "sps", "acc", "accL",
              "accR", "return", "ep_return", "ent", "ev", "r1", "maxlr",
              "kl_first", "kl_last", "clipfrac_first", "clipfrac_last",
              "gnorm", "clipc", "ent_min", "ent_min_head", "afsd",
              "falls", "tilts", "lifts", "lifts_L", "lifts_R", "flail",
              "flail_L", "flail_R", "flail_ticks", "limitrest",
              "limitrest_legs", "limitrest_spine", "limitrest_arms",
              "activation", "activation_spine", "activation_arms",
              "activation_legs", "torque", "torque_spine",
              "torque_arms", "torque_legs"):
        assert rec[k] == pytest.approx(parsed[k]), k
    for g in ("spine", "armL", "armR", "legL", "legR"):
        assert rec["ent_" + g] == pytest.approx(parsed["ent_" + g],
                                                abs=2e-3), g
    assert rec["entfull"] == pytest.approx(parsed["entfull"], abs=5e-2)


def test_head_groups_match_trainer():
    torch = pytest.importorskip("torch")  # noqa: F841
    import train_batch
    assert tuple(train_batch.HEAD_GROUPS) == ml.HEAD_GROUPS
    eh = [0.1 * (i % 7) + 1.0 for i in range(29)]
    groups, lo, lo_v, total = train_batch.entropy_report(eh)
    g2, lo2, lo_v2, total2 = ml.entropy_groups(eh)
    assert dict(groups) == pytest.approx(g2)
    assert (lo, lo_v, total) == (lo2, lo_v2, pytest.approx(total2))


def test_load_metrics_skips_bad_lines(tmp_path):
    p = tmp_path / "metrics.jsonl"
    p.write_text('{"upd":1,"steps":10}\nnot json\n\n{"upd":2,"steps":20}\n')
    assert [r["upd"] for r in ml.load_metrics(p)] == [1, 2]


@pytest.mark.parametrize("rel", [
    "beat-arena-af50n/logs/af50n_trunk.log",
    "beat-arena/runs/v11_contact_8h/train.log",
    "beat-arena-v38/logs/v38_chain.log",
])
def test_archive_logs_if_present(rel):
    """Smoke over the real archive when it is on this machine: every
    update line becomes a record and no key escapes the schema."""
    path = os.path.expanduser("~/beat-arena-volume-backup/" + rel)
    if not os.path.exists(path):
        pytest.skip("archive not present")
    segs = ml.parse_log_file(path)
    n_lines = sum(1 for line in open(path, errors="replace")
                  if line.startswith("update ")
                  and ml._RE_UPDATE.match(line))
    assert sum(len(s.updates) for s in segs) == n_lines
    for s in segs:
        for r in s.updates:
            assert set(r) <= set(ml.UPDATE_FIELDS)
