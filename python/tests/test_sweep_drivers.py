"""Sweep drivers vs the CURRENT trainer/bench CLIs (2026-09-30).

All three Protein drivers once died on argparse in seconds (removed
--lr-decay / --fused-forward) and logged each leg as a failure at cost
1.0 s. Pinned here, CPU-only (no CUDA, no gpytorch — the drivers import
Protein inside main()):

1. Every command a driver builds for a sample config (trunk, leaf,
   bench) parses against the real train_batch / bench_scoresaber
   parse_args(), and the parsed values carry the driver's intent.
2. The checkpoint names the drivers wait for are the ones the trainer
   writes, and the metric field sweep_contact reads is the one the
   trainer's metrics.jsonl record carries.
3. Failures are honest: a sub-FAST_FAIL_S crash aborts the sweep, a
   slower one returns its real wall; replay refuses pre-port lines.
"""
import inspect
import json
import os
import sys

import pytest

import bench_scoresaber
import metrics_log
import sweep_contact
import sweep_protein
import sweep_ranked
import train_batch

DRIVERS = [sweep_protein, sweep_ranked, sweep_contact]
SAMPLE = {"lr": 1.3e-3, "ent_coef": 0.02, "ent_final": 3e-4,
          "gamma": 0.965, "gae_lambda": 0.95, "bomb_prob": 0.05,
          "total_steps": 3e8}


def _parse(parse_args, cmd, script, monkeypatch):
    assert cmd[0] == sys.executable and cmd[1] == script, cmd[:2]
    monkeypatch.setattr(sys, "argv", [script] + cmd[2:])
    return parse_args()


@pytest.mark.parametrize("drv", DRIVERS, ids=lambda m: m.__name__)
def test_train_cmds_parse(drv, monkeypatch, tmp_path):
    run_dir = str(tmp_path / "run_000")
    trunk, leaf, leaf_dir = drv.train_cmds(dict(SAMPLE), run_dir)
    ta = _parse(train_batch.parse_args, trunk, "train_batch.py", monkeypatch)
    la = _parse(train_batch.parse_args, leaf, "train_batch.py", monkeypatch)
    trunk_steps, cd = sweep_protein.split_budget(
        SAMPLE["total_steps"], drv.NUM_ENVS, drv.HORIZON)
    batch = drv.NUM_ENVS * drv.HORIZON
    # the swept dims land where the driver means them
    assert ta.lr == pytest.approx(SAMPLE["lr"])
    assert ta.lr_final == pytest.approx(sweep_protein.LR_FINAL)
    assert ta.ent_coef == pytest.approx(SAMPLE["ent_coef"])
    for a in (ta, la):
        assert a.ent_final == pytest.approx(SAMPLE["ent_final"])
        assert a.gamma == pytest.approx(SAMPLE["gamma"])
        assert a.gae_lambda == pytest.approx(SAMPLE["gae_lambda"])
        assert (a.num_envs, a.horizon) == (drv.NUM_ENVS, drv.HORIZON)
        assert a.seed == 0 and a.foot_contact
        # main()'s post-parse shape checks
        mb = batch // a.minibatches
        assert mb > 0 and mb % a.mb_accum == 0
    # WSD: fresh trunk, then a leaf branched off its final checkpoint
    assert ta.resume is None and ta.cooldown_updates == 0
    assert ta.total_steps == trunk_steps and ta.run_dir == run_dir
    assert la.resume == os.path.join(run_dir, "trunk_final.pt")
    assert la.cooldown_updates == cd > 0
    assert la.run_dir == leaf_dir and not la.wall_despawn
    assert trunk_steps // batch + cd == int(SAMPLE["total_steps"]) // batch
    if drv is sweep_protein:   # procedural: the bomb dial must be live
        assert not ta.ranked_maps and not la.ranked_maps
        assert not ta.ep_stats and ta.chart_pool == 65536
        assert ta.bomb_prob == la.bomb_prob == pytest.approx(
            SAMPLE["bomb_prob"])
    else:
        assert ta.ranked_maps and la.ranked_maps
        assert ta.chart_pool == la.chart_pool == drv.SEG_POOL


@pytest.mark.parametrize("drv", [sweep_protein, sweep_ranked],
                         ids=lambda m: m.__name__)
def test_bench_cmd_parses(drv, monkeypatch):
    cmd = drv.bench_cmd("runs/x/leaf/leaf_from_upd00100.pt", "b.json")
    a = _parse(bench_scoresaber.parse_args, cmd, "bench_scoresaber.py",
               monkeypatch)
    assert a.ckpt.endswith("leaf_from_upd00100.pt") and a.out == "b.json"
    assert a.num_envs == 2048


def test_checkpoint_names_match_trainer():
    src = inspect.getsource(train_batch.main)
    assert '"trunk_final.pt"' in src
    assert 'f"leaf_from_upd{leaf_from:05d}.pt"' in src


def test_contact_metric_field(tmp_path):
    # the trainer passes ret= to write_metrics; the record calls it
    # "return" — the field final_return reads
    assert "ret=ret_v" in inspect.getsource(train_batch.main)
    trunk, leaf = tmp_path / "t.jsonl", tmp_path / "l.jsonl"
    with open(trunk, "w") as f:
        for u in range(1, 181):
            f.write(metrics_log.dumps_record(metrics_log.update_record(
                u, 200, u * 10, ret=-1.0)) + "\n")
    with open(leaf, "w") as f:
        for u in range(181, 201):
            f.write(metrics_log.dumps_record(metrics_log.update_record(
                u, 200, u * 10, ret=2.0)) + "\n")
        f.write('{"upd": 201, "ret')   # torn line from a killed run
    # 200 records -> tail max(20, 20) = the 20 leaf records
    assert sweep_contact.final_return([str(trunk), str(leaf)]) == 2.0
    assert sweep_contact.final_return([str(tmp_path / "none")]) is None


def test_fast_failure_aborts(tmp_path):
    cmd = [sys.executable, "-c", "import sys; print('usage: boom'); "
           "sys.exit(2)"]
    with pytest.raises(SystemExit) as e:
        sweep_protein.run_step(0, "TRUNK", cmd, str(tmp_path / "l.log"), 60)
    assert "ABORTED" in str(e.value) and "usage: boom" in str(e.value)


def test_slow_failure_costs_real_wall(tmp_path, monkeypatch):
    monkeypatch.setattr(sweep_protein, "FAST_FAIL_S", 0.0)
    cmd = [sys.executable, "-c",
           "import sys, time; time.sleep(0.3); sys.exit(1)"]
    ok, wall = sweep_protein.run_step(0, "TRUNK", cmd,
                                      str(tmp_path / "l.log"), 60)
    assert not ok and wall >= 0.3


def test_replay_refuses_pre_port_lines(tmp_path):
    class P:
        suggestion_idx = 0
        def observe(self, *a):
            pass
    log = tmp_path / "observations.jsonl"
    sweep_protein.log_observation(str(log), 0, {"train": {}}, 1.0, 5.0,
                                  False)
    assert sweep_protein.replay(P(), str(log)) == 1
    with open(log, "a") as f:
        f.write(json.dumps({"idx": 1, "hypers": {"train": {}},
                            "score": 0.0, "cost": 1.0,
                            "is_failure": True}) + "\n")
    with pytest.raises(SystemExit):
        sweep_protein.replay(P(), str(log))
