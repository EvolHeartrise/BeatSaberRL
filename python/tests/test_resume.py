"""--resume / --init-from correctness (2026-09-30 review bugs 1-3):
flag inheritance (explicit vs inherited vs never-inherited vs refused),
RNG-stream round trip through a weights_only checkpoint, the shared
critic loss the [diag] probe now differentiates, the diag row index,
and a tiny Rust-env CPU trainer run exercising resume end to end
(RNG continuity, args.json preservation, the finished-trunk refusal,
--critic-lr-mult after the optimizer load, init-from value norm)."""
import json
import os
import random
import subprocess
import sys

import numpy as np
import pytest
import torch

import policy as P
import train_batch as tb

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# --- flag resolution (pure) -------------------------------------------

def test_explicit_dests_sees_typed_flags_only():
    ex = tb.explicit_dests(["--gamma", "0.9", "--no-wall-despawn",
                            "--num-env", "8", "--resume", "x.pt"])
    # abbreviation (--num-env) and the --no-X form both count as typed
    assert ex == {"gamma", "wall_despawn", "num_envs", "resume"}
    assert tb.explicit_dests([]) == set()


def _ckpt_args(**over):
    a = vars(tb.parse_args([]))
    a.update(over)
    return a


SCHED = {"lr": 7e-4, "lr_final": 1e-4, "ent_coef": 0.03, "ent_tau": 125,
         "warmup_updates": 34, "num_updates": 100, "lr_decay": "wsd"}


def _resolve(argv, leaf=False, **ck):
    args = tb.parse_args(argv)
    rep = tb.resolve_resume_args(args, tb.explicit_dests(argv),
                                 _ckpt_args(**ck), dict(SCHED), leaf)
    return args, rep


def test_untyped_flags_inherit_the_checkpoint_recipe():
    args, rep = _resolve(["--resume", "t.pt"],
                         num_envs=16384, horizon=8, gae_lambda=0.98,
                         gamma=0.99, limit_coef=0.0, tau_rand=0.0,
                         wall_despawn=False, seed=3)
    assert (args.num_envs, args.horizon, args.gae_lambda, args.gamma,
            args.limit_coef, args.tau_rand, args.wall_despawn,
            args.seed) == (16384, 8, 0.98, 0.99, 0.0, 0.0, False, 3)
    inh = {d for d, _, _ in rep["inherited"]}
    assert {"num_envs", "horizon", "gamma", "seed"} <= inh
    assert not rep["diffs"] and not rep["arch"] and not rep["ignored"]


def test_typed_flags_win_and_are_listed_when_they_differ():
    args, rep = _resolve(["--resume", "t.pt", "--no-wall-despawn",
                          "--tau-rand", "0.25", "--gamma", "0.99"],
                         wall_despawn=True, tau_rand=0.1, gamma=0.99)
    assert args.wall_despawn is False and args.tau_rand == 0.25
    diffs = {d: (m, c) for d, m, c in rep["diffs"]}
    assert diffs == {"wall_despawn": (False, True),
                     "tau_rand": (0.25, 0.1)}    # gamma typed but equal


def test_never_inherited_flags_keep_the_cli():
    args, _ = _resolve(["--resume", "t.pt"],
                       run_dir="/old/run", total_steps=5, ckpt_every=250,
                       time_cap=60.0, cooldown_updates=400,
                       ent_final=3e-6, device="cuda", env="cuda",
                       bench=True, init_from="w.pt")
    d = vars(tb.parse_args([]))
    for k in ("run_dir", "total_steps", "ckpt_every", "time_cap",
              "cooldown_updates", "ent_final", "device", "env", "bench",
              "init_from"):
        assert getattr(args, k) == d[k], k
    assert args.resume == "t.pt"


def test_schedule_state_comes_from_the_schedule():
    args, rep = _resolve(["--resume", "t.pt", "--lr", "1e-3",
                          "--ent-coef", "0.05", "--lr-final", "5e-5"])
    assert args.lr == 7e-4 and args.ent_coef == 0.03     # schedule wins
    assert args.lr_final == 1e-4                          # trunk: ignored
    ign = {d for d, _, _ in rep["ignored"]}
    assert ign == {"lr", "ent_coef", "lr_final"}
    # a LEAF's typed --lr-final is its own floor (listed as a diff)
    args, rep = _resolve(["--resume", "t.pt", "--cooldown-updates", "4",
                          "--lr-final", "5e-5"], leaf=True)
    assert args.lr_final == 5e-5
    assert ("lr_final", 5e-5, 1e-4) in rep["diffs"]


def test_typed_arch_change_is_reported_for_refusal():
    args, rep = _resolve(["--resume", "t.pt", "--embed", "256",
                          "--n-blocks", "6"], embed=192, n_blocks=6)
    assert rep["arch"] == [("embed", 256, 192)]
    # untyped arch flags just inherit (the checkpoint's arch wins anyway)
    args, rep = _resolve(["--resume", "t.pt"], trunk_hidden=1536)
    assert args.trunk_hidden == 1536 and not rep["arch"]


def test_checkpoint_args_predating_a_flag_keep_the_cli_value():
    ck = _ckpt_args()
    del ck["target_kl"]
    args = tb.parse_args(["--resume", "t.pt"])
    rep = tb.resolve_resume_args(args, {"resume"}, ck, dict(SCHED), False)
    assert args.target_kl == 1.0
    assert ("target_kl", 1.0, None) in rep["missing"]


def test_tau_center_inherits_the_resolved_body_not_the_name():
    from beat_arena_env.constants import TAU_FEAT, parse_tau_scale
    body = [1.0 + 0.01 * i for i in range(TAU_FEAT)]   # a moved default
    args, rep = _resolve(["--resume", "t.pt"], tau_center="default",
                         tau_center_body=body)
    assert parse_tau_scale(args.tau_center) == body
    assert any(d == "tau_center" for d, _, _ in rep["inherited"])
    # a typed --tau-center still wins
    args, rep = _resolve(["--resume", "t.pt", "--tau-center", "nominal"],
                         tau_center="default", tau_center_body=body)
    assert args.tau_center == "nominal"
    # an older checkpoint without the resolved body keeps the old path
    args, _ = _resolve(["--resume", "t.pt"], tau_center="default")
    assert args.tau_center == "default"


# --- RNG streams --------------------------------------------------------

def test_rng_state_round_trips_through_a_weights_only_checkpoint(tmp_path):
    torch.manual_seed(5)
    np.random.seed(5)
    random.seed(5)
    torch.rand(3), np.random.rand(3), np.random.randn()   # stir the state
    path = tmp_path / "rng.pt"
    torch.save({"rng": tb.rng_state()}, path)
    want = (torch.rand(4), np.random.rand(4), np.random.randn(),
            random.random())
    torch.manual_seed(0)
    np.random.seed(0)
    random.seed(0)
    got_st = torch.load(path, weights_only=True)["rng"]
    assert set(tb.restore_rng(got_st)) >= {"torch", "numpy", "python"}
    got = (torch.rand(4), np.random.rand(4), np.random.randn(),
           random.random())
    assert torch.equal(want[0], got[0])
    assert np.array_equal(want[1], got[1])
    assert want[2] == got[2] and want[3] == got[3]


def test_restore_rng_env_gen():
    class _Env:
        _gen = torch.Generator().manual_seed(1)
    env = _Env()
    torch.rand(2, generator=env._gen)
    st = tb.rng_state(env)
    want = torch.rand(3, generator=env._gen)
    env._gen.manual_seed(99)
    assert tb.restore_rng(st, env, what=("env_gen",)) == ["env_gen"]
    assert torch.equal(torch.rand(3, generator=env._gen), want)


# --- the [diag] critic loss and rows -------------------------------------

def _tiny_policy():
    torch.manual_seed(0)
    return P.NotePolicy(embed=32, trunk_dim=48, n_blocks=1, n_heads=2,
                        n_gears=5, ffn_mult=2, note_geo=True,
                        value_bins=51, value_vmax=6.0)


def test_value_loss_is_the_training_hl_loss_on_normalized_returns():
    pol = _tiny_policy()
    obs = torch.randn(16, P.OBS_DIM) * 0.5
    ret = torch.randn(16) * 3.0 + 1.0
    mu, sig = torch.tensor(0.7), torch.tensor(2.5)
    pol.get_action_and_value(obs)
    got = tb.value_loss(pol, ret, mu, sig)
    # the minibatch loop's expression before the refactor, verbatim
    want = P.hl_loss(pol._value_logits, (ret - mu) / sig,
                     pol.value_centers, pol.value_sigma)
    assert torch.equal(got, want)
    # ...and NOT the old diag proxy 0.5*MSE on the HL expectation
    v = torch.softmax(pol._value_logits, -1) @ pol.value_centers
    old = 0.5 * ((v - (ret - mu) / sig) ** 2).mean()
    assert not torch.allclose(got, old)
    # its gradient is the critic gradient training applies (x vf_coef)
    got.backward()
    g = [p.grad.clone() for p in pol.parameters() if p.grad is not None]
    assert g and any(x.abs().sum() > 0 for x in g)


def test_diag_index_spans_all_ticks_and_leaves_the_global_rng_alone():
    T, N, dn = 32, 4096, 2048
    torch.manual_seed(11)
    before = torch.get_rng_state()
    idx = tb.diag_index(T * N, dn)
    assert torch.equal(torch.get_rng_state(), before)
    assert torch.equal(idx, tb.diag_index(T * N, dn))        # fixed
    assert idx.numel() == dn and idx.unique().numel() == dn
    assert int(idx.max()) < T * N
    ticks = (idx // N).unique()
    assert ticks.numel() == T                 # not just t = 0
    assert (idx % N).unique().numel() > dn // 2
    assert tb.diag_index(64, 2048).numel() == 64


# --- end to end: tiny Rust-env CPU trainer ------------------------------

COMMON = ["--device", "cpu", "--env", "rust", "--num-envs", "8",
          "--horizon", "8", "--minibatches", "1", "--n-blocks", "1",
          "--no-ranked-maps", "--no-mirror-augment", "--no-ep-stats",
          "--tau-rand", "0"]


def _run(*argv, ok=True):
    r = subprocess.run([sys.executable, os.path.join(HERE, "train_batch.py"),
                        *argv], cwd=HERE, capture_output=True, text=True,
                       timeout=600)
    out = r.stdout + r.stderr
    if ok:
        assert r.returncode == 0, out[-4000:]
    return r.returncode, out


@pytest.fixture(scope="module")
def trunk(tmp_path_factory):
    """3 updates at gamma 0.9 (off-default, to see it inherited), a
    numbered checkpoint every update."""
    d = tmp_path_factory.mktemp("trunk")
    _run(*COMMON, "--total-steps", "192", "--ckpt-every", "1",
         "--gamma", "0.9", "--run-dir", str(d))
    return d


def _load(p):
    return torch.load(p, map_location="cpu", weights_only=True)


def test_checkpoint_carries_rng_and_args(trunk):
    ck = _load(trunk / "trunk_upd00002.pt")
    assert {"torch", "numpy", "python"} <= set(ck["rng"])
    assert ck["args"]["gamma"] == 0.9
    assert "diag_probe_obs" in ck        # None on CPU (the diag is CUDA)
    assert "diag_probe_upd" in ck        # capture update (None until then)


def test_resume_continues_the_rng_stream_and_inherits(trunk, tmp_path):
    """Resume at update 2 -> run update 3: the sampling/shuffle draws
    are data-independent in count, so the torch RNG after the resumed
    update 3 must equal the uninterrupted run's after update 3 (before
    the fix both runs reseeded to --seed and this was update 1's)."""
    run = tmp_path / "cont"
    _, out = _run("--device", "cpu", "--resume",
                  str(trunk / "trunk_upd00002.pt"), "--total-steps", "192",
                  "--clip", "0.2", "--critic-lr-mult", "2",
                  "--run-dir", str(run))
    assert "RNG streams restored" in out
    assert "--gamma 0.9 (default 0.965)" in out            # inherited
    assert "--clip 0.2   (checkpoint 0.3)" in out           # diff list
    assert "critic lr multiplier 3 -> 2" in out
    assert "update   3/3" in out
    a = _load(run / "trunk_final.pt")
    b = _load(trunk / "trunk_upd00003.pt")
    assert torch.equal(a["rng"]["torch"], b["rng"]["torch"])
    assert torch.equal(a["rng"]["numpy"]["key"], b["rng"]["numpy"]["key"])
    assert a["args"]["gamma"] == 0.9 and a["args"]["clip"] == 0.2
    assert a["args"]["critic_lr_mult"] == 2.0
    # the Adam critic group carries the new multiplier
    assert a["optimizer"]["param_groups"][1]["lr_mult"] == 2.0


def test_resume_in_place_keeps_the_original_args_json(trunk, tmp_path):
    import shutil
    d = tmp_path / "inplace"
    shutil.copytree(trunk, d)
    orig = (d / "args.json").read_text()
    _run("--device", "cpu", "--resume", str(d / "trunk_upd00002.pt"),
         "--total-steps", "192")
    assert (d / "args.json").read_text() == orig
    rec = json.loads((d / "args_resume_00002.json").read_text())
    assert rec["inherited"]["gamma"] == [0.965, 0.9]
    assert rec["resume_from"].endswith("trunk_upd00002.pt")
    _run("--device", "cpu", "--resume", str(d / "trunk_upd00002.pt"),
         "--total-steps", "192")
    assert (d / "args_resume_00002_2.json").exists()


def test_finished_trunk_without_extension_is_refused(trunk, tmp_path):
    rc, out = _run("--device", "cpu", "--resume",
                   str(trunk / "trunk_upd00003.pt"), "--run-dir",
                   str(tmp_path / "x"), ok=False)
    assert rc != 0 and "nothing to run" in out, out[-2000:]
    assert "Traceback" not in out


def test_typed_arch_flag_is_refused_on_resume(trunk, tmp_path):
    rc, out = _run("--device", "cpu", "--resume",
                   str(trunk / "trunk_upd00002.pt"), "--n-blocks", "3",
                   "--total-steps", "192", "--run-dir",
                   str(tmp_path / "x"), ok=False)
    assert rc != 0 and "--ignore-arch-flags" in out, out[-2000:]


def test_leaf_resume_keeps_its_own_departures(trunk, tmp_path):
    run = tmp_path / "leaf"
    _, out = _run("--device", "cpu", "--resume",
                  str(trunk / "trunk_upd00002.pt"), "--cooldown-updates",
                  "1", "--ent-final", "1e-4", "--no-wall-despawn",
                  "--run-dir", str(run))
    assert "WSD LEAF" in out and "update   3/3" in out
    leaf = _load(run / "leaf_from_upd00002.pt")
    assert leaf["args"]["wall_despawn"] is False
    assert leaf["args"]["gamma"] == 0.9 and leaf["args"]["horizon"] == 8


def test_init_from_loads_the_value_normalizer(trunk, tmp_path):
    src = _load(trunk / "trunk_final.pt")["value_norm"]
    run = tmp_path / "warm"
    _, out = _run(*COMMON, "--total-steps", "64", "--init-from",
                  str(trunk / "trunk_final.pt"), "--run-dir", str(run))
    assert "init-from: value normalizer loaded" in out, out[-3000:]
    # one EMA step (beta 0.01) from the source stats, not a re-init
    vn = _load(run / "trunk_final.pt")["value_norm"]
    assert abs(vn["mu"] - src["mu"]) <= 0.02 * max(1.0, abs(src["mu"])) + 0.2
