"""--trunk-arch simba (2026-09-15, brainstorm item 2): SimBa-style
residual + LayerNorm readout trunk. Pins: exact param deltas, simba
forward shapes / finiteness / LN-scaled h, checkpoint round trip, a
2-update CPU trainer smoke, and (CUDA) fp16-vs-fp32 agreement."""
import os
import subprocess
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from policy import (N_ACTION_HEADS, N_NOTE_SLOTS, OBS_DIM, OFF_NOTE_MASK,  # noqa: E402
                    NotePolicy, load_policy)

cuda = pytest.mark.skipif(not torch.cuda.is_available(),
                          reason="needs a CUDA device")
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _obs(n, seed=0):
    g = torch.Generator().manual_seed(seed)
    obs = torch.randn(n, OBS_DIM, generator=g)
    obs[:, OFF_NOTE_MASK:OFF_NOTE_MASK + N_NOTE_SLOTS] = (
        torch.rand(n, N_NOTE_SLOTS, generator=g) < 0.6).float()
    obs[:, OFF_NOTE_MASK] = 1.0
    return obs


def _trunk_params(p):
    return sum(x.numel() for n, x in p.named_parameters()
               if n.startswith("trunk"))


def test_param_deltas_exact():
    mlp = NotePolicy(embed=192, trunk_dim=384, n_gears=5, note_geo=True)
    s384 = NotePolicy(embed=192, trunk_dim=384, n_gears=5, note_geo=True,
                      trunk_arch="simba")
    s1536 = NotePolicy(embed=192, trunk_dim=384, n_gears=5, note_geo=True,
                       trunk_arch="simba", trunk_hidden=1536)
    # 5 lead tokens since the 3-token body (2026-09-22): 960 -> 384 -> 384
    assert _trunk_params(mlp) == 516_864
    assert _trunk_params(s384) == 666_240
    assert _trunk_params(s384) - _trunk_params(mlp) == 149_376
    assert _trunk_params(s1536) - _trunk_params(mlp) == 1_035_264
    tot = lambda p: sum(x.numel() for x in p.parameters())  # noqa: E731
    # env v46 (2026-09-16) widened the encoders: saber 37 -> 44 lanes
    # (+7 x 192 x 2 tokens... one shared saber_mlp: +1,344) and body
    # 78 -> 87 (+9 x 192 = 1,728): totals +3,072 over the v45 counts
    # (2,418,578 / 2,567,954). The 3-token body (2026-09-22) replaced
    # body_mlp (53,952) with torso_mlp + leg_mlp (86,592: +32,640) and
    # widened the trunk input 576 -> 960 (+147,456): +180,096 in total.
    # The ctor's critic head is HL-Gauss 101 bins since the scalar head
    # was removed (2026-09-22): final layer 384 -> 101 instead of -> 1,
    # +38,500 over the scalar-head counts (2,601,746 / 2,751,122)
    assert tot(mlp) == 2_640_246
    assert tot(s384) == 2_789_622
    # the block's new 2-D matrices are the only additions besides LN/bias
    names = {n for n, _ in s384.named_parameters() if n.startswith("trunk")}
    assert names == {"trunk_proj.weight", "trunk_proj.bias",
                     "trunk_ln1.weight", "trunk_ln1.bias",
                     "trunk_fc1.weight", "trunk_fc1.bias",
                     "trunk_fc2.weight", "trunk_fc2.bias",
                     "trunk_ln2.weight", "trunk_ln2.bias"}


def test_simba_forward_shapes_and_ln_scale():
    torch.manual_seed(0)
    p = NotePolicy(n_gears=5, trunk_arch="simba")
    obs = _obs(16)
    p._diag_stash = True
    with torch.no_grad():
        logits, value = p.forward_stacked(obs)
    p._diag_stash = False
    assert logits.shape == (16, N_ACTION_HEADS, 5)
    assert value.shape == (16,)
    assert torch.isfinite(logits).all() and torch.isfinite(value).all()
    h = p._diag_h
    assert h.shape == (16, p.arch["trunk_dim"])
    # h is the LN2 output: per-row mean ~ 0, var ~ 1 (affine at init)
    assert h.mean(dim=1).abs().max() < 1e-4
    assert (h.var(dim=1, unbiased=False) - 1.0).abs().max() < 1e-3
    # gradients reach every trunk parameter
    logits, value = p.forward_stacked(obs)
    (logits.float().pow(2).mean() + value.pow(2).mean()).backward()
    for n, prm in p.named_parameters():
        if n.startswith("trunk"):
            assert prm.grad is not None and torch.isfinite(prm.grad).all(), n
            assert prm.grad.abs().sum() > 0, n


def test_simba_checkpoint_round_trip():
    torch.manual_seed(1)
    p = NotePolicy(n_gears=5, trunk_arch="simba", trunk_hidden=512)
    ckpt = {"arch": dict(p.arch), "state_dict": p.state_dict()}
    q = load_policy(ckpt)
    assert q.arch["trunk_arch"] == "simba" and q.arch["trunk_hidden"] == 512
    assert q.trunk_fc1.weight.shape == (512, p.arch["trunk_dim"])
    obs = _obs(6, seed=3)
    with torch.no_grad():
        a, va = p.forward_stacked(obs)
        b, vb = q.forward_stacked(obs)
    assert torch.equal(a, b) and torch.equal(va, vb)


def test_cpu_trainer_smoke_simba(tmp_path):
    """2 updates, 8 envs x T=8 on the Rust env, --trunk-arch simba: runs
    clean, reports the simba trunk and puts its matrices under Muon."""
    run = tmp_path / "simba_smoke"
    cmd = [sys.executable, os.path.join(HERE, "train_batch.py"),
           "--device", "cpu", "--env", "rust", "--num-envs", "8",
           "--horizon", "8", "--minibatches", "1", "--total-steps", "128",
           "--gears", "5", "--foot-contact", "--n-blocks", "3",
           "--no-ranked-maps", "--no-mirror-augment", "--no-ep-stats",
           "--trunk-arch", "simba", "--ckpt-every", "0",
           "--run-dir", str(run)]
    r = subprocess.run(cmd, cwd=HERE, capture_output=True, text=True,
                       timeout=900)
    out = r.stdout + r.stderr
    assert r.returncode == 0, out[-4000:]
    assert "readout trunk: simba" in out, out[-4000:]
    assert "'trunk_arch': 'simba'" in out
    assert "update   2/2" in out or "update 2/2" in out.replace("  ", " "), out[-2000:]
    m = [ln for ln in out.splitlines() if "Muon on" in ln]
    assert m, out[-2000:]
    # mlp: 18 block + 2 trunk + 6 critic = 26 hidden matrices; simba's 3
    # trunk matrices (proj, fc1, fc2) replace the mlp's 2 -> 27
    assert "Muon on 27 hidden" in m[0], m[0]


@cuda
def test_simba_fp16_matches_fp32_cuda():
    torch.manual_seed(0)
    p32 = NotePolicy(n_gears=5, trunk_arch="simba").cuda()
    p16 = NotePolicy(n_gears=5, trunk_arch="simba").cuda()
    p16.load_state_dict(p32.state_dict())
    p16.fp16 = True
    obs = _obs(256, seed=1).cuda()
    with torch.no_grad():
        l32, v32 = p32.forward_stacked(obs)
        l16, v16 = p16.forward_stacked(obs)
    # the fp16-storage class the actor tests accept (test_critic_fp16)
    scale = l32.abs().max().clamp_min(1e-3)
    assert ((l16 - l32).abs().max() / scale) < 2e-2
    vs = v32.abs().max().clamp_min(1e-3)
    assert ((v16 - v32).abs().max() / vs) < 2e-2
    # flat fp16 mode picks the new Linears up
    p16.flatten_fp16()
    with torch.no_grad():
        l16b, _ = p16.forward_stacked(obs)
    assert torch.allclose(l16b, l16, atol=1e-2, rtol=1e-2)
