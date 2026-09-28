"""fp16-accumulate GEMM modes (issue #3, fp16_linear.fp16_accum /
NotePolicy.fp16_accum / train_batch --fp16-accum).

Pins: the modes really flip the torch forward (fwd, all) and dgrad (all)
GEMMs and nothing else — the wgrad / bias-grad reductions and the Lt
ReLU-epilogue GEMMs stay bitwise the fp32-accumulate result in every
mode; the torch flag never leaks past a call; a Function created under a
mode runs its backward under that mode even when backward() is called
outside the context (the trainer's pattern); fp16-accumulate error vs an
fp32 reference stays in the fp16 class; the rollout (grad-free) and
update (grad-mode, half obs) forwards stay BITWISE identical at the same
row count in every mode (the r1 contract); and the mode is an arch
field that survives load_policy (absent = "off")."""

import pytest
import torch
import torch.nn as nn

if not torch.cuda.is_available():
    pytest.skip("fp16 accumulate paths need CUDA", allow_module_level=True)

import fp16_linear as fl

if fl._load() is None:
    pytest.skip("fp16_gemm extension unavailable", allow_module_level=True)


def _flag():
    return torch._C._get_cublas_allow_fp16_accumulation()


def _rel(a, b):
    return ((a.float() - b.float()).abs().max()
            / b.float().abs().max().clamp_min(1e-6)).item()


def _lin(K, N, seed):
    torch.manual_seed(seed)
    m = nn.Linear(K, N).cuda()
    with torch.no_grad():
        m.bias.normal_(0, 0.1)
    return m


def _x(R, K, seed):
    g = torch.Generator(device="cuda").manual_seed(seed)
    return torch.randn(R, K, device="cuda", generator=g).half()


def _run_linear(mode, m, x, gy, outside=False):
    """fp16_linear fwd+bwd under `mode`; outside=True calls backward()
    after leaving the context (as the trainer does)."""
    fl.bump_weights_version()
    for p in m.parameters():
        p.grad = None
    xg = x.clone().requires_grad_(True)
    with fl.fp16_accum(mode):
        y = fl.fp16_linear(xg, m.weight, m.bias)
        if not outside:
            y.backward(gy)
    if outside:
        y.backward(gy)
    return y.detach(), xg.grad, m.weight.grad.clone(), m.bias.grad.clone()


@pytest.mark.parametrize("K,N", [(192, 576), (768, 192)])
def test_modes_flip_fwd_and_dgrad_only(K, N):
    m = _lin(K, N, 0)
    R = 29 * 2048
    x = _x(R, K, 1)
    gy = (_x(R, N, 2).float() * 0.05).half()
    y0, dx0, dw0, db0 = _run_linear("off", m, x, gy)
    y1, dx1, dw1, db1 = _run_linear("fwd", m, x, gy)
    y2, dx2, dw2, db2 = _run_linear("all", m, x, gy)
    # forward flips in fwd and all (identically), dgrad only in all
    assert not torch.equal(y0, y1)
    assert torch.equal(y1, y2)
    assert torch.equal(dx0, dx1)
    assert not torch.equal(dx0, dx2)
    # the weight / bias grads (reductions over R rows) never flip
    for dw, db in ((dw1, db1), (dw2, db2)):
        assert torch.equal(dw, dw0)
        assert torch.equal(db, db0)
    # fp16-accumulate stays in the fp16 error class vs fp32
    ref_y = x.float() @ m.weight.detach().t() + m.bias.detach()
    ref_dx = gy.float() @ m.weight.detach()
    assert _rel(y0, ref_y) < 2e-3
    assert _rel(y2, ref_y) < 1e-2
    assert _rel(dx2, ref_dx) < 1e-2
    assert not _flag()


def test_backward_outside_context_keeps_the_forward_mode():
    m = _lin(192, 192, 3)
    x = _x(29 * 1024, 192, 4)
    gy = (_x(29 * 1024, 192, 5).float() * 0.05).half()
    _, dx_in, _, _ = _run_linear("all", m, x, gy)
    _, dx_out, _, _ = _run_linear("all", m, x, gy, outside=True)
    assert torch.equal(dx_in, dx_out)
    _, dx_off, _, _ = _run_linear("off", m, x, gy)
    assert not torch.equal(dx_out, dx_off)


def test_flag_never_leaks():
    m = _lin(192, 384, 6)
    x = _x(4096, 192, 7)
    assert not _flag()
    with fl.fp16_accum("all"):
        with torch.no_grad():
            fl.fp16_linear(x, m.weight, m.bias)
        assert not _flag()          # set per call, not per context
    assert not _flag()
    with pytest.raises(ValueError):
        fl.fp16_accum("on")


def test_lt_epilogue_and_wgrad_gemms_stay_fp32():
    """The ReLU-epilogue GEMMs (sm_120 has no fast COMPUTE_16F aux
    kernel) and the wgrad GEMMs are explicit cuBLAS(Lt) calls with
    COMPUTE_32F: the torch flag must not touch them."""
    ext = fl._load()
    R = 29 * 1024
    x = _x(R, 192, 8)
    w = (_x(768, 192, 9).float() * 0.07).half()
    b = (_x(1, 768, 10).float() * 0.1).half().view(-1)
    g = (_x(R, 768, 11).float() * 0.05).half()
    base = [ext.linear_relu_aux(x, w, b)[0], ext.wgrad(g, x)]
    torch._C._set_cublas_allow_fp16_accumulation(True)
    try:
        flipped = [ext.linear_relu_aux(x, w, b)[0], ext.wgrad(g, x)]
    finally:
        torch._C._set_cublas_allow_fp16_accumulation(False)
    for a, c in zip(base, flipped):
        assert torch.equal(a, c)


def _policy(mode):
    from policy import NotePolicy
    torch.manual_seed(0)
    p = NotePolicy(embed=192, trunk_dim=384, n_blocks=2, n_heads=4,
                   n_gears=5, note_geo=True, fp16_accum=mode).cuda()
    p.fp16 = True
    p.flatten_fp16()
    return p


@pytest.mark.parametrize("mode", ["off", "fwd", "all"])
def test_rollout_and_update_forwards_agree_bitwise(mode):
    """Same rows: the rollout's grad-free forward on fp32 obs and the
    update's grad-mode forward on the half copy must be bit-identical
    under every mode (the kernels r1 rides on)."""
    from policy import OBS_DIM
    p = _policy(mode)
    obs = torch.randn(2048, OBS_DIM, device="cuda") * 0.7
    with torch.no_grad():
        l_roll, v_roll = p.forward_stacked(obs)
    l_upd, v_upd = p.forward_stacked(obs.half())
    assert torch.equal(l_roll, l_upd.detach())
    assert torch.equal(v_roll, v_upd.detach())
    assert not _flag()


def test_policy_mode_changes_the_fp16_forward_only():
    from policy import OBS_DIM
    obs = torch.randn(1024, OBS_DIM, device="cuda") * 0.7
    out = {}
    for mode in ("off", "fwd"):
        p = _policy(mode)
        with torch.no_grad():
            out[mode] = p.forward_stacked(obs)[0]
        p.fp16 = False              # fp32 record path ignores the mode
        with torch.no_grad():
            out[mode + "32"] = p.forward_stacked(obs)[0]
    assert not torch.equal(out["off"], out["fwd"])
    assert torch.equal(out["off32"], out["fwd32"])
    # error vs the fp32 record path: the fp16 class, at most a few x the
    # fp32-accumulate fp16 stream's (random obs on a random net put the
    # max-abs metric at the mercy of the note-geo lanes; the mean is the
    # stable one — tools/fp16_accum_probe.py error has the real-obs,
    # trained-checkpoint numbers)
    e_off = (out["off"] - out["off32"]).abs().mean().item()
    e_fwd = (out["fwd"] - out["off32"]).abs().mean().item()
    assert e_fwd < 4 * e_off + 1e-4, (e_off, e_fwd)


def test_arch_field_round_trips():
    from policy import NotePolicy, load_policy
    p = NotePolicy(embed=64, trunk_dim=128, n_blocks=1, n_gears=5,
                   fp16_accum="all")
    assert p.arch["fp16_accum"] == "all" and p.fp16_accum == "all"
    q = load_policy({"arch": dict(p.arch), "state_dict": p.state_dict()})
    assert q.fp16_accum == "all"
    arch = dict(p.arch)
    arch.pop("fp16_accum")               # every pre-#3 checkpoint
    assert load_policy({"arch": arch,
                        "state_dict": p.state_dict()}).fp16_accum == "off"
    with pytest.raises(ValueError):
        NotePolicy(embed=64, trunk_dim=128, n_blocks=1, fp16_accum="yes")
