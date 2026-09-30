"""Fused block FFN (csrc/fused_ffn.cu, fp16_linear.fused_ffn, NotePolicy.
fused_ffn / train_batch --fused-ffn, 2026-09-30, review item I13): one
kernel runs x + relu(h2 W0^T + b0) W2^T + b2 and the next block's
LayerNorm, the 768-wide hidden kept on-chip.

CPU pins (run everywhere): the mode is an arch field that round-trips
through load_policy (absent = "off"), bad modes raise, and off the fp16
CUDA stream the flag changes nothing (the kernel never applies).

CUDA pins (the pod): kernel outputs and every gradient sit as close to an
fp32 reference as the unfused path's (fp16 last-rounding class — the
tolerances below compare against the unfused path's OWN error, so they
scale with the accumulate mode); the grad-free call is bitwise the grad
call's forward and a row's result does not depend on the batch it rides
in or its position in a 128-row tile (odd row counts, tail tiles) — the
rollout-vs-update r1 contract; the saved ReLU mask is cuBLASLt's
RELU_AUX layout (so the backward's DRELU dgrad reads it) and the saved
hidden is exactly the fp16 GEMM2 operand; at the policy level the
rollout (grad-free, fp32 obs) and update (grad mode, half obs) forwards
agree bitwise with the flag on, under every fp16 accumulate mode, and
the whole-policy logits / value / grads stay in the unfused path's
error class vs the fp32 record path."""

import pytest
import torch
import torch.nn as nn

import fp16_linear as fl
from policy import NotePolicy, OBS_DIM, load_policy

D, HID = 192, 768


# ---- CPU ------------------------------------------------------------------

def test_arch_field_round_trips():
    p = NotePolicy(embed=64, trunk_dim=128, n_blocks=1, n_gears=5,
                   fused_ffn="on")
    assert p.arch["fused_ffn"] == "on" and p.fused_ffn == "on"
    q = load_policy({"arch": dict(p.arch), "state_dict": p.state_dict()})
    assert q.fused_ffn == "on"
    arch = dict(p.arch)
    arch.pop("fused_ffn")                # every checkpoint before I13
    assert load_policy({"arch": arch,
                        "state_dict": p.state_dict()}).fused_ffn == "off"
    with pytest.raises(ValueError):
        NotePolicy(embed=64, trunk_dim=128, n_blocks=1, fused_ffn="yes")
    with pytest.raises(ValueError):
        fl.ffn_fusion("yes")


def test_mode_is_inert_off_the_fp16_cuda_stream():
    """CPU / fp32 record path: fused_ffn never applies — bitwise the
    unfused forward."""
    torch.manual_seed(0)
    p = NotePolicy(embed=192, trunk_dim=384, n_blocks=2, n_heads=4,
                   n_gears=5, note_geo=True)
    obs = torch.randn(8, OBS_DIM) * 0.7
    with torch.no_grad():
        a = p.forward_stacked(obs)
        p.fused_ffn = "on"
        b = p.forward_stacked(obs)
    assert torch.equal(a[0], b[0]) and torch.equal(a[1], b[1])
    blk = p.blocks[0]
    with fl.ffn_fusion("on"):
        assert not fl.fused_ffn_ok(torch.zeros(4, D), blk.ffn, blk.ln_attn)
        assert not fl.fused_ffn_ok(torch.zeros(4, D, dtype=torch.float16),
                                   blk.ffn, blk.ln_attn)
    assert fl._FFN_MODE == "off"         # the context never leaks


class _FakeExt:
    """torch stand-in for the fp16_gemm / layer_norm_cuda entry points the
    fused path calls (CPU, fp32 math, fp16 storage) — pins the Python
    autograd wiring (saved tensors, needs_input_grad slots, grad routing,
    the ReLU-mask layout the backward unpacks) without a GPU."""

    @staticmethod
    def ffn_fused(x, h2, w0, b0, w2, b2, lnw, lnb, eps, acc, save):
        pre = h2.float() @ w0.float().t() + b0
        hid = torch.relu(pre).half()
        out = (x.float() + (hid.float() @ w2.float().t() + b2)).half()
        ho = mean = rstd = None
        if lnw is not None:
            xf = out.float()
            mean = xf.mean(1)
            rstd = (xf.var(1, unbiased=False) + eps).rsqrt()
            ho = ((xf - mean[:, None]) * rstd[:, None] * lnw + lnb).half()
        if not save:
            return [out, ho, None, None, None, None]
        bits = (pre > 0).to(torch.uint8).view(pre.shape[0], -1, 8)
        mask = (bits << torch.arange(8, dtype=torch.uint8)).sum(
            -1, dtype=torch.uint8)
        return [out, ho, mean if lnw is not None else None,
                rstd if lnw is not None else None, hid, mask]

    @staticmethod
    def dgrad_drelu(g, w, mask):
        R = g.shape[0]
        bits = ((mask.view(R, -1, 1) >> torch.arange(8, dtype=torch.uint8))
                & 1).view(R, -1)[:, :w.shape[1]]
        return ((g.float() @ w.float()) * bits).half()

    @staticmethod
    def wgrad(dy, x):
        return dy.float().t() @ x.float()

    @staticmethod
    def ln_bwd(x, dy, w, mean, rstd, gx, want_dbb):
        xh = (x.float() - mean[:, None]) * rstd[:, None]
        gw = dy.float() * w
        dx = rstd[:, None] * (gw - gw.mean(1, keepdim=True)
                              - xh * (gw * xh).mean(1, keepdim=True))
        dx = (dx + gx.float()).half()
        out = [dx, (dy.float() * xh).sum(0), dy.float().sum(0)]
        if want_dbb:
            out.append(dx.float().sum(0))
        return out


@pytest.mark.parametrize("with_ln", [True, False], ids=["ln", "res"])
def test_autograd_wiring_on_a_fake_extension(with_ln, monkeypatch):
    """fused_ffn's Function vs autograd through the same math: every
    input's gradient lands in its own slot (x, h2, W0, b0, W2, b2, LN w/b),
    the grad-free call equals the grad call's forward, and the ReLU mask
    round-trips through the Lt bit layout."""
    import fused_ln
    from fused_ln import FusedLayerNorm
    fake = _FakeExt()
    monkeypatch.setattr(fl, "_load", lambda: fake)
    monkeypatch.setattr(fused_ln, "_load", lambda: fake)
    torch.manual_seed(0)
    ffn = nn.Sequential(nn.Linear(D, HID), nn.ReLU(), nn.Linear(HID, D))
    ln = FusedLayerNorm(D) if with_ln else None
    with torch.no_grad():
        ffn[0].bias.normal_(0, 0.1)
        ffn[2].bias.normal_(0, 0.1)
        if ln is not None:
            ln.weight.normal_(1.0, 0.1)
            ln.bias.normal_(0, 0.1)
    fl.bump_weights_version()
    B, T = 3, 7                          # (B, T, d) activations, 21 rows
    x0 = (torch.randn(B, T, D) * 0.7).half()
    h0 = torch.randn(B, T, D).half()
    params = list(ffn.parameters()) + (list(ln.parameters()) if ln else [])
    x = x0.clone().requires_grad_(True)
    h2 = h0.clone().requires_grad_(True)
    out = fl.fused_ffn(x, h2, ffn, ln)
    outs = out if with_ln else (out,)
    assert all(o.shape == (B, T, D) for o in outs)
    with torch.no_grad():
        nog = fl.fused_ffn(x0, h0, ffn, ln)
    for a, b in zip(outs, nog if with_ln else (nog,)):
        assert torch.equal(a.detach(), b)
    gs = [(torch.randn(B, T, D) * 0.1).half() for _ in outs]
    torch.autograd.backward(list(outs), gs)
    got = [x.grad, h2.grad] + [p_.grad for p_ in params]
    # reference: plain autograd through the fake's math
    rp = [p_.detach().clone().requires_grad_(True) for p_ in params]
    xr = x0.float().requires_grad_(True)
    hr = h0.float().requires_grad_(True)
    pre = hr @ rp[0].half().float().t() + rp[1]
    o = xr + torch.relu(pre) @ rp[2].half().float().t() + rp[3]
    routs = [o]
    if with_ln:
        routs.append(torch.nn.functional.layer_norm(o, (D,), rp[4], rp[5],
                                                    ln.eps))
    torch.autograd.backward(routs, [g_.float() for g_ in gs])
    ref = [xr.grad, hr.grad] + [p_.grad for p_ in rp]
    for n, a, r in zip(["x", "h2", "w0", "b0", "w2", "b2", "lnw", "lnb"],
                       got, ref):
        assert a is not None and a.shape == r.shape, n
        assert _fro_cpu(a, r) < 2e-2, (n, _fro_cpu(a, r))


def _fro_cpu(a, b):
    return ((a.float() - b.float()).norm() / b.float().norm()).item()


# ---- CUDA -----------------------------------------------------------------

cuda = pytest.mark.skipif(not torch.cuda.is_available(),
                          reason="fused FFN kernel needs CUDA")


def _need_kernel():
    ext = fl._load()
    if ext is None or not hasattr(ext, "ffn_fused"):
        pytest.skip("fp16_gemm extension lacks ffn_fused")
    if not ext.ffn_fused_supported():
        pytest.skip("device lacks the kernel's shared memory")
    return ext


def _mods(seed):
    """Block FFN [Linear, ReLU, Linear] + a LayerNorm with non-trivial
    affine (the next block's ln_attn)."""
    from fused_ln import FusedLayerNorm
    torch.manual_seed(seed)
    ffn = nn.Sequential(nn.Linear(D, HID), nn.ReLU(), nn.Linear(HID, D))
    ln = FusedLayerNorm(D)
    with torch.no_grad():
        ffn[0].bias.normal_(0, 0.1)
        ffn[2].bias.normal_(0, 0.1)
        ln.weight.normal_(1.0, 0.1)
        ln.bias.normal_(0, 0.1)
    return ffn.cuda(), ln.cuda()


def _act(R, seed, scale=1.0):
    g = torch.Generator(device="cuda").manual_seed(seed)
    return (torch.randn(R, D, device="cuda", generator=g) * scale).half()


def _rel(a, b):
    return ((a.float() - b.float()).abs().max()
            / b.float().abs().max().clamp_min(1e-6)).item()


def _fro(a, b):
    """Relative Frobenius error — robust to the rare ReLU mask flips two
    implementations disagree on (a pre-activation within rounding noise of
    0 carries a full gradient element; see test_relu_chain._fro)."""
    return ((a.float() - b.float()).norm() / b.float().norm()).item()


def _unpack(mask):
    R = mask.shape[0]
    shifts = torch.arange(8, device=mask.device, dtype=torch.uint8)
    return ((mask.view(R, -1, 1) >> shifts) & 1).view(R, -1)[:, :HID]


def _fused(x, h2, ffn, ln, mode, acc_mode, grad):
    """fl.fused_ffn under (ffn mode, fp16 accumulate mode)."""
    with fl.fp16_accum(acc_mode), fl.ffn_fusion(mode):
        assert fl.fused_ffn_ok(h2, ffn, ln)
        with torch.set_grad_enabled(grad):
            return fl.fused_ffn(x, h2, ffn, ln)


def _unfused(x, h2, ffn, ln, acc_mode):
    """The production unfused path: epilogue chain (tail bias-free) + the
    next LN's fused residual pass with the folded ffn.2 bias."""
    with fl.fp16_accum(acc_mode):
        d = fl.fp16_relu_chain(h2, [ffn[0], ffn[2]], tail_bias=False)
        assert d is not None
        if ln is None:
            return x + (d.float() + ffn[2].bias).half()
        return ln.forward_add(x, d, ffn[2].bias)


def _ref(x, h2, ffn, ln, mask=None):
    """fp32 reference from the fp16 inputs and the fp16-rounded weights;
    the hidden is rounded to fp16 like both kernels' stored hidden. `mask`
    (unpacked bits) pins the ReLU decisions to one implementation's."""
    w0 = ffn[0].weight.detach().half().float()
    w2 = ffn[2].weight.detach().half().float()
    pre = h2.float() @ w0.t() + ffn[0].bias
    hid = (pre * mask.float()) if mask is not None else torch.relu(pre)
    hid = hid.half().float()
    out = x.float() + hid @ w2.t() + ffn[2].bias
    if ln is None:
        return out
    hn = torch.nn.functional.layer_norm(out.half().float(), (D,),
                                        ln.weight, ln.bias, ln.eps)
    return out, hn


MODES = [("on", "off"), ("on", "fwd"), ("up16", "all")]


@cuda
@pytest.mark.parametrize("mode,acc", MODES, ids=["-".join(m) for m in MODES])
@pytest.mark.parametrize("with_ln", [True, False], ids=["ln", "res"])
def test_forward_vs_reference_and_unfused(mode, acc, with_ln):
    _need_kernel()
    fl.bump_weights_version()
    ffn, ln = _mods(1)
    ln = ln if with_ln else None
    R = 29 * 1000 + 37                   # odd: a partial tail tile
    x, h2 = _act(R, 2, 0.7), _act(R, 3)
    got = _fused(x, h2, ffn, ln, mode, acc, False)
    old = _unfused(x, h2, ffn, ln, acc)
    ref = _ref(x, h2, ffn, ln)
    got = got if with_ln else (got,)
    old = old if with_ln else (old,)
    ref = ref if with_ln else (ref,)
    for g_, o_, r_ in zip(got, old, ref):
        assert g_.shape == (R, D) and g_.dtype is torch.float16
        assert torch.isfinite(g_).all()
        e_new, e_old = _rel(g_, r_), _rel(o_, r_)
        if mode == "up16":
            # no unfused counterpart: the fp16-accumulate class of
            # test_fp16_accum (K = 192 up, 768 down)
            assert e_new < 1e-2, e_new
        elif acc == "off":
            # same class as the unfused path (fp32 accumulate: the error
            # is the output's own fp16 rounding, ~2^-11 of max); the fused
            # kernel rounds once fewer (no fp16 ffn.2 output)
            assert e_new <= max(1.1 * e_old, 2e-3), (e_new, e_old)
            assert _rel(g_, o_) < 2e-2
        else:
            # fp16-accumulated down-projection: the kernel's accumulation
            # chain (48 sequential k16 MMAs over K = 768) is not cuBLAS's
            # COMPUTE_16F order — same class, looser constant
            assert e_new <= max(1.5 * e_old, 5e-3), (e_new, e_old)
            assert _rel(g_, o_) < 2e-2


@cuda
@pytest.mark.parametrize("mode,acc", MODES, ids=["-".join(m) for m in MODES])
def test_grad_free_forward_is_bitwise_the_grad_forward(mode, acc):
    _need_kernel()
    fl.bump_weights_version()
    ffn, ln = _mods(4)
    R = 29 * 517                         # odd row count
    x, h2 = _act(R, 5, 0.7), _act(R, 6)
    a = _fused(x, h2, ffn, ln, mode, acc, False)
    b = _fused(x.clone().requires_grad_(True),
               h2.clone().requires_grad_(True), ffn, ln, mode, acc, True)
    assert b[0].grad_fn is not None
    assert torch.equal(a[0], b[0].detach())
    assert torch.equal(a[1], b[1].detach())
    ra = _fused(x, h2, ffn, None, mode, acc, False)
    rb = _fused(x.clone().requires_grad_(True),
                h2.clone().requires_grad_(True), ffn, None, mode, acc, True)
    assert torch.equal(ra, rb.detach())


@cuda
@pytest.mark.parametrize("mode,acc", MODES, ids=["-".join(m) for m in MODES])
def test_rows_are_batch_and_position_invariant(mode, acc):
    """A row's bits do not depend on R or on where it falls in a 128-row
    tile: rows [37, 37 + 1001) computed alone (a different tile phase, an
    odd tail) equal the same rows of the full call (rollout N = 32768 vs
    the update's micro-batch — the r1 contract)."""
    _need_kernel()
    fl.bump_weights_version()
    ffn, ln = _mods(7)
    R = 29 * 300 + 5
    x, h2 = _act(R, 8, 0.7), _act(R, 9)
    full = _fused(x, h2, ffn, ln, mode, acc, False)
    sl = slice(37, 37 + 1001)
    part = _fused(x[sl].contiguous(), h2[sl].contiguous(), ffn, ln, mode,
                  acc, False)
    assert torch.equal(full[0][sl], part[0])
    assert torch.equal(full[1][sl], part[1])
    for r in (1, 127, 129):              # sub-tile and one-past-tile calls
        one = _fused(x[:r].contiguous(), h2[:r].contiguous(), ffn, ln, mode,
                     acc, False)
        assert torch.equal(full[0][:r], one[0])
        assert torch.equal(full[1][:r], one[1])


@cuda
def test_saved_hidden_and_mask_layout():
    """Grad mode's saved tensors: the hidden is relu(pre) in fp16 (== the
    reference hidden to fp16 rounding), the mask is (pre > 0) in
    cuBLASLt's RELU_AUX layout — bit-identical to linear_relu_aux's mask
    wherever the pre-activation is clear of the rounding noise."""
    ext = _need_kernel()
    fl.bump_weights_version()
    ffn, ln = _mods(10)
    R = 29 * 64 + 3
    x, h2 = _act(R, 11, 0.7), _act(R, 12)
    w0h, _ = fl._half_params(ffn[0].weight, ffn[0].bias)
    w2h, _ = fl._half_params(ffn[2].weight, ffn[2].bias)
    xo, ho, mean, rstd, hid, mask = ext.ffn_fused(
        x, h2, w0h, ffn[0].bias.detach(), w2h, ffn[2].bias.detach(),
        ln.weight.detach(), ln.bias.detach(), ln.eps, 0, True)
    assert hid.shape == (R, HID) and mask.shape == (R, HID // 8)
    assert mean.shape == rstd.shape == (R,)
    pre = h2.float() @ w0h.float().t() + ffn[0].bias
    clear = pre.abs() > 1e-2
    bits = _unpack(mask).bool()
    assert torch.equal(bits[clear], (pre > 0)[clear])
    assert _rel(hid, torch.relu(pre)) < 2e-3
    assert torch.equal(hid > 0, (hid > 0) & bits)   # hid > 0 => bit set
    # same layout as the cuBLASLt epilogue's own mask
    _, lt_mask = ext.linear_relu_aux(h2, w0h, fl._half_params(
        ffn[0].weight, ffn[0].bias)[1])
    lt_bits = _unpack(lt_mask).bool()
    assert torch.equal(bits[clear], lt_bits[clear])
    # grad-free call: same outputs, nothing saved
    r = ext.ffn_fused(x, h2, w0h, ffn[0].bias.detach(), w2h,
                      ffn[2].bias.detach(), ln.weight.detach(),
                      ln.bias.detach(), ln.eps, 0, False)
    assert torch.equal(r[0], xo) and torch.equal(r[1], ho)
    assert r[2] is None and r[3] is None and r[4] is None and r[5] is None
    # stats are the LN of the ROUNDED residual output
    xf = xo.float()
    assert (mean - xf.mean(1)).abs().max() < 1e-4
    assert _rel(rstd, 1 / (xf.var(1, unbiased=False) + ln.eps).sqrt()) < 1e-3


@cuda
@pytest.mark.parametrize("acc", ["off", "all"])
@pytest.mark.parametrize("with_ln", [True, False], ids=["ln", "res"])
def test_backward_vs_reference_and_unfused(acc, with_ln):
    """Every gradient (x, h2, W0, b0, W2, b2, LN w/b) vs an fp32 reference
    using each path's own ReLU mask, at most the unfused path's error
    class (same backward kernels; only the forward roundings differ)."""
    ext = _need_kernel()
    ffn, ln = _mods(13)
    ln = ln if with_ln else None
    R = 29 * 2048 + 11
    x0, h0 = _act(R, 14, 0.7), _act(R, 15)
    g = torch.Generator(device="cuda").manual_seed(16)
    gx = (torch.randn(R, D, device="cuda", generator=g) * 0.01).half()
    gh = (torch.randn(R, D, device="cuda", generator=g) * 0.01).half()
    params = list(ffn.parameters()) + (list(ln.parameters()) if ln else [])

    def run(path):
        fl.bump_weights_version()
        for p_ in params:
            p_.grad = None
        x = x0.clone().requires_grad_(True)
        h2 = h0.clone().requires_grad_(True)
        if path == "fused":
            out = _fused(x, h2, ffn, ln, "on", acc, True)
        else:
            out = _unfused(x, h2, ffn, ln, acc)
        outs = out if with_ln else (out,)
        torch.autograd.backward(list(outs), [gx, gh][:len(outs)])
        return [x.grad, h2.grad] + [p_.grad.clone() for p_ in params]

    got, old = run("fused"), run("unfused")
    # reference with the fused kernel's mask / with the Lt chain's mask
    w0h, bh0 = fl._half_params(ffn[0].weight, ffn[0].bias)
    w2h, _ = fl._half_params(ffn[2].weight, ffn[2].bias)
    fmask = _unpack(ext.ffn_fused(
        x0, h0, w0h, ffn[0].bias.detach(), w2h, ffn[2].bias.detach(),
        None, None, 0.0, 0, True)[5])
    lmask = _unpack(ext.linear_relu_aux(h0, w0h, bh0)[1])

    def ref_grads(mask):
        rp = [p_.detach().clone().requires_grad_(True) for p_ in params]
        x = x0.float().requires_grad_(True)
        h2 = h0.float().requires_grad_(True)
        w0, b0, w2, b2 = rp[:4]
        pre = h2 @ w0.half().float().t() + b0
        out = x + (pre * mask.float()) @ w2.half().float().t() + b2
        outs = [out]
        if with_ln:
            outs.append(torch.nn.functional.layer_norm(
                out, (D,), rp[4], rp[5], ln.eps))
        torch.autograd.backward(outs, [gx.float(), gh.float()][:len(outs)])
        return [x.grad, h2.grad] + [p_.grad for p_ in rp]

    ref_f, ref_o = ref_grads(fmask), ref_grads(lmask)
    names = ["x", "h2", "w0", "b0", "w2", "b2", "lnw", "lnb"]
    for n, a, o, rf, ro in zip(names, got, old, ref_f, ref_o):
        e_new, e_old = _fro(a, rf), _fro(o, ro)
        assert e_new <= max(1.25 * e_old, 3e-3), (n, e_new, e_old)
        assert _fro(a, o) < 2e-2, n


def _policy(fused, acc, n_blocks=3):
    torch.manual_seed(0)
    p = NotePolicy(embed=192, trunk_dim=384, n_blocks=n_blocks, n_heads=4,
                   n_gears=5, note_geo=True, fp16_accum=acc,
                   fused_ffn=fused).cuda()
    p.fp16 = True
    p.flatten_fp16()
    return p


@cuda
@pytest.mark.parametrize("fused,acc", [("on", "off"), ("on", "all"),
                                       ("up16", "all")])
def test_policy_rollout_and_update_forwards_agree_bitwise(fused, acc,
                                                          monkeypatch):
    """The r1 contract with the flag on: the rollout's grad-free forward
    on fp32 obs and the update's grad-mode forward on the half copy are
    bit-identical (odd B: 1000 x 29 rows, a partial tail tile), and the
    fused kernel really ran at every block (n_blocks - 1 LN variants + the
    final block's residual variant, per forward)."""
    _need_kernel()
    calls = []
    orig = fl.fused_ffn

    def counted(*a, **k):
        calls.append(k.get("ln", a[3] if len(a) > 3 else None) is not None)
        return orig(*a, **k)
    monkeypatch.setattr(fl, "fused_ffn", counted)
    p = _policy(fused, acc)
    obs = torch.randn(1000, OBS_DIM, device="cuda") * 0.7
    with torch.no_grad():
        l_roll, v_roll = p.forward_stacked(obs)
    assert calls == [True, True, False]
    l_upd, v_upd = p.forward_stacked(obs.half())
    assert torch.equal(l_roll, l_upd.detach())
    assert torch.equal(v_roll, v_upd.detach())
    assert fl._FFN_MODE == "off"         # context restored


@cuda
def test_policy_falls_back_where_the_kernel_does_not_apply():
    """embed 128 (hidden 512): the flag is inert — bitwise the unfused
    forward, and no fused call."""
    _need_kernel()
    obs = torch.randn(256, OBS_DIM, device="cuda") * 0.7
    out = {}
    for mode in ("off", "on"):
        torch.manual_seed(0)
        p = NotePolicy(embed=128, trunk_dim=256, n_blocks=2, n_gears=5,
                       fused_ffn=mode).cuda()
        p.fp16 = True
        p.flatten_fp16()
        with torch.no_grad():
            out[mode] = p.forward_stacked(obs)
    assert torch.equal(out["off"][0], out["on"][0])
    assert torch.equal(out["off"][1], out["on"][1])


@cuda
def test_policy_grads_vs_fp32_reference():
    """Whole policy, one PPO fwd+bwd (6 blocks, the recipe): the fused
    path's logits, value and every param grad sit as close to the fp32
    record path's as the unfused fp16 path's (test_relu_chain's gate)."""
    _need_kernel()
    import bench_policy_update as bpu
    torch.manual_seed(0)
    src = NotePolicy(embed=192, trunk_dim=384, n_blocks=6, n_heads=4,
                     n_gears=5, ffn_mult=4, note_geo=True)
    data = bpu.make_minibatch(2048, torch.device("cuda"), 5)

    def run(mode):
        src.fused_ffn = src.arch["fused_ffn"] = (
            "off" if mode == "fp32" else mode)
        pol = bpu.fresh_policy(src, torch.device("cuda"))
        if mode != "fp32":
            pol.fp16 = True
            pol.flatten_fp16()
        logits, value = pol.forward_stacked(data[0])
        for p_ in pol.parameters():
            p_.grad = None
        bpu.eager_update(pol, None, *data, step=False, scale=1024.0)
        return logits.float(), value.float(), {
            n: p_.grad.float() / 1024.0 for n, p_ in pol.named_parameters()}

    lr, vr, gr = run("fp32")
    la, va, ga = run("on")
    lb, vb, gb = run("off")
    assert _fro(la, lr) <= 1.3 * _fro(lb, lr) + 1e-4
    assert _fro(va, vr) <= 1.3 * _fro(vb, vr) + 1e-4
    for n in gr:
        e_new, e_old = _fro(ga[n], gr[n]), _fro(gb[n], gr[n])
        assert e_new <= 1.3 * e_old + 2e-3, (n, e_new, e_old)
