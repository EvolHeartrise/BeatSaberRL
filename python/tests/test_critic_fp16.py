"""adapool_v1 critic on the fp16 stream (2026-09-02, Tier 1.1 of the
09-02 static audit, A §1).

Until 2026-09-02 the critic silently ran in fp32 through stock torch:
forward_stacked recomputed its input as `tokens + delta + delta_b`,
and delta_b is the fp32 ffn.2.bias Parameter — dimensioned fp16 +
dimensioned fp32 promotes to fp32. These tests pin the fix: the
critic's inputs on the fp16 stream ARE fp16, the input is the very
tensor the final block's fused ln_attn pass wrote (reused, not
recomputed), and the critic's parameters still receive grads through
the fp16 machinery (packed k/v, SDPA, bias_relu/relu_bwd).
"""

import pytest
import torch
import torch.nn.functional as F

if not torch.cuda.is_available():
    pytest.skip("fp16 critic path needs CUDA", allow_module_level=True)

from policy import N_NOTE_SLOTS, NotePolicy, OBS_DIM, OFF_NOTE_MASK, hl_loss


def _obs(n, seed=0):
    g = torch.Generator().manual_seed(seed)
    obs = (torch.randn(n, OBS_DIM, generator=g) * 0.7)
    obs[:, OFF_NOTE_MASK:OFF_NOTE_MASK + N_NOTE_SLOTS] = (
        torch.rand(n, N_NOTE_SLOTS, generator=g) < 0.6).float()
    return obs.cuda()


def _value_loss(p, n, seed=7):
    """The trainer's critic loss: HL-Gauss cross-entropy of the head's
    bin logits against non-trivial normalized-return targets. (Until the
    scalar head was removed, 2026-09-22, this test used value**2; on the
    HL head at init that loss is ~0 by symmetry — v = softmax . centers
    ~ 0, so dL/dv = 2v/B — and the score-path grads it produces, already
    ~1e-5 under the scalar head, fall below the fp16 range with no loss
    scale: critic_ln.weight came back exactly 0.)"""
    g = torch.Generator().manual_seed(seed)
    tgt = (torch.randn(n, generator=g) * 1.5).cuda()
    return hl_loss(p._value_logits, tgt, p.value_centers, p.value_sigma)


def _policy(fp16=True, flat=False):
    torch.manual_seed(0)
    p = NotePolicy(n_gears=5).cuda()
    p.fp16 = fp16
    if flat:
        p.flatten_fp16()
    return p


class _Spy:
    """Records what reaches _critic and what critic_ln produces, and
    which tensor the final block's fused ln_attn pass returned as x."""

    def __init__(self, p):
        self.p = p
        self.calls = []
        self.ln_out = []
        self.add_x = []
        orig_critic = p._critic
        orig_ln = p.critic_ln.forward
        orig_add = p.blocks[-1].ln_attn.forward_add

        def critic(x_res, h, attn_bias):
            self.calls.append((x_res, h, attn_bias))
            return orig_critic(x_res, h, attn_bias)

        def ln(x):
            y = orig_ln(x)
            self.ln_out.append(y)
            return y

        def add(a, b, b_bias=None):
            x, y = orig_add(a, b, b_bias)
            self.add_x.append(x)
            return x, y

        p._critic = critic
        p.critic_ln.forward = ln
        p.blocks[-1].ln_attn.forward_add = add


@pytest.mark.parametrize("flat", [False, True])
@pytest.mark.parametrize("grad", [False, True])
def test_critic_inputs_are_fp16_on_fp16_stream(flat, grad):
    p = _policy(flat=flat)
    spy = _Spy(p)
    obs = _obs(64)
    with torch.set_grad_enabled(grad):
        logits, value = p.forward_stacked(obs)
    (x_res, h, ab), = spy.calls
    assert x_res.dtype is torch.float16, x_res.dtype
    assert h.dtype is torch.float16 and ab.dtype is torch.float16
    assert x_res.shape == (64, p.n_lead + N_NOTE_SLOTS, p.embed)
    assert x_res.is_contiguous()
    (hv,) = spy.ln_out
    assert hv.dtype is torch.float16, "critic_ln output promoted"
    assert value.dtype is torch.float32 and value.shape == (64,)
    assert logits.dtype is torch.float32
    assert torch.isfinite(value).all()


def test_critic_input_is_the_final_blocks_resolved_stream():
    """Reused, not recomputed: the tensor handed to _critic IS the x
    the final block's ln_attn.forward_add returned."""
    p = _policy()
    spy = _Spy(p)
    with torch.no_grad():
        p.forward_stacked(_obs(16))
    (x_res, _, _), = spy.calls
    (x_add,) = spy.add_x
    assert x_res is x_add


def test_fp16_value_tracks_fp32_value():
    """fp16-rounding class, not a routing bug: the fp16 critic's value
    stays within a few 1e-2 (relative to the value range) of the fp32
    record path on the same weights."""
    p32 = _policy(fp16=False)
    p16 = _policy(fp16=True, flat=True)
    obs = _obs(2048, seed=3)
    with torch.no_grad():
        _, v32 = p32.forward_stacked(obs)
        _, v16 = p16.forward_stacked(obs)
    scale = v32.abs().max().clamp_min(1e-3)
    rel = (v16 - v32).abs().max() / scale
    assert rel < 2e-2, f"fp16 vs fp32 value rel diff {rel:.3e}"


@pytest.mark.parametrize("flat", [False, True])
def test_critic_params_receive_grads_through_fp16_path(flat):
    p = _policy(flat=flat)
    obs = _obs(256, seed=1)
    p.forward_stacked(obs)
    _value_loss(p, 256).backward()
    crit = ["critic_ln", "critic_q", "critic_k", "critic_v",
            "critic_out", "critic_mlp"]
    for name, prm in p.named_parameters():
        if any(name.startswith(c) for c in crit):
            assert prm.grad is not None, name
            assert torch.isfinite(prm.grad).all(), name
            assert prm.grad.abs().sum() > 0, name
    # the value loss reaches the shared blocks (through the resolved
    # stream's _LnAddFn gx edge) but never the action heads
    assert p.blocks[0].q_proj.weight.grad is not None
    assert p.blocks[-1].k_proj.weight.grad is not None
    assert p.heads[0].weight.grad is None
    # fp16-rounding class, not a routing bug: on the critic grads that
    # carry signal at random init (v / out / value MLP, |g| ~ 1e-4..1e-2)
    # the fp16 path tracks the fp32 record path at cos > 0.998
    # (measured 2026-09-02, both SDPA and legacy attention). The
    # score-path grads (critic_q / critic_k / critic_ln) are NOT
    # compared here: at init they sit at ~1e-5 (critic_k.bias is
    # mathematically 0 — a per-key constant is softmax-invariant), where
    # any fp16 path is at the noise floor (cos 0.3-0.4 measured, rising
    # to 0.85-0.97 with q/k scaled x8). The run-level gate for those is
    # the 5-update ev/acc comparison in the audit's Tier 1 results.
    p32 = _policy(fp16=False)
    p32.forward_stacked(obs)
    _value_loss(p32, 256).backward()
    for name in ("critic_v", "critic_out", "critic_mlp.0", "critic_mlp.2",
                 "critic_mlp.4"):
        m16 = dict(p.named_modules())[name]
        m32 = dict(p32.named_modules())[name]
        cos = F.cosine_similarity(m16.weight.grad.flatten(),
                                  m32.weight.grad.flatten(), dim=0)
        assert cos > 0.98, f"{name}: grad cos {cos:.4f}"
