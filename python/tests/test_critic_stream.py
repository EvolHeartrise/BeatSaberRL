"""2.2 of the 09-02 static audit (B §2, 2026-09-03): the rollout critic
on a side CUDA stream.

Pins: (1) the value from the side-stream path is BIT-IDENTICAL to the
inline path on the same weights and obs (same kernels, same batch,
same per-stream order) — on both dtype streams, flat and non-flat;
(2) the actor path (logits, sampled actions, logprobs) is untouched
and the sampling RNG stream is not consumed by the critic; (3) the
side stream is actually used and the inputs are held until the join;
(4) the grad-enabled forward ignores critic_async (the update phase is
untouched); (5) a missed join is an error, not silent corruption; (6)
allocator churn on the main stream between the async forward and the
join cannot corrupt the critic's inputs (the reason the policy holds
them on _critic_pending).
"""

import pytest
import torch

if not torch.cuda.is_available():
    pytest.skip("side-stream critic needs CUDA", allow_module_level=True)

from policy import N_NOTE_SLOTS, NotePolicy, OBS_DIM, OFF_NOTE_MASK


def _obs(n, seed=0):
    g = torch.Generator().manual_seed(seed)
    obs = (torch.randn(n, OBS_DIM, generator=g) * 0.7)
    obs[:, OFF_NOTE_MASK:OFF_NOTE_MASK + N_NOTE_SLOTS] = (
        torch.rand(n, N_NOTE_SLOTS, generator=g) < 0.6).float()
    return obs.cuda()


def _policy(fp16=True, flat=False, critic="adapool_v1"):
    torch.manual_seed(0)
    p = NotePolicy(n_gears=5, critic=critic).cuda()
    p.fp16 = fp16
    if flat:
        p.flatten_fp16()
    return p


@pytest.mark.parametrize("flat", [False, True])
@pytest.mark.parametrize("fp16", [False, True])
def test_side_stream_value_is_bit_identical(fp16, flat):
    p = _policy(fp16=fp16, flat=flat and fp16)
    obs = _obs(1024, seed=1)
    with torch.no_grad():
        lg0, v0 = p.forward_stacked(obs)
    p.critic_side_stream(True)
    with torch.no_grad():
        lg1, v1 = p.forward_stacked(obs, critic_async=True)
        p.critic_join()
    torch.cuda.synchronize()
    assert torch.equal(lg0, lg1)
    assert torch.equal(v0, v1)
    # and back to inline: same answer, no stream involved
    p.critic_side_stream(False)
    with torch.no_grad():
        lg2, v2 = p.forward_stacked(obs, critic_async=True)
    assert p._critic_pending is None
    assert torch.equal(v0, v2) and torch.equal(lg0, lg2)


def test_actions_and_logprobs_identical_same_seed():
    """The sampling RNG stream is untouched by where the critic runs:
    same seed -> same actions, logprobs AND values."""
    p = _policy(flat=True)
    obs = _obs(2048, seed=2)
    torch.manual_seed(123)
    with torch.no_grad():
        a0, lp0, _, v0 = p.get_action_and_value(obs, need_entropy=False)
    p.critic_side_stream(True)
    torch.manual_seed(123)
    with torch.no_grad():
        a1, lp1, _, v1 = p.get_action_and_value(obs, need_entropy=False,
                                                critic_async=True)
        p.critic_join()
    torch.cuda.synchronize()
    assert torch.equal(a0, a1) and torch.equal(lp0, lp1)
    assert torch.equal(v0, v1)


def test_critic_actually_runs_on_the_side_stream_and_holds_inputs():
    p = _policy(flat=True)
    p.critic_side_stream(True)
    seen = []
    orig = p._critic

    def spy(x_res, h, ab):
        seen.append(torch.cuda.current_stream())
        return orig(x_res, h, ab)
    p._critic = spy
    with torch.no_grad():
        _, v = p.forward_stacked(_obs(64), critic_async=True)
        assert p._critic_pending is not None
        ev, x_res, h, ab = p._critic_pending
        assert x_res.shape == (64, p.n_lead + N_NOTE_SLOTS, p.embed)
        assert h.shape == (64, 256) and ab.shape == (64, 1, p.n_lead + N_NOTE_SLOTS)
        p.critic_join()
    assert p._critic_pending is None
    assert len(seen) == 1 and seen[0] == p._critic_stream
    assert seen[0] != torch.cuda.default_stream()
    torch.cuda.synchronize()
    assert torch.isfinite(v).all()


def test_grad_enabled_forward_ignores_critic_async():
    """The update phase is untouched: with grad on, critic_async is a
    no-op — inline critic on the current stream, value has a grad_fn,
    nothing pending."""
    p = _policy(flat=True)
    p.critic_side_stream(True)
    seen = []
    orig = p._critic

    def spy(x_res, h, ab):
        seen.append(torch.cuda.current_stream())
        return orig(x_res, h, ab)
    p._critic = spy
    _, v = p.forward_stacked(_obs(32), critic_async=True)
    assert p._critic_pending is None
    assert v.requires_grad and v.grad_fn is not None
    assert seen[0] == torch.cuda.current_stream() != p._critic_stream
    v.sum().backward()
    assert p.critic_mlp[0].weight.grad is not None


def test_missed_join_raises():
    p = _policy(flat=True)
    p.critic_side_stream(True)
    with torch.no_grad():
        p.forward_stacked(_obs(16), critic_async=True)
        with pytest.raises(RuntimeError, match="critic_join"):
            p.forward_stacked(_obs(16))
        p.critic_join()
        p.forward_stacked(_obs(16))     # fine after the join
    with pytest.raises(AssertionError):
        with torch.no_grad():
            p.forward_stacked(_obs(16), critic_async=True)
            p.critic_side_stream(False)
    p.critic_join()


def test_main_stream_churn_between_forward_and_join_cannot_corrupt():
    """Simulates the rollout: async forward, then heavy allocate/free/
    write traffic on the main stream (what venv.step and the sampling
    tail do) BEFORE the join, for many ticks. If the critic's inputs
    were released early the allocator would recycle their blocks into
    this traffic while the side stream still reads them and the value
    would diverge from the inline reference."""
    p = _policy(flat=True)
    obs = [_obs(4096, seed=10 + i) for i in range(4)]
    with torch.no_grad():
        ref = [p.forward_stacked(o)[1].clone() for o in obs]
    p.critic_side_stream(True)
    with torch.no_grad():
        for it in range(24):
            o = obs[it % 4]
            _, v = p.forward_stacked(o, critic_async=True)
            # churn: several tensors the size of x_res / h / attn_bias,
            # written on the main stream, freed immediately
            for _ in range(6):
                junk = torch.empty(4096, 29, 128, device="cuda",
                                   dtype=torch.float16)
                junk.fill_(float(it))
                junk2 = torch.empty(4096, 256, device="cuda",
                                    dtype=torch.float16).fill_(-1.0)
                del junk, junk2
            p.critic_join()
            out = v * 1.0            # consume on the main stream
            assert torch.equal(out, ref[it % 4]), f"tick {it}"
    torch.cuda.synchronize()


def test_legacy_critic_stays_inline():
    p = _policy(critic="linear_v0", flat=True)
    p.critic_side_stream(True)
    with torch.no_grad():
        _, v0 = p.forward_stacked(_obs(8))
        _, v1 = p.forward_stacked(_obs(8), critic_async=True)
    assert p._critic_pending is None
    assert torch.equal(v0, v1)


def test_deferred_launch_probe_is_bit_identical_and_grad_free():
    """BEAT_CRITIC_LATE (the 2.3 go/no-go probe): forward returns
    value=None, critic_launch_deferred() — called OUTSIDE no_grad, as
    the trainer does — enqueues the same critic on the side stream,
    grad-free, bit-identical to inline."""
    p = _policy(flat=True)
    obs = _obs(512, seed=5)
    with torch.no_grad():
        _, v0 = p.forward_stacked(obs)
    p.critic_side_stream(True)
    p._critic_late = True
    with torch.no_grad():
        lg, v = p.forward_stacked(obs, critic_async=True)
    assert v is None and p._critic_pending is not None
    assert p._critic_pending[0] is None
    v = p.critic_launch_deferred()          # grad mode here, on purpose
    assert not v.requires_grad and v.grad_fn is None
    assert p._critic_pending is not None and p._critic_pending[0] is not None
    p.critic_join()
    torch.cuda.synchronize()
    assert torch.equal(v, v0)
    p._critic_late = False
