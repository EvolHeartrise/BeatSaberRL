"""Architecture A/B options (2026-10-03): --emb-ln, --value-residual,
--final-ln, --dwa, --trunk-arch simbav2.

Pinned here: all off = the pre-option network bit for bit (its own
source, git 1ac0254); value residual at b = 0 and DWA at the identity
are the baseline on the fp32 path; emb-LN / final-LN / SimBaV2 are
finite and train every parameter; the SimBaV2 rows stay unit-norm after
a real Muon + Adam step; every option survives save -> load_policy and
is inherited / refused on --resume like the other arch flags; a tiny
CPU trainer run with all five on, then resumed. CUDA-only: each option
on the production fp16 stream (fp16 all, fp16-accum all, fused FFN) —
rollout and update forwards bitwise equal (the r1 contract), finite
backward."""
import importlib.util
import json
import os
import subprocess
import sys

import pytest
import torch

import policy as P
import train_batch as tb

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROOT = os.path.dirname(HERE)

KW = dict(embed=64, trunk_dim=96, n_blocks=3, n_heads=4, n_gears=5,
          note_geo=True)
OPTS = {"emb_ln": dict(emb_ln=True),
        "value_residual": dict(value_residual=True),
        "final_ln": dict(final_ln=True),
        "dwa": dict(dwa=True),
        "simbav2": dict(trunk_arch="simbav2"),
        "all": dict(emb_ln=True, value_residual=True, final_ln=True,
                    dwa=True, trunk_arch="simbav2")}
NEW_KEYS = {"emb_ln", "value_residual", "final_ln", "dwa"}


def _obs(n=16, seed=1):
    g = torch.Generator().manual_seed(seed)
    obs = torch.randn(n, P.OBS_DIM, generator=g) * 0.5
    # a few masked note slots, like real windows
    obs[:, P.OFF_NOTE_MASK:P.OFF_NOTE_MASK + P.N_NOTE_SLOTS] = (
        torch.rand(n, P.N_NOTE_SLOTS, generator=g) > 0.3).float()
    return obs


def _pol(seed=0, **opts):
    torch.manual_seed(seed)
    return P.NotePolicy(**KW, **opts)


def _fwd_bwd(p, obs):
    p.zero_grad(set_to_none=True)
    lg, v = p.forward_stacked(obs)
    (lg.square().mean() + v.square().mean()).backward()
    return lg.detach(), v.detach()


# --- all off == the pre-option code ----------------------------------------

def _main_policy_module(tmp_path):
    """policy.py at 1ac0254 (main before the options) as a module; None
    when git or the commit is unavailable (shallow CI clones)."""
    try:
        src = subprocess.run(
            ["git", "-C", ROOT, "show", "1ac0254:python/policy.py"],
            check=True, capture_output=True, text=True).stdout
    except (OSError, subprocess.CalledProcessError):
        return None
    path = tmp_path / "policy_main_1ac0254.py"
    path.write_text(src)
    spec = importlib.util.spec_from_file_location("policy_main_1ac0254",
                                                  path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_all_off_is_bit_identical_to_the_pre_option_policy(tmp_path):
    old_mod = _main_policy_module(tmp_path)
    if old_mod is None:
        pytest.skip("policy.py at git 1ac0254 unavailable")
    torch.manual_seed(0)
    old = old_mod.NotePolicy(**KW)
    new = _pol()
    # same parameters, same order, same init stream
    so, sn = old.state_dict(), new.state_dict()
    assert list(so) == list(sn)
    assert all(torch.equal(so[k], sn[k]) for k in so)
    assert [n for n, _ in old.named_parameters()] == [
        n for n, _ in new.named_parameters()]
    # the arch only gains the four option markers (all off); trunk simba
    assert set(new.arch) - set(old.arch) == NEW_KEYS
    assert not any(new.arch[k] for k in NEW_KEYS)
    assert {k: v for k, v in new.arch.items() if k in old.arch} == old.arch
    obs = _obs()
    lo, vo = _fwd_bwd(old, obs)
    ln, vn = _fwd_bwd(new, obs)
    assert torch.equal(lo, ln) and torch.equal(vo, vn)
    for (n, a), b in zip(old.named_parameters(), new.parameters()):
        assert torch.equal(a.grad, b.grad), n
    with torch.no_grad():
        ao = old.get_action_and_value(obs, action=lo.argmax(-1))
        an = new.get_action_and_value(obs, action=lo.argmax(-1))
    assert all(torch.equal(x, y) for x, y in zip(ao[1:], an[1:]))


def test_flags_default_off():
    a = tb.build_parser().parse_args([])
    assert (a.emb_ln, a.value_residual, a.final_ln, a.dwa) == (
        False, False, False, False)
    assert a.trunk_arch == "simba"


# --- baseline-equivalent at init -------------------------------------------

def _baseline_and(opts):
    base = _pol()
    p = _pol(**opts)
    # the options' own params draw no RNG: every shared tensor is equal
    sb = base.state_dict()
    sp = p.state_dict()
    assert all(torch.equal(sb[k], sp[k]) for k in sb)
    return base, p


def test_value_residual_at_b0_is_the_baseline_and_v0_gets_gradient():
    base, p = _baseline_and(dict(value_residual=True))
    assert p.blocks[0].vres_a is None             # block 0 provides v_0
    for blk in p.blocks[1:]:
        assert blk.vres_a.item() == 1.0 and blk.vres_b.item() == 0.0
    obs = _obs()
    lb, vb = _fwd_bwd(base, obs)
    lp, vp = _fwd_bwd(p, obs)
    assert torch.equal(lb, lp) and torch.equal(vb, vp)
    # b is learnable from the start: dL/db = <dL/dv_mix, v_0> != 0
    for blk in p.blocks[1:]:
        assert blk.vres_b.grad is not None and blk.vres_b.grad.abs() > 0
    g0 = p.blocks[0].v_proj.weight.grad.clone()
    assert torch.equal(g0, base.blocks[0].v_proj.weight.grad)
    # with b != 0 the gradient reaches v_0 through the later blocks
    with torch.no_grad():
        for blk in p.blocks[1:]:
            blk.vres_b.fill_(0.5)
    _fwd_bwd(p, obs)
    assert not torch.allclose(p.blocks[0].v_proj.weight.grad, g0)


def test_dwa_at_identity_is_the_baseline():
    base, p = _baseline_and(dict(dwa=True))
    for i, blk in enumerate(p.blocks):
        w = torch.zeros(i + 2)
        w[-1] = 1.0
        assert torch.equal(blk.dwa.detach(), w)
    obs = _obs()
    lb, vb = _fwd_bwd(base, obs)
    lp, vp = _fwd_bwd(p, obs)
    assert torch.equal(lb, lp) and torch.equal(vb, vp)
    pg = dict(p.named_parameters())
    for n, a in base.named_parameters():
        assert torch.equal(a.grad, pg[n].grad), n
    # every mixing weight trains, the off-diagonal ones included
    for blk in p.blocks:
        assert (blk.dwa.grad.abs() > 0).all()


def test_vmix_function_matches_plain_autograd():
    torch.manual_seed(3)
    leaves = [torch.randn(4, 29, 8, requires_grad=True),
              torch.randn(4, 29, 8, requires_grad=True),
              torch.tensor([0.7], requires_grad=True),
              torch.tensor([0.3], requires_grad=True)]
    g = torch.randn(4, 29, 8)
    out = P._VMix.apply(*leaves)
    got = torch.autograd.grad(out, leaves, g)
    v, v0, a, b = leaves
    ref = a * v + b * v0
    want = torch.autograd.grad(ref, leaves, g)
    assert torch.allclose(out, ref, atol=1e-6)
    for x, y in zip(got, want):
        assert torch.allclose(x, y, rtol=1e-5, atol=1e-5)


def test_dwa_mix_is_the_weighted_sum():
    xs = [torch.randn(4, 29, 8) for _ in range(3)]
    w = torch.tensor([0.2, -0.5, 1.3])
    want = 0.2 * xs[0] - 0.5 * xs[1] + 1.3 * xs[2]
    assert torch.allclose(P._dwa_mix(w, xs), want, atol=1e-6)


@pytest.mark.parametrize("name", ["emb_ln", "final_ln", "simbav2", "all"])
def test_non_equivalent_options_are_finite_and_train_everything(name):
    p = _pol(**OPTS[name])
    lg, v = _fwd_bwd(p, _obs())
    assert lg.shape == (16, P.N_ACTION_HEADS, 5) and v.shape == (16,)
    assert torch.isfinite(lg).all() and torch.isfinite(v).all()
    for n, q in p.named_parameters():
        if not q.requires_grad:
            assert n.startswith("trunk_") and n.endswith(".bias"), n
            continue
        assert q.grad is not None and torch.isfinite(q.grad).all(), n
        assert q.grad.abs().sum() > 0, n


def test_modules_exist_only_when_on():
    p = _pol()
    assert p.emb_ln is None and p.final_ln is None
    assert all(b.vres_a is None and b.dwa is None for b in p.blocks)
    q = _pol(**OPTS["all"])
    names = {n for n, _ in q.named_parameters()}
    assert {"emb_ln.weight", "final_ln.weight", "blocks.1.vres_b",
            "blocks.2.dwa", "trunk_alpha", "trunk_in.weight"} <= names
    assert not any(n.startswith(("trunk_proj", "trunk_ln", "trunk_fc"))
                   for n in names)
    # registration order: emb_ln before the blocks (enc diag group),
    # final_ln right before the trunk
    order = [n for n, _ in q.named_parameters()]
    assert order.index("emb_ln.bias") < order.index("blocks.0.ln_attn.weight")
    assert (order.index("blocks.2.dwa") < order.index("final_ln.weight")
            < order.index("trunk_in.weight"))


# --- optimizer layout + SimBaV2 projection ---------------------------------

def test_new_params_go_to_adam_and_simbav2_matrices_to_muon():
    p = _pol(**OPTS["all"])
    muon = set(tb.muon_param_names(p))
    for n, q in p.named_parameters():
        if n.startswith(("emb_ln", "final_ln", "trunk_s_", "trunk_alpha")) \
                or ".vres_" in n or n.endswith(".dwa"):
            assert q.ndim == 1 and n not in muon, n
    assert {"trunk_in.weight", "trunk_w1.weight", "trunk_w2.weight"} <= muon


def test_simbav2_rows_stay_unit_norm_after_a_muon_and_adam_step():
    p = _pol(trunk_arch="simbav2")
    lins = p.trunk_projected()
    for m in lins:
        assert torch.allclose(m.weight.norm(dim=1),
                              torch.ones(m.weight.shape[0]), atol=1e-6)
        assert not m.bias.requires_grad and torch.count_nonzero(m.bias) == 0
    muon_names = set(tb.muon_param_names(p))
    named = dict(p.named_parameters())
    adam = torch.optim.Adam([q for n, q in named.items()
                             if n not in muon_names and q.requires_grad],
                            lr=1e-2)
    muon = tb.Muon([{"params": [named[n] for n in sorted(muon_names)],
                     "lr": 1e-2, "lr_mult": 1.0}])
    s_before = p.trunk_s_in.detach().clone()
    _fwd_bwd(p, _obs())
    adam.step()
    muon.step()
    drift = max((m.weight.detach().norm(dim=1) - 1).abs().max().item()
                for m in lins)
    assert drift > 1e-4                 # the step did leave the sphere
    assert p.project_trunk_weights()
    for m in lins:
        assert torch.allclose(m.weight.norm(dim=1),
                              torch.ones(m.weight.shape[0]), atol=1e-6)
        assert torch.count_nonzero(m.bias) == 0
    assert not torch.equal(p.trunk_s_in.detach(), s_before)  # Adam owns it
    assert not _pol().project_trunk_weights()              # simba: no-op


def test_simbav2_output_scale_and_init():
    p = _pol(trunk_arch="simbav2")
    h = p._trunk(torch.randn(16, 5 * 64) * 30)
    # h = sqrt(D) * unit vector: RMS exactly 1 per element, like LN2
    assert torch.allclose(h.square().mean(-1), torch.ones(16), atol=1e-5)
    assert torch.allclose(p.trunk_alpha * p._v2_c_alpha,
                          torch.full((96,), 0.5))


def test_depth_init_leaves_the_simbav2_trunk_alone():
    p, ref = _pol(trunk_arch="simbav2"), _pol(trunk_arch="simbav2")
    msg = P.apply_depth_init(p)
    for a, b in zip(p.trunk_projected(), ref.trunk_projected()):
        assert torch.equal(a.weight, b.weight)
    assert "x0.01" in msg
    f = (2 * 3) ** -0.5
    assert torch.allclose(p.blocks[0].ffn[-1].weight,
                          ref.blocks[0].ffn[-1].weight * f)


# --- checkpoints ------------------------------------------------------------

@pytest.mark.parametrize("name", sorted(OPTS))
def test_save_load_policy_round_trip(name, tmp_path):
    p = _pol(**OPTS[name])
    with torch.no_grad():               # move off the init so it matters
        for q in p.parameters():
            if q.requires_grad and q.ndim == 1:
                q.add_(0.05 * torch.randn_like(q))
    path = tmp_path / "ck.pt"
    torch.save({"arch": p.arch, "state_dict": p.state_dict()}, path)
    ck = torch.load(path, map_location="cpu", weights_only=True)
    q = P.load_policy(ck)
    assert q.arch == p.arch
    for k, v in OPTS[name].items():
        assert q.arch[k] == v
    obs = _obs()
    with torch.no_grad():
        a, va = p.forward_stacked(obs)
        b, vb = q.forward_stacked(obs)
    assert torch.equal(a, b) and torch.equal(va, vb)


def test_old_checkpoint_without_the_keys_loads_with_options_off():
    p = _pol()
    arch = {k: v for k, v in p.arch.items() if k not in NEW_KEYS}
    q = P.load_policy({"arch": arch, "state_dict": p.state_dict()})
    assert not any(q.arch[k] for k in NEW_KEYS)
    assert q.arch["trunk_arch"] == "simba" and q.emb_ln is None


# --- --resume inheritance (pure) -------------------------------------------

def _ckpt_args(**over):
    a = vars(tb.parse_args([]))
    a.update(over)
    return a


SCHED = {"lr": 7e-4, "lr_final": 1e-4, "ent_coef": 0.03, "ent_tau": 125,
         "warmup_updates": 34, "num_updates": 100, "lr_decay": "wsd"}
ON = dict(emb_ln=True, value_residual=True, final_ln=True, dwa=True,
          trunk_arch="simbav2")


def test_resume_inherits_every_option():
    args = tb.parse_args(["--resume", "t.pt"])
    rep = tb.resolve_resume_args(args, tb.explicit_dests(["--resume", "t.pt"]),
                                 _ckpt_args(**ON), dict(SCHED), False)
    assert {d: getattr(args, d) for d in ON} == ON
    assert set(ON) <= {d for d, _, _ in rep["inherited"]}
    assert set(ON) <= tb.RESUME_ARCH


@pytest.mark.parametrize("argv,dest", [
    (["--no-emb-ln"], "emb_ln"), (["--no-value-residual"], "value_residual"),
    (["--no-final-ln"], "final_ln"), (["--no-dwa"], "dwa"),
    (["--trunk-arch", "simba"], "trunk_arch")])
def test_resume_refuses_a_typed_option_change(argv, dest):
    argv = ["--resume", "t.pt", *argv]
    args = tb.parse_args(argv)
    rep = tb.resolve_resume_args(args, tb.explicit_dests(argv),
                                 _ckpt_args(**ON), dict(SCHED), False)
    assert dest in {d for d, _, _ in rep["arch"]}


def test_check_arch_options_follows_the_checkpoint_arch():
    p = _pol(**OPTS["all"])
    ck = {"arch": dict(p.arch)}
    args = tb.parse_args([])
    changed = tb.check_arch_options(args, set(), ck, "--resume")
    assert {d: getattr(args, d) for d in ON} == ON
    assert {d for d, _, _ in changed} == set(ON)
    # a pre-option checkpoint (no keys) with a typed flag: refused ...
    old = {"arch": {k: v for k, v in _pol().arch.items()
                    if k not in NEW_KEYS}}
    args = tb.parse_args(["--emb-ln"])
    with pytest.raises(SystemExit, match="emb-ln"):
        tb.check_arch_options(args, {"emb_ln"}, old, "--resume")
    # ... or dropped with --ignore-arch-flags; untyped stays off
    args = tb.parse_args(["--emb-ln", "--ignore-arch-flags"])
    tb.check_arch_options(args, {"emb_ln"}, old, "--init-from")
    assert args.emb_ln is False
    args = tb.parse_args([])
    assert tb.check_arch_options(args, set(), old, "--resume") == []


# --- end to end: tiny CPU trainer, all five on, then resumed ---------------

COMMON = ["--device", "cpu", "--env", "rust", "--num-envs", "8",
          "--horizon", "8", "--minibatches", "1", "--n-blocks", "2",
          "--no-ranked-maps", "--no-mirror-augment", "--no-ep-stats",
          "--tau-rand", "0"]


def _run(*argv):
    r = subprocess.run([sys.executable, os.path.join(HERE, "train_batch.py"),
                        *argv], cwd=HERE, capture_output=True, text=True,
                       timeout=900)
    out = r.stdout + r.stderr
    assert r.returncode == 0, out[-4000:]
    return out


def test_cpu_trainer_all_options_and_resume(tmp_path):
    d = tmp_path / "run"
    out = _run(*COMMON, "--total-steps", "128", "--ckpt-every", "1",
               "--emb-ln", "--value-residual", "--final-ln", "--dwa",
               "--trunk-arch", "simbav2", "--depth-init",
               "--run-dir", str(d))
    assert "arch options: emb_ln, value_residual, final_ln, dwa" in out
    assert "readout trunk: simbav2" in out
    ck = torch.load(d / "trunk_upd00001.pt", map_location="cpu",
                    weights_only=True)
    for k, v in ON.items():
        assert ck["arch"][k] == v
    q = P.load_policy(ck)
    for m in q.trunk_projected():       # projected after every step
        assert torch.allclose(m.weight.norm(dim=1),
                              torch.ones(m.weight.shape[0]), atol=1e-5)
    # resume (untyped flags): the options come back from the checkpoint
    d2 = tmp_path / "res"
    out = _run(*COMMON, "--resume", str(d / "trunk_upd00001.pt"),
               "--total-steps", "192", "--run-dir", str(d2))
    rec = json.load(open(d2 / "args.json"))["resolved"]
    assert {k: rec[k] for k in ON} == ON
    assert "arch options: emb_ln, value_residual, final_ln, dwa" in out


# --- CUDA: the production fp16 stream ---------------------------------------

@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_fused_vres_attention_matches_the_rebuilt_buffer_path():
    import fp16_linear as fl
    if not fl.attn29_ok(29, 192, 4):
        pytest.skip("attn29 kernel unavailable")
    torch.manual_seed(5)
    B, T, d = 64, 29, 192
    bias = torch.zeros(B, 1, T, device="cuda", dtype=torch.float16)
    bias[:, :, 20:] = -P.MASK_BIAS
    a = torch.tensor([0.8], device="cuda", requires_grad=True)
    b = torch.tensor([0.4], device="cuda", requires_grad=True)
    src0 = (torch.randn(B, T, 3 * d, device="cuda") * 0.5).half()
    gout = torch.randn(B, T, d, device="cuda").half()
    gq = torch.randn(B, 5, d, device="cuda").half()

    def run(fused, final):
        qkv = qkv0.clone().requires_grad_(True)
        z = src0.clone().requires_grad_(True)
        v0 = z[..., 2 * d:]
        for p_ in (a, b):
            p_.grad = None
        if final:
            q = qkv[:, :5, :d]
            kv = qkv[..., d:]
            if fused:
                out = fl.attn29_qkv_vres(q, kv, v0, a, b, bias)
            else:
                vm = fl.vres_mix(kv[..., d:], v0, a, b)
                out = fl.attn29_qkv(q, torch.cat([kv[..., :d], vm], -1),
                                    bias)
            out.backward(gq)
        else:
            if fused:
                out = fl.attn29_packed_vres(qkv, v0, a, b, bias)
            else:
                vm = fl.vres_mix(qkv[..., 2 * d:], v0, a, b)
                out = fl.attn29_packed(
                    torch.cat([qkv[..., :2 * d], vm], -1), bias)
            out.backward(gout)
        return (out.detach(), qkv.grad, z.grad, a.grad.clone(),
                b.grad.clone())

    qkv0 = (torch.randn(B, T, 3 * d, device="cuda") * 0.5).half()
    for final in (False, True):
        f, r = run(True, final), run(False, final)
        assert torch.equal(f[0], r[0])               # same forward bits
        assert torch.allclose(f[1].float(), r[1].float(), atol=2e-3)
        assert torch.allclose(f[2].float(), r[2].float(), atol=2e-3)
        assert torch.allclose(f[3], r[3], rtol=1e-3, atol=1e-3)
        assert torch.allclose(f[4], r[4], rtol=1e-3, atol=1e-3)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
@pytest.mark.parametrize("name", ["off"] + sorted(OPTS))
def test_fp16_production_stream_r1_and_backward(name):
    opts = {} if name == "off" else OPTS[name]
    torch.manual_seed(0)
    p = P.NotePolicy(embed=192, trunk_dim=384, n_blocks=4, n_heads=4,
                     n_gears=5, note_geo=True, fp16_accum="all",
                     fused_ffn="on", **opts).cuda()
    p.fp16 = True
    p.flatten_fp16()
    obs = _obs(256).cuda().half()
    with torch.no_grad():
        lr_, vr = p.forward_stacked(obs)          # rollout
    lu, vu = p.forward_stacked(obs)               # update (grad mode)
    # r1: same arithmetic in both phases
    assert torch.equal(lr_, lu.detach()) and torch.equal(vr, vu.detach())
    assert torch.isfinite(lu).all() and torch.isfinite(vu).all()
    (lu.float().square().mean() + vu.square().mean()).backward()
    for n, q in p.named_parameters():
        if q.requires_grad:
            assert q.grad is not None and torch.isfinite(q.grad).all(), n
    # fp16 vs the fp32 record path: same network, fp16 rounding only
    p.fp16 = False
    with torch.no_grad():
        l32, _ = p.forward_stacked(obs.float())
    assert (l32 - lu.detach()).abs().max() < 0.05
    if name in ("value_residual", "dwa"):
        # baseline-equivalent at init on the fp16 stream too (near-tie
        # numerics for dwa: the resolved residual instead of the fused
        # delayed add)
        torch.manual_seed(0)
        b = P.NotePolicy(embed=192, trunk_dim=384, n_blocks=4, n_heads=4,
                         n_gears=5, note_geo=True, fp16_accum="all",
                         fused_ffn="on").cuda()
        b.fp16 = True
        with torch.no_grad():
            lb, _ = b.forward_stacked(obs)
            p.fp16 = True
            lp, _ = p.forward_stacked(obs)
        if name == "value_residual":
            assert torch.equal(lb, lp)
        else:
            assert (lb - lp).abs().max() < 0.05
