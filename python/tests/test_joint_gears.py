"""Per-joint gear counts (2026-10-04).

A build may give some joints extra gear ids on top of the base table
(constants.JOINT_EXTRA_GEARS, mirrored by lib.rs JOINT_EXTRA_N / _FRAC and
the kernel's JOINT_EXTRA_*). Main has none — every joint keeps n_gears
ids and every table, policy and checkpoint is today's exactly; a body
branch (exp/bend7) gives the wrist bends 7.

Pinned here, build-generically (the same tests pass on main and on a
branch with extras):
  * the tables: joint_gear_table / joint_gear_counts, incl. a simulated
    extras build (monkeypatched) and the n_gears that cannot carry them;
  * the Rust sim decodes every (joint, id) to joint_gear_table's
    fraction (commanded activation of a single-joint action), ids past a
    joint's count coast; the CUDA kernel likewise (GPU);
  * the policy: padded heads, masked lanes (never sampled, no
    probability, no gradient, no entropy), the arch record, load_policy
    rebuilding the heads and refusing a layout this build cannot decode;
  * replay_format.derive_effort on ragged per-joint rows.
"""
import math

import numpy as np
import pytest
import torch

import beat_arena_rs
from beat_arena_env import constants as C
from beat_arena_env import replay_format
from beat_arena_env.chart import note_tuple
from tests._helpers import FAR_NOTE, cuda

N = C.N_JOINTS
BEND = (8, 15)


def _bend7(monkeypatch):
    """Simulate a build whose wrist bends carry one extra +/-1/4 pair."""
    extra = tuple((0.25,) if j in BEND else () for j in range(N))
    monkeypatch.setattr(C, "JOINT_EXTRA_GEARS", extra)


# --- tables ----------------------------------------------------------------

def test_tables_on_this_build():
    for k in (3, 5, 7):
        try:
            counts = C.joint_gear_counts(k)
        except ValueError:
            # a build with extras cannot put them on top of K=7
            assert any(C.JOINT_EXTRA_GEARS) and k == 7
            continue
        for j in range(N):
            t = C.joint_gear_table(k, j)
            assert len(t) == counts[j] == k + 2 * len(C.JOINT_EXTRA_GEARS[j])
            # the base prefix: today's table (fine ids x FINE_GEAR_MULT)
            base = C.gear_table(k)
            assert t[:k] == tuple(g * C.FINE_GEAR_MULT[j] if i >= 3 else g
                                  for i, g in enumerate(base))
            # extras: +/- pairs after the base ids
            for p, m in enumerate(C.JOINT_EXTRA_GEARS[j]):
                assert t[k + 2 * p:k + 2 * p + 2] == (m, -m)


def test_main_has_no_extras_or_a_bend7_layout():
    """Main: no joint has extras. A body branch may only give the bend
    joints extras (exp/bend7) — anything else is a new decision."""
    for j in range(N):
        if j not in BEND:
            assert C.JOINT_EXTRA_GEARS[j] == ()
    assert C.JOINT_EXTRA_GEARS[8] == C.JOINT_EXTRA_GEARS[15]


def test_simulated_extras_build(monkeypatch):
    _bend7(monkeypatch)
    assert C.joint_gear_table(5, 15) == (0.0, 1.0, -1.0, 0.5, -0.5,
                                         0.25, -0.25)
    assert C.joint_gear_table(5, 14) == (0.0, 1.0, -1.0, 0.25, -0.25)
    counts = C.joint_gear_counts(5)
    assert counts[8] == counts[15] == 7
    assert sum(counts) == 5 * N + 4
    with pytest.raises(ValueError):
        C.joint_gear_counts(7)          # 7 + 2 > MAX_GEARS


# --- the Rust sim's decode -------------------------------------------------

def _single_joint_activation(sim, j, a):
    sim.set_chart([note_tuple(FAR_NOTE)])            # fresh episode
    act = [0] * N
    act[j] = a
    sim.step(act)
    return sim.last_activation()[0]


@pytest.mark.parametrize("gears", [3, 5])
def test_rust_decodes_every_joint_id(gears):
    """Each (joint, id): the sim's commanded activation (= |fraction|) is
    joint_gear_table's; ids past the joint's count coast."""
    counts = C.joint_gear_counts(gears)
    sim = beat_arena_rs.RawBeatSim(max_steps=690, gears=gears)
    for j in range(N):
        t = C.joint_gear_table(gears, j)
        for a in range(C.MAX_GEARS):
            want = abs(t[a]) if a < counts[j] else 0.0
            got = _single_joint_activation(sim, j, a)
            assert got == want, (gears, j, a, got, want)


# --- policy ----------------------------------------------------------------

def _obs(b, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(b, C.OBS_DIM, generator=g) * 0.3


def _policy(head_gears, n_gears=5, depth_init=False):
    from policy import NotePolicy, apply_depth_init
    torch.manual_seed(0)
    p = NotePolicy(embed=32, trunk_dim=48, n_blocks=1, n_heads=4,
                   n_gears=n_gears, ffn_mult=2, note_geo=True,
                   head_gears=head_gears)
    if depth_init:
        apply_depth_init(p)
    return p


HG7 = tuple(7 if j in BEND else 5 for j in range(N))


def test_uniform_policy_is_unchanged():
    p = _policy((5,) * N)
    assert p.head_sizes == (5,) * N and p.k_max == 5
    assert "head_gears" not in p.arch and p._gear_mask is None
    assert all(h.weight.shape == (5, 48) for h in p.heads)


def test_default_is_this_builds_layout(monkeypatch):
    """head_gears=None = constants.joint_gear_counts(n_gears): on main
    the uniform policy (no arch record), on an extras build its layout."""
    for hg in (None, C.joint_gear_counts(5)):
        p, q = _policy(None), _policy(hg)
        assert p.head_sizes == C.joint_gear_counts(5) and q.arch == p.arch
        for (n1, t1), (n2, t2) in zip(p.state_dict().items(),
                                      q.state_dict().items()):
            assert n1 == n2 and torch.equal(t1, t2)
    _bend7(monkeypatch)
    assert _policy(None).head_sizes == HG7


def test_masked_heads():
    p = _policy(HG7, depth_init=True)
    assert p.k_max == 7 and p.arch["head_gears"] == list(HG7)
    obs = _obs(64)
    with torch.no_grad():
        logits, _ = p.forward_stacked(obs)
        assert logits.shape == (64, N, 7)
        for j in range(N):
            assert (logits[:, j, HG7[j]:] < -1e8).all()
            assert (logits[:, j, :HG7[j]].abs() < 1e3).all()
        # sampling never picks a masked lane; every live lane is reached
        torch.manual_seed(1)
        seen = torch.zeros(N, 7, dtype=torch.long)
        for _ in range(40):
            a, lp, ent, _ = p.get_action_and_value(obs)
            seen += torch.nn.functional.one_hot(a, 7).sum(0)
            assert torch.isfinite(lp).all() and torch.isfinite(ent).all()
        for j in range(N):
            assert seen[j, HG7[j]:].sum() == 0
            assert (seen[j, :HG7[j]] > 0).all()
        # depth-init heads are ~uniform over each head's OWN count
        ent_h = p._ent_h.mean(0)
        for j in range(N):
            assert ent_h[j] == pytest.approx(math.log(HG7[j]), abs=1e-3)
        # an E = 0 Gumbel draw (log 0 = -inf -> +inf score on every lane)
        # still cannot select a masked lane
        torch.manual_seed(2)
        orig = torch.Tensor.exponential_
        try:
            torch.Tensor.exponential_ = lambda self, *a, **k: self.zero_()
            a, _, _, _ = p.get_action_and_value(obs[:4])
        finally:
            torch.Tensor.exponential_ = orig
        for j in range(N):
            assert (a[:, j] < HG7[j]).all()


def test_masked_lanes_get_no_gradient():
    p = _policy(HG7, depth_init=True)
    obs = _obs(32)
    with torch.no_grad():
        act, _, _, _ = p.get_action_and_value(obs)
    _, lp, ent, v = p.get_action_and_value(obs, act)
    (lp.sum() + 0.1 * ent.sum() + v.sum()).backward()
    for j, h in enumerate(p.heads):
        k = HG7[j]
        assert (h.weight.grad[k:] == 0).all() and (h.bias.grad[k:] == 0).all()
        assert h.weight.grad[:k].abs().sum() > 0


def test_load_policy_rebuilds_and_checks_the_layout(monkeypatch):
    from policy import load_policy
    p = _policy(HG7)
    ck = {"state_dict": p.state_dict(), "arch": p.arch}
    if C.joint_gear_counts(5) != HG7:
        # this build decodes uniform (or other) gears: refused
        with pytest.raises(SystemExit, match="per-joint gear counts"):
            load_policy(ck)
    _bend7(monkeypatch)                  # a build that decodes them
    q = load_policy(ck)
    assert q.head_sizes == HG7 and q.arch == p.arch
    obs = _obs(8)
    with torch.no_grad():
        assert torch.equal(p.forward_stacked(obs)[0],
                           q.forward_stacked(obs)[0])
    # and on that build a uniform checkpoint (no head_gears record) is
    # refused — load_policy never applies the build default to it
    u = _policy((5,) * N)
    with pytest.raises(SystemExit, match="per-joint gear counts"):
        load_policy({"state_dict": u.state_dict(), "arch": u.arch})


# --- replay format -----------------------------------------------------------

def test_derive_effort_with_ragged_rows(monkeypatch):
    _bend7(monkeypatch)
    body = replay_format.body_tables(5, [1.0] * C.TAU_FEAT)
    assert [len(r) for r in body["gear_frac"]] == list(C.joint_gear_counts(5))
    act = np.zeros((3, N), dtype=np.int64)
    act[0, 15] = 5          # +1/4 on the R bend (an extra id)
    act[1, 15] = 3          # +1/2 (the 2x fine id)
    act[2, 14] = 4          # -1/4 on the R roll
    a, t = replay_format.derive_effort(act, body)
    assert a[:, 0].tolist() == [0.25, 0.5, 0.25]
    assert t[0, 0] == pytest.approx(0.25 * C.TAU_MAX[15])


# --- CUDA ------------------------------------------------------------------

@cuda
def test_cuda_decodes_every_joint_id():
    """The kernel's per-joint decode (JOINT_EXTRA_* mirror): env e
    commands one (joint, id); its activation telemetry after one tick is
    |joint_gear_table[id]|, 0 past the joint's count."""
    from beat_arena_env.cuda_env import CudaBeatVecEnv
    gears = 5
    counts = C.joint_gear_counts(gears)
    pairs = [(j, a) for j in range(N) for a in range(C.MAX_GEARS)]
    env = CudaBeatVecEnv(len(pairs), charts=[[FAR_NOTE]] * len(pairs),
                         max_steps=690, gears=gears, device="cuda",
                         dtype=torch.float64, auto_reset=False)
    assert env.joint_gears == counts
    env.reset()
    env.act_stat.zero_()
    act = torch.zeros((len(pairs), N), dtype=torch.int64)
    for e, (j, a) in enumerate(pairs):
        act[e, j] = a
    env.step(act)
    got = env.act_stat[:, 0].cpu().numpy()
    for e, (j, a) in enumerate(pairs):
        t = C.joint_gear_table(gears, j)
        want = abs(t[a]) if a < counts[j] else 0.0
        assert got[e] == want, (j, a, got[e], want)
