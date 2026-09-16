"""Policy + vec-env tests (transformer-only since 2026-07-20; env v10
shapes since P4: obs 340, 19 tokens incl. the body token, 29 gear heads)."""

import numpy as np
import pytest
import torch

from beat_arena_env import RustBeatEnv, OFF_NOTES, OFF_NOTE_MASK, NOTE_FEAT
from beat_arena_env.constants import N_NOTE_SLOTS
from beat_arena_env.constants import N_JOINTS
from beat_arena_env.vec_env import BeatVecEnv
from policy import HEAD_SIZES, N_ACTION_HEADS, NotePolicy, OBS_DIM

ACTION_NVEC = [3] * N_JOINTS


def _obs_batch(n=6):
    # short chart -> fewer pending notes than slots, so masked slots exist
    env = RustBeatEnv(seed=1, chart_len_s=6.0)
    obs, _ = env.reset()
    rows = [obs]
    rng = np.random.default_rng(0)
    for _ in range(n - 1):
        obs, _, done, _, _ = env.step(rng.integers(0, ACTION_NVEC))
        if done:
            obs, _ = env.reset()
        rows.append(obs)
    return torch.as_tensor(np.stack(rows), dtype=torch.float32)


def test_output_shapes():
    torch.manual_seed(0)
    policy = NotePolicy()
    obs = _obs_batch()
    logits, value = policy(obs)
    assert [l.shape for l in logits] == [(6, n) for n in HEAD_SIZES]
    assert value.shape == (6,)
    a, logp, ent, v = policy.get_action_and_value(obs)
    assert a.shape == (6, N_ACTION_HEADS) and logp.shape == (6,)


def test_junk_in_masked_slots_cannot_leak():
    """Garbage in invalid note slots must not change any output."""
    torch.manual_seed(0)
    policy = NotePolicy()
    obs = _obs_batch()
    mask = obs[:, OFF_NOTE_MASK:OFF_NOTE_MASK + N_NOTE_SLOTS]
    assert (mask == 0).any(), "test needs at least one empty slot"

    junk = obs.clone()
    g = torch.Generator().manual_seed(42)
    for b in range(obs.shape[0]):
        for slot in torch.nonzero(mask[b] == 0).flatten():
            s = OFF_NOTES + int(slot) * NOTE_FEAT
            junk[b, s:s + NOTE_FEAT] = torch.randn(NOTE_FEAT, generator=g) * 5

    action = torch.stack(
        [torch.randint(0, n, (obs.shape[0],)) for n in HEAD_SIZES], dim=1)
    _, lp1, ent1, v1 = policy.get_action_and_value(obs, action)
    _, lp2, ent2, v2 = policy.get_action_and_value(junk, action)
    assert torch.allclose(lp1, lp2, atol=1e-5)
    assert torch.allclose(ent1, ent2, atol=1e-5)
    assert torch.allclose(v1, v2, atol=1e-5)


def test_obs_dim_contract():
    assert OBS_DIM == 624   # env v46: tau-scale lanes
    assert N_ACTION_HEADS == N_JOINTS == 29


def test_pool_encoder_removed():
    with pytest.raises(AssertionError):
        NotePolicy(encoder="pool")


def test_vec_env_auto_reset():
    venv = BeatVecEnv(4, colors="both", chart_len_s=5.0, seed=3)
    obs = venv.reset()
    assert obs.shape == (4, OBS_DIM)
    rng = np.random.default_rng(0)
    finished = 0
    for _ in range(400):
        obs, rew, done, infos = venv.step(
            rng.integers(0, ACTION_NVEC, size=(4, N_JOINTS)))
        finished += len(infos)
        for info in infos:
            assert info["n_notes"] > 0
            assert obs[info["env"], OFF_NOTE_MASK:OFF_NOTE_MASK + N_NOTE_SLOTS].sum() > 0  # fresh notes
    assert finished >= 4


def test_vec_matches_single():
    """VecBeatSim(1) must step identically to RawBeatSim."""
    import beat_arena_rs
    from beat_arena_env import ChartConfig, generate_chart
    chart_notes = generate_chart(ChartConfig(colors="both"),
                                 np.random.default_rng(21))
    from beat_arena_env.chart import note_tuple
    chart = [note_tuple(n) for n in chart_notes]
    single = beat_arena_rs.RawBeatSim(max_steps=630)
    vec = beat_arena_rs.VecBeatSim(1, max_steps=630)
    o1 = single.set_chart(chart)
    vec.set_chart(0, chart)
    np.testing.assert_array_equal(vec.obs(), o1)
    rng = np.random.default_rng(4)
    done1 = False
    while not done1:
        a = rng.integers(0, ACTION_NVEC, size=(1, N_JOINTS))
        o1, r1, done1, ev1 = single.step([int(x) for x in a[0]])
        ov, rv, dv, evv = vec.step(a)
        np.testing.assert_array_equal(ov, o1)
        assert rv[0] == np.float32(r1) and bool(dv[0]) == done1
        # vectorized events == the single sim's, env-tagged, order kept
        assert evv == [(0, *e) for e in ev1]
    assert vec.counters(0) == (single.good_cuts, single.bad_cuts,
                               single.misses, single.n_notes, single.score,
                               single.bombs_hit)


# --- hand-relative note geometry lanes (2026-09-14, --note-geo) ----------

def test_note_geo_normalizers_pinned():
    """policy._note_geo de-normalizes obs lanes with its own copies of the
    sim's normalizers; pin them to constants.py so an obs change cannot
    silently skew the geometry."""
    import policy as P
    from beat_arena_env import constants as C
    assert P._Z_SCALE == C.Z_SCALE and P._V_NORM == C.V_NORM
    if hasattr(C, "Y_NORM"):
        assert P._Y_NORM == C.Y_NORM
    if hasattr(C, "X_NORM"):
        assert P._X_NORM == C.X_NORM
    # hand block lanes inside the 32-lane saber token: q(8)+qd(7) then
    # [wrist pos(3), wrist vel(3), blade(3), meter, post_rem]
    assert (P._S_WPOS, P._S_WVEL, P._S_BLADE) == (15, 18, 21)
    assert P._SABER_L_IDX[15:26] == tuple(range(C.OFF_HAND_L, C.OFF_HAND_L + 11))
    assert P.NOTE_GEO_FEAT == 14


def test_note_geo_lanes_geometry():
    """A hand-crafted obs: left wrist at (0.2, -0.3, -0.5) head-relative
    x/y and world z, moving +x at 3 m/s with the blade along +y; note slot
    0 at head-relative (0.5, 0.15), world z 4.0, speed 16 m/s, cut
    direction (cos, sin) = (1, 0). The 14 lanes must be the metre-space
    offset/2, |dp|/2, z/speed, and the two direction dots."""
    import policy as P
    from beat_arena_env import constants as C
    obs = torch.zeros(1, OBS_DIM)
    hl = C.OFF_HAND_L
    obs[0, hl:hl + 3] = torch.tensor([0.2, -0.3 / 0.75, -0.5])
    obs[0, hl + 3:hl + 6] = torch.tensor([3.0 / C.V_NORM, 0.0, 0.0])
    obs[0, hl + 6:hl + 9] = torch.tensor([0.0, 1.0, 0.0])
    n0 = OFF_NOTES
    obs[0, n0:n0 + 4] = torch.tensor([0.5, 0.15 / 0.75, 4.0 / C.Z_SCALE,
                                      -16.0 / C.Z_SCALE])
    obs[0, n0 + 6], obs[0, n0 + 7] = 1.0, 0.0
    obs[0, OFF_NOTE_MASK] = 1.0
    pol = NotePolicy(note_geo=True)
    sabers, body, notes, mask = pol._split(obs)
    geo = P._note_geo(notes, sabers.reshape(1, 2, P.SABER_FEAT))
    assert geo.shape == (1, N_NOTE_SLOTS, P.NOTE_GEO_FEAT)
    g = geo[0, 0]
    dp = torch.tensor([0.5 - 0.2, 0.15 - (-0.3), 4.0 - (-0.5)]) * 0.5
    assert torch.allclose(g[0:3], dp, atol=1e-5)
    assert abs(g[3] - dp.norm()) < 1e-5
    assert abs(g[4] - 4.0 / 16.0) < 1e-5              # seconds to plane
    assert abs(g[5] - 3.0 / (3.0 + 0.05)) < 1e-5      # vhat.cutd, +x vs (1,0)
    assert abs(g[6]) < 1e-6                            # blade +y . (1,0) = 0
    # right hand sits at the origin with zero velocity: its dots are ~0
    assert abs(g[7 + 5]) < 1e-6
    # wall rows have no cut direction: dots masked to 0
    obs[0, n0 + 14] = 1.0
    sabers, body, notes, mask = pol._split(obs)
    g2 = P._note_geo(notes, sabers.reshape(1, 2, P.SABER_FEAT))[0, 0]
    assert g2[5] == 0 and g2[6] == 0 and torch.allclose(g2[0:3], dp, atol=1e-5)


def test_note_geo_forward_and_arch_marker():
    torch.manual_seed(0)
    pol = NotePolicy(note_geo=True)
    assert pol.arch["note_geo"] is True
    assert pol.note_mlp[0].in_features == NOTE_FEAT + 14
    obs = _obs_batch()
    logits, value = pol(obs)
    assert [l.shape for l in logits] == [(6, n) for n in HEAD_SIZES]
    assert torch.isfinite(value).all()
    # the default model is untouched
    assert NotePolicy().note_mlp[0].in_features == NOTE_FEAT
    assert NotePolicy().arch["note_geo"] is False
