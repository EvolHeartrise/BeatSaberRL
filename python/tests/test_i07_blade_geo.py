"""I07 (env v49, 2026-09-30): give the policy the geometry the game scores.

Env side: 6 blade-TIP velocity obs lanes at 624..630 ([L xyz, R xyz] /
V_NORM) — the finite difference of wrist + (HANDLE_LEN + SABER_LEN)·blade
across the tick, i.e. the judge's tip_move / DT (lib.rs step_inner), zero
at reset / set_state and in the Flux game; Rust + CUDA.

Policy side (_note_geo): per hand × note, after the 14 v55 lanes, the
block [along, perp, cut, vt.d] per hand — the note's position along the
blade (dp·b − HANDLE_LEN), its distance to the blade line, its SIGNED
distance to the swept cut plane dp·norm(b × v_tip), and v_tip · cutdir.
Plus B12: t_plane = (z − z_off) / speed with z_off = obs lane 99.

Warm start: a pre-I07 checkpoint (obs 624, no "blade_geo" marker) loads
into the new policy with zero weights on every new input column — output
identical to the old net except for the t_plane fix.

The CUDA twins at the bottom skip without CUDA (run them on the pod).
"""

import math
import os
import subprocess

import numpy as np
import pytest
import torch

import beat_arena_rs
from beat_arena_env import constants as C
from beat_arena_env.chart import note_tuple
import policy as P
from policy import NotePolicy, load_policy
from tests._helpers import FAR_NOTE, cuda, resync, rust_state

DT = 1.0 / 30.0
TIP_L = C.HANDLE_LEN + C.SABER_LEN
# gears-5 ids: 1 = +full, 2 = -full — both arms + wrists swing, so the
# blades ROTATE and the tip outruns the wrist
SWING = [0] * C.N_JOINTS
for _j, _a in ((3, 1), (8, 2), (10, 1), (13, 2), (15, 1)):
    SWING[_j] = _a


def _rust_sim(flux=False):
    sim = beat_arena_rs.RawBeatSim(max_steps=690, gears=5, flux=flux)
    if flux:
        sim.set_flux_chart([])
    else:
        sim.set_chart([note_tuple(FAR_NOTE)])
    return sim


def _tips(sim):
    s = np.asarray(sim.sabers(), dtype=np.float64)       # (2, 6) w | b
    return s[:, :3] + TIP_L * s[:, 3:], s


def _tip_lanes(obs):
    return np.asarray(obs)[C.OFF_TIP_VEL:C.OFF_TIP_VEL + 6].reshape(2, 3)


# --- layout ------------------------------------------------------------------

def test_layout_contract():
    assert C.OFF_TIP_VEL == 624 and C.TIP_VEL_FEAT == 6 and C.OBS_DIM == 630
    assert C.OFF_TAU + C.TAU_FEAT == C.OFF_TIP_VEL   # appended, no shift
    assert P.OBS_DIM == C.OBS_DIM and P.SABER_FEAT == 47
    assert P._SABER_L_IDX[-3:] == (624, 625, 626)
    assert P._SABER_R_IDX[-3:] == (627, 628, 629)
    # every earlier lane keeps its index: the pre-I07 routing is a prefix
    assert P._SABER_L_IDX[:P.SABER_FEAT_PRE_I07] == tuple(
        i for i in P._SABER_L_IDX if i < 624)
    assert not set(P._BODY3_IDX) & set(range(624, 630))


# --- Rust env lanes ------------------------------------------------------------

def test_rust_tip_lanes_are_the_tip_finite_difference():
    sim = _rust_sim()
    assert np.all(_tip_lanes(sim.obs()) == 0.0), "zero at reset"
    gap = 0.0
    for _ in range(6):
        t0, _ = _tips(sim)
        obs, *_ = sim.step(SWING)
        t1, s1 = _tips(sim)
        want = (t1 - t0) / DT / C.V_NORM
        got = _tip_lanes(obs)
        np.testing.assert_allclose(got, want, rtol=1e-5, atol=1e-6)
        wv = np.asarray(obs)[[C.OFF_HAND_L + 3, C.OFF_HAND_R + 3]]
        gap = max(gap, float(np.abs(got[:, 0] - wv).max()) * C.V_NORM)
    assert gap > 0.5, f"tip vs wrist velocity gap only {gap} m/s"
    # set_joints (set_state) zeroes it, like wrist_vel
    q, v = rust_state(sim)
    sim.set_joints(q, v)
    assert np.all(_tip_lanes(sim.obs()) == 0.0)


def test_rust_flux_tip_lanes_stay_zero():
    sim = _rust_sim(flux=True)
    for _ in range(4):
        obs, *_ = sim.step(SWING)
        assert np.all(_tip_lanes(obs) == 0.0)


# --- _note_geo on hand-built obs ------------------------------------------------

def _geo(obs, z_asinh=8.0):
    pol = NotePolicy(note_geo=True)
    sabers, _body, notes, _mask = pol._split(obs)
    return P._note_geo(notes, sabers.reshape(obs.shape[0], 2, P.SABER_FEAT),
                       z_asinh, z_off=obs[:, P._OBS_HEAD_Z])


def _hand_obs(wrist, blade, tipv=(0.0, 0.0, 0.0), hand=0):
    """obs with hand `hand` at world wrist (x, y relative to a head at the
    origin/x 0, y HEAD0 — head lanes stay 0), blade dir, tip velocity m/s."""
    obs = torch.zeros(1, C.OBS_DIM)
    hb = C.OFF_HAND_L if hand == 0 else C.OFF_HAND_R
    obs[0, hb:hb + 3] = torch.tensor([wrist[0] / C.X_NORM,
                                      wrist[1] / C.Y_NORM, wrist[2]])
    obs[0, hb + 6:hb + 9] = torch.tensor(blade)
    t0 = C.OFF_TIP_VEL + 3 * hand
    obs[0, t0:t0 + 3] = torch.tensor(tipv) / C.V_NORM
    return obs


def _put_note(obs, slot, rel, cut=(0.0, -1.0), speed=16.0, wall=False,
              z_asinh=8.0):
    """note at head-relative x/y rel[0:2] (m), world z rel[2]."""
    b = C.OFF_NOTES + slot * C.NOTE_FEAT
    obs[0, b] = rel[0] / C.X_NORM
    obs[0, b + 1] = rel[1] / C.Y_NORM
    obs[0, b + 2] = math.asinh(rel[2] / z_asinh)
    obs[0, b + 3] = -speed / C.Z_SCALE
    obs[0, b + 6], obs[0, b + 7] = cut
    obs[0, b + 14] = float(wall)
    obs[0, C.OFF_NOTE_MASK + slot] = 1.0


def _i07(g, hand):
    """(along, perp, cut) decoded back to metres + vt.d (obs units)."""
    base = P.NOTE_GEO_FEAT_V55 + 4 * hand
    lanes = g[base:base + 4].double()
    return (2.0 * torch.sinh(lanes[0]).item(), 2.0 * torch.sinh(lanes[1]).item(),
            2.0 * torch.sinh(lanes[2]).item(), lanes[3].item())


def test_note_geo_shape_and_order():
    assert P.NOTE_GEO_FEAT == 22 and P.NOTE_GEO_FEAT_V55 == 14
    g = _geo(_hand_obs((0.1, -0.2, -0.4), (0.0, 1.0, 0.0)))
    assert g.shape == (1, C.N_NOTE_SLOTS, 22)


def test_note_on_the_blade_line():
    """A note ON the blade line: perpendicular distance 0 and the
    along-blade coordinate = its distance from the wrist minus the
    handle; off the line by 0.2 m sideways: perp 0.2, along unchanged."""
    w = (0.2, -0.3, -0.5)
    b = torch.tensor([0.6, 0.0, 0.8])            # unit, tilted forward
    obs = _hand_obs(w, b.tolist())
    for slot, s in enumerate((0.16, 0.66, 1.16, 2.0)):
        p = torch.tensor(w) + s * b
        _put_note(obs, slot, p.tolist())
    side = torch.tensor(w) + 0.7 * b + 0.2 * torch.tensor([0.0, 1.0, 0.0])
    _put_note(obs, 4, side.tolist())
    g = _geo(obs)[0]
    for slot, s in enumerate((0.16, 0.66, 1.16, 2.0)):
        along, perp, _cut, _vt = _i07(g[slot], 0)
        assert abs(perp) < 1e-5, (slot, perp)
        assert abs(along - (s - C.HANDLE_LEN)) < 1e-5, (slot, along)
    along, perp, _cut, _vt = _i07(g[4], 0)
    assert abs(perp - 0.2) < 1e-5 and abs(along - (0.7 - C.HANDLE_LEN)) < 1e-5


def test_cut_plane_sign_and_tip_dot():
    """Blade +y, tip moving +x at 5 m/s: b × v = (0, 1, 0) × (5, 0, 0) =
    (0, 0, -5) — the plane is x-y, the normal points -z. A note 0.2 m on
    the -z side reads cut = +0.2 (x 5/5.05, the soft normaliser), +z reads
    −0.2, in-plane 0. vt.d = v_tip · cutdir / V_NORM, masked on walls; a
    still blade (no plane) reads cut 0."""
    w = (0.0, 0.0, 0.0)
    obs = _hand_obs(w, (0.0, 1.0, 0.0), tipv=(5.0, 0.0, 0.0))
    _put_note(obs, 0, (0.3, 0.5, -0.2), cut=(1.0, 0.0))
    _put_note(obs, 1, (0.3, 0.5, 0.2), cut=(0.0, -1.0))
    _put_note(obs, 2, (0.3, 0.5, 0.0), cut=(1.0, 0.0), wall=True)
    g = _geo(obs)[0]
    soft = 5.0 / 5.05
    assert abs(_i07(g[0], 0)[2] - 0.2 * soft) < 1e-5
    assert abs(_i07(g[1], 0)[2] + 0.2 * soft) < 1e-5
    assert abs(_i07(g[2], 0)[2]) < 1e-6
    assert abs(_i07(g[0], 0)[3] - 5.0 / C.V_NORM) < 1e-6    # +x . (1, 0)
    assert abs(_i07(g[1], 0)[3]) < 1e-6                     # +x . (0, -1)
    assert _i07(g[2], 0)[3] == 0.0                          # wall: masked
    # the other hand (at the origin, no blade, no velocity): cut / vt 0
    assert _i07(g[0], 1)[2] == 0.0 and _i07(g[0], 1)[3] == 0.0
    still = _hand_obs(w, (0.0, 1.0, 0.0))
    _put_note(still, 0, (0.3, 0.5, -0.2))
    assert _i07(_geo(still)[0, 0], 0)[2] == 0.0


def test_cut_lane_is_the_rust_judge_plane():
    """End to end against the Rust sim: after a real swinging tick, a
    note at wrist + a·blade + c·n̂ (n̂ = norm(blade × tip_move), the
    judge's plane normal, from the sim's own FK) reads cut = +c through
    the obs → _note_geo path, and along = a − HANDLE_LEN, perp = |c|."""
    sim = _rust_sim()
    for _ in range(4):
        sim.step(SWING)
    w0b0 = np.asarray(sim.sabers(), dtype=np.float64)
    obs_np, *_ = sim.step(SWING)
    w1b1 = np.asarray(sim.sabers(), dtype=np.float64)
    obs = torch.as_tensor(np.asarray(obs_np), dtype=torch.float32)[None]
    head = np.array([obs_np[C.OFF_HEAD] * C.X_NORM,
                     obs_np[C.OFF_HEAD + 1] * C.Y_NORM + C.HEAD0[1]])
    cases = [(0.5, 0.15), (0.9, -0.1), (1.1, 0.25)]
    for hand in range(2):
        w0, b0 = w0b0[hand, :3], w0b0[hand, 3:]
        w1, b1 = w1b1[hand, :3], w1b1[hand, 3:]
        tm = w1 + TIP_L * b1 - w0 - TIP_L * b0
        pn = np.cross(b1, tm)
        nh = pn / np.linalg.norm(pn)
        vt = np.asarray(obs_np)[C.OFF_TIP_VEL + 3 * hand:
                                C.OFF_TIP_VEL + 3 * hand + 3] * C.V_NORM
        bv = np.linalg.norm(np.cross(b1, vt))
        soft = bv / (bv + 0.05)
        o = obs.clone()
        o[0, C.OFF_NOTES:C.OFF_NOTE_MASK + C.N_NOTE_SLOTS] = 0.0
        for slot, (a, c) in enumerate(cases):
            p = w1 + a * b1 + c * nh
            _put_note(o, slot, [p[0] - head[0], p[1] - head[1], p[2]])
        g = _geo(o)[0]
        for slot, (a, c) in enumerate(cases):
            along, perp, cut, _vt = _i07(g[slot], hand)
            assert abs(cut - c * soft) < 2e-4, (hand, slot, cut, c)
            assert abs(along - (a - C.HANDLE_LEN)) < 2e-4, (hand, slot, along)
            assert abs(perp - abs(c)) < 2e-4, (hand, slot, perp)


def test_t_plane_uses_the_head_z_offset():
    """B12: the note is judged at z_off = head_z − PLANE_Z0 (obs lane 99),
    so t_plane = (z − z_off) / speed, not z / speed."""
    obs = _hand_obs((0.0, 0.0, 0.0), (0.0, 1.0, 0.0))
    _put_note(obs, 0, (0.2, 0.1, 4.0), speed=16.0)
    obs[0, P._OBS_HEAD_Z] = 0.0
    g0 = _geo(obs)[0, 0]
    assert abs(g0[4].item() - 4.0 / 16.0 / 3.0) < 1e-6
    obs[0, P._OBS_HEAD_Z] = 0.4          # head 0.4 m forward of PLANE_Z0
    g1 = _geo(obs)[0, 0]
    assert abs(g1[4].item() - (4.0 - 0.4) / 16.0 / 3.0) < 1e-6
    assert abs(g1[7 + 4].item() - (4.0 - 0.4) / 16.0 / 3.0) < 1e-6
    # nothing else in the v55 block moved
    keep = [i for i in range(14) if i not in (4, 11)]
    assert torch.equal(g0[keep], g1[keep])


# --- the v55 lanes are unchanged (git-free reference) ----------------------------

def _note_geo_v55(notes, sabers, z_asinh):
    """policy._note_geo as of 0fbfa96 (pre-I07), verbatim minus flux."""
    z_m = torch.sinh(notes[..., 2]) * float(z_asinh)
    tx, ty = notes[..., 0], notes[..., 1]
    npos = torch.stack([tx * P._X_NORM, ty * P._Y_NORM, z_m], -1)
    cutd = torch.stack([notes[..., 6], notes[..., 7],
                        torch.zeros_like(notes[..., 6])], -1)
    not_wall = 1.0 - notes[..., 14:15]
    speed = (-notes[..., 3:4] * P._Z_SCALE).clamp_min(1e-3)
    tpl = (z_m[..., None] / speed).clamp(-0.5, 3.0)
    tpl = tpl / 3.0
    feats = []
    for hnd in range(2):
        w = sabers[:, hnd]
        wpos = torch.stack([w[:, 15] * P._X_NORM, w[:, 16] * P._Y_NORM,
                            w[:, 17]], -1)
        dp = (npos - wpos[:, None]) * 0.5
        dist = dp.norm(dim=-1, keepdim=True)
        dp = torch.cat([dp[..., :2], torch.asinh(dp[..., 2:3] * 2.0 / 8.0)], -1)
        dist = torch.asinh(dist * 2.0 / 8.0)
        v = w[:, 18:21] * P._V_NORM
        vhat = v / (v.norm(dim=-1, keepdim=True) + 0.05)
        vdot = (vhat[:, None] * cutd).sum(-1, keepdim=True) * not_wall
        bdot = (w[:, 21:24][:, None] * cutd).sum(-1, keepdim=True) * not_wall
        feats += [dp, dist, tpl, vdot, bdot]
    return torch.cat(feats, -1)


def _real_obs(n=8, seed=0):
    """Real v49 obs rows from the Rust env (swinging actions), with the
    tip lanes live."""
    from beat_arena_env import RustBeatEnv
    env = RustBeatEnv(seed=seed, chart_len_s=6.0, gears=5)
    obs, _ = env.reset()
    rows = []
    rng = np.random.default_rng(seed)
    for t in range(40):
        a = rng.integers(0, 5, size=C.N_JOINTS)
        obs, _r, done, _tr, _i = env.step(a)
        if done:
            obs, _ = env.reset()
        if t >= 40 - n:
            rows.append(obs)
    return torch.as_tensor(np.stack(rows), dtype=torch.float32)


def test_v55_lanes_are_bitwise_the_old_ones_at_zero_z_off():
    obs = _real_obs()
    obs[:, P._OBS_HEAD_Z] = 0.0
    pol = NotePolicy(note_geo=True)
    sabers, _b, notes, _m = pol._split(obs)
    sab = sabers.reshape(-1, 2, P.SABER_FEAT)
    new = P._note_geo(notes, sab, 8.0, z_off=obs[:, P._OBS_HEAD_Z])
    old = _note_geo_v55(notes, sab[..., :P.SABER_FEAT_PRE_I07], 8.0)
    assert torch.equal(new[..., :14], old)
    assert (_tip_lanes(obs[0].numpy()) != 0).any(), "tip lanes should be live"


# --- warm start: pre-I07 checkpoints ---------------------------------------------

_SMALL = dict(embed=32, trunk_dim=64, n_blocks=2, n_heads=2, n_gears=5,
              ffn_mult=2, note_geo=True)


def _pre_i07_ckpt(pol_new):
    """A pre-I07 checkpoint derived from a NEW policy by deleting the new
    input columns (the exact inverse of upgrade_pre_i07_state_dict)."""
    sd = {k: v.clone() for k, v in pol_new.state_dict().items()}
    w = sd["saber_mlp.0.weight"]
    sd["saber_mlp.0.weight"] = torch.cat([w[:, :44], w[:, 47:]], 1)
    sd["note_mlp.0.weight"] = sd["note_mlp.0.weight"][:, :32].clone()
    arch = dict(pol_new.arch, obs_dim=624)
    arch.pop("blade_geo")
    return {"state_dict": sd, "arch": arch}


def test_upgrade_zero_fills_and_marks():
    torch.manual_seed(0)
    ck = _pre_i07_ckpt(NotePolicy(**_SMALL))
    pol = load_policy(ck)
    assert pol.upgraded_from == "pre-i07"
    assert pol.arch["obs_dim"] == 630 and pol.arch["blade_geo"] == 1
    ws = pol.saber_mlp[0].weight
    assert ws.shape[1] == 49 and ws[:, 44:47].abs().max() == 0
    assert torch.equal(ws[:, :44], ck["state_dict"]["saber_mlp.0.weight"][:, :44])
    assert torch.equal(ws[:, 47:], ck["state_dict"]["saber_mlp.0.weight"][:, 44:])
    wn = pol.note_mlp[0].weight
    assert wn.shape[1] == 40 and wn[:, 32:].abs().max() == 0
    # a new-format checkpoint loads as-is
    again = load_policy({"state_dict": pol.state_dict(), "arch": pol.arch})
    assert again.upgraded_from is None
    # unknown marker / mismatched obs are refused
    with pytest.raises(SystemExit):
        load_policy({"state_dict": pol.state_dict(),
                     "arch": dict(pol.arch, blade_geo=2)})
    with pytest.raises(SystemExit):
        load_policy({"state_dict": pol.state_dict(),
                     "arch": dict(pol.arch, obs_dim=624)})
    bad = dict(ck["arch"], obs_dim=630)        # no marker but obs 630
    with pytest.raises(SystemExit):
        load_policy({"state_dict": ck["state_dict"], "arch": bad})


def test_upgraded_policy_is_blind_to_the_new_lanes():
    """Zero columns: the tip lanes (and every I07 geo lane derived from
    them) cannot move the upgraded net's output — bitwise."""
    torch.manual_seed(1)
    pol = load_policy(_pre_i07_ckpt(NotePolicy(**_SMALL)))
    obs = _real_obs()
    obs2 = obs.clone()
    obs2[:, C.OFF_TIP_VEL:] = torch.randn(obs.shape[0], 6) * 3.0
    with torch.no_grad():
        l1, v1 = pol.forward_stacked(obs)
        l2, v2 = pol.forward_stacked(obs2)
    assert torch.equal(l1, l2) and torch.equal(v1, v2)


def _old_policy_module(tmp_path):
    """The pre-I07 policy.py (commit 0fbfa96, on main) as a module; None
    when git or the commit is unavailable."""
    root = os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))))
    try:
        src = subprocess.run(
            ["git", "-C", root, "show", "0fbfa96:python/policy.py"],
            check=True, capture_output=True, text=True).stdout
    except (OSError, subprocess.CalledProcessError):
        return None
    path = tmp_path / "policy_pre_i07.py"
    path.write_text(src)
    import importlib.util
    spec = importlib.util.spec_from_file_location("policy_pre_i07", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_pre_i07_checkpoint_output_is_unchanged(tmp_path, monkeypatch):
    """The real thing: the pre-I07 NotePolicy (its own source), its
    checkpoint loaded through today's load_policy. On the same state the
    two nets agree to fp32 rounding (the widened GEMMs sum 11 extra
    exact zeros) once t_plane is computed the old way (z_off forced to
    0); with the B12 fix live they differ only through t_plane."""
    old_mod = _old_policy_module(tmp_path)
    if old_mod is None:
        pytest.skip("pre-I07 policy.py (git 0fbfa96) unavailable")
    assert old_mod.OBS_DIM == 624 and "blade_geo" not in old_mod.NotePolicy().arch
    torch.manual_seed(2)
    old = old_mod.NotePolicy(**_SMALL)
    ck = {"state_dict": old.state_dict(), "arch": dict(old.arch)}
    new = load_policy(ck)
    obs = _real_obs(n=12, seed=3)
    assert obs[:, P._OBS_HEAD_Z].abs().max() > 1e-4, "want a live z_off"
    with torch.no_grad():
        lo, vo = old.forward_stacked(obs[:, :624])
        ln, vn = new.forward_stacked(obs)
        # B12 on: the outputs move (notes are in the t_plane clip range)
        assert not torch.allclose(lo, ln, atol=1e-7, rtol=0)
        # t_plane the old way: identical up to fp32 rounding
        orig = P._note_geo

        def old_tpl(*a, **k):
            k["z_off"] = torch.zeros_like(k["z_off"])
            return orig(*a, **k)
        monkeypatch.setattr(P, "_note_geo", old_tpl)
        ln0, vn0 = new.forward_stacked(obs)
    torch.testing.assert_close(ln0, lo, atol=2e-6, rtol=1e-5)
    torch.testing.assert_close(vn0, vo, atol=2e-6, rtol=1e-5)


def test_resume_refuses_an_upgraded_checkpoint():
    """train_batch --resume must not pair an upgraded net with the old
    optimizer state: load_policy flags it (upgraded_from) and the trainer
    refuses; --init-from takes it."""
    src = open(os.path.join(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))), "train_batch.py")).read()
    i = src.index("policy = load_policy(resume_ckpt, game=args.game)")
    assert "policy.upgraded_from" in src[i:i + 300]


# --- CUDA twins (run on the pod) ---------------------------------------------------

def _cuda_env(n=16, flux=False, dtype=torch.float64):
    from beat_arena_env.cuda_env import CudaBeatVecEnv
    if flux:
        return CudaBeatVecEnv(n, game="flux", seed=3, chart_pool=8,
                              chart_len_s=4.0, device="cuda", dtype=dtype,
                              gears=5)
    return CudaBeatVecEnv(n, chart_len_s=8.0, seed=0, device="cuda",
                          dtype=dtype, gears=5)


def _swing_acts(n):
    return torch.tensor([SWING] * n, dtype=torch.int64, device="cuda")


@cuda
def test_cuda_tip_lanes_fd_reset_and_set_state():
    env = _cuda_env()
    obs = env.reset()
    assert obs.shape[1] == 630
    assert obs[:, C.OFF_TIP_VEL:].abs().max().item() == 0.0
    gap = 0.0
    for _ in range(4):
        t0 = (env.wrist + TIP_L * env.blade).clone()
        obs, *_ = env.step(_swing_acts(env.num_envs))
        t1 = env.wrist + TIP_L * env.blade
        want = ((t1 - t0) / DT / C.V_NORM).reshape(-1, 6).float()
        got = obs[:, C.OFF_TIP_VEL:C.OFF_TIP_VEL + 6]
        assert torch.allclose(got, want, atol=1e-5), (got - want).abs().max()
        assert torch.allclose(env.tip_vel.reshape(-1, 6).float() / C.V_NORM,
                              got, atol=1e-6)
        wv = obs[:, [C.OFF_HAND_L + 3, C.OFF_HAND_R + 3]]
        gap = max(gap, (got[:, [0, 3]] - wv).abs().max().item() * C.V_NORM)
    assert gap > 0.5
    q36 = [0.0, 0.94, -0.65] + [1.0, 0.0, 0.0, 0.0] + [0.0] * 29
    env.set_state(q36, [0.0] * 35)
    assert env._obs[:, C.OFF_TIP_VEL:].abs().max().item() == 0.0
    assert env.tip_vel.abs().max().item() == 0.0


@cuda
def test_cuda_tip_lanes_match_rust_lockstep():
    """Per-tick Rust-vs-CUDA parity of the 6 new lanes (f64, lockstep
    resync — tests/_helpers.resync carries tip_vel across set_state)."""
    from beat_arena_env.cuda_env import CudaBeatVecEnv
    from beat_arena_env.chart import ChartConfig, generate_chart
    n = 6
    charts = [generate_chart(ChartConfig(colors="both", chart_len_s=4.0),
                             np.random.default_rng(10 + s)) for s in range(n)]
    env = CudaBeatVecEnv(n, charts=charts, max_steps=400, gears=5,
                         device="cuda", dtype=torch.float64,
                         auto_reset=False, want_contacts=True)
    env.reset()
    sims = []
    for ch in charts:
        s = beat_arena_rs.RawBeatSim(max_steps=400, gears=5)
        s.set_chart([note_tuple(x) for x in ch])
        sims.append(s)
    rng = np.random.default_rng(7)
    worst, live, flips, checked = 0.0, 0.0, 0, 0
    for t in range(60):
        # random ARM gears only (legs coast: no falls inside 60 ticks)
        acts = np.zeros((n, C.N_JOINTS), dtype=np.int64)
        acts[:, 3:17] = rng.integers(0, 5, size=(n, 14))
        tobs, _r, tdone, _i = env.step(torch.as_tensor(acts, device="cuda"))
        rdone = []
        for e, sim in enumerate(sims):
            robs, _rr, rd, _ev = sim.step([int(x) for x in acts[e]])
            rdone.append(rd)
            d = np.abs(_tip_lanes(robs)
                       - tobs[e, C.OFF_TIP_VEL:].cpu().numpy().reshape(2, 3))
            worst = max(worst, float(d.max()))
            checked += 1
            # the f64 tier's knife edges (a 1-ulp seed flipping a limit /
            # contact row mid-tick, test_v10_parity_cuda) are tolerated
            # as rare, bounded flips; everything else is the tight band
            if d.max() >= 1e-5:
                flips += 1
                assert d.max() < 1.0, f"t{t} env{e} tip lanes {d.max()}"
        live = max(live, float(tobs[:, C.OFF_TIP_VEL:].abs().max()))
        if any(rdone) or bool(tdone.any()):
            break
        q36s, v35s = zip(*(rust_state(s) for s in sims))
        resync(env, list(q36s), list(v35s))
    print(f"tip-lane lockstep worst diff {worst:.3e} (max |lane| {live:.3f})")
    assert live > 0.1, "the arms never swung"
    assert flips <= max(1, checked // 30), f"{flips}/{checked} flips"


@cuda
def test_cuda_flux_tip_lanes_stay_zero():
    env = _cuda_env(n=4, flux=True, dtype=torch.float32)
    env.reset()
    for _ in range(4):
        obs, *_ = env.step(_swing_acts(4))
        assert obs[:, C.OFF_TIP_VEL:].abs().max().item() == 0.0


@cuda
def test_cuda_fp16_forward_matches_fp32_with_i07_lanes():
    """The widened encoders on the fp16 stream (note_mlp K = 40, the
    8-aligned Lt chain; saber K = 49, the odd-K fallback) agree with the
    fp32 record path at the fp16 last-rounding band."""
    torch.manual_seed(0)
    pol = NotePolicy(embed=64, trunk_dim=128, n_blocks=2, n_heads=4,
                     n_gears=5, note_geo=True).cuda()
    obs = _real_obs().cuda()
    with torch.no_grad():
        l32, v32 = pol.forward_stacked(obs)
        pol.fp16 = True
        l16, v16 = pol.forward_stacked(obs)
    assert torch.allclose(l16.float(), l32, atol=5e-2), (l16 - l32).abs().max()
    assert torch.allclose(v16.float(), v32, atol=5e-2)
