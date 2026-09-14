"""env v42 "hilt grip" (2026-09-14, user-directed): the hand holds the
0.16 m HANDLE, so the cutting blade spans [HANDLE_LEN, HANDLE_LEN +
SABER_LEN] along the blade dir from the wrist; the handle does not cut.

Fixture: the scenario family's FLAT pose (right blade horizontal along
+-x at z ~ -0.195, behind the note plane). A note whose good box
(0.8 m wide) overlaps ONLY the handle segment [wrist, wrist + 0.16*dir]
must produce no cut event as it drifts through the blade line; the same
note moved 0.05 m further along the blade dir overlaps the blade and is
cut (a near-static blade -> "slow" bad cut, code 1 — still a cut event).
The note centres are placed from the sim's OWN sabers() so the test is
pinned to the live FK, not to a literal.
"""
import numpy as np
import pytest
import torch

from beat_arena_env.chart import Note, note_tuple
from beat_arena_env.constants import GOOD_HALF_X, HANDLE_LEN, N_JOINTS

from beat_arena_rs import RawBeatSim

from test_env import _set_flat, _coast, _codes, DT

TICKS = 8
COAST = [0] * N_JOINTS


def _flat_sim():
    sim = RawBeatSim(max_steps=690)
    sim.set_chart([note_tuple(Note(hit_time=50.0, col=2, row=1, x=0.3,
                                   y=1.4, color=1, angle=4.712,
                                   speed=4.0))])
    _set_flat(sim, 1, 0.0)
    return sim


def _note_along_blade(sim, along):
    """A right-hand note centred `along` metres from the wrist along the
    live blade dir (in the horizontal plane), arriving at the plane one
    tick in: its box crosses the flat blade's z within a few ticks."""
    w = np.asarray(sim.sabers()[1][:3])
    d = np.asarray(sim.sabers()[1][3:])
    c = w + along * d
    return Note(hit_time=DT, col=2, row=1, x=float(c[0]), y=float(w[1]),
                color=1, angle=4.712, speed=4.0)


def _run(sim, ticks=TICKS):
    events = []
    for _ in range(ticks):
        ev, _ = _coast(sim)
        events += _codes(ev)
    return events


# note box (half-width GOOD_HALF_X) touching the handle only: its far edge
# ends 4 cm short of the blade base
HANDLE_ONLY = HANDLE_LEN - GOOD_HALF_X - 0.04
BLADE_EDGE = HANDLE_LEN - GOOD_HALF_X + 0.05


def test_handle_does_not_cut_but_blade_does():
    probe = _flat_sim()
    n_handle = _note_along_blade(probe, HANDLE_ONLY)
    n_blade = _note_along_blade(probe, BLADE_EDGE)

    sim = RawBeatSim(max_steps=690)
    sim.set_chart([note_tuple(n_handle)])
    _set_flat(sim, 1, 0.0)
    assert [c for c in _run(sim) if c[1] in (0, 1, 2)] == [], \
        "a note within the handle segment must not be cut"

    sim = RawBeatSim(max_steps=690)
    sim.set_chart([note_tuple(n_blade)])
    _set_flat(sim, 1, 0.0)
    cuts = [c for c in _run(sim) if c[1] in (0, 1, 2)]
    assert cuts and cuts[0][0] == 0, \
        f"a note 5 cm into the blade segment must be cut, got {cuts}"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_cuda_handle_does_not_cut_but_blade_does():
    from beat_arena_env.cuda_env import CudaBeatVecEnv
    probe = _flat_sim()
    q36 = list(probe.qpos())
    v35 = list(probe.twist()) + list(probe.omega())
    n_handle = _note_along_blade(probe, HANDLE_ONLY)
    n_blade = _note_along_blade(probe, BLADE_EDGE)
    env = CudaBeatVecEnv(2, charts=[[n_handle], [n_blade]], gears=5,
                         max_steps=690, foot_contact=False,
                         device="cuda", dtype=torch.float64,
                         auto_reset=False)
    env.reset()
    env.set_state(torch.tensor([q36] * 2, dtype=torch.float64),
                  torch.tensor([v35] * 2, dtype=torch.float64))
    act = torch.zeros((2, N_JOINTS), dtype=torch.long)
    for _ in range(TICKS):
        env.step(act)
    # per-env cut counters (good + bad), the same readout the parity gate
    # uses against sim.good_cuts / bad_cuts
    cut = [int(env.good[e]) + int(env.bad[e]) > 0 for e in range(2)]
    assert cut == [False, True], f"CUDA handle/blade cut pattern {cut}"
