"""Training-acc components (2026-10-04, user): the CUDA env's acc_part
accumulators (per env x hand x [pre, pre max, post, post max, precision,
precision max]) against its own acc_pts flow. Needs CUDA."""
import math

import numpy as np
import pytest
import torch

from beat_arena_env.constants import N_JOINTS


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_acc_parts_track_the_acc_flow():
    from beat_arena_env.cuda_env import CudaBeatVecEnv
    env = CudaBeatVecEnv(256, chart_len_s=6.0, seed=5, device="cuda",
                         dtype=torch.float64, gears=5)
    env.reset()
    rng = np.random.default_rng(0)
    for _ in range(240):
        a = torch.as_tensor(rng.integers(0, 5, (256, N_JOINTS)),
                            device="cuda")
        env.step(a)
    part = env.acc_part.sum(dim=(0, 1)).tolist()
    pts_l, pts_r, cuts_l, cuts_r = env.pop_acc()
    pre, post, prec = env.pop_acc_parts()
    assert cuts_l + cuts_r > 0                       # notes were cut
    # each component is points / max, within [0, 1]
    for v in (pre, post, prec):
        assert 0.0 <= v <= 1.0, (pre, post, prec)
    # maxima: 70 / 15 per scored note cut (same count), 30 per post
    assert part[1] / 70.0 == pytest.approx(part[5] / 15.0)
    assert part[3] % 30.0 == pytest.approx(0.0, abs=1e-6)
    # generated charts have no chains (chain links' flat points are in
    # acc_pts only), so here the three components ARE the acc flow
    comp = part[0] + part[2] + part[4]
    assert comp == pytest.approx(pts_l + pts_r, rel=1e-9)
    # popped + zeroed
    assert all(math.isnan(v) for v in env.pop_acc_parts())
