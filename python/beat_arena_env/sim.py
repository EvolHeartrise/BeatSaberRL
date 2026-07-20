"""Beat Arena sim core (SPEC.md §Sim rules). Pure Python/NumPy, no I/O.

Two point-mass sabers on the cut plane, a schedule of notes, per-tick
discrete acceleration control. All constants here are the SPEC contract.
"""

from __future__ import annotations

import json

import numpy as np

from .chart import ChartConfig, Note, generate_chart

DT = 1.0 / 30.0
ACCEL = 60.0          # u/s^2
DRAG = 0.02           # per-tick multiplicative velocity decay
V_MAX = 12.0
V_MIN_CUT = 6.0       # min speed for a good cut
CUT_RADIUS = 0.55
CUT_WINDOW = 0.15     # seconds either side of hit_time
COS_TOL = 0.5         # cos(60 deg): max angle between swing and required dir
POS_LO = np.array([-1.0, -1.0])
POS_HI = np.array([5.0, 4.0])

REWARD_GOOD = 1.0
REWARD_BAD = -0.25
REWARD_MISS = -0.5

# potential-based shaping (Ng et al. 1999): reward += GAMMA*phi(s') - phi(s)
# with phi = -coef * normalized distance from each saber to its next
# matching note. Provably does not change the optimal policy; exists purely
# to give exploration a dense gradient (raw cut rewards are too sparse for
# random motion to ever discover).
SHAPE_GAMMA = 0.99
SHAPE_DIST_CLIP = 4.0

N_SABERS = 2
LEFT, RIGHT = 0, 1

# action k in 1..8 accelerates at (k-1)*45 deg; 0 = coast
_DIRS = np.array(
    [[np.cos(k * np.pi / 4), np.sin(k * np.pi / 4)] for k in range(8)]
)


def dir_vec(d: int) -> np.ndarray:
    """Unit vector of cut direction d (d*45 deg, 0 = +x, CCW, y up)."""
    return _DIRS[d]


def note_center(n: Note) -> np.ndarray:
    return np.array([n.col + 0.5, n.row + 0.5])


def _seg_point_dist(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    """Distance from point c to segment a->b."""
    ab = b - a
    denom = float(ab @ ab)
    if denom < 1e-12:
        return float(np.linalg.norm(c - a))
    t = float(np.clip((c - a) @ ab / denom, 0.0, 1.0))
    return float(np.linalg.norm(a + t * ab - c))


class BeatSim:
    """One episode: a chart plus saber state, stepped one tick at a time."""

    def __init__(self, cfg: ChartConfig, seed: int):
        self.cfg = cfg
        self.rng = np.random.default_rng(seed)
        self.chart: list[Note] = generate_chart(cfg, self.rng)
        self.max_steps = int(cfg.chart_len_s * 30) + 30
        self.tick = 0
        self.time = 0.0
        # sabers start at rest, centered in their half of the grid
        self.pos = np.array([[1.0, 1.5], [3.0, 1.5]], dtype=np.float64)
        self.vel = np.zeros((N_SABERS, 2), dtype=np.float64)
        # note status: 0 = pending, 1 = good cut, 2 = bad cut, 3 = missed
        self.status = np.zeros(len(self.chart), dtype=np.int64)
        self.good_cuts = 0
        self.bad_cuts = 0
        self.misses = 0
        self._record_path = None
        self._record_lines = None
        self._phi = self._potential()

    def _potential(self) -> float:
        """phi(s): -shaping_coef * sum over hands of clipped, normalized
        distance to that hand's earliest pending note (0 if it has none)."""
        if self.cfg.shaping_coef == 0.0:
            return 0.0
        phi = 0.0
        seen = 0
        for i in self.pending_notes():
            n = self.chart[i]
            if seen & (1 << n.color):
                continue
            seen |= 1 << n.color
            d = float(np.linalg.norm(self.pos[n.color] - note_center(n)))
            phi -= self.cfg.shaping_coef * min(d, SHAPE_DIST_CLIP) / SHAPE_DIST_CLIP
            if seen == 3:
                break
        return phi

    # -- recording -----------------------------------------------------------
    def set_record(self, path: str):
        self._record_path = path
        header = {
            "header": {
                "version": 1,
                "config": {
                    "note_rate": self.cfg.note_rate,
                    "chart_len_s": self.cfg.chart_len_s,
                    "colors": self.cfg.colors,
                    "dirs": list(self.cfg.dirs),
                    "min_hand_gap_s": self.cfg.min_hand_gap_s,
                },
                "chart": [
                    {"t": n.hit_time, "col": n.col, "row": n.row,
                     "color": n.color, "dir": n.dir}
                    for n in self.chart
                ],
            }
        }
        self._record_lines = [json.dumps(header)]

    def _flush_record(self):
        if self._record_path is not None:
            with open(self._record_path, "w") as f:
                f.write("\n".join(self._record_lines) + "\n")

    # -- stepping ------------------------------------------------------------
    def pending_notes(self):
        """Indices of pending notes in hit-time order (chart is sorted)."""
        return np.flatnonzero(self.status == 0)

    def step(self, action_left: int, action_right: int):
        """Advance one tick. Returns (reward, done, events)."""
        assert 0 <= action_left <= 8 and 0 <= action_right <= 8
        prev_pos = self.pos.copy()

        # integrate sabers: accel -> drag -> speed cap -> move -> clamp
        for s, a in ((LEFT, action_left), (RIGHT, action_right)):
            if a > 0:
                self.vel[s] += ACCEL * DT * _DIRS[a - 1]
            self.vel[s] *= 1.0 - DRAG
            speed = np.linalg.norm(self.vel[s])
            if speed > V_MAX:
                self.vel[s] *= V_MAX / speed
            self.pos[s] += self.vel[s] * DT
            for ax in range(2):
                if self.pos[s, ax] < POS_LO[ax] or self.pos[s, ax] > POS_HI[ax]:
                    self.pos[s, ax] = np.clip(self.pos[s, ax], POS_LO[ax], POS_HI[ax])
                    self.vel[s, ax] = 0.0

        self.time = (self.tick + 1) * DT
        reward = 0.0
        events = []

        # evaluate active notes (window straddles hit_time)
        for i in self.pending_notes():
            n = self.chart[i]
            if abs(self.time - n.hit_time) > CUT_WINDOW:
                if self.time - n.hit_time > CUT_WINDOW:
                    self.status[i] = 3
                    self.misses += 1
                    reward += REWARD_MISS
                    events.append({"miss": {"note": int(i)}})
                continue
            center = note_center(n)
            # matching-color saber evaluated first (SPEC)
            for s in (n.color, 1 - n.color):
                crossed = (
                    _seg_point_dist(prev_pos[s], self.pos[s], center) <= CUT_RADIUS
                )
                if not crossed:
                    continue
                if s != n.color:
                    good, reason = False, "wrong_hand"
                else:
                    speed = float(np.linalg.norm(self.vel[s]))
                    if speed < V_MIN_CUT:
                        good, reason = False, "slow"
                    elif (self.vel[s] @ dir_vec(n.dir)) / max(speed, 1e-9) < COS_TOL:
                        good, reason = False, "angle"
                    else:
                        good, reason = True, "ok"
                self.status[i] = 1 if good else 2
                if good:
                    self.good_cuts += 1
                    reward += REWARD_GOOD
                else:
                    self.bad_cuts += 1
                    reward += REWARD_BAD
                events.append(
                    {"cut": {"note": int(i), "good": good, "reason": reason}}
                )
                break

        self.tick += 1
        last_hit = self.chart[-1].hit_time if self.chart else 0.0
        done = (
            len(self.pending_notes()) == 0
            and (len(self.chart) == 0 or self.time > last_hit)
        ) or self.time > last_hit + 0.5 or self.tick >= self.max_steps

        # potential-based shaping; phi(terminal) = 0 by convention
        new_phi = 0.0 if done else self._potential()
        reward += SHAPE_GAMMA * new_phi - self._phi
        self._phi = new_phi

        if self._record_lines is not None:
            self._record_lines.append(json.dumps({
                "t": self.tick - 1,
                "time": round(self.time - DT, 5),
                "sabers": [[round(float(x), 4) for x in
                            (*self.pos[s], *self.vel[s])] for s in range(2)],
                "action": [int(action_left), int(action_right)],
                "reward": round(float(reward), 4),
                "events": events,
            }))
            if done:
                self._flush_record()

        return reward, done, events
