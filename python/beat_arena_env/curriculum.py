"""Map-ladder curriculum over ScoreSaber-ranked maps (SPEC §Curriculum).

The ladder starts with one random ~`start_star` map. Training samples
every episode's chart from the active ladder — the newest (hardest,
un-FC'd) map gets `new_map_weight` of episodes, the rest is uniform over
the older maps so they can't regress. When a greedy (argmax) rollout
full-combos the newest map (zero bad cuts, zero misses — checked by the
trainer), `promote()` marks it FC'd and fetches a new random map
`star_step` harder. State persists as JSON so ladders survive restarts
(`--curriculum-state` in train_batch.py).
"""

from __future__ import annotations

import json
import os
from collections import deque

from .beatsaver import load_beatmap
from . import scoresaber

STATS_WINDOW = 20  # rolling per-stage episode window for the summary line


class Curriculum:
    def __init__(self, map_dir, state_path, start_star=1.0, star_step=0.2,
                 star_band=0.1, speed_scale=0.3, new_map_weight=0.5,
                 max_song_s=400.0, acquire_rng=None):
        self.map_dir = map_dir
        self.state_path = state_path
        self.start_star = start_star
        self.star_step = star_step
        self.star_band = star_band
        self.speed_scale = speed_scale
        self.new_map_weight = new_map_weight
        self.max_song_s = max_song_s
        self._acq_rng = acquire_rng
        os.makedirs(map_dir, exist_ok=True)

        self.stages = []   # meta dicts (scoresaber.acquire_* + fc/target_star)
        self.charts = []   # parsed list[Note] per stage
        self.stats = []    # per-stage deque of (good, bad, miss, n) episodes
        if os.path.exists(state_path):
            self._load()
        else:
            self._add_map(start_star)

        # env-index bookkeeping for per-stage episode stats: sample() runs at
        # auto-reset time, i.e. right after env i's episode ended, so the
        # outgoing stage is snapshotted into completed_stage before reassign.
        self.env_stage = {}
        self.completed_stage = {}

    # -- ladder ---------------------------------------------------------------
    @property
    def newest(self):
        return len(self.stages) - 1

    def _add_map(self, target_star):
        exclude = {(s["hash"], s["difficulty"]) for s in self.stages}
        meta = scoresaber.acquire_random_map(
            target_star, self.star_band, self.map_dir, self._acq_rng,
            speed_scale=self.speed_scale, max_len_s=self.max_song_s,
            exclude=exclude)
        meta.update(fc=False, target_star=round(target_star, 2))
        self.stages.append(meta)
        self.charts.append(self._load_chart(meta))
        self.stats.append(deque(maxlen=STATS_WINDOW))
        self._save()
        print(f"[curriculum] stage {len(self.stages)}: {meta['song']!r} "
              f"[{meta['difficulty']}] {meta['stars']:.2f}★  "
              f"{meta['n_notes']} notes / {meta['duration_s']:.0f}s "
              f"({meta['nps']:.2f} nps)", flush=True)

    def promote(self):
        """Newest map FC'd -> lock it in, fetch the next rung (+star_step)."""
        stage = self.stages[self.newest]
        stage["fc"] = True
        self._add_map(stage["target_star"] + self.star_step)

    def _load_chart(self, meta):
        path = os.path.join(self.map_dir, meta["zip"])
        if not os.path.exists(path):      # cache purged: re-fetch by hash
            scoresaber.download_zip(meta["hash"], self.map_dir)
        notes, _ = load_beatmap(path, meta["difficulty"],
                                speed_scale=self.speed_scale, first_note_s=2.0)
        return notes

    # -- training hooks -------------------------------------------------------
    def sample(self, i, rng):
        """chart_sampler for BeatVecEnv: newest map with new_map_weight
        probability, else uniform over the already-FC'd rungs."""
        if i in self.env_stage:
            self.completed_stage[i] = self.env_stage[i]
        k = len(self.stages)
        if k == 1:
            s = 0
        elif rng.random() < self.new_map_weight:
            s = k - 1
        else:
            s = int(rng.integers(0, k - 1))
        self.env_stage[i] = s
        return self.charts[s]

    def note_result(self, info):
        """Feed a finished-episode info dict (from BeatVecEnv) into the
        per-stage rolling stats."""
        s = self.completed_stage.get(info["env"])
        if s is not None:
            self.stats[s].append((info["good_cuts"], info["bad_cuts"],
                                  info["misses"], info["n_notes"]))

    def summary(self):
        lines = []
        for s, (meta, st) in enumerate(zip(self.stages, self.stats)):
            tag = "FC" if meta["fc"] else "<- current"
            if st:
                g = sum(e[0] for e in st)
                n = max(1, sum(e[3] for e in st))
                acc = f"{100 * g / n:5.1f}% good (last {len(st)} eps)"
            else:
                acc = "no episodes yet"
            lines.append(f"  [{s}] {meta['stars']:.2f}★ {meta['song'][:32]!r} "
                         f"[{meta['difficulty']}]  {acc}  {tag}")
        return "\n".join(lines)

    # -- persistence ----------------------------------------------------------
    def _save(self):
        state = {"version": 1, "start_star": self.start_star,
                 "star_step": self.star_step, "speed_scale": self.speed_scale,
                 "stages": self.stages}
        tmp = self.state_path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(state, f, indent=1)
        os.replace(tmp, self.state_path)

    def _load(self):
        with open(self.state_path) as f:
            state = json.load(f)
        assert state.get("version") == 1, f"unknown curriculum state {state}"
        if state["speed_scale"] != self.speed_scale:
            raise ValueError(
                f"state was built with speed_scale={state['speed_scale']}, "
                f"got --speed-scale {self.speed_scale}")
        self.stages = state["stages"]
        self.charts = [self._load_chart(m) for m in self.stages]
        self.stats = [deque(maxlen=STATS_WINDOW) for _ in self.stages]
        print(f"[curriculum] resumed {len(self.stages)}-stage ladder from "
              f"{self.state_path}", flush=True)
        # a crash between FC-marking and add_map can leave the newest FC'd:
        if self.stages and self.stages[self.newest]["fc"]:
            self._add_map(self.stages[self.newest]["target_star"]
                          + self.star_step)
