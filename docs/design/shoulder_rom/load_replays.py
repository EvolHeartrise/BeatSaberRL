"""Load the scratch replays (v61/v64/v65) into per-arm canonical (left-frame)
shoulder coordinates. Cached to an npz next to the replays."""
import glob
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "..", "python"))
from beat_arena_env.replay_format import decode, read_raw  # noqa: E402

RDIR = os.environ.get("SROM_REPLAYS",
                      "/tmp/claude-1000/-home-evolheartrise-git-projects-BeatSaberRL/"
                      "dcb1a0a2-73f1-4c8b-9ec9-f6f9911b2ca5/scratchpad/twist")


def version_map():
    vm = {}
    for f in glob.glob(os.path.join(RDIR, "pairs*.json")):
        for k, v in json.load(open(f)).items():
            if k.startswith("v"):
                for rid in v:
                    vm[rid] = k
    return vm


def arm_block(q, om, side):
    """Canonical left-frame arm coords (T, 7): pitch, yaw, twist, elbow,
    roll, bend, dev (yaw/twist/roll/dev negated for the right arm), and
    the same for omega."""
    j0 = 3 if side == 0 else 10
    sgn = np.array([1, -1, -1, 1, -1, 1, -1], float) if side else np.ones(7)
    return q[:, 7 + j0:7 + j0 + 7] * sgn, om[:, j0:j0 + 7] * sgn


def load(cache=os.path.join(RDIR, "srom_arms.npz")):
    if os.path.exists(cache):
        z = np.load(cache, allow_pickle=True)
        return {k: z[k] for k in z.files}
    vm = version_map()
    Q, W, V, RID, SIDE = [], [], [], [], []
    for path in sorted(glob.glob(os.path.join(RDIR, "*.rpl.gz"))):
        rid = os.path.basename(path).split(".")[0]
        if rid not in vm:
            continue
        d = decode(read_raw(path))
        for side in (0, 1):
            a, w = arm_block(d["q"], d["omega"], side)
            Q.append(a)
            W.append(w)
            n = len(a)
            V += [vm[rid]] * n
            RID += [rid] * n
            SIDE += [side] * n
    out = dict(q=np.concatenate(Q), w=np.concatenate(W), ver=np.array(V),
               rid=np.array(RID), side=np.array(SIDE))
    np.savez(cache, **out)
    return out


if __name__ == "__main__":
    z = load()
    for v in sorted(set(z["ver"])):
        m = z["ver"] == v
        print(v, "arm-ticks", m.sum(), "replays", len(set(z["rid"][m])))
