"""The official ScoreSaber curve, lifted verbatim from
python/beat_arena_env/scoresaber.py (whose package __init__ needs gymnasium
and the Rust module, so it cannot be imported on a bare laptop)."""
import ast, os, re

_SRC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "python",
                    "beat_arena_env", "scoresaber.py")
_txt = open(_SRC).read()
PP_CURVE = ast.literal_eval(re.search(r"PP_CURVE = (\[.*?\n\])", _txt, re.S).group(1))
PP_DECAY = 0.965


def pp_multiplier(acc):
    if acc >= PP_CURVE[0][0]:
        return PP_CURVE[0][1]
    for (hi_a, hi_m), (lo_a, lo_m) in zip(PP_CURVE, PP_CURVE[1:]):
        if acc >= lo_a:
            t = (acc - lo_a) / (hi_a - lo_a)
            return lo_m + (hi_m - lo_m) * t
    return 0.0


def pp_value(acc, max_pp):
    return pp_multiplier(acc) * max_pp
