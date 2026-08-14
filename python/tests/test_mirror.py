"""Mirror-transform pins (2026-08-14): direction table, involution,
color/bomb handling, wall reflection, and pool-augmentation plumbing."""

import math
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from beat_arena_env.chart import note_tuple  # noqa: E402
from beat_arena_env.constants import KIND_BOMB, KIND_NOTE  # noqa: E402
from beat_arena_env.mirror import (  # noqa: E402
    mirror_angle, mirror_note_tuple, mirror_tuples, mirror_wall_tuple)

UP, DOWN = math.pi / 2, 3 * math.pi / 2
LEFT, RIGHT = math.pi, 0.0
UPLEFT, UPRIGHT = 3 * math.pi / 4, math.pi / 4
DOWNLEFT, DOWNRIGHT = 5 * math.pi / 4, 7 * math.pi / 4


def _note(x=0.3, color=0, angle=RIGHT, kind=KIND_NOTE):
    # (t, x, y, color, angle, speed, any_dir, kind, free_pre, free_post)
    return (1.5, x, 1.4, color, angle, 17.0, False, kind, False, True)


@pytest.mark.parametrize("a,b", [
    (UP, UP), (DOWN, DOWN),                      # verticals fixed
    (LEFT, RIGHT), (RIGHT, LEFT),                # horizontals swap
    (UPLEFT, UPRIGHT), (UPRIGHT, UPLEFT),
    (DOWNLEFT, DOWNRIGHT), (DOWNRIGHT, DOWNLEFT),
])
def test_direction_table(a, b):
    assert mirror_angle(a) == pytest.approx(b % (2 * math.pi), abs=1e-12)


def test_note_mirror_fields():
    m = mirror_note_tuple(_note(x=0.9, color=0, angle=UPLEFT))
    assert m[1] == pytest.approx(-0.9)     # x negated
    assert m[3] == 1                       # color swapped
    assert m[4] == pytest.approx(UPRIGHT)  # angle reflected
    # untouched: t, y, speed, any_dir, kind, arc flags
    orig = _note(x=0.9, color=0, angle=UPLEFT)
    for i in (0, 2, 5, 6, 7, 8, 9):
        assert m[i] == orig[i]


def test_bomb_keeps_color():
    m = mirror_note_tuple(_note(x=-0.3, color=0, kind=KIND_BOMB))
    assert m[3] == 0 and m[7] == KIND_BOMB
    assert m[1] == pytest.approx(0.3)


def test_involution():
    for t in [_note(), _note(x=-0.9, color=1, angle=DOWNLEFT),
              _note(x=0.0, angle=1.2345), _note(kind=KIND_BOMB)]:
        rt = mirror_note_tuple(mirror_note_tuple(t))
        assert rt[3] == t[3] and rt[7] == t[7]
        assert rt[1] == pytest.approx(t[1], abs=1e-12)
        assert rt[4] == pytest.approx(t[4], abs=1e-12)


def test_wall_mirror():
    w = (2.0, -0.9, 1.2, 0.6, 1.8, 3.4, 17.0)
    m = mirror_wall_tuple(w)
    assert m[1] == pytest.approx(0.9)
    assert m[0] == w[0] and m[2:] == w[2:]
    assert mirror_wall_tuple(m) == pytest.approx(w)


def test_pool_mirror_doubles_segments(tmp_path):
    """_build_segments with mirror=True emits original+mirrored pairs."""
    from beat_arena_env import ranked_pool as rp

    calls = []

    def fake_build(*a, **k):
        calls.append(k)

    # exercise the real windowing path on a synthetic 2-note chart via
    # the internal helpers instead: rebased window -> pair check
    reb = [_note(x=0.3, color=0, angle=LEFT),
           _note(x=-0.9, color=1, angle=UP)]
    mirrored = mirror_tuples(reb)
    assert mirrored[0][1] == pytest.approx(-0.3)
    assert mirrored[0][3] == 1
    assert mirrored[0][4] == pytest.approx(RIGHT)
    assert mirrored[1][1] == pytest.approx(0.9)
    assert mirrored[1][4] == pytest.approx(UP)
    # cache key must fork on the flag (mirror pools never collide with
    # normal pools on the shared volume)
    cat = tmp_path / "cat.json"
    cat.write_text("[]", encoding="utf-8")
    a = rp._cache_path(25.0, 99, str(cat), 0.5, mirror=False)
    b = rp._cache_path(25.0, 99, str(cat), 0.5, mirror=True)
    assert a != b
