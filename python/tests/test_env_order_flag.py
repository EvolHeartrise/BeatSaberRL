"""--env-order / --no-env-order resolution (2026-09-02, Tier 0.1 of
PERF_AUDIT_2026-09-02_static.md §B 1).

The R12 dynamics-key permutation shipped 2026-08-21 behind a bare
BEAT_ENV_ORDER env var, default OFF, that no launcher ever set — so a
measured +5.4% (on the previous pod) sat uncollected for 12 days and no
run log said which ordering it used. The flag is now the interface and
defaults ON; the env var stays in the chain so the standalone contract
(and a pod that deliberately exports BEAT_ENV_ORDER=0) still works.

Precedence pinned here: explicit flag > shell BEAT_ENV_ORDER > ON.
"""

from train_batch import parse_args, resolve_env_order


# ---------------------------------------------------------- precedence

def test_default_is_on_when_nothing_is_set():
    assert resolve_env_order(None, {}) is True


def test_shell_var_wins_when_no_flag_is_given():
    assert resolve_env_order(None, {"BEAT_ENV_ORDER": "0"}) is False
    assert resolve_env_order(None, {"BEAT_ENV_ORDER": "1"}) is True


def test_anything_other_than_1_reads_as_off():
    # cuda_env.py's own reader is `== "1"`; keep the two in lockstep
    for v in ("0", "", "true", "yes", "2"):
        assert resolve_env_order(None, {"BEAT_ENV_ORDER": v}) is (v == "1")


def test_explicit_flag_beats_the_shell_var_both_ways():
    assert resolve_env_order(True, {"BEAT_ENV_ORDER": "0"}) is True
    assert resolve_env_order(False, {"BEAT_ENV_ORDER": "1"}) is False


# --------------------------------------------------------------- argparse

def _env_order(argv):
    return parse_args_with(argv).env_order


def parse_args_with(argv):
    import sys
    old = sys.argv
    try:
        sys.argv = ["train_batch.py"] + argv
        return parse_args()
    finally:
        sys.argv = old


def test_flag_absent_parses_as_none_not_false():
    # None is the "nothing said" sentinel resolve_env_order needs; a
    # plain store_true default of False would silently outrank the
    # shell var
    assert _env_order([]) is None


def test_flags_parse_to_true_and_false():
    assert _env_order(["--env-order"]) is True
    assert _env_order(["--no-env-order"]) is False


def test_last_flag_wins():
    assert _env_order(["--env-order", "--no-env-order"]) is False
    assert _env_order(["--no-env-order", "--env-order"]) is True
