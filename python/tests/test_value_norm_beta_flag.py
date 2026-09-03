"""--value-norm-beta (2026-09-03, audit 2.6 T=64 probes).

VALUE_NORM_BETA was a module constant (0.01 per update, halflife ~69
updates). The EMA is update-indexed, so a recipe that doubles the
samples per update (T=64) halves the halflife in samples unless beta is
doubled. The flag exposes it; the default is the old constant, so every
existing argv is bit-identical (same float, same two in-place ops).
"""

from train_batch import VALUE_NORM_BETA, parse_args


def _parse(argv):
    import sys
    old = sys.argv
    try:
        sys.argv = ["train_batch.py"] + argv
        return parse_args()
    finally:
        sys.argv = old


def test_default_is_the_old_constant():
    assert VALUE_NORM_BETA == 0.01
    assert _parse([]).value_norm_beta == VALUE_NORM_BETA
    # the same float object semantics: 1.0 - beta must reproduce the
    # exact multiplier the constant produced
    assert 1.0 - _parse([]).value_norm_beta == 1.0 - VALUE_NORM_BETA


def test_flag_parses():
    assert _parse(["--value-norm-beta", "0.02"]).value_norm_beta == 0.02
