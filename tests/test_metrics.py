import numpy as np
import pandas as pd
import pytest

from metrics import (cagr, christoffersen_ind, kupiec_pof, max_drawdown,
                     stationary_bootstrap_index)


def test_max_drawdown_known_path():
    # 100 -> 120 -> 60 -> 90: worst fall is 120 -> 60 = -50%
    r = pd.Series([0.20, -0.50, 0.50])
    assert max_drawdown(r) == pytest.approx(-0.50)


def test_cagr_doubling_in_one_year():
    r = pd.Series([2 ** (1 / 252) - 1] * 252)
    assert cagr(r) == pytest.approx(1.0)


def test_kupiec_passes_at_expected_rate_and_fails_far_from_it():
    _, p_ok = kupiec_pof(breaches=50, n=1000, p=0.05)
    _, p_bad = kupiec_pof(breaches=100, n=1000, p=0.05)
    assert p_ok > 0.99
    assert p_bad < 0.001


def test_christoffersen_flags_clustered_breaches():
    # independent breaches: the test should reject only about 5% of the time
    rng = np.random.default_rng(0)
    pvals = [christoffersen_ind((rng.random(1000) < 0.05).astype(int))[1] for _ in range(200)]
    assert 0.01 < np.mean(np.array(pvals) < 0.05) < 0.12
    clustered = np.zeros(1000, dtype=int)
    clustered[100:150] = 1  # 50 breaches, back to back
    assert christoffersen_ind(clustered)[1] < 0.001


def test_stationary_bootstrap_shape_and_blocks():
    idx = stationary_bootstrap_index(500, 50, block=20, rng=np.random.default_rng(0))
    assert idx.shape == (50, 500)
    assert idx.min() >= 0 and idx.max() < 500
    # most steps continue the block (next day), about 1 in 20 jumps
    cont = (np.diff(idx, axis=1) % 500 == 1).mean()
    assert 0.9 < cont < 0.99
