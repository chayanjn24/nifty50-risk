import numpy as np
import pandas as pd
import pytest

from portfolio import METHODS, hrp, min_variance, risk_parity


@pytest.fixture
def cov():
    rng = np.random.default_rng(1)
    x = rng.normal(size=(500, 6)) * np.array([0.01, 0.015, 0.02, 0.012, 0.03, 0.008])
    x[:, 1] += 0.7 * x[:, 0]  # some correlation
    names = list("ABCDEF")
    return pd.DataFrame(np.cov(x.T), index=names, columns=names)


@pytest.mark.parametrize("name", list(METHODS))
def test_weights_are_long_only_and_fully_invested(cov, name):
    w = METHODS[name](cov)
    assert w.sum() == pytest.approx(1.0)
    assert (w >= -1e-9).all()


def test_risk_parity_equalises_risk_contributions(cov):
    w = risk_parity(cov)
    rc = w * (cov.values @ w)
    assert np.allclose(rc / rc.sum(), 1 / len(w), atol=1e-4)


def test_min_variance_respects_cap(cov):
    w = min_variance(cov, cap=0.25)
    assert w.max() <= 0.25 + 1e-6


def test_hrp_uncorrelated_two_assets_is_inverse_variance():
    cov = pd.DataFrame([[0.04, 0.0], [0.0, 0.01]], index=["A", "B"], columns=["A", "B"])
    w = hrp(cov)
    assert w == pytest.approx([0.2, 0.8])
