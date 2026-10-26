import numpy as np
import pandas as pd
import pytest

from nifty import data
from nifty.stats import CASH, RF_DAILY, christoffersen, kupiec, max_drawdown, simulate
from nifty.strategies import PORTFOLIOS, hrp, risk_parity

DATES = pd.bdate_range("2024-01-01", periods=5)


def test_max_drawdown():
    assert max_drawdown(pd.Series([0.20, -0.50, 0.50])) == pytest.approx(-0.50)  # 120 -> 60


def test_kupiec():
    assert kupiec(50, 1000, 0.05) > 0.99
    assert kupiec(100, 1000, 0.05) < 0.001


def test_christoffersen():
    rng = np.random.default_rng(0)
    false_alarms = np.mean([christoffersen(rng.random(1000) < 0.05) < 0.05 for _ in range(200)])
    assert 0.01 < false_alarms < 0.12  # independent breaches rejected about 5% of the time
    clustered = np.zeros(1000)
    clustered[100:150] = 1
    assert christoffersen(clustered) < 0.001


def test_signal_trades_next_day():
    rets = pd.DataFrame({"A": [0, 0, 0.10, 0, 0.05]}, index=DATES)
    out = simulate(pd.DataFrame({"A": [1.0], CASH: [0.0]}, index=[DATES[2]]), rets, cost_bps=0)
    assert out.ret.iloc[2] == pytest.approx(RF_DAILY)  # decided at day 2's close, misses day 2's jump
    assert out.ret.iloc[4] == pytest.approx(0.05)


def test_costs_and_drift():
    rets = pd.DataFrame({"A": [0, 1.0, 0], "B": [0, 0, 0]}, index=DATES[:3])
    targets = pd.DataFrame({"A": [0.5, 0.5], "B": [0.5, 0.5]}, index=[DATES[0], DATES[2]])
    out = simulate(targets, rets, cost_bps=10)
    assert out.turnover.iloc[2] == pytest.approx(2 * (2 / 3 - 0.5))  # A drifted to 2/3
    assert out.cost.iloc[2] == pytest.approx(out.turnover.iloc[2] * 0.001)


@pytest.fixture
def cov():
    x = np.random.default_rng(1).normal(size=(500, 6)) * [0.01, 0.015, 0.02, 0.012, 0.03, 0.008]
    x[:, 1] += 0.7 * x[:, 0]
    return pd.DataFrame(np.cov(x.T))


@pytest.mark.parametrize("method", PORTFOLIOS)
def test_weights_valid(cov, method):
    w = PORTFOLIOS[method](cov)
    assert w.sum() == pytest.approx(1) and (w >= -1e-9).all()


def test_risk_parity_equal_contributions(cov):
    w = risk_parity(cov)
    rc = w * (cov.values @ w)
    assert np.allclose(rc / rc.sum(), 1 / 6, atol=1e-4)


def test_hrp_two_assets_inverse_variance():
    assert hrp(np.diag([0.04, 0.01])) == pytest.approx([0.2, 0.8])


@pytest.mark.skipif(not data.DB.exists(), reason="run `python run.py` first")
def test_database():
    assert data.query("SELECT date, ticker FROM returns GROUP BY ALL HAVING count(*) > 1").empty
    fixed = data.query("""SELECT r.ret FROM returns r JOIN price_fixes f
                          ON f.ticker = r.ticker AND r.date BETWEEN f.date_from AND f.date_to""")
    assert (fixed.ret.abs() < 0.05).all()  # no fake crash left on a fixed day
    r = data.query("SELECT ret FROM returns WHERE ticker = 'HDFCBANK.NS' ORDER BY date").ret
    sql = data.query("SELECT max_dd FROM stock_metrics WHERE ticker = 'HDFCBANK.NS'").max_dd[0]
    assert sql == pytest.approx(max_drawdown(r))  # SQL and Python agree
