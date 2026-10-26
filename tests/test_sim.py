import numpy as np
import pandas as pd
import pytest

from metrics import RF_DAILY
from sim import CASH, simulate

DATES = pd.bdate_range("2024-01-01", periods=5)


def test_signal_earns_from_the_next_day_only():
    # the asset jumps 10% on day 2; a signal taken at the close of day 2 must not catch it
    rets = pd.DataFrame({"A": [0, 0, 0.10, 0, 0.05]}, index=DATES)
    targets = pd.DataFrame({"A": [1.0], CASH: [0.0]}, index=[DATES[2]])
    out = simulate(targets, rets, cost_bps=0)
    assert out.gross.iloc[2] == pytest.approx(RF_DAILY)  # still in cash on the jump day
    assert out.gross.iloc[4] == pytest.approx(0.05)  # invested afterwards


def test_costs_charged_on_turnover():
    rets = pd.DataFrame({"A": [0.0] * 5}, index=DATES)
    targets = pd.DataFrame({"A": [1.0, 0.0], CASH: [0.0, 1.0]}, index=[DATES[0], DATES[3]])
    out = simulate(targets, rets, cost_bps=10)
    assert out.turnover.sum() == pytest.approx(2.0)  # in, then out
    assert out.cost.sum() == pytest.approx(0.002)


def test_weights_drift_between_rebalances():
    # 50/50; A doubles, so before the next rebalance A is 2/3 of the book
    rets = pd.DataFrame({"A": [0, 1.0, 0], "B": [0, 0, 0]}, index=DATES[:3])
    targets = pd.DataFrame({"A": [0.5, 0.5], "B": [0.5, 0.5]}, index=[DATES[0], DATES[2]])
    out = simulate(targets, rets, cost_bps=0)
    assert out.turnover.iloc[2] == pytest.approx(2 * (2 / 3 - 0.5))
    assert np.isclose(out.gross.iloc[1], 0.5)
