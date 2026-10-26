"""Portfolio simulator: target weights in, daily net returns out.

Weights decided with data up to the close of day t are traded at that close and
earn returns from day t+1 (the one-day signal lag). Between rebalances weights
drift with prices, and turnover is measured against the drifted weights.
"""
import numpy as np
import pandas as pd

from config import COST_BPS
from metrics import RF_DAILY

CASH = "CASH"


def simulate(targets: pd.DataFrame, rets: pd.DataFrame, cost_bps: float = COST_BPS,
             start=None) -> pd.DataFrame:
    """targets: rows = decision dates, columns = tickers (and optionally CASH), rows sum to 1.

    Returns a frame indexed by date with columns ret (net), gross, cost, turnover.
    """
    rets = rets.copy()
    rets[CASH] = RF_DAILY
    cols = list(dict.fromkeys(list(targets.columns) + [CASH]))
    targets = targets.reindex(columns=cols, fill_value=0.0).fillna(0.0)
    r = rets.reindex(columns=cols).fillna(0.0)

    dates = r.index
    if start is not None:
        dates = dates[dates >= pd.Timestamp(start)]
    decision = targets.index

    cash_i = cols.index(CASH)
    w = np.zeros(len(cols))
    w[cash_i] = 1.0  # sit in cash until the first decision
    earlier = decision[decision < dates[0]]
    if len(earlier):  # a decision taken before the window opens is already in place
        w = targets.loc[earlier.max()].values.astype(float)
    out = np.zeros((len(dates), 4))
    r_arr = r.loc[dates].values
    t_rows = {d: row for d, row in zip(decision, targets.values)}
    for i, d in enumerate(dates):
        day_r = r_arr[i]
        gross = float(w @ day_r)
        w = w * (1 + day_r)
        w = w / w.sum()
        cost = turnover = 0.0
        if d in t_rows:
            new = t_rows[d]
            # cash leg trades for free; only stock/ETF legs pay costs
            turnover = float(np.abs(np.delete(new - w, cash_i)).sum())
            cost = turnover * cost_bps / 1e4
            w = new.copy()
        out[i] = [gross - cost, gross, cost, turnover]
    return pd.DataFrame(out, index=dates, columns=["ret", "gross", "cost", "turnover"])


def month_starts(index: pd.DatetimeIndex, start=None) -> pd.DatetimeIndex:
    """Last trading day of each month: decisions taken at that close, held through the next month."""
    s = pd.Series(index, index=index)
    last = s.groupby([index.year, index.month]).max()
    out = pd.DatetimeIndex(last.values)
    if start is not None:
        out = out[out >= pd.Timestamp(start) - pd.offsets.MonthBegin(1)]
    return out
