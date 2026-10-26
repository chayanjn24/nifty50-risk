"""Performance metrics, VaR tests, bootstrap and the portfolio simulator."""
import numpy as np
import pandas as pd
from scipy import stats

from .data import COST_BPS, DAYS, RISK_FREE

RF_DAILY = (1 + RISK_FREE) ** (1 / DAYS) - 1
CASH = "CASH"


def max_drawdown(r: pd.Series) -> float:
    w = (1 + r).cumprod()
    return float((w / w.cummax() - 1).min())


def cagr(r: pd.Series) -> float:
    return float((1 + r).prod() ** (DAYS / len(r)) - 1)


def ann_vol(r: pd.Series) -> float:
    return float(r.std() * np.sqrt(DAYS))


def sharpe(r) -> float:
    ex = np.asarray(r) - RF_DAILY
    return float(ex.mean() / ex.std(ddof=1) * np.sqrt(DAYS))


def summary(r: pd.Series) -> dict:
    return {"cagr": cagr(r), "ann_vol": ann_vol(r), "sharpe": sharpe(r), "max_dd": max_drawdown(r)}


def kupiec(breaches: int, n: int, p: float) -> float:
    """p-value for 'the breach rate equals p'."""
    x, phat = breaches, breaches / n
    if x in (0, n):
        lr = -2 * n * np.log(1 - p) if x == 0 else -2 * n * np.log(p)
    else:
        lr = -2 * ((n - x) * np.log((1 - p) / (1 - phat)) + x * np.log(p / phat))
    return float(1 - stats.chi2.cdf(lr, 1))


def christoffersen(hits) -> float:
    """p-value for 'a breach today says nothing about tomorrow'."""
    h = np.asarray(hits, dtype=int)
    prev, cur = h[:-1], h[1:]
    n00, n01 = np.sum((prev == 0) & (cur == 0)), np.sum((prev == 0) & (cur == 1))
    n10, n11 = np.sum((prev == 1) & (cur == 0)), np.sum((prev == 1) & (cur == 1))
    pi = (n01 + n11) / (n00 + n01 + n10 + n11)
    if pi in (0, 1):
        return 1.0
    pi0, pi1 = n01 / max(n00 + n01, 1), n11 / max(n10 + n11, 1)

    def ll(p, stay, breach):
        return (stay * np.log(1 - p) if stay else 0) + (breach * np.log(p) if breach else 0)

    lr = -2 * (ll(pi, n00 + n10, n01 + n11) - ll(pi0, n00, n01) - ll(pi1, n10, n11))
    return float(1 - stats.chi2.cdf(lr, 1))


def bootstrap_index(n: int, reps: int, block: float, rng) -> np.ndarray:
    """Stationary bootstrap: random-length blocks of days, so volatility clusters survive."""
    idx = np.empty((reps, n), dtype=np.int64)
    idx[:, 0] = rng.integers(0, n, reps)
    jump = rng.random((reps, n)) < 1 / block
    starts = rng.integers(0, n, (reps, n))
    for t in range(1, n):
        idx[:, t] = np.where(jump[:, t], starts[:, t], (idx[:, t - 1] + 1) % n)
    return idx


def simulate(targets: pd.DataFrame, rets: pd.DataFrame, start=None, cost_bps: float = COST_BPS) -> pd.DataFrame:
    """Daily net returns for target weights decided at each date's close.

    New weights earn returns from the next day (one-day lag). Between decisions the
    weights drift with prices; turnover is measured from the drifted weights.
    """
    cols = list(dict.fromkeys([*targets.columns, CASH]))
    targets = targets.reindex(columns=cols, fill_value=0.0).fillna(0.0)
    r = rets.assign(**{CASH: RF_DAILY}).reindex(columns=cols).fillna(0.0)
    dates = r.index if start is None else r.index[r.index >= pd.Timestamp(start)]
    cash = cols.index(CASH)

    earlier = targets.index[targets.index < dates[0]]
    if len(earlier):
        w = targets.loc[earlier.max()].values.astype(float)
    else:
        w = np.eye(len(cols))[cash]
    plan = dict(zip(targets.index, targets.values))
    out = np.zeros((len(dates), 3))
    for i, (d, day) in enumerate(zip(dates, r.loc[dates].values)):
        gross = w @ day
        w = w * (1 + day) / (w * (1 + day)).sum()
        turnover = 0.0
        if d in plan:
            turnover = np.abs(np.delete(plan[d] - w, cash)).sum()  # moving cash is free
            w = plan[d].copy()
        cost = turnover * cost_bps / 1e4
        out[i] = [gross - cost, cost, turnover]
    return pd.DataFrame(out, index=dates, columns=["ret", "cost", "turnover"])


def month_ends(index: pd.DatetimeIndex) -> pd.DatetimeIndex:
    s = pd.Series(index, index=index)
    return pd.DatetimeIndex(s.groupby([index.year, index.month]).max().values)
