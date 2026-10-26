"""Performance and risk statistics shared by every step."""
import numpy as np
import pandas as pd
from scipy import stats

from config import RISK_FREE, TRADING_DAYS

RF_DAILY = (1 + RISK_FREE) ** (1 / TRADING_DAYS) - 1


def wealth(r: pd.Series) -> pd.Series:
    return (1 + r).cumprod()


def drawdown(r: pd.Series) -> pd.Series:
    w = wealth(r)
    return w / w.cummax() - 1


def max_drawdown(r: pd.Series) -> float:
    return float(drawdown(r).min())


def cagr(r: pd.Series) -> float:
    return float(wealth(r).iloc[-1] ** (TRADING_DAYS / len(r)) - 1)


def ann_vol(r: pd.Series) -> float:
    return float(r.std(ddof=1) * np.sqrt(TRADING_DAYS))


def sharpe(r, rf_daily: float = RF_DAILY) -> float:
    ex = np.asarray(r) - rf_daily
    return float(ex.mean() / ex.std(ddof=1) * np.sqrt(TRADING_DAYS))


def sortino(r: pd.Series, rf_daily: float = RF_DAILY) -> float:
    ex = r - rf_daily
    downside = np.sqrt((np.minimum(ex, 0) ** 2).mean())
    return float(ex.mean() / downside * np.sqrt(TRADING_DAYS))


def summary(r: pd.Series) -> dict:
    mdd = max_drawdown(r)
    g = cagr(r)
    return {
        "cagr": g,
        "ann_vol": ann_vol(r),
        "sharpe": sharpe(r),
        "sortino": sortino(r),
        "max_dd": mdd,
        "calmar": g / abs(mdd) if mdd < 0 else np.nan,
        "start": r.index.min(),
        "end": r.index.max(),
        "days": len(r),
    }


def stationary_bootstrap_index(n: int, reps: int, block: float, rng: np.random.Generator) -> np.ndarray:
    """Politis-Romano stationary bootstrap: random-length blocks (mean `block` days) keep
    volatility clustering intact. Returns a reps x n array of row positions."""
    idx = np.empty((reps, n), dtype=np.int64)
    idx[:, 0] = rng.integers(0, n, reps)
    new_block = rng.random((reps, n)) < 1 / block
    jumps = rng.integers(0, n, (reps, n))
    for t in range(1, n):
        idx[:, t] = np.where(new_block[:, t], jumps[:, t], (idx[:, t - 1] + 1) % n)
    return idx


def kupiec_pof(breaches: int, n: int, p: float) -> tuple[float, float]:
    """Kupiec proportion-of-failures test. H0: breach rate equals p."""
    x = breaches
    phat = x / n
    if x == 0:
        lr = -2 * n * np.log(1 - p)
    elif x == n:
        lr = -2 * n * np.log(p)
    else:
        lr = -2 * ((n - x) * np.log((1 - p) / (1 - phat)) + x * np.log(p / phat))
    return float(lr), float(1 - stats.chi2.cdf(lr, 1))


def christoffersen_ind(hits) -> tuple[float, float]:
    """Christoffersen independence test. H0: a breach today says nothing about tomorrow."""
    h = np.asarray(hits, dtype=int)
    prev, cur = h[:-1], h[1:]
    n00 = np.sum((prev == 0) & (cur == 0))
    n01 = np.sum((prev == 0) & (cur == 1))
    n10 = np.sum((prev == 1) & (cur == 0))
    n11 = np.sum((prev == 1) & (cur == 1))
    pi0 = n01 / max(n00 + n01, 1)
    pi1 = n11 / max(n10 + n11, 1)
    pi = (n01 + n11) / max(n00 + n01 + n10 + n11, 1)

    def ll(p, a, b):  # log-likelihood of a non-breaches and b breaches at rate p
        return (a * np.log(1 - p) if a else 0) + (b * np.log(p) if b else 0)

    if pi in (0, 1):
        return 0.0, 1.0
    lr = -2 * (ll(pi, n00 + n10, n01 + n11) - ll(pi0, n00, n01) - ll(pi1, n10, n11))
    return float(lr), float(1 - stats.chi2.cdf(lr, 1))
