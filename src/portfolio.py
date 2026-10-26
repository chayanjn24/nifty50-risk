"""Portfolio construction, walk-forward: each month-end, weights come from the last 252 days only.

  Equal weight       1/N of every stock with a full year of history
  Inverse volatility weight by 1 / volatility
  Minimum variance   long-only, max 10% per stock, Ledoit-Wolf shrunk covariance
  Risk parity (ERC)  every stock adds the same amount of risk
  HRP                hierarchical risk parity (Lopez de Prado, 2016)

Outputs (data/processed/): portfolios.parquet (daily), weights.parquet (each rebalance)
"""
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import leaves_list, linkage
from scipy.optimize import minimize
from scipy.spatial.distance import squareform
from sklearn.covariance import LedoitWolf

import data
from config import EVAL_START, PROCESSED
from sim import month_starts, simulate

LOOKBACK = 252
MAX_WEIGHT = 0.10


def equal_weight(cov: pd.DataFrame) -> np.ndarray:
    n = len(cov)
    return np.full(n, 1 / n)


def inverse_vol(cov: pd.DataFrame) -> np.ndarray:
    iv = 1 / np.sqrt(np.diag(cov))
    return iv / iv.sum()


def min_variance(cov: pd.DataFrame, cap: float = MAX_WEIGHT) -> np.ndarray:
    s = cov.values
    n = len(s)
    res = minimize(lambda w: w @ s @ w, np.full(n, 1 / n), jac=lambda w: 2 * s @ w,
                   bounds=[(0, cap)] * n, constraints=[{"type": "eq", "fun": lambda w: w.sum() - 1}],
                   method="SLSQP", options={"maxiter": 500, "ftol": 1e-12})
    w = np.clip(res.x, 0, None)
    return w / w.sum()


def risk_parity(cov: pd.DataFrame) -> np.ndarray:
    """Equal risk contribution via the convex form: min 0.5 y'Sy - sum(log y), then rescale."""
    s = cov.values
    n = len(s)
    res = minimize(lambda y: 0.5 * y @ s @ y - np.log(y).sum() / n, np.full(n, 1 / n),
                   jac=lambda y: s @ y - 1 / (n * y), bounds=[(1e-10, None)] * n, method="L-BFGS-B")
    return res.x / res.x.sum()


def hrp(cov: pd.DataFrame) -> np.ndarray:
    s = cov.values
    sd = np.sqrt(np.diag(s))
    corr = s / np.outer(sd, sd)
    dist = np.sqrt(np.clip((1 - corr) / 2, 0, None))
    order = leaves_list(linkage(squareform(dist, checks=False), method="single"))
    w = np.ones(len(s))
    clusters = [list(order)]
    while clusters:
        nxt = []
        for c in clusters:
            if len(c) < 2:
                continue
            half = len(c) // 2
            left, right = c[:half], c[half:]

            def cluster_var(idx):
                sub = s[np.ix_(idx, idx)]
                iv = 1 / np.diag(sub)
                iv /= iv.sum()
                return iv @ sub @ iv

            vl, vr = cluster_var(left), cluster_var(right)
            alpha = 1 - vl / (vl + vr)
            w[left] *= alpha
            w[right] *= 1 - alpha
            nxt += [left, right]
        clusters = nxt
    return w / w.sum()


METHODS = {
    "Equal weight": equal_weight,
    "Inverse volatility": inverse_vol,
    "Minimum variance": min_variance,
    "Risk parity (ERC)": risk_parity,
    "HRP": hrp,
}


def weights_over_time(rets: pd.DataFrame, tickers: list[str]) -> dict[str, pd.DataFrame]:
    out = {m: {} for m in METHODS}
    for d in month_starts(rets.index, EVAL_START):
        i = rets.index.get_loc(d)
        window = rets[tickers].iloc[i - LOOKBACK + 1:i + 1].dropna(axis=1)  # full history only
        cov = pd.DataFrame(LedoitWolf().fit(window.values).covariance_,
                           index=window.columns, columns=window.columns)
        for m, fn in METHODS.items():
            out[m][d] = pd.Series(fn(cov), index=cov.index)
    return {m: pd.DataFrame(v).T.fillna(0.0) for m, v in out.items()}


def main():
    PROCESSED.mkdir(parents=True, exist_ok=True)
    rets = data.returns_wide()
    tickers = data.stocks(rets)
    W = weights_over_time(rets, tickers)

    daily, wlong = [], []
    for m, w in W.items():
        sim = simulate(w, rets[tickers], start=EVAL_START)
        daily.append(sim.assign(strategy=m).rename_axis("date").reset_index())
        x = w.stack().rename("weight").reset_index()
        x.columns = ["date", "ticker", "weight"]
        wlong.append(x[x.weight > 1e-6].assign(method=m))  # drop solver dust
        eff_n = (1 / (w ** 2).sum(axis=1)).mean()
        print(f"{m:20s} avg stocks held {(w > 1e-4).sum(axis=1).mean():.1f}  effective N {eff_n:.1f}"
              f"  max weight {w.max(axis=1).mean():.3f}")
    pd.concat(daily).to_parquet(PROCESSED / "portfolios.parquet")
    pd.concat(wlong).to_parquet(PROCESSED / "weights.parquet")


if __name__ == "__main__":
    main()
