"""Trading rules and portfolio weights, run walk-forward and compared with bootstrap ranges."""
import itertools

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import leaves_list, linkage
from scipy.optimize import minimize
from scipy.spatial.distance import squareform
from sklearn.covariance import LedoitWolf

from . import data
from .data import DAYS, ETF, EVAL_START, OUT, SEED
from .stats import CASH, RF_DAILY, bootstrap_index, month_ends, sharpe, simulate, summary

BENCH = "Buy and hold (Nifty ETF)"
SMA_PAIRS = [(f, s) for f, s in itertools.product([20, 50, 100], [100, 150, 200, 250]) if f < s]
MOM_GRID = list(itertools.product([63, 126, 189, 252], [5, 10, 15, 20]))  # lookback days, stocks held
REGIME_THRESHOLDS = [0.3, 0.5, 0.7, 0.9]
LOOKBACK = 252
MAX_WEIGHT = 0.10


# ---- trading rules -------------------------------------------------------------------------

def etf_or_cash(signal: pd.Series) -> pd.DataFrame:
    s = signal.astype(float)
    return pd.DataFrame({ETF: s, CASH: 1 - s})


def sma(rets, close, fast, slow):
    """Hold the ETF while the index's fast average is above its slow one, else cash."""
    signal = (close.rolling(fast).mean() > close.rolling(slow).mean())[close.rolling(slow).count() == slow]
    return simulate(etf_or_cash(signal), rets[[ETF]])


def sma_walk_forward(rets, close):
    """Each January, use the SMA pair with the best Sharpe on all earlier years."""
    runs = {p: sma(rets, close, *p) for p in SMA_PAIRS}
    pieces = []
    for year in range(pd.Timestamp(EVAL_START).year, rets.index.max().year + 1):
        best = max(runs, key=lambda p: sharpe(runs[p].ret[runs[p].index.year < year]))
        pieces.append(runs[best][runs[best].index.year == year])
    return pd.concat(pieces)


def momentum(rets, tickers, lookback=252, top=10):
    """Each month hold the `top` stocks with the best return from `lookback` days to 1 month ago."""
    wealth = (1 + rets[tickers].fillna(0)).cumprod()
    listed = rets[tickers].notna().rolling(lookback + 1).sum() == lookback + 1
    rows = {}
    for d in month_ends(rets.index):
        i = rets.index.get_loc(d)
        if i > lookback:
            score = (wealth.iloc[i - 21] / wealth.iloc[i - lookback] - 1)[listed.iloc[i]]
            pick = score.nlargest(top).index
            rows[d] = pd.Series(1 / len(pick), index=pick)
    return simulate(pd.DataFrame(rows).T.fillna(0.0), rets[tickers])


# ---- portfolio weights from a covariance matrix ----------------------------------------------

def equal_weight(cov):
    return np.full(len(cov), 1 / len(cov))


def inverse_vol(cov):
    iv = 1 / np.sqrt(np.diag(cov))
    return iv / iv.sum()


def min_variance(cov, cap=MAX_WEIGHT):
    s = np.asarray(cov)
    n = len(s)
    res = minimize(lambda w: w @ s @ w, np.full(n, 1 / n), jac=lambda w: 2 * s @ w, method="SLSQP",
                   bounds=[(0, cap)] * n, constraints=[{"type": "eq", "fun": lambda w: w.sum() - 1}],
                   options={"maxiter": 500, "ftol": 1e-12})
    w = np.clip(res.x, 0, None)
    return w / w.sum()


def risk_parity(cov):
    """Equal risk contribution, solved as min 0.5 y'Sy - mean(log y) and rescaled."""
    s = np.asarray(cov)
    n = len(s)
    res = minimize(lambda y: 0.5 * y @ s @ y - np.log(y).sum() / n, np.full(n, 1 / n),
                   jac=lambda y: s @ y - 1 / (n * y), bounds=[(1e-10, None)] * n, method="L-BFGS-B")
    return res.x / res.x.sum()


def hrp(cov):
    """Hierarchical risk parity (Lopez de Prado, 2016)."""
    s = np.asarray(cov)
    sd = np.sqrt(np.diag(s))
    dist = np.sqrt(np.clip((1 - s / np.outer(sd, sd)) / 2, 0, None))
    order = list(leaves_list(linkage(squareform(dist, checks=False), "single")))

    def cluster_var(idx):
        sub = s[np.ix_(idx, idx)]
        iv = 1 / np.diag(sub) / (1 / np.diag(sub)).sum()
        return iv @ sub @ iv

    w = np.ones(len(s))
    queue = [order]
    while queue:
        c = queue.pop()
        if len(c) > 1:
            left, right = c[:len(c) // 2], c[len(c) // 2:]
            a = 1 - cluster_var(left) / (cluster_var(left) + cluster_var(right))
            w[left] *= a
            w[right] *= 1 - a
            queue += [left, right]
    return w / w.sum()


PORTFOLIOS = {"Equal weight": equal_weight, "Inverse volatility": inverse_vol,
              "Minimum variance": min_variance, "Risk parity (ERC)": risk_parity, "HRP": hrp}


def portfolio_weights(rets, tickers):
    """Month-end weights from the last 252 days only (Ledoit-Wolf covariance)."""
    out = {m: {} for m in PORTFOLIOS}
    ends = month_ends(rets.index)
    for d in ends[ends >= pd.Timestamp(EVAL_START) - pd.offsets.MonthBegin(1)]:
        i = rets.index.get_loc(d)
        window = rets[tickers].iloc[i - LOOKBACK + 1:i + 1].dropna(axis=1)
        cov = pd.DataFrame(LedoitWolf().fit(window.values).covariance_, window.columns, window.columns)
        for m, fn in PORTFOLIOS.items():
            out[m][d] = pd.Series(fn(cov), index=cov.index)
    return {m: pd.DataFrame(v).T.fillna(0.0) for m, v in out.items()}


# ---- run and compare --------------------------------------------------------------------------

def run():
    rets = data.returns_wide()
    close = data.index_close()
    tickers = data.stocks(rets)
    stress = pd.read_parquet(OUT / "regimes.parquet").stress_prob_wf.dropna()

    runs = {
        BENCH: simulate(pd.DataFrame({ETF: [1.0]}, index=rets.index[:1]), rets[[ETF]]),
        "SMA 50/200": sma(rets, close, 50, 200),
        "SMA walk-forward": sma_walk_forward(rets, close),
        "Momentum 12-1, top 10": momentum(rets, tickers),
        "Regime filter": simulate(etf_or_cash(stress < 0.5), rets[[ETF]]),
    }
    weights = portfolio_weights(rets, tickers)
    for m, w in weights.items():
        runs[m] = simulate(w, rets[tickers], start=EVAL_START)

    daily = pd.concat(df[df.index >= EVAL_START].assign(strategy=name) for name, df in runs.items())
    daily = daily.rename_axis("date").reset_index()
    daily.to_parquet(OUT / "strategies.parquet")
    wide = daily.pivot(index="date", columns="strategy", values="ret").dropna()

    # 95% ranges for each Sharpe and for the gap vs buy-and-hold / equal weight (same resampled days)
    idx = bootstrap_index(len(wide), 2000, 21, np.random.default_rng(SEED))
    ex = wide.values - RF_DAILY
    boot = {c: (lambda s: s.mean(1) / s.std(1, ddof=1) * np.sqrt(DAYS))(ex[:, j][idx])
            for j, c in enumerate(wide.columns)}
    rows = []
    for c in wide.columns:
        d = daily[daily.strategy == c]
        rows.append({"strategy": c, **summary(wide[c]),
                     "sharpe_lo": np.percentile(boot[c], 2.5), "sharpe_hi": np.percentile(boot[c], 97.5),
                     "beats_bh": (boot[c] > boot[BENCH]).mean(),
                     "beats_ew": (boot[c] > boot["Equal weight"]).mean(),
                     "turnover_per_year": d.turnover.sum() * DAYS / len(d),
                     "cost_per_year": d.cost.sum() * DAYS / len(d)})
    table = pd.DataFrame(rows).sort_values("sharpe", ascending=False)
    table.to_parquet(OUT / "summary.parquet")

    regime = pd.read_parquet(OUT / "regimes.parquet").regime
    by = wide.join(regime).groupby("regime").agg(lambda r: (1 + r).prod() ** (DAYS / len(r)) - 1)
    by.T.rename_axis("strategy").reset_index().to_parquet(OUT / "by_regime.parquet")

    sens = [("SMA", f, s, sma(rets, close, f, s).ret) for f, s in SMA_PAIRS]
    sens += [("Momentum", lb, top, momentum(rets, tickers, lb, top).ret) for lb, top in MOM_GRID]
    sens += [("Regime", th, 0, simulate(etf_or_cash(stress < th), rets[[ETF]]).ret) for th in REGIME_THRESHOLDS]
    pd.DataFrame([{"family": f, "p1": a, "p2": b, "sharpe": sharpe(r[r.index >= EVAL_START])}
                  for f, a, b, r in sens]).to_parquet(OUT / "sensitivity.parquet")

    w = pd.concat(v.stack().rename("weight").rename_axis(["date", "ticker"]).reset_index().assign(method=m)
                  for m, v in weights.items())
    w[w.weight > 1e-6].to_parquet(OUT / "weights.parquet")

    print(table[["strategy", "cagr", "sharpe", "sharpe_lo", "sharpe_hi", "max_dd", "beats_bh", "beats_ew"]]
          .round(3).to_string(index=False))
