"""Market model, dependence, VaR backtests and volatility forecasts.

Outputs (data/processed/):
  capm.parquet, rolling_beta.parquet, clusters.parquet, corr.parquet,
  avg_corr.parquet, var_backtest.parquet, var_summary.parquet, vol_forecast.parquet
"""
import warnings

import numpy as np
import pandas as pd
import statsmodels.api as sm
from arch import arch_model
from scipy.cluster.hierarchy import fcluster, leaves_list, linkage
from scipy.spatial.distance import squareform
from scipy.stats import norm, t as student_t

import data
from config import INDEX, PROCESSED, SEED
from metrics import RF_DAILY, christoffersen_ind, kupiec_pof

warnings.filterwarnings("ignore")

VAR_WINDOW = 250
VAR_LEVELS = [0.95, 0.99]
GARCH_REFIT = 21  # re-estimate GARCH parameters every month; forecast daily
N_CLUSTERS = 10


def capm(rets: pd.DataFrame, tickers: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    mkt = rets[INDEX] - RF_DAILY
    rows, rolling = [], {}
    for t in tickers:
        y = (rets[t] - RF_DAILY).dropna()
        x = sm.add_constant(mkt.reindex(y.index))
        fit = sm.OLS(y, x).fit(cov_type="HAC", cov_kwds={"maxlags": 5})
        rows.append({
            "ticker": t,
            "beta": fit.params.iloc[1],
            "beta_t": fit.tvalues.iloc[1],
            "alpha_ann": (1 + fit.params.iloc[0]) ** 252 - 1,
            "alpha_t": fit.tvalues.iloc[0],
            "r2": fit.rsquared,
            "n": int(fit.nobs),
        })
        cov = rets[t].rolling(252).cov(rets[INDEX])
        rolling[t] = cov / rets[INDEX].rolling(252).var()
    rb = pd.DataFrame(rolling).stack().rename("beta_252d").reset_index()
    rb.columns = ["date", "ticker", "beta_252d"]
    return pd.DataFrame(rows), rb


def clusters(rets: pd.DataFrame, tickers: list[str]) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray]:
    """Correlation over the last 3 years (all 50 stocks listed), clustered by distance sqrt((1 - rho) / 2)."""
    recent = rets[tickers].loc[rets.index >= rets.index.max() - pd.DateOffset(years=3)].dropna(axis=1)
    corr = recent.corr()
    dist = np.sqrt(((1 - corr) / 2).clip(lower=0))
    link = linkage(squareform(dist.values, checks=False), method="ward")
    order = leaves_list(link)
    labels = fcluster(link, N_CLUSTERS, criterion="maxclust")
    cl = pd.DataFrame({"ticker": corr.index, "cluster": labels,
                       "dendro_order": np.argsort(order)})
    long = corr.rename_axis("ticker_a").rename_axis("ticker_b", axis=1).stack().rename("rho").reset_index()
    return cl, long, link


def avg_pairwise_corr(rets: pd.DataFrame, tickers: list[str], window: int = 63) -> pd.Series:
    """Mean off-diagonal correlation across stocks on a rolling window."""
    r = rets[tickers]
    out = {}
    idx = r.index
    for i in range(window, len(idx) + 1, 5):  # every 5th day is enough for a chart
        block = r.iloc[i - window:i].dropna(axis=1)
        c = np.corrcoef(block.values.T)
        n = c.shape[0]
        out[idx[i - 1]] = (c.sum() - n) / (n * (n - 1))
    return pd.Series(out, name="avg_corr_63d")


def var_backtest(r: pd.Series, name: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """One-day-ahead VaR from data up to yesterday, three ways, compared with today's return."""
    r = r.dropna()
    rows = []
    garch_params = None
    am = None
    for i in range(VAR_WINDOW, len(r)):
        hist = r.iloc[i - VAR_WINDOW:i]
        today = r.index[i]
        mu, sd = hist.mean(), hist.std(ddof=1)
        # GARCH(1,1) with Student-t errors on an expanding window, refit monthly
        if garch_params is None or (i - VAR_WINDOW) % GARCH_REFIT == 0:
            am = arch_model(r.iloc[:i] * 100, mean="Constant", vol="GARCH", p=1, q=1, dist="t")
            garch_params = am.fit(disp="off").params
        f = arch_model(r.iloc[:i] * 100, mean="Constant", vol="GARCH", p=1, q=1, dist="t") \
            .fix(garch_params).forecast(horizon=1, reindex=False)
        g_mu = f.mean.values[-1, 0] / 100
        g_sd = np.sqrt(f.variance.values[-1, 0]) / 100
        nu = garch_params["nu"]
        t_scale = np.sqrt((nu - 2) / nu)  # unit-variance Student-t
        row = {"date": today, "portfolio": name, "ret": r.iloc[i],
               "garch_vol": g_sd * np.sqrt(252), "roll_vol": sd * np.sqrt(252)}
        for lvl in VAR_LEVELS:
            q = 1 - lvl
            row[f"hist_{lvl}"] = -np.quantile(hist, q)
            row[f"normal_{lvl}"] = -(mu + sd * norm.ppf(q))
            row[f"garch_{lvl}"] = -(g_mu + g_sd * t_scale * student_t.ppf(q, nu))
        rows.append(row)
    bt = pd.DataFrame(rows)

    long = []
    summ = []
    for method in ["hist", "normal", "garch"]:
        for lvl in VAR_LEVELS:
            col = f"{method}_{lvl}"
            hits = (bt["ret"] < -bt[col]).astype(int)
            n, x = len(hits), int(hits.sum())
            lr_uc, p_uc = kupiec_pof(x, n, 1 - lvl)
            lr_ind, p_ind = christoffersen_ind(hits.values)
            summ.append({"portfolio": name, "method": method, "level": lvl, "days": n,
                         "breaches": x, "expected": n * (1 - lvl), "breach_rate": x / n,
                         "kupiec_lr": lr_uc, "kupiec_p": p_uc,
                         "christoffersen_lr": lr_ind, "christoffersen_p": p_ind,
                         "avg_var": bt[col].mean()})
            long.append(pd.DataFrame({"date": bt["date"], "portfolio": name, "method": method,
                                      "level": lvl, "var": bt[col], "ret": bt["ret"],
                                      "breach": hits}))
    vol = bt[["date", "portfolio", "ret", "garch_vol", "roll_vol"]]
    return pd.concat(long), pd.DataFrame(summ), vol


def vol_forecast_scores(vol: pd.DataFrame) -> pd.DataFrame:
    """Which forecast tracks next-day variance better? MSE and QLIKE against squared returns."""
    out = []
    for name, g in vol.groupby("portfolio"):
        realized = g["ret"] ** 2
        for col in ["garch_vol", "roll_vol"]:
            h = (g[col] / np.sqrt(252)) ** 2
            out.append({"portfolio": name, "forecast": col.replace("_vol", ""),
                        "mse": float(((realized - h) ** 2).mean() * 1e8),
                        "qlike": float((realized / h - np.log(realized / h + 1e-12) - 1).mean())})
    return pd.DataFrame(out)


def main():
    np.random.seed(SEED)
    PROCESSED.mkdir(parents=True, exist_ok=True)
    rets = data.returns_wide()
    tickers = data.stocks(rets)

    cp, rb = capm(rets, tickers)
    cp.to_parquet(PROCESSED / "capm.parquet")
    rb.to_parquet(PROCESSED / "rolling_beta.parquet")
    print("CAPM: median beta", round(cp.beta.median(), 2), "| R2 median", round(cp.r2.median(), 2))

    cl, corr_long, _ = clusters(rets, tickers)
    cl.to_parquet(PROCESSED / "clusters.parquet")
    corr_long.to_parquet(PROCESSED / "corr.parquet")
    ac = avg_pairwise_corr(rets, tickers)
    ac.to_frame().to_parquet(PROCESSED / "avg_corr.parquet")
    print("clusters:", cl.groupby("cluster").ticker.apply(list).to_dict())

    ew = rets[tickers].mean(axis=1, skipna=True).rename("ew")  # daily-rebalanced equal weight
    parts = [var_backtest(rets[INDEX], "Nifty 50"), var_backtest(ew, "Equal weight 50")]
    var_long = pd.concat([p[0] for p in parts])
    var_sum = pd.concat([p[1] for p in parts])
    vol = pd.concat([p[2] for p in parts])
    var_long.to_parquet(PROCESSED / "var_backtest.parquet")
    var_sum.to_parquet(PROCESSED / "var_summary.parquet")
    vol.to_parquet(PROCESSED / "vol_forecast.parquet")
    vol_forecast_scores(vol).to_parquet(PROCESSED / "vol_scores.parquet")
    print(var_sum[["portfolio", "method", "level", "breaches", "expected", "kupiec_p", "christoffersen_p"]]
          .round(3).to_string(index=False))
    print(vol_forecast_scores(vol).round(4).to_string(index=False))


if __name__ == "__main__":
    main()
