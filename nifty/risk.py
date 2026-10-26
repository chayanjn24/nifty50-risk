"""Market model, clusters, VaR backtests, GARCH and market regimes."""
import warnings

import numpy as np
import pandas as pd
import statsmodels.api as sm
from arch import arch_model
from scipy.cluster.hierarchy import fcluster, leaves_list, linkage
from scipy.spatial.distance import squareform
from scipy.stats import norm, t as student_t

from . import data
from .data import INDEX, OUT
from .stats import RF_DAILY, christoffersen, kupiec

warnings.filterwarnings("ignore")

VAR_WINDOW = 250
LEVELS = [0.95, 0.99]
GARCH_REFIT = 21
N_CLUSTERS = 10


def capm(rets, tickers):
    """Beta, alpha and R² of each stock against the index (Newey-West errors), plus a rolling beta."""
    mkt = sm.add_constant(rets[INDEX] - RF_DAILY)
    rows = []
    for t in tickers:
        y = (rets[t] - RF_DAILY).dropna()
        fit = sm.OLS(y, mkt.loc[y.index]).fit(cov_type="HAC", cov_kwds={"maxlags": 5})
        rows.append({"ticker": t, "beta": fit.params.iloc[1], "alpha_ann": (1 + fit.params.iloc[0]) ** 252 - 1,
                     "alpha_t": fit.tvalues.iloc[0], "r2": fit.rsquared})
    rolling = rets[tickers].rolling(252).cov(rets[INDEX]).div(rets[INDEX].rolling(252).var(), axis=0)
    rolling = rolling.stack().rename("beta_252d").rename_axis(["date", "ticker"]).reset_index()
    return pd.DataFrame(rows), rolling


def clusters(rets, tickers):
    """Ward clusters on correlation distance over the last 3 years."""
    recent = rets[tickers].loc[rets.index >= rets.index.max() - pd.DateOffset(years=3)].dropna(axis=1)
    corr = recent.corr()
    link = linkage(squareform(np.sqrt(((1 - corr) / 2).clip(lower=0)).values, checks=False), "ward")
    groups = pd.DataFrame({"ticker": corr.index, "cluster": fcluster(link, N_CLUSTERS, "maxclust"),
                           "order": np.argsort(leaves_list(link))})
    pairs = corr.rename_axis("ticker_a").rename_axis("ticker_b", axis=1).stack().rename("rho").reset_index()
    return groups, pairs


def avg_correlation(rets, tickers, window=63):
    """Mean correlation between stocks on a rolling window (every 5th day)."""
    out = {}
    for i in range(window, len(rets) + 1, 5):
        c = np.corrcoef(rets[tickers].iloc[i - window:i].dropna(axis=1).values.T)
        out[rets.index[i - 1]] = (c.sum() - len(c)) / (len(c) * (len(c) - 1))
    return pd.Series(out, name="avg_corr_63d")


def var_backtest(r, name):
    """One-day VaR from data up to yesterday (historical, normal, GARCH-t) against today's return."""
    r = r.dropna()
    rows, params = [], None
    for i in range(VAR_WINDOW, len(r)):
        hist = r.iloc[i - VAR_WINDOW:i]
        model = arch_model(r.iloc[:i] * 100, vol="GARCH", p=1, q=1, dist="t")
        if params is None or (i - VAR_WINDOW) % GARCH_REFIT == 0:
            params = model.fit(disp="off").params
        f = model.fix(params).forecast(horizon=1, reindex=False)
        g_mu, g_sd = f.mean.values[-1, 0] / 100, np.sqrt(f.variance.values[-1, 0]) / 100
        nu = params["nu"]
        row = {"date": r.index[i], "portfolio": name, "ret": r.iloc[i],
               "garch_vol": g_sd * np.sqrt(252), "roll_vol": hist.std() * np.sqrt(252)}
        for lvl in LEVELS:
            q = 1 - lvl
            row[f"Historical {lvl}"] = -np.quantile(hist, q)
            row[f"Normal {lvl}"] = -(hist.mean() + hist.std() * norm.ppf(q))
            row[f"GARCH-t {lvl}"] = -(g_mu + g_sd * np.sqrt((nu - 2) / nu) * student_t.ppf(q, nu))
        rows.append(row)
    bt = pd.DataFrame(rows)

    daily, tests = [], []
    for method in ["Historical", "Normal", "GARCH-t"]:
        for lvl in LEVELS:
            var = bt[f"{method} {lvl}"]
            hits = (bt.ret < -var).astype(int)
            tests.append({"portfolio": name, "method": method, "level": lvl, "days": len(hits),
                          "breaches": hits.sum(), "expected": len(hits) * (1 - lvl),
                          "kupiec_p": kupiec(hits.sum(), len(hits), 1 - lvl),
                          "christoffersen_p": christoffersen(hits)})
            daily.append(pd.DataFrame({"date": bt.date, "portfolio": name, "method": method,
                                       "level": lvl, "var": var, "ret": bt.ret, "breach": hits}))

    # forecast loss vs next-day squared return (lower is better)
    realized = bt.ret ** 2
    scores = []
    for col in ["garch_vol", "roll_vol"]:
        h = (bt[col] / np.sqrt(252)) ** 2
        scores.append({"portfolio": name, "forecast": col[:-4], "mse": ((realized - h) ** 2).mean() * 1e8,
                       "qlike": (realized / h - np.log(realized / h + 1e-12) - 1).mean()})
    return pd.concat(daily), pd.DataFrame(tests), pd.DataFrame(scores)


def markov(r):
    mod = sm.tsa.MarkovRegression(r * 100, k_regimes=2, trend="c", switching_variance=True)
    res = mod.fit(search_reps=20, disp=False)
    return res, int(np.argmax([res.params["sigma2[0]"], res.params["sigma2[1]"]]))


def regimes(r):
    """Calm vs stressed. Full-sample probabilities describe history; the walk-forward ones
    (refit each January on earlier years only) are safe to trade on."""
    res, stress = markov(r)
    wf = []
    for year in range(pd.Timestamp(data.EVAL_START).year, r.index.max().year + 1):
        fit, s = markov(r[r.index.year < year])
        upto = r[r.index.year <= year] * 100
        p = sm.tsa.MarkovRegression(upto, k_regimes=2, trend="c", switching_variance=True) \
            .filter(fit.params).filtered_marginal_probabilities[s]
        wf.append(p[p.index.year == year])
    out = pd.DataFrame({"index_ret": r, "stress_prob": res.smoothed_marginal_probabilities[stress],
                        "stress_prob_wf": pd.concat(wf)})
    out["regime"] = np.where(out.stress_prob > 0.5, "Stressed", "Calm")
    return out.rename_axis("date")


def run():
    OUT.mkdir(parents=True, exist_ok=True)
    rets = data.returns_wide()
    tickers = data.stocks(rets)

    cp, rolling_beta = capm(rets, tickers)
    groups, pairs = clusters(rets, tickers)
    cp.to_parquet(OUT / "capm.parquet")
    rolling_beta.to_parquet(OUT / "rolling_beta.parquet")
    groups.to_parquet(OUT / "clusters.parquet")
    pairs.to_parquet(OUT / "corr.parquet")
    avg_correlation(rets, tickers).to_frame().to_parquet(OUT / "avg_corr.parquet")

    equal_weight = rets[tickers].mean(axis=1)
    parts = [var_backtest(rets[INDEX], "Nifty 50"), var_backtest(equal_weight, "Equal weight 50")]
    for k, name in enumerate(["var_backtest", "var_summary", "vol_scores"]):
        pd.concat(p[k] for p in parts).to_parquet(OUT / f"{name}.parquet")

    reg = regimes(rets[INDEX].dropna())
    reg.to_parquet(OUT / "regimes.parquet")
    rows = []
    for name, days in reg.groupby("regime").groups.items():
        block = rets.loc[days, tickers].dropna(axis=1, thresh=int(len(days) * 0.9))
        c = block.corr().values
        rows.append({"regime": name, "days": len(days),
                     "avg_corr": (np.nansum(c) - len(c)) / (len(c) * (len(c) - 1)),
                     "index_vol": reg.loc[days, "index_ret"].std() * np.sqrt(252)})
    pd.DataFrame(rows).to_parquet(OUT / "regime_summary.parquet")
    print(f"risk: median beta {cp.beta.median():.2f}, stressed days {(reg.regime == 'Stressed').mean():.1%}")
