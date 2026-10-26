"""Calm vs stressed market regimes from a 2-state Markov switching model on Nifty 50 returns.

Two versions:
  smoothed  - fitted on the full sample; uses future data, so it is for description only
  walk-forward filtered - parameters fitted on data before each year, probability at day t
                          uses returns up to day t only; safe to trade on (with a one-day lag)

Outputs (data/processed/): regimes.parquet, regime_corr.parquet
"""
import warnings

import numpy as np
import pandas as pd
import statsmodels.api as sm

import data
from config import INDEX, PROCESSED

warnings.filterwarnings("ignore")

FIRST_OOS_YEAR = 2017  # two years of training before the first walk-forward year


def fit(r: pd.Series):
    mod = sm.tsa.MarkovRegression(r * 100, k_regimes=2, trend="c", switching_variance=True)
    res = mod.fit(search_reps=20, disp=False)
    stress = int(np.argmax(res.params[[f"sigma2[{i}]" for i in range(2)]].values))
    return res, stress


def walk_forward_probs(r: pd.Series) -> pd.Series:
    out = []
    for year in range(FIRST_OOS_YEAR, r.index.max().year + 1):
        train = r[r.index.year < year]
        res, stress = fit(train)
        upto = r[r.index.year <= year]
        filt = sm.tsa.MarkovRegression(upto * 100, k_regimes=2, trend="c", switching_variance=True) \
            .filter(res.params).filtered_marginal_probabilities[stress]
        out.append(filt[filt.index.year == year])
    return pd.concat(out).rename("stress_prob_wf")


def main():
    PROCESSED.mkdir(parents=True, exist_ok=True)
    rets = data.returns_wide()
    r = rets[INDEX].dropna()

    res, stress = fit(r)
    smoothed = res.smoothed_marginal_probabilities[stress].rename("stress_prob")
    wf = walk_forward_probs(r)
    reg = pd.concat([r.rename("index_ret"), smoothed, wf], axis=1)
    reg["regime"] = np.where(reg.stress_prob > 0.5, "Stressed", "Calm")
    reg.index.name = "date"
    reg.to_parquet(PROCESSED / "regimes.parquet")

    sig = res.params[[f"sigma2[{i}]" for i in range(2)]].values
    print("daily vol by regime (%):", np.sqrt(sig).round(2), "| stressed share of days:",
          round((reg.regime == "Stressed").mean(), 3))
    print("persistence p[calm->calm], p[stress->stress]:",
          res.params.filter(like="p[").round(3).to_dict())

    # Does diversification hold up when it is needed? Average pairwise correlation by regime.
    tickers = data.stocks(rets)
    rows = []
    for name, days in reg.groupby("regime").groups.items():
        block = rets.loc[days, tickers].dropna(axis=1, thresh=int(len(days) * 0.9))
        c = block.corr().values
        n = c.shape[0]
        rows.append({"regime": name, "days": len(days), "stocks": n,
                     "avg_pairwise_corr": (np.nansum(c) - n) / (n * (n - 1)),
                     "index_ann_vol": reg.loc[days, "index_ret"].std() * np.sqrt(252),
                     "index_ann_ret": (1 + reg.loc[days, "index_ret"]).prod() ** (252 / len(days)) - 1})
    rc = pd.DataFrame(rows)
    rc.to_parquet(PROCESSED / "regime_corr.parquet")
    print(rc.round(3).to_string(index=False))

    # stressed spells longer than 20 days
    s = (reg.regime == "Stressed").astype(int)
    spell = (s.diff() != 0).cumsum()
    spells = reg[s == 1].groupby(spell[s == 1]).apply(
        lambda g: pd.Series({"start": g.index.min().date(), "end": g.index.max().date(), "days": len(g)}))
    print(spells[spells.days > 20].to_string(index=False))


if __name__ == "__main__":
    main()
