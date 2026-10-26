"""One table for every strategy and portfolio, with bootstrap ranges and a regime split.

Sharpe ratios on ~10 years of daily data are noisy. The stationary bootstrap resamples
blocks of days (same days for a strategy and its benchmark) to get a 95% range for each
Sharpe and for the Sharpe gap vs buy-and-hold.

Outputs (data/processed/): summary.parquet, by_regime.parquet, yearly.parquet, daily_all.parquet
"""
import numpy as np
import pandas as pd

from config import PROCESSED, SEED, TRADING_DAYS
from metrics import RF_DAILY, stationary_bootstrap_index, summary

BENCH = "Buy and hold (Nifty ETF)"
SAME_STOCKS = "Equal weight"  # fair comparator: same survivor universe, no signal
REPS = 2000
BLOCK = 21


def boot_sharpe(ex: np.ndarray, idx: np.ndarray) -> np.ndarray:
    s = ex[idx]
    return s.mean(axis=1) / s.std(axis=1, ddof=1) * np.sqrt(TRADING_DAYS)


def main():
    s = pd.read_parquet(PROCESSED / "strategies.parquet")
    p = pd.read_parquet(PROCESSED / "portfolios.parquet")
    p = p[p.strategy != "Equal weight"].assign(group="Portfolio")
    ew = pd.read_parquet(PROCESSED / "portfolios.parquet").query("strategy == 'Equal weight'")
    allx = pd.concat([s.assign(group="Strategy"), ew.assign(group="Portfolio"), p])
    allx.loc[allx.strategy == BENCH, "group"] = "Benchmark"
    start = max(allx.groupby("strategy").date.min())
    allx = allx[allx.date >= start]
    wide = allx.pivot(index="date", columns="strategy", values="ret").dropna()
    allx.to_parquet(PROCESSED / "daily_all.parquet")

    rng = np.random.default_rng(SEED)
    idx = stationary_bootstrap_index(len(wide), REPS, BLOCK, rng)
    ex = wide.values - RF_DAILY
    boots = {c: boot_sharpe(ex[:, j], idx) for j, c in enumerate(wide.columns)}

    rows = []
    groups = allx.drop_duplicates("strategy").set_index("strategy").group
    for c in wide.columns:
        g = allx[allx.strategy == c]
        st = summary(wide[c])
        diff = boots[c] - boots[BENCH]
        diff_ew = boots[c] - boots[SAME_STOCKS]
        rows.append({
            "strategy": c, "group": groups[c], **st,
            "sharpe_lo": np.percentile(boots[c], 2.5), "sharpe_hi": np.percentile(boots[c], 97.5),
            "sharpe_gap_vs_bh": st["sharpe"] - summary(wide[BENCH])["sharpe"],
            "gap_lo": np.percentile(diff, 2.5), "gap_hi": np.percentile(diff, 97.5),
            "share_beats_bh": float((diff > 0).mean()),
            "sharpe_gap_vs_ew": st["sharpe"] - summary(wide[SAME_STOCKS])["sharpe"],
            "gap_ew_lo": np.percentile(diff_ew, 2.5), "gap_ew_hi": np.percentile(diff_ew, 97.5),
            "share_beats_ew": float((diff_ew > 0).mean()),
            "turnover_per_year": g.turnover.sum() / (len(g) / TRADING_DAYS),
            "cost_per_year": g.cost.sum() / (len(g) / TRADING_DAYS),
        })
    summ = pd.DataFrame(rows).sort_values("sharpe", ascending=False)
    summ.to_parquet(PROCESSED / "summary.parquet")

    reg = pd.read_parquet(PROCESSED / "regimes.parquet")[["regime"]]
    j = wide.join(reg, how="left")
    by = []
    for name, g in j.groupby("regime"):
        for c in wide.columns:
            r = g[c]
            by.append({"strategy": c, "regime": name, "days": len(r),
                       "ann_return": (1 + r).prod() ** (TRADING_DAYS / len(r)) - 1,
                       "ann_vol": r.std() * np.sqrt(TRADING_DAYS),
                       "worst_day": r.min()})
    pd.DataFrame(by).to_parquet(PROCESSED / "by_regime.parquet")

    yearly = (1 + wide).groupby(wide.index.year).prod() - 1
    yearly.index.name = "year"
    yearly.stack().rename("ret").reset_index().to_parquet(PROCESSED / "yearly.parquet")

    pd.set_option("display.width", 200)
    print(f"window {wide.index.min().date()} to {wide.index.max().date()} ({len(wide)} days)")
    print(summ[["strategy", "cagr", "ann_vol", "sharpe", "sharpe_lo", "sharpe_hi", "max_dd",
                "gap_lo", "gap_hi", "share_beats_bh", "gap_ew_lo", "gap_ew_hi", "share_beats_ew",
                "turnover_per_year"]]
          .round(3).to_string(index=False))
    print(pd.DataFrame(by).pivot(index="strategy", columns="regime", values="ann_return").round(3))


if __name__ == "__main__":
    main()
