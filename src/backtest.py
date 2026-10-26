"""Rule-based strategies vs buy-and-hold, with a one-day signal lag and 10 bps costs.

  Buy and hold     NIFTYBEES (Nifty 50 ETF, dividends inside the price)
  SMA 50/200       hold the ETF while the index 50-day average is above the 200-day, else cash
  SMA walk-forward each January, pick the SMA pair with the best Sharpe on all earlier data
  Momentum 12-1    each month hold the 10 stocks with the best return from 12 months to 1 month ago
  Regime filter    hold the ETF unless the walk-forward Markov model says "stressed" (prob > 0.5)

Outputs (data/processed/): strategies.parquet (daily), sensitivity.parquet, wf_choices.parquet
"""
import itertools

import numpy as np
import pandas as pd

import data
from config import EVAL_START, PROCESSED, TR_BENCH
from metrics import sharpe
from sim import CASH, month_starts, simulate

SMA_FAST = [20, 50, 100]
SMA_SLOW = [100, 150, 200, 250]
MOM_LOOKBACK = [63, 126, 189, 252]  # 3, 6, 9, 12 months
MOM_TOP = [5, 10, 15, 20]
MOM_SKIP = 21
REGIME_THRESHOLDS = [0.3, 0.5, 0.7, 0.9]  # exit to cash when stress probability is at or above this


def timing_targets(signal: pd.Series) -> pd.DataFrame:
    """Daily target weights for an ETF-or-cash rule."""
    s = signal.astype(float)
    return pd.DataFrame({TR_BENCH: s, CASH: 1 - s})


def sma_signal(close: pd.Series, fast: int, slow: int) -> pd.Series:
    sig = close.rolling(fast).mean() > close.rolling(slow).mean()
    return sig[close.rolling(slow).count() == slow]


def sma_strategy(rets, close, fast, slow):
    return simulate(timing_targets(sma_signal(close, fast, slow)), rets[[TR_BENCH]])


def sma_walk_forward(rets, close) -> tuple[pd.DataFrame, pd.DataFrame]:
    pairs = [(f, s) for f, s in itertools.product(SMA_FAST, SMA_SLOW) if f < s]
    runs = {p: sma_strategy(rets, close, *p) for p in pairs}
    pieces, choices = [], []
    for year in range(pd.Timestamp(EVAL_START).year, rets.index.max().year + 1):
        # rank pairs on everything before this year (each run only uses past prices)
        scores = {p: sharpe(run.ret[run.index.year < year].dropna()) for p, run in runs.items()}
        best = max(scores, key=scores.get)
        choices.append({"year": year, "fast": best[0], "slow": best[1], "train_sharpe": scores[best]})
        pieces.append(runs[best][runs[best].index.year == year])
    return pd.concat(pieces), pd.DataFrame(choices)


def momentum_targets(rets: pd.DataFrame, tickers, lookback=252, top=10) -> pd.DataFrame:
    wealth = (1 + rets[tickers].fillna(0)).cumprod()
    listed = rets[tickers].notna().rolling(lookback + 1).sum() == lookback + 1
    rows = {}
    for d in month_starts(rets.index):
        i = rets.index.get_loc(d)
        if i < lookback + 1:
            continue
        score = wealth.iloc[i - MOM_SKIP] / wealth.iloc[i - lookback] - 1
        score = score[listed.iloc[i]]
        pick = score.nlargest(top).index
        rows[d] = pd.Series(1 / len(pick), index=pick)
    return pd.DataFrame(rows).T.fillna(0.0)


def main():
    PROCESSED.mkdir(parents=True, exist_ok=True)
    rets = data.returns_wide()
    close = data.index_close()
    tickers = data.stocks(rets)
    regimes = pd.read_parquet(PROCESSED / "regimes.parquet")

    out = {}
    bh = pd.DataFrame({TR_BENCH: 1.0}, index=rets.index[:1])
    out["Buy and hold (Nifty ETF)"] = simulate(bh, rets[[TR_BENCH]])
    out["SMA 50/200"] = sma_strategy(rets, close, 50, 200)
    wf, choices = sma_walk_forward(rets, close)
    out["SMA walk-forward"] = wf
    out["Momentum 12-1, top 10"] = simulate(momentum_targets(rets, tickers), rets[tickers])
    stress = regimes["stress_prob_wf"].dropna()
    out["Regime filter"] = simulate(timing_targets(stress < 0.5), rets[[TR_BENCH]])

    frames = []
    for name, df in out.items():
        df = df[df.index >= EVAL_START]
        frames.append(df.assign(strategy=name).rename_axis("date").reset_index())
    pd.concat(frames).to_parquet(PROCESSED / "strategies.parquet")
    choices.to_parquet(PROCESSED / "wf_choices.parquet")
    print(choices.to_string(index=False))

    # Sensitivity: every parameter combination over the same window. Shown as a range, never as a headline.
    sens = []
    for f, s in itertools.product(SMA_FAST, SMA_SLOW):
        if f < s:
            r = sma_strategy(rets, close, f, s).ret
            sens.append({"family": "SMA", "p1": f, "p2": s, "sharpe": sharpe(r[r.index >= EVAL_START])})
    for lb, top in itertools.product(MOM_LOOKBACK, MOM_TOP):
        r = simulate(momentum_targets(rets, tickers, lb, top), rets[tickers]).ret
        sens.append({"family": "Momentum", "p1": lb, "p2": top, "sharpe": sharpe(r[r.index >= EVAL_START])})
    for th in REGIME_THRESHOLDS:
        r = simulate(timing_targets(stress < th), rets[[TR_BENCH]]).ret
        sens.append({"family": "Regime", "p1": th, "p2": 0, "sharpe": sharpe(r[r.index >= EVAL_START])})
    sens = pd.DataFrame(sens)
    sens.to_parquet(PROCESSED / "sensitivity.parquet")
    print(sens.pivot_table(index=["family", "p1"], columns="p2", values="sharpe").round(2))

    for name, df in out.items():
        r = df.ret[df.index >= EVAL_START]
        print(f"{name:28s} sharpe {sharpe(r):.2f}  cagr {(1 + r).prod() ** (252 / len(r)) - 1:.3f}"
              f"  turnover/yr {df.turnover[df.index >= EVAL_START].sum() / (len(r) / 252):.1f}")


if __name__ == "__main__":
    main()
