"""Write tidy CSVs for Tableau Public (one file per dashboard need). See docs/tableau_guide.md."""
import warnings

import numpy as np
import pandas as pd

import data
from config import INDEX, PROCESSED, TABLEAU, TR_BENCH

warnings.filterwarnings("ignore", category=UserWarning)

LABELS = {INDEX: "Nifty 50 index", TR_BENCH: "Nifty 50 ETF (NIFTYBEES)"}


def short(t: str) -> str:
    return LABELS.get(t, t.removesuffix(".NS"))


def main():
    TABLEAU.mkdir(parents=True, exist_ok=True)
    P = lambda name: pd.read_parquet(PROCESSED / f"{name}.parquet")  # noqa: E731

    # 1. One row per stock: risk, market model, cluster, sector
    m = data.query("SELECT * FROM stock_metrics")
    m = m.merge(P("capm"), on="ticker", how="left").merge(P("clusters")[["ticker", "cluster", "dendro_order"]],
                                                          on="ticker", how="left")
    m["symbol"] = m.ticker.map(short)
    m["is_benchmark"] = m.ticker.isin([INDEX, TR_BENCH])
    m["industry"] = m.industry.fillna("Benchmark")
    m.to_csv(TABLEAU / "stock_metrics.csv", index=False)

    # 2. Daily per stock: wealth, drawdown, rolling vol, rolling beta
    d = data.query("""
        SELECT r.date, r.ticker, r.ret, r.wealth, d.drawdown, v.vol_21d
        FROM returns r
        JOIN drawdowns d USING (date, ticker)
        LEFT JOIN rolling_vol v USING (date, ticker)
        ORDER BY r.ticker, r.date""")
    rb = P("rolling_beta")
    rb["date"] = pd.to_datetime(rb.date)
    d["date"] = pd.to_datetime(d.date)
    d = d.merge(rb, on=["date", "ticker"], how="left")
    d["symbol"] = d.ticker.map(short)
    d.round(6).to_csv(TABLEAU / "daily_stock.csv", index=False)

    # 3. Correlation matrix (last 3 years) in long form, with dendrogram order for sorting
    c = P("corr")
    cl = P("clusters").set_index("ticker")
    c["order_a"] = c.ticker_a.map(cl.dendro_order)
    c["order_b"] = c.ticker_b.map(cl.dendro_order)
    c["cluster_a"] = c.ticker_a.map(cl.cluster)
    c["symbol_a"] = c.ticker_a.map(short)
    c["symbol_b"] = c.ticker_b.map(short)
    c.round(4).to_csv(TABLEAU / "correlation.csv", index=False)

    # 4. VaR backtest (daily) and test results
    v = P("var_backtest")
    v["method"] = v.method.map({"hist": "Historical", "normal": "Normal", "garch": "GARCH-t"})
    v.round(6).to_csv(TABLEAU / "var_backtest.csv", index=False)
    vs = P("var_summary")
    vs["method"] = vs.method.map({"hist": "Historical", "normal": "Normal", "garch": "GARCH-t"})
    vs["kupiec_pass"] = vs.kupiec_p > 0.05
    vs["christoffersen_pass"] = vs.christoffersen_p > 0.05
    vs.round(4).to_csv(TABLEAU / "var_summary.csv", index=False)

    # 5. Strategies and portfolios: daily wealth and drawdown, summary, by regime, yearly
    a = P("daily_all").sort_values(["strategy", "date"])
    a["wealth"] = a.groupby("strategy").ret.transform(lambda r: (1 + r).cumprod())
    a["drawdown"] = a.groupby("strategy").wealth.transform(lambda w: w / w.cummax() - 1)
    reg = P("regimes")[["regime", "stress_prob", "stress_prob_wf"]].reset_index()
    a = a.merge(reg, on="date", how="left")
    a[["date", "strategy", "group", "ret", "wealth", "drawdown", "turnover", "cost", "regime"]] \
        .round(6).to_csv(TABLEAU / "strategies_daily.csv", index=False)
    P("summary").round(4).to_csv(TABLEAU / "strategy_summary.csv", index=False)
    P("by_regime").round(4).to_csv(TABLEAU / "strategy_by_regime.csv", index=False)
    P("yearly").round(4).to_csv(TABLEAU / "strategy_yearly.csv", index=False)
    P("sensitivity").round(4).to_csv(TABLEAU / "sensitivity.csv", index=False)

    # 6. Portfolio weights at each rebalance, with sector
    w = P("weights")
    w = w.merge(data.constituents()[["ticker", "industry"]], on="ticker", how="left")
    w["symbol"] = w.ticker.map(short)
    w.round(6).to_csv(TABLEAU / "portfolio_weights.csv", index=False)

    # 7. Regimes: index level, stress probabilities, average correlation
    close = data.index_close().rename("nifty_close")
    rg = P("regimes").join(close).join(P("avg_corr"), how="left")
    rg["avg_corr_63d"] = rg.avg_corr_63d.ffill()
    rg.reset_index().round(6).to_csv(TABLEAU / "regimes.csv", index=False)
    P("regime_corr").round(4).to_csv(TABLEAU / "regime_summary.csv", index=False)

    for f in sorted(TABLEAU.glob("*.csv")):
        print(f"{f.name:28s} {sum(1 for _ in open(f)) - 1:>8,} rows  {f.stat().st_size / 1e6:6.2f} MB")


if __name__ == "__main__":
    main()
