# Tableau Public build guide

Four dashboards from the CSVs in `tableau/`. Built in Tableau Public Desktop, published to Tableau Public. About 2 hours.

## Setup

1. Open Tableau Public, **Connect > Text file**, pick `tableau/stock_metrics.csv`.
2. Add the other CSVs as **separate data sources** (Data > New Data Source), not joins. Each sheet uses one source.
3. In every source, check dates are Date type and rates are Number (decimal). Format rates as **Percentage, 1 decimal**.
4. Colours: one blue `#2a78d6` for single series. For a few named series use, in this order: blue `#2a78d6`, orange `#eb6834`, aqua `#1baf7a`, yellow `#eda100`. Grey `#b9b8b2` for "everything else".

## Dashboard 1: Stock risk explorer (`stock_metrics.csv`, `daily_stock.csv`)

| Sheet | Build |
|---|---|
| Risk vs return | Columns `ann_vol`, Rows `cagr`, Detail `symbol`, Color `cluster` (discrete), Label `symbol`. Filter `is_benchmark` = False, then add the two benchmark rows as a second mark layer in orange, or simply show them via a reference line at the ETF's CAGR (10.1%). Tooltip: sharpe, max_dd, beta, recovery_days |
| Drawdown leaders | Bar: `symbol` by `max_dd`, sorted ascending, top 15 |
| Beta vs R² | Columns `beta`, Rows `r2`, Label `symbol` |
| Growth of ₹1 | From `daily_stock.csv`: Columns `date` (exact, continuous), Rows `wealth` (log axis), Color `symbol`. Filter `symbol` (multi-select dropdown, default HDFCBANK + Nifty 50 ETF) |
| Drawdown path | Same filter, Rows `drawdown` as area |
| Rolling risk | Rows `vol_21d` and `beta_252d` as two separate sheets (never on a dual axis) |

Dashboard actions: clicking a dot in **Risk vs return** filters Growth, Drawdown and Rolling risk (Dashboard > Actions > Filter, target field `symbol`).

## Dashboard 2: Market regimes (`regimes.csv`, `regime_summary.csv`, `correlation.csv`)

| Sheet | Build |
|---|---|
| Index with regimes | Columns `date`, Rows `nifty_close` (line, blue). Add a second sheet underneath with Rows `stress_prob` as area, orange, 0 to 1 axis. Title: "Stress probability (Markov switching model)" |
| Average correlation | Columns `date`, Rows `avg_corr_63d` (line). Color by `regime` |
| Calm vs stressed | Text table from `regime_summary.csv`: days, avg_pairwise_corr, index_ann_vol, index_ann_ret |
| Correlation heatmap | `correlation.csv`: Columns `symbol_b`, Rows `symbol_a`, Color `rho` (sequential blue, 0 to 0.8), Mark = Square. Sort `symbol_a` by `order_a` (min, ascending) and `symbol_b` by `order_b` |

## Dashboard 3: VaR backtest (`var_backtest.csv`, `var_summary.csv`)

| Sheet | Build |
|---|---|
| Returns vs VaR | Columns `date`. Rows: `ret` as bars (grey) and a calculated field `-[var]` as a line (blue) in the **same** axis (Measure Values, not dual axis). Color the bars with calculated field `IF [breach] = 1 THEN "Breach" ELSE "Within VaR" END` (orange / grey). Filters: `portfolio`, `method`, `level` (single value dropdowns; defaults Nifty 50, GARCH-t, 0.99) |
| Test scorecard | Text table from `var_summary.csv`: method x level, showing breaches, expected, kupiec_p, christoffersen_p. Add a calculated field `IF [kupiec_pass] AND [christoffersen_pass] THEN "Pass" ELSE "Fail" END` and show it as text with an icon shape, not colour alone |

## Dashboard 4: Strategies and portfolios (`strategies_daily.csv`, `strategy_summary.csv`, `strategy_by_regime.csv`, `portfolio_weights.csv`, `sensitivity.csv`)

| Sheet | Build |
|---|---|
| Growth of ₹1 | Columns `date`, Rows `wealth` (log axis), Color `strategy`. Filter `strategy`, default: Buy and hold, Equal weight, Momentum, SMA 50/200 |
| Drawdowns | Same, Rows `drawdown` (area) |
| Sharpe with range | Rows `strategy` sorted by `sharpe`. Columns `sharpe_lo` as Gantt bar with size calculated field `[sharpe_hi] - [sharpe_lo]`; add `sharpe` as a circle on the same axis via Measure Values. Reference line at the buy-and-hold Sharpe (0.48) |
| Summary table | strategy, cagr, ann_vol, sharpe, max_dd, turnover_per_year, share_beats_ew |
| By regime | Highlight table: `strategy` x `regime`, value `ann_return`, diverging colour (orange negative, blue positive, grey zero) |
| Portfolio mix | `portfolio_weights.csv`: Columns `date`, Rows SUM(`weight`), Color `industry`, mark Area. Filter `method` (single value) |
| Sensitivity | `sensitivity.csv`, filter `family` = Momentum: Columns `p2`, Rows `p1`, Color `sharpe` (sequential blue), Label `sharpe` |

## Publish

1. File > Save to Tableau Public As > "Nifty 50 Risk, Regimes and Backtests".
2. On Tableau Public, set **Show sheets as tabs**, add the repo link in the description.
3. Paste the public link into the README (top line, replace "link added once published").
