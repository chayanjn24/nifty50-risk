# Data dictionary

All returns are daily simple returns on adjusted close (splits and dividends), aligned to the Nifty 50 index calendar. Rates are decimals (0.12 = 12%). Annualised figures use 252 trading days.

## DuckDB tables (`data/nifty.duckdb`)

| Table | Grain | Key columns |
|---|---|---|
| `constituents` | 1 row per stock | company, industry (NSE), symbol, isin, ticker (`SYMBOL.NS`) |
| `prices_raw` | ticker x date as downloaded | open, high, low, close, adj_close, volume |
| `price_fixes` | 1 row per fix | ticker, date_from, date_to, action (`drop_row`, `scale_before`, `flat_return`), factor, reason |
| `calendar` | 1 row per index trading day | date, day_idx |
| `prices_fixed` | raw minus dropped rows, rescaled where needed | adj_close |
| `prices_clean` | ticker x calendar day from first trade | adj_close (gaps carry the last price), carried_forward |
| `returns` | ticker x day | ret, log_ret, wealth (growth of 1 from clean returns) |
| `drawdowns` | ticker x day | peak, drawdown (wealth / peak - 1), spell (underwater spell id) |
| `rolling_vol` | ticker x day | vol_21d (annualised) |
| `yearly_returns` | ticker x year | ret, days |
| `underwater` | 1 row per ticker | max_dd, trough_date, recovered_date (NULL = not yet), recovery_days, longest_underwater_days |
| `stock_metrics` | 1 row per ticker | cagr, ann_vol, sharpe, sortino, max_dd, calmar, worst_day, best_day, skew, excess_kurtosis |

## Tableau extracts (`tableau/`)

| File | Grain | Columns |
|---|---|---|
| `stock_metrics.csv` | ticker | everything in `stock_metrics` + beta, beta_t, alpha_ann, alpha_t, r2 (CAPM vs index, Newey-West), cluster (1-10), dendro_order, symbol, is_benchmark |
| `daily_stock.csv` | ticker x day | ret, wealth, drawdown, vol_21d, beta_252d, symbol |
| `correlation.csv` | stock pair | rho (last 3 years), order_a, order_b (sort both axes by these), cluster_a |
| `var_backtest.csv` | portfolio x method x level x day | var (positive number = loss threshold), ret, breach (1 if ret < -var) |
| `var_summary.csv` | portfolio x method x level | breaches, expected, breach_rate, kupiec_p, christoffersen_p, kupiec_pass, christoffersen_pass (p > 0.05) |
| `strategies_daily.csv` | strategy x day | group (Benchmark / Strategy / Portfolio), ret (net of costs), wealth, drawdown, turnover, cost, regime |
| `strategy_summary.csv` | strategy | cagr, ann_vol, sharpe, sharpe_lo, sharpe_hi (95% bootstrap), max_dd, sharpe gap vs buy-and-hold and vs equal weight with ranges, share_beats_bh, share_beats_ew, turnover_per_year, cost_per_year |
| `strategy_by_regime.csv` | strategy x regime | ann_return, ann_vol, worst_day, days |
| `strategy_yearly.csv` | strategy x year | ret |
| `sensitivity.csv` | parameter combination | family (SMA: p1 fast, p2 slow; Momentum: p1 lookback days, p2 stocks held; Regime: p1 threshold), sharpe |
| `portfolio_weights.csv` | method x rebalance date x stock | weight, industry, symbol |
| `regimes.csv` | day | index_ret, nifty_close, stress_prob (full sample, descriptive), stress_prob_wf (walk-forward, tradable), regime, avg_corr_63d |
| `regime_summary.csv` | regime | days, avg_pairwise_corr, index_ann_vol, index_ann_ret |

## Profiling notes

- 50 constituents + `^NSEI` + `NIFTYBEES.NS`; 144,371 raw rows; 143,897 clean daily returns.
- Late listings: IndiGo (Nov 2015), BSE (Feb 2017), SBI Life (Oct 2017), HDFC Life (Nov 2017), Max Healthcare (Aug 2020), Eternal (Jul 2021), Jio Financial (Aug 2023).
- 9 stock dates (Jan 1s, a few holidays, weekend muhurat sessions) exist for stocks but not for the index; they are skipped so every return spans the same interval as the index return.
- NIFTYBEES is missing 24 Oct 2025; the last price carries forward.
- Moves over 18% in a day were all checked; see the fixes table in the README.
