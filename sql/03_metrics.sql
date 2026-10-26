-- Risk layer: drawdowns, rolling volatility, yearly returns, one summary row per ticker.
-- getvariable('rf') = annual risk-free rate, set by build_db.py from config.py.

CREATE OR REPLACE TABLE drawdowns AS
WITH peaks AS (
    SELECT *,
        max(wealth) OVER (PARTITION BY ticker ORDER BY date) AS peak,
        -- a new "underwater spell" starts every time wealth sets a new high
        count(*) FILTER (WHERE wealth >= running_high) OVER (PARTITION BY ticker ORDER BY date) AS spell
    FROM (
        SELECT *, max(wealth) OVER (PARTITION BY ticker ORDER BY date) AS running_high FROM returns
    )
)
SELECT date, ticker, wealth, peak, wealth / peak - 1 AS drawdown, spell
FROM peaks;

CREATE OR REPLACE TABLE rolling_vol AS
SELECT
    date,
    ticker,
    stddev_samp(ret) OVER w * sqrt(252) AS vol_21d,
    count(*) OVER w AS n
FROM returns
WINDOW w AS (PARTITION BY ticker ORDER BY date ROWS BETWEEN 20 PRECEDING AND CURRENT ROW)
QUALIFY n = 21;

CREATE OR REPLACE TABLE yearly_returns AS
SELECT ticker, year(date) AS year, exp(sum(log_ret)) - 1 AS ret, count(*) AS days
FROM returns
GROUP BY ALL;

-- Longest time under water, and how long the worst drawdown took to recover.
CREATE OR REPLACE TABLE underwater AS
WITH spells AS (
    SELECT ticker, spell, min(date) AS start_date, max(date) AS end_date,
           count(*) - 1 AS days_under, min(drawdown) AS depth
    FROM drawdowns
    GROUP BY ticker, spell
),
worst AS (
    SELECT ticker, arg_min(date, drawdown) AS trough_date, min(drawdown) AS max_dd
    FROM drawdowns GROUP BY ticker
),
recovery AS (
    SELECT w.ticker, w.trough_date, w.max_dd,
        (SELECT min(d.date) FROM drawdowns d
         WHERE d.ticker = w.ticker AND d.date > w.trough_date AND d.drawdown >= 0) AS recovered_date
    FROM worst w
)
SELECT
    r.ticker, r.max_dd, r.trough_date, r.recovered_date,
    CASE WHEN r.recovered_date IS NOT NULL THEN
        (SELECT count(*) FROM calendar c WHERE c.date > r.trough_date AND c.date <= r.recovered_date)
    END AS recovery_days,  -- NULL = not yet recovered
    (SELECT max(days_under) FROM spells s WHERE s.ticker = r.ticker) AS longest_underwater_days
FROM recovery r;

CREATE OR REPLACE TABLE stock_metrics AS
WITH rf AS (SELECT pow(1 + getvariable('rf'), 1 / 252) - 1 AS rf_d),
base AS (
    SELECT
        ticker,
        min(date) AS first_date,
        max(date) AS last_date,
        count(*) AS n_days,
        arg_max(wealth, date) AS final_wealth,
        avg(ret) AS mean_d,
        stddev_samp(ret) AS sd_d,
        sqrt(avg(pow(least(ret - (SELECT rf_d FROM rf), 0), 2))) AS downside_d,
        min(ret) AS worst_day,
        max(ret) AS best_day,
        skewness(ret) AS skew,
        kurtosis(ret) AS excess_kurtosis
    FROM returns
    GROUP BY ticker
)
SELECT
    b.ticker,
    c.company,
    c.industry,
    b.first_date,
    b.last_date,
    b.n_days,
    pow(b.final_wealth, 252.0 / b.n_days) - 1 AS cagr,
    b.sd_d * sqrt(252) AS ann_vol,
    (b.mean_d - rf.rf_d) / b.sd_d * sqrt(252) AS sharpe,
    (b.mean_d - rf.rf_d) / b.downside_d * sqrt(252) AS sortino,
    u.max_dd,
    u.trough_date,
    u.recovered_date,
    u.recovery_days,
    u.longest_underwater_days,
    (pow(b.final_wealth, 252.0 / b.n_days) - 1) / abs(u.max_dd) AS calmar,
    b.worst_day,
    b.best_day,
    b.skew,
    b.excess_kurtosis
FROM base b
CROSS JOIN rf
JOIN underwater u USING (ticker)
LEFT JOIN constituents c ON c.ticker = b.ticker;
