-- Drawdowns, rolling volatility and one risk summary row per stock.
-- getvariable('rf') is the annual risk-free rate, set from Python.

CREATE OR REPLACE TABLE drawdowns AS
SELECT
    date,
    ticker,
    wealth,
    wealth / max(wealth) OVER w - 1 AS drawdown,
    -- a new underwater spell starts each time wealth sets a new high
    count(*) FILTER (WHERE wealth >= high) OVER w AS spell
FROM (SELECT *, max(wealth) OVER (PARTITION BY ticker ORDER BY date) AS high FROM returns)
WINDOW w AS (PARTITION BY ticker ORDER BY date);

CREATE OR REPLACE TABLE rolling_vol AS
SELECT date, ticker, stddev_samp(ret) OVER w * sqrt(252) AS vol_21d
FROM returns
WINDOW w AS (PARTITION BY ticker ORDER BY date ROWS BETWEEN 20 PRECEDING AND CURRENT ROW)
QUALIFY count(*) OVER w = 21;

CREATE OR REPLACE TABLE underwater AS
WITH worst AS (
    SELECT ticker, arg_min(date, drawdown) AS trough_date, min(drawdown) AS max_dd
    FROM drawdowns GROUP BY ticker
),
recovered AS (
    SELECT w.*,
        (SELECT min(d.date) FROM drawdowns d
         WHERE d.ticker = w.ticker AND d.date > w.trough_date AND d.drawdown >= 0) AS recovered_date
    FROM worst w
)
SELECT
    r.*,
    CASE WHEN r.recovered_date IS NOT NULL THEN
        (SELECT count(*) FROM calendar c WHERE c.date > r.trough_date AND c.date <= r.recovered_date)
    END AS recovery_days,  -- NULL = not recovered yet
    (SELECT max(n) FROM (SELECT count(*) - 1 AS n FROM drawdowns d
                         WHERE d.ticker = r.ticker GROUP BY spell)) AS longest_underwater_days
FROM recovered r;

CREATE OR REPLACE TABLE stock_metrics AS
WITH base AS (
    SELECT
        ticker,
        min(date) AS first_date,
        count(*) AS n_days,
        arg_max(wealth, date) AS final_wealth,
        avg(ret) - (pow(1 + getvariable('rf'), 1 / 252) - 1) AS excess_d,
        stddev_samp(ret) AS sd_d,
        sqrt(avg(pow(least(ret - (pow(1 + getvariable('rf'), 1 / 252) - 1), 0), 2))) AS downside_d,
        min(ret) AS worst_day,
        max(ret) AS best_day
    FROM returns
    GROUP BY ticker
)
SELECT
    b.ticker,
    c.company,
    c.industry,
    b.first_date,
    b.n_days,
    pow(b.final_wealth, 252.0 / b.n_days) - 1 AS cagr,
    b.sd_d * sqrt(252) AS ann_vol,
    b.excess_d / b.sd_d * sqrt(252) AS sharpe,
    b.excess_d / b.downside_d * sqrt(252) AS sortino,
    u.max_dd,
    u.trough_date,
    u.recovered_date,
    u.recovery_days,
    u.longest_underwater_days,
    b.worst_day,
    b.best_day
FROM base b
JOIN underwater u USING (ticker)
LEFT JOIN constituents c USING (ticker);
