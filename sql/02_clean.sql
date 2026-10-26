-- Clean layer: apply fixes, align to the calendar, compute returns.

-- 1. Drop bad prints and rescale history where an adjustment was applied from the wrong date.
CREATE OR REPLACE TABLE prices_fixed AS
SELECT
    p.date,
    p.ticker,
    p.close,
    p.adj_close * coalesce((
        SELECT product(f.factor)
        FROM price_fixes f
        WHERE f.action = 'scale_before' AND f.ticker = p.ticker AND p.date < f.date_from
    ), 1) AS adj_close,
    p.volume
FROM prices_raw p
WHERE NOT EXISTS (
    SELECT 1 FROM price_fixes f
    WHERE f.action = 'drop_row' AND f.ticker = p.ticker
      AND p.date BETWEEN f.date_from AND f.date_to
);

-- 2. One row per ticker per calendar day from its first trade; gaps carry the last price.
CREATE OR REPLACE TABLE prices_clean AS
WITH first_seen AS (
    SELECT ticker, min(date) AS first_date FROM prices_fixed GROUP BY ticker
),
grid AS (
    SELECT c.date, c.day_idx, f.ticker
    FROM calendar c
    JOIN first_seen f ON c.date >= f.first_date
)
SELECT
    g.date,
    g.day_idx,
    g.ticker,
    last_value(p.adj_close IGNORE NULLS) OVER (
        PARTITION BY g.ticker ORDER BY g.date
        ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
    ) AS adj_close,
    p.adj_close IS NULL AS carried_forward,
    p.volume
FROM grid g
LEFT JOIN prices_fixed p USING (date, ticker);

-- 3. Daily simple and log returns, plus a wealth index rebuilt from clean returns.
CREATE OR REPLACE TABLE returns AS
WITH r AS (
    SELECT
        c.date,
        c.ticker,
        c.adj_close,
        CASE WHEN f.ticker IS NOT NULL THEN 0.0
             ELSE c.adj_close / lag(c.adj_close) OVER w - 1 END AS ret
    FROM prices_clean c
    LEFT JOIN price_fixes f
        ON f.action = 'flat_return' AND f.ticker = c.ticker AND f.date_from = c.date
    WINDOW w AS (PARTITION BY c.ticker ORDER BY c.date)
)
SELECT
    date,
    ticker,
    adj_close,
    ret,
    ln(1 + ret) AS log_ret,
    exp(sum(ln(1 + ret)) OVER (PARTITION BY ticker ORDER BY date)) AS wealth
FROM r
WHERE ret IS NOT NULL;
