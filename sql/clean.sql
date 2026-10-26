-- Raw files in, clean daily returns out.

CREATE OR REPLACE TABLE constituents AS
SELECT * FROM read_parquet('data/raw/constituents.parquet');

CREATE OR REPLACE TABLE prices_raw AS
SELECT * FROM read_parquet('data/raw/prices.parquet');

CREATE OR REPLACE TABLE price_fixes AS
SELECT * FROM read_csv('data/price_fixes.csv', header = true,
    columns = {'ticker': 'VARCHAR', 'date_from': 'DATE', 'date_to': 'DATE',
               'action': 'VARCHAR', 'factor': 'DOUBLE', 'reason': 'VARCHAR'});

-- Trading calendar = days the index has a close, so every stock return
-- covers the same interval as the index return.
CREATE OR REPLACE TABLE calendar AS
SELECT date FROM prices_raw WHERE ticker = '^NSEI' ORDER BY date;

-- Drop bad prints, rescale history where Yahoo applied an adjustment from the wrong date.
CREATE OR REPLACE TABLE prices_fixed AS
SELECT
    p.date,
    p.ticker,
    p.adj_close * coalesce((
        SELECT product(f.factor) FROM price_fixes f
        WHERE f.action = 'scale_before' AND f.ticker = p.ticker AND p.date < f.date_from
    ), 1) AS adj_close
FROM prices_raw p
WHERE NOT EXISTS (
    SELECT 1 FROM price_fixes f
    WHERE f.action = 'drop_row' AND f.ticker = p.ticker AND p.date BETWEEN f.date_from AND f.date_to
);

-- One row per stock per trading day from its first trade; a missing day carries the last price.
CREATE OR REPLACE TABLE prices_clean AS
WITH grid AS (
    SELECT c.date, s.ticker
    FROM calendar c
    JOIN (SELECT ticker, min(date) AS first_date FROM prices_fixed GROUP BY ticker) s
      ON c.date >= s.first_date
)
SELECT
    g.date,
    g.ticker,
    last_value(p.adj_close IGNORE NULLS) OVER (PARTITION BY g.ticker ORDER BY g.date) AS adj_close
FROM grid g
LEFT JOIN prices_fixed p USING (date, ticker);

-- Daily returns (demerger days set to flat) and growth of 1.
CREATE OR REPLACE TABLE returns AS
WITH r AS (
    SELECT
        c.date,
        c.ticker,
        c.adj_close,
        CASE WHEN f.ticker IS NOT NULL THEN 0.0
             ELSE c.adj_close / lag(c.adj_close) OVER (PARTITION BY c.ticker ORDER BY c.date) - 1
        END AS ret
    FROM prices_clean c
    LEFT JOIN price_fixes f
      ON f.action = 'flat_return' AND f.ticker = c.ticker AND f.date_from = c.date
)
SELECT
    date,
    ticker,
    adj_close,
    ret,
    exp(sum(ln(1 + ret)) OVER (PARTITION BY ticker ORDER BY date)) AS wealth
FROM r
WHERE ret IS NOT NULL;
