-- Raw layer: files as downloaded, plus the hand-checked fixes list.
CREATE OR REPLACE TABLE constituents AS
SELECT * FROM read_parquet('data/raw/constituents.parquet');

CREATE OR REPLACE TABLE prices_raw AS
SELECT * FROM read_parquet('data/raw/prices.parquet');

CREATE OR REPLACE TABLE price_fixes AS
SELECT * FROM read_csv('data/price_fixes.csv', header = true,
    columns = {'ticker': 'VARCHAR', 'date_from': 'DATE', 'date_to': 'DATE',
               'action': 'VARCHAR', 'factor': 'DOUBLE', 'reason': 'VARCHAR'});

-- Master calendar = days the Nifty 50 index has a close. Stock-only days
-- (weekend muhurat sessions, a few holidays Yahoo lacks for the index) are skipped,
-- so every stock return covers exactly the same interval as the index return.
CREATE OR REPLACE TABLE calendar AS
SELECT date, row_number() OVER (ORDER BY date) AS day_idx
FROM prices_raw
WHERE ticker = '^NSEI'
ORDER BY date;
