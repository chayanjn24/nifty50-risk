"""Run the SQL layers in order and build data/nifty.duckdb."""
import os

import duckdb

from config import DB_PATH, RISK_FREE, ROOT

SQL_FILES = ["01_stage.sql", "02_clean.sql", "03_metrics.sql"]


def connect(read_only: bool = False) -> duckdb.DuckDBPyConnection:
    return duckdb.connect(str(DB_PATH), read_only=read_only)


def main():
    os.chdir(ROOT)  # SQL files use paths relative to the repo root
    con = connect()
    con.execute(f"SET VARIABLE rf = {RISK_FREE}")
    for name in SQL_FILES:
        con.execute((ROOT / "sql" / name).read_text())
        print(f"ran {name}")
    for t in ["prices_raw", "prices_clean", "returns", "stock_metrics"]:
        print(f"  {t}: {con.execute(f'SELECT count(*) FROM {t}').fetchone()[0]:,} rows")
    con.close()


if __name__ == "__main__":
    main()
