"""Read clean tables from DuckDB into pandas."""
import duckdb
import pandas as pd

from config import DB_PATH, INDEX, TR_BENCH


def query(sql: str) -> pd.DataFrame:
    with duckdb.connect(str(DB_PATH), read_only=True) as con:
        return con.execute(sql).df()


def returns_wide() -> pd.DataFrame:
    """Date x ticker matrix of daily simple returns (NaN before a stock listed)."""
    df = query("SELECT date, ticker, ret FROM returns")
    wide = df.pivot(index="date", columns="ticker", values="ret").sort_index()
    wide.index = pd.to_datetime(wide.index)
    return wide


def index_close() -> pd.Series:
    df = query(f"SELECT date, adj_close FROM prices_clean WHERE ticker = '{INDEX}' ORDER BY date")
    return pd.Series(df.adj_close.values, index=pd.to_datetime(df.date), name="index")


def stocks(wide: pd.DataFrame) -> list[str]:
    return [c for c in wide.columns if c not in (INDEX, TR_BENCH)]


def constituents() -> pd.DataFrame:
    return query("SELECT ticker, company, industry FROM constituents")
