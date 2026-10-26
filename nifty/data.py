"""Settings, download, DuckDB build and loaders."""
import warnings
from datetime import date, timedelta
from pathlib import Path

import duckdb
import pandas as pd
import requests
import yfinance as yf

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
OUT = ROOT / "data" / "processed"
DB = ROOT / "data" / "nifty.duckdb"
TABLEAU = ROOT / "tableau"
IMAGES = ROOT / "images"

START, END = "2015-01-01", "2026-10-23"
EVAL_START = "2017-01-01"  # strategy results are out of sample from here
INDEX = "^NSEI"  # Nifty 50 price index: beta and regimes
ETF = "NIFTYBEES.NS"  # Nifty 50 ETF, dividends kept in the price: buy-and-hold benchmark
RISK_FREE = 0.06  # assumed 91-day T-bill rate
COST_BPS = 10
DAYS = 252
SEED = 42

CONSTITUENTS_URL = "https://nsearchives.nseindia.com/content/indices/ind_nifty50list.csv"


def download():
    RAW.mkdir(parents=True, exist_ok=True)
    r = requests.get(CONSTITUENTS_URL, headers={"User-Agent": "Mozilla/5.0"}, timeout=30)
    r.raise_for_status()
    (RAW / "ind_nifty50list.csv").write_bytes(r.content)
    cons = pd.read_csv(RAW / "ind_nifty50list.csv")
    cons.columns = ["company", "industry", "symbol", "series", "isin"]
    cons["ticker"] = cons.symbol + ".NS"
    cons.to_parquet(RAW / "constituents.parquet", index=False)

    tickers = cons.ticker.tolist() + [INDEX, ETF]
    end = (date.fromisoformat(END) + timedelta(days=1)).isoformat()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        wide = yf.download(tickers, start=START, end=end, auto_adjust=False,
                           progress=False, group_by="ticker")
    frames = []
    for t in tickers:
        d = wide[t].dropna(subset=["Adj Close"]).reset_index()
        d.columns = [c.lower().replace(" ", "_") for c in d.columns]
        frames.append(d.assign(ticker=t))
    prices = pd.concat(frames, ignore_index=True)
    prices["date"] = pd.to_datetime(prices.date).dt.date
    prices = prices[["date", "ticker", "open", "high", "low", "close", "adj_close", "volume"]]
    prices.to_parquet(RAW / "prices.parquet", index=False)
    print(f"downloaded {len(prices):,} price rows for {prices.ticker.nunique()} tickers")


def build_db():
    with duckdb.connect(str(DB)) as con:
        con.execute(f"SET file_search_path = '{ROOT}'")
        con.execute(f"SET VARIABLE rf = {RISK_FREE}")
        for name in ["clean.sql", "metrics.sql"]:
            con.execute((ROOT / "sql" / name).read_text())
        n = con.execute("SELECT count(*) FROM returns").fetchone()[0]
    print(f"built {DB.name}: {n:,} daily returns")


def query(sql: str) -> pd.DataFrame:
    with duckdb.connect(str(DB), read_only=True) as con:
        return con.execute(sql).df()


def returns_wide() -> pd.DataFrame:
    """Date x ticker matrix of daily returns (NaN before a stock listed)."""
    df = query("SELECT date, ticker, ret FROM returns")
    wide = df.pivot(index="date", columns="ticker", values="ret").sort_index()
    wide.index = pd.to_datetime(wide.index)
    return wide


def index_close() -> pd.Series:
    df = query(f"SELECT date, adj_close FROM prices_clean WHERE ticker = '{INDEX}' ORDER BY date")
    return pd.Series(df.adj_close.values, index=pd.to_datetime(df.date))


def stocks(wide: pd.DataFrame) -> list[str]:
    return [c for c in wide.columns if c not in (INDEX, ETF)]
