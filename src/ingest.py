"""Download Nifty 50 constituents and daily prices to Parquet.

Constituents: NSE's published list (today's members, so survivorship bias applies).
Prices: Yahoo Finance via yfinance, unadjusted close and adjusted close kept side by side.
"""
import warnings
from datetime import date, timedelta

import pandas as pd
import requests
import yfinance as yf

from config import END, INDEX, RAW, START, TR_BENCH

warnings.filterwarnings("ignore")

CONSTITUENTS_URL = "https://nsearchives.nseindia.com/content/indices/ind_nifty50list.csv"


def fetch_constituents() -> pd.DataFrame:
    path = RAW / "ind_nifty50list.csv"
    if not path.exists():
        r = requests.get(CONSTITUENTS_URL, headers={"User-Agent": "Mozilla/5.0"}, timeout=30)
        r.raise_for_status()
        path.write_bytes(r.content)
    df = pd.read_csv(path)
    df.columns = ["company", "industry", "symbol", "series", "isin"]
    df["ticker"] = df["symbol"] + ".NS"
    return df


def fetch_prices(tickers: list[str]) -> pd.DataFrame:
    end_exclusive = (date.fromisoformat(END) + timedelta(days=1)).isoformat()
    wide = yf.download(tickers, start=START, end=end_exclusive, auto_adjust=False,
                       progress=False, group_by="ticker", threads=True)
    frames = []
    for t in tickers:
        d = wide[t].dropna(subset=["Adj Close"]).reset_index()
        d.columns = [c.lower().replace(" ", "_") for c in d.columns]
        d["ticker"] = t
        frames.append(d)
    out = pd.concat(frames, ignore_index=True)
    out["date"] = pd.to_datetime(out["date"]).dt.date
    return out[["date", "ticker", "open", "high", "low", "close", "adj_close", "volume"]]


def main():
    RAW.mkdir(parents=True, exist_ok=True)
    cons = fetch_constituents()
    cons.to_parquet(RAW / "constituents.parquet", index=False)
    tickers = cons["ticker"].tolist() + [INDEX, TR_BENCH]
    prices = fetch_prices(tickers)
    prices.to_parquet(RAW / "prices.parquet", index=False)
    print(f"constituents: {len(cons)}  price rows: {len(prices):,}  tickers: {prices.ticker.nunique()}")
    print(f"dates: {prices.date.min()} to {prices.date.max()}")


if __name__ == "__main__":
    main()
