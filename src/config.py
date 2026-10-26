"""Fixed settings for every step. Change them here, never inside a script."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"
DB_PATH = ROOT / "data" / "nifty.duckdb"
TABLEAU = ROOT / "tableau"
IMAGES = ROOT / "docs" / "images"

START = "2015-01-01"
END = "2026-10-23"  # last trading day used (inclusive)

INDEX = "^NSEI"  # Nifty 50 price index (no dividends): used for beta and regimes
TR_BENCH = "NIFTYBEES.NS"  # Nifty 50 ETF, dividends reinvested: used for return comparisons

EVAL_START = "2017-01-01"  # every headline number is out of sample from here (2 years of warm-up)

TRADING_DAYS = 252
RISK_FREE = 0.06  # stated assumption: 91-day T-bill, rough 2015-2026 average
COST_BPS = 10  # one-way cost per unit of turnover

SEED = 42
