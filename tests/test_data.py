"""Checks on the built database. Skipped on a fresh clone until `make data db` has run."""
import pandas as pd
import pytest

from config import DB_PATH

pytestmark = pytest.mark.skipif(not DB_PATH.exists(), reason="run `make data db` first")


@pytest.fixture(scope="module")
def q():
    import data
    return data.query


def test_one_row_per_ticker_per_day(q):
    dupes = q("SELECT date, ticker, count(*) n FROM returns GROUP BY ALL HAVING n > 1")
    assert dupes.empty


def test_wealth_matches_price_ratio_where_no_fix(q):
    # cumulative return must equal last / first adjusted price, except where a fix was applied
    df = q("""
        WITH ends AS (
            SELECT ticker, arg_min(adj_close, date) p0, arg_max(adj_close, date) p1,
                   arg_max(wealth, date) w
            FROM prices_clean JOIN returns USING (date, ticker, adj_close)
            GROUP BY ticker),
        first_px AS (SELECT ticker, arg_min(adj_close, date) p_first FROM prices_clean GROUP BY ticker)
        SELECT e.ticker, e.p1 / f.p_first AS ratio, e.w
        FROM ends e JOIN first_px f USING (ticker)
        WHERE e.ticker NOT IN (SELECT ticker FROM price_fixes WHERE action = 'flat_return')""")
    assert (abs(df.ratio / df.w - 1) < 1e-9).all()


def test_fixed_days_have_no_fake_crash(q):
    df = q("""SELECT r.ticker, r.date, r.ret FROM returns r
              JOIN price_fixes f ON f.ticker = r.ticker AND r.date BETWEEN f.date_from AND f.date_to""")
    assert (df.ret.abs() < 0.05).all()


def test_sql_metrics_match_python(q):
    from metrics import ann_vol, cagr, max_drawdown
    r = q("SELECT date, ret FROM returns WHERE ticker = 'HDFCBANK.NS' ORDER BY date").set_index("date").ret
    m = q("SELECT * FROM stock_metrics WHERE ticker = 'HDFCBANK.NS'").iloc[0]
    assert m.cagr == pytest.approx(cagr(r), rel=1e-9)
    assert m.ann_vol == pytest.approx(ann_vol(r), rel=1e-9)
    assert m.max_dd == pytest.approx(max_drawdown(r), rel=1e-9)


def test_all_fifty_constituents_present(q):
    n = q("SELECT count(DISTINCT ticker) n FROM returns WHERE ticker LIKE '%.NS' AND ticker <> 'NIFTYBEES.NS'")
    assert n.n.iloc[0] == 50
