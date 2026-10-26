"""Run the whole project: python run.py  (add --no-download to reuse saved prices)."""
import sys

from nifty import data, report, risk, strategies

if __name__ == "__main__":
    if "--no-download" not in sys.argv:
        data.download()
    data.build_db()
    risk.run()
    strategies.run()
    report.run()
