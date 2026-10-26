PY := .venv/bin/python

.PHONY: setup data db analyse tableau charts test all

setup:
	python3 -m venv .venv && .venv/bin/pip install -r requirements.txt

data:
	cd src && ../$(PY) ingest.py

db:
	cd src && ../$(PY) build_db.py

analyse:
	cd src && ../$(PY) risk.py && ../$(PY) regimes.py && ../$(PY) backtest.py \
		&& ../$(PY) portfolio.py && ../$(PY) evaluate.py

tableau:
	cd src && ../$(PY) export_tableau.py

charts:
	cd src && ../$(PY) charts.py

test:
	$(PY) -m pytest -q

all: db analyse tableau charts test
