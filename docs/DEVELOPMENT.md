# Development Guide

## Main Modules

- `app.py` owns Flask routes and HTTP response handling.
- `stant/market_data.py` owns Yahoo Finance fetching, ETF symbol fallback, cache setup, and chart payload formatting.
- `stock_analyzer.py` is a CLI utility that reuses the same market-data resolver as the web app.
- `templates/index.html` contains the dashboard UI and browser-side chart logic.

## Local Commands

Install dependencies:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Run the app:

```powershell
.\.venv\Scripts\python.exe -m flask --app app run --host 127.0.0.1 --port 5000
```

Run tests:

```powershell
.\.venv\Scripts\python.exe -m unittest -v test_api_stock.py
```

## Testing Notes

The unit tests mock Yahoo Finance calls where possible so CI does not depend on live market data. For manual live checks, use a short date range:

```powershell
.\.venv\Scripts\python.exe -c "from app import app; c=app.test_client(); r=c.get('/api/stock?ticker=VUSA&interval=1d&start=2026-06-01&end=2026-06-10'); print(r.status_code); print(r.get_json().get('ticker'))"
```

## GitHub Notes

The repository ignores local virtual environments, Python caches, Yahoo Finance cache databases, generated CSVs, and generated plots. Keep source code, docs, tests, and workflow files committed.

If you want sample images or CSVs in the repository later, move them to a dedicated `examples/` folder and update `.gitignore` intentionally.

