# StAnT

**StAnT** is a lightweight **Stock Analysis Tool** for exploring historical market data in a browser. It uses Flask, Yahoo Finance data through `yfinance`, and TradingView Lightweight Charts for the interactive chart.

The app supports stocks and many ETFs. For non-US ETFs, it can automatically try common Yahoo Finance exchange suffixes. For example, entering `VUSA` resolves to `VUSA.L` when Yahoo Finance has data for the London-listed ETF.

## Features

- Interactive candlestick and line charts
- Volume overlay
- Date range filtering
- Daily, weekly, monthly, and intraday intervals
- ETF suffix fallback for symbols such as `VUSA`
- Summary metrics: last price, range, volume, and period return
- Optional CLI script for exporting CSV files and static plots

## Project Structure

```text
.
├── app.py                  # Flask web application entry point
├── stock_analyzer.py       # Command-line CSV and plot generator
├── stant/
│   ├── __init__.py         # Project metadata
│   └── market_data.py      # Yahoo Finance fetching and chart payload logic
├── templates/
│   └── index.html          # Web dashboard UI
├── test_api_stock.py       # API tests
├── requirements.txt        # Python dependencies
├── docs/
│   ├── API.md              # API endpoint reference
│   └── DEVELOPMENT.md      # Development notes
└── .github/workflows/
    └── tests.yml           # GitHub Actions test workflow
```

## Requirements

- Python 3.11 or newer
- Internet access for Yahoo Finance data
- Internet access in the browser for the TradingView Lightweight Charts CDN

## Setup

From PowerShell:

```powershell
cd C:\Users\ebrah\Documents\Projects\Stock_Analysis
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

If the `.venv` folder already exists, you can skip creating it and just install dependencies.

## Run The Web App

```powershell
.\.venv\Scripts\python.exe -m flask --app app run --host 127.0.0.1 --port 5000
```

Open this in your browser:

```text
http://127.0.0.1:5000
```

## Run Tests

```powershell
.\.venv\Scripts\python.exe -m unittest -v test_api_stock.py
```

## Publish To GitHub

If this folder is not already a valid Git repository, initialize it first:

```powershell
git init
git add .
git commit -m "Initial StAnT project"
```

Then create an empty GitHub repository and connect it:

```powershell
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPOSITORY.git
git push -u origin main
```

If `git init` complains because a broken `.git` folder already exists, move that folder out of the project first or recreate the project folder before initializing Git.

## CLI Usage

Generate a CSV file and plot for a ticker:

```powershell
.\.venv\Scripts\python.exe stock_analyzer.py AAPL
```

Generate ETF data using the fallback resolver:

```powershell
.\.venv\Scripts\python.exe stock_analyzer.py VUSA --start 2026-06-01 --end 2026-06-10
```

Generated CSV files go to `data/` and generated plots go to `plots/`.

## Notes

- Yahoo Finance intraday data has strict history limits. For example, `1m` data is usually limited to the last 7 days.
- This project is for learning and analysis. It is not financial advice.
- Generated caches, virtual environments, CSVs, and plots are ignored by Git so the repository stays clean.
