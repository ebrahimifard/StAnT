import os
import sys
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from stant.market_data import get_stock_data
from stant.strategies import run_strategy_analysis, STRATEGY_NAMES
from stant.backtest import backtest_strategy
from stant.multi_analysis import run_multi_stock_screener

def fetch_historical_data(ticker: str, interval: str = "1d", start: str = None, end: str = None) -> tuple[pd.DataFrame, str]:
    if start or end:
        print(f"Fetching historical data for ticker '{ticker}' (interval='{interval}', start={start}, end={end})...")
    else:
        print(f"Fetching historical data for ticker '{ticker}' (interval='{interval}', period='max')...")

    stock_data = get_stock_data(ticker, interval=interval, start=start, end=end)
    
    if stock_data.frame.empty:
        raise ValueError(
            f"No data returned for ticker '{ticker}'. Please check if the symbol is valid, "
            f"and that the selected interval ('{interval}') is compatible with the requested date range.\n"
            f"Tried Yahoo Finance symbols: {', '.join(stock_data.tried_symbols)}."
        )

    if stock_data.resolved_symbol != stock_data.requested_symbol:
        print(f"Resolved '{stock_data.requested_symbol}' to Yahoo Finance symbol '{stock_data.resolved_symbol}'.")
        
    return stock_data.frame, stock_data.resolved_symbol

def save_data(df: pd.DataFrame, ticker: str, data_dir: str) -> str:
    os.makedirs(data_dir, exist_ok=True)
    df_to_save = df.copy()
    if isinstance(df_to_save.index, pd.DatetimeIndex):
        df_to_save.index = df_to_save.index.tz_localize(None)
    
    file_path = os.path.join(data_dir, f"{ticker.upper()}_history.csv")
    df_to_save.to_csv(file_path)
    print(f"Historical data successfully saved to: {file_path}")
    return file_path

def plot_data(df: pd.DataFrame, ticker: str, plot_dir: str, show_plot: bool, interval: str = "1d", start: str = None, end: str = None) -> str:
    os.makedirs(plot_dir, exist_ok=True)
    ticker_upper = ticker.upper()
    
    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True, gridspec_kw={'height_ratios': [3, 1]})
    
    dates = df.index.tz_localize(None) if isinstance(df.index, pd.DatetimeIndex) else df.index
    ax1.plot(dates, df['Close'], label='Close Price', color='#0f9755', linewidth=1.5)
    
    unit = "period" if interval != "1d" else "day"
    if len(df) >= 50:
        ma50 = df['Close'].rolling(window=50).mean()
        ax1.plot(dates, ma50, label=f'50-{unit} SMA', color='#e67e22', linewidth=1.0, linestyle='--')
    if len(df) >= 200:
        ma200 = df['Close'].rolling(window=200).mean()
        ax1.plot(dates, ma200, label=f'200-{unit} SMA', color='#2980b9', linewidth=1.0, linestyle='--')
        
    period_str = f"{start or 'Start'} to {end or 'End'}" if start or end else "Full History"
    ax1.set_title(f"{ticker_upper} Historical Close Price & Volume ({period_str}, interval={interval})", fontsize=14, fontweight='bold', pad=15)
    ax1.set_ylabel("Price (USD)", fontsize=11, fontweight='semibold')
    ax1.legend(loc='upper left', frameon=True, facecolor='white', edgecolor='none')
    ax1.tick_params(labelsize=9)
    ax1.grid(True, linestyle=':', alpha=0.6)
    
    ax2.bar(dates, df['Volume'], color='#bdc3c7', alpha=0.7, width=1.0, label='Volume')
    ax2.set_ylabel("Volume", fontsize=11, fontweight='semibold')
    ax2.set_xlabel("Date", fontsize=11, fontweight='semibold')
    ax2.tick_params(labelsize=9)
    ax2.grid(True, linestyle=':', alpha=0.6)
    
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    ax2.xaxis.set_major_locator(mdates.AutoDateLocator())
    fig.autofmt_xdate()
    plt.tight_layout()
    
    plot_file = os.path.join(plot_dir, f"{ticker_upper}_plot.png")
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    print(f"Plot successfully saved to: {plot_file}")
    
    if show_plot:
        plt.show()
        
    return plot_file

def display_signals_report(ticker: str, analysis: dict):
    active = analysis["active_strategy"]
    ensemble = analysis["ensemble_consensus"]
    setup = active["trade_setup"]

    print("\n" + "=" * 65)
    print(f" [REPORT] TRADING STRATEGY & SIGNALS REPORT — {ticker.upper()}")
    print("=" * 65)
    print(f" Strategy Model    : {active['name']}")
    print(f" Active Signal     : {active['signal']} (Score: {active['score']*100:+.1f}%)")
    print(f" Consensus Signal  : {ensemble['signal']} (Score: {ensemble['score']*100:+.1f}%)")
    print(f" Indicator Summary : {active['indicator_summary']}")
    print("-" * 65)
    print(" [SETUP] CALCULATED TRADE SETUP & RISK TARGETS:")
    print(f"   * Action            : {setup['action']}")
    print(f"   * Entry Price       : ${setup['entry_price']:.2f}")
    print(f"   * Stop-Loss Level   : ${setup['stop_loss']:.2f}  (Risk/Share: ${setup['risk_per_share']:.2f})")
    print(f"   * Take-Profit Target: ${setup['target_1']:.2f}  (Reward/Share: ${setup['reward_per_share']:.2f})")
    print(f"   * Risk/Reward Ratio : {setup['risk_reward_ratio']}")
    print("-" * 65)
    print(" [SUB-STRATEGIES BREAKDOWN]:")
    for sub in analysis["sub_strategies"]:
        print(f"   [{sub['signal']:^9}] {sub['name']:<35} Score: {sub['score']*100:+.0f}%")
    print("=" * 65 + "\n")

def display_backtest_report(ticker: str, bt: dict):
    print("\n" + "=" * 65)
    print(f" [BACKTEST] HISTORICAL BACKTEST SIMULATION REPORT — {ticker.upper()}")
    print("=" * 65)
    print(f" Strategy Name         : {bt['strategy_name']}")
    print(f" Initial Capital       : ${bt['initial_capital']:,.2f}")
    print(f" Final Portfolio Equity: ${bt['final_equity']:,.2f}")
    print(f" Strategy Total Return : {bt['strategy_return_pct']:+.2f}%")
    print(f" Buy & Hold Return     : {bt['buy_hold_return_pct']:+.2f}%")
    print(f" Total Trades Executed : {bt['total_trades']} (Win: {bt['winning_trades']}, Loss: {bt['losing_trades']})")
    print(f" Strategy Win Rate     : {bt['win_rate_pct']}%")
    print(f" Profit Factor         : {bt['profit_factor']}")
    print(f" Max Drawdown          : {bt['max_drawdown_pct']}%")
    print(f" Sharpe Ratio          : {bt['sharpe_ratio']}")
    print("=" * 65 + "\n")

from stant.portfolio import build_optimized_portfolio

def parse_arguments():
    parser = argparse.ArgumentParser(description="Fetch, save, plot, run strategies, and optimize portfolios.")
    parser.add_argument("ticker", type=str, nargs="?", default="AAPL", help="The stock ticker symbol (e.g., AAPL, MSFT, TSLA)")
    parser.add_argument("--data-dir", type=str, default="data", help="Directory to save the CSV data file (default: data)")
    parser.add_argument("--plot-dir", type=str, default="plots", help="Directory to save the plot image file (default: plots)")
    parser.add_argument("--show", action="store_true", help="Display the plot window after saving")
    parser.add_argument(
        "--interval", 
        type=str, 
        default="1d", 
        choices=["1m", "2m", "5m", "15m", "30m", "60m", "90m", "1h", "1d", "5d", "1wk", "1mo", "3mo"],
        help="Data resolution/interval. Default: 1d"
    )
    parser.add_argument(
        "--strategy",
        type=str,
        default="ensemble",
        choices=list(STRATEGY_NAMES.keys()),
        help="Trading strategy to evaluate (default: ensemble)"
    )
    parser.add_argument("--signals", action="store_true", help="Display detailed Buy/Sell signals & trade setup targets in CLI")
    parser.add_argument("--backtest", action="store_true", help="Run historical backtest simulation and output win rate & return statistics")
    parser.add_argument("--screener", type=str, default=None, help="Run multi-stock batch screener across comma-separated tickers (e.g. --screener AAPL,MSFT,TSLA,NVDA)")
    parser.add_argument("--portfolio", type=str, default=None, help="Optimize portfolio across comma-separated tickers (e.g. --portfolio AAPL,MSFT,NVDA,TSLA,SPY)")
    parser.add_argument("--capital", type=float, default=10000.0, help="Initial investment capital amount for portfolio (default: 10000)")
    parser.add_argument("--model", type=str, default="max_sharpe", choices=["max_sharpe", "risk_parity", "black_litterman", "min_variance", "max_div"], help="Portfolio optimization model")
    parser.add_argument("--start", type=str, default=None, help="Start date (YYYY-MM-DD).")
    parser.add_argument("--end", type=str, default=None, help="End date (YYYY-MM-DD).")
    return parser.parse_args()

def display_portfolio_report(res: dict):
    print("\n" + "=" * 80)
    print(f" [PORTFOLIO] INSTITUTIONAL PORTFOLIO OPTIMIZATION REPORT (Model: {res['model_type'].upper()})")
    print("=" * 80)
    print(f" Capital Invested    : ${res['capital']:,.2f}")
    print(f" Expected Return     : {res['exp_return_pct']:+.2f}% / year")
    print(f" Portfolio Risk (Vol): {res['port_volatility_pct']:.2f}% / year")
    print(f" Sharpe Ratio        : {res['sharpe_ratio']}")
    print(f" Sortino Ratio       : {res['sortino_ratio']}")
    print(f" Daily 95% VaR       : ${res['var_95_daily_dollar']:,.2f}")
    print(f" Monthly 95% VaR     : ${res['var_95_monthly_dollar']:,.2f}")
    print(f" Max Drawdown        : -{res['max_drawdown_pct']:.2f}%")
    print("-" * 80)
    print(f" {'TICKER':<8} {'WEIGHT':<10} {'INVESTED ($)':<14} {'SHARES':<10} {'PRICE':<10} {'QUANT SIGNAL':<15}")
    print("-" * 80)
    for a in res["allocations"]:
        print(f" {a['ticker']:<8} {a['weight_pct']:>6.2f}%    ${a['target_dollar']:>11.2f}   {a['share_count']:>6} sh   ${a['last_price']:>8.2f}  {a['quant_signal']:<15}")
    print("=" * 80 + "\n")

def main():
    if hasattr(sys.stdout, 'reconfigure'):
        try:
            sys.stdout.reconfigure(encoding='utf-8')
        except Exception:
            pass

    args = parse_arguments()

    if args.portfolio:
        tickers = [t.strip() for t in args.portfolio.split(",") if t.strip()]
        print(f"\nBuilding optimized portfolio across {len(tickers)} tickers using '{args.model}' model...")
        res = build_optimized_portfolio(tickers=tickers, model_type=args.model, capital=args.capital)
        if "error" in res:
            print(f"Portfolio Error: {res['error']}", file=sys.stderr)
            sys.exit(1)
        display_portfolio_report(res)
        return
    
    if args.screener:
        tickers = [t.strip() for t in args.screener.split(",") if t.strip()]
        print(f"\nRunning Multi-Stock Screener across {len(tickers)} tickers...")
        res = run_multi_stock_screener(tickers=tickers, interval=args.interval, strategy_key=args.strategy)
        print("\n" + "=" * 80)
        print(f" [SCREENER] MULTI-STOCK SCREENER RESULTS (Interval: {args.interval}, Strategy: {args.strategy})")
        print("=" * 80)
        print(f"{'TICKER':<10} {'PRICE':<10} {'CHANGE':<10} {'SIGNAL':<12} {'SCORE':<10} {'STOP LOSS':<10} {'TARGET':<10}")
        print("-" * 80)
        for r in res["results"]:
            if r.get("status") == "ok":
                print(f"{r['ticker']:<10} ${r['last_price']:<9.2f} {r['change_pct']:+6.2f}%   {r['signal']:<12} {r['score']:+6.1f}%    ${r['stop_loss']:<9.2f} ${r['target_1']:<9.2f}")
            else:
                print(f"{r['ticker']:<10} ERROR: {r.get('error')}")
        print("=" * 80 + "\n")
        return

    ticker = args.ticker.strip()
    if not ticker:
        print("Error: Ticker symbol cannot be empty.", file=sys.stderr)
        sys.exit(1)
        
    try:
        df, resolved_ticker = fetch_historical_data(ticker, interval=args.interval, start=args.start, end=args.end)
        csv_path = save_data(df, resolved_ticker, args.data_dir)
        plot_path = plot_data(df, resolved_ticker, args.plot_dir, args.show, interval=args.interval, start=args.start, end=args.end)
        
        analysis = run_strategy_analysis(df, strategy_key=args.strategy, interval=args.interval)
        display_signals_report(resolved_ticker, analysis)

        if args.backtest:
            bt_res = backtest_strategy(df, strategy_key=args.strategy, interval=args.interval)
            display_backtest_report(resolved_ticker, bt_res)

        print(f"Data File: {os.path.abspath(csv_path)}")
        print(f"Plot File: {os.path.abspath(plot_path)}")
        
    except ValueError as ve:
        print(f"\nError: {ve}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"\nAn unexpected error occurred: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()

