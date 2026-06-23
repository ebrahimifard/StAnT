import os
import sys
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from stant.market_data import get_stock_data

def parse_arguments():
    parser = argparse.ArgumentParser(description="Fetch, save, and plot historical stock data from Yahoo Finance.")
    parser.add_argument("ticker", type=str, help="The stock ticker symbol (e.g., AAPL, MSFT, TSLA)")
    parser.add_argument("--data-dir", type=str, default="data", help="Directory to save the CSV data file (default: data)")
    parser.add_argument("--plot-dir", type=str, default="plots", help="Directory to save the plot image file (default: plots)")
    parser.add_argument("--show", action="store_true", help="Display the plot window after saving")
    parser.add_argument(
        "--interval", 
        type=str, 
        default="1d", 
        choices=["1m", "2m", "5m", "15m", "30m", "60m", "90m", "1h", "1d", "5d", "1wk", "1mo", "3mo"],
        help="Data resolution/interval. Note: Intraday intervals (1m-1h) have historical limits in Yahoo Finance (e.g. 1m is max 7 days, 1h is max 730 days). Default: 1d"
    )
    parser.add_argument("--start", type=str, default=None, help="Start date (YYYY-MM-DD). If not specified, downloads from the first available day.")
    parser.add_argument("--end", type=str, default=None, help="End date (YYYY-MM-DD). If not specified, downloads up to the last available moment.")
    return parser.parse_args()

def fetch_historical_data(ticker: str, interval: str = "1d", start: str = None, end: str = None) -> tuple[pd.DataFrame, str]:
    """
    Fetch historical data for a given ticker with a specified interval and optional start/end dates.
    
    Yahoo Finance Historical Data Limits:
    - 1-minute (1m) resolution: max 7 days of history.
    - 2m, 5m, 15m, 30m, 90m resolution: max 60 days of history.
    - 1-hour (1h) resolution: max 730 days of history.
    - Daily (1d), weekly (1wk), monthly (1mo) or longer resolutions support full history.
    """
    if start or end:
        print(f"Fetching historical data for ticker '{ticker}' (interval='{interval}', start={start}, end={end})...")
    else:
        print(f"Fetching historical data for ticker '{ticker}' (interval='{interval}', period='max')...")

    stock_data = get_stock_data(ticker, interval=interval, start=start, end=end)
    
    if stock_data.frame.empty:
        raise ValueError(
            f"No data returned for ticker '{ticker}'. Please check if the symbol is valid, "
            f"and that the selected interval ('{interval}') is compatible with the requested date range.\n"
            f"Note: Intraday intervals have strict limits (1m: max 7 days, 1h: max 730 days).\n"
            f"Tried Yahoo Finance symbols: {', '.join(stock_data.tried_symbols)}."
        )

    if stock_data.resolved_symbol != stock_data.requested_symbol:
        print(f"Resolved '{stock_data.requested_symbol}' to Yahoo Finance symbol '{stock_data.resolved_symbol}'.")
        
    return stock_data.frame, stock_data.resolved_symbol

def save_data(df: pd.DataFrame, ticker: str, data_dir: str) -> str:
    # Ensure directory exists
    os.makedirs(data_dir, exist_ok=True)
    
    # Clean index name and save
    df_to_save = df.copy()
    
    # Convert index (which is timezone-aware DatetimeIndex) to timezone-naive dates for cleaner CSV
    if isinstance(df_to_save.index, pd.DatetimeIndex):
        df_to_save.index = df_to_save.index.tz_localize(None)
    
    file_path = os.path.join(data_dir, f"{ticker.upper()}_history.csv")
    df_to_save.to_csv(file_path)
    print(f"Historical data successfully saved to: {file_path}")
    return file_path

def plot_data(df: pd.DataFrame, ticker: str, plot_dir: str, show_plot: bool, interval: str = "1d", start: str = None, end: str = None) -> str:
    os.makedirs(plot_dir, exist_ok=True)
    
    print("Generating historical data plot...")
    ticker_upper = ticker.upper()
    
    # Set up matplotlib style for a clean, modern look
    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
    
    # Create figure with two subplots: Price (top) and Volume (bottom)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True, gridspec_kw={'height_ratios': [3, 1]})
    
    # Convert index to timezone-naive for matplotlib compatibility
    dates = df.index.tz_localize(None) if isinstance(df.index, pd.DatetimeIndex) else df.index
    
    # Plot Close Price
    ax1.plot(dates, df['Close'], label='Close Price', color='#0f9755', linewidth=1.5)
    
    # Add simple moving averages for aesthetic and functional depth
    unit = "period" if interval != "1d" else "day"
    if len(df) >= 50:
        ma50 = df['Close'].rolling(window=50).mean()
        ax1.plot(dates, ma50, label=f'50-{unit} SMA', color='#e67e22', linewidth=1.0, linestyle='--')
    if len(df) >= 200:
        ma200 = df['Close'].rolling(window=200).mean()
        ax1.plot(dates, ma200, label=f'200-{unit} SMA', color='#2980b9', linewidth=1.0, linestyle='--')
        
    # Dynamic period subtitle
    if start or end:
        period_str = f"{start or 'Start'} to {end or 'End'}"
    else:
        period_str = "Full History"
    ax1.set_title(f"{ticker_upper} Historical Close Price & Volume ({period_str}, interval={interval})", fontsize=14, fontweight='bold', pad=15)
    ax1.set_ylabel("Price (USD)", fontsize=11, fontweight='semibold')
    ax1.legend(loc='upper left', frameon=True, facecolor='white', edgecolor='none')
    ax1.tick_params(labelsize=9)
    ax1.grid(True, linestyle=':', alpha=0.6)
    
    # Plot Volume
    ax2.bar(dates, df['Volume'], color='#bdc3c7', alpha=0.7, width=1.0, label='Volume')
    ax2.set_ylabel("Volume", fontsize=11, fontweight='semibold')
    ax2.set_xlabel("Date", fontsize=11, fontweight='semibold')
    ax2.tick_params(labelsize=9)
    ax2.grid(True, linestyle=':', alpha=0.6)
    
    # Format x-axis dates
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    ax2.xaxis.set_major_locator(mdates.AutoDateLocator())
    fig.autofmt_xdate()
    
    # Adjust layout
    plt.tight_layout()
    
    # Save the plot
    plot_file = os.path.join(plot_dir, f"{ticker_upper}_plot.png")
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    print(f"Plot successfully saved to: {plot_file}")
    
    if show_plot:
        print("Displaying plot window. Close the window to complete the program execution.")
        plt.show()
        
    return plot_file

def main():
    args = parse_arguments()
    ticker = args.ticker.strip()
    
    if not ticker:
        print("Error: Ticker symbol cannot be empty.", file=sys.stderr)
        sys.exit(1)
        
    try:
        # Fetch data
        df, resolved_ticker = fetch_historical_data(ticker, interval=args.interval, start=args.start, end=args.end)
        
        # Save CSV
        csv_path = save_data(df, resolved_ticker, args.data_dir)
        
        # Plot data
        plot_path = plot_data(df, resolved_ticker, args.plot_dir, args.show, interval=args.interval, start=args.start, end=args.end)
        
        print("\nProcess completed successfully!")
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
