"""Market data helpers for StAnT.

The web app and CLI both use this module so ticker resolution, Yahoo Finance
cache handling, and response formatting stay consistent.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import yfinance as yf

PROJECT_ROOT = Path(__file__).resolve().parent.parent
YFINANCE_CACHE_DIR = PROJECT_ROOT / ".yfinance-cache"

EXCHANGE_SUFFIX_FALLBACKS = (
    ".L",   # London Stock Exchange
    ".AS",  # Euronext Amsterdam
    ".DE",  # Deutsche Boerse Xetra
    ".MI",  # Borsa Italiana
    ".PA",  # Euronext Paris
    ".SW",  # SIX Swiss Exchange
)


@dataclass(frozen=True)
class ResolvedStockData:
    """Historical prices plus the Yahoo symbol that actually returned data."""

    frame: pd.DataFrame
    requested_symbol: str
    resolved_symbol: str
    tried_symbols: list[str]


def configure_yfinance_cache(cache_dir: Path = YFINANCE_CACHE_DIR) -> None:
    """Store yfinance cache files inside the project instead of user globals."""

    cache_dir.mkdir(parents=True, exist_ok=True)
    yf.set_tz_cache_location(str(cache_dir))


INTRADAY_INTERVALS = frozenset(
    {"1m", "2m", "5m", "15m", "30m", "60m", "90m", "1h"}
)


def is_intraday(interval: str) -> bool:
    """Return True for minute/hour intervals used by Lightweight Charts.

    Must be an exact match against yfinance's intraday interval vocabulary -
    a substring check (e.g. "m" in interval) would misclassify "1mo"/"3mo"
    (monthly) as intraday, since they contain "m" too.
    """

    return interval in INTRADAY_INTERVALS


def get_ticker_candidates(ticker: str) -> list[str]:
    """Return Yahoo Finance symbols to try for a user-entered ticker.

    Yahoo Finance usually requires exchange suffixes for non-US ETFs and
    listings. For example, VUSA on the London Stock Exchange is VUSA.L.
    """

    normalized = ticker.strip().upper()
    candidates = [normalized]

    if "." not in normalized and not normalized.startswith("^"):
        candidates.extend(f"{normalized}{suffix}" for suffix in EXCHANGE_SUFFIX_FALLBACKS)

    return candidates


def fetch_stock_history(
    symbol: str,
    interval: str,
    start: str | None = None,
    end: str | None = None,
) -> pd.DataFrame:
    """Fetch historical OHLCV data from Yahoo Finance."""

    ticker_obj = yf.Ticker(symbol)

    if start or end:
        return ticker_obj.history(
            interval=interval,
            start=start or None,
            end=end or None,
            timeout=10,
            auto_adjust=True,
        )

    return ticker_obj.history(
        interval=interval, period="max", timeout=10, auto_adjust=True
    )


def get_stock_data(
    ticker: str,
    interval: str,
    start: str | None = None,
    end: str | None = None,
) -> ResolvedStockData:
    """Fetch the first non-empty result across direct and fallback symbols."""

    requested_symbol = ticker.strip().upper()
    tried_symbols: list[str] = []
    last_frame = pd.DataFrame()

    for candidate in get_ticker_candidates(ticker):
        tried_symbols.append(candidate)
        try:
            frame = fetch_stock_history(candidate, interval, start, end)
        except Exception:
            # A slow/rate-limited candidate (e.g. a Yahoo exchange-suffix guess
            # that doesn't exist) shouldn't abort the whole lookup - move on to
            # the next fallback symbol instead.
            frame = pd.DataFrame()
        if not frame.empty:
            return ResolvedStockData(
                frame=frame,
                requested_symbol=requested_symbol,
                resolved_symbol=candidate,
                tried_symbols=tried_symbols,
            )
        last_frame = frame

    return ResolvedStockData(
        frame=last_frame,
        requested_symbol=requested_symbol,
        resolved_symbol=requested_symbol,
        tried_symbols=tried_symbols,
    )


def build_chart_payload(stock_data: ResolvedStockData, interval: str) -> dict:
    """Convert a resolved OHLCV frame into the JSON shape used by the UI."""

    frame = stock_data.frame
    chart_data = []
    volume_data = []

    frame_reset = frame.reset_index()
    time_col = frame_reset.columns[0]

    for _, row in frame_reset.iterrows():
        timestamp = row[time_col]
        if pd.isna(row["Close"]) or pd.isna(row["Open"]):
            continue

        if is_intraday(interval):
            time_value = int(timestamp.timestamp())
        else:
            time_value = timestamp.strftime("%Y-%m-%d")

        chart_data.append(
            {
                "time": time_value,
                "open": float(row["Open"]),
                "high": float(row["High"]),
                "low": float(row["Low"]),
                "close": float(row["Close"]),
            }
        )

        color = (
            "rgba(38, 166, 154, 0.5)"
            if row["Close"] >= row["Open"]
            else "rgba(239, 83, 80, 0.5)"
        )
        volume_data.append(
            {
                "time": time_value,
                "value": float(row["Volume"]),
                "color": color,
            }
        )

    close_prices = frame["Close"].tolist()
    return_pct = 0.0
    if len(close_prices) >= 2:
        return_pct = float(((close_prices[-1] - close_prices[0]) / close_prices[0]) * 100)

    return {
        "ticker": stock_data.resolved_symbol,
        "requested_ticker": stock_data.requested_symbol,
        "tried_tickers": stock_data.tried_symbols,
        "interval": interval,
        "chart_data": chart_data,
        "volume_data": volume_data,
        "stats": {
            "highest": float(frame["High"].max()),
            "lowest": float(frame["Low"].min()),
            "avg": float(frame["Close"].mean()),
            "volume": int(frame["Volume"].sum()),
            "return_pct": return_pct,
        },
    }


configure_yfinance_cache()
