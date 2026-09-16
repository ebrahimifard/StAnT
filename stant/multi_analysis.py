"""Multi-stock screener and batch strategy evaluator for StAnT.

Processes a batch of stock tickers, computes technical indicators, consensus
trading signals, risk setups, and ranks them by opportunity score.
"""

from __future__ import annotations

from typing import Any

from stant.market_data import get_stock_data
from stant.strategies import run_strategy_analysis


DEFAULT_SCREENER_TICKERS = ["AAPL", "MSFT", "TSLA", "NVDA", "GOOGL", "AMZN", "SPY"]


def run_multi_stock_screener(
    tickers: list[str] | None = None,
    interval: str = "1d",
    strategy_key: str = "ensemble",
) -> dict[str, Any]:
    """Run batch analysis on multiple stock tickers."""
    if not tickers:
        tickers = DEFAULT_SCREENER_TICKERS

    results = []

    for raw_ticker in tickers:
        t = raw_ticker.strip().upper()
        if not t:
            continue

        try:
            stock_data = get_stock_data(t, interval=interval)
            df = stock_data.frame

            if df.empty or len(df) < 5:
                results.append({
                    "ticker": t,
                    "resolved_ticker": stock_data.resolved_symbol,
                    "status": "error",
                    "error": "No sufficient price data returned",
                })
                continue

            analysis = run_strategy_analysis(df, strategy_key=strategy_key, interval=interval)
            active_strat = analysis["active_strategy"]
            ensemble = analysis["ensemble_consensus"]

            last_close = float(df["Close"].iloc[-1])
            prev_close = float(df["Close"].iloc[-2]) if len(df) >= 2 else last_close
            change_pct = ((last_close - prev_close) / prev_close) * 100

            results.append({
                "ticker": stock_data.requested_symbol,
                "resolved_ticker": stock_data.resolved_symbol,
                "status": "ok",
                "last_price": round(last_close, 2),
                "change_pct": round(change_pct, 2),
                "signal": active_strat["signal"],
                "score": round(active_strat["score"] * 100, 1),
                "consensus_signal": ensemble["signal"],
                "consensus_score": round(ensemble["score"] * 100, 1),
                "entry_price": active_strat["trade_setup"]["entry_price"],
                "stop_loss": active_strat["trade_setup"]["stop_loss"],
                "target_1": active_strat["trade_setup"]["target_1"],
                "risk_reward": active_strat["trade_setup"]["risk_reward_ratio"],
                "indicator_summary": active_strat["indicator_summary"],
            })
        except Exception as e:
            results.append({
                "ticker": t,
                "status": "error",
                "error": str(e),
            })

    # Sort results by consensus score descending (Highest conviction Buy to Sell)
    results.sort(key=lambda x: x.get("consensus_score", -999.0), reverse=True)

    top_buys = [r for r in results if r.get("status") == "ok" and r.get("consensus_score", 0) >= 20.0]
    top_sells = [r for r in results if r.get("status") == "ok" and r.get("consensus_score", 0) <= -20.0]

    return {
        "interval": interval,
        "strategy_key": strategy_key,
        "total_analyzed": len(results),
        "results": results,
        "top_buys_count": len(top_buys),
        "top_sells_count": len(top_sells),
    }
