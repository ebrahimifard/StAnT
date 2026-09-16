"""Historical strategy backtesting engine for StAnT.

Simulates strategy executions over historical stock data and calculates key
performance indicators (Win Rate, Total Return vs Buy&Hold, Profit Factor,
Max Drawdown, Sharpe Ratio, Equity Curve).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from stant.indicators import calculate_all_indicators
from stant.market_data import is_intraday
from stant.strategies import (
    evaluate_bollinger,
    evaluate_ensemble_consensus,
    evaluate_ichimoku,
    evaluate_keltner_squeeze,
    evaluate_macd_cross,
    evaluate_ml_quant,
    evaluate_parabolic_sar,
    evaluate_rsi_reversion,
    evaluate_sma_cross,
    evaluate_stochastic,
    evaluate_supertrend,
    evaluate_vwap_profile,
)


RISK_FREE_RATE = 0.04


def _format_time(timestamp: pd.Timestamp, interval: str) -> str | int:
    if is_intraday(interval):
        return int(timestamp.timestamp())
    return timestamp.strftime("%Y-%m-%d")


def backtest_strategy(
    df: pd.DataFrame,
    strategy_key: str = "ensemble",
    interval: str = "1d",
    initial_capital: float = 10000.0,
    cost_bps: float = 5.0,
) -> dict[str, Any]:
    """Run historical simulation for chosen strategy and return metrics."""
    if len(df) < 5:
        return {
            "error": "Insufficient data to run backtest simulation (minimum 5 data points required)"
        }

    df_ind = calculate_all_indicators(df)

    if strategy_key == "sma_cross":
        res = evaluate_sma_cross(df_ind, interval)
    elif strategy_key == "rsi_reversion":
        res = evaluate_rsi_reversion(df_ind, interval)
    elif strategy_key == "macd_cross":
        res = evaluate_macd_cross(df_ind, interval)
    elif strategy_key == "bollinger":
        res = evaluate_bollinger(df_ind, interval)
    elif strategy_key == "stochastic":
        res = evaluate_stochastic(df_ind, interval)
    elif strategy_key == "supertrend":
        res = evaluate_supertrend(df_ind, interval)
    elif strategy_key == "ichimoku":
        res = evaluate_ichimoku(df_ind, interval)
    elif strategy_key == "keltner_squeeze":
        res = evaluate_keltner_squeeze(df_ind, interval)
    elif strategy_key == "parabolic_sar":
        res = evaluate_parabolic_sar(df_ind, interval)
    elif strategy_key == "vwap_profile":
        res = evaluate_vwap_profile(df_ind, interval)
    elif strategy_key == "ml_quant":
        res = evaluate_ml_quant(df_ind, interval)
    else:
        res, _ = evaluate_ensemble_consensus(df_ind, interval)

    marker_map = {}
    for m in res.chart_markers:
        marker_map[str(m["time"])] = "BUY" if "BUY" in m["text"] or "Bullish" in m["text"] else "SELL"

    frame_reset = df_ind.reset_index()
    time_col = frame_reset.columns[0]
    close_prices = df_ind["Close"].values
    open_prices = df_ind["Open"].values
    cost_rate = cost_bps / 10000.0

    position = 0
    entry_price = 0.0
    cash = initial_capital
    holdings = 0.0

    trades = []
    equity_curve = []

    buy_and_hold_shares = initial_capital / close_prices[0]

    # A signal derived from bar i's close is only actionable from bar i+1 onward -
    # it fills at bar i+1's open, never at the same close that produced it.
    pending_signal = None

    for i in range(len(df_ind)):
        t_val = _format_time(frame_reset.loc[i, time_col], interval)
        close_price = close_prices[i]
        fill_price = open_prices[i]

        sig = pending_signal

        if sig == "BUY" and position == 0:
            position = 1
            entry_price = fill_price * (1 + cost_rate)
            holdings = cash / entry_price
            cash = 0.0
        elif sig == "SELL" and position == 1:
            position = 0
            exit_price = fill_price * (1 - cost_rate)
            cash = holdings * exit_price
            pnl_pct = ((exit_price - entry_price) / entry_price) * 100
            pnl_val = cash - (holdings * entry_price)
            trades.append({"entry": entry_price, "exit": exit_price, "pnl_pct": pnl_pct, "pnl_val": pnl_val})
            holdings = 0.0

        current_equity = cash + (holdings * close_price) if position == 1 else cash
        current_buy_hold = buy_and_hold_shares * close_price

        equity_curve.append({
            "time": t_val,
            "equity": round(current_equity, 2),
            "buy_hold": round(current_buy_hold, 2),
        })

        pending_signal = marker_map.get(str(t_val), None)

    final_equity = cash + (holdings * close_prices[-1]) if position == 1 else cash
    final_buy_hold = buy_and_hold_shares * close_prices[-1]

    strategy_return_pct = ((final_equity - initial_capital) / initial_capital) * 100
    buy_hold_return_pct = ((final_buy_hold - initial_capital) / initial_capital) * 100

    winning_trades = [t for t in trades if t["pnl_val"] > 0]
    losing_trades = [t for t in trades if t["pnl_val"] <= 0]

    total_trades = len(trades)
    win_rate_pct = (len(winning_trades) / total_trades * 100) if total_trades > 0 else 0.0

    gross_profit = sum(t["pnl_val"] for t in winning_trades)
    gross_loss = abs(sum(t["pnl_val"] for t in losing_trades))

    if gross_loss > 0:
        profit_factor = gross_profit / gross_loss
    elif gross_profit > 0:
        profit_factor = 99.0
    else:
        profit_factor = 1.0

    eq_values = np.array([pt["equity"] for pt in equity_curve])
    running_max = np.maximum.accumulate(eq_values)
    drawdowns = (eq_values - running_max) / running_max
    max_drawdown_pct = abs(float(np.min(drawdowns))) * 100 if len(drawdowns) > 0 else 0.0

    period_returns = np.diff(eq_values) / eq_values[:-1] if len(eq_values) > 1 else np.array([])
    period_returns = period_returns[np.isfinite(period_returns)]
    if len(period_returns) > 1 and np.std(period_returns) > 0:
        sharpe_ratio = float(
            ((np.mean(period_returns) * 252) - RISK_FREE_RATE) / (np.std(period_returns) * np.sqrt(252))
        )
    else:
        sharpe_ratio = 0.0

    return {
        "strategy_key": strategy_key,
        "strategy_name": res.name,
        "initial_capital": initial_capital,
        "final_equity": round(final_equity, 2),
        "strategy_return_pct": round(strategy_return_pct, 2),
        "buy_hold_return_pct": round(buy_hold_return_pct, 2),
        "total_trades": total_trades,
        "winning_trades": len(winning_trades),
        "losing_trades": len(losing_trades),
        "win_rate_pct": round(win_rate_pct, 1),
        "profit_factor": round(profit_factor, 2),
        "max_drawdown_pct": round(max_drawdown_pct, 2),
        "sharpe_ratio": round(sharpe_ratio, 2),
        "cost_bps_per_trade": cost_bps,
        "equity_curve": equity_curve,
    }
