"""Correctness tests for stant.backtest: the engine a real trading decision
would be based on, so these check the mechanics independently of any single
strategy's signal-generation logic (which is covered in test_strategies.py).

Where a test needs a fully controlled sequence of BUY/SELL signals, it mocks
the relevant `evaluate_*` function that backtest_strategy dispatches to, so
the trade-simulation math can be verified against hand-computed numbers
without depending on any strategy's actual indicator crossovers.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from stant.backtest import PERIODS_PER_YEAR, _periods_per_year, backtest_strategy
from stant.config import RISK_FREE_RATE_ANNUAL
from stant.strategies import STRATEGY_NAMES, StrategyResult


def _make_df(opens, highs, lows, closes, start="2025-01-01"):
    n = len(opens)
    dates = pd.date_range(start=start, periods=n, freq="D")
    return pd.DataFrame(
        {
            "Open": opens,
            "High": highs,
            "Low": lows,
            "Close": closes,
            "Volume": [1_000_000] * n,
        },
        index=dates,
    )


def _forced_result(key, markers):
    return StrategyResult(
        key=key,
        name="forced-for-test",
        signal="BUY",
        score=1.0,
        indicator_summary="forced",
        chart_markers=markers,
        trade_setup={},
    )


def _marker(time, bullish, text):
    return {
        "time": time,
        "position": "belowBar" if bullish else "aboveBar",
        "color": "#26a69a" if bullish else "#ef5350",
        "shape": "arrowUp" if bullish else "arrowDown",
        "text": text,
    }


# --------------------------------------------------------------------------
# Look-ahead bias
# --------------------------------------------------------------------------


def test_signal_fills_at_next_bar_open_not_same_bar_close(mocker):
    df = _make_df(
        opens=[100, 101, 102, 103, 104, 105, 106, 107, 108, 109],
        highs=[101, 102, 103, 104, 105, 106, 107, 108, 109, 110],
        lows=[99, 100, 101, 102, 103, 104, 105, 106, 107, 108],
        closes=[100.5, 101.5, 102.5, 103.5, 104.5, 105.5, 106.5, 107.5, 108.5, 109.5],
    )
    # Marker at bar 4's date ("2025-01-05") - a signal derived from bar 4's
    # data must only be actionable from bar 5 onward.
    forced = _forced_result("sma_cross", [_marker("2025-01-05", True, "Golden Cross (BUY)")])
    mocker.patch("stant.backtest.evaluate_sma_cross", return_value=forced)

    result = backtest_strategy(df, strategy_key="sma_cross", interval="1d", initial_capital=10000.0, cost_bps=0.0)

    curve = result["equity_curve"]
    assert curve[4]["equity"] == 10000.0  # unchanged on the signal's own bar
    entry_price = df["Open"].iloc[5]
    expected_equity = round((10000.0 / entry_price) * df["Close"].iloc[5], 2)
    assert curve[5]["equity"] == pytest.approx(expected_equity)
    assert curve[5]["equity"] != pytest.approx(10000.0)


def test_trade_entry_and_exit_times_are_one_bar_after_their_signals(mocker):
    df = _make_df(
        opens=[100, 101, 102, 103, 104, 105],
        highs=[101, 102, 103, 104, 105, 106],
        lows=[99, 100, 101, 102, 103, 104],
        closes=[100.5, 101.5, 102.5, 103.5, 104.5, 105.5],
    )
    forced = _forced_result(
        "sma_cross",
        [
            _marker("2025-01-01", True, "Golden Cross (BUY)"),
            _marker("2025-01-02", False, "Death Cross (SELL)"),
        ],
    )
    mocker.patch("stant.backtest.evaluate_sma_cross", return_value=forced)

    result = backtest_strategy(df, strategy_key="sma_cross", interval="1d", initial_capital=10000.0, cost_bps=0.0)

    assert result["total_trades"] == 1
    trade = result["trades"][0]
    assert trade["entry_time"] == "2025-01-02"  # bar after the BUY signal's bar
    assert trade["exit_time"] == "2025-01-03"  # bar after the SELL signal's bar
    assert trade["entry"] == pytest.approx(df["Open"].iloc[1])
    assert trade["exit"] == pytest.approx(df["Open"].iloc[2])


def test_last_bar_pending_signal_is_never_executed(mocker):
    df = _make_df(
        opens=[100, 101, 102, 103, 104],
        highs=[101, 102, 103, 104, 105],
        lows=[99, 100, 101, 102, 103],
        closes=[100.5, 101.5, 102.5, 103.5, 104.5],
    )
    # Marker on the FINAL bar - there is no next bar for it to fill on.
    forced = _forced_result("sma_cross", [_marker("2025-01-05", True, "Golden Cross (BUY)")])
    mocker.patch("stant.backtest.evaluate_sma_cross", return_value=forced)

    result = backtest_strategy(df, strategy_key="sma_cross", interval="1d", initial_capital=10000.0, cost_bps=0.0)

    assert result["total_trades"] == 0
    assert result["final_equity"] == 10000.0


# --------------------------------------------------------------------------
# Marker -> side classification (regression test for a confirmed live bug:
# text-substring matching silently misclassified several real bullish
# markers as SELL - see stant/backtest.py's classification comment)
# --------------------------------------------------------------------------


def test_bullish_marker_without_buy_or_bullish_in_text_still_triggers_buy(mocker):
    df = _make_df(
        opens=[100, 101, 102, 103, 104],
        highs=[101, 102, 103, 104, 105],
        lows=[99, 100, 101, 102, 103],
        closes=[100.5, 101.5, 102.5, 103.5, 104.5],
    )
    # This is rsi_reversion's real oversold marker text - it contains neither
    # "BUY" nor "Bullish". Before the fix, the old text-substring classifier
    # would have mapped this to SELL, and since SELL is ignored while flat
    # (position == 0), the position would NEVER have opened.
    forced = _forced_result("rsi_reversion", [_marker("2025-01-01", True, "RSI Oversold (25.0)")])
    mocker.patch("stant.backtest.evaluate_rsi_reversion", return_value=forced)

    result = backtest_strategy(df, strategy_key="rsi_reversion", interval="1d", initial_capital=10000.0, cost_bps=0.0)

    assert result["equity_curve"][1]["equity"] != pytest.approx(10000.0)


def test_ml_quant_bullish_marker_triggers_buy(mocker):
    df = _make_df(
        opens=[100, 101, 102, 103, 104],
        highs=[101, 102, 103, 104, 105],
        lows=[99, 100, 101, 102, 103],
        closes=[100.5, 101.5, 102.5, 103.5, 104.5],
    )
    # ml_quant's marker text ("Quant Score NN%") never contains "BUY" or
    # "Bullish" regardless of direction - under the old text-based
    # classifier this strategy could never open a position via its own
    # marker at all.
    forced = _forced_result("ml_quant", [_marker("2025-01-01", True, "Quant Score 82.0%")])
    mocker.patch("stant.backtest.evaluate_ml_quant", return_value=forced)

    result = backtest_strategy(df, strategy_key="ml_quant", interval="1d", initial_capital=10000.0, cost_bps=0.0)

    assert result["equity_curve"][1]["equity"] != pytest.approx(10000.0)


def test_all_strategy_markers_use_consistent_position_color_convention(synthetic_ohlcv):
    """Every strategy must place bullish markers at belowBar/green and
    bearish markers at aboveBar/red - backtest.py's marker classifier keys
    off `position` alone, so a strategy violating this convention would
    silently misfire trades in the wrong direction."""
    from stant.indicators import calculate_all_indicators
    from stant.strategies import (
        evaluate_bollinger,
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

    df = calculate_all_indicators(synthetic_ohlcv(250))
    evaluators = [
        evaluate_sma_cross, evaluate_rsi_reversion, evaluate_macd_cross, evaluate_bollinger,
        evaluate_stochastic, evaluate_supertrend, evaluate_ichimoku, evaluate_keltner_squeeze,
        evaluate_parabolic_sar, evaluate_vwap_profile, evaluate_ml_quant,
    ]
    checked_any = False
    for fn in evaluators:
        res = fn(df, "1d")
        for m in res.chart_markers:
            checked_any = True
            if m["color"] == "#26a69a":
                assert m["position"] == "belowBar", f"{fn.__name__}: green marker not belowBar: {m}"
            elif m["color"] == "#ef5350":
                assert m["position"] == "aboveBar", f"{fn.__name__}: red marker not aboveBar: {m}"
    assert checked_any, "no markers were generated across any strategy - test fixture needs more volatility"


# --------------------------------------------------------------------------
# Hand-calculated profit factor / drawdown / Sharpe
# --------------------------------------------------------------------------


def test_profit_factor_drawdown_and_sharpe_match_hand_calculation(mocker):
    """One winning trade (+10%) then one losing trade (-10%), fully
    controlled, with the resulting 8-point equity curve hand-verified:
    [10000, 10500, 11000, 11000, 10500, 9900, 9900, 9900].
    """
    df = _make_df(
        opens=[100, 100, 110, 110, 110, 99, 99, 99],
        highs=[101, 101, 111, 111, 111, 100, 100, 100],
        lows=[99, 99, 109, 109, 109, 98, 98, 98],
        closes=[100, 105, 110, 110, 105, 99, 99, 99],
    )
    forced = _forced_result(
        "sma_cross",
        [
            _marker("2025-01-01", True, "Golden Cross (BUY)"),   # -> fills bar1 open (100)
            _marker("2025-01-02", False, "Death Cross (SELL)"),  # -> fills bar2 open (110)
            _marker("2025-01-04", True, "Golden Cross (BUY)"),   # -> fills bar4 open (110)
            _marker("2025-01-05", False, "Death Cross (SELL)"),  # -> fills bar5 open (99)
        ],
    )
    mocker.patch("stant.backtest.evaluate_sma_cross", return_value=forced)

    result = backtest_strategy(df, strategy_key="sma_cross", interval="1d", initial_capital=10000.0, cost_bps=0.0)

    eq = np.array([pt["equity"] for pt in result["equity_curve"]])
    assert eq.tolist() == [10000.0, 10500.0, 11000.0, 11000.0, 10500.0, 9900.0, 9900.0, 9900.0]

    assert result["total_trades"] == 2
    assert result["winning_trades"] == 1
    assert result["losing_trades"] == 1
    assert result["win_rate_pct"] == pytest.approx(50.0)

    # gross_profit=1000 (10000->11000), gross_loss=1100 (11000->9900)
    assert result["profit_factor"] == pytest.approx(round(1000 / 1100, 2))

    # min drawdown is exactly 1100/11000 = 10% from the post-trade-1 peak
    assert result["max_drawdown_pct"] == pytest.approx(10.0)

    # Independently reimplement the documented Sharpe formula against the
    # same (hand-verified) equity curve, rather than hand-computing a 7-term
    # variance by hand (too error-prone to serve as a trustworthy oracle).
    period_returns = np.diff(eq) / eq[:-1]
    expected_sharpe = round(
        float(
            (period_returns.mean() * 252 - RISK_FREE_RATE_ANNUAL)
            / (period_returns.std() * np.sqrt(252))
        ),
        2,
    )
    assert result["sharpe_ratio"] == pytest.approx(expected_sharpe)


def test_profit_factor_is_99_when_no_losing_trades(mocker):
    df = _make_df(
        opens=[100, 100, 110, 110, 110],
        highs=[101, 101, 111, 111, 111],
        lows=[99, 99, 109, 109, 109],
        closes=[100, 105, 110, 110, 110],
    )
    forced = _forced_result(
        "sma_cross",
        [_marker("2025-01-01", True, "Golden Cross (BUY)"), _marker("2025-01-02", False, "Death Cross (SELL)")],
    )
    mocker.patch("stant.backtest.evaluate_sma_cross", return_value=forced)

    result = backtest_strategy(df, strategy_key="sma_cross", interval="1d", initial_capital=10000.0, cost_bps=0.0)
    assert result["total_trades"] == 1
    assert result["winning_trades"] == 1
    assert result["profit_factor"] == pytest.approx(99.0)


def test_profit_factor_is_1_when_no_trades_at_all(mocker):
    df = _make_df(
        opens=[100, 101, 102, 103, 104],
        highs=[101, 102, 103, 104, 105],
        lows=[99, 100, 101, 102, 103],
        closes=[100.5, 101.5, 102.5, 103.5, 104.5],
    )
    forced = _forced_result("sma_cross", [])
    mocker.patch("stant.backtest.evaluate_sma_cross", return_value=forced)

    result = backtest_strategy(df, strategy_key="sma_cross", interval="1d", initial_capital=10000.0, cost_bps=0.0)
    assert result["total_trades"] == 0
    assert result["profit_factor"] == pytest.approx(1.0)


def test_insufficient_data_returns_error_dict_not_exception(synthetic_ohlcv):
    df = synthetic_ohlcv(4)
    result = backtest_strategy(df, strategy_key="sma_cross", interval="1d")
    assert "error" in result


# --------------------------------------------------------------------------
# Sharpe annualization (regression test for a confirmed live bug: Sharpe was
# hardcoded to a 252-trading-day annualization regardless of interval, which
# silently misreports Sharpe for any intraday backtest)
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "interval,expected",
    [
        ("1d", 252.0),
        ("1wk", 52.0),
        ("1mo", 12.0),
        ("1h", 252.0 * 390 / 60),
        ("60m", 252.0 * 390 / 60),
        ("30m", 252.0 * 390 / 30),
        ("5m", 252.0 * 390 / 5),
        ("1m", 252.0 * 390 / 1),
    ],
)
def test_periods_per_year_lookup(interval, expected):
    assert _periods_per_year(interval) == pytest.approx(expected)


def test_intraday_backtest_uses_intraday_annualization_not_hardcoded_252(mocker):
    n = 8
    dates = pd.date_range("2025-01-01", periods=n, freq="h")
    df = pd.DataFrame(
        {
            "Open": [100, 100, 110, 110, 110, 99, 99, 99],
            "High": [101, 101, 111, 111, 111, 100, 100, 100],
            "Low": [99, 99, 109, 109, 109, 98, 98, 98],
            "Close": [100, 105, 110, 110, 105, 99, 99, 99],
            "Volume": [1_000_000] * n,
        },
        index=dates,
    )
    epoch_times = [int(ts.timestamp()) for ts in dates]

    forced = _forced_result(
        "sma_cross",
        [
            {"time": epoch_times[0], "position": "belowBar", "color": "#26a69a", "shape": "arrowUp", "text": "Golden Cross (BUY)"},
            {"time": epoch_times[1], "position": "aboveBar", "color": "#ef5350", "shape": "arrowDown", "text": "Death Cross (SELL)"},
            {"time": epoch_times[3], "position": "belowBar", "color": "#26a69a", "shape": "arrowUp", "text": "Golden Cross (BUY)"},
            {"time": epoch_times[4], "position": "aboveBar", "color": "#ef5350", "shape": "arrowDown", "text": "Death Cross (SELL)"},
        ],
    )
    mocker.patch("stant.backtest.evaluate_sma_cross", return_value=forced)

    result = backtest_strategy(df, strategy_key="sma_cross", interval="1h", initial_capital=10000.0, cost_bps=0.0)

    eq = np.array([pt["equity"] for pt in result["equity_curve"]])
    period_returns = np.diff(eq) / eq[:-1]

    hourly_periods_per_year = _periods_per_year("1h")
    wrong_sharpe_hardcoded_252 = round(
        float((period_returns.mean() * 252 - RISK_FREE_RATE_ANNUAL) / (period_returns.std() * np.sqrt(252))), 2
    )
    correct_sharpe = round(
        float(
            (period_returns.mean() * hourly_periods_per_year - RISK_FREE_RATE_ANNUAL)
            / (period_returns.std() * np.sqrt(hourly_periods_per_year))
        ),
        2,
    )

    assert result["sharpe_ratio"] == pytest.approx(correct_sharpe)
    assert result["sharpe_ratio"] != pytest.approx(wrong_sharpe_hardcoded_252)


# --------------------------------------------------------------------------
# Warm-up guard matrix (characterization, not a mandate to add guards
# everywhere - documents current behavior on short histories)
# --------------------------------------------------------------------------


@pytest.mark.parametrize("n_bars", [5, 10, 20])
@pytest.mark.parametrize("strategy_key", list(STRATEGY_NAMES.keys()))
def test_backtest_does_not_crash_on_short_history(strategy_key, n_bars, synthetic_ohlcv):
    df = synthetic_ohlcv(n_bars)
    result = backtest_strategy(df, strategy_key=strategy_key, interval="1d", initial_capital=10000.0)
    assert "error" not in result or n_bars < 5
