"""Correctness tests for stant.strategies beyond what tests/test_signals.py
already covers (all-11-strategies-run, ensemble-aggregates-11 smoke tests).
"""

from __future__ import annotations

import zoneinfo

import pandas as pd
import pytest

from stant.strategies import ENSEMBLE_WEIGHTS, STRATEGY_NAMES, _format_time, run_strategy_analysis


def test_ensemble_weights_sum_to_one():
    assert sum(ENSEMBLE_WEIGHTS.values()) == pytest.approx(1.0)


def test_ensemble_weights_cover_every_non_ensemble_strategy():
    non_ensemble_keys = set(STRATEGY_NAMES.keys()) - {"ensemble"}
    assert set(ENSEMBLE_WEIGHTS.keys()) == non_ensemble_keys


@pytest.mark.parametrize("n_bars", [5, 10, 20])
@pytest.mark.parametrize("strategy_key", list(STRATEGY_NAMES.keys()))
def test_run_strategy_analysis_does_not_crash_on_short_history(strategy_key, n_bars, synthetic_ohlcv):
    df = synthetic_ohlcv(n_bars)
    analysis = run_strategy_analysis(df, strategy_key=strategy_key, interval="1d")
    active = analysis["active_strategy"]
    assert active["signal"] in {"BUY", "STRONG BUY", "HOLD", "SELL", "STRONG SELL"}
    assert isinstance(active["score"], float)


def test_sma_cross_and_ichimoku_hold_below_their_warmup_length(synthetic_ohlcv):
    """The two strategies with explicit warm-up guards should stay neutral
    (not emit a confident directional signal off insufficient history)."""
    df = synthetic_ohlcv(20)
    analysis = run_strategy_analysis(df, strategy_key="sma_cross", interval="1d")
    assert analysis["active_strategy"]["signal"] == "HOLD"
    assert analysis["active_strategy"]["score"] == pytest.approx(0.0)

    analysis = run_strategy_analysis(df, strategy_key="ichimoku", interval="1d")
    assert analysis["active_strategy"]["signal"] == "HOLD"
    assert analysis["active_strategy"]["score"] == pytest.approx(0.0)


def test_format_time_daily_is_date_string_not_epoch():
    ts = pd.Timestamp("2025-06-15")
    assert _format_time(ts, "1d") == "2025-06-15"


def test_format_time_intraday_is_epoch_int():
    ts = pd.Timestamp("2025-06-15 09:30:00")
    result = _format_time(ts, "1h")
    assert isinstance(result, int)


def test_format_time_monthly_interval_is_not_misclassified_as_intraday():
    """Regression test for the is_intraday() substring bug: '1mo' contains
    'm' and was previously (wrongly) treated as intraday."""
    ts = pd.Timestamp("2025-06-15")
    assert _format_time(ts, "1mo") == "2025-06-15"


def test_format_time_across_dst_transition_is_consistent():
    """Both tz-naive (daily) and tz-aware (intraday) timestamps around a US
    DST transition (2025-03-09) should format without raising and should be
    internally consistent (an intraday epoch value increases monotonically
    with wall-clock time)."""
    before = pd.Timestamp("2025-03-09 01:30:00", tz=zoneinfo.ZoneInfo("America/New_York"))
    after = pd.Timestamp("2025-03-09 03:30:00", tz=zoneinfo.ZoneInfo("America/New_York"))

    t_before = _format_time(before, "1h")
    t_after = _format_time(after, "1h")
    assert isinstance(t_before, int) and isinstance(t_after, int)
    assert t_after > t_before

    naive_before = pd.Timestamp("2025-03-08")
    naive_after = pd.Timestamp("2025-03-10")
    assert _format_time(naive_before, "1d") == "2025-03-08"
    assert _format_time(naive_after, "1d") == "2025-03-10"
