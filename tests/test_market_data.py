"""Correctness tests for stant.market_data: interval classification,
auto_adjust pinning (validated against real historical split dates), and the
exchange-suffix fallback loop's behavior under slow/erroring candidates.
"""

from __future__ import annotations

import pandas as pd
import pytest

from stant.market_data import get_stock_data, is_intraday


@pytest.mark.parametrize(
    "interval,expected",
    [
        ("1m", True), ("2m", True), ("5m", True), ("15m", True),
        ("30m", True), ("60m", True), ("90m", True), ("1h", True),
        ("1d", False), ("1wk", False), ("1mo", False), ("3mo", False),
    ],
)
def test_is_intraday(interval, expected):
    assert is_intraday(interval) is expected


def test_is_intraday_does_not_misclassify_monthly_interval():
    """Regression test for a confirmed live bug: the old substring check
    (`"m" in interval`) matched "1mo"/"3mo" because they contain "m", which
    misformatted monthly-interval chart/backtest timestamps as epoch ints."""
    assert is_intraday("1mo") is False
    assert is_intraday("3mo") is False


def test_ohlc_continuity_across_real_aapl_splits(golden_aapl_df):
    """The golden CSV is already split-adjusted (yfinance's auto_adjust
    behavior, now pinned explicitly in fetch_stock_history). Verify there's
    no discontinuous price jump around AAPL's real 2014-06-09 (7:1) and
    2020-08-31 (4:1) splits - an unadjusted feed would show a ~7x/~4x jump."""
    for split_date in ("2014-06-09", "2020-08-31"):
        idx = golden_aapl_df.index.get_indexer([pd.Timestamp(split_date)], method="nearest")[0]
        before = golden_aapl_df["Close"].iloc[idx - 1]
        after = golden_aapl_df["Close"].iloc[idx + 1]
        ratio = after / before
        assert 0.5 < ratio < 2.0, f"Discontinuous jump around {split_date}: ratio={ratio}"


def test_fallback_loop_tries_suffixes_in_order_and_stops_at_first_hit(mocker):
    calls = []

    def fake_history(self, *args, **kwargs):
        calls.append(self.ticker)
        if self.ticker == "VUSA.L":
            return pd.DataFrame({"Close": [1.0], "Open": [1.0], "High": [1.0], "Low": [1.0], "Volume": [1]})
        return pd.DataFrame()

    mocker.patch("yfinance.Ticker.history", fake_history)
    result = get_stock_data("VUSA", interval="1d")

    assert result.resolved_symbol == "VUSA.L"
    assert calls == ["VUSA", "VUSA.L"]


def test_fallback_loop_continues_past_a_raising_candidate(mocker):
    def fake_history(self, *args, **kwargs):
        if self.ticker == "XYZ":
            raise TimeoutError("simulated slow/erroring candidate")
        if self.ticker == "XYZ.L":
            return pd.DataFrame({"Close": [1.0], "Open": [1.0], "High": [1.0], "Low": [1.0], "Volume": [1]})
        return pd.DataFrame()

    mocker.patch("yfinance.Ticker.history", fake_history)
    result = get_stock_data("XYZ", interval="1d")

    assert result.resolved_symbol == "XYZ.L"


def test_fetch_stock_history_passes_auto_adjust_true(mocker):
    captured = {}

    def fake_history(self, **kwargs):
        captured.update(kwargs)
        return pd.DataFrame({"Close": [1.0]})

    mocker.patch("yfinance.Ticker.history", fake_history)

    from stant.market_data import fetch_stock_history

    fetch_stock_history("AAPL", "1d")
    assert captured.get("auto_adjust") is True
