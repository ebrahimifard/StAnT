"""Correctness tests for stant.indicators.

Covers: RSI Wilder-seed numerical accuracy against an independently
hand-computed reference, RSI boundedness over 40+ years of real data, and
NaN-injection characterization (does a single missing bar poison downstream
indicator values, and if so for how long).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from stant.indicators import calc_rsi, calculate_all_indicators

GOLDEN_DIR = Path(__file__).resolve().parent / "golden"


def test_rsi_matches_hand_computed_wilder_reference():
    """Validate calc_rsi's Wilder-smoothing implementation against a small
    series computed independently by hand (not derived from calc_rsi itself).

    Method, matching Wilder's RSI definition: seed the average gain/loss at
    bar `period` with the simple mean of the first `period` gains/losses,
    then smooth every bar after with an EWM of alpha=1/period. RS = avg_gain
    / avg_loss, RSI = 100 - 100/(1+RS). Bars before the seed have no signal
    and the code fills them with a neutral 50.0.
    """
    golden = pd.read_csv(GOLDEN_DIR / "hand_calc_series.csv")
    df = pd.DataFrame({"Close": golden["close"]})

    rsi = calc_rsi(df, period=5)

    for i, expected in enumerate(golden["expected_rsi_period5"]):
        assert rsi.iloc[i] == pytest.approx(expected, abs=0.05), (
            f"RSI mismatch at bar {i}: got {rsi.iloc[i]}, expected {expected}"
        )


def test_rsi_bounded_0_100_over_real_44yr_history(golden_aapl_df):
    rsi = calc_rsi(golden_aapl_df)
    assert rsi.between(0, 100).all()
    assert not rsi.isna().any()


def test_rsi_bounded_and_no_nan_on_synthetic_data(synthetic_ohlcv):
    df = synthetic_ohlcv(250)
    rsi = calc_rsi(df)
    assert rsi.between(0, 100).all()
    assert not rsi.isna().any()


def test_nan_close_mid_series_characterization(synthetic_ohlcv):
    """Characterize (not assert-as-correct) how a single missing bar (e.g. a
    holiday/halt gap in real data) propagates through indicators.

    Verified actual behavior (not the naive assumption that NaN propagates
    forever): pandas' ewm()/rolling() with min_periods=1 treat a single NaN
    input as "skip", not "propagate NaN" or "reset" - no NaN leaks into any
    indicator column at all. Delta-based indicators (RSI, MACD) need two
    consecutive real closes to produce a fresh gain/loss reading, so a single
    missing close causes a brief ~2-bar "flatline" (the NaN bar and the bar
    right after it repeat the last valid reading) rather than a crash or a
    garbage value. This is documented in docs/LIMITATIONS.md as a known gap:
    raw NaN rows are not filtered before indicator calculation, but the
    actual failure mode is a brief stale reading, not corruption.
    """
    df = synthetic_ohlcv(100)
    df.iloc[50, df.columns.get_loc("Close")] = np.nan

    result = calculate_all_indicators(df)

    for col in ("SMA_20", "EMA_12", "RSI_14", "MACD_Line", "ATR_14"):
        assert not result[col].isna().any(), f"{col} leaked a NaN from the single missing bar"

    # RSI flatlines across the NaN bar (50) and the bar right after it (51),
    # since both need a real prior close to compute a delta, then resumes
    # producing fresh values from bar 52 onward.
    assert result["RSI_14"].iloc[49] == pytest.approx(result["RSI_14"].iloc[50])
    assert result["RSI_14"].iloc[50] == pytest.approx(result["RSI_14"].iloc[51])
    assert result["RSI_14"].iloc[51] != pytest.approx(result["RSI_14"].iloc[52])


def test_ichimoku_and_keltner_and_psar_shapes(synthetic_ohlcv):
    from stant.indicators import calc_ichimoku, calc_keltner_channels, calc_parabolic_sar

    df = synthetic_ohlcv(100)
    tenkan, kijun, senkou_a, senkou_b, chikou = calc_ichimoku(df)
    assert len(tenkan) == 100

    ku, km, kl = calc_keltner_channels(df)
    assert (ku >= km).all()
    assert (km >= kl).all()

    sar, sar_dir = calc_parabolic_sar(df)
    assert len(sar) == 100
    assert set(sar_dir.unique()).issubset({1, -1})
