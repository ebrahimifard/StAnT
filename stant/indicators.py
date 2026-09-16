"""Technical indicators calculation engine for StAnT.

Provides vectorised calculation of common and advanced quantitative technical indicators
including Moving Averages, RSI, MACD, Bollinger Bands, Stochastic Oscillator,
ATR, Supertrend, ADX, VWAP, Ichimoku Kinko Hyo, Keltner Channels, Parabolic SAR,
Money Flow Index (MFI), and Volume-Price Trend (VPT).
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def calc_sma(df: pd.DataFrame, window: int = 20, column: str = "Close", min_periods: int = 1) -> pd.Series:
    """Calculate Simple Moving Average."""
    return df[column].rolling(window=window, min_periods=min_periods).mean()


def calc_ema(df: pd.DataFrame, span: int = 20, column: str = "Close") -> pd.Series:
    """Calculate Exponential Moving Average."""
    return df[column].ewm(span=span, adjust=False, min_periods=1).mean()


def calc_rsi(df: pd.DataFrame, period: int = 14, column: str = "Close") -> pd.Series:
    """Calculate Relative Strength Index (RSI) using Wilder's smoothing."""
    delta = df[column].diff()
    gain = delta.clip(lower=0)
    loss = -1 * delta.clip(upper=0)

    # Wilder's method: seed the first average with a simple mean of the first
    # `period` gains/losses, then smooth every bar after that with alpha=1/period.
    # Setting everything before the seed bar to NaN makes ewm() reset its
    # recursion exactly at the seed instead of blending in the pre-seed bars.
    gain_seeded = gain.copy()
    loss_seeded = loss.copy()
    if len(gain_seeded) > period:
        seed_gain = gain.iloc[1 : period + 1].mean()
        seed_loss = loss.iloc[1 : period + 1].mean()
        gain_seeded.iloc[:period] = np.nan
        gain_seeded.iloc[period] = seed_gain
        loss_seeded.iloc[:period] = np.nan
        loss_seeded.iloc[period] = seed_loss
    else:
        gain_seeded.iloc[:] = np.nan
        loss_seeded.iloc[:] = np.nan

    avg_gain = gain_seeded.ewm(alpha=1 / period, adjust=False).mean()
    avg_loss = loss_seeded.ewm(alpha=1 / period, adjust=False).mean()

    rs = avg_gain / (avg_loss.replace(0, np.nan))
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50.0)


def calc_macd(
    df: pd.DataFrame, fast: int = 12, slow: int = 26, signal: int = 9, column: str = "Close"
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Calculate MACD line, Signal line, and MACD Histogram."""
    ema_fast = calc_ema(df, span=fast, column=column)
    ema_slow = calc_ema(df, span=slow, column=column)
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False, min_periods=1).mean()
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram


def calc_bollinger_bands(
    df: pd.DataFrame, window: int = 20, num_std: float = 2.0, column: str = "Close"
) -> tuple[pd.Series, pd.Series, pd.Series, pd.Series, pd.Series]:
    """Calculate Bollinger Bands (Upper, Middle, Lower, %B, Bandwidth)."""
    middle = calc_sma(df, window=window, column=column)
    std = df[column].rolling(window=window, min_periods=1).std().fillna(0.0)
    upper = middle + (std * num_std)
    lower = middle - (std * num_std)

    denom = upper - lower
    pct_b = np.where(denom != 0, (df[column] - lower) / denom, 0.5)
    pct_b_series = pd.Series(pct_b, index=df.index)

    bandwidth = np.where(middle != 0, (upper - lower) / middle, 0.0)
    bandwidth_series = pd.Series(bandwidth, index=df.index)

    return upper, middle, lower, pct_b_series, bandwidth_series


def calc_stochastic(
    df: pd.DataFrame, k_window: int = 14, d_window: int = 3
) -> tuple[pd.Series, pd.Series]:
    """Calculate Stochastic Oscillator (%K and %D)."""
    low_min = df["Low"].rolling(window=k_window, min_periods=1).min()
    high_max = df["High"].rolling(window=k_window, min_periods=1).max()
    denom = high_max - low_min

    stoch_k = np.where(denom != 0, 100 * ((df["Close"] - low_min) / denom), 50.0)
    stoch_k_series = pd.Series(stoch_k, index=df.index)
    stoch_d_series = stoch_k_series.rolling(window=d_window, min_periods=1).mean()

    return stoch_k_series, stoch_d_series


def calc_atr(df: pd.DataFrame, window: int = 14) -> pd.Series:
    """Calculate Average True Range (ATR)."""
    high = df["High"]
    low = df["Low"]
    close = df["Close"]

    prev_close = close.shift(1)
    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()

    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1 / window, min_periods=1, adjust=False).mean()
    return atr


def calc_supertrend(
    df: pd.DataFrame, period: int = 10, multiplier: float = 3.0
) -> tuple[pd.Series, pd.Series]:
    """Calculate Supertrend indicator and Trend Direction (+1 for Up, -1 for Down)."""
    atr = calc_atr(df, window=period)
    hl2 = (df["High"] + df["Low"]) / 2

    basic_ub = hl2 + (multiplier * atr)
    basic_lb = hl2 - (multiplier * atr)

    n = len(df)
    final_ub = np.zeros(n)
    final_lb = np.zeros(n)
    supertrend = np.zeros(n)
    direction = np.ones(n, dtype=int)

    close = df["Close"].values

    for i in range(1, n):
        if basic_ub.iloc[i] < final_ub[i - 1] or close[i - 1] > final_ub[i - 1]:
            final_ub[i] = basic_ub.iloc[i]
        else:
            final_ub[i] = final_ub[i - 1]

        if basic_lb.iloc[i] > final_lb[i - 1] or close[i - 1] < final_lb[i - 1]:
            final_lb[i] = basic_lb.iloc[i]
        else:
            final_lb[i] = final_lb[i - 1]

        if direction[i - 1] == 1:
            if close[i] < final_lb[i]:
                direction[i] = -1
                supertrend[i] = final_ub[i]
            else:
                direction[i] = 1
                supertrend[i] = final_lb[i]
        else:
            if close[i] > final_ub[i]:
                direction[i] = 1
                supertrend[i] = final_lb[i]
            else:
                direction[i] = -1
                supertrend[i] = final_ub[i]

    st_series = pd.Series(supertrend, index=df.index)
    dir_series = pd.Series(direction, index=df.index)
    return st_series, dir_series


def calc_adx(df: pd.DataFrame, window: int = 14) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Calculate Average Directional Index (ADX), +DI, and -DI."""
    high = df["High"]
    low = df["Low"]
    close = df["Close"]

    up_move = high.diff()
    down_move = -low.diff()

    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)

    tr = calc_atr(df, window=1)
    atr = tr.ewm(alpha=1 / window, min_periods=1, adjust=False).mean()

    plus_di = 100 * (pd.Series(plus_dm, index=df.index).ewm(alpha=1 / window, min_periods=1).mean() / atr.replace(0, np.nan))
    minus_di = 100 * (pd.Series(minus_dm, index=df.index).ewm(alpha=1 / window, min_periods=1).mean() / atr.replace(0, np.nan))

    di_sum = (plus_di + minus_di).replace(0, np.nan)
    dx = 100 * ((plus_di - minus_di).abs() / di_sum)
    adx = dx.ewm(alpha=1 / window, min_periods=1).mean().fillna(0.0)

    return adx.fillna(0.0), plus_di.fillna(0.0), minus_di.fillna(0.0)


def calc_vwap(df: pd.DataFrame) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Calculate VWAP and Upper/Lower 1 std-dev bands."""
    tp = (df["High"] + df["Low"] + df["Close"]) / 3
    cum_tp_vol = (tp * df["Volume"]).cumsum()
    cum_vol = df["Volume"].cumsum().replace(0, np.nan)
    vwap = cum_tp_vol / cum_vol

    dev = (tp - vwap) ** 2
    cum_dev = (dev * df["Volume"]).cumsum()
    std_dev = np.sqrt(cum_dev / cum_vol).fillna(0.0)

    vwap_upper = vwap + std_dev
    vwap_lower = vwap - std_dev

    return vwap.fillna(df["Close"]), vwap_upper.fillna(df["Close"]), vwap_lower.fillna(df["Close"])


def calc_ichimoku(
    df: pd.DataFrame, tenkan_period: int = 9, kijun_period: int = 26, senkou_b_period: int = 52
) -> tuple[pd.Series, pd.Series, pd.Series, pd.Series, pd.Series]:
    """Calculate Ichimoku Kinko Hyo components (Tenkan-sen, Kijun-sen, Senkou A, Senkou B, Chikou)."""
    high = df["High"]
    low = df["Low"]
    close = df["Close"]

    tenkan_sen = (high.rolling(window=tenkan_period, min_periods=1).max() + low.rolling(window=tenkan_period, min_periods=1).min()) / 2
    kijun_sen = (high.rolling(window=kijun_period, min_periods=1).max() + low.rolling(window=kijun_period, min_periods=1).min()) / 2

    senkou_span_a = ((tenkan_sen + kijun_sen) / 2).shift(kijun_period)
    senkou_span_b = ((high.rolling(window=senkou_b_period, min_periods=1).max() + low.rolling(window=senkou_b_period, min_periods=1).min()) / 2).shift(kijun_period)

    chikou_span = close.shift(-kijun_period)

    return (
        tenkan_sen,
        kijun_sen,
        senkou_span_a.fillna(tenkan_sen),
        senkou_span_b.fillna(kijun_sen),
        chikou_span.fillna(close),
    )


def calc_keltner_channels(
    df: pd.DataFrame, window: int = 20, atr_multiplier: float = 2.0
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Calculate Keltner Channels (Upper, Middle EMA, Lower)."""
    middle = calc_ema(df, span=window)
    atr = calc_atr(df, window=10)
    upper = middle + (atr * atr_multiplier)
    lower = middle - (atr * atr_multiplier)
    return upper, middle, lower


def calc_parabolic_sar(
    df: pd.DataFrame, af_start: float = 0.02, af_step: float = 0.02, af_max: float = 0.20
) -> tuple[pd.Series, pd.Series]:
    """Calculate Parabolic SAR (Stop and Reverse) and Direction (+1 Up, -1 Down)."""
    high = df["High"].values
    low = df["Low"].values
    n = len(df)

    sar = np.zeros(n)
    direction = np.ones(n, dtype=int)

    if n < 2:
        return pd.Series(high, index=df.index), pd.Series(direction, index=df.index)

    is_uptrend = True
    af = af_start
    ep = high[0]
    sar[0] = low[0]

    for i in range(1, n):
        prev_sar = sar[i - 1]

        if is_uptrend:
            sar[i] = prev_sar + af * (ep - prev_sar)
            sar[i] = min(sar[i], low[i - 1], low[i - 2] if i >= 2 else low[i - 1])

            if low[i] < sar[i]:
                is_uptrend = False
                sar[i] = ep
                ep = low[i]
                af = af_start
            else:
                if high[i] > ep:
                    ep = high[i]
                    af = min(af + af_step, af_max)
        else:
            sar[i] = prev_sar + af * (ep - prev_sar)
            sar[i] = max(sar[i], high[i - 1], high[i - 2] if i >= 2 else high[i - 1])

            if high[i] > sar[i]:
                is_uptrend = True
                sar[i] = ep
                ep = high[i]
                af = af_start
            else:
                if low[i] < ep:
                    ep = low[i]
                    af = min(af + af_step, af_max)

        direction[i] = 1 if is_uptrend else -1

    return pd.Series(sar, index=df.index), pd.Series(direction, index=df.index)


def calc_mfi(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Calculate Money Flow Index (MFI)."""
    tp = (df["High"] + df["Low"] + df["Close"]) / 3
    rmf = tp * df["Volume"]

    pos_flow = np.where(tp > tp.shift(1), rmf, 0.0)
    neg_flow = np.where(tp < tp.shift(1), rmf, 0.0)

    pos_mf = pd.Series(pos_flow, index=df.index).rolling(window=period, min_periods=1).sum()
    neg_mf = pd.Series(neg_flow, index=df.index).rolling(window=period, min_periods=1).sum().replace(0, np.nan)

    mfr = pos_mf / neg_mf
    mfi = 100 - (100 / (1 + mfr))
    return mfi.fillna(50.0)


def calc_vpt(df: pd.DataFrame) -> pd.Series:
    """Calculate Volume-Price Trend (VPT)."""
    close_diff = df["Close"].pct_change().fillna(0.0)
    vpt = (close_diff * df["Volume"]).cumsum()
    return vpt.fillna(0.0)


def calculate_all_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Enrich dataframe with full suite of standard and SOTA technical indicators."""
    res = df.copy()

    res["SMA_20"] = calc_sma(res, window=20)
    res["SMA_50"] = calc_sma(res, window=50)
    res["SMA_200"] = calc_sma(res, window=200, min_periods=200)

    res["EMA_12"] = calc_ema(res, span=12)
    res["EMA_26"] = calc_ema(res, span=26)
    res["EMA_50"] = calc_ema(res, span=50)

    res["RSI_14"] = calc_rsi(res, period=14)

    macd, macd_sig, macd_hist = calc_macd(res)
    res["MACD_Line"] = macd
    res["MACD_Signal"] = macd_sig
    res["MACD_Hist"] = macd_hist

    bb_u, bb_m, bb_l, bb_pct, bb_bw = calc_bollinger_bands(res)
    res["BB_Upper"] = bb_u
    res["BB_Middle"] = bb_m
    res["BB_Lower"] = bb_l
    res["BB_PctB"] = bb_pct
    res["BB_Bandwidth"] = bb_bw

    stoch_k, stoch_d = calc_stochastic(res)
    res["Stoch_K"] = stoch_k
    res["Stoch_D"] = stoch_d

    res["ATR_14"] = calc_atr(res, window=14)

    st, st_dir = calc_supertrend(res)
    res["Supertrend"] = st
    res["Supertrend_Dir"] = st_dir

    adx, pdi, mdi = calc_adx(res)
    res["ADX"] = adx
    res["Plus_DI"] = pdi
    res["Minus_DI"] = mdi

    vwap, vwap_u, vwap_l = calc_vwap(res)
    res["VWAP"] = vwap
    res["VWAP_Upper"] = vwap_u
    res["VWAP_Lower"] = vwap_l

    tenkan, kijun, senkou_a, senkou_b, chikou = calc_ichimoku(res)
    res["Tenkan_Sen"] = tenkan
    res["Kijun_Sen"] = kijun
    res["Senkou_Span_A"] = senkou_a
    res["Senkou_Span_B"] = senkou_b
    res["Chikou_Span"] = chikou

    kc_u, kc_m, kc_l = calc_keltner_channels(res)
    res["KC_Upper"] = kc_u
    res["KC_Middle"] = kc_m
    res["KC_Lower"] = kc_l

    psar, psar_dir = calc_parabolic_sar(res)
    res["PSAR"] = psar
    res["PSAR_Dir"] = psar_dir

    res["MFI_14"] = calc_mfi(res)
    res["VPT"] = calc_vpt(res)

    return res
