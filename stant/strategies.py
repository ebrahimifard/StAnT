"""Trading strategies engine and signal generator for StAnT.

Provides 11 institutional strategy models (SMA Crossover, RSI Reversion, MACD,
Bollinger Bands, Stochastic Oscillator, Supertrend, Ichimoku Kinko Hyo Cloud,
Keltner Channel Volatility Squeeze, Parabolic SAR Reversal, VWAP Order-Flow Profile,
and a hand-weighted Quant Score), Multi-Factor Ensemble Consensus model,
Chart Markers generator, and Risk Management Trade Setup targets.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import pandas as pd

from stant.indicators import calculate_all_indicators
from stant.market_data import is_intraday


@dataclass
class TradeSetup:
    action: str
    entry_price: float
    stop_loss: float
    target_1: float
    target_2: float
    risk_reward_ratio: str
    risk_per_share: float
    reward_per_share: float


@dataclass
class StrategyResult:
    key: str
    name: str
    signal: str
    score: float
    indicator_summary: str
    chart_markers: list[dict[str, Any]]
    trade_setup: dict[str, Any]


STRATEGY_NAMES = {
    "ensemble": "✨ Multi-Factor Ensemble Consensus",
    "sma_cross": "📈 SMA 50/200 Golden & Death Cross",
    "rsi_reversion": "📊 RSI Mean Reversion (14-period)",
    "macd_cross": "⚡ MACD Signal Line Crossover",
    "bollinger": "🌊 Bollinger Bands Reversion & Squeeze",
    "stochastic": "🎯 Stochastic %K/%D Momentum",
    "supertrend": "🚀 Supertrend & ADX Trend Following",
    "ichimoku": "⛩️ Ichimoku Kinko Hyo Cloud Breakout",
    "keltner_squeeze": "💎 Keltner Channel Volatility Compression",
    "parabolic_sar": "📍 Parabolic SAR Trailing Reversal",
    "vwap_profile": "📊 Session VWAP Band Profile Reversion",
    "ml_quant": "🤖 Weighted Quant Score (Multi-Factor)",
}


def _format_time(timestamp: pd.Timestamp, interval: str) -> str | int:
    if is_intraday(interval):
        return int(timestamp.timestamp())
    return timestamp.strftime("%Y-%m-%d")


def compute_trade_setup(df: pd.DataFrame, action: str, last_close: float, atr_val: float) -> TradeSetup:
    """Calculate entry, stop-loss, take-profit targets based on volatility (ATR)."""
    if atr_val <= 0 or np.isnan(atr_val):
        atr_val = last_close * 0.02

    if action in ("BUY", "STRONG BUY"):
        stop_loss = round(last_close - (1.5 * atr_val), 2)
        risk = last_close - stop_loss
        if risk <= 0:
            risk = last_close * 0.02
            stop_loss = round(last_close - risk, 2)
        target_1 = round(last_close + (2.0 * risk), 2)
        target_2 = round(last_close + (3.0 * risk), 2)
        rr_str = "1 : 2.0"
        return TradeSetup(
            action="BUY",
            entry_price=round(last_close, 2),
            stop_loss=stop_loss,
            target_1=target_1,
            target_2=target_2,
            risk_reward_ratio=rr_str,
            risk_per_share=round(risk, 2),
            reward_per_share=round(target_1 - last_close, 2),
        )

    elif action in ("SELL", "STRONG SELL"):
        stop_loss = round(last_close + (1.5 * atr_val), 2)
        risk = stop_loss - last_close
        if risk <= 0:
            risk = last_close * 0.02
            stop_loss = round(last_close + risk, 2)
        target_1 = round(last_close - (2.0 * risk), 2)
        target_2 = round(last_close - (3.0 * risk), 2)
        rr_str = "1 : 2.0"
        return TradeSetup(
            action="SELL",
            entry_price=round(last_close, 2),
            stop_loss=stop_loss,
            target_1=target_1,
            target_2=target_2,
            risk_reward_ratio=rr_str,
            risk_per_share=round(risk, 2),
            reward_per_share=round(last_close - target_1, 2),
        )

    else:
        return TradeSetup(
            action="HOLD",
            entry_price=round(last_close, 2),
            stop_loss=round(last_close * 0.95, 2),
            target_1=round(last_close * 1.05, 2),
            target_2=round(last_close * 1.10, 2),
            risk_reward_ratio="1 : 1.0",
            risk_per_share=round(last_close * 0.05, 2),
            reward_per_share=round(last_close * 0.05, 2),
        )


def evaluate_sma_cross(df: pd.DataFrame, interval: str = "1d") -> StrategyResult:
    sma50 = df["SMA_50"]
    sma200 = df["SMA_200"]

    last_close = float(df["Close"].iloc[-1])
    atr_val = float(df["ATR_14"].iloc[-1])

    if pd.isna(sma200.iloc[-1]):
        signal = "HOLD"
        summary = f"Insufficient history for 200-period SMA ({len(df)} bars available, 200 required)"
        setup = asdict(compute_trade_setup(df, signal, last_close, atr_val))
        return StrategyResult("sma_cross", STRATEGY_NAMES["sma_cross"], signal, 0.0, summary, [], setup)

    frame_reset = df.reset_index()
    time_col = frame_reset.columns[0]

    markers = []
    scores = []

    for i in range(1, len(df)):
        time_val = _format_time(frame_reset.loc[i, time_col], interval)
        curr_diff = sma50.iloc[i] - sma200.iloc[i]
        prev_diff = sma50.iloc[i - 1] - sma200.iloc[i - 1]

        if prev_diff <= 0 and curr_diff > 0:
            markers.append({"time": time_val, "position": "belowBar", "color": "#26a69a", "shape": "arrowUp", "text": "Golden Cross (BUY)"})
            scores.append(1.0)
        elif prev_diff >= 0 and curr_diff < 0:
            markers.append({"time": time_val, "position": "aboveBar", "color": "#ef5350", "shape": "arrowDown", "text": "Death Cross (SELL)"})
            scores.append(-1.0)
        else:
            scores.append(1.0 if curr_diff > 0 else -1.0)

    last_score = scores[-1] if scores else 0.0
    signal = "BUY" if last_score > 0.3 else ("SELL" if last_score < -0.3 else "HOLD")
    summary = f"SMA 50: ${sma50.iloc[-1]:.2f} vs SMA 200: ${sma200.iloc[-1]:.2f}"
    setup = asdict(compute_trade_setup(df, signal, last_close, atr_val))

    return StrategyResult("sma_cross", STRATEGY_NAMES["sma_cross"], signal, float(last_score), summary, markers, setup)


def evaluate_rsi_reversion(df: pd.DataFrame, interval: str = "1d") -> StrategyResult:
    rsi = df["RSI_14"]
    frame_reset = df.reset_index()
    time_col = frame_reset.columns[0]

    markers = []
    last_rsi = float(rsi.iloc[-1])

    for i in range(1, len(df)):
        time_val = _format_time(frame_reset.loc[i, time_col], interval)
        curr_rsi = rsi.iloc[i]
        prev_rsi = rsi.iloc[i - 1]

        if prev_rsi <= 30 and curr_rsi > 30:
            markers.append({"time": time_val, "position": "belowBar", "color": "#26a69a", "shape": "arrowUp", "text": f"RSI Oversold ({curr_rsi:.1f})"})
        elif prev_rsi >= 70 and curr_rsi < 70:
            markers.append({"time": time_val, "position": "aboveBar", "color": "#ef5350", "shape": "arrowDown", "text": f"RSI Overbought ({curr_rsi:.1f})"})

    if last_rsi < 30:
        score = 1.0
        signal = "BUY"
    elif last_rsi > 70:
        score = -1.0
        signal = "SELL"
    elif last_rsi < 45:
        score = 0.4
        signal = "BUY"
    elif last_rsi > 55:
        score = -0.4
        signal = "SELL"
    else:
        score = 0.0
        signal = "HOLD"

    summary = f"RSI (14): {last_rsi:.1f}"
    last_close = float(df["Close"].iloc[-1])
    atr_val = float(df["ATR_14"].iloc[-1])
    setup = asdict(compute_trade_setup(df, signal, last_close, atr_val))

    return StrategyResult("rsi_reversion", STRATEGY_NAMES["rsi_reversion"], signal, score, summary, markers, setup)


def evaluate_macd_cross(df: pd.DataFrame, interval: str = "1d") -> StrategyResult:
    macd = df["MACD_Line"]
    signal_line = df["MACD_Signal"]
    hist = df["MACD_Hist"]

    frame_reset = df.reset_index()
    time_col = frame_reset.columns[0]
    markers = []

    for i in range(1, len(df)):
        time_val = _format_time(frame_reset.loc[i, time_col], interval)
        curr_h = hist.iloc[i]
        prev_h = hist.iloc[i - 1]

        if prev_h <= 0 and curr_h > 0:
            markers.append({"time": time_val, "position": "belowBar", "color": "#26a69a", "shape": "arrowUp", "text": "MACD Bullish Cross"})
        elif prev_h >= 0 and curr_h < 0:
            markers.append({"time": time_val, "position": "aboveBar", "color": "#ef5350", "shape": "arrowDown", "text": "MACD Bearish Cross"})

    last_hist = float(hist.iloc[-1])
    last_macd = float(macd.iloc[-1])
    last_sig = float(signal_line.iloc[-1])

    if last_hist > 0 and last_macd > 0:
        score = 0.8
        signal = "BUY"
    elif last_hist > 0:
        score = 0.4
        signal = "BUY"
    elif last_hist < 0 and last_macd < 0:
        score = -0.8
        signal = "SELL"
    else:
        score = -0.4
        signal = "SELL"

    summary = f"MACD: {last_macd:.2f}, Signal: {last_sig:.2f}, Hist: {last_hist:.2f}"
    last_close = float(df["Close"].iloc[-1])
    atr_val = float(df["ATR_14"].iloc[-1])
    setup = asdict(compute_trade_setup(df, signal, last_close, atr_val))

    return StrategyResult("macd_cross", STRATEGY_NAMES["macd_cross"], signal, score, summary, markers, setup)


def evaluate_bollinger(df: pd.DataFrame, interval: str = "1d") -> StrategyResult:
    close = df["Close"]
    upper = df["BB_Upper"]
    lower = df["BB_Lower"]
    pct_b = df["BB_PctB"]

    frame_reset = df.reset_index()
    time_col = frame_reset.columns[0]
    markers = []

    for i in range(1, len(df)):
        time_val = _format_time(frame_reset.loc[i, time_col], interval)
        c = close.iloc[i]
        l = lower.iloc[i]
        u = upper.iloc[i]

        if c <= l:
            markers.append({"time": time_val, "position": "belowBar", "color": "#26a69a", "shape": "arrowUp", "text": "Lower BB Touch (BUY)"})
        elif c >= u:
            markers.append({"time": time_val, "position": "aboveBar", "color": "#ef5350", "shape": "arrowDown", "text": "Upper BB Touch (SELL)"})

    last_pct = float(pct_b.iloc[-1])
    if last_pct < 0.1:
        score = 0.9
        signal = "BUY"
    elif last_pct < 0.35:
        score = 0.4
        signal = "BUY"
    elif last_pct > 0.9:
        score = -0.9
        signal = "SELL"
    elif last_pct > 0.65:
        score = -0.4
        signal = "SELL"
    else:
        score = 0.0
        signal = "HOLD"

    summary = f"BB Upper: ${upper.iloc[-1]:.2f}, Lower: ${lower.iloc[-1]:.2f}, %B: {last_pct:.2f}"
    last_close = float(df["Close"].iloc[-1])
    atr_val = float(df["ATR_14"].iloc[-1])
    setup = asdict(compute_trade_setup(df, signal, last_close, atr_val))

    return StrategyResult("bollinger", STRATEGY_NAMES["bollinger"], signal, score, summary, markers, setup)


def evaluate_stochastic(df: pd.DataFrame, interval: str = "1d") -> StrategyResult:
    k = df["Stoch_K"]
    d = df["Stoch_D"]

    frame_reset = df.reset_index()
    time_col = frame_reset.columns[0]
    markers = []

    for i in range(1, len(df)):
        time_val = _format_time(frame_reset.loc[i, time_col], interval)
        prev_diff = k.iloc[i - 1] - d.iloc[i - 1]
        curr_diff = k.iloc[i] - d.iloc[i]

        if prev_diff <= 0 and curr_diff > 0 and k.iloc[i] < 30:
            markers.append({"time": time_val, "position": "belowBar", "color": "#26a69a", "shape": "arrowUp", "text": "Stoch Oversold Cross"})
        elif prev_diff >= 0 and curr_diff < 0 and k.iloc[i] > 70:
            markers.append({"time": time_val, "position": "aboveBar", "color": "#ef5350", "shape": "arrowDown", "text": "Stoch Overbought Cross"})

    last_k = float(k.iloc[-1])
    last_d = float(d.iloc[-1])

    if last_k < 20 and last_k > last_d:
        score = 0.85
        signal = "BUY"
    elif last_k > 80 and last_k < last_d:
        score = -0.85
        signal = "SELL"
    elif last_k > last_d:
        score = 0.3
        signal = "BUY"
    elif last_k < last_d:
        score = -0.3
        signal = "SELL"
    else:
        score = 0.0
        signal = "HOLD"

    summary = f"Stoch %K: {last_k:.1f}, %D: {last_d:.1f}"
    last_close = float(df["Close"].iloc[-1])
    atr_val = float(df["ATR_14"].iloc[-1])
    setup = asdict(compute_trade_setup(df, signal, last_close, atr_val))

    return StrategyResult("stochastic", STRATEGY_NAMES["stochastic"], signal, score, summary, markers, setup)


def evaluate_supertrend(df: pd.DataFrame, interval: str = "1d") -> StrategyResult:
    st = df["Supertrend"]
    st_dir = df["Supertrend_Dir"]
    adx = df["ADX"]

    frame_reset = df.reset_index()
    time_col = frame_reset.columns[0]
    markers = []

    for i in range(1, len(df)):
        time_val = _format_time(frame_reset.loc[i, time_col], interval)
        if st_dir.iloc[i - 1] == -1 and st_dir.iloc[i] == 1:
            markers.append({"time": time_val, "position": "belowBar", "color": "#26a69a", "shape": "arrowUp", "text": "Supertrend Bullish"})
        elif st_dir.iloc[i - 1] == 1 and st_dir.iloc[i] == -1:
            markers.append({"time": time_val, "position": "aboveBar", "color": "#ef5350", "shape": "arrowDown", "text": "Supertrend Bearish"})

    last_dir = int(st_dir.iloc[-1])
    last_adx = float(adx.iloc[-1])
    last_st = float(st.iloc[-1])

    strength_mult = 1.0 if last_adx > 25 else 0.6
    score = last_dir * strength_mult

    signal = "BUY" if score >= 0.5 else ("SELL" if score <= -0.5 else "HOLD")
    summary = f"Supertrend: ${last_st:.2f} ({'Uptrend' if last_dir==1 else 'Downtrend'}), ADX: {last_adx:.1f}"
    last_close = float(df["Close"].iloc[-1])
    atr_val = float(df["ATR_14"].iloc[-1])
    setup = asdict(compute_trade_setup(df, signal, last_close, atr_val))

    return StrategyResult("supertrend", STRATEGY_NAMES["supertrend"], signal, float(score), summary, markers, setup)


def evaluate_ichimoku(df: pd.DataFrame, interval: str = "1d") -> StrategyResult:
    """Ichimoku Kinko Hyo Cloud Breakout strategy."""
    close = df["Close"]
    tenkan = df["Tenkan_Sen"]
    kijun = df["Kijun_Sen"]
    senkou_a = df["Senkou_Span_A"]
    senkou_b = df["Senkou_Span_B"]

    if len(df) < 52:
        signal = "HOLD"
        last_c = float(close.iloc[-1])
        atr_val = float(df["ATR_14"].iloc[-1])
        summary = f"Insufficient history for Ichimoku Cloud (52-period Senkou B) ({len(df)} bars available, 52 required)"
        setup = asdict(compute_trade_setup(df, signal, last_c, atr_val))
        return StrategyResult("ichimoku", STRATEGY_NAMES["ichimoku"], signal, 0.0, summary, [], setup)

    frame_reset = df.reset_index()
    time_col = frame_reset.columns[0]
    markers = []

    for i in range(1, len(df)):
        time_val = _format_time(frame_reset.loc[i, time_col], interval)
        prev_tk_diff = tenkan.iloc[i - 1] - kijun.iloc[i - 1]
        curr_tk_diff = tenkan.iloc[i] - kijun.iloc[i]

        if prev_tk_diff <= 0 and curr_tk_diff > 0:
            markers.append({"time": time_val, "position": "belowBar", "color": "#26a69a", "shape": "arrowUp", "text": "Ichimoku TK Bullish Cross"})
        elif prev_tk_diff >= 0 and curr_tk_diff < 0:
            markers.append({"time": time_val, "position": "aboveBar", "color": "#ef5350", "shape": "arrowDown", "text": "Ichimoku TK Bearish Cross"})

    last_c = float(close.iloc[-1])
    last_tk = float(tenkan.iloc[-1])
    last_kj = float(kijun.iloc[-1])
    cloud_top = max(float(senkou_a.iloc[-1]), float(senkou_b.iloc[-1]))
    cloud_bottom = min(float(senkou_a.iloc[-1]), float(senkou_b.iloc[-1]))

    if last_c > cloud_top and last_tk > last_kj:
        score = 0.95
        signal = "BUY"
    elif last_c > cloud_top:
        score = 0.50
        signal = "BUY"
    elif last_c < cloud_bottom and last_tk < last_kj:
        score = -0.95
        signal = "SELL"
    elif last_c < cloud_bottom:
        score = -0.50
        signal = "SELL"
    else:
        score = 0.0
        signal = "HOLD"

    summary = f"Tenkan: ${last_tk:.2f}, Kijun: ${last_kj:.2f}, Cloud Top: ${cloud_top:.2f}"
    atr_val = float(df["ATR_14"].iloc[-1])
    setup = asdict(compute_trade_setup(df, signal, last_c, atr_val))

    return StrategyResult("ichimoku", STRATEGY_NAMES["ichimoku"], signal, score, summary, markers, setup)


def evaluate_keltner_squeeze(df: pd.DataFrame, interval: str = "1d") -> StrategyResult:
    """Keltner Channel Volatility Compression strategy."""
    close = df["Close"]
    kc_u = df["KC_Upper"]
    kc_l = df["KC_Lower"]
    bb_u = df["BB_Upper"]
    bb_l = df["BB_Lower"]

    frame_reset = df.reset_index()
    time_col = frame_reset.columns[0]
    markers = []

    for i in range(1, len(df)):
        time_val = _format_time(frame_reset.loc[i, time_col], interval)
        c = close.iloc[i]

        if c > kc_u.iloc[i]:
            markers.append({"time": time_val, "position": "belowBar", "color": "#26a69a", "shape": "arrowUp", "text": "Keltner Upper Breakout"})
        elif c < kc_l.iloc[i]:
            markers.append({"time": time_val, "position": "aboveBar", "color": "#ef5350", "shape": "arrowDown", "text": "Keltner Lower Breakdown"})

    last_c = float(close.iloc[-1])
    is_squeeze = (float(bb_u.iloc[-1]) < float(kc_u.iloc[-1])) and (float(bb_l.iloc[-1]) > float(kc_l.iloc[-1]))

    if last_c > float(kc_u.iloc[-1]):
        score = 0.9 if not is_squeeze else 1.0
        signal = "BUY"
    elif last_c < float(kc_l.iloc[-1]):
        score = -0.9 if not is_squeeze else -1.0
        signal = "SELL"
    else:
        score = 0.0
        signal = "HOLD"

    summary = f"KC Upper: ${kc_u.iloc[-1]:.2f}, Lower: ${kc_l.iloc[-1]:.2f} ({'Squeeze Active' if is_squeeze else 'Normal'})"
    atr_val = float(df["ATR_14"].iloc[-1])
    setup = asdict(compute_trade_setup(df, signal, last_c, atr_val))

    return StrategyResult("keltner_squeeze", STRATEGY_NAMES["keltner_squeeze"], signal, score, summary, markers, setup)


def evaluate_parabolic_sar(df: pd.DataFrame, interval: str = "1d") -> StrategyResult:
    """Parabolic SAR Reversal Strategy."""
    sar = df["PSAR"]
    sar_dir = df["PSAR_Dir"]

    frame_reset = df.reset_index()
    time_col = frame_reset.columns[0]
    markers = []

    for i in range(1, len(df)):
        time_val = _format_time(frame_reset.loc[i, time_col], interval)
        if sar_dir.iloc[i - 1] == -1 and sar_dir.iloc[i] == 1:
            markers.append({"time": time_val, "position": "belowBar", "color": "#26a69a", "shape": "arrowUp", "text": "SAR Bullish Reversal"})
        elif sar_dir.iloc[i - 1] == 1 and sar_dir.iloc[i] == -1:
            markers.append({"time": time_val, "position": "aboveBar", "color": "#ef5350", "shape": "arrowDown", "text": "SAR Bearish Reversal"})

    last_dir = int(sar_dir.iloc[-1])
    last_sar = float(sar.iloc[-1])
    score = 0.85 if last_dir == 1 else -0.85
    signal = "BUY" if last_dir == 1 else "SELL"

    summary = f"Parabolic SAR Stop: ${last_sar:.2f} ({'Bullish Trailing' if last_dir==1 else 'Bearish Trailing'})"
    last_close = float(df["Close"].iloc[-1])
    atr_val = float(df["ATR_14"].iloc[-1])
    setup = asdict(compute_trade_setup(df, signal, last_close, atr_val))

    return StrategyResult("parabolic_sar", STRATEGY_NAMES["parabolic_sar"], signal, score, summary, markers, setup)


def evaluate_vwap_profile(df: pd.DataFrame, interval: str = "1d") -> StrategyResult:
    """Session VWAP Band Reversion strategy."""
    close = df["Close"]
    vwap = df["VWAP"]
    vwap_u = df["VWAP_Upper"]
    vwap_l = df["VWAP_Lower"]

    frame_reset = df.reset_index()
    time_col = frame_reset.columns[0]
    markers = []

    for i in range(1, len(df)):
        time_val = _format_time(frame_reset.loc[i, time_col], interval)
        c = close.iloc[i]
        vl = vwap_l.iloc[i]
        vu = vwap_u.iloc[i]

        if c <= vl:
            markers.append({"time": time_val, "position": "belowBar", "color": "#26a69a", "shape": "arrowUp", "text": "Below VWAP -1 Std (BUY)"})
        elif c >= vu:
            markers.append({"time": time_val, "position": "aboveBar", "color": "#ef5350", "shape": "arrowDown", "text": "Above VWAP +1 Std (SELL)"})

    last_c = float(close.iloc[-1])
    last_v = float(vwap.iloc[-1])
    last_u = float(vwap_u.iloc[-1])
    last_l = float(vwap_l.iloc[-1])

    if last_c < last_l:
        score = 0.9
        signal = "BUY"
    elif last_c > last_u:
        score = -0.9
        signal = "SELL"
    elif last_c > last_v:
        score = 0.3
        signal = "BUY"
    else:
        score = -0.3
        signal = "SELL"

    summary = f"VWAP: ${last_v:.2f}, Upper Band: ${last_u:.2f}, Lower Band: ${last_l:.2f}"
    atr_val = float(df["ATR_14"].iloc[-1])
    setup = asdict(compute_trade_setup(df, signal, last_c, atr_val))

    return StrategyResult("vwap_profile", STRATEGY_NAMES["vwap_profile"], signal, score, summary, markers, setup)


def evaluate_ml_quant(df: pd.DataFrame, interval: str = "1d") -> StrategyResult:
    """Hand-weighted multi-factor quant score (static weights, not a trained/fitted model)."""
    rsi = df["RSI_14"].iloc[-1]
    macd_h = df["MACD_Hist"].iloc[-1]
    stoch_k = df["Stoch_K"].iloc[-1]
    adx = df["ADX"].iloc[-1]
    mfi = df["MFI_14"].iloc[-1]
    pct_b = df["BB_PctB"].iloc[-1]

    # Feature weights
    score = 0.0
    score += (50.0 - rsi) / 50.0 * 0.20
    score += (1.0 if macd_h > 0 else -1.0) * 0.20
    score += (50.0 - stoch_k) / 50.0 * 0.15
    score += (50.0 - mfi) / 50.0 * 0.15
    score += (0.5 - pct_b) * 2.0 * 0.15
    score += (0.15 if adx > 25 else 0.0)

    score = float(np.clip(score, -1.0, 1.0))
    prob = round((score + 1.0) / 2.0 * 100, 1)

    if score >= 0.35:
        signal = "STRONG BUY" if score >= 0.7 else "BUY"
    elif score <= -0.35:
        signal = "STRONG SELL" if score <= -0.7 else "SELL"
    else:
        signal = "HOLD"

    frame_reset = df.reset_index()
    time_col = frame_reset.columns[0]
    time_val = _format_time(frame_reset.iloc[-1][time_col], interval)
    markers = [{"time": time_val, "position": "belowBar" if score>0 else "aboveBar", "color": "#26a69a" if score>0 else "#ef5350", "shape": "circle", "text": f"ML Prob {prob}%"}]

    summary = f"Predictive Bullish Probability: {prob}% (Multi-Factor Quant Model)"
    last_close = float(df["Close"].iloc[-1])
    atr_val = float(df["ATR_14"].iloc[-1])
    setup = asdict(compute_trade_setup(df, signal, last_close, atr_val))

    return StrategyResult("ml_quant", STRATEGY_NAMES["ml_quant"], signal, score, summary, markers, setup)


def evaluate_ensemble_consensus(df: pd.DataFrame, interval: str = "1d") -> tuple[StrategyResult, list[StrategyResult]]:
    """Multi-Factor Ensemble Consensus combining all 11 strategy models."""
    sub_results = [
        evaluate_sma_cross(df, interval),
        evaluate_rsi_reversion(df, interval),
        evaluate_macd_cross(df, interval),
        evaluate_bollinger(df, interval),
        evaluate_stochastic(df, interval),
        evaluate_supertrend(df, interval),
        evaluate_ichimoku(df, interval),
        evaluate_keltner_squeeze(df, interval),
        evaluate_parabolic_sar(df, interval),
        evaluate_vwap_profile(df, interval),
        evaluate_ml_quant(df, interval),
    ]

    weights = {
        "sma_cross": 0.10,
        "rsi_reversion": 0.08,
        "macd_cross": 0.10,
        "bollinger": 0.08,
        "stochastic": 0.08,
        "supertrend": 0.10,
        "ichimoku": 0.12,
        "keltner_squeeze": 0.08,
        "parabolic_sar": 0.08,
        "vwap_profile": 0.08,
        "ml_quant": 0.10,
    }

    total_score = sum(res.score * weights.get(res.key, 0.09) for res in sub_results)

    if total_score >= 0.55:
        recommendation = "STRONG BUY"
    elif total_score >= 0.20:
        recommendation = "BUY"
    elif total_score <= -0.55:
        recommendation = "STRONG SELL"
    elif total_score <= -0.20:
        recommendation = "SELL"
    else:
        recommendation = "HOLD"

    merged_markers = []
    seen_marker_keys = set()

    for sub in sub_results:
        for m in sub.chart_markers:
            m_key = (m["time"], m["text"])
            if m_key not in seen_marker_keys:
                seen_marker_keys.add(m_key)
                merged_markers.append(m)

    merged_markers.sort(key=lambda x: str(x["time"]))

    last_close = float(df["Close"].iloc[-1])
    atr_val = float(df["ATR_14"].iloc[-1])
    setup = asdict(compute_trade_setup(df, recommendation, last_close, atr_val))

    summary = f"Consensus Score: {total_score * 100:+.1f}% across 11 Quantitative Strategy Models"

    ensemble_res = StrategyResult(
        key="ensemble",
        name=STRATEGY_NAMES["ensemble"],
        signal=recommendation,
        score=float(total_score),
        indicator_summary=summary,
        chart_markers=merged_markers,
        trade_setup=setup,
    )

    return ensemble_res, sub_results


def run_strategy_analysis(
    df: pd.DataFrame, strategy_key: str = "ensemble", interval: str = "1d"
) -> dict[str, Any]:
    """Run full strategy pipeline across all 11 models."""
    df_ind = calculate_all_indicators(df)

    ensemble_res, sub_results = evaluate_ensemble_consensus(df_ind, interval)

    if strategy_key == "ensemble" or strategy_key not in STRATEGY_NAMES:
        active_result = ensemble_res
    else:
        match_res = [r for r in sub_results if r.key == strategy_key]
        active_result = match_res[0] if match_res else ensemble_res

    indicators_table = []
    last_row = df_ind.iloc[-1]
    indicators_table.append({"name": "RSI (14)", "value": f"{last_row['RSI_14']:.2f}", "signal": "Oversold" if last_row['RSI_14']<30 else ("Overbought" if last_row['RSI_14']>70 else "Neutral")})
    indicators_table.append({"name": "MACD Line / Signal", "value": f"{last_row['MACD_Line']:.2f} / {last_row['MACD_Signal']:.2f}", "signal": "Bullish" if last_row['MACD_Hist']>0 else "Bearish"})
    indicators_table.append({"name": "SMA 50 / 200", "value": f"${last_row['SMA_50']:.2f} / ${last_row['SMA_200']:.2f}", "signal": "Bullish" if last_row['SMA_50']>last_row['SMA_200'] else "Bearish"})
    indicators_table.append({"name": "Ichimoku TK Cross", "value": f"${last_row['Tenkan_Sen']:.2f} / ${last_row['Kijun_Sen']:.2f}", "signal": "Bullish" if last_row['Tenkan_Sen']>last_row['Kijun_Sen'] else "Bearish"})
    indicators_table.append({"name": "Parabolic SAR", "value": f"${last_row['PSAR']:.2f}", "signal": "Bullish" if last_row['PSAR_Dir']==1 else "Bearish"})
    indicators_table.append({"name": "Keltner Bands", "value": f"${last_row['KC_Upper']:.2f} / ${last_row['KC_Lower']:.2f}", "signal": "Overbought" if last_row['Close']>last_row['KC_Upper'] else ("Oversold" if last_row['Close']<last_row['KC_Lower'] else "Neutral")})

    sub_summaries = [asdict(r) for r in sub_results]

    return {
        "active_strategy": asdict(active_result),
        "ensemble_consensus": asdict(ensemble_res),
        "sub_strategies": sub_summaries,
        "indicators_table": indicators_table,
    }
