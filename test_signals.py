"""Unit tests for StAnT trading strategies, indicators, backtester, portfolio engine, and endpoints."""

from __future__ import annotations

import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from app import app
from stant.indicators import calculate_all_indicators, calc_rsi, calc_ichimoku, calc_keltner_channels, calc_parabolic_sar
from stant.strategies import run_strategy_analysis, compute_trade_setup, STRATEGY_NAMES
from stant.backtest import backtest_strategy
from stant.multi_analysis import run_multi_stock_screener
from stant.portfolio import build_optimized_portfolio
from stant.market_data import ResolvedStockData


def _generate_synthetic_stock_data(n: int = 250) -> pd.DataFrame:
    dates = pd.date_range(start="2025-01-01", periods=n, freq="D")
    np.random.seed(42)

    t = np.linspace(0, 4 * np.pi, n)
    trend = np.linspace(100, 150, n)
    close = trend + 15 * np.sin(t)
    open_price = close - np.random.uniform(-2, 2, n)
    high = np.maximum(close, open_price) + np.random.uniform(0.5, 3.0, n)
    low = np.minimum(close, open_price) - np.random.uniform(0.5, 3.0, n)
    volume = np.random.randint(1_000_000, 10_000_000, n)

    df = pd.DataFrame(
        {
            "Open": open_price,
            "High": high,
            "Low": low,
            "Close": close,
            "Volume": volume,
        },
        index=dates,
    )
    return df


class IndicatorTests(unittest.TestCase):
    def setUp(self):
        self.df = _generate_synthetic_stock_data(100)

    def test_indicators_calculation_attaches_all_columns(self):
        res = calculate_all_indicators(self.df)
        expected_cols = [
            "SMA_20", "SMA_50", "SMA_200", "EMA_12", "EMA_26",
            "RSI_14", "MACD_Line", "MACD_Signal", "MACD_Hist",
            "BB_Upper", "BB_Lower", "BB_PctB", "Stoch_K", "Stoch_D",
            "ATR_14", "Supertrend", "ADX", "VWAP",
            "Tenkan_Sen", "Kijun_Sen", "Senkou_Span_A", "Senkou_Span_B",
            "KC_Upper", "KC_Lower", "PSAR", "MFI_14", "VPT"
        ]
        for col in expected_cols:
            self.assertIn(col, res.columns, f"Missing indicator column: {col}")

    def test_sota_indicators_math(self):
        t, k, sa, sb, ch = calc_ichimoku(self.df)
        self.assertEqual(len(t), 100)
        
        ku, km, kl = calc_keltner_channels(self.df)
        self.assertTrue((ku >= km).all())
        self.assertTrue((km >= kl).all())

        sar, sdir = calc_parabolic_sar(self.df)
        self.assertEqual(len(sar), 100)


class StrategyAndSignalsTests(unittest.TestCase):
    def setUp(self):
        self.df = _generate_synthetic_stock_data(250)

    def test_all_11_strategies_evaluation(self):
        for strat_key in STRATEGY_NAMES.keys():
            analysis = run_strategy_analysis(self.df, strategy_key=strat_key, interval="1d")
            active = analysis["active_strategy"]
            self.assertIn(active["signal"], ["BUY", "STRONG BUY", "HOLD", "SELL", "STRONG SELL"])
            self.assertIsInstance(active["score"], float)

    def test_ensemble_aggregates_sub_strategies(self):
        analysis = run_strategy_analysis(self.df, strategy_key="ensemble", interval="1d")
        self.assertEqual(len(analysis["sub_strategies"]), 11)


class BacktestingTests(unittest.TestCase):
    def setUp(self):
        self.df = _generate_synthetic_stock_data(250)

    def test_backtest_all_strategies(self):
        for strat_key in ["ensemble", "ichimoku", "keltner_squeeze", "parabolic_sar", "vwap_profile", "ml_quant"]:
            bt = backtest_strategy(self.df, strategy_key=strat_key, interval="1d", initial_capital=10000.0)
            self.assertIn("strategy_return_pct", bt)
            self.assertIn("win_rate_pct", bt)
            self.assertEqual(bt["initial_capital"], 10000.0)


class PortfolioOptimizationTests(unittest.TestCase):
    @patch("stant.portfolio.get_stock_data")
    def test_build_optimized_portfolio_models(self, mock_get_stock_data):
        df_aapl = _generate_synthetic_stock_data(100)
        df_msft = _generate_synthetic_stock_data(100)

        def side_effect(ticker, interval="1d", start=None, end=None):
            df = df_aapl if ticker == "AAPL" else df_msft
            return ResolvedStockData(frame=df, requested_symbol=ticker, resolved_symbol=ticker, tried_symbols=[ticker])

        mock_get_stock_data.side_effect = side_effect

        for model in ["max_sharpe", "risk_parity", "black_litterman", "min_variance", "max_div"]:
            res = build_optimized_portfolio(
                tickers=["AAPL", "MSFT"],
                model_type=model,
                capital=10000.0,
            )
            self.assertNotIn("error", res, f"Portfolio error in model {model}")
            self.assertEqual(len(res["allocations"]), 2)
            self.assertIn("var_95_daily_dollar", res)
            self.assertIn("correlation_matrix", res)


class StrategyApiTests(unittest.TestCase):
    def setUp(self):
        self.client = app.test_client()

    @patch("app.get_stock_data")
    def test_api_stock_includes_strategies(self, mock_get_stock_data):
        df = _generate_synthetic_stock_data(50)
        mock_get_stock_data.return_value = ResolvedStockData(
            frame=df,
            requested_symbol="AAPL",
            resolved_symbol="AAPL",
            tried_symbols=["AAPL"],
        )

        resp = self.client.get("/api/stock?ticker=AAPL&strategy=ichimoku")
        self.assertEqual(resp.status_code, 200)
        data = resp.get_json()
        self.assertIn("active_strategy", data)
        self.assertEqual(data["active_strategy"]["key"], "ichimoku")


if __name__ == "__main__":
    unittest.main(verbosity=2)
