"""Correctness tests for stant.portfolio: optimizer quality against an
independent reference solver, risk-parity convergence, and the two silent
beta==1.0 fallback paths.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy.optimize import minimize

from stant.market_data import ResolvedStockData
from stant.portfolio import (
    RISK_FREE_RATE_ANNUAL,
    _optimize_max_sharpe,
    _optimize_risk_parity,
    build_optimized_portfolio,
)


def _portfolio_sharpe(w, mean_returns, cov_matrix, rf=RISK_FREE_RATE_ANNUAL):
    port_ret = np.dot(w, mean_returns)
    port_vol = np.sqrt(np.dot(w.T, np.dot(cov_matrix, w)))
    return (port_ret - rf) / port_vol if port_vol > 0 else -999.0


def _solve_max_sharpe_via_scipy(mean_returns, cov_matrix, rf=RISK_FREE_RATE_ANNUAL):
    n = len(mean_returns)

    def neg_sharpe(w):
        return -_portfolio_sharpe(w, mean_returns, cov_matrix, rf)

    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
    bounds = [(0.0, 1.0)] * n
    x0 = np.ones(n) / n
    res = minimize(neg_sharpe, x0, method="SLSQP", bounds=bounds, constraints=constraints)
    return res.x


def test_monte_carlo_max_sharpe_close_to_scipy_optimum():
    mean_returns = np.array([0.08, 0.12, 0.15])
    cov_matrix = np.array(
        [
            [0.0400, 0.0100, 0.0150],
            [0.0100, 0.0900, 0.0200],
            [0.0150, 0.0200, 0.1600],
        ]
    )

    mc_weights = _optimize_max_sharpe(mean_returns, cov_matrix)
    scipy_weights = _solve_max_sharpe_via_scipy(mean_returns, cov_matrix)

    mc_sharpe = _portfolio_sharpe(mc_weights, mean_returns, cov_matrix)
    scipy_sharpe = _portfolio_sharpe(scipy_weights, mean_returns, cov_matrix)

    assert scipy_sharpe > 0  # sanity: the reference solution is meaningful
    assert mc_sharpe >= scipy_sharpe * 0.95, (
        f"Monte-Carlo Sharpe {mc_sharpe:.4f} is more than 5% below the "
        f"scipy-optimal Sharpe {scipy_sharpe:.4f}"
    )


def test_risk_parity_converges_to_roughly_equal_risk_contributions():
    cov_matrix = np.array(
        [
            [0.0400, 0.0250, 0.0100],
            [0.0250, 0.0900, 0.0150],
            [0.0100, 0.0150, 0.1225],
        ]
    )
    w = _optimize_risk_parity(cov_matrix)

    port_vol = np.sqrt(np.dot(w.T, np.dot(cov_matrix, w)))
    marginal_risk = np.dot(cov_matrix, w) / port_vol
    risk_contrib = w * marginal_risk
    target = port_vol / len(w)

    # 20 fixed-step gradient iterations is a crude solver - assert it lands
    # within a documented tolerance band of equal risk contribution, not
    # exact equality.
    assert risk_contrib == pytest.approx(target, rel=0.15)


def _stock_frame(n=100, seed=1):
    dates = pd.date_range("2024-01-01", periods=n, freq="D")
    rng = np.random.default_rng(seed)
    close = 100 + np.cumsum(rng.normal(0, 1, n))
    return pd.DataFrame(
        {
            "Open": close,
            "High": close + 1,
            "Low": close - 1,
            "Close": close,
            "Volume": rng.integers(1_000_000, 5_000_000, n),
        },
        index=dates,
    )


def test_portfolio_beta_falls_back_to_1_when_spy_fetch_raises(mocker):
    aapl_df = _stock_frame(seed=1)
    msft_df = _stock_frame(seed=2)

    def side_effect(ticker, interval="1d", start=None, end=None):
        if ticker == "SPY":
            raise RuntimeError("simulated SPY fetch failure")
        df = aapl_df if ticker == "AAPL" else msft_df
        return ResolvedStockData(frame=df, requested_symbol=ticker, resolved_symbol=ticker, tried_symbols=[ticker])

    mocker.patch("stant.portfolio.get_stock_data", side_effect=side_effect)
    mocker.patch(
        "stant.portfolio.run_strategy_analysis",
        return_value={"ensemble_consensus": {"score": 0.1, "signal": "BUY"}},
    )

    result = build_optimized_portfolio(tickers=["AAPL", "MSFT"], model_type="max_sharpe")
    assert "error" not in result
    assert result["portfolio_beta"] == pytest.approx(1.0)


def test_portfolio_beta_falls_back_to_1_when_aligned_spy_history_too_short(mocker):
    aapl_df = _stock_frame(seed=1)
    msft_df = _stock_frame(seed=2)
    # SPY data with a completely non-overlapping date range - after
    # reindexing to the AAPL/MSFT index and dropping NaNs, zero points align.
    spy_df = _stock_frame(seed=3)
    spy_df.index = pd.date_range("2099-01-01", periods=len(spy_df), freq="D")

    def side_effect(ticker, interval="1d", start=None, end=None):
        if ticker == "SPY":
            return ResolvedStockData(frame=spy_df, requested_symbol="SPY", resolved_symbol="SPY", tried_symbols=["SPY"])
        df = aapl_df if ticker == "AAPL" else msft_df
        return ResolvedStockData(frame=df, requested_symbol=ticker, resolved_symbol=ticker, tried_symbols=[ticker])

    mocker.patch("stant.portfolio.get_stock_data", side_effect=side_effect)
    mocker.patch(
        "stant.portfolio.run_strategy_analysis",
        return_value={"ensemble_consensus": {"score": 0.1, "signal": "BUY"}},
    )

    result = build_optimized_portfolio(tickers=["AAPL", "MSFT"], model_type="max_sharpe")
    assert "error" not in result
    assert result["portfolio_beta"] == pytest.approx(1.0)
