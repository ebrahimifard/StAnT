"""Institutional portfolio construction and risk optimization engine for StAnT.

Provides vectorised portfolio optimization algorithms (Markowitz MPT Max Sharpe,
Equal Risk Contribution Risk Parity, Black-Litterman Quant Alpha, Minimum Variance,
and Maximum Diversification Ratio), Risk & Return Metrics (Sharpe, Sortino, VaR 95%,
CVaR 95%, Beta vs SPY, Drawdown), and Actionable Share Order Allocations.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from stant.config import RISK_FREE_RATE_ANNUAL
from stant.market_data import get_stock_data
from stant.strategies import run_strategy_analysis


DEFAULT_PORTFOLIO_TICKERS = ["AAPL", "MSFT", "NVDA", "TSLA", "GOOGL", "AMZN", "SPY"]


def _optimize_max_sharpe(
    mean_returns: np.ndarray,
    cov_matrix: np.ndarray,
    risk_free_rate: float = RISK_FREE_RATE_ANNUAL,
) -> np.ndarray:
    """Iterative optimization to find Max Sharpe Ratio weights."""
    n = len(mean_returns)
    best_sharpe = -999.0
    best_weights = np.ones(n) / n

    # Monte Carlo & Dirichlet optimization search
    np.random.seed(42)
    num_portfolios = 5000

    for _ in range(num_portfolios):
        w = np.random.dirichlet(np.ones(n))
        p_ret = np.dot(w, mean_returns)
        p_vol = np.sqrt(np.dot(w.T, np.dot(cov_matrix, w)))
        if p_vol > 0:
            sharpe = (p_ret - risk_free_rate) / p_vol
            if sharpe > best_sharpe:
                best_sharpe = sharpe
                best_weights = w

    return best_weights


def _optimize_min_variance(cov_matrix: np.ndarray) -> np.ndarray:
    """Optimization for Minimum Variance portfolio weights."""
    n = cov_matrix.shape[0]
    inv_cov = np.linalg.pinv(cov_matrix)
    ones = np.ones(n)
    w = np.dot(inv_cov, ones) / np.dot(ones.T, np.dot(inv_cov, ones))
    w = np.clip(w, 0, None)
    w_sum = np.sum(w)
    return w / w_sum if w_sum > 0 else np.ones(n) / n


def _optimize_risk_parity(cov_matrix: np.ndarray) -> np.ndarray:
    """Equal Risk Contribution (Risk Parity) portfolio solver."""
    n = cov_matrix.shape[0]
    w = 1.0 / np.sqrt(np.diag(cov_matrix))
    w = w / np.sum(w)

    for _ in range(20):
        p_vol = np.sqrt(np.dot(w.T, np.dot(cov_matrix, w)))
        marginal_risk = np.dot(cov_matrix, w) / (p_vol if p_vol > 0 else 1.0)
        risk_contrib = w * marginal_risk
        target_risk = p_vol / n
        diff = risk_contrib - target_risk
        w = w - 0.1 * diff
        w = np.clip(w, 0.01, None)
        w = w / np.sum(w)

    return w


def _optimize_black_litterman(
    mean_returns: np.ndarray, cov_matrix: np.ndarray, quant_scores: np.ndarray, tau: float = 0.05
) -> np.ndarray:
    """Black-Litterman model blending equilibrium returns with quantitative strategy views."""
    n = len(mean_returns)
    # Convert quant scores (-1 to +1) into views vector
    views = mean_returns + (quant_scores * 0.15)
    p_matrix = np.eye(n)
    omega = np.diag(np.diag(np.dot(p_matrix, np.dot(tau * cov_matrix, p_matrix.T))))

    tau_sigma_inv = np.linalg.pinv(tau * cov_matrix)
    omega_inv = np.linalg.pinv(omega)

    post_cov_inv = tau_sigma_inv + np.dot(p_matrix.T, np.dot(omega_inv, p_matrix))
    post_cov = np.linalg.pinv(post_cov_inv)

    post_mean = np.dot(post_cov, np.dot(tau_sigma_inv, mean_returns) + np.dot(p_matrix.T, np.dot(omega_inv, views)))

    return _optimize_max_sharpe(post_mean, cov_matrix)


def _optimize_max_diversification(mean_returns: np.ndarray, cov_matrix: np.ndarray) -> np.ndarray:
    """Maximize Portfolio Diversification Ratio."""
    n = len(mean_returns)
    stds = np.sqrt(np.diag(cov_matrix))
    best_div = -999.0
    best_weights = np.ones(n) / n

    np.random.seed(42)
    for _ in range(4000):
        w = np.random.dirichlet(np.ones(n))
        weighted_stds = np.dot(w, stds)
        p_vol = np.sqrt(np.dot(w.T, np.dot(cov_matrix, w)))
        if p_vol > 0:
            div_ratio = weighted_stds / p_vol
            if div_ratio > best_div:
                best_div = div_ratio
                best_weights = w

    return best_weights


def build_optimized_portfolio(
    tickers: list[str] | None = None,
    model_type: str = "max_sharpe",
    capital: float = 10000.0,
    risk_tolerance: str = "moderate",
    goal: str = "growth",
    interval: str = "1d",
) -> dict[str, Any]:
    """Build and optimize portfolio based on user criteria."""
    if not tickers:
        tickers = DEFAULT_PORTFOLIO_TICKERS

    valid_tickers = []
    prices_dict = {}
    quant_dict = {}

    for t_raw in tickers:
        symbol = t_raw.strip().upper()
        if not symbol:
            continue
        try:
            stock_data = get_stock_data(symbol, interval=interval)
            df = stock_data.frame
            if not df.empty and len(df) >= 30:
                valid_tickers.append(stock_data.resolved_symbol)
                prices_dict[stock_data.resolved_symbol] = df["Close"]
                strat_res = run_strategy_analysis(df, strategy_key="ensemble", interval=interval)
                quant_dict[stock_data.resolved_symbol] = strat_res["ensemble_consensus"]
        except Exception:
            continue

    if len(valid_tickers) < 2:
        return {
            "error": "Insufficient valid stock tickers for portfolio optimization (minimum 2 tickers with historical data required)."
        }

    close_df = pd.DataFrame(prices_dict).dropna()
    if len(close_df) < 20:
        return {"error": "Insufficient overlapping price history across selected tickers."}

    returns_df = close_df.pct_change().dropna()
    mean_daily = returns_df.mean()
    cov_daily = returns_df.cov()

    mean_annual = (mean_daily * 252).values
    cov_annual = (cov_daily * 252).values
    stds_annual = np.sqrt(np.diag(cov_annual))

    quant_scores = np.array([quant_dict[t]["score"] for t in valid_tickers])

    # Select portfolio optimization model
    if model_type == "risk_parity":
        raw_weights = _optimize_risk_parity(cov_annual)
    elif model_type == "black_litterman":
        raw_weights = _optimize_black_litterman(mean_annual, cov_annual, quant_scores)
    elif model_type == "min_variance":
        raw_weights = _optimize_min_variance(cov_annual)
    elif model_type == "max_div":
        raw_weights = _optimize_max_diversification(mean_annual, cov_annual)
    else:  # max_sharpe
        raw_weights = _optimize_max_sharpe(mean_annual, cov_annual)

    # Adjust weights based on Risk Tolerance & Investment Goal
    if risk_tolerance == "conservative" or goal == "preservation":
        min_var_w = _optimize_min_variance(cov_annual)
        final_weights = 0.6 * min_var_w + 0.4 * raw_weights
    elif risk_tolerance == "aggressive" or goal == "high_beta":
        high_alpha_w = np.where(quant_scores > 0, quant_scores, 0)
        if np.sum(high_alpha_w) > 0:
            high_alpha_w = high_alpha_w / np.sum(high_alpha_w)
            final_weights = 0.5 * raw_weights + 0.5 * high_alpha_w
        else:
            final_weights = raw_weights
    else:
        final_weights = raw_weights

    final_weights = final_weights / np.sum(final_weights)

    # Portfolio metrics calculations
    exp_return_pct = float(np.dot(final_weights, mean_annual) * 100)
    port_volatility_pct = float(np.sqrt(np.dot(final_weights.T, np.dot(cov_annual, final_weights))) * 100)

    rf_rate = RISK_FREE_RATE_ANNUAL * 100  # stant.config stores this as a decimal; portfolio metrics use percent
    sharpe_ratio = (exp_return_pct - rf_rate) / port_volatility_pct if port_volatility_pct > 0 else 0.0

    # Downside Semi-Deviation for Sortino Ratio
    port_daily_returns = np.dot(returns_df.values, final_weights)
    downside_returns = np.clip(port_daily_returns, None, 0)
    downside_std_annual = float(np.std(downside_returns) * np.sqrt(252) * 100)
    sortino_ratio = (exp_return_pct - rf_rate) / downside_std_annual if downside_std_annual > 0 else 0.0

    # Value at Risk (VaR 95%) & CVaR (95%)
    var_95_daily_pct = abs(float(np.percentile(port_daily_returns, 5))) * 100
    var_95_monthly_pct = var_95_daily_pct * np.sqrt(21)
    var_95_daily_dollar = capital * (var_95_daily_pct / 100)
    var_95_monthly_dollar = capital * (var_95_monthly_pct / 100)

    cvar_tail = port_daily_returns[port_daily_returns <= np.percentile(port_daily_returns, 5)]
    cvar_95_daily_pct = abs(float(np.mean(cvar_tail))) * 100 if len(cvar_tail) > 0 else var_95_daily_pct
    cvar_95_dollar = capital * (cvar_95_daily_pct / 100)

    # Maximum Drawdown calculation
    cum_returns = np.cumprod(1 + port_daily_returns)
    running_max = np.maximum.accumulate(cum_returns)
    drawdowns = (cum_returns - running_max) / running_max
    max_drawdown_pct = abs(float(np.min(drawdowns))) * 100 if len(drawdowns) > 0 else 0.0

    # Benchmark Beta vs SPY - fetch SPY independently when it isn't already one
    # of the requested tickers, instead of silently assuming beta == 1.0.
    portfolio_beta = 1.0
    if "SPY" in close_df.columns:
        spy_ret_aligned = returns_df["SPY"].values
        port_ret_aligned = port_daily_returns
    else:
        try:
            spy_data = get_stock_data("SPY", interval=interval)
            spy_close = spy_data.frame["Close"].reindex(close_df.index).ffill().dropna()
            spy_returns = spy_close.pct_change().dropna()
            aligned_index = returns_df.index.intersection(spy_returns.index)
            spy_ret_aligned = spy_returns.reindex(aligned_index).values
            port_ret_aligned = pd.Series(port_daily_returns, index=returns_df.index).reindex(aligned_index).values
        except Exception:
            spy_ret_aligned = None
            port_ret_aligned = None

    if spy_ret_aligned is not None and len(spy_ret_aligned) > 5:
        cov_spy = np.cov(port_ret_aligned, spy_ret_aligned)[0, 1]
        var_spy = np.var(spy_ret_aligned)
        portfolio_beta = cov_spy / var_spy if var_spy > 0 else 1.0

    # Correlation Matrix
    corr_df = returns_df.corr().round(2)
    corr_matrix_data = {
        "columns": list(corr_df.columns),
        "values": corr_df.values.tolist(),
    }

    # Actionable Order Allocations
    allocations = []
    for i, t in enumerate(valid_tickers):
        w_pct = float(final_weights[i] * 100)
        dollar_alloc = capital * (final_weights[i])
        last_price = float(close_df[t].iloc[-1])
        shares = int(dollar_alloc // last_price) if last_price > 0 else 0
        actual_dollar = round(shares * last_price, 2)

        allocations.append({
            "ticker": t,
            "weight_pct": round(w_pct, 2),
            "target_dollar": round(dollar_alloc, 2),
            "actual_dollar": actual_dollar,
            "share_count": shares,
            "last_price": round(last_price, 2),
            "annual_volatility": round(stds_annual[i] * 100, 1),
            "quant_score": round(quant_scores[i] * 100, 1),
            "quant_signal": quant_dict[t]["signal"],
        })

    # Sort allocations by weight descending
    allocations.sort(key=lambda x: x["weight_pct"], reverse=True)

    return {
        "model_type": model_type,
        "capital": capital,
        "risk_tolerance": risk_tolerance,
        "goal": goal,
        "total_assets": len(valid_tickers),
        "exp_return_pct": round(exp_return_pct, 2),
        "port_volatility_pct": round(port_volatility_pct, 2),
        "sharpe_ratio": round(sharpe_ratio, 2),
        "sortino_ratio": round(sortino_ratio, 2),
        "var_95_daily_dollar": round(var_95_daily_dollar, 2),
        "var_95_monthly_dollar": round(var_95_monthly_dollar, 2),
        "cvar_95_dollar": round(cvar_95_dollar, 2),
        "portfolio_beta": round(portfolio_beta, 2),
        "max_drawdown_pct": round(max_drawdown_pct, 2),
        "allocations": allocations,
        "correlation_matrix": corr_matrix_data,
    }
