"""Shared numerical assumptions used across StAnT's analytical modules.

Keeping these in one place avoids the same assumption being hardcoded twice
with different units (a real bug found in this codebase: backtest.py stored
0.04 as a decimal while portfolio.py stored 4.0 as a percent).
"""

from __future__ import annotations

RISK_FREE_RATE_ANNUAL: float = 0.04
"""Annual risk-free rate, as a decimal (0.04 == 4%). Used by Sharpe/Sortino
calculations in both stant.backtest and stant.portfolio; each module converts
to whatever unit it needs (decimal vs. percent) at the point of use."""
