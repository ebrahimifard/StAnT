# Known Limitations

StAnT's backtest, strategy, and portfolio engines have been tested for
internal correctness (see `tests/`), but a correct implementation of a
simplified model is still a simplified model. Read this before trusting any
number here with real money.

## Backtest engine is a directional-signal simulator, not an execution simulator

- **No shorting.** Every strategy is long/flat only (`position` is 0 or 1 in
  `stant/backtest.py`). A "SELL" signal only closes a long position; it never
  opens a short.
- **No partial fills, no slippage model.** Every trade fills 100% of
  available capital at the next bar's open, adjusted only by a flat
  `cost_bps` (default 5 bps). Real fills in fast-moving or illiquid names
  will differ, sometimes substantially.
- **No position sizing.** There's no concept of risking a fixed % of capital
  per trade, no stop-loss enforcement within the simulation loop itself (the
  `trade_setup` stop/target shown in the UI is informational, not simulated).

## Portfolio and screener defaults are survivorship-biased

`DEFAULT_PORTFOLIO_TICKERS` (`stant/portfolio.py`) and the screener's default
list (`stant/multi_analysis.py`) are current mega-cap survivors (AAPL, MSFT,
NVDA, TSLA, GOOGL, AMZN, SPY). Any performance shown using these defaults is
inflated relative to running the same strategy against an unbiased universe
that includes historical laggards and delistings.

## Ensemble is always fully computed

`run_strategy_analysis` evaluates all 11 sub-strategies internally even when
only one strategy is requested (needed to populate the indicators table).
This is a known performance characteristic, tied to the separate production
timeout work, not addressed by this testing pass.

## Raw NaN rows are not filtered before indicator calculation

`build_chart_payload` drops rows with a NaN Open/Close for the chart display
only. The raw frame passed to indicators/backtest/strategies elsewhere is not
filtered. Verified behavior (`tests/test_indicators.py::test_nan_close_mid_series_characterization`):
this is more benign than it sounds - pandas' `ewm()`/`rolling()` with
`min_periods=1` treat a single missing input as "skip", not "propagate NaN
forever", so no NaN leaks into any indicator column. Delta-based indicators
(RSI, MACD) flatline for about 2 bars around the gap (repeating the last
valid reading) before resuming normally. A single missing bar (a holiday or
halt) is unlikely to meaningfully distort a signal; a *string* of missing
bars has not been tested and should be treated with more caution.

## `ml_quant` is not a trained model

Despite the name, `evaluate_ml_quant` (`stant/strategies.py`) is a static,
hand-weighted linear combination of RSI/MACD/Stochastic/MFI/%B/ADX. It has
never been fit or backtested as a model in its own right beyond the same
regression tests applied to every other strategy here. Its labels were
updated (see git history) to say "Composite Quant Score" rather than
"Predictive ... Probability" to avoid implying a calibrated probability.

## Most strategies lack an explicit short-history guard

Only `sma_cross` (needs 200 bars) and `ichimoku` (needs 52 bars) explicitly
HOLD when there isn't enough history. The other 9 strategies will still
produce a directional signal on very short histories (5-20 bars), based on
indicator values computed with degenerate rolling windows.
`tests/test_strategies.py` and `tests/test_backtest.py` characterize this as
a baseline (no crash), not as validated behavior - treat any signal from a
strategy running on fewer than ~50 bars with caution.
