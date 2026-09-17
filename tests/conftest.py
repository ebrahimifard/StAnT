"""Shared pytest fixtures for StAnT's test suite.

Two data tiers are used across the suite:
  - synthetic series (this file's `synthetic_ohlcv` fixture / helper): fast,
    deterministic, no network - used for the bulk of math/logic tests.
  - golden CSVs (`golden_aapl_df` / `golden_msft_df`): real historical OHLCV
    data already tracked in data/*.csv, loaded directly with pandas (no
    network call). Used only for things synthetic data can't exercise, such
    as real split-date continuity or 40+ years of bounded-indicator checks.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"


def generate_synthetic_ohlcv(n: int = 250, seed: int = 42) -> pd.DataFrame:
    """Deterministic synthetic OHLCV series for fast, network-free tests."""
    dates = pd.date_range(start="2025-01-01", periods=n, freq="D")
    rng = np.random.default_rng(seed)

    t = np.linspace(0, 4 * np.pi, n)
    trend = np.linspace(100, 150, n)
    close = trend + 15 * np.sin(t)
    open_price = close - rng.uniform(-2, 2, n)
    high = np.maximum(close, open_price) + rng.uniform(0.5, 3.0, n)
    low = np.minimum(close, open_price) - rng.uniform(0.5, 3.0, n)
    volume = rng.integers(1_000_000, 10_000_000, n)

    return pd.DataFrame(
        {
            "Open": open_price,
            "High": high,
            "Low": low,
            "Close": close,
            "Volume": volume,
        },
        index=dates,
    )


@pytest.fixture
def synthetic_ohlcv():
    """Factory fixture: synthetic_ohlcv(n=250, seed=42) -> DataFrame."""
    return generate_synthetic_ohlcv


def _load_golden_csv(filename: str) -> pd.DataFrame:
    path = DATA_DIR / filename
    df = pd.read_csv(path, index_col="Date", parse_dates=True)
    return df[["Open", "High", "Low", "Close", "Volume"]]


@pytest.fixture(scope="session")
def golden_aapl_df() -> pd.DataFrame:
    return _load_golden_csv("AAPL_history.csv")


@pytest.fixture(scope="session")
def golden_msft_df() -> pd.DataFrame:
    return _load_golden_csv("MSFT_history.csv")
