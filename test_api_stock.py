import unittest
from unittest.mock import patch

import pandas as pd

from app import app
from stant.market_data import ResolvedStockData


def _sample_ohlcv_df() -> pd.DataFrame:
    idx = pd.to_datetime(["2025-01-02", "2025-01-03"])
    return pd.DataFrame(
        {
            "Open": [100.0, 101.0],
            "High": [102.0, 104.0],
            "Low": [99.0, 100.0],
            "Close": [101.0, 103.0],
            "Volume": [1_000_000, 1_500_000],
        },
        index=idx,
    )


class ApiStockTests(unittest.TestCase):
    def setUp(self):
        self.client = app.test_client()

    @patch("app.get_stock_data")
    def test_api_stock_success_payload_shape(self, mock_get_stock_data):
        mock_get_stock_data.return_value = ResolvedStockData(
            frame=_sample_ohlcv_df(),
            requested_symbol="AAPL",
            resolved_symbol="AAPL",
            tried_symbols=["AAPL"],
        )

        resp = self.client.get("/api/stock?ticker=AAPL&interval=1d")
        self.assertEqual(resp.status_code, 200)

        data = resp.get_json()
        self.assertIsInstance(data, dict)
        self.assertEqual(data["ticker"], "AAPL")
        self.assertEqual(data["requested_ticker"], "AAPL")
        self.assertIn("chart_data", data)
        self.assertIn("volume_data", data)
        self.assertIn("stats", data)
        self.assertEqual(len(data["chart_data"]), 2)
        self.assertEqual(len(data["volume_data"]), 2)

    @patch("stant.market_data.fetch_stock_history")
    def test_api_stock_falls_back_to_london_etf_suffix(self, mock_fetch_stock_history):
        empty_df = pd.DataFrame()

        def fake_fetch(symbol, interval, start=None, end=None):
            if symbol == "VUSA.L":
                return _sample_ohlcv_df()
            return empty_df

        mock_fetch_stock_history.side_effect = fake_fetch

        resp = self.client.get("/api/stock?ticker=VUSA&interval=1d")
        self.assertEqual(resp.status_code, 200)

        data = resp.get_json()
        self.assertEqual(data["requested_ticker"], "VUSA")
        self.assertEqual(data["ticker"], "VUSA.L")
        self.assertEqual(data["tried_tickers"][:2], ["VUSA", "VUSA.L"])

    def test_api_stock_rejects_invalid_date_range(self):
        resp = self.client.get(
            "/api/stock?ticker=AAPL&interval=1d&start=2025-12-31&end=2025-01-01"
        )
        self.assertEqual(resp.status_code, 400)

        data = resp.get_json()
        self.assertEqual(
            data["error"], "Start date must be earlier than or equal to end date"
        )

    def test_api_stock_requires_ticker(self):
        resp = self.client.get("/api/stock?interval=1d")
        self.assertEqual(resp.status_code, 400)

        data = resp.get_json()
        self.assertEqual(data["error"], "Ticker symbol is required")


if __name__ == "__main__":
    unittest.main(verbosity=2)
