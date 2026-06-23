# API Reference

## `GET /api/stock`

Fetch historical OHLCV data for a ticker and return chart-ready JSON.

### Query Parameters

| Name | Required | Default | Description |
| --- | --- | --- | --- |
| `ticker` | Yes | None | Stock, ETF, index, or Yahoo Finance symbol. Examples: `AAPL`, `VUSA`, `VUSA.L`. |
| `interval` | No | `1d` | Yahoo Finance interval. Examples: `1m`, `5m`, `15m`, `1h`, `1d`, `1wk`, `1mo`. |
| `start` | No | None | Start date in `YYYY-MM-DD` format. |
| `end` | No | None | End date in `YYYY-MM-DD` format. |

### Example

```text
/api/stock?ticker=VUSA&interval=1d&start=2026-06-01&end=2026-06-10
```

### Successful Response

```json
{
  "ticker": "VUSA.L",
  "requested_ticker": "VUSA",
  "tried_tickers": ["VUSA", "VUSA.L"],
  "interval": "1d",
  "chart_data": [
    {
      "time": "2026-06-01",
      "open": 100.0,
      "high": 101.0,
      "low": 99.0,
      "close": 100.5
    }
  ],
  "volume_data": [
    {
      "time": "2026-06-01",
      "value": 1000000.0,
      "color": "rgba(38, 166, 154, 0.5)"
    }
  ],
  "stats": {
    "highest": 101.0,
    "lowest": 99.0,
    "avg": 100.5,
    "volume": 1000000,
    "return_pct": 0.0
  }
}
```

### Error Responses

Missing ticker:

```json
{
  "error": "Ticker symbol is required"
}
```

Invalid date range:

```json
{
  "error": "Start date must be earlier than or equal to end date"
}
```

No data:

```json
{
  "error": "No data returned for ticker 'XYZ'. Please verify the ticker symbol ..."
}
```

## ETF Symbol Resolution

StAnT first tries the exact ticker the user typed. If that returns no data and the ticker does not already contain a suffix, it tries common Yahoo Finance exchange suffixes:

```text
.L, .AS, .DE, .MI, .PA, .SW
```

This supports common ETF inputs such as `VUSA`, which resolves to `VUSA.L`.

