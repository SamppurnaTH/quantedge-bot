# Data Pipeline & Feeding Mechanism

This document explains how the QuantEdge Bot acquires, processes, and stores market data.

## 1. Data Source
The primary data source for the bot is **Yahoo Finance**, accessed via the [`yfinance`](https://github.com/ranaroussi/yfinance) library.

## 2. Data Acquisition Process

### Automatic Fetching
When the bot runs (either locally or via GitHub Actions), it automatically fetches historical data for all symbols listed in `DATA_CONFIG["symbols"]` (defined in `config/settings.py`).

- **Period**: 10 years of historical data (`DATA_CONFIG["period"]`). This covers multiple bull runs, bear markets, and sideways periods for robust strategy validation.
- **Interval**: Daily bars (`DATA_CONFIG["interval"]`).
- **Adjustments**: All prices are auto-adjusted for dividends and splits.

### Market Regime Data
The bot also fetches data for the **NIFTY 50 Index (`^NSEI`)** to determine the overall market regime. This determines whether new BUY signals are allowed and how much capital should be risked.

## 3. Caching & Persistence

To optimize performance and allow for offline analysis, the bot implements a simple CSV-based caching system:

- **Storage Location**: `data/raw/`
- **Format**: Standard CSV with `Date`, `Open`, `High`, `Low`, `Close`, and `Volume`.
- **Logic**:
  1. The bot checks if a CSV file for the symbol exists in `data/raw/`.
  2. If found, it loads the data from the CSV.
  3. If not found (or when running a fresh scan), it downloads the latest data and overwrites the CSV.

## 4. Signal Processing Pipeline
Once data is loaded into a Pandas DataFrame, it goes through the following steps:
1. **Indicator Computation**: SMA, RSI, ATR, and Volume MAs are calculated.
2. **Regime Detection**: The market trend and volatility are analyzed.
3. **Strategy Filtering**: Entry/Exit conditions are checked against the processed data.

## 5. Automation (GitHub Actions)
In the automated GitHub Actions workflow (`.github/workflows/daily_scan.yml`):
- The `data/raw/` directory is **not** committed to the repository (it is in `.gitignore`).
- Every run downloads fresh data to ensure the signals are based on the latest market close.
- Persistent state (like paper trading portfolio) is saved separately in the `state/` directory.

---
*For questions regarding data accuracy or adding new tickers, refer to `config/settings.py`.*
