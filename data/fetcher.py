"""
Data Fetcher
Fetches OHLCV data from Yahoo Finance and caches it locally as CSV.
Also provides market index data for the market trend filter.
"""

import os
import logging
import pandas as pd
import yfinance as yf

from config.settings import DATA_CONFIG

logger = logging.getLogger(__name__)


def fetch_stock_data(
    symbol: str,
    period: str = DATA_CONFIG["period"],
    interval: str = DATA_CONFIG["interval"],
    save: bool = True,
) -> pd.DataFrame:
    """
    Download historical OHLCV data for a symbol.

    Args:
        symbol:   Ticker symbol (e.g. 'RELIANCE.NS', 'AAPL')
        period:   yfinance period string ('1y', '2y', 'max', …)
        interval: Bar size ('1d', '1h', …)
        save:     If True, cache the CSV to data/raw/

    Returns:
        DataFrame with columns: Open, High, Low, Close, Volume
    """
    logger.info("Fetching %s | period=%s interval=%s", symbol, period, interval)

    ticker = yf.Ticker(symbol)
    df = ticker.history(period=period, interval=interval, auto_adjust=True)

    if df.empty:
        raise ValueError(f"No data returned for symbol '{symbol}'. Check the ticker.")

    # Keep only OHLCV columns
    df = df[["Open", "High", "Low", "Close", "Volume"]].copy()
    df.index.name = "Date"
    df.index = pd.to_datetime(df.index)

    # Remove timezone info for cleaner downstream handling
    if df.index.tz is not None:
        df.index = df.index.tz_localize(None)

    logger.info("Fetched %d rows for %s", len(df), symbol)

    if save:
        _save_csv(df, symbol)

    return df


def load_stock_data(symbol: str) -> pd.DataFrame:
    """
    Load cached CSV data for a symbol.
    Falls back to fetching if the file doesn't exist.
    """
    path = _csv_path(symbol)
    if os.path.exists(path):
        logger.info("Loading cached data for %s from %s", symbol, path)
        df = pd.read_csv(path, index_col="Date", parse_dates=True)
        return df
    logger.warning("No cache found for %s — fetching from Yahoo Finance.", symbol)
    return fetch_stock_data(symbol)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _csv_path(symbol: str) -> str:
    safe_name = symbol.replace(".", "_")
    return os.path.join(DATA_CONFIG["data_dir"], f"{safe_name}.csv")


def _save_csv(df: pd.DataFrame, symbol: str) -> None:
    os.makedirs(DATA_CONFIG["data_dir"], exist_ok=True)
    path = _csv_path(symbol)
    df.to_csv(path)
    logger.info("Saved data to %s", path)


def fetch_market_trend(
    index: str = DATA_CONFIG["market_index"],
    period: str = DATA_CONFIG["period"],
    ma_period: int = 50,
) -> bool:
    """
    Determine if the broad market is in an uptrend.

    Logic: NIFTY (or any index) Close > SMA(ma_period) → market is UP

    Args:
        index:     Yahoo Finance ticker for the index (e.g. '^NSEI')
        period:    History period to fetch
        ma_period: SMA period for trend determination

    Returns:
        True if market is in uptrend, False otherwise
    """
    try:
        ticker = yf.Ticker(index)
        df = ticker.history(period=period, interval="1d", auto_adjust=True)
        if df.empty or len(df) < ma_period:
            logger.warning("Insufficient index data for %s — assuming market UP.", index)
            return True

        close = df["Close"]
        sma   = close.rolling(window=ma_period).mean()
        market_up = bool(close.iloc[-1] > sma.iloc[-1])
        direction = "UP" if market_up else "DOWN"
        logger.info(
            "Market trend (%s): Close=%.2f SMA%d=%.2f → %s",
            index, close.iloc[-1], ma_period, sma.iloc[-1], direction,
        )
        return market_up
    except Exception as exc:
        logger.warning("Could not fetch market index %s: %s — assuming UP.", index, exc)
        return True
