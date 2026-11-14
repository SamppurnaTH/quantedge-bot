"""
Data Fetcher
Fetches OHLCV data from Yahoo Finance and caches it locally as CSV.
Also provides market index data for the market trend filter.
"""

import os
import time
import logging
from datetime import datetime, timedelta, timezone

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

    df = pd.DataFrame()
    last_exc = None
    for attempt in range(3):                      # Yahoo is flaky — retry with backoff
        try:
            df = yf.Ticker(symbol).history(period=period, interval=interval, auto_adjust=True)
            if not df.empty:
                break
        except Exception as exc:                  # network / rate limit
            last_exc = exc
        time.sleep(2 ** attempt)

    if df.empty:
        raise ValueError(f"No data returned for symbol '{symbol}' ({last_exc or 'empty response'}).")

    # Keep only OHLCV columns
    df = df[["Open", "High", "Low", "Close", "Volume"]].copy()
    df.index.name = "Date"
    df.index = pd.to_datetime(df.index)

    # Remove timezone info for cleaner downstream handling
    if df.index.tz is not None:
        df.index = df.index.tz_localize(None)

    df = df[~df.index.duplicated(keep="last")].sort_index()
    df = df.dropna(subset=["Open", "High", "Low", "Close"])
    df = drop_incomplete_bar(df) if interval == "1d" else df

    logger.info("Fetched %d rows for %s", len(df), symbol)

    if save:
        _save_csv(df, symbol)

    return df


def load_stock_data(symbol: str, max_age_hours: float = 12.0) -> pd.DataFrame:
    """
    Load cached CSV data for a symbol.

    Re-fetches when the cache is missing OR older than `max_age_hours`.
    (Previously a cache file was trusted forever, so learning/exits could run
    on weeks-old prices without any warning.)
    """
    path = _csv_path(symbol)
    if os.path.exists(path):
        age_h = (time.time() - os.path.getmtime(path)) / 3600.0
        if age_h <= max_age_hours:
            logger.info("Loading cached data for %s from %s (%.1fh old)", symbol, path, age_h)
            return pd.read_csv(path, index_col="Date", parse_dates=True)
        logger.info("Cache for %s is %.1fh old — refreshing.", symbol, age_h)
        try:
            return fetch_stock_data(symbol)
        except Exception as exc:
            logger.warning("Refresh failed for %s (%s) — using stale cache.", symbol, exc)
            return pd.read_csv(path, index_col="Date", parse_dates=True)
    logger.warning("No cache found for %s — fetching from Yahoo Finance.", symbol)
    return fetch_stock_data(symbol)


# NSE cash session closes 15:30 IST; Yahoo's daily bar is final a bit later.
_IST = timezone(timedelta(hours=5, minutes=30))
_BAR_FINAL_HOUR_IST = 16


def drop_incomplete_bar(df: pd.DataFrame, now: datetime = None) -> pd.DataFrame:
    """
    Remove today's daily bar if the session is not finished yet.

    During market hours Yahoo returns a *partial* bar for today whose
    Close/High/Low/Volume keep changing. Treating it as a completed day is
    look-ahead/repainting: signals appear and disappear intraday and the
    learning engine records outcomes on prices that never closed there.
    """
    if df.empty:
        return df
    now = now or datetime.now(_IST)
    last_date = pd.Timestamp(df.index[-1]).date()
    if last_date == now.date() and now.hour < _BAR_FINAL_HOUR_IST:
        logger.info("Dropping incomplete intraday bar for %s", last_date)
        return df.iloc[:-1]
    return df


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
        df = fetch_stock_data(index, period=period, save=False)
        if df.empty or len(df) < ma_period:
            logger.warning("Insufficient index data for %s — assuming market DOWN (fail safe).", index)
            return False

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
        logger.warning("Could not fetch market index %s: %s — assuming DOWN (fail safe).", index, exc)
        return False
