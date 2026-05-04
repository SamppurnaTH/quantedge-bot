"""
Phase 2 — Moving Averages
Simple Moving Average (SMA) and Exponential Moving Average (EMA).
"""

import pandas as pd
from config.settings import INDICATOR_CONFIG


def compute_sma(close: pd.Series, period: int) -> pd.Series:
    """
    Simple Moving Average.

    Args:
        close:  Series of closing prices
        period: Look-back window

    Returns:
        Series of SMA values
    """
    sma = close.rolling(window=period, min_periods=period).mean()
    sma.name = f"SMA_{period}"
    return sma


def compute_ema(close: pd.Series, period: int) -> pd.Series:
    """
    Exponential Moving Average.

    Args:
        close:  Series of closing prices
        period: Look-back window

    Returns:
        Series of EMA values
    """
    ema = close.ewm(span=period, min_periods=period, adjust=False).mean()
    ema.name = f"EMA_{period}"
    return ema


def add_moving_averages(df: pd.DataFrame) -> pd.DataFrame:
    """
    Attach MA_50 and MA_200 columns to the DataFrame in-place.

    Returns:
        DataFrame with added SMA_50 and SMA_200 columns
    """
    df = df.copy()
    df[f"SMA_{INDICATOR_CONFIG['ma_fast']}"] = compute_sma(df["Close"], INDICATOR_CONFIG["ma_fast"])
    df[f"SMA_{INDICATOR_CONFIG['ma_slow']}"] = compute_sma(df["Close"], INDICATOR_CONFIG["ma_slow"])
    return df
