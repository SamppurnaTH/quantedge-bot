"""
ATR — Average True Range
Used for volatility-based stop-loss and take-profit calculation.
"""

import pandas as pd
from config.settings import INDICATOR_CONFIG


def compute_atr(df: pd.DataFrame, period: int = INDICATOR_CONFIG["atr_period"]) -> pd.Series:
    """
    Compute ATR using Wilder's smoothing.

    True Range = max(High-Low, |High-PrevClose|, |Low-PrevClose|)
    ATR = Wilder EMA of True Range over `period` bars

    Args:
        df:     DataFrame with High, Low, Close columns
        period: Look-back window (default 14)

    Returns:
        Series of ATR values
    """
    high  = df["High"]
    low   = df["Low"]
    close = df["Close"]

    prev_close = close.shift(1)

    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low  - prev_close).abs(),
    ], axis=1).max(axis=1)

    atr = tr.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    atr.name = f"ATR_{period}"
    return atr
