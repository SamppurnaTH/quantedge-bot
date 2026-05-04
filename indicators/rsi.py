"""
Phase 2 — RSI (Relative Strength Index)
Pure pandas implementation — no external TA library dependency.
"""

import pandas as pd
from config.settings import INDICATOR_CONFIG


def compute_rsi(close: pd.Series, period: int = INDICATOR_CONFIG["rsi_period"]) -> pd.Series:
    """
    Compute RSI using Wilder's smoothing (EMA-based).

    Args:
        close:  Series of closing prices
        period: Look-back window (default 14)

    Returns:
        Series of RSI values (0–100), NaN for the first `period` rows
    """
    if len(close) < period + 1:
        raise ValueError(f"Need at least {period + 1} data points to compute RSI.")

    delta = close.diff()

    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    # Wilder's smoothing = EMA with alpha = 1/period
    avg_gain = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()

    rs = avg_gain / avg_loss.replace(0, float("inf"))
    rsi = 100 - (100 / (1 + rs))
    rsi.name = f"RSI_{period}"
    return rsi
