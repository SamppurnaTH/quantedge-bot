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

    # When avg_loss == 0 the stock has had no down days in the window: RSI is
    # 100 (maximally overbought), NOT 0. The previous implementation replaced
    # 0 with +inf in the denominator, which produced RS=0 -> RSI=0 and flagged
    # straight-up stocks as "deeply oversold" buy candidates.
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.where(avg_loss != 0, 100.0)
    rsi = rsi.where(~((avg_loss == 0) & (avg_gain == 0)), 50.0)   # flat price
    rsi = rsi.where(avg_gain.notna())                             # keep warm-up NaNs
    rsi.name = f"RSI_{period}"
    return rsi
