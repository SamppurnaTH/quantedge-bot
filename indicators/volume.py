"""
Volume Indicators
Volume confirmation: current volume vs N-day average.
"""

import pandas as pd
from config.settings import INDICATOR_CONFIG


def compute_volume_ma(volume: pd.Series, period: int = INDICATOR_CONFIG["volume_ma_period"]) -> pd.Series:
    """
    Simple moving average of volume.

    Args:
        volume: Series of daily volume
        period: Look-back window (default 20)

    Returns:
        Series of volume SMA values
    """
    vma = volume.rolling(window=period, min_periods=period).mean()
    vma.name = f"Volume_MA_{period}"
    return vma


def volume_confirmed(df: pd.DataFrame, multiplier: float = 1.0) -> pd.Series:
    """
    Boolean Series: True where current volume > multiplier × volume MA.

    Args:
        df:         DataFrame with Volume and Volume_MA_20 columns
        multiplier: Threshold multiplier (default 1.0 = above average)

    Returns:
        Boolean Series
    """
    vol_ma_col = f"Volume_MA_{INDICATOR_CONFIG['volume_ma_period']}"
    if vol_ma_col not in df.columns:
        raise ValueError(f"Column '{vol_ma_col}' not found. Run compute_volume_ma first.")
    return df["Volume"] > (df[vol_ma_col] * multiplier)
