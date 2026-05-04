"""
Indicator Engine — single entry point.
Attaches all indicators to a raw OHLCV DataFrame.
"""

import pandas as pd
from indicators.rsi import compute_rsi
from indicators.moving_averages import add_moving_averages
from indicators.atr import compute_atr
from indicators.volume import compute_volume_ma
from config.settings import INDICATOR_CONFIG


def compute_all_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute and attach all indicators.

    Input:  Raw OHLCV DataFrame (Open, High, Low, Close, Volume)
    Output: Same DataFrame + RSI_14, SMA_50, SMA_200, ATR_14, Volume_MA_20
    """
    df = df.copy()
    df[f"RSI_{INDICATOR_CONFIG['rsi_period']}"]          = compute_rsi(df["Close"])
    df = add_moving_averages(df)
    df[f"ATR_{INDICATOR_CONFIG['atr_period']}"]          = compute_atr(df)
    df[f"Volume_MA_{INDICATOR_CONFIG['volume_ma_period']}"] = compute_volume_ma(df["Volume"])
    return df
