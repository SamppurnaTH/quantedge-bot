"""
Indicator Engine — single entry point.
Attaches all indicators to a raw OHLCV DataFrame.
"""

import pandas as pd
from indicators.rsi import compute_rsi
from indicators.moving_averages import add_moving_averages
from indicators.atr import compute_atr
from indicators.volume import compute_volume_ma
from indicators.adx import compute_adx
from indicators.macd import compute_macd
from indicators.bollinger import compute_bollinger_bands
from config.settings import INDICATOR_CONFIG


def compute_all_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute and attach all indicators.

    Input:  Raw OHLCV DataFrame (Open, High, Low, Close, Volume)
    Output: Same DataFrame + RSI, SMA, ATR, Volume_MA, ADX, MACD, BB
    """
    df = df.copy()
    
    # Core indicators
    df[f"RSI_{INDICATOR_CONFIG['rsi_period']}"] = compute_rsi(df["Close"])
    df = add_moving_averages(df)
    df[f"ATR_{INDICATOR_CONFIG['atr_period']}"] = compute_atr(df)
    df[f"Volume_MA_{INDICATOR_CONFIG['volume_ma_period']}"] = compute_volume_ma(df["Volume"])
    
    # New indicators
    df["ADX_14"] = compute_adx(df)
    
    macd_df = compute_macd(df["Close"])
    df["MACD"] = macd_df["macd"]
    df["MACD_Signal"] = macd_df["signal"]
    df["MACD_Hist"] = macd_df["histogram"]
    
    bb_df = compute_bollinger_bands(df["Close"])
    df["BB_Upper"] = bb_df["bb_upper"]
    df["BB_Mid"] = bb_df["bb_mid"]
    df["BB_Lower"] = bb_df["bb_lower"]
    
    return df
