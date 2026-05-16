"""
Bollinger Bands
Measures volatility and potential overextended price levels.
"""

import pandas as pd

def compute_bollinger_bands(series: pd.Series, period: int = 20, std_dev: int = 2) -> pd.DataFrame:
    """
    Compute Upper, Middle (SMA), and Lower Bollinger Bands.
    """
    sma = series.rolling(window=period).mean()
    std = series.rolling(window=period).std()
    
    upper = sma + (std * std_dev)
    lower = sma - (std * std_dev)
    
    return pd.DataFrame({
        'bb_upper':  upper.round(2),
        'bb_mid':    sma.round(2),
        'bb_lower':  lower.round(2)
    })
