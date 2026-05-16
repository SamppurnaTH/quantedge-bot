"""
MACD (Moving Average Convergence Divergence)
Measures momentum and trend shifts.
"""

import pandas as pd

def compute_macd(series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
    """
    Compute MACD, Signal line, and Histogram.
    """
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    histogram = macd_line - signal_line
    
    return pd.DataFrame({
        'macd':      macd_line.round(4),
        'signal':    signal_line.round(4),
        'histogram': histogram.round(4)
    })
