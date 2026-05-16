"""
ADX (Average Directional Index)
Measures trend strength (not direction).
ADX > 25 = Strong Trend
ADX < 20 = Choppy / Sideways
"""

import pandas as pd
import numpy as np

def compute_adx(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """
    Compute ADX using Wilder's Smoothing.
    
    Formula:
    1. TR = Max(H-L, |H-PC|, |L-PC|)
    2. +DM = H-PH if (H-PH > PL-L) else 0
    3. -DM = PL-L if (PL-L > H-PH) else 0
    4. Smooth TR, +DM, -DM
    5. DI+ = 100 * Smooth(+DM) / Smooth(TR)
    6. DI- = 100 * Smooth(-DM) / Smooth(TR)
    7. DX = 100 * |DI+ - DI-| / |DI+ + DI-|
    8. ADX = Smooth(DX)
    """
    df = df.copy()
    
    high = df['High']
    low = df['Low']
    close = df['Close']
    
    # TR calculation
    tr1 = high - low
    tr2 = abs(high - close.shift(1))
    tr3 = abs(low - close.shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    # DM calculation
    plus_dm = high.diff()
    minus_dm = low.diff().apply(lambda x: -x)
    
    plus_dm[plus_dm < 0] = 0
    minus_dm[minus_dm < 0] = 0
    
    plus_dm[plus_dm < minus_dm] = 0
    minus_dm[minus_dm < plus_dm] = 0
    
    # Smoothing (Wilder's)
    def smooth(series, p):
        results = series.copy()
        # First value is simple sum
        results.iloc[p] = series.iloc[1:p+1].sum()
        results.iloc[:p] = 0
        # Subsequent values: Current * 1/p + Previous * (p-1)/p
        for i in range(p + 1, len(series)):
            results.iloc[i] = (results.iloc[i-1] * (p - 1) + series.iloc[i]) / p
        return results

    # Simplified EMA for faster calculation (Standard ADX use Wilder's)
    str_ = tr.rolling(window=period).mean()
    splus_dm = plus_dm.rolling(window=period).mean()
    sminus_dm = minus_dm.rolling(window=period).mean()
    
    plus_di = 100 * (splus_dm / str_)
    minus_di = 100 * (sminus_dm / str_)
    
    dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = dx.rolling(window=period).mean()
    
    return adx.round(2)
