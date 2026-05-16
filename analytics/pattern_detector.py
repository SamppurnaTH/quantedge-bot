"""
Pattern Detector
Automatically identifies market patterns from OHLCV data.

Patterns detected:
  1. Support & Resistance   — swing lows/highs acting as price magnets
  2. Trend Channel          — rising, falling, or sideways price structure
  3. Candlestick Patterns   — hammer, shooting star, engulfing, inside bar
  4. RSI Divergence         — price/RSI disagreement (leading reversal signal)
  5. Volume Spikes          — unusual volume vs 20-bar average
  6. Key Pivots             — multi-week price extremes (pin-to-pin levels)
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


# ── Support & Resistance ────────────────────────────────────────────────────────

def find_support_resistance(df: pd.DataFrame, window: int = 10) -> Dict[str, List[float]]:
    """
    Detect key support and resistance levels using local swing highs/lows.

    Args:
        df:     OHLCV DataFrame with indicator columns
        window: Number of bars to look left/right for a local extreme

    Returns:
        dict with 'support' and 'resistance' lists (price levels)
    """
    highs = df["High"].values
    lows  = df["Low"].values
    n     = len(df)

    supports    = []
    resistances = []

    for i in range(window, n - window):
        # Swing low (support): lower than all bars in window on both sides
        if lows[i] == min(lows[i - window:i + window + 1]):
            supports.append(round(float(lows[i]), 2))
        # Swing high (resistance): higher than all bars in window on both sides
        if highs[i] == max(highs[i - window:i + window + 1]):
            resistances.append(round(float(highs[i]), 2))

    # Keep the 3 most recent levels only (most relevant)
    return {
        "support":    sorted(set(supports[-6:]))[-3:],
        "resistance": sorted(set(resistances[-6:]))[:3],
    }


def price_near_level(price: float, levels: List[float], tolerance_pct: float = 0.02) -> Optional[float]:
    """Return the nearest support/resistance level if within tolerance, else None."""
    for level in levels:
        if abs(price - level) / level <= tolerance_pct:
            return level
    return None


# ── Trend Channel ───────────────────────────────────────────────────────────────

def detect_trend_channel(df: pd.DataFrame, lookback: int = 20) -> Dict[str, object]:
    """
    Determine current trend channel using linear regression over the last N bars.

    Returns:
        dict with 'direction' (RISING/FALLING/SIDEWAYS), 'slope', 'r_squared'
    """
    if len(df) < lookback:
        return {"direction": "SIDEWAYS", "slope": 0.0, "r_squared": 0.0}

    closes = df["Close"].tail(lookback).values
    x      = np.arange(len(closes))

    # Linear regression
    coeffs   = np.polyfit(x, closes, 1)
    slope    = float(coeffs[0])
    y_hat    = np.polyval(coeffs, x)
    ss_res   = float(np.sum((closes - y_hat) ** 2))
    ss_tot   = float(np.sum((closes - closes.mean()) ** 2))
    r_sq     = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

    # Normalise slope by price level
    price     = float(closes[-1])
    norm_slope = slope / price if price > 0 else 0.0

    if norm_slope > 0.002:
        direction = "RISING"
    elif norm_slope < -0.002:
        direction = "FALLING"
    else:
        direction = "SIDEWAYS"

    return {
        "direction": direction,
        "slope":     round(norm_slope, 6),
        "r_squared": round(r_sq, 3),
    }


# ── Candlestick Patterns ────────────────────────────────────────────────────────

def detect_candlestick_patterns(df: pd.DataFrame) -> List[str]:
    """
    Detect candlestick patterns on the last two bars.

    Returns:
        List of detected pattern names (e.g., ['HAMMER', 'BULLISH_ENGULFING'])
    """
    if len(df) < 2:
        return []

    patterns = []

    curr = df.iloc[-1]
    prev = df.iloc[-2]

    o, h, l, c = float(curr["Open"]), float(curr["High"]), float(curr["Low"]), float(curr["Close"])
    po, ph, pl, pc = float(prev["Open"]), float(prev["High"]), float(prev["Low"]), float(prev["Close"])

    body        = abs(c - o)
    upper_shadow = h - max(o, c)
    lower_shadow = min(o, c) - l
    total_range  = h - l

    if total_range == 0:
        return []

    # Hammer: small body near top, long lower shadow (≥2× body), bullish
    if (lower_shadow >= 2 * body and upper_shadow <= 0.3 * body
            and c > o and body / total_range < 0.4):
        patterns.append("HAMMER")

    # Shooting Star: small body near bottom, long upper shadow (≥2× body), bearish
    if (upper_shadow >= 2 * body and lower_shadow <= 0.3 * body
            and c < o and body / total_range < 0.4):
        patterns.append("SHOOTING_STAR")

    # Bullish Engulfing: current bullish bar engulfs previous bearish bar
    if (c > o and pc > po  # curr is bullish, prev was bearish
            and c > po and o < pc):
        patterns.append("BULLISH_ENGULFING")

    # Bearish Engulfing: current bearish bar engulfs previous bullish bar
    if (c < o and pc < po  # curr is bearish, prev was bullish
            and c < po and o > pc):
        patterns.append("BEARISH_ENGULFING")

    # Inside Bar: current bar fully inside previous bar's range
    if h < ph and l > pl:
        patterns.append("INSIDE_BAR")

    # Doji: tiny body relative to range
    if body / total_range < 0.1:
        patterns.append("DOJI")

    return patterns


# ── RSI Divergence ──────────────────────────────────────────────────────────────

def detect_rsi_divergence(df: pd.DataFrame, lookback: int = 14) -> Optional[str]:
    """
    Detect bullish or bearish RSI divergence.

    Bullish divergence:  price makes lower low, RSI makes higher low  → buy signal
    Bearish divergence:  price makes higher high, RSI makes lower high → sell signal

    Returns: 'BULLISH_DIVERGENCE', 'BEARISH_DIVERGENCE', or None
    """
    rsi_col = "RSI_14"
    if rsi_col not in df.columns or len(df) < lookback * 2:
        return None

    recent = df.tail(lookback * 2)
    mid    = len(recent) // 2

    price_first_half = recent["Close"].iloc[:mid]
    price_second_half = recent["Close"].iloc[mid:]
    rsi_first_half  = recent[rsi_col].iloc[:mid]
    rsi_second_half = recent[rsi_col].iloc[mid:]

    price_low_1  = float(price_first_half.min())
    price_low_2  = float(price_second_half.min())
    rsi_low_1    = float(rsi_first_half.min())
    rsi_low_2    = float(rsi_second_half.min())

    price_high_1 = float(price_first_half.max())
    price_high_2 = float(price_second_half.max())
    rsi_high_1   = float(rsi_first_half.max())
    rsi_high_2   = float(rsi_second_half.max())

    # Bullish: lower price low but higher RSI low
    if price_low_2 < price_low_1 and rsi_low_2 > rsi_low_1:
        return "BULLISH_DIVERGENCE"

    # Bearish: higher price high but lower RSI high
    if price_high_2 > price_high_1 and rsi_high_2 < rsi_high_1:
        return "BEARISH_DIVERGENCE"

    return None


# ── Volume Spikes ───────────────────────────────────────────────────────────────

def detect_volume_spike(df: pd.DataFrame, threshold: float = 2.0) -> bool:
    """
    Detect if today's volume is significantly above average.

    Args:
        threshold: Multiple of 20-bar average to qualify as a spike

    Returns:
        True if volume spike detected
    """
    if "Volume" not in df.columns or len(df) < 21:
        return False

    avg_vol  = float(df["Volume"].tail(21).iloc[:-1].mean())
    curr_vol = float(df["Volume"].iloc[-1])

    if avg_vol == 0:
        return False

    return (curr_vol / avg_vol) >= threshold


# ── Key Pivots ──────────────────────────────────────────────────────────────────

def detect_key_pivots(df: pd.DataFrame, weeks: int = 4) -> Dict[str, float]:
    """
    Detect pin-to-pin price extremes over the past N weeks.

    Returns:
        dict with 'pivot_high' and 'pivot_low'
    """
    bars = weeks * 5  # ~5 trading days per week
    recent = df.tail(bars)

    return {
        "pivot_high": round(float(recent["High"].max()), 2),
        "pivot_low":  round(float(recent["Low"].min()), 2),
        "period_bars": bars,
    }


# ── Full Pattern Snapshot ────────────────────────────────────────────────────────

def get_pattern_snapshot(df: pd.DataFrame) -> dict:
    """
    Run all pattern detectors and return a full snapshot for a symbol.

    Returns:
        dict with all detected patterns for the latest bar.
    """
    sr      = find_support_resistance(df)
    channel = detect_trend_channel(df)
    candles = detect_candlestick_patterns(df)
    diverg  = detect_rsi_divergence(df)
    vol_spk = detect_volume_spike(df)
    pivots  = detect_key_pivots(df)

    last_bar = df.iloc[-1]
    price       = float(last_bar["Close"])
    near_supp   = price_near_level(price, sr["support"])
    near_resist = price_near_level(price, sr["resistance"])

    # Trend Strength (ADX)
    adx = float(last_bar.get("ADX_14", 0))
    if adx > 25:
        trend_strength = "STRONG"
    elif adx < 20:
        trend_strength = "WEAK"
    else:
        trend_strength = "MODERATE"

    # Momentum (MACD)
    macd_hist = float(last_bar.get("MACD_Hist", 0))
    momentum  = "BULLISH" if macd_hist > 0 else "BEARISH"

    # Volatility (Bollinger Bands)
    bb_upper = float(last_bar.get("BB_Upper", 0))
    bb_lower = float(last_bar.get("BB_Lower", 0))
    is_overbought = price >= bb_upper
    is_oversold   = price <= bb_lower

    return {
        "support_levels":     sr["support"],
        "resistance_levels":  sr["resistance"],
        "near_support":       near_supp,
        "near_resistance":    near_resist,
        "trend_channel":      channel["direction"],
        "trend_strength":     trend_strength,
        "adx_value":          adx,
        "momentum":           momentum,
        "macd_hist":          macd_hist,
        "bb_overbought":      is_overbought,
        "bb_oversold":        is_oversold,
        "channel_slope":      channel["slope"],
        "channel_r2":         channel["r_squared"],
        "candlestick":        candles,
        "rsi_divergence":     diverg,
        "volume_spike":       vol_spk,
        "pivot_high":         pivots["pivot_high"],
        "pivot_low":          pivots["pivot_low"],
    }
