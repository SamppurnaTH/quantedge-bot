"""Benchmark how fast a single symbol seeds, and where the time goes."""
import time
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import numpy as np
from data.fetcher import load_stock_data
from indicators.engine import compute_all_indicators
from indicators.regime import Regime
from strategies.rsi_ma_strategy import get_rsi_threshold, compute_pullback_depth
from analytics.pattern_detector import get_pattern_snapshot
from config.settings import REGIME_CONFIG, INDICATOR_CONFIG, STRATEGY_CONFIG

OUTCOME_LOOKAHEAD = 20
MIN_ROWS = 250
SYMBOL = "TCS.NS"

t0 = time.time()
df = load_stock_data(SYMBOL)
df = compute_all_indicators(df)
print(f"Load+indicators: {time.time()-t0:.2f}s, rows={len(df)}")

# Vectorized regime
ma_fast_col = f"SMA_{INDICATOR_CONFIG['ma_fast']}"
ma_slow_col = f"SMA_{INDICATOR_CONFIG['ma_slow']}"
atr_col     = f"ATR_{INDICATOR_CONFIG['atr_period']}"
rsi_col     = f"RSI_{INDICATOR_CONFIG['rsi_period']}"
vol_col     = f"Volume_MA_{INDICATOR_CONFIG['volume_ma_period']}"

atr_series = df[atr_col]
atr_ratio  = atr_series / atr_series.rolling(50).mean()
slope_norm = df[ma_fast_col].diff().rolling(REGIME_CONFIG["slope_window"]).mean() / df[ma_fast_col]

p_above = df["Close"] > df[ma_slow_col]
p_below = (df["Close"] < df[ma_fast_col]) & (df["Close"] < df[ma_slow_col])

s_up, w_up = REGIME_CONFIG["slope_strong_up"], REGIME_CONFIG["slope_weak_up"]
s_dn, w_dn = REGIME_CONFIG["slope_strong_down"], REGIME_CONFIG["slope_weak_down"]
atr_th      = REGIME_CONFIG["atr_ratio_threshold"]

conds   = [atr_ratio > atr_th, p_above & (slope_norm >= s_up), p_above & (slope_norm >= w_up), p_below & (slope_norm <= s_dn), p_below & (slope_norm <= w_dn)]
choices = [Regime.VOLATILE, Regime.STRONG_TREND_UP, Regime.WEAK_TREND_UP, Regime.STRONG_TREND_DOWN, Regime.WEAK_TREND_DOWN]
df["regime"] = np.select(conds, choices, default=Regime.SIDEWAYS)

df["Pullback"]    = compute_pullback_depth(df)
rsi_thresholds    = df["regime"].apply(lambda r: get_rsi_threshold(Regime(r)))
df["Score"]       = (
    (df["Close"] > df[ma_slow_col]).astype(int)
    + (df[rsi_col] < rsi_thresholds).astype(int)
    + (df["Volume"] > df[vol_col] * STRATEGY_CONFIG["volume_multiplier"]).astype(int)
    + (df["Pullback"] >= STRATEGY_CONFIG["min_pullback_pct"]).astype(int)
)
min_scores  = df["regime"].apply(lambda r: Regime(r).min_score_to_buy)
buy_allowed = df["regime"].apply(lambda r: Regime(r).allows_buy)
df["Signal"] = np.select(
    [(df["Score"] >= min_scores) & buy_allowed, df[rsi_col] > STRATEGY_CONFIG["rsi_sell_threshold"]],
    ["BUY", "SELL"], default="HOLD"
)

buy_bars = [i for i in range(MIN_ROWS, len(df) - OUTCOME_LOOKAHEAD) if df.iloc[i]["Signal"] == "BUY"]
print(f"Vectorize done: {time.time()-t0:.2f}s  |  BUY signals found: {len(buy_bars)}")

# Time one pattern snapshot
t1 = time.time()
_ = get_pattern_snapshot(df.iloc[:buy_bars[0] + 1])
one_snap = time.time() - t1
print(f"One get_pattern_snapshot: {one_snap*1000:.1f}ms")
print(f"Estimated for {len(buy_bars)} signals (1 symbol): {one_snap * len(buy_bars):.1f}s")
print(f"Estimated for 48 symbols: {one_snap * len(buy_bars) * 48:.0f}s  ({one_snap * len(buy_bars) * 48 / 60:.1f} min)")
