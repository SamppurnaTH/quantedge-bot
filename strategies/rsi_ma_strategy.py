"""
Strategy Engine — Regime-Aware RSI + Pullback Depth + Volume + Scoring

Signal scoring (0–4 points):
  +1  price > SMA_200              (trend structure)
  +1  RSI < regime_rsi_threshold   (regime-aware oversold level)
  +1  volume > 20-day average      (volume confirmation)
  +1  pullback_depth >= 3%         (meaningful dip, not a shallow entry)

RSI thresholds by regime:
  STRONG_TREND_UP   → RSI < 35   (moderate pullback in strong trend)
  WEAK_TREND_UP     → RSI < 30   (deeper pullback required)
  SIDEWAYS          → RSI < 25   (very oversold only)
  VOLATILE          → RSI < 25   (very oversold only)
  TRENDING_DOWN     → no longs

BUY threshold by regime:
  STRONG_TREND_UP   → score ≥ 2
  all others        → score = 3 (or blocked)

SELL → RSI > 70 (overbought exit)
HOLD → everything else
"""

from enum import Enum
import numpy as np
import pandas as pd
import logging

from config.settings import STRATEGY_CONFIG, INDICATOR_CONFIG
from indicators.regime import Regime, regime_allows_trade

logger = logging.getLogger(__name__)


class Signal(str, Enum):
    BUY  = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"

    def __str__(self):
        return self.value


# ── RSI threshold lookup ──────────────────────────────────────────────────────

def get_rsi_threshold(regime: Regime) -> int:
    """
    Return the RSI buy threshold for the given regime.
    Tighter thresholds in weaker regimes = only deep pullbacks qualify.
    """
    thresholds = STRATEGY_CONFIG["rsi_thresholds"]
    val = thresholds.get(str(regime))
    if val is None:
        return 0   # no longs allowed — threshold impossible to meet
    return val


# ── Pullback depth ────────────────────────────────────────────────────────────

def compute_pullback_depth(df: pd.DataFrame) -> pd.Series:
    """
    Pullback depth = (rolling_high - close) / rolling_high

    Measures how far price has pulled back from its recent peak.
    A value >= 0.03 means price is at least 3% below its recent high.

    Args:
        df: DataFrame with Close column

    Returns:
        Series of pullback depth values (0.0 to 1.0)
    """
    window = INDICATOR_CONFIG["pullback_window"]
    rolling_high = df["Close"].rolling(window=window, min_periods=1).max()
    pullback = (rolling_high - df["Close"]) / rolling_high
    pullback.name = "Pullback"
    return pullback.clip(lower=0)


# ── Signal scoring ────────────────────────────────────────────────────────────

def compute_signal_score(df: pd.DataFrame, regime: Regime = Regime.STRONG_TREND_UP) -> pd.Series:
    """
    Score each bar 0–4 based on how many conditions are met.

    Scoring:
      +1  Close > SMA_200
      +1  RSI < regime_rsi_threshold   (regime-aware)
      +1  Volume > Volume_MA_20
      +1  Pullback >= min_pullback_pct  (meaningful dip)
    """
    rsi_col = f"RSI_{INDICATOR_CONFIG['rsi_period']}"
    ma_col  = f"SMA_{INDICATOR_CONFIG['ma_slow']}"
    vol_col = f"Volume_MA_{INDICATOR_CONFIG['volume_ma_period']}"

    rsi_threshold  = get_rsi_threshold(regime)
    min_pullback   = STRATEGY_CONFIG["min_pullback_pct"]

    # Compute pullback if not already present
    if "Pullback" not in df.columns:
        df = df.copy()
        df["Pullback"] = compute_pullback_depth(df)

    score = (
        (df["Close"] > df[ma_col]).astype(int)
        + (df[rsi_col] < rsi_threshold).astype(int)
        + (df["Volume"] > df[vol_col] * STRATEGY_CONFIG["volume_multiplier"]).astype(int)
        + (df["Pullback"] >= min_pullback).astype(int)
    )
    score.name = "Score"
    return score


# ── Signal generation ─────────────────────────────────────────────────────────

def generate_signals(
    df: pd.DataFrame,
    market_up: bool = True,
    regime: Regime = Regime.STRONG_TREND_UP,
) -> pd.DataFrame:
    """
    Apply the full strategy to a DataFrame that already has indicators.

    Args:
        df:        DataFrame with Close, RSI_14, SMA_200, Volume, Volume_MA_20
        market_up: True if the market index is in an uptrend
        regime:    Current effective regime — controls RSI threshold + min score

    Returns:
        DataFrame with added 'Pullback', 'Score', and 'Signal' columns
    """
    rsi_col = f"RSI_{INDICATOR_CONFIG['rsi_period']}"
    ma_col  = f"SMA_{INDICATOR_CONFIG['ma_slow']}"

    required = ["Close", rsi_col, ma_col, "Volume"]
    missing  = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns for strategy: {missing}")

    df = df.copy()
    df["Pullback"] = compute_pullback_depth(df)
    df["Score"]    = compute_signal_score(df, regime=regime)

    min_score  = regime.min_score_to_buy
    rsi_sell   = STRATEGY_CONFIG["rsi_sell_threshold"]
    buy_allowed = regime.allows_buy and market_up

    conditions = [
        (df["Score"] >= min_score) & buy_allowed,
        df[rsi_col] > rsi_sell,
    ]
    choices = [Signal.BUY, Signal.SELL]

    df["Signal"] = np.select(conditions, choices, default=Signal.HOLD)
    return df


# ── Latest signal extraction ──────────────────────────────────────────────────

def get_latest_signal(df: pd.DataFrame) -> dict:
    """Return the most recent bar's signal with all supporting data."""
    rsi_col = f"RSI_{INDICATOR_CONFIG['rsi_period']}"
    ma_col  = f"SMA_{INDICATOR_CONFIG['ma_slow']}"
    atr_col = f"ATR_{INDICATOR_CONFIG['atr_period']}"
    vol_col = f"Volume_MA_{INDICATOR_CONFIG['volume_ma_period']}"

    drop_cols = [c for c in [rsi_col, ma_col] if c in df.columns]
    last = df.dropna(subset=drop_cols).iloc[-1]

    trend         = "UP"  if last["Close"] > last[ma_col] else "DOWN"
    vol_confirmed = bool(last["Volume"] > last.get(vol_col, 0)) if vol_col in df.columns else False
    atr_val       = round(float(last[atr_col]), 2) if atr_col in df.columns else None
    pullback_val  = round(float(last["Pullback"]) * 100, 2) if "Pullback" in df.columns else 0.0

    return {
        "date":          str(last.name.date()) if hasattr(last.name, "date") else str(last.name),
        "close":         round(float(last["Close"]), 2),
        "rsi":           round(float(last[rsi_col]), 2),
        "ma_200":        round(float(last[ma_col]), 2),
        "atr":           atr_val,
        "volume":        int(last["Volume"]),
        "volume_ma":     round(float(last[vol_col]), 0) if vol_col in df.columns else None,
        "vol_confirmed": vol_confirmed,
        "pullback_pct":  pullback_val,
        "score":         int(last.get("Score", 0)),
        "signal":        last["Signal"],
        "trend":         trend,
    }


# ── Signal ranking ────────────────────────────────────────────────────────────

def rank_signals(results: list) -> list:
    """
    Rank BUY signals by composite quality score.

    Ranking formula:
      rank = (score × 10) + rr_ratio + (pullback_pct × 2)

    Higher score → better
    Higher R:R   → better
    Deeper pullback → better entry (more room to recover)

    Args:
        results: List of signal dicts from scan_watchlist

    Returns:
        Same list with 'rank_score' added, BUYs sorted best-first
    """
    for r in results:
        if r.get("signal") != "BUY" or "error" in r:
            r["rank_score"] = 0
            continue

        rr       = r.get("risk_report", {}).get("risk_reward", 0) or 0
        score    = r.get("score", 0)
        pullback = r.get("pullback_pct", 0) or 0

        r["rank_score"] = round((score * 10) + rr + (pullback * 2), 2)

    # Sort: BUY (by rank_score desc) → SELL → HOLD
    order = {"BUY": 0, "SELL": 1, "HOLD": 2}
    results.sort(
        key=lambda r: (order.get(r.get("signal", "HOLD"), 3), -r.get("rank_score", 0))
    )
    return results


# ── Structured logging ────────────────────────────────────────────────────────

def log_signal_decision(
    symbol: str,
    score: int,
    regime: Regime,
    signal: str,
    rsi: float,
    close: float,
    ma200: float,
    pullback_pct: float = 0.0,
    rsi_threshold: int = 40,
) -> None:
    """Structured log entry for every signal decision — full audit trail."""
    allowed, reason = regime_allows_trade(regime, score)
    decision = signal if allowed else "HOLD (blocked)"

    logger.info(
        "SIGNAL | %-18s | %-18s | Score:%d/4 | RSI:%5.1f(<%d) | "
        "Pullback:%4.1f%% | Close:%8.2f | MA200:%8.2f | %s | %s",
        symbol, regime, score, rsi, rsi_threshold,
        pullback_pct, close, ma200, decision, reason,
    )
