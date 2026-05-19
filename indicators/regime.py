"""
Market Regime Detection — Trend Strength Tiering

Six regimes (expanded from four):
  STRONG_TREND_UP   — slope > 0.0015, price above both MAs  → score ≥ 2, full size
  WEAK_TREND_UP     — slope > 0.0005, price above both MAs  → score = 3, full size
  SIDEWAYS          — slope flat or mixed price structure    → score = 3, half size
  VOLATILE          — ATR ratio > 1.4 (overrides trend)     → score = 3, half size, wider SL
  WEAK_TREND_DOWN   — slope < -0.0005, price below both MAs → no longs
  STRONG_TREND_DOWN — slope < -0.0015, price below both MAs → no longs

Detection priority (strict):
  1. ATR Ratio  → VOLATILE if elevated (overrides everything)
  2. MA Slope   → slope = ma50.diff().rolling(5).mean() / ma50  (normalised)
  3. Price      → price vs SMA(50) and SMA(200)

Regime → Behavior mapping:
  Regime             Min Score  Size   SL Mult
  STRONG_TREND_UP       2/3    ×1.0    ×2.0
  WEAK_TREND_UP         3/3    ×1.0    ×2.0
  SIDEWAYS              3/3    ×0.5    ×2.0
  VOLATILE              3/3    ×0.5    ×2.5
  WEAK_TREND_DOWN       —      ×0.0     —
  STRONG_TREND_DOWN     —      ×0.0     —
"""

from dataclasses import dataclass
from enum import Enum
import pandas as pd
import logging

from config.settings import INDICATOR_CONFIG, REGIME_CONFIG

logger = logging.getLogger(__name__)


# ── Regime enum ───────────────────────────────────────────────────────────────

class Regime(str, Enum):
    STRONG_TREND_UP   = "STRONG_TREND_UP"
    WEAK_TREND_UP     = "WEAK_TREND_UP"
    SIDEWAYS          = "SIDEWAYS"
    VOLATILE          = "VOLATILE"
    WEAK_TREND_DOWN   = "WEAK_TREND_DOWN"
    STRONG_TREND_DOWN = "STRONG_TREND_DOWN"

    def __str__(self):
        return self.value

    @property
    def allows_buy(self) -> bool:
        return self in (
            Regime.STRONG_TREND_UP,
            Regime.WEAK_TREND_UP,
            Regime.SIDEWAYS,
            Regime.VOLATILE,
        )

    @property
    def min_score_to_buy(self) -> int:
        """
        Minimum signal score (0–3) required to open a long position.
        Higher regime uncertainty → stricter score requirement.
        """
        return {
            Regime.STRONG_TREND_UP:   2,   # strong trend: score ≥ 2 is enough
            Regime.WEAK_TREND_UP:     3,   # weak trend: only perfect score
            Regime.SIDEWAYS:          3,   # range-bound: only perfect score
            Regime.VOLATILE:          3,   # volatile: only perfect score
            Regime.WEAK_TREND_DOWN:   99,  # no longs
            Regime.STRONG_TREND_DOWN: 99,  # no longs
        }[self]

    @property
    def position_size_multiplier(self) -> float:
        return {
            Regime.STRONG_TREND_UP:   1.0,
            Regime.WEAK_TREND_UP:     0.5,   # half size — weak conviction
            Regime.SIDEWAYS:          0.5,
            Regime.VOLATILE:          0.5,
            Regime.WEAK_TREND_DOWN:   0.0,
            Regime.STRONG_TREND_DOWN: 0.0,
        }[self]

    @property
    def atr_sl_multiplier(self) -> float:
        """Stop-loss ATR multiplier — wider in volatile regimes."""
        return {
            Regime.STRONG_TREND_UP:   2.0,
            Regime.WEAK_TREND_UP:     2.0,
            Regime.SIDEWAYS:          2.0,
            Regime.VOLATILE:          2.5,   # wider: avoid noise shakeouts
            Regime.WEAK_TREND_DOWN:   2.0,
            Regime.STRONG_TREND_DOWN: 2.0,
        }[self]

    @property
    def emoji(self) -> str:
        return {
            Regime.STRONG_TREND_UP:   "🚀",
            Regime.WEAK_TREND_UP:     "📈",
            Regime.SIDEWAYS:          "↔️ ",
            Regime.VOLATILE:          "⚡",
            Regime.WEAK_TREND_DOWN:   "📉",
            Regime.STRONG_TREND_DOWN: "🔻",
        }[self]

    @property
    def description(self) -> str:
        return {
            Regime.STRONG_TREND_UP:   "Strong uptrend — score ≥ 2, full size",
            Regime.WEAK_TREND_UP:     "Weak uptrend — score = 3 only, full size",
            Regime.SIDEWAYS:          "Range-bound — score = 3 only, half size",
            Regime.VOLATILE:          "High volatility — score = 3, wider SL, half size",
            Regime.WEAK_TREND_DOWN:   "Weak downtrend — no new longs",
            Regime.STRONG_TREND_DOWN: "Strong downtrend — no new longs, protect capital",
        }[self]

    @property
    def trend_strength(self) -> str:
        """Human-readable trend strength label."""
        return {
            Regime.STRONG_TREND_UP:   "STRONG",
            Regime.WEAK_TREND_UP:     "WEAK",
            Regime.SIDEWAYS:          "FLAT",
            Regime.VOLATILE:          "VOLATILE",
            Regime.WEAK_TREND_DOWN:   "WEAK",
            Regime.STRONG_TREND_DOWN: "STRONG",
        }[self]


# ── Detection result ──────────────────────────────────────────────────────────

@dataclass
class RegimeResult:
    regime:    Regime
    slope:     float      # normalised slope of SMA_50
    atr_ratio: float      # current ATR / 50-bar ATR average
    price:     float
    ma50:      float
    ma200:     float

    def __str__(self):
        return (
            f"{self.regime.emoji} {self.regime} [{self.regime.trend_strength}] | "
            f"slope={self.slope:+.5f}  atr_ratio={self.atr_ratio:.2f}  "
            f"price={self.price:.2f}  ma50={self.ma50:.2f}  ma200={self.ma200:.2f}"
        )


# ── Core detection ────────────────────────────────────────────────────────────

def detect_regime(df: pd.DataFrame) -> Regime:
    """Detect regime, return Regime enum only."""
    return detect_regime_detailed(df).regime


def detect_regime_detailed(df: pd.DataFrame) -> RegimeResult:
    """
    Full regime detection with all intermediate values.

    Required columns: Close, SMA_50, SMA_200, ATR_14
    (produced by indicators.engine.compute_all_indicators)
    """
    ma_fast_col = f"SMA_{INDICATOR_CONFIG['ma_fast']}"
    ma_slow_col = f"SMA_{INDICATOR_CONFIG['ma_slow']}"
    atr_col     = f"ATR_{INDICATOR_CONFIG['atr_period']}"

    required = ["Close", ma_fast_col, ma_slow_col, atr_col]
    missing  = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Regime detection missing columns: {missing}")

    df_clean = df.dropna(subset=required)
    if len(df_clean) < 55:
        logger.warning("Insufficient data (%d rows) — defaulting SIDEWAYS", len(df_clean))
        last = df_clean.iloc[-1] if len(df_clean) > 0 else df.iloc[-1]
        return RegimeResult(
            regime=Regime.SIDEWAYS, slope=0.0, atr_ratio=1.0,
            price=float(last["Close"]),
            ma50=float(last.get(ma_fast_col, 0)),
            ma200=float(last.get(ma_slow_col, 0)),
        )

    # ── Signal 1: ATR Ratio (override check) ─────────────────────────────────
    atr_series  = df_clean[atr_col]
    atr_current = float(atr_series.iloc[-1])
    atr_avg_50  = float(atr_series.tail(50).mean())
    atr_ratio   = atr_current / atr_avg_50 if atr_avg_50 > 0 else 1.0

    # ── Signal 2: MA Slope (normalised) ──────────────────────────────────────
    ma50_series  = df_clean[ma_fast_col]
    slope_raw    = ma50_series.diff().rolling(REGIME_CONFIG["slope_window"]).mean()
    slope_abs    = float(slope_raw.iloc[-1])
    ma50_last    = float(ma50_series.iloc[-1])
    slope_norm   = slope_abs / ma50_last if ma50_last != 0 else 0.0

    # ── Signal 3: Price Structure ─────────────────────────────────────────────
    # Key insight: RSI pullback entries happen when price dips below MA50
    # (that's what creates the oversold reading). Requiring price > MA50 AND
    # price > MA200 blocks every valid pullback entry.
    # Fix: use price > MA200 as the long-term uptrend condition.
    #      MA50 slope already captures the medium-term direction.
    last      = df_clean.iloc[-1]
    price     = float(last["Close"])
    ma50_val  = float(last[ma_fast_col])
    ma200_val = float(last[ma_slow_col])

    # Uptrend: price above long-term MA (MA200) — allows pullbacks below MA50
    price_above_ma200 = price > ma200_val
    # Downtrend: price below both MAs (confirmed breakdown)
    price_below_both  = price < ma50_val and price < ma200_val

    # ── Regime decision ───────────────────────────────────────────────────────
    s_up   = REGIME_CONFIG["slope_strong_up"]
    w_up   = REGIME_CONFIG["slope_weak_up"]
    s_dn   = REGIME_CONFIG["slope_strong_down"]
    w_dn   = REGIME_CONFIG["slope_weak_down"]
    atr_th = REGIME_CONFIG["atr_ratio_threshold"]

    if atr_ratio > atr_th:
        regime = Regime.VOLATILE
    elif price_above_ma200 and slope_norm >= s_up:
        regime = Regime.STRONG_TREND_UP
    elif price_above_ma200 and slope_norm >= w_up:
        regime = Regime.WEAK_TREND_UP
    elif price_below_both and slope_norm <= s_dn:
        regime = Regime.STRONG_TREND_DOWN
    elif price_below_both and slope_norm <= w_dn:
        regime = Regime.WEAK_TREND_DOWN
    else:
        regime = Regime.SIDEWAYS

    result = RegimeResult(
        regime=regime, slope=slope_norm, atr_ratio=atr_ratio,
        price=price, ma50=ma50_val, ma200=ma200_val,
    )
    logger.info("Regime: %s", result)
    return result


def detect_regime_for_index(df: pd.DataFrame) -> Regime:
    return detect_regime(df)


# ── Helpers ───────────────────────────────────────────────────────────────────

def regime_summary(regime: Regime) -> str:
    return f"{regime.emoji} {regime}  ({regime.description})"


def regime_allows_trade(regime: Regime, score: int) -> tuple:
    min_score = regime.min_score_to_buy
    if score >= min_score:
        return True, f"Score {score}/4 meets {regime} threshold ({min_score})"
    return False, f"Score {score}/4 below {regime} threshold ({min_score})"


def _more_conservative(r1: Regime, r2: Regime) -> Regime:
    """Return the regime with the smaller position_size_multiplier."""
    if r1.position_size_multiplier <= r2.position_size_multiplier:
        return r1
    return r2


def apply_regime_hysteresis(new_regime: Regime) -> Regime:
    """
    Prevents rapid regime flipping by requiring 2 consecutive days of confirmation.
    Saves and reads history from state/regime_history.json.
    """
    import os
    import json
    
    history_file = os.path.join("state", "regime_history.json")
    os.makedirs(os.path.dirname(history_file), exist_ok=True)
    
    history = {}
    if os.path.exists(history_file):
        try:
            with open(history_file, "r") as f:
                history = json.load(f)
        except Exception:
            history = {}

    prev_regime_str = history.get("active_regime")
    pending_regime_str = history.get("pending_regime")
    pending_days = history.get("pending_days", 0)

    if not prev_regime_str:
        # First run: initialize
        history = {
            "active_regime": str(new_regime),
            "pending_regime": str(new_regime),
            "pending_days": 0
        }
        with open(history_file, "w") as f:
            json.dump(history, f, indent=2)
        return new_regime

    prev_regime = Regime(prev_regime_str)

    if str(new_regime) == prev_regime_str:
        # Regime matches active, reset pending
        history["pending_regime"] = str(new_regime)
        history["pending_days"] = 0
        with open(history_file, "w") as f:
            json.dump(history, f, indent=2)
        return new_regime

    # Different regime detected
    if pending_regime_str == str(new_regime):
        # We've seen this before, increment days
        pending_days += 1
    else:
        # New candidate regime
        pending_regime_str = str(new_regime)
        pending_days = 1

    if pending_days >= 2:
        # Confirmed transition!
        active_regime = new_regime
        pending_days = 0
        logger.info("Regime Transition Confirmed: %s -> %s", prev_regime, active_regime)
    else:
        # Stick to previous active regime for stability
        active_regime = prev_regime
        logger.info("Regime Transition Pending: candidate=%s, active stays=%s (pending %d/2 days)", new_regime, prev_regime, pending_days)

    history["active_regime"] = str(active_regime)
    history["pending_regime"] = pending_regime_str
    history["pending_days"] = pending_days
    
    with open(history_file, "w") as f:
        json.dump(history, f, indent=2)

    return active_regime
