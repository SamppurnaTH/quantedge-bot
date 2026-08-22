"""
Learning Engine
Tracks detected patterns against their outcomes and builds an institutional knowledge base.

Five knowledge states per pattern condition:
  PROVEN      — ≥100 trades, ≥60% win rate, Profit Factor ≥1.2, expectancy > 0 → elite trusted edge
  VALIDATED   — 50–99 trades, ≥55% win rate, Profit Factor ≥1.1, expectancy > 0 → robust edge
  LEARNING    — 20–49 trades, any win rate → gathering evidence
  WATCHING    — <20 trades → monitoring
  UNRELIABLE  — ≥50 trades, fails to meet validated/proven performance metrics → avoid / skip

The journal is seeded from 10-year backtest signal history on first run.
"""

import os
import json
import logging
from datetime import datetime
from typing import Dict, List, Optional

import pandas as pd

from analytics.pattern_detector import get_pattern_snapshot
from data.fetcher import fetch_stock_data, load_stock_data
from indicators.engine import compute_all_indicators
from indicators.regime import detect_regime_detailed
from strategies.rsi_ma_strategy import generate_signals, get_latest_signal
from config.settings import DATA_CONFIG, RISK_CONFIG, STRATEGY_CONFIG

logger = logging.getLogger(__name__)

LEARNING_JOURNAL_FILE = os.path.join("state", "learning_journal.json")

OUTCOME_LOOKAHEAD = 20   # bars to look ahead to determine outcome


# ── Journal I/O ────────────────────────────────────────────────────────────────

def load_journal() -> dict:
    """Load the learning journal from disk."""
    if not os.path.exists(LEARNING_JOURNAL_FILE):
        return {"patterns": {}, "metadata": {"last_updated": None, "total_observations": 0}}
    with open(LEARNING_JOURNAL_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


def save_journal(journal: dict) -> None:
    """Save the learning journal to disk."""
    os.makedirs("state", exist_ok=True)
    journal["metadata"]["last_updated"] = datetime.now().strftime("%Y-%m-%d %H:%M")
    with open(LEARNING_JOURNAL_FILE, "w", encoding="utf-8") as f:
        json.dump(journal, f, indent=2)
    logger.info("Learning journal saved (%d pattern types)", len(journal["patterns"]))


# ── Pattern Key Builder ────────────────────────────────────────────────────────

def build_condition_key(
    regime: str,
    rsi_bucket: str,
    score: int,
    channel: str,
    candles: List[str],
    near_support: bool,
    volume_spike: bool,
) -> str:
    """
    Build a simplified canonical 7-factor key to prevent overfitting.
    Example: 'STRONG_TREND_UP|RSI<30|S3|RISING|HAMMER|SUPP|VOLSPK'
    """
    parts = [
        regime,
        rsi_bucket,
        f"S{score}",
        channel,
        "+".join(candles) if candles else "NOCNDLE",
        "SUPP" if near_support else "NOSUPP",
        "VOLSPK" if volume_spike else "NOSPK",
    ]
    return "|".join(parts)


def rsi_to_bucket(rsi: float) -> str:
    """Bucket RSI into readable ranges."""
    if rsi < 20:   return "RSI<20"
    if rsi < 25:   return "RSI<25"
    if rsi < 30:   return "RSI<30"
    if rsi < 35:   return "RSI<35"
    if rsi < 40:   return "RSI<40"
    if rsi < 50:   return "RSI<50"
    if rsi < 60:   return "RSI<60"
    if rsi < 70:   return "RSI<70"
    return "RSI>=70"


# ── Observation Recording ──────────────────────────────────────────────────────

# ── Behavioral State Classifier ───────────────────────────────────────────────

def classify_behavioral_state(key: str) -> str:
    """
    Compress a technical 7-factor pattern key into one of 6 behavioral/institutional states
    to prevent overfitting and provide operator-focused intelligence.
    """
    parts = key.split("|")
    if len(parts) < 7:
        return "UNKNOWN_NOISE"
    
    regime, rsi_bkt, score_str, channel, candles_str, supp_str, vol_str = parts[:7]
    score = int(score_str[1]) if (len(score_str) > 1 and score_str[1].isdigit()) else 0
    near_support = supp_str == "SUPP"
    volume_spike = vol_str == "VOLSPK"
    has_bull_candle = candles_str not in ("NOCNDLE", "SHOOTING_STAR", "BEARISH_ENGULFING")

    # 1. Panic Exhaustion (Oversold capitulation bounce)
    if rsi_bkt in ("RSI<20", "RSI<25", "RSI<30") and (near_support or has_bull_candle):
        return "PANIC_EXHAUSTION"
        
    # 2. Forced Momentum (Trend chasing)
    if "UP" in regime and channel == "RISING" and score >= 2 and rsi_bkt not in ("RSI<20", "RSI<25", "RSI<30"):
        return "FORCED_MOMENTUM"
        
    # 3. Volatility Compression (Low volatility squeeze buildup)
    if regime == "SIDEWAYS" and channel == "SIDEWAYS" and not volume_spike:
        return "VOLATILITY_COMPRESSION"
        
    # 4. Rotational Strength (Outperformance in weak market)
    if ("DOWN" in regime or regime == "SIDEWAYS") and score >= 2 and channel == "RISING":
        return "ROTATIONAL_STRENGTH"
        
    # 5. Failed Breakout (Trapped buyers)
    if rsi_bkt == "RSI>=70" or (score < 2 and channel == "RISING" and not volume_spike):
        return "FAILED_BREAKOUT"
        
    # 6. Liquidity Vacuum (Low volume, high risk environment)
    if not volume_spike and ("DOWN" in regime or regime == "VOLATILE"):
        return "LIQUIDITY_VACUUM"
        
    # Fallback
    return "UNKNOWN_NOISE"


def classify_archetype(key: str) -> str:
    """
    Map a detailed pattern key into one of 6 canonical archetypes used by the cockpit.

    Returns one of:
      - MEAN_REVERSION_PULLBACK
      - TREND_BREAKOUT_CONTINUATION
      - RANGE_BOUND_SUPPORT
      - VOLATILITY_BREAKOUT
      - BEAR_REGIME_EXHAUSTION
      - UNKNOWN_NOISE
    """
    parts = key.split("|")
    if len(parts) < 7:
        return "UNKNOWN_NOISE"

    regime, rsi_bkt, score_str, channel, candles_str, supp_str, vol_str = parts[:7]
    score = int(score_str[1]) if (len(score_str) > 1 and score_str[1].isdigit()) else 0
    near_support = supp_str == "SUPP"
    volume_spike = vol_str == "VOLSPK"

    # Mean reversion pullbacks: setups near support, oversold RSI and in up-trend regimes
    if near_support and (rsi_bkt in ("RSI<20", "RSI<25", "RSI<30", "RSI<35")) and ("UP" in regime or channel == "RISING"):
        return "MEAN_REVERSION_PULLBACK"

    # Trend breakout / continuation: strong up regimes, rising channel, decent score
    if ("UP" in regime or channel == "RISING") and score >= 2 and not near_support:
        return "TREND_BREAKOUT_CONTINUATION"

    # Range-bound support: sideways regime, near support or low channel slope
    if regime == "SIDEWAYS" and (near_support or channel == "SIDEWAYS"):
        return "RANGE_BOUND_SUPPORT"

    # Volatility breakout: high volume spikes or volatile regime with large moves
    if volume_spike or "VOLATILE" in regime:
        return "VOLATILITY_BREAKOUT"

    # Bear-regime exhaustion: in down regimes but oversold and showing support/bounce
    if ("DOWN" in regime) and (rsi_bkt in ("RSI<20", "RSI<25", "RSI<30")) and near_support:
        return "BEAR_REGIME_EXHAUSTION"

    return "UNKNOWN_NOISE"


def get_blended_expectancy(pattern_data: dict, archetype_name: str, regime_name: str, journal: dict) -> float:
    """
    Compute Bayesian smoothed expectancy for a pattern using the regime-aware archetype prior.
    Formula: Blended = (Pattern_Trades * Pattern_Expectancy + C * Prior_Expectancy) / (Pattern_Trades + C)
    Where C is the smoothing constant (e.g. 30 trades).
    """
    pattern_trades = pattern_data.get("trades", 0.0)
    pattern_exp = pattern_data.get("expectancy", 0.0)
    
    prior_exp = 0.0
    c_constant = 30.0
    
    archetypes = journal.get("canonical_archetypes", {})
    if archetype_name not in archetypes:
        archetypes = journal.get("archetypes", {})
    if archetype_name in archetypes:
        arch_data = archetypes[archetype_name]
        regime_data = arch_data.get("regimes", {}).get(regime_name, {})
        if regime_data.get("trades", 0.0) >= 20:
            prior_exp = regime_data.get("expectancy", 0.0)
        else:
            prior_exp = arch_data.get("expectancy", 0.0)
            
    blended = (pattern_trades * pattern_exp + c_constant * prior_exp) / (pattern_trades + c_constant)
    return round(blended, 3)


# ── Observation Recording & Decayed Update ─────────────────────────────────────

def _update_stats(p: dict, won: bool, pnl: float, trade_date: str, hold_time: int) -> None:
    """Helper to update a stats dictionary with exponential time decay."""
    import math
    if "history" not in p:
        p["history"] = []
    p["history"].append({
        "date": trade_date,
        "won": won,
        "pnl": pnl,
        "hold_time": hold_time
    })
    if len(p["history"]) > 200:
        p["history"] = p["history"][-200:]
        
    decay_lambda = 0.0019  # ~365 days half-life
    current_dt = datetime.now()
    
    decayed_wins = 0.0
    decayed_losses = 0.0
    decayed_gp = 0.0
    decayed_gl = 0.0
    decayed_trades = 0.0
    decayed_hold_time = 0.0
    
    for t_item in p["history"]:
        try:
            t_dt = datetime.strptime(t_item["date"], "%Y-%m-%d")
        except ValueError:
            t_dt = current_dt
        
        age_days = max((current_dt - t_dt).days, 0)
        weight = math.exp(-decay_lambda * age_days)
        
        decayed_trades += weight
        decayed_hold_time += t_item.get("hold_time", 0) * weight
        if t_item["won"]:
            decayed_wins += weight
            if t_item["pnl"] > 0:
                decayed_gp += t_item["pnl"] * weight
        else:
            decayed_losses += weight
            decayed_gl += abs(t_item["pnl"]) * weight

    p["trades"] = round(decayed_trades, 2)
    p["wins"] = round(decayed_wins, 2)
    p["losses"] = round(decayed_losses, 2)
    p["win_rate"] = round(decayed_wins / decayed_trades, 3) if decayed_trades > 0 else 0.0
    p["gross_profit"] = round(decayed_gp, 2)
    p["gross_loss"] = round(decayed_gl, 2)
    p["avg_hold_time"] = round(decayed_hold_time / decayed_trades, 1) if decayed_trades > 0 else 0.0

    if decayed_gl > 0:
        p["profit_factor"] = round(decayed_gp / decayed_gl, 2)
    else:
        p["profit_factor"] = round(decayed_gp, 2) if decayed_gp > 0 else 1.0

    p["expectancy"] = round((decayed_gp - decayed_gl) / decayed_trades, 3) if decayed_trades > 0 else 0.0


def record_observation(
    journal: dict,
    key: str,
    won: bool,
    context: dict,
    pnl: Optional[float] = None,
    trade_date: Optional[str] = None,
    hold_time: int = 0,
    source: str = "historical_seed",
) -> None:
    """
    Add one outcome observation to a pattern key.
    Calculates performance metrics (Profit Factor, Expectancy) and updates
    multi-tier aggregations (Specific Pattern, Behavioral Archetype, and Regime-Specific Archetype).
    """
    if key not in journal["patterns"]:
        journal["patterns"][key] = {
            "trades":        0.0,
            "wins":          0.0,
            "losses":        0.0,
            "win_rate":      0.0,
            "gross_profit":  0.0,
            "gross_loss":    0.0,
            "profit_factor": 1.0,
            "expectancy":    0.0,
            "avg_hold_time": 0.0,
            "state":         "WATCHING",
            "context":       context,
            "history":       []
        }

    p = journal["patterns"][key]
    p["context"] = {**p.get("context", {}), **context}
    trade_pnl = pnl if pnl is not None else (1.0 if won else -1.0)
    t_date = trade_date if trade_date is not None else datetime.now().strftime("%Y-%m-%d")

    # 1. Update pattern-level stats
    _update_stats(p, won, trade_pnl, t_date, hold_time)

    # Reclassify technical pattern state
    trades = p["trades"]
    wr = p["win_rate"]
    pf = p["profit_factor"]
    ev = p["expectancy"]

    if trades >= 100:
        if wr >= 0.60 and pf >= 1.2 and ev > 0:
            p["state"] = "PROVEN"
        else:
            p["state"] = "UNRELIABLE"
    elif trades >= 50:
        if wr >= 0.55 and pf >= 1.1 and ev > 0:
            p["state"] = "VALIDATED"
        else:
            p["state"] = "UNRELIABLE"
    elif trades >= 20:
        p["state"] = "LEARNING"
    else:
        p["state"] = "WATCHING"

    # 2. Update parent behavioral state (legacy archetype) level stats (kept for compatibility)
    archetype = classify_behavioral_state(key)
    if "archetypes" not in journal:
        journal["archetypes"] = {}
    if archetype not in journal["archetypes"]:
        journal["archetypes"][archetype] = {
            "trades": 0.0, "wins": 0.0, "losses": 0.0, "win_rate": 0.0,
            "gross_profit": 0.0, "gross_loss": 0.0, "profit_factor": 1.0, "expectancy": 0.0,
            "avg_hold_time": 0.0, "history": [], "regimes": {}
        }
    a = journal["archetypes"][archetype]
    _update_stats(a, won, trade_pnl, t_date, hold_time)

    # 2b. Update canonical compressed archetype stats (new 6-state taxonomy)
    canonical = classify_archetype(key)
    if "canonical_archetypes" not in journal:
        journal["canonical_archetypes"] = {}
    if canonical not in journal["canonical_archetypes"]:
        journal["canonical_archetypes"][canonical] = {
            "trades": 0.0, "wins": 0.0, "losses": 0.0, "win_rate": 0.0,
            "gross_profit": 0.0, "gross_loss": 0.0, "profit_factor": 1.0, "expectancy": 0.0,
            "avg_hold_time": 0.0, "history": [], "regimes": {}
        }
    ca = journal["canonical_archetypes"][canonical]
    _update_stats(ca, won, trade_pnl, t_date, hold_time)

    # 3. Update regime-specific archetype level stats
    regime = key.split("|")[0] if "|" in key else "UNKNOWN"
    if "regimes" not in a:
        a["regimes"] = {}
    if regime not in a["regimes"]:
        a["regimes"][regime] = {
            "trades": 0.0, "wins": 0.0, "losses": 0.0, "win_rate": 0.0,
            "gross_profit": 0.0, "gross_loss": 0.0, "profit_factor": 1.0, "expectancy": 0.0,
            "avg_hold_time": 0.0, "history": []
        }
    ar = a["regimes"][regime]
    _update_stats(ar, won, trade_pnl, t_date, hold_time)

    # 3b. Update regime-specific canonical archetype stats
    if "regimes" not in ca:
        ca["regimes"] = {}
    if regime not in ca["regimes"]:
        ca["regimes"][regime] = {
            "trades": 0.0, "wins": 0.0, "losses": 0.0, "win_rate": 0.0,
            "gross_profit": 0.0, "gross_loss": 0.0, "profit_factor": 1.0, "expectancy": 0.0,
            "avg_hold_time": 0.0, "history": []
        }
    car = ca["regimes"][regime]
    _update_stats(car, won, trade_pnl, t_date, hold_time)

    # Track garbage ratio to prevent statistical landfilling
    journal["metadata"]["total_observations"] = journal["metadata"].get("total_observations", 0) + 1
    source_counts = journal["metadata"].setdefault("observation_sources", {})
    source_counts[source] = source_counts.get(source, 0) + 1
    
    unknown_trades = journal["archetypes"].get("UNKNOWN_NOISE", {}).get("trades", 0.0)
    total_o = journal["metadata"]["total_observations"]
    garbage_ratio = (unknown_trades / total_o) if total_o > 0 else 0.0
    journal["metadata"]["garbage_ratio"] = round(garbage_ratio, 3)
    if garbage_ratio > 0.40:
        logger.warning(
            "⚠️ Statistical Landfill Warning: UNKNOWN_NOISE ratio is elevated (%.1f%% > 40%%). "
            "Consider model reclustering or adding new behavioral states.",
            garbage_ratio * 100
        )



# ── Backtest Seeding ───────────────────────────────────────────────────────────

def seed_from_history(symbols: Optional[List[str]] = None) -> dict:
    """
    Seed the learning journal from 10-year historical data.

    Fully vectorized — precomputes regime, signals, and all four pattern key
    fields (trend_channel, volume_spike, near_support, candlestick) across every
    bar BEFORE entering the per-signal loop.  get_pattern_snapshot is never
    called inside the loop, giving ~30x speedup over the naive approach.

    Returns the populated journal dict.
    """
    import numpy as np
    from indicators.regime import Regime
    from config.settings import REGIME_CONFIG, INDICATOR_CONFIG, STRATEGY_CONFIG
    from strategies.rsi_ma_strategy import get_rsi_threshold, compute_pullback_depth, Signal

    symbols = symbols or DATA_CONFIG["symbols"]
    journal = load_journal()

    logger.info("Seeding learning journal (vectorized) for %d symbols", len(symbols))

    for symbol in symbols:
        try:
            df = load_stock_data(symbol)
            df = compute_all_indicators(df)

            min_rows = 250
            if len(df) < min_rows + OUTCOME_LOOKAHEAD:
                logger.warning("%s: not enough data (need %d bars)", symbol, min_rows + OUTCOME_LOOKAHEAD)
                continue

            # ── Column names ─────────────────────────────────────────────────
            ma_fast_col = f"SMA_{INDICATOR_CONFIG['ma_fast']}"
            ma_slow_col = f"SMA_{INDICATOR_CONFIG['ma_slow']}"
            atr_col     = f"ATR_{INDICATOR_CONFIG['atr_period']}"
            rsi_col     = f"RSI_{INDICATOR_CONFIG['rsi_period']}"
            vol_col     = f"Volume_MA_{INDICATOR_CONFIG['volume_ma_period']}"

            # ── Vectorized regime ─────────────────────────────────────────────
            atr_series = df[atr_col]
            atr_ratio  = atr_series / atr_series.rolling(50).mean()
            slope_norm = df[ma_fast_col].diff().rolling(REGIME_CONFIG["slope_window"]).mean() / df[ma_fast_col]
            price      = df["Close"]
            ma200      = df[ma_slow_col]
            ma50       = df[ma_fast_col]
            p_above    = price > ma200
            p_below    = (price < ma50) & (price < ma200)
            s_up, w_up = REGIME_CONFIG["slope_strong_up"],   REGIME_CONFIG["slope_weak_up"]
            s_dn, w_dn = REGIME_CONFIG["slope_strong_down"], REGIME_CONFIG["slope_weak_down"]
            atr_th     = REGIME_CONFIG["atr_ratio_threshold"]
            df["regime"] = np.select(
                [atr_ratio > atr_th,
                 p_above & (slope_norm >= s_up), p_above & (slope_norm >= w_up),
                 p_below & (slope_norm <= s_dn), p_below & (slope_norm <= w_dn)],
                [Regime.VOLATILE, Regime.STRONG_TREND_UP, Regime.WEAK_TREND_UP,
                 Regime.STRONG_TREND_DOWN, Regime.WEAK_TREND_DOWN],
                default=Regime.SIDEWAYS,
            )

            # ── Vectorized scoring & signals ──────────────────────────────────
            df["Pullback"]  = compute_pullback_depth(df)
            rsi_thresh      = df["regime"].apply(lambda r: get_rsi_threshold(Regime(r)))
            df["Score"]     = (
                (price > ma200).astype(int)
                + (df[rsi_col] < rsi_thresh).astype(int)
                + (df["Volume"] > df[vol_col] * STRATEGY_CONFIG["volume_multiplier"]).astype(int)
                + (df["Pullback"] >= STRATEGY_CONFIG["min_pullback_pct"]).astype(int)
            )
            min_scores  = df["regime"].apply(lambda r: Regime(r).min_score_to_buy)
            buy_allowed = df["regime"].apply(lambda r: Regime(r).allows_buy)
            df["Signal"] = np.select(
                [(df["Score"] >= min_scores) & buy_allowed,
                 df[rsi_col] > STRATEGY_CONFIG["rsi_sell_threshold"]],
                ["BUY", "SELL"], default="HOLD",
            )

            # ── Vectorized pattern features (no get_pattern_snapshot in loop) ─
            # 1. Trend channel: 20-bar rolling OLS slope (fully vectorized)
            #    For x=[0..N-1]: slope = (N*sum(x*y) - sum(x)*sum(y)) / (N*sum(x^2) - sum(x)^2)
            #    With N=20: sum(x)=190, sum(x^2)=2470, denom=20*2470-190^2=13400 (constants)
            lookback   = 20
            closes     = df["Close"].values
            n          = len(closes)
            _N         = float(lookback)
            _sx        = _N * (_N - 1) / 2.0          # sum(x) for x=0..N-1
            _sx2       = _N * (_N - 1) * (2*_N - 1) / 6.0  # sum(x^2)
            _denom     = _N * _sx2 - _sx ** 2          # = 13400 for N=20
            # x-weights for dot product: w_i = N*i - sum(x)
            _x_weights = np.arange(_N) * _N - _sx      # shape (20,)
            # Rolling dot product via convolution (mode='full', then slice)
            _conv      = np.convolve(closes, _x_weights[::-1], mode='valid')  # len = n - lookback + 1
            _roll_mean = np.convolve(closes, np.ones(lookback)/lookback, mode='valid')
            _raw_slopes = _conv / _denom                # un-normalised slope
            # Normalise by rolling last-bar price (pad with nan for first lookback-1 bars)
            _last_price = closes[lookback - 1:]        # length = n - lookback + 1
            _norm_slopes = np.where(_last_price > 0, _raw_slopes / _last_price, 0.0)
            # Build channel array
            ch_full = np.full(n, "SIDEWAYS", dtype=object)
            ch_full[lookback - 1:] = np.where(
                _norm_slopes > 0.002, "RISING",
                np.where(_norm_slopes < -0.002, "FALLING", "SIDEWAYS")
            )
            df["_channel"] = ch_full

            # 2. Volume spike: current volume > 2x prior 20-bar average
            vol_avg_20      = df["Volume"].rolling(21).mean().shift(1)
            df["_vol_spike"] = (df["Volume"] > 2.0 * vol_avg_20).fillna(False)

            # 3. Near support: close within 2% of 10-bar swing low
            swing_low      = df["Low"].rolling(10).min()
            near_ratio     = (df["Close"] - swing_low).abs() / df["Close"].clip(lower=1e-6)
            df["_near_sup"] = (near_ratio <= 0.02)

            # 4. Candlestick: classify last bar only (simplified)
            op = df["Open"]; hi = df["High"]; lo = df["Low"]; cl = df["Close"]
            body         = (cl - op).abs()
            upper_shadow = hi - np.maximum(op, cl)
            lower_shadow = np.minimum(op, cl) - lo
            tot_range    = (hi - lo).clip(lower=1e-9)
            safe_body    = body.clip(lower=1e-9)
            is_hammer    = (lower_shadow >= 2 * safe_body) & (upper_shadow <= 0.3 * safe_body) & (cl > op) & (body / tot_range < 0.4)
            is_bull_eng  = (cl > op) & (cl.shift(1) > op.shift(1)) & (cl > op.shift(1)) & (op < cl.shift(1))
            is_doji      = (body / tot_range < 0.1)
            df["_candle"] = np.select(
                [is_hammer, is_bull_eng, is_doji],
                ["HAMMER",  "BULLISH_ENGULFING", "DOJI"],
                default="NOCNDLE",
            )

            # ── Per-signal loop (all lookups are O(1) now) ────────────────────
            wins = losses = 0
            sig_arr = df["Signal"].values
            reg_arr = df["regime"].values
            rsi_arr = df[rsi_col].values
            scr_arr = df["Score"].values
            atr_arr = df[atr_col].values
            chn_arr = df["_channel"].values
            vol_arr = df["_vol_spike"].values
            sup_arr = df["_near_sup"].values
            cdl_arr = df["_candle"].values
            idx_arr = df.index

            for i in range(min_rows, n - OUTCOME_LOOKAHEAD):
                if sig_arr[i] != "BUY":
                    continue

                regime = str(reg_arr[i])
                rsi    = float(rsi_arr[i])
                score  = int(scr_arr[i])
                atr    = float(atr_arr[i])
                entry  = float(closes[i])
                cdl    = str(cdl_arr[i])

                key = build_condition_key(
                    regime       = regime,
                    rsi_bucket   = rsi_to_bucket(rsi),
                    score        = score,
                    channel      = str(chn_arr[i]),
                    candles      = [cdl] if cdl != "NOCNDLE" else [],
                    near_support = bool(sup_arr[i]),
                    volume_spike = bool(vol_arr[i]),
                )

                future = closes[i + 1: i + 1 + OUTCOME_LOOKAHEAD]
                if len(future) == 0:
                    continue

                target    = entry + atr
                stop      = entry - atr
                won       = None
                hold_time = 0
                for bar_idx, p_val in enumerate(future):
                    if p_val >= target:
                        won = True;  hold_time = bar_idx + 1; break
                    elif p_val <= stop:
                        won = False; hold_time = bar_idx + 1; break

                if won is None:
                    continue

                rsi_bkt    = rsi_to_bucket(rsi)
                context    = {"symbol": symbol, "regime": regime, "rsi_bkt": rsi_bkt, "score": score}
                trade_date = str(idx_arr[i].strftime("%Y-%m-%d"))
                record_observation(journal, key, won, context, trade_date=trade_date, hold_time=hold_time)
                if won: wins += 1
                else:   losses += 1

            logger.info("%s: seeded %d wins + %d losses", symbol, wins, losses)

        except Exception as exc:
            logger.warning("Seed failed for %s: %s", symbol, exc)

    save_journal(journal)
    return journal


# ── Live Trade Recording ───────────────────────────────────────────────────────

def record_live_trade(
    symbol: str,
    signal_result: dict,
    won: bool,
    df: Optional[pd.DataFrame] = None,
    pnl: Optional[float] = None,
    hold_time: int = 0,
    trade: Optional[dict] = None,
) -> None:
    """
    Record the outcome of a live paper trade into the learning journal.
    Call this when a paper trade is closed.
    """
    journal = load_journal()

    regime   = signal_result.get("regime", "UNKNOWN")
    rsi      = float(signal_result.get("rsi", 50))
    score    = int(signal_result.get("score", 0))
    rsi_bkt  = rsi_to_bucket(rsi)
    key = signal_result.get("pattern_key")
    snap = signal_result.get("pattern_snapshot", {})

    if not key:
        if df is not None:
            try:
                snap = get_pattern_snapshot(df)
            except Exception:
                snap = {}

        key = build_condition_key(
            regime       = regime,
            rsi_bucket   = rsi_bkt,
            score        = score,
            channel      = snap.get("trend_channel", "SIDEWAYS"),
            candles      = snap.get("candlestick", []),
            near_support = snap.get("near_support") is not None,
            volume_spike = snap.get("volume_spike", False),
        )

    context = {
        "symbol": symbol,
        "regime": regime,
        "rsi_bkt": rsi_bkt,
        "score": score,
        "source": "live_paper",
    }
    trade_date = (trade or {}).get("exit_date") or datetime.now().strftime("%Y-%m-%d")
    record_observation(
        journal,
        key,
        won,
        context,
        pnl=pnl,
        trade_date=trade_date,
        hold_time=hold_time,
        source="live_paper",
    )

    if trade is not None:
        journal.setdefault("closed_trades", []).append({
            **trade,
            "pattern_key": key,
            "pattern_snapshot": snap,
        })
        if len(journal["closed_trades"]) > 1000:
            journal["closed_trades"] = journal["closed_trades"][-1000:]

    save_journal(journal)
    logger.info("Recorded live trade outcome for %s: %s", symbol, "WIN" if won else "LOSS")


# ── State Summary ──────────────────────────────────────────────────────────────

def get_knowledge_summary(journal: dict) -> dict:
    """
    Summarise the knowledge base into PROVEN / VALIDATED / LEARNING / WATCHING / UNRELIABLE buckets.
    """
    proven      = []
    validated   = []
    learning    = []
    watching    = []
    unreliable  = []

    for key, data in journal["patterns"].items():
        entry = {
            "key":           key,
            "trades":        data["trades"],
            "wins":          data["wins"],
            "win_rate":      data["win_rate"],
            "profit_factor": data.get("profit_factor", 1.0),
            "expectancy":    data.get("expectancy", 0.0),
            "context":       data.get("context", {}),
        }
        state = data.get("state", "WATCHING")

        if state == "PROVEN":
            proven.append(entry)
        elif state == "VALIDATED":
            validated.append(entry)
        elif state == "LEARNING":
            learning.append(entry)
        elif state == "UNRELIABLE":
            unreliable.append(entry)
        else:
            watching.append(entry)

    # Sort: highest win rate & trades first
    proven.sort(key=lambda x: (x["expectancy"], x["trades"]), reverse=True)
    validated.sort(key=lambda x: (x["expectancy"], x["trades"]), reverse=True)
    learning.sort(key=lambda x: x["trades"], reverse=True)

    # Bayesian prior blending for sparse patterns: smooth expectancy & win_rate
    canonical = journal.get("canonical_archetypes", {})
    for group in (proven, validated, learning, watching, unreliable):
        for e in group:
            trades = e.get("trades", 0.0)
            if trades < 50:
                # determine parent archetype and blend
                parent = classify_archetype(e["key"])
                # Blend expectancy using regime-aware prior
                blended_exp = get_blended_expectancy(e, parent, e.get("context", {}).get("regime", "UNKNOWN"), journal)
                e["expectancy_blended"] = blended_exp
                # Blend win rate with archetype prior (simple linear smoothing)
                arch = canonical.get(parent, {})
                arch_wr = arch.get("win_rate", 0.0)
                c = 30.0
                e_wr = e.get("win_rate", 0.0)
                e["win_rate_blended"] = round(((trades * e_wr) + (c * arch_wr)) / (trades + c), 3) if (trades + c) > 0 else e_wr
            else:
                e["expectancy_blended"] = e.get("expectancy", 0.0)
                e["win_rate_blended"] = e.get("win_rate", 0.0)

    observation_sources = dict(journal["metadata"].get("observation_sources", {}))
    known_sources = sum(observation_sources.values())
    total_observations = journal["metadata"].get("total_observations", 0)
    if total_observations > known_sources:
        observation_sources["historical_seed"] = observation_sources.get("historical_seed", 0) + (total_observations - known_sources)

    return {
        "proven":             proven,
        "validated":          validated,
        "learning":           learning,
        "unreliable":         unreliable,
        "watching":           watching[:10],   # top 10 most recent watching patterns
        "total_patterns":     len(journal["patterns"]),
        "total_observations": total_observations,
        "observation_sources": observation_sources,
        "last_updated":       journal["metadata"].get("last_updated"),
        # Prefer canonical compressed archetypes for reporting; fall back to legacy archetypes
        "archetypes":         journal.get("canonical_archetypes", journal.get("archetypes", {})),
    }
