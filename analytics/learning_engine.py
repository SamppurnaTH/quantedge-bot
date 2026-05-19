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
from data.fetcher import fetch_stock_data
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

def record_observation(journal: dict, key: str, won: bool, context: dict, pnl: Optional[float] = None, trade_date: Optional[str] = None) -> None:
    """
    Add one outcome observation to a pattern key.
    Calculates institutional performance metrics (Profit Factor, Expectancy)
    with Exponential Time Decay built-in.
    """
    import math
    
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
            "state":         "WATCHING",
            "context":       context,
            "history":       []
        }

    p = journal["patterns"][key]
    
    # Track trade outcome in history list
    if "history" not in p:
        p["history"] = []
        
    # PnL tracking for exact profit factor and expectancy
    if pnl is None:
        trade_pnl = 1.0 if won else -1.0
    else:
        trade_pnl = pnl

    if trade_date is None:
        trade_date = datetime.now().strftime("%Y-%m-%d")

    p["history"].append({
        "date": trade_date,
        "won": won,
        "pnl": trade_pnl
    })

    # Limit history list to last 200 trades to prevent file bloat
    if len(p["history"]) > 200:
        p["history"] = p["history"][-200:]

    # Apply Exponential Decay (CRITICAL ISSUE #5)
    decay_lambda = 0.0019  # ~365 days half-life (older trades slowly lose significance)
    current_dt = datetime.now()
    
    decayed_wins = 0.0
    decayed_losses = 0.0
    decayed_gp = 0.0
    decayed_gl = 0.0
    decayed_trades = 0.0
    
    for t_item in p["history"]:
        try:
            t_dt = datetime.strptime(t_item["date"], "%Y-%m-%d")
        except ValueError:
            t_dt = current_dt
        
        age_days = max((current_dt - t_dt).days, 0)
        weight = math.exp(-decay_lambda * age_days)
        
        decayed_trades += weight
        if t_item["won"]:
            decayed_wins += weight
            if t_item["pnl"] > 0:
                decayed_gp += t_item["pnl"] * weight
        else:
            decayed_losses += weight
            decayed_gl += abs(t_item["pnl"]) * weight

    # Set decayed metrics
    p["trades"] = round(decayed_trades, 2)
    p["wins"] = round(decayed_wins, 2)
    p["losses"] = round(decayed_losses, 2)
    p["win_rate"] = round(decayed_wins / decayed_trades, 3) if decayed_trades > 0 else 0.0
    p["gross_profit"] = round(decayed_gp, 2)
    p["gross_loss"] = round(decayed_gl, 2)

    # Calculate Profit Factor
    if decayed_gl > 0:
        p["profit_factor"] = round(decayed_gp / decayed_gl, 2)
    else:
        p["profit_factor"] = round(decayed_gp, 2) if decayed_gp > 0 else 1.0

    # Calculate Expectancy
    p["expectancy"] = round((decayed_gp - decayed_gl) / decayed_trades, 3) if decayed_trades > 0 else 0.0

    # Institutional knowledge state classification
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

    journal["metadata"]["total_observations"] = journal["metadata"].get("total_observations", 0) + 1


# ── Backtest Seeding ───────────────────────────────────────────────────────────

def seed_from_history(symbols: Optional[List[str]] = None) -> dict:
    """
    Seed the learning journal from 10-year historical data.

    For each BUY signal generated in backtesting, we simulate the outcome
    by checking if price rose > ATR (win) or fell > ATR (loss) in the next
    OUTCOME_LOOKAHEAD bars.

    Returns the populated journal dict.
    """
    symbols = symbols or DATA_CONFIG["symbols"]
    journal = load_journal()

    logger.info("Seeding learning journal from 10-year backtest data for: %s", symbols)

    for symbol in symbols:
        try:
            df = fetch_stock_data(symbol, save=True)
            df = compute_all_indicators(df)

            # Slide window through history to find BUY signals and their outcomes
            min_rows = 250   # need enough bars for all indicators to warm up
            if len(df) < min_rows + OUTCOME_LOOKAHEAD:
                logger.warning("%s: not enough data to seed (need %d bars)", symbol, min_rows + OUTCOME_LOOKAHEAD)
                continue

            wins = losses = 0

            for i in range(min_rows, len(df) - OUTCOME_LOOKAHEAD):
                slice_df = df.iloc[:i + 1].copy()

                # Detect regime
                try:
                    regime_result = detect_regime_detailed(slice_df)
                    regime = str(regime_result.regime)
                except Exception:
                    continue

                # Generate signal at this bar
                try:
                    sig_df = generate_signals(slice_df, market_up=regime_result.regime.allows_buy,
                                              regime=regime_result.regime)
                    latest = get_latest_signal(sig_df)
                except Exception:
                    continue

                if latest.get("signal") != "BUY":
                    continue

                # Detect patterns at this bar
                try:
                    snap = get_pattern_snapshot(slice_df)
                except Exception:
                    continue

                # Build condition key
                rsi       = float(latest.get("rsi", 50))
                score     = int(latest.get("score", 0))
                rsi_bkt   = rsi_to_bucket(rsi)
                atr       = float(latest.get("atr") or slice_df["ATR_14"].iloc[-1])
                entry     = float(latest.get("close", 0))

                key = build_condition_key(
                    regime       = regime,
                    rsi_bucket   = rsi_bkt,
                    score        = score,
                    channel      = snap["trend_channel"],
                    candles      = snap["candlestick"],
                    near_support = snap["near_support"] is not None,
                    volume_spike = snap["volume_spike"],
                )

                # Determine outcome: did price rise > 1× ATR in next N bars?
                future_prices = df["Close"].iloc[i + 1: i + 1 + OUTCOME_LOOKAHEAD].values
                if len(future_prices) == 0:
                    continue

                max_future  = float(future_prices.max())
                min_future  = float(future_prices.min())
                target      = entry + atr
                stop        = entry - atr

                if max_future >= target:
                    won = True
                elif min_future <= stop:
                    won = False
                else:
                    # Neither hit — skip ambiguous outcome
                    continue

                trade_date = str(slice_df.index[-1].strftime("%Y-%m-%d"))
                record_observation(journal, key, won, context, trade_date=trade_date)
                if won: wins += 1
                else:   losses += 1

            logger.info("%s: seeded %d wins + %d losses from history", symbol, wins, losses)

        except Exception as exc:
            logger.warning("Seed failed for %s: %s", symbol, exc)

    save_journal(journal)
    return journal


# ── Live Trade Recording ───────────────────────────────────────────────────────

def record_live_trade(symbol: str, signal_result: dict, won: bool, df: pd.DataFrame, pnl: Optional[float] = None) -> None:
    """
    Record the outcome of a live paper trade into the learning journal.
    Call this when a paper trade is closed.
    """
    journal = load_journal()

    try:
        snap = get_pattern_snapshot(df)
    except Exception:
        snap = {}

    regime   = signal_result.get("regime", "UNKNOWN")
    rsi      = float(signal_result.get("rsi", 50))
    score    = int(signal_result.get("score", 0))
    rsi_bkt  = rsi_to_bucket(rsi)

    key = build_condition_key(
        regime       = regime,
        rsi_bucket   = rsi_bkt,
        score        = score,
        channel      = snap.get("trend_channel", "SIDEWAYS"),
        candles      = snap.get("candlestick", []),
        near_support = snap.get("near_support") is not None,
        volume_spike = snap.get("volume_spike", False),
    )

    context = {"symbol": symbol, "regime": regime, "rsi_bkt": rsi_bkt, "score": score}
    trade_date = datetime.now().strftime("%Y-%m-%d")
    record_observation(journal, key, won, context, pnl=pnl, trade_date=trade_date)
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

    return {
        "proven":             proven,
        "validated":          validated,
        "learning":           learning,
        "unreliable":         unreliable,
        "watching":           watching[:10],   # top 10 most recent watching patterns
        "total_patterns":     len(journal["patterns"]),
        "total_observations": journal["metadata"].get("total_observations", 0),
        "last_updated":       journal["metadata"].get("last_updated"),
    }
