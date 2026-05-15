"""
Learning Engine
Tracks detected patterns against their outcomes and builds a knowledge base.

Three knowledge states per pattern condition:
  LEARNED   — ≥5 trades, ≥60% win rate → trusted, used to guide decisions
  LEARNING  — 2–4 trades, any win rate → gathering evidence
  WATCHING  — pattern detected but 0 trades taken yet → monitoring

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

# Thresholds for knowledge state
MIN_TRADES_LEARNED  = 5
MIN_WIN_RATE_LEARNED = 0.60
MIN_TRADES_LEARNING  = 2

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
    divergence: Optional[str],
    volume_spike: bool,
) -> str:
    """
    Build a canonical string key for a pattern condition.
    Example: 'SIDEWAYS|RSI<30|S3|RISING|HAMMER|SUPP|BULL_DIV|VOLSPK'
    """
    parts = [
        regime,
        rsi_bucket,
        f"S{score}",
        channel,
        "+".join(candles) if candles else "NOCNDLE",
        "SUPP" if near_support else "NOSUPP",
        divergence if divergence else "NODIV",
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

def record_observation(journal: dict, key: str, won: bool, context: dict) -> None:
    """Add one outcome observation to a pattern key."""
    if key not in journal["patterns"]:
        journal["patterns"][key] = {
            "trades":   0,
            "wins":     0,
            "losses":   0,
            "win_rate": 0.0,
            "state":    "WATCHING",
            "context":  context,
        }

    p = journal["patterns"][key]
    p["trades"] += 1
    p["wins"]   += 1 if won else 0
    p["losses"] += 0 if won else 1
    p["win_rate"] = round(p["wins"] / p["trades"], 3) if p["trades"] > 0 else 0.0

    # Classify state
    if p["trades"] >= MIN_TRADES_LEARNED and p["win_rate"] >= MIN_WIN_RATE_LEARNED:
        p["state"] = "LEARNED"
    elif p["trades"] >= MIN_TRADES_LEARNING:
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
                    divergence   = snap["rsi_divergence"],
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

                context = {
                    "symbol":  symbol,
                    "regime":  regime,
                    "rsi_bkt": rsi_bkt,
                    "score":   score,
                }

                record_observation(journal, key, won, context)
                if won: wins += 1
                else:   losses += 1

            logger.info("%s: seeded %d wins + %d losses from history", symbol, wins, losses)

        except Exception as exc:
            logger.warning("Seed failed for %s: %s", symbol, exc)

    save_journal(journal)
    return journal


# ── Live Trade Recording ───────────────────────────────────────────────────────

def record_live_trade(symbol: str, signal_result: dict, won: bool, df: pd.DataFrame) -> None:
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
        divergence   = snap.get("rsi_divergence"),
        volume_spike = snap.get("volume_spike", False),
    )

    context = {"symbol": symbol, "regime": regime, "rsi_bkt": rsi_bkt, "score": score}
    record_observation(journal, key, won, context)
    save_journal(journal)
    logger.info("Recorded live trade outcome for %s: %s", symbol, "WIN" if won else "LOSS")


# ── State Summary ──────────────────────────────────────────────────────────────

def get_knowledge_summary(journal: dict) -> dict:
    """
    Summarise the knowledge base into LEARNED / LEARNING / WATCHING buckets.
    """
    learned  = []
    learning = []
    watching = []

    for key, data in journal["patterns"].items():
        entry = {
            "key":      key,
            "trades":   data["trades"],
            "wins":     data["wins"],
            "win_rate": data["win_rate"],
            "context":  data.get("context", {}),
        }
        state = data.get("state", "WATCHING")

        if state == "LEARNED":
            learned.append(entry)
        elif state == "LEARNING":
            learning.append(entry)
        else:
            watching.append(entry)

    # Sort: highest win rate first
    learned.sort(key=lambda x: x["win_rate"], reverse=True)
    learning.sort(key=lambda x: x["trades"], reverse=True)

    return {
        "learned":  learned,
        "learning": learning,
        "watching": watching[:10],   # top 10 most recent watching patterns
        "total_patterns": len(journal["patterns"]),
        "total_observations": journal["metadata"].get("total_observations", 0),
        "last_updated": journal["metadata"].get("last_updated"),
    }
