"""
Learning Engine  (journal v2 — R-multiple based)
Tracks every entry condition the bot trades against the outcome of the trade
the bot ACTUALLY takes, and turns that into a statistically honest edge
estimate that drives trade gating and position sizing.

What changed vs v1 (and why it matters):
  * One unit: every outcome is an R-multiple (net P&L / initial risk). v1
    mixed ±1 "R" seed outcomes with live rupee P&L (e.g. −138), so a single
    live loss outweighed hundreds of historical trades.
  * Same trade: history is simulated with execution/trade_model.py — next-
    open entry, 2×ATR stop, 2R target, gap/intraday handling, RSI & time
    exits, costs — instead of a symmetric ±1 ATR close-only label.
  * Same signal: history uses strategies.build_signal_frame (index regime,
    market filter, effective regime) and the same feature function as live.
  * Independent samples: one position per symbol at a time + cooldown. v1
    counted every consecutive BUY bar of the same dip as a new "trade".
  * Timeouts count: trades that hit neither stop nor target are exited by the
    time stop and recorded. v1 silently dropped them (survivorship).
  * No 200-observation cap: decayed sufficient statistics are kept
    incrementally (exact exponential decay, O(1) storage). v1 truncated every
    archetype to its last 200 observations, so ~24k seeded trades collapsed
    to ~30 effective trades and nothing could ever reach VALIDATED.
  * States from significance, not raw win rate: a 2R-target strategy is
    profitable at a 40% win rate, but v1 required ≥55–60% to be "valid".
  * Hierarchical Bayesian shrinkage: pattern → archetype×regime → archetype →
    regime → global, so sparse patterns borrow strength instead of being noise.

Knowledge states (n = effective sample size, bounds are 90% one-sided):
  PROVEN      n ≥ 100, lower bound of mean R > 0, profit factor ≥ 1.2
  VALIDATED   n ≥ 50,  lower bound of mean R > 0
  UNRELIABLE  n ≥ 30 and upper bound < 0, or n ≥ 50 and mean R ≤ 0
  LEARNING    n ≥ 20
  WATCHING    n < 20
"""

import os
import json
import math
import logging
import shutil
from datetime import datetime
from typing import Dict, List, Optional

import pandas as pd

from analytics.pattern_detector import get_pattern_snapshot
from data.fetcher import fetch_stock_data, load_stock_data
from indicators.engine import compute_all_indicators
from indicators.regime import detect_regime_detailed
from strategies.rsi_ma_strategy import generate_signals, get_latest_signal
from config.settings import DATA_CONFIG, RISK_CONFIG, STRATEGY_CONFIG, LEARNING_CONFIG, EXECUTION_CONFIG

logger = logging.getLogger(__name__)

LEARNING_JOURNAL_FILE = os.path.join("state", "learning_journal.json")
JOURNAL_VERSION = LEARNING_CONFIG["journal_version"]
DECAY_LAMBDA = math.log(2) / LEARNING_CONFIG["half_life_days"]
SHRINK_C = float(LEARNING_CONFIG["shrinkage_trades"])
Z = LEARNING_CONFIG["z_one_sided"]

OUTCOME_LOOKAHEAD = EXECUTION_CONFIG["max_hold_bars"]   # kept for backward compatibility


# ── Journal I/O ────────────────────────────────────────────────────────────────

def _empty_journal() -> dict:
    return {
        "patterns": {}, "archetypes": {}, "canonical_archetypes": {},
        "regimes": {}, "global": {}, "closed_trades": [],
        "live_drift": {"trades": 0, "sum_expected_r": 0.0, "sum_actual_r": 0.0},
        "metadata": {"version": JOURNAL_VERSION, "last_updated": None,
                     "total_observations": 0, "observation_sources": {}},
    }


def load_journal() -> dict:
    """
    Load the learning journal. A journal from an older schema (v1 mixed
    rupee/R units) is moved aside and an empty v2 journal is returned, which
    makes run_daily re-seed it from history with the corrected simulator.
    """
    if not os.path.exists(LEARNING_JOURNAL_FILE):
        return _empty_journal()
    with open(LEARNING_JOURNAL_FILE, "r", encoding="utf-8") as f:
        journal = json.load(f)
    if journal.get("metadata", {}).get("version") != JOURNAL_VERSION:
        os.makedirs("logs", exist_ok=True)
        backup = os.path.join("logs", "learning_journal_v1_backup.json")
        shutil.copyfile(LEARNING_JOURNAL_FILE, backup)
        logger.warning("Learning journal schema is outdated — backed up to %s; it will be re-seeded.", backup)
        return _empty_journal()
    return journal


def save_journal(journal: dict) -> None:
    """Save the learning journal to disk (atomically)."""
    os.makedirs("state", exist_ok=True)
    journal["metadata"]["last_updated"] = datetime.now().strftime("%Y-%m-%d %H:%M")
    journal["metadata"]["version"] = JOURNAL_VERSION
    tmp = LEARNING_JOURNAL_FILE + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(journal, f, indent=1)
    os.replace(tmp, LEARNING_JOURNAL_FILE)
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


# ── Decayed sufficient statistics ─────────────────────────────────────────────

_SUMS = ("sw", "sw2", "swr", "swr2", "sww", "sgp", "sgl", "swh")


def _new_node() -> dict:
    node = {k: 0.0 for k in _SUMS}
    node.update({"ref_date": None, "count": 0, "trades": 0.0, "wins": 0.0, "losses": 0.0,
                 "win_rate": 0.0, "gross_profit": 0.0, "gross_loss": 0.0, "profit_factor": 1.0,
                 "expectancy": 0.0, "std_r": 0.0, "lower_r": 0.0, "upper_r": 0.0,
                 "avg_hold_time": 0.0})
    return node


def _add(node: dict, r: float, won: bool, date_str: str, hold: float) -> None:
    """
    Add one observation with exact exponential time decay. Sums are stored
    relative to node['ref_date']; out-of-order dates (seeding symbol by
    symbol) are handled by down-weighting the older observation instead.
    """
    try:
        t = datetime.strptime(date_str[:10], "%Y-%m-%d")
    except Exception:
        t = datetime.now()
    ref = datetime.strptime(node["ref_date"], "%Y-%m-%d") if node.get("ref_date") else None
    if ref is None:
        node["ref_date"], w = t.strftime("%Y-%m-%d"), 1.0
    elif t > ref:
        f = math.exp(-DECAY_LAMBDA * (t - ref).days)
        for k in _SUMS:
            node[k] *= (f * f) if k == "sw2" else f
        node["ref_date"], w = t.strftime("%Y-%m-%d"), 1.0
    else:
        w = math.exp(-DECAY_LAMBDA * (ref - t).days)

    node["sw"]   += w
    node["sw2"]  += w * w
    node["swr"]  += w * r
    node["swr2"] += w * r * r
    node["sww"]  += w * (1.0 if won else 0.0)
    node["sgp"]  += w * max(r, 0.0)
    node["sgl"]  += w * max(-r, 0.0)
    node["swh"]  += w * hold
    node["count"] = node.get("count", 0) + 1
    _derive(node)


def _derive(node: dict) -> None:
    sw = node["sw"]
    if sw <= 0:
        return
    n_eff = sw * sw / node["sw2"] if node["sw2"] > 0 else 0.0
    mean = node["swr"] / sw
    var = max(node["swr2"] / sw - mean * mean, 0.0)
    se = math.sqrt(var / n_eff) if n_eff > 0 else float("inf")
    node["trades"] = round(n_eff, 2)                      # effective sample size
    node["win_rate"] = round(node["sww"] / sw, 4)
    node["wins"] = round(node["win_rate"] * n_eff, 2)
    node["losses"] = round(n_eff - node["wins"], 2)
    node["expectancy"] = round(mean, 4)                  # mean R per trade
    node["std_r"] = round(math.sqrt(var), 4)
    node["lower_r"] = round(mean - Z * se, 4) if math.isfinite(se) else -99.0
    node["upper_r"] = round(mean + Z * se, 4) if math.isfinite(se) else 99.0
    node["gross_profit"] = round(node["sgp"] / sw * n_eff, 3)   # in R, scaled to n_eff
    node["gross_loss"] = round(node["sgl"] / sw * n_eff, 3)
    node["profit_factor"] = round(node["sgp"] / node["sgl"], 3) if node["sgl"] > 0 else (99.0 if node["sgp"] > 0 else 1.0)
    node["avg_hold_time"] = round(node["swh"] / sw, 1)


def _classify(node: dict) -> str:
    n, mean = node["trades"], node["expectancy"]
    lo, hi, pf = node["lower_r"], node["upper_r"], node["profit_factor"]
    if n >= 100 and lo > 0 and pf >= 1.2:
        return "PROVEN"
    if n >= 50 and lo > 0:
        return "VALIDATED"
    if (n >= 30 and hi < 0) or (n >= 50 and mean <= 0):
        return "UNRELIABLE"
    if n >= 20:
        return "LEARNING"
    return "WATCHING"


def _node(container: dict, key: str) -> dict:
    if key not in container or "sw" not in container[key]:
        container[key] = _new_node()
    return container[key]


# ── Observation Recording ─────────────────────────────────────────────────────

def record_observation(
    journal: dict,
    key: str,
    won: bool,
    context: dict,
    pnl: Optional[float] = None,
    trade_date: Optional[str] = None,
    hold_time: int = 0,
    source: str = "historical_seed",
    r: Optional[float] = None,
) -> None:
    """
    Add one trade outcome to the pattern and all of its parent levels.
    `r` is the R-multiple of the trade. `pnl` (rupees) is accepted for
    backward compatibility but is NEVER mixed into the statistics.
    """
    if r is None:
        r = 1.0 if won else -1.0
    r = float(max(min(r, 10.0), -10.0))          # clip pathological fills
    won = r > 0
    t_date = trade_date or datetime.now().strftime("%Y-%m-%d")
    regime = key.split("|")[0] if "|" in key else "UNKNOWN"

    p = _node(journal["patterns"], key)
    p["context"] = {**p.get("context", {}), **context}
    _add(p, r, won, t_date, hold_time)
    p["state"] = _classify(p)
    recent = p.setdefault("recent", [])
    recent.append({"date": t_date, "r": round(r, 3), "src": source})
    del recent[:-20]

    for bucket, name in (("archetypes", classify_behavioral_state(key)),
                         ("canonical_archetypes", classify_archetype(key))):
        node = _node(journal.setdefault(bucket, {}), name)
        _add(node, r, won, t_date, hold_time)
        node["state"] = _classify(node)
        sub = _node(node.setdefault("regimes", {}), regime)
        _add(sub, r, won, t_date, hold_time)

    _add(_node(journal.setdefault("regimes", {}), regime), r, won, t_date, hold_time)
    g = journal.get("global") or _new_node()
    journal["global"] = g
    _add(g, r, won, t_date, hold_time)

    meta = journal["metadata"]
    meta["total_observations"] = meta.get("total_observations", 0) + 1
    src = meta.setdefault("observation_sources", {})
    src[source] = src.get(source, 0) + 1
    unknown = journal["canonical_archetypes"].get("UNKNOWN_NOISE", {}).get("count", 0)
    meta["garbage_ratio"] = round(unknown / meta["total_observations"], 3)


def get_blended_expectancy(pattern_data: dict, archetype_name: str, regime_name: str, journal: dict) -> float:
    """Backward-compatible wrapper — see get_edge_estimate()."""
    key = None
    for k, v in journal.get("patterns", {}).items():
        if v is pattern_data:
            key = k
            break
    if key is None:
        return round(pattern_data.get("expectancy", 0.0), 3)
    return round(get_edge_estimate(journal, key)["expected_r"], 3)


def get_edge_estimate(journal: dict, key: str) -> dict:
    """
    Hierarchically shrunk estimate of the R-expectancy of a setup.

        global → regime → canonical archetype → archetype×regime → pattern

    At each level: posterior = (n·level_mean + C·prior_mean) / (n + C).
    A pattern with 3 trades is therefore mostly its archetype's edge, while a
    pattern with 300 trades speaks for itself.

    Returns expected_r, win_prob, lower_r (90% one-sided), n (pattern's own
    effective sample), evidence (deepest level with ≥ C effective trades),
    state.
    """
    regime = key.split("|")[0] if "|" in key else "UNKNOWN"
    arch = classify_archetype(key)
    ca = journal.get("canonical_archetypes", {}).get(arch, {})
    chain = [
        ("global", journal.get("global") or {}),
        ("regime", journal.get("regimes", {}).get(regime, {})),
        ("archetype", ca),
        ("archetype_regime", ca.get("regimes", {}).get(regime, {})),
        ("pattern", journal.get("patterns", {}).get(key, {})),
    ]
    mean, wr, var, n_tot = 0.0, 0.5, 1.0, 0.0
    evidence = "none"
    first = True
    for name, node in chain:
        n = float(node.get("trades", 0.0) or 0.0)
        if n <= 0:
            continue
        m, w, sd = node.get("expectancy", 0.0), node.get("win_rate", 0.5), node.get("std_r", 1.0)
        if first:
            mean, wr, var, n_tot, first = m, w, sd * sd, n, False
        else:
            mean = (n * m + SHRINK_C * mean) / (n + SHRINK_C)
            wr = (n * w + SHRINK_C * wr) / (n + SHRINK_C)
            var = (n * sd * sd + SHRINK_C * var) / (n + SHRINK_C)
            n_tot = n + SHRINK_C
        if n >= SHRINK_C:
            evidence = name
    pattern = journal.get("patterns", {}).get(key, {})
    se = math.sqrt(var / n_tot) if n_tot > 0 else float("inf")
    return {
        "expected_r": round(mean, 4),
        "win_prob": round(wr, 4),
        "lower_r": round(mean - Z * se, 4) if math.isfinite(se) else -99.0,
        "n": round(float(pattern.get("trades", 0.0) or 0.0), 2),
        "evidence": evidence,
        "state": pattern.get("state", "NEW"),
        "archetype": arch,
    }


# ── Backtest Seeding ───────────────────────────────────────────────────────────

def simulate_history(symbols: Optional[List[str]] = None):
    """
    Yield one record per historical trade, simulated with the exact live
    rules (signal frame + trade model, one position per symbol + cooldown).
    """
    import numpy as np
    from indicators.regime import Regime
    from strategies.rsi_ma_strategy import build_signal_frame
    from analytics.pattern_detector import compute_pattern_features
    from execution import trade_model
    from config.settings import INDICATOR_CONFIG

    symbols = symbols or DATA_CONFIG["symbols"]
    cooldown = EXECUTION_CONFIG["cooldown_days"]
    atr_col = f"ATR_{INDICATOR_CONFIG['atr_period']}"
    rsi_col = f"RSI_{INDICATOR_CONFIG['rsi_period']}"
    try:
        index_df = compute_all_indicators(load_stock_data(DATA_CONFIG["market_index"]))
    except Exception as exc:
        logger.warning("Index data unavailable (%s) — simulating without market filter", exc)
        index_df = None

    for symbol in symbols:
        try:
            df = compute_all_indicators(load_stock_data(symbol))
            if len(df) < 250 + OUTCOME_LOOKAHEAD:
                logger.warning("%s: not enough data", symbol)
                continue
            sf = build_signal_frame(df, index_df)
            feats = compute_pattern_features(df)
            bars = {"open": df["Open"].to_numpy(float), "high": df["High"].to_numpy(float),
                    "low": df["Low"].to_numpy(float), "close": df["Close"].to_numpy(float),
                    "rsi": df[rsi_col].to_numpy(float)}
            sig, reg, score = sf["Signal"].to_numpy(), sf["regime"].to_numpy(), sf["Score"].to_numpy()
            atr, idx = df[atr_col].to_numpy(float), df.index
            next_allowed, rec = 250, 0
            for i in range(250, len(df) - 1):
                if i < next_allowed or sig[i] != "BUY" or not np.isfinite(atr[i]):
                    continue
                regime = str(reg[i])
                res = trade_model.simulate(bars, i + 1, bars["close"][i], float(atr[i]),
                                           Regime(regime).atr_sl_multiplier)
                if res is None:
                    continue
                cdl = str(feats["_candle"].iat[i])
                rsi_v = float(bars["rsi"][i])
                key = build_condition_key(
                    regime=regime, rsi_bucket=rsi_to_bucket(rsi_v), score=int(score[i]),
                    channel=str(feats["_channel"].iat[i]), candles=[cdl] if cdl != "NOCNDLE" else [],
                    near_support=bool(feats["_near_sup"].iat[i]), volume_spike=bool(feats["_vol_spike"].iat[i]),
                )
                rec += 1
                next_allowed = res["exit_idx"] + cooldown
                yield {"symbol": symbol, "date": idx[i].strftime("%Y-%m-%d"), "key": key, "regime": regime,
                       "score": int(score[i]), "rsi": rsi_v, "r": res["r"], "hold": res["hold"],
                       "exit_reason": res["exit_reason"]}
            logger.info("%s: simulated %d independent trades", symbol, rec)
        except Exception as exc:
            logger.warning("Simulation failed for %s: %s", symbol, exc)


def seed_from_history(symbols: Optional[List[str]] = None, journal: Optional[dict] = None) -> dict:
    """
    (Re)build the learning journal from ~10 years of history by simulating
    the exact live rules (see simulate_history / execution.trade_model).
    """
    journal = journal if journal is not None else _empty_journal()
    total = 0
    for t in simulate_history(symbols):
        record_observation(
            journal, t["key"], t["r"] > 0,
            {"symbol": t["symbol"], "regime": t["regime"], "rsi_bkt": rsi_to_bucket(t["rsi"]), "score": t["score"]},
            trade_date=t["date"], hold_time=t["hold"], r=t["r"],
        )
        total += 1
    journal["metadata"]["seeded_at"] = datetime.now().strftime("%Y-%m-%d %H:%M")
    journal["metadata"]["seeded_trades"] = total
    save_journal(journal)
    return journal


def refresh_journal_if_needed(max_age_days: int = 30) -> dict:
    """
    Rebuild the journal from history when it is empty, from an old schema,
    or older than `max_age_days` (so new market data keeps being learned),
    then replay live paper trades on top — live evidence is never lost.
    """
    journal = load_journal()
    seeded = journal.get("metadata", {}).get("seeded_at")
    stale = True
    if seeded and journal["metadata"].get("total_observations", 0) > 0:
        try:
            stale = (datetime.now() - datetime.strptime(seeded, "%Y-%m-%d %H:%M")).days > max_age_days
        except ValueError:
            stale = True
    if not stale:
        return journal

    live = [t for t in journal.get("closed_trades", []) if t.get("r_multiple") is not None]
    drift = journal.get("live_drift")
    logger.info("Re-seeding learning journal from history (%d live trades will be replayed)", len(live))
    fresh = seed_from_history(journal=_empty_journal())
    for t in live:
        key = t.get("pattern_key")
        if not key:
            continue
        record_observation(fresh, key, t["r_multiple"] > 0,
                           {"symbol": t.get("symbol"), "regime": t.get("regime"), "source": "live_paper"},
                           trade_date=t.get("exit_date"), hold_time=t.get("bars_held", 0),
                           source="live_paper", r=t["r_multiple"])
    fresh["closed_trades"] = live
    if drift:
        fresh["live_drift"] = drift
    save_journal(fresh)
    return fresh


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
    Record a closed paper trade. Uses the trade's R-multiple; also tracks
    live-vs-model drift (actual R vs the expected R at entry), the first
    thing a professional checks to know whether an edge is decaying.
    """
    journal = load_journal()
    trade = trade or {}

    regime   = signal_result.get("regime", "UNKNOWN")
    rsi      = float(signal_result.get("rsi", 50))
    score    = int(signal_result.get("score", 0))
    rsi_bkt  = rsi_to_bucket(rsi)
    key = signal_result.get("pattern_key")
    snap = signal_result.get("pattern_snapshot", {})

    if not key:
        feats = {"channel": "SIDEWAYS", "candles": [], "near_support": False, "volume_spike": False}
        if df is not None:
            try:
                from analytics.pattern_detector import latest_key_features
                feats = latest_key_features(df)
            except Exception:
                pass
        key = build_condition_key(regime=regime, rsi_bucket=rsi_bkt, score=score, **feats)

    r = trade.get("r_multiple")
    if r is None:
        r = 1.0 if won else -1.0
        logger.warning("Live trade %s has no r_multiple — recording ±1R", symbol)

    record_observation(
        journal, key, r > 0,
        {"symbol": symbol, "regime": regime, "rsi_bkt": rsi_bkt, "score": score, "source": "live_paper"},
        trade_date=trade.get("exit_date") or datetime.now().strftime("%Y-%m-%d"),
        hold_time=hold_time, source="live_paper", r=r,
    )

    exp_r = trade.get("expected_r")
    if exp_r is not None:
        d = journal.setdefault("live_drift", {"trades": 0, "sum_expected_r": 0.0, "sum_actual_r": 0.0})
        d["trades"] += 1
        d["sum_expected_r"] += float(exp_r)
        d["sum_actual_r"] += float(r)

    if trade:
        journal.setdefault("closed_trades", []).append({**trade, "pattern_key": key, "pattern_snapshot": snap})
        del journal["closed_trades"][:-1000]

    save_journal(journal)
    logger.info("Recorded live trade outcome for %s: %+.2fR", symbol, r)


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

    # Hierarchically shrunk edge for every pattern (sparse patterns borrow strength)
    for group in (proven, validated, learning, watching, unreliable):
        for e in group:
            est = get_edge_estimate(journal, e["key"])
            e["expectancy_blended"] = est["expected_r"]
            e["win_rate_blended"] = est["win_prob"]
            e["lower_r"] = est["lower_r"]

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
        "global":             journal.get("global", {}),
        "live_drift":         _drift_summary(journal),
    }


def _drift_summary(journal: dict) -> dict:
    d = journal.get("live_drift", {}) or {}
    n = d.get("trades", 0)
    if not n:
        return {"trades": 0}
    exp_avg = d["sum_expected_r"] / n
    act_avg = d["sum_actual_r"] / n
    return {"trades": n, "avg_expected_r": round(exp_avg, 3), "avg_actual_r": round(act_avg, 3),
            "drift_r": round(act_avg - exp_avg, 3)}
