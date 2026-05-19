"""
Probabilistic Intelligence Cycles
Coordinates the 8:30 AM Pre-Market preparation scan and the 3:15 PM EOD post-market analysis.
"""

import os
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import numpy as np
import pandas as pd
import yfinance as yf

from indicators.regime import Regime, detect_regime_detailed, regime_summary
from data.fetcher import fetch_stock_data, load_stock_data, fetch_market_trend
from config.settings import DATA_CONFIG, RISK_CONFIG

logger = logging.getLogger(__name__)

PREDICTION_HISTORY_FILE = os.path.join("state", "prediction_history.json")

# Standard Sector Groupings for Watchlist
SYMBOL_SECTORS = {
    "RELIANCE.NS": "Energy/Infra",
    "TCS.NS": "IT",
    "INFY.NS": "IT",
    "HDFCBANK.NS": "Banking",
    "ICICIBANK.NS": "Banking",
    "SBIN.NS": "Banking",
    "BHARTIARTL.NS": "Energy/Infra",
    "ITC.NS": "FMCG",
    "HINDUNILVR.NS": "FMCG",
    "LTIM.NS": "IT",
    "MARUTI.NS": "Auto",
    "TATAMOTORS.NS": "Auto",
    "TATASTEEL.NS": "Metals",
    "JIOFIN.NS": "Banking",
}


# ── Global Market Snapshot Tickers ─────────────────────────────────────────────
GLOBAL_TICKERS = {
    "US (S&P 500)": "^GSPC",
    "US (Nasdaq)": "^IXIC",
    "VIX (Volatility)": "^VIX",
    "Crude Oil": "CL=F",
    "Dollar Index": "DX-Y.NYB",
    "Bond Yields (10Y)": "^TNX",
    "Japan (Nikkei 225)": "^N225"
}


# ── Pre-Market Cycle (8:30 AM IST) ─────────────────────────────────────────────

def run_pre_market_cycle(symbols: List[str], capital: float, active_trades: int) -> dict:
    """
    Executes the morning 8:30 AM IST intelligence preparation routine.
    """
    logger.info("Executing 8:30 AM IST Pre-Market Intelligence Cycle...")

    # 1. Global Market Snapshot
    snapshot = []
    bullish_factors = 0
    bearish_factors = 0
    vix_val = 15.0
    dxy_val = 101.0
    
    for name, ticker in GLOBAL_TICKERS.items():
        try:
            t = yf.Ticker(ticker)
            df = t.history(period="2d", interval="1d", auto_adjust=True)
            if not df.empty and len(df) >= 2:
                prev_close = df["Close"].iloc[-2]
                curr_close = df["Close"].iloc[-1]
                pct_change = ((curr_close - prev_close) / prev_close) * 100
                
                # Capture major metrics
                if ticker == "^VIX": vix_val = curr_close
                if ticker == "DX-Y.NYB": dxy_val = curr_close
                
                # Classify impact
                impact = "NEUTRAL"
                impact_text = "No major impact"
                
                if ticker in ("^GSPC", "^IXIC", "^N225"):
                    if pct_change < -0.5:
                        impact = "BEARISH"
                        impact_text = "bearish pressure"
                        bearish_factors += 1.5
                    elif pct_change > 0.5:
                        impact = "BULLISH"
                        impact_text = "bullish support"
                        bullish_factors += 1.5
                elif ticker == "^VIX":
                    if curr_close > 20.0:
                        impact = "BEARISH"
                        impact_text = "volatility expansion"
                        bearish_factors += 2.0
                    elif curr_close < 15.0:
                        impact = "BULLISH"
                        impact_text = "volatility compression"
                        bullish_factors += 1.0
                elif ticker == "DX-Y.NYB":
                    if curr_close > 103.0:
                        impact = "BEARISH"
                        impact_text = "pressure on emerging markets"
                        bearish_factors += 1.5
                    elif curr_close < 100.0:
                        impact = "BULLISH"
                        impact_text = "supportive currency flow"
                        bullish_factors += 1.0
                elif ticker == "CL=F":
                    if pct_change > 2.0:
                        impact = "BEARISH"
                        impact_text = "crude oil surge / inflation risk"
                        bearish_factors += 1.0
                    elif pct_change < -2.0:
                        impact = "BULLISH"
                        impact_text = "softening crude oil / supportive"
                        bullish_factors += 0.8
                
                snapshot.append({
                    "name": name,
                    "ticker": ticker,
                    "value": round(curr_close, 2),
                    "change": round(pct_change, 2),
                    "impact": impact,
                    "impact_text": impact_text
                })
        except Exception as e:
            logger.warning("Failed to fetch %s: %s", name, e)

    # Compute opening bias & confidence
    total_weights = bullish_factors + bearish_factors
    if total_weights > 0:
        bullish_pct = (bullish_factors / total_weights) * 100
    else:
        bullish_pct = 50.0
        
    confidence = int(max(bullish_pct, 100 - bullish_pct))
    
    if bullish_pct > 65:
        opening_bias = "BULLISH"
        gap_prob = "High Gap Up Probability"
        risk_level = "LOW"
    elif bullish_pct > 52:
        opening_bias = "WEAK BULLISH"
        gap_prob = "Mild Gap Up / Flat"
        risk_level = "MEDIUM"
    elif bullish_pct < 35:
        opening_bias = "BEARISH"
        gap_prob = "High Gap Down Probability"
        risk_level = "HIGH"
    elif bullish_pct < 48:
        opening_bias = "WEAK BEARISH"
        gap_prob = "Mild Gap Down / Flat"
        risk_level = "HIGH"
    else:
        opening_bias = "NEUTRAL"
        gap_prob = "Flat Opening"
        risk_level = "MEDIUM"

    if vix_val > 22.0:
        risk_level = "HIGH"
        
    # 2. Sector Strength Map
    sector_perf = {}
    for sym in symbols:
        try:
            df = load_stock_data(sym)
            if not df.empty and len(df) >= 6:
                ret_5d = ((df["Close"].iloc[-1] - df["Close"].iloc[-6]) / df["Close"].iloc[-6]) * 100
                sec = SYMBOL_SECTORS.get(sym, "Others")
                if sec not in sector_perf:
                    sector_perf[sec] = []
                sector_perf[sec].append(ret_5d)
        except Exception:
            continue

    sector_rankings = []
    for sec, rets in sector_perf.items():
        avg_ret = float(np.mean(rets))
        sector_rankings.append({
            "sector": sec,
            "strength": round(avg_ret, 2),
            "status": "STRONG" if avg_ret > 0.5 else ("WEAK" if avg_ret < -0.5 else "NEUTRAL")
        })
    sector_rankings.sort(key=lambda x: x["strength"], reverse=True)

    # 3. Watchlist Quality Ranking
    from signals.generator import generate_signal_for_symbol
    market_up = fetch_market_trend()
    
    # Detailed 7-regime detection
    try:
        nifty_df = fetch_stock_data(DATA_CONFIG["market_index"], save=True)
        regime_result = detect_regime_detailed(nifty_df)
        market_regime = regime_result.regime
    except Exception:
        market_regime = Regime.SIDEWAYS

    watchlist_scans = []
    for sym in symbols:
        try:
            res = generate_signal_for_symbol(
                sym,
                capital=capital,
                active_trades=active_trades,
                market_up=market_up,
                market_regime=market_regime
            )
            watchlist_scans.append(res)
        except Exception as e:
            logger.warning("Watchlist scan failed for %s: %s", sym, e)

    # Sort watchlist by Quality Score
    watchlist_scans.sort(key=lambda x: x.get("confidence_score", 0), reverse=True)

    # 4. Do Not Trade Warnings
    warnings = []
    if vix_val > 24.0:
        warnings.append(f"⚠️ Extreme Volatility: VIX is {vix_val:.1f} (reduce capital exposure & position sizes immediately)")
    if dxy_val > 104.0:
        warnings.append(f"⚠️ Strong US Dollar: DXY is {dxy_val:.1f} (heavy pressure on emerging markets, avoid aggressive longs)")
    
    # Calculate watch list average pullback
    pullbacks = [r.get("pullback_pct", 0.0) for r in watchlist_scans if "pullback_pct" in r]
    if pullbacks and np.mean(pullbacks) < 0.015:
        warnings.append("⚠️ Shallow Watchlist Dips: Average watchlist pullback is very low (risk of buying at short-term peaks)")

    return {
        "cycle": "PRE-MARKET",
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "global_snapshot": snapshot,
        "opening_bias": opening_bias,
        "confidence_pct": confidence,
        "gap_probability": gap_prob,
        "risk_level": risk_level,
        "market_regime": market_regime,
        "sector_rankings": sector_rankings,
        "watchlist": watchlist_scans,
        "warnings": warnings
    }


# ── End-of-Day Cycle (3:15 PM IST) ─────────────────────────────────────────────

def run_eod_cycle(symbols: List[str], pre_market_predictions: Optional[dict] = None) -> dict:
    """
    Executes the afternoon 3:15 PM IST closing intelligence routine.
    """
    logger.info("Executing 3:15 PM IST End-of-Day Intelligence Cycle...")
    
    # 1. Closing Structure Analysis (Nifty 50 index)
    close_structure = "RANGE-BOUND"
    structure_desc = "Price trading in a narrow range."
    nifty_chg = 0.0
    
    try:
        df = fetch_stock_data(DATA_CONFIG["market_index"], save=True)
        if not df.empty and len(df) >= 1:
            o, h, l, c = df["Open"].iloc[-1], df["High"].iloc[-1], df["Low"].iloc[-1], df["Close"].iloc[-1]
            nifty_chg = ((c - df["Close"].iloc[-2]) / df["Close"].iloc[-2]) * 100 if len(df) >= 2 else 0.0
            
            rng = h - l
            pct_in_range = (c - l) / rng if rng > 0 else 0.5
            
            if pct_in_range >= 0.90:
                close_structure = "STRONG CLOSE"
                structure_desc = "Institutional buying likely continued aggressively into the market close."
            elif pct_in_range <= 0.10:
                close_structure = "WEAK CLOSE"
                structure_desc = "Heavy institutional distribution occurred, pulling the index to daily lows."
            elif h - max(o, c) > abs(o - c) * 2 and pct_in_range < 0.40:
                close_structure = "FAILED RALLY"
                structure_desc = "Late-day selloff completely erased early gains, signaling distribution."
            elif min(o, c) - l > abs(o - c) * 2 and pct_in_range > 0.60:
                close_structure = "ACCUMULATION"
                structure_desc = "Strong intra-day rebound; buyers absorbed panic selling at lows."
    except Exception as e:
        logger.warning("Nifty EOD analysis failed: %s", e)

    # 2. Watchlist Breadth Analysis
    advances = 0
    declines = 0
    breakout_failures = []
    unusual_volume = []
    
    for sym in symbols:
        try:
            df = load_stock_data(sym)
            if not df.empty and len(df) >= 2:
                prev_c = df["Close"].iloc[-2]
                curr_c = df["Close"].iloc[-1]
                daily_pct = ((curr_c - prev_c) / prev_c) * 100
                
                if daily_pct > 0: advances += 1
                else: declines += 1
                
                # Check unusual volume (vol > 2.0x of 20-day MA)
                vol = df["Volume"].iloc[-1]
                vol_ma = df["Volume"].rolling(window=20).mean().iloc[-1]
                if vol_ma > 0 and vol / vol_ma >= 2.0:
                    unusual_volume.append({
                        "symbol": sym,
                        "ratio": round(vol / vol_ma, 2),
                        "change": round(daily_pct, 2)
                    })
                    
                # Breakout failure: high volume spike but closed in bottom 30% of day's range
                rng = df["High"].iloc[-1] - df["Low"].iloc[-1]
                pct_range = (curr_c - df["Low"].iloc[-1]) / rng if rng > 0 else 0.5
                if vol_ma > 0 and vol / vol_ma >= 1.5 and pct_range <= 0.30 and daily_pct > -1.0:
                    breakout_failures.append(sym)
        except Exception:
            continue

    total_watchlist = advances + declines
    adv_dec_ratio = advances / declines if declines > 0 else advances
    
    # ── DYNAMIC SECTOR LEADER/LAGGARD CALCULATION (CRITICAL ISSUE #1) ──────────
    sector_perf = {}
    for sym in symbols:
        try:
            df = load_stock_data(sym)
            if not df.empty and len(df) >= 6:
                ret_5d = ((df["Close"].iloc[-1] - df["Close"].iloc[-6]) / df["Close"].iloc[-6]) * 100
                sec = SYMBOL_SECTORS.get(sym, "Others")
                if sec not in sector_perf:
                    sector_perf[sec] = []
                sector_perf[sec].append(ret_5d)
        except Exception:
            continue
            
    sector_averages = []
    for sec, rets in sector_perf.items():
        if rets:
            sector_averages.append((sec, float(np.mean(rets))))
    
    sector_averages.sort(key=lambda x: x[1], reverse=True)
    sector_leaders = [x[0] for x in sector_averages[:2]] if len(sector_averages) >= 2 else ["Others"]
    sector_laggards = [x[0] for x in sector_averages[-2:]] if len(sector_averages) >= 2 else ["Others"]

    # Load persistent market regime (CRITICAL ISSUE #5)
    try:
        nifty_df = fetch_stock_data(DATA_CONFIG["market_index"], save=False)
        regime_result = detect_regime_detailed(nifty_df)
        from indicators.regime import apply_regime_hysteresis
        market_regime = apply_regime_hysteresis(regime_result.regime)
    except Exception:
        market_regime = Regime.SIDEWAYS

    # ── ENSEMBLE PREDICTION MODEL (CRITICAL ISSUE #3) ───────────────────────────
    # Submodel 1: Breadth (participation)
    breadth_score = (advances / total_watchlist * 100) if total_watchlist > 0 else 50.0
    b_bull = breadth_score
    b_bear = 100.0 - breadth_score
    
    # Submodel 2: Momentum (directional force)
    m_bull = 33.3
    m_bear = 33.3
    if close_structure == "STRONG CLOSE":
        m_bull += 30.0; m_bear -= 20.0
    elif close_structure == "ACCUMULATION":
        m_bull += 15.0; m_bear -= 10.0
    elif close_structure == "WEAK CLOSE":
        m_bear += 30.0; m_bull -= 20.0
    elif close_structure == "FAILED RALLY":
        m_bear += 20.0; m_bull -= 15.0
        
    if nifty_chg > 0.5:
        m_bull += 15.0; m_bear -= 10.0
    elif nifty_chg < -0.5:
        m_bear += 15.0; m_bull -= 10.0

    # Submodel 3: Volatility (noise and uncertainty)
    vix_val = 15.0
    try:
        vix_t = yf.Ticker("^VIX")
        vix_df = vix_t.history(period="1d", auto_adjust=True)
        if not vix_df.empty:
            vix_val = float(vix_df["Close"].iloc[-1])
    except Exception:
        pass
        
    vol_bull = 33.3
    vol_bear = 33.3
    if vix_val > 20.0:
        vol_bear += 10.0; vol_bull -= 25.0
    elif vix_val < 14.0:
        vol_bull += 15.0; vol_bear -= 5.0

    # Submodel 4: Sector Rotation (defensive vs aggressive lead)
    agg_lead = 0.0
    def_lead = 0.0
    for sec, val in sector_averages:
        if sec in ("Banking", "Auto", "Metals"):
            agg_lead += val
        elif sec in ("FMCG", "IT", "Energy/Infra"):
            def_lead += val
            
    rot_bull = 33.3
    rot_bear = 33.3
    if agg_lead > def_lead + 0.5:
        rot_bull += 15.0; rot_bear -= 10.0
    elif def_lead > agg_lead + 0.5:
        rot_bear += 10.0; rot_bull -= 20.0

    # Submodel 5: Market Regime
    reg_bull = 33.3
    reg_bear = 33.3
    if market_regime == Regime.STRONG_TREND_UP:
        reg_bull += 30.0; reg_bear -= 20.0
    elif market_regime == Regime.WEAK_TREND_UP:
        reg_bull += 15.0; reg_bear -= 10.0
    elif market_regime == Regime.STRONG_TREND_DOWN:
        reg_bear += 30.0; reg_bull -= 20.0
    elif market_regime == Regime.WEAK_TREND_DOWN:
        reg_bear += 15.0; reg_bull -= 10.0
    elif market_regime == Regime.SIDEWAYS:
        reg_bull -= 15.0; reg_bear -= 10.0

    # Ensemble Weighted Aggregation
    w_breadth = 0.20
    w_momentum = 0.25
    w_vol = 0.15
    w_rot = 0.15
    w_reg = 0.25
    
    final_bull = (b_bull * w_breadth) + (m_bull * w_momentum) + (vol_bull * w_vol) + (rot_bull * w_rot) + (reg_bull * w_reg)
    final_bear = (b_bear * w_breadth) + (m_bear * w_momentum) + (vol_bear * w_vol) + (rot_bear * w_rot) + (reg_bear * w_reg)
    
    final_bull = min(max(final_bull, 5.0), 90.0)
    final_bear = min(max(final_bear, 5.0), 90.0)
    final_range = 100.0 - (final_bull + final_bear)

    # Core forecast label & narrative description (CRITICAL ISSUE #3)
    if final_bull > 55.0:
        forecast = "Bullish Continuation Likelihood Elevated"
        desc = "Technical metrics indicate positive upward momentum has a high probability of follow-through, though opening prints remain subject to global indices."
    elif final_bear > 55.0:
        forecast = "Bearish Continuation Likelihood Elevated"
        desc = "Elevated distribution structures point to soft closing momentum, though morning opening gaps may be altered by global asset changes."
    elif final_bull > 42.0:
        forecast = "Neutral to Bullish Reversal Bias"
        desc = "Subtle buyer absorption suggests a mild upward bias, though overall directional commitment is relatively low."
    elif final_bear > 42.0:
        forecast = "Bearish to Neutral Distribution Bias"
        desc = "Mild selling pressure indicates a softer tone, likely consolidating unless new institutional support materializes."
    else:
        forecast = "Range-Bound Consolidation Likelihood Elevated"
        desc = "Equally balanced internal breadth vectors suggest higher probability of a standard range-bound consolidation day."

    prediction_report = {
        "date": datetime.now().strftime("%Y-%m-%d"),
        "forecast": forecast,
        "narrative": desc,
        "probs": {
            "BULLISH": round(final_bull, 1),
            "RANGE": round(final_range, 1),
            "BEARISH": round(final_bear, 1)
        }
    }

    # ── DEDICATED MARKET STATE STORE (CRITICAL ISSUE #1) ───────────────────────
    # Determine general volatility state
    if vix_val > 22.0:
        vol_state = "HIGH_VOLATILITY"
        risk_level = "HIGH"
    elif vix_val < 14.0:
        vol_state = "LOW_VOLATILITY"
        risk_level = "LOW"
    else:
        vol_state = "NORMAL"
        risk_level = "MEDIUM"

    if market_regime in (Regime.STRONG_TREND_DOWN, Regime.VOLATILE):
        risk_level = "HIGH"

    market_state = {
        "date": datetime.now().strftime("%Y-%m-%d"),
        "market_regime": str(market_regime),
        "risk_level": risk_level,
        "breadth_score": int(breadth_score),
        "volatility_state": vol_state,
        "sector_leaders": sector_leaders,
        "sector_laggards": sector_laggards,
        "tomorrow_outlook": {
            "bearish": round(final_bear / 100.0, 2),
            "sideways": round(final_range / 100.0, 2),
            "bullish": round(final_bull / 100.0, 2)
        }
    }

    market_state_file = os.path.join("state", "market_state.json")
    try:
        os.makedirs(os.path.dirname(market_state_file), exist_ok=True)
        with open(market_state_file, "w") as f:
            json.dump(market_state, f, indent=2)
        logger.info("Centralized Market State successfully persisted: %s", market_state_file)
    except Exception as e:
        logger.error("Failed to persist Market State snapshot: %s", e)

    # 4. Historical Accuracy Tracker: score yesterday's forecast
    accuracy_log = score_yesterday_prediction(nifty_chg)

    # 5. Alert Memory System: persist today's market stress conditions
    try:
        update_alert_memory(market_state)
    except Exception as e:
        logger.warning("Alert memory update failed (non-critical): %s", e)

    return {
        "cycle": "EOD",
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "nifty_change": round(nifty_chg, 2),
        "close_structure": close_structure,
        "structure_desc": structure_desc,
        "advances": advances,
        "declines": declines,
        "adv_dec_ratio": round(adv_dec_ratio, 2),
        "unusual_volume": unusual_volume,
        "breakout_failures": breakout_failures,
        "prediction": prediction_report,
        "accuracy_tracking": accuracy_log
    }


# ── Prediction Scoring Engine ──────────────────────────────────────────────────

def score_yesterday_prediction(today_nifty_chg: float) -> dict:
    """
    Compares yesterday's forecast against today's Nifty actual price action to measure precision.
    Saves and returns cumulative tracker logs.
    """
    history = []
    if os.path.exists(PREDICTION_HISTORY_FILE):
        try:
            with open(PREDICTION_HISTORY_FILE, "r") as f:
                history = json.load(f)
        except Exception:
            history = []

    # Identify if yesterday has a pending score
    scored_record = None
    if history and "result" not in history[-1]:
        # Yesterday's prediction is at the tail
        yesterday = history[-1]
        forecast = yesterday.get("forecast", "NEUTRAL")
        
        # Scoring logic
        result = "FAILED"
        if today_nifty_chg > 0.15 and "BULLISH" in forecast:
            result = "SUCCESS"
        elif today_nifty_chg < -0.15 and "BEARISH" in forecast:
            result = "SUCCESS"
        elif -0.15 <= today_nifty_chg <= 0.15 and ("RANGE" in forecast or "NEUTRAL" in forecast):
            result = "SUCCESS"
            
        yesterday["actual_change"] = round(today_nifty_chg, 2)
        yesterday["result"] = result
        scored_record = yesterday
        
        logger.info("Yesterday's forecast [%s] scored against Nifty change [%.2f%%]: %s", forecast, today_nifty_chg, result)

    # Write tomorrow's prediction shell (will be updated when EOD forecast runs)
    # Return cumulative accuracy
    total_scored = sum(1 for h in history if "result" in h)
    successful = sum(1 for h in history if h.get("result") == "SUCCESS")
    accuracy_pct = (successful / total_scored * 100) if total_scored > 0 else 0.0

    return {
        "scored_yesterday": scored_record is not None,
        "yesterday_result": scored_record.get("result", "N/A") if scored_record else "N/A",
        "yesterday_forecast": scored_record.get("forecast", "N/A") if scored_record else "N/A",
        "actual_change": round(today_nifty_chg, 2),
        "total_predictions": total_scored,
        "successful_predictions": successful,
        "accuracy_pct": round(accuracy_pct, 2)
    }


def save_today_forecast(forecast_report: dict) -> None:
    """
    Saves today's forecast into history so it can be evaluated tomorrow.
    """
    history = []
    if os.path.exists(PREDICTION_HISTORY_FILE):
        try:
            with open(PREDICTION_HISTORY_FILE, "r") as f:
                history = json.load(f)
        except Exception:
            history = []

    # Avoid duplicate saves for the same date
    today_str = datetime.now().strftime("%Y-%m-%d")
    if history and history[-1].get("date") == today_str:
        # Update existing record
        history[-1] = forecast_report
    else:
        history.append(forecast_report)

    # Restrict to last 100 forecasts
    history = history[-100:]

    try:
        os.makedirs(os.path.dirname(PREDICTION_HISTORY_FILE), exist_ok=True)
        with open(PREDICTION_HISTORY_FILE, "w") as f:
            json.dump(history, f, indent=2)
    except Exception as e:
        logger.error("Failed to save forecast history: %s", e)


# ── Alert Memory System (Upgrade 3) ───────────────────────────────────────────

ALERT_MEMORY_FILE = os.path.join("state", "alert_memory.json")

# Alert type definitions: (condition_check_fn, threshold_label, resolve_check_fn)
_ALERT_TYPES = {
    "WEAK_BREADTH": {
        "label": "Narrow Market Breadth",
        "severity": "WARNING",
        "action": "Avoid index-momentum longs; focus on defensive leaders only.",
    },
    "CONSENSUS_SPLIT": {
        "label": "Model Consensus Split",
        "severity": "CRITICAL",
        "action": "Reduce active position sizing by 50% immediately.",
    },
    "HIGH_VOLATILITY": {
        "label": "Volatility Expansion Detected",
        "severity": "WARNING",
        "action": "Tighten hard stops on open positions; expect gap risk.",
    },
    "REGIME_DOWNTREND": {
        "label": "Macro Index in Downtrend",
        "severity": "CRITICAL",
        "action": "Long lock active — no new BUY signals until regime recovers.",
    },
}


def update_alert_memory(market_state: dict) -> None:
    """
    Reads the current market_state snapshot and appends/resolves alert events
    into state/alert_memory.json.

    Called at the end of every EOD cycle so the dashboard can show:
        "⚠️ Weak Breadth persisted 4 sessions — Escalating"
    rather than just a fresh point-in-time alert.
    """
    today = datetime.now().strftime("%Y-%m-%d")

    # Load existing memory
    memory = {"alerts": []}
    if os.path.exists(ALERT_MEMORY_FILE):
        try:
            with open(ALERT_MEMORY_FILE, "r") as f:
                memory = json.load(f)
        except Exception:
            memory = {"alerts": []}

    existing_alerts = memory.get("alerts", [])

    # --- Determine which alert types are active today ---
    breadth = market_state.get("breadth_score", 50)
    vol_state = market_state.get("volatility_state", "NORMAL")
    regime = market_state.get("market_regime", "SIDEWAYS")
    outlook = market_state.get("tomorrow_outlook", {})
    max_prob = max(outlook.values()) if outlook else 0.34
    consensus_pct = int(max_prob * 100)

    active_today = set()
    if breadth < 45:
        active_today.add("WEAK_BREADTH")
    if consensus_pct < 45:
        active_today.add("CONSENSUS_SPLIT")
    if vol_state in ("HIGH_VOLATILITY", "EXPANDING"):
        active_today.add("HIGH_VOLATILITY")
    if "DOWN" in regime:
        active_today.add("REGIME_DOWNTREND")

    # --- Resolve alerts that are no longer active ---
    for alert in existing_alerts:
        if not alert.get("resolved") and alert["type"] not in active_today:
            alert["resolved"] = True
            alert["resolved_date"] = today

    # --- Append new alert entries for today (one per type per day) ---
    today_types_logged = {a["type"] for a in existing_alerts if a.get("date") == today}
    for alert_type in active_today:
        if alert_type not in today_types_logged:
            meta = _ALERT_TYPES.get(alert_type, {"label": alert_type, "severity": "WARNING", "action": ""})
            existing_alerts.append({
                "date": today,
                "type": alert_type,
                "label": meta["label"],
                "severity": meta["severity"],
                "action": meta["action"],
                "value": {
                    "breadth": breadth,
                    "consensus_pct": consensus_pct,
                    "volatility": vol_state,
                    "regime": regime,
                },
                "resolved": False,
                "resolved_date": None,
            })

    # Keep only the last 90 days of alert records
    cutoff = (datetime.now() - timedelta(days=90)).strftime("%Y-%m-%d")
    existing_alerts = [a for a in existing_alerts if a["date"] >= cutoff]

    memory["alerts"] = existing_alerts
    memory["last_updated"] = today

    try:
        os.makedirs(os.path.dirname(ALERT_MEMORY_FILE), exist_ok=True)
        with open(ALERT_MEMORY_FILE, "w") as f:
            json.dump(memory, f, indent=2)
        logger.info("Alert memory updated: %d active alert types today", len(active_today))
    except Exception as e:
        logger.error("Failed to persist alert memory: %s", e)


def get_alert_persistence_summary(memory: dict) -> list:
    """
    Reads alert_memory.json and returns a list of dicts describing
    each currently-active alert with its persistence count.

    Returns:
        [{"type": str, "label": str, "severity": str, "action": str,
          "days_active": int, "trend": "ESCALATING"|"STABLE"|"NEW"}]
    """
    alerts = memory.get("alerts", [])
    unresolved = [a for a in alerts if not a.get("resolved")]

    # Group by type and count consecutive days
    from collections import defaultdict
    by_type: dict = defaultdict(list)
    for a in unresolved:
        by_type[a["type"]].append(a["date"])

    result = []
    for alert_type, dates in by_type.items():
        days_active = len(set(dates))
        # Determine escalation trend
        if days_active >= 4:
            trend = "ESCALATING 🔴"
        elif days_active >= 2:
            trend = "PERSISTING ⚠️"
        else:
            trend = "NEW ℹ️"

        meta = _ALERT_TYPES.get(alert_type, {"label": alert_type, "severity": "WARNING", "action": ""})
        result.append({
            "type": alert_type,
            "label": meta["label"],
            "severity": meta["severity"],
            "action": meta["action"],
            "days_active": days_active,
            "trend": trend,
        })

    # Sort: CRITICAL first, then by days_active desc
    result.sort(key=lambda x: (0 if x["severity"] == "CRITICAL" else 1, -x["days_active"]))
    return result

