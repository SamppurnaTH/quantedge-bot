"""
Daily Paper Trading & Market Intelligence Runner

Determines execution mode dynamically based on the current time (IST):
  - Morning Mode (before 12:00 PM IST): Pre-Market Intelligence & Order Execution Cycle.
  - Afternoon Mode (12:00 PM IST or later): End-of-Day Analysis & Predictive Outlook Cycle.
"""

import sys
import os
import json
import logging
from datetime import datetime, timezone

try:
    from zoneinfo import ZoneInfo
except ImportError:
    ZoneInfo = None

# Force UTF-8 encoding for stdout/stderr to handle emojis on all platforms
if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')
if sys.stderr.encoding != 'utf-8':
    sys.stderr.reconfigure(encoding='utf-8')


def get_ist_now() -> datetime:
    """Return the current time in India Standard Time (IST)."""
    if ZoneInfo is not None:
        return datetime.now(timezone.utc).astimezone(ZoneInfo("Asia/Kolkata"))
    return datetime.now()

sys.path.insert(0, os.path.dirname(__file__))

from logger.env_loader import load_env
load_env()

from logger.setup import setup_logging
from indicators.regime import Regime, regime_summary
from execution.paper_trader import PaperTrader
from notifications.telegram import TelegramNotifier
from analytics.dashboard import run_auto_optimization
from analytics.learning_engine import load_journal, seed_from_history
from analytics.learning_report import run_learning_report
from config.settings import DATA_CONFIG, RISK_CONFIG
from data.fetcher import load_stock_data
from analytics.intelligence_cycles import (
    run_pre_market_cycle,
    run_eod_cycle,
    save_today_forecast
)


def save_markdown_report(filepath: str, content: str) -> None:
    """Helper to save generated markdown reports to disk."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(content)
    print(f"  📝 Saved markdown report: {filepath}")


def save_json_state(filepath: str, data: dict) -> None:
    """Helper to persist JSON state files for dashboard integration."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, default=str)


def _regime_execution_summary(regime: Regime) -> tuple:
    """Return (mode_label, guideline_text, size_pct) for a given regime."""
    sm = regime.position_size_multiplier
    if sm >= 1.0:
        return (
            "🚀 FULL EXPOSURE (×1.0 size)",
            "Optimal trending environment — capital sizing is maximised for standard portfolio weights.",
            100,
        )
    elif regime == Regime.VOLATILE:
        return (
            "⚡ VOLATILE MODE — HALF SIZE, WIDER STOPS (×0.5)",
            "Elevated volatility detected — stops widened to ×2.5 ATR, position size throttled by 50%.",
            50,
        )
    elif sm > 0:
        return (
            "⚠️ REDUCED EXPOSURE — HALF SIZE (×0.5)",
            "High-uncertainty or range-bound environment — sizing throttled by 50% to prevent churn.",
            50,
        )
    else:
        return (
            "🛑 CAPITAL PRESERVATION — NO NEW LONGS (×0.0)",
            "Confirmed index downtrend — strategy blocks all new long entries to protect portfolio capital.",
            0,
        )


ARCHETYPE_LABELS = {
    "PANIC_EXHAUSTION": "Panic Exhaustion",
    "FORCED_MOMENTUM": "Forced Momentum",
    "VOLATILITY_COMPRESSION": "Volatility Compression",
    "ROTATIONAL_STRENGTH": "Rotational Strength",
    "FAILED_BREAKOUT": "Failed Breakout",
    "LIQUIDITY_VACUUM": "Liquidity Vacuum",
    "UNKNOWN_NOISE": "Noise / Unclassified"
}


def run_daily():
    setup_logging()

    local_now = datetime.now()
    ist_now = get_ist_now()
    logger = logging.getLogger(__name__)
    logger.info(
        "Daily runner start | local=%s | IST=%s",
        local_now.strftime("%Y-%m-%d %H:%M:%S"),
        ist_now.strftime("%Y-%m-%d %H:%M:%S %Z"),
    )

    now = ist_now
    # Before 12:00 PM IST is Morning Pre-Market Scan. 12:00 PM IST or after is EOD.
    is_afternoon = now.hour >= 12

    # Initialize Notifier & Trader
    notifier = TelegramNotifier()
    if not notifier.enabled:
        logger.warning(
            "Telegram notifications disabled. Verify TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID in the scheduled task environment."
        )
    trader = PaperTrader(capital=RISK_CONFIG["default_capital"])

    if not is_afternoon:
        # =======================================================================
        # 🌅 MORNING MODE: PRE-MARKET INTELLIGENCE COCKPIT
        # =======================================================================
        report_title = "PRE-MARKET INTELLIGENCE COCKPIT"
        print(f"\n{'='*70}")
        print(f"  {report_title} — {now.strftime('%A, %d %b %Y  %H:%M %Z')}")
        print(f"{'='*70}\n")

        # 1. Run Auto-Optimization
        print("  ⚙️  Running auto-optimization from paper trade history...")
        opt_rules = run_auto_optimization()

        # 2. Seed and Generate Learning Report
        journal = load_journal()
        if journal.get("metadata", {}).get("total_observations", 0) == 0:
            print("  🌱 Seeding learning journal from 10-year history (first run)...")
            journal = seed_from_history()

        # Update confidence calibration file dynamically
        from analytics.confidence import update_confidence_calibration
        update_confidence_calibration(journal)

        print("  📚 Generating daily learning report...")
        full_report, tg_summary = run_learning_report(journal)

        # 3. Run Pre-Market Intelligence Cycle
        print("  🌍 Fetching global market snapshot & calculating bias...")
        pm_report = run_pre_market_cycle(
            symbols=DATA_CONFIG["symbols"],
            capital=RISK_CONFIG["default_capital"],
            active_trades=len(trader.positions)
        )

        # ── Derive key metrics for report ─────────────────────────────────────
        regime: Regime = pm_report["market_regime"]
        exec_mode, exec_guideline, size_pct = _regime_execution_summary(regime)
        min_score = regime.min_score_to_buy
        atr_mult  = regime.atr_sl_multiplier

        # Top BUY opportunities (max 5, sorted by rank_key)
        buys = [x for x in pm_report["watchlist"] if x.get("signal") == "BUY" and "error" not in x]
        top5 = sorted(buys, key=lambda x: x.get("rank_key", 0), reverse=True)[:5]
        best_idea = top5[0] if top5 else None

        # ── Section 1: MARKET OPENING STATUS ──────────────────────────────────
        print(f"\n  ┌─ 🌍 MARKET OPENING STATUS {'─'*41}┐")
        print(f"  │  Regime      : {regime_summary(regime)}")
        print(f"  │  Risk Level  : {pm_report['risk_level']}")
        print(f"  │  Opening Bias: {pm_report['opening_bias']}  (confidence {pm_report['confidence_pct']}%)")
        print(f"  │  Gap Outlook : {pm_report['gap_probability']}")

        # 1-line interpretation
        bias_txt = pm_report['opening_bias'].upper()
        if "BULLISH" in bias_txt:
            interp = "Positive global cues support an upside open — watch for early continuation setups."
        elif "BEARISH" in bias_txt:
            interp = "Risk-off global cues — protect open positions, avoid chasing early momentum."
        else:
            interp = "Mixed global signals — standard mean-reversion rules apply today."
        print(f"  │  Outlook     : {interp}")
        print(f"  └{'─'*68}┘")

        # ── Section 2: CAPITAL PROTECTION ALERTS ──────────────────────────────
        print(f"\n  ┌─ 🚫 CAPITAL PROTECTION ALERTS {'─'*37}┐")
        if pm_report["warnings"]:
            for w in pm_report["warnings"]:
                print(f"  │  ⚠️  {w}")
        else:
            print("  │  ✅ No critical risk warnings detected for this session.")
        print(f"  └{'─'*68}┘")

        # ── Section 3: TODAY'S BEST OPPORTUNITIES ─────────────────────────────
        print(f"\n  ┌─ 🎯 TODAY'S BEST OPPORTUNITIES {'─'*36}┐")
        if top5:
            print(f"  │  {'Stock':<12} {'Bias':<6} {'Q-Score':>8} {'Exp(R)':>8} {'Win%':>6}  {'Risk'}")
            print(f"  │  {'─'*66}")
            for sym in top5:
                rr    = sym.get("risk_report", {})
                sl    = rr.get("stop_loss", 0)
                tp    = rr.get("take_profit", 0)
                close = sym.get("close", 0)
                risk_str = f"SL {sl:.0f} / TP {tp:.0f}" if sl and tp else "—"
                exp_r    = sym.get("expectancy_r", 0.0)
                cal_p    = sym.get("calibrated_prob", 50.0)
                exp_sign = "+" if exp_r >= 0 else ""
                print(
                    f"  │  {sym['symbol']:<12} {'BUY':<6} {sym.get('confidence_score', 0):>7}/100"
                    f" {exp_sign}{exp_r:>6.2f}R {cal_p:>5.1f}%  {risk_str}"
                )
        else:
            print("  │  ❌ No approved BUY setups in today's watchlist.")
            print("  │     This is expected in low-regime / capital-preservation modes.")
        print(f"  └{'─'*68}┘")

        # ── Section 4: WHY THIS STOCK? ─────────────────────────────────────────
        if top5:
            print(f"\n  ┌─ 🧠 WHY THESE SETUPS? {'─'*46}┐")
            for sym in top5[:3]:
                print(f"  │  {sym['symbol']} (Q-Score {sym.get('confidence_score',0)}/100):")
                for pro in sym.get("pros", [])[:2]:
                    print(f"  │    ✅ {pro}")
                for con in sym.get("cons", [])[:1]:
                    print(f"  │    ⚠️  {con}")
                print("  │")
            print(f"  └{'─'*68}┘")

        # ── Section 5: MARKET INTERNALS ────────────────────────────────────────
        print(f"\n  ┌─ 📊 MARKET INTERNALS {'─'*47}┐")
        print(f"  │  {'Asset':<22} {'Value':>10}  {'Change':>8}  Impact")
        print(f"  │  {'─'*60}")
        for s in pm_report.get("global_snapshot", [])[:6]:
            sign = "+" if s["change"] >= 0 else ""
            impact_icon = "🟢" if s["impact"] == "BULLISH" else ("🔴" if s["impact"] == "BEARISH" else "🟡")
            print(
                f"  │  {s['name']:<22} {str(s['value']):>10}  {sign}{s['change']:>6.2f}%  {impact_icon} {s['impact_text']}"
            )
        print(f"  └{'─'*68}┘")

        # ── Section 6: TODAY'S EXECUTION MODE ─────────────────────────────────
        print(f"\n  ┌─ 🚦 TODAY'S EXECUTION MODE {'─'*41}┐")
        print(f"  │  Mode      : {exec_mode}")
        print(f"  │  Guideline : {exec_guideline}")
        print(f"  │  Min Score : {min_score}/3 required | Stop Width: {atr_mult:.1f}x ATR | Size: {size_pct}%")
        print(f"  └{'─'*68}┘")

        # ── Section 7: BEST SINGLE IDEA OF THE DAY ────────────────────────────
        if best_idea:
            rr      = best_idea.get("risk_report", {})
            exp_r   = best_idea.get("expectancy_r", 0.0)
            cal_p   = best_idea.get("calibrated_prob", 50.0)
            exp_sign = "+" if exp_r >= 0 else ""
            print(f"\n  ┌─ 🚀 BEST SINGLE IDEA OF THE DAY {'─'*35}┐")
            print(f"  │  Symbol   : {best_idea['symbol']}")
            print(f"  │  Q-Score  : {best_idea.get('confidence_score', 0)}/100  |  Win Probability: {cal_p:.1f}%  |  Expectancy: {exp_sign}{exp_r:.2f}R")
            print(f"  │  Entry    : ₹{best_idea.get('close', 0):.2f}  |  RSI: {best_idea.get('rsi', 0):.1f}")
            if rr.get("approved"):
                print(f"  │  Stop Loss: ₹{rr['stop_loss']:.2f}  |  Target: ₹{rr['take_profit']:.2f}  |  R/R: {rr.get('risk_reward', 0):.1f}x")
                print(f"  │  Trade Cost: ₹{rr.get('trade_cost', 0):,.0f}  |  Shares: {rr.get('shares', 0)}")
            if best_idea.get("pros"):
                print(f"  │  Edge     : {best_idea['pros'][0]}")
            print(f"  └{'─'*68}┘")
        else:
            print(f"\n  ┌─ 🚀 BEST SINGLE IDEA OF THE DAY {'─'*35}┐")
            print("  │  No approved trade ideas today — system is in capital protection mode.")
            print(f"  └{'─'*68}┘")

        # ── Build Premium Pre-Market Markdown Report ───────────────────────────
        pm_md = f"""# 🌅 QuantEdge Intelligence Cockpit — Pre-Market Briefing
_Generated: {pm_report['timestamp']}_

---

## 🌍 Section 1 — Market Opening Status
| Metric | Value |
| :--- | :--- |
| **Index Regime** | {regime_summary(regime)} |
| **Risk Level** | {pm_report['risk_level']} |
| **Opening Bias** | {pm_report['opening_bias']} ({pm_report['confidence_pct']}% confidence) |
| **Gap Outlook** | {pm_report['gap_probability']} |
| **Interpretation** | *{interp}* |

---

## 🚫 Section 2 — Capital Protection Alerts
"""
        if pm_report["warnings"]:
            for w in pm_report["warnings"]:
                pm_md += f"- ⚠️ {w}\n"
        else:
            pm_md += "_No critical risk warnings detected for this session._\n"

        pm_md += f"""
---

## 🎯 Section 3 — Today's Best Opportunities (Top 5 BUY Setups)
| Symbol | Q-Score | Signal | Expectancy (R) | Win% | Entry | Stop | Target |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
"""
        for sym in top5:
            rr_s  = sym.get("risk_report", {})
            sl_s  = rr_s.get("stop_loss", 0)
            tp_s  = rr_s.get("take_profit", 0)
            exp_r_s = sym.get("expectancy_r", 0.0)
            cal_p_s = sym.get("calibrated_prob", 50.0)
            exp_sign_s = "+" if exp_r_s >= 0 else ""
            pm_md += (
                f"| **{sym['symbol']}** | {sym.get('confidence_score',0)}/100 | BUY"
                f" | {exp_sign_s}{exp_r_s:.2f}R | {cal_p_s:.1f}% "
                f"| ₹{sym.get('close',0):.2f} | ₹{sl_s:.2f} | ₹{tp_s:.2f} |\n"
            )
        if not top5:
            pm_md += "| *No approved BUY setups today* | — | — | — | — | — | — | — |\n"

        pm_md += "\n---\n\n## 🧠 Section 4 — Setup Reasoning (Pros & Cons)\n"
        for sym in top5[:3]:
            pm_md += f"\n### {sym['symbol']} — Q-Score {sym.get('confidence_score',0)}/100\n"
            pm_md += "**Pros:**\n"
            for pro in sym.get("pros", []):
                pm_md += f"- ✅ {pro}\n"
            pm_md += "**Cons:**\n"
            for con in sym.get("cons", []):
                pm_md += f"- ⚠️ {con}\n"

        pm_md += "\n---\n\n## 📊 Section 5 — Market Internals (Global Snapshot)\n"
        pm_md += "| Asset | Value | Change | Impact |\n| :--- | :--- | :--- | :--- |\n"
        for s in pm_report.get("global_snapshot", []):
            sign = "+" if s["change"] >= 0 else ""
            pm_md += f"| {s['name']} | {s['value']} | {sign}{s['change']}% | {s['impact']} — {s['impact_text']} |\n"

        pm_md += f"""
---

## 🚦 Section 6 — Today's Execution Mode
- **Mode:** {exec_mode}
- **Operational Guideline:** {exec_guideline}
- **Minimum Signal Quality Score:** {min_score}/3
- **Stop Loss Width:** {atr_mult:.1f}× ATR
- **Position Size:** {size_pct}% of standard weight

---

## ⚡ Sector Strength Rankings
| Sector | 5D Avg Return | Status |
| :--- | :--- | :--- |
"""
        for sec in pm_report.get("sector_rankings", []):
            sign = "+" if sec["strength"] >= 0 else ""
            pm_md += f"| {sec['sector']} | {sign}{sec['strength']}% | {sec['status']} |\n"

        if best_idea:
            rr_b    = best_idea.get("risk_report", {})
            exp_r_b = best_idea.get("expectancy_r", 0.0)
            cal_p_b = best_idea.get("calibrated_prob", 50.0)
            exp_sign_b = "+" if exp_r_b >= 0 else ""
            pm_md += f"""
---

## 🚀 Section 7 — Best Single Idea of the Day
> **{best_idea['symbol']}** — Q-Score {best_idea.get('confidence_score',0)}/100

| Metric | Value |
| :--- | :--- |
| **Entry Price** | ₹{best_idea.get('close', 0):.2f} |
| **Win Probability** | {cal_p_b:.1f}% |
| **Expectancy** | {exp_sign_b}{exp_r_b:.2f}R |
| **RSI** | {best_idea.get('rsi', 0):.1f} |
"""
            if rr_b.get("approved"):
                pm_md += f"| **Stop Loss** | ₹{rr_b['stop_loss']:.2f} |\n"
                pm_md += f"| **Take Profit** | ₹{rr_b['take_profit']:.2f} |\n"
                pm_md += f"| **Risk/Reward** | {rr_b.get('risk_reward', 0):.1f}x |\n"
                pm_md += f"| **Trade Cost** | ₹{rr_b.get('trade_cost', 0):,.0f} |\n"

        # Save Reports to Disk
        save_markdown_report(os.path.join("state", "pre_market_report.md"), pm_md)

        # Save JSON state for Streamlit dashboard
        save_json_state(os.path.join("state", "pre_market_state.json"), pm_report)
        print("  📊 Pre-market state saved for dashboard.")

        # 5. Process order entry/exits through paper trader
        print("\n  📥 Morning Mode: Processing watchlist signals through paper trader...")
        trader.process_signals(pm_report["watchlist"])
        trader.print_portfolio()

        # 6. Dispatch Telegram alerts
        if notifier.enabled:
            print("  📨 Dispatching pre-market alerts to Telegram...")
            notifier.send_optimization_alert(opt_rules)
            notifier.send_learning_report(tg_summary)
            notifier.send_pre_market_report(pm_report)

            # Send individual BUY/SELL signals
            for r in pm_report["watchlist"]:
                if r.get("signal") in ("BUY", "SELL") and "error" not in r:
                    notifier.send_signal_alert(r)

            # Portfolio Summary Telegram Dispatch
            s = trader.portfolio_summary()
            sign = "+" if s["total_pnl"] >= 0 else ""
            portfolio_msg = (
                f"📊 <b>PAPER PORTFOLIO STATUS</b>\n"
                f"Current Value: ₹{s['total_value']:,.2f}\n"
                f"P&L: {sign}₹{s['total_pnl']:,.2f} ({sign}{s['total_pnl_pct']:.2f}%)\n"
                f"Open Positions: {s['open_positions']}"
            )
            notifier._send(portfolio_msg)
            print("  ✅ Telegram reports sent successfully.")
        else:
            print("  ℹ️ Telegram not configured.")

    else:
        # =======================================================================
        # 🌇 AFTERNOON MODE: END-OF-DAY INTELLIGENCE AUDIT
        # =======================================================================
        report_title = "END-OF-DAY INTELLIGENCE AUDIT"
        print(f"\n{'='*70}")
        print(f"  {report_title} — {now.strftime('%A, %d %b %Y  %H:%M')}")
        print(f"{'='*70}\n")

        # Load journal for start observations count
        journal = load_journal()
        start_obs = journal.get("metadata", {}).get("total_observations", 0)

        # 1. Run EOD Intelligence Cycle
        print("  🕯️ Analyzing Nifty/BankNifty close structure and breadth...")
        eod_report = run_eod_cycle(DATA_CONFIG["symbols"])

        # 2. Save tomorrow forecast to prediction history
        save_today_forecast(eod_report["prediction"])

        # ── Construct EOD Watchlist & Check exits through Paper Trader ──
        print("\n  📥 Afternoon Mode: Processing watchlist exits through paper trader...")
        eod_watchlist = []
        for sym in DATA_CONFIG["symbols"]:
            try:
                df = load_stock_data(sym)  # loads the EOD data fetched in run_eod_cycle
                if not df.empty and len(df) >= 2:
                    close = float(df["Close"].iloc[-1])
                    open_price = float(df["Open"].iloc[-1])
                    eod_watchlist.append({
                        "symbol": sym,
                        "signal": "HOLD",  # exits are checked via SL/TP
                        "close": close,
                        "open_price": open_price,
                        "regime": str(eod_report.get("market_regime", "SIDEWAYS"))
                    })
            except Exception as e:
                print(f"  ⚠️ Failed to prepare EOD signal for {sym}: {e}")

        # Process EOD exits through paper trader
        trader.process_signals(eod_watchlist)
        trader.print_portfolio()

        # Reload journal for EOD updates
        journal = load_journal()
        end_obs = journal.get("metadata", {}).get("total_observations", 0)
        new_obs = end_obs - start_obs

        # Load pre-market state for audit comparison
        pm_state = {}
        pm_state_file = os.path.join("state", "pre_market_state.json")
        if os.path.exists(pm_state_file):
            try:
                with open(pm_state_file, "r", encoding="utf-8") as f:
                    pm_state = json.load(f)
            except Exception:
                pass

        morning_bias    = pm_state.get("opening_bias", "UNKNOWN")
        morning_gap_out = pm_state.get("gap_probability", "UNKNOWN")
        morning_conf    = pm_state.get("confidence_pct", 0)

        nifty_chg   = eod_report["nifty_change"]
        nifty_gap   = eod_report.get("nifty_gap", 0.0)
        bnf_chg     = eod_report.get("banknifty_change", 0.0)
        n_sign      = "+" if nifty_chg >= 0 else ""
        g_sign      = "+" if nifty_gap >= 0 else ""
        b_sign      = "+" if bnf_chg >= 0 else ""

        # Determine directional bias correctness
        actual_dir = "BULLISH" if nifty_chg > 0.15 else ("BEARISH" if nifty_chg < -0.15 else "NEUTRAL")
        bias_correct = False
        if actual_dir == "BULLISH" and "BULL" in morning_bias.upper():
            bias_correct = True
        elif actual_dir == "BEARISH" and "BEAR" in morning_bias.upper():
            bias_correct = True
        elif actual_dir == "NEUTRAL" and ("NEUTRAL" in morning_bias.upper() or "RANGE" in morning_bias.upper()):
            bias_correct = True
            
        bias_icon = "✓" if bias_correct else "✗"
        bias_verdict = f"{bias_icon} Directional Bias: Predicted {morning_bias}, Market closed {actual_dir} (Nifty {n_sign}{nifty_chg:.2f}%)"

        # Determine gap match
        actual_gap_dir = "GAP-UP" if nifty_gap > 0.1 else ("GAP-DOWN" if nifty_gap < -0.1 else "FLAT OPEN")
        gap_match = actual_gap_dir in morning_gap_out.upper() or (
            "RANGE" in morning_gap_out.upper() and actual_gap_dir == "FLAT OPEN"
        )
        gap_icon = "✓" if gap_match else "✗"
        gap_verdict = f"{gap_icon} Opening Gap: Predicted {morning_gap_out}, Actual open was {actual_gap_dir} ({g_sign}{nifty_gap:.2f}%)"

        # Calculate morning's signal performances
        pm_signals = pm_state.get("watchlist", [])
        active_signals = [s for s in pm_signals if s.get("signal") in ("BUY", "SELL") and "error" not in s]
        signal_perf = []
        for s in active_signals:
            sym = s["symbol"]
            try:
                df = load_stock_data(sym)
                if not df.empty and len(df) >= 2:
                    m_close = s.get("close", 0.0)
                    eod_close = float(df["Close"].iloc[-1])
                    ret = ((eod_close - m_close) / m_close) * 100 if m_close > 0 else 0.0
                    
                    # Calculate R-Multiple if we had a stop-loss
                    rr = s.get("risk_report", {})
                    sl = rr.get("stop_loss", 0.0)
                    r_mult = 0.0
                    if sl > 0 and m_close > sl:
                        r_mult = (eod_close - m_close) / (m_close - sl)
                    
                    signal_perf.append({
                        "symbol": sym,
                        "signal": s.get("signal"),
                        "m_close": m_close,
                        "eod_close": eod_close,
                        "return": ret,
                        "r_multiple": r_mult
                    })
            except Exception:
                pass

        # ── Section 1: MARKET CLOSE SUMMARY ──────────────────────────────────
        print(f"\n  ┌─ 📉 MARKET CLOSE SUMMARY {'─'*40}┐")
        print(f"  │  Nifty 50     : {n_sign}{nifty_chg:.2f}%  |  Opening Gap: {g_sign}{nifty_gap:.2f}%")
        print(f"  │  Bank Nifty   : {b_sign}{bnf_chg:.2f}%")
        print(f"  │  Close Type   : {eod_report['close_structure']}")
        print(f"  │  Structure    : {eod_report['structure_desc']}")
        print(f"  │  Advances     : {eod_report['advances']}  |  Declines: {eod_report['declines']}  |  A/D Ratio: {eod_report['adv_dec_ratio']:.2f}")
        print(f"  └{'─'*68}┘")

        # ── Section 2: DID PRE-MARKET PREDICTIONS WORK? ───────────────────────
        acc = eod_report["accuracy_tracking"]
        print(f"\n  ┌─ 🎯 DID PRE-MARKET PREDICTIONS WORK? {'─'*29}┐")
        print(f"  │  {bias_verdict}")
        print(f"  │  {gap_verdict}")
        if acc.get("scored_yesterday"):
            result_icon = "✓" if acc["yesterday_result"] == "SUCCESS" else "✗"
            print(f"  │  {result_icon} Yesterday's EOD Forecast: {acc['yesterday_result']} (Expected: {acc['yesterday_forecast']})")
            print(f"  │  📊 Model Rolling Accuracy: {acc['accuracy_pct']}%  ({acc['successful_predictions']}/{acc['total_predictions']} scored)")
        print(f"  └{'─'*68}┘")

        # ── Section 3: WHAT THE MARKET ACTUALLY DID ───────────────────────────
        print(f"\n  ┌─ 🧠 WHAT THE MARKET ACTUALLY DID {'─'*32}┐")
        print(f"  │  • Price Action: Closed with a {eod_report['close_structure']} ({eod_report['structure_desc']})")
        print(f"  │  • Breadth     : A/D ratio of {eod_report['adv_dec_ratio']:.2f} ({eod_report['advances']} Adv / {eod_report['declines']} Dec)")
        if eod_report["unusual_volume"]:
            v_syms = [uv["symbol"] for uv in eod_report["unusual_volume"][:4]]
            print(f"  │  • Smart Money : High volume activity in {', '.join(v_syms)}")
        if eod_report["breakout_failures"]:
            print(f"  │  • Weakness    : Breakout failures in {', '.join(eod_report['breakout_failures'])}")
        print(f"  └{'─'*68}┘")

        # ── Section 4: TODAY'S BEST/WORST SIGNALS ─────────────────────────────
        print(f"\n  ┌─ 📊 TODAY'S BEST/WORST SIGNALS {'─'*35}┐")
        if signal_perf:
            print(f"  │  {'Stock':<12} {'Signal':<6} {'Morning Close':>13} {'EOD Close':>10} {'Change %':>9} {'R-Mult'}")
            print(f"  │  {'─'*66}")
            for sp in sorted(signal_perf, key=lambda x: x["return"], reverse=True):
                r_sign = "+" if sp["return"] >= 0 else ""
                m_sign = "+" if sp["r_multiple"] >= 0 else ""
                print(
                    f"  │  {sp['symbol']:<12} {sp['signal']:<6} {sp['m_close']:>13.2f} "
                    f"{sp['eod_close']:>10.2f} {r_sign}{sp['return']:>8.2f}% {m_sign}{sp['r_multiple']:>6.2f}R"
                )
        else:
            print("  │  ❌ No actionable BUY/SELL signals triggered this morning.")
        print(f"  └{'─'*68}┘")

        # ── Section 5: LEARNING ENGINE UPDATE ──────────────────────────────────
        print(f"\n  ┌─ 🔬 LEARNING ENGINE UPDATE {'─'*39}┐")
        print(f"  │  • Today's closed trades processed: {new_obs}")
        print(f"  │  • Cumulative dataset size: {end_obs} historical signal evaluations analyzed")
        print(f"  │")
        print(f"  │  Archetype Expectancy Metrics:")
        print(f"  │  {'Archetype':<26} {'n':>5} {'Win Rate':>8} {'PF':>6} {'Expectancy':>10}")
        print(f"  │  {'─'*60}")
        archetypes = journal.get("archetypes", {})
        for name, arch in sorted(archetypes.items(), key=lambda x: x[1].get("expectancy", 0), reverse=True):
            if name == "UNKNOWN_NOISE":
                continue
            wr_pct = arch.get("win_rate", 0) * 100
            pf = arch.get("profit_factor", 1.0)
            exp = arch.get("expectancy", 0.0)
            e_sign = "+" if exp >= 0 else ""
            label = ARCHETYPE_LABELS.get(name, name)[:26]
            print(
                f"  │  {label:<26} {int(round(arch.get('trades',0))):>5} {wr_pct:>7.1f}% {pf:>6.2f} {e_sign}{exp:>9.3f}R"
            )
        print(f"  └{'─'*68}┘")

        # ── Section 6: TOMORROW'S PROBABILITY MAP ─────────────────────────────
        probs = eod_report["prediction"]["probs"]
        forecast = eod_report["prediction"]["forecast"]
        print(f"\n  ┌─ 🔮 TOMORROW'S PROBABILITY MAP {'─'*36}┐")
        print(f"  │  Forecast : {forecast}")
        print(f"  │  Narrative: {eod_report['prediction']['narrative']}")
        print(f"  │")
        # ASCII probability bars
        for label, key, bar_char in [("BULLISH ", "BULLISH", "█"), ("SIDEWAYS", "RANGE", "▓"), ("BEARISH ", "BEARISH", "░")]:
            pct = probs.get(key, 0)
            bar = bar_char * int(pct / 3)
            print(f"  │  {label}: {bar:<34} {pct:.1f}%")
        print(f"  └{'─'*68}┘")

        # ── Build EOD Markdown Report ──────────────────────────────────────────
        sig_rows = []
        for sp in sorted(signal_perf, key=lambda x: x["return"], reverse=True):
            r_sign = "+" if sp["return"] >= 0 else ""
            m_sign = "+" if sp["r_multiple"] >= 0 else ""
            sig_rows.append(
                f"| **{sp['symbol']}** | {sp['signal']} | ₹{sp['m_close']:.2f} | ₹{sp['eod_close']:.2f} | {r_sign}{sp['return']:.2f}% | {m_sign}{sp['r_multiple']:.2f}R |"
            )
        sig_md_table = "\n".join(sig_rows) if sig_rows else "| _No buy/sell signals triggered this morning_ | | | | | |"

        arch_md_rows = []
        for name, arch in sorted(archetypes.items(), key=lambda x: x[1].get("expectancy", 0), reverse=True):
            if name == "UNKNOWN_NOISE":
                continue
            wr_pct = arch.get("win_rate", 0.0) * 100
            pf = arch.get("profit_factor", 1.0)
            exp = arch.get("expectancy", 0.0)
            e_sign = "+" if exp >= 0 else ""
            label = ARCHETYPE_LABELS.get(name, name)
            hold = arch.get("avg_hold_time", 0.0)
            arch_md_rows.append(
                f"| **{label}** | {int(round(arch.get('trades', 0)))} | {wr_pct:.1f}% | {pf:.2f} | {e_sign}{exp:.3f}R | {hold:.1f} bars |"
            )
        arch_md_table = "\n".join(arch_md_rows)

        eod_md = f"""# 🌇 QuantEdge Intelligence Audit — End-of-Day Analysis
_Generated: {eod_report['timestamp']}_

---

## 📉 Section 1 — Market Close Summary

| Index | Session Change | Opening Gap | Structure |
| :--- | :--- | :--- | :--- |
| **Nifty 50** | {n_sign}{nifty_chg:.2f}% | {g_sign}{nifty_gap:.2f}% | {eod_report['close_structure']} |
| **Bank Nifty** | {b_sign}{bnf_chg:.2f}% | — | — |

- **Structural Summary:** *{eod_report['structure_desc']}*
- **Advances:** {eod_report['advances']} symbols  |  **Declines:** {eod_report['declines']} symbols  |  **A/D Ratio:** {eod_report['adv_dec_ratio']:.2f}

---

## 🎯 Section 2 — Pre-Market Prediction Audit

- {bias_verdict}
- {gap_verdict}
"""
        if acc.get("scored_yesterday"):
            result_icon = "✅" if acc["yesterday_result"] == "SUCCESS" else "❌"
            eod_md += f"""
**Yesterday's EOD Prediction Score:** {result_icon} {acc['yesterday_result']}
- Yesterday's Forecast: *{acc['yesterday_forecast']}*
- Today's Actual Nifty Change: {n_sign}{nifty_chg:.2f}%
- **Cumulative Model Accuracy: {acc['accuracy_pct']}%** ({acc['successful_predictions']}/{acc['total_predictions']} scored predictions)
"""

        eod_md += f"""
---

## 🧠 Section 3 — What the Market Actually Did
- **Price Action:** Market closed with a {eod_report['close_structure']}. {eod_report['structure_desc']}
- **Breadth:** Advances/Declines stood at {eod_report['advances']}/{eod_report['declines']} with an A/D Ratio of {eod_report['adv_dec_ratio']:.2f}.
- **Smart Money:** Unusual volume detected in {', '.join([uv['symbol'] for uv in eod_report['unusual_volume']]) if eod_report['unusual_volume'] else 'none'}.
- **Weakness:** Breakout failures occurred in {', '.join(eod_report['breakout_failures']) if eod_report['breakout_failures'] else 'none'}.

---

## 📊 Section 4 — Today's Best/Worst Signals

| Symbol | Signal | Morning Close | EOD Close | Change % | R-Multiple |
| :--- | :--- | :--- | :--- | :--- | :--- |
{sig_md_table}

---

## 🔬 Section 5 — Learning Engine Update
- **Closed Trades Processed Today:** {new_obs}
- **Dataset Size:** {end_obs} historical signal evaluations analyzed.

### Archetype Expectancy Performance
| Archetype State | Observations (n) | Decayed Win Rate | Profit Factor | Expectancy (R) | Avg Hold Period |
| :--- | :---: | :---: | :---: | :---: | :---: |
{arch_md_table}

---

## 🔮 Section 6 — Tomorrow's Early Probability Map

> **{forecast}**

*{eod_report['prediction']['narrative']}*

| Scenario | Model Probability |
| :--- | :--- |
| 🟢 Bullish Continuation | **{probs.get('BULLISH', 0):.1f}%** |
| 🟡 Sideways / Range-Bound | **{probs.get('RANGE', 0):.1f}%** |
| 🔴 Bearish Continuation | **{probs.get('BEARISH', 0):.1f}%** |
"""

        # Save Report to Disk
        save_markdown_report(os.path.join("state", "eod_report.md"), eod_md)

        # 4. Dispatch Telegram report
        if notifier.enabled:
            print("  📨 Dispatching EOD report to Telegram...")
            notifier.send_eod_report(eod_report)
            print("  ✅ Telegram EOD report sent.")
        else:
            print("  ℹ️ Telegram not configured.")

    print(f"\n{'='*70}")
    print(f"  {report_title} complete — {now.strftime('%H:%M:%S')}")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    try:
        run_daily()
    except Exception as exc:
        logger = logging.getLogger(__name__)
        logger.exception("Daily runner failed unexpectedly: %s", exc)
        raise
