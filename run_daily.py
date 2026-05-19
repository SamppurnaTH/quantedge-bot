"""
Daily Paper Trading & Market Intelligence Runner

Determines execution mode dynamically based on the current time (IST):
  - Morning Mode (before 12:00 PM IST): Pre-Market Intelligence & Order Execution Cycle.
  - Afternoon Mode (12:00 PM IST or later): End-of-Day Analysis & Predictive Outlook Cycle.
"""

import sys
import os
from datetime import datetime

# Force UTF-8 encoding for stdout/stderr to handle emojis on all platforms
if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')
if sys.stderr.encoding != 'utf-8':
    sys.stderr.reconfigure(encoding='utf-8')

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


def run_daily():
    setup_logging()

    now = datetime.now()
    # Before 12:00 PM IST is Morning Pre-Market Scan. 12:00 PM IST or after is EOD.
    is_afternoon = now.hour >= 12

    # Initialize Notifier & Trader
    notifier = TelegramNotifier()
    trader = PaperTrader(capital=RISK_CONFIG["default_capital"])

    if not is_afternoon:
        # =======================================================================
        # 🌅 MORNING MODE: PRE-MARKET INTEL & EXECUTION CYCLE
        # =======================================================================
        report_title = "PRE-MARKET PREPARATION SCAN"
        print(f"\n{'='*70}")
        print(f"  {report_title} — {now.strftime('%A, %d %b %Y  %H:%M')}")
        print(f"{'='*70}\n")

        # 1. Run Auto-Optimization
        print("  ⚙️  Running auto-optimization from paper trade history...")
        opt_rules = run_auto_optimization()

        # 2. Seed and Generate Learning Report
        journal = load_journal()
        if journal.get("metadata", {}).get("total_observations", 0) == 0:
            print("  🌱 Seeding learning journal from 10-year history (first run)...")
            journal = seed_from_history()
        
        # Update confidence calibration file dynamically (CRITICAL ISSUE #4)
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

        # 4. Generate Pre-Market Markdown Report
        pm_md = f"""# 🔮 QuantEdge — Pre-Market Preparation Report
_Generated: {pm_report['timestamp']}_

---

## 📈 Broad Market Context
- **Index Regime:** {regime_summary(pm_report['market_regime'])}
- **Opening Bias:** **{pm_report['opening_bias']}**
- **Confidence Rating:** **{pm_report['confidence_pct']}%**
- **Gap Expectation:** **{pm_report['gap_probability']}**
- **Risk Level:** **{pm_report['risk_level']}**

---

## 🌍 Global Assets Snapshot
| Asset Name | Value | Daily Change | Impact |
| :--- | :--- | :--- | :--- |
"""
        for s in pm_report["global_snapshot"]:
            sign = "+" if s['change'] >= 0 else ""
            pm_md += f"| {s['name']} | {s['value']} | {sign}{s['change']}% | {s['impact']} ({s['impact_text']}) |\n"

        pm_md += """
---

## ⚡ Sector Strength Rankings
| Sector Name | Average 5D Return | Status |
| :--- | :--- | :--- |
"""
        for sec in pm_report["sector_rankings"]:
            sign = "+" if sec['strength'] >= 0 else ""
            pm_md += f"| {sec['sector']} | {sign}{sec['strength']}% | {sec['status']} |\n"

        pm_md += """
---

## 🏆 Watchlist Quality Rankings
| Symbol | Q-Score (0-100) | Signal | Current Close | RSI |
| :--- | :--- | :--- | :--- | :--- |
"""
        for sym in pm_report["watchlist"]:
            pm_md += f"| {sym['symbol']} | **{sym.get('confidence_score',0)}/100** | {sym.get('signal','HOLD')} | {sym['close']:.2f} | {sym['rsi']:.2f} |\n"

        if pm_report["warnings"]:
            pm_md += "\n---\n\n## ⚠️ Do Not Trade Warnings\n"
            for w in pm_report["warnings"]:
                pm_md += f"- {w}\n"

        # Save Report to Disk
        save_markdown_report(os.path.join("state", "pre_market_report.md"), pm_md)

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
        # 🌇 AFTERNOON MODE: END-OF-DAY INTELLIGENCE CYCLE
        # =======================================================================
        report_title = "POST-MARKET END-OF-DAY ANALYSIS"
        print(f"\n{'='*70}")
        print(f"  {report_title} — {now.strftime('%A, %d %b %Y  %H:%M')}")
        print(f"{'='*70}\n")

        # 1. Run EOD Intelligence Cycle
        print("  🕯️ Analyzing Nifty candle close structure and breadth...")
        eod_report = run_eod_cycle(DATA_CONFIG["symbols"])

        # 2. Save tomorrow forecast to prediction history
        save_today_forecast(eod_report["prediction"])

        # 3. Generate End-of-Day Markdown Report
        eod_md = f"""# 🌇 QuantEdge — End-of-Day Market Analysis
_Generated: {eod_report['timestamp']}_

---

## 🕯️ Index Session Close
- **Nifty 50 Change:** {eod_report['nifty_change']}%
- **Candle Structure:** **{eod_report['close_structure']}**
- **Structural Summary:** *{eod_report['structure_desc']}*

---

## 📈 Watchlist Breadth Analysis
- **Advances:** {eod_report['advances']} symbols
- **Declines:** {eod_report['declines']} symbols
- **Advances/Declines Ratio:** **{eod_report['adv_dec_ratio']}**

---

## 🔮 Tomorrow Outlook Forecast
- **Tomorrow Forecast:** **{eod_report['prediction']['forecast']}**
- **Calibrated Prediction Narrative:** *{eod_report['prediction']['narrative']}*
- **Bullish Continuation Probability:** {eod_report['prediction']['probs']['BULLISH']}%
- **Range-bound Probability:** {eod_report['prediction']['probs']['RANGE']}%
- **Bearish Continuation Probability:** {eod_report['prediction']['probs']['BEARISH']}%

---

## 🐋 Smart Money Spikes (Unusual Volume)
| Symbol | Volume Ratio (vs 20D MA) | Daily Price Change |
| :--- | :--- | :--- |
"""
        for uv in eod_report["unusual_volume"]:
            uv_sign = "+" if uv['change'] >= 0 else ""
            eod_md += f"| {uv['symbol']} | {uv['ratio']}x | {uv_sign}{uv['change']}% |\n"

        if eod_report["breakout_failures"]:
            eod_md += "\n---\n\n## ⚠️ Breakout Failure Warnings\n"
            for bf in eod_report["breakout_failures"]:
                eod_md += f"- **{bf}** closed weak despite elevated intraday breakout volume.\n"

        acc = eod_report["accuracy_tracking"]
        if acc.get("scored_yesterday"):
            eod_md += f"""
---

## 🎯 Prediction Accuracy Tracking
- **Yesterday's Forecast:** {acc['yesterday_forecast']}
- **Today's Actual Outcome:** **{acc['yesterday_result']}**
- **Cumulative Model Precision:** **{acc['accuracy_pct']}%** ({acc['successful_predictions']}/{acc['total_predictions']})
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
    run_daily()
