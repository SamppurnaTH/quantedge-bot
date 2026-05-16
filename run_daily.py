"""
Daily Paper Trading Runner

Run this every morning before market open (9:00–9:15 AM IST):
    python run_daily.py

What it does:
  1. Checks market regime
  2. Scans all watchlist symbols
  3. Processes signals through paper trader
  4. Sends Telegram alert with today's signals
  5. Logs everything to CSV

Weekend review (run manually):
    python main.py --dashboard
    python main.py --filter-analysis
    python main.py --journal
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
from signals.generator import scan_watchlist, print_signal_report, fetch_index_regime
from indicators.regime import Regime, regime_summary
from execution.paper_trader import PaperTrader, PAPER_STATE_FILE
from logger.trade_tracker import log_signals
from notifications.telegram import TelegramNotifier
from analytics.dashboard import run_auto_optimization
from analytics.learning_engine import load_journal, seed_from_history
from analytics.learning_report import run_learning_report
from config.settings import DATA_CONFIG, RISK_CONFIG, PORTFOLIO_CONFIG


def get_market_outlook(regime, result):
    """Generates a predictive narrative for the next trading day."""
    slope = result.slope
    price = result.price
    ma50 = result.ma50
    
    outlook = {
        "sentiment": "NEUTRAL",
        "narrative": "",
        "action": "WAIT"
    }

    if regime == Regime.STRONG_TREND_UP:
        outlook["sentiment"] = "BULLISH"
        outlook["narrative"] = "Market is in a powerful uptrend. Expect continued momentum, but watch for minor profit-taking."
        outlook["action"] = "BUY ON DIPS"
    elif regime == Regime.WEAK_TREND_UP:
        outlook["sentiment"] = "CAUTIOUSLY BULLISH"
        outlook["narrative"] = "Uptrend is slowing down. Volatility might increase tomorrow as it tests resistance levels."
        outlook["action"] = "SELECTIVE BUYS"
    elif regime == Regime.SIDEWAYS:
        outlook["sentiment"] = "NEUTRAL"
        outlook["narrative"] = "Market is range-bound. Expect a flat opening tomorrow unless global cues are strong."
        outlook["action"] = "RANGE TRADING"
    elif regime == Regime.VOLATILE:
        outlook["sentiment"] = "HIGH UNCERTAINTY"
        outlook["narrative"] = "High volatility detected. Tomorrow could see wild swings in both directions."
        outlook["action"] = "REDUCE QUANTITY"
    elif regime == Regime.STRONG_TREND_DOWN:
        outlook["sentiment"] = "BEARISH"
        outlook["narrative"] = "Strong downward pressure. Expect a weak opening tomorrow."
        outlook["action"] = "STAY IN CASH / SHORT"
    
    # Add technical context
    if price > ma50 * 1.05:
        outlook["narrative"] += " (Note: Trading far above 50-DMA, potential pullback candidate)"
    elif price < ma50 * 0.95:
        outlook["narrative"] += " (Note: Deeply oversold relative to 50-DMA, watch for bounce)"

    return outlook


def run_daily():
    setup_logging()

    now = datetime.now()
    # Determine mode based on time if not specified
    # Morning: 0:00 to 12:00, Evening: 12:00 to 23:59
    is_evening = now.hour >= 12
    report_title = "POST-MARKET OUTLOOK" if is_evening else "PRE-MARKET SCAN"
    target_day = "Tomorrow" if is_evening else "Today"

    print(f"\n{'='*60}")
    print(f"  {report_title} — {now.strftime('%A, %d %b %Y  %H:%M')}")
    print(f"{'='*60}\n")

    # ── 0. Auto-Optimize: learn from paper trading history ────────────────────
    print("  ⚙️  Running auto-optimization from paper trade history...")
    opt_rules = run_auto_optimization()
    
    # ── 0.1 Learning Engine: seed if needed, then build report ────────────────
    journal = load_journal()
    if journal.get("metadata", {}).get("total_observations", 0) == 0:
        print("  🌱 Seeding learning journal from 10-year history (first run)...")
        journal = seed_from_history()
    
    print("  📚 Generating daily learning report...")
    full_report, tg_summary = run_learning_report(journal)

    # ── 1. Market regime & Outlook ────────────────────────────────────────────
    market_up, regime, result = fetch_index_regime()
    outlook = get_market_outlook(regime, result)
    
    print(f"  Market Regime : {regime_summary(regime)}")
    print(f"  Outlook for {target_day}: {outlook['sentiment']} ({outlook['action']})")
    print(f"  Narrative     : {outlook['narrative']}\n")

    # ── 2. Load paper portfolio ───────────────────────────────────────────────
    trader = PaperTrader(capital=RISK_CONFIG["default_capital"])
    summary = trader.portfolio_summary()
    print(f"  Portfolio     : ₹{summary['total_value']:,.2f}  "
          f"(P&L: {'+' if summary['total_pnl'] >= 0 else ''}{summary['total_pnl']:,.2f}  "
          f"{'+' if summary['total_pnl_pct'] >= 0 else ''}{summary['total_pnl_pct']:.2f}%)")
    print(f"  Open positions: {summary['open_positions']}   "
          f"Closed trades: {summary['closed_trades']}   "
          f"Win rate: {summary['win_rate_pct']:.1f}%\n")

    # ── 3. Scan signals ───────────────────────────────────────────────────────
    results = scan_watchlist(
        symbols=DATA_CONFIG["symbols"],
        capital=RISK_CONFIG["default_capital"],
        active_trades=len(trader.positions),
        top_n=PORTFOLIO_CONFIG["top_n_signals"],
        send_telegram=False,
    )

    print_signal_report(results, market_regime=regime)
    log_signals(results)

    # ── 4. Process through paper trader ──────────────────────────────────────
    if not is_evening:
        print("  Morning Mode: Processing signals through paper trader...")
        trader.process_signals(results)
        trader.print_portfolio()
    else:
        print("  Evening Mode: Signals are for tomorrow's preparation. No orders placed.")

    # ── 5. Telegram summary ───────────────────────────────────────────────────
    notifier = TelegramNotifier()
    if notifier.enabled:
        # Outlook Header
        outlook_msg = (
            f"🔮 <b>{report_title} FOR {target_day.upper()}</b>\n"
            f"Market: <b>{regime_summary(regime)}</b>\n"
            f"Sentiment: <b>{outlook['sentiment']}</b>\n"
            f"Strategy: <i>{outlook['action']}</i>\n\n"
            f"📝 {outlook['narrative']}"
        )
        notifier._send(outlook_msg)

        # Optimization & Learning
        notifier.send_optimization_alert(opt_rules)
        notifier.send_learning_report(tg_summary)

        # Daily summary
        notifier.send_daily_summary(results, regime_str=str(regime))

        # Individual BUY/SELL alerts (Only in morning or for prep)
        for r in results:
            if r.get("signal") in ("BUY", "SELL") and "error" not in r:
                notifier.send_signal_alert(r)

        # Portfolio update
        s = trader.portfolio_summary()
        sign = "+" if s["total_pnl"] >= 0 else ""
        portfolio_msg = (
            f"📊 <b>PAPER PORTFOLIO STATUS</b>\n"
            f"Current Value: ₹{s['total_value']:,.2f}\n"
            f"P&L: {sign}₹{s['total_pnl']:,.2f} ({sign}{s['total_pnl_pct']:.2f}%)\n"
            f"Open Positions: {s['open_positions']}"
        )
        notifier._send(portfolio_msg)
        print("\n  ✅ Telegram reports sent.")
    else:
        print("\n  ℹ️  Telegram not configured.")

    print(f"\n{'='*60}")
    print(f"  {report_title} complete — {now.strftime('%H:%M:%S')}")
    print(f"{'='*60}\n")



if __name__ == "__main__":
    run_daily()
