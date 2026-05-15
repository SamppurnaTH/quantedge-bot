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
from config.settings import DATA_CONFIG, RISK_CONFIG, PORTFOLIO_CONFIG


def run_daily():
    setup_logging()

    now = datetime.now()
    print(f"\n{'='*60}")
    print(f"  DAILY RUN — {now.strftime('%A, %d %b %Y  %H:%M')}")
    print(f"{'='*60}\n")

    # ── 1. Market regime ──────────────────────────────────────────────────────
    market_up, regime, result = fetch_index_regime()
    print(f"  Market Regime : {regime_summary(regime)}")
    print(f"  Slope: {result.slope:+.5f}   ATR Ratio: {result.atr_ratio:.2f}")
    print(f"  Buys allowed  : {'Yes' if market_up else 'No'}\n")

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
        send_telegram=False,   # we send manually below with richer context
    )

    print_signal_report(results, market_regime=regime)
    log_signals(results)

    # ── 4. Process through paper trader ──────────────────────────────────────
    print("  Processing signals through paper trader...")
    trader.process_signals(results)
    trader.print_portfolio()

    # ── 5. Telegram summary ───────────────────────────────────────────────────
    notifier = TelegramNotifier()
    if notifier.enabled:
        # Regime alert
        notifier.send_regime_alert(str(regime), regime.description)

        # Daily summary
        notifier.send_daily_summary(results, regime_str=str(regime))

        # Individual BUY/SELL alerts
        for r in results:
            if r.get("signal") in ("BUY", "SELL") and "error" not in r:
                notifier.send_signal_alert(r)

        # Portfolio update
        s = trader.portfolio_summary()
        sign = "+" if s["total_pnl"] >= 0 else ""
        portfolio_msg = (
            f"📊 <b>PAPER PORTFOLIO UPDATE</b>\n"
            f"Value: ₹{s['total_value']:,.2f}\n"
            f"P&L: {sign}₹{s['total_pnl']:,.2f} ({sign}{s['total_pnl_pct']:.2f}%)\n"
            f"Open: {s['open_positions']}  Closed: {s['closed_trades']}  "
            f"Win: {s['win_rate_pct']:.1f}%\n"
            f"Expectancy: {sign}₹{s['expectancy']:,.2f}/trade"
        )
        notifier._send(portfolio_msg)
        print("\n  ✅ Telegram alerts sent.")
    else:
        print("\n  ℹ️  Telegram not configured — add TELEGRAM_BOT_TOKEN to .env")

    print(f"\n{'='*60}")
    print(f"  Daily run complete — {now.strftime('%H:%M:%S')}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    run_daily()
