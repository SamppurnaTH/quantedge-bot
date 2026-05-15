"""
Trading Bot — Main Entry Point

Usage:
    python main.py                          # scan full watchlist
    python main.py --symbol TCS.NS          # single symbol
    python main.py --backtest               # backtest full watchlist
    python main.py --compare                # regime filter: with vs without
    python main.py --ab-test                # A/B test all strategy versions
    python main.py --filter-analysis        # which trades did the filter remove?
    python main.py --dashboard              # performance dashboard
    python main.py --paper                  # paper trading simulation
    python main.py --paper --reset          # reset paper portfolio
    python main.py --journal                # trade journal analytics
    python main.py --regime                 # show current market regime only
    python main.py --telegram               # scan + send Telegram alerts
    python main.py --test-telegram          # verify Telegram connection
"""

import sys
import os
import argparse
import logging

# Force UTF-8 encoding for stdout/stderr to handle emojis on all platforms
if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')
if sys.stderr.encoding != 'utf-8':
    sys.stderr.reconfigure(encoding='utf-8')

sys.path.insert(0, os.path.dirname(__file__))

# Load .env before any module reads environment variables
from logger.env_loader import load_env
load_env()

from logger.setup import setup_logging
from signals.generator import scan_watchlist, print_signal_report, fetch_index_regime
from indicators.regime import Regime, regime_summary
from backtests.runner import run_backtest, run_multi_backtest, print_backtest_summary
from backtests.regime_comparison import run_regime_comparison, print_regime_comparison
from data.fetcher import fetch_stock_data
from indicators.engine import compute_all_indicators
from logger.trade_tracker import log_signals, log_backtest
from execution.paper_trader import PaperTrader, PAPER_STATE_FILE
from analytics.journal import print_journal_report
from analytics.dashboard import print_dashboard
from analytics.ab_test import run_ab_test, print_ab_report, VERSIONS
from analytics.filter_analysis import run_filter_analysis, print_filter_analysis
from config.settings import DATA_CONFIG, RISK_CONFIG, PORTFOLIO_CONFIG, REGIME_CONFIG


def parse_args():
    parser = argparse.ArgumentParser(description="Trading Decision-Support Bot")
    parser.add_argument("--symbol",   type=str,   help="Single ticker to analyse")
    parser.add_argument("--backtest", action="store_true", help="Run backtest(s)")
    parser.add_argument("--compare",  action="store_true", help="Regime filter: with vs without comparison")
    parser.add_argument("--paper",    action="store_true", help="Run paper trading")
    parser.add_argument("--reset",    action="store_true", help="Reset paper portfolio")
    parser.add_argument("--journal",   action="store_true", help="Show trade journal analytics")
    parser.add_argument("--dashboard", action="store_true", help="Show performance dashboard")
    parser.add_argument("--ab-test",      action="store_true", help="Run A/B test across strategy versions")
    parser.add_argument("--filter-analysis", action="store_true", help="Analyse which trades the regime filter removed")
    parser.add_argument("--regime",    action="store_true", help="Show market regime only")
    parser.add_argument("--telegram", action="store_true", help="Send Telegram alerts")
    parser.add_argument("--test-telegram", action="store_true", help="Send a test Telegram message")
    parser.add_argument("--top",      type=int, default=PORTFOLIO_CONFIG["top_n_signals"],
                        help="Max BUY signals to act on (portfolio filter)")
    parser.add_argument("--capital",  type=float, default=RISK_CONFIG["default_capital"],
                        help="Capital for risk sizing")
    return parser.parse_args()


def run_signal_scan(symbols, capital, top_n, send_telegram=False):
    logger = logging.getLogger(__name__)
    logger.info("Signal scan | symbols=%s | top_n=%d", symbols, top_n)

    results = scan_watchlist(
        symbols=symbols, capital=capital, top_n=top_n,
        send_telegram=send_telegram,
    )
    regime = None
    for r in results:
        if "market_regime" in r:
            try:
                regime = Regime(r["market_regime"])
            except Exception:
                pass
            break

    print_signal_report(results, market_regime=regime)
    log_signals(results)
    return results


def run_backtests(symbols, capital):
    logger = logging.getLogger(__name__)
    if len(symbols) == 1:
        symbol = symbols[0]
        logger.info("Single backtest: %s", symbol)
        df      = fetch_stock_data(symbol, save=False)
        df      = compute_all_indicators(df)
        metrics = run_backtest(df, symbol=symbol)
        log_backtest(metrics)
        _print_single_backtest(metrics)
    else:
        logger.info("Multi-stock backtest: %s", symbols)
        results = run_multi_backtest(symbols=symbols)
        for m in results:
            if "error" not in m:
                log_backtest(m)
        print_backtest_summary(results)


def run_paper_trading(symbols, capital, top_n, reset=False):
    if reset and os.path.exists(PAPER_STATE_FILE):
        os.remove(PAPER_STATE_FILE)
        print("  Paper portfolio reset.\n")

    trader  = PaperTrader(capital=capital)
    results = scan_watchlist(
        symbols=symbols,
        capital=capital,
        active_trades=len(trader.positions),
        top_n=top_n,
    )

    regime = None
    for r in results:
        if "market_regime" in r:
            try:
                regime = Regime(r["market_regime"])
            except Exception:
                pass
            break

    print_signal_report(results, market_regime=regime)
    log_signals(results)

    print("\n  Processing signals through paper trader...")
    trader.process_signals(results)
    trader.print_portfolio()


def show_regime():
    market_up, regime, result = fetch_index_regime()
    print(f"\n  {regime_summary(regime)}")
    print(f"  Trend Strength : {regime.trend_strength}")
    print(f"  Slope          : {result.slope:+.5f}  "
          f"(strong>{REGIME_CONFIG['slope_strong_up']:.4f}  "
          f"weak>{REGIME_CONFIG['slope_weak_up']:.4f})")
    print(f"  ATR Ratio      : {result.atr_ratio:.2f}  (volatile>{REGIME_CONFIG['atr_ratio_threshold']})")
    print(f"  Price          : {result.price:.2f}  MA50: {result.ma50:.2f}  MA200: {result.ma200:.2f}")
    print(f"  Buys allowed   : {'Yes' if market_up else 'No'}")
    print(f"  Min score      : {regime.min_score_to_buy}/3")
    print(f"  Size mult      : ×{regime.position_size_multiplier}")
    print(f"  SL mult        : ×{regime.atr_sl_multiplier}\n")


def _print_single_backtest(m: dict) -> None:
    print(f"\n{'='*60}")
    print(f"  BACKTEST: {m['symbol']}")
    print(f"{'='*60}")
    print(f"  Start Value  : {m['start_value']:>12,.2f}")
    print(f"  End Value    : {m['end_value']:>12,.2f}")
    sign = "+" if m["pnl"] >= 0 else ""
    print(f"  P&L          : {sign}{m['pnl']:>11,.2f}  ({sign}{m['pnl_pct']:.2f}%)")
    print(f"  Max Drawdown : {m['max_drawdown']:>11.2f}%")
    print(f"  Sharpe Ratio : {str(m['sharpe_ratio']):>12}")
    print(f"  Total Trades : {m['total_trades']:>12}")
    print(f"  Won / Lost   : {m['won_trades']} / {m['lost_trades']}")
    print(f"  Win Rate     : {m['win_rate_pct']:>11.2f}%")
    print(f"  Avg Win      : {m['avg_win']:>12,.2f}")
    print(f"  Avg Loss     : {m['avg_loss']:>12,.2f}")
    print(f"{'='*60}\n")


def main():
    setup_logging()
    logger = logging.getLogger(__name__)
    args    = parse_args()
    symbols = [args.symbol] if args.symbol else DATA_CONFIG["symbols"]

    if args.journal:
        print_journal_report()
    elif args.dashboard:
        print_dashboard()
    elif getattr(args, "ab_test", False):
        logger.info("A/B test: %s", symbols)
        results = run_ab_test(symbols=symbols)
        print_ab_report(results, versions=VERSIONS)
    elif getattr(args, "filter_analysis", False):
        logger.info("Filter analysis: %s", symbols)
        results = run_filter_analysis(symbols=symbols)
        print_filter_analysis(results)
    elif args.regime:
        show_regime()
    elif getattr(args, "test_telegram", False):
        from notifications.telegram import TelegramNotifier
        ok = TelegramNotifier().test_connection()
        print(f"  Telegram test: {'✅ sent' if ok else '❌ failed (check .env credentials)'}")
    elif args.compare:
        logger.info("Regime comparison backtest: %s", symbols)
        comparisons = run_regime_comparison(symbols=symbols)
        print_regime_comparison(comparisons)
    elif args.paper:
        run_paper_trading(symbols, args.capital, args.top, reset=args.reset)
    elif args.backtest:
        run_backtests(symbols, args.capital)
    else:
        run_signal_scan(symbols, args.capital, args.top, send_telegram=args.telegram)


if __name__ == "__main__":
    main()
