"""
Backtest Runner
Supports single-symbol and multi-symbol backtests with full metrics.
"""

import logging
from typing import List
import pandas as pd
import backtrader as bt

from backtests.bt_strategy import RSI_MA_Strategy
from data.fetcher import fetch_stock_data
from indicators.engine import compute_all_indicators
from config.settings import BACKTEST_CONFIG

logger = logging.getLogger(__name__)


def _build_cerebro(df: pd.DataFrame, symbol: str) -> bt.Cerebro:
    """Create and configure a Cerebro instance for one symbol."""
    cerebro = bt.Cerebro()
    cerebro.addstrategy(RSI_MA_Strategy)

    feed = bt.feeds.PandasData(dataname=df)
    cerebro.adddata(feed, name=symbol)

    cerebro.broker.setcash(BACKTEST_CONFIG["initial_cash"])
    cerebro.broker.setcommission(commission=BACKTEST_CONFIG["commission"])
    cerebro.addsizer(bt.sizers.FixedSize, stake=BACKTEST_CONFIG["stake"])

    cerebro.addanalyzer(bt.analyzers.SharpeRatio,  _name="sharpe",   riskfreerate=0.05)
    cerebro.addanalyzer(bt.analyzers.DrawDown,      _name="drawdown")
    cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name="trades")
    cerebro.addanalyzer(bt.analyzers.Returns,       _name="returns")
    return cerebro


def _extract_metrics(cerebro: bt.Cerebro, strat, symbol: str, start_value: float) -> dict:
    """Pull metrics out of analyzers into a clean dict."""
    end_value = cerebro.broker.getvalue()
    pnl       = end_value - start_value
    pnl_pct   = (pnl / start_value) * 100

    dd_info    = strat.analyzers.drawdown.get_analysis()
    sharpe_info = strat.analyzers.sharpe.get_analysis()
    trade_info  = strat.analyzers.trades.get_analysis()

    max_dd = dd_info.get("max", {}).get("drawdown", 0.0)
    sharpe = sharpe_info.get("sharperatio", None)

    total  = trade_info.get("total",  {}).get("total",  0)
    won    = trade_info.get("won",    {}).get("total",  0)
    lost   = trade_info.get("lost",   {}).get("total",  0)
    win_rt = (won / total * 100) if total else 0.0

    avg_win  = trade_info.get("won",  {}).get("pnl", {}).get("average", 0.0) or 0.0
    avg_loss = trade_info.get("lost", {}).get("pnl", {}).get("average", 0.0) or 0.0

    return {
        "symbol":        symbol,
        "start_value":   round(start_value, 2),
        "end_value":     round(end_value, 2),
        "pnl":           round(pnl, 2),
        "pnl_pct":       round(pnl_pct, 2),
        "max_drawdown":  round(max_dd, 2),
        "sharpe_ratio":  round(sharpe, 4) if sharpe else "N/A",
        "total_trades":  total,
        "won_trades":    won,
        "lost_trades":   lost,
        "win_rate_pct":  round(win_rt, 2),
        "avg_win":       round(avg_win, 2),
        "avg_loss":      round(avg_loss, 2),
    }


def run_backtest(df: pd.DataFrame, symbol: str = "STOCK") -> dict:
    """Run a backtest on a pre-loaded DataFrame."""
    cerebro     = _build_cerebro(df, symbol)
    start_value = cerebro.broker.getvalue()
    logger.info("Backtest start | %s | Capital: %.2f", symbol, start_value)

    results = cerebro.run()
    metrics = _extract_metrics(cerebro, results[0], symbol, start_value)
    logger.info("Backtest done  | %s | PnL: %.2f (%.2f%%)", symbol, metrics["pnl"], metrics["pnl_pct"])
    return metrics


def run_multi_backtest(symbols: List[str] = None, period: str = "3y") -> List[dict]:
    """
    Run independent backtests for multiple symbols and return all results.

    Args:
        symbols: List of tickers (defaults to DATA_CONFIG watchlist)
        period:  History period to fetch

    Returns:
        List of metric dicts, sorted by P&L % descending
    """
    from config.settings import DATA_CONFIG
    symbols = symbols or DATA_CONFIG["symbols"]

    all_metrics = []
    for symbol in symbols:
        try:
            logger.info("Fetching data for backtest: %s", symbol)
            df      = fetch_stock_data(symbol, period=period, save=False)
            df      = compute_all_indicators(df)
            metrics = run_backtest(df, symbol=symbol)
            all_metrics.append(metrics)
        except Exception as exc:
            logger.error("Backtest failed for %s: %s", symbol, exc)
            all_metrics.append({"symbol": symbol, "error": str(exc)})

    # Sort by P&L % descending (errors go last)
    all_metrics.sort(key=lambda m: m.get("pnl_pct", float("-inf")), reverse=True)
    return all_metrics


def print_backtest_summary(results: List[dict]) -> None:
    """Print a comparison table of multi-stock backtest results."""
    print("\n" + "=" * 75)
    print("  MULTI-STOCK BACKTEST SUMMARY")
    print("=" * 75)
    print(f"  {'Symbol':<18} {'P&L':>10} {'P&L%':>7} {'MaxDD%':>8} {'Trades':>7} {'WinRate':>8} {'Sharpe':>8}")
    print("  " + "-" * 71)

    for m in results:
        if "error" in m:
            print(f"  {m['symbol']:<18}  ERROR: {m['error']}")
            continue
        pnl_sign = "+" if m["pnl"] >= 0 else ""
        print(
            f"  {m['symbol']:<18} "
            f"{pnl_sign}{m['pnl']:>9,.2f} "
            f"{pnl_sign}{m['pnl_pct']:>6.2f}% "
            f"{m['max_drawdown']:>7.2f}% "
            f"{m['total_trades']:>7} "
            f"{m['win_rate_pct']:>7.2f}% "
            f"{str(m['sharpe_ratio']):>8}"
        )

    print("=" * 75 + "\n")
