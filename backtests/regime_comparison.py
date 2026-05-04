"""
Regime Filter Comparison Backtest

Three-way comparison per symbol:
  A) BASELINE  — RSI + MA + Volume, no regime filter, no controls
  B) REGIME    — + trend strength tiering (6 regimes)
  C) FULL      — + cooldown (5 bars) + min hold (3 bars)

Metrics compared: P&L%, Max Drawdown, Win Rate, Trade Count
"""

import logging
from typing import List
import backtrader as bt

from backtests.bt_strategy import RSI_MA_Strategy
from backtests.regime_bt_strategy import RegimeAwareStrategy
from data.fetcher import fetch_stock_data
from indicators.engine import compute_all_indicators
from config.settings import BACKTEST_CONFIG, DATA_CONFIG, EXECUTION_CONFIG

logger = logging.getLogger(__name__)


def _run_single(df, strategy_cls, symbol: str, extra_params: dict = None, silent: bool = True) -> dict:
    """Run one backtest and return metrics dict."""
    cerebro = bt.Cerebro()

    kwargs = {"printlog": not silent}
    if extra_params:
        kwargs.update(extra_params)
    cerebro.addstrategy(strategy_cls, **kwargs)

    cerebro.adddata(bt.feeds.PandasData(dataname=df), name=symbol)
    cerebro.broker.setcash(BACKTEST_CONFIG["initial_cash"])
    cerebro.broker.setcommission(commission=BACKTEST_CONFIG["commission"])
    cerebro.addsizer(bt.sizers.FixedSize, stake=BACKTEST_CONFIG["stake"])
    cerebro.addanalyzer(bt.analyzers.SharpeRatio,  _name="sharpe", riskfreerate=0.05)
    cerebro.addanalyzer(bt.analyzers.DrawDown,      _name="drawdown")
    cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name="trades")

    start   = cerebro.broker.getvalue()
    results = cerebro.run()
    strat   = results[0]
    end     = cerebro.broker.getvalue()

    pnl     = end - start
    pnl_pct = pnl / start * 100

    dd_info = strat.analyzers.drawdown.get_analysis()
    sh_info = strat.analyzers.sharpe.get_analysis()
    tr_info = strat.analyzers.trades.get_analysis()

    max_dd = dd_info.get("max", {}).get("drawdown", 0.0)
    sharpe = sh_info.get("sharperatio", None)
    total  = tr_info.get("total", {}).get("total", 0)
    won    = tr_info.get("won",   {}).get("total", 0)
    win_rt = (won / total * 100) if total else 0.0

    avg_win  = tr_info.get("won",  {}).get("pnl", {}).get("average", 0.0) or 0.0
    avg_loss = tr_info.get("lost", {}).get("pnl", {}).get("average", 0.0) or 0.0

    return {
        "pnl":          round(pnl, 2),
        "pnl_pct":      round(pnl_pct, 2),
        "max_drawdown": round(max_dd, 2),
        "sharpe":       round(sharpe, 4) if sharpe else None,
        "total_trades": total,
        "won_trades":   won,
        "win_rate_pct": round(win_rt, 2),
        "avg_win":      round(avg_win, 2),
        "avg_loss":     round(avg_loss, 2),
    }


def run_regime_comparison(
    symbols: List[str] = None,
    period: str = "3y",
    silent: bool = True,
) -> List[dict]:
    """
    Three-way comparison: Baseline vs Regime vs Full (regime + cooldown + hold).
    """
    symbols = symbols or DATA_CONFIG["symbols"]
    comparisons = []

    for symbol in symbols:
        logger.info("Comparison: %s", symbol)
        try:
            df = fetch_stock_data(symbol, period=period, save=False)
            df = compute_all_indicators(df)

            # A: Baseline — no regime, no controls
            baseline = _run_single(df, RSI_MA_Strategy, symbol, silent=silent)

            # B: Regime only — trend tiering, no cooldown/hold
            regime_only = _run_single(
                df, RegimeAwareStrategy, symbol,
                extra_params={"cooldown_bars": 0, "min_hold_bars": 0},
                silent=silent,
            )

            # C: Full — regime + cooldown + min hold
            full = _run_single(
                df, RegimeAwareStrategy, symbol,
                extra_params={
                    "cooldown_bars": EXECUTION_CONFIG["cooldown_days"],
                    "min_hold_bars": EXECUTION_CONFIG["min_hold_bars"],
                },
                silent=silent,
            )

            comparisons.append({
                "symbol":   symbol,
                "baseline": baseline,
                "regime":   regime_only,
                "full":     full,
            })

        except Exception as exc:
            logger.error("Comparison failed for %s: %s", symbol, exc)
            comparisons.append({"symbol": symbol, "error": str(exc)})

    return comparisons


def print_regime_comparison(comparisons: List[dict]) -> None:
    """Print a three-way comparison table."""
    W = 90
    print("\n" + "=" * W)
    print("  REGIME FILTER IMPACT — BASELINE vs REGIME vs FULL (regime + cooldown + hold)")
    print("=" * W)
    print(
        f"  {'Symbol':<16} {'Metric':<14} "
        f"{'Baseline':>10} {'Regime':>10} {'Full':>10}  "
        f"{'vs Base':>8}  {'Verdict'}"
    )
    print("  " + "-" * (W - 2))

    full_improved = 0
    total_symbols = 0

    for c in comparisons:
        if "error" in c:
            print(f"  {c['symbol']:<16}  ERROR: {c['error']}")
            continue

        total_symbols += 1
        sym = c["symbol"]
        b   = c["baseline"]
        r   = c["regime"]
        f   = c["full"]

        metrics = [
            ("P&L %",    f"{b['pnl_pct']:+.2f}%",    f"{r['pnl_pct']:+.2f}%",    f"{f['pnl_pct']:+.2f}%",    f['pnl_pct'] - b['pnl_pct'],   True),
            ("Max DD %", f"{b['max_drawdown']:.2f}%", f"{r['max_drawdown']:.2f}%", f"{f['max_drawdown']:.2f}%", b['max_drawdown'] - f['max_drawdown'], True),
            ("Win Rate", f"{b['win_rate_pct']:.1f}%", f"{r['win_rate_pct']:.1f}%", f"{f['win_rate_pct']:.1f}%", f['win_rate_pct'] - b['win_rate_pct'], True),
            ("Trades",   str(b['total_trades']),       str(r['total_trades']),       str(f['total_trades']),       f['total_trades'] - b['total_trades'], False),
            ("Avg Win",  f"{b['avg_win']:,.0f}",       f"{r['avg_win']:,.0f}",       f"{f['avg_win']:,.0f}",       f['avg_win'] - b['avg_win'],   True),
        ]

        if f['pnl_pct'] > b['pnl_pct']:
            full_improved += 1

        for i, (metric, b_val, r_val, f_val, delta, higher_better) in enumerate(metrics):
            label = sym if i == 0 else ""
            if abs(delta) < 0.01 and isinstance(delta, float):
                verdict = "  —"
            elif (delta > 0) == higher_better:
                verdict = "  ✅"
            else:
                verdict = "  ❌"

            delta_str = f"{delta:+.2f}" if isinstance(delta, float) else f"{delta:+d}"
            print(
                f"  {label:<16} {metric:<14} "
                f"{b_val:>10} {r_val:>10} {f_val:>10}  "
                f"{delta_str:>8}{verdict}"
            )

        print("  " + "-" * (W - 2))

    # Summary
    print(f"\n  Full controls improved P&L in {full_improved}/{total_symbols} symbols")
    print(f"  Controls: trend tiering + {EXECUTION_CONFIG['cooldown_days']}-bar cooldown "
          f"+ {EXECUTION_CONFIG['min_hold_bars']}-bar min hold")
    print(f"\n  Key insight: regime filter gates entries — if trade count increases,")
    print(f"  the regime is opening new entries in WEAK_TREND bars not taken by baseline.")
    print(f"  Tune slope thresholds in REGIME_CONFIG to control this.")
    print("=" * W + "\n")
