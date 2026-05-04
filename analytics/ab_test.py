"""
A/B Testing Framework

Tests strategy versions against each other on the same data.
Each version is a named configuration — one parameter changes at a time.

Versions tested:
  V1_BASELINE    — RSI < 40, no regime, no controls
  V2_REGIME      — RSI adaptive, regime tiering only
  V3_FULL        — regime + cooldown + min hold
  V4_PULLBACK    — V3 + pullback depth filter (3%)
  V5_TOP2        — V4 + portfolio cap at top 2 signals

Output:
  - Side-by-side metrics table
  - Winner per metric
  - Recommendation: which version to use
  - Results saved to logs/ab_test_results.json
"""

import logging
from typing import List, Dict
from dataclasses import dataclass, field

import backtrader as bt

from backtests.bt_strategy import RSI_MA_Strategy
from backtests.regime_bt_strategy import RegimeAwareStrategy
from data.fetcher import fetch_stock_data
from indicators.engine import compute_all_indicators
from config.settings import BACKTEST_CONFIG, DATA_CONFIG, EXECUTION_CONFIG
from analytics.dashboard import save_ab_result

logger = logging.getLogger(__name__)


# ── Version definitions ───────────────────────────────────────────────────────

@dataclass
class StrategyVersion:
    name:        str
    description: str
    strategy_cls: type
    params:      dict = field(default_factory=dict)


VERSIONS: List[StrategyVersion] = [
    StrategyVersion(
        name="V1_BASELINE",
        description="RSI<40, no regime, no controls",
        strategy_cls=RSI_MA_Strategy,
        params={"rsi_buy": 40},
    ),
    StrategyVersion(
        name="V2_REGIME",
        description="Regime-aware RSI, no cooldown/hold",
        strategy_cls=RegimeAwareStrategy,
        params={"cooldown_bars": 0, "min_hold_bars": 0},
    ),
    StrategyVersion(
        name="V3_FULL",
        description="Regime + cooldown + min hold",
        strategy_cls=RegimeAwareStrategy,
        params={
            "cooldown_bars": EXECUTION_CONFIG["cooldown_days"],
            "min_hold_bars": EXECUTION_CONFIG["min_hold_bars"],
        },
    ),
    StrategyVersion(
        name="V4_STRICT_RSI",
        description="V3 + tighter RSI (strong=30, weak=25, sideways=20)",
        strategy_cls=RegimeAwareStrategy,
        params={
            "cooldown_bars":    EXECUTION_CONFIG["cooldown_days"],
            "min_hold_bars":    EXECUTION_CONFIG["min_hold_bars"],
            "rsi_buy_strong":   30,
            "rsi_buy_weak":     25,
            "rsi_buy_sideways": 20,
        },
    ),
    StrategyVersion(
        name="V5_WIDER_HOLD",
        description="V3 + longer min hold (5 bars)",
        strategy_cls=RegimeAwareStrategy,
        params={
            "cooldown_bars": EXECUTION_CONFIG["cooldown_days"],
            "min_hold_bars": 5,
        },
    ),
]


# ── Runner ────────────────────────────────────────────────────────────────────

def _run_version(df, version: StrategyVersion, symbol: str, silent: bool = True) -> dict:
    """Run one version and return metrics."""
    cerebro = bt.Cerebro()
    kwargs  = {"printlog": not silent, **version.params}
    cerebro.addstrategy(version.strategy_cls, **kwargs)
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
    avg_w  = tr_info.get("won",  {}).get("pnl", {}).get("average", 0.0) or 0.0
    avg_l  = tr_info.get("lost", {}).get("pnl", {}).get("average", 0.0) or 0.0

    # Expectancy
    wr  = won / total if total else 0
    exp = round((wr * avg_w) - ((1 - wr) * abs(avg_l)), 2)

    return {
        "pnl":          round(pnl, 2),
        "pnl_pct":      round(pnl_pct, 2),
        "max_drawdown": round(max_dd, 2),
        "sharpe":       round(sharpe, 4) if sharpe else None,
        "total_trades": total,
        "won_trades":   won,
        "win_rate_pct": round(win_rt, 2),
        "avg_win":      round(avg_w, 2),
        "avg_loss":     round(avg_l, 2),
        "expectancy":   exp,
    }


def run_ab_test(
    symbols: List[str] = None,
    versions: List[StrategyVersion] = None,
    period: str = "3y",
    silent: bool = True,
    save: bool = True,
) -> Dict[str, List[dict]]:
    """
    Run all versions on all symbols.

    Args:
        symbols:  Tickers to test
        versions: Strategy versions to compare (defaults to VERSIONS)
        period:   History period
        silent:   Suppress per-trade logs
        save:     Save results to ab_test_results.json

    Returns:
        dict: {symbol: [result_per_version, ...]}
    """
    symbols  = symbols or DATA_CONFIG["symbols"]
    versions = versions or VERSIONS
    all_results: Dict[str, List[dict]] = {}

    for symbol in symbols:
        logger.info("A/B test: %s", symbol)
        try:
            df = fetch_stock_data(symbol, period=period, save=False)
            df = compute_all_indicators(df)
        except Exception as exc:
            logger.error("Data fetch failed for %s: %s", symbol, exc)
            continue

        symbol_results = []
        for version in versions:
            try:
                metrics = _run_version(df, version, symbol, silent=silent)
                metrics["version"] = version.name
                metrics["symbol"]  = symbol
                symbol_results.append(metrics)

                if save:
                    save_ab_result(version.name, symbol, metrics)

            except Exception as exc:
                logger.error("Version %s failed for %s: %s", version.name, symbol, exc)
                symbol_results.append({
                    "version": version.name, "symbol": symbol, "error": str(exc)
                })

        all_results[symbol] = symbol_results

    return all_results


# ── Report printer ────────────────────────────────────────────────────────────

def print_ab_report(all_results: Dict[str, List[dict]], versions: List[StrategyVersion] = None) -> None:
    """Print a comprehensive A/B test report."""
    versions = versions or VERSIONS
    W = 100

    print("\n" + "=" * W)
    print("  A/B TEST RESULTS — STRATEGY VERSION COMPARISON")
    print("=" * W)

    # Print version legend
    print("\n  Versions tested:")
    for v in versions:
        print(f"    {v.name:<20} — {v.description}")

    # Per-symbol results
    for symbol, results in all_results.items():
        print(f"\n  {'─'*W}")
        print(f"  {symbol}")
        print(f"  {'─'*W}")
        print(
            f"  {'Version':<20} {'P&L%':>8} {'MaxDD%':>8} {'Win%':>7} "
            f"{'Trades':>7} {'Expect':>8} {'Sharpe':>8}"
        )
        print("  " + "─" * 70)

        for r in results:
            if "error" in r:
                print(f"  {r['version']:<20}  ERROR: {r['error']}")
                continue
            sign = "+" if r["pnl_pct"] >= 0 else ""
            exp_icon = "✅" if r["expectancy"] > 0 else "❌"
            sharpe_str = f"{r['sharpe']:.2f}" if r["sharpe"] else "N/A"
            print(
                f"  {r['version']:<20} {sign}{r['pnl_pct']:>7.2f}% "
                f"{r['max_drawdown']:>7.2f}% {r['win_rate_pct']:>6.1f}% "
                f"{r['total_trades']:>7} {r['expectancy']:>+7.2f} {exp_icon} "
                f"{sharpe_str:>7}"
            )

    # Aggregate winner
    print(f"\n  {'─'*W}")
    print("  AGGREGATE WINNER (avg across all symbols)")
    print(f"  {'─'*W}")

    import pandas as pd
    all_flat = [r for results in all_results.values() for r in results if "error" not in r]
    if all_flat:
        agg = pd.DataFrame(all_flat).groupby("version").agg(
            avg_pnl_pct=("pnl_pct", "mean"),
            avg_drawdown=("max_drawdown", "mean"),
            avg_win_rate=("win_rate_pct", "mean"),
            avg_trades=("total_trades", "mean"),
            avg_expectancy=("expectancy", "mean"),
        ).reset_index().sort_values("avg_expectancy", ascending=False)

        print(f"  {'Version':<20} {'Avg P&L%':>9} {'Avg DD%':>8} {'Avg Win%':>9} {'Avg Trades':>11} {'Avg Expect':>11}")
        print("  " + "─" * 72)
        for i, (_, row) in enumerate(agg.iterrows()):
            sign    = "+" if row["avg_pnl_pct"] >= 0 else ""
            winner  = "  ← WINNER" if i == 0 else ""
            print(
                f"  {str(row['version']):<20} {sign}{row['avg_pnl_pct']:>8.2f}% "
                f"{row['avg_drawdown']:>7.2f}% {row['avg_win_rate']:>8.1f}% "
                f"{row['avg_trades']:>11.1f} {row['avg_expectancy']:>+10.2f}{winner}"
            )

        winner_version = agg.iloc[0]["version"]
        winner_desc    = next((v.description for v in versions if v.name == winner_version), "")
        print(f"\n  Recommendation: Use {winner_version}")
        print(f"  Description   : {winner_desc}")

    print("=" * W + "\n")
