"""
Filter Analysis — "Which trades did the regime filter remove, and were they good?"

This is the core diagnostic tool. It runs both strategies on the same data,
captures every trade from each, then classifies each baseline trade as:

  KEPT     — regime strategy also took this trade
  REMOVED  — baseline took it, regime strategy skipped it
  ADDED    — regime strategy took it, baseline didn't (new entries)

For each category it reports: win rate, avg P&L, expectancy.

If REMOVED trades have POSITIVE expectancy → filter is hurting you.
If REMOVED trades have NEGATIVE expectancy → filter is helping you.
That's the answer.
"""

import logging
from dataclasses import dataclass, field
from typing import List, Dict, Tuple
from datetime import date, timedelta

import pandas as pd
import backtrader as bt

from data.fetcher import fetch_stock_data
from indicators.engine import compute_all_indicators
from config.settings import BACKTEST_CONFIG, DATA_CONFIG, EXECUTION_CONFIG

logger = logging.getLogger(__name__)

# Tolerance window: trades within N days of each other = same trade
MATCH_WINDOW_DAYS = 3


# ── Trade capture strategy ────────────────────────────────────────────────────

class TradeCaptureStrategy(bt.Strategy):
    """
    Wraps any strategy logic and captures every trade as a structured record.
    Used as a mixin base — subclass and implement _should_buy() / _should_sell().
    """

    params = (
        ("rsi_period",    14),
        ("ma_slow",       200),
        ("atr_period",    14),
        ("vol_ma_period", 20),
        ("rsi_buy",       40),
        ("rsi_sell",      70),
        ("vol_multiplier", 1.0),
        ("atr_sl_mult",   2.0),
        ("printlog",      False),   # accepted but ignored — keeps cerebro happy
    )

    def __init__(self):
        self.rsi    = bt.indicators.RSI(self.data.close, period=self.p.rsi_period)
        self.ma200  = bt.indicators.SMA(self.data.close, period=self.p.ma_slow)
        self.atr    = bt.indicators.ATR(self.data,       period=self.p.atr_period)
        self.vol_ma = bt.indicators.SMA(self.data.volume, period=self.p.vol_ma_period)

        self.order       = None
        self.stop_price  = None
        self.tp_price    = None
        self.entry_price = None
        self.entry_date  = None
        self.trades: List[dict] = []   # captured trade records

    def notify_order(self, order):
        if order.status in [order.Submitted, order.Accepted]:
            return
        if order.status == order.Completed:
            if order.isbuy():
                self.entry_price = order.executed.price
                self.entry_date  = self.datas[0].datetime.date(0)
                self.stop_price  = round(self.entry_price - self.atr[0] * self.p.atr_sl_mult, 2)
                self.tp_price    = round(self.entry_price + self.atr[0] * self.p.atr_sl_mult * 2, 2)
            else:
                exit_date = self.datas[0].datetime.date(0)
                pnl       = order.executed.pnl
                # Determine exit reason
                close = self.data.close[0]
                if self.stop_price and close <= self.stop_price:
                    reason = "STOP-LOSS"
                elif self.tp_price and close >= self.tp_price:
                    reason = "TAKE-PROFIT"
                else:
                    reason = "RSI-EXIT"

                self.trades.append({
                    "entry_date":  self.entry_date,
                    "exit_date":   exit_date,
                    "entry_price": round(self.entry_price, 2),
                    "exit_price":  round(order.executed.price, 2),
                    "pnl":         round(pnl, 2),
                    "win":         pnl > 0,
                    "exit_reason": reason,
                    "hold_days":   (exit_date - self.entry_date).days,
                })
                self.stop_price  = None
                self.tp_price    = None
                self.entry_price = None
                self.entry_date  = None
        self.order = None

    def next(self):
        if self.order:
            return
        vol_ok = self.data.volume[0] > self.vol_ma[0] * self.p.vol_multiplier
        if not self.position:
            if self._should_buy(vol_ok):
                self.order = self.buy()
        else:
            close  = self.data.close[0]
            hit_sl = self.stop_price and close <= self.stop_price
            hit_tp = self.tp_price   and close >= self.tp_price
            if hit_sl or hit_tp or self._should_sell():
                self.order = self.sell()

    def _should_buy(self, vol_ok: bool) -> bool:
        return (
            self.data.close[0] > self.ma200[0]
            and self.rsi[0] < self.p.rsi_buy
            and vol_ok
        )

    def _should_sell(self) -> bool:
        return self.rsi[0] > self.p.rsi_sell


class RegimeCaptureStrategy(TradeCaptureStrategy):
    """
    Same as TradeCaptureStrategy but with regime gating.
    Only enters in STRONG_TREND_UP.
    """

    params = (
        ("rsi_period",        14),
        ("ma_slow",           200),
        ("atr_period",        14),
        ("vol_ma_period",     20),
        ("rsi_buy",           40),    # kept for base class compat (unused here)
        ("rsi_sell",          70),
        ("vol_multiplier",    1.0),
        ("atr_sl_mult",       2.0),
        ("printlog",          False),
        # Regime-specific
        ("ma_fast",           50),
        ("rsi_buy_strong",    35),
        ("slope_strong_up",   0.0015),
        ("atr_ratio_thresh",  1.4),
        ("cooldown_bars",     5),
        ("min_hold_bars",     3),
    )

    def __init__(self):
        super().__init__()
        self.ma50          = bt.indicators.SMA(self.data.close, period=self.p.ma_fast)
        self.cooldown_left = 0
        self.bars_in_trade = 0

    def _detect_strong_trend(self) -> bool:
        """Returns True only if regime is STRONG_TREND_UP."""
        if len(self.ma50) < 55:
            return False
        atr_vals  = [self.atr[-i] for i in range(min(50, len(self.atr)))]
        atr_avg   = sum(atr_vals) / len(atr_vals) if atr_vals else 1.0
        atr_ratio = self.atr[0] / atr_avg if atr_avg > 0 else 1.0
        if atr_ratio > self.p.atr_ratio_thresh:
            return False   # VOLATILE — not STRONG_TREND_UP
        ma_vals    = [self.ma50[-i] for i in range(self.p.ma_fast + 1)]
        diffs      = [ma_vals[i] - ma_vals[i + 1] for i in range(self.p.ma_fast)]
        slope_norm = (sum(diffs[-5:]) / 5) / self.ma50[0] if self.ma50[0] else 0
        price      = self.data.close[0]
        # Uptrend: price above MA200 (allows pullbacks below MA50)
        above_ma200 = price > self.ma200[0]
        below_both  = price < self.ma50[0] and price < self.ma200[0]
        return (above_ma200 and slope_norm >= self.p.slope_strong_up)

    def notify_order(self, order):
        super().notify_order(order)
        if order.status == order.Completed and not order.isbuy():
            self.cooldown_left = self.p.cooldown_bars

    def next(self):
        if self.order:
            return
        if self.cooldown_left > 0:
            self.cooldown_left -= 1
        vol_ok = self.data.volume[0] > self.vol_ma[0] * self.p.vol_multiplier
        if not self.position:
            if self.cooldown_left == 0 and self._should_buy(vol_ok):
                self.order = self.buy()
        else:
            self.bars_in_trade += 1
            close  = self.data.close[0]
            hit_sl = self.stop_price and close <= self.stop_price
            hit_tp = self.tp_price   and close >= self.tp_price
            if hit_sl:
                self.order = self.sell()
                return
            if self.bars_in_trade < self.p.min_hold_bars:
                return
            if hit_tp or self._should_sell():
                self.order = self.sell()

    def _should_buy(self, vol_ok: bool) -> bool:
        return (
            self._detect_strong_trend()
            and self.rsi[0] < self.p.rsi_buy_strong
            and vol_ok
        )


# ── Trade capture runner ──────────────────────────────────────────────────────

def _capture_trades(df: pd.DataFrame, strategy_cls, params: dict) -> List[dict]:
    """Run a strategy and return its trade list."""
    cerebro = bt.Cerebro()
    cerebro.addstrategy(strategy_cls, printlog=False, **params)
    cerebro.adddata(bt.feeds.PandasData(dataname=df))
    cerebro.broker.setcash(BACKTEST_CONFIG["initial_cash"])
    cerebro.broker.setcommission(commission=BACKTEST_CONFIG["commission"])
    cerebro.addsizer(bt.sizers.FixedSize, stake=BACKTEST_CONFIG["stake"])
    results = cerebro.run()
    return results[0].trades


# ── Trade matching ────────────────────────────────────────────────────────────

def _match_trades(
    baseline_trades: List[dict],
    regime_trades: List[dict],
    window: int = MATCH_WINDOW_DAYS,
) -> Tuple[List[dict], List[dict], List[dict]]:
    """
    Match baseline trades to regime trades by entry date proximity.

    Returns:
        kept    — baseline trades also taken by regime strategy
        removed — baseline trades NOT taken by regime strategy
        added   — regime trades NOT in baseline
    """
    regime_dates = [t["entry_date"] for t in regime_trades]

    kept    = []
    removed = []

    for bt_trade in baseline_trades:
        bd = bt_trade["entry_date"]
        # Check if any regime trade is within window days
        matched = any(
            abs((bd - rd).days) <= window
            for rd in regime_dates
        )
        if matched:
            kept.append(bt_trade)
        else:
            removed.append(bt_trade)

    baseline_dates = [t["entry_date"] for t in baseline_trades]
    added = [
        rt for rt in regime_trades
        if not any(abs((rt["entry_date"] - bd).days) <= window for bd in baseline_dates)
    ]

    return kept, removed, added


# ── Metrics ───────────────────────────────────────────────────────────────────

def _trade_metrics(trades: List[dict], label: str) -> dict:
    if not trades:
        return {
            "label": label, "count": 0, "wins": 0, "win_rate": 0,
            "total_pnl": 0, "avg_pnl": 0, "avg_win": 0, "avg_loss": 0,
            "expectancy": 0,
        }
    wins   = [t for t in trades if t["win"]]
    losses = [t for t in trades if not t["win"]]
    wr     = len(wins) / len(trades)
    avg_w  = sum(t["pnl"] for t in wins)   / len(wins)   if wins   else 0
    avg_l  = sum(t["pnl"] for t in losses) / len(losses) if losses else 0
    exp    = (wr * avg_w) - ((1 - wr) * abs(avg_l))
    return {
        "label":     label,
        "count":     len(trades),
        "wins":      len(wins),
        "win_rate":  round(wr * 100, 1),
        "total_pnl": round(sum(t["pnl"] for t in trades), 2),
        "avg_pnl":   round(sum(t["pnl"] for t in trades) / len(trades), 2),
        "avg_win":   round(avg_w, 2),
        "avg_loss":  round(avg_l, 2),
        "expectancy": round(exp, 2),
    }


# ── Main analysis ─────────────────────────────────────────────────────────────

def run_filter_analysis(
    symbols: List[str] = None,
    period: str = "3y",
) -> List[dict]:
    """
    Run filter analysis for each symbol.

    Returns:
        List of analysis dicts per symbol with kept/removed/added metrics
    """
    symbols = symbols or DATA_CONFIG["symbols"]
    results = []

    for symbol in symbols:
        logger.info("Filter analysis: %s", symbol)
        try:
            df = fetch_stock_data(symbol, period=period, save=False)
            df = compute_all_indicators(df)

            baseline_trades = _capture_trades(df, TradeCaptureStrategy,  {"rsi_buy": 40})
            regime_trades   = _capture_trades(df, RegimeCaptureStrategy, {
                "cooldown_bars": EXECUTION_CONFIG["cooldown_days"],
                "min_hold_bars": EXECUTION_CONFIG["min_hold_bars"],
            })

            kept, removed, added = _match_trades(baseline_trades, regime_trades)

            results.append({
                "symbol":  symbol,
                "all":     _trade_metrics(baseline_trades, "ALL BASELINE"),
                "kept":    _trade_metrics(kept,    "KEPT by regime"),
                "removed": _trade_metrics(removed, "REMOVED by regime"),
                "added":   _trade_metrics(added,   "ADDED by regime"),
                "verdict": _verdict(removed, kept),
                "removed_trades": removed,   # full trade list for detail view
                "added_trades":   added,
            })

        except Exception as exc:
            logger.error("Filter analysis failed for %s: %s", symbol, exc)
            results.append({"symbol": symbol, "error": str(exc)})

    return results


def _verdict(removed: List[dict], kept: List[dict]) -> str:
    """
    Determine whether the regime filter is helping or hurting.
    """
    if not removed:
        return "NEUTRAL — no trades removed"

    removed_exp = _trade_metrics(removed, "")["expectancy"]
    kept_exp    = _trade_metrics(kept,    "")["expectancy"] if kept else 0

    if removed_exp > 50:
        return f"⚠️  FILTER HURTING — removed trades had +{removed_exp:.0f} expectancy"
    elif removed_exp > 0:
        return f"⚠️  FILTER SLIGHTLY HURTING — removed trades had +{removed_exp:.0f} expectancy"
    elif removed_exp < -50:
        return f"✅  FILTER HELPING — removed trades had {removed_exp:.0f} expectancy (bad trades blocked)"
    elif removed_exp < 0:
        return f"✅  FILTER SLIGHTLY HELPING — removed trades had {removed_exp:.0f} expectancy"
    else:
        return "NEUTRAL — removed trades had near-zero expectancy"


# ── Report printer ────────────────────────────────────────────────────────────

def print_filter_analysis(results: List[dict]) -> None:
    """Print the filter analysis report."""
    W = 80
    print("\n" + "=" * W)
    print("  FILTER ANALYSIS — Which baseline trades did the regime filter remove?")
    print("  If REMOVED trades have positive expectancy → filter is hurting you.")
    print("  If REMOVED trades have negative expectancy → filter is helping you.")
    print("=" * W)

    for r in results:
        if "error" in r:
            print(f"\n  {r['symbol']}: ERROR — {r['error']}")
            continue

        print(f"\n  ── {r['symbol']} {'─'*(W-6-len(r['symbol']))}")
        print(f"  Verdict: {r['verdict']}")
        print()

        # Metrics table
        header = f"  {'Category':<24} {'Trades':>7} {'Win%':>7} {'Avg P&L':>9} {'Expect':>9}"
        print(header)
        print("  " + "─" * 60)

        for key in ("all", "kept", "removed", "added"):
            m = r[key]
            if m["count"] == 0:
                print(f"  {m['label']:<24} {'—':>7}")
                continue
            exp_icon = "✅" if m["expectancy"] > 0 else ("❌" if m["expectancy"] < 0 else "—")
            sign = "+" if m["avg_pnl"] >= 0 else ""
            print(
                f"  {m['label']:<24} {m['count']:>7} {m['win_rate']:>6.1f}% "
                f"{sign}{m['avg_pnl']:>8,.2f} {m['expectancy']:>+8.2f} {exp_icon}"
            )

        # Show removed trades detail
        if r["removed_trades"]:
            print(f"\n  Removed trades detail ({len(r['removed_trades'])} trades):")
            print(f"  {'Entry':>12} {'Exit':>12} {'Entry ₹':>10} {'Exit ₹':>10} {'P&L':>9} {'Days':>5} {'Reason'}")
            print("  " + "─" * 68)
            for t in sorted(r["removed_trades"], key=lambda x: x["entry_date"]):
                sign = "+" if t["pnl"] >= 0 else ""
                icon = "✅" if t["win"] else "❌"
                print(
                    f"  {str(t['entry_date']):>12} {str(t['exit_date']):>12} "
                    f"{t['entry_price']:>10.2f} {t['exit_price']:>10.2f} "
                    f"{sign}{t['pnl']:>8,.2f} {t['hold_days']:>5}  {t['exit_reason']} {icon}"
                )

    # Cross-symbol summary
    valid = [r for r in results if "error" not in r]
    if len(valid) > 1:
        print(f"\n  {'─'*W}")
        print("  CROSS-SYMBOL SUMMARY")
        print(f"  {'─'*W}")

        all_removed = [t for r in valid for t in r["removed_trades"]]
        all_added   = [t for r in valid for t in r["added_trades"]]

        if all_removed:
            rm = _trade_metrics(all_removed, "All removed")
            exp_icon = "✅" if rm["expectancy"] < 0 else "⚠️ "
            print(f"  All REMOVED trades: {rm['count']} trades | "
                  f"Win: {rm['win_rate']:.1f}% | Avg P&L: {rm['avg_pnl']:+,.2f} | "
                  f"Expectancy: {rm['expectancy']:+.2f} {exp_icon}")

        if all_added:
            ad = _trade_metrics(all_added, "All added")
            exp_icon = "✅" if ad["expectancy"] > 0 else "⚠️ "
            print(f"  All ADDED trades  : {ad['count']} trades | "
                  f"Win: {ad['win_rate']:.1f}% | Avg P&L: {ad['avg_pnl']:+,.2f} | "
                  f"Expectancy: {ad['expectancy']:+.2f} {exp_icon}")

        # Final recommendation
        if all_removed:
            rm_exp = _trade_metrics(all_removed, "")["expectancy"]
            print(f"\n  RECOMMENDATION:")
            if rm_exp > 20:
                print(f"  The regime filter is removing profitable trades (expectancy {rm_exp:+.2f}).")
                print(f"  Consider: relax slope threshold OR use baseline RSI=40 for STRONG_TREND_UP.")
            elif rm_exp < -20:
                print(f"  The regime filter is correctly blocking bad trades (expectancy {rm_exp:+.2f}).")
                print(f"  Keep the filter. Focus on improving entry quality within allowed regimes.")
            else:
                print(f"  The regime filter has minimal impact on trade quality (expectancy {rm_exp:+.2f}).")
                print(f"  The filter is reducing exposure without clear benefit or harm.")
                print(f"  Consider: widen slope threshold to allow more STRONG_TREND_UP entries.")

    print("=" * W + "\n")
