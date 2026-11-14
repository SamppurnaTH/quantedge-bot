"""
Paper Trader — realistic, bar-by-bar execution simulation.

How it works
  Morning run  : process_signals() turns approved BUY signals into PENDING
                 next-open orders (respecting slots, sector cap, cooldown, cash).
  Any run      : update() replays every new daily bar for every pending order
                 and open position using execution/trade_model.py:
                   - pending orders fill at the first bar AFTER the signal bar
                     (its open + slippage), or are cancelled (gap-up chase /
                     opened through stop)
                   - stops/targets are checked against the bar's LOW/HIGH, with
                     gap-through handling at the open
                   - RSI-overbought exit after the minimum hold, time stop
                   - statutory costs + DP charges are charged to cash
  Every closed trade records its R-multiple and feeds the learning journal.

Fixes vs the previous version
  * Entries were filled at the PREVIOUS day's close at 8:30 AM — a price no
    longer available. Now they fill at the next open.
  * Exits only looked at the close, so intraday stop/target hits were missed;
    open_price was never supplied so gap detection never ran.
  * Exits were only checked for symbols in the scan list — a held stock that
    left the list (ADANIPOWER) could never be stopped out.
  * Regime size multiplier was applied twice (risk manager AND here), and a
    minimum of 1 share was forced even when the risk budget said 0.
  * No transaction costs; drawdown was never tracked (equity was never
    marked between runs); missed runs silently skipped bars.
"""

import os
import json
import logging
from datetime import datetime
from typing import Callable, List, Optional

import numpy as np
import pandas as pd

from config.settings import EXECUTION_CONFIG, PORTFOLIO_CONFIG, INDICATOR_CONFIG
from execution import trade_model

logger = logging.getLogger(__name__)

PAPER_STATE_FILE = os.path.join("logs", "paper_portfolio.json")
STATE_VERSION = 2

# Kept for backward compatibility with imports elsewhere
SLIPPAGE_PCT  = EXECUTION_CONFIG["slippage_pct"]
GAP_THRESHOLD = EXECUTION_CONFIG["gap_threshold"]


def _default_loader(symbol: str) -> pd.DataFrame:
    from data.fetcher import load_stock_data
    return load_stock_data(symbol)


def _sector_of(symbol: str) -> str:
    try:
        from analytics.intelligence_cycles import SYMBOL_SECTORS
        return SYMBOL_SECTORS.get(symbol, "Others")
    except Exception:
        return "Others"


class PaperTrader:
    """Simulated brokerage account with realistic next-open execution."""

    def __init__(self, capital: float = 100_000.0, loader: Optional[Callable] = None):
        self.initial_capital = capital
        self.cash            = capital
        self.positions: dict = {}
        self.pending_orders: dict = {}
        self.closed_trades: list = []
        self.cancelled_orders: list = []
        self.equity_curve: list = []
        self.last_exit: dict = {}
        self.max_portfolio_value = capital
        self.max_drawdown = 0.0
        self.loader = loader or _default_loader
        self._load_state()

    # ── Account values ────────────────────────────────────────────────────────

    def equity(self) -> float:
        return self.cash + sum(p["shares"] * p["current_price"] for p in self.positions.values())

    def reserved_cash(self) -> float:
        return sum(o["shares"] * o["signal_close"] * 1.01 for o in self.pending_orders.values())

    def free_cash(self) -> float:
        return self.cash - self.reserved_cash()

    def slots_used(self) -> int:
        return len(self.positions) + len(self.pending_orders)

    # ── Order entry ───────────────────────────────────────────────────────────

    def process_signals(self, signals: List[dict]) -> List[dict]:
        """
        Queue next-open BUY orders for approved signals, best expected R first.
        Returns the list of orders placed. Exits are handled by update().
        """
        placed = []
        buys = [s for s in signals
                if "error" not in s and s.get("signal") == "BUY" and s.get("risk_report", {}).get("approved")]
        buys.sort(key=lambda s: (s.get("expectancy_r", 0.0), s.get("confidence_score", 0)), reverse=True)

        max_slots = PORTFOLIO_CONFIG["max_active_trades"]
        sector_cap = PORTFOLIO_CONFIG["max_sector_exposure_pct"] / 100.0
        eq = self.equity()

        for s in buys:
            sym = s["symbol"]
            rr = s["risk_report"]
            why = None
            if sym in self.positions or sym in self.pending_orders:
                why = "already held / pending"
            elif self.slots_used() >= max_slots:
                why = f"max active trades ({max_slots})"
            elif self._in_cooldown(sym, s.get("date")):
                why = f"cooldown ({EXECUTION_CONFIG['cooldown_days']} bars after last exit)"
            else:
                cost = rr["shares"] * s["close"]
                sector = _sector_of(sym)
                sec_val = sum(p["shares"] * p["current_price"] for p in self.positions.values()
                              if _sector_of(p["symbol"]) == sector)
                sec_val += sum(o["shares"] * o["signal_close"] for o in self.pending_orders.values()
                               if _sector_of(o["symbol"]) == sector)
                if (sec_val + cost) > sector_cap * eq:
                    why = f"sector cap {sector} {sector_cap:.0%}"
                elif cost * 1.01 > self.free_cash():
                    why = "insufficient free cash"
            if why:
                logger.info("ORDER SKIP | %s | %s", sym, why)
                continue

            order = {
                "symbol": sym,
                "signal_date": str(s.get("date"))[:10],
                "signal_close": float(s["close"]),
                "atr": float(rr.get("atr_used") or s.get("atr") or 0.0),
                "sl_mult": float(rr.get("sl_multiplier", 2.0)),
                "shares": int(rr["shares"]),
                "regime": s.get("regime", "?"),
                "score": s.get("score", 0),
                "rsi": s.get("rsi", 50.0),
                "confidence_score": s.get("confidence_score"),
                "expected_r": s.get("expectancy_r"),
                "pattern_key": s.get("pattern_key"),
                "pattern_snapshot": s.get("pattern_snapshot", {}),
                "placed_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
            }
            self.pending_orders[sym] = order
            placed.append(order)
            print(f"  📝 ORDER PLACED | {sym} | BUY {order['shares']} @ next open "
                  f"(signal close {order['signal_close']:.2f}, ATR {order['atr']:.2f}, {order['regime']})")
        self._save_state()
        return placed

    def _in_cooldown(self, symbol: str, signal_date) -> bool:
        last = self.last_exit.get(symbol)
        if not last or not signal_date:
            return False
        try:
            gap = np.busday_count(str(last)[:10], str(signal_date)[:10])
        except Exception:
            return False
        return gap < EXECUTION_CONFIG["cooldown_days"]

    # ── Bar-by-bar update ─────────────────────────────────────────────────────

    def update(self) -> None:
        """Fill pending orders and manage open positions over all new bars."""
        symbols = sorted(set(self.positions) | set(self.pending_orders))
        last_date = None
        for sym in symbols:
            try:
                df = self._bars(sym)
            except Exception as exc:
                logger.error("UPDATE | %s | could not load bars: %s", sym, exc)
                continue
            if df.empty:
                continue
            last_date = max(last_date, df.index[-1]) if last_date is not None else df.index[-1]
            if sym in self.pending_orders:
                self._try_fill(sym, df)
            if sym in self.positions:
                self._manage(sym, df)
        self._mark_equity(last_date)
        self._save_state()

    def _bars(self, sym: str) -> pd.DataFrame:
        from indicators.rsi import compute_rsi
        df = self.loader(sym)
        df = df[["Open", "High", "Low", "Close"]].copy()
        df.index = pd.to_datetime(df.index)
        df["RSI"] = compute_rsi(df["Close"], INDICATOR_CONFIG["rsi_period"]) if len(df) > 15 else np.nan
        return df

    def _try_fill(self, sym: str, df: pd.DataFrame) -> None:
        o = self.pending_orders[sym]
        after = df[df.index > pd.Timestamp(o["signal_date"])]
        if after.empty:
            return                                           # next session not traded yet
        bar = after.iloc[0]
        plan = trade_model.plan_entry(o["signal_close"], float(bar["Open"]), o["atr"], o["sl_mult"])
        del self.pending_orders[sym]
        if plan is None:
            reason = "gap-up beyond chase limit / opened below stop"
            self.cancelled_orders.append({**o, "cancel_date": str(after.index[0].date()), "reason": reason})
            del self.cancelled_orders[:-200]
            print(f"  ✖ ORDER CANCELLED | {sym} | open {bar['Open']:.2f} | {reason}")
            return
        shares = o["shares"]
        cost = plan["entry_price"] * shares * (1 + trade_model.COST_BUY)
        if cost > self.cash:
            shares = int(self.cash / (plan["entry_price"] * (1 + trade_model.COST_BUY)))
            if shares <= 0:
                print(f"  ✖ ORDER CANCELLED | {sym} | insufficient cash at fill")
                return
            cost = plan["entry_price"] * shares * (1 + trade_model.COST_BUY)
        self.cash -= cost
        entry_date = str(after.index[0].date())
        self.positions[sym] = {
            **{k: o[k] for k in ("symbol", "regime", "score", "rsi", "confidence_score",
                                 "expected_r", "pattern_key", "pattern_snapshot", "atr")},
            "shares": shares,
            "entry_price": plan["entry_price"],
            "quoted_price": plan["quoted_price"],
            "current_price": float(bar["Close"]),
            "stop_loss": round(plan["stop_loss"], 2),
            "take_profit": round(plan["take_profit"], 2),
            "initial_risk": plan["initial_risk"],
            "entry_date": entry_date,
            "signal_date": o["signal_date"],
            "last_bar_date": None,                 # entry bar is processed by _manage
            "bars_held": 0,
            "entry_costs": round(cost - plan["entry_price"] * shares, 2),
        }
        print(f"  📥 FILLED  | {sym} | {shares} @ {plan['entry_price']:.2f} on {entry_date} "
              f"| SL {plan['stop_loss']:.2f} | TP {plan['take_profit']:.2f}")

    def _manage(self, sym: str, df: pd.DataFrame) -> None:
        pos = self.positions[sym]
        self._migrate_position(pos, df)
        start = pd.Timestamp(pos["last_bar_date"]) if pos.get("last_bar_date") else pd.Timestamp(pos["entry_date"]) - pd.Timedelta(days=1)
        new_bars = df[df.index > start]
        for ts, bar in new_bars.iterrows():
            pos["bars_held"] = pos.get("bars_held", 0) + 1
            is_entry = str(ts.date()) == pos["entry_date"] and pos["bars_held"] == 1
            rsi = None if pd.isna(bar["RSI"]) else float(bar["RSI"])
            res = trade_model.step(pos, float(bar["Open"]), float(bar["High"]), float(bar["Low"]),
                                   float(bar["Close"]), rsi, pos["bars_held"], is_entry_bar=is_entry)
            pos["current_price"] = float(bar["Close"])
            pos["last_bar_date"] = str(ts.date())
            if res is not None:
                self._close_position(sym, res[0], res[1], str(ts.date()))
                return

    @staticmethod
    def _migrate_position(pos: dict, df: pd.DataFrame) -> None:
        """Bring positions written by the old trader up to the v2 schema."""
        if "initial_risk" in pos and "bars_held" in pos:
            return
        pos["initial_risk"] = max(pos["entry_price"] - pos["stop_loss"], 1e-6)
        pos["signal_date"] = pos.get("signal_date", pos["entry_date"])
        pos["bars_held"] = 0
        pos["last_bar_date"] = None
        # Replay every bar since entry, so a legacy position that was never
        # checked (e.g. a symbol no longer in the scan list) gets its real exit.
        held = df[df.index >= pd.Timestamp(pos["entry_date"])]
        if not held.empty:
            pos["entry_date"] = str(held.index[0].date())

    def _close_position(self, symbol: str, raw_price: float, reason: str, exit_date: str) -> None:
        pos = self.positions.pop(symbol)
        xf = trade_model.exit_fill(raw_price)
        shares = pos["shares"]
        pnl = trade_model.net_pnl(pos["entry_price"], xf, shares, include_dp=True)
        r = pnl / (pos["initial_risk"] * shares) if pos["initial_risk"] > 0 and shares > 0 else 0.0
        self.cash += xf * shares * (1 - trade_model.COST_SELL) - trade_model.DP_CHARGE
        self.last_exit[symbol] = exit_date

        trade = {
            "symbol": symbol, "shares": shares,
            "entry_price": round(pos["entry_price"], 2), "exit_price": round(xf, 2),
            "stop_loss": pos["stop_loss"], "take_profit": pos["take_profit"],
            "entry_date": pos["entry_date"], "exit_date": exit_date,
            "pnl": round(pnl, 2), "r_multiple": round(r, 3), "exit_reason": reason,
            "bars_held": pos.get("bars_held", 0),
            "score": pos.get("score", 0), "regime": pos.get("regime", "?"),
            "confidence_score": pos.get("confidence_score"), "expected_r": pos.get("expected_r"),
            "pattern_key": pos.get("pattern_key"),
            "slippage_cost": round((pos["entry_price"] - pos.get("quoted_price", pos["entry_price"])) * shares
                                   + (raw_price - xf) * shares, 2),
        }
        self.closed_trades.append(trade)
        sign = "+" if pnl >= 0 else ""
        logger.info("PAPER SELL | %s | %s | PnL %s%.2f | %+.2fR", symbol, reason, sign, pnl, r)
        print(f"  📤 EXIT    | {symbol} | {reason} on {exit_date} | PnL {sign}{pnl:.2f} ({r:+.2f}R)")

        try:
            from analytics.learning_engine import record_live_trade
            record_live_trade(
                symbol=symbol,
                signal_result={k: pos.get(k) for k in ("regime", "score", "rsi", "confidence_score",
                                                       "pattern_key", "pattern_snapshot")},
                won=pnl > 0, pnl=pnl, hold_time=pos.get("bars_held", 0), trade=trade,
            )
        except Exception as e:
            logger.warning("Failed to record trade to learning journal: %s", e)

    def _mark_equity(self, as_of) -> None:
        eq = self.equity()
        if as_of is None:                 # nothing held / no new bars → nothing to mark
            return
        date = str(pd.Timestamp(as_of).date())
        if self.equity_curve and self.equity_curve[-1]["date"] == date:
            self.equity_curve[-1]["equity"] = round(eq, 2)
        else:
            self.equity_curve.append({"date": date, "equity": round(eq, 2)})
        self.max_portfolio_value = max(self.max_portfolio_value, eq)
        dd = (self.max_portfolio_value - eq) / self.max_portfolio_value * 100 if self.max_portfolio_value > 0 else 0.0
        self.max_drawdown = max(self.max_drawdown, dd)

    # ── Reporting ─────────────────────────────────────────────────────────────

    def portfolio_summary(self) -> dict:
        open_value  = sum(p["shares"] * p["current_price"] for p in self.positions.values())
        total_value = self.cash + open_value
        total_pnl   = total_value - self.initial_capital

        ct = self.closed_trades
        wins   = [t for t in ct if t["pnl"] > 0]
        losses = [t for t in ct if t["pnl"] <= 0]
        n = len(ct)
        win_rate = len(wins) / n * 100 if n else 0.0
        avg_win  = sum(t["pnl"] for t in wins) / len(wins) if wins else 0.0
        avg_loss = sum(abs(t["pnl"]) for t in losses) / len(losses) if losses else 0.0
        gp, gl = sum(t["pnl"] for t in wins), sum(abs(t["pnl"]) for t in losses)
        rs = [t["r_multiple"] for t in ct if t.get("r_multiple") is not None]

        streak = max_streak = 0
        for t in ct:
            streak = streak + 1 if t["pnl"] <= 0 else 0
            max_streak = max(max_streak, streak)

        holds = []
        for t in ct:
            try:
                holds.append(max((datetime.strptime(t["exit_date"], "%Y-%m-%d")
                                  - datetime.strptime(t["entry_date"], "%Y-%m-%d")).days, 1))
            except Exception:
                pass

        regime_perf = {}
        for t in ct:
            regime_perf[t.get("regime", "UNKNOWN")] = round(regime_perf.get(t.get("regime", "UNKNOWN"), 0.0) + t["pnl"], 2)

        return {
            "cash": round(self.cash, 2),
            "open_value": round(open_value, 2),
            "total_value": round(total_value, 2),
            "total_pnl": round(total_pnl, 2),
            "total_pnl_pct": round(total_pnl / self.initial_capital * 100, 2),
            "closed_pnl": round(sum(t["pnl"] for t in ct), 2),
            "open_positions": len(self.positions),
            "pending_orders": len(self.pending_orders),
            "closed_trades": n,
            "wins": len(wins), "losses": len(losses),
            "win_rate_pct": round(win_rate, 2),
            "avg_win": round(avg_win, 2), "avg_loss": round(avg_loss, 2),
            "expectancy": round((win_rate / 100) * avg_win - (1 - win_rate / 100) * avg_loss, 2),
            "expectancy_r": round(float(np.mean(rs)), 3) if rs else 0.0,
            "exposure_pct": round(open_value / total_value * 100, 2) if total_value > 0 else 0.0,
            "profit_factor": round(gp / gl, 2) if gl > 0 else (round(gp, 2) if gp > 0 else 1.0),
            "max_drawdown": round(self.max_drawdown, 2),
            "max_consecutive_losses": max_streak,
            "avg_holding_days": round(float(np.mean(holds)), 1) if holds else 0.0,
            "regime_performance": regime_perf,
        }

    def print_portfolio(self) -> None:
        s = self.portfolio_summary()
        sign = "+" if s["total_pnl"] >= 0 else ""
        print("\n" + "=" * 65)
        print("  PAPER PORTFOLIO STATUS")
        print("=" * 65)
        print(f"  Cash           : {s['cash']:>12,.2f}")
        print(f"  Open Value     : {s['open_value']:>12,.2f}")
        print(f"  Total Value    : {s['total_value']:>12,.2f}")
        print(f"  Total P&L      : {sign}{s['total_pnl']:>11,.2f}  ({sign}{s['total_pnl_pct']:.2f}%)")
        print(f"  Exposure       : {s['exposure_pct']:>11.1f}%     Max Drawdown: {s['max_drawdown']:.2f}%")
        print(f"  Profit Factor  : {s['profit_factor']:>12.2f}  Max Consecutive Losses: {s['max_consecutive_losses']}")
        print(f"  Closed Trades  : {s['closed_trades']}  (W:{s['wins']} / L:{s['losses']})  Win Rate: {s['win_rate_pct']:.1f}%")
        print(f"  Expectancy     : {s['expectancy_r']:+.2f}R per trade  (₹{s['expectancy']:+,.2f})")
        print(f"  Pending Orders : {s['pending_orders']}   Avg Hold: {s['avg_holding_days']} days")

        if self.pending_orders:
            print("\n  Pending next-open orders:")
            for sym, o in self.pending_orders.items():
                print(f"  {sym:<18} BUY {o['shares']:>5}  signal {o['signal_date']} close {o['signal_close']:.2f}")
        if self.positions:
            print("\n  Open Positions:")
            print(f"  {'Symbol':<18} {'Shares':>6} {'Entry':>9} {'Current':>9} {'SL':>9} {'TP':>9} {'Unreal P&L':>12} {'Bars':>4}")
            print("  " + "-" * 82)
            for sym, p in self.positions.items():
                unreal = (p["current_price"] - p["entry_price"]) * p["shares"]
                print(f"  {sym:<18} {p['shares']:>6} {p['entry_price']:>9.2f} {p['current_price']:>9.2f} "
                      f"{p['stop_loss']:>9.2f} {p['take_profit']:>9.2f} {unreal:>+12,.2f} {p.get('bars_held', 0):>4}")
        if self.closed_trades:
            print("\n  Recent Closed Trades (last 5):")
            for t in self.closed_trades[-5:]:
                r = t.get("r_multiple")
                r_txt = f"{r:+.2f}R" if r is not None else "  n/a"
                print(f"  {t['symbol']:<18} {t['exit_date']:<12} {t['pnl']:>+10,.2f} {r_txt:>7}  {t['exit_reason']}")
        print("=" * 65 + "\n")

    # ── Persistence ───────────────────────────────────────────────────────────

    def _save_state(self) -> None:
        os.makedirs(os.path.dirname(PAPER_STATE_FILE), exist_ok=True)
        state = {
            "version": STATE_VERSION,
            "initial_capital": self.initial_capital,
            "cash": self.cash,
            "positions": self.positions,
            "pending_orders": self.pending_orders,
            "closed_trades": self.closed_trades,
            "cancelled_orders": self.cancelled_orders,
            "equity_curve": self.equity_curve[-2000:],
            "last_exit": self.last_exit,
            "max_portfolio_value": self.max_portfolio_value,
            "max_drawdown": self.max_drawdown,
        }
        tmp = PAPER_STATE_FILE + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(state, f, indent=2)
        os.replace(tmp, PAPER_STATE_FILE)

    def _load_state(self) -> None:
        if not os.path.exists(PAPER_STATE_FILE):
            return
        try:
            with open(PAPER_STATE_FILE, "r", encoding="utf-8") as f:
                state = json.load(f)
            self.initial_capital  = state.get("initial_capital", self.initial_capital)
            self.cash             = state.get("cash", self.cash)
            self.positions        = state.get("positions", {})
            self.pending_orders   = state.get("pending_orders", {})
            self.closed_trades    = state.get("closed_trades", [])
            self.cancelled_orders = state.get("cancelled_orders", [])
            self.equity_curve     = state.get("equity_curve", [])
            self.last_exit        = state.get("last_exit", {})
            self.max_portfolio_value = state.get("max_portfolio_value", self.initial_capital)
            self.max_drawdown     = state.get("max_drawdown", 0.0)
            if not self.last_exit:
                for t in self.closed_trades:
                    self.last_exit[t["symbol"]] = t.get("exit_date")
            logger.info("Paper portfolio loaded | Cash: %.2f | Open: %d | Pending: %d | Closed: %d",
                        self.cash, len(self.positions), len(self.pending_orders), len(self.closed_trades))
        except Exception as exc:
            logger.warning("Could not load paper portfolio: %s", exc)
