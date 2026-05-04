"""
Paper Trader — Realistic execution simulation

Execution realism upgrades:
  1. Slippage simulation  — entry/exit price is nudged by a configurable %
  2. Gap handling         — if price gaps below SL, exit at open (not SL)
  3. Regime-aware sizing  — position size multiplied by regime.position_size_multiplier
  4. Persistent state     — survives restarts via JSON

Tracks open positions, closed trades, equity curve, and full audit trail.
"""

import os
import json
import logging
from datetime import datetime
from typing import List, Optional

logger = logging.getLogger(__name__)

PAPER_STATE_FILE = os.path.join("logs", "paper_portfolio.json")

# ── Execution realism constants ───────────────────────────────────────────────
SLIPPAGE_PCT   = 0.001   # 0.1% slippage on every fill (buy higher, sell lower)
GAP_THRESHOLD  = 0.005   # if price gaps > 0.5% below SL → gap exit at open price


class PaperTrader:
    """
    Simulates a brokerage account with realistic execution.
    Call process_signals() each day with fresh signal + OHLC data.
    """

    def __init__(self, capital: float = 100_000.0):
        self.initial_capital = capital
        self.cash            = capital
        self.positions: dict = {}
        self.closed_trades: list = []
        self._load_state()

    # ── Public API ────────────────────────────────────────────────────────────

    def process_signals(self, signals: List[dict]) -> None:
        """
        Process a list of signal dicts (output of scan_watchlist).

        Each signal dict may optionally include:
          - 'open_price': today's open (for gap detection)
          - 'regime':     current regime string (for position sizing)
        """
        for s in signals:
            if "error" in s:
                continue

            symbol     = s["symbol"]
            signal     = s.get("signal", "HOLD")
            close      = s.get("close", 0)
            open_price = s.get("open_price", close)   # fallback to close if not provided
            rr         = s.get("risk_report", {})
            regime_str = s.get("regime", "TRENDING_UP")

            # ── Check exits on existing positions first ────────────────────
            if symbol in self.positions:
                self._check_exit(symbol, close, open_price, signal)

            # ── Check entry ────────────────────────────────────────────────
            if signal == "BUY" and rr.get("approved") and symbol not in self.positions:
                self._open_position(symbol, close, rr, s, regime_str)

        self._save_state()

    def portfolio_summary(self) -> dict:
        open_value  = sum(p["shares"] * p["current_price"] for p in self.positions.values())
        total_value = self.cash + open_value
        total_pnl   = total_value - self.initial_capital

        wins         = sum(1 for t in self.closed_trades if t["pnl"] > 0)
        losses       = sum(1 for t in self.closed_trades if t["pnl"] <= 0)
        total_closed = len(self.closed_trades)
        win_rate     = (wins / total_closed * 100) if total_closed else 0.0
        closed_pnl   = sum(t["pnl"] for t in self.closed_trades)

        # Expectancy = (win_rate × avg_win) - (loss_rate × avg_loss)
        avg_win  = (sum(t["pnl"] for t in self.closed_trades if t["pnl"] > 0) / wins) if wins else 0
        avg_loss = (sum(abs(t["pnl"]) for t in self.closed_trades if t["pnl"] <= 0) / losses) if losses else 0
        win_rate_dec = win_rate / 100
        expectancy   = (win_rate_dec * avg_win) - ((1 - win_rate_dec) * avg_loss)

        return {
            "cash":           round(self.cash, 2),
            "open_value":     round(open_value, 2),
            "total_value":    round(total_value, 2),
            "total_pnl":      round(total_pnl, 2),
            "total_pnl_pct":  round((total_pnl / self.initial_capital) * 100, 2),
            "closed_pnl":     round(closed_pnl, 2),
            "open_positions": len(self.positions),
            "closed_trades":  total_closed,
            "wins":           wins,
            "losses":         losses,
            "win_rate_pct":   round(win_rate, 2),
            "avg_win":        round(avg_win, 2),
            "avg_loss":       round(avg_loss, 2),
            "expectancy":     round(expectancy, 2),
        }

    def print_portfolio(self) -> None:
        s    = self.portfolio_summary()
        sign = "+" if s["total_pnl"] >= 0 else ""

        print("\n" + "=" * 65)
        print("  PAPER PORTFOLIO")
        print("=" * 65)
        print(f"  Cash           : {s['cash']:>12,.2f}")
        print(f"  Open Value     : {s['open_value']:>12,.2f}")
        print(f"  Total Value    : {s['total_value']:>12,.2f}")
        print(f"  Total P&L      : {sign}{s['total_pnl']:>11,.2f}  ({sign}{s['total_pnl_pct']:.2f}%)")
        print(f"  Closed Trades  : {s['closed_trades']}  (W:{s['wins']} / L:{s['losses']})  Win Rate: {s['win_rate_pct']:.1f}%")
        print(f"  Expectancy/trade: {s['expectancy']:>+,.2f}  (avg win: {s['avg_win']:,.2f}  avg loss: {s['avg_loss']:,.2f})")

        if self.positions:
            print(f"\n  Open Positions:")
            print(f"  {'Symbol':<18} {'Shares':>6} {'Entry':>9} {'Current':>9} {'SL':>9} {'TP':>9} {'Unreal P&L':>12} {'Regime'}")
            print("  " + "-" * 82)
            for sym, p in self.positions.items():
                unreal = (p["current_price"] - p["entry_price"]) * p["shares"]
                usign  = "+" if unreal >= 0 else ""
                print(
                    f"  {sym:<18} {p['shares']:>6} {p['entry_price']:>9.2f} "
                    f"{p['current_price']:>9.2f} {p['stop_loss']:>9.2f} "
                    f"{p['take_profit']:>9.2f} {usign}{unreal:>11,.2f}  {p.get('regime','?')}"
                )

        if self.closed_trades:
            print(f"\n  Recent Closed Trades (last 5):")
            print(f"  {'Symbol':<18} {'Exit Date':<12} {'P&L':>10} {'Exit Reason':<16} {'Slippage'}")
            print("  " + "-" * 68)
            for t in self.closed_trades[-5:]:
                tsign = "+" if t["pnl"] >= 0 else ""
                slip  = f"{t.get('slippage_cost', 0):.2f}"
                print(
                    f"  {t['symbol']:<18} {t['exit_date']:<12} "
                    f"{tsign}{t['pnl']:>9,.2f}  {t['exit_reason']:<16} -{slip}"
                )

        print("=" * 65 + "\n")

    # ── Internal ──────────────────────────────────────────────────────────────

    def _apply_slippage(self, price: float, is_buy: bool) -> float:
        """
        Simulate market impact:
          Buy  → fill slightly above quoted price (you pay more)
          Sell → fill slightly below quoted price (you receive less)
        """
        if is_buy:
            return round(price * (1 + SLIPPAGE_PCT), 4)
        else:
            return round(price * (1 - SLIPPAGE_PCT), 4)

    def _regime_size_multiplier(self, regime_str: str) -> float:
        """Map regime string to position size multiplier."""
        multipliers = {
            "TRENDING_UP":   1.0,
            "SIDEWAYS":      0.5,
            "VOLATILE":      0.5,
            "TRENDING_DOWN": 0.0,
        }
        return multipliers.get(regime_str, 1.0)

    def _open_position(
        self,
        symbol: str,
        price: float,
        rr: dict,
        signal_data: dict,
        regime_str: str,
    ) -> None:
        # Regime-aware position sizing
        size_mult = self._regime_size_multiplier(regime_str)
        if size_mult == 0.0:
            logger.info("Skipping %s — regime %s blocks longs", symbol, regime_str)
            return

        base_shares = rr["shares"]
        shares      = max(int(base_shares * size_mult), 1)

        # Apply slippage to entry
        fill_price     = self._apply_slippage(price, is_buy=True)
        slippage_cost  = round((fill_price - price) * shares, 2)
        cost           = shares * fill_price

        if cost > self.cash:
            logger.warning("Insufficient cash for %s. Need %.2f, have %.2f", symbol, cost, self.cash)
            return

        self.cash -= cost
        self.positions[symbol] = {
            "symbol":        symbol,
            "shares":        shares,
            "entry_price":   fill_price,
            "quoted_price":  price,
            "current_price": fill_price,
            "stop_loss":     rr["stop_loss"],
            "take_profit":   rr["take_profit"],
            "entry_date":    datetime.now().strftime("%Y-%m-%d"),
            "score":         signal_data.get("score", 0),
            "regime":        regime_str,
            "slippage_cost": slippage_cost,
        }

        size_note = f" (×{size_mult} regime)" if size_mult < 1.0 else ""
        logger.info(
            "PAPER BUY | %s | %d shares @ %.2f (slip: %.2f) | SL: %.2f | TP: %.2f | %s",
            symbol, shares, fill_price, slippage_cost, rr["stop_loss"], rr["take_profit"], regime_str,
        )
        print(
            f"  📥 PAPER BUY  | {symbol} | {shares} shares @ {fill_price:.2f}"
            f" (slip: -{slippage_cost:.2f}){size_note} | SL: {rr['stop_loss']:.2f} | TP: {rr['take_profit']:.2f}"
        )

    def _check_exit(
        self,
        symbol: str,
        close: float,
        open_price: float,
        signal: str,
    ) -> None:
        pos = self.positions[symbol]
        pos["current_price"] = close

        sl = pos["stop_loss"]
        tp = pos["take_profit"]

        # Gap detection: if open gapped below SL, exit at open (not SL)
        gap_below_sl = open_price < sl * (1 - GAP_THRESHOLD)

        if gap_below_sl:
            # Gapped down through stop — exit at open price (worse than SL)
            self._close_position(symbol, open_price, "GAP-DOWN-SL")
        elif close <= sl:
            self._close_position(symbol, sl, "STOP-LOSS")
        elif close >= tp:
            self._close_position(symbol, tp, "TAKE-PROFIT")
        elif signal == "SELL":
            self._close_position(symbol, close, "SIGNAL-SELL")

    def _close_position(self, symbol: str, price: float, reason: str) -> None:
        pos        = self.positions.pop(symbol)
        fill_price = self._apply_slippage(price, is_buy=False)

        gross_pnl     = (fill_price - pos["entry_price"]) * pos["shares"]
        slippage_cost = round((price - fill_price) * pos["shares"], 2)  # cost of exit slip
        total_slip    = round(pos.get("slippage_cost", 0) + slippage_cost, 2)

        self.cash += fill_price * pos["shares"]

        trade = {
            "symbol":        symbol,
            "shares":        pos["shares"],
            "entry_price":   pos["entry_price"],
            "exit_price":    fill_price,
            "entry_date":    pos["entry_date"],
            "exit_date":     datetime.now().strftime("%Y-%m-%d"),
            "pnl":           round(gross_pnl, 2),
            "exit_reason":   reason,
            "score":         pos.get("score", 0),
            "regime":        pos.get("regime", "?"),
            "slippage_cost": total_slip,
        }
        self.closed_trades.append(trade)

        sign = "+" if gross_pnl >= 0 else ""
        logger.info("PAPER SELL | %s | %s | PnL: %s%.2f | Slip: -%.2f", symbol, reason, sign, gross_pnl, total_slip)
        print(f"  📤 PAPER SELL | {symbol} | {reason} | PnL: {sign}{gross_pnl:.2f} | Slip: -{total_slip:.2f}")

    # ── Persistence ───────────────────────────────────────────────────────────

    def _save_state(self) -> None:
        os.makedirs(os.path.dirname(PAPER_STATE_FILE), exist_ok=True)
        state = {
            "initial_capital": self.initial_capital,
            "cash":            self.cash,
            "positions":       self.positions,
            "closed_trades":   self.closed_trades,
        }
        with open(PAPER_STATE_FILE, "w", encoding="utf-8") as f:
            json.dump(state, f, indent=2)

    def _load_state(self) -> None:
        if not os.path.exists(PAPER_STATE_FILE):
            return
        try:
            with open(PAPER_STATE_FILE, "r", encoding="utf-8") as f:
                state = json.load(f)
            self.initial_capital = state.get("initial_capital", self.initial_capital)
            self.cash            = state.get("cash", self.cash)
            self.positions       = state.get("positions", {})
            self.closed_trades   = state.get("closed_trades", [])
            logger.info(
                "Paper portfolio loaded | Cash: %.2f | Open: %d | Closed: %d",
                self.cash, len(self.positions), len(self.closed_trades),
            )
        except Exception as exc:
            logger.warning("Could not load paper portfolio: %s", exc)
