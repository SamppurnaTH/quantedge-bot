"""
Trade Model — the ONE definition of how a trade is entered, managed and exited.

Used by:
  * execution/paper_trader.py      (live paper trading, bar by bar)
  * analytics/learning_engine.py   (historical seeding of the learning journal)

Because both use this module, the learning journal learns the statistics of
exactly the trade the bot takes. Previously the journal was seeded with a
symmetric ±1 ATR close-only label while the bot traded a 2 ATR stop / 4 ATR
target with RSI exits — it was learning a different game.

Rules (daily bars):
  Entry : signal on bar T's close → market order filled at bar T+1's OPEN
          (+ slippage). Skipped if T+1 gaps up more than `max_entry_gap_atr`
          ATRs above the signal close (no chasing) or opens at/below the stop.
  Stop  : fill − SL_mult × ATR      (SL_mult from the regime, 2.0 / 2.5)
  Target: fill + SL_mult × ATR × (atr_tp_multiplier / atr_sl_multiplier)
  Each bar, in this order (conservative):
    1. Open ≤ stop            → exit at Open   (gap through stop)
    2. Open ≥ target          → exit at Open   (gap through target)
    3. Low  ≤ stop            → exit at stop   (if stop and target are both
                                               inside the bar we assume the
                                               stop was hit first)
    4. High ≥ target          → exit at target
    5. held ≥ min_hold and RSI > rsi_sell → exit at Close (overbought exit)
    6. held ≥ max_hold        → exit at Close  (time stop)
  Costs : slippage on both fills + Indian delivery charges (STT, stamp, …).

Outcome is measured in R = net P&L per share / initial risk per share.
R is unit-free, so trades on a ₹200 stock and a ₹5,000 stock — and
historical vs live trades — are comparable and can be aggregated.
"""

from typing import Optional, Tuple

from config.settings import EXECUTION_CONFIG, RISK_CONFIG, STRATEGY_CONFIG

SLIPPAGE     = EXECUTION_CONFIG["slippage_pct"]
COST_BUY     = EXECUTION_CONFIG["cost_buy_pct"]
COST_SELL    = EXECUTION_CONFIG["cost_sell_pct"]
DP_CHARGE    = EXECUTION_CONFIG["dp_charge_per_sell"]
MIN_HOLD     = EXECUTION_CONFIG["min_hold_bars"]
MAX_HOLD     = EXECUTION_CONFIG["max_hold_bars"]
MAX_GAP_ATR  = EXECUTION_CONFIG["max_entry_gap_atr"]
RSI_SELL     = STRATEGY_CONFIG["rsi_sell_threshold"]
TP_RATIO     = RISK_CONFIG["atr_tp_multiplier"] / RISK_CONFIG["atr_sl_multiplier"]


def stop_and_target(fill: float, atr: float, sl_mult: float) -> Tuple[float, float]:
    risk = atr * sl_mult
    return fill - risk, fill + risk * TP_RATIO


def plan_entry(signal_close: float, next_open: float, atr: float, sl_mult: float) -> Optional[dict]:
    """
    Decide whether a next-open order fills and at what levels.
    Returns None if the order is skipped.
    """
    if not (atr and atr > 0 and next_open > 0):
        return None
    if next_open > signal_close + MAX_GAP_ATR * atr:
        return None                                   # gapped away — don't chase
    fill = next_open * (1 + SLIPPAGE)
    stop, target = stop_and_target(fill, atr, sl_mult)
    if next_open <= stop:
        return None                                   # opened through the stop
    return {"entry_price": fill, "stop_loss": stop, "take_profit": target,
            "initial_risk": fill - stop, "quoted_price": next_open}


def step(pos: dict, o: float, h: float, l: float, c: float, rsi: Optional[float],
         bars_held: int, is_entry_bar: bool = False) -> Optional[Tuple[float, str]]:
    """
    Evaluate one bar for an open position. `bars_held` counts the entry bar as 1.
    Returns (raw_exit_price, reason) or None to keep holding.
    """
    sl, tp = pos["stop_loss"], pos["take_profit"]
    if not is_entry_bar:
        if o <= sl:
            return o, "GAP-DOWN-SL"
        if o >= tp:
            return o, "GAP-UP-TP"
    if l <= sl:
        return sl, "STOP-LOSS"
    if h >= tp:
        return tp, "TAKE-PROFIT"
    if rsi is not None and bars_held >= MIN_HOLD and rsi > RSI_SELL:
        return c, "SIGNAL-SELL"
    if bars_held >= MAX_HOLD:
        return c, "TIME-STOP"
    return None


def exit_fill(raw_price: float) -> float:
    return raw_price * (1 - SLIPPAGE)


def net_pnl(entry_fill: float, exit_fill_price: float, shares: float, include_dp: bool = True) -> float:
    """Net rupee P&L after statutory costs (slippage already in the fills)."""
    gross = (exit_fill_price - entry_fill) * shares
    costs = entry_fill * shares * COST_BUY + exit_fill_price * shares * COST_SELL
    if include_dp:
        costs += DP_CHARGE
    return gross - costs


def r_multiple(entry_fill: float, exit_fill_price: float, initial_risk: float,
               shares: float = 1.0, include_dp: bool = False) -> float:
    if initial_risk <= 0 or shares <= 0:
        return 0.0
    return net_pnl(entry_fill, exit_fill_price, shares, include_dp) / (initial_risk * shares)


def simulate(bars, start: int, signal_close: float, atr: float, sl_mult: float) -> Optional[dict]:
    """
    Simulate a trade on historical arrays. `bars` = dict of numpy arrays
    (open, high, low, close, rsi); the signal is on index start-1 and the
    order fills at bars["open"][start]. Returns None if the order is skipped
    or there is not enough future data to finish the trade.
    """
    n = len(bars["close"])
    if start >= n:
        return None
    plan = plan_entry(signal_close, float(bars["open"][start]), atr, sl_mult)
    if plan is None:
        return None
    for k in range(start, min(n, start + MAX_HOLD)):
        res = step(plan, float(bars["open"][k]), float(bars["high"][k]), float(bars["low"][k]),
                   float(bars["close"][k]), float(bars["rsi"][k]), bars_held=k - start + 1,
                   is_entry_bar=(k == start))
        if res is not None:
            px, reason = res
            xf = exit_fill(px)
            return {"exit_idx": k, "exit_reason": reason, "hold": k - start + 1,
                    "r": r_multiple(plan["entry_price"], xf, plan["initial_risk"]),
                    **plan}
    return None   # trade still open at end of data → not a finished observation
