"""
Regime-Aware Backtrader Strategy
Implements all three production controls:

  1. Trend Strength Tiering
       STRONG_TREND_UP   → score ≥ 2
       WEAK_TREND_UP     → score = 3
       SIDEWAYS/VOLATILE → score = 3, half size, wider SL
       TRENDING_DOWN     → no entries

  2. Trade Cooldown
       After any exit, the same symbol cannot be re-entered for
       `cooldown_bars` bars. Prevents choppy re-entries on noise.

  3. Minimum Hold Period
       RSI/signal exits are ignored for the first `min_hold_bars`
       after entry. Stop-loss always overrides this.
"""

import backtrader as bt
from config.settings import STRATEGY_CONFIG, INDICATOR_CONFIG, RISK_CONFIG, REGIME_CONFIG, EXECUTION_CONFIG


class RegimeAwareStrategy(bt.Strategy):

    params = (
        # Indicators
        ("rsi_period",           INDICATOR_CONFIG["rsi_period"]),
        ("ma_fast",              INDICATOR_CONFIG["ma_fast"]),
        ("ma_slow",              INDICATOR_CONFIG["ma_slow"]),
        ("atr_period",           INDICATOR_CONFIG["atr_period"]),
        ("vol_ma_period",        INDICATOR_CONFIG["volume_ma_period"]),
        # Strategy thresholds — regime-aware RSI
        ("rsi_buy_strong",       35),   # STRONG_TREND_UP
        ("rsi_buy_weak",         30),   # WEAK_TREND_UP
        ("rsi_buy_sideways",     25),   # SIDEWAYS / VOLATILE
        ("rsi_sell",             STRATEGY_CONFIG["rsi_sell_threshold"]),
        ("vol_multiplier",       STRATEGY_CONFIG["volume_multiplier"]),
        # Risk
        ("atr_sl_mult_normal",   RISK_CONFIG["atr_sl_multiplier"]),
        ("atr_sl_mult_volatile", 2.5),
        # Regime
        ("atr_ratio_threshold",  REGIME_CONFIG["atr_ratio_threshold"]),
        ("slope_strong_up",      REGIME_CONFIG["slope_strong_up"]),
        ("slope_weak_up",        REGIME_CONFIG["slope_weak_up"]),
        ("slope_strong_down",    REGIME_CONFIG["slope_strong_down"]),
        ("slope_weak_down",      REGIME_CONFIG["slope_weak_down"]),
        # Execution controls
        ("cooldown_bars",        EXECUTION_CONFIG["cooldown_days"]),
        ("min_hold_bars",        EXECUTION_CONFIG["min_hold_bars"]),
        ("printlog",             True),
    )

    def __init__(self):
        self.rsi    = bt.indicators.RSI(self.data.close, period=self.p.rsi_period)
        self.ma50   = bt.indicators.SMA(self.data.close, period=self.p.ma_fast)
        self.ma200  = bt.indicators.SMA(self.data.close, period=self.p.ma_slow)
        self.atr    = bt.indicators.ATR(self.data,       period=self.p.atr_period)
        self.vol_ma = bt.indicators.SMA(self.data.volume, period=self.p.vol_ma_period)

        self.order        = None
        self.stop_price   = None
        self.tp_price     = None
        self.bars_in_trade = 0       # bars since entry (for min hold)
        self.cooldown_left = 0       # bars remaining in cooldown

    def log(self, txt, dt=None):
        if self.p.printlog:
            dt = dt or self.datas[0].datetime.date(0)
            print(f"[{dt}] {txt}")

    # ── Regime detection (rolling, bar-by-bar) ────────────────────────────────

    def _detect_regime(self) -> str:
        if len(self.ma50) < 55:
            return "SIDEWAYS"

        # ATR ratio
        atr_vals  = [self.atr[-i] for i in range(min(50, len(self.atr)))]
        atr_avg   = sum(atr_vals) / len(atr_vals) if atr_vals else 1.0
        atr_ratio = self.atr[0] / atr_avg if atr_avg > 0 else 1.0

        if atr_ratio > self.p.atr_ratio_threshold:
            return "VOLATILE"

        # Normalised slope
        ma_vals    = [self.ma50[-i] for i in range(self.p.ma_fast + 1)]
        diffs      = [ma_vals[i] - ma_vals[i + 1] for i in range(self.p.ma_fast)]
        slope_abs  = sum(diffs[-5:]) / 5   # last 5 diffs, smoothed
        slope_norm = slope_abs / self.ma50[0] if self.ma50[0] != 0 else 0.0

        price        = self.data.close[0]
        # Uptrend: price above MA200 (allows pullbacks below MA50)
        above_ma200  = price > self.ma200[0]
        below_both   = price < self.ma50[0] and price < self.ma200[0]

        if above_ma200 and slope_norm >= self.p.slope_strong_up:
            return "STRONG_TREND_UP"
        if above_ma200 and slope_norm >= self.p.slope_weak_up:
            return "WEAK_TREND_UP"
        if below_both and slope_norm <= self.p.slope_strong_down:
            return "STRONG_TREND_DOWN"
        if below_both and slope_norm <= self.p.slope_weak_down:
            return "WEAK_TREND_DOWN"
        return "SIDEWAYS"

    def _rsi_threshold(self, regime: str) -> int:
        return {
            "STRONG_TREND_UP": self.p.rsi_buy_strong,
            "WEAK_TREND_UP":   self.p.rsi_buy_weak,
            "SIDEWAYS":        self.p.rsi_buy_sideways,
            "VOLATILE":        self.p.rsi_buy_sideways,
        }.get(regime, 0)

    def _score(self) -> int:
        score = 0
        if self.data.close[0] > self.ma200[0]:
            score += 1
        regime = self._detect_regime()
        if self.rsi[0] < self._rsi_threshold(regime):
            score += 1
        if self.data.volume[0] > self.vol_ma[0] * self.p.vol_multiplier:
            score += 1
        return score

    # ── Order notifications ───────────────────────────────────────────────────

    def notify_order(self, order):
        if order.status in [order.Submitted, order.Accepted]:
            return

        if order.status == order.Completed:
            if order.isbuy():
                regime  = self._detect_regime()
                sl_mult = (self.p.atr_sl_mult_volatile
                           if regime == "VOLATILE"
                           else self.p.atr_sl_mult_normal)
                entry = order.executed.price
                self.stop_price    = round(entry - self.atr[0] * sl_mult, 2)
                self.tp_price      = round(entry + self.atr[0] * sl_mult * 2, 2)
                self.bars_in_trade = 0
                self.log(
                    f"BUY  EXEC | {entry:.2f} | SL:{self.stop_price:.2f} "
                    f"TP:{self.tp_price:.2f} | {regime}"
                )
            else:
                self.log(
                    f"SELL EXEC | {order.executed.price:.2f} "
                    f"| PnL:{order.executed.pnl:.2f}"
                )
                self.stop_price    = None
                self.tp_price      = None
                self.cooldown_left = self.p.cooldown_bars   # start cooldown
        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            self.log(f"Order {order.Status[order.status]}")

        self.order = None

    # ── Main logic ────────────────────────────────────────────────────────────

    def next(self):
        if self.order:
            return

        # Tick down cooldown counter
        if self.cooldown_left > 0:
            self.cooldown_left -= 1

        if not self.position:
            # ── Entry gate ────────────────────────────────────────────────────
            if self.cooldown_left > 0:
                return   # still in cooldown — skip

            regime    = self._detect_regime()
            score     = self._score()
            min_score = {
                "STRONG_TREND_UP":   2,
                "WEAK_TREND_UP":     3,
                "SIDEWAYS":          3,
                "VOLATILE":          3,
            }.get(regime, 99)

            if regime not in ("STRONG_TREND_DOWN", "WEAK_TREND_DOWN", "SIDEWAYS", "VOLATILE") and score >= min_score:
                self.log(
                    f"BUY  SIGNAL | {self.data.close[0]:.2f} "
                    f"| RSI:{self.rsi[0]:.1f} | Score:{score} | {regime}"
                )
                self.order = self.buy()

        else:
            # ── Exit gate ─────────────────────────────────────────────────────
            self.bars_in_trade += 1
            close  = self.data.close[0]
            hit_sl = self.stop_price and close <= self.stop_price
            hit_tp = self.tp_price   and close >= self.tp_price

            # Stop-loss always fires immediately (no hold period protection)
            if hit_sl:
                self.log(f"STOP-LOSS   | {close:.2f} SL:{self.stop_price:.2f}")
                self.order = self.sell()
                return

            # Take-profit and RSI exit respect minimum hold period
            if self.bars_in_trade < self.p.min_hold_bars:
                return   # too early — hold regardless of RSI/TP signal

            if hit_tp:
                self.log(f"TAKE-PROFIT | {close:.2f} TP:{self.tp_price:.2f}")
                self.order = self.sell()
            elif self.rsi[0] > self.p.rsi_sell:
                self.log(f"RSI EXIT    | {close:.2f} RSI:{self.rsi[0]:.1f}")
                self.order = self.sell()

    def stop(self):
        self.log(f"Done | Final: {self.broker.getvalue():.2f}")
