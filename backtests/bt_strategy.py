"""
Backtrader Strategy — RSI + MA + Volume + ATR stops

Entry:  Close > SMA(200)  AND  RSI < rsi_buy  AND  Volume > Volume_MA × multiplier
Exit:   RSI > rsi_sell  OR  price hits ATR-based stop-loss
"""

import backtrader as bt
from config.settings import STRATEGY_CONFIG, INDICATOR_CONFIG, RISK_CONFIG


class RSI_MA_Strategy(bt.Strategy):

    params = (
        ("rsi_period",       INDICATOR_CONFIG["rsi_period"]),
        ("ma_slow",          INDICATOR_CONFIG["ma_slow"]),
        ("atr_period",       INDICATOR_CONFIG["atr_period"]),
        ("vol_ma_period",    INDICATOR_CONFIG["volume_ma_period"]),
        ("rsi_buy",          35),   # use STRONG_TREND_UP threshold as default
        ("rsi_sell",         STRATEGY_CONFIG["rsi_sell_threshold"]),
        ("vol_multiplier",   STRATEGY_CONFIG["volume_multiplier"]),
        ("atr_sl_mult",      RISK_CONFIG["atr_sl_multiplier"]),
        ("atr_tp_mult",      RISK_CONFIG["atr_tp_multiplier"]),
        ("printlog",         True),
    )

    def __init__(self):
        self.rsi     = bt.indicators.RSI(self.data.close, period=self.p.rsi_period)
        self.ma200   = bt.indicators.SMA(self.data.close, period=self.p.ma_slow)
        self.atr     = bt.indicators.ATR(self.data,       period=self.p.atr_period)
        self.vol_ma  = bt.indicators.SMA(self.data.volume, period=self.p.vol_ma_period)

        self.order      = None
        self.stop_price = None
        self.tp_price   = None

    def log(self, txt: str, dt=None):
        if self.p.printlog:
            dt = dt or self.datas[0].datetime.date(0)
            print(f"[{dt}] {txt}")

    def notify_order(self, order):
        if order.status in [order.Submitted, order.Accepted]:
            return

        if order.status == order.Completed:
            if order.isbuy():
                entry = order.executed.price
                self.stop_price = round(entry - self.atr[0] * self.p.atr_sl_mult, 2)
                self.tp_price   = round(entry + self.atr[0] * self.p.atr_tp_mult, 2)
                self.log(
                    f"BUY  EXECUTED | Price: {entry:.2f} | Size: {order.executed.size} "
                    f"| SL: {self.stop_price:.2f} | TP: {self.tp_price:.2f}"
                )
            else:
                self.log(
                    f"SELL EXECUTED | Price: {order.executed.price:.2f} "
                    f"| PnL: {order.executed.pnl:.2f}"
                )
                self.stop_price = None
                self.tp_price   = None
        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            self.log(f"Order {order.Status[order.status]}")

        self.order = None

    def next(self):
        if self.order:
            return

        vol_confirmed = self.data.volume[0] > self.vol_ma[0] * self.p.vol_multiplier

        if not self.position:
            # Entry: all three conditions must be met
            if (
                self.data.close[0] > self.ma200[0]
                and self.rsi[0] < self.p.rsi_buy
                and vol_confirmed
            ):
                self.log(
                    f"BUY  SIGNAL | Close: {self.data.close[0]:.2f} "
                    f"| RSI: {self.rsi[0]:.2f} | ATR: {self.atr[0]:.2f}"
                )
                self.order = self.buy()
        else:
            close = self.data.close[0]
            # Exit: RSI overbought OR stop-loss hit OR take-profit hit
            hit_sl = self.stop_price and close <= self.stop_price
            hit_tp = self.tp_price   and close >= self.tp_price
            rsi_exit = self.rsi[0] > self.p.rsi_sell

            if hit_sl:
                self.log(f"STOP-LOSS HIT | Close: {close:.2f} | SL: {self.stop_price:.2f}")
                self.order = self.sell()
            elif hit_tp:
                self.log(f"TAKE-PROFIT HIT | Close: {close:.2f} | TP: {self.tp_price:.2f}")
                self.order = self.sell()
            elif rsi_exit:
                self.log(f"RSI EXIT | Close: {close:.2f} | RSI: {self.rsi[0]:.2f}")
                self.order = self.sell()

    def stop(self):
        self.log(f"Backtest complete | Final Value: {self.broker.getvalue():.2f}")
