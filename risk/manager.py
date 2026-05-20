"""
Risk Management Module

Regime-aware risk calculation:
  - ATR-based SL/TP with multiplier sourced from the current regime
  - VOLATILE regime uses wider SL (ATR × 2.5 vs normal ATR × 2.0)
  - Position size scaled by regime.position_size_multiplier
  - Portfolio-level trade approval (max active trades, capital limits)
"""

import logging
from indicators.regime import Regime
from config.settings import RISK_CONFIG, PORTFOLIO_CONFIG

logger = logging.getLogger(__name__)


class RiskManager:

    def __init__(
        self,
        capital: float = RISK_CONFIG["default_capital"],
        active_trades: int = 0,
    ):
        self.capital          = capital
        self.active_trades    = active_trades
        self.max_risk_pct     = RISK_CONFIG["max_risk_per_trade_pct"]
        self.fb_sl_pct        = RISK_CONFIG["fallback_stop_loss_pct"]
        self.fb_tp_pct        = RISK_CONFIG["fallback_take_profit_pct"]
        self.max_position_pct = RISK_CONFIG["max_position_pct"]
        self.max_trades       = PORTFOLIO_CONFIG["max_active_trades"]
        self.capital_per_pct  = PORTFOLIO_CONFIG["capital_per_trade_pct"]

    # ── SL / TP ───────────────────────────────────────────────────────────────

    def stop_loss_price(
        self,
        entry: float,
        atr: float = None,
        regime: Regime = Regime.STRONG_TREND_UP,
    ) -> float:
        """
        ATR-based stop-loss. Multiplier comes from the regime:
          TRENDING_UP / SIDEWAYS → ATR × 2.0
          VOLATILE               → ATR × 2.5  (wider — avoids noise shakeouts)
        """
        sl_mult = regime.atr_sl_multiplier
        if atr and atr > 0:
            sl = entry - (atr * sl_mult)
        else:
            sl = entry * (1 - self.fb_sl_pct / 100)
        return round(max(sl, 0.01), 2)

    def take_profit_price(
        self,
        entry: float,
        atr: float = None,
        regime: Regime = Regime.STRONG_TREND_UP,
    ) -> float:
        """
        Take-profit = entry + (ATR × tp_multiplier).
        TP multiplier is always 2× the SL multiplier to maintain 2:1 R:R.
        """
        sl_mult = regime.atr_sl_multiplier
        tp_mult = sl_mult * 2   # 2:1 R:R regardless of regime
        if atr and atr > 0:
            tp = entry + (atr * tp_mult)
        else:
            tp = entry * (1 + self.fb_tp_pct / 100)
        return round(tp, 2)

    def risk_reward_ratio(self, entry: float, sl: float, tp: float) -> float:
        risk   = entry - sl
        reward = tp - entry
        if risk <= 0:
            return 0.0
        return round(reward / risk, 2)

    # ── Position sizing ───────────────────────────────────────────────────────

    def position_size(
        self,
        entry: float,
        atr: float = None,
        regime: Regime = Regime.STRONG_TREND_UP,
        confidence_score: float = None,
    ) -> int:
        """
        Base shares = risk_amount / risk_per_share
        Then scaled by regime.position_size_multiplier and confidence_score (Q-Score):
          - Q-Score >= 90: 100% of regime size (multiplier ×1.0)
          - Q-Score 75-89: 50% of regime size (multiplier ×0.5)
          - Q-Score 60-74: 25% of regime size (multiplier ×0.25)
          - Q-Score < 60:  0% of regime size (rejected)
        Effective multiplier is the minimum of regime and confidence multipliers.
        """
        if entry <= 0:
            raise ValueError("Entry price must be positive.")

        risk_amount = self.capital * (self.max_risk_pct / 100)
        sl_mult     = regime.atr_sl_multiplier

        if atr and atr > 0:
            risk_per_share = atr * sl_mult
        else:
            risk_per_share = entry * (self.fb_sl_pct / 100)

        base_shares = risk_amount / risk_per_share if risk_per_share > 0 else 0

        # Cap by capital allocation per trade
        alloc_value   = self.capital * (self.capital_per_pct / 100)
        shares_by_cap = alloc_value / entry

        base_shares = int(min(base_shares, shares_by_cap))

        # Determine confidence-based sizing multiplier
        if confidence_score is not None:
            if confidence_score >= 90:
                conf_mult = 1.0
            elif confidence_score >= 75:
                conf_mult = 0.5
            elif confidence_score >= 60:
                conf_mult = 0.25
            else:
                conf_mult = 0.0
            size_mult = min(regime.position_size_multiplier, conf_mult)
        else:
            size_mult = regime.position_size_multiplier

        shares = int(base_shares * size_mult)
        return max(shares, 1) if size_mult > 0 else 0

    # ── Trade approval ────────────────────────────────────────────────────────

    def approve_trade(
        self,
        entry: float,
        signal: str,
        atr: float = None,
        regime: Regime = Regime.STRONG_TREND_UP,
        confidence_score: float = None,
    ) -> dict:
        """
        Full trade validation with regime-aware and confidence-aware sizing.

        Args:
            entry:  Current market price
            signal: 'BUY' | 'SELL' | 'HOLD'
            atr:    Current ATR value
            regime: Effective regime for this symbol
            confidence_score: Q-Score from confidence engine (0-100)

        Returns:
            dict with approved flag and complete risk parameters
        """
        if signal != "BUY":
            return {"approved": False, "reason": f"Signal is {signal}, not BUY."}

        if entry <= 0:
            return {"approved": False, "reason": "Invalid entry price."}

        if not regime.allows_buy:
            return {
                "approved": False,
                "reason": f"Regime {regime} does not allow long entries.",
            }

        if confidence_score is not None and confidence_score < 60:
            return {
                "approved": False,
                "reason": f"Confidence score ({confidence_score}) is below the minimum threshold of 60.",
            }

        if self.active_trades >= self.max_trades:
            return {
                "approved": False,
                "reason": f"Max active trades reached ({self.max_trades}).",
            }

        # Calculate sizing multiplier for report logging
        if confidence_score is not None:
            if confidence_score >= 90:
                conf_mult = 1.0
            elif confidence_score >= 75:
                conf_mult = 0.5
            elif confidence_score >= 60:
                conf_mult = 0.25
            else:
                conf_mult = 0.0
            size_mult = min(regime.position_size_multiplier, conf_mult)
        else:
            size_mult = regime.position_size_multiplier

        sl     = self.stop_loss_price(entry, atr, regime)
        tp     = self.take_profit_price(entry, atr, regime)
        shares = self.position_size(entry, atr, regime, confidence_score)
        cost   = shares * entry
        rr     = self.risk_reward_ratio(entry, sl, tp)

        if shares == 0:
            return {"approved": False, "reason": f"Regime {regime} or Confidence {confidence_score} → position size = 0."}

        if cost > self.capital:
            return {
                "approved": False,
                "reason": f"Trade cost {cost:,.2f} exceeds capital {self.capital:,.2f}.",
            }

        report = {
            "approved":            True,
            "entry_price":         round(entry, 2),
            "shares":              shares,
            "trade_cost":          round(cost, 2),
            "stop_loss":           sl,
            "take_profit":         tp,
            "atr_used":            round(atr, 2) if atr else None,
            "sl_multiplier":       regime.atr_sl_multiplier,
            "size_multiplier":     size_mult,
            "confidence_score":    confidence_score,
            "max_loss":            round(shares * (entry - sl), 2),
            "max_gain":            round(shares * (tp - entry), 2),
            "risk_reward":         rr,
            "rr_warning":          rr < 1.5,
            "capital_at_risk_pct": round((shares * (entry - sl) / self.capital) * 100, 2),
            "sl_method":           "ATR" if (atr and atr > 0) else "Fixed%",
            "regime":              str(regime),
        }

        logger.info(
            "Trade approved | entry=%.2f shares=%d SL=%.2f TP=%.2f "
            "R:R=%.2f regime=%s sl_mult=×%.1f size_mult=×%.2f Q-Score=%s",
            entry, shares, sl, tp, rr, regime,
            regime.atr_sl_multiplier, size_mult, str(confidence_score),
        )
        return report
