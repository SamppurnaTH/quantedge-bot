"""
Telegram Alert System

Sends formatted trading signals to your Telegram bot.
Credentials are loaded from environment variables (never hardcoded).

Setup:
  1. Create bot via @BotFather → get TELEGRAM_BOT_TOKEN
  2. Message your bot → get TELEGRAM_CHAT_ID from getUpdates
  3. Add to trading_bot/.env:
       TELEGRAM_BOT_TOKEN=your_token_here
       TELEGRAM_CHAT_ID=your_chat_id_here

Usage:
  from notifications.telegram import TelegramNotifier
  notifier = TelegramNotifier()
  notifier.send_signal_alert(signal_result)
  notifier.send_daily_summary(results)
"""

import os
import logging
from typing import List, Optional
import requests

from config.settings import TELEGRAM_CONFIG

logger = logging.getLogger(__name__)

# Load from environment — never hardcode credentials
_TOKEN   = os.getenv("TELEGRAM_BOT_TOKEN", "")
_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")
_API_URL = f"https://api.telegram.org/bot{_TOKEN}/sendMessage"


class TelegramNotifier:
    """
    Sends formatted alerts to Telegram.
    Fails silently if credentials are missing — never crashes the main system.
    """

    def __init__(self):
        self.enabled  = TELEGRAM_CONFIG["enabled"] and bool(_TOKEN) and bool(_CHAT_ID)
        self.parse_mode = TELEGRAM_CONFIG["parse_mode"]

        if TELEGRAM_CONFIG["enabled"] and not self.enabled:
            logger.warning(
                "Telegram enabled in config but TELEGRAM_BOT_TOKEN or "
                "TELEGRAM_CHAT_ID not set in environment. Alerts disabled."
            )

    # ── Public API ────────────────────────────────────────────────────────────

    def send_signal_alert(self, result: dict) -> bool:
        """
        Send a formatted signal alert for a single symbol.

        Args:
            result: Signal dict from scan_watchlist / generate_signal_for_symbol

        Returns:
            True if sent successfully, False otherwise
        """
        if not self.enabled or "error" in result:
            return False

        signal = result.get("signal", "HOLD")
        if signal == "HOLD":
            return False   # don't spam HOLD signals

        msg = self._format_signal(result)
        return self._send(msg)

    def send_daily_summary(self, results: List[dict], regime_str: str = "") -> bool:
        """
        Send a compact daily summary of all signals.

        Args:
            results:    List of signal dicts
            regime_str: Market regime string for context

        Returns:
            True if sent successfully
        """
        if not self.enabled:
            return False

        msg = self._format_summary(results, regime_str)
        return self._send(msg)

    def send_paper_trade(self, action: str, symbol: str, price: float,
                         shares: int, sl: float, tp: float, pnl: float = None) -> bool:
        """Send a paper trade execution alert."""
        if not self.enabled:
            return False

        if action == "BUY":
            msg = (
                f"📥 <b>PAPER BUY</b>\n"
                f"Stock: <b>{symbol}</b>\n"
                f"Price: {price:.2f}  |  Shares: {shares}\n"
                f"SL: {sl:.2f}  |  TP: {tp:.2f}"
            )
        else:
            sign = "+" if (pnl or 0) >= 0 else ""
            msg = (
                f"📤 <b>PAPER SELL</b>\n"
                f"Stock: <b>{symbol}</b>\n"
                f"Price: {price:.2f}  |  P&L: {sign}{pnl:.2f}"
            )
        return self._send(msg)

    def send_regime_alert(self, regime_str: str, description: str) -> bool:
        """Send a regime change alert."""
        if not self.enabled:
            return False

        emoji_map = {
            "STRONG_TREND_UP":   "🚀",
            "WEAK_TREND_UP":     "📈",
            "SIDEWAYS":          "↔️",
            "VOLATILE":          "⚡",
            "WEAK_TREND_DOWN":   "📉",
            "STRONG_TREND_DOWN": "🔻",
        }
        emoji = emoji_map.get(regime_str, "📊")
        msg = (
            f"{emoji} <b>MARKET REGIME</b>\n"
            f"<b>{regime_str}</b>\n"
            f"{description}"
        )
        return self._send(msg)

    def test_connection(self) -> bool:
        """Send a test message to verify the bot is working."""
        return self._send("🤖 <b>Trading Bot Connected</b>\nAlerts are active.")

    # ── Formatters ────────────────────────────────────────────────────────────

    def _format_signal(self, r: dict) -> str:
        signal   = r.get("signal", "HOLD")
        score    = r.get("score", 0)
        regime   = r.get("regime", "?")
        rsi      = r.get("rsi", 0)
        pullback = r.get("pullback_pct", 0)
        rr       = r.get("risk_report", {})

        icon = {"BUY": "🟢", "SELL": "🔴"}.get(signal, "🟡")
        score_bar = "●" * score + "○" * (4 - score)

        lines = [
            f"{icon} <b>{signal} SIGNAL — {r['symbol']}</b>",
            f"Date: {r.get('date', '?')}",
            f"Score: [{score_bar}] {score}/4",
            f"Regime: <b>{regime}</b>",
            f"Close: {r.get('close', 0):.2f}  |  RSI: {rsi:.1f}",
            f"Pullback: {pullback:.1f}%  |  ATR: {r.get('atr', 'N/A')}",
        ]

        if rr.get("approved"):
            lines += [
                f"",
                f"<b>Risk ({rr.get('sl_method','?')})</b>",
                f"Shares: {rr['shares']}  |  Cost: {rr['trade_cost']:,.0f}",
                f"SL: {rr['stop_loss']:.2f}  |  TP: {rr['take_profit']:.2f}",
                f"R:R: {rr['risk_reward']}  |  Risk: {rr['capital_at_risk_pct']:.2f}%",
            ]

        lines.append("\n⚠️ <i>Decision support only — not a trade order</i>")
        return "\n".join(lines)

    def _format_summary(self, results: List[dict], regime_str: str) -> str:
        buys  = [r for r in results if r.get("signal") == "BUY" and "error" not in r]
        sells = [r for r in results if r.get("signal") == "SELL" and "error" not in r]
        holds = [r for r in results if r.get("signal") == "HOLD" and "error" not in r]

        lines = [
            f"📊 <b>DAILY SIGNAL SUMMARY</b>",
            f"Market: <b>{regime_str}</b>",
            f"",
        ]

        if buys:
            lines.append(f"🟢 <b>BUY ({len(buys)})</b>")
            for r in buys:
                rsi_threshold = r.get("rsi_threshold", "?")
                lines.append(
                    f"  • {r['symbol']}  Score:{r.get('score',0)}/4  "
                    f"RSI:{r.get('rsi',0):.1f}  Pull:{r.get('pullback_pct',0):.1f}%"
                )

        if sells:
            lines.append(f"\n🔴 <b>SELL ({len(sells)})</b>")
            for r in sells:
                lines.append(f"  • {r['symbol']}  RSI:{r.get('rsi',0):.1f}")

        if holds:
            syms = ", ".join(r["symbol"] for r in holds[:5])
            lines.append(f"\n🟡 HOLD: {syms}")

        lines.append("\n⚠️ <i>Decision support only</i>")
        return "\n".join(lines)

    # ── HTTP sender ───────────────────────────────────────────────────────────

    def _send(self, text: str) -> bool:
        """Send a message via Telegram Bot API. Returns True on success."""
        if not self.enabled:
            return False
        try:
            resp = requests.post(
                _API_URL,
                json={
                    "chat_id":    _CHAT_ID,
                    "text":       text,
                    "parse_mode": self.parse_mode,
                },
                timeout=10,
            )
            data = resp.json()
            if data.get("ok"):
                logger.info("Telegram alert sent successfully")
                return True
            else:
                logger.warning("Telegram API error: %s", data.get("description", "unknown"))
                return False
        except requests.exceptions.ConnectionError:
            logger.warning("Telegram: no internet connection")
            return False
        except Exception as exc:
            logger.warning("Telegram send failed: %s", exc)
            return False
