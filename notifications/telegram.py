"""
Telegram Alert System
Sends beautifully formatted, institutional-grade alerts to your Telegram bot.
Supports the 8:30 AM Pre-Market Snapshot, 3:15 PM EOD Analysis, and 0-100 Confidence pros/cons.
"""

import os
import logging
from typing import List, Optional
import requests

from config.settings import TELEGRAM_CONFIG

logger = logging.getLogger(__name__)

# Load from environment — never hardcode credentials
def _get_token():
    return os.getenv("TELEGRAM_BOT_TOKEN", "").strip()

def _get_chat_id():
    return os.getenv("TELEGRAM_CHAT_ID", "").strip()


class TelegramNotifier:
    """
    Sends formatted alerts to Telegram.
    Fails silently if credentials are missing — never crashes the main system.
    """

    def __init__(self):
        token = _get_token()
        chat_id = _get_chat_id()
        
        self.enabled = TELEGRAM_CONFIG["enabled"] and bool(token) and bool(chat_id)
        self.parse_mode = TELEGRAM_CONFIG["parse_mode"]

        if TELEGRAM_CONFIG["enabled"] and not self.enabled:
            missing = []
            if not token: missing.append("TELEGRAM_BOT_TOKEN")
            if not chat_id: missing.append("TELEGRAM_CHAT_ID")
            logger.warning(
                f"Telegram enabled in config but {', '.join(missing)} not set in environment. Alerts disabled."
            )

    # ── Public API ────────────────────────────────────────────────────────────

    def send_signal_alert(self, result: dict) -> bool:
        """Send a formatted signal alert for a single symbol."""
        if not self.enabled or "error" in result:
            return False

        signal = result.get("signal", "HOLD")
        if signal == "HOLD":
            return False   # don't spam HOLD signals

        msg = self._format_signal(result)
        return self._send(msg)

    def send_daily_summary(self, results: List[dict], regime_str: str = "") -> bool:
        """Send a compact daily summary of all signals."""
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
                f"📥 <b>PAPER BUY ENTRY</b>\n"
                f"Stock: <b>{symbol}</b>\n"
                f"Price: ₹{price:,.2f} | Shares: {shares}\n"
                f"SL: ₹{sl:,.2f} | TP: ₹{tp:,.2f}"
            )
        else:
            sign = "+" if (pnl or 0) >= 0 else ""
            msg = (
                f"📤 <b>PAPER SELL EXIT</b>\n"
                f"Stock: <b>{symbol}</b>\n"
                f"Price: ₹{price:,.2f} | P&L: {sign}₹{pnl:,.2f}"
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
            f"{emoji} <b>MARKET REGIME CHANGE</b>\n"
            f"<b>{regime_str}</b>\n"
            f"{description}"
        )
        return self._send(msg)

    def send_pre_market_report(self, report: dict) -> bool:
        """Send a comprehensive 8:30 AM pre-market intelligence report."""
        if not self.enabled:
            return False

        lines = [
            f"🔮 <b>PRE-MARKET REPORT FOR TODAY</b>",
            f"Mode: <b>PREPARATION & PROBABILITY</b>",
            f"Regime: <b>{report['market_regime']}</b>\n",
            f"🌍 <b>GLOBAL MARKET SNAPSHOT</b>",
        ]

        for s in report["global_snapshot"]:
            sign = "+" if s["change"] >= 0 else ""
            impact_emoji = "🟢" if s["impact"] == "BULLISH" else ("🔴" if s["impact"] == "BEARISH" else "⚪")
            lines.append(f"  {impact_emoji} {s['name']}: {s['value']} ({sign}{s['change']}%) | <i>{s['impact_text']}</i>")

        lines += [
            f"\n📊 <b>BIAS & RISK EXPECTATION</b>",
            f"Opening Bias: <b>{report['opening_bias']}</b>",
            f"Confidence: <b>{report['confidence_pct']}%</b>",
            f"Gap Probability: <b>{report['gap_probability']}</b>",
            f"Risk Level: <b>{report['risk_level']}</b>\n",
            f"⚡ <b>SECTOR STRENGTH RANKINGS</b>",
        ]

        for sec in report["sector_rankings"][:3]:  # Top 3
            sign = "+" if sec["strength"] >= 0 else ""
            lines.append(f"  • {sec['sector']}: {sign}{sec['strength']}% ({sec['status']})")

        lines.append(f"\n🏆 <b>WATCHLIST QUALITY RANKINGS</b>")
        for sym in report["watchlist"][:5]:  # Top 5
            signal_sig = sym.get("signal", "HOLD")
            lines.append(
                f"  • {sym['symbol']} | Q-Score: <b>{sym.get('confidence_score',0)}/100</b> | {signal_sig}"
            )

        if report["warnings"]:
            lines.append(f"\n⚠️ <b>DO NOT TRADE WARNINGS</b>")
            for w in report["warnings"]:
                lines.append(f"  {w}")

        lines.append(f"\n📁 Detailed report: <code>state/pre_market_report.md</code>")
        return self._send("\n".join(lines))

    def send_eod_report(self, report: dict) -> bool:
        """Send a comprehensive 3:15 PM EOD intelligence report."""
        if not self.enabled:
            return False

        acc = report["accuracy_tracking"]
        pred = report["prediction"]
        nifty_sign = "+" if report["nifty_change"] >= 0 else ""

        lines = [
            f"🌇 <b>END-OF-DAY MARKET ANALYSIS</b>",
            f"Nifty 50: <b>{nifty_sign}{report['nifty_change']}%</b>\n",
            f"🕯️ <b>CLOSING STRUCTURE</b>",
            f"Candle: <b>{report['close_structure']}</b>",
            f"Description: <i>{report['structure_desc']}</i>\n",
            f"📈 <b>MARKET BREADTH</b>",
            f"Advances: {report['advances']} | Declines: {report['declines']}",
            f"A/D Ratio: <b>{report['adv_dec_ratio']}</b>\n",
            f"🔮 <b>TOMORROW OUTLOOK FORECAST</b>",
            f"Forecast: <b>{pred['forecast']}</b>",
            f"Narrative: <i>{pred.get('narrative', 'N/A')}</i>",
            f"  • Bullish: {pred['probs']['BULLISH']}%",
            f"  • Range-bound: {pred['probs']['RANGE']}%",
            f"  • Bearish: {pred['probs']['BEARISH']}%",
        ]

        if report["unusual_volume"]:
            lines.append(f"\n🐋 <b>SMART MONEY VOLUME SPIKES</b>")
            for uv in report["unusual_volume"][:3]:
                uv_sign = "+" if uv["change"] >= 0 else ""
                lines.append(f"  • {uv['symbol']}: {uv['ratio']}x average | Change: {uv_sign}{uv['change']}%")

        if acc.get("scored_yesterday"):
            yesterday_res_emoji = "🎯" if acc["yesterday_result"] == "SUCCESS" else "❌"
            lines.append(
                f"\n🎯 <b>FORECAST ACCURACY TRACKER</b>\n"
                f"  Yesterday Forecast: {acc['yesterday_forecast']}\n"
                f"  Today Outcome: {yesterday_res_emoji} <b>{acc['yesterday_result']}</b>\n"
                f"  Cumulative Accuracy: <b>{acc['accuracy_pct']}%</b> ({acc['successful_predictions']}/{acc['total_predictions']})"
            )

        lines.append(f"\n📁 Detailed report: <code>state/eod_report.md</code>")
        return self._send("\n".join(lines))

    def test_connection(self) -> bool:
        """Send a test message to verify the bot is working."""
        return self._send("🤖 <b>Trading Bot Connected</b>\nAlerts are active.")

    def send_optimization_alert(self, rules: dict) -> bool:
        """Send a daily auto-optimization summary."""
        if not self.enabled:
            return False

        skip_regimes = rules.get("skip_regimes", [])
        skip_symbols = rules.get("skip_symbols", [])
        skip_score   = rules.get("skip_score_below", 0)
        generated_at = rules.get("generated_at", "?")

        lines = ["🤖 <b>AUTO-OPTIMIZATION COMPLETE</b>"]
        lines.append(f"Based on paper trading history as of {generated_at}:\n")

        if skip_regimes:
            lines.append(f"⛔ <b>Blocked regimes:</b> {', '.join(skip_regimes)}")
        else:
            lines.append("✅ All market regimes still active")

        if skip_symbols:
            lines.append(f"⛔ <b>Blocked symbols:</b> {', '.join(skip_symbols)}")
        else:
            lines.append("✅ All symbols still active")

        if skip_score > 0:
            lines.append(f"⛔ <b>Min score raised to:</b> {skip_score}/4")
        else:
            lines.append("✅ No score filter needed yet")

        if not skip_regimes and not skip_symbols and skip_score == 0:
            lines.append("\n📊 Not enough closed trades to learn from yet. Keep paper trading!")
        else:
            lines.append("\n📈 These rules will apply to today's scan automatically.")

        return self._send("\n".join(lines))

    def send_learning_report(self, summary_text: str) -> bool:
        """Send a compact learning report alert."""
        if not self.enabled:
            return False
        return self._send(summary_text)

    # ── Formatters ────────────────────────────────────────────────────────────

    def _format_signal(self, r: dict) -> str:
        signal   = r.get("signal", "HOLD")
        score    = r.get("score", 0)
        regime   = r.get("regime", "?")
        rsi      = r.get("rsi", 0)
        pullback = r.get("pullback_pct", 0)
        rr       = r.get("risk_report", {})
        conf     = r.get("confidence_score", 0)
        pros     = r.get("pros", [])
        cons     = r.get("cons", [])

        icon = {"BUY": "🟢", "SELL": "🔴"}.get(signal, "🟡")
        score_bar = "●" * score + "○" * (4 - score)

        lines = [
            f"{icon} <b>{signal} SIGNAL — {r['symbol']}</b>",
            f"Date: {r.get('date', '?')}",
            f"Score: [{score_bar}] {score}/4",
            f"Confidence Score: <b>{conf}/100</b>",
            f"Regime: <b>{regime}</b>",
            f"Close: {r.get('close', 0):.2f}  |  RSI: {rsi:.1f}",
            f"Pullback: {pullback:.1f}%  |  ATR: {r.get('atr', 'N/A')}\n",
        ]

        if pros:
            lines.append("<b>✓ Pros:</b>")
            for p in pros[:3]:
                lines.append(f"  {p}")
        if cons:
            lines.append("\n<b>✗ Cons:</b>")
            for c in cons[:3]:
                lines.append(f"  {c}")

        if rr.get("approved"):
            lines += [
                f"",
                f"<b>Risk ({rr.get('sl_method','?')})</b>",
                f"Shares: {rr['shares']}  |  Cost: ₹{rr['trade_cost']:,.0f}",
                f"SL: ₹{rr['stop_loss']:.2f}  |  TP: ₹{rr['take_profit']:.2f}",
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
                lines.append(
                    f"  • {r['symbol']} | Q-Score: {r.get('confidence_score',0)} | RSI:{r.get('rsi',0):.1f}"
                )

        if sells:
            lines.append(f"\n🔴 <b>SELL ({len(sells)})</b>")
            for r in sells:
                lines.append(f"  • {r['symbol']} | RSI:{r.get('rsi',0):.1f}")

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
            
        token = _get_token()
        chat_id = _get_chat_id()
        api_url = f"https://api.telegram.org/bot{token}/sendMessage"

        try:
            resp = requests.post(
                api_url,
                json={
                    "chat_id":    chat_id,
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
