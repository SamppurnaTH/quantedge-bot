"""
Learning Report Generator
Produces a comprehensive daily learning report in five sections:

  🔥 WHAT I HAVE PROVEN    — proven elite edges (≥100 trades, ≥60% win, PF ≥1.2)
  ✅ WHAT I HAVE VALIDATED  — robust validated patterns (50–99 trades, ≥55% win, PF ≥1.1)
  🔄 WHAT I AM LEARNING    — patterns being actively tracked (20–49 trades)
  🎯 WHAT I PLAN TO LEARN  — patterns detected but under watch (<20 trades)
  ⚠️ UNRELIABLE PATTERNS   — patterns with large samples that fail performance metrics

The report is saved as state/learning_report.md and a compact version is sent to Telegram.
"""

import os
import logging
from datetime import datetime
from typing import Dict

from analytics.learning_engine import get_knowledge_summary

logger = logging.getLogger(__name__)

REPORT_FILE = os.path.join("state", "learning_report.md")


# ── Human-Readable Key Parser ──────────────────────────────────────────────────

def _parse_key(key: str) -> str:
    """Convert a technical 7-factor pattern key into a plain-English story for beginners."""
    parts = key.split("|")
    if len(parts) < 7:
        return key

    regime_raw, rsi_raw, score, channel_raw, candle_raw, supp, vol = parts[:7]

    # 1. Market Context (The Environment)
    context = ""
    if regime_raw == "STRONG_TREND_UP": context = "When the overall market is in a powerful surge,"
    elif regime_raw == "WEAK_TREND_UP": context = "When the market is slowly but steadily climbing,"
    elif regime_raw == "SIDEWAYS": context = "When the market is stuck moving sideways,"
    elif regime_raw == "VOLATILE": context = "When the market is jumping up and down wildly,"
    elif regime_raw == "STRONG_TREND_DOWN": context = "When the market is in a severe crash,"
    elif regime_raw == "WEAK_TREND_DOWN": context = "When the market is sliding down in a minor downtrend,"
    else: context = "When the market environment is highly uncertain,"

    # 2. Condition (The Setup)
    condition = ""
    if rsi_raw == "RSI<20": condition = "the stock becomes extremely cheap and heavily oversold"
    elif rsi_raw == "RSI<25": condition = "the stock is deeply oversold"
    elif rsi_raw == "RSI<30": condition = "the stock becomes very cheap"
    elif rsi_raw == "RSI<35": condition = "the stock is starting to look like a bargain"
    elif rsi_raw == "RSI<40": condition = "the stock is entering a minor pullback"
    elif rsi_raw == "RSI<50": condition = "the stock is trading at a fair price"
    elif rsi_raw == "RSI>=70": condition = "the stock is getting too expensive (overbought)"
    else: condition = "the stock is at normal levels"

    if channel_raw == "RISING": condition += " while climbing a steady uptrend channel"
    elif channel_raw == "FALLING": condition += " while sliding down a falling channel"
    else: condition += " while trading in a flat range"

    # 3. Trigger (The Signal)
    trigger = f"I spotted a {score} signal"
    if candle_raw != "NOCNDLE":
        clean_candle = candle_raw.replace("+", " and ")
        trigger = f"I spotted a {score} signal triggered by a '{clean_candle}' pattern"

    if supp == "SUPP": trigger += " right at a historical floor (Support)"
    
    # 4. Confirmation
    confirmation = ""
    if vol == "VOLSPK": confirmation = "on a sudden burst of high volume"

    # Assemble
    story = f"{context} and {condition}, {trigger}"
    if confirmation:
        story += f" {confirmation}"
    
    return story.strip() + "."


# ── Report Sections ─────────────────────────────────────────────────────────────

def _section_proven(proven: list) -> str:
    if not proven:
        return "  _No proven elite edges yet — need more paper trade data._\n"

    lines = []
    for p in proven[:10]:   # top 10
        desc     = _parse_key(p["key"])
        win_rate = int(p["win_rate"] * 100)
        lines.append(f"  • **{p['key'].split('|')[0]} Setup** | Win: {win_rate}% | PF: {p['profit_factor']} | Expectancy: {p['expectancy']}")
        lines.append(f"    _{desc}_")
    return "\n".join(lines) + "\n"


def _section_validated(validated: list) -> str:
    if not validated:
        return "  _No validated patterns yet — gathering more trade metrics._\n"

    lines = []
    for p in validated[:10]:
        desc     = _parse_key(p["key"])
        win_rate = int(p["win_rate"] * 100)
        lines.append(f"  • **{p['key'].split('|')[0]} Setup** | Win: {win_rate}% | PF: {p['profit_factor']} | Expectancy: {p['expectancy']}")
        lines.append(f"    _{desc}_")
    return "\n".join(lines) + "\n"


def _section_learning(learning: list) -> str:
    if not learning:
        return "  _No patterns under active learning yet._\n"

    lines = []
    for p in learning[:10]:
        desc      = _parse_key(p["key"])
        win_rate  = int(p["win_rate"] * 100)
        needed    = max(0, 50 - p["trades"])
        lines.append(f"  • **{p['key'].split('|')[0]} Setup** | Win: {win_rate}% | {p['trades']} trades | (need {needed} trades to validate)")
        lines.append(f"    _{desc}_")
    return "\n".join(lines) + "\n"


def _section_watching(watching: list) -> str:
    if not watching:
        return "  _No new patterns being monitored yet._\n"

    lines = []
    for p in watching[:10]:
        desc = _parse_key(p["key"])
        lines.append(f"  • _{desc}_")
    return "\n".join(lines) + "\n"


def _section_unreliable(unreliable: list) -> str:
    if not unreliable:
        return "  _No unreliable/failed patterns identified yet (that is good!)._\n"

    lines = []
    for p in unreliable[:10]:
        desc     = _parse_key(p["key"])
        win_rate = int(p["win_rate"] * 100)
        lines.append(f"  • **{p['key'].split('|')[0]} Setup** | Win: {win_rate}% | PF: {p['profit_factor']} | Expectancy: {p['expectancy']} | **AVOID**")
        lines.append(f"    _{desc}_")
    return "\n".join(lines) + "\n"


# ── Full Report Builder ─────────────────────────────────────────────────────────

def generate_learning_report(summary: dict) -> str:
    """Build the full markdown learning report."""
    now = datetime.now().strftime("%d %b %Y  %H:%M")

    proven      = summary["proven"]
    validated   = summary["validated"]
    learning    = summary["learning"]
    watching    = summary["watching"]
    unreliable  = summary["unreliable"]
    total_p     = summary["total_patterns"]
    total_o     = summary["total_observations"]

    report = f"""# 📚 QuantEdge Bot — Intelligence Learning Report
_Generated: {now}_

---

## Knowledge Summary
- **Total pattern conditions tracked**: {total_p}
- **Total trade observations**: {total_o}
- **Proven Elite Edges (PROVEN)**: {len(proven)}
- **Validated Patterns (VALIDATED)**: {len(validated)}
- **Patterns in Training (LEARNING)**: {len(learning)}
- **Monitoring Patterns (WATCHING)**: {len(watching)}
- **Unreliable Patterns (AVOID)**: {len(unreliable)}

---

## 🔥 What I Have Proven (Elite Edges)
> These pattern setups have strong statistical significance (100+ trades) and met strict institutional criteria (Win Rate ≥ 60%, Profit Factor ≥ 1.2).
> The bot treats these setups as premium high-confidence entries.

{_section_proven(proven)}

---

## ✅ What I Have Validated (Robust Setups)
> These pattern setups are validated (50–99 trades) with positive metrics (Win Rate ≥ 55%, Profit Factor ≥ 1.1).
> They form a robust part of our daily strategy framework.

{_section_validated(validated)}

---

## 🔄 What I Am Currently Learning
> These patterns are in the active data-gathering phase (20–49 trades) to establish reliable profit factors.

{_section_learning(learning)}

---

## 🎯 What I Plan to Learn Next
> These patterns have been detected (<20 trades) but lack sufficient history to analyze. 

{_section_watching(watching)}

---

## ⚠️ Unreliable Patterns (Danger Zones to Avoid)
> These pattern setups have sufficient sample sizes (50+ trades) but failed to achieve sustainable expectancy.
> **The bot automatically filters out and blocks buy signals matching these conditions.**

{_section_unreliable(unreliable)}

---

## 📈 How the Bot Adapts
1. **Regime Filtering:** Every signal is evaluated under both Index and Stock regimes.
2. **Key Simplification:** Patterns are grouped using a simplified 7-factor key to avoid mathematical overfitting.
3. **Outcome Auditing:** After paper trades close, exact P&L, Win Rate, and Profit Factors are logged.
4. **State Transition:** Patterns dynamically advance or demote across states based on performance thresholds.
5. **Dynamic Blocking:** Proven failed/unreliable setups are blocked from entries, preserving capital.

---
_This report is automatically generated and committed to the repository daily._
"""
    return report


def generate_telegram_summary(summary: dict) -> str:
    """Build a compact Telegram version of the learning report."""
    proven      = summary["proven"]
    validated   = summary["validated"]
    learning    = summary["learning"]
    watching    = summary["watching"]
    unreliable  = summary["unreliable"]
    total_o     = summary["total_observations"]

    lines = [
        "📚 <b>DAILY LEARNING REPORT</b>",
        f"Based on <b>{total_o}</b> total trade observations\n",
    ]

    # Proven
    lines.append(f"🔥 <b>PROVEN EDGES ({len(proven)})</b>")
    if proven:
        for p in proven[:2]:
            desc = _parse_key(p["key"])
            lines.append(f"  • {desc} → {int(p['win_rate'] * 100)}% Win | PF: {p['profit_factor']}")
        if len(proven) > 2:
            lines.append(f"  ... and {len(proven) - 2} more proven edges")
    else:
        lines.append("  No proven elite edges yet")

    # Validated
    lines.append(f"\n✅ <b>VALIDATED SETUPS ({len(validated)})</b>")
    if validated:
        for p in validated[:2]:
            desc = _parse_key(p["key"])
            lines.append(f"  • {desc} → {int(p['win_rate'] * 100)}% Win | PF: {p['profit_factor']}")
    else:
        lines.append("  No validated setups yet")

    # Learning
    lines.append(f"\n🔄 <b>LEARNING ({len(learning)} patterns)</b>")
    if learning:
        for p in learning[:2]:
            desc = _parse_key(p["key"])
            lines.append(f"  • {desc} → {p['trades']}/50 trades")
    else:
        lines.append("  None yet")

    # Unreliable
    if unreliable:
        lines.append(f"\n⛔ <b>BLOCKED DANGER ZONES ({len(unreliable)})</b>")
        for p in unreliable[:2]:
            desc = _parse_key(p["key"])
            lines.append(f"  • {desc} (Win: {int(p['win_rate']*100)}% | PF: {p['profit_factor']})")

    lines.append("\n📁 Full report committed to: <code>state/learning_report.md</code>")

    return "\n".join(lines)


# ── Save Report ─────────────────────────────────────────────────────────────────

def save_report(report: str) -> None:
    """Save the learning report to state/learning_report.md."""
    os.makedirs("state", exist_ok=True)
    with open(REPORT_FILE, "w", encoding="utf-8") as f:
        f.write(report)
    logger.info("Learning report saved to %s", REPORT_FILE)


def run_learning_report(journal: dict) -> tuple:
    """
    Build, save and return the full report + telegram summary.

    Args:
        journal: The learning journal dict

    Returns:
        (full_report_str, telegram_summary_str)
    """
    summary    = get_knowledge_summary(journal)
    report     = generate_learning_report(summary)
    tg_summary = generate_telegram_summary(summary)
    save_report(report)
    return report, tg_summary
