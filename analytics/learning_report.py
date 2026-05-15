"""
Learning Report Generator
Produces a comprehensive daily learning report in three sections:

  ✅ WHAT I HAVE LEARNED   — proven patterns (≥5 trades, ≥60% win rate)
  🔄 WHAT I AM LEARNING    — patterns being tracked (2–4 trades)
  🎯 WHAT I PLAN TO LEARN  — patterns detected but no trades taken yet

The report is saved as state/learning_report.md and a compact version
is sent to Telegram.
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
    """Convert a pattern key like 'SIDEWAYS|RSI<30|S3|RISING|HAMMER|SUPP|NODIV|NOSPK'
    into a readable description."""
    parts = key.split("|")
    if len(parts) < 8:
        return key

    regime, rsi, score, channel, candle, supp, div, vol = parts[:8]

    desc = []
    desc.append(f"{regime} regime")
    desc.append(f"{rsi}")
    desc.append(f"Score {score[-1]}/4")
    if channel != "SIDEWAYS":
        desc.append(f"{channel.capitalize()} channel")
    if candle not in ("NOCNDLE", ""):
        desc.append(f"{candle.replace('+', ' + ').replace('_', ' ').title()}")
    if supp == "SUPP":
        desc.append("at support")
    if div not in ("NODIV", "None", ""):
        desc.append(f"{div.replace('_', ' ').title()}")
    if vol == "VOLSPK":
        desc.append("+ volume spike")

    return ", ".join(desc)


# ── Report Sections ─────────────────────────────────────────────────────────────

def _section_learned(learned: list) -> str:
    if not learned:
        return "  _No proven patterns yet — need more paper trade data._\n"

    lines = []
    for p in learned[:10]:   # top 10
        desc     = _parse_key(p["key"])
        win_rate = int(p["win_rate"] * 100)
        lines.append(f"  • {desc}")
        lines.append(f"    → {win_rate}% win rate over {p['trades']} trades")
    return "\n".join(lines) + "\n"


def _section_learning(learning: list) -> str:
    if not learning:
        return "  _No patterns under active tracking yet._\n"

    lines = []
    for p in learning[:10]:
        desc      = _parse_key(p["key"])
        win_rate  = int(p["win_rate"] * 100)
        needed    = max(0, 5 - p["trades"])
        if needed > 0:
            status = f"need {needed} more to confirm"
        else:
            status = "pending re-classification"
        lines.append(f"  • {desc}")
        lines.append(f"    → {win_rate}% win rate | {p['trades']} trades | {status}")
    return "\n".join(lines) + "\n"


def _section_watching(watching: list) -> str:
    if not watching:
        return "  _No new patterns detected yet._\n"

    lines = []
    for p in watching[:10]:
        desc = _parse_key(p["key"])
        lines.append(f"  • {desc}")
        lines.append(f"    → Detected, waiting for a trade signal to track outcome")
    return "\n".join(lines) + "\n"


# ── Full Report Builder ─────────────────────────────────────────────────────────

def generate_learning_report(summary: dict) -> str:
    """Build the full markdown learning report."""
    now = datetime.now().strftime("%d %b %Y  %H:%M")

    learned  = summary["learned"]
    learning = summary["learning"]
    watching = summary["watching"]
    total_p  = summary["total_patterns"]
    total_o  = summary["total_observations"]

    report = f"""# 📚 QuantEdge Bot — Learning Report
_Generated: {now}_

---

## Summary
- **Total pattern conditions tracked**: {total_p}
- **Total trade observations**: {total_o}
- **Proven patterns (LEARNED)**: {len(learned)}
- **Patterns in training (LEARNING)**: {len(learning)}
- **Patterns being watched**: {len(watching)}

---

## ✅ What I Have Learned
> These patterns have been confirmed through sufficient trade history.
> The bot will prioritise signals matching these conditions.

{_section_learned(learned)}

---

## 🔄 What I Am Currently Learning
> These patterns have been seen in 2–4 trades. Win rates are preliminary.
> More data is needed before the bot adjusts its behaviour.

{_section_learning(learning)}

---

## 🎯 What I Plan to Learn Next
> These patterns have been detected in recent market data but no trades
> have been taken yet. The bot is monitoring them actively.

{_section_watching(watching)}

---

## 📈 How the Bot Improves
1. Every day, new BUY/SELL signals are matched against pattern conditions
2. When a paper trade closes, the outcome (win/loss) is recorded
3. After 5 confirmed outcomes, a pattern moves from LEARNING → LEARNED
4. LEARNED patterns influence the dynamic filter (blocking losing conditions)
5. As more stocks are added, the knowledge base expands automatically

---
_This report is automatically generated and committed to the repository daily._
"""
    return report


def generate_telegram_summary(summary: dict) -> str:
    """Build a compact Telegram version of the learning report."""
    learned  = summary["learned"]
    learning = summary["learning"]
    watching = summary["watching"]
    total_o  = summary["total_observations"]

    lines = [
        "📚 <b>DAILY LEARNING REPORT</b>",
        f"Based on {total_o} total trade observations\n",
    ]

    # Learned
    lines.append(f"✅ <b>LEARNED ({len(learned)} proven patterns)</b>")
    if learned:
        for p in learned[:3]:
            desc = _parse_key(p["key"])
            lines.append(f"  • {desc} → {int(p['win_rate'] * 100)}% ({p['trades']} trades)")
        if len(learned) > 3:
            lines.append(f"  ... and {len(learned) - 3} more")
    else:
        lines.append("  Not enough closed trades yet")

    # Learning
    lines.append(f"\n🔄 <b>LEARNING ({len(learning)} patterns)</b>")
    if learning:
        for p in learning[:3]:
            desc = _parse_key(p["key"])
            lines.append(f"  • {desc} → {p['trades']}/5 trades")
        if len(learning) > 3:
            lines.append(f"  ... and {len(learning) - 3} more")
    else:
        lines.append("  None yet")

    # Watching
    lines.append(f"\n🎯 <b>WATCHING ({len(watching)} patterns)</b>")
    if watching:
        for p in watching[:2]:
            desc = _parse_key(p["key"])
            lines.append(f"  • {desc}")
    else:
        lines.append("  None yet")

    lines.append("\n📁 Full report committed to repo: state/learning_report.md")

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
