"""
Phase 7 — Trade Tracker
Persists signal history and backtest results to CSV for review.
"""

import os
import csv
import logging
from datetime import datetime
from typing import List

logger = logging.getLogger(__name__)

SIGNALS_LOG  = os.path.join("logs", "signals_history.csv")
BACKTEST_LOG = os.path.join("logs", "backtest_results.csv")

_SIGNAL_FIELDS = [
    "timestamp", "symbol", "date", "close", "rsi",
    "ma_200", "atr", "score", "signal", "trend",
    "market_up", "vol_confirmed",
    "shares", "stop_loss", "take_profit", "risk_reward", "sl_method",
]

_BACKTEST_FIELDS = [
    "timestamp", "symbol", "start_value", "end_value",
    "pnl", "pnl_pct", "max_drawdown", "sharpe_ratio",
    "total_trades", "won_trades", "lost_trades", "win_rate_pct",
    "avg_win", "avg_loss",
]


def log_signals(results: List[dict]) -> None:
    """Append signal scan results to the signals history CSV."""
    _ensure_csv(SIGNALS_LOG, _SIGNAL_FIELDS)
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    with open(SIGNALS_LOG, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=_SIGNAL_FIELDS, extrasaction="ignore")
        for r in results:
            if "error" in r:
                continue
            rr = r.get("risk_report", {})
            row = {
                "timestamp":    ts,
                "symbol":       r.get("symbol"),
                "date":         r.get("date"),
                "close":        r.get("close"),
                "rsi":          r.get("rsi"),
                "ma_200":       r.get("ma_200"),
                "atr":          r.get("atr"),
                "score":        r.get("score"),
                "signal":       r.get("signal"),
                "trend":        r.get("trend"),
                "market_up":    r.get("market_up"),
                "vol_confirmed": r.get("vol_confirmed"),
                "shares":       rr.get("shares", ""),
                "stop_loss":    rr.get("stop_loss", ""),
                "take_profit":  rr.get("take_profit", ""),
                "risk_reward":  rr.get("risk_reward", ""),
                "sl_method":    rr.get("sl_method", ""),
            }
            writer.writerow(row)

    logger.info("Signals logged to %s", SIGNALS_LOG)


def log_backtest(metrics: dict) -> None:
    """Append backtest metrics to the backtest results CSV."""
    _ensure_csv(BACKTEST_LOG, _BACKTEST_FIELDS)
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    with open(BACKTEST_LOG, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=_BACKTEST_FIELDS, extrasaction="ignore")
        writer.writerow({"timestamp": ts, **metrics})

    logger.info("Backtest results logged to %s", BACKTEST_LOG)


def _ensure_csv(path: str, fields: List[str]) -> None:
    """Create CSV with header row if it doesn't exist."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if not os.path.exists(path):
        with open(path, "w", newline="", encoding="utf-8") as f:
            csv.DictWriter(f, fieldnames=fields).writeheader()
