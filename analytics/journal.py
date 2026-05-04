"""
Trade Journal Analytics

Reads the signals_history.csv and paper_portfolio.json to extract
actionable intelligence — not just raw logs.

Key questions answered:
  - Win rate by signal score (score 3 vs 2 vs 1)
  - Win rate by regime
  - Win rate by stock
  - Average R:R achieved vs planned
  - Best and worst performing conditions
  - Equity curve from paper trades
"""

import os
import json
import logging
from typing import List, Dict, Any

import pandas as pd

from logger.trade_tracker import SIGNALS_LOG, BACKTEST_LOG
from execution.paper_trader import PAPER_STATE_FILE

logger = logging.getLogger(__name__)


# ── Data loaders ──────────────────────────────────────────────────────────────

def load_signal_history() -> pd.DataFrame:
    """Load the signals_history.csv into a DataFrame."""
    if not os.path.exists(SIGNALS_LOG):
        logger.warning("No signal history found at %s", SIGNALS_LOG)
        return pd.DataFrame()
    df = pd.read_csv(SIGNALS_LOG, parse_dates=["timestamp"])
    return df


def load_backtest_history() -> pd.DataFrame:
    """Load the backtest_results.csv into a DataFrame."""
    if not os.path.exists(BACKTEST_LOG):
        logger.warning("No backtest history found at %s", BACKTEST_LOG)
        return pd.DataFrame()
    df = pd.read_csv(BACKTEST_LOG, parse_dates=["timestamp"])
    return df


def load_paper_trades() -> List[dict]:
    """Load closed trades from the paper portfolio JSON."""
    if not os.path.exists(PAPER_STATE_FILE):
        logger.warning("No paper portfolio found at %s", PAPER_STATE_FILE)
        return []
    with open(PAPER_STATE_FILE, "r", encoding="utf-8") as f:
        state = json.load(f)
    return state.get("closed_trades", [])


# ── Analytics ─────────────────────────────────────────────────────────────────

def win_rate_by_score(trades: List[dict]) -> pd.DataFrame:
    """
    Win rate broken down by signal score (0–3).

    Args:
        trades: List of closed trade dicts from paper portfolio

    Returns:
        DataFrame with columns: score, total, wins, losses, win_rate_pct, avg_pnl
    """
    if not trades:
        return pd.DataFrame(columns=["score", "total", "wins", "losses", "win_rate_pct", "avg_pnl"])

    df = pd.DataFrame(trades)
    if "score" not in df.columns:
        df["score"] = 0

    rows = []
    for score, group in df.groupby("score"):
        wins     = (group["pnl"] > 0).sum()
        losses   = (group["pnl"] <= 0).sum()
        total    = len(group)
        win_rate = wins / total * 100 if total else 0
        avg_pnl  = group["pnl"].mean()
        rows.append({
            "score":        int(score),
            "total":        total,
            "wins":         int(wins),
            "losses":       int(losses),
            "win_rate_pct": round(win_rate, 1),
            "avg_pnl":      round(avg_pnl, 2),
        })

    result = pd.DataFrame(rows).sort_values("score", ascending=False)
    return result


def win_rate_by_symbol(trades: List[dict]) -> pd.DataFrame:
    """Win rate and average P&L per stock."""
    if not trades:
        return pd.DataFrame()

    df = pd.DataFrame(trades)
    rows = []
    for symbol, group in df.groupby("symbol"):
        wins     = (group["pnl"] > 0).sum()
        total    = len(group)
        win_rate = wins / total * 100 if total else 0
        rows.append({
            "symbol":       symbol,
            "total":        total,
            "wins":         int(wins),
            "losses":       int(total - wins),
            "win_rate_pct": round(win_rate, 1),
            "total_pnl":    round(group["pnl"].sum(), 2),
            "avg_pnl":      round(group["pnl"].mean(), 2),
            "best_trade":   round(group["pnl"].max(), 2),
            "worst_trade":  round(group["pnl"].min(), 2),
        })

    return pd.DataFrame(rows).sort_values("total_pnl", ascending=False)


def win_rate_by_exit_reason(trades: List[dict]) -> pd.DataFrame:
    """How often does each exit type (SL / TP / Signal) win?"""
    if not trades:
        return pd.DataFrame()

    df = pd.DataFrame(trades)
    rows = []
    for reason, group in df.groupby("exit_reason"):
        wins     = (group["pnl"] > 0).sum()
        total    = len(group)
        win_rate = wins / total * 100 if total else 0
        rows.append({
            "exit_reason":  reason,
            "total":        total,
            "wins":         int(wins),
            "win_rate_pct": round(win_rate, 1),
            "avg_pnl":      round(group["pnl"].mean(), 2),
            "total_pnl":    round(group["pnl"].sum(), 2),
        })

    return pd.DataFrame(rows).sort_values("total_pnl", ascending=False)


def equity_curve(trades: List[dict], initial_capital: float = 100_000) -> pd.DataFrame:
    """
    Build an equity curve from closed paper trades.

    Returns:
        DataFrame with columns: exit_date, pnl, cumulative_pnl, equity
    """
    if not trades:
        return pd.DataFrame()

    df = pd.DataFrame(trades)
    df["exit_date"]      = pd.to_datetime(df["exit_date"])
    df                   = df.sort_values("exit_date")
    df["cumulative_pnl"] = df["pnl"].cumsum()
    df["equity"]         = initial_capital + df["cumulative_pnl"]
    return df[["exit_date", "symbol", "pnl", "cumulative_pnl", "equity", "exit_reason"]]


def signal_quality_report(signal_df: pd.DataFrame) -> pd.DataFrame:
    """
    Analyse signal history to find which conditions produce the most BUY signals.
    Useful for tuning thresholds.

    Returns:
        DataFrame summarising BUY signal frequency by score and trend
    """
    if signal_df.empty:
        return pd.DataFrame()

    buys = signal_df[signal_df["signal"] == "BUY"].copy()
    if buys.empty:
        return pd.DataFrame({"message": ["No BUY signals in history yet."]})

    summary = (
        buys.groupby(["score", "trend"])
        .agg(
            count=("symbol", "count"),
            avg_rsi=("rsi", "mean"),
            avg_atr=("atr", "mean"),
        )
        .reset_index()
    )
    summary["avg_rsi"] = summary["avg_rsi"].round(2)
    summary["avg_atr"] = summary["avg_atr"].round(2)
    return summary.sort_values(["score", "count"], ascending=[False, False])


# ── Report printer ────────────────────────────────────────────────────────────

def print_journal_report() -> None:
    """Print the full trade journal analytics to console."""
    trades     = load_paper_trades()
    signal_df  = load_signal_history()
    backtest_df = load_backtest_history()

    print("\n" + "=" * 65)
    print("  TRADE JOURNAL ANALYTICS")
    print("=" * 65)

    # ── Paper trade summary ───────────────────────────────────────────────────
    if not trades:
        print("\n  No closed paper trades yet. Run --paper mode to accumulate data.")
    else:
        print(f"\n  Total closed paper trades: {len(trades)}")

        # Win rate by score
        score_df = win_rate_by_score(trades)
        if not score_df.empty:
            print("\n  Win Rate by Signal Score:")
            print(f"  {'Score':>6} {'Trades':>7} {'Wins':>6} {'Losses':>7} {'Win%':>7} {'Avg P&L':>10}")
            print("  " + "-" * 48)
            for _, row in score_df.iterrows():
                insight = " ← best" if row["score"] == score_df.loc[score_df["win_rate_pct"].idxmax(), "score"] else ""
                print(
                    f"  {int(row['score']):>6} {int(row['total']):>7} {int(row['wins']):>6} "
                    f"{int(row['losses']):>7} {row['win_rate_pct']:>6.1f}% {row['avg_pnl']:>10,.2f}{insight}"
                )

        # Win rate by symbol
        sym_df = win_rate_by_symbol(trades)
        if not sym_df.empty:
            print("\n  Win Rate by Symbol:")
            print(f"  {'Symbol':<18} {'Trades':>7} {'Win%':>7} {'Total P&L':>12} {'Avg P&L':>10}")
            print("  " + "-" * 58)
            for _, row in sym_df.iterrows():
                sign = "+" if row["total_pnl"] >= 0 else ""
                print(
                    f"  {row['symbol']:<18} {int(row['total']):>7} {row['win_rate_pct']:>6.1f}% "
                    f"  {sign}{row['total_pnl']:>10,.2f}  {sign}{row['avg_pnl']:>8,.2f}"
                )

        # Win rate by exit reason
        exit_df = win_rate_by_exit_reason(trades)
        if not exit_df.empty:
            print("\n  Exit Reason Analysis:")
            print(f"  {'Exit Reason':<16} {'Trades':>7} {'Win%':>7} {'Avg P&L':>10} {'Total P&L':>12}")
            print("  " + "-" * 56)
            for _, row in exit_df.iterrows():
                sign = "+" if row["total_pnl"] >= 0 else ""
                print(
                    f"  {row['exit_reason']:<16} {int(row['total']):>7} {row['win_rate_pct']:>6.1f}% "
                    f"{row['avg_pnl']:>10,.2f}  {sign}{row['total_pnl']:>10,.2f}"
                )

        # Equity curve summary
        eq_df = equity_curve(trades)
        if not eq_df.empty:
            peak    = eq_df["equity"].max()
            trough  = eq_df["equity"].min()
            final   = eq_df["equity"].iloc[-1]
            max_dd  = round((peak - trough) / peak * 100, 2)
            print(f"\n  Equity Curve:")
            print(f"  Peak: {peak:,.2f}   Trough: {trough:,.2f}   Final: {final:,.2f}   Max DD: {max_dd:.2f}%")

    # ── Signal quality ────────────────────────────────────────────────────────
    if not signal_df.empty:
        sq = signal_quality_report(signal_df)
        if not sq.empty and "message" not in sq.columns:
            print("\n  BUY Signal Frequency by Score + Trend:")
            print(sq.to_string(index=False))

    # ── Backtest comparison ───────────────────────────────────────────────────
    if not backtest_df.empty:
        latest = backtest_df.sort_values("timestamp").groupby("symbol").last().reset_index()
        print("\n  Latest Backtest Results per Symbol:")
        print(f"  {'Symbol':<18} {'P&L%':>8} {'MaxDD%':>8} {'Trades':>7} {'Win%':>7}")
        print("  " + "-" * 52)
        for _, row in latest.sort_values("pnl_pct", ascending=False).iterrows():
            sign = "+" if row["pnl_pct"] >= 0 else ""
            print(
                f"  {row['symbol']:<18} {sign}{row['pnl_pct']:>7.2f}% "
                f"{row['max_drawdown']:>7.2f}% {int(row['total_trades']):>7} "
                f"{row['win_rate_pct']:>6.1f}%"
            )

    print("\n" + "=" * 65 + "\n")
