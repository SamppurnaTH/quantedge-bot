"""
Performance Dashboard

Answers the one question that matters:
  "Where is my edge actually coming from?"

Sections:
  1. Edge Summary        — which conditions produce positive expectancy
  2. Regime Performance  — P&L and win rate broken down by regime
  3. Score Performance   — does higher score = better outcome?
  4. Version Comparison  — A/B test results across strategy versions
  5. Dynamic Filter      — which symbol/regime/score combos to skip
  6. Equity Curve        — visual ASCII equity curve in terminal
"""

import os
import json
import logging
from typing import List, Dict, Optional
from datetime import datetime

import pandas as pd
import numpy as np

from logger.trade_tracker import SIGNALS_LOG, BACKTEST_LOG
from execution.paper_trader import PAPER_STATE_FILE

logger = logging.getLogger(__name__)

# Where A/B test results are stored
AB_RESULTS_FILE = os.path.join("logs", "ab_test_results.json")


# ── Data loading ──────────────────────────────────────────────────────────────

def _load_trades() -> pd.DataFrame:
    if not os.path.exists(PAPER_STATE_FILE):
        return pd.DataFrame()
    with open(PAPER_STATE_FILE, "r", encoding="utf-8") as f:
        state = json.load(f)
    trades = state.get("closed_trades", [])
    if not trades:
        return pd.DataFrame()
    df = pd.DataFrame(trades)
    df["exit_date"]  = pd.to_datetime(df["exit_date"])
    df["entry_date"] = pd.to_datetime(df["entry_date"])
    df["hold_days"]  = (df["exit_date"] - df["entry_date"]).dt.days
    df["win"]        = df["pnl"] > 0
    return df


def _load_backtest() -> pd.DataFrame:
    if not os.path.exists(BACKTEST_LOG):
        return pd.DataFrame()
    return pd.read_csv(BACKTEST_LOG, parse_dates=["timestamp"])


def _load_ab_results() -> List[dict]:
    if not os.path.exists(AB_RESULTS_FILE):
        return []
    with open(AB_RESULTS_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


# ── Core metrics ──────────────────────────────────────────────────────────────

def _expectancy(df: pd.DataFrame) -> float:
    """
    Expectancy = (win_rate × avg_win) - (loss_rate × avg_loss)
    Positive expectancy = edge exists.
    """
    if df.empty:
        return 0.0
    wins   = df[df["win"]]
    losses = df[~df["win"]]
    wr     = len(wins) / len(df)
    avg_w  = wins["pnl"].mean() if len(wins) else 0
    avg_l  = abs(losses["pnl"].mean()) if len(losses) else 0
    return round((wr * avg_w) - ((1 - wr) * avg_l), 2)


def _metrics(df: pd.DataFrame) -> dict:
    """Compute standard metrics for a group of trades."""
    if df.empty:
        return {"total": 0, "wins": 0, "win_rate": 0, "total_pnl": 0,
                "avg_pnl": 0, "avg_win": 0, "avg_loss": 0, "expectancy": 0,
                "max_loss": 0, "best_trade": 0}
    wins   = df[df["win"]]
    losses = df[~df["win"]]
    return {
        "total":      len(df),
        "wins":       len(wins),
        "win_rate":   round(len(wins) / len(df) * 100, 1),
        "total_pnl":  round(df["pnl"].sum(), 2),
        "avg_pnl":    round(df["pnl"].mean(), 2),
        "avg_win":    round(wins["pnl"].mean(), 2) if len(wins) else 0,
        "avg_loss":   round(losses["pnl"].mean(), 2) if len(losses) else 0,
        "expectancy": _expectancy(df),
        "max_loss":   round(df["pnl"].min(), 2),
        "best_trade": round(df["pnl"].max(), 2),
    }


# ── Dashboard sections ────────────────────────────────────────────────────────

def edge_summary(df: pd.DataFrame) -> dict:
    """
    Top-level edge summary.
    Returns overall metrics + identifies best and worst conditions.
    """
    overall = _metrics(df)

    # Best condition = highest expectancy among regime × score combos
    best_condition  = None
    worst_condition = None

    if not df.empty and "regime" in df.columns and "score" in df.columns:
        groups = []
        for (regime, score), group in df.groupby(["regime", "score"]):
            if len(group) >= 2:   # need at least 2 trades to be meaningful
                m = _metrics(group)
                groups.append({"regime": regime, "score": score, **m})

        if groups:
            groups_df = pd.DataFrame(groups)
            best_idx  = groups_df["expectancy"].idxmax()
            worst_idx = groups_df["expectancy"].idxmin()
            best_condition  = groups_df.loc[best_idx].to_dict()
            worst_condition = groups_df.loc[worst_idx].to_dict()

    return {
        "overall":         overall,
        "best_condition":  best_condition,
        "worst_condition": worst_condition,
    }


def regime_performance(df: pd.DataFrame) -> pd.DataFrame:
    """P&L and win rate broken down by regime."""
    if df.empty or "regime" not in df.columns:
        return pd.DataFrame()

    rows = []
    for regime, group in df.groupby("regime"):
        m = _metrics(group)
        rows.append({"regime": regime, **m})

    result = pd.DataFrame(rows)
    result = result.sort_values("expectancy", ascending=False)
    return result


def score_performance(df: pd.DataFrame) -> pd.DataFrame:
    """Does higher score = better outcome? This answers that."""
    if df.empty or "score" not in df.columns:
        return pd.DataFrame()

    rows = []
    for score, group in df.groupby("score"):
        m = _metrics(group)
        rows.append({"score": int(score), **m})

    result = pd.DataFrame(rows)
    result = result.sort_values("score", ascending=False)
    return result


def symbol_performance(df: pd.DataFrame) -> pd.DataFrame:
    """Which symbols are actually profitable?"""
    if df.empty or "symbol" not in df.columns:
        return pd.DataFrame()

    rows = []
    for symbol, group in df.groupby("symbol"):
        m = _metrics(group)
        rows.append({"symbol": symbol, **m})

    result = pd.DataFrame(rows)
    result = result.sort_values("total_pnl", ascending=False)
    return result


def equity_curve_ascii(df: pd.DataFrame, width: int = 60, height: int = 12) -> str:
    """
    Render an ASCII equity curve for terminal display.

    Args:
        df:     Trades DataFrame with pnl column
        width:  Chart width in characters
        height: Chart height in lines

    Returns:
        Multi-line string with the ASCII chart
    """
    if df.empty:
        return "  No trades to plot."

    equity = df.sort_values("exit_date")["pnl"].cumsum().values
    if len(equity) < 2:
        return "  Need at least 2 trades to plot."

    # Downsample to width
    indices = np.linspace(0, len(equity) - 1, min(width, len(equity))).astype(int)
    values  = equity[indices]

    v_min, v_max = values.min(), values.max()
    v_range = v_max - v_min if v_max != v_min else 1

    # Build grid
    grid = [[" "] * len(values) for _ in range(height)]

    for x, val in enumerate(values):
        y = int((val - v_min) / v_range * (height - 1))
        y = height - 1 - y   # flip: top = high
        grid[y][x] = "█" if val >= 0 else "▓"

    # Add zero line
    zero_y = int((0 - v_min) / v_range * (height - 1))
    zero_y = height - 1 - zero_y
    if 0 <= zero_y < height:
        for x in range(len(values)):
            if grid[zero_y][x] == " ":
                grid[zero_y][x] = "─"

    lines = []
    for i, row in enumerate(grid):
        if i == 0:
            label = f"{v_max:+8.0f} │"
        elif i == height - 1:
            label = f"{v_min:+8.0f} │"
        elif i == height // 2:
            label = f"{'0':>8} │"
        else:
            label = "         │"
        lines.append(label + "".join(row))

    lines.append("         └" + "─" * len(values))
    lines.append(f"           {df['exit_date'].min().strftime('%Y-%m')} → {df['exit_date'].max().strftime('%Y-%m')}")
    return "\n".join(lines)


# ── Dynamic filter ────────────────────────────────────────────────────────────

def build_dynamic_filter(df: pd.DataFrame, min_trades: int = 3) -> dict:
    """
    Build a filter dict from journal data.
    Conditions with negative expectancy and enough trades → SKIP.

    Args:
        df:         Trades DataFrame
        min_trades: Minimum trades needed before a condition is filtered

    Returns:
        dict with 'skip_regimes', 'skip_symbols', 'skip_score_below'
    """
    filter_rules = {
        "skip_regimes":    [],
        "skip_symbols":    [],
        "skip_score_below": 0,
        "source":          "journal",
        "generated_at":    datetime.now().strftime("%Y-%m-%d %H:%M"),
    }

    if df.empty:
        return filter_rules

    # Regimes with negative expectancy (enough data)
    if "regime" in df.columns:
        for regime, group in df.groupby("regime"):
            if len(group) >= min_trades and _expectancy(group) < 0:
                filter_rules["skip_regimes"].append(str(regime))
                logger.info("Dynamic filter: skip regime %s (expectancy=%.2f)", regime, _expectancy(group))

    # Symbols with negative total P&L (enough data)
    if "symbol" in df.columns:
        for symbol, group in df.groupby("symbol"):
            if len(group) >= min_trades and group["pnl"].sum() < 0:
                filter_rules["skip_symbols"].append(str(symbol))
                logger.info("Dynamic filter: skip symbol %s (total_pnl=%.2f)", symbol, group["pnl"].sum())

    # Minimum score: find lowest score with positive expectancy
    if "score" in df.columns:
        best_min_score = 0
        # Check cumulative: if using score X and above gives positive expectancy
        for score in sorted(df["score"].unique()):
            group = df[df["score"] >= score]
            if len(group) >= min_trades and _expectancy(group) > 0:
                best_min_score = int(score)
                break
        filter_rules["skip_score_below"] = best_min_score

    return filter_rules


def run_auto_optimization() -> dict:
    """
    Load latest trades, rebuild filter, and save it.
    Returns the new filter rules.
    """
    df = _load_trades()
    if df.empty:
        return load_dynamic_filter()
        
    new_filter = build_dynamic_filter(df)
    save_dynamic_filter(new_filter)
    return new_filter


def save_dynamic_filter(filter_rules: dict) -> None:
    """Save dynamic filter to logs for use by signal generator."""
    path = os.path.join("logs", "dynamic_filter.json")
    os.makedirs("logs", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(filter_rules, f, indent=2)
    logger.info("Dynamic filter saved to %s", path)


def load_dynamic_filter() -> dict:
    """Load dynamic filter. Returns empty filter if not found."""
    path = os.path.join("logs", "dynamic_filter.json")
    if not os.path.exists(path):
        return {"skip_regimes": [], "skip_symbols": [], "skip_score_below": 0}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# ── A/B test storage ──────────────────────────────────────────────────────────

def save_ab_result(version: str, symbol: str, metrics: dict) -> None:
    """Append an A/B test result to the results file."""
    results = _load_ab_results()
    results.append({
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "version":   version,
        "symbol":    symbol,
        **metrics,
    })
    os.makedirs("logs", exist_ok=True)
    with open(AB_RESULTS_FILE, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)


# ── Main dashboard printer ────────────────────────────────────────────────────

def print_dashboard() -> None:
    """Print the full performance dashboard to console."""
    df          = _load_trades()
    backtest_df = _load_backtest()
    ab_results  = _load_ab_results()

    W = 70
    print("\n" + "═" * W)
    print("  PERFORMANCE DASHBOARD")
    print("═" * W)

    # ── Section 1: Edge Summary ───────────────────────────────────────────────
    print("\n  ┌─ EDGE SUMMARY ─────────────────────────────────────────────┐")
    if df.empty:
        print("  │  No closed paper trades yet.                               │")
        print("  │  Run: python main.py --paper  to start accumulating data.  │")
        print("  └────────────────────────────────────────────────────────────┘")
    else:
        summary = edge_summary(df)
        o = summary["overall"]
        exp_icon = "✅" if o["expectancy"] > 0 else "❌"
        print(f"  │  Trades: {o['total']}   Win Rate: {o['win_rate']:.1f}%   "
              f"Expectancy: {o['expectancy']:+.2f} {exp_icon}")
        print(f"  │  Total P&L: {o['total_pnl']:+,.2f}   "
              f"Avg Win: {o['avg_win']:,.2f}   Avg Loss: {o['avg_loss']:,.2f}")

        if summary["best_condition"]:
            bc = summary["best_condition"]
            print(f"  │  Best condition : {bc['regime']} score={int(bc['score'])}  "
                  f"→ expectancy {bc['expectancy']:+.2f}")
        if summary["worst_condition"]:
            wc = summary["worst_condition"]
            print(f"  │  Worst condition: {wc['regime']} score={int(wc['score'])}  "
                  f"→ expectancy {wc['expectancy']:+.2f}")
        print("  └────────────────────────────────────────────────────────────┘")

    # ── Section 2: Regime Performance ────────────────────────────────────────
    if not df.empty and "regime" in df.columns:
        print("\n  ┌─ REGIME PERFORMANCE ───────────────────────────────────────┐")
        rp = regime_performance(df)
        if not rp.empty:
            print(f"  │  {'Regime':<22} {'Trades':>6} {'Win%':>6} {'Avg P&L':>9} {'Expect':>8}")
            print("  │  " + "─" * 56)
            for _, row in rp.iterrows():
                exp_icon = "✅" if row["expectancy"] > 0 else "❌"
                print(
                    f"  │  {str(row['regime']):<22} {int(row['total']):>6} "
                    f"{row['win_rate']:>5.1f}% {row['avg_pnl']:>9,.2f} "
                    f"{row['expectancy']:>+7.2f} {exp_icon}"
                )
        print("  └────────────────────────────────────────────────────────────┘")

    # ── Section 3: Score Performance ─────────────────────────────────────────
    if not df.empty and "score" in df.columns:
        print("\n  ┌─ SCORE PERFORMANCE (does higher score = better?) ──────────┐")
        sp = score_performance(df)
        if not sp.empty:
            print(f"  │  {'Score':>6} {'Trades':>7} {'Win%':>6} {'Avg P&L':>9} {'Expect':>8}")
            print("  │  " + "─" * 42)
            for _, row in sp.iterrows():
                exp_icon = "✅" if row["expectancy"] > 0 else "❌"
                print(
                    f"  │  {int(row['score']):>6} {int(row['total']):>7} "
                    f"{row['win_rate']:>5.1f}% {row['avg_pnl']:>9,.2f} "
                    f"{row['expectancy']:>+7.2f} {exp_icon}"
                )
        print("  └────────────────────────────────────────────────────────────┘")

    # ── Section 4: Symbol Performance ────────────────────────────────────────
    if not df.empty:
        print("\n  ┌─ SYMBOL PERFORMANCE ───────────────────────────────────────┐")
        symp = symbol_performance(df)
        if not symp.empty:
            print(f"  │  {'Symbol':<18} {'Trades':>6} {'Win%':>6} {'Total P&L':>11} {'Expect':>8}")
            print("  │  " + "─" * 56)
            for _, row in symp.iterrows():
                exp_icon = "✅" if row["expectancy"] > 0 else "❌"
                sign = "+" if row["total_pnl"] >= 0 else ""
                print(
                    f"  │  {str(row['symbol']):<18} {int(row['total']):>6} "
                    f"{row['win_rate']:>5.1f}% {sign}{row['total_pnl']:>10,.2f} "
                    f"{row['expectancy']:>+7.2f} {exp_icon}"
                )
        print("  └────────────────────────────────────────────────────────────┘")

    # ── Section 5: Equity Curve ───────────────────────────────────────────────
    if not df.empty:
        print("\n  ┌─ EQUITY CURVE ─────────────────────────────────────────────┐")
        chart = equity_curve_ascii(df)
        for line in chart.split("\n"):
            print(f"  │  {line}")
        print("  └────────────────────────────────────────────────────────────┘")

    # ── Section 6: Dynamic Filter ─────────────────────────────────────────────
    if not df.empty:
        print("\n  ┌─ DYNAMIC FILTER (auto-generated from journal) ─────────────┐")
        filt = build_dynamic_filter(df)
        save_dynamic_filter(filt)

        if filt["skip_regimes"]:
            print(f"  │  ⛔ Skip regimes  : {', '.join(filt['skip_regimes'])}")
        else:
            print("  │  ✅ No regimes to skip yet")

        if filt["skip_symbols"]:
            print(f"  │  ⛔ Skip symbols  : {', '.join(filt['skip_symbols'])}")
        else:
            print("  │  ✅ No symbols to skip yet")

        if filt["skip_score_below"] > 0:
            print(f"  │  ⛔ Skip score <  : {filt['skip_score_below']}")
        else:
            print("  │  ✅ No score filter yet")

        print(f"  │  Generated: {filt['generated_at']}")
        print("  └────────────────────────────────────────────────────────────┘")

    # ── Section 7: A/B Test Results ───────────────────────────────────────────
    if ab_results:
        print("\n  ┌─ A/B TEST RESULTS ─────────────────────────────────────────┐")
        ab_df = pd.DataFrame(ab_results)
        summary = ab_df.groupby("version").agg(
            symbols=("symbol", "count"),
            avg_pnl_pct=("pnl_pct", "mean"),
            avg_win_rate=("win_rate_pct", "mean"),
            avg_drawdown=("max_drawdown", "mean"),
            avg_trades=("total_trades", "mean"),
        ).reset_index()

        print(f"  │  {'Version':<20} {'Symbols':>7} {'Avg P&L%':>9} {'Win%':>7} {'MaxDD%':>7} {'Trades':>7}")
        print("  │  " + "─" * 60)
        for _, row in summary.sort_values("avg_pnl_pct", ascending=False).iterrows():
            sign = "+" if row["avg_pnl_pct"] >= 0 else ""
            print(
                f"  │  {str(row['version']):<20} {int(row['symbols']):>7} "
                f"{sign}{row['avg_pnl_pct']:>8.2f}% {row['avg_win_rate']:>6.1f}% "
                f"{row['avg_drawdown']:>6.2f}% {row['avg_trades']:>7.1f}"
            )
        print("  └────────────────────────────────────────────────────────────┘")

    # ── Section 8: Backtest Summary ───────────────────────────────────────────
    if not backtest_df.empty:
        print("\n  ┌─ LATEST BACKTEST RESULTS ──────────────────────────────────┐")
        latest = backtest_df.sort_values("timestamp").groupby("symbol").last().reset_index()
        print(f"  │  {'Symbol':<18} {'P&L%':>8} {'MaxDD%':>8} {'Trades':>7} {'Win%':>7}")
        print("  │  " + "─" * 52)
        for _, row in latest.sort_values("pnl_pct", ascending=False).iterrows():
            sign = "+" if row["pnl_pct"] >= 0 else ""
            print(
                f"  │  {str(row['symbol']):<18} {sign}{row['pnl_pct']:>7.2f}% "
                f"{row['max_drawdown']:>7.2f}% {int(row['total_trades']):>7} "
                f"{row['win_rate_pct']:>6.1f}%"
            )
        print("  └────────────────────────────────────────────────────────────┘")

    print("\n" + "═" * W + "\n")
