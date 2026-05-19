"""
Regime Performance Analytics
Slices the learning journal by market regime to reveal where the edge
actually lives — and where it breaks down.

Key outputs:
  - regime_performance_table: win rate, profit factor, expectancy, sample count per regime
  - failure_taxonomy: ranked breakdown of UNRELIABLE pattern clusters
  - confidence_interval: Wilson Score 90% CI on each regime's win rate
"""

import logging
from typing import Dict, List
import pandas as pd

from analytics.confidence import wilson_ci

logger = logging.getLogger(__name__)

# Minimum trades required before reporting a regime as statistically meaningful
_MINIMUM_TRADES_FOR_REPORTING = 10


def _sample_strength(n: float) -> str:
    """Classify sample size robustness for display."""
    if n >= 100:
        return "STRONG ✅"
    if n >= 30:
        return "MODERATE ⚠️"
    return "WEAK 🔴"


def compute_regime_performance_table(journal: dict) -> pd.DataFrame:
    """
    Aggregates all pattern observations grouped by their leading regime token
    (the first field of the condition key: e.g. 'STRONG_TREND_UP').

    Returns a DataFrame with columns:
        Regime | Trades | Win Rate | CI Lower | CI Upper | Profit Factor | Expectancy | Sample Strength

    Statistical note:
        Win rates are computed from decayed trade weights, not raw counts.
        Wilson CI uses integer approximations (round to nearest trade) to remain valid.
        Regimes with fewer than MINIMUM_TRADES_FOR_REPORTING are excluded.
    """
    regime_buckets: Dict[str, Dict] = {}

    for key, p in journal.get("patterns", {}).items():
        # Extract regime prefix (first segment of pipe-separated key)
        regime = key.split("|")[0] if "|" in key else "UNKNOWN"

        if regime not in regime_buckets:
            regime_buckets[regime] = {
                "trades": 0.0,
                "wins": 0.0,
                "gross_profit": 0.0,
                "gross_loss": 0.0,
            }

        b = regime_buckets[regime]
        b["trades"] += p.get("trades", 0.0)
        b["wins"] += p.get("wins", 0.0)
        b["gross_profit"] += p.get("gross_profit", 0.0)
        b["gross_loss"] += p.get("gross_loss", 0.0)

    rows = []
    for regime, b in regime_buckets.items():
        n = b["trades"]
        if n < _MINIMUM_TRADES_FOR_REPORTING:
            continue

        wins = b["wins"]
        win_rate = wins / n if n > 0 else 0.0

        # Wilson CI — approximate integer wins from decayed float counts
        lower_ci, upper_ci = wilson_ci(int(round(wins)), int(round(n)))

        gp = b["gross_profit"]
        gl = b["gross_loss"]
        pf = round(gp / gl, 2) if gl > 0 else (round(gp, 2) if gp > 0 else 1.0)
        expectancy = round((gp - gl) / n, 3) if n > 0 else 0.0

        rows.append({
            "Regime": regime,
            "Trades": int(round(n)),
            "Win Rate": f"{win_rate*100:.1f}%",
            "90% CI": f"[{lower_ci*100:.0f}% – {upper_ci*100:.0f}%]",
            "Profit Factor": pf,
            "Expectancy": expectancy,
            "Sample Strength": _sample_strength(n),
            # Internal sort key
            "_wr": win_rate,
            "_n": n,
        })

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows).sort_values("_wr", ascending=False).drop(
        columns=["_wr", "_n"]
    )
    df.reset_index(drop=True, inplace=True)
    return df


def compute_failure_taxonomy(journal: dict) -> pd.DataFrame:
    """
    Analyses UNRELIABLE patterns to identify recurring failure clusters.

    Groups failing patterns by their dominant condition factors:
      - RSI bucket (position 1 in key)
      - Trend channel (position 3 in key)
      - Volume spike presence (position 6 in key)

    Returns a DataFrame:
        Failure Type | Condition | Occurrences | Avg Win Rate | Recommendation
    """
    failures: List[Dict] = []

    for key, p in journal.get("patterns", {}).items():
        if p.get("state") != "UNRELIABLE":
            continue

        parts = key.split("|")
        regime = parts[0] if len(parts) > 0 else "?"
        rsi_bucket = parts[1] if len(parts) > 1 else "?"
        channel = parts[3] if len(parts) > 3 else "?"
        vol_spike = parts[6] if len(parts) > 6 else "?"

        # Classify failure type by dominant stressor
        if "RSI>70" in rsi_bucket or "RSI>=70" in rsi_bucket:
            failure_type = "Overbought Entry Trap"
        elif "DOWN" in regime:
            failure_type = "Counter-Trend Buy in Downtrend"
        elif vol_spike == "NOSPK" and channel == "FALLING":
            failure_type = "Low-Volume Falling Knife"
        elif channel == "FALLING":
            failure_type = "Falling Channel False Reversal"
        else:
            failure_type = "Regime Mismatch / Mixed Signal"

        failures.append({
            "failure_type": failure_type,
            "regime": regime,
            "rsi": rsi_bucket,
            "win_rate": p.get("win_rate", 0.0),
            "trades": p.get("trades", 0.0),
        })

    if not failures:
        return pd.DataFrame()

    df = pd.DataFrame(failures)

    # Aggregate by failure type
    agg = (
        df.groupby("failure_type")
        .agg(
            Occurrences=("trades", "sum"),
            Avg_Win_Rate=("win_rate", "mean"),
        )
        .reset_index()
        .rename(columns={"failure_type": "Failure Pattern", "Avg_Win_Rate": "Avg Win Rate"})
        .sort_values("Occurrences", ascending=False)
    )

    agg["Avg Win Rate"] = agg["Avg Win Rate"].apply(lambda x: f"{x*100:.1f}%")
    agg["Occurrences"] = agg["Occurrences"].apply(lambda x: int(round(x)))
    agg["Recommendation"] = agg["Failure Pattern"].apply(_failure_recommendation)

    return agg.reset_index(drop=True)


def _failure_recommendation(failure_type: str) -> str:
    """Map failure pattern to a brief operational recommendation."""
    recs = {
        "Overbought Entry Trap": "Skip entries when RSI ≥ 70 regardless of other signals.",
        "Counter-Trend Buy in Downtrend": "Enforce long lock: no BUY signals in DOWNTREND regime.",
        "Low-Volume Falling Knife": "Require volume spike confirmation before entry.",
        "Falling Channel False Reversal": "Wait for channel breakout confirmation (2 closes above).",
        "Regime Mismatch / Mixed Signal": "Require regime alignment between stock and index.",
    }
    return recs.get(failure_type, "Review conditions manually.")
