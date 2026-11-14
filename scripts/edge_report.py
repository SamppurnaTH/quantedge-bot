"""
Edge Report — does the strategy actually make money, and does it hold out of sample?

Replays the exact live rules (signal frame + trade model, costs included)
over the full history of the watchlist and prints R-multiple statistics:
overall, by regime, by archetype and by exit reason — split into an
in-sample and an out-of-sample period. An edge that only exists in-sample is
curve-fit; trust only what survives out-of-sample.

    python scripts/edge_report.py                 # split at 2022-01-01
    python scripts/edge_report.py --split 2023-01-01
"""
import argparse
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd

from analytics.learning_engine import simulate_history, classify_archetype


def _stats(r: pd.Series) -> dict:
    n = len(r)
    if n == 0:
        return {"n": 0}
    mean, sd = r.mean(), r.std(ddof=1) if n > 1 else 0.0
    gp, gl = r[r > 0].sum(), -r[r <= 0].sum()
    return {"n": n, "win%": round((r > 0).mean() * 100, 1), "avg R": round(mean, 3),
            "t-stat": round(mean / (sd / math.sqrt(n)), 2) if sd > 0 else float("nan"),
            "PF": round(gp / gl, 2) if gl > 0 else float("inf"), "total R": round(r.sum(), 1)}


def _table(df: pd.DataFrame, by: str, split: str) -> pd.DataFrame:
    rows = []
    for name, g in df.groupby(by):
        ins, oos = g[g["date"] < split]["r"], g[g["date"] >= split]["r"]
        rows.append({by: name, **{f"IS {k}": v for k, v in _stats(ins).items()},
                     **{f"OOS {k}": v for k, v in _stats(oos).items()}})
    return pd.DataFrame(rows).set_index(by)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="2022-01-01", help="out-of-sample start date")
    ap.add_argument("--csv", default=os.path.join("logs", "edge_report_trades.csv"))
    args = ap.parse_args()

    df = pd.DataFrame(list(simulate_history()))
    if df.empty:
        print("No trades simulated — check data access.")
        return
    df["archetype"] = df["key"].map(classify_archetype)
    df["all"] = "ALL"
    os.makedirs(os.path.dirname(args.csv), exist_ok=True)
    df.to_csv(args.csv, index=False)

    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", 20)
    for by in ("all", "regime", "archetype", "exit_reason"):
        print(f"\n=== by {by}  (IS < {args.split} ≤ OOS) ===")
        print(_table(df, by, args.split).to_string())
    print(f"\nTrade list saved to {args.csv}")
    print("Rule of thumb: |t-stat| ≥ 2 with ≥ 100 trades is evidence; anything less is noise.")


if __name__ == "__main__":
    main()
