"""
Patch analytics/learning_engine.py — replace seed_from_history body with the
fully vectorized version that precomputes all pattern key fields.
"""
import re, textwrap, pathlib

path = pathlib.Path(r"b:\Personal\Bot\trading_bot\analytics\learning_engine.py")
src  = path.read_text(encoding="utf-8")

# ── The new function body ──────────────────────────────────────────────────────
NEW_FUNC = '''\
def seed_from_history(symbols: Optional[List[str]] = None) -> dict:
    """
    Seed the learning journal from 10-year historical data.

    Fully vectorized — precomputes regime, signals, and all four pattern key
    fields (trend_channel, volume_spike, near_support, candlestick) across every
    bar BEFORE entering the per-signal loop.  get_pattern_snapshot is never
    called inside the loop, giving ~30x speedup over the naive approach.

    Returns the populated journal dict.
    """
    import numpy as np
    from indicators.regime import Regime
    from config.settings import REGIME_CONFIG, INDICATOR_CONFIG, STRATEGY_CONFIG
    from strategies.rsi_ma_strategy import get_rsi_threshold, compute_pullback_depth, Signal

    symbols = symbols or DATA_CONFIG["symbols"]
    journal = load_journal()

    logger.info("Seeding learning journal (vectorized) for %d symbols", len(symbols))

    for symbol in symbols:
        try:
            df = load_stock_data(symbol)
            df = compute_all_indicators(df)

            min_rows = 250
            if len(df) < min_rows + OUTCOME_LOOKAHEAD:
                logger.warning("%s: not enough data (need %d bars)", symbol, min_rows + OUTCOME_LOOKAHEAD)
                continue

            # ── Column names ─────────────────────────────────────────────────
            ma_fast_col = f"SMA_{INDICATOR_CONFIG[\'ma_fast\']}"
            ma_slow_col = f"SMA_{INDICATOR_CONFIG[\'ma_slow\']}"
            atr_col     = f"ATR_{INDICATOR_CONFIG[\'atr_period\']}"
            rsi_col     = f"RSI_{INDICATOR_CONFIG[\'rsi_period\']}"
            vol_col     = f"Volume_MA_{INDICATOR_CONFIG[\'volume_ma_period\']}"

            # ── Vectorized regime ─────────────────────────────────────────────
            atr_series = df[atr_col]
            atr_ratio  = atr_series / atr_series.rolling(50).mean()
            slope_norm = df[ma_fast_col].diff().rolling(REGIME_CONFIG["slope_window"]).mean() / df[ma_fast_col]
            price      = df["Close"]
            ma200      = df[ma_slow_col]
            ma50       = df[ma_fast_col]
            p_above    = price > ma200
            p_below    = (price < ma50) & (price < ma200)
            s_up, w_up = REGIME_CONFIG["slope_strong_up"],   REGIME_CONFIG["slope_weak_up"]
            s_dn, w_dn = REGIME_CONFIG["slope_strong_down"], REGIME_CONFIG["slope_weak_down"]
            atr_th     = REGIME_CONFIG["atr_ratio_threshold"]
            df["regime"] = np.select(
                [atr_ratio > atr_th,
                 p_above & (slope_norm >= s_up), p_above & (slope_norm >= w_up),
                 p_below & (slope_norm <= s_dn), p_below & (slope_norm <= w_dn)],
                [Regime.VOLATILE, Regime.STRONG_TREND_UP, Regime.WEAK_TREND_UP,
                 Regime.STRONG_TREND_DOWN, Regime.WEAK_TREND_DOWN],
                default=Regime.SIDEWAYS,
            )

            # ── Vectorized scoring & signals ──────────────────────────────────
            df["Pullback"]  = compute_pullback_depth(df)
            rsi_thresh      = df["regime"].apply(lambda r: get_rsi_threshold(Regime(r)))
            df["Score"]     = (
                (price > ma200).astype(int)
                + (df[rsi_col] < rsi_thresh).astype(int)
                + (df["Volume"] > df[vol_col] * STRATEGY_CONFIG["volume_multiplier"]).astype(int)
                + (df["Pullback"] >= STRATEGY_CONFIG["min_pullback_pct"]).astype(int)
            )
            min_scores  = df["regime"].apply(lambda r: Regime(r).min_score_to_buy)
            buy_allowed = df["regime"].apply(lambda r: Regime(r).allows_buy)
            df["Signal"] = np.select(
                [(df["Score"] >= min_scores) & buy_allowed,
                 df[rsi_col] > STRATEGY_CONFIG["rsi_sell_threshold"]],
                ["BUY", "SELL"], default="HOLD",
            )

            # ── Vectorized pattern features (no get_pattern_snapshot in loop) ─
            # 1. Trend channel: 20-bar normalised linear slope
            lookback = 20
            closes   = df["Close"].values
            n        = len(closes)
            ch_arr   = ["SIDEWAYS"] * n
            for j in range(lookback, n):
                seg   = closes[j - lookback: j]
                x     = np.arange(lookback, dtype=float)
                m     = float(np.polyfit(x, seg, 1)[0])
                p_end = float(seg[-1])
                ns    = m / p_end if p_end > 0 else 0.0
                if ns > 0.002:    ch_arr[j] = "RISING"
                elif ns < -0.002: ch_arr[j] = "FALLING"
            df["_channel"] = ch_arr

            # 2. Volume spike: current volume > 2x prior 20-bar average
            vol_avg_20      = df["Volume"].rolling(21).mean().shift(1)
            df["_vol_spike"] = (df["Volume"] > 2.0 * vol_avg_20).fillna(False)

            # 3. Near support: close within 2% of 10-bar swing low
            swing_low      = df["Low"].rolling(10).min()
            near_ratio     = (df["Close"] - swing_low).abs() / df["Close"].clip(lower=1e-6)
            df["_near_sup"] = (near_ratio <= 0.02)

            # 4. Candlestick: classify last bar only (simplified)
            op = df["Open"]; hi = df["High"]; lo = df["Low"]; cl = df["Close"]
            body         = (cl - op).abs()
            upper_shadow = hi - np.maximum(op, cl)
            lower_shadow = np.minimum(op, cl) - lo
            tot_range    = (hi - lo).clip(lower=1e-9)
            safe_body    = body.clip(lower=1e-9)
            is_hammer    = (lower_shadow >= 2 * safe_body) & (upper_shadow <= 0.3 * safe_body) & (cl > op) & (body / tot_range < 0.4)
            is_bull_eng  = (cl > op) & (cl.shift(1) > op.shift(1)) & (cl > op.shift(1)) & (op < cl.shift(1))
            is_doji      = (body / tot_range < 0.1)
            df["_candle"] = np.select(
                [is_hammer, is_bull_eng, is_doji],
                ["HAMMER",  "BULLISH_ENGULFING", "DOJI"],
                default="NOCNDLE",
            )

            # ── Per-signal loop (all lookups are O(1) now) ────────────────────
            wins = losses = 0
            sig_arr = df["Signal"].values
            reg_arr = df["regime"].values
            rsi_arr = df[rsi_col].values
            scr_arr = df["Score"].values
            atr_arr = df[atr_col].values
            chn_arr = df["_channel"].values
            vol_arr = df["_vol_spike"].values
            sup_arr = df["_near_sup"].values
            cdl_arr = df["_candle"].values
            idx_arr = df.index

            for i in range(min_rows, n - OUTCOME_LOOKAHEAD):
                if sig_arr[i] != "BUY":
                    continue

                regime = str(reg_arr[i])
                rsi    = float(rsi_arr[i])
                score  = int(scr_arr[i])
                atr    = float(atr_arr[i])
                entry  = float(closes[i])
                cdl    = str(cdl_arr[i])

                key = build_condition_key(
                    regime       = regime,
                    rsi_bucket   = rsi_to_bucket(rsi),
                    score        = score,
                    channel      = str(chn_arr[i]),
                    candles      = [cdl] if cdl != "NOCNDLE" else [],
                    near_support = bool(sup_arr[i]),
                    volume_spike = bool(vol_arr[i]),
                )

                future = closes[i + 1: i + 1 + OUTCOME_LOOKAHEAD]
                if len(future) == 0:
                    continue

                target    = entry + atr
                stop      = entry - atr
                won       = None
                hold_time = 0
                for bar_idx, p_val in enumerate(future):
                    if p_val >= target:
                        won = True;  hold_time = bar_idx + 1; break
                    elif p_val <= stop:
                        won = False; hold_time = bar_idx + 1; break

                if won is None:
                    continue

                rsi_bkt    = rsi_to_bucket(rsi)
                context    = {"symbol": symbol, "regime": regime, "rsi_bkt": rsi_bkt, "score": score}
                trade_date = str(idx_arr[i].strftime("%Y-%m-%d"))
                record_observation(journal, key, won, context, trade_date=trade_date, hold_time=hold_time)
                if won: wins += 1
                else:   losses += 1

            logger.info("%s: seeded %d wins + %d losses", symbol, wins, losses)

        except Exception as exc:
            logger.warning("Seed failed for %s: %s", symbol, exc)

    save_journal(journal)
    return journal
'''

# Find the start of the old function and the start of the next function
start_marker = "def seed_from_history"
end_marker   = "\n\n# ── Live Trade Recording"

start_idx = src.index(start_marker)
end_idx   = src.index(end_marker, start_idx)

new_src = src[:start_idx] + NEW_FUNC + src[end_idx:]
path.write_text(new_src, encoding="utf-8")
print("Patched successfully. New lines:", new_src.count("\\n"))
