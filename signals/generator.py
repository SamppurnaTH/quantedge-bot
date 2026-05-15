"""
Signal Generator
Full pipeline: data → indicators → regime → strategy → risk → ranking

Regime flows through every layer:
  - detect_regime_detailed() gives slope + atr_ratio for logging
  - regime.min_score_to_buy controls strategy entry threshold
  - regime.atr_sl_multiplier controls stop-loss width in risk manager
  - regime.position_size_multiplier scales shares
  - Every decision is logged with full context
"""

import logging
from typing import List, Optional

import yfinance as yf

from data.fetcher import fetch_stock_data
from indicators.engine import compute_all_indicators
from indicators.regime import (
    Regime, RegimeResult,
    detect_regime, detect_regime_detailed, detect_regime_for_index,
    regime_summary, regime_allows_trade, _more_conservative,
)
from strategies.rsi_ma_strategy import (
    generate_signals, get_latest_signal, log_signal_decision,
    rank_signals, get_rsi_threshold,
)
from risk.manager import RiskManager
from config.settings import DATA_CONFIG, PORTFOLIO_CONFIG, STRATEGY_CONFIG
from analytics.dashboard import load_dynamic_filter

logger = logging.getLogger(__name__)


# ── Index regime detection ────────────────────────────────────────────────────

def fetch_index_regime() -> tuple:
    """
    Fetch NIFTY data, compute indicators, detect regime with full detail.

    Returns:
        (market_up: bool, regime: Regime, regime_result: RegimeResult)
    """
    index = DATA_CONFIG["market_index"]
    try:
        ticker = yf.Ticker(index)
        df     = ticker.history(period=DATA_CONFIG["period"], interval="1d", auto_adjust=True)
        if df.empty or len(df) < 210:
            logger.warning("Insufficient index data — defaulting to STRONG_TREND_UP")
            dummy = RegimeResult(Regime.STRONG_TREND_UP, 0.0, 1.0, 0.0, 0.0, 0.0)
            return True, Regime.STRONG_TREND_UP, dummy

        df = df[["Open", "High", "Low", "Close", "Volume"]].copy()
        df.index = df.index.tz_localize(None) if df.index.tz else df.index
        df = compute_all_indicators(df)

        result    = detect_regime_detailed(df)
        regime    = result.regime
        market_up = regime.allows_buy

        logger.info(
            "INDEX REGIME | %s | slope=%+.5f atr_ratio=%.2f | allows_buy=%s",
            result, market_up, result.slope, result.atr_ratio,
        )
        return market_up, regime, result
    except Exception as exc:
        logger.warning("Could not detect index regime: %s — defaulting STRONG_TREND_UP", exc)
        dummy = RegimeResult(Regime.STRONG_TREND_UP, 0.0, 1.0, 0.0, 0.0, 0.0)
        return True, Regime.STRONG_TREND_UP, dummy


# ── Per-symbol pipeline ───────────────────────────────────────────────────────

def generate_signal_for_symbol(
    symbol: str,
    capital: float = None,
    active_trades: int = 0,
    market_up: bool = True,
    market_regime: Regime = Regime.STRONG_TREND_UP,
) -> dict:
    """
    Full pipeline for a single symbol.
    Regime flows through strategy (score threshold) and risk (SL width, size).
    """
    df = fetch_stock_data(symbol, save=True)
    df = compute_all_indicators(df)

    # Per-stock regime
    stock_result = detect_regime_detailed(df)
    stock_regime = stock_result.regime

    # Effective regime = more conservative of index vs stock
    effective_regime = _more_conservative(market_regime, stock_regime)    # Strategy: regime controls RSI threshold + min_score_to_buy
    df = generate_signals(df, market_up=market_up, regime=effective_regime)
    latest = get_latest_signal(df)

    # Structured signal decision log (now includes pullback + RSI threshold)
    log_signal_decision(
        symbol=symbol,
        score=latest.get("score", 0),
        regime=effective_regime,
        signal=latest["signal"],
        rsi=latest["rsi"],
        close=latest["close"],
        ma200=latest["ma_200"],
        pullback_pct=latest.get("pullback_pct", 0),
        rsi_threshold=get_rsi_threshold(effective_regime),
    )

    # Risk: regime controls SL multiplier and position size
    rm = RiskManager(
        capital=capital if capital else 100_000,
        active_trades=active_trades,
    )
    risk_report = rm.approve_trade(
        entry=latest["close"],
        signal=latest["signal"],
        atr=latest.get("atr"),
        regime=effective_regime,
    )

    rr_val   = risk_report.get("risk_reward", 0) if risk_report.get("approved") else 0
    rank_key = (latest.get("score", 0) * 10) + rr_val

    return {
        "symbol":          symbol,
        "date":            latest["date"],
        "close":           latest["close"],
        "rsi":             latest["rsi"],
        "rsi_threshold":   get_rsi_threshold(effective_regime),
        "ma_200":          latest["ma_200"],
        "atr":             latest.get("atr"),
        "volume":          latest.get("volume"),
        "volume_ma":       latest.get("volume_ma"),
        "vol_confirmed":   latest.get("vol_confirmed"),
        "pullback_pct":    latest.get("pullback_pct", 0),
        "score":           latest.get("score", 0),
        "signal":          latest["signal"],
        "trend":           latest["trend"],
        "market_up":       market_up,
        "market_regime":   str(market_regime),
        "stock_regime":    str(stock_regime),
        "stock_slope":     round(stock_result.slope, 5),
        "stock_atr_ratio": round(stock_result.atr_ratio, 2),
        "regime":          str(effective_regime),
        "regime_emoji":    effective_regime.emoji,
        "size_multiplier": effective_regime.position_size_multiplier,
        "min_score_req":   effective_regime.min_score_to_buy,
        "risk_report":     risk_report,
        "rank_key":        rank_key,
    }


def scan_watchlist(
    symbols: List[str] = None,
    capital: float = None,
    active_trades: int = 0,
    top_n: Optional[int] = None,
    send_telegram: bool = False,
    use_dynamic_filter: bool = True,
) -> List[dict]:
    """
    Scan all symbols. Detects market regime once, reuses for all symbols.
    Applies composite ranking and optional dynamic journal-based filter.
    """
    symbols = symbols or DATA_CONFIG["symbols"]
    market_up, market_regime, market_result = fetch_index_regime()

    # Load dynamic filter from journal (if available)
    dyn_filter = load_dynamic_filter() if use_dynamic_filter else {}
    skip_regimes = dyn_filter.get("skip_regimes", [])
    skip_symbols = dyn_filter.get("skip_symbols", [])
    skip_score   = dyn_filter.get("skip_score_below", 0)

    if skip_regimes or skip_symbols or skip_score:
        logger.info(
            "Dynamic filter active | skip_regimes=%s skip_symbols=%s skip_score<%d",
            skip_regimes, skip_symbols, skip_score,
        )

    results = []
    for symbol in symbols:
        try:
            result = generate_signal_for_symbol(
                symbol,
                capital=capital,
                active_trades=active_trades,
                market_up=market_up,
                market_regime=market_regime,
            )

            # Apply dynamic filter — demote to HOLD if journal says skip
            if result.get("signal") == "BUY":
                regime_str = result.get("regime", "")
                score      = result.get("score", 0)
                if regime_str in skip_regimes:
                    result = {**result, "signal": "HOLD", "_filtered": f"regime {regime_str} blocked by journal"}
                elif symbol in skip_symbols:
                    result = {**result, "signal": "HOLD", "_filtered": f"{symbol} blocked by journal"}
                elif score < skip_score:
                    result = {**result, "signal": "HOLD", "_filtered": f"score {score} < journal min {skip_score}"}

            results.append(result)
        except Exception as exc:
            logger.error("Failed to process %s: %s", symbol, exc)
            results.append({"symbol": symbol, "error": str(exc)})

    # Apply composite ranking
    results = rank_signals(results)

    if top_n is not None:
        buy_count, filtered = 0, []
        for r in results:
            if r.get("signal") == "BUY":
                if buy_count < top_n:
                    filtered.append(r)
                    buy_count += 1
                else:
                    filtered.append({**r, "signal": "HOLD", "_demoted": True})
            else:
                filtered.append(r)
        results = filtered

    # Telegram alerts
    if send_telegram:
        try:
            from notifications.telegram import TelegramNotifier
            notifier = TelegramNotifier()
            notifier.send_daily_summary(results, regime_str=str(market_regime))
            for r in results:
                if r.get("signal") in ("BUY", "SELL"):
                    notifier.send_signal_alert(r)
        except Exception as exc:
            logger.warning("Telegram notification failed: %s", exc)

    return results


# ── Report printer ────────────────────────────────────────────────────────────

def print_signal_report(results: List[dict], market_regime: Regime = None) -> None:
    """Pretty-print the full signal report with regime context."""
    print("\n" + "=" * 70)
    print("  TRADING SIGNAL REPORT")
    if market_regime is not None:
        print(f"  Market Regime : {regime_summary(market_regime)}")
    print("=" * 70)

    for r in results:
        if "error" in r:
            print(f"\n  {r['symbol']:20s}  ⚠  ERROR: {r['error']}")
            continue

        signal      = r.get("signal", "HOLD")
        demoted     = r.get("_demoted", False)
        sig_icon    = {"BUY": "🟢", "SELL": "🔴", "HOLD": "🟡"}.get(signal, "⚪")
        vol_icon    = "✅" if r.get("vol_confirmed") else "❌"
        score       = r.get("score", 0)
        min_req     = r.get("min_score_req", 2)
        score_bar   = "●" * score + "○" * (3 - score)
        size_mult   = r.get("size_multiplier", 1.0)
        regime_str  = r.get("regime", "?")
        regime_icon = r.get("regime_emoji", "")
        demote_note = "  (portfolio cap)" if demoted else ""

        # Explain why signal was blocked if HOLD despite score
        block_note = ""
        if signal == "HOLD" and score >= 2 and not demoted:
            if r.get("_filtered"):
                block_note = f"  ← journal filter: {r['_filtered']}"
            else:
                block_note = f"  ← blocked: need {min_req}/3 in {regime_str}"

        print(f"\n  Stock    : {r['symbol']}  [{regime_icon}{regime_str}]")
        print(f"  Date     : {r['date']}")
        print(f"  Signal   : {sig_icon} {signal}   Score: [{score_bar}] {score}/4  (need {min_req}){demote_note}{block_note}")
        print(f"  Close    : {r['close']:.2f}   ATR: {r.get('atr') or 'N/A'}")
        print(f"  RSI      : {r['rsi']:.2f} (threshold <{r.get('rsi_threshold','?')})   MA-200: {r['ma_200']:.2f}")
        print(f"  Pullback : {r.get('pullback_pct', 0):.1f}%  (need ≥{STRATEGY_CONFIG['min_pullback_pct']*100:.0f}%)")
        print(f"  Trend    : {r['trend']}   Vol: {vol_icon}   Size: ×{size_mult}")
        print(f"  Slope    : {r.get('stock_slope', 0):+.5f}   ATR Ratio: {r.get('stock_atr_ratio', 1.0):.2f}")

        rr = r.get("risk_report", {})
        if rr.get("approved") and not demoted:
            rr_flag = " ⚠ Low R:R" if rr.get("rr_warning") else ""
            print(f"  ── Risk ({rr['sl_method']} ×{rr.get('sl_multiplier', 2.0)}) ──────────────────────────")
            print(f"  Shares      : {rr['shares']}   Cost: {rr['trade_cost']:,.2f}")
            print(f"  Stop-Loss   : {rr['stop_loss']:.2f}   Take-Profit: {rr['take_profit']:.2f}")
            print(f"  Max Loss    : {rr['max_loss']:,.2f}   Max Gain: {rr['max_gain']:,.2f}")
            print(f"  R:R Ratio   : {rr['risk_reward']}{rr_flag}")
            print(f"  Capital Risk: {rr['capital_at_risk_pct']:.2f}%")
        elif not rr.get("approved"):
            print(f"  Risk     : ⛔ {rr.get('reason', 'N/A')}")

    print("\n" + "=" * 70)
    print("  ⚠  SIGNALS ARE FOR DECISION SUPPORT ONLY — NOT TRADE ORDERS")
    print("=" * 70 + "\n")
