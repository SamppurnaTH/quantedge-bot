"""Offline tests for the trading / learning core (no network)."""
import json
import math
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import pytest

from conftest import make_ohlcv


# ── Indicators ────────────────────────────────────────────────────────────────

def test_rsi_is_100_when_there_are_no_losses():
    from indicators.rsi import compute_rsi
    close = pd.Series(np.linspace(100, 200, 60))
    rsi = compute_rsi(close)
    assert rsi.iloc[-1] == pytest.approx(100.0)
    assert rsi.iloc[:13].isna().all()


def test_rsi_is_0_when_there_are_no_gains():
    from indicators.rsi import compute_rsi
    rsi = compute_rsi(pd.Series(np.linspace(200, 100, 60)))
    assert rsi.iloc[-1] == pytest.approx(0.0)


def test_drop_incomplete_bar():
    from data.fetcher import drop_incomplete_bar
    ist = timezone(timedelta(hours=5, minutes=30))
    today = datetime(2026, 11, 13, 11, 0, tzinfo=ist)
    df = make_ohlcv(5, start="2026-11-09")
    assert df.index[-1].date() == today.date()
    assert len(drop_incomplete_bar(df, now=today)) == 4                      # intraday → dropped
    assert len(drop_incomplete_bar(df, now=today.replace(hour=17))) == 5     # after close → kept


# ── Regime ────────────────────────────────────────────────────────────────────

def test_regime_series_matches_point_detector():
    from indicators.engine import compute_all_indicators
    from indicators.regime import compute_regime_series, detect_regime_detailed
    df = compute_all_indicators(make_ohlcv(800, seed=3))
    series = compute_regime_series(df)
    for end in (400, 550, 799):
        assert series.iloc[end] == str(detect_regime_detailed(df.iloc[: end + 1]).regime)


def test_more_conservative_is_order_independent():
    from indicators.regime import Regime, _more_conservative
    assert _more_conservative(Regime.SIDEWAYS, Regime.VOLATILE) == Regime.VOLATILE
    assert _more_conservative(Regime.VOLATILE, Regime.SIDEWAYS) == Regime.VOLATILE
    assert _more_conservative(Regime.STRONG_TREND_UP, Regime.WEAK_TREND_DOWN) == Regime.WEAK_TREND_DOWN


def test_hysteresis_advances_once_per_bar_date(workdir):
    from indicators.regime import Regime, apply_regime_hysteresis as h
    assert h(Regime.STRONG_TREND_UP, "2026-01-01") == Regime.STRONG_TREND_UP
    # same bar date seen by morning AND evening run must not count twice
    assert h(Regime.SIDEWAYS, "2026-01-02") == Regime.STRONG_TREND_UP
    assert h(Regime.SIDEWAYS, "2026-01-02") == Regime.STRONG_TREND_UP
    assert h(Regime.SIDEWAYS, "2026-01-05") == Regime.SIDEWAYS


# ── Strategy ──────────────────────────────────────────────────────────────────

def test_buy_requires_uptrend_and_oversold():
    from indicators.engine import compute_all_indicators
    from indicators.regime import Regime
    from strategies.rsi_ma_strategy import generate_signals
    df = compute_all_indicators(make_ohlcv(600, seed=1))
    out = generate_signals(df, market_up=True, regime=Regime.STRONG_TREND_UP)
    buys = out[out["Signal"] == "BUY"]
    assert ((buys["Close"] > buys["SMA_200"]) & (buys["RSI_14"] < 35)).all()


def test_signal_frame_matches_live_generator_on_last_bar():
    from indicators.engine import compute_all_indicators
    from indicators.regime import Regime
    from strategies.rsi_ma_strategy import build_signal_frame, generate_signals
    df = compute_all_indicators(make_ohlcv(700, seed=5))
    sf = build_signal_frame(df, None)
    for end in (450, 600, 699):
        sub = df.iloc[: end + 1]
        reg = Regime(sf["regime"].iloc[end])
        live = generate_signals(sub, market_up=True, regime=reg)
        assert live["Signal"].iloc[-1] == sf["Signal"].iloc[end]


# ── Trade model ───────────────────────────────────────────────────────────────

def test_trade_model_rules():
    from execution import trade_model as tm
    pos = {"stop_loss": 95.0, "take_profit": 110.0}
    assert tm.step(pos, 94, 96, 93, 95, 50, 2) == (94, "GAP-DOWN-SL")
    assert tm.step(pos, 100, 111, 94, 105, 50, 2) == (95.0, "STOP-LOSS")      # both inside bar → stop first
    assert tm.step(pos, 100, 111, 99, 105, 50, 2) == (110.0, "TAKE-PROFIT")
    assert tm.step(pos, 100, 104, 99, 103, 75, 2) is None                      # RSI exit waits for min hold
    assert tm.step(pos, 100, 104, 99, 103, 75, 3) == (103, "SIGNAL-SELL")
    assert tm.step(pos, 100, 104, 99, 103, 50, tm.MAX_HOLD) == (103, "TIME-STOP")


def test_plan_entry_skips_gap_chase_and_r_includes_costs():
    from execution import trade_model as tm
    assert tm.plan_entry(100, 100 + 0.6 * 4, atr=4, sl_mult=2) is None
    p = tm.plan_entry(100, 100.5, atr=4, sl_mult=2)
    assert p["initial_risk"] == pytest.approx(8.0)
    assert p["take_profit"] - p["entry_price"] == pytest.approx(16.0)
    r = tm.r_multiple(p["entry_price"], tm.exit_fill(p["take_profit"]), p["initial_risk"])
    assert 1.9 < r < 2.0                                                       # 2R minus costs


# ── Risk ──────────────────────────────────────────────────────────────────────

def test_risk_manager_uses_edge_and_never_forces_a_share():
    from indicators.regime import Regime
    from risk.manager import RiskManager
    rm = RiskManager(capital=100_000)
    neg = {"expected_r": -0.1, "lower_r": -0.3, "evidence": "archetype"}
    assert not rm.approve_trade(500, "BUY", atr=10, regime=Regime.STRONG_TREND_UP, edge=neg)["approved"]
    proven = {"expected_r": 0.3, "lower_r": 0.1, "evidence": "pattern"}
    full = rm.approve_trade(500, "BUY", atr=25, regime=Regime.STRONG_TREND_UP, edge=proven)
    half = rm.approve_trade(500, "BUY", atr=25, regime=Regime.SIDEWAYS, edge=proven)
    assert full["approved"] and full["capital_at_risk_pct"] == pytest.approx(1.0, abs=0.05)
    assert half["shares"] == full["shares"] // 2
    # risk budget too small for one share → rejected, not forced to 1
    tiny = RiskManager(capital=1_000)
    assert not tiny.approve_trade(5000, "BUY", atr=200, regime=Regime.SIDEWAYS, edge=proven)["approved"]


# ── Learning engine ───────────────────────────────────────────────────────────

def test_decayed_stats_and_states():
    from analytics.learning_engine import _empty_journal, record_observation, get_edge_estimate
    j = _empty_journal()
    key = "STRONG_TREND_UP|RSI<35|S3|SIDEWAYS|NOCNDLE|NOSUPP|NOSPK"
    for i in range(400):                      # 40% win rate, 2R winners, −1R losers → +0.2R edge
        r = 2.0 if i % 5 in (0, 2) else -1.0
        d = (datetime(2020, 1, 1) + timedelta(days=i * 3)).strftime("%Y-%m-%d")
        record_observation(j, key, r > 0, {}, trade_date=d, r=r)
    p = j["patterns"][key]
    assert p["state"] in ("PROVEN", "VALIDATED")      # profitable at a 40% win rate
    assert 0.35 < p["win_rate"] < 0.45
    est = get_edge_estimate(j, key)
    assert est["expected_r"] > 0 and est["evidence"] == "pattern"


def test_rupee_pnl_is_never_mixed_into_r_stats():
    from analytics.learning_engine import _empty_journal, record_observation
    j = _empty_journal()
    key = "SIDEWAYS|RSI<25|S3|SIDEWAYS|NOCNDLE|NOSUPP|NOSPK"
    record_observation(j, key, False, {}, pnl=-138.64, trade_date="2026-06-02", r=-1.05)
    assert j["patterns"][key]["expectancy"] == pytest.approx(-1.05)


def test_losing_pattern_becomes_unreliable():
    from analytics.learning_engine import _empty_journal, record_observation
    j = _empty_journal()
    key = "WEAK_TREND_UP|RSI<30|S3|FALLING|NOCNDLE|NOSUPP|NOSPK"
    for i in range(80):
        r = 2.0 if i % 5 == 0 else -1.0       # 20% win rate → −0.4R
        record_observation(j, key, r > 0, {}, trade_date=f"2024-{1 + i % 12:02d}-{1 + i % 28:02d}", r=r)
    assert j["patterns"][key]["state"] == "UNRELIABLE"


def test_seed_and_paper_trader_end_to_end(workdir, monkeypatch):
    import data.fetcher as fetcher
    from analytics import learning_engine as le
    from execution.paper_trader import PaperTrader

    frames = {"^NSEI": make_ohlcv(2500, seed=99, vol=0.009, drift=0.0006)}
    for k in range(12):
        frames[f"S{k}.NS"] = make_ohlcv(2500, seed=k + 10, drift=0.0007, vol=0.017)
    monkeypatch.setattr(fetcher, "load_stock_data", lambda s, **kw: frames[s].copy())
    monkeypatch.setattr(le, "load_stock_data", lambda s, **kw: frames[s].copy())

    j = le.seed_from_history(symbols=[s for s in frames if s != "^NSEI"], journal=le._empty_journal())
    assert j["metadata"]["seeded_trades"] >= 10
    g = j["global"]
    assert g["count"] == j["metadata"]["seeded_trades"]
    assert -1.5 < g["expectancy"] < 2.5                  # R units, not rupees
    assert json.load(open("state/learning_journal.json"))["metadata"]["version"] == 2

    # Paper trader: order on the second-to-last bar, filled at the last bar's open
    sym = "S0.NS"
    full = frames[sym]
    frames[sym] = full.iloc[:-1]
    trader = PaperTrader(capital=100_000, loader=lambda s: frames[s].copy())
    sig = {"symbol": sym, "signal": "BUY", "date": str(full.index[-2].date()), "close": float(full["Close"].iloc[-2]),
           "atr": 10.0, "regime": "STRONG_TREND_UP", "expectancy_r": 0.2,
           "risk_report": {"approved": True, "shares": 10, "atr_used": 10.0, "sl_multiplier": 2.0}}
    trader.process_signals([sig])
    assert sym in trader.pending_orders
    trader.update()
    assert sym in trader.pending_orders                    # next bar does not exist yet
    frames[sym] = full
    open_px = float(full["Open"].iloc[-1])
    trader.update()
    if sym in trader.positions:
        assert trader.positions[sym]["entry_price"] == pytest.approx(open_px * 1.001)
    else:
        assert trader.closed_trades or trader.cancelled_orders
    assert trader.equity_curve
