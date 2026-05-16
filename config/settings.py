"""
Central configuration — all parameters in one place, no magic numbers in code.
"""

# ── Data ──────────────────────────────────────────────────────────────────────
DATA_CONFIG = {
    "symbols": [
        "ADANIENT.NS", "ADANIPORTS.NS", "APOLLOHOSP.NS", "ASIANPAINT.NS", "AXISBANK.NS", 
        "BAJAJ-AUTO.NS", "BAJAJFINSV.NS", "BAJFINANCE.NS", "BHARTIARTL.NS", "BPCL.NS", 
        "BRITANNIA.NS", "CIPLA.NS", "COALINDIA.NS", "DIVISLAB.NS", "DRREDDY.NS", 
        "EICHERMOT.NS", "GRASIM.NS", "HCLTECH.NS", "HDFCBANK.NS", "HDFCLIFE.NS", 
        "HEROMOTOCO.NS", "HINDALCO.NS", "HINDUNILVR.NS", "ICICIBANK.NS", "INDUSINDBK.NS", 
        "INFY.NS", "ITC.NS", "JSWSTEEL.NS", "KOTAKBANK.NS", "LT.NS", 
        "LTIM.NS", "M&M.NS", "MARUTI.NS", "NESTLEIND.NS", "NTPC.NS", 
        "ONGC.NS", "POWERGRID.NS", "RELIANCE.NS", "SBILIFE.NS", "SBIN.NS", 
        "SUNPHARMA.NS", "TATACONSUM.NS", "TATAMOTORS.NS", "TATASTEEL.NS", "TCS.NS", 
        "TECHM.NS", "TITAN.NS", "ULTRACEMCO.NS", "UPL.NS", "WIPRO.NS"
    ],
    "market_index": "^NSEI",
    "period": "10y",      # extended from 3y — more data = better strategy calibration
    "interval": "1d",
    "data_dir": "data/raw",
}

# ── Indicators ────────────────────────────────────────────────────────────────
INDICATOR_CONFIG = {
    "rsi_period": 14,
    "ma_fast": 50,
    "ma_slow": 200,
    "atr_period": 14,
    "volume_ma_period": 20,
    "pullback_window": 20,          # bars to look back for recent high (pullback depth)
}

# ── Strategy ──────────────────────────────────────────────────────────────────
STRATEGY_CONFIG = {
    # RSI thresholds — regime-aware (tighter in weaker regimes)
    "rsi_thresholds": {
        "STRONG_TREND_UP":   35,    # strong trend: enter on moderate pullback
        "WEAK_TREND_UP":     30,    # weak trend: only deep pullbacks
        "SIDEWAYS":          25,    # sideways: very oversold only
        "VOLATILE":          25,    # volatile: very oversold only
        "WEAK_TREND_DOWN":   None,  # no longs
        "STRONG_TREND_DOWN": None,  # no longs
    },
    "rsi_sell_threshold": 70,
    "trend_ma": "SMA_200",
    "volume_multiplier": 1.0,
    "market_trend_ma": 50,
    "min_score_to_buy": 2,
    # Pullback depth filter
    "min_pullback_pct": 0.03,       # price must be ≥ 3% below recent high to qualify
}

# ── Risk Management ───────────────────────────────────────────────────────────
RISK_CONFIG = {
    "max_risk_per_trade_pct": 1.0,
    "atr_sl_multiplier": 2.0,
    "atr_tp_multiplier": 4.0,
    "fallback_stop_loss_pct": 3.0,
    "fallback_take_profit_pct": 6.0,
    "max_position_pct": 10.0,
    "default_capital": 100_000,
}

# ── Portfolio ─────────────────────────────────────────────────────────────────
PORTFOLIO_CONFIG = {
    "max_active_trades": 5,
    "max_sector_exposure_pct": 30.0,
    "capital_per_trade_pct": 20.0,
    "top_n_signals": 3,
}

# ── Regime Detection ──────────────────────────────────────────────────────────
REGIME_CONFIG = {
    "slope_window": 5,
    "atr_ratio_threshold": 1.4,
    "slope_strong_up":    0.0015,       # data-calibrated: actual slope range ≈ ±0.004
    "slope_weak_up":      0.0005,
    "slope_strong_down": -0.0015,
    "slope_weak_down":   -0.0005,
}

# ── Trade Execution Controls ──────────────────────────────────────────────────
EXECUTION_CONFIG = {
    "slippage_pct": 0.001,
    "gap_threshold": 0.005,
    "cooldown_days": 5,
    "min_hold_bars": 3,
}

# ── Telegram Alerts ───────────────────────────────────────────────────────────
TELEGRAM_CONFIG = {
    "enabled": True,                # set False to disable without removing code
    "parse_mode": "HTML",           # HTML formatting in messages
}

# ── Backtesting ───────────────────────────────────────────────────────────────
BACKTEST_CONFIG = {
    "initial_cash": 100_000,
    "commission": 0.001,
    "stake": 10,
}

# ── Logging ───────────────────────────────────────────────────────────────────
LOG_CONFIG = {
    "log_dir": "logs",
    "log_file": "trading_bot.log",
    "level": "INFO",
}
