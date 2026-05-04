# QuantEdge Bot

Production-grade stock market decision-support system for Indian equities.

> **Not a trade executor.** Generates signals, manages risk, and tracks performance. All trading decisions remain with the human.

---

## What it does

```
Market → Regime Detection → Entry Filter → Risk Sizing → Signal → Paper Trade → Telegram Alert
```

- **6-tier regime detection** — STRONG_TREND_UP → STRONG_TREND_DOWN, using MA slope + ATR ratio
- **Regime-aware RSI thresholds** — RSI < 35 in strong trends, < 25 in sideways
- **4-point signal scoring** — trend structure + RSI + volume + pullback depth
- **ATR-based risk management** — dynamic SL/TP that adapts to volatility
- **Paper trading** — slippage simulation, gap handling, persistent state
- **A/B testing framework** — compare strategy versions on 3 years of data
- **Filter analysis** — identifies which trades the regime filter removes and whether they were profitable
- **Telegram alerts** — daily signals sent to your phone

---

## Setup

```bash
pip install pandas numpy yfinance backtrader requests
```

Copy `.env.example` to `.env` and fill in your Telegram credentials:

```
TELEGRAM_BOT_TOKEN=your_token_here
TELEGRAM_CHAT_ID=your_chat_id_here
```

---

## Usage

```bash
# Daily signal scan
python main.py

# Daily scan + Telegram alerts
python main.py --telegram

# Automated daily runner (run at 9:00 AM IST)
python run_daily.py

# Paper trading
python main.py --paper
python main.py --paper --reset     # fresh start

# Backtesting
python main.py --backtest
python main.py --symbol TCS.NS --backtest

# Analysis
python main.py --regime            # current market regime
python main.py --dashboard         # full performance dashboard
python main.py --ab-test           # compare 5 strategy versions
python main.py --filter-analysis   # which trades did the filter remove?
python main.py --journal           # trade journal analytics
python main.py --compare           # regime filter: with vs without
```

---

## Project structure

```
trading_bot/
├── config/settings.py          ← all parameters in one place
├── data/fetcher.py             ← yfinance + CSV cache
├── indicators/
│   ├── rsi.py                  ← Wilder's RSI
│   ├── moving_averages.py      ← SMA / EMA
│   ├── atr.py                  ← Average True Range
│   ├── volume.py               ← volume confirmation
│   ├── regime.py               ← 6-tier regime detection
│   └── engine.py               ← attaches all indicators to DataFrame
├── strategies/
│   └── rsi_ma_strategy.py      ← regime-aware RSI + pullback + scoring
├── risk/manager.py             ← ATR-based SL/TP, position sizing
├── signals/generator.py        ← full pipeline + ranking + dynamic filter
├── backtests/
│   ├── bt_strategy.py          ← Backtrader baseline strategy
│   ├── regime_bt_strategy.py   ← regime-aware Backtrader strategy
│   ├── runner.py               ← single + multi-symbol backtest
│   └── regime_comparison.py    ← 3-way comparison table
├── execution/paper_trader.py   ← paper trading with slippage + gap handling
├── analytics/
│   ├── journal.py              ← trade journal analytics
│   ├── dashboard.py            ← performance dashboard + equity curve
│   ├── ab_test.py              ← A/B testing framework
│   └── filter_analysis.py      ← regime filter diagnostic
├── notifications/telegram.py   ← Telegram alerts
├── logger/
│   ├── setup.py                ← rotating file + console logging
│   ├── trade_tracker.py        ← CSV signal + backtest history
│   └── env_loader.py           ← .env loader
├── run_daily.py                ← daily automation script
└── main.py                     ← CLI entry point
```

---

## Watchlist (default)

`RELIANCE.NS`, `TCS.NS`, `INFY.NS`, `HDFCBANK.NS`, `ADANIPOWER.NS`

Edit `DATA_CONFIG["symbols"]` in `config/settings.py` to change it.

---

## Key design decisions

**Why MA200 for regime, not MA50?**
RSI pullback entries happen when price dips below MA50 — that's what creates the oversold reading. Requiring `price > MA50` as a filter contradicts the entry signal. MA200 captures long-term trend direction; MA50 slope captures momentum.

**Why V1_BASELINE still wins on backtests?**
TCS's two largest wins (115-day and 41-day holds) were counter-trend entries — price was below MA200. The regime filter correctly blocks those as high-risk. Whether that's right depends on your risk tolerance.

**Why paper trade before live?**
The A/B test uses 3 years of historical data. Real-time signals will encounter market conditions not in that window. Paper trading for 8–10 weeks builds a live track record before risking capital.

---

## Disclaimer

This system is for research and decision support only. It does not execute trades. Past backtest performance does not guarantee future results.
