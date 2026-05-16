import streamlit as st
import pandas as pd
import json
import os
import io
import sys
from datetime import datetime
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import yfinance as yf
from config.settings import DATA_CONFIG, RISK_CONFIG
from config.tickers import NIFTY50_NAMES

# ── PAGE CONFIG ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="QuantEdge | Intelligence Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── STYLING ──────────────────────────────────────────────────────────────────
st.markdown("""
    <style>
    .main { background-color: #0e1117; }
    .stMetric { background-color: #1e2130; padding: 15px; border-radius: 10px; border: 1px solid #3e4150; }
    .stPlotlyChart { background-color: #1e2130; border-radius: 10px; padding: 10px; }
    </style>
    """, unsafe_allow_html=True)

# ── DATA LOADING ──────────────────────────────────────────────────────────────
PAPER_PORTFOLIO_FILE = os.path.join("logs", "paper_portfolio.json")
LEARNING_JOURNAL_FILE = os.path.join("state", "learning_journal.json")

def load_json(filepath):
    if not os.path.exists(filepath): return {}
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
            return data if data is not None else {}
    except (json.JSONDecodeError, EOFError):
        return {}

from analytics.learning_report import _parse_key

def get_market_status():
    """Checks NSE Market Hours (9:15 AM - 3:30 PM IST)."""
    now = datetime.now()
    # NSE is Monday (0) to Friday (4)
    is_weekday = now.weekday() < 5
    market_open = now.replace(hour=9, minute=15, second=0, microsecond=0)
    market_close = now.replace(hour=15, minute=30, second=0, microsecond=0)
    
    if not is_weekday:
        return "🔴 MARKET CLOSED (Weekend)", "#ff4b4b"
    if market_open <= now <= market_close:
        return "🟢 MARKET OPEN (NSE)", "#00cc96"
    else:
        return "🔴 MARKET CLOSED", "#ff4b4b"

# ── SIDEBAR ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("🛡️ QuantEdge")
    status_text, status_color = get_market_status()
    st.markdown(f"<p style='color:{status_color}; font-weight:bold; font-size:1.1em;'>{status_text}</p>", unsafe_allow_html=True)
    st.caption(f"Current Time: {datetime.now().strftime('%H:%M:%S IST')}")
    st.divider()
    page = st.radio("Navigation", ["📊 Live Dashboard", "🔍 Stock Explorer", "📘 Intelligence Report", "🏫 Education Center"])
    st.divider()
    portfolio = load_json(PAPER_PORTFOLIO_FILE)
    journal = load_json(LEARNING_JOURNAL_FILE)

    # ── GLOBAL DATA PROCESSING ──────────────────────────────────────────────────
    closed_trades = []
    positions = {}
    if journal is None: journal = {}
    if portfolio:
        closed_trades = portfolio.get("closed_trades", [])
        positions = portfolio.get("positions", {})

    if portfolio: st.success("✅ Portfolio Connected")
    else: st.error("❌ Portfolio Data Not Found")
    if journal: st.success("✅ Knowledge Base Connected")
    else: st.warning("⚠️ Journal Not Found")
    st.divider()
    st.info("Current Mode: **Paper Trading**")
    if st.button("🔄 Refresh Data"): st.rerun()
    
    # Excel Export Helper
    if closed_trades or positions:
        st.divider()
        st.subheader("📥 Data Export")
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            if closed_trades:
                pd.DataFrame(closed_trades).to_excel(writer, index=False, sheet_name='Trade_Journal')
            if positions:
                pd.DataFrame(positions).transpose().to_excel(writer, index=True, sheet_name='Open_Positions')
            if journal and "patterns" in journal:
                pd.DataFrame(journal["patterns"]).transpose().to_excel(writer, index=True, sheet_name='Bot_Knowledge')
        
        st.download_button(
            label="📊 Download Excel Report",
            data=output.getvalue(),
            file_name=f"QuantEdge_Report_{datetime.now().strftime('%Y%m%d')}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )

# ── PAGE LOGIC ───────────────────────────────────────────────────────────────

if page == "📊 Live Dashboard":
    st.title("📈 Performance Intelligence Dashboard")
    st.caption("Monitoring self-learning trading patterns and paper portfolio health.")

    if not portfolio:
        st.info("No data available yet. Run `python run_daily.py` to start generating insights.")
        st.stop()

    # ── BOT PULSE ──────────────────────────────────────────────────────────────
    st.divider()
    cp1, cp2 = st.columns([2, 1])
    
    with cp1:
        st.subheader("⚡ Live Bot Activity")
        log_path = os.path.join("logs", "trading_bot.log")
        if os.path.exists(log_path):
            with open(log_path, "r", encoding="utf-8") as f:
                lines = f.readlines()[-10:]
                log_text = "".join(lines)
                st.code(log_text, language="text")
        else:
            st.info("No activity logs found yet.")

    with cp2:
        st.subheader("🏁 Current Focus")
        st.write(f"**Target Symbols:** {len(portfolio.get('symbols', DATA_CONFIG['symbols']))}")
        st.write(f"**Last Scan:** {datetime.now().strftime('%Y-%m-%d %H:%M')}")
        
        # Human-friendly Maturity
        p_count = len(journal.get('patterns', {})) if journal else 0
        if p_count < 50:
            stage, desc = "Infant 👶", "I'm still learning the basics. Use extreme caution."
        elif p_count < 200:
            stage, desc = "Adolescent 👦", "I've seen many patterns and am becoming more reliable."
        else:
            stage, desc = "Adult 🧔", "I have a deep knowledge base and high optimization."
            
        st.write(f"**Maturity Stage:** {stage}")
        st.caption(f"_{desc}_")

    initial_cap = portfolio.get("initial_capital", RISK_CONFIG["default_capital"])
    cash = portfolio.get("cash", RISK_CONFIG["default_capital"])
    
    open_value = sum(p["shares"] * p["current_price"] for p in positions.values())
    total_value = cash + open_value
    total_pnl = total_value - initial_cap
    total_pnl_pct = (total_pnl / initial_cap) * 100

    wins = [t for t in closed_trades if t["pnl"] > 0]
    losses = [t for t in closed_trades if t["pnl"] <= 0]
    win_rate = (len(wins) / len(closed_trades) * 100) if closed_trades else 0.0

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Total Portfolio Value", f"₹{total_value:,.2f}", f"{total_pnl_pct:+.2f}%")
    m2.metric("Available Cash", f"₹{cash:,.2f}")
    m3.metric("Win Rate", f"{win_rate:.1f}%", f"{len(closed_trades)} Trades")
    m4.metric("Active Positions", len(positions))

    st.divider()
    st.subheader("📊 Equity Curve & Growth")
    if closed_trades:
        df_trades = pd.DataFrame(closed_trades)
        df_trades['exit_date'] = pd.to_datetime(df_trades['exit_date'])
        df_trades = df_trades.sort_values('exit_date')
        df_trades['cum_pnl'] = df_trades['pnl'].cumsum()
        df_trades['equity'] = initial_cap + df_trades['cum_pnl']
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=[df_trades['exit_date'].min(), df_trades['exit_date'].max()], y=[initial_cap, initial_cap], mode='lines', name='Starting Capital', line=dict(color='gray', dash='dash')))
        fig.add_trace(go.Scatter(x=df_trades['exit_date'], y=df_trades['equity'], mode='lines+markers', name='Portfolio Equity', line=dict(color='#00ffcc', width=3), fill='tozeroy', fillcolor='rgba(0, 255, 204, 0.1)'))
        fig.update_layout(template="plotly_dark", margin=dict(l=20, r=20, t=20, b=20), height=400, hovermode="x unified", xaxis_title="Date", yaxis_title="Equity (₹)")
        st.plotly_chart(fig, width='stretch')
    else:
        st.info("Equity curve will appear after the first closed trade.")

    c1, c2 = st.columns([1, 1])
    with c1:
        st.subheader("🧠 Intelligence Hub")
        if journal and journal.get("patterns"):
            p_df = []
            counts = {"LEARNED": 0, "LEARNING": 0, "WATCHING": 0}
            for key, data in journal["patterns"].items():
                state = data["state"]
                counts[state] += 1
                if data["trades"] > 0:
                    p_df.append({"Description": _parse_key(key), "Trades": data["trades"], "Win Rate": data["win_rate"] * 100, "State": state})
            fig_dist = px.pie(values=list(counts.values()), names=list(counts.keys()), color=list(counts.keys()), color_discrete_map={'LEARNED': '#00cc96', 'LEARNING': '#ffa500', 'WATCHING': '#ff4b4b'}, hole=0.4, title="Knowledge Maturity")
            fig_dist.update_layout(template="plotly_dark", height=300, margin=dict(l=20, r=20, t=40, b=20))
            st.plotly_chart(fig_dist, width='stretch')
            if p_df:
                df_p = pd.DataFrame(p_df).sort_values("Win Rate", ascending=False)
                st.dataframe(df_p, width='stretch', hide_index=True, column_config={"Win Rate": st.column_config.NumberColumn(format="%.1f%%"), "Trades": st.column_config.NumberColumn(format="%d 📈"), "State": st.column_config.TextColumn(help="Knowledge state of this pattern")})
        else:
            st.info("Intelligence Hub will populate as the bot learns from history.")
    with c2:
        st.subheader("📉 Regime Performance")
        if closed_trades:
            df_regime = pd.DataFrame(closed_trades)
            regime_stats = df_regime.groupby("regime")["pnl"].sum().reset_index()
            fig_regime = px.bar(regime_stats, x="regime", y="pnl", color="pnl", color_continuous_scale=["#ff4b4b", "#00cc96"], title="PnL by Market Regime")
            fig_regime.update_layout(template="plotly_dark", margin=dict(l=20, r=20, t=40, b=20))
            st.plotly_chart(fig_regime, width='stretch')
        else:
            st.info("Regime diagnostics will appear after trades are closed.")

    # ── STRATEGY FORECASTER ────────────────────────────────────────────────────
    st.divider()
    st.subheader("🔮 Probabilistic Strategy Forecaster")
    st.caption("Using historical outcomes to predict future probability of success.")
    
    if journal and journal.get("patterns"):
        # Pre-fetch symbols for price lookup
        unique_syms = list(set([data.get("context", {}).get("symbol") for data in journal["patterns"].values() if data.get("context")]))
        with st.spinner("Fetching live prices..."):
            prices = {}
            if unique_syms:
                latest_data = yf.download(unique_syms, period="1d", interval="1m", progress=False)
                if not latest_data.empty:
                    # Flatten if multi-index
                    if isinstance(latest_data.columns, pd.MultiIndex):
                        price_df = latest_data['Close']
                        # Handle multi-stock or single-stock returns
                        if isinstance(price_df, pd.Series):
                            prices = {unique_syms[0]: price_df.iloc[-1]}
                        else:
                            prices = {s: price_df[s].iloc[-1] for s in price_df.columns}
                    else:
                        prices = {unique_syms[0]: latest_data['Close'].iloc[-1]}

        p_data = []
        for key, data in journal["patterns"].items():
            if data["trades"] >= 2:
                win_rate = data["win_rate"]
                loss_rate = 1 - win_rate
                ev = (win_rate * 2.0) - (loss_rate * 1.0)
                
                # Dynamic Risk Narrative
                risk_why = "Unknown"
                if "RSI>70" in key: risk_why = "Extreme Overbought (High reversal risk)"
                elif "RSI<30" in key and "DOWN" in key: risk_why = "Catching a falling knife in a bear market"
                elif "VOLATILE" in key: risk_why = "Wild swings making stops easy to hit"
                elif "SIDEWAYS" in key: risk_why = "Stuck in a range with no clear breakout"
                else: risk_why = "General market noise and false signals"

                primary_sym = data.get("context", {}).get("symbol", "All Nifty 50")
                curr_price = prices.get(primary_sym, 0)
                company_name = NIFTY50_NAMES.get(primary_sym, primary_sym)

                p_data.append({
                    "Target Company": company_name,
                    "Price": curr_price,
                    "Pattern Story": _parse_key(key),
                    "Win Prob.": f"{win_rate*100:.0f}%",
                    "Loss Prob.": f"{loss_rate*100:.0f}%",
                    "Expectancy Score": round(ev, 2),
                    "Risk Analysis": risk_why,
                    "Obs.": data["trades"]
                })
        
        if p_data:
            df_ev = pd.DataFrame(p_data).sort_values("Expectancy Score", ascending=False)
            st.dataframe(df_ev, hide_index=True, width='stretch', column_config={
                "Price": st.column_config.NumberColumn("Current Price", format="₹%.2f"),
                "Win Prob.": st.column_config.TextColumn(help="Probability of Profit"),
                "Loss Prob.": st.column_config.TextColumn(help="Probability of Loss"),
                "Risk Analysis": st.column_config.TextColumn(help="Why this pattern might fail")
            })
            st.info("💡 **Risk Insight:** Patterns with a high **Loss Prob.** and a negative **Expectancy Score** are your 'Danger Zones'. The bot uses this data to skip these trades automatically!")
        else:
            st.info("Analyzing market history to build probability maps...")
    else:
        st.info("Forecasting will begin after the bot processes more data.")

    # ── PERFORMANCE LEADERBOARD ────────────────────────────────────────────────
    st.divider()
    st.subheader("🏆 Performance Leaderboard (Top Winners & Losers)")
    if closed_trades:
        df_lead = pd.DataFrame(closed_trades)
        leaderboard = df_lead.groupby("symbol").agg({
            "pnl": ["sum", "count", "mean"],
            "regime": "first"
        }).reset_index()
        leaderboard.columns = ["Ticker", "Total PnL", "Trades", "Avg PnL", "Primary Regime"]
        leaderboard["Company"] = leaderboard["Ticker"].map(NIFTY50_NAMES).fillna(leaderboard["Ticker"])
        
        # Sort by PnL
        leaderboard = leaderboard.sort_values("Total PnL", ascending=False)
        
        l_col1, l_col2 = st.columns(2)
        with l_col1:
            st.markdown("### 🟢 Top Profitable Stocks")
            st.dataframe(leaderboard.head(5), hide_index=True, width='stretch', column_config={"Total PnL": st.column_config.NumberColumn(format="₹%.2f"), "Avg PnL": st.column_config.NumberColumn(format="₹%.2f")})
        with l_col2:
            st.markdown("### 🔴 Top Loss-Making Stocks")
            st.dataframe(leaderboard.tail(5).sort_values("Total PnL"), hide_index=True, width='stretch', column_config={"Total PnL": st.column_config.NumberColumn(format="₹%.2f"), "Avg PnL": st.column_config.NumberColumn(format="₹%.2f")})
    else:
        st.info("Leaderboard will populate as stocks complete their trade cycles.")

    st.divider()
    st.subheader("📓 Trade Journal")
    if closed_trades:
        df_all = pd.DataFrame(closed_trades)
        df_all['Company'] = df_all['symbol'].map(NIFTY50_NAMES).fillna(df_all['symbol'])
        display_cols = ['Company', 'symbol', 'exit_date', 'pnl', 'exit_reason', 'regime']
        st.dataframe(df_all[display_cols].sort_values('exit_date', ascending=False), width='stretch', hide_index=True, column_config={
            "pnl": st.column_config.NumberColumn("PnL (₹)", format="₹%.2f"),
            "exit_date": "Date",
            "symbol": "Ticker",
            "exit_reason": "Reason",
            "regime": "Regime"
        })
    else:
        st.info("No closed trades in history yet.")

    # ── OPEN POSITIONS ───────────────────────────────────────────────────────────
    st.divider()
    st.subheader("🔓 Active Positions")
    if positions:
        ap1, ap2 = st.columns([1, 2])
        p_list = []
        for sym, p in positions.items():
            unreal = (p["current_price"] - p["entry_price"]) * p["shares"]
            current_val = p["current_price"] * p["shares"]
            company_name = NIFTY50_NAMES.get(sym, sym)
            p_list.append({
                "Company": company_name,
                "Symbol": sym,
                "Value": current_val,
                "Entry": p["entry_price"],
                "Current": p["current_price"],
                "Unreal P&L": unreal,
                "Regime": p["regime"]
            })
        
        df_pos = pd.DataFrame(p_list)
        
        with ap1:
            fig_alloc = px.pie(df_pos, values='Value', names='Symbol', title="Portfolio Allocation", hole=0.4, color_discrete_sequence=px.colors.qualitative.Pastel)
            fig_alloc.update_layout(template="plotly_dark", margin=dict(l=10, r=10, t=40, b=10), height=300)
            st.plotly_chart(fig_alloc, width='stretch')
            
        with ap2:
            st.dataframe(df_pos, width='stretch', hide_index=True, column_config={
                "Value": st.column_config.NumberColumn(format="₹%.2f"),
                "Unreal P&L": st.column_config.NumberColumn(format="₹%.2f"),
                "Entry": st.column_config.NumberColumn(format="₹%.2f"),
                "Current": st.column_config.NumberColumn(format="₹%.2f")
            })
    else:
        st.info("No open positions at the moment.")

# ── STOCK EXPLORER ───────────────────────────────────────────────────────────

elif page == "🔍 Stock Explorer":
    st.title("🔍 Technical Stock Explorer")
    st.caption("Deep-dive into Nifty 50 price action with professional indicators.")
    
    selected_sym = st.selectbox("Select a Stock to Analyze", DATA_CONFIG["symbols"], format_func=lambda x: f"{x} ({NIFTY50_NAMES.get(x, 'Nifty 50')})")
    
    with st.spinner(f"Fetching live data for {selected_sym}..."):
        df = yf.download(selected_sym, period="6mo", interval="1d", progress=False)
        if not df.empty:
            # Flatten columns if MultiIndex (yf fix)
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            
            # Simple Indicator Calculations (for visual only)
            df['MA20'] = df['Close'].rolling(window=20).mean()
            df['STD'] = df['Close'].rolling(window=20).std()
            df['Upper'] = df['MA20'] + (df['STD'] * 2)
            df['Lower'] = df['MA20'] - (df['STD'] * 2)
            
            # Create Chart
            fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.05, row_heights=[0.6, 0.2, 0.2], subplot_titles=("Price Action & Bollinger Bands", "Volume", "RSI (Relative Strength)"))
            
            # 1. Candlestick
            fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name="Price"), row=1, col=1)
            
            # 2. Bollinger Bands
            fig.add_trace(go.Scatter(x=df.index, y=df['Upper'], name='Upper Band', line=dict(color='rgba(173, 216, 230, 0.4)', width=1)), row=1, col=1)
            fig.add_trace(go.Scatter(x=df.index, y=df['Lower'], name='Lower Band', line=dict(color='rgba(173, 216, 230, 0.4)', width=1), fill='tonexty'), row=1, col=1)
            
            # 3. Volume
            colors = ['red' if df['Open'].iloc[i] > df['Close'].iloc[i] else 'green' for i in range(len(df))]
            fig.add_trace(go.Bar(x=df.index, y=df['Volume'], name="Volume", marker_color=colors, opacity=0.5), row=2, col=1)
            
            # 4. RSI (Placeholder logic for display)
            delta = df['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            df['RSI'] = 100 - (100 / (1 + rs))
            
            fig.add_trace(go.Scatter(x=df.index, y=df['RSI'], name="RSI", line=dict(color='#ff9900')), row=3, col=1)
            fig.add_hline(y=70, line_dash="dash", line_color="red", row=3, col=1)
            fig.add_hline(y=30, line_dash="dash", line_color="green", row=3, col=1)
            
            fig.update_layout(template="plotly_dark", height=800, xaxis_rangeslider_visible=False, showlegend=False, margin=dict(l=10, r=10, t=50, b=10))
            st.plotly_chart(fig, width='stretch')
            
            # Beginner Tip
            st.info("💡 **Beginner Tip:** When the price touches the **Lower Band** (shaded area) and the **RSI** is below the green line, the stock might be 'On Sale'. When it hits the **Upper Band** and the red line, it might be 'Overpriced'.")
        else:
            st.error("Could not fetch data. Check your internet connection or the symbol name.")

# ── INTELLIGENCE REPORT ───────────────────────────────────────────────────────

elif page == "📘 Intelligence Report":
    st.title("📘 Detailed Intelligence Report")
    report_path = os.path.join("state", "learning_report.md")
    
    if os.path.exists(report_path):
        with open(report_path, "r", encoding="utf-8") as f:
            content = f.read()
        
        # Professional rendering
        st.markdown(content)
        
        st.divider()
        if st.button("🔄 Regenerate Full Report"):
            import subprocess
            with st.spinner("Analyzing patterns and rebuilding report..."):
                subprocess.run([sys.executable, "main.py", "--journal"])
                st.rerun()
    else:
        st.warning("Intelligence report not found. Please click below to generate it for the first time.")
        if st.button("🚀 Generate First Report"):
            import subprocess
            subprocess.run([sys.executable, "main.py", "--journal"])
            st.rerun()

elif page == "🏫 Education Center":
    st.title("🏫 QuantEdge Academy")
    st.subheader("How I Learn to Trade (Step-by-Step)")
    st.caption("I am designed to learn like a human trader—by making mistakes, observing outcomes, and remembering what works.")
    st.header("🧠 1. The Bot's Brain (Indicators)")
    col_a, col_b = st.columns(2)
    with col_a:
        with st.expander("📉 RSI (Relative Strength Index)", expanded=True):
            st.write("**What is it?** Measures if a stock is over-excited (Overbought) or depressed (Oversold).")
            st.write("**How I use it:** I look for 'Oversold' dips (RSI < 30) as potential buying opportunities.")
        with st.expander("📊 ADX (Trend Strength)"):
            st.write("**What is it?** Measures how strong the current move is.")
            st.write("**How I use it:** I don't like messy, sideways markets. I prefer to trade when ADX > 25 (Strong Trend).")
    with col_b:
        with st.expander("📈 MACD (Momentum)"):
            st.write("**What is it?** Shows the 'speed' of price changes.")
            st.write("**How I use it:** I wait for the MACD line to cross its signal line to confirm the momentum is on our side.")
        with st.expander("🔘 Bollinger Bands"):
            st.write("**What is it?** Creates a 'tunnel' around the price.")
            st.write("**How I use it:** If price touches the bottom of the tunnel, I look for a rebound.")
    st.divider()
    st.header("👶 2. My Learning Stages (Baby Steps)")
    s1, s2, s3 = st.columns(3)
    with s1:
        st.markdown("### 👀 1. WATCHING")
        st.write("I've seen a pattern (like a hammer candle at support), but I haven't traded it yet. I'm just keeping an eye on it.")
    with s2:
        st.markdown("### 🧪 2. LEARNING")
        st.write("I've taken a few trades (2–4). I'm starting to see if this pattern is actually profitable or just lucky.")
    with s3:
        st.markdown("### ✅ 3. LEARNED")
        st.write("I've seen this 5+ times and it has a >60% success rate. I now trust this pattern and will prioritize it.")
    st.divider()
    st.header("🌍 3. Market Regimes (The Weather)")
    st.write("Just like you wear a jacket when it's raining, I change my strategy based on the 'Market Weather'.")
    r1, r2, r3 = st.columns(3)
    r1.info("**Trending Up**\n\nThe sun is out. I buy aggressively and hold for big profits.")
    r2.warning("**Sideways / Volatile**\n\nIt's cloudy. I trade with half-size and take profits quickly.")
    r3.error("**Trending Down**\n\nA storm is here. I stop buying entirely to protect your capital.")
    st.divider()
    st.header("🔬 4. The Perfect Learning Loop")
    st.write("I follow a strict 4-step process to ensure I learn only the most profitable patterns.")
    l1, l2, l3, l4 = st.columns(4)
    with l1:
        st.markdown("📸 **Step 1: Snapshot**")
        st.write("When a 'BUY' signal appears, I take a high-resolution snapshot of **11 different market variables** (RSI, ADX, Trend, etc.).")
    with l2:
        st.markdown("🎯 **Step 2: Execution**")
        st.write("I take the trade in your Paper Portfolio. I don't hesitate. I follow the rules perfectly every time.")
    with l3:
        st.markdown("📝 **Step 3: Journaling**")
        st.write("Once the trade is closed, I record the result back into my **Knowledge Base**. I link the outcome directly to the initial snapshot.")
    with l4:
        st.markdown("🛡️ **Step 4: Filtering**")
        st.write("Tomorrow, before I take a trade, I check my memory. If I see a pattern that failed last time, **I skip it.**")

    st.divider()
    st.subheader("🤖 My Current Progress")
    if journal:
        total_p = len(journal.get("patterns", {}))
        total_obs = journal.get("metadata", {}).get("total_observations", 0)
        st.write(f"I am currently tracking **{total_p}** different pattern combinations across **{total_obs}** historical observations.")
        st.progress(min(total_obs / 500, 1.0), text="Bot Maturity Progress")
    st.info("💡 **Pro Tip:** Keep running me every morning. Every trade I take makes me smarter and more accurate for tomorrow.")

st.divider()
st.caption("QuantEdge Bot Framework © 2026")
