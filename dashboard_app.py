import streamlit as st
import pandas as pd
import json
import os
import sys
from datetime import datetime
import plotly.graph_objects as go
import plotly.express as px
from config.settings import DATA_CONFIG, RISK_CONFIG

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
    if not os.path.exists(filepath): return None
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)

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
    page = st.radio("Navigation", ["📊 Live Dashboard", "📘 Intelligence Report", "🏫 Education Center"])
    st.divider()
    portfolio = load_json(PAPER_PORTFOLIO_FILE)
    journal = load_json(LEARNING_JOURNAL_FILE)
    if portfolio: st.success("✅ Portfolio Connected")
    else: st.error("❌ Portfolio Data Not Found")
    if journal: st.success("✅ Knowledge Base Connected")
    else: st.warning("⚠️ Journal Not Found")
    st.divider()
    st.info("Current Mode: **Paper Trading**")
    if st.button("🔄 Refresh Data"): st.rerun()

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
        p_count = len(journal.get('patterns', {}))
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
    positions = portfolio.get("positions", {})
    closed_trades = portfolio.get("closed_trades", [])

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

    st.divider()
    st.subheader("📓 Trade Journal")
    if closed_trades:
        df_all = pd.DataFrame(closed_trades)
        display_cols = ['symbol', 'exit_date', 'pnl', 'exit_reason', 'regime', 'slippage_cost']
        st.dataframe(df_all[display_cols].sort_values('exit_date', ascending=False), width='stretch', hide_index=True, column_config={"pnl": st.column_config.NumberColumn("PnL (₹)", format="₹%.2f"), "slippage_cost": st.column_config.NumberColumn("Slippage (₹)", format="₹%.2f"), "exit_date": "Date", "symbol": "Symbol", "exit_reason": "Reason", "regime": "Regime"})
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
            p_list.append({
                "Symbol": sym,
                "Value": current_val,
                "Entry": p["entry_price"],
                "Current": p["current_price"],
                "Shares": p["shares"],
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
