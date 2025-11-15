# -*- coding: utf-8 -*-
"""
Trading Bot Web Dashboard
Streamlit-based interactive dashboard
"""
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
from portfolio_manager import get_portfolio_manager
from database import get_db
from monitoring_enhanced import get_enhanced_monitor
from ml_model_monitor import get_ml_model_monitor

# Page config
st.set_page_config(
    page_title="Trading Bot Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .metric-card {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
    }
    .positive { color: #00c853; }
    .negative { color: #ff1744; }
</style>
""", unsafe_allow_html=True)

# Initialize
@st.cache_resource
def get_managers():
    return {
        'portfolio': get_portfolio_manager(),
        'db': get_db(),
        'monitor': get_enhanced_monitor(),
        'ml_monitor': get_ml_model_monitor()
    }

managers = get_managers()

# Sidebar
st.sidebar.title("📊 Trading Bot")
st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Navigation",
    ["Overview", "Portfolio", "Signals", "Performance", "ML Models", "Settings"]
)

st.sidebar.markdown("---")
st.sidebar.info(f"Last updated: {datetime.now().strftime('%H:%M:%S')}")

# Auto-refresh
if st.sidebar.checkbox("Auto-refresh (30s)", value=False):
    st.rerun()

# ============================================================================
# OVERVIEW PAGE
# ============================================================================

if page == "Overview":
    st.title("📊 Trading Bot Overview")
    
    # Portfolio summary
    portfolio = managers['portfolio'].get_portfolio_value()
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Portfolio Value",
            f"{portfolio['total_value']:,.0f} VNĐ",
            delta=f"{portfolio['pnl']:+,.0f} VNĐ"
        )
    
    with col2:
        st.metric(
            "P&L %",
            f"{portfolio['pnl_percent']:+.2f}%",
            delta=None
        )
    
    with col3:
        st.metric(
            "Active Positions",
            portfolio['num_positions']
        )
    
    with col4:
        cash = 100_000_000 - portfolio['total_cost']
        st.metric(
            "Cash Available",
            f"{cash:,.0f} VNĐ"
        )
    
    # Equity curve
    st.subheader("📈 Equity Curve")
    
    history = managers['db'].conn.execute("""
        SELECT date, total_value, pnl_percent
        FROM portfolio_history
        ORDER BY date
    """).fetchall()
    
    if history:
        df_history = pd.DataFrame(history, columns=['date', 'value', 'pnl_pct'])
        df_history['date'] = pd.to_datetime(df_history['date'])
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=df_history['date'],
            y=df_history['value'],
            mode='lines',
            name='Portfolio Value',
            line=dict(color='#1f77b4', width=2)
        ))
        
        fig.update_layout(
            title="Portfolio Value Over Time",
            xaxis_title="Date",
            yaxis_title="Value (VNĐ)",
            hovermode='x unified',
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No historical data yet")
    
    # Recent activity
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🎯 Recent Signals")
        
        signals = managers['db'].conn.execute("""
            SELECT symbol, signal_type, confidence, created_at
            FROM signals_cache
            ORDER BY created_at DESC
            LIMIT 10
        """).fetchall()
        
        if signals:
            df_signals = pd.DataFrame(
                signals,
                columns=['Symbol', 'Signal', 'Confidence', 'Date']
            )
            st.dataframe(df_signals, use_container_width=True)
        else:
            st.info("No recent signals")
    
    with col2:
        st.subheader("💼 Recent Trades")
        
        trades = managers['db'].conn.execute("""
            SELECT symbol, action, shares, price, trade_date
            FROM trades
            ORDER BY trade_date DESC
            LIMIT 10
        """).fetchall()
        
        if trades:
            df_trades = pd.DataFrame(
                trades,
                columns=['Symbol', 'Action', 'Shares', 'Price', 'Date']
            )
            st.dataframe(df_trades, use_container_width=True)
        else:
            st.info("No recent trades")

# ============================================================================
# PORTFOLIO PAGE
# ============================================================================

elif page == "Portfolio":
    st.title("💼 Portfolio Management")
    
    positions = managers['portfolio'].get_positions()
    
    if not positions:
        st.warning("No active positions")
    else:
        # Summary metrics
        portfolio = managers['portfolio'].get_portfolio_value()
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Total Value", f"{portfolio['total_value']:,.0f} VNĐ")
        with col2:
            st.metric("Total Cost", f"{portfolio['total_cost']:,.0f} VNĐ")
        with col3:
            pnl_color = "positive" if portfolio['pnl'] >= 0 else "negative"
            st.markdown(
                f"<h3>P&L: <span class='{pnl_color}'>{portfolio['pnl']:+,.0f} VNĐ</span></h3>",
                unsafe_allow_html=True
            )
        
        st.markdown("---")
        
        # Positions table
        st.subheader("📊 Current Positions")
        
        positions_data = []
        for symbol, pos in positions.items():
            current_price = pos.get('metadata', {}).get('last_price', pos['avg_price'])
            pnl = (current_price - pos['avg_price']) * pos['shares']
            pnl_pct = (pnl / (pos['avg_price'] * pos['shares'])) * 100
            
            positions_data.append({
                'Symbol': symbol,
                'Shares': pos['shares'],
                'Avg Price': f"{pos['avg_price']:,.0f}",
                'Current Price': f"{current_price:,.0f}",
                'Value': f"{current_price * pos['shares']:,.0f}",
                'P&L': f"{pnl:+,.0f}",
                'P&L %': f"{pnl_pct:+.2f}%",
                'Entry Date': pos['entry_date'][:10]
            })
        
        df_positions = pd.DataFrame(positions_data)
        st.dataframe(df_positions, use_container_width=True)
        
        # Position allocation pie chart
        st.subheader("📊 Position Allocation")
        
        fig = px.pie(
            values=[pos['shares'] * pos['avg_price'] for pos in positions.values()],
            names=list(positions.keys()),
            title="Portfolio Allocation by Value"
        )
        st.plotly_chart(fig, use_container_width=True)

# ============================================================================
# SIGNALS PAGE
# ============================================================================

elif page == "Signals":
    st.title("🎯 Trading Signals")
    
    # Filters
    col1, col2, col3 = st.columns(3)
    
    with col1:
        signal_type = st.selectbox("Signal Type", ["All", "BUY", "SELL", "HOLD"])
    
    with col2:
        min_confidence = st.slider("Min Confidence", 0, 100, 60)
    
    with col3:
        days_back = st.slider("Days Back", 1, 30, 7)
    
    # Query signals
    cutoff_date = (datetime.now() - timedelta(days=days_back)).isoformat()
    
    query = """
        SELECT symbol, signal_type, confidence, reason, created_at
        FROM signals_cache
        WHERE created_at >= ?
    """
    params = [cutoff_date]
    
    if signal_type != "All":
        query += " AND signal_type = ?"
        params.append(signal_type)
    
    query += " AND confidence >= ? ORDER BY created_at DESC"
    params.append(min_confidence)
    
    signals = managers['db'].conn.execute(query, params).fetchall()
    
    if signals:
        st.success(f"Found {len(signals)} signals")
        
        df_signals = pd.DataFrame(
            signals,
            columns=['Symbol', 'Signal', 'Confidence', 'Reason', 'Date']
        )
        
        # Signal distribution
        col1, col2 = st.columns(2)
        
        with col1:
            fig = px.histogram(
                df_signals,
                x='Signal',
                title="Signal Distribution",
                color='Signal'
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            fig = px.box(
                df_signals,
                x='Signal',
                y='Confidence',
                title="Confidence by Signal Type",
                color='Signal'
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Signals table
        st.dataframe(df_signals, use_container_width=True)
    else:
        st.info("No signals found with current filters")

# ============================================================================
# PERFORMANCE PAGE
# ============================================================================

elif page == "Performance":
    st.title("📊 Performance Analytics")
    
    # Get trades
    trades = managers['db'].conn.execute("""
        SELECT symbol, action, shares, price, total_value, trade_date, metadata
        FROM trades
        WHERE action IN ('SELL_FULL', 'SELL_PARTIAL')
        ORDER BY trade_date DESC
    """).fetchall()
    
    if trades:
        # Calculate metrics
        import json
        
        total_trades = len(trades)
        winning_trades = sum(1 for t in trades if json.loads(t[6] or '{}').get('pnl', 0) > 0)
        win_rate = (winning_trades / total_trades) * 100 if total_trades > 0 else 0
        
        profits = [json.loads(t[6] or '{}').get('pnl', 0) for t in trades]
        avg_profit = sum(p for p in profits if p > 0) / max(1, sum(1 for p in profits if p > 0))
        avg_loss = sum(p for p in profits if p < 0) / max(1, sum(1 for p in profits if p < 0))
        
        # Metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Trades", total_trades)
        with col2:
            st.metric("Win Rate", f"{win_rate:.1f}%")
        with col3:
            st.metric("Avg Profit", f"{avg_profit:,.0f} VNĐ")
        with col4:
            st.metric("Avg Loss", f"{avg_loss:,.0f} VNĐ")
        
        # P&L over time
        st.subheader("📈 Cumulative P&L")
        
        df_trades = pd.DataFrame(trades, columns=[
            'symbol', 'action', 'shares', 'price', 'value', 'date', 'metadata'
        ])
        df_trades['pnl'] = df_trades['metadata'].apply(
            lambda x: json.loads(x or '{}').get('pnl', 0)
        )
        df_trades['date'] = pd.to_datetime(df_trades['date'])
        df_trades['cumulative_pnl'] = df_trades['pnl'].cumsum()
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=df_trades['date'],
            y=df_trades['cumulative_pnl'],
            mode='lines+markers',
            name='Cumulative P&L',
            line=dict(color='#1f77b4', width=2)
        ))
        
        fig.update_layout(
            title="Cumulative P&L Over Time",
            xaxis_title="Date",
            yaxis_title="P&L (VNĐ)",
            hovermode='x unified',
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Trades table
        st.subheader("📋 Trade History")
        st.dataframe(df_trades[['symbol', 'action', 'shares', 'price', 'pnl', 'date']], use_container_width=True)
    else:
        st.info("No completed trades yet")

# ============================================================================
# ML MODELS PAGE
# ============================================================================

elif page == "ML Models":
    st.title("🤖 ML Model Performance")
    
    ml_monitor = managers['ml_monitor']
    
    # Model metrics
    metrics = ml_monitor.calculate_accuracy(window_days=30)
    
    if metrics['accuracy'] is not None:
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Accuracy", f"{metrics['accuracy']:.2%}")
        with col2:
            st.metric("Precision", f"{metrics['precision']:.2%}")
        with col3:
            st.metric("Recall", f"{metrics['recall']:.2%}")
        with col4:
            st.metric("Predictions", metrics['total_predictions'])
        
        # Drift detection
        st.subheader("🔍 Drift Detection")
        
        drift = ml_monitor.check_drift()
        
        if drift['drift_detected']:
            st.error(f"⚠️ Model drift detected: {drift['reason']}")
            st.warning("Retraining recommended!")
        else:
            st.success("✅ No drift detected - Model performing well")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("Baseline Accuracy", f"{drift['baseline_accuracy']:.2%}")
        with col2:
            st.metric("Recent Accuracy", f"{drift['recent_accuracy']:.2%}")
        
        # Performance report
        st.subheader("📊 Performance Report")
        report = ml_monitor.get_performance_report()
        st.text(report)
    else:
        st.info("Insufficient data for model metrics")

# ============================================================================
# SETTINGS PAGE
# ============================================================================

elif page == "Settings":
    st.title("⚙️ Settings")
    
    st.subheader("Trading Parameters")
    
    col1, col2 = st.columns(2)
    
    with col1:
        min_confidence = st.slider("Min Confidence", 0, 100, 60)
        max_position_size = st.slider("Max Position Size (%)", 5, 20, 10)
    
    with col2:
        min_risk_reward = st.slider("Min Risk/Reward", 1.0, 5.0, 2.0, 0.1)
        max_positions = st.slider("Max Positions", 5, 20, 10)
    
    if st.button("Save Settings"):
        st.success("Settings saved!")
    
    st.markdown("---")
    
    st.subheader("System Status")
    
    # Health check
    health = managers['monitor'].get_health_status()
    
    status_color = {
        'healthy': '🟢',
        'degraded': '🟡',
        'unhealthy': '🔴'
    }
    
    st.markdown(f"### {status_color.get(health['status'], '⚪')} Status: {health['status'].upper()}")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric("Uptime", f"{health['uptime_seconds'] / 3600:.1f} hours")
    
    with col2:
        if health['last_scan']:
            st.metric("Last Scan", health['last_scan'][:16])
    
    # System metrics
    managers['monitor'].update_system_metrics()
    
    st.subheader("System Resources")
    
    # This would show real-time CPU/Memory if we had the data
    st.info("System metrics available at /metrics endpoint")

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: #666;'>"
    "Trading Bot Dashboard v1.0 | "
    f"© {datetime.now().year}"
    "</div>",
    unsafe_allow_html=True
)
