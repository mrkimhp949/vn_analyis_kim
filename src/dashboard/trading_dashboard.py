# -*- coding: utf-8 -*-
"""
Real-time Trading Monitoring Dashboard

Flask-based real-time dashboard for monitoring:
- Portfolio positions and P&L
- Active signals and trades
- Risk metrics and circuit breakers
- Foreign flow monitoring
- ML model performance

Usage:
    from src.dashboard.trading_dashboard import start_dashboard
    
    # Start dashboard on port 8050
    start_dashboard(port=8050)
    
    # Or run directly
    python -m src.dashboard.trading_dashboard
"""

import json
import logging
import os
import threading
from collections import deque
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

from flask import Flask, jsonify, render_template_string
from flask_socketio import SocketIO, emit

logger = logging.getLogger(__name__)

# =============================================================================
# DASHBOARD TEMPLATE
# =============================================================================

DASHBOARD_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>🇻🇳 VN Trading Dashboard</title>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/socket.io/4.6.1/socket.io.min.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        :root {
            --bg-primary: #0d1117;
            --bg-secondary: #161b22;
            --bg-card: #21262d;
            --border-color: #30363d;
            --text-primary: #c9d1d9;
            --text-secondary: #8b949e;
            --green: #3fb950;
            --red: #f85149;
            --yellow: #d29922;
            --blue: #58a6ff;
            --purple: #a371f7;
        }
        
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: var(--bg-primary);
            color: var(--text-primary);
            min-height: 100vh;
        }
        
        .header {
            background: var(--bg-secondary);
            padding: 1rem 2rem;
            border-bottom: 1px solid var(--border-color);
            display: flex;
            justify-content: space-between;
            align-items: center;
        }
        
        .header h1 {
            font-size: 1.5rem;
            display: flex;
            align-items: center;
            gap: 0.5rem;
        }
        
        .status-indicator {
            display: flex;
            align-items: center;
            gap: 0.5rem;
        }
        
        .status-dot {
            width: 10px;
            height: 10px;
            border-radius: 50%;
            background: var(--green);
            animation: pulse 2s infinite;
        }
        
        @keyframes pulse {
            0%, 100% { opacity: 1; }
            50% { opacity: 0.5; }
        }
        
        .container {
            display: grid;
            grid-template-columns: repeat(12, 1fr);
            gap: 1rem;
            padding: 1rem;
        }
        
        .card {
            background: var(--bg-card);
            border: 1px solid var(--border-color);
            border-radius: 8px;
            padding: 1rem;
        }
        
        .card-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 1rem;
            padding-bottom: 0.5rem;
            border-bottom: 1px solid var(--border-color);
        }
        
        .card-title {
            font-size: 0.875rem;
            font-weight: 600;
            color: var(--text-secondary);
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }
        
        /* Portfolio Card */
        .portfolio { grid-column: span 4; }
        
        .metric-grid {
            display: grid;
            grid-template-columns: repeat(2, 1fr);
            gap: 1rem;
        }
        
        .metric {
            text-align: center;
        }
        
        .metric-value {
            font-size: 1.5rem;
            font-weight: 700;
        }
        
        .metric-label {
            font-size: 0.75rem;
            color: var(--text-secondary);
            margin-top: 0.25rem;
        }
        
        .positive { color: var(--green); }
        .negative { color: var(--red); }
        .neutral { color: var(--text-primary); }
        
        /* Positions Card */
        .positions { grid-column: span 8; }
        
        .positions-table {
            width: 100%;
            border-collapse: collapse;
        }
        
        .positions-table th,
        .positions-table td {
            padding: 0.5rem;
            text-align: left;
            border-bottom: 1px solid var(--border-color);
        }
        
        .positions-table th {
            color: var(--text-secondary);
            font-size: 0.75rem;
            text-transform: uppercase;
        }
        
        /* Signals Card */
        .signals { grid-column: span 6; }
        
        .signal-item {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 0.75rem;
            border-radius: 6px;
            margin-bottom: 0.5rem;
            background: var(--bg-secondary);
        }
        
        .signal-symbol {
            font-weight: 600;
        }
        
        .signal-type {
            padding: 0.25rem 0.5rem;
            border-radius: 4px;
            font-size: 0.75rem;
            font-weight: 600;
        }
        
        .signal-buy { background: rgba(63, 185, 80, 0.2); color: var(--green); }
        .signal-sell { background: rgba(248, 81, 73, 0.2); color: var(--red); }
        
        /* Chart Card */
        .chart { grid-column: span 6; }
        
        .chart-container {
            height: 200px;
        }
        
        /* Foreign Flow Card */
        .foreign-flow { grid-column: span 6; }
        
        .flow-bar {
            display: flex;
            height: 30px;
            border-radius: 4px;
            overflow: hidden;
            margin: 1rem 0;
        }
        
        .flow-buy {
            background: var(--green);
            display: flex;
            align-items: center;
            justify-content: center;
            color: white;
            font-size: 0.75rem;
            font-weight: 600;
        }
        
        .flow-sell {
            background: var(--red);
            display: flex;
            align-items: center;
            justify-content: center;
            color: white;
            font-size: 0.75rem;
            font-weight: 600;
        }
        
        /* Risk Card */
        .risk { grid-column: span 6; }
        
        .risk-meter {
            height: 8px;
            background: var(--bg-secondary);
            border-radius: 4px;
            margin: 0.5rem 0;
            overflow: hidden;
        }
        
        .risk-fill {
            height: 100%;
            border-radius: 4px;
            transition: width 0.3s;
        }
        
        .risk-low { background: var(--green); }
        .risk-medium { background: var(--yellow); }
        .risk-high { background: var(--red); }
        
        /* Logs Card */
        .logs { grid-column: span 12; }
        
        .log-container {
            max-height: 200px;
            overflow-y: auto;
            font-family: 'Fira Code', monospace;
            font-size: 0.75rem;
        }
        
        .log-entry {
            padding: 0.25rem 0;
            display: flex;
            gap: 1rem;
        }
        
        .log-time {
            color: var(--text-secondary);
            white-space: nowrap;
        }
        
        .log-info { color: var(--blue); }
        .log-warning { color: var(--yellow); }
        .log-error { color: var(--red); }
        .log-success { color: var(--green); }
        
        /* Circuit Breaker */
        .circuit-breaker { grid-column: span 4; }
        
        .breaker-item {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 0.5rem 0;
            border-bottom: 1px solid var(--border-color);
        }
        
        .breaker-status {
            display: inline-block;
            padding: 0.25rem 0.5rem;
            border-radius: 4px;
            font-size: 0.75rem;
            font-weight: 600;
        }
        
        .breaker-closed { background: rgba(63, 185, 80, 0.2); color: var(--green); }
        .breaker-open { background: rgba(248, 81, 73, 0.2); color: var(--red); }
        .breaker-half-open { background: rgba(210, 153, 34, 0.2); color: var(--yellow); }
        
        /* ML Performance */
        .ml-performance { grid-column: span 4; }
        
        .ml-model {
            padding: 0.75rem;
            margin-bottom: 0.5rem;
            background: var(--bg-secondary);
            border-radius: 6px;
        }
        
        .ml-model-name {
            font-weight: 600;
            margin-bottom: 0.5rem;
        }
        
        .ml-stats {
            display: flex;
            gap: 1rem;
            font-size: 0.75rem;
        }
    </style>
</head>
<body>
    <div class="header">
        <h1>🇻🇳 Vietnam Trading Dashboard</h1>
        <div class="status-indicator">
            <div class="status-dot" id="connectionStatus"></div>
            <span id="statusText">Connected</span>
            <span style="margin-left: 1rem; color: var(--text-secondary);" id="lastUpdate"></span>
        </div>
    </div>
    
    <div class="container">
        <!-- Portfolio Summary -->
        <div class="card portfolio">
            <div class="card-header">
                <span class="card-title">📊 Portfolio Summary</span>
            </div>
            <div class="metric-grid">
                <div class="metric">
                    <div class="metric-value" id="totalValue">₫0</div>
                    <div class="metric-label">Total Value</div>
                </div>
                <div class="metric">
                    <div class="metric-value" id="dayPnL">₫0</div>
                    <div class="metric-label">Day P&L</div>
                </div>
                <div class="metric">
                    <div class="metric-value" id="totalPnL">₫0</div>
                    <div class="metric-label">Total P&L</div>
                </div>
                <div class="metric">
                    <div class="metric-value" id="positionCount">0</div>
                    <div class="metric-label">Positions</div>
                </div>
            </div>
        </div>
        
        <!-- Circuit Breaker Status -->
        <div class="card circuit-breaker">
            <div class="card-header">
                <span class="card-title">🔌 Circuit Breakers</span>
            </div>
            <div id="circuitBreakers">
                <div class="breaker-item">
                    <span>Loading...</span>
                </div>
            </div>
        </div>
        
        <!-- ML Model Performance -->
        <div class="card ml-performance">
            <div class="card-header">
                <span class="card-title">🤖 ML Performance</span>
            </div>
            <div id="mlModels">
                <div class="ml-model">
                    <div class="ml-model-name">Loading...</div>
                </div>
            </div>
        </div>
        
        <!-- Positions -->
        <div class="card positions">
            <div class="card-header">
                <span class="card-title">📋 Open Positions</span>
            </div>
            <table class="positions-table">
                <thead>
                    <tr>
                        <th>Symbol</th>
                        <th>Qty</th>
                        <th>Entry</th>
                        <th>Current</th>
                        <th>P&L</th>
                        <th>P&L %</th>
                        <th>Stop Loss</th>
                    </tr>
                </thead>
                <tbody id="positionsTable">
                    <tr>
                        <td colspan="7" style="text-align: center; color: var(--text-secondary);">
                            No open positions
                        </td>
                    </tr>
                </tbody>
            </table>
        </div>
        
        <!-- Active Signals -->
        <div class="card signals">
            <div class="card-header">
                <span class="card-title">📡 Active Signals</span>
            </div>
            <div id="activeSignals">
                <div class="signal-item">
                    <span style="color: var(--text-secondary);">No active signals</span>
                </div>
            </div>
        </div>
        
        <!-- P&L Chart -->
        <div class="card chart">
            <div class="card-header">
                <span class="card-title">📈 P&L Chart (Today)</span>
            </div>
            <div class="chart-container">
                <canvas id="pnlChart"></canvas>
            </div>
        </div>
        
        <!-- Foreign Flow -->
        <div class="card foreign-flow">
            <div class="card-header">
                <span class="card-title">🌍 Foreign Flow</span>
            </div>
            <div id="foreignFlowContent">
                <div class="flow-bar">
                    <div class="flow-buy" id="foreignBuy" style="width: 50%;">BUY</div>
                    <div class="flow-sell" id="foreignSell" style="width: 50%;">SELL</div>
                </div>
                <div style="display: flex; justify-content: space-between;">
                    <span id="foreignBuyValue">₫0</span>
                    <span id="foreignSellValue">₫0</span>
                </div>
                <div style="text-align: center; margin-top: 1rem; font-size: 1.25rem; font-weight: 700;">
                    Net: <span id="foreignNet">₫0</span>
                </div>
            </div>
        </div>
        
        <!-- Risk Metrics -->
        <div class="card risk">
            <div class="card-header">
                <span class="card-title">⚠️ Risk Metrics</span>
            </div>
            <div id="riskMetrics">
                <div style="margin-bottom: 1rem;">
                    <div style="display: flex; justify-content: space-between;">
                        <span>Portfolio Risk</span>
                        <span id="portfolioRiskValue">0%</span>
                    </div>
                    <div class="risk-meter">
                        <div class="risk-fill risk-low" id="portfolioRiskBar" style="width: 0%"></div>
                    </div>
                </div>
                <div style="margin-bottom: 1rem;">
                    <div style="display: flex; justify-content: space-between;">
                        <span>Max Drawdown</span>
                        <span id="maxDrawdownValue">0%</span>
                    </div>
                    <div class="risk-meter">
                        <div class="risk-fill risk-medium" id="maxDrawdownBar" style="width: 0%"></div>
                    </div>
                </div>
                <div style="margin-bottom: 1rem;">
                    <div style="display: flex; justify-content: space-between;">
                        <span>Daily Loss Limit</span>
                        <span id="dailyLossValue">0%</span>
                    </div>
                    <div class="risk-meter">
                        <div class="risk-fill risk-high" id="dailyLossBar" style="width: 0%"></div>
                    </div>
                </div>
            </div>
        </div>
        
        <!-- Activity Logs -->
        <div class="card logs">
            <div class="card-header">
                <span class="card-title">📝 Activity Log</span>
            </div>
            <div class="log-container" id="activityLog">
                <div class="log-entry">
                    <span class="log-time">--:--:--</span>
                    <span class="log-info">Dashboard initialized</span>
                </div>
            </div>
        </div>
    </div>
    
    <script>
        // Socket.IO connection
        const socket = io();
        
        // P&L Chart
        const ctx = document.getElementById('pnlChart').getContext('2d');
        const pnlChart = new Chart(ctx, {
            type: 'line',
            data: {
                labels: [],
                datasets: [{
                    label: 'P&L',
                    data: [],
                    borderColor: '#3fb950',
                    backgroundColor: 'rgba(63, 185, 80, 0.1)',
                    fill: true,
                    tension: 0.4
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: { display: false }
                },
                scales: {
                    x: {
                        grid: { color: '#30363d' },
                        ticks: { color: '#8b949e' }
                    },
                    y: {
                        grid: { color: '#30363d' },
                        ticks: { color: '#8b949e' }
                    }
                }
            }
        });
        
        // Format currency
        function formatVND(value) {
            return '₫' + Math.abs(value).toLocaleString();
        }
        
        // Format percentage
        function formatPct(value) {
            return (value >= 0 ? '+' : '') + value.toFixed(2) + '%';
        }
        
        // Set value class
        function valueClass(value) {
            if (value > 0) return 'positive';
            if (value < 0) return 'negative';
            return 'neutral';
        }
        
        // Add log entry
        function addLog(message, type = 'info') {
            const logContainer = document.getElementById('activityLog');
            const now = new Date().toLocaleTimeString();
            
            const entry = document.createElement('div');
            entry.className = 'log-entry';
            entry.innerHTML = `
                <span class="log-time">${now}</span>
                <span class="log-${type}">${message}</span>
            `;
            
            logContainer.insertBefore(entry, logContainer.firstChild);
            
            // Keep only last 50 entries
            while (logContainer.children.length > 50) {
                logContainer.removeChild(logContainer.lastChild);
            }
        }
        
        // Socket event handlers
        socket.on('connect', () => {
            document.getElementById('statusText').textContent = 'Connected';
            document.getElementById('connectionStatus').style.background = '#3fb950';
            addLog('Connected to server', 'success');
        });
        
        socket.on('disconnect', () => {
            document.getElementById('statusText').textContent = 'Disconnected';
            document.getElementById('connectionStatus').style.background = '#f85149';
            addLog('Disconnected from server', 'error');
        });
        
        // Portfolio update
        socket.on('portfolio_update', (data) => {
            document.getElementById('lastUpdate').textContent = 
                'Updated: ' + new Date().toLocaleTimeString();
            
            // Update metrics
            document.getElementById('totalValue').textContent = formatVND(data.total_value);
            
            const dayPnLEl = document.getElementById('dayPnL');
            dayPnLEl.textContent = (data.day_pnl >= 0 ? '+' : '') + formatVND(data.day_pnl);
            dayPnLEl.className = 'metric-value ' + valueClass(data.day_pnl);
            
            const totalPnLEl = document.getElementById('totalPnL');
            totalPnLEl.textContent = (data.total_pnl >= 0 ? '+' : '') + formatVND(data.total_pnl);
            totalPnLEl.className = 'metric-value ' + valueClass(data.total_pnl);
            
            document.getElementById('positionCount').textContent = data.position_count;
            
            // Update P&L chart
            const now = new Date().toLocaleTimeString();
            pnlChart.data.labels.push(now);
            pnlChart.data.datasets[0].data.push(data.day_pnl);
            
            // Keep only last 50 points
            if (pnlChart.data.labels.length > 50) {
                pnlChart.data.labels.shift();
                pnlChart.data.datasets[0].data.shift();
            }
            
            // Update chart color based on P&L
            pnlChart.data.datasets[0].borderColor = data.day_pnl >= 0 ? '#3fb950' : '#f85149';
            pnlChart.data.datasets[0].backgroundColor = data.day_pnl >= 0 
                ? 'rgba(63, 185, 80, 0.1)' 
                : 'rgba(248, 81, 73, 0.1)';
            
            pnlChart.update();
        });
        
        // Positions update
        socket.on('positions_update', (data) => {
            const tbody = document.getElementById('positionsTable');
            
            if (!data.positions || data.positions.length === 0) {
                tbody.innerHTML = `
                    <tr>
                        <td colspan="7" style="text-align: center; color: var(--text-secondary);">
                            No open positions
                        </td>
                    </tr>
                `;
                return;
            }
            
            tbody.innerHTML = data.positions.map(pos => `
                <tr>
                    <td><strong>${pos.symbol}</strong></td>
                    <td>${pos.quantity.toLocaleString()}</td>
                    <td>${pos.entry_price.toLocaleString()}</td>
                    <td>${pos.current_price.toLocaleString()}</td>
                    <td class="${valueClass(pos.pnl)}">${(pos.pnl >= 0 ? '+' : '')}${formatVND(pos.pnl)}</td>
                    <td class="${valueClass(pos.pnl_pct)}">${formatPct(pos.pnl_pct)}</td>
                    <td>${pos.stop_loss ? pos.stop_loss.toLocaleString() : '-'}</td>
                </tr>
            `).join('');
        });
        
        // Signals update
        socket.on('signals_update', (data) => {
            const container = document.getElementById('activeSignals');
            
            if (!data.signals || data.signals.length === 0) {
                container.innerHTML = `
                    <div class="signal-item">
                        <span style="color: var(--text-secondary);">No active signals</span>
                    </div>
                `;
                return;
            }
            
            container.innerHTML = data.signals.map(sig => `
                <div class="signal-item">
                    <div>
                        <span class="signal-symbol">${sig.symbol}</span>
                        <span style="color: var(--text-secondary); margin-left: 0.5rem;">
                            ${sig.confidence.toFixed(0)}% confidence
                        </span>
                    </div>
                    <span class="signal-type signal-${sig.type.toLowerCase()}">${sig.type}</span>
                </div>
            `).join('');
        });
        
        // Foreign flow update
        socket.on('foreign_flow_update', (data) => {
            const total = Math.abs(data.buy_value) + Math.abs(data.sell_value);
            const buyPct = total > 0 ? (data.buy_value / total) * 100 : 50;
            const sellPct = 100 - buyPct;
            
            document.getElementById('foreignBuy').style.width = buyPct + '%';
            document.getElementById('foreignSell').style.width = sellPct + '%';
            
            document.getElementById('foreignBuyValue').textContent = formatVND(data.buy_value);
            document.getElementById('foreignSellValue').textContent = formatVND(data.sell_value);
            
            const netEl = document.getElementById('foreignNet');
            netEl.textContent = (data.net_value >= 0 ? '+' : '') + formatVND(data.net_value);
            netEl.className = valueClass(data.net_value);
        });
        
        // Risk metrics update
        socket.on('risk_update', (data) => {
            // Portfolio risk
            document.getElementById('portfolioRiskValue').textContent = formatPct(data.portfolio_risk);
            const portfolioBar = document.getElementById('portfolioRiskBar');
            portfolioBar.style.width = Math.min(data.portfolio_risk, 100) + '%';
            portfolioBar.className = 'risk-fill ' + 
                (data.portfolio_risk < 30 ? 'risk-low' : data.portfolio_risk < 60 ? 'risk-medium' : 'risk-high');
            
            // Max drawdown
            document.getElementById('maxDrawdownValue').textContent = formatPct(data.max_drawdown);
            const ddBar = document.getElementById('maxDrawdownBar');
            ddBar.style.width = Math.min(Math.abs(data.max_drawdown), 100) + '%';
            ddBar.className = 'risk-fill ' + 
                (Math.abs(data.max_drawdown) < 10 ? 'risk-low' : Math.abs(data.max_drawdown) < 20 ? 'risk-medium' : 'risk-high');
            
            // Daily loss
            document.getElementById('dailyLossValue').textContent = formatPct(data.daily_loss_used);
            const dailyBar = document.getElementById('dailyLossBar');
            dailyBar.style.width = Math.min(data.daily_loss_used, 100) + '%';
            dailyBar.className = 'risk-fill ' + 
                (data.daily_loss_used < 50 ? 'risk-low' : data.daily_loss_used < 80 ? 'risk-medium' : 'risk-high');
        });
        
        // Circuit breaker update
        socket.on('circuit_breaker_update', (data) => {
            const container = document.getElementById('circuitBreakers');
            
            container.innerHTML = data.breakers.map(cb => `
                <div class="breaker-item">
                    <span>${cb.name}</span>
                    <span class="breaker-status breaker-${cb.status.toLowerCase()}">${cb.status}</span>
                </div>
            `).join('');
        });
        
        // ML performance update
        socket.on('ml_performance_update', (data) => {
            const container = document.getElementById('mlModels');
            
            container.innerHTML = data.models.map(model => `
                <div class="ml-model">
                    <div class="ml-model-name">${model.name}</div>
                    <div class="ml-stats">
                        <span>Accuracy: <strong>${model.accuracy.toFixed(1)}%</strong></span>
                        <span>Win Rate: <strong>${model.win_rate.toFixed(1)}%</strong></span>
                        <span>Signals: <strong>${model.signal_count}</strong></span>
                    </div>
                </div>
            `).join('');
        });
        
        // Log events
        socket.on('log_event', (data) => {
            addLog(data.message, data.type);
        });
        
        // Initial data request
        socket.emit('request_initial_data');
        
        addLog('Dashboard initialized', 'info');
    </script>
</body>
</html>
"""


# =============================================================================
# DATA COLLECTORS
# =============================================================================


class DashboardDataCollector:
    """Collects data from trading system components."""

    def __init__(self):
        self._pnl_history: deque = deque(maxlen=1000)
        self._log_buffer: deque = deque(maxlen=100)

    def get_portfolio_data(self) -> Dict[str, Any]:
        """Get portfolio summary data."""
        # Try to get from actual portfolio manager
        try:
            from src.portfolio.manager import get_portfolio_manager

            pm = get_portfolio_manager()

            return {
                "total_value": pm.total_value if hasattr(pm, "total_value") else 0,
                "day_pnl": pm.day_pnl if hasattr(pm, "day_pnl") else 0,
                "total_pnl": pm.total_pnl if hasattr(pm, "total_pnl") else 0,
                "position_count": len(pm.positions) if hasattr(pm, "positions") else 0,
            }
        except:
            # Mock data for demo
            import random

            pnl = random.uniform(-50000, 100000)
            return {
                "total_value": 1_000_000_000 + pnl * 100,
                "day_pnl": pnl,
                "total_pnl": pnl * 10,
                "position_count": random.randint(3, 10),
            }

    def get_positions_data(self) -> Dict[str, Any]:
        """Get open positions data."""
        try:
            from src.portfolio.manager import get_portfolio_manager

            pm = get_portfolio_manager()

            positions = []
            for symbol, pos in pm.positions.items():
                positions.append(
                    {
                        "symbol": symbol,
                        "quantity": pos.quantity,
                        "entry_price": pos.entry_price,
                        "current_price": pos.current_price,
                        "pnl": pos.unrealized_pnl,
                        "pnl_pct": pos.unrealized_pnl_pct,
                        "stop_loss": pos.stop_loss,
                    }
                )

            return {"positions": positions}
        except:
            # Mock data
            import random

            return {
                "positions": [
                    {
                        "symbol": "VNM",
                        "quantity": 1000,
                        "entry_price": 78000,
                        "current_price": 80000,
                        "pnl": 2000000,
                        "pnl_pct": 2.56,
                        "stop_loss": 74000,
                    },
                    {
                        "symbol": "HPG",
                        "quantity": 2000,
                        "entry_price": 25000,
                        "current_price": 24500,
                        "pnl": -1000000,
                        "pnl_pct": -2.0,
                        "stop_loss": 23000,
                    },
                ]
            }

    def get_signals_data(self) -> Dict[str, Any]:
        """Get active signals."""
        try:
            from src.signals.signal_manager import get_signal_manager

            sm = get_signal_manager()

            signals = []
            for sig in sm.active_signals:
                signals.append(
                    {
                        "symbol": sig.symbol,
                        "type": sig.signal_type.value,
                        "confidence": sig.confidence * 100,
                    }
                )

            return {"signals": signals}
        except:
            # Mock data
            import random

            return {
                "signals": [
                    {"symbol": "VCB", "type": "BUY", "confidence": 85},
                    {"symbol": "FPT", "type": "BUY", "confidence": 72},
                    {"symbol": "MBB", "type": "SELL", "confidence": 68},
                ]
            }

    def get_foreign_flow_data(self) -> Dict[str, Any]:
        """Get foreign flow data."""
        try:
            from src.data.hose_hnx_realtime_api import get_foreign_flow_api

            api = get_foreign_flow_api()
            summary = api.get_market_summary()

            return {
                "buy_value": summary.get("total_buy_value", 0),
                "sell_value": summary.get("total_sell_value", 0),
                "net_value": summary.get("net_value", 0),
            }
        except:
            import random

            buy = random.uniform(100e9, 500e9)
            sell = random.uniform(100e9, 500e9)
            return {
                "buy_value": buy,
                "sell_value": sell,
                "net_value": buy - sell,
            }

    def get_risk_data(self) -> Dict[str, Any]:
        """Get risk metrics."""
        try:
            from src.risk.risk_manager import get_risk_manager

            rm = get_risk_manager()

            return {
                "portfolio_risk": rm.current_risk_pct,
                "max_drawdown": rm.max_drawdown,
                "daily_loss_used": rm.daily_loss_pct,
            }
        except:
            import random

            return {
                "portfolio_risk": random.uniform(20, 60),
                "max_drawdown": random.uniform(-5, -15),
                "daily_loss_used": random.uniform(10, 40),
            }

    def get_circuit_breaker_data(self) -> Dict[str, Any]:
        """Get circuit breaker status."""
        try:
            from src.risk.circuit_breaker import CircuitBreakerManager

            # Read from stats file
            with open("circuit_breaker_stats.json") as f:
                stats = json.load(f)

            breakers = []
            for name, info in stats.items():
                breakers.append(
                    {
                        "name": name,
                        "status": info.get("state", "CLOSED").upper(),
                    }
                )

            return {"breakers": breakers}
        except:
            return {
                "breakers": [
                    {"name": "API", "status": "CLOSED"},
                    {"name": "Trading", "status": "CLOSED"},
                    {"name": "Risk", "status": "HALF-OPEN"},
                ]
            }

    def get_ml_performance_data(self) -> Dict[str, Any]:
        """Get ML model performance."""
        try:
            from src.ml.ensemble import get_ensemble_predictor

            ep = get_ensemble_predictor()

            models = []
            for name, perf in ep.model_performance.items():
                models.append(
                    {
                        "name": name,
                        "accuracy": perf.get("accuracy", 0) * 100,
                        "win_rate": perf.get("win_rate", 0) * 100,
                        "signal_count": perf.get("signal_count", 0),
                    }
                )

            return {"models": models}
        except:
            import random

            return {
                "models": [
                    {
                        "name": "LSTM Predictor",
                        "accuracy": random.uniform(55, 70),
                        "win_rate": random.uniform(50, 65),
                        "signal_count": random.randint(10, 50),
                    },
                    {
                        "name": "Random Forest",
                        "accuracy": random.uniform(55, 70),
                        "win_rate": random.uniform(50, 65),
                        "signal_count": random.randint(10, 50),
                    },
                    {
                        "name": "XGBoost",
                        "accuracy": random.uniform(55, 70),
                        "win_rate": random.uniform(50, 65),
                        "signal_count": random.randint(10, 50),
                    },
                ]
            }

    def add_log(self, message: str, log_type: str = "info"):
        """Add log entry."""
        self._log_buffer.append(
            {
                "time": datetime.now().isoformat(),
                "message": message,
                "type": log_type,
            }
        )


# =============================================================================
# FLASK APP
# =============================================================================

app = Flask(__name__)
app.config["SECRET_KEY"] = "vn-trading-dashboard-secret"
socketio = SocketIO(app, cors_allowed_origins="*")

data_collector = DashboardDataCollector()
update_thread = None
stop_updates = threading.Event()


@app.route("/")
def dashboard():
    """Serve main dashboard."""
    return render_template_string(DASHBOARD_HTML)


@app.route("/api/status")
def api_status():
    """API status endpoint."""
    return jsonify(
        {
            "status": "running",
            "timestamp": datetime.now().isoformat(),
        }
    )


@app.route("/api/portfolio")
def api_portfolio():
    """API portfolio endpoint."""
    return jsonify(data_collector.get_portfolio_data())


@app.route("/api/positions")
def api_positions():
    """API positions endpoint."""
    return jsonify(data_collector.get_positions_data())


@socketio.on("connect")
def handle_connect():
    """Handle client connection."""
    logger.info("Client connected to dashboard")
    emit("log_event", {"message": "Connected to server", "type": "success"})


@socketio.on("disconnect")
def handle_disconnect():
    """Handle client disconnection."""
    logger.info("Client disconnected from dashboard")


@socketio.on("request_initial_data")
def handle_initial_data():
    """Send initial data to client."""
    emit("portfolio_update", data_collector.get_portfolio_data())
    emit("positions_update", data_collector.get_positions_data())
    emit("signals_update", data_collector.get_signals_data())
    emit("foreign_flow_update", data_collector.get_foreign_flow_data())
    emit("risk_update", data_collector.get_risk_data())
    emit("circuit_breaker_update", data_collector.get_circuit_breaker_data())
    emit("ml_performance_update", data_collector.get_ml_performance_data())


def background_updates():
    """Background thread for pushing updates."""
    while not stop_updates.is_set():
        try:
            socketio.emit("portfolio_update", data_collector.get_portfolio_data())
            socketio.emit("positions_update", data_collector.get_positions_data())
            socketio.emit("signals_update", data_collector.get_signals_data())
            socketio.emit("foreign_flow_update", data_collector.get_foreign_flow_data())
            socketio.emit("risk_update", data_collector.get_risk_data())
            socketio.emit("circuit_breaker_update", data_collector.get_circuit_breaker_data())
            socketio.emit("ml_performance_update", data_collector.get_ml_performance_data())
        except Exception as e:
            logger.error(f"Error in background updates: {e}")

        stop_updates.wait(timeout=2.0)  # Update every 2 seconds


def start_dashboard(
    host: str = "0.0.0.0",
    port: int = 8050,
    debug: bool = False,
):
    """
    Start the trading dashboard.

    Args:
        host: Host to bind to
        port: Port to run on
        debug: Enable debug mode
    """
    global update_thread

    stop_updates.clear()

    # Start background update thread
    update_thread = threading.Thread(target=background_updates, daemon=True)
    update_thread.start()

    logger.info(f"🚀 Starting Trading Dashboard on http://{host}:{port}")

    socketio.run(app, host=host, port=port, debug=debug)


def stop_dashboard():
    """Stop the dashboard."""
    stop_updates.set()
    logger.info("Dashboard stopped")


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    print("\n" + "=" * 60)
    print("🇻🇳 VIETNAM TRADING DASHBOARD")
    print("=" * 60)
    print("\nStarting dashboard server...")
    print("Open http://localhost:8050 in your browser")
    print("\nPress Ctrl+C to stop\n")

    start_dashboard(port=8050, debug=True)
