# -*- coding: utf-8 -*-
"""
Prometheus Metrics Exporter for Trading Bot

Exports trading metrics to Prometheus for Grafana dashboards:
- Trade performance metrics
- Position metrics
- Risk metrics
- System health metrics

Author: Trading Bot Team
Version: 1.0.0
"""

import logging
import time
from typing import Dict, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# Try to import prometheus_client, fallback to mock if not available
try:
    from prometheus_client import (
        Counter,
        Gauge,
        Histogram,
        Summary,
        Info,
        start_http_server,
        CollectorRegistry,
        generate_latest,
        CONTENT_TYPE_LATEST,
    )

    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False
    logger.warning("prometheus_client not installed. Metrics will be disabled.")


@dataclass
class MetricLabels:
    """Standard metric labels"""

    symbol: str = ""
    side: str = ""  # BUY/SELL
    strategy: str = ""
    broker: str = ""
    market_regime: str = ""


class TradingMetrics:
    """
    Trading metrics for Prometheus

    Usage:
        metrics = TradingMetrics()
        metrics.start_server(port=8000)

        # Record trade
        metrics.record_trade("VNM", "BUY", 100, 85000)

        # Update position
        metrics.update_position("VNM", 100, 85000, 2.5)

        # Update portfolio
        metrics.update_portfolio(
            total_value=1_000_000_000,
            cash=300_000_000,
            unrealized_pnl=50_000_000
        )
    """

    def __init__(self, prefix: str = "trading_bot"):
        self.prefix = prefix
        self._registry = CollectorRegistry() if PROMETHEUS_AVAILABLE else None
        self._server_started = False

        if PROMETHEUS_AVAILABLE:
            self._init_metrics()

    def _init_metrics(self):
        """Initialize all Prometheus metrics"""

        # ============================================================
        # Trade Metrics
        # ============================================================

        self.trades_total = Counter(
            f"{self.prefix}_trades_total",
            "Total number of trades executed",
            ["symbol", "side", "strategy"],
            registry=self._registry,
        )

        self.trade_value_total = Counter(
            f"{self.prefix}_trade_value_total",
            "Total trade value in VND",
            ["symbol", "side"],
            registry=self._registry,
        )

        self.trade_pnl_total = Counter(
            f"{self.prefix}_trade_pnl_total",
            "Total realized P&L in VND",
            ["symbol"],
            registry=self._registry,
        )

        self.winning_trades = Counter(
            f"{self.prefix}_winning_trades_total",
            "Total winning trades",
            ["symbol", "strategy"],
            registry=self._registry,
        )

        self.losing_trades = Counter(
            f"{self.prefix}_losing_trades_total",
            "Total losing trades",
            ["symbol", "strategy"],
            registry=self._registry,
        )

        self.trade_duration = Histogram(
            f"{self.prefix}_trade_duration_hours",
            "Trade holding duration in hours",
            ["symbol", "strategy"],
            buckets=[1, 4, 8, 24, 48, 72, 168, 336],  # Up to 2 weeks
            registry=self._registry,
        )

        self.trade_return_pct = Histogram(
            f"{self.prefix}_trade_return_percent",
            "Trade return percentage",
            ["symbol", "strategy"],
            buckets=[-10, -5, -3, -2, -1, 0, 1, 2, 3, 5, 10, 20],
            registry=self._registry,
        )

        # ============================================================
        # Position Metrics
        # ============================================================

        self.position_count = Gauge(
            f"{self.prefix}_positions_count",
            "Current number of open positions",
            registry=self._registry,
        )

        self.position_value = Gauge(
            f"{self.prefix}_position_value",
            "Current position market value in VND",
            ["symbol"],
            registry=self._registry,
        )

        self.position_pnl_pct = Gauge(
            f"{self.prefix}_position_pnl_percent",
            "Current position P&L percentage",
            ["symbol"],
            registry=self._registry,
        )

        self.position_days_held = Gauge(
            f"{self.prefix}_position_days_held",
            "Days position has been held",
            ["symbol"],
            registry=self._registry,
        )

        # ============================================================
        # Portfolio Metrics
        # ============================================================

        self.portfolio_value = Gauge(
            f"{self.prefix}_portfolio_value",
            "Total portfolio value in VND",
            registry=self._registry,
        )

        self.portfolio_cash = Gauge(
            f"{self.prefix}_portfolio_cash",
            "Available cash in VND",
            registry=self._registry,
        )

        self.portfolio_unrealized_pnl = Gauge(
            f"{self.prefix}_portfolio_unrealized_pnl",
            "Total unrealized P&L in VND",
            registry=self._registry,
        )

        self.portfolio_realized_pnl = Gauge(
            f"{self.prefix}_portfolio_realized_pnl",
            "Total realized P&L in VND",
            registry=self._registry,
        )

        self.portfolio_drawdown = Gauge(
            f"{self.prefix}_portfolio_drawdown_percent",
            "Current drawdown from peak",
            registry=self._registry,
        )

        self.portfolio_exposure = Gauge(
            f"{self.prefix}_portfolio_exposure_percent",
            "Portfolio exposure percentage",
            registry=self._registry,
        )

        # ============================================================
        # Risk Metrics
        # ============================================================

        self.daily_pnl = Gauge(
            f"{self.prefix}_daily_pnl",
            "Today's P&L in VND",
            registry=self._registry,
        )

        self.daily_trades = Gauge(
            f"{self.prefix}_daily_trades_count",
            "Number of trades today",
            registry=self._registry,
        )

        self.consecutive_losses = Gauge(
            f"{self.prefix}_consecutive_losses",
            "Current consecutive losing trades",
            registry=self._registry,
        )

        self.circuit_breaker_status = Gauge(
            f"{self.prefix}_circuit_breaker_active",
            "Circuit breaker status (1=active, 0=inactive)",
            registry=self._registry,
        )

        self.risk_per_trade = Gauge(
            f"{self.prefix}_risk_per_trade_percent",
            "Current risk per trade setting",
            registry=self._registry,
        )

        # ============================================================
        # Market Metrics
        # ============================================================

        self.vnindex_value = Gauge(
            f"{self.prefix}_vnindex_value",
            "Current VNINDEX value",
            registry=self._registry,
        )

        self.vnindex_change = Gauge(
            f"{self.prefix}_vnindex_change_percent",
            "VNINDEX change from open",
            registry=self._registry,
        )

        self.market_regime = Info(
            f"{self.prefix}_market_regime",
            "Current market regime",
            registry=self._registry,
        )

        # ============================================================
        # Signal Metrics
        # ============================================================

        self.signals_generated = Counter(
            f"{self.prefix}_signals_generated_total",
            "Total entry signals generated",
            ["symbol", "signal_type"],
            registry=self._registry,
        )

        self.signal_confidence = Histogram(
            f"{self.prefix}_signal_confidence",
            "Signal confidence distribution",
            buckets=[30, 40, 50, 60, 70, 80, 90, 100],
            registry=self._registry,
        )

        self.signals_rejected = Counter(
            f"{self.prefix}_signals_rejected_total",
            "Signals rejected by filters",
            ["reason"],
            registry=self._registry,
        )

        # ============================================================
        # System Metrics
        # ============================================================

        self.api_requests = Counter(
            f"{self.prefix}_api_requests_total",
            "Total API requests made",
            ["endpoint", "status"],
            registry=self._registry,
        )

        self.api_latency = Histogram(
            f"{self.prefix}_api_latency_seconds",
            "API request latency",
            ["endpoint"],
            buckets=[0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0],
            registry=self._registry,
        )

        self.errors_total = Counter(
            f"{self.prefix}_errors_total",
            "Total errors by type",
            ["error_type"],
            registry=self._registry,
        )

        self.bot_status = Info(
            f"{self.prefix}_bot_info",
            "Bot version and status information",
            registry=self._registry,
        )

    def start_server(self, port: int = 8000) -> bool:
        """Start Prometheus HTTP server"""
        if not PROMETHEUS_AVAILABLE:
            logger.warning("Cannot start Prometheus server - prometheus_client not installed")
            return False

        if self._server_started:
            logger.warning("Prometheus server already started")
            return True

        try:
            start_http_server(port, registry=self._registry)
            self._server_started = True
            logger.info(f"✅ Prometheus metrics server started on port {port}")
            return True
        except Exception as e:
            logger.error(f"Failed to start Prometheus server: {e}")
            return False

    def get_metrics(self) -> bytes:
        """Get metrics in Prometheus format"""
        if not PROMETHEUS_AVAILABLE:
            return b""
        return generate_latest(self._registry)

    # ============================================================
    # Trade Recording Methods
    # ============================================================

    def record_trade(
        self,
        symbol: str,
        side: str,
        quantity: int,
        price: float,
        strategy: str = "default",
        pnl: Optional[float] = None,
        pnl_pct: Optional[float] = None,
        duration_hours: Optional[float] = None,
    ):
        """Record a completed trade"""
        if not PROMETHEUS_AVAILABLE:
            return

        self.trades_total.labels(symbol=symbol, side=side, strategy=strategy).inc()
        self.trade_value_total.labels(symbol=symbol, side=side).inc(quantity * price)

        if pnl is not None:
            self.trade_pnl_total.labels(symbol=symbol).inc(pnl)

            if pnl > 0:
                self.winning_trades.labels(symbol=symbol, strategy=strategy).inc()
            else:
                self.losing_trades.labels(symbol=symbol, strategy=strategy).inc()

        if pnl_pct is not None:
            self.trade_return_pct.labels(symbol=symbol, strategy=strategy).observe(pnl_pct)

        if duration_hours is not None:
            self.trade_duration.labels(symbol=symbol, strategy=strategy).observe(duration_hours)

    def update_position(
        self,
        symbol: str,
        quantity: int,
        market_price: float,
        pnl_pct: float,
        days_held: int = 0,
    ):
        """Update position metrics"""
        if not PROMETHEUS_AVAILABLE:
            return

        self.position_value.labels(symbol=symbol).set(quantity * market_price)
        self.position_pnl_pct.labels(symbol=symbol).set(pnl_pct)
        self.position_days_held.labels(symbol=symbol).set(days_held)

    def clear_position(self, symbol: str):
        """Clear position metrics when closed"""
        if not PROMETHEUS_AVAILABLE:
            return

        self.position_value.labels(symbol=symbol).set(0)
        self.position_pnl_pct.labels(symbol=symbol).set(0)
        self.position_days_held.labels(symbol=symbol).set(0)

    def update_portfolio(
        self,
        total_value: float,
        cash: float,
        unrealized_pnl: float,
        realized_pnl: float = 0,
        drawdown_pct: float = 0,
        exposure_pct: float = 0,
        position_count: int = 0,
    ):
        """Update portfolio metrics"""
        if not PROMETHEUS_AVAILABLE:
            return

        self.portfolio_value.set(total_value)
        self.portfolio_cash.set(cash)
        self.portfolio_unrealized_pnl.set(unrealized_pnl)
        self.portfolio_realized_pnl.set(realized_pnl)
        self.portfolio_drawdown.set(drawdown_pct)
        self.portfolio_exposure.set(exposure_pct)
        self.position_count.set(position_count)

    def update_risk_metrics(
        self,
        daily_pnl: float,
        daily_trades: int,
        consecutive_losses: int,
        circuit_breaker_active: bool,
        risk_per_trade: float,
    ):
        """Update risk metrics"""
        if not PROMETHEUS_AVAILABLE:
            return

        self.daily_pnl.set(daily_pnl)
        self.daily_trades.set(daily_trades)
        self.consecutive_losses.set(consecutive_losses)
        self.circuit_breaker_status.set(1 if circuit_breaker_active else 0)
        self.risk_per_trade.set(risk_per_trade)

    def update_market(
        self,
        vnindex: float,
        vnindex_change_pct: float,
        regime: str,
    ):
        """Update market metrics"""
        if not PROMETHEUS_AVAILABLE:
            return

        self.vnindex_value.set(vnindex)
        self.vnindex_change.set(vnindex_change_pct)
        self.market_regime.info({"regime": regime})

    def record_signal(
        self,
        symbol: str,
        signal_type: str,
        confidence: float,
        rejected: bool = False,
        reject_reason: str = "",
    ):
        """Record entry signal"""
        if not PROMETHEUS_AVAILABLE:
            return

        self.signals_generated.labels(symbol=symbol, signal_type=signal_type).inc()
        self.signal_confidence.observe(confidence)

        if rejected:
            self.signals_rejected.labels(reason=reject_reason).inc()

    def record_api_request(
        self,
        endpoint: str,
        status: str,
        latency: float,
    ):
        """Record API request metrics"""
        if not PROMETHEUS_AVAILABLE:
            return

        self.api_requests.labels(endpoint=endpoint, status=status).inc()
        self.api_latency.labels(endpoint=endpoint).observe(latency)

    def record_error(self, error_type: str):
        """Record error"""
        if not PROMETHEUS_AVAILABLE:
            return

        self.errors_total.labels(error_type=error_type).inc()

    def set_bot_info(self, version: str, status: str, start_time: str):
        """Set bot info"""
        if not PROMETHEUS_AVAILABLE:
            return

        self.bot_status.info(
            {
                "version": version,
                "status": status,
                "start_time": start_time,
            }
        )


# Singleton instance
_trading_metrics: Optional[TradingMetrics] = None


def get_trading_metrics() -> TradingMetrics:
    """Get singleton trading metrics instance"""
    global _trading_metrics
    if _trading_metrics is None:
        _trading_metrics = TradingMetrics()
    return _trading_metrics
