# -*- coding: utf-8 -*-
"""
Enhanced Monitoring with Prometheus Metrics
"""
import time
import logging
from typing import Dict, Optional
from datetime import datetime
from prometheus_client import Counter, Histogram, Gauge, Info, generate_latest
from prometheus_client import CONTENT_TYPE_LATEST
import psutil
import os

logger = logging.getLogger(__name__)

# ============================================================================
# PROMETHEUS METRICS
# ============================================================================

# Counters
signals_generated = Counter(
    "trading_signals_generated_total",
    "Total number of trading signals generated",
    ["signal_type", "symbol"],
)

trades_executed = Counter(
    "trading_trades_executed_total",
    "Total number of trades executed",
    ["action", "symbol"],
)

errors_total = Counter("trading_errors_total", "Total number of errors", ["error_type"])

api_calls_total = Counter(
    "trading_api_calls_total", "Total number of API calls", ["api_name", "status"]
)

# Histograms
scan_duration = Histogram(
    "trading_scan_duration_seconds",
    "Time spent scanning for signals",
    buckets=[1, 5, 10, 30, 60, 120, 300],
)

ml_inference_duration = Histogram(
    "trading_ml_inference_duration_seconds",
    "Time spent on ML inference",
    buckets=[0.1, 0.5, 1, 2, 5],
)

# Gauges
portfolio_value = Gauge("trading_portfolio_value", "Current portfolio value in VND")

portfolio_pnl = Gauge("trading_portfolio_pnl", "Current portfolio P&L in VND")

portfolio_pnl_percent = Gauge(
    "trading_portfolio_pnl_percent", "Current portfolio P&L percentage"
)

active_positions = Gauge("trading_active_positions", "Number of active positions")

cash_available = Gauge("trading_cash_available", "Available cash in VND")

system_cpu_percent = Gauge("trading_system_cpu_percent", "System CPU usage percentage")

system_memory_percent = Gauge(
    "trading_system_memory_percent", "System memory usage percentage"
)

# Info
bot_info = Info("trading_bot", "Trading bot information")

# ============================================================================
# MONITORING CLASS
# ============================================================================


class EnhancedMonitor:
    """Enhanced monitoring with Prometheus metrics"""

    def __init__(self):
        self.start_time = time.time()
        self.last_scan_time = None
        self.last_error_time = None
        self.health_status = "healthy"

        # Set bot info
        bot_info.info(
            {
                "version": "1.0.0",
                "python_version": os.sys.version.split()[0],
                "start_time": datetime.now().isoformat(),
            }
        )

    # ========================================================================
    # SIGNAL TRACKING
    # ========================================================================

    def track_signal(self, signal_type: str, symbol: str):
        """Track signal generation"""
        signals_generated.labels(signal_type=signal_type, symbol=symbol).inc()

    def track_trade(self, action: str, symbol: str):
        """Track trade execution"""
        trades_executed.labels(action=action, symbol=symbol).inc()

    # ========================================================================
    # ERROR TRACKING
    # ========================================================================

    def track_error(self, error_type: str):
        """Track error occurrence"""
        errors_total.labels(error_type=error_type).inc()
        self.last_error_time = datetime.now()

    # ========================================================================
    # API TRACKING
    # ========================================================================

    def track_api_call(self, api_name: str, status: str):
        """Track API call"""
        api_calls_total.labels(api_name=api_name, status=status).inc()

    # ========================================================================
    # PERFORMANCE TRACKING
    # ========================================================================

    def track_scan_duration(self, duration: float):
        """Track scan duration"""
        scan_duration.observe(duration)
        self.last_scan_time = datetime.now()

    def track_ml_inference(self, duration: float):
        """Track ML inference duration"""
        ml_inference_duration.observe(duration)

    # ========================================================================
    # PORTFOLIO TRACKING
    # ========================================================================

    def update_portfolio_metrics(self, portfolio_data: Dict):
        """Update portfolio metrics"""
        portfolio_value.set(portfolio_data.get("total_value", 0))
        portfolio_pnl.set(portfolio_data.get("pnl", 0))
        portfolio_pnl_percent.set(portfolio_data.get("pnl_percent", 0))
        active_positions.set(portfolio_data.get("num_positions", 0))
        cash_available.set(portfolio_data.get("cash", 0))

    # ========================================================================
    # SYSTEM METRICS
    # ========================================================================

    def update_system_metrics(self):
        """Update system resource metrics"""
        try:
            cpu_percent = psutil.cpu_percent(interval=1)
            memory_percent = psutil.virtual_memory().percent

            system_cpu_percent.set(cpu_percent)
            system_memory_percent.set(memory_percent)
        except Exception as e:
            logger.error(f"Error updating system metrics: {e}")

    # ========================================================================
    # HEALTH CHECK
    # ========================================================================

    def get_health_status(self) -> Dict:
        """Get comprehensive health status"""
        uptime = time.time() - self.start_time

        # Check if last scan was recent (within 1 hour)
        scan_healthy = True
        if self.last_scan_time:
            time_since_scan = (datetime.now() - self.last_scan_time).total_seconds()
            scan_healthy = time_since_scan < 3600

        # Check if errors are recent
        error_healthy = True
        if self.last_error_time:
            time_since_error = (datetime.now() - self.last_error_time).total_seconds()
            error_healthy = time_since_error > 300  # No errors in last 5 min

        # Overall health
        if scan_healthy and error_healthy:
            self.health_status = "healthy"
        elif scan_healthy or error_healthy:
            self.health_status = "degraded"
        else:
            self.health_status = "unhealthy"

        return {
            "status": self.health_status,
            "uptime_seconds": uptime,
            "last_scan": (
                self.last_scan_time.isoformat() if self.last_scan_time else None
            ),
            "last_error": (
                self.last_error_time.isoformat() if self.last_error_time else None
            ),
            "checks": {"scan_recent": scan_healthy, "no_recent_errors": error_healthy},
        }

    # ========================================================================
    # METRICS EXPORT
    # ========================================================================

    def export_metrics(self) -> bytes:
        """Export Prometheus metrics"""
        return generate_latest()


# Singleton
_monitor = None


def get_enhanced_monitor() -> EnhancedMonitor:
    """Get enhanced monitor singleton"""
    global _monitor
    if _monitor is None:
        _monitor = EnhancedMonitor()
    return _monitor


# ============================================================================
# CONTEXT MANAGERS FOR TIMING
# ============================================================================


class ScanTimer:
    """Context manager for timing scans"""

    def __init__(self, monitor: EnhancedMonitor):
        self.monitor = monitor
        self.start_time = None

    def __enter__(self):
        self.start_time = time.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        duration = time.time() - self.start_time
        self.monitor.track_scan_duration(duration)


class MLTimer:
    """Context manager for timing ML inference"""

    def __init__(self, monitor: EnhancedMonitor):
        self.monitor = monitor
        self.start_time = None

    def __enter__(self):
        self.start_time = time.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        duration = time.time() - self.start_time
        self.monitor.track_ml_inference(duration)


if __name__ == "__main__":
    # Test monitoring
    monitor = get_enhanced_monitor()

    # Track some metrics
    monitor.track_signal("BUY", "VNM")
    monitor.track_trade("BUY", "VNM")
    monitor.track_api_call("TCBS", "success")

    # Update portfolio
    monitor.update_portfolio_metrics(
        {
            "total_value": 105_000_000,
            "pnl": 5_000_000,
            "pnl_percent": 5.0,
            "num_positions": 3,
            "cash": 40_000_000,
        }
    )

    # Update system metrics
    monitor.update_system_metrics()

    # Get health
    health = monitor.get_health_status()
    print(f"Health: {health}")

    # Export metrics
    metrics = monitor.export_metrics()
    print(f"\nMetrics exported: {len(metrics)} bytes")
