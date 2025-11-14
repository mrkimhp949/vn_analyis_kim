"""
Prometheus Metrics Exporter
Export trading bot metrics for Prometheus monitoring
"""

from prometheus_client import Counter, Gauge, Histogram, Info, generate_latest, REGISTRY
from typing import Dict, Optional
import time


# ============================================================================
# TRADING METRICS
# ============================================================================

# Portfolio metrics
portfolio_value = Gauge("portfolio_total_value", "Total portfolio value in VND")
portfolio_pnl = Gauge("portfolio_pnl", "Portfolio profit/loss in VND")
portfolio_pnl_percent = Gauge("portfolio_pnl_percent", "Portfolio P&L percentage")
num_positions = Gauge("portfolio_positions", "Number of active positions")

# Trading signals
signals_generated = Counter(
    "signals_generated_total", "Total trading signals generated", ["signal_type"]
)
signals_confidence = Histogram(
    "signal_confidence",
    "Signal confidence distribution",
    buckets=[50, 60, 70, 80, 90, 100],
)

# Trades
trades_executed = Counter("trades_executed_total", "Total trades executed", ["action"])
trade_pnl = Histogram(
    "trade_pnl",
    "Trade P&L distribution",
    buckets=[-1000000, -500000, 0, 500000, 1000000, 2000000],
)

# Paper trading
paper_trades = Counter("paper_trades_total", "Total paper trades", ["action"])
paper_pnl = Gauge("paper_trading_pnl", "Paper trading P&L")
paper_win_rate = Gauge("paper_trading_win_rate", "Paper trading win rate percentage")

# ============================================================================
# SYSTEM METRICS
# ============================================================================

# ML Models
ml_predictions = Counter("ml_predictions_total", "Total ML predictions made")
ml_prediction_time = Histogram("ml_prediction_seconds", "ML prediction latency")
ml_model_accuracy = Gauge("ml_model_accuracy", "ML model accuracy", ["model"])

# API
api_requests = Counter(
    "api_requests_total", "Total API requests", ["endpoint", "status"]
)
api_latency = Histogram("api_latency_seconds", "API request latency", ["endpoint"])

# Database
db_queries = Counter("db_queries_total", "Total database queries", ["operation"])
db_query_time = Histogram("db_query_seconds", "Database query latency")

# Cache
cache_hits = Counter("cache_hits_total", "Cache hit count")
cache_misses = Counter("cache_misses_total", "Cache miss count")

# Errors
errors_total = Counter("errors_total", "Total errors", ["component", "error_type"])

# ============================================================================
# RISK METRICS
# ============================================================================

max_drawdown = Gauge("risk_max_drawdown", "Maximum drawdown percentage")
sharpe_ratio = Gauge("risk_sharpe_ratio", "Sharpe ratio")
sortino_ratio = Gauge("risk_sortino_ratio", "Sortino ratio")
sector_exposure = Gauge(
    "risk_sector_exposure", "Sector exposure percentage", ["sector"]
)
correlation_risk = Gauge("risk_correlation", "Portfolio correlation risk")

# ============================================================================
# SYSTEM INFO
# ============================================================================

system_info = Info("trading_bot", "Trading bot information")
system_info.info(
    {"version": "1.0.0", "python_version": "3.11", "environment": "production"}
)

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================


def update_portfolio_metrics(portfolio_data: Dict) -> None:
    """Update portfolio metrics"""
    portfolio_value.set(portfolio_data.get("total_value", 0))
    portfolio_pnl.set(portfolio_data.get("pnl", 0))
    portfolio_pnl_percent.set(portfolio_data.get("pnl_percent", 0))
    num_positions.set(portfolio_data.get("num_positions", 0))


def record_signal(signal_type: str, confidence: float) -> None:
    """Record trading signal"""
    signals_generated.labels(signal_type=signal_type).inc()
    signals_confidence.observe(confidence)


def record_trade(action: str, pnl: Optional[float] = None) -> None:
    """Record executed trade"""
    trades_executed.labels(action=action).inc()
    if pnl is not None:
        trade_pnl.observe(pnl)


def record_paper_trade(action: str) -> None:
    """Record paper trade"""
    paper_trades.labels(action=action).inc()


def update_paper_trading_metrics(stats: Dict) -> None:
    """Update paper trading metrics"""
    paper_pnl.set(stats.get("total_pnl", 0))
    paper_win_rate.set(stats.get("win_rate", 0))


def record_ml_prediction(duration: float) -> None:
    """Record ML prediction"""
    ml_predictions.inc()
    ml_prediction_time.observe(duration)


def update_ml_accuracy(model_name: str, accuracy: float) -> None:
    """Update ML model accuracy"""
    ml_model_accuracy.labels(model=model_name).set(accuracy)


def record_api_request(endpoint: str, status: int, duration: float) -> None:
    """Record API request"""
    api_requests.labels(endpoint=endpoint, status=str(status)).inc()
    api_latency.labels(endpoint=endpoint).observe(duration)


def record_db_query(operation: str, duration: float) -> None:
    """Record database query"""
    db_queries.labels(operation=operation).inc()
    db_query_time.observe(duration)


def record_cache_access(hit: bool) -> None:
    """Record cache access"""
    if hit:
        cache_hits.inc()
    else:
        cache_misses.inc()


def record_error(component: str, error_type: str) -> None:
    """Record error"""
    errors_total.labels(component=component, error_type=error_type).inc()


def update_risk_metrics(metrics: Dict) -> None:
    """Update risk metrics"""
    max_drawdown.set(metrics.get("max_drawdown", 0))
    sharpe_ratio.set(metrics.get("sharpe_ratio", 0))
    sortino_ratio.set(metrics.get("sortino_ratio", 0))
    correlation_risk.set(metrics.get("correlation_risk", 0))


def update_sector_exposure(sector_exposures: Dict[str, float]) -> None:
    """Update sector exposure metrics"""
    for sector, exposure in sector_exposures.items():
        sector_exposure.labels(sector=sector).set(exposure)


def get_metrics() -> bytes:
    """
    Get Prometheus metrics in text format

    Returns:
        Metrics in Prometheus text format
    """
    return generate_latest(REGISTRY)


# ============================================================================
# METRICS COLLECTION
# ============================================================================


def collect_all_metrics() -> None:
    """Collect all current metrics from the trading bot"""
    try:
        # Portfolio metrics
        from portfolio_manager import get_portfolio_manager

        manager = get_portfolio_manager()
        portfolio = manager.get_portfolio_value()
        update_portfolio_metrics(portfolio)

    except Exception as e:
        record_error("portfolio", type(e).__name__)

    try:
        # Paper trading metrics
        from paper_trading import get_paper_account

        account = get_paper_account()
        stats = account.get_statistics()
        if stats:
            update_paper_trading_metrics(stats)

    except Exception as e:
        record_error("paper_trading", type(e).__name__)

    try:
        # ML model metrics
        from ml_models import MLPredictor

        predictor = MLPredictor()
        if predictor.rf_model is not None:
            update_ml_accuracy("random_forest", 0.75)  # Placeholder

    except Exception as e:
        record_error("ml_models", type(e).__name__)


# ============================================================================
# FASTAPI INTEGRATION
# ============================================================================


def add_metrics_endpoint(app):
    """
    Add Prometheus metrics endpoint to FastAPI app

    Usage:
        from prometheus_metrics import add_metrics_endpoint
        add_metrics_endpoint(app)

    Then access metrics at: http://localhost:8080/metrics
    """
    from fastapi import Response

    @app.get("/metrics")
    async def metrics():
        """Prometheus metrics endpoint"""
        collect_all_metrics()
        return Response(content=get_metrics(), media_type="text/plain")


if __name__ == "__main__":
    print("📊 Prometheus Metrics")
    print("=" * 70)

    # Collect and display current metrics
    collect_all_metrics()

    print("\nMetrics collected. Access via /metrics endpoint when running.")
    print("\nExample Prometheus config:")
    print(
        """
scrape_configs:
  - job_name: 'trading-bot'
    scrape_interval: 15s
    static_configs:
      - targets: ['localhost:8080']
    """
    )
