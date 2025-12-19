# -*- coding: utf-8 -*-
"""
Execution Integration - Integrate execution tracking with brokers

Wraps broker execution to automatically track:
- Actual slippage vs expected
- Commission costs
- Fill times
- Execution quality

Author: Trading Bot Team
Version: 1.0.0
"""

import logging
import time
from datetime import datetime
from functools import wraps
from typing import Any, Callable, Dict, Optional

logger = logging.getLogger(__name__)

# Import execution tracker
try:
    from src.monitoring.execution_tracker import get_execution_tracker

    EXECUTION_TRACKER_AVAILABLE = True
except ImportError:
    EXECUTION_TRACKER_AVAILABLE = False
    logger.warning("ExecutionTracker not available - execution tracking disabled")

# Import slippage estimation
try:
    from src.config.constants import get_order_impact_slippage, estimate_execution_cost

    SLIPPAGE_ESTIMATION_AVAILABLE = True
except ImportError:
    SLIPPAGE_ESTIMATION_AVAILABLE = False


def track_execution(
    symbol: str,
    order_type: str,
    side: str,
    expected_price: float,
    shares: int,
    avg_daily_volume: int = 0,
    session: str = "CONTINUOUS",
):
    """
    Decorator to track execution of broker order methods.

    Usage:
        @track_execution("VNM", "LIMIT", "BUY", 80000, 500)
        def place_order(...):
            ...

    Or use track_order_execution() function directly.
    """

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            if not EXECUTION_TRACKER_AVAILABLE:
                return func(*args, **kwargs)

            start_time = time.time()

            # Estimate expected slippage
            expected_slippage = 0.003  # Default 0.3%
            if SLIPPAGE_ESTIMATION_AVAILABLE and avg_daily_volume > 0:
                expected_slippage = get_order_impact_slippage(
                    symbol=symbol,
                    order_value=expected_price * shares,
                    avg_daily_volume=avg_daily_volume,
                    avg_price=expected_price,
                    is_market_order=(order_type.upper() == "MARKET"),
                )

            # Execute order
            result = func(*args, **kwargs)

            execution_time_ms = int((time.time() - start_time) * 1000)

            # Extract executed price from result
            executed_price = _extract_executed_price(result, expected_price)
            actual_commission = _extract_commission(result, expected_price * shares)

            # Record execution
            tracker = get_execution_tracker()
            tracker.record_execution(
                symbol=symbol,
                order_type=order_type,
                side=side,
                expected_price=expected_price,
                executed_price=executed_price,
                shares=shares,
                expected_slippage_pct=expected_slippage,
                expected_commission=expected_price * shares * 0.003,  # ~0.3% commission
                actual_commission=actual_commission,
                execution_time_ms=execution_time_ms,
                session=session,
                avg_daily_volume=avg_daily_volume,
            )

            return result

        return wrapper

    return decorator


def track_order_execution(
    symbol: str,
    order_type: str,
    side: str,
    expected_price: float,
    executed_price: float,
    shares: int,
    actual_commission: float,
    execution_time_ms: int = 0,
    avg_daily_volume: int = 0,
    session: str = "CONTINUOUS",
) -> Optional[Dict]:
    """
    Track an order execution directly (without decorator).

    Call this after order is filled to record execution metrics.

    Args:
        symbol: Stock symbol
        order_type: "MARKET" or "LIMIT"
        side: "BUY" or "SELL"
        expected_price: Price expected when order was placed
        executed_price: Actual fill price
        shares: Number of shares
        actual_commission: Actual commission charged
        execution_time_ms: Time to fill in milliseconds
        avg_daily_volume: Average daily volume
        session: Trading session

    Returns:
        ExecutionRecord dict or None if tracking unavailable
    """
    if not EXECUTION_TRACKER_AVAILABLE:
        logger.debug("Execution tracking not available")
        return None

    # Estimate expected slippage
    expected_slippage = 0.003  # Default
    if SLIPPAGE_ESTIMATION_AVAILABLE and avg_daily_volume > 0:
        expected_slippage = get_order_impact_slippage(
            symbol=symbol,
            order_value=expected_price * shares,
            avg_daily_volume=avg_daily_volume,
            avg_price=expected_price,
            is_market_order=(order_type.upper() == "MARKET"),
        )

    # Estimate expected commission
    expected_commission = expected_price * shares * 0.003  # ~0.3%

    tracker = get_execution_tracker()
    record = tracker.record_execution(
        symbol=symbol,
        order_type=order_type,
        side=side,
        expected_price=expected_price,
        executed_price=executed_price,
        shares=shares,
        expected_slippage_pct=expected_slippage,
        expected_commission=expected_commission,
        actual_commission=actual_commission,
        execution_time_ms=execution_time_ms,
        session=session,
        avg_daily_volume=avg_daily_volume,
    )

    return {
        "symbol": record.symbol,
        "side": record.side,
        "expected_slippage": record.expected_slippage_pct,
        "actual_slippage": record.actual_slippage_pct,
        "slippage_deviation": record.slippage_deviation,
        "total_cost_pct": record.total_cost_pct,
    }


def get_calibrated_slippage_for_order(
    symbol: str,
    order_value: float,
    avg_daily_volume: int,
    avg_price: float,
    order_type: str = "LIMIT",
) -> float:
    """
    Get calibrated slippage estimate for an order.

    Uses historical execution data if available, otherwise falls back
    to model-based estimation.

    Args:
        symbol: Stock symbol
        order_value: Order value in VND
        avg_daily_volume: Average daily volume
        avg_price: Average price
        order_type: "MARKET" or "LIMIT"

    Returns:
        Estimated slippage as decimal (e.g., 0.003 = 0.3%)
    """
    # Try to get calibrated slippage from historical data
    if EXECUTION_TRACKER_AVAILABLE:
        tracker = get_execution_tracker()
        calibrated = tracker.get_calibrated_slippage(
            symbol,
            order_type,
            fallback=None,
        )
        if calibrated is not None:
            return calibrated

    # Fall back to model-based estimation
    if SLIPPAGE_ESTIMATION_AVAILABLE:
        return get_order_impact_slippage(
            symbol=symbol,
            order_value=order_value,
            avg_daily_volume=avg_daily_volume,
            avg_price=avg_price,
            is_market_order=(order_type.upper() == "MARKET"),
        )

    # Default fallback
    if order_type.upper() == "MARKET":
        return 0.004  # 0.4% for market orders
    else:
        return 0.002  # 0.2% for limit orders


def get_execution_quality_report(symbol: Optional[str] = None) -> Dict:
    """
    Get execution quality report.

    Args:
        symbol: Optional symbol to filter by

    Returns:
        Dict with execution quality metrics
    """
    if not EXECUTION_TRACKER_AVAILABLE:
        return {"error": "Execution tracking not available"}

    tracker = get_execution_tracker()

    if symbol:
        score, desc = tracker.get_execution_quality_score(symbol)
        return {
            "symbol": symbol,
            "quality_score": score,
            "description": desc,
            "calibrated_slippage": tracker.get_calibrated_slippage(symbol),
        }
    else:
        # Get recommendations for all symbols
        recommendations = tracker.get_calibration_recommendations()
        daily_report = tracker.get_daily_report()

        return {
            "daily_report": daily_report,
            "calibration_recommendations": recommendations,
            "symbols_with_issues": len(
                [r for r in recommendations.values() if abs(r.get("deviation", 0)) > 0.003]
            ),
        }


def _extract_executed_price(result: Any, default: float) -> float:
    """Extract executed price from broker result."""
    if result is None:
        return default

    if isinstance(result, dict):
        return result.get("executed_price", result.get("price", default))

    if hasattr(result, "executed_price"):
        return result.executed_price

    if hasattr(result, "price"):
        return result.price

    return default


def _extract_commission(result: Any, order_value: float) -> float:
    """Extract commission from broker result."""
    if result is None:
        return order_value * 0.003  # Default 0.3%

    if isinstance(result, dict):
        return result.get("commission", result.get("fee", order_value * 0.003))

    if hasattr(result, "commission"):
        return result.commission

    if hasattr(result, "fee"):
        return result.fee

    return order_value * 0.003


# Integration with ML validator
def track_ml_prediction(
    symbol: str,
    signal: str,
    confidence: float,
    entry_price: float,
) -> Optional[str]:
    """
    Track ML prediction for performance validation.

    Args:
        symbol: Stock symbol
        signal: "BUY", "SELL", or "HOLD"
        confidence: Model confidence 0-100
        entry_price: Entry price

    Returns:
        prediction_id for tracking outcome, or None if unavailable
    """
    try:
        from src.monitoring.ml_performance_validator import get_ml_validator

        validator = get_ml_validator()
        return validator.record_prediction(
            symbol=symbol,
            signal=signal,
            confidence=confidence,
            entry_price=entry_price,
        )
    except ImportError:
        logger.debug("ML validator not available")
        return None


def track_ml_outcome(
    prediction_id: str,
    exit_price: float,
    exit_reason: str = "NORMAL",
) -> Optional[Dict]:
    """
    Track ML prediction outcome.

    Args:
        prediction_id: ID from track_ml_prediction
        exit_price: Exit price
        exit_reason: Reason for exit

    Returns:
        Outcome dict or None if unavailable
    """
    try:
        from src.monitoring.ml_performance_validator import get_ml_validator

        validator = get_ml_validator()
        result = validator.record_outcome(prediction_id, exit_price, exit_reason)

        if result:
            return {
                "symbol": result.symbol,
                "outcome": result.outcome,
                "return_pct": result.actual_return_pct,
                "holding_days": result.holding_days,
            }
        return None
    except ImportError:
        logger.debug("ML validator not available")
        return None


def get_ml_health_check() -> Dict:
    """
    Get ML model health check.

    Returns:
        Dict with model health status
    """
    try:
        from src.monitoring.ml_performance_validator import get_ml_validator

        validator = get_ml_validator()
        health = validator.get_model_health()

        return {
            "is_healthy": health.is_healthy,
            "accuracy_status": health.accuracy_status,
            "ev_status": health.ev_status,
            "calibration_status": health.calibration_status,
            "recommendation": health.recommendation,
            "details": health.details,
        }
    except ImportError:
        return {"error": "ML validator not available"}
