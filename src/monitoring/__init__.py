# -*- coding: utf-8 -*-
"""
Monitoring Module - Performance tracking and validation

Components:
- execution_tracker: Track actual execution costs vs estimates
- ml_performance_validator: Validate ML model accuracy in production
- filter_performance: Track entry filter effectiveness
- performance: General performance monitoring

Author: Trading Bot Team
"""

from src.monitoring.execution_tracker import (
    ExecutionCostTracker,
    ExecutionRecord,
    get_execution_tracker,
)

from src.monitoring.ml_performance_validator import (
    MLPerformanceValidator,
    MLPrediction,
    ModelHealthStatus,
    get_ml_validator,
)

# Optional imports (may not exist in all installations)
try:
    from src.monitoring.filter_performance import (
        FilterPerformanceTracker,
        get_filter_performance_tracker,
    )
except ImportError:
    FilterPerformanceTracker = None
    get_filter_performance_tracker = None

try:
    from src.monitoring.performance import (
        PerformanceMonitor,
        get_performance_monitor,
    )
except ImportError:
    PerformanceMonitor = None
    get_performance_monitor = None

__all__ = [
    # Execution tracking
    "ExecutionCostTracker",
    "ExecutionRecord",
    "get_execution_tracker",
    # ML validation
    "MLPerformanceValidator",
    "MLPrediction",
    "ModelHealthStatus",
    "get_ml_validator",
    # Filter performance
    "FilterPerformanceTracker",
    "get_filter_performance_tracker",
    # General performance
    "PerformanceMonitor",
    "get_performance_monitor",
]
