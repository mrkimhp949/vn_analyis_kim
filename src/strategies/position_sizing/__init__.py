# -*- coding: utf-8 -*-
"""
Position Sizing Package

This package provides position sizing functionality for Vietnam stock market trading.

Modules:
- constants: Position sizing constants and configuration
- protocols: Protocol definitions for dependency injection
- models: Data classes for position sizing results
- cache: Correlation cache implementation
- sizer: Main EnhancedPositionSizer class

Usage:
    from src.strategies.position_sizing import EnhancedPositionSizer, EnhancedPositionSize

    # Or import specific components:
    from src.strategies.position_sizing.constants import PositionSizingConstants
    from src.strategies.position_sizing.models import MarketRegimeInfo
"""

from .cache import CorrelationCache
from .constants import PositionSizingConstants
from .models import EnhancedPositionSize, MarketRegimeInfo, PositionSize
from .protocols import CircuitBreakerProtocol, DataLoaderProtocol, RegimeDetectorProtocol
from .sizer import EnhancedPositionSizer

__all__ = [
    # Constants
    "PositionSizingConstants",
    # Protocols
    "DataLoaderProtocol",
    "RegimeDetectorProtocol",
    "CircuitBreakerProtocol",
    # Models
    "EnhancedPositionSize",
    "MarketRegimeInfo",
    "PositionSize",
    # Cache
    "CorrelationCache",
    # Main class
    "EnhancedPositionSizer",
]
