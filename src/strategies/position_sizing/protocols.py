# -*- coding: utf-8 -*-
"""
Position Sizing Protocols

Protocol definitions for dependency injection interfaces.
"""

from typing import Optional, Protocol

import pandas as pd


class DataLoaderProtocol(Protocol):
    """Protocol for data loading dependency."""

    def __call__(self, symbol: str, lookback: int = 60) -> Optional[pd.DataFrame]: ...


class RegimeDetectorProtocol(Protocol):
    """Protocol for market regime detection."""

    def __call__(self, df: pd.DataFrame) -> object: ...


class CircuitBreakerProtocol(Protocol):
    """Protocol for circuit breaker."""

    def is_caution_mode(self) -> bool: ...
