# -*- coding: utf-8 -*-
"""
Data Staleness Manager - Consistent Data Quality Handling

Provides a unified approach to handle data staleness across all data sources.
This ensures consistent behavior when data is outdated:
- Reduce score/weight by configurable factor
- Log warnings about stale data
- Track data age for monitoring

Author: Trading Bot Team
Version: 1.0.0
"""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, Optional, Tuple, Any
from threading import RLock
from enum import Enum

logger = logging.getLogger(__name__)


# =============================================================================
# CONSTANTS
# =============================================================================


# Default staleness thresholds by data type (in minutes)
class DataFreshness(Enum):
    """Data freshness levels."""

    FRESH = "fresh"  # Data is current
    SLIGHTLY_STALE = "slightly_stale"  # Data is a bit old
    STALE = "stale"  # Data is old, reduce weight
    VERY_STALE = "very_stale"  # Data is very old, minimal weight
    EXPIRED = "expired"  # Data should not be used


# Weight reduction factors for different freshness levels
FRESHNESS_WEIGHT_FACTORS = {
    DataFreshness.FRESH: 1.0,
    DataFreshness.SLIGHTLY_STALE: 0.85,
    DataFreshness.STALE: 0.5,
    DataFreshness.VERY_STALE: 0.2,
    DataFreshness.EXPIRED: 0.0,
}


@dataclass
class StalenessConfig:
    """Configuration for data staleness handling."""

    # Threshold in minutes
    fresh_threshold_minutes: int = 5  # Under this = FRESH
    slightly_stale_minutes: int = 15  # Under this = SLIGHTLY_STALE
    stale_threshold_minutes: int = 30  # Under this = STALE
    very_stale_threshold_minutes: int = 60  # Under this = VERY_STALE
    # Above very_stale = EXPIRED

    # Weight reduction factors (override defaults if needed)
    fresh_weight: float = 1.0
    slightly_stale_weight: float = 0.85
    stale_weight: float = 0.5
    very_stale_weight: float = 0.2
    expired_weight: float = 0.0

    # Behavior flags
    log_stale_warnings: bool = True
    track_staleness_metrics: bool = True

    def get_weight_for_freshness(self, freshness: DataFreshness) -> float:
        """Get weight factor for freshness level."""
        return {
            DataFreshness.FRESH: self.fresh_weight,
            DataFreshness.SLIGHTLY_STALE: self.slightly_stale_weight,
            DataFreshness.STALE: self.stale_weight,
            DataFreshness.VERY_STALE: self.very_stale_weight,
            DataFreshness.EXPIRED: self.expired_weight,
        }.get(freshness, 0.5)


# Predefined configs for different data types
STALENESS_CONFIGS = {
    # Real-time data needs to be very fresh
    "realtime": StalenessConfig(
        fresh_threshold_minutes=1,
        slightly_stale_minutes=3,
        stale_threshold_minutes=5,
        very_stale_threshold_minutes=10,
    ),
    # Market breadth can be slightly older
    "market_breadth": StalenessConfig(
        fresh_threshold_minutes=5,
        slightly_stale_minutes=15,
        stale_threshold_minutes=30,
        very_stale_threshold_minutes=60,
    ),
    # Foreign flow data - daily data, longer thresholds
    "foreign_flow": StalenessConfig(
        fresh_threshold_minutes=15,
        slightly_stale_minutes=30,
        stale_threshold_minutes=60,
        very_stale_threshold_minutes=120,
    ),
    # Sentiment data - can be a bit older
    "sentiment": StalenessConfig(
        fresh_threshold_minutes=15,
        slightly_stale_minutes=30,
        stale_threshold_minutes=60,
        very_stale_threshold_minutes=120,
    ),
    # Order book - needs to be fresh
    "orderbook": StalenessConfig(
        fresh_threshold_minutes=1,
        slightly_stale_minutes=2,
        stale_threshold_minutes=5,
        very_stale_threshold_minutes=10,
    ),
    # Fundamental data - can be quite old
    "fundamental": StalenessConfig(
        fresh_threshold_minutes=60,
        slightly_stale_minutes=240,
        stale_threshold_minutes=480,
        very_stale_threshold_minutes=1440,  # 1 day
    ),
}


# =============================================================================
# DATA STALENESS MIXIN
# =============================================================================


class DataStalenessMixin:
    """
    Mixin class providing consistent data staleness handling.

    Usage:
        class MyDataProvider(DataStalenessMixin):
            def __init__(self):
                super().__init__()
                self._init_staleness("realtime")  # Use realtime config

            def get_data(self):
                data = self._fetch_data()
                self._update_cache_timestamp()  # Mark data as fresh
                return data

            def get_adjusted_score(self, raw_score: float) -> float:
                return self._apply_staleness_weight(raw_score)
    """

    def _init_staleness(
        self,
        data_type: str = "market_breadth",
        custom_config: Optional[StalenessConfig] = None,
    ):
        """
        Initialize staleness tracking.

        Args:
            data_type: Type of data for preset config
            custom_config: Custom config to override preset
        """
        self._staleness_config = custom_config or STALENESS_CONFIGS.get(
            data_type, StalenessConfig()
        )
        self._cache_time: Optional[datetime] = None
        self._staleness_lock = RLock()
        self._staleness_metrics = {
            "checks": 0,
            "stale_count": 0,
            "last_freshness": None,
        }

    def _update_cache_timestamp(self, timestamp: Optional[datetime] = None):
        """Update cache timestamp to current time or specified time."""
        with self._staleness_lock:
            self._cache_time = timestamp or datetime.now()

    def get_data_age_minutes(self) -> float:
        """Get age of cached data in minutes."""
        if self._cache_time is None:
            return float("inf")

        age = datetime.now() - self._cache_time
        return age.total_seconds() / 60

    def get_data_freshness(self) -> DataFreshness:
        """Get freshness level of cached data."""
        age = self.get_data_age_minutes()
        config = self._staleness_config

        if age < config.fresh_threshold_minutes:
            return DataFreshness.FRESH
        elif age < config.slightly_stale_minutes:
            return DataFreshness.SLIGHTLY_STALE
        elif age < config.stale_threshold_minutes:
            return DataFreshness.STALE
        elif age < config.very_stale_threshold_minutes:
            return DataFreshness.VERY_STALE
        else:
            return DataFreshness.EXPIRED

    def is_data_stale(self, max_delay_minutes: Optional[int] = None) -> bool:
        """
        Check if cached data is stale.

        Args:
            max_delay_minutes: Override threshold (uses stale_threshold_minutes if None)

        Returns:
            True if data is stale (older than threshold)
        """
        threshold = max_delay_minutes or self._staleness_config.stale_threshold_minutes
        age = self.get_data_age_minutes()

        is_stale = age > threshold

        # Track metrics
        if hasattr(self, "_staleness_metrics"):
            self._staleness_metrics["checks"] += 1
            if is_stale:
                self._staleness_metrics["stale_count"] += 1
            self._staleness_metrics["last_freshness"] = self.get_data_freshness().value

        return is_stale

    def is_data_expired(self) -> bool:
        """Check if data is expired and should not be used."""
        return self.get_data_freshness() == DataFreshness.EXPIRED

    def get_staleness_weight(self) -> float:
        """
        Get weight factor based on data freshness.

        Returns:
            Weight factor 0.0-1.0
        """
        freshness = self.get_data_freshness()
        weight = self._staleness_config.get_weight_for_freshness(freshness)

        # Log warning if stale
        if freshness in (DataFreshness.STALE, DataFreshness.VERY_STALE):
            if self._staleness_config.log_stale_warnings:
                logger.warning(
                    f"⚠️ Data staleness: {self.__class__.__name__} is {freshness.value} "
                    f"(age={self.get_data_age_minutes():.1f}m, weight={weight:.2f})"
                )

        return weight

    def _apply_staleness_weight(
        self,
        raw_score: float,
        min_score: float = 0.0,
    ) -> float:
        """
        Apply staleness weight to a raw score.

        Args:
            raw_score: Original score
            min_score: Minimum score to return

        Returns:
            Adjusted score based on data freshness
        """
        weight = self.get_staleness_weight()
        adjusted = raw_score * weight
        return max(min_score, adjusted)

    def get_staleness_status(self) -> Dict[str, Any]:
        """Get comprehensive staleness status."""
        freshness = self.get_data_freshness()
        return {
            "freshness": freshness.value,
            "age_minutes": self.get_data_age_minutes(),
            "weight_factor": self.get_staleness_weight(),
            "is_stale": self.is_data_stale(),
            "is_expired": self.is_data_expired(),
            "cache_time": self._cache_time.isoformat() if self._cache_time else None,
            "config": {
                "stale_threshold": self._staleness_config.stale_threshold_minutes,
                "expired_threshold": self._staleness_config.very_stale_threshold_minutes,
            },
            "metrics": self._staleness_metrics if hasattr(self, "_staleness_metrics") else {},
        }


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================


def apply_staleness_to_score(
    score: float,
    cache_time: Optional[datetime],
    config: Optional[StalenessConfig] = None,
) -> Tuple[float, DataFreshness]:
    """
    Standalone function to apply staleness weight to a score.

    Args:
        score: Raw score
        cache_time: When data was cached
        config: Staleness config (uses default if None)

    Returns:
        Tuple of (adjusted_score, freshness_level)
    """
    if cache_time is None:
        return 0.0, DataFreshness.EXPIRED

    config = config or StalenessConfig()
    age = (datetime.now() - cache_time).total_seconds() / 60

    if age < config.fresh_threshold_minutes:
        freshness = DataFreshness.FRESH
    elif age < config.slightly_stale_minutes:
        freshness = DataFreshness.SLIGHTLY_STALE
    elif age < config.stale_threshold_minutes:
        freshness = DataFreshness.STALE
    elif age < config.very_stale_threshold_minutes:
        freshness = DataFreshness.VERY_STALE
    else:
        freshness = DataFreshness.EXPIRED

    weight = config.get_weight_for_freshness(freshness)
    return score * weight, freshness


def create_staleness_config(
    data_type: str,
    **overrides,
) -> StalenessConfig:
    """
    Create staleness config with optional overrides.

    Args:
        data_type: Base config type
        **overrides: Override specific values

    Returns:
        StalenessConfig instance
    """
    base = STALENESS_CONFIGS.get(data_type, StalenessConfig())

    # Apply overrides
    config_dict = {
        "fresh_threshold_minutes": base.fresh_threshold_minutes,
        "slightly_stale_minutes": base.slightly_stale_minutes,
        "stale_threshold_minutes": base.stale_threshold_minutes,
        "very_stale_threshold_minutes": base.very_stale_threshold_minutes,
        "fresh_weight": base.fresh_weight,
        "slightly_stale_weight": base.slightly_stale_weight,
        "stale_weight": base.stale_weight,
        "very_stale_weight": base.very_stale_weight,
        "expired_weight": base.expired_weight,
        "log_stale_warnings": base.log_stale_warnings,
        "track_staleness_metrics": base.track_staleness_metrics,
    }
    config_dict.update(overrides)

    return StalenessConfig(**config_dict)


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    "DataFreshness",
    "StalenessConfig",
    "DataStalenessMixin",
    "STALENESS_CONFIGS",
    "FRESHNESS_WEIGHT_FACTORS",
    "apply_staleness_to_score",
    "create_staleness_config",
]
