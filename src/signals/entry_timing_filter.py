"""
Entry Timing Filter
Validates entry timing based on time-of-day, volume, and market conditions
"""

import logging
from dataclasses import dataclass
from datetime import time as Time
from typing import Dict, Optional

import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class TimingFilterResult:
    """Result of timing filter validation"""

    allowed: bool
    confidence_adjustment: float  # Multiplier for confidence (0.5 - 1.2)
    reason: str
    components: Dict[str, bool]  # Individual check results


class EntryTimingFilter:
    """
    Entry Timing Filter

    Validates optimal entry timing based on:
    1. Time of day (avoid opening/closing volatility)
    2. Volume confirmation (minimum volume threshold)
    3. Mid-session preference (10:00-13:30 is optimal)

    Helps avoid:
    - Opening volatility/manipulation (first 15 min)
    - Closing manipulation (last 15 min)
    - Low liquidity periods
    - Abnormal volume conditions
    """

    def __init__(
        self,
        avoid_first_minutes: int = 15,
        avoid_last_minutes: int = 15,
        min_volume_ratio: float = 0.5,  # 50% of avg volume
        optimal_volume_ratio: float = 1.0,  # 100% of avg volume
        market_open_time: Time = Time(9, 0),
        market_close_time: Time = Time(14, 45),
        optimal_start_time: Time = Time(10, 0),
        optimal_end_time: Time = Time(13, 30),
    ):
        """
        Args:
            avoid_first_minutes: Avoid first N minutes after open (default: 15)
            avoid_last_minutes: Avoid last N minutes before close (default: 15)
            min_volume_ratio: Minimum volume vs avg (default: 0.5 = 50%)
            optimal_volume_ratio: Optimal volume vs avg (default: 1.0 = 100%)
            market_open_time: Market open time (default: 9:00)
            market_close_time: Market close time (default: 14:45)
            optimal_start_time: Optimal entry start (default: 10:00)
            optimal_end_time: Optimal entry end (default: 13:30)
        """
        self.avoid_first_minutes = avoid_first_minutes
        self.avoid_last_minutes = avoid_last_minutes
        self.min_volume_ratio = min_volume_ratio
        self.optimal_volume_ratio = optimal_volume_ratio
        self.market_open_time = market_open_time
        self.market_close_time = market_close_time
        self.optimal_start_time = optimal_start_time
        self.optimal_end_time = optimal_end_time

    def validate_entry_timing(
        self,
        current_time: pd.Timestamp,
        current_volume: float,
        avg_volume: float,
        strict_mode: bool = False,
    ) -> TimingFilterResult:
        """
        Validate if current timing is good for entry

        Args:
            current_time: Current timestamp
            current_volume: Current bar volume
            avg_volume: Average volume (e.g., 20-day SMA)
            strict_mode: If True, reject non-optimal times (default: False - only warn)

        Returns:
            TimingFilterResult with allowed flag and adjustments
        """
        components = {}
        reasons = []
        confidence_adjustment = 1.0

        # =================================================================
        # CHECK 1: TIME OF DAY
        # =================================================================
        time_check = self._check_time_of_day(current_time, strict_mode)
        components["time_of_day"] = time_check["allowed"]
        confidence_adjustment *= time_check["adjustment"]

        if not time_check["allowed"]:
            reasons.append(time_check["reason"])
        elif time_check["adjustment"] < 1.0:
            reasons.append(f"Suboptimal time: {time_check['reason']}")

        # =================================================================
        # CHECK 2: VOLUME CONFIRMATION
        # =================================================================
        volume_check = self._check_volume(current_volume, avg_volume, strict_mode)
        components["volume"] = volume_check["allowed"]
        confidence_adjustment *= volume_check["adjustment"]

        if not volume_check["allowed"]:
            reasons.append(volume_check["reason"])
        elif volume_check["adjustment"] < 1.0:
            reasons.append(f"Volume concern: {volume_check['reason']}")

        # =================================================================
        # OVERALL DECISION
        # =================================================================
        allowed = all(components.values())

        if allowed:
            if confidence_adjustment == 1.0:
                reason = "✅ Optimal entry timing (good time + good volume)"
            else:
                reason = f"⚠️ Entry allowed but suboptimal: {'; '.join(reasons)}"
        else:
            reason = f"❌ Entry NOT allowed: {'; '.join(reasons)}"

        # Clamp adjustment to reasonable range
        confidence_adjustment = max(0.5, min(confidence_adjustment, 1.2))

        return TimingFilterResult(
            allowed=allowed,
            confidence_adjustment=confidence_adjustment,
            reason=reason,
            components=components,
        )

    def _check_time_of_day(self, current_time: pd.Timestamp, strict_mode: bool) -> Dict[str, any]:
        """Check if time of day is suitable for entry"""
        current_time_only = current_time.time()

        # Calculate avoid times
        from datetime import datetime, timedelta

        # Avoid first N minutes
        avoid_until = (
            datetime.combine(datetime.today(), self.market_open_time)
            + timedelta(minutes=self.avoid_first_minutes)
        ).time()

        # Avoid last N minutes
        avoid_from = (
            datetime.combine(datetime.today(), self.market_close_time)
            - timedelta(minutes=self.avoid_last_minutes)
        ).time()

        # CHECK: First 15 minutes (NOT ALLOWED)
        if current_time_only < avoid_until:
            return {
                "allowed": False,
                "adjustment": 0.0,
                "reason": f"First {self.avoid_first_minutes}min after open (high volatility)",
            }

        # CHECK: Last 15 minutes (NOT ALLOWED)
        if current_time_only > avoid_from:
            return {
                "allowed": False,
                "adjustment": 0.0,
                "reason": f"Last {self.avoid_last_minutes}min before close (manipulation risk)",
            }

        # CHECK: Optimal window (10:00-13:30)
        if self.optimal_start_time <= current_time_only <= self.optimal_end_time:
            return {
                "allowed": True,
                "adjustment": 1.1,  # Bonus for optimal time
                "reason": "Optimal mid-session time",
            }

        # CHECK: Acceptable but not optimal
        if strict_mode:
            return {
                "allowed": False,
                "adjustment": 0.8,
                "reason": f"Outside optimal window ({self.optimal_start_time}-{self.optimal_end_time})",
            }
        else:
            return {
                "allowed": True,
                "adjustment": 0.9,  # Slight penalty
                "reason": f"Acceptable but outside optimal window",
            }

    def _check_volume(
        self, current_volume: float, avg_volume: float, strict_mode: bool
    ) -> Dict[str, any]:
        """Check if volume is sufficient"""
        if avg_volume <= 0:
            # No avg volume data - allow but with penalty
            return {
                "allowed": True,
                "adjustment": 0.95,
                "reason": "No volume data for validation",
            }

        volume_ratio = current_volume / avg_volume

        # CHECK: Too low volume (NOT ALLOWED in strict mode)
        if volume_ratio < self.min_volume_ratio:
            if strict_mode:
                return {
                    "allowed": False,
                    "adjustment": 0.0,
                    "reason": f"Volume too low ({volume_ratio:.1%} of avg, need {self.min_volume_ratio:.0%})",
                }
            else:
                return {
                    "allowed": True,
                    "adjustment": 0.7,
                    "reason": f"Low volume ({volume_ratio:.1%} of avg)",
                }

        # CHECK: Optimal volume (>= 100% of avg)
        if volume_ratio >= self.optimal_volume_ratio:
            bonus = min(1.2, 1.0 + (volume_ratio - 1.0) * 0.1)  # Max 1.2x bonus
            return {
                "allowed": True,
                "adjustment": bonus,
                "reason": f"Good volume ({volume_ratio:.1%} of avg)",
            }

        # CHECK: Acceptable volume (50%-100%)
        # Linear interpolation between 0.9 and 1.0
        adjustment = (
            0.9
            + (volume_ratio - self.min_volume_ratio)
            / (self.optimal_volume_ratio - self.min_volume_ratio)
            * 0.1
        )

        return {
            "allowed": True,
            "adjustment": adjustment,
            "reason": f"Moderate volume ({volume_ratio:.1%} of avg)",
        }

    def get_next_optimal_time(self, current_time: pd.Timestamp) -> Optional[pd.Timestamp]:
        """
        Get next optimal entry time

        Returns:
            Next optimal time, or None if already optimal
        """
        current_time_only = current_time.time()

        # Already in optimal window
        if self.optimal_start_time <= current_time_only <= self.optimal_end_time:
            return None

        # Before optimal window - return optimal start
        if current_time_only < self.optimal_start_time:
            return pd.Timestamp(
                year=current_time.year,
                month=current_time.month,
                day=current_time.day,
                hour=self.optimal_start_time.hour,
                minute=self.optimal_start_time.minute,
            )

        # After optimal window - return next day optimal start
        next_day = current_time + pd.Timedelta(days=1)
        return pd.Timestamp(
            year=next_day.year,
            month=next_day.month,
            day=next_day.day,
            hour=self.optimal_start_time.hour,
            minute=self.optimal_start_time.minute,
        )


# Singleton instance
_timing_filter_instance: Optional[EntryTimingFilter] = None


def get_timing_filter() -> EntryTimingFilter:
    """Get singleton instance of timing filter"""
    global _timing_filter_instance
    if _timing_filter_instance is None:
        _timing_filter_instance = EntryTimingFilter()
    return _timing_filter_instance


def validate_entry_timing(
    current_time: pd.Timestamp,
    current_volume: float,
    avg_volume: float,
    strict_mode: bool = False,
) -> TimingFilterResult:
    """
    Convenience function to validate entry timing

    Args:
        current_time: Current timestamp
        current_volume: Current bar volume
        avg_volume: Average volume
        strict_mode: Strict validation mode

    Returns:
        TimingFilterResult
    """
    filter = get_timing_filter()
    return filter.validate_entry_timing(current_time, current_volume, avg_volume, strict_mode)
