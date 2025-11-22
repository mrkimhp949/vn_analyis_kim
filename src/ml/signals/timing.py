"""
Timing Analysis Module
Provides optimal entry/exit timing hints for signals
"""

import logging
from dataclasses import dataclass
from datetime import datetime, time
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class TimingHint:
    """Container for timing analysis results"""

    is_good_timing: bool
    timing_score: float  # 0-1, higher is better
    best_entry_time: str  # Human-readable
    time_until_best: Optional[str]
    reasons: list
    current_time_category: str  # "OPEN", "MID_DAY", "CLOSE", "AFTER_HOURS"


class TimingAnalyzer:
    """
    Analyze optimal entry/exit timing for Vietnamese stock market

    Market hours: 9:00 AM - 3:00 PM
    Best practices:
    - Avoid first 15-30 min (gap noise, emotional trading)
    - Best entry: 9:30-10:30 AM (after opening volatility settles)
    - Best exit: 2:00-2:45 PM (before closing rush)
    - Avoid last 15 min (closing auction manipulation)
    """

    def __init__(self):
        # Vietnamese stock market hours (24-hour format)
        self.market_open = time(9, 0)
        self.market_close = time(15, 0)

        # Optimal time windows
        self.best_entry_start = time(9, 30)
        self.best_entry_end = time(10, 30)

        self.best_exit_start = time(14, 0)
        self.best_exit_end = time(14, 45)

        # Avoid times
        self.avoid_open_until = time(9, 30)  # First 30 min
        self.avoid_close_from = time(14, 45)  # Last 15 min

    def analyze_timing(
        self, current_time: Optional[datetime] = None, action: str = "BUY"
    ) -> TimingHint:
        """
        Analyze if current time is good for trading

        Args:
            current_time: Time to analyze (default: now)
            action: "BUY" or "SELL"

        Returns:
            TimingHint with analysis
        """
        if current_time is None:
            current_time = datetime.now()

        ct = current_time.time()
        reasons = []
        timing_score = 0.5  # Neutral
        is_good_timing = True

        # =================================================================
        # MARKET HOURS CHECK
        # =================================================================
        if ct < self.market_open:
            category = "BEFORE_MARKET"
            is_good_timing = False
            timing_score = 0.2
            reasons.append("⏰ Market not open yet")
            time_until_best = self._time_diff(ct, self.best_entry_start)
            best_time = "9:30 AM (after opening volatility)"

        elif ct > self.market_close:
            category = "AFTER_HOURS"
            is_good_timing = False
            timing_score = 0.0
            reasons.append("⏰ Market closed")
            time_until_best = "Next trading day at 9:30 AM"
            best_time = "9:30 AM (tomorrow)"

        # =================================================================
        # OPENING PERIOD (9:00-9:30 AM) - HIGH VOLATILITY
        # =================================================================
        elif self.market_open <= ct < self.avoid_open_until:
            category = "OPEN"
            timing_score = 0.3
            is_good_timing = False
            reasons.append("⚠️ Opening period - high volatility")
            reasons.append("💡 Wait for price to settle")
            time_until_best = self._time_diff(ct, self.best_entry_start)
            best_time = "9:30-10:30 AM"

        # =================================================================
        # BEST ENTRY WINDOW (9:30-10:30 AM)
        # =================================================================
        elif self.best_entry_start <= ct <= self.best_entry_end:
            category = "BEST_ENTRY"
            timing_score = 1.0 if action == "BUY" else 0.7
            is_good_timing = True
            reasons.append("✅ Optimal entry time")
            reasons.append("📈 Post-open volatility settled")
            time_until_best = None
            best_time = "Now (optimal window)"

        # =================================================================
        # MID-DAY TRADING (10:30 AM - 2:00 PM)
        # =================================================================
        elif self.best_entry_end < ct < self.best_exit_start:
            category = "MID_DAY"
            timing_score = 0.7  # Good but not optimal
            is_good_timing = True
            reasons.append("✅ Good trading time")
            reasons.append("📊 Stable market conditions")
            time_until_best = None
            best_time = "Now (acceptable)"

        # =================================================================
        # BEST EXIT WINDOW (2:00-2:45 PM)
        # =================================================================
        elif self.best_exit_start <= ct <= self.best_exit_end:
            category = "BEST_EXIT"
            timing_score = 1.0 if action == "SELL" else 0.7
            is_good_timing = True
            reasons.append("✅ Optimal exit time" if action == "SELL" else "✅ Good entry time")
            reasons.append("💰 Before closing rush")
            time_until_best = None
            best_time = "Now (optimal for exits)"

        # =================================================================
        # CLOSING PERIOD (2:45-3:00 PM) - AVOID
        # =================================================================
        elif self.avoid_close_from < ct <= self.market_close:
            category = "CLOSE"
            timing_score = 0.2
            is_good_timing = False
            reasons.append("⚠️ Closing period - avoid trading")
            reasons.append("🎭 High risk of manipulation")
            time_until_best = "Next trading day at 9:30 AM"
            best_time = "9:30 AM (tomorrow)"

        else:
            # Fallback
            category = "UNKNOWN"
            timing_score = 0.5
            is_good_timing = True
            reasons.append("📊 Normal trading hours")
            time_until_best = None
            best_time = "Now"

        # =================================================================
        # ACTION-SPECIFIC RECOMMENDATIONS
        # =================================================================
        if action == "BUY" and category in ["BEST_ENTRY", "MID_DAY"]:
            reasons.append("🎯 Good time to enter positions")
        elif action == "SELL" and category in ["BEST_EXIT", "MID_DAY"]:
            reasons.append("🎯 Good time to exit positions")

        return TimingHint(
            is_good_timing=is_good_timing,
            timing_score=timing_score,
            best_entry_time=best_time,
            time_until_best=time_until_best,
            reasons=reasons,
            current_time_category=category,
        )

    def _time_diff(self, time1: time, time2: time) -> str:
        """Calculate human-readable time difference"""
        # Convert to minutes from midnight
        t1_mins = time1.hour * 60 + time1.minute
        t2_mins = time2.hour * 60 + time2.minute

        diff_mins = t2_mins - t1_mins

        if diff_mins < 0:
            return "Next day"

        hours = diff_mins // 60
        mins = diff_mins % 60

        if hours > 0:
            return f"{hours}h {mins}m"
        else:
            return f"{mins}m"

    def get_timing_score_for_signal(self, signal: dict) -> dict:
        """
        Add timing information to a signal

        Args:
            signal: Signal dict with 'action'

        Returns:
            Updated signal with timing info
        """
        action = signal.get("action", "BUY")
        timing = self.analyze_timing(action=action)

        # Add timing info to signal
        signal["timing"] = {
            "is_good_timing": timing.is_good_timing,
            "timing_score": timing.timing_score,
            "best_entry_time": timing.best_entry_time,
            "time_until_best": timing.time_until_best,
            "category": timing.current_time_category,
            "reasons": timing.reasons,
        }

        # Adjust confidence based on timing
        if "confidence" in signal:
            original_confidence = signal["confidence"]

            # Boost/reduce confidence based on timing
            if timing.timing_score >= 0.9:
                # Optimal timing - boost confidence
                signal["confidence"] = min(original_confidence * 1.05, 1.0)
                signal["timing"]["adjustment"] = "+5%"
            elif timing.timing_score <= 0.3:
                # Bad timing - reduce confidence
                signal["confidence"] = original_confidence * 0.85
                signal["timing"]["adjustment"] = "-15%"
            else:
                signal["timing"]["adjustment"] = "0%"

            logger.debug(
                f"Timing adjustment for {signal.get('symbol')}: "
                f"{original_confidence:.2%} → {signal['confidence']:.2%} "
                f"(timing_score: {timing.timing_score:.2f})"
            )

        return signal


# Singleton instance
_timing_analyzer = None


def get_timing_analyzer() -> TimingAnalyzer:
    """Get timing analyzer singleton"""
    global _timing_analyzer
    if _timing_analyzer is None:
        _timing_analyzer = TimingAnalyzer()
    return _timing_analyzer


# Convenience function
def add_timing_to_signal(signal: dict) -> dict:
    """Add timing analysis to a signal"""
    analyzer = get_timing_analyzer()
    return analyzer.get_timing_score_for_signal(signal)


# Testing
if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("🕐 TESTING TIMING ANALYZER")
    print("=" * 70 + "\n")

    analyzer = TimingAnalyzer()

    # Test different times
    test_times = [
        datetime(2024, 1, 15, 9, 0),  # Market open
        datetime(2024, 1, 15, 9, 15),  # Opening volatility
        datetime(2024, 1, 15, 9, 45),  # Best entry
        datetime(2024, 1, 15, 12, 0),  # Mid-day
        datetime(2024, 1, 15, 14, 30),  # Best exit
        datetime(2024, 1, 15, 14, 50),  # Closing
        datetime(2024, 1, 15, 16, 0),  # After hours
    ]

    for test_time in test_times:
        timing = analyzer.analyze_timing(test_time, action="BUY")
        print(f"\n⏰ Time: {test_time.strftime('%H:%M')}")
        print(f"  Category: {timing.current_time_category}")
        print(f"  Good timing: {timing.is_good_timing}")
        print(f"  Score: {timing.timing_score:.2f}")
        print(f"  Best time: {timing.best_entry_time}")
        if timing.time_until_best:
            print(f"  Wait: {timing.time_until_best}")
        print(f"  Reasons:")
        for reason in timing.reasons:
            print(f"    • {reason}")

    print("\n" + "=" * 70)
    print("✅ Timing analysis complete!")
    print("=" * 70 + "\n")
