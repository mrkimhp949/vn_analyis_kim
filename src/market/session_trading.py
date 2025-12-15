# -*- coding: utf-8 -*-
"""
Vietnam Market Session Trading Logic
ATO/ATC và Entry Timing Optimization

FEATURES:
- ATO (At The Open) session detection và strategy
- ATC (At The Close) session detection và strategy
- Optimal entry timing windows
- Session-specific risk adjustments
- Order type recommendations (ATO/ATC/LO/MP)

IMPROVEMENTS v2.0:
- Added constants for magic numbers
- Thread-safe singleton
- Added __all__ export list
"""

import logging
import threading
from dataclasses import dataclass
from datetime import datetime, time
from enum import Enum
from typing import Dict, List, Optional, Tuple

import pytz

logger = logging.getLogger(__name__)

__all__ = [
    "SessionType",
    "OrderType",
    "SessionInfo",
    "EntryTimingResult",
    "SessionTradingManager",
    "get_session_manager",
    "get_current_session",
    "analyze_entry_timing",
    "is_optimal_entry_time",
]

# =============================================================================
# CONSTANTS
# =============================================================================

# Quality score thresholds
QUALITY_SCORE_OPTIMAL = 85.0
QUALITY_SCORE_ACCEPTABLE = 60.0
QUALITY_SCORE_AVOID = 20.0
QUALITY_SCORE_CLOSED = 0.0
QUALITY_SCORE_BASE = 50.0

# Quality bonuses/penalties
QUALITY_BONUS_SWEET_SPOT = 10
QUALITY_PENALTY_PRE_LUNCH = 15
QUALITY_PENALTY_EARLY_AFTERNOON = 5
QUALITY_BONUS_HIGH_URGENCY = 20

# Position size multipliers
SIZE_MULTIPLIER_EXCELLENT = 1.0
SIZE_MULTIPLIER_GOOD = 0.8
SIZE_MULTIPLIER_ACCEPTABLE = 0.6
SIZE_MULTIPLIER_POOR = 0.5

# Thresholds
OPTIMAL_QUALITY_THRESHOLD = 70
STRONG_SIGNAL_THRESHOLD = 80
LARGE_POSITION_THRESHOLD = 0.15  # 15% of portfolio

# Quality score boundaries for position sizing
QUALITY_EXCELLENT_THRESHOLD = 80
QUALITY_GOOD_THRESHOLD = 60
QUALITY_ACCEPTABLE_THRESHOLD = 40

# Vietnam timezone
VN_TZ = pytz.timezone("Asia/Ho_Chi_Minh")


class SessionType(Enum):
    """Trading session types"""

    PRE_OPEN = "PRE_OPEN"  # Before 9:00
    ATO = "ATO"  # 9:00-9:15 (Opening auction)
    MORNING_CONTINUOUS = "MORNING"  # 9:15-11:30
    LUNCH_BREAK = "LUNCH"  # 11:30-13:00
    AFTERNOON_CONTINUOUS = "AFTERNOON"  # 13:00-14:30
    ATC = "ATC"  # 14:30-14:45 (Closing auction)
    POST_CLOSE = "POST_CLOSE"  # After 14:45
    CLOSED = "CLOSED"  # Weekend/Holiday


class OrderType(Enum):
    """Order types for Vietnam market"""

    ATO = "ATO"  # At The Open order
    ATC = "ATC"  # At The Close order
    LO = "LO"  # Limit Order
    MP = "MP"  # Market Price (best available)
    MTL = "MTL"  # Market To Limit
    MOK = "MOK"  # Match Or Kill
    MAK = "MAK"  # Match And Kill


@dataclass
class SessionInfo:
    """Current session information"""

    session_type: SessionType
    time_remaining: int  # Minutes remaining in session
    is_auction: bool  # True if ATO/ATC
    is_continuous: bool  # True if continuous trading
    recommended_order_type: OrderType
    risk_level: str  # LOW, MEDIUM, HIGH
    entry_quality: str  # OPTIMAL, ACCEPTABLE, AVOID
    warnings: List[str]
    recommendations: List[str]


@dataclass
class EntryTimingResult:
    """Entry timing analysis result"""

    is_optimal: bool
    quality_score: float  # 0-100
    session: SessionType
    recommended_order_type: OrderType
    position_size_multiplier: float  # 0.5-1.5
    reasons: List[str]
    warnings: List[str]


class SessionTradingManager:
    """
    Manages session-specific trading logic for Vietnam market

    Vietnam Market Sessions:
    - Pre-open: Before 9:00
    - ATO (Opening Auction): 9:00-9:15
    - Morning Continuous: 9:15-11:30
    - Lunch Break: 11:30-13:00
    - Afternoon Continuous: 13:00-14:30
    - ATC (Closing Auction): 14:30-14:45
    - Post-close: After 14:45

    Optimal Entry Windows:
    - 9:30-10:30: After opening volatility settles
    - 13:30-14:15: After lunch gap settles, before ATC

    Avoid Entry:
    - 9:00-9:15: ATO auction (high volatility)
    - 11:00-11:30: Pre-lunch selling pressure
    - 14:30-14:45: ATC auction (high volatility)
    """

    # Session times
    SESSIONS = {
        "ATO_START": time(9, 0),
        "ATO_END": time(9, 15),
        "MORNING_END": time(11, 30),
        "AFTERNOON_START": time(13, 0),
        "ATC_START": time(14, 30),
        "ATC_END": time(14, 45),
    }

    # Optimal entry windows
    OPTIMAL_WINDOWS = [
        (time(9, 30), time(10, 30), "Morning optimal - post-ATO stability"),
        (time(13, 30), time(14, 15), "Afternoon optimal - pre-ATC window"),
    ]

    # Avoid windows
    AVOID_WINDOWS = [
        (time(9, 0), time(9, 15), "ATO auction - high volatility"),
        (time(11, 0), time(11, 30), "Pre-lunch - selling pressure"),
        (time(14, 30), time(14, 45), "ATC auction - high volatility"),
    ]

    # Acceptable windows (not optimal but OK)
    ACCEPTABLE_WINDOWS = [
        (time(9, 15), time(9, 30), "Post-ATO settling"),
        (time(10, 30), time(11, 0), "Mid-morning"),
        (time(13, 0), time(13, 30), "Early afternoon"),
        (time(14, 15), time(14, 30), "Pre-ATC"),
    ]

    def __init__(self):
        pass  # Stateless manager - no caching needed

    def get_current_session(self, dt: Optional[datetime] = None) -> SessionInfo:
        """
        Get current trading session information

        Args:
            dt: Datetime to check (None = now)

        Returns:
            SessionInfo object
        """
        if dt is None:
            dt = datetime.now(VN_TZ)
        else:
            if dt.tzinfo is None:
                dt = VN_TZ.localize(dt)
            else:
                dt = dt.astimezone(VN_TZ)

        current_time = dt.time()
        weekday = dt.weekday()

        # Weekend check
        if weekday >= 5:
            return SessionInfo(
                session_type=SessionType.CLOSED,
                time_remaining=0,
                is_auction=False,
                is_continuous=False,
                recommended_order_type=OrderType.LO,
                risk_level="N/A",
                entry_quality="CLOSED",
                warnings=["Market closed (weekend)"],
                recommendations=["Wait for market open"],
            )

        # Determine session
        session_type, time_remaining = self._determine_session(current_time)

        # Session-specific info
        is_auction = session_type in [SessionType.ATO, SessionType.ATC]
        is_continuous = session_type in [
            SessionType.MORNING_CONTINUOUS,
            SessionType.AFTERNOON_CONTINUOUS,
        ]

        # Recommended order type
        if session_type == SessionType.ATO:
            order_type = OrderType.ATO
        elif session_type == SessionType.ATC:
            order_type = OrderType.ATC
        elif is_continuous:
            order_type = OrderType.LO  # Limit order for better execution
        else:
            order_type = OrderType.LO

        # Risk level and entry quality
        risk_level, entry_quality = self._assess_session_risk(session_type, current_time)

        # Warnings and recommendations
        warnings, recommendations = self._get_session_guidance(session_type, time_remaining)

        return SessionInfo(
            session_type=session_type,
            time_remaining=time_remaining,
            is_auction=is_auction,
            is_continuous=is_continuous,
            recommended_order_type=order_type,
            risk_level=risk_level,
            entry_quality=entry_quality,
            warnings=warnings,
            recommendations=recommendations,
        )

    def _determine_session(self, current_time: time) -> Tuple[SessionType, int]:
        """Determine current session and time remaining"""

        # Pre-open
        if current_time < self.SESSIONS["ATO_START"]:
            mins_to_open = self._time_diff_minutes(current_time, self.SESSIONS["ATO_START"])
            return SessionType.PRE_OPEN, mins_to_open

        # ATO
        if current_time < self.SESSIONS["ATO_END"]:
            mins_remaining = self._time_diff_minutes(current_time, self.SESSIONS["ATO_END"])
            return SessionType.ATO, mins_remaining

        # Morning continuous
        if current_time < self.SESSIONS["MORNING_END"]:
            mins_remaining = self._time_diff_minutes(current_time, self.SESSIONS["MORNING_END"])
            return SessionType.MORNING_CONTINUOUS, mins_remaining

        # Lunch break
        if current_time < self.SESSIONS["AFTERNOON_START"]:
            mins_remaining = self._time_diff_minutes(current_time, self.SESSIONS["AFTERNOON_START"])
            return SessionType.LUNCH_BREAK, mins_remaining

        # Afternoon continuous
        if current_time < self.SESSIONS["ATC_START"]:
            mins_remaining = self._time_diff_minutes(current_time, self.SESSIONS["ATC_START"])
            return SessionType.AFTERNOON_CONTINUOUS, mins_remaining

        # ATC
        if current_time < self.SESSIONS["ATC_END"]:
            mins_remaining = self._time_diff_minutes(current_time, self.SESSIONS["ATC_END"])
            return SessionType.ATC, mins_remaining

        # Post-close
        return SessionType.POST_CLOSE, 0

    def _time_diff_minutes(self, t1: time, t2: time) -> int:
        """Calculate minutes between two times"""
        mins1 = t1.hour * 60 + t1.minute
        mins2 = t2.hour * 60 + t2.minute
        return max(0, mins2 - mins1)

    def _assess_session_risk(
        self, session_type: SessionType, current_time: time
    ) -> Tuple[str, str]:
        """Assess risk level and entry quality for current session"""

        # Auction sessions = HIGH risk
        if session_type in [SessionType.ATO, SessionType.ATC]:
            return "HIGH", "AVOID"

        # Non-trading = N/A
        if session_type in [
            SessionType.PRE_OPEN,
            SessionType.LUNCH_BREAK,
            SessionType.POST_CLOSE,
            SessionType.CLOSED,
        ]:
            return "N/A", "CLOSED"

        # Check optimal windows
        for start, end, _ in self.OPTIMAL_WINDOWS:
            if start <= current_time <= end:
                return "LOW", "OPTIMAL"

        # Check avoid windows
        for start, end, _ in self.AVOID_WINDOWS:
            if start <= current_time <= end:
                return "HIGH", "AVOID"

        # Check acceptable windows
        for start, end, _ in self.ACCEPTABLE_WINDOWS:
            if start <= current_time <= end:
                return "MEDIUM", "ACCEPTABLE"

        # Default
        return "MEDIUM", "ACCEPTABLE"

    def _get_session_guidance(
        self, session_type: SessionType, time_remaining: int
    ) -> Tuple[List[str], List[str]]:
        """Get warnings and recommendations for session"""
        warnings = []
        recommendations = []

        if session_type == SessionType.ATO:
            warnings.append("⚠️ ATO auction - High volatility expected")
            warnings.append("⚠️ Price discovery phase - spreads may be wide")
            recommendations.append("Use ATO orders only for strong conviction")
            recommendations.append("Consider waiting for continuous session")

        elif session_type == SessionType.ATC:
            warnings.append("⚠️ ATC auction - High volatility expected")
            warnings.append("⚠️ Large institutional orders may move prices")
            recommendations.append("Use ATC orders for end-of-day positioning")
            recommendations.append("Avoid if not necessary")

        elif session_type == SessionType.MORNING_CONTINUOUS:
            if time_remaining <= 30:
                warnings.append("⚠️ Approaching lunch break - potential selling")
            recommendations.append("Use limit orders for better execution")

        elif session_type == SessionType.AFTERNOON_CONTINUOUS:
            if time_remaining <= 15:
                warnings.append("⚠️ Approaching ATC - volatility may increase")
            recommendations.append("Complete trades before 14:30 if possible")

        elif session_type == SessionType.LUNCH_BREAK:
            recommendations.append("Prepare orders for afternoon session")
            recommendations.append("Review morning price action")

        elif session_type == SessionType.PRE_OPEN:
            recommendations.append("Review overnight news and global markets")
            recommendations.append("Prepare ATO orders if needed")

        return warnings, recommendations

    def analyze_entry_timing(
        self,
        dt: Optional[datetime] = None,
        urgency: str = "NORMAL",  # LOW, NORMAL, HIGH
    ) -> EntryTimingResult:
        """
        Analyze if current time is good for entry

        Args:
            dt: Datetime to check (None = now)
            urgency: Trade urgency level

        Returns:
            EntryTimingResult with recommendations
        """
        session_info = self.get_current_session(dt)

        if dt is None:
            dt = datetime.now(VN_TZ)
        else:
            if dt.tzinfo is None:
                dt = VN_TZ.localize(dt)

        current_time = dt.time()

        # Base quality score
        quality_score = QUALITY_SCORE_BASE
        reasons = []
        warnings = []

        # Session-based scoring
        if session_info.entry_quality == "OPTIMAL":
            quality_score = QUALITY_SCORE_OPTIMAL
            reasons.append("✅ Optimal entry window")
        elif session_info.entry_quality == "ACCEPTABLE":
            quality_score = QUALITY_SCORE_ACCEPTABLE
            reasons.append("📊 Acceptable entry window")
        elif session_info.entry_quality == "AVOID":
            quality_score = QUALITY_SCORE_AVOID
            warnings.append("⚠️ Suboptimal entry timing")
        elif session_info.entry_quality == "CLOSED":
            quality_score = QUALITY_SCORE_CLOSED
            warnings.append("🚫 Market closed")

        # Time-specific adjustments
        if session_info.session_type == SessionType.MORNING_CONTINUOUS:
            # Best: 9:30-10:30
            if time(9, 30) <= current_time <= time(10, 30):
                quality_score += QUALITY_BONUS_SWEET_SPOT
                reasons.append("🌅 Morning sweet spot (9:30-10:30)")
            # Avoid: 11:00-11:30
            elif time(11, 0) <= current_time:
                quality_score -= QUALITY_PENALTY_PRE_LUNCH
                warnings.append("⚠️ Pre-lunch selling pressure zone")

        elif session_info.session_type == SessionType.AFTERNOON_CONTINUOUS:
            # Best: 13:30-14:15
            if time(13, 30) <= current_time <= time(14, 15):
                quality_score += QUALITY_BONUS_SWEET_SPOT
                reasons.append("🌆 Afternoon sweet spot (13:30-14:15)")
            # Early afternoon gap risk
            elif current_time < time(13, 30):
                quality_score -= QUALITY_PENALTY_EARLY_AFTERNOON
                warnings.append("📊 Early afternoon - watch for gap fill")

        # Urgency adjustments
        if urgency == "HIGH":
            # High urgency = accept lower quality
            quality_score = min(quality_score + QUALITY_BONUS_HIGH_URGENCY, 100)
            reasons.append("⚡ High urgency trade")
        elif urgency == "LOW":
            # Low urgency = be more selective
            if quality_score < 70:
                warnings.append("💡 Consider waiting for better timing")

        # Position size multiplier based on timing
        if quality_score >= QUALITY_EXCELLENT_THRESHOLD:
            size_multiplier = SIZE_MULTIPLIER_EXCELLENT
        elif quality_score >= QUALITY_GOOD_THRESHOLD:
            size_multiplier = SIZE_MULTIPLIER_GOOD
        elif quality_score >= QUALITY_ACCEPTABLE_THRESHOLD:
            size_multiplier = SIZE_MULTIPLIER_ACCEPTABLE
        else:
            size_multiplier = SIZE_MULTIPLIER_POOR

        # Determine if optimal
        is_optimal = (
            quality_score >= OPTIMAL_QUALITY_THRESHOLD and session_info.entry_quality != "AVOID"
        )

        return EntryTimingResult(
            is_optimal=is_optimal,
            quality_score=quality_score,
            session=session_info.session_type,
            recommended_order_type=session_info.recommended_order_type,
            position_size_multiplier=size_multiplier,
            reasons=reasons,
            warnings=warnings + session_info.warnings,
        )

    def get_next_optimal_window(self, dt: Optional[datetime] = None) -> Optional[Dict]:
        """
        Get next optimal entry window

        Returns:
            Dict with window info or None if no more windows today
        """
        if dt is None:
            dt = datetime.now(VN_TZ)
        else:
            if dt.tzinfo is None:
                dt = VN_TZ.localize(dt)

        current_time = dt.time()

        for start, end, description in self.OPTIMAL_WINDOWS:
            if current_time < start:
                # This window is upcoming
                mins_until = self._time_diff_minutes(current_time, start)
                return {
                    "start": start.strftime("%H:%M"),
                    "end": end.strftime("%H:%M"),
                    "description": description,
                    "minutes_until": mins_until,
                }
            elif start <= current_time <= end:
                # Currently in this window
                mins_remaining = self._time_diff_minutes(current_time, end)
                return {
                    "start": start.strftime("%H:%M"),
                    "end": end.strftime("%H:%M"),
                    "description": description,
                    "minutes_remaining": mins_remaining,
                    "is_current": True,
                }

        return None  # No more optimal windows today

    def should_use_ato_order(
        self,
        signal_strength: float,
        overnight_gap_expected: bool = False,
    ) -> Tuple[bool, str]:
        """
        Determine if ATO order should be used

        Args:
            signal_strength: Signal strength (0-100)
            overnight_gap_expected: True if expecting gap up/down

        Returns:
            (should_use, reason)
        """
        # ATO orders are risky - only use for strong signals
        if signal_strength < STRONG_SIGNAL_THRESHOLD:
            return False, "Signal not strong enough for ATO order"

        if overnight_gap_expected:
            return True, "Strong signal + expected gap - ATO may capture move"

        # Default: prefer waiting for continuous session
        return False, "Consider waiting for continuous session for better execution"

    def should_use_atc_order(
        self,
        is_exit: bool,
        position_size_pct: float,
        urgency: str = "NORMAL",
    ) -> Tuple[bool, str]:
        """
        Determine if ATC order should be used

        Args:
            is_exit: True if this is an exit order
            position_size_pct: Position size as % of portfolio
            urgency: Trade urgency

        Returns:
            (should_use, reason)
        """
        # ATC good for exits to ensure execution
        if is_exit and urgency == "HIGH":
            return True, "High urgency exit - ATC ensures execution"

        # Large positions may benefit from ATC liquidity
        if position_size_pct > LARGE_POSITION_THRESHOLD:
            return True, "Large position - ATC provides better liquidity"

        # Default: prefer continuous session
        return False, "Use limit order in continuous session for better price"

    def get_adaptive_entry_timing(
        self,
        signal_confidence: float,
        market_regime: Optional[Dict] = None,
        is_volatile_day: bool = False,
        symbol: Optional[str] = None,
    ) -> Dict:
        """
        Get adaptive entry timing recommendation based on market conditions.

        This enhanced method considers:
        - Current session timing
        - Market regime (BULL/BEAR/SIDEWAYS)
        - Intraday volatility
        - Signal confidence level

        Args:
            signal_confidence: Signal confidence (0-100)
            market_regime: Market regime from regime_detector
            is_volatile_day: True if high intraday volatility
            symbol: Stock symbol for logging

        Returns:
            Dict with timing recommendation and adjustments
        """
        session_info = self.get_current_session()
        timing_result = self.analyze_entry_timing()

        result = {
            "session": session_info.session_type.value,
            "base_quality": timing_result.quality_score,
            "adjusted_quality": timing_result.quality_score,
            "position_multiplier": timing_result.position_size_multiplier,
            "order_type": timing_result.recommended_order_type.value,
            "is_tradeable": True,
            "adjustments": [],
            "recommendations": [],
        }

        # 1. Market Regime Adjustments
        if market_regime:
            regime = market_regime.get("regime", "SIDEWAYS")
            regime_confidence = market_regime.get("confidence", 50)

            if regime == "BEAR":
                # In bear market, prefer morning entries (before afternoon selling)
                if session_info.session_type == SessionType.AFTERNOON_CONTINUOUS:
                    result["adjusted_quality"] -= 15
                    result["position_multiplier"] *= 0.8
                    result["adjustments"].append("Bear market: prefer morning entry")

                # Avoid pre-lunch in bear market (strong selling pressure)
                current_time = datetime.now(VN_TZ).time()
                if time(10, 30) <= current_time <= time(11, 30):
                    result["adjusted_quality"] -= 20
                    result["adjustments"].append("Bear market: avoid pre-lunch selling")

            elif regime == "BULL":
                # In bull market, afternoon entries can catch momentum
                if session_info.session_type == SessionType.AFTERNOON_CONTINUOUS:
                    result["adjusted_quality"] += 5
                    result["adjustments"].append("Bull market: afternoon momentum OK")

            elif regime == "HIGH_VOLATILITY":
                # High volatility: prefer optimal windows only
                if not timing_result.is_optimal:
                    result["adjusted_quality"] -= 25
                    result["position_multiplier"] *= 0.6
                    result["adjustments"].append("High volatility: wait for optimal window")

            # Low regime confidence: be more cautious
            if regime_confidence < 60:
                result["position_multiplier"] *= 0.9
                result["adjustments"].append("Low regime confidence: reduced size")

        # 2. Volatility Adjustments
        if is_volatile_day:
            result["adjusted_quality"] -= 15
            result["position_multiplier"] *= 0.7

            # On volatile days, prefer mid-session entries
            current_time = datetime.now(VN_TZ).time()
            if time(9, 45) <= current_time <= time(10, 30):
                result["adjusted_quality"] += 10
                result["recommendations"].append("Volatile day: 9:45-10:30 sweet spot")
            elif time(13, 30) <= current_time <= time(14, 0):
                result["adjusted_quality"] += 10
                result["recommendations"].append("Volatile day: 13:30-14:00 sweet spot")
            else:
                result["recommendations"].append("Volatile day: wait for mid-session")

            result["adjustments"].append("Volatile day: reduced exposure")

        # 3. Signal Confidence Integration
        if signal_confidence < 50:
            # Low confidence: require optimal timing
            if not timing_result.is_optimal:
                result["adjusted_quality"] -= 20
                result["recommendations"].append("Low confidence: wait for optimal timing")
        elif signal_confidence >= 80:
            # High confidence: more flexible timing
            result["adjusted_quality"] += 10
            result["recommendations"].append("High confidence: timing less critical")

        # 4. Order Type Recommendations
        if session_info.is_auction:
            if signal_confidence < 70:
                result["order_type"] = "LO"  # Use limit order instead
                result["recommendations"].append("Auction: use limit order for safety")
        elif is_volatile_day:
            result["order_type"] = "LO"
            result["recommendations"].append("Volatile: limit order recommended")

        # 5. Tradeable Decision
        result["adjusted_quality"] = max(0, min(100, result["adjusted_quality"]))

        if result["adjusted_quality"] < 30:
            result["is_tradeable"] = False
            result["recommendations"].append("Timing quality too low - wait")
        elif session_info.session_type in [
            SessionType.CLOSED,
            SessionType.LUNCH_BREAK,
            SessionType.POST_CLOSE,
        ]:
            result["is_tradeable"] = False
            result["recommendations"].append("Market not open for trading")

        # 6. Final position multiplier
        result["position_multiplier"] = max(0.3, min(1.2, result["position_multiplier"]))

        # Add timing to next optimal window
        next_window = self.get_next_optimal_window()
        if next_window and not timing_result.is_optimal:
            if "minutes_until" in next_window:
                result["next_optimal"] = next_window
                result["recommendations"].append(
                    f"Next optimal: {next_window['start']} ({next_window['minutes_until']} mins)"
                )

        return result


# Singleton instance with thread-safe initialization
_session_manager = None
_session_manager_lock = threading.Lock()


def get_session_manager() -> SessionTradingManager:
    """Get singleton instance (thread-safe)"""
    global _session_manager
    if _session_manager is None:
        with _session_manager_lock:
            if _session_manager is None:
                _session_manager = SessionTradingManager()
    return _session_manager


# Convenience functions
def get_current_session() -> SessionInfo:
    """Get current trading session"""
    return get_session_manager().get_current_session()


def analyze_entry_timing(urgency: str = "NORMAL") -> EntryTimingResult:
    """Analyze current entry timing"""
    return get_session_manager().analyze_entry_timing(urgency=urgency)


def is_optimal_entry_time() -> bool:
    """Quick check if now is optimal for entry"""
    result = analyze_entry_timing()
    return result.is_optimal


# Test
if __name__ == "__main__":
    print("Testing Session Trading Manager...")

    manager = SessionTradingManager()

    # Test current session
    session = manager.get_current_session()
    print(f"\nCurrent Session: {session.session_type.value}")
    print(f"Time Remaining: {session.time_remaining} minutes")
    print(f"Risk Level: {session.risk_level}")
    print(f"Entry Quality: {session.entry_quality}")
    print(f"Recommended Order: {session.recommended_order_type.value}")

    print("\nWarnings:")
    for w in session.warnings:
        print(f"  {w}")

    print("\nRecommendations:")
    for r in session.recommendations:
        print(f"  {r}")

    # Test entry timing
    timing = manager.analyze_entry_timing()
    print(f"\nEntry Timing Analysis:")
    print(f"  Is Optimal: {timing.is_optimal}")
    print(f"  Quality Score: {timing.quality_score:.1f}")
    print(f"  Position Size Multiplier: {timing.position_size_multiplier:.2f}")

    # Test next optimal window
    next_window = manager.get_next_optimal_window()
    if next_window:
        print(f"\nNext Optimal Window: {next_window}")

    print("\n✅ Session Trading Manager test completed!")
