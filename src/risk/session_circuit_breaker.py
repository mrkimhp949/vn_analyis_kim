# -*- coding: utf-8 -*-
"""
Session Circuit Breaker for Vietnam Market

Implements session-level circuit breakers:
- VNINDEX drops more than 5% from opening -> trading halted
- Market-wide circuit breaker based on index movement
- Session-specific risk limits

Author: Trading Bot Team
Version: 1.0.0
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime, time
from enum import Enum
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class SessionBreakLevel(Enum):
    """Circuit breaker levels for market-wide halts"""

    NORMAL = "NORMAL"  # Trading as normal
    WARNING = "WARNING"  # Warning level - reduce activity
    CAUTION = "CAUTION"  # Caution - restrict new positions
    HALT = "HALT"  # Trading halted
    RESUME = "RESUME"  # Trading resuming after halt


@dataclass
class SessionBreakConfig:
    """Configuration for session circuit breaker"""

    # VNINDEX drop thresholds (from session open)
    warning_threshold_pct: float = -2.0  # Warning at -2%
    caution_threshold_pct: float = -3.5  # Caution at -3.5%
    halt_threshold_pct: float = -5.0  # Halt at -5%

    # Recovery thresholds
    resume_threshold_pct: float = -3.0  # Resume when recovers to -3%

    # Time-based rules
    halt_duration_minutes: int = 30  # Minimum halt duration
    max_halts_per_session: int = 2  # Max halts per session

    # Cooldown
    cooldown_after_resume_minutes: int = 15  # Reduced activity after resume


@dataclass
class SessionState:
    """Current session state"""

    level: SessionBreakLevel = SessionBreakLevel.NORMAL
    session_open_price: float = 0.0
    current_price: float = 0.0
    session_low: float = 0.0
    session_high: float = 0.0
    change_pct: float = 0.0

    # Halt tracking
    halt_count: int = 0
    last_halt_time: Optional[datetime] = None
    last_resume_time: Optional[datetime] = None

    # Messages
    status_message: str = ""
    recommendations: List[str] = field(default_factory=list)


class SessionCircuitBreaker:
    """
    Session-level circuit breaker for Vietnam market

    Monitors VNINDEX and triggers circuit breakers:
    - WARNING (-2%): Reduce position sizes, tighten stops
    - CAUTION (-3.5%): No new positions, only exits allowed
    - HALT (-5%): All trading suspended

    Usage:
        breaker = SessionCircuitBreaker()

        # Update with VNINDEX price
        state = breaker.update(vnindex_open=1200, vnindex_current=1164)

        if state.level == SessionBreakLevel.CAUTION:
            # Only allow exits
            pass
        elif state.level == SessionBreakLevel.HALT:
            # All trading suspended
            pass
    """

    def __init__(self, config: Optional[SessionBreakConfig] = None):
        self.config = config or SessionBreakConfig()
        self._state = SessionState()
        self._session_date: Optional[datetime] = None

    @property
    def state(self) -> SessionState:
        """Get current session state"""
        return self._state

    @property
    def is_trading_allowed(self) -> bool:
        """Check if trading is allowed"""
        return self._state.level not in [SessionBreakLevel.HALT]

    @property
    def is_new_position_allowed(self) -> bool:
        """Check if new positions are allowed"""
        return self._state.level in [SessionBreakLevel.NORMAL, SessionBreakLevel.WARNING]

    def reset_session(self, session_open_price: float) -> SessionState:
        """Reset for new trading session"""
        self._state = SessionState(
            level=SessionBreakLevel.NORMAL,
            session_open_price=session_open_price,
            current_price=session_open_price,
            session_low=session_open_price,
            session_high=session_open_price,
            change_pct=0.0,
            status_message="Session started - monitoring active",
        )
        self._session_date = datetime.now()
        logger.info(f"🔄 Session reset - Open: {session_open_price:,.2f}")
        return self._state

    def update(
        self,
        vnindex_current: float,
        vnindex_open: Optional[float] = None,
    ) -> SessionState:
        """
        Update circuit breaker with current VNINDEX price

        Args:
            vnindex_current: Current VNINDEX value
            vnindex_open: Session opening VNINDEX (optional, uses stored value)

        Returns:
            Updated SessionState
        """
        # Update open price if provided
        if vnindex_open is not None:
            self._state.session_open_price = vnindex_open

        # Validate
        if self._state.session_open_price <= 0:
            logger.warning("Session open price not set - call reset_session() first")
            return self._state

        # Update current values
        self._state.current_price = vnindex_current
        self._state.session_low = min(self._state.session_low, vnindex_current)
        self._state.session_high = max(self._state.session_high, vnindex_current)

        # Calculate change
        self._state.change_pct = (
            (vnindex_current - self._state.session_open_price)
            / self._state.session_open_price
            * 100
        )

        # Determine level
        new_level = self._determine_level()

        # Handle state transitions
        if new_level != self._state.level:
            self._handle_transition(new_level)

        # Generate recommendations
        self._state.recommendations = self._get_recommendations()

        return self._state

    def _determine_level(self) -> SessionBreakLevel:
        """Determine circuit breaker level based on current change"""
        change = self._state.change_pct
        current_level = self._state.level

        # Currently halted - check for resume
        if current_level == SessionBreakLevel.HALT:
            if change >= self.config.resume_threshold_pct:
                # Check minimum halt duration
                if self._state.last_halt_time:
                    elapsed = (datetime.now() - self._state.last_halt_time).total_seconds() / 60
                    if elapsed >= self.config.halt_duration_minutes:
                        return SessionBreakLevel.RESUME
                return SessionBreakLevel.HALT
            return SessionBreakLevel.HALT

        # Check for resume cooldown
        if current_level == SessionBreakLevel.RESUME:
            if self._state.last_resume_time:
                elapsed = (datetime.now() - self._state.last_resume_time).total_seconds() / 60
                if elapsed >= self.config.cooldown_after_resume_minutes:
                    # Return to appropriate level based on change
                    pass  # Fall through to normal level determination
                else:
                    return SessionBreakLevel.RESUME

        # Determine level based on change
        if change <= self.config.halt_threshold_pct:
            if self._state.halt_count < self.config.max_halts_per_session:
                return SessionBreakLevel.HALT
            else:
                logger.warning("Max halts reached - staying at CAUTION")
                return SessionBreakLevel.CAUTION
        elif change <= self.config.caution_threshold_pct:
            return SessionBreakLevel.CAUTION
        elif change <= self.config.warning_threshold_pct:
            return SessionBreakLevel.WARNING
        else:
            return SessionBreakLevel.NORMAL

    def _handle_transition(self, new_level: SessionBreakLevel) -> None:
        """Handle state transition"""
        old_level = self._state.level

        if new_level == SessionBreakLevel.HALT:
            self._state.halt_count += 1
            self._state.last_halt_time = datetime.now()
            self._state.status_message = (
                f"🛑 TRADING HALTED - VNINDEX down {self._state.change_pct:.2f}% "
                f"(Halt #{self._state.halt_count})"
            )
            logger.critical(self._state.status_message)

        elif new_level == SessionBreakLevel.RESUME:
            self._state.last_resume_time = datetime.now()
            self._state.status_message = (
                f"🟢 TRADING RESUMING - VNINDEX recovered to {self._state.change_pct:.2f}%"
            )
            logger.warning(self._state.status_message)

        elif new_level == SessionBreakLevel.CAUTION:
            self._state.status_message = (
                f"🟠 CAUTION - VNINDEX down {self._state.change_pct:.2f}% - "
                "New positions restricted"
            )
            logger.warning(self._state.status_message)

        elif new_level == SessionBreakLevel.WARNING:
            self._state.status_message = (
                f"⚠️ WARNING - VNINDEX down {self._state.change_pct:.2f}% - " "Reduce activity"
            )
            logger.warning(self._state.status_message)

        elif new_level == SessionBreakLevel.NORMAL:
            self._state.status_message = "✅ NORMAL - Trading as usual"
            logger.info(self._state.status_message)

        self._state.level = new_level

    def _get_recommendations(self) -> List[str]:
        """Get trading recommendations based on current state"""
        recs = []
        level = self._state.level

        if level == SessionBreakLevel.HALT:
            recs.extend(
                [
                    "🛑 DO NOT place any orders",
                    "Wait for official resume announcement",
                    "Monitor news for market-moving events",
                ]
            )

        elif level == SessionBreakLevel.RESUME:
            recs.extend(
                [
                    "⏳ Cooldown period - limited trading",
                    "Only close existing positions if needed",
                    "Wait for market stabilization",
                ]
            )

        elif level == SessionBreakLevel.CAUTION:
            recs.extend(
                [
                    "❌ DO NOT open new positions",
                    "Review existing positions for exit",
                    "Tighten stop losses by 50%",
                    "Consider protective puts if available",
                ]
            )

        elif level == SessionBreakLevel.WARNING:
            recs.extend(
                [
                    "⚠️ Reduce position sizes by 30-50%",
                    "Increase required confidence to 70+",
                    "Prioritize defensive sectors",
                    "Tighten stop losses",
                ]
            )

        else:  # NORMAL
            recs.append("✅ Normal trading conditions")

        return recs

    def get_position_size_multiplier(self) -> float:
        """Get position size multiplier based on circuit breaker level"""
        multipliers = {
            SessionBreakLevel.NORMAL: 1.0,
            SessionBreakLevel.WARNING: 0.6,
            SessionBreakLevel.CAUTION: 0.0,  # No new positions
            SessionBreakLevel.HALT: 0.0,
            SessionBreakLevel.RESUME: 0.3,
        }
        return multipliers.get(self._state.level, 1.0)

    def get_confidence_threshold_adjustment(self) -> int:
        """Get confidence threshold adjustment based on level"""
        adjustments = {
            SessionBreakLevel.NORMAL: 0,
            SessionBreakLevel.WARNING: 10,
            SessionBreakLevel.CAUTION: 20,  # Much higher bar
            SessionBreakLevel.HALT: 100,  # Effectively block all
            SessionBreakLevel.RESUME: 15,
        }
        return adjustments.get(self._state.level, 0)

    def should_emergency_exit(self, current_pnl_pct: float) -> Tuple[bool, str]:
        """
        Check if position should be emergency exited based on session state

        Args:
            current_pnl_pct: Current position P&L percentage

        Returns:
            Tuple of (should_exit, reason)
        """
        level = self._state.level

        if level == SessionBreakLevel.HALT:
            if current_pnl_pct < -2.0:
                return True, "Market halt + negative P&L - exit immediately when trading resumes"

        elif level == SessionBreakLevel.CAUTION:
            if current_pnl_pct < 0:
                return True, "Caution level + loss - consider exit"
            elif current_pnl_pct > 3.0:
                return True, "Caution level + profit - lock in gains"

        elif level == SessionBreakLevel.WARNING:
            if current_pnl_pct < -3.0:
                return True, "Warning level + significant loss - exit recommended"

        return False, ""

    def get_summary(self) -> Dict:
        """Get summary of current session state"""
        return {
            "level": self._state.level.value,
            "is_trading_allowed": self.is_trading_allowed,
            "is_new_position_allowed": self.is_new_position_allowed,
            "vnindex_open": self._state.session_open_price,
            "vnindex_current": self._state.current_price,
            "change_pct": self._state.change_pct,
            "session_low": self._state.session_low,
            "session_high": self._state.session_high,
            "halt_count": self._state.halt_count,
            "position_size_mult": self.get_position_size_multiplier(),
            "confidence_adjustment": self.get_confidence_threshold_adjustment(),
            "status_message": self._state.status_message,
            "recommendations": self._state.recommendations,
        }


# Singleton instance
_session_breaker: Optional[SessionCircuitBreaker] = None


def get_session_circuit_breaker() -> SessionCircuitBreaker:
    """Get singleton session circuit breaker"""
    global _session_breaker
    if _session_breaker is None:
        _session_breaker = SessionCircuitBreaker()
    return _session_breaker
