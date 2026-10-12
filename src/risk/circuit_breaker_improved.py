# -*- coding: utf-8 -*-
"""
Circuit Breaker Improved - Enhanced risk protection with better logging

This module provides improved circuit breaker that:
1. Better error handling and logging
2. Gradual response system with clear thresholds
3. Regime-aware consecutive loss limits
4. Comprehensive statistics tracking

Author: Trading Bot Team
Version: 12.0.0
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field, asdict
from datetime import date, datetime, timedelta
from threading import RLock
from typing import Dict, List, Optional, Tuple, Any
from enum import Enum

logger = logging.getLogger(__name__)


class CircuitBreakerState(Enum):
    """Circuit breaker states."""
    NORMAL = "normal"           # All systems go
    WARNING = "warning"         # Approaching limits
    CAUTION = "caution"         # Reduce position sizes
    TRIPPED = "tripped"         # No new trades


class TripReason(Enum):
    """Reasons for circuit breaker trip."""
    MAX_DAILY_LOSS = "max_daily_loss"
    MAX_TRADES = "max_trades"
    CONSECUTIVE_LOSSES = "consecutive_losses"
    VNINDEX_DROP = "vnindex_drop"
    PORTFOLIO_HEAT = "portfolio_heat"
    MAX_DRAWDOWN = "max_drawdown"
    SESSION_LIMIT = "session_limit"
    MANUAL = "manual"


@dataclass
class CircuitBreakerStats:
    """Daily statistics for circuit breaker."""
    date: str
    trades_count: int = 0
    total_loss: float = 0.0
    total_profit: float = 0.0
    net_pnl: float = 0.0
    consecutive_losses: int = 0
    consecutive_wins: int = 0
    morning_trades: int = 0
    afternoon_trades: int = 0
    last_updated: str = ""
    
    def __post_init__(self):
        if not self.last_updated:
            self.last_updated = datetime.now().isoformat()


@dataclass
class CircuitBreakerConfig:
    """Configuration for circuit breaker - BALANCED v12.0."""
    
    # Trade limits - BALANCED
    max_trades_per_day: int = 8
    max_trades_per_session: int = 4
    
    # Loss limits - BALANCED
    max_loss_per_day_pct: float = 0.03      # 3% max daily loss
    max_drawdown_pct: float = 0.12          # 12% max drawdown
    
    # Consecutive loss limits - REGIME AWARE
    max_consecutive_losses_bull: int = 4     # More lenient in bull
    max_consecutive_losses_sideways: int = 3 # Standard
    max_consecutive_losses_bear: int = 2     # Strict in bear
    max_consecutive_losses_high_vol: int = 2 # Strict in high vol
    
    # VNINDEX thresholds - BALANCED
    vnindex_warning_threshold: float = -0.015   # -1.5% warning
    vnindex_caution_threshold: float = -0.020   # -2.0% caution
    vnindex_trip_threshold: float = -0.025      # -2.5% trip
    
    # Portfolio heat - BALANCED
    max_portfolio_heat: float = 0.60        # 60% max exposure
    
    # Winning streak protection
    max_consecutive_wins: int = 7           # Pause after 7 wins
    
    # Cooldown
    cooldown_minutes: int = 30              # 30 min cooldown after trip
    
    def get_max_consecutive_losses(self, regime: str) -> int:
        """Get regime-specific consecutive loss limit."""
        regime = regime.upper() if regime else "SIDEWAYS"
        
        if regime == "BULL":
            return self.max_consecutive_losses_bull
        elif regime == "BEAR":
            return self.max_consecutive_losses_bear
        elif regime == "HIGH_VOLATILITY":
            return self.max_consecutive_losses_high_vol
        else:
            return self.max_consecutive_losses_sideways


class ImprovedCircuitBreaker:
    """
    Improved Circuit Breaker v12.0 with better logging and error handling.
    
    Key improvements:
    1. Clear state machine (NORMAL -> WARNING -> CAUTION -> TRIPPED)
    2. Regime-aware consecutive loss limits
    3. Comprehensive logging for all decisions
    4. Better statistics tracking
    5. Thread-safe operations
    """
    
    def __init__(
        self,
        config: Optional[CircuitBreakerConfig] = None,
        total_capital: float = 100_000_000,
        stats_file: str = "circuit_breaker_stats.json",
    ):
        """
        Initialize improved circuit breaker.
        
        Args:
            config: Optional configuration, uses defaults if not provided
            total_capital: Total trading capital in VND
            stats_file: Path to stats persistence file
        """
        self.config = config or CircuitBreakerConfig()
        self.total_capital = total_capital
        self.stats_file = stats_file
        
        # State
        self._state = CircuitBreakerState.NORMAL
        self._trip_reason: Optional[TripReason] = None
        self._trip_time: Optional[datetime] = None
        self._current_regime: str = "SIDEWAYS"
        
        # Statistics
        self._stats = self._load_stats()
        self._check_new_day()
        
        # Peak tracking for drawdown
        self._peak_portfolio_value = total_capital
        self._current_drawdown = 0.0
        
        # Thread safety
        self._lock = RLock()
        
        logger.info(
            f"✅ ImprovedCircuitBreaker initialized: "
            f"max_trades={self.config.max_trades_per_day}, "
            f"max_loss={self.config.max_loss_per_day_pct:.1%}"
        )
    
    # =========================================================================
    # STATE MANAGEMENT
    # =========================================================================
    
    @property
    def state(self) -> CircuitBreakerState:
        """Get current circuit breaker state."""
        return self._state
    
    @property
    def is_tripped(self) -> bool:
        """Check if circuit breaker is tripped."""
        return self._state == CircuitBreakerState.TRIPPED
    
    @property
    def is_caution(self) -> bool:
        """Check if in caution mode."""
        return self._state == CircuitBreakerState.CAUTION
    
    @property
    def is_warning(self) -> bool:
        """Check if in warning mode."""
        return self._state == CircuitBreakerState.WARNING
    
    def get_position_multiplier(self) -> float:
        """
        Get position size multiplier based on current state.
        
        Returns:
            1.0 for NORMAL
            0.75 for WARNING
            0.5 for CAUTION
            0.0 for TRIPPED
        """
        if self._state == CircuitBreakerState.TRIPPED:
            return 0.0
        elif self._state == CircuitBreakerState.CAUTION:
            return 0.5
        elif self._state == CircuitBreakerState.WARNING:
            return 0.75
        else:
            return 1.0
    
    def set_regime(self, regime: str) -> None:
        """Set current market regime for adaptive limits."""
        self._current_regime = regime.upper() if regime else "SIDEWAYS"
        logger.debug(f"Circuit breaker regime set to: {self._current_regime}")
    
    # =========================================================================
    # MAIN CHECK METHOD
    # =========================================================================
    
    def check_and_update(
        self,
        portfolio_pnl_pct: float,
        vnindex_change_pct: float,
        portfolio_heat: float = 0.0,
        market_volatility: float = 0.0,
        regime: Optional[str] = None,
    ) -> Tuple[bool, str]:
        """
        Check all circuit breaker conditions and update state.
        
        Args:
            portfolio_pnl_pct: Portfolio P&L as decimal (e.g., -0.02 for -2%)
            vnindex_change_pct: VNINDEX change as decimal
            portfolio_heat: Portfolio exposure (0-1)
            market_volatility: Market volatility
            regime: Optional market regime override
        
        Returns:
            Tuple of (is_tripped, reason_message)
        """
        with self._lock:
            # Update regime if provided
            if regime:
                self._current_regime = regime.upper()
            
            # Check if already tripped
            if self._state == CircuitBreakerState.TRIPPED:
                # Check if cooldown has passed
                if self._check_cooldown_expired():
                    self._reset_trip()
                else:
                    return True, f"Circuit breaker tripped: {self._trip_reason.value if self._trip_reason else 'unknown'}"
            
            # Run all checks
            checks = [
                self._check_daily_loss(portfolio_pnl_pct),
                self._check_vnindex(vnindex_change_pct),
                self._check_trade_count(),
                self._check_consecutive_losses(),
                self._check_portfolio_heat(portfolio_heat),
                self._check_drawdown(portfolio_pnl_pct),
            ]
            
            # Process results
            for is_tripped, reason, message in checks:
                if is_tripped:
                    self._trip(reason, message)
                    return True, message
            
            # Update state based on warnings
            self._update_state_from_checks(vnindex_change_pct, portfolio_pnl_pct)
            
            return False, self._get_status_message()
    
    def _check_daily_loss(self, pnl_pct: float) -> Tuple[bool, Optional[TripReason], str]:
        """Check daily loss limit."""
        if pnl_pct < 0 and abs(pnl_pct) >= self.config.max_loss_per_day_pct:
            msg = (
                f"Daily loss {pnl_pct:.2%} exceeds limit {self.config.max_loss_per_day_pct:.2%}"
            )
            logger.warning(f"🚨 CIRCUIT BREAKER: {msg}")
            return True, TripReason.MAX_DAILY_LOSS, msg
        return False, None, ""
    
    def _check_vnindex(self, change_pct: float) -> Tuple[bool, Optional[TripReason], str]:
        """Check VNINDEX drop threshold."""
        if change_pct <= self.config.vnindex_trip_threshold:
            msg = (
                f"VNINDEX drop {change_pct:.2%} exceeds threshold "
                f"{self.config.vnindex_trip_threshold:.2%}"
            )
            logger.warning(f"🚨 CIRCUIT BREAKER: {msg}")
            return True, TripReason.VNINDEX_DROP, msg
        return False, None, ""
    
    def _check_trade_count(self) -> Tuple[bool, Optional[TripReason], str]:
        """Check daily trade count limit."""
        if self._stats.trades_count >= self.config.max_trades_per_day:
            msg = (
                f"Trade count {self._stats.trades_count} reached limit "
                f"{self.config.max_trades_per_day}"
            )
            logger.warning(f"🚨 CIRCUIT BREAKER: {msg}")
            return True, TripReason.MAX_TRADES, msg
        return False, None, ""
    
    def _check_consecutive_losses(self) -> Tuple[bool, Optional[TripReason], str]:
        """Check consecutive losses with regime-aware limit."""
        max_losses = self.config.get_max_consecutive_losses(self._current_regime)
        
        if self._stats.consecutive_losses >= max_losses:
            msg = (
                f"Consecutive losses {self._stats.consecutive_losses} reached limit "
                f"{max_losses} (regime: {self._current_regime})"
            )
            logger.warning(f"🚨 CIRCUIT BREAKER: {msg}")
            return True, TripReason.CONSECUTIVE_LOSSES, msg
        return False, None, ""
    
    def _check_portfolio_heat(self, heat: float) -> Tuple[bool, Optional[TripReason], str]:
        """Check portfolio heat (exposure)."""
        if heat > self.config.max_portfolio_heat:
            msg = (
                f"Portfolio heat {heat:.1%} exceeds limit "
                f"{self.config.max_portfolio_heat:.1%}"
            )
            logger.warning(f"🚨 CIRCUIT BREAKER: {msg}")
            return True, TripReason.PORTFOLIO_HEAT, msg
        return False, None, ""
    
    def _check_drawdown(self, pnl_pct: float) -> Tuple[bool, Optional[TripReason], str]:
        """Check max drawdown."""
        # Update peak
        current_value = self.total_capital * (1 + pnl_pct)
        if current_value > self._peak_portfolio_value:
            self._peak_portfolio_value = current_value
        
        # Calculate drawdown
        if self._peak_portfolio_value > 0:
            self._current_drawdown = (
                (self._peak_portfolio_value - current_value) / self._peak_portfolio_value
            )
        
        if self._current_drawdown >= self.config.max_drawdown_pct:
            msg = (
                f"Drawdown {self._current_drawdown:.2%} exceeds limit "
                f"{self.config.max_drawdown_pct:.2%}"
            )
            logger.warning(f"🚨 CIRCUIT BREAKER: {msg}")
            return True, TripReason.MAX_DRAWDOWN, msg
        return False, None, ""
    
    def _update_state_from_checks(self, vnindex_pct: float, pnl_pct: float) -> None:
        """Update state based on warning thresholds."""
        # Check VNINDEX levels
        if vnindex_pct <= self.config.vnindex_caution_threshold:
            if self._state != CircuitBreakerState.CAUTION:
                self._state = CircuitBreakerState.CAUTION
                logger.warning(
                    f"⚠️ CAUTION MODE: VNINDEX {vnindex_pct:.2%} "
                    f"(threshold: {self.config.vnindex_caution_threshold:.2%})"
                )
        elif vnindex_pct <= self.config.vnindex_warning_threshold:
            if self._state == CircuitBreakerState.NORMAL:
                self._state = CircuitBreakerState.WARNING
                logger.info(
                    f"⚠️ WARNING MODE: VNINDEX {vnindex_pct:.2%} "
                    f"(threshold: {self.config.vnindex_warning_threshold:.2%})"
                )
        else:
            # Check if we can return to normal
            if self._state in (CircuitBreakerState.WARNING, CircuitBreakerState.CAUTION):
                self._state = CircuitBreakerState.NORMAL
                logger.info("✅ Returned to NORMAL mode")
    
    # =========================================================================
    # TRADE RECORDING
    # =========================================================================
    
    def record_trade(self, pnl: float, is_win: Optional[bool] = None) -> None:
        """
        Record a completed trade.
        
        Args:
            pnl: Profit/Loss amount in VND
            is_win: Optional explicit win/loss flag
        """
        with self._lock:
            self._check_new_day()
            
            # Update counts
            self._stats.trades_count += 1
            
            # Update P&L
            if pnl >= 0:
                self._stats.total_profit += pnl
                self._stats.consecutive_losses = 0
                self._stats.consecutive_wins += 1
            else:
                self._stats.total_loss += abs(pnl)
                self._stats.consecutive_losses += 1
                self._stats.consecutive_wins = 0
            
            self._stats.net_pnl = self._stats.total_profit - self._stats.total_loss
            self._stats.last_updated = datetime.now().isoformat()
            
            # Update session trades
            session = self._get_current_session()
            if session == "morning":
                self._stats.morning_trades += 1
            elif session == "afternoon":
                self._stats.afternoon_trades += 1
            
            # Save stats
            self._save_stats()
            
            # Log
            win_loss = "WIN" if pnl >= 0 else "LOSS"
            logger.info(
                f"📊 Trade recorded: {win_loss} {pnl:+,.0f} VND | "
                f"Day: {self._stats.trades_count} trades, {self._stats.net_pnl:+,.0f} net | "
                f"Streak: {self._stats.consecutive_wins}W / {self._stats.consecutive_losses}L"
            )
    
    # =========================================================================
    # SESSION MANAGEMENT
    # =========================================================================
    
    def check_session_limit(self) -> Tuple[bool, str]:
        """
        Check if current session trade limit reached.
        
        Returns:
            Tuple of (can_trade, message)
        """
        session = self._get_current_session()
        
        if session == "closed":
            return False, "Market is closed"
        
        if session == "morning":
            trades = self._stats.morning_trades
        else:
            trades = self._stats.afternoon_trades
        
        if trades >= self.config.max_trades_per_session:
            return False, f"Session limit reached: {trades}/{self.config.max_trades_per_session}"
        
        remaining = self.config.max_trades_per_session - trades
        return True, f"Session OK: {trades}/{self.config.max_trades_per_session} ({remaining} remaining)"
    
    def _get_current_session(self) -> str:
        """Get current trading session."""
        try:
            from datetime import datetime
            import pytz
            vn_tz = pytz.timezone("Asia/Ho_Chi_Minh")
            now = datetime.now(vn_tz)
        except ImportError:
            now = datetime.now()
        
        hour, minute = now.hour, now.minute
        time_mins = hour * 60 + minute
        
        # Morning: 9:00 - 11:30
        if 540 <= time_mins <= 690:
            return "morning"
        # Afternoon: 13:00 - 14:45
        elif 780 <= time_mins <= 885:
            return "afternoon"
        else:
            return "closed"
    
    # =========================================================================
    # STATISTICS & PERSISTENCE
    # =========================================================================
    
    def get_stats(self) -> Dict[str, Any]:
        """Get current statistics."""
        return {
            "state": self._state.value,
            "regime": self._current_regime,
            "position_multiplier": self.get_position_multiplier(),
            "trades_today": self._stats.trades_count,
            "max_trades": self.config.max_trades_per_day,
            "net_pnl": self._stats.net_pnl,
            "consecutive_losses": self._stats.consecutive_losses,
            "consecutive_wins": self._stats.consecutive_wins,
            "max_consecutive_losses": self.config.get_max_consecutive_losses(self._current_regime),
            "current_drawdown": self._current_drawdown,
            "morning_trades": self._stats.morning_trades,
            "afternoon_trades": self._stats.afternoon_trades,
            "trip_reason": self._trip_reason.value if self._trip_reason else None,
        }
    
    def _load_stats(self) -> CircuitBreakerStats:
        """Load stats from file."""
        if os.path.exists(self.stats_file):
            try:
                with open(self.stats_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    return CircuitBreakerStats(**data.get("today", {}))
            except Exception as e:
                logger.warning(f"Failed to load stats: {e}")
        
        return CircuitBreakerStats(date=date.today().isoformat())
    
    def _save_stats(self) -> None:
        """Save stats to file."""
        try:
            with open(self.stats_file, "w", encoding="utf-8") as f:
                json.dump({"today": asdict(self._stats)}, f, indent=2)
        except Exception as e:
            logger.warning(f"Failed to save stats: {e}")
    
    def _check_new_day(self) -> None:
        """Check if new day and reset stats."""
        today = date.today().isoformat()
        if self._stats.date != today:
            logger.info(f"📅 New trading day: {today}")
            self._stats = CircuitBreakerStats(date=today)
            self._reset_trip()
            self._save_stats()
    
    # =========================================================================
    # TRIP MANAGEMENT
    # =========================================================================
    
    def _trip(self, reason: TripReason, message: str) -> None:
        """Trip the circuit breaker."""
        self._state = CircuitBreakerState.TRIPPED
        self._trip_reason = reason
        self._trip_time = datetime.now()
        
        logger.error(f"🚨 CIRCUIT BREAKER TRIPPED: {message}")
        self._save_stats()
    
    def _reset_trip(self) -> None:
        """Reset circuit breaker from tripped state."""
        if self._state == CircuitBreakerState.TRIPPED:
            logger.info("✅ Circuit breaker reset")
        
        self._state = CircuitBreakerState.NORMAL
        self._trip_reason = None
        self._trip_time = None
    
    def _check_cooldown_expired(self) -> bool:
        """Check if cooldown period has expired."""
        if self._trip_time is None:
            return True
        
        cooldown = timedelta(minutes=self.config.cooldown_minutes)
        return datetime.now() > self._trip_time + cooldown
    
    def manual_reset(self) -> None:
        """Manually reset circuit breaker (use with caution)."""
        logger.warning("⚠️ Manual circuit breaker reset")
        self._reset_trip()
    
    def _get_status_message(self) -> str:
        """Get current status message."""
        return (
            f"State: {self._state.value} | "
            f"Trades: {self._stats.trades_count}/{self.config.max_trades_per_day} | "
            f"Losses: {self._stats.consecutive_losses}/{self.config.get_max_consecutive_losses(self._current_regime)}"
        )


# =============================================================================
# SINGLETON
# =============================================================================

_improved_circuit_breaker: Optional[ImprovedCircuitBreaker] = None


def get_improved_circuit_breaker(
    config: Optional[CircuitBreakerConfig] = None,
    force_new: bool = False,
) -> ImprovedCircuitBreaker:
    """Get singleton instance of improved circuit breaker."""
    global _improved_circuit_breaker
    
    if _improved_circuit_breaker is None or force_new:
        _improved_circuit_breaker = ImprovedCircuitBreaker(config=config)
    
    return _improved_circuit_breaker
