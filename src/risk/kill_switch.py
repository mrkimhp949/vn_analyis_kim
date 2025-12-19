# -*- coding: utf-8 -*-
"""
Kill Switch - Emergency Trading Stop

Provides immediate ability to:
- Stop all trading activities
- Close all positions (market orders)
- Prevent any new orders

Critical for live/paper trading safety.

Author: Trading Bot Team
Version: 1.0.0
"""

import json
import logging
import os
from dataclasses import dataclass, asdict
from datetime import datetime
from enum import Enum
from threading import RLock
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class KillSwitchState(Enum):
    """Kill switch states"""

    ACTIVE = "ACTIVE"  # Trading allowed
    PAUSED = "PAUSED"  # Trading paused (can resume)
    KILLED = "KILLED"  # All trading stopped, positions being closed


@dataclass
class KillSwitchEvent:
    """Kill switch event record"""

    timestamp: str
    action: str  # ACTIVATE, PAUSE, RESUME, KILL
    reason: str
    triggered_by: str  # MANUAL, AUTO, SYSTEM
    positions_affected: int = 0


class KillSwitch:
    """
    Emergency Kill Switch for Trading Bot

    Features:
    - Instant trading halt
    - Force close all positions
    - Audit trail of all actions
    - Thread-safe operations
    - Persistent state across restarts

    Usage:
        kill_switch = get_kill_switch()

        # Check before any trade
        if not kill_switch.can_trade():
            return "Trading is disabled"

        # Emergency stop
        kill_switch.kill_all("Market crash detected")

        # Pause trading
        kill_switch.pause("Taking a break")

        # Resume
        kill_switch.resume("Back to normal")
    """

    STATE_FILE = "kill_switch_state.json"

    def __init__(
        self,
        state_file: str = None,
        auto_close_on_kill: bool = True,
    ):
        self.state_file = state_file or self.STATE_FILE
        self.auto_close_on_kill = auto_close_on_kill
        self._lock = RLock()

        # Load or initialize state
        self._state = KillSwitchState.ACTIVE
        self._kill_reason: Optional[str] = None
        self._events: List[Dict] = []
        self._positions_to_close: List[str] = []

        self._load_state()

    def _load_state(self):
        """Load state from file"""
        if os.path.exists(self.state_file):
            try:
                with open(self.state_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    self._state = KillSwitchState(data.get("state", "ACTIVE"))
                    self._kill_reason = data.get("kill_reason")
                    self._events = data.get("events", [])[-100:]  # Keep last 100

                    # If was KILLED, stay KILLED until manual resume
                    if self._state == KillSwitchState.KILLED:
                        logger.warning(
                            "⚠️ Kill switch was KILLED - trading disabled until manual resume"
                        )
            except Exception as e:
                logger.error(f"Failed to load kill switch state: {e}")

    def _save_state(self):
        """Save state to file"""
        try:
            with open(self.state_file, "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "state": self._state.value,
                        "kill_reason": self._kill_reason,
                        "events": self._events[-100:],  # Keep last 100
                        "last_updated": datetime.now().isoformat(),
                    },
                    f,
                    indent=2,
                    ensure_ascii=False,
                )
        except Exception as e:
            logger.error(f"Failed to save kill switch state: {e}")

    def _record_event(
        self, action: str, reason: str, triggered_by: str, positions_affected: int = 0
    ):
        """Record an event"""
        event = KillSwitchEvent(
            timestamp=datetime.now().isoformat(),
            action=action,
            reason=reason,
            triggered_by=triggered_by,
            positions_affected=positions_affected,
        )
        self._events.append(asdict(event))
        self._save_state()

    def can_trade(self) -> Tuple[bool, str]:
        """
        Check if trading is allowed

        Returns:
            (can_trade, reason)
        """
        with self._lock:
            if self._state == KillSwitchState.ACTIVE:
                return True, "Trading active"
            elif self._state == KillSwitchState.PAUSED:
                return False, f"Trading paused: {self._kill_reason or 'No reason'}"
            else:  # KILLED
                return False, f"🚨 KILL SWITCH ACTIVE: {self._kill_reason or 'Emergency stop'}"

    def is_active(self) -> bool:
        """Check if trading is active"""
        return self._state == KillSwitchState.ACTIVE

    def is_killed(self) -> bool:
        """Check if kill switch has been triggered"""
        return self._state == KillSwitchState.KILLED

    def pause(self, reason: str = "Manual pause", triggered_by: str = "MANUAL") -> bool:
        """
        Pause trading (can be resumed)

        Args:
            reason: Why trading is being paused
            triggered_by: Who triggered (MANUAL, AUTO, SYSTEM)

        Returns:
            True if state changed
        """
        with self._lock:
            if self._state == KillSwitchState.KILLED:
                logger.warning("Cannot pause - kill switch is active")
                return False

            prev_state = self._state
            self._state = KillSwitchState.PAUSED
            self._kill_reason = reason

            self._record_event("PAUSE", reason, triggered_by)

            logger.warning(f"⏸️ Trading PAUSED: {reason}")
            return prev_state != KillSwitchState.PAUSED

    def resume(self, reason: str = "Manual resume", triggered_by: str = "MANUAL") -> bool:
        """
        Resume trading

        Args:
            reason: Why trading is being resumed
            triggered_by: Who triggered

        Returns:
            True if state changed
        """
        with self._lock:
            prev_state = self._state
            self._state = KillSwitchState.ACTIVE
            self._kill_reason = None

            self._record_event("RESUME", reason, triggered_by)

            logger.info(f"▶️ Trading RESUMED: {reason}")
            return prev_state != KillSwitchState.ACTIVE

    def kill_all(
        self,
        reason: str = "Emergency stop",
        triggered_by: str = "MANUAL",
        close_positions: bool = True,
    ) -> Dict:
        """
        EMERGENCY: Kill all trading and optionally close positions

        Args:
            reason: Why kill switch is triggered
            triggered_by: Who triggered (MANUAL, AUTO, SYSTEM)
            close_positions: Whether to close all positions

        Returns:
            Dict with results
        """
        with self._lock:
            self._state = KillSwitchState.KILLED
            self._kill_reason = reason

            positions_closed = 0
            close_results = []

            if close_positions and self.auto_close_on_kill:
                # Get portfolio manager and close all positions
                try:
                    from src.portfolio.manager import get_portfolio_manager
                    from src.portfolio.paper_trading import get_paper_account

                    pm = get_portfolio_manager()
                    paper = get_paper_account()

                    # Get all positions
                    positions = pm.get_active_positions()

                    for symbol, pos_data in positions.items():
                        try:
                            # Get current price (approximate)
                            current_price = pos_data.get(
                                "current_price", pos_data.get("avg_price", 0)
                            )
                            shares = pos_data.get("shares", 0)

                            if shares > 0:
                                # Execute emergency sell
                                success, msg, _ = paper.execute_sell(
                                    symbol=symbol,
                                    shares=shares,
                                    price=current_price,
                                    signal_reason=f"KILL SWITCH: {reason}",
                                )
                                close_results.append(
                                    {
                                        "symbol": symbol,
                                        "shares": shares,
                                        "success": success,
                                        "message": msg,
                                    }
                                )
                                if success:
                                    positions_closed += 1
                        except Exception as e:
                            close_results.append(
                                {
                                    "symbol": symbol,
                                    "success": False,
                                    "error": str(e),
                                }
                            )

                except Exception as e:
                    logger.error(f"Error closing positions: {e}")

            self._record_event("KILL", reason, triggered_by, positions_closed)

            logger.critical(
                f"🚨 KILL SWITCH ACTIVATED: {reason} | Positions closed: {positions_closed}"
            )

            return {
                "state": self._state.value,
                "reason": reason,
                "positions_closed": positions_closed,
                "close_results": close_results,
                "timestamp": datetime.now().isoformat(),
            }

    def get_status(self) -> Dict:
        """Get current kill switch status"""
        with self._lock:
            return {
                "state": self._state.value,
                "can_trade": self._state == KillSwitchState.ACTIVE,
                "reason": self._kill_reason,
                "recent_events": self._events[-10:],
                "last_updated": datetime.now().isoformat(),
            }

    def get_events(self, limit: int = 50) -> List[Dict]:
        """Get recent events"""
        return self._events[-limit:]


# Singleton instance
_kill_switch_instance: Optional[KillSwitch] = None
_kill_switch_lock = RLock()


def get_kill_switch() -> KillSwitch:
    """Get singleton kill switch instance"""
    global _kill_switch_instance

    with _kill_switch_lock:
        if _kill_switch_instance is None:
            _kill_switch_instance = KillSwitch()
        return _kill_switch_instance


def reset_kill_switch():
    """Reset kill switch (for testing)"""
    global _kill_switch_instance
    with _kill_switch_lock:
        _kill_switch_instance = None
