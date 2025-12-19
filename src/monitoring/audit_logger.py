# -*- coding: utf-8 -*-
"""
Trading Audit Logger - Complete Trade Audit Trail

Logs all trading activities to a separate audit file for:
- Compliance and review
- Debugging trade issues
- Performance analysis
- Legal record keeping

Author: Trading Bot Team
Version: 1.0.0
"""

import json
import logging
import os
from dataclasses import dataclass, asdict
from datetime import datetime
from enum import Enum
from pathlib import Path
from threading import RLock
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class AuditEventType(Enum):
    """Types of audit events"""

    # Orders
    ORDER_PLACED = "ORDER_PLACED"
    ORDER_FILLED = "ORDER_FILLED"
    ORDER_PARTIAL = "ORDER_PARTIAL"
    ORDER_CANCELLED = "ORDER_CANCELLED"
    ORDER_REJECTED = "ORDER_REJECTED"
    ORDER_TIMEOUT = "ORDER_TIMEOUT"

    # Positions
    POSITION_OPENED = "POSITION_OPENED"
    POSITION_INCREASED = "POSITION_INCREASED"
    POSITION_REDUCED = "POSITION_REDUCED"
    POSITION_CLOSED = "POSITION_CLOSED"

    # Risk events
    STOP_LOSS_HIT = "STOP_LOSS_HIT"
    TAKE_PROFIT_HIT = "TAKE_PROFIT_HIT"
    TRAILING_STOP_HIT = "TRAILING_STOP_HIT"

    # System events
    KILL_SWITCH_ACTIVATED = "KILL_SWITCH_ACTIVATED"
    KILL_SWITCH_RESUMED = "KILL_SWITCH_RESUMED"
    CIRCUIT_BREAKER_TRIGGERED = "CIRCUIT_BREAKER_TRIGGERED"
    EMERGENCY_STOP = "EMERGENCY_STOP"

    # Signals
    SIGNAL_GENERATED = "SIGNAL_GENERATED"
    SIGNAL_EXECUTED = "SIGNAL_EXECUTED"
    SIGNAL_SKIPPED = "SIGNAL_SKIPPED"

    # Session
    SESSION_START = "SESSION_START"
    SESSION_END = "SESSION_END"

    # Errors
    ERROR = "ERROR"
    WARNING = "WARNING"


@dataclass
class AuditEvent:
    """Single audit event"""

    timestamp: str
    event_type: str
    symbol: Optional[str]
    action: Optional[str]  # BUY, SELL, etc.
    details: Dict[str, Any]
    source: str  # Which module generated this
    session_id: str

    def to_dict(self) -> Dict:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False)


class TradingAuditLogger:
    """
    Audit logger for all trading activities

    Features:
    - Writes to dedicated audit log file
    - JSON format for easy parsing
    - Daily rotation
    - Session tracking
    - Thread-safe

    Usage:
        audit = get_audit_logger()

        # Log order
        audit.log_order_placed("VNM", "BUY", 100, 85000, order_id="ORD_001")

        # Log position change
        audit.log_position_opened("VNM", 100, 85000)

        # Log signal
        audit.log_signal("VNM", "BUY", confidence=75, reasons=["RSI oversold"])
    """

    AUDIT_DIR = "logs/audit"

    def __init__(
        self,
        audit_dir: str = None,
        session_id: str = None,
    ):
        self.audit_dir = Path(audit_dir or self.AUDIT_DIR)
        self.audit_dir.mkdir(parents=True, exist_ok=True)

        # Generate session ID if not provided
        self.session_id = session_id or datetime.now().strftime("%Y%m%d_%H%M%S")

        self._lock = RLock()
        self._current_file = None
        self._current_date = None

        # Log session start
        self._log_event(
            event_type=AuditEventType.SESSION_START,
            symbol=None,
            action=None,
            details={"session_id": self.session_id},
            source="AuditLogger",
        )

    def _get_audit_file(self) -> Path:
        """Get current audit file (daily rotation)"""
        today = datetime.now().strftime("%Y%m%d")

        if self._current_date != today:
            self._current_date = today
            self._current_file = self.audit_dir / f"trading_audit_{today}.jsonl"

        return self._current_file

    def _log_event(
        self,
        event_type: AuditEventType,
        symbol: Optional[str],
        action: Optional[str],
        details: Dict[str, Any],
        source: str,
    ):
        """Internal method to log an event"""
        with self._lock:
            event = AuditEvent(
                timestamp=datetime.now().isoformat(),
                event_type=event_type.value,
                symbol=symbol,
                action=action,
                details=details,
                source=source,
                session_id=self.session_id,
            )

            # Write to file
            audit_file = self._get_audit_file()
            try:
                with open(audit_file, "a", encoding="utf-8") as f:
                    f.write(event.to_json() + "\n")
            except Exception as e:
                logger.error(f"Failed to write audit log: {e}")

    # =========================================================================
    # ORDER LOGGING
    # =========================================================================

    def log_order_placed(
        self,
        symbol: str,
        side: str,
        shares: int,
        price: float,
        order_type: str = "MARKET",
        order_id: str = None,
        source: str = "OrderManager",
        **extra,
    ):
        """Log order placed"""
        self._log_event(
            event_type=AuditEventType.ORDER_PLACED,
            symbol=symbol,
            action=side,
            details={
                "shares": shares,
                "price": price,
                "order_type": order_type,
                "order_id": order_id,
                "value": shares * price,
                **extra,
            },
            source=source,
        )

    def log_order_filled(
        self,
        symbol: str,
        side: str,
        shares: int,
        price: float,
        order_id: str = None,
        commission: float = 0,
        slippage: float = 0,
        source: str = "OrderManager",
        **extra,
    ):
        """Log order filled"""
        self._log_event(
            event_type=AuditEventType.ORDER_FILLED,
            symbol=symbol,
            action=side,
            details={
                "shares": shares,
                "price": price,
                "order_id": order_id,
                "value": shares * price,
                "commission": commission,
                "slippage": slippage,
                **extra,
            },
            source=source,
        )

    def log_order_cancelled(
        self,
        symbol: str,
        side: str,
        order_id: str,
        reason: str = "Manual",
        source: str = "OrderManager",
        **extra,
    ):
        """Log order cancelled"""
        self._log_event(
            event_type=AuditEventType.ORDER_CANCELLED,
            symbol=symbol,
            action=side,
            details={
                "order_id": order_id,
                "reason": reason,
                **extra,
            },
            source=source,
        )

    def log_order_rejected(
        self,
        symbol: str,
        side: str,
        shares: int,
        price: float,
        reason: str,
        source: str = "OrderManager",
        **extra,
    ):
        """Log order rejected"""
        self._log_event(
            event_type=AuditEventType.ORDER_REJECTED,
            symbol=symbol,
            action=side,
            details={
                "shares": shares,
                "price": price,
                "reason": reason,
                **extra,
            },
            source=source,
        )

    # =========================================================================
    # POSITION LOGGING
    # =========================================================================

    def log_position_opened(
        self,
        symbol: str,
        shares: int,
        entry_price: float,
        stop_loss: float = None,
        take_profit: float = None,
        source: str = "PortfolioManager",
        **extra,
    ):
        """Log position opened"""
        self._log_event(
            event_type=AuditEventType.POSITION_OPENED,
            symbol=symbol,
            action="OPEN",
            details={
                "shares": shares,
                "entry_price": entry_price,
                "value": shares * entry_price,
                "stop_loss": stop_loss,
                "take_profit": take_profit,
                **extra,
            },
            source=source,
        )

    def log_position_closed(
        self,
        symbol: str,
        shares: int,
        entry_price: float,
        exit_price: float,
        pnl: float,
        pnl_percent: float,
        exit_reason: str,
        holding_days: int = 0,
        source: str = "PortfolioManager",
        **extra,
    ):
        """Log position closed"""
        self._log_event(
            event_type=AuditEventType.POSITION_CLOSED,
            symbol=symbol,
            action="CLOSE",
            details={
                "shares": shares,
                "entry_price": entry_price,
                "exit_price": exit_price,
                "pnl": pnl,
                "pnl_percent": pnl_percent,
                "exit_reason": exit_reason,
                "holding_days": holding_days,
                **extra,
            },
            source=source,
        )

    # =========================================================================
    # SIGNAL LOGGING
    # =========================================================================

    def log_signal(
        self,
        symbol: str,
        signal: str,
        confidence: float,
        reasons: List[str] = None,
        ml_confidence: float = None,
        technical_score: float = None,
        executed: bool = False,
        source: str = "SignalGenerator",
        **extra,
    ):
        """Log signal generated"""
        event_type = AuditEventType.SIGNAL_EXECUTED if executed else AuditEventType.SIGNAL_GENERATED

        self._log_event(
            event_type=event_type,
            symbol=symbol,
            action=signal,
            details={
                "confidence": confidence,
                "ml_confidence": ml_confidence,
                "technical_score": technical_score,
                "reasons": reasons or [],
                **extra,
            },
            source=source,
        )

    def log_signal_skipped(
        self,
        symbol: str,
        signal: str,
        reason: str,
        source: str = "EntryLogic",
        **extra,
    ):
        """Log signal skipped"""
        self._log_event(
            event_type=AuditEventType.SIGNAL_SKIPPED,
            symbol=symbol,
            action=signal,
            details={
                "skip_reason": reason,
                **extra,
            },
            source=source,
        )

    # =========================================================================
    # RISK EVENT LOGGING
    # =========================================================================

    def log_stop_loss(
        self,
        symbol: str,
        shares: int,
        entry_price: float,
        stop_price: float,
        loss: float,
        source: str = "ExitLogic",
        **extra,
    ):
        """Log stop loss hit"""
        self._log_event(
            event_type=AuditEventType.STOP_LOSS_HIT,
            symbol=symbol,
            action="STOP_LOSS",
            details={
                "shares": shares,
                "entry_price": entry_price,
                "stop_price": stop_price,
                "loss": loss,
                **extra,
            },
            source=source,
        )

    def log_take_profit(
        self,
        symbol: str,
        shares: int,
        entry_price: float,
        tp_price: float,
        profit: float,
        tp_level: int = 1,
        source: str = "ExitLogic",
        **extra,
    ):
        """Log take profit hit"""
        self._log_event(
            event_type=AuditEventType.TAKE_PROFIT_HIT,
            symbol=symbol,
            action=f"TAKE_PROFIT_{tp_level}",
            details={
                "shares": shares,
                "entry_price": entry_price,
                "tp_price": tp_price,
                "profit": profit,
                "tp_level": tp_level,
                **extra,
            },
            source=source,
        )

    # =========================================================================
    # SYSTEM EVENT LOGGING
    # =========================================================================

    def log_kill_switch(
        self,
        action: str,  # ACTIVATED, RESUMED
        reason: str,
        positions_affected: int = 0,
        source: str = "KillSwitch",
        **extra,
    ):
        """Log kill switch event"""
        event_type = (
            AuditEventType.KILL_SWITCH_ACTIVATED
            if action == "ACTIVATED"
            else AuditEventType.KILL_SWITCH_RESUMED
        )

        self._log_event(
            event_type=event_type,
            symbol=None,
            action=action,
            details={
                "reason": reason,
                "positions_affected": positions_affected,
                **extra,
            },
            source=source,
        )

    def log_circuit_breaker(
        self,
        trigger_type: str,
        reason: str,
        metrics: Dict = None,
        source: str = "CircuitBreaker",
        **extra,
    ):
        """Log circuit breaker event"""
        self._log_event(
            event_type=AuditEventType.CIRCUIT_BREAKER_TRIGGERED,
            symbol=None,
            action=trigger_type,
            details={
                "reason": reason,
                "metrics": metrics or {},
                **extra,
            },
            source=source,
        )

    def log_error(
        self,
        error_type: str,
        message: str,
        symbol: str = None,
        source: str = "System",
        **extra,
    ):
        """Log error"""
        self._log_event(
            event_type=AuditEventType.ERROR,
            symbol=symbol,
            action=error_type,
            details={
                "message": message,
                **extra,
            },
            source=source,
        )

    def log_warning(
        self,
        warning_type: str,
        message: str,
        symbol: str = None,
        source: str = "System",
        **extra,
    ):
        """Log warning"""
        self._log_event(
            event_type=AuditEventType.WARNING,
            symbol=symbol,
            action=warning_type,
            details={
                "message": message,
                **extra,
            },
            source=source,
        )

    # =========================================================================
    # UTILITY METHODS
    # =========================================================================

    def end_session(self):
        """Log session end"""
        self._log_event(
            event_type=AuditEventType.SESSION_END,
            symbol=None,
            action=None,
            details={"session_id": self.session_id},
            source="AuditLogger",
        )

    def get_today_events(self, event_type: AuditEventType = None) -> List[Dict]:
        """Get today's events"""
        events = []
        audit_file = self._get_audit_file()

        if audit_file.exists():
            try:
                with open(audit_file, "r", encoding="utf-8") as f:
                    for line in f:
                        try:
                            event = json.loads(line.strip())
                            if event_type is None or event.get("event_type") == event_type.value:
                                events.append(event)
                        except json.JSONDecodeError:
                            continue
            except Exception as e:
                logger.error(f"Failed to read audit log: {e}")

        return events

    def get_session_summary(self) -> Dict:
        """Get summary for current session"""
        events = self.get_today_events()
        session_events = [e for e in events if e.get("session_id") == self.session_id]

        # Count by type
        type_counts = {}
        for event in session_events:
            event_type = event.get("event_type", "UNKNOWN")
            type_counts[event_type] = type_counts.get(event_type, 0) + 1

        # Calculate P&L
        total_pnl = 0
        for event in session_events:
            if event.get("event_type") == "POSITION_CLOSED":
                total_pnl += event.get("details", {}).get("pnl", 0)

        return {
            "session_id": self.session_id,
            "total_events": len(session_events),
            "event_counts": type_counts,
            "total_pnl": total_pnl,
        }


# Singleton instance
_audit_logger_instance: Optional[TradingAuditLogger] = None
_audit_logger_lock = RLock()


def get_audit_logger() -> TradingAuditLogger:
    """Get singleton audit logger instance"""
    global _audit_logger_instance

    with _audit_logger_lock:
        if _audit_logger_instance is None:
            _audit_logger_instance = TradingAuditLogger()
        return _audit_logger_instance


def reset_audit_logger():
    """Reset audit logger (for testing)"""
    global _audit_logger_instance
    with _audit_logger_lock:
        if _audit_logger_instance:
            _audit_logger_instance.end_session()
        _audit_logger_instance = None
