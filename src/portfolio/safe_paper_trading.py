# -*- coding: utf-8 -*-
"""
Safe Paper Trading - Paper Trading with Full Safety Guards

Integrates all safety modules:
- Kill Switch
- Order Guard (duplicate prevention)
- Audit Logger
- Position Reconciliation
- Circuit Breaker

Author: Trading Bot Team
Version: 1.0.0
"""

import logging
from dataclasses import dataclass
from datetime import datetime
from threading import RLock
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


@dataclass
class SafetyCheckResult:
    """Result of safety checks"""

    can_proceed: bool
    checks_passed: List[str]
    checks_failed: List[str]
    warnings: List[str]

    @property
    def message(self) -> str:
        if self.can_proceed:
            return "All safety checks passed"
        return f"Safety check failed: {', '.join(self.checks_failed)}"


class SafePaperTrading:
    """
    Safe Paper Trading wrapper with all safety guards

    Features:
    - Pre-trade safety checks
    - Kill switch integration
    - Duplicate order prevention
    - Full audit trail
    - Position reconciliation
    - Circuit breaker respect

    Usage:
        safe_trader = get_safe_paper_trader()

        # Execute buy with safety checks
        result = safe_trader.safe_buy(
            symbol="VNM",
            shares=100,
            price=85000,
            signal_confidence=75,
        )

        # Force sell (emergency)
        safe_trader.emergency_sell_all()

        # Check status
        status = safe_trader.get_safety_status()
    """

    def __init__(self):
        self._lock = RLock()
        self._initialized = False

        # Lazy load components
        self._kill_switch = None
        self._order_guard = None
        self._audit_logger = None
        self._reconciler = None
        self._circuit_breaker = None
        self._paper_account = None

    def _ensure_initialized(self):
        """Lazy initialization of components"""
        if self._initialized:
            return

        try:
            from src.risk.kill_switch import get_kill_switch

            self._kill_switch = get_kill_switch()
        except ImportError as e:
            logger.warning(f"Kill switch not available: {e}")

        try:
            from src.risk.order_guard import get_order_guard

            self._order_guard = get_order_guard()
        except ImportError as e:
            logger.warning(f"Order guard not available: {e}")

        try:
            from src.monitoring.audit_logger import get_audit_logger

            self._audit_logger = get_audit_logger()
        except ImportError as e:
            logger.warning(f"Audit logger not available: {e}")

        try:
            from src.portfolio.reconciliation import get_position_reconciler

            self._reconciler = get_position_reconciler()
        except ImportError as e:
            logger.warning(f"Position reconciler not available: {e}")

        try:
            from src.risk.circuit_breaker import get_circuit_breaker

            self._circuit_breaker = get_circuit_breaker()
        except ImportError as e:
            logger.warning(f"Circuit breaker not available: {e}")

        try:
            from src.portfolio.paper_trading import get_paper_account

            self._paper_account = get_paper_account()
        except ImportError as e:
            logger.error(f"Paper account not available: {e}")
            raise

        self._initialized = True
        logger.info("✅ SafePaperTrading initialized with all safety guards")

    def _run_safety_checks(
        self,
        symbol: str,
        side: str,
        shares: int,
    ) -> SafetyCheckResult:
        """
        Run all safety checks before trading

        Returns:
            SafetyCheckResult
        """
        self._ensure_initialized()

        checks_passed = []
        checks_failed = []
        warnings = []

        # Check 1: Kill Switch
        if self._kill_switch:
            can_trade, reason = self._kill_switch.can_trade()
            if can_trade:
                checks_passed.append("kill_switch")
            else:
                checks_failed.append(f"kill_switch: {reason}")
        else:
            warnings.append("Kill switch not available")

        # Check 2: Circuit Breaker
        if self._circuit_breaker:
            cb_ok, cb_reason = self._circuit_breaker.can_trade()
            if cb_ok:
                checks_passed.append("circuit_breaker")
            else:
                checks_failed.append(f"circuit_breaker: {cb_reason}")
        else:
            warnings.append("Circuit breaker not available")

        # Check 3: Order Guard (duplicate prevention)
        if self._order_guard:
            can_place, og_reason = self._order_guard.can_place_order(symbol, side, shares)
            if can_place:
                checks_passed.append("order_guard")
            else:
                checks_failed.append(f"order_guard: {og_reason}")
        else:
            warnings.append("Order guard not available")

        # Check 4: Position Reconciliation (only for buys)
        if self._reconciler and side == "BUY":
            can_trade_safe, recon_reason = self._reconciler.can_trade_safely()
            if can_trade_safe:
                checks_passed.append("reconciliation")
            else:
                # Only warn, don't block
                warnings.append(f"reconciliation: {recon_reason}")
                checks_passed.append("reconciliation (warning)")

        can_proceed = len(checks_failed) == 0

        return SafetyCheckResult(
            can_proceed=can_proceed,
            checks_passed=checks_passed,
            checks_failed=checks_failed,
            warnings=warnings,
        )

    def safe_buy(
        self,
        symbol: str,
        shares: int,
        price: float,
        signal_confidence: Optional[float] = None,
        signal_reason: Optional[str] = None,
        stop_loss: Optional[float] = None,
        take_profit: Optional[float] = None,
        metadata: Optional[Dict] = None,
        bypass_checks: bool = False,
    ) -> Tuple[bool, str, Optional[Any]]:
        """
        Execute buy with safety checks

        Args:
            symbol: Stock symbol
            shares: Number of shares
            price: Price per share
            signal_confidence: ML/Technical confidence
            signal_reason: Why buying
            stop_loss: Stop loss price
            take_profit: Take profit price
            metadata: Additional metadata
            bypass_checks: Skip safety checks (dangerous!)

        Returns:
            (success, message, trade)
        """
        with self._lock:
            self._ensure_initialized()

            # Run safety checks
            if not bypass_checks:
                safety = self._run_safety_checks(symbol, "BUY", shares)
                if not safety.can_proceed:
                    if self._audit_logger:
                        self._audit_logger.log_order_rejected(
                            symbol=symbol,
                            side="BUY",
                            shares=shares,
                            price=price,
                            reason=safety.message,
                            source="SafePaperTrading",
                        )
                    return False, safety.message, None

            # Register order with guard
            pending_order = None
            if self._order_guard:
                try:
                    pending_order = self._order_guard.register_order(
                        symbol=symbol,
                        side="BUY",
                        shares=shares,
                        price=price,
                        order_type="MARKET",
                    )
                except Exception as e:
                    return False, str(e), None

            # Log order placed
            if self._audit_logger:
                self._audit_logger.log_order_placed(
                    symbol=symbol,
                    side="BUY",
                    shares=shares,
                    price=price,
                    order_id=pending_order.order_id if pending_order else None,
                    source="SafePaperTrading",
                    signal_confidence=signal_confidence,
                )

            # Execute via paper trading
            try:
                success, message, trade = self._paper_account.execute_buy(
                    symbol=symbol,
                    shares=shares,
                    price=price,
                    signal_confidence=signal_confidence,
                    signal_reason=signal_reason,
                    stop_loss=stop_loss,
                    take_profit=take_profit,
                    metadata=metadata,
                )

                # Update order status
                if self._order_guard and pending_order:
                    from src.risk.order_guard import PendingOrderStatus

                    self._order_guard.update_order_status(
                        pending_order.order_id,
                        PendingOrderStatus.FILLED if success else PendingOrderStatus.REJECTED,
                        filled_shares=shares if success else 0,
                        filled_price=price if success else 0,
                    )

                # Log result
                if self._audit_logger:
                    if success:
                        self._audit_logger.log_order_filled(
                            symbol=symbol,
                            side="BUY",
                            shares=shares,
                            price=price,
                            order_id=pending_order.order_id if pending_order else None,
                            source="SafePaperTrading",
                        )
                        self._audit_logger.log_position_opened(
                            symbol=symbol,
                            shares=shares,
                            entry_price=price,
                            stop_loss=stop_loss,
                            take_profit=take_profit,
                            source="SafePaperTrading",
                        )
                    else:
                        self._audit_logger.log_order_rejected(
                            symbol=symbol,
                            side="BUY",
                            shares=shares,
                            price=price,
                            reason=message,
                            source="SafePaperTrading",
                        )

                return success, message, trade

            except Exception as e:
                logger.error(f"Error executing buy: {e}")
                if self._audit_logger:
                    self._audit_logger.log_error(
                        error_type="BUY_EXECUTION_ERROR",
                        message=str(e),
                        symbol=symbol,
                        source="SafePaperTrading",
                    )
                return False, str(e), None

    def safe_sell(
        self,
        symbol: str,
        shares: int,
        price: float,
        signal_reason: Optional[str] = None,
        exit_reason: str = "Manual",
        bypass_checks: bool = False,
    ) -> Tuple[bool, str, Optional[Any]]:
        """
        Execute sell with safety checks

        Args:
            symbol: Stock symbol
            shares: Number of shares to sell
            price: Price per share
            signal_reason: Why selling
            exit_reason: Exit reason type
            bypass_checks: Skip safety checks

        Returns:
            (success, message, trade)
        """
        with self._lock:
            self._ensure_initialized()

            # Run safety checks (sell always allowed if kill switch not active)
            if not bypass_checks:
                # Only check kill switch for sells
                if self._kill_switch:
                    can_trade, reason = self._kill_switch.can_trade()
                    if not can_trade and not self._kill_switch.is_killed():
                        # Paused but not killed - block sell
                        return False, f"Trading paused: {reason}", None
                    # If killed, allow sells (closing positions)

            # Register order
            pending_order = None
            if self._order_guard:
                try:
                    pending_order = self._order_guard.register_order(
                        symbol=symbol,
                        side="SELL",
                        shares=shares,
                        price=price,
                        order_type="MARKET",
                    )
                except Exception as e:
                    # For sells, we might want to proceed anyway
                    logger.warning(f"Order guard warning (proceeding): {e}")

            # Log order placed
            if self._audit_logger:
                self._audit_logger.log_order_placed(
                    symbol=symbol,
                    side="SELL",
                    shares=shares,
                    price=price,
                    order_id=pending_order.order_id if pending_order else None,
                    source="SafePaperTrading",
                )

            # Execute via paper trading
            try:
                success, message, trade = self._paper_account.execute_sell(
                    symbol=symbol,
                    shares=shares,
                    price=price,
                    signal_reason=signal_reason or exit_reason,
                )

                # Update order status
                if self._order_guard and pending_order:
                    from src.risk.order_guard import PendingOrderStatus

                    self._order_guard.update_order_status(
                        pending_order.order_id,
                        PendingOrderStatus.FILLED if success else PendingOrderStatus.REJECTED,
                        filled_shares=shares if success else 0,
                        filled_price=price if success else 0,
                    )

                # Log result
                if self._audit_logger and success:
                    self._audit_logger.log_order_filled(
                        symbol=symbol,
                        side="SELL",
                        shares=shares,
                        price=price,
                        order_id=pending_order.order_id if pending_order else None,
                        source="SafePaperTrading",
                    )

                return success, message, trade

            except Exception as e:
                logger.error(f"Error executing sell: {e}")
                return False, str(e), None

    def emergency_sell_all(self, reason: str = "Emergency sell all") -> Dict:
        """
        Emergency: Sell all positions immediately

        Returns:
            Dict with results
        """
        with self._lock:
            self._ensure_initialized()

            results = {
                "success": True,
                "positions_sold": 0,
                "total_value": 0,
                "errors": [],
            }

            # Log emergency
            if self._audit_logger:
                self._audit_logger.log_kill_switch(
                    action="ACTIVATED",
                    reason=reason,
                    source="SafePaperTrading",
                )

            try:
                from src.portfolio.manager import get_portfolio_manager

                pm = get_portfolio_manager()
                positions = pm.get_active_positions()

                for symbol, pos_data in positions.items():
                    shares = pos_data.get("shares", 0)
                    current_price = pos_data.get("current_price", pos_data.get("avg_price", 0))

                    if shares > 0 and current_price > 0:
                        success, msg, _ = self.safe_sell(
                            symbol=symbol,
                            shares=shares,
                            price=current_price,
                            exit_reason=f"EMERGENCY: {reason}",
                            bypass_checks=True,
                        )

                        if success:
                            results["positions_sold"] += 1
                            results["total_value"] += shares * current_price
                        else:
                            results["errors"].append(f"{symbol}: {msg}")
                            results["success"] = False

            except Exception as e:
                results["success"] = False
                results["errors"].append(str(e))

            return results

    def pause_trading(self, reason: str = "Manual pause") -> bool:
        """Pause all trading"""
        self._ensure_initialized()
        if self._kill_switch:
            return self._kill_switch.pause(reason)
        return False

    def resume_trading(self, reason: str = "Manual resume") -> bool:
        """Resume trading"""
        self._ensure_initialized()
        if self._kill_switch:
            return self._kill_switch.resume(reason)
        return False

    def kill_trading(self, reason: str = "Kill switch activated") -> Dict:
        """Activate kill switch and close all positions"""
        self._ensure_initialized()

        if self._kill_switch:
            return self._kill_switch.kill_all(reason, close_positions=True)
        else:
            return self.emergency_sell_all(reason)

    def get_safety_status(self) -> Dict:
        """Get comprehensive safety status"""
        self._ensure_initialized()

        status = {
            "timestamp": datetime.now().isoformat(),
            "can_trade": True,
            "reasons": [],
        }

        # Kill switch status
        if self._kill_switch:
            ks_status = self._kill_switch.get_status()
            status["kill_switch"] = ks_status
            if not ks_status["can_trade"]:
                status["can_trade"] = False
                status["reasons"].append(f"Kill switch: {ks_status['reason']}")

        # Circuit breaker status
        if self._circuit_breaker:
            cb_ok, cb_reason = self._circuit_breaker.can_trade()
            status["circuit_breaker"] = {
                "can_trade": cb_ok,
                "reason": cb_reason if not cb_ok else None,
            }
            if not cb_ok:
                status["can_trade"] = False
                status["reasons"].append(f"Circuit breaker: {cb_reason}")

        # Order guard status
        if self._order_guard:
            status["order_guard"] = self._order_guard.get_statistics()

        # Reconciliation status
        if self._reconciler:
            status["reconciliation"] = self._reconciler.get_status()

        # Paper account status
        if self._paper_account:
            status["paper_account"] = {
                "cash": self._paper_account.account.get("cash", 0),
                "initial_capital": self._paper_account.account.get("initial_capital", 0),
            }

        return status

    def run_health_check(self) -> Tuple[bool, List[str]]:
        """
        Run comprehensive health check

        Returns:
            (all_healthy, issues)
        """
        self._ensure_initialized()

        issues = []

        # Check kill switch
        if self._kill_switch:
            if self._kill_switch.is_killed():
                issues.append("Kill switch is KILLED - trading disabled")
        else:
            issues.append("Kill switch not available")

        # Check circuit breaker
        if self._circuit_breaker:
            cb_ok, cb_reason = self._circuit_breaker.can_trade()
            if not cb_ok:
                issues.append(f"Circuit breaker: {cb_reason}")
        else:
            issues.append("Circuit breaker not available")

        # Check reconciliation
        if self._reconciler:
            has_critical, critical = self._reconciler.has_critical_mismatches()
            if has_critical:
                issues.append(f"Critical position mismatches: {len(critical)}")

        # Check paper account
        if self._paper_account:
            cash = self._paper_account.account.get("cash", 0)
            if cash <= 0:
                issues.append("Paper account has no cash")
        else:
            issues.append("Paper account not available")

        all_healthy = len(issues) == 0

        return all_healthy, issues


# Singleton instance
_safe_paper_trading_instance: Optional[SafePaperTrading] = None
_safe_paper_trading_lock = RLock()


def get_safe_paper_trader() -> SafePaperTrading:
    """Get singleton safe paper trading instance"""
    global _safe_paper_trading_instance

    with _safe_paper_trading_lock:
        if _safe_paper_trading_instance is None:
            _safe_paper_trading_instance = SafePaperTrading()
        return _safe_paper_trading_instance


def reset_safe_paper_trading():
    """Reset safe paper trading (for testing)"""
    global _safe_paper_trading_instance
    with _safe_paper_trading_lock:
        _safe_paper_trading_instance = None
