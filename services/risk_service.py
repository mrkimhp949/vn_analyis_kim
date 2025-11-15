"""
Risk Management Service
Centralized risk checks and circuit breaker logic
"""

import logging
from typing import Dict, Tuple

from circuit_breaker import get_circuit_breaker
from emergency_stop import get_emergency_stop
from portfolio_manager import get_portfolio_manager

logger = logging.getLogger(__name__)


class RiskManagementService:
    """
    Service for risk management operations

    Responsibilities:
    - Check circuit breaker status
    - Check emergency stop conditions
    - Validate portfolio risk levels
    - Enforce trading limits
    """

    def __init__(self):
        self.circuit_breaker = get_circuit_breaker()
        self.emergency_stop = get_emergency_stop()
        self.portfolio_manager = get_portfolio_manager()

        logger.info("✅ Risk Management Service initialized")

    async def can_trade(self) -> Tuple[bool, str]:
        """
        Check if trading is allowed

        Returns:
            (can_trade, reason) tuple
        """
        # Check 1: Emergency stop
        emergency_ok, emergency_reason = self.emergency_stop.can_trade()
        if not emergency_ok:
            logger.warning(f"Trading blocked by emergency stop: {emergency_reason}")
            return False, emergency_reason

        # Check 2: Circuit breaker
        circuit_ok, circuit_reason = self.circuit_breaker.can_trade()
        if not circuit_ok:
            logger.warning(f"Trading blocked by circuit breaker: {circuit_reason}")
            return False, circuit_reason

        return True, "✅ OK to trade"

    async def check_and_update_circuit_breaker(
        self, portfolio_pnl_pct: float, vnindex_change_pct: float
    ) -> bool:
        """
        Check and update circuit breaker status

        Args:
            portfolio_pnl_pct: Portfolio P&L percentage
            vnindex_change_pct: VNINDEX change percentage

        Returns:
            True if circuit breaker tripped, False otherwise
        """
        tripped = self.circuit_breaker.check_and_update(
            portfolio_pnl_pct=portfolio_pnl_pct, vnindex_change_pct=vnindex_change_pct
        )

        if tripped:
            logger.critical(
                f"🚨 Circuit breaker tripped: {self.circuit_breaker.tripped_reason}"
            )

        return tripped

    def record_trade(self, pnl: float) -> None:
        """Record a trade for circuit breaker tracking"""
        self.circuit_breaker.record_trade(pnl)

    def get_risk_status(self) -> Dict:
        """Get comprehensive risk status"""
        return {
            "circuit_breaker": {
                "tripped": self.circuit_breaker.tripped,
                "reason": self.circuit_breaker.tripped_reason,
                "stats": self.circuit_breaker.get_daily_stats(),
            },
            "emergency_stop": {
                "active": self.emergency_stop.is_emergency_active(),
                "status": self.emergency_stop.get_status_message(),
            },
            "portfolio": {
                "daily_pnl_pct": self.portfolio_manager.get_daily_pnl_pct(),
            },
        }


# Singleton
_risk_service = None


def get_risk_service() -> RiskManagementService:
    """Get risk service singleton"""
    global _risk_service
    if _risk_service is None:
        _risk_service = RiskManagementService()
    return _risk_service
