# -*- coding: utf-8 -*-
"""
Odd-Lot Trading Handler for Vietnam Market

Handles trading of shares less than standard lot size (100 shares).
Odd-lot trading was enabled on Vietnam stock market since 2021.

Key features:
- Odd-lot validation
- Spread premium calculation
- Commission calculation (minimum commission applies)
- Position reconciliation

Author: Trading Bot Team
Version: 1.0.0
"""

import logging
from dataclasses import dataclass
from typing import Optional, Tuple

from src.config.constants import (
    VIETNAM_LOT_SIZE,
    VN_ODD_LOT_ENABLED,
    VN_ODD_LOT_MIN_QTY,
    VN_ODD_LOT_MAX_QTY,
    VN_ODD_LOT_SPREAD_PREMIUM,
    VN_ODD_LOT_MIN_COMMISSION,
    VN_BROKERAGE_FEE,
)

logger = logging.getLogger(__name__)


@dataclass
class OddLotResult:
    """Result of odd-lot calculation"""

    is_odd_lot: bool
    quantity: int
    standard_lots: int  # Full 100-share lots
    odd_shares: int  # Remaining odd shares

    # Cost estimates
    spread_premium_pct: float = 0.0
    estimated_slippage: float = 0.0
    commission: float = 0.0

    # Recommendations
    can_trade: bool = True
    warning: str = ""
    recommendation: str = ""


class OddLotHandler:
    """
    Handle odd-lot trading for Vietnam market

    Vietnam market rules:
    - Standard lot size: 100 shares
    - Odd-lot: 1-99 shares
    - Odd-lots can only be traded through special odd-lot board
    - Wider spreads and less liquidity for odd-lots
    - Minimum commission may apply

    Usage:
        handler = OddLotHandler()

        # Check if order is odd-lot
        result = handler.analyze_order(150)  # 1 lot + 50 odd shares

        # Split order into lots and odd-lot
        lots, odds = handler.split_order(150)  # Returns (100, 50)
    """

    def __init__(
        self,
        lot_size: int = VIETNAM_LOT_SIZE,
        odd_lot_enabled: bool = VN_ODD_LOT_ENABLED,
        spread_premium: float = VN_ODD_LOT_SPREAD_PREMIUM,
        min_commission: float = VN_ODD_LOT_MIN_COMMISSION,
    ):
        self.lot_size = lot_size
        self.odd_lot_enabled = odd_lot_enabled
        self.spread_premium = spread_premium
        self.min_commission = min_commission

    def is_odd_lot(self, quantity: int) -> bool:
        """Check if quantity is odd-lot"""
        return 0 < quantity < self.lot_size

    def has_odd_portion(self, quantity: int) -> bool:
        """Check if quantity has odd portion"""
        return quantity % self.lot_size != 0

    def split_order(self, quantity: int) -> Tuple[int, int]:
        """
        Split order into standard lots and odd-lot portion

        Args:
            quantity: Total shares to trade

        Returns:
            Tuple of (standard_lot_shares, odd_lot_shares)

        Example:
            split_order(150) -> (100, 50)
            split_order(100) -> (100, 0)
            split_order(50) -> (0, 50)
        """
        standard_lots = (quantity // self.lot_size) * self.lot_size
        odd_shares = quantity % self.lot_size
        return standard_lots, odd_shares

    def analyze_order(
        self,
        quantity: int,
        price: float,
        is_sell: bool = False,
    ) -> OddLotResult:
        """
        Analyze order for odd-lot implications

        Args:
            quantity: Number of shares
            price: Share price
            is_sell: True if selling

        Returns:
            OddLotResult with analysis
        """
        standard_lots, odd_shares = self.split_order(quantity)
        is_odd = odd_shares > 0

        result = OddLotResult(
            is_odd_lot=self.is_odd_lot(quantity),
            quantity=quantity,
            standard_lots=standard_lots // self.lot_size,
            odd_shares=odd_shares,
        )

        if not is_odd:
            result.recommendation = "Standard lot order - no special handling needed"
            return result

        # Check if odd-lot trading is enabled
        if not self.odd_lot_enabled:
            result.can_trade = False
            result.warning = "Odd-lot trading is disabled"
            result.recommendation = f"Round to nearest lot: {standard_lots} shares"
            return result

        # Calculate costs for odd-lot portion
        odd_value = odd_shares * price

        # Spread premium
        result.spread_premium_pct = self.spread_premium
        result.estimated_slippage = odd_value * self.spread_premium

        # Commission (minimum applies for small orders)
        standard_commission = odd_value * VN_BROKERAGE_FEE
        result.commission = max(standard_commission, self.min_commission)

        # Generate recommendation
        if result.is_odd_lot:
            # Pure odd-lot order
            total_cost_pct = (result.commission + result.estimated_slippage) / odd_value * 100

            if total_cost_pct > 2.0:
                result.warning = f"High cost for odd-lot: {total_cost_pct:.1f}% of value"
                result.recommendation = (
                    "Consider waiting to accumulate 100+ shares for better execution"
                )
            else:
                result.recommendation = "Odd-lot order acceptable"
        else:
            # Mixed order (lots + odd-lot)
            if odd_shares < 50:
                result.recommendation = (
                    f"Consider selling {odd_shares} odd shares separately or "
                    f"rounding down to {standard_lots} shares"
                )
            else:
                result.recommendation = (
                    f"Order will be split: {standard_lots} in standard board, "
                    f"{odd_shares} in odd-lot board"
                )

        return result

    def calculate_odd_lot_cost(
        self,
        quantity: int,
        price: float,
    ) -> dict:
        """
        Calculate total cost for odd-lot trade

        Returns:
            Dict with cost breakdown
        """
        value = quantity * price

        # Standard fees
        commission = max(value * VN_BROKERAGE_FEE, self.min_commission)
        spread_cost = value * self.spread_premium

        # Estimated total
        total_cost = commission + spread_cost
        total_cost_pct = (total_cost / value * 100) if value > 0 else 0

        return {
            "quantity": quantity,
            "price": price,
            "value": value,
            "commission": commission,
            "spread_cost": spread_cost,
            "total_cost": total_cost,
            "total_cost_pct": total_cost_pct,
            "is_expensive": total_cost_pct > 2.0,
        }

    def reconcile_position(
        self,
        current_qty: int,
        target_qty: int,
    ) -> dict:
        """
        Calculate how to reconcile position with odd-lot handling

        Args:
            current_qty: Current position quantity
            target_qty: Desired position quantity

        Returns:
            Dict with reconciliation plan
        """
        diff = target_qty - current_qty

        if diff == 0:
            return {
                "action": "NONE",
                "standard_lot_qty": 0,
                "odd_lot_qty": 0,
                "total_qty": 0,
            }

        action = "BUY" if diff > 0 else "SELL"
        abs_diff = abs(diff)

        standard_lots, odd_shares = self.split_order(abs_diff)

        return {
            "action": action,
            "standard_lot_qty": standard_lots,
            "odd_lot_qty": odd_shares,
            "total_qty": abs_diff,
            "has_odd_lot": odd_shares > 0,
            "recommendation": (
                f"{action} {standard_lots} via standard board"
                + (f" + {odd_shares} via odd-lot board" if odd_shares > 0 else "")
            ),
        }

    def round_to_lot(self, quantity: int, round_up: bool = False) -> int:
        """
        Round quantity to nearest lot size

        Args:
            quantity: Number of shares
            round_up: If True, round up; otherwise round down

        Returns:
            Rounded quantity
        """
        if round_up:
            return ((quantity + self.lot_size - 1) // self.lot_size) * self.lot_size
        return (quantity // self.lot_size) * self.lot_size


# Singleton instance
_odd_lot_handler: Optional[OddLotHandler] = None


def get_odd_lot_handler() -> OddLotHandler:
    """Get singleton odd-lot handler"""
    global _odd_lot_handler
    if _odd_lot_handler is None:
        _odd_lot_handler = OddLotHandler()
    return _odd_lot_handler
