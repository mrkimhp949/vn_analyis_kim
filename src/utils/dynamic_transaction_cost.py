# -*- coding: utf-8 -*-
"""
Dynamic Transaction Cost Calculator - Priority 1 Improvement

Calculates transaction costs dynamically based on:
- Order size (large orders = higher slippage)
- Session type (ATO/ATC = higher slippage)
- Stock liquidity tier
- Market conditions

Replaces static 1.48% round trip cost with dynamic calculation.
"""

import logging
from typing import Dict, Optional, Tuple

from src.config.constants import (
    VN_BROKERAGE_FEE,
    VN_EXCHANGE_FEE,
    VN_ROUND_TRIP_COST_MARKET,
    VN_SLIPPAGE_LIMIT_ORDER,
    VN_SLIPPAGE_MARKET_ORDER,
    VN_STOCK_TAX,
    VN_TRANSFER_FEE,
    get_dynamic_slippage,
)

logger = logging.getLogger(__name__)


class DynamicTransactionCostCalculator:
    """
    Dynamic Transaction Cost Calculator

    Calculates realistic transaction costs based on order characteristics
    and market conditions.

    Base costs (per side):
    - Brokerage: 0.25%
    - Exchange: 0.03%
    - Transfer: 0.02%
    - Stock tax (sell only): 0.10%
    - Slippage: Dynamic (0.15%-1.0% depending on conditions)

    Adjustments:
    - Large orders (> 5% of daily volume): +0.3% slippage
    - ATO/ATC sessions: +0.2% slippage
    - Illiquid stocks: Higher slippage (already in get_dynamic_slippage)
    """

    # Base costs (per side)
    BASE_BROKERAGE = VN_BROKERAGE_FEE  # 0.25%
    BASE_EXCHANGE = VN_EXCHANGE_FEE  # 0.03%
    BASE_TRANSFER = VN_TRANSFER_FEE  # 0.02%
    BASE_STOCK_TAX = VN_STOCK_TAX  # 0.10% (sell only)

    # Adjustment thresholds
    LARGE_ORDER_THRESHOLD = 0.05  # > 5% of daily volume = large order
    LARGE_ORDER_SLIPPAGE_PREMIUM = 0.003  # +0.3% slippage for large orders
    ATO_ATC_SLIPPAGE_PREMIUM = 0.002  # +0.2% slippage for ATO/ATC

    def calculate_buy_cost(
        self,
        symbol: str,
        order_size: int,
        order_price: float,
        avg_daily_volume: Optional[float] = None,
        liquidity_value: Optional[float] = None,
        session_type: str = "CONTINUOUS",
        order_type: str = "MARKET",  # MARKET or LIMIT
    ) -> Dict:
        """
        Calculate buy-side transaction cost.

        Args:
            symbol: Stock symbol
            order_size: Number of shares
            order_price: Order price per share
            avg_daily_volume: Average daily volume (for size adjustment)
            liquidity_value: Daily liquidity value in VND (for tier detection)
            session_type: Trading session (ATO, ATC, CONTINUOUS)
            order_type: Order type (MARKET or LIMIT)

        Returns:
            Dict with cost breakdown
        """
        order_value = order_size * order_price

        # Base fees
        brokerage = order_value * self.BASE_BROKERAGE
        exchange_fee = order_value * self.BASE_EXCHANGE
        transfer_fee = order_value * self.BASE_TRANSFER

        # Base slippage (by liquidity tier)
        if liquidity_value:
            base_slippage_pct = get_dynamic_slippage(symbol, liquidity_value)
        else:
            base_slippage_pct = VN_SLIPPAGE_MARKET_ORDER  # Default 0.4%

        # Order type adjustment
        if order_type == "LIMIT":
            base_slippage_pct = VN_SLIPPAGE_LIMIT_ORDER  # 0.15% for limit orders

        # Size adjustment (large orders = higher slippage)
        size_adjustment = 0.0
        if avg_daily_volume and avg_daily_volume > 0:
            order_pct_of_volume = order_size / avg_daily_volume
            if order_pct_of_volume > self.LARGE_ORDER_THRESHOLD:
                size_adjustment = self.LARGE_ORDER_SLIPPAGE_PREMIUM
                logger.debug(
                    f"Large order adjustment: {order_pct_of_volume:.1%} of daily volume → "
                    f"+{size_adjustment*100:.2f}% slippage"
                )

        # Session adjustment (ATO/ATC = higher slippage)
        session_adjustment = 0.0
        if session_type in ["ATO", "ATC"]:
            session_adjustment = self.ATO_ATC_SLIPPAGE_PREMIUM
            logger.debug(
                f"Session adjustment: {session_type} → +{session_adjustment*100:.2f}% slippage"
            )

        # Total slippage
        total_slippage_pct = base_slippage_pct + size_adjustment + session_adjustment
        slippage_cost = order_value * total_slippage_pct

        # Total buy cost
        total_cost = brokerage + exchange_fee + transfer_fee + slippage_cost
        total_cost_pct = (total_cost / order_value) * 100 if order_value > 0 else 0

        return {
            "total_cost": total_cost,
            "total_cost_pct": total_cost_pct,
            "breakdown": {
                "brokerage": brokerage,
                "exchange_fee": exchange_fee,
                "transfer_fee": transfer_fee,
                "slippage": slippage_cost,
            },
            "slippage_pct": total_slippage_pct * 100,
            "adjustments": {
                "base_slippage_pct": base_slippage_pct * 100,
                "size_adjustment_pct": size_adjustment * 100,
                "session_adjustment_pct": session_adjustment * 100,
            },
        }

    def calculate_sell_cost(
        self,
        symbol: str,
        order_size: int,
        order_price: float,
        avg_daily_volume: Optional[float] = None,
        liquidity_value: Optional[float] = None,
        session_type: str = "CONTINUOUS",
        order_type: str = "MARKET",
    ) -> Dict:
        """
        Calculate sell-side transaction cost (includes stock tax).

        Args:
            symbol: Stock symbol
            order_size: Number of shares
            order_price: Order price per share
            avg_daily_volume: Average daily volume
            liquidity_value: Daily liquidity value in VND
            session_type: Trading session
            order_type: Order type

        Returns:
            Dict with cost breakdown
        """
        order_value = order_size * order_price

        # Base fees (same as buy)
        brokerage = order_value * self.BASE_BROKERAGE
        exchange_fee = order_value * self.BASE_EXCHANGE
        transfer_fee = order_value * self.BASE_TRANSFER

        # Stock tax (sell only)
        stock_tax = order_value * self.BASE_STOCK_TAX

        # Slippage (same calculation as buy)
        if liquidity_value:
            base_slippage_pct = get_dynamic_slippage(symbol, liquidity_value)
        else:
            base_slippage_pct = VN_SLIPPAGE_MARKET_ORDER

        if order_type == "LIMIT":
            base_slippage_pct = VN_SLIPPAGE_LIMIT_ORDER

        # Size adjustment
        size_adjustment = 0.0
        if avg_daily_volume and avg_daily_volume > 0:
            order_pct_of_volume = order_size / avg_daily_volume
            if order_pct_of_volume > self.LARGE_ORDER_THRESHOLD:
                size_adjustment = self.LARGE_ORDER_SLIPPAGE_PREMIUM

        # Session adjustment
        session_adjustment = 0.0
        if session_type in ["ATO", "ATC"]:
            session_adjustment = self.ATO_ATC_SLIPPAGE_PREMIUM

        total_slippage_pct = base_slippage_pct + size_adjustment + session_adjustment
        slippage_cost = order_value * total_slippage_pct

        # Total sell cost
        total_cost = brokerage + exchange_fee + transfer_fee + stock_tax + slippage_cost
        total_cost_pct = (total_cost / order_value) * 100 if order_value > 0 else 0

        return {
            "total_cost": total_cost,
            "total_cost_pct": total_cost_pct,
            "breakdown": {
                "brokerage": brokerage,
                "exchange_fee": exchange_fee,
                "transfer_fee": transfer_fee,
                "stock_tax": stock_tax,
                "slippage": slippage_cost,
            },
            "slippage_pct": total_slippage_pct * 100,
            "adjustments": {
                "base_slippage_pct": base_slippage_pct * 100,
                "size_adjustment_pct": size_adjustment * 100,
                "session_adjustment_pct": session_adjustment * 100,
            },
        }

    def calculate_round_trip_cost(
        self,
        symbol: str,
        order_size: int,
        order_price: float,
        avg_daily_volume: Optional[float] = None,
        liquidity_value: Optional[float] = None,
        session_type: str = "CONTINUOUS",
        order_type: str = "MARKET",
    ) -> Dict:
        """
        Calculate round trip transaction cost (buy + sell).

        Args:
            symbol: Stock symbol
            order_size: Number of shares
            order_price: Order price per share
            avg_daily_volume: Average daily volume
            liquidity_value: Daily liquidity value in VND
            session_type: Trading session
            order_type: Order type

        Returns:
            Dict with round trip cost breakdown
        """
        buy_cost = self.calculate_buy_cost(
            symbol=symbol,
            order_size=order_size,
            order_price=order_price,
            avg_daily_volume=avg_daily_volume,
            liquidity_value=liquidity_value,
            session_type=session_type,
            order_type=order_type,
        )

        sell_cost = self.calculate_sell_cost(
            symbol=symbol,
            order_size=order_size,
            order_price=order_price,
            avg_daily_volume=avg_daily_volume,
            liquidity_value=liquidity_value,
            session_type=session_type,
            order_type=order_type,
        )

        order_value = order_size * order_price
        total_round_trip_cost = buy_cost["total_cost"] + sell_cost["total_cost"]
        total_round_trip_cost_pct = (
            (total_round_trip_cost / order_value) * 100 if order_value > 0 else 0
        )

        return {
            "total_cost": total_round_trip_cost,
            "total_cost_pct": total_round_trip_cost_pct,
            "buy_cost": buy_cost,
            "sell_cost": sell_cost,
            "breakdown": {
                "buy_cost_pct": buy_cost["total_cost_pct"],
                "sell_cost_pct": sell_cost["total_cost_pct"],
                "round_trip_pct": total_round_trip_cost_pct,
            },
        }

    def get_cost_for_rr_calculation(
        self,
        symbol: str,
        order_size: int,
        entry_price: float,
        take_profit_price: float,
        avg_daily_volume: Optional[float] = None,
        liquidity_value: Optional[float] = None,
        session_type: str = "CONTINUOUS",
        order_type: str = "MARKET",
    ) -> Tuple[float, float]:
        """
        Get transaction costs for risk/reward calculation.

        Returns:
            Tuple of (entry_cost_pct, exit_cost_pct) for R:R calculation
        """
        buy_cost = self.calculate_buy_cost(
            symbol=symbol,
            order_size=order_size,
            order_price=entry_price,
            avg_daily_volume=avg_daily_volume,
            liquidity_value=liquidity_value,
            session_type=session_type,
            order_type=order_type,
        )

        sell_cost = self.calculate_sell_cost(
            symbol=symbol,
            order_size=order_size,
            order_price=take_profit_price,
            avg_daily_volume=avg_daily_volume,
            liquidity_value=liquidity_value,
            session_type=session_type,
            order_type=order_type,
        )

        return (buy_cost["total_cost_pct"], sell_cost["total_cost_pct"])


# Singleton instance
_cost_calculator: Optional[DynamicTransactionCostCalculator] = None


def get_dynamic_cost_calculator() -> DynamicTransactionCostCalculator:
    """Get singleton instance of DynamicTransactionCostCalculator."""
    global _cost_calculator
    if _cost_calculator is None:
        _cost_calculator = DynamicTransactionCostCalculator()
    return _cost_calculator


# Convenience functions
def calculate_round_trip_cost_dynamic(
    symbol: str,
    order_size: int,
    order_price: float,
    avg_daily_volume: Optional[float] = None,
    liquidity_value: Optional[float] = None,
    session_type: str = "CONTINUOUS",
    order_type: str = "MARKET",
) -> float:
    """
    Quick function to get round trip cost percentage.

    Returns:
        Round trip cost as percentage (e.g., 1.48 for 1.48%)
    """
    calculator = get_dynamic_cost_calculator()
    result = calculator.calculate_round_trip_cost(
        symbol=symbol,
        order_size=order_size,
        order_price=order_price,
        avg_daily_volume=avg_daily_volume,
        liquidity_value=liquidity_value,
        session_type=session_type,
        order_type=order_type,
    )
    return result["total_cost_pct"]


# Test
if __name__ == "__main__":
    print("Testing Dynamic Transaction Cost Calculator...")

    calculator = DynamicTransactionCostCalculator()

    # Test case 1: Normal order, VN30 stock
    print("\n1️⃣ Test: Normal order, VN30 stock (VNM)")
    cost = calculator.calculate_round_trip_cost(
        symbol="VNM",
        order_size=1000,
        order_price=80_000,
        avg_daily_volume=500_000,
        liquidity_value=10_000_000_000,  # 10B VND
        session_type="CONTINUOUS",
        order_type="MARKET",
    )
    print(f"   Round trip cost: {cost['total_cost_pct']:.2f}%")
    print(f"   Buy cost: {cost['buy_cost']['total_cost_pct']:.2f}%")
    print(f"   Sell cost: {cost['sell_cost']['total_cost_pct']:.2f}%")

    # Test case 2: Large order (> 5% of volume)
    print("\n2️⃣ Test: Large order (> 5% of volume)")
    cost = calculator.calculate_round_trip_cost(
        symbol="VNM",
        order_size=50_000,  # 10% of daily volume
        order_price=80_000,
        avg_daily_volume=500_000,
        liquidity_value=10_000_000_000,
        session_type="CONTINUOUS",
        order_type="MARKET",
    )
    print(f"   Round trip cost: {cost['total_cost_pct']:.2f}%")
    print(f"   Size adjustment: +{cost['buy_cost']['adjustments']['size_adjustment_pct']:.2f}%")

    # Test case 3: ATO session
    print("\n3️⃣ Test: ATO session")
    cost = calculator.calculate_round_trip_cost(
        symbol="VNM",
        order_size=1000,
        order_price=80_000,
        avg_daily_volume=500_000,
        liquidity_value=10_000_000_000,
        session_type="ATO",
        order_type="MARKET",
    )
    print(f"   Round trip cost: {cost['total_cost_pct']:.2f}%")
    print(
        f"   Session adjustment: +{cost['buy_cost']['adjustments']['session_adjustment_pct']:.2f}%"
    )

    # Test case 4: Limit order (lower slippage)
    print("\n4️⃣ Test: Limit order")
    cost = calculator.calculate_round_trip_cost(
        symbol="VNM",
        order_size=1000,
        order_price=80_000,
        avg_daily_volume=500_000,
        liquidity_value=10_000_000_000,
        session_type="CONTINUOUS",
        order_type="LIMIT",
    )
    print(f"   Round trip cost: {cost['total_cost_pct']:.2f}%")
    print(f"   (Lower than market order due to limit order slippage)")

    print("\n✅ Dynamic Transaction Cost Calculator test completed!")
