# -*- coding: utf-8 -*-
"""
Odd-Lot Trading Support for Vietnam Stock Market

Vietnam Odd-Lot Rules (since 2021):
- Regular lot size: 100 shares
- Odd-lot: 1-99 shares (lô lẻ)
- Odd-lot trading allowed since Jan 4, 2021

Key differences for odd-lot trading:
1. Trading sessions: Only during continuous trading (not ATO/ATC)
2. Order types: Only limit orders (LO) allowed
3. Price: Can only sell at bid price or below (less favorable)
4. Priority: Lower priority than regular lot orders
5. Matching: Odd-lots are matched separately from regular lots

Odd-lot order rules:
- HOSE: Odd-lot orders matched in separate book
- HNX: Similar odd-lot handling
- UPCOM: Follows HOSE rules

Use cases:
1. Retail investors with small capital
2. DCA (Dollar Cost Averaging) strategies
3. Selling fractional positions from stock dividends
4. Portfolio rebalancing with precise allocations
"""

import logging
from dataclasses import dataclass
from datetime import datetime, time
from enum import Enum
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class OrderLotType(Enum):
    """Order lot type classification"""

    REGULAR_LOT = "REGULAR"  # >= 100 shares, multiple of 100
    ODD_LOT = "ODD_LOT"  # 1-99 shares
    MIXED_LOT = "MIXED"  # Has both regular and odd-lot components


class OddLotOrderStatus(Enum):
    """Odd-lot order execution status"""

    PENDING = "PENDING"
    PARTIALLY_FILLED = "PARTIAL"
    FILLED = "FILLED"
    CANCELLED = "CANCELLED"
    REJECTED = "REJECTED"


@dataclass
class OddLotOrder:
    """Odd-lot order record"""

    order_id: str
    symbol: str
    side: str  # "BUY" or "SELL"
    quantity: int  # 1-99 shares
    price: float
    order_time: datetime
    status: OddLotOrderStatus = OddLotOrderStatus.PENDING
    filled_quantity: int = 0
    filled_price: float = 0.0
    fill_time: Optional[datetime] = None
    reject_reason: Optional[str] = None

    @property
    def remaining_quantity(self) -> int:
        return self.quantity - self.filled_quantity


@dataclass
class LotBreakdown:
    """Breakdown of order into regular and odd-lot components"""

    total_quantity: int
    regular_lots: int  # Number of regular lots (100-share units)
    regular_quantity: int  # Total shares in regular lots
    odd_lot_quantity: int  # Remaining odd-lot shares (0-99)
    lot_type: OrderLotType
    recommendations: List[str]


class OddLotTrader:
    """
    Odd-lot trading handler for Vietnam stock market.

    Handles the complexities of odd-lot trading including:
    - Order splitting (regular + odd-lot)
    - Session validation (no ATO/ATC for odd-lots)
    - Price constraints (bid price or lower for sells)
    - Order priority considerations

    Usage:
        trader = OddLotTrader()
        breakdown = trader.analyze_order(250)  # 2 regular lots + 50 odd-lot
        if trader.is_odd_lot_session():
            trader.place_odd_lot_order("VNM", "SELL", 50, 79500)
    """

    REGULAR_LOT_SIZE = 100

    # Odd-lot trading hours (continuous session only)
    ODD_LOT_START_AM = time(9, 15)  # After ATO
    ODD_LOT_END_AM = time(11, 30)  # Before lunch
    ODD_LOT_START_PM = time(13, 0)  # After lunch
    ODD_LOT_END_PM = time(14, 30)  # Before ATC

    # Odd-lot specific costs (may vary by broker)
    ODD_LOT_SPREAD_PREMIUM = 0.005  # 0.5% wider spread for odd-lots
    ODD_LOT_MIN_COMMISSION = 11_000  # 11,000 VND minimum commission

    def __init__(
        self,
        enable_odd_lot: bool = True,
        prefer_regular_lots: bool = True,
        max_odd_lot_pct: float = 0.10,  # Max 10% of position as odd-lot
    ):
        """
        Initialize odd-lot trader.

        Args:
            enable_odd_lot: Whether to enable odd-lot trading
            prefer_regular_lots: Prefer regular lots when possible
            max_odd_lot_pct: Maximum portfolio percentage for odd-lots
        """
        self.enable_odd_lot = enable_odd_lot
        self.prefer_regular_lots = prefer_regular_lots
        self.max_odd_lot_pct = max_odd_lot_pct

        # Order tracking
        self._pending_orders: Dict[str, OddLotOrder] = {}
        self._filled_orders: List[OddLotOrder] = []

        logger.info(
            f"✅ OddLotTrader initialized: "
            f"enabled={enable_odd_lot}, prefer_regular={prefer_regular_lots}"
        )

    def analyze_order(self, quantity: int) -> LotBreakdown:
        """
        Analyze an order quantity and break down into lots.

        Args:
            quantity: Total number of shares

        Returns:
            LotBreakdown with regular and odd-lot components
        """
        if quantity <= 0:
            return LotBreakdown(
                total_quantity=0,
                regular_lots=0,
                regular_quantity=0,
                odd_lot_quantity=0,
                lot_type=OrderLotType.REGULAR_LOT,
                recommendations=["Invalid quantity"],
            )

        regular_lots = quantity // self.REGULAR_LOT_SIZE
        regular_quantity = regular_lots * self.REGULAR_LOT_SIZE
        odd_lot_quantity = quantity % self.REGULAR_LOT_SIZE

        recommendations = []

        if odd_lot_quantity == 0:
            lot_type = OrderLotType.REGULAR_LOT
            recommendations.append("✅ Order is in regular lots - standard execution")
        elif regular_lots == 0:
            lot_type = OrderLotType.ODD_LOT
            recommendations.append("⚠️ Pure odd-lot order - may have wider spread")
            recommendations.append("📋 Only limit orders allowed during continuous session")
            recommendations.append("💡 Consider rounding up to 100 shares if possible")
        else:
            lot_type = OrderLotType.MIXED_LOT
            recommendations.append(
                f"📊 Mixed order: {regular_lots} regular lots ({regular_quantity} shares) + "
                f"{odd_lot_quantity} odd-lot shares"
            )
            recommendations.append("💡 Will be split into 2 separate orders")
            recommendations.append("⚠️ Odd-lot portion may fill at less favorable price")

        return LotBreakdown(
            total_quantity=quantity,
            regular_lots=regular_lots,
            regular_quantity=regular_quantity,
            odd_lot_quantity=odd_lot_quantity,
            lot_type=lot_type,
            recommendations=recommendations,
        )

    def is_odd_lot_session(self, check_time: Optional[datetime] = None) -> bool:
        """
        Check if current time is valid for odd-lot trading.

        Odd-lot trading is only allowed during continuous sessions:
        - Morning: 9:15 - 11:30
        - Afternoon: 13:00 - 14:30

        NOT allowed during:
        - ATO: 9:00 - 9:15
        - ATC: 14:30 - 14:45

        Args:
            check_time: Time to check (default: now)

        Returns:
            True if odd-lot trading is allowed
        """
        if check_time is None:
            check_time = datetime.now()

        current_time = check_time.time()

        # Check morning session
        if self.ODD_LOT_START_AM <= current_time <= self.ODD_LOT_END_AM:
            return True

        # Check afternoon session
        if self.ODD_LOT_START_PM <= current_time <= self.ODD_LOT_END_PM:
            return True

        return False

    def get_session_status(self) -> Dict:
        """
        Get current odd-lot session status.

        Returns:
            Dict with session information
        """
        now = datetime.now()
        current_time = now.time()
        is_valid = self.is_odd_lot_session(now)

        session = "CLOSED"
        next_open = None

        if current_time < self.ODD_LOT_START_AM:
            session = "PRE_MARKET"
            next_open = datetime.combine(now.date(), self.ODD_LOT_START_AM)
        elif self.ODD_LOT_START_AM <= current_time <= self.ODD_LOT_END_AM:
            session = "MORNING_CONTINUOUS"
        elif self.ODD_LOT_END_AM < current_time < self.ODD_LOT_START_PM:
            session = "LUNCH_BREAK"
            next_open = datetime.combine(now.date(), self.ODD_LOT_START_PM)
        elif self.ODD_LOT_START_PM <= current_time <= self.ODD_LOT_END_PM:
            session = "AFTERNOON_CONTINUOUS"
        elif current_time > self.ODD_LOT_END_PM:
            session = "AFTER_HOURS"

        return {
            "current_time": now.strftime("%H:%M:%S"),
            "session": session,
            "odd_lot_allowed": is_valid,
            "next_open": next_open.strftime("%H:%M") if next_open else None,
            "message": (
                "✅ Odd-lot trading available"
                if is_valid
                else f"❌ Odd-lot not available during {session}"
            ),
        }

    def calculate_odd_lot_price(
        self, side: str, current_price: float, bid_price: float, ask_price: float
    ) -> float:
        """
        Calculate appropriate price for odd-lot order.

        Rules:
        - SELL: Must be at bid price or lower (less favorable)
        - BUY: Must be at ask price or higher (less favorable)

        Args:
            side: "BUY" or "SELL"
            current_price: Current market price
            bid_price: Current bid price
            ask_price: Current ask price

        Returns:
            Recommended price for odd-lot order
        """
        if side.upper() == "SELL":
            # For sells, use bid price (lower)
            return bid_price
        else:
            # For buys, use ask price (higher)
            return ask_price

    def estimate_execution_cost(self, quantity: int, price: float, side: str) -> Dict:
        """
        Estimate execution cost for odd-lot order.

        Odd-lot orders typically have:
        - Wider spread (0.5% premium)
        - Minimum commission
        - Lower fill priority

        Args:
            quantity: Number of shares (1-99)
            price: Order price
            side: "BUY" or "SELL"

        Returns:
            Dict with cost estimates
        """
        if quantity >= self.REGULAR_LOT_SIZE:
            logger.warning(f"Quantity {quantity} is not an odd-lot")

        value = quantity * price

        # Standard commission (0.15%)
        commission = max(value * 0.0015, self.ODD_LOT_MIN_COMMISSION)

        # Spread cost (wider for odd-lots)
        spread_cost = value * self.ODD_LOT_SPREAD_PREMIUM

        # Tax (sell only)
        tax = value * 0.001 if side.upper() == "SELL" else 0

        total_cost = commission + spread_cost + tax

        return {
            "order_value": value,
            "commission": commission,
            "spread_cost": spread_cost,
            "tax": tax,
            "total_cost": total_cost,
            "cost_pct": total_cost / value * 100 if value > 0 else 0,
            "note": "⚠️ Odd-lot costs are higher than regular lots",
        }

    def can_place_odd_lot(self, symbol: str, side: str, quantity: int) -> Tuple[bool, str]:
        """
        Validate if odd-lot order can be placed.

        Args:
            symbol: Stock symbol
            side: "BUY" or "SELL"
            quantity: Number of shares

        Returns:
            Tuple of (can_place, reason)
        """
        # Check if odd-lot trading is enabled
        if not self.enable_odd_lot:
            return False, "Odd-lot trading is disabled"

        # Check quantity is actually an odd-lot
        if quantity <= 0:
            return False, "Invalid quantity"

        if quantity >= self.REGULAR_LOT_SIZE:
            return False, f"Quantity {quantity} is not an odd-lot (must be 1-99)"

        # Check if odd-lot session
        if not self.is_odd_lot_session():
            session_status = self.get_session_status()
            return False, session_status["message"]

        # Check for warrants/ETFs (some may have different odd-lot rules)
        if any(suffix in symbol.upper() for suffix in ["_WFT", "-WR", "ETF"]):
            return False, f"Odd-lot trading not available for {symbol} (warrant/ETF)"

        return True, "✅ Odd-lot order can be placed"

    def place_odd_lot_order(
        self, symbol: str, side: str, quantity: int, price: float
    ) -> Optional[OddLotOrder]:
        """
        Place an odd-lot order.

        Args:
            symbol: Stock symbol
            side: "BUY" or "SELL"
            quantity: Number of shares (1-99)
            price: Limit price

        Returns:
            OddLotOrder if successful, None if failed
        """
        can_place, reason = self.can_place_odd_lot(symbol, side, quantity)
        if not can_place:
            logger.warning(f"❌ Cannot place odd-lot order: {reason}")
            return None

        order_id = f"OL-{datetime.now().strftime('%H%M%S')}-{symbol}"

        order = OddLotOrder(
            order_id=order_id,
            symbol=symbol,
            side=side.upper(),
            quantity=quantity,
            price=price,
            order_time=datetime.now(),
        )

        self._pending_orders[order_id] = order

        logger.info(
            f"📝 Odd-lot order placed: {side} {quantity} {symbol} @ {price:,.0f} "
            f"(ID: {order_id})"
        )

        return order

    def split_mixed_order(
        self,
        symbol: str,
        side: str,
        total_quantity: int,
        price: float,
        bid_price: float,
        ask_price: float,
    ) -> Dict:
        """
        Split a mixed order into regular and odd-lot components.

        Args:
            symbol: Stock symbol
            side: "BUY" or "SELL"
            total_quantity: Total shares to trade
            price: Regular lot price
            bid_price: Current bid price (for odd-lot sell)
            ask_price: Current ask price (for odd-lot buy)

        Returns:
            Dict with regular and odd-lot order details
        """
        breakdown = self.analyze_order(total_quantity)

        result = {
            "breakdown": breakdown,
            "regular_order": None,
            "odd_lot_order": None,
            "total_cost_estimate": 0.0,
        }

        # Regular lot component
        if breakdown.regular_quantity > 0:
            result["regular_order"] = {
                "type": "REGULAR_LOT",
                "symbol": symbol,
                "side": side,
                "quantity": breakdown.regular_quantity,
                "price": price,
                "order_type": "LO",  # Or "MP" for market
            }

        # Odd-lot component
        if breakdown.odd_lot_quantity > 0:
            odd_lot_price = self.calculate_odd_lot_price(side, price, bid_price, ask_price)
            cost_estimate = self.estimate_execution_cost(
                breakdown.odd_lot_quantity, odd_lot_price, side
            )

            result["odd_lot_order"] = {
                "type": "ODD_LOT",
                "symbol": symbol,
                "side": side,
                "quantity": breakdown.odd_lot_quantity,
                "price": odd_lot_price,
                "order_type": "LO",  # Only limit orders for odd-lots
                "cost_estimate": cost_estimate,
            }
            result["total_cost_estimate"] += cost_estimate["total_cost"]

        return result

    def round_to_lot(self, quantity: int, direction: str = "down") -> int:
        """
        Round quantity to nearest lot size.

        Args:
            quantity: Original quantity
            direction: "down", "up", or "nearest"

        Returns:
            Rounded quantity (multiple of 100)
        """
        if direction == "down":
            return (quantity // self.REGULAR_LOT_SIZE) * self.REGULAR_LOT_SIZE
        elif direction == "up":
            return (
                (quantity + self.REGULAR_LOT_SIZE - 1) // self.REGULAR_LOT_SIZE
            ) * self.REGULAR_LOT_SIZE
        else:  # nearest
            return round(quantity / self.REGULAR_LOT_SIZE) * self.REGULAR_LOT_SIZE

    def get_status_message(self) -> str:
        """Get formatted status message."""
        session = self.get_session_status()

        lines = [
            "=" * 50,
            "📊 ODD-LOT TRADING STATUS",
            "=" * 50,
            f"Time: {session['current_time']}",
            f"Session: {session['session']}",
            f"Odd-Lot Allowed: {session['message']}",
        ]

        if session["next_open"]:
            lines.append(f"Next Opening: {session['next_open']}")

        pending_count = len(self._pending_orders)
        filled_count = len(self._filled_orders)

        lines.extend(
            [
                "-" * 50,
                f"Pending Orders: {pending_count}",
                f"Filled Orders:  {filled_count}",
            ]
        )

        if self._pending_orders:
            lines.append("-" * 50)
            lines.append("📋 PENDING ORDERS:")
            for order_id, order in self._pending_orders.items():
                lines.append(
                    f"   {order.side} {order.quantity} {order.symbol} @ "
                    f"{order.price:,.0f} ({order.status.value})"
                )

        lines.append("=" * 50)
        return "\n".join(lines)


# Singleton instance
_odd_lot_trader: Optional[OddLotTrader] = None


def get_odd_lot_trader(enable: bool = True) -> OddLotTrader:
    """Get singleton odd-lot trader instance."""
    global _odd_lot_trader
    if _odd_lot_trader is None:
        _odd_lot_trader = OddLotTrader(enable_odd_lot=enable)
    return _odd_lot_trader


def validate_and_split_order(
    quantity: int, symbol: str, side: str, price: float, enable_odd_lot: bool = True
) -> Dict:
    """
    Convenience function to validate and split an order.

    Args:
        quantity: Total shares
        symbol: Stock symbol
        side: "BUY" or "SELL"
        price: Target price
        enable_odd_lot: Whether to handle odd-lots

    Returns:
        Dict with order splitting information
    """
    trader = get_odd_lot_trader(enable_odd_lot)
    breakdown = trader.analyze_order(quantity)

    result = {"breakdown": breakdown, "can_execute": True, "orders": []}

    if breakdown.lot_type == OrderLotType.REGULAR_LOT:
        result["orders"].append({"type": "REGULAR", "quantity": quantity, "price": price})
    elif breakdown.lot_type == OrderLotType.ODD_LOT:
        if enable_odd_lot and trader.is_odd_lot_session():
            result["orders"].append(
                {
                    "type": "ODD_LOT",
                    "quantity": quantity,
                    "price": price,
                    "note": "Odd-lot order - limit order only",
                }
            )
        else:
            result["can_execute"] = False
            result["reason"] = "Odd-lot trading not available"
    else:  # MIXED_LOT
        result["orders"].append(
            {"type": "REGULAR", "quantity": breakdown.regular_quantity, "price": price}
        )
        if enable_odd_lot and trader.is_odd_lot_session():
            result["orders"].append(
                {
                    "type": "ODD_LOT",
                    "quantity": breakdown.odd_lot_quantity,
                    "price": price,
                    "note": "Odd-lot portion",
                }
            )
        else:
            result["orders"].append(
                {
                    "type": "SKIPPED",
                    "quantity": breakdown.odd_lot_quantity,
                    "reason": "Odd-lot skipped (not available)",
                }
            )

    return result


# Test
if __name__ == "__main__":
    print("Testing Odd-Lot Trader...")

    trader = OddLotTrader(enable_odd_lot=True)

    print("\n1️⃣ Session Status:")
    print(trader.get_status_message())

    print("\n2️⃣ Analyzing Orders:")
    test_quantities = [100, 150, 50, 99, 250, 1]
    for qty in test_quantities:
        breakdown = trader.analyze_order(qty)
        print(f"\n   Quantity: {qty}")
        print(f"   Type: {breakdown.lot_type.value}")
        print(f"   Regular: {breakdown.regular_quantity} ({breakdown.regular_lots} lots)")
        print(f"   Odd-lot: {breakdown.odd_lot_quantity}")
        for rec in breakdown.recommendations:
            print(f"   → {rec}")

    print("\n3️⃣ Cost Estimate for Odd-Lot:")
    cost = trader.estimate_execution_cost(50, 80_000, "SELL")
    print(f"   Order value: {cost['order_value']:,.0f} VND")
    print(f"   Commission: {cost['commission']:,.0f} VND")
    print(f"   Spread cost: {cost['spread_cost']:,.0f} VND")
    print(f"   Tax: {cost['tax']:,.0f} VND")
    print(f"   Total cost: {cost['total_cost']:,.0f} VND ({cost['cost_pct']:.2f}%)")

    print("\n4️⃣ Order Splitting:")
    split = validate_and_split_order(250, "VNM", "SELL", 80_000)
    print(f"   Can execute: {split['can_execute']}")
    for order in split["orders"]:
        print(f"   → {order['type']}: {order['quantity']} shares")

    print("\n✅ Test completed!")
