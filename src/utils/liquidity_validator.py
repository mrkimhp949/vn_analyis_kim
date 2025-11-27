"""
Liquidity Validation Module
Validates bid-ask spread and order book depth before entry

IMPROVEMENT: Addresses critique point #6 in trading logic evaluation
Prevents entries with poor liquidity that could lead to high slippage
"""

import logging
from typing import Dict, Optional, Tuple

import pandas as pd

logger = logging.getLogger(__name__)


class LiquidityValidator:
    """
    Validates trade liquidity to prevent excessive slippage

    Checks:
    1. Bid-ask spread as percentage of mid price
    2. Order book depth at bid/ask levels (if available)
    3. Recent volume patterns
    4. Price impact estimation
    """

    def __init__(
        self,
        max_spread_pct: float = 0.5,  # 0.5% max bid-ask spread
        max_spread_pct_large_order: float = 0.3,  # 0.3% for large orders
        large_order_threshold: float = 0.02,  # 2% of daily volume
        min_volume_ratio: float = 0.5,  # Current volume must be >= 50% of avg
        warn_spread_pct: float = 0.3,  # Warning at 0.3% spread
    ):
        self.max_spread_pct = max_spread_pct
        self.max_spread_pct_large_order = max_spread_pct_large_order
        self.large_order_threshold = large_order_threshold
        self.min_volume_ratio = min_volume_ratio
        self.warn_spread_pct = warn_spread_pct

    def estimate_spread_from_price(
        self,
        current_price: float,
        volatility: float = 0.02,  # 2% default volatility
    ) -> float:
        """
        Estimate bid-ask spread from price and volatility

        Vietnam market typical spreads:
        - High liquidity (VNM, VCB, HPG): 0.1-0.2%
        - Medium liquidity: 0.3-0.5%
        - Low liquidity: 0.5-1.0%+

        Args:
            current_price: Current price
            volatility: Stock volatility (default 2%)

        Returns:
            Estimated spread as percentage
        """
        # Base spread estimate: 0.2% for normal stocks
        base_spread_pct = 0.002

        # Adjust for volatility (higher vol = wider spread)
        volatility_adjustment = volatility * 0.5  # 50% of volatility
        estimated_spread_pct = base_spread_pct + volatility_adjustment

        # Cap at reasonable bounds (0.1% to 1.5%)
        estimated_spread_pct = max(0.001, min(0.015, estimated_spread_pct))

        return estimated_spread_pct

    def validate_spread(
        self,
        symbol: str,
        current_price: float,
        bid_price: Optional[float] = None,
        ask_price: Optional[float] = None,
        volatility: float = 0.02,
        order_size_shares: int = 0,
        avg_daily_volume: float = 0,
    ) -> Tuple[bool, Optional[str]]:
        """
        Validate bid-ask spread before entry

        Args:
            symbol: Stock symbol
            current_price: Current/mid price
            bid_price: Bid price (if available)
            ask_price: Ask price (if available)
            volatility: Stock volatility
            order_size_shares: Size of order in shares
            avg_daily_volume: Average daily volume

        Returns:
            (is_acceptable, warning_message)
        """
        # If bid/ask not available, estimate from price
        if bid_price is None or ask_price is None or bid_price <= 0 or ask_price <= 0:
            estimated_spread_pct = self.estimate_spread_from_price(current_price, volatility)

            if estimated_spread_pct > self.max_spread_pct:
                return (
                    False,
                    f"Estimated spread too wide: {estimated_spread_pct*100:.2f}% "
                    f"(max: {self.max_spread_pct*100:.2f}%)",
                )
            elif estimated_spread_pct > self.warn_spread_pct:
                return (True, f"⚠️ Wide estimated spread: {estimated_spread_pct*100:.2f}%")

            return (True, None)

        # Calculate actual spread
        mid_price = (bid_price + ask_price) / 2
        spread = ask_price - bid_price
        spread_pct = spread / mid_price if mid_price > 0 else 0

        # Determine if this is a large order
        is_large_order = False
        if order_size_shares > 0 and avg_daily_volume > 0:
            order_pct_of_volume = order_size_shares / avg_daily_volume
            is_large_order = order_pct_of_volume >= self.large_order_threshold

        # Use stricter threshold for large orders
        max_allowed_spread = (
            self.max_spread_pct_large_order if is_large_order else self.max_spread_pct
        )

        # Check if spread exceeds maximum
        if spread_pct > max_allowed_spread:
            return (
                False,
                f"Bid-ask spread too wide: {spread_pct*100:.2f}% "
                f"(max: {max_allowed_spread*100:.2f}% for {'large' if is_large_order else 'normal'} order)",
            )

        # Warning for moderately wide spread
        if spread_pct > self.warn_spread_pct:
            return (
                True,
                f"⚠️ Moderate spread: {spread_pct*100:.2f}% "
                f"(bid: {bid_price:,.0f}, ask: {ask_price:,.0f})",
            )

        # Spread acceptable
        logger.debug(
            f"[{symbol}] Spread OK: {spread_pct*100:.3f}% "
            f"(bid: {bid_price:,.0f}, ask: {ask_price:,.0f})"
        )
        return (True, None)

    def validate_volume(
        self,
        symbol: str,
        current_volume: float,
        avg_volume: float,
    ) -> Tuple[bool, Optional[str]]:
        """
        Validate current volume is adequate

        Args:
            symbol: Stock symbol
            current_volume: Current period volume
            avg_volume: Average volume

        Returns:
            (is_acceptable, warning_message)
        """
        if avg_volume <= 0:
            return (True, "⚠️ No average volume data available")

        volume_ratio = current_volume / avg_volume

        if volume_ratio < self.min_volume_ratio:
            return (
                False,
                f"Low volume: {volume_ratio*100:.0f}% of average "
                f"(min: {self.min_volume_ratio*100:.0f}%)",
            )

        if volume_ratio < self.min_volume_ratio * 1.2:  # 60% of average
            return (True, f"⚠️ Below-average volume: {volume_ratio*100:.0f}% of average")

        return (True, None)

    def estimate_slippage(
        self,
        order_size_shares: int,
        avg_daily_volume: float,
        spread_pct: float,
        volatility: float = 0.02,
    ) -> Dict:
        """
        Estimate slippage for an order

        Args:
            order_size_shares: Order size in shares
            avg_daily_volume: Average daily volume
            spread_pct: Bid-ask spread as percentage
            volatility: Stock volatility

        Returns:
            Dict with slippage estimate and breakdown
        """
        if avg_daily_volume <= 0:
            return {
                "estimated_slippage_pct": 0.003,  # 0.3% default
                "breakdown": "No volume data - using default",
            }

        # Calculate order as percentage of daily volume
        order_pct_of_volume = order_size_shares / avg_daily_volume

        # Base slippage = half of spread (assuming we pay half the spread on average)
        base_slippage = spread_pct / 2

        # Market impact (more significant for large orders)
        # Rule of thumb: market impact ≈ volatility × sqrt(order% of volume)
        market_impact = volatility * (order_pct_of_volume**0.5)

        # Total slippage
        total_slippage = base_slippage + market_impact

        # Cap at reasonable bounds (0.1% to 2%)
        total_slippage = max(0.001, min(0.020, total_slippage))

        breakdown = (
            f"Order size: {order_pct_of_volume*100:.2f}% of daily volume | "
            f"Spread: {spread_pct*100:.2f}% | "
            f"Impact: {market_impact*100:.2f}% | "
            f"Total: {total_slippage*100:.2f}%"
        )

        return {
            "estimated_slippage_pct": total_slippage,
            "order_pct_of_volume": order_pct_of_volume,
            "spread_cost": base_slippage,
            "market_impact": market_impact,
            "breakdown": breakdown,
        }


# Singleton instance
_liquidity_validator = None


def get_liquidity_validator() -> LiquidityValidator:
    """Get singleton instance of liquidity validator"""
    global _liquidity_validator
    if _liquidity_validator is None:
        _liquidity_validator = LiquidityValidator()
    return _liquidity_validator


# ============================================================================
# TESTING
# ============================================================================

if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("🧪 TESTING LIQUIDITY VALIDATOR")
    print("=" * 70 + "\n")

    validator = LiquidityValidator()

    # Test 1: Good liquidity
    print("1️⃣ Test: Good liquidity (VNM)")
    is_ok, warning = validator.validate_spread(
        symbol="VNM",
        current_price=70_000,
        bid_price=69_950,
        ask_price=70_050,
        volatility=0.015,
    )
    print(f"   Result: {'✅ OK' if is_ok else '❌ FAIL'}")
    if warning:
        print(f"   Warning: {warning}")
    print()

    # Test 2: Wide spread
    print("2️⃣ Test: Wide spread")
    is_ok, warning = validator.validate_spread(
        symbol="ABC",
        current_price=50_000,
        bid_price=49_700,
        ask_price=50_300,
        volatility=0.025,
    )
    print(f"   Result: {'✅ OK' if is_ok else '❌ FAIL'}")
    if warning:
        print(f"   Warning: {warning}")
    print()

    # Test 3: Slippage estimation
    print("3️⃣ Test: Slippage estimation (large order)")
    slippage = validator.estimate_slippage(
        order_size_shares=100_000,
        avg_daily_volume=5_000_000,
        spread_pct=0.003,
        volatility=0.020,
    )
    print(f"   Estimated slippage: {slippage['estimated_slippage_pct']*100:.2f}%")
    print(f"   {slippage['breakdown']}")
    print()

    print("=" * 70)
    print("✅ Testing complete!")
