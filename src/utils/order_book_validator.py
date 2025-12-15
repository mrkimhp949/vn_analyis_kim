# -*- coding: utf-8 -*-
"""
Order Book Depth Validator for Vietnam Stock Market

IMPROVED v10.0: Complete order book validation for large orders.

Features:
- Order size vs depth validation
- Market impact estimation
- Optimal order splitting
- VWAP execution planning
- Iceberg order detection

Vietnam Market Specifics:
- 3 best bid/ask levels typically visible
- Lot size: 100 shares
- Price limits: ±7% (HOSE), ±10% (HNX), ±15% (UPCOM)
- ATO/ATC auction impact

Author: Trading Bot Team
Version: 10.0.0
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime, time
from typing import Dict, List, Optional, Tuple
from enum import Enum

logger = logging.getLogger(__name__)


class OrderImpactLevel(Enum):
    """Order impact classification"""

    NEGLIGIBLE = "NEGLIGIBLE"  # < 1% of depth
    LOW = "LOW"  # 1-5% of depth
    MODERATE = "MODERATE"  # 5-10% of depth
    HIGH = "HIGH"  # 10-25% of depth
    SEVERE = "SEVERE"  # > 25% of depth


@dataclass
class OrderBookLevel:
    """Single price level in order book"""

    price: float
    volume: int
    num_orders: int = 1


@dataclass
class OrderBook:
    """
    Order book snapshot for a symbol.

    Vietnam market typically shows 3 best bid/ask levels.
    """

    symbol: str
    timestamp: datetime
    bids: List[OrderBookLevel] = field(default_factory=list)  # Sorted desc by price
    asks: List[OrderBookLevel] = field(default_factory=list)  # Sorted asc by price
    last_price: float = 0.0
    reference_price: float = 0.0  # Previous close

    @property
    def best_bid(self) -> Optional[float]:
        return self.bids[0].price if self.bids else None

    @property
    def best_ask(self) -> Optional[float]:
        return self.asks[0].price if self.asks else None

    @property
    def spread(self) -> float:
        if self.best_bid and self.best_ask:
            return self.best_ask - self.best_bid
        return 0.0

    @property
    def spread_pct(self) -> float:
        if self.best_bid and self.best_ask and self.best_bid > 0:
            return (self.spread / self.best_bid) * 100
        return 0.0

    @property
    def total_bid_volume(self) -> int:
        return sum(level.volume for level in self.bids)

    @property
    def total_ask_volume(self) -> int:
        return sum(level.volume for level in self.asks)

    @property
    def bid_ask_imbalance(self) -> float:
        """
        Bid/Ask volume imbalance ratio.

        > 1.0 = More buying pressure
        < 1.0 = More selling pressure
        """
        total_ask = self.total_ask_volume
        if total_ask == 0:
            return float("inf")
        return self.total_bid_volume / total_ask


@dataclass
class OrderValidationResult:
    """Result of order book validation"""

    is_valid: bool
    can_execute: bool
    impact_level: OrderImpactLevel
    estimated_fill_price: float
    estimated_slippage_pct: float
    market_impact_pct: float
    warnings: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    split_orders: List[Dict] = field(default_factory=list)  # Recommended order splits


class OrderBookValidator:
    """
    Order Book Depth Validator for Vietnam Market.

    IMPROVED v10.0: Complete order book analysis.

    Features:
    - Validate order size against visible depth
    - Estimate market impact
    - Suggest order splitting for large orders
    - Detect potential manipulation (spoofing)

    Usage:
        validator = OrderBookValidator()

        # Create order book from market data
        order_book = OrderBook(
            symbol="VNM",
            timestamp=datetime.now(),
            bids=[
                OrderBookLevel(price=85000, volume=50000),
                OrderBookLevel(price=84900, volume=30000),
                OrderBookLevel(price=84800, volume=20000),
            ],
            asks=[
                OrderBookLevel(price=85100, volume=40000),
                OrderBookLevel(price=85200, volume=25000),
                OrderBookLevel(price=85300, volume=15000),
            ],
        )

        # Validate order
        result = validator.validate_order(order_book, shares=10000, side="BUY")
    """

    # Thresholds for impact levels
    IMPACT_THRESHOLDS = {
        "NEGLIGIBLE": 0.01,  # < 1% of depth
        "LOW": 0.05,  # 1-5% of depth
        "MODERATE": 0.10,  # 5-10% of depth
        "HIGH": 0.25,  # 10-25% of depth
        # > 25% = SEVERE
    }

    # Maximum order size as percentage of visible depth
    MAX_ORDER_PCT_OF_DEPTH = 0.10  # 10% of visible depth

    # Vietnam lot size
    LOT_SIZE = 100

    def __init__(
        self,
        max_order_pct_of_depth: float = MAX_ORDER_PCT_OF_DEPTH,
        max_slippage_pct: float = 0.5,  # 0.5% max acceptable slippage
    ):
        """
        Initialize Order Book Validator.

        Args:
            max_order_pct_of_depth: Max order size as % of visible depth
            max_slippage_pct: Maximum acceptable slippage percentage
        """
        self.max_order_pct_of_depth = max_order_pct_of_depth
        self.max_slippage_pct = max_slippage_pct

    def validate_order(
        self,
        order_book: OrderBook,
        shares: int,
        side: str,  # "BUY" or "SELL"
        limit_price: Optional[float] = None,
    ) -> OrderValidationResult:
        """
        Validate order against order book depth.

        Args:
            order_book: Current order book snapshot
            shares: Number of shares to trade
            side: "BUY" or "SELL"
            limit_price: Limit price (optional, for limit orders)

        Returns:
            OrderValidationResult with validation details
        """
        warnings = []
        recommendations = []

        # Get relevant side of order book
        if side.upper() == "BUY":
            levels = order_book.asks
            total_depth = order_book.total_ask_volume
            best_price = order_book.best_ask
        else:
            levels = order_book.bids
            total_depth = order_book.total_bid_volume
            best_price = order_book.best_bid

        if not levels or total_depth == 0:
            return OrderValidationResult(
                is_valid=False,
                can_execute=False,
                impact_level=OrderImpactLevel.SEVERE,
                estimated_fill_price=0,
                estimated_slippage_pct=0,
                market_impact_pct=0,
                warnings=["No liquidity available on this side of the order book"],
            )

        # Calculate order as percentage of visible depth
        order_pct_of_depth = shares / total_depth

        # Determine impact level
        impact_level = self._get_impact_level(order_pct_of_depth)

        # Estimate fill price and slippage
        fill_price, slippage_pct = self._estimate_fill_price(levels, shares, best_price, side)

        # Market impact estimation (simplified model)
        # Market impact ≈ 0.5 * sqrt(order_size / avg_volume) for VN market
        market_impact_pct = 0.5 * (order_pct_of_depth**0.5) * 100

        # Generate warnings
        if order_pct_of_depth > self.max_order_pct_of_depth:
            warnings.append(
                f"⚠️ Large order: {order_pct_of_depth:.1%} of visible depth "
                f"(max recommended: {self.max_order_pct_of_depth:.1%})"
            )

        if slippage_pct > self.max_slippage_pct:
            warnings.append(
                f"⚠️ High slippage risk: {slippage_pct:.2f}% "
                f"(max acceptable: {self.max_slippage_pct:.1f}%)"
            )

        if impact_level in [OrderImpactLevel.HIGH, OrderImpactLevel.SEVERE]:
            warnings.append(f"🚨 {impact_level.value} market impact expected")

        if order_book.spread_pct > 0.5:
            warnings.append(f"⚠️ Wide spread: {order_book.spread_pct:.2f}%")

        # Generate recommendations
        if order_pct_of_depth > 0.05:  # > 5% of depth
            split_orders = self._suggest_order_splits(shares, total_depth, best_price)
            recommendations.append(f"Consider splitting into {len(split_orders)} smaller orders")
        else:
            split_orders = []

        if impact_level == OrderImpactLevel.SEVERE:
            recommendations.append("Use VWAP or TWAP execution strategy")
            recommendations.append("Consider executing over multiple sessions")

        if side == "BUY" and order_book.bid_ask_imbalance < 0.8:
            recommendations.append("⚠️ Selling pressure detected - consider waiting")
        elif side == "SELL" and order_book.bid_ask_imbalance > 1.2:
            recommendations.append("⚠️ Buying pressure detected - may get better price")

        # Determine if order is valid
        is_valid = (
            order_pct_of_depth <= self.max_order_pct_of_depth
            and slippage_pct <= self.max_slippage_pct
        )

        can_execute = shares <= total_depth

        return OrderValidationResult(
            is_valid=is_valid,
            can_execute=can_execute,
            impact_level=impact_level,
            estimated_fill_price=fill_price,
            estimated_slippage_pct=slippage_pct,
            market_impact_pct=market_impact_pct,
            warnings=warnings,
            recommendations=recommendations,
            split_orders=split_orders,
        )

    def _get_impact_level(self, order_pct: float) -> OrderImpactLevel:
        """Determine impact level from order percentage of depth."""
        if order_pct < self.IMPACT_THRESHOLDS["NEGLIGIBLE"]:
            return OrderImpactLevel.NEGLIGIBLE
        elif order_pct < self.IMPACT_THRESHOLDS["LOW"]:
            return OrderImpactLevel.LOW
        elif order_pct < self.IMPACT_THRESHOLDS["MODERATE"]:
            return OrderImpactLevel.MODERATE
        elif order_pct < self.IMPACT_THRESHOLDS["HIGH"]:
            return OrderImpactLevel.HIGH
        else:
            return OrderImpactLevel.SEVERE

    def _estimate_fill_price(
        self,
        levels: List[OrderBookLevel],
        shares: int,
        best_price: float,
        side: str,
    ) -> Tuple[float, float]:
        """
        Estimate fill price by walking through order book levels.

        Returns:
            (estimated_fill_price, slippage_percentage)
        """
        remaining = shares
        total_cost = 0.0

        for level in levels:
            fill_at_level = min(remaining, level.volume)
            total_cost += fill_at_level * level.price
            remaining -= fill_at_level

            if remaining <= 0:
                break

        if shares > 0:
            avg_fill_price = (
                total_cost / (shares - remaining) if (shares - remaining) > 0 else best_price
            )
        else:
            avg_fill_price = best_price

        # Calculate slippage
        if best_price > 0:
            if side == "BUY":
                slippage_pct = ((avg_fill_price - best_price) / best_price) * 100
            else:
                slippage_pct = ((best_price - avg_fill_price) / best_price) * 100
        else:
            slippage_pct = 0.0

        return avg_fill_price, max(0, slippage_pct)

    def _suggest_order_splits(
        self,
        total_shares: int,
        total_depth: int,
        base_price: float,
    ) -> List[Dict]:
        """
        Suggest order splits for large orders.

        Returns list of recommended smaller orders.
        """
        # Target each split to be < 3% of visible depth
        target_pct = 0.03
        target_size = int(total_depth * target_pct)

        # Round to lot size
        target_size = max(self.LOT_SIZE, (target_size // self.LOT_SIZE) * self.LOT_SIZE)

        splits = []
        remaining = total_shares

        while remaining > 0:
            split_size = min(remaining, target_size)
            split_size = (split_size // self.LOT_SIZE) * self.LOT_SIZE

            if split_size > 0:
                splits.append(
                    {
                        "shares": split_size,
                        "estimated_pct_of_depth": split_size / total_depth * 100,
                        "delay_minutes": len(splits) * 5,  # 5 minutes between splits
                    }
                )

            remaining -= split_size

            if split_size == 0:
                break

        return splits

    def detect_spoofing(
        self,
        order_book: OrderBook,
        historical_books: List[OrderBook] = None,
    ) -> Dict:
        """
        Detect potential spoofing/layering in order book.

        Spoofing indicators:
        - Large orders that disappear quickly
        - Imbalanced depth that shifts rapidly
        - Orders at unusual price levels

        Args:
            order_book: Current order book
            historical_books: Previous order book snapshots

        Returns:
            Dict with spoofing analysis
        """
        result = {
            "spoofing_detected": False,
            "confidence": 0.0,
            "indicators": [],
        }

        # Check for unusual depth imbalance
        imbalance = order_book.bid_ask_imbalance

        if imbalance > 3.0:
            result["indicators"].append(
                f"Extreme bid imbalance: {imbalance:.2f}x (possible fake bids)"
            )
            result["confidence"] += 0.3

        if imbalance < 0.33:
            result["indicators"].append(
                f"Extreme ask imbalance: {1/imbalance:.2f}x (possible fake asks)"
            )
            result["confidence"] += 0.3

        # Check for unusually large orders at single level
        for level in order_book.bids[:3]:
            if level.volume > order_book.total_bid_volume * 0.5:
                result["indicators"].append(
                    f"Single large bid: {level.volume:,} shares at {level.price:,.0f}"
                )
                result["confidence"] += 0.2

        for level in order_book.asks[:3]:
            if level.volume > order_book.total_ask_volume * 0.5:
                result["indicators"].append(
                    f"Single large ask: {level.volume:,} shares at {level.price:,.0f}"
                )
                result["confidence"] += 0.2

        # Historical analysis (if available)
        if historical_books and len(historical_books) >= 3:
            # Check for disappearing large orders
            prev_book = historical_books[-1]

            # Compare depth changes
            bid_change = abs(order_book.total_bid_volume - prev_book.total_bid_volume) / max(
                prev_book.total_bid_volume, 1
            )

            ask_change = abs(order_book.total_ask_volume - prev_book.total_ask_volume) / max(
                prev_book.total_ask_volume, 1
            )

            if bid_change > 0.5:  # > 50% change in bid depth
                result["indicators"].append(f"Rapid bid depth change: {bid_change:.1%}")
                result["confidence"] += 0.25

            if ask_change > 0.5:
                result["indicators"].append(f"Rapid ask depth change: {ask_change:.1%}")
                result["confidence"] += 0.25

        result["spoofing_detected"] = result["confidence"] >= 0.5
        result["confidence"] = min(1.0, result["confidence"])

        return result


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================


def validate_order_vs_depth(
    symbol: str,
    shares: int,
    side: str,
    order_book: Optional[OrderBook] = None,
) -> OrderValidationResult:
    """
    Quick validation of order against order book depth.

    Args:
        symbol: Stock symbol
        shares: Number of shares
        side: "BUY" or "SELL"
        order_book: Order book (optional, will use mock if not provided)

    Returns:
        OrderValidationResult
    """
    validator = OrderBookValidator()

    if order_book is None:
        # Create mock order book for testing
        # In production, this should fetch real order book data
        order_book = _create_mock_order_book(symbol)

    return validator.validate_order(order_book, shares, side)


def _create_mock_order_book(symbol: str) -> OrderBook:
    """Create mock order book for testing."""
    import random

    base_price = 50000  # Mock base price

    return OrderBook(
        symbol=symbol,
        timestamp=datetime.now(),
        bids=[
            OrderBookLevel(price=base_price - i * 100, volume=random.randint(10000, 50000))
            for i in range(3)
        ],
        asks=[
            OrderBookLevel(price=base_price + (i + 1) * 100, volume=random.randint(10000, 50000))
            for i in range(3)
        ],
        last_price=base_price,
        reference_price=base_price - 500,
    )


def estimate_market_impact(
    shares: int,
    avg_daily_volume: float,
    volatility: float = 0.02,
) -> float:
    """
    Estimate market impact using simplified model.

    Args:
        shares: Order size in shares
        avg_daily_volume: Average daily volume
        volatility: Daily volatility (default 2%)

    Returns:
        Estimated market impact as percentage
    """
    if avg_daily_volume <= 0:
        return 0.0

    # Simplified market impact model:
    # Impact = volatility * sqrt(order_size / avg_volume)
    participation_rate = shares / avg_daily_volume
    impact = volatility * (participation_rate**0.5) * 100

    return min(7.0, impact)  # Cap at 7% (VN price limit)


class AdvancedSlippageEstimator:
    """
    Advanced slippage estimator using order book depth analysis.

    This class provides more accurate slippage estimation by combining:
    - Order book depth analysis
    - Market impact models
    - Urgency adjustments
    - Time of day factors (Vietnam trading sessions)

    Usage:
        estimator = AdvancedSlippageEstimator()

        # With order book
        slippage = estimator.estimate_slippage(
            symbol="VNM",
            shares=10000,
            side="BUY",
            order_book=order_book,
        )

        # Without order book (uses statistical model)
        slippage = estimator.estimate_slippage(
            symbol="VNM",
            shares=10000,
            side="BUY",
            avg_daily_volume=500000,
            avg_spread=0.15,
        )
    """

    # Session-based slippage multipliers (Vietnam market)
    SESSION_MULTIPLIERS = {
        "ATO": 1.5,  # High volatility at open
        "AM_SESSION": 1.0,  # Normal morning session
        "LUNCH": 0.8,  # Lower activity during lunch
        "PM_SESSION": 1.0,  # Normal afternoon session
        "ATC": 1.3,  # Higher volatility at close
    }

    # Urgency-based slippage adjustments
    URGENCY_MULTIPLIERS = {
        "low": 0.7,  # Patient execution
        "normal": 1.0,  # Standard execution
        "high": 1.3,  # Need to execute quickly
        "immediate": 1.6,  # Must execute now
    }

    def __init__(
        self,
        base_slippage_bps: float = 10.0,  # Base slippage in basis points
        use_order_book: bool = True,
    ):
        """
        Initialize Advanced Slippage Estimator.

        Args:
            base_slippage_bps: Base slippage in basis points (default 10 = 0.1%)
            use_order_book: Whether to use order book depth when available
        """
        self.base_slippage_bps = base_slippage_bps
        self.use_order_book = use_order_book
        self._order_book_validator = OrderBookValidator()

    def get_current_session(self) -> str:
        """Determine current trading session based on Vietnam time."""
        from datetime import datetime
        import pytz

        vn_tz = pytz.timezone("Asia/Ho_Chi_Minh")
        now = datetime.now(vn_tz).time()

        # Vietnam trading hours
        if time(9, 0) <= now < time(9, 15):
            return "ATO"
        elif time(9, 15) <= now < time(11, 30):
            return "AM_SESSION"
        elif time(11, 30) <= now < time(13, 0):
            return "LUNCH"
        elif time(13, 0) <= now < time(14, 30):
            return "PM_SESSION"
        elif time(14, 30) <= now < time(15, 0):
            return "ATC"
        else:
            return "CLOSED"

    def estimate_slippage(
        self,
        symbol: str,
        shares: int,
        side: str,
        order_book: Optional[OrderBook] = None,
        avg_daily_volume: Optional[float] = None,
        avg_spread: Optional[float] = None,  # Spread as percentage
        urgency: str = "normal",
        volatility: Optional[float] = None,  # Current volatility if known
    ) -> Dict:
        """
        Estimate slippage for an order.

        Args:
            symbol: Stock symbol
            shares: Number of shares to trade
            side: "BUY" or "SELL"
            order_book: Order book snapshot (optional but recommended)
            avg_daily_volume: Average daily volume (optional)
            avg_spread: Average bid-ask spread as percentage (optional)
            urgency: Execution urgency ("low", "normal", "high", "immediate")
            volatility: Current volatility (optional)

        Returns:
            Dict with slippage estimates and breakdown
        """
        result = {
            "symbol": symbol,
            "shares": shares,
            "side": side,
            "estimated_slippage_pct": 0.0,
            "estimated_slippage_bps": 0.0,
            "breakdown": {},
            "confidence": "low",
            "recommendations": [],
        }

        # Get session multiplier
        session = self.get_current_session()
        session_mult = self.SESSION_MULTIPLIERS.get(session, 1.0)

        # Get urgency multiplier
        urgency_mult = self.URGENCY_MULTIPLIERS.get(urgency, 1.0)

        # Start with base slippage
        total_slippage_bps = self.base_slippage_bps
        breakdown = {"base": self.base_slippage_bps}

        # Method 1: Order book based estimation (most accurate)
        if order_book is not None and self.use_order_book:
            validation = self._order_book_validator.validate_order(order_book, shares, side)

            # Add order book slippage
            ob_slippage_bps = validation.estimated_slippage_pct * 100  # Convert to bps
            breakdown["order_book"] = ob_slippage_bps
            total_slippage_bps += ob_slippage_bps

            # Add market impact
            breakdown["market_impact"] = validation.market_impact_pct * 100
            total_slippage_bps += validation.market_impact_pct * 100

            result["confidence"] = "high"
            result["order_book_analysis"] = {
                "impact_level": validation.impact_level.value,
                "can_execute": validation.can_execute,
                "warnings": validation.warnings,
            }

            if validation.split_orders:
                result["recommendations"].append(
                    f"Consider splitting into {len(validation.split_orders)} orders"
                )

        # Method 2: Statistical model (when no order book)
        elif avg_daily_volume is not None:
            # Participation rate based slippage
            participation_rate = shares / avg_daily_volume if avg_daily_volume > 0 else 0

            # Square-root impact model
            impact_bps = 50 * (participation_rate**0.5)  # 50 bps for 1% participation
            breakdown["participation_impact"] = impact_bps
            total_slippage_bps += impact_bps

            result["confidence"] = "medium"

            if participation_rate > 0.01:
                result["recommendations"].append(
                    f"High participation rate ({participation_rate:.1%}) - consider splitting"
                )

        # Add spread component
        if avg_spread is not None:
            # Half the spread is typically paid
            spread_bps = avg_spread * 50  # Convert % to bps and take half
            breakdown["spread"] = spread_bps
            total_slippage_bps += spread_bps

        # Add volatility component
        if volatility is not None:
            vol_bps = volatility * 100 * 0.3  # 30% of daily volatility as slippage risk
            breakdown["volatility"] = vol_bps
            total_slippage_bps += vol_bps

        # Apply multipliers
        breakdown["session_adjustment"] = total_slippage_bps * (session_mult - 1)
        breakdown["urgency_adjustment"] = total_slippage_bps * (urgency_mult - 1)

        total_slippage_bps *= session_mult * urgency_mult

        # Cap at reasonable level (500 bps = 5%)
        total_slippage_bps = min(500, total_slippage_bps)

        result["estimated_slippage_bps"] = round(total_slippage_bps, 2)
        result["estimated_slippage_pct"] = round(total_slippage_bps / 100, 4)
        result["breakdown"] = breakdown
        result["session"] = session
        result["multipliers"] = {
            "session": session_mult,
            "urgency": urgency_mult,
        }

        # Add recommendations based on slippage level
        if total_slippage_bps > 100:  # > 1%
            result["recommendations"].append("⚠️ High slippage expected - use limit orders")
        if total_slippage_bps > 200:  # > 2%
            result["recommendations"].append("🔴 Very high slippage - consider reducing order size")

        return result

    def get_optimal_execution_strategy(
        self,
        symbol: str,
        shares: int,
        side: str,
        order_book: Optional[OrderBook] = None,
        avg_daily_volume: Optional[float] = None,
        max_slippage_pct: float = 0.5,
    ) -> Dict:
        """
        Get optimal execution strategy to minimize slippage.

        Returns:
            Dict with execution strategy recommendations
        """
        # Estimate slippage with different approaches
        estimates = {}

        # Single order
        single = self.estimate_slippage(symbol, shares, side, order_book, avg_daily_volume)
        estimates["single_order"] = single["estimated_slippage_pct"]

        # Calculate participation rate
        participation_rate = 0.0
        if avg_daily_volume and avg_daily_volume > 0:
            participation_rate = shares / avg_daily_volume

        # Determine optimal strategy
        if participation_rate > 0.05:  # > 5% of daily volume
            strategy = "VWAP"
            reason = "Order size > 5% of ADV - use VWAP over full session"
            num_splits = max(5, int(participation_rate * 20))
        elif participation_rate > 0.02:  # 2-5%
            strategy = "TWAP"
            reason = "Order size 2-5% of ADV - use TWAP with 3-5 splits"
            num_splits = 4
        elif participation_rate > 0.01:  # 1-2%
            strategy = "SPLIT"
            reason = "Order size 1-2% of ADV - split into 2-3 orders"
            num_splits = 2
        else:
            strategy = "SINGLE"
            reason = "Order size < 1% of ADV - single market order OK"
            num_splits = 1

        # Calculate split orders
        split_size = shares // num_splits
        split_size = max(100, (split_size // 100) * 100)  # Round to lot size

        splits = []
        remaining = shares
        for i in range(num_splits):
            size = min(remaining, split_size)
            if i == num_splits - 1:
                size = remaining  # Last split gets remainder

            size = (size // 100) * 100  # Round to lot
            if size > 0:
                splits.append(
                    {
                        "order_num": i + 1,
                        "shares": size,
                        "delay_minutes": i * 5 if strategy == "TWAP" else i * 15,
                    }
                )
                remaining -= size

        # Estimate slippage reduction
        split_slippage = single["estimated_slippage_pct"] * (0.7 ** (num_splits - 1))

        return {
            "strategy": strategy,
            "reason": reason,
            "original_slippage_pct": single["estimated_slippage_pct"],
            "estimated_reduced_slippage_pct": round(split_slippage, 4),
            "slippage_reduction_pct": round(
                (
                    (1 - split_slippage / single["estimated_slippage_pct"]) * 100
                    if single["estimated_slippage_pct"] > 0
                    else 0
                ),
                1,
            ),
            "num_splits": num_splits,
            "splits": splits,
            "participation_rate": round(participation_rate * 100, 2),
        }
