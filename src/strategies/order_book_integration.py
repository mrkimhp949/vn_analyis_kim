# -*- coding: utf-8 -*-
"""
Order Book Integration for Entry Timing - Vietnam Market

Integrates order book depth analysis into entry timing decisions:
1. Analyze bid/ask spread and depth
2. Detect large orders and institutional activity
3. Optimize entry price using order book signals
4. Real-time slippage estimation
5. Market impact prediction

Vietnam Market Order Book Characteristics:
- 3 best bid/ask levels visible
- Order types: LO (Limit), ATO, ATC, MP (Market)
- Minimum order: 100 shares (1 lot)
- Price step: 10/50/100 VND depending on price

Author: Trading Bot Team
Version: 1.0.0
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime, time
from enum import Enum
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# =============================================================================
# CONSTANTS & ENUMS
# =============================================================================


class OrderBookSignal(Enum):
    """Order book derived signal"""

    STRONG_BUY_PRESSURE = "STRONG_BUY_PRESSURE"
    BUY_PRESSURE = "BUY_PRESSURE"
    NEUTRAL = "NEUTRAL"
    SELL_PRESSURE = "SELL_PRESSURE"
    STRONG_SELL_PRESSURE = "STRONG_SELL_PRESSURE"


class EntryTimingSignal(Enum):
    """Entry timing recommendation"""

    ENTER_NOW = "ENTER_NOW"  # Good entry point
    WAIT_FOR_DIP = "WAIT_FOR_DIP"  # Wait for better price
    SCALE_IN = "SCALE_IN"  # Enter in tranches
    AVOID = "AVOID"  # Poor timing
    USE_LIMIT_ORDER = "USE_LIMIT_ORDER"  # Use limit instead of market


class OrderBookImbalance(Enum):
    """Order book imbalance level"""

    HEAVY_BID = "HEAVY_BID"  # Much more buying interest
    SLIGHT_BID = "SLIGHT_BID"
    BALANCED = "BALANCED"
    SLIGHT_ASK = "SLIGHT_ASK"
    HEAVY_ASK = "HEAVY_ASK"  # Much more selling interest


@dataclass
class OrderBookLevel:
    """Single price level in order book"""

    price: float
    volume: int  # Number of shares
    order_count: int  # Number of orders at this level


@dataclass
class OrderBook:
    """Full order book snapshot"""

    symbol: str
    timestamp: datetime
    bids: List[OrderBookLevel]  # Best bid first (highest price)
    asks: List[OrderBookLevel]  # Best ask first (lowest price)
    last_price: float
    last_volume: int

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
        return 0

    @property
    def spread_pct(self) -> float:
        if self.best_bid and self.spread:
            return self.spread / self.best_bid
        return 0

    @property
    def mid_price(self) -> float:
        if self.best_bid and self.best_ask:
            return (self.best_bid + self.best_ask) / 2
        return self.last_price

    @property
    def total_bid_volume(self) -> int:
        return sum(level.volume for level in self.bids)

    @property
    def total_ask_volume(self) -> int:
        return sum(level.volume for level in self.asks)


@dataclass
class OrderBookAnalysis:
    """Results of order book analysis"""

    symbol: str
    timestamp: datetime
    spread_bps: float  # Spread in basis points
    imbalance: OrderBookImbalance
    imbalance_ratio: float  # Bid volume / Ask volume
    signal: OrderBookSignal
    large_orders_detected: bool
    estimated_slippage_pct: float
    recommended_entry_price: float
    entry_timing: EntryTimingSignal
    confidence: float
    notes: List[str]


@dataclass
class OrderBookConfig:
    """Configuration for order book analysis"""

    # Spread thresholds
    tight_spread_bps: float = 10  # 0.1% = tight spread
    wide_spread_bps: float = 50  # 0.5% = wide spread

    # Imbalance thresholds
    heavy_imbalance_ratio: float = 2.0  # 2:1 = heavy imbalance
    slight_imbalance_ratio: float = 1.3  # 1.3:1 = slight imbalance

    # Large order detection
    large_order_threshold: float = 0.05  # 5% of ADV = large order
    institutional_order_threshold: float = 0.10  # 10% of ADV = institutional

    # Entry optimization
    max_slippage_for_market_order: float = 0.003  # 0.3% max slippage for market
    min_depth_for_entry: int = 10000  # Min 10K shares at best bid

    # Timing parameters
    avoid_entry_spread_bps: float = 100  # Avoid entry if spread > 1%
    optimal_entry_spread_bps: float = 20  # Optimal entry if spread < 0.2%


class OrderBookIntegration:
    """
    Integrates order book analysis into entry timing

    Key Features:
    1. Spread analysis for optimal entry timing
    2. Depth analysis for slippage estimation
    3. Imbalance analysis for direction bias
    4. Large order detection for institutional activity
    5. Dynamic entry price recommendation
    """

    def __init__(self, config: Optional[OrderBookConfig] = None):
        self.config = config or OrderBookConfig()

        # Cache
        self._order_book_cache: Dict[str, OrderBook] = {}
        self._analysis_cache: Dict[str, OrderBookAnalysis] = {}

        # Historical data for pattern detection
        self._imbalance_history: Dict[str, List[Tuple[datetime, float]]] = {}
        self._spread_history: Dict[str, List[Tuple[datetime, float]]] = {}

        # ADV cache for large order detection
        self._adv_cache: Dict[str, float] = {}

    def update_order_book(
        self,
        symbol: str,
        bids: List[Tuple[float, int, int]],  # (price, volume, order_count)
        asks: List[Tuple[float, int, int]],
        last_price: float,
        last_volume: int,
    ) -> OrderBook:
        """
        Update order book data for a symbol

        Args:
            symbol: Stock symbol
            bids: List of (price, volume, order_count) tuples, best first
            asks: List of (price, volume, order_count) tuples, best first
            last_price: Last traded price
            last_volume: Last traded volume

        Returns:
            Updated OrderBook object
        """
        bid_levels = [OrderBookLevel(price=p, volume=v, order_count=c) for p, v, c in bids]
        ask_levels = [OrderBookLevel(price=p, volume=v, order_count=c) for p, v, c in asks]

        order_book = OrderBook(
            symbol=symbol,
            timestamp=datetime.now(),
            bids=bid_levels,
            asks=ask_levels,
            last_price=last_price,
            last_volume=last_volume,
        )

        self._order_book_cache[symbol] = order_book

        # Update history
        self._update_history(symbol, order_book)

        return order_book

    def _update_history(self, symbol: str, order_book: OrderBook):
        """Update historical data for pattern detection"""
        timestamp = order_book.timestamp

        # Imbalance history
        if order_book.total_ask_volume > 0:
            imbalance = order_book.total_bid_volume / order_book.total_ask_volume
        else:
            imbalance = 1.0

        if symbol not in self._imbalance_history:
            self._imbalance_history[symbol] = []
        self._imbalance_history[symbol].append((timestamp, imbalance))

        # Keep last 100 snapshots
        if len(self._imbalance_history[symbol]) > 100:
            self._imbalance_history[symbol] = self._imbalance_history[symbol][-100:]

        # Spread history
        spread_bps = order_book.spread_pct * 10000

        if symbol not in self._spread_history:
            self._spread_history[symbol] = []
        self._spread_history[symbol].append((timestamp, spread_bps))

        if len(self._spread_history[symbol]) > 100:
            self._spread_history[symbol] = self._spread_history[symbol][-100:]

    def set_adv(self, symbol: str, average_daily_volume: float):
        """Set average daily volume for large order detection"""
        self._adv_cache[symbol] = average_daily_volume

    def analyze_order_book(
        self, symbol: str, order_size: int = 1000, side: str = "BUY"
    ) -> Optional[OrderBookAnalysis]:
        """
        Comprehensive order book analysis

        Args:
            symbol: Stock symbol
            order_size: Intended order size in shares
            side: "BUY" or "SELL"

        Returns:
            OrderBookAnalysis with recommendations
        """
        order_book = self._order_book_cache.get(symbol)
        if not order_book or not order_book.bids or not order_book.asks:
            return None

        notes = []

        # 1. Spread analysis
        spread_bps = order_book.spread_pct * 10000

        if spread_bps < self.config.tight_spread_bps:
            notes.append(f"Tight spread: {spread_bps:.1f} bps - good liquidity")
        elif spread_bps > self.config.wide_spread_bps:
            notes.append(f"Wide spread: {spread_bps:.1f} bps - consider limit order")

        # 2. Imbalance analysis
        imbalance_ratio = (
            order_book.total_bid_volume / order_book.total_ask_volume
            if order_book.total_ask_volume > 0
            else 1.0
        )

        if imbalance_ratio >= self.config.heavy_imbalance_ratio:
            imbalance = OrderBookImbalance.HEAVY_BID
            notes.append(f"Heavy buying pressure: {imbalance_ratio:.1f}x bid/ask ratio")
        elif imbalance_ratio >= self.config.slight_imbalance_ratio:
            imbalance = OrderBookImbalance.SLIGHT_BID
        elif imbalance_ratio <= 1 / self.config.heavy_imbalance_ratio:
            imbalance = OrderBookImbalance.HEAVY_ASK
            notes.append(f"Heavy selling pressure: {1/imbalance_ratio:.1f}x ask/bid ratio")
        elif imbalance_ratio <= 1 / self.config.slight_imbalance_ratio:
            imbalance = OrderBookImbalance.SLIGHT_ASK
        else:
            imbalance = OrderBookImbalance.BALANCED

        # 3. Signal from imbalance
        signal = self._imbalance_to_signal(imbalance)

        # 4. Large order detection
        adv = self._adv_cache.get(symbol, 1_000_000)  # Default 1M ADV
        large_orders = self._detect_large_orders(order_book, adv)

        if large_orders:
            notes.append("Large orders detected - possible institutional activity")

        # 5. Slippage estimation
        estimated_slippage = self._estimate_slippage(order_book, order_size, side)

        # 6. Entry timing recommendation
        entry_timing, entry_price = self._recommend_entry(
            order_book, spread_bps, imbalance, estimated_slippage, side
        )

        # 7. Calculate confidence
        confidence = self._calculate_confidence(
            spread_bps, imbalance_ratio, large_orders, estimated_slippage
        )

        analysis = OrderBookAnalysis(
            symbol=symbol,
            timestamp=datetime.now(),
            spread_bps=spread_bps,
            imbalance=imbalance,
            imbalance_ratio=imbalance_ratio,
            signal=signal,
            large_orders_detected=large_orders,
            estimated_slippage_pct=estimated_slippage,
            recommended_entry_price=entry_price,
            entry_timing=entry_timing,
            confidence=confidence,
            notes=notes,
        )

        self._analysis_cache[symbol] = analysis
        return analysis

    def _imbalance_to_signal(self, imbalance: OrderBookImbalance) -> OrderBookSignal:
        """Convert imbalance to trading signal"""
        mapping = {
            OrderBookImbalance.HEAVY_BID: OrderBookSignal.STRONG_BUY_PRESSURE,
            OrderBookImbalance.SLIGHT_BID: OrderBookSignal.BUY_PRESSURE,
            OrderBookImbalance.BALANCED: OrderBookSignal.NEUTRAL,
            OrderBookImbalance.SLIGHT_ASK: OrderBookSignal.SELL_PRESSURE,
            OrderBookImbalance.HEAVY_ASK: OrderBookSignal.STRONG_SELL_PRESSURE,
        }
        return mapping.get(imbalance, OrderBookSignal.NEUTRAL)

    def _detect_large_orders(self, order_book: OrderBook, adv: float) -> bool:
        """Detect large orders indicating institutional activity"""
        large_threshold = adv * self.config.large_order_threshold

        # Check bid side
        for level in order_book.bids:
            if level.volume >= large_threshold:
                return True

        # Check ask side
        for level in order_book.asks:
            if level.volume >= large_threshold:
                return True

        return False

    def _estimate_slippage(self, order_book: OrderBook, order_size: int, side: str) -> float:
        """
        Estimate slippage for an order of given size

        Returns slippage as percentage of price
        """
        if side == "BUY":
            levels = order_book.asks
            reference_price = order_book.best_ask or order_book.last_price
        else:
            levels = order_book.bids
            reference_price = order_book.best_bid or order_book.last_price

        if not levels or reference_price <= 0:
            return 0.01  # Default 1% if no data

        remaining = order_size
        total_cost = 0

        for level in levels:
            if remaining <= 0:
                break

            fill_size = min(remaining, level.volume)
            total_cost += fill_size * level.price
            remaining -= fill_size

        # If order not fully filled by visible levels, add penalty
        if remaining > 0:
            # Assume remaining fills at 0.5% worse price
            worst_level = levels[-1] if levels else level
            penalty_price = worst_level.price * (1.005 if side == "BUY" else 0.995)
            total_cost += remaining * penalty_price

        avg_fill_price = total_cost / order_size
        slippage = abs(avg_fill_price - reference_price) / reference_price

        return slippage

    def _recommend_entry(
        self,
        order_book: OrderBook,
        spread_bps: float,
        imbalance: OrderBookImbalance,
        estimated_slippage: float,
        side: str,
    ) -> Tuple[EntryTimingSignal, float]:
        """
        Recommend entry timing and price

        Returns:
            (timing_signal, recommended_price)
        """
        # For BUY orders
        if side == "BUY":
            reference_price = order_book.best_ask

            # Wide spread - use limit order at mid or below
            if spread_bps > self.config.wide_spread_bps:
                entry_price = order_book.mid_price
                return EntryTimingSignal.USE_LIMIT_ORDER, entry_price

            # Very wide spread - avoid
            if spread_bps > self.config.avoid_entry_spread_bps:
                entry_price = order_book.best_bid
                return EntryTimingSignal.AVOID, entry_price

            # Heavy selling pressure - wait for dip
            if imbalance in [OrderBookImbalance.HEAVY_ASK]:
                entry_price = order_book.best_bid * 1.001  # Just above bid
                return EntryTimingSignal.WAIT_FOR_DIP, entry_price

            # Heavy buying pressure - enter now before price runs
            if imbalance in [OrderBookImbalance.HEAVY_BID, OrderBookImbalance.SLIGHT_BID]:
                entry_price = order_book.best_ask
                return EntryTimingSignal.ENTER_NOW, entry_price

            # High slippage expected - scale in
            if estimated_slippage > self.config.max_slippage_for_market_order:
                entry_price = order_book.mid_price
                return EntryTimingSignal.SCALE_IN, entry_price

            # Optimal conditions - tight spread, balanced
            if spread_bps < self.config.optimal_entry_spread_bps:
                entry_price = order_book.best_ask
                return EntryTimingSignal.ENTER_NOW, entry_price

            # Default - use limit order at mid
            entry_price = order_book.mid_price
            return EntryTimingSignal.USE_LIMIT_ORDER, entry_price

        # For SELL orders (mirror logic)
        else:
            reference_price = order_book.best_bid

            if spread_bps > self.config.wide_spread_bps:
                entry_price = order_book.mid_price
                return EntryTimingSignal.USE_LIMIT_ORDER, entry_price

            if imbalance in [OrderBookImbalance.HEAVY_BID]:
                entry_price = order_book.best_ask * 0.999
                return EntryTimingSignal.WAIT_FOR_DIP, entry_price  # Wait for rally

            if imbalance in [OrderBookImbalance.HEAVY_ASK, OrderBookImbalance.SLIGHT_ASK]:
                entry_price = order_book.best_bid
                return EntryTimingSignal.ENTER_NOW, entry_price

            entry_price = order_book.mid_price
            return EntryTimingSignal.USE_LIMIT_ORDER, entry_price

    def _calculate_confidence(
        self, spread_bps: float, imbalance_ratio: float, large_orders: bool, slippage: float
    ) -> float:
        """Calculate confidence in the analysis"""
        confidence = 0.5  # Base confidence

        # Tight spread increases confidence
        if spread_bps < self.config.tight_spread_bps:
            confidence += 0.15
        elif spread_bps < self.config.wide_spread_bps:
            confidence += 0.05
        else:
            confidence -= 0.10

        # Clear imbalance increases confidence
        if imbalance_ratio > 1.5 or imbalance_ratio < 0.67:
            confidence += 0.10

        # Large orders detected increases confidence
        if large_orders:
            confidence += 0.10

        # Low slippage increases confidence
        if slippage < 0.002:  # 0.2%
            confidence += 0.10
        elif slippage > 0.005:  # 0.5%
            confidence -= 0.10

        return max(0.1, min(0.95, confidence))

    def get_optimal_entry_price(
        self,
        symbol: str,
        order_size: int,
        side: str = "BUY",
        aggressiveness: float = 0.5,  # 0 = passive, 1 = aggressive
    ) -> Optional[float]:
        """
        Get optimal entry price based on order book state

        Args:
            symbol: Stock symbol
            order_size: Order size in shares
            side: "BUY" or "SELL"
            aggressiveness: How aggressive to be (0-1)

        Returns:
            Recommended limit price
        """
        order_book = self._order_book_cache.get(symbol)
        if not order_book:
            return None

        if side == "BUY":
            # Range from bid (passive) to ask (aggressive)
            price_range = order_book.spread
            optimal_price = order_book.best_bid + (price_range * aggressiveness)
        else:
            # Range from ask (passive) to bid (aggressive)
            price_range = order_book.spread
            optimal_price = order_book.best_ask - (price_range * aggressiveness)

        # Round to tick size (simplified - use vietnam_market module in production)
        if optimal_price >= 50000:
            tick = 100
        elif optimal_price >= 10000:
            tick = 50
        else:
            tick = 10

        optimal_price = round(optimal_price / tick) * tick

        return optimal_price

    def should_use_market_order(
        self, symbol: str, order_size: int, max_slippage: float = 0.005
    ) -> Tuple[bool, str]:
        """
        Determine if market order is appropriate

        Args:
            symbol: Stock symbol
            order_size: Order size in shares
            max_slippage: Maximum acceptable slippage

        Returns:
            (should_use_market, reason)
        """
        analysis = self._analysis_cache.get(symbol)
        if not analysis:
            analysis = self.analyze_order_book(symbol, order_size)

        if not analysis:
            return False, "No order book data available"

        # Check slippage
        if analysis.estimated_slippage_pct > max_slippage:
            return False, f"Slippage {analysis.estimated_slippage_pct:.2%} > {max_slippage:.2%}"

        # Check spread
        if analysis.spread_bps > self.config.wide_spread_bps:
            return False, f"Wide spread: {analysis.spread_bps:.0f} bps"

        # Check entry timing recommendation
        if analysis.entry_timing == EntryTimingSignal.ENTER_NOW:
            return True, "Good conditions for market order"
        elif analysis.entry_timing == EntryTimingSignal.USE_LIMIT_ORDER:
            return False, "Limit order recommended for better price"
        elif analysis.entry_timing == EntryTimingSignal.AVOID:
            return False, "Poor entry conditions"

        return False, "Use limit order for better execution"

    def get_entry_adjustment(
        self, symbol: str, base_confidence: float, order_size: int
    ) -> Tuple[int, str]:
        """
        Get confidence adjustment based on order book analysis

        Used to integrate with entry_logic.py

        Returns:
            (adjustment, reason) - adjustment in confidence points (-20 to +20)
        """
        analysis = self.analyze_order_book(symbol, order_size)

        if not analysis:
            return 0, "No order book data"

        adjustment = 0
        reasons = []

        # Spread-based adjustment
        if analysis.spread_bps < self.config.tight_spread_bps:
            adjustment += 10
            reasons.append("tight spread (+10)")
        elif analysis.spread_bps > self.config.wide_spread_bps:
            adjustment -= 10
            reasons.append("wide spread (-10)")

        # Imbalance-based adjustment (for BUY signals)
        if analysis.imbalance == OrderBookImbalance.HEAVY_BID:
            adjustment += 10
            reasons.append("strong buying pressure (+10)")
        elif analysis.imbalance == OrderBookImbalance.HEAVY_ASK:
            adjustment -= 15
            reasons.append("strong selling pressure (-15)")

        # Large order detection
        if analysis.large_orders_detected:
            if analysis.signal in [
                OrderBookSignal.STRONG_BUY_PRESSURE,
                OrderBookSignal.BUY_PRESSURE,
            ]:
                adjustment += 5
                reasons.append("institutional buying (+5)")
            elif analysis.signal in [
                OrderBookSignal.STRONG_SELL_PRESSURE,
                OrderBookSignal.SELL_PRESSURE,
            ]:
                adjustment -= 5
                reasons.append("institutional selling (-5)")

        # Slippage penalty
        if analysis.estimated_slippage_pct > 0.005:
            adjustment -= 5
            reasons.append("high slippage expected (-5)")

        reason = "; ".join(reasons) if reasons else "neutral order book"

        # Clamp to ±20
        adjustment = max(-20, min(20, adjustment))

        return adjustment, reason


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

_order_book_integration: Optional[OrderBookIntegration] = None


def get_order_book_integration(config: Optional[OrderBookConfig] = None) -> OrderBookIntegration:
    """Get singleton order book integration instance"""
    global _order_book_integration
    if _order_book_integration is None:
        _order_book_integration = OrderBookIntegration(config)
    return _order_book_integration


def analyze_entry_with_order_book(
    symbol: str,
    order_size: int,
    bids: List[Tuple[float, int, int]],
    asks: List[Tuple[float, int, int]],
    last_price: float,
) -> Dict[str, Any]:
    """
    Quick analysis for entry with order book data

    Returns dict with:
    - timing: Entry timing recommendation
    - price: Recommended entry price
    - slippage: Estimated slippage
    - use_market: Whether to use market order
    - adjustment: Confidence adjustment
    """
    integration = get_order_book_integration()

    # Update order book
    integration.update_order_book(
        symbol=symbol, bids=bids, asks=asks, last_price=last_price, last_volume=0
    )

    # Analyze
    analysis = integration.analyze_order_book(symbol, order_size, "BUY")

    if not analysis:
        return {"error": "Could not analyze order book"}

    use_market, market_reason = integration.should_use_market_order(symbol, order_size)
    adjustment, adj_reason = integration.get_entry_adjustment(symbol, 50, order_size)

    return {
        "symbol": symbol,
        "timing": analysis.entry_timing.value,
        "price": analysis.recommended_entry_price,
        "slippage_pct": analysis.estimated_slippage_pct,
        "spread_bps": analysis.spread_bps,
        "imbalance": analysis.imbalance.value,
        "signal": analysis.signal.value,
        "use_market_order": use_market,
        "market_order_reason": market_reason,
        "confidence_adjustment": adjustment,
        "adjustment_reason": adj_reason,
        "notes": analysis.notes,
    }


# =============================================================================
# TEST
# =============================================================================

if __name__ == "__main__":
    print("Testing Order Book Integration...\n")

    # Create integration
    integration = OrderBookIntegration()

    # Sample order book data (VNM)
    symbol = "VNM"
    bids = [
        (75000, 50000, 25),  # Best bid: 75,000 - 50K shares, 25 orders
        (74900, 30000, 15),
        (74800, 20000, 10),
    ]
    asks = [
        (75100, 20000, 12),  # Best ask: 75,100 - 20K shares, 12 orders
        (75200, 25000, 14),
        (75300, 35000, 18),
    ]
    last_price = 75050

    # Update order book
    order_book = integration.update_order_book(
        symbol=symbol, bids=bids, asks=asks, last_price=last_price, last_volume=1000
    )

    print(f"Order Book Summary - {symbol}:")
    print(f"  Best Bid: {order_book.best_bid:,.0f} VND")
    print(f"  Best Ask: {order_book.best_ask:,.0f} VND")
    print(f"  Spread: {order_book.spread:,.0f} VND ({order_book.spread_pct:.3%})")
    print(f"  Mid Price: {order_book.mid_price:,.0f} VND")
    print(f"  Total Bid Volume: {order_book.total_bid_volume:,}")
    print(f"  Total Ask Volume: {order_book.total_ask_volume:,}")

    # Analyze for different order sizes
    print("\nOrder Book Analysis:")
    for order_size in [1000, 10000, 50000]:
        analysis = integration.analyze_order_book(symbol, order_size, "BUY")
        if analysis:
            print(f"\n  Order Size: {order_size:,} shares")
            print(f"    Spread: {analysis.spread_bps:.1f} bps")
            print(f"    Imbalance: {analysis.imbalance.value} ({analysis.imbalance_ratio:.2f}x)")
            print(f"    Signal: {analysis.signal.value}")
            print(f"    Est. Slippage: {analysis.estimated_slippage_pct:.3%}")
            print(f"    Recommended Entry: {analysis.recommended_entry_price:,.0f} VND")
            print(f"    Timing: {analysis.entry_timing.value}")
            print(f"    Confidence: {analysis.confidence:.0%}")

    # Test quick analysis function
    print("\nQuick Analysis Test:")
    result = analyze_entry_with_order_book(
        symbol="FPT",
        order_size=5000,
        bids=[(110000, 30000, 20), (109900, 25000, 15), (109800, 20000, 10)],
        asks=[(110200, 15000, 10), (110300, 20000, 12), (110400, 25000, 15)],
        last_price=110100,
    )
    print(f"  {result}")

    # Test market order decision
    print("\nMarket Order Decision:")
    use_market, reason = integration.should_use_market_order(symbol, 5000)
    print(f"  Use Market Order: {use_market}")
    print(f"  Reason: {reason}")

    # Test entry adjustment
    print("\nEntry Confidence Adjustment:")
    adjustment, adj_reason = integration.get_entry_adjustment(symbol, 50, 5000)
    print(f"  Adjustment: {adjustment:+d} points")
    print(f"  Reason: {adj_reason}")

    # Test optimal entry price
    print("\nOptimal Entry Prices:")
    for aggressiveness in [0.0, 0.5, 1.0]:
        price = integration.get_optimal_entry_price(symbol, 5000, "BUY", aggressiveness)
        print(f"  Aggressiveness {aggressiveness}: {price:,.0f} VND")

    print("\n✅ Order Book Integration test completed!")
