# -*- coding: utf-8 -*-
"""
Enhanced Order Book Integration with Broker Connectivity

IMPROVEMENT #3.3: Order Book Analysis with real/simulated data support

Features:
- Abstract broker interface for real-time data
- Simulated order book for backtesting/paper trading
- Multiple broker support (SSI, VNDirect, TCBS)
- Intraday pattern detection
- Smart order routing suggestions
- Market impact estimation
- Data staleness handling with confidence adjustment

Vietnam Market Order Book Characteristics:
- 3 best bid/ask levels visible publicly
- Full depth available through broker API
- Order types: LO (Limit), ATO, ATC, MP (Market)
- Minimum order: 100 shares (1 lot)
- Price step: 10/50/100 VND depending on price range

Author: Trading Bot Team
Version: 2.1.0 - Added Data Staleness Handling
"""

import logging
import random
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, time, timedelta
from enum import Enum
from typing import Dict, List, Optional, Tuple, Any, Callable
from threading import Thread, Event, RLock
import json
import os

import numpy as np
import pandas as pd

from src.utils.data_staleness import (
    DataStalenessMixin,
    DataFreshness,
    StalenessConfig,
    STALENESS_CONFIGS,
)

logger = logging.getLogger(__name__)


# =============================================================================
# DATA STRUCTURES
# =============================================================================


@dataclass
class OrderBookLevel:
    """Single price level in order book"""

    price: float
    volume: int
    order_count: int = 1

    @property
    def value(self) -> float:
        return self.price * self.volume


@dataclass
class OrderBookSnapshot:
    """Full order book snapshot"""

    symbol: str
    timestamp: datetime

    # Bid/Ask levels (best first)
    bids: List[OrderBookLevel] = field(default_factory=list)
    asks: List[OrderBookLevel] = field(default_factory=list)

    # Last trade
    last_price: float = 0.0
    last_volume: int = 0

    # Aggregates
    total_bid_volume: int = 0
    total_ask_volume: int = 0

    # Reference price (previous close)
    reference_price: float = 0.0

    # Metadata
    source: str = ""
    is_simulated: bool = False

    def __post_init__(self):
        if not self.total_bid_volume:
            self.total_bid_volume = sum(l.volume for l in self.bids)
        if not self.total_ask_volume:
            self.total_ask_volume = sum(l.volume for l in self.asks)

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
    def imbalance_ratio(self) -> float:
        """Bid/Ask volume ratio. >1 = buying pressure, <1 = selling pressure"""
        if self.total_ask_volume > 0:
            return self.total_bid_volume / self.total_ask_volume
        return 1.0


class OrderBookSignal(Enum):
    """Trading signal from order book analysis"""

    STRONG_BUY = "STRONG_BUY"
    BUY = "BUY"
    NEUTRAL = "NEUTRAL"
    SELL = "SELL"
    STRONG_SELL = "STRONG_SELL"


class EntryRecommendation(Enum):
    """Entry timing recommendation"""

    ENTER_NOW = "ENTER_NOW"
    WAIT_FOR_DIP = "WAIT_FOR_DIP"
    USE_LIMIT_ORDER = "USE_LIMIT_ORDER"
    SCALE_IN = "SCALE_IN"
    AVOID = "AVOID"


@dataclass
class OrderBookAnalysisResult:
    """Result of order book analysis"""

    symbol: str
    timestamp: datetime

    # Order book state
    spread_bps: float
    imbalance_ratio: float

    # Signal
    signal: OrderBookSignal
    confidence: float  # 0-100

    # Entry recommendation
    entry_recommendation: EntryRecommendation
    recommended_price: float
    estimated_slippage_pct: float

    # Detected patterns
    large_orders_detected: bool = False
    institutional_activity: bool = False
    spoofing_detected: bool = False

    # Notes
    notes: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)


# =============================================================================
# ABSTRACT BROKER INTERFACE
# =============================================================================


class BrokerOrderBookProvider(ABC):
    """
    Abstract interface for broker order book data.

    Implement this for each broker (SSI, VNDirect, TCBS, etc.)
    """

    @abstractmethod
    def connect(self) -> bool:
        """Connect to broker API"""
        pass

    @abstractmethod
    def disconnect(self):
        """Disconnect from broker"""
        pass

    @abstractmethod
    def is_connected(self) -> bool:
        """Check connection status"""
        pass

    @abstractmethod
    def get_order_book(self, symbol: str) -> Optional[OrderBookSnapshot]:
        """Get current order book for symbol"""
        pass

    @abstractmethod
    def subscribe(self, symbol: str, callback: Callable[[OrderBookSnapshot], None]):
        """Subscribe to real-time updates"""
        pass

    @abstractmethod
    def unsubscribe(self, symbol: str):
        """Unsubscribe from updates"""
        pass


# =============================================================================
# SIMULATED ORDER BOOK PROVIDER
# =============================================================================


class SimulatedOrderBookProvider(BrokerOrderBookProvider):
    """
    Simulated order book for backtesting and paper trading.

    Generates realistic order book based on:
    - Historical price/volume data
    - Typical Vietnam market patterns
    - Intraday volume profile
    - Volatility state
    """

    # Vietnam market tick sizes
    TICK_SIZES = {(0, 10000): 10, (10000, 50000): 50, (50000, float("inf")): 100}

    # Intraday volume profile (typical Vietnam market)
    VOLUME_PROFILE = {
        (9, 0): 2.5,  # ATO - high activity
        (9, 15): 1.8,  # Post-ATO
        (9, 30): 1.2,
        (10, 0): 1.0,
        (10, 30): 0.9,
        (11, 0): 1.1,  # Pre-lunch
        (11, 15): 0.8,  # Lunch wind-down
        (13, 0): 1.3,  # Afternoon open
        (13, 30): 1.0,
        (14, 0): 1.1,
        (14, 15): 1.5,  # Pre-ATC
        (14, 30): 2.0,  # ATC - high activity
    }

    def __init__(
        self,
        base_spread_bps: float = 20,
        base_depth_shares: int = 50000,
        noise_factor: float = 0.3,
        seed: Optional[int] = None,
    ):
        """
        Initialize simulated provider.

        Args:
            base_spread_bps: Base spread in basis points
            base_depth_shares: Base order book depth in shares
            noise_factor: Random noise factor (0-1)
            seed: Random seed for reproducibility
        """
        self.base_spread_bps = base_spread_bps
        self.base_depth_shares = base_depth_shares
        self.noise_factor = noise_factor

        if seed:
            random.seed(seed)
            np.random.seed(seed)

        self._connected = False
        self._price_cache: Dict[str, float] = {}
        self._volatility_cache: Dict[str, float] = {}
        self._subscriptions: Dict[str, Callable] = {}

    def connect(self) -> bool:
        self._connected = True
        logger.info("📊 Simulated order book provider connected")
        return True

    def disconnect(self):
        self._connected = False
        self._subscriptions.clear()

    def is_connected(self) -> bool:
        return self._connected

    def set_price(self, symbol: str, price: float, volatility: float = 0.02):
        """Set reference price for simulation"""
        self._price_cache[symbol] = price
        self._volatility_cache[symbol] = volatility

    def get_order_book(self, symbol: str) -> Optional[OrderBookSnapshot]:
        """Generate simulated order book"""
        if not self._connected:
            return None

        # Get base price
        base_price = self._price_cache.get(symbol, 50000)  # Default 50K VND
        volatility = self._volatility_cache.get(symbol, 0.02)

        # Get tick size
        tick_size = self._get_tick_size(base_price)

        # Get current time factor
        time_factor = self._get_time_factor()

        # Generate spread
        spread = self._generate_spread(base_price, volatility, time_factor)

        # Calculate mid price with some random walk
        mid_price = base_price * (1 + np.random.normal(0, volatility / 10))
        mid_price = self._round_to_tick(mid_price, tick_size)

        # Generate bid/ask levels
        bids = self._generate_levels(mid_price, spread, tick_size, "BID", time_factor)
        asks = self._generate_levels(mid_price, spread, tick_size, "ASK", time_factor)

        return OrderBookSnapshot(
            symbol=symbol,
            timestamp=datetime.now(),
            bids=bids,
            asks=asks,
            last_price=mid_price,
            last_volume=int(self.base_depth_shares * 0.1 * time_factor),
            reference_price=base_price,
            source="simulated",
            is_simulated=True,
        )

    def subscribe(self, symbol: str, callback: Callable[[OrderBookSnapshot], None]):
        """Subscribe to simulated updates (no real-time for simulation)"""
        self._subscriptions[symbol] = callback

    def unsubscribe(self, symbol: str):
        self._subscriptions.pop(symbol, None)

    def _get_tick_size(self, price: float) -> float:
        """Get tick size for price"""
        for (low, high), tick in self.TICK_SIZES.items():
            if low <= price < high:
                return tick
        return 100

    def _round_to_tick(self, price: float, tick_size: float) -> float:
        """Round price to nearest tick"""
        return round(price / tick_size) * tick_size

    def _get_time_factor(self) -> float:
        """Get volume multiplier based on time of day"""
        now = datetime.now().time()
        hour, minute = now.hour, now.minute

        # Find closest time in profile
        best_match = 1.0
        for (h, m), factor in self.VOLUME_PROFILE.items():
            if h == hour and m <= minute:
                best_match = factor

        # Add some noise
        noise = np.random.normal(1, self.noise_factor * 0.3)
        return best_match * max(0.5, min(2.0, noise))

    def _generate_spread(self, price: float, volatility: float, time_factor: float) -> float:
        """Generate realistic spread"""
        # Higher volatility = wider spread
        vol_factor = 1 + volatility * 10

        # Less liquid times = wider spread
        liquidity_factor = 1 / max(0.5, time_factor)

        # Calculate spread in price terms
        spread_bps = self.base_spread_bps * vol_factor * liquidity_factor
        spread_bps *= 1 + np.random.normal(0, self.noise_factor)

        return price * spread_bps / 10000

    def _generate_levels(
        self, mid_price: float, spread: float, tick_size: float, side: str, time_factor: float
    ) -> List[OrderBookLevel]:
        """Generate order book levels"""
        levels = []
        num_levels = 3  # Vietnam typically shows 3 levels

        for i in range(num_levels):
            # Price
            if side == "BID":
                price = mid_price - spread / 2 - i * tick_size
            else:
                price = mid_price + spread / 2 + i * tick_size

            price = self._round_to_tick(price, tick_size)

            # Volume (decreasing with distance from best)
            volume_factor = 1 / (1 + i * 0.5)
            volume = int(self.base_depth_shares * volume_factor * time_factor)
            volume = max(100, volume)  # Min 1 lot

            # Add noise
            volume = int(volume * (1 + np.random.normal(0, self.noise_factor)))
            volume = max(100, (volume // 100) * 100)  # Round to lots

            # Order count
            order_count = max(1, int(volume / 5000))

            levels.append(OrderBookLevel(price=price, volume=volume, order_count=order_count))

        return levels


# =============================================================================
# SSI BROKER IMPLEMENTATION (STUB)
# =============================================================================


class SSIOrderBookProvider(BrokerOrderBookProvider):
    """
    SSI Broker order book provider.

    Requires SSI iBoard API credentials.
    This is a stub - implement with actual SSI API.
    """

    def __init__(self, api_key: str = "", api_secret: str = ""):
        self.api_key = api_key
        self.api_secret = api_secret
        self._connected = False
        self._session = None

    def connect(self) -> bool:
        """Connect to SSI API"""
        if not self.api_key or not self.api_secret:
            logger.warning("⚠️ SSI API credentials not provided")
            return False

        try:
            # TODO: Implement actual SSI API connection
            # from ssi_api import SSIClient
            # self._session = SSIClient(self.api_key, self.api_secret)
            # self._session.connect()

            logger.info("📊 SSI order book provider connected")
            self._connected = True
            return True

        except Exception as e:
            logger.error(f"SSI connection failed: {e}")
            return False

    def disconnect(self):
        self._connected = False
        self._session = None

    def is_connected(self) -> bool:
        return self._connected

    def get_order_book(self, symbol: str) -> Optional[OrderBookSnapshot]:
        """Get order book from SSI"""
        if not self._connected:
            return None

        try:
            # TODO: Implement actual SSI API call
            # data = self._session.get_order_book(symbol)
            # return self._parse_ssi_order_book(data)
            return None

        except Exception as e:
            logger.error(f"SSI order book fetch failed: {e}")
            return None

    def subscribe(self, symbol: str, callback: Callable[[OrderBookSnapshot], None]):
        """Subscribe to SSI real-time updates"""
        # TODO: Implement SSI WebSocket subscription
        pass

    def unsubscribe(self, symbol: str):
        pass


# =============================================================================
# VNDIRECT BROKER IMPLEMENTATION (STUB)
# =============================================================================


class VNDirectOrderBookProvider(BrokerOrderBookProvider):
    """
    VNDirect broker order book provider.

    Requires VNDirect API access.
    This is a stub - implement with actual VNDirect API.
    """

    def __init__(self, username: str = "", password: str = ""):
        self.username = username
        self.password = password
        self._connected = False

    def connect(self) -> bool:
        logger.warning("⚠️ VNDirect provider not implemented - use simulated")
        return False

    def disconnect(self):
        self._connected = False

    def is_connected(self) -> bool:
        return self._connected

    def get_order_book(self, symbol: str) -> Optional[OrderBookSnapshot]:
        return None

    def subscribe(self, symbol: str, callback: Callable[[OrderBookSnapshot], None]):
        pass

    def unsubscribe(self, symbol: str):
        pass


# =============================================================================
# ENHANCED ORDER BOOK ANALYZER
# =============================================================================


class EnhancedOrderBookAnalyzer(DataStalenessMixin):
    """
    Enhanced order book analysis with pattern detection.

    Features:
    - Spread analysis
    - Imbalance detection
    - Large order detection
    - Institutional activity detection
    - Spoofing detection
    - Entry timing optimization
    - Smart order routing suggestions
    - Data staleness handling (confidence reduction for stale data)

    Staleness handling:
    - Fresh (<1 min): 100% confidence
    - Slightly stale (1-2 min): 85% confidence
    - Stale (2-5 min): 50% confidence
    - Very stale (5-10 min): 20% confidence
    - Expired (>10 min): Do not use for trading decisions
    """

    # Thresholds
    TIGHT_SPREAD_BPS = 10
    WIDE_SPREAD_BPS = 50
    HEAVY_IMBALANCE_RATIO = 2.0
    LARGE_ORDER_PCT = 0.05  # 5% of ADV
    INSTITUTIONAL_ORDER_PCT = 0.10  # 10% of ADV

    def __init__(self):
        # Initialize staleness tracking with orderbook config (strict)
        self._init_staleness("orderbook")

        # Historical data for pattern detection
        self._imbalance_history: Dict[str, List[Tuple[datetime, float]]] = {}
        self._spread_history: Dict[str, List[Tuple[datetime, float]]] = {}

        # Per-symbol cache timestamps
        self._orderbook_timestamps: Dict[str, datetime] = {}

        # ADV cache
        self._adv_cache: Dict[str, float] = {}

    def set_adv(self, symbol: str, adv: float):
        """Set average daily volume for analysis"""
        self._adv_cache[symbol] = adv

    def analyze(
        self, order_book: OrderBookSnapshot, order_size: int = 1000, side: str = "BUY"
    ) -> OrderBookAnalysisResult:
        """
        Comprehensive order book analysis.

        Args:
            order_book: Order book snapshot
            order_size: Intended order size
            side: "BUY" or "SELL"

        Returns:
            OrderBookAnalysisResult
        """
        symbol = order_book.symbol
        notes = []
        warnings = []

        # 1. Spread analysis
        spread_bps = order_book.spread_pct * 10000

        if spread_bps < self.TIGHT_SPREAD_BPS:
            notes.append(f"Tight spread ({spread_bps:.1f} bps) - good liquidity")
        elif spread_bps > self.WIDE_SPREAD_BPS:
            warnings.append(f"Wide spread ({spread_bps:.1f} bps) - use limit order")

        # 2. Imbalance analysis
        imbalance = order_book.imbalance_ratio
        signal = self._imbalance_to_signal(imbalance)

        if imbalance > self.HEAVY_IMBALANCE_RATIO:
            notes.append(f"Heavy buying pressure ({imbalance:.1f}x)")
        elif imbalance < 1 / self.HEAVY_IMBALANCE_RATIO:
            notes.append(f"Heavy selling pressure ({1/imbalance:.1f}x)")

        # 3. Large order detection
        adv = self._adv_cache.get(symbol, 1_000_000)
        large_orders = self._detect_large_orders(order_book, adv)
        institutional = self._detect_institutional_activity(order_book, adv)

        if institutional:
            notes.append("🏛️ Possible institutional activity detected")
        elif large_orders:
            notes.append("📊 Large orders detected")

        # 4. Spoofing detection
        spoofing = self._detect_spoofing(order_book, symbol)
        if spoofing:
            warnings.append("⚠️ Possible spoofing detected - be cautious")

        # 5. Slippage estimation
        slippage = self._estimate_slippage(order_book, order_size, side)

        if slippage > 0.005:  # > 0.5%
            warnings.append(f"High slippage risk ({slippage*100:.2f}%)")

        # 6. Entry recommendation
        recommendation, price = self._get_entry_recommendation(
            order_book, spread_bps, imbalance, slippage, side
        )

        # 7. Confidence
        confidence = self._calculate_confidence(spread_bps, imbalance, large_orders, slippage)

        # Update history
        self._update_history(symbol, order_book)

        return OrderBookAnalysisResult(
            symbol=symbol,
            timestamp=datetime.now(),
            spread_bps=spread_bps,
            imbalance_ratio=imbalance,
            signal=signal,
            confidence=confidence,
            entry_recommendation=recommendation,
            recommended_price=price,
            estimated_slippage_pct=slippage,
            large_orders_detected=large_orders,
            institutional_activity=institutional,
            spoofing_detected=spoofing,
            notes=notes,
            warnings=warnings,
        )

    def _imbalance_to_signal(self, imbalance: float) -> OrderBookSignal:
        """Convert imbalance ratio to signal"""
        if imbalance >= 2.5:
            return OrderBookSignal.STRONG_BUY
        elif imbalance >= 1.5:
            return OrderBookSignal.BUY
        elif imbalance <= 0.4:
            return OrderBookSignal.STRONG_SELL
        elif imbalance <= 0.67:
            return OrderBookSignal.SELL
        else:
            return OrderBookSignal.NEUTRAL

    def _detect_large_orders(self, order_book: OrderBookSnapshot, adv: float) -> bool:
        """Detect large orders in book"""
        threshold = adv * self.LARGE_ORDER_PCT

        for level in order_book.bids + order_book.asks:
            if level.volume >= threshold:
                return True
        return False

    def _detect_institutional_activity(self, order_book: OrderBookSnapshot, adv: float) -> bool:
        """Detect institutional activity patterns"""
        threshold = adv * self.INSTITUTIONAL_ORDER_PCT

        # Single large order
        for level in order_book.bids + order_book.asks:
            if level.volume >= threshold:
                return True

        # Concentrated volume at one price (many orders same price)
        for level in order_book.bids + order_book.asks:
            if level.order_count >= 10 and level.volume >= adv * 0.03:
                return True

        return False

    def _detect_spoofing(self, order_book: OrderBookSnapshot, symbol: str) -> bool:
        """
        Detect potential spoofing.

        Spoofing indicators:
        - Large orders that appear/disappear quickly
        - Extreme imbalance that reverses
        """
        history = self._imbalance_history.get(symbol, [])
        if len(history) < 10:
            return False

        # Check for rapid imbalance reversals
        recent = history[-10:]
        imbalances = [i[1] for i in recent]

        # High variance in imbalance = possible manipulation
        if np.std(imbalances) > 0.8:
            # Check for reversals
            reversals = sum(
                1
                for i in range(1, len(imbalances))
                if (imbalances[i] > 1.5 and imbalances[i - 1] < 0.67)
                or (imbalances[i] < 0.67 and imbalances[i - 1] > 1.5)
            )
            if reversals >= 3:
                return True

        return False

    def _estimate_slippage(
        self, order_book: OrderBookSnapshot, order_size: int, side: str
    ) -> float:
        """Estimate slippage for order"""
        if side == "BUY":
            levels = order_book.asks
            reference = order_book.best_ask or order_book.last_price
        else:
            levels = order_book.bids
            reference = order_book.best_bid or order_book.last_price

        if not levels or reference <= 0:
            return 0.01  # Default 1%

        remaining = order_size
        total_cost = 0

        for level in levels:
            if remaining <= 0:
                break
            fill = min(remaining, level.volume)
            total_cost += fill * level.price
            remaining -= fill

        # Unfilled portion
        if remaining > 0:
            worst_price = levels[-1].price if levels else reference
            if side == "BUY":
                penalty_price = worst_price * 1.01
            else:
                penalty_price = worst_price * 0.99
            total_cost += remaining * penalty_price

        avg_price = total_cost / order_size
        slippage = abs(avg_price - reference) / reference

        return slippage

    def _get_entry_recommendation(
        self,
        order_book: OrderBookSnapshot,
        spread_bps: float,
        imbalance: float,
        slippage: float,
        side: str,
    ) -> Tuple[EntryRecommendation, float]:
        """Get entry timing and price recommendation"""

        if side == "BUY":
            # Wide spread - limit order
            if spread_bps > 50:
                return EntryRecommendation.USE_LIMIT_ORDER, order_book.mid_price

            # Very wide - avoid
            if spread_bps > 100:
                return EntryRecommendation.AVOID, order_book.best_bid or order_book.last_price

            # Heavy selling pressure - wait for dip
            if imbalance < 0.5:
                return (
                    EntryRecommendation.WAIT_FOR_DIP,
                    order_book.best_bid or order_book.last_price,
                )

            # High slippage - scale in
            if slippage > 0.005:
                return EntryRecommendation.SCALE_IN, order_book.mid_price

            # Good conditions
            return EntryRecommendation.ENTER_NOW, order_book.best_ask or order_book.last_price

        else:  # SELL
            if spread_bps > 50:
                return EntryRecommendation.USE_LIMIT_ORDER, order_book.mid_price

            if imbalance > 2:
                return (
                    EntryRecommendation.WAIT_FOR_DIP,
                    order_book.best_ask or order_book.last_price,
                )

            return EntryRecommendation.ENTER_NOW, order_book.best_bid or order_book.last_price

    def _calculate_confidence(
        self, spread_bps: float, imbalance: float, large_orders: bool, slippage: float
    ) -> float:
        """Calculate analysis confidence"""
        confidence = 50.0

        # Tight spread = higher confidence
        if spread_bps < 20:
            confidence += 15
        elif spread_bps < 50:
            confidence += 5
        else:
            confidence -= 10

        # Clear imbalance = higher confidence
        if imbalance > 2 or imbalance < 0.5:
            confidence += 10

        # Large orders = more information
        if large_orders:
            confidence += 5

        # Low slippage = higher confidence
        if slippage < 0.003:
            confidence += 10
        elif slippage > 0.01:
            confidence -= 10

        return max(0, min(100, confidence))

    def _update_history(self, symbol: str, order_book: OrderBookSnapshot):
        """Update historical data"""
        timestamp = order_book.timestamp

        # Update per-symbol timestamp for staleness tracking
        self._orderbook_timestamps[symbol] = timestamp

        # Imbalance history
        if symbol not in self._imbalance_history:
            self._imbalance_history[symbol] = []
        self._imbalance_history[symbol].append((timestamp, order_book.imbalance_ratio))

        if len(self._imbalance_history[symbol]) > 100:
            self._imbalance_history[symbol] = self._imbalance_history[symbol][-100:]

        # Spread history
        if symbol not in self._spread_history:
            self._spread_history[symbol] = []
        self._spread_history[symbol].append((timestamp, order_book.spread_pct * 10000))

        if len(self._spread_history[symbol]) > 100:
            self._spread_history[symbol] = self._spread_history[symbol][-100:]

    # =========================================================================
    # STALENESS-ADJUSTED METHODS
    # =========================================================================

    def is_orderbook_stale(self, symbol: str, max_delay_seconds: int = 60) -> bool:
        """
        Check if order book data for a symbol is stale.

        Args:
            symbol: Stock symbol
            max_delay_seconds: Threshold in seconds (default 60)

        Returns:
            True if data is older than threshold
        """
        if symbol not in self._orderbook_timestamps:
            return True

        age = (datetime.now() - self._orderbook_timestamps[symbol]).total_seconds()
        return age > max_delay_seconds

    def get_orderbook_age_seconds(self, symbol: str) -> float:
        """Get age of order book data in seconds."""
        if symbol not in self._orderbook_timestamps:
            return float("inf")

        return (datetime.now() - self._orderbook_timestamps[symbol]).total_seconds()

    def get_orderbook_freshness(self, symbol: str) -> DataFreshness:
        """
        Get freshness level of order book data.

        Args:
            symbol: Stock symbol

        Returns:
            DataFreshness level
        """
        age_seconds = self.get_orderbook_age_seconds(symbol)
        age_minutes = age_seconds / 60

        config = self._staleness_config
        if age_minutes < config.fresh_threshold_minutes:
            return DataFreshness.FRESH
        elif age_minutes < config.slightly_stale_minutes:
            return DataFreshness.SLIGHTLY_STALE
        elif age_minutes < config.stale_threshold_minutes:
            return DataFreshness.STALE
        elif age_minutes < config.very_stale_threshold_minutes:
            return DataFreshness.VERY_STALE
        else:
            return DataFreshness.EXPIRED

    def get_adjusted_confidence(
        self,
        raw_confidence: float,
        symbol: str,
    ) -> float:
        """
        Get staleness-adjusted confidence score.

        Automatically reduces confidence when order book data is stale:
        - Fresh (<1 min): 100% of raw confidence
        - Slightly stale (1-2 min): 85% of raw confidence
        - Stale (2-5 min): 50% of raw confidence
        - Very stale (5-10 min): 20% of raw confidence
        - Expired (>10 min): 0% confidence

        Args:
            raw_confidence: Original confidence score (0-100)
            symbol: Stock symbol

        Returns:
            Adjusted confidence score
        """
        freshness = self.get_orderbook_freshness(symbol)
        weight = self._staleness_config.get_weight_for_freshness(freshness)

        adjusted = raw_confidence * weight

        # Log warning if stale
        if freshness in (DataFreshness.STALE, DataFreshness.VERY_STALE):
            logger.warning(
                f"⚠️ Order book data stale for {symbol}: "
                f"age={self.get_orderbook_age_seconds(symbol):.1f}s, "
                f"confidence reduced {raw_confidence:.1f}→{adjusted:.1f}"
            )

        return adjusted

    def get_data_quality_status(self, symbol: Optional[str] = None) -> Dict[str, Any]:
        """
        Get data quality status for order book analysis.

        Args:
            symbol: Specific symbol or None for overall status

        Returns:
            Dict with freshness, age, and trading readiness
        """
        if symbol:
            freshness = self.get_orderbook_freshness(symbol)
            weight = self._staleness_config.get_weight_for_freshness(freshness)

            return {
                "symbol": symbol,
                "freshness": freshness.value,
                "age_seconds": self.get_orderbook_age_seconds(symbol),
                "confidence_factor": weight,
                "is_stale": self.is_orderbook_stale(symbol),
                "ready_for_trading": freshness
                in (DataFreshness.FRESH, DataFreshness.SLIGHTLY_STALE),
            }
        else:
            # Overall status
            total = len(self._orderbook_timestamps)
            stale_count = sum(1 for s in self._orderbook_timestamps if self.is_orderbook_stale(s))

            return {
                "tracked_symbols": total,
                "fresh_orderbooks": total - stale_count,
                "stale_orderbooks": stale_count,
                "staleness_config": {
                    "fresh_threshold_min": self._staleness_config.fresh_threshold_minutes,
                    "stale_threshold_min": self._staleness_config.stale_threshold_minutes,
                },
            }


# =============================================================================
# MAIN INTEGRATION CLASS
# =============================================================================


class EnhancedOrderBookIntegration:
    """
    Main order book integration class.

    Features:
    - Multiple broker support with automatic failover
    - Simulated mode for backtesting
    - Real-time analysis
    - Entry timing optimization

    Usage:
        # Paper trading with simulation
        ob = EnhancedOrderBookIntegration(use_simulated=True)

        # Real trading with broker
        ob = EnhancedOrderBookIntegration(
            broker="ssi",
            api_key="xxx",
            api_secret="xxx"
        )

        # Analyze
        analysis = ob.analyze_for_entry("VNM", order_size=1000)
    """

    def __init__(
        self, use_simulated: bool = True, broker: Optional[str] = None, **broker_credentials
    ):
        """
        Initialize order book integration.

        Args:
            use_simulated: Use simulated order book
            broker: Broker name ("ssi", "vndirect")
            broker_credentials: Broker API credentials
        """
        self.analyzer = EnhancedOrderBookAnalyzer()
        self._provider: Optional[BrokerOrderBookProvider] = None

        # Try to connect to real broker first
        if broker and not use_simulated:
            self._provider = self._create_broker_provider(broker, broker_credentials)
            if self._provider and self._provider.connect():
                logger.info(f"✅ Connected to {broker} broker")
            else:
                logger.warning(f"⚠️ Failed to connect to {broker}, using simulation")
                self._provider = None

        # Fallback to simulation
        if self._provider is None:
            self._provider = SimulatedOrderBookProvider()
            self._provider.connect()
            logger.info("📊 Using simulated order book")

        # Cache for price data
        self._price_cache: Dict[str, float] = {}
        self._volume_cache: Dict[str, float] = {}

    def _create_broker_provider(
        self, broker: str, credentials: Dict
    ) -> Optional[BrokerOrderBookProvider]:
        """Create broker provider"""
        broker = broker.lower()

        if broker == "ssi":
            return SSIOrderBookProvider(
                api_key=credentials.get("api_key", ""), api_secret=credentials.get("api_secret", "")
            )
        elif broker == "vndirect":
            return VNDirectOrderBookProvider(
                username=credentials.get("username", ""), password=credentials.get("password", "")
            )
        else:
            logger.warning(f"Unknown broker: {broker}")
            return None

    def set_price_data(
        self, symbol: str, price: float, adv: float = 1_000_000, volatility: float = 0.02
    ):
        """
        Set price data for analysis.

        For simulated mode, this sets the reference price.
        For real mode, this sets ADV for large order detection.
        """
        self._price_cache[symbol] = price
        self._volume_cache[symbol] = adv

        # Set for simulated provider
        if isinstance(self._provider, SimulatedOrderBookProvider):
            self._provider.set_price(symbol, price, volatility)

        # Set ADV for analyzer
        self.analyzer.set_adv(symbol, adv)

    def get_order_book(self, symbol: str) -> Optional[OrderBookSnapshot]:
        """Get current order book"""
        if self._provider:
            return self._provider.get_order_book(symbol)
        return None

    def analyze_for_entry(
        self,
        symbol: str,
        order_size: int = 1000,
        side: str = "BUY",
        df: Optional[pd.DataFrame] = None,
    ) -> Optional[OrderBookAnalysisResult]:
        """
        Analyze order book for entry timing.

        Args:
            symbol: Stock symbol
            order_size: Intended order size
            side: "BUY" or "SELL"
            df: Optional price DataFrame for context

        Returns:
            OrderBookAnalysisResult or None
        """
        # Set price from DataFrame if provided
        if df is not None and not df.empty:
            price = df["close"].iloc[-1]
            adv = df["volume"].tail(20).mean() if "volume" in df else 1_000_000
            volatility = df["close"].pct_change().std() if len(df) > 5 else 0.02
            self.set_price_data(symbol, price, adv, volatility)

        # Get order book
        order_book = self.get_order_book(symbol)
        if not order_book:
            return None

        # Analyze
        return self.analyzer.analyze(order_book, order_size, side)

    def check_entry_conditions(
        self, symbol: str, df: Optional[pd.DataFrame] = None
    ) -> Tuple[bool, int, str]:
        """
        Check order book conditions for entry filter.

        Compatible with entry filter pipeline.

        Returns:
            Tuple of (should_proceed, confidence_adjustment, message)
        """
        try:
            analysis = self.analyze_for_entry(symbol, df=df)

            if analysis is None:
                return (True, 0, "Order book data unavailable")

            # Check for AVOID recommendation
            if analysis.entry_recommendation == EntryRecommendation.AVOID:
                return (False, -15, f"🚫 Order book: {'; '.join(analysis.warnings)}")

            # Calculate adjustment based on signal
            adjustment = 0
            if analysis.signal == OrderBookSignal.STRONG_BUY:
                adjustment = 10
            elif analysis.signal == OrderBookSignal.BUY:
                adjustment = 5
            elif analysis.signal == OrderBookSignal.SELL:
                adjustment = -5
            elif analysis.signal == OrderBookSignal.STRONG_SELL:
                adjustment = -10

            # Adjust for slippage
            if analysis.estimated_slippage_pct > 0.01:
                adjustment -= 5

            # Build message
            msg_parts = analysis.notes + analysis.warnings
            message = "; ".join(msg_parts) if msg_parts else f"Order book: {analysis.signal.value}"

            return (True, adjustment, message)

        except Exception as e:
            logger.debug(f"Order book check failed for {symbol}: {e}")
            return (True, 0, "Order book analysis failed")

    def get_smart_order_recommendation(
        self, symbol: str, total_shares: int, side: str = "BUY", urgency: str = "NORMAL"
    ) -> Dict:
        """
        Get smart order execution recommendation.

        Args:
            symbol: Stock symbol
            total_shares: Total shares to trade
            side: "BUY" or "SELL"
            urgency: "LOW", "NORMAL", "HIGH"

        Returns:
            Dict with execution strategy recommendation
        """
        analysis = self.analyze_for_entry(symbol, total_shares, side)

        if analysis is None:
            return {
                "strategy": "MARKET",
                "slices": [{"shares": total_shares, "price": None}],
                "reason": "No order book data",
            }

        # Determine strategy based on analysis
        if analysis.entry_recommendation == EntryRecommendation.USE_LIMIT_ORDER:
            return {
                "strategy": "LIMIT",
                "slices": [{"shares": total_shares, "price": analysis.recommended_price}],
                "reason": "Wide spread - use limit order",
            }

        elif analysis.entry_recommendation == EntryRecommendation.SCALE_IN:
            # Split into 3 slices
            slice_size = (total_shares // 300) * 100  # Round to lots
            remainder = total_shares - slice_size * 3

            return {
                "strategy": "TWAP",
                "slices": [
                    {"shares": slice_size, "price": analysis.recommended_price},
                    {"shares": slice_size, "price": None, "delay_minutes": 5},
                    {"shares": slice_size + remainder, "price": None, "delay_minutes": 10},
                ],
                "reason": "High slippage - scale in gradually",
            }

        else:
            return {
                "strategy": "MARKET",
                "slices": [{"shares": total_shares, "price": None}],
                "reason": "Good liquidity - market order OK",
            }


# =============================================================================
# SINGLETON
# =============================================================================


_integration_instance: Optional[EnhancedOrderBookIntegration] = None
_lock = RLock()


def get_order_book_integration(
    use_simulated: bool = True, **kwargs
) -> EnhancedOrderBookIntegration:
    """Get or create singleton order book integration"""
    global _integration_instance
    with _lock:
        if _integration_instance is None:
            _integration_instance = EnhancedOrderBookIntegration(
                use_simulated=use_simulated, **kwargs
            )
        return _integration_instance


def reset_order_book_integration():
    """Reset integration instance"""
    global _integration_instance
    with _lock:
        _integration_instance = None


# =============================================================================
# TESTING
# =============================================================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    print("\n" + "=" * 60)
    print("🧪 TESTING ENHANCED ORDER BOOK INTEGRATION")
    print("=" * 60)

    # Test with simulated provider
    integration = EnhancedOrderBookIntegration(use_simulated=True)

    # Test symbols
    test_symbols = [
        ("VNM", 100000, 500000),  # symbol, price, adv
        ("VCB", 85000, 800000),
        ("HPG", 25000, 2000000),
    ]

    for symbol, price, adv in test_symbols:
        print(f"\n📊 Testing {symbol} (price: {price:,}, ADV: {adv:,})...")

        # Set price data
        integration.set_price_data(symbol, price, adv)

        # Get order book
        ob = integration.get_order_book(symbol)
        if ob:
            print(f"   Best Bid: {ob.best_bid:,.0f}")
            print(f"   Best Ask: {ob.best_ask:,.0f}")
            print(f"   Spread: {ob.spread_pct*10000:.1f} bps")
            print(f"   Imbalance: {ob.imbalance_ratio:.2f}x")

        # Analyze for entry
        analysis = integration.analyze_for_entry(symbol, order_size=1000)
        if analysis:
            print(f"   Signal: {analysis.signal.value}")
            print(f"   Recommendation: {analysis.entry_recommendation.value}")
            print(f"   Est. Slippage: {analysis.estimated_slippage_pct*100:.2f}%")
            print(f"   Confidence: {analysis.confidence:.0f}%")

        # Check entry conditions
        can_enter, adj, msg = integration.check_entry_conditions(symbol)
        print(f"   Can Enter: {can_enter}")
        print(f"   Adjustment: {adj:+d}")
        print(f"   Message: {msg}")

    print("\n✅ Test complete!")
