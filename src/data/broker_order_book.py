# -*- coding: utf-8 -*-
"""
Broker Order Book Integration

This module provides real-time order book integration for Vietnam brokers:
- SSI, VNDirect, TCBS, VPS API integration
- Order book depth analysis
- Bid/Ask spread analysis
- Order flow analytics

Usage:
    from src.data.broker_order_book import (
        BrokerOrderBookManager,
        get_order_book_manager,
    )
    
    manager = get_order_book_manager()
    
    # Get order book
    order_book = manager.get_order_book("VNM")
    
    # Analyze spread
    spread_info = manager.analyze_spread("VNM")
"""

import asyncio
import hashlib
import hmac
import json
import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from threading import RLock, Thread
from typing import Any, Callable, Dict, List, Optional, Tuple

import pandas as pd
import requests

try:
    import websocket
except ImportError:
    websocket = None

logger = logging.getLogger(__name__)


# =============================================================================
# ENUMS AND CONSTANTS
# =============================================================================


class BrokerType(Enum):
    """Supported brokers."""

    SSI = "ssi"
    VNDIRECT = "vndirect"
    TCBS = "tcbs"
    VPS = "vps"
    MOCK = "mock"  # For testing


class OrderSide(Enum):
    """Order side."""

    BID = "bid"
    ASK = "ask"


class OrderBookLevel(Enum):
    """Order book depth level."""

    TOP = 1  # Top of book only
    THREE = 3  # 3 levels
    FIVE = 5  # 5 levels (standard)
    TEN = 10  # 10 levels (deep)
    FULL = 100  # Full book


# Broker API configurations
BROKER_CONFIGS = {
    BrokerType.SSI: {
        "name": "SSI Securities",
        "rest_url": "https://iboard-api.ssi.com.vn",
        "ws_url": "wss://ssi.signalr.iboard.ssi.com.vn",
        "auth_url": "https://fc-auth.ssi.com.vn/api/v2/Auth/AccessToken",
        "order_book_endpoint": "/market/orderbook",
        "max_depth": 10,
        "rate_limit": 5,  # requests per second
    },
    BrokerType.VNDIRECT: {
        "name": "VNDirect",
        "rest_url": "https://finfo-api.vndirect.com.vn",
        "ws_url": "wss://stock-api.vndirect.com.vn/websocket",
        "order_book_endpoint": "/v4/stock_prices/order_book",
        "max_depth": 10,
        "rate_limit": 10,
    },
    BrokerType.TCBS: {
        "name": "TCBS",
        "rest_url": "https://apipubaws.tcbs.com.vn",
        "ws_url": "wss://apipubaws.tcbs.com.vn/ws/market/quote",
        "order_book_endpoint": "/stock-insight/v1/stock/order-book",
        "max_depth": 10,
        "rate_limit": 10,
    },
    BrokerType.VPS: {
        "name": "VPS Securities",
        "rest_url": "https://bgapidatafeeds.vps.com.vn",
        "ws_url": "wss://bgapidatafeeds.vps.com.vn/realtime",
        "order_book_endpoint": "/getorderbook",
        "max_depth": 10,
        "rate_limit": 10,
    },
}


# =============================================================================
# DATA CLASSES
# =============================================================================


@dataclass
class OrderBookLevel:
    """Single level in order book."""

    price: float
    volume: int
    order_count: int = 0

    @property
    def value(self) -> float:
        """Total value at this level."""
        return self.price * self.volume


@dataclass
class OrderBook:
    """Complete order book snapshot."""

    symbol: str
    timestamp: datetime

    # Bid/Ask sides
    bids: List[OrderBookLevel] = field(default_factory=list)  # Highest first
    asks: List[OrderBookLevel] = field(default_factory=list)  # Lowest first

    # Reference prices
    reference_price: float = 0.0
    ceiling_price: float = 0.0
    floor_price: float = 0.0

    # Last trade
    last_price: float = 0.0
    last_volume: int = 0

    # Totals
    total_bid_volume: int = 0
    total_ask_volume: int = 0
    total_trading_volume: int = 0

    def __post_init__(self):
        if not self.total_bid_volume:
            self.total_bid_volume = sum(b.volume for b in self.bids)
        if not self.total_ask_volume:
            self.total_ask_volume = sum(a.volume for a in self.asks)

    @property
    def best_bid(self) -> Optional[float]:
        """Best bid price."""
        return self.bids[0].price if self.bids else None

    @property
    def best_ask(self) -> Optional[float]:
        """Best ask price."""
        return self.asks[0].price if self.asks else None

    @property
    def mid_price(self) -> Optional[float]:
        """Mid price between best bid and ask."""
        if self.best_bid and self.best_ask:
            return (self.best_bid + self.best_ask) / 2
        return None

    @property
    def spread(self) -> Optional[float]:
        """Bid-ask spread in absolute terms."""
        if self.best_bid and self.best_ask:
            return self.best_ask - self.best_bid
        return None

    @property
    def spread_pct(self) -> Optional[float]:
        """Bid-ask spread as percentage of mid price."""
        if self.spread and self.mid_price:
            return (self.spread / self.mid_price) * 100
        return None

    @property
    def imbalance(self) -> float:
        """Order book imbalance (-1 to 1). Positive = more buying pressure."""
        total = self.total_bid_volume + self.total_ask_volume
        if total == 0:
            return 0.0
        return (self.total_bid_volume - self.total_ask_volume) / total

    def to_dict(self) -> Dict[str, Any]:
        return {
            "symbol": self.symbol,
            "timestamp": self.timestamp.isoformat(),
            "best_bid": self.best_bid,
            "best_ask": self.best_ask,
            "spread": self.spread,
            "spread_pct": self.spread_pct,
            "imbalance": self.imbalance,
            "total_bid_volume": self.total_bid_volume,
            "total_ask_volume": self.total_ask_volume,
            "bids": [{"price": b.price, "volume": b.volume} for b in self.bids[:5]],
            "asks": [{"price": a.price, "volume": a.volume} for a in self.asks[:5]],
        }


@dataclass
class SpreadAnalysis:
    """Analysis of bid-ask spread."""

    symbol: str
    timestamp: datetime

    # Current spread
    spread: float = 0.0
    spread_pct: float = 0.0
    spread_ticks: int = 0  # Spread in tick sizes

    # Spread statistics
    avg_spread_5m: float = 0.0
    avg_spread_15m: float = 0.0
    avg_spread_today: float = 0.0

    # Spread rating
    is_tight: bool = False  # < 0.1%
    is_normal: bool = False  # 0.1% - 0.3%
    is_wide: bool = False  # > 0.3%

    # Liquidity assessment
    liquidity_score: float = 0.0  # 0-100
    recommended_slippage: float = 0.0  # Recommended slippage for order


@dataclass
class OrderFlowMetrics:
    """Order flow analytics."""

    symbol: str
    timestamp: datetime

    # Volume ratios
    buy_volume: int = 0
    sell_volume: int = 0
    buy_ratio: float = 0.0  # % of total

    # Large orders
    large_buy_orders: int = 0
    large_sell_orders: int = 0
    large_order_threshold: int = 10000  # shares

    # Order book depth
    depth_bid_10k: int = 0  # How many price levels to absorb 10k shares
    depth_ask_10k: int = 0

    # Aggression metrics
    trade_at_ask_pct: float = 0.0  # % of trades at ask (aggressive buying)
    trade_at_bid_pct: float = 0.0  # % of trades at bid (aggressive selling)

    # Signals
    buying_pressure: str = "neutral"  # strong, moderate, neutral, weak
    trend_direction: str = "neutral"  # up, down, neutral


# =============================================================================
# BROKER API CONNECTORS
# =============================================================================


class BrokerAPIConnector(ABC):
    """Abstract base class for broker API connectors."""

    def __init__(self, broker_type: BrokerType, credentials: Optional[Dict] = None):
        self.broker_type = broker_type
        self.credentials = credentials or {}
        self.config = BROKER_CONFIGS.get(broker_type, {})

        self._session = requests.Session()
        self._last_request_time = 0.0
        self._access_token: Optional[str] = None
        self._token_expiry: Optional[datetime] = None

    @abstractmethod
    def get_order_book(self, symbol: str, depth: int = 5) -> Optional[OrderBook]:
        """Get order book for symbol."""
        pass

    @abstractmethod
    def subscribe_order_book(
        self,
        symbols: List[str],
        callback: Callable[[OrderBook], None],
    ) -> bool:
        """Subscribe to order book updates."""
        pass

    @abstractmethod
    def unsubscribe_order_book(self, symbols: List[str]) -> bool:
        """Unsubscribe from order book updates."""
        pass

    def _rate_limit(self):
        """Apply rate limiting."""
        rate_limit = self.config.get("rate_limit", 10)
        min_interval = 1.0 / rate_limit

        elapsed = time.time() - self._last_request_time
        if elapsed < min_interval:
            time.sleep(min_interval - elapsed)

        self._last_request_time = time.time()


class SSIConnector(BrokerAPIConnector):
    """SSI Securities API connector."""

    def __init__(self, credentials: Optional[Dict] = None):
        super().__init__(BrokerType.SSI, credentials)
        self._ws = None
        self._ws_thread = None
        self._callbacks: Dict[str, Callable] = {}

    def _authenticate(self) -> bool:
        """Authenticate with SSI API."""
        if self._access_token and self._token_expiry:
            if datetime.now() < self._token_expiry:
                return True

        # SSI uses public API for market data
        # Real authentication would use consumer_id and consumer_secret
        self._access_token = "public"
        self._token_expiry = datetime.now() + timedelta(hours=1)
        return True

    def get_order_book(self, symbol: str, depth: int = 5) -> Optional[OrderBook]:
        """Get order book from SSI."""
        self._rate_limit()

        try:
            # SSI iBoard public API
            url = f"https://iboard-api.ssi.com.vn/statistics/current"

            response = self._session.get(
                url,
                params={"symbols": symbol},
                headers={"Accept": "application/json"},
                timeout=10,
            )

            if response.status_code != 200:
                logger.warning(f"SSI API error: {response.status_code}")
                return None

            data = response.json()
            if not data or "data" not in data:
                return None

            stock_data = data["data"][0] if data["data"] else None
            if not stock_data:
                return None

            # Parse order book
            bids = []
            asks = []

            # SSI provides 3 levels: bid1-3, ask1-3
            for i in range(1, 4):
                bid_price = stock_data.get(f"bid{i}Price", 0)
                bid_vol = stock_data.get(f"bid{i}Vol", 0)
                if bid_price > 0:
                    bids.append(
                        OrderBookLevel(
                            price=bid_price * 1000,  # Convert to VND
                            volume=bid_vol * 10,  # Convert to shares
                        )
                    )

                ask_price = stock_data.get(f"ask{i}Price", 0)
                ask_vol = stock_data.get(f"ask{i}Vol", 0)
                if ask_price > 0:
                    asks.append(
                        OrderBookLevel(
                            price=ask_price * 1000,
                            volume=ask_vol * 10,
                        )
                    )

            return OrderBook(
                symbol=symbol,
                timestamp=datetime.now(),
                bids=bids,
                asks=asks,
                reference_price=stock_data.get("refPrice", 0) * 1000,
                ceiling_price=stock_data.get("ceiling", 0) * 1000,
                floor_price=stock_data.get("floor", 0) * 1000,
                last_price=stock_data.get("lastPrice", 0) * 1000,
                last_volume=stock_data.get("lastVol", 0) * 10,
                total_trading_volume=stock_data.get("totalVol", 0) * 10,
            )

        except Exception as e:
            logger.warning(f"SSI get_order_book error: {e}")
            return None

    def subscribe_order_book(
        self,
        symbols: List[str],
        callback: Callable[[OrderBook], None],
    ) -> bool:
        """Subscribe to SSI WebSocket order book."""
        if websocket is None:
            logger.warning("websocket-client not installed")
            return False

        for symbol in symbols:
            self._callbacks[symbol] = callback

        # Start WebSocket thread if not running
        if self._ws_thread is None or not self._ws_thread.is_alive():
            self._start_websocket()

        return True

    def unsubscribe_order_book(self, symbols: List[str]) -> bool:
        """Unsubscribe from order book updates."""
        for symbol in symbols:
            self._callbacks.pop(symbol, None)
        return True

    def _start_websocket(self):
        """Start WebSocket connection."""

        def run_ws():
            while self._callbacks:
                try:
                    # Poll mode fallback
                    for symbol in list(self._callbacks.keys()):
                        order_book = self.get_order_book(symbol)
                        if order_book and symbol in self._callbacks:
                            self._callbacks[symbol](order_book)
                    time.sleep(1)
                except Exception as e:
                    logger.warning(f"SSI WebSocket error: {e}")
                    time.sleep(5)

        self._ws_thread = Thread(target=run_ws, daemon=True)
        self._ws_thread.start()


class VNDirectConnector(BrokerAPIConnector):
    """VNDirect API connector."""

    def __init__(self, credentials: Optional[Dict] = None):
        super().__init__(BrokerType.VNDIRECT, credentials)
        self._callbacks: Dict[str, Callable] = {}
        self._poll_thread = None

    def get_order_book(self, symbol: str, depth: int = 5) -> Optional[OrderBook]:
        """Get order book from VNDirect."""
        self._rate_limit()

        try:
            # VNDirect finfo API
            url = f"https://finfo-api.vndirect.com.vn/v4/stock_prices"

            response = self._session.get(
                url,
                params={
                    "q": f"code:{symbol}",
                    "size": 1,
                    "sort": "date",
                },
                timeout=10,
            )

            if response.status_code != 200:
                return None

            data = response.json()
            if not data.get("data"):
                return None

            stock_data = data["data"][0]

            # VNDirect provides limited order book in main API
            # Would need authenticated API for full depth
            bids = []
            asks = []

            # Parse available bid/ask levels
            if stock_data.get("bidPrice1"):
                for i in range(1, 4):
                    bid_p = stock_data.get(f"bidPrice{i}", 0)
                    bid_v = stock_data.get(f"bidVol{i}", 0)
                    if bid_p > 0:
                        bids.append(OrderBookLevel(price=bid_p, volume=bid_v))

                    ask_p = stock_data.get(f"offerPrice{i}", 0)
                    ask_v = stock_data.get(f"offerVol{i}", 0)
                    if ask_p > 0:
                        asks.append(OrderBookLevel(price=ask_p, volume=ask_v))

            return OrderBook(
                symbol=symbol,
                timestamp=datetime.now(),
                bids=bids,
                asks=asks,
                reference_price=stock_data.get("basicPrice", 0),
                ceiling_price=stock_data.get("ceilingPrice", 0),
                floor_price=stock_data.get("floorPrice", 0),
                last_price=stock_data.get("close", 0),
                last_volume=stock_data.get("nmVolume", 0),
            )

        except Exception as e:
            logger.warning(f"VNDirect get_order_book error: {e}")
            return None

    def subscribe_order_book(
        self,
        symbols: List[str],
        callback: Callable[[OrderBook], None],
    ) -> bool:
        """Subscribe to order book updates via polling."""
        for symbol in symbols:
            self._callbacks[symbol] = callback

        if self._poll_thread is None or not self._poll_thread.is_alive():
            self._start_polling()

        return True

    def unsubscribe_order_book(self, symbols: List[str]) -> bool:
        for symbol in symbols:
            self._callbacks.pop(symbol, None)
        return True

    def _start_polling(self):
        def poll():
            while self._callbacks:
                for symbol in list(self._callbacks.keys()):
                    try:
                        order_book = self.get_order_book(symbol)
                        if order_book and symbol in self._callbacks:
                            self._callbacks[symbol](order_book)
                    except Exception as e:
                        logger.debug(f"Poll error: {e}")
                time.sleep(1)

        self._poll_thread = Thread(target=poll, daemon=True)
        self._poll_thread.start()


class TCBSConnector(BrokerAPIConnector):
    """TCBS API connector."""

    def __init__(self, credentials: Optional[Dict] = None):
        super().__init__(BrokerType.TCBS, credentials)
        self._callbacks: Dict[str, Callable] = {}
        self._poll_thread = None

    def get_order_book(self, symbol: str, depth: int = 5) -> Optional[OrderBook]:
        """Get order book from TCBS."""
        self._rate_limit()

        try:
            url = f"https://apipubaws.tcbs.com.vn/stock-insight/v1/intraday/{symbol}/his/paging"

            response = self._session.get(
                url,
                params={"page": 0, "size": 20, "headIndex": -1},
                timeout=10,
            )

            if response.status_code != 200:
                return None

            data = response.json()

            # TCBS provides intraday data, need to construct book from recent trades
            # For actual order book, would need authenticated API

            return OrderBook(
                symbol=symbol,
                timestamp=datetime.now(),
                bids=[],
                asks=[],
            )

        except Exception as e:
            logger.warning(f"TCBS get_order_book error: {e}")
            return None

    def subscribe_order_book(
        self,
        symbols: List[str],
        callback: Callable[[OrderBook], None],
    ) -> bool:
        for symbol in symbols:
            self._callbacks[symbol] = callback
        return True

    def unsubscribe_order_book(self, symbols: List[str]) -> bool:
        for symbol in symbols:
            self._callbacks.pop(symbol, None)
        return True


class MockConnector(BrokerAPIConnector):
    """Mock connector for testing."""

    def __init__(self, credentials: Optional[Dict] = None):
        super().__init__(BrokerType.MOCK, credentials)
        self._mock_books: Dict[str, OrderBook] = {}

    def set_mock_order_book(self, symbol: str, order_book: OrderBook):
        """Set mock order book for testing."""
        self._mock_books[symbol] = order_book

    def get_order_book(self, symbol: str, depth: int = 5) -> Optional[OrderBook]:
        if symbol in self._mock_books:
            return self._mock_books[symbol]

        # Generate mock order book
        import random

        base_price = 50000  # 50k VND

        bids = [
            OrderBookLevel(
                price=base_price - (i * 100),
                volume=random.randint(1000, 10000),
            )
            for i in range(depth)
        ]

        asks = [
            OrderBookLevel(
                price=base_price + 100 + (i * 100),
                volume=random.randint(1000, 10000),
            )
            for i in range(depth)
        ]

        return OrderBook(
            symbol=symbol,
            timestamp=datetime.now(),
            bids=bids,
            asks=asks,
            reference_price=base_price,
            ceiling_price=base_price * 1.07,
            floor_price=base_price * 0.93,
            last_price=base_price,
        )

    def subscribe_order_book(
        self,
        symbols: List[str],
        callback: Callable[[OrderBook], None],
    ) -> bool:
        return True

    def unsubscribe_order_book(self, symbols: List[str]) -> bool:
        return True


# =============================================================================
# ORDER BOOK MANAGER
# =============================================================================


class BrokerOrderBookManager:
    """
    Unified Order Book Manager

    Manages multiple broker connections and provides unified order book access
    with spread analysis and order flow metrics.

    Usage:
        manager = BrokerOrderBookManager()

        # Get order book from any available source
        book = manager.get_order_book("VNM")

        # Analyze spread
        spread_info = manager.analyze_spread("VNM")

        # Get order flow metrics
        flow = manager.get_order_flow_metrics("VNM")
    """

    def __init__(
        self,
        primary_broker: BrokerType = BrokerType.SSI,
        fallback_brokers: Optional[List[BrokerType]] = None,
    ):
        self._lock = RLock()

        # Initialize connectors
        self._connectors: Dict[BrokerType, BrokerAPIConnector] = {}
        self._primary_broker = primary_broker
        self._fallback_brokers = fallback_brokers or [
            BrokerType.VNDIRECT,
            BrokerType.TCBS,
        ]

        self._init_connectors()

        # Cache
        self._order_book_cache: Dict[str, OrderBook] = {}
        self._cache_ttl = timedelta(seconds=2)

        # Spread history for analysis
        self._spread_history: Dict[str, List[Tuple[datetime, float]]] = {}

        logger.info("📖 Broker Order Book Manager initialized")

    def _init_connectors(self):
        """Initialize broker connectors."""
        connector_classes = {
            BrokerType.SSI: SSIConnector,
            BrokerType.VNDIRECT: VNDirectConnector,
            BrokerType.TCBS: TCBSConnector,
            BrokerType.MOCK: MockConnector,
        }

        brokers = [self._primary_broker] + self._fallback_brokers

        for broker in brokers:
            if broker in connector_classes:
                try:
                    self._connectors[broker] = connector_classes[broker]()
                    logger.debug(f"Initialized {broker.value} connector")
                except Exception as e:
                    logger.warning(f"Failed to initialize {broker.value}: {e}")

    def get_order_book(
        self,
        symbol: str,
        depth: int = 5,
        use_cache: bool = True,
    ) -> Optional[OrderBook]:
        """
        Get order book for symbol.

        Args:
            symbol: Stock symbol
            depth: Order book depth (1-10)
            use_cache: Use cached data if fresh

        Returns:
            OrderBook or None
        """
        symbol = symbol.upper()

        # Check cache
        if use_cache and symbol in self._order_book_cache:
            cached = self._order_book_cache[symbol]
            if datetime.now() - cached.timestamp < self._cache_ttl:
                return cached

        # Try connectors in order
        brokers = [self._primary_broker] + self._fallback_brokers

        for broker in brokers:
            if broker not in self._connectors:
                continue

            try:
                order_book = self._connectors[broker].get_order_book(symbol, depth)
                if order_book:
                    self._order_book_cache[symbol] = order_book
                    self._update_spread_history(symbol, order_book)
                    return order_book
            except Exception as e:
                logger.debug(f"{broker.value} failed: {e}")
                continue

        logger.warning(f"No order book available for {symbol}")
        return None

    def get_batch_order_books(
        self,
        symbols: List[str],
        depth: int = 5,
    ) -> Dict[str, OrderBook]:
        """Get order books for multiple symbols."""
        results = {}

        for symbol in symbols:
            order_book = self.get_order_book(symbol, depth)
            if order_book:
                results[symbol] = order_book

        return results

    def analyze_spread(self, symbol: str) -> SpreadAnalysis:
        """
        Analyze bid-ask spread for symbol.

        Returns comprehensive spread analysis including:
        - Current spread
        - Historical averages
        - Liquidity assessment
        """
        order_book = self.get_order_book(symbol)

        analysis = SpreadAnalysis(symbol=symbol, timestamp=datetime.now())

        if not order_book or order_book.spread is None:
            return analysis

        analysis.spread = order_book.spread
        analysis.spread_pct = order_book.spread_pct or 0.0

        # Calculate spread in ticks
        tick_size = self._get_tick_size(order_book.reference_price)
        analysis.spread_ticks = int(analysis.spread / tick_size) if tick_size > 0 else 0

        # Historical spread analysis
        history = self._spread_history.get(symbol, [])

        now = datetime.now()

        # 5-minute average
        spreads_5m = [s for t, s in history if now - t < timedelta(minutes=5)]
        if spreads_5m:
            analysis.avg_spread_5m = sum(spreads_5m) / len(spreads_5m)

        # 15-minute average
        spreads_15m = [s for t, s in history if now - t < timedelta(minutes=15)]
        if spreads_15m:
            analysis.avg_spread_15m = sum(spreads_15m) / len(spreads_15m)

        # Today's average
        today = now.date()
        spreads_today = [s for t, s in history if t.date() == today]
        if spreads_today:
            analysis.avg_spread_today = sum(spreads_today) / len(spreads_today)

        # Classify spread
        if analysis.spread_pct < 0.1:
            analysis.is_tight = True
        elif analysis.spread_pct < 0.3:
            analysis.is_normal = True
        else:
            analysis.is_wide = True

        # Liquidity score (0-100)
        # Based on spread, depth, and volume
        spread_score = max(0, 50 - analysis.spread_pct * 100)
        depth_score = min(50, order_book.total_bid_volume / 100)  # Normalize
        analysis.liquidity_score = min(100, spread_score + depth_score)

        # Recommended slippage
        analysis.recommended_slippage = max(0.1, analysis.spread_pct * 2)

        return analysis

    def get_order_flow_metrics(self, symbol: str) -> OrderFlowMetrics:
        """
        Get order flow analytics for symbol.

        Analyzes buying vs selling pressure and large order activity.
        """
        order_book = self.get_order_book(symbol)

        metrics = OrderFlowMetrics(symbol=symbol, timestamp=datetime.now())

        if not order_book:
            return metrics

        # Volume analysis
        metrics.buy_volume = order_book.total_bid_volume
        metrics.sell_volume = order_book.total_ask_volume

        total = metrics.buy_volume + metrics.sell_volume
        if total > 0:
            metrics.buy_ratio = metrics.buy_volume / total

        # Large orders analysis
        large_threshold = metrics.large_order_threshold

        for bid in order_book.bids:
            if bid.volume >= large_threshold:
                metrics.large_buy_orders += 1

        for ask in order_book.asks:
            if ask.volume >= large_threshold:
                metrics.large_sell_orders += 1

        # Depth analysis - how many levels to absorb 10k shares
        target_volume = 10000

        cumulative = 0
        for i, bid in enumerate(order_book.bids):
            cumulative += bid.volume
            if cumulative >= target_volume:
                metrics.depth_bid_10k = i + 1
                break

        cumulative = 0
        for i, ask in enumerate(order_book.asks):
            cumulative += ask.volume
            if cumulative >= target_volume:
                metrics.depth_ask_10k = i + 1
                break

        # Buying pressure classification
        imbalance = order_book.imbalance

        if imbalance > 0.3:
            metrics.buying_pressure = "strong"
            metrics.trend_direction = "up"
        elif imbalance > 0.1:
            metrics.buying_pressure = "moderate"
            metrics.trend_direction = "up"
        elif imbalance < -0.3:
            metrics.buying_pressure = "weak"
            metrics.trend_direction = "down"
        elif imbalance < -0.1:
            metrics.buying_pressure = "neutral"
            metrics.trend_direction = "down"
        else:
            metrics.buying_pressure = "neutral"
            metrics.trend_direction = "neutral"

        return metrics

    def check_entry_conditions(
        self,
        symbol: str,
        side: str = "buy",  # "buy" or "sell"
    ) -> Dict[str, Any]:
        """
        Check order book conditions for entry.

        Returns dict with:
        - is_favorable: bool
        - confidence_adjustment: float (-0.1 to +0.1)
        - reasons: List[str]
        """
        order_book = self.get_order_book(symbol)
        spread_analysis = self.analyze_spread(symbol)
        flow_metrics = self.get_order_flow_metrics(symbol)

        result = {
            "is_favorable": True,
            "confidence_adjustment": 0.0,
            "reasons": [],
        }

        if not order_book:
            result["is_favorable"] = False
            result["confidence_adjustment"] = -0.1
            result["reasons"].append("Order book unavailable")
            return result

        # Spread check
        if spread_analysis.is_wide:
            result["confidence_adjustment"] -= 0.05
            result["reasons"].append(f"Wide spread: {spread_analysis.spread_pct:.2f}%")
        elif spread_analysis.is_tight:
            result["confidence_adjustment"] += 0.03
            result["reasons"].append("Tight spread")

        # Imbalance check
        imbalance = order_book.imbalance

        if side == "buy":
            if imbalance < -0.3:  # Heavy selling pressure
                result["confidence_adjustment"] -= 0.05
                result["reasons"].append("High selling pressure")
            elif imbalance > 0.3:  # Heavy buying pressure
                result["confidence_adjustment"] += 0.05
                result["reasons"].append("Strong buying pressure")
        else:  # sell
            if imbalance > 0.3:
                result["confidence_adjustment"] -= 0.05
                result["reasons"].append("High buying pressure (bad for short)")
            elif imbalance < -0.3:
                result["confidence_adjustment"] += 0.05
                result["reasons"].append("Strong selling pressure")

        # Liquidity check
        if spread_analysis.liquidity_score < 30:
            result["is_favorable"] = False
            result["confidence_adjustment"] -= 0.1
            result["reasons"].append("Low liquidity")

        # Large orders check
        if side == "buy" and flow_metrics.large_buy_orders > flow_metrics.large_sell_orders:
            result["confidence_adjustment"] += 0.02
            result["reasons"].append("More large buy orders")

        return result

    def subscribe_order_book(
        self,
        symbols: List[str],
        callback: Callable[[OrderBook], None],
    ) -> bool:
        """Subscribe to real-time order book updates."""
        if self._primary_broker in self._connectors:
            return self._connectors[self._primary_broker].subscribe_order_book(symbols, callback)
        return False

    def unsubscribe_order_book(self, symbols: List[str]) -> bool:
        """Unsubscribe from order book updates."""
        if self._primary_broker in self._connectors:
            return self._connectors[self._primary_broker].unsubscribe_order_book(symbols)
        return False

    def _update_spread_history(self, symbol: str, order_book: OrderBook):
        """Update spread history for analysis."""
        if symbol not in self._spread_history:
            self._spread_history[symbol] = []

        if order_book.spread_pct:
            self._spread_history[symbol].append((order_book.timestamp, order_book.spread_pct))

        # Keep only last hour
        cutoff = datetime.now() - timedelta(hours=1)
        self._spread_history[symbol] = [
            (t, s) for t, s in self._spread_history[symbol] if t > cutoff
        ]

    def _get_tick_size(self, price: float) -> float:
        """Get tick size based on price level (Vietnam rules)."""
        if price < 10000:
            return 10
        elif price < 50000:
            return 50
        else:
            return 100

    def get_status(self) -> Dict[str, Any]:
        """Get manager status."""
        return {
            "primary_broker": self._primary_broker.value,
            "active_connectors": [b.value for b in self._connectors.keys()],
            "cached_symbols": list(self._order_book_cache.keys()),
            "spread_history_symbols": list(self._spread_history.keys()),
        }


# =============================================================================
# SINGLETON & CONVENIENCE FUNCTIONS
# =============================================================================

_manager_instance: Optional[BrokerOrderBookManager] = None
_manager_lock = RLock()


def get_order_book_manager(
    primary_broker: BrokerType = BrokerType.SSI,
) -> BrokerOrderBookManager:
    """Get singleton order book manager."""
    global _manager_instance
    with _manager_lock:
        if _manager_instance is None:
            _manager_instance = BrokerOrderBookManager(primary_broker)
        return _manager_instance


def reset_order_book_manager():
    """Reset singleton (for testing)."""
    global _manager_instance
    with _manager_lock:
        _manager_instance = None


def get_order_book(symbol: str) -> Optional[OrderBook]:
    """Convenience function to get order book."""
    return get_order_book_manager().get_order_book(symbol)


def analyze_spread(symbol: str) -> SpreadAnalysis:
    """Convenience function to analyze spread."""
    return get_order_book_manager().analyze_spread(symbol)


def check_entry_liquidity(symbol: str, side: str = "buy") -> Dict[str, Any]:
    """Convenience function to check entry conditions."""
    return get_order_book_manager().check_entry_conditions(symbol, side)


# =============================================================================
# TESTING
# =============================================================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    print("\n" + "=" * 60)
    print("🧪 TESTING BROKER ORDER BOOK MANAGER")
    print("=" * 60)

    manager = get_order_book_manager()

    test_symbols = ["VNM", "VCB", "HPG"]

    print("\n📖 Fetching order books:")
    print("-" * 60)

    for symbol in test_symbols:
        order_book = manager.get_order_book(symbol)

        if order_book:
            print(f"\n{symbol}:")
            print(
                f"  Best Bid: {order_book.best_bid:,.0f}"
                if order_book.best_bid
                else "  Best Bid: N/A"
            )
            print(
                f"  Best Ask: {order_book.best_ask:,.0f}"
                if order_book.best_ask
                else "  Best Ask: N/A"
            )
            print(
                f"  Spread: {order_book.spread_pct:.3f}%"
                if order_book.spread_pct
                else "  Spread: N/A"
            )
            print(f"  Imbalance: {order_book.imbalance:+.2f}")
            print(f"  Bid Volume: {order_book.total_bid_volume:,}")
            print(f"  Ask Volume: {order_book.total_ask_volume:,}")
        else:
            print(f"\n{symbol}: No order book available")

    print("\n📊 Spread Analysis:")
    print("-" * 60)

    for symbol in test_symbols[:1]:
        spread_info = manager.analyze_spread(symbol)
        print(f"\n{symbol}:")
        print(f"  Spread: {spread_info.spread_pct:.3f}%")
        print(f"  Liquidity Score: {spread_info.liquidity_score:.1f}")
        print(
            f"  Classification: {'Tight' if spread_info.is_tight else 'Normal' if spread_info.is_normal else 'Wide'}"
        )
        print(f"  Recommended Slippage: {spread_info.recommended_slippage:.2f}%")

    print("\n📈 Entry Condition Check:")
    print("-" * 60)

    for symbol in test_symbols[:1]:
        conditions = manager.check_entry_conditions(symbol, "buy")
        print(f"\n{symbol}:")
        print(f"  Favorable: {conditions['is_favorable']}")
        print(f"  Confidence Adj: {conditions['confidence_adjustment']:+.3f}")
        print(f"  Reasons: {conditions['reasons']}")

    print("\n" + "=" * 60)
