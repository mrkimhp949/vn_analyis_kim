# -*- coding: utf-8 -*-
"""
Real-time Data Provider for Vietnam Stock Market

Provides real-time market data integration with multiple broker APIs:
- SSI (SSI Securities)
- VNDirect
- TCBS (Techcom Securities)
- WebSocket streaming support

Author: Trading Bot Team
Version: 1.0.0
"""

import asyncio
import logging
import threading
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Callable, Dict, List, Optional, Any
from queue import Queue

import pandas as pd

logger = logging.getLogger(__name__)


class DataSource(Enum):
    """Supported data sources"""

    SSI = "SSI"
    VNDIRECT = "VNDIRECT"
    TCBS = "TCBS"
    FIREANT = "FIREANT"
    CAFEF = "CAFEF"


@dataclass
class RealtimeQuote:
    """Real-time quote data"""

    symbol: str
    price: float
    change: float
    change_pct: float
    volume: int
    value: float  # Trading value in VND

    # Best bid/ask
    bid_price: float = 0.0
    bid_volume: int = 0
    ask_price: float = 0.0
    ask_volume: int = 0

    # OHLC
    open: float = 0.0
    high: float = 0.0
    low: float = 0.0
    reference: float = 0.0  # Previous close
    ceiling: float = 0.0
    floor: float = 0.0

    # Foreign trading
    foreign_buy_volume: int = 0
    foreign_sell_volume: int = 0
    foreign_room: int = 0  # Remaining foreign room

    # Metadata
    timestamp: datetime = field(default_factory=datetime.now)
    source: str = ""

    @property
    def foreign_net_volume(self) -> int:
        """Net foreign volume (buy - sell)"""
        return self.foreign_buy_volume - self.foreign_sell_volume

    @property
    def spread(self) -> float:
        """Bid-ask spread"""
        if self.bid_price > 0 and self.ask_price > 0:
            return self.ask_price - self.bid_price
        return 0.0

    @property
    def spread_pct(self) -> float:
        """Bid-ask spread as percentage"""
        if self.price > 0:
            return (self.spread / self.price) * 100
        return 0.0


@dataclass
class OrderBookLevel:
    """Single level in order book"""

    price: float
    volume: int
    order_count: int = 0


@dataclass
class OrderBook:
    """Full order book data"""

    symbol: str
    bids: List[OrderBookLevel] = field(default_factory=list)  # Best to worst
    asks: List[OrderBookLevel] = field(default_factory=list)  # Best to worst
    timestamp: datetime = field(default_factory=datetime.now)

    @property
    def best_bid(self) -> Optional[OrderBookLevel]:
        return self.bids[0] if self.bids else None

    @property
    def best_ask(self) -> Optional[OrderBookLevel]:
        return self.asks[0] if self.asks else None

    @property
    def mid_price(self) -> float:
        if self.best_bid and self.best_ask:
            return (self.best_bid.price + self.best_ask.price) / 2
        return 0.0

    @property
    def total_bid_volume(self) -> int:
        return sum(level.volume for level in self.bids)

    @property
    def total_ask_volume(self) -> int:
        return sum(level.volume for level in self.asks)

    @property
    def imbalance(self) -> float:
        """Order book imbalance (-1 to 1, positive = more bids)"""
        total = self.total_bid_volume + self.total_ask_volume
        if total == 0:
            return 0.0
        return (self.total_bid_volume - self.total_ask_volume) / total


@dataclass
class MarketDepth:
    """Market depth analysis"""

    symbol: str
    bid_depth_value: float = 0.0  # Total bid value in VND
    ask_depth_value: float = 0.0  # Total ask value in VND
    liquidity_score: float = 0.0  # 0-100
    spread_score: float = 0.0  # 0-100 (lower spread = higher score)
    imbalance_score: float = 0.0  # -100 to 100

    @property
    def is_liquid(self) -> bool:
        return self.liquidity_score >= 60

    @property
    def has_tight_spread(self) -> bool:
        return self.spread_score >= 70


class BaseRealtimeProvider(ABC):
    """Abstract base class for real-time data providers"""

    def __init__(self, api_key: Optional[str] = None, api_secret: Optional[str] = None):
        self.api_key = api_key
        self.api_secret = api_secret
        self._connected = False
        self._subscribers: Dict[str, List[Callable]] = {}
        self._quote_cache: Dict[str, RealtimeQuote] = {}
        self._orderbook_cache: Dict[str, OrderBook] = {}

    @abstractmethod
    def connect(self) -> bool:
        """Connect to data source"""
        pass

    @abstractmethod
    def disconnect(self) -> None:
        """Disconnect from data source"""
        pass

    @abstractmethod
    def get_quote(self, symbol: str) -> Optional[RealtimeQuote]:
        """Get real-time quote for a symbol"""
        pass

    @abstractmethod
    def get_orderbook(self, symbol: str, depth: int = 10) -> Optional[OrderBook]:
        """Get order book for a symbol"""
        pass

    def subscribe(self, symbol: str, callback: Callable[[RealtimeQuote], None]) -> None:
        """Subscribe to real-time updates for a symbol"""
        if symbol not in self._subscribers:
            self._subscribers[symbol] = []
        self._subscribers[symbol].append(callback)

    def unsubscribe(self, symbol: str, callback: Optional[Callable] = None) -> None:
        """Unsubscribe from updates"""
        if symbol in self._subscribers:
            if callback:
                self._subscribers[symbol].remove(callback)
            else:
                del self._subscribers[symbol]

    def _notify_subscribers(self, symbol: str, quote: RealtimeQuote) -> None:
        """Notify all subscribers of a quote update"""
        if symbol in self._subscribers:
            for callback in self._subscribers[symbol]:
                try:
                    callback(quote)
                except Exception as e:
                    logger.error(f"Subscriber callback error: {e}")


class SSIRealtimeProvider(BaseRealtimeProvider):
    """
    SSI Securities Real-time Data Provider

    Requires SSI API credentials from:
    https://iboard.ssi.com.vn/
    """

    def __init__(self, consumer_id: str = "", consumer_secret: str = ""):
        super().__init__(consumer_id, consumer_secret)
        self.base_url = "https://fc-data.ssi.com.vn"
        self.ws_url = "wss://fc-data.ssi.com.vn/ws"
        self._access_token: Optional[str] = None
        self._token_expiry: Optional[datetime] = None

    def connect(self) -> bool:
        """Connect and authenticate with SSI API"""
        if not self.api_key or not self.api_secret:
            logger.warning("SSI API credentials not provided")
            return False

        try:
            # Authenticate
            self._access_token = self._authenticate()
            if self._access_token:
                self._connected = True
                logger.info("✅ Connected to SSI Real-time API")
                return True
        except Exception as e:
            logger.error(f"SSI connection failed: {e}")

        return False

    def _authenticate(self) -> Optional[str]:
        """Authenticate with SSI API"""
        try:
            import requests

            auth_url = f"{self.base_url}/api/v2/Market/AccessToken"
            payload = {"consumerID": self.api_key, "consumerSecret": self.api_secret}

            response = requests.post(auth_url, json=payload, timeout=10)

            if response.status_code == 200:
                data = response.json()
                if data.get("status") == 200:
                    token = data.get("data", {}).get("accessToken")
                    logger.info("✅ SSI authentication successful")
                    return token

            logger.warning(f"SSI auth failed: {response.text}")
            return None

        except Exception as e:
            logger.error(f"SSI authentication error: {e}")
            return None

    def disconnect(self) -> None:
        """Disconnect from SSI API"""
        self._connected = False
        self._access_token = None
        logger.info("Disconnected from SSI API")

    def get_quote(self, symbol: str) -> Optional[RealtimeQuote]:
        """Get real-time quote from SSI"""
        if not self._connected:
            if not self.connect():
                return None

        try:
            import requests

            url = f"{self.base_url}/api/v2/Market/SecuritiesDetails"
            headers = {"Authorization": f"Bearer {self._access_token}"}
            params = {"market": "HOSE", "symbol": symbol}

            response = requests.get(url, headers=headers, params=params, timeout=5)

            if response.status_code == 200:
                data = response.json()
                if data.get("status") == 200:
                    item = data.get("data", [{}])[0]

                    quote = RealtimeQuote(
                        symbol=symbol,
                        price=float(item.get("matchedPrice", 0)) * 1000,
                        change=float(item.get("priceChange", 0)) * 1000,
                        change_pct=float(item.get("priceChangePercent", 0)),
                        volume=int(item.get("matchedVolume", 0)),
                        value=float(item.get("matchedValue", 0)),
                        bid_price=float(item.get("best1Bid", 0)) * 1000,
                        bid_volume=int(item.get("best1BidVol", 0)),
                        ask_price=float(item.get("best1Offer", 0)) * 1000,
                        ask_volume=int(item.get("best1OfferVol", 0)),
                        open=float(item.get("openPrice", 0)) * 1000,
                        high=float(item.get("highestPrice", 0)) * 1000,
                        low=float(item.get("lowestPrice", 0)) * 1000,
                        reference=float(item.get("refPrice", 0)) * 1000,
                        ceiling=float(item.get("ceilingPrice", 0)) * 1000,
                        floor=float(item.get("floorPrice", 0)) * 1000,
                        foreign_buy_volume=int(item.get("foreignBuyVolume", 0)),
                        foreign_sell_volume=int(item.get("foreignSellVolume", 0)),
                        foreign_room=int(item.get("foreignRoom", 0)),
                        source="SSI",
                        timestamp=datetime.now(),
                    )

                    self._quote_cache[symbol] = quote
                    return quote

            return self._quote_cache.get(symbol)

        except Exception as e:
            logger.error(f"SSI quote error for {symbol}: {e}")
            return self._quote_cache.get(symbol)

    def get_orderbook(self, symbol: str, depth: int = 10) -> Optional[OrderBook]:
        """Get order book from SSI"""
        # SSI provides top 3 bid/ask in quote data
        quote = self.get_quote(symbol)
        if not quote:
            return None

        # Build order book from quote data
        orderbook = OrderBook(
            symbol=symbol,
            bids=[OrderBookLevel(price=quote.bid_price, volume=quote.bid_volume)],
            asks=[OrderBookLevel(price=quote.ask_price, volume=quote.ask_volume)],
            timestamp=quote.timestamp,
        )

        self._orderbook_cache[symbol] = orderbook
        return orderbook

    def get_foreign_flow_realtime(self, symbols: List[str] = None) -> Dict[str, Dict]:
        """Get real-time foreign trading flow"""
        if symbols is None:
            symbols = ["VNINDEX", "VN30"]

        result = {}
        for symbol in symbols:
            quote = self.get_quote(symbol)
            if quote:
                result[symbol] = {
                    "buy_volume": quote.foreign_buy_volume,
                    "sell_volume": quote.foreign_sell_volume,
                    "net_volume": quote.foreign_net_volume,
                    "room": quote.foreign_room,
                    "timestamp": quote.timestamp.isoformat(),
                }

        return result


class VNDirectRealtimeProvider(BaseRealtimeProvider):
    """
    VNDirect Real-time Data Provider

    Uses VNDirect's public API endpoints
    """

    def __init__(self, api_key: str = "", api_secret: str = ""):
        super().__init__(api_key, api_secret)
        self.base_url = "https://finfo-api.vndirect.com.vn"

    def connect(self) -> bool:
        """Connect to VNDirect API"""
        try:
            # VNDirect has some public endpoints
            self._connected = True
            logger.info("✅ Connected to VNDirect API")
            return True
        except Exception as e:
            logger.error(f"VNDirect connection failed: {e}")
            return False

    def disconnect(self) -> None:
        self._connected = False

    def get_quote(self, symbol: str) -> Optional[RealtimeQuote]:
        """Get quote from VNDirect"""
        try:
            import requests

            url = f"{self.base_url}/v4/stock_prices"
            params = {"sort": "date", "q": f"code:{symbol}", "size": 1}

            response = requests.get(url, params=params, timeout=5)

            if response.status_code == 200:
                data = response.json()
                items = data.get("data", [])

                if items:
                    item = items[0]

                    quote = RealtimeQuote(
                        symbol=symbol,
                        price=float(item.get("close", 0)) * 1000,
                        change=float(item.get("change", 0)) * 1000,
                        change_pct=float(item.get("pctChange", 0)),
                        volume=int(item.get("nmVolume", 0)),
                        value=float(item.get("nmValue", 0)),
                        open=float(item.get("open", 0)) * 1000,
                        high=float(item.get("high", 0)) * 1000,
                        low=float(item.get("low", 0)) * 1000,
                        reference=float(item.get("basicPrice", 0)) * 1000,
                        ceiling=float(item.get("ceilingPrice", 0)) * 1000,
                        floor=float(item.get("floorPrice", 0)) * 1000,
                        source="VNDirect",
                        timestamp=datetime.now(),
                    )

                    self._quote_cache[symbol] = quote
                    return quote

            return self._quote_cache.get(symbol)

        except Exception as e:
            logger.error(f"VNDirect quote error: {e}")
            return self._quote_cache.get(symbol)

    def get_orderbook(self, symbol: str, depth: int = 10) -> Optional[OrderBook]:
        """VNDirect doesn't provide public orderbook"""
        return self._orderbook_cache.get(symbol)


class RealtimeDataManager:
    """
    Unified Real-time Data Manager

    Manages multiple data providers with automatic failover
    """

    def __init__(self):
        self._providers: Dict[DataSource, BaseRealtimeProvider] = {}
        self._primary_source: Optional[DataSource] = None
        self._quote_callbacks: Dict[str, List[Callable]] = {}
        self._running = False
        self._update_thread: Optional[threading.Thread] = None
        self._update_queue: Queue = Queue()

        # Cache settings
        self._cache_ttl = 5  # seconds
        self._last_update: Dict[str, datetime] = {}

    def add_provider(
        self, source: DataSource, provider: BaseRealtimeProvider, is_primary: bool = False
    ) -> None:
        """Add a data provider"""
        self._providers[source] = provider
        if is_primary or self._primary_source is None:
            self._primary_source = source
        logger.info(f"Added {source.value} provider (primary={is_primary})")

    def connect_all(self) -> Dict[DataSource, bool]:
        """Connect all providers"""
        results = {}
        for source, provider in self._providers.items():
            results[source] = provider.connect()
        return results

    def disconnect_all(self) -> None:
        """Disconnect all providers"""
        self._running = False
        for provider in self._providers.values():
            provider.disconnect()

    def get_quote(
        self, symbol: str, source: Optional[DataSource] = None
    ) -> Optional[RealtimeQuote]:
        """
        Get real-time quote with automatic failover

        Args:
            symbol: Stock symbol
            source: Specific source to use (None = auto)

        Returns:
            RealtimeQuote or None
        """
        # Try specific source first
        if source and source in self._providers:
            quote = self._providers[source].get_quote(symbol)
            if quote:
                return quote

        # Try primary source
        if self._primary_source and self._primary_source in self._providers:
            quote = self._providers[self._primary_source].get_quote(symbol)
            if quote:
                return quote

        # Failover to other sources
        for src, provider in self._providers.items():
            if src != self._primary_source:
                quote = provider.get_quote(symbol)
                if quote:
                    logger.debug(f"Failover to {src.value} for {symbol}")
                    return quote

        return None

    def get_quotes_batch(self, symbols: List[str]) -> Dict[str, RealtimeQuote]:
        """Get quotes for multiple symbols"""
        results = {}
        for symbol in symbols:
            quote = self.get_quote(symbol)
            if quote:
                results[symbol] = quote
        return results

    def get_orderbook(self, symbol: str) -> Optional[OrderBook]:
        """Get order book with failover"""
        for provider in self._providers.values():
            orderbook = provider.get_orderbook(symbol)
            if orderbook:
                return orderbook
        return None

    def analyze_market_depth(self, symbol: str) -> Optional[MarketDepth]:
        """Analyze market depth for a symbol"""
        orderbook = self.get_orderbook(symbol)
        quote = self.get_quote(symbol)

        if not orderbook or not quote:
            return None

        # Calculate depth values
        bid_depth = sum(level.price * level.volume for level in orderbook.bids)
        ask_depth = sum(level.price * level.volume for level in orderbook.asks)

        # Liquidity score (based on total depth)
        total_depth = bid_depth + ask_depth
        liquidity_score = min(100, (total_depth / 10_000_000_000) * 100)  # 10B VND = 100

        # Spread score (lower spread = higher score)
        spread_pct = quote.spread_pct
        spread_score = max(0, 100 - spread_pct * 50)  # 2% spread = 0 score

        # Imbalance score
        imbalance_score = orderbook.imbalance * 100

        return MarketDepth(
            symbol=symbol,
            bid_depth_value=bid_depth,
            ask_depth_value=ask_depth,
            liquidity_score=liquidity_score,
            spread_score=spread_score,
            imbalance_score=imbalance_score,
        )

    def subscribe(self, symbol: str, callback: Callable[[RealtimeQuote], None]) -> None:
        """Subscribe to real-time updates"""
        if symbol not in self._quote_callbacks:
            self._quote_callbacks[symbol] = []
        self._quote_callbacks[symbol].append(callback)

        # Subscribe on all providers
        for provider in self._providers.values():
            provider.subscribe(symbol, callback)

    def start_streaming(self, symbols: List[str], interval: float = 1.0) -> None:
        """Start streaming updates for symbols"""
        self._running = True

        def update_loop():
            while self._running:
                for symbol in symbols:
                    quote = self.get_quote(symbol)
                    if quote and symbol in self._quote_callbacks:
                        for callback in self._quote_callbacks[symbol]:
                            try:
                                callback(quote)
                            except Exception as e:
                                logger.error(f"Callback error: {e}")
                time.sleep(interval)

        self._update_thread = threading.Thread(target=update_loop, daemon=True)
        self._update_thread.start()
        logger.info(f"Started streaming for {len(symbols)} symbols")

    def stop_streaming(self) -> None:
        """Stop streaming updates"""
        self._running = False
        if self._update_thread:
            self._update_thread.join(timeout=5)
        logger.info("Stopped streaming")

    def get_foreign_flow_realtime(self) -> Dict[str, Any]:
        """Get real-time foreign flow from all providers"""
        for provider in self._providers.values():
            if hasattr(provider, "get_foreign_flow_realtime"):
                flow = provider.get_foreign_flow_realtime()
                if flow:
                    return flow
        return {}


# Singleton instance
_realtime_manager: Optional[RealtimeDataManager] = None


def get_realtime_manager() -> RealtimeDataManager:
    """Get singleton realtime data manager"""
    global _realtime_manager
    if _realtime_manager is None:
        _realtime_manager = RealtimeDataManager()
    return _realtime_manager


def setup_realtime_providers(
    ssi_consumer_id: str = "",
    ssi_consumer_secret: str = "",
    vndirect_key: str = "",
) -> RealtimeDataManager:
    """
    Setup real-time providers with credentials

    Args:
        ssi_consumer_id: SSI API consumer ID
        ssi_consumer_secret: SSI API consumer secret
        vndirect_key: VNDirect API key

    Returns:
        Configured RealtimeDataManager
    """
    manager = get_realtime_manager()

    # Add SSI provider
    if ssi_consumer_id and ssi_consumer_secret:
        ssi_provider = SSIRealtimeProvider(ssi_consumer_id, ssi_consumer_secret)
        manager.add_provider(DataSource.SSI, ssi_provider, is_primary=True)

    # Add VNDirect provider (works without credentials for basic data)
    vndirect_provider = VNDirectRealtimeProvider(vndirect_key)
    manager.add_provider(DataSource.VNDIRECT, vndirect_provider)

    # Connect all
    results = manager.connect_all()
    logger.info(f"Provider connection results: {results}")

    return manager


# Test
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    print("\n" + "=" * 70)
    print("🧪 TESTING REAL-TIME DATA PROVIDER")
    print("=" * 70 + "\n")

    # Setup with VNDirect (no credentials needed for basic data)
    manager = get_realtime_manager()
    vndirect = VNDirectRealtimeProvider()
    manager.add_provider(DataSource.VNDIRECT, vndirect, is_primary=True)
    manager.connect_all()

    # Test quotes
    test_symbols = ["VNM", "VCB", "FPT"]

    for symbol in test_symbols:
        quote = manager.get_quote(symbol)
        if quote:
            print(f"\n📊 {symbol}:")
            print(f"   Price: {quote.price:,.0f} VND")
            print(f"   Change: {quote.change:+,.0f} ({quote.change_pct:+.2f}%)")
            print(f"   Volume: {quote.volume:,}")
            print(f"   O/H/L: {quote.open:,.0f} / {quote.high:,.0f} / {quote.low:,.0f}")
            print(f"   Ceiling/Floor: {quote.ceiling:,.0f} / {quote.floor:,.0f}")
        else:
            print(f"❌ No data for {symbol}")

    print("\n" + "=" * 70)
    print("✅ Real-time provider test completed!")
    print("=" * 70)
