# -*- coding: utf-8 -*-
"""
Real-time Data Provider for Vietnam Stock Market

Provides real-time market data integration with multiple broker APIs:
- SSI (SSI Securities)
- VNDirect
- TCBS (Techcom Securities)
- WebSocket streaming support

Features:
- Automatic retry with exponential backoff
- Circuit breaker pattern for API failures
- Multi-source failover
- Connection health monitoring
- Data staleness handling with quality scoring

Author: Trading Bot Team
Version: 2.1.0 - Added Data Staleness Handling
"""

import asyncio
import logging
import threading
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Callable, Dict, List, Optional, Any, Tuple
from queue import Queue
from functools import wraps

import pandas as pd

from src.utils.data_staleness import (
    DataStalenessMixin,
    DataFreshness,
    StalenessConfig,
    STALENESS_CONFIGS,
)

logger = logging.getLogger(__name__)


# =============================================================================
# ERROR HANDLING & RETRY UTILITIES
# =============================================================================


@dataclass
class RetryConfig:
    """Configuration for retry behavior"""

    max_retries: int = 3
    base_delay: float = 1.0  # seconds
    max_delay: float = 30.0  # seconds
    exponential_base: float = 2.0
    jitter: bool = True  # Add random jitter to prevent thundering herd


@dataclass
class CircuitBreakerConfig:
    """Configuration for circuit breaker"""

    failure_threshold: int = 5  # Failures before opening circuit
    recovery_timeout: float = 60.0  # Seconds before trying again
    half_open_requests: int = 3  # Requests to try in half-open state


class CircuitState(Enum):
    """Circuit breaker states"""

    CLOSED = "CLOSED"  # Normal operation
    OPEN = "OPEN"  # Failing, reject requests
    HALF_OPEN = "HALF_OPEN"  # Testing if recovered


class CircuitBreaker:
    """
    Circuit breaker pattern for API calls

    Prevents cascading failures by stopping requests to failing services
    """

    def __init__(self, name: str, config: Optional[CircuitBreakerConfig] = None):
        self.name = name
        self.config = config or CircuitBreakerConfig()
        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._success_count = 0
        self._last_failure_time: Optional[datetime] = None
        self._half_open_successes = 0
        self._lock = threading.Lock()

    @property
    def state(self) -> CircuitState:
        with self._lock:
            if self._state == CircuitState.OPEN:
                # Check if recovery timeout has passed
                if self._last_failure_time:
                    elapsed = (datetime.now() - self._last_failure_time).total_seconds()
                    if elapsed >= self.config.recovery_timeout:
                        self._state = CircuitState.HALF_OPEN
                        self._half_open_successes = 0
                        logger.info(f"Circuit {self.name}: OPEN -> HALF_OPEN")
            return self._state

    def can_execute(self) -> bool:
        """Check if request can be executed"""
        return self.state != CircuitState.OPEN

    def record_success(self):
        """Record a successful request"""
        with self._lock:
            self._failure_count = 0
            self._success_count += 1

            if self._state == CircuitState.HALF_OPEN:
                self._half_open_successes += 1
                if self._half_open_successes >= self.config.half_open_requests:
                    self._state = CircuitState.CLOSED
                    logger.info(f"Circuit {self.name}: HALF_OPEN -> CLOSED (recovered)")

    def record_failure(self):
        """Record a failed request"""
        with self._lock:
            self._failure_count += 1
            self._last_failure_time = datetime.now()

            if self._state == CircuitState.HALF_OPEN:
                # Immediately open on failure in half-open state
                self._state = CircuitState.OPEN
                logger.warning(f"Circuit {self.name}: HALF_OPEN -> OPEN (still failing)")
            elif self._failure_count >= self.config.failure_threshold:
                self._state = CircuitState.OPEN
                logger.warning(
                    f"Circuit {self.name}: CLOSED -> OPEN " f"(failures: {self._failure_count})"
                )

    def get_stats(self) -> Dict:
        """Get circuit breaker statistics"""
        return {
            "name": self.name,
            "state": self.state.value,
            "failure_count": self._failure_count,
            "success_count": self._success_count,
            "last_failure": (
                self._last_failure_time.isoformat() if self._last_failure_time else None
            ),
        }


def retry_with_backoff(
    config: Optional[RetryConfig] = None,
    exceptions: Tuple = (Exception,),
    on_retry: Optional[Callable] = None,
):
    """
    Decorator for retry with exponential backoff

    Args:
        config: Retry configuration
        exceptions: Tuple of exceptions to catch
        on_retry: Callback function on retry (attempt, exception, delay)
    """
    config = config or RetryConfig()

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            import random

            last_exception = None

            for attempt in range(config.max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e

                    if attempt == config.max_retries:
                        break

                    # Calculate delay with exponential backoff
                    delay = min(
                        config.base_delay * (config.exponential_base**attempt), config.max_delay
                    )

                    # Add jitter
                    if config.jitter:
                        delay = delay * (0.5 + random.random())

                    if on_retry:
                        on_retry(attempt + 1, e, delay)
                    else:
                        logger.warning(
                            f"Retry {attempt + 1}/{config.max_retries} " f"after {delay:.1f}s: {e}"
                        )

                    time.sleep(delay)

            raise last_exception

        return wrapper

    return decorator


class ConnectionHealthMonitor:
    """Monitor connection health across providers"""

    def __init__(self):
        self._health: Dict[str, Dict] = {}
        self._lock = threading.Lock()

    def record_request(self, provider: str, success: bool, latency_ms: float):
        """Record a request result"""
        with self._lock:
            if provider not in self._health:
                self._health[provider] = {
                    "total_requests": 0,
                    "successful_requests": 0,
                    "failed_requests": 0,
                    "total_latency_ms": 0,
                    "last_success": None,
                    "last_failure": None,
                }

            stats = self._health[provider]
            stats["total_requests"] += 1
            stats["total_latency_ms"] += latency_ms

            if success:
                stats["successful_requests"] += 1
                stats["last_success"] = datetime.now()
            else:
                stats["failed_requests"] += 1
                stats["last_failure"] = datetime.now()

    def get_health(self, provider: str) -> Dict:
        """Get health statistics for a provider"""
        with self._lock:
            if provider not in self._health:
                return {"status": "unknown"}

            stats = self._health[provider]
            total = stats["total_requests"]

            if total == 0:
                return {"status": "no_data"}

            success_rate = stats["successful_requests"] / total
            avg_latency = stats["total_latency_ms"] / total

            # Determine status
            if success_rate >= 0.95:
                status = "healthy"
            elif success_rate >= 0.80:
                status = "degraded"
            else:
                status = "unhealthy"

            return {
                "status": status,
                "success_rate": success_rate,
                "avg_latency_ms": avg_latency,
                "total_requests": total,
                "last_success": stats["last_success"],
                "last_failure": stats["last_failure"],
            }

    def get_all_health(self) -> Dict[str, Dict]:
        """Get health for all providers"""
        return {p: self.get_health(p) for p in self._health.keys()}


# Global health monitor
_health_monitor = ConnectionHealthMonitor()


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


class BaseRealtimeProvider(DataStalenessMixin, ABC):
    """Abstract base class for real-time data providers with enhanced error handling

    Features data staleness handling:
    - Per-symbol quote freshness tracking
    - Stale data (>1 min) reduces weight by 15%
    - Very stale data (>3 min) reduces weight by 50%
    - Expired data (>5 min) not used for trading decisions
    """

    def __init__(self, api_key: Optional[str] = None, api_secret: Optional[str] = None):
        self.api_key = api_key
        self.api_secret = api_secret
        self._connected = False
        self._subscribers: Dict[str, List[Callable]] = {}
        self._quote_cache: Dict[str, RealtimeQuote] = {}
        self._orderbook_cache: Dict[str, OrderBook] = {}

        # Initialize staleness tracking with realtime config (strict)
        self._init_staleness("realtime")

        # Error handling
        self._retry_config = RetryConfig()
        self._circuit_breaker = CircuitBreaker(self.__class__.__name__)
        self._last_error: Optional[str] = None
        self._error_count = 0
        self._consecutive_failures = 0

        # Cache settings - per-symbol timestamps
        self._cache_ttl_seconds = 5
        self._cache_timestamps: Dict[str, datetime] = {}

    @property
    def provider_name(self) -> str:
        """Provider name for logging"""
        return self.__class__.__name__

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

    def _is_cache_valid(self, symbol: str) -> bool:
        """Check if cached data is still valid"""
        if symbol not in self._cache_timestamps:
            return False
        elapsed = (datetime.now() - self._cache_timestamps[symbol]).total_seconds()
        return elapsed < self._cache_ttl_seconds

    def _update_cache(self, symbol: str, quote: RealtimeQuote):
        """Update cache with new quote"""
        self._quote_cache[symbol] = quote
        self._cache_timestamps[symbol] = datetime.now()

    def _record_success(self):
        """Record successful request"""
        self._consecutive_failures = 0
        self._circuit_breaker.record_success()

    def _record_failure(self, error: str):
        """Record failed request"""
        self._consecutive_failures += 1
        self._error_count += 1
        self._last_error = error
        self._circuit_breaker.record_failure()

    def get_health_status(self) -> Dict:
        """Get provider health status"""
        return {
            "provider": self.provider_name,
            "connected": self._connected,
            "circuit_state": self._circuit_breaker.state.value,
            "consecutive_failures": self._consecutive_failures,
            "total_errors": self._error_count,
            "last_error": self._last_error,
            "cache_size": len(self._quote_cache),
        }

    def is_quote_stale(self, symbol: str, max_delay_seconds: int = 60) -> bool:
        """
        Check if quote data for a symbol is stale.

        Args:
            symbol: Stock symbol
            max_delay_seconds: Threshold in seconds (default 60)

        Returns:
            True if data is older than threshold
        """
        if symbol not in self._cache_timestamps:
            return True

        age = (datetime.now() - self._cache_timestamps[symbol]).total_seconds()
        return age > max_delay_seconds

    def get_quote_age_seconds(self, symbol: str) -> float:
        """Get age of cached quote in seconds."""
        if symbol not in self._cache_timestamps:
            return float("inf")

        return (datetime.now() - self._cache_timestamps[symbol]).total_seconds()

    def get_quote_freshness(self, symbol: str) -> DataFreshness:
        """
        Get freshness level of quote data.

        Args:
            symbol: Stock symbol

        Returns:
            DataFreshness level (FRESH, SLIGHTLY_STALE, STALE, VERY_STALE, EXPIRED)
        """
        age_seconds = self.get_quote_age_seconds(symbol)
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

    def get_quote_quality_weight(self, symbol: str) -> float:
        """
        Get quality weight for quote data based on freshness.

        Args:
            symbol: Stock symbol

        Returns:
            Weight factor 0.0-1.0
        """
        freshness = self.get_quote_freshness(symbol)
        return self._staleness_config.get_weight_for_freshness(freshness)

    def get_data_quality_status(self, symbol: Optional[str] = None) -> Dict[str, Any]:
        """
        Get data quality status for quote cache.

        Args:
            symbol: Specific symbol or None for overall status

        Returns:
            Dict with freshness, age, weight for decision making
        """
        if symbol:
            return {
                "symbol": symbol,
                "freshness": self.get_quote_freshness(symbol).value,
                "age_seconds": self.get_quote_age_seconds(symbol),
                "weight_factor": self.get_quote_quality_weight(symbol),
                "is_stale": self.is_quote_stale(symbol),
                "has_data": symbol in self._quote_cache,
            }
        else:
            # Overall status
            stale_count = sum(1 for s in self._cache_timestamps if self.is_quote_stale(s))
            fresh_count = len(self._cache_timestamps) - stale_count
            return {
                "cached_symbols": len(self._quote_cache),
                "fresh_quotes": fresh_count,
                "stale_quotes": stale_count,
                "connected": self._connected,
                "provider": self.provider_name,
            }


class SSIRealtimeProvider(BaseRealtimeProvider):
    """
    SSI Securities Real-time Data Provider with enhanced error handling

    Requires SSI API credentials from:
    https://iboard.ssi.com.vn/

    Features:
    - Automatic retry with exponential backoff
    - Circuit breaker for API failures
    - Token refresh handling
    - Request caching
    """

    def __init__(self, consumer_id: str = "", consumer_secret: str = ""):
        super().__init__(consumer_id, consumer_secret)
        self.base_url = "https://fc-data.ssi.com.vn"
        self.ws_url = "wss://fc-data.ssi.com.vn/ws"
        self._access_token: Optional[str] = None
        self._token_expiry: Optional[datetime] = None
        self._token_refresh_margin = 300  # Refresh 5 min before expiry

        # Request session with connection pooling
        self._session: Optional[Any] = None

    def _get_session(self):
        """Get or create requests session"""
        if self._session is None:
            import requests

            self._session = requests.Session()
            # Configure connection pooling
            adapter = requests.adapters.HTTPAdapter(
                pool_connections=10, pool_maxsize=10, max_retries=0  # We handle retries ourselves
            )
            self._session.mount("https://", adapter)
        return self._session

    def connect(self) -> bool:
        """Connect and authenticate with SSI API"""
        if not self.api_key or not self.api_secret:
            logger.warning("SSI API credentials not provided")
            return False

        # Check circuit breaker
        if not self._circuit_breaker.can_execute():
            logger.warning(f"SSI circuit breaker is OPEN, skipping connection")
            return False

        try:
            self._access_token = self._authenticate_with_retry()
            if self._access_token:
                self._connected = True
                self._record_success()
                logger.info("✅ Connected to SSI Real-time API")
                return True
        except Exception as e:
            self._record_failure(str(e))
            logger.error(f"SSI connection failed: {e}")

        return False

    def _authenticate_with_retry(self) -> Optional[str]:
        """Authenticate with retry logic"""
        import requests

        last_error = None

        for attempt in range(self._retry_config.max_retries + 1):
            try:
                start_time = time.time()

                auth_url = f"{self.base_url}/api/v2/Market/AccessToken"
                payload = {"consumerID": self.api_key, "consumerSecret": self.api_secret}

                response = self._get_session().post(auth_url, json=payload, timeout=10)

                latency = (time.time() - start_time) * 1000
                _health_monitor.record_request("SSI", response.status_code == 200, latency)

                if response.status_code == 200:
                    data = response.json()
                    if data.get("status") == 200:
                        token = data.get("data", {}).get("accessToken")
                        # Set token expiry (SSI tokens typically last 1 hour)
                        self._token_expiry = datetime.now() + timedelta(hours=1)
                        logger.info("✅ SSI authentication successful")
                        return token

                last_error = f"Auth failed: {response.status_code} - {response.text[:100]}"

            except requests.exceptions.Timeout:
                last_error = "Request timeout"
            except requests.exceptions.ConnectionError as e:
                last_error = f"Connection error: {e}"
            except Exception as e:
                last_error = str(e)

            if attempt < self._retry_config.max_retries:
                delay = self._retry_config.base_delay * (2**attempt)
                logger.warning(
                    f"SSI auth retry {attempt + 1}/{self._retry_config.max_retries} after {delay:.1f}s: {last_error}"
                )
                time.sleep(delay)

        logger.error(f"SSI authentication failed after retries: {last_error}")
        return None

    def _ensure_token_valid(self) -> bool:
        """Ensure access token is valid, refresh if needed"""
        if not self._access_token:
            return self.connect()

        # Check if token needs refresh
        if self._token_expiry:
            time_until_expiry = (self._token_expiry - datetime.now()).total_seconds()
            if time_until_expiry < self._token_refresh_margin:
                logger.info("SSI token expiring soon, refreshing...")
                return self.connect()

        return True

    def disconnect(self) -> None:
        """Disconnect from SSI API"""
        self._connected = False
        self._access_token = None
        if self._session:
            self._session.close()
            self._session = None
        logger.info("Disconnected from SSI API")

    def get_quote(self, symbol: str) -> Optional[RealtimeQuote]:
        """Get real-time quote from SSI with error handling"""
        # Check cache first
        if self._is_cache_valid(symbol):
            return self._quote_cache.get(symbol)

        # Check circuit breaker
        if not self._circuit_breaker.can_execute():
            logger.debug(f"SSI circuit open, returning cached data for {symbol}")
            return self._quote_cache.get(symbol)

        if not self._ensure_token_valid():
            return self._quote_cache.get(symbol)

        try:
            start_time = time.time()

            url = f"{self.base_url}/api/v2/Market/SecuritiesDetails"
            headers = {"Authorization": f"Bearer {self._access_token}"}
            params = {"market": "HOSE", "symbol": symbol}

            response = self._get_session().get(url, headers=headers, params=params, timeout=5)

            latency = (time.time() - start_time) * 1000
            _health_monitor.record_request("SSI", response.status_code == 200, latency)

            if response.status_code == 200:
                data = response.json()
                if data.get("status") == 200:
                    items = data.get("data", [])
                    if items:
                        item = items[0]

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

                        self._update_cache(symbol, quote)
                        self._record_success()
                        return quote

            elif response.status_code == 401:
                # Token expired, reconnect
                logger.info("SSI token expired, reconnecting...")
                self._access_token = None
                if self.connect():
                    return self.get_quote(symbol)

            self._record_failure(f"HTTP {response.status_code}")
            return self._quote_cache.get(symbol)

        except Exception as e:
            self._record_failure(str(e))
            logger.error(f"SSI quote error for {symbol}: {e}")
            return self._quote_cache.get(symbol)

    def get_orderbook(self, symbol: str, depth: int = 10) -> Optional[OrderBook]:
        """Get order book from SSI"""
        # SSI provides top 3 bid/ask in quote data
        quote = self.get_quote(symbol)
        if not quote:
            return self._orderbook_cache.get(symbol)

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
    VNDirect Real-time Data Provider with enhanced error handling

    Uses VNDirect's public API endpoints

    Features:
    - Automatic retry with exponential backoff
    - Circuit breaker for API failures
    - Request caching
    """

    def __init__(self, api_key: str = "", api_secret: str = ""):
        super().__init__(api_key, api_secret)
        self.base_url = "https://finfo-api.vndirect.com.vn"
        self._session: Optional[Any] = None

    def _get_session(self):
        """Get or create requests session"""
        if self._session is None:
            import requests

            self._session = requests.Session()
            adapter = requests.adapters.HTTPAdapter(
                pool_connections=10, pool_maxsize=10, max_retries=0
            )
            self._session.mount("https://", adapter)
        return self._session

    def connect(self) -> bool:
        """Connect to VNDirect API"""
        # Check circuit breaker
        if not self._circuit_breaker.can_execute():
            logger.warning("VNDirect circuit breaker is OPEN")
            return False

        try:
            # Test connection with a simple request
            import requests

            response = self._get_session().get(
                f"{self.base_url}/v4/stock_prices", params={"q": "code:VNM", "size": 1}, timeout=5
            )

            if response.status_code == 200:
                self._connected = True
                self._record_success()
                logger.info("✅ Connected to VNDirect API")
                return True

            self._record_failure(f"HTTP {response.status_code}")
            return False

        except Exception as e:
            self._record_failure(str(e))
            logger.error(f"VNDirect connection failed: {e}")
            return False

    def disconnect(self) -> None:
        self._connected = False
        if self._session:
            self._session.close()
            self._session = None

    def get_quote(self, symbol: str) -> Optional[RealtimeQuote]:
        """Get quote from VNDirect with error handling"""
        # Check cache first
        if self._is_cache_valid(symbol):
            return self._quote_cache.get(symbol)

        # Check circuit breaker
        if not self._circuit_breaker.can_execute():
            logger.debug(f"VNDirect circuit open, returning cached data for {symbol}")
            return self._quote_cache.get(symbol)

        last_error = None

        for attempt in range(self._retry_config.max_retries + 1):
            try:
                start_time = time.time()

                url = f"{self.base_url}/v4/stock_prices"
                params = {"sort": "date", "q": f"code:{symbol}", "size": 1}

                response = self._get_session().get(url, params=params, timeout=5)

                latency = (time.time() - start_time) * 1000
                _health_monitor.record_request("VNDirect", response.status_code == 200, latency)

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

                        self._update_cache(symbol, quote)
                        self._record_success()
                        return quote

                elif response.status_code == 429:
                    # Rate limited
                    retry_after = int(response.headers.get("Retry-After", 5))
                    logger.warning(f"VNDirect rate limited, waiting {retry_after}s")
                    time.sleep(retry_after)
                    continue

                last_error = f"HTTP {response.status_code}"

            except Exception as e:
                last_error = str(e)
                logger.debug(f"VNDirect quote attempt {attempt + 1} failed: {e}")

            if attempt < self._retry_config.max_retries:
                delay = self._retry_config.base_delay * (2**attempt)
                time.sleep(delay)

        self._record_failure(last_error or "Unknown error")
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
