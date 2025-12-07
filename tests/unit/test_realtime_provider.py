# -*- coding: utf-8 -*-
"""
Tests for Real-time Data Provider

Tests:
- Circuit breaker pattern
- Retry logic
- Health monitoring
- Provider failover
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime, timedelta
import time

from src.data.realtime_provider import (
    RetryConfig,
    CircuitBreakerConfig,
    CircuitState,
    CircuitBreaker,
    ConnectionHealthMonitor,
    RealtimeQuote,
    OrderBook,
    OrderBookLevel,
    MarketDepth,
    SSIRealtimeProvider,
    VNDirectRealtimeProvider,
    RealtimeDataManager,
    DataSource,
    get_realtime_manager,
)


class TestCircuitBreaker:
    """Tests for Circuit Breaker pattern"""

    def test_initial_state(self):
        """Test initial state is CLOSED"""
        cb = CircuitBreaker("test")
        assert cb.state == CircuitState.CLOSED
        assert cb.can_execute() is True

    def test_record_success(self):
        """Test recording success"""
        cb = CircuitBreaker("test")
        cb.record_success()

        assert cb._failure_count == 0
        assert cb._success_count == 1

    def test_record_failure(self):
        """Test recording failure"""
        cb = CircuitBreaker("test")
        cb.record_failure()

        assert cb._failure_count == 1
        assert cb._last_failure_time is not None

    def test_circuit_opens_after_threshold(self):
        """Test circuit opens after failure threshold"""
        config = CircuitBreakerConfig(failure_threshold=3)
        cb = CircuitBreaker("test", config)

        # Record failures
        for _ in range(3):
            cb.record_failure()

        assert cb.state == CircuitState.OPEN
        assert cb.can_execute() is False

    def test_circuit_half_open_after_timeout(self):
        """Test circuit goes to HALF_OPEN after recovery timeout"""
        config = CircuitBreakerConfig(
            failure_threshold=2, recovery_timeout=0.1  # 100ms for testing
        )
        cb = CircuitBreaker("test", config)

        # Open the circuit
        cb.record_failure()
        cb.record_failure()
        assert cb.state == CircuitState.OPEN

        # Wait for recovery timeout
        time.sleep(0.15)

        assert cb.state == CircuitState.HALF_OPEN
        assert cb.can_execute() is True

    def test_circuit_closes_after_success_in_half_open(self):
        """Test circuit closes after success in HALF_OPEN state"""
        config = CircuitBreakerConfig(
            failure_threshold=2, recovery_timeout=0.1, half_open_requests=2
        )
        cb = CircuitBreaker("test", config)

        # Open the circuit
        cb.record_failure()
        cb.record_failure()

        # Wait for recovery
        time.sleep(0.15)
        assert cb.state == CircuitState.HALF_OPEN

        # Record successes
        cb.record_success()
        cb.record_success()

        assert cb.state == CircuitState.CLOSED

    def test_circuit_reopens_on_failure_in_half_open(self):
        """Test circuit reopens on failure in HALF_OPEN state"""
        config = CircuitBreakerConfig(failure_threshold=2, recovery_timeout=0.1)
        cb = CircuitBreaker("test", config)

        # Open the circuit
        cb.record_failure()
        cb.record_failure()

        # Wait for recovery
        time.sleep(0.15)
        assert cb.state == CircuitState.HALF_OPEN

        # Record failure
        cb.record_failure()

        assert cb.state == CircuitState.OPEN

    def test_get_stats(self):
        """Test getting circuit breaker stats"""
        cb = CircuitBreaker("test")
        cb.record_success()
        cb.record_failure()

        stats = cb.get_stats()

        assert stats["name"] == "test"
        assert stats["state"] == "CLOSED"
        assert stats["failure_count"] == 1
        assert stats["success_count"] == 1


class TestConnectionHealthMonitor:
    """Tests for Connection Health Monitor"""

    def test_record_request(self):
        """Test recording requests"""
        monitor = ConnectionHealthMonitor()

        monitor.record_request("SSI", True, 100)
        monitor.record_request("SSI", True, 150)
        monitor.record_request("SSI", False, 200)

        health = monitor.get_health("SSI")

        assert health["total_requests"] == 3
        assert health["success_rate"] == 2 / 3
        assert health["avg_latency_ms"] == 150

    def test_health_status_healthy(self):
        """Test healthy status"""
        monitor = ConnectionHealthMonitor()

        # 95%+ success rate
        for _ in range(19):
            monitor.record_request("SSI", True, 100)
        monitor.record_request("SSI", False, 100)

        health = monitor.get_health("SSI")
        assert health["status"] == "healthy"

    def test_health_status_degraded(self):
        """Test degraded status"""
        monitor = ConnectionHealthMonitor()

        # 80-95% success rate
        for _ in range(85):
            monitor.record_request("SSI", True, 100)
        for _ in range(15):
            monitor.record_request("SSI", False, 100)

        health = monitor.get_health("SSI")
        assert health["status"] == "degraded"

    def test_health_status_unhealthy(self):
        """Test unhealthy status"""
        monitor = ConnectionHealthMonitor()

        # <80% success rate
        for _ in range(70):
            monitor.record_request("SSI", True, 100)
        for _ in range(30):
            monitor.record_request("SSI", False, 100)

        health = monitor.get_health("SSI")
        assert health["status"] == "unhealthy"

    def test_unknown_provider(self):
        """Test unknown provider"""
        monitor = ConnectionHealthMonitor()

        health = monitor.get_health("UNKNOWN")
        assert health["status"] == "unknown"


class TestRealtimeQuote:
    """Tests for RealtimeQuote dataclass"""

    def test_foreign_net_volume(self):
        """Test foreign net volume calculation"""
        quote = RealtimeQuote(
            symbol="VNM",
            price=85000,
            change=1000,
            change_pct=1.2,
            volume=1000000,
            value=85000000000,
            foreign_buy_volume=100000,
            foreign_sell_volume=80000,
        )

        assert quote.foreign_net_volume == 20000

    def test_spread(self):
        """Test spread calculation"""
        quote = RealtimeQuote(
            symbol="VNM",
            price=85000,
            change=0,
            change_pct=0,
            volume=0,
            value=0,
            bid_price=84900,
            ask_price=85100,
        )

        assert quote.spread == 200
        assert abs(quote.spread_pct - 0.235) < 0.01


class TestOrderBook:
    """Tests for OrderBook dataclass"""

    def test_best_bid_ask(self):
        """Test best bid/ask"""
        orderbook = OrderBook(
            symbol="VNM",
            bids=[
                OrderBookLevel(price=84900, volume=1000),
                OrderBookLevel(price=84800, volume=2000),
            ],
            asks=[
                OrderBookLevel(price=85100, volume=1500),
                OrderBookLevel(price=85200, volume=2500),
            ],
        )

        assert orderbook.best_bid.price == 84900
        assert orderbook.best_ask.price == 85100

    def test_mid_price(self):
        """Test mid price calculation"""
        orderbook = OrderBook(
            symbol="VNM",
            bids=[OrderBookLevel(price=84900, volume=1000)],
            asks=[OrderBookLevel(price=85100, volume=1000)],
        )

        assert orderbook.mid_price == 85000

    def test_imbalance(self):
        """Test order book imbalance"""
        orderbook = OrderBook(
            symbol="VNM",
            bids=[OrderBookLevel(price=84900, volume=3000)],
            asks=[OrderBookLevel(price=85100, volume=1000)],
        )

        # More bids = positive imbalance
        assert orderbook.imbalance == 0.5  # (3000-1000)/(3000+1000)


class TestSSIRealtimeProvider:
    """Tests for SSI Realtime Provider"""

    def test_init(self):
        """Test initialization"""
        provider = SSIRealtimeProvider("consumer_id", "consumer_secret")

        assert provider.api_key == "consumer_id"
        assert provider.api_secret == "consumer_secret"
        assert provider._connected is False

    def test_connect_without_credentials(self):
        """Test connection without credentials"""
        provider = SSIRealtimeProvider()

        result = provider.connect()
        assert result is False

    @patch("requests.Session.post")
    def test_connect_success(self, mock_post):
        """Test successful connection"""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"status": 200, "data": {"accessToken": "test_token"}}
        mock_post.return_value = mock_response

        provider = SSIRealtimeProvider("consumer_id", "consumer_secret")
        result = provider.connect()

        assert result is True
        assert provider._connected is True

    @patch("requests.Session.get")
    def test_get_quote_from_cache(self, mock_get):
        """Test getting quote from cache"""
        provider = SSIRealtimeProvider("id", "secret")
        provider._connected = True
        provider._access_token = "token"

        # Add to cache
        cached_quote = RealtimeQuote(
            symbol="VNM", price=85000, change=0, change_pct=0, volume=0, value=0, source="SSI"
        )
        provider._quote_cache["VNM"] = cached_quote
        provider._cache_timestamps["VNM"] = datetime.now()

        # Should return cached
        quote = provider.get_quote("VNM")

        assert quote is not None
        assert quote.symbol == "VNM"
        mock_get.assert_not_called()

    def test_get_health_status(self):
        """Test health status"""
        provider = SSIRealtimeProvider("id", "secret")
        provider._connected = True
        provider._consecutive_failures = 2

        status = provider.get_health_status()

        assert status["provider"] == "SSIRealtimeProvider"
        assert status["connected"] is True
        assert status["consecutive_failures"] == 2


class TestVNDirectRealtimeProvider:
    """Tests for VNDirect Realtime Provider"""

    def test_init(self):
        """Test initialization"""
        provider = VNDirectRealtimeProvider()

        assert provider._connected is False

    @patch("requests.Session.get")
    def test_connect_success(self, mock_get):
        """Test successful connection"""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"data": [{"code": "VNM"}]}
        mock_get.return_value = mock_response

        provider = VNDirectRealtimeProvider()
        result = provider.connect()

        assert result is True


class TestRealtimeDataManager:
    """Tests for Realtime Data Manager"""

    def test_add_provider(self):
        """Test adding provider"""
        manager = RealtimeDataManager()
        provider = Mock(spec=SSIRealtimeProvider)

        manager.add_provider(DataSource.SSI, provider, is_primary=True)

        assert DataSource.SSI in manager._providers
        assert manager._primary_source == DataSource.SSI

    def test_get_quote_with_failover(self):
        """Test quote with failover"""
        manager = RealtimeDataManager()

        # Primary fails
        primary = Mock()
        primary.get_quote.return_value = None

        # Secondary succeeds
        secondary = Mock()
        secondary.get_quote.return_value = RealtimeQuote(
            symbol="VNM", price=85000, change=0, change_pct=0, volume=0, value=0, source="VNDirect"
        )

        manager.add_provider(DataSource.SSI, primary, is_primary=True)
        manager.add_provider(DataSource.VNDIRECT, secondary)

        quote = manager.get_quote("VNM")

        assert quote is not None
        assert quote.source == "VNDirect"

    def test_get_quotes_batch(self):
        """Test batch quote retrieval"""
        manager = RealtimeDataManager()

        provider = Mock()
        provider.get_quote.side_effect = lambda s: RealtimeQuote(
            symbol=s, price=85000, change=0, change_pct=0, volume=0, value=0
        )

        manager.add_provider(DataSource.SSI, provider, is_primary=True)

        quotes = manager.get_quotes_batch(["VNM", "FPT", "VCB"])

        assert len(quotes) == 3
        assert "VNM" in quotes

    def test_subscribe(self):
        """Test subscription"""
        manager = RealtimeDataManager()
        callback = Mock()

        manager.subscribe("VNM", callback)

        assert "VNM" in manager._quote_callbacks
        assert callback in manager._quote_callbacks["VNM"]


class TestRetryConfig:
    """Tests for RetryConfig"""

    def test_default_values(self):
        """Test default configuration values"""
        config = RetryConfig()

        assert config.max_retries == 3
        assert config.base_delay == 1.0
        assert config.max_delay == 30.0
        assert config.exponential_base == 2.0
        assert config.jitter is True

    def test_custom_values(self):
        """Test custom configuration"""
        config = RetryConfig(max_retries=5, base_delay=0.5, max_delay=60.0)

        assert config.max_retries == 5
        assert config.base_delay == 0.5
        assert config.max_delay == 60.0
